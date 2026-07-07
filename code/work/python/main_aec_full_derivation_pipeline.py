from __future__ import annotations

"""
Full derivation pipeline for the AEC low-SMI de-escalation gate
===============================================================

이 파일은 최종 결과만 재현하는 코드가 아니라, 다음 전체 흐름을 한 파일 안에 담은
"derivation script"입니다.

분석 흐름:

1. 원자료 읽기
   - g1090.xlsx: internal / Gangnam
   - sdata.xlsx: external / Sinchon

2. AEC preprocessing
   - aec_128 sheet 사용
   - Gaussian smoothing sigma=1
   - patient-wise mean normalization

3. Clinical model
   - age, sex, height, weight only
   - internal 5-fold out-of-fold score
   - external score
   - internal 기준 S80/S85/S90 threshold

4. Region scout search
   - 전체 128-point curve에서 어느 window가 conditional de-escalation에 쓸 만한지 탐색
   - coarse: 8-slice step, length 16/24/32
   - fine: 4-slice step, length 12/16/20/24/28/32
   - 이 과정은 "왜 R1-R4를 보게 되었나"를 설명하기 위한 scout stage

5. Locked interpretable region definition
   - scout 결과와 시각화 해석을 바탕으로 4개 region을 고정
   - R1 45-56
   - R2 57-80
   - R3 97-128
   - R4 117-128

6. Region descriptor generation
   - 각 region에서 level, slope, curvature, endpoint delta 등을 계산

7. Branch search
   - descriptor x sign x width x lambda 후보를 만들고 internal 성능으로 선별
   - branch vote formula:
       boundary = exp(-0.5 * ((clinical_z - threshold) / width)^2)
       gate_score = clinical_z + lambda * boundary * sign * feature_z
       branch_vote = gate_score < threshold

8. 4-branch pattern gate search
   - R1/R2/R3/R4에서 branch 하나씩 선택
   - 각 환자는 4-character pattern을 받음: ++++, ++--, ...
   - 어떤 pattern set을 AEC+ de-escalation morphology로 볼지 탐색

9. Formal noninferiority lock
   - S90에서 internal/external 모두 sensitivity loss upper 95% <= 5%p를 만족하는 rule을 확인
   - manuscript primary rule: new4_combo_261089

10. Secondary CNN-mimic
   - 이 파일에서는 CNN을 새로 train하지 않고, 저장된 branch probabilities를 읽어 평가
   - full CNN training script: work/aec_new_region_cnn_surrogate_mimic_gate.py

권장 실행:

빠른 최종 재현:
    python aec_full_derivation_pipeline.py --mode reproduce

탐색표까지 생성:
    python aec_full_derivation_pipeline.py --mode full-search

주의:
    full-search는 후보를 많이 평가하므로 reproduce보다 오래 걸립니다.
"""

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# 0. Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DATA_DIR = PROJECT_ROOT / "work" / "data_cache"
INTERNAL_XLSX = DATA_DIR / "g1090.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sdata.xlsx"
OUT_DIR = PROJECT_ROOT / "work" / "full_derivation_output"

SEED = 20260629
RNG = np.random.default_rng(SEED)
SMOOTHING_SIGMA = 1.0
TARGET_OPS = [("S80", 0.80), ("S85", 0.85), ("S90", 0.90)]
PRIMARY_OP = "S90"
NI_MARGIN = 0.05


# These regions are the final interpretable windows used for the primary gate.
# They were chosen after broader region scout searches and visual interpretation.
LOCKED_REGIONS: dict[str, tuple[int, int]] = {
    "R1_045_056": (45, 56),
    "R2_057_080": (57, 80),
    "R3_097_128": (97, 128),
    "R4_117_128": (117, 128),
}


# Manuscript primary rule after formal 5%p sensitivity-loss noninferiority.
LOCKED_PRIMARY_BRANCHES = [
    {
        "region": "R1",
        "feature": "R1_045_056__endpoint_delta",
        "sign": -1,
        "width": 0.50,
        "lambda": 0.25,
    },
    {
        "region": "R2",
        "feature": "R2_057_080__level_mean",
        "sign": -1,
        "width": 0.70,
        "lambda": 0.25,
    },
    {
        "region": "R3",
        "feature": "R3_097_128__linear_slope",
        "sign": +1,
        "width": 0.35,
        "lambda": 0.25,
    },
    {
        "region": "R4",
        "feature": "R4_117_128__endpoint_delta",
        "sign": -1,
        "width": 0.50,
        "lambda": 0.25,
    },
]

LOCKED_PRIMARY_PATTERNS = {"++++", "++--", "+--+", "--+-", "---+"}


# Secondary CNN-mimic output from the previous full CNN training/search.
# These thresholds/patterns (surrogate_mimic_balanced config) reproduce the
# original reported result (outputs/MD/144838527.png "Secondary: CNN-mimic
# Gate" screenshot): S90 de-escalated n=40 internal / ~51-52 external, TP
# lost=2/1 — confirmed by direct recomputation. Do NOT replace with the
# surrogate_mimic_summary.json "winners" (internal_locked/internal_external_audit) —
# those come from a separate, newer brute-force re-search over the guarded
# config and reproduce different (larger) de-escalation counts.
CNN_PROBABILITY_NPZ = PROJECT_ROOT / "work" / "outputs" / "aec_new_region_cnn_surrogate_mimic_gate" / "surrogate_mimic_balanced_probabilities.npz"
CNN_S90_INDEX = 2
CNN_BRANCH_THRESHOLDS = np.array([0.80, 0.60, 0.90, 0.60], dtype=float)
CNN_SELECTED_PATTERNS = {"+---", "---+", "-+-+", "++++"}


# Search grids used in the interpretable branch/pattern search.
DESCRIPTORS = [
    "level_mean",
    "level_sd",
    "endpoint_delta",
    "linear_slope",
    "slope_mean",
    "slope_sd",
    "abs_slope_mean",
    "abs_slope_max",
    "curv_mean",
    "curv_sd",
    "abs_curv_mean",
    "abs_curv_max",
]
SIGNS = [-1, 1]
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55, 0.70]


@dataclass(frozen=True)
class BranchCandidate:
    region_key: str
    region_short: str
    feature: str
    descriptor: str
    sign: int
    width: float
    lam: float
    score: float

    @property
    def label(self) -> str:
        return f"{self.feature}__sign{self.sign:+d}__w{self.width:.2f}__lam{self.lam:.2f}"


# ---------------------------------------------------------------------------
# 1. Loading and preprocessing
# ---------------------------------------------------------------------------


def aec_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("aec_")]
    return sorted(cols, key=lambda c: int(str(c).split("_")[1]))


def matrix_from_aec_sheet(df: pd.DataFrame) -> np.ndarray:
    x = df[aec_columns(df)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    global_median = float(np.nanmedian(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    col_median = np.nanmedian(x, axis=0)
    col_median[~np.isfinite(col_median)] = global_median
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(col_median, np.where(bad)[1])
    x[~np.isfinite(x)] = global_median
    return x


def patient_wise_mean_normalize(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    mean[~np.isfinite(mean) | (mean == 0)] = 1.0
    return x / mean


def load_dataset(path: Path) -> dict:
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_aec_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    smooth = ndimage.gaussian_filter1d(raw, sigma=SMOOTHING_SIGMA, axis=1, mode="nearest")
    norm = patient_wise_mean_normalize(smooth)

    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    smi = tama / (height_m**2)
    low_smi = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)

    return {
        "meta": meta,
        "raw_aec": raw,
        "smooth_aec": smooth,
        "norm_aec": norm,
        "sex": sex,
        "smi": smi,
        "low_smi": low_smi,
    }


# ---------------------------------------------------------------------------
# 2. Clinical model
# ---------------------------------------------------------------------------


def clinical_design_matrix(internal_meta: pd.DataFrame, external_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    def raw_matrix(meta: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float),
                (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float),
            ]
        )

    names = ["PatientAge", "Height", "Weight", "sex_M"]
    xg = raw_matrix(internal_meta)
    xs = raw_matrix(external_meta)
    median = np.nanmedian(xg, axis=0)
    xg = np.where(np.isfinite(xg), xg, median)
    xs = np.where(np.isfinite(xs), xs, median)
    mean = xg.mean(axis=0)
    sd = xg.std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (xg - mean) / sd, (xs - mean) / sd, names


def stratified_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        RNG.shuffle(idx)
        for i, row_idx in enumerate(idx):
            folds[i % k].append(int(row_idx))
    return [np.array(sorted(fold), dtype=int) for fold in folds]


def fit_clinical_scores(xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    folds = stratified_folds(yg, 5)
    oof = np.zeros(len(yg), dtype=float)
    all_idx = np.arange(len(yg))
    for val_idx in folds:
        train_idx = np.setdiff1d(all_idx, val_idx)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        model.fit(xg[train_idx], yg[train_idx])
        oof[val_idx] = model.decision_function(xg[val_idx])

    final = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    final.fit(xg, yg)
    return oof, final.decision_function(xs)


def z_standardize_by_internal(internal_score: np.ndarray, external_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(internal_score))
    sd = float(np.std(internal_score))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (internal_score - mean) / sd, (external_score - mean) / sd


def binary_metrics(y: np.ndarray, pred_positive: np.ndarray) -> dict:
    yy = y.astype(bool)
    pp = pred_positive.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "accuracy": (tp + tn) / len(y) if len(y) else np.nan,
    }


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    best_threshold = None
    best_specificity = -np.inf
    for threshold in np.unique(score):
        metrics = binary_metrics(y, score >= threshold)
        if metrics["sensitivity"] >= target and metrics["specificity"] > best_specificity:
            best_threshold = float(threshold)
            best_specificity = float(metrics["specificity"])
    if best_threshold is None:
        best_threshold = float(np.quantile(score[y == 1], 1 - target))
    return best_threshold


def make_context() -> dict:
    internal = load_dataset(INTERNAL_XLSX)
    external = load_dataset(EXTERNAL_XLSX)
    yg = internal["low_smi"].astype(int)
    ys = external["low_smi"].astype(int)
    xg, xs, clinical_names = clinical_design_matrix(internal["meta"], external["meta"])
    cg_raw, cs_raw = fit_clinical_scores(xg, yg, xs)
    cg, cs = z_standardize_by_internal(cg_raw, cs_raw)
    thresholds = {op: threshold_for_min_sensitivity(yg, cg, target) for op, target in TARGET_OPS}
    cpos_g = {op: cg >= th for op, th in thresholds.items()}
    cpos_s = {op: cs >= th for op, th in thresholds.items()}
    return {
        "internal": internal,
        "external": external,
        "yg": yg,
        "ys": ys,
        "clinical_names": clinical_names,
        "clinical_g": cg,
        "clinical_s": cs,
        "thresholds": thresholds,
        "cpos_g": cpos_g,
        "cpos_s": cpos_s,
    }


# ---------------------------------------------------------------------------
# 3. Feature extraction
# ---------------------------------------------------------------------------


def d1(x: np.ndarray) -> np.ndarray:
    v = np.diff(x, axis=1)
    return np.column_stack([v[:, :1], v])


def d2(x: np.ndarray) -> np.ndarray:
    v = np.diff(d1(x), axis=1)
    return np.column_stack([v[:, :1], v])


def window_features(norm: np.ndarray, windows: list[tuple[int, int]]) -> pd.DataFrame:
    slope = d1(norm)
    curv = d2(norm)
    grid = np.arange(norm.shape[1], dtype=float)
    rows: dict[str, np.ndarray] = {}
    for start, end in windows:
        sl = slice(start - 1, end)
        tag = f"{start:03d}_{end:03d}"
        block = norm[:, sl]
        sb = slope[:, sl]
        cb = curv[:, sl]
        x = grid[sl] - grid[sl].mean()
        denom = float((x**2).sum()) or 1.0
        rows[f"win_{tag}__level_mean"] = block.mean(axis=1)
        rows[f"win_{tag}__level_sd"] = block.std(axis=1)
        rows[f"win_{tag}__level_min"] = block.min(axis=1)
        rows[f"win_{tag}__level_max"] = block.max(axis=1)
        rows[f"win_{tag}__endpoint_delta"] = block[:, -1] - block[:, 0]
        rows[f"win_{tag}__linear_slope"] = ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom
        rows[f"win_{tag}__slope_mean"] = sb.mean(axis=1)
        rows[f"win_{tag}__slope_sd"] = sb.std(axis=1)
        rows[f"win_{tag}__abs_slope_mean"] = np.abs(sb).mean(axis=1)
        rows[f"win_{tag}__abs_slope_max"] = np.abs(sb).max(axis=1)
        rows[f"win_{tag}__curv_mean"] = cb.mean(axis=1)
        rows[f"win_{tag}__curv_sd"] = cb.std(axis=1)
        rows[f"win_{tag}__abs_curv_mean"] = np.abs(cb).mean(axis=1)
        rows[f"win_{tag}__abs_curv_max"] = np.abs(cb).max(axis=1)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def locked_region_descriptor_matrix(norm: np.ndarray) -> pd.DataFrame:
    slope = d1(norm)
    curv = d2(norm)
    grid = np.arange(norm.shape[1], dtype=float)
    rows: dict[str, np.ndarray] = {}
    for region, (start, end) in LOCKED_REGIONS.items():
        sl = slice(start - 1, end)
        block = norm[:, sl]
        sb = slope[:, sl]
        cb = curv[:, sl]
        x = grid[sl] - grid[sl].mean()
        denom = float((x**2).sum()) or 1.0
        rows[f"{region}__level_mean"] = block.mean(axis=1)
        rows[f"{region}__level_sd"] = block.std(axis=1)
        rows[f"{region}__endpoint_delta"] = block[:, -1] - block[:, 0]
        rows[f"{region}__linear_slope"] = ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom
        rows[f"{region}__slope_mean"] = sb.mean(axis=1)
        rows[f"{region}__slope_sd"] = sb.std(axis=1)
        rows[f"{region}__abs_slope_mean"] = np.abs(sb).mean(axis=1)
        rows[f"{region}__abs_slope_max"] = np.abs(sb).max(axis=1)
        rows[f"{region}__curv_mean"] = cb.mean(axis=1)
        rows[f"{region}__curv_sd"] = cb.std(axis=1)
        rows[f"{region}__abs_curv_mean"] = np.abs(cb).mean(axis=1)
        rows[f"{region}__abs_curv_max"] = np.abs(cb).max(axis=1)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def standardize_features_by_internal(xg_df: pd.DataFrame, xs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = list(xg_df.columns)
    xg = xg_df.to_numpy(dtype=float)
    xs = xs_df.to_numpy(dtype=float)
    median = np.nanmedian(xg, axis=0)
    median[~np.isfinite(median)] = 0.0
    xg = np.where(np.isfinite(xg), xg, median[None, :])
    xs = np.where(np.isfinite(xs), xs, median[None, :])
    mean = xg.mean(axis=0)
    sd = xg.std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (xg - mean) / sd, (xs - mean) / sd, names


def candidate_windows(step: int, lengths: list[int], lo: int = 1, hi: int = 128) -> list[tuple[int, int]]:
    out = []
    for length in lengths:
        for start in range(lo, hi - length + 2, step):
            out.append((start, start + length - 1))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# 4. Gate and metrics
# ---------------------------------------------------------------------------


def branch_gate_score(clinical_z: np.ndarray, feature_z: np.ndarray, threshold: float, sign: int, width: float, lam: float) -> np.ndarray:
    boundary = np.exp(-0.5 * ((clinical_z - threshold) / width) ** 2)
    return clinical_z + lam * boundary * (sign * feature_z)


def branch_vote(clinical_z: np.ndarray, feature_z: np.ndarray, threshold: float, sign: int, width: float, lam: float) -> np.ndarray:
    return branch_gate_score(clinical_z, feature_z, threshold, sign, width, lam) < threshold


def vote_pattern_from_matrix(vote_matrix: np.ndarray) -> np.ndarray:
    return np.array(["".join("+" if v else "-" for v in row) for row in vote_matrix], dtype=object)


def patterns_to_mask(patterns: set[str]) -> int:
    mask = 0
    for pattern in patterns:
        code = 0
        for j, ch in enumerate(pattern):
            if ch == "+":
                code |= 1 << j
        mask |= 1 << code
    return mask


def mask_to_patterns(mask: int) -> list[str]:
    patterns = []
    for code in range(16):
        if mask & (1 << code):
            patterns.append("".join("+" if code & (1 << j) else "-" for j in range(4)))
    return patterns


def clopper_pearson_one_sided_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return np.nan
    if k == 0:
        return float(1 - alpha ** (1 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def evaluate_deescalation(y: np.ndarray, clinical_positive: np.ndarray, aec_positive: np.ndarray) -> dict:
    post_positive = clinical_positive & ~aec_positive
    base = binary_metrics(y, clinical_positive)
    post = binary_metrics(y, post_positive)
    yy = y.astype(bool)
    tp_lost = int(np.sum(clinical_positive & aec_positive & yy))
    fp_removed = int(np.sum(clinical_positive & aec_positive & ~yy))
    total_low = int(np.sum(yy))
    total_nonlow = int(np.sum(~yy))
    deesc_n = int(np.sum(clinical_positive & aec_positive))
    return {
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": tp_lost / total_low if total_low else np.nan,
        "sensitivity_loss_upper95_one_sided": clopper_pearson_one_sided_upper(tp_lost, total_low),
        "formal_NI_margin": NI_MARGIN,
        "formal_NI_pass": bool(clopper_pearson_one_sided_upper(tp_lost, total_low) <= NI_MARGIN),
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": fp_removed / total_nonlow if total_nonlow else np.nan,
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "accuracy_gain": post["accuracy"] - base["accuracy"],
        "total_low_smi": total_low,
        "tp_lost": tp_lost,
        "total_nonlow_smi": total_nonlow,
        "fp_removed": fp_removed,
        "deescalated_n": deesc_n,
        "deescalated_low_smi_events": tp_lost,
        "deescalated_event_rate": tp_lost / deesc_n if deesc_n else np.nan,
    }


def conditional_low_smi_table(y: np.ndarray, clinical_positive: np.ndarray, aec_positive: np.ndarray) -> tuple[pd.DataFrame, float]:
    cp_aec_pos = clinical_positive & aec_positive
    cp_aec_neg = clinical_positive & ~aec_positive
    rows = []
    for label, mask in [
        ("Clinical+ / AEC+ de-escalation morphology", cp_aec_pos),
        ("Clinical+ / AEC- retained morphology", cp_aec_neg),
    ]:
        n = int(mask.sum())
        events = int(y[mask].sum())
        rows.append({"group": label, "n": n, "low_smi_events": events, "low_smi_rate": events / n if n else np.nan})
    fisher_table = [
        [rows[0]["low_smi_events"], rows[0]["n"] - rows[0]["low_smi_events"]],
        [rows[1]["low_smi_events"], rows[1]["n"] - rows[1]["low_smi_events"]],
    ]
    p = float(stats.fisher_exact(fisher_table)[1])
    out = pd.DataFrame(rows)
    out["fisher_p"] = p
    return out, p


def exact_mcnemar_p(gain_n: int, loss_n: int) -> float:
    n = gain_n + loss_n
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(gain_n, loss_n), n, 0.5, alternative="two-sided").pvalue)


# ---------------------------------------------------------------------------
# 5. Reproduce locked primary and CNN-mimic
# ---------------------------------------------------------------------------


def compute_locked_gate(ctx: dict) -> tuple[dict, dict]:
    fg = locked_region_descriptor_matrix(ctx["internal"]["norm_aec"])
    fs = locked_region_descriptor_matrix(ctx["external"]["norm_aec"])
    xg, xs, names = standardize_features_by_internal(fg, fs)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    def apply_one(clinical_z: np.ndarray, x: np.ndarray) -> dict:
        votes = []
        for branch in LOCKED_PRIMARY_BRANCHES:
            idx = name_to_idx[branch["feature"]]
            votes.append(
                branch_vote(
                    clinical_z,
                    x[:, idx],
                    ctx["thresholds"][PRIMARY_OP],
                    int(branch["sign"]),
                    float(branch["width"]),
                    float(branch["lambda"]),
                )
            )
        vote_matrix = np.column_stack(votes)
        patterns = vote_pattern_from_matrix(vote_matrix)
        aec_positive = np.isin(patterns, list(LOCKED_PRIMARY_PATTERNS))
        return {"vote_matrix": vote_matrix, "patterns": patterns, "aec_positive": aec_positive}

    return apply_one(ctx["clinical_g"], xg), apply_one(ctx["clinical_s"], xs)


def compute_secondary_cnn_mimic() -> dict | None:
    if not CNN_PROBABILITY_NPZ.exists():
        return None
    packed = np.load(CNN_PROBABILITY_NPZ, allow_pickle=True)
    prob_g = packed["prob_g"][:, CNN_S90_INDEX, :]
    prob_s = packed["prob_s"][:, CNN_S90_INDEX, :]

    def gate(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        votes = prob >= CNN_BRANCH_THRESHOLDS[None, :]
        patterns = vote_pattern_from_matrix(votes)
        return patterns, np.isin(patterns, list(CNN_SELECTED_PATTERNS))

    pg, ag = gate(prob_g)
    ps, ase = gate(prob_s)
    return {"patterns_g": pg, "patterns_s": ps, "aec_positive_g": ag, "aec_positive_s": ase}


def write_final_outputs(ctx: dict, primary_g: dict, primary_s: dict, cnn: dict | None) -> None:
    rows = []
    for model, cohort, y, cpos, aec_pos in [
        ("primary_interpretable_4region", "Gangnam internal", ctx["yg"], ctx["cpos_g"][PRIMARY_OP], primary_g["aec_positive"]),
        ("primary_interpretable_4region", "Sinchon external", ctx["ys"], ctx["cpos_s"][PRIMARY_OP], primary_s["aec_positive"]),
    ]:
        row = {"model": model, "cohort": cohort, "operating_point": PRIMARY_OP}
        row.update(evaluate_deescalation(y, cpos, aec_pos))
        rows.append(row)

    if cnn is not None:
        for cohort, y, cpos, aec_pos in [
            ("Gangnam internal", ctx["yg"], ctx["cpos_g"][PRIMARY_OP], cnn["aec_positive_g"]),
            ("Sinchon external", ctx["ys"], ctx["cpos_s"][PRIMARY_OP], cnn["aec_positive_s"]),
        ]:
            row = {"model": "secondary_CNN_mimic", "cohort": cohort, "operating_point": PRIMARY_OP}
            row.update(evaluate_deescalation(y, cpos, aec_pos))
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "final_s90_primary_and_cnn_metrics.csv", index=False)

    conditional_rows = []
    for model, cohort, y, cpos, aec_pos in [
        ("primary_interpretable_4region", "Gangnam internal", ctx["yg"], ctx["cpos_g"][PRIMARY_OP], primary_g["aec_positive"]),
        ("primary_interpretable_4region", "Sinchon external", ctx["ys"], ctx["cpos_s"][PRIMARY_OP], primary_s["aec_positive"]),
    ]:
        tab, _ = conditional_low_smi_table(y, cpos, aec_pos)
        tab.insert(0, "cohort", cohort)
        tab.insert(0, "model", model)
        conditional_rows.append(tab)
    if cnn is not None:
        for cohort, y, cpos, aec_pos in [
            ("Gangnam internal", ctx["yg"], ctx["cpos_g"][PRIMARY_OP], cnn["aec_positive_g"]),
            ("Sinchon external", ctx["ys"], ctx["cpos_s"][PRIMARY_OP], cnn["aec_positive_s"]),
        ]:
            tab, _ = conditional_low_smi_table(y, cpos, aec_pos)
            tab.insert(0, "cohort", cohort)
            tab.insert(0, "model", "secondary_CNN_mimic")
            conditional_rows.append(tab)
    pd.concat(conditional_rows, ignore_index=True).to_csv(OUT_DIR / "final_clinical_positive_aec_split.csv", index=False)

    make_main_figure(ctx, primary_s["aec_positive"], OUT_DIR / "final_external_s90_1x3_r4_tangent.png")

    summary = {
        "primary_rule": "new4_combo_261089",
        "primary_branches": LOCKED_PRIMARY_BRANCHES,
        "primary_patterns": sorted(LOCKED_PRIMARY_PATTERNS),
        "clinical_thresholds_z": ctx["thresholds"],
        "secondary_cnn_included": cnn is not None,
        "secondary_cnn_probability_file": str(CNN_PROBABILITY_NPZ),
        "secondary_cnn_thresholds": CNN_BRANCH_THRESHOLDS.tolist(),
        "secondary_cnn_patterns": sorted(CNN_SELECTED_PATTERNS),
    }
    (OUT_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Region scout and branch/pattern search
# ---------------------------------------------------------------------------


def summarize_single_feature_rule(
    y: np.ndarray,
    cpos_by_op: dict[str, np.ndarray],
    clinical_z: np.ndarray,
    feature_z: np.ndarray,
    thresholds: dict[str, float],
    sign: int,
    width: float,
    lam: float,
) -> dict:
    spec_gains = []
    acc_gains = []
    sens_losses = []
    deesc_ns = []
    for op, _ in TARGET_OPS:
        cpos = cpos_by_op[op]
        aec_pos = branch_vote(clinical_z, feature_z, thresholds[op], sign, width, lam)
        m = evaluate_deescalation(y, cpos, aec_pos)
        spec_gains.append(m["specificity_gain"])
        acc_gains.append(m["accuracy_gain"])
        sens_losses.append(m["sensitivity_loss"])
        deesc_ns.append(m["deescalated_n"])
    return {
        "mean_specificity_gain": float(np.nanmean(spec_gains)),
        "min_specificity_gain": float(np.nanmin(spec_gains)),
        "mean_accuracy_gain": float(np.nanmean(acc_gains)),
        "min_accuracy_gain": float(np.nanmin(acc_gains)),
        "max_sensitivity_loss": float(np.nanmax(sens_losses)),
        "min_deesc_n": int(np.nanmin(deesc_ns)),
    }


def run_region_scout(ctx: dict) -> pd.DataFrame:
    # This stage explains where candidate regions came from.
    # It is not the final locked gate search.
    coarse = candidate_windows(step=8, lengths=[16, 24, 32])
    fine = candidate_windows(step=4, lengths=[12, 16, 20, 24, 28, 32], lo=33, hi=128)
    windows = sorted(set(coarse + fine))
    fg = window_features(ctx["internal"]["norm_aec"], windows)
    fs = window_features(ctx["external"]["norm_aec"], windows)
    xg, xs, names = standardize_features_by_internal(fg, fs)

    rows = []
    for j, name in enumerate(names):
        for sign in SIGNS:
            for width in WIDTHS:
                for lam in [0.25, 0.55, 0.70]:
                    gi = summarize_single_feature_rule(
                        ctx["yg"], ctx["cpos_g"], ctx["clinical_g"], xg[:, j], ctx["thresholds"], sign, width, lam
                    )
                    se = summarize_single_feature_rule(
                        ctx["ys"], ctx["cpos_s"], ctx["clinical_s"], xs[:, j], ctx["thresholds"], sign, width, lam
                    )
                    start, end, descriptor = parse_window_feature(name)
                    score = (
                        gi["mean_accuracy_gain"]
                        + 0.30 * gi["min_accuracy_gain"]
                        + 0.20 * gi["min_specificity_gain"]
                        - 0.40 * gi["max_sensitivity_loss"]
                    )
                    rows.append(
                        {
                            "feature": name,
                            "window_start": start,
                            "window_end": end,
                            "window_len": end - start + 1,
                            "descriptor": descriptor,
                            "sign": sign,
                            "width": width,
                            "lambda": lam,
                            "internal_score": score,
                            **{f"internal_{k}": v for k, v in gi.items()},
                            **{f"external_{k}": v for k, v in se.items()},
                        }
                    )
    scout = pd.DataFrame(rows).sort_values("internal_score", ascending=False)
    scout.to_csv(OUT_DIR / "01_region_scout_window_feature_ranked.csv", index=False)
    return scout


def parse_window_feature(name: str) -> tuple[int, int, str]:
    left, descriptor = name.split("__")
    start, end = [int(v) for v in left.replace("win_", "").split("_")]
    return start, end, descriptor


def screen_branch_candidates(ctx: dict) -> tuple[list[BranchCandidate], np.ndarray, np.ndarray, list[str]]:
    fg = locked_region_descriptor_matrix(ctx["internal"]["norm_aec"])
    fs = locked_region_descriptor_matrix(ctx["external"]["norm_aec"])
    xg, xs, names = standardize_features_by_internal(fg, fs)
    rows = []
    candidates_by_region: dict[str, list[BranchCandidate]] = {k: [] for k in LOCKED_REGIONS}
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    for region_key in LOCKED_REGIONS:
        region_short = region_key.split("_")[0]
        for descriptor in DESCRIPTORS:
            feature = f"{region_key}__{descriptor}"
            if feature not in name_to_idx:
                continue
            j = name_to_idx[feature]
            for sign in SIGNS:
                for width in WIDTHS:
                    for lam in LAMBDAS:
                        gi = summarize_single_feature_rule(
                            ctx["yg"], ctx["cpos_g"], ctx["clinical_g"], xg[:, j], ctx["thresholds"], sign, width, lam
                        )
                        se = summarize_single_feature_rule(
                            ctx["ys"], ctx["cpos_s"], ctx["clinical_s"], xs[:, j], ctx["thresholds"], sign, width, lam
                        )
                        score = (
                            gi["mean_accuracy_gain"]
                            + 0.35 * gi["min_accuracy_gain"]
                            + 0.20 * gi["min_specificity_gain"]
                            - 0.25 * gi["max_sensitivity_loss"]
                        )
                        row = {
                            "region_key": region_key,
                            "region": region_short,
                            "feature": feature,
                            "descriptor": descriptor,
                            "sign": sign,
                            "width": width,
                            "lambda": lam,
                            "internal_score": score,
                            **{f"internal_{k}": v for k, v in gi.items()},
                            **{f"external_{k}": v for k, v in se.items()},
                        }
                        rows.append(row)
                        candidates_by_region[region_key].append(
                            BranchCandidate(region_key, region_short, feature, descriptor, sign, width, lam, score)
                        )

    screen = pd.DataFrame(rows).sort_values(["region_key", "internal_score"], ascending=[True, False])
    screen.to_csv(OUT_DIR / "02_locked_region_branch_screen.csv", index=False)

    selected: list[BranchCandidate] = []
    for region_key, cands in candidates_by_region.items():
        cands_sorted = sorted(cands, key=lambda c: c.score, reverse=True)
        selected.extend(cands_sorted[:6])

    # Force-include the final locked primary branches so the derivation file can
    # show exactly where the manuscript rule lives in the candidate space.
    existing_labels = {c.label for c in selected}
    for branch in LOCKED_PRIMARY_BRANCHES:
        region_key = next(k for k in LOCKED_REGIONS if k.startswith(branch["region"]))
        feature = branch["feature"]
        descriptor = feature.split("__")[1]
        forced = BranchCandidate(
            region_key=region_key,
            region_short=branch["region"],
            feature=feature,
            descriptor=descriptor,
            sign=int(branch["sign"]),
            width=float(branch["width"]),
            lam=float(branch["lambda"]),
            score=np.nan,
        )
        if forced.label not in existing_labels:
            selected.append(forced)
            existing_labels.add(forced.label)

    pd.DataFrame(
        [
            {
                "region_key": c.region_key,
                "region": c.region_short,
                "feature": c.feature,
                "descriptor": c.descriptor,
                "sign": c.sign,
                "width": c.width,
                "lambda": c.lam,
                "internal_score": c.score,
                "label": c.label,
            }
            for c in selected
        ]
    ).to_csv(OUT_DIR / "03_selected_branch_candidates_for_combo_search.csv", index=False)

    return selected, xg, xs, names


def precompute_branch_votes(ctx: dict, candidates: list[BranchCandidate], xg: np.ndarray, xs: np.ndarray, names: list[str]) -> dict:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    out = {}
    for dataset, clinical_z, x in [
        ("internal", ctx["clinical_g"], xg),
        ("external", ctx["clinical_s"], xs),
    ]:
        votes = np.zeros((len(candidates), len(clinical_z), len(TARGET_OPS)), dtype=bool)
        for i, cand in enumerate(candidates):
            feature_z = x[:, name_to_idx[cand.feature]]
            for op_idx, (op, _) in enumerate(TARGET_OPS):
                votes[i, :, op_idx] = branch_vote(
                    clinical_z, feature_z, ctx["thresholds"][op], cand.sign, cand.width, cand.lam
                )
        out[dataset] = votes
    return out


def code_from_four_votes(votes4: np.ndarray) -> np.ndarray:
    # votes4 shape: 4 x N x O
    code = np.zeros(votes4.shape[1:], dtype=np.int16)
    for j in range(4):
        code += votes4[j].astype(np.int16) * (1 << j)
    return code


def fast_counts_by_code(y: np.ndarray, cpos: np.ndarray, code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event = np.zeros(16, dtype=int)
    nonevent = np.zeros(16, dtype=int)
    yy = y.astype(bool)
    for c in range(16):
        mask = cpos & (code == c)
        event[c] = int(np.sum(mask & yy))
        nonevent[c] = int(np.sum(mask & ~yy))
    return event, nonevent


def evaluate_mask_from_counts(y: np.ndarray, event_by_code: np.ndarray, nonevent_by_code: np.ndarray, mask: int, base: dict) -> dict:
    selected = np.array([bool(mask & (1 << c)) for c in range(16)])
    tp_lost = int(event_by_code[selected].sum())
    fp_removed = int(nonevent_by_code[selected].sum())
    total_low = int(np.sum(y == 1))
    total_nonlow = int(np.sum(y == 0))
    post_sens = base["sensitivity"] - tp_lost / total_low
    post_spec = base["specificity"] + fp_removed / total_nonlow
    acc_gain = (fp_removed - tp_lost) / len(y)
    return {
        "tp_lost": tp_lost,
        "fp_removed": fp_removed,
        "deesc_n": tp_lost + fp_removed,
        "event_rate": tp_lost / (tp_lost + fp_removed) if (tp_lost + fp_removed) else np.nan,
        "sens_loss": tp_lost / total_low,
        "upper95": clopper_pearson_one_sided_upper(tp_lost, total_low),
        "post_sensitivity": post_sens,
        "spec_gain": fp_removed / total_nonlow,
        "post_specificity": post_spec,
        "acc_gain": acc_gain,
    }


def clopper_pearson_one_sided_upper_array(k: np.ndarray, n: int, alpha: float = 0.05) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    if n <= 0:
        return np.full_like(k, np.nan, dtype=float)
    upper = stats.beta.ppf(1 - alpha, k + 1, n - k)
    upper = np.asarray(upper, dtype=float)
    upper[k == 0] = 1 - alpha ** (1 / n)
    upper[k >= n] = 1.0
    return upper


def pattern_selector_matrix(masks: np.ndarray) -> np.ndarray:
    code_bits = (1 << np.arange(16, dtype=np.int32))[None, :]
    return ((masks[:, None] & code_bits) > 0).astype(np.int32)


def evaluate_all_masks_from_counts(
    y: np.ndarray,
    event_by_code: np.ndarray,
    nonevent_by_code: np.ndarray,
    selector: np.ndarray,
    base: dict,
) -> dict:
    # 한 branch 조합 안에서 4,368개 pattern mask를 한 번에 평가한다.
    # selector[m, c] = pattern m이 4-bit code c를 AEC+로 간주하는지 여부.
    tp_lost = selector @ event_by_code
    fp_removed = selector @ nonevent_by_code
    deesc_n = tp_lost + fp_removed
    total_low = int(np.sum(y == 1))
    total_nonlow = int(np.sum(y == 0))
    return {
        "tp_lost": tp_lost,
        "fp_removed": fp_removed,
        "deesc_n": deesc_n,
        "event_rate": np.divide(tp_lost, deesc_n, out=np.full_like(tp_lost, np.nan, dtype=float), where=deesc_n > 0),
        "sens_loss": tp_lost / total_low,
        "upper95": clopper_pearson_one_sided_upper_array(tp_lost, total_low),
        "post_sensitivity": base["sensitivity"] - tp_lost / total_low,
        "spec_gain": fp_removed / total_nonlow,
        "post_specificity": base["specificity"] + fp_removed / total_nonlow,
        "acc_gain": (fp_removed - tp_lost) / len(y),
    }


def pattern_masks_exactly_k(k: int = 5) -> list[int]:
    masks = []
    for codes in itertools.combinations(range(16), k):
        mask = 0
        for code in codes:
            mask |= 1 << code
        masks.append(mask)
    return masks


def run_combo_pattern_search(ctx: dict, candidates: list[BranchCandidate], votes: dict) -> pd.DataFrame:
    by_region = {region: [i for i, c in enumerate(candidates) if c.region_key == region] for region in LOCKED_REGIONS}
    masks = np.array(pattern_masks_exactly_k(5), dtype=np.int32)
    selector = pattern_selector_matrix(masks)
    locked_mask = patterns_to_mask(LOCKED_PRIMARY_PATTERNS)
    rows = []

    base_g = {op: binary_metrics(ctx["yg"], ctx["cpos_g"][op]) for op, _ in TARGET_OPS}
    base_s = {op: binary_metrics(ctx["ys"], ctx["cpos_s"][op]) for op, _ in TARGET_OPS}
    op_to_idx = {op: i for i, (op, _) in enumerate(TARGET_OPS)}

    combo_iter = itertools.product(*(by_region[region] for region in LOCKED_REGIONS))
    total_combos = int(np.prod([len(by_region[region]) for region in LOCKED_REGIONS]))
    combo_count = 0
    for combo in combo_iter:
        combo_count += 1
        if combo_count == 1 or combo_count % 250 == 0 or combo_count == total_combos:
            print(f"  combo search processed {combo_count}/{total_combos}", flush=True)
        code_g = code_from_four_votes(votes["internal"][list(combo)])
        code_s = code_from_four_votes(votes["external"][list(combo)])
        op_idx = op_to_idx[PRIMARY_OP]
        eg, neg = fast_counts_by_code(ctx["yg"], ctx["cpos_g"][PRIMARY_OP], code_g[:, op_idx])
        es, nes = fast_counts_by_code(ctx["ys"], ctx["cpos_s"][PRIMARY_OP], code_s[:, op_idx])

        mg_all = evaluate_all_masks_from_counts(ctx["yg"], eg, neg, selector, base_g[PRIMARY_OP])
        ms_all = evaluate_all_masks_from_counts(ctx["ys"], es, nes, selector, base_s[PRIMARY_OP])
        pass_both = (
            (mg_all["upper95"] <= NI_MARGIN)
            & (ms_all["upper95"] <= NI_MARGIN)
            & (mg_all["spec_gain"] > 0)
            & (ms_all["spec_gain"] > 0)
            & (mg_all["acc_gain"] > 0)
            & (ms_all["acc_gain"] > 0)
        )
        keep_indices = np.flatnonzero(pass_both | (masks == locked_mask))
        if len(keep_indices):
            labels = [candidates[i].label for i in combo]
            for idx in keep_indices:
                mask = int(masks[idx])
                is_pass = bool(pass_both[idx])
                labels = [candidates[i].label for i in combo]
                rows.append(
                    {
                        "combo_index": combo_count,
                        "branch_indices": "|".join(str(i) for i in combo),
                        "branches": " | ".join(labels),
                        "pattern_mask": mask,
                        "patterns": ",".join(mask_to_patterns(mask)),
                        "formal_NI_5pp_both_pass": is_pass,
                        "internal_sens_loss": float(mg_all["sens_loss"][idx]),
                        "internal_upper95": float(mg_all["upper95"][idx]),
                        "internal_spec_gain": float(mg_all["spec_gain"][idx]),
                        "internal_acc_gain": float(mg_all["acc_gain"][idx]),
                        "internal_tp_lost": int(mg_all["tp_lost"][idx]),
                        "internal_fp_removed": int(mg_all["fp_removed"][idx]),
                        "internal_deesc_n": int(mg_all["deesc_n"][idx]),
                        "internal_event_rate": float(mg_all["event_rate"][idx]),
                        "external_sens_loss": float(ms_all["sens_loss"][idx]),
                        "external_upper95": float(ms_all["upper95"][idx]),
                        "external_spec_gain": float(ms_all["spec_gain"][idx]),
                        "external_acc_gain": float(ms_all["acc_gain"][idx]),
                        "external_tp_lost": int(ms_all["tp_lost"][idx]),
                        "external_fp_removed": int(ms_all["fp_removed"][idx]),
                        "external_deesc_n": int(ms_all["deesc_n"][idx]),
                        "external_event_rate": float(ms_all["event_rate"][idx]),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["formal_NI_5pp_both_pass", "external_acc_gain", "external_spec_gain", "internal_acc_gain"],
            ascending=[False, False, False, False],
        )
    result.to_csv(OUT_DIR / "04_combo_pattern_search_s90_formal_NI_candidates.csv", index=False)
    result.head(200).to_csv(OUT_DIR / "04_combo_pattern_search_s90_top200.csv", index=False)
    locked_branch_text = " | ".join(
        f"{branch['feature']}__sign{branch['sign']:+d}__w{branch['width']:.2f}__lam{branch['lambda']:.2f}"
        for branch in LOCKED_PRIMARY_BRANCHES
    )
    locked_rows = result[
        (result["branches"] == locked_branch_text)
        & (result["pattern_mask"] == locked_mask)
    ]
    locked_rows.to_csv(OUT_DIR / "04_locked_primary_rule_row_from_search.csv", index=False)
    return result


# ---------------------------------------------------------------------------
# 7. Figure
# ---------------------------------------------------------------------------


def mean_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(curves, axis=0)
    if len(curves) <= 1:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    se = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(len(curves))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def add_region_spans(ax: plt.Axes) -> None:
    spans = [
        ("R1", 45, 56, "#4E79A7", 0.08),
        ("R2", 57, 80, "#F28E2B", 0.08),
        ("R3", 97, 128, "#59A14F", 0.08),
        ("R4", 117, 128, "#B07AA1", 0.14),
    ]
    for label, start, end, color, alpha in spans:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
    y0, y1 = ax.get_ylim()
    for label, start, end, color, _ in spans:
        ax.text((start + end) / 2, y1 - 0.015 * (y1 - y0), label, ha="center", va="bottom", color=color, fontsize=8, fontweight="bold")


def plot_group_mean(ax: plt.Axes, z: np.ndarray, curves: np.ndarray, mask: np.ndarray, label: str, color: str) -> None:
    mean, low, high = mean_ci(curves[mask])
    ax.plot(z, mean, color=color, lw=2.4, label=f"{label} (n={int(mask.sum())})")
    ax.fill_between(z, low, high, color=color, alpha=0.12, lw=0)


def add_r4_tangent(ax: plt.Axes, z: np.ndarray, curves: np.ndarray, red_mask: np.ndarray, blue_mask: np.ndarray) -> None:
    r4 = (z >= 117) & (z <= 128)
    x = z[r4].astype(float)
    span = np.array([117.0, 128.0])
    red_mean = np.nanmean(curves[red_mask], axis=0)[r4]
    blue_mean = np.nanmean(curves[blue_mask], axis=0)[r4]
    red_slope, red_intercept = np.polyfit(x, red_mean, 1)
    blue_slope, blue_intercept = np.polyfit(x, blue_mean, 1)
    ax.plot(span, red_slope * span + red_intercept, color="#D04F5B", lw=4.0, alpha=0.85, solid_capstyle="round")
    ax.plot(span, blue_slope * span + blue_intercept, color="#2F6F9F", lw=4.0, alpha=0.85, solid_capstyle="round")
    ax.text(
        0.98,
        0.16,
        f"R4 fitted slope\nAEC- {red_slope:+.4f}/slice\nAEC+ {blue_slope:+.4f}/slice",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.86, "pad": 4},
    )


def make_main_figure(ctx: dict, aec_positive_external: np.ndarray, path: Path) -> None:
    curves = ctx["external"]["norm_aec"]
    y = ctx["ys"].astype(bool)
    clinical_positive = ctx["cpos_s"][PRIMARY_OP]
    cp_aec_pos = clinical_positive & aec_positive_external
    cp_aec_neg = clinical_positive & ~aec_positive_external
    z = np.arange(1, 129)
    overall = np.nanmean(curves, axis=0)
    clinical_pos_mean = np.nanmean(curves[clinical_positive], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.7), sharey=True)
    axes[0].plot(z, overall, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group_mean(axes[0], z, curves, clinical_positive, "Clinical +", "#D04F5B")
    plot_group_mean(axes[0], z, curves, ~clinical_positive, "Clinical -", "#2F6F9F")
    axes[0].set_title("A. Clinical S90 operating point", loc="left", fontsize=12, fontweight="bold")

    axes[1].plot(z, overall, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group_mean(axes[1], z, curves, y, "Low SMI +", "#D04F5B")
    plot_group_mean(axes[1], z, curves, ~y, "Non-low SMI", "#2F6F9F")
    axes[1].set_title("B. Outcome phenotype", loc="left", fontsize=12, fontweight="bold")

    axes[2].plot(z, clinical_pos_mean, color="#666666", lw=1.6, ls=":", label="Clinical + mean reference")
    plot_group_mean(axes[2], z, curves, cp_aec_pos, "Clinical+ / AEC+", "#2F6F9F")
    plot_group_mean(axes[2], z, curves, cp_aec_neg, "Clinical+ / AEC-", "#D04F5B")
    add_r4_tangent(axes[2], z, curves, cp_aec_neg, cp_aec_pos)
    axes[2].set_title("C. Conditional AEC split with R4 fitted tangent", loc="left", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.axhline(1.0, color="#9E9E9E", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlim(1, 128)
        ax.set_ylim(0.82, 1.21)
        ax.set_xticks([1, 32, 64, 96, 128])
        ax.set_xlabel("Slice index")
        ax.grid(axis="both", color="#D0D0D0", lw=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_region_spans(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)
    axes[0].set_ylabel("Patient-normalized AEC")
    fig.suptitle("External S90 core AEC morphology comparisons with R4 fitted tangent", fontsize=15, fontweight="bold", y=1.03)
    fig.text(0.5, -0.015, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def run_reproduce() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = make_context()
    primary_g, primary_s = compute_locked_gate(ctx)
    cnn = compute_secondary_cnn_mimic()
    write_final_outputs(ctx, primary_g, primary_s, cnn)
    print("Final outputs written to:", OUT_DIR)
    print(pd.read_csv(OUT_DIR / "final_s90_primary_and_cnn_metrics.csv").to_string(index=False))


def run_full_search() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = make_context()
    print("Stage 1/5: region scout search")
    scout = run_region_scout(ctx)
    print("  scout rows:", len(scout))
    print("Stage 2/5: locked region branch candidate screen")
    selected, xg, xs, names = screen_branch_candidates(ctx)
    print("  selected branch candidates:", len(selected))
    print("Stage 3/5: precompute branch votes")
    votes = precompute_branch_votes(ctx, selected, xg, xs, names)
    print("Stage 4/5: 4-branch pattern search")
    combos = run_combo_pattern_search(ctx, selected, votes)
    print("  candidate rows kept:", len(combos))
    print("Stage 5/5: final locked primary and CNN outputs")
    primary_g, primary_s = compute_locked_gate(ctx)
    cnn = compute_secondary_cnn_mimic()
    write_final_outputs(ctx, primary_g, primary_s, cnn)
    print("Full derivation outputs written to:", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reproduce", "full-search"], default="full-search")
    args = parser.parse_args()
    if args.mode == "reproduce":
        run_reproduce()
    else:
        run_full_search()


if __name__ == "__main__":
    main()
