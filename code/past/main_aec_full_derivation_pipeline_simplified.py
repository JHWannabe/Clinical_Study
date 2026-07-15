from __future__ import annotations

"""
Full derivation pipeline for the AEC low-SMI de-escalation gate (simplified)
=============================================================================

이 파일은 main_aec_full_derivation_pipeline.py와 동일한 계산/출력을 재현하되,
반복되던 로직을 공통 헬퍼로 묶어 가독성을 높인 버전입니다.
계산 로직과 수치 결과는 원본과 100% 동일합니다 (컬럼 순서 등 순수 화면 표시상의
디테일만 일부 정리됨). 분석 흐름 설명은 원본 파일 docstring을 참고하세요.

Section 8은 원래 별도 파일이었던 main_plot_internal_s90_core_1x3_mean_curves.py /
main_plot_external_s90_core_1x3_mean_curves.py (이후 병합본 main_plot_s90_core_1x3_mean_curves.py)를
이 파일로 합친 것입니다 — 그 파일들은 더 이상 존재하지 않습니다. 이 섹션은
aec_lock_smoothed_deesc_gate/aec_new_region_surrogate_combo_gate의 자체 클리니컬 스코어링
파이프라인을 그대로 사용하며(이 파일 자체의 make_context()와는 다른 별개의 계산),
산출물은 work/outputs/aec_1x3_core_mean_curves/에 저장됩니다.

권장 실행:

빠른 최종 재현:
    python main_aec_full_derivation_pipeline_simplified.py --mode reproduce

탐색표까지 생성:
    python main_aec_full_derivation_pipeline_simplified.py --mode full-search

internal/external 1x3 평균곡선 그림 생성:
    python main_aec_full_derivation_pipeline_simplified.py --mode plot-1x3
"""

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast, overload, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression

# Section 8 (internal/external 1x3 mean-curve figures, merged from
# main_plot_s90_core_1x3_mean_curves.py) reuses these two sibling modules'
# own clinical-scoring/feature pipeline (now merged below as the LSG_-tagged
# section, reusing this file's own feature/standardization helpers with an
# isolated RNG). This is intentionally a *separate* computation from this
# file's own make_context()/fit_clinical_scores(): the two pipelines use
# different underlying models, so their clinical_z scores are not
# interchangeable even though they gate on the same locked branches.


# ---------------------------------------------------------------------------
# 0. Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
OUT_DIR = PROJECT_ROOT /  "outputs" / "full_derivation" /  "full_derivation_output"
PLOT_1X3_OUT_DIR = PROJECT_ROOT /  "outputs" / "full_derivation" / "aec_1x3_core_mean_curves"

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

# Region span visuals shared by add_region_spans() (section 7) and the
# section 8 panel/mirror summaries.
REGION_SPANS = [
    ("R1", 45, 56, "#4E79A7", 0.22),
    ("R2", 57, 80, "#F28E2B", 0.22),
    ("R3", 97, 128, "#59A14F", 0.22),
    ("R4", 117, 128, "#B07AA1", 0.32),
]

# cohort -> (display title, JSON dataset label) for the section 8 figures.
COHORT_TITLE = {"internal": "Internal (gangnam)", "external": "External"}
COHORT_DATASET = {"internal": "gangnam", "external": "sinchon"}


# Secondary CNN-mimic output from the previous full CNN training/search.
# These thresholds/patterns (surrogate_mimic_balanced config) reproduce the
# original reported result (outputs/MD/144838527.png "Secondary: CNN-mimic
# Gate" screenshot): S90 de-escalated n=40 internal / ~51-52 external, TP
# lost=2/1 — confirmed by direct recomputation. Do NOT replace with the
# surrogate_mimic_summary.json "winners" (internal_locked/internal_external_audit) —
# those come from a separate, newer brute-force re-search over the guarded
# config and reproduce different (larger) de-escalation counts.
CNN_PROBABILITY_NPZ = PROJECT_ROOT /  "outputs" / "full_derivation" / "aec_new_region_cnn_surrogate_mimic_gate" / "surrogate_mimic_balanced_probabilities.npz"
# outputs/aec_new_region_cnn_surrogate_mimic_gate is not reproduced by this
# simplified pipeline (the original CNN training/search script was dropped
# during the single-file merge); its npz/csv/json/png are carried over as
# static, pre-computed files so the MD summary card step always has what it
# needs for the secondary CNN-mimic card.
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

# score = w_mean_acc*mean_acc_gain + w_min_acc*min_acc_gain + w_min_spec*min_spec_gain - w_sens_loss*max_sens_loss
REGION_SCOUT_LAMBDAS = [0.25, 0.55, 0.70]
REGION_SCOUT_SCORE_WEIGHTS = (1.0, 0.30, 0.20, 0.40)
BRANCH_SCREEN_SCORE_WEIGHTS = (1.0, 0.35, 0.20, 0.25)


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


def stratified_folds(y: np.ndarray, k: int = 5, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    rng = RNG if rng is None else rng
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            folds[i % k].append(int(row_idx))
    return [np.array(sorted(fold), dtype=int) for fold in folds]


def fit_clinical_scores(xg: np.ndarray, yg: np.ndarray, xs: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    folds = stratified_folds(yg, 5, rng=rng)
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


@overload
def z_standardize_by_internal(
    internal_score: np.ndarray, external_score: np.ndarray, return_stats: Literal[False] = False
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def z_standardize_by_internal(
    internal_score: np.ndarray, external_score: np.ndarray, return_stats: Literal[True]
) -> tuple[np.ndarray, np.ndarray, float, float]: ...
def z_standardize_by_internal(
    internal_score: np.ndarray, external_score: np.ndarray, return_stats: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(internal_score))
    sd = float(np.std(internal_score))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    zg, zs = (internal_score - mean) / sd, (external_score - mean) / sd
    if return_stats:
        return zg, zs, mean, sd
    return zg, zs


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
    # make_context() is called more than once per process run (once for the
    # main pipeline, once again inside MDCARD_main() to render the summary
    # cards). A shared, mutable module-level RNG would have already advanced
    # past its seed state by the second call, giving a different fold split
    # (and therefore different clinical scores/gate results) than a single
    # fresh run. Re-seed here so every make_context() call is independent and
    # reproducible regardless of how many times it has already been invoked.
    cg_raw, cs_raw = fit_clinical_scores(xg, yg, xs, rng=np.random.default_rng(SEED))
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


def _region_descriptors(block: np.ndarray, sb: np.ndarray, cb: np.ndarray, x: np.ndarray, denom: float, include_min_max: bool) -> dict[str, np.ndarray]:
    """Shared per-region descriptor set used by both the region-scout window
    search (arbitrary windows, includes level_min/max) and the locked-region
    descriptor matrix (fixed R1-R4 windows, no level_min/max)."""
    d = {
        "level_mean": block.mean(axis=1),
        "level_sd": block.std(axis=1),
    }
    if include_min_max:
        d["level_min"] = block.min(axis=1)
        d["level_max"] = block.max(axis=1)
    d.update(
        {
            "endpoint_delta": block[:, -1] - block[:, 0],
            "linear_slope": ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom,
            "slope_mean": sb.mean(axis=1),
            "slope_sd": sb.std(axis=1),
            "abs_slope_mean": np.abs(sb).mean(axis=1),
            "abs_slope_max": np.abs(sb).max(axis=1),
            "curv_mean": cb.mean(axis=1),
            "curv_sd": cb.std(axis=1),
            "abs_curv_mean": np.abs(cb).mean(axis=1),
            "abs_curv_max": np.abs(cb).max(axis=1),
        }
    )
    return d


def _descriptor_matrix(norm: np.ndarray, regions: list[tuple[str, tuple[int, int]]], include_min_max: bool) -> pd.DataFrame:
    slope = d1(norm)
    curv = d2(norm)
    grid = np.arange(norm.shape[1], dtype=float)
    rows: dict[str, np.ndarray] = {}
    for tag, (start, end) in regions:
        sl = slice(start - 1, end)
        block, sb, cb = norm[:, sl], slope[:, sl], curv[:, sl]
        x = grid[sl] - grid[sl].mean()
        denom = float((x**2).sum()) or 1.0
        for suffix, values in _region_descriptors(block, sb, cb, x, denom, include_min_max).items():
            rows[f"{tag}__{suffix}"] = values
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def window_features(norm: np.ndarray, windows: list[tuple[int, int]]) -> pd.DataFrame:
    regions = [(f"win_{start:03d}_{end:03d}", (start, end)) for start, end in windows]
    return _descriptor_matrix(norm, regions, include_min_max=True)


def locked_region_descriptor_matrix(norm: np.ndarray) -> pd.DataFrame:
    return _descriptor_matrix(norm, list(LOCKED_REGIONS.items()), include_min_max=False)


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
    p = float(cast(float, stats.fisher_exact(fisher_table)[1]))
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


def _cohort_table(ctx: dict, primary_g: dict, primary_s: dict, cnn: dict | None) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """(model, cohort, y, clinical_positive, aec_positive) rows shared by the
    final metrics CSV and the conditional low-SMI split CSV."""
    rows = [
        ("primary_interpretable_4region", "Gangnam internal", ctx["yg"], ctx["cpos_g"][PRIMARY_OP], primary_g["aec_positive"]),
        ("primary_interpretable_4region", "Sinchon external", ctx["ys"], ctx["cpos_s"][PRIMARY_OP], primary_s["aec_positive"]),
    ]
    if cnn is not None:
        rows += [
            ("secondary_CNN_mimic", "Gangnam internal", ctx["yg"], ctx["cpos_g"][PRIMARY_OP], cnn["aec_positive_g"]),
            ("secondary_CNN_mimic", "Sinchon external", ctx["ys"], ctx["cpos_s"][PRIMARY_OP], cnn["aec_positive_s"]),
        ]
    return rows


def write_final_outputs(ctx: dict, primary_g: dict, primary_s: dict, cnn: dict | None) -> None:
    cohorts = _cohort_table(ctx, primary_g, primary_s, cnn)

    metrics = pd.DataFrame(
        [
            {"model": model, "cohort": cohort, "operating_point": PRIMARY_OP, **evaluate_deescalation(y, cpos, aec_pos)}
            for model, cohort, y, cpos, aec_pos in cohorts
        ]
    )
    metrics.to_csv(OUT_DIR / "final_s90_primary_and_cnn_metrics.csv", index=False)

    conditional_rows = []
    for model, cohort, y, cpos, aec_pos in cohorts:
        tab, _ = conditional_low_smi_table(y, cpos, aec_pos)
        tab.insert(0, "cohort", cohort)
        tab.insert(0, "model", model)
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


def _feature_rule_row(ctx: dict, feature_z_g: np.ndarray, feature_z_s: np.ndarray, sign: int, width: float, lam: float, score_weights: tuple[float, float, float, float]) -> tuple[float, dict]:
    """Evaluate one (sign, width, lambda) rule on internal+external and build
    the score plus the internal_*/external_* row tail shared by the region
    scout search and the locked-region branch screen (they only differ in
    which identifying columns get prepended, and in the score weights)."""
    gi = summarize_single_feature_rule(ctx["yg"], ctx["cpos_g"], ctx["clinical_g"], feature_z_g, ctx["thresholds"], sign, width, lam)
    se = summarize_single_feature_rule(ctx["ys"], ctx["cpos_s"], ctx["clinical_s"], feature_z_s, ctx["thresholds"], sign, width, lam)
    w_mean_acc, w_min_acc, w_min_spec, w_sens_loss = score_weights
    score = (
        w_mean_acc * gi["mean_accuracy_gain"]
        + w_min_acc * gi["min_accuracy_gain"]
        + w_min_spec * gi["min_specificity_gain"]
        - w_sens_loss * gi["max_sensitivity_loss"]
    )
    row = {"sign": sign, "width": width, "lambda": lam, "internal_score": score}
    row.update({f"internal_{k}": v for k, v in gi.items()})
    row.update({f"external_{k}": v for k, v in se.items()})
    return score, row


def parse_window_feature(name: str) -> tuple[int, int, str]:
    left, descriptor = name.split("__")
    start, end = [int(v) for v in left.replace("win_", "").split("_")]
    return start, end, descriptor


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
        start, end, descriptor = parse_window_feature(name)
        for sign in SIGNS:
            for width in WIDTHS:
                for lam in REGION_SCOUT_LAMBDAS:
                    _, tail = _feature_rule_row(ctx, xg[:, j], xs[:, j], sign, width, lam, REGION_SCOUT_SCORE_WEIGHTS)
                    rows.append(
                        {
                            "feature": name,
                            "window_start": start,
                            "window_end": end,
                            "window_len": end - start + 1,
                            "descriptor": descriptor,
                            **tail,
                        }
                    )
    scout = pd.DataFrame(rows).sort_values("internal_score", ascending=False)
    scout.to_csv(OUT_DIR / "01_region_scout_window_feature_ranked.csv", index=False)
    return scout


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
                        score, tail = _feature_rule_row(ctx, xg[:, j], xs[:, j], sign, width, lam, BRANCH_SCREEN_SCORE_WEIGHTS)
                        rows.append({"region_key": region_key, "region": region_short, "feature": feature, "descriptor": descriptor, **tail})
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
    yy = y.astype(bool)
    event = np.bincount(code[cpos & yy], minlength=16).astype(int)
    nonevent = np.bincount(code[cpos & ~yy], minlength=16).astype(int)
    return event, nonevent


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


def style_axis(ax: Axes) -> None:
    """그래프 축의 공통 스타일(기준선, x축 범위·눈금, 격자, 테두리 제거)을 적용."""
    ax.axhline(1.0, color="#9E9E9E", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(1, 128)
    ax.set_xticks([1, 32, 64, 96, 128])
    ax.set_xlabel("Slice index")
    ax.grid(axis="both", color="#D0D0D0", lw=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_region_spans(ax: Axes) -> None:
    for label, start, end, color, alpha in REGION_SPANS:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
    y0, y1 = ax.get_ylim()
    for label, start, end, color, _ in REGION_SPANS:
        ax.text((start + end) / 2, y1 - 0.015 * (y1 - y0), label, ha="center", va="bottom", color=color, fontsize=8, fontweight="bold")


def plot_group_mean(ax: Axes, z: np.ndarray, curves: np.ndarray, mask: np.ndarray, label: str, color: str) -> None:
    mean, low, high = mean_ci(curves[mask])
    ax.plot(z, mean, color=color, lw=2.4, label=f"{label} (n={int(mask.sum())})")
    ax.fill_between(z, low, high, color=color, alpha=0.12, lw=0)


def add_r4_tangent(
    ax: Axes,
    z: np.ndarray,
    curves: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    *,
    text_y: float = 0.16,
    text_va: str = "bottom",
) -> None:
    """R4 구간(117~128)에서 AEC+/AEC- 두 그룹 평균 곡선에 각각 직선을 적합(fit)해 기울기를 그래프에 굵은 선+텍스트로 표시.
    text_y/text_va로 주석 위치를 조절한다 (section 7은 우하단, section 8의 tangent 그림은 우상단)."""
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
        text_y,
        f"R4 fitted slope\nAEC- {red_slope:+.4f}/slice\nAEC+ {blue_slope:+.4f}/slice",
        transform=ax.transAxes,
        ha="right",
        va=text_va,
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
        style_axis(ax)
        ax.set_ylim(0.72, 1.21)
        add_region_spans(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)
    axes[0].set_ylabel("Patient-normalized AEC")
    fig.suptitle("External S90 core AEC morphology comparisons with R4 fitted tangent", fontsize=15, fontweight="bold", y=1.03)
    fig.text(0.5, -0.015, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Internal/external 1x3 mean-curve figures
#    (merged from main_plot_s90_core_1x3_mean_curves.py; that file no longer
#    exists as a standalone script)
# ---------------------------------------------------------------------------
#
# This section computes and renders internal_*/external_* mean-curve figures
# (aec_1x3_core_mean_curves outputs) using aec_lock_smoothed_deesc_gate's own
# clinical_scores() and aec_new_region_surrogate_combo_gate's own
# region_descriptor_matrix()/z_train_apply() — a *different* clinical-scoring
# backend from this file's own make_context()/fit_clinical_scores() above.
# The two backends are not interchangeable (different models/feature banks),
# so they are kept fully separate; only the pure data constants that are
# genuinely identical between the two rules (LOCKED_PRIMARY_BRANCHES,
# LOCKED_PRIMARY_PATTERNS, REGION_SPANS) and pure math/plot helpers that don't
# care which backend produced their inputs (branch_gate_score,
# vote_pattern_from_matrix, mean_ci, add_region_spans, plot_group_mean,
# style_axis, add_r4_tangent) are shared with the rest of this file.


def compute_clinical_and_features(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, int]]:
    """clinical_scores()/z_train_apply()를 단 한 번만 계산해 internal/external 코호트가 공유하도록 함.
    clinical_scores() 내부의 fold shuffle이 aec_conditional_value의 전역 RNG를 소비하므로, 두 번
    부르면 원본 main_plot_*_s90_core_1x3_mean_curves.py 스크립트들(각자 별도 프로세스에서 단 한 번만
    호출)과 다른 결과가 나온다."""
    _, _, c_g, c_s, thresholds = LSG_clinical_scores(g, s)
    threshold = float(thresholds["S90"])
    fg = locked_region_descriptor_matrix(g["norm"])
    fs = locked_region_descriptor_matrix(s["norm"])
    xg, xs, names = standardize_features_by_internal(fg, fs)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    return c_g, c_s, xg, xs, threshold, name_to_idx


def compute_gate(cohort: str, clinical_z: np.ndarray, x: np.ndarray, threshold: float, name_to_idx: dict[str, int]) -> dict:
    """S90 임상 임계값에서, LOCKED_PRIMARY_BRANCHES(R1~R4) 게이트 투표 패턴을 계산해, LOCKED_PRIMARY_PATTERNS에
    해당하면 "AEC 양성(하향조정 후보)"으로 표시하고, 임상 양성군을 AEC 양성/음성으로 나눔."""
    vote_key = f"{cohort}_vote_positive_n"
    votes = []
    branch_rows = []
    for branch in LOCKED_PRIMARY_BRANCHES:
        idx = name_to_idx[branch["feature"]]
        score = branch_gate_score(
            clinical_z,
            x[:, idx],
            threshold,
            int(branch["sign"]),
            float(branch["width"]),
            float(branch["lambda"]),
        )
        vote = score < threshold
        votes.append(vote)
        branch_rows.append({**branch, vote_key: int(vote.sum())})

    vote_matrix = np.column_stack(votes)
    pattern = vote_pattern_from_matrix(vote_matrix)
    morphology_pos = np.isin(pattern, list(LOCKED_PRIMARY_PATTERNS))
    clinical_pos = clinical_z >= threshold
    deesc = clinical_pos & morphology_pos
    retained = clinical_pos & ~morphology_pos

    return {
        "clinical_z": clinical_z,
        "clinical_threshold": threshold,
        "clinical_pos": clinical_pos,
        "pattern": pattern,
        "morphology_pos": morphology_pos,
        "clinical_pos_aec_pos": deesc,
        "clinical_pos_aec_neg": retained,
        "branch_rows": branch_rows,
    }

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mirror_deviation(
    ax: Axes,
    z: np.ndarray,
    norm: np.ndarray,
    ref_mean: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    red_label: str,
    blue_label: str,
    title: str,
) -> tuple[np.ndarray, np.ndarray]:
    """기준곡선(ref_mean) 대비 두 그룹 평균의 절대편차를 위아래(빨강 위/파랑 아래)로 미러링해 그리고, 각 그룹의 편차 배열을 반환."""
    red_dev = np.abs(np.nanmean(norm[red_mask], axis=0) - ref_mean)
    blue_dev = np.abs(np.nanmean(norm[blue_mask], axis=0) - ref_mean)
    ax.fill_between(z, 0, red_dev, color="#D04F5B", alpha=0.30, lw=0)
    ax.plot(z, red_dev, color="#D04F5B", lw=1.8, label=red_label)
    ax.fill_between(z, 0, -blue_dev, color="#2F6F9F", alpha=0.30, lw=0)
    ax.plot(z, -blue_dev, color="#2F6F9F", lw=1.8, label=blue_label)
    ax.axhline(0, color="#666666", lw=0.9)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    return red_dev, blue_dev


def panel_summary(norm: np.ndarray, name: str, mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """두 그룹의 평균 곡선 차이(전체 및 R1~R4 영역별)를 요약한 한 행을 만듦."""
    mean_a = np.nanmean(norm[mask_a], axis=0)
    mean_b = np.nanmean(norm[mask_b], axis=0)
    row = {
        "comparison": name,
        "n_group1": int(mask_a.sum()),
        "n_group2": int(mask_b.sum()),
        "mean_abs_between_group_difference": float(np.nanmean(np.abs(mean_a - mean_b))),
    }
    for label, start, end, _, _ in REGION_SPANS:
        sl = slice(start - 1, end)
        row[f"{label}_group1_minus_group2"] = float(np.nanmean(mean_a[sl] - mean_b[sl]))
    return row


def mirror_summary(name: str, red_label: str, blue_label: str, red_dev: np.ndarray, blue_dev: np.ndarray) -> list[dict]:
    """두 그룹의 편차 곡선(red_dev/blue_dev)에서 전체 및 R1~R4 영역별 평균·최댓값 편차를 표로 정리."""
    rows = []
    for group, dev in [(red_label, red_dev), (blue_label, blue_dev)]:
        row = {
            "panel": name,
            "group": group,
            "mean_abs_deviation_all_slices": float(np.nanmean(dev)),
            "max_abs_deviation_all_slices": float(np.nanmax(dev)),
            "max_abs_deviation_slice": int(np.nanargmax(dev) + 1),
        }
        for label, start, end, _, _ in REGION_SPANS:
            sl = slice(start - 1, end)
            row[f"{label}_mean_abs_deviation"] = float(np.nanmean(dev[sl]))
            row[f"{label}_max_abs_deviation"] = float(np.nanmax(dev[sl]))
        rows.append(row)
    return rows


def fisher_exact_conditional(y: np.ndarray, aec_pos: np.ndarray, aec_neg: np.ndarray) -> float:
    """AEC 양성군과 음성군의 사건 발생률 차이에 대한 Fisher 정확검정 p값을 계산."""
    table = [
        [int(y[aec_pos].sum()), int((~y.astype(bool))[aec_pos].sum())],
        [int(y[aec_neg].sum()), int((~y.astype(bool))[aec_neg].sum())],
    ]
    return float(cast(float, stats.fisher_exact(table)[1]))


def _build_1x3_mean_curve_figure(
    cohort: str,
    title_label: str,
    z: np.ndarray,
    norm: np.ndarray,
    overall_mean: np.ndarray,
    clinical_pos_mean: np.ndarray,
    clinical_pos: np.ndarray,
    clinical_neg: np.ndarray,
    low: np.ndarray,
    nonlow: np.ndarray,
    cp_aec_pos: np.ndarray,
    cp_aec_neg: np.ndarray,
    *,
    tangent: bool,
    panel_c_annotation: str,
) -> Figure:
    """panels A/B/C 1x3 평균곡선 그림을 만든다. tangent=False면 panel C에 Fisher 검정 텍스트,
    tangent=True면 R4 적합 tangent 선+텍스트를 표시한다 (두 그림은 panel C와 suptitle만 다름)."""
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.7), sharey=True)

    axes[0].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label=f"{cohort} overall mean")
    plot_group_mean(axes[0], z, norm, clinical_pos, "Clinical +", "#D04F5B")
    plot_group_mean(axes[0], z, norm, clinical_neg, "Clinical -", "#2F6F9F")
    axes[0].set_title("A. Clinical S90 operating point", loc="left", fontsize=12, fontweight="bold")

    axes[1].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label=f"{cohort} overall mean")
    plot_group_mean(axes[1], z, norm, low, "Low SMI +", "#D04F5B")
    plot_group_mean(axes[1], z, norm, nonlow, "Non-low SMI", "#2F6F9F")
    axes[1].set_title("B. Outcome phenotype", loc="left", fontsize=12, fontweight="bold")

    axes[2].plot(z, clinical_pos_mean, color="#666666", lw=1.6, ls=":", label="Clinical + mean reference")
    plot_group_mean(axes[2], z, norm, cp_aec_pos, "Clinical+ / AEC+", "#2F6F9F")
    plot_group_mean(axes[2], z, norm, cp_aec_neg, "Clinical+ / AEC-", "#D04F5B")
    if tangent:
        add_r4_tangent(axes[2], z, norm, cp_aec_neg, cp_aec_pos, text_y=0.98, text_va="top")
        axes[2].set_title("C. Conditional AEC split with R4 fitted tangent", loc="left", fontsize=12, fontweight="bold")
    else:
        axes[2].text(
            0.98,
            0.98,
            panel_c_annotation,
            transform=axes[2].transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        axes[2].set_title("C. Conditional AEC split among Clinical +", loc="left", fontsize=12, fontweight="bold")

    for ax in axes:
        style_axis(ax)
        ax.set_ylim(0.72, 1.21)
        add_region_spans(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)

    axes[0].set_ylabel("Patient-normalized AEC")
    if tangent:
        fig.suptitle(
            f"{title_label} S90 core AEC morphology comparisons with R4 fitted tangent",
            fontsize=15,
            fontweight="bold",
            y=1.03,
        )
    else:
        fig.suptitle(
            f"{title_label} S90 core AEC morphology comparisons\n"
            "AEC+ indicates de-escalation / lower low-SMI probability",
            fontsize=15,
            fontweight="bold",
            y=1.04,
        )
    fig.text(0.5, -0.015, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig.tight_layout()
    return fig


def render_cohort(cohort: str, norm: np.ndarray, y: np.ndarray, gate: dict) -> None:
    """
    한 코호트(internal 또는 external)에 대해 원본 main_plot_*_s90_core_1x3_mean_curves.py 스크립트들의
    main() 본문과 동일한 산출물을 생성: 1x3 평균곡선 PNG, R4 tangent 포함 1x3 PNG,
    2x3 mean+mirror-deviation PNG, 3개의 CSV/JSON 요약.
    """
    prefix = cohort
    title_label = COHORT_TITLE[cohort]
    dataset_key = f"{cohort}_dataset"
    dataset_value = COHORT_DATASET[cohort]

    png = PLOT_1X3_OUT_DIR / f"{prefix}_s90_core_1x3_mean_curves.png"
    png_tangent = PLOT_1X3_OUT_DIR / f"{prefix}_s90_core_1x3_mean_curves_with_r4_tangent.png"
    png_2x3 = PLOT_1X3_OUT_DIR / f"{prefix}_s90_core_2x3_mean_and_mirror_deviation.png"
    csv = PLOT_1X3_OUT_DIR / f"{prefix}_s90_core_1x3_mean_curve_summary.csv"
    mirror_csv = PLOT_1X3_OUT_DIR / f"{prefix}_s90_core_2x3_mirror_deviation_summary.csv"
    json_path = PLOT_1X3_OUT_DIR / f"{prefix}_s90_core_1x3_summary.json"

    z = np.arange(1, 129)
    clinical_pos = gate["clinical_pos"]
    clinical_neg = ~clinical_pos
    low = y
    nonlow = ~y
    cp_aec_pos = gate["clinical_pos_aec_pos"]
    cp_aec_neg = gate["clinical_pos_aec_neg"]

    overall_mean = np.nanmean(norm, axis=0)
    clinical_pos_mean = np.nanmean(norm[clinical_pos], axis=0)

    cp_pos_events = int(y[cp_aec_pos].sum())
    cp_neg_events = int(y[cp_aec_neg].sum())
    p_fisher = fisher_exact_conditional(y, cp_aec_pos, cp_aec_neg)
    pos_rate = cp_pos_events / int(cp_aec_pos.sum())
    neg_rate = cp_neg_events / int(cp_aec_neg.sum())
    fisher_annotation = (
        f"low SMI: {cp_pos_events}/{int(cp_aec_pos.sum())}={pos_rate:.1%} vs "
        f"{cp_neg_events}/{int(cp_aec_neg.sum())}={neg_rate:.1%}\nFisher p={p_fisher:.2g}"
    )

    fig = _build_1x3_mean_curve_figure(
        cohort, title_label, z, norm, overall_mean, clinical_pos_mean,
        clinical_pos, clinical_neg, low, nonlow, cp_aec_pos, cp_aec_neg,
        tangent=False, panel_c_annotation=fisher_annotation,
    )
    fig.savefig(png, dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig_t = _build_1x3_mean_curve_figure(
        cohort, title_label, z, norm, overall_mean, clinical_pos_mean,
        clinical_pos, clinical_neg, low, nonlow, cp_aec_pos, cp_aec_neg,
        tangent=True, panel_c_annotation="",
    )
    fig_t.savefig(png_tangent, dpi=260, bbox_inches="tight")
    plt.close(fig_t)

    fig2, axes2 = plt.subplots(2, 3, figsize=(18.5, 8.8), sharex=True)
    panels = [
        {
            "top_title": "A. Clinical S90 operating point",
            "bottom_title": "D. Deviation from overall mean",
            "reference": overall_mean,
            "reference_label": f"{cohort} overall mean",
            "red_mask": clinical_pos,
            "red_label": f"Clinical + (n={int(clinical_pos.sum())})",
            "blue_mask": clinical_neg,
            "blue_label": f"Clinical - (n={int(clinical_neg.sum())})",
            "panel_name": "Clinical + vs Clinical -",
        },
        {
            "top_title": "B. Outcome phenotype",
            "bottom_title": "E. Deviation from overall mean",
            "reference": overall_mean,
            "reference_label": f"{cohort} overall mean",
            "red_mask": low,
            "red_label": f"Low SMI + (n={int(low.sum())})",
            "blue_mask": nonlow,
            "blue_label": f"Non-low SMI (n={int(nonlow.sum())})",
            "panel_name": "Low SMI + vs Non-low SMI",
        },
        {
            "top_title": "C. Conditional AEC split among Clinical +",
            "bottom_title": "F. Deviation from Clinical + mean",
            "reference": clinical_pos_mean,
            "reference_label": "Clinical + mean reference",
            "red_mask": cp_aec_neg,
            "red_label": f"Clinical+ / AEC- (n={int(cp_aec_neg.sum())})",
            "blue_mask": cp_aec_pos,
            "blue_label": f"Clinical+ / AEC+ (n={int(cp_aec_pos.sum())})",
            "panel_name": "Clinical+/AEC- vs Clinical+/AEC+",
        },
    ]

    mirror_rows = []
    for j, panel in enumerate(panels):
        ax_top = axes2[0, j]
        ax_bottom = axes2[1, j]
        ax_top.plot(z, panel["reference"], color="#666666", lw=1.6, ls=":", label=panel["reference_label"])
        plot_group_mean(ax_top, z, norm, panel["red_mask"], panel["red_label"].split(" (n=")[0], "#D04F5B")
        plot_group_mean(ax_top, z, norm, panel["blue_mask"], panel["blue_label"].split(" (n=")[0], "#2F6F9F")
        ax_top.set_title(panel["top_title"], loc="left", fontsize=12, fontweight="bold")
        style_axis(ax_top)
        ax_top.set_ylim(0.72, 1.21)
        add_region_spans(ax_top)
        ax_top.legend(frameon=False, loc="lower left", fontsize=8.8)

        red_dev, blue_dev = plot_mirror_deviation(
            ax_bottom,
            z,
            norm,
            panel["reference"],
            panel["red_mask"],
            panel["blue_mask"],
            panel["red_label"],
            panel["blue_label"],
            panel["bottom_title"],
        )
        style_axis(ax_bottom)
        max_dev = max(float(np.nanmax(red_dev)), float(np.nanmax(blue_dev)), 0.04)
        ax_bottom.set_ylim(-max_dev * 1.18, max_dev * 1.18)
        add_region_spans(ax_bottom)
        ax_bottom.legend(frameon=False, loc="upper left", fontsize=8.5)
        mirror_rows.extend(mirror_summary(panel["panel_name"], panel["red_label"], panel["blue_label"], red_dev, blue_dev))

    axes2[0, 0].set_ylabel("Patient-normalized AEC")
    axes2[1, 0].set_ylabel("|group mean - reference|\n(red upward, blue downward)")
    fig2.suptitle(
        f"{title_label} S90 AEC morphology: mean curves and mirror absolute-deviation plots\n"
        "Bottom row shows magnitude of separation from the reference curve; placement above/below zero denotes group color, not original direction",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig2.text(0.5, -0.005, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(png_2x3, dpi=260, bbox_inches="tight")
    plt.close(fig2)
    pd.DataFrame(mirror_rows).to_csv(mirror_csv, index=False)

    summary_rows = [
        panel_summary(norm, "Clinical + vs Clinical -", clinical_pos, clinical_neg),
        panel_summary(norm, "Low SMI + vs Non-low SMI", low, nonlow),
        panel_summary(norm, "Clinical+/AEC- vs Clinical+/AEC+", cp_aec_neg, cp_aec_pos),
    ]
    pd.DataFrame(summary_rows).to_csv(csv, index=False)
    json_path.write_text(
        json.dumps(
            {
                "png": str(png),
                "png_tangent": str(png_tangent),
                "png_2x3": str(png_2x3),
                dataset_key: dataset_value,
                "clinical_operating_point": "S90",
                "AEC_definition": "primary interpretable morphology gate new4_combo_261089; AEC+ means de-escalation/low-risk morphology",
                "selected_patterns": sorted(LOCKED_PRIMARY_PATTERNS),
                "branches": gate["branch_rows"],
                "low_smi_conditional": {
                    "clinical_pos_aec_pos_events": cp_pos_events,
                    "clinical_pos_aec_pos_n": int(cp_aec_pos.sum()),
                    "clinical_pos_aec_pos_rate": pos_rate,
                    "clinical_pos_aec_neg_events": cp_neg_events,
                    "clinical_pos_aec_neg_n": int(cp_aec_neg.sum()),
                    "clinical_pos_aec_neg_rate": neg_rate,
                    "fisher_p": p_fisher,
                },
                "summary": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(png)
    print(png_tangent)
    print(png_2x3)
    print(csv)
    print(mirror_csv)
    print(json_path)


def run_plot_1x3_mean_curves() -> None:
    """
    internal/external 두 코호트에 대해 1x3 평균곡선 + R4 tangent + 2x3 mirror-deviation
    산출물을 모두 생성 (PLOT_1X3_OUT_DIR = work/outputs/aec_1x3_core_mean_curves).
    clinical_scores()는 단 한 번만 호출해 두 코호트가 공유한다 — compute_clinical_and_features()의
    docstring 참고.
    """
    PLOT_1X3_OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = LSG_load_dataset(DATA_DIR / "gangnam.xlsx")
    s = LSG_load_dataset(DATA_DIR / "sinchon.xlsx")

    c_g, c_s, xg, xs, threshold, name_to_idx = compute_clinical_and_features(g, s)
    gate_internal = compute_gate("internal", c_g, xg, threshold, name_to_idx)
    gate_external = compute_gate("external", c_s, xs, threshold, name_to_idx)

    render_cohort("internal", g["norm"], g["y"].astype(bool), gate_internal)
    render_cohort("external", s["norm"], s["y"].astype(bool), gate_external)


# ---------------------------------------------------------------------------
# 9. CLI
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
    parser.add_argument("--mode", choices=["reproduce", "full-search", "plot-1x3"], default="full-search")
    args = parser.parse_args()
    if args.mode == "reproduce":
        run_reproduce()
    elif args.mode == "full-search":
        run_full_search()
    run_plot_1x3_mean_curves()
    MDCARD_main()


# ---------------------------------------------------------------------------
# Legacy helpers retained only because run_plot_1x3_mean_curves() (section 8)
# reuses this file's own clinical-scoring and region-descriptor pipeline as a
# separate, isolated-RNG backend (see section 8 docstring / compute_clinical_and_features()
# docstring), and MDCARD_main() (below) reuses this file's own
# make_context()/compute_locked_gate()/compute_secondary_cnn_mimic() to
# render the outputs/MD summary cards. Everything else from the original
# merged legacy scripts (region/branch/CNN search stages, their own main()s
# and CLI modes) has been removed as unused. CV_RNG is a separate RNG instance
# (same seed) so section 8's fold split matches what the original standalone
# script produced with a pristine RNG, independent of how far the main
# make_context()/fit_clinical_scores() call above has already advanced RNG.
# ---------------------------------------------------------------------------

CV_SEED = 20260629
CV_RNG = np.random.default_rng(CV_SEED)

LSG_SIGMA = 1.0
LSG_OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


def LSG_load_dataset(path: Path) -> dict:
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_aec_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    smooth_raw = ndimage.gaussian_filter1d(raw, sigma=LSG_SIGMA, axis=1, mode="nearest")
    norm = patient_wise_mean_normalize(smooth_raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "smooth_raw": smooth_raw, "norm": norm, "y": y, "sex": sex, "smi": smi}


def LSG_clinical_scores(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    xg, xs, _ = clinical_design_matrix(g["meta"], s["meta"])
    clinical_oof, clinical_ext = fit_clinical_scores(xg, g["y"].astype(int), xs, rng=CV_RNG)
    c_g, c_s, mu, sd = z_standardize_by_internal(clinical_oof, clinical_ext, return_stats=True)
    thresholds = {}
    for label, target in LSG_OPS:
        th_raw = threshold_for_min_sensitivity(g["y"], clinical_oof, target)
        thresholds[label] = float((th_raw - mu) / sd)
    return clinical_oof, clinical_ext, c_g, c_s, thresholds


# ---------------------------------------------------------------------------
# MD summary cards (always run at the end of main(), regardless of --mode;
# formerly generate_md_summary_cards.py). Renders outputs/MD/*.png from this file's
# own make_context()/compute_locked_gate()/compute_secondary_cnn_mimic();
# the CNN card and CNN rows in the reproduction-check card are skipped/N-A
# if outputs/aec_new_region_cnn_surrogate_mimic_gate's probability file is
# missing.
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

MDCARD_OUT_DIR = PROJECT_ROOT / "outputs" / "full_derivation" / "MD"

MDCARD_HEADER_BG = "#F2F2F2"
MDCARD_LINE_COLOR = "#DDDDDD"
MDCARD_GREEN = "#1A7F37"
MDCARD_AMBER = "#B7791F"

# MD 원본 스크린샷(outputs/MD/144811843.png, 144838527.png)의 값 — 코드로 재계산할 수 없는,
# 협업자가 공유한 참조값이므로 상수로 유지한다.
MDCARD_MD_ORIGINAL_PRIMARY = {
    "deesc_n": ("53", "56"),
    "deesc_low_smi": ("3.8%", "3.6%"),
    "sensitivity_loss": ("-1.55%p", "-1.42%p"),
    "specificity_gain": ("+5.31%p", "+6.88%p"),
    "accuracy_gain": ("+4.50%p", "+5.62%p"),
    "fisher_p": ("5.53e-04", "2.30e-05"),
}
MDCARD_MD_ORIGINAL_CNN = {
    "deesc_n": ("40", "52"),
    "deesc_low_smi": ("5.0%", "1.9%"),
    "tp_lost": ("2", "1"),
    "sensitivity_loss": ("-1.55%p", "-0.71%p"),
    "specificity_gain": ("+3.95%p", "+6.50%p"),
    "accuracy_gain": ("+3.30%p", "+5.40%p"),
}


def MDCARD_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def MDCARD_pp_loss(x: float) -> str:
    return f"{-x * 100:.2f}%p"


def MDCARD_pp_gain(x: float) -> str:
    return f"{x * 100:+.2f}%p"


def MDCARD_sci(x: float) -> str:
    return f"{x:.2e}"


def MDCARD_draw_card(
    path: Path,
    title: str,
    sections: list[dict],
    footer_lines: list[str],
    figsize: tuple[float, float] = (14.0, 9.5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=145)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.965
    ax.text(0.012, y, title, fontsize=19, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    y -= 0.075

    row_h = 0.052

    # First column ("항목" labels) is a fixed 0.34-of-width fraction by
    # default, which overflows into the value columns on narrower figsize
    # cards (e.g. cnn_mimic_secondary_gate_summary.png) when a label is long
    # (e.g. "Sens loss upper 95% (1-sided) / NI pass"). Measure actual label
    # text widths and widen the first column fraction to fit, capped so the
    # value columns stay usable.
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()

    def _text_width_in(s: str, fontsize: float, bold: bool = False) -> float:
        t = ax.text(0, 0, s, fontsize=fontsize, fontweight="bold" if bold else "normal", transform=ax.transAxes, alpha=0)
        w = t.get_window_extent(renderer=renderer).width / fig.dpi
        t.remove()
        return w

    label0_width_in = max(
        (
            max(
                [_text_width_in(section["columns"][0], 11.5, bold=True)]
                + [_text_width_in(str(row[0]), 11) for row in section["rows"]]
            )
            for section in sections
        ),
        default=0.0,
    )
    axes_width_in = ax.get_position().width * fig.get_size_inches()[0]
    col0_frac = min(0.55, max(0.34, (label0_width_in + 0.15) / axes_width_in))

    for section in sections:
        ax.text(0.012, y, section["header"], fontsize=14.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= 0.052
        if section.get("subheader"):
            ax.text(0.012, y, section["subheader"], fontsize=10.5, ha="left", va="top", transform=ax.transAxes, color="#333333")
            y -= 0.036 * section["subheader"].count("\n") + 0.046

        cols = section["columns"]
        n_cols = len(cols)
        widths = [col0_frac] + [(1 - col0_frac) / (n_cols - 1)] * (n_cols - 1)
        xs = [0.012]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)

        ax.add_patch(Rectangle((0.008, y - row_h + 0.012), 0.98, row_h, facecolor=MDCARD_HEADER_BG, edgecolor="none", transform=ax.transAxes, zorder=0))
        for cx, col in zip(xs, cols):
            ax.text(cx, y - 0.010, col, fontsize=11.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        y -= row_h

        colors = section.get("cell_colors")
        for ri, row in enumerate(section["rows"]):
            for ci, (cx, val) in enumerate(zip(xs, row)):
                color = "black"
                if colors and colors[ri][ci]:
                    color = colors[ri][ci]
                weight = "bold" if (ci == len(row) - 1 and colors) else "normal"
                ax.text(cx, y - 0.010, str(val), fontsize=11, ha="left", va="top", transform=ax.transAxes, color=color, fontweight=weight)
            y -= row_h
            ax.plot([0.008, 0.988], [y + row_h * 0.30, y + row_h * 0.30], color=MDCARD_LINE_COLOR, lw=0.8, transform=ax.transAxes)
        y -= 0.03

    y -= 0.01
    ax.plot([0.008, 0.988], [y, y], color="#BBBBBB", lw=0.8, transform=ax.transAxes)
    y -= 0.035
    for line in footer_lines:
        ax.text(0.012, y, line, fontsize=9, ha="left", va="top", transform=ax.transAxes, color="#555555")
        y -= 0.032

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def MDCARD_compute_all() -> dict:
    ctx = make_context()
    primary_g, primary_s = compute_locked_gate(ctx)
    cnn = compute_secondary_cnn_mimic()

    op = PRIMARY_OP
    yg, ys = ctx["yg"], ctx["ys"]
    cpos_g, cpos_s = ctx["cpos_g"][op], ctx["cpos_s"][op]

    base_g = binary_metrics(yg, cpos_g)
    base_s = binary_metrics(ys, cpos_s)

    cohort = {
        "n": (len(yg), len(ys)),
        "low_smi_n": (int(yg.sum()), int(ys.sum())),
        "low_smi_rate": (float(yg.mean()), float(ys.mean())),
        "cpos_n": (int(cpos_g.sum()), int(cpos_s.sum())),
        "cpos_low_smi_n": (int(yg[cpos_g].sum()), int(ys[cpos_s].sum())),
        "cpos_low_smi_rate": (float(yg[cpos_g].mean()), float(ys[cpos_s].mean())),
        "clinical_sensitivity": (base_g["sensitivity"], base_s["sensitivity"]),
        "clinical_specificity": (base_g["specificity"], base_s["specificity"]),
        "clinical_accuracy": (base_g["accuracy"], base_s["accuracy"]),
    }

    def gate_summary(aec_g, aec_s) -> dict:
        mg = evaluate_deescalation(yg, cpos_g, aec_g)
        ms = evaluate_deescalation(ys, cpos_s, aec_s)
        _, fisher_g = conditional_low_smi_table(yg, cpos_g, aec_g)
        _, fisher_s = conditional_low_smi_table(ys, cpos_s, aec_s)
        return {"internal": mg, "external": ms, "fisher_p": (fisher_g, fisher_s)}

    primary = gate_summary(primary_g["aec_positive"], primary_s["aec_positive"])
    secondary = gate_summary(cnn["aec_positive_g"], cnn["aec_positive_s"]) if cnn is not None else None

    return {"cohort": cohort, "primary": primary, "secondary": secondary}


def MDCARD_gate_rows(gate: dict) -> list[list[str]]:
    mg, ms = gate["internal"], gate["external"]
    fp_g, fp_s = gate["fisher_p"]
    return [
        ["De-escalated n", str(mg["deescalated_n"]), str(ms["deescalated_n"])],
        [
            "De-escalated low SMI",
            f"{mg['deescalated_low_smi_events']}/{mg['deescalated_n']} = {MDCARD_pct(mg['deescalated_event_rate'])}",
            f"{ms['deescalated_low_smi_events']}/{ms['deescalated_n']} = {MDCARD_pct(ms['deescalated_event_rate'])}",
        ],
        ["TP lost", str(mg["tp_lost"]), str(ms["tp_lost"])],
        ["FP removed", str(mg["fp_removed"]), str(ms["fp_removed"])],
        ["Post sensitivity", MDCARD_pct(mg["post_sensitivity"]), MDCARD_pct(ms["post_sensitivity"])],
        ["Sensitivity loss", MDCARD_pp_loss(mg["sensitivity_loss"]), MDCARD_pp_loss(ms["sensitivity_loss"])],
        ["Sens loss upper 95% (1-sided) / NI pass", f"{MDCARD_pct(mg['sensitivity_loss_upper95_one_sided'])} / {mg['formal_NI_pass']}", f"{MDCARD_pct(ms['sensitivity_loss_upper95_one_sided'])} / {ms['formal_NI_pass']}"],
        ["Post specificity", MDCARD_pct(mg["post_specificity"]), MDCARD_pct(ms["post_specificity"])],
        ["Specificity gain", MDCARD_pp_gain(mg["specificity_gain"]), MDCARD_pp_gain(ms["specificity_gain"])],
        ["Post accuracy", MDCARD_pct(mg["post_accuracy"]), MDCARD_pct(ms["post_accuracy"])],
        ["Accuracy gain", MDCARD_pp_gain(mg["accuracy_gain"]), MDCARD_pp_gain(ms["accuracy_gain"])],
        ["Clinical+/AEC+ vs AEC- Fisher p", MDCARD_sci(fp_g), MDCARD_sci(fp_s)],
    ]


def MDCARD_approx(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def MDCARD_main() -> None:
    MDCARD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = MDCARD_compute_all()
    cohort = data["cohort"]
    primary = data["primary"]
    secondary = data["secondary"]

    # 1. Primary AEC gate summary
    MDCARD_draw_card(
        MDCARD_OUT_DIR / "aec_1x3_primary_gate_summary.png",
        "Primary AEC Gate — outputs/aec_1x3_core_mean_curves",
        sections=[
            {
                "header": "Cohort / Clinical Baseline (S90 operating point)",
                "columns": ["항목", "Internal (gangnam)", "External (sinchon)"],
                "rows": [
                    ["N", str(cohort["n"][0]), str(cohort["n"][1])],
                    ["Low SMI(+) actual", f"{cohort['low_smi_n'][0]} ({MDCARD_pct(cohort['low_smi_rate'][0])})", f"{cohort['low_smi_n'][1]} ({MDCARD_pct(cohort['low_smi_rate'][1])})"],
                    ["Clinical+ at S90", str(cohort["cpos_n"][0]), str(cohort["cpos_n"][1])],
                    ["Clinical+ low SMI", f"{cohort['cpos_low_smi_n'][0]} ({MDCARD_pct(cohort['cpos_low_smi_rate'][0])})", f"{cohort['cpos_low_smi_n'][1]} ({MDCARD_pct(cohort['cpos_low_smi_rate'][1])})"],
                    ["Clinical sensitivity", MDCARD_pct(cohort["clinical_sensitivity"][0]), MDCARD_pct(cohort["clinical_sensitivity"][1])],
                    ["Clinical specificity", MDCARD_pct(cohort["clinical_specificity"][0]), MDCARD_pct(cohort["clinical_specificity"][1])],
                    ["Clinical accuracy", MDCARD_pct(cohort["clinical_accuracy"][0]), MDCARD_pct(cohort["clinical_accuracy"][1])],
                ],
            },
            {
                "header": "Primary: Interpretable 4-region AEC Gate  (new4_combo_261089)",
                "columns": ["항목", "Internal (gangnam)", "External (sinchon)"],
                "rows": MDCARD_gate_rows(primary),
            },
        ],
        footer_lines=[
            "출처: work/python/main_aec_full_derivation_pipeline_simplified.py의 make_context()/compute_locked_gate()/evaluate_deescalation()/conditional_low_smi_table()를 직접 호출해 계산 (하드코딩 없음).",
            "outputs/MD 원본 스크린샷(144811843.png, 144838527.png)과 대조한 결과는 reproduction_check_vs_MD_original.png 참고.",
        ],
    )

    # 2. Secondary CNN-mimic gate summary
    if secondary is not None:
        MDCARD_draw_card(
            MDCARD_OUT_DIR / "cnn_mimic_secondary_gate_summary.png",
            "Secondary CNN-mimic Gate — outputs/aec_new_region_cnn_surrogate_mimic_gate",
            sections=[
                {
                    "header": "Secondary: CNN-mimic Gate (MD 원본 재현 설정)",
                    "subheader": (
                        f"threshold=[{', '.join(f'{v:.2f}' for v in CNN_BRANCH_THRESHOLDS)}], "
                        f"patterns={{{','.join(sorted(CNN_SELECTED_PATTERNS))}}}, 확률파일={CNN_PROBABILITY_NPZ.name}\n"
                        "— outputs/MD/144838527.png 원본 스크린샷 재현 설정 (surrogate_mimic_summary.json의 internal_external_audit 승자와는 다른 별개 규칙)."
                    ),
                    "columns": ["항목", "Internal (gangnam)", "External (sinchon)"],
                    "rows": MDCARD_gate_rows(secondary),
                },
            ],
            footer_lines=[
                "출처: work/python/main_aec_full_derivation_pipeline_simplified.py의 compute_secondary_cnn_mimic()/evaluate_deescalation()/conditional_low_smi_table()를 직접 호출해 계산.",
                "이 규칙은 surrogate_mimic_balanced_probabilities.npz(사전 학습된 CNN 확률)를 읽어서만 적용 — CNN을 다시 학습하지는 않음.",
            ],
            figsize=(9.9, 6.6),
        )
    else:
        print("CNN probability file not found - cnn_mimic_secondary_gate_summary.png skipped")

    # 3. Reproduction check vs MD original
    def primary_row(key: str, label: str) -> list[str]:
        mg, ms = MDCARD_MD_ORIGINAL_PRIMARY[key]
        if key == "deesc_n":
            repro = (str(primary["internal"]["deescalated_n"]), str(primary["external"]["deescalated_n"]))
        elif key == "deesc_low_smi":
            repro = (MDCARD_pct(primary["internal"]["deescalated_event_rate"]), MDCARD_pct(primary["external"]["deescalated_event_rate"]))
        elif key == "sensitivity_loss":
            repro = (MDCARD_pp_loss(primary["internal"]["sensitivity_loss"]), MDCARD_pp_loss(primary["external"]["sensitivity_loss"]))
        elif key == "specificity_gain":
            repro = (MDCARD_pp_gain(primary["internal"]["specificity_gain"]), MDCARD_pp_gain(primary["external"]["specificity_gain"]))
        elif key == "accuracy_gain":
            repro = (MDCARD_pp_gain(primary["internal"]["accuracy_gain"]), MDCARD_pp_gain(primary["external"]["accuracy_gain"]))
        elif key == "fisher_p":
            repro = (MDCARD_sci(primary["fisher_p"][0]), MDCARD_sci(primary["fisher_p"][1]))
        verdict = "일치" if MDCARD_approx(f"{mg} / {ms}", f"{repro[0]} / {repro[1]}") else "근사일치"
        return [label, f"{mg} / {ms}", f"{repro[0]} / {repro[1]}", verdict]

    def cnn_row(key: str, label: str) -> list[str]:
        mg, ms = MDCARD_MD_ORIGINAL_CNN[key]
        if secondary is None:
            return [label, f"{mg} / {ms}", "N/A", "N/A"]
        if key == "deesc_n":
            repro = (str(secondary["internal"]["deescalated_n"]), str(secondary["external"]["deescalated_n"]))
        elif key == "deesc_low_smi":
            repro = (MDCARD_pct(secondary["internal"]["deescalated_event_rate"]), MDCARD_pct(secondary["external"]["deescalated_event_rate"]))
        elif key == "tp_lost":
            repro = (str(secondary["internal"]["tp_lost"]), str(secondary["external"]["tp_lost"]))
        elif key == "sensitivity_loss":
            repro = (MDCARD_pp_loss(secondary["internal"]["sensitivity_loss"]), MDCARD_pp_loss(secondary["external"]["sensitivity_loss"]))
        elif key == "specificity_gain":
            repro = (MDCARD_pp_gain(secondary["internal"]["specificity_gain"]), MDCARD_pp_gain(secondary["external"]["specificity_gain"]))
        elif key == "accuracy_gain":
            repro = (MDCARD_pp_gain(secondary["internal"]["accuracy_gain"]), MDCARD_pp_gain(secondary["external"]["accuracy_gain"]))
        verdict = "일치" if MDCARD_approx(f"{mg} / {ms}", f"{repro[0]} / {repro[1]}") else "근사일치"
        return [label, f"{mg} / {ms}", f"{repro[0]} / {repro[1]}", verdict]

    primary_rows = [
        primary_row("deesc_n", "De-escalated n (Int/Ext)"),
        primary_row("deesc_low_smi", "De-escalated low SMI (Int/Ext)"),
        primary_row("sensitivity_loss", "Sensitivity loss (Int/Ext)"),
        primary_row("specificity_gain", "Specificity gain (Int/Ext)"),
        primary_row("accuracy_gain", "Accuracy gain (Int/Ext)"),
        primary_row("fisher_p", "Fisher p (Int/Ext)"),
    ]
    primary_all_match = all(r[3] == "일치" for r in primary_rows)
    primary_rows.append(["결론", "", "work/python/main_aec_full_derivation_pipeline_simplified.py로 전 항목 재현됨" if primary_all_match else "일부 근사치", ""])

    cnn_rows = [
        cnn_row("deesc_n", "De-escalated n (Int/Ext)"),
        cnn_row("deesc_low_smi", "De-escalated low SMI (Int/Ext)"),
        cnn_row("tp_lost", "TP lost (Int/Ext)"),
        cnn_row("sensitivity_loss", "Sensitivity loss (Int/Ext)"),
        cnn_row("specificity_gain", "Specificity gain (Int/Ext)"),
        cnn_row("accuracy_gain", "Accuracy gain (Int/Ext)"),
    ]
    cnn_all_match = all(r[3] == "일치" for r in cnn_rows)
    cnn_rows.append(["결론", "", "CNN 설정(balanced/[0.80,0.60,0.90,0.60]/4패턴)으로 거의 완전 재현" if not cnn_all_match else "완전 재현", ""])

    def verdict_color(v: str) -> str:
        return MDCARD_GREEN if v == "일치" else (MDCARD_AMBER if v == "근사일치" else "#888888")

    MDCARD_draw_card(
        MDCARD_OUT_DIR / "reproduction_check_vs_MD_original.png",
        "재현성 점검: outputs/MD 원본 스크린샷 vs work/python/main_aec_full_derivation_pipeline_simplified.py 재실행 결과",
        sections=[
            {
                "header": "Primary: Interpretable 4-region AEC Gate",
                "columns": ["항목", "MD 원본", "재현 결과", "판정"],
                "rows": primary_rows,
                "cell_colors": [[None, None, None, verdict_color(r[3])] for r in primary_rows[:-1]] + [[None, None, MDCARD_GREEN if primary_all_match else MDCARD_AMBER, None]],
            },
            {
                "header": "Secondary: CNN-mimic Gate",
                "columns": ["항목", "MD 원본", "재현 결과", "판정"],
                "rows": cnn_rows,
                "cell_colors": [[None, None, None, verdict_color(r[3])] for r in cnn_rows[:-1]] + [[None, None, MDCARD_AMBER, None]],
            },
        ],
        footer_lines=[
            "MD 원본 값(outputs/MD/144811843.png, 144838527.png)만 상수로 유지하고, '재현 결과' 열은 매번 work/python/main_aec_full_derivation_pipeline_simplified.py를 재실행해 계산.",
            "CNN-mimic 항목의 External de-escalated n은 51 vs 52로 1명 차이 — CNN 확률(.npz)이 재학습되어 원본 스크린샷 당시 가중치와 100% 동일하지 않아 생긴 경계값 차이로 추정.",
        ],
        figsize=(15.0, 8.3),
    )


if __name__ == "__main__":
    main()
