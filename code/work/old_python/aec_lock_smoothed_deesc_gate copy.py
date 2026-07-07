from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_mass_feature_combinations import build_feature_bank  # noqa: E402
from aec_conditional_value import clinical_estimator, clinical_matrix, make_folds, matrix_from_sheet, oof_and_external, zfit_apply  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent.parent / "data_cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_lock_smoothed_deesc_gate"
SEED = 20260701
SIGMA = 1.0
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55, 0.70]
TOP_FEATURES_FOR_COMBO = 18
MAX_COMBO_M = 4
MAX_FEATURES_SCREEN = 600


def sanitize(name: str) -> str:
    """파일명으로 안전하게 쓸 수 있도록 특수문자를 밑줄로 치환하고 150자로 자름."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:150]


def row_norm(x: np.ndarray) -> np.ndarray:
    """환자별 행 평균으로 나눠 스캔마다 다른 전체 강도 스케일을 정규화."""
    m = np.nanmean(x, axis=1, keepdims=True)
    m[~np.isfinite(m) | (m == 0)] = 1.0
    return x / m


def load_dataset(path: Path) -> dict:
    """엑셀에서 메타데이터/AEC-128 원자료를 읽고, 가우시안 스무딩과 행정규화를 적용한 뒤 성별 기준 저SMI 라벨(y)까지 계산."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    smooth_raw = ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")
    norm = row_norm(smooth_raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "smooth_raw": smooth_raw, "norm": norm, "y": y, "sex": sex, "smi": smi}


def add_window_stats(out: dict[str, np.ndarray], x: np.ndarray, prefix: str, lengths: list[int], step: int) -> None:
    """여러 길이의 슬라이딩 윈도우마다 평균/표준편차/최소/최대를 계산해 out 딕셔너리에 채워 넣음."""
    n = x.shape[1]
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            end0 = start0 + length
            block = x[:, start0:end0]
            tag = f"{prefix}_{start0 + 1:03d}_{end0:03d}"
            out[f"{tag}_mean"] = np.nanmean(block, axis=1)
            out[f"{tag}_sd"] = np.nanstd(block, axis=1)
            out[f"{tag}_min"] = np.nanmin(block, axis=1)
            out[f"{tag}_max"] = np.nanmax(block, axis=1)


def build_visual_norm_bank(norm: np.ndarray) -> pd.DataFrame:
    """초반(지지)/중반(복부 저점)/후반(회복) 구간 조합으로 만든 트로프 깊이·평탄도 등 '시각적으로 해석 가능한' 형태 특징과 기울기/곡률 윈도우 통계를 계산."""
    rows: dict[str, np.ndarray] = {}
    # Keep these coarse and semantic: early support, mid abdomen trough, late recovery.
    mids = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(37, 82, 4) for length in [16, 24, 32] if s + length - 1 <= 104]
    tails = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(81, 118, 4) for length in [12, 20, 28] if s + length - 1 <= 128]
    early = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(1, 50, 4) for length in [16, 24, 32] if s + length - 1 <= 68]
    for es, ee, ename in early[::2]:
        e = norm[:, es - 1 : ee].mean(axis=1)
        for ms, me, mname in mids[::2]:
            m = norm[:, ms - 1 : me].mean(axis=1)
            for ts, te, tname in tails[::2]:
                t = norm[:, ts - 1 : te].mean(axis=1)
                rows[f"visual_trough_depth__early_{ename}__mid_{mname}__tail_{tname}"] = 0.5 * (e + t) - m
                rows[f"visual_mid_flatness__early_{ename}__mid_{mname}__tail_{tname}"] = np.abs(m - 0.5 * (e + t))
                rows[f"visual_tail_minus_mid__tail_{tname}__mid_{mname}"] = t - m
    d1 = np.diff(norm, axis=1)
    d2 = np.diff(d1, axis=1)
    add_window_stats(rows, d1, "visual_norm_slope", [6, 10, 14, 18, 24, 32], 3)
    add_window_stats(rows, np.abs(d1), "visual_norm_abs_slope", [6, 10, 14, 18, 24, 32], 3)
    add_window_stats(rows, d2, "visual_norm_curv", [6, 10, 14, 18, 24], 3)
    rows["visual_global_waviness_abs_slope_mean"] = np.abs(d1).mean(axis=1)
    rows["visual_global_waviness_abs_curv_mean"] = np.abs(d2).mean(axis=1)
    rows["visual_global_waviness_curve_sd"] = norm.std(axis=1)
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return df


def build_candidate_bank(norm: np.ndarray) -> pd.DataFrame:
    """공용 대규모 특징뱅크(build_feature_bank)와 시각적 형태 특징뱅크를 이어 붙여 이 프로토콜의 전체 후보 특징 테이블을 구성."""
    dense = build_feature_bank(norm).add_prefix("smooth_norm__")
    visual = build_visual_norm_bank(norm).add_prefix("smooth_visual__")
    # No raw-level features in this lock protocol.
    return pd.concat([dense, visual], axis=1)


def standardize_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """훈련셋 기준 결측치 대치·1~99% 분위수 클리핑·z-표준화를 계산해 훈련/테스트 양쪽에 동일하게 적용하고, 표준편차가 0에 가까운 특징은 제거."""
    xg = train.to_numpy(dtype=float)
    xs = test.to_numpy(dtype=float)
    med = np.nanmedian(xg, axis=0)
    med[~np.isfinite(med)] = 0.0
    xg = np.where(np.isfinite(xg), xg, med)
    xs = np.where(np.isfinite(xs), xs, med)
    lo = np.nanquantile(xg, 0.01, axis=0)
    hi = np.nanquantile(xg, 0.99, axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    xg[:, ok] = np.clip(xg[:, ok], lo[ok], hi[ok])
    xs[:, ok] = np.clip(xs[:, ok], lo[ok], hi[ok])
    mu = xg.mean(axis=0)
    sd = xg.std(axis=0)
    keep = np.isfinite(sd) & (sd > 1e-10)
    xg = (xg[:, keep] - mu[keep]) / sd[keep]
    xs = (xs[:, keep] - mu[keep]) / sd[keep]
    names = [str(c) for c, k in zip(train.columns, keep) if k]
    return xg, xs, names


def clinical_scores(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """임상변수만으로 5-fold OOF/외부 예측 점수를 만들고 z-표준화한 뒤, 각 민감도 목표(OPS)에 대응하는 표준화 임계값을 계산."""
    xg, xs, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"].astype(int), 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xg, g["y"].astype(int), xs, folds)
    c_g, c_s, mu, sd = zfit_apply(clinical_oof, clinical_ext)
    thresholds = {}
    for label, target in OPS:
        th_raw = threshold_for_min_sensitivity(g["y"], clinical_oof, target)
        thresholds[label] = float((th_raw - mu) / sd)
    return clinical_oof, clinical_ext, c_g, c_s, thresholds


def auc_with_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUC를 계산하고(0.5 미만이면 부호를 뒤집어 방향을 통일), Mann-Whitney U 검정으로 유의성 p값도 함께 반환."""
    auc = float(roc_auc_score(y, score))
    oriented = score.copy()
    if auc < 0.5:
        oriented = -oriented
        auc = 1.0 - auc
    p = float(stats.mannwhitneyu(oriented[y == 1], oriented[y == 0], alternative="two-sided").pvalue)
    return auc, p


def risk_direction(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """임상점수로 설명되지 않는 라벨 잔차와 각 특징의 상관 부호를 구해, 특징값이 커질수록 위험이 커지는 방향으로 정렬하는 +-1 부호 벡터를 만듦."""
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    resid = y.astype(float) - np.asarray(fit.predict(base), dtype=float)
    score = x.T @ resid
    direction = np.sign(score)
    fallback = np.sign(x.T @ (y.astype(float) - y.mean()))
    direction[direction == 0] = fallback[direction == 0]
    direction[direction == 0] = 1.0
    return direction.astype(float)


def prescreen_feature_indices(
    y: np.ndarray,
    clinical_z: np.ndarray,
    x: np.ndarray,
    names: list[str],
    thresholds: dict[str, float],
    max_n: int,
) -> np.ndarray:
    """전체 특징 후보 중, 임상점수로 설명 안 되는 잔차와의 상관(전역 점수)과 각 임상양성 구간에서의 상관(구간별 점수), 형태 관련 키워드 가산점을 합쳐 상위 max_n개 특징의 인덱스만 남김."""
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    resid = y.astype(float) - np.asarray(fit.predict(base), dtype=float)
    denom = np.sqrt(np.sum(x * x, axis=0) + 1e-12)
    global_score = np.abs(x.T @ resid) / denom
    cp_score = np.zeros(x.shape[1], dtype=float)
    for op, _ in OPS:
        cp = clinical_z >= thresholds[op]
        yy = y[cp].astype(float)
        if yy.size < 20 or np.unique(yy).size < 2:
            continue
        yc = yy - yy.mean()
        xx = x[cp]
        cp_score += np.abs(xx.T @ yc) / np.sqrt(np.sum(xx * xx, axis=0) + 1e-12)
    name_arr = np.asarray(names)
    semantic = np.array(
        [
            0.08
            if any(token in name for token in ["curv", "slope", "haar", "trough", "waviness", "dct", "autocorr"])
            else 0.0
            for name in name_arr
        ],
        dtype=float,
    )
    score = global_score + 0.7 * cp_score + semantic
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    return order[: min(max_n, len(order))]


def company_from_manufacturer(value: object) -> str:
    """DICOM 제조사 문자열을 Siemens/Philips/GE/Other 4개 회사군으로 매핑."""
    s = str(value).upper()
    if any(token in s for token in ["SOMATOM", "SENSATION", "SIEMENS"]):
        return "Siemens"
    if any(token in s for token in ["INGENUITY", "ICT", "PHILIPS"]):
        return "Philips"
    if any(token in s for token in ["REVOLUTION", "LIGHTSPEED", "GE"]):
        return "GE"
    return "Other"


def company_eta2(values: np.ndarray, company: np.ndarray) -> float:
    """값의 총분산 중 제조사 그룹 간 분산이 차지하는 비율(에타제곱)을 계산 — 특징이 제조사 흔적을 얼마나 담고 있는지 측정."""
    ok = np.isfinite(values) & (company != "Other")
    values = values[ok]
    company = company[ok]
    if len(values) < 10:
        return np.nan
    grand = values.mean()
    total = np.sum((values - grand) ** 2)
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for c in np.unique(company):
        v = values[company == c]
        between += len(v) * (v.mean() - grand) ** 2
    return float(between / total)


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """실제 라벨과 예측(양성 플래그)로부터 혼동행렬 셀 수, 민감도/특이도/정확도/균형정확도/PPV/NPV를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def exact_p(a: int, b: int) -> float:
    """두 방향 변화 건수(a,b)가 동전던지기(50:50)로부터 유의하게 벗어나는지 이항검정으로 계산."""
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def deesc_metric_row(dataset: str, rule: str, features: str, op: str, y: np.ndarray, cpos: np.ndarray, deesc: np.ndarray) -> dict:
    """임상양성군에서 de-escalation(위험강등) 규칙을 적용하기 전/후의 민감도·특이도·정확도 변화와, 그 변화가 우연이 아닌지 검증하는 이항/피셔 정확검정 p값들을 한 행으로 정리."""
    final = cpos & ~deesc
    base = counts(y, cpos)
    post = counts(y, final)
    yy = y.astype(bool)
    sens_loss_n = int(np.sum(yy & cpos & ~final))
    sens_gain_n = int(np.sum(yy & ~cpos & final))
    spec_gain_n = int(np.sum(~yy & cpos & ~final))
    spec_loss_n = int(np.sum(~yy & ~cpos & final))
    correct_base = cpos == yy
    correct_post = final == yy
    acc_gain_n = int(np.sum(~correct_base & correct_post))
    acc_loss_n = int(np.sum(correct_base & ~correct_post))
    kept_e = int(np.sum(y[final] == 1))
    kept_ne = int(np.sum(y[final] == 0))
    de_e = int(np.sum(y[deesc] == 1))
    de_ne = int(np.sum(y[deesc] == 0))
    fisher_p = float(stats.fisher_exact([[kept_e, kept_ne], [de_e, de_ne]])[1]) if (kept_e + kept_ne and de_e + de_ne) else np.nan
    return {
        "dataset": dataset,
        "rule": rule,
        "features": features,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "accuracy_delta": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "balanced_accuracy_delta": post["balanced_accuracy"] - base["balanced_accuracy"],
        "clinical_positive_n": int(np.sum(cpos)),
        "clinical_positive_events": int(np.sum(y[cpos] == 1)),
        "clinical_positive_event_rate": float(np.mean(y[cpos])) if np.any(cpos) else np.nan,
        "deesc_n": int(np.sum(deesc)),
        "deesc_events": de_e,
        "deesc_event_rate": de_e / (de_e + de_ne) if de_e + de_ne else np.nan,
        "fp_removed": de_ne,
        "tp_lost": de_e,
        "sensitivity_loss_p_exact": exact_p(sens_loss_n, sens_gain_n),
        "specificity_gain_p_exact": exact_p(spec_gain_n, spec_loss_n),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain_n, acc_loss_n),
        "deesc_event_fisher_p": fisher_p,
    }


def make_single_deesc(
    clinical_z: np.ndarray,
    feature_z: np.ndarray,
    th: float,
    width: float,
    lam: float,
) -> np.ndarray:
    """단일 특징 하나로 만드는 de-escalation 게이트: 임상양성(cpos) 중 임계값 근방에서 가우시안 가중치(boundary)를 특징값에 곱해 게이트 점수를 만들고, 그 점수가 다시 임계값 아래로 내려가면 강등."""
    cpos = clinical_z >= th
    boundary = np.exp(-0.5 * ((clinical_z - th) / width) ** 2)
    gate_score = clinical_z + lam * boundary * feature_z
    return cpos & (gate_score < th)


def summarize_internal(rows: list[dict], op_labels: set[str]) -> dict:
    """여러 운영점(민감도 목표)에 걸친 결과 행들 중 가장 나쁜 경우(최소 p값, 최대 민감도손실 등)를 뽑아 안전성 제약 판정에 쓸 요약통계로 압축."""
    sub = [r for r in rows if r["operating_point"] in op_labels]
    if not sub:
        return {}
    return {
        "min_p_loss": float(np.nanmin([r["sensitivity_loss_p_exact"] for r in sub])),
        "max_sens_loss": float(np.nanmax([r["sensitivity_loss"] for r in sub])),
        "min_spec_gain": float(np.nanmin([r["specificity_gain"] for r in sub])),
        "mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in sub])),
        "min_ba_delta": float(np.nanmin([r["balanced_accuracy_delta"] for r in sub])),
        "mean_ba_delta": float(np.nanmean([r["balanced_accuracy_delta"] for r in sub])),
        "max_fisher_p": float(np.nanmax([r["deesc_event_fisher_p"] for r in sub])),
        "min_deesc_n": int(np.nanmin([r["deesc_n"] for r in sub])),
        "mean_deesc_event_rate": float(np.nanmean([r["deesc_event_rate"] for r in sub])),
    }


def feature_screen(
    y: np.ndarray,
    clinical_z: np.ndarray,
    x: np.ndarray,
    names: list[str],
    thresholds: dict[str, float],
    company: np.ndarray,
) -> pd.DataFrame:
    """각 특징 x (폭, 람다) 조합마다 단일-특징 de-escalation 게이트를 5개 운영점에서 평가하고, 안전성 제약 통과 여부와 게이트 품질 점수를 매겨 전수 스크리닝 결과표를 만듦."""
    rows = []
    for j, name in enumerate(names):
        z = x[:, j]
        eta = company_eta2(z, company)
        for width in WIDTHS:
            for lam in LAMBDAS:
                metrics = []
                for op, _ in OPS:
                    th = thresholds[op]
                    cpos = clinical_z >= th
                    deesc = make_single_deesc(clinical_z, z, th, width, lam)
                    metrics.append(deesc_metric_row("g1090_internal", "single", name, op, y, cpos, deesc))
                s = summarize_internal(metrics, {op for op, _ in OPS})
                fail = (
                    s["min_p_loss"] < 0.05
                    or s["min_spec_gain"] <= 0
                    or s["max_fisher_p"] >= 0.05
                    or s["min_deesc_n"] < 25
                    or s["max_sens_loss"] > 0.08
                )
                score = (
                    2.5 * s["min_spec_gain"]
                    + 1.0 * s["mean_spec_gain"]
                    + 0.8 * s["min_ba_delta"]
                    - 0.45 * s["max_sens_loss"]
                    - 0.05 * np.nan_to_num(eta, nan=0.0)
                )
                if fail:
                    score -= 10.0
                rows.append({"feature": name, "feature_index": j, "width": width, "lambda": lam, "company_eta2": eta, "screen_score": score, **s})
    out = pd.DataFrame(rows).sort_values("screen_score", ascending=False)
    return out


def feature_family(name: str) -> str:
    """특징 이름에 포함된 키워드를 보고 형태-대비/곡률/기울기/haar/스펙트럼/레벨 등 특징군으로 분류."""
    if "trough" in name or "tail_minus_mid" in name or "flatness" in name:
        return "shape_contrast"
    if "curv" in name:
        return "curvature"
    if "abs_slope" in name:
        return "absolute_slope"
    if "slope" in name:
        return "signed_slope"
    if "haar" in name:
        return "haar"
    if "dct" in name or "fft" in name or "autocorr" in name:
        return "spectral"
    if "level" in name or "ratio" in name:
        return "level"
    return "other"


def diverse_combo_pool(screen: pd.DataFrame, x: np.ndarray, n: int) -> pd.DataFrame:
    """스크리닝 점수 상위 특징들 중, 특징군당 최대 5개·상관계수 0.92 미만이 되도록 다양성을 유지하며 조합 탐색용 상위 n개 후보 풀을 뽑음."""
    rows = []
    used_features: set[str] = set()
    family_counts: dict[str, int] = {}
    selected_cols: list[np.ndarray] = []
    for _, row in screen.sort_values("screen_score", ascending=False).iterrows():
        feature = str(row["feature"])
        if feature in used_features:
            continue
        fam = feature_family(feature)
        if family_counts.get(fam, 0) >= 5:
            continue
        col = x[:, int(row["feature_index"])]
        if selected_cols:
            corrs = []
            for prev in selected_cols:
                if np.std(prev) <= 1e-12 or np.std(col) <= 1e-12:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(prev, col)[0, 1])
                corrs.append(abs(corr))
            if max(corrs) >= 0.92:
                continue
        item = row.copy()
        item["feature_family"] = fam
        rows.append(item)
        used_features.add(feature)
        family_counts[fam] = family_counts.get(fam, 0) + 1
        selected_cols.append(col)
        if len(rows) >= n:
            break
    if len(rows) < n:
        for _, row in screen.sort_values("screen_score", ascending=False).iterrows():
            feature = str(row["feature"])
            if feature in used_features:
                continue
            item = row.copy()
            item["feature_family"] = feature_family(feature)
            rows.append(item)
            used_features.add(feature)
            if len(rows) >= n:
                break
    return pd.DataFrame(rows).reset_index(drop=True)


def precompute_votes(
    selected: pd.DataFrame,
    y_by: dict[str, np.ndarray],
    c_by: dict[str, np.ndarray],
    x_by: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    """선택된 후보 특징 각각에 대해 (코호트, 운영점)별로 개별 de-escalation 투표(0/1) 행렬과 임상양성 플래그를 미리 계산해, 이후 조합 탐색에서 반복 계산을 피함."""
    votes: dict[tuple[str, str], np.ndarray] = {}
    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    for dataset in y_by:
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c_by[dataset] >= th
            cpos_by[(dataset, op)] = cpos
            mat = np.zeros((len(selected), len(y_by[dataset])), dtype=np.int8)
            for i, r in selected.reset_index(drop=True).iterrows():
                z = x_by[dataset][:, int(r["feature_index"])]
                mat[i] = make_single_deesc(c_by[dataset], z, th, float(r["width"]), float(r["lambda"])).astype(np.int8)
            votes[(dataset, op)] = mat
    return votes, cpos_by


def evaluate_rule(
    selected: pd.DataFrame,
    subset: tuple[int, ...],
    k: int,
    votes: dict[tuple[str, str], np.ndarray],
    cpos_by: dict[tuple[str, str], np.ndarray],
    y_by: dict[str, np.ndarray],
    datasets: list[str],
) -> list[dict]:
    """선택된 특징 부분집합(subset)에 대해 'k표 이상이면 강등' 투표 규칙을 적용하고, 지정된 데이터셋들과 모든 운영점에 대해 de-escalation 성능 지표 행들을 계산."""
    features = " + ".join(selected.iloc[list(subset)]["feature"].astype(str).tolist())
    rule = f"{k}-of-{len(subset)}"
    rows = []
    for dataset in datasets:
        for op, _ in OPS:
            deesc = cpos_by[(dataset, op)] & (votes[(dataset, op)][list(subset)].sum(axis=0) >= k)
            rows.append(deesc_metric_row(dataset, rule, features, op, y_by[dataset], cpos_by[(dataset, op)], deesc))
    return rows


def combo_search(selected: pd.DataFrame, votes: dict, cpos_by: dict, y_by: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1~MAX_COMBO_M개 특징으로 이루어진 모든 부분집합 x 다수결 임계값(k) 조합을 g1090 내부 데이터에서만 평가해, 안전성 제약을 통과하는지와 선택 점수를 매긴 전체 요약표(및 생존 규칙의 상세 행)를 만듦 — 이후 '잠금(lock)'될 최종 규칙을 여기서 고른다."""
    summary_rows = []
    detail_rows = []
    n = len(selected)
    for m in range(1, min(MAX_COMBO_M, n) + 1):
        k_values = [1] if m == 1 else [k for k in range((m + 1) // 2, m + 1)]
        for subset in itertools.combinations(range(n), m):
            for k in k_values:
                rows = evaluate_rule(selected, subset, k, votes, cpos_by, y_by, ["g1090_internal"])
                s = summarize_internal(rows, {op for op, _ in OPS})
                survives = (
                    s["min_p_loss"] >= 0.05
                    and s["min_spec_gain"] > 0
                    and s["max_fisher_p"] < 0.05
                    and s["min_deesc_n"] >= 25
                    and s["max_sens_loss"] <= 0.08
                )
                mean_eta = float(np.nanmean(selected.iloc[list(subset)]["company_eta2"]))
                score = (
                    3.0 * s["min_spec_gain"]
                    + 1.3 * s["mean_spec_gain"]
                    + 0.8 * s["min_ba_delta"]
                    - 0.6 * s["max_sens_loss"]
                    - 0.04 * mean_eta
                    + 0.01 * min(m, 3)
                )
                if not survives:
                    score -= 10.0
                summary = {
                    "rule": f"{k}-of-{m}",
                    "m": m,
                    "k": k,
                    "subset_indices": "|".join(map(str, subset)),
                    "features": " + ".join(selected.iloc[list(subset)]["feature"].astype(str).tolist()),
                    "mean_company_eta2": mean_eta,
                    "survives_internal_constraints": survives,
                    "lock_selection_score": score,
                    **{f"internal_{kk}": vv for kk, vv in s.items()},
                }
                summary_rows.append(summary)
                if survives and score > 0:
                    for r in rows:
                        rr = dict(r)
                        rr["subset_indices"] = summary["subset_indices"]
                        rr["lock_selection_score"] = score
                        detail_rows.append(rr)
    summary_df = pd.DataFrame(summary_rows).sort_values(["survives_internal_constraints", "lock_selection_score"], ascending=False)
    detail_df = pd.DataFrame(detail_rows)
    return summary_df, detail_df


def adjusted_deesc_logit(y: np.ndarray, clinical_z: np.ndarray, deesc: np.ndarray, manufacturer: np.ndarray) -> dict:
    """사용되지 않는 자리표시 함수 — 실제 로직은 adjusted_p_for_row에서 cpos를 별도로 받아 수행하므로, 직접 호출되면 항상 오류를 던짐."""
    cpos = deesc | (clinical_z >= np.nanmin(clinical_z[deesc]) if np.any(deesc) else np.zeros_like(deesc, dtype=bool))
    # The caller passes deesc inside clinical-positive rows, so rebuild cpos outside this helper.
    raise RuntimeError("adjusted_deesc_logit should not be called directly")


def adjusted_p_for_row(y: np.ndarray, clinical_z: np.ndarray, cpos: np.ndarray, deesc: np.ndarray, manufacturer: np.ndarray) -> dict:
    """임상양성군 안에서 de-escalation 여부가 제조사 회사 더미변수(및 임상점수)를 통제한 뒤에도 라벨과 유의하게 연관되는지 로지스틱회귀 우도비검정(LRT)으로 확인 — 게이트 효과가 순수 회사효과로 설명되지 않는지 감사."""
    yy = y[cpos].astype(int)
    if len(yy) < 20 or np.unique(yy).size < 2 or np.sum(deesc[cpos]) == 0:
        return {
            "scanner_only_or": np.nan,
            "scanner_only_lrt_p": np.nan,
            "scanner_plus_clinical_or": np.nan,
            "scanner_plus_clinical_lrt_p": np.nan,
        }
    m = pd.Series(manufacturer[cpos].astype(str)).map(company_from_manufacturer)
    m = m.where(m.map(m.value_counts()) >= 20, "Other")
    dummies = pd.get_dummies(m, prefix="company", drop_first=True, dtype=float).reset_index(drop=True)
    out = {}
    for include_clinical, label in [(False, "scanner_only"), (True, "scanner_plus_clinical")]:
        base = dummies.copy()
        full = dummies.copy()
        if include_clinical:
            base.insert(0, "clinical_z", clinical_z[cpos])
            full.insert(0, "clinical_z", clinical_z[cpos])
        full["deesc"] = deesc[cpos].astype(float)
        try:
            fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=1000)
            fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=1000)
            lrt = 2.0 * (fit1.llf - fit0.llf)
            out[f"{label}_or"] = float(np.exp(fit1.params["deesc"]))
            out[f"{label}_wald_p"] = float(fit1.pvalues["deesc"])
            out[f"{label}_lrt_p"] = float(stats.chi2.sf(lrt, 1))
        except Exception:
            out[f"{label}_or"] = np.nan
            out[f"{label}_wald_p"] = np.nan
            out[f"{label}_lrt_p"] = np.nan
    return out


def locked_score_auc_table(
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    selected: pd.DataFrame,
    locked_summary: pd.Series,
    xg_risk: np.ndarray,
    xs_risk: np.ndarray,
) -> pd.DataFrame:
    """잠긴(locked) 특징 조합의 평균을 'AEC 점수'로 삼아, 임상단독/AEC단독/임상+AEC 결합 로지스틱 모델의 내부(OOF)·외부 AUC와 임상단독 대비 AUC 증분을 비교표로 만듦."""
    subset = [int(v) for v in str(locked_summary["subset_indices"]).split("|")]
    feature_indices = selected.iloc[subset]["feature_index"].astype(int).to_numpy()
    aec_g = xg_risk[:, feature_indices].mean(axis=1)
    aec_s = xs_risk[:, feature_indices].mean(axis=1)
    if roc_auc_score(g["y"], aec_g) < 0.5:
        aec_g = -aec_g
        aec_s = -aec_s
    folds = make_folds(g["y"].astype(int), 5)
    combo_oof = np.zeros(len(g["y"]), dtype=float)
    all_idx = np.arange(len(g["y"]))
    for fold_id, va in enumerate(folds):
        tr = np.setdiff1d(all_idx, va)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], aec_g[tr]]), g["y"][tr])
        combo_oof[va] = model.decision_function(np.column_stack([clinical_oof[va], aec_g[va]]))
    final = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED + 99)
    final.fit(np.column_stack([clinical_oof, aec_g]), g["y"])
    combo_ext = final.decision_function(np.column_stack([clinical_ext, aec_s]))
    rows = []
    for model_name, sg, ss in [
        ("clinical_only", clinical_oof, clinical_ext),
        ("locked_aec_score_only", aec_g, aec_s),
        ("clinical_plus_locked_aec_score", combo_oof, combo_ext),
    ]:
        auc_g, p_g = auc_with_p(g["y"], sg)
        auc_s, p_s = auc_with_p(s["y"], ss)
        rows.append(
            {
                "model": model_name,
                "internal_auc": auc_g,
                "internal_auc_p": p_g,
                "external_auc": auc_s,
                "external_auc_p": p_s,
            }
        )
    base_g = rows[0]["internal_auc"]
    base_s = rows[0]["external_auc"]
    for row in rows:
        row["internal_delta_vs_clinical_auc"] = row["internal_auc"] - base_g
        row["external_delta_vs_clinical_auc"] = row["external_auc"] - base_s
    return pd.DataFrame(rows)


def plot_locked(details: pd.DataFrame, path: Path) -> None:
    """내부/외부 코호트 각각에 대해 (임상 vs 게이트적용 후 특이도, 특이도 이득 vs 민감도 손실) 4패널 그래프를 그려 잠긴 게이트의 운영점별 효과를 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), constrained_layout=True)
    colors = {"g1090_internal": "#2F6B9A", "sdata_external": "#C54E2C"}
    x = np.arange(len(OPS))
    labels = [op for op, _ in OPS]
    for dataset, ax in zip(["g1090_internal", "sdata_external"], axes[0]):
        sub = details[details["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        ax.plot(x, sub["clinical_specificity"] * 100, marker="o", color="#999999", label="Clinical specificity")
        ax.plot(x, sub["post_specificity"] * 100, marker="o", color=colors[dataset], label="Post-gate specificity")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Specificity (%)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for dataset, ax in zip(["g1090_internal", "sdata_external"], axes[1]):
        sub = details[details["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        ax.bar(x - 0.18, sub["specificity_gain"] * 100, width=0.34, color=colors[dataset], label="Specificity gain")
        ax.bar(x + 0.18, sub["sensitivity_loss"] * 100, width=0.34, color="#D95F02", label="Sensitivity loss")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Percentage points")
        ax.axhline(0, color="black", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Locked smoothed patient-normalized AEC de-escalation gate", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상 기준으로 "양성"(고위험) 판정된 환자 중 일부를, AEC-128
    곡선의 형태 특징을 근거로 안전하게 "위험 강등(de-escalation)"할 수 있는가? — g1090 내부 데이터에서만
    규칙을 탐색/고정(lock)하고, sdata 외부 데이터에서는 검증만 하는 사전등록형 프로토콜):

    1. g1090/sdata를 로드(가우시안 스무딩 + 환자별 정규화)하고, 임상변수만으로 만든 위험점수와 5개
       민감도 목표(S80~S90)에 대응하는 임계값을 계산.
    2. 공용 특징뱅크 + 이 프로토콜 전용 시각적 형태 특징뱅크로 대규모 후보 특징을 만들고, 훈련(g1090)
       기준으로 표준화한 뒤, 임상 잔차와의 상관 기준으로 최대 600개까지 1차 스크리닝.
    3. 각 특징의 위험 방향을 정렬(risk_direction)하고, 단일-특징 de-escalation 게이트를 폭/람다 조합별로
       전수 평가(feature_screen)해 안전성 제약(민감도손실 유의, 특이도이득 양수, 최소 강등 인원 등)을
       통과하는 특징을 우선 선별.
    4. diverse_combo_pool로 특징군 다양성을 유지하며 상위 18개 후보를 뽑고, 이들의 de-escalation 투표를
       미리 계산(precompute_votes)한 뒤, 1~4개 특징 조합 x 다수결 임계값(k) 전수 탐색(combo_search)을
       g1090 내부에서만 수행 — 안전성 제약을 통과하면서 점수가 가장 높은 규칙을 "잠금(locked)".
    5. 잠긴 규칙을 g1090(내부)과 sdata(외부) 양쪽에 적용해 운영점별 민감도/특이도/정확도 변화를 계산하고,
       제조사 더미변수를 통제해도 강등 효과가 유의한지(adjusted_p_for_row) 확인.
    6. 잠긴 특징들의 평균을 AEC 점수로 삼아 임상단독/AEC단독/결합 모델의 내부·외부 AUC를 비교.
    7. 운영점별 효과를 4패널 그래프로 시각화하고, 스크리닝/조합탐색/잠긴규칙/AUC/조정된 p값을 모두 CSV로,
       선택 프로토콜 요약을 JSON으로 저장한 뒤 콘솔에 전체 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)

    fg = build_candidate_bank(g["norm"])
    fs = build_candidate_bank(s["norm"])
    xg_all, xs_all, names_all = standardize_train_test(fg, fs)
    prescreen_idx = prescreen_feature_indices(g["y"], c_g, xg_all, names_all, thresholds, MAX_FEATURES_SCREEN)
    pd.DataFrame(
        {
            "rank": np.arange(1, len(prescreen_idx) + 1),
            "original_feature_index": prescreen_idx,
            "feature": [names_all[i] for i in prescreen_idx],
        }
    ).to_csv(OUT_DIR / "internal_prescreen_feature_pool.csv", index=False)
    xg = xg_all[:, prescreen_idx]
    xs = xs_all[:, prescreen_idx]
    names = [names_all[i] for i in prescreen_idx]
    direction = risk_direction(g["y"], c_g, xg)
    xg_risk = xg * direction[None, :]
    xs_risk = xs * direction[None, :]

    company_g = g["meta"]["Manufacturer"].map(company_from_manufacturer).to_numpy()
    screen = feature_screen(g["y"], c_g, xg_risk, names, thresholds, company_g)
    screen.to_csv(OUT_DIR / "internal_single_feature_screen.csv", index=False)
    selected = diverse_combo_pool(screen, xg_risk, TOP_FEATURES_FOR_COMBO)
    selected.to_csv(OUT_DIR / "internal_combo_feature_pool.csv", index=False)

    y_by = {"g1090_internal": g["y"].astype(int), "sdata_external": s["y"].astype(int)}
    c_by = {"g1090_internal": c_g, "sdata_external": c_s}
    x_by = {"g1090_internal": xg_risk, "sdata_external": xs_risk}
    votes, cpos_by = precompute_votes(selected, y_by, c_by, x_by, thresholds)
    combo_summary, combo_internal_details = combo_search(selected, votes, cpos_by, y_by)
    combo_summary.to_csv(OUT_DIR / "internal_combo_search_summary.csv", index=False)
    combo_internal_details.to_csv(OUT_DIR / "internal_combo_search_survivor_details.csv", index=False)

    locked = combo_summary[combo_summary["survives_internal_constraints"]].head(1)
    if locked.empty:
        locked = combo_summary.head(1)
    locked_row = locked.iloc[0]
    subset = tuple(int(v) for v in str(locked_row["subset_indices"]).split("|"))
    k = int(locked_row["k"])
    locked_details = pd.DataFrame(
        evaluate_rule(selected, subset, k, votes, cpos_by, y_by, ["g1090_internal", "sdata_external"])
    )
    locked_details.to_csv(OUT_DIR / "locked_gate_operating_point_details.csv", index=False)

    adj_rows = []
    for _, r in locked_details.iterrows():
        dataset = str(r["dataset"])
        op = str(r["operating_point"])
        th = thresholds[op]
        cpos = c_by[dataset] >= th
        votes_sum = votes[(dataset, op)][list(subset)].sum(axis=0)
        deesc = cpos & (votes_sum >= k)
        meta = g["meta"] if dataset == "g1090_internal" else s["meta"]
        adj = adjusted_p_for_row(y_by[dataset], c_by[dataset], cpos, deesc, meta["Manufacturer"].astype(str).to_numpy())
        adj_rows.append({"dataset": dataset, "operating_point": op, **adj})
    adjusted_df = pd.DataFrame(adj_rows)
    adjusted_df.to_csv(OUT_DIR / "locked_gate_adjusted_pvalues.csv", index=False)

    auc_df = locked_score_auc_table(g, s, clinical_oof, clinical_ext, selected, locked_row, xg_risk, xs_risk)
    auc_df.to_csv(OUT_DIR / "locked_gate_auc_summary.csv", index=False)

    plot_locked(locked_details, OUT_DIR / "locked_gate_operating_points.png")

    feature_rows = selected.iloc[list(subset)].copy()
    feature_rows.to_csv(OUT_DIR / "locked_gate_features.csv", index=False)
    summary = {
        "preprocessing": {
            "source_sheet": "aec_128",
            "smoothing": f"gaussian_filter1d sigma={SIGMA}, axis=1, mode=nearest",
            "normalization": "patient-wise mean normalization after smoothing",
            "raw_level_features_used": False,
        },
        "selection": {
            "derivation_dataset": "g1090 internal only",
            "external_dataset": "sdata used only after lock",
            "operating_points": OPS,
            "single_feature_pool_size": int(len(selected)),
            "max_combo_m": MAX_COMBO_M,
            "constraints": "min sensitivity-loss p >= 0.05, min specificity gain > 0, max de-escalated event Fisher p < 0.05, min de-escalated n >= 25, max sensitivity loss <= 8 percentage points",
        },
        "locked_rule": locked_row.to_dict(),
        "locked_features": feature_rows[["feature", "width", "lambda", "company_eta2", "screen_score"]].to_dict(orient="records"),
    }
    (OUT_DIR / "locked_gate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("AUC summary")
    print(auc_df.to_string(index=False))
    print("\nLocked rule")
    print(locked_row.to_string())
    print("\nLocked features")
    print(feature_rows[["feature", "width", "lambda", "company_eta2", "screen_score"]].to_string(index=False))
    print("\nOperating points")
    show_cols = [
        "dataset",
        "operating_point",
        "clinical_sensitivity",
        "post_sensitivity",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "clinical_specificity",
        "post_specificity",
        "specificity_gain",
        "specificity_gain_p_exact",
        "accuracy_delta",
        "accuracy_delta_p_mcnemar",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
    ]
    print(locked_details[show_cols].to_string(index=False))
    print("\nAdjusted p-values")
    print(adjusted_df.to_string(index=False))
    print("\nout_dir", OUT_DIR)


if __name__ == "__main__":
    # g1090 내부 데이터에서만 de-escalation 게이트 규칙을 탐색해 잠그고(lock), sdata 외부
    # 데이터에서는 검증만 수행하는 사전등록형 파이프라인을 실행한다.
    main()
