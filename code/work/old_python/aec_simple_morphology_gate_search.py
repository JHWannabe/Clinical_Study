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

from aec_conditional_value import clinical_estimator, clinical_matrix, make_folds, matrix_from_sheet, oof_and_external, zfit_apply  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent / "data_cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_simple_morphology_gate_search"
SEED = 20260701
SIGMA = 1.0
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
QUANTILES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TOP_SINGLE_FOR_COMBO = 24
MAX_COMBO_M = 3
TOP_COMBOS_FOR_ADJUSTED = 250


def row_norm(x: np.ndarray) -> np.ndarray:
    """환자별 행 평균으로 나눠 스캔마다 다른 전체 강도 스케일을 정규화."""
    m = np.nanmean(x, axis=1, keepdims=True)
    m[~np.isfinite(m) | (m == 0)] = 1.0
    return x / m


def load_dataset(path: Path) -> dict:
    """엑셀에서 메타데이터/AEC-128 원자료를 읽고, 가우시안 스무딩과 행정규화를 적용한 뒤 성별 기준 저SMI 라벨(y)까지 계산."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    smooth = ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")
    norm = row_norm(smooth)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "smooth": smooth, "norm": norm, "y": y}


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


def add_stats(rows: dict[str, np.ndarray], x: np.ndarray, prefix: str, lengths: list[int], starts: range, stats_list: tuple[str, ...]) -> None:
    """여러 길이x시작위치 윈도우 조합마다 요청된 통계량(평균/표준편차/최소/최대)만 골라 계산해 rows 딕셔너리에 채워 넣음."""
    n = x.shape[1]
    for length in lengths:
        for start in starts:
            end = start + length - 1
            if start < 1 or end > n:
                continue
            block = x[:, start - 1 : end]
            tag = f"{prefix}_{start:03d}_{end:03d}"
            if "mean" in stats_list:
                rows[f"{tag}_mean"] = np.nanmean(block, axis=1)
            if "sd" in stats_list:
                rows[f"{tag}_sd"] = np.nanstd(block, axis=1)
            if "min" in stats_list:
                rows[f"{tag}_min"] = np.nanmin(block, axis=1)
            if "max" in stats_list:
                rows[f"{tag}_max"] = np.nanmax(block, axis=1)


def build_simple_bank(norm: np.ndarray) -> pd.DataFrame:
    """레벨/기울기/곡률 윈도우 통계, 구간별 '활력도'(vitality) 통계, 초반-중반-후반 조합 트로프 깊이 등 단순하고 해석 가능한 후보 특징들을 계산."""
    rows: dict[str, np.ndarray] = {}
    d1 = np.diff(norm, axis=1)
    d2 = np.diff(d1, axis=1)
    abs_d1 = np.abs(d1)
    abs_d2 = np.abs(d2)

    starts = range(1, 121, 4)
    add_stats(rows, norm, "level", [8, 12, 16, 24, 32], starts, ("mean", "sd"))
    add_stats(rows, d1, "slope", [4, 8, 12, 16, 24], starts, ("mean", "sd", "min", "max"))
    add_stats(rows, abs_d1, "abs_slope", [4, 8, 12, 16, 24], starts, ("mean", "max"))
    add_stats(rows, d2, "curv", [4, 8, 12, 16, 24], starts, ("mean", "sd", "min", "max"))
    add_stats(rows, abs_d2, "abs_curv", [4, 8, 12, 16, 24], starts, ("mean", "max"))

    zones = {
        "early_001_040": (1, 40),
        "mid_045_076": (45, 76),
        "midlate_077_108": (77, 108),
        "late_093_124": (93, 124),
        "tail_105_128": (105, 128),
    }
    for zn, (a, b) in zones.items():
        dd = d1[:, max(a - 1, 0) : min(b - 1, d1.shape[1])]
        cc = d2[:, max(a - 1, 0) : min(b - 2, d2.shape[1])]
        rows[f"vitality_abs_slope_mean_{zn}"] = np.mean(np.abs(dd), axis=1)
        rows[f"vitality_abs_slope_sd_{zn}"] = np.std(np.abs(dd), axis=1)
        rows[f"vitality_abs_curv_mean_{zn}"] = np.mean(np.abs(cc), axis=1)
        rows[f"vitality_abs_curv_sd_{zn}"] = np.std(np.abs(cc), axis=1)
        rows[f"vitality_signed_slope_sd_{zn}"] = np.std(dd, axis=1)
        rows[f"vitality_signed_curv_sd_{zn}"] = np.std(cc, axis=1)

    early = [(1, 32), (9, 40), (17, 48), (25, 56), (33, 64)]
    mids = [(45, 68), (53, 76), (61, 84), (69, 92)]
    lates = [(85, 108), (93, 116), (101, 124), (105, 128)]
    for ea, eb in early:
        e = norm[:, ea - 1 : eb].mean(axis=1)
        for ma, mb in mids:
            m = norm[:, ma - 1 : mb].mean(axis=1)
            for la, lb in lates:
                l = norm[:, la - 1 : lb].mean(axis=1)
                rows[f"shape_trough_depth_e{ea:03d}_{eb:03d}_m{ma:03d}_{mb:03d}_l{la:03d}_{lb:03d}"] = 0.5 * (e + l) - m
                rows[f"shape_late_minus_mid_m{ma:03d}_{mb:03d}_l{la:03d}_{lb:03d}"] = l - m

    rows["vitality_global_abs_slope_mean"] = abs_d1.mean(axis=1)
    rows["vitality_global_abs_curv_mean"] = abs_d2.mean(axis=1)
    rows["vitality_curve_sd"] = norm.std(axis=1)
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return out


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
    return (xg[:, keep] - mu[keep]) / sd[keep], (xs[:, keep] - mu[keep]) / sd[keep], [str(c) for c, k in zip(train.columns, keep) if k]


def clinical_scores(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """임상변수만으로 5-fold OOF/외부 예측 점수를 만들고 z-표준화한 뒤, 각 민감도 목표(OPS)에 대응하는 표준화 임계값을 계산."""
    xg, xs, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"].astype(int), 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xg, g["y"].astype(int), xs, folds)
    c_g, c_s, mu, sd = zfit_apply(clinical_oof, clinical_ext)
    thresholds = {}
    for label, target in OPS:
        thresholds[label] = float((threshold_for_min_sensitivity(g["y"], clinical_oof, target) - mu) / sd)
    return clinical_oof, clinical_ext, c_g, c_s, thresholds


def lowrisk_orient(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """임상점수로 설명 안 되는 라벨 잔차와 특징의 상관 부호를 구해, 특징값이 커질수록 "저위험"(de-escalation 후보) 쪽을 가리키도록 부호를 뒤집은 벡터를 만듦."""
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    resid = y.astype(float) - np.asarray(fit.predict(base), dtype=float)
    association_with_risk = x.T @ resid
    sign = -np.sign(association_with_risk)
    sign[sign == 0] = 1.0
    return sign


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """실제 라벨과 예측(양성 플래그)로부터 혼동행렬 셀 수, 민감도/특이도/정확도/균형정확도를 계산."""
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
        "balanced_accuracy": 0.5 * (sens + spec),
    }


def exact_p(a: int, b: int) -> float:
    """두 방향 변화 건수(a,b)가 동전던지기(50:50)로부터 유의하게 벗어나는지 이항검정으로 계산."""
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def adjusted_p(y: np.ndarray, clinical_z: np.ndarray, cpos: np.ndarray, deesc: np.ndarray, manufacturer: np.ndarray) -> dict:
    """임상양성군 안에서 de-escalation 여부가 제조사 회사 더미변수와 임상점수를 통제한 뒤에도 라벨과 유의하게 연관되는지 로지스틱회귀 우도비검정(LRT)으로 확인."""
    yy = y[cpos].astype(int)
    if len(yy) < 30 or np.unique(yy).size < 2 or int(np.sum(deesc[cpos])) < 5:
        return {"adj_or": np.nan, "adj_wald_p": np.nan, "adj_lrt_p": np.nan}
    company = pd.Series(manufacturer[cpos].astype(str)).map(company_from_manufacturer)
    company = company.where(company.map(company.value_counts()) >= 20, "Other")
    dummies = pd.get_dummies(company, prefix="company", drop_first=True, dtype=float).reset_index(drop=True)
    base = dummies.copy()
    base.insert(0, "clinical_z", clinical_z[cpos])
    full = base.copy()
    full["aec_lowrisk"] = deesc[cpos].astype(float)
    try:
        fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=500)
        fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=500)
        lrt = 2.0 * (fit1.llf - fit0.llf)
        return {
            "adj_or": float(np.exp(fit1.params["aec_lowrisk"])),
            "adj_wald_p": float(fit1.pvalues["aec_lowrisk"]),
            "adj_lrt_p": float(stats.chi2.sf(lrt, 1)),
        }
    except Exception:
        return {"adj_or": np.nan, "adj_wald_p": np.nan, "adj_lrt_p": np.nan}


def metric_row(
    dataset: str,
    op: str,
    y: np.ndarray,
    clinical_z: np.ndarray,
    th: float,
    lowrisk_mask: np.ndarray,
    manufacturer: np.ndarray,
    score_label: str,
    q: float,
    do_adjusted: bool,
) -> dict:
    """단순 저위험 점수 기준 분위수 컷오프(quantile)를 적용한 de-escalation 게이트 전/후의 민감도·특이도·정확도 변화, 유의성 검정 p값, (요청 시) 제조사 조정 p값까지 한 행으로 정리."""
    cpos = clinical_z >= th
    deesc = cpos & lowrisk_mask
    post = cpos & ~deesc
    base = counts(y, cpos)
    new = counts(y, post)
    yy = y.astype(bool)
    sens_loss_n = int(np.sum(yy & cpos & ~post))
    sens_gain_n = int(np.sum(yy & ~cpos & post))
    spec_gain_n = int(np.sum(~yy & cpos & ~post))
    spec_loss_n = int(np.sum(~yy & ~cpos & post))
    correct_base = cpos == yy
    correct_new = post == yy
    acc_gain_n = int(np.sum(~correct_base & correct_new))
    acc_loss_n = int(np.sum(correct_base & ~correct_new))
    kept_e = int(np.sum(y[post] == 1))
    kept_ne = int(np.sum(y[post] == 0))
    de_e = int(np.sum(y[deesc] == 1))
    de_ne = int(np.sum(y[deesc] == 0))
    fisher_p = float(stats.fisher_exact([[kept_e, kept_ne], [de_e, de_ne]])[1]) if kept_e + kept_ne and de_e + de_ne else np.nan
    adj = adjusted_p(y, clinical_z, cpos, deesc, manufacturer) if do_adjusted else {
        "adj_or": np.nan,
        "adj_wald_p": np.nan,
        "adj_lrt_p": np.nan,
    }
    return {
        "dataset": dataset,
        "score_label": score_label,
        "quantile": q,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": new["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - new["sensitivity"],
        "sensitivity_loss_p_exact": exact_p(sens_loss_n, sens_gain_n),
        "clinical_specificity": base["specificity"],
        "post_specificity": new["specificity"],
        "specificity_gain": new["specificity"] - base["specificity"],
        "specificity_gain_p_exact": exact_p(spec_gain_n, spec_loss_n),
        "accuracy_delta": new["accuracy"] - base["accuracy"],
        "accuracy_delta_p_mcnemar": exact_p(acc_gain_n, acc_loss_n),
        "deesc_n": int(np.sum(deesc)),
        "deesc_events": de_e,
        "deesc_event_rate": de_e / (de_e + de_ne) if de_e + de_ne else np.nan,
        "deesc_event_fisher_p": fisher_p,
        **adj,
    }


def summarize(rows: list[dict], dataset: str) -> dict:
    """지정된 데이터셋(내부/외부)에 속한 여러 운영점 결과 중 가장 나쁜 경우(최소 p값, 최대 민감도손실 등)를 뽑아 안전성 제약 판정용 요약통계로 압축."""
    sub = [r for r in rows if r["dataset"] == dataset]
    adj_vals = np.asarray([r["adj_lrt_p"] for r in sub], dtype=float)
    finite_adj = adj_vals[np.isfinite(adj_vals)]
    return {
        f"{dataset}_min_p_loss": float(np.nanmin([r["sensitivity_loss_p_exact"] for r in sub])),
        f"{dataset}_max_sens_loss": float(np.nanmax([r["sensitivity_loss"] for r in sub])),
        f"{dataset}_min_spec_gain": float(np.nanmin([r["specificity_gain"] for r in sub])),
        f"{dataset}_mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in sub])),
        f"{dataset}_max_fisher_p": float(np.nanmax([r["deesc_event_fisher_p"] for r in sub])),
        f"{dataset}_max_adj_lrt_p": float(np.max(finite_adj)) if finite_adj.size else np.nan,
        f"{dataset}_min_deesc_n": int(np.nanmin([r["deesc_n"] for r in sub])),
        f"{dataset}_mean_event_rate": float(np.nanmean([r["deesc_event_rate"] for r in sub])),
    }


def eval_score(
    label: str,
    score_g: np.ndarray,
    score_s: np.ndarray,
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    thresholds: dict[str, float],
    q: float,
    do_adjusted: bool = False,
) -> tuple[list[dict], float]:
    """훈련(g1090) 점수 분포의 q분위수를 컷오프로 삼아 저위험 플래그를 만들고, 두 코호트x모든 운영점에 대한 metric_row 결과 행들과 컷오프값을 반환."""
    cutoff = float(np.quantile(score_g, q))
    low_g = score_g >= cutoff
    low_s = score_s >= cutoff
    rows = []
    for dataset, d, c, low in [("g1090_internal", g, c_g, low_g), ("sdata_external", s, c_s, low_s)]:
        manufacturer = d["meta"]["Manufacturer"].astype(str).to_numpy()
        for op, _ in OPS:
            rows.append(metric_row(dataset, op, d["y"], c, thresholds[op], low, manufacturer, label, q, do_adjusted=do_adjusted))
    return rows, cutoff


def auc_with_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUC를 계산하고(0.5 미만이면 부호를 뒤집어 방향을 통일), Mann-Whitney U 검정으로 유의성 p값도 함께 반환."""
    auc = float(roc_auc_score(y, score))
    oriented = score.copy()
    if auc < 0.5:
        oriented = -oriented
        auc = 1.0 - auc
    p = float(stats.mannwhitneyu(oriented[y == 1], oriented[y == 0], alternative="two-sided").pvalue)
    return auc, p


def clinical_plus_score_auc(g: dict, s: dict, clinical_oof: np.ndarray, clinical_ext: np.ndarray, sg: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """임상점수와 단순 AEC 점수를 함께 넣은 로지스틱 모델을 5-fold OOF로 학습해, 내부 OOF/외부 결합 예측 점수를 반환."""
    folds = make_folds(g["y"].astype(int), 5)
    oof = np.zeros(len(g["y"]), dtype=float)
    all_idx = np.arange(len(g["y"]))
    for fold_id, va in enumerate(folds):
        tr = np.setdiff1d(all_idx, va)
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=SEED + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], sg[tr]]), g["y"][tr])
        oof[va] = model.decision_function(np.column_stack([clinical_oof[va], sg[va]]))
    final = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=SEED + 99)
    final.fit(np.column_stack([clinical_oof, sg]), g["y"])
    ext = final.decision_function(np.column_stack([clinical_ext, ss]))
    return oof, ext


def plot_result(details: pd.DataFrame, path: Path) -> None:
    """운영점별 특이도이득/민감도손실(좌)과 제조사조정 LRT p값의 -log10(우)을 코호트별로 비교하는 2패널 그래프를 그려 PNG로 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), constrained_layout=True)
    labels = [op for op, _ in OPS]
    x = np.arange(len(labels))
    colors = {"g1090_internal": "#2F6B9A", "sdata_external": "#C54E2C"}
    for dataset in ["g1090_internal", "sdata_external"]:
        sub = details[details["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", color=colors[dataset], label=f"{dataset} spec gain")
        axes[0].plot(x, sub["sensitivity_loss"] * 100, marker="x", color=colors[dataset], ls="--", label=f"{dataset} sens loss")
        axes[1].plot(x, -np.log10(np.clip(sub["adj_lrt_p"], 1e-12, 1.0)), marker="o", color=colors[dataset], label=dataset)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Percentage points")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(-np.log10(0.05), color="black", lw=0.8, ls="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("-log10(scanner + clinical adjusted LRT p)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Simple AEC-only morphology gate", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_lock_smoothed_deesc_gate와 같은 de-escalation 아이디어를,
    가우시안 커널 기반 게이트 대신 "단순 분위수 컷오프" 규칙으로 구현해도 여전히 안전하게 작동하는가? —
    더 단순하고 설명하기 쉬운 대안 프로토콜의 g1090 내부 탐색 -> sdata 외부 검증):

    1. g1090/sdata를 로드(가우시안 스무딩 + 환자별 정규화)하고, 임상변수만으로 만든 위험점수와 5개
       민감도 목표(S80~S90)에 대응하는 임계값을 계산.
    2. 레벨/기울기/곡률 윈도우 통계와 초중후반 조합 트로프 특징으로 단순 후보뱅크를 만들고, 훈련 기준
       표준화 후 각 특징의 방향을 "저위험 쪽"으로 정렬(lowrisk_orient).
    3. 단일 특징 x 여러 분위수 컷오프(50~80%) 조합을 g1090 내부에서 전수 평가해 안전성 제약(민감도손실
       유의, 특이도이득 양수 등)을 통과하는 특징 위주로 점수를 매겨 스크리닝 CSV로 저장.
    4. 상위 24개 특징으로 조합 풀을 만들고, 1~3개 조합의 평균 점수 x 분위수 컷오프를 다시 전수 탐색
       (combo_search 단계)해 내부 생존 규칙들을 CSV로 저장.
    5. 생존 규칙 상위 250개에 대해서만 제조사(회사) 더미변수를 통제한 조정 p값(LRT)까지 계산하고, 내부와
       외부 양쪽에서 제약을 통과하는 규칙 하나를 최종 "잠금(locked)"으로 선택.
    6. 잠긴 규칙을 g1090/sdata 양쪽 운영점에 적용한 상세 결과와, 임상단독/단순AEC단독/결합 모델의 AUC
       비교표를 계산.
    7. 결과를 2패널 그래프로 시각화하고, 스크리닝/조합탐색/조정된 요약/잠긴규칙/AUC를 모두 CSV로, 선택
       프로토콜 요약을 JSON으로 저장한 뒤 콘솔에 전체 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_simple_bank(g["norm"])
    fs = build_simple_bank(s["norm"])
    xg, xs, names = standardize_train_test(fg, fs)
    sign = lowrisk_orient(g["y"], c_g, xg)
    xg_low = xg * sign[None, :]
    xs_low = xs * sign[None, :]

    # Single-score internal screen.
    candidates = []
    detail_cache: dict[tuple[str, float], list[dict]] = {}
    cutoff_cache: dict[tuple[str, float], float] = {}
    for j, name in enumerate(names):
        sg, ss = xg_low[:, j], xs_low[:, j]
        for q in QUANTILES:
            rows, cutoff = eval_score(name, sg, ss, g, s, c_g, c_s, thresholds, q, do_adjusted=False)
            si = summarize(rows, "g1090_internal")
            survives_internal = (
                si["g1090_internal_min_p_loss"] >= 0.05
                and si["g1090_internal_min_spec_gain"] > 0
                and si["g1090_internal_max_fisher_p"] < 0.05
                and si["g1090_internal_min_deesc_n"] >= 25
                and si["g1090_internal_max_sens_loss"] <= 0.08
            )
            score = (
                2.5 * si["g1090_internal_min_spec_gain"]
                + si["g1090_internal_mean_spec_gain"]
                - 0.6 * si["g1090_internal_max_sens_loss"]
            )
            if not survives_internal:
                score -= 10.0
            candidates.append({"score_label": name, "feature_indices": str(j), "quantile": q, "cutoff": cutoff, "internal_survives": survives_internal, "internal_score": score, **si})
            detail_cache[(name, q)] = rows
            cutoff_cache[(name, q)] = cutoff
    single = pd.DataFrame(candidates).sort_values(["internal_survives", "internal_score"], ascending=False)
    single.to_csv(OUT_DIR / "single_simple_gate_internal_screen.csv", index=False)

    pool = single.drop_duplicates("score_label").head(TOP_SINGLE_FOR_COMBO).copy()
    pool.to_csv(OUT_DIR / "combo_pool_from_internal_screen.csv", index=False)
    pool_indices = [int(v.split("|")[0]) for v in pool["feature_indices"]]

    combo_rows = []
    combo_details = {}
    for m in range(1, min(MAX_COMBO_M, len(pool_indices)) + 1):
        for subset in itertools.combinations(pool_indices, m):
            label = " + ".join(names[i] for i in subset)
            sg = xg_low[:, list(subset)].mean(axis=1)
            ss = xs_low[:, list(subset)].mean(axis=1)
            for q in QUANTILES:
                rows, cutoff = eval_score(label, sg, ss, g, s, c_g, c_s, thresholds, q, do_adjusted=False)
                si = summarize(rows, "g1090_internal")
                survives_internal = (
                    si["g1090_internal_min_p_loss"] >= 0.05
                    and si["g1090_internal_min_spec_gain"] > 0
                    and si["g1090_internal_max_fisher_p"] < 0.05
                    and si["g1090_internal_min_deesc_n"] >= 25
                    and si["g1090_internal_max_sens_loss"] <= 0.08
                )
                score = (
                    3.0 * si["g1090_internal_min_spec_gain"]
                    + 1.2 * si["g1090_internal_mean_spec_gain"]
                    - 0.7 * si["g1090_internal_max_sens_loss"]
                    + 0.015 * m
                )
                if not survives_internal:
                    score -= 10.0
                combo_rows.append(
                    {
                        "score_label": label,
                        "feature_indices": "|".join(map(str, subset)),
                        "m": m,
                        "quantile": q,
                        "cutoff": cutoff,
                        "internal_survives": survives_internal,
                        "internal_score": score,
                        **si,
                    }
                )
                combo_details[(label, q)] = rows
    combo = pd.DataFrame(combo_rows).sort_values(["internal_survives", "internal_score"], ascending=False)
    combo.to_csv(OUT_DIR / "combo_simple_gate_internal_search.csv", index=False)

    adj_candidates = combo[combo["internal_survives"]].head(TOP_COMBOS_FOR_ADJUSTED)
    if adj_candidates.empty:
        adj_candidates = combo.head(TOP_COMBOS_FOR_ADJUSTED)
    adjusted_summary_rows = []
    adjusted_detail_rows = []
    for _, cand in adj_candidates.iterrows():
        cand_label = str(cand["score_label"])
        cand_q = float(cand["quantile"])
        cand_subset = [int(v) for v in str(cand["feature_indices"]).split("|")]
        cand_sg = xg_low[:, cand_subset].mean(axis=1)
        cand_ss = xs_low[:, cand_subset].mean(axis=1)
        cand_rows, cand_cutoff = eval_score(cand_label, cand_sg, cand_ss, g, s, c_g, c_s, thresholds, cand_q, do_adjusted=True)
        si_adj = summarize(cand_rows, "g1090_internal")
        se_adj = summarize(cand_rows, "sdata_external")
        internal_adjusted_survives = (
            si_adj["g1090_internal_min_p_loss"] >= 0.05
            and si_adj["g1090_internal_min_spec_gain"] > 0
            and si_adj["g1090_internal_max_fisher_p"] < 0.05
            and si_adj["g1090_internal_max_adj_lrt_p"] < 0.05
            and si_adj["g1090_internal_min_deesc_n"] >= 25
            and si_adj["g1090_internal_max_sens_loss"] <= 0.08
        )
        external_adjusted_survives = (
            se_adj["sdata_external_min_p_loss"] >= 0.05
            and se_adj["sdata_external_min_spec_gain"] > 0
            and se_adj["sdata_external_max_fisher_p"] < 0.05
            and se_adj["sdata_external_max_adj_lrt_p"] < 0.05
            and se_adj["sdata_external_min_deesc_n"] >= 20
            and se_adj["sdata_external_max_sens_loss"] <= 0.08
        )
        adj_score = float(cand["internal_score"]) - 0.04 * float(np.nan_to_num(si_adj["g1090_internal_max_adj_lrt_p"], nan=1.0))
        adjusted_summary_rows.append(
            {
                **cand.to_dict(),
                "cutoff_adjusted_eval": cand_cutoff,
                "internal_adjusted_survives": internal_adjusted_survives,
                "external_adjusted_survives": external_adjusted_survives,
                "adjusted_selection_score": adj_score,
                **{f"adj_{k}": v for k, v in si_adj.items()},
                **{f"adj_{k}": v for k, v in se_adj.items()},
            }
        )
        if internal_adjusted_survives or external_adjusted_survives:
            for row in cand_rows:
                rr = dict(row)
                rr["candidate_label"] = cand_label
                rr["candidate_quantile"] = cand_q
                rr["candidate_feature_indices"] = str(cand["feature_indices"])
                adjusted_detail_rows.append(rr)
    adjusted_summary = pd.DataFrame(adjusted_summary_rows).sort_values(
        ["internal_adjusted_survives", "adjusted_selection_score"], ascending=False
    )
    adjusted_summary.to_csv(OUT_DIR / "adjusted_top_combo_summary.csv", index=False)
    pd.DataFrame(adjusted_detail_rows).to_csv(OUT_DIR / "adjusted_top_combo_survivor_details.csv", index=False)

    locked = adjusted_summary[adjusted_summary["internal_adjusted_survives"]].head(1)
    if locked.empty:
        locked = adjusted_summary.head(1)
    locked_row = locked.iloc[0]
    label = str(locked_row["score_label"])
    q = float(locked_row["quantile"])
    subset = [int(v) for v in str(locked_row["feature_indices"]).split("|")]
    sg = xg_low[:, subset].mean(axis=1)
    ss = xs_low[:, subset].mean(axis=1)
    rows, cutoff = eval_score(label, sg, ss, g, s, c_g, c_s, thresholds, q, do_adjusted=True)
    details = pd.DataFrame(rows)
    details.to_csv(OUT_DIR / "locked_simple_gate_operating_point_details.csv", index=False)
    summary = {**locked_row.to_dict(), **summarize(rows, "sdata_external")}

    combo_oof, combo_ext = clinical_plus_score_auc(g, s, clinical_oof, clinical_ext, -sg, -ss)
    auc_rows = []
    for model, ig, es in [
        ("clinical_only", clinical_oof, clinical_ext),
        ("simple_aec_lowrisk_score_only", -sg, -ss),
        ("clinical_plus_simple_aec_score", combo_oof, combo_ext),
    ]:
        ai, pi = auc_with_p(g["y"], ig)
        ae, pe = auc_with_p(s["y"], es)
        auc_rows.append({"model": model, "internal_auc": ai, "internal_auc_p": pi, "external_auc": ae, "external_auc_p": pe})
    auc_df = pd.DataFrame(auc_rows)
    auc_df["internal_delta_vs_clinical"] = auc_df["internal_auc"] - auc_df.loc[0, "internal_auc"]
    auc_df["external_delta_vs_clinical"] = auc_df["external_auc"] - auc_df.loc[0, "external_auc"]
    auc_df.to_csv(OUT_DIR / "locked_simple_gate_auc_summary.csv", index=False)

    feature_rows = pd.DataFrame({"feature_index": subset, "feature": [names[i] for i in subset], "lowrisk_sign": sign[subset]})
    feature_rows.to_csv(OUT_DIR / "locked_simple_gate_features.csv", index=False)
    (OUT_DIR / "locked_simple_gate_summary.json").write_text(
        json.dumps(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "gate": "clinical_positive AND AEC_lowrisk_score >= training_cutoff",
                "clinical_score_not_used_inside_aec_lowrisk_score": True,
                "selection_dataset": "g1090 internal only",
                "locked": summary,
                "features": feature_rows.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_result(details, OUT_DIR / "locked_simple_gate_plot.png")

    print("Locked simple gate")
    print(pd.Series(summary).to_string())
    print("\nFeatures")
    print(feature_rows.to_string(index=False))
    print("\nAUC")
    print(auc_df.to_string(index=False))
    show = [
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
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
        "adj_or",
        "adj_lrt_p",
    ]
    print("\nDetails")
    print(details[show].to_string(index=False))
    print("\nout_dir", OUT_DIR)


if __name__ == "__main__":
    # 단순 분위수 컷오프 기반 de-escalation 게이트를 g1090 내부에서 탐색/잠금하고 sdata 외부에서
    # 검증하는, 더 단순하고 설명하기 쉬운 대안 파이프라인을 실행한다.
    main()
