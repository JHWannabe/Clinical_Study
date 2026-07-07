from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_mass_feature_combinations import build_feature_bank  # noqa: E402
from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    clinical_estimator,
    clinical_matrix,
    make_folds,
    matrix_from_sheet,
    oof_and_external,
    row_norm,
    threshold_youden,
    zfit_apply,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_midrange_feature_refit"
SEED = 20260630
OPS = [("youden", None), ("sens80", 0.80), ("sens85", 0.85), ("sens90", 0.90), ("sens95", 0.95)]
PRIMARY_OPS = {"youden", "sens80", "sens85"}


def load_aec128(path: Path) -> dict:
    """엑셀에서 aec_128 원시행렬·행정규화 곡선과 저근감소증 라벨을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "norm": norm, "y": y}


def add_window_stats(out: dict[str, np.ndarray], x: np.ndarray, prefix: str, lengths: list[int], step: int) -> None:
    """여러 길이(lengths)의 겹치는 슬라이딩 윈도우 평균·표준편차를 out 딕셔너리에 채워 넣음 (in-place)."""
    n = x.shape[1]
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            end0 = start0 + length
            tag = f"{prefix}_{start0 + 1:03d}_{end0:03d}"
            block = x[:, start0:end0]
            out[f"{tag}_mean"] = np.nanmean(block, axis=1)
            out[f"{tag}_sd"] = np.nanstd(block, axis=1)


def build_visual_contrast_bank(norm: np.ndarray, raw: np.ndarray) -> pd.DataFrame:
    """중간/후반/초반 구간을 성긴 간격으로 슬라이딩하며 만든 tail-minus-mid, tail-over-mid,
    mid-minus-early, trough-depth 등 "시각적 대비" 특징들과, 기울기/원시레벨 특징까지 모두 계산."""
    rows: dict[str, np.ndarray] = {}

    norm_windows: dict[str, np.ndarray] = {}
    for length in [8, 12, 16, 20, 24, 32]:
        for start in range(1, 128 - length + 2, 4):
            end = start + length - 1
            if end > 128:
                continue
            name = f"{start:03d}_{end:03d}"
            norm_windows[name] = norm[:, start - 1 : end].mean(axis=1)

    # Coarse windows keep the search interpretable and avoid turning the feature
    # definition itself into a high-dimensional model.
    mids = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(37, 82, 8) for length in [16, 24, 32] if s + length - 1 <= 100]
    tails = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(85, 118, 8) for length in [12, 20, 28] if s + length - 1 <= 128]
    early = [(s, s + length - 1, f"{s:03d}_{s + length - 1:03d}") for s in range(1, 50, 8) for length in [16, 24, 32] if s + length - 1 <= 64]

    for ms, me, mname in mids:
        m = norm[:, ms - 1 : me].mean(axis=1)
        for ts, te, tname in tails:
            t = norm[:, ts - 1 : te].mean(axis=1)
            rows[f"visual_tail_minus_mid__tail_{tname}__mid_{mname}"] = t - m
            rows[f"visual_tail_over_mid__tail_{tname}__mid_{mname}"] = t / np.where(np.abs(m) < 1e-8, 1e-8, m)
        for es, ee, ename in early:
            e = norm[:, es - 1 : ee].mean(axis=1)
            rows[f"visual_mid_minus_early__mid_{mname}__early_{ename}"] = m - e

    trough_early = early[::3]
    trough_tail = tails[::2]
    for es, ee, ename in trough_early:
        e = norm[:, es - 1 : ee].mean(axis=1)
        for ms, me, mname in mids:
            m = norm[:, ms - 1 : me].mean(axis=1)
            for ts, te, tname in trough_tail:
                t = norm[:, ts - 1 : te].mean(axis=1)
                rows[f"visual_trough_depth__early_{ename}__mid_{mname}__tail_{tname}"] = 0.5 * (e + t) - m

    d1 = np.diff(norm, axis=1)
    add_window_stats(rows, d1, "visual_norm_slope", [6, 10, 14, 18, 24], step=4)
    add_window_stats(rows, np.abs(d1), "visual_norm_abs_slope", [6, 10, 14, 18, 24], step=4)

    raw_rows: dict[str, np.ndarray] = {}
    raw_log = np.log(np.clip(raw, 1e-6, None))
    add_window_stats(raw_rows, raw, "raw_level", [8, 16, 24, 32, 48], step=8)
    add_window_stats(raw_rows, raw_log, "raw_log_level", [8, 16, 24, 32, 48], step=8)
    raw_rows["raw_global_mean"] = raw.mean(axis=1)
    raw_rows["raw_global_sd"] = raw.std(axis=1)
    raw_rows["raw_global_range"] = raw.max(axis=1) - raw.min(axis=1)
    for name, val in raw_rows.items():
        rows[name] = val

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_candidate_bank(d: dict) -> pd.DataFrame:
    """aec128_mass_feature_combinations의 초대형 특징 은행과, 이 스크립트의 시각적 대비 은행을 합쳐 전체 후보 특징 테이블을 만듦."""
    norm_bank = build_feature_bank(d["norm"]).add_prefix("bank_norm__")
    visual_bank = build_visual_contrast_bank(d["norm"], d["raw"]).add_prefix("midrange__")
    return pd.concat([norm_bank, visual_bank], axis=1)


def standardize_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """train 중앙값으로 결측 대체, 1~99% 분위수로 이상치 클리핑, train 평균/표준편차로 표준화하고, 분산이 0에 가까운 특징은 제거."""
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


def clinical_scores(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, dict[str, dict]]:
    """임상 단독 모델의 표준화된 OOF/외부 점수와, Youden/민감도80~95% 5개 운영점의 임상 임계값(z-스케일)을 함께 계산."""
    xg, xs, _ = clinical_matrix(g["meta"], s["meta"])
    folds = make_folds(g["y"], 5)
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xg, g["y"], xs, folds)
    c_g, c_s, mu, sd = zfit_apply(clinical_oof, clinical_ext)
    thresholds: dict[str, dict] = {}
    for label, target in OPS:
        if target is None:
            th_raw = threshold_youden(g["y"], clinical_oof)
            method = "youden"
        else:
            th_raw = threshold_for_min_sensitivity(g["y"], clinical_oof, target)
            method = "minimum_training_sensitivity"
        thresholds[label] = {
            "target_sensitivity": target,
            "method": method,
            "clinical_z": float((th_raw - mu) / sd),
        }
    return c_g, c_s, thresholds


def risk_direction(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """임상점수로 설명 안 되는 잔차와 각 특징의 상관 부호를 구해, 모든 특징을 "높을수록 고위험"이 되도록 방향(부호)을 통일."""
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    resid = y.astype(float) - np.asarray(fit.predict(base), dtype=float)
    score = x.T @ resid
    direction = np.sign(score)
    fallback = np.sign(x.T @ (y.astype(float) - y.mean()))
    direction[direction == 0] = fallback[direction == 0]
    direction[direction == 0] = 1.0
    return direction.astype(float)


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측(pred)과 실제 라벨로 TP/FP/FN/TN과 민감도·특이도·PPV·NPV를 계산."""
    yb = y.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.sum(yb & pred))
    fp = int(np.sum(~yb & pred))
    fn = int(np.sum(yb & ~pred))
    tn = int(np.sum(~yb & ~pred))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def gate_metrics(y: np.ndarray, clinical_z: np.ndarray, aec_risk_z: np.ndarray, th: float, width: float, lam: float) -> dict:
    """가우시안 경계 게이트로 임상양성군을 유지/하향조정한 뒤, 민감도손실/특이도이득/균형이득
    (balanced_gain)과 Fisher p값까지 모두 계산."""
    boundary = np.exp(-0.5 * ((clinical_z - th) / width) ** 2)
    gate = clinical_z + lam * boundary * aec_risk_z
    clinical_pos = clinical_z >= th
    final_pos = clinical_pos & (gate >= th)
    deesc = clinical_pos & ~final_pos
    keep = final_pos
    base = counts(y, clinical_pos)
    rule = counts(y, final_pos)
    a = int(np.sum(y[keep] == 1))
    b = int(np.sum(y[keep] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    fisher_p = stats.fisher_exact([[a, b], [c, d]])[1] if (a + b) and (c + d) else np.nan
    return {
        **{f"clinical_{k}": v for k, v in base.items()},
        **{f"rule_{k}": v for k, v in rule.items()},
        "clinical_positive_n": int(clinical_pos.sum()),
        "clinical_positive_events": int(y[clinical_pos].sum()),
        "clinical_positive_prevalence": float(np.mean(y[clinical_pos])) if clinical_pos.any() else np.nan,
        "deesc_n": int(deesc.sum()),
        "deesc_events": c,
        "deesc_prevalence": c / (c + d) if c + d else np.nan,
        "fp_removed": d,
        "tp_lost": c,
        "specificity_gain": rule["specificity"] - base["specificity"],
        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
        "balanced_gain": (rule["specificity"] - base["specificity"]) - (base["sensitivity"] - rule["sensitivity"]),
        "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
    }


def summarize_train_metrics(metrics: list[dict]) -> dict:
    """youden/sens80/sens85 3개 주요 운영점의 지표를 하나의 train 선택 점수로 압축 — 하향조정 인원
    20명 미만/특이도이득 0.5%p 미만/최대 민감도손실 7.5% 초과 중 하나라도 있으면 크게 감점."""
    primary = [m for m in metrics if m["operating_point"] in PRIMARY_OPS]
    spec_gain = np.asarray([m["specificity_gain"] for m in primary], dtype=float)
    sens_loss = np.asarray([m["sensitivity_loss"] for m in primary], dtype=float)
    balanced = np.asarray([m["balanced_gain"] for m in primary], dtype=float)
    deesc_n = np.asarray([m["deesc_n"] for m in primary], dtype=float)
    pvals = np.asarray([m["fisher_p"] for m in primary], dtype=float)
    pscore = np.nanmean(-np.log10(np.clip(pvals, 1e-12, 1.0)))
    fail = False
    if np.any(deesc_n < 20):
        fail = True
    if np.any(spec_gain < 0.005):
        fail = True
    if np.max(sens_loss) > 0.075:
        fail = True
    score = float(np.nanmean(balanced) + 0.35 * np.nanmin(balanced) + 0.012 * pscore)
    if fail:
        score -= 10.0
    return {
        "train_selection_score": score,
        "train_primary_avg_spec_gain": float(np.nanmean(spec_gain)),
        "train_primary_avg_sens_loss": float(np.nanmean(sens_loss)),
        "train_primary_avg_balanced_gain": float(np.nanmean(balanced)),
        "train_primary_min_balanced_gain": float(np.nanmin(balanced)),
        "train_primary_min_deesc_n": float(np.nanmin(deesc_n)),
        "train_primary_mean_neglog10_fisher_p": float(pscore),
        "train_constraint_fail": fail,
    }


def adjusted_deesc_p(
    y: np.ndarray,
    clinical_z: np.ndarray,
    aec_risk_z: np.ndarray,
    manufacturer: np.ndarray,
    thresholds: dict[str, dict],
    width: float,
    lam: float,
    include_clinical: bool,
) -> pd.DataFrame:
    """제조사(스캐너) 더미변수를 통제한 로지스틱 회귀로, "하향조정 여부"가 스캐너 차이로 설명되는
    가짜 신호가 아닌지 LRT와 Wald 검정으로 확인 — include_clinical로 임상점수 통제 여부 선택."""
    rows = []
    for label, cfg in thresholds.items():
        th = cfg["clinical_z"]
        boundary = np.exp(-0.5 * ((clinical_z - th) / width) ** 2)
        gate = clinical_z + lam * boundary * aec_risk_z
        clinical_pos = clinical_z >= th
        deesc = clinical_pos & (gate < th)
        yy = y[clinical_pos].astype(int)
        if np.unique(yy).size < 2 or deesc[clinical_pos].sum() == 0:
            continue
        base = pd.DataFrame()
        full = pd.DataFrame({"deesc": deesc[clinical_pos].astype(float)})
        if include_clinical:
            base["clinical_z"] = clinical_z[clinical_pos]
            full.insert(0, "clinical_z", clinical_z[clinical_pos])
        m = pd.Series(manufacturer[clinical_pos].astype(str))
        m = m.where(m.map(m.value_counts()) >= 20, "OTHER")
        dummies = pd.get_dummies(m, prefix="scanner", drop_first=True, dtype=float)
        base = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        full = pd.concat([full.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        try:
            fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=1000)
            fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=1000)
            lrt = 2 * (fit1.llf - fit0.llf)
            rows.append(
                {
                    "operating_point": label,
                    "adjustment": "scanner_plus_clinical" if include_clinical else "scanner_only",
                    "n_clinical_positive": int(clinical_pos.sum()),
                    "deesc_or": float(np.exp(fit1.params["deesc"])),
                    "deesc_coef": float(fit1.params["deesc"]),
                    "deesc_wald_p": float(fit1.pvalues["deesc"]),
                    "deesc_lrt_p": float(stats.chi2.sf(lrt, 1)),
                    "scanner_groups": int(m.nunique()),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "operating_point": label,
                    "adjustment": "scanner_plus_clinical" if include_clinical else "scanner_only",
                    "n_clinical_positive": int(clinical_pos.sum()),
                    "deesc_or": np.nan,
                    "deesc_coef": np.nan,
                    "deesc_wald_p": np.nan,
                    "deesc_lrt_p": np.nan,
                    "scanner_groups": int(m.nunique()),
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metrics(
    y: np.ndarray,
    clinical_z: np.ndarray,
    aec_risk_z: np.ndarray,
    thresholds: dict[str, dict],
    width: float,
    lam: float,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """운영점별 게이트의 특이도이득/민감도손실/균형이득/하향조정군 유병률을 부트스트랩 재표본추출로 신뢰구간(및 부호 방향 p값)과 함께 추정."""
    rng = np.random.default_rng(SEED + 41)
    rows = []
    for label, cfg in thresholds.items():
        vals = []
        th = cfg["clinical_z"]
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if np.unique(yy).size < 2:
                continue
            m = gate_metrics(yy, clinical_z[idx], aec_risk_z[idx], th, width, lam)
            vals.append([m["specificity_gain"], m["sensitivity_loss"], m["balanced_gain"], m["deesc_prevalence"]])
        arr = np.asarray(vals)
        for j, metric in enumerate(["specificity_gain", "sensitivity_loss", "balanced_gain", "deesc_prevalence"]):
            x = arr[:, j]
            rows.append(
                {
                    "operating_point": label,
                    "metric": metric,
                    "mean": float(np.mean(x)),
                    "ci2.5": float(np.quantile(x, 0.025)),
                    "ci97.5": float(np.quantile(x, 0.975)),
                    "p_le_0": float(np.mean(x <= 0)),
                    "p_ge_0": float(np.mean(x >= 0)),
                }
            )
    return pd.DataFrame(rows)


def plot_selected(result: pd.DataFrame, path: Path, title: str) -> None:
    """선택된 특징의 운영점별 (임상 vs 결합모델 특이도) 막대그래프와, (특이도이득 vs 민감도손실) 막대그래프를 나란히 그려 PNG로 저장."""
    ext = result[result["dataset"].eq("sdata_external")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6), sharex=True)
    x = np.arange(len(ext))
    labels = ext["operating_point"].tolist()
    width = 0.36
    axes[0].bar(x - width / 2, ext["clinical_specificity"] * 100, width=width, color="#8DA0CB", label="Clinical")
    axes[0].bar(x + width / 2, ext["rule_specificity"] * 100, width=width, color="#66A61E", label="Clinical + AEC gate")
    axes[0].set_ylabel("Specificity (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, ext["specificity_gain"] * 100, width=width, color="#66A61E", label="Specificity gain")
    axes[1].bar(x + width / 2, ext["sensitivity_loss"] * 100, width=width, color="#D95F02", label="Sensitivity loss")
    axes[1].axhline(0, color="#555555", lw=1)
    axes[1].set_ylabel("Percentage points")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: "중간대역(midrange)" 시각적 대비 특징과 초대형 특징은행을
    합친 거대한 후보군에서, train 안에서만 최적의 특징+게이트폭+람다 조합을 골라내면 외부에서도
    통하는가? — 이 파일은 이후 스크립트들(reference, curves, top3000 등)이 재사용하는 핵심 결과를 만든다):

    1. g1090/sdata를 로드하고 임상 단독 모델의 표준화 점수와 5개 운영점 임계값을 준비.
    2. build_candidate_bank로 초대형 특징은행 + 시각적 대비은행을 합친 거대한 후보 특징 테이블을
       만들고, standardize_train_test로 정리한 뒤 risk_direction으로 모든 특징의 방향을 통일.
    3. 모든 특징 x 게이트폭 3종 x 람다 3종 조합(수만 개)에 대해 gate_metrics로 5개 운영점 성능을
       계산하고, summarize_train_metrics로 "youden/sens80/sens85 평균 균형이득"이라는 단일
       train 선택 점수로 압축해 전체를 순위 매긴다 (외부 데이터는 전혀 관여하지 않음).
    4. train 선택 점수 상위 80개를 뽑아, 그 각각을 g1090 OOF와 sdata 외부 양쪽 5개 운영점에서 평가.
    5. 외부 데이터에서 "youden/sens80/sens85 평균" 기준으로 상위 80개를 다시 정렬한 요약을 만들고,
       train 1위(=외부 요약에서도 1위인 것)를 "최종 채택 특징"으로 고정.
    6. 채택된 특징에 대해 adjusted_deesc_p로 스캐너(제조사) 통제 하에서도 하향조정이 유의한지
       확인하고, bootstrap_metrics로 외부 신뢰구간을 추정하며, 운영점별 성능 그래프를 저장.
    7. 선택 규칙, 채택된 특징·폭·람다, 외부 성능 요약을 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, thresholds = clinical_scores(g, s)

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]

    widths = [0.35, 0.50, 0.70]
    lambdas = [0.25, 0.40, 0.55]
    train_rows = []
    for j, name in enumerate(names):
        zg = xg[:, j]
        for width in widths:
            for lam in lambdas:
                metrics = []
                for label, cfg in thresholds.items():
                    m = gate_metrics(g["y"], c_g, zg, cfg["clinical_z"], width, lam)
                    metrics.append({"operating_point": label, **m})
                row = {"feature": name, "width": width, "lambda": lam}
                row.update(summarize_train_metrics(metrics))
                for m in metrics:
                    if m["operating_point"] in PRIMARY_OPS:
                        op = m["operating_point"]
                        row[f"{op}_spec_gain"] = m["specificity_gain"]
                        row[f"{op}_sens_loss"] = m["sensitivity_loss"]
                        row[f"{op}_balanced_gain"] = m["balanced_gain"]
                        row[f"{op}_fisher_p"] = m["fisher_p"]
                        row[f"{op}_deesc_n"] = m["deesc_n"]
                        row[f"{op}_deesc_prevalence"] = m["deesc_prevalence"]
                train_rows.append(row)

    train_ranked = pd.DataFrame(train_rows).sort_values("train_selection_score", ascending=False)
    train_ranked.to_csv(OUT_DIR / "midrange_feature_search_train_ranked.csv", index=False)
    top = train_ranked.head(80).copy()
    top.to_csv(OUT_DIR / "midrange_feature_search_train_top80.csv", index=False)

    eval_rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        feature_name = str(row["feature"])
        param_width = float(row["width"])
        param_lambda = float(row["lambda"])
        j = names.index(feature_name)
        for dataset, d, c, z in [("g1090_oof", g, c_g, xg[:, j]), ("sdata_external", s, c_s, xs[:, j])]:
            for label, cfg in thresholds.items():
                m = gate_metrics(d["y"], c, z, cfg["clinical_z"], param_width, param_lambda)
                eval_rows.append(
                    {
                        "train_rank": rank,
                        "dataset": dataset,
                        "feature": feature_name,
                        "width": param_width,
                        "lambda": param_lambda,
                        "operating_point": label,
                        "target_sensitivity": cfg["target_sensitivity"],
                        **m,
                    }
                )
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT_DIR / "midrange_train_selected_top80_external_eval.csv", index=False)

    primary_ext = eval_df[(eval_df["dataset"].eq("sdata_external")) & (eval_df["operating_point"].isin(PRIMARY_OPS))]
    ext_summary = (
        primary_ext.groupby(["train_rank", "feature", "width", "lambda"], as_index=False)
        .agg(
            external_primary_avg_spec_gain=("specificity_gain", "mean"),
            external_primary_avg_sens_loss=("sensitivity_loss", "mean"),
            external_primary_avg_balanced_gain=("balanced_gain", "mean"),
            external_primary_min_balanced_gain=("balanced_gain", "min"),
            external_primary_min_deesc_n=("deesc_n", "min"),
            external_primary_mean_deesc_prevalence=("deesc_prevalence", "mean"),
            external_primary_max_fisher_p=("fisher_p", "max"),
        )
        .sort_values(["external_primary_avg_balanced_gain", "external_primary_avg_spec_gain"], ascending=False)
    )
    ext_summary.to_csv(OUT_DIR / "midrange_train_selected_external_primary_summary.csv", index=False)

    chosen_row = ext_summary[ext_summary["train_rank"].eq(1)]
    if chosen_row.empty:
        chosen_row = ext_summary.head(1)
    chosen = chosen_row.iloc[0]
    chosen_eval = eval_df[eval_df["train_rank"].eq(int(chosen["train_rank"]))].copy()
    chosen_eval.to_csv(OUT_DIR / "chosen_train_feature_all_operating_points.csv", index=False)

    feature = str(chosen["feature"])
    width = float(chosen["width"])
    lam = float(chosen["lambda"])
    j = names.index(feature)
    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(s["y"]))).astype(str).to_numpy()
    adj = pd.concat(
        [
            adjusted_deesc_p(s["y"], c_s, xs[:, j], scanner_s, thresholds, width, lam, include_clinical=False),
            adjusted_deesc_p(s["y"], c_s, xs[:, j], scanner_s, thresholds, width, lam, include_clinical=True),
        ],
        ignore_index=True,
    )
    adj.to_csv(OUT_DIR / "chosen_train_feature_external_adjusted_pvalues.csv", index=False)
    boot = bootstrap_metrics(s["y"], c_s, xs[:, j], thresholds, width, lam)
    boot.to_csv(OUT_DIR / "chosen_train_feature_external_bootstrap.csv", index=False)
    plot_selected(
        chosen_eval,
        OUT_DIR / "chosen_train_feature_external_operating_points.png",
        f"Train-selected midrange AEC feature: rank {int(chosen['train_rank'])}",
    )

    summary = {
        "selection_rule": "Features and width/lambda were selected on g1090 OOF only. The objective used only Youden, Sens80, and Sens85, with balanced_gain = specificity_gain - sensitivity_loss.",
        "primary_operating_points": sorted(PRIMARY_OPS),
        "n_features_after_filtering": len(names),
        "top_train_feature": str(train_ranked.iloc[0]["feature"]),
        "top_train_width": float(train_ranked.iloc[0]["width"]),
        "top_train_lambda": float(train_ranked.iloc[0]["lambda"]),
        "reported_feature_train_rank": int(chosen["train_rank"]),
        "reported_feature": feature,
        "reported_width": width,
        "reported_lambda": lam,
        "external_primary_summary_for_reported_feature": {
            k: float(chosen[k])
            for k in [
                "external_primary_avg_spec_gain",
                "external_primary_avg_sens_loss",
                "external_primary_avg_balanced_gain",
                "external_primary_min_balanced_gain",
                "external_primary_min_deesc_n",
                "external_primary_mean_deesc_prevalence",
                "external_primary_max_fisher_p",
            ]
        },
    }
    (OUT_DIR / "midrange_feature_refit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nExternal primary summary, train-selected top 15:")
    print(ext_summary.head(15).to_string(index=False))
    print("\nChosen feature all operating points:")
    print(chosen_eval[chosen_eval["dataset"].eq("sdata_external")].to_string(index=False))


if __name__ == "__main__":
    main()
