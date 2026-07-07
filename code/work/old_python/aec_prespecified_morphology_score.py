from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec_conditional_value import clinical_estimator, clinical_matrix, make_folds, oof_and_external, threshold_youden, zfit_apply  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_prespecified_morphology_score"
SEED = 20260630
OPERATING_POINTS = [("youden", None), ("sens80", 0.80), ("sens85", 0.85), ("sens90", 0.90), ("sens95", 0.95)]


def mean_seg(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """1-based 구간 [start, end]의 평균값(행마다)을 계산."""
    return x[:, start - 1 : end].mean(axis=1)


def prespecified_features(x: np.ndarray) -> pd.DataFrame:
    """사전에 지정된 5개 형태 특징(중간레벨, 회복레벨, 회복-중간차, 전환기울기, 후반 평탄도)을 계산 — 데이터를 보고 고른 게 아니라 미리 정의해둔 특징이라는 점이 핵심."""
    d1 = np.diff(x, axis=1)
    mid = mean_seg(x, 55, 84)
    recovery = mean_seg(x, 88, 116)
    transition_slope = d1[:, 46 - 1 : 62 - 1].mean(axis=1)
    tail_rough = np.abs(d1[:, 114 - 1 : 127 - 1]).mean(axis=1)
    return pd.DataFrame(
        {
            "mid_level_55_84": mid,
            "recovery_level_88_116": recovery,
            "recovery_minus_mid": recovery - mid,
            "transition_slope_46_62": transition_slope,
            "tail_flatness_114_127": -tail_rough,
        }
    )


def winsorize_standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """train 중앙값으로 결측 대체, 1~99% 분위수로 이상치 클리핑(winsorize), train 평균/표준편차로 표준화."""
    xtr = train.to_numpy(dtype=float)
    xte = test.to_numpy(dtype=float)
    med = np.nanmedian(xtr, axis=0)
    med[~np.isfinite(med)] = 0.0
    xtr = np.where(np.isfinite(xtr), xtr, med)
    xte = np.where(np.isfinite(xte), xte, med)
    lo = np.nanquantile(xtr, 0.01, axis=0)
    hi = np.nanquantile(xtr, 0.99, axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    xtr[:, ok] = np.clip(xtr[:, ok], lo[ok], hi[ok])
    xte[:, ok] = np.clip(xte[:, ok], lo[ok], hi[ok])
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd, list(train.columns)


def fit_oof_logit_score(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, c: float = 1e6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-fold 로지스틱 회귀로 train의 OOF 점수를 만들고, 전체 train으로 재학습한 모델의 외부 점수와, 폴드별+최종 계수 행렬을 함께 반환."""
    folds = make_folds(ytr.astype(int), 5)
    oof = np.zeros(len(ytr), dtype=float)
    all_idx = np.arange(len(ytr))
    coefs = []
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED + fold_id)
        model.fit(xtr[tr_idx], ytr[tr_idx])
        oof[val_idx] = model.decision_function(xtr[val_idx])
        coefs.append(model.coef_.ravel())
    final = LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED + 99)
    final.fit(xtr, ytr)
    return oof, final.decision_function(xte), np.vstack(coefs + [final.coef_.ravel()])


def zfit(train_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 train·test 점수를 함께 z-표준화."""
    mu = float(np.mean(train_score))
    sd = float(np.std(train_score)) or 1.0
    return (train_score - mu) / sd, (test_score - mu) / sd


def binary_counts(y: np.ndarray, pred: np.ndarray) -> dict:
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


def threshold_at_min_sens(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """목표 민감도(target)를 만족하는 임계값을 계산 (aec_universal_boundary_gate의 함수를 감싼 래퍼)."""
    return threshold_for_min_sensitivity(y.astype(int), score, target)


def threshold_eval_rows(y: np.ndarray, score_clin: np.ndarray, score_full: np.ndarray, thresholds: dict[str, dict]) -> list[dict]:
    """여러 운영점(youden/sens80~95)마다 임상 단독 vs (임상+형태특징) 모델의 혼동행렬과 민감도·특이도·PPV 차이를 계산."""
    rows = []
    for label, cfg in thresholds.items():
        t_clin = cfg["clinical_z"]
        t_full = cfg["full_z"]
        clinical = binary_counts(y, score_clin >= t_clin)
        full = binary_counts(y, score_full >= t_full)
        rows.append(
            {
                "operating_point": label,
                "target_sensitivity": cfg["target_sensitivity"],
                "threshold_method": cfg["method"],
                **{f"clinical_{k}": v for k, v in clinical.items()},
                **{f"full_{k}": v for k, v in full.items()},
                "sensitivity_delta_full_minus_clinical": full["sensitivity"] - clinical["sensitivity"],
                "specificity_delta_full_minus_clinical": full["specificity"] - clinical["specificity"],
                "ppv_delta_full_minus_clinical": full["ppv"] - clinical["ppv"],
            }
        )
    return rows


def deesc_rows(y: np.ndarray, clinical: np.ndarray, morphology: np.ndarray, thresholds: dict[str, dict], width: float = 0.40, lam: float = 0.70) -> list[dict]:
    """여러 운영점마다 가우시안 경계 게이트로 임상양성군을 하향조정했을 때의 유지/하향조정군 통계, 민감도손실/특이도이득, Fisher p값을 계산."""
    rows = []
    for label, cfg in thresholds.items():
        t = cfg["clinical_z"]
        boundary = np.exp(-0.5 * ((clinical - t) / width) ** 2)
        gate = clinical + lam * boundary * morphology
        clinical_pos = clinical >= t
        final_pos = clinical_pos & (gate >= t)
        deesc = clinical_pos & ~final_pos
        base = binary_counts(y, clinical_pos)
        rule = binary_counts(y, final_pos)
        keep = final_pos
        a = int(np.sum(y[keep] == 1))
        b = int(np.sum(y[keep] == 0))
        c = int(np.sum(y[deesc] == 1))
        d = int(np.sum(y[deesc] == 0))
        fisher_p = stats.fisher_exact([[a, b], [c, d]])[1] if np.sum(keep) and np.sum(deesc) else np.nan
        rows.append(
            {
                "operating_point": label,
                "target_sensitivity": cfg["target_sensitivity"],
                "threshold_method": cfg["method"],
                "width": width,
                "lambda": lam,
                **{f"clinical_{k}": v for k, v in base.items()},
                **{f"rule_{k}": v for k, v in rule.items()},
                "deesc_n": int(np.sum(deesc)),
                "deesc_events": c,
                "deesc_prevalence": c / (c + d) if c + d else np.nan,
                "fp_removed": d,
                "tp_lost": c,
                "specificity_gain": rule["specificity"] - base["specificity"],
                "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
                "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
            }
        )
    return rows


def deesc_adjusted_p(
    y: np.ndarray,
    clinical: np.ndarray,
    morphology: np.ndarray,
    manufacturer: np.ndarray,
    thresholds: dict[str, dict],
    width: float = 0.40,
    lam: float = 0.70,
    include_clinical: bool = True,
) -> pd.DataFrame:
    """제조사(스캐너) 더미변수를 통제한 로지스틱 회귀로, "하향조정 여부"가 결과와 독립적으로 연관되는지
    (스캐너 차이로 설명되는 게 아닌지) LRT와 Wald 검정으로 확인 — include_clinical로 임상점수 통제 여부 선택."""
    rows = []
    for label, cfg in thresholds.items():
        t = cfg["clinical_z"]
        boundary = np.exp(-0.5 * ((clinical - t) / width) ** 2)
        gate = clinical + lam * boundary * morphology
        clinical_pos = clinical >= t
        deesc = clinical_pos & (gate < t)
        yy = y[clinical_pos].astype(int)
        if np.unique(yy).size < 2 or deesc[clinical_pos].sum() == 0:
            rows.append(
                {
                    "operating_point": label,
                    "target_sensitivity": cfg["target_sensitivity"],
                    "threshold_method": cfg["method"],
                    "adjustment": "manufacturer_plus_clinical" if include_clinical else "manufacturer_only",
                    "deesc_or": np.nan,
                    "deesc_wald_p": np.nan,
                    "deesc_lrt_p": np.nan,
                }
            )
            continue
        base = pd.DataFrame()
        full = pd.DataFrame({"deesc": deesc[clinical_pos].astype(float)})
        if include_clinical:
            base["clinical_z"] = clinical[clinical_pos]
            full.insert(0, "clinical_z", clinical[clinical_pos])
        m = pd.Series(manufacturer[clinical_pos].astype(str))
        m = m.where(m.map(m.value_counts()) >= 20, "OTHER")
        dummies = pd.get_dummies(m, prefix="manufacturer", drop_first=True, dtype=float)
        base = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        full = pd.concat([full.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        try:
            fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=1000)
            fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=1000)
            lrt = 2 * (fit1.llf - fit0.llf)
            rows.append(
                {
                    "operating_point": label,
                    "target_sensitivity": cfg["target_sensitivity"],
                    "threshold_method": cfg["method"],
                    "adjustment": "manufacturer_plus_clinical" if include_clinical else "manufacturer_only",
                    "n_clinical_positive": int(clinical_pos.sum()),
                    "deesc_or": float(np.exp(fit1.params["deesc"])),
                    "deesc_coef": float(fit1.params["deesc"]),
                    "deesc_wald_p": float(fit1.pvalues["deesc"]),
                    "deesc_lrt_p": float(stats.chi2.sf(lrt, 1)),
                    "manufacturer_groups": int(m.nunique()),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "operating_point": label,
                    "target_sensitivity": cfg["target_sensitivity"],
                    "threshold_method": cfg["method"],
                    "adjustment": "manufacturer_plus_clinical" if include_clinical else "manufacturer_only",
                    "deesc_or": np.nan,
                    "deesc_wald_p": np.nan,
                    "deesc_lrt_p": np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_deesc_gate(
    y: np.ndarray,
    clinical: np.ndarray,
    morphology: np.ndarray,
    thresholds: dict[str, dict],
    width: float = 0.40,
    lam: float = 0.70,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """운영점별 하향조정 게이트의 특이도이득/민감도손실/하향조정군 통계를 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 22)
    rows = []
    for label, cfg in thresholds.items():
        t = cfg["clinical_z"]
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if np.unique(yy).size < 2:
                continue
            m = deesc_rows(yy, clinical[idx], morphology[idx], {label: cfg}, width=width, lam=lam)[0]
            vals.append([m["specificity_gain"], m["sensitivity_loss"], m["deesc_n"], m["deesc_events"], m["deesc_prevalence"], m["fp_removed"], m["tp_lost"]])
        arr = np.asarray(vals)
        for i, metric in enumerate(["specificity_gain", "sensitivity_loss", "deesc_n", "deesc_events", "deesc_prevalence", "fp_removed", "tp_lost"]):
            x = arr[:, i]
            rows.append(
                {
                    "operating_point": label,
                    "target_sensitivity": cfg["target_sensitivity"],
                    "threshold_method": cfg["method"],
                    "metric": metric,
                    "mean": float(np.mean(x)),
                    "ci2.5": float(np.quantile(x, 0.025)),
                    "ci97.5": float(np.quantile(x, 0.975)),
                    "p_le_0": float((np.sum(x <= 0) + 1) / (len(x) + 1)) if metric in {"specificity_gain"} else np.nan,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metric_delta(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, th_a: float, th_b: float, n_boot: int = 2000) -> dict:
    """두 점수(a vs b)의 AUC·특이도·민감도 차이를 부트스트랩 재표본추출로 신뢰구간과 단측 p값과 함께 추정."""
    rng = np.random.default_rng(SEED + 11)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        ca = binary_counts(yy, score_a[idx] >= th_a)
        cb = binary_counts(yy, score_b[idx] >= th_b)
        vals.append(
            [
                roc_auc_score(yy, score_b[idx]) - roc_auc_score(yy, score_a[idx]),
                cb["specificity"] - ca["specificity"],
                cb["sensitivity"] - ca["sensitivity"],
            ]
        )
    arr = np.asarray(vals)
    rows = {}
    for i, name in enumerate(["delta_auc", "delta_specificity", "delta_sensitivity"]):
        x = arr[:, i]
        rows[f"{name}_mean"] = float(np.mean(x))
        rows[f"{name}_ci2.5"] = float(np.quantile(x, 0.025))
        rows[f"{name}_ci97.5"] = float(np.quantile(x, 0.975))
        rows[f"{name}_p_le_0"] = float((np.sum(x <= 0) + 1) / (len(x) + 1))
    return rows


def logistic_lrt(y: np.ndarray, base_cols: dict[str, np.ndarray], add_cols: dict[str, np.ndarray], manufacturer: np.ndarray | None = None) -> dict:
    """기본 변수만 넣은 모델과 추가 변수까지 넣은 모델의 우도비검정(LRT)을 수행하고, manufacturer가 주어지면 제조사 더미변수까지 통제해 계수·오즈비·p값을 계산."""
    base = pd.DataFrame(base_cols)
    full = pd.DataFrame({**base_cols, **add_cols})
    if manufacturer is not None:
        m = pd.Series(manufacturer.astype(str)).where(pd.Series(manufacturer.astype(str)).map(pd.Series(manufacturer.astype(str)).value_counts()) >= 20, "OTHER")
        dummies = pd.get_dummies(m, prefix="manufacturer", drop_first=True, dtype=float)
        base = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        full = pd.concat([full.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    x0 = sm.add_constant(base.astype(float), has_constant="add")
    x1 = sm.add_constant(full.astype(float), has_constant="add")
    try:
        m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
        m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
        stat = 2 * (m1.llf - m0.llf)
        df = x1.shape[1] - x0.shape[1]
        out = {
            "lrt_chi2": float(stat),
            "df": int(df),
            "lrt_p": float(stats.chi2.sf(stat, df)),
        }
        for col in add_cols:
            out[f"{col}_coef"] = float(m1.params[col])
            out[f"{col}_or"] = float(np.exp(m1.params[col]))
            out[f"{col}_wald_p"] = float(m1.pvalues[col])
        return out
    except Exception as exc:
        return {"lrt_chi2": np.nan, "df": len(add_cols), "lrt_p": np.nan, "error": str(exc)}


def manufacturer_r2(score: np.ndarray, manufacturer: np.ndarray) -> dict:
    """점수를 제조사 더미변수로 회귀시켜, 점수가 스캐너(제조사) 차이로 얼마나 설명되는지 R²와 F검정 p값으로 확인."""
    m = pd.Series(manufacturer.astype(str))
    m = m.where(m.map(m.value_counts()) >= 20, "OTHER")
    x = sm.add_constant(pd.get_dummies(m, drop_first=True, dtype=float), has_constant="add")
    fit = sm.OLS(score, x).fit()
    return {"r2": float(fit.rsquared), "f_p": float(fit.f_pvalue) if fit.f_pvalue is not None else np.nan, "n_groups": int(m.nunique())}


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 데이터를 보고 나서 고른 특징이 아니라, "사전에 지정된"
    5개 형태 특징만으로도 임상변수 대비 이득이 있고, 그 이득이 스캐너 제조사 차이 때문이 아닌
    진짜 신호인가?):

    1. g1090/sdata를 로드하고 prespecified_features로 5개 사전지정 형태 특징을 계산해 CSV로 저장.
    2. 임상 단독 모델, 형태특징 단독 로지스틱 모델, 임상+형태특징 결합 모델을 각각 5-fold OOF로
       학습해 표준화된 점수를 만들고, 계수들도 CSV로 저장.
    3. Youden/민감도80~95% 5개 운영점마다 임상 임계값을 g1090에서 고정하고, threshold_eval_rows로
       임상 단독 vs 결합모델의 성능을 train/외부에서 비교.
    4. deesc_rows로 각 운영점에서 가우시안 게이트 하향조정 규칙의 성능을 계산.
    5. deesc_adjusted_p로 "하향조정 여부"가 제조사(스캐너) 더미변수를 통제해도 여전히 결과와
       유의하게 연관되는지 확인 (스캐너 차이로 설명되는 가짜 신호가 아닌지 검증) — 임상점수
       포함/미포함 두 버전으로.
    6. 외부 데이터에서 하향조정 게이트와 결합모델의 부트스트랩 신뢰구간을 계산.
    7. logistic_lrt로 임상점수(+제조사 통제) 대비 형태점수의 조건부 연관성을 검정하고,
       manufacturer_r2로 형태점수 자체가 제조사 차이로 얼마나 설명되는지도 확인.
    8. 외부 운영점별 특이도 비교 그래프를 저장하고, 특징 정의·AUC 요약·전체 결과를 JSON으로
       저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_aec128(FILES["g1090"])
    test = load_aec128(FILES["sdata"])
    y_g = train["y"].astype(int)
    y_s = test["y"].astype(int)

    f_g = prespecified_features(train["x"])
    f_s = prespecified_features(test["x"])
    f_g.to_csv(OUT_DIR / "g1090_prespecified_features.csv", index=False)
    f_s.to_csv(OUT_DIR / "sdata_prespecified_features.csv", index=False)
    x_g, x_s, names = winsorize_standardize(f_g, f_s)

    xclin_g, xclin_s, _ = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(y_g, 5)
    clinical_oof_raw, clinical_ext_raw = oof_and_external(lambda seed: clinical_estimator(), xclin_g, y_g, xclin_s, folds)
    clinical_g, clinical_s, _, _ = zfit_apply(clinical_oof_raw, clinical_ext_raw)

    aec_oof_raw, aec_ext_raw, coef_mat = fit_oof_logit_score(x_g, y_g, x_s)
    aec_g, aec_s = zfit(aec_oof_raw, aec_ext_raw)
    if np.corrcoef(aec_g, y_g)[0, 1] < 0:
        aec_g = -aec_g
        aec_s = -aec_s
        coef_mat = -coef_mat

    full_oof_raw, full_ext_raw, full_coef = fit_oof_logit_score(np.column_stack([clinical_g, x_g]), y_g, np.column_stack([clinical_s, x_s]))
    full_g, full_s = zfit(full_oof_raw, full_ext_raw)
    if np.corrcoef(full_g, y_g)[0, 1] < 0:
        full_g = -full_g
        full_s = -full_s
        full_coef = -full_coef

    pd.DataFrame(coef_mat, columns=names, index=[f"fold_{i}" for i in range(1, 6)] + ["final"]).to_csv(OUT_DIR / "aec_morphology_logit_coefficients.csv")
    pd.DataFrame(full_coef, columns=["clinical_z", *names], index=[f"fold_{i}" for i in range(1, 6)] + ["final"]).to_csv(OUT_DIR / "clinical_plus_morphology_logit_coefficients.csv")

    clinical_mu = float(np.mean(clinical_oof_raw))
    clinical_sd = float(np.std(clinical_oof_raw)) or 1.0
    full_mu = float(np.mean(full_oof_raw))
    full_sd = float(np.std(full_oof_raw)) or 1.0
    thresholds_pair = {}
    for label, target in OPERATING_POINTS:
        if target is None:
            clinical_raw_th = threshold_youden(y_g, clinical_oof_raw)
            full_raw_th = threshold_youden(y_g, full_oof_raw)
            method = "youden"
            target_value = np.nan
        else:
            clinical_raw_th = threshold_at_min_sens(y_g, clinical_oof_raw, target)
            full_raw_th = threshold_at_min_sens(y_g, full_oof_raw, target)
            method = "minimum_training_sensitivity"
            target_value = target
        thresholds_pair[label] = {
            "target_sensitivity": target_value,
            "method": method,
            "clinical_raw": float(clinical_raw_th),
            "clinical_z": float((clinical_raw_th - clinical_mu) / clinical_sd),
            "full_raw": float(full_raw_th),
            "full_z": float((full_raw_th - full_mu) / full_sd),
        }

    perf_rows = []
    for dataset, y, cscore, fscore in [("g1090_oof", y_g, clinical_g, full_g), ("sdata_external", y_s, clinical_s, full_s)]:
        for r in threshold_eval_rows(y, cscore, fscore, thresholds_pair):
            perf_rows.append({"dataset": dataset, **r})
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(OUT_DIR / "clinical_vs_clinical_plus_morphology_thresholds.csv", index=False)

    de_rows = []
    for dataset, y, cscore, ascore in [("g1090_oof", y_g, clinical_g, aec_g), ("sdata_external", y_s, clinical_s, aec_s)]:
        for r in deesc_rows(y, cscore, ascore, thresholds_pair, width=0.40, lam=0.70):
            de_rows.append({"dataset": dataset, **r})
    deesc = pd.DataFrame(de_rows)
    deesc.to_csv(OUT_DIR / "morphology_score_deescalation_gate.csv", index=False)
    deesc_adj_rows = []
    for dataset, y, cscore, ascore, meta in [
        ("g1090_oof", y_g, clinical_g, aec_g, train["meta"]),
        ("sdata_external", y_s, clinical_s, aec_s, test["meta"]),
    ]:
        for include_clinical in [False, True]:
            tmp = deesc_adjusted_p(
                y,
                cscore,
                ascore,
                meta["Manufacturer"].astype(str).to_numpy(),
                thresholds_pair,
                width=0.40,
                lam=0.70,
                include_clinical=include_clinical,
            )
            tmp["dataset"] = dataset
            deesc_adj_rows.append(tmp)
    deesc_adj = pd.concat(deesc_adj_rows, ignore_index=True)
    deesc_adj.to_csv(OUT_DIR / "morphology_deescalation_manufacturer_adjusted_pvalues.csv", index=False)
    deesc_boot = bootstrap_deesc_gate(y_s, clinical_s, aec_s, thresholds_pair, width=0.40, lam=0.70)
    deesc_boot["dataset"] = "sdata_external"
    deesc_boot.to_csv(OUT_DIR / "external_deescalation_bootstrap.csv", index=False)

    boot_rows = []
    for label, cfg in thresholds_pair.items():
        boot_rows.append(
            {
                "dataset": "sdata_external",
                "operating_point": label,
                "target_sensitivity": cfg["target_sensitivity"],
                "threshold_method": cfg["method"],
                **bootstrap_metric_delta(y_s, clinical_s, full_s, cfg["clinical_z"], cfg["full_z"]),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT_DIR / "external_threshold_bootstrap_clinical_plus_morphology.csv", index=False)

    lrt_rows = []
    for dataset, y, cscore, ascore, meta in [
        ("g1090_oof", y_g, clinical_g, aec_g, train["meta"]),
        ("sdata_external", y_s, clinical_s, aec_s, test["meta"]),
    ]:
        lrt_rows.append({"dataset": dataset, "adjustment": "clinical_only", **logistic_lrt(y, {"clinical_z": cscore}, {"aec_morphology_z": ascore})})
        lrt_rows.append(
            {
                "dataset": dataset,
                "adjustment": "clinical_plus_manufacturer_dummies",
                **logistic_lrt(y, {"clinical_z": cscore}, {"aec_morphology_z": ascore}, manufacturer=meta["Manufacturer"].astype(str).to_numpy()),
            }
        )
    lrt = pd.DataFrame(lrt_rows)
    lrt.to_csv(OUT_DIR / "morphology_score_conditional_pvalues_with_manufacturer.csv", index=False)

    mr_rows = []
    for dataset, score, meta in [("g1090_oof", aec_g, train["meta"]), ("sdata_external", aec_s, test["meta"])]:
        mr_rows.append({"dataset": dataset, "score": "aec_morphology_z", **manufacturer_r2(score, meta["Manufacturer"].astype(str).to_numpy())})
    mr = pd.DataFrame(mr_rows)
    mr.to_csv(OUT_DIR / "manufacturer_association_with_morphology_score.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ext = perf[perf["dataset"].eq("sdata_external")].copy()
    order = {label: i for i, (label, _target) in enumerate(OPERATING_POINTS)}
    ext["xpos"] = ext["operating_point"].map(order)
    ext = ext.sort_values("xpos")
    ax.plot(ext["xpos"], ext["clinical_specificity"] * 100, marker="o", lw=2, label="Clinical only")
    ax.plot(ext["xpos"], ext["full_specificity"] * 100, marker="o", lw=2, label="Clinical + AEC morphology")
    for _, r in ext.iterrows():
        ax.annotate(f'{r["full_sensitivity"] * 100:.1f}/{r["full_specificity"] * 100:.1f}', (r["xpos"], r["full_specificity"] * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xticks(list(order.values()))
    ax.set_xticklabels(list(order.keys()))
    ax.set_xlabel("g1090 operating point")
    ax.set_ylabel("sdata specificity (%)")
    ax.set_title("Prespecified AEC morphology score: external operating points", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_specificity_by_operating_point.png", dpi=220)
    plt.close(fig)

    summary = {
        "feature_family": {
            "mid_level_55_84": "mean normalized AEC 55-84",
            "recovery_level_88_116": "mean normalized AEC 88-116",
            "recovery_minus_mid": "recovery_level_88_116 - mid_level_55_84",
            "transition_slope_46_62": "mean first difference across AEC 46-62",
            "tail_flatness_114_127": "-mean absolute first difference across AEC 114-127",
        },
        "scanner_policy": "Manufacturer is not used in the deployable score. Manufacturer dummies are used only as nuisance adjustment in inference/sensitivity analysis because future scanners may be unseen.",
        "auc": {
            "clinical_g1090_oof": float(roc_auc_score(y_g, clinical_g)),
            "clinical_sdata_external": float(roc_auc_score(y_s, clinical_s)),
            "aec_morphology_g1090_oof": float(roc_auc_score(y_g, aec_g)),
            "aec_morphology_sdata_external": float(roc_auc_score(y_s, aec_s)),
            "clinical_plus_morphology_g1090_oof": float(roc_auc_score(y_g, full_g)),
            "clinical_plus_morphology_sdata_external": float(roc_auc_score(y_s, full_s)),
        },
        "outputs": {
            "thresholds": str(OUT_DIR / "clinical_vs_clinical_plus_morphology_thresholds.csv"),
            "deescalation": str(OUT_DIR / "morphology_score_deescalation_gate.csv"),
            "deescalation_adjusted_p": str(OUT_DIR / "morphology_deescalation_manufacturer_adjusted_pvalues.csv"),
            "deescalation_bootstrap": str(OUT_DIR / "external_deescalation_bootstrap.csv"),
            "pvalues": str(OUT_DIR / "morphology_score_conditional_pvalues_with_manufacturer.csv"),
            "manufacturer_r2": str(OUT_DIR / "manufacturer_association_with_morphology_score.csv"),
            "plot": str(OUT_DIR / "external_specificity_by_operating_point.png"),
        },
    }
    (OUT_DIR / "prespecified_morphology_score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nAUC summary")
    print(pd.Series(summary["auc"]).to_string())
    print("\nExternal operating points")
    print(ext.to_string(index=False))
    print("\nDe-escalation gate")
    print(deesc[deesc["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nDe-escalation manufacturer-adjusted p-values")
    print(deesc_adj[deesc_adj["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nExternal de-escalation bootstrap")
    print(deesc_boot.to_string(index=False))
    print("\nConditional p-values")
    print(lrt.to_string(index=False))
    print("\nManufacturer association with morphology score")
    print(mr.to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
