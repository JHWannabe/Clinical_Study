from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec_offset_score import clinical_raw, sigmoid  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_five_strategy_audit"
SEED = 20260629


FEATURE_SPECS = {
    "tail_rebound_114_128_max_minus_60_95_min": "max(AEC_114..128) - min(AEC_60..95)",
    "late_rebound_91_128_max_minus_60_95_min": "max(AEC_91..128) - min(AEC_60..95)",
    "tail_level_120_128": "mean(AEC_120..128)",
    "tail_level_114_128": "mean(AEC_114..128)",
    "recovery_slope_81_113": "mean(AEC_{j+1} - AEC_j), j=81..113",
    "roughness_75_90": "mean(abs(AEC_{j+1} - AEC_j)), j=75..90",
    "trough_range_60_95": "max(AEC_60..95) - min(AEC_60..95)",
    "early_slope_31_38": "mean(AEC_{j+1} - AEC_j), j=31..38",
}


def smi_from_meta(meta: pd.DataFrame) -> np.ndarray:
    """메타데이터에 SMI 컬럼이 있고 대부분 유효하면 그대로 쓰고, 아니면 TAMA/키^2로 SMI를 직접 계산."""
    if "SMI" in meta.columns:
        smi = pd.to_numeric(meta["SMI"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(smi).mean() > 0.95:
            return smi
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    return tama / (height_m**2)


def endpoint_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    """SMI와 함께 TAMA/IMATA/BMI/체중/키/나이 등 신체계측 관련 결과변수들을 하나의 표로 모음."""
    out = pd.DataFrame(index=meta.index)
    out["SMI"] = smi_from_meta(meta)
    for col in ["TAMA", "IMATA", "BMI", "Weight", "Height", "PatientAge"]:
        if col in meta.columns:
            out[col] = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
    return out


def build_aec_features(x: np.ndarray) -> pd.DataFrame:
    """FEATURE_SPECS에 정의된 8개 사전 지정 AEC_128 특징(후반 반등, 후반 레벨, 회복 기울기, 거칠기, 트로프 범위, 초반 기울기)을 계산."""
    x = np.asarray(x, dtype=float)
    d1 = np.diff(x, axis=1)
    feats = {
        "tail_rebound_114_128_max_minus_60_95_min": x[:, 113:128].max(axis=1) - x[:, 59:95].min(axis=1),
        "late_rebound_91_128_max_minus_60_95_min": x[:, 90:128].max(axis=1) - x[:, 59:95].min(axis=1),
        "tail_level_120_128": x[:, 119:128].mean(axis=1),
        "tail_level_114_128": x[:, 113:128].mean(axis=1),
        "recovery_slope_81_113": d1[:, 80:113].mean(axis=1),
        "roughness_75_90": np.abs(d1[:, 74:90]).mean(axis=1),
        "trough_range_60_95": x[:, 59:95].max(axis=1) - x[:, 59:95].min(axis=1),
        "early_slope_31_38": d1[:, 30:38].mean(axis=1),
    }
    return pd.DataFrame(feats)


def clinical_pipeline() -> Pipeline:
    """결측대체→표준화→로지스틱 회귀(정규화 거의 없음)로 이어지는 임상 모델 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def linear_pipeline() -> Pipeline:
    """결측대체→표준화→선형회귀로 이어지는, 임상변수로 연속형 결과변수를 예측하는 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lin", LinearRegression()),
        ]
    )


def zfit_apply(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """train의 평균/표준편차로 train·test를 함께 z-표준화."""
    mu = float(np.nanmean(xtr))
    sd = float(np.nanstd(xtr))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd, mu, sd


def metric_row(dataset: str, model: str, y: np.ndarray, score: np.ndarray) -> dict:
    """데이터셋/모델 이름과 점수로부터 AUC/AP/로그손실/Brier를 한 행으로 정리."""
    prob = np.clip(sigmoid(score), 1e-6, 1 - 1e-6)
    return {
        "dataset": dataset,
        "model": model,
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """주어진 임계값에서 TP/FP/FN/TN과 민감도·특이도·PPV·NPV를 계산."""
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """목표 민감도(target) 이상을 유지하면서 특이도가 가장 높은 임계값을 찾음 (해당하는 값이 없으면 분위수로 근사)."""
    best = None
    for th in np.unique(score):
        m = binary_metrics(y, score, float(th))
        if m["sensitivity"] >= target and (best is None or m["specificity"] > best[1]):
            best = (float(th), m["specificity"])
    if best is None:
        return float(np.quantile(score[y == 1], 1 - target))
    return best[0]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 3000) -> tuple[float, float]:
    """값들의 평균에 대한 부트스트랩 95% 신뢰구간을 계산."""
    rng = np.random.default_rng(SEED)
    vals = np.asarray(values, dtype=float)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        boots[i] = float(np.mean(vals[idx]))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def nearest_pairs(y: np.ndarray, clinical_score: np.ndarray, caliper_sd: float = 0.20) -> pd.DataFrame:
    """각 저근감소증 환자를 임상점수가 가장 가까운 비저근감소증 환자와 1:1 매칭하고, 거리가 caliper 이내인지 표시."""
    yb = y.astype(bool)
    low_idx = np.flatnonzero(yb)
    non_idx = np.flatnonzero(~yb)
    non_scores = clinical_score[non_idx]
    order = np.argsort(non_scores)
    sorted_non_idx = non_idx[order]
    sorted_non_scores = non_scores[order]
    caliper = float(caliper_sd * np.nanstd(clinical_score))
    rows = []
    for i in low_idx:
        pos = int(np.searchsorted(sorted_non_scores, clinical_score[i]))
        cand = []
        if pos < len(sorted_non_idx):
            cand.append(sorted_non_idx[pos])
        if pos > 0:
            cand.append(sorted_non_idx[pos - 1])
        if not cand:
            continue
        j = min(cand, key=lambda jj: abs(clinical_score[i] - clinical_score[jj]))
        dist = float(abs(clinical_score[i] - clinical_score[j]))
        rows.append(
            {
                "low_index": int(i),
                "matched_nonlow_index": int(j),
                "abs_clinical_score_distance": dist,
                "within_caliper": bool(dist <= caliper),
                "caliper": caliper,
            }
        )
    return pd.DataFrame(rows)


def load_all() -> dict:
    """g1090/sdata를 로드하고, 각각에 임상변수·8개 AEC 특징·결과변수 표·여성 여부까지 미리 계산해 붙여둠."""
    data = {}
    for cohort, path in FILES.items():
        d = load_aec128(path)
        d["y"] = d["y"].astype(int)
        d["clinical_x"] = clinical_raw(d["meta"])
        d["features"] = build_aec_features(d["x"])
        d["endpoints"] = endpoint_matrix(d["meta"])
        sex = d["meta"]["PatientSex"].astype(str).str.upper().to_numpy()
        d["female"] = (sex == "F").astype(float)
        data[cohort] = d
    return data


def clinical_scores(data: dict) -> None:
    """g1090 5-fold OOF 임상점수와, 전체 g1090으로 학습한 모델의 g1090/sdata 점수를 계산해 data 딕셔너리에 채워 넣음 (in-place)."""
    g = data["g1090"]
    s = data["sdata"]
    y = g["y"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(g["clinical_x"], y):
        model = clinical_pipeline()
        model.fit(g["clinical_x"][tr], y[tr])
        oof[va] = model.decision_function(g["clinical_x"][va])
    final = clinical_pipeline()
    final.fit(g["clinical_x"], y)
    g["clinical_score_oof"] = oof
    g["clinical_score_full"] = final.decision_function(g["clinical_x"])
    s["clinical_score_external"] = final.decision_function(s["clinical_x"])


def strategy_1_residual_endpoint(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[전략1] SMI/TAMA/IMATA/BMI 각 결과변수를 임상변수로 예측한 뒤 그 잔차(임상으로 설명 안 되는 부분)를
    구하고, 8개 AEC 특징이 그 잔차와 상관이 있는지(Pearson/Spearman) 검정 — "임상변수로 설명 안 되는
    체성분 변동을 AEC가 잡아내는가?"를 보는 첫 번째 접근."""
    g = data["g1090"]
    s = data["sdata"]
    endpoints = ["SMI", "TAMA", "IMATA", "BMI"]
    rows = []
    pred_rows = []

    for endpoint in endpoints:
        if endpoint not in g["endpoints"].columns or endpoint not in s["endpoints"].columns:
            continue
        yg = g["endpoints"][endpoint].to_numpy(dtype=float)
        ys = s["endpoints"][endpoint].to_numpy(dtype=float)
        ok_g = np.isfinite(yg)
        ok_s = np.isfinite(ys)

        # OOF residual in g1090.
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        pred = np.full(len(yg), np.nan)
        for tr, va in kf.split(g["clinical_x"]):
            tr = tr[ok_g[tr]]
            va_ok = va[ok_g[va]]
            model = linear_pipeline()
            model.fit(g["clinical_x"][tr], yg[tr])
            pred[va_ok] = model.predict(g["clinical_x"][va_ok])
        resid_g = yg - pred

        # External residual using g1090 clinical fit.
        model = linear_pipeline()
        model.fit(g["clinical_x"][ok_g], yg[ok_g])
        pred_g_full = np.full(len(yg), np.nan)
        pred_s = np.full(len(ys), np.nan)
        pred_g_full[ok_g] = model.predict(g["clinical_x"][ok_g])
        pred_s[ok_s] = model.predict(s["clinical_x"][ok_s])
        resid_s = ys - pred_s

        pred_rows.extend(
            [
                {
                    "endpoint": endpoint,
                    "cohort": "g1090",
                    "clinical_residual_sd": float(np.nanstd(resid_g)),
                    "clinical_r2_oof": float(1 - np.nansum((yg - pred) ** 2) / np.nansum((yg - np.nanmean(yg)) ** 2)),
                },
                {
                    "endpoint": endpoint,
                    "cohort": "sdata",
                    "clinical_residual_sd": float(np.nanstd(resid_s)),
                    "clinical_r2_external": float(1 - np.nansum((ys - pred_s) ** 2) / np.nansum((ys - np.nanmean(ys)) ** 2)),
                },
            ]
        )

        for feature in FEATURE_SPECS:
            for cohort, d, resid, ok in [("g1090", g, resid_g, ok_g), ("sdata", s, resid_s, ok_s)]:
                x = d["features"][feature].to_numpy(dtype=float)
                mask = ok & np.isfinite(resid) & np.isfinite(x)
                pear = stats.pearsonr(x[mask], resid[mask])
                spear = stats.spearmanr(x[mask], resid[mask])
                rows.append(
                    {
                        "strategy": "1_residual_endpoint",
                        "endpoint": endpoint,
                        "cohort": cohort,
                        "feature": feature,
                        "feature_definition": FEATURE_SPECS[feature],
                        "n": int(np.sum(mask)),
                        "pearson_r_feature_vs_clinical_residual": float(pear.statistic),
                        "pearson_p": float(pear.pvalue),
                        "spearman_r": float(spear.statistic),
                        "spearman_p": float(spear.pvalue),
                    }
                )

    assoc = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    assoc.to_csv(OUT_DIR / "strategy1_residual_endpoint_associations.csv", index=False)
    pred_df.to_csv(OUT_DIR / "strategy1_clinical_residual_quality.csv", index=False)

    # Plot SMI residual associations for the main feature family.
    plot_df = assoc[(assoc["endpoint"].eq("SMI")) & (assoc["feature"].isin(list(FEATURE_SPECS.keys())[:6]))]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    labels = plot_df["feature"].drop_duplicates().tolist()
    x = np.arange(len(labels))
    width = 0.36
    for offset, cohort, color in [(-width / 2, "g1090", "#4C78A8"), (width / 2, "sdata", "#F58518")]:
        vals = [plot_df[(plot_df["feature"].eq(f)) & (plot_df["cohort"].eq(cohort))]["pearson_r_feature_vs_clinical_residual"].iloc[0] for f in labels]
        ax.bar(x + offset, vals, width=width, color=color, label=cohort)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Pearson r with clinical residual SMI")
    ax.set_title("Strategy 1: AEC vs Clinical-Residual SMI", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "strategy1_smi_residual_associations.png", dpi=220)
    plt.close(fig)
    return assoc, pred_df


def strategy_2_matched_contrast(data: dict) -> pd.DataFrame:
    """[전략2] 임상점수로 매칭된 저근감소증-비저근감소증 쌍에서, 8개 AEC 특징과 곡선 전체의 위치별
    값이 통계적으로 유의하게 다른지(대응t검정/Wilcoxon/부트스트랩) 검정 — "임상적으로 같은 사람들
    끼리 비교해도 AEC가 다른가?"를 보는 두 번째 접근."""
    rows = []
    curve_rows = []
    for cohort, score_key in [("g1090", "clinical_score_oof"), ("sdata", "clinical_score_external")]:
        d = data[cohort]
        pairs = nearest_pairs(d["y"], d[score_key])
        pairs = pairs[pairs["within_caliper"]].copy()
        pairs.to_csv(OUT_DIR / f"strategy2_{cohort}_clinical_matched_pairs.csv", index=False)
        low_idx = pairs["low_index"].to_numpy(dtype=int)
        non_idx = pairs["matched_nonlow_index"].to_numpy(dtype=int)
        for feature in FEATURE_SPECS:
            v = d["features"][feature].to_numpy(dtype=float)
            diff = v[low_idx] - v[non_idx]
            ci_low, ci_high = bootstrap_mean_ci(diff)
            paired = stats.ttest_rel(v[low_idx], v[non_idx], nan_policy="omit")
            wil = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            rows.append(
                {
                    "strategy": "2_clinical_matched_contrast",
                    "cohort": cohort,
                    "feature": feature,
                    "feature_definition": FEATURE_SPECS[feature],
                    "n_pairs": int(len(diff)),
                    "matched_low_mean": float(np.mean(v[low_idx])),
                    "matched_nonlow_mean": float(np.mean(v[non_idx])),
                    "matched_diff_low_minus_nonlow": float(np.mean(diff)),
                    "diff_ci2.5": ci_low,
                    "diff_ci97.5": ci_high,
                    "paired_t_p": float(paired.pvalue),
                    "wilcoxon_p": float(wil.pvalue),
                }
            )

        # Curve-level matched difference.
        diffs = d["x"][low_idx] - d["x"][non_idx]
        mean = diffs.mean(axis=0)
        se = diffs.std(axis=0, ddof=1) / np.sqrt(diffs.shape[0])
        for j in range(128):
            curve_rows.append(
                {
                    "cohort": cohort,
                    "point": j + 1,
                    "matched_low_minus_nonlow_mean": mean[j],
                    "ci95_low": mean[j] - 1.96 * se[j],
                    "ci95_high": mean[j] + 1.96 * se[j],
                }
            )
    out = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    out.to_csv(OUT_DIR / "strategy2_matched_feature_contrasts.csv", index=False)
    curves.to_csv(OUT_DIR / "strategy2_matched_curve_difference.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for cohort, color in [("g1090", "#4C78A8"), ("sdata", "#F58518")]:
        sub = curves[curves["cohort"].eq(cohort)]
        ax.plot(sub["point"], sub["matched_low_minus_nonlow_mean"], color=color, lw=2.2, label=cohort)
        ax.fill_between(sub["point"], sub["ci95_low"], sub["ci95_high"], color=color, alpha=0.16, lw=0)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xlabel("AEC_128 point")
    ax.set_ylabel("Matched low - non-low normalized AEC")
    ax.set_title("Strategy 2: Clinical-Matched Curve Contrast", loc="left", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "strategy2_matched_curve_contrast.png", dpi=220)
    plt.close(fig)
    return out


def fit_interaction_score(
    c_tr: np.ndarray,
    f_tr: np.ndarray,
    female_tr: np.ndarray,
    y_tr: np.ndarray,
    c_va: np.ndarray,
    f_va: np.ndarray,
    female_va: np.ndarray,
    model_type: str,
) -> np.ndarray:
    """임상점수+AEC특징(z표준화)에, model_type에 따라 (여성여부 상호작용) 및/또는 (임상점수 상호작용)
    항을 추가한 로지스틱 회귀를 학습해 검증 데이터 점수를 반환."""
    fz_tr, fz_va, _, _ = zfit_apply(f_tr, f_va)
    cz_tr, cz_va, _, _ = zfit_apply(c_tr, c_va)
    cols_tr = [c_tr, fz_tr]
    cols_va = [c_va, fz_va]
    if model_type in {"aec_x_female", "aec_x_both"}:
        cols_tr.append(fz_tr * female_tr)
        cols_va.append(fz_va * female_va)
    if model_type in {"aec_x_clinical", "aec_x_both"}:
        cols_tr.append(fz_tr * cz_tr)
        cols_va.append(fz_va * cz_va)
    ztr = np.column_stack(cols_tr)
    zva = np.column_stack(cols_va)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )
    pipe.fit(ztr, y_tr)
    return pipe.decision_function(zva)


def strategy_3_interactions(data: dict) -> pd.DataFrame:
    """[전략3] 8개 AEC 특징 x 4가지 모델형태(주효과만/여성상호작용/임상점수상호작용/둘다)를 5-fold OOF로
    학습해, 어떤 특징x상호작용 조합이 임상 단독 대비 AUC를 개선하는지 총 32개 모델을 비교 —
    "AEC 효과가 특정 하위집단(여성)이나 임상점수 구간에서만 나타나는가?"를 보는 세 번째 접근."""
    g = data["g1090"]
    s = data["sdata"]
    rows = []
    model_types = ["main_aec", "aec_x_female", "aec_x_clinical", "aec_x_both"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    clinical_oof = g["clinical_score_oof"]
    clinical_ext = s["clinical_score_external"]
    rows.append(metric_row("g1090_oof", "clinical_score_only", g["y"], clinical_oof))
    rows.append(metric_row("sdata_external", "clinical_score_only", s["y"], clinical_ext))

    for feature in FEATURE_SPECS:
        f_g = g["features"][feature].to_numpy(dtype=float)
        f_s = s["features"][feature].to_numpy(dtype=float)
        for mt in model_types:
            oof = np.zeros(len(g["y"]), dtype=float)
            for tr, va in skf.split(g["clinical_x"], g["y"]):
                cm = clinical_pipeline()
                cm.fit(g["clinical_x"][tr], g["y"][tr])
                c_tr = cm.decision_function(g["clinical_x"][tr])
                c_va = cm.decision_function(g["clinical_x"][va])
                oof[va] = fit_interaction_score(
                    c_tr, f_g[tr], g["female"][tr], g["y"][tr], c_va, f_g[va], g["female"][va], mt
                )
            cm = clinical_pipeline()
            cm.fit(g["clinical_x"], g["y"])
            c_tr = cm.decision_function(g["clinical_x"])
            c_te = cm.decision_function(s["clinical_x"])
            ext = fit_interaction_score(c_tr, f_g, g["female"], g["y"], c_te, f_s, s["female"], mt)
            rows.append(metric_row("g1090_oof", f"{feature}_{mt}", g["y"], oof))
            rows.append(metric_row("sdata_external", f"{feature}_{mt}", s["y"], ext))

    perf = pd.DataFrame(rows)
    perf.to_csv(OUT_DIR / "strategy3_interaction_model_performance.csv", index=False)
    return perf


def choose_deescalation_threshold(
    y: np.ndarray,
    clinical_score: np.ndarray,
    aec_score: np.ndarray,
    clinical_th: float,
    max_sensitivity_loss: float,
) -> tuple[float, dict]:
    """임상 양성군 내에서, 민감도 손실이 max_sensitivity_loss를 넘지 않는 한도 내에서 특이도 이득이
    가장 큰 AEC 하향조정 임계값을 탐색."""
    base = binary_metrics(y, clinical_score, clinical_th)
    clinical_pos = clinical_score >= clinical_th
    best = None
    for ath in np.unique(aec_score[clinical_pos]):
        keep = clinical_pos & (aec_score >= ath)
        score_rule = keep.astype(float)
        m = binary_metrics(y, score_rule, 0.5)
        sens_loss = base["sensitivity"] - m["sensitivity"]
        spec_gain = m["specificity"] - base["specificity"]
        deesc_n = int(np.sum(clinical_pos & (aec_score < ath)))
        row = {
            **m,
            "aec_threshold": float(ath),
            "baseline_sensitivity": base["sensitivity"],
            "baseline_specificity": base["specificity"],
            "sensitivity_loss": float(sens_loss),
            "specificity_gain": float(spec_gain),
            "deescalated_n": deesc_n,
        }
        if sens_loss <= max_sensitivity_loss:
            if best is None or row["specificity_gain"] > best["specificity_gain"]:
                best = row
    if best is None:
        ath = float(np.min(aec_score[clinical_pos]))
        keep = clinical_pos & (aec_score >= ath)
        best = {
            **binary_metrics(y, keep.astype(float), 0.5),
            "aec_threshold": ath,
            "baseline_sensitivity": base["sensitivity"],
            "baseline_specificity": base["specificity"],
            "sensitivity_loss": 0.0,
            "specificity_gain": 0.0,
            "deescalated_n": 0,
        }
    return float(best["aec_threshold"]), best


def net_benefit(y: np.ndarray, prob: np.ndarray, pt: float) -> float:
    """의사결정곡선분석(DCA)의 순이익(net benefit) 공식으로, 임계확률 pt에서의 순이익을 계산."""
    pred = prob >= pt
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    n = len(y)
    return float(tp / n - fp / n * (pt / (1 - pt)))


def strategy_4_deescalation(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[전략4] "후반 반등" AEC 특징을 이용해, 여러 임상 민감도 목표(85/90/95%)와 허용 민감도손실
    한도(0~5%)마다 하향조정 규칙의 민감도손실/특이도이득을 표로 만들고, 임상 단독 vs 임상+AEC
    결합모델의 의사결정곡선(순이익)까지 비교 — "실제 임상 활용 시나리오에서 순이익이 있는가?"를
    보는 네 번째 접근."""
    g = data["g1090"]
    s = data["sdata"]
    feature = "tail_rebound_114_128_max_minus_60_95_min"
    fg = g["features"][feature].to_numpy(dtype=float)
    fs = s["features"][feature].to_numpy(dtype=float)
    rows = []
    for target in [0.85, 0.90, 0.95]:
        cth = threshold_for_min_sensitivity(g["y"], g["clinical_score_oof"], target)
        for max_loss in [0.00, 0.01, 0.02, 0.05]:
            ath, train_rule = choose_deescalation_threshold(g["y"], g["clinical_score_oof"], fg, cth, max_loss)
            for cohort, y, cscore, fscore in [
                ("g1090_oof", g["y"], g["clinical_score_oof"], fg),
                ("sdata_external", s["y"], s["clinical_score_external"], fs),
            ]:
                base = binary_metrics(y, cscore, cth)
                clinical_pos = cscore >= cth
                keep = clinical_pos & (fscore >= ath)
                rule = binary_metrics(y, keep.astype(float), 0.5)
                rows.append(
                    {
                        "strategy": "4_deescalation",
                        "feature": feature,
                        "clinical_target_sensitivity_g1090": target,
                        "max_allowed_sensitivity_loss_g1090": max_loss,
                        "dataset": cohort,
                        "clinical_threshold_from_g1090": cth,
                        "aec_threshold_from_g1090": ath,
                        "baseline_sensitivity": base["sensitivity"],
                        "baseline_specificity": base["specificity"],
                        "rule_sensitivity": rule["sensitivity"],
                        "rule_specificity": rule["specificity"],
                        "sensitivity_loss": base["sensitivity"] - rule["sensitivity"],
                        "specificity_gain": rule["specificity"] - base["specificity"],
                        "clinical_positive_n": int(np.sum(clinical_pos)),
                        "deescalated_n": int(np.sum(clinical_pos & (fscore < ath))),
                        "deescalated_events": int(np.sum(y[clinical_pos & (fscore < ath)])),
                        "kept_events": int(np.sum(y[keep])),
                    }
                )

    deesc = pd.DataFrame(rows)
    deesc.to_csv(OUT_DIR / "strategy4_deescalation_tradeoff.csv", index=False)

    # Decision curve for clinical-only vs clinical + fixed feature.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clinical_prob_oof = np.zeros(len(g["y"]), dtype=float)
    combined_prob_oof = np.zeros(len(g["y"]), dtype=float)
    for tr, va in skf.split(g["clinical_x"], g["y"]):
        cm = clinical_pipeline()
        cm.fit(g["clinical_x"][tr], g["y"][tr])
        clinical_prob_oof[va] = cm.predict_proba(g["clinical_x"][va])[:, 1]
        fz_tr, fz_va, _, _ = zfit_apply(fg[tr], fg[va])
        ztr = np.column_stack([g["clinical_x"][tr], fz_tr])
        zva = np.column_stack([g["clinical_x"][va], fz_va])
        pipe = clinical_pipeline()
        pipe.fit(ztr, g["y"][tr])
        combined_prob_oof[va] = pipe.predict_proba(zva)[:, 1]

    cm = clinical_pipeline()
    cm.fit(g["clinical_x"], g["y"])
    clinical_prob_ext = cm.predict_proba(s["clinical_x"])[:, 1]
    fz_g, fz_s, _, _ = zfit_apply(fg, fs)
    pipe = clinical_pipeline()
    pipe.fit(np.column_stack([g["clinical_x"], fz_g]), g["y"])
    combined_prob_ext = pipe.predict_proba(np.column_stack([s["clinical_x"], fz_s]))[:, 1]

    dca_rows = []
    for dataset, y, p_clin, p_comb in [
        ("g1090_oof", g["y"], clinical_prob_oof, combined_prob_oof),
        ("sdata_external", s["y"], clinical_prob_ext, combined_prob_ext),
    ]:
        prevalence = float(np.mean(y))
        for pt in np.linspace(0.03, 0.30, 28):
            dca_rows.append(
                {
                    "dataset": dataset,
                    "threshold_probability": float(pt),
                    "net_benefit_clinical": net_benefit(y, p_clin, float(pt)),
                    "net_benefit_clinical_plus_aec": net_benefit(y, p_comb, float(pt)),
                    "net_benefit_treat_all": prevalence - (1 - prevalence) * (pt / (1 - pt)),
                    "net_benefit_treat_none": 0.0,
                }
            )
    dca = pd.DataFrame(dca_rows)
    dca.to_csv(OUT_DIR / "strategy4_decision_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, dataset in zip(axes, ["g1090_oof", "sdata_external"]):
        sub = dca[dca["dataset"].eq(dataset)]
        ax.plot(sub["threshold_probability"], sub["net_benefit_clinical"], lw=2.2, label="Clinical", color="#4C78A8")
        ax.plot(
            sub["threshold_probability"],
            sub["net_benefit_clinical_plus_aec"],
            lw=2.2,
            label="Clinical + AEC",
            color="#F58518",
        )
        ax.plot(sub["threshold_probability"], sub["net_benefit_treat_all"], lw=1.4, ls="--", color="#777777", label="Treat all")
        ax.axhline(0, color="#555555", lw=1)
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.set_xlabel("Threshold probability")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Net benefit")
    axes[0].legend(frameon=False)
    fig.suptitle("Strategy 4: Decision Curve", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "strategy4_decision_curve.png", dpi=220)
    plt.close(fig)
    return deesc, dca


def strategy_5_measurement_surrogate(data: dict) -> pd.DataFrame:
    """[전략5] 8개 AEC 특징이 SMI뿐 아니라 TAMA/IMATA/BMI/체중 등 다른 신체계측 결과변수와도
    상관이 있는지 확인 — "AEC가 특정 진단기준의 대리측정치로 쓰일 수 있는가?"를 보는 다섯 번째 접근."""
    rows = []
    endpoints = ["SMI", "TAMA", "IMATA", "BMI", "Weight"]
    for cohort, d in data.items():
        for feature in FEATURE_SPECS:
            x = d["features"][feature].to_numpy(dtype=float)
            for endpoint in endpoints:
                if endpoint not in d["endpoints"].columns:
                    continue
                y = d["endpoints"][endpoint].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                pear = stats.pearsonr(x[mask], y[mask])
                spear = stats.spearmanr(x[mask], y[mask])
                rows.append(
                    {
                        "strategy": "5_measurement_surrogate",
                        "cohort": cohort,
                        "feature": feature,
                        "feature_definition": FEATURE_SPECS[feature],
                        "endpoint": endpoint,
                        "n": int(np.sum(mask)),
                        "pearson_r": float(pear.statistic),
                        "pearson_p": float(pear.pvalue),
                        "spearman_r": float(spear.statistic),
                        "spearman_p": float(spear.pvalue),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "strategy5_measurement_surrogate_correlations.csv", index=False)

    plot = out[(out["endpoint"].isin(["TAMA", "IMATA", "BMI", "SMI"])) & (out["feature"].eq("tail_rebound_114_128_max_minus_60_95_min"))]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    endpoints = ["TAMA", "IMATA", "BMI", "SMI"]
    xloc = np.arange(len(endpoints))
    for offset, cohort, color in [(-0.18, "g1090", "#4C78A8"), (0.18, "sdata", "#F58518")]:
        vals = [plot[(plot["endpoint"].eq(ep)) & (plot["cohort"].eq(cohort))]["pearson_r"].iloc[0] for ep in endpoints]
        ax.bar(xloc + offset, vals, width=0.36, color=color, label=cohort)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks(xloc)
    ax.set_xticklabels(endpoints)
    ax.set_ylabel("Pearson r")
    ax.set_title("Strategy 5: Tail Rebound as Measurement Surrogate", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "strategy5_tail_rebound_surrogate_correlations.png", dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 8개의 사전 지정 AEC 특징에 대해, 서로 다른 5가지 각도에서
    타당성을 검증했을 때 어느 정도까지 일관된 결론이 나오는가? — 하나의 방법에 기대지 않고
    다각도 감사(audit)를 수행):

    1. g1090/sdata를 로드하고 8개 AEC 특징(FEATURE_SPECS)과 임상점수를 준비한다.
    2. strategy_1_residual_endpoint: SMI/TAMA/IMATA/BMI를 임상변수로 예측한 잔차와 AEC 특징의 상관 검정.
    3. strategy_2_matched_contrast: 임상점수로 매칭된 쌍에서 AEC 특징·곡선 전체의 대응비교 검정.
    4. strategy_3_interactions: AEC 특징 x (주효과/여성상호작용/임상점수상호작용/둘다) 32개 모델의
       train OOF·외부 AUC 비교.
    5. strategy_4_deescalation: "후반 반등" 특징으로 임상 하향조정 규칙의 민감도-특이도 트레이드오프와
       의사결정곡선(순이익) 분석.
    6. strategy_5_measurement_surrogate: AEC 특징과 여러 신체계측 결과변수 간의 상관 확인.
    7. 5가지 전략의 결과를 각각 CSV/PNG로 저장하고, 임상 기준모델 성능과 최상위 상호작용 모델을
       요약 JSON으로 저장한 뒤, 각 전략의 핵심 결과를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    clinical_scores(data)

    s1, s1_quality = strategy_1_residual_endpoint(data)
    s2 = strategy_2_matched_contrast(data)
    s3 = strategy_3_interactions(data)
    s4, dca = strategy_4_deescalation(data)
    s5 = strategy_5_measurement_surrogate(data)

    clinical_ref = s3[s3["model"].eq("clinical_score_only")].copy()
    best_interactions = (
        s3[~s3["model"].eq("clinical_score_only")]
        .sort_values(["dataset", "auc"], ascending=[True, False])
        .groupby("dataset")
        .head(10)
    )
    summary = {
        "features": FEATURE_SPECS,
        "clinical_reference": clinical_ref.to_dict(orient="records"),
        "best_interactions_by_dataset": best_interactions.to_dict(orient="records"),
        "key_outputs": {
            "strategy1_residual_endpoint": str(OUT_DIR / "strategy1_residual_endpoint_associations.csv"),
            "strategy2_matched_contrast": str(OUT_DIR / "strategy2_matched_feature_contrasts.csv"),
            "strategy3_interactions": str(OUT_DIR / "strategy3_interaction_model_performance.csv"),
            "strategy4_deescalation": str(OUT_DIR / "strategy4_deescalation_tradeoff.csv"),
            "strategy4_decision_curve": str(OUT_DIR / "strategy4_decision_curve.csv"),
            "strategy5_surrogate": str(OUT_DIR / "strategy5_measurement_surrogate_correlations.csv"),
        },
    }
    (OUT_DIR / "five_strategy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Clinical reference")
    print(clinical_ref.to_string(index=False))
    print("\nStrategy 1: SMI residual associations")
    print(
        s1[s1["endpoint"].eq("SMI")]
        .sort_values(["cohort", "pearson_p"])
        .groupby("cohort")
        .head(8)
        .to_string(index=False)
    )
    print("\nStrategy 2: matched feature contrasts")
    print(s2.sort_values(["cohort", "paired_t_p"]).groupby("cohort").head(8).to_string(index=False))
    print("\nStrategy 3: best interaction models")
    print(best_interactions.to_string(index=False))
    print("\nStrategy 4: selected 90% sensitivity de-escalation rows")
    print(
        s4[
            s4["clinical_target_sensitivity_g1090"].eq(0.90)
            & s4["max_allowed_sensitivity_loss_g1090"].isin([0.01, 0.02, 0.05])
        ].to_string(index=False)
    )
    print("\nStrategy 5: tail rebound surrogate correlations")
    print(
        s5[s5["feature"].eq("tail_rebound_114_128_max_minus_60_95_min")]
        .sort_values(["endpoint", "cohort"])
        .to_string(index=False)
    )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
