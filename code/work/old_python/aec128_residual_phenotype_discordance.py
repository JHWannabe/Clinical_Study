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

from aec128_five_strategy_audit import FEATURE_SPECS, build_aec_features, clinical_pipeline, load_all  # noqa: E402
from aec_offset_score import sigmoid  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_residual_phenotype_discordance"
SEED = 20260629


def add_shape_features(x: np.ndarray) -> pd.DataFrame:
    """aec128_five_strategy_audit의 8개 기본 특징에, 트로프/회복/후반 구간의 평균·최소·최대·기울기·
    거칠기·위치 등 20여 개 추가 모양 특징을 더해 최종 특징 테이블을 구성."""
    x = np.asarray(x, dtype=float)
    d1 = np.diff(x, axis=1)
    d2 = np.diff(x, n=2, axis=1)

    def mean(a: int, b: int) -> np.ndarray:
        return x[:, a - 1 : b].mean(axis=1)

    def minv(a: int, b: int) -> np.ndarray:
        return x[:, a - 1 : b].min(axis=1)

    def maxv(a: int, b: int) -> np.ndarray:
        return x[:, a - 1 : b].max(axis=1)

    def rough(a: int, b: int) -> np.ndarray:
        return np.abs(d1[:, a - 1 : b - 1]).mean(axis=1)

    def slope(a: int, b: int) -> np.ndarray:
        yy = x[:, a - 1 : b]
        grid = np.linspace(-1.0, 1.0, yy.shape[1])
        denom = float(np.sum(grid**2)) or 1.0
        return (yy - yy.mean(axis=1, keepdims=True)) @ grid / denom

    eps = 1e-6
    trough_mean = mean(60, 95)
    recovery_mean = mean(96, 113)
    tail_mean = mean(114, 128)
    early_mean = mean(1, 38)
    shoulder_mean = mean(39, 59)
    trough_min = minv(60, 95)
    tail_max = maxv(114, 128)
    late_max = maxv(96, 128)

    out = build_aec_features(x)
    out["early_mean_1_38"] = early_mean
    out["shoulder_mean_39_59"] = shoulder_mean
    out["trough_mean_60_95"] = trough_mean
    out["recovery_mean_96_113"] = recovery_mean
    out["tail_mean_114_128"] = tail_mean
    out["tail_minus_trough_mean"] = tail_mean - trough_mean
    out["recovery_minus_trough_mean"] = recovery_mean - trough_mean
    out["tail_to_trough_ratio"] = tail_mean / (trough_mean + eps)
    out["trough_depth_min_below_1"] = 1.0 - trough_min
    out["trough_area_below_1_60_95"] = np.maximum(0.0, 1.0 - x[:, 59:95]).mean(axis=1)
    out["tail_area_above_1_114_128"] = np.maximum(0.0, x[:, 113:128] - 1.0).mean(axis=1)
    out["tail_above1_minus_trough_below1"] = out["tail_area_above_1_114_128"] - out["trough_area_below_1_60_95"]
    out["late_max_minus_trough_min"] = late_max - trough_min
    out["slope_60_95"] = slope(60, 95)
    out["slope_96_128"] = slope(96, 128)
    out["slope_60_128"] = slope(60, 128)
    out["roughness_60_95"] = rough(60, 95)
    out["roughness_96_128"] = rough(96, 128)
    out["curvature_abs_60_128"] = np.abs(d2[:, 58:127]).mean(axis=1)
    out["trough_min_pos_60_95"] = np.argmin(x[:, 59:95], axis=1) + 60
    out["tail_max_pos_114_128"] = np.argmax(x[:, 113:128], axis=1) + 114
    out["tailmax_minus_troughmin_pos"] = out["tail_max_pos_114_128"] - out["trough_min_pos_60_95"]
    return out


def endpoint_matrix(meta: pd.DataFrame) -> pd.DataFrame:
    """SMI/TAMA/IMATA/BMI/체중과, 그로부터 파생된 비율·로그비 등 체성분 관련 결과변수 9개를 계산."""
    out = pd.DataFrame(index=meta.index)
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    weight = pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float)
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    imata = pd.to_numeric(meta["IMATA"], errors="coerce").to_numpy(dtype=float)
    bmi = pd.to_numeric(meta["BMI"], errors="coerce").to_numpy(dtype=float) if "BMI" in meta.columns else weight / (height_m**2)
    smi = pd.to_numeric(meta["SMI"], errors="coerce").to_numpy(dtype=float) if "SMI" in meta.columns else tama / (height_m**2)

    out["SMI"] = smi
    out["TAMA"] = tama
    out["IMATA"] = imata
    out["BMI"] = bmi
    out["Weight"] = weight
    out["IMATA_fraction"] = imata / (tama + imata)
    out["muscle_quality_TAMA_fraction"] = tama / (tama + imata)
    out["TAMA_per_weight"] = tama / weight
    out["IMATA_per_weight"] = imata / weight
    out["log_TAMA_to_IMATA"] = np.log((tama + 1e-3) / (imata + 1e-3))
    return out.replace([np.inf, -np.inf], np.nan)


def linear_pipeline() -> Pipeline:
    """결측대체→표준화→선형회귀로 이어지는, 임상변수로 연속형 결과변수를 예측하는 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lin", LinearRegression()),
        ]
    )


def ridge_pipeline() -> Pipeline:
    """결측대체→표준화→RidgeCV(정규화 강도 자동탐색)로 이어지는, AEC 특징으로 잔차를 예측하는 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 25))),
        ]
    )


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """여러 배열(1차원 또는 2차원)을 받아, 모든 배열에서 유효한(유한한) 값을 가진 행만 True인 마스크를 만듦."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            mask &= np.isfinite(arr)
        else:
            mask &= np.all(np.isfinite(arr), axis=1)
    return mask


def clinical_endpoint_predictions(
    x_clin: np.ndarray,
    y: np.ndarray,
    x_ext: np.ndarray | None = None,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray | None, Pipeline]:
    """임상변수로 결과변수 y를 예측하는 선형회귀를 K-fold로 학습해 OOF 예측을, 전체 데이터로
    재학습한 모델로 외부(x_ext) 예측까지 계산."""
    ok = finite_mask(y)
    pred = np.full(len(y), np.nan)
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    for tr, va in kf.split(x_clin):
        tr_ok = tr[ok[tr]]
        va_ok = va[ok[va]]
        if len(tr_ok) < 20 or len(va_ok) == 0:
            continue
        model = linear_pipeline()
        model.fit(x_clin[tr_ok], y[tr_ok])
        pred[va_ok] = model.predict(x_clin[va_ok])
    final = linear_pipeline()
    final.fit(x_clin[ok], y[ok])
    ext_pred = None if x_ext is None else final.predict(x_ext)
    return pred, ext_pred, final


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """결측을 제외하고 결정계수(R²)를 계산."""
    mask = finite_mask(y, pred)
    yy = y[mask]
    pp = pred[mask]
    return float(1.0 - np.sum((yy - pp) ** 2) / np.sum((yy - np.mean(yy)) ** 2))


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """표본별 가중치 w를 반영한 가중 피어슨 상관계수를 계산 (표본 5개 미만이거나 분산이 0이면 NaN)."""
    mask = finite_mask(x, y, w) & (w > 0)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if len(x) < 5:
        return np.nan
    w = w / np.sum(w)
    mx = np.sum(w * x)
    my = np.sum(w * y)
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy))


def gate_weights(p: np.ndarray) -> dict[str, np.ndarray]:
    """임상 확률 p로부터 4가지 threshold-free 게이트 가중치(균등/불확실성중간/고위험쪽불확실성/고위험단조증가)를 계산."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "all_unweighted": np.ones_like(p),
        "uncertainty_mid_4p1mp": 4.0 * p * (1.0 - p),
        "soft_high_uncertainty_6p75p2": 6.75 * p**2 * (1.0 - p),
        "soft_high_risk_p": p,
    }


def add_data_fields(data: dict) -> None:
    """각 코호트 데이터에 확장 모양 특징·체성분 결과변수·여성 여부를 계산해 채워 넣음 (in-place)."""
    for d in data.values():
        d["features"] = add_shape_features(d["x"])
        d["endpoints"] = endpoint_matrix(d["meta"])
        sex = d["meta"]["PatientSex"].astype(str).str.upper().to_numpy()
        d["female"] = (sex == "F").astype(float)


def clinical_low_smi_scores(data: dict) -> None:
    """g1090 5-fold OOF 임상점수와, 전체 g1090으로 학습한 모델의 sdata 외부 점수를 계산해 data 딕셔너리에 채워 넣음 (in-place)."""
    g = data["g1090"]
    s = data["sdata"]
    y = g["y"].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(g["clinical_x"], y):
        model = clinical_pipeline()
        model.fit(g["clinical_x"][tr], y[tr])
        oof[va] = model.decision_function(g["clinical_x"][va])
    final = clinical_pipeline()
    final.fit(g["clinical_x"], y)
    g["clinical_score_oof"] = oof
    s["clinical_score_external"] = final.decision_function(s["clinical_x"])


def endpoint_residual_correlations(data: dict) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """[핵심 질문] 9개 체성분 결과변수(SMI, TAMA 등)를 임상변수로 예측한 "잔차"(임상으로 설명 안 되는
    부분)를 구하고, 전체/여성/남성/BMI 상하위/임상 불확실 구간 등 여러 하위그룹에서 모든 AEC 특징이
    그 잔차와 상관이 있는지(Pearson/Spearman) 검정 — "AEC가 잡아내는 것이 임상변수로 설명되지
    않는 진짜 체성분 정보인가?"를 확인."""
    endpoints = [
        "SMI",
        "TAMA",
        "IMATA",
        "BMI",
        "IMATA_fraction",
        "muscle_quality_TAMA_fraction",
        "TAMA_per_weight",
        "IMATA_per_weight",
        "log_TAMA_to_IMATA",
    ]
    feature_names = data["g1090"]["features"].columns.tolist()
    rows = []
    residuals: dict[str, dict[str, np.ndarray]] = {cohort: {} for cohort in data}

    for endpoint in endpoints:
        g = data["g1090"]
        s = data["sdata"]
        y_g = g["endpoints"][endpoint].to_numpy(dtype=float)
        y_s = s["endpoints"][endpoint].to_numpy(dtype=float)
        pred_g, pred_s, _ = clinical_endpoint_predictions(g["clinical_x"], y_g, s["clinical_x"])
        resid_g = y_g - pred_g
        resid_s = y_s - pred_s
        residuals["g1090"][endpoint] = resid_g
        residuals["sdata"][endpoint] = resid_s

        for cohort, d, resid in [("g1090", g, resid_g), ("sdata", s, resid_s)]:
            bmi = d["endpoints"]["BMI"].to_numpy(dtype=float)
            female = d["female"].astype(bool)
            p = sigmoid(d["clinical_score_oof"] if cohort == "g1090" else d["clinical_score_external"])
            groups = {
                "all": np.ones(len(resid), dtype=bool),
                "female": female,
                "male": ~female,
                "bmi_high_half": bmi >= np.nanmedian(bmi),
                "bmi_low_half": bmi < np.nanmedian(bmi),
                "clinical_uncertain_0.2_0.8": (p >= 0.2) & (p <= 0.8),
                "clinical_high_ge_0.5": p >= 0.5,
            }
            for group_name, gm in groups.items():
                for feature in feature_names:
                    x = d["features"][feature].to_numpy(dtype=float)
                    mask = gm & finite_mask(x, resid)
                    if int(np.sum(mask)) < 12 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(resid[mask]) < 1e-12:
                        continue
                    pear = stats.pearsonr(x[mask], resid[mask])
                    spear = stats.spearmanr(x[mask], resid[mask])
                    rows.append(
                        {
                            "cohort": cohort,
                            "endpoint": endpoint,
                            "group": group_name,
                            "feature": feature,
                            "n": int(np.sum(mask)),
                            "pearson_r_feature_vs_clinical_residual": float(pear.statistic),
                            "pearson_p": float(pear.pvalue),
                            "spearman_r": float(spear.statistic),
                            "spearman_p": float(spear.pvalue),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "endpoint_residual_feature_correlations_by_group.csv", index=False)
    return out, residuals


def endpoint_residual_ridge(data: dict) -> pd.DataFrame:
    """임상변수만 쓴 예측(선형회귀) 대비, 임상 잔차를 AEC 특징(3가지 그룹: 단일특징/핵심5개/전체)으로
    RidgeCV 예측해 더했을 때 R²가 얼마나 개선되는지 9개 결과변수 x 3개 특징그룹에 대해 계산
    (g1090 OOF와 sdata 외부 양쪽 모두)."""
    endpoints = [
        "SMI",
        "TAMA",
        "IMATA",
        "BMI",
        "IMATA_fraction",
        "muscle_quality_TAMA_fraction",
        "TAMA_per_weight",
        "IMATA_per_weight",
        "log_TAMA_to_IMATA",
    ]
    feature_groups = {
        "tail_rebound_only": ["tail_rebound_114_128_max_minus_60_95_min"],
        "trough_to_tail_core": [
            "tail_rebound_114_128_max_minus_60_95_min",
            "tail_minus_trough_mean",
            "tail_area_above_1_114_128",
            "trough_area_below_1_60_95",
            "slope_96_128",
        ],
        "all_interpretable_shape": data["g1090"]["features"].columns.tolist(),
    }
    rows = []
    g = data["g1090"]
    s = data["sdata"]
    for endpoint in endpoints:
        y_g = g["endpoints"][endpoint].to_numpy(dtype=float)
        y_s = s["endpoints"][endpoint].to_numpy(dtype=float)
        for group_name, features in feature_groups.items():
            pred_clin_oof = np.full(len(y_g), np.nan)
            pred_comb_oof = np.full(len(y_g), np.nan)
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            for tr, va in kf.split(g["clinical_x"]):
                ok_tr = finite_mask(y_g[tr])
                tr_ok = tr[ok_tr]
                ok_va = finite_mask(y_g[va])
                va_ok = va[ok_va]
                if len(tr_ok) < 30 or len(va_ok) == 0:
                    continue
                cm = linear_pipeline()
                cm.fit(g["clinical_x"][tr_ok], y_g[tr_ok])
                c_tr = cm.predict(g["clinical_x"][tr_ok])
                c_va = cm.predict(g["clinical_x"][va_ok])
                resid_tr = y_g[tr_ok] - c_tr
                rm = ridge_pipeline()
                rm.fit(g["features"].loc[tr_ok, features].to_numpy(dtype=float), resid_tr)
                resid_va = rm.predict(g["features"].loc[va_ok, features].to_numpy(dtype=float))
                pred_clin_oof[va_ok] = c_va
                pred_comb_oof[va_ok] = c_va + resid_va

            ok_g = finite_mask(y_g)
            cm = linear_pipeline()
            cm.fit(g["clinical_x"][ok_g], y_g[ok_g])
            c_g_full = cm.predict(g["clinical_x"])
            c_s = cm.predict(s["clinical_x"])
            resid_g_full = y_g - c_g_full
            rm = ridge_pipeline()
            rm.fit(g["features"].loc[ok_g, features].to_numpy(dtype=float), resid_g_full[ok_g])
            combined_s = c_s + rm.predict(s["features"][features].to_numpy(dtype=float))

            rows.append(
                {
                    "endpoint": endpoint,
                    "feature_group": group_name,
                    "n_features": len(features),
                    "g1090_clinical_oof_r2": r2(y_g, pred_clin_oof),
                    "g1090_clinical_plus_aec_oof_r2": r2(y_g, pred_comb_oof),
                    "g1090_delta_r2": r2(y_g, pred_comb_oof) - r2(y_g, pred_clin_oof),
                    "sdata_clinical_external_r2": r2(y_s, c_s),
                    "sdata_clinical_plus_aec_external_r2": r2(y_s, combined_s),
                    "sdata_delta_r2": r2(y_s, combined_s) - r2(y_s, c_s),
                    "selected_alpha_full_g1090": float(rm.named_steps["ridge"].alpha_),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "endpoint_residual_ridge_r2.csv", index=False)
    return out


def binary_clinical_residual_analysis(data: dict) -> pd.DataFrame:
    """저근감소증 이진 라벨과 임상 예측확률의 차이(y - p_clinical, 이진 버전의 "잔차")가 AEC 특징과
    상관있는지, 여러 하위그룹 x 4개 게이트 가중치 조합으로 비가중/가중 피어슨 상관을 모두 계산."""
    rows = []
    for cohort, score_key in [("g1090", "clinical_score_oof"), ("sdata", "clinical_score_external")]:
        d = data[cohort]
        y = d["y"].astype(int)
        p = sigmoid(d[score_key])
        resid = y - p
        female = d["female"].astype(bool)
        bmi = d["endpoints"]["BMI"].to_numpy(dtype=float)
        group_masks = {
            "all": np.ones(len(y), dtype=bool),
            "female": female,
            "male": ~female,
            "bmi_high_half": bmi >= np.nanmedian(bmi),
            "bmi_low_half": bmi < np.nanmedian(bmi),
        }
        for group_name, gm in group_masks.items():
            for gate_name, w_base in gate_weights(p).items():
                w = w_base * gm.astype(float)
                for feature in d["features"].columns:
                    x = d["features"][feature].to_numpy(dtype=float)
                    mask = gm & finite_mask(x, resid)
                    if int(np.sum(mask)) < 12:
                        continue
                    pear = stats.pearsonr(x[mask], resid[mask]) if np.nanstd(x[mask]) > 0 else (np.nan, np.nan)
                    rows.append(
                        {
                            "cohort": cohort,
                            "group": group_name,
                            "gate_weight": gate_name,
                            "feature": feature,
                            "n": int(np.sum(mask)),
                            "unweighted_pearson_r_feature_vs_y_minus_pclinical": float(pear.statistic)
                            if hasattr(pear, "statistic")
                            else np.nan,
                            "unweighted_pearson_p": float(pear.pvalue) if hasattr(pear, "pvalue") else np.nan,
                            "weighted_pearson_r_feature_vs_y_minus_pclinical": weighted_pearson(x, resid, w),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "binary_low_smi_clinical_residual_aec_associations.csv", index=False)
    return out


def logistic_discordance_models(data: dict) -> pd.DataFrame:
    """5개 대표 AEC 특징 x (여성상호작용 유무) x (고위험쪽 소프트게이트 유무) = 20개 로지스틱 모델을
    임상점수에 결합해 학습하고, 임상 단독 대비 AUC/로그손실/Brier 개선을 train OOF·외부 양쪽에서 비교
    (하드 임계값 없이 AEC가 임상확률의 잔차 방향을 예측하는지 확인)."""
    # Tests whether AEC predicts clinical probability residual direction without forcing a hard threshold.
    g = data["g1090"]
    s = data["sdata"]
    features = [
        "tail_rebound_114_128_max_minus_60_95_min",
        "tail_minus_trough_mean",
        "late_rebound_91_128_max_minus_60_95_min",
        "roughness_75_90",
        "IMATA_surrogate_combo" if False else "trough_range_60_95",
    ]
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clinical_oof = g["clinical_score_oof"]
    clinical_ext = s["clinical_score_external"]
    base_oof = {
        "auc": float(roc_auc_score(g["y"], clinical_oof)),
        "ap": float(average_precision_score(g["y"], clinical_oof)),
        "log_loss": float(log_loss(g["y"], sigmoid(clinical_oof))),
        "brier": float(brier_score_loss(g["y"], sigmoid(clinical_oof))),
    }
    base_ext = {
        "auc": float(roc_auc_score(s["y"], clinical_ext)),
        "ap": float(average_precision_score(s["y"], clinical_ext)),
        "log_loss": float(log_loss(s["y"], sigmoid(clinical_ext))),
        "brier": float(brier_score_loss(s["y"], sigmoid(clinical_ext))),
    }
    for feature in features:
        for with_female in [False, True]:
            for with_gate in [False, True]:
                model_name = feature + ("__female" if with_female else "") + ("__soft_high_gate" if with_gate else "")
                oof = np.zeros(len(g["y"]), dtype=float)
                for tr, va in skf.split(g["clinical_x"], g["y"]):
                    cm = clinical_pipeline()
                    cm.fit(g["clinical_x"][tr], g["y"][tr])
                    c_tr = cm.decision_function(g["clinical_x"][tr])
                    c_va = cm.decision_function(g["clinical_x"][va])
                    f_tr = g["features"].loc[tr, feature].to_numpy(dtype=float)
                    f_va = g["features"].loc[va, feature].to_numpy(dtype=float)
                    fz_tr, fz_va = standardize_1d(f_tr, f_va)
                    cols_tr = [c_tr, fz_tr]
                    cols_va = [c_va, fz_va]
                    if with_gate:
                        cols_tr.append(fz_tr * (6.75 * sigmoid(c_tr) ** 2 * (1 - sigmoid(c_tr))))
                        cols_va.append(fz_va * (6.75 * sigmoid(c_va) ** 2 * (1 - sigmoid(c_va))))
                    if with_female:
                        cols_tr.append(fz_tr * g["female"][tr])
                        cols_va.append(fz_va * g["female"][va])
                    pipe = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED)),
                        ]
                    )
                    pipe.fit(np.column_stack(cols_tr), g["y"][tr])
                    oof[va] = pipe.decision_function(np.column_stack(cols_va))

                cm = clinical_pipeline()
                cm.fit(g["clinical_x"], g["y"])
                c_tr = cm.decision_function(g["clinical_x"])
                c_te = cm.decision_function(s["clinical_x"])
                fz_g, fz_s = standardize_1d(
                    g["features"][feature].to_numpy(dtype=float), s["features"][feature].to_numpy(dtype=float)
                )
                cols_g = [c_tr, fz_g]
                cols_s = [c_te, fz_s]
                if with_gate:
                    cols_g.append(fz_g * (6.75 * sigmoid(c_tr) ** 2 * (1 - sigmoid(c_tr))))
                    cols_s.append(fz_s * (6.75 * sigmoid(c_te) ** 2 * (1 - sigmoid(c_te))))
                if with_female:
                    cols_g.append(fz_g * g["female"])
                    cols_s.append(fz_s * s["female"])
                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("logit", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED)),
                    ]
                )
                pipe.fit(np.column_stack(cols_g), g["y"])
                ext = pipe.decision_function(np.column_stack(cols_s))
                for dataset, y, score, base in [
                    ("g1090_oof", g["y"], oof, base_oof),
                    ("sdata_external", s["y"], ext, base_ext),
                ]:
                    prob = sigmoid(score)
                    auc = float(roc_auc_score(y, score))
                    ap = float(average_precision_score(y, score))
                    ll = float(log_loss(y, prob))
                    br = float(brier_score_loss(y, prob))
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "feature": feature,
                            "female_interaction": with_female,
                            "soft_high_gate": with_gate,
                            "auc": auc,
                            "average_precision": ap,
                            "log_loss": ll,
                            "brier": br,
                            "delta_auc": auc - base["auc"],
                            "delta_average_precision": ap - base["ap"],
                            "log_loss_reduction": base["log_loss"] - ll,
                            "brier_reduction": base["brier"] - br,
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "binary_low_smi_logistic_discordance_models.csv", index=False)
    return out


def standardize_1d(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 1차원 train·test 배열을 함께 표준화."""
    mu = float(np.nanmean(xtr))
    sd = float(np.nanstd(xtr))
    if not np.isfinite(sd) or sd < 1e-10:
        sd = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def common_hit_summary(corr: pd.DataFrame) -> pd.DataFrame:
    """전체표본(group="all") 기준으로 g1090과 sdata의 상관계수를 나란히 놓고, 두 코호트 모두 방향이
    같고 p<0.05인 "강건한 공통 히트"를 판별력 순으로 정렬."""
    all_group = corr[corr["group"].eq("all")].copy()
    g = all_group[all_group["cohort"].eq("g1090")]
    s = all_group[all_group["cohort"].eq("sdata")]
    merged = g.merge(
        s,
        on=["endpoint", "group", "feature"],
        suffixes=("_g1090", "_sdata"),
    )
    merged["same_direction"] = np.sign(merged["pearson_r_feature_vs_clinical_residual_g1090"]) == np.sign(
        merged["pearson_r_feature_vs_clinical_residual_sdata"]
    )
    merged["both_p_lt_0.05"] = (merged["pearson_p_g1090"] < 0.05) & (merged["pearson_p_sdata"] < 0.05)
    merged["min_abs_r"] = np.minimum(
        np.abs(merged["pearson_r_feature_vs_clinical_residual_g1090"]),
        np.abs(merged["pearson_r_feature_vs_clinical_residual_sdata"]),
    )
    merged["mean_abs_r"] = (
        np.abs(merged["pearson_r_feature_vs_clinical_residual_g1090"])
        + np.abs(merged["pearson_r_feature_vs_clinical_residual_sdata"])
    ) / 2.0
    out = merged.sort_values(["both_p_lt_0.05", "same_direction", "min_abs_r"], ascending=[False, False, False])
    out.to_csv(OUT_DIR / "common_endpoint_residual_feature_hits.csv", index=False)
    return out


def plot_residual_heatmap(hits: pd.DataFrame) -> None:
    """강건한 공통 히트 상위 특징들 x 7개 결과변수의 상관계수를 g1090/sdata 나란히 히트맵으로 그려 PNG로 저장."""
    endpoints = ["SMI", "TAMA", "IMATA", "IMATA_fraction", "TAMA_per_weight", "IMATA_per_weight", "log_TAMA_to_IMATA"]
    top_features = (
        hits[hits["same_direction"] & hits["both_p_lt_0.05"]]
        .sort_values("min_abs_r", ascending=False)["feature"]
        .drop_duplicates()
        .head(14)
        .tolist()
    )
    if len(top_features) < 8:
        top_features = hits.sort_values("min_abs_r", ascending=False)["feature"].drop_duplicates().head(14).tolist()
    mat_g = np.full((len(endpoints), len(top_features)), np.nan)
    mat_s = np.full_like(mat_g, np.nan)
    for i, ep in enumerate(endpoints):
        for j, feat in enumerate(top_features):
            row = hits[(hits["endpoint"].eq(ep)) & (hits["feature"].eq(feat))]
            if not row.empty:
                mat_g[i, j] = row["pearson_r_feature_vs_clinical_residual_g1090"].iloc[0]
                mat_s[i, j] = row["pearson_r_feature_vs_clinical_residual_sdata"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), sharey=True)
    vmax = np.nanmax(np.abs(np.r_[mat_g.ravel(), mat_s.ravel()]))
    vmax = max(vmax, 0.05)
    for ax, mat, title in [(axes[0], mat_g, "g1090 OOF residual"), (axes[1], mat_s, "sdata external residual")]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(top_features)))
        ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(endpoints)))
        ax.set_yticklabels(endpoints)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Pearson r: AEC feature vs clinical residual endpoint")
    fig.suptitle("AEC morphology vs clinical-residual body-composition phenotypes", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "endpoint_residual_feature_heatmap.png", dpi=220)
    plt.close(fig)


def plot_r2_delta(ridge: pd.DataFrame) -> None:
    """"핵심 5특징" 그룹의 결과변수별 delta R²(임상 대비 AEC 추가 이득)를 g1090/sdata 나란히 막대그래프로 그려 PNG로 저장."""
    plot = ridge[ridge["feature_group"].eq("trough_to_tail_core")].copy()
    order = plot.sort_values("sdata_delta_r2", ascending=False)["endpoint"].tolist()
    plot["endpoint"] = pd.Categorical(plot["endpoint"], categories=order, ordered=True)
    plot = plot.sort_values("endpoint")
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 0.18, plot["g1090_delta_r2"], width=0.36, label="g1090 OOF", color="#4C78A8")
    ax.bar(x + 0.18, plot["sdata_delta_r2"], width=0.36, label="sdata external", color="#F58518")
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["endpoint"], rotation=35, ha="right")
    ax.set_ylabel("Delta R2 vs clinical-only")
    ax.set_title("Does AEC explain continuous clinical residual phenotype?", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "endpoint_residual_ridge_delta_r2.png", dpi=220)
    plt.close(fig)


def plot_binary_residual(binary: pd.DataFrame) -> None:
    """전체표본 x (비가중/고위험게이트) 조건에서 이진 잔차와 가장 강하게 연관된 상위 14개 특징을 g1090/sdata 나란히 막대그래프로 그려 PNG로 저장."""
    plot = binary[
        binary["group"].eq("all")
        & binary["gate_weight"].isin(["all_unweighted", "soft_high_uncertainty_6p75p2"])
    ].copy()
    g = plot[plot["cohort"].eq("g1090")]
    s = plot[plot["cohort"].eq("sdata")]
    merged = g.merge(s, on=["group", "gate_weight", "feature"], suffixes=("_g1090", "_sdata"))
    merged["min_abs_weighted_r"] = np.minimum(
        np.abs(merged["weighted_pearson_r_feature_vs_y_minus_pclinical_g1090"]),
        np.abs(merged["weighted_pearson_r_feature_vs_y_minus_pclinical_sdata"]),
    )
    top = merged.sort_values("min_abs_weighted_r", ascending=False).head(14)
    labels = top["feature"] + "\n" + top["gate_weight"].str.replace("_", " ")
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    ax.bar(x - 0.18, top["weighted_pearson_r_feature_vs_y_minus_pclinical_g1090"], width=0.36, label="g1090", color="#4C78A8")
    ax.bar(x + 0.18, top["weighted_pearson_r_feature_vs_y_minus_pclinical_sdata"], width=0.36, label="sdata", color="#F58518")
    ax.axhline(0, color="#555555", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weighted Pearson r with y - p(clinical)")
    ax.set_title("AEC vs clinical probability residual for low SMI", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "binary_clinical_residual_aec_top_features.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (6/29 작업의 최종 집대성 — 질문: AEC 형태 특징이 임상변수로는
    설명되지 않는 "진짜" 체성분 표현형 정보(잔차)를 담고 있는가? 연속형 결과변수와 이진 저근감소증
    라벨 양쪽에서, 그리고 하위그룹/게이트별로 일관되게 나타나는가?):

    1. g1090/sdata를 로드하고, add_data_fields로 확장 모양 특징(20여개)·9개 체성분 결과변수·
       여성 여부를 계산하며, clinical_low_smi_scores로 임상 저근감소증 점수(OOF/외부)도 준비한다.
    2. endpoint_residual_correlations: 9개 연속형 결과변수(SMI, TAMA 등)를 임상변수로 예측한 잔차와,
       모든 AEC 특징의 상관을 전체/여성/남성/BMI상하위/임상불확실구간 등 여러 하위그룹에서 검정.
    3. endpoint_residual_ridge: 같은 잔차를 이번엔 AEC 특징 3개 그룹(단일/핵심5개/전체)으로 RidgeCV
       예측해, 임상 단독 대비 R² 개선(delta R²)을 g1090 OOF와 sdata 외부 양쪽에서 계산.
    4. binary_clinical_residual_analysis: 이진 저근감소증 라벨의 "잔차"(y - 임상예측확률)와 AEC
       특징의 상관을, 4개 threshold-free 게이트 가중치를 적용해 하위그룹별로 검정.
    5. logistic_discordance_models: 5개 대표 특징 x 여성상호작용 x 소프트게이트 조합(20개) 로지스틱
       모델의 임상 단독 대비 AUC/로그손실 개선을 train OOF·외부 양쪽에서 비교.
    6. common_hit_summary로 두 코호트 모두에서 방향이 같고 유의한(p<0.05) "강건한 공통 히트"를 추려내고,
       결과변수x특징 히트맵, delta R² 막대그래프, 이진 잔차 상위특징 막대그래프를 각각 PNG로 저장.
    7. 강건한 공통 히트 상위 80개, ridge R² 전체 결과, 이진 로지스틱 모델 상위 20개를 CSV로 저장하고,
       방법론 설명과 함께 종합 요약을 JSON으로 저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    add_data_fields(data)
    clinical_low_smi_scores(data)

    corr, residuals = endpoint_residual_correlations(data)
    ridge = endpoint_residual_ridge(data)
    binary = binary_clinical_residual_analysis(data)
    logit = logistic_discordance_models(data)
    hits = common_hit_summary(corr)

    plot_residual_heatmap(hits)
    plot_r2_delta(ridge)
    plot_binary_residual(binary)

    # Compact interpretation tables.
    robust_hits = hits[hits["same_direction"] & hits["both_p_lt_0.05"]].copy()
    robust_hits = robust_hits.sort_values("min_abs_r", ascending=False)
    robust_hits.head(80).to_csv(OUT_DIR / "robust_common_residual_hits_top80.csv", index=False)

    paired_logit = logit.pivot_table(
        index=["model", "feature", "female_interaction", "soft_high_gate"],
        columns="dataset",
        values=["auc", "delta_auc", "log_loss_reduction", "brier_reduction"],
        aggfunc="first",
    )
    paired_logit.columns = [f"{m}_{d}" for m, d in paired_logit.columns]
    paired_logit = paired_logit.reset_index().sort_values(
        ["delta_auc_g1090_oof", "delta_auc_sdata_external"], ascending=[False, False]
    )
    paired_logit.to_csv(OUT_DIR / "binary_logistic_discordance_models_paired.csv", index=False)

    summary = {
        "method": {
            "core_question": "Does AEC explain clinical residual body-composition phenotypes or clinical low-SMI probability residuals?",
            "clinical_variables": ["age", "sex", "height", "weight"],
            "endpoint_residual": "observed endpoint - clinical-predicted endpoint",
            "binary_residual": "low_smi_label - clinical_predicted_probability",
            "feature_strategy": "interpretable normalized AEC_128 trough-to-tail morphology features",
        },
        "top_robust_common_residual_hits": robust_hits.head(30).to_dict(orient="records"),
        "endpoint_ridge_r2": ridge.to_dict(orient="records"),
        "top_binary_logistic_models": paired_logit.head(20).to_dict(orient="records"),
        "outputs": {
            "endpoint_correlations": str(OUT_DIR / "endpoint_residual_feature_correlations_by_group.csv"),
            "common_hits": str(OUT_DIR / "common_endpoint_residual_feature_hits.csv"),
            "robust_hits": str(OUT_DIR / "robust_common_residual_hits_top80.csv"),
            "ridge_r2": str(OUT_DIR / "endpoint_residual_ridge_r2.csv"),
            "binary_residual": str(OUT_DIR / "binary_low_smi_clinical_residual_aec_associations.csv"),
            "binary_logit": str(OUT_DIR / "binary_logistic_discordance_models_paired.csv"),
            "heatmap": str(OUT_DIR / "endpoint_residual_feature_heatmap.png"),
            "r2_plot": str(OUT_DIR / "endpoint_residual_ridge_delta_r2.png"),
            "binary_residual_plot": str(OUT_DIR / "binary_clinical_residual_aec_top_features.png"),
        },
    }
    (OUT_DIR / "residual_phenotype_discordance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nTop robust endpoint residual hits")
    print(
        robust_hits[
            [
                "endpoint",
                "feature",
                "pearson_r_feature_vs_clinical_residual_g1090",
                "pearson_p_g1090",
                "pearson_r_feature_vs_clinical_residual_sdata",
                "pearson_p_sdata",
                "min_abs_r",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    print("\nEndpoint residual ridge delta R2")
    print(
        ridge.sort_values(["sdata_delta_r2", "g1090_delta_r2"], ascending=False)[
            [
                "endpoint",
                "feature_group",
                "g1090_delta_r2",
                "sdata_delta_r2",
                "g1090_clinical_oof_r2",
                "sdata_clinical_external_r2",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    print("\nBinary logistic discordance top")
    print(
        paired_logit[
            [
                "model",
                "delta_auc_g1090_oof",
                "delta_auc_sdata_external",
                "log_loss_reduction_g1090_oof",
                "log_loss_reduction_sdata_external",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
