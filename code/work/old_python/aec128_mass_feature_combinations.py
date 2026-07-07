from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec_offset_score import clinical_raw, sigmoid  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_signal_audit"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_mass_feature_combinations"
SEED = 20260629


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """분모가 0에 가까우면 부호를 살린 작은 값으로 대체해 0 나눗셈을 방지하는 나눗셈."""
    return num / np.where(np.abs(den) < eps, eps * np.sign(den + eps), den)


def pseg(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """1-based 구간 [a, b]의 평균값(행마다)을 계산."""
    return x[:, a - 1 : b].mean(axis=1)


def add_window_stats(
    out: dict[str, np.ndarray],
    signal: np.ndarray,
    prefix: str,
    lengths: list[int],
    step: int,
    stats_list: tuple[str, ...] = ("mean",),
) -> None:
    """여러 길이(lengths)의 겹치는 슬라이딩 윈도우에 대해 평균/표준편차/최소/최대 중 지정된 통계량들을 out 딕셔너리에 채워 넣음 (in-place)."""
    n = signal.shape[1]
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            end0 = start0 + length
            block = signal[:, start0:end0]
            tag = f"{prefix}_{start0 + 1:03d}_{end0:03d}"
            if "mean" in stats_list:
                out[f"{tag}_mean"] = block.mean(axis=1)
            if "sd" in stats_list:
                out[f"{tag}_sd"] = block.std(axis=1)
            if "min" in stats_list:
                out[f"{tag}_min"] = block.min(axis=1)
            if "max" in stats_list:
                out[f"{tag}_max"] = block.max(axis=1)


def add_haar_edges(out: dict[str, np.ndarray], signal: np.ndarray, prefix: str, blocks: list[int], step: int) -> None:
    """여러 블록 크기(blocks)에 대해 인접한 좌/우 블록 평균 차이(Haar 엣지)를 out 딕셔너리에 채워 넣음 (in-place)."""
    n = signal.shape[1]
    for block in blocks:
        length = 2 * block
        for start0 in range(0, n - length + 1, step):
            mid0 = start0 + block
            end0 = start0 + length
            out[f"{prefix}_haar_b{block:02d}_{start0 + 1:03d}_{end0:03d}"] = (
                signal[:, mid0:end0].mean(axis=1) - signal[:, start0:mid0].mean(axis=1)
            )


def longest_run(mask: np.ndarray) -> np.ndarray:
    """각 행에서 True가 가장 길게 연속되는 구간의 길이를 반환."""
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        best = cur = 0
        for val in row:
            if val:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[i] = best
    return out


def run_count(mask: np.ndarray) -> np.ndarray:
    """각 행에서 True가 연속되는 구간(run)이 몇 번 나타나는지 개수를 세어 반환."""
    out = np.zeros(mask.shape[0], dtype=float)
    for i, row in enumerate(mask):
        runs = 0
        in_run = False
        for val in row:
            if val and not in_run:
                runs += 1
                in_run = True
            elif not val:
                in_run = False
        out[i] = runs
    return out


def autocorr_features(x: np.ndarray, prefix: str, lags: list[int]) -> dict[str, np.ndarray]:
    """여러 지연(lags)에서의 자기상관 값을 계산해 딕셔너리로 반환."""
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(centered**2, axis=1)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    out = {}
    for lag in lags:
        out[f"{prefix}_autocorr_lag_{lag:02d}"] = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / denom
    return out


def build_feature_bank(x_norm: np.ndarray) -> pd.DataFrame:
    """다중스케일 구간 레벨/기울기/곡률, Haar 엣지, 고정 해부학적 구간 간의 모든 쌍별 대비·비율,
    극값·기하, 거칠기·평탄구간, 전역 통계, DCT/FFT/자기상관까지 총망라한 초대형(수천 차원)
    AEC_128 특징 은행을 계산 (라벨 미사용, "특징을 최대한 많이" 만드는 것이 목적)."""
    x = np.asarray(x_norm, dtype=float)
    logx = np.log(np.clip(x, 1e-6, None))
    d1 = np.diff(x, axis=1)
    dlog = np.diff(logx, axis=1)
    d2 = np.diff(d1, axis=1)
    coeff = dct(logx, type=2, norm="ortho", axis=1)
    fft_mag = np.abs(np.fft.rfft(logx - logx.mean(axis=1, keepdims=True), axis=1))

    rows: dict[str, np.ndarray] = {}

    # Dense multiscale levels and slopes.
    add_window_stats(rows, x, "norm_level", [4, 8, 12, 16, 24, 32, 48, 64], step=4, stats_list=("mean",))
    add_window_stats(rows, logx, "log_level", [4, 8, 12, 16, 24, 32, 48, 64], step=4, stats_list=("mean",))
    add_window_stats(rows, d1, "norm_slope", [4, 8, 12, 16, 24, 32], step=3, stats_list=("mean", "sd"))
    add_window_stats(rows, dlog, "log_slope", [4, 8, 12, 16, 24, 32], step=3, stats_list=("mean", "sd"))
    add_window_stats(rows, d2, "norm_curv", [4, 8, 12, 16, 24], step=3, stats_list=("mean", "sd", "min", "max"))
    add_haar_edges(rows, x, "norm", [2, 4, 8, 12, 16, 24], step=2)
    add_haar_edges(rows, logx, "log", [2, 4, 8, 12, 16, 24], step=2)

    # Fixed interpretable atlas segments.
    segs = {
        "early_001_032": (1, 32),
        "earlymid_001_058": (1, 58),
        "pretrough_050_065": (50, 65),
        "transition_058_074": (58, 74),
        "trough_060_095": (60, 95),
        "troughcore_070_085": (70, 85),
        "recover_075_090": (75, 90),
        "recover_081_113": (81, 113),
        "late_091_113": (91, 113),
        "tail_114_128": (114, 128),
        "tail_120_128": (120, 128),
    }
    level_vals = {name: pseg(x, a, b) for name, (a, b) in segs.items()}
    log_vals = {f"log_{name}": pseg(logx, a, b) for name, (a, b) in segs.items()}
    rows.update({f"level_{k}": v for k, v in level_vals.items()})
    rows.update(log_vals)
    keys = list(level_vals.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            rows[f"contrast_{k2}_minus_{k1}"] = level_vals[k2] - level_vals[k1]
            rows[f"ratio_{k2}_over_{k1}"] = safe_div(level_vals[k2], level_vals[k1])
    log_keys = list(log_vals.keys())
    for i, k1 in enumerate(log_keys):
        for k2 in log_keys[i + 1 :]:
            rows[f"logcontrast_{k2}_minus_{k1}"] = log_vals[k2] - log_vals[k1]

    # Extrema and geometry.
    ranges = {
        "early_001_040": (1, 40),
        "mid_040_080": (40, 80),
        "trough_060_095": (60, 95),
        "late_091_128": (91, 128),
        "tail_114_128": (114, 128),
    }
    extrema = {}
    for name, (a, b) in ranges.items():
        block = x[:, a - 1 : b]
        extrema[f"{name}_min"] = block.min(axis=1)
        extrema[f"{name}_max"] = block.max(axis=1)
        extrema[f"{name}_argmin"] = np.argmin(block, axis=1).astype(float) + a
        extrema[f"{name}_argmax"] = np.argmax(block, axis=1).astype(float) + a
        extrema[f"{name}_range"] = block.max(axis=1) - block.min(axis=1)
    rows.update(extrema)
    rows["late_rebound_height_max91_128_minus_min60_95"] = extrema["late_091_128_max"] - extrema["trough_060_095_min"]
    rows["tail_rebound_height_max114_128_minus_min60_95"] = extrema["tail_114_128_max"] - extrema["trough_060_095_min"]
    rows["early_to_trough_drop_max1_40_minus_min60_95"] = extrema["early_001_040_max"] - extrema["trough_060_095_min"]
    rows["tailpeak_minus_earlypeak"] = extrema["late_091_128_max"] - extrema["early_001_040_max"]
    rows["trough_to_latepeak_distance"] = extrema["late_091_128_argmax"] - extrema["trough_060_095_argmin"]
    rows["earlypeak_to_trough_distance"] = extrema["trough_060_095_argmin"] - extrema["early_001_040_argmax"]
    rows["recovery_fraction_late"] = safe_div(
        extrema["late_091_128_max"] - extrema["trough_060_095_min"],
        extrema["early_001_040_max"] - extrema["trough_060_095_min"],
    )

    # Roughness, plateau, sign-change.
    for name, (a, b) in {
        "early_001_032": (1, 32),
        "transition_058_074": (58, 74),
        "recover_075_090": (75, 90),
        "recover_081_113": (81, 113),
        "tail_114_127": (114, 127),
        "late_081_127": (81, 127),
    }.items():
        dd = d1[:, a - 1 : b]
        rows[f"rough_abs_d1_{name}"] = np.abs(dd).mean(axis=1)
        rows[f"rough_sd_d1_{name}"] = dd.std(axis=1)
        rows[f"slope_pos_fraction_{name}"] = (dd > 0).mean(axis=1)
        rows[f"slope_sign_change_count_{name}"] = np.sum(np.diff(np.sign(dd), axis=1) != 0, axis=1).astype(float)
        flat = np.abs(dd) < 0.001
        rows[f"nearflat_count_{name}"] = flat.sum(axis=1).astype(float)
        rows[f"nearflat_run_count_{name}"] = run_count(flat)
        rows[f"nearflat_longest_run_{name}"] = longest_run(flat)

    # Global descriptors.
    rows["global_norm_sd"] = x.std(axis=1)
    rows["global_log_sd"] = logx.std(axis=1)
    rows["global_norm_range"] = x.max(axis=1) - x.min(axis=1)
    rows["global_norm_skew"] = stats.skew(x, axis=1)
    rows["global_norm_kurtosis"] = stats.kurtosis(x, axis=1)
    rows["global_abs_slope_mean"] = np.abs(d1).mean(axis=1)
    rows["global_abs_curv_mean"] = np.abs(d2).mean(axis=1)

    # Frequency and shift-tolerant descriptors.
    for i in range(1, 49):
        rows[f"dct_log_{i:02d}"] = coeff[:, i]
    for i in range(1, min(49, fft_mag.shape[1])):
        rows[f"fftmag_log_{i:02d}"] = fft_mag[:, i]
    rows.update(autocorr_features(logx, "log", [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64]))

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def load_data() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """g1090/sdata를 로드하고 각각에 대해 초대형 특징 은행을 만들어 함께 반환."""
    train = load_aec128(FILES["g1090"])
    test = load_aec128(FILES["sdata"])
    train["y"] = train["y"].astype(int)
    test["y"] = test["y"].astype(int)
    ftr = build_feature_bank(train["x"])
    fte = build_feature_bank(test["x"])
    return train, test, ftr, fte


def impute_arrays(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train 중앙값 기준으로 train·test 결측치를 함께 대체."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(xtr), imp.transform(xte)


def top_k_f_classif(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """분산분석 F통계량(f_classif) 기준으로 상위 k개 특징의 인덱스를 선택."""
    k = min(k, x.shape[1])
    if k <= 0:
        return np.array([], dtype=int)
    scores, _ = f_classif(x, y)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(scores)[::-1]
    return order[:k]


def clinical_pipeline() -> Pipeline:
    """결측대체→표준화→로지스틱 회귀(정규화 거의 없음)로 이어지는 임상 모델 파이프라인 생성."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def metric_row(dataset: str, model: str, y: np.ndarray, score: np.ndarray) -> dict:
    """데이터셋/모델 이름과 점수로부터 AUC/AP/로그손실/Brier를 한 행으로 정리."""
    prob = np.clip(sigmoid(score), 1e-6, 1.0 - 1e-6)
    return {
        "dataset": dataset,
        "model": model,
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def fit_score_selected(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    k: int,
    c: float,
    class_weight: str | None,
) -> tuple[np.ndarray, object]:
    """train에서 상위 k개 특징을 F통계량으로 선택해 표준화 후 로지스틱 회귀를 학습하고, 검증 데이터 점수를 반환 (AEC 특징 단독 버전)."""
    xtr_imp, xva_imp = impute_arrays(xtr, xva)
    idx = top_k_f_classif(xtr_imp, ytr, k)
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr_imp[:, idx])
    xva_s = scaler.transform(xva_imp[:, idx])
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000, class_weight=class_weight, random_state=SEED)
    model.fit(xtr_s, ytr)
    return model.decision_function(xva_s), (idx, scaler, model)


def fit_score_clinical_plus_selected(
    aec_tr: np.ndarray,
    clin_tr: np.ndarray,
    ytr: np.ndarray,
    aec_va: np.ndarray,
    clin_va: np.ndarray,
    k: int,
    c: float,
    class_weight: str | None,
) -> tuple[np.ndarray, object]:
    """임상 모델 로짓 + train에서 F통계량으로 선택한 상위 k개 AEC 특징을 함께 표준화·결합해 로지스틱 회귀를 학습하고, 검증 데이터 점수를 반환."""
    clinical = clinical_pipeline()
    clinical.fit(clin_tr, ytr)
    c_tr = clinical.decision_function(clin_tr)
    c_va = clinical.decision_function(clin_va)
    atr_imp, ava_imp = impute_arrays(aec_tr, aec_va)
    idx = top_k_f_classif(atr_imp, ytr, k)
    ztr = np.column_stack([c_tr, atr_imp[:, idx]])
    zva = np.column_stack([c_va, ava_imp[:, idx]])
    scaler = StandardScaler()
    ztr_s = scaler.fit_transform(ztr)
    zva_s = scaler.transform(zva)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000, class_weight=class_weight, random_state=SEED)
    model.fit(ztr_s, ytr)
    return model.decision_function(zva_s), (clinical, idx, scaler, model)


def evaluate_grid(
    model_name: str,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
    feature_names: list[str],
    clinical_xtr: np.ndarray | None = None,
    clinical_xte: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """k(선택 특징 수) x C(정규화) 그리드를 5-fold 교차검증으로 평가해 최적 하이퍼파라미터를 고르고,
    그 값으로 재적합한 OOF 점수·외부 예측·최종 선택된 특징·폴드별 선택 특징까지 모두 계산."""
    k_grid = [5, 10, 20, 40, 80, 160, 320]
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    cw_grid: list[str | None] = [None]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_rows = []

    for k in k_grid:
        for c in c_grid:
            for cw in cw_grid:
                oof = np.zeros(len(ytr), dtype=float)
                for fold_id, (tr_idx, va_idx) in enumerate(skf.split(xtr, ytr), start=1):
                    if clinical_xtr is None:
                        score, _ = fit_score_selected(xtr[tr_idx], ytr[tr_idx], xtr[va_idx], k, c, cw)
                    else:
                        score, _ = fit_score_clinical_plus_selected(
                            xtr[tr_idx],
                            clinical_xtr[tr_idx],
                            ytr[tr_idx],
                            xtr[va_idx],
                            clinical_xtr[va_idx],
                            k,
                            c,
                            cw,
                        )
                    oof[va_idx] = score
                grid_rows.append(
                    {
                        "model": model_name,
                        "k": k,
                        "C": c,
                        "class_weight": "balanced" if cw == "balanced" else "none",
                        "oof_auc": float(roc_auc_score(ytr, oof)),
                        "oof_average_precision": float(average_precision_score(ytr, oof)),
                        "oof_log_loss_recalibrated": float(log_loss(ytr, np.clip(sigmoid(oof), 1e-6, 1 - 1e-6))),
                    }
                )

    grid = pd.DataFrame(grid_rows)
    best = grid.sort_values(["oof_auc", "oof_average_precision"], ascending=[False, False]).iloc[0]
    best_k = int(best["k"])
    best_c = float(best["C"])
    best_cw = None if best["class_weight"] == "none" else "balanced"

    # Refit OOF for chosen hyperparameters and record selected features by fold.
    oof = np.zeros(len(ytr), dtype=float)
    fold_rows = []
    feature_counts = pd.Series(0, index=feature_names, dtype=int)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(xtr, ytr), start=1):
        if clinical_xtr is None:
            score, fitted = fit_score_selected(xtr[tr_idx], ytr[tr_idx], xtr[va_idx], best_k, best_c, best_cw)
            idx = fitted[0]
        else:
            score, fitted = fit_score_clinical_plus_selected(
                xtr[tr_idx],
                clinical_xtr[tr_idx],
                ytr[tr_idx],
                xtr[va_idx],
                clinical_xtr[va_idx],
                best_k,
                best_c,
                best_cw,
            )
            idx = fitted[1]
        oof[va_idx] = score
        selected = [feature_names[i] for i in idx]
        feature_counts.loc[selected] += 1
        fold_rows.append({"model": model_name, "fold": fold_id, "selected_features": "|".join(selected)})

    # External final fit.
    if clinical_xtr is None:
        xtr_imp, xte_imp = impute_arrays(xtr, xte)
        idx = top_k_f_classif(xtr_imp, ytr, best_k)
        scaler = StandardScaler()
        xtr_s = scaler.fit_transform(xtr_imp[:, idx])
        xte_s = scaler.transform(xte_imp[:, idx])
        final = LogisticRegression(C=best_c, solver="lbfgs", max_iter=5000, class_weight=best_cw, random_state=SEED)
        final.fit(xtr_s, ytr)
        ext_score = final.decision_function(xte_s)
        final_feature_names = [feature_names[i] for i in idx]
    else:
        clinical = clinical_pipeline()
        clinical.fit(clinical_xtr, ytr)
        c_tr = clinical.decision_function(clinical_xtr)
        c_te = clinical.decision_function(clinical_xte)
        xtr_imp, xte_imp = impute_arrays(xtr, xte)
        idx = top_k_f_classif(xtr_imp, ytr, best_k)
        ztr = np.column_stack([c_tr, xtr_imp[:, idx]])
        zte = np.column_stack([c_te, xte_imp[:, idx]])
        scaler = StandardScaler()
        ztr_s = scaler.fit_transform(ztr)
        zte_s = scaler.transform(zte)
        final = LogisticRegression(C=best_c, solver="lbfgs", max_iter=5000, class_weight=best_cw, random_state=SEED)
        final.fit(ztr_s, ytr)
        ext_score = final.decision_function(zte_s)
        final_feature_names = [feature_names[i] for i in idx]

    perf = pd.DataFrame(
        [
            metric_row("g1090_oof", model_name, ytr, oof),
            metric_row("sdata_external", model_name, yte, ext_score),
        ]
    )
    perf["selected_k"] = best_k
    perf["selected_C"] = best_c
    perf["selected_class_weight"] = "balanced" if best_cw == "balanced" else "none"
    final_features = pd.DataFrame(
        {
            "model": model_name,
            "feature": final_feature_names,
            "selected_in_final": 1,
            "cv_selection_count_0_to_5": [int(feature_counts.get(f, 0)) for f in final_feature_names],
        }
    )
    folds = pd.DataFrame(fold_rows)
    return perf, grid, final_features, folds


def discovery_stats(ftr: pd.DataFrame, fte: pd.DataFrame, train: dict, test: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """초대형 특징 은행의 모든 특징 각각에 대해 코호트별 AUC·평균차와 임상매칭쌍 대응차이를 계산하고,
    두 코호트 모두 방향이 일치하는 특징들만 골라 판별력 기준으로 정렬한 "공통방향 발견" 표를 만듦."""
    rows = []
    pairs = {
        "g1090": pd.read_csv(BASE_DIR / "g1090_clinical_matched_pairs.csv"),
        "sdata": pd.read_csv(BASE_DIR / "sdata_clinical_matched_pairs.csv"),
    }
    pairs = {k: v[v["within_caliper"]].copy() for k, v in pairs.items()}
    for cohort, feat_df, d in [("g1090", ftr, train), ("sdata", fte, test)]:
        y = d["y"].astype(bool)
        low_idx = pairs[cohort]["low_index"].to_numpy(dtype=int)
        non_idx = pairs[cohort]["matched_nonlow_index"].to_numpy(dtype=int)
        for feature in feat_df.columns:
            values = feat_df[feature].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=np.nanmedian(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 0.0)
            auc_high = roc_auc_score(y.astype(int), values)
            low = values[y]
            non = values[~y]
            pdiff = values[low_idx] - values[non_idx]
            rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "diff_low_minus_nonlow": float(np.mean(low) - np.mean(non)),
                    "auc_if_higher_predicts_low": float(auc_high),
                    "auc_best_direction": float(max(auc_high, 1 - auc_high)),
                    "matched_diff_low_minus_nonlow": float(np.mean(pdiff)),
                    "matched_diff_abs": float(abs(np.mean(pdiff))),
                }
            )
    stats_df = pd.DataFrame(rows)
    wide_all = stats_df.pivot(index="feature", columns="cohort", values="diff_low_minus_nonlow")
    wide_pair = stats_df.pivot(index="feature", columns="cohort", values="matched_diff_low_minus_nonlow")
    common_rows = []
    for feat in wide_all.index:
        g_all, s_all = float(wide_all.loc[feat, "g1090"]), float(wide_all.loc[feat, "sdata"])
        g_pair, s_pair = float(wide_pair.loc[feat, "g1090"]), float(wide_pair.loc[feat, "sdata"])
        same_all = np.sign(g_all) == np.sign(s_all) and g_all != 0 and s_all != 0
        same_pair = np.sign(g_pair) == np.sign(s_pair) and g_pair != 0 and s_pair != 0
        if same_all and same_pair:
            g = stats_df[(stats_df["cohort"].eq("g1090")) & (stats_df["feature"].eq(feat))].iloc[0]
            s = stats_df[(stats_df["cohort"].eq("sdata")) & (stats_df["feature"].eq(feat))].iloc[0]
            common_rows.append(
                {
                    "feature": feat,
                    "direction": "higher_in_low" if g_all > 0 else "lower_in_low",
                    "g1090_auc_best": float(g["auc_best_direction"]),
                    "sdata_auc_best": float(s["auc_best_direction"]),
                    "min_site_auc_best": float(min(g["auc_best_direction"], s["auc_best_direction"])),
                    "mean_site_auc_best": float((g["auc_best_direction"] + s["auc_best_direction"]) / 2),
                    "g1090_all_diff": g_all,
                    "sdata_all_diff": s_all,
                    "g1090_matched_diff": g_pair,
                    "sdata_matched_diff": s_pair,
                    "min_abs_matched_diff": float(min(abs(g_pair), abs(s_pair))),
                }
            )
    common_df = pd.DataFrame(common_rows)
    common_df = common_df.sort_values(
        ["min_site_auc_best", "mean_site_auc_best", "min_abs_matched_diff"], ascending=[False, False, False]
    )
    return stats_df, common_df


def bootstrap_delta(y: np.ndarray, clinical: np.ndarray, combined: np.ndarray, n_boot: int = 2000) -> dict:
    """결합모델과 임상모델의 AUC 차이를 부트스트랩 재표본추출로 신뢰구간과 p값과 함께 추정."""
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], combined[idx]) - roc_auc_score(y[idx], clinical[idx]))
    arr = np.asarray(vals)
    return {
        "delta_auc_mean": float(np.mean(arr)),
        "delta_auc_ci2.5": float(np.quantile(arr, 0.025)),
        "delta_auc_ci97.5": float(np.quantile(arr, 0.975)),
        "p_delta_le_0": float(np.mean(arr <= 0)),
    }


def plot_performance(perf: pd.DataFrame) -> None:
    """모델별 train OOF vs 외부 AUC를 나란히 막대그래프로 비교해 PNG로 저장."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    order = perf.pivot(index="model", columns="dataset", values="auc").sort_values("sdata_external", ascending=False)
    x = np.arange(len(order))
    ax.bar(x - 0.18, order["g1090_oof"], width=0.36, color="#4C78A8", label="g1090 OOF")
    ax.bar(x + 0.18, order["sdata_external"], width=0.36, color="#F58518", label="sdata external")
    ax.set_xticks(x)
    ax.set_xticklabels(order.index, rotation=25, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Mass AEC Feature Combination Models", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mass_feature_combination_auc.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 지금까지 만든 어떤 특징 집합보다 훨씬 더 많은 후보
    (수천 차원)를 만들어 놓고, "train 안에서 상위 k개를 자동으로 골라 쓰는" 모델을 학습하면
    임상변수 대비 얼마나 이득을 보는가?):

    1. g1090/sdata를 로드하고, build_feature_bank로 초대형 특징 은행(다중스케일 레벨/기울기/곡률/
       Haar엣지, 고정구간 쌍별 대비·비율, 극값/기하, 거칠기/평탄구간, 전역통계, DCT/FFT/자기상관)을
       만들어 CSV로 저장.
    2. discovery_stats로 모든 특징 각각의 코호트별 AUC·평균차와, 이전 신호감사에서 만든 임상매칭쌍
       기준 대응차이를 계산해, 두 코호트 모두 방향이 일치하는 "공통방향" 특징들을 판별력 순으로
       정렬한 발견 표를 CSV로 저장 (이게 순수 tone/방향 확인용 사전 스캔).
    3. 임상 단독 기준모델을 학습하고, evaluate_grid를 두 번 실행: (a) AEC 특징만 써서 k/C 그리드서치
       (b) 임상변수+AEC 특징을 합쳐 k/C 그리드서치. 둘 다 5-fold CV로 train OOF 점수를 만들고
       전체 train으로 재학습해 외부 예측까지 수행.
    4. 3개 모델(임상단독/AEC단독/결합)의 성능표, 그리드서치 전체 결과, 최종 선택된 특징 목록,
       폴드별 선택 특징을 모두 CSV로 저장하고, 모델별 AUC 막대그래프를 저장.
    5. 특징 은행 크기, 상위 공통방향 발견 특징, 전체 성능을 JSON으로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, ftr, fte = load_data()
    ftr.to_csv(OUT_DIR / "g1090_mass_feature_bank.csv", index=False)
    fte.to_csv(OUT_DIR / "sdata_mass_feature_bank.csv", index=False)

    stats_df, common_df = discovery_stats(ftr, fte, train, test)
    stats_df.to_csv(OUT_DIR / "mass_feature_discovery_stats_by_cohort.csv", index=False)
    common_df.to_csv(OUT_DIR / "mass_feature_common_direction_discovery.csv", index=False)

    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    xtr = ftr.to_numpy(dtype=float)
    xte = fte.to_numpy(dtype=float)
    feature_names = list(ftr.columns)
    clinical_xtr = clinical_raw(train["meta"])
    clinical_xte = clinical_raw(test["meta"])

    # Clinical-only reference.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clinical_oof = np.zeros(len(ytr), dtype=float)
    for tr_idx, va_idx in skf.split(clinical_xtr, ytr):
        m = clinical_pipeline()
        m.fit(clinical_xtr[tr_idx], ytr[tr_idx])
        clinical_oof[va_idx] = m.decision_function(clinical_xtr[va_idx])
    clinical_final = clinical_pipeline()
    clinical_final.fit(clinical_xtr, ytr)
    clinical_ext = clinical_final.decision_function(clinical_xte)
    clinical_perf = pd.DataFrame(
        [
            metric_row("g1090_oof", "clinical_only", ytr, clinical_oof),
            metric_row("sdata_external", "clinical_only", yte, clinical_ext),
        ]
    )

    aec_perf, aec_grid, aec_final_features, aec_folds = evaluate_grid(
        "aec_only_selected_mass_features", xtr, ytr, xte, yte, feature_names
    )
    comb_perf, comb_grid, comb_final_features, comb_folds = evaluate_grid(
        "clinical_plus_selected_mass_features",
        xtr,
        ytr,
        xte,
        yte,
        feature_names,
        clinical_xtr=clinical_xtr,
        clinical_xte=clinical_xte,
    )

    perf = pd.concat([clinical_perf, aec_perf, comb_perf], ignore_index=True)
    perf.to_csv(OUT_DIR / "mass_feature_combination_performance.csv", index=False)
    aec_grid.to_csv(OUT_DIR / "aec_only_grid.csv", index=False)
    comb_grid.to_csv(OUT_DIR / "clinical_plus_aec_grid.csv", index=False)
    pd.concat([aec_final_features, comb_final_features], ignore_index=True).to_csv(
        OUT_DIR / "final_selected_features.csv", index=False
    )
    pd.concat([aec_folds, comb_folds], ignore_index=True).to_csv(OUT_DIR / "cv_fold_selected_features.csv", index=False)
    plot_performance(perf)

    # Bootstrap external AUC delta for the best combined model is easier to recompute from final selected features.
    summary = {
        "n_features": int(ftr.shape[1]),
        "n_train": int(len(ytr)),
        "n_external": int(len(yte)),
        "top_common_discovery_features": common_df.head(25).to_dict(orient="records"),
        "performance": perf.to_dict(orient="records"),
    }
    (OUT_DIR / "mass_feature_combination_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Feature bank:", ftr.shape)
    print("\nTop common-direction discovery features")
    print(common_df.head(25).to_string(index=False))
    print("\nPerformance")
    print(perf.to_string(index=False))
    print("\nFinal selected features")
    print(pd.concat([aec_final_features, comb_final_features], ignore_index=True).to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
