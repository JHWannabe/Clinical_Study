from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.interpolate import BSpline
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402
from aec_vendor_neutral_preprocessing_audit import company_from_manufacturer  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_whole_curve_models"
WORK_DATA_DIR = Path(__file__).resolve().parent / "data_cache"
SEED = 20260701
SIGMA = 1.0
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


@dataclass(frozen=True)
class Dataset:
    """한 코호트의 메타데이터, 원시 AEC_128 행렬, 라벨, 제조사 범주를 담는 묶음."""

    meta: pd.DataFrame
    raw: np.ndarray
    y: np.ndarray
    company: np.ndarray


def load_dataset(path: Path) -> Dataset:
    """엑셀에서 원시 AEC_128 곡선, 라벨, 제조사 범주를 읽어 Dataset으로 반환."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    company = meta["Manufacturer"].map(company_from_manufacturer).to_numpy()
    return Dataset(meta=meta, raw=raw, y=y, company=company)


def smooth_norm(raw: np.ndarray) -> np.ndarray:
    """원시곡선을 가우시안 평활화한 뒤 환자 평균으로 정규화."""
    return row_norm(ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest"))


def company_harmonize(
    x_train_source: np.ndarray,
    x_apply: np.ndarray,
    company_train_source: np.ndarray,
    company_apply: np.ndarray,
) -> np.ndarray:
    """train 소스에서 회사별/전체 평균 템플릿을 만들어, 적용 대상 곡선의 회사 평균을 전체 평균으로 치환(harmonize)."""
    keep = company_train_source != "Other"
    global_template = x_train_source[keep].mean(axis=0)
    templates = {
        label: x_train_source[company_train_source == label].mean(axis=0)
        for label in np.unique(company_train_source[keep])
    }
    out = np.empty_like(x_apply)
    for i, label in enumerate(company_apply):
        out[i] = x_apply[i] - templates.get(label, global_template) + global_template
    return out


def residual(x: np.ndarray) -> np.ndarray:
    """곡선의 양 끝점을 잇는 직선을 빼서 전반적 기울기 성분을 제거한 잔차를 계산."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    line = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    return x - line


def d1(x: np.ndarray) -> np.ndarray:
    """1차 도함수를 계산 (길이를 맞추기 위해 첫 값을 복제)."""
    out = np.diff(x, axis=1)
    return np.column_stack([out[:, :1], out])


def d2(x: np.ndarray) -> np.ndarray:
    """1차 도함수를 한 번 더 미분해 2차 도함수(곡률)를 계산."""
    first = d1(x)
    out = np.diff(first, axis=1)
    return np.column_stack([out[:, :1], out])


def z_rows(x: np.ndarray) -> np.ndarray:
    """각 행(환자)을 자기 자신의 평균/표준편차로 z-표준화."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x - mu) / sd


def channel_dict(curve: np.ndarray) -> dict[str, np.ndarray]:
    """원곡선과 (표준화된) 잔차/기울기/곡률 4가지 채널을 딕셔너리로 묶음."""
    r = residual(curve)
    s = d1(curve)
    c = d2(curve)
    return {
        "curve": curve,
        "residual_z": z_rows(r),
        "slope_z": z_rows(s),
        "curvature_z": z_rows(c),
    }


def make_channels_for_fold(g: Dataset, s: Dataset, train_idx: np.ndarray | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """폴드의 train 인덱스(없으면 전체)만으로 회사 보정 템플릿을 만들어 g/s 양쪽에 적용한 뒤, 4채널 딕셔너리로 변환 (데이터 유출 방지를 위해 템플릿은 항상 train에서만 학습)."""
    g_norm = smooth_norm(g.raw)
    s_norm = smooth_norm(s.raw)
    source_idx = np.arange(len(g.y)) if train_idx is None else train_idx
    g_h = company_harmonize(g_norm[source_idx], g_norm, g.company[source_idx], g.company)
    s_h = company_harmonize(g_norm[source_idx], s_norm, g.company[source_idx], s.company)
    return channel_dict(g_h), channel_dict(s_h)


def full_matrix(channels: dict[str, np.ndarray]) -> np.ndarray:
    """4채널(원곡선/잔차/기울기/곡률)을 옆으로 이어붙여 512차원 특징 행렬을 만듦."""
    return np.column_stack([channels["curve"], channels["residual_z"], channels["slope_z"], channels["curvature_z"]])


def spline_basis(n: int = 128, n_basis: int = 12, degree: int = 3) -> np.ndarray:
    """128개 점 위에서 B-스플라인 기저 함수 12개를 계산 — 함수형 데이터 분석(FDA)의 저차원 표현 기반."""
    n_internal = n_basis - degree - 1
    interior = np.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else np.array([])
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    x = np.linspace(0.0, 1.0, n)
    basis = []
    for i in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        basis.append(BSpline(knots, coeff, degree, extrapolate=True)(x))
    return np.column_stack(basis)


SPLINE_BASIS = spline_basis()
SPLINE_PINV = np.linalg.pinv(SPLINE_BASIS)


def spline_features(channels: dict[str, np.ndarray]) -> np.ndarray:
    """4채널 각각을 B-스플라인 기저에 투영해 12개 계수로 압축(함수형 로지스틱 회귀의 입력)."""
    coeffs = []
    for name in ["curve", "residual_z", "slope_z", "curvature_z"]:
        coeffs.append(channels[name] @ SPLINE_PINV.T)
    return np.column_stack(coeffs)


def haar_row(row: np.ndarray) -> np.ndarray:
    """한 곡선에 대해 Haar 웨이블릿 변환(재귀적 평균/차분)을 수행해 전체 스케일의 계수를 계산."""
    x = row.astype(float).copy()
    coeffs = []
    while len(x) > 1:
        avg = (x[0::2] + x[1::2]) / np.sqrt(2.0)
        diff = (x[0::2] - x[1::2]) / np.sqrt(2.0)
        coeffs.append(diff)
        x = avg
    coeffs.append(x)
    return np.concatenate(coeffs)


def wavelet_features(channels: dict[str, np.ndarray]) -> np.ndarray:
    """4채널 각각에 Haar 웨이블릿 변환을 적용해 이어붙인 특징 행렬을 만듦."""
    out = []
    for name in ["curve", "residual_z", "slope_z", "curvature_z"]:
        out.append(np.vstack([haar_row(row) for row in channels[name]]))
    return np.column_stack(out)


def auc_p(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U 검정으로 AUC의 유의성(p값)을 계산."""
    if len(np.unique(y)) < 2:
        return np.nan
    return float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)


def score_estimator(model, x: np.ndarray) -> np.ndarray:
    """모델에 decision_function이 있으면 그것을, 없으면 양성 클래스 확률을 점수로 사용."""
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def inner_cv(y: np.ndarray) -> StratifiedKFold:
    """하이퍼파라미터 튜닝용 내부 3-fold 교차검증 분할기를 생성."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def functional_model(y: np.ndarray):
    """스플라인 계수 특징에 L2 정규화 로지스틱(내부 CV로 C 자동선택)을 적용하는 "함수형 데이터 분석" 모델."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=[0.03, 0.1, 0.3, 1.0, 3.0],
            cv=inner_cv(y),
            scoring="roc_auc",
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            refit=True,
        ),
    )


def fpca_model(y: np.ndarray):
    """4채널 전체(512차원)에 PCA로 차원을 줄인 뒤 로지스틱 회귀를 적용하는 "함수형 주성분분석(FPCA)" 모델 (주성분 개수·C를 그리드서치로 튜닝)."""
    pipe = make_pipeline(
        StandardScaler(),
        PCA(random_state=SEED),
        LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000),
    )
    return GridSearchCV(
        pipe,
        {
            "pca__n_components": [4, 6, 10, 16, 24],
            "logisticregression__C": [0.03, 0.1, 0.3, 1.0],
        },
        scoring="roc_auc",
        cv=inner_cv(y),
        n_jobs=1,
        refit=True,
    )


def wavelet_model(y: np.ndarray):
    """웨이블릿 특징 중 상위 80개를 F통계량으로 골라 L1 정규화 로지스틱(내부 CV로 C 자동선택)을 적용하는 모델."""
    return make_pipeline(
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=80),
        LogisticRegressionCV(
            Cs=[0.02, 0.05, 0.1, 0.3, 1.0],
            cv=inner_cv(y),
            scoring="roc_auc",
            solver="liblinear",
            penalty="l1",
            class_weight="balanced",
            max_iter=5000,
            refit=True,
        ),
    )


def kernel_svm_model(y: np.ndarray):
    """4채널 전체를 PCA(24차원)로 줄인 뒤 RBF 커널 SVM을 적용하는 비선형 모델 (C·gamma를 그리드서치로 튜닝)."""
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=24, random_state=SEED),
        SVC(kernel="rbf", class_weight="balanced"),
    )
    return GridSearchCV(
        pipe,
        {
            "svc__C": [0.1, 0.3, 1.0, 3.0],
            "svc__gamma": ["scale", 0.01, 0.03, 0.1],
        },
        scoring="roc_auc",
        cv=inner_cv(y),
        n_jobs=1,
        refit=True,
    )


def tabular_feature_builder(model_name: str, channels_g: dict[str, np.ndarray], channels_s: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """모델 이름에 맞는 특징 추출 방식(스플라인/전체행렬/웨이블릿)을 선택해 train·test 특징 행렬을 만듦."""
    if model_name == "functional_logistic":
        return spline_features(channels_g), spline_features(channels_s)
    if model_name == "fpca_logistic":
        return full_matrix(channels_g), full_matrix(channels_s)
    if model_name == "wavelet_logistic":
        return wavelet_features(channels_g), wavelet_features(channels_s)
    if model_name == "kernel_svm":
        return full_matrix(channels_g), full_matrix(channels_s)
    raise ValueError(model_name)


def estimator_factory(model_name: str, y_train: np.ndarray):
    """모델 이름에 맞는 추정기(함수형/FPCA/웨이블릿/커널SVM)를 생성."""
    if model_name == "functional_logistic":
        return functional_model(y_train)
    if model_name == "fpca_logistic":
        return fpca_model(y_train)
    if model_name == "wavelet_logistic":
        return wavelet_model(y_train)
    if model_name == "kernel_svm":
        return kernel_svm_model(y_train)
    raise ValueError(model_name)


def run_tabular_model(model_name: str, g: Dataset, s: Dataset) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """지정된 전곡선 모델(함수형/FPCA/웨이블릿/커널SVM)을 5-fold로 학습해(폴드마다 회사보정 채널을
    다시 만들어 유출 방지) train OOF 점수와 외부 예측을 계산."""
    yg = g.y.astype(int)
    oof = np.zeros(len(yg), dtype=float)
    logs = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(yg)), yg)):
        ch_g, ch_s = make_channels_for_fold(g, s, tr)
        xg, _ = tabular_feature_builder(model_name, ch_g, ch_s)
        model = estimator_factory(model_name, yg[tr])
        model.fit(xg[tr], yg[tr])
        oof[va] = score_estimator(model, xg[va])
        logs.append({"model": model_name, "fold": fold, "best_params": getattr(model, "best_params_", None)})
    ch_g, ch_s = make_channels_for_fold(g, s, None)
    xg, xs = tabular_feature_builder(model_name, ch_g, ch_s)
    final = estimator_factory(model_name, yg)
    final.fit(xg, yg)
    ext = score_estimator(final, xs)
    logs.append({"model": model_name, "fold": "final", "best_params": getattr(final, "best_params_", None)})
    return oof, ext, logs


def window_specs(n: int = 128) -> list[tuple[str, int, int]]:
    """4채널 x 5개 폭(8~32) 조합으로 겹치는 슬라이딩 윈도우 후보(채널명, 시작, 끝)들을 나열 — shapelet 탐색 대상."""
    specs = []
    for ch in ["curve", "residual_z", "slope_z", "curvature_z"]:
        for length in [8, 12, 16, 24, 32]:
            for start in range(0, n - length + 1, 4):
                specs.append((ch, start, start + length))
    return specs


SHAPELET_WINDOWS = window_specs()


def shapelet_fit_features(
    channels_train_source: dict[str, np.ndarray],
    y_train: np.ndarray,
    channels_apply: dict[str, np.ndarray],
    top_k: int = 80,
) -> tuple[np.ndarray, list[dict]]:
    """각 윈도우 후보마다 train에서 "저근감소증 평균 모양(prototype)"과 "비저근감소증 평균 모양"을 만들고,
    각 환자 구간이 어느 프로토타입에 더 가까운지(거리 차이)를 shapelet 특징으로 계산. train 효과크기가
    큰 상위 top_k개 윈도우만 선택해 적용 대상 데이터의 특징 행렬과 선택 메타데이터를 반환."""
    train_feats = []
    apply_feats = []
    meta_rows = []
    yb = y_train.astype(bool)
    for ch_name, start, end in SHAPELET_WINDOWS:
        train_seg = channels_train_source[ch_name][:, start:end]
        apply_seg = channels_apply[ch_name][:, start:end]
        low_proto = train_seg[yb].mean(axis=0)
        non_proto = train_seg[~yb].mean(axis=0)
        train_feat = np.linalg.norm(train_seg - non_proto, axis=1) - np.linalg.norm(train_seg - low_proto, axis=1)
        apply_feat = np.linalg.norm(apply_seg - non_proto, axis=1) - np.linalg.norm(apply_seg - low_proto, axis=1)
        if np.std(train_feat) == 0:
            score = 0.0
            p = np.nan
        else:
            score = abs(np.mean(train_feat[yb]) - np.mean(train_feat[~yb])) / (np.std(train_feat) + 1e-8)
            p = auc_p(y_train, train_feat)
        train_feats.append(train_feat)
        apply_feats.append(apply_feat)
        meta_rows.append({"channel": ch_name, "start": start + 1, "end": end, "effect_score": score, "p_train": p})
    train_mat = np.column_stack(train_feats)
    apply_mat = np.column_stack(apply_feats)
    order = np.argsort([-r["effect_score"] for r in meta_rows])[:top_k]
    selected = [{**meta_rows[i], "selected_rank": rank + 1} for rank, i in enumerate(order)]
    return apply_mat[:, order], selected


def shapelet_model(y: np.ndarray):
    """선택된 shapelet 거리 특징에 L2 정규화 로지스틱(내부 CV로 C 자동선택)을 적용하는 모델."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=[0.03, 0.1, 0.3, 1.0],
            cv=inner_cv(y),
            scoring="roc_auc",
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            refit=True,
        ),
    )


def run_shapelet_model(g: Dataset, s: Dataset) -> tuple[np.ndarray, np.ndarray, list[dict], pd.DataFrame]:
    """shapelet(프로토타입 거리) 모델을 5-fold로 학습해 train OOF/외부 점수와, 각 폴드에서 선택된 윈도우 메타데이터를 함께 반환."""
    yg = g.y.astype(int)
    oof = np.zeros(len(yg), dtype=float)
    logs = []
    selected_rows = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(yg)), yg)):
        ch_g, _ = make_channels_for_fold(g, s, tr)
        train_source = {k: v[tr] for k, v in ch_g.items()}
        xg_selected, selected = shapelet_fit_features(train_source, yg[tr], ch_g)
        model = shapelet_model(yg[tr])
        model.fit(xg_selected[tr], yg[tr])
        oof[va] = score_estimator(model, xg_selected[va])
        logs.append({"model": "shapelet_prototype", "fold": fold, "best_params": None})
        selected_rows.extend([{**row, "fold": fold} for row in selected])
    ch_g, ch_s = make_channels_for_fold(g, s, None)
    xg_selected, selected = shapelet_fit_features(ch_g, yg, ch_g)
    _, selected_ext = shapelet_fit_features(ch_g, yg, ch_s)
    # shapelet_fit_features re-selects the same windows because train source is identical,
    # but it returns only transformed apply features. Use selected metadata from train.
    xs_selected, _ = shapelet_fit_features(ch_g, yg, ch_s)
    final = shapelet_model(yg)
    final.fit(xg_selected, yg)
    ext = score_estimator(final, xs_selected)
    selected_rows.extend([{**row, "fold": "final"} for row in selected])
    logs.append({"model": "shapelet_prototype", "fold": "final", "best_params": None})
    return oof, ext, logs, pd.DataFrame(selected_rows)


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """주어진 임계값에서 정확도·민감도·특이도·균형정확도와 혼동행렬을 계산."""
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def model_metrics(dataset: str, model_name: str, y: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict:
    """데이터셋/모델 이름별로 AUC·AP·Brier와 (임계값 없으면 Youden 자동계산) 혼동행렬 지표를 한 행으로 정리."""
    if threshold is None:
        threshold = threshold_youden(y.astype(int), score)
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return {
        "dataset": dataset,
        "model": model_name,
        "auc": float(roc_auc_score(y, score)),
        "auc_p_mannwhitney": auc_p(y, score),
        "average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))),
        **binary_metrics(y, score, threshold),
    }


def bootstrap_auc_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, seed: int, n_boot: int = 3000) -> dict:
    """두 점수(base vs candidate)의 AUC 차이를 부트스트랩으로 신뢰구간과 양측 p값과 함께 추정."""
    rng = np.random.default_rng(seed)
    obs = float(roc_auc_score(y, candidate) - roc_auc_score(y, base))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], base[idx])))
    arr = np.asarray(vals)
    p = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)) if len(arr) else np.nan
    return {
        "delta_auc": obs,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "delta_auc_p_boot": float(min(1.0, p)) if np.isfinite(p) else np.nan,
    }


def exact_p(a: int, b: int) -> float:
    """두 카운트에 대한 이항 정확검정(양측) p값을 계산."""
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상 양성 판정을 최종 판정으로 바꿨을 때 민감도손실/특이도이득/정확도변화에 대한 대응 이항검정 p값을 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(~yy & ~clinical_pos & final_pos))
    cc = clinical_pos == yy
    fc = final_pos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
        "tp_lost_n": sens_loss,
        "fp_removed_n": spec_gain,
    }


def fisher_event_p(y: np.ndarray, kept: np.ndarray, deesc: np.ndarray) -> float:
    """유지군과 하향조정군의 사건 발생률 차이에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[kept] == 1))
    b = int(np.sum(y[kept] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    return np.nan if not (a + b and c + d) else float(stats.fisher_exact([[a, b], [c, d]])[1])


def deesc_row(dataset: str, op: str, y: np.ndarray, clinical_pos: np.ndarray, aec_score: np.ndarray, threshold: float) -> dict:
    """임상 양성군 중 AEC 점수가 임계값 이하인 환자를 하향조정군으로 분류하고, 임상 단독 대비 민감도손실/특이도이득/정확도변화와 하향조정군 통계를 계산."""
    deesc = clinical_pos & (aec_score <= threshold)
    final_pos = clinical_pos & ~deesc
    base = binary_metrics(y, clinical_pos.astype(float), 0.5)
    post = binary_metrics(y, final_pos.astype(float), 0.5)
    return {
        "dataset": dataset,
        "operating_point": op,
        "aec_low_threshold": float(threshold),
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_event_p(y, final_pos, deesc),
        **paired_pvalues(y, clinical_pos, final_pos),
    }


def choose_deesc_threshold(y: np.ndarray, clinical_pos: np.ndarray, aec_score: np.ndarray, op: str) -> tuple[float, dict]:
    """train 안에서 여러 하향조정 임계값 후보를 스캔해, 민감도손실 유의(p<0.05)·특이도이득 양수
    조건을 만족하면서 "선택 점수"가 가장 높은 임계값을 선택."""
    values = aec_score[clinical_pos]
    candidates = np.unique(np.quantile(values, np.linspace(0.05, 0.45, 41)))
    best = None
    for th in candidates:
        row = deesc_row("Gangnam internal OOF", op, y, clinical_pos, aec_score, float(th))
        if row["deesc_n"] < 20 or row["sensitivity_loss_p_exact"] < 0.05 or row["specificity_gain"] <= 0:
            continue
        score = row["specificity_gain"] + 0.35 * row["delta_accuracy"] + 0.25 * row["delta_balanced_accuracy"] - 0.20 * row["sensitivity_loss"]
        candidate = {**row, "selection_score": float(score)}
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    if best is None:
        th = float(np.quantile(values, 0.20))
        best = {**deesc_row("Gangnam internal OOF", op, y, clinical_pos, aec_score, th), "selection_score": np.nan}
    return float(best["aec_low_threshold"]), best


def evaluate_scores(model_name: str, aec_g: np.ndarray, aec_s: np.ndarray, g: Dataset, s: Dataset, cg: np.ndarray, cs: np.ndarray) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """한 전곡선 모델의 AEC 점수를 임상점수와 스택해 임상/AEC단독/결합 3개 모델 성능을 비교하고, 민감도 목표별 하향조정 분석까지 계산."""
    yg = g.y.astype(int)
    ys = s.y.astype(int)
    clinical_th = threshold_youden(yg, cg)
    aec_th = threshold_youden(yg, aec_g)
    stack = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)
    stack.fit(np.column_stack([cg, aec_g]), yg)
    fusion_g = stack.decision_function(np.column_stack([cg, aec_g]))
    fusion_s = stack.decision_function(np.column_stack([cs, aec_s]))
    fusion_th = threshold_youden(yg, fusion_g)
    rows = [
        {"whole_curve_model": model_name, **model_metrics("Gangnam internal OOF", "clinical", yg, cg, clinical_th)},
        {"whole_curve_model": model_name, **model_metrics("Sinchon external", "clinical", ys, cs, clinical_th)},
        {"whole_curve_model": model_name, **model_metrics("Gangnam internal OOF", "aec_only", yg, aec_g, aec_th)},
        {"whole_curve_model": model_name, **model_metrics("Sinchon external", "aec_only", ys, aec_s, aec_th)},
        {"whole_curve_model": model_name, **model_metrics("Gangnam internal OOF", "clinical_plus_aec", yg, fusion_g, fusion_th), **bootstrap_auc_delta(yg, cg, fusion_g, SEED + 1)},
        {"whole_curve_model": model_name, **model_metrics("Sinchon external", "clinical_plus_aec", ys, fusion_s, fusion_th), **bootstrap_auc_delta(ys, cs, fusion_s, SEED + 2)},
    ]
    deesc_rows = []
    for op, target in OPS:
        cth = threshold_for_min_sensitivity(yg, cg, target)
        cpos_g = cg >= cth
        cpos_s = cs >= cth
        ath, train_row = choose_deesc_threshold(yg, cpos_g, aec_g, op)
        deesc_rows.append({"whole_curve_model": model_name, **train_row})
        deesc_rows.append({"whole_curve_model": model_name, **deesc_row("Sinchon external", op, ys, cpos_s, aec_s, ath)})
    scores = pd.concat(
        [
            pd.DataFrame({"whole_curve_model": model_name, "cohort": "Gangnam", "y": yg, "clinical": cg, "aec_score": aec_g, "clinical_plus_aec": fusion_g, "company": g.company}),
            pd.DataFrame({"whole_curve_model": model_name, "cohort": "Sinchon", "y": ys, "clinical": cs, "aec_score": aec_s, "clinical_plus_aec": fusion_s, "company": s.company}),
        ],
        ignore_index=True,
    )
    return rows, deesc_rows, scores


def summarize_deesc(deesc: pd.DataFrame) -> pd.DataFrame:
    """모델x데이터셋별로 여러 민감도 목표에 걸친 최소/평균 하향조정 지표를 요약."""
    return (
        deesc.groupby(["whole_curve_model", "dataset"])
        .agg(
            min_p_loss=("sensitivity_loss_p_exact", "min"),
            max_sens_loss=("sensitivity_loss", "max"),
            min_spec_gain=("specificity_gain", "min"),
            mean_spec_gain=("specificity_gain", "mean"),
            min_delta_ba=("delta_balanced_accuracy", "min"),
            mean_delta_ba=("delta_balanced_accuracy", "mean"),
            max_fisher_p=("deesc_event_fisher_p", "max"),
            min_deesc_event_rate=("deesc_event_rate", "min"),
            max_deesc_event_rate=("deesc_event_rate", "max"),
            mean_deesc_event_rate=("deesc_event_rate", "mean"),
        )
        .reset_index()
    )


def plot_model_auc(metrics: pd.DataFrame, out_path: Path) -> None:
    """5가지 전곡선 모델(AEC단독/결합)의 외부 AUC를 임상 단독 기준선과 비교하는 가로 막대그래프를 PNG로 저장."""
    sub = metrics[(metrics["dataset"].eq("Sinchon external")) & (metrics["model"].isin(["aec_only", "clinical_plus_aec"]))].copy()
    sub["label"] = sub["whole_curve_model"] + "\n" + sub["model"]
    sub = sub.sort_values(["model", "auc"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    colors = np.where(sub["model"].eq("aec_only"), "#7b3294", "#008837")
    ax.barh(np.arange(len(sub)), sub["auc"], color=colors)
    ax.axvline(0.834521, color="black", ls="--", lw=1.2, label="clinical external AUC")
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.86)
    ax.set_xlabel("Sinchon external AUC")
    ax.set_title("Whole-curve visual models: external AUC", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_fpca_components(g: Dataset, s: Dataset, out_path: Path) -> None:
    """4채널 전체 행렬에 PCA를 적용해 상위 4개 주성분이 각 채널에서 어떤 모양을 갖는지 4x4 그리드로 그려 PNG로 저장 (FPCA가 무엇을 포착하는지 해석용)."""
    ch_g, ch_s = make_channels_for_fold(g, s, None)
    xg = full_matrix(ch_g)
    pca = make_pipeline(StandardScaler(), PCA(n_components=4, random_state=SEED))
    pcs = pca.fit(xg).named_steps["pca"].components_
    xs = np.arange(1, 129)
    fig, axes = plt.subplots(4, 4, figsize=(13, 9), sharex=True)
    names = ["curve", "residual", "slope", "curvature"]
    for pc in range(4):
        vec = pcs[pc]
        for ch in range(4):
            ax = axes[pc, ch]
            ax.plot(xs, vec[ch * 128 : (ch + 1) * 128], color="#333333")
            ax.axhline(0, color="0.7", lw=0.8, ls="--")
            ax.set_title(f"PC{pc + 1}: {names[ch]}", fontsize=9)
            ax.grid(alpha=0.18)
    fig.suptitle("FPCA/PCA components from smoothed, normalized, company-harmonized whole-curve representation", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 곡선을 이미지로 그려 CNN에 넣는 대신, 통계학의 "함수형
    데이터 분석(FDA)" 전통 기법들 — 스플라인/FPCA/웨이블릿/shapelet/커널SVM — 로 전체 곡선을
    직접 모델링하면 임상변수 대비 추가 정보를 더 잘 뽑아내는가? — CNN이 아닌 고전적 전곡선 모델
    5종 비교):

    1. g1090/sdata를 로드하고 임상점수를 준비한다.
    2. 5가지 전곡선 모델을 각각 실행: functional_logistic(스플라인 계수+로지스틱),
       fpca_logistic(PCA+로지스틱), wavelet_logistic(Haar 웨이블릿+L1로지스틱),
       shapelet_prototype(구간별 프로토타입 거리 특징), kernel_svm(PCA+RBF SVM).
       각 모델은 5-fold로 train OOF 점수와 외부 예측을 만들되, 폴드마다 회사보정 채널을 다시
       만들어 데이터 유출을 방지한다.
    3. evaluate_scores로 각 모델의 AEC 점수를 임상점수와 스택해 임상단독/AEC단독/결합 3개 모델의
       성능(AUC 등)을 비교하고, 5개 민감도 목표별 하향조정 분석까지 계산.
    4. 5개 모델의 AUC 결과, 하향조정 상세·요약, 환자별 점수, 학습로그를 각각 CSV로 저장하고,
       shapelet 모델이 선택한 윈도우 목록도 별도 CSV로 저장.
    5. 외부 AUC 비교 막대그래프와, FPCA 주성분이 각 채널에서 어떤 모양을 갖는지 보여주는 그래프를 PNG로 저장.
    6. 외부 AUC와 하향조정 요약을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_path = WORK_DATA_DIR / "g1090.xlsx" if (WORK_DATA_DIR / "g1090.xlsx").exists() else DATA_DIR / "g1090.xlsx"
    s_path = WORK_DATA_DIR / "sdata.xlsx" if (WORK_DATA_DIR / "sdata.xlsx").exists() else DATA_DIR / "sdata.xlsx"
    g = load_dataset(g_path)
    s = load_dataset(s_path)
    cg, cs, _ = clinical_scores({"meta": g.meta, "raw": g.raw, "norm": row_norm(g.raw), "y": g.y}, {"meta": s.meta, "raw": s.raw, "norm": row_norm(s.raw), "y": s.y})

    model_names = ["functional_logistic", "fpca_logistic", "wavelet_logistic", "shapelet_prototype", "kernel_svm"]
    metric_rows = []
    deesc_rows = []
    score_frames = []
    logs = []
    shapelet_selected = []
    for model_name in model_names:
        print(f"\n=== {model_name} ===", flush=True)
        if model_name == "shapelet_prototype":
            aec_g, aec_s, model_logs, selected = run_shapelet_model(g, s)
            shapelet_selected.append(selected)
        else:
            aec_g, aec_s, model_logs = run_tabular_model(model_name, g, s)
        rows, drows, scores = evaluate_scores(model_name, aec_g, aec_s, g, s, cg, cs)
        metric_rows.extend(rows)
        deesc_rows.extend(drows)
        score_frames.append(scores)
        logs.extend(model_logs)
        print(pd.DataFrame(rows)[["dataset", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot"]].to_string(index=False))

    metrics = pd.DataFrame(metric_rows)
    deesc = pd.DataFrame(deesc_rows)
    scores = pd.concat(score_frames, ignore_index=True)
    deesc_summary = summarize_deesc(deesc)
    metrics.to_csv(OUT_DIR / "whole_curve_model_auc_metrics.csv", index=False)
    deesc.to_csv(OUT_DIR / "whole_curve_model_deescalation_details.csv", index=False)
    deesc_summary.to_csv(OUT_DIR / "whole_curve_model_deescalation_summary.csv", index=False)
    scores.to_csv(OUT_DIR / "whole_curve_model_scores.csv", index=False)
    pd.DataFrame(logs).to_csv(OUT_DIR / "whole_curve_model_training_log.csv", index=False)
    if shapelet_selected:
        pd.concat(shapelet_selected, ignore_index=True).to_csv(OUT_DIR / "shapelet_selected_windows.csv", index=False)
    plot_model_auc(metrics, OUT_DIR / "whole_curve_external_auc_summary.png")
    plot_fpca_components(g, s, OUT_DIR / "fpca_components.png")

    print("\nSINCHON EXTERNAL AUC")
    print(
        metrics[metrics["dataset"].eq("Sinchon external")][
            ["whole_curve_model", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot", "sensitivity", "specificity", "accuracy"]
        ].to_string(index=False)
    )
    print("\nDE-ESCALATION SUMMARY")
    print(deesc_summary[deesc_summary["dataset"].eq("Sinchon external")].to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스플라인/FPCA/웨이블릿/shapelet/커널SVM 5종 전곡선 모델을 학습해 임상변수 대비 추가 정보를
    # 얼마나 뽑아내는지 비교하는 파이프라인을 실행한다.
    main()
