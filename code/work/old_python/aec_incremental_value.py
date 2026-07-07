from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data_cache"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec_incremental_value"
RNG = np.random.default_rng(20260629)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """로짓 값을 0~1 확률로 변환 (오버플로 방지를 위해 입력을 -40~40으로 클리핑)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """실제 라벨 y와 예측 점수 s로부터 ROC AUC를 순위(rank) 기반으로 직접 계산."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """예측 점수를 내림차순 정렬한 뒤 정밀도-재현율 곡선의 평균 정밀도(AP)를 계산."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    rank = np.arange(1, len(y) + 1)
    precision = tp / rank
    return float((precision * yy).sum() / n_pos)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """이진 라벨 y와 예측 확률 p 사이의 평균 로그 손실(cross-entropy)을 계산."""
    p = np.clip(np.asarray(p).astype(float), 1e-9, 1 - 1e-9)
    y = np.asarray(y).astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """실제 라벨과 예측 확률 간의 평균 제곱오차(Brier score)를 계산."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((y - p) ** 2))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-8,
    penalize_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """뉴턴-랩슨(IRLS) 방식으로 L2 정규화 로지스틱 회귀를 직접 학습하고, 계수와 공분산 행렬을 반환."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.ones(x.shape[1], dtype=float) * float(l2)
    if not penalize_intercept:
        penalty[0] = 0.0
    cov = np.full((x.shape[1], x.shape[1]), np.nan)
    for _ in range(max_iter):
        eta = x @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1 - p), 1e-8, None)
        grad = x.T @ (p - y) + penalty * beta
        hess = (x.T * w) @ x + np.diag(penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = x @ beta
    p = sigmoid(eta)
    w = np.clip(p * (1 - p), 1e-8, None)
    hess = (x.T * w) @ x + np.diag(penalty)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    return beta, cov


def loglik(y: np.ndarray, p: np.ndarray) -> float:
    """이진 라벨과 예측 확률로부터 로그우도(log-likelihood) 총합을 계산."""
    p = np.clip(np.asarray(p).astype(float), 1e-9, 1 - 1e-9)
    y = np.asarray(y).astype(float)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def chi2_df1_pvalue(stat: float) -> float:
    """자유도 1인 카이제곱 통계량에 대한 p-value를 오차함수(erfc)로 근사 계산."""
    return float(math.erfc(math.sqrt(max(stat, 0.0) / 2.0)))


def stratified_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """클래스 비율을 유지하며 데이터를 k개의 교차검증 폴드 인덱스로 분할."""
    y = np.asarray(y).astype(int)
    folds = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        RNG.shuffle(idx)
        for i, v in enumerate(idx):
            folds[i % k].append(int(v))
    return [np.array(sorted(f), dtype=int) for f in folds]


def standardize_train_apply(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train 데이터의 평균/표준편차로 train·test를 함께 표준화(z-score)."""
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def make_clinical(meta_train: pd.DataFrame, meta_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """메타데이터(나이, 성별, 키, 몸무게)로부터 표준화된 임상 변수 설계행렬(절편 포함)을 생성."""
    def build(meta: pd.DataFrame) -> np.ndarray:
        sex = meta["PatientSex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).to_numpy()
        arr = np.column_stack(
            [
                meta["PatientAge"].astype(float).to_numpy(),
                sex,
                meta["Height"].astype(float).to_numpy(),
                meta["Weight"].astype(float).to_numpy(),
            ]
        )
        return arr

    xtr_raw = build(meta_train)
    xte_raw = build(meta_test)
    xtr_z, xte_z, _, _ = standardize_train_apply(xtr_raw, xte_raw)
    xtr = np.column_stack([np.ones(len(xtr_z)), xtr_z])
    xte = np.column_stack([np.ones(len(xte_z)), xte_z])
    return xtr, xte, ["intercept", "age_z", "male_z", "height_z", "weight_z"]


def aec_columns(df: pd.DataFrame) -> list[str]:
    """데이터프레임에서 'aec_'로 시작하는 컬럼명을 뽑아 번호 순서로 정렬."""
    cols = [c for c in df.columns if str(c).startswith("aec_")]
    return sorted(cols, key=lambda x: int(str(x).split("_")[1]))


def resample_row(vals: np.ndarray, n: int = 128) -> np.ndarray:
    """길이가 제각각인 1차원 신호를 선형보간으로 길이 n(기본 128)으로 리샘플링."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.zeros(n, dtype=float)
    if len(vals) == 1:
        return np.full(n, vals[0], dtype=float)
    old = np.linspace(0.0, 1.0, len(vals))
    new = np.linspace(0.0, 1.0, n)
    return np.interp(new, old, vals)


def normalized_shape_matrix(df: pd.DataFrame, resample: bool) -> tuple[np.ndarray, list[int]]:
    """각 행의 AEC 신호를 (옵션에 따라 리샘플링 후) 자기 평균으로 정규화한 행렬과, 유효 길이 목록을 반환."""
    cols = aec_columns(df)
    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out = []
    lengths = []
    for row in mat:
        finite = row[np.isfinite(row)]
        lengths.append(int(len(finite)))
        if resample:
            r = resample_row(finite, 128)
        else:
            r = row[:128].astype(float)
            if np.isnan(r).any():
                r = resample_row(finite, 128)
        mean = np.nanmean(r)
        if not np.isfinite(mean) or abs(mean) < 1e-8:
            mean = 1.0
        out.append(r / mean - 1.0)
    return np.asarray(out, dtype=float), lengths


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """엑셀 파일(metadata/aec_128/aec_cropped 시트)을 읽어 라벨(y, 저근감소증 여부)과 AEC 특징 행렬을 구성."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    a128 = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl")
    crop = pd.read_excel(path, sheet_name="aec_cropped", engine="openpyxl")
    meta = meta.copy()
    meta["PatientID"] = meta["PatientID"].astype(str)
    a128["PatientID"] = a128["PatientID"].astype(str)
    crop["PatientID"] = crop["PatientID"].astype(str)
    a128 = a128.set_index("PatientID").loc[meta["PatientID"]].reset_index()
    crop = crop.set_index("PatientID").loc[meta["PatientID"]].reset_index()
    smi = meta["TAMA"].astype(float).to_numpy() / (meta["Height"].astype(float).to_numpy() / 100.0) ** 2
    male = meta["PatientSex"].astype(str).str.upper().to_numpy() == "M"
    y = np.where(male, smi < 45.4, smi < 34.4).astype(int)
    x128, _ = normalized_shape_matrix(a128, resample=False)
    xcrop, lengths = normalized_shape_matrix(crop, resample=True)
    xaec = np.column_stack([x128, xcrop])
    return meta, y, xaec, np.asarray(lengths)


def cv_lambda(x: np.ndarray, y: np.ndarray, lambdas: list[float], k: int = 5) -> tuple[float, list[dict]]:
    """여러 L2 정규화 강도(lambda) 후보를 교차검증 로그손실 기준으로 비교해 최적값을 선택."""
    folds = stratified_folds(y, k=k)
    rows = []
    for lam in lambdas:
        fold_losses = []
        fold_aucs = []
        for test_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            beta, _ = fit_logistic(x[train_idx], y[train_idx], l2=lam)
            p = sigmoid(x[test_idx] @ beta)
            fold_losses.append(log_loss(y[test_idx], p))
            fold_aucs.append(auc_score(y[test_idx], p))
        rows.append(
            {
                "lambda": lam,
                "cv_log_loss": float(np.mean(fold_losses)),
                "cv_auc": float(np.mean(fold_aucs)),
            }
        )
    best = min(rows, key=lambda r: r["cv_log_loss"])
    return float(best["lambda"]), rows


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC, average precision, log loss, Brier score를 한 번에 묶어서 반환."""
    return {
        "auc": auc_score(y, p),
        "average_precision": average_precision(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
    }


def bootstrap_deltas(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, n_boot: int = 2000) -> dict:
    """두 모델(p0 vs p1)의 성능 지표 차이를 부트스트랩 리샘플링으로 신뢰구간과 함께 추정."""
    y = np.asarray(y)
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append(
            [
                auc_score(y[idx], p1[idx]) - auc_score(y[idx], p0[idx]),
                average_precision(y[idx], p1[idx]) - average_precision(y[idx], p0[idx]),
                log_loss(y[idx], p0[idx]) - log_loss(y[idx], p1[idx]),
                brier(y[idx], p0[idx]) - brier(y[idx], p1[idx]),
            ]
        )
    arr = np.asarray(rows)
    names = ["delta_auc", "delta_average_precision", "log_loss_reduction", "brier_reduction"]
    out = {}
    for i, name in enumerate(names):
        vals = arr[:, i]
        out[name] = {
            "mean": float(np.mean(vals)),
            "ci2.5": float(np.quantile(vals, 0.025)),
            "ci97.5": float(np.quantile(vals, 0.975)),
            "p_le_0": float(np.mean(vals <= 0)),
        }
    return out


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.90) -> float:
    """양성군 점수 분포에서 목표 민감도(기본 90%)를 만족하는 최소 임계값을 찾음."""
    pos_scores = np.sort(np.asarray(p)[np.asarray(y) == 1])
    if len(pos_scores) == 0:
        return float("nan")
    q = max(0, int(math.floor((1 - target) * len(pos_scores))))
    return float(pos_scores[q])


def confusion_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """주어진 임계값에서 TP/FP/FN/TN과 민감도·특이도·PPV·NPV를 계산."""
    pred = np.asarray(p) >= thr
    y = np.asarray(y).astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv}


def pca_train_apply(xtr: np.ndarray, xte: np.ndarray, n_pc: int) -> tuple[np.ndarray, np.ndarray]:
    """train 데이터로 PCA(SVD)를 학습해 train·test를 상위 n_pc개 주성분으로 투영."""
    mu = np.mean(xtr, axis=0)
    xc = xtr - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:n_pc].T
    return xc @ comp, (xte - mu) @ comp


def permutation_lrt(
    y: np.ndarray,
    clinical_score: np.ndarray,
    aec_score: np.ndarray,
    observed_stat: float,
    n_perm: int = 2000,
) -> float:
    # Tests whether the locked AEC score adds beyond locked clinical score in the external cohort.
    """AEC 점수를 무작위로 섞어 우도비 검정 통계량의 순열분포를 만들고, 관측 통계량의 순열 p-value를 계산."""
    x0 = np.column_stack([np.ones(len(y)), clinical_score])
    stats = []
    for _ in range(n_perm):
        perm_score = RNG.permutation(aec_score)
        x1 = np.column_stack([np.ones(len(y)), clinical_score, perm_score])
        b0, _ = fit_logistic(x0, y, l2=1e-8)
        b1, _ = fit_logistic(x1, y, l2=1e-8)
        stat = 2 * (loglik(y, sigmoid(x1 @ b1)) - loglik(y, sigmoid(x0 @ b0)))
        stats.append(stat)
    return float((np.sum(np.asarray(stats) >= observed_stat) + 1) / (len(stats) + 1))


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름:

    1. load_dataset으로 g1090.xlsx(학습)와 sdata.xlsx(외부검증)를 읽어 라벨 y(저근감소증 여부)와
       정규화된 AEC 곡선(128구간 + 크롭 128구간)을 만든다.
    2. make_clinical으로 나이/성별/키/몸무게 임상 설계행렬(절편 포함, 표준화)을 만든다.
    3. 임상변수만으로 fit_logistic(직접 구현한 뉴턴법 로지스틱 회귀)을 학습해 "임상 베이스라인"
       확률(p_clin_g, p_clin_s)을 구한다.
    4. AEC 신호를 지도학습 없이(PCA만으로) 5/10/20개 주성분으로 축약해 임상모델에 추가했을 때
       train 우도비(LRT)와 외부 AUC/로그손실이 얼마나 개선되는지 확인한다 (가장 보수적인 비교군).
    5. AEC 전체 차원을 그대로 써서 능선회귀(cv_lambda로 λ 교차검증) 로지스틱을 학습한 "AEC 단독 모델"과,
       임상변수+AEC를 합친 "결합 모델"을 각각 train에서만 튜닝한다 (외부 데이터는 튜닝에 관여 안 함).
    6. 외부 데이터에서 임상점수와 AEC점수를 각각 z-표준화한 뒤, "임상점수만" 모델과 "임상점수+AEC점수"
       모델의 우도비검정(LRT)을 수행해 AEC가 임상변수를 통제한 후에도 독립적 연관성이 있는지 확인하고,
       permutation_lrt로 라벨을 섞은 순열분포 대비 p-value도 계산한다.
    7. train에서 90% 민감도를 만족하는 임계값(threshold_for_sensitivity)을 정하고, 그 임계값을
       외부 데이터에 그대로 적용해(잠금 임계값) 임상 단독 vs 결합모델의 혼동행렬을 비교한다.
    8. bootstrap_deltas로 두 모델 간 AUC/AP/로그손실/Brier 차이를 2000회 재표본추출로 신뢰구간과
       함께 추정한다.
    9. 이 모든 결과(모델별 지표, 교차검증 결과, PCA 추가 결과, 임계값 기반 혼동행렬, 부트스트랩 델타)를
       요약 딕셔너리로 모아 OUT_DIR 아래 CSV 여러 개와 summary JSON 파일로 저장하고 콘솔에 출력한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_meta, y_g, xaec_g_raw, len_g = load_dataset(DATA_DIR / "g1090.xlsx")
    s_meta, y_s, xaec_s_raw, len_s = load_dataset(DATA_DIR / "sdata.xlsx")

    xclin_g, xclin_s, clin_names = make_clinical(g_meta, s_meta)
    xaec_g_z, xaec_s_z, _, _ = standardize_train_apply(xaec_g_raw, xaec_s_raw)

    # Clinical baseline.
    b_clin, cov_clin = fit_logistic(xclin_g, y_g, l2=1e-6)
    p_clin_g = sigmoid(xclin_g @ b_clin)
    p_clin_s = sigmoid(xclin_s @ b_clin)

    # Low-dimensional unsupervised AEC PC additions, a conventional nested check.
    pc_results = []
    for k_pc in [5, 10, 20]:
        pc_g, pc_s = pca_train_apply(xaec_g_z, xaec_s_z, k_pc)
        pc_g_z, pc_s_z, _, _ = standardize_train_apply(pc_g, pc_s)
        xpc_g = np.column_stack([xclin_g, pc_g_z])
        xpc_s = np.column_stack([xclin_s, pc_s_z])
        b_pc, _ = fit_logistic(xpc_g, y_g, l2=1e-6)
        p_pc_g = sigmoid(xpc_g @ b_pc)
        p_pc_s = sigmoid(xpc_s @ b_pc)
        lrt_train = 2 * (loglik(y_g, p_pc_g) - loglik(y_g, p_clin_g))
        pc_results.append(
            {
                "model": f"clinical_plus_first_{k_pc}_unsupervised_aec_pcs",
                "train_lrt_vs_clinical": float(lrt_train),
                "train_lrt_df": k_pc,
                "train_log_loss": log_loss(y_g, p_pc_g),
                "external_auc": auc_score(y_s, p_pc_s),
                "external_ap": average_precision(y_s, p_pc_s),
                "external_log_loss": log_loss(y_s, p_pc_s),
                "external_brier": brier(y_s, p_pc_s),
                "external_delta_auc_vs_clinical": auc_score(y_s, p_pc_s) - auc_score(y_s, p_clin_s),
                "external_log_loss_reduction_vs_clinical": log_loss(y_s, p_clin_s) - log_loss(y_s, p_pc_s),
            }
        )

    # Supervised ridge AEC and full clinical+AEC models tuned only in g1090.
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    xaec_design_g = np.column_stack([np.ones(len(xaec_g_z)), xaec_g_z])
    xaec_design_s = np.column_stack([np.ones(len(xaec_s_z)), xaec_s_z])
    best_aec_lam, aec_cv = cv_lambda(xaec_design_g, y_g, lambdas)
    b_aec, _ = fit_logistic(xaec_design_g, y_g, l2=best_aec_lam)
    aec_score_g = xaec_design_g @ b_aec
    aec_score_s = xaec_design_s @ b_aec
    p_aec_s = sigmoid(aec_score_s)

    xfull_g = np.column_stack([xclin_g, xaec_g_z])
    xfull_s = np.column_stack([xclin_s, xaec_s_z])
    best_full_lam, full_cv = cv_lambda(xfull_g, y_g, lambdas)
    b_full, _ = fit_logistic(xfull_g, y_g, l2=best_full_lam)
    p_full_g = sigmoid(xfull_g @ b_full)
    p_full_s = sigmoid(xfull_s @ b_full)

    # External conditional association of a locked AEC score beyond locked clinical score.
    clin_score_s = xclin_s @ b_clin
    clin_score_g = xclin_g @ b_clin
    clin_score_s_z = (clin_score_s - clin_score_g.mean()) / clin_score_g.std()
    aec_score_s_z = (aec_score_s - aec_score_g.mean()) / aec_score_g.std()
    x0_s = np.column_stack([np.ones(len(y_s)), clin_score_s_z])
    x1_s = np.column_stack([np.ones(len(y_s)), clin_score_s_z, aec_score_s_z])
    b0_s, _ = fit_logistic(x0_s, y_s, l2=1e-8)
    b1_s, cov1_s = fit_logistic(x1_s, y_s, l2=1e-8)
    ll0_s = loglik(y_s, sigmoid(x0_s @ b0_s))
    ll1_s = loglik(y_s, sigmoid(x1_s @ b1_s))
    lrt_s = 2 * (ll1_s - ll0_s)
    se_aec = math.sqrt(max(cov1_s[2, 2], 0.0))
    z_aec = b1_s[2] / se_aec if se_aec > 0 else float("nan")
    wald_p = math.erfc(abs(z_aec) / math.sqrt(2.0)) if np.isfinite(z_aec) else float("nan")
    perm_p = permutation_lrt(y_s, clin_score_s_z, aec_score_s_z, lrt_s, n_perm=2000)

    # Train-selected operating threshold at high sensitivity.
    thr_clin = threshold_for_sensitivity(y_g, p_clin_g, target=0.90)
    thr_full = threshold_for_sensitivity(y_g, p_full_g, target=0.90)
    operating = {
        "clinical_threshold_train_sens_0.90": {"threshold": thr_clin, **confusion_at(y_s, p_clin_s, thr_clin)},
        "clinical_plus_aec_threshold_train_sens_0.90": {"threshold": thr_full, **confusion_at(y_s, p_full_s, thr_full)},
    }

    summary = {
        "n_train": int(len(y_g)),
        "events_train": int(y_g.sum()),
        "prevalence_train": float(y_g.mean()),
        "n_external": int(len(y_s)),
        "events_external": int(y_s.sum()),
        "prevalence_external": float(y_s.mean()),
        "clinical_coefficients": dict(zip(clin_names, map(float, b_clin))),
        "clinical_external": metrics(y_s, p_clin_s),
        "aec_only_external": metrics(y_s, p_aec_s),
        "clinical_plus_direct_aec_external": metrics(y_s, p_full_s),
        "best_aec_lambda_by_train_cv_logloss": best_aec_lam,
        "best_full_lambda_by_train_cv_logloss": best_full_lam,
        "external_deltas_full_vs_clinical": bootstrap_deltas(y_s, p_clin_s, p_full_s),
        "external_conditional_locked_aec_score_test": {
            "model": "external y ~ locked clinical logit + locked supervised AEC logit",
            "lrt_chi_square_1df": float(lrt_s),
            "lrt_p_chi_square_1df": chi2_df1_pvalue(lrt_s),
            "permutation_p": perm_p,
            "aec_beta_per_train_sd": float(b1_s[2]),
            "aec_or_per_train_sd": float(math.exp(b1_s[2])),
            "aec_wald_z": float(z_aec),
            "aec_wald_p": float(wald_p),
            "ll_clinical_score_only": float(ll0_s),
            "ll_clinical_plus_aec_score": float(ll1_s),
        },
        "train_cv_rows_aec_only": aec_cv,
        "train_cv_rows_clinical_plus_direct_aec": full_cv,
        "unsupervised_pca_addition_results": pc_results,
        "external_operating_at_train_selected_90pct_sensitivity": operating,
        "aec_cropped_lengths": {
            "train_min": int(len_g.min()),
            "train_median": float(np.median(len_g)),
            "train_max": int(len_g.max()),
            "external_min": int(len_s.min()),
            "external_median": float(np.median(len_s)),
            "external_max": int(len_s.max()),
        },
    }

    pd.DataFrame(
        [
            {"model": "clinical", **summary["clinical_external"]},
            {"model": "aec_only_direct_ridge", **summary["aec_only_external"]},
            {"model": "clinical_plus_direct_aec_ridge", **summary["clinical_plus_direct_aec_external"]},
            *pc_results,
        ]
    ).to_csv(OUT_DIR / "incremental_model_metrics.csv", index=False)
    pd.DataFrame(aec_cv).to_csv(OUT_DIR / "train_cv_aec_only_lambda.csv", index=False)
    pd.DataFrame(full_cv).to_csv(OUT_DIR / "train_cv_clinical_plus_aec_lambda.csv", index=False)
    pd.DataFrame(pc_results).to_csv(OUT_DIR / "unsupervised_pca_nested_checks.csv", index=False)
    pd.DataFrame.from_dict(operating, orient="index").to_csv(OUT_DIR / "external_operating_90pct_train_sensitivity.csv")
    with open(OUT_DIR / "incremental_value_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
