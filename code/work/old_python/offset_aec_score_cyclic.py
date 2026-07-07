from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, clinical_matrix, load_dataset  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "offset_aec_score_cyclic"
SEED = 20260629


def sigmoid(x: np.ndarray) -> np.ndarray:
    """로짓 값을 0~1 확률로 변환 (오버플로 방지를 위해 -40~40으로 클리핑)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_clinical(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """임상 변수 전용 로지스틱 회귀 모델(정규화 거의 없음)을 학습."""
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)
    model.fit(x, y)
    return model


def as_curves(aec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """결합 AEC 배열을 a128 곡선과 crop 곡선으로 분리."""
    return aec[:, :128], aec[:, 128:]


def fixed_basis_features(aec: np.ndarray) -> np.ndarray:
    """DCT 저주파 계수, 8구간 평균·기울기, 거칠기/곡률, 초반-후반 차이 등 위치 고정 기저 특징을 추출."""
    feats = []
    for mat in as_curves(aec):
        coeff = dct(mat, type=2, norm="ortho", axis=1)[:, 1:13]
        edges = np.linspace(0, 128, 9).astype(int)
        seg_means = []
        seg_slopes = []
        for a, b in zip(edges[:-1], edges[1:]):
            block = mat[:, a:b]
            seg_means.append(block.mean(axis=1))
            pos = np.linspace(-1, 1, block.shape[1])
            denom = np.sum(pos**2) or 1.0
            seg_slopes.append(((block - block.mean(axis=1, keepdims=True)) @ pos) / denom)
        seg_means = np.column_stack(seg_means)
        seg_slopes = np.column_stack(seg_slopes)
        rough = np.abs(np.diff(mat, axis=1)).mean(axis=1)[:, None]
        curvature = np.abs(np.diff(mat, n=2, axis=1)).mean(axis=1)[:, None]
        early_late = (seg_means[:, -1] - seg_means[:, 0])[:, None]
        mid_tail = (seg_means[:, 6:8].mean(axis=1) - seg_means[:, 3:5].mean(axis=1))[:, None]
        feats.append(np.column_stack([coeff, seg_means, seg_slopes, rough, curvature, early_late, mid_tail]))
    return np.column_stack(feats)


def fft_lag_invariant_features(aec: np.ndarray) -> np.ndarray:
    """위치(위상) 이동에 덜 민감하도록 FFT 진폭 스펙트럼과 여러 지연(lag)에서의 자기상관 값을 특징으로 추출."""
    feats = []
    for mat in as_curves(aec):
        fft = np.fft.rfft(mat, axis=1)
        amp = np.abs(fft[:, 1:17]) / mat.shape[1]
        power_ratio = amp[:, :4].sum(axis=1, keepdims=True) / (amp.sum(axis=1, keepdims=True) + 1e-8)
        autocorr_feats = []
        centered = mat - mat.mean(axis=1, keepdims=True)
        denom = np.sum(centered**2, axis=1) + 1e-8
        for lag in [1, 2, 4, 8, 16, 32, 48, 64]:
            rolled = np.roll(centered, shift=lag, axis=1)
            autocorr_feats.append(np.sum(centered * rolled, axis=1) / denom)
        autocorr_feats = np.column_stack(autocorr_feats)
        feats.append(np.column_stack([amp, power_ratio, autocorr_feats]))
    return np.column_stack(feats)


@dataclass
class CyclicTemplate:
    """환자군(case)-대조군(control) 평균 곡선 차이로 만든, 원형 이동(circular shift) 상관 비교용 템플릿."""

    template_a128: np.ndarray
    template_crop: np.ndarray
    max_lag: int = 32


def fit_cyclic_template(aec: np.ndarray, y: np.ndarray, max_lag: int = 32) -> CyclicTemplate:
    """라벨 y의 양성(case)-음성(control) 평균 곡선 차이를 정규화해 원형 상관용 템플릿을 만듦."""
    a128, crop = as_curves(aec)
    templates = []
    for mat in [a128, crop]:
        case = mat[y == 1].mean(axis=0)
        ctrl = mat[y == 0].mean(axis=0)
        t = case - ctrl
        t = t - t.mean()
        norm = np.linalg.norm(t)
        if norm == 0 or not np.isfinite(norm):
            norm = 1.0
        templates.append(t / norm)
    return CyclicTemplate(templates[0], templates[1], max_lag=max_lag)


def cyclic_template_features(aec: np.ndarray, tpl: CyclicTemplate) -> np.ndarray:
    """템플릿을 -max_lag~+max_lag만큼 원형 이동시키며 각 곡선과의 상관을 계산해, 위치 0 상관/최대·최소 상관/그 위치(lag) 등을 특징으로 추출."""
    out = []
    lags = np.arange(-tpl.max_lag, tpl.max_lag + 1)
    for mat, template in [(as_curves(aec)[0], tpl.template_a128), (as_curves(aec)[1], tpl.template_crop)]:
        centered = mat - mat.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(centered, axis=1) + 1e-8
        corrs = []
        for lag in lags:
            rolled_template = np.roll(template, lag)
            corrs.append((centered @ rolled_template) / denom)
        corrs = np.column_stack(corrs)
        max_idx = np.argmax(corrs, axis=1)
        min_idx = np.argmin(corrs, axis=1)
        zero_idx = int(np.where(lags == 0)[0][0])
        out.append(
            np.column_stack(
                [
                    corrs[:, zero_idx],
                    corrs.max(axis=1),
                    corrs.min(axis=1),
                    lags[max_idx] / tpl.max_lag,
                    lags[min_idx] / tpl.max_lag,
                    corrs.max(axis=1) - corrs[:, zero_idx],
                    corrs.std(axis=1),
                ]
            )
        )
    return np.column_stack(out)


def clinical_offset_ridge_fit(x: np.ndarray, y: np.ndarray, offset: np.ndarray, lam: float) -> np.ndarray:
    """임상 로짓 점수를 오프셋(고정)으로 두고, AEC 특징에 대해서만 L2 정규화 로지스틱 계수를 L-BFGS로 학습 (임상 점수는 그대로 두고 AEC가 그 위에 무엇을 더하는지만 추정)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)

    def obj(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + x @ beta
        p = sigmoid(eta)
        loss = -np.sum(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        loss += 0.5 * lam * float(beta @ beta)
        grad = x.T @ (p - y) + lam * beta
        return loss, grad

    beta0 = np.zeros(x.shape[1], dtype=float)
    res = optimize.minimize(lambda b: obj(b)[0], beta0, jac=lambda b: obj(b)[1], method="L-BFGS-B", options={"maxiter": 1000})
    return res.x


def choose_lambda(x: np.ndarray, y: np.ndarray, offset: np.ndarray, lambdas: list[float]) -> tuple[float, pd.DataFrame]:
    """4-fold 교차검증으로 여러 정규화 강도 lambda 후보의 로그손실을 비교해 최적값을 선택."""
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 11)
    rows = []
    for lam in lambdas:
        probs = np.zeros(len(y), dtype=float)
        scores = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in skf.split(x, y):
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x[tr_idx])
            xva = scaler.transform(x[va_idx])
            beta = clinical_offset_ridge_fit(xtr, y[tr_idx], offset[tr_idx], lam)
            scores[va_idx] = xva @ beta
            probs[va_idx] = sigmoid(offset[va_idx] + scores[va_idx])
        rows.append(
            {
                "lambda": lam,
                "cv_log_loss": float(log_loss(y, probs)),
                "cv_auc_combined": float(roc_auc_score(y, offset + scores)),
                "cv_auc_aec_score": float(roc_auc_score(y, scores)),
            }
        )
    df = pd.DataFrame(rows)
    best = df.sort_values(["cv_log_loss", "lambda"], ascending=[True, True]).iloc[0]
    return float(best["lambda"]), df


def lrt_add_score(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray) -> dict:
    """임상점수만 넣은 모델과 AEC점수까지 넣은 모델의 우도비검정(LRT) 카이제곱/p값/계수/오즈비/Wald p값을 계산."""
    c = (clinical_score - clinical_score.mean()) / (clinical_score.std() or 1.0)
    a = (aec_score - aec_score.mean()) / (aec_score.std() or 1.0)
    x0 = sm.add_constant(c, has_constant="add")
    x1 = sm.add_constant(np.column_stack([c, a]), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    chi2 = 2 * (m1.llf - m0.llf)
    return {
        "chi2": float(chi2),
        "p": float(stats.chi2.sf(chi2, 1)),
        "beta": float(m1.params[2]),
        "or_per_sd": float(np.exp(m1.params[2])),
        "wald_p": float(m1.pvalues[2]),
    }


def transform_features(
    variant: str,
    train_aec: np.ndarray,
    apply_aec: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """variant 이름에 따라 fixed_basis/fft_lag_invariant/cyclic_template 특징(또는 그 조합)을 train/apply 데이터 양쪽에 동일하게 적용."""
    if variant == "fixed_basis":
        return fixed_basis_features(train_aec), fixed_basis_features(apply_aec)
    if variant == "fft_lag_invariant":
        return fft_lag_invariant_features(train_aec), fft_lag_invariant_features(apply_aec)
    if variant == "fixed_plus_fft":
        return (
            np.column_stack([fixed_basis_features(train_aec), fft_lag_invariant_features(train_aec)]),
            np.column_stack([fixed_basis_features(apply_aec), fft_lag_invariant_features(apply_aec)]),
        )
    if variant == "cyclic_template":
        tpl = fit_cyclic_template(train_aec, y_train, max_lag=32)
        return cyclic_template_features(train_aec, tpl), cyclic_template_features(apply_aec, tpl)
    if variant == "fixed_plus_cyclic_template":
        tpl = fit_cyclic_template(train_aec, y_train, max_lag=32)
        return (
            np.column_stack([fixed_basis_features(train_aec), cyclic_template_features(train_aec, tpl)]),
            np.column_stack([fixed_basis_features(apply_aec), cyclic_template_features(apply_aec, tpl)]),
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_variant(
    variant: str,
    train: dict,
    test: dict,
    xclin_tr: np.ndarray,
    xclin_te: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    lambdas: list[float],
) -> dict:
    """한 특징 변형(variant)에 대해 5-fold로 임상 오프셋+AEC 릿지 모델을 학습해 train OOF 점수를 만들고,
    전체 train으로 재학습한 모델로 외부 데이터 점수까지 구해 결과 딕셔너리로 반환."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clinical_oof = np.zeros(len(ytr), dtype=float)
    aec_oof = np.zeros(len(ytr), dtype=float)
    combined_oof = np.zeros(len(ytr), dtype=float)
    chosen_lambdas = []

    for fold_id, (tr_idx, va_idx) in enumerate(outer.split(xclin_tr, ytr)):
        clin = fit_clinical(xclin_tr[tr_idx], ytr[tr_idx])
        c_tr = clin.decision_function(xclin_tr[tr_idx])
        c_va = clin.decision_function(xclin_tr[va_idx])

        f_tr, f_va = transform_features(variant, train["aec"][tr_idx], train["aec"][va_idx], ytr[tr_idx])
        scaler = StandardScaler()
        f_tr_s = scaler.fit_transform(f_tr)
        f_va_s = scaler.transform(f_va)
        lam, _ = choose_lambda(f_tr_s, ytr[tr_idx], c_tr, lambdas)
        beta = clinical_offset_ridge_fit(f_tr_s, ytr[tr_idx], c_tr, lam)
        clinical_oof[va_idx] = c_va
        aec_oof[va_idx] = f_va_s @ beta
        combined_oof[va_idx] = c_va + aec_oof[va_idx]
        chosen_lambdas.append(lam)

    clin_full = fit_clinical(xclin_tr, ytr)
    clinical_ext = clin_full.decision_function(xclin_te)
    clinical_train_full = clin_full.decision_function(xclin_tr)
    f_tr_full, f_te = transform_features(variant, train["aec"], test["aec"], ytr)
    scaler_full = StandardScaler()
    f_tr_full_s = scaler_full.fit_transform(f_tr_full)
    f_te_s = scaler_full.transform(f_te)
    lam_full, cv_table = choose_lambda(f_tr_full_s, ytr, clinical_train_full, lambdas)
    beta_full = clinical_offset_ridge_fit(f_tr_full_s, ytr, clinical_train_full, lam_full)
    aec_ext = f_te_s @ beta_full
    combined_ext = clinical_ext + aec_ext

    cv_table.to_csv(OUT_DIR / f"{variant}_lambda_cv.csv", index=False)
    return {
        "variant": variant,
        "n_features": int(f_tr_full.shape[1]),
        "oof_lambdas": chosen_lambdas,
        "full_lambda": lam_full,
        "clinical_oof": clinical_oof,
        "aec_oof": aec_oof,
        "combined_oof": combined_oof,
        "clinical_ext": clinical_ext,
        "aec_ext": aec_ext,
        "combined_ext": combined_ext,
    }


def metrics(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray, combined_score: np.ndarray) -> dict:
    """임상/AEC/결합 점수 각각의 AUC·AP와, 결합 점수의 로그손실·Brier, 임상 대비 delta AUC/AP를 계산."""
    return {
        "clinical_auc": float(roc_auc_score(y, clinical_score)),
        "aec_score_auc": float(roc_auc_score(y, aec_score)),
        "combined_auc": float(roc_auc_score(y, combined_score)),
        "delta_auc": float(roc_auc_score(y, combined_score) - roc_auc_score(y, clinical_score)),
        "clinical_ap": float(average_precision_score(y, clinical_score)),
        "aec_score_ap": float(average_precision_score(y, aec_score)),
        "combined_ap": float(average_precision_score(y, combined_score)),
        "delta_ap": float(average_precision_score(y, combined_score) - average_precision_score(y, clinical_score)),
        "combined_log_loss": float(log_loss(y, sigmoid(combined_score))),
        "combined_brier": float(brier_score_loss(y, sigmoid(combined_score))),
    }


def bootstrap_delta_auc(y: np.ndarray, clinical: np.ndarray, combined: np.ndarray, n_boot: int = 2000) -> dict:
    """결합 점수와 임상 점수의 AUC 차이를 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 99)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], combined[idx]) - roc_auc_score(y[idx], clinical[idx]))
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "ci2.5": float(np.quantile(vals, 0.025)),
        "ci97.5": float(np.quantile(vals, 0.975)),
        "p_le_0": float(np.mean(vals <= 0)),
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: AEC 곡선의 "위치"가 아니라 "위상 이동에 둔감한" 형태로
    특징을 표현해도(원형 이동/주파수 기반) 임상변수 대비 추가 정보가 유지되는가?):

    1. train(g1090)/test(sdata)를 로드하고 임상 설계행렬을 만든다.
    2. 5개 특징 변형(variant)을 준비한다: fixed_basis(DCT+구간특징, 위치 고정), fft_lag_invariant
       (푸리에 진폭+자기상관, 위치이동에 덜 민감), 두 개를 합친 fixed_plus_fft, cyclic_template
       (train에서 만든 case-control 평균차 템플릿과의 원형 상관), fixed_plus_cyclic_template.
    3. 각 variant마다 run_variant를 실행: 5-fold 내부에서 임상 로짓 점수를 오프셋으로 고정하고
       clinical_offset_ridge_fit으로 AEC 특징에 대해서만 릿지 로지스틱을 학습해(임상점수 자체는
       건드리지 않고 그 위에 AEC가 더하는 부분만 추정), train OOF와 외부 점수를 모두 구한다.
       lambda는 choose_lambda로 내부 교차검증 로그손실 기준 튜닝한다.
    4. 각 variant에 대해 metrics로 임상/AEC/결합 점수의 AUC·AP·로그손실·Brier를 계산하고,
       lrt_add_score로 임상점수 대비 AEC점수의 조건부 연관성(LRT)을, bootstrap_delta_auc로
       결합-임상 간 AUC 차이의 신뢰구간을 추정한다.
    5. 5개 variant를 외부 delta AUC 기준으로 정렬한 요약표를 CSV로 저장하고, "원형 이동(cyclic
       wrapping)은 수학적 민감도 분석이지 해부학적 가정이 아니며, 성능이 비슷하면 원래 z축
       위치 기반 해석을 우선해야 한다"는 주의사항과 함께 결과를 JSON으로 저장한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]
    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    variants = ["fixed_basis", "fft_lag_invariant", "fixed_plus_fft", "cyclic_template", "fixed_plus_cyclic_template"]
    lambdas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    rows = []
    full_results = {}
    for variant in variants:
        print(f"Running {variant}", flush=True)
        res = run_variant(variant, train, test, xclin_tr, xclin_te, ytr, yte, lambdas)
        full_results[variant] = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        train_metrics = metrics(ytr, res["clinical_oof"], res["aec_oof"], res["combined_oof"])
        ext_metrics = metrics(yte, res["clinical_ext"], res["aec_ext"], res["combined_ext"])
        row = {
            "variant": variant,
            "n_features": res["n_features"],
            "full_lambda": res["full_lambda"],
            **{f"train_oof_{k}": v for k, v in train_metrics.items()},
            **{f"external_{k}": v for k, v in ext_metrics.items()},
            "train_lrt_p": lrt_add_score(ytr, res["clinical_oof"], res["aec_oof"])["p"],
            "train_lrt_beta": lrt_add_score(ytr, res["clinical_oof"], res["aec_oof"])["beta"],
            "external_lrt_p": lrt_add_score(yte, res["clinical_ext"], res["aec_ext"])["p"],
            "external_lrt_beta": lrt_add_score(yte, res["clinical_ext"], res["aec_ext"])["beta"],
            "external_delta_auc_boot_mean": bootstrap_delta_auc(yte, res["clinical_ext"], res["combined_ext"])["mean"],
            "external_delta_auc_boot_ci2.5": bootstrap_delta_auc(yte, res["clinical_ext"], res["combined_ext"])["ci2.5"],
            "external_delta_auc_boot_ci97.5": bootstrap_delta_auc(yte, res["clinical_ext"], res["combined_ext"])["ci97.5"],
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("external_delta_auc", ascending=False)
    summary.to_csv(OUT_DIR / "offset_aec_cyclic_summary_table.csv", index=False)
    result = {
        "model": "clinical-offset ridge logistic AEC score",
        "loss": "sum logistic_loss(y, clinical_logit + AEC_features @ beta) + 0.5 * lambda * ||beta||^2",
        "cyclic_lagging_interpretation": "fft_lag_invariant uses cyclic-shift invariant amplitudes/autocorrelation; cyclic_template computes max circular correlation against train-derived case-control AEC templates.",
        "summary_table": summary.to_dict(orient="records"),
        "caveat": "Cyclic wrapping is a mathematical sensitivity analysis, not an anatomical assumption; z-axis location should remain primary if model performance is similar.",
    }
    with open(OUT_DIR / "offset_aec_cyclic_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print(OUT_DIR / "offset_aec_cyclic_summary.json")


if __name__ == "__main__":
    main()
