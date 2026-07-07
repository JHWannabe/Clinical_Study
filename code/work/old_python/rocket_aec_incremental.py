from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    binary_metrics,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    oof_and_external,
    threshold_youden,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "rocket_aec"
SEED = 20260629


@dataclass
class RocketKernel:
    channels: np.ndarray
    weights: np.ndarray
    length: int
    dilation: int
    padding: int
    bias: float


class RocketTransformer:
    """A compact ROCKET-style random convolution feature extractor for 1D AEC curves."""

    def __init__(self, n_kernels: int = 3000, seed: int = SEED):
        self.n_kernels = int(n_kernels)
        self.seed = int(seed)
        self.kernels: list[RocketKernel] = []

    def fit(self, x: np.ndarray) -> "RocketTransformer":
        """길이/팽창(dilation)/채널/가중치가 무작위인 합성곱 커널 n_kernels개를 생성하고, 샘플 하나의 응답 분위수로 각 커널의 편향(bias)을 정함."""
        rng = np.random.default_rng(self.seed)
        n, n_channels, series_len = x.shape
        lengths = np.array([7, 9, 11], dtype=int)
        self.kernels = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(lengths))
            max_dilation = max(1, (series_len - 1) // (length - 1))
            dilation = int(2 ** rng.uniform(0, math.log2(max_dilation))) if max_dilation > 1 else 1
            dilation = max(1, min(dilation, max_dilation))
            padding = int(rng.choice([0, ((length - 1) * dilation) // 2]))
            n_sel = int(rng.integers(1, n_channels + 1))
            channels = np.sort(rng.choice(n_channels, size=n_sel, replace=False))
            weights = rng.normal(size=(n_sel, length)).astype(np.float32)
            weights = weights - weights.mean(axis=1, keepdims=True)
            sample_idx = int(rng.integers(0, n))
            sample = x[sample_idx : sample_idx + 1]
            conv = self._apply_kernel(sample, channels, weights, length, dilation, padding)
            if conv.size:
                bias = float(np.quantile(conv.ravel(), rng.uniform(0, 1)))
            else:
                bias = 0.0
            self.kernels.append(RocketKernel(channels, weights, length, dilation, padding, bias))
        return self

    @staticmethod
    def _apply_kernel(
        x: np.ndarray,
        channels: np.ndarray,
        weights: np.ndarray,
        length: int,
        dilation: int,
        padding: int,
    ) -> np.ndarray:
        """선택된 채널들에 대해 팽창 합성곱을 직접 계산해(길이 length, 팽창 dilation) 응답 시계열을 반환."""
        xx = x[:, channels, :]
        if padding > 0:
            xx = np.pad(xx, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        out_len = xx.shape[2] - (length - 1) * dilation
        if out_len <= 0:
            return np.zeros((x.shape[0], 1), dtype=np.float32)
        out = np.zeros((x.shape[0], out_len), dtype=np.float32)
        for c in range(len(channels)):
            for j in range(length):
                out += weights[c, j] * xx[:, c, j * dilation : j * dilation + out_len]
        return out

    def transform(self, x: np.ndarray) -> np.ndarray:
        """각 커널의 합성곱 응답에서 ROCKET 특유의 두 통계량(최댓값, 편향을 넘는 비율 PPV)을 뽑아 커널당 2개씩 특징 벡터를 구성."""
        feats = np.empty((x.shape[0], self.n_kernels * 2), dtype=np.float32)
        for i, kernel in enumerate(self.kernels):
            conv = self._apply_kernel(x, kernel.channels, kernel.weights, kernel.length, kernel.dilation, kernel.padding)
            feats[:, 2 * i] = conv.max(axis=1)
            feats[:, 2 * i + 1] = (conv > kernel.bias).mean(axis=1)
        return feats

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """fit과 transform을 이어서 실행하는 편의 메서드."""
        return self.fit(x).transform(x)


def as_two_channel(aec: np.ndarray) -> np.ndarray:
    """길이 256짜리 결합 AEC 벡터를 (a128, crop) 2채널 x 128길이 시계열 형태로 재구성."""
    return np.stack([aec[:, :128], aec[:, 128:]], axis=1).astype(np.float32)


def make_model(c: float) -> LogisticRegression:
    """L2 정규화 강도 c를 받아 클래스 균형 로지스틱 회귀(liblinear) 모델을 생성."""
    return LogisticRegression(
        C=float(c),
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )


def score_metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None = None) -> dict:
    """AUC/AP를 계산하고, 확률값(prob)이 주어지면 로그손실·Brier도 추가."""
    out = {
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }
    if prob is not None:
        pp = np.clip(prob, 1e-8, 1 - 1e-8)
        out["log_loss"] = float(log_loss(y, pp))
        out["brier"] = float(brier_score_loss(y, pp))
    return out


def tune_c(x: np.ndarray, y: np.ndarray, c_grid: list[float], label: str) -> tuple[float, pd.DataFrame]:
    """5-fold 교차검증으로 여러 정규화 강도 C 후보의 AUC/AP를 비교해 최적 C를 선택."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for c in c_grid:
        scores = np.zeros(len(y), dtype=float)
        probs = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in skf.split(x, y):
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x[tr_idx])
            xva = scaler.transform(x[va_idx])
            model = make_model(c)
            model.fit(xtr, y[tr_idx])
            scores[va_idx] = model.decision_function(xva)
            probs[va_idx] = model.predict_proba(xva)[:, 1]
        row = {"model": label, "C": c, **score_metrics(y, scores, probs)}
        rows.append(row)
    df = pd.DataFrame(rows)
    # AUC is used for selecting a ranking/de-escalation marker; loss is still the fitted objective.
    best = df.sort_values(["auc", "average_precision"], ascending=False).iloc[0]
    return float(best["C"]), df


def cross_val_scores(x: np.ndarray, y: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """고정된 C값으로 5-fold 교차검증을 돌려 train 전체에 대한 out-of-fold 점수와 확률을 만듦."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    scores = np.zeros(len(y), dtype=float)
    probs = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(x, y):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x[tr_idx])
        xva = scaler.transform(x[va_idx])
        model = make_model(c)
        model.fit(xtr, y[tr_idx])
        scores[va_idx] = model.decision_function(xva)
        probs[va_idx] = model.predict_proba(xva)[:, 1]
    return scores, probs


def fit_predict_external(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """train 전체로 모델을 학습해 외부(test) 데이터에 대한 점수와 확률을 예측."""
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr)
    xte_s = scaler.transform(xte)
    model = make_model(c)
    model.fit(xtr_s, ytr)
    return model.decision_function(xte_s), model.predict_proba(xte_s)[:, 1]


def logit_lrt(y: np.ndarray, base_score: np.ndarray, add_score: np.ndarray) -> dict:
    """기본 점수만 넣은 모델과 추가 점수까지 넣은 모델의 우도비검정(LRT)을 수행해, 추가 점수의 독립적 연관성(계수·오즈비·p값)을 계산."""
    base_z = (base_score - base_score.mean()) / (base_score.std() or 1.0)
    add_z = (add_score - add_score.mean()) / (add_score.std() or 1.0)
    x0 = sm.add_constant(base_z, has_constant="add")
    x1 = sm.add_constant(np.column_stack([base_z, add_z]), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    stat = 2 * (m1.llf - m0.llf)
    return {
        "lrt_chi2_1df": float(stat),
        "lrt_p": float(stats.chi2.sf(stat, 1)),
        "add_score_beta": float(m1.params[2]),
        "add_score_or_per_sd": float(np.exp(m1.params[2])),
        "add_score_wald_p": float(m1.pvalues[2]),
        "ll_base": float(m0.llf),
        "ll_full": float(m1.llf),
    }


def bootstrap_delta(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 2500) -> dict:
    """두 점수(a vs b)의 AUC/AP 차이를 부트스트랩 재표본추출로 신뢰구간과 함께 추정."""
    rng = np.random.default_rng(SEED + 4)
    arr = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        arr.append(
            [
                roc_auc_score(yy, score_b[idx]) - roc_auc_score(yy, score_a[idx]),
                average_precision_score(yy, score_b[idx]) - average_precision_score(yy, score_a[idx]),
            ]
        )
    arr = np.asarray(arr)
    out = {}
    for i, name in enumerate(["delta_auc", "delta_average_precision"]):
        vals = arr[:, i]
        out[name] = {
            "mean": float(vals.mean()),
            "ci2.5": float(np.quantile(vals, 0.025)),
            "ci97.5": float(np.quantile(vals, 0.975)),
            "p_le_0": float(np.mean(vals <= 0)),
        }
    return out


def decile_table(y: np.ndarray, clinical_score: np.ndarray, aec_score: np.ndarray, clinical_threshold: float) -> pd.DataFrame:
    """임상 양성 환자만 골라 AEC(ROCKET) 점수 10분위로 나누고, 분위별 표본수·이벤트수·유병률을 표로 만듦."""
    clinical_pos = clinical_score >= clinical_threshold
    df = pd.DataFrame({"y": y[clinical_pos], "aec_score": aec_score[clinical_pos]})
    df["aec_decile"] = pd.qcut(df["aec_score"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, g in df.groupby("aec_decile", observed=True):
        rows.append(
            {
                "aec_score_decile_among_clinical_positive": int(decile),
                "n": int(len(g)),
                "events": int(g["y"].sum()),
                "prevalence": float(g["y"].mean()),
                "aec_score_min": float(g["aec_score"].min()),
                "aec_score_max": float(g["aec_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 손으로 설계한 특징 대신, ROCKET 방식의 무작위 합성곱
    특징으로 AEC 곡선을 표현하면 임상변수 대비 추가 정보가 있는가?):

    1. train(g1090)/test(sdata)를 로드하고, 임상 단독 모델의 out-of-fold/외부 점수를 구한다.
    2. as_two_channel로 AEC 곡선(a128+crop)을 2채널 시계열로 만들고, RocketTransformer로 3000개의
       무작위 합성곱 커널을 학습(fit)한 뒤 train/test 각각에 대해 커널당 2개(최댓값, PPV)씩,
       총 6000차원 ROCKET 특징을 추출(transform)한다.
    3. tune_c로 "ROCKET 특징만 쓰는 모델"과 "임상변수+ROCKET 특징을 합친 모델" 각각에 대해
       C 그리드를 5-fold 교차검증으로 튜닝한다.
    4. cross_val_scores/fit_predict_external로 두 모델의 train out-of-fold 점수와 외부 예측을 구하고,
       별도로 "임상점수 z + ROCKET점수 z"를 로지스틱으로 결합한 단순 스택 모델도 학습한다.
    5. 4개 모델(임상 단독/ROCKET 단독/스택/임상+ROCKET 직접결합)의 train·외부 성능 지표와, 임상
       Youden 임계값에서의 혼동행렬을 표로 정리한다.
    6. logit_lrt로 "임상점수만" vs "임상점수+ROCKET점수"의 우도비검정을 train/외부 양쪽에서 수행하고,
       bootstrap_delta로 각 대안 모델과 임상 단독 모델 간 AUC/AP 차이의 신뢰구간을 추정한다.
    7. decile_table로 외부 데이터의 임상 양성군을 ROCKET 점수 10분위로 나눠 분위별 유병률을 계산하고,
       분위-유병률 그래프를 PNG로 저장한다.
    8. 커널 개수, 특징 차원, 최적 C, 검정 결과, 부트스트랩 델타, 분위별 위험도를 모두 JSON으로 저장.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]

    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    fold_indices = [va for _, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(xclin_tr, ytr)]
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, fold_indices)

    xrocket_tr_series = as_two_channel(train["aec"])
    xrocket_te_series = as_two_channel(test["aec"])
    rocket = RocketTransformer(n_kernels=3000, seed=SEED)
    xrocket_tr = rocket.fit_transform(xrocket_tr_series)
    xrocket_te = rocket.transform(xrocket_te_series)

    c_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    best_aec_c, aec_cv = tune_c(xrocket_tr, ytr, c_grid, "rocket_aec_only")
    xclinrocket_tr = np.column_stack([xclin_tr, xrocket_tr])
    xclinrocket_te = np.column_stack([xclin_te, xrocket_te])
    best_full_c, full_cv = tune_c(xclinrocket_tr, ytr, c_grid, "clinical_plus_rocket_aec")

    aec_oof, aec_oof_prob = cross_val_scores(xrocket_tr, ytr, best_aec_c)
    aec_ext, aec_ext_prob = fit_predict_external(xrocket_tr, ytr, xrocket_te, best_aec_c)

    full_oof, full_oof_prob = cross_val_scores(xclinrocket_tr, ytr, best_full_c)
    full_ext, full_ext_prob = fit_predict_external(xclinrocket_tr, ytr, xclinrocket_te, best_full_c)

    # A simple score-level stack: clinical score + AEC ROCKET score, trained only on OOF g1090 scores.
    c_oof_z, c_ext_z, _, _ = zfit_apply(clinical_oof, clinical_ext)
    a_oof_z, a_ext_z, _, _ = zfit_apply(aec_oof, aec_ext)
    stack = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
    stack.fit(np.column_stack([c_oof_z, a_oof_z]), ytr)
    stack_oof = stack.decision_function(np.column_stack([c_oof_z, a_oof_z]))
    stack_ext = stack.decision_function(np.column_stack([c_ext_z, a_ext_z]))
    stack_ext_prob = stack.predict_proba(np.column_stack([c_ext_z, a_ext_z]))[:, 1]

    clinical_prob_cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    clinical_prob_cal.fit(clinical_oof.reshape(-1, 1), ytr)
    clinical_ext_prob = clinical_prob_cal.predict_proba(clinical_ext.reshape(-1, 1))[:, 1]

    clinical_th = threshold_youden(ytr, clinical_oof)
    stack_th = threshold_youden(ytr, stack_oof)
    full_th = threshold_youden(ytr, full_oof)

    model_rows = [
        {
            "model": "clinical_only",
            **{f"train_oof_{k}": v for k, v in score_metrics(ytr, clinical_oof).items()},
            **{f"external_{k}": v for k, v in score_metrics(yte, clinical_ext, clinical_ext_prob).items()},
            **{f"external_{k}": v for k, v in binary_metrics(yte, clinical_ext, clinical_th).items()},
        },
        {
            "model": "rocket_aec_only",
            "best_C": best_aec_c,
            **{f"train_oof_{k}": v for k, v in score_metrics(ytr, aec_oof, aec_oof_prob).items()},
            **{f"external_{k}": v for k, v in score_metrics(yte, aec_ext, aec_ext_prob).items()},
        },
        {
            "model": "clinical_score_plus_rocket_score_stack",
            **{f"train_oof_{k}": v for k, v in score_metrics(ytr, stack_oof).items()},
            **{f"external_{k}": v for k, v in score_metrics(yte, stack_ext, stack_ext_prob).items()},
            **{f"external_{k}": v for k, v in binary_metrics(yte, stack_ext, stack_th).items()},
        },
        {
            "model": "clinical_variables_plus_rocket_features_direct",
            "best_C": best_full_c,
            **{f"train_oof_{k}": v for k, v in score_metrics(ytr, full_oof, full_oof_prob).items()},
            **{f"external_{k}": v for k, v in score_metrics(yte, full_ext, full_ext_prob).items()},
            **{f"external_{k}": v for k, v in binary_metrics(yte, full_ext, full_th).items()},
        },
    ]
    model_df = pd.DataFrame(model_rows)

    cond_train = logit_lrt(ytr, clinical_oof, aec_oof)
    cond_external = logit_lrt(yte, clinical_ext, aec_ext)
    deltas = {
        "stack_vs_clinical": bootstrap_delta(yte, clinical_ext, stack_ext),
        "direct_full_vs_clinical": bootstrap_delta(yte, clinical_ext, full_ext),
        "aec_only_vs_clinical": bootstrap_delta(yte, clinical_ext, aec_ext),
    }

    deciles = decile_table(yte, clinical_ext, aec_ext, clinical_th)

    model_df.to_csv(OUT_DIR / "rocket_model_metrics.csv", index=False)
    aec_cv.to_csv(OUT_DIR / "rocket_aec_only_cv_C.csv", index=False)
    full_cv.to_csv(OUT_DIR / "clinical_plus_rocket_cv_C.csv", index=False)
    deciles.to_csv(OUT_DIR / "clinical_positive_rocket_aec_deciles.csv", index=False)

    result = {
        "loss_function": "class-balanced L2-penalized logistic loss on ROCKET features",
        "n_kernels": rocket.n_kernels,
        "n_rocket_features": int(xrocket_tr.shape[1]),
        "cohorts": {
            "g1090_n": int(len(ytr)),
            "g1090_events": int(ytr.sum()),
            "sdata_n": int(len(yte)),
            "sdata_events": int(yte.sum()),
        },
        "best_C": {"rocket_aec_only": best_aec_c, "clinical_plus_rocket_direct": best_full_c},
        "conditional_lrt": {
            "train_oof_y_on_clinical_score_plus_rocket_aec_score": cond_train,
            "external_y_on_clinical_score_plus_rocket_aec_score": cond_external,
        },
        "bootstrap_external_deltas": deltas,
        "model_metrics": model_rows,
        "clinical_positive_aec_deciles": deciles.to_dict(orient="records"),
    }
    with open(OUT_DIR / "rocket_aec_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(deciles["aec_score_decile_among_clinical_positive"], deciles["prevalence"], marker="o", lw=2)
    ax.axhline(yte[clinical_ext >= clinical_th].mean(), color="#777777", ls="--", lw=1.2, label="Clinical-positive prevalence")
    ax.set_xlabel("ROCKET-AEC score decile among clinical-positive patients")
    ax.set_ylabel("Observed low-SMI prevalence in sdata")
    ax.set_title("External clinical-positive risk gradient by ROCKET-AEC score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clinical_positive_rocket_aec_decile_risk.png", dpi=180)
    plt.close(fig)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
