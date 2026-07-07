from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aec128_common_shape_feature import FILES, load_aec128


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_highdim_aec_only"
SEED = 20260629


def smooth_log(x_norm: np.ndarray) -> np.ndarray:
    """정규화된 곡선을 로그 스케일로 바꾼 뒤 Savitzky-Golay로 평활화."""
    return savgol_filter(np.log(np.clip(x_norm, 1e-6, None)), window_length=9, polyorder=2, axis=1, mode="interp")


def add_window_means(rows: dict[str, np.ndarray], signal: np.ndarray, prefix: str, lengths: list[int], step: int = 2) -> None:
    """여러 길이(lengths)의 겹치는 슬라이딩 윈도우 평균들을 rows 딕셔너리에 채워 넣음 (in-place)."""
    n = signal.shape[1]
    for length in lengths:
        for start0 in range(0, n - length + 1, step):
            end0 = start0 + length
            rows[f"{prefix}_mean_{start0 + 1:03d}_{end0:03d}"] = signal[:, start0:end0].mean(axis=1)


def add_haar_edges(rows: dict[str, np.ndarray], signal: np.ndarray, prefix: str, blocks: list[int], step: int = 1) -> None:
    """여러 블록 크기(blocks)에 대해 인접한 좌/우 블록 평균 차이(Haar 엣지)를 rows 딕셔너리에 채워 넣음 (in-place)."""
    n = signal.shape[1]
    for block in blocks:
        length = 2 * block
        for start0 in range(0, n - length + 1, step):
            mid0 = start0 + block
            end0 = start0 + length
            rows[f"{prefix}_edge_b{block:02d}_{start0 + 1:03d}_{mid0:03d}_to_{mid0 + 1:03d}_{end0:03d}"] = (
                signal[:, mid0:end0].mean(axis=1) - signal[:, start0:mid0].mean(axis=1)
            )


def build_feature_banks(x_norm: np.ndarray) -> dict[str, pd.DataFrame]:
    """로그 프로파일/도함수/DCT로부터 7가지 서로 다른 크기·구성의 "특징 은행"(원시128차원부터
    관심영역 밀집특징까지)을 만들어 이름별 DataFrame 딕셔너리로 반환."""
    p = smooth_log(x_norm)
    d1 = np.gradient(p, axis=1)
    d2 = np.gradient(d1, axis=1)
    coeff = dct(p, type=2, norm="ortho", axis=1)

    banks: dict[str, pd.DataFrame] = {}
    banks["raw_log128"] = pd.DataFrame({f"logp_{i + 1:03d}": p[:, i] for i in range(128)})

    rows = {f"logp_{i + 1:03d}": p[:, i] for i in range(128)}
    rows.update({f"d1_{i + 1:03d}": d1[:, i] for i in range(128)})
    rows.update({f"d2_{i + 1:03d}": d2[:, i] for i in range(128)})
    banks["profile_d1_d2"] = pd.DataFrame(rows)

    banks["dct48"] = pd.DataFrame({f"dct_{i:02d}": coeff[:, i] for i in range(1, 49)})

    rows = {}
    add_haar_edges(rows, p, "logp", blocks=[2, 4, 8, 16], step=1)
    banks["haar_edges_log"] = pd.DataFrame(rows)

    rows = {}
    add_window_means(rows, p, "logp", lengths=[4, 8, 16, 32], step=2)
    add_haar_edges(rows, p, "logp", blocks=[2, 4, 8, 16], step=1)
    banks["multiscale_log"] = pd.DataFrame(rows)

    rows = {}
    add_window_means(rows, p, "logp", lengths=[4, 8, 16, 32], step=2)
    add_haar_edges(rows, p, "logp", blocks=[2, 4, 8, 16], step=1)
    add_window_means(rows, d1, "d1", lengths=[4, 8, 16, 32], step=2)
    add_haar_edges(rows, d1, "d1", blocks=[2, 4, 8, 16], step=1)
    banks["multiscale_log_d1"] = pd.DataFrame(rows)

    # Mean-graph-inspired but not point-selected: dense local windows only around the mid trough and late rebound zones.
    roi = {}
    for signal, prefix in [(p, "logp"), (d1, "d1"), (d2, "d2")]:
        for length in [4, 6, 8, 12, 16, 24]:
            for start0 in range(35, 128 - length + 1):
                end0 = start0 + length
                if (start0 < 82 and end0 > 38) or (start0 < 128 and end0 > 84):
                    roi[f"{prefix}_roi_mean_{start0 + 1:03d}_{end0:03d}"] = signal[:, start0:end0].mean(axis=1)
        add_haar_edges(roi, signal[:, 35:128], f"{prefix}_roi35_128", blocks=[2, 4, 8, 12, 16], step=1)
    banks["mean_graph_roi_dense"] = pd.DataFrame(roi)

    return banks


def metric_row(dataset: str, y: np.ndarray, score: np.ndarray, prob: np.ndarray) -> dict:
    """데이터셋 이름과 점수·확률로부터 AUC/AP/로그손실/Brier를 한 행으로 정리."""
    return {
        "dataset": dataset,
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def make_model(penalty: str, c: float, class_weight: str | None) -> Pipeline:
    """표준화 후 지정된 페널티(l1/l2)·C·클래스 가중치로 로지스틱 회귀(liblinear)를 구성한 파이프라인 생성."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    penalty=penalty,
                    C=c,
                    solver="liblinear",
                    class_weight=class_weight,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def cv_model_grid(x: np.ndarray, y: np.ndarray, bank_name: str) -> pd.DataFrame:
    """penalty(l1/l2) x C(6개) x class_weight(2개) 전체 격자를 5-fold 교차검증으로 평가해 그리드 서치 결과표를 만듦."""
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for penalty in ["l2", "l1"]:
        for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            for class_weight in [None, "balanced"]:
                score = np.zeros(len(y), dtype=float)
                prob = np.zeros(len(y), dtype=float)
                for tr_idx, va_idx in skf.split(x, y):
                    model = make_model(penalty, c, class_weight)
                    model.fit(x[tr_idx], y[tr_idx])
                    score[va_idx] = model.decision_function(x[va_idx])
                    prob[va_idx] = model.predict_proba(x[va_idx])[:, 1]
                rows.append(
                    {
                        "bank": bank_name,
                        "penalty": penalty,
                        "C": c,
                        "class_weight": "balanced" if class_weight == "balanced" else "none",
                        "cv_auc": float(roc_auc_score(y, score)),
                        "cv_average_precision": float(average_precision_score(y, score)),
                        "cv_log_loss": float(log_loss(y, prob)),
                        "cv_brier": float(brier_score_loss(y, prob)),
                    }
                )
    return pd.DataFrame(rows)


def crossfit_fixed_hyper(x: np.ndarray, y: np.ndarray, penalty: str, c: float, class_weight_label: str) -> tuple[np.ndarray, np.ndarray]:
    """이미 정해진 하이퍼파라미터로 5-fold 교차검증을 돌려 train 전체에 대한 out-of-fold 점수와 확률을 만듦."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 100)
    score = np.zeros(len(y), dtype=float)
    prob = np.zeros(len(y), dtype=float)
    class_weight = "balanced" if class_weight_label == "balanced" else None
    for tr_idx, va_idx in skf.split(x, y):
        model = make_model(penalty, c, class_weight)
        model.fit(x[tr_idx], y[tr_idx])
        score[va_idx] = model.decision_function(x[va_idx])
        prob[va_idx] = model.predict_proba(x[va_idx])[:, 1]
    return score, prob


def top_coefficients(model: Pipeline, feature_names: list[str], top_n: int = 60) -> pd.DataFrame:
    """학습된 로지스틱 모델의 계수 중 절댓값이 큰 상위 top_n개 특징을 표로 정리."""
    coef = model.named_steps["logit"].coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coef": coef, "abs_coef": np.abs(coef)})
    return df.sort_values("abs_coef", ascending=False).head(top_n)


def evaluate_bank(
    bank_name: str,
    xtr_df: pd.DataFrame,
    xte_df: pd.DataFrame,
    ytr: np.ndarray,
    yte: np.ndarray,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """한 특징 은행에 대해 하이퍼파라미터 그리드서치→최적값으로 OOF 재적합→전체 train으로 외부 예측까지
    수행하고, 상위 계수 표와 함께 성능 결과 딕셔너리를 반환."""
    xtr = xtr_df.to_numpy(dtype=float)
    xte = xte_df.to_numpy(dtype=float)
    cv_df = cv_model_grid(xtr, ytr, bank_name)
    best = cv_df.sort_values(["cv_auc", "cv_average_precision", "cv_log_loss"], ascending=[False, False, True]).iloc[0]

    oof_score, oof_prob = crossfit_fixed_hyper(
        xtr,
        ytr,
        str(best["penalty"]),
        float(best["C"]),
        str(best["class_weight"]),
    )
    class_weight = "balanced" if best["class_weight"] == "balanced" else None
    final = make_model(str(best["penalty"]), float(best["C"]), class_weight)
    final.fit(xtr, ytr)
    ext_score = final.decision_function(xte)
    ext_prob = final.predict_proba(xte)[:, 1]

    coef_df = top_coefficients(final, list(xtr_df.columns))
    result = {
        "bank": bank_name,
        "n_features": int(xtr.shape[1]),
        "selected_penalty": str(best["penalty"]),
        "selected_C": float(best["C"]),
        "selected_class_weight": str(best["class_weight"]),
        "selection_cv_auc": float(best["cv_auc"]),
        "selection_cv_average_precision": float(best["cv_average_precision"]),
        "oof": metric_row("g1090_oof_fixed_hyper", ytr, oof_score, oof_prob),
        "external": metric_row("sdata_external", yte, ext_score, ext_prob),
        "nonzero_coefficients": int(np.sum(np.abs(final.named_steps["logit"].coef_.ravel()) > 1e-10)),
    }
    return result, cv_df, coef_df


def plot_summary(perf: pd.DataFrame) -> None:
    """각 특징 은행의 train OOF AUC와 외부 AUC를 나란히 막대그래프로 비교해 PNG로 저장."""
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    order = perf.sort_values("external_auc", ascending=False)
    x = np.arange(len(order))
    ax.bar(x - 0.18, order["oof_auc"], width=0.36, color="#4C78A8", label="g1090 OOF")
    ax.bar(x + 0.18, order["external_auc"], width=0.36, color="#F58518", label="sdata external")
    ax.axhline(0.5, color="#555555", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order["bank"], rotation=35, ha="right")
    ax.set_ylabel("AEC-only AUC")
    ax.set_title("High-dimensional AEC_128-only feature banks", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_highdim_aec_only_auc.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 고차원(수백~수천 차원) AEC 특징 은행 여러 종류를 각각
    정규화된 로지스틱 회귀로 학습하면, 어떤 표현 방식이 AEC 단독 판별력을 가장 잘 살리는가?):

    1. g1090/sdata를 로드하고, build_feature_banks로 7가지 특징 은행(원시 로그128차원,
       로그+도함수, DCT48, Haar 엣지, 다중스케일, 다중스케일+도함수, 관심영역 밀집특징)을 만든다.
    2. 각 은행마다 evaluate_bank로: cv_model_grid(penalty x C x class_weight 그리드서치)로
       최적 하이퍼파라미터를 고르고, 그 값으로 5-fold OOF 점수를 만들고, 전체 train으로 재학습한
       모델로 외부 데이터를 예측하며, 상위 계수(어느 위치/구간이 중요한지)를 뽑는다.
    3. 7개 은행의 성능(선택 CV AUC, train OOF/외부 AUC·AP·로그손실·Brier, 0이 아닌 계수 개수)을
       표로 모아 CSV로 저장하고, 그리드서치 전체 결과와 상위 계수도 각각 CSV로 저장.
    4. 은행별 train OOF vs 외부 AUC를 막대그래프로 비교해 PNG로 저장.
    5. 전체 결과를 JSON으로 저장하고 성능 요약표를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}
    train = datasets["g1090"]
    test = datasets["sdata"]
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    train_banks = build_feature_banks(train["x"])
    test_banks = build_feature_banks(test["x"])

    results = []
    cv_all = []
    for bank_name, xtr_df in train_banks.items():
        print(f"Running {bank_name}: {xtr_df.shape[1]} features")
        result, cv_df, coef_df = evaluate_bank(bank_name, xtr_df, test_banks[bank_name], ytr, yte)
        results.append(result)
        cv_all.append(cv_df)
        coef_df.to_csv(OUT_DIR / f"{bank_name}_top_coefficients.csv", index=False)

    perf_rows = []
    for r in results:
        perf_rows.append(
            {
                "bank": r["bank"],
                "n_features": r["n_features"],
                "selected_penalty": r["selected_penalty"],
                "selected_C": r["selected_C"],
                "selected_class_weight": r["selected_class_weight"],
                "nonzero_coefficients": r["nonzero_coefficients"],
                "selection_cv_auc": r["selection_cv_auc"],
                "selection_cv_average_precision": r["selection_cv_average_precision"],
                "oof_auc": r["oof"]["auc"],
                "oof_average_precision": r["oof"]["average_precision"],
                "oof_log_loss": r["oof"]["log_loss"],
                "external_auc": r["external"]["auc"],
                "external_average_precision": r["external"]["average_precision"],
                "external_log_loss": r["external"]["log_loss"],
                "external_brier": r["external"]["brier"],
            }
        )
    perf = pd.DataFrame(perf_rows).sort_values("external_auc", ascending=False)
    perf.to_csv(OUT_DIR / "aec128_highdim_aec_only_performance.csv", index=False)
    pd.concat(cv_all, ignore_index=True).to_csv(OUT_DIR / "aec128_highdim_aec_only_cv_grid.csv", index=False)
    with open(OUT_DIR / "aec128_highdim_aec_only_summary.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    plot_summary(perf)

    print(perf.to_string(index=False))
    print(OUT_DIR / "aec128_highdim_aec_only_performance.csv")
    print(OUT_DIR / "aec128_highdim_aec_only_auc.png")


if __name__ == "__main__":
    main()
