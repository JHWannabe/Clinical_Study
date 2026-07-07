from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores, load_aec128  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402
from aec_waviness_feature_test import waviness_features  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_waviness_comprehensive_metrics"
SEED = 20260630
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
FEATURES = [
    "wave_regional_contrast_sum",
    "wave_trough_rebound_amp_053_118",
    "wave_mid_trough_depth_linear_053_076",
    "wave_linear_abs_tv_041_118",
    "wave_linear_rms_041_118",
    "wave_linear_curv_abs_041_118",
]


def zfit_apply(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train(Gangnam)의 평균·표준편차로 표준화 파라미터를 정하고, train과 test(Sinchon) 양쪽에 동일하게 적용해 z-score로 변환."""
    mu = float(np.nanmean(train))
    sd = float(np.nanstd(train))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd


def auc_p_mannwhitney(y: np.ndarray, score: np.ndarray) -> float:
    """점수의 AUC가 0.5와 다른지를 Mann-Whitney U 검정으로 확인한 양측 p값을 계산."""
    y = y.astype(int)
    a = score[y == 1]
    b = score[y == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def bootstrap_auc_delta(
    y: np.ndarray,
    base_score: np.ndarray,
    new_score: np.ndarray,
    n_boot: int = 3000,
    seed: int = SEED,
) -> dict:
    """부트스트랩 재표본추출로 새 점수와 기준 점수의 AUC 차이 분포를 만들어, 관측된 delta AUC와 95% 신뢰구간, 양측 부트스트랩 p값을 계산."""
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    observed = float(roc_auc_score(y, new_score) - roc_auc_score(y, base_score))
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(float(roc_auc_score(y[idx], new_score[idx]) - roc_auc_score(y[idx], base_score[idx])))
    arr = np.asarray(diffs, dtype=float)
    if len(arr) == 0:
        return {"delta_auc": observed, "delta_auc_ci_low": np.nan, "delta_auc_ci_high": np.nan, "delta_auc_p_boot": np.nan}
    p = 2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))
    return {
        "delta_auc": observed,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)),
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)),
        "delta_auc_p_boot": float(min(1.0, p)),
    }


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """지정된 임계값으로 이진화한 예측의 정확도, 민감도, 특이도, 균형정확도, PPV, NPV, tp/fn/tn/fp를 계산."""
    y = y.astype(bool)
    pred = score >= threshold
    tp = int(np.sum(y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def continuous_model_metrics(y: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict:
    """연속 점수의 AUC, average precision, AUC의 Mann-Whitney p값, Brier score, 그리고 (Youden 임계값 또는 지정 임계값 기준) 이진 성능 지표를 한 번에 계산."""
    if threshold is None:
        threshold = threshold_youden(y.astype(int), score)
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return {
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "auc_p_mannwhitney": auc_p_mannwhitney(y, score),
        "brier": float(brier_score_loss(y, np.clip(prob, 1e-6, 1.0 - 1e-6))),
        **binary_metrics(y, score, threshold),
    }


def fit_oof_external(xg: np.ndarray, yg: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """5-fold OOF로 g1090(train)에서 로지스틱회귀(임상점수+굴곡특징)를 학습해 train OOF 점수를 만들고, 5개 fold 모델의 예측을 평균해 sdata(external) 점수를 만든다."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(yg), dtype=float)
    test_scores = []
    for tr, va in folds.split(xg, yg):
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)
        model.fit(xg[tr], yg[tr])
        oof[va] = model.decision_function(xg[va])
        test_scores.append(model.decision_function(xs))
    final_test = np.mean(test_scores, axis=0)
    return oof, final_test


def exact_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상단독 판정과 게이트 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
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
    }


def fisher_p(y: np.ndarray, final_pos: np.ndarray, deesc: np.ndarray) -> float:
    """최종 유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[final_pos] == 1))
    b = int(np.sum(y[final_pos] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def deesc_metric_row(
    dataset: str,
    feature: str,
    op: str,
    direction: int,
    gate_threshold: float,
    y: np.ndarray,
    clinical_pos: np.ndarray,
    feature_z: np.ndarray,
) -> dict:
    """지정된 방향·컷오프로 임상양성군을 하향조정/유지로 나누고, 임상단독 대비 성능 변화, 하향조정군 사건비율·Fisher p값, 각종 대응검정 p값을 계산."""
    deesc = clinical_pos & (direction * feature_z >= gate_threshold)
    final_pos = clinical_pos & ~deesc
    base = binary_metrics(y, clinical_pos.astype(float), 0.5)
    post = binary_metrics(y, final_pos.astype(float), 0.5)
    return {
        "dataset": dataset,
        "feature": feature,
        "operating_point": op,
        "direction": direction,
        "gate_threshold": float(gate_threshold),
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
        "deesc_event_fisher_p": fisher_p(y, final_pos, deesc),
        **paired_pvalues(y, clinical_pos, final_pos),
    }


def choose_deesc_threshold(
    y: np.ndarray,
    clinical_pos: np.ndarray,
    feature_z: np.ndarray,
    feature: str,
    op: str,
) -> tuple[int, float, dict]:
    """Gangnam(train)에서 방향(1/-1)과 여러 분위수 컷오프를 모두 시도해, 하향조정표본≥25, 민감도손실 p≥0.05, 특이도이득>0, Fisher p<0.05 조건을 만족하는 후보 중 selection_score가 최대인 방향·컷오프를 고른다(조건을 만족하는 후보가 없으면 단계적으로 완화된 대체 기준으로 재시도)."""
    best = None
    values = feature_z[clinical_pos]
    thresholds = np.unique(np.quantile(values, np.linspace(0.50, 0.95, 46)))
    for direction in [1, -1]:
        for th in thresholds:
            row = deesc_metric_row("Gangnam", feature, op, direction, float(th), y, clinical_pos, feature_z)
            if row["deesc_n"] < 25:
                continue
            if row["sensitivity_loss_p_exact"] < 0.05:
                continue
            if row["specificity_gain"] <= 0:
                continue
            if row["deesc_event_fisher_p"] >= 0.05:
                continue
            score = (
                row["specificity_gain"]
                + 0.4 * row["delta_balanced_accuracy"]
                + 0.25 * row["delta_accuracy"]
                - 0.2 * row["sensitivity_loss"]
            )
            candidate = {**row, "selection_score": float(score)}
            if best is None or candidate["selection_score"] > best["selection_score"]:
                best = candidate
    if best is None:
        # Fallback: still report the best specificity gain under non-significant sensitivity loss.
        for direction in [1, -1]:
            for th in thresholds:
                row = deesc_metric_row("Gangnam", feature, op, direction, float(th), y, clinical_pos, feature_z)
                if row["deesc_n"] < 25 or row["sensitivity_loss_p_exact"] < 0.05 or row["specificity_gain"] <= 0:
                    continue
                score = row["specificity_gain"] - 0.2 * row["sensitivity_loss"]
                candidate = {**row, "selection_score": float(score)}
                if best is None or candidate["selection_score"] > best["selection_score"]:
                    best = candidate
    if best is None:
        # Last fallback for degenerate cases.
        direction = 1
        th = float(np.quantile(values, 0.75))
        best = {**deesc_metric_row("Gangnam", feature, op, direction, th, y, clinical_pos, feature_z), "selection_score": np.nan}
    return int(best["direction"]), float(best["gate_threshold"]), best


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_waviness_feature_test가 정의한 6개 굴곡 특징 각각을
    (1) 단독 위험점수로서, (2) 임상점수에 더하는 결합모델로서, (3) 임상양성군 하향조정 게이트로서
    - 세 가지 방식 모두에서 g1090/sdata 양쪽에 걸쳐 얼마나 유의미하고 강건하게 작동하는지 종합
    평가):

    1. g1090(train)/sdata(external)를 로드해 임상점수와 6개 굴곡특징을 계산.
    2. 임상단독 모델의 Youden 임계값·성능을 기준선으로 기록.
    3. 각 굴곡특징에 대해: (a) train에서 학습한 부호로 방향을 정한 단독 점수의 AUC 등 연속모델
       지표를 두 데이터셋에서 계산, (b) 임상점수+굴곡특징을 5-fold OOF 로지스틱회귀로 결합한
       모델의 성능과 임상단독 대비 delta AUC 부트스트랩 신뢰구간/p값을 계산, (c) 5개 운영점마다
       choose_deesc_threshold로 train에서 최적 방향·컷오프를 고른 하향조정 게이트를 external에
       적용한 성능을 계산.
    4. 특징 x 데이터셋별로 하향조정 게이트 성능의 범위(최소/최대/평균)를 요약.
    5. 단독 AUC 결과, 결합모델 결과, 하향조정 요약을 각각 CSV로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    yg = g["y"].astype(int)
    ys = s["y"].astype(int)
    cg, cs, _ = clinical_scores(g, s)
    wg = waviness_features(g["norm"])
    ws = waviness_features(s["norm"])

    auc_rows = []
    additive_rows = []
    deesc_rows = []

    clinical_threshold = threshold_youden(yg, cg)
    clinical_train_metrics = continuous_model_metrics(yg, cg, clinical_threshold)
    clinical_external_metrics = continuous_model_metrics(ys, cs, clinical_threshold)

    additive_rows.append({"model": "clinical", "feature": "none", "dataset": "Gangnam internal OOF", **clinical_train_metrics})
    additive_rows.append({"model": "clinical", "feature": "none", "dataset": "Sinchon external", **clinical_external_metrics})

    for feature in FEATURES:
        fzg, fzs = zfit_apply(wg[feature].to_numpy(dtype=float), ws[feature].to_numpy(dtype=float))

        # Continuous AUC of the waviness scalar alone. Direction is learned in Gangnam and applied to Sinchon.
        raw_auc_g = roc_auc_score(yg, fzg)
        risk_direction = 1 if raw_auc_g >= 0.5 else -1
        wave_risk_g = risk_direction * fzg
        wave_risk_s = risk_direction * fzs
        for dataset, y, score in [
            ("Gangnam internal", yg, wave_risk_g),
            ("Sinchon external", ys, wave_risk_s),
        ]:
            auc_rows.append(
                {
                    "feature": feature,
                    "dataset": dataset,
                    "risk_direction_from_gangnam": risk_direction,
                    **continuous_model_metrics(y, score),
                }
            )

        # Add one scalar feature to the clinical score.
        xg = np.column_stack([cg, fzg])
        xs = np.column_stack([cs, fzs])
        combo_g, combo_s = fit_oof_external(xg, yg, xs)
        combo_threshold = threshold_youden(yg, combo_g)
        combo_train_metrics = continuous_model_metrics(yg, combo_g, combo_threshold)
        combo_external_metrics = continuous_model_metrics(ys, combo_s, combo_threshold)
        additive_rows.append(
            {
                "model": "clinical_plus_waviness",
                "feature": feature,
                "dataset": "Gangnam internal OOF",
                **combo_train_metrics,
                **bootstrap_auc_delta(yg, cg, combo_g, seed=SEED + 11),
            }
        )
        additive_rows.append(
            {
                "model": "clinical_plus_waviness",
                "feature": feature,
                "dataset": "Sinchon external",
                **combo_external_metrics,
                **bootstrap_auc_delta(ys, cs, combo_s, seed=SEED + 17),
            }
        )

        # Scalar clinical-positive de-escalation gate, threshold selected in Gangnam per operating point.
        for op, target in OPS:
            th_clinical = threshold_for_min_sensitivity(yg, cg, target)
            cpos_g = cg >= th_clinical
            cpos_s = cs >= th_clinical
            direction, gate_th, train_row = choose_deesc_threshold(yg, cpos_g, fzg, feature, op)
            deesc_rows.append({**train_row, "selection_dataset": "Gangnam"})
            deesc_rows.append(
                {
                    **deesc_metric_row("Sinchon", feature, op, direction, gate_th, ys, cpos_s, fzs),
                    "selection_dataset": "Gangnam",
                }
            )

    pd.DataFrame(auc_rows).to_csv(OUT_DIR / "waviness_scalar_alone_auc_metrics.csv", index=False)
    pd.DataFrame(additive_rows).to_csv(OUT_DIR / "clinical_plus_waviness_additive_metrics.csv", index=False)
    pd.DataFrame(deesc_rows).to_csv(OUT_DIR / "waviness_scalar_deescalation_metrics.csv", index=False)

    deesc = pd.DataFrame(deesc_rows)
    summary = (
        deesc.groupby(["feature", "dataset"])
        .agg(
            min_p_loss=("sensitivity_loss_p_exact", "min"),
            max_sens_loss=("sensitivity_loss", "max"),
            min_spec_gain=("specificity_gain", "min"),
            mean_spec_gain=("specificity_gain", "mean"),
            min_delta_ba=("delta_balanced_accuracy", "min"),
            mean_delta_ba=("delta_balanced_accuracy", "mean"),
            max_fisher_p=("deesc_event_fisher_p", "max"),
            mean_deesc_event_rate=("deesc_event_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "waviness_scalar_deescalation_range_summary.csv", index=False)

    print("\nSCALAR AUC ALONE")
    auc_df = pd.DataFrame(auc_rows)
    print(
        auc_df[["feature", "dataset", "auc", "auc_p_mannwhitney", "accuracy", "sensitivity", "specificity"]]
        .sort_values(["feature", "dataset"])
        .to_string(index=False)
    )

    print("\nCLINICAL + WAVINESS ADDITIVE MODELS")
    add_df = pd.DataFrame(additive_rows)
    print(
        add_df[
            [
                "model",
                "feature",
                "dataset",
                "auc",
                "delta_auc",
                "delta_auc_ci_low",
                "delta_auc_ci_high",
                "delta_auc_p_boot",
                "accuracy",
                "sensitivity",
                "specificity",
            ]
        ]
        .sort_values(["feature", "dataset", "model"])
        .to_string(index=False)
    )

    print("\nDE-ESCALATION RANGE SUMMARY")
    print(summary.sort_values(["feature", "dataset"]).to_string(index=False))


if __name__ == "__main__":
    main()
