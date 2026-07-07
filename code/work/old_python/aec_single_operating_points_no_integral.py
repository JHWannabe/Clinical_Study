from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    risk_direction,
    standardize_train_test,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_single_operating_points_no_integral"
CONFIG_PATH = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_individual_feature_full_metrics" / "selected_individual_feature_configs.csv"
)
EXISTING_PATH = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_individual_feature_full_metrics" / "individual_features_all_metrics_with_pvalues.csv"
)
FEATURE_ORDER = [
    "norm_curv_010_025_max",
    "norm_slope_013_016_sd",
    "norm_curv_010_021_max",
    "norm_curv_007_010_min",
    "dct_log_17",
]
OPS = [
    ("sens75", 0.75),
    ("sens80", 0.80),
    ("sens85", 0.85),
    ("sens90", 0.90),
    ("sens95", 0.95),
]


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측 양성/음성과 실제 결과로부터 tp/fp/fn/tn, 정확도, 민감도, 특이도, 균형정확도를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(y),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (sens + spec),
    }


def exact_paired_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상단독 판정과 게이트 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
    yy = y.astype(bool)
    clinical_pos = clinical_pos.astype(bool)
    final_pos = final_pos.astype(bool)

    pos = yy
    sens_loss = int(np.sum(pos & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(pos & ~clinical_pos & final_pos))

    neg = ~yy
    spec_gain = int(np.sum(neg & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(neg & ~clinical_pos & final_pos))

    clinical_correct = clinical_pos == yy
    final_correct = final_pos == yy
    acc_loss = int(np.sum(clinical_correct & ~final_correct))
    acc_gain = int(np.sum(~clinical_correct & final_correct))

    return {
        "sensitivity_loss_p_exact": exact_paired_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_paired_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_paired_p(acc_gain, acc_loss),
    }


def bootstrap_delta_ba_p(
    y: np.ndarray,
    clinical_pos: np.ndarray,
    final_pos: np.ndarray,
    n_boot: int = 5000,
    seed: int = 20260630,
) -> float:
    """부트스트랩 재표본추출로 균형정확도 변화량(게이트 후 - 임상단독)의 분포를 만들고, 그 값이 0 이하일 확률(단측 부트스트랩 p값 근사)을 계산."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        c = counts(y[idx], clinical_pos[idx])["balanced_accuracy"]
        f = counts(y[idx], final_pos[idx])["balanced_accuracy"]
        vals.append(f - c)
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals <= 0.0))


def gate_predictions(
    clinical_z: np.ndarray,
    aec_risk_z: np.ndarray,
    threshold_z: float,
    width: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """가우시안 경계가중치 게이트로 임상양성 예측과, 게이트 통과 후 최종양성 예측을 계산."""
    boundary = np.exp(-0.5 * ((clinical_z - threshold_z) / width) ** 2)
    gate = clinical_z + lam * boundary * aec_risk_z
    clinical_pos = clinical_z >= threshold_z
    final_pos = clinical_pos & (gate >= threshold_z)
    return clinical_pos, final_pos


def row_metrics(
    feature_label: str,
    feature: str,
    y: np.ndarray,
    clinical_z: np.ndarray,
    aec_risk_z: np.ndarray,
    op: str,
    target_sensitivity: float,
    threshold_z: float,
    width: float,
    lam: float,
) -> dict:
    """한 특징·한 운영점에 대해 임상단독 vs 게이트적용 후 성능 지표, 하향조정군의 사건비율·Fisher p, 각종 대응검정 p값을 모두 계산해 하나의 결과 행으로 반환."""
    clinical_pos, final_pos = gate_predictions(clinical_z, aec_risk_z, threshold_z, width, lam)
    clinical = counts(y, clinical_pos)
    post = counts(y, final_pos)
    pv = pvalues(y, clinical_pos, final_pos)

    keep = final_pos
    deesc = clinical_pos & ~final_pos
    a = int(np.sum(y[keep] == 1))
    b = int(np.sum(y[keep] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    fisher_p = stats.fisher_exact([[a, b], [c, d]])[1] if (a + b) and (c + d) else np.nan

    return {
        "feature_label": feature_label,
        "feature": feature,
        "operating_point": op,
        "target_sensitivity": target_sensitivity,
        "clinical_threshold_z": threshold_z,
        "width": width,
        "lambda": lam,
        "clinical_balanced_accuracy": clinical["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - clinical["balanced_accuracy"],
        "clinical_accuracy": clinical["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - clinical["accuracy"],
        "clinical_sensitivity": clinical["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": clinical["sensitivity"] - post["sensitivity"],
        "clinical_specificity": clinical["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - clinical["specificity"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": c,
        "deesc_event_rate": c / (c + d) if c + d else np.nan,
        "fp_removed": d,
        "tp_lost": c,
        "deesc_event_fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
        **pv,
        "balanced_accuracy_delta_p_bootstrap": bootstrap_delta_ba_p(y, clinical_pos, final_pos),
    }


def merge_existing_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """aec_individual_feature_full_metrics가 이미 계산해둔 p값이 있으면 그 값을 우선 사용하도록 새로 계산한 결과와 병합."""
    if not EXISTING_PATH.exists():
        return df
    existing = pd.read_csv(EXISTING_PATH)
    cols = [
        "feature_label",
        "operating_point",
        "auc_delta_p_bootstrap",
        "accuracy_delta_p_mcnemar",
        "sensitivity_loss_p_exact",
        "specificity_gain_p_exact",
        "deesc_event_fisher_p",
    ]
    existing = existing[[c for c in cols if c in existing.columns]].copy()
    existing = existing.rename(columns={"auc_delta_p_bootstrap": "existing_balanced_accuracy_delta_p_bootstrap"})
    out = df.merge(existing, on=["feature_label", "operating_point"], how="left")
    has_existing = out["existing_balanced_accuracy_delta_p_bootstrap"].notna()
    out.loc[has_existing, "balanced_accuracy_delta_p_bootstrap"] = out.loc[
        has_existing, "existing_balanced_accuracy_delta_p_bootstrap"
    ]
    for col in [
        "accuracy_delta_p_mcnemar",
        "sensitivity_loss_p_exact",
        "specificity_gain_p_exact",
        "deesc_event_fisher_p",
    ]:
        right = f"{col}_y"
        left = f"{col}_x"
        if left in out.columns and right in out.columns:
            out[col] = out[right].where(out[right].notna(), out[left])
            out = out.drop(columns=[left, right])
    return out.drop(columns=["existing_balanced_accuracy_delta_p_bootstrap"], errors="ignore")


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 여러 운영점에 걸친 "구간적분" 요약 대신, 민감도75~95%
    5개 개별 운영점 각각에 대한 성능과 p값을 있는 그대로 표로 남겨, 값이 뭉개지지 않고 그대로
    보이게 하려는 목적):

    1. g1090/sdata를 로드하고 임상점수(c_g, c_s)를 계산한 뒤, g1090 기준 5개 운영점
       (sens75~sens95)에 해당하는 임상 임계값을 threshold_for_min_sensitivity로 구한다.
    2. 후보 특징뱅크를 만들어 표준화하고 방향을 위험증가 쪽으로 통일한다.
    3. aec_individual_feature_full_metrics가 저장해둔 "선택된 개별 특징 설정" 표에서 5개 관심
       특징(FEATURE_ORDER)의 폭(width)·람다(lambda)를 읽어와, 각 특징 x 5개 운영점 조합마다
       row_metrics로 sdata(external)에서의 임상단독 대 게이트적용 성능과 각종 p값을 계산.
    4. 기존에 계산되어 있던 p값이 있으면 그것으로 덮어써 일관성을 유지(merge_existing_pvalues)하고,
       특징/운영점 순서를 고정한 뒤 전체표·주요특징(norm_curv_010_025_max)표·요약(compact)표를
       각각 CSV로 저장.
    5. 요약표를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, _ = clinical_scores(g, s)
    thresholds = {label: threshold_for_min_sensitivity(g["y"], c_g, target) for label, target in OPS}

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    xg = xg * direction[None, :]
    xs = xs * direction[None, :]
    name_to_idx = {name: i for i, name in enumerate(names)}

    config = pd.read_csv(CONFIG_PATH)
    config = config[config["label"].isin(FEATURE_ORDER)].copy()
    rows = []
    for label in FEATURE_ORDER:
        r = config[config["label"].eq(label)].iloc[0]
        feature = str(r["feature"])
        idx = name_to_idx[feature]
        for op, target in OPS:
            rows.append(
                row_metrics(
                    feature_label=label,
                    feature=feature,
                    y=s["y"].astype(int),
                    clinical_z=c_s,
                    aec_risk_z=xs[:, idx],
                    op=op,
                    target_sensitivity=target,
                    threshold_z=thresholds[op],
                    width=float(r["width"]),
                    lam=float(r["lambda"]),
                )
            )

    df = pd.DataFrame(rows)
    df = merge_existing_pvalues(df)
    df["feature_label"] = pd.Categorical(df["feature_label"], categories=FEATURE_ORDER, ordered=True)
    df["operating_point"] = pd.Categorical(df["operating_point"], categories=[x[0] for x in OPS], ordered=True)
    df = df.sort_values(["feature_label", "operating_point"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "single_operating_points_75_95_with_pvalues.csv", index=False)

    main_feature = df[df["feature_label"].eq("norm_curv_010_025_max")].copy()
    main_feature.to_csv(OUT_DIR / "main_feature_single_operating_points_75_95.csv", index=False)

    compact_cols = [
        "feature_label",
        "operating_point",
        "clinical_balanced_accuracy",
        "post_balanced_accuracy",
        "delta_balanced_accuracy",
        "balanced_accuracy_delta_p_bootstrap",
        "clinical_sensitivity",
        "post_sensitivity",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "clinical_specificity",
        "post_specificity",
        "specificity_gain",
        "specificity_gain_p_exact",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
    ]
    df[compact_cols].to_csv(OUT_DIR / "compact_single_operating_points_75_95.csv", index=False)
    print(df[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
