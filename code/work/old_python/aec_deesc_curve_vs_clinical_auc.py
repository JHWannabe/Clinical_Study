from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import clinical_scores, load_aec128  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_deesc_curve_vs_clinical_auc"
METRICS_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_individual_feature_full_metrics" / "individual_features_all_metrics_with_pvalues.csv"
OPS_TARGET = ["sens80", "sens85", "sens90", "sens95"]
OPS_PRIMARY = ["youden", "sens80", "sens85"]
TARGET_X = {"sens80": 0.80, "sens85": 0.85, "sens90": 0.90, "sens95": 0.95}
FEATURES = [
    "norm_curv_010_025_max",
    "norm_curv_010_021_max",
    "norm_slope_013_016_sd",
    "norm_curv_007_010_min",
    "dct_log_17",
    "norm_curv_055_058_mean",
]


def fisher_combine(pvals: list[float]) -> float:
    """여러 p값을 Fisher's method로 결합해 하나의 통합 p값을 계산."""
    p = np.asarray([x for x in pvals if np.isfinite(x) and x > 0], dtype=float)
    if len(p) == 0:
        return np.nan
    stat = -2 * np.sum(np.log(np.clip(p, 1e-300, 1.0)))
    return float(stats.chi2.sf(stat, 2 * len(p)))


def integrated(x: np.ndarray, y: np.ndarray) -> float:
    """x 구간에 대한 y값의 사다리꼴 적분을 구간 길이로 나눠, 여러 운영점에 걸친 평균적인 y값을 계산."""
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def clinical_auc_bootstrap(y: np.ndarray, score: np.ndarray, n_boot: int = 3000) -> dict:
    """임상 연속점수의 external AUC와 부트스트랩 95% 신뢰구간, AUC가 0.5 이하일 확률(부트스트랩 p값)을 계산."""
    rng = np.random.default_rng(20260630)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    vals = np.asarray(vals)
    return {
        "auc": float(roc_auc_score(y, score)),
        "ci2.5": float(np.quantile(vals, 0.025)),
        "ci97.5": float(np.quantile(vals, 0.975)),
        "p_auc_le_0.5_bootstrap": float((np.sum(vals <= 0.5) + 1) / (len(vals) + 1)),
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상 연속점수 자체의 AUC와 비교했을 때, 개별 AEC 특징을
    쓴 하향조정이 이진(문턱값 기반) AUC를 얼마나 더 끌어올리는지, 그리고 그 개선이 민감도
    80~95% 전 구간에서 일관되게 나타나는지 확인):

    1. g1090/sdata를 로드해 임상점수(c_s)를 계산하고, sdata에서 임상 연속점수의 AUC와 부트스트랩
       신뢰구간을 구한다.
    2. aec_individual_feature_full_metrics가 저장해둔 전체 지표 CSV에서, 미리 지정한 6개 관심
       특징(FEATURES)의 행만 추려낸다.
    3. 특징별로 (a) 3개 주요 운영점에서의 평균 이진 AUC delta와 Fisher 결합 p값, (b) 민감도
       80~95% 4개 목표점에 대한 이진 AUC/특이도이득/민감도손실/균형이득의 구간적분(integrated)과
       Fisher 결합 p값을 계산해 요약표로 정리하고 delta_auc_80_95 기준으로 정렬해 저장.
    4. 가장 유명한 단일 특징(norm_curv_010_025_max)에 대해 임상 연속 ROC 곡선과 하향조정 전후
       운영점을 겹쳐 그리고, 상위 5개 특징의 목표 운영점별 특이도이득 곡선을 나란히 그려 PNG로 저장.
    5. 임상 연속 AUC와 특징별 요약을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    _, c_s, _ = clinical_scores(g, s)
    y_s = s["y"].astype(int)
    clinical_auc = clinical_auc_bootstrap(y_s, c_s)

    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics["feature_label"].isin(FEATURES)].copy()
    rows = []
    for label, sub in metrics.groupby("feature_label"):
        target = sub[sub["operating_point"].isin(OPS_TARGET)].copy()
        primary = sub[sub["operating_point"].isin(OPS_PRIMARY)].copy()
        target["target_x"] = target["operating_point"].map(TARGET_X)
        rows.append(
            {
                "feature_label": label,
                "clinical_continuous_auc": clinical_auc["auc"],
                "clinical_continuous_auc_ci2.5": clinical_auc["ci2.5"],
                "clinical_continuous_auc_ci97.5": clinical_auc["ci97.5"],
                "clinical_continuous_auc_p_le_0.5": clinical_auc["p_auc_le_0.5_bootstrap"],
                "primary_mean_clinical_binary_auc": float(primary["clinical_auc_binary"].mean()),
                "primary_mean_post_binary_auc": float(primary["post_auc_binary"].mean()),
                "primary_mean_delta_binary_auc": float(primary["delta_auc_binary"].mean()),
                "primary_delta_auc_combined_p": fisher_combine(primary["auc_delta_p_bootstrap"].tolist()),
                "target_integrated_clinical_binary_auc_80_95": integrated(target["target_x"].to_numpy(), target["clinical_auc_binary"].to_numpy()),
                "target_integrated_post_binary_auc_80_95": integrated(target["target_x"].to_numpy(), target["post_auc_binary"].to_numpy()),
                "target_integrated_delta_binary_auc_80_95": integrated(target["target_x"].to_numpy(), target["delta_auc_binary"].to_numpy()),
                "target_integrated_specificity_gain_80_95": integrated(target["target_x"].to_numpy(), target["delta_specificity"].to_numpy()),
                "target_integrated_sensitivity_loss_80_95": integrated(target["target_x"].to_numpy(), -target["delta_sensitivity"].to_numpy()),
                "target_integrated_balanced_gain_80_95": integrated(
                    target["target_x"].to_numpy(),
                    (target["delta_specificity"] + target["delta_sensitivity"]).to_numpy(),
                ),
                "target_integrated_fisher_p": fisher_combine(target["deesc_event_fisher_p"].tolist()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("target_integrated_delta_binary_auc_80_95", ascending=False)
    summary.to_csv(OUT_DIR / "deesc_curve_vs_clinical_auc_summary.csv", index=False)

    point_cols = [
        "feature_label",
        "operating_point",
        "clinical_auc_binary",
        "post_auc_binary",
        "delta_auc_binary",
        "auc_delta_p_bootstrap",
        "clinical_sensitivity",
        "clinical_specificity",
        "post_sensitivity",
        "post_specificity",
        "delta_sensitivity",
        "delta_specificity",
        "deesc_event_rate",
        "deesc_event_fisher_p",
    ]
    metrics[point_cols].to_csv(OUT_DIR / "deesc_operating_point_binary_auc_points.csv", index=False)

    # Plot clinical continuous ROC and de-escalation operating points.
    fpr, tpr, _ = roc_curve(y_s, c_s)
    best = "norm_curv_010_025_max"
    best_points = metrics[metrics["feature_label"].eq(best)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.4))
    ax = axes[0]
    ax.plot(fpr, tpr, color="#333333", lw=2.2, label=f"Clinical continuous ROC AUC={clinical_auc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    for op in OPS_TARGET:
        r = best_points[best_points["operating_point"].eq(op)].iloc[0]
        ax.scatter(1 - r["clinical_specificity"], r["clinical_sensitivity"], color="#8DA0CB", s=58)
        ax.scatter(1 - r["post_specificity"], r["post_sensitivity"], color="#4DAF4A", s=58)
        ax.annotate(op.replace("sens", "S"), (1 - r["post_specificity"], r["post_sensitivity"]), fontsize=8, xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Clinical ROC vs de-escalated operating points", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False)

    ax = axes[1]
    for label in summary["feature_label"].head(5):
        sub = metrics[(metrics["feature_label"].eq(label)) & (metrics["operating_point"].isin(OPS_TARGET))].copy()
        sub["target_x"] = sub["operating_point"].map(TARGET_X)
        sub = sub.sort_values("target_x")
        ax.plot(sub["target_x"], sub["delta_specificity"] * 100, marker="o", lw=2.0, label=label)
    ax.axhline(0, color="#555555", lw=1)
    ax.set_xlabel("Clinical sensitivity operating point")
    ax.set_ylabel("Specificity gain after de-escalation (%p)")
    ax.set_title("De-escalation curve: specificity gain", loc="left", fontweight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "clinical_auc_vs_deesc_curve.png", dpi=220)
    plt.close(fig)

    print("Clinical continuous AUC")
    print(json.dumps(clinical_auc, indent=2))
    print("\nDe-escalation curve summary")
    print(summary.to_string(index=False))
    print(OUT_DIR / "clinical_auc_vs_deesc_curve.png")


if __name__ == "__main__":
    main()
