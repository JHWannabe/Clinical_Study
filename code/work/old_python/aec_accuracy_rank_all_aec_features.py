from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_accuracy_max_reclassification_search import (  # noqa: E402
    LOSS_BUDGETS,
    build_combo_candidates,
    metric_for_candidate,
    pass_constraints,
    pattern_candidates,
    score_model_candidates,
    score_threshold_candidates,
)
from aec_lock_smoothed_deesc_gate import DATA_DIR, OPS, clinical_scores, load_dataset  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_accuracy_rank_all_aec_features"
CONSTRAINT_BUDGET = 0.08


def add_external_metrics(candidates, g, s, cpos_g, cpos_s) -> pd.DataFrame:
    """각 후보에 대해 내부(g1090)와 외부(sdata) 데이터셋 모두에서 정확도/민감도/특이도 관련 지표를 계산해 하나의 비교용 데이터프레임으로 정리한다."""
    rows = []
    for cand in candidates:
        int_row = metric_for_candidate("g1090_internal", g, cpos_g[cand.op], cand)
        ext_row = metric_for_candidate("sdata_external", s, cpos_s[cand.op], cand)
        rows.append(
            {
                "candidate": cand.name,
                "candidate_kind": cand.kind,
                "operating_point": cand.op,
                "candidate_detail": cand.detail,
                "internal_accuracy": int_row["post_accuracy"],
                "internal_accuracy_gain": int_row["accuracy_delta"],
                "internal_accuracy_p": int_row["accuracy_delta_p_mcnemar"],
                "internal_sensitivity": int_row["post_sensitivity"],
                "internal_sensitivity_loss": int_row["sensitivity_loss"],
                "internal_sensitivity_loss_p": int_row["sensitivity_loss_p_exact"],
                "internal_specificity": int_row["post_specificity"],
                "internal_specificity_gain": int_row["specificity_gain"],
                "internal_specificity_p": int_row["specificity_gain_p_exact"],
                "internal_deesc_n": int_row["deesc_n"],
                "internal_deesc_events": int_row["deesc_events"],
                "internal_deesc_event_rate": int_row["deesc_event_rate"],
                "external_accuracy": ext_row["post_accuracy"],
                "external_accuracy_gain": ext_row["accuracy_delta"],
                "external_accuracy_p": ext_row["accuracy_delta_p_mcnemar"],
                "external_sensitivity": ext_row["post_sensitivity"],
                "external_sensitivity_loss": ext_row["sensitivity_loss"],
                "external_sensitivity_loss_p": ext_row["sensitivity_loss_p_exact"],
                "external_specificity": ext_row["post_specificity"],
                "external_specificity_gain": ext_row["specificity_gain"],
                "external_specificity_p": ext_row["specificity_gain_p_exact"],
                "external_deesc_n": ext_row["deesc_n"],
                "external_deesc_events": ext_row["deesc_events"],
                "external_deesc_event_rate": ext_row["deesc_event_rate"],
            }
        )
    return pd.DataFrame(rows)


def constraint_ok(df: pd.DataFrame, prefix: str, budget: float) -> pd.Series:
    """prefix(internal/external)가 붙은 컬럼들을 이용해 최소 재분류 표본 수, 양(+)의 특이도 증가, 예산 이내 민감도 손실, 손실 유의성 조건을 만족하는지 행 단위 불리언 시리즈로 반환한다."""
    p = pd.to_numeric(df[f"{prefix}_sensitivity_loss_p"], errors="coerce").fillna(1.0)
    return (
        (df[f"{prefix}_deesc_n"] >= 10)
        & (df[f"{prefix}_specificity_gain"] > 0)
        & (df[f"{prefix}_sensitivity_loss"] <= budget + 1e-12)
        & (p >= 0.05)
    )


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """내부/외부 제약 조건 충족 여부를 계산하고, 내부 정확도 증가 기준 순위와 외부 정확도 증가 기준 순위를 각각 매겨 컬럼으로 추가한다."""
    out = df.copy()
    out["internal_constraint_ok"] = constraint_ok(out, "internal", CONSTRAINT_BUDGET)
    out["external_constraint_ok"] = constraint_ok(out, "external", CONSTRAINT_BUDGET)
    out = out.sort_values(["internal_accuracy_gain", "internal_accuracy"], ascending=False).reset_index(drop=True)
    out["internal_accuracy_rank"] = np.arange(1, len(out) + 1)
    out = out.sort_values(["external_accuracy_gain", "external_accuracy"], ascending=False).reset_index(drop=True)
    out["external_accuracy_rank"] = np.arange(1, len(out) + 1)
    return out


def plot_top(rank_df: pd.DataFrame, out_path: Path) -> None:
    """제약 조건을 통과한(없으면 전체 중) 상위 20개 후보의 내부/외부 정확도 증가, 민감도 손실, 특이도 증가를 3개 가로 막대 그래프로 그려 저장한다."""
    top = rank_df[rank_df["internal_constraint_ok"] & rank_df["external_constraint_ok"]].head(20).iloc[::-1]
    if top.empty:
        top = rank_df.head(20).iloc[::-1]
    labels = [f"{r.operating_point} {r.candidate_kind}" for r in top.itertuples()]
    y = np.arange(len(top))
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, 0.35 * len(top))), constrained_layout=True)
    axes[0].barh(y - 0.18, top["internal_accuracy_gain"] * 100, height=0.34, color="#4c78a8", label="OOF")
    axes[0].barh(y + 0.18, top["external_accuracy_gain"] * 100, height=0.34, color="#f58518", label="External")
    axes[1].barh(y - 0.18, top["internal_sensitivity_loss"] * 100, height=0.34, color="#4c78a8")
    axes[1].barh(y + 0.18, top["external_sensitivity_loss"] * 100, height=0.34, color="#f58518")
    axes[2].barh(y - 0.18, top["internal_specificity_gain"] * 100, height=0.34, color="#4c78a8")
    axes[2].barh(y + 0.18, top["external_specificity_gain"] * 100, height=0.34, color="#f58518")
    for ax, title in zip(axes, ["Accuracy gain", "Sensitivity loss", "Specificity gain"]):
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """AEC 점수/패턴 기반 재분류 후보와 그 결합 후보를 모두 생성해 내부·외부 정확도 증가 기준으로 순위를 매기고,
    제약 조건을 통과한 후보들을 정렬한 CSV와 상위 20개 그래프, 요약 JSON을 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    clinical_thresholds = {
        op: float(threshold_for_min_sensitivity(g["y"].astype(int), c_g, target))
        for op, target in OPS
    }
    cpos_g = {op: c_g >= clinical_thresholds[op] for op, _ in OPS}
    cpos_s = {op: c_s >= clinical_thresholds[op] for op, _ in OPS}

    score_models = score_model_candidates(g, s)
    base_candidates = score_threshold_candidates(score_models, cpos_g, cpos_s)
    base_candidates.extend(pattern_candidates(cpos_g, cpos_s))
    base_internal = []
    for cand in base_candidates:
        row = metric_for_candidate("g1090_internal", g, cpos_g[cand.op], cand)
        base_internal.append(row)
    base_internal_df = pd.DataFrame(base_internal)

    combo_candidates = []
    for op, _ in OPS:
        combo_candidates.extend(build_combo_candidates(op, base_candidates, base_internal_df, CONSTRAINT_BUDGET))
    all_candidates = base_candidates + combo_candidates
    rank_df = add_ranks(add_external_metrics(all_candidates, g, s, cpos_g, cpos_s))
    rank_df.to_csv(OUT_DIR / "aec_accuracy_all_candidates_ranked.csv", index=False)

    constrained = rank_df[rank_df["internal_constraint_ok"] & rank_df["external_constraint_ok"]].copy()
    constrained = constrained.sort_values(["external_accuracy_gain", "external_accuracy"], ascending=False)
    constrained.to_csv(OUT_DIR / "aec_accuracy_constrained_external_ranked.csv", index=False)
    oof_ranked = rank_df[rank_df["internal_constraint_ok"]].sort_values(["internal_accuracy_gain", "internal_accuracy"], ascending=False)
    oof_ranked.to_csv(OUT_DIR / "aec_accuracy_constrained_oof_ranked.csv", index=False)
    plot_top(constrained if not constrained.empty else rank_df, OUT_DIR / "aec_accuracy_top20_tradeoff.png")
    with (OUT_DIR / "aec_accuracy_rank_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "definition": "AEC-derived clinical-positive de-escalation candidates ranked by post-classification accuracy gain.",
                "aec_sources": [
                    "AEC-all logistic scores across k/C",
                    "direct-vote CNN branch pattern gates",
                    "score OR pattern combinations",
                    "pattern AND score guard combinations",
                ],
                "constraint_budget": CONSTRAINT_BUDGET,
                "constraint": "deesc_n>=10, specificity_gain>0, sensitivity_loss<=budget, exact sensitivity-loss p>=0.05",
                "note": "direct-vote CNN probabilities are AEC-derived but clinical-conditioned.",
                "n_base_candidates": len(base_candidates),
                "n_combo_candidates": len(combo_candidates),
                "n_all_candidates": len(all_candidates),
            },
            f,
            indent=2,
        )
    show_cols = [
        "external_accuracy_rank",
        "internal_accuracy_rank",
        "operating_point",
        "candidate_kind",
        "external_accuracy",
        "external_accuracy_gain",
        "external_accuracy_p",
        "external_sensitivity_loss",
        "external_sensitivity_loss_p",
        "external_specificity_gain",
        "external_deesc_n",
        "external_deesc_events",
        "external_deesc_event_rate",
        "internal_accuracy_gain",
        "internal_sensitivity_loss",
        "candidate_detail",
    ]
    print("\nTOP EXTERNAL ACCURACY, CONSTRAINED")
    print(constrained[show_cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nTOP OOF ACCURACY, INTERNAL-CONSTRAINED")
    print(oof_ranked[show_cols].head(30).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# 모든 AEC 기반 재분류 후보(및 결합 후보)를 생성해 내부/외부 정확도 증가 기준으로 순위를 매기고 제약 조건 통과 결과를 저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
