from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_all_reclassification import OPS_EXTENDED, build_aec_all_matrices  # noqa: E402
from aec_lock_smoothed_deesc_gate import DATA_DIR, clinical_scores, deesc_metric_row, load_dataset  # noqa: E402
from aec_oof_auc_max_search import Candidate, crossfit_candidate  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_all_reclassification_threshold_search"
LOSS_BUDGETS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
MIN_DEESC_N = 10


def candidate_cutoffs(score: np.ndarray, cpos: np.ndarray) -> np.ndarray:
    """임상 양성(cpos) 대상자의 AEC 점수 분포에서 후보 컷오프(분위수 기반)를 생성한다."""
    vals = np.asarray(score[cpos], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    qs = np.unique(np.r_[np.linspace(0.01, 0.95, 95), np.linspace(0.955, 0.995, 9)])
    cuts = np.quantile(vals, qs)
    return np.unique(cuts)


def search_op_thresholds(
    g: dict,
    c_g: np.ndarray,
    score_g: np.ndarray,
    clinical_thresholds: dict[str, float],
) -> pd.DataFrame:
    """각 임상 운영점(operating point)마다 후보 AEC 컷오프들을 모두 적용해 재분류 지표(민감도 손실, 특이도 증가 등)를 계산한 표를 만든다."""
    rows = []
    for op, _target in OPS_EXTENDED:
        cpos = c_g >= clinical_thresholds[op]
        for cut in candidate_cutoffs(score_g, cpos):
            deesc = cpos & (score_g < cut)
            row = deesc_metric_row(
                "g1090_internal",
                "aec_all_threshold_search",
                "aec_all__l2_k40_C0.1",
                op,
                g["y"].astype(int),
                cpos,
                deesc,
            )
            row["aec_cutoff"] = float(cut)
            rows.append(row)
    return pd.DataFrame(rows)


def select_by_budget(search: pd.DataFrame, budget: float, require_p_loss: bool = True) -> pd.DataFrame:
    """민감도 손실 예산(budget), 최소 재분류 표본 수, (옵션으로) 손실 유의성 조건을 만족하는 후보 중 특이도 증가가 가장 큰 AEC 컷오프를 운영점별로 선택한다. 조건을 만족하는 후보가 없으면 표본이 있는 것 중 안전한 값으로 대체한다."""
    selected = []
    for op, _target in OPS_EXTENDED:
        sub = search[search["operating_point"].eq(op)].copy()
        ok = sub["deesc_n"].ge(MIN_DEESC_N) & sub["sensitivity_loss"].le(budget + 1e-12)
        if require_p_loss:
            ok &= sub["sensitivity_loss_p_exact"].fillna(1.0).ge(0.05)
        ok &= sub["specificity_gain"].gt(0)
        cand = sub[ok].copy()
        if cand.empty:
            # Fall back to the safest nonempty option so the table is complete.
            cand = sub[sub["deesc_n"].ge(1)].copy()
            cand["fallback_no_candidate_under_budget"] = True
        else:
            cand["fallback_no_candidate_under_budget"] = False
        cand["selection_score"] = (
            cand["specificity_gain"]
            - 0.50 * cand["sensitivity_loss"]
            - 0.001 * cand["deesc_event_rate"].fillna(0)
        )
        best = cand.sort_values(["specificity_gain", "selection_score"], ascending=False).iloc[0].to_dict()
        best["loss_budget"] = budget
        best["require_p_loss_ge_0.05"] = require_p_loss
        selected.append(best)
    return pd.DataFrame(selected)


def apply_selected(
    selected: pd.DataFrame,
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    score_g: np.ndarray,
    score_s: np.ndarray,
    clinical_thresholds: dict[str, float],
) -> pd.DataFrame:
    """budget별로 선택된 AEC 컷오프를 내부(g)와 외부(s) 데이터셋 양쪽에 적용해 재분류 지표를 계산한 표를 만든다."""
    rows = []
    for _, sel in selected.iterrows():
        op = str(sel["operating_point"])
        cut = float(sel["aec_cutoff"])
        for dataset, d, clinical_z, aec_score in [
            ("g1090_internal", g, c_g, score_g),
            ("sdata_external", s, c_s, score_s),
        ]:
            cpos = clinical_z >= clinical_thresholds[op]
            deesc = cpos & (aec_score < cut)
            row = deesc_metric_row(
                dataset,
                f"aec_all_budget_{float(sel['loss_budget']):.2f}",
                "aec_all__l2_k40_C0.1",
                op,
                d["y"].astype(int),
                cpos,
                deesc,
            )
            row["loss_budget"] = float(sel["loss_budget"])
            row["aec_cutoff"] = cut
            row["fallback_no_candidate_under_budget"] = bool(sel["fallback_no_candidate_under_budget"])
            row["net_reclassification_delta"] = row["specificity_gain"] - row["sensitivity_loss"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_budget(applied: pd.DataFrame) -> pd.DataFrame:
    """민감도 손실 예산(budget)과 데이터셋별로 특이도 증가/민감도 손실/재분류 이벤트 등의 요약 통계(평균, 최소/최대, 합계)를 계산한다."""
    rows = []
    for (budget, dataset), sub in applied.groupby(["loss_budget", "dataset"]):
        rows.append(
            {
                "loss_budget": budget,
                "dataset": dataset,
                "mean_specificity_gain": sub["specificity_gain"].mean(),
                "min_specificity_gain": sub["specificity_gain"].min(),
                "mean_sensitivity_loss": sub["sensitivity_loss"].mean(),
                "max_sensitivity_loss": sub["sensitivity_loss"].max(),
                "min_sensitivity_loss_p": sub["sensitivity_loss_p_exact"].fillna(1.0).min(),
                "max_deesc_event_rate": sub["deesc_event_rate"].fillna(0.0).max(),
                "mean_deesc_n": sub["deesc_n"].mean(),
                "sum_deesc_events": sub["deesc_events"].sum(),
            }
        )
    return pd.DataFrame(rows)


def plot_external(applied: pd.DataFrame, out_path: Path) -> None:
    """외부 데이터셋에 대해 여러 민감도 손실 예산(budget)별로 운영점에 따른 특이도 증가와 민감도 손실 추이를 두 개의 선그래프로 그려 저장한다."""
    labels = [op for op, _ in OPS_EXTENDED]
    budgets = [0.02, 0.03, 0.04, 0.05]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for budget, color in zip(budgets, colors):
        sub = applied[
            applied["dataset"].eq("sdata_external") & applied["loss_budget"].eq(budget)
        ].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", color=color, label=f"budget {budget:.0%}")
        axes[1].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=color, label=f"budget {budget:.0%}")
    for ax, title in [(axes[0], "External specificity gain"), (axes[1], "External sensitivity loss")]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """aec_all 모델의 내부 OOF 점수로 운영점별 후보 컷오프를 탐색하고, 여러 민감도 손실 예산(LOSS_BUDGETS)마다 최적 컷오프를 선택,
    내부/외부 데이터에 동일 컷오프를 적용한 재분류 결과와 예산별 요약을 계산하여 CSV/그래프/JSON으로 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    xg, xs = build_aec_all_matrices(g, s)
    cand = Candidate("aec_all__l2_k40_C0.1", "aec_all", "logit_l2", k=40, c=0.1)
    score_g, _score_s_fold, score_s = crossfit_candidate(cand, xg, g["y"].astype(int), xs)

    clinical_thresholds = {
        op: float(threshold_for_min_sensitivity(g["y"].astype(int), c_g, target))
        for op, target in OPS_EXTENDED
    }
    search = search_op_thresholds(g, c_g, score_g, clinical_thresholds)
    search.to_csv(OUT_DIR / "aec_all_threshold_search_internal_candidates.csv", index=False)

    all_selected = []
    all_applied = []
    for budget in LOSS_BUDGETS:
        selected = select_by_budget(search, budget, require_p_loss=True)
        applied = apply_selected(selected, g, s, c_g, c_s, score_g, score_s, clinical_thresholds)
        all_selected.append(selected)
        all_applied.append(applied)
    selected_all = pd.concat(all_selected, ignore_index=True)
    applied_all = pd.concat(all_applied, ignore_index=True)
    summary = summarize_budget(applied_all)
    selected_all.to_csv(OUT_DIR / "aec_all_threshold_search_selected_internal.csv", index=False)
    applied_all.to_csv(OUT_DIR / "aec_all_threshold_search_applied_details.csv", index=False)
    summary.to_csv(OUT_DIR / "aec_all_threshold_search_budget_summary.csv", index=False)
    plot_external(applied_all, OUT_DIR / "aec_all_threshold_search_external_tradeoff.png")
    with (OUT_DIR / "aec_all_threshold_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "aec_all__l2_k40_C0.1",
                "selection": "For each clinical operating point, choose the AEC-all cutoff on g1090 internal OOF that maximizes specificity gain under a sensitivity-loss budget and exact p_loss >= 0.05.",
                "applied_to": "same cutoff applied to sdata external",
                "loss_budgets": LOSS_BUDGETS,
                "min_deesc_n": MIN_DEESC_N,
            },
            f,
            indent=2,
        )
    show = applied_all[
        applied_all["dataset"].eq("sdata_external")
        & applied_all["loss_budget"].isin([0.02, 0.03, 0.04, 0.05])
    ][
        [
            "loss_budget",
            "operating_point",
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
            "net_reclassification_delta",
        ]
    ]
    print("\nEXTERNAL THRESHOLD SEARCH DETAILS")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nBUDGET SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# 내부 데이터에서 민감도 손실 예산별 최적 AEC 컷오프를 탐색하고, 이를 내부/외부 데이터에 적용해 재분류 성능을 비교·저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
