from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    build_candidate_bank,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
)
from aec_oof_auc_max_search import (  # noqa: E402
    Candidate,
    auc_p,
    crossfit_candidate,
    curve_features,
    load_direct_vote_features,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_all_reclassification"
OPS_EXTENDED = [
    ("S75", 0.75),
    ("S80", 0.80),
    ("S82.5", 0.825),
    ("S85", 0.85),
    ("S87.5", 0.875),
    ("S90", 0.90),
    ("S95", 0.95),
]


def build_aec_all_matrices(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """AEC 곡선 특징, 후보뱅크 특징, direct-vote 특징을 결합해 내부(g)/외부(s) 데이터셋의 aec_all 특징 행렬을 만든다."""
    curve_g = curve_features(g)
    curve_s = curve_features(s)
    bank_g = build_candidate_bank(g["norm"]).to_numpy(dtype=float)
    bank_s = build_candidate_bank(s["norm"]).to_numpy(dtype=float)
    vote_g, vote_s, _ = load_direct_vote_features()
    return np.column_stack([curve_g, bank_g, vote_g]), np.column_stack([curve_s, bank_s, vote_s])


def plot_details(details: pd.DataFrame, out_path: Path) -> None:
    """운영점(operating point)별로 내부/외부 데이터셋의 특이도 증가, 민감도 손실, 저(低)SMI 재분류율을 3개 서브플롯 선그래프로 그려 저장한다."""
    labels = [x[0] for x in OPS_EXTENDED]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for dataset, color, ls in [
        ("g1090_internal", "#4c78a8", "-"),
        ("sdata_external", "#f58518", "--"),
    ]:
        sub = details[details["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[1].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[2].plot(x, sub["deesc_event_rate"] * 100, marker="o", color=color, ls=ls, label=dataset)
    for ax, title, ylabel in [
        (axes[0], "Specificity gain", "percentage points"),
        (axes[1], "Sensitivity loss", "percentage points"),
        (axes[2], "De-escalated low SMI rate", "%"),
    ]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """aec_all 특징으로 로지스틱 회귀(l2, k40, C0.1)를 학습해 내부 OOF/외부 재적합/외부 fold 앙상블 점수와 AUC를 구하고,
    임상 양성이면서 AEC 음성인 경우를 저(低)위험군으로 재분류하는 규칙을 여러 목표 민감도(operating point)에서 적용해 특이도 증가/민감도 손실을 계산, 결과를 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    xg, xs = build_aec_all_matrices(g, s)
    cand = Candidate("aec_all__l2_k40_C0.1", "aec_all", "logit_l2", k=40, c=0.1)
    score_g, score_s_fold, score_s = crossfit_candidate(cand, xg, g["y"].astype(int), xs)

    auc_rows = []
    for name, y, score in [
        ("aec_all_oof_internal", g["y"].astype(int), score_g),
        ("aec_all_external_refit", s["y"].astype(int), score_s),
        ("aec_all_external_fold_ensemble", s["y"].astype(int), score_s_fold),
        ("clinical_internal", g["y"].astype(int), clinical_oof),
        ("clinical_external", s["y"].astype(int), clinical_ext),
    ]:
        auc, p = auc_p(y, score)
        auc_rows.append({"score": name, "auc": auc, "p": p})
    pd.DataFrame(auc_rows).to_csv(OUT_DIR / "aec_all_reclassification_auc.csv", index=False)

    rows = []
    threshold_rows = []
    for op, target in OPS_EXTENDED:
        clinical_th = float(threshold_for_min_sensitivity(g["y"].astype(int), c_g, target))
        aec_th = float(threshold_for_min_sensitivity(g["y"].astype(int), score_g, target))
        threshold_rows.append(
            {
                "operating_point": op,
                "target_internal_sensitivity": target,
                "clinical_z_threshold_internal": clinical_th,
                "aec_all_score_threshold_internal": aec_th,
            }
        )
        for dataset, d, clinical_z, aec_score in [
            ("g1090_internal", g, c_g, score_g),
            ("sdata_external", s, c_s, score_s),
        ]:
            cpos = clinical_z >= clinical_th
            deesc = cpos & (aec_score < aec_th)
            row = deesc_metric_row(
                dataset,
                "clinical_positive_AND_aec_all_positive",
                "aec_all__l2_k40_C0.1",
                op,
                d["y"].astype(int),
                cpos,
                deesc,
            )
            row["target_internal_sensitivity"] = target
            row["aec_threshold_source"] = "internal OOF same target sensitivity"
            row["net_reclassification_delta"] = row["specificity_gain"] - row["sensitivity_loss"]
            rows.append(row)
    details = pd.DataFrame(rows)
    details.to_csv(OUT_DIR / "aec_all_reclassification_details.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "aec_all_reclassification_thresholds.csv", index=False)
    plot_details(details, OUT_DIR / "aec_all_reclassification_plot.png")
    with (OUT_DIR / "aec_all_reclassification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "aec_all__l2_k40_C0.1",
                "features": "smooth patient-wise norm curve, slope, curvature, handcrafted AEC bank, direct-vote CNN probabilities",
                "reclassification_rule": "clinical positive AND AEC-all positive; de-escalate clinical-positive/AEC-negative",
                "thresholds": "clinical and AEC thresholds set on g1090 internal OOF to each target sensitivity, then applied unchanged to sdata external",
            },
            f,
            indent=2,
        )

    show = details[
        [
            "dataset",
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
    print("\nAEC-ALL RECLASSIFICATION")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# aec_all 모델을 학습해 임상 양성/AEC 음성 환자를 재분류(de-escalate)했을 때의 특이도 증가·민감도 손실을 내부/외부 데이터에서 평가하고 결과를 저장한다.
if __name__ == "__main__":
    main()
