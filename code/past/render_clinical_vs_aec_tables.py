from __future__ import annotations

# Renders the saved result files of the two "past" pipelines
# (run_from_raw_standalone.py, main_aec_full_derivation_pipeline_simplified.py)
# as a clinical-vs-AEC-assisted summary table image, in the same visual style
# as outputs/9_aec_cnn_fcn/clinical_vs_aec_assisted_table.png.
#
# Run: python code/past/render_clinical_vs_aec_tables.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

TABLE_HEADER_BG = "#1c1c1c"
TABLE_HEADER_FG = "#ffffff"
TABLE_HEADER_SUB = "#b9b8b3"
TABLE_BAND_BG = "#f6f6f4"
TABLE_GRID = "#d9d8d3"
TABLE_DIVIDER = "#2a2a2a"
TABLE_GOOD = "#1a7a4c"
TABLE_BAD = "#c0392b"
TABLE_NRI_BG = "#d9e8fb"
TABLE_NRI_FG = "#1553b6"
TABLE_TEXT = "#161616"
TABLE_MUTED = "#4d4c48"
TABLE_SUBTEXT = "#6b6a66"
TABLE_PASS_BG = "#dcefe1"
TABLE_FAIL_BG = "#f8dcd8"


def exact_mcnemar_p(gain_n: int, loss_n: int) -> float:
    n = gain_n + loss_n
    if n == 0:
        return float("nan")
    return float(stats.binomtest(min(gain_n, loss_n), n, 0.5, alternative="two-sided").pvalue)


def clopper_pearson_one_sided_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return float("nan")
    if k == 0:
        return float(1 - alpha ** (1 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def confusion_result(cohort: str, sens: float, spec: float, n_event: int, n_nonevent: int) -> dict:
    # Recovers exact tp/fn/tn/fp counts from the reported sensitivity/specificity
    # fractions and the (integer) event/non-event totals -- same convention as
    # code/1_aec_residual_reclassify.py::evaluate_combined's matrix layout ([[tp,fn],[fp,tn]]).
    tp = int(round(sens * n_event))
    fn = n_event - tp
    tn = int(round(spec * n_nonevent))
    fp = n_nonevent - tn
    return {"cohort": cohort, "matrix": np.array([[tp, fn], [fp, tn]])}


def plot_confusion_matrix(ax: Axes, result: dict, title: str) -> None:
    # Same layout as code/1_aec_residual_reclassify.py::plot_confusion_matrix.
    matrix = result["matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center",
                    fontsize=13, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    ax.set_title(title, fontsize=11, fontweight="bold")


def save_before_after_confusion_matrices(rows: list[dict], out_path: Path, suptitle: str) -> None:
    # 2x2 grid matching the sibling pipelines' stage1_vs_stage2_confusion_matrix.png:
    # row 0 = internal cohort (before/after), row 1 = external cohort (before/after).
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 12.5))
    for ri, r in enumerate(rows):
        n_event = r["event"]
        n_nonevent = r["n"] - r["event"]
        before = confusion_result(r["cohort"], r["sens_clin"], r["spec_clin"], n_event, n_nonevent)
        after = confusion_result(r["cohort"], r["sens_aec"], r["spec_aec"], n_event, n_nonevent)
        plot_confusion_matrix(axes[ri, 0], before, f"{r['cohort']}: Clinical only (before)")
        plot_confusion_matrix(axes[ri, 1], after, f"{r['cohort']}: AEC-assisted (after)")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved confusion matrices to {out_path}")


def plot_clinical_vs_aec_table(rows: list[dict], out_path: Path, title: str) -> None:
    # Same layout as code/1_aec_residual_reclassify.py::plot_clinical_vs_aec_table.
    # Each row shows a cohort's clinical-only vs AEC-assisted sens/spec/acc, McNemar
    # p-values and Net NRI. The subtitle under the cohort name is the score AUC when
    # available, otherwise a free-text label (e.g. de-escalated n for gate-style
    # models that don't produce a continuous risk score).
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [("sens", "Sensitivity"), ("spec", "Specificity"), ("acc", "Accuracy")]
    row_h, header_h, footer_h, ni_row_h = 1.0, 1.7, 0.55, 0.6
    block_h = len(metrics) * row_h + ni_row_h
    total_h = header_h + len(rows) * block_h + footer_h

    col = {"cohort": (0.00, 0.15), "n": (0.15, 0.205), "event": (0.205, 0.26),
           "metric": (0.26, 0.40), "clin": (0.40, 0.62), "aec": (0.62, 0.90), "nri": (0.90, 1.00)}
    cx = lambda key: (col[key][0] + col[key][1]) / 2

    fig, ax = plt.subplots(figsize=(13.5, total_h * 0.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    header_bottom = total_h - header_h
    ax.add_patch(Rectangle((0, header_bottom), 1, header_h, facecolor=TABLE_HEADER_BG, edgecolor="none", zorder=1))
    header_main_y = header_bottom + header_h * 0.68
    header_sub_y = header_bottom + header_h * 0.28
    for key, label in [("cohort", "코호트"), ("n", "N"), ("event", "Event"), ("metric", "지표")]:
        ax.text(cx(key), header_bottom + header_h / 2, label, ha="center", va="center",
                color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("clin"), header_main_y, "Clinical only", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("clin"), header_sub_y, "sens / spec / acc", ha="center", va="center",
            color=TABLE_HEADER_SUB, fontsize=9.5)
    ax.text(cx("aec"), header_main_y, "AEC-assisted", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("aec"), header_sub_y, "sens / spec / acc (p)", ha="center", va="center",
            color=TABLE_HEADER_SUB, fontsize=9.5)
    ax.text(cx("nri"), header_bottom + header_h / 2, "Net\nNRI", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")

    def pfmt(p: float) -> str:
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    y_cursor = header_bottom
    for gi, r in enumerate(rows):
        block_top = y_cursor
        block_bottom = y_cursor - block_h
        if gi % 2 == 0:
            ax.add_patch(Rectangle((0, block_bottom), 1, block_h,
                                    facecolor=TABLE_BAND_BG, edgecolor="none", zorder=0))

        mid_y = (block_top + block_bottom) / 2
        ax.text(cx("cohort"), mid_y + 0.12, r["cohort"], ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=TABLE_TEXT)
        subtitle = f"AUC {r['auc']:.3f}" if r.get("auc") is not None else str(r.get("subtitle", ""))
        ax.text(cx("cohort"), mid_y - 0.22, subtitle, ha="center", va="center",
                fontsize=9.5, color=TABLE_GOOD)
        ax.text(cx("n"), mid_y, f"{r['n']}", ha="center", va="center", fontsize=12, color=TABLE_TEXT)
        ax.text(cx("event"), mid_y, f"{r['event']}", ha="center", va="center", fontsize=12, color=TABLE_TEXT)

        nri = r["net_nri"]
        box_w, box_h = 0.07, 0.9
        ax.add_patch(FancyBboxPatch((cx("nri") - box_w / 2, mid_y - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     linewidth=0, facecolor=TABLE_NRI_BG, zorder=2))
        ax.text(cx("nri"), mid_y, f"{nri:+d}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=TABLE_NRI_FG, zorder=3)

        for mi, (mkey, mlabel) in enumerate(metrics):
            row_top = block_top - mi * row_h
            row_bottom = row_top - row_h
            row_mid = (row_top + row_bottom) / 2

            ax.text(cx("metric"), row_mid, mlabel, ha="center", va="center", fontsize=11.5, color=TABLE_TEXT)

            clin_val, aec_val, p_val = r[f"{mkey}_clin"], r[f"{mkey}_aec"], r[f"{mkey}_p"]
            delta = aec_val - clin_val
            dcolor = TABLE_GOOD if delta >= 0 else TABLE_BAD

            ax.text(cx("clin"), row_mid, f"{clin_val:.3f}", ha="center", va="center",
                    fontsize=12, color=TABLE_MUTED)
            aec_x0, aec_x1 = col["aec"]
            ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.28, row_mid, f"{aec_val:.3f}",
                    ha="center", va="center", fontsize=12, color=TABLE_TEXT)
            ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.72, row_mid, f"({delta:+.3f}) {pfmt(p_val)}",
                    ha="center", va="center", fontsize=9.5, color=dcolor)

            ax.plot([col["metric"][0], 1], [row_bottom, row_bottom], color=TABLE_GRID,
                    linewidth=0.8, zorder=1)

        # NI test row (below the sens/spec/acc rows): each pipeline's own
        # pre-specified non-inferiority / sensitivity-preservation criterion
        # (these differ between the two source scripts -- see build_* below).
        ni_top = block_top - len(metrics) * row_h
        ni_bottom = ni_top - ni_row_h
        ni_mid = (ni_top + ni_bottom) / 2
        ni_pass = bool(r["ni_pass"])
        ni_color = TABLE_GOOD if ni_pass else TABLE_BAD

        ax.text(cx("metric"), ni_mid, "NI Test", ha="center", va="center",
                fontsize=11.5, color=TABLE_TEXT)
        ax.text(cx("clin"), ni_mid, r["ni_label"], ha="center", va="center",
                fontsize=10.5, color=TABLE_MUTED)
        aec_x0, aec_x1 = col["aec"]
        ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.5, ni_mid, r["ni_detail"],
                ha="center", va="center", fontsize=10.5, color=TABLE_TEXT)

        badge_w, badge_h = 0.07, 0.44 if ni_pass else 0.52
        ax.add_patch(FancyBboxPatch((cx("nri") - badge_w / 2, ni_mid - badge_h / 2), badge_w, badge_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     linewidth=0, facecolor=TABLE_PASS_BG if ni_pass else TABLE_FAIL_BG, zorder=2))
        if ni_pass:
            ax.text(cx("nri"), ni_mid, "PASS", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ni_color, zorder=3)
        else:
            ax.text(cx("nri"), ni_mid + 0.13, f"{r['ni_ci_upper'] * 100:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=ni_color, zorder=3)
            ax.text(cx("nri"), ni_mid - 0.11, "FAIL", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ni_color, zorder=3)

        y_cursor = block_bottom
        ax.plot([0, 1], [block_bottom, block_bottom], color=TABLE_DIVIDER, linewidth=1.4, zorder=2)

    footnote = ("* p < 0.05 (유의)    n.s. p ≥ 0.05 (비유의)    Net NRI: AEC 추가 시 순 재분류 개선 환자 수    "
                "NI Test: 민감도 손실 one-sided 95% CI 상한이 margin 이하이면 PASS (비열등성, 두 파이프라인 동일 기준)")
    ax.text(0.0, footer_h * 0.4, footnote, ha="left", va="center", fontsize=9, color=TABLE_SUBTEXT)

    fig.suptitle(title, x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=TABLE_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved clinical-vs-AEC table to {out_path}")


def build_full_derivation_table() -> None:
    # main_aec_full_derivation_pipeline_simplified.py: primary interpretable
    # 4-region AEC gate, reproduced/derived from raw AEC-128 curves.
    out_dir = ROOT / "outputs" / "full_derivation" / "full_derivation_output"
    metrics = pd.read_csv(out_dir / "final_s90_primary_and_cnn_metrics.csv")
    metrics = metrics[metrics["model"] == "primary_interpretable_4region"].set_index("cohort")

    rows = []
    for cohort_key, label in [("Gangnam internal", "internal"), ("Sinchon external", "external")]:
        m = metrics.loc[cohort_key]
        tp_lost, fp_removed = int(m["tp_lost"]), int(m["fp_removed"])
        rows.append({
            "cohort": label,
            "n": int(m["total_low_smi"] + m["total_nonlow_smi"]),
            "event": int(m["total_low_smi"]),
            "subtitle": f"De-esc n={int(m['deescalated_n'])}",
            "sens_clin": m["clinical_sensitivity"], "sens_aec": m["post_sensitivity"],
            "sens_p": exact_mcnemar_p(0, tp_lost),
            "spec_clin": m["clinical_specificity"], "spec_aec": m["post_specificity"],
            "spec_p": exact_mcnemar_p(fp_removed, 0),
            "acc_clin": m["clinical_accuracy"], "acc_aec": m["post_accuracy"],
            "acc_p": exact_mcnemar_p(fp_removed, tp_lost),
            "net_nri": fp_removed - tp_lost,
            "ni_label": f"formal NI, margin ≤{float(m['formal_NI_margin']) * 100:.1f}%p",
            "ni_detail": f"sens loss upper 95% CI = {float(m['sensitivity_loss_upper95_one_sided']) * 100:.2f}%p",
            "ni_ci_upper": float(m["sensitivity_loss_upper95_one_sided"]),
            "ni_pass": bool(m["formal_NI_pass"]),
        })

    out_dir_top = ROOT / "outputs" / "full_derivation"
    plot_clinical_vs_aec_table(
        rows, out_dir_top / "clinical_vs_aec_assisted_table.png",
        "clinical-only vs. AEC-assisted(Interpretable 4-region Gate) 성능 비교 (main_aec_full_derivation_pipeline_simplified)",
    )
    save_before_after_confusion_matrices(
        rows, out_dir_top / "stage1_vs_stage2_confusion_matrix.png",
        "Clinical-only vs. AEC-assisted(Interpretable 4-region Gate) confusion matrices (main_aec_full_derivation_pipeline_simplified)",
    )


def build_run_from_raw_standalone_table() -> None:
    # run_from_raw_standalone.py: locked, smoothed de-escalation gate (2-of-4 rule)
    # at the S90 operating point.
    gate_dir = ROOT / "outputs" / "run_from_raw_standalone" / "aec_lock_smoothed_deesc_gate"
    details = pd.read_csv(gate_dir / "locked_gate_operating_point_details.csv")
    details = details[details["operating_point"] == "S90"].set_index("dataset")
    auc = pd.read_csv(gate_dir / "locked_gate_auc_summary.csv").set_index("model")

    # Same underlying cohorts as the full-derivation pipeline (both scripts run on
    # the same Gangnam-internal / Sinchon-external patients).
    total_n = {"gangnam_internal": 1090, "sinchon_external": 926}
    total_event = {"gangnam_internal": 129, "sinchon_external": 141}

    # Judged against the SAME formal non-inferiority test as the full-derivation
    # pipeline (Clopper-Pearson one-sided 95% CI on sensitivity loss <= 5%p
    # margin), for apples-to-apples comparability across pipelines. As of the
    # combo_search update in run_from_raw_standalone.py, this formal margin is
    # now a hard constraint on the S90 locking decision itself (not just a
    # looser proxy), so the PASS/FAIL shown here reflects the actual locking
    # criterion -- see locked_gate_summary.json's "constraints" field.
    ni_margin = 0.05

    rows = []
    for dataset, auc_col, label in [
        ("gangnam_internal", "internal_auc", "internal"),
        ("sinchon_external", "external_auc", "external"),
    ]:
        d = details.loc[dataset]
        tp_lost, fp_removed = int(d["tp_lost"]), int(d["fp_removed"])
        ni_upper95 = clopper_pearson_one_sided_upper(tp_lost, total_event[dataset])
        rows.append({
            "cohort": label,
            "n": total_n[dataset],
            "event": total_event[dataset],
            "auc": float(auc.loc["clinical_only", auc_col]),
            "sens_clin": d["clinical_sensitivity"], "sens_aec": d["post_sensitivity"],
            "sens_p": d["sensitivity_loss_p_exact"],
            "spec_clin": d["clinical_specificity"], "spec_aec": d["post_specificity"],
            "spec_p": d["specificity_gain_p_exact"],
            "acc_clin": d["clinical_accuracy"], "acc_aec": d["post_accuracy"],
            "acc_p": d["accuracy_delta_p_mcnemar"],
            "net_nri": fp_removed - tp_lost,
            "ni_label": f"formal NI, margin ≤{ni_margin * 100:.1f}%p",
            "ni_detail": f"sens loss upper 95% CI = {ni_upper95 * 100:.2f}%p",
            "ni_ci_upper": ni_upper95,
            "ni_pass": ni_upper95 <= ni_margin,
        })

    out_dir_top = ROOT / "outputs" / "run_from_raw_standalone"
    plot_clinical_vs_aec_table(
        rows, out_dir_top / "clinical_vs_aec_assisted_table.png",
        "clinical-only vs. AEC-assisted(Locked Smoothed De-esc Gate) 성능 비교 (run_from_raw_standalone, S90)",
    )
    save_before_after_confusion_matrices(
        rows, out_dir_top / "stage1_vs_stage2_confusion_matrix.png",
        "Clinical-only vs. AEC-assisted(Locked Smoothed De-esc Gate) confusion matrices (run_from_raw_standalone, S90)",
    )


if __name__ == "__main__":
    build_full_derivation_table()
    build_run_from_raw_standalone_table()
