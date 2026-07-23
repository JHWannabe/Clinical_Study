from __future__ import annotations

# Cohort-swapped rerun of code/stage2_model.py (clinical + AEC late-fusion multimodal
# classifier): the original fits/OOFs Stage 2 on internal=gangnam.xlsx and transfers
# the frozen ensemble to external=sinchon.xlsx. This script swaps the roles --
# internal=sinchon.xlsx (5-fold OOF), external=gangnam.xlsx (frozen-model transfer) --
# reusing every model/training/evaluation function from stage2_model.py as-is; only the
# Stage-1/Stage-2 data source (stage2_dataset_swap instead of stage2_dataset) and
# OUTPUT_DIR are swapped.
# Run: python code/0723/stage2_model_multimodal_swap.py

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
baseline = import_module("clinic-only_baseline")
s2m = import_module("stage2_model")
stage2_swap = import_module("stage2_dataset_swap")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0723" / "stage2_model_swap"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal=sinchon: 5-fold OOF for an unbiased internal estimate ---
    screen = stage2_swap.fit_internal_screen()
    stage1_rows_all_int, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = stage2_swap.build_stage2_inputs(screen)
    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    score_standardizer = s2m.fit_score_standardizer(stage1_rows_int["score"].to_numpy(dtype=np.float32))
    x_clin_int, x_aec_int = s2m._to_tensors(stage2_input_clin_int, stage2_input_aec_int, stage1_rows_int,
                                             score_standardizer=score_standardizer)

    oof, fold_loss_histories = s2m.oof_scores(x_clin_int, x_aec_int, y_int)

    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    th = s2m.choose_stage2_threshold(y_all_int, pos_mask_int, oof, stage1_only_int["sens"], stage1_only_int["spec"])

    # --- freeze: refit on the full internal(=sinchon) Stage-2 cohort, transfer to external(=gangnam) ---
    model, final_loss_history = s2m.fit_final_model(x_clin_int, x_aec_int, y_int)
    s2m.plot_loss_curves(fold_loss_histories, final_loss_history, OUTPUT_DIR / "loss_curve.png")

    stage1_rows_all_ext, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = stage2_swap.build_stage2_inputs_external(screen)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)
    x_clin_ext, x_aec_ext = s2m._to_tensors(stage2_input_clin_ext, stage2_input_aec_ext, stage1_rows_ext,
                                             score_standardizer=score_standardizer)
    score_ext = model.predict_proba(x_clin_ext, x_aec_ext)

    result_int = baseline.evaluate("internal", y_int, oof >= th, th)
    result_ext = baseline.evaluate("external", y_ext, score_ext >= th, th)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Stage-2 Late Fusion (clinical + AEC-128), swapped cohorts, screen-positive only", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_int)
    baseline.plot_confusion_matrix(axes[1], result_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    auc_int = baseline.auc_significance_stats(y_int, oof)
    baseline.plot_roc_curve(y_int, oof, auc_int, OUTPUT_DIR / "roc_curve_internal.png",
                             title="Stage-2 Late Fusion: ROC (internal=sinchon, OOF)")

    auc_ext = baseline.auc_significance_stats(y_ext, score_ext)
    baseline.plot_roc_curve(y_ext, score_ext, auc_ext, OUTPUT_DIR / "roc_curve_external.png",
                             title="Stage-2 Late Fusion: ROC (external=gangnam, frozen internal model)")

    pred_all_int = s2m.combine_predictions(pos_mask_int, oof, th)
    y_all_ext, pos_mask_ext, pred_all_ext = s2m.final_pipeline_labels(stage1_rows_all_ext, score_ext, th)

    stage1_score_int = stage1_rows_all_int["score"].to_numpy()
    stage1_score_ext = stage1_rows_all_ext["score"].to_numpy()
    full_score_int = s2m.combine_full_pipeline_score(stage1_score_int, pos_mask_int, oof, screen["th"])
    full_score_ext = s2m.combine_full_pipeline_score(stage1_score_ext, pos_mask_ext, score_ext, screen["th"])

    auc_stage1_int = baseline.auc_significance_stats(y_all_int, stage1_score_int)
    auc_full_int = baseline.auc_significance_stats(y_all_int, full_score_int)
    auc_stage1_ext = baseline.auc_significance_stats(y_all_ext, stage1_score_ext)
    auc_full_ext = baseline.auc_significance_stats(y_all_ext, full_score_ext)

    # Whole-cohort (not screen-positive-only) full-pipeline ROC, alongside the
    # Stage-2-only roc_curve_internal/external.png plotted above.
    baseline.plot_roc_curve(y_all_int, full_score_int, auc_full_int, OUTPUT_DIR / "roc_curve_internal_full_pipeline.png",
                             title="Full Pipeline: ROC (internal=sinchon, whole cohort)")
    baseline.plot_roc_curve(y_all_ext, full_score_ext, auc_full_ext, OUTPUT_DIR / "roc_curve_external_full_pipeline.png",
                             title="Full Pipeline: ROC (external=gangnam, whole cohort)")

    delong_int = s2m.delong_paired_auc_test(y_all_int.astype(float), stage1_score_int, full_score_int)
    delong_ext = s2m.delong_paired_auc_test(y_all_ext.astype(float), stage1_score_ext, full_score_ext)
    print(f"[internal=sinchon] Stage-1 AUC={auc_stage1_int['auc']:.3f} [{auc_stage1_int['ci_lower']:.3f}, {auc_stage1_int['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full_int['auc']:.3f} [{auc_full_int['ci_lower']:.3f}, {auc_full_int['ci_upper']:.3f}]  "
          f"DeLong diff={delong_int['diff']:+.4f} p={delong_int['p_value']:.4f}")
    print(f"[external=gangnam] Stage-1 AUC={auc_stage1_ext['auc']:.3f} [{auc_stage1_ext['ci_lower']:.3f}, {auc_stage1_ext['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full_ext['auc']:.3f} [{auc_full_ext['ci_lower']:.3f}, {auc_full_ext['ci_upper']:.3f}]  "
          f"DeLong diff={delong_ext['diff']:+.4f} p={delong_ext['p_value']:.4f}")

    s2m.plot_stage1_vs_full_pipeline_roc([
        {"label": "internal=sinchon", "y": y_all_int, "stage1_score": stage1_score_int, "stage1_auc": auc_stage1_int,
         "full_score": full_score_int, "full_auc": auc_full_int, "delong_p": delong_int["p_value"]},
        {"label": "external=gangnam", "y": y_all_ext, "stage1_score": stage1_score_ext, "stage1_auc": auc_stage1_ext,
         "full_score": full_score_ext, "full_auc": auc_full_ext, "delong_p": delong_ext["p_value"]},
    ], OUTPUT_DIR / "roc_comparison_stage1_vs_full_pipeline.png")

    pd.DataFrame([
        {"cohort": "internal(sinchon)", "stage1_auc": auc_stage1_int["auc"], "stage1_ci_lower": auc_stage1_int["ci_lower"],
         "stage1_ci_upper": auc_stage1_int["ci_upper"], "full_pipeline_auc": auc_full_int["auc"],
         "full_pipeline_ci_lower": auc_full_int["ci_lower"], "full_pipeline_ci_upper": auc_full_int["ci_upper"],
         "auc_diff": delong_int["diff"], "delong_z": delong_int["z"], "delong_p_value": delong_int["p_value"],
         "significant_p05": bool(np.isfinite(delong_int["p_value"]) and delong_int["p_value"] < 0.05)},
        {"cohort": "external(gangnam)", "stage1_auc": auc_stage1_ext["auc"], "stage1_ci_lower": auc_stage1_ext["ci_lower"],
         "stage1_ci_upper": auc_stage1_ext["ci_upper"], "full_pipeline_auc": auc_full_ext["auc"],
         "full_pipeline_ci_lower": auc_full_ext["ci_lower"], "full_pipeline_ci_upper": auc_full_ext["ci_upper"],
         "auc_diff": delong_ext["diff"], "delong_z": delong_ext["z"], "delong_p_value": delong_ext["p_value"],
         "significant_p05": bool(np.isfinite(delong_ext["p_value"]) and delong_ext["p_value"] < 0.05)},
    ]).to_csv(OUTPUT_DIR / "stage1_vs_full_pipeline_auc.csv", index=False)

    stage1_only_ext = baseline.evaluate("external / stage-1 only", y_all_ext, pos_mask_ext, screen["th"])
    result_final_int = baseline.evaluate("internal", y_all_int, pred_all_int, th)
    result_final_ext = baseline.evaluate("external", y_all_ext, pred_all_ext, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Stage 1 only (Clinical-only S90 screen), swapped cohorts", fontsize=13, fontweight="bold")
    for ax, result, cohort_label in [(axes[0], stage1_only_int, "internal=sinchon"), (axes[1], stage1_only_ext, "external=gangnam")]:
        matrix = result["matrix"]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            label = [["TP", "FN"], ["FP", "TN"]][i][j]
            ax.text(j, i, f"{label}\n{matrix[i, j]}", ha="center", va="center", fontsize=13,
                    color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Positive", "Actual Negative"])
        ax.set_title(f"{cohort_label}\n(threshold={result['th']:.3f})", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage1_only.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Full Pipeline (Stage 1 screen + Stage 2 late fusion), swapped cohorts", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_final_int)
    baseline.plot_confusion_matrix(axes[1], result_final_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    sens_delta_int = result_final_int["sens"] - stage1_only_int["sens"]
    spec_delta_int = result_final_int["spec"] - stage1_only_int["spec"]
    sens_delta_ext = result_final_ext["sens"] - stage1_only_ext["sens"]
    spec_delta_ext = result_final_ext["spec"] - stage1_only_ext["spec"]

    ok_int = s2m.ni_pass_fail(stage1_only_int["sens"], result_final_int["sens"], stage1_only_int["spec"], result_final_int["spec"])
    ok_ext = s2m.ni_pass_fail(stage1_only_ext["sens"], result_final_ext["sens"], stage1_only_ext["spec"], result_final_ext["spec"])

    for cohort, sens_delta, spec_delta, stage1_res, ok in [
        ("internal=sinchon", sens_delta_int, spec_delta_int, stage1_only_int, ok_int),
        ("external=gangnam", sens_delta_ext, spec_delta_ext, stage1_only_ext, ok_ext),
    ]:
        sens_floor = stage1_res["sens"] * (1 - s2m.SENS_LOSS_RATIO_MARGIN)
        print(f"[{cohort}] sens_delta={sens_delta:+.3f} spec_delta={spec_delta:+.3f} "
              f"(sens floor={sens_floor:.3f}, margin={s2m.SENS_LOSS_RATIO_MARGIN:.0%} relative) -> {'PASS' if ok else 'FAIL'}")

    pipeline_summary = pd.DataFrame([
        {"cohort": "internal(sinchon)", "sens_before": stage1_only_int["sens"], "sens_after": result_final_int["sens"],
         "sens_delta": sens_delta_int, "sens_floor": stage1_only_int["sens"] * (1 - s2m.SENS_LOSS_RATIO_MARGIN),
         "spec_before": stage1_only_int["spec"], "spec_after": result_final_int["spec"],
         "spec_delta": spec_delta_int, "pass": ok_int},
        {"cohort": "external(gangnam)", "sens_before": stage1_only_ext["sens"], "sens_after": result_final_ext["sens"],
         "sens_delta": sens_delta_ext, "sens_floor": stage1_only_ext["sens"] * (1 - s2m.SENS_LOSS_RATIO_MARGIN),
         "spec_before": stage1_only_ext["spec"], "spec_after": result_final_ext["spec"],
         "spec_delta": spec_delta_ext, "pass": ok_ext},
    ])
    pipeline_summary_path = OUTPUT_DIR / "final_pipeline_summary.csv"
    pipeline_summary.to_csv(pipeline_summary_path, index=False)
    print(f"Saved final pipeline summary to {pipeline_summary_path}")

    table_rows = [
        s2m.build_clinical_vs_aec_row("internal(sinchon)", y_all_int, pos_mask_int, pred_all_int, stage1_only_int, result_final_int, auc_int["auc"]),
        s2m.build_clinical_vs_aec_row("external(gangnam)", y_all_ext, pos_mask_ext, pred_all_ext, stage1_only_ext, result_final_ext, auc_ext["auc"]),
    ]
    s2m.plot_clinical_vs_aec_table(
        table_rows, OUTPUT_DIR / "clinical_vs_aec_assisted_table.png",
        "clinical-only vs. AEC-assisted(Late Fusion) 성능 비교, 코호트 스왑 (Stage-1 vs Stage-1+Stage-2)",
    )

    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "clinical_vs_aec_assisted_summary.csv", index=False)
    print("\n=== PRIMARY significance test: NRI / McNemar (reclassification) ===")
    for r in table_rows:
        print(f"[{r['cohort']}] Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
              f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
              f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})  "
              f"acc: {r['acc_clin']:.3f}->{r['acc_aec']:.3f} (p={r['acc_p']:.4f})")
    print("=== Secondary: whole-curve AUC / DeLong ===")
    print(f"[internal=sinchon] DeLong p={delong_int['p_value']:.4f}   [external=gangnam] DeLong p={delong_ext['p_value']:.4f}")


if __name__ == "__main__":
    main()
