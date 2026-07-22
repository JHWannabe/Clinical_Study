from __future__ import annotations

# Experiment: does residualizing the AEC-128 curve against clinical variables (age/
# height/weight/sex) before feeding it into Stage-2's late-fusion model recover more
# incremental signal than the raw (patient-normalized only) curve does?
#
# Motivation: [[project_aec_curve_bmi_confound]] showed the raw AEC-128 curve mostly
# re-encodes BMI (Simpson's-paradox signature) rather than carrying genuinely new
# information beyond clinical vars. stage2_model_no_aec_ablation.py confirmed this
# empirically at the Stage-2 level: with the AEC branch removed entirely (z_clin-only),
# Net NRI is +63 internal / +20 external, vs. +71 / +34 with the raw-AEC branch --
# i.e. raw AEC's OWN incremental contribution is only +8 internal / +14 external, most
# of it re-deriving what z_clin already knows.
#
# Design: fit a per-slice OLS regression of aec_i ~ CLIN_COLS + intercept on the internal
# Stage-2 (screen-positive) cohort only; freeze those coefficients and apply them to
# external too (same "fit internal, freeze, transfer" pattern as fit_score_standardizer /
# fit_internal_screen elsewhere in this codebase). Residual curves then replace raw
# curves as AecBranch's input, with every other part of the pipeline (frozen_lr clinical
# branch, convpool AEC branch, training protocol, threshold selection, Net NRI) held
# identical to stage2_model.py -- this script mirrors stage2_model.py's main() end to
# end (same plots/tables/CSVs, saved under its own OUTPUT_DIR) so the two are directly
# comparable artifact-for-artifact.
#
# RESULT (2026-07-22): Net NRI +77 internal / +52 external -- both higher than raw AEC's
# +71 / +34. BUT the NI test (sens retains >=95% of stage-1-only, chosen on internal only,
# never touched by external) PASSES internal (0.868 vs floor 0.862) and FAILS external by
# a hair (0.879 vs floor 0.883, short by 0.32pp) -- see final_pipeline_summary.csv "pass"
# column. Decision: production stage2_model.py was NOT switched to residualized AEC
# because of this external NI fail; this script stays exploratory/appendix-only. Higher
# Net NRI does not imply NI PASS -- they measure different things (net reclassification
# count vs. a specific sensitivity-floor constraint) and the threshold is chosen on
# internal only by design, so it isn't guaranteed to generalize to external.
#
# Run: python code/stage2_model_residualized_aec.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2 = import_module("stage2_dataset")
s2model = import_module("stage2_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_model_residualized_aec"

CLIN_COLS = stage2.CLIN_COLS
AEC_COLS = stage2.AEC_COLS


def fit_aec_residualizer(clin_df: pd.DataFrame, aec_df: pd.DataFrame) -> np.ndarray:
    # OLS: aec_i ~ CLIN_COLS + intercept, fit jointly across all 128 slices via a single
    # lstsq call (n x 5 design matrix, n x 128 targets) -- fit on internal Stage-2 cohort
    # only, coefficients frozen and reused for external (see module docstring).
    x = clin_df[CLIN_COLS].to_numpy(dtype=np.float64)
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    y = aec_df[AEC_COLS].to_numpy(dtype=np.float64)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)  # (5, 128)
    return coef


def residualize(clin_df: pd.DataFrame, aec_df: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    x = clin_df[CLIN_COLS].to_numpy(dtype=np.float64)
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    pred = x @ coef
    resid = aec_df[AEC_COLS].to_numpy(dtype=np.float64) - pred
    out = aec_df.copy()
    out[AEC_COLS] = resid
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal: 5-fold OOF for an unbiased internal estimate ---
    screen = stage2.fit_internal_screen()
    stage1_rows_all_int, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = stage2.build_stage2_inputs(screen)
    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    score_standardizer = s2model.fit_score_standardizer(stage1_rows_int["score"].to_numpy(dtype=np.float32))
    x_clin_int, _ = s2model._to_tensors(stage2_input_clin_int, stage2_input_aec_int, stage1_rows_int,
                                         score_standardizer=score_standardizer)

    stage1_rows_all_ext, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = stage2.build_stage2_inputs_external(screen)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)
    x_clin_ext, _ = s2model._to_tensors(stage2_input_clin_ext, stage2_input_aec_ext, stage1_rows_ext,
                                         score_standardizer=score_standardizer)

    # --- residualize AEC-128 against clinical vars: fit on internal Stage-2 cohort only,
    # freeze coefficients, apply to both internal and external ---
    coef = fit_aec_residualizer(stage2_input_clin_int, stage2_input_aec_int)
    resid_aec_int = residualize(stage2_input_clin_int, stage2_input_aec_int, coef)
    resid_aec_ext = residualize(stage2_input_clin_ext, stage2_input_aec_ext, coef)
    var_explained = 1.0 - (resid_aec_int[AEC_COLS].to_numpy().var() / stage2_input_aec_int[AEC_COLS].to_numpy().var())
    print(f"Clinical vars explain {var_explained:.1%} of raw AEC-128 curve variance (internal Stage-2 cohort).")

    x_aec_int = torch.tensor(resid_aec_int[AEC_COLS].to_numpy(dtype=np.float32))
    x_aec_ext = torch.tensor(resid_aec_ext[AEC_COLS].to_numpy(dtype=np.float32))

    oof, fold_loss_histories = s2model.oof_scores(x_clin_int, x_aec_int, y_int)

    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    th = s2model.choose_stage2_threshold(y_all_int, pos_mask_int, oof, stage1_only_int["sens"], stage1_only_int["spec"])

    # --- freeze: refit on the full internal Stage-2 cohort, transfer to external ---
    model, final_loss_history = s2model.fit_final_model(x_clin_int, x_aec_int, y_int)
    s2model.plot_loss_curves(fold_loss_histories, final_loss_history, OUTPUT_DIR / "loss_curve.png")

    score_ext = model.predict_proba(x_clin_ext, x_aec_ext)

    result_int = baseline.evaluate("internal", y_int, oof >= th, th)
    result_ext = baseline.evaluate("external", y_ext, score_ext >= th, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Stage-2 Late Fusion (clinical + RESIDUALIZED AEC-128), screen-positive only", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_int)
    baseline.plot_confusion_matrix(axes[1], result_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    auc_int = baseline.auc_significance_stats(y_int, oof)
    baseline.plot_roc_curve(y_int, oof, auc_int, OUTPUT_DIR / "roc_curve_internal.png",
                             title="Stage-2 Late Fusion (residualized AEC): ROC (internal, OOF)")

    auc_ext = baseline.auc_significance_stats(y_ext, score_ext)
    baseline.plot_roc_curve(y_ext, score_ext, auc_ext, OUTPUT_DIR / "roc_curve_external.png",
                             title="Stage-2 Late Fusion (residualized AEC): ROC (external, frozen internal model)")

    pred_all_int = s2model.combine_predictions(pos_mask_int, oof, th)
    y_all_ext, pos_mask_ext, pred_all_ext = s2model.final_pipeline_labels(stage1_rows_all_ext, score_ext, th)

    stage1_score_int = stage1_rows_all_int["score"].to_numpy()
    stage1_score_ext = stage1_rows_all_ext["score"].to_numpy()
    full_score_int = s2model.combine_full_pipeline_score(stage1_score_int, pos_mask_int, oof, screen["th"])
    full_score_ext = s2model.combine_full_pipeline_score(stage1_score_ext, pos_mask_ext, score_ext, screen["th"])

    auc_stage1_int = baseline.auc_significance_stats(y_all_int, stage1_score_int)
    auc_full_int = baseline.auc_significance_stats(y_all_int, full_score_int)
    auc_stage1_ext = baseline.auc_significance_stats(y_all_ext, stage1_score_ext)
    auc_full_ext = baseline.auc_significance_stats(y_all_ext, full_score_ext)

    delong_int = s2model.delong_paired_auc_test(y_all_int.astype(float), stage1_score_int, full_score_int)
    delong_ext = s2model.delong_paired_auc_test(y_all_ext.astype(float), stage1_score_ext, full_score_ext)
    print(f"[internal] Stage-1 AUC={auc_stage1_int['auc']:.3f}  Full-pipeline AUC={auc_full_int['auc']:.3f}  "
          f"DeLong diff={delong_int['diff']:+.4f} p={delong_int['p_value']:.4f}")
    print(f"[external] Stage-1 AUC={auc_stage1_ext['auc']:.3f}  Full-pipeline AUC={auc_full_ext['auc']:.3f}  "
          f"DeLong diff={delong_ext['diff']:+.4f} p={delong_ext['p_value']:.4f}")

    s2model.plot_stage1_vs_full_pipeline_roc([
        {"label": "internal", "y": y_all_int, "stage1_score": stage1_score_int, "stage1_auc": auc_stage1_int,
         "full_score": full_score_int, "full_auc": auc_full_int, "delong_p": delong_int["p_value"]},
        {"label": "external", "y": y_all_ext, "stage1_score": stage1_score_ext, "stage1_auc": auc_stage1_ext,
         "full_score": full_score_ext, "full_auc": auc_full_ext, "delong_p": delong_ext["p_value"]},
    ], OUTPUT_DIR / "roc_comparison_stage1_vs_full_pipeline.png")

    pd.DataFrame([
        {"cohort": "internal", "stage1_auc": auc_stage1_int["auc"], "full_pipeline_auc": auc_full_int["auc"],
         "auc_diff": delong_int["diff"], "delong_p_value": delong_int["p_value"]},
        {"cohort": "external", "stage1_auc": auc_stage1_ext["auc"], "full_pipeline_auc": auc_full_ext["auc"],
         "auc_diff": delong_ext["diff"], "delong_p_value": delong_ext["p_value"]},
    ]).to_csv(OUTPUT_DIR / "stage1_vs_full_pipeline_auc.csv", index=False)

    stage1_only_ext = baseline.evaluate("external / stage-1 only", y_all_ext, pos_mask_ext, screen["th"])
    result_final_int = baseline.evaluate("internal", y_all_int, pred_all_int, th)
    result_final_ext = baseline.evaluate("external", y_all_ext, pred_all_ext, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Full Pipeline (Stage 1 screen + Stage 2 residualized-AEC late fusion)", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_final_int)
    baseline.plot_confusion_matrix(axes[1], result_final_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    # --- NI test vs. stage-1-only (see module docstring: threshold is chosen on
    # internal only and never touched by external, so external PASS is not guaranteed) ---
    ok_int = s2model.ni_pass_fail(stage1_only_int["sens"], result_final_int["sens"], stage1_only_int["spec"], result_final_int["spec"])
    ok_ext = s2model.ni_pass_fail(stage1_only_ext["sens"], result_final_ext["sens"], stage1_only_ext["spec"], result_final_ext["spec"])
    for cohort, stage1_res, result_final, ok in [
        ("internal", stage1_only_int, result_final_int, ok_int),
        ("external", stage1_only_ext, result_final_ext, ok_ext),
    ]:
        sens_floor = stage1_res["sens"] * (1 - s2model.SENS_LOSS_RATIO_MARGIN)
        print(f"[{cohort}] sens {stage1_res['sens']:.4f}->{result_final['sens']:.4f} (floor={sens_floor:.4f})  "
              f"spec {stage1_res['spec']:.4f}->{result_final['spec']:.4f}  -> {'PASS' if ok else 'FAIL'}")

    pipeline_summary = pd.DataFrame([
        {"cohort": "internal", "sens_before": stage1_only_int["sens"], "sens_after": result_final_int["sens"],
         "sens_floor": stage1_only_int["sens"] * (1 - s2model.SENS_LOSS_RATIO_MARGIN),
         "spec_before": stage1_only_int["spec"], "spec_after": result_final_int["spec"], "pass": ok_int},
        {"cohort": "external", "sens_before": stage1_only_ext["sens"], "sens_after": result_final_ext["sens"],
         "sens_floor": stage1_only_ext["sens"] * (1 - s2model.SENS_LOSS_RATIO_MARGIN),
         "spec_before": stage1_only_ext["spec"], "spec_after": result_final_ext["spec"], "pass": ok_ext},
    ])
    pipeline_summary.to_csv(OUTPUT_DIR / "final_pipeline_summary.csv", index=False)

    table_rows = [
        s2model.build_clinical_vs_aec_row("internal", y_all_int, pos_mask_int, pred_all_int, stage1_only_int, result_final_int, auc_int["auc"]),
        s2model.build_clinical_vs_aec_row("external", y_all_ext, pos_mask_ext, pred_all_ext, stage1_only_ext, result_final_ext, auc_ext["auc"]),
    ]
    s2model.plot_clinical_vs_aec_table(
        table_rows, OUTPUT_DIR / "clinical_vs_aec_assisted_table.png",
        "clinical-only vs. Residualized-AEC-assisted(Late Fusion) 성능 비교 (Stage-1 vs Stage-1+Stage-2)",
    )
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "clinical_vs_aec_assisted_summary.csv", index=False)

    print("\n=== residualized-AEC branch: PRIMARY significance test (Net NRI / McNemar) ===")
    for r in table_rows:
        print(f"[{r['cohort']}] Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
              f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
              f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})")
    print(f"NI test: internal {'PASS' if ok_int else 'FAIL'} / external {'PASS' if ok_ext else 'FAIL'}")
    print("\nCompare: z_clin-only Net NRI = +63 internal / +20 external (see stage2_model_no_aec_ablation.py)")
    print("         raw-AEC (production) Net NRI = +71 internal / +34 external, NI PASS both cohorts")
    print("         residualized-AEC Net NRI = higher, but NI FAILS external -> NOT adopted for production")


if __name__ == "__main__":
    main()
