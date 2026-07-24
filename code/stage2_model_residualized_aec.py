from __future__ import annotations

# Experiment: classify Stage-2 screen-positive patients (TP vs FP) directly from the
# clinic-covariate-adjusted AEC-128 residual curve, using the two-step method described
# in Li, Chiou & Shyr (2017, Computational Statistics & Data Analysis 115:21-34,
# "Functional Data Classification Using Covariate-Adjusted Subspace Projection"):
#
#   Step 1 (covariate-adjusted mean function, Chiou, Muller & Wang 2003 lineage):
#     fit aec_i ~ CLIN_COLS + intercept per slice (unchanged from the previous version
#     of this script -- see fit_aec_residualizer/residualize below) and take the
#     residual curve = observed - clinic-predicted mean.
#   Step 2 (covariate-adjusted subspace projection, Chiou & Li 2007 lineage):
#     for EACH class (TP, FP) separately, run FPCA (via SVD) on that class's residual
#     curves to get a class-specific mean + a small set of eigenfunctions -- i.e. a
#     low-dimensional linear subspace that best reconstructs curves from that class.
#     A new patient's residual curve is projected onto both subspaces; whichever
#     subspace reconstructs it with less leftover (unexplained) energy wins
#     (nearest-subspace classification by L2 distance, same principle as Eigenfaces).
#
# HONESTY NOTE: the CSDA paper itself is paywalled and its exact formulas were not
# directly verified -- this is a good-faith reconstruction from (a) the paper's own
# abstract, and (b) the two predecessor methods it explicitly builds on (Chiou, Muller
# & Wang 2003 for the covariate-adjusted mean function; Chiou & Li 2007 for the
# FPC-subspace nearest-class classification rule). See docs/aec_residual_related_papers.md
# section 3 for the citation trail. In particular the per-class variance-explained
# truncation threshold (VAR_EXPLAINED_THRESHOLD below) is our own reasonable choice,
# not something taken from the paper.
#
# This REPLACES the previous version of this script, which fed the 128-dim residual
# curve into a 1D-CNN branch (same architecture as stage2_model.py's AecBranch) and
# late-fused it with the clinical branch. That version scored Net NRI +77 internal /
# +52 external -- higher than raw-AEC's +71/+34 -- but its external NI test FAILED by
# 0.32pp (see git history for the exact numbers/plots). Net NRI beating NI-fail
# suggests a variance/overfitting problem (128-dim conv branch, n=564) rather than a
# signal problem, which is the motivation for trying this lower-parameter, curve-shape-
# preserving alternative instead of either the CNN or a flattened AEC-sum scalar.
#
# Design mirrors the rest of the Stage-2 pipeline (fit_internal_screen, 5-fold OOF for
# internal, freeze-and-transfer to external, NI test vs. stage-1-only, Net NRI/McNemar
# as primary significance test) by reusing stage2_model.py's shared helpers
# (choose_stage2_threshold, combine_predictions, ni_pass_fail, build_clinical_vs_aec_row,
# etc.) -- only the score-generation step changes, from a neural net to the FPCA-subspace
# classifier below, so this script's outputs stay artifact-for-artifact comparable with
# stage2_model.py and the CNN version this replaces.
#
# Run: python code/stage2_model_residualized_aec.py

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2 = import_module("stage2_dataset")
s2model = import_module("stage2_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_model_residualized_aec"

CLIN_COLS = stage2.CLIN_COLS
AEC_COLS = stage2.AEC_COLS

# Cumulative variance-explained target for each class's FPCA subspace truncation. Not
# taken from the paper (paywalled) -- 90% is a common practical default for FPCA rank
# selection; kept as a single named constant so it's easy to sweep/ablate.
VAR_EXPLAINED_THRESHOLD = 0.90


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


@dataclass
class ClassSubspace:
    # A class's covariate-adjusted mean curve + a truncated set of FPCA eigenfunctions
    # (rows of `components`, orthonormal) spanning the low-dimensional subspace that
    # best reconstructs that class's residual curves.
    mean: np.ndarray                    # (n_slices,)
    components: np.ndarray              # (k, n_slices)
    explained_variance_ratio: np.ndarray  # (k,)


def fit_class_subspace(residual_curves: np.ndarray, var_threshold: float = VAR_EXPLAINED_THRESHOLD,
                        min_components: int = 1) -> ClassSubspace:
    # FPCA via SVD of the class-mean-centered residual curves (Chiou & Li 2007-style):
    # right singular vectors are the eigenfunctions of the empirical covariance operator.
    # Keeps the fewest leading components whose cumulative variance-explained clears
    # var_threshold (at least min_components, at most n_samples-comparable rank).
    mean = residual_curves.mean(axis=0)
    centered = residual_curves - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = s ** 2
    total = var.sum()
    ratio = var / total if total > 0 else np.zeros_like(var)
    cum = np.cumsum(ratio)
    k = int(np.searchsorted(cum, var_threshold) + 1)
    k = max(min_components, min(k, vt.shape[0]))
    return ClassSubspace(mean=mean, components=vt[:k], explained_variance_ratio=ratio[:k])


def subspace_residual_energy(x: np.ndarray, subspace: ClassSubspace) -> np.ndarray:
    # x: (n, n_slices). Squared L2 norm of the part of (x - class mean) NOT reconstructed
    # by the class subspace -- Li et al.'s classification distance. Smaller means x looks
    # more like a curve drawn from this class.
    centered = x - subspace.mean
    proj_coef = centered @ subspace.components.T   # (n, k)
    reconstructed = proj_coef @ subspace.components  # (n, n_slices)
    leftover = centered - reconstructed
    return np.sum(leftover ** 2, axis=1)


def fpca_subspace_score(x: np.ndarray, sub_tp: ClassSubspace, sub_fp: ClassSubspace) -> np.ndarray:
    # score > 0 <=> dist_to_FP_subspace > dist_to_TP_subspace <=> curve looks more TP-like.
    # Continuous (not just argmin class) so it plugs into the same threshold-sweep/NI
    # machinery (choose_stage2_threshold) as every other Stage-2 branch's score.
    dist_tp = subspace_residual_energy(x, sub_tp)
    dist_fp = subspace_residual_energy(x, sub_fp)
    return dist_fp - dist_tp


def oof_fpca_scores(x_aec: np.ndarray, y: np.ndarray, var_threshold: float = VAR_EXPLAINED_THRESHOLD) -> np.ndarray:
    # 5-fold OOF (same N_FOLDS/SEED as the rest of the pipeline): class subspaces are
    # refit on the training fold only, so a validation row never informs the very
    # subspace it's scored against.
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=baseline.N_FOLDS, shuffle=True, random_state=baseline.SEED)
    for tr_idx, va_idx in skf.split(x_aec, y):
        sub_tp = fit_class_subspace(x_aec[tr_idx][y[tr_idx] == 1], var_threshold)
        sub_fp = fit_class_subspace(x_aec[tr_idx][y[tr_idx] == 0], var_threshold)
        oof[va_idx] = fpca_subspace_score(x_aec[va_idx], sub_tp, sub_fp)
    return oof


def fit_final_fpca(x_aec: np.ndarray, y: np.ndarray,
                    var_threshold: float = VAR_EXPLAINED_THRESHOLD) -> tuple[ClassSubspace, ClassSubspace]:
    # Refit on the FULL internal Stage-2 cohort, frozen and reused for external --
    # mirrors fit_final_model's role in the CNN version this replaces.
    sub_tp = fit_class_subspace(x_aec[y == 1], var_threshold)
    sub_fp = fit_class_subspace(x_aec[y == 0], var_threshold)
    return sub_tp, sub_fp


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    screen = stage2.fit_internal_screen()
    stage1_rows_all_int, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = stage2.build_stage2_inputs(screen)
    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)

    stage1_rows_all_ext, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = stage2.build_stage2_inputs_external(screen)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)

    # --- Step 1: residualize AEC-128 against clinical vars -- fit on internal Stage-2
    # cohort only, freeze coefficients, apply to both internal and external ---
    coef = fit_aec_residualizer(stage2_input_clin_int, stage2_input_aec_int)
    resid_aec_int_df = residualize(stage2_input_clin_int, stage2_input_aec_int, coef)
    resid_aec_ext_df = residualize(stage2_input_clin_ext, stage2_input_aec_ext, coef)
    var_explained = 1.0 - (resid_aec_int_df[AEC_COLS].to_numpy().var() / stage2_input_aec_int[AEC_COLS].to_numpy().var())
    print(f"Clinical vars explain {var_explained:.1%} of raw AEC-128 curve variance (internal Stage-2 cohort).")

    x_aec_int = resid_aec_int_df[AEC_COLS].to_numpy(dtype=np.float64)
    x_aec_ext = resid_aec_ext_df[AEC_COLS].to_numpy(dtype=np.float64)

    # --- Step 2: covariate-adjusted subspace projection (Li, Chiou & Shyr 2017-style) ---
    # 5-fold OOF for an unbiased internal estimate.
    oof = oof_fpca_scores(x_aec_int, y_int)

    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    th = s2model.choose_stage2_threshold(y_all_int, pos_mask_int, oof, stage1_only_int["sens"], stage1_only_int["spec"])

    # --- freeze: refit class subspaces on the full internal Stage-2 cohort, transfer to external ---
    sub_tp, sub_fp = fit_final_fpca(x_aec_int, y_int)
    print(f"Final TP subspace: {sub_tp.components.shape[0]} eigenfunctions "
          f"(cum. var explained={sub_tp.explained_variance_ratio.sum():.1%}, n_TP={int(y_int.sum())})")
    print(f"Final FP subspace: {sub_fp.components.shape[0]} eigenfunctions "
          f"(cum. var explained={sub_fp.explained_variance_ratio.sum():.1%}, n_FP={int((y_int == 0).sum())})")

    score_ext = fpca_subspace_score(x_aec_ext, sub_tp, sub_fp)

    result_int = baseline.evaluate("internal", y_int, oof >= th, th)
    result_ext = baseline.evaluate("external", y_ext, score_ext >= th, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Stage-2 FPCA-Subspace Classifier (Li, Chiou & Shyr 2017-style, residualized AEC-128), screen-positive only",
                 fontsize=12, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_int)
    baseline.plot_confusion_matrix(axes[1], result_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    auc_int = baseline.auc_significance_stats(y_int, oof)
    baseline.plot_roc_curve(y_int, oof, auc_int, OUTPUT_DIR / "roc_curve_internal.png",
                             title="Stage-2 FPCA-Subspace Classifier: ROC (internal, OOF)")

    auc_ext = baseline.auc_significance_stats(y_ext, score_ext)
    baseline.plot_roc_curve(y_ext, score_ext, auc_ext, OUTPUT_DIR / "roc_curve_external.png",
                             title="Stage-2 FPCA-Subspace Classifier: ROC (external, frozen internal model)")

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
    fig.suptitle("Full Pipeline (Stage 1 screen + Stage 2 FPCA-subspace classifier)", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_final_int)
    baseline.plot_confusion_matrix(axes[1], result_final_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    # --- NI test vs. stage-1-only (threshold chosen on internal only, never touched by
    # external, so external PASS is not guaranteed) ---
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
        "clinical-only vs. FPCA-Subspace-AEC-assisted 성능 비교 (Stage-1 vs Stage-1+Stage-2)",
    )
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "clinical_vs_aec_assisted_summary.csv", index=False)

    print("\n=== FPCA-subspace branch (Li, Chiou & Shyr 2017-style): PRIMARY significance test (Net NRI / McNemar) ===")
    for r in table_rows:
        print(f"[{r['cohort']}] Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
              f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
              f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})")
    print(f"NI test: internal {'PASS' if ok_int else 'FAIL'} / external {'PASS' if ok_ext else 'FAIL'}")
    print("\nCompare: z_clin-only Net NRI = +63 internal / +20 external (see stage2_model_no_aec_ablation.py)")
    print("         raw-AEC CNN (production, stage2_model.py) Net NRI = +71 internal / +34 external, NI PASS both cohorts")
    print("         residualized-AEC CNN (previous version of this script) Net NRI = +77 internal / +52 external, NI FAILS external by 0.32pp")


if __name__ == "__main__":
    main()
