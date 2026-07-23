from __future__ import annotations

# Whole-cohort (Sinchon, no Stage-1/Stage-2 screening funnel) comparison: clinical-only
# baseline vs. clinical+AEC-128 model, both scored 5-fold OOF over the SAME full cohort.
# Unlike sinchon_only_pipeline.py -- where only Stage-1's screen-positive subset ever
# reaches the AEC model -- every one of the 926 patients gets a clinic+AEC prediction
# directly, low-SMI label vs. everyone else, and the two whole-cohort models are compared
# head-to-head: AUC (DeLong, paired since both scores share the same patients), and at a
# matched S90 threshold, sensitivity/specificity/accuracy with NRI/McNemar.
#
# Reuses baseline (clinic-only_baseline.py) for the clinical-only LR baseline + eval/plot
# utilities, stage2_dataset for AEC-128 loading + column names, stage2_model for DeLong /
# McNemar / accuracy / table-plotting utilities (generic over any two OOF scores, not
# specific to a screening funnel), and sinchon_only_pipeline's AecCNN + grid_search_stage2 /
# aec_cnn_oof_scores / final_model_oof_scores for the clinic+AEC model -- those are already
# generic over any (x_aec, y, x_clin) triple, so they work unchanged against the whole
# cohort's own low-SMI label instead of a screen-positive TP/FP subset.
#
# Run: python code/sinchon_clinic_aec_vs_baseline.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2_dataset = import_module("stage2_dataset")
stage2_model = import_module("stage2_model")
sinchon_pipeline = import_module("sinchon_only_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sinchon_clinic_aec_vs_baseline"

SINCHON_XLSX = DATA_DIR / "sinchon.xlsx"
CLIN_COLS = stage2_dataset.CLIN_COLS
AEC_COLS = stage2_dataset.AEC_COLS


def plot_roc_overlay(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, auc_a: dict, auc_b: dict,
                      label_a: str, label_b: str, delong_p: float, out_path: Path, title: str) -> None:
    # Same visual recipe as stage2_model.plot_stage1_vs_full_pipeline_roc, but with
    # explicit curve labels instead of that function's hardcoded "Stage 1 only" /
    # "Full pipeline" legend text -- this script has no screening funnel to name.
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fpr_a, tpr_a, _ = roc_curve(y, score_a)
    fpr_b, tpr_b, _ = roc_curve(y, score_b)
    ax.plot(fpr_a, tpr_a, color="#9a9a9a", linewidth=2, label=f"{label_a} (AUC={auc_a['auc']:.3f})")
    ax.plot(fpr_b, tpr_b, color="#2a78d6", linewidth=2.5, label=f"{label_b} (AUC={auc_b['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    p_str = "p<0.001" if delong_p < 0.001 else f"p={delong_p:.3f}"
    ax.set_title(f"{title}  (DeLong {p_str})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC comparison to {out_path}")


def build_baseline_vs_clinic_aec_row(cohort: str, y: np.ndarray, pred_baseline: np.ndarray, pred_clinic_aec: np.ndarray,
                                      result_baseline: dict, result_clinic_aec: dict, auc_clinic_aec: float) -> dict:
    # Bidirectional NRI (unlike stage2_model.build_clinical_vs_aec_row, which assumes a
    # one-directional screen-positive-only funnel where reclassification only ever
    # removes positives): both models classify every patient independently, so a
    # patient can be gained OR lost in either direction, in either the actual-positive
    # or actual-negative subgroup.
    pos = y.astype(bool)
    tp_gained = int(np.sum(pos & ~pred_baseline & pred_clinic_aec))   # actual positive, baseline missed, clinic+AEC catches
    tp_lost = int(np.sum(pos & pred_baseline & ~pred_clinic_aec))     # actual positive, baseline caught, clinic+AEC misses
    tn_gained = int(np.sum(~pos & pred_baseline & ~pred_clinic_aec))  # actual negative, baseline false-positive, clinic+AEC corrects
    tn_lost = int(np.sum(~pos & ~pred_baseline & pred_clinic_aec))    # actual negative, baseline correct, clinic+AEC false-positives
    return {
        "cohort": cohort, "n": int(len(y)), "event": int(y.sum()), "auc": auc_clinic_aec,
        "sens_clin": result_baseline["sens"], "sens_aec": result_clinic_aec["sens"],
        "sens_p": stage2_model.exact_mcnemar_p(tp_gained, tp_lost),
        "spec_clin": result_baseline["spec"], "spec_aec": result_clinic_aec["spec"],
        "spec_p": stage2_model.exact_mcnemar_p(tn_gained, tn_lost),
        "acc_clin": stage2_model.accuracy(result_baseline), "acc_aec": stage2_model.accuracy(result_clinic_aec),
        "acc_p": stage2_model.exact_mcnemar_p(tp_gained + tn_gained, tp_lost + tn_lost),
        "net_nri": (tp_gained - tp_lost) + (tn_gained - tn_lost),
        "n_deesc": tp_gained + tp_lost + tn_gained + tn_lost,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Whole Sinchon cohort, clinical + AEC-128 for every patient -- no screening funnel ---
    meta, y = baseline.load_cohort(SINCHON_XLSX)
    print(f"Sinchon cohort: n={len(y)} (event={int(y.sum())})")

    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x_clin_std = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)

    aec = stage2_dataset.load_aec_for_patients(SINCHON_XLSX, meta["PatientID"])
    x_aec_t = torch.tensor(aec[AEC_COLS].to_numpy(dtype=np.float32))

    # --- Baseline: clinical-only LR, 5-fold OOF over the whole cohort ---
    oof_baseline = baseline.oof_scores(x_clin_std, y)
    auc_baseline = baseline.auc_significance_stats(y, oof_baseline)
    print(f"[baseline: clinical-only LR] AUC={auc_baseline['auc']:.3f} "
          f"[{auc_baseline['ci_lower']:.3f}, {auc_baseline['ci_upper']:.3f}]")

    # --- Clinic+AEC model: same AEC-CNN-feature + best-final-classifier stack as
    # sinchon_only_pipeline.py's Stage 2, but scored directly against the whole cohort's
    # own low-SMI label y (not a screen-positive TP/FP subset) -- grid_search_stage2 and
    # its building blocks are generic over the label they're given. ---
    best_hp = sinchon_pipeline.grid_search_stage2(x_aec_t, y, x_clin_std, OUTPUT_DIR)
    print(f"\n=== Using grid-search-selected hyperparameters: {best_hp} ===")

    aec_cnn_oof, fold_loss_histories = sinchon_pipeline.aec_cnn_oof_scores(
        x_aec_t, y, embed_dim=best_hp["embed_dim"], dropout=best_hp["dropout"], lr=best_hp["lr"])
    sinchon_pipeline.plot_fold_loss_curves(
        fold_loss_histories, OUTPUT_DIR / "loss_curve_aec_cnn.png",
        title="AEC-CNN (whole cohort): training loss vs. epoch (Sinchon, 5-fold OOF)")

    aec_mean, aec_std = stage2_model.fit_score_standardizer(aec_cnn_oof)
    x_final = np.column_stack([x_clin_std, (aec_cnn_oof - aec_mean) / aec_std])
    oof_clinic_aec = sinchon_pipeline.final_model_oof_scores(best_hp["final_model"], best_hp["final_params"], x_final, y)
    auc_clinic_aec = baseline.auc_significance_stats(y, oof_clinic_aec)
    print(f"[clinic+AEC ({best_hp['final_model']})] AUC={auc_clinic_aec['auc']:.3f} "
          f"[{auc_clinic_aec['ci_lower']:.3f}, {auc_clinic_aec['ci_upper']:.3f}]")

    # --- Head-to-head AUC: DeLong paired test (same patients, correlated scores) ---
    delong = stage2_model.delong_paired_auc_test(y.astype(float), oof_baseline, oof_clinic_aec)
    print(f"DeLong diff (baseline - clinic+AEC) ={delong['diff']:+.4f} p={delong['p_value']:.4f}")

    baseline.plot_roc_curve_dual([
        (y, oof_baseline, auc_baseline, "Clinical-only LR: ROC (Sinchon, whole cohort, 5-fold OOF)"),
        (y, oof_clinic_aec, auc_clinic_aec, f"Clinic+AEC ({best_hp['final_model']}): ROC (Sinchon, whole cohort, 5-fold OOF)"),
    ], OUTPUT_DIR / "roc_curve_baseline_vs_clinic_aec.png")
    plot_roc_overlay(y, oof_baseline, oof_clinic_aec, auc_baseline, auc_clinic_aec,
                      "Clinical-only LR", f"Clinic+AEC ({best_hp['final_model']})", delong["p_value"],
                      OUTPUT_DIR / "roc_comparison_overlay.png",
                      "Sinchon (whole cohort, 5-fold OOF): Clinical-only vs. Clinic+AEC")

    # --- Matched-threshold comparison (S90 on each model's own OOF score) + NRI/McNemar ---
    th_baseline = baseline.threshold_for_sensitivity(y, oof_baseline, baseline.TARGET_SENSITIVITY)
    th_clinic_aec = baseline.threshold_for_sensitivity(y, oof_clinic_aec, baseline.TARGET_SENSITIVITY)
    result_baseline = baseline.evaluate("baseline (clinical-only LR)", y, oof_baseline >= th_baseline, th_baseline)
    result_clinic_aec = baseline.evaluate(f"clinic+AEC ({best_hp['final_model']})", y,
                                           oof_clinic_aec >= th_clinic_aec, th_clinic_aec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle(f"Sinchon (whole cohort, 5-fold OOF, S{int(baseline.TARGET_SENSITIVITY * 100)} each): "
                 "Clinical-only LR vs. Clinic+AEC", fontsize=13, fontweight="bold")
    sinchon_pipeline.plot_confusion_matrix_custom(axes[0], result_baseline, "Clinical-only LR")
    sinchon_pipeline.plot_confusion_matrix_custom(axes[1], result_clinic_aec, f"Clinic+AEC ({best_hp['final_model']})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_baseline_vs_clinic_aec.png", dpi=220)
    plt.close(fig)

    pred_baseline = (oof_baseline >= th_baseline)
    pred_clinic_aec = (oof_clinic_aec >= th_clinic_aec)
    table_row = build_baseline_vs_clinic_aec_row(
        "sinchon (whole cohort, 5-fold OOF)", y, pred_baseline, pred_clinic_aec,
        result_baseline, result_clinic_aec, auc_clinic_aec["auc"])
    stage2_model.plot_clinical_vs_aec_table(
        [table_row], OUTPUT_DIR / "clinical_vs_aec_assisted_table.png",
        f"clinical-only vs. clinic+AEC({best_hp['final_model']}) 성능 비교 (Sinchon, 전체 코호트, 5-fold OOF)")
    pd.DataFrame([table_row]).to_csv(OUTPUT_DIR / "clinical_vs_aec_assisted_summary.csv", index=False)

    pd.DataFrame([{
        "cohort": "sinchon (whole cohort, 5-fold OOF)", "n": len(y), "event": int(y.sum()),
        "auc_baseline": auc_baseline["auc"], "auc_baseline_ci_lower": auc_baseline["ci_lower"],
        "auc_baseline_ci_upper": auc_baseline["ci_upper"],
        "auc_clinic_aec": auc_clinic_aec["auc"], "auc_clinic_aec_ci_lower": auc_clinic_aec["ci_lower"],
        "auc_clinic_aec_ci_upper": auc_clinic_aec["ci_upper"],
        "delong_diff": delong["diff"], "delong_z": delong["z"], "delong_p_value": delong["p_value"],
        "final_model": best_hp["final_model"], "final_params": str(best_hp["final_params"]),
    }]).to_csv(OUTPUT_DIR / "auc_comparison_summary.csv", index=False)

    print("\n=== PRIMARY significance test: NRI / McNemar (reclassification), whole cohort ===")
    r = table_row
    print(f"Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']}, reclassified={r['n_deesc']})  "
          f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
          f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})  "
          f"acc: {r['acc_clin']:.3f}->{r['acc_aec']:.3f} (p={r['acc_p']:.4f})")
    print("=== Secondary: whole-curve AUC / DeLong ===")
    print(f"Baseline AUC={auc_baseline['auc']:.3f}  Clinic+AEC AUC={auc_clinic_aec['auc']:.3f}  "
          f"DeLong p={delong['p_value']:.4f}")


if __name__ == "__main__":
    main()
