from __future__ import annotations

# Cohort-swapped rerun of code/baseline/clinic-only_baseline.py: the original baseline
# fits on internal=gangnam.xlsx and transfers to external=sinchon.xlsx. This script
# swaps which cohort plays which role -- internal=sinchon.xlsx (fit + OOF), external=
# gangnam.xlsx (frozen-model transfer) -- to check whether the clinical-only screen's
# performance depends on that internal/external assignment. All modeling logic is
# reused as-is from clinic-only_baseline.py; only INTERNAL_XLSX/EXTERNAL_XLSX/OUTPUT_DIR
# are swapped.
# Run: python code/0723/clinic_only_baseline_swap.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0723" / "0_clinic-only_baseline_swap"

INTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
EXTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal (sinchon): fit standardizer once on internal only ---
    meta_int, y_int = baseline.load_cohort(INTERNAL_XLSX)
    x_raw_int = baseline.raw_clinical_matrix(meta_int)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw_int)
    x_int = baseline.apply_clinical_standardizer(x_raw_int, med, mu, sd)
    oof = baseline.oof_scores(x_int, y_int)

    auc_stats = baseline.auc_significance_stats(y_int, oof)
    print(f"[internal=sinchon / OOF] AUC={auc_stats['auc']:.4f} "
          f"95%CI=[{auc_stats['ci_lower']:.4f}, {auc_stats['ci_upper']:.4f}] "
          f"Mann-Whitney p={auc_stats['p_value']:.3e}")

    model = baseline.fit_baseline_model(x_int, y_int)
    meta_ext, y_ext = baseline.load_cohort(EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), med, mu, sd)
    score_ext = model.decision_function(x_ext)

    auc_stats_ext = baseline.auc_significance_stats(y_ext, score_ext)
    print(f"[external=gangnam / frozen internal model] AUC={auc_stats_ext['auc']:.4f} "
          f"95%CI=[{auc_stats_ext['ci_lower']:.4f}, {auc_stats_ext['ci_upper']:.4f}] "
          f"Mann-Whitney p={auc_stats_ext['p_value']:.3e}")

    pd.DataFrame([{"cohort": "internal(sinchon)", **auc_stats}, {"cohort": "external(gangnam)", **auc_stats_ext}]) \
        .to_csv(OUTPUT_DIR / "clinical_only_auc_significance.csv", index=False)
    baseline.plot_roc_curve_dual([
        (y_int, oof, auc_stats, "ROC (internal=sinchon, OOF)"),
        (y_ext, score_ext, auc_stats_ext, "ROC (external=gangnam, frozen model)"),
    ], OUTPUT_DIR / "clinical_only_roc_curve.png")

    all_results = []
    summary_rows = []
    th90 = None
    for target in baseline.TARGET_SENSITIVITIES:
        th = baseline.threshold_for_sensitivity(y_int, oof, target)
        if target == 0.90:
            th90 = th
        result_int = baseline.evaluate("internal", y_int, oof >= th, th)
        result_ext = baseline.evaluate("external", y_ext, score_ext >= th, th)
        result_int["target"] = target
        result_ext["target"] = target
        all_results.extend([result_int, result_ext])

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle(f"Clinical-only Logistic Regression, swapped cohorts (S{target * 100:.0f})", fontsize=13, fontweight="bold")
        baseline.plot_confusion_matrix(axes[0], result_int)
        baseline.plot_confusion_matrix(axes[1], result_ext)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = OUTPUT_DIR / f"clinical_only_confusion_matrix_sens{int(target * 100)}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Saved confusion matrix to {out_path}")

        for result in (result_int, result_ext):
            summary_rows.append({
                "target_sensitivity": target,
                "cohort": result["cohort"],
                "threshold": result["th"],
                "sensitivity": result["sens"],
                "specificity": result["spec"],
                "ppv": result["ppv"],
                "npv": result["npv"],
                "n": int(result["matrix"].sum()),
            })

    assert th90 is not None, "0.90 must be in TARGET_SENSITIVITIES for error_feature_analysis"
    baseline.error_feature_analysis(meta_int, y_int, oof, th90, model, OUTPUT_DIR)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "clinical_only_sensitivity_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved comparison table to {summary_path}")

    comparison_fig_path = OUTPUT_DIR / "clinical_only_sensitivity_comparison.png"
    baseline.plot_comparison_summary(all_results, comparison_fig_path)
    print(f"Saved comparison figure to {comparison_fig_path}")


if __name__ == "__main__":
    main()
