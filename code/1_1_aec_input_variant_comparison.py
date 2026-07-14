from __future__ import annotations

# Two comparisons requested on top of 1_aec_residual_reclassify.py:
#
#   Axis 1: AEC-128 input representation, both combined with the clinical
#           stage-1+stage-2 architecture --
#             - "patient-normalized": AEC curves divided by patient mean only
#               (no residualization against clinical vars).
#             - "BMI-residualized": current pipeline's residual-vs-clinical
#               curves (see project_aec_curve_bmi_confound memory).
#           Same stage-1 gate, same stage-2 sweep/selection logic for both, so
#           the only thing that differs is the AEC feature source.
#
#   Axis 2: AEC-128 alone (no clinical features at all, single-stage LR,
#           same OOF/threshold-at-target-sensitivity recipe as
#           clinic-only_baseline.py) vs the clinical-only baseline and vs the
#           two clinical+AEC combined models above.
#
# Reuses baseline.py and 1_aec_residual_reclassify.py functions throughout --
# no pipeline logic is reimplemented (feedback_no_legacy_pipeline_reuse,
# feedback_unified_baseline_model).
#
# Run: python code/1_1_aec_input_variant_comparison.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")
stage2 = importlib.import_module("1_aec_residual_reclassify")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1_1_aec_input_variant_comparison"

INTERNAL_XLSX = stage2.INTERNAL_XLSX
EXTERNAL_XLSX = stage2.EXTERNAL_XLSX
TARGET_SENSITIVITY = baseline.TARGET_SENSITIVITY

# Fixed categorical colors, one per model variant (reused from the project's
# existing pipeline palette in 1_aec_residual_reclassify.py for consistency).
MODEL_COLORS = {
    "Clinical only (stage-1)": stage2.PIPE_INK_MUTED,
    "AEC only - patient-normalized": stage2.PIPE_COL_B,
    "AEC only - BMI-residualized": "#c0447a",
    "Clinical + AEC - patient-normalized": stage2.PIPE_COL_C,
    "Clinical + AEC - BMI-residualized": stage2.PIPE_COL_A,
}


def run_combined_variant(name: str, aec_int: np.ndarray, aec_ext: np.ndarray,
                          x_int: np.ndarray, x_ext: np.ndarray, y_int: np.ndarray, y_ext: np.ndarray,
                          oof1: np.ndarray, pos_mask_int: np.ndarray, stage1_model,
                          score1_ext: np.ndarray, pos_mask_ext: np.ndarray,
                          baseline_int: dict, baseline_ext: dict) -> dict:
    # Mirrors 1_aec_residual_reclassify.main()'s sweep -> select -> freeze -> external
    # flow exactly, but with `aec_int`/`aec_ext` swapped in as the 128-dim AEC
    # input to PCA (patient-normalized curves vs BMI-residualized curves).
    sweep_state = {}
    for cfg in stage2.SWEEP_CONFIGS:
        pca = stage2.fit_residual_pca(aec_int, var_target=cfg["pca_var_target"])
        aec_pca_int = pca.transform(aec_int)
        stage1_feat = oof1 if cfg["use_stage1_score"] else None
        x2_int = stage2.stage2_feature_matrix(x_int, aec_pca_int, stage1_feat)

        s2_oof = stage2.stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"])
        th2 = stage2.choose_stage2_threshold(y_int, pos_mask_int, s2_oof)
        pred_int = stage2.combine_predictions(pos_mask_int, s2_oof, th2)
        comb_int = stage2.evaluate_combined(f"{name}/internal sweep {cfg}", y_int, pred_int)
        ni = stage2.noninferiority_test_sensitivity(pos_mask_int, pred_int, y_int)
        sens_delta, spec_delta, ok = stage2.pass_fail(baseline_int, comb_int, ni)
        key = (cfg["model_type"], cfg["pca_var_target"], cfg["use_stage1_score"])
        sweep_state[key] = {"pca": pca, "th2": th2, "sens_delta": sens_delta,
                             "spec_delta": spec_delta, "ok": ok, "cfg": cfg}

    passing = [s for s in sweep_state.values() if s["ok"]]
    pool = passing if passing else list(sweep_state.values())
    best = max(pool, key=lambda s: s["spec_delta"])
    cfg, pca, th2 = best["cfg"], best["pca"], best["th2"]

    aec_pca_int = pca.transform(aec_int)
    stage1_feat_int = oof1 if cfg["use_stage1_score"] else None
    x2_int = stage2.stage2_feature_matrix(x_int, aec_pca_int, stage1_feat_int)
    s2_oof = stage2.stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"])
    pred_int = stage2.combine_predictions(pos_mask_int, s2_oof, th2)
    combined_int = stage2.with_accuracy(stage2.evaluate_combined(f"{name}/internal (OOF, selected)", y_int, pred_int))
    ni_int = stage2.noninferiority_test_sensitivity(pos_mask_int, pred_int, y_int)
    sens_d_int, spec_d_int, ok_int = stage2.pass_fail(baseline_int, combined_int, ni_int)

    score1_int_frozen = stage1_model.decision_function(x_int)
    stage1_feat_frozen = score1_int_frozen if cfg["use_stage1_score"] else None
    x2_int_frozen = stage2.stage2_feature_matrix(x_int, aec_pca_int, stage1_feat_frozen)
    stage2_model = stage2.make_stage2_model(cfg["model_type"], stage2.SEED)
    stage2_model.fit(x2_int_frozen[pos_mask_int], y_int[pos_mask_int])

    aec_pca_ext = pca.transform(aec_ext)
    stage1_feat_ext = score1_ext if cfg["use_stage1_score"] else None
    x2_ext = stage2.stage2_feature_matrix(x_ext, aec_pca_ext, stage1_feat_ext)
    s2_score_ext = stage2.stage2_score_fn(stage2_model, x2_ext)
    pred_ext = stage2.combine_predictions(pos_mask_ext, s2_score_ext, th2)
    combined_ext = stage2.with_accuracy(stage2.evaluate_combined(f"{name}/external (frozen)", y_ext, pred_ext))
    ni_ext = stage2.noninferiority_test_sensitivity(pos_mask_ext, pred_ext, y_ext)
    sens_d_ext, spec_d_ext, ok_ext = stage2.pass_fail(baseline_ext, combined_ext, ni_ext)

    return {
        "name": name, "cfg": cfg, "pca_k": pca.n_components_,
        "internal": combined_int, "external": combined_ext,
        "sens_delta_internal": sens_d_int, "spec_delta_internal": spec_d_int, "pass_internal": ok_int,
        "sens_delta_external": sens_d_ext, "spec_delta_external": spec_d_ext, "pass_external": ok_ext,
    }


def run_aec_only_variant(name: str, aec_int: np.ndarray, aec_ext: np.ndarray,
                          y_int: np.ndarray, y_ext: np.ndarray) -> dict:
    # Same OOF / threshold-at-target-sensitivity / frozen-external recipe as
    # clinic-only_baseline.py's main(), but the feature matrix is AEC PCA scores
    # only -- no clinical variables, no stage-1 gate.
    pca = stage2.fit_residual_pca(aec_int, var_target=0.90)
    feat_int = pca.transform(aec_int)
    feat_ext = pca.transform(aec_ext)

    oof = baseline.oof_scores(feat_int, y_int)
    th = baseline.threshold_for_sensitivity(y_int, oof, TARGET_SENSITIVITY)
    result_int = stage2.with_accuracy(baseline.evaluate(f"{name}/internal", y_int, oof >= th, th))

    model = baseline.fit_baseline_model(feat_int, y_int)
    score_ext = model.decision_function(feat_ext)
    result_ext = stage2.with_accuracy(baseline.evaluate(f"{name}/external", y_ext, score_ext >= th, th))

    return {
        "name": name, "pca_k": pca.n_components_,
        "internal": result_int, "external": result_ext,
        "auc_internal": roc_auc_score(y_int, oof), "auc_external": roc_auc_score(y_ext, score_ext),
    }


def plot_comparison_figure(report: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [("sens", "Sensitivity"), ("spec", "Specificity"), ("ppv", "PPV"), ("acc", "Accuracy")]
    models = report["model"].tolist()
    colors = [MODEL_COLORS[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, cohort, title in zip(axes, ["internal", "external"], ["Internal (OOF)", "External (frozen)"]):
        n_metrics = len(metrics)
        n_models = len(models)
        group_w = 0.8
        bar_w = group_w / n_models
        x = np.arange(n_metrics)
        for mi, (model, color) in enumerate(zip(models, colors)):
            vals = [report.loc[report["model"] == model, f"{key}_{cohort}"].iloc[0] for key, _ in metrics]
            offset = (mi - (n_models - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w * 0.92, color=color,
                           label=model if ax is axes[0] else None, zorder=3)
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, color=stage2.PIPE_INK_SECONDARY, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=12, fontweight="bold", color=stage2.PIPE_INK_PRIMARY)
        stage2._pipeline_style_axes(ax)
        ax.yaxis.grid(True, color=stage2.PIPE_GRID, linewidth=0.8, zorder=0)

    fig.legend(loc="lower center", ncol=len(models), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("AEC-128 input variant comparison: clinical-only vs AEC-only vs combined "
                 "(patient-normalized vs BMI-residualized)",
                 fontsize=13, fontweight="bold", color=stage2.PIPE_INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- internal: clinical baseline + both AEC representations ----------
    meta_int, y_int, curves_int = stage2.load_cohort_with_aec(INTERNAL_XLSX)
    x_raw_int = baseline.raw_clinical_matrix(meta_int)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw_int)
    x_int = baseline.apply_clinical_standardizer(x_raw_int, med, mu, sd)

    oof1 = baseline.oof_scores(x_int, y_int)
    th1 = baseline.threshold_for_sensitivity(y_int, oof1, TARGET_SENSITIVITY)
    baseline_int = stage2.with_accuracy(baseline.evaluate("internal/clinical-only", y_int, oof1 >= th1, th1))
    pos_mask_int = oof1 >= th1

    reg = stage2.fit_aec_residualizer(x_int, curves_int)
    resid_int = stage2.apply_aec_residualizer(reg, x_int, curves_int)

    stage1_model = baseline.fit_baseline_model(x_int, y_int)

    # ---------- external: pure held-out test, frozen internal-fit parameters ----------
    meta_ext, y_ext, curves_ext = stage2.load_cohort_with_aec(EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), med, mu, sd)
    score1_ext = stage1_model.decision_function(x_ext)
    baseline_ext = stage2.with_accuracy(baseline.evaluate("external/clinical-only (frozen)", y_ext, score1_ext >= th1, th1))
    pos_mask_ext = score1_ext >= th1

    resid_ext = stage2.apply_aec_residualizer(reg, x_ext, curves_ext)

    # ---------- Axis 1: patient-normalized vs BMI-residualized AEC, combined w/ clinical ----------
    print("\n=== Axis 1: clinical + AEC, patient-normalized vs BMI-residualized ===")
    combined_norm = run_combined_variant(
        "Clinical + AEC - patient-normalized", curves_int, curves_ext, x_int, x_ext, y_int, y_ext,
        oof1, pos_mask_int, stage1_model, score1_ext, pos_mask_ext, baseline_int, baseline_ext)
    combined_resid = run_combined_variant(
        "Clinical + AEC - BMI-residualized", resid_int, resid_ext, x_int, x_ext, y_int, y_ext,
        oof1, pos_mask_int, stage1_model, score1_ext, pos_mask_ext, baseline_int, baseline_ext)

    print(f"patient-norm: cfg={combined_norm['cfg']} pca_k={combined_norm['pca_k']} "
          f"sens_delta_int={combined_norm['sens_delta_internal']:+.3f} spec_delta_int={combined_norm['spec_delta_internal']:+.3f} "
          f"pass_int={combined_norm['pass_internal']} | sens_delta_ext={combined_norm['sens_delta_external']:+.3f} "
          f"spec_delta_ext={combined_norm['spec_delta_external']:+.3f} pass_ext={combined_norm['pass_external']}")
    print(f"residual:     cfg={combined_resid['cfg']} pca_k={combined_resid['pca_k']} "
          f"sens_delta_int={combined_resid['sens_delta_internal']:+.3f} spec_delta_int={combined_resid['spec_delta_internal']:+.3f} "
          f"pass_int={combined_resid['pass_internal']} | sens_delta_ext={combined_resid['sens_delta_external']:+.3f} "
          f"spec_delta_ext={combined_resid['spec_delta_external']:+.3f} pass_ext={combined_resid['pass_external']}")

    # ---------- Axis 2: AEC alone (no clinical vars), both representations ----------
    print("\n=== Axis 2: AEC-only (no clinical data) ===")
    aec_only_norm = run_aec_only_variant("AEC only - patient-normalized", curves_int, curves_ext, y_int, y_ext)
    aec_only_resid = run_aec_only_variant("AEC only - BMI-residualized", resid_int, resid_ext, y_int, y_ext)

    auc_clin_int = roc_auc_score(y_int, oof1)
    auc_clin_ext = roc_auc_score(y_ext, score1_ext)

    # ---------- summary table ----------
    rows = []

    def add_row(name: str, res_int: dict, res_ext: dict, auc_int: float, auc_ext: float) -> None:
        rows.append({
            "model": name,
            "sens_internal": res_int["sens"], "spec_internal": res_int["spec"], "acc_internal": res_int["acc"],
            "ppv_internal": res_int["ppv"], "npv_internal": res_int["npv"], "auc_internal": auc_int,
            "sens_external": res_ext["sens"], "spec_external": res_ext["spec"], "acc_external": res_ext["acc"],
            "ppv_external": res_ext["ppv"], "npv_external": res_ext["npv"], "auc_external": auc_ext,
        })

    add_row("Clinical only (stage-1)", baseline_int, baseline_ext, auc_clin_int, auc_clin_ext)
    add_row("AEC only - patient-normalized", aec_only_norm["internal"], aec_only_norm["external"],
             aec_only_norm["auc_internal"], aec_only_norm["auc_external"])
    add_row("AEC only - BMI-residualized", aec_only_resid["internal"], aec_only_resid["external"],
             aec_only_resid["auc_internal"], aec_only_resid["auc_external"])
    add_row("Clinical + AEC - patient-normalized", combined_norm["internal"], combined_norm["external"],
             float("nan"), float("nan"))
    add_row("Clinical + AEC - BMI-residualized", combined_resid["internal"], combined_resid["external"],
             float("nan"), float("nan"))

    report = pd.DataFrame(rows)
    report_path = OUTPUT_DIR / "aec_input_variant_comparison.csv"
    report.to_csv(report_path, index=False)
    print("\n=== Full comparison table ===")
    print(report.to_string(index=False))
    print(f"Saved summary to {report_path}")

    plot_comparison_figure(report, OUTPUT_DIR / "aec_input_variant_comparison.png")


if __name__ == "__main__":
    main()
