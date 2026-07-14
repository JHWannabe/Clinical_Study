from __future__ import annotations

# Compares two FIXED PCA component counts (k=4 vs k=6) for the AEC-128 residual
# features, both plugged into the same clinical stage-1 + stage-2 architecture
# as 1_aec_residual_reclassify.py (which instead picks k automatically from a
# cumulative-explained-variance target). Everything except the PCA k is shared:
# same clinical standardizer/stage-1 model/threshold, same residualizer, same
# stage-2 model-selection sweep over {model_type, use_stage1_score}, same
# frozen-external evaluation recipe.
#
# Reuses 1_aec_residual_reclassify.py functions throughout -- no pipeline logic
# is reimplemented (feedback_no_legacy_pipeline_reuse, feedback_unified_baseline_model).
#
# Run: python code/1_2_aec_pca_k_comparison.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")
stage2 = importlib.import_module("1_aec_residual_reclassify")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1_2_aec_pca_k_comparison"

INTERNAL_XLSX = stage2.INTERNAL_XLSX
EXTERNAL_XLSX = stage2.EXTERNAL_XLSX
TARGET_SENSITIVITY = baseline.TARGET_SENSITIVITY

PCA_K_VALUES = [4, 6]

MODEL_COLORS = {
    "Clinical only (stage-1)": stage2.PIPE_INK_MUTED,
    "Clinical + AEC (PCA k=4)": stage2.PIPE_COL_A,
    "Clinical + AEC (PCA k=6)": stage2.PIPE_COL_B,
}

# stage-2 model-selection sweep at fixed PCA k: mirrors stage2.SWEEP_CONFIGS'
# {model_type, use_stage1_score} combos, but pca_var_target is meaningless here
# since k is fixed, so the two var_target duplicates collapse to one each.
FIXED_K_MODEL_CONFIGS = [
    {"model_type": "logreg", "use_stage1_score": False},
    {"model_type": "logreg", "use_stage1_score": True},
    {"model_type": "hgb", "use_stage1_score": True},
]


def run_fixed_k_variant(name: str, k: int, resid_int: np.ndarray, resid_ext: np.ndarray,
                         x_int: np.ndarray, x_ext: np.ndarray, y_int: np.ndarray, y_ext: np.ndarray,
                         oof1: np.ndarray, pos_mask_int: np.ndarray, stage1_model,
                         score1_ext: np.ndarray, pos_mask_ext: np.ndarray,
                         baseline_int: dict, baseline_ext: dict) -> dict:
    pca = stage2.fit_residual_pca(resid_int, fixed_k=k)
    aec_pca_int = pca.transform(resid_int)

    sweep_state = {}
    for cfg in FIXED_K_MODEL_CONFIGS:
        stage1_feat = oof1 if cfg["use_stage1_score"] else None
        x2_int = stage2.stage2_feature_matrix(x_int, aec_pca_int, stage1_feat)
        s2_oof = stage2.stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"])
        th2 = stage2.choose_stage2_threshold(y_int, pos_mask_int, s2_oof)
        pred_int = stage2.combine_predictions(pos_mask_int, s2_oof, th2)
        comb_int = stage2.evaluate_combined(f"{name}/internal sweep {cfg}", y_int, pred_int)
        ni = stage2.noninferiority_test_sensitivity(pos_mask_int, pred_int, y_int)
        sens_delta, spec_delta, ok = stage2.pass_fail(baseline_int, comb_int, ni)
        key = (cfg["model_type"], cfg["use_stage1_score"])
        sweep_state[key] = {"th2": th2, "sens_delta": sens_delta, "spec_delta": spec_delta, "ok": ok, "cfg": cfg}

    passing = [s for s in sweep_state.values() if s["ok"]]
    pool = passing if passing else list(sweep_state.values())
    best = max(pool, key=lambda s: s["spec_delta"])
    cfg, th2 = best["cfg"], best["th2"]

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

    aec_pca_ext = pca.transform(resid_ext)
    stage1_feat_ext = score1_ext if cfg["use_stage1_score"] else None
    x2_ext = stage2.stage2_feature_matrix(x_ext, aec_pca_ext, stage1_feat_ext)
    s2_score_ext = stage2.stage2_score_fn(stage2_model, x2_ext)
    pred_ext = stage2.combine_predictions(pos_mask_ext, s2_score_ext, th2)
    combined_ext = stage2.with_accuracy(stage2.evaluate_combined(f"{name}/external (frozen)", y_ext, pred_ext))
    ni_ext = stage2.noninferiority_test_sensitivity(pos_mask_ext, pred_ext, y_ext)
    sens_d_ext, spec_d_ext, ok_ext = stage2.pass_fail(baseline_ext, combined_ext, ni_ext)

    return {
        "name": name, "pca_k": k, "cfg": cfg,
        "explained_var_int": float(np.sum(pca.explained_variance_ratio_)),
        "internal": combined_int, "external": combined_ext,
        "sens_delta_internal": sens_d_int, "spec_delta_internal": spec_d_int, "pass_internal": ok_int,
        "sens_delta_external": sens_d_ext, "spec_delta_external": spec_d_ext, "pass_external": ok_ext,
    }


def plot_comparison_figure(report: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [("sens", "Sensitivity"), ("spec", "Specificity"), ("ppv", "PPV"), ("acc", "Accuracy")]
    models = report["model"].tolist()
    colors = [MODEL_COLORS[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
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

    fig.legend(loc="lower center", ncol=len(models), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("AEC residual PCA component count comparison: k=4 vs k=6",
                 fontsize=13, fontweight="bold", color=stage2.PIPE_INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- internal: fit clinical standardizer, stage-1, aec residualizer ----------
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

    # ---------- run both fixed-k variants ----------
    results = {}
    for k in PCA_K_VALUES:
        print(f"\n=== PCA k={k} ===")
        res = run_fixed_k_variant(
            f"Clinical + AEC (PCA k={k})", k, resid_int, resid_ext, x_int, x_ext, y_int, y_ext,
            oof1, pos_mask_int, stage1_model, score1_ext, pos_mask_ext, baseline_int, baseline_ext)
        results[k] = res
        print(f"cfg={res['cfg']} explained_var(internal)={res['explained_var_int']:.3f} "
              f"sens_delta_int={res['sens_delta_internal']:+.3f} spec_delta_int={res['spec_delta_internal']:+.3f} "
              f"pass_int={res['pass_internal']} | sens_delta_ext={res['sens_delta_external']:+.3f} "
              f"spec_delta_ext={res['spec_delta_external']:+.3f} pass_ext={res['pass_external']}")

    # ---------- summary table ----------
    rows = [{
        "model": "Clinical only (stage-1)", "pca_k": np.nan, "explained_var_internal": np.nan,
        "sens_internal": baseline_int["sens"], "spec_internal": baseline_int["spec"],
        "acc_internal": baseline_int["acc"], "ppv_internal": baseline_int["ppv"], "npv_internal": baseline_int["npv"],
        "sens_external": baseline_ext["sens"], "spec_external": baseline_ext["spec"],
        "acc_external": baseline_ext["acc"], "ppv_external": baseline_ext["ppv"], "npv_external": baseline_ext["npv"],
        "sens_delta_internal": 0.0, "spec_delta_internal": 0.0, "pass_internal": True,
        "sens_delta_external": 0.0, "spec_delta_external": 0.0, "pass_external": True,
    }]
    for k in PCA_K_VALUES:
        res = results[k]
        ri, re_ = res["internal"], res["external"]
        rows.append({
            "model": res["name"], "pca_k": res["pca_k"], "explained_var_internal": res["explained_var_int"],
            "sens_internal": ri["sens"], "spec_internal": ri["spec"], "acc_internal": ri["acc"],
            "ppv_internal": ri["ppv"], "npv_internal": ri["npv"],
            "sens_external": re_["sens"], "spec_external": re_["spec"], "acc_external": re_["acc"],
            "ppv_external": re_["ppv"], "npv_external": re_["npv"],
            "sens_delta_internal": res["sens_delta_internal"], "spec_delta_internal": res["spec_delta_internal"],
            "pass_internal": res["pass_internal"],
            "sens_delta_external": res["sens_delta_external"], "spec_delta_external": res["spec_delta_external"],
            "pass_external": res["pass_external"],
        })

    report = pd.DataFrame(rows)
    report_path = OUTPUT_DIR / "pca_k_comparison.csv"
    report.to_csv(report_path, index=False)
    print("\n=== Full comparison table ===")
    print(report.to_string(index=False))
    print(f"Saved summary to {report_path}")

    plot_comparison_figure(report, OUTPUT_DIR / "pca_k_comparison.png")


if __name__ == "__main__":
    main()
