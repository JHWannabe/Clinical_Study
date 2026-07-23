from __future__ import annotations

# Ablation: how well do plain clinical variables alone -- sex, height, weight only (no
# age, no vendor, no AEC-128) -- separate Stage-1 TP from FP via a single plain
# LogisticRegression, within the Sinchon-only 5-fold OOF protocol (see
# sinchon_only_pipeline.py)?
#
# Reuses sinchon_only_pipeline.py's Stage 1 (clinical-only LR screen, identical code
# path) to get the screen-positive (TP/FP) subset, then replaces its whole Stage 2
# (AEC-CNN + grid-searched final classifier) with one LogisticRegression(C=1.0) on 3
# raw clinical features (CLIN_COLS minus age_std), scored via the same 5-fold OOF
# protocol as every other classifier in this codebase (baseline.oof_scores).
#
# Run: python code/sinchon_stage2_sex_height_weight_logreg.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2_dataset = import_module("stage2_dataset")
stage2_model = import_module("stage2_model")
sinchon_only_pipeline = import_module("sinchon_only_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sinchon_stage2_sex_height_weight_logreg"

SINCHON_XLSX = DATA_DIR / "sinchon.xlsx"
CLIN_COLS = ["sex_m", "height_std", "weight_std"]  # stage2_dataset.CLIN_COLS minus age_std


def oof_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Plain LogisticRegression, 5-fold OOF -- same protocol/hyperparameters as
    # baseline.oof_scores, just predict_proba instead of decision_function so the
    # score sits on [0,1] like every other Stage-2 candidate in this codebase.
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=baseline.N_FOLDS, shuffle=True, random_state=baseline.SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x, y)):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=baseline.SEED + fold_id)
        model.fit(x[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(x[va_idx])[:, 1]
    return oof


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: clinical-only LR, 5-fold OOF over the whole Sinchon cohort
    # (identical to sinchon_only_pipeline.py's Stage 1) ---
    meta, y = baseline.load_cohort(SINCHON_XLSX)
    print(f"Sinchon cohort: n={len(y)} (event={int(y.sum())})")

    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)
    oof1 = baseline.oof_scores(x, y)
    th = baseline.threshold_for_sensitivity(y, oof1, baseline.TARGET_SENSITIVITY)
    print(f"[Sinchon, S{int(baseline.TARGET_SENSITIVITY * 100)}, 5-fold OOF] threshold={th:.4f}")

    stage1_only = baseline.evaluate("sinchon (5-fold OOF)", y, oof1 >= th, th)

    # --- Stage-2 inputs: screen-positive (TP/FP) rows only, clinic features from
    # Stage-1's OOF score (same helper sinchon_only_pipeline.py uses) ---
    stage1_rows_all, stage1_rows_pos, stage2_clin, _ = \
        stage2_dataset._stage1_positive_rows(SINCHON_XLSX, meta, y, oof1, th, x)

    y2 = (stage1_rows_pos["group"] == "TP").to_numpy().astype(int)
    x_clin = stage2_clin[CLIN_COLS].to_numpy(dtype=np.float64)
    print(f"Stage-2 clinical features: {CLIN_COLS} (n={len(y2)}, TP={int(y2.sum())}, FP={int((1 - y2).sum())})")

    # --- Stage 2: plain logistic regression on sex+height+weight only, 5-fold OOF ---
    oof2 = oof_scores(x_clin, y2)

    y_all = stage1_rows_all["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask = stage1_rows_all["group"].isin(["TP", "FP"]).to_numpy()
    th2 = stage2_model.choose_stage2_threshold(y_all, pos_mask, oof2, stage1_only["sens"], stage1_only["spec"])

    result2 = baseline.evaluate("sinchon (5-fold OOF)", y2, oof2 >= th2, th2)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sinchon_only_pipeline.plot_confusion_matrix_custom(ax, result2, "Stage 2 only: sex+height+weight LR (Sinchon, 5-fold OOF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage2_only.png", dpi=220)
    plt.close(fig)

    auc2 = baseline.auc_significance_stats(y2, oof2)
    baseline.plot_roc_curve(y2, oof2, auc2, OUTPUT_DIR / "roc_curve_stage2.png",
                             title="Stage 2 (sex+height+weight, plain LogisticRegression): ROC (Sinchon, 5-fold OOF)")

    # --- Full pipeline: Stage-1 FN/TN (screen-negative, untouched) + Stage-2's OOF
    # reclassification of Stage-1 TP/FP (screen-positive) ---
    pred_all = stage2_model.combine_predictions(pos_mask, oof2, th2)
    result_final = baseline.evaluate("sinchon (5-fold OOF)", y_all, pred_all, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Sinchon (5-fold OOF): Stage 1 only vs. Full Pipeline (Stage 1 + sex/height/weight LR)",
                 fontsize=13, fontweight="bold")
    sinchon_only_pipeline.plot_confusion_matrix_custom(axes[0], stage1_only, "Stage 1 only")
    sinchon_only_pipeline.plot_confusion_matrix_custom(axes[1], result_final, "Full pipeline")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    # --- Stage-1-only vs. full-pipeline AUROC, whole cohort, on a directly comparable
    # continuous score (see stage2_model.combine_full_pipeline_score) ---
    stage1_score_all = stage1_rows_all["score"].to_numpy()
    full_score = stage2_model.combine_full_pipeline_score(stage1_score_all, pos_mask, oof2, th)
    auc_stage1_all = baseline.auc_significance_stats(y_all, stage1_score_all)
    auc_full = baseline.auc_significance_stats(y_all, full_score)
    delong = stage2_model.delong_paired_auc_test(y_all.astype(float), stage1_score_all, full_score)
    print(f"\n[sinchon] Stage-1 AUC={auc_stage1_all['auc']:.3f} "
          f"[{auc_stage1_all['ci_lower']:.3f}, {auc_stage1_all['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full['auc']:.3f} [{auc_full['ci_lower']:.3f}, {auc_full['ci_upper']:.3f}]  "
          f"DeLong diff={delong['diff']:+.4f} p={delong['p_value']:.4f}")

    stage2_model.plot_stage1_vs_full_pipeline_roc([
        {"label": "sinchon (5-fold OOF)", "y": y_all, "stage1_score": stage1_score_all, "stage1_auc": auc_stage1_all,
         "full_score": full_score, "full_auc": auc_full, "delong_p": delong["p_value"]},
    ], OUTPUT_DIR / "roc_comparison_stage1_vs_full_pipeline.png")

    table_row = stage2_model.build_clinical_vs_aec_row(
        "sinchon (5-fold OOF)", y_all, pos_mask, pred_all, stage1_only, result_final, auc2["auc"])
    pd.DataFrame([table_row]).to_csv(OUTPUT_DIR / "clinical_vs_sex_height_weight_lr_summary.csv", index=False)

    print("\n=== Stage 2 (sex + height + weight, plain LogisticRegression): TP vs FP ===")
    print(f"AUC={auc2['auc']:.3f} [{auc2['ci_lower']:.3f}, {auc2['ci_upper']:.3f}] "
          f"Mann-Whitney p={auc2['p_value']:.3e}  "
          f"sens={result2['sens']:.3f} spec={result2['spec']:.3f} ppv={result2['ppv']:.3f} npv={result2['npv']:.3f}")

    r = table_row
    print("\n=== Full-pipeline effect vs. Stage-1-only (NRI / McNemar) ===")
    print(f"Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
          f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
          f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})  "
          f"acc: {r['acc_clin']:.3f}->{r['acc_aec']:.3f} (p={r['acc_p']:.4f})")

    ok = stage2_model.ni_pass_fail(stage1_only["sens"], result_final["sens"], stage1_only["spec"], result_final["spec"])
    sens_floor = stage1_only["sens"] * (1 - stage2_model.SENS_LOSS_RATIO_MARGIN)
    print(f"NI test vs. stage-1-only: sens {stage1_only['sens']:.3f}->{result_final['sens']:.3f} "
          f"(floor={sens_floor:.3f}), spec {stage1_only['spec']:.3f}->{result_final['spec']:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
