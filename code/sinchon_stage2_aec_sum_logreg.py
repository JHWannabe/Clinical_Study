from __future__ import annotations

# Ablation: replace the AEC-CNN feature extractor with a trivial one -- collapse the
# whole AEC-128 curve into a single scalar (raw, unnormalized sum across all 128
# slices) -- and see how well Stage-2's input (the frozen Stage-1 clinical-only LR
# score, standardized, PLUS this one AEC-sum scalar) separates Stage-1 TP from FP via
# a single plain LogisticRegression, within the Sinchon-only 5-fold OOF protocol.
#
# Stage-1 score (not the 4 raw clinical features) is the clinical-side input here --
# same convention as stage2_model.py's default CLIN_BRANCH_VARIANT="frozen_lr"
# (IdentityBranch: x_clin IS the frozen Stage-1 LR score) rather than
# sinchon_only_pipeline.py's re-exposed raw clinical features + vendor.
#
# Uses the RAW aec_128 sheet values, not stage2_dataset.load_aec_for_patients's
# patient-normalized (mean~1) curve -- that normalization rescales every patient's
# curve to sum to ~128 regardless of true magnitude, which would make "sum across
# slices" carry ~zero signal (curve totals a bit above/below 128 seat noise instead
# of the between-patient average-attenuation signal this ablation is meant to test).
#
# Reuses sinchon_only_pipeline.py's Stage 1 (clinical-only LR screen, identical code
# path) to get the screen-positive (TP/FP) subset, then replaces its whole Stage 2
# (AEC-CNN + grid-searched final classifier) with one LogisticRegression(C=1.0) on
# [standardized Stage-1 score, standardized AEC-sum], scored via the same 5-fold OOF
# protocol as every other classifier in this codebase (baseline.oof_scores).
#
# Run: python code/sinchon_stage2_aec_sum_logreg.py

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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sinchon_stage2_aec_sum_logreg"

SINCHON_XLSX = DATA_DIR / "sinchon.xlsx"
AEC_COLS = stage2_dataset.AEC_COLS


def load_aec_sum_for_patients(xlsx_path: Path, patient_ids: pd.Series) -> np.ndarray:
    # Raw (unnormalized) per-patient sum across all 128 AEC slices, row-aligned to
    # patient_ids -- deliberately bypasses stage2_dataset.load_aec_for_patients's
    # per-patient mean-1 normalization (see module docstring for why).
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    aec_sum_df = pd.DataFrame({
        "PatientID": aec["PatientID"].to_numpy(),
        "aec_sum": aec[AEC_COLS].astype(float).sum(axis=1).to_numpy(),
    })
    order = pd.DataFrame({"PatientID": patient_ids.to_numpy(), "__row__": np.arange(len(patient_ids))})
    merged = order.merge(aec_sum_df, on="PatientID", how="left").sort_values("__row__").drop(columns="__row__")

    missing = int(merged["aec_sum"].isna().sum())
    if missing:
        raise ValueError(f"{missing} patients in {xlsx_path.name} have no matching aec_128 row")
    return merged["aec_sum"].to_numpy()


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

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sinchon_only_pipeline.plot_confusion_matrix_custom(ax, stage1_only, "Stage 1 only (Sinchon, 5-fold OOF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage1_only.png", dpi=220)
    plt.close(fig)

    auc1 = baseline.auc_significance_stats(y, oof1)
    baseline.plot_roc_curve(y, oof1, auc1, OUTPUT_DIR / "roc_curve_stage1.png",
                             title="Stage 1 (Clinical-only LR): ROC (Sinchon, 5-fold OOF)")

    # --- Stage-2 inputs: screen-positive (TP/FP) rows only. Clinical side = the frozen
    # Stage-1 score itself ("score" column of stage1_rows_pos, i.e. oof1 restricted to
    # the screen-positive subset), standardized -- not the raw clinical features. ---
    stage1_rows_all, stage1_rows_pos, stage2_clin, _ = \
        stage2_dataset._stage1_positive_rows(SINCHON_XLSX, meta, y, oof1, th, x)

    y2 = (stage1_rows_pos["group"] == "TP").to_numpy().astype(int)

    stage1_score = stage1_rows_pos["score"].to_numpy(dtype=np.float64)
    stage1_score_mean, stage1_score_std_dev = stage2_model.fit_score_standardizer(stage1_score)
    stage1_score_std = (stage1_score - stage1_score_mean) / stage1_score_std_dev

    aec_sum = load_aec_sum_for_patients(SINCHON_XLSX, stage2_clin["PatientID"])
    aec_sum_mean, aec_sum_std_dev = stage2_model.fit_score_standardizer(aec_sum)
    aec_sum_std = (aec_sum - aec_sum_mean) / aec_sum_std_dev

    x_final = np.column_stack([stage1_score_std, aec_sum_std])
    print(f"Stage-2 features: ['stage1_score', 'aec_sum'] (n={len(y2)}, TP={int(y2.sum())}, FP={int((1 - y2).sum())})")

    # --- Stage 2: plain logistic regression on stage1-score+AEC-sum, 5-fold OOF ---
    oof2 = oof_scores(x_final, y2)

    y_all = stage1_rows_all["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask = stage1_rows_all["group"].isin(["TP", "FP"]).to_numpy()
    th2 = stage2_model.choose_stage2_threshold(y_all, pos_mask, oof2, stage1_only["sens"], stage1_only["spec"])

    result2 = baseline.evaluate("sinchon (5-fold OOF)", y2, oof2 >= th2, th2)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sinchon_only_pipeline.plot_confusion_matrix_custom(
        ax, result2, "Stage 2 only: stage1-score+AEC-sum LR (Sinchon, 5-fold OOF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage2_only.png", dpi=220)
    plt.close(fig)

    auc2 = baseline.auc_significance_stats(y2, oof2)
    baseline.plot_roc_curve(y2, oof2, auc2, OUTPUT_DIR / "roc_curve_stage2.png",
                             title="Stage 2 (stage1-score+AEC-sum, plain LogisticRegression): ROC (Sinchon, 5-fold OOF)")

    # --- Full pipeline: Stage-1 FN/TN (screen-negative, untouched) + Stage-2's OOF
    # reclassification of Stage-1 TP/FP (screen-positive) ---
    pred_all = stage2_model.combine_predictions(pos_mask, oof2, th2)
    result_final = baseline.evaluate("sinchon (5-fold OOF)", y_all, pred_all, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Sinchon (5-fold OOF): Stage 1 only vs. Full Pipeline (Stage 1 + stage1-score/AEC-sum LR)",
                 fontsize=13, fontweight="bold")
    sinchon_only_pipeline.plot_confusion_matrix_custom(axes[0], stage1_only, "Stage 1 only")
    sinchon_only_pipeline.plot_confusion_matrix_custom(axes[1], result_final, "Full pipeline")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    stage1_score_all = stage1_rows_all["score"].to_numpy()
    full_score = stage2_model.combine_full_pipeline_score(stage1_score_all, pos_mask, oof2, th)
    auc_stage1_all = baseline.auc_significance_stats(y_all, stage1_score_all)
    auc_full = baseline.auc_significance_stats(y_all, full_score)
    delong = stage2_model.delong_paired_auc_test(y_all.astype(float), stage1_score_all, full_score)

    stage2_model.plot_stage1_vs_full_pipeline_roc([
        {"label": "sinchon (5-fold OOF)", "y": y_all, "stage1_score": stage1_score_all, "stage1_auc": auc_stage1_all,
         "full_score": full_score, "full_auc": auc_full, "delong_p": delong["p_value"]},
    ], OUTPUT_DIR / "roc_comparison_stage1_vs_full_pipeline.png")

    ok = stage2_model.ni_pass_fail(stage1_only["sens"], result_final["sens"], stage1_only["spec"], result_final["spec"])
    sens_floor = stage1_only["sens"] * (1 - stage2_model.SENS_LOSS_RATIO_MARGIN)
    pd.DataFrame([{
        "cohort": "sinchon (5-fold OOF)", "sens_before": stage1_only["sens"], "sens_after": result_final["sens"],
        "sens_floor": sens_floor, "spec_before": stage1_only["spec"], "spec_after": result_final["spec"],
        "pass": ok,
    }]).to_csv(OUTPUT_DIR / "final_pipeline_summary.csv", index=False)

    table_row = stage2_model.build_clinical_vs_aec_row(
        "sinchon (5-fold OOF)", y_all, pos_mask, pred_all, stage1_only, result_final, auc2["auc"])
    stage2_model.plot_clinical_vs_aec_table(
        [table_row], OUTPUT_DIR / "clinical_vs_aec_assisted_table.png",
        "clinical-only vs. stage1-score+AEC-sum LR 성능 비교 (Sinchon, 5-fold OOF)")
    pd.DataFrame([table_row]).to_csv(OUTPUT_DIR / "clinical_vs_aec_sum_lr_summary.csv", index=False)

    print("\n=== Stage 2 (stage1-score + raw AEC-128 sum, plain LogisticRegression): TP vs FP ===")
    print(f"AUC={auc2['auc']:.3f} [{auc2['ci_lower']:.3f}, {auc2['ci_upper']:.3f}] "
          f"Mann-Whitney p={auc2['p_value']:.3e}  "
          f"sens={result2['sens']:.3f} spec={result2['spec']:.3f} ppv={result2['ppv']:.3f} npv={result2['npv']:.3f}")

    print("\n=== Full-pipeline effect vs. Stage-1-only ===")
    print(f"Stage-1 AUC={auc_stage1_all['auc']:.3f} [{auc_stage1_all['ci_lower']:.3f}, {auc_stage1_all['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full['auc']:.3f} [{auc_full['ci_lower']:.3f}, {auc_full['ci_upper']:.3f}]  "
          f"DeLong diff={delong['diff']:+.4f} p={delong['p_value']:.4f}")
    r = table_row
    print(f"Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
          f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
          f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})")

    print(f"NI test vs. stage-1-only: sens {stage1_only['sens']:.3f}->{result_final['sens']:.3f} "
          f"(floor={sens_floor:.3f}), spec {stage1_only['spec']:.3f}->{result_final['spec']:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
