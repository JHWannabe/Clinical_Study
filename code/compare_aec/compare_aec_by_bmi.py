from __future__ import annotations

# TP vs FP (Low-SMI vs Normal) AEC-128 curve comparison, faceted by sex (M/F)
# and by within-sex BMI quartile (Q1-Q4) -- i.e. within each sex x BMI-quartile
# cell, compare the same TP/FP groups compare_aec.py / compare_aec_by_sex.py
# already compare (ground-truth Low-SMI patients the classifier caught, TP,
# vs Normal patients it incorrectly flagged, FP; both from
# code/baseline/clinic-only_baseline.py's clinical-only classifier, same
# threshold, same OOF/frozen-model scores it already computes).
# BMI is a body-composition measure whose distribution differs by sex, so
# quartiles are computed within each sex's own TP+FP BMI distribution (not one
# pooled cutoff) -- matching this project's existing convention of splitting
# body-composition variables (TAMA/BMI) by sex-specific cutoffs rather than a
# single pooled cutoff (see code/baseline/aec_curve_comparison_low_smi.py's
# sex_median_group2).
# In addition, run_cohort_by_sex_who_bmi() repeats the same sex-faceted TP-vs-FP
# comparison using fixed WHO Asian-population BMI cutoffs (WHO Expert Consultation,
# Lancet 2004;363:157-163, Table 3: <18.5 / 18.5-22.9 / 23.0-24.9 / 25.0-29.9 / >=30.0)
# instead of within-sex quartiles, so both a cohort-relative and an absolute-cutoff
# view are available. A universal cutoff previously left female Obese TP+FP n=0
# (female Low-SMI-positive predictions skew toward low BMI), so WHO-cutoff groups
# with zero TP+FP patients in a cohort are dropped from that cohort's grid entirely
# rather than shown as a low-n placeholder.
# Reuses code/compare_aec.py's raw-AEC loader and the TP-vs-FP whole-curve
# RMSD permutation test + plot (base.curve_diff_test / base.plot_tp_vs_fp)
# unchanged, raw AEC only (no patient_norm / global_zscore / standard_scaler).
# Each sex x BMI-group cell's own TP+FP matrix is normalized on
# its own (not the whole sex or whole cohort), matching compare_aec_by_sex.py's
# convention of normalizing the population actually being compared.
# Run: python code/compare_aec/compare_aec_by_bmi.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_aec as base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "bmi"
WHO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "bmi_who_asian"

SEXES = ["M", "F"]
BMI_LABELS = ["Q1 (lowest BMI)", "Q2", "Q3", "Q4 (highest BMI)"]
FACET_MIN_N = 3  # minimum TP and FP n within a sex x BMI-group cell to run the permutation test

# WHO Expert Consultation. Appropriate body-mass index for Asian populations and its
# implications for policy and intervention strategies. Lancet 2004;363:157-163, Table 3
# (proposed classification for Asian populations) -- fixed absolute cutoffs, not
# sex-specific and not derived from this cohort's own distribution.
WHO_ASIAN_BMI_BINS = [-np.inf, 18.5, 23.0, 25.0, 30.0, np.inf]
WHO_ASIAN_BMI_LABELS = [
    "Underweight (<18.5)",
    "Normal (18.5-22.9)",
    "Overweight/at-risk (23.0-24.9)",
    "Obese I (25.0-29.9)",
    "Obese II (>=30.0)",
]
# Short forms for per-subplot titles only (WHO_ASIAN_BMI_LABELS above stays the
# canonical label used in CSV output) -- the full labels are too long to fit
# side-by-side subplot titles at this figure width and overlap into neighboring panels.
WHO_ASIAN_BMI_LABELS_SHORT = {
    "Underweight (<18.5)": "Underweight <18.5",
    "Normal (18.5-22.9)": "Normal 18.5-22.9",
    "Overweight/at-risk (23.0-24.9)": "Overweight 23.0-24.9",
    "Obese I (25.0-29.9)": "Obese I 25.0-29.9",
    "Obese II (>=30.0)": "Obese II ≥30.0",
}


def assign_bmi_quartile_by_sex(sex: np.ndarray, bmi: np.ndarray) -> np.ndarray:
    # Quartiles of BMI computed within each sex separately (not one pooled cutoff)
    # so both sexes get 4 roughly-equal-n groups regardless of how the TP+FP
    # subset's BMI distribution differs by sex.
    out = pd.Series(index=np.arange(len(bmi)), dtype=object)
    bmi_s = pd.Series(bmi)
    for s in pd.unique(sex):
        mask = sex == s
        out.loc[mask] = pd.qcut(bmi_s.loc[mask], q=4, labels=BMI_LABELS).astype(str)
    return out.to_numpy()


def assign_bmi_who_asian(bmi: np.ndarray) -> np.ndarray:
    # Fixed WHO Asian-population cutoffs (see WHO_ASIAN_BMI_BINS docstring above),
    # same cutoffs for both sexes -- unlike assign_bmi_quartile_by_sex, this is not
    # computed from this cohort's own BMI distribution.
    return pd.cut(pd.Series(bmi), bins=WHO_ASIAN_BMI_BINS, labels=WHO_ASIAN_BMI_LABELS, right=False).astype(str).to_numpy()


def run_cohort_by_sex_bmi(cohort: str, meta: pd.DataFrame, y, score, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    tp_fp = rows[rows["group"].isin(["TP", "FP"])].copy()
    tp_fp["BMIGroup"] = assign_bmi_quartile_by_sex(tp_fp["sex"].to_numpy(), tp_fp["bmi"].to_numpy())

    aec = base.load_raw_aec(xlsx_path)
    df = tp_fp[["PatientID", "sex", "group", "bmi", "BMIGroup"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(tp_fp), f"{cohort}: TP/FP merge dropped rows"

    results = []
    for variant in ["raw"]:
        ylabel = base.NORM_LABELS[variant]
        out_dir = OUTPUT_DIR / cohort / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(len(SEXES), len(BMI_LABELS), figsize=(7 * len(BMI_LABELS), 5.5 * len(SEXES)), sharey="row")
        variant_results = []
        for row_idx, sex in enumerate(SEXES):
            for col_idx, bmi_label in enumerate(BMI_LABELS):
                ax = axes[row_idx, col_idx]
                cell = df[(df["sex"] == sex) & (df["BMIGroup"] == bmi_label)]
                mat = cell[base.AEC_COLS].to_numpy()
                group_arr = cell["group"].to_numpy()

                # Normalize this sex x BMI-quartile cell's own pooled TP+FP matrix, then split.
                norm_mat = base.normalize_curves(mat, variant)
                mat_tp = norm_mat[group_arr == "TP"]
                mat_fp = norm_mat[group_arr == "FP"]

                title = f"{cohort} ({title_suffix}), sex={sex}, BMI {bmi_label}: {ylabel}-128, TP vs FP"
                if min(len(mat_tp), len(mat_fp)) >= FACET_MIN_N:
                    r = base.curve_diff_test(mat_tp, mat_fp)
                    print(f"[{cohort}/{variant}/sex={sex}/bmi={bmi_label}] n_TP={r['n_tp']} n_FP={r['n_fp']} "
                          f"curve_RMSD={r['curve_rmsd']:.4f} perm_p={r['p_value']:.4g} peak_slice={r['peak_slice']} "
                          f"peak_delta={r['peak_deviation']:.4f}")
                    variant_results.append({"cohort": cohort, "normalization": variant, "sex": sex, "bmi_group": bmi_label, **r})
                    base.plot_tp_vs_fp(ax, mat_tp, mat_fp, r, title, ylabel)
                else:
                    print(f"[{cohort}/{variant}/sex={sex}/bmi={bmi_label}] skipped (n_TP={len(mat_tp)}, n_FP={len(mat_fp)} < {FACET_MIN_N})")
                    ax.text(0.5, 0.5, f"표본 수 부족 (TP={len(mat_tp)}, FP={len(mat_fp)})",
                            transform=ax.transAxes, ha="center", va="center", fontsize=10, color=base.INK_MUTED)
                    ax.set_title(title, color=base.INK_PRIMARY, fontsize=11, fontweight="bold")
                    base.style_axes(ax)

        fig.tight_layout()
        fig_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_bmi.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")

        summary_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_bmi_summary.csv"
        pd.DataFrame(variant_results).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved {summary_path}")

        results += variant_results
    return results


def run_cohort_by_sex_who_bmi(cohort: str, meta: pd.DataFrame, y, score, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    tp_fp = rows[rows["group"].isin(["TP", "FP"])].copy()
    tp_fp["BMIGroup"] = assign_bmi_who_asian(tp_fp["bmi"].to_numpy())

    aec = base.load_raw_aec(xlsx_path)
    df = tp_fp[["PatientID", "sex", "group", "bmi", "BMIGroup"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(tp_fp), f"{cohort}: TP/FP merge dropped rows"

    # Groups with zero TP+FP patients in this cohort (e.g. no Underweight or no Obese II
    # patients at all) are dropped from the grid entirely, not just flagged as low-n --
    # a WHO cutoff group with n=0 isn't a small sample, there's no dataset for it here.
    present_labels = [lbl for lbl in WHO_ASIAN_BMI_LABELS if (df["BMIGroup"] == lbl).any()]
    if not present_labels:
        print(f"[{cohort}] no WHO Asian BMI groups present in TP+FP data, skipping")
        return []

    results = []
    for variant in ["raw"]:
        ylabel = base.NORM_LABELS[variant]
        out_dir = WHO_OUTPUT_DIR / cohort / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(len(SEXES), len(present_labels), figsize=(7.5 * len(present_labels), 5.5 * len(SEXES)),
                                  sharey="row", squeeze=False)
        fig.suptitle(f"{cohort} ({title_suffix}): WHO-Asian BMI, {ylabel}-128 TP vs FP",
                     color=base.INK_PRIMARY, fontsize=13, fontweight="bold")
        variant_results = []
        for row_idx, sex in enumerate(SEXES):
            for col_idx, bmi_label in enumerate(present_labels):
                ax = axes[row_idx, col_idx]
                cell = df[(df["sex"] == sex) & (df["BMIGroup"] == bmi_label)]
                mat = cell[base.AEC_COLS].to_numpy()
                group_arr = cell["group"].to_numpy()

                # Normalize this sex x WHO-BMI-group cell's own pooled TP+FP matrix, then split.
                norm_mat = base.normalize_curves(mat, variant)
                mat_tp = norm_mat[group_arr == "TP"]
                mat_fp = norm_mat[group_arr == "FP"]

                title = f"sex={sex}, {WHO_ASIAN_BMI_LABELS_SHORT[bmi_label]}"
                if min(len(mat_tp), len(mat_fp)) >= FACET_MIN_N:
                    r = base.curve_diff_test(mat_tp, mat_fp)
                    print(f"[{cohort}/{variant}/sex={sex}/who_bmi={bmi_label}] n_TP={r['n_tp']} n_FP={r['n_fp']} "
                          f"curve_RMSD={r['curve_rmsd']:.4f} perm_p={r['p_value']:.4g} peak_slice={r['peak_slice']} "
                          f"peak_delta={r['peak_deviation']:.4f}")
                    variant_results.append({"cohort": cohort, "normalization": variant, "sex": sex, "bmi_group": bmi_label, **r})
                    base.plot_tp_vs_fp(ax, mat_tp, mat_fp, r, title, ylabel)
                else:
                    print(f"[{cohort}/{variant}/sex={sex}/who_bmi={bmi_label}] skipped (n_TP={len(mat_tp)}, n_FP={len(mat_fp)} < {FACET_MIN_N})")
                    ax.text(0.5, 0.5, f"표본 수 부족 (TP={len(mat_tp)}, FP={len(mat_fp)})",
                            transform=ax.transAxes, ha="center", va="center", fontsize=10, color=base.INK_MUTED)
                    ax.set_title(title, color=base.INK_PRIMARY, fontsize=11, fontweight="bold")
                    base.style_axes(ax)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.subplots_adjust(wspace=0.3, hspace=0.35)
        fig_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_who_bmi.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")

        summary_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_who_bmi_summary.csv"
        pd.DataFrame(variant_results).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved {summary_path}")

        results += variant_results
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WHO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = base.baseline

    meta_int, y_int = baseline.load_cohort(baseline.INTERNAL_XLSX)
    x_raw_int = baseline.raw_clinical_matrix(meta_int)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw_int)
    x_int = baseline.apply_clinical_standardizer(x_raw_int, med, mu, sd)
    oof = baseline.oof_scores(x_int, y_int)

    model = baseline.fit_baseline_model(x_int, y_int)
    meta_ext, y_ext = baseline.load_cohort(baseline.EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), med, mu, sd)
    score_ext = model.decision_function(x_ext)

    th = baseline.threshold_for_sensitivity(y_int, oof, baseline.TARGET_SENSITIVITY)

    results = []
    results += run_cohort_by_sex_bmi("gangnam", meta_int, y_int, oof, th, baseline.INTERNAL_XLSX, "internal, OOF")
    results += run_cohort_by_sex_bmi("sinchon", meta_ext, y_ext, score_ext, th, baseline.EXTERNAL_XLSX, "external, frozen internal model")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_curve_by_sex_bmi_summary_all.csv", index=False, encoding="utf-8-sig")

    who_results = []
    who_results += run_cohort_by_sex_who_bmi("gangnam", meta_int, y_int, oof, th, baseline.INTERNAL_XLSX, "internal, OOF")
    who_results += run_cohort_by_sex_who_bmi("sinchon", meta_ext, y_ext, score_ext, th, baseline.EXTERNAL_XLSX, "external, frozen internal model")

    pd.DataFrame(who_results).to_csv(WHO_OUTPUT_DIR / "tp_vs_fp_aec_curve_by_sex_who_bmi_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
