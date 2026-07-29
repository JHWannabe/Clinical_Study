from __future__ import annotations

# TP vs FP (Low-SMI vs Normal) AEC-128 curve comparison, faceted by sex (M/F)
# and by within-sex quartile of Height and, separately, of Weight -- direct
# body-size variables rather than the composite BMI = Weight/Height^2 used by
# compare_aec_by_bmi.py.
# Candidate stratification condition #3 from docs/260729_Stratified Analysis of
# AEC.pptx Slide 12 ("Next Plans" -> "추가 층화 조건 후보"): BMI compresses two
# body-size axes into one ratio, which does not match the physical definition of
# WED (Water Equivalent Diameter, a cross-sectional size metric AEC/tube-current
# modulation actually responds to). Height and Weight quartiles are tested
# separately here as the two direct body-size axes available in metadata, as a
# WED proxy/complement to BMI quartiles.
# References:
# - AAPM Report 204: Size-Specific Dose Estimates (SSDE) in Pediatric and Adult
#   Body CT Examinations. American Association of Physicists in Medicine, 2011.
# - Bostani M, et al. Attenuation-based size metric for estimating organ dose to
#   patients undergoing tube current modulated CT exams. Med Phys. 2015;42(2):958-968
#   -- regional/attenuation-based WED predicts organ dose significantly better than
#   middle-slice or global size metrics, motivating a body-size (not BMI-ratio) view.
# Reuses code/compare_aec.py's raw-AEC loader and the TP-vs-FP whole-curve RMSD
# permutation test + plot (base.curve_diff_test / base.plot_tp_vs_fp) unchanged,
# raw AEC only. Quartiles are computed within each sex separately (matching
# compare_aec_by_bmi.py's assign_bmi_quartile_by_sex convention), and each
# sex x quartile cell's own TP+FP matrix is normalized on its own.
# Run: python code/compare_aec/compare_aec_by_body_size.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_aec as base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "body_size"

SEXES = ["M", "F"]
QUARTILE_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
FACET_MIN_N = 3  # minimum TP and FP n within a sex x quartile cell to run the permutation test
FEATURES = [("height", "Height (cm)"), ("weight", "Weight (kg)")]


def assign_quartile_by_sex(sex: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Quartiles computed within each sex separately (not one pooled cutoff), matching
    # compare_aec_by_bmi.py's assign_bmi_quartile_by_sex -- Height/Weight distributions
    # differ by sex, so a pooled cutoff would badly imbalance the two sexes' cells.
    out = pd.Series(index=np.arange(len(values)), dtype=object)
    val_s = pd.Series(values)
    for s in pd.unique(sex):
        mask = sex == s
        out.loc[mask] = pd.qcut(val_s.loc[mask], q=4, labels=QUARTILE_LABELS).astype(str)
    return out.to_numpy()


def run_cohort_by_feature(cohort: str, feature: str, feature_label: str, meta: pd.DataFrame,
                            y, score, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    tp_fp = rows[rows["group"].isin(["TP", "FP"])].copy()
    tp_fp["QGroup"] = assign_quartile_by_sex(tp_fp["sex"].to_numpy(), tp_fp[feature].to_numpy())

    aec = base.load_raw_aec(xlsx_path)
    df = tp_fp[["PatientID", "sex", "group", feature, "QGroup"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(tp_fp), f"{cohort}/{feature}: TP/FP merge dropped rows"

    out_dir = OUTPUT_DIR / feature / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SEXES), len(QUARTILE_LABELS), figsize=(7 * len(QUARTILE_LABELS), 5.5 * len(SEXES)), sharey="row")
    results = []
    for row_idx, sex in enumerate(SEXES):
        for col_idx, q_label in enumerate(QUARTILE_LABELS):
            ax = axes[row_idx, col_idx]
            cell = df[(df["sex"] == sex) & (df["QGroup"] == q_label)]
            mat = cell[base.AEC_COLS].to_numpy()
            group_arr = cell["group"].to_numpy()

            norm_mat = base.normalize_curves(mat, "raw")
            mat_tp = norm_mat[group_arr == "TP"]
            mat_fp = norm_mat[group_arr == "FP"]

            title = f"{cohort} ({title_suffix}), sex={sex}, {feature_label} {q_label}: Raw AEC-128, TP vs FP"
            if min(len(mat_tp), len(mat_fp)) >= FACET_MIN_N:
                r = base.curve_diff_test(mat_tp, mat_fp)
                print(f"[{cohort}/{feature}/sex={sex}/{q_label}] n_TP={r['n_tp']} n_FP={r['n_fp']} "
                      f"curve_RMSD={r['curve_rmsd']:.4f} perm_p={r['p_value']:.4g} peak_slice={r['peak_slice']} "
                      f"peak_delta={r['peak_deviation']:.4f}")
                results.append({"cohort": cohort, "feature": feature, "sex": sex, "quartile": q_label, **r})
                base.plot_tp_vs_fp(ax, mat_tp, mat_fp, r, title, "Raw AEC")
            else:
                print(f"[{cohort}/{feature}/sex={sex}/{q_label}] skipped (n_TP={len(mat_tp)}, n_FP={len(mat_fp)} < {FACET_MIN_N})")
                ax.text(0.5, 0.5, f"표본 수 부족 (TP={len(mat_tp)}, FP={len(mat_fp)})",
                        transform=ax.transAxes, ha="center", va="center", fontsize=10, color=base.INK_MUTED)
                ax.set_title(title, color=base.INK_PRIMARY, fontsize=11, fontweight="bold")
                base.style_axes(ax)

    fig.tight_layout()
    fig_path = out_dir / f"tp_vs_fp_aec_curve_raw_by_sex_{feature}_quartile.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    summary_path = out_dir / f"tp_vs_fp_aec_curve_raw_by_sex_{feature}_quartile_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved {summary_path}")

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    all_results = []
    for feature, feature_label in FEATURES:
        for cohort, meta, y, score, xlsx, suffix in [
            ("gangnam", meta_int, y_int, oof, baseline.INTERNAL_XLSX, "internal, OOF"),
            ("sinchon", meta_ext, y_ext, score_ext, baseline.EXTERNAL_XLSX, "external, frozen internal model"),
        ]:
            all_results += run_cohort_by_feature(cohort, feature, feature_label, meta, y, score, th, xlsx, suffix)

    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_curve_by_body_size_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
