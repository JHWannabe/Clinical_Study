from __future__ import annotations

# TP vs FP AEC-128 curve comparison, stratified by sex (M/F separately).
# Reuses code/compare_aec.py's pipeline (same TP/FP labels from
# code/baseline/clinic-only_baseline.py, same whole-curve RMSD permutation test),
# raw AEC only (no patient_norm / global_zscore / standard_scaler),
# and just splits each cohort by PatientSex before running/plotting the TP-vs-FP
# comparison. Each sex's TP+FP pooled matrix is normalized on its own (not the
# whole cohort) so global_zscore/standard_scaler stats reflect the population
# actually being compared within that sex, matching compare_aec.py's convention.
# Run: python code/compare_aec/compare_aec_by_sex.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import compare_aec as base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "sex"

SEXES = ["M", "F"]


def run_cohort_by_sex(cohort: str, meta: pd.DataFrame, y, score, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    aec = base.load_raw_aec(xlsx_path)

    df = rows[["PatientID", "group", "sex"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(rows), f"{cohort}: TP/FP merge dropped rows"

    results = []
    for variant in ["raw"]:
        ylabel = base.NORM_LABELS[variant]
        out_dir = OUTPUT_DIR / cohort / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        fig_s, axes_s = plt.subplots(1, 2, figsize=(15, 5.5))
        sampled_all = []
        variant_results = []
        for ax, ax_s, sex in zip(axes, axes_s, SEXES):
            sub = df[df["sex"] == sex]
            sub_mat = sub[base.AEC_COLS].to_numpy()
            group_arr = sub["group"].to_numpy()
            id_arr = sub["PatientID"].to_numpy()

            # Normalize this sex's pooled TP+FP matrix, then split.
            norm_mat = base.normalize_curves(sub_mat, variant)
            mat_tp = norm_mat[group_arr == "TP"]
            mat_fp = norm_mat[group_arr == "FP"]
            ids_tp = id_arr[group_arr == "TP"]
            ids_fp = id_arr[group_arr == "FP"]

            r = base.curve_diff_test(mat_tp, mat_fp)
            base.plot_tp_vs_fp(ax, mat_tp, mat_fp, r, f"{cohort} ({title_suffix}), sex={sex}: {ylabel}-128, TP vs FP", ylabel)
            print(f"[{cohort}/{variant}/{sex}] n_TP={r['n_tp']} n_FP={r['n_fp']} curve_RMSD={r['curve_rmsd']:.4f} "
                  f"perm_p={r['p_value']:.4g} peak_slice={r['peak_slice']} peak_delta={r['peak_deviation']:.4f}")
            variant_results.append({"cohort": cohort, "normalization": variant, "sex": sex, **r})

            sampled = base.plot_tp_vs_fp_samples(ax_s, mat_tp, ids_tp, mat_fp, ids_fp,
                                                  f"{cohort} ({title_suffix}), sex={sex}: {ylabel}-128 samples, TP vs FP",
                                                  ylabel)
            for row in sampled:
                row["sex"] = sex
            sampled_all += sampled

        fig.tight_layout()
        fig_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")

        fig_s.tight_layout()
        samples_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_samples.png"
        fig_s.savefig(samples_path, dpi=200, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved {samples_path}")

        sampled_ids_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_samples_patient_ids.csv"
        pd.DataFrame(sampled_all).to_csv(sampled_ids_path, index=False, encoding="utf-8-sig")
        print(f"Saved {sampled_ids_path}")

        summary_path = out_dir / f"tp_vs_fp_aec_curve_{variant}_by_sex_summary.csv"
        pd.DataFrame(variant_results).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved {summary_path}")

        results += variant_results
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

    results = []
    results += run_cohort_by_sex("gangnam", meta_int, y_int, oof, th, baseline.INTERNAL_XLSX, "internal, OOF")
    results += run_cohort_by_sex("sinchon", meta_ext, y_ext, score_ext, th, baseline.EXTERNAL_XLSX, "external, frozen internal model")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_curve_by_sex_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
