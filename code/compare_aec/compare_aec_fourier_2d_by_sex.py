from __future__ import annotations

# TP vs FP AEC-128 2D-image + 2D DFT comparison, stratified by sex (M/F separately).
# Reuses code/compare_aec_fourier_2d.py's pipeline (same TP/FP labels from
# code/baseline/clinic-only_baseline.py, same density-image + fft2 + whole-image RMSD
# permutation test, same three normalizations: raw / patient_norm / global_zscore, same two FFT
# test scales: linear |FFT| and log1p(|FFT|)), and just splits each cohort by PatientSex before
# building each sex's own TP/FP density images. Each sex's TP+FP pooled curves get their own
# value-axis bin edges (not the whole cohort's), so the image reflects the population actually
# being compared within that sex.
# Run: python code/compare_aec/compare_aec_fourier_2d_by_sex.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_aec_fourier_2d as base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "fourier_2d_by_sex"

SEXES = ["M", "F"]


def run_cohort_by_sex(cohort: str, meta: pd.DataFrame, y, score, th: float, xlsx_path: Path, title_suffix: str) -> list[dict]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    aec = base.load_raw_aec(xlsx_path)

    df = rows[["PatientID", "group", "sex"]].merge(aec, on="PatientID", how="inner")
    assert len(df) == len(rows), f"{cohort}: TP/FP merge dropped rows"
    df = df[df["group"].isin(["TP", "FP"])].reset_index(drop=True)

    results = []
    for variant in base.NORMALIZATIONS:
        out_dir = OUTPUT_DIR / cohort / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        for sex in SEXES:
            sub = df[df["sex"] == sex]
            sub_mat = sub[base.AEC_COLS].to_numpy()
            is_tp = (sub["group"].to_numpy() == "TP")

            norm_mat = base.normalize_curves(sub_mat, variant)
            bin_edges = np.linspace(norm_mat.min(), norm_mat.max(), base.N_Y_BINS + 1)
            bin_idx = base.digitize_curves(norm_mat, bin_edges)

            for fft_scale in base.FFT_SCALES:
                r = base.image_2d_diff_test(bin_idx, is_tp, base.N_Y_BINS, base.N_SLICES, log_fft=(fft_scale == "log"))

                fig = base.plot_image_and_spectrum(cohort, variant, r, bin_edges, title_extra=f" ({title_suffix}), sex={sex}")
                fig_path = out_dir / f"tp_vs_fp_aec_2dfft_{variant}_{fft_scale}fft_sex_{sex}.png"
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"[{cohort}/{variant}/{fft_scale}fft/{sex}] n_TP={r['n_tp']} n_FP={r['n_fp']} image_RMSD={r['image_rmsd']:.5f} "
                      f"perm_p={r['p_value']:.4g} peak=(x={r['peak_freq_x_cycle']:.0f},y={r['peak_freq_y_cycle']:.0f}) "
                      f"peak_delta={r['peak_deviation']:.5f}")
                print(f"Saved {fig_path}")

                row_out = {k: v for k, v in r.items() if not k.startswith("_")}
                summary_path = out_dir / f"tp_vs_fp_aec_2dfft_{variant}_{fft_scale}fft_sex_{sex}_summary.csv"
                pd.DataFrame([{"cohort": cohort, "normalization": variant, "sex": sex, **row_out}]).to_csv(summary_path, index=False, encoding="utf-8-sig")
                print(f"Saved {summary_path}")

                results.append({"cohort": cohort, "normalization": variant, "sex": sex, **row_out})
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

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_2dfft_by_sex_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
