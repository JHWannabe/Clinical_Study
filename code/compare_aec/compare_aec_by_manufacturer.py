from __future__ import annotations

# TP vs FP AEC-128 curve comparison, stratified by CT scanner model (metadata's
# "Manufacturer" column -- despite the name this is the scanner model string,
# e.g. "Sensation 64", "Revolution CT", not the company name).
# Candidate stratification condition #1 from docs/260729_Stratified Analysis of
# AEC.pptx Slide 12 ("Next Plans" -> "추가 층화 조건 후보"): the Manufacturer
# column exists in metadata but has never been tested against AEC-128.
# Reference: Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
# expression data using empirical Bayes methods. Biostatistics. 2007;8(1):118-127
# (ComBat) -- scanner/site is a canonical source of systematic "batch" offset in
# imaging-derived measurements, which is the rationale for testing a vendor main
# effect on AEC-128 before (or instead of) assuming any TP/FP AEC difference is
# purely biological.
#
# Two tests per cohort:
# 1. Vendor main effect (omnibus): among the pooled TP+FP screen-positive curves,
#    do scanner models with enough n differ systematically in raw AEC-128 -- a
#    permutation test on the weighted between-vendor RMSD-to-grand-mean (ANOVA-like
#    stat, but distribution-free), analogous to compare_aec.py's / compare_aec_by_bmi.py's
#    two-group RMSD permutation test extended to >2 groups.
# 2. TP vs FP within each vendor with enough n (reuses base.curve_diff_test /
#    base.plot_tp_vs_fp unchanged, raw AEC only), same convention as
#    compare_aec_by_sex.py / compare_aec_by_bmi.py: normalize each vendor's own
#    pooled TP+FP matrix, then split.
# Run: python code/compare_aec/compare_aec_by_manufacturer.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_aec as base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "manufacturer"

FACET_MIN_N = 3  # minimum TP and FP n within a vendor cell to run the TP-vs-FP permutation test
OMNIBUS_MIN_N = 10  # minimum pooled TP+FP n within a vendor to include it in the vendor main-effect test
N_PERM = 2000
SEED = 20260709

VENDOR_COLORS = ["#1b4fd8", "#d8271b", "#2e8b57", "#7a4fd8", "#e0762c", "#00acc1", "#b8860b", "#c22a68"]


def load_manufacturer(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl")
    return meta[["PatientID", "Manufacturer"]].copy()


def vendor_main_effect_test(group_mats: dict[str, np.ndarray], n_perm: int = N_PERM, seed: int = SEED) -> dict:
    # Weighted between-vendor RMSD-to-grand-mean, permutation-tested by shuffling
    # vendor labels across the pooled rows (n per vendor held fixed each draw) --
    # an ANOVA-like "does the group mean curve move more than chance" statistic,
    # but with a permutation null instead of an F-distribution assumption.
    labels = np.concatenate([[name] * len(mat) for name, mat in group_mats.items()])
    mat = np.vstack(list(group_mats.values()))
    names = list(group_mats.keys())

    def stat(lab: np.ndarray) -> float:
        grand_mean = mat.mean(axis=0)
        total = 0.0
        for name in names:
            sub = mat[lab == name]
            dev = sub.mean(axis=0) - grand_mean
            total += len(sub) * np.mean(dev ** 2)
        return float(np.sqrt(total / len(mat)))

    obs_stat = stat(labels)
    rng = np.random.default_rng(seed)
    perm_labels = labels.copy()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(perm_labels)
        perm_stats[i] = stat(perm_labels)
    p_value = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)

    return {"vendors": names, "n_per_vendor": [int(len(group_mats[n])) for n in names],
            "between_vendor_rmsd": obs_stat, "n_perm": n_perm, "p_value": float(p_value)}


def plot_vendor_means(ax, group_mats: dict[str, np.ndarray], title: str, ylabel: str) -> None:
    x = np.arange(1, base.N_SLICES + 1)
    for i, (name, mat) in enumerate(group_mats.items()):
        color = VENDOR_COLORS[i % len(VENDOR_COLORS)]
        ax.plot(x, mat.mean(axis=0), color=color, linewidth=2, label=f"{name} (n={len(mat)})")
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Slice index (1-128)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=base.INK_PRIMARY, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=base.INK_SECONDARY, loc="best")
    base.style_axes(ax)


def run_cohort(cohort: str, meta: pd.DataFrame, y, score, th: float, xlsx_path: Path, title_suffix: str) -> tuple[list[dict], list[dict]]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    tp_fp = rows[rows["group"].isin(["TP", "FP"])].copy()
    vendor = load_manufacturer(xlsx_path)
    aec = base.load_raw_aec(xlsx_path)

    df = tp_fp[["PatientID", "group"]].merge(vendor, on="PatientID", how="inner").merge(aec, on="PatientID", how="inner")
    assert len(df) == len(tp_fp), f"{cohort}: TP/FP merge dropped rows"

    out_dir = OUTPUT_DIR / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    vendor_counts = df["Manufacturer"].value_counts()
    omnibus_vendors = vendor_counts[vendor_counts >= OMNIBUS_MIN_N].index.tolist()
    print(f"[{cohort}] vendor n (pooled TP+FP): {vendor_counts.to_dict()}")
    print(f"[{cohort}] vendors with n>={OMNIBUS_MIN_N} used in main-effect test: {omnibus_vendors}")

    omnibus_results = []
    if len(omnibus_vendors) >= 2:
        group_mats = {v: df.loc[df["Manufacturer"] == v, base.AEC_COLS].to_numpy() for v in omnibus_vendors}
        r = vendor_main_effect_test(group_mats)
        print(f"[{cohort}/vendor_main_effect] vendors={r['vendors']} n={r['n_per_vendor']} "
              f"between_vendor_RMSD={r['between_vendor_rmsd']:.4f} perm_p={r['p_value']:.4g}")
        omnibus_results.append({"cohort": cohort, **r})

        fig, ax = plt.subplots(figsize=(9, 5.5))
        note = f"between-vendor RMSD={r['between_vendor_rmsd']:.4f}, p={r['p_value']:.3g}"
        plot_vendor_means(ax, group_mats, f"{cohort} ({title_suffix}): raw AEC-128 mean by scanner model\n{note}", "Raw AEC")
        fig.tight_layout()
        fig_path = out_dir / "vendor_main_effect_aec_curve_raw.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")
    else:
        print(f"[{cohort}] fewer than 2 vendors with n>={OMNIBUS_MIN_N}, skipping main-effect test")

    tp_fp_vendors = [v for v in vendor_counts.index
                      if ((df["Manufacturer"] == v) & (df["group"] == "TP")).sum() >= FACET_MIN_N
                      and ((df["Manufacturer"] == v) & (df["group"] == "FP")).sum() >= FACET_MIN_N]
    print(f"[{cohort}] vendors with n_TP,n_FP>={FACET_MIN_N} used in TP-vs-FP tests: {tp_fp_vendors}")

    tp_fp_results = []
    if tp_fp_vendors:
        n_cols = len(tp_fp_vendors)
        fig, axes = plt.subplots(1, n_cols, figsize=(7.5 * n_cols, 5.5), squeeze=False)
        fig.suptitle(f"{cohort} ({title_suffix}): Raw AEC-128, TP vs FP by scanner model",
                     color=base.INK_PRIMARY, fontsize=13, fontweight="bold")
        for col_idx, v in enumerate(tp_fp_vendors):
            ax = axes[0, col_idx]
            cell = df[df["Manufacturer"] == v]
            mat = cell[base.AEC_COLS].to_numpy()
            group_arr = cell["group"].to_numpy()
            norm_mat = base.normalize_curves(mat, "raw")
            mat_tp = norm_mat[group_arr == "TP"]
            mat_fp = norm_mat[group_arr == "FP"]

            r = base.curve_diff_test(mat_tp, mat_fp)
            print(f"[{cohort}/vendor={v}] n_TP={r['n_tp']} n_FP={r['n_fp']} curve_RMSD={r['curve_rmsd']:.4f} "
                  f"perm_p={r['p_value']:.4g} peak_slice={r['peak_slice']} peak_delta={r['peak_deviation']:.4f}")
            tp_fp_results.append({"cohort": cohort, "vendor": v, **r})
            base.plot_tp_vs_fp(ax, mat_tp, mat_fp, r, f"vendor={v}", "Raw AEC")

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.subplots_adjust(wspace=0.3)
        fig_path = out_dir / "tp_vs_fp_aec_curve_raw_by_vendor.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")

    summary_path = out_dir / "tp_vs_fp_aec_curve_raw_by_vendor_summary.csv"
    pd.DataFrame(tp_fp_results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved {summary_path}")

    return omnibus_results, tp_fp_results


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

    omnibus_all, tp_fp_all = [], []
    for cohort, meta, y, score, xlsx, suffix in [
        ("gangnam", meta_int, y_int, oof, baseline.INTERNAL_XLSX, "internal, OOF"),
        ("sinchon", meta_ext, y_ext, score_ext, baseline.EXTERNAL_XLSX, "external, frozen internal model"),
    ]:
        o, t = run_cohort(cohort, meta, y, score, th, xlsx, suffix)
        omnibus_all += o
        tp_fp_all += t

    pd.DataFrame(omnibus_all).to_csv(OUTPUT_DIR / "vendor_main_effect_summary_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tp_fp_all).to_csv(OUTPUT_DIR / "tp_vs_fp_aec_curve_raw_by_vendor_summary_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
