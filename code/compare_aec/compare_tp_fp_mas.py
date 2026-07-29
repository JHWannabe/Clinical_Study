from __future__ import annotations

# TP vs FP comparison of the CT acquisition metadata variable mAs, plus its
# correlation with AEC-sum (the per-patient scalar sum of the raw AEC-128 curve).
# Candidate stratification condition #2 from docs/260729_Stratified Analysis of
# AEC.pptx Slide 12 ("Next Plans" -> "추가 층화 조건 후보"): metadata already has
# mAs but it has never been tested directly against TP/FP or against AEC.
# kVp (also in metadata) is excluded -- it is a constant 100 across every patient
# in both cohorts, so it carries no comparable signal.
# Rationale (docs/stage1_tp_fp_subclassification_ideas.md, 2순위): AEC-128 is
# effectively a per-slice mA(z) tube-current profile, so it may substantially
# overlap with mAs -- checking the correlation first is the cheapest way to find
# out whether mAs carries any TP/FP signal AEC-sum doesn't already carry.
# Not a curve stratification like compare_aec_by_sex.py / compare_aec_by_bmi.py --
# mAs is itself the compared variable here, tested the same way
# code/baseline/clinic-only_baseline.py's error_feature_analysis Welch-t-tests
# age/height/weight/bmi between groups, plus a Mann-Whitney U (matches this
# project's other TP/FP scalar comparisons, e.g. AEC-sum vs clinical in
# stage2_aec_sum_logreg-style scripts, which used rank-based tests for a single
# scalar predictor).
# Run: python code/compare_aec/compare_tp_fp_mas.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import compare_aec as base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compare_aec" / "mas"

COL_TP = base.COL_TP
COL_FP = base.COL_FP


def load_mas(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl")
    out = meta[["PatientID", "mAs"]].copy()
    out["mAs"] = pd.to_numeric(out["mAs"], errors="coerce")
    return out


def compare_mas(df: pd.DataFrame) -> dict:
    tp = df.loc[df["group"] == "TP", "mAs"].to_numpy(dtype=float)
    fp = df.loc[df["group"] == "FP", "mAs"].to_numpy(dtype=float)
    t_stat, t_p = stats.ttest_ind(tp, fp, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(tp, fp, alternative="two-sided")
    return {"n_tp": int(len(tp)), "n_fp": int(len(fp)),
            "tp_mean": float(np.mean(tp)), "fp_mean": float(np.mean(fp)),
            "t_stat": t_stat, "t_p": t_p,
            "mannwhitney_u": u_stat, "mannwhitney_p": u_p}


def correlation_with_aec_sum(df: pd.DataFrame) -> dict:
    sub = df[["mAs", "aec_sum"]].dropna()
    r, p = stats.pearsonr(sub["mAs"].to_numpy(dtype=float), sub["aec_sum"].to_numpy(dtype=float))
    return {"n": int(len(sub)), "pearson_r": r, "pearson_p": p}


def plot_mas(ax_box, ax_scatter, df: pd.DataFrame) -> None:
    tp = df.loc[df["group"] == "TP", "mAs"].dropna().to_numpy(dtype=float)
    fp = df.loc[df["group"] == "FP", "mAs"].dropna().to_numpy(dtype=float)
    ax_box.boxplot([tp, fp], tick_labels=[f"TP (n={len(tp)})", f"FP (n={len(fp)})"],
                    patch_artist=True,
                    boxprops={"facecolor": "#e1e0d9", "edgecolor": base.INK_SECONDARY},
                    medianprops={"color": base.INK_PRIMARY})
    ax_box.scatter(np.full(len(tp), 1) + np.random.default_rng(0).uniform(-0.08, 0.08, len(tp)), tp, color=COL_TP, alpha=0.5, s=14, zorder=3)
    ax_box.scatter(np.full(len(fp), 2) + np.random.default_rng(1).uniform(-0.08, 0.08, len(fp)), fp, color=COL_FP, alpha=0.5, s=14, zorder=3)
    ax_box.set_ylabel("mAs")
    base.style_axes(ax_box)

    colors = np.where(df["group"] == "TP", COL_TP, COL_FP)
    ax_scatter.scatter(df["mAs"], df["aec_sum"], c=colors, alpha=0.4, s=14)
    ax_scatter.set_xlabel("mAs")
    ax_scatter.set_ylabel("AEC-sum (sum of raw AEC-128)")
    base.style_axes(ax_scatter)


def run_cohort(cohort: str, meta: pd.DataFrame, y, score, th: float, xlsx_path: Path, title_suffix: str) -> tuple[dict, dict]:
    rows = base.baseline.build_group_rows(meta, y, score, th)
    tp_fp = rows[rows["group"].isin(["TP", "FP"])].copy()
    mas = load_mas(xlsx_path)
    aec = base.load_raw_aec(xlsx_path)
    aec = aec.assign(aec_sum=aec[base.AEC_COLS].sum(axis=1))

    df = tp_fp[["PatientID", "group"]].merge(mas, on="PatientID", how="inner").merge(aec[["PatientID", "aec_sum"]], on="PatientID", how="inner")
    assert len(df) == len(tp_fp), f"{cohort}: TP/FP merge dropped rows"

    out_dir = OUTPUT_DIR / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    r_cmp = compare_mas(df)
    print(f"[{cohort}/mAs] TP_mean={r_cmp['tp_mean']:.3f} FP_mean={r_cmp['fp_mean']:.3f} "
          f"Welch_t_p={r_cmp['t_p']:.4g} MannWhitney_p={r_cmp['mannwhitney_p']:.4g}")

    r_corr = correlation_with_aec_sum(df)
    print(f"[{cohort}/mAs] Pearson r with AEC-sum = {r_corr['pearson_r']:.4f} (p={r_corr['pearson_p']:.4g}, n={r_corr['n']})")

    fig, (ax_box, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))
    plot_mas(ax_box, ax_scatter, df)
    ax_box.set_title(f"{cohort} ({title_suffix}): mAs by TP/FP", fontsize=11, fontweight="bold", color=base.INK_PRIMARY)
    ax_scatter.set_title(f"{cohort} ({title_suffix}): mAs vs AEC-sum (r={r_corr['pearson_r']:.3f})",
                          fontsize=11, fontweight="bold", color=base.INK_PRIMARY)
    fig.tight_layout()
    fig_path = out_dir / "tp_vs_fp_mas.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    compare_row = {"cohort": cohort, **r_cmp}
    corr_row = {"cohort": cohort, **r_corr}
    pd.DataFrame([compare_row]).to_csv(out_dir / "tp_vs_fp_mas_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([corr_row]).to_csv(out_dir / "mas_aec_sum_correlation.csv", index=False, encoding="utf-8-sig")

    return compare_row, corr_row


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

    compare_all, corr_all = [], []
    for cohort, meta, y, score, xlsx, suffix in [
        ("gangnam", meta_int, y_int, oof, baseline.INTERNAL_XLSX, "internal, OOF"),
        ("sinchon", meta_ext, y_ext, score_ext, baseline.EXTERNAL_XLSX, "external, frozen internal model"),
    ]:
        c, r = run_cohort(cohort, meta, y, score, th, xlsx, suffix)
        compare_all.append(c)
        corr_all.append(r)

    pd.DataFrame(compare_all).to_csv(OUTPUT_DIR / "tp_vs_fp_mas_compare_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(corr_all).to_csv(OUTPUT_DIR / "mas_aec_sum_correlation_all.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
