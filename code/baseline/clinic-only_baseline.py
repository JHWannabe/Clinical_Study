from __future__ import annotations

# Clinical-only low-SMI classifier on the internal cohort (g1090.xlsx) only.
# Fits a Logistic Regression on {PatientAge, Height, Weight, sex} via 5-fold
# cross-validation, picks the score threshold that hits >=90% sensitivity on the
# out-of-fold predictions, and saves the resulting confusion matrix as a figure + csv.
# Run: python code/new_hypothesis.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0_clinic-only_baseline"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
TARGET_SENSITIVITIES = [0.85, 0.90, 0.95]
TARGET_SENSITIVITY = 0.90
N_FOLDS = 5
SEED = 20260709


def load_cohort(xlsx_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return meta, y


def raw_clinical_matrix(meta: pd.DataFrame) -> np.ndarray:
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(int)
    age = pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=int)
    height = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([sex_m, age, height, weight, ])


def fit_clinical_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Column 0 is sex_m (binary indicator) -- left unscaled (mu=0, sd=1),
    # only age/height/weight (columns 1:) are standardized.
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    scaler = StandardScaler().fit(x[:, 1:])
    mu = np.concatenate([[0.0], np.asarray(scaler.mean_)])
    sd = np.concatenate([[1.0], np.asarray(scaler.scale_)])
    return med, mu, sd


def apply_clinical_standardizer(x: np.ndarray, med: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = np.where(np.isfinite(x), x, med)
    return (x - mu) / sd


def clinical_features(meta: pd.DataFrame) -> np.ndarray:
    x = raw_clinical_matrix(meta)
    med, mu, sd = fit_clinical_standardizer(x)
    return apply_clinical_standardizer(x, med, mu, sd)


def oof_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x, y)):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED + fold_id)
        model.fit(x[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(x[va_idx])[:, 1]
    return oof


def fit_baseline_model(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> LogisticRegression:
    # Canonical clinical-only baseline (fit on the full cohort, no CV holdout) --
    # every script needing a "baseline" model must reuse this, not refit its own.
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed).fit(x, y)


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    pred = pred.astype(bool)
    pos = y.astype(bool)
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def threshold_for_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    best = None
    for th in np.unique(score):
        tp, fp, fn, tn = confusion_counts(y, score >= th)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        if np.isfinite(sens) and sens >= target and (best is None or spec > best[1]):
            best = (float(th), spec)
    if best is None:
        return float(np.quantile(score[y == 1], 1 - target))
    return best[0]


def evaluate(cohort: str, y: np.ndarray, pred: np.ndarray, th: float) -> dict:
    tp, fp, fn, tn = confusion_counts(y, pred)
    n = len(y)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    print(f"[{cohort}] threshold={th:.4f} sensitivity={sens:.3f} specificity={spec:.3f} ppv={ppv:.3f} npv={npv:.3f} n={n}")
    return {"cohort": cohort, "matrix": np.array([[tp, fn], [fp, tn]]), "th": th, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv}


def plot_confusion_matrix(ax: Axes, result: dict) -> None:
    matrix = result["matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center", fontsize=13, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    label = "internal, OOF" if result["cohort"] == "internal" else "external, frozen internal model"
    ax.set_title(f"{label}\n(threshold={result['th']:.3f})", fontsize=11, fontweight="bold")


def plot_comparison_summary(results: list[dict], out_path: Path) -> None:
    metrics = ["sens", "spec", "ppv", "npv"]
    metric_labels = ["Sensitivity", "Specificity", "PPV", "NPV"]
    cohorts = ["internal", "external"]
    targets = sorted({r["target"] for r in results})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    width = 0.2
    x = np.arange(len(metrics))
    for ax, cohort in zip(axes, cohorts):
        for i, target in enumerate(targets):
            result = next(r for r in results if r["cohort"] == cohort and r["target"] == target)
            values = [result[m] for m in metrics]
            ax.bar(x + (i - (len(targets) - 1) / 2) * width, values, width, label=f"S{target * 100:0.0f}")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(cohort)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[1].legend(title="Sensitivity target", loc="lower right")
    fig.suptitle("Clinical-only Logistic Regression: S85 vs S90 vs S95", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def auc_significance_stats(y: np.ndarray, score: np.ndarray, n_boot: int = 3000, seed: int = SEED) -> dict:
    # AUC vs 0.5 significance: Mann-Whitney U on the score is the rank-based test
    # equivalent to "is AUC different from chance" (U relates to AUC by AUC = U / (n_pos*n_neg)).
    # CI on AUC itself comes from a percentile bootstrap stratified on y (resample
    # positives and negatives separately so every resample keeps both classes present).
    auc = roc_auc_score(y, score)
    pos, neg = score[y == 1], score[y == 0]
    u_stat, p_value = stats.mannwhitneyu(pos, neg, alternative="two-sided")

    rng = np.random.default_rng(seed)
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boot_aucs = np.empty(n_boot)
    for i in range(n_boot):
        bi = np.concatenate([rng.choice(idx_pos, size=len(idx_pos), replace=True),
                              rng.choice(idx_neg, size=len(idx_neg), replace=True)])
        boot_aucs[i] = roc_auc_score(y[bi], score[bi])
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    return {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "auc": float(auc),
            "ci_lower": float(ci_lo), "ci_upper": float(ci_hi),
            "mannwhitney_u": float(u_stat), "p_value": float(p_value)}


def plot_roc_curve(y: np.ndarray, score: np.ndarray, auc_stats: dict, out_path: Path, title: str) -> None:
    # AUC is the headline number, so it gets hero-figure treatment (large, bold, in
    # the plot's dead space below the curve) instead of being buried at legend font
    # size -- CI/p-value/n ride underneath it as a de-emphasized secondary line.
    INK_PRIMARY = "#161616"
    INK_MUTED = "#6b6a66"
    CURVE_COLOR = "#2a78d6"

    fpr, tpr, _ = roc_curve(y, score)
    p = auc_stats["p_value"]
    p_str = "p<1e-300" if p == 0 else f"p={p:.2e}"

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(fpr, tpr, color=CURVE_COLOR, linewidth=2.5, solid_capstyle="round")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)

    for target_sens in TARGET_SENSITIVITIES:
        ax.axhline(target_sens, color=INK_MUTED, linestyle=":", linewidth=1, alpha=0.7, zorder=0)
        ax.text(0.02, target_sens + 0.012, f"Se={target_sens:.2f}", ha="left", va="bottom",
                fontsize=8.5, color=INK_MUTED)

    ax.text(0.97, 0.16, f"AUC = {auc_stats['auc']:.3f}", ha="right", va="bottom",
            fontsize=26, fontweight="bold", color=INK_PRIMARY, transform=ax.transAxes)
    ax.text(0.97, 0.10, f"95% CI [{auc_stats['ci_lower']:.3f}, {auc_stats['ci_upper']:.3f}]",
            ha="right", va="bottom", fontsize=10, color=INK_MUTED, transform=ax.transAxes)
    ax.text(0.97, 0.045, f"Mann-Whitney U {p_str}  (n={auc_stats['n_pos']}/{auc_stats['n_neg']})",
            ha="right", va="bottom", fontsize=9, color=INK_MUTED, transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK_PRIMARY)
    ax.grid(alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve to {out_path}")


def build_group_rows(meta: pd.DataFrame, y: np.ndarray, score: np.ndarray, th: float) -> pd.DataFrame:
    # Per-patient feature/score table with TP/FN/FP/TN group labels, for reuse by
    # error_feature_analysis and any downstream stage that needs the same rows.
    age = pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float)
    height = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (height / 100.0) ** 2
    bmi = weight / (height / 100.0) ** 2

    pred = score >= th
    pos = y.astype(bool)
    group = np.select(
        [pos & pred, pos & ~pred, ~pos & pred, ~pos & ~pred],
        ["TP", "FN", "FP", "TN"],
        default="",
    )

    return pd.DataFrame({
        "PatientID": meta["PatientID"].to_numpy(),
        "group": group, "age": age, "height": height, "weight": weight,
        "sex": sex, "smi": smi, "bmi": bmi, "score": score,
    })


def error_feature_analysis(meta: pd.DataFrame, y: np.ndarray, score: np.ndarray, th: float, model: LogisticRegression, out_dir: Path) -> None:
    rows = build_group_rows(meta, y, score, th)
    rows.to_csv(out_dir / "error_feature_analysis_rows.csv", index=False)
    group = rows["group"].to_numpy()
    age = rows["age"].to_numpy()
    height = rows["height"].to_numpy()
    weight = rows["weight"].to_numpy()
    bmi = rows["bmi"].to_numpy()

    lines = ["# Error feature analysis: TP/FN/TN/FP", "", "Clinical-only classifier analyzed on internal OOF predictions.", ""]

    lines += ["## Group sizes", ""]
    lines.append(rows["group"].value_counts().rename_axis("group").reset_index(name="n").to_markdown(index=False))
    lines.append("")

    lines += ["## Feature means by group", ""]
    means = rows.groupby("group")[["age", "height", "weight", "bmi", "smi", "score"]].mean().round(2)
    lines.append(means.reset_index().to_markdown(index=False))
    lines.append("")

    lines += ["## Sex composition by group (fraction Male)", ""]
    frac_male = rows.groupby("group")["sex"].apply(lambda s: (s == "M").mean()).round(3)
    lines.append(frac_male.rename("frac_male").reset_index().to_markdown(index=False))
    lines.append("")

    lines += ["## Full-data LR coefficients (standardized features: age, height, weight, sex_M)", ""]
    coef_df = pd.DataFrame({
        "coefficient": np.concatenate([model.coef_.ravel(), model.intercept_]),
    }, index=["age", "height", "weight", "sex_M", "intercept"]).round(4)
    lines.append(coef_df.to_markdown())
    lines.append("")

    lines += ["## Correlation of OOF score with derived BMI / raw features", ""]
    corr = pd.Series({
        "bmi": np.corrcoef(score, bmi)[0, 1],
        "height": np.corrcoef(score, height)[0, 1],
        "weight": np.corrcoef(score, weight)[0, 1],
    }).round(4)
    lines.append(corr.rename("correlation_with_score").to_frame().to_markdown())
    lines.append("")

    def welch_table(group_a: str, group_b: str) -> pd.DataFrame:
        a_mask, b_mask = group == group_a, group == group_b
        out_rows = []
        for feat, arr in [("age", age), ("height", height), ("weight", weight), ("bmi", bmi)]:
            a, b = arr[a_mask], arr[b_mask]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            out_rows.append({f"{group_a}_mean": a.mean(), f"{group_b}_mean": b.mean(), "diff": a.mean() - b.mean(), "t": t, "p": p})
        return pd.DataFrame(out_rows, index=["age", "height", "weight", "bmi"]).round(4)

    lines += ["## TP vs FN (among actual low-SMI positives): Welch t-test", ""]
    lines.append(welch_table("TP", "FN").reset_index(names="feature").to_markdown(index=False))
    lines.append("")

    lines += ["## TN vs FP (among actual negatives): Welch t-test", ""]
    lines.append(welch_table("TN", "FP").reset_index(names="feature").to_markdown(index=False))
    lines.append("")

    def chi_square(group_a: str, group_b: str) -> tuple[pd.DataFrame, float, float]:
        table = pd.crosstab(rows["group"], rows["sex"]).loc[[group_a, group_b]]
        res = stats.chi2_contingency(table)
        return table, float(res.statistic), float(res.pvalue)  # type: ignore[attr-defined]

    lines += ["## Sex distribution chi-square: TP vs FN", ""]
    table, chi2, p = chi_square("TP", "FN")
    lines.append(table.reset_index().to_markdown(index=False))
    lines.append("")
    lines.append(f"chi2={chi2:.2f}, p={p:.4f}")
    lines.append("")

    lines += ["## Sex distribution chi-square: TN vs FP", ""]
    table, chi2, p = chi_square("TN", "FP")
    lines.append(table.reset_index().to_markdown(index=False))
    lines.append("")
    lines.append(f"chi2={chi2:.2f}, p={p:.4f}")
    lines.append("")

    (out_dir / "error_feature_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved error feature analysis to {out_dir / 'error_feature_analysis.md'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal: fit standardizer once on internal only ---
    meta_int, y_int = load_cohort(INTERNAL_XLSX)
    x_raw_int = raw_clinical_matrix(meta_int)
    med, mu, sd = fit_clinical_standardizer(x_raw_int)
    x_int = apply_clinical_standardizer(x_raw_int, med, mu, sd)
    oof = oof_scores(x_int, y_int)

    auc_stats = auc_significance_stats(y_int, oof)
    print(f"[internal / OOF] AUC={auc_stats['auc']:.4f} "
          f"95%CI=[{auc_stats['ci_lower']:.4f}, {auc_stats['ci_upper']:.4f}] "
          f"Mann-Whitney p={auc_stats['p_value']:.3e}")
    pd.DataFrame([auc_stats]).to_csv(OUTPUT_DIR / "clinical_only_auc_significance.csv", index=False)
    plot_roc_curve(y_int, oof, auc_stats, OUTPUT_DIR / "clinical_only_roc_curve.png",
                   "Clinical-only Logistic Regression: ROC (internal, OOF)")

    model = fit_baseline_model(x_int, y_int)
    meta_ext, y_ext = load_cohort(EXTERNAL_XLSX)
    x_ext = apply_clinical_standardizer(raw_clinical_matrix(meta_ext), med, mu, sd)
    score_ext = model.predict_proba(x_ext)[:, 1]

    all_results = []
    summary_rows = []
    th90 = None
    for target in TARGET_SENSITIVITIES:
        th = threshold_for_sensitivity(y_int, oof, target)
        if target == 0.90:
            th90 = th
        result_int = evaluate("internal", y_int, oof >= th, th)
        result_ext = evaluate("external", y_ext, score_ext >= th, th)
        result_int["target"] = target
        result_ext["target"] = target
        all_results.extend([result_int, result_ext])

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle(f"Clinical-only Logistic Regression (S{target * 100:.0f})", fontsize=13, fontweight="bold")
        plot_confusion_matrix(axes[0], result_int)
        plot_confusion_matrix(axes[1], result_ext)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = OUTPUT_DIR / f"clinical_only_confusion_matrix_sens{int(target * 100)}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Saved confusion matrix to {out_path}")

        for result in (result_int, result_ext):
            summary_rows.append({
                "target_sensitivity": target,
                "cohort": result["cohort"],
                "threshold": result["th"],
                "sensitivity": result["sens"],
                "specificity": result["spec"],
                "ppv": result["ppv"],
                "npv": result["npv"],
                "n": int(result["matrix"].sum()),
            })

    assert th90 is not None, "0.90 must be in TARGET_SENSITIVITIES for error_feature_analysis"
    error_feature_analysis(meta_int, y_int, oof, th90, model, OUTPUT_DIR)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "clinical_only_sensitivity_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved comparison table to {summary_path}")

    comparison_fig_path = OUTPUT_DIR / "clinical_only_sensitivity_comparison.png"
    plot_comparison_summary(all_results, comparison_fig_path)
    print(f"Saved comparison figure to {comparison_fig_path}")


if __name__ == "__main__":
    main()
