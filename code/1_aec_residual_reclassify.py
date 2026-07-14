from __future__ import annotations

# Stage-2 reclassification of the clinical-only low-SMI screen.
#
# Stage 1 (clinic-only_baseline.py): LR on {age, height, weight, sex}, threshold
# chosen at Sensitivity>=90% on internal OOF. This yields a large "screen positive"
# group (TP+FP) with poor PPV.
#
# Stage 2: for patients the stage-1 screen calls Positive (score >= th1), refit a
# second classifier using clinical features + AEC-128 curve information that is
# NOT already explained by clinical variables (residualized against age/height/
# weight/sex, then reduced via PCA fit on internal only). The stage-2 threshold is
# chosen to maximize PPV of the screen-positive group subject to a hard floor:
# global sensitivity must not drop more than 5pp below the stage-1-only baseline
# (see acceptance-criteria memory). Predicted-negative patients (score < th1) are
# left untouched by design -- the FN group is too small (n=12 internal) to support
# a reliable stage-2 correction, and touching TN/FN risks hurting specificity or
# sensitivity for no reliable gain.
#
# Run: python code/1_aec_residual_reclassify.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1_aec_residual_reclassify"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"

N_FOLDS = baseline.N_FOLDS
SEED = baseline.SEED
TARGET_SENSITIVITY = baseline.TARGET_SENSITIVITY
SENS_NONINF_MARGIN = 0.05  # acceptance-criteria memory: sensitivity may not drop more than 5pp
NI_ALPHA = 0.025  # one-sided; equivalent to checking the bound of a two-sided 95% CI
NI_Z = float(norm.ppf(1 - NI_ALPHA))

N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
PCA_VAR_TARGET = 0.90
PCA_N_MAX = 10

PIPE_COL_A = "#2a78d6"   # Low group / PC1 (blue)
PIPE_COL_B = "#eda100"   # High group / PC2 (yellow)
PIPE_COL_C = "#1baf7a"   # accent (aqua)
PIPE_INK_PRIMARY = "#0b0b0b"
PIPE_INK_SECONDARY = "#52514e"
PIPE_INK_MUTED = "#898781"
PIPE_GRID = "#e1e0d9"
PIPE_SURFACE = "#fcfcfb"


def load_cohort_with_aec(xlsx_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta, y = baseline.load_cohort(xlsx_path)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")

    curves_raw = aec[AEC_COLS].astype(float).to_numpy()
    patient_mean = curves_raw.mean(axis=1, keepdims=True)
    norm_curves = curves_raw / patient_mean  # patient-normalized AEC (same convention as aec_curve_comparison.py)

    aec_df = pd.DataFrame(norm_curves, columns=AEC_COLS)
    aec_df.insert(0, "PatientID", aec["PatientID"].to_numpy())

    order = meta[["PatientID"]].copy()
    order["__row__"] = np.arange(len(meta))
    merged = order.merge(aec_df, on="PatientID", how="left").sort_values("__row__")
    curves = merged[AEC_COLS].to_numpy(dtype=float)

    if not np.all(np.isfinite(curves)):
        missing = int((~np.all(np.isfinite(curves), axis=1)).sum())
        raise ValueError(f"{missing} patients in {xlsx_path.name} have no matching aec_128 row")

    return meta, y, curves


def fit_aec_residualizer(clin_std: np.ndarray, curves: np.ndarray) -> LinearRegression:
    # Regress each of the 128 patient-normalized slice points on the standardized
    # clinical features, fit on internal only. The residual keeps only the AEC
    # information clinical variables don't already explain (see project_aec_curve_bmi_confound).
    return LinearRegression().fit(clin_std, curves)


def apply_aec_residualizer(reg: LinearRegression, clin_std: np.ndarray, curves: np.ndarray) -> np.ndarray:
    return curves - reg.predict(clin_std)


def fit_residual_pca(resid: np.ndarray, var_target: float = PCA_VAR_TARGET, n_max: int = PCA_N_MAX,
                      fixed_k: int | None = None) -> PCA:
    if fixed_k is not None:
        return PCA(n_components=fixed_k).fit(resid)
    probe = PCA(n_components=min(n_max, resid.shape[1])).fit(resid)
    cum = np.cumsum(probe.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = max(1, min(k, n_max))
    return PCA(n_components=k).fit(resid)


def stage2_feature_matrix(clin_std: np.ndarray, aec_pca_scores: np.ndarray, stage1_score: np.ndarray | None = None) -> np.ndarray:
    cols = [clin_std, aec_pca_scores]
    if stage1_score is not None:
        cols.append(stage1_score.reshape(-1, 1))
    return np.column_stack(cols)


def make_stage2_model(model_type: str, seed: int):
    if model_type == "logreg":
        return LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    if model_type == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, random_state=seed)
    raise ValueError(model_type)


def stage2_score_fn(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def stage2_oof_scores(x2: np.ndarray, y: np.ndarray, pos_mask: np.ndarray, model_type: str) -> np.ndarray:
    # Reuses the exact same StratifiedKFold(n_splits, shuffle, seed) split as stage-1's
    # oof_scores (fold assignment depends only on y/seed, not on x), so a patient's
    # "screen positive" membership and stage-2 fold membership never leak into each other.
    scores = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x2, y)):
        tr_pos = tr_idx[pos_mask[tr_idx]]
        va_pos = va_idx[pos_mask[va_idx]]
        if len(va_pos) == 0 or len(np.unique(y[tr_pos])) < 2:
            continue
        model = make_stage2_model(model_type, SEED + fold_id)
        model.fit(x2[tr_pos], y[tr_pos])
        scores[va_pos] = stage2_score_fn(model, x2[va_pos])
    return scores


def combine_predictions(pos_mask: np.ndarray, stage2_score: np.ndarray, th2: float) -> np.ndarray:
    final_pred = np.zeros(len(pos_mask), dtype=bool)
    decided = pos_mask & np.isfinite(stage2_score)
    final_pred[decided] = stage2_score[decided] >= th2
    final_pred[pos_mask & ~np.isfinite(stage2_score)] = True  # no stage-2 score -> keep stage-1 positive call
    return final_pred


def choose_stage2_threshold(y: np.ndarray, pos_mask: np.ndarray, stage2_score: np.ndarray,
                             margin: float = SENS_NONINF_MARGIN, z: float = NI_Z) -> float:
    # Selects the threshold with the best PPV among thresholds whose CI-based
    # non-inferiority test (noninferiority_test_sensitivity) actually passes -- not
    # just the point-estimate sensitivity drop -- so the criterion baked into model
    # selection is the same one reported/plotted as the final NI verdict.
    finite = pos_mask & np.isfinite(stage2_score)
    candidates = np.concatenate([[-np.inf], np.unique(stage2_score[finite])])
    best = None
    for th in candidates:
        pred = combine_predictions(pos_mask, stage2_score, th)
        tp, fp, fn, tn = baseline.confusion_counts(y, pred)
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        ni = noninferiority_test_sensitivity(pos_mask, pred, y, margin=margin, z=z)
        if np.isfinite(ppv) and ni["noninferior"]:
            if best is None or ppv > best[1]:
                best = (float(th), ppv)
    assert best is not None, "sentinel th=-inf candidate reproduces stage-1 exactly (sens_drop=0) and must always pass"
    return best[0]


def evaluate_combined(cohort: str, y: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn, tn = baseline.confusion_counts(y, pred)
    n = len(y)
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    print(f"[{cohort}] acc={acc:.3f} sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f} npv={npv:.3f} "
          f"n={n} tp={tp} fp={fp} fn={fn} tn={tn}")
    return {"cohort": cohort, "matrix": np.array([[tp, fn], [fp, tn]]),
            "acc": acc, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv}


def plot_confusion_matrix(ax: Axes, result: dict, title: str, baseline_res: dict | None = None,
                            mcnemar_res: dict | None = None, ni_res: dict | None = None) -> None:
    matrix = result["matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center",
                     fontsize=13, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    ax.set_title(title, fontsize=11, fontweight="bold")

    def fmt(name: str, key: str) -> str:
        val = result[key]
        if baseline_res is None:
            return f"{name}={val:.3f}"
        delta = val - baseline_res[key]
        return f"{name}={val:.3f} ({delta * 100:+.1f}%)"

    line1 = f"{fmt('Acc', 'acc')}  {fmt('Sens', 'sens')}  {fmt('Spec', 'spec')}"
    line2 = f"{fmt('PPV', 'ppv')}  {fmt('NPV', 'npv')}"
    xlabel = f"{line1}\n{line2}"
    if mcnemar_res is not None:
        def pfmt(p: float) -> str:
            return "<0.001" if p < 0.001 else f"{p:.3f}"
        line3 = (f"McNemar p (Sens)={pfmt(mcnemar_res['sens_p'])}  "
                 f"McNemar p (Spec)={pfmt(mcnemar_res['spec_p'])}")
        xlabel = f"{xlabel}\n{line3}"
    if ni_res is not None:
        verdict = "PASS" if ni_res["noninferior"] else "FAIL"
        line4 = (f"Sens NI (margin={ni_res['margin']:.2f}): drop={ni_res['sens_drop']:.3f}  "
                 f"97.5%CI upper={ni_res['ci_upper']:.3f}  -> {verdict}")
        xlabel = f"{xlabel}\n{line4}"
    ax.set_xlabel(xlabel, fontsize=9.5)


TABLE_HEADER_BG = "#1c1c1c"
TABLE_HEADER_FG = "#ffffff"
TABLE_HEADER_SUB = "#b9b8b3"
TABLE_BAND_BG = "#f6f6f4"
TABLE_GRID = "#d9d8d3"
TABLE_DIVIDER = "#2a2a2a"
TABLE_GOOD = "#1a7a4c"
TABLE_BAD = "#c0392b"
TABLE_NRI_BG = "#d9e8fb"
TABLE_NRI_FG = "#1553b6"
TABLE_TEXT = "#161616"
TABLE_MUTED = "#4d4c48"
TABLE_SUBTEXT = "#6b6a66"


def plot_clinical_vs_aec_table(rows: list[dict], out_path: Path, title: str) -> None:
    # reference_image.png-style summary table: per cohort, Clinical-only (stage-1) vs
    # AEC-assisted (stage-1+stage-2) sens/spec/acc with McNemar p-values and Net NRI
    # (= specificity flips that improved minus sensitivity flips that worsened; see
    # noninferiority_test_sensitivity / mcnemar_pvalue for where those flip counts
    # come from -- stage-2 only ever flips a stage-1-positive call to negative).
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [("sens", "Sensitivity"), ("spec", "Specificity"), ("acc", "Accuracy")]
    row_h, header_h, footer_h = 1.0, 1.7, 0.55
    n_metric_rows = len(rows) * len(metrics)
    total_h = header_h + n_metric_rows * row_h + footer_h

    col = {"cohort": (0.00, 0.15), "n": (0.15, 0.205), "event": (0.205, 0.26),
           "metric": (0.26, 0.40), "clin": (0.40, 0.62), "aec": (0.62, 0.90), "nri": (0.90, 1.00)}
    cx = lambda key: (col[key][0] + col[key][1]) / 2

    fig, ax = plt.subplots(figsize=(13.5, total_h * 0.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    header_bottom = total_h - header_h
    ax.add_patch(Rectangle((0, header_bottom), 1, header_h, facecolor=TABLE_HEADER_BG, edgecolor="none", zorder=1))
    header_main_y = header_bottom + header_h * 0.68
    header_sub_y = header_bottom + header_h * 0.28
    for key, label in [("cohort", "코호트"), ("n", "N"), ("event", "Event"), ("metric", "지표")]:
        ax.text(cx(key), header_bottom + header_h / 2, label, ha="center", va="center",
                color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("clin"), header_main_y, "Clinical only", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("clin"), header_sub_y, "sens / spec / acc", ha="center", va="center",
            color=TABLE_HEADER_SUB, fontsize=9.5)
    ax.text(cx("aec"), header_main_y, "AEC-assisted", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("aec"), header_sub_y, "sens / spec / acc (p)", ha="center", va="center",
            color=TABLE_HEADER_SUB, fontsize=9.5)
    ax.text(cx("nri"), header_bottom + header_h / 2, "Net\nNRI", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")

    def pfmt(p: float) -> str:
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    y_cursor = header_bottom
    for gi, r in enumerate(rows):
        block_top = y_cursor
        block_bottom = y_cursor - len(metrics) * row_h
        if gi % 2 == 0:
            ax.add_patch(Rectangle((0, block_bottom), 1, len(metrics) * row_h,
                                    facecolor=TABLE_BAND_BG, edgecolor="none", zorder=0))

        mid_y = (block_top + block_bottom) / 2
        ax.text(cx("cohort"), mid_y + 0.12, r["cohort"], ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=TABLE_TEXT)
        ax.text(cx("cohort"), mid_y - 0.22, f"AUC {r['auc']:.3f}", ha="center", va="center",
                fontsize=9.5, color=TABLE_GOOD)
        ax.text(cx("n"), mid_y, f"{r['n']}", ha="center", va="center", fontsize=12, color=TABLE_TEXT)
        ax.text(cx("event"), mid_y, f"{r['event']}", ha="center", va="center", fontsize=12, color=TABLE_TEXT)

        nri = r["net_nri"]
        box_w, box_h = 0.07, 0.9
        ax.add_patch(FancyBboxPatch((cx("nri") - box_w / 2, mid_y - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     linewidth=0, facecolor=TABLE_NRI_BG, zorder=2))
        ax.text(cx("nri"), mid_y, f"{nri:+d}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=TABLE_NRI_FG, zorder=3)

        for mi, (mkey, mlabel) in enumerate(metrics):
            row_top = block_top - mi * row_h
            row_bottom = row_top - row_h
            row_mid = (row_top + row_bottom) / 2

            ax.text(cx("metric"), row_mid, mlabel, ha="center", va="center", fontsize=11.5, color=TABLE_TEXT)

            clin_val, aec_val, p_val = r[f"{mkey}_clin"], r[f"{mkey}_aec"], r[f"{mkey}_p"]
            delta = aec_val - clin_val
            dcolor = TABLE_GOOD if delta >= 0 else TABLE_BAD

            ax.text(cx("clin"), row_mid, f"{clin_val:.3f}", ha="center", va="center",
                    fontsize=12, color=TABLE_MUTED)
            aec_x0, aec_x1 = col["aec"]
            ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.28, row_mid, f"{aec_val:.3f}",
                    ha="center", va="center", fontsize=12, color=TABLE_TEXT)
            ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.72, row_mid, f"({delta:+.3f}) {pfmt(p_val)}",
                    ha="center", va="center", fontsize=9.5, color=dcolor)

            if mi < len(metrics) - 1:
                ax.plot([col["metric"][0], 1], [row_bottom, row_bottom], color=TABLE_GRID,
                        linewidth=0.8, zorder=1)

        y_cursor = block_bottom
        ax.plot([0, 1], [block_bottom, block_bottom], color=TABLE_DIVIDER, linewidth=1.4, zorder=2)

    footnote = "* p < 0.05 (유의)    n.s. p ≥ 0.05 (비유의)    Net NRI: AEC 추가 시 순 재분류 개선 환자 수"
    ax.text(0.0, footer_h * 0.4, footnote, ha="left", va="center", fontsize=9, color=TABLE_SUBTEXT)

    fig.suptitle(title, x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=TABLE_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved clinical-vs-AEC table to {out_path}")


def with_accuracy(result: dict) -> dict:
    matrix = result["matrix"]
    tp, fn, fp, tn = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    result["acc"] = (tp + tn) / matrix.sum()
    return result


def pass_fail(baseline_res: dict, combined_res: dict, ni_res: dict) -> tuple[float, float, bool]:
    sens_delta = combined_res["sens"] - baseline_res["sens"]
    spec_delta = combined_res["spec"] - baseline_res["spec"]
    ok = ni_res["noninferior"] and (spec_delta > 0)
    return sens_delta, spec_delta, ok


def mcnemar_pvalue(pred_before: np.ndarray, pred_after: np.ndarray, subset_mask: np.ndarray) -> tuple[int, int, float]:
    # Paired McNemar test on the subset (actual positives for sensitivity, actual
    # negatives for specificity). By construction stage-2 only ever flips a stage-1
    # positive call to negative (never the reverse), so the table is one-directional:
    # b = positive->negative flips, c = negative->positive flips (always 0 here).
    before = pred_before[subset_mask].astype(bool)
    after = pred_after[subset_mask].astype(bool)
    b = int(np.sum(before & ~after))
    c = int(np.sum(~before & after))
    table = np.array([[np.sum(before & after), b], [c, np.sum(~before & ~after)]])
    result = mcnemar(table, exact=(b + c < 25), correction=True)
    return b, c, float(getattr(result, "pvalue"))


def _wilson_ci(count: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = count / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return center - half, center + half


def noninferiority_test_sensitivity(pred_before: np.ndarray, pred_after: np.ndarray, y: np.ndarray,
                                      margin: float = SENS_NONINF_MARGIN, z: float = NI_Z) -> dict:
    # Newcombe (1998) "Method 10" score-based CI for the difference between two paired
    # proportions, applied to sensitivity (subset = actual positives). This tests the
    # same acceptance criterion as pass_fail()'s point-estimate check
    # (sens_delta >= -margin) but accounts for sampling uncertainty: non-inferiority is
    # only declared if the upper confidence bound on the sensitivity DROP stays within
    # margin, not just the observed drop itself. By design c=0 here (stage-2 never
    # turns a stage-1 negative into a positive), same as in mcnemar_pvalue.
    subset = y == 1
    before = pred_before[subset].astype(bool)
    after = pred_after[subset].astype(bool)
    n = len(before)
    a = int(np.sum(before & after))
    b = int(np.sum(before & ~after))
    c = int(np.sum(~before & after))
    d = int(np.sum(~before & ~after))

    p1 = (a + b) / n  # sensitivity before (stage-1 only)
    p2 = (a + c) / n  # sensitivity after (stage-1+stage-2)
    drop = p1 - p2    # positive value = sensitivity fell

    l1, u1 = _wilson_ci(a + b, n, z)
    l2, u2 = _wilson_ci(a + c, n, z)

    denom = float(np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))
    phi = (a * d - b * c) / denom if denom > 0 else 0.0

    ci_lower = drop - np.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    ci_upper = drop + np.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)

    return {"n": n, "a": a, "b": b, "c": c, "d": d,
            "sens_before": p1, "sens_after": p2, "sens_drop": drop,
            "ci_lower": ci_lower, "ci_upper": ci_upper, "margin": margin,
            "noninferior": bool(ci_upper <= margin)}


def _pipeline_style_axes(ax: Axes) -> None:
    ax.set_facecolor(PIPE_SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PIPE_GRID)
    ax.tick_params(colors=PIPE_INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=PIPE_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(PIPE_INK_SECONDARY)
    ax.yaxis.label.set_color(PIPE_INK_SECONDARY)


def _pipeline_group_mean_ci(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy import stats
    mean = mat.mean(axis=0)
    ci = 1.96 * stats.sem(mat, axis=0)
    return mean, ci


def _pipeline_plot_group_curves(ax: Axes, x: np.ndarray, curves: np.ndarray, smi_group: np.ndarray,
                                 title: str, ylabel: str) -> None:
    for val, label, color in [("Low", "Low SMI (sarcopenia)", PIPE_COL_A), ("High", "Non-low SMI", PIPE_COL_B)]:
        mask = smi_group == val
        mean, ci = _pipeline_group_mean_ci(curves[mask])
        ax.plot(x, mean, color=color, linewidth=2, label=f"{label} (n={mask.sum()})")
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.0 if "residual" in ylabel.lower() else 1.0, color=PIPE_INK_MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(1, N_SLICES)
    ax.set_xlabel("Slice index (1-128)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=PIPE_INK_PRIMARY, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=PIPE_INK_SECONDARY, loc="best")
    _pipeline_style_axes(ax)


def plot_aec_preprocessing_pipeline_figure(curves_int: np.ndarray,
                                            x_int: np.ndarray, reg: LinearRegression,
                                            resid_int: np.ndarray, y_int: np.ndarray) -> None:
    # 4.2 AEC-128 곡선 전처리 파이프라인을 시각적으로 설명하는 그림.
    #
    #   Step 1 (정규화): 환자별 128-slice raw AEC를 환자 평균으로 나눠 스케일 차이를 제거.
    #   Step 2 (잔차화): 정규화 곡선을 표준화된 임상변수(나이/키/몸무게/성별)로 선형회귀 예측하고
    #                    실제값 - 예측값(잔차)만 남겨, 체질량(임상변수)에 의한 Simpson's paradox식
    #                    교란을 제거. 회귀는 내부 코호트에서만 학습.
    #   Step 3 (PCA):    잔차 128차원을 내부 코호트에서 PCA로 축소 (누적 설명분산 90%/95% 지점,
    #                     최대 10개 성분).
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    aec = pd.read_excel(INTERNAL_XLSX, sheet_name="aec_128", engine="openpyxl")
    curves_raw_all = aec[AEC_COLS].astype(float).to_numpy()

    # Low-SMI 임상 cutoff (성별 고정값, baseline.load_cohort()의 y 정의와 동일:
    # SMI = TAMA / Height[m]^2, Low-SMI(y=1)는 남성 SMI<45.4, 여성 SMI<34.4).
    smi_group = np.where(y_int == 1, "Low", "High")

    pca_probe = fit_residual_pca(resid_int, var_target=0.999, n_max=min(30, resid_int.shape[1]))
    cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
    pca90 = fit_residual_pca(resid_int, var_target=0.90)
    pca95 = fit_residual_pca(resid_int, var_target=0.95)

    x_idx = np.arange(1, N_SLICES + 1)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))

    # --- Panel A: raw vs patient-normalized curves for 4 example patients ---
    ax = axes[0, 0]
    rng = np.random.default_rng()
    sample_idx = rng.choice(len(curves_raw_all), size=4, replace=False)
    colors4 = [PIPE_COL_A, PIPE_COL_B, PIPE_COL_C, "#4a3aa7"]
    for i, c in zip(sample_idx, colors4):
        ax.plot(x_idx, curves_raw_all[i], color=c, linewidth=1.3, alpha=0.85)
    ax.set_xlim(1, N_SLICES)
    ax.set_xlabel("Slice index (1-128)")
    ax.set_ylabel("Raw AEC (mAs)")
    ax.set_title("Step 1a. Raw AEC-128 (환자 4명 예시)", color=PIPE_INK_PRIMARY, fontsize=11, fontweight="bold")
    _pipeline_style_axes(ax)

    ax = axes[0, 1]
    for i, c in zip(sample_idx, colors4):
        ax.plot(x_idx, curves_int[i], color=c, linewidth=1.3, alpha=0.85)
    ax.axhline(1.0, color=PIPE_INK_MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(1, N_SLICES)
    ax.set_xlabel("Slice index (1-128)")
    ax.set_ylabel("Patient-normalized AEC (raw / patient mean)")
    ax.set_title("Step 1b. 환자 평균 정규화 후\n(스케일 차이 제거, 형태만 남음)",
                  color=PIPE_INK_PRIMARY, fontsize=11, fontweight="bold")
    _pipeline_style_axes(ax)

    # --- Panel B: Low-SMI confound BEFORE residualization ---
    _pipeline_plot_group_curves(axes[0, 2], x_idx, curves_int, smi_group,
                                 "Step 2a. 정규화 곡선 (잔차화 전)\nLow-SMI에 의한 교란(Simpson's paradox) 존재",
                                 "Patient-normalized AEC")

    # --- Panel C: Low-SMI confound AFTER residualization ---
    _pipeline_plot_group_curves(axes[1, 0], x_idx, resid_int, smi_group,
                                 "Step 2b. 잔차화 후\n(나이/키/몸무게/성별로 설명되는 부분 제거)",
                                 "Residual AEC (actual - predicted)")

    # --- Panel D: cumulative explained variance / PCA component count ---
    ax = axes[1, 1]
    k = np.arange(1, len(cum_var) + 1)
    ax.plot(k, cum_var, color=PIPE_COL_A, linewidth=2, marker="o", markersize=3)
    ax.axhline(0.90, color=PIPE_COL_C, linewidth=1, linestyle="--", label=f"90% (k={pca90.n_components_})")
    ax.axhline(0.95, color=PIPE_COL_B, linewidth=1, linestyle="--", label=f"95% (k={pca95.n_components_})")
    ax.axvline(pca90.n_components_, color=PIPE_COL_C, linewidth=1, linestyle=":")
    ax.axvline(pca95.n_components_, color=PIPE_COL_B, linewidth=1, linestyle=":")
    ax.axvline(PCA_N_MAX, color=PIPE_INK_MUTED, linewidth=1, linestyle="-", alpha=0.5,
               label=f"n_max={PCA_N_MAX}")
    ax.set_xlim(1, len(cum_var))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PCA component count (k)")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Step 3a. 잔차 128차원 -> PCA 누적 설명분산",
                  color=PIPE_INK_PRIMARY, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=PIPE_INK_SECONDARY, loc="lower right")
    _pipeline_style_axes(ax)

    # --- Panel E: PCA component loadings (shape) -- all k components used by stage-2 ---
    ax = axes[1, 2]
    n_show = pca90.n_components_
    comp_colors = [PIPE_COL_A, PIPE_COL_B, PIPE_COL_C, "#4a3aa7", "#c0447a", "#5aa6a1"][:n_show]
    for comp_i in range(n_show):
        ax.plot(x_idx, pca90.components_[comp_i], color=comp_colors[comp_i], linewidth=1.8,
                label=f"PC{comp_i + 1} ({pca90.explained_variance_ratio_[comp_i]:.1%})")
    ax.axhline(0.0, color=PIPE_INK_MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(1, N_SLICES)
    ax.set_xlabel("Slice index (1-128)")
    ax.set_ylabel("PCA loading")
    ax.set_title(f"Step 3b. 전체 성분(loading) 형태 (90%-PCA, k={pca90.n_components_}개 모두 stage-2 feature로 사용)",
                  color=PIPE_INK_PRIMARY, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, labelcolor=PIPE_INK_SECONDARY, loc="best")
    _pipeline_style_axes(ax)

    fig.suptitle("4.2 AEC-128 곡선 전처리: 정규화 -> 잔차화(임상변수 교란 제거) -> PCA 차원축소",
                 fontsize=14, fontweight="bold", color=PIPE_INK_PRIMARY)
    fig.patch.set_facecolor(PIPE_SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    fig_path = OUTPUT_DIR / "aec_preprocessing_pipeline.png"
    fig.savefig(fig_path, dpi=220, facecolor=PIPE_SURFACE)
    plt.close(fig)
    print(f"Saved pipeline figure to {fig_path}")
    print(f"PCA @90% var: k={pca90.n_components_}, @95% var: k={pca95.n_components_}")


SWEEP_CONFIGS = [
    {"model_type": "logreg", "pca_var_target": 0.90, "use_stage1_score": False},
    {"model_type": "logreg", "pca_var_target": 0.90, "use_stage1_score": True},
    {"model_type": "logreg", "pca_var_target": 0.95, "use_stage1_score": True},
    {"model_type": "hgb", "pca_var_target": 0.90, "use_stage1_score": True},
    {"model_type": "hgb", "pca_var_target": 0.95, "use_stage1_score": True},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- internal: fit everything (clinical standardizer, stage-1, aec residualizer) ----------
    meta_int, y_int, curves_int = load_cohort_with_aec(INTERNAL_XLSX)
    x_raw_int = baseline.raw_clinical_matrix(meta_int)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw_int)
    x_int = baseline.apply_clinical_standardizer(x_raw_int, med, mu, sd)

    oof1 = baseline.oof_scores(x_int, y_int)
    th1 = baseline.threshold_for_sensitivity(y_int, oof1, TARGET_SENSITIVITY)
    baseline_int = with_accuracy(baseline.evaluate("internal / stage-1 only", y_int, oof1 >= th1, th1))
    pos_mask_int = oof1 >= th1

    reg = fit_aec_residualizer(x_int, curves_int)
    resid_int = apply_aec_residualizer(reg, x_int, curves_int)

    # ---------- model-selection sweep: internal OOF only, external is never touched here ----------
    sweep_rows = []
    sweep_state = {}
    for cfg in SWEEP_CONFIGS:
        pca = fit_residual_pca(resid_int, var_target=cfg["pca_var_target"])
        aec_pca_int = pca.transform(resid_int)
        stage1_feat = oof1 if cfg["use_stage1_score"] else None
        x2_int = stage2_feature_matrix(x_int, aec_pca_int, stage1_feat)

        stage2_oof = stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"])
        th2 = choose_stage2_threshold(y_int, pos_mask_int, stage2_oof)
        final_pred_int = combine_predictions(pos_mask_int, stage2_oof, th2)
        combined_int = evaluate_combined(
            f"internal sweep [{cfg['model_type']}, pca_var={cfg['pca_var_target']}, "
            f"stage1_feat={cfg['use_stage1_score']}]", y_int, final_pred_int
        )
        ni_sweep = noninferiority_test_sensitivity(pos_mask_int, final_pred_int, y_int)
        sens_delta, spec_delta, ok = pass_fail(baseline_int, combined_int, ni_sweep)
        key = (cfg["model_type"], cfg["pca_var_target"], cfg["use_stage1_score"])
        sweep_rows.append({**cfg, "pca_k": pca.n_components_, "sens_delta": sens_delta,
                            "spec_delta": spec_delta, "ppv": combined_int["ppv"],
                            "ni_ci_upper": ni_sweep["ci_upper"], "pass": ok})
        sweep_state[key] = {"pca": pca, "th2": th2, "sens_delta": sens_delta, "spec_delta": spec_delta,
                             "ok": ok, "cfg": cfg}

    sweep_df = pd.DataFrame(sweep_rows)
    print("\n=== internal OOF model-selection sweep ===")
    print(sweep_df.to_string(index=False))

    passing = [s for s in sweep_state.values() if s["ok"]]
    pool = passing if passing else list(sweep_state.values())
    best = max(pool, key=lambda s: s["spec_delta"])
    cfg, pca, th2 = best["cfg"], best["pca"], best["th2"]
    print(f"\nSelected config: {cfg} (internal spec_delta={best['spec_delta']:+.3f}, "
          f"sens_delta={best['sens_delta']:+.3f})")

    aec_pca_int = pca.transform(resid_int)
    stage1_feat_int = oof1 if cfg["use_stage1_score"] else None
    x2_int = stage2_feature_matrix(x_int, aec_pca_int, stage1_feat_int)
    stage2_oof = stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"])
    final_pred_int = combine_predictions(pos_mask_int, stage2_oof, th2)
    combined_int = evaluate_combined("internal / stage-1+stage-2 (OOF, selected config)", y_int, final_pred_int)

    ni_int = noninferiority_test_sensitivity(pos_mask_int, final_pred_int, y_int)
    sens_delta_int, spec_delta_int, ok_int = pass_fail(baseline_int, combined_int, ni_int)
    print(f"[internal] sens_delta={sens_delta_int:+.3f} spec_delta={spec_delta_int:+.3f} "
          f"-> {'PASS' if ok_int else 'FAIL'}")

    sens_b_int, sens_c_int, sens_p_int = mcnemar_pvalue(pos_mask_int, final_pred_int, y_int == 1)
    spec_b_int, spec_c_int, spec_p_int = mcnemar_pvalue(pos_mask_int, final_pred_int, y_int == 0)
    mcnemar_int = {"sens_p": sens_p_int, "spec_p": spec_p_int}
    print(f"[internal] McNemar sens: b={sens_b_int} c={sens_c_int} p={sens_p_int:.4g} | "
          f"spec: b={spec_b_int} c={spec_c_int} p={spec_p_int:.4g}")

    print(f"[internal] Non-inferiority (sens): drop={ni_int['sens_drop']:.3f} "
          f"97.5%CI upper={ni_int['ci_upper']:.3f} (margin={ni_int['margin']:.2f}) "
          f"-> {'NON-INFERIOR' if ni_int['noninferior'] else 'NOT NON-INFERIOR'}")

    # ---------- freeze final models on ALL internal data for external application ----------
    stage1_model = baseline.fit_baseline_model(x_int, y_int)
    score1_int_frozen = stage1_model.decision_function(x_int)  # for the frozen stage-2 feature, if used
    stage1_feat_frozen = score1_int_frozen if cfg["use_stage1_score"] else None
    x2_int_frozen = stage2_feature_matrix(x_int, aec_pca_int, stage1_feat_frozen)
    stage2_model = make_stage2_model(cfg["model_type"], SEED)
    stage2_model.fit(x2_int_frozen[pos_mask_int], y_int[pos_mask_int])

    # ---------- external: pure held-out test, frozen internal-fit parameters only ----------
    meta_ext, y_ext, curves_ext = load_cohort_with_aec(EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), med, mu, sd)

    score1_ext = stage1_model.decision_function(x_ext)
    baseline_ext = with_accuracy(baseline.evaluate("external / stage-1 only", y_ext, score1_ext >= th1, th1))
    pos_mask_ext = score1_ext >= th1

    resid_ext = apply_aec_residualizer(reg, x_ext, curves_ext)
    aec_pca_ext = pca.transform(resid_ext)
    stage1_feat_ext = score1_ext if cfg["use_stage1_score"] else None
    x2_ext = stage2_feature_matrix(x_ext, aec_pca_ext, stage1_feat_ext)
    stage2_score_ext = stage2_score_fn(stage2_model, x2_ext)
    final_pred_ext = combine_predictions(pos_mask_ext, stage2_score_ext, th2)
    combined_ext = evaluate_combined("external / stage-1+stage-2 (frozen)", y_ext, final_pred_ext)

    ni_ext = noninferiority_test_sensitivity(pos_mask_ext, final_pred_ext, y_ext)
    sens_delta_ext, spec_delta_ext, ok_ext = pass_fail(baseline_ext, combined_ext, ni_ext)
    print(f"[external] sens_delta={sens_delta_ext:+.3f} spec_delta={spec_delta_ext:+.3f} "
          f"-> {'PASS' if ok_ext else 'FAIL'}")

    sens_b_ext, sens_c_ext, sens_p_ext = mcnemar_pvalue(pos_mask_ext, final_pred_ext, y_ext == 1)
    spec_b_ext, spec_c_ext, spec_p_ext = mcnemar_pvalue(pos_mask_ext, final_pred_ext, y_ext == 0)
    mcnemar_ext = {"sens_p": sens_p_ext, "spec_p": spec_p_ext}
    print(f"[external] McNemar sens: b={sens_b_ext} c={sens_c_ext} p={sens_p_ext:.4g} | "
          f"spec: b={spec_b_ext} c={spec_c_ext} p={spec_p_ext:.4g}")

    print(f"[external] Non-inferiority (sens): drop={ni_ext['sens_drop']:.3f} "
          f"97.5%CI upper={ni_ext['ci_upper']:.3f} (margin={ni_ext['margin']:.2f}) "
          f"-> {'NON-INFERIOR' if ni_ext['noninferior'] else 'NOT NON-INFERIOR'}")

    # ---------- figures ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12.5))
    plot_confusion_matrix(axes[0, 0], baseline_int, "Internal: Stage-1 only (OOF)")
    plot_confusion_matrix(axes[0, 1], combined_int, "Internal: Stage-1+Stage-2 (OOF)",
                          baseline_res=baseline_int, mcnemar_res=mcnemar_int, ni_res=ni_int)
    plot_confusion_matrix(axes[1, 0], baseline_ext, "External: Stage-1 only (frozen)")
    plot_confusion_matrix(axes[1, 1], combined_ext, "External: Stage-1+Stage-2 (frozen)",
                          baseline_res=baseline_ext, mcnemar_res=mcnemar_ext, ni_res=ni_ext)
    fig.suptitle("Stage-2 reclassification of screen-positives (clinical + AEC residual PCA)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = OUTPUT_DIR / "stage1_vs_stage2_confusion_matrix.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"Saved confusion matrices to {fig_path}")

    # ---------- summary report ----------
    rows = []
    for cohort, base_res, comb_res, sd, spd, ok, mc, ni in [
        ("internal", baseline_int, combined_int, sens_delta_int, spec_delta_int, ok_int, mcnemar_int, ni_int),
        ("external", baseline_ext, combined_ext, sens_delta_ext, spec_delta_ext, ok_ext, mcnemar_ext, ni_ext),
    ]:
        rows.append({
            "cohort": cohort,
            "sens_stage1": base_res["sens"], "spec_stage1": base_res["spec"],
            "sens_combined": comb_res["sens"], "spec_combined": comb_res["spec"],
            "acc_combined": comb_res["acc"], "ppv_combined": comb_res["ppv"], "npv_combined": comb_res["npv"],
            "sens_delta": sd, "spec_delta": spd,
            "mcnemar_sens_p": mc["sens_p"], "mcnemar_spec_p": mc["spec_p"],
            "verdict": "PASS" if ok else "FAIL",
            "ni_sens_drop": ni["sens_drop"], "ni_ci_upper_97.5": ni["ci_upper"], "ni_margin": ni["margin"],
            "ni_verdict": "NON-INFERIOR" if ni["noninferior"] else "NOT NON-INFERIOR",
        })
    report = pd.DataFrame(rows)
    report_path = OUTPUT_DIR / "stage1_vs_stage2_summary.csv"
    report.to_csv(report_path, index=False)
    print(f"Saved summary to {report_path}")

    # ---------- clinical-only vs AEC-assisted summary table (reference_image.png style) ----------
    auc_int = roc_auc_score(y_int, oof1)
    auc_ext = roc_auc_score(y_ext, score1_ext)

    # Accuracy McNemar test: reuse mcnemar_pvalue on "was this patient classified
    # correctly" (rather than "was this patient called positive") across ALL patients,
    # so it captures both the sensitivity-losing and specificity-gaining flips.
    acc_b_int, acc_c_int, acc_p_int = mcnemar_pvalue(
        pos_mask_int == y_int.astype(bool), final_pred_int == y_int.astype(bool), np.ones_like(y_int, dtype=bool))
    acc_b_ext, acc_c_ext, acc_p_ext = mcnemar_pvalue(
        pos_mask_ext == y_ext.astype(bool), final_pred_ext == y_ext.astype(bool), np.ones_like(y_ext, dtype=bool))

    # Net NRI (count of patients): specificity flips that improved (FP->TN, actual
    # negatives) minus sensitivity flips that worsened (TP->FN, actual positives).
    # Stage-2 never flips a stage-1 negative to positive (c=0 by design), so this is
    # exactly the net number of patients whose classification improved.
    net_nri_int = spec_b_int - sens_b_int
    net_nri_ext = spec_b_ext - sens_b_ext

    table_rows = [
        {"cohort": "internal", "n": len(y_int), "event": int(y_int.sum()), "auc": auc_int,
         "sens_clin": baseline_int["sens"], "spec_clin": baseline_int["spec"], "acc_clin": baseline_int["acc"],
         "sens_aec": combined_int["sens"], "spec_aec": combined_int["spec"], "acc_aec": combined_int["acc"],
         "sens_p": mcnemar_int["sens_p"], "spec_p": mcnemar_int["spec_p"], "acc_p": acc_p_int,
         "net_nri": net_nri_int},
        {"cohort": "external", "n": len(y_ext), "event": int(y_ext.sum()), "auc": auc_ext,
         "sens_clin": baseline_ext["sens"], "spec_clin": baseline_ext["spec"], "acc_clin": baseline_ext["acc"],
         "sens_aec": combined_ext["sens"], "spec_aec": combined_ext["spec"], "acc_aec": combined_ext["acc"],
         "sens_p": mcnemar_ext["sens_p"], "spec_p": mcnemar_ext["spec_p"], "acc_p": acc_p_ext,
         "net_nri": net_nri_ext},
    ]
    table_path = OUTPUT_DIR / "clinical_vs_aec_assisted_table.png"
    plot_clinical_vs_aec_table(table_rows, table_path,
                                "clinical-only vs. AEC-assisted(PCA) 성능 비교 (Stage-1 vs Stage-1+Stage-2)")

    # ---------- 4.2 preprocessing pipeline explainer figure ----------
    plot_aec_preprocessing_pipeline_figure(curves_int, x_int, reg, resid_int, y_int)


if __name__ == "__main__":
    main()
