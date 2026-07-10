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
# Run: python code/stage2_aec_residual_reclassify.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_aec_residual_reclassify"

INTERNAL_XLSX = DATA_DIR / "g1090.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sdata.xlsx"

N_FOLDS = baseline.N_FOLDS
SEED = baseline.SEED
TARGET_SENSITIVITY = baseline.TARGET_SENSITIVITY
SENS_NONINF_MARGIN = 0.05  # acceptance-criteria memory: sensitivity may not drop more than 5pp

N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
PCA_VAR_TARGET = 0.90
PCA_N_MAX = 10


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


def fit_residual_pca(resid: np.ndarray, var_target: float = PCA_VAR_TARGET, n_max: int = PCA_N_MAX) -> PCA:
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


def choose_stage2_threshold(y: np.ndarray, pos_mask: np.ndarray, stage2_score: np.ndarray, baseline_sens: float) -> float:
    finite = pos_mask & np.isfinite(stage2_score)
    candidates = np.concatenate([[-np.inf], np.unique(stage2_score[finite])])
    best = None
    for th in candidates:
        pred = combine_predictions(pos_mask, stage2_score, th)
        tp, fp, fn, tn = baseline.confusion_counts(y, pred)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        if np.isfinite(sens) and np.isfinite(ppv) and sens >= baseline_sens - SENS_NONINF_MARGIN:
            if best is None or ppv > best[1]:
                best = (float(th), ppv)
    assert best is not None, "sentinel th=-inf candidate reproduces stage-1 exactly and must always pass"
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
                            mcnemar_res: dict | None = None) -> None:
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
    ax.set_xlabel(xlabel, fontsize=9.5)


def with_accuracy(result: dict) -> dict:
    matrix = result["matrix"]
    tp, fn, fp, tn = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    result["acc"] = (tp + tn) / matrix.sum()
    return result


def pass_fail(baseline_res: dict, combined_res: dict) -> tuple[float, float, bool]:
    sens_delta = combined_res["sens"] - baseline_res["sens"]
    spec_delta = combined_res["spec"] - baseline_res["spec"]
    ok = (sens_delta >= -SENS_NONINF_MARGIN) and (spec_delta > 0)
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
        th2 = choose_stage2_threshold(y_int, pos_mask_int, stage2_oof, baseline_int["sens"])
        final_pred_int = combine_predictions(pos_mask_int, stage2_oof, th2)
        combined_int = evaluate_combined(
            f"internal sweep [{cfg['model_type']}, pca_var={cfg['pca_var_target']}, "
            f"stage1_feat={cfg['use_stage1_score']}]", y_int, final_pred_int
        )
        sens_delta, spec_delta, ok = pass_fail(baseline_int, combined_int)
        key = (cfg["model_type"], cfg["pca_var_target"], cfg["use_stage1_score"])
        sweep_rows.append({**cfg, "pca_k": pca.n_components_, "sens_delta": sens_delta,
                            "spec_delta": spec_delta, "ppv": combined_int["ppv"], "pass": ok})
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
    sens_delta_int, spec_delta_int, ok_int = pass_fail(baseline_int, combined_int)
    print(f"[internal] sens_delta={sens_delta_int:+.3f} spec_delta={spec_delta_int:+.3f} "
          f"-> {'PASS' if ok_int else 'FAIL'}")

    sens_b_int, sens_c_int, sens_p_int = mcnemar_pvalue(pos_mask_int, final_pred_int, y_int == 1)
    spec_b_int, spec_c_int, spec_p_int = mcnemar_pvalue(pos_mask_int, final_pred_int, y_int == 0)
    mcnemar_int = {"sens_p": sens_p_int, "spec_p": spec_p_int}
    print(f"[internal] McNemar sens: b={sens_b_int} c={sens_c_int} p={sens_p_int:.4g} | "
          f"spec: b={spec_b_int} c={spec_c_int} p={spec_p_int:.4g}")

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

    sens_delta_ext, spec_delta_ext, ok_ext = pass_fail(baseline_ext, combined_ext)
    print(f"[external] sens_delta={sens_delta_ext:+.3f} spec_delta={spec_delta_ext:+.3f} "
          f"-> {'PASS' if ok_ext else 'FAIL'}")

    sens_b_ext, sens_c_ext, sens_p_ext = mcnemar_pvalue(pos_mask_ext, final_pred_ext, y_ext == 1)
    spec_b_ext, spec_c_ext, spec_p_ext = mcnemar_pvalue(pos_mask_ext, final_pred_ext, y_ext == 0)
    mcnemar_ext = {"sens_p": sens_p_ext, "spec_p": spec_p_ext}
    print(f"[external] McNemar sens: b={sens_b_ext} c={sens_c_ext} p={sens_p_ext:.4g} | "
          f"spec: b={spec_b_ext} c={spec_c_ext} p={spec_p_ext:.4g}")

    # ---------- figures ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.5))
    plot_confusion_matrix(axes[0, 0], baseline_int, "Internal: Stage-1 only (OOF)")
    plot_confusion_matrix(axes[0, 1], combined_int, "Internal: Stage-1+Stage-2 (OOF)",
                          baseline_res=baseline_int, mcnemar_res=mcnemar_int)
    plot_confusion_matrix(axes[1, 0], baseline_ext, "External: Stage-1 only (frozen)")
    plot_confusion_matrix(axes[1, 1], combined_ext, "External: Stage-1+Stage-2 (frozen)",
                          baseline_res=baseline_ext, mcnemar_res=mcnemar_ext)
    fig.suptitle("Stage-2 reclassification of screen-positives (clinical + AEC residual PCA)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = OUTPUT_DIR / "stage1_vs_stage2_confusion_matrix.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"Saved confusion matrices to {fig_path}")

    # ---------- summary report ----------
    rows = []
    for cohort, base_res, comb_res, sd, spd, ok, mc in [
        ("internal", baseline_int, combined_int, sens_delta_int, spec_delta_int, ok_int, mcnemar_int),
        ("external", baseline_ext, combined_ext, sens_delta_ext, spec_delta_ext, ok_ext, mcnemar_ext),
    ]:
        rows.append({
            "cohort": cohort,
            "sens_stage1": base_res["sens"], "spec_stage1": base_res["spec"],
            "sens_combined": comb_res["sens"], "spec_combined": comb_res["spec"],
            "acc_combined": comb_res["acc"], "ppv_combined": comb_res["ppv"], "npv_combined": comb_res["npv"],
            "sens_delta": sd, "spec_delta": spd,
            "mcnemar_sens_p": mc["sens_p"], "mcnemar_spec_p": mc["spec_p"],
            "verdict": "PASS" if ok else "FAIL",
        })
    report = pd.DataFrame(rows)
    report_path = OUTPUT_DIR / "stage1_vs_stage2_summary.csv"
    report.to_csv(report_path, index=False)
    print(f"Saved summary to {report_path}")


if __name__ == "__main__":
    main()
