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
# weight/sex, fit on internal only). The stage-2 threshold is chosen to maximize
# PPV of the screen-positive group subject to a hard floor: global sensitivity
# must not drop more than 5pp below the stage-1-only baseline (see acceptance-
# criteria memory). Predicted-negative patients (score < th1) are left untouched
# by design -- the FN group is too small (n=12 internal) to support a reliable
# stage-2 correction, and touching TN/FN risks hurting specificity or sensitivity
# for no reliable gain.
#
# This variant extends stage2_aec_residual_reclassify.py's whole-curve PCA
# summary with alternative curve featurizations -- mean residual within contiguous
# equal-width slice bands (8/16/32), data-driven cluster bands, and 1D-radiomics
# texture features (first-order + GLCM/GLRLM adapted to a 1D curve) -- swept
# alongside PCA in the same internal-OOF model-selection sweep. See
# fit_curve_featurizer() for why the raw `aec_cropped` sheet was considered and
# rejected as a feature source.
#
# Run: python code/2_aec_residual_reclassify_bandfeat.py

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.cluster import AgglomerativeClustering
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "2_aec_residual_reclassify_bandfeat"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"

N_FOLDS = baseline.N_FOLDS
SEED = baseline.SEED
TARGET_SENSITIVITY = baseline.TARGET_SENSITIVITY
SENS_NONINF_MARGIN = 0.05  # acceptance-criteria memory: sensitivity may not drop more than 5pp
NI_ALPHA = 0.025  # one-sided; equivalent to checking the bound of a two-sided 95% CI
NI_Z = float(norm.ppf(1 - NI_ALPHA))
# Tighter margin used ONLY for internal model/threshold selection (choose_stage2_threshold,
# sweep pass/fail) -- not for the officially reported NI verdict below, which always uses
# SENS_NONINF_MARGIN. This wider curve-feature search space (band/cluster_band configs, not
# just whole-curve PCA) can find configs whose internal CI-based NI test passes right at the
# 0.05 edge but that don't leave enough headroom for the external cohort's own sampling noise.
# Selecting against a stricter internal margin buys that headroom without ever looking at the
# external cohort's outcome to pick among candidate configs.
SELECTION_MARGIN = 0.6 * SENS_NONINF_MARGIN

N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
PCA_VAR_TARGET = 0.90
PCA_N_MAX = 10
N_RADIOMICS_BINS = 8


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


# Ways to summarize the AEC residual curve into stage-2 features:
#   "pca"          -- global functional PCA (whole-curve modes of residual variation)
#   "band"         -- mean residual within n_bands contiguous, equal-width slice bands
#                     (localized signal -- e.g. a tail-region effect only shows up in
#                     the last band instead of being diluted across a whole-curve PCA
#                     mode).
#   "cluster_band" -- like "band", but the band boundaries are data-driven: adjacent
#                     slices are merged via connectivity-constrained (chain-graph)
#                     Ward clustering on their internal-cohort residual profile, so
#                     boundaries fall where the residual curve's behavior actually
#                     changes instead of at arbitrary equal-width cut points.
#   "combo"        -- band features concatenated with PCA features (param is a
#                     (n_bands, pca_var_target) tuple).
#   "radiomics1d"  -- classic radiomics feature families adapted to a 1D curve:
#                     first-order statistics (computed directly on the continuous
#                     residual) plus 1D GLCM and GLRLM texture features (computed on
#                     the residual curve discretized into n_bins quantile levels).
#                     Captures shape/heterogeneity of the residual curve (e.g. how
#                     "jagged" vs "smooth" it is) that neither whole-curve PCA modes
#                     nor per-band means represent directly. Bin edges are quantiles
#                     of the pooled internal-cohort residual distribution, frozen the
#                     same way PCA/regressor above are.
# All are fit/derived from internal-only data, consistent with fit_aec_residualizer.


def _first_order_features_1d(resid: np.ndarray) -> np.ndarray:
    # Standard radiomics first-order statistics -- computed directly on the
    # continuous residual curve, no discretization needed.
    mean = resid.mean(axis=1)
    std = resid.std(axis=1)
    q75 = np.percentile(resid, 75, axis=1)
    q25 = np.percentile(resid, 25, axis=1)
    iqr = q75 - q25
    mad = np.mean(np.abs(resid - mean[:, None]), axis=1)
    energy = np.sum(resid ** 2, axis=1)
    rms = np.sqrt(np.mean(resid ** 2, axis=1))
    centered = resid - mean[:, None]
    std_safe = np.where(std == 0, 1.0, std)
    skew = np.mean(centered ** 3, axis=1) / std_safe ** 3
    kurt = np.mean(centered ** 4, axis=1) / std_safe ** 4 - 3
    return np.column_stack([mean, std, iqr, mad, energy, rms, skew, kurt])


def fit_radiomics1d_binner(resid: np.ndarray, n_bins: int = N_RADIOMICS_BINS) -> np.ndarray:
    # Equal-probability (quantile) bin edges from the pooled internal-cohort
    # residual value distribution -- frozen here, applied unchanged to external data,
    # same fit-internal/apply-external discipline as fit_aec_residualizer.
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(resid.ravel(), quantiles)


def _discretize_for_texture(resid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(resid, edges).astype(int)  # levels in [0, n_bins-1]


def _glcm_1d_features(levels_row: np.ndarray, n_bins: int) -> np.ndarray:
    # 1D adaptation of GLCM: co-occurrence of adjacent-slice level pairs (symmetric,
    # offset=1), then the standard contrast/energy/correlation/homogeneity/entropy
    # descriptors on the normalized co-occurrence matrix.
    glcm = np.zeros((n_bins, n_bins))
    a, b = levels_row[:-1], levels_row[1:]
    np.add.at(glcm, (a, b), 1)
    np.add.at(glcm, (b, a), 1)
    total = glcm.sum()
    if total == 0:
        return np.zeros(5)
    p = glcm / total
    idx = np.arange(n_bins)
    i_idx, j_idx = np.meshgrid(idx, idx, indexing="ij")
    contrast = np.sum(p * (i_idx - j_idx) ** 2)
    energy = np.sum(p ** 2)
    mu_i = np.sum(i_idx * p)
    mu_j = np.sum(j_idx * p)
    sigma_i = np.sqrt(np.sum(p * (i_idx - mu_i) ** 2))
    sigma_j = np.sqrt(np.sum(p * (j_idx - mu_j) ** 2))
    correlation = (np.sum(p * (i_idx - mu_i) * (j_idx - mu_j)) / (sigma_i * sigma_j)
                   if sigma_i > 0 and sigma_j > 0 else 0.0)
    homogeneity = np.sum(p / (1 + np.abs(i_idx - j_idx)))
    nz = p > 0
    entropy = -np.sum(p[nz] * np.log2(p[nz]))
    return np.array([contrast, energy, correlation, homogeneity, entropy])


def _glrlm_1d_features(levels_row: np.ndarray, n_bins: int) -> np.ndarray:
    # 1D GLRLM: run-length encode the discretized curve, then the standard short/long
    # run emphasis, gray-level/run-length non-uniformity, and run-percentage descriptors.
    change = np.flatnonzero(np.diff(levels_row) != 0)
    run_start = np.concatenate([[0], change + 1])
    run_end = np.concatenate([change + 1, [len(levels_row)]])
    run_lengths = (run_end - run_start).astype(float)
    run_levels = levels_row[run_start]
    n_runs = len(run_lengths)
    n_points = len(levels_row)

    sre = np.mean(1.0 / run_lengths ** 2)
    lre = np.mean(run_lengths ** 2)
    level_run_counts = np.bincount(run_levels, minlength=n_bins).astype(float)
    gln = np.sum(level_run_counts ** 2) / n_runs
    length_run_counts = np.bincount(run_lengths.astype(int)).astype(float)
    rln = np.sum(length_run_counts ** 2) / n_runs
    rp = n_runs / n_points
    return np.array([sre, lre, gln, rln, rp])


def radiomics1d_features(resid: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    first_order = _first_order_features_1d(resid)
    levels = _discretize_for_texture(resid, edges)
    glcm_feats = np.array([_glcm_1d_features(row, n_bins) for row in levels])
    glrlm_feats = np.array([_glrlm_1d_features(row, n_bins) for row in levels])
    return np.column_stack([first_order, glcm_feats, glrlm_feats])
#
# Note: the raw `aec_cropped` sheet (pre-resampling) was considered but rejected as
# a feature source -- n_slices_cropped ranges 110-238 across patients (variable
# scan length), so its per-index columns are NOT slice-registered across patients
# the way aec_128 is. Using it column-wise would reintroduce a scan-length
# confound instead of removing one.
def fit_curve_featurizer(resid: np.ndarray, kind: str, param) -> dict:
    if kind == "pca":
        pca = fit_residual_pca(resid, var_target=param)
        return {"kind": "pca", "pca": pca, "k": pca.n_components_}
    if kind == "band":
        n_bands = int(param)
        if N_SLICES % n_bands != 0:
            raise ValueError(f"N_SLICES={N_SLICES} not divisible by n_bands={n_bands}")
        return {"kind": "band", "n_bands": n_bands, "k": n_bands}
    if kind == "cluster_band":
        n_bands = int(param)
        connectivity = np.zeros((N_SLICES, N_SLICES))
        idx = np.arange(N_SLICES - 1)
        connectivity[idx, idx + 1] = 1
        connectivity[idx + 1, idx] = 1
        clustering = AgglomerativeClustering(n_clusters=n_bands, connectivity=connectivity, linkage="ward")
        labels = clustering.fit_predict(resid.T)  # cluster slices by their residual profile across patients
        return {"kind": "cluster_band", "labels": labels, "k": n_bands}
    if kind == "combo":
        n_bands, pca_var_target = param
        band_state = fit_curve_featurizer(resid, "band", n_bands)
        pca_state = fit_curve_featurizer(resid, "pca", pca_var_target)
        return {"kind": "combo", "band": band_state, "pca": pca_state,
                 "k": band_state["k"] + pca_state["k"]}
    if kind == "radiomics1d":
        n_bins = int(param)
        edges = fit_radiomics1d_binner(resid, n_bins=n_bins)
        return {"kind": "radiomics1d", "edges": edges, "n_bins": n_bins, "k": 8 + 5 + 5}
    raise ValueError(kind)


def transform_curve_featurizer(state: dict, resid: np.ndarray) -> np.ndarray:
    if state["kind"] == "pca":
        return state["pca"].transform(resid)
    if state["kind"] == "band":
        n_bands = state["n_bands"]
        band_width = N_SLICES // n_bands
        return resid.reshape(resid.shape[0], n_bands, band_width).mean(axis=2)
    if state["kind"] == "cluster_band":
        labels = state["labels"]
        feats = np.column_stack([resid[:, labels == b].mean(axis=1) for b in range(state["k"])])
        return feats
    if state["kind"] == "combo":
        band_feat = transform_curve_featurizer(state["band"], resid)
        pca_feat = transform_curve_featurizer(state["pca"], resid)
        return np.column_stack([band_feat, pca_feat])
    if state["kind"] == "radiomics1d":
        return radiomics1d_features(resid, state["edges"], state["n_bins"])
    raise ValueError(state["kind"])


def stage2_feature_matrix(clin_std: np.ndarray, curve_feat: np.ndarray, stage1_score: np.ndarray | None = None) -> np.ndarray:
    cols = [clin_std, curve_feat]
    if stage1_score is not None:
        cols.append(stage1_score.reshape(-1, 1))
    return np.column_stack(cols)


MODEL_LABELS = {"logreg": "Logistic Regression", "hgb": "HGB"}


def make_stage2_model(model_type: str, seed: int, model_kwargs: dict | None = None):
    if model_type == "logreg":
        defaults = {"C": 1.0, "solver": "lbfgs", "max_iter": 5000}
        return LogisticRegression(random_state=seed, **{**defaults, **(model_kwargs or {})})
    if model_type == "hgb":
        defaults = {"max_depth": 3, "learning_rate": 0.06, "max_iter": 150}
        return HistGradientBoostingClassifier(random_state=seed, **{**defaults, **(model_kwargs or {})})
    raise ValueError(model_type)


def stage2_score_fn(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def stage2_oof_scores(x2: np.ndarray, y: np.ndarray, pos_mask: np.ndarray, model_type: str,
                        model_kwargs: dict | None = None) -> np.ndarray:
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
        model = make_stage2_model(model_type, SEED + fold_id, model_kwargs)
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


def plot_confusion_matrix(ax: Axes, result: dict, title: str) -> None:
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
TABLE_PASS_BG = "#dcefe1"
TABLE_FAIL_BG = "#f8dcd8"


def plot_clinical_vs_aec_table(rows: list[dict], out_path: Path, title: str) -> None:
    # reference_image.png-style summary table: per cohort, Clinical-only (stage-1) vs
    # AEC-assisted (stage-1+stage-2) sens/spec/acc with McNemar p-values and Net NRI
    # (= specificity flips that improved minus sensitivity flips that worsened; see
    # noninferiority_test_sensitivity / mcnemar_pvalue for where those flip counts
    # come from -- stage-2 only ever flips a stage-1-positive call to negative).
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [("sens", "Sensitivity"), ("spec", "Specificity"), ("acc", "Accuracy")]
    row_h, header_h, footer_h, ni_row_h = 1.0, 1.7, 0.55, 0.6
    block_h = len(metrics) * row_h + ni_row_h
    total_h = header_h + len(rows) * block_h + footer_h

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
        block_bottom = y_cursor - block_h
        if gi % 2 == 0:
            ax.add_patch(Rectangle((0, block_bottom), 1, block_h,
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

            ax.plot([col["metric"][0], 1], [row_bottom, row_bottom], color=TABLE_GRID,
                    linewidth=0.8, zorder=1)

        # NI test row: non-inferiority of sensitivity loss (noninferiority_test_sensitivity,
        # Wilson-score 97.5% CI vs SENS_NONINF_MARGIN), same PASS/FAIL already printed to
        # console as NON-INFERIOR/NOT NON-INFERIOR.
        ni_top = block_top - len(metrics) * row_h
        ni_bottom = ni_top - ni_row_h
        ni_mid = (ni_top + ni_bottom) / 2
        ni_pass = bool(r["ni_pass"])
        ni_color = TABLE_GOOD if ni_pass else TABLE_BAD

        ax.text(cx("metric"), ni_mid, "NI Test", ha="center", va="center",
                fontsize=11.5, color=TABLE_TEXT)
        ax.text(cx("clin"), ni_mid, f"margin ≤{r['ni_margin'] * 100:.1f}%p", ha="center", va="center",
                fontsize=10.5, color=TABLE_MUTED)
        aec_x0, aec_x1 = col["aec"]
        ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.5, ni_mid,
                f"sens loss 97.5% CI upper = {r['ni_ci_upper'] * 100:.2f}%p",
                ha="center", va="center", fontsize=10.5, color=TABLE_TEXT)

        badge_w, badge_h = 0.07, 0.44 if ni_pass else 0.52
        ax.add_patch(FancyBboxPatch((cx("nri") - badge_w / 2, ni_mid - badge_h / 2), badge_w, badge_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     linewidth=0, facecolor=TABLE_PASS_BG if ni_pass else TABLE_FAIL_BG, zorder=2))
        if ni_pass:
            ax.text(cx("nri"), ni_mid, "PASS", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ni_color, zorder=3)
        else:
            ax.text(cx("nri"), ni_mid + 0.13, f"{r['ni_ci_upper'] * 100:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=ni_color, zorder=3)
            ax.text(cx("nri"), ni_mid - 0.11, "FAIL", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ni_color, zorder=3)

        y_cursor = block_bottom
        ax.plot([0, 1], [block_bottom, block_bottom], color=TABLE_DIVIDER, linewidth=1.4, zorder=2)

    footnote = ("* p < 0.05 (유의)    n.s. p ≥ 0.05 (비유의)    Net NRI: AEC 추가 시 순 재분류 개선 환자 수    "
                "NI Test: 민감도 손실 97.5% CI 상한이 margin 이하이면 PASS (비열등성)")
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


SWEEP_CONFIGS = [
    {"model_type": "logreg", "curve_feat": ("pca", 0.90), "use_stage1_score": False},
    {"model_type": "logreg", "curve_feat": ("pca", 0.90), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("pca", 0.95), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("pca", 0.90), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("pca", 0.95), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("band", 4), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("band", 8), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("band", 16), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("band", 32), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("band", 8), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("band", 16), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("band", 32), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("cluster_band", 8), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("cluster_band", 16), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("cluster_band", 8), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("combo", (8, 0.90)), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("combo", (8, 0.90)), "use_stage1_score": True},
    {"model_type": "logreg", "curve_feat": ("radiomics1d", N_RADIOMICS_BINS), "use_stage1_score": True},
    {"model_type": "hgb", "curve_feat": ("radiomics1d", N_RADIOMICS_BINS), "use_stage1_score": True},
]

# Hyperparameter grid search. Earlier versions of this script only tuned hgb's
# learning_rate/max_iter for the single cluster_band:8 config that happened to win under
# default hyperparameters -- a biased search, since a curve_feat that looked mediocre with
# hgb's defaults could still hide a better internal spec_delta once tuned. So every
# curve_feat below (paired with use_stage1_score=True, the setting that beats
# use_stage1_score=False for every curve_feat in SWEEP_CONFIGS above) gets the same
# tuning treatment for BOTH model types, and model selection below picks the true internal
# OOF maximum across all of them rather than assuming untuned hgb/logreg is representative.
HGB_TUNE_GRID = [
    {"max_depth": d, "learning_rate": lr, "max_iter": mi}
    for d in (2, 3, 4)
    for lr in (0.03, 0.06, 0.1, 0.15)
    for mi in (100, 150, 300, 500)
]
LOGREG_TUNE_GRID = [{"C": c} for c in (0.1, 0.3, 1.0, 3.0, 10.0)]
MODEL_TUNE_GRIDS = {"hgb": HGB_TUNE_GRID, "logreg": LOGREG_TUNE_GRID}
# Grid points identical to a model's own default are skipped -- that combination is
# already covered by the untuned row in SWEEP_CONFIGS for the same curve_feat.
MODEL_DEFAULT_KWARGS = {"hgb": {"max_depth": 3, "learning_rate": 0.06, "max_iter": 150}, "logreg": {"C": 1.0}}
TUNE_CURVE_FEATS = [
    ("pca", 0.90), ("pca", 0.95),
    ("band", 4), ("band", 8), ("band", 16), ("band", 32),
    ("cluster_band", 8), ("cluster_band", 16),
    ("combo", (8, 0.90)),
    ("radiomics1d", N_RADIOMICS_BINS),
]


def build_tuned_sweep_configs() -> list[dict]:
    configs = []
    for feat_kind, feat_param in TUNE_CURVE_FEATS:
        for model_type, grid in MODEL_TUNE_GRIDS.items():
            default = MODEL_DEFAULT_KWARGS[model_type]
            for kwargs in grid:
                if kwargs == default:
                    continue
                configs.append({"model_type": model_type, "curve_feat": (feat_kind, feat_param),
                                 "use_stage1_score": True, "model_kwargs": kwargs})
    return configs


SWEEP_CONFIGS = SWEEP_CONFIGS + build_tuned_sweep_configs()


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
    for key, cfg in enumerate(SWEEP_CONFIGS):
        feat_kind, feat_param = cfg["curve_feat"]
        model_kwargs = cfg.get("model_kwargs")
        feat_state = fit_curve_featurizer(resid_int, feat_kind, feat_param)
        curve_feat_int = transform_curve_featurizer(feat_state, resid_int)
        stage1_feat = oof1 if cfg["use_stage1_score"] else None
        x2_int = stage2_feature_matrix(x_int, curve_feat_int, stage1_feat)

        stage2_oof = stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"], model_kwargs)
        th2 = choose_stage2_threshold(y_int, pos_mask_int, stage2_oof, margin=SELECTION_MARGIN)
        final_pred_int = combine_predictions(pos_mask_int, stage2_oof, th2)
        combined_int = evaluate_combined(
            f"internal sweep [{cfg['model_type']}, curve_feat={feat_kind}:{feat_param}, "
            f"stage1_feat={cfg['use_stage1_score']}, model_kwargs={model_kwargs}]", y_int, final_pred_int
        )
        ni_sweep = noninferiority_test_sensitivity(pos_mask_int, final_pred_int, y_int, margin=SELECTION_MARGIN)
        sens_delta, spec_delta, ok = pass_fail(baseline_int, combined_int, ni_sweep)
        sweep_rows.append({"model_type": cfg["model_type"], "curve_feat_kind": feat_kind,
                            "curve_feat_param": feat_param, "use_stage1_score": cfg["use_stage1_score"],
                            "model_kwargs": model_kwargs, "feat_k": feat_state["k"], "sens_delta": sens_delta,
                            "spec_delta": spec_delta, "ppv": combined_int["ppv"],
                            "ni_ci_upper": ni_sweep["ci_upper"], "pass": ok})
        sweep_state[key] = {"feat_state": feat_state, "th2": th2, "sens_delta": sens_delta,
                             "spec_delta": spec_delta, "ok": ok, "cfg": cfg}

    sweep_df = pd.DataFrame(sweep_rows)
    print("\n=== internal OOF model-selection sweep ===")
    print(sweep_df.to_string(index=False))

    sweep_ranked = sweep_df.sort_values(["pass", "spec_delta"], ascending=[False, False]).reset_index(drop=True)
    sweep_ranked.insert(0, "rank", np.arange(1, len(sweep_ranked) + 1))
    sweep_ranking_path = OUTPUT_DIR / "stage2_sweep_ranking.csv"
    sweep_ranked.to_csv(sweep_ranking_path, index=False)
    print(f"Saved sweep ranking to {sweep_ranking_path}")

    passing = [s for s in sweep_state.values() if s["ok"]]
    pool = passing if passing else list(sweep_state.values())
    best = max(pool, key=lambda s: s["spec_delta"])
    cfg, feat_state, th2 = best["cfg"], best["feat_state"], best["th2"]
    print(f"\nSelected config: {cfg} (internal spec_delta={best['spec_delta']:+.3f}, "
          f"sens_delta={best['sens_delta']:+.3f})")

    curve_feat_int = transform_curve_featurizer(feat_state, resid_int)
    stage1_feat_int = oof1 if cfg["use_stage1_score"] else None
    x2_int = stage2_feature_matrix(x_int, curve_feat_int, stage1_feat_int)
    stage2_oof = stage2_oof_scores(x2_int, y_int, pos_mask_int, cfg["model_type"], cfg.get("model_kwargs"))
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
    x2_int_frozen = stage2_feature_matrix(x_int, curve_feat_int, stage1_feat_frozen)
    stage2_model = make_stage2_model(cfg["model_type"], SEED, cfg.get("model_kwargs"))
    stage2_model.fit(x2_int_frozen[pos_mask_int], y_int[pos_mask_int])

    # ---------- external: pure held-out test, frozen internal-fit parameters only ----------
    meta_ext, y_ext, curves_ext = load_cohort_with_aec(EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), med, mu, sd)

    score1_ext = stage1_model.decision_function(x_ext)
    baseline_ext = with_accuracy(baseline.evaluate("external / stage-1 only", y_ext, score1_ext >= th1, th1))
    pos_mask_ext = score1_ext >= th1

    resid_ext = apply_aec_residualizer(reg, x_ext, curves_ext)
    curve_feat_ext = transform_curve_featurizer(feat_state, resid_ext)
    stage1_feat_ext = score1_ext if cfg["use_stage1_score"] else None
    x2_ext = stage2_feature_matrix(x_ext, curve_feat_ext, stage1_feat_ext)
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
    plot_confusion_matrix(axes[0, 1], combined_int, "Internal: Stage-1+Stage-2 (OOF)")
    plot_confusion_matrix(axes[1, 0], baseline_ext, "External: Stage-1 only (frozen)")
    plot_confusion_matrix(axes[1, 1], combined_ext, "External: Stage-1+Stage-2 (frozen)")
    fig.suptitle(f"Stage-2 reclassification of screen-positives "
                 f"(clinical + AEC residual, {MODEL_LABELS[cfg['model_type']]}, "
                 f"{cfg['curve_feat'][0]}:{cfg['curve_feat'][1]})",
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
            "model_type": MODEL_LABELS[cfg["model_type"]],
            "curve_feat": f"{cfg['curve_feat'][0]}:{cfg['curve_feat'][1]}",
            "model_kwargs": cfg.get("model_kwargs"),
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
         "net_nri": net_nri_int,
         "ni_margin": ni_int["margin"], "ni_ci_upper": ni_int["ci_upper"], "ni_pass": ni_int["noninferior"]},
        {"cohort": "external", "n": len(y_ext), "event": int(y_ext.sum()), "auc": auc_ext,
         "sens_clin": baseline_ext["sens"], "spec_clin": baseline_ext["spec"], "acc_clin": baseline_ext["acc"],
         "sens_aec": combined_ext["sens"], "spec_aec": combined_ext["spec"], "acc_aec": combined_ext["acc"],
         "sens_p": mcnemar_ext["sens_p"], "spec_p": mcnemar_ext["spec_p"], "acc_p": acc_p_ext,
         "net_nri": net_nri_ext,
         "ni_margin": ni_ext["margin"], "ni_ci_upper": ni_ext["ci_upper"], "ni_pass": ni_ext["noninferior"]},
    ]
    table_path = OUTPUT_DIR / "clinical_vs_aec_assisted_table.png"
    model_label = MODEL_LABELS[cfg["model_type"]]
    plot_clinical_vs_aec_table(
        table_rows, table_path,
        f"clinical-only vs. AEC-assisted ({model_label}, {cfg['curve_feat'][0]}:{cfg['curve_feat'][1]})"
        f"성능 비교 (Stage-1 vs Stage-1+Stage-2)")


if __name__ == "__main__":
    main()
