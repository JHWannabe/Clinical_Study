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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinic-only_baseline"

INTERNAL_XLSX = DATA_DIR / "g1090.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sdata.xlsx"
TARGET_SENSITIVITY = 0.90
N_FOLDS = 5
SEED = 20260709


def load_cohort(xlsx_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return meta, y


def load_internal(xlsx_path: Path = INTERNAL_XLSX) -> tuple[pd.DataFrame, np.ndarray]:
    return load_cohort(xlsx_path)


def load_external(xlsx_path: Path = EXTERNAL_XLSX) -> tuple[pd.DataFrame, np.ndarray]:
    return load_cohort(xlsx_path)


def raw_clinical_matrix(meta: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float)
    height = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float)
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([age, height, weight, sex_m])


def fit_clinical_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
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
        oof[va_idx] = model.decision_function(x[va_idx])
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
    ax.set_title(f"{label}\nThreshold @ Sensitivity>={TARGET_SENSITIVITY:.0%} (th={result['th']:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Sensitivity={result['sens']:.3f}  Specificity={result['spec']:.3f}  PPV={result['ppv']:.3f}  NPV={result['npv']:.3f}", fontsize=10)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal: fit standardizer + threshold via 5-fold OOF on internal only ---
    meta_int, y_int = load_internal()
    x_raw_int = raw_clinical_matrix(meta_int)
    med, mu, sd = fit_clinical_standardizer(x_raw_int)
    x_int = apply_clinical_standardizer(x_raw_int, med, mu, sd)

    oof = oof_scores(x_int, y_int)
    th = threshold_for_sensitivity(y_int, oof, TARGET_SENSITIVITY)
    result_int = evaluate("internal", y_int, oof >= th, th)

    # --- external: pure test set, scored by the model trained on all of internal ---
    model = fit_baseline_model(x_int, y_int)
    meta_ext, y_ext = load_external()
    x_ext = apply_clinical_standardizer(raw_clinical_matrix(meta_ext), med, mu, sd)
    score_ext = model.decision_function(x_ext)
    result_ext = evaluate("external", y_ext, score_ext >= th, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Clinical-only Logistic Regression", fontsize=13, fontweight="bold")
    plot_confusion_matrix(axes[0], result_int)
    plot_confusion_matrix(axes[1], result_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUTPUT_DIR / "clinical_only_confusion_matrix_sens90.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
