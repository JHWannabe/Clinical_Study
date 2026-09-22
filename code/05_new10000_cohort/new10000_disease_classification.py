"""사용자 요청(2026-09-18): "data/new10000_final_dataset.xlsx만 사용해서 DM, HTN 이외에 DL,
Osteoporosis, MI, Stroke에 대한 분류 성능도 확인" - 다른 코호트(gangnam/sinchon) 없이 new10000
metadata 단독으로 clinic4(age/height/weight/sex) 로지스틱 회귀 분류 성능만 확인.
후속(2026-09-18): "internal처럼 5-fold OOF로 성능 보여줘도 되고" -> 8:2 단일 split 대신 5-fold
StratifiedKFold cross_val_predict OOF 채택(0910 스크립트의 internal 평가 방식과 동일, MI처럼
양성이 84명뿐인 질환에서 단일 test split보다 안정적).
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_XLSX = PROJECT_ROOT / "data" / "new10000_final_dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0918" / "new10000_disease_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASES = ["DM", "HTN", "DL", "Osteoporosis", "MI", "Stroke"]
CLINICAL_COLS = ["PatientAge", "Height", "Weight"]
LOGREG_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 5000}
SEED = 20260709
N_FOLDS = 5


def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
             "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
             "accuracy": (tp + tn) / len(y)}


def run_disease(disease: str, meta: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    mask = meta[CLINICAL_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    m = meta.loc[mask].reset_index(drop=True)
    y = m[disease].astype(int).to_numpy()

    x_rest = StandardScaler().fit_transform(m[CLINICAL_COLS].to_numpy(dtype=float))
    sex_m = (m["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    x = np.column_stack([sex_m, x_rest])

    cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(LogisticRegression(**LOGREG_PARAMS), x, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, oof_proba)
    thr = youden_threshold(y, oof_proba)
    cstats = classification_stats(y, oof_proba, thr)
    n_pos = int(y.sum())
    row = {"disease": disease, "n": len(y), "n_pos": n_pos, "prevalence": n_pos / len(y),
           "auc": auc, "threshold": thr, **cstats}
    print(f"[{disease}] n={len(y)} n_pos={n_pos} ({n_pos/len(y):.1%}) AUC={auc:.3f} "
          f"sens={cstats['sensitivity']:.3f} spec={cstats['specificity']:.3f} acc={cstats['accuracy']:.3f}")
    return row, y, oof_proba


def plot_roc_grid(curves: dict[str, tuple[np.ndarray, np.ndarray]], aucs: dict[str, float], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, disease in zip(axes.flat, DISEASES):
        y, score = curves[disease]
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color="#2c7fb8", linewidth=1.8, label=f"AUC={aucs[disease]:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(disease, fontweight="bold")
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("new10000 clinic4(age/height/weight/sex) — 5-fold OOF ROC", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    meta = pd.read_excel(NEW_XLSX, sheet_name="metadata")
    meta = meta[meta["PatientSex"].astype(str).str.upper().isin(["M", "F"])].reset_index(drop=True)
    print(f"new10000 metadata: n={len(meta)}")

    rows, curves, aucs = [], {}, {}
    for disease in DISEASES:
        row, y, oof_proba = run_disease(disease, meta)
        rows.append(row)
        curves[disease] = (y, oof_proba)
        aucs[disease] = row["auc"]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "new10000_disease_classification_summary.csv", index=False)
    plot_roc_grid(curves, aucs, OUTPUT_DIR / "new10000_disease_classification_roc.png")


if __name__ == "__main__":
    main()
