from __future__ import annotations

# aec_cnn_vatsat_disease_compare.py의 1D CNN(ConcatFusionModel)에 대해 learning rate(1e-4/5e-4/1e-3/5e-3) x
# epoch 상한 확대(300, patience 30) x SMOTE 유무 스윕을 수행(2026-09-03, 사용자 요청). 모델선택은 internal
# CV로만 해야 하므로([[feedback_internal_external_validation_discipline]]) external은 건드리지 않고 internal
# 5-fold OOF AUC만으로 비교한다. 스크리닝 단계라 ENSEMBLE_SIZE=1(단일 시드)로 실행 비용을 낮춤.
# SMOTE는 각 fold의 (train/val 분리 후) train subset에만 적용해 validation/타 fold/external로의 leakage를
# 막는다. curve는 patient-wise z-score([[feedback_aec_preprocessing_methods]]) 이후 값 위에서 보간되며,
# SMOTE 적용 시 클래스가 balanced되므로 pos_weight는 1로 되돌린다(가중치 이중 보정 방지).

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from aec_cnn_vatsat_disease_compare import (
    AEC_COLS,
    DEVICE,
    FEATURES,
    MIN_POSITIVES,
    PROJECT_ROOT,
    SEED,
    VAL_FRACTION,
    ConcatFusionModel,
    clinical_matrix,
    load_cohort,
    n_splits_for,
    patient_zscore,
    predict_model,
    valid_rows,
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = PROJECT_ROOT / "data"
INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "lr_epoch_sweep"

LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3]
SMOTE_GRID = [False, True]
SWEEP_EPOCHS = 300
SWEEP_PATIENCE = 30
SWEEP_ENSEMBLE_SIZE = 1
BATCH_SIZE = 64
SMOTE_K_NEIGHBORS_MAX = 5


# fold의 train subset(curve+clinic 결합 벡터)에만 SMOTE 오버샘플링 적용. 소수 클래스가 너무 적어
# k_neighbors를 못 채우면(<2) SMOTE를 건너뛰고 원본을 그대로 반환한다
def apply_smote(curve: np.ndarray, clinic: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pos = int(y.sum())
    if n_pos < 2:
        return curve, clinic, y
    k = min(SMOTE_K_NEIGHBORS_MAX, n_pos - 1)
    x = np.column_stack([curve, clinic])
    x_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(x, y)
    return x_res[:, :curve.shape[1]], x_res[:, curve.shape[1]:], y_res


def train_model_config(curve: np.ndarray, clinic: np.ndarray, y: np.ndarray, seed: int,
                        lr: float, epochs: int, patience: int, use_smote: bool) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y)), test_size=VAL_FRACTION, random_state=seed, stratify=y)

    curve_tr, clinic_tr, y_tr = curve[tr_idx], clinic[tr_idx], y[tr_idx]
    if use_smote:
        curve_tr, clinic_tr, y_tr = apply_smote(curve_tr, clinic_tr, y_tr, seed)
        pos_weight_val = 1.0
    else:
        n_pos = float(y_tr.sum())
        n_neg = float(len(y_tr)) - n_pos
        pos_weight_val = n_neg / max(n_pos, 1.0)

    model = ConcatFusionModel(n_clinic=clinic.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([pos_weight_val], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve_t = torch.tensor(curve_tr, dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic_tr, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y[val_idx]
    curve_std = curve_t.std()

    n = curve_t.shape[0]
    best_auc, best_state, epochs_no_improve = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch_curve = curve_t[idx] + torch.randn_like(curve_t[idx]) * curve_std * 0.05
            opt.zero_grad()
            loss = bce(model(batch_curve, clinic_t[idx]), y_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(curve_val_t, clinic_val_t)).cpu().numpy().ravel()
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else float("nan")
        if val_auc > best_auc:
            best_auc, best_state, epochs_no_improve = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def internal_oof_for_config(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray, cv: StratifiedKFold,
                             lr: float, epochs: int, patience: int, ensemble_size: int,
                             use_smote: bool) -> np.ndarray:
    curve = patient_zscore(curve_raw)
    oof = np.full(len(y), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(curve, y)):
        seed_preds = []
        for s in range(ensemble_size):
            model = train_model_config(curve[tr_idx], clinic[tr_idx], y[tr_idx],
                                        seed=SEED + fold_i * 10 + s, lr=lr, epochs=epochs, patience=patience,
                                        use_smote=use_smote)
            seed_preds.append(predict_model(model, curve[va_idx], clinic[va_idx]))
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return oof


def plot_lr_sweep(summary: pd.DataFrame, out_path: Path) -> None:
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, use_smote in zip(axes, SMOTE_GRID):
        sub = summary[summary["use_smote"] == use_smote]
        for feat in features:
            rows = sub[sub["feature"] == feat].sort_values("lr")
            ax.plot(rows["lr"], rows["auc"], marker="o", linewidth=2, markersize=9, label=feat)
        ax.set_xscale("log")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Learning rate", fontsize=24)
        ax.set_title("SMOTE" if use_smote else "No SMOTE (pos_weight)", fontsize=20, fontweight="bold", color="#161616")
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Internal 5-fold OOF AUC", fontsize=24)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(features), bbox_to_anchor=(0.5, -0.05),
               fontsize=16, frameon=False)
    fig.suptitle(f"1D CNN LR x SMOTE sweep (epochs<={SWEEP_EPOCHS}, patience={SWEEP_PATIENCE})",
                 fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LR x SMOTE sweep plot to {out_path}")


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE}")
    print(f"[스윕 설정] lr_grid={LR_GRID} smote_grid={SMOTE_GRID} epochs<={SWEEP_EPOCHS} patience={SWEEP_PATIENCE} "
          f"ensemble_size={SWEEP_ENSEMBLE_SIZE} (internal-only screening)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int = load_cohort(INTERNAL_XLSX)
    meta_int = meta_int[valid_rows(meta_int)].reset_index(drop=True)
    print(f"Cohort: internal n={len(meta_int)}")

    summary_rows = []
    for feat in FEATURES:
        y_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        mask_val = np.isfinite(y_all)
        n_pos, n_neg = int(y_all[mask_val].sum()), int(mask_val.sum() - y_all[mask_val].sum())
        if min(n_pos, n_neg) < MIN_POSITIVES:
            print(f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만")
            continue

        meta_m = meta_int.loc[mask_val].reset_index(drop=True)
        y = y_all[mask_val].astype(int)
        cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)

        curve_raw = meta_m[AEC_COLS].astype(float).to_numpy()
        clinic, _ = clinical_matrix(meta_m, None)

        for use_smote in SMOTE_GRID:
            for lr in LR_GRID:
                tag = "SMOTE" if use_smote else "pos_weight"
                print(f"[{feat}/{tag}] lr={lr:g} internal 5-fold OOF 학습 시작 (n={len(y)}, n_pos={y.sum()})")
                oof = internal_oof_for_config(curve_raw, clinic, y, cv, lr=lr, epochs=SWEEP_EPOCHS,
                                               patience=SWEEP_PATIENCE, ensemble_size=SWEEP_ENSEMBLE_SIZE,
                                               use_smote=use_smote)
                auc = float(roc_auc_score(y, oof))
                summary_rows.append({"feature": feat, "use_smote": use_smote, "lr": lr, "epochs_cap": SWEEP_EPOCHS,
                                      "patience": SWEEP_PATIENCE, "ensemble_size": SWEEP_ENSEMBLE_SIZE,
                                      "n": int(len(y)), "n_pos": int(y.sum()), "auc": auc})
                print(f"[{feat}/{tag}] lr={lr:g} internal AUC={auc:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "lr_epoch_sweep_summary.csv", index=False)
    print(f"Saved sweep summary to {OUTPUT_DIR / 'lr_epoch_sweep_summary.csv'}")

    if not summary.empty:
        plot_lr_sweep(summary, OUTPUT_DIR / "lr_sweep_auc.png")
        best = summary.loc[summary.groupby("feature")["auc"].idxmax()]
        print("\n[feature별 최고 internal AUC의 (lr, smote)]")
        print(best[["feature", "use_smote", "lr", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
