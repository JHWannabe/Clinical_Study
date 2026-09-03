from __future__ import annotations

# 1D CNN 학습에 ReduceLROnPlateau lr scheduler를 추가했을 때 internal 5-fold OOF AUC가 개선되는지
# 확인하는 ablation (2026-09-02, 사용자 질문: "1d cnn의 scheduler를 사용해서 lr을 조정하면 조금더
# 개선되지 않을까?"). feedback_internal_external_validation_discipline에 따라 internal CV만
# 비교하고 external은 건드리지 않는다. aec_cnn_vatsat_disease_compare.py의 데이터 로딩/모델 구조를
# 그대로 재사용하고, train_model만 scheduler 유무 두 버전으로 나눠 같은 fold/시드로 학습해 DeLong test로
# 비교한다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from aec_cnn_vatsat_disease_compare import (
    AEC_COLS, DEVICE, EARLY_STOP_PATIENCE, ENSEMBLE_SIZE, EPOCHS, FEATURES, INTERNAL_XLSX,
    MIN_POSITIVES, SEED, VAL_FRACTION, ConcatFusionModel, clinical_matrix, delong_paired_auc_test,
    load_cohort, n_splits_for, patient_zscore, predict_model, valid_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "lr_scheduler"


# aec_cnn_vatsat_disease_compare.train_model과 동일하되 ReduceLROnPlateau(val AUC 기준) 적용 여부만 분기
def train_model_variant(curve: np.ndarray, clinic: np.ndarray, y: np.ndarray, seed: int,
                         use_scheduler: bool) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y)), test_size=VAL_FRACTION, random_state=seed, stratify=y)

    model = ConcatFusionModel(n_clinic=clinic.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-5)
        if use_scheduler else None
    )
    n_pos = float(y[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve_t = torch.tensor(curve[tr_idx], dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y[val_idx]
    curve_std = curve_t.std()

    n = curve_t.shape[0]
    batch_size = 64
    best_auc, best_state, epochs_no_improve = -np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_curve = curve_t[idx] + torch.randn_like(curve_t[idx]) * curve_std * 0.05
            opt.zero_grad()
            loss = bce(model(batch_curve, clinic_t[idx]), y_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(curve_val_t, clinic_val_t)).cpu().numpy().ravel()
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else float("nan")
        if scheduler is not None and not np.isnan(val_auc):
            scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc, best_state, epochs_no_improve = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def cnn_internal_oof_variant(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray,
                              cv: StratifiedKFold, use_scheduler: bool) -> np.ndarray:
    curve = patient_zscore(curve_raw)
    oof = np.full(len(y), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(curve, y)):
        seed_preds = []
        for s in range(ENSEMBLE_SIZE):
            model = train_model_variant(curve[tr_idx], clinic[tr_idx], y[tr_idx],
                                         seed=SEED + fold_i * 10 + s, use_scheduler=use_scheduler)
            seed_preds.append(predict_model(model, curve[va_idx], clinic[va_idx]))
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return oof


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int = load_cohort(INTERNAL_XLSX)
    meta_int = meta_int[valid_rows(meta_int)].reset_index(drop=True)
    print(f"Cohort: internal n={len(meta_int)} (internal-only ablation, external 미사용)")

    rows = []
    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        mask_val = np.isfinite(y_int_all)
        n_pos = int(y_int_all[mask_val].sum())
        n_neg = int(mask_val.sum() - n_pos)
        if min(n_pos, n_neg) < MIN_POSITIVES:
            print(f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만")
            continue

        meta_m = meta_int.loc[mask_val].reset_index(drop=True)
        y = y_int_all[mask_val].astype(int)
        cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)

        curve_raw = meta_m[AEC_COLS].astype(float).to_numpy()
        clinic, _ = clinical_matrix(meta_m, None)

        print(f"[{feat}] baseline(no scheduler) 5-fold OOF 학습 (n={len(y)}, n_pos={y.sum()})")
        oof_base = cnn_internal_oof_variant(curve_raw, clinic, y, cv, use_scheduler=False)
        print(f"[{feat}] ReduceLROnPlateau 5-fold OOF 학습")
        oof_sched = cnn_internal_oof_variant(curve_raw, clinic, y, cv, use_scheduler=True)

        auc_base = float(roc_auc_score(y, oof_base))
        auc_sched = float(roc_auc_score(y, oof_sched))
        d = delong_paired_auc_test(y, oof_base, oof_sched)
        print(f"[{feat}] internal AUC: no-scheduler={auc_base:.4f}  ReduceLROnPlateau={auc_sched:.4f}  "
              f"diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")

        rows.append({"feature": feat, "n": int(len(y)), "n_pos": int(y.sum()),
                      "auc_no_scheduler": auc_base, "auc_reduce_on_plateau": auc_sched,
                      "auc_diff": d["diff"], "z": d["z"], "p_value": d["p_value"]})

        pd.DataFrame({"feature": feat, "patient_id": meta_m["PatientID"].to_numpy(), "y": y,
                       "score_no_scheduler": oof_base, "score_reduce_on_plateau": oof_sched}
                      ).to_csv(OUTPUT_DIR / f"predictions_{slug}.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "lr_scheduler_ablation_summary.csv", index=False)
    print(f"Saved summary to {OUTPUT_DIR / 'lr_scheduler_ablation_summary.csv'}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
