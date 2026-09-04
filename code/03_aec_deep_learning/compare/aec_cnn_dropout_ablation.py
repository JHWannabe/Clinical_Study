from __future__ import annotations

# aec_cnn_vatsat_disease_compare.py(Intermediate Fusion, ConcatFusionModel dropout=0.3)의 dropout이
# 실제로 필요한지 확인하는 ablation(2026-09-04 사용자 요청: "dropout이 굳이 필요한가" -> "확인해").
# 동일 구조·동일 cohort·동일 fold에서 dropout=0.0으로만 다시 학습해, 이미 저장된 dropout=0.3 예측
# (outputs/03_aec_deep_learning/compare/cnn_vatsat/predictions_cnn_*.csv)과 DeLong paired test로 비교한다.
# dropout=0.3 쪽은 재학습하지 않고 기존 산출물을 그대로 재사용(중복 학습 방지).

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DROPOUT03_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "cnn_vatsat"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "dropout_ablation"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
AGE_CUTOFF = 20
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]

VAT_COL = "VAT(내장지방)_SUM"
SAT_COL = "SAT(피하지방)_SUM"
MEAN_MAS_COL = "mean_mAs"
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
EXTRA_COLS = [VAT_COL, SAT_COL]

FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}
MIN_POSITIVES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
VAL_FRACTION = 0.15
EARLY_STOP_PATIENCE = 15
ENSEMBLE_SIZE = 3


def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    merged[MEAN_MAS_COL] = merged[AEC_COLS].astype(float).mean(axis=1)
    return merged


def valid_rows(meta: pd.DataFrame) -> np.ndarray:
    required_cols = CLINICAL_BASE_COLS + [VAT_COL, SAT_COL, MEAN_MAS_COL]
    vals = meta[required_cols].apply(pd.to_numeric, errors="coerce")
    mask = vals.notna().all(axis=1).to_numpy()
    valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
    return mask & valid_sex


def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None) -> tuple[np.ndarray, StandardScaler]:
    cols = CLINICAL_BASE_COLS + EXTRA_COLS
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    sex = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex, scaled]), scaler


def patient_zscore(curve: np.ndarray) -> np.ndarray:
    mean = curve.mean(axis=1, keepdims=True)
    std = curve.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (curve - mean) / std


# aec_cnn_vatsat_disease_compare.py와 동일 구조, dropout만 인자화(0.0으로 실행)
class CurveEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 64), kernel_sizes=(11, 9, 7), embed_dim=16, dropout=0.0):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2), nn.BatchNorm1d(c1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2), nn.BatchNorm1d(c2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2), nn.BatchNorm1d(c3), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(c3, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.unsqueeze(1))
        x = self.gap(x).squeeze(-1)
        return self.proj(x)


class ClinicEncoder(nn.Module):
    def __init__(self, n_in: int, embed_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, embed_dim), nn.ReLU())

    def forward(self, clinic: torch.Tensor) -> torch.Tensor:
        return self.net(clinic)


class ConcatFusionModel(nn.Module):
    def __init__(self, n_clinic: int, embed_dim: int = 16, head_hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.curve_encoder = CurveEncoder(embed_dim=embed_dim, dropout=dropout)
        self.clinic_encoder = ClinicEncoder(n_clinic, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.curve_encoder(curve), self.clinic_encoder(clinic)], dim=1)
        return self.head(z)


def train_model(curve: np.ndarray, clinic: np.ndarray, y: np.ndarray, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y)), test_size=VAL_FRACTION, random_state=seed, stratify=y)

    model = ConcatFusionModel(n_clinic=clinic.shape[1], dropout=0.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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


def predict_model(model: nn.Module, curve: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t, clinic_t)).cpu().numpy().ravel()


def cnn_internal_oof(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    curve = patient_zscore(curve_raw)
    oof = np.full(len(y), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(curve, y)):
        seed_preds = []
        for s in range(ENSEMBLE_SIZE):
            model = train_model(curve[tr_idx], clinic[tr_idx], y[tr_idx], seed=SEED + fold_i * 10 + s)
            seed_preds.append(predict_model(model, curve[va_idx], clinic[va_idx]))
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return oof


def cnn_external_frozen(curve_int_raw: np.ndarray, clinic_int: np.ndarray, y_int: np.ndarray,
                         curve_ext_raw: np.ndarray, clinic_ext: np.ndarray) -> np.ndarray:
    curve_int = patient_zscore(curve_int_raw)
    curve_ext = patient_zscore(curve_ext_raw)
    preds = []
    for s in range(ENSEMBLE_SIZE):
        model = train_model(curve_int, clinic_int, y_int, seed=SEED + 1000 + s)
        preds.append(predict_model(model, curve_ext, clinic_ext))
    return np.mean(preds, axis=0)


def _delong_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_covariance(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    n_neg = scores.shape[1] - n_pos
    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    k = scores.shape[0]
    tx = np.vstack([_delong_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_delong_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_delong_midrank(scores[r]) for r in range(k)])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


def delong_paired_auc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    mask_int, mask_ext = valid_rows(meta_int), valid_rows(meta_ext)
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"Cohort: internal n={len(meta_int)}, external n={len(meta_ext)}")

    rows = []
    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_val_int, mask_val_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        meta_int_m = meta_int.loc[mask_val_int].reset_index(drop=True)
        meta_ext_m = meta_ext.loc[mask_val_ext].reset_index(drop=True)
        y_int = y_int_all[mask_val_int].astype(int)
        y_ext = y_ext_all[mask_val_ext].astype(int)

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        curve_int_raw = meta_int_m[AEC_COLS].astype(float).to_numpy()
        curve_ext_raw = meta_ext_m[AEC_COLS].astype(float).to_numpy()
        clinic_int, scaler = clinical_matrix(meta_int_m, None)
        clinic_ext, _ = clinical_matrix(meta_ext_m, scaler)

        print(f"[{feat}] dropout=0.0 internal 5-fold OOF 학습 시작")
        oof0 = cnn_internal_oof(curve_int_raw, clinic_int, y_int, cv)
        print(f"[{feat}] dropout=0.0 external 동결평가")
        ext0 = cnn_external_frozen(curve_int_raw, clinic_int, y_int, curve_ext_raw, clinic_ext)

        pred03 = pd.read_csv(DROPOUT03_DIR / f"predictions_cnn_{slug}.csv")

        for cohort, meta_m, y, score0 in (("internal", meta_int_m, y_int, oof0), ("external", meta_ext_m, y_ext, ext0)):
            base_rows = pred03[(pred03["feature"] == feat) & (pred03["cohort"] == cohort)]
            base_aligned = base_rows.set_index("patient_id").reindex(meta_m["PatientID"].to_numpy())
            assert base_aligned["score"].notna().all(), f"{feat}/{cohort}: patient_id 정렬 실패"
            score03 = base_aligned["score"].to_numpy()

            auc0 = float(roc_auc_score(y, score0))
            auc03 = float(roc_auc_score(y, score03))
            d = delong_paired_auc_test(y, score03, score0)
            print(f"[{feat}/{cohort}] AUC dropout=0.0: {auc0:.4f}  dropout=0.3: {auc03:.4f}  "
                  f"diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
            rows.append({"feature": feat, "cohort": cohort, "auc_dropout00": auc0, "auc_dropout03": auc03,
                         "diff_00_minus_03": d["diff"], "z": d["z"], "p_value": d["p_value"]})

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "dropout_ablation_delong.csv", index=False)
    print(f"Saved to {OUTPUT_DIR / 'dropout_ablation_delong.csv'}")


if __name__ == "__main__":
    main()
