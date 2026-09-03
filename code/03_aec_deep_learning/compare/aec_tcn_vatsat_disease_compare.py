from __future__ import annotations

# HTN/DM/CKD 3질환에 대해 clinic4 / clinic4+VAT+SAT+AEC(FPCA) / clinic4+VAT+SAT+AEC(1D CNN, concat
# fusion 기본형, aec_cnn_vatsat_disease_compare.py 260902 결과) / clinic4+VAT+SAT+AEC(TCN) 4-way 비교
# (2026-09-03, 사용자 요청: "1d cnn 성능이 best인 것과 비교한거야?" -> "그렇게 해봐"). 기본형 CNN은
# 0901_multimodal_fusion 아키텍처 탐색 결과를 참조하지 않고 새로 작성된 것이었는데, 그 탐색(10개
# 아키텍처 로테이션, [[project_aec_10day_architecture_rotation_result]]) 중 유일하게 family Bonferroni를
# 통과했던 TCN(dilated causal conv, day3)을 이번엔 VAT+SAT까지 포함한 clinic 벡터에 맞춰 다시 적용한다.
# TCN의 구조적 하이퍼파라미터(n_levels/width/kernel_size)는 day3와 동일한 그리드로 internal 5-fold CV
# 스크리닝 -> 상위 3개 ensemble 재검증 -> 확정 설정으로 external 1회 동결평가한다
# ([[feedback_internal_external_validation_discipline]]).
#
# 코호트 마스크·fold 정의는 aec_cnn_vatsat_disease_compare.py와 완전히 동일하게 재현해 predictions.csv
# (clinic4/FPCA) 및 predictions_cnn_*.csv(기본형 CNN)를 patient_id로 정렬 후 TCN과 DeLong 비교한다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "01_disease_logistic" / "logistic"
CNN_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "cnn_vatsat"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "tcn_vatsat"

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

# day3 TCN 아키텍처 탐색과 동일 그리드(docs/aec_architecture_rotation_plan.md 참고)
TCN_SEARCH_SPACE = {"n_levels": [3, 4], "width": [16, 32], "kernel_size": [3, 5]}
TCN_DROPOUT = 0.3
TCN_EMBED_DIM = 16
TCN_HEAD_HIDDEN = 16
N_REFINE_TOP_K = 3

MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_vatsat_aec_fpca": "clinic4+VAT+SAT+AEC (FPCA)",
    "clinic4_vatsat_aec_cnn": "clinic4+VAT+SAT+AEC (1D CNN concat)",
    "clinic4_vatsat_aec_tcn": "clinic4+VAT+SAT+AEC (TCN)",
}
MODEL_COLORS = {
    "clinic4": "#898781",
    "clinic4_vatsat_aec_fpca": "#e2622e",
    "clinic4_vatsat_aec_cnn": "#6a9fb5",
    "clinic4_vatsat_aec_tcn": "#1b3d8f",
}
MODEL_ORDER = ["clinic4", "clinic4_vatsat_aec_fpca", "clinic4_vatsat_aec_cnn", "clinic4_vatsat_aec_tcn"]


# step_disease_logistic.py의 load_cohort와 동일: 연령<20 제외 후 aec_128 원시곡선 병합, mean_mAs 계산
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


# clinic4(age/height/weight+sex) + VAT/SAT 6열 tabular 행렬. scaler는 internal에서 fit해 external에 frozen 적용
def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None) -> tuple[np.ndarray, StandardScaler]:
    cols = CLINICAL_BASE_COLS + EXTRA_COLS
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    sex = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex, scaled]), scaler


# 환자별(patient-wise) z-score. cohort 통계 미참조([[feedback_aec_preprocessing_methods]])
def patient_zscore(curve: np.ndarray) -> np.ndarray:
    mean = curve.mean(axis=1, keepdims=True)
    std = curve.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (curve - mean) / std


# TCN(dilated causal conv) curve encoder. code/0901_multimodal_fusion/aec_arch_day3_tcn.py와 동일 구조.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.relu(self.chomp1(self.conv1(x))))
        out = self.drop(self.relu(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CurveEncoderTCN(nn.Module):
    def __init__(self, width: int, kernel_size: int, n_levels: int, dropout: float, embed_dim: int):
        super().__init__()
        blocks = []
        in_ch = 1
        for i in range(n_levels):
            blocks.append(TemporalBlock(in_ch, width, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = width
        self.network = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(nn.Linear(width, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.network(curve.unsqueeze(1))
        x = self.gap(x).squeeze(-1)
        return self.encoder(x)


class ClinicEncoder(nn.Module):
    def __init__(self, n_in: int, embed_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, embed_dim), nn.ReLU())

    def forward(self, clinic: torch.Tensor) -> torch.Tensor:
        return self.net(clinic)


class ConcatFusionTCN(nn.Module):
    def __init__(self, n_clinic: int, width: int, kernel_size: int, n_levels: int,
                 dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderTCN(width, kernel_size, n_levels, dropout, embed_dim)
        self.clinic_encoder = ClinicEncoder(n_clinic, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.curve_encoder(curve), self.clinic_encoder(clinic)], dim=1)
        return self.head(z)


def build_tcn_model(cfg: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionTCN(
        n_clinic=n_clinic, width=cfg["width"], kernel_size=cfg["kernel_size"], n_levels=cfg["n_levels"],
        dropout=TCN_DROPOUT, embed_dim=TCN_EMBED_DIM, head_hidden=TCN_HEAD_HIDDEN,
    )


def train_model(curve: np.ndarray, clinic: np.ndarray, y: np.ndarray, seed: int, cfg: dict) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y)), test_size=VAL_FRACTION, random_state=seed, stratify=y)

    model = build_tcn_model(cfg, n_clinic=clinic.shape[1]).to(DEVICE)
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


def evaluate_cfg(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray, cv: StratifiedKFold,
                  cfg: dict, ensemble_size: int) -> tuple[float, np.ndarray]:
    curve = patient_zscore(curve_raw)
    oof = np.full(len(y), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(curve, y)):
        seed_preds = [
            predict_model(train_model(curve[tr_idx], clinic[tr_idx], y[tr_idx], SEED + fold_i * 10 + s, cfg),
                          curve[va_idx], clinic[va_idx])
            for s in range(ensemble_size)
        ]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return float(roc_auc_score(y, oof)), oof


# internal 5-fold CV로 TCN 구조 하이퍼파라미터(n_levels/width/kernel_size) 탐색: 1-seed 스크리닝 ->
# 상위 N_REFINE_TOP_K개만 ENSEMBLE_SIZE-seed로 재검증해 최종 설정 확정 (day3와 동일 절차, VAT+SAT
# 포함 clinic 벡터에 맞춰 재탐색)
def search_best_tcn(feat: str, curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray,
                     cv: StratifiedKFold) -> tuple[dict, np.ndarray, float, list[dict], list[dict]]:
    keys = list(TCN_SEARCH_SPACE.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*(TCN_SEARCH_SPACE[k] for k in keys))]

    stage1 = []
    for i, cfg in enumerate(combos):
        auc, _ = evaluate_cfg(curve_raw, clinic, y, cv, cfg, ensemble_size=1)
        print(f"[TCN/{feat} stage1 {i + 1}/{len(combos)}] {cfg} internal_OOF_AUC={auc:.4f}")
        stage1.append({"feature": feat, **cfg, "internal_oof_auc": auc})

    top = sorted(stage1, key=lambda r: r["internal_oof_auc"], reverse=True)[:N_REFINE_TOP_K]
    stage2, best = [], None
    for r in top:
        cfg = {k: r[k] for k in keys}
        auc, oof = evaluate_cfg(curve_raw, clinic, y, cv, cfg, ensemble_size=ENSEMBLE_SIZE)
        print(f"[TCN/{feat} stage2-refine] {cfg} internal_OOF_AUC={auc:.4f} (ensemble={ENSEMBLE_SIZE})")
        stage2.append({"feature": feat, **cfg, "internal_oof_auc": auc})
        if best is None or auc > best["auc"]:
            best = {"cfg": cfg, "auc": auc, "oof": oof}
    print(f"[TCN/{feat} 선택] {best['cfg']} internal_OOF_AUC={best['auc']:.4f}")
    return best["cfg"], best["oof"], best["auc"], stage1, stage2


# internal 전체로 학습한 ENSEMBLE_SIZE개 모델을 external에 딱 1번 동결 평가
def tcn_external_frozen(cfg: dict, curve_int_raw: np.ndarray, clinic_int: np.ndarray, y_int: np.ndarray,
                         curve_ext_raw: np.ndarray, clinic_ext: np.ndarray) -> np.ndarray:
    curve_int = patient_zscore(curve_int_raw)
    curve_ext = patient_zscore(curve_ext_raw)
    preds = [
        predict_model(train_model(curve_int, clinic_int, y_int, SEED + 1000 + s, cfg), curve_ext, clinic_ext)
        for s in range(ENSEMBLE_SIZE)
    ]
    return np.mean(preds, axis=0)


# code/질병예측/step_disease_logistic.py와 동일 구현(중복 허용 - 코드베이스 관례)
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


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 3000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for bi in rng.integers(0, n, size=(n_boot, n)):
        y_bi = y[bi]
        if len(np.unique(y_bi)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_bi, score[bi]))
    if len(boot_aucs) < n_boot * 0.5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


def classification_row(feat: str, model_name: str, cohort: str, y: np.ndarray, score: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, score))
    ci_lo, ci_hi = bootstrap_auc_ci(y, score) if cohort == "external" else (float("nan"), float("nan"))
    return {"feature": feat, "model": model_name, "cohort": cohort, "n": int(len(y)), "n_pos": int(y.sum()),
            "prevalence": float(y.mean()), "auc": auc, "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi}


def plot_auc_grouped(summary: pd.DataFrame, out_path: Path) -> None:
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    x = np.arange(len(features))
    width = 0.8 / len(MODEL_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(6 + 3 * len(features), 8))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, name in enumerate(MODEL_ORDER):
            rows = sub[sub["model"] == name].set_index("feature").reindex(features)
            offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
            ax.bar(x + offset, rows["auc"], width, label=MODEL_LABELS[name], color=MODEL_COLORS[name])
            if rows["auc_ci_lower"].notna().any():
                ax.errorbar(x + offset, rows["auc"],
                             yerr=[rows["auc"] - rows["auc_ci_lower"], rows["auc_ci_upper"] - rows["auc"]],
                             fmt="none", ecolor="black", capsize=4)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=24)
        ax.set_title(cohort, fontsize=24, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=24)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14), fontsize=14, frameon=False)
    fig.suptitle("clinic4 vs FPCA vs 1D CNN(concat) vs TCN (clinic4+VAT+SAT+AEC)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_pred = pd.read_csv(BASELINE_DIR / "predictions.csv")

    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    mask_int, mask_ext = valid_rows(meta_int), valid_rows(meta_ext)
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"Cohort (step_disease_logistic.py와 동일 마스크): internal n={len(meta_int)}, external n={len(meta_ext)}")

    summary_rows, delong_rows, all_stage1, all_stage2 = [], [], [], []

    for feat, slug in FEATURES.items():
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_val_int, mask_val_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        n_pos_int, n_neg_int = int(y_int_all[mask_val_int].sum()), int(mask_val_int.sum() - y_int_all[mask_val_int].sum())
        n_pos_ext, n_neg_ext = int(y_ext_all[mask_val_ext].sum()), int(mask_val_ext.sum() - y_ext_all[mask_val_ext].sum())
        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            print(f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만")
            continue

        meta_int_m = meta_int.loc[mask_val_int].reset_index(drop=True)
        meta_ext_m = meta_ext.loc[mask_val_ext].reset_index(drop=True)
        y_int = y_int_all[mask_val_int].astype(int)
        y_ext = y_ext_all[mask_val_ext].astype(int)

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        curve_int_raw = meta_int_m[AEC_COLS].astype(float).to_numpy()
        curve_ext_raw = meta_ext_m[AEC_COLS].astype(float).to_numpy()
        clinic_int, scaler = clinical_matrix(meta_int_m, None)
        clinic_ext, _ = clinical_matrix(meta_ext_m, scaler)

        print(f"[{feat}] TCN internal 하이퍼파라미터 탐색 시작 (n={len(y_int)}, n_pos={y_int.sum()})")
        best_cfg, tcn_oof, tcn_oof_auc, stage1, stage2 = search_best_tcn(feat, curve_int_raw, clinic_int, y_int, cv)
        all_stage1.extend(stage1)
        all_stage2.extend(stage2)

        print(f"[{feat}] TCN external 동결평가 시작 (설정={best_cfg})")
        tcn_ext = tcn_external_frozen(best_cfg, curve_int_raw, clinic_int, y_int, curve_ext_raw, clinic_ext)

        summary_rows.append(classification_row(feat, "clinic4_vatsat_aec_tcn", "internal", y_int, tcn_oof))
        summary_rows.append(classification_row(feat, "clinic4_vatsat_aec_tcn", "external", y_ext, tcn_ext))
        print(f"[{feat}/TCN] internal AUC={summary_rows[-2]['auc']:.4f}  external AUC={summary_rows[-1]['auc']:.4f} "
              f"95%CI=[{summary_rows[-1]['auc_ci_lower']:.4f}, {summary_rows[-1]['auc_ci_upper']:.4f}]")

        pred_tcn = pd.concat([
            pd.DataFrame({"feature": feat, "model": "clinic4_vatsat_aec_tcn", "cohort": "internal",
                          "patient_id": meta_int_m["PatientID"].to_numpy(), "y": y_int, "score": tcn_oof}),
            pd.DataFrame({"feature": feat, "model": "clinic4_vatsat_aec_tcn", "cohort": "external",
                          "patient_id": meta_ext_m["PatientID"].to_numpy(), "y": y_ext, "score": tcn_ext}),
        ], ignore_index=True)
        pred_tcn.to_csv(OUTPUT_DIR / f"predictions_tcn_{slug}.csv", index=False)

        # baseline 3개(clinic4, FPCA, 기본형 CNN)를 patient_id로 정렬해 TCN과 DeLong 비교
        cnn_pred = pd.read_csv(CNN_DIR / f"predictions_cnn_{slug}.csv")
        for base_model, out_model_name, base_source in [
            ("clinic4", "clinic4", baseline_pred),
            ("clinic4_vatsat_aec", "clinic4_vatsat_aec_fpca", baseline_pred),
            ("clinic4_vatsat_aec_cnn", "clinic4_vatsat_aec_cnn", cnn_pred),
        ]:
            for cohort, meta_m, y in (("internal", meta_int_m, y_int), ("external", meta_ext_m, y_ext)):
                base_rows = base_source[(base_source["feature"] == feat) & (base_source["model"] == base_model)
                                         & (base_source["cohort"] == cohort)]
                base_aligned = base_rows.set_index("patient_id").reindex(meta_m["PatientID"].to_numpy())
                assert base_aligned["score"].notna().all(), f"{feat}/{base_model}/{cohort}: patient_id 정렬 실패"
                base_score = base_aligned["score"].to_numpy()

                if out_model_name not in [r["model"] for r in summary_rows if r["feature"] == feat and r["cohort"] == cohort]:
                    summary_rows.append(classification_row(feat, out_model_name, cohort, y, base_score))

                tcn_score = tcn_oof if cohort == "internal" else tcn_ext
                d = delong_paired_auc_test(y, base_score, tcn_score)
                print(f"[{feat}/{cohort}] DeLong clinic4_vatsat_aec_tcn vs {out_model_name}: "
                      f"AUC diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
                delong_rows.append({"feature": feat, "cohort": cohort, "baseline_model": out_model_name,
                                     "extended_model": "clinic4_vatsat_aec_tcn", "auc_baseline": d["auc_a"],
                                     "auc_extended": d["auc_b"], "auc_diff": d["diff"], "z": d["z"],
                                     "p_value": d["p_value"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary_4way.csv", index=False)
    print(f"Saved 4-way summary to {OUTPUT_DIR / 'classification_summary_4way.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(OUTPUT_DIR / "delong_4way.csv", index=False)
    print(f"Saved DeLong comparison to {OUTPUT_DIR / 'delong_4way.csv'}")

    pd.DataFrame(all_stage1).to_csv(OUTPUT_DIR / "search_stage1_grid.csv", index=False)
    pd.DataFrame(all_stage2).to_csv(OUTPUT_DIR / "search_stage2_refine.csv", index=False)
    print(f"Saved TCN hyperparameter search to {OUTPUT_DIR}")

    plot_auc_grouped(summary, OUTPUT_DIR / "auc_comparison_4way.png")


if __name__ == "__main__":
    main()
