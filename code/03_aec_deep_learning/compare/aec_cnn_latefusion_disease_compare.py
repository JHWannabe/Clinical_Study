from __future__ import annotations

# HTN/DM/CKD 3질환에 대해 "Late Fusion"(모달리티별 독립 예측 후 확률 평균)을 실제로 구현·평가한다
# (2026-09-04 사용자 요청: "late fusion도 한번 진행해보고 내용 추가해줘" — slide8에서 우리 concat 구조를
# "Late Fusion"이라 잘못 표기했다가 Intermediate Fusion으로 정정한 뒤, 진짜 Late Fusion과 비교해보자는 취지).
#
# aec_cnn_vatsat_disease_compare.py의 ConcatFusionModel(curve/clinic embedding을 concat한 뒤 공유
# Fusion Head로 처리 = Intermediate Fusion)과 대비되는 구조: curve 전용 모델과 clinic 전용 모델을
# "완전히 독립적으로" 학습(서로의 gradient에 관여하지 않음)해 각자 P(질환)을 내고, 두 확률의 평균만으로
# 최종 예측을 만든다 — 특징 벡터 결합이 아니라 결정(확률) 단계에서만 결합하는 것이 Late Fusion의 정의.
#
# 공정한 비교를 위해 코호트 마스크·fold 정의는 step_disease_logistic.py/aec_cnn_vatsat_disease_compare.py와
# 완전히 동일하게 재현하고, clinic4/FPCA/concat-CNN(Intermediate Fusion) 예측은 각각의 predictions 파일에서
# patient_id로 정렬해 그대로 가져와 DeLong test를 수행한다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
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
CONCAT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "cnn_vatsat"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "latefusion"

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

MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_vatsat_aec_fpca": "clinic4+VAT+SAT+AEC (FPCA)",
    "clinic4_vatsat_aec_cnn": "clinic4+VAT+SAT+AEC (1D CNN, Intermediate Fusion)",
    "clinic4_vatsat_aec_latefusion": "clinic4+VAT+SAT+AEC (1D CNN, Late Fusion)",
}
MODEL_COLORS = {
    "clinic4": "#898781",
    "clinic4_vatsat_aec_fpca": "#e2622e",
    "clinic4_vatsat_aec_cnn": "#1b6ba8",
    "clinic4_vatsat_aec_latefusion": "#7a3ea1",
}
MODEL_ORDER = ["clinic4", "clinic4_vatsat_aec_fpca", "clinic4_vatsat_aec_cnn", "clinic4_vatsat_aec_latefusion"]


# step_disease_logistic.py의 load_cohort와 동일
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


# aec_cnn_vatsat_disease_compare.py의 CurveEncoder와 완전히 동일한 구조(공정 비교를 위해 capacity 통일)
class CurveEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 64), kernel_sizes=(11, 9, 7), embed_dim=16, dropout=0.3):
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


# curve 전용 모델: encoder(16) -> 자체 head(16->1). clinic 정보는 전혀 보지 않음
class CurveOnlyModel(nn.Module):
    def __init__(self, embed_dim: int = 16, head_hidden: int = 16, dropout: float = 0.3):
        super().__init__()
        self.encoder = CurveEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(curve))


# clinic 전용 모델: encoder(16) -> 자체 head(16->1). curve 정보는 전혀 보지 않음
class ClinicOnlyModel(nn.Module):
    def __init__(self, n_clinic: int, embed_dim: int = 16, head_hidden: int = 16, dropout: float = 0.3):
        super().__init__()
        self.encoder = ClinicEncoder(n_clinic, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, clinic: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(clinic))


def _train_single_modal(build_model, x_tr: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, val_idx: np.ndarray,
                         seed: int, noise_on_x: bool) -> nn.Module:
    torch.manual_seed(seed)
    model = build_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = float(y[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_t = torch.tensor(x_tr[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    x_val_t = torch.tensor(x_tr[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y[val_idx]
    x_std = x_t.std()

    n = x_t.shape[0]
    batch_size = 64
    best_auc, best_state, epochs_no_improve = -np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_x = x_t[idx]
            if noise_on_x:
                batch_x = batch_x + torch.randn_like(batch_x) * x_std * 0.05
            opt.zero_grad()
            loss = bce(model(batch_x), y_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(x_val_t)).cpu().numpy().ravel()
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


def _predict_single_modal(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(x_t)).cpu().numpy().ravel()


# 5-fold internal OOF: curve 전용/clinic 전용 모델을 완전히 독립적으로 학습(서로 다른 train/val split도
# 독립적으로 뽑음)한 뒤, fold마다 두 확률의 단순 평균(late fusion)으로 채운다. ENSEMBLE_SIZE개 시드 배깅은
# aec_cnn_vatsat_disease_compare.py와 동일하게 유지
def latefusion_internal_oof(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray,
                             cv: StratifiedKFold) -> np.ndarray:
    curve = patient_zscore(curve_raw)
    oof = np.full(len(y), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(curve, y)):
        curve_preds, clinic_preds = [], []
        for s in range(ENSEMBLE_SIZE):
            seed = SEED + fold_i * 10 + s
            tr_sub, val_sub = train_test_split(tr_idx, test_size=VAL_FRACTION, random_state=seed,
                                                stratify=y[tr_idx])
            curve_model = _train_single_modal(lambda: CurveOnlyModel(), curve, y, tr_sub, val_sub,
                                               seed=seed, noise_on_x=True)
            clinic_model = _train_single_modal(lambda: ClinicOnlyModel(n_clinic=clinic.shape[1]), clinic, y,
                                                tr_sub, val_sub, seed=seed + 500, noise_on_x=False)
            curve_preds.append(_predict_single_modal(curve_model, curve[va_idx]))
            clinic_preds.append(_predict_single_modal(clinic_model, clinic[va_idx]))
        p_curve = np.mean(curve_preds, axis=0)
        p_clinic = np.mean(clinic_preds, axis=0)
        oof[va_idx] = 0.5 * p_curve + 0.5 * p_clinic
    return oof


def latefusion_external_frozen(curve_int_raw: np.ndarray, clinic_int: np.ndarray, y_int: np.ndarray,
                                curve_ext_raw: np.ndarray, clinic_ext: np.ndarray) -> np.ndarray:
    curve_int = patient_zscore(curve_int_raw)
    curve_ext = patient_zscore(curve_ext_raw)
    curve_preds, clinic_preds = [], []
    for s in range(ENSEMBLE_SIZE):
        seed = SEED + 1000 + s
        tr_sub, val_sub = train_test_split(np.arange(len(y_int)), test_size=VAL_FRACTION, random_state=seed,
                                            stratify=y_int)
        curve_model = _train_single_modal(lambda: CurveOnlyModel(), curve_int, y_int, tr_sub, val_sub,
                                           seed=seed, noise_on_x=True)
        clinic_model = _train_single_modal(lambda: ClinicOnlyModel(n_clinic=clinic_int.shape[1]), clinic_int,
                                            y_int, tr_sub, val_sub, seed=seed + 500, noise_on_x=False)
        curve_preds.append(_predict_single_modal(curve_model, curve_ext))
        clinic_preds.append(_predict_single_modal(clinic_model, clinic_ext))
    return 0.5 * np.mean(curve_preds, axis=0) + 0.5 * np.mean(clinic_preds, axis=0)


# code/01_disease_logistic/step_disease_logistic.py와 동일 구현(중복 허용 - 코드베이스 관례)
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

    fig, axes = plt.subplots(1, 2, figsize=(8 + 3 * len(features), 8))
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
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18), fontsize=14, frameon=False)
    fig.suptitle("clinic4 vs FPCA vs 1D CNN(Intermediate Fusion) vs 1D CNN(Late Fusion)",
                 fontsize=18, fontweight="bold")
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

    summary_rows, delong_rows = [], []

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

        print(f"[{feat}] Late Fusion internal 5-fold OOF 학습 시작 (n={len(y_int)}, n_pos={y_int.sum()})")
        lf_oof = latefusion_internal_oof(curve_int_raw, clinic_int, y_int, cv)
        print(f"[{feat}] Late Fusion external 동결평가 학습 시작")
        lf_ext = latefusion_external_frozen(curve_int_raw, clinic_int, y_int, curve_ext_raw, clinic_ext)

        summary_rows.append(classification_row(feat, "clinic4_vatsat_aec_latefusion", "internal", y_int, lf_oof))
        summary_rows.append(classification_row(feat, "clinic4_vatsat_aec_latefusion", "external", y_ext, lf_ext))
        print(f"[{feat}/LateFusion] internal AUC={summary_rows[-2]['auc']:.4f}  external AUC={summary_rows[-1]['auc']:.4f} "
              f"95%CI=[{summary_rows[-1]['auc_ci_lower']:.4f}, {summary_rows[-1]['auc_ci_upper']:.4f}]")

        pred_lf = pd.concat([
            pd.DataFrame({"feature": feat, "model": "clinic4_vatsat_aec_latefusion", "cohort": "internal",
                          "patient_id": meta_int_m["PatientID"].to_numpy(), "y": y_int, "score": lf_oof}),
            pd.DataFrame({"feature": feat, "model": "clinic4_vatsat_aec_latefusion", "cohort": "external",
                          "patient_id": meta_ext_m["PatientID"].to_numpy(), "y": y_ext, "score": lf_ext}),
        ], ignore_index=True)
        pred_lf.to_csv(OUTPUT_DIR / f"predictions_latefusion_{slug}.csv", index=False)

        # clinic4 / FPCA(baseline_pred) + concat-CNN(Intermediate Fusion, CONCAT_DIR) 예측을 patient_id로
        # 정렬해 Late Fusion과 DeLong 비교
        other_sources = [
            ("clinic4", "clinic4", baseline_pred),
            ("clinic4_vatsat_aec", "clinic4_vatsat_aec_fpca", baseline_pred),
        ]
        concat_pred = pd.read_csv(CONCAT_DIR / f"predictions_cnn_{slug}.csv").rename(columns={"model": "model_orig"})
        concat_pred["model"] = "clinic4_vatsat_aec_cnn"
        other_sources.append(("clinic4_vatsat_aec_cnn", "clinic4_vatsat_aec_cnn", concat_pred))

        for base_model, out_model_name, source_df in other_sources:
            for cohort, meta_m, y in (("internal", meta_int_m, y_int), ("external", meta_ext_m, y_ext)):
                base_rows = source_df[(source_df["feature"] == feat) & (source_df["model"] == base_model)
                                       & (source_df["cohort"] == cohort)]
                base_aligned = base_rows.set_index("patient_id").reindex(meta_m["PatientID"].to_numpy())
                assert base_aligned["score"].notna().all(), f"{feat}/{base_model}/{cohort}: patient_id 정렬 실패"
                base_score = base_aligned["score"].to_numpy()

                if out_model_name not in [r["model"] for r in summary_rows if r["feature"] == feat and r["cohort"] == cohort]:
                    summary_rows.append(classification_row(feat, out_model_name, cohort, y, base_score))

                lf_score = lf_oof if cohort == "internal" else lf_ext
                d = delong_paired_auc_test(y, base_score, lf_score)
                print(f"[{feat}/{cohort}] DeLong clinic4_vatsat_aec_latefusion vs {out_model_name}: "
                      f"AUC diff={d['diff']:+.4f} z={d['z']:.3f} p={d['p_value']:.4f}")
                delong_rows.append({"feature": feat, "cohort": cohort, "baseline_model": out_model_name,
                                     "extended_model": "clinic4_vatsat_aec_latefusion", "auc_baseline": d["auc_a"],
                                     "auc_extended": d["auc_b"], "auc_diff": d["diff"], "z": d["z"],
                                     "p_value": d["p_value"]})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary_4way.csv", index=False)
    print(f"Saved 4-way summary to {OUTPUT_DIR / 'classification_summary_4way.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(OUTPUT_DIR / "delong_4way.csv", index=False)
    print(f"Saved DeLong comparison to {OUTPUT_DIR / 'delong_4way.csv'}")

    plot_auc_grouped(summary, OUTPUT_DIR / "auc_comparison_4way.png")


if __name__ == "__main__":
    main()
