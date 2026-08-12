from __future__ import annotations

# AEC-128 raw curve + clinic4를 입력으로 하는 1D CNN / Transformer 멀티태스크 신경망을 시도한다
# (이홍선 교수 요청, 항목9). 7개 연속형 체성분 SUM feature를 동시에 예측하는 회귀 네트워크와,
# step3와 동일한 성별 mean±1SD cutoff으로 이분화한 7개 feature를 동시에 예측하는 분류 네트워크를
# 아키텍처별(CNN/Transformer)로 각각 하나씩 학습한다. 개별 feature마다 별도 모델을 쓰지 않고
# shared trunk 멀티태스크로 묶어 n~1000대 소표본에서 표본 효율을 높인다.
# internal(Gangnam) 5-fold로 재학습해 OOF 예측을 만들고, 전체 internal로 최종 재학습한 모델을
# external(Sinchon)에 동결 적용(재학습 금지) — 하이퍼파라미터는 고정값을 미리 정하고 external로
# 튜닝하지 않는다([[feedback_internal_external_validation_discipline]]).
# n~1000 규모에서는 고전적 선형/로지스틱 회귀를 능가하지 못할 가능성이 높음 — 그 자체가 유효한
# 실험 결과이므로 있는 그대로 보고한다(train R^2와 OOF R^2 격차로 과적합 여부도 함께 로깅).

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 로컬 환경 OpenMP 중복 초기화 오류 회피

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step6"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
MIN_POSITIVES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
D_MODEL = 32

REG_FEATURES = {
    "IMATA_SUM": "imata", "NAMA_SUM": "nama", "LAMA_SUM": "lama", "TAMA_SUM": "tama",
    "SAT(피하지방)_SUM": "sat", "VAT(내장지방)_SUM": "vat", "Total Fat_SUM": "total_fat",
}
# (slug, cutoff 방향) — step3_clinic_aec_logistic.py의 FEATURES와 동일한 정의(단, 이 스크립트는 근육계열도
# 그대로 포함한 옛 7-feature 세트를 씀 — output feature를 지방계열로 좁히기 전 실험이라 아직 미반영)
CLF_FEATURES: dict[str, tuple[str, str]] = {
    "TAMA_SUM": ("tama", "low"), "NAMA_SUM": ("nama", "low"), "LAMA_SUM": ("lama", "high"),
    "IMATA_SUM": ("imata", "high"), "SAT(피하지방)_SUM": ("sat", "high"),
    "VAT(내장지방)_SUM": ("vat", "high"), "Total Fat_SUM": ("total_fat", "high"),
}


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# 환자별로 자기 자신의 128개 값 기준 z-score(patient-wise normalization) — 코호트 통계를 쓰지 않으므로
# raw/patient-wise만 허용하는 프로젝트 규칙([[feedback_aec_preprocessing_methods]])을 그대로 만족하고,
# 신경망 최적화 안정성을 위해 스케일도 맞춘다
def patient_wise_normalize(aec_matrix: np.ndarray) -> np.ndarray:
    mean = aec_matrix.mean(axis=1, keepdims=True)
    std = aec_matrix.std(axis=1, keepdims=True)
    return (aec_matrix - mean) / (std + 1e-6)


# age/height/weight 행렬 구성 + 표준화 + (include_sex시) sex 열 결합
def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None, include_sex: bool) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = scaled if not include_sex else np.column_stack(
        [(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    return clinic, scaler


# internal 성별 mean±1SD cutoff을 산출/고정 후 internal·external에 동일 적용해 이분형 라벨 행렬(n,7) 생성.
# 한쪽 클래스가 MIN_POSITIVES 미만인 feature는 전체 코호트에서 해당 열을 NaN 처리(멀티태스크에서 제외)
def build_classification_targets(meta_int: pd.DataFrame, meta_ext: pd.DataFrame,
                                  sex_int: np.ndarray, sex_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid_feats = []
    y_int_cols, y_ext_cols = [], []
    for feat, (slug, direction) in CLF_FEATURES.items():
        v_int = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        v_ext = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_int = np.isfinite(v_int)

        cutoffs = {}
        for s in ("M", "F"):
            v = v_int[mask_int & (sex_int == s)]
            if len(v) == 0:
                continue
            mean, sd = float(v.mean()), float(v.std(ddof=1))
            cutoffs[s] = mean - sd if direction == "low" else mean + sd

        def to_label(values: np.ndarray, sex: np.ndarray) -> np.ndarray:
            th = np.full(len(sex), np.nan)
            for s, c in cutoffs.items():
                th[sex == s] = c
            lab = np.where(values < th, 1.0, 0.0) if direction == "low" else np.where(values > th, 1.0, 0.0)
            lab[~np.isfinite(values) | ~np.isfinite(th)] = np.nan
            return lab

        y_int = to_label(v_int, sex_int)
        y_ext = to_label(v_ext, sex_ext)
        n_pos_int, n_neg_int = int(np.nansum(y_int == 1)), int(np.nansum(y_int == 0))
        n_pos_ext, n_neg_ext = int(np.nansum(y_ext == 1)), int(np.nansum(y_ext == 0))
        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            print(f"[{feat}] SKIP(분류): 한쪽 클래스가 {MIN_POSITIVES}명 미만 "
                  f"(internal pos={n_pos_int} neg={n_neg_int}, external pos={n_pos_ext} neg={n_neg_ext})")
            continue
        valid_feats.append(feat)
        y_int_cols.append(y_int)
        y_ext_cols.append(y_ext)

    return np.column_stack(y_int_cols), np.column_stack(y_ext_cols), valid_feats


# 1D CNN 멀티태스크 trunk: Conv1d 3층(16->32->64) + global average pool -> clinic4와 concat -> MLP head
class CNNTrunk(nn.Module):
    def __init__(self, n_clinic: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + n_clinic, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_out),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.unsqueeze(1)).mean(dim=2)
        return self.head(torch.cat([x, clinic], dim=1))


# Transformer 멀티태스크 trunk: 128 스칼라 시퀀스를 d_model로 projection + 학습형 위치 임베딩 ->
# TransformerEncoder(2layer) -> mean pool -> clinic4와 concat -> MLP head
class TransformerTrunk(nn.Module):
    def __init__(self, n_clinic: int, n_out: int, d_model: int = D_MODEL, nhead: int = 4,
                 n_layers: int = 2, seq_len: int = N_SLICES):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_clinic, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_out),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        x = self.proj(curve.unsqueeze(-1)) + self.pos_emb
        x = self.encoder(x).mean(dim=1)
        return self.head(torch.cat([x, clinic], dim=1))


ARCHITECTURES = {"cnn": CNNTrunk, "transformer": TransformerTrunk}


# 회귀 멀티태스크 학습: target을 fold의 train 통계로 z-score(feature별 NaN 허용, 결측은 loss에서 마스킹)
# 후 고정 하이퍼파라미터(EPOCHS/LR/WEIGHT_DECAY)로 학습. 반환된 mean/std로 예측 시 역변환한다
def train_regression_net(model_cls, curve: np.ndarray, clinic: np.ndarray, y_raw: np.ndarray):
    mean = np.nanmean(y_raw, axis=0)
    std = np.nanstd(y_raw, axis=0)
    std = np.where(std == 0, 1.0, std)
    y_scaled = (y_raw - mean) / std
    mask = np.isfinite(y_scaled)
    y_filled = np.where(mask, y_scaled, 0.0)

    torch.manual_seed(SEED)
    model = model_cls(n_clinic=clinic.shape[1], n_out=y_raw.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_filled, dtype=torch.float32, device=DEVICE)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE)

    n = curve_t.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            pred = model(curve_t[idx], clinic_t[idx])
            loss = (((pred - y_t[idx]) ** 2) * mask_t[idx]).sum() / mask_t[idx].sum().clamp(min=1)
            loss.backward()
            opt.step()
    model.eval()
    return model, mean, std


# 학습된 회귀 모델로 예측 후 표준화를 역변환해 원래 스케일로 반환
def predict_regression(model, mean: np.ndarray, std: np.ndarray, curve: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        pred_scaled = model(curve_t, clinic_t).cpu().numpy()
    return pred_scaled * std + mean


# 분류 멀티태스크 학습: 결측(NaN) 라벨은 loss에서 마스킹한 BCEWithLogitsLoss로 학습
def train_classification_net(model_cls, curve: np.ndarray, clinic: np.ndarray, y_raw: np.ndarray):
    mask = np.isfinite(y_raw)
    y_filled = np.where(mask, y_raw, 0.0)

    torch.manual_seed(SEED)
    model = model_cls(n_clinic=clinic.shape[1], n_out=y_raw.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_filled, dtype=torch.float32, device=DEVICE)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE)

    n = curve_t.shape[0]
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            logit = model(curve_t[idx], clinic_t[idx])
            loss = (bce(logit, y_t[idx]) * mask_t[idx]).sum() / mask_t[idx].sum().clamp(min=1)
            loss.backward()
            opt.step()
    model.eval()
    return model


# 학습된 분류 모델로 확률(sigmoid) 예측
def predict_classification(model, curve: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t, clinic_t)).cpu().numpy()


# R^2/RMSE/MAE/Pearson r(유의성)과 R^2의 bootstrap 95% CI를 산출
def regression_significance_stats(y: np.ndarray, pred: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(np.mean(np.abs(y - pred)))
    r, p_value = stats.pearsonr(y, pred)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_r2 = np.array([r2_score(y[bi], pred[bi]) for bi in rng.integers(0, n, size=(n_boot, n))])
    ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])
    return {"n": int(n), "r2": float(r2_score(y, pred)), "r2_ci_lower": float(ci_lo), "r2_ci_upper": float(ci_hi),
            "rmse": rmse, "mae": mae, "pearson_r": r, "p_value": p_value}


# 회귀: clinic4-only(고전 baseline) vs CNN vs Transformer를 internal 5-fold OOF + external frozen으로 비교.
# train R^2(OOF가 아닌 전체 internal 재적합 기준)와 OOF R^2 격차를 과적합 진단으로 함께 로깅
def run_regression(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, curve_int: np.ndarray, curve_ext: np.ndarray,
                    clinic_int: np.ndarray, clinic_ext: np.ndarray, output_dir: Path) -> None:
    y_int = meta_int[list(REG_FEATURES.keys())].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_ext = meta_ext[list(REG_FEATURES.keys())].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_by_model = {name: np.full_like(y_int, np.nan) for name in ("clinic4", "cnn", "transformer")}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(curve_int)):
        oof_by_model["clinic4"][va_idx] = LinearRegression().fit(clinic_int[tr_idx], np.nan_to_num(y_int[tr_idx])).predict(clinic_int[va_idx])
        for arch_name, model_cls in ARCHITECTURES.items():
            model, mean, std = train_regression_net(model_cls, curve_int[tr_idx], clinic_int[tr_idx], y_int[tr_idx])
            oof_by_model[arch_name][va_idx] = predict_regression(model, mean, std, curve_int[va_idx], clinic_int[va_idx])
        print(f"[회귀 fold {fold + 1}/{N_FOLDS}] 완료")

    ext_by_model = {}
    train_r2_by_model: dict[str, np.ndarray] = {}
    ext_by_model["clinic4"] = LinearRegression().fit(clinic_int, np.nan_to_num(y_int)).predict(clinic_ext)
    for arch_name, model_cls in ARCHITECTURES.items():
        model, mean, std = train_regression_net(model_cls, curve_int, clinic_int, y_int)
        ext_by_model[arch_name] = predict_regression(model, mean, std, curve_ext, clinic_ext)
        train_pred = predict_regression(model, mean, std, curve_int, clinic_int)
        train_r2_by_model[arch_name] = train_pred

    rows = []
    for j, feat in enumerate(REG_FEATURES):
        mask_int = np.isfinite(y_int[:, j])
        mask_ext = np.isfinite(y_ext[:, j])
        for model_name in ("clinic4", "cnn", "transformer"):
            s_int = regression_significance_stats(y_int[mask_int, j], oof_by_model[model_name][mask_int, j])
            s_ext = regression_significance_stats(y_ext[mask_ext, j], ext_by_model[model_name][mask_ext, j])
            print(f"[{feat} / {model_name} / internal OOF] R2={s_int['r2']:.4f} 95%CI=[{s_int['r2_ci_lower']:.4f}, {s_int['r2_ci_upper']:.4f}]")
            print(f"[{feat} / {model_name} / external frozen] R2={s_ext['r2']:.4f} 95%CI=[{s_ext['r2_ci_lower']:.4f}, {s_ext['r2_ci_upper']:.4f}]")
            rows.append({"feature": feat, "model": model_name, "cohort": "internal", **s_int})
            rows.append({"feature": feat, "model": model_name, "cohort": "external", **s_ext})
            if model_name in train_r2_by_model:
                train_r2 = r2_score(y_int[mask_int, j], train_r2_by_model[model_name][mask_int, j])
                print(f"[{feat} / {model_name}] 과적합 진단: train R^2={train_r2:.4f} vs internal OOF R^2={s_int['r2']:.4f} "
                      f"(격차={train_r2 - s_int['r2']:+.4f})")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "regression_summary.csv", index=False)
    print(f"Saved regression summary to {output_dir / 'regression_summary.csv'}")
    plot_comparison(summary, list(REG_FEATURES.keys()), [REG_FEATURES[f] for f in REG_FEATURES], "r2", "R²",
                     output_dir / "regression_r2_comparison.png", "회귀 R² 비교 (clinic4 vs CNN vs Transformer)")


# 분류: clinic4-only(고전 baseline) vs CNN vs Transformer를 internal 5-fold OOF + external frozen AUC로 비교
def run_classification(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, curve_int: np.ndarray, curve_ext: np.ndarray,
                        clinic_int: np.ndarray, clinic_ext: np.ndarray, sex_int: np.ndarray, sex_ext: np.ndarray,
                        output_dir: Path) -> None:
    y_int, y_ext, valid_feats = build_classification_targets(meta_int, meta_ext, sex_int, sex_ext)
    if not valid_feats:
        print("[분류] 유효한 feature가 없어 분류 실험을 생략합니다.")
        return

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_by_model = {name: np.full_like(y_int, np.nan) for name in ("clinic4", "cnn", "transformer")}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(curve_int)):
        for j in range(y_int.shape[1]):
            col = y_int[tr_idx, j]
            mask = np.isfinite(col)
            if len(np.unique(col[mask])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx][mask], col[mask])
            oof_by_model["clinic4"][va_idx, j] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        for arch_name, model_cls in ARCHITECTURES.items():
            model = train_classification_net(model_cls, curve_int[tr_idx], clinic_int[tr_idx], y_int[tr_idx])
            oof_by_model[arch_name][va_idx] = predict_classification(model, curve_int[va_idx], clinic_int[va_idx])
        print(f"[분류 fold {fold + 1}/{N_FOLDS}] 완료")

    ext_by_model = {}
    clinic4_ext = np.full_like(y_ext, np.nan)
    for j in range(y_int.shape[1]):
        col = y_int[:, j]
        mask = np.isfinite(col)
        clf = LogisticRegression(max_iter=2000).fit(clinic_int[mask], col[mask])
        clinic4_ext[:, j] = clf.predict_proba(clinic_ext)[:, 1]
    ext_by_model["clinic4"] = clinic4_ext
    for arch_name, model_cls in ARCHITECTURES.items():
        model = train_classification_net(model_cls, curve_int, clinic_int, y_int)
        ext_by_model[arch_name] = predict_classification(model, curve_ext, clinic_ext)

    rows = []
    for j, feat in enumerate(valid_feats):
        mask_int = np.isfinite(y_int[:, j])
        mask_ext = np.isfinite(y_ext[:, j])
        for model_name in ("clinic4", "cnn", "transformer"):
            auc_int = float(roc_auc_score(y_int[mask_int, j], oof_by_model[model_name][mask_int, j]))
            auc_ext = float(roc_auc_score(y_ext[mask_ext, j], ext_by_model[model_name][mask_ext, j]))
            print(f"[{feat} / {model_name}] internal OOF AUC={auc_int:.4f} / external frozen AUC={auc_ext:.4f}")
            rows.append({"feature": feat, "model": model_name, "cohort": "internal", "n": int(mask_int.sum()),
                         "n_pos": int(y_int[mask_int, j].sum()), "auc": auc_int})
            rows.append({"feature": feat, "model": model_name, "cohort": "external", "n": int(mask_ext.sum()),
                         "n_pos": int(y_ext[mask_ext, j].sum()), "auc": auc_ext})

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "classification_summary.csv", index=False)
    print(f"Saved classification summary to {output_dir / 'classification_summary.csv'}")
    plot_comparison(summary, valid_feats, [CLF_FEATURES[f][0] for f in valid_feats], "auc", "AUC",
                     output_dir / "classification_auc_comparison.png", "분류 AUC 비교 (clinic4 vs CNN vs Transformer)")


# clinic4/CNN/Transformer 3개 모델의 internal·external 지표를 feature별 막대그래프로 비교(회귀/분류 공용)
def plot_comparison(summary: pd.DataFrame, features: list[str], slugs: list[str], value_col: str, ylabel: str,
                     out_path: Path, title: str) -> None:
    model_order = ["clinic4", "cnn", "transformer"]
    colors = {"clinic4": "#6b6a66", "cnn": "#2a78d6", "transformer": "#e2622e"}
    x = np.arange(len(features))
    width = 0.8 / len(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(6 * len(features) / 2 + 3, 6))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_order):
            rows = sub[sub["model"] == model_name].set_index("feature").reindex(features)
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, rows[value_col], width, label=model_name, color=colors[model_name])
        if value_col == "auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=24, rotation=15)
        ax.set_title(cohort, fontsize=20, fontweight="bold", color="#161616")
        ax.set_ylabel(ylabel, fontsize=24)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=16)

    if value_col != "auc":
        y_min = min(0.0, min(ax.get_ylim()[0] for ax in axes))
        y_max = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


# 곡선 patient-wise 정규화 + clinic4 행렬 구성 후 회귀/분류 실험을 순서대로 실행
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_int = patient_wise_normalize(meta_int[AEC_COLS].astype(float).to_numpy())
    curve_ext = patient_wise_normalize(meta_ext[AEC_COLS].astype(float).to_numpy())
    clinic_int, scaler = clinical_matrix(meta_int, None, include_sex)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler, include_sex)

    sex_int = meta_int["PatientSex"].astype(str).str.upper().to_numpy()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper().to_numpy()

    run_regression(meta_int, meta_ext, curve_int, curve_ext, clinic_int, clinic_ext, output_dir)
    run_classification(meta_int, meta_ext, curve_int, curve_ext, clinic_int, clinic_ext, sex_int, sex_ext, output_dir)


# internal/external 코호트를 로드/전처리 후 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행
def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    clinical_cols = ["PatientAge", "Height", "Weight"]

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[clinical_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_clinical_int = valid_rows(meta_int)
    mask_clinical_ext = valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_clinical_int).sum()}/{len(mask_clinical_int)}, "
          f"external {(~mask_clinical_ext).sum()}/{len(mask_clinical_ext)}")
    meta_int = meta_int[mask_clinical_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_clinical_ext].reset_index(drop=True)

    sex_int = meta_int["PatientSex"].astype(str).str.upper()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper()

    run(meta_int, meta_ext, OUTPUT_DIR / "total", include_sex=True)
    for sex_label, sub_dir in (("M", OUTPUT_DIR / "male"), ("F", OUTPUT_DIR / "female")):
        print(f"\n=== sex={sex_label} ({sub_dir.name}) ===")
        run(meta_int[sex_int == sex_label].reset_index(drop=True),
            meta_ext[sex_ext == sex_label].reset_index(drop=True), sub_dir, include_sex=False)


if __name__ == "__main__":
    main()
