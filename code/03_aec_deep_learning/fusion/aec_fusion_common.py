from __future__ import annotations

# aec_fusion_concat/gated/attnpool/crossattn.py 4개 스크립트가 공유하는 데이터 로딩·학습 루프·
# DeLong/bootstrap·plot 유틸. curve/clinic 인코더 backbone(channels/kernel/dropout/embed_dim 등)은
# aec_deep_learning.py의 확장 탐색(400 stage1 random + 10 refine)에서 찾은 최종 설정으로 고정한다
# (outputs/1d_cnn/search_stage2_refine.csv 최상단 행, internal_oof_auc=0.7254). 4개 스크립트는 이
# backbone을 그대로 두고 fusion 메커니즘(및 그에 딸린 소수 하이퍼파라미터)만 바꿔 internal 5-fold CV로
# 비교한다 — [[project_step6_multimodal_fusion_references]] 비교표(Concat/Gated fusion/Attention
# pooling/Cross-attention)를 코드로 옮긴 것.
#
# 각 fusion 스크립트는 자기 output_dir에만 쓴다([[feedback_output_dir_single_producer]]).
# 모델 선택은 전부 internal CV로만 하고 external은 최종 선택된 설정으로 딱 1번만 동결 평가한다
# ([[feedback_internal_external_validation_discipline]]).

import copy
import itertools
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
# code/질병예측/step_disease_logistic.py의 FEATURES와 동일 정의(metadata의 HTN/DM/CKD 컬럼, 이미 0/1).
# 값(slug)은 파일명 등에 쓰는 소문자 식별자.
FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
VAL_FRACTION = 0.15
EARLY_STOP_PATIENCE = 12
ENSEMBLE_SIZE = 3  # 최종 확정 설정은 시드 다른 모델 3개를 배깅

META_COLS = ["PatientID", "PatientAge", "Height", "Weight", "PatientSex", *FEATURES.keys()]

# aec_deep_learning.py 확장 탐색에서 찾은 최종 backbone(고정값, fusion 스크립트에서 바꾸지 않음)
BACKBONE = {
    "curve_prep": "patient_zscore",
    "channels": (128, 256, 128),
    "kernel_sizes": (11, 9, 7),
    "dropout": 0.6,
    "embed_dim": 8,
    "augment_std": 0.05,
    "lr": 3e-3,
    "weight_decay": 1e-3,
    "batch_size": 64,
}


# 엑셀 metadata 시트에서 clinic4(+PatientID/HTN/DM/CKD)만, aec_128 시트에서 raw 128포인트만 골라 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl", usecols=META_COLS).reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl", usecols=["PatientID"] + AEC_COLS)
    merged = meta.merge(aec, on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# age/height/weight/sex 4열 clinical 행렬 구성(clinic4와 동일 정의), scaler는 internal에서만 fit
def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    sex = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex, scaled]), scaler


# curve 전처리: raw/patient-wise z-score(기존) + 2026-09-02 전처리 스크리닝에서 추가한 3종
# (robust/winsorize/smoothing). 전부 행(환자) 단위 연산만 사용 — cohort 통계 참조 없음
# ([[feedback_aec_preprocessing_methods]]).
def prepare_curve(curve: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return curve
    if mode == "patient_zscore":
        mean = curve.mean(axis=1, keepdims=True)
        std = curve.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (curve - mean) / std
    if mode == "patient_zscore_robust":
        median = np.median(curve, axis=1, keepdims=True)
        q75 = np.percentile(curve, 75, axis=1, keepdims=True)
        q25 = np.percentile(curve, 25, axis=1, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        return (curve - median) / iqr
    if mode == "patient_zscore_winsor":
        lo = np.percentile(curve, 2, axis=1, keepdims=True)
        hi = np.percentile(curve, 98, axis=1, keepdims=True)
        clipped = np.clip(curve, lo, hi)
        mean = clipped.mean(axis=1, keepdims=True)
        std = clipped.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (clipped - mean) / std
    if mode == "patient_zscore_smooth":
        from scipy.signal import savgol_filter
        smoothed = savgol_filter(curve, window_length=9, polyorder=2, axis=1)
        mean = smoothed.mean(axis=1, keepdims=True)
        std = smoothed.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (smoothed - mean) / std
    raise ValueError(f"unknown curve prep mode: {mode}")


# curve 인코더 A: conv 3층+BN+ReLU 후 GAP(균등 평균)으로 풀링 — 4개 fusion 스크립트 중 concat/gated/crossattn이 사용
class CurveEncoderGAP(nn.Module):
    def __init__(self, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int], embed_dim: int):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2), nn.BatchNorm1d(c3), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(nn.Linear(c3, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.unsqueeze(1))
        x = self.gap(x).squeeze(-1)
        return self.encoder(x)


# curve 인코더 B: GAP 대신 슬라이스별 학습된 attention 가중치 α(t)로 가중합(DeepSpiro SpiroEncoder 방식) — attnpool 전용
class CurveEncoderAttnPool(nn.Module):
    def __init__(self, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int], embed_dim: int, attn_hidden: int):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2), nn.BatchNorm1d(c3), nn.ReLU(),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(c3, attn_hidden, kernel_size=1), nn.Tanh(), nn.Conv1d(attn_hidden, 1, kernel_size=1),
        )
        self.encoder = nn.Sequential(nn.Linear(c3, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.unsqueeze(1))            # (B, C, 128)
        scores = self.attn(x)                          # (B, 1, 128)
        alpha = torch.softmax(scores, dim=-1)           # (B, 1, 128) — 시점별 가중치
        pooled = (x * alpha).sum(dim=-1)                # (B, C)
        return self.encoder(pooled)


# 1D residual block(conv-BN-ReLU-conv-BN + skip, He et al. 스타일) — CurveEncoderResNet의 depth 단위
class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# curve 인코더 C: stem(1->width) + ResBlock1D n_blocks개(각 block=conv 2층, 즉 depth=1+2*n_blocks층) + GAP.
# "학습이 너무 빨리 끝나는데 depth를 늘리면 개선되지 않을까" 질문에 대한 진단용 — 3층 CurveEncoderGAP보다
# 훨씬 깊은 구조를 residual로 안정적으로 학습시켜 internal CV에서 실제로 신호가 느는지 확인한다.
class CurveEncoderResNet(nn.Module):
    def __init__(self, width: int, kernel_size: int, n_blocks: int, embed_dim: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, padding=3), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(width, kernel_size, dropout) for _ in range(n_blocks)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(nn.Linear(width, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = self.stem(curve.unsqueeze(1))
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.encoder(x)


# clinic4 인코더: 4개 스크립트 공통(소형 MLP, curve 인코더와 동일 embed_dim으로 대칭)
class ClinicEncoderMLP(nn.Module):
    def __init__(self, n_clinic: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_clinic, embed_dim), nn.ReLU())

    def forward(self, clinic: torch.Tensor) -> torch.Tensor:
        return self.encoder(clinic)


# 단일 모델 학습: build_model()로 fusion별 아키텍처를 생성해 fold의 train split만으로 학습하고
# (외부 val/test/external은 건드리지 않음) validation AUC가 EARLY_STOP_PATIENCE epoch 연속 개선 없으면
# 조기종료 후 best validation AUC 시점 가중치로 복원. config는 BACKBONE ∪ fusion별 그리드 조합.
def train_model(build_model: Callable[[], nn.Module], curve: np.ndarray, clinic: np.ndarray,
                 y_raw: np.ndarray, config: dict, seed: int, history: list[dict] | None = None) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_raw)), test_size=VAL_FRACTION, random_state=seed, stratify=y_raw,
    )

    model = build_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    augment_std = config["augment_std"]
    batch_size = config["batch_size"]
    n_pos = float(y_raw[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve_t = torch.tensor(curve[tr_idx], dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_raw[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y_raw[val_idx]
    curve_std = curve_t.std()

    n = curve_t.shape[0]
    best_auc = -np.inf
    best_state = None
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_curve = curve_t[idx]
            if augment_std > 0:
                batch_curve = batch_curve + torch.randn_like(batch_curve) * curve_std * augment_std
            opt.zero_grad()
            logit = model(batch_curve, clinic_t[idx])
            loss = bce(logit, y_t[idx])
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(curve_val_t, clinic_val_t)).cpu().numpy().ravel()
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else float("nan")
        if history is not None:
            history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_auc": float(val_auc)})
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
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


# DeLong 관련 함수 — code/질병예측/step_disease_logistic.py, aec_deep_learning.py와 동일 구현 재사용
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


# AUC bootstrap 95% CI(external frozen 평가에만 적용, internal은 5-fold CV OOF 점추정만 사용)
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


# 주어진 설정(backbone ∪ fusion 그리드 조합)을 internal 5-fold CV로 평가. fold마다 ensemble_size개
# 시드로 학습한 모델의 예측을 평균(배깅)해 validation fold에 채운다. external은 여기서 참조하지 않는다.
def evaluate_config(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray, folds: list,
                     config: dict, ensemble_size: int, build_model_fn: Callable[[dict, int], nn.Module]) -> tuple[float, np.ndarray]:
    curve = prepare_curve(curve_raw, config["curve_prep"])
    n_clinic = clinic.shape[1]
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        seed_preds = [
            predict_model(
                train_model(lambda: build_model_fn(config, n_clinic), curve[tr_idx], clinic[tr_idx], y[tr_idx], config, SEED + s),
                curve[va_idx], clinic[va_idx],
            )
            for s in range(ensemble_size)
        ]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return float(roc_auc_score(y, oof)), oof


# external 동결평가에 쓰인 최종모델(ENSEMBLE_SIZE개 시드)의 epoch별 train loss/val AUC를 시드별로 겹쳐 그림.
# stage1/stage2 그리드서치 중 학습된 모델들은 그리지 않고(수가 너무 많음) 최종 선택 모델만 대상으로 함.
def plot_loss_curve(histories: list[list[dict]], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, hist in enumerate(histories):
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], marker="o", markersize=3, label=f"seed{i}")
        axes[1].plot(epochs, [h["val_auc"] for h in hist], marker="o", markersize=3, label=f"seed{i}")
    axes[0].set_xlabel("epoch", fontsize=18)
    axes[0].set_ylabel("train BCE loss", fontsize=18)
    axes[0].set_title("Train loss", fontsize=18, fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=12)
    axes[1].set_xlabel("epoch", fontsize=18)
    axes[1].set_ylabel("validation AUC", fontsize=18)
    axes[1].set_title("Validation AUC", fontsize=18, fontweight="bold")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=12)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {out_path}")


# internal·external AUC를 (feature × model) 그룹 막대그래프로 비교. x축=HTN/DM/CKD, 막대 그룹=model_order.
# aec_fusion_compare.py도 이 함수를 그대로 재사용(model_order에 fusion 4종을 다 넣으면 됨).
# 폰트 크기는 기존 대비 3배 고정 유지.
def plot_auc_grouped(summary: pd.DataFrame, out_path: Path, model_order: list[str], colors: dict[str, str], title: str) -> None:
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    x = np.arange(len(features))
    width = 0.8 / len(model_order)

    fig, axes = plt.subplots(1, 2, figsize=(6 + 3 * len(features), 7))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, name in enumerate(model_order):
            rows = sub[sub["model"] == name].set_index("feature").reindex(features)
            offset = (i - (len(model_order) - 1) / 2) * width
            ax.bar(x + offset, rows["auc"], width, label=name, color=colors[name])
            if rows["auc_ci_lower"].notna().any():
                ax.errorbar(x + offset, rows["auc"],
                             yerr=[rows["auc"] - rows["auc_ci_lower"], rows["auc_ci_upper"] - rows["auc"]],
                             fmt="none", ecolor="black", capsize=4)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=22)
        ax.set_title(cohort, fontsize=20, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=24)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=14, loc="lower right")

    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


# 하나의 feature(HTN/DM/CKD 중 하나)·하나의 backbone∪fusion 조합 설정을 internal 5-fold CV로 평가하고
# (1-seed 스크리닝 or ensemble 재검증은 evaluate_config가 처리) stage1/stage2 결과 리스트를 반환.
def _search_configs_for_feature(feature: str, curve_int_raw: np.ndarray, clinic_int: np.ndarray, y_int: np.ndarray,
                                 folds: list, combos: list[dict], fusion_name: str, n_refine_top_k: int,
                                 build_model_fn: Callable[[dict, int], nn.Module], clinic4_auc_int: float) -> tuple[list, list, dict]:
    stage1_results = []
    for i, combo in enumerate(combos):
        cfg = {**BACKBONE, **combo}
        auc, _ = evaluate_config(curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=1, build_model_fn=build_model_fn)
        delta = auc - clinic4_auc_int
        print(f"[{fusion_name}/{feature} stage1 {i + 1}/{len(combos)}] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
        stage1_results.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})

    ranked = sorted(stage1_results, key=lambda r: r["auc"], reverse=True)
    top = ranked[:n_refine_top_k]
    stage2_results = []
    best = None
    for r in top:
        combo = r["combo"]
        cfg = {**BACKBONE, **combo}
        auc, oof = evaluate_config(curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=ENSEMBLE_SIZE, build_model_fn=build_model_fn)
        delta = auc - clinic4_auc_int
        print(f"[{fusion_name}/{feature} stage2-refine] {combo} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f}, ensemble={ENSEMBLE_SIZE})")
        stage2_results.append({"feature": feature, "combo": combo, "auc": auc, "delta": delta})
        if best is None or auc > best["auc"]:
            best = {"combo": combo, "cfg": cfg, "auc": auc, "oof": oof}
    print(f"[{fusion_name}/{feature} 선택] {best['combo']} internal_OOF_AUC={best['auc']:.4f} "
          f"(delta={best['auc'] - clinic4_auc_int:+.4f} vs clinic4)")
    return stage1_results, stage2_results, best


# fusion 스크립트 4개가 공유하는 실행 파이프라인. HTN/DM/CKD(기본값 FEATURES) 각각에 대해:
# fusion-specific search_space를 전수 그리드 탐색(공간이 작아 random sampling 불필요) -> 1-seed 5-fold CV로
# 스크리닝 -> 상위 n_refine_top_k만 ENSEMBLE_SIZE-seed로 재검증해 확정 -> 확정 설정으로 external 1회 동결
# 평가. 세 질환 결과를 하나의 output_dir(CSV 4개+plot 1개)에 "feature" 컬럼으로 구분해 함께 저장한다
# ([[feedback_output_dir_single_producer]] — 이 output_dir을 쓰는 스크립트는 이 파일 하나뿐).
def run_fusion_pipeline(*, fusion_name: str, output_dir: Path, search_space: dict,
                         build_model_fn: Callable[[dict, int], nn.Module], n_refine_top_k: int = 3,
                         features: dict[str, str] | None = None) -> pd.DataFrame:
    features = features or FEATURES
    print(f"[환경] torch={torch.__version__} device={DEVICE} fusion={fusion_name} features={list(features)}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    print(f"internal n={len(meta_int)}, external n={len(meta_ext)}")

    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    clinic_int, scaler = clinical_matrix(meta_int, None)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)
    n_clinic = clinic_int.shape[1]

    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))
    keys = list(search_space.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*(search_space[k] for k in keys))]
    print(f"[{fusion_name}] fusion-specific grid: {len(combos)}개 조합 (backbone={BACKBONE} 고정)")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_stage1, all_stage2, summary_rows, delong_rows = [], [], [], []
    for feature in features:
        y_int = meta_int[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y_ext = meta_ext[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        clinic4_oof = np.full(len(y_int), np.nan)
        for tr_idx, va_idx in folds:
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int[tr_idx])
            clinic4_oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        clinic4_auc_int = float(roc_auc_score(y_int, clinic4_oof))
        print(f"[baseline/{feature}] clinic4 internal OOF AUC={clinic4_auc_int:.4f}")

        stage1_results, stage2_results, best = _search_configs_for_feature(
            feature, curve_int_raw, clinic_int, y_int, folds, combos, fusion_name, n_refine_top_k, build_model_fn, clinic4_auc_int,
        )
        all_stage1.extend(stage1_results)
        all_stage2.extend(stage2_results)

        # ---- external: 위에서 선택된 설정으로 딱 1번만 동결 평가(재조정 금지) ----
        best_cfg = best["cfg"]
        curve_int_final = prepare_curve(curve_int_raw, best_cfg["curve_prep"])
        curve_ext_final = prepare_curve(curve_ext_raw, best_cfg["curve_prep"])
        loss_histories = []
        ext_preds = []
        for s in range(ENSEMBLE_SIZE):
            hist: list[dict] = []
            model = train_model(lambda: build_model_fn(best_cfg, n_clinic), curve_int_final, clinic_int, y_int, best_cfg, SEED + s, history=hist)
            loss_histories.append(hist)
            ext_preds.append(predict_model(model, curve_ext_final, clinic_ext))
        model_ext_pred = np.mean(ext_preds, axis=0)
        plot_loss_curve(loss_histories, output_dir / f"loss_curve_{feature}.png", f"{fusion_name} / {feature} 최종모델 학습곡선")

        clinic4_full = LogisticRegression(max_iter=2000).fit(clinic_int, y_int)
        clinic4_ext_pred = clinic4_full.predict_proba(clinic_ext)[:, 1]

        model_auc_ext = float(roc_auc_score(y_ext, model_ext_pred))
        clinic4_auc_ext = float(roc_auc_score(y_ext, clinic4_ext_pred))
        model_ci = bootstrap_auc_ci(y_ext, model_ext_pred)
        clinic4_ci = bootstrap_auc_ci(y_ext, clinic4_ext_pred)
        print(f"[{fusion_name}/{feature} external, 1회 동결평가] clinic4 AUC={clinic4_auc_ext:.4f} "
              f"95%CI=[{clinic4_ci[0]:.4f}, {clinic4_ci[1]:.4f}] / {fusion_name} AUC={model_auc_ext:.4f} "
              f"95%CI=[{model_ci[0]:.4f}, {model_ci[1]:.4f}]")

        summary_rows.extend([
            {"feature": feature, "model": "clinic4", "cohort": "internal", "n": int(len(y_int)),
             "n_pos": int(y_int.sum()), "auc": clinic4_auc_int, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": fusion_name, "cohort": "internal", "n": int(len(y_int)),
             "n_pos": int(y_int.sum()), "auc": best["auc"], "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": "clinic4", "cohort": "external", "n": int(len(y_ext)),
             "n_pos": int(y_ext.sum()), "auc": clinic4_auc_ext, "auc_ci_lower": clinic4_ci[0], "auc_ci_upper": clinic4_ci[1]},
            {"feature": feature, "model": fusion_name, "cohort": "external", "n": int(len(y_ext)),
             "n_pos": int(y_ext.sum()), "auc": model_auc_ext, "auc_ci_lower": model_ci[0], "auc_ci_upper": model_ci[1]},
        ])

        for cohort, y, score_a, score_b in (
            ("internal", y_int, clinic4_oof, best["oof"]),
            ("external", y_ext, clinic4_ext_pred, model_ext_pred),
        ):
            res = delong_paired_auc_test(y, score_a, score_b)
            print(f"[{fusion_name} vs clinic4 / {feature} / {cohort}] delta_auc={res['diff']:+.4f} z={res['z']:.4f} p={res['p_value']:.4f}")
            delong_rows.append({"feature": feature, "comparison": f"{fusion_name}_minus_clinic4", "cohort": cohort, **res})

    pd.DataFrame([{**r["combo"], "feature": r["feature"], "internal_oof_auc": r["auc"], "delta_vs_clinic4": r["delta"]} for r in all_stage1]).to_csv(
        output_dir / "search_stage1_grid.csv", index=False,
    )
    pd.DataFrame([{**r["combo"], "feature": r["feature"], "internal_oof_auc": r["auc"], "delta_vs_clinic4": r["delta"]} for r in all_stage2]).to_csv(
        output_dir / "search_stage2_refine.csv", index=False,
    )
    print(f"Saved stage1/stage2 search to {output_dir}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "classification_summary.csv", index=False)
    print(f"Saved classification summary to {output_dir / 'classification_summary.csv'}")

    pd.DataFrame(delong_rows).to_csv(output_dir / "delong_vs_clinic4.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_vs_clinic4.csv'}")

    plot_auc_grouped(
        summary, output_dir / "classification_auc_comparison.png",
        model_order=["clinic4", fusion_name], colors={"clinic4": "#6b6a66", fusion_name: "#2a78d6"},
        title=f"AUC 비교 (clinic4 vs {fusion_name})",
    )
    print(f"[{fusion_name}] 결과 저장 완료: {output_dir}")
    return summary
