from __future__ import annotations

# AEC-128 raw curve(128포인트 전체)를 clinic4(age/height/weight/sex)와 함께 1D CNN에 넣어 DM을
# 예측하고, clinic4만 쓰는 로지스틱 baseline과 비교한다. 이전 버전(2026-08-31 삭제)은
# (a) clinic4를 4차원 그대로 curve GAP 임베딩(128차원)에 concat해 두 입력의 스케일이
# 비대칭이었고, (b) 같은 파일에서 CNN과 FPCA-DNN 두 아키텍처를 동시에 비교해 구조가 복잡했다.
# 이번 버전은 curve/clinic 인코더를 동일 임베딩 차원(embed_dim)으로 맞추고, 아키텍처도
# CNN 하나만 다룬다.
#
# 목표: clinic4-only 대비 internal AUC +0.05~0.1 개선. 커브 전처리(raw/patient-wise z-score,
# 둘 다 [[feedback_aec_preprocessing_methods]]가 허용하는 방식)·conv 채널 폭·커널 크기·dropout·
# embed_dim·augmentation 강도·learning rate로 구성된 넓은 공간을 2단계로 탐색한다: 1단계는
# N_RANDOM_TRIALS개를 무작위 샘플링해 1-seed 5-fold CV로 스크리닝하고, 2단계는 상위
# N_REFINE_TOP_K개만 ENSEMBLE_SIZE-seed 배깅으로 재검증해 최종 설정을 고른다. 전 과정이
# internal 데이터만 사용한다. external은 그렇게 선택된 설정의 동결 모델로 정확히 1번만
# 평가하며, 이 결과를 보고 탐색 공간이나 하이퍼파라미터를 다시 바꾸지 않는다
# ([[feedback_internal_external_validation_discipline]] — external을 반복 조회하며 설정을
# 맞추면 optimistic bias가 생긴다). 2026-08-31 세션에서 좁은 그리드(8개 설정)로 시도했을 때
# 최선의 internal OOF AUC가 clinic4-only와 사실상 동일(Δ-0.0004)했던 것을 이 확장 탐색으로
# 재확인한다 — [[project_step6_deep_learning_reimplementation]] 참고.

import copy
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step6"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
TARGET = "DM"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
VAL_FRACTION = 0.15
EARLY_STOP_PATIENCE = 12
ENSEMBLE_SIZE = 3  # 최종 확정 설정은 시드 다른 모델 3개를 배깅해 분산을 줄임(고정값, 그리드서치 대상 아님)

# 하이퍼파라미터 탐색 공간 — 전부 internal 5-fold CV로만 평가한다(external 미접촉).
# 1단계: 이 공간에서 N_RANDOM_TRIALS개를 무작위 샘플링해 1-seed 5-fold CV로 빠르게 스크리닝.
# 2단계: 상위 N_REFINE_TOP_K개만 ENSEMBLE_SIZE-seed 5-fold CV로 재검증해 최종 설정 확정.
# 2026-08-31 1차 탐색(80개, weight_decay/batch_size 고정)에서 delta_vs_clinic4가 전부 +0.01
# 이하로 수렴해(최대 +0.0099) 아직 안 훑은 축(weight_decay/batch_size)을 추가하고 기존 축도
# 더 세분화해 재탐색.
SEARCH_SPACE = {
    "curve_prep": ["raw", "patient_zscore"],
    "channels": [(16, 32, 16), (32, 64, 32), (64, 128, 64), (128, 256, 128)],
    "kernel_sizes": [(5, 3, 3), (7, 5, 3), (11, 9, 7), (15, 11, 7)],
    "dropout": [0.1, 0.2, 0.4, 0.6],
    "embed_dim": [8, 16, 32, 64],
    "augment_std": [0.0, 0.02, 0.05, 0.1],  # 학습 배치에 더할 가우시안 노이즈 표준편차(curve std 대비 비율)
    "lr": [3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],
}
N_RANDOM_TRIALS = 400
N_REFINE_TOP_K = 10

META_COLS = ["PatientID", "PatientAge", "Height", "Weight", "PatientSex", TARGET]


# 엑셀 metadata 시트에서 clinic4(+PatientID/TARGET)만, aec_128 시트에서 raw 128포인트만 골라 PatientID 기준으로 병합
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


# curve 전처리: raw 그대로, 또는 환자별(행별) z-score. 둘 다 환자 자신의 값만 쓰므로 fold 분할과
# 무관하게 미리 적용해도 leakage가 없다(cohort 통계를 쓰는 정규화는 사용하지 않음).
def prepare_curve(curve: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return curve
    if mode == "patient_zscore":
        mean = curve.mean(axis=1, keepdims=True)
        std = curve.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (curve - mean) / std
    raise ValueError(f"unknown curve prep mode: {mode}")


# curve(1D CNN, conv 3층+GAP)와 clinic4(소형 MLP)를 동일 embed_dim으로 인코딩한 뒤 concat -> MLP head.
# 이전 버전은 curve 임베딩(128차원)과 clinic 임베딩(16차원)의 스케일이 비대칭이었던 것을 대칭으로 수정.
# channels/kernel_sizes/dropout/embed_dim은 모두 SEARCH_SPACE에서 internal CV로 탐색되는 값이다.
class CurveClinicCNN(nn.Module):
    def __init__(self, n_clinic: int, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 dropout: float, embed_dim: int):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2), nn.BatchNorm1d(c3), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.curve_encoder = nn.Sequential(nn.Linear(c3, embed_dim), nn.ReLU())
        self.clinic_encoder = nn.Sequential(nn.Linear(n_clinic, embed_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        x = self.conv(curve.unsqueeze(1))
        x = self.gap(x).squeeze(-1)
        x = self.curve_encoder(x)
        c = self.clinic_encoder(clinic)
        return self.head(torch.cat([x, c], dim=1))


# 단일 CNN 모델 학습: 입력을 다시 train/inner-val로 나눠(내부적으로만, 이 함수는 fold의 train
# split만 받으므로 외부 val/test/external은 절대 건드리지 않음) validation AUC가
# EARLY_STOP_PATIENCE epoch 연속 개선 없으면 조기종료하고 best validation AUC 시점 가중치로 복원.
# config는 SEARCH_SPACE 키(channels/kernel_sizes/dropout/embed_dim/augment_std/lr)를 담은 dict.
def train_cnn_model(curve: np.ndarray, clinic: np.ndarray, y_raw: np.ndarray, config: dict, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_raw)), test_size=VAL_FRACTION, random_state=seed, stratify=y_raw,
    )

    model = CurveClinicCNN(n_clinic=clinic.shape[1], channels=config["channels"], kernel_sizes=config["kernel_sizes"],
                            dropout=config["dropout"], embed_dim=config["embed_dim"]).to(DEVICE)
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
    for _ in range(EPOCHS):
        model.train()
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

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(curve_val_t, clinic_val_t)).cpu().numpy().ravel()
        val_auc = roc_auc_score(y_val, val_pred) if len(np.unique(y_val)) > 1 else float("nan")
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


def predict_cnn(model: nn.Module, curve: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t, clinic_t)).cpu().numpy().ravel()


# SEARCH_SPACE에서 n개 설정을 중복 없이 무작위 샘플링(curve_prep 포함, config dict 그대로 담아 반환)
def sample_configs(space: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    keys = list(space.keys())
    seen = set()
    configs = []
    while len(configs) < n:
        cfg = {k: rng.choice(space[k]) for k in keys}
        sig = tuple(cfg[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(cfg)
    return configs


# 주어진 설정(curve 전처리 포함)을 internal 5-fold CV로 평가. fold마다 ensemble_size개 시드로 학습한
# 모델의 예측을 평균(배깅)해 validation fold에 채운다. external은 여기서 전혀 참조하지 않는다.
def evaluate_config(curve_raw: np.ndarray, clinic: np.ndarray, y: np.ndarray, folds: list,
                     config: dict, ensemble_size: int) -> tuple[float, np.ndarray]:
    curve = prepare_curve(curve_raw, config["curve_prep"])
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        seed_preds = [
            predict_cnn(
                train_cnn_model(curve[tr_idx], clinic[tr_idx], y[tr_idx], config, SEED + s),
                curve[va_idx], clinic[va_idx],
            )
            for s in range(ensemble_size)
        ]
        oof[va_idx] = np.mean(seed_preds, axis=0)
    return float(roc_auc_score(y, oof)), oof


def _config_row(cfg: dict, auc: float, delta: float) -> dict:
    return {**cfg, "internal_oof_auc": auc, "delta_vs_clinic4": delta}


# DeLong 관련 함수 — code/질병예측/step_disease_logistic.py와 동일 구현 재사용
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


# clinic4/CNN의 internal·external AUC를 막대그래프로 비교
def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    model_order = ["clinic4", "cnn"]
    colors = {"clinic4": "#6b6a66", "cnn": "#2a78d6"}
    x = np.arange(1)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_order):
            row = sub[sub["model"] == model_name].iloc[0]
            offset = (i - 0.5) * width
            ax.bar(x + offset, row["auc"], width, label=model_name, color=colors[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([TARGET], fontsize=24)
        ax.set_title(cohort, fontsize=20, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=24)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=16)

    fig.suptitle("DM AUC 비교 (clinic4 vs CNN-raw curve)", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE}")
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    print(f"internal n={len(meta_int)}, external n={len(meta_ext)}")

    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    clinic_int, scaler = clinical_matrix(meta_int, None)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)
    y_int = meta_int[TARGET].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_ext = meta_ext[TARGET].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))

    clinic4_oof = np.full(len(y_int), np.nan)
    for tr_idx, va_idx in folds:
        clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int[tr_idx])
        clinic4_oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
    clinic4_auc_int = float(roc_auc_score(y_int, clinic4_oof))
    print(f"[baseline] clinic4 internal OOF AUC={clinic4_auc_int:.4f}")

    # ---- 1단계: SEARCH_SPACE에서 N_RANDOM_TRIALS개 무작위 샘플을 1-seed 5-fold CV로 스크리닝.
    # internal만 사용, external은 아래에서 전혀 참조하지 않음. ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_configs = sample_configs(SEARCH_SPACE, N_RANDOM_TRIALS, SEED)
    stage1_results = []
    for i, cfg in enumerate(stage1_configs):
        auc, _ = evaluate_config(curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=1)
        delta = auc - clinic4_auc_int
        print(f"[stage1 {i + 1}/{len(stage1_configs)}] {cfg} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
        stage1_results.append({"config": cfg, "auc": auc, "delta": delta})

    pd.DataFrame([_config_row(r["config"], r["auc"], r["delta"]) for r in stage1_results]).to_csv(
        OUTPUT_DIR / "search_stage1_random.csv", index=False,
    )
    print(f"Saved stage1 random search to {OUTPUT_DIR / 'search_stage1_random.csv'}")

    # ---- 2단계: stage1 상위 N_REFINE_TOP_K개만 ENSEMBLE_SIZE-seed로 재검증해 최종 설정 확정 ----
    stage1_results.sort(key=lambda r: r["auc"], reverse=True)
    top_configs = stage1_results[:N_REFINE_TOP_K]
    stage2_results = []
    best = None
    for r in top_configs:
        cfg = r["config"]
        auc, oof = evaluate_config(curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=ENSEMBLE_SIZE)
        delta = auc - clinic4_auc_int
        print(f"[stage2-refine] {cfg} internal_OOF_AUC={auc:.4f} (delta={delta:+.4f}, ensemble={ENSEMBLE_SIZE})")
        stage2_results.append({"config": cfg, "auc": auc, "delta": delta})
        if best is None or auc > best["auc"]:
            best = {"config": cfg, "auc": auc, "oof": oof}

    pd.DataFrame([_config_row(r["config"], r["auc"], r["delta"]) for r in stage2_results]).to_csv(
        OUTPUT_DIR / "search_stage2_refine.csv", index=False,
    )
    print(f"Saved stage2 refine search to {OUTPUT_DIR / 'search_stage2_refine.csv'}")
    print(f"[선택] {best['config']} internal_OOF_AUC={best['auc']:.4f} "
          f"(delta={best['auc'] - clinic4_auc_int:+.4f} vs clinic4)")

    # ---- external: 위에서 선택된 설정으로 딱 1번만 동결 평가(재조정 금지) ----
    best_config = best["config"]
    curve_int_final = prepare_curve(curve_int_raw, best_config["curve_prep"])
    curve_ext_final = prepare_curve(curve_ext_raw, best_config["curve_prep"])
    ext_preds = [
        predict_cnn(
            train_cnn_model(curve_int_final, clinic_int, y_int, best_config, SEED + s),
            curve_ext_final, clinic_ext,
        )
        for s in range(ENSEMBLE_SIZE)
    ]
    cnn_ext_pred = np.mean(ext_preds, axis=0)

    clinic4_full = LogisticRegression(max_iter=2000).fit(clinic_int, y_int)
    clinic4_ext_pred = clinic4_full.predict_proba(clinic_ext)[:, 1]

    cnn_auc_ext = float(roc_auc_score(y_ext, cnn_ext_pred))
    clinic4_auc_ext = float(roc_auc_score(y_ext, clinic4_ext_pred))
    cnn_ci = bootstrap_auc_ci(y_ext, cnn_ext_pred)
    clinic4_ci = bootstrap_auc_ci(y_ext, clinic4_ext_pred)
    print(f"[external, 1회 동결평가] clinic4 AUC={clinic4_auc_ext:.4f} 95%CI=[{clinic4_ci[0]:.4f}, {clinic4_ci[1]:.4f}] / "
          f"cnn AUC={cnn_auc_ext:.4f} 95%CI=[{cnn_ci[0]:.4f}, {cnn_ci[1]:.4f}]")

    rows = [
        {"feature": TARGET, "model": "clinic4", "cohort": "internal", "n": int(len(y_int)),
         "n_pos": int(y_int.sum()), "auc": clinic4_auc_int, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
        {"feature": TARGET, "model": "cnn", "cohort": "internal", "n": int(len(y_int)),
         "n_pos": int(y_int.sum()), "auc": best["auc"], "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
        {"feature": TARGET, "model": "clinic4", "cohort": "external", "n": int(len(y_ext)),
         "n_pos": int(y_ext.sum()), "auc": clinic4_auc_ext, "auc_ci_lower": clinic4_ci[0], "auc_ci_upper": clinic4_ci[1]},
        {"feature": TARGET, "model": "cnn", "cohort": "external", "n": int(len(y_ext)),
         "n_pos": int(y_ext.sum()), "auc": cnn_auc_ext, "auc_ci_lower": cnn_ci[0], "auc_ci_upper": cnn_ci[1]},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary.csv", index=False)
    print(f"Saved classification summary to {OUTPUT_DIR / 'classification_summary.csv'}")

    delong_rows = []
    for cohort, y, score_a, score_b in (
        ("internal", y_int, clinic4_oof, best["oof"]),
        ("external", y_ext, clinic4_ext_pred, cnn_ext_pred),
    ):
        res = delong_paired_auc_test(y, score_a, score_b)
        print(f"[cnn vs clinic4 / {cohort}] delta_auc={res['diff']:+.4f} z={res['z']:.4f} p={res['p_value']:.4f}")
        delong_rows.append({"feature": TARGET, "comparison": "cnn_minus_clinic4", "cohort": cohort, **res})
    pd.DataFrame(delong_rows).to_csv(OUTPUT_DIR / "delong_vs_clinic4.csv", index=False)
    print(f"Saved DeLong comparison to {OUTPUT_DIR / 'delong_vs_clinic4.csv'}")

    plot_comparison(summary, OUTPUT_DIR / "classification_auc_comparison.png")


if __name__ == "__main__":
    main()
