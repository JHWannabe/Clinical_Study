from __future__ import annotations

# AEC-128 curve를 GAF(gramian angular field) 대신 "그냥 xy 라인플롯을 래스터화한 이미지"로 인코딩하는
# 실험(2026-09-02, 사용자 요청: "이미지를 그냥 xy그래프를 넣도록 하면 안되나? x,y축 범위를 동일하게 해서").
# x축 = 슬라이스 인덱스 1..128(이미지 가로 128칸에 1:1 대응, 모든 환자 동일), y축 = raw mA값을 고정된
# [0, 700] 구간(관측 범위 49~646에 여유를 둔 물리적 상한, 코호트 통계로 fit한 값이 아님 — 스캐너 종류에 따라
# cutoff이 달라지는 것을 피하기 위해 데이터 관측 이전에 정할 수 있는 상수로 고정)에 선형 매핑해 128x128
# 1채널 이진 라인 이미지를 만든다. matplotlib으로 환자마다 개별 렌더링하면 배치당 학습 augmentation이 너무
# 느려지므로, GAF 스크립트와 동일하게 배치 텐서 연산만으로 라인을 그리는 curve_to_lineplot 함수를 직접
# 구현한다(각 슬라이스 열에서 인접 슬라이스 값까지 세로로 채워 연결된 선처럼 보이게 함).
#
# GAF와 마찬가지로 CNN2D 인코더·clinic4 concat fusion·backbone hyperparameter·internal 5-fold CV+external
# 1회 동결평가 프로토콜은 전부 동일하게 유지해, "표현 방식(GAF vs 라인플롯)"만 분리해서 비교할 수 있게 한다.
# GAF 자체도 21번째 방법론이 신호 없음으로 수렴했고([[project_aec_gaf_image_fusion_no_signal]]), 라인플롯
# 래스터화는 128개 숫자를 "선의 픽셀 좌표"라는 더 손실이 큰 형태로 다시 인코딩하는 것이라 GAF보다 나을
# 근거는 약하다고 사전에 안내했으나, 사용자가 실제 결과를 보고 싶다고 해 22번째 방법론으로 실행한다.
#
# [[feedback_output_dir_single_producer]] — outputs/arch/fusion_image_lineplot는 이 스크립트 전용.

import copy
import os

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "imaging" / "lineplot"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}
META_COLS = ["PatientID", "PatientAge", "Height", "Weight", "PatientSex", *FEATURES.keys()]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
VAL_FRACTION = 0.15
EARLY_STOP_PATIENCE = 12
ENSEMBLE_SIZE = 3

# aec_fusion_common.BACKBONE과 동일 값 — GAF 실험과 표현 방식만 분리 비교하기 위해 그대로 고정
CONFIG = {"embed_dim": 8, "dropout": 0.6, "augment_std": 0.05, "lr": 3e-3, "weight_decay": 1e-3, "batch_size": 64}
MODEL_NAME = "image_lineplot_cnn"

IMG_H = 128
IMG_W = N_SLICES  # x축(슬라이스) 1:1 대응이라 별도 보간 불필요
Y_MIN, Y_MAX = 0.0, 700.0  # 관측 raw 범위(49~646)에 여유를 둔 고정 물리 상한 — 코호트 통계로 fit한 값 아님


def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl", usecols=META_COLS).reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl", usecols=["PatientID"] + AEC_COLS)
    merged = meta.merge(aec, on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    sex = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex, scaled]), scaler


# curve(B,L=128, raw mA) -> (B,1,H,W) 이진 라인플롯 이미지. x축은 슬라이스 인덱스에 1:1 대응(열=슬라이스),
# y축은 [Y_MIN,Y_MAX] 고정 구간을 [0,H-1] 행으로 선형 매핑(값이 클수록 위쪽 행). 인접 슬라이스 사이는
# 세로 구간을 채워 연결된 선처럼 보이게 한다(floor/ceil로 최소 1px 두께 보장). 배치 단위 GPU 텐서 연산이라
# 매 epoch/배치마다 raw curve에 노이즈를 더한 뒤 즉시 재렌더링해도 비용이 작다(matplotlib 개별 렌더 대비).
def curve_to_lineplot(curve: torch.Tensor) -> torch.Tensor:
    frac = ((curve - Y_MIN) / (Y_MAX - Y_MIN)).clamp(0.0, 1.0)
    pos = (1.0 - frac) * (IMG_H - 1)  # row position, 값이 클수록 위쪽(작은 row index)
    pos_next = torch.cat([pos[:, 1:], pos[:, -1:]], dim=1)
    seg_lo = torch.floor(torch.minimum(pos, pos_next))
    seg_hi = torch.ceil(torch.maximum(pos, pos_next))
    rows = torch.arange(IMG_H, device=curve.device, dtype=curve.dtype).view(1, IMG_H, 1)
    image = ((rows >= seg_lo.unsqueeze(1)) & (rows <= seg_hi.unsqueeze(1))).to(curve.dtype)
    return image.unsqueeze(1)  # (B,1,H,W)


class ImageEncoderLinePlot(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(128, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        image = curve_to_lineplot(curve)
        x = self.conv(image)
        x = self.gap(x).flatten(1)
        return self.proj(x)


class ClinicEncoderMLP(nn.Module):
    def __init__(self, n_clinic: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_clinic, embed_dim), nn.ReLU())

    def forward(self, clinic: torch.Tensor) -> torch.Tensor:
        return self.encoder(clinic)


class ImageClinicConcatModel(nn.Module):
    def __init__(self, n_clinic: int, embed_dim: int, dropout: float):
        super().__init__()
        self.image_encoder = ImageEncoderLinePlot(embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.image_encoder(curve), self.clinic_encoder(clinic)], dim=1)
        return self.head(z)


def train_model(curve_raw: np.ndarray, clinic: np.ndarray, y_raw: np.ndarray, n_clinic: int,
                 seed: int, history: list[dict] | None = None) -> nn.Module:
    torch.manual_seed(seed)
    tr_idx, val_idx = train_test_split(np.arange(len(y_raw)), test_size=VAL_FRACTION, random_state=seed, stratify=y_raw)

    model = ImageClinicConcatModel(n_clinic=n_clinic, embed_dim=CONFIG["embed_dim"], dropout=CONFIG["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    n_pos = float(y_raw[tr_idx].sum())
    n_neg = float(len(tr_idx)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    curve_t = torch.tensor(curve_raw[tr_idx], dtype=torch.float32, device=DEVICE)
    clinic_t = torch.tensor(clinic[tr_idx], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_raw[tr_idx].reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    curve_val_t = torch.tensor(curve_raw[val_idx], dtype=torch.float32, device=DEVICE)
    clinic_val_t = torch.tensor(clinic[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = y_raw[val_idx]
    curve_std = curve_t.std()

    n = curve_t.shape[0]
    batch_size = CONFIG["batch_size"]
    augment_std = CONFIG["augment_std"]
    best_auc, best_state, epochs_no_improve = -np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_curve = curve_t[idx] + torch.randn_like(curve_t[idx]) * curve_std * augment_std
            opt.zero_grad()
            loss = bce(model(batch_curve, clinic_t[idx]), y_t[idx])
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
            best_auc, best_state, epochs_no_improve = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_model(model: nn.Module, curve_raw: np.ndarray, clinic: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        curve_t = torch.tensor(curve_raw, dtype=torch.float32, device=DEVICE)
        clinic_t = torch.tensor(clinic, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(curve_t, clinic_t)).cpu().numpy().ravel()


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


def plot_sample_images(curve_raw: np.ndarray, out_path: Path, n_samples: int = 4) -> None:
    idx = np.linspace(0, len(curve_raw) - 1, n_samples).astype(int)
    curve_t = torch.tensor(curve_raw[idx], dtype=torch.float32, device=DEVICE)
    images = curve_to_lineplot(curve_t).cpu().numpy()
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    for ax, img in zip(axes, images):
        ax.imshow(img[0], cmap="gray_r", origin="upper", aspect="auto")
        ax.set_xlabel("slice(1-128)", fontsize=14)
        ax.set_yticks([0, IMG_H - 1])
        ax.set_yticklabels([f"{Y_MAX:.0f}", f"{Y_MIN:.0f}"], fontsize=12)
    fig.suptitle("AEC curve -> 라인플롯 이미지 예시(고정 x/y축)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample images to {out_path}")


def plot_auc_grouped(summary: pd.DataFrame, out_path: Path) -> None:
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    model_order = ["clinic4", MODEL_NAME]
    colors = {"clinic4": "#6b6a66", MODEL_NAME: "#2a78d6"}
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

    fig.suptitle(f"AUC 비교 (clinic4 vs {MODEL_NAME})", fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def main() -> None:
    print(f"[환경] torch={torch.__version__} device={DEVICE} model={MODEL_NAME}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    print(f"internal n={len(meta_int)}, external n={len(meta_ext)}")

    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    curve_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()
    clinic_int, scaler = clinical_matrix(meta_int, None)
    clinic_ext, _ = clinical_matrix(meta_ext, scaler)
    n_clinic = clinic_int.shape[1]

    plot_sample_images(curve_int_raw, OUTPUT_DIR / "sample_lineplot_images.png")

    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))

    summary_rows, delong_rows = [], []
    for feature in FEATURES:
        y_int = meta_int[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y_ext = meta_ext[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        clinic4_oof = np.full(len(y_int), np.nan)
        for tr_idx, va_idx in folds:
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int[tr_idx])
            clinic4_oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        clinic4_auc_int = float(roc_auc_score(y_int, clinic4_oof))
        print(f"[baseline/{feature}] clinic4 internal OOF AUC={clinic4_auc_int:.4f}")

        print(f"[{feature}] {MODEL_NAME} internal 5-fold OOF 학습 시작 (n={len(y_int)}, n_pos={int(y_int.sum())})")
        image_oof = np.full(len(y_int), np.nan)
        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            seed_preds = [
                predict_model(
                    train_model(curve_int_raw[tr_idx], clinic_int[tr_idx], y_int[tr_idx], n_clinic, SEED + fold_i * 10 + s),
                    curve_int_raw[va_idx], clinic_int[va_idx],
                )
                for s in range(ENSEMBLE_SIZE)
            ]
            image_oof[va_idx] = np.mean(seed_preds, axis=0)
        image_auc_int = float(roc_auc_score(y_int, image_oof))
        print(f"[{feature}] {MODEL_NAME} internal OOF AUC={image_auc_int:.4f} (delta={image_auc_int - clinic4_auc_int:+.4f} vs clinic4)")

        print(f"[{feature}] {MODEL_NAME} external 동결평가 학습 시작")
        loss_histories, ext_preds = [], []
        for s in range(ENSEMBLE_SIZE):
            hist: list[dict] = []
            model = train_model(curve_int_raw, clinic_int, y_int, n_clinic, SEED + 1000 + s, history=hist)
            loss_histories.append(hist)
            ext_preds.append(predict_model(model, curve_ext_raw, clinic_ext))
        image_ext_pred = np.mean(ext_preds, axis=0)
        plot_loss_curve(loss_histories, OUTPUT_DIR / f"loss_curve_{feature}.png", f"AEC image(lineplot)+CNN2D / {feature} 최종모델 학습곡선")

        clinic4_full = LogisticRegression(max_iter=2000).fit(clinic_int, y_int)
        clinic4_ext_pred = clinic4_full.predict_proba(clinic_ext)[:, 1]

        image_auc_ext = float(roc_auc_score(y_ext, image_ext_pred))
        clinic4_auc_ext = float(roc_auc_score(y_ext, clinic4_ext_pred))
        image_ci = bootstrap_auc_ci(y_ext, image_ext_pred)
        clinic4_ci = bootstrap_auc_ci(y_ext, clinic4_ext_pred)
        print(f"[{feature} external, 1회 동결평가] clinic4 AUC={clinic4_auc_ext:.4f} "
              f"95%CI=[{clinic4_ci[0]:.4f}, {clinic4_ci[1]:.4f}] / {MODEL_NAME} AUC={image_auc_ext:.4f} "
              f"95%CI=[{image_ci[0]:.4f}, {image_ci[1]:.4f}]")

        summary_rows.extend([
            {"feature": feature, "model": "clinic4", "cohort": "internal", "n": int(len(y_int)),
             "n_pos": int(y_int.sum()), "auc": clinic4_auc_int, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": MODEL_NAME, "cohort": "internal", "n": int(len(y_int)),
             "n_pos": int(y_int.sum()), "auc": image_auc_int, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan")},
            {"feature": feature, "model": "clinic4", "cohort": "external", "n": int(len(y_ext)),
             "n_pos": int(y_ext.sum()), "auc": clinic4_auc_ext, "auc_ci_lower": clinic4_ci[0], "auc_ci_upper": clinic4_ci[1]},
            {"feature": feature, "model": MODEL_NAME, "cohort": "external", "n": int(len(y_ext)),
             "n_pos": int(y_ext.sum()), "auc": image_auc_ext, "auc_ci_lower": image_ci[0], "auc_ci_upper": image_ci[1]},
        ])

        for cohort, y, score_a, score_b in (
            ("internal", y_int, clinic4_oof, image_oof),
            ("external", y_ext, clinic4_ext_pred, image_ext_pred),
        ):
            res = delong_paired_auc_test(y, score_a, score_b)
            print(f"[{MODEL_NAME} vs clinic4 / {feature} / {cohort}] delta_auc={res['diff']:+.4f} z={res['z']:.4f} p={res['p_value']:.4f}")
            delong_rows.append({"feature": feature, "comparison": f"{MODEL_NAME}_minus_clinic4", "cohort": cohort, **res})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "classification_summary.csv", index=False)
    print(f"Saved classification summary to {OUTPUT_DIR / 'classification_summary.csv'}")

    pd.DataFrame(delong_rows).to_csv(OUTPUT_DIR / "delong_vs_clinic4.csv", index=False)
    print(f"Saved DeLong comparison to {OUTPUT_DIR / 'delong_vs_clinic4.csv'}")

    plot_auc_grouped(summary, OUTPUT_DIR / "classification_auc_comparison.png")
    print(f"[{MODEL_NAME}] 결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
