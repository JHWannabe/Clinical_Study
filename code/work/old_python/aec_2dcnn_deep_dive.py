from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aec_image_cnn_exploratory as cnn  # noqa: E402
from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402
from aec_vendor_neutral_preprocessing_audit import company_from_manufacturer  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_2dcnn_deep_dive"
WORK_DATA_DIR = Path(__file__).resolve().parent / "data_cache"
SIGMA = 1.0
SEEDS = [20260701, 20260711]


@dataclass(frozen=True)
class ExperimentConfig:
    """CNN 학습 정규화 강도(드롭아웃/가중치감쇠/학습률/에폭/patience) 조합 하나를 담는 실험 설정."""

    name: str
    dropout: float
    weight_decay: float
    lr: float
    max_epochs: int = 60
    patience: int = 8


CONFIGS = [
    ExperimentConfig("default_reg", dropout=0.25, weight_decay=1.0e-3, lr=1.0e-3),
    ExperimentConfig("strong_reg", dropout=0.45, weight_decay=5.0e-3, lr=6.0e-4),
]


class AnisoImageCNN(nn.Module):
    """Small 2D CNN with horizontally long kernels to favor z-axis slope/interval patterns."""

    def __init__(self, dropout: float = 0.25) -> None:
        """가로로 긴 비대칭 커널의 합성곱 3개 층과 평균+최댓값 풀링 결합 후 이진 분류용 선형 헤드를 구성."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5, 11), padding=(2, 5))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 14, kernel_size=(3, 9), padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(14)
        self.conv3 = nn.Conv2d(14, 20, kernel_size=(3, 7), padding=(1, 3))
        self.bn3 = nn.BatchNorm2d(20)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """가로로 긴 비대칭 커널 3개 합성곱 블록을 통과시킨 뒤 평균·최댓값 풀링을 합쳐 로짓을 출력."""
        x = F.avg_pool2d(F.silu(self.bn1(self.conv1(x))), 2)
        x = F.avg_pool2d(F.silu(self.bn2(self.conv2(x))), 2)
        x = F.silu(self.bn3(self.conv3(x)))
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.head(self.drop(pooled)).squeeze(1)


def load_dataset(path: Path) -> dict:
    """엑셀에서 원시/정규화 AEC_128 곡선, 라벨, 제조사 범주를 함께 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    company = meta["Manufacturer"].map(company_from_manufacturer).to_numpy()
    return {"meta": meta, "raw": raw, "norm": row_norm(raw), "y": y, "company": company}


def rank_rows(x: np.ndarray) -> np.ndarray:
    """각 행의 값을 순위(0~1로 정규화)로 바꿔 절대 스케일에 무관한 표현으로 변환."""
    ranked = np.vstack([stats.rankdata(row, method="average") for row in x])
    ranked = (ranked - 1.0) / (x.shape[1] - 1.0)
    return ranked - ranked.mean(axis=1, keepdims=True)


def z_rows(x: np.ndarray) -> np.ndarray:
    """각 행(환자)을 자기 자신의 평균/표준편차로 z-표준화."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x - mu) / sd


def harmonize_by_company(
    x_train: np.ndarray,
    x_test: np.ndarray,
    company_train: np.ndarray,
    company_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """train에서 회사별 평균 템플릿과 전체 평균 템플릿을 만들어, 각 환자 곡선의 회사 평균을 전체 평균으로 치환(harmonize)."""
    keep = company_train != "Other"
    global_template = x_train[keep].mean(axis=0)
    templates = {label: x_train[company_train == label].mean(axis=0) for label in np.unique(company_train[keep])}

    def apply(x: np.ndarray, company: np.ndarray) -> np.ndarray:
        """각 행에서 그 행의 회사 템플릿을 빼고 전체 템플릿을 더해 회사 차이를 보정."""
        out = np.empty_like(x)
        for i, label in enumerate(company):
            out[i] = x[i] - templates.get(label, global_template) + global_template
        return out

    return apply(x_train, company_train), apply(x_test, company_test)


def residual(x: np.ndarray) -> np.ndarray:
    """곡선의 양 끝점을 잇는 직선을 빼서 전반적 기울기 성분을 제거한 잔차를 계산."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    line = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    return x - line


def d1(x: np.ndarray) -> np.ndarray:
    """1차 도함수를 계산 (길이를 맞추기 위해 첫 값을 복제)."""
    out = np.diff(x, axis=1)
    return np.column_stack([out[:, :1], out])


def d2(x: np.ndarray) -> np.ndarray:
    """1차 도함수를 한 번 더 미분해 2차 도함수(곡률)를 계산."""
    first = d1(x)
    out = np.diff(first, axis=1)
    return np.column_stack([out[:, :1], out])


def midlate_mask(width: int = 128) -> np.ndarray:
    """중반~후반(41~118) 구간에 가중치 1.0, 나머지는 0.18을 주고 경계를 부드럽게 평활화한 마스크를 생성 (CNN이 인위적인 경계를 학습하지 않도록)."""
    z = np.arange(1, width + 1)
    mask = np.full(width, 0.18, dtype=float)
    mask[(z >= 41) & (z <= 118)] = 1.0
    # soft edges so the CNN does not learn a hard artificial boundary.
    return ndimage.gaussian_filter1d(mask, sigma=3.0, mode="nearest")


def make_three_channels(curve: np.ndarray, channel_set: str) -> np.ndarray:
    """channel_set 이름에 따라 (원곡선/잔차/기울기), (잔차/기울기/곡률), (기울기/곡률/절대기울기),
    (고역통과/기울기/곡률), (중반후반가중 잔차/기울기/곡률) 중 하나의 3채널 조합을 만듦."""
    r = residual(curve)
    s1 = d1(curve)
    s2 = d2(curve)
    hp = curve - ndimage.gaussian_filter1d(curve, sigma=8.0, axis=1, mode="nearest")
    if channel_set == "curve_resid_slope":
        return np.stack([curve, r, s1], axis=1)
    if channel_set == "resid_slope_curv":
        return np.stack([z_rows(r), z_rows(s1), z_rows(s2)], axis=1)
    if channel_set == "slope_curv_abs":
        return np.stack([z_rows(s1), z_rows(s2), z_rows(np.abs(s1))], axis=1)
    if channel_set == "highpass_slope_curv":
        return np.stack([z_rows(hp), z_rows(s1), z_rows(s2)], axis=1)
    if channel_set == "midlate_resid_slope_curv":
        mask = midlate_mask(curve.shape[1])[None, :]
        return np.stack([z_rows(r) * mask, z_rows(s1) * mask, z_rows(s2) * mask], axis=1)
    raise ValueError(channel_set)


def rasterize_channels(train_channels: np.ndarray, test_channels: np.ndarray, height: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """train 채널 값 범위를 기준으로 train/test 3채널 신호를 이미지(세로축=값, 가로축=위치)로 래스터화."""
    cfg = cnn.CnnConfig(height=height, width=128, line_sigma=1.05)
    limits = cnn.fit_channel_limits(train_channels)

    def convert(channels: np.ndarray) -> np.ndarray:
        """채널 배열을 픽셀 이미지 배열로 변환."""
        imgs = np.zeros((channels.shape[0], 3, cfg.height, cfg.width), dtype=np.float32)
        for i in range(channels.shape[0]):
            for ch, (lo, hi) in enumerate(limits):
                imgs[i, ch] = cnn.rasterize_signal(channels[i, ch], lo, hi, cfg)
        return imgs

    return convert(train_channels), convert(test_channels)


def build_curves(g: dict, s: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """평활화된 원시곡선으로부터 5가지 버전(평균정규화, 회사보정 평균정규화/로그중심화/순위모양,
    원시로그절대값)의 train/test 곡선 쌍을 만들어 이름별 딕셔너리로 반환."""
    raw_g = ndimage.gaussian_filter1d(g["raw"], sigma=SIGMA, axis=1, mode="nearest")
    raw_s = ndimage.gaussian_filter1d(s["raw"], sigma=SIGMA, axis=1, mode="nearest")
    mean_g = row_norm(raw_g)
    mean_s = row_norm(raw_s)
    mean_h_g, mean_h_s = harmonize_by_company(mean_g, mean_s, g["company"], s["company"])
    log_g = np.log(np.clip(mean_g, 1e-6, None))
    log_s = np.log(np.clip(mean_s, 1e-6, None))
    log_g = log_g - log_g.mean(axis=1, keepdims=True)
    log_s = log_s - log_s.mean(axis=1, keepdims=True)
    log_h_g, log_h_s = harmonize_by_company(log_g, log_s, g["company"], s["company"])
    rank_h_g, rank_h_s = harmonize_by_company(rank_rows(mean_g), rank_rows(mean_s), g["company"], s["company"])
    raw_log_g = np.log(np.clip(raw_g, 1e-6, None))
    raw_log_s = np.log(np.clip(raw_s, 1e-6, None))
    return {
        "mean_norm": (mean_g, mean_s),
        "mean_norm_company_harmonized": (mean_h_g, mean_h_s),
        "log_centered_company_harmonized": (log_h_g, log_h_s),
        "rank_shape_company_harmonized": (rank_h_g, rank_h_s),
        "raw_log_absolute": (raw_log_g, raw_log_s),
    }


def set_seed(seed: int) -> None:
    """numpy/torch 난수 시드를 고정하고, GPU 사용 시 결정론적 연산 모드를 켬."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_one_fold(
    model_name: str,
    cfg: ExperimentConfig,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """지정된 CNN 아키텍처(tiny/aniso)를 한 폴드에 대해 미니배치+조기종료로 학습하고, 검증 점수와 외부 예측을 반환."""
    set_seed(seed)
    if model_name == "tiny":
        model = cnn.TinyImageCNN(dropout=cfg.dropout).to(cnn.DEVICE)
    elif model_name == "aniso":
        model = AnisoImageCNN(dropout=cfg.dropout).to(cnn.DEVICE)
    else:
        raise ValueError(model_name)
    pos = float(np.sum(y[train_idx] == 1))
    neg = float(np.sum(y[train_idx] == 0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=cnn.DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.tensor(x[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx].astype(float), dtype=torch.float32)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=cnn.DEVICE)
    yv = torch.tensor(y[val_idx].astype(float), dtype=torch.float32, device=cnn.DEVICE)
    rng = np.random.default_rng(seed)
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    patience = cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), 64):
            idx = order[start : start + 64]
            xb = xt[idx].to(cnn.DEVICE)
            yb = yt[idx].to(cnn.DEVICE)
            xb = torch.clamp(xb + 0.012 * torch.randn_like(xb), 0.0, 1.0)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).item())
        if val_loss < best_loss - 1.0e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_score = model(torch.tensor(x[val_idx], dtype=torch.float32, device=cnn.DEVICE)).cpu().numpy()
        ext_score = cnn.predict_torch(model, x_ext)
    return val_score, ext_score, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss)}


def crossfit(
    model_name: str,
    cfg: ExperimentConfig,
    x: np.ndarray,
    y: np.ndarray,
    x_ext: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드 x 5-fold로 지정된 CNN을 반복 학습해 train OOF 점수와 외부 예측을 시드 평균으로 앙상블."""
    all_oof = []
    all_ext = []
    logs = []
    for seed in SEEDS:
        oof = np.zeros(len(y), dtype=float)
        ext_scores = []
        for fold_id, (tr, va) in enumerate(cnn.stratified_folds(y, seed)):
            val, ext, info = train_one_fold(model_name, cfg, x, y, tr, va, x_ext, seed + 100 * fold_id)
            oof[va] = val
            ext_scores.append(ext)
            logs.append({"seed": seed, "fold": fold_id, "model_arch": model_name, "regime": cfg.name, **info})
        all_oof.append(oof)
        all_ext.append(np.mean(ext_scores, axis=0))
    return np.mean(all_oof, axis=0), np.mean(all_ext, axis=0), pd.DataFrame(logs)


def eval_experiment(
    exp_name: str,
    model_arch: str,
    regime: ExperimentConfig,
    img_g: np.ndarray,
    img_s: np.ndarray,
    y_g: np.ndarray,
    y_s: np.ndarray,
    c_g: np.ndarray,
    c_s: np.ndarray,
) -> tuple[list[dict], list[dict], pd.DataFrame, pd.DataFrame]:
    """한 실험 설정(채널 조합x아키텍처x정규화 강도)에 대해 CNN을 학습하고, 임상점수와 스택한 결합모델까지
    만들어 AUC 지표·하향조정 분석·학습로그·환자별 점수를 모두 계산해 반환."""
    oof, ext, logs = crossfit(model_arch, regime, img_g, y_g, img_s)
    logs.insert(0, "experiment", exp_name)
    clinical_th = threshold_youden(y_g, c_g)
    cnn_th = threshold_youden(y_g, oof)
    stack = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEEDS[0])
    stack.fit(np.column_stack([c_g, oof]), y_g)
    fusion_g = stack.decision_function(np.column_stack([c_g, oof]))
    fusion_s = stack.decision_function(np.column_stack([c_s, ext]))
    fusion_th = threshold_youden(y_g, fusion_g)
    metric_rows = [
        {"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, **cnn.model_metrics("Gangnam internal OOF", "clinical", y_g, c_g, clinical_th)},
        {"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, **cnn.model_metrics("Sinchon external", "clinical", y_s, c_s, clinical_th)},
        {"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, **cnn.model_metrics("Gangnam internal OOF", "aec_2dcnn", y_g, oof, cnn_th)},
        {"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, **cnn.model_metrics("Sinchon external", "aec_2dcnn", y_s, ext, cnn_th)},
        {
            "experiment": exp_name,
            "regime": regime.name,
            "model_arch": model_arch,
            **cnn.model_metrics("Gangnam internal OOF", "clinical_plus_aec_2dcnn", y_g, fusion_g, fusion_th),
            **cnn.bootstrap_auc_delta(y_g, c_g, fusion_g, seed=SEEDS[0] + 50),
        },
        {
            "experiment": exp_name,
            "regime": regime.name,
            "model_arch": model_arch,
            **cnn.model_metrics("Sinchon external", "clinical_plus_aec_2dcnn", y_s, fusion_s, fusion_th),
            **cnn.bootstrap_auc_delta(y_s, c_s, fusion_s, seed=SEEDS[0] + 51),
        },
    ]
    deesc_rows = []
    for op, target in cnn.OPS:
        cth = threshold_for_min_sensitivity(y_g, c_g, target)
        cpos_g = c_g >= cth
        cpos_s = c_s >= cth
        ath, train_row = cnn.choose_deesc_threshold(y_g, cpos_g, oof, op)
        deesc_rows.append({"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, **train_row})
        deesc_rows.append({"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, **cnn.deesc_row("Sinchon external", op, y_s, cpos_s, ext, ath)})
    scores = pd.concat(
        [
            pd.DataFrame({"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, "cohort": "Gangnam", "y": y_g, "clinical": c_g, "aec_2dcnn": oof, "clinical_plus_aec_2dcnn": fusion_g}),
            pd.DataFrame({"experiment": exp_name, "regime": regime.name, "model_arch": model_arch, "cohort": "Sinchon", "y": y_s, "clinical": c_s, "aec_2dcnn": ext, "clinical_plus_aec_2dcnn": fusion_s}),
        ],
        ignore_index=True,
    )
    return metric_rows, deesc_rows, logs, scores


def summarize_deesc(details: pd.DataFrame) -> pd.DataFrame:
    """실험x아키텍처x정규화강도x데이터셋별로 여러 민감도 목표에 걸친 최소/평균 하향조정 지표를 요약."""
    return (
        details.groupby(["experiment", "regime", "model_arch", "dataset"])
        .agg(
            min_p_loss=("sensitivity_loss_p_exact", "min"),
            max_sens_loss=("sensitivity_loss", "max"),
            min_spec_gain=("specificity_gain", "min"),
            mean_spec_gain=("specificity_gain", "mean"),
            min_delta_ba=("delta_balanced_accuracy", "min"),
            mean_delta_ba=("delta_balanced_accuracy", "mean"),
            max_fisher_p=("deesc_event_fisher_p", "max"),
            min_deesc_event_rate=("deesc_event_rate", "min"),
            max_deesc_event_rate=("deesc_event_rate", "max"),
            mean_deesc_event_rate=("deesc_event_rate", "mean"),
        )
        .reset_index()
    )


def plot_auc_summary(auc: pd.DataFrame, out_path: Path) -> None:
    """외부(Sinchon) AUC가 가장 높은 상위 24개 실험(AEC단독/결합)을 가로 막대그래프로 정렬해 임상 단독
    AUC 기준선과 비교하는 그래프를 PNG로 저장."""
    sub = auc[(auc["dataset"].eq("Sinchon external")) & (auc["model"].isin(["aec_2dcnn", "clinical_plus_aec_2dcnn"]))].copy()
    sub["label"] = sub["experiment"] + "\n" + sub["model_arch"] + "/" + sub["regime"] + "\n" + sub["model"]
    sub = sub.sort_values("auc", ascending=False).head(24)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(np.arange(len(sub)), sub["auc"], color=np.where(sub["model"].eq("aec_2dcnn"), "#7b3294", "#008837"))
    ax.axvline(0.834521, color="black", lw=1.2, ls="--", label="clinical external AUC")
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0.40, 0.86)
    ax.set_xlabel("Sinchon external AUC")
    ax.set_title("2D CNN deep dive: best external AUC candidates", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 이미지 CNN을 "곡선 그대로"가 아니라 "잔차/기울기/곡률 등
    파생 채널"과 "비대칭(가로로 긴) 커널", "중반후반 가중 마스크" 등으로 바꿔가며 훨씬 깊게 파보면,
    임상변수 대비 외부 AUC를 더 끌어올릴 수 있는가? — CNN 관련 탐색 중 가장 방대한 조합 스캔):

    1. g1090/sdata를 로드하고, build_curves로 5가지 곡선 표현(평균정규화, 회사보정 버전들, 원시로그)을 준비.
    2. 9개의 (곡선표현, 채널조합) 실험 사양(experiment_specs)을 정의하고, 각각에 대해 make_three_channels로
       3채널을 만들어 rasterize_channels로 이미지화한 뒤, 2개 아키텍처(tiny/aniso) x 2개 정규화강도
       (default_reg/strong_reg) = 4가지 학습 설정으로 eval_experiment를 실행 — 총 36개 모델 조합.
    3. 각 조합마다 CNN 단독/임상+CNN 결합 모델의 AUC를 비교하고, 부트스트랩 delta AUC와 민감도
       목표별 하향조정 분석까지 함께 계산.
    4. 모든 조합의 AUC 결과, 하향조정 상세·요약, 환자별 점수, 학습로그를 각각 CSV로 저장.
    5. 외부 AUC가 가장 높은 상위 24개 조합을 막대그래프로 시각화해 PNG로 저장.
    6. 외부에서 AEC 단독/결합 각각 상위 15개 조합을 콘솔에 출력해 어떤 채널·아키텍처·정규화 조합이 가장 유망한지 확인.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_path = WORK_DATA_DIR / "g1090.xlsx" if (WORK_DATA_DIR / "g1090.xlsx").exists() else DATA_DIR / "g1090.xlsx"
    s_path = WORK_DATA_DIR / "sdata.xlsx" if (WORK_DATA_DIR / "sdata.xlsx").exists() else DATA_DIR / "sdata.xlsx"
    g = load_dataset(g_path)
    s = load_dataset(s_path)
    y_g = g["y"].astype(int)
    y_s = s["y"].astype(int)
    c_g, c_s, _ = clinical_scores(g, s)
    curves = build_curves(g, s)
    experiment_specs = [
        ("mean_norm_curve_resid_slope", "mean_norm", "curve_resid_slope"),
        ("company_curve_resid_slope", "mean_norm_company_harmonized", "curve_resid_slope"),
        ("company_resid_slope_curv", "mean_norm_company_harmonized", "resid_slope_curv"),
        ("company_slope_curv_abs", "mean_norm_company_harmonized", "slope_curv_abs"),
        ("company_highpass_slope_curv", "mean_norm_company_harmonized", "highpass_slope_curv"),
        ("company_midlate_resid_slope_curv", "mean_norm_company_harmonized", "midlate_resid_slope_curv"),
        ("log_company_curve_resid_slope", "log_centered_company_harmonized", "curve_resid_slope"),
        ("rank_company_curve_resid_slope", "rank_shape_company_harmonized", "curve_resid_slope"),
        ("raw_log_absolute_curve_resid_slope", "raw_log_absolute", "curve_resid_slope"),
    ]
    metric_rows: list[dict] = []
    deesc_rows: list[dict] = []
    logs = []
    scores = []
    for exp_name, curve_name, channel_set in experiment_specs:
        train_curve, test_curve = curves[curve_name]
        train_channels = make_three_channels(train_curve, channel_set)
        test_channels = make_three_channels(test_curve, channel_set)
        img_g, img_s = rasterize_channels(train_channels, test_channels)
        for arch in ["tiny", "aniso"]:
            for regime in CONFIGS:
                label = f"{exp_name} | {arch}/{regime.name}"
                print(f"\n=== {label} ===", flush=True)
                a, d, log, sc = eval_experiment(exp_name, arch, regime, img_g, img_s, y_g, y_s, c_g, c_s)
                metric_rows.extend(a)
                deesc_rows.extend(d)
                logs.append(log)
                scores.append(sc)
                print(pd.DataFrame(a)[["dataset", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot"]].to_string(index=False))
    auc = pd.DataFrame(metric_rows)
    deesc = pd.DataFrame(deesc_rows)
    deesc_summary = summarize_deesc(deesc)
    score_df = pd.concat(scores, ignore_index=True)
    log_df = pd.concat(logs, ignore_index=True)
    auc.to_csv(OUT_DIR / "cnn_deep_dive_auc_metrics.csv", index=False)
    deesc.to_csv(OUT_DIR / "cnn_deep_dive_deescalation_details.csv", index=False)
    deesc_summary.to_csv(OUT_DIR / "cnn_deep_dive_deescalation_summary.csv", index=False)
    score_df.to_csv(OUT_DIR / "cnn_deep_dive_scores.csv", index=False)
    log_df.to_csv(OUT_DIR / "cnn_deep_dive_training_log.csv", index=False)
    plot_auc_summary(auc, OUT_DIR / "cnn_deep_dive_external_auc_summary.png")
    print("\nTOP SINCHON AEC-ONLY CNN")
    top_aec = auc[(auc["dataset"].eq("Sinchon external")) & (auc["model"].eq("aec_2dcnn"))].sort_values("auc", ascending=False).head(15)
    print(top_aec[["experiment", "model_arch", "regime", "auc", "auc_p_mannwhitney", "sensitivity", "specificity", "accuracy"]].to_string(index=False))
    print("\nTOP SINCHON CLINICAL+CNN")
    top_fusion = auc[(auc["dataset"].eq("Sinchon external")) & (auc["model"].eq("clinical_plus_aec_2dcnn"))].sort_values("auc", ascending=False).head(15)
    print(top_fusion[["experiment", "model_arch", "regime", "auc", "delta_auc", "delta_auc_p_boot", "sensitivity", "specificity", "accuracy"]].to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 여러 곡선 표현 x 채널 조합 x 아키텍처 x 정규화 강도(총 36개 조합)에 대해
    # 2D CNN을 학습하고 AUC/하향조정 결과를 비교하는 대규모 탐색 파이프라인이 수행된다.
    main()
