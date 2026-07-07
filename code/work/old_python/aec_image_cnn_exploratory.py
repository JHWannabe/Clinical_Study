from __future__ import annotations

import json
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores, load_aec128  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_image_cnn_exploratory"
SEEDS = [20260701, 20260711, 20260721]
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class CnnConfig:
    """이미지 기반 CNN 학습에 필요한 이미지 크기·배치·학습률·조기종료 하이퍼파라미터 묶음."""

    height: int = 64
    width: int = 128
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 12
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    dropout: float = 0.25
    line_sigma: float = 1.05


class TinyImageCNN(nn.Module):
    """AEC 곡선을 이미지로 그려낸 3채널 입력을 받는 4층 소형 합성곱 신경망 (평균+최댓값 풀링 결합)."""

    def __init__(self, dropout: float = 0.25) -> None:
        """4개의 합성곱+배치정규화 레이어와 평균+최댓값 풀링 결합 후 이진 분류용 선형 헤드를 구성."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """4개 합성곱+배치정규화+평균풀링 블록을 통과시킨 뒤 평균·최댓값 풀링을 합쳐 로짓 하나를 출력."""
        x = F.avg_pool2d(F.silu(self.bn1(self.conv1(x))), 2)
        x = F.avg_pool2d(F.silu(self.bn2(self.conv2(x))), 2)
        x = F.avg_pool2d(F.silu(self.bn3(self.conv3(x))), 2)
        x = F.silu(self.bn4(self.conv4(x)))
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.head(self.drop(pooled)).squeeze(1)


def set_seed(seed: int) -> None:
    """numpy/torch 난수 시드를 고정하고, GPU 사용 시 결정론적 연산 모드를 켜 재현성을 확보."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_signal_channels(x: np.ndarray) -> np.ndarray:
    """원곡선, (양 끝을 잇는 직선 대비) 중심선 잔차, 1차 도함수 3채널을 만들어 이미지화 준비."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    line = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    resid = x - line
    d1 = np.diff(x, axis=1)
    d1 = np.column_stack([d1[:, :1], d1])
    return np.stack([x, resid, d1], axis=1)


def fit_channel_limits(train_channels: np.ndarray) -> list[tuple[float, float]]:
    """train 채널별로 0.5~99.5 분위수를 이미지 래스터화 시 사용할 값 범위(lo, hi)로 계산."""
    limits = []
    for ch in range(train_channels.shape[1]):
        vals = train_channels[:, ch].ravel()
        lo, hi = np.quantile(vals[np.isfinite(vals)], [0.005, 0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1.0
        limits.append((float(lo), float(hi)))
    return limits


def rasterize_signal(signal: np.ndarray, lo: float, hi: float, cfg: CnnConfig) -> np.ndarray:
    """1차원 신호를 세로축 위치로 변환해, 가우시안 선 굵기를 가진 2차원 이미지(축 없는 곡선 그림)로 그림."""
    clipped = np.clip(signal, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    y_pos = (1.0 - scaled) * (cfg.height - 1)
    rows = np.arange(cfg.height, dtype=float)[:, None]
    img = np.exp(-0.5 * ((rows - y_pos[None, :]) / cfg.line_sigma) ** 2)
    return img.astype(np.float32)


def make_images(train_x: np.ndarray, test_x: np.ndarray, cfg: CnnConfig) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """train/test 곡선을 3채널로 변환하고, train 기준 값 범위로 둘 다 3채널 이미지로 래스터화."""
    train_channels = make_signal_channels(train_x)
    test_channels = make_signal_channels(test_x)
    limits = fit_channel_limits(train_channels)

    def convert(channels: np.ndarray) -> np.ndarray:
        """채널별로 rasterize_signal을 적용해 (N, 3, height, width) 이미지 배열을 만든다."""
        imgs = np.zeros((channels.shape[0], 3, cfg.height, cfg.width), dtype=np.float32)
        for i in range(channels.shape[0]):
            for ch, (lo, hi) in enumerate(limits):
                imgs[i, ch] = rasterize_signal(channels[i, ch], lo, hi, cfg)
        return imgs

    return convert(train_channels), convert(test_channels), limits


def stratified_folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """클래스 비율을 유지하는 5-fold 교차검증 분할(학습/검증 인덱스 쌍)을 생성."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    seed: int,
    cfg: CnnConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """한 폴드에 대해 CNN을 미니배치+조기종료로 학습하고(검증 손실 기준 최적 가중치 복원), 검증 점수와 외부 데이터 예측을 반환."""
    set_seed(seed)
    model = TinyImageCNN(dropout=cfg.dropout).to(DEVICE)
    pos = float(np.sum(y[train_idx] == 1))
    neg = float(np.sum(y[train_idx] == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    xt = torch.tensor(x[train_idx], dtype=torch.float32)
    yt = torch.tensor(y[train_idx].astype(float), dtype=torch.float32)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y[val_idx].astype(float), dtype=torch.float32, device=DEVICE)

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience_left = cfg.patience
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = xt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)
            # Very light intensity noise only; no shift/flip because anatomy is aligned.
            xb = torch.clamp(xb + 0.015 * torch.randn_like(xb), 0.0, 1.0)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_score = model(torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        ext_score = predict_torch(model, x_ext)
    return val_score, ext_score, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss)}


def predict_torch(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """학습된 모델로 배치 단위 추론을 수행해 전체 예측 점수를 이어붙여 반환."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def crossfit_cnn(x: np.ndarray, y: np.ndarray, x_ext: np.ndarray, cfg: CnnConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드(SEEDS) x 5-fold로 CNN을 반복 학습해 train OOF 점수와 외부 예측을 시드 평균으로 앙상블."""
    all_oof = []
    all_ext = []
    log_rows = []
    for seed in SEEDS:
        oof = np.zeros(len(y), dtype=float)
        ext_scores = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            val_score, ext_score, info = train_one_fold(x, y, tr, va, x_ext, seed + 100 * fold_id, cfg)
            oof[va] = val_score
            ext_scores.append(ext_score)
            log_rows.append({"seed": seed, "fold": fold_id, **info})
        all_oof.append(oof)
        all_ext.append(np.mean(ext_scores, axis=0))
    return np.mean(all_oof, axis=0), np.mean(all_ext, axis=0), pd.DataFrame(log_rows)


def auc_p_mannwhitney(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney U 검정으로 두 그룹 점수 분포가 다른지에 대한 p값을 계산 (AUC 유의성 검정과 동치)."""
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """주어진 임계값에서 정확도·민감도·특이도·균형정확도와 혼동행렬을 계산."""
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def model_metrics(dataset: str, model: str, y: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict:
    """데이터셋/모델 이름별로 AUC·AP·Brier와 (임계값 없으면 Youden 기준 자동 계산 후) 혼동행렬 지표를 한 행으로 정리."""
    if threshold is None:
        threshold = threshold_youden(y.astype(int), score)
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return {
        "dataset": dataset,
        "model": model,
        "auc": float(roc_auc_score(y, score)),
        "auc_p_mannwhitney": auc_p_mannwhitney(y, score),
        "average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(prob, 1e-6, 1.0 - 1e-6))),
        **binary_metrics(y, score, threshold),
    }


def bootstrap_auc_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, seed: int = 888, n_boot: int = 3000) -> dict:
    """두 점수(base vs candidate)의 AUC 차이를 부트스트랩으로 신뢰구간과 양측 p값과 함께 추정."""
    rng = np.random.default_rng(seed)
    obs = float(roc_auc_score(y, candidate) - roc_auc_score(y, base))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(float(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], base[idx])))
    arr = np.asarray(diffs)
    p = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)) if len(arr) else np.nan
    return {
        "delta_auc": obs,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "delta_auc_p_boot": float(min(1.0, p)) if np.isfinite(p) else np.nan,
    }


def exact_p(a: int, b: int) -> float:
    """두 카운트(a, b)에 대한 이항 정확검정(양측) p값을 계산 (McNemar류 대응비교에 사용)."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상 양성 판정을 최종(하향조정 후) 판정으로 바꿨을 때 민감도손실/특이도이득/정확도변화에 대한 대응 이항검정 p값을 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(~yy & ~clinical_pos & final_pos))
    cc = clinical_pos == yy
    fc = final_pos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
        "tp_lost_n": sens_loss,
        "fp_removed_n": spec_gain,
    }


def fisher_event_p(y: np.ndarray, kept: np.ndarray, deesc: np.ndarray) -> float:
    """유지군과 하향조정군의 사건 발생률 차이에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[kept] == 1))
    b = int(np.sum(y[kept] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def deesc_row(
    dataset: str,
    op: str,
    y: np.ndarray,
    clinical_pos: np.ndarray,
    aec_score: np.ndarray,
    aec_threshold: float,
) -> dict:
    """임상 양성군 중 CNN 점수가 임계값 이하인 환자를 하향조정군으로 분류하고, 임상 단독 대비 민감도손실/특이도이득/정확도변화와 하향조정군의 사건 통계를 계산."""
    deesc = clinical_pos & (aec_score <= aec_threshold)
    final_pos = clinical_pos & ~deesc
    base = binary_metrics(y, clinical_pos.astype(float), 0.5)
    post = binary_metrics(y, final_pos.astype(float), 0.5)
    return {
        "dataset": dataset,
        "operating_point": op,
        "aec_low_threshold": float(aec_threshold),
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_event_p(y, final_pos, deesc),
        **paired_pvalues(y, clinical_pos, final_pos),
    }


def choose_deesc_threshold(y: np.ndarray, clinical_pos: np.ndarray, aec_score: np.ndarray, op: str) -> tuple[float, dict]:
    """train 안에서 여러 CNN 하향조정 임계값 후보를 스캔해, 민감도손실 유의(p<0.05)·특이도이득 양수 조건을
    만족하면서 "선택 점수"(특이도이득+정확도개선-민감도손실 가중합)가 가장 높은 임계값을 선택."""
    values = aec_score[clinical_pos]
    qs = np.linspace(0.05, 0.45, 41)
    candidates = np.unique(np.quantile(values, qs))
    best = None
    for th in candidates:
        row = deesc_row("Gangnam internal OOF", op, y, clinical_pos, aec_score, float(th))
        if row["deesc_n"] < 20:
            continue
        if row["sensitivity_loss_p_exact"] < 0.05:
            continue
        if row["specificity_gain"] <= 0:
            continue
        # Train selection emphasizes usable de-escalation, not global AUC.
        score = (
            row["specificity_gain"]
            + 0.35 * row["delta_accuracy"]
            + 0.25 * row["delta_balanced_accuracy"]
            - 0.20 * row["sensitivity_loss"]
        )
        if np.isfinite(row["deesc_event_fisher_p"]) and row["deesc_event_fisher_p"] >= 0.05:
            score -= 0.02
        candidate = {**row, "selection_score": float(score)}
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    if best is None:
        th = float(np.quantile(values, 0.20))
        best = {**deesc_row("Gangnam internal OOF", op, y, clinical_pos, aec_score, th), "selection_score": np.nan}
    return float(best["aec_low_threshold"]), best


def plot_example_images(images: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """저근감소증군과 비저근감소증군 각각의 평균 이미지(3채널)를 나란히 그려 PNG로 저장 (CNN이 실제로 무엇을 보는지 확인)."""
    low = images[y == 1].mean(axis=0)
    non = images[y == 0].mean(axis=0)
    names = ["normalized curve", "centerline residual", "first derivative"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.3))
    for col in range(3):
        axes[0, col].imshow(non[col], cmap="magma", aspect="auto")
        axes[0, col].set_title(f"Non-low: {names[col]}", fontsize=9)
        axes[1, col].imshow(low[col], cmap="magma", aspect="auto")
        axes[1, col].set_title(f"Low SMI: {names[col]}", fontsize=9)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("Axis-free AEC morphology images used by 2D CNN", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (7/1 CNN 계열 분석의 출발점 — 질문: AEC 곡선을 "숫자 특징"이
    아니라 아예 "이미지"로 그려서 CNN에 직접 보여주면 임상변수 대비 추가 정보를 얻을 수 있는가?):

    1. g1090/sdata를 로드하고 임상점수를 준비한다.
    2. make_images로 각 환자의 AEC 곡선을 (원곡선, 중심선 잔차, 1차 도함수) 3채널의 "축 없는" 이미지로
       변환한다 (train 값 범위 기준으로 test도 동일하게 스케일링).
    3. 저근감소증군/비저근감소증군 평균 이미지를 그려 CNN이 실제로 어떤 모양을 보는지 확인.
    4. crossfit_cnn으로 여러 시드 x 5-fold로 TinyImageCNN을 학습해 train OOF와 외부 예측을 앙상블.
    5. 임상점수+CNN점수를 로지스틱으로 결합(스택)한 모델도 만들어, 임상 단독/CNN 단독/결합 3개 모델의
       AUC 등 성능을 비교하고 결합모델은 부트스트랩으로 delta AUC 신뢰구간까지 계산.
    6. 5개 민감도 목표(S80~S90)마다 choose_deesc_threshold로 train에서 CNN 하향조정 임계값을 고르고,
       임상 양성군 중 하향조정된 환자들의 사건율·통계적 유의성을 외부 데이터에서 검증.
    7. 하향조정 결과를 목표별로 요약(최소/평균 지표)해 CSV로 저장하고, 환자별 점수·이미지 스케일링
       한계값을 CSV/JSON으로 저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CnnConfig()
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    yg = g["y"].astype(int)
    ys = s["y"].astype(int)
    cg, cs, _ = clinical_scores(g, s)

    img_g, img_s, limits = make_images(g["norm"], s["norm"], cfg)
    plot_example_images(img_g, yg, OUT_DIR / "aec_morphology_image_representation_mean_examples.png")

    cnn_oof, cnn_ext, train_log = crossfit_cnn(img_g, yg, img_s, cfg)
    train_log.to_csv(OUT_DIR / "cnn_training_log.csv", index=False)

    # Stack clinical score + CNN score using train OOF scores only.
    stack = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEEDS[0])
    stack.fit(np.column_stack([cg, cnn_oof]), yg)
    fusion_oof = stack.decision_function(np.column_stack([cg, cnn_oof]))
    fusion_ext = stack.decision_function(np.column_stack([cs, cnn_ext]))

    clinical_th = threshold_youden(yg, cg)
    cnn_th = threshold_youden(yg, cnn_oof)
    fusion_th = threshold_youden(yg, fusion_oof)

    metric_rows = [
        model_metrics("Gangnam internal OOF", "clinical", yg, cg, clinical_th),
        model_metrics("Sinchon external", "clinical", ys, cs, clinical_th),
        model_metrics("Gangnam internal OOF", "aec_image_cnn", yg, cnn_oof, cnn_th),
        model_metrics("Sinchon external", "aec_image_cnn", ys, cnn_ext, cnn_th),
        {
            **model_metrics("Gangnam internal OOF", "clinical_plus_aec_image_cnn", yg, fusion_oof, fusion_th),
            **bootstrap_auc_delta(yg, cg, fusion_oof, seed=SEEDS[0] + 1),
        },
        {
            **model_metrics("Sinchon external", "clinical_plus_aec_image_cnn", ys, fusion_ext, fusion_th),
            **bootstrap_auc_delta(ys, cs, fusion_ext, seed=SEEDS[0] + 2),
        },
    ]
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "image_cnn_auc_metrics.csv", index=False)

    deesc_rows = []
    for op, target in OPS:
        cth = threshold_for_min_sensitivity(yg, cg, target)
        cpos_g = cg >= cth
        cpos_s = cs >= cth
        ath, train_row = choose_deesc_threshold(yg, cpos_g, cnn_oof, op)
        deesc_rows.append(train_row)
        deesc_rows.append(deesc_row("Sinchon external", op, ys, cpos_s, cnn_ext, ath))
    deesc = pd.DataFrame(deesc_rows)
    deesc.to_csv(OUT_DIR / "image_cnn_deescalation_metrics.csv", index=False)

    summary = (
        deesc.groupby("dataset")
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
    summary.to_csv(OUT_DIR / "image_cnn_deescalation_range_summary.csv", index=False)

    pd.DataFrame(
        {
            "y_gangnam": yg,
            "clinical_oof": cg,
            "aec_image_cnn_oof": cnn_oof,
            "clinical_plus_aec_image_cnn_oof": fusion_oof,
        }
    ).to_csv(OUT_DIR / "image_cnn_gangnam_scores.csv", index=False)
    pd.DataFrame(
        {
            "y_sinchon": ys,
            "clinical_external": cs,
            "aec_image_cnn_external": cnn_ext,
            "clinical_plus_aec_image_cnn_external": fusion_ext,
        }
    ).to_csv(OUT_DIR / "image_cnn_sinchon_scores.csv", index=False)
    (OUT_DIR / "image_scaling_limits.json").write_text(json.dumps(limits, indent=2), encoding="utf-8")

    print("\nAUC METRICS")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nDE-ESCALATION RANGE SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 main()의 전체 파이프라인(이미지 변환 -> CNN 크로스핏 학습 -> 임상/CNN/결합 성능 비교 ->
    # 민감도 목표별 하향조정 임계값 산출 및 검증)이 순서대로 수행된다.
    main()
