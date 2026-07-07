from __future__ import annotations

import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec_offset_score import clinical_raw, sigmoid  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_transformer_offset"
SEED = 20260629


@dataclass(frozen=True)
class Config:
    """미니 트랜스포머 학습에 필요한 구조·정규화·손실 가중치 하이퍼파라미터 묶음."""

    name: str
    d_model: int = 24
    n_heads: int = 2
    n_layers: int = 1
    ff_mult: int = 2
    dropout: float = 0.30
    lr: float = 8e-4
    weight_decay: float = 3e-3
    max_epochs: int = 260
    patience: int = 55
    lambda_pair: float = 0.10
    lambda_score: float = 2e-3
    lambda_alpha: float = 1e-2
    lambda_noise: float = 2e-2
    noise_sd: float = 0.025
    tau: float = 0.75
    alpha_init: float = 0.10


CONFIGS = [
    Config(
        name="tiny_pair100_fast",
        d_model=16,
        dropout=0.20,
        max_epochs=140,
        patience=30,
        lambda_pair=1.00,
        lambda_score=1e-3,
        lambda_alpha=1e-4,
        lambda_noise=0.0,
        noise_sd=0.0,
        alpha_init=0.75,
    ),
]


def set_seed(seed: int) -> None:
    """python/numpy/torch 난수 시드를 모두 고정해 재현성을 확보."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def clinical_model() -> Pipeline:
    """임상 변수 전용 로지스틱 회귀 파이프라인(표준화+정규화 거의 없음)을 생성."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def fit_clinical(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """결측을 중앙값으로 채운 뒤 임상 모델을 학습."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    model = clinical_model()
    model.fit(x, y)
    return model


def apply_clinical(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """학습된 임상 모델을 새 데이터(결측은 그 데이터 중앙값으로 대체)에 적용해 로짓 점수를 산출."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    return model.decision_function(x)


def curve_channels(x_norm: np.ndarray) -> np.ndarray:
    """정규화된 곡선을 (로그값, 1차 도함수, 2차 도함수) 3채널 시계열로 변환 — 트랜스포머 입력용."""
    logx = np.log(np.clip(np.asarray(x_norm, dtype=float), 1e-6, None))
    d1 = np.gradient(logx, axis=1)
    d2 = np.gradient(d1, axis=1)
    return np.stack([logx, d1, d2], axis=-1)


def standardize_seq_train_apply(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """train의 채널별 평균/표준편차(시점 축 포함)로 train·test 시퀀스를 함께 표준화."""
    mu = np.nanmean(xtr, axis=0, keepdims=True)
    sd = np.nanstd(xtr, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    xtr2 = np.where(np.isfinite(xtr), xtr, mu)
    xte2 = np.where(np.isfinite(xte), xte, mu)
    return (xtr2 - mu) / sd, (xte2 - mu) / sd, mu, sd


def metric_row(dataset: str, model: str, y: np.ndarray, score: np.ndarray) -> dict:
    """데이터셋/모델 이름과 점수로부터 AUC/AP/로그손실/Brier와 점수 평균·표준편차를 한 행으로 정리."""
    p = np.clip(sigmoid(score), 1e-6, 1.0 - 1e-6)
    return {
        "dataset": dataset,
        "model": model,
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "score_mean": float(np.mean(score)),
        "score_sd": float(np.std(score)),
    }


def pair_bandwidth(y: np.ndarray, clinical_score: np.ndarray) -> float:
    """양성-음성 쌍 간 임상점수 거리의 중앙값을 계산 (조건부 페어 손실의 커널 폭 h로 사용)."""
    pos = clinical_score[y == 1]
    neg = clinical_score[y == 0]
    diffs = np.abs(pos[:, None] - neg[None, :]).ravel()
    h = float(np.median(diffs[np.isfinite(diffs)]))
    return h if np.isfinite(h) and h > 1e-6 else 1.0


def conditional_pair_auc(y: np.ndarray, score: np.ndarray, clinical_score: np.ndarray, h: float) -> float:
    """임상점수가 비슷한 양성-음성 쌍에 더 큰 가중치를 줘서, "임상점수가 같을 때 순위가 맞는 정도"를 재는 가중 AUC를 계산."""
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    d_score = score[pos][:, None] - score[neg][None, :]
    d_clinical = np.abs(clinical_score[pos][:, None] - clinical_score[neg][None, :])
    weight = np.exp(-d_clinical / max(h, 1e-6))
    wins = (d_score > 0).astype(float) + 0.5 * (d_score == 0).astype(float)
    return float(np.sum(weight * wins) / np.sum(weight))


def bootstrap_delta_auc(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    """두 점수(a vs b)의 AUC 차이를 부트스트랩 재표본추출로 신뢰구간과 양방향 p값과 함께 추정."""
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        deltas.append(roc_auc_score(yi, b[idx]) - roc_auc_score(yi, a[idx]))
    arr = np.asarray(deltas, dtype=float)
    return {
        "n_boot": int(arr.size),
        "delta_auc_mean": float(np.mean(arr)),
        "delta_auc_ci2.5": float(np.quantile(arr, 0.025)),
        "delta_auc_ci97.5": float(np.quantile(arr, 0.975)),
        "p_delta_le_0": float(np.mean(arr <= 0)),
        "p_delta_ge_0": float(np.mean(arr >= 0)),
    }


def permutation_alignment_p(
    y: np.ndarray,
    clinical_score: np.ndarray,
    correction: np.ndarray,
    observed_delta: float,
    n_perm: int = 5000,
    seed: int = SEED,
) -> dict:
    """트랜스포머의 보정항(correction)을 무작위로 섞어, 관측된 AUC 개선폭이 순열분포 대비 얼마나 극단적인지 순열검정으로 확인."""
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_perm, dtype=float)
    base_auc = roc_auc_score(y, clinical_score)
    for i in range(n_perm):
        perm = rng.permutation(correction)
        deltas[i] = roc_auc_score(y, clinical_score + perm) - base_auc
    return {
        "n_perm": n_perm,
        "observed_delta_auc": float(observed_delta),
        "perm_mean_delta_auc": float(np.mean(deltas)),
        "perm_ci97.5_delta_auc": float(np.quantile(deltas, 0.975)),
        "one_sided_p_perm_delta_ge_observed": float((np.sum(deltas >= observed_delta) + 1) / (n_perm + 1)),
    }


class TinyAECTransformer(nn.Module):
    """AEC 곡선(3채널 시퀀스)을 입력받아 CLS 토큰 표현으로 잠재 점수를 뽑고, alpha·delta로 임상 로짓에
    더해질 보정항을 만드는 소형 트랜스포머 인코더."""

    def __init__(self, cfg: Config, in_channels: int = 3, seq_len: int = 128):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 1, cfg.d_model))
        self.input = nn.Linear(in_channels, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha_init))
        self.delta = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor, clinical_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CLS 토큰 표현으로 AEC 잠재점수(score)를 계산하고, delta+alpha*score를 보정항(correction)으로 임상 로짓(offset)에 더해 최종 로짓(eta)을 반환."""
        z = self.input(x)
        cls = self.cls.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos
        z = self.encoder(z)
        score = self.head(self.norm(z[:, 0])).squeeze(1)
        correction = self.delta + self.alpha * score
        eta = clinical_offset + correction
        return eta, score, correction


def conditional_pair_loss_torch(
    eta: torch.Tensor,
    y: torch.Tensor,
    clinical_offset: torch.Tensor,
    h: float,
    tau: float,
) -> torch.Tensor:
    """conditional_pair_auc와 같은 가중치(임상점수 거리 기반 커널)로, 양성 로짓이 음성 로짓보다 커지도록 유도하는 미분가능한 페어 순위 손실(softplus)을 계산."""
    pos = y > 0.5
    neg = ~pos
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        return eta.new_tensor(0.0)
    d_eta = eta[pos].unsqueeze(1) - eta[neg].unsqueeze(0)
    d_clinical = torch.abs(clinical_offset[pos].unsqueeze(1) - clinical_offset[neg].unsqueeze(0))
    weight = torch.exp(-d_clinical / max(h, 1e-6))
    loss = F.softplus(-d_eta / tau)
    return (weight * loss).sum() / weight.sum().clamp_min(1e-8)


def eval_scores(model: TinyAECTransformer, x: np.ndarray, clinical_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """평가 모드로 모델을 실행해 (결합 로짓, AEC 잠재점수, 보정항)을 numpy 배열로 반환."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        ct = torch.tensor(clinical_score, dtype=torch.float32)
        eta, score, correction = model(xt, ct)
    return eta.cpu().numpy(), score.cpu().numpy(), correction.cpu().numpy()


def train_transformer(
    xtr: np.ndarray,
    ytr: np.ndarray,
    ctr: np.ndarray,
    cfg: Config,
    seed: int,
    xva: np.ndarray | None = None,
    yva: np.ndarray | None = None,
    cva: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[TinyAECTransformer, pd.DataFrame, int, float]:
    """트랜스포머를 학습 (BCE + 조건부 페어 손실 + 점수/alpha 정규화 + 노이즈 안정성 항의 가중합).
    검증세트(xva 등)가 주어지면 매 5에폭마다 검증 AUC를 확인해 조기종료·최적 가중치 복원을 하고,
    fixed_epochs가 주어지면 검증 없이 그 에폭 수만큼 고정 학습 (최종 전체-train 재학습용)."""
    set_seed(seed)
    model = TinyAECTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr.astype(float), dtype=torch.float32)
    ct = torch.tensor(ctr, dtype=torch.float32)
    h = pair_bandwidth(ytr.astype(int), ctr)

    max_epochs = fixed_epochs if fixed_epochs is not None else cfg.max_epochs
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_auc = -math.inf
    best_val_log_loss = math.inf
    bad_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        eta, score, _ = model(xt, ct)
        bce = F.binary_cross_entropy_with_logits(eta, yt)
        pair = conditional_pair_loss_torch(eta, yt, ct, h=h, tau=cfg.tau)
        score_penalty = torch.mean(score.pow(2))
        alpha_penalty = model.alpha.pow(2)
        if cfg.lambda_noise > 0.0 and cfg.noise_sd > 0.0:
            noise = torch.randn_like(xt) * cfg.noise_sd
            _, score_noise, _ = model(xt + noise, ct)
            noise_loss = torch.mean((score_noise - score.detach()).pow(2))
        else:
            noise_loss = eta.new_tensor(0.0)
        loss = (
            bce
            + cfg.lambda_pair * pair
            + cfg.lambda_score * score_penalty
            + cfg.lambda_alpha * alpha_penalty
            + cfg.lambda_noise * noise_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == max_epochs:
            row = {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_bce": float(bce.detach().cpu()),
                "train_pair": float(pair.detach().cpu()),
                "train_score_penalty": float(score_penalty.detach().cpu()),
                "train_noise": float(noise_loss.detach().cpu()),
                "alpha": float(model.alpha.detach().cpu()),
                "delta": float(model.delta.detach().cpu()),
                "pair_bandwidth_h": float(h),
            }
            if xva is not None and yva is not None and cva is not None:
                va_score, _, _ = eval_scores(model, xva, cva)
                va_prob = np.clip(sigmoid(va_score), 1e-6, 1.0 - 1e-6)
                row["valid_auc"] = float(roc_auc_score(yva, va_score))
                row["valid_log_loss"] = float(log_loss(yva, va_prob))
                row["valid_delta_auc_vs_clinical"] = float(roc_auc_score(yva, va_score) - roc_auc_score(yva, cva))
                improved = (row["valid_auc"] > best_val_auc + 1e-4) or (
                    abs(row["valid_auc"] - best_val_auc) <= 1e-4 and row["valid_log_loss"] < best_val_log_loss
                )
                if improved:
                    best_val_auc = row["valid_auc"]
                    best_val_log_loss = row["valid_log_loss"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 5
            history.append(row)
            if fixed_epochs is None and xva is not None and bad_epochs >= cfg.patience:
                break

    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = max_epochs
    return model, pd.DataFrame(history), best_epoch, h


def run_config(cfg: Config, train: dict, test: dict) -> dict:
    """한 하이퍼파라미터 설정(cfg)에 대해: 5-fold로 (임상 오프셋 고정) 트랜스포머를 학습해 train
    OOF 점수를 만들고, 폴드별 최적 에폭의 중앙값으로 전체 train에 재학습해 외부 데이터를 예측한 뒤,
    성능표·부트스트랩 델타 AUC·순열검정·환자별 예측값까지 모두 모아 결과 딕셔너리로 반환."""
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    xtr_raw = curve_channels(train["x"])
    xte_raw = curve_channels(test["x"])
    clinical_xtr = clinical_raw(train["meta"])
    clinical_xte = clinical_raw(test["meta"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clinical_oof = np.zeros(len(ytr), dtype=float)
    combined_oof = np.zeros(len(ytr), dtype=float)
    aec_score_oof = np.zeros(len(ytr), dtype=float)
    correction_oof = np.zeros(len(ytr), dtype=float)
    fold_rows = []
    histories = []

    for fold_id, (idx_tr, idx_va) in enumerate(skf.split(xtr_raw, ytr), start=1):
        clinical = fit_clinical(clinical_xtr[idx_tr], ytr[idx_tr])
        c_tr = apply_clinical(clinical, clinical_xtr[idx_tr])
        c_va = apply_clinical(clinical, clinical_xtr[idx_va])
        x_tr_s, x_va_s, _, _ = standardize_seq_train_apply(xtr_raw[idx_tr], xtr_raw[idx_va])

        model, hist, best_epoch, h = train_transformer(
            x_tr_s,
            ytr[idx_tr],
            c_tr,
            cfg,
            seed=SEED + fold_id * 101,
            xva=x_va_s,
            yva=ytr[idx_va],
            cva=c_va,
        )
        eta_va, score_va, corr_va = eval_scores(model, x_va_s, c_va)
        clinical_oof[idx_va] = c_va
        combined_oof[idx_va] = eta_va
        aec_score_oof[idx_va] = score_va
        correction_oof[idx_va] = corr_va
        histories.append(hist.assign(config=cfg.name, fold=fold_id))
        fold_rows.append(
            {
                "config": cfg.name,
                "fold": fold_id,
                "best_epoch": int(best_epoch),
                "pair_bandwidth_h": float(h),
                "clinical_auc": float(roc_auc_score(ytr[idx_va], c_va)),
                "combined_auc": float(roc_auc_score(ytr[idx_va], eta_va)),
                "delta_auc": float(roc_auc_score(ytr[idx_va], eta_va) - roc_auc_score(ytr[idx_va], c_va)),
                "correction_sd": float(np.std(corr_va)),
            }
        )

    median_epoch = int(np.median([r["best_epoch"] for r in fold_rows]))
    median_epoch = max(20, median_epoch)

    final_clinical = fit_clinical(clinical_xtr, ytr)
    c_tr_full = apply_clinical(final_clinical, clinical_xtr)
    c_te = apply_clinical(final_clinical, clinical_xte)
    xtr_s, xte_s, _, _ = standardize_seq_train_apply(xtr_raw, xte_raw)
    final_model, final_hist, _, h_full = train_transformer(
        xtr_s,
        ytr,
        c_tr_full,
        cfg,
        seed=SEED + 999,
        fixed_epochs=median_epoch,
    )
    combined_te, aec_score_te, correction_te = eval_scores(final_model, xte_s, c_te)
    combined_tr_full, aec_score_tr_full, correction_tr_full = eval_scores(final_model, xtr_s, c_tr_full)

    perf_rows = [
        metric_row("g1090_oof", f"{cfg.name}_clinical", ytr, clinical_oof),
        metric_row("g1090_oof", f"{cfg.name}_clinical_plus_aec_transformer", ytr, combined_oof),
        metric_row("sdata_external", f"{cfg.name}_clinical", yte, c_te),
        metric_row("sdata_external", f"{cfg.name}_clinical_plus_aec_transformer", yte, combined_te),
    ]
    h_eval = pair_bandwidth(ytr, clinical_oof)
    for row in perf_rows:
        if row["dataset"] == "g1090_oof":
            score = clinical_oof if row["model"].endswith("_clinical") else combined_oof
            clinical_score = clinical_oof
            y = ytr
        else:
            score = c_te if row["model"].endswith("_clinical") else combined_te
            clinical_score = c_te
            y = yte
        row["conditional_pair_auc_h"] = float(h_eval)
        row["conditional_pair_auc"] = conditional_pair_auc(y, score, clinical_score, h_eval)

    observed_delta_ext = roc_auc_score(yte, combined_te) - roc_auc_score(yte, c_te)
    observed_delta_oof = roc_auc_score(ytr, combined_oof) - roc_auc_score(ytr, clinical_oof)
    bootstrap_ext = bootstrap_delta_auc(yte, c_te, combined_te, seed=SEED + 31)
    bootstrap_oof = bootstrap_delta_auc(ytr, clinical_oof, combined_oof, seed=SEED + 32)
    perm_ext = permutation_alignment_p(yte, c_te, correction_te, observed_delta_ext, seed=SEED + 33)
    perm_oof = permutation_alignment_p(ytr, clinical_oof, correction_oof, observed_delta_oof, seed=SEED + 34)

    pred_oof = train["meta"][["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "SMI"]].copy()
    pred_oof["y_low_smi"] = ytr
    pred_oof["clinical_score_oof"] = clinical_oof
    pred_oof["aec_transformer_latent_oof"] = aec_score_oof
    pred_oof["aec_transformer_correction_oof"] = correction_oof
    pred_oof["combined_score_oof"] = combined_oof

    pred_te = test["meta"][["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "SMI"]].copy()
    pred_te["y_low_smi"] = yte
    pred_te["clinical_score_external"] = c_te
    pred_te["aec_transformer_latent_external"] = aec_score_te
    pred_te["aec_transformer_correction_external"] = correction_te
    pred_te["combined_score_external"] = combined_te

    pred_tr_full = train["meta"][["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "SMI"]].copy()
    pred_tr_full["y_low_smi"] = ytr
    pred_tr_full["clinical_score_full_fit"] = c_tr_full
    pred_tr_full["aec_transformer_latent_full_fit"] = aec_score_tr_full
    pred_tr_full["aec_transformer_correction_full_fit"] = correction_tr_full
    pred_tr_full["combined_score_full_fit"] = combined_tr_full

    return {
        "config": cfg,
        "folds": pd.DataFrame(fold_rows),
        "history": pd.concat(histories + [final_hist.assign(config=cfg.name, fold="final_all_g1090")], ignore_index=True),
        "performance": pd.DataFrame(perf_rows),
        "pred_oof": pred_oof,
        "pred_external": pred_te,
        "pred_train_full": pred_tr_full,
        "bootstrap": pd.DataFrame(
            [
                {"config": cfg.name, "dataset": "g1090_oof", **bootstrap_oof},
                {"config": cfg.name, "dataset": "sdata_external", **bootstrap_ext},
            ]
        ),
        "permutation": pd.DataFrame(
            [
                {"config": cfg.name, "dataset": "g1090_oof", **perm_oof},
                {"config": cfg.name, "dataset": "sdata_external", **perm_ext},
            ]
        ),
        "summary": {
            "config": cfg.__dict__,
            "median_epoch_from_oof": median_epoch,
            "full_train_pair_bandwidth_h": float(h_full),
            "oof_delta_auc": float(observed_delta_oof),
            "external_delta_auc": float(observed_delta_ext),
        },
    }


def plot_performance(perf: pd.DataFrame) -> None:
    """설정x데이터셋별로 임상 단독 vs 결합모델 AUC를 나란히 막대그래프로 비교해 PNG로 저장."""
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = []
    x = []
    clinical = []
    combined = []
    for i, (cfg, dataset) in enumerate([(c, d) for c in perf["config"].unique() for d in ["g1090_oof", "sdata_external"]]):
        sub = perf[(perf["config"].eq(cfg)) & (perf["dataset"].eq(dataset))]
        labels.append(f"{cfg}\n{dataset}")
        x.append(i)
        clinical.append(float(sub[sub["model_type"].eq("clinical")]["auc"].iloc[0]))
        combined.append(float(sub[sub["model_type"].eq("combined")]["auc"].iloc[0]))
    xarr = np.asarray(x)
    ax.bar(xarr - 0.18, clinical, width=0.36, label="Clinical", color="#4C78A8")
    ax.bar(xarr + 0.18, combined, width=0.36, label="Clinical + AEC transformer", color="#F58518")
    ax.set_xticks(xarr)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.50, min(0.95, max(max(clinical), max(combined)) + 0.04))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_transformer_auc_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 손으로 설계한 특징이 아니라, 작은 트랜스포머가 AEC 곡선
    전체를 직접 보고 학습한 "잠재 점수"를 임상 로짓에 더하면 판별력이 개선되는가?):

    1. g1090/sdata를 로드하고, CONFIGS에 정의된 트랜스포머 설정(현재는 tiny_pair100_fast 1개)마다
       run_config를 실행한다.
    2. run_config는 5-fold 안에서: 임상 로짓을 고정 오프셋으로 두고, TinyAECTransformer가 AEC
       3채널 시퀀스(로그값+1차+2차 도함수)를 보고 만든 보정항(delta+alpha*score)을 더해 결합
       로짓을 만든다. 손실은 BCE + "임상점수가 비슷한 쌍끼리는 순위도 맞아야 한다"는 조건부 페어
       손실 + 정규화항들의 합이다. 폴드별 최적 에폭의 중앙값으로 전체 train에 재학습해 외부
       데이터를 예측한다.
    3. 임상 단독 vs 결합모델의 성능(AUC/AP/로그손실/Brier/조건부페어AUC)을 비교하고, 부트스트랩으로
       delta AUC 신뢰구간을, 순열검정으로 "보정항이 우연히 이 정도 개선을 만들 수 있는지"의 p값을 구한다.
    4. 환자별 예측값(train OOF/외부/전체재학습)을 CSV로 저장하고, 폴드별 요약·학습 곡선(history)·
       부트스트랩·순열검정 결과를 모두 CSV로 저장.
    5. 설정별 임상 vs 결합 AUC를 막대그래프로 비교해 저장하고, 손실 함수 정의와 전체 결과를
       JSON으로 저장한 뒤 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    train = load_aec128(FILES["g1090"])
    test = load_aec128(FILES["sdata"])
    train["y"] = train["y"].astype(int)
    test["y"] = test["y"].astype(int)

    all_perf = []
    all_folds = []
    all_history = []
    all_boot = []
    all_perm = []
    summaries = []

    for cfg in CONFIGS:
        result = run_config(cfg, train, test)
        perf = result["performance"].copy()
        perf["config"] = cfg.name
        perf["model_type"] = np.where(perf["model"].str.contains("plus_aec"), "combined", "clinical")
        all_perf.append(perf)
        all_folds.append(result["folds"])
        all_history.append(result["history"])
        all_boot.append(result["bootstrap"])
        all_perm.append(result["permutation"])
        summaries.append(result["summary"])
        result["pred_oof"].to_csv(OUT_DIR / f"{cfg.name}_g1090_oof_predictions.csv", index=False)
        result["pred_external"].to_csv(OUT_DIR / f"{cfg.name}_sdata_external_predictions.csv", index=False)
        result["pred_train_full"].to_csv(OUT_DIR / f"{cfg.name}_g1090_full_fit_predictions.csv", index=False)

    perf_df = pd.concat(all_perf, ignore_index=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    hist_df = pd.concat(all_history, ignore_index=True)
    boot_df = pd.concat(all_boot, ignore_index=True)
    perm_df = pd.concat(all_perm, ignore_index=True)

    perf_df.to_csv(OUT_DIR / "aec128_transformer_performance.csv", index=False)
    fold_df.to_csv(OUT_DIR / "aec128_transformer_fold_summary.csv", index=False)
    hist_df.to_csv(OUT_DIR / "aec128_transformer_training_history.csv", index=False)
    boot_df.to_csv(OUT_DIR / "aec128_transformer_bootstrap_delta_auc.csv", index=False)
    perm_df.to_csv(OUT_DIR / "aec128_transformer_permutation_alignment.csv", index=False)
    plot_performance(perf_df)

    loss_definition = {
        "final_logit": "eta_i = clinical_logit_i + delta + alpha * transformer_score(AEC_128_i)",
        "total_loss": "BCEWithLogits(eta, y) + lambda_pair * conditional_pairwise_rank_loss + lambda_score * mean(score^2) + lambda_alpha * alpha^2 + lambda_noise * mean((score(AEC+noise)-score(AEC))^2)",
        "conditional_pairwise_rank_loss": "weighted mean over event/non-event pairs of softplus(-(eta_event - eta_nonevent)/tau)",
        "pair_weight": "K_ij = exp(-abs(clinical_logit_i - clinical_logit_j) / h), h = median event/non-event clinical-logit distance in training data",
        "input": "patient-normalized AEC_128 only; channels are log(AEC/row_mean), first derivative, second derivative",
        "validation": "g1090 5-fold OOF; final model trained on all g1090 for median best OOF epoch and tested once on sdata",
    }
    summary = {
        "seed": SEED,
        "configs": summaries,
        "loss_definition": loss_definition,
        "outputs": {
            "performance_csv": str(OUT_DIR / "aec128_transformer_performance.csv"),
            "fold_summary_csv": str(OUT_DIR / "aec128_transformer_fold_summary.csv"),
            "bootstrap_csv": str(OUT_DIR / "aec128_transformer_bootstrap_delta_auc.csv"),
            "permutation_csv": str(OUT_DIR / "aec128_transformer_permutation_alignment.csv"),
            "plot": str(OUT_DIR / "aec128_transformer_auc_comparison.png"),
        },
    }
    (OUT_DIR / "aec128_transformer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Performance")
    print(perf_df[["config", "dataset", "model_type", "auc", "average_precision", "log_loss", "brier", "conditional_pair_auc"]])
    print("\nBootstrap delta AUC")
    print(boot_df)
    print("\nPermutation alignment")
    print(perm_df)
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
