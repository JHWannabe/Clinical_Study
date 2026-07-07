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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, clinical_scores, load_dataset  # noqa: E402
from aec_region_constrained_cnn_gate import (  # noqa: E402
    DEVICE,
    make_channels,
    set_seed,
    standardize_channels_train_apply,
    stratified_folds,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_guided_auc_cnn"
SEEDS = [20260701, 20260711]
BOOT_N = 2000


REGION_SETS: dict[str, dict[str, tuple[int, int]]] = {
    "late4": {
        "R1_late_fall_076_092": (76, 92),
        "R2_late_slope_088_106": (88, 106),
        "R3_tail_curve_096_118": (96, 118),
        "R4_tail_rebound_090_110": (90, 110),
    },
    "tail8": {
        "T1_065_080": (65, 80),
        "T2_073_088": (73, 88),
        "T3_081_096": (81, 96),
        "T4_089_104": (89, 104),
        "T5_097_112": (97, 112),
        "T6_105_120": (105, 120),
        "T7_113_128": (113, 128),
        "T8_late_context_081_128": (81, 128),
    },
    "anatomy6": {
        "A1_pelvis_001_024": (1, 24),
        "A2_lower_abd_025_048": (25, 48),
        "A3_mid_abd_049_072": (49, 72),
        "A4_upper_mid_073_096": (73, 96),
        "A5_upper_abd_097_112": (97, 112),
        "A6_liver_dome_113_128": (113, 128),
    },
    "multiscale10": {
        "M1_001_032": (1, 32),
        "M2_017_048": (17, 48),
        "M3_033_064": (33, 64),
        "M4_049_080": (49, 80),
        "M5_065_096": (65, 96),
        "M6_081_112": (81, 112),
        "M7_097_128": (97, 128),
        "M8_lower_half_001_064": (1, 64),
        "M9_mid_context_033_096": (33, 96),
        "M10_upper_half_065_128": (65, 128),
    },
}


@dataclass(frozen=True)
class AucConfig:
    name: str
    region_set: str
    use_clinical: bool
    anchor_clinical: bool = False
    hidden: int = 12
    embed: int = 12
    dropout: float = 0.30
    lr: float = 6.0e-4
    weight_decay: float = 3.0e-3
    rank_weight: float = 0.45
    noise_sd: float = 0.01
    max_epochs: int = 150
    patience: int = 18
    batch_size: int = 96


CONFIGS = [
    AucConfig("late4_aec_only", "late4", False),
    AucConfig("late4_clinical_aware", "late4", True, dropout=0.25, rank_weight=0.30),
    AucConfig("late4_clinical_anchor", "late4", True, anchor_clinical=True, dropout=0.25, rank_weight=0.80),
    AucConfig("tail8_aec_only", "tail8", False),
    AucConfig("tail8_clinical_aware", "tail8", True, dropout=0.25, rank_weight=0.30),
    AucConfig("tail8_clinical_anchor", "tail8", True, anchor_clinical=True, dropout=0.25, rank_weight=0.80),
    AucConfig("anatomy6_aec_only", "anatomy6", False),
    AucConfig("anatomy6_clinical_aware", "anatomy6", True, dropout=0.25, rank_weight=0.30),
    AucConfig("anatomy6_clinical_anchor", "anatomy6", True, anchor_clinical=True, dropout=0.25, rank_weight=0.80),
    AucConfig("multiscale10_aec_only", "multiscale10", False, dropout=0.35),
    AucConfig("multiscale10_clinical_aware", "multiscale10", True, dropout=0.30, rank_weight=0.30),
    AucConfig("multiscale10_clinical_anchor", "multiscale10", True, anchor_clinical=True, dropout=0.30, rank_weight=0.80),
]


class RegionEncoder(nn.Module):
    """AEC 곡선의 한 구간(region)을 입력받아 소형 1D CNN으로 특징맵을 뽑고, 평균/최댓값/표준편차 풀링 후 임베딩 벡터로 투영하는 인코더."""

    def __init__(self, in_channels: int, hidden: int, embed: int, dropout: float) -> None:
        """구간 입력을 처리할 합성곱 몸통(net)과, 풀링된 특징을 embed 차원으로 줄이는 투영 레이어(proj)를 초기화."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 3, embed),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """구간 입력을 합성곱한 뒤 평균/최댓값/표준편차 3가지로 풀링하고 투영해 이 구간의 임베딩 벡터를 계산."""
        z = self.net(x)
        pooled = torch.cat(
            [
                z.mean(dim=2),
                z.amax(dim=2),
                z.std(dim=2, unbiased=False),
            ],
            dim=1,
        )
        return self.proj(pooled)


class RegionGuidedAucCnn(nn.Module):
    """설정된 구간 집합(REGION_SETS)마다 RegionEncoder를 두어 임베딩을 뽑고, 필요하면 임상점수까지 이어붙여 MLP head로 AUC용 위험점수를 예측 — clinical_anchor 모드에서는 CNN이 임상점수 주변의 보정치(잔차)만 학습하도록 제약."""

    def __init__(self, cfg: AucConfig) -> None:
        """설정된 구간 집합의 구간 수만큼 RegionEncoder를 만들고, (임상점수 포함 여부에 따라 입력 차원이 달라지는) MLP head를 구성. anchor_clinical이면 임상점수에 더할 보정 크기(scale)와 편향(bias) 파라미터도 추가."""
        super().__init__()
        self.cfg = cfg
        self.regions = list(REGION_SETS[cfg.region_set].items())
        self.encoders = nn.ModuleList(
            [RegionEncoder(3, cfg.hidden, cfg.embed, cfg.dropout) for _ in self.regions]
        )
        in_dim = cfg.embed * len(self.regions) + (1 if cfg.use_clinical else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(16, 1),
        )
        if cfg.anchor_clinical:
            self.anchor_bias = nn.Parameter(torch.zeros(1))
            self.anchor_scale_raw = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor, clinical_z: torch.Tensor | None = None) -> torch.Tensor:
        """각 구간 인코더의 임베딩(및 use_clinical이면 임상점수)을 이어붙여 head에 통과시키고, anchor_clinical 모드면 그 출력을 tanh로 제한한 뒤 스케일링해 임상점수에 더한 값을, 아니면 head 출력 자체를 위험 로짓으로 반환."""
        parts = []
        for enc, (_, (start, end)) in zip(self.encoders, self.regions):
            parts.append(enc(x[:, :, start - 1 : end]))
        if self.cfg.use_clinical:
            if clinical_z is None:
                raise ValueError("clinical_z is required for clinical-aware config")
            parts.append(clinical_z[:, None])
        residual = self.head(torch.cat(parts, dim=1)).squeeze(1)
        if self.cfg.anchor_clinical:
            if clinical_z is None:
                raise ValueError("clinical_z is required for clinical-anchor config")
            scale = torch.clamp(F.softplus(self.anchor_scale_raw), max=0.75)
            return clinical_z + self.anchor_bias.squeeze(0) + scale.squeeze(0) * torch.tanh(residual)
        return residual


def auc_raw_with_p(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """점수로 AUC를 계산하고, Mann-Whitney U 검정으로 두 그룹(양성/음성) 점수 분포 차이의 p값도 함께 반환."""
    auc = float(roc_auc_score(y, score))
    p = float(stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").pvalue)
    return auc, p


def paired_auc_delta_bootstrap(
    y: np.ndarray,
    score_new: np.ndarray,
    score_ref: np.ndarray,
    seed: int,
    n_boot: int = BOOT_N,
) -> tuple[float, float, float, float]:
    """새 점수와 기준(임상) 점수의 AUC 차이를 관측하고, 환자를 복원추출로 재표본화한 부트스트랩 분포로 그 차이의 양측 p값과 95% 신뢰구간을 계산."""
    obs = float(roc_auc_score(y, score_new) - roc_auc_score(y, score_ref))
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], score_new[idx]) - roc_auc_score(y[idx], score_ref[idx])))
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return obs, np.nan, np.nan, np.nan
    ci_low, ci_high = np.quantile(arr, [0.025, 0.975])
    p = 2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))
    return obs, float(min(1.0, p)), float(ci_low), float(ci_high)


def ranking_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """모든 양성-음성 쌍에 대해 양성의 로짓이 음성보다 커지도록 유도하는 pairwise 순위 손실(softplus 기반 힌지 근사) — AUC를 직접 겨냥한 보조 손실."""
    pos = logit[target > 0.5]
    neg = logit[target <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logit.new_tensor(0.0)
    return F.softplus(-(pos[:, None] - neg[None, :])).mean()


def train_loss(logit: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, cfg: AucConfig) -> torch.Tensor:
    """클래스 불균형을 보정한 가중 이진교차엔트로피와 순위손실(ranking_loss)을 cfg.rank_weight 비율로 합쳐 최종 학습 손실을 계산."""
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
    return bce + cfg.rank_weight * ranking_loss(logit, target)


def predict_model(
    model: RegionGuidedAucCnn,
    x: np.ndarray,
    clinical_z: np.ndarray | None,
    batch_size: int = 512,
) -> np.ndarray:
    """배치 단위로 모델을 평가모드로 실행해 전체 데이터에 대한 위험 로짓 점수를 반환."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            cb = None
            if clinical_z is not None:
                cb = torch.tensor(clinical_z[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            out.append(model(xb, cb).detach().cpu().numpy())
    return np.concatenate(out)


def train_one_fold(
    cfg: AucConfig,
    x: np.ndarray,
    clinical_z: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    clinical_ext: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """한 fold에 대해 RegionGuidedAucCnn을 조기종료로 학습(클래스 불균형에 맞춘 pos_weight 적용, 입력에 가우시안 노이즈 추가)시키고, 검증셋(OOF)과 외부셋에 대한 위험점수 및 학습 로그를 반환."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = RegionGuidedAucCnn(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y[train_idx].astype(np.float32), dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(clinical_z[train_idx], dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y[val_idx].astype(np.float32), dtype=torch.float32, device=DEVICE)
    cv = torch.tensor(clinical_z[val_idx], dtype=torch.float32, device=DEVICE)

    pos = max(float(np.sum(y[train_idx] == 1)), 1.0)
    neg = max(float(np.sum(y[train_idx] == 0)), 1.0)
    pos_weight = torch.tensor(min(neg / pos, 15.0), dtype=torch.float32, device=DEVICE)

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = cfg.patience
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            xb = xt[batch]
            if cfg.noise_sd > 0:
                xb = xb + cfg.noise_sd * torch.randn_like(xb)
            cb = ct[batch] if cfg.use_clinical else None
            opt.zero_grad(set_to_none=True)
            loss = train_loss(model(xb, cb), yt[batch], pos_weight, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            cbv = cv if cfg.use_clinical else None
            val_loss = float(train_loss(model(xv, cbv), yv, pos_weight, cfg).item())
        if val_loss < best_loss - 1e-4:
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
    val_score = predict_model(model, x[val_idx], clinical_z[val_idx] if cfg.use_clinical else None)
    ext_score = predict_model(model, x_ext, clinical_ext if cfg.use_clinical else None)
    return val_score, ext_score, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss), "pos_weight": float(pos_weight.item())}


def crossfit_config(
    cfg: AucConfig,
    xg: np.ndarray,
    yg: np.ndarray,
    cg: np.ndarray,
    xs: np.ndarray,
    cs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드 x 5-fold로 train_one_fold를 반복해, 시드에 걸쳐 평균낸 내부 OOF/외부 위험점수와 전체 학습 로그를 반환."""
    oof_runs = []
    ext_runs = []
    logs = []
    for seed in SEEDS:
        oof = np.zeros(len(yg), dtype=float)
        ext_folds = []
        for fold_id, (tr, va) in enumerate(stratified_folds(yg, seed)):
            val_score, ext_score, info = train_one_fold(cfg, xg, cg, yg, tr, va, xs, cs, seed + fold_id * 101)
            oof[va] = val_score
            ext_folds.append(ext_score)
            logs.append({"config": cfg.name, "seed": seed, "fold": fold_id, **info})
        oof_runs.append(oof)
        ext_runs.append(np.mean(ext_folds, axis=0))
    return np.mean(oof_runs, axis=0), np.mean(ext_runs, axis=0), pd.DataFrame(logs)


def stacked_clinical_plus_aec(
    y: np.ndarray,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    aec_oof: np.ndarray,
    aec_ext: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """임상점수와 AEC(CNN) 점수를 함께 넣은 로지스틱 결합모델을 5-fold OOF로 학습해 내부 OOF/외부 결합 점수를 반환."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260701)
    oof = np.zeros(len(y), dtype=float)
    x = np.column_stack([clinical_oof, aec_oof])
    for fold_id, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260701 + fold_id)
        model.fit(x[tr], y[tr])
        oof[va] = model.decision_function(x[va])
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260799)
    model.fit(x, y)
    ext = model.decision_function(np.column_stack([clinical_ext, aec_ext]))
    return oof, ext


def metric_row(
    config: str,
    model: str,
    region_set: str,
    uses_clinical_in_cnn: bool,
    clinical_anchor: bool,
    score_g: np.ndarray,
    score_s: np.ndarray,
    y_g: np.ndarray,
    y_s: np.ndarray,
    clinical_g: np.ndarray,
    clinical_s: np.ndarray,
    seed_offset: int,
) -> dict:
    """한 모델(설정)에 대해 내부/외부 AUC(및 p값)와, 임상점수 대비 부트스트랩 AUC 증분(및 p값, 신뢰구간)을 계산해 결과표 한 행으로 정리."""
    ig_auc, ig_p = auc_raw_with_p(y_g, score_g)
    es_auc, es_p = auc_raw_with_p(y_s, score_s)
    idelta, idelta_p, idelta_ci_l, idelta_ci_u = paired_auc_delta_bootstrap(
        y_g, score_g, clinical_g, 20260701 + seed_offset
    )
    edelta, edelta_p, edelta_ci_l, edelta_ci_u = paired_auc_delta_bootstrap(
        y_s, score_s, clinical_s, 20261701 + seed_offset
    )
    return {
        "config": config,
        "model": model,
        "region_set": region_set,
        "uses_clinical_in_cnn": uses_clinical_in_cnn,
        "clinical_anchor": clinical_anchor,
        "internal_auc": ig_auc,
        "internal_auc_p": ig_p,
        "external_auc": es_auc,
        "external_auc_p": es_p,
        "internal_delta_vs_clinical": idelta,
        "internal_delta_p_bootstrap": idelta_p,
        "internal_delta_ci_low": idelta_ci_l,
        "internal_delta_ci_high": idelta_ci_u,
        "external_delta_vs_clinical": edelta,
        "external_delta_p_bootstrap": edelta_p,
        "external_delta_ci_low": edelta_ci_l,
        "external_delta_ci_high": edelta_ci_u,
    }


def plot_auc(summary: pd.DataFrame, out_path: Path) -> None:
    """임상단독을 제외한 모든 모델의 내부/외부 AUC를 외부 AUC 내림차순으로 가로 막대그래프로 그리고, 임상단독 외부 AUC를 기준선으로 표시해 PNG로 저장."""
    rows = summary[summary["model"].ne("clinical_only")].copy()
    rows = rows.sort_values("external_auc", ascending=False)
    labels = rows["model"].tolist()
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(rows))), constrained_layout=True)
    ax.barh(y - 0.18, rows["internal_auc"], height=0.34, color="#4c78a8", label="Internal/Gangnam")
    ax.barh(y + 0.18, rows["external_auc"], height=0.34, color="#f58518", label="External/Sinchon")
    clinical_ext = float(summary.loc[summary["model"].eq("clinical_only"), "external_auc"].iloc[0])
    ax.axvline(clinical_ext, color="black", lw=1.2, ls="--", label=f"Clinical external {clinical_ext:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.45, max(0.88, rows[["internal_auc", "external_auc"]].max().max() + 0.03))
    ax.set_xlabel("AUC")
    ax.set_title("Region-guided CNN AUC", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: de-escalation 게이트가 아니라 순수하게 "AUC를 임상단독보다
    올릴 수 있는가"에 집중해, 4가지 구간 분할 방식(late4/tail8/anatomy6/multiscale10) x 3가지
    임상정보 활용 방식(AEC만/임상인지형/임상앵커형)의 CNN들을 비교):

    1. g1090/sdata를 로드하고 3채널(곡선/기울기/곡률) 텐서를 준비.
    2. CONFIGS의 12가지 설정마다 RegionGuidedAucCnn을 여러 시드 x 5-fold로 학습(crossfit_config)해
       내부 OOF/외부 위험점수를 얻고, AEC-only 설정에는 임상점수와의 로지스틱 결합 점수
       (stacked_clinical_plus_aec)도 추가로 계산.
    3. 임상단독을 포함해 모든 모델(설정별 CNN 단독 점수, 결합 점수)의 내부/외부 AUC와, 임상단독 대비
       부트스트랩 기반 AUC 증분(p값, 신뢰구간)을 계산.
    4. 외부 AUC 기준으로 정렬한 요약표를 그래프로 시각화.
    5. 요약표/원점수/학습로그를 CSV로, 구간 분할 정의를 JSON으로 저장한 뒤 콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    xg = make_channels(g["norm"])
    xs = make_channels(s["norm"])
    xg, xs = standardize_channels_train_apply(xg, xs)

    rows = []
    score_rows = pd.DataFrame(
        {
            "dataset": ["g1090_internal"] * len(g["y"]) + ["sdata_external"] * len(s["y"]),
            "row_index": list(range(len(g["y"]))) + list(range(len(s["y"]))),
            "y_low_smi": np.r_[g["y"], s["y"]],
            "clinical_score": np.r_[clinical_oof, clinical_ext],
            "clinical_z": np.r_[c_g, c_s],
        }
    )
    cg_auc, cg_p = auc_raw_with_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_raw_with_p(s["y"], clinical_ext)
    rows.append(
        {
            "config": "clinical_only",
            "model": "clinical_only",
            "region_set": "none",
            "uses_clinical_in_cnn": True,
            "clinical_anchor": False,
            "internal_auc": cg_auc,
            "internal_auc_p": cg_p,
            "external_auc": cs_auc,
            "external_auc_p": cs_p,
            "internal_delta_vs_clinical": 0.0,
            "internal_delta_p_bootstrap": np.nan,
            "internal_delta_ci_low": np.nan,
            "internal_delta_ci_high": np.nan,
            "external_delta_vs_clinical": 0.0,
            "external_delta_p_bootstrap": np.nan,
            "external_delta_ci_low": np.nan,
            "external_delta_ci_high": np.nan,
        }
    )

    all_logs = []
    for i, cfg in enumerate(CONFIGS, start=1):
        print(f"[{i}/{len(CONFIGS)}] {cfg.name}", flush=True)
        score_g, score_s, logs = crossfit_config(cfg, xg, g["y"].astype(int), c_g, xs, c_s)
        all_logs.append(logs)
        score_rows.loc[score_rows["dataset"].eq("g1090_internal"), cfg.name] = score_g
        score_rows.loc[score_rows["dataset"].eq("sdata_external"), cfg.name] = score_s
        rows.append(
            metric_row(
                cfg.name,
                f"{cfg.name}_score",
                cfg.region_set,
                cfg.use_clinical,
                cfg.anchor_clinical,
                score_g,
                score_s,
                g["y"].astype(int),
                s["y"].astype(int),
                clinical_oof,
                clinical_ext,
                i * 10,
            )
        )
        if not cfg.use_clinical:
            stack_g, stack_s = stacked_clinical_plus_aec(
                g["y"].astype(int), clinical_oof, clinical_ext, score_g, score_s
            )
            stack_name = f"clinical_plus_{cfg.name}"
            score_rows.loc[score_rows["dataset"].eq("g1090_internal"), stack_name] = stack_g
            score_rows.loc[score_rows["dataset"].eq("sdata_external"), stack_name] = stack_s
            rows.append(
                metric_row(
                    cfg.name,
                    stack_name,
                    cfg.region_set,
                    False,
                    False,
                    stack_g,
                    stack_s,
                    g["y"].astype(int),
                    s["y"].astype(int),
                    clinical_oof,
                    clinical_ext,
                    i * 10 + 1,
                )
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["external_auc", "internal_auc"], ascending=False).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "region_guided_auc_summary.csv", index=False)
    score_rows.to_csv(OUT_DIR / "region_guided_scores.csv", index=False)
    if all_logs:
        pd.concat(all_logs, ignore_index=True).to_csv(OUT_DIR / "region_guided_training_log.csv", index=False)
    plot_auc(summary, OUT_DIR / "region_guided_auc_plot.png")
    with open(OUT_DIR / "region_guided_auc_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization, then channel-wise z standardization",
                "target": "low SMI",
                "internal": "g1090/Gangnam cross-fitted OOF",
                "external": "sdata/Sinchon held-out",
                "seeds": SEEDS,
                "region_sets": REGION_SETS,
            },
            f,
            indent=2,
        )

    show = summary[
        [
            "model",
            "region_set",
            "uses_clinical_in_cnn",
            "clinical_anchor",
            "internal_auc",
            "internal_auc_p",
            "internal_delta_vs_clinical",
            "internal_delta_p_bootstrap",
            "external_auc",
            "external_auc_p",
            "external_delta_vs_clinical",
            "external_delta_p_bootstrap",
        ]
    ]
    print("\nREGION-GUIDED AUC SUMMARY")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    # 데이터 로드 -> 4가지 구간분할 x 3가지 임상정보 활용방식(총 12설정)의 CNN을 다중시드 x 5-fold로
    # 학습 -> 각 설정 단독 및 임상결합 점수의 내부/외부 AUC와 임상단독 대비 부트스트랩 증분 계산 ->
    # 결과 정렬/그래프화 및 저장 순으로 실행된다.
    main()
