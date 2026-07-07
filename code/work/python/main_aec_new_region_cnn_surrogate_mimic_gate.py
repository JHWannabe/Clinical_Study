from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
    make_single_deesc,
)
from aec_region_constrained_cnn_gate import make_channels, standardize_channels_train_apply, stratified_folds  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "aec_new_region_cnn_surrogate_mimic_gate"
PROGRESS_PATH = OUT_DIR / "progress.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_OPS = ["S80", "S85", "S90"]
SEEDS = [20260701, 20260711]
MIN_DEESC_N = 10
MAX_SENS_LOSS = 0.08

REGIONS: dict[str, tuple[int, int]] = {
    "R1_045_056": (45, 56),
    "R2_057_080": (57, 80),
    "R3_097_128": (97, 128),
    "R4_117_128": (117, 128),
}

# Teacher copied from the portable surrogate audit winner:
# outputs/aec_new_region_surrogate_combo_gate/new4_combo_summary.json
TEACHER_BRANCHES = [
    {"region": "R1_045_056", "descriptor": "endpoint_delta", "sign": -1, "width": 0.35, "lambda": 0.25},
    {"region": "R2_057_080", "descriptor": "level_mean", "sign": -1, "width": 0.35, "lambda": 0.25},
    {"region": "R3_097_128", "descriptor": "endpoint_delta", "sign": -1, "width": 0.70, "lambda": 0.25},
    {"region": "R4_117_128", "descriptor": "linear_slope", "sign": -1, "width": 0.35, "lambda": 0.25},
]
TEACHER_PATTERNS = ["--+-", "---+", "++-+", "--++", "++++"]


@dataclass(frozen=True)
class MimicConfig:
    name: str
    hidden: int = 10
    dropout: float = 0.20
    lr: float = 8.0e-4
    weight_decay: float = 1.0e-3
    consensus_weight: float = 0.65
    non_cpos_weight: float = 0.04
    max_epochs: int = 180
    patience: int = 22
    batch_size: int = 96


CONFIGS = [
    MimicConfig("surrogate_mimic_balanced", dropout=0.20, consensus_weight=0.65, non_cpos_weight=0.04),
    MimicConfig("surrogate_mimic_guarded", hidden=12, dropout=0.30, lr=6.0e-4, weight_decay=2.0e-3, consensus_weight=0.85, non_cpos_weight=0.02),
]


def write_progress(**kwargs: object) -> None:
    """현재 진행 상황(단계, 학습/탐색 진행률 등)을 타임스탬프와 함께 progress.json에 기록."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    """numpy/torch 난수 시드를 고정하고, GPU 사용 시 재현성을 위해 cuDNN 결정론적 모드를 설정."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def d1(x: np.ndarray) -> np.ndarray:
    """각 행(환자)에 대해 1차 차분(기울기)을 계산. 맨 앞 값은 첫 차분값으로 채워 길이를 원본과 동일하게 유지."""
    v = np.diff(x, axis=1)
    return np.column_stack([v[:, :1], v])


def d2(x: np.ndarray) -> np.ndarray:
    """1차 차분(d1)의 차분을 계산해 2차 차분(곡률)을 구함. 맨 앞 값은 첫 값으로 채워 길이를 유지."""
    v = np.diff(d1(x), axis=1)
    return np.column_stack([v[:, :1], v])


def region_descriptor_matrix(norm: np.ndarray) -> pd.DataFrame:
    """정규화된 AEC 곡선에서 미리 정의된 4개 구간(REGIONS)마다 레벨/기울기/곡률 관련 12개 서술자(descriptor)를 계산해 특징 DataFrame을 만듦."""
    slope = d1(norm)
    curv = d2(norm)
    grid = np.arange(norm.shape[1], dtype=float)
    rows: dict[str, np.ndarray] = {}
    for region, (start, end) in REGIONS.items():
        sl = slice(start - 1, end)
        block = norm[:, sl]
        sb = slope[:, sl]
        cb = curv[:, sl]
        x = grid[sl] - grid[sl].mean()
        denom = float((x**2).sum()) or 1.0
        rows[f"{region}__level_mean"] = block.mean(axis=1)
        rows[f"{region}__level_sd"] = block.std(axis=1)
        rows[f"{region}__endpoint_delta"] = block[:, -1] - block[:, 0]
        rows[f"{region}__linear_slope"] = ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom
        rows[f"{region}__slope_mean"] = sb.mean(axis=1)
        rows[f"{region}__slope_sd"] = sb.std(axis=1)
        rows[f"{region}__abs_slope_mean"] = np.abs(sb).mean(axis=1)
        rows[f"{region}__abs_slope_max"] = np.abs(sb).max(axis=1)
        rows[f"{region}__curv_mean"] = cb.mean(axis=1)
        rows[f"{region}__curv_sd"] = cb.std(axis=1)
        rows[f"{region}__abs_curv_mean"] = np.abs(cb).mean(axis=1)
        rows[f"{region}__abs_curv_max"] = np.abs(cb).max(axis=1)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def z_train_apply(xg_df: pd.DataFrame, xs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """내부(xg_df) 데이터의 중앙값으로 결측을 채우고 내부 데이터 기준 평균/표준편차로 z-표준화한 뒤, 같은 통계를 외부(xs_df) 데이터에도 적용."""
    names = list(xg_df.columns)
    xg = xg_df.to_numpy(dtype=float)
    xs = xs_df.to_numpy(dtype=float)
    med = np.nanmedian(xg, axis=0)
    xg = np.where(np.isfinite(xg), xg, med[None, :])
    xs = np.where(np.isfinite(xs), xs, med[None, :])
    mu = xg.mean(axis=0)
    sd = xg.std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (xg - mu) / sd, (xs - mu) / sd, names


def teacher_features(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """지역 서술자 행렬을 표준화한 뒤 TEACHER_BRANCHES에 지정된 4개 특징만 골라 부호(sign)를 적용한 내부/외부 특징 배열과 라벨 목록을 만듦 — 교사(teacher) 대리 규칙을 재현하기 위한 입력."""
    fg = region_descriptor_matrix(g["norm"])
    fs = region_descriptor_matrix(s["norm"])
    xg_all, xs_all, names = z_train_apply(fg, fs)
    name_to_idx = {name: i for i, name in enumerate(names)}
    idx = []
    labels = []
    signs = []
    for branch in TEACHER_BRANCHES:
        name = f"{branch['region']}__{branch['descriptor']}"
        idx.append(name_to_idx[name])
        labels.append(name)
        signs.append(float(branch["sign"]))
    sign_arr = np.asarray(signs, dtype=float)
    return xg_all[:, idx] * sign_arr[None, :], xs_all[:, idx] * sign_arr[None, :], labels


def clinical_positive_matrix(score: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """임상 점수를 각 목표 운영점(TARGET_OPS)의 임계값과 비교해, 환자별 임상양성(cpos) 여부를 나타내는 불리언 행렬을 만듦."""
    return np.column_stack([score >= float(thresholds[op]) for op in TARGET_OPS])


def branch_votes(feature_z: np.ndarray, clinical_z: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """TEACHER_BRANCHES 각 특징으로 만든 단일-특징 de-escalation 게이트(교사 규칙)를 모든 목표 운영점에 대해 계산해, (환자수 x 운영점수 x 4브랜치) 형태의 교사 투표(distillation 타깃)를 만듦."""
    out = np.zeros((len(feature_z), len(TARGET_OPS), len(TEACHER_BRANCHES)), dtype=np.float32)
    for op_idx, op in enumerate(TARGET_OPS):
        th = float(thresholds[op])
        for j, branch in enumerate(TEACHER_BRANCHES):
            out[:, op_idx, j] = make_single_deesc(
                clinical_z,
                feature_z[:, j],
                th,
                float(branch["width"]),
                float(branch["lambda"]),
            ).astype(np.float32)
    return out


def pattern_str(code: int) -> str:
    """4비트 정수 코드를 "+/-" 4글자 패턴 문자열(각 자리가 지역 투표 여부)로 변환."""
    return "".join("+" if code & (1 << j) else "-" for j in range(len(REGIONS)))


def pattern_mask_to_text(mask: int) -> str:
    """여러 패턴 코드를 포함하는 비트마스크를, 콤마로 구분된 패턴 문자열 목록으로 변환."""
    return ",".join(pattern_str(code) for code in range(16) if mask & (1 << code))


def mask_from_patterns(patterns: list[str]) -> int:
    """"+/-" 4글자 패턴 문자열 목록을 각각 4비트 코드로 해석해, 그 코드들을 포함하는 비트마스크로 합침."""
    mask = 0
    for pat in patterns:
        code = 0
        for j, ch in enumerate(pat):
            if ch == "+":
                code |= 1 << j
        mask |= 1 << code
    return mask


def popcount(x: int) -> int:
    """정수의 이진수 표현에서 1의 개수(투표한 지역 수)를 셈."""
    return int(bin(int(x)).count("1"))


def soft_atleast2_prob(logits: torch.Tensor) -> torch.Tensor:
    """4개 지역 branch 로짓으로부터 "정확히 0개 또는 1개만 투표"할 확률을 1에서 빼서, "2개 이상 투표"할 미분가능한 확률(합의 확률)을 계산."""
    p = torch.sigmoid(logits)
    q = 1.0 - p
    p0 = q.prod(dim=-1)
    p1 = torch.zeros_like(p0)
    for j in range(p.shape[-1]):
        p1 = p1 + p[..., j] * torch.cat([q[..., :j], q[..., j + 1 :]], dim=-1).prod(dim=-1)
    return torch.clamp(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)


class RegionBranch(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        """단일 지역(region) 구간의 형태(morphology)만 요약하는 1D CNN 분기(conv 2층 + 평균/최대 풀링 + 선형 헤드)를 구성."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv 네트워크로 지역 구간 특징을 추출한 뒤 평균/최대로 풀링하고, 드롭아웃과 선형 헤드를 통과시켜 형태 점수(스칼라)를 출력."""
        z = self.net(x)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)
        return self.head(self.drop(pooled)).squeeze(1)


class DirectVoteMimicCnn(nn.Module):
    thresholds: torch.Tensor
    widths: torch.Tensor

    def __init__(self, cfg: MimicConfig, threshold_vec: np.ndarray) -> None:
        """4개 지역별 RegionBranch(형태 점수)와, 교사(surrogate) de-escalation 규칙의 boundary-gated 구조를 흉내내는 선형 결합 헤드를 구성.
        헤드 가중치는 교사 규칙의 부호(clinical delta, boundary, cpos 항의 방향)를 반영해 초기화."""
        super().__init__()
        self.regions = list(REGIONS.items())
        self.branches = nn.ModuleList([RegionBranch(cfg.hidden, cfg.dropout) for _ in self.regions])
        self.head_weight = nn.Parameter(torch.zeros(len(REGIONS), 5))
        self.head_bias = nn.Parameter(torch.zeros(len(REGIONS)))
        widths = np.asarray([branch["width"] for branch in TEACHER_BRANCHES], dtype=np.float32)
        with torch.no_grad():
            self.head_weight[:, 1] = -1.5
            self.head_weight[:, 2] = -2.0
            self.head_weight[:, 3] = 0.5
            self.head_weight[:, 4] = 0.5
            self.head_bias[:] = -1.0
        self.register_buffer("thresholds", torch.tensor(threshold_vec, dtype=torch.float32))
        self.register_buffer("widths", torch.tensor(widths, dtype=torch.float32))

    def forward(self, x: torch.Tensor, clinical_z: torch.Tensor) -> torch.Tensor:
        """각 지역의 CNN 형태 점수와, 임상 점수-임계값 거리(delta)·경계 가중치(boundary)·임상양성 여부(cpos)로 만든 5개 특징을 선형 결합해, (환자 x 운영점 x 지역) 로짓을 계산 — 교사의 단일-특징 de-escalation 게이트 구조를 모방."""
        morph = []
        for branch, (_, (start, end)) in zip(self.branches, self.regions):
            morph.append(branch(x[:, :, start - 1 : end]))
        morph_t = torch.stack(morph, dim=-1)
        delta = clinical_z[:, None] - self.thresholds[None, :]
        boundary = torch.exp(-0.5 * (delta[:, :, None] / self.widths[None, None, :]) ** 2)
        cpos = (delta >= 0).float()[:, :, None]
        feats = torch.stack(
            [
                morph_t[:, None, :].expand(-1, len(TARGET_OPS), -1),
                morph_t[:, None, :] * boundary,
                delta[:, :, None].expand(-1, -1, len(REGIONS)),
                boundary,
                cpos.expand(-1, -1, len(REGIONS)),
            ],
            dim=-1,
        )
        return (feats * self.head_weight[None, None, :, :]).sum(dim=-1) + self.head_bias[None, None, :]


def loss_fn(logits: torch.Tensor, target: torch.Tensor, cpos_weight: torch.Tensor, pos_weight: torch.Tensor, cfg: MimicConfig) -> torch.Tensor:
    """교사 투표(target)를 흉내내도록 지역별 가중 BCE 손실과, 2-of-4 합의 확률에 대한 BCE 손실을 결합한 distillation 학습 손실을 계산."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = cpos_weight[:, :, None] * (1.0 + (pos_weight[None, None, :] - 1.0) * target)
    branch_loss = (bce * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    prob2 = soft_atleast2_prob(logits)
    target2 = (target.sum(dim=-1) >= 2).float()
    consensus_loss = (F.binary_cross_entropy(prob2, target2, reduction="none") * cpos_weight).sum() / torch.clamp(cpos_weight.sum(), min=1.0)
    return branch_loss + cfg.consensus_weight * consensus_loss


def train_fold(
    cfg: MimicConfig,
    x: np.ndarray,
    clinical_z: np.ndarray,
    target: np.ndarray,
    cpos: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    clinical_ext: np.ndarray,
    threshold_vec: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """한 fold에 대해 DirectVoteMimicCnn을 교사 투표를 흉내내도록(distillation) 조기종료로 학습시키고, 최적 검증 손실 상태로 복원한 뒤 검증세트와 외부세트에 대한 로짓과 학습 정보를 반환."""
    set_seed(seed)
    model = DirectVoteMimicCnn(cfg, threshold_vec).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(clinical_z[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(target[train_idx], dtype=torch.float32, device=DEVICE)
    wt_np = np.where(cpos[train_idx], 1.0, cfg.non_cpos_weight).astype(np.float32)
    wt = torch.tensor(wt_np, dtype=torch.float32, device=DEVICE)

    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    cv = torch.tensor(clinical_z[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(target[val_idx], dtype=torch.float32, device=DEVICE)
    wv_np = np.where(cpos[val_idx], 1.0, cfg.non_cpos_weight).astype(np.float32)
    wv = torch.tensor(wv_np, dtype=torch.float32, device=DEVICE)

    pos = (target[train_idx] * wt_np[:, :, None]).sum(axis=(0, 1))
    neg = ((1.0 - target[train_idx]) * wt_np[:, :, None]).sum(axis=(0, 1))
    pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 40.0).astype(np.float32)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = cfg.patience
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[batch], ct[batch]), yt[batch], wt[batch], pos_weight, cfg)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv, cv), yv, wv, pos_weight, cfg).item())
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
    model.eval()
    with torch.no_grad():
        val_logits = model(xv, cv).detach().cpu().numpy()
        ext_logits = model(
            torch.tensor(x_ext, dtype=torch.float32, device=DEVICE),
            torch.tensor(clinical_ext, dtype=torch.float32, device=DEVICE),
        ).detach().cpu().numpy()
    return val_logits, ext_logits, {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss), "pos_weight_mean": float(pw.mean())}


def crossfit_config(
    cfg: MimicConfig,
    xg: np.ndarray,
    c_g: np.ndarray,
    target_g: np.ndarray,
    cpos_g: np.ndarray,
    xs: np.ndarray,
    c_s: np.ndarray,
    y: np.ndarray,
    threshold_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드 x 5-fold 교차검증으로 설정(cfg) 하나를 반복 학습시켜, 내부 OOF 로짓과 외부 로짓(fold/시드 평균)과 학습 로그를 반환."""
    oof_runs = []
    ext_runs = []
    logs = []
    total_folds = len(SEEDS) * 5
    done = 0
    for seed in SEEDS:
        oof = np.zeros_like(target_g, dtype=float)
        ext_folds = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            val_logits, ext_logits, info = train_fold(cfg, xg, c_g, target_g, cpos_g, tr, va, xs, c_s, threshold_vec, seed + fold_id * 101)
            oof[va] = val_logits
            ext_folds.append(ext_logits)
            logs.append({"config": cfg.name, "seed": seed, "fold": fold_id, **info})
            done += 1
            write_progress(stage="training", config=cfg.name, fold_done=done, fold_total=total_folds, device=str(DEVICE))
            print(f"  {cfg.name}: seed={seed} fold={fold_id + 1}/5 best_epoch={info['best_epoch']} val_loss={info['best_val_loss']:.5f}", flush=True)
        oof_runs.append(oof)
        ext_runs.append(np.mean(ext_folds, axis=0))
    return np.mean(oof_runs, axis=0), np.mean(ext_runs, axis=0), pd.DataFrame(logs)


def codes_from_prob(prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """(환자 x 운영점 x 지역) 확률을 지역별 임계값과 비교해 0/1 투표로 만들고, 4개 투표를 하나의 4비트 패턴 코드(0~15)로 합침."""
    votes = prob >= thresholds[None, None, :]
    code = np.zeros(prob.shape[:2], dtype=np.int16)
    for j in range(prob.shape[-1]):
        code += votes[:, :, j].astype(np.int16) * (1 << j)
    return code


def votes_to_codes(votes: np.ndarray) -> np.ndarray:
    """이미 0/1로 확정된 (환자 x 운영점 x 지역) 투표 배열을 하나의 4비트 패턴 코드(0~15)로 합침."""
    code = np.zeros(votes.shape[:2], dtype=np.int16)
    for j in range(votes.shape[-1]):
        code += votes[:, :, j].astype(np.int16) * (1 << j)
    return code


def threshold_vectors() -> list[np.ndarray]:
    """4개 지역에 적용할 임계값 조합 후보들을 생성. 균일 임계값 스캔과 지역별 값이 다른 두 격자 조합을 합쳐 중복 제거 후 정렬된 리스트로 반환."""
    rows: set[tuple[float, float, float, float]] = set()
    for p in np.round(np.arange(0.35, 0.96, 0.05), 2):
        rows.add((float(p), float(p), float(p), float(p)))
    for a, b, c, d in itertools.product([0.45, 0.55, 0.65, 0.75, 0.85], repeat=4):
        rows.add((float(a), float(b), float(c), float(d)))
    for a, b, c, d in itertools.product([0.50, 0.60, 0.70, 0.80, 0.90], repeat=4):
        rows.add((float(a), float(b), float(c), float(d)))
    return [np.array(v, dtype=float) for v in sorted(rows)]


def exact_loss_p(tp_lost: int) -> float:
    """놓친 진양성(사건) 수를 바탕으로, 그 손실이 우연히 발생했을 확률(이항분포 근사, 2^(1-n))의 상한을 근사 계산."""
    if tp_lost <= 0:
        return 1.0
    return float(min(1.0, 2.0 ** (1 - int(tp_lost))))


def baseline_accuracy(y_bool: np.ndarray, pred: np.ndarray) -> float:
    """예측값과 실제 라벨이 일치하는 비율(정확도)을 계산."""
    return float(np.mean(y_bool == pred))


def fast_dataset_summary(y: np.ndarray, cpos: np.ndarray, code: np.ndarray, mask: int) -> dict:
    """패턴 마스크로 선택된 코드에 해당하는 환자를 강등 대상으로 삼아, 각 운영점에서의 민감도 손실/특이도 증가/정확도 변화 등을 계산해 데이터셋 전체 요약 딕셔너리로 반환."""
    selected_codes = [k for k in range(16) if mask & (1 << k)]
    yy = y.astype(bool)
    total_pos = max(int(yy.sum()), 1)
    total_neg = max(int((~yy).sum()), 1)
    n_total = len(yy)
    sens_loss = []
    p_loss = []
    spec_gain = []
    acc_delta = []
    post_acc = []
    deesc_n = []
    event_rate = []
    for op_idx in range(cpos.shape[1]):
        deesc = cpos[:, op_idx] & np.isin(code[:, op_idx], selected_codes)
        tp_lost = int(np.sum(deesc & yy))
        fp_removed = int(np.sum(deesc & ~yy))
        n = tp_lost + fp_removed
        sens_loss.append(tp_lost / total_pos)
        p_loss.append(exact_loss_p(tp_lost))
        spec_gain.append(fp_removed / total_neg)
        delta = (fp_removed - tp_lost) / n_total
        acc_delta.append(delta)
        post_acc.append(baseline_accuracy(yy, cpos[:, op_idx]) + delta)
        deesc_n.append(n)
        event_rate.append(tp_lost / n if n else np.nan)
    return {
        "mean_accuracy": float(np.nanmean(post_acc)),
        "mean_accuracy_gain": float(np.nanmean(acc_delta)),
        "min_accuracy_gain": float(np.nanmin(acc_delta)),
        "mean_specificity_gain": float(np.nanmean(spec_gain)),
        "min_specificity_gain": float(np.nanmin(spec_gain)),
        "max_sensitivity_loss": float(np.nanmax(sens_loss)),
        "min_sensitivity_loss_p": float(np.nanmin(p_loss)),
        "min_deesc_n": int(np.nanmin(deesc_n)),
        "mean_event_rate": float(np.nanmean(event_rate)),
    }


def dataset_pass(row: dict, prefix: str) -> bool:
    """fast_dataset_summary 요약이 최소 de-escalation 건수, 민감도 손실 한도/유의성, 특이도/정확도 증가 조건을 모두 만족하는지 판정."""
    return bool(
        row[f"{prefix}_min_deesc_n"] >= MIN_DEESC_N
        and row[f"{prefix}_max_sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12
        and row[f"{prefix}_min_sensitivity_loss_p"] >= 0.05
        and row[f"{prefix}_min_specificity_gain"] > 0
        and row[f"{prefix}_min_accuracy_gain"] > 0
    )


def rank_codes_internal(y: np.ndarray, cpos: np.ndarray, code: np.ndarray) -> list[int]:
    """내부 데이터 기준으로 16개 투표 패턴 각각에 대해 모든 목표 운영점에서의 비사건수/사건수/사건율로 점수를 매겨, 상위 8개 패턴 코드를 반환."""
    yy = y.astype(bool)
    rows = []
    for pat in range(16):
        nonevents = []
        events = []
        event_rates = []
        for op_idx in range(cpos.shape[1]):
            idx = cpos[:, op_idx] & (code[:, op_idx] == pat)
            n = int(idx.sum())
            e = int(np.sum(idx & yy))
            nonevents.append(n - e)
            events.append(e)
            event_rates.append(e / n if n else 1.0)
        score = min(nonevents) - 4.0 * max(events) - 18.0 * float(np.mean(event_rates))
        rows.append((score, pat))
    rows.sort(reverse=True)
    return [pat for _, pat in rows[:8]]


def candidate_masks(top_codes: list[int]) -> list[int]:
    """de-escalation 후보 비트마스크 집합을 생성: 교사(TEACHER_PATTERNS) 마스크, 단일 패턴, popcount 기준 이상/정확히 k개 패턴, 상위 패턴 조합(2~5개)을 모두 모아 정렬된 리스트로 반환."""
    masks: set[int] = {mask_from_patterns(TEACHER_PATTERNS)}
    for code in range(16):
        masks.add(1 << code)
    for k in [1, 2, 3, 4]:
        at_least = 0
        exactly = 0
        for code in range(16):
            if popcount(code) >= k:
                at_least |= 1 << code
            if popcount(code) == k:
                exactly |= 1 << code
        masks.add(at_least)
        masks.add(exactly)
    for size in range(2, min(5, len(top_codes)) + 1):
        for combo in itertools.combinations(top_codes, size):
            m = 0
            for code in combo:
                m |= 1 << code
            masks.add(m)
    return sorted(masks)


def row_from_fast(config: str, thresholds: np.ndarray, mask: int, g_sum: dict, s_sum: dict) -> dict:
    """한 규칙(설정+임계값+패턴 마스크)에 대한 내부/외부 fast_dataset_summary 결과를 합쳐, 통과 여부와 선택 점수를 포함한 한 행짜리 딕셔너리로 만듦."""
    row = {
        "config": config,
        "rule": f"{config}_t{'_'.join(f'{x:.2f}' for x in thresholds)}_m{mask}",
        "threshold_R1": float(thresholds[0]),
        "threshold_R2": float(thresholds[1]),
        "threshold_R3": float(thresholds[2]),
        "threshold_R4": float(thresholds[3]),
        "pattern_mask": int(mask),
        "patterns": pattern_mask_to_text(mask),
        "n_patterns": popcount(mask),
    }
    for prefix, summ in [("internal", g_sum), ("external", s_sum)]:
        for key, value in summ.items():
            row[f"{prefix}_{key}"] = value
    row["internal_pass"] = dataset_pass(row, "internal")
    row["external_pass"] = dataset_pass(row, "external")
    row["internal_selection_score"] = (
        row["internal_mean_accuracy"]
        + 0.45 * row["internal_min_accuracy_gain"]
        + 0.20 * row["internal_min_specificity_gain"]
        - 0.20 * row["internal_max_sensitivity_loss"]
        - 0.03 * row["internal_mean_event_rate"]
    )
    if not row["internal_pass"]:
        row["internal_selection_score"] -= 10.0
    return row


def evaluate_gate_detail(
    rule: str,
    features: str,
    mask: int,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    code_g: np.ndarray,
    code_s: np.ndarray,
) -> pd.DataFrame:
    """주어진 패턴 마스크로 내부/외부 데이터셋의 각 목표 운영점에서 de-escalation 상세 지표(deesc_metric_row)를 계산해 하나의 DataFrame으로 합침."""
    selected_codes = [code for code in range(16) if mask & (1 << code)]
    rows = []
    for dataset, d, cpos, code in [
        ("g1090_internal", g, cpos_g, code_g),
        ("sdata_external", s, cpos_s, code_s),
    ]:
        for op_idx, op in enumerate(TARGET_OPS):
            deesc = cpos[:, op_idx] & np.isin(code[:, op_idx], selected_codes)
            rows.append(deesc_metric_row(dataset, rule, features, op, d["y"].astype(int), cpos[:, op_idx], deesc))
    return pd.DataFrame(rows)


def search_gates(
    config: str,
    prob_g: np.ndarray,
    prob_s: np.ndarray,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    config_index: int,
    n_configs: int,
) -> pd.DataFrame:
    """한 CNN 설정의 확률값에 대해 모든 임계값 조합과, 내부 기준 상위 패턴 마스크(+교사 마스크) 조합을 전수 평가하여 후보 규칙들의 DataFrame을 만들고 진행상황을 주기적으로 출력/기록."""
    ths = threshold_vectors()
    rows = []
    start = time.time()
    for th_idx, th in enumerate(ths, start=1):
        code_g = codes_from_prob(prob_g, th)
        code_s = codes_from_prob(prob_s, th)
        masks = candidate_masks(rank_codes_internal(g["y"], cpos_g, code_g))
        for mask in masks:
            rows.append(row_from_fast(config, th, mask, fast_dataset_summary(g["y"], cpos_g, code_g, mask), fast_dataset_summary(s["y"], cpos_s, code_s, mask)))
        if th_idx == 1 or th_idx % 50 == 0 or th_idx == len(ths):
            elapsed = time.time() - start
            rate = th_idx / elapsed if elapsed > 0 else 0.0
            eta = (len(ths) - th_idx) / rate if rate > 0 else None
            internal_pass_n = sum(1 for r in rows if r["internal_pass"])
            both_pass_n = sum(1 for r in rows if r["internal_pass"] and r["external_pass"])
            print(
                f"[{config}] gate search {th_idx}/{len(ths)}; candidates={len(rows)}; "
                f"internal_pass={internal_pass_n}; both_pass={both_pass_n}; ETA_min={eta / 60 if eta else np.nan:.1f}",
                flush=True,
            )
            write_progress(
                stage="gate_search",
                config=config,
                config_index=config_index,
                n_configs=n_configs,
                threshold_index=th_idx,
                threshold_total=len(ths),
                candidates=len(rows),
                internal_pass=internal_pass_n,
                internal_external_pass=both_pass_n,
                eta_seconds=eta,
            )
    return pd.DataFrame(rows)


def detail_for_winner(row: pd.Series, g: dict, s: dict, cpos_g: np.ndarray, cpos_s: np.ndarray, prob_g: np.ndarray, prob_s: np.ndarray) -> pd.DataFrame:
    """선택된 최종 규칙 행(row)의 임계값과 패턴 마스크를 다시 적용해, 내부/외부 데이터셋에 대한 상세 de-escalation 지표 DataFrame을 재계산."""
    thresholds = np.array([row["threshold_R1"], row["threshold_R2"], row["threshold_R3"], row["threshold_R4"]], dtype=float)
    code_g = codes_from_prob(prob_g, thresholds)
    code_s = codes_from_prob(prob_s, thresholds)
    features = f"thresholds={','.join(f'{x:.2f}' for x in thresholds)}; patterns={row['patterns']}"
    return evaluate_gate_detail(str(row["rule"]), features, int(row["pattern_mask"]), g, s, cpos_g, cpos_s, code_g, code_s)


def plot_detail(detail: pd.DataFrame, out_path: Path, title: str) -> None:
    """내부/외부 데이터셋별 정확도 증가·특이도 증가·민감도 손실을 운영점별로 나란히 3개 패널 선 그래프로 그려 제목과 함께 이미지 파일로 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    colors = {"g1090_internal": "#2F6B9A", "sdata_external": "#C54E2C"}
    for dataset, ls in [("g1090_internal", "-"), ("sdata_external", "--")]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[TARGET_OPS].reset_index()
        x = np.arange(len(TARGET_OPS))
        axes[0].plot(x, sub["accuracy_delta"] * 100, marker="o", color=colors[dataset], ls=ls, label=dataset)
        axes[1].plot(x, sub["specificity_gain"] * 100, marker="o", color=colors[dataset], ls=ls, label=dataset)
        axes[2].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=colors[dataset], ls=ls, label=dataset)
    for ax, label in zip(axes, ["Accuracy gain", "Specificity gain", "Sensitivity loss"]):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(TARGET_OPS)))
        ax.set_xticklabels(TARGET_OPS)
        ax.set_ylabel("percentage points")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def agreement_table(label: str, target_g: np.ndarray, target_s: np.ndarray, prob_g: np.ndarray, prob_s: np.ndarray, th: np.ndarray, cpos_g: np.ndarray, cpos_s: np.ndarray) -> pd.DataFrame:
    """임상양성 환자들에 한해, CNN 임계값 예측이 교사 투표(target)와 브랜치별로/2-of-4 합의 기준으로 얼마나 일치하는지를 운영점별로 계산해 DataFrame으로 반환."""
    pred_g = prob_g >= th[None, None, :]
    pred_s = prob_s >= th[None, None, :]
    rows = []
    for dataset, target, pred, cpos in [
        ("g1090_internal", target_g.astype(bool), pred_g, cpos_g),
        ("sdata_external", target_s.astype(bool), pred_s, cpos_s),
    ]:
        for op_idx, op in enumerate(TARGET_OPS):
            cp = cpos[:, op_idx]
            rows.append(
                {
                    "rule": label,
                    "dataset": dataset,
                    "operating_point": op,
                    "branch_vote_agreement_cpos": float((target[cp, op_idx, :] == pred[cp, op_idx, :]).mean()) if cp.any() else np.nan,
                    "consensus_agreement_cpos": float(((target[cp, op_idx, :].sum(axis=1) >= 2) == (pred[cp, op_idx, :].sum(axis=1) >= 2)).mean()) if cp.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """고정된 교사(surrogate combo) 규칙의 투표를 distillation 타깃으로 삼아 지역별 CNN(DirectVoteMimicCnn)을 학습시키고,
    정확한 교사 게이트 결과를 기록한 뒤, 학습된 CNN 확률에 대해 임계값×패턴마스크 조합을 탐색하여 내부(및 내부+외부) 통과 규칙과
    교사 투표와의 일치도(agreement)까지 계산해 결과표/그래프/요약 JSON을 저장."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    print(f"teacher_patterns={TEACHER_PATTERNS}", flush=True)
    write_progress(stage="started", device=str(DEVICE), teacher_patterns=TEACHER_PATTERNS)

    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    threshold_vec = np.array([thresholds[op] for op in TARGET_OPS], dtype=np.float32)
    cpos_g = clinical_positive_matrix(c_g, thresholds)
    cpos_s = clinical_positive_matrix(c_s, thresholds)
    feat_g, feat_s, feature_labels = teacher_features(g, s)
    target_g = branch_votes(feat_g, c_g, thresholds)
    target_s = branch_votes(feat_s, c_s, thresholds)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))

    teacher_mask = mask_from_patterns(TEACHER_PATTERNS)
    exact_detail = evaluate_gate_detail(
        "exact_surrogate_teacher_gate",
        f"teacher_features={';'.join(feature_labels)}; patterns={','.join(TEACHER_PATTERNS)}",
        teacher_mask,
        g,
        s,
        cpos_g,
        cpos_s,
        votes_to_codes(target_g),
        votes_to_codes(target_s),
    )
    exact_detail.to_csv(OUT_DIR / "exact_surrogate_teacher_details.csv", index=False)

    prob_by_config: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    logs_all = []
    for cfg_idx, cfg in enumerate(CONFIGS, start=1):
        print(f"\ntraining {cfg.name} ({cfg_idx}/{len(CONFIGS)})", flush=True)
        logits_g, logits_s, logs = crossfit_config(cfg, xg, c_g, target_g, cpos_g, xs, c_s, g["y"], threshold_vec)
        prob_g = 1.0 / (1.0 + np.exp(-logits_g))
        prob_s = 1.0 / (1.0 + np.exp(-logits_s))
        prob_by_config[cfg.name] = (prob_g, prob_s)
        logs.to_csv(OUT_DIR / f"{cfg.name}_training_log.csv", index=False)
        logs_all.append(logs)
        np.savez_compressed(OUT_DIR / f"{cfg.name}_probabilities.npz", prob_g=prob_g, prob_s=prob_s, regions=np.array(list(REGIONS.keys())))
    pd.concat(logs_all, ignore_index=True).to_csv(OUT_DIR / "surrogate_mimic_training_log.csv", index=False)

    all_candidates = []
    for cfg_idx, cfg in enumerate(CONFIGS, start=1):
        print(f"\nsearching same-rule gate for {cfg.name} ({cfg_idx}/{len(CONFIGS)})", flush=True)
        prob_g, prob_s = prob_by_config[cfg.name]
        cand = search_gates(cfg.name, prob_g, prob_s, g, s, cpos_g, cpos_s, cfg_idx, len(CONFIGS))
        cand.to_csv(OUT_DIR / f"{cfg.name}_same_rule_candidates.csv", index=False)
        all_candidates.append(cand)

    summary = pd.concat(all_candidates, ignore_index=True).sort_values(
        ["internal_pass", "internal_selection_score", "internal_mean_accuracy"],
        ascending=[False, False, False],
    )
    internal_passing = summary[summary["internal_pass"]].copy()
    both_passing = internal_passing[internal_passing["external_pass"]].sort_values(
        ["external_mean_accuracy_gain", "external_mean_specificity_gain", "internal_mean_accuracy_gain"],
        ascending=False,
    )
    summary.to_csv(OUT_DIR / "surrogate_mimic_same_rule_all_candidates.csv", index=False)
    internal_passing.to_csv(OUT_DIR / "surrogate_mimic_internal_passing_ranked.csv", index=False)
    both_passing.to_csv(OUT_DIR / "surrogate_mimic_internal_external_passing_ranked.csv", index=False)

    winners = {
        "internal_locked": internal_passing.iloc[0] if not internal_passing.empty else None,
        "internal_external_audit": both_passing.iloc[0] if not both_passing.empty else None,
    }
    for tag, winner in winners.items():
        if winner is None:
            continue
        prob_g, prob_s = prob_by_config[str(winner["config"])]
        detail = detail_for_winner(winner, g, s, cpos_g, cpos_s, prob_g, prob_s)
        detail.to_csv(OUT_DIR / f"{tag}_winner_details.csv", index=False)
        plot_detail(detail, OUT_DIR / f"{tag}_winner_plot.png", f"{tag}: {winner['config']}")
        th = np.array([winner["threshold_R1"], winner["threshold_R2"], winner["threshold_R3"], winner["threshold_R4"]], dtype=float)
        agreement_table(str(winner["rule"]), target_g, target_s, prob_g, prob_s, th, cpos_g, cpos_s).to_csv(OUT_DIR / f"{tag}_agreement.csv", index=False)

    with (OUT_DIR / "surrogate_mimic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(DEVICE),
                "preprocessing": "aec_128, Gaussian smoothing sigma=1, patient-wise mean normalization, row-z curve/slope/curvature channels",
                "teacher_branches": TEACHER_BRANCHES,
                "teacher_patterns": TEACHER_PATTERNS,
                "teacher_pattern_mask": teacher_mask,
                "target_ops": TARGET_OPS,
                "configs": [asdict(cfg) for cfg in CONFIGS],
                "n_candidates": int(len(summary)),
                "n_internal_passing": int(len(internal_passing)),
                "n_internal_external_passing": int(len(both_passing)),
                "winners": {k: (None if v is None else v.to_dict()) for k, v in winners.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nEXACT TEACHER DETAIL")
    print(exact_detail.to_string(index=False))
    print("\nTOP SAME-RULE CANDIDATES")
    show_cols = [
        "config",
        "rule",
        "threshold_R1",
        "threshold_R2",
        "threshold_R3",
        "threshold_R4",
        "patterns",
        "internal_pass",
        "external_pass",
        "internal_mean_accuracy_gain",
        "internal_mean_specificity_gain",
        "internal_max_sensitivity_loss",
        "internal_min_sensitivity_loss_p",
        "external_mean_accuracy_gain",
        "external_mean_specificity_gain",
        "external_max_sensitivity_loss",
        "external_min_sensitivity_loss_p",
    ]
    print(summary.head(30)[show_cols].to_string(index=False))
    for tag in ["internal_locked", "internal_external_audit"]:
        p = OUT_DIR / f"{tag}_winner_details.csv"
        if p.exists():
            print(f"\n{tag.upper()} DETAILS")
            print(pd.read_csv(p).to_string(index=False))
            a = OUT_DIR / f"{tag}_agreement.csv"
            if a.exists():
                print(f"\n{tag.upper()} AGREEMENT")
                print(pd.read_csv(a).to_string(index=False))
    print("out_dir", OUT_DIR)
    write_progress(stage="done", out_dir=str(OUT_DIR), n_candidates=int(len(summary)))


# 고정된 surrogate-combo 교사 규칙의 지역별 투표를 흉내내도록 CNN(DirectVoteMimicCnn)을 distillation 방식으로 학습시키고,
# 학습된 CNN 확률로 임계값·패턴마스크를 재탐색해 내부/외부에서 통과하는 de-escalation 규칙과 교사 규칙과의 일치도를 함께 산출하는 파이프라인을 실행.
if __name__ == "__main__":
    main()
