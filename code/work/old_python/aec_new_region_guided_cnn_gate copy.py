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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    auc_with_p,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
)
from aec_region_constrained_cnn_gate import (  # noqa: E402
    make_channels,
    standardize_channels_train_apply,
    stratified_folds,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_new_region_guided_cnn_gate"
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


@dataclass(frozen=True)
class TrainConfig:
    name: str
    hidden: int = 12
    dropout: float = 0.25
    lr: float = 7.0e-4
    weight_decay: float = 1.5e-3
    low_guard_weight: float = 5.0
    consensus_weight: float = 0.45
    diversity_weight: float = 0.00
    clinical_focus_floor: float = 0.35
    max_epochs: int = 180
    patience: int = 22
    batch_size: int = 96
    noise_sd: float = 0.015


CONFIGS = [
    TrainConfig("new4_outcome_balanced", dropout=0.25, low_guard_weight=4.0, consensus_weight=0.35),
    TrainConfig("new4_outcome_guarded", dropout=0.35, lr=6.0e-4, weight_decay=2.5e-3, low_guard_weight=8.0, consensus_weight=0.50),
    TrainConfig("new4_outcome_diverse", hidden=14, dropout=0.30, low_guard_weight=6.0, consensus_weight=0.45, diversity_weight=0.025),
    TrainConfig("new4_outcome_strict", hidden=16, dropout=0.40, lr=5.0e-4, weight_decay=3.0e-3, low_guard_weight=10.0, consensus_weight=0.60),
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


def soft_atleast2_prob(logits: torch.Tensor) -> torch.Tensor:
    """4개 지역 branch 로짓으로부터 "정확히 0개 또는 1개만 투표"할 확률을 1에서 빼서, "2개 이상 투표"할 미분가능한 확률(합의 확률)을 계산."""
    p = torch.sigmoid(logits)
    q = 1.0 - p
    p0 = q.prod(dim=1)
    p1 = torch.zeros_like(p0)
    for j in range(p.shape[1]):
        p1 = p1 + p[:, j] * torch.cat([q[:, :j], q[:, j + 1 :]], dim=1).prod(dim=1)
    return torch.clamp(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)


def soft_atleast2_np(prob: np.ndarray) -> np.ndarray:
    """soft_atleast2_prob의 넘파이 버전: 4개 지역 확률로부터 "2개 이상 투표"할 확률을 계산."""
    q = 1.0 - prob
    p0 = np.prod(q, axis=1)
    p1 = np.zeros_like(p0)
    for j in range(prob.shape[1]):
        p1 += prob[:, j] * np.prod(np.concatenate([q[:, :j], q[:, j + 1 :]], axis=1), axis=1)
    return np.clip(1.0 - p0 - p1, 1e-6, 1.0 - 1e-6)


class RegionBranch(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        """단일 지역(region) 구간을 입력받아 처리하는 1D CNN 분기(3개 conv 층 + 평균/최대/표준편차 풀링 + 완전연결 헤드)를 구성."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv 네트워크로 지역 구간 특징을 추출한 뒤 평균/최대/표준편차로 풀링하고, 완전연결 헤드를 통과시켜 스칼라 로짓을 출력."""
        z = self.net(x)
        pooled = torch.cat([z.mean(dim=2), z.amax(dim=2), z.std(dim=2, unbiased=False)], dim=1)
        return self.head(pooled).squeeze(1)


class NewRegionGuidedCnn(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        """REGIONS에 정의된 4개 지역마다 독립된 RegionBranch를 만들어 모듈 리스트로 구성."""
        super().__init__()
        self.regions = list(REGIONS.items())
        self.branches = nn.ModuleList([RegionBranch(cfg.hidden, cfg.dropout) for _ in self.regions])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 곡선을 각 지역 구간으로 잘라 해당 branch에 통과시키고, 4개 지역의 로짓을 하나의 텐서로 쌓아 반환."""
        logits = []
        for branch, (_, (start, end)) in zip(self.branches, self.regions):
            logits.append(branch(x[:, :, start - 1 : end]))
        return torch.stack(logits, dim=1)


def diversity_penalty(logits: torch.Tensor) -> torch.Tensor:
    """4개 지역 branch 확률들 간의 상관관계를 계산해, 서로 너무 비슷하게 예측하지 않도록 벌점을 주는 다양성 손실(비대각 상관의 제곱평균)을 반환."""
    p = torch.sigmoid(logits)
    p = p - p.mean(dim=0, keepdim=True)
    sd = p.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = p / sd
    corr = (z.T @ z) / max(int(z.shape[0]), 1)
    off_diag = corr - torch.eye(corr.shape[0], dtype=corr.dtype, device=corr.device)
    return (off_diag**2).mean()


def train_loss(
    logits: torch.Tensor,
    target_nonlow: torch.Tensor,
    sample_weight: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """저SMI(low) 환자에 가중치를 더 주는 branch별 BCE 손실과, 4개 지역 합의(2-of-4) 확률에 대한 BCE 손실을 결합하고, 필요시 다양성 벌점까지 더한 총 학습 손실을 계산."""
    target4 = target_nonlow[:, None].expand_as(logits)
    low_guard = torch.where(
        target_nonlow < 0.5,
        torch.full_like(target_nonlow, cfg.low_guard_weight),
        torch.ones_like(target_nonlow),
    )
    weight = sample_weight * low_guard
    branch_loss = F.binary_cross_entropy_with_logits(logits, target4, weight=weight[:, None].expand_as(logits))
    consensus_prob = soft_atleast2_prob(logits)
    consensus_loss = F.binary_cross_entropy(consensus_prob, target_nonlow, weight=weight)
    loss = branch_loss + cfg.consensus_weight * consensus_loss
    if cfg.diversity_weight > 0:
        loss = loss + cfg.diversity_weight * diversity_penalty(logits)
    return loss


def clinical_positive_matrix(score: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """임상 점수를 각 목표 운영점(TARGET_OPS)의 임계값과 비교해, 환자별 임상양성(cpos) 여부를 나타내는 불리언 행렬을 만듦."""
    return np.column_stack([score >= float(thresholds[op]) for op in TARGET_OPS])


def clinical_focus_weights(cpos: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    """임상양성 운영점 비율이 높을수록(즉 임상적으로 더 관심 대상일수록) 학습 샘플 가중치를 높게 주는 값을 계산."""
    return (cfg.clinical_focus_floor + cpos.mean(axis=1)).astype(np.float32)


def predict_logits(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """모델을 평가 모드로 배치 단위 실행해 전체 입력에 대한 지역별 로짓을 계산해 반환."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            out.append(model(xb).detach().cpu().numpy())
    return np.vstack(out)


def train_one_fold(
    cfg: TrainConfig,
    x: np.ndarray,
    y_low: np.ndarray,
    sample_weight: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """한 fold에 대해 NewRegionGuidedCnn을 조기종료(patience)로 학습시키고, 최적 검증 손실 상태로 복원한 뒤 검증세트와 외부세트에 대한 로짓과 학습 정보를 반환."""
    set_seed(seed)
    model = NewRegionGuidedCnn(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    target = (1 - y_low).astype(np.float32)

    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(target[train_idx], dtype=torch.float32, device=DEVICE)
    wt = torch.tensor(sample_weight[train_idx], dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(target[val_idx], dtype=torch.float32, device=DEVICE)
    wv = torch.tensor(sample_weight[val_idx], dtype=torch.float32, device=DEVICE)

    best_state = None
    best_epoch = 0
    best_loss = math.inf
    patience = cfg.patience
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = xt[idx]
            if cfg.noise_sd > 0:
                xb = xb + cfg.noise_sd * torch.randn_like(xb)
            opt.zero_grad(set_to_none=True)
            loss = train_loss(model(xb), yt[idx], wt[idx], cfg)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(train_loss(model(xv), yv, wv, cfg).item())
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
    return (
        predict_logits(model, x[val_idx]),
        predict_logits(model, x_ext),
        {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss)},
    )


def crossfit_config(
    cfg: TrainConfig,
    xg: np.ndarray,
    y: np.ndarray,
    cpos_g: np.ndarray,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드 x 5-fold 교차검증으로 설정(cfg) 하나를 반복 학습시켜, 내부 OOF 로짓과 외부 로짓(fold/시드 평균)과 학습 로그를 반환."""
    oof_runs = []
    ext_runs = []
    logs = []
    sample_weight = clinical_focus_weights(cpos_g, cfg)
    total_folds = len(SEEDS) * 5
    done = 0
    for seed in SEEDS:
        oof = np.zeros((len(y), len(REGIONS)), dtype=float)
        ext_fold_logits = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            val_logits, ext_logits, info = train_one_fold(cfg, xg, y, sample_weight, tr, va, xs, seed + fold_id * 101)
            oof[va] = val_logits
            ext_fold_logits.append(ext_logits)
            logs.append({"config": cfg.name, "seed": seed, "fold": fold_id, **info})
            done += 1
            write_progress(stage="training", config=cfg.name, fold_done=done, fold_total=total_folds, device=str(DEVICE))
            print(f"  {cfg.name}: seed={seed} fold={fold_id + 1}/5 best_epoch={info['best_epoch']} val_loss={info['best_val_loss']:.5f}", flush=True)
        oof_runs.append(oof)
        ext_runs.append(np.mean(ext_fold_logits, axis=0))
    return np.mean(oof_runs, axis=0), np.mean(ext_runs, axis=0), pd.DataFrame(logs)


def logistic_clinical_plus_auc(
    y_g: np.ndarray,
    y_s: np.ndarray,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    lowrisk_g: np.ndarray,
    lowrisk_s: np.ndarray,
) -> tuple[float, float, float, float]:
    """CNN의 저위험(비-저SMI) 확률을 로짓 스케일 위험 점수로 변환해 임상 점수와 결합한 로지스틱 회귀를 5-fold OOF로 학습·평가하고, 내부/외부 AUC와 p값을 반환."""
    aec_risk_g = -np.log(np.clip(lowrisk_g, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - lowrisk_g, 1e-6, 1.0))
    aec_risk_s = -np.log(np.clip(lowrisk_s, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - lowrisk_s, 1e-6, 1.0))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260701)
    oof = np.zeros(len(y_g), dtype=float)
    for fold_id, (tr, va) in enumerate(folds.split(np.zeros(len(y_g)), y_g)):
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260701 + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], aec_risk_g[tr]]), y_g[tr])
        oof[va] = model.decision_function(np.column_stack([clinical_oof[va], aec_risk_g[va]]))
    final = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260799)
    final.fit(np.column_stack([clinical_oof, aec_risk_g]), y_g)
    ext = final.decision_function(np.column_stack([clinical_ext, aec_risk_s]))
    ig_auc, ig_p = auc_with_p(y_g, oof)
    es_auc, es_p = auc_with_p(y_s, ext)
    return ig_auc, ig_p, es_auc, es_p


def auc_rows(
    config: str,
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    prob_g: np.ndarray,
    prob_s: np.ndarray,
) -> pd.DataFrame:
    """임상 단독, AEC 지역 평균/합의(2-of-4) 저위험 점수 단독, 그리고 임상+AEC 결합 모델 각각의 내부/외부 AUC를 계산해 임상 단독 대비 향상폭까지 담은 DataFrame을 만듦."""
    clinical_i_auc, clinical_i_p = auc_with_p(g["y"], clinical_oof)
    clinical_e_auc, clinical_e_p = auc_with_p(s["y"], clinical_ext)
    mean_lowrisk_g = prob_g.mean(axis=1)
    mean_lowrisk_s = prob_s.mean(axis=1)
    vote_lowrisk_g = soft_atleast2_np(prob_g)
    vote_lowrisk_s = soft_atleast2_np(prob_s)
    rows = [
        {
            "config": config,
            "model": "clinical_only",
            "internal_auc": clinical_i_auc,
            "internal_auc_p": clinical_i_p,
            "external_auc": clinical_e_auc,
            "external_auc_p": clinical_e_p,
        }
    ]
    for label, lg, ls in [
        ("aec_branch_mean_lowrisk_score", mean_lowrisk_g, mean_lowrisk_s),
        ("aec_soft_2of4_lowrisk_score", vote_lowrisk_g, vote_lowrisk_s),
    ]:
        ia, ip = auc_with_p(g["y"], -lg)
        ea, ep = auc_with_p(s["y"], -ls)
        rows.append({"config": config, "model": label, "internal_auc": ia, "internal_auc_p": ip, "external_auc": ea, "external_auc_p": ep})
        cia, cip, cea, cep = logistic_clinical_plus_auc(g["y"], s["y"], clinical_oof, clinical_ext, lg, ls)
        rows.append(
            {
                "config": config,
                "model": f"clinical_plus_{label}",
                "internal_auc": cia,
                "internal_auc_p": cip,
                "external_auc": cea,
                "external_auc_p": cep,
            }
        )
    df = pd.DataFrame(rows)
    df["internal_delta_vs_clinical"] = df["internal_auc"] - clinical_i_auc
    df["external_delta_vs_clinical"] = df["external_auc"] - clinical_e_auc
    return df


def pattern_str(code: int) -> str:
    """4비트 정수 코드를 "+/-" 4글자 패턴 문자열(각 자리가 지역 투표 여부)로 변환."""
    return "".join("+" if code & (1 << j) else "-" for j in range(len(REGIONS)))


def pattern_mask_to_text(mask: int) -> str:
    """여러 패턴 코드를 포함하는 비트마스크를, 콤마로 구분된 패턴 문자열 목록으로 변환."""
    return ",".join(pattern_str(code) for code in range(16) if mask & (1 << code))


def popcount(x: int) -> int:
    """정수의 이진수 표현에서 1의 개수(투표한 지역 수)를 셈."""
    return int(bin(int(x)).count("1"))


def codes_from_prob(prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """각 지역 확률을 지역별 임계값과 비교해 0/1 투표로 만들고, 4개 투표를 하나의 4비트 패턴 코드(0~15)로 합침."""
    votes = prob >= thresholds[None, :]
    code = np.zeros(len(prob), dtype=np.int16)
    for j in range(prob.shape[1]):
        code += votes[:, j].astype(np.int16) * (1 << j)
    return code


def threshold_vectors() -> list[np.ndarray]:
    """4개 지역에 적용할 임계값 조합 후보들을 생성. 균일 임계값 스캔과 지역별 값이 다른 두 격자 조합을 합쳐 중복 제거 후 정렬된 리스트로 반환."""
    rows: set[tuple[float, float, float, float]] = set()
    for p in np.round(np.arange(0.35, 0.96, 0.05), 2):
        rows.add((float(p), float(p), float(p), float(p)))
    for vals in itertools.product([0.45, 0.55, 0.65, 0.75, 0.85], repeat=4):
        rows.add(tuple(float(v) for v in vals))
    for vals in itertools.product([0.50, 0.60, 0.70, 0.80, 0.90], repeat=4):
        rows.add(tuple(float(v) for v in vals))
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
    """패턴 마스크로 선택된 코드에 해당하는 환자를 강등 대상으로 삼아, 각 운영점에서의 민감도 손실/특이도 증가/정확도 변화 등을 계산해 데이터셋 전체 요약 딕셔너리로 반환 (evaluate_gate_detail보다 빠른 벡터화 버전)."""
    selected = np.zeros_like(code, dtype=bool)
    for pat in range(16):
        if mask & (1 << pat):
            selected |= code == pat
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
        deesc = cpos[:, op_idx] & selected
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


def dataset_pass(summary: dict, prefix: str) -> bool:
    """fast_dataset_summary 요약 딕셔너리가 최소 de-escalation 건수, 민감도 손실 한도/유의성, 특이도/정확도 증가 조건을 모두 만족하는지 판정."""
    return bool(
        summary[f"{prefix}_min_deesc_n"] >= MIN_DEESC_N
        and summary[f"{prefix}_max_sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12
        and summary[f"{prefix}_min_sensitivity_loss_p"] >= 0.05
        and summary[f"{prefix}_min_specificity_gain"] > 0
        and summary[f"{prefix}_min_accuracy_gain"] > 0
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
            idx = cpos[:, op_idx] & (code == pat)
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
    """de-escalation 후보 비트마스크 집합을 생성: 단일 패턴, popcount(투표 수) 기준 이상/정확히 k개 패턴, 그리고 상위 패턴들의 조합(2~5개)을 모두 모아 정렬된 리스트로 반환."""
    masks: set[int] = set()
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
            deesc = cpos[:, op_idx] & np.isin(code, selected_codes)
            rows.append(deesc_metric_row(dataset, rule, features, op, d["y"].astype(int), cpos[:, op_idx], deesc))
    return pd.DataFrame(rows)


def search_gates_for_config(
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
    """한 CNN 설정의 확률값에 대해 모든 임계값 조합과, 내부 기준 상위 패턴 마스크 조합을 전수 평가하여 후보 규칙들의 DataFrame을 만들고 진행상황을 주기적으로 출력/기록."""
    ths = threshold_vectors()
    rows = []
    start = time.time()
    for th_idx, th in enumerate(ths, start=1):
        code_g = codes_from_prob(prob_g, th)
        code_s = codes_from_prob(prob_s, th)
        masks = candidate_masks(rank_codes_internal(g["y"], cpos_g, code_g))
        for mask in masks:
            g_sum = fast_dataset_summary(g["y"], cpos_g, code_g, mask)
            s_sum = fast_dataset_summary(s["y"], cpos_s, code_s, mask)
            rows.append(row_from_fast(config, th, mask, g_sum, s_sum))
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


def plot_detail(detail: pd.DataFrame, out_path: Path, title: str) -> None:
    """내부/외부 데이터셋별 정확도 증가·특이도 증가·민감도 손실을 운영점별로 나란히 3개 패널 선 그래프로 그려 제목과 함께 이미지 파일로 저장."""
    labels = TARGET_OPS
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    colors = {"g1090_internal": "#2F6B9A", "sdata_external": "#C54E2C"}
    for dataset, ls in [("g1090_internal", "-"), ("sdata_external", "--")]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["accuracy_delta"] * 100, marker="o", color=colors[dataset], ls=ls, label=dataset)
        axes[1].plot(x, sub["specificity_gain"] * 100, marker="o", color=colors[dataset], ls=ls, label=dataset)
        axes[2].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=colors[dataset], ls=ls, label=dataset)
    for ax, label in zip(axes, ["Accuracy gain", "Specificity gain", "Sensitivity loss"]):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("percentage points")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def detail_for_winner(row: pd.Series, g: dict, s: dict, cpos_g: np.ndarray, cpos_s: np.ndarray, prob_g: np.ndarray, prob_s: np.ndarray) -> pd.DataFrame:
    """선택된 최종 규칙 행(row)의 임계값과 패턴 마스크를 다시 적용해, 내부/외부 데이터셋에 대한 상세 de-escalation 지표 DataFrame을 재계산."""
    thresholds = np.array([row["threshold_R1"], row["threshold_R2"], row["threshold_R3"], row["threshold_R4"]], dtype=float)
    code_g = codes_from_prob(prob_g, thresholds)
    code_s = codes_from_prob(prob_s, thresholds)
    features = f"thresholds={','.join(f'{x:.2f}' for x in thresholds)}; patterns={row['patterns']}"
    return evaluate_gate_detail(str(row["rule"]), features, int(row["pattern_mask"]), g, s, cpos_g, cpos_s, code_g, code_s)


def main() -> None:
    """4개 지역별 CNN branch(NewRegionGuidedCnn)를 여러 설정(CONFIGS)으로 교차검증 학습시켜 확률을 얻고, AUC를 비교한 뒤,
    각 설정의 확률에 대해 임계값×패턴마스크 조합을 탐색해 내부(및 내부+외부) 통과 de-escalation 규칙을 찾아 결과표/그래프/요약 JSON을 저장."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    print(f"regions={REGIONS}", flush=True)
    write_progress(stage="started", device=str(DEVICE), regions=REGIONS)

    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    cpos_g = clinical_positive_matrix(c_g, thresholds)
    cpos_s = clinical_positive_matrix(c_s, thresholds)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))

    prob_by_config: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_auc = []
    all_logs = []
    for cfg_idx, cfg in enumerate(CONFIGS, start=1):
        print(f"\ntraining {cfg.name} ({cfg_idx}/{len(CONFIGS)})", flush=True)
        logits_g, logits_s, logs = crossfit_config(cfg, xg, g["y"], cpos_g, xs)
        prob_g = 1.0 / (1.0 + np.exp(-logits_g))
        prob_s = 1.0 / (1.0 + np.exp(-logits_s))
        prob_by_config[cfg.name] = (prob_g, prob_s)
        logs.to_csv(OUT_DIR / f"{cfg.name}_training_log.csv", index=False)
        all_logs.append(logs)
        all_auc.append(auc_rows(cfg.name, g, s, clinical_oof, clinical_ext, prob_g, prob_s))
        np.savez_compressed(OUT_DIR / f"{cfg.name}_probabilities.npz", prob_g=prob_g, prob_s=prob_s, regions=np.array(list(REGIONS.keys())))

    auc_all = pd.concat(all_auc, ignore_index=True)
    logs_all = pd.concat(all_logs, ignore_index=True)
    auc_all.to_csv(OUT_DIR / "new_region_guided_cnn_auc_summary.csv", index=False)
    logs_all.to_csv(OUT_DIR / "new_region_guided_cnn_training_log.csv", index=False)

    all_candidates = []
    for cfg_idx, cfg in enumerate(CONFIGS, start=1):
        print(f"\nsearching same-rule gate for {cfg.name} ({cfg_idx}/{len(CONFIGS)})", flush=True)
        prob_g, prob_s = prob_by_config[cfg.name]
        cand = search_gates_for_config(cfg.name, prob_g, prob_s, g, s, cpos_g, cpos_s, cfg_idx, len(CONFIGS))
        cand.to_csv(OUT_DIR / f"{cfg.name}_same_rule_candidates.csv", index=False)
        all_candidates.append(cand)

    summary = pd.concat(all_candidates, ignore_index=True).sort_values(
        ["internal_pass", "internal_selection_score", "internal_mean_accuracy"],
        ascending=[False, False, False],
    )
    summary.to_csv(OUT_DIR / "new_region_guided_cnn_same_rule_all_candidates.csv", index=False)
    internal_passing = summary[summary["internal_pass"]].copy()
    both_passing = internal_passing[internal_passing["external_pass"]].sort_values(
        ["external_mean_accuracy_gain", "external_mean_specificity_gain", "internal_mean_accuracy_gain"],
        ascending=False,
    )
    internal_passing.to_csv(OUT_DIR / "new_region_guided_cnn_internal_passing_ranked.csv", index=False)
    both_passing.to_csv(OUT_DIR / "new_region_guided_cnn_internal_external_passing_ranked.csv", index=False)

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

    with (OUT_DIR / "new_region_guided_cnn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(DEVICE),
                "preprocessing": "aec_128, Gaussian smoothing sigma=1, patient-wise mean normalization, row-z curve/slope/curvature channels",
                "regions_1_indexed_inclusive": REGIONS,
                "target_ops": TARGET_OPS,
                "training_target": "Each region branch predicts non-low SMI from that region only; final de-escalation is a fixed threshold/pattern gate selected on internal OOF.",
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

    print("\nAUC SUMMARY")
    print(auc_all.to_string(index=False))
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
    print("out_dir", OUT_DIR)
    write_progress(stage="done", out_dir=str(OUT_DIR), n_candidates=int(len(summary)))


# 4개 지역별 CNN branch를 여러 학습 설정으로 교차검증 학습해 저위험 확률을 산출하고, AUC를 비교한 뒤 임계값·패턴마스크 조합을
# 전수 탐색하여 내부/외부 데이터에서 통과하는 de-escalation 게이트 규칙을 찾는 전체 파이프라인을 실행.
if __name__ == "__main__":
    main()
