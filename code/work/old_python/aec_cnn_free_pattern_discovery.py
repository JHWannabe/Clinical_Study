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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, OPS, auc_with_p, clinical_scores, deesc_metric_row, load_dataset  # noqa: E402
from aec_region_constrained_cnn_gate import DEVICE, make_channels, standardize_channels_train_apply, stratified_folds  # noqa: E402
from aec_region_cnn_direct_vote_gate import soft_atleast2_np  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_cnn_free_pattern_discovery"
SEEDS = [20260701, 20260711]
K_BRANCHES = 6
GRID_GLOBAL_THRESHOLDS = np.round(np.arange(0.45, 0.81, 0.05), 2)
SAFETY_LIMITS = [0.12, 0.15, 0.18]
UPPER_CI_LIMITS = [0.20, 0.25, 0.30]


@dataclass
class FreeConfig:
    name: str = "free_k6_guarded"
    k: int = K_BRANCHES
    dropout: float = 0.25
    lr: float = 7.0e-4
    weight_decay: float = 2.0e-3
    low_weight: float = 6.0
    global_weight: float = 0.75
    overlap_weight: float = 0.08
    max_epochs: int = 130
    patience: int = 18
    batch_size: int = 96


CFG = FreeConfig()


def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    """시그모이드 함수의 역함수(로짓 변환) — 학습 가능한 중심 위치를 (0,1) 범위 시그모이드 파라미터로 초기화하기 위해 사용."""
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def inverse_softplus(x: np.ndarray) -> np.ndarray:
    """softplus 함수의 역함수 — 학습 가능한 폭(width)이 항상 양수가 되도록 softplus로 재매개변수화한 값을 초기화하기 위해 사용."""
    return np.log(np.expm1(np.maximum(x, 1e-4)))


class SoftWindowCnn(torch.nn.Module):
    """손수 정한 해부학적 구간(REGIONS) 없이, K개의 창(window) 중심/폭 자체를 학습 가능한 파라미터로 두고 CNN 특징을 부드럽게(가우시안 마스크로) 풀링해 각 창의 저위험 투표 로짓과 전역 결합 로짓을 함께 내는 모델."""

    def __init__(self, cfg: FreeConfig, length: int = 128) -> None:
        """K개 창의 중심/폭 초기 파라미터(처음엔 128 위치에 고르게 분산), 공유 CNN 몸통(trunk), 창별 분류 head, 그리고 창들의 로짓을 합쳐 최종 확률을 내는 global_head를 초기화."""
        super().__init__()
        self.cfg = cfg
        self.length = length
        init_centers = np.linspace(12, length - 12, cfg.k)
        init_widths = np.full(cfg.k, 14.0)
        self.center_raw = torch.nn.Parameter(torch.tensor(inverse_sigmoid((init_centers - 1) / (length - 1)), dtype=torch.float32))
        self.width_raw = torch.nn.Parameter(torch.tensor(inverse_softplus(init_widths - 4.0), dtype=torch.float32))
        self.register_buffer("positions", torch.arange(length, dtype=torch.float32))
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv1d(3, 18, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(18),
            torch.nn.SiLU(),
            torch.nn.Conv1d(18, 18, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(18),
            torch.nn.SiLU(),
            torch.nn.Conv1d(18, 18, kernel_size=3, padding=1),
            torch.nn.SiLU(),
        )
        self.heads = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(36, 18),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(cfg.dropout),
                    torch.nn.Linear(18, 1),
                )
                for _ in range(cfg.k)
            ]
        )
        self.global_head = torch.nn.Sequential(
            torch.nn.Linear(cfg.k, 12),
            torch.nn.SiLU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(12, 1),
        )

    def centers_widths_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """재매개변수화된 원시 파라미터로부터 실제 중심 위치(1~length)와 폭을 복원하고, 각 창마다 정규화된 가우시안 마스크(길이 방향 가중치)를 계산."""
        centers = 1.0 + (self.length - 1.0) * torch.sigmoid(self.center_raw)
        widths = 4.0 + F.softplus(self.width_raw)
        dist = (self.positions[None, :] + 1.0 - centers[:, None]) / widths[:, None]
        masks = torch.exp(-0.5 * dist * dist)
        masks = masks / torch.clamp(masks.sum(dim=1, keepdim=True), min=1e-6)
        return centers, widths, masks

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """공유 CNN 몸통으로 특징맵을 뽑은 뒤, 각 창의 가우시안 마스크로 가중 평균/표준편차를 풀링해 창별 head에 통과시켜 K개 로짓을 만들고, 이를 global_head로 결합한 전역 로짓과 함께 반환."""
        z = self.trunk(x)
        _, _, masks = self.centers_widths_masks()
        pooled = []
        for j in range(self.cfg.k):
            m = masks[j][None, None, :]
            mean = (z * m).sum(dim=-1)
            var = ((z - mean[:, :, None]) ** 2 * m).sum(dim=-1)
            pooled.append(torch.cat([mean, torch.sqrt(torch.clamp(var, min=1e-8))], dim=1))
        logits = torch.cat([head(pooled[j]) for j, head in enumerate(self.heads)], dim=1)
        global_logit = self.global_head(logits).squeeze(1)
        return logits, global_logit

    def overlap_penalty(self) -> torch.Tensor:
        """서로 다른 창들의 마스크 간 내적(그람행렬의 비대각 성분) 평균을 계산해, 창들이 서로 겹치지 않고 각기 다른 위치를 보도록 유도하는 정규화 항으로 사용."""
        _, _, masks = self.centers_widths_masks()
        gram = masks @ masks.T
        off = gram - torch.diag(torch.diag(gram))
        return off.sum() / (self.cfg.k * (self.cfg.k - 1))


def clinical_positive_weights(c_g: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """5개 운영점 임계값 중 몇 개를 넘는 임상양성인지에 비례해 훈련 샘플 가중치를 매김 — 임상양성군 학습에 더 집중시키기 위함."""
    mat = np.column_stack([(c_g >= thresholds[op]).astype(float) for op, _ in OPS])
    return 1.0 + 1.5 * mat.mean(axis=1)


def train_loss(
    branch_logits: torch.Tensor,
    global_logit: torch.Tensor,
    target_nonlow: torch.Tensor,
    sample_weight: torch.Tensor,
    cfg: FreeConfig,
    model: SoftWindowCnn,
) -> torch.Tensor:
    """창별 로짓과 전역 로짓 각각에 대해 저SMI(위험군)에 더 큰 가중치(low_weight)를 준 가중 이진교차엔트로피 손실을 계산하고, 창 간 겹침 페널티(overlap_penalty)까지 더해 최종 학습 손실을 만듦."""
    low_guard = torch.where(target_nonlow < 0.5, torch.full_like(target_nonlow, cfg.low_weight), torch.ones_like(target_nonlow))
    weight = sample_weight * low_guard
    target_mat = target_nonlow[:, None].expand_as(branch_logits)
    branch_loss = F.binary_cross_entropy_with_logits(branch_logits, target_mat, weight=weight[:, None].expand_as(branch_logits))
    global_loss = F.binary_cross_entropy_with_logits(global_logit, target_nonlow, weight=weight)
    return branch_loss + cfg.global_weight * global_loss + cfg.overlap_weight * model.overlap_penalty()


def train_fold(
    cfg: FreeConfig,
    x: np.ndarray,
    y_low: np.ndarray,
    sample_weight: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """한 fold에 대해 SoftWindowCnn을 조기종료로 학습시키고, 검증셋(OOF)·외부셋의 창별/전역 확률과, 학습된 창 중심·폭을 포함한 로그를 반환."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SoftWindowCnn(cfg, length=x.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(1 - y_low[train_idx], dtype=torch.float32, device=DEVICE)
    wt = torch.tensor(sample_weight[train_idx], dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(1 - y_low[val_idx], dtype=torch.float32, device=DEVICE)
    wv = torch.tensor(sample_weight[val_idx], dtype=torch.float32, device=DEVICE)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = cfg.patience
    for epoch in range(cfg.max_epochs):
        order = rng.permutation(len(train_idx))
        model.train()
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            bl, gl = model(xt[batch])
            loss = train_loss(bl, gl, yt[batch], wt[batch], cfg, model)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            blv, glv = model(xv)
            val_loss = float(train_loss(blv, glv, yv, wv, cfg, model).item())
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
        val_logits, val_global = model(torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE))
        ext_logits, ext_global = model(torch.tensor(x_ext, dtype=torch.float32, device=DEVICE))
        centers, widths, _ = model.centers_widths_masks()
    return (
        torch.sigmoid(val_logits).detach().cpu().numpy(),
        torch.sigmoid(val_global).detach().cpu().numpy(),
        torch.sigmoid(ext_logits).detach().cpu().numpy(),
        torch.sigmoid(ext_global).detach().cpu().numpy(),
        {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_loss),
            **{f"center_{j+1}": float(centers[j].detach().cpu()) for j in range(cfg.k)},
            **{f"width_{j+1}": float(widths[j].detach().cpu()) for j in range(cfg.k)},
        },
    )


def crossfit_free_cnn(
    cfg: FreeConfig,
    xg: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드 x 5-fold로 train_fold를 반복해, 시드에 걸쳐 평균낸 내부 OOF/외부 창별·전역 확률과 전체 학습 로그를 반환."""
    oof_prob_runs = []
    oof_global_runs = []
    ext_prob_runs = []
    ext_global_runs = []
    logs = []
    for seed in SEEDS:
        oof_prob = np.zeros((len(y), cfg.k), dtype=float)
        oof_global = np.zeros(len(y), dtype=float)
        ext_probs = []
        ext_globals = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            vp, vg, ep, eg, info = train_fold(cfg, xg, y, sample_weight, tr, va, xs, seed + fold_id * 101)
            oof_prob[va] = vp
            oof_global[va] = vg
            ext_probs.append(ep)
            ext_globals.append(eg)
            logs.append({"seed": seed, "fold": fold_id, **info})
        oof_prob_runs.append(oof_prob)
        oof_global_runs.append(oof_global)
        ext_prob_runs.append(np.mean(ext_probs, axis=0))
        ext_global_runs.append(np.mean(ext_globals, axis=0))
    return (
        np.mean(oof_prob_runs, axis=0),
        np.mean(oof_global_runs, axis=0),
        np.mean(ext_prob_runs, axis=0),
        np.mean(ext_global_runs, axis=0),
        pd.DataFrame(logs),
    )


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """이항비율 k/n에 대한 Wilson 신뢰구간의 상한(기본 95%)을 계산 — 표본이 적은 패턴의 사건율 안전성을 보수적으로 평가하기 위함."""
    if n <= 0:
        return np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return min(1.0, center + half)


def codes_from_probs(prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """각 창 확률을 창별 임계값과 비교해 +/- 투표로 변환하고, K개 비트를 하나의 정수 패턴 코드로 인코딩."""
    votes = prob >= thresholds[None, :]
    code = np.zeros(len(prob), dtype=np.int16)
    for j in range(prob.shape[1]):
        code += votes[:, j].astype(np.int16) * (1 << j)
    return code


def pattern_str(code: int, k: int = K_BRANCHES) -> str:
    """정수 패턴 코드를 "+/-" 문자열(각 창의 투표 방향)로 사람이 읽기 쉽게 변환."""
    return "".join("+" if code & (1 << j) else "-" for j in range(k))


def selected_patterns_by_risk(
    g: dict,
    cpos_g: np.ndarray,
    code_g: np.ndarray,
    k: int,
    safety_limit: float,
    upper_ci_limit: float,
) -> int:
    """가능한 모든 2^k개 패턴 각각에 대해, 임상양성이면서 그 패턴에 해당하는 g1090 환자군의 사건율과 Wilson 상한을 확인해, 둘 다 안전 기준 이하인 패턴들만 비트마스크로 모아 반환 — 표본이 12명 미만인 패턴은 제외."""
    y = g["y"].astype(bool)
    cpos_any = cpos_g.any(axis=1)
    mask = 0
    for code in range(1 << k):
        idx = cpos_any & (code_g == code)
        n = int(idx.sum())
        events = int((idx & y).sum())
        if n < 12:
            continue
        event_rate = events / n
        upper = wilson_upper(events, n)
        if event_rate <= safety_limit and upper <= upper_ci_limit:
            mask |= 1 << code
    return mask


def evaluate_mask(
    rule: str,
    features: str,
    mask: int,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    code_g: np.ndarray,
    code_s: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """선택된 패턴 집합(mask)에 해당하는 환자를 강등 대상으로 삼아 내부/외부 x 모든 운영점에서 de-escalation 성능 지표를 계산."""
    selected = [code for code in range(1 << k) if mask & (1 << code)]
    rows = []
    for dataset, d, cpos, code in [("g1090_internal", g, cpos_g, code_g), ("sdata_external", s, cpos_s, code_s)]:
        for op_idx, (op, _) in enumerate(OPS):
            deesc = cpos[:, op_idx] & np.isin(code, selected)
            rows.append(deesc_metric_row(dataset, rule, features, op, d["y"].astype(int), cpos[:, op_idx], deesc))
    return pd.DataFrame(rows)


def summarize_internal(detail: pd.DataFrame) -> dict:
    """g1090 내부 결과에서 여러 운영점 중 가장 나쁜 경우를 뽑아 안전성 제약 판정용 요약통계로 압축."""
    gi = detail[detail["dataset"].eq("g1090_internal")]
    return {
        "internal_min_p_loss": float(gi["sensitivity_loss_p_exact"].min(skipna=True)),
        "internal_max_sens_loss": float(gi["sensitivity_loss"].max(skipna=True)),
        "internal_min_spec_gain": float(gi["specificity_gain"].min(skipna=True)),
        "internal_mean_spec_gain": float(gi["specificity_gain"].mean(skipna=True)),
        "internal_max_fisher_p": float(gi["deesc_event_fisher_p"].max(skipna=True)),
        "internal_min_deesc_n": int(gi["deesc_n"].min(skipna=True)),
        "internal_mean_event_rate": float(gi["deesc_event_rate"].mean(skipna=True)),
    }


def internal_score(summary: dict) -> tuple[bool, float]:
    """내부 요약통계가 안전성 제약(민감도손실 상한, 특이도이득 양수, Fisher p, 표본수, 평균사건율 등)을 모두 통과하는지 판정하고, 특이도이득 중심의 선택점수를 계산 — 위반 시 큰 페널티를 줘서 순위 밖으로 밀어냄."""
    survives = (
        summary["internal_min_p_loss"] >= 0.05
        and summary["internal_max_sens_loss"] <= 0.08
        and summary["internal_min_spec_gain"] > 0
        and summary["internal_max_fisher_p"] < 0.05
        and summary["internal_min_deesc_n"] >= 25
        and summary["internal_mean_event_rate"] <= 0.12
    )
    score = (
        3.0 * summary["internal_min_spec_gain"]
        + 1.3 * summary["internal_mean_spec_gain"]
        - 0.9 * summary["internal_max_sens_loss"]
        - 0.25 * summary["internal_mean_event_rate"]
        - 0.02 * summary["internal_max_fisher_p"]
    )
    if not survives:
        score -= 10.0
    return survives, float(score)


def search_logical_pattern_gate(
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    prob_g: np.ndarray,
    prob_s: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """여러 전역 임계값(모든 창에 동일 적용) 후보와, 창별로 안전 기준을 처음 만족하는 임계값 조합 하나를 만든 뒤, 각 임계값 조합 x 안전기준(safety/upper CI limit) 조합마다 안전 패턴 집합을 골라 de-escalation을 평가 — g1090 내부 기준으로 가장 좋은 조합을 골라 그 상세결과와 패턴분포까지 반환."""
    thresholds_list = [np.full(k, t, dtype=float) for t in GRID_GLOBAL_THRESHOLDS]
    # Add branch-wise safety thresholds. These are not optimized for specificity;
    # they are the first branch cutoffs that make the branch-positive group look low-risk.
    branch_thresholds = []
    y = g["y"].astype(bool)
    cpos_any = cpos_g.any(axis=1)
    for j in range(k):
        chosen = 0.80
        for t in GRID_GLOBAL_THRESHOLDS:
            idx = cpos_any & (prob_g[:, j] >= t)
            n = int(idx.sum())
            e = int((idx & y).sum())
            if n >= 25 and e / n <= 0.12 and wilson_upper(e, n) <= 0.30:
                chosen = float(t)
                break
        branch_thresholds.append(chosen)
    thresholds_list.append(np.array(branch_thresholds, dtype=float))

    summary_rows = []
    detail_map = {}
    for thresholds in thresholds_list:
        code_g = codes_from_probs(prob_g, thresholds)
        code_s = codes_from_probs(prob_s, thresholds)
        for safety in SAFETY_LIMITS:
            for upper in UPPER_CI_LIMITS:
                mask = selected_patterns_by_risk(g, cpos_g, code_g, k, safety, upper)
                if mask == 0:
                    continue
                patterns = ",".join(pattern_str(code, k) for code in range(1 << k) if mask & (1 << code))
                rule = f"free_cnn_pattern_s{safety:.2f}_u{upper:.2f}"
                detail = evaluate_mask(rule, patterns, mask, g, s, cpos_g, cpos_s, code_g, code_s, k)
                summary = summarize_internal(detail)
                survives, score = internal_score(summary)
                key = (tuple(np.round(thresholds, 3)), safety, upper, mask)
                detail_map[key] = detail
                summary_rows.append(
                    {
                        "thresholds": "|".join(f"{x:.2f}" for x in thresholds),
                        "safety_limit": safety,
                        "upper_ci_limit": upper,
                        "pattern_mask": int(mask),
                        "n_patterns": int(bin(mask).count("1")),
                        "patterns": patterns,
                        "survives_internal_constraints": survives,
                        "internal_selection_score": score,
                        **summary,
                    }
                )
    summary_df = pd.DataFrame(summary_rows).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    best = summary_df.iloc[0]
    thresholds = np.array([float(x) for x in str(best["thresholds"]).split("|")], dtype=float)
    key = (tuple(np.round(thresholds, 3)), float(best["safety_limit"]), float(best["upper_ci_limit"]), int(best["pattern_mask"]))
    best_detail = detail_map[key].copy()
    dist = pattern_distribution(g, s, cpos_g, cpos_s, codes_from_probs(prob_g, thresholds), codes_from_probs(prob_s, thresholds), int(best["pattern_mask"]), CFG.k)
    return summary_df, best_detail, dist


def pattern_distribution(
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    code_g: np.ndarray,
    code_s: np.ndarray,
    mask: int,
    k: int,
) -> pd.DataFrame:
    """가능한 모든 패턴 코드별로, 내부/외부 x 각 운영점에서 그 패턴에 속하는 임상양성 환자 수/사건수/사건율/Wilson 상한 및 최종 선택 여부를 표로 정리."""
    rows = []
    for dataset, d, cpos, code in [("g1090_internal", g, cpos_g, code_g), ("sdata_external", s, cpos_s, code_s)]:
        y = d["y"].astype(bool)
        for op_idx, (op, _) in enumerate(OPS):
            cp = cpos[:, op_idx]
            for pat in range(1 << k):
                idx = cp & (code == pat)
                n = int(idx.sum())
                e = int((idx & y).sum())
                rows.append(
                    {
                        "dataset": dataset,
                        "operating_point": op,
                        "pattern_code": pat,
                        "pattern": pattern_str(pat, k),
                        "selected": bool(mask & (1 << pat)),
                        "n": n,
                        "events": e,
                        "event_rate": e / n if n else np.nan,
                        "upper95": wilson_upper(e, n) if n else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def auc_table(
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    branch_prob_g: np.ndarray,
    branch_prob_s: np.ndarray,
    global_g: np.ndarray,
    global_s: np.ndarray,
) -> pd.DataFrame:
    """임상단독/창확률평균단독/전역head단독/임상+창평균 결합 모델의 내부·외부 AUC와 임상단독 대비 증분을 비교표로 만듦."""
    rows = []
    cg_auc, cg_p = auc_with_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_with_p(s["y"], clinical_ext)
    # Higher score means low-SMI risk. CNN outputs low-risk probability.
    aec_risk_g = -np.mean(branch_prob_g, axis=1)
    aec_risk_s = -np.mean(branch_prob_s, axis=1)
    ag_auc, ag_p = auc_with_p(g["y"], aec_risk_g)
    as_auc, as_p = auc_with_p(s["y"], aec_risk_s)
    gg_auc, gg_p = auc_with_p(g["y"], -global_g)
    gs_auc, gs_p = auc_with_p(s["y"], -global_s)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260701)
    oof = np.zeros(len(g["y"]), dtype=float)
    for fold_id, (tr, va) in enumerate(folds.split(np.zeros(len(g["y"])), g["y"])):
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260701 + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], aec_risk_g[tr]]), g["y"][tr])
        oof[va] = model.decision_function(np.column_stack([clinical_oof[va], aec_risk_g[va]]))
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260799)
    model.fit(np.column_stack([clinical_oof, aec_risk_g]), g["y"])
    ext = model.decision_function(np.column_stack([clinical_ext, aec_risk_s]))
    pg_auc, pg_p = auc_with_p(g["y"], oof)
    ps_auc, ps_p = auc_with_p(s["y"], ext)
    rows.extend(
        [
            {"model": "clinical_only", "internal_auc": cg_auc, "internal_auc_p": cg_p, "external_auc": cs_auc, "external_auc_p": cs_p, "internal_delta_vs_clinical": 0.0, "external_delta_vs_clinical": 0.0},
            {"model": "free_cnn_branch_mean", "internal_auc": ag_auc, "internal_auc_p": ag_p, "external_auc": as_auc, "external_auc_p": as_p, "internal_delta_vs_clinical": ag_auc - cg_auc, "external_delta_vs_clinical": as_auc - cs_auc},
            {"model": "free_cnn_global_head", "internal_auc": gg_auc, "internal_auc_p": gg_p, "external_auc": gs_auc, "external_auc_p": gs_p, "internal_delta_vs_clinical": gg_auc - cg_auc, "external_delta_vs_clinical": gs_auc - cs_auc},
            {"model": "clinical_plus_free_cnn_branch_mean", "internal_auc": pg_auc, "internal_auc_p": pg_p, "external_auc": ps_auc, "external_auc_p": ps_p, "internal_delta_vs_clinical": pg_auc - cg_auc, "external_delta_vs_clinical": ps_auc - cs_auc},
        ]
    )
    return pd.DataFrame(rows)


def plot_result(detail: pd.DataFrame, dist: pd.DataFrame, train_log: pd.DataFrame, out_path: Path) -> None:
    """운영점별 특이도이득/민감도손실, 학습된 창들의 중심 위치와 폭, 외부 S85 운영점에서 선택된 패턴들의 사건율을 3패널 그래프로 그려 PNG로 저장."""
    labels = [op for op, _ in OPS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {"g1090_internal": "#2c7fb8", "sdata_external": "#d95f02"}
    for dataset in ["g1090_internal", "sdata_external"]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", color=colors[dataset], label=f"{dataset} spec gain")
        axes[0].plot(x, sub["sensitivity_loss"] * 100, marker="x", color=colors[dataset], ls="--", label=f"{dataset} sens loss")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Percentage points")
    axes[0].set_title("Free CNN pattern gate", loc="left", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    centers = train_log[[f"center_{j+1}" for j in range(CFG.k)]].mean(axis=0).to_numpy()
    widths = train_log[[f"width_{j+1}" for j in range(CFG.k)]].mean(axis=0).to_numpy()
    order = np.argsort(centers)
    axes[1].errorbar(np.arange(CFG.k), centers[order], yerr=widths[order], fmt="o", color="#756bb1")
    axes[1].set_xticks(np.arange(CFG.k))
    axes[1].set_xticklabels([f"B{j+1}" for j in order])
    axes[1].set_ylim(1, 128)
    axes[1].set_ylabel("AEC position")
    axes[1].set_title("Learned soft-window centers", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.25)

    sub = dist[dist["dataset"].eq("sdata_external") & dist["operating_point"].eq("S85") & dist["selected"]].copy()
    if sub.empty:
        sub = dist[dist["dataset"].eq("sdata_external") & dist["operating_point"].eq("S85")].sort_values("n", ascending=False).head(8)
    axes[2].bar(np.arange(len(sub)), sub["event_rate"] * 100, color="#31a354")
    axes[2].set_xticks(np.arange(len(sub)))
    axes[2].set_xticklabels(sub["pattern"].tolist(), rotation=45, ha="right")
    axes[2].set_ylabel("Low SMI %")
    axes[2].set_title("External S85 selected patterns", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 손수 고른 해부학적 구간이나 손수 만든 특징 없이, CNN이 곡선
    위 어디를 볼지(K개 창의 중심/폭)까지 스스로 학습하고, 그 창들의 +/- 투표 패턴 중 실제로 안전한
    조합만 골라 de-escalation에 쓰면 어떤가? — 완전한 자유 탐색(free discovery) 접근):

    1. g1090/sdata를 로드하고 3채널(곡선/기울기/곡률) 텐서와 임상양성 가중치를 준비.
    2. SoftWindowCnn을 여러 시드 x 5-fold로 학습(crossfit_free_cnn) — K=6개의 학습 가능한 소프트
       창이 각각 저위험 여부에 투표하고, 창들의 로짓을 합친 전역 확률도 함께 산출.
    3. 여러 전역/창별 임계값 조합 x 안전기준(사건율 상한, Wilson 상한) 조합에 대해, g1090 내부에서
       사건율이 충분히 낮은 +/- 패턴들만 골라(selected_patterns_by_risk) de-escalation을 평가하고,
       내부 기준으로 가장 좋은 조합을 최종 선택(search_logical_pattern_gate).
    4. 임상단독/창평균/전역head/결합 모델의 AUC 비교표와 패턴별 분포표도 계산.
    5. 결과를 그래프로 시각화하고, 학습로그/확률/AUC/패턴선택요약/de-escalation상세/패턴분포를
       CSV(및 npz)로, 선택된 설정을 JSON으로 저장한 뒤 콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    cpos_g = np.column_stack([(c_g >= thresholds[op]) for op, _ in OPS])
    cpos_s = np.column_stack([(c_s >= thresholds[op]) for op, _ in OPS])
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
    sample_weight = clinical_positive_weights(c_g, thresholds)
    print("training free CNN", flush=True)
    prob_g, global_g, prob_s, global_s, train_log = crossfit_free_cnn(CFG, xg, g["y"], sample_weight, xs)
    train_log.to_csv(OUT_DIR / "free_cnn_training_log.csv", index=False)
    np.savez_compressed(OUT_DIR / "free_cnn_probabilities.npz", prob_g=prob_g, global_g=global_g, prob_s=prob_s, global_s=global_s)
    auc_df = auc_table(g, s, clinical_oof, clinical_ext, prob_g, prob_s, global_g, global_s)
    auc_df.to_csv(OUT_DIR / "free_cnn_auc_summary.csv", index=False)

    print("searching logical patterns", flush=True)
    summary, detail, dist = search_logical_pattern_gate(g, s, cpos_g, cpos_s, prob_g, prob_s, CFG.k)
    summary.to_csv(OUT_DIR / "free_cnn_pattern_selection_summary.csv", index=False)
    detail.to_csv(OUT_DIR / "free_cnn_pattern_deescalation_details.csv", index=False)
    dist.to_csv(OUT_DIR / "free_cnn_pattern_distribution.csv", index=False)
    plot_result(detail, dist, train_log, OUT_DIR / "free_cnn_pattern_plot.png")

    best = summary.iloc[0]
    with (OUT_DIR / "free_cnn_pattern_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "model": CFG.__dict__,
                "thresholds": str(best["thresholds"]),
                "safety_limit": float(best["safety_limit"]),
                "upper_ci_limit": float(best["upper_ci_limit"]),
                "selected_patterns": str(best["patterns"]),
                "logic": "No hand-crafted/SVM features are used. CNN soft-window branches are trained from outcome only; branch signs are converted to patterns; patterns are selected on internal OOF by event-rate and Wilson upper-CI safety limits.",
            },
            f,
            indent=2,
        )

    print("\nAUC SUMMARY")
    print(auc_df.to_string(index=False))
    print("\nPATTERN SELECTION TOP")
    print(summary.head(20).to_string(index=False))
    print("\nDE-ESCALATION")
    show = [
        "dataset",
        "operating_point",
        "clinical_sensitivity",
        "post_sensitivity",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "clinical_specificity",
        "post_specificity",
        "specificity_gain",
        "specificity_gain_p_exact",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
        "features",
    ]
    print(detail[show].to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    # 데이터 로드 -> 창 위치까지 스스로 학습하는 SoftWindowCnn을 다중시드 x 5-fold로 학습 -> 임계값x
    # 안전기준 조합별로 내부에서 안전한 +/- 패턴 집합을 탐색해 최적 조합 선택 -> AUC/패턴분포 계산과
    # 결과 저장 순으로 실행된다.
    main()
