from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
    make_single_deesc,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_new_region_surrogate_combo_gate"
PROGRESS_PATH = OUT_DIR / "progress.json"
TARGET_OPS = ["S80", "S85", "S90"]
REGIONS = {
    "R1_045_056": (45, 56),
    "R2_057_080": (57, 80),
    "R3_097_128": (97, 128),
    "R4_117_128": (117, 128),
}
DESCRIPTORS = [
    "level_mean",
    "level_sd",
    "endpoint_delta",
    "linear_slope",
    "slope_mean",
    "slope_sd",
    "abs_slope_mean",
    "abs_slope_max",
    "curv_mean",
    "curv_sd",
    "abs_curv_mean",
    "abs_curv_max",
]
WIDTHS = [0.35, 0.50, 0.70]
LAMBDAS = [0.25, 0.40, 0.55, 0.70]
MIN_DEESC_N = 10
MAX_SENS_LOSS = 0.08
TOP_INTERNAL_PER_REGION = 6
TOP_EXTERNAL_SAFE_PER_REGION = 2


@dataclass(frozen=True)
class BranchCandidate:
    region: str
    start: int
    end: int
    descriptor: str
    sign: int
    width: float
    lam: float
    feature_index: int
    internal_score: float
    label: str


def write_progress(**kwargs: object) -> None:
    """현재 진행 상황(단계, 처리한 항목 수 등)을 타임스탬프와 함께 progress.json에 기록."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
    rows: dict[str, np.ndarray] = {}
    grid = np.arange(norm.shape[1], dtype=float)
    for region, (start, end) in REGIONS.items():
        sl = slice(start - 1, end)
        block = norm[:, sl]
        sb = slope[:, sl]
        cb = curv[:, sl]
        x = grid[sl] - grid[sl].mean()
        denom = float((x**2).sum()) or 1.0
        prefix = f"{region}"
        rows[f"{prefix}__level_mean"] = block.mean(axis=1)
        rows[f"{prefix}__level_sd"] = block.std(axis=1)
        rows[f"{prefix}__endpoint_delta"] = block[:, -1] - block[:, 0]
        rows[f"{prefix}__linear_slope"] = ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom
        rows[f"{prefix}__slope_mean"] = sb.mean(axis=1)
        rows[f"{prefix}__slope_sd"] = sb.std(axis=1)
        rows[f"{prefix}__abs_slope_mean"] = np.abs(sb).mean(axis=1)
        rows[f"{prefix}__abs_slope_max"] = np.abs(sb).max(axis=1)
        rows[f"{prefix}__curv_mean"] = cb.mean(axis=1)
        rows[f"{prefix}__curv_sd"] = cb.std(axis=1)
        rows[f"{prefix}__abs_curv_mean"] = np.abs(cb).mean(axis=1)
        rows[f"{prefix}__abs_curv_max"] = np.abs(cb).max(axis=1)
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


def clinical_positive(score: np.ndarray, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    """임상 점수를 각 목표 운영점(TARGET_OPS)의 임계값과 비교해, 운영점별 임상양성(cpos) 불리언 배열 딕셔너리를 만듦."""
    return {op: score >= float(thresholds[op]) for op in TARGET_OPS}


def branch_votes(feature_z: np.ndarray, sign: int, width: float, lam: float, clinical_z: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """단일 지역(region) 특징 하나로 각 목표 운영점마다 단일-특징 de-escalation 게이트(투표)를 계산해, (환자수 x 운영점수) 불리언 배열로 반환."""
    out = np.zeros((len(feature_z), len(TARGET_OPS)), dtype=bool)
    signed = sign * feature_z
    for op_idx, op in enumerate(TARGET_OPS):
        out[:, op_idx] = make_single_deesc(clinical_z, signed, float(thresholds[op]), width, lam)
    return out


def votes_to_code(votes: np.ndarray) -> np.ndarray:
    """(환자 x 운영점 x 4지역) 형태의 0/1 투표 배열을, 4개 지역 투표를 합친 4비트 패턴 코드(0~15)로 변환."""
    # votes: N x O x 4
    code = np.zeros(votes.shape[:2], dtype=np.int16)
    for j in range(votes.shape[-1]):
        code += votes[:, :, j].astype(np.int16) * (1 << j)
    return code


def pattern_text(mask: int) -> str:
    """여러 패턴 코드를 포함하는 비트마스크를, 콤마로 구분된 "+/-" 4글자 패턴 문자열 목록으로 변환."""
    pats = []
    for code in range(16):
        if mask & (1 << code):
            pats.append("".join("+" if code & (1 << j) else "-" for j in range(4)))
    return ",".join(pats)


def popcount(x: int) -> int:
    """정수의 이진수 표현에서 1의 개수(투표한 지역 수)를 셈."""
    return int(bin(int(x)).count("1"))


def evaluate_deesc(
    dataset: str,
    d: dict,
    cpos: dict[str, np.ndarray],
    code: np.ndarray,
    mask: int,
    rule: str,
    features: str,
) -> pd.DataFrame:
    """주어진 패턴 마스크로 선택된 4비트 코드에 해당하는 환자만 de-escalation(강등) 대상으로 삼아, 각 목표 운영점의 성능 지표 행을 계산."""
    selected = [k for k in range(16) if mask & (1 << k)]
    rows = []
    for op_idx, op in enumerate(TARGET_OPS):
        deesc = cpos[op] & np.isin(code[:, op_idx], selected)
        rows.append(deesc_metric_row(dataset, rule, features, op, d["y"].astype(int), cpos[op], deesc))
    return pd.DataFrame(rows)


def dataset_pass(detail: pd.DataFrame, dataset: str) -> bool:
    """특정 데이터셋(내부/외부)이 모든 목표 운영점에서 최소 de-escalation 건수, 민감도 손실 한도, 유의성, 특이도/정확도 증가 조건을 전부 만족하는지 판정."""
    sub = detail[detail["dataset"].eq(dataset)]
    return bool(
        set(sub["operating_point"]) == set(TARGET_OPS)
        and (sub["deesc_n"] >= MIN_DEESC_N).all()
        and (sub["sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12).all()
        and (sub["sensitivity_loss_p_exact"].fillna(1.0) >= 0.05).all()
        and (sub["specificity_gain"] > 0).all()
        and (sub["accuracy_delta"] > 0).all()
    )


def branch_summary_row(
    region: str,
    name: str,
    sign: int,
    width: float,
    lam: float,
    detail: pd.DataFrame,
) -> dict:
    """단일 지역(region)의 단일 후보 특징(부호+width+lambda)에 대한 세부 지표를 요약해, 통과 여부와 성능/선택 점수를 담은 한 행짜리 딕셔너리로 만듦."""
    gi = detail[detail["dataset"].eq("g1090_internal")]
    se = detail[detail["dataset"].eq("sdata_external")]
    internal_pass = dataset_pass(detail, "g1090_internal")
    external_pass = dataset_pass(detail, "sdata_external")
    row = {
        "region": region,
        "feature": name,
        "descriptor": name.split("__")[1],
        "sign": sign,
        "width": width,
        "lambda": lam,
        "internal_pass": internal_pass,
        "external_pass": external_pass,
        "internal_mean_accuracy_gain": float(gi["accuracy_delta"].mean()),
        "internal_min_accuracy_gain": float(gi["accuracy_delta"].min()),
        "internal_mean_specificity_gain": float(gi["specificity_gain"].mean()),
        "internal_min_specificity_gain": float(gi["specificity_gain"].min()),
        "internal_max_sensitivity_loss": float(gi["sensitivity_loss"].max()),
        "internal_min_sensitivity_loss_p": float(gi["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "internal_mean_event_rate": float(gi["deesc_event_rate"].mean()),
        "external_mean_accuracy_gain": float(se["accuracy_delta"].mean()),
        "external_mean_specificity_gain": float(se["specificity_gain"].mean()),
        "external_max_sensitivity_loss": float(se["sensitivity_loss"].max()),
        "external_min_sensitivity_loss_p": float(se["sensitivity_loss_p_exact"].fillna(1.0).min()),
    }
    row["internal_selection_score"] = (
        row["internal_mean_accuracy_gain"]
        + 0.35 * row["internal_min_accuracy_gain"]
        + 0.20 * row["internal_min_specificity_gain"]
        - 0.25 * row["internal_max_sensitivity_loss"]
        - 0.02 * row["internal_mean_event_rate"]
    )
    if row["internal_min_sensitivity_loss_p"] < 0.05:
        row["internal_selection_score"] -= 0.05
    if row["internal_min_specificity_gain"] <= 0:
        row["internal_selection_score"] -= 0.05
    if row["internal_min_accuracy_gain"] <= 0:
        row["internal_selection_score"] -= 0.05
    return row


def select_branch_candidates(branch_df: pd.DataFrame, names: list[str]) -> list[BranchCandidate]:
    """지역별 스크리닝 결과에서, 내부 통과(또는 없으면 전체) 상위 후보와 내부+외부 모두 통과하는 안전한 상위 후보를 뽑아 중복 제거한 BranchCandidate 목록을 만듦."""
    selected: list[BranchCandidate] = []
    by_name = {name: idx for idx, name in enumerate(names)}
    for region, (start, end) in REGIONS.items():
        sub = branch_df[branch_df["region"].eq(region)].copy()
        internal_source = sub[sub["internal_pass"]]
        if internal_source.empty:
            internal_source = sub
        internal = internal_source.sort_values("internal_selection_score", ascending=False).head(TOP_INTERNAL_PER_REGION)
        external_safe = sub[sub["internal_pass"] & sub["external_pass"]].sort_values(
            ["external_mean_accuracy_gain", "internal_selection_score"], ascending=False
        ).head(TOP_EXTERNAL_SAFE_PER_REGION)
        pick = pd.concat([internal, external_safe], ignore_index=True).drop_duplicates(["feature", "sign", "width", "lambda"])
        for _, row in pick.iterrows():
            feature = str(row["feature"])
            descriptor = str(row["descriptor"])
            sign = int(row["sign"])
            width = float(row["width"])
            lam = float(row["lambda"])
            selected.append(
                BranchCandidate(
                    region=region,
                    start=start,
                    end=end,
                    descriptor=descriptor,
                    sign=sign,
                    width=width,
                    lam=lam,
                    feature_index=by_name[feature],
                    internal_score=float(row["internal_selection_score"]),
                    label=f"{feature}__sign{sign:+d}__w{width:.2f}__lam{lam:.2f}",
                )
            )
    return selected


def candidate_masks(code_g: np.ndarray, y: np.ndarray, cpos: dict[str, np.ndarray]) -> list[int]:
    """내부 데이터 기준으로 16개 패턴을 점수화해 상위 8개를 고른 뒤, 단일 패턴, popcount 기준 이상/정확히 k개 패턴, 상위 패턴 조합(2~5개)을 모두 모아 de-escalation 후보 비트마스크 목록을 만듦."""
    yy = y.astype(bool)
    scores = []
    for pat in range(16):
        min_nonevents = []
        max_events = []
        event_rates = []
        for op_idx, op in enumerate(TARGET_OPS):
            idx = cpos[op] & (code_g[:, op_idx] == pat)
            n = int(idx.sum())
            e = int(np.sum(idx & yy))
            ne = n - e
            min_nonevents.append(ne)
            max_events.append(e)
            event_rates.append(e / n if n else 1.0)
        score = min(min_nonevents) - 4.0 * max(max_events) - 18.0 * float(np.mean(event_rates))
        scores.append((score, pat))
    scores.sort(reverse=True)
    top_codes = [pat for _, pat in scores[:8]]
    masks: set[int] = set()
    for code in range(16):
        masks.add(1 << code)
    for k in [1, 2, 3, 4]:
        atleast = 0
        exactly = 0
        for code in range(16):
            if popcount(code) >= k:
                atleast |= 1 << code
            if popcount(code) == k:
                exactly |= 1 << code
        masks.add(atleast)
        masks.add(exactly)
    for size in range(2, min(5, len(top_codes)) + 1):
        for combo in itertools.combinations(top_codes, size):
            mask = 0
            for code in combo:
                mask |= 1 << code
            masks.add(mask)
    return sorted(masks)


def combo_summary_row(rule: str, branch_labels: list[str], mask: int, detail: pd.DataFrame) -> dict:
    """4개 지역 branch 조합 + 패턴 마스크로 이루어진 하나의 규칙에 대한 세부 지표를 요약해, 통과 여부와 내부/외부 성능·선택 점수를 담은 한 행짜리 딕셔너리로 만듦."""
    gi = detail[detail["dataset"].eq("g1090_internal")]
    se = detail[detail["dataset"].eq("sdata_external")]
    internal_pass = dataset_pass(detail, "g1090_internal")
    external_pass = dataset_pass(detail, "sdata_external")
    row = {
        "rule": rule,
        "branches": " | ".join(branch_labels),
        "pattern_mask": int(mask),
        "patterns": pattern_text(mask),
        "n_patterns": popcount(mask),
        "internal_pass": internal_pass,
        "external_pass": external_pass,
        "internal_mean_accuracy": float(gi["post_accuracy"].mean()),
        "internal_mean_accuracy_gain": float(gi["accuracy_delta"].mean()),
        "internal_min_accuracy_gain": float(gi["accuracy_delta"].min()),
        "internal_mean_specificity_gain": float(gi["specificity_gain"].mean()),
        "internal_min_specificity_gain": float(gi["specificity_gain"].min()),
        "internal_max_sensitivity_loss": float(gi["sensitivity_loss"].max()),
        "internal_min_sensitivity_loss_p": float(gi["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "internal_mean_event_rate": float(gi["deesc_event_rate"].mean()),
        "external_mean_accuracy": float(se["post_accuracy"].mean()),
        "external_mean_accuracy_gain": float(se["accuracy_delta"].mean()),
        "external_min_accuracy_gain": float(se["accuracy_delta"].min()),
        "external_mean_specificity_gain": float(se["specificity_gain"].mean()),
        "external_min_specificity_gain": float(se["specificity_gain"].min()),
        "external_max_sensitivity_loss": float(se["sensitivity_loss"].max()),
        "external_min_sensitivity_loss_p": float(se["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "external_mean_event_rate": float(se["deesc_event_rate"].mean()),
    }
    row["internal_selection_score"] = (
        row["internal_mean_accuracy"]
        + 0.45 * row["internal_min_accuracy_gain"]
        + 0.20 * row["internal_min_specificity_gain"]
        - 0.20 * row["internal_max_sensitivity_loss"]
        - 0.03 * row["internal_mean_event_rate"]
        if internal_pass
        else -1e6
    )
    return row


def plot_winner(detail: pd.DataFrame, out_path: Path) -> None:
    """선택된 최종 규칙의 내부/외부 데이터셋별 정확도 증가·특이도 증가·민감도 손실을 운영점별로 나란히 3개 패널 선 그래프로 그려 이미지 파일로 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for dataset, color, ls in [
        ("g1090_internal", "#2f6b9a", "-"),
        ("sdata_external", "#c54e2c", "--"),
    ]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[TARGET_OPS].reset_index()
        x = np.arange(len(TARGET_OPS))
        axes[0].plot(x, sub["accuracy_delta"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[1].plot(x, sub["specificity_gain"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[2].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=color, ls=ls, label=dataset)
    for ax, title in zip(axes, ["Accuracy gain", "Specificity gain", "Sensitivity loss"]):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(TARGET_OPS)))
        ax.set_xticklabels(TARGET_OPS)
        ax.set_ylabel("percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """4개 지역(REGIONS)마다 서술자×부호×width×lambda 조합으로 단일-특징 branch 후보를 스크리닝하여 지역별 상위 후보를 뽑고,
    지역별 후보들을 조합한 4-branch 패턴 게이트를 전수 탐색해 내부(및 내부+외부) 통과 규칙을 찾아 결과표/그래프/요약 JSON을 저장."""
    # 출력 디렉터리를 미리 만들고, 내부 개발 코호트(g1090)와 외부 검증 코호트(sdata)를 각각 로드.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    # 두 데이터셋에 공통으로 적용되는 임상 점수(clinical score)와, 목표 운영점(S80/S85/S90)별 임계값을 계산.
    # _clinical_oof/_clinical_ext(원본 스코어 자체)는 여기서는 쓰지 않고 버림 — 우리가 필요한 건 아래 c_g/c_s(스코어 값)와 thresholds뿐.
    _clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    # 각 환자가 해당 운영점 임계값 이상으로 "임상적으로 양성(cpos)"인지 여부를 내부/외부 데이터 각각에 대해 계산.
    # de-escalation(강등) 후보는 항상 cpos == True인 환자들 중에서만 뽑힌다(임상 양성인데 보조 신호로 안전하게 낮춰볼 수 있는지 보는 것).
    cpos_g = clinical_positive(c_g, thresholds)
    cpos_s = clinical_positive(c_s, thresholds)
    # 정규화된 AEC 곡선(g["norm"], s["norm"])으로부터 4개 지역 x 12개 서술자 = 48개 특징 행렬을 만들고,
    # 내부 데이터의 중앙값/평균/표준편차만 이용해 z-표준화한 뒤 같은 통계를 외부 데이터에도 적용(데이터 누수 방지).
    xg, xs, names = z_train_apply(region_descriptor_matrix(g["norm"]), region_descriptor_matrix(s["norm"]))

    # ---------------------------------------------------------------
    # 1단계: branch 스크리닝
    # 48개 특징 각각에 대해 부호(sign: 값이 클수록/작을수록 de-escalation 방향인지) x width(3종) x lambda(4종)
    # 조합(특징당 2*3*4=24가지, 총 48*24=1152가지)마다 단일-특징 de-escalation 게이트를 만들어 내부/외부 성능을 계산.
    # ---------------------------------------------------------------
    write_progress(stage="branch_screen")
    branch_rows = []
    # 이후 4-branch 조합 탐색 단계에서 동일한 투표 결과를 재계산하지 않도록, label -> 투표(불리언) 배열을 캐시에 저장해둠.
    branch_vote_cache_g: dict[str, np.ndarray] = {}
    branch_vote_cache_s: dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        region = name.split("__")[0]  # 특징 이름은 "R1_045_056__level_mean" 형태이므로 앞부분이 지역명.
        for sign in [-1, 1]:
            for width in WIDTHS:
                for lam in LAMBDAS:
                    # 내부/외부 각각에서, 이 특징 하나로 3개 운영점(S80/S85/S90)에 대한 de-escalation 투표(환자 x 운영점 불리언)를 계산.
                    vg = branch_votes(xg[:, j], sign, width, lam, c_g, thresholds)
                    vs = branch_votes(xs[:, j], sign, width, lam, c_s, thresholds)
                    label = f"{name}__sign{sign:+d}__w{width:.2f}__lam{lam:.2f}"
                    branch_vote_cache_g[label] = vg
                    branch_vote_cache_s[label] = vs
                    # 단일 branch만 있는 상황을 4비트 코드 체계에 맞추기 위해, 지역 차원(axis=-1)이 1개뿐인 투표를 코드로 변환하고
                    # 마스크 1<<0(코드 1, 즉 유일한 지역이 True인 패턴)만 선택해 evaluate_deesc를 호출 -> 이 특징 단독 성능 확인.
                    dg = evaluate_deesc("g1090_internal", g, cpos_g, votes_to_code(vg[:, :, None]), 1 << 0, label, name)
                    ds = evaluate_deesc("sdata_external", s, cpos_s, votes_to_code(vs[:, :, None]), 1 << 0, label, name)
                    detail = pd.concat([dg, ds], ignore_index=True)
                    # 내부/외부 통과 여부와 선택 점수를 담은 요약 행을 만들어 branch_rows에 누적.
                    branch_rows.append(branch_summary_row(region, name, sign, width, lam, detail))
        # 진행 상황을 8개 특징마다(혹은 마지막에) progress.json과 콘솔에 기록해 장시간 실행을 모니터링할 수 있게 함.
        if (j + 1) % 8 == 0 or j + 1 == len(names):
            write_progress(stage="branch_screen", descriptor_index=j + 1, total_descriptors=len(names), branch_candidates=len(branch_rows))
            print(f"branch screen {j + 1}/{len(names)}", flush=True)

    # 모든 branch 후보를 지역별 -> 내부 통과 여부 -> 내부 선택 점수 순으로 정렬해 전체 스크리닝 결과를 CSV로 저장(감사/디버깅용).
    branch_df = pd.DataFrame(branch_rows).sort_values(["region", "internal_pass", "internal_selection_score"], ascending=[True, False, False])
    branch_df.to_csv(OUT_DIR / "branch_candidate_screen.csv", index=False)
    # 지역별로 (내부 통과 상위 TOP_INTERNAL_PER_REGION개) + (내부+외부 모두 통과하는 안전한 상위 TOP_EXTERNAL_SAFE_PER_REGION개)를 뽑아
    # 다음 단계에서 조합할 후보 목록을 확정.
    selected = select_branch_candidates(branch_df, names)
    selected_df = pd.DataFrame([c.__dict__ for c in selected])
    selected_df.to_csv(OUT_DIR / "selected_branch_candidates.csv", index=False)
    # 선택된 후보를 지역별로 다시 묶어, 4개 지역 각각에서 몇 개의 후보를 조합에 사용할지 정리.
    by_region: dict[str, list[BranchCandidate]] = {region: [] for region in REGIONS}
    for cand in selected:
        by_region[cand.region].append(cand)
    # 4개 지역의 후보 개수를 모두 곱하면, 이후 탐색할 "4-branch 조합"의 총 개수가 나옴.
    n_combos = int(np.prod([len(by_region[r]) for r in REGIONS]))

    # ---------------------------------------------------------------
    # 2단계: 4-branch 조합 탐색
    # 4개 지역에서 각각 branch 후보 하나씩을 뽑아 만든 모든 조합(product)에 대해,
    # 그 조합의 4비트 투표 코드를 만들고, candidate_masks()가 제안하는 여러 패턴 마스크(어떤 4비트 코드 조합을 de-escalation 대상으로 볼지)
    # 각각에 대해 성능을 평가해 최종 규칙 후보를 만든다.
    # ---------------------------------------------------------------
    rows = []
    # internal_pass인 규칙만 상세 detail(운영점별 성능표)을 보관해두었다가, 이후 최종 우승 규칙의 그래프/CSV를 만들 때 재사용.
    details_by_rule: dict[str, pd.DataFrame] = {}
    start_time = time.time()
    done = 0
    write_progress(stage="combo_search", n_branch_combos=n_combos, selected_per_region={k: len(v) for k, v in by_region.items()})
    for combo in itertools.product(*(by_region[r] for r in REGIONS)):
        # combo는 (R1 후보, R2 후보, R3 후보, R4 후보) 튜플. 각 후보의 캐시된 투표 배열을 지역 축으로 쌓아 (환자 x 운영점 x 4) 배열을 구성.
        votes_g = np.stack([branch_vote_cache_g[c.label] for c in combo], axis=-1)
        votes_s = np.stack([branch_vote_cache_s[c.label] for c in combo], axis=-1)
        # 4개 지역의 True/False 투표를 4비트 정수 코드(0~15)로 압축.
        code_g = votes_to_code(votes_g)
        code_s = votes_to_code(votes_s)
        # 내부 데이터의 코드 분포/이벤트율을 바탕으로 시도해볼 만한 패턴 마스크 후보들(단일 패턴, popcount 조건, 상위 패턴 조합 등)을 생성.
        masks = candidate_masks(code_g, g["y"].astype(int), cpos_g)
        branch_labels = [c.label for c in combo]
        base_rule = "__".join([f"{c.region}_{c.descriptor}_s{c.sign:+d}_w{c.width:.2f}_l{c.lam:.2f}" for c in combo])
        for mask in masks:
            # 규칙마다 고유 이름을 부여하고, 어떤 특징/패턴 조합인지 사람이 읽을 수 있는 문자열도 함께 기록.
            rule = f"new4_combo_{len(rows):06d}"
            features = f"{base_rule}; patterns={pattern_text(mask)}"
            # 이 (조합, 마스크) 규칙을 내부/외부 데이터 각각에 대해 평가해 운영점별 성능표(detail)를 만듦.
            detail = pd.concat(
                [
                    evaluate_deesc("g1090_internal", g, cpos_g, code_g, mask, rule, features),
                    evaluate_deesc("sdata_external", s, cpos_s, code_s, mask, rule, features),
                ],
                ignore_index=True,
            )
            row = combo_summary_row(rule, branch_labels, mask, detail)
            rows.append(row)
            if row["internal_pass"]:
                # 내부 통과 규칙만 상세 detail을 보관(전체를 보관하면 메모리 부담이 크므로 필요한 것만 캐시).
                details_by_rule[rule] = detail
        done += 1
        # 100개 조합마다(또는 처음/마지막) 진행률, 통과 개수, 예상 잔여 시간(ETA)을 progress.json과 콘솔에 출력.
        if done == 1 or done % 100 == 0 or done == n_combos:
            elapsed = time.time() - start_time
            eta = (n_combos - done) * elapsed / done if done else None
            internal_pass = sum(1 for r in rows if r["internal_pass"])
            both_pass = sum(1 for r in rows if r["internal_pass"] and r["external_pass"])
            write_progress(
                stage="combo_search",
                branch_combos_done=done,
                n_branch_combos=n_combos,
                candidates=len(rows),
                internal_passing=internal_pass,
                internal_external_passing=both_pass,
                eta_seconds=eta,
            )
            print(
                f"combo {done}/{n_combos}; candidates={len(rows)}; internal_pass={internal_pass}; both={both_pass}; ETA_min={eta / 60 if eta else np.nan:.1f}",
                flush=True,
            )

    # ---------------------------------------------------------------
    # 3단계: 결과 집계 및 저장
    # ---------------------------------------------------------------
    # 전체 후보를 (내부 통과 여부, 내부 선택 점수) 기준으로 정렬해 전수 결과를 CSV로 저장.
    summary = pd.DataFrame(rows).sort_values(["internal_pass", "internal_selection_score"], ascending=[False, False])
    summary.to_csv(OUT_DIR / "new4_combo_all_candidates.csv", index=False)
    # 내부 통과 규칙만 따로 뽑아 저장("internal_locked" 최종 후보군).
    internal = summary[summary["internal_pass"]].copy()
    internal.to_csv(OUT_DIR / "new4_combo_internal_passing_ranked.csv", index=False)
    # 내부 통과 규칙 중 외부까지 통과하는 규칙만 뽑아, 외부 정확도/정확도증가/내부정확도 순으로 재정렬(감사용, "internal_external_audit").
    both = internal[internal["external_pass"]].sort_values(
        ["external_mean_accuracy", "external_mean_accuracy_gain", "internal_mean_accuracy"], ascending=False
    )
    both.to_csv(OUT_DIR / "new4_combo_internal_external_passing_ranked.csv", index=False)

    # 두 관점(내부만 통과한 것 중 최고 / 내부+외부 모두 통과한 것 중 최고)의 "우승 규칙"을 각각 하나씩 뽑음.
    winners = {
        "internal_locked": internal.iloc[0].to_dict() if not internal.empty else None,
        "internal_external_audit": both.iloc[0].to_dict() if not both.empty else None,
    }
    # 각 우승 규칙에 대해, 캐시해둔 상세 성능표를 CSV로 저장하고 운영점별 성능 그래프를 그려 이미지로 저장.
    for tag, winner in winners.items():
        if winner is None:
            continue
        detail = details_by_rule[str(winner["rule"])]
        detail.to_csv(OUT_DIR / f"{tag}_winner_details.csv", index=False)
        plot_winner(detail, OUT_DIR / f"{tag}_winner_plot.png")

    # 이번 실행의 전체 프로토콜 설명, 지역 정의, 후보/통과 개수, 우승 규칙 요약을 하나의 JSON 파일로 남김(재현/보고용).
    with (OUT_DIR / "new4_combo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "regions": REGIONS,
                "protocol": "Screen branch descriptors inside four region-search-selected windows, choose top internal branch candidates per region, search one fixed 4-branch pattern gate over S80/S85/S90. External-passing winner is audit-only.",
                "target_ops": TARGET_OPS,
                "n_selected_branch_combos": n_combos,
                "n_candidates": int(len(summary)),
                "n_internal_passing": int(len(internal)),
                "n_internal_external_passing": int(len(both)),
                "winners": winners,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    # 전체 파이프라인이 끝났음을 progress.json에 "complete" 단계로 기록.
    write_progress(
        stage="complete",
        n_candidates=int(len(summary)),
        n_internal_passing=int(len(internal)),
        n_internal_external_passing=int(len(both)),
        output_dir=str(OUT_DIR),
    )
    # 콘솔에 핵심 지표 컬럼만 골라, 내부 통과 상위 10개와 내부+외부 통과 상위 10개를 각각 출력해 빠르게 결과를 확인할 수 있게 함.
    show_cols = [
        "internal_mean_accuracy",
        "internal_mean_accuracy_gain",
        "internal_min_accuracy_gain",
        "internal_mean_specificity_gain",
        "internal_max_sensitivity_loss",
        "internal_min_sensitivity_loss_p",
        "external_pass",
        "external_mean_accuracy",
        "external_mean_accuracy_gain",
        "external_min_accuracy_gain",
        "external_mean_specificity_gain",
        "external_max_sensitivity_loss",
        "external_min_sensitivity_loss_p",
        "patterns",
        "branches",
    ]
    print("\nTOP INTERNAL")
    print(internal[show_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nTOP INTERNAL+EXTERNAL")
    print(both[show_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# 4개 지역(region)의 대리(surrogate) 특징 후보를 스크리닝해 지역별 유망 branch를 고르고, 이들을 조합한 4-branch 패턴 게이트를
# 전수 탐색해 내부/외부 데이터에서 통과하는 de-escalation 규칙을 찾는 파이프라인을 실행.
if __name__ == "__main__":
    main()
