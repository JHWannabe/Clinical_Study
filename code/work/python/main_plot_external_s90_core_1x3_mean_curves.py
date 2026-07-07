from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, clinical_scores, load_dataset  # noqa: E402
from aec_new_region_surrogate_combo_gate import region_descriptor_matrix, z_train_apply  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "aec_1x3_core_mean_curves"
PNG = OUT_DIR / "external_s90_core_1x3_mean_curves.png"
PNG_TANGENT = OUT_DIR / "external_s90_core_1x3_mean_curves_with_r4_tangent.png"
PNG_2X3 = OUT_DIR / "external_s90_core_2x3_mean_and_mirror_deviation.png"
CSV = OUT_DIR / "external_s90_core_1x3_mean_curve_summary.csv"
MIRROR_CSV = OUT_DIR / "external_s90_core_2x3_mirror_deviation_summary.csv"
JSON = OUT_DIR / "external_s90_core_1x3_summary.json"

# gate_scores()에 전달되는 4개 영역(R1~R4) 게이트 브랜치 설정.
# - region: 영역 이름
# - feature: z_train_apply가 반환하는 형태 특징 이름 (region_descriptor_matrix의 컬럼)
# - sign: 특징 점수의 방향(+1/-1). gate_scores에서 sign * feature_z로 곱해져 "낮을수록 위험"
#         방향으로 맞춰줌
# - width: 임상 임계값 근처에서만 게이트를 활성화하는 가우시안 창의 폭(표준편차 역할)
# - lambda: 게이트가 최대로 열렸을 때 feature_z가 clinical_z에 더해지는 가중치
BRANCHES = [
    {
        "region": "R1",
        "feature": "R1_045_056__endpoint_delta",
        "sign": -1,
        "width": 0.50,
        "lambda": 0.25,
    },
    {
        "region": "R2",
        "feature": "R2_057_080__level_mean",
        "sign": -1,
        "width": 0.70,
        "lambda": 0.25,
    },
    {
        "region": "R3",
        "feature": "R3_097_128__linear_slope",
        "sign": 1,
        "width": 0.35,
        "lambda": 0.25,
    },
    {
        "region": "R4",
        "feature": "R4_117_128__endpoint_delta",
        "sign": -1,
        "width": 0.50,
        "lambda": 0.25,
    },
]

# 4개 브랜치 투표(+/-) 조합 중 "AEC 양성(하향조정 후보)"으로 인정할 패턴 집합.
# 문자열 순서는 BRANCHES 리스트 순서(R1,R2,R3,R4)와 대응하며, 각 자리가 "+"면 해당
# 영역이 vote=True(게이트 통과)임을 의미. pattern_from_votes()가 만든 패턴과 비교됨.
SELECTED_PATTERNS = {"++--", "--+-", "---+", "+--+", "++++"}

# 그래프에 그릴 R1~R4 영역 밴드 정의: (라벨, 시작 슬라이스, 끝 슬라이스, 색상, 음영 투명도).
# add_regions()에서 배경 밴드/라벨을 그리고, panel_summary()/mirror_summary()에서
# 영역별 통계를 낼 때 슬라이스 범위로도 재사용됨.
REGION_SPANS = [
    ("R1", 45, 56, "#4E79A7", 0.08),
    ("R2", 57, 80, "#F28E2B", 0.08),
    ("R3", 97, 128, "#59A14F", 0.08),
    ("R4", 117, 128, "#B07AA1", 0.14),
]


def mean_ci(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """위치별 평균과 95% 신뢰구간(정규근사)을 계산 (표본이 1개 이하면 신뢰구간은 NaN)."""
    mean = np.nanmean(x, axis=0)
    if len(x) <= 1:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(len(x))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def pattern_from_votes(votes: np.ndarray) -> np.ndarray:
    """4개 브랜치의 불리언 투표 행렬을 "++--" 같은 부호 문자열 패턴으로 변환."""
    return np.array(["".join("+" if v else "-" for v in row) for row in votes], dtype=object)


def gate_scores(
    feature_z: np.ndarray,
    sign: int,
    clinical_z: np.ndarray,
    threshold: float,
    width: float,
    lam: float,
) -> np.ndarray:
    """임상 임계값 근처에서만 가우시안 가중치로 특징 점수를 더하는 게이트 점수를 계산."""
    boundary = np.exp(-0.5 * ((clinical_z - threshold) / width) ** 2)
    return clinical_z + lam * boundary * (sign * feature_z)


def compute_external_s90_gate(g: dict, s: dict) -> dict:
    """S90 임상 임계값에서, 4개 영역(R1~R4) 형태 특징 브랜치의 게이트 투표 패턴을 계산해, 사전 선택된
    패턴(SELECTED_PATTERNS)에 해당하면 "AEC 양성(하향조정 후보)"로 표시하고, 임상 양성군을 AEC 양성/음성으로 나눔."""
    _, _, c_g, c_s, thresholds = clinical_scores(g, s)
    threshold = float(thresholds["S90"])

    fg = region_descriptor_matrix(g["norm"])
    fs = region_descriptor_matrix(s["norm"])
    xg, xs, names = z_train_apply(fg, fs)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    votes = []
    branch_rows = []
    for branch in BRANCHES:
        idx = name_to_idx[branch["feature"]]
        score = gate_scores(
            xs[:, idx],
            int(branch["sign"]),
            c_s,
            threshold,
            float(branch["width"]),
            float(branch["lambda"]),
        )
        vote = score < threshold
        votes.append(vote)
        branch_rows.append(
            {
                **branch,
                "external_vote_positive_n": int(vote.sum()),
            }
        )

    vote_matrix = np.column_stack(votes)
    pattern = pattern_from_votes(vote_matrix)
    morphology_pos = np.isin(pattern, list(SELECTED_PATTERNS))
    clinical_pos = c_s >= threshold
    deesc = clinical_pos & morphology_pos
    retained = clinical_pos & ~morphology_pos

    return {
        "clinical_z": c_s,
        "clinical_threshold": threshold,
        "clinical_pos": clinical_pos,
        "pattern": pattern,
        "morphology_pos": morphology_pos,
        "clinical_pos_aec_pos": deesc,
        "clinical_pos_aec_neg": retained,
        "branch_rows": branch_rows,
    }


def add_regions(ax: Axes) -> None:
    """그래프 위에 R1~R4 영역을 색칠된 세로 밴드와 라벨로 표시."""
    for label, start, end, color, alpha in REGION_SPANS:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
    y0, y1 = ax.get_ylim()
    label_y = y1 - 0.015 * (y1 - y0)
    for label, start, end, color, _ in REGION_SPANS:
        ax.text(
            (start + end) / 2,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
        )


def style_axis(ax: Axes) -> None:
    """그래프 축의 공통 스타일(기준선, x축 범위·눈금, 격자, 테두리 제거)을 적용."""
    ax.axhline(1.0, color="#9E9E9E", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(1, 128)
    ax.set_xticks([1, 32, 64, 96, 128])
    ax.set_xlabel("Slice index")
    ax.grid(axis="both", color="#D0D0D0", lw=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_group(ax: Axes, z: np.ndarray, x: np.ndarray, mask: np.ndarray, label: str, color: str) -> None:
    """mask로 선택된 그룹의 평균 곡선과 95% 신뢰구간 음영을 그래프에 그림."""
    mean, lo, hi = mean_ci(x[mask])
    ax.plot(z, mean, color=color, lw=2.4, label=f"{label} (n={int(mask.sum())})")
    ax.fill_between(z, lo, hi, color=color, alpha=0.12, lw=0)


def add_r4_tangent_annotation(
    ax: Axes,
    z: np.ndarray,
    norm: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
) -> None:
    """R4 구간(117~128)에서 AEC+/AEC- 두 그룹 평균 곡선에 각각 직선을 적합(fit)해 기울기를 그래프에 굵은 선+텍스트로 표시."""
    r4 = (z >= 117) & (z <= 128)
    x = z[r4].astype(float)
    x_center = x.mean()
    x_span = np.array([117.0, 128.0])
    red_mean = np.nanmean(norm[red_mask], axis=0)[r4]
    blue_mean = np.nanmean(norm[blue_mask], axis=0)[r4]

    red_slope, red_intercept = np.polyfit(x, red_mean, 1)
    blue_slope, blue_intercept = np.polyfit(x, blue_mean, 1)
    red_y = red_slope * x_span + red_intercept
    blue_y = blue_slope * x_span + blue_intercept

    ax.plot(x_span, red_y, color="#D04F5B", lw=4.0, alpha=0.85, solid_capstyle="round")
    ax.plot(x_span, blue_y, color="#2F6F9F", lw=4.0, alpha=0.85, solid_capstyle="round")
    ax.text(
        0.98,
        0.98,
        f"R4 fitted slope\nAEC- {red_slope:+.4f}/slice\nAEC+ {blue_slope:+.4f}/slice",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.86, "pad": 4},
    )


def plot_mirror_deviation(
    ax: Axes,
    z: np.ndarray,
    norm: np.ndarray,
    ref_mean: np.ndarray,
    red_mask: np.ndarray,
    blue_mask: np.ndarray,
    red_label: str,
    blue_label: str,
    title: str,
) -> tuple[np.ndarray, np.ndarray]:
    """기준곡선(ref_mean) 대비 두 그룹 평균의 절대편차를 위아래(빨강 위/파랑 아래)로 미러링해 그리고, 각 그룹의 편차 배열을 반환."""
    red_dev = np.abs(np.nanmean(norm[red_mask], axis=0) - ref_mean)
    blue_dev = np.abs(np.nanmean(norm[blue_mask], axis=0) - ref_mean)
    ax.fill_between(z, 0, red_dev, color="#D04F5B", alpha=0.30, lw=0)
    ax.plot(z, red_dev, color="#D04F5B", lw=1.8, label=red_label)
    ax.fill_between(z, 0, -blue_dev, color="#2F6F9F", alpha=0.30, lw=0)
    ax.plot(z, -blue_dev, color="#2F6F9F", lw=1.8, label=blue_label)
    ax.axhline(0, color="#666666", lw=0.9)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    return red_dev, blue_dev


def panel_summary(norm: np.ndarray, name: str, mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """두 그룹의 평균 곡선 차이(전체 및 R1~R4 영역별)를 요약한 한 행을 만듦."""
    mean_a = np.nanmean(norm[mask_a], axis=0)
    mean_b = np.nanmean(norm[mask_b], axis=0)
    row = {
        "comparison": name,
        "n_group1": int(mask_a.sum()),
        "n_group2": int(mask_b.sum()),
        "mean_abs_between_group_difference": float(np.nanmean(np.abs(mean_a - mean_b))),
    }
    for label, start, end, _, _ in REGION_SPANS:
        sl = slice(start - 1, end)
        row[f"{label}_group1_minus_group2"] = float(np.nanmean(mean_a[sl] - mean_b[sl]))
    return row


def mirror_summary(name: str, red_label: str, blue_label: str, red_dev: np.ndarray, blue_dev: np.ndarray) -> list[dict]:
    """두 그룹의 편차 곡선(red_dev/blue_dev)에서 전체 및 R1~R4 영역별 평균·최댓값 편차를 표로 정리."""
    rows = []
    for group, dev in [(red_label, red_dev), (blue_label, blue_dev)]:
        row = {
            "panel": name,
            "group": group,
            "mean_abs_deviation_all_slices": float(np.nanmean(dev)),
            "max_abs_deviation_all_slices": float(np.nanmax(dev)),
            "max_abs_deviation_slice": int(np.nanargmax(dev) + 1),
        }
        for label, start, end, _, _ in REGION_SPANS:
            sl = slice(start - 1, end)
            row[f"{label}_mean_abs_deviation"] = float(np.nanmean(dev[sl]))
            row[f"{label}_max_abs_deviation"] = float(np.nanmax(dev[sl]))
        rows.append(row)
    return rows


def fisher_exact_conditional(y: np.ndarray, aec_pos: np.ndarray, aec_neg: np.ndarray) -> float:
    """AEC 양성군과 음성군의 사건 발생률 차이에 대한 Fisher 정확검정 p값을 계산."""
    table = [
        [int(y[aec_pos].sum()), int((~y.astype(bool))[aec_pos].sum())],
        [int(y[aec_neg].sum()), int((~y.astype(bool))[aec_neg].sum())],
    ]
    return float(cast(float, stats.fisher_exact(table)[1]))


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (7/2 최종 산출물 — 질문: 여러 날짜에 걸쳐 찾아낸 4개 영역(R1~R4)
    형태 게이트를 외부(sdata) 데이터 하나에 적용했을 때, 최종 그림으로 봤을 때도 이야기가 말이
    되는가? — 논문/보고용 최종 시각화):

    1. g1090/sdata를 로드하고, compute_external_s90_gate로 S90 임상 임계값과 4개 영역 게이트
       투표 패턴을 계산해 외부 데이터의 "임상 양성 중 AEC 양성(하향조정 후보)/AEC 음성(유지)"을 나눈다.
    2. (A) 임상 양성/음성, (B) 실제 저근감소증/비저근감소증, (C) 임상양성 내 AEC+/AEC- 세 가지 대비를
       나란히 놓은 1x3 평균곡선 그래프를 그려 PNG로 저장 (패널 C에는 Fisher 정확검정 p값과 사건율도 함께 표시).
    3. 같은 1x3 그래프에 R4 구간(117~128)의 적합 직선(기울기)을 추가로 표시한 버전도 저장.
    4. 위 3가지 대비 각각에 대해, 위쪽엔 평균곡선/아래쪽엔 기준곡선 대비 절대편차를 미러링해서 보여주는
       2x3 그래프를 그려 저장하고, 편차 요약을 CSV로 저장.
    5. 세 가지 대비의 전체/영역별 평균 차이를 요약표로 CSV 저장.
    6. 게이트 정의(선택된 패턴, 브랜치 조건), 임상양성 내 AEC 조건부 사건율·Fisher p값, 요약표를
       모두 JSON으로 저장하고, 생성된 파일 경로들을 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    gate = compute_external_s90_gate(g, s)

    z = np.arange(1, 129)
    norm = s["norm"]
    y = s["y"].astype(bool)
    clinical_pos = gate["clinical_pos"]
    clinical_neg = ~clinical_pos
    low = y
    nonlow = ~y
    cp_aec_pos = gate["clinical_pos_aec_pos"]
    cp_aec_neg = gate["clinical_pos_aec_neg"]

    overall_mean = np.nanmean(norm, axis=0)
    clinical_pos_mean = np.nanmean(norm[clinical_pos], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.7), sharey=True)

    axes[0].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group(axes[0], z, norm, clinical_pos, "Clinical +", "#D04F5B")
    plot_group(axes[0], z, norm, clinical_neg, "Clinical -", "#2F6F9F")
    axes[0].set_title("A. Clinical S90 operating point", loc="left", fontsize=12, fontweight="bold")

    axes[1].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group(axes[1], z, norm, low, "Low SMI +", "#D04F5B")
    plot_group(axes[1], z, norm, nonlow, "Non-low SMI", "#2F6F9F")
    axes[1].set_title("B. Outcome phenotype", loc="left", fontsize=12, fontweight="bold")

    axes[2].plot(z, clinical_pos_mean, color="#666666", lw=1.6, ls=":", label="Clinical + mean reference")
    plot_group(axes[2], z, norm, cp_aec_pos, "Clinical+ / AEC+", "#2F6F9F")
    plot_group(axes[2], z, norm, cp_aec_neg, "Clinical+ / AEC-", "#D04F5B")
    axes[2].set_title("C. Conditional AEC split among Clinical +", loc="left", fontsize=12, fontweight="bold")

    cp_pos_events = int(y[cp_aec_pos].sum())
    cp_neg_events = int(y[cp_aec_neg].sum())
    p_fisher = fisher_exact_conditional(y, cp_aec_pos, cp_aec_neg)
    pos_rate = cp_pos_events / int(cp_aec_pos.sum())
    neg_rate = cp_neg_events / int(cp_aec_neg.sum())
    axes[2].text(
        0.98,
        0.98,
        f"low SMI: {cp_pos_events}/{int(cp_aec_pos.sum())}={pos_rate:.1%} vs "
        f"{cp_neg_events}/{int(cp_aec_neg.sum())}={neg_rate:.1%}\nFisher p={p_fisher:.2g}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    for ax in axes:
        style_axis(ax)
        ax.set_ylim(0.82, 1.21)
        add_regions(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)

    axes[0].set_ylabel("Patient-normalized AEC")
    fig.suptitle(
        "External S90 core AEC morphology comparisons\n"
        "AEC+ indicates de-escalation / lower low-SMI probability",
        fontsize=15,
        fontweight="bold",
        y=1.04,
    )
    fig.text(0.5, -0.015, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig_t, axes_t = plt.subplots(1, 3, figsize=(18.5, 4.7), sharey=True)

    axes_t[0].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group(axes_t[0], z, norm, clinical_pos, "Clinical +", "#D04F5B")
    plot_group(axes_t[0], z, norm, clinical_neg, "Clinical -", "#2F6F9F")
    axes_t[0].set_title("A. Clinical S90 operating point", loc="left", fontsize=12, fontweight="bold")

    axes_t[1].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group(axes_t[1], z, norm, low, "Low SMI +", "#D04F5B")
    plot_group(axes_t[1], z, norm, nonlow, "Non-low SMI", "#2F6F9F")
    axes_t[1].set_title("B. Outcome phenotype", loc="left", fontsize=12, fontweight="bold")

    axes_t[2].plot(z, clinical_pos_mean, color="#666666", lw=1.6, ls=":", label="Clinical + mean reference")
    plot_group(axes_t[2], z, norm, cp_aec_pos, "Clinical+ / AEC+", "#2F6F9F")
    plot_group(axes_t[2], z, norm, cp_aec_neg, "Clinical+ / AEC-", "#D04F5B")
    add_r4_tangent_annotation(axes_t[2], z, norm, cp_aec_neg, cp_aec_pos)
    axes_t[2].set_title("C. Conditional AEC split with R4 fitted tangent", loc="left", fontsize=12, fontweight="bold")

    for ax in axes_t:
        style_axis(ax)
        ax.set_ylim(0.82, 1.21)
        add_regions(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)

    axes_t[0].set_ylabel("Patient-normalized AEC")
    fig_t.suptitle(
        "External S90 core AEC morphology comparisons with R4 fitted tangent",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig_t.text(0.5, -0.015, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig_t.tight_layout()
    fig_t.savefig(PNG_TANGENT, dpi=260, bbox_inches="tight")
    plt.close(fig_t)

    fig2, axes2 = plt.subplots(2, 3, figsize=(18.5, 8.8), sharex=True)
    panels = [
        {
            "top_title": "A. Clinical S90 operating point",
            "bottom_title": "D. Deviation from overall mean",
            "reference": overall_mean,
            "reference_label": "external overall mean",
            "red_mask": clinical_pos,
            "red_label": f"Clinical + (n={int(clinical_pos.sum())})",
            "blue_mask": clinical_neg,
            "blue_label": f"Clinical - (n={int(clinical_neg.sum())})",
            "panel_name": "Clinical + vs Clinical -",
        },
        {
            "top_title": "B. Outcome phenotype",
            "bottom_title": "E. Deviation from overall mean",
            "reference": overall_mean,
            "reference_label": "external overall mean",
            "red_mask": low,
            "red_label": f"Low SMI + (n={int(low.sum())})",
            "blue_mask": nonlow,
            "blue_label": f"Non-low SMI (n={int(nonlow.sum())})",
            "panel_name": "Low SMI + vs Non-low SMI",
        },
        {
            "top_title": "C. Conditional AEC split among Clinical +",
            "bottom_title": "F. Deviation from Clinical + mean",
            "reference": clinical_pos_mean,
            "reference_label": "Clinical + mean reference",
            "red_mask": cp_aec_neg,
            "red_label": f"Clinical+ / AEC- (n={int(cp_aec_neg.sum())})",
            "blue_mask": cp_aec_pos,
            "blue_label": f"Clinical+ / AEC+ (n={int(cp_aec_pos.sum())})",
            "panel_name": "Clinical+/AEC- vs Clinical+/AEC+",
        },
    ]

    mirror_rows = []
    for j, panel in enumerate(panels):
        ax_top = axes2[0, j]
        ax_bottom = axes2[1, j]
        ax_top.plot(z, panel["reference"], color="#666666", lw=1.6, ls=":", label=panel["reference_label"])
        plot_group(ax_top, z, norm, panel["red_mask"], panel["red_label"].split(" (n=")[0], "#D04F5B")
        plot_group(ax_top, z, norm, panel["blue_mask"], panel["blue_label"].split(" (n=")[0], "#2F6F9F")
        ax_top.set_title(panel["top_title"], loc="left", fontsize=12, fontweight="bold")
        style_axis(ax_top)
        ax_top.set_ylim(0.82, 1.21)
        add_regions(ax_top)
        ax_top.legend(frameon=False, loc="lower left", fontsize=8.8)

        red_dev, blue_dev = plot_mirror_deviation(
            ax_bottom,
            z,
            norm,
            panel["reference"],
            panel["red_mask"],
            panel["blue_mask"],
            panel["red_label"],
            panel["blue_label"],
            panel["bottom_title"],
        )
        style_axis(ax_bottom)
        max_dev = max(float(np.nanmax(red_dev)), float(np.nanmax(blue_dev)), 0.04)
        ax_bottom.set_ylim(-max_dev * 1.18, max_dev * 1.18)
        add_regions(ax_bottom)
        ax_bottom.legend(frameon=False, loc="upper left", fontsize=8.5)
        mirror_rows.extend(mirror_summary(panel["panel_name"], panel["red_label"], panel["blue_label"], red_dev, blue_dev))

    axes2[0, 0].set_ylabel("Patient-normalized AEC")
    axes2[1, 0].set_ylabel("|group mean - reference|\n(red upward, blue downward)")
    fig2.suptitle(
        "External S90 AEC morphology: mean curves and mirror absolute-deviation plots\n"
        "Bottom row shows magnitude of separation from the reference curve; placement above/below zero denotes group color, not original direction",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig2.text(0.5, -0.005, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(PNG_2X3, dpi=260, bbox_inches="tight")
    plt.close(fig2)
    pd.DataFrame(mirror_rows).to_csv(MIRROR_CSV, index=False)

    summary_rows = [
        panel_summary(norm, "Clinical + vs Clinical -", clinical_pos, clinical_neg),
        panel_summary(norm, "Low SMI + vs Non-low SMI", low, nonlow),
        panel_summary(norm, "Clinical+/AEC- vs Clinical+/AEC+", cp_aec_neg, cp_aec_pos),
    ]
    pd.DataFrame(summary_rows).to_csv(CSV, index=False)
    JSON.write_text(
        json.dumps(
            {
                "png": str(PNG),
                "png_tangent": str(PNG_TANGENT),
                "png_2x3": str(PNG_2X3),
                "external_dataset": "sdata",
                "clinical_operating_point": "S90",
                "AEC_definition": "primary interpretable morphology gate new4_combo_261089; AEC+ means de-escalation/low-risk morphology",
                "selected_patterns": sorted(SELECTED_PATTERNS),
                "branches": gate["branch_rows"],
                "low_smi_conditional": {
                    "clinical_pos_aec_pos_events": cp_pos_events,
                    "clinical_pos_aec_pos_n": int(cp_aec_pos.sum()),
                    "clinical_pos_aec_pos_rate": pos_rate,
                    "clinical_pos_aec_neg_events": cp_neg_events,
                    "clinical_pos_aec_neg_n": int(cp_aec_neg.sum()),
                    "clinical_pos_aec_neg_rate": neg_rate,
                    "fisher_p": p_fisher,
                },
                "summary": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(PNG)
    print(PNG_TANGENT)
    print(PNG_2X3)
    print(CSV)
    print(MIRROR_CSV)
    print(JSON)


if __name__ == "__main__":
    main()
