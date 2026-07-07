from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    risk_direction,
    standardize_train_test,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_clinical_positive_aec_gate_curves"
RANKED_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_midrange_feature_refit" / "midrange_feature_search_train_ranked.csv"

FEATURE_SHORTS = [
    "visual_trough_depth__early_041_056__mid_053_076__tail_101_128",
    "norm_slope_085_096_mean",
    "norm_haar_b08_045_060",
    "norm_haar_b08_103_118",
]
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


def choose_feature_settings() -> pd.DataFrame:
    """aec_midrange_feature_refit의 순위표에서 미리 지정한 4개 "중후반 구간" 특징(FEATURE_SHORTS)의 최적 폭·람다 설정을 가져온다."""
    ranked = pd.read_csv(RANKED_PATH)
    ranked["feature_short"] = (
        ranked["feature"].astype(str).str.replace("bank_norm__", "", regex=False).str.replace("midrange__", "", regex=False)
    )
    rows = []
    for short in FEATURE_SHORTS:
        rows.append(ranked[ranked["feature_short"].eq(short)].iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def fisher_p(y: np.ndarray, kept: np.ndarray, deesc: np.ndarray) -> float:
    """유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[kept] == 1))
    b = int(np.sum(y[kept] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    return float(stats.fisher_exact([[a, b], [c, d]])[1]) if (a + b and c + d) else np.nan


def mean_ci(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """행렬의 위치별 평균과 95% 신뢰구간(하한/상한)을 계산."""
    m = np.nanmean(x, axis=0)
    if len(x) <= 1:
        return m, np.full_like(m, np.nan), np.full_like(m, np.nan)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


def add_anatomy_spans(ax: plt.Axes, label: bool = False) -> None:
    """그래프에 4개 관심 구간(하복부/중복부 골/상복부 회복/횡격막하 꼬리)을 색칠한 배경 영역으로 표시하고, 필요시 구간 이름 라벨을 추가."""
    spans = [
        (41, 56, "#D6EFD8", "lower abdomen"),
        (53, 76, "#F8D6D2", "mid-abdominal trough"),
        (85, 96, "#D6E4F5", "upper-abdominal recovery"),
        (103, 118, "#E8DCF5", "subdiaphragmatic tail"),
    ]
    for a, b, color, text in spans:
        ax.axvspan(a, b, color=color, alpha=0.26, lw=0)
        if label:
            ymax = ax.get_ylim()[1]
            ax.text((a + b) / 2, ymax, text, ha="center", va="top", fontsize=7, color="#333333", rotation=90)


def plot_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, kept: np.ndarray, deesc: np.ndarray, title: str, show_ci: bool) -> None:
    """한 서브플롯에 유지군(AEC+)과 하향조정군(AEC-)의 평균 곡선(옵션에 따라 95% CI 포함)을 겹쳐 그리고, 제목에 두 군의 표본수·사건수·Fisher p값을 표시."""
    z = np.arange(1, 129)
    m_keep, lo_keep, hi_keep = mean_ci(x[kept])
    m_deesc, lo_deesc, hi_deesc = mean_ci(x[deesc])

    ax.plot(z, m_keep, color="#B23A48", lw=2.2, label="AEC(+): kept")
    ax.plot(z, m_deesc, color="#2F6F9F", lw=2.2, label="AEC(-): de-escalated")
    if show_ci:
        ax.fill_between(z, lo_keep, hi_keep, color="#B23A48", alpha=0.12, lw=0)
        ax.fill_between(z, lo_deesc, hi_deesc, color="#2F6F9F", alpha=0.12, lw=0)

    p = fisher_p(y, kept, deesc)
    deesc_events = int(y[deesc].sum())
    kept_events = int(y[kept].sum())
    ax.set_title(
        f"{title}\nAEC(-) {deesc_events}/{int(deesc.sum())}; AEC(+) {kept_events}/{int(kept.sum())}; Fisher p={p:.2g}",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlim(1, 128)
    ax.set_xticks([1, 32, 64, 96, 128])
    ax.grid(axis="y", color="#dddddd", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_block_or_consensus_search 등에서 사용한 "중후반 구간"
    4개 특징의 2-of-4 합의 하향조정 게이트가, 실제로 정규화/원본 AEC 곡선 자체에서 유지군과
    하향조정군을 시각적으로 뚜렷하게 갈라놓는지 확인하려는 목적):

    1. g1090(Gangnam)/sdata(Sinchon)를 로드하고 choose_feature_settings로 4개 특징의 폭·람다
       설정을 가져온 뒤, 임상점수·특징뱅크·표준화값·방향을 계산하고 g1090 기준 5개 운영점
       (S80~S90)의 임상 임계값을 구한다.
    2. 두 코호트 x 5개 운영점 전체에 대해, 4개 특징 중 2개 이상이 하향조정 신호를 내는 임상양성
       표본을 "하향조정(deesc)"으로, 나머지를 "유지(kept)"로 분류하고 그 표본수·사건수·Fisher
       p값을 표로 저장.
    3. S85 운영점에서 두 코호트의 정규화 AEC 평균곡선(+95%CI)을 유지군 대 하향조정군으로 겹쳐
       그리고, 해부학적 관심구간을 배경 음영으로 표시한 그림을 저장.
    4. 두 코호트 x 5개 운영점 전체(2x5 격자)에 대해 같은 비교를 반복한 종합 그림을 저장.
    5. S85 운영점에서 원본(비정규화) AEC 곡선으로도 같은 비교 그림을 저장.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    settings = choose_feature_settings()

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    cg, cs, _ = clinical_scores(g, s)
    direction = risk_direction(g["y"].astype(int), cg, xg)
    name_to_idx = {name: i for i, name in enumerate(names)}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], cg, target) for op, target in OPS}

    datasets = {
        "Gangnam internal": {"y": g["y"].astype(int), "clinical": cg, "features": xg, "norm": g["norm"], "raw": g["raw"]},
        "Sinchon external": {"y": s["y"].astype(int), "clinical": cs, "features": xs, "norm": s["norm"], "raw": s["raw"]},
    }

    memberships: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    membership_rows = []
    for cohort, d in datasets.items():
        for op, _ in OPS:
            th = thresholds[op]
            cpos = d["clinical"] >= th
            votes = np.zeros(len(d["y"]), dtype=int)
            for _, r in settings.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((d["clinical"] - th) / float(r["width"])) ** 2)
                gate = d["clinical"] + float(r["lambda"]) * boundary * d["features"][:, idx] * direction[idx]
                votes += (cpos & (gate < th)).astype(int)
            deesc = cpos & (votes >= 2)
            kept = cpos & ~deesc
            memberships[(cohort, op)] = (kept, deesc)
            membership_rows.append(
                {
                    "cohort": cohort,
                    "operating_point": op,
                    "clinical_positive_n": int(cpos.sum()),
                    "aec_positive_kept_n": int(kept.sum()),
                    "aec_positive_kept_events": int(d["y"][kept].sum()),
                    "aec_negative_deescalated_n": int(deesc.sum()),
                    "aec_negative_deescalated_events": int(d["y"][deesc].sum()),
                    "fisher_p": fisher_p(d["y"], kept, deesc),
                }
            )
    pd.DataFrame(membership_rows).to_csv(OUT_DIR / "clinical_positive_aec_gate_membership_counts.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, cohort in zip(axes, datasets):
        kept, deesc = memberships[(cohort, "S85")]
        plot_panel(ax, datasets[cohort]["norm"], datasets[cohort]["y"], kept, deesc, f"{cohort}: clinical-positive at S85", True)
        add_anatomy_spans(ax, label=True)
        ax.set_xlabel("Craniocaudal index: 1 inferior pubic margin -> 128 liver dome")
    axes[0].set_ylabel("Patient-normalized AEC")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Clinical-Positive Patients Split by AEC Gate: Mean Normalized AEC Curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "clinical_positive_aec_pos_neg_mean_curves_S85_normalized.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(18.5, 7.8), sharex=True, sharey=True)
    for row, cohort in enumerate(datasets):
        for col, (op, _) in enumerate(OPS):
            ax = axes[row, col]
            kept, deesc = memberships[(cohort, op)]
            plot_panel(ax, datasets[cohort]["norm"], datasets[cohort]["y"], kept, deesc, f"{cohort} {op}", False)
            add_anatomy_spans(ax, label=False)
            if row == 1:
                ax.set_xlabel("1 pubis -> 128 liver")
            if col == 0:
                ax.set_ylabel("Normalized AEC")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("Clinical-Positive AEC(+) Kept vs AEC(-) De-escalated: All Sensitivity Operating Points", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "clinical_positive_aec_pos_neg_mean_curves_all_ops_normalized.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
    for ax, cohort in zip(axes, datasets):
        kept, deesc = memberships[(cohort, "S85")]
        plot_panel(ax, datasets[cohort]["raw"], datasets[cohort]["y"], kept, deesc, f"{cohort}: clinical-positive at S85", True)
        add_anatomy_spans(ax, label=True)
        ax.set_xlabel("Craniocaudal index: 1 inferior pubic margin -> 128 liver dome")
    axes[0].set_ylabel("Raw AEC")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Clinical-Positive Patients Split by AEC Gate: Mean Raw AEC Curves", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "clinical_positive_aec_pos_neg_mean_curves_S85_raw.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
