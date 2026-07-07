from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    OUT_DIR,
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    risk_direction,
    standardize_train_test,
)


FEATURE = "bank_norm__norm_curv_055_058_mean"
WIDTH = 0.50
LAMBDA = 0.55


def mean_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """행렬의 위치별 평균과 표준오차(SE)를 계산."""
    return x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_midrange_feature_refit에서 채택된 특징(중간 곡률
    구간)이 실제로 임상 양성군을 "유지"와 "하향조정"으로 얼마나 다르게 갈라놓는지, 곡선 자체를
    보고 확인 — youden/sens80/sens85 3개 운영점 각각에 대해):

    1. g1090/sdata를 로드하고, 채택된 특징(FEATURE)의 표준화·방향고정 값을 sdata에서 계산.
    2. 3개 운영점(youden/sens80/sens85) 각각의 임상 임계값에서, 가우시안 게이트(폭=WIDTH,
       람다=LAMBDA)로 sdata 임상양성군을 "유지"와 "하향조정"으로 나눈다.
    3. 3개 패널(운영점별)에 걸쳐, 두 그룹의 정규화 AEC 평균곡선(+95%CI)을 겹쳐 그리고, 채택된
       특징의 탐색 구간(55-58)을 음영으로 표시하며, 각 패널 제목에 하향조정/유지 그룹의 표본수·
       사건수를 함께 표시한 그래프를 PNG로 저장.
    """
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    z_s = xs[:, names.index(FEATURE)] * direction[names.index(FEATURE)]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5), sharey=True)
    xgrid = np.arange(1, 129)
    colors = {"kept": "#4C78A8", "de-escalated": "#D95F02"}
    for ax, op in zip(axes, ["youden", "sens80", "sens85"]):
        th = thresholds[op]["clinical_z"]
        boundary = np.exp(-0.5 * ((c_s - th) / WIDTH) ** 2)
        gate = c_s + LAMBDA * boundary * z_s
        cp = c_s >= th
        masks = {
            "kept": cp & (gate >= th),
            "de-escalated": cp & (gate < th),
        }
        for label, mask in masks.items():
            mean, se = mean_se(s["norm"][mask])
            ax.plot(xgrid, mean, lw=2.2, color=colors[label], label=label)
            ax.fill_between(xgrid, mean - 1.96 * se, mean + 1.96 * se, color=colors[label], alpha=0.13, lw=0)
        ax.axvspan(55, 58, color="#111111", alpha=0.12, label="feature window" if op == "youden" else None)
        de = masks["de-escalated"]
        kept = masks["kept"]
        ax.set_title(
            f"{op}: deesc {int(de.sum())}, events {int(s['y'][de].sum())}/{int(de.sum())}\n"
            f"kept events {int(s['y'][kept].sum())}/{int(kept.sum())}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("AEC_128 point")
        ax.grid(alpha=0.24)
    axes[0].set_ylabel("Patient-normalized AEC")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("sdata clinical-positive groups split by exploratory curvature gate", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_balanced_reference_curves.png", dpi=220)
    plt.close(fig)
    print(OUT_DIR / "external_balanced_reference_curves.png")


if __name__ == "__main__":
    main()
