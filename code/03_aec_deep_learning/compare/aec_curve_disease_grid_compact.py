from __future__ import annotations

# 260904 "AEC FPCA 및 1D CNN 결과 비교" pptx slide7용 압축 그리드 figure. aec_curve_disease_mean_compare.py가
# 만든 outputs/03_aec_deep_learning/compare/curve_mean/aec_mean_curve_by_slice.csv(질환별 PNG 3장, 병렬 배치 시
# 슬라이드 한 장에 넣기엔 세로로 너무 큼)를 그대로 읽어 HTN/DM/CKD x internal/external 6패널을 한 장에 압축
# 재배치만 한다. 이 스크립트는 curve_mean/ 폴더에 쓰지 않고 별도 폴더에 저장(출력 폴더 단일 생산자 원칙).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_CSV = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "curve_mean" / "aec_mean_curve_by_slice.csv"
PERM_CSV = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "curve_mean" / "curve_whole_permutation_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "curve_mean_grid"

FEATURES = ["HTN", "DM", "CKD"]
COHORTS = ["internal", "external"]
GROUP_COLORS = {"neg": "#6b6a66", "pos": "#d63a3a"}
GROUP_LABELS = {"neg": "없음", "pos": "있음"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC_CSV)
    perm = pd.read_csv(PERM_CSV).set_index(["feature", "cohort"])["p_value"]

    fig, axes = plt.subplots(3, 2, figsize=(22, 12), sharey=True)
    y_min, y_max = df["mean"].sub(df["std"]).min(), df["mean"].add(df["std"]).max()
    pad = (y_max - y_min) * 0.05

    for row, feat in enumerate(FEATURES):
        for col, cohort in enumerate(COHORTS):
            ax = axes[row, col]
            sub = df[(df["feature"] == feat) & (df["cohort"] == cohort)]
            for group in ["neg", "pos"]:
                g = sub[sub["group"] == group].sort_values("slice")
                n = g["n"].iloc[0]
                ax.plot(g["slice"], g["mean"], color=GROUP_COLORS[group], linewidth=2.2,
                         label=f"{feat} {GROUP_LABELS[group]} (n={n})")
                ax.fill_between(g["slice"], g["mean"] - g["std"], g["mean"] + g["std"],
                                 color=GROUP_COLORS[group], alpha=0.15, linewidth=0)
            ax.set_ylim(y_min - pad, y_max + pad)
            p_value = perm.loc[(feat, cohort)]
            p_label = "p<0.0001" if p_value < 0.0001 else f"p={p_value:.4f}"
            ax.set_title(f"{feat} — {cohort}  (곡선 전체 permutation {p_label})",
                         fontsize=18, fontweight="bold", color="#161616")
            ax.legend(fontsize=13, loc="lower right", frameon=False)
            ax.tick_params(labelsize=14)
            ax.grid(alpha=0.3)
            if row == 2:
                ax.set_xlabel("AEC slice index (liver→hip)", fontsize=16)
            if col == 0:
                ax.set_ylabel("Patient-wise normalized AEC (z-score)", fontsize=15)

    fig.suptitle("질환 유무별 평균 AEC 곡선 비교 (patient-wise 정규화)", fontsize=24, fontweight="bold", y=1.0)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "aec_mean_curve_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
