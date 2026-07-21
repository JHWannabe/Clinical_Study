"""
Stage-1 screen-positive(TP/FP) 환자들을 clinic 변수(height/weight/age/bmi) 공간에
scatter로 시각화한다 -- error_feature_analysis.md의 TP vs FP Welch t-test 결과를
그림으로 보여주는 보조 자료.

error_feature_analysis()가 저장한 error_feature_analysis_rows.csv(내부/gangnam
코호트, OOF 예측 기준 TP/FN/TN/FP 라벨)를 그대로 재사용하고, 색상은 이 저장소의
AEC 곡선 비교(aec_curve_comparison.py)에서 이미 쓰는 두 그룹 색(COL_A/COL_B)을
그대로 맞춰 덱 전체에서 TP/FP 색이 일관되게 보이도록 한다.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

COL_TP = "#2d2ad6"   # blue -- aec_curve_comparison.py의 COL_A와 동일
COL_FP = "#af1b1b"   # red  -- aec_curve_comparison.py의 COL_B와 동일
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

ROWS_CSV = os.path.join(os.path.dirname(__file__), "../..", "outputs", "0_clinic-only_baseline",
                         "error_feature_analysis_rows.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "../..", "outputs", "0_clinic-only_baseline",
                         "tp_vs_fp_scatter.png")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def scatter_panel(ax, df, x_col, y_col, x_label, y_label, title):
    for group, color in [("FP", COL_FP), ("TP", COL_TP)]:
        sub = df[df["group"] == group]
        ax.scatter(sub[x_col], sub[y_col], s=26, linewidths=0.6, edgecolors=SURFACE,
                   color=color, alpha=0.75, label=f"{group} (n={len(sub)})", zorder=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="best")
    style_axes(ax)


def main():
    rows = pd.read_csv(ROWS_CSV)
    df = rows[rows["group"].isin(["TP", "FP"])].copy()

    fig, axes = plt.subplots(2, 1, figsize=(7, 8.6))
    scatter_panel(axes[0], df, "height", "weight", "Height (cm)", "Weight (kg)",
                  "Height vs Weight")
    scatter_panel(axes[1], df, "age", "bmi", "Age (yr)", "BMI (kg/m²)",
                  "Age vs BMI")
    fig.suptitle("Stage-1 screen-positive (TP vs FP): clinic 변수 scatter",
                 fontsize=13, color=INK_PRIMARY)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
