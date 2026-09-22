"""사용자 요청(2026-09-17): 강남/신촌(internal/external) 5개 AUC 결과 섹션(스칼라/비율/FPCA/
Upper-Lower ratio/slope) 각각에 대해, AUC 슬라이드 뒤에 붙일 Confusion Matrix를 섹션별로 추가.
대상 모델은 clinic4 baseline과 해당 섹션 내 external ΔAUC가 가장 높은 모델(사용자 확인:
"clinic4 + 해당 섹션 내 best 모델"). 임계값은 Youden index(사용자 확인) - 이미
outputs/0910/logistic_regression_summary.csv에 저장된 threshold/sensitivity/specificity가
code/0910/clinic_body_composition_individual_compare.py의 youden_threshold()로 계산된 값이므로
재적합 없이 그대로 재사용한다. TP/FP/FN/TN은 정수이고 sensitivity=tp/n_pos, specificity=tn/n_neg는
그 정수비의 부동소수 표현이므로 반올림으로 정확히 복원 가능(검증: 최대 오차 1e-13 수준).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = PROJECT_ROOT / "outputs" / "0910" / "logistic_regression_summary.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "0910" / "confusion_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "bc_compare", PROJECT_ROOT / "code" / "0910" / "clinic_body_composition_individual_compare.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def confusion_counts(row: pd.Series) -> tuple[int, int, int, int]:
    n_pos, n_neg = int(row["n_pos"]), int(row["n"] - row["n_pos"])
    tp = int(round(row["sensitivity"] * n_pos))
    fn = n_pos - tp
    tn = int(round(row["specificity"] * n_neg))
    fp = n_neg - tn
    return tn, fp, fn, tp


# 사용자 요청(2026-09-18): 행=Predicted(Positive 위/Negative 아래), 열=Actual(Positive 왼쪽/Negative
# 오른쪽) 배치로 변경 - TP가 좌상단, TN이 우하단에 오도록
def draw_cm(ax, tn: int, fp: int, fn: int, tp: int, title: str) -> None:
    mat = np.array([[tp, fp], [fn, tn]])
    mat_norm = mat / mat.sum()
    ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=mat_norm.max() * 1.15)
    for (i, j), v in np.ndenumerate(mat):
        color = "white" if mat_norm[i, j] > mat_norm.max() * 0.6 else "black"
        ax.text(j, i, f"{v}\n({v / mat.sum():.1%})", ha="center", va="center", fontsize=11, color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive (1)", "Negative (0)"], fontsize=9)
    ax.set_yticklabels(["Positive (1)", "Negative (0)"], fontsize=9)
    ax.set_xlabel("Actual Values", fontsize=9)
    ax.set_ylabel("Predicted Values", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")


def best_model_in_section(ext: pd.DataFrame, feat: str, models: list[str]) -> str:
    sub = ext[(ext["feature"] == feat) & (ext["model"].isin(models)) & (ext["model"] != "clinic4")]
    return sub.loc[sub["auc"].idxmax(), "model"]


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    ext = summary[summary["cohort"] == "external"].copy()
    features = [f for f in bc.FEATURES if f in ext["feature"].unique()]

    # 슬라이드의 기존 그림 자리(가로:세로 ≈ 2.36:1)에 그대로 맞도록 질환을 열, 모델(clinic4/best)을
    # 행으로 배치(3x2 세로 배치는 원래의 넓은 자리에 넣으면 심하게 왜곡됨)
    for section_name, models in bc.MODEL_SECTIONS.items():
        fig, axes = plt.subplots(2, len(features), figsize=(4.1 * len(features), 7.2))
        for c, feat in enumerate(features):
            best_model = best_model_in_section(ext, feat, models)
            for r, model_name in enumerate(["clinic4", best_model]):
                row = ext[(ext["feature"] == feat) & (ext["model"] == model_name)].iloc[0]
                tn, fp, fn, tp = confusion_counts(row)
                title = (f"{feat}: {bc.MODEL_LABELS[model_name]}\n"
                         f"AUC={row['auc']:.3f} thr={row['threshold']:.3f}")
                draw_cm(axes[r, c], tn, fp, fn, tp, title)
        fig.suptitle(f"{bc.SECTION_TITLES[section_name]} — Confusion Matrix (external, Youden threshold)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = OUT_DIR / f"{section_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
