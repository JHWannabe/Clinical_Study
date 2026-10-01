from __future__ import annotations

# outputs/clinic4/predictions_5fold_unbalanced.xlsx(clinic4_5fold_unbalanced.py 산출물)의 patient-level 예측으로 질병별
# calibration plot을 그린다. docs/261002_개인연구미팅자료.pptx 슬라이드 8(Figure 2)에 이미 박혀 있던
# Best(+AEC+체성분) 단독 그림과 같은 스타일(1x3 subplot, 5 quantile bin, gangnam=실선/sinchon=점선,
# 회색 점선 대각선 "Perfect calibration")을 그대로 유지한 채, Baseline과 Best를 한 그림에 4선으로
# 합치지 않고 모델별로 별도 파일 2장(calibration_plot_baseline.png / calibration_plot_best.png)을
# 각각 gangnam/sinchon 2선짜리로 저장한다(2026-09-30 재요청: 4선은 색이 헷갈려서 되돌림).

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트(없으면 그래프 한글 라벨이 깨짐)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from clinic4_logistic_regression import CLINIC4_DIR, DISEASES, f3

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PRED_XLSX = CLINIC4_DIR / "predictions_5fold_unbalanced.xlsx"  # 덱 분석과 같은 전체 표본(n=1,260) 예측값
DISEASE_LABEL = {"HTN": "고혈압(HTN)", "DM": "당뇨병(DM)", "CKD": "만성신장질환(CKD)"}
N_BINS = 5
# model -> (파일명, legend/title에 쓸 이름)
MODEL_FILES = {
    "baseline": (CLINIC4_DIR / "calibration_plot_baseline.png", "Baseline"),
    "best": (CLINIC4_DIR / "calibration_plot_best.png", "Best (+AEC+체성분)"),
}
# cohort -> (색, 선스타일, legend에 쓸 이름) - 원래 단일모델 플롯 스타일(gangnam=파랑 실선/sinchon=주황 점선) 유지
COHORT_STYLE = {"gangnam": ("tab:blue", "-", "gangnam"), "sinchon": ("tab:orange", "--", "sinchon")}


# score를 5분위(quantile)로 나눠 bin별 (평균 predicted prob, 관측 빈도) 좌표를 반환.
# score에 동률(duplicate)이 많아 5개 고유 경계가 안 나오면 pd.qcut이 자동으로 bin 수를 줄인다(duplicates="drop")
def calibration_curve_points(y: np.ndarray, score: np.ndarray, n_bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray]:
    bins = pd.qcut(score, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "score": score, "bin": bins})
    grouped = df.groupby("bin", observed=True).agg(pred=("score", "mean"), obs=("y", "mean"))
    return grouped["pred"].to_numpy(), grouped["obs"].to_numpy()


def main() -> None:
    predictions = pd.read_excel(PRED_XLSX, sheet_name="predictions")
    summary = pd.read_excel(PRED_XLSX, sheet_name="summary")
    brier_col = {"gangnam": "brier_gangnam_oof", "sinchon": "brier_sinchon_frozen"}

    for model_name, (out_png, model_label) in MODEL_FILES.items():
        fig, axes = plt.subplots(1, len(DISEASES), figsize=(5 * len(DISEASES), 5))
        for ax, disease in zip(axes, DISEASES):
            summary_row = summary[(summary["disease"] == disease) & (summary["model"] == model_name)].iloc[0]
            for cohort, (color, linestyle, cohort_label) in COHORT_STYLE.items():
                pred_c = predictions[(predictions["disease"] == disease) & (predictions["model"] == model_name)
                                     & (predictions["cohort"] == cohort)]
                pred_pts, obs_pts = calibration_curve_points(pred_c["y"].to_numpy(), pred_c["score"].to_numpy())
                brier = summary_row[brier_col[cohort]]
                ax.plot(pred_pts, obs_pts, marker="o", linestyle=linestyle, color=color,
                        label=f"{cohort_label} (Brier={f3(brier)})")
            ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1, label="Perfect calibration")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(DISEASE_LABEL[disease])
            ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Calibration Plot — {model_label}, "
                     "Internal=5-fold OOF / External=full-refit, 5 quantile bins")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
