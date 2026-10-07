from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# Table S1(전체 확장모델 AUC 비교)에 이미 확정된 숫자로 Figure S1(Baseline 대비 ΔAUC 그래프)을 재생성한다.
# 기존 Figure S1은 생성 스크립트가 남아있지 않은 채 그림만 존재했고, HTN +AEC+LAMA의 External(Sinchon)
# 막대가 실제로는 NAMA 열의 값(0.732, 비유의)으로 잘못 들어가 있었음(정답은 LAMA 0.743*, 유의) - 이를
# Table S1 값에서 직접 재계산해 바로잡는다. 5-fold CV 재현 문제(Table 2/predictions_5fold.xlsx)와는 무관하게,
# Table S1은 이미 결정된 숫자이므로 predictions.xlsx를 다시 돌리지 않고 그 숫자를 그대로 사용한다.

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "outputs" / "clinic4" / "figures" / "delta_auc_vs_baseline.png"

MODELS = ["+AEC (FPCA+Ratio)", "+AEC (FPCA)", "+AEC (Ratio)", "+AEC +VAT", "+AEC +LAMA",
          "+AEC +NAMA", "+AEC +체성분", "체성분만"]
DISEASES = ["HTN", "DM", "CKD"]

# Table S1 (docs/261002_개인연구미팅자료.pptx, slide index 15)의 셀 값을 그대로 옮김.
# (baseline, [모델별 (auc, 유의여부)]) 순서로 MODELS와 매칭
TABLE_S1 = {
    "HTN": {
        "gangnam": (0.796, [(0.804, False), (0.805, False), (0.800, False), (0.810, False),
                             (0.808, True), (0.811, True), (0.811, True), (0.812, True)]),
        "sinchon": (0.718, [(0.730, True), (0.730, True), (0.728, False), (0.735, True),
                             (0.743, True), (0.732, False), (0.741, True), (0.740, True)]),
    },
    "DM": {
        "gangnam": (0.712, [(0.733, True), (0.732, False), (0.730, False), (0.732, False),
                             (0.738, True), (0.732, False), (0.736, False), (0.736, False)]),
        "sinchon": (0.665, [(0.679, False), (0.674, False), (0.665, False), (0.681, False),
                             (0.683, False), (0.673, False), (0.706, True), (0.698, True)]),
    },
    "CKD": {
        "gangnam": (0.770, [(0.774, False), (0.754, False), (0.776, False), (0.772, False),
                             (0.793, True), (0.783, False), (0.784, False), (0.785, False)]),
        "sinchon": (0.615, [(0.622, False), (0.643, True), (0.633, True), (0.638, True),
                             (0.669, True), (0.646, True), (0.681, True), (0.675, True)]),
    },
}


def main() -> None:
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(15, 5))
    x = range(len(MODELS))
    width = 0.35
    for ax, disease in zip(axes, DISEASES):
        for offset, cohort, color in ((-width / 2, "gangnam", "tab:blue"), (width / 2, "sinchon", "tab:orange")):
            baseline, values = TABLE_S1[disease][cohort]
            deltas = [auc - baseline for auc, _ in values]
            xs = [xi + offset for xi in x]
            ax.bar(xs, deltas, width=width, color=color, label=cohort)
            for xi, (auc, sig) in zip(xs, values):
                if sig:
                    ax.annotate("*", (xi, auc - baseline), ha="center", va="bottom", fontsize=13)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(MODELS, rotation=30, ha="right")
        ax.set_title(disease)
        ax.set_ylabel("Δ AUC vs Baseline")
        ax.set_ylim(-0.02, 0.07)
        ax.legend(fontsize=9, loc="upper left")
    fig.text(0.5, -0.02, "* p<0.05 vs Baseline (paired DeLong), Internal=5-fold CV OOF / External=full-refit",
              ha="center", fontsize=10, color="dimgray")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
