from __future__ import annotations

# 260904 "AEC FPCA 및 1D CNN 결과 비교" pptx slide5(FPCA 알고리즘 설명)용 PC1-3 loading curve(eigenfunction)
# 시각화. step_disease_logistic.py의 select_fpca_n_by_elbow + PCA(n_components=3).fit(aec_int_raw)와 동일한
# 입력(연령<20 제외 gangnam_final_dataset.xlsx raw AEC-128, patient-wise 정규화 아님)으로 PCA를 그대로
# 재현해 components_(로딩 곡선)만 새로 뽑아 그린다 — 기존 산출물에는 FPCA score(fpca_scores.xlsx)만 있고
# 로딩 곡선 자체는 저장되어 있지 않았음. outputs/01_disease_logistic/logistic/은 step_disease_logistic.py
# 전용 출력 폴더라 여기 쓰지 않고 별도 폴더에 저장(출력 폴더 단일 생산자 원칙).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "01_disease_logistic" / "fpca_loading"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
AGE_CUTOFF = 20
N_SLICES = 128
N_FPCA = 3
SEED = 20260709
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
PC_COLORS = {1: "#1f5fa8", 2: "#d9791e", 3: "#2ea86b"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_excel(INTERNAL_XLSX, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(INTERNAL_XLSX, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    aec_raw = merged[AEC_COLS].astype(float).to_numpy()

    pca = PCA(n_components=N_FPCA, random_state=SEED).fit(aec_raw)
    evr = pca.explained_variance_ratio_
    mean_curve = pca.mean_
    slices = list(range(1, N_SLICES + 1))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.plot(slices, mean_curve, color="#161616", linewidth=2.5, label="internal 평균 곡선")
    ax.set_title("internal 코호트 평균 AEC 곡선(PCA 기준선)", fontsize=19, fontweight="bold")
    ax.set_xlabel("AEC slice index (liver→hip)", fontsize=16)
    ax.set_ylabel("AEC (raw mA)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(N_FPCA):
        pc = i + 1
        ax.plot(slices, pca.components_[i], color=PC_COLORS[pc], linewidth=2.5,
                 label=f"PC{pc} (explained var={evr[i]*100:.1f}%)")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title(f"PC1–{N_FPCA} loading curve(누적 explained variance={evr.sum()*100:.1f}%)",
                 fontsize=19, fontweight="bold")
    ax.set_xlabel("AEC slice index (liver→hip)", fontsize=16)
    ax.set_ylabel("Loading (eigenfunction)", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fpca_loading_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"n_patients={aec_raw.shape[0]}, explained_variance_ratio(PC1-{N_FPCA})={evr.round(4)}, "
          f"cumulative={evr.sum():.4f}")


if __name__ == "__main__":
    main()
