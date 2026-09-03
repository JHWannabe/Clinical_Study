from __future__ import annotations

# HTN/DM/CKD 각 질환 유무(0/1, metadata 컬럼)에 따라 환자군을 둘로 나누고, patient-wise z-score로
# 정규화한 AEC-128 곡선의 그룹 평균(±SD 밴드)을 internal/external 코호트별로 겹쳐 그려 비교한다
# (2026-09-03 사용자 요청: "raw로 하면 단순 비교가 힘드니까 patient-wise로 전처리 후에 비교" ->
# "그래프에 std도 포함해서 저장해" -> "SEM은 제거해"). patient-wise 정규화 정의는
# aec_fusion_common.prepare_curve의 "patient_zscore"와 동일(행 단위 계산만 사용, 코호트 통계 미참조 —
# [[feedback_aec_preprocessing_methods]]). [[feedback_aec_curve_wholistic]]에 따라 슬라이스별 유의성
# 검정은 하지 않고 곡선 형태 비교만 시각화한다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "compare" / "curve_mean"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
AGE_CUTOFF = 20
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]

FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}
GROUP_COLORS = {"neg": "#6b6a66", "pos": "#d63a3a"}
GROUP_LABELS = {"neg": "없음", "pos": "있음"}


# step_disease_logistic.py의 load_cohort와 동일 정의(연령<20 제외, metadata+aec_128 병합)
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# patient-wise z-score: 환자(행) 단위로만 평균/표준편차를 계산 — aec_fusion_common.prepare_curve의
# "patient_zscore"와 동일 구현
def patient_zscore(curve: np.ndarray) -> np.ndarray:
    mean = curve.mean(axis=1, keepdims=True)
    std = curve.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (curve - mean) / std


def group_stats(curve_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = curve_norm.mean(axis=0)
    std = curve_norm.std(axis=0, ddof=1)
    return mean, std


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = {"internal": load_cohort(INTERNAL_XLSX), "external": load_cohort(EXTERNAL_XLSX)}
    for name, df in cohorts.items():
        print(f"[{name}] n={len(df)}")

    x = np.arange(1, N_SLICES + 1)
    summary_rows: list[dict] = []
    stats_rows: list[dict] = []
    curves_by_panel: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for feature in FEATURES:
        for cohort_name, df in cohorts.items():
            y = pd.to_numeric(df[feature], errors="coerce")
            valid = y.notna()
            curve_raw = df.loc[valid, AEC_COLS].astype(float).to_numpy()
            curve_norm = patient_zscore(curve_raw)
            y_valid = y[valid].to_numpy()

            group_data: dict[str, np.ndarray] = {}
            for group_key, mask_val in [("neg", 0), ("pos", 1)]:
                sub = curve_norm[y_valid == mask_val]
                group_data[group_key] = sub
                mean, std = group_stats(sub)
                for i, (m, sd) in enumerate(zip(mean, std), start=1):
                    summary_rows.append({
                        "feature": feature, "cohort": cohort_name, "group": group_key,
                        "n": sub.shape[0], "slice": i, "mean": m, "std": sd,
                    })
            curves_by_panel[(feature, cohort_name)] = group_data

            n_pos, n_neg = group_data["pos"].shape[0], group_data["neg"].shape[0]
            print(f"[{feature}/{cohort_name}] 있음 n={n_pos}, 없음 n={n_neg}")
            stats_rows.append({"feature": feature, "cohort": cohort_name, "n_pos": n_pos, "n_neg": n_neg})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "aec_mean_curve_by_slice.csv", index=False)
    print(f"Saved per-slice mean/SD to {OUTPUT_DIR / 'aec_mean_curve_by_slice.csv'}")
    pd.DataFrame(stats_rows).to_csv(OUTPUT_DIR / "group_n.csv", index=False)

    # 세 질환 x 두 코호트 전체 패널에서 공통 y축 범위를 구해 통일(feedback_code_plot_unified_ylim)
    all_bounds = []
    for group_data in curves_by_panel.values():
        for sub in group_data.values():
            mean, std = group_stats(sub)
            all_bounds.append(mean - std)
            all_bounds.append(mean + std)
    all_bounds = np.concatenate(all_bounds)
    margin = 0.05 * (all_bounds.max() - all_bounds.min())
    ylim = (all_bounds.min() - margin, all_bounds.max() + margin)

    for feature in FEATURES:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        for ax, cohort_name in zip(axes, ["internal", "external"]):
            group_data = curves_by_panel[(feature, cohort_name)]
            for group_key in ["neg", "pos"]:
                sub = group_data[group_key]
                mean, std = group_stats(sub)
                color = GROUP_COLORS[group_key]
                ax.plot(x, mean, color=color, linewidth=2.5, label=f"{feature} {GROUP_LABELS[group_key]} (n={sub.shape[0]})")
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
            ax.set_ylim(*ylim)
            ax.set_xlabel("AEC slice index (liver→hip)", fontsize=24)
            ax.set_title(cohort_name, fontsize=20, fontweight="bold", color="#161616")
            ax.tick_params(labelsize=18)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=16, loc="best")
        axes[0].set_ylabel("Patient-wise normalized AEC (z-score)", fontsize=24)

        fig.suptitle(f"{feature} 유무에 따른 평균 AEC 곡선 비교 (patient-wise 정규화)", fontsize=22, fontweight="bold")
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"aec_mean_curve_{feature}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"[완료] 결과 저장 디렉터리: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
