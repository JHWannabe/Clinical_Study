from __future__ import annotations

# 체성분 스칼라(bodycomp_pvalue.png)에 이어, 원본 128구간(liver->pubis) 곡선 자체도 질환 유무에 따라
# 형태가 다른지 확인. 방법론은 code/03_aec_deep_learning/compare/aec_curve_disease_mean_compare.py와
# 동일하되 정규화만 patient-wise ratio(값/환자 자신의 128구간 평균)로 사용(사용자 요청 2026-09-22: z-score
# 대신 patient-wise ratio) 후 그룹 평균±SD 밴드, 슬라이스별 t-test 대신 곡선 전체 L2^2 permutation
# test 1개 p-value - [[feedback_aec_curve_wholistic]]: 슬라이스별 유의성 검정은 하지 않는다는 기존 결정을 따름).
# AEC 하나만 보던 기존 스크립트를 clinic4_bodycomp_compare.py의 CURVE_SOURCES(7개 곡선)로 확장하고,
# 이번 세션에서 다루던 성별 필터(SEX)를 그대로 적용. 코호트는 gangnam(internal)/sinchon/new10000 중 선택
# 가능(new10000은 CKD 컬럼이 없어 HTN/DM만 - [[new10000_auc_ceiling]]과 동일 제약, 해당 질환은 자동 skip).
# 사용법: python clinic4_bodycomp_curve128_compare.py [M|F|ALL] [gangnam|sinchon|new10000] (기본 M gangnam)
# ALL은 성별 필터 없이(M/F 둘 다 포함, 성별 미상만 제외) 통합 비교

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from clinic4_baseline_logistic import CLINICAL_BASE_COLS, DATA_DIR, DISEASES, INTERNAL_XLSX, METADATA_SHEET
from clinic4_bodycomp_compare import CURVE_SOURCES, N_SLICES, curve_cols

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

SEX = sys.argv[1].upper() if len(sys.argv) > 1 else "M"
assert SEX in ("M", "F", "ALL"), "SEX는 M/F/ALL 중 하나"
DATASETS = {"gangnam": INTERNAL_XLSX, "sinchon": DATA_DIR / "sinchon_final_dataset.xlsx",
            "new10000": DATA_DIR / "new10000_final_dataset.xlsx"}
DATASET = sys.argv[2] if len(sys.argv) > 2 else "gangnam"
assert DATASET in DATASETS, f"DATASET은 {list(DATASETS)} 중 하나"
SEX_DIR_NAME = {"M": "male", "F": "female", "ALL": "all"}[SEX]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1002" / f"clinic4_{SEX_DIR_NAME}_curve128_compare_{DATASET}"

GROUP_COLORS = {"neg": "#6b6a66", "pos": "#d63a3a"}
GROUP_LABELS = {"neg": "없음", "pos": "있음"}
N_PERMUTATIONS = 20000
PERM_SEED = 20260709


# SEX(M/F/ALL)로 필터링한 뒤 CURVE_SOURCES 7개 곡선을 전부 병합(clinic4_bodycomp_compare.load_cohort_with_curves와
# 동일하되 성별 제한만 추가). ALL은 성별 미상만 제외하고 M/F 둘 다 포함
def load_cohort_curves_by_sex(path: Path) -> pd.DataFrame:
    meta = pd.read_excel(path, sheet_name=METADATA_SHEET, engine="openpyxl").reset_index(drop=True)
    sex_col = meta["PatientSex"].astype(str).str.upper()
    valid_sex = sex_col.isin(["M", "F"]) if SEX == "ALL" else sex_col.eq(SEX)
    valid_clinic = meta[CLINICAL_BASE_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    meta = meta[valid_sex & valid_clinic].reset_index(drop=True)
    n0 = len(meta)
    for _key, (sheet, prefix) in CURVE_SOURCES.items():
        curve = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        meta = meta.merge(curve[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
    meta = meta.reset_index(drop=True)
    print(f"[{path.stem}] 성별({SEX})/clinic4 결측/곡선 없음 제외 후 n={len(meta)} (성별+clinic4 필터 후 {n0}명)")
    return meta


# patient-wise ratio: 환자(행) 자신의 128구간 평균으로 나눠 비율로 표현(z-score처럼 표준편차로 나누지 않음)
def patient_ratio(curve: np.ndarray) -> np.ndarray:
    mean = curve.mean(axis=1, keepdims=True)
    mean[mean == 0] = 1.0
    return curve / mean


# 곡선 전체(128포인트) 평균곡선 간 L2^2 거리를 단일 통계량으로 삼는 permutation test(슬라이스별 t-test 대신
# 곡선 전체 단위 검정 - aec_curve_disease_mean_compare.py와 동일)
def curve_level_permutation_test(pos: np.ndarray, neg: np.ndarray, n_perm: int = N_PERMUTATIONS,
                                  seed: int = PERM_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs_stat = float(np.sum((pos.mean(axis=0) - neg.mean(axis=0)) ** 2))
    pooled = np.vstack([pos, neg])
    n_pos, n_total = pos.shape[0], pooled.shape[0]
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n_total)
        perm_stats[i] = np.sum((pooled[idx[:n_pos]].mean(axis=0) - pooled[idx[n_pos:]].mean(axis=0)) ** 2)
    p_value = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)
    return obs_stat, p_value


def group_stats(curve_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return curve_norm.mean(axis=0), curve_norm.std(axis=0, ddof=1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_cohort_curves_by_sex(DATASETS[DATASET])
    sex_label = {"M": "남자", "F": "여자", "ALL": "전체(성별 통합)"}[SEX]
    diseases = [d for d in DISEASES if d in meta.columns]
    if len(diseases) < len(DISEASES):
        print(f"[{DATASET}] 컬럼 없어 skip: {[d for d in DISEASES if d not in diseases]}")
    x = np.arange(1, N_SLICES + 1)

    summary_rows, perm_rows = [], []
    for curve_key, (_sheet, prefix) in CURVE_SOURCES.items():
        cols = curve_cols(prefix)
        curve_norm_all = patient_ratio(meta[cols].astype(float).to_numpy())

        # 공통 y축 범위를 대상 질환 전체에서 통일
        panel_data = {}
        all_bounds = []
        for disease in diseases:
            y = meta[disease].to_numpy(dtype=int)
            group_data = {g: curve_norm_all[y == v] for g, v in [("neg", 0), ("pos", 1)]}
            panel_data[disease] = group_data
            for sub in group_data.values():
                mean, std = group_stats(sub)
                all_bounds += [mean - std, mean + std]
        all_bounds = np.concatenate(all_bounds)
        margin = 0.05 * (all_bounds.max() - all_bounds.min())
        ylim = (all_bounds.min() - margin, all_bounds.max() + margin)

        fig, axes = plt.subplots(1, len(diseases), figsize=(7 * len(diseases), 6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, disease in zip(axes, diseases):
            group_data = panel_data[disease]
            obs_stat, p_value = curve_level_permutation_test(group_data["pos"], group_data["neg"])
            perm_rows.append({"variable": curve_key, "disease": disease, "l2_stat": obs_stat,
                               "n_perm": N_PERMUTATIONS, "p_value": p_value})
            for group_key in ["neg", "pos"]:
                sub = group_data[group_key]
                mean, std = group_stats(sub)
                color = GROUP_COLORS[group_key]
                ax.plot(x, mean, color=color, linewidth=2, label=f"{GROUP_LABELS[group_key]} (n={sub.shape[0]})")
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
                for i, (m, sd) in enumerate(zip(mean, std), start=1):
                    summary_rows.append({"variable": curve_key, "disease": disease, "group": group_key,
                                          "slice": i, "mean": m, "std": sd})
            ax.set_ylim(*ylim)
            ax.set_xlabel(f"{curve_key} slice index (liver→pubis)")
            ax.set_title(f"{disease} (permutation p={p_value:.4f})")
            ax.grid(alpha=0.3)
            ax.legend()
        axes[0].set_ylabel(f"Patient-wise ratio {curve_key} (value / patient mean)")
        fig.suptitle(f"{sex_label}: 질환 유무에 따른 {curve_key} 128구간 곡선 비교", fontsize=16, fontweight="bold")
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"curve128_{curve_key.lower()}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "curve128_mean_by_slice.csv", index=False)
    pd.DataFrame(perm_rows).round(6).to_csv(OUTPUT_DIR / "curve128_permutation_test.csv", index=False)
    print(f"Saved curve128_mean_by_slice.csv, curve128_permutation_test.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
