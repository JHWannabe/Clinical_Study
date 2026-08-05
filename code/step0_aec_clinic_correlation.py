from __future__ import annotations

# clinic4(sex/age/height/weight) 각 변수와 AEC-128을 1~128개 구간(2배씩 증가)으로 나눈 구간별 평균값 간
# 단순 Pearson r/p를 산출. output feature(체성분)를 거치지 않고 AEC 신호 자체가 임상변수와
# 얼마나 겹치는지(체형에 의한 confound 크기)를 직접 확인하기 위함.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step0"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]  # 128슬라이스를 몇 구간으로 나눠 구간별 평균을 낼지

CLINIC_VARS = ["sex_M", "age", "height", "weight"]


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")

    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return merged


# age/height/weight(+include_sex시 sex)를 raw 값(sex는 0/1)으로 구성. Pearson r은 선형변환에 불변이므로 표준화 불필요.
# 성별을 고정한 남/여 개별 실행에서는 sex_M이 상수가 되어 상관계수가 정의되지 않으므로 include_sex=False로 제외
def clinic4_raw(meta: pd.DataFrame, include_sex: bool = True) -> pd.DataFrame:
    rest = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame()
    if include_sex:
        out["sex_M"] = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    out["age"] = rest["PatientAge"].to_numpy()
    out["height"] = rest["Height"].to_numpy()
    out["weight"] = rest["Weight"].to_numpy()
    return out


# raw AEC-128 행렬(n x 128)을 n_seg개 구간으로 나눠 구간별 평균 행렬(n x n_seg)을 산출
def segment_means(aec_matrix: np.ndarray, n_seg: int) -> np.ndarray:
    chunks = np.array_split(aec_matrix, n_seg, axis=1)
    return np.column_stack([c.mean(axis=1) for c in chunks])


# clinic4 변수 각각과 구간 수 n_seg의 AEC 구간평균 각 구간 간 단순 Pearson r/p를 산출
def clinic_aec_correlations(clinic: pd.DataFrame, aec_seg: np.ndarray, n_seg: int, cohort: str) -> list[dict]:
    rows = []
    for var in clinic.columns:
        y_all = clinic[var].to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        y, x = y_all[mask], aec_seg[mask]
        for seg_i in range(aec_seg.shape[1]):
            r, p = stats.pearsonr(x[:, seg_i], y)
            rows.append({"clinic_var": var, "cohort": cohort, "n_seg": n_seg,
                         "segment": seg_i + 1, "r": float(r), "p_value": float(p), "n": int(mask.sum())})
    return rows


# 상관계수 CSV 저장 (predictor 컬럼 = seg{n_seg}_{segment} 형식으로 다른 correlation 파일과 표기 통일)
def write_correlation_rows(rows: list[dict], correlation_csv: Path) -> None:
    df = pd.DataFrame(rows)
    df["predictor"] = df.apply(lambda row: f"seg{int(row['n_seg'])}_{int(row['segment'])}", axis=1)
    cols = ["clinic_var", "cohort", "predictor", "n_seg", "segment", "r", "p_value", "n"]
    df[cols].to_csv(correlation_csv, index=False)
    print(f"Saved correlation rows to {correlation_csv}")


# 구간 수(행)별 clinic4 x AEC 구간 상관 히트맵을 internal/external 나란히 그림.
# 구간이 8개 이하면 셀에 r 표시, 많으면 색만 표시
def plot_clinic_aec_correlation_heatmap(corr_df: pd.DataFrame, segment_counts: list[int],
                                         clinic_vars: list[str], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    cohorts = ["internal", "external"]
    nrows = len(segment_counts)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 2.0 * nrows), squeeze=False)

    im = None
    for row, n_seg in enumerate(segment_counts):
        for col, cohort in enumerate(cohorts):
            ax = axes[row, col]
            sub = corr_df[(corr_df["n_seg"] == n_seg) & (corr_df["cohort"] == cohort)]
            mat = sub.pivot(index="clinic_var", columns="segment", values="r").loc[clinic_vars]
            im = ax.imshow(mat.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            if n_seg <= 8:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        r_val = mat.iat[i, j]
                        ax.text(j, i, f"{r_val:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(r_val) > 0.5 else INK_PRIMARY)
            ax.set_yticks(range(len(clinic_vars)))
            ax.set_yticklabels(clinic_vars if col == 0 else [], fontsize=8)
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(f"n_seg={n_seg}", fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(cohort, fontsize=11, fontweight="bold", color=INK_PRIMARY)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Pearson r")
    fig.suptitle("clinic4 vs AEC segment-mean Pearson r (row=n_seg)",
                 fontsize=12, fontweight="bold", color=INK_PRIMARY)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved clinic4-AEC segment correlation heatmap to {out_path}")


# clinic4(sex/age/height/weight, include_sex=False면 age/height/weight만) 각 변수와
# AEC-128 구간평균(SEGMENT_COUNTS 구간) 간 단순 상관을 internal/external 코호트별로 산출, CSV와 히트맵으로 저장
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clinic_vars = CLINIC_VARS if include_sex else CLINIC_VARS[1:]
    clinic_int, clinic_ext = clinic4_raw(meta_int, include_sex), clinic4_raw(meta_ext, include_sex)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    corr_rows = []
    for n_seg in SEGMENT_COUNTS:
        corr_rows += clinic_aec_correlations(clinic_int, segment_means(aec_int_raw, n_seg), n_seg, "internal")
        corr_rows += clinic_aec_correlations(clinic_ext, segment_means(aec_ext_raw, n_seg), n_seg, "external")

    write_correlation_rows(corr_rows, output_dir / "aec_clinic_correlations.csv")
    corr_df = pd.DataFrame(corr_rows)

    # 각 clinic 변수별 |r| 최댓값(코호트/구간 전체 중)을 요약 출력
    for var in clinic_vars:
        sub = corr_df[corr_df["clinic_var"] == var]
        top = sub.loc[sub["r"].abs().idxmax()]
        print(f"[{var}] max|r|={top['r']:.4f} (cohort={top['cohort']}, n_seg={int(top['n_seg'])}, "
              f"segment={int(top['segment'])}, p={top['p_value']:.3e}, n={int(top['n'])})")

    plot_clinic_aec_correlation_heatmap(corr_df, SEGMENT_COUNTS, clinic_vars,
                                         output_dir / "aec_clinic_correlation_heatmap.png")


# 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행
def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    sex_int = meta_int["PatientSex"].astype(str).str.upper()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper()

    run(meta_int, meta_ext, OUTPUT_DIR / "total", include_sex=True)
    for sex_label, sub_dir in (("M", OUTPUT_DIR / "male"), ("F", OUTPUT_DIR / "female")):
        print(f"\n=== sex={sex_label} ({sub_dir.name}) ===")
        run(meta_int[sex_int == sex_label].reset_index(drop=True),
            meta_ext[sex_ext == sex_label].reset_index(drop=True), sub_dir, include_sex=False)


if __name__ == "__main__":
    main()
