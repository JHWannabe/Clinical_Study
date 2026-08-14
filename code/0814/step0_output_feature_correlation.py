from __future__ import annotations

# output feature를 지방계열(SAT/VAT/Total Fat 절대값 + 비율 6종)로 결정한 근거 확인용 correlation 스크립트.
# step2_clinic_aec_ratio.py와 동일한 9개 output feature 각각과 clinic4(sex/age/height/weight),
# AEC-128을 1~128개 구간(2배씩 증가)으로 나눈 구간별 평균값 간 단순 Pearson r/p를 산출한다.
# step0_aec_clinic_correlation.py(clinic4 vs AEC, 체성분 미개입)와 짝을 이루는 파일로,
# 여기서는 실제 예측 대상인 output feature가 clinic4/AEC 각각과 얼마나 겹치는지를 직접 확인한다.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step0_output_feature_correlation"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]  # 128슬라이스를 몇 구간으로 나눠 구간별 평균을 낼지

CLINIC_VARS = ["sex_M", "age", "height", "weight"]

# step2_clinic_aec_ratio.py와 동일한 지방계열 output feature 9종(절대값 3 + 비율 6) -> 그래프 slug.
# 근육계열(TAMA/NAMA/LAMA/IMATA)은 라벨변수와 상관이 높아 순환논리가 되므로 제외([[feedback_no_circular_label_feature]])
FEATURES = {
    "SAT(피하지방)_SUM": "sat",
    "VAT(내장지방)_SUM": "vat",
    "Total Fat_SUM": "total_fat",
    "VAT_SAT_ratio": "vat_sat",
    "VAT_TotalFat_ratio": "vat_total_fat",
    "VAT_TAMA_ratio": "vat_tama",
    "SAT_TotalFat_ratio": "sat_total_fat",
    "SAT_TAMA_ratio": "sat_tama",
    "TotalFat_TAMA_ratio": "total_fat_tama",
}


# VAT/SAT, VAT/Total Fat, VAT/TAMA, SAT/Total Fat, SAT/TAMA, Total Fat/TAMA 6개 비율 컬럼을 원본 SUM 컬럼으로부터
# 산출해 추가 (step2_clinic_aec_ratio.py/step3_clinic_aec_logistic.py와 동일)
def add_ratio_features(meta: pd.DataFrame) -> pd.DataFrame:
    vat = pd.to_numeric(meta["VAT(내장지방)_SUM"], errors="coerce")
    sat = pd.to_numeric(meta["SAT(피하지방)_SUM"], errors="coerce")
    tama = pd.to_numeric(meta["TAMA_SUM"], errors="coerce")
    total_fat = pd.to_numeric(meta["Total Fat_SUM"], errors="coerce")

    meta = meta.copy()
    meta["VAT_SAT_ratio"] = vat / sat
    meta["VAT_TotalFat_ratio"] = vat / total_fat
    meta["VAT_TAMA_ratio"] = vat / tama
    meta["SAT_TotalFat_ratio"] = sat / total_fat
    meta["SAT_TAMA_ratio"] = sat / tama
    meta["TotalFat_TAMA_ratio"] = total_fat / tama
    return meta


# 엑셀 metadata 시트를 로드하고 aec_128 시트의 raw 128포인트를 PatientID 기준으로 병합, 체성분 비율 컬럼을 추가
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")

    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    return add_ratio_features(merged)


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


# output feature 각각과 clinic4 변수 각각의 단순 Pearson |r|/p를 산출(부호는 버리고 절대값만 사용)
def feature_clinic4_correlations(clinic: pd.DataFrame, meta: pd.DataFrame, cohort: str) -> list[dict]:
    rows = []
    for feat in FEATURES:
        y_all = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
        for var in clinic.columns:
            x_all = clinic[var].to_numpy(dtype=float)
            mask = np.isfinite(y_all) & np.isfinite(x_all)
            r, p = stats.pearsonr(x_all[mask], y_all[mask])
            rows.append({"feature": FEATURES[feat], "predictor_group": "clinic4", "predictor": var,
                         "cohort": cohort, "n_seg": np.nan, "segment": np.nan,
                         "r": float(abs(r)), "p_value": float(p), "n": int(mask.sum())})
    return rows


# output feature 각각과 구간 수 n_seg의 AEC 구간평균 각 구간 간 단순 Pearson |r|/p를 산출(부호는 버리고 절대값만 사용)
def feature_aec_correlations(aec_seg: np.ndarray, meta: pd.DataFrame, n_seg: int, cohort: str) -> list[dict]:
    rows = []
    for feat in FEATURES:
        y_all = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
        mask_y = np.isfinite(y_all)
        y, x = y_all[mask_y], aec_seg[mask_y]
        for seg_i in range(aec_seg.shape[1]):
            r, p = stats.pearsonr(x[:, seg_i], y)
            rows.append({"feature": FEATURES[feat], "predictor_group": "aec_segment",
                         "predictor": f"seg{n_seg}_{seg_i + 1}", "cohort": cohort, "n_seg": n_seg,
                         "segment": seg_i + 1, "r": float(abs(r)), "p_value": float(p), "n": int(mask_y.sum())})
    return rows


# 상관계수 CSV 저장 (clinic4/aec_segment 행을 predictor_group으로 구분해 한 파일에 저장)
def write_correlation_rows(rows: list[dict], correlation_csv: Path) -> None:
    df = pd.DataFrame(rows)
    cols = ["feature", "cohort", "predictor_group", "predictor", "n_seg", "segment", "r", "p_value", "n"]
    df[cols].to_csv(correlation_csv, index=False)
    print(f"Saved correlation rows to {correlation_csv}")


# output feature x clinic4 상관계수를 internal/external 나란히 히트맵으로 시각화(칸에 |r|, 0~1 순차 컬러맵).
# feature 9개 기준 셀 텍스트가 겹치지 않도록 행 수(n_feat)에 비례해 figure 높이를 잡고,
# suptitle/서브플롯 title이 겹치지 않도록 상단 여백을 inch 단위로 직접 확보한다
def plot_feature_clinic4_correlation_heatmap(corr_df: pd.DataFrame, clinic_vars: list[str], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    features = list(FEATURES.values())
    cohorts = ["internal", "external"]
    n_feat, n_clinic = len(features), len(clinic_vars)

    cell_w, cell_h = 1.3, 0.62
    top_margin_in = 1.9  # suptitle + 서브플롯 title 두 줄이 들어갈 여백
    fig_w = cell_w * n_clinic * 2 + 3.0
    fig_h = cell_h * n_feat + top_margin_in
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    im = None
    for ax, cohort in zip(axes, cohorts):
        mat = corr_df[corr_df["cohort"] == cohort].pivot(index="feature", columns="predictor", values="r")
        mat = mat.loc[features, clinic_vars]

        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="Reds", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                r_val = mat.iat[i, j]
                ax.text(j, i, f"{r_val:.2f}", ha="center", va="center",
                        fontsize=20, color="white" if r_val > 0.5 else INK_PRIMARY)

        ax.set_xticks(range(n_clinic))
        ax.set_xticklabels(clinic_vars, fontsize=18)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(features, fontsize=18)
        ax.set_title(cohort, fontsize=20, fontweight="bold", color=INK_PRIMARY, pad=10)

    fig.subplots_adjust(top=1 - top_margin_in / fig_h, wspace=0.5)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="|Pearson r|")
    fig.suptitle("Output feature vs clinic4 |Pearson r|", fontsize=20, fontweight="bold", color=INK_PRIMARY, y=0.99)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-clinic4 correlation heatmap to {out_path}")


# 구간 수(행)별 output feature x AEC 구간 상관 히트맵을 internal/external 나란히 그림.
# 구간이 8개 이하면 셀에 r 표시, 많으면 색만 표시. feature 9개 기준 각 n_seg 행의 높이를 n_feat에
# 비례시키고, n_seg 라벨은 rotated ylabel 대신 axes 밖 좌측에 별도 텍스트로 배치해 ytick과 겹치지 않게 함
def plot_feature_aec_correlation_heatmap(corr_df: pd.DataFrame, segment_counts: list[int], out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    features = list(FEATURES.values())
    cohorts = ["internal", "external"]
    nrows, n_feat = len(segment_counts), len(features)

    row_h = 0.4 * n_feat
    top_margin_in = 1.1
    fig, axes = plt.subplots(nrows, 2, figsize=(11, row_h * nrows + top_margin_in), squeeze=False)

    im = None
    for row, n_seg in enumerate(segment_counts):
        for col, cohort in enumerate(cohorts):
            ax = axes[row, col]
            sub = corr_df[(corr_df["n_seg"] == n_seg) & (corr_df["cohort"] == cohort)]
            mat = sub.pivot(index="feature", columns="segment", values="r").loc[features]
            im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="Reds", aspect="auto")
            if n_seg <= 8:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        r_val = mat.iat[i, j]
                        ax.text(j, i, f"{r_val:.2f}", ha="center", va="center", fontsize=11,
                                color="white" if r_val > 0.5 else INK_PRIMARY)
            ax.set_yticks(range(n_feat))
            ax.set_yticklabels(features if col == 0 else [], fontsize=15)
            ax.set_xticks([])
            if col == 0:
                ax.text(-0.42, 0.5, f"n_seg={n_seg}", transform=ax.transAxes, fontsize=16,
                        fontweight="bold", rotation=90, ha="center", va="center", color=INK_PRIMARY)
            if row == 0:
                ax.set_title(cohort, fontsize=20, fontweight="bold", color=INK_PRIMARY, pad=8)

    fig_h = row_h * nrows + top_margin_in
    fig.subplots_adjust(top=1 - top_margin_in / fig_h, right=0.88, left=0.16, hspace=0.7)

    # 기본 fig.colorbar(im, ax=axes)는 전체 8행 높이 중 중앙 근처에만 짧게 뜨는 문제가 있어,
    # 모든 axes의 실제 bounding box(y0~y1)에 맞춰 colorbar 전용 axes를 직접 만들어 전체 높이에 걸치게 함
    all_pos = [ax.get_position() for ax in axes.ravel()]
    y0, y1 = min(p.y0 for p in all_pos), max(p.y1 for p in all_pos)
    x1 = max(p.x1 for p in all_pos)
    cax = fig.add_axes((x1 + 0.02, y0, 0.02, y1 - y0))
    fig.colorbar(im, cax=cax, label="|Pearson r|")
    fig.suptitle("Output feature vs AEC segment-mean |Pearson r| (row=n_seg)",
                 fontsize=20, fontweight="bold", color=INK_PRIMARY, y=0.995)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-AEC segment correlation heatmap to {out_path}")


# output feature 9종(지방계열 절대값 3 + 비율 6) 각각에 대해 clinic4(include_sex=False면 age/height/weight만)와
# AEC-128 구간평균(SEGMENT_COUNTS 구간) 간 단순 상관을 internal/external 코호트별로 산출, CSV와 히트맵으로 저장
def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path, include_sex: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clinic_vars = CLINIC_VARS if include_sex else CLINIC_VARS[1:]
    clinic_int, clinic_ext = clinic4_raw(meta_int, include_sex), clinic4_raw(meta_ext, include_sex)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    corr_rows = (feature_clinic4_correlations(clinic_int, meta_int, "internal")
                 + feature_clinic4_correlations(clinic_ext, meta_ext, "external"))
    for n_seg in SEGMENT_COUNTS:
        corr_rows += feature_aec_correlations(segment_means(aec_int_raw, n_seg), meta_int, n_seg, "internal")
        corr_rows += feature_aec_correlations(segment_means(aec_ext_raw, n_seg), meta_ext, n_seg, "external")

    write_correlation_rows(corr_rows, output_dir / "output_feature_correlations.csv")
    corr_df = pd.DataFrame(corr_rows)

    # feature별 |r| 최댓값(predictor_group별, 코호트/구간 전체 중)을 요약 출력
    for feat_slug in FEATURES.values():
        sub = corr_df[corr_df["feature"] == feat_slug]
        for group in ("clinic4", "aec_segment"):
            grp = sub[sub["predictor_group"] == group]
            top = grp.loc[grp["r"].abs().idxmax()]
            print(f"[{feat_slug} / {group}] max|r|={top['r']:.4f} (predictor={top['predictor']}, "
                  f"cohort={top['cohort']}, p={top['p_value']:.3e}, n={int(top['n'])})")

    plot_feature_clinic4_correlation_heatmap(corr_df[corr_df["predictor_group"] == "clinic4"], clinic_vars,
                                              output_dir / "output_feature_clinic4_correlation_heatmap.png")
    plot_feature_aec_correlation_heatmap(corr_df[corr_df["predictor_group"] == "aec_segment"], SEGMENT_COUNTS,
                                          output_dir / "output_feature_aec_segment_correlation_heatmap.png")


# internal/external 코호트를 로드 후 전체(sex 포함)/남성만/여성만 3가지로 나눠 run()을 각각 실행
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
