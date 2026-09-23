from __future__ import annotations

# clinic4_logistic_regression.py의 baseline(성별/나이/신장/체중)에 aec_128(liver->pubis 128구간 AEC 곡선)을
# 추가 input으로 넣는다. AEC는 전처리 없이 원본 곡선 그대로 FPCA(n=3, code/01_disease_logistic/
# clinic4_bodycomp_compare.py에서 elbow로 검증된 값)와 Upper/Lower ratio(곡선을 앞/뒤 절반으로 나눈
# 구간평균의 비) 두 형태로 파생시켜 clinic4와 함께 StandardScaler로 표준화 후 로지스틱 회귀에 넣는다.
# uplow_ratio와 aec_fpca_pc2가 VIF>10으로 서로 겹쳐(2026-09-22 결과) 어느 쪽이 억눌렸는지 구분이 안 되므로
# ratio 포함(fpca_ratio)/FPCA만(fpca_only) 두 변형을 같이 돌려 AUC를 비교한다. 추가로 fpca_only 변형의
# PC 계수를 128-slice 곡선 축으로 역변환(PCA는 선형변환이라 score_k=(curve-mean).component_k, 로지스틱
# 표준화 계수 beta_k를 대입하면 logit 기여분 = curve.[sum_k (beta_k/score_std_k)*component_k] + const로
# 재전개됨 - [[fpca_coefficient_recovery_technique]])해 어느 slice(해부학적 위치)가 기여하는지 확인한다.
# 그 외 split/외부검증/출력 로직은 baseline과 전부 동일해 클래스 재사용 없이 함수만 가져다 쓴다.

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clinic4_logistic_regression import (
    DATA_XLSX, DISEASES, EXTERNAL_COHORTS, PROJECT_ROOT, SEED, format_floats, load_data, run_and_save,
)

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinic4" / "aec"

AEC_SHEET, AEC_PREFIX = "aec_128", "aec"
N_SLICES = 128
AEC_FPCA_N = 3  # AEC elbow로 검증된 PCA 성분 수(clinic4_bodycomp_compare.py CURVE_FPCA_N_ELBOW["AEC"]와 동일)
AEC_COLS = [f"{AEC_PREFIX}_{i}" for i in range(1, N_SLICES + 1)]
FPCA_COLS = [f"aec_fpca_pc{i}" for i in range(1, AEC_FPCA_N + 1)]
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
# ratio 포함/FPCA만/ratio만 세 변형(VIF 다중공선성 비교 + FPCA 압축이 실제로 필요한지 확인용). key가 출력
# 서브폴더명이 됨
VARIANTS: dict[str, list[str]] = {
    "fpca_ratio": ["aec_uplow_ratio"] + FPCA_COLS, "fpca_only": FPCA_COLS, "ratio_only": ["aec_uplow_ratio"],
}


# baseline의 load_data(clinic4 결측 제외)에 aec_128 원본 곡선을 PatientID로 inner join(곡선 없는 환자는 제외)
def load_data_with_aec(path: Path) -> pd.DataFrame:
    df = load_data(path)
    aec = pd.read_excel(path, sheet_name=AEC_SHEET)[["PatientID"] + AEC_COLS]
    n0 = len(df)
    df = df.merge(aec, on="PatientID", how="inner").reset_index(drop=True)
    if len(df) < n0:
        print(f"[{path.stem}] aec_128 없음 {n0 - len(df)}/{n0}명 제외")
    return df


# AEC 128구간 곡선(전처리 없이 원본 그대로)에서 Upper/Lower ratio와 FPCA 성분을 뽑아 df에 새 컬럼으로
# 붙인다. PCA는 gangnam에서만 fit하고 외부 코호트에는 frozen으로 넘겨 transform만 함(누수 방지)
def add_aec_features(df: pd.DataFrame, pca: PCA | None = None) -> tuple[pd.DataFrame, PCA]:
    curve = df[AEC_COLS].to_numpy(dtype=float)
    half = N_SLICES // 2
    df = df.copy()
    df["aec_uplow_ratio"] = curve[:, :half].mean(axis=1) / curve[:, half:].mean(axis=1)
    if pca is None:
        pca = PCA(n_components=AEC_FPCA_N, random_state=SEED).fit(curve)
    fpca = pca.transform(curve)
    for i in range(AEC_FPCA_N):
        df[f"aec_fpca_pc{i + 1}"] = fpca[:, i]
    return df, pca


# 성별(M=1/F=0) + 표준화된 나이/신장/체중/extra_cols로 입력 행렬 구성. scaler는 gangnam에서 fit해
# 외부 코호트에는 frozen으로 transform만 함(baseline clinical_matrix와 동일 원칙)
def clinical_matrix(df: pd.DataFrame, extra_cols: list[str],
                     scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_BASE_COLS + extra_cols].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


# 변형 하나(extra_cols)에 대해 전체 질병 학습/외부검증을 돌리고 OUTPUT_DIR(clinic4 루트)에 결과를 저장.
# summary/coef_df/scaler를 반환(변형간 AUC 비교, FPCA 계수 역변환에 재사용)
def run_variant(variant: str, extra_cols: list[str], df: pd.DataFrame,
                 externals_meta: dict[str, pd.DataFrame],
                 balance: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    terms = ["intercept", "sex_M", "age", "height", "weight"] + extra_cols
    x, scaler = clinical_matrix(df, extra_cols)

    externals: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for cohort, df_ext in externals_meta.items():
        x_ext, _ = clinical_matrix(df_ext, extra_cols, scaler=scaler)
        externals[cohort] = (x_ext, df_ext)

    # save_roc_individual.py MODELS 키(aec_fpca_ratio 등)와 맞춘 시트명
    summary, coef_df = run_and_save(f"aec_{variant}", x, df, externals, terms=terms, balance=balance)
    return summary, coef_df, scaler


# fpca_only 변형의 질병별 PC 계수(표준화 스케일)를 128-slice 곡선 축으로 역변환. PCA는 선형변환이라
# score_k=(curve-pca.mean_).component_k이고 로지스틱은 표준화 score 위에서 logit=...+beta_k*(score_k-
# score_mean_k)/score_std_k이므로 대입하면 logit 기여분 = curve.[sum_k (beta_k/score_std_k)*component_k]+const.
# raw curve의 slice별 표준편차를 곱해 "그 slice가 +1SD 증가할 때 logit 변화량"으로 정규화(서로 다른 slice를
# 한 그래프에서 비교 가능하게)
def recover_curve_coefficients(coef_df: pd.DataFrame, pca: PCA, scaler: StandardScaler,
                                curve: np.ndarray) -> pd.DataFrame:
    cols = CLINICAL_BASE_COLS + FPCA_COLS  # fpca_only 변형에서 scaler를 fit할 때 쓴 컬럼 순서
    curve_std = curve.std(axis=0, ddof=1)
    rows = []
    for disease in DISEASES:
        d = coef_df[coef_df["disease"] == disease].set_index("term")
        w_curve = np.zeros(N_SLICES)
        for i, col in enumerate(FPCA_COLS):
            beta = d.loc[col, "coefficient"]
            score_std = scaler.scale_[cols.index(col)]
            w_curve += (beta / score_std) * pca.components_[i]
        w_curve_per_1sd = w_curve * curve_std
        for slice_i in range(1, N_SLICES + 1):
            rows.append({"disease": disease, "slice": slice_i,
                         "logit_per_1sd_raw_aec": w_curve_per_1sd[slice_i - 1]})
    return pd.DataFrame(rows)


# 질병별 recovered curve(logit 기여도 vs slice)를 한 그래프에 겹쳐 그려 저장
def save_recovered_curve_plot(recovered: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.16))  # 261002.pptx 그림 슬라이드 box(11176000x3806168 EMU)와 비율 일치
    for disease in DISEASES:
        d = recovered[recovered["disease"] == disease]
        ax.plot(d["slice"], d["logit_per_1sd_raw_aec"], label=disease, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("AEC slice index (liver -> pubis)")
    ax.set_ylabel("logit change per +1 SD raw AEC (fpca_only model)")
    ax.set_title("FPCA 계수 역변환: slice별 logit 기여도")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data_with_aec(DATA_XLSX)
    df, pca = add_aec_features(df)

    externals_meta: dict[str, pd.DataFrame] = {}
    for cohort, path in EXTERNAL_COHORTS.items():
        df_ext = load_data_with_aec(path)
        df_ext, _ = add_aec_features(df_ext, pca=pca)
        externals_meta[cohort] = df_ext

    summaries, coefs, scalers = {}, {}, {}
    for variant, extra_cols in VARIANTS.items():
        print(f"\n=== variant: {variant} (extra_cols={extra_cols}) ===")
        summaries[variant], coefs[variant], scalers[variant] = run_variant(
            variant, extra_cols, df, externals_meta)
        run_variant(variant, extra_cols, df, externals_meta, balance=False)  # 원본 유병률 결과도 함께 저장

    # 변형간 AUC 비교(disease x cohort별로 fpca_ratio vs fpca_only 나란히)
    compare = pd.concat(
        [s[["disease", "cohort", "auc"]].assign(variant=v) for v, s in summaries.items()], ignore_index=True)
    compare_wide = compare.pivot_table(index=["disease", "cohort"], columns="variant", values="auc").reset_index()
    format_floats(compare_wide).to_csv(OUTPUT_DIR / "variant_auc_compare.csv", index=False)
    print("\n=== variant AUC 비교 ===")
    print(format_floats(compare_wide).to_string(index=False))

    # FPCA 계수 slice 역변환(uplow_ratio와 안 얽힌 fpca_only 변형의 계수/scaler 사용)
    recovered = recover_curve_coefficients(coefs["fpca_only"], pca, scalers["fpca_only"], df[AEC_COLS].to_numpy(dtype=float))
    recovered.to_csv(OUTPUT_DIR / "fpca_recovered_curve_coefficients.csv", index=False)
    save_recovered_curve_plot(recovered, OUTPUT_DIR / "fpca_recovered_curve.png")
    print(f"\nSaved variant_auc_compare.csv, fpca_recovered_curve_coefficients.csv, "
          f"fpca_recovered_curve.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
