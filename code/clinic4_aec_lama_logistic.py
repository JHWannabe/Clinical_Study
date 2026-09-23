from __future__ import annotations

# clinic4_logistic_regression.py의 clinic4(성별/나이/신장/체중) baseline에 AEC/LAMA 128구간 곡선(liver->pubis,
# aec_128/LAMA_128 시트)을 추가 input으로 넣은 확장 모델. AEC는 FPCA(n=3), LAMA는 FPCA(n=2)로 넣는다(둘 다
# Kneedle/elbow로 재확인한 CURVE_FPCA_N_ELBOW 값). clinic4_aec_vat_logistic.py와 동일 구조에서 VAT curve만
# LAMA(하지 저감쇠 근육, low attenuation muscle area)로 바꾼 버전. PCA/scaler는 gangnam 전체에서 한 번만
# fit하고, split/grid search/외부검증/출력은 전부 clinic4_logistic_regression.py의 run_disease(terms
# 파라미터로 확장된 버전)를 그대로 재사용한다.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clinic4_logistic_regression import (
    DATA_XLSX, EXTERNAL_COHORTS, PROJECT_ROOT, SEED, load_data, run_and_save,
)

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinic4" / "aec_lama"
N_SLICES = 128
CURVE_SHEETS = {"AEC": ("aec_128", "aec"), "LAMA": ("LAMA_128", "LAMA")}
AEC_FPCA_N = 3  # CURVE_FPCA_N_ELBOW["AEC"](Kneedle/elbow로 재확인)
LAMA_FPCA_N = 2  # CURVE_FPCA_N_ELBOW["LAMA"](Kneedle/elbow로 재확인)
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
EXTRA_COLS = ([f"aec_fpca_pc{i}" for i in range(1, AEC_FPCA_N + 1)]
              + [f"lama_fpca_pc{i}" for i in range(1, LAMA_FPCA_N + 1)])


def curve_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_SLICES + 1)]


# baseline의 load_data(clinic4 결측 제외)에 AEC/LAMA 128구간 곡선을 PatientID로 병합(곡선 없는 환자는 자동 제외)
def load_data_with_curves(path: Path) -> pd.DataFrame:
    df = load_data(path)
    n0 = len(df)
    for sheet, prefix in CURVE_SHEETS.values():
        curve = pd.read_excel(path, sheet_name=sheet)
        df = df.merge(curve[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
    df = df.reset_index(drop=True)
    if len(df) < n0:
        print(f"[{path.stem}] AEC/LAMA 곡선 없는 환자 제외: {n0 - len(df)}/{n0}명")
    return df


# AEC/LAMA 128구간 곡선(전처리 없이 원본 그대로)에서 두 곡선의 FPCA 성분을 뽑아 df에 새 컬럼으로 붙인다.
# PCA는 gangnam에서만 fit하고 외부 코호트에는 frozen으로 넘겨 transform만 함(누수 방지)
def add_curve_features(df: pd.DataFrame, aec_pca: PCA | None = None,
                        lama_pca: PCA | None = None) -> tuple[pd.DataFrame, PCA, PCA]:
    df = df.copy()
    aec_curve = df[curve_cols(CURVE_SHEETS["AEC"][1])].to_numpy(dtype=float)
    if aec_pca is None:
        aec_pca = PCA(n_components=AEC_FPCA_N, random_state=SEED).fit(aec_curve)
    for i, score in enumerate(aec_pca.transform(aec_curve).T, 1):
        df[f"aec_fpca_pc{i}"] = score

    lama_curve = df[curve_cols(CURVE_SHEETS["LAMA"][1])].to_numpy(dtype=float)
    if lama_pca is None:
        lama_pca = PCA(n_components=LAMA_FPCA_N, random_state=SEED).fit(lama_curve)
    for i, score in enumerate(lama_pca.transform(lama_curve).T, 1):
        df[f"lama_fpca_pc{i}"] = score
    return df, aec_pca, lama_pca


# 성별(M=1/F=0) + 표준화된 나이/신장/체중/extra_cols로 입력 행렬 구성. scaler는 gangnam에서 fit해
# 외부 코호트에는 frozen으로 transform만 함(clinic4_aec_vat_logistic.py clinical_matrix와 동일)
def clinical_matrix(df: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_BASE_COLS + EXTRA_COLS].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


def main() -> None:
    terms = ["intercept", "sex_M", "age", "height", "weight"] + EXTRA_COLS

    df, aec_pca, lama_pca = add_curve_features(load_data_with_curves(DATA_XLSX))
    x, scaler = clinical_matrix(df)

    externals: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for cohort, path in EXTERNAL_COHORTS.items():
        df_ext, _, _ = add_curve_features(load_data_with_curves(path), aec_pca=aec_pca, lama_pca=lama_pca)
        x_ext, _ = clinical_matrix(df_ext, scaler=scaler)
        externals[cohort] = (x_ext, df_ext)

    for balance in (True, False):  # 1:1 언더샘플링 결과와 원본 유병률(_unbalanced) 결과를 둘 다 저장
        run_and_save(OUTPUT_DIR.name, x, df, externals, terms=terms, balance=balance)


if __name__ == "__main__":
    main()
