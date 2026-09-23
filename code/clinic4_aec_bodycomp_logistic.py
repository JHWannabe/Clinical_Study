from __future__ import annotations

# clinic4_aec_vat_logistic.py(clinic4 + AEC FPCA(n=3) + VAT FPCA(n=2))에 나머지 체성분 128구간 곡선
# (SAT/TAMA/LAMA/NAMA/IMATA, liver->pubis)도 각각 FPCA로 추가해 VAT 하나만 넣었을 때보다 더 개선되는지
# 확인한다. 성분 수는 전부 Kneedle/elbow 기준으로 gangnam 전체 curve에서 재확인한 값(SAT=2, TAMA=2, LAMA=2,
# NAMA=2, IMATA=3 - code/01_disease_logistic/clinic4_bodycomp_compare.py CURVE_FPCA_N_ELBOW와 동일).
# AEC Upper/Lower ratio는 clinic4_aec_vat_logistic.py와 같은 이유(aec_fpca_pc2와 VIF>10 중복)로 뺐다.
# split/grid search/외부검증/출력은 clinic4_logistic_regression.py의 run_disease(terms 파라미터)를 재사용.

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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clinic4" / "aec_bodycomp"
N_SLICES = 128
AEC_SHEET, AEC_PREFIX, AEC_FPCA_N = "aec_128", "aec", 3
# name -> (sheet, prefix, FPCA n). n은 전부 Kneedle/elbow로 gangnam에서 재확인한 값(CURVE_FPCA_N_ELBOW와 동일).
# TAMA(total abdominal muscle area)는 정의상 LAMA+NAMA와 완전히 같아서(실측 max diff=0.01, 반올림 오차
# 수준) 뺐다 - 셋을 같이 넣으면 tama_fpca_pc1의 VIF가 6만대까지 치솟는 완전 다중공선성이 발생함(2026-09-22 확인)
BODYCOMP_CURVES: dict[str, tuple[str, str, int]] = {
    "vat": ("VFA_128", "VFA", 2), "sat": ("SFA_128", "SFA", 2),
    "lama": ("LAMA_128", "LAMA", 2), "nama": ("NAMA_128", "NAMA", 2), "imata": ("IMATA_128", "IMATA", 3),
}
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
EXTRA_COLS = ([f"aec_fpca_pc{i}" for i in range(1, AEC_FPCA_N + 1)]
              + [f"{name}_fpca_pc{i}" for name, (_, _, n) in BODYCOMP_CURVES.items() for i in range(1, n + 1)])


def curve_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_SLICES + 1)]


# baseline의 load_data(clinic4 결측 제외)에 AEC + 6개 체성분 128구간 곡선을 PatientID로 병합
# (곡선 하나라도 없는 환자는 inner join으로 자동 제외)
def load_data_with_curves(path: Path) -> pd.DataFrame:
    df = load_data(path)
    n0 = len(df)
    sheets = [(AEC_SHEET, AEC_PREFIX)] + [(sheet, prefix) for sheet, prefix, _ in BODYCOMP_CURVES.values()]
    for sheet, prefix in sheets:
        curve = pd.read_excel(path, sheet_name=sheet)
        df = df.merge(curve[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
    df = df.reset_index(drop=True)
    if len(df) < n0:
        print(f"[{path.stem}] AEC/체성분 곡선 없는 환자 제외: {n0 - len(df)}/{n0}명")
    return df


# AEC + 6개 체성분 곡선(전처리 없이 원본 그대로)을 각각 FPCA로 압축해 df에 새 컬럼으로 붙인다. PCA는
# gangnam에서만 fit하고 외부 코호트에는 frozen으로 넘겨 transform만 함(누수 방지)
def add_curve_features(df: pd.DataFrame, pcas: dict[str, PCA] | None = None) -> tuple[pd.DataFrame, dict[str, PCA]]:
    df = df.copy()
    pcas = dict(pcas) if pcas else {}
    curves = {"aec": (AEC_PREFIX, AEC_FPCA_N)} | {name: (prefix, n) for name, (_, prefix, n) in BODYCOMP_CURVES.items()}
    for name, (prefix, n) in curves.items():
        curve = df[curve_cols(prefix)].to_numpy(dtype=float)
        if name not in pcas:
            pcas[name] = PCA(n_components=n, random_state=SEED).fit(curve)
        for i, score in enumerate(pcas[name].transform(curve).T, 1):
            df[f"{name}_fpca_pc{i}"] = score
    return df, pcas


# 성별(M=1/F=0) + 표준화된 나이/신장/체중/extra_cols로 입력 행렬 구성. scaler는 gangnam에서 fit해
# 외부 코호트에는 frozen으로 transform만 함
def clinical_matrix(df: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_BASE_COLS + EXTRA_COLS].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


def main() -> None:
    terms = ["intercept", "sex_M", "age", "height", "weight"] + EXTRA_COLS

    df, pcas = add_curve_features(load_data_with_curves(DATA_XLSX))
    x, scaler = clinical_matrix(df)

    externals: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for cohort, path in EXTERNAL_COHORTS.items():
        df_ext, _ = add_curve_features(load_data_with_curves(path), pcas=pcas)
        x_ext, _ = clinical_matrix(df_ext, scaler=scaler)
        externals[cohort] = (x_ext, df_ext)

    for balance in (True, False):  # 1:1 언더샘플링 결과와 원본 유병률(_unbalanced) 결과를 둘 다 저장
        run_and_save(OUTPUT_DIR.name, x, df, externals, terms=terms, balance=balance)


if __name__ == "__main__":
    main()
