from __future__ import annotations

# clinic4(baseline: 성별/나이/신장/체중)에 AEC 없이 체성분 128구간 곡선(VAT/SAT/LAMA/NAMA/IMATA,
# liver->pubis)만 추가한 모델군. aec_vat/aec_lama/aec_nama/aec_bodycomp와 동일 구조에서 AEC curve만
# 뺐다 - AEC의 기여도를 체성분만의 기여도와 분리해서 보기 위함. curve별 단일 모델(vat/sat/lama/nama/imata)
# 5개 + 전부 합친 combined 모델(bodycomp) 1개 + VAT/SAT 비율 계열 3개(vsr/vat_sat/vat_sat_vsr)를
# 한 스크립트에서 순서대로 생성한다. FPCA 성분 수는
# clinic4_aec_bodycomp_logistic.py BODYCOMP_CURVES와 동일(Kneedle/elbow로 gangnam에서 재확인한 값).
# TAMA는 LAMA+NAMA와 완전 공선성이라 같은 이유로 뺐다(clinic4_aec_bodycomp_logistic.py 참고).
# split/grid search/외부검증/출력은 clinic4_logistic_regression.py의 run_disease를 그대로 재사용.

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

CLINIC4_DIR = PROJECT_ROOT / "outputs" / "clinic4"
N_SLICES = 128
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]
# name -> (sheet, prefix, FPCA n). clinic4_aec_bodycomp_logistic.py BODYCOMP_CURVES와 동일 값
CURVES: dict[str, tuple[str, str, int]] = {
    "vat": ("VFA_128", "VFA", 2), "sat": ("SFA_128", "SFA", 2),
    "lama": ("LAMA_128", "LAMA", 2), "nama": ("NAMA_128", "NAMA", 2), "imata": ("IMATA_128", "IMATA", 3),
}
VSR_COL = "vat_sat_ratio"  # 임상 표준 visceral-to-subcutaneous ratio = 128구간 VAT 합 / SAT 합
VSR_CURVES = ["vat", "sat"]  # VSR 계산에 필요한 curve(모델이 FPCA로 안 쓰더라도 merge는 해야 함)
# 슬라이스별 VAT_i/SAT_i 곡선은 쓰지 않는다: SAT에 정확히 0인 슬라이스가 있어(gangnam 38, sinchon 33,
# new10000 368셀) 비율이 234~905까지 튀어 FPCA가 극단 슬라이스 몇 개에 지배됨. 곡선 전체 합의 비율은
# 분모가 0이 되지 않아(실측 VSR 범위 0.05~5.9) 안정적 - aec_uplow_ratio가 곡선 평균의 스칼라 비율인 것과 동일한 방식
# 출력 모델명 -> (FPCA로 넣을 curve 목록, 스칼라 변수 목록)
MODELS: dict[str, tuple[list[str], list[str]]] = (
    {name: ([name], []) for name in CURVES}
    | {"bodycomp": (list(CURVES), []),
       "vsr": ([], [VSR_COL]),  # clinic4 + 비율만
       "vat_sat": (["vat", "sat"], []),  # 비율이 두 곡선 대비 뭘 더 주는지 판단할 비교군
       "vat_sat_vsr": (["vat", "sat"], [VSR_COL])}
)


def curve_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_SLICES + 1)]


# baseline의 load_data(clinic4 결측 제외)에 지정된 체성분 128구간 곡선들을 PatientID로 병합
# (곡선 하나라도 없는 환자는 inner join으로 자동 제외)
def load_data_with_curves(path: Path, curve_names: list[str], scalars: list[str] = []) -> pd.DataFrame:
    df = load_data(path)
    n0 = len(df)
    needed = list(dict.fromkeys(curve_names + (VSR_CURVES if VSR_COL in scalars else [])))
    for name in needed:
        sheet, prefix, _ = CURVES[name]
        curve = pd.read_excel(path, sheet_name=sheet)
        df = df.merge(curve[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
    df = df.reset_index(drop=True)
    if len(df) < n0:
        print(f"[{path.stem}] 체성분 곡선 없는 환자 제외: {n0 - len(df)}/{n0}명")
    return df


# 체성분 곡선(전처리 없이 원본 그대로)을 각각 FPCA로 압축해 df에 새 컬럼으로 붙인다. PCA는 gangnam에서만
# fit하고 외부 코호트에는 frozen으로 넘겨 transform만 함(누수 방지)
def add_curve_features(df: pd.DataFrame, curve_names: list[str], scalars: list[str] = [],
                        pcas: dict[str, PCA] | None = None) -> tuple[pd.DataFrame, dict[str, PCA]]:
    df = df.copy()
    pcas = dict(pcas) if pcas else {}
    for name in curve_names:
        _, prefix, n = CURVES[name]
        curve = df[curve_cols(prefix)].to_numpy(dtype=float)
        if name not in pcas:
            pcas[name] = PCA(n_components=n, random_state=SEED).fit(curve)
        for i, score in enumerate(pcas[name].transform(curve).T, 1):
            df[f"{name}_fpca_pc{i}"] = score
    if VSR_COL in scalars:
        vat = df[curve_cols(CURVES["vat"][1])].to_numpy(dtype=float).sum(axis=1)
        sat = df[curve_cols(CURVES["sat"][1])].to_numpy(dtype=float).sum(axis=1)
        df[VSR_COL] = vat / sat
    return df, pcas


# 성별(M=1/F=0) + 표준화된 나이/신장/체중/extra_cols로 입력 행렬 구성. scaler는 gangnam에서 fit해
# 외부 코호트에는 frozen으로 transform만 함
def clinical_matrix(df: pd.DataFrame, extra_cols: list[str],
                     scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_BASE_COLS + extra_cols].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


# 지정된 curve 조합 하나로 baseline+체성분 모델 전체 파이프라인(학습/외부검증/저장)을 실행. 모델별 하위
# 폴더는 만들지 않고(변형이 많아 빈 폴더만 늘어남) clinic4_aec_logistic_regression.py의 variant 저장
# 방식처럼 roc_curve를 CLINIC4_DIR에 모델명을 붙인 파일명으로 평평하게 저장한다.
def run_model(model_name: str, curve_names: list[str], scalars: list[str] = []) -> None:
    extra_cols = ([f"{name}_fpca_pc{i}" for name in curve_names for i in range(1, CURVES[name][2] + 1)]
                  + scalars)
    terms = ["intercept", "sex_M", "age", "height", "weight"] + extra_cols

    df, pcas = add_curve_features(load_data_with_curves(DATA_XLSX, curve_names, scalars), curve_names, scalars)
    x, scaler = clinical_matrix(df, extra_cols)

    externals: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for cohort, path in EXTERNAL_COHORTS.items():
        df_ext, _ = add_curve_features(load_data_with_curves(path, curve_names, scalars), curve_names,
                                       scalars, pcas=pcas)
        x_ext, _ = clinical_matrix(df_ext, extra_cols, scaler=scaler)
        externals[cohort] = (x_ext, df_ext)

    for balance in (True, False):  # 1:1 언더샘플링 결과와 원본 유병률(_unbalanced) 결과를 둘 다 저장
        run_and_save(model_name, x, df, externals, terms=terms, balance=balance)


def main() -> None:
    for model_name, (curve_names, scalars) in MODELS.items():
        print(f"\n=== {model_name} ===")
        run_model(model_name, curve_names, scalars)


if __name__ == "__main__":
    main()
