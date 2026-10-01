from __future__ import annotations

# baseline(성별/나이/신장/체중)에 AEC FPCA + 체성분 곡선 1개(vat/sat/lama/nama/imata)의 FPCA를 더한 확장 모델.
# 곡선별 FPCA n/시트는 clinic4_aec_bodycomp_logistic.BODYCOMP_CURVES(elbow 확정값)를 그대로 쓰고, PCA/scaler는
# gangnam에서만 fit해 외부 코호트에는 frozen으로 적용한다. split/grid search/외부검증/출력은 run_and_save 재사용.
#
#   python clinic4_aec_single_curve_logistic.py vat lama   # 지정한 곡선만(생략하면 vat lama nama imata 전부)
# 결과는 outputs/clinic4/aec_<곡선>/ 에 저장(옛 clinic4_aec_{vat,lama,nama,imata}_logistic.py와 동일).

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clinic4_logistic_regression import DATA_XLSX, EXTERNAL_COHORTS, SEED, load_data, run_and_save
from clinic4_aec_bodycomp_logistic import (
    AEC_FPCA_N, AEC_PREFIX, AEC_SHEET, BODYCOMP_CURVES, CLINICAL_BASE_COLS, curve_cols,
)

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

DEFAULT_CURVES = ["vat", "lama", "nama", "imata"]


# baseline에 AEC + 해당 곡선 128구간을 PatientID로 병합(곡선 없는 환자는 inner join으로 제외)
def load_with_curves(path, curves: dict[str, tuple[str, str]]) -> pd.DataFrame:
    df = load_data(path)
    n0 = len(df)
    for sheet, prefix in curves.values():
        df = df.merge(pd.read_excel(path, sheet_name=sheet)[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
    if len(df) < n0:
        print(f"[{path.stem}] 곡선 없는 환자 제외: {n0 - len(df)}/{n0}명")
    return df.reset_index(drop=True)


def run_curve(name: str) -> None:
    sheet, prefix, n = BODYCOMP_CURVES[name]
    spec = {"aec": (AEC_SHEET, AEC_PREFIX, AEC_FPCA_N), name: (sheet, prefix, n)}
    extra = [f"{k}_fpca_pc{i}" for k, (_, _, kn) in spec.items() for i in range(1, kn + 1)]
    pcas: dict[str, PCA] = {}

    # PCA는 pcas가 비어있을 때(gangnam)만 fit, 이후 외부 코호트는 frozen transform(누수 방지)
    def features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for k, (_, p, kn) in spec.items():
            curve = df[curve_cols(p)].to_numpy(float)
            pcas.setdefault(k, PCA(n_components=kn, random_state=SEED).fit(curve))
            for i, score in enumerate(pcas[k].transform(curve).T, 1):
                df[f"{k}_fpca_pc{i}"] = score
        return df

    def matrix(df: pd.DataFrame, scaler: StandardScaler | None = None):
        sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
        rest = df[CLINICAL_BASE_COLS + extra].to_numpy(float)
        scaler = scaler or StandardScaler().fit(rest)
        return np.column_stack([sex_m, scaler.transform(rest)]), scaler

    loader = {k: (s, p) for k, (s, p, _) in spec.items()}
    df = features(load_with_curves(DATA_XLSX, loader))
    x, scaler = matrix(df)
    externals = {}
    for cohort, path in EXTERNAL_COHORTS.items():
        df_ext = features(load_with_curves(path, loader))
        externals[cohort] = (matrix(df_ext, scaler)[0], df_ext)

    terms = ["intercept", "sex_M", "age", "height", "weight"] + extra
    for balance in (True, False):  # 1:1 언더샘플링 결과와 원본 유병률(_unbalanced) 결과를 둘 다 저장
        run_and_save(f"aec_{name}", x, df, externals, terms=terms, balance=balance)


if __name__ == "__main__":
    for name in sys.argv[1:] or DEFAULT_CURVES:
        run_curve(name)
