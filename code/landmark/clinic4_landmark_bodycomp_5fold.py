from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# data/{gangnam,sinchon}_landmark_filtered.xlsx의 landmark별 체성분 단면적(VAT/SAT/NAMA/LAMA/IMATA, cm2)을
# clinic4(성별/나이/신장/체중)에 스칼라 feature로 더했을 때 AUC가 baseline 대비 달라지는지 5-fold CV(+sinchon 외부검증)로 비교.
# 같은 landmark 필터 코호트 안에서 baseline과 비교하므로 환자 구성 차이는 통제된다. 라벨(HTN/DM/CKD)은 patient_metadata 시트에 있음.
# 5-fold/외부검증 로직은 clinic4_5fold_cv.run_disease_5fold를 그대로 재사용.

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clinic4_5fold_cv import run_disease_5fold
from clinic4_logistic_regression import CLINIC4_DIR, CLINICAL_COLS, DATA_DIR, DISEASES, f3, format_floats, save_sheet

sys.stdout.reconfigure(encoding="utf-8")

TISSUES = ["VAT", "SAT", "NAMA", "LAMA", "IMATA"]
ALL_LANDMARKS = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center",
                 "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
LANDMARK_SETS = {"L3": ["L3_center"], "L1_L3_L5": ["L1_center", "L3_center", "L5_center"], "all": ALL_LANDMARKS}
TISSUE_SETS = {**{t: [t] for t in TISSUES}, "VAT+SAT": ["VAT", "SAT"], "muscle(NAMA+LAMA+IMATA)": ["NAMA", "LAMA", "IMATA"],
               "all5": TISSUES}
REF = "L3_center"  # landmark 간 비율의 기준(분모)
MUSCLE = ["NAMA", "LAMA", "IMATA"]


# 모델명 -> feature 컬럼 목록. 수준(단면적) 모델은 "조직|landmark", 비율 모델은 "ratio:..."
def feature_sets() -> dict[str, list[str]]:
    sets = {"baseline": []}
    for l, lms in LANDMARK_SETS.items():
        for t, ts in TISSUE_SETS.items():
            sets[f"{t}|{l}"] = [f"{a}__{x}" for a in lms for x in ts]
        sets[f"ratio:VAT/SAT|{l}"] = [f"{a}__VSR" for a in lms]  # landmark 안에서 조직 간 비율
        lms_other = [a for a in lms if a != REF]
        if lms_other:  # landmark 간 비율(같은 조직의 landmark/L3). L3 단독은 분모뿐이라 해당 없음
            sets[f"ratio:all5/L3|{l}"] = [f"{a}__{x}/L3" for a in lms_other for x in TISSUES]
            sets[f"ratio:VAT,SAT/L3|{l}"] = [f"{a}__{x}/L3" for a in lms_other for x in ("VAT", "SAT")]
    sets["ratio:VAT/SAT@L3 + VAT,SAT/L3@all"] = ["L3_center__VSR"] + [f"{a}__{x}/L3" for a in ALL_LANDMARKS if a != REF
                                                                       for x in ("VAT", "SAT")]
    return sets


# 비율은 SAT 등이 0일 수 있어 log((a+1)/(b+1))로 안정화
def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    new = {}
    for a in ALL_LANDMARKS:
        new[f"{a}__VSR"] = np.log((df[f"{a}__VAT"] + 1) / (df[f"{a}__SAT"] + 1))
        if a != REF:
            for x in TISSUES:
                new[f"{a}__{x}/L3"] = np.log((df[f"{a}__{x}"] + 1) / (df[f"{REF}__{x}"] + 1))
    return pd.concat([df, pd.DataFrame(new)], axis=1)


# landmark xlsx의 patient_metadata + body_composition(wide) + final_dataset의 HTN/DM/CKD를 PatientID로 병합(결측 환자 제외)
def load_cohort(cohort: str) -> pd.DataFrame:
    path = DATA_DIR / f"{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    bc = pd.read_excel(path, sheet_name="body_composition").pivot(index="PatientID", columns="anchor",
                                                                   values=[f"{t}_sum_cm2" for t in TISSUES])
    bc.columns = [f"{a}__{v.split('_')[0]}" for v, a in bc.columns]
    df = meta.merge(bc.reset_index(), on="PatientID")
    df = df.dropna(subset=CLINICAL_COLS + list(bc.columns) + DISEASES).reset_index(drop=True)  # 라벨 없는 환자 제외
    print(f"[{cohort}] landmark 환자 {len(meta)} -> 라벨/결측 제외 후 {len(df)}명 "
          f"({', '.join(f'{d}={int(df[d].sum())}' for d in DISEASES)})")
    return add_ratios(df)


def matrix(df: pd.DataFrame, cols: list[str], scaler: StandardScaler | None = None):
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_COLS + cols].to_numpy(float)
    scaler = scaler or StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


def main() -> None:
    df, df_ext = load_cohort("gangnam"), load_cohort("sinchon")
    rows, preds = [], []
    for name, cols in feature_sets().items():
        x, scaler = matrix(df, cols)
        x_ext, _ = matrix(df_ext, cols, scaler)
        for disease in DISEASES:
            row, pred = run_disease_5fold(disease, name, x, df, x_ext, df_ext)
            rows.append(row)
            preds.append(pred)
    summary = pd.DataFrame(rows)
    save_sheet(pd.concat(preds, ignore_index=True), "landmark/landmark_bodycomp_5fold.xlsx", "predictions")
    save_sheet(format_floats(summary), "landmark/landmark_bodycomp_5fold.xlsx", "summary")

    # baseline 대비 AUC 차이(internal=5-fold OOF pooled, external=sinchon)
    cols = ["internal_pooled_auc", "external_auc"]
    piv = summary.pivot(index="model", columns="disease", values=cols)
    base = piv.loc["baseline"]
    print("baseline:", ", ".join(f"{m[:3]}-{d}={f3(base[(m, d)])}" for m in cols for d in DISEASES))
    delta = piv.sub(base, axis=1).drop(index="baseline")
    delta.columns = [f"{'int' if m.startswith('int') else 'ext'}_{d}" for m, d in delta.columns]
    order = [f"int_{d}" for d in DISEASES] + [f"ext_{d}" for d in DISEASES]
    print("\ndelta vs baseline\n" + delta.loc[list(feature_sets())[1:], order].map(f3).to_string())


if __name__ == "__main__":
    main()
