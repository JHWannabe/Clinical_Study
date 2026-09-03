from __future__ import annotations

# data/aec_cropped.xlsx(metadata 시트)의 신장/체중/BMI가 명백히 잘못 저장돼 있음을 발견
# (예: patientID=3315174 체중=0.65kg, BMI=0.21). 두 파일 모두 신규 추가본(git 미추적)이라 기존
# step 스크립트와 무관. patientID == 통합 문서.xlsx의 연구등록번호로 1:1 매칭됨을 사용자가 확인(2026-08-19).
#
# 신장/체중은 두 소스에서 가져온다: '1-2.키몸무게BMI_예진정보'(wide, 행 하나=측정 1회, 1차 소스),
# '1-1.키몸무게BMI_임상관찰'(long, 임상관찰코드별 개별 실측값, 2차 소스). 단순 median 집계만으로는
# 개별 이상치가 섞여 들어가는 문제를 발견해(예: 연구등록번호=3251106의 1-1 체중 중복 기록이 [68, 10.5]로
# 상충, median하면 39.25라는 실존하지 않는 값이 나옴) 다음 2단계로 검증한다.
#   1) 개별 측정값(행) 단위로 신장/체중 각각이 임상적으로 가능한 범위 밖이면 그 측정값 자체를 버린다.
#   2) 소스별로 (신장,체중) 쌍의 BMI가 가능한 범위(12~60) 밖이면 그 소스의 쌍 전체를 무효로 본다.
# 두 소스 모두 유효한 쌍을 내면 서로 비교(재검토)해 합의(체중 차 15kg 이하 & 신장 차 10cm 이하)하면
# 1차 소스를 채택하고, 상충하면(예: 연구등록번호=5975789 → 1-2:165cm/90kg vs 1-1:158cm/51.8kg)
# 어느 쪽이 옳은지 판단할 근거가 없으므로 그 행을 제거한다(사용자 요청 3번 "재검토 후에도 동일하면 제거"에
# 해당 — 상충이 재검토 후에도 해소되지 않는 경우로 취급).
#
# 질병 6종(당뇨병/고혈압/이상지질혈증/골다공증/심근경색/뇌졸중)은 metadata의 기존 _여부 컬럼을 신뢰하지
# 않고, 통합 문서.xlsx의 진단 시트 5-1~5-6에 해당 연구등록번호가 존재하는지로 재계산해
# "다시한번 확인"한다(사용자 요청 4번). 결과는 data/aec_cropped.xlsx에 'metadata_cleaned' 새 시트로
# 추가 저장한다(사용자 확인, 2026-08-19). 원본은 실행 전 data/aec_cropped_backup_before_cleaning.xlsx로
# 백업해둔 상태.

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글/±를 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

AEC_XLSX = DATA_DIR / "aec_cropped.xlsx"
INTEGRATED_XLSX = DATA_DIR / "통합 문서.xlsx"

# 개별 측정값(행) 단위로 걸러낼 성인 신장/체중 허용 범위
HEIGHT_MIN, HEIGHT_MAX = 100.0, 250.0
WEIGHT_MIN, WEIGHT_MAX = 20.0, 250.0
# 신장·체중 쌍의 BMI가 이 범위 밖이면 그 쌍(소스) 자체를 무효로 간주(극단적 오기록 필터링)
BMI_MIN, BMI_MAX = 12.0, 60.0
# 두 소스가 모두 유효한 쌍을 냈을 때, 이 이내 차이는 같은 측정으로 보고 1차 소스를 채택
WEIGHT_AGREE_TOL = 15.0
HEIGHT_AGREE_TOL = 10.0

# 질병 6종 -> 통합 문서.xlsx 내 해당 진단 시트명, 최종 컬럼명(metadata 기존 컬럼명과 동일하게 유지)
DISEASE_SHEETS = {
    "당뇨병_여부": "5-1. DM_f_u",
    "고혈압_여부": "5-2. HTN_f_u",
    "이상지질혈증_여부": "5-3. DL_f_u",
    "골다공증_여부": "5-4. 골다공증_f_u",
    "심근경색_여부": "5-5. 심근경색_f_u",
    "뇌졸중_여부": "5-6. 뇌졸중_f_u",
}

FINAL_COLUMNS = [
    "patientID",
    "성별",
    "나이",
    "신장",
    "체중",
    "BMI",
    *DISEASE_SHEETS.keys(),
]


# 문자열 측정값에서 선행 숫자만 추출(예: "66.5/구두" -> 66.5, "거절" -> NaN)
def parse_leading_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*(-?\d+\.?\d*)", str(value))
    return float(match.group(1)) if match else np.nan


# (신장,체중) 쌍 DataFrame에서 개별 범위를 벗어나거나 BMI가 비정상인 쌍을 제거하고 환자별 median으로 집계
def clean_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    valid = pairs[
        (pairs["신장"] >= HEIGHT_MIN) & (pairs["신장"] <= HEIGHT_MAX)
        & (pairs["체중"] >= WEIGHT_MIN) & (pairs["체중"] <= WEIGHT_MAX)
    ].copy()
    valid["bmi"] = valid["체중"] / ((valid["신장"] / 100.0) ** 2)
    valid = valid[(valid["bmi"] >= BMI_MIN) & (valid["bmi"] <= BMI_MAX)]
    return valid.groupby("연구등록번호")[["신장", "체중"]].median()


# '1-2.키몸무게BMI_예진정보' 시트: 한 행이 실제 측정 1회이므로 행 단위 쌍을 그대로 검증
def load_source_1_2(xls: pd.ExcelFile) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name="1-2. 키몸무게BMI_예진정보")
    return clean_pairs(df[["연구등록번호", "신장", "체중"]])


# '1-1.키몸무게BMI_임상관찰' 시트(임상관찰코드별 long format)를 환자별 신장/체중 쌍으로 만들어 검증
# 신장/체중이 서로 다른 행에 기록돼 있어 행 단위가 아니라 환자 단위로 먼저 median 매칭한 뒤 쌍 검증한다
def load_source_1_1(xls: pd.ExcelFile) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name="1-1.키몸무게BMI_임상관찰")
    df["측정값_num"] = df["측정값"].map(parse_leading_number)
    height_raw = df.loc[df["임상관찰코드(코드명)"] == "신체계측/신장(cm)", ["연구등록번호", "측정값_num"]]
    weight_raw = df.loc[df["임상관찰코드(코드명)"] == "신체계측/체중(kg)", ["연구등록번호", "측정값_num"]]
    # median을 내기 전에 개별 측정값 단위로 범위를 벗어난 기록부터 제거(예: 같은 환자의 체중 중복기록이
    # [68, 10.5]처럼 상충할 때 10.5를 먼저 걸러내지 않으면 median이 39.25라는 존재하지 않는 값이 됨)
    height = height_raw.loc[
        height_raw["측정값_num"].between(HEIGHT_MIN, HEIGHT_MAX)
    ].groupby("연구등록번호")["측정값_num"].median()
    weight = weight_raw.loc[
        weight_raw["측정값_num"].between(WEIGHT_MIN, WEIGHT_MAX)
    ].groupby("연구등록번호")["측정값_num"].median()
    pairs = pd.concat([height.rename("신장"), weight.rename("체중")], axis=1).dropna().reset_index()
    return clean_pairs(pairs)


# 1차(1-2)/2차(1-1) 소스를 환자 단위로 병합해 최종 신장/체중을 확정. 결측 사유(단일소스 없음/상충)를 함께 반환
def reconcile(src12: pd.DataFrame, src11: pd.DataFrame, patient_ids: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    both = src12.join(src11, lsuffix="_12", rsuffix="_11", how="inner")
    conflict_mask = (
        (both["체중_12"] - both["체중_11"]).abs() > WEIGHT_AGREE_TOL
    ) | (
        (both["신장_12"] - both["신장_11"]).abs() > HEIGHT_AGREE_TOL
    )
    conflict_ids = set(both.index[conflict_mask])

    height = pd.Series(index=patient_ids, dtype=float)
    weight = pd.Series(index=patient_ids, dtype=float)
    reason = pd.Series(index=patient_ids, dtype=object)

    for pid in patient_ids:
        if pid in conflict_ids:
            reason[pid] = "소스간_상충"
            continue
        if pid in src12.index:
            height[pid], weight[pid] = src12.loc[pid, "신장"], src12.loc[pid, "체중"]
        elif pid in src11.index:
            height[pid], weight[pid] = src11.loc[pid, "신장"], src11.loc[pid, "체중"]
        else:
            reason[pid] = "결측"
    return height, weight, reason


# 질병 시트 하나에서 연구등록번호 집합을 뽑음(해당 시트에 행이 있으면 유병으로 판정)
def load_disease_ids(xls: pd.ExcelFile, sheet_name: str) -> set[int]:
    df = pd.read_excel(xls, sheet_name=sheet_name, usecols=["연구등록번호"])
    return set(df["연구등록번호"].dropna().astype(int))


def main() -> None:
    meta = pd.read_excel(AEC_XLSX, sheet_name="metadata")
    meta = meta.set_index(meta["patientID"].astype(int), drop=False)

    integrated = pd.ExcelFile(INTEGRATED_XLSX)
    src12 = load_source_1_2(integrated)
    src11 = load_source_1_1(integrated)
    print(f"1-2 소스 유효 환자: {len(src12)}명, 1-1 소스 유효 환자: {len(src11)}명")

    height, weight, missing_reason = reconcile(src12, src11, meta.index)
    n_conflict = int((missing_reason == "소스간_상충").sum())
    n_missing = int((missing_reason == "결측").sum())
    print(f"신장/체중 확정 불가 사유: 소스간_상충 {n_conflict}명(체중 차>{WEIGHT_AGREE_TOL}kg 또는 신장 차>{HEIGHT_AGREE_TOL}cm, 재검토 후에도 상충), 결측 {n_missing}명(두 소스 모두 유효값 없음)")

    bmi = (weight / ((height / 100.0) ** 2)).round(2)

    result = pd.DataFrame(index=meta.index)
    result["patientID"] = meta["patientID"]
    result["성별"] = meta["성별"]
    result["나이"] = meta["나이"]
    result["신장"] = height.round(1)
    result["체중"] = weight.round(1)
    result["BMI"] = bmi

    # 질병 6종: 기존 metadata의 _여부 컬럼을 신뢰하지 않고 통합 문서.xlsx 진단 시트 존재 여부로 재계산
    for col, sheet_name in DISEASE_SHEETS.items():
        positive_ids = load_disease_ids(integrated, sheet_name)
        recomputed = meta["patientID"].astype(int).isin(positive_ids).astype(int)
        mismatch = int((recomputed.values != meta[col].fillna(0).astype(int).values).sum())
        print(f"[{col}] 재계산 결과 기존 metadata 값과 불일치: {mismatch}명 / 재계산 유병: {int(recomputed.sum())}명 (기존: {int(meta[col].sum())}명)")
        result[col] = recomputed.values

    n_before = len(result)
    result = result.dropna(subset=["신장", "체중"])
    n_after = len(result)
    print(f"신장/체중 확정 불가 행 제거: {n_before}명 -> {n_after}명 (제거 {n_before - n_after}명)")

    result = result[FINAL_COLUMNS].reset_index(drop=True)

    # 기존 aec_cropped.xlsx의 다른 시트(metadata/aec_cropped/aec_128)는 그대로 두고 새 시트만 추가/갱신
    with pd.ExcelWriter(AEC_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        result.to_excel(writer, sheet_name="metadata_cleaned", index=False)

    print(f"저장 완료: {AEC_XLSX} -> 시트 'metadata_cleaned' ({len(result)}행 x {len(result.columns)}열)")


if __name__ == "__main__":
    main()
