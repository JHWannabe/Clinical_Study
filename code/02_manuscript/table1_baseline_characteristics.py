from __future__ import annotations

# m&m 초안(docs/m&m 초안_한글.docx) Table 1(환자 기저 특성)을 만드는 스크립트. 2026-08-24 기준 code/질병예측와
# code/질병예측_v2 두 폴더를 하나로 통합하며 이 파일은 v2(스캐너/벤더 제한 해제 코호트) 내용으로 교체됐다.
#
# 2026-08-24(2차): 사용자가 제공한 참고 이미지(Table 1: Baseline Characteristics of the Participants by
# Prevalent Diabetes)의 구성을 따르도록 표 자체의 데이터 구성을 재설계했다(사용자 확인: "이미지처럼 구성해줘"
# -> "데이터 구성 자체" -> 성별/질병유무로도 층화, 질환 3개(HTN/DM/CKD)는 참고 이미지처럼 성별x질환 조합
# 컬럼을 만들지 않고 한 표 안에 각각 별도 행으로 n(%)+p-value만 추가, 참고 이미지의 Anthropometry/Impedance
# analysis/CT-derived measures 같은 구역(section) 헤더는 생략, Internal/External은 참고 이미지처럼 단일
# 코호트로 합치지 않고 기존처럼 코호트별로 별도 표 유지). 그 결과 비교축이 "Internal vs External"에서
# "Men vs Women"(코호트 내부)으로 바뀌었고, Internal/External 코호트마다 독립적인 표 이미지/csv/xlsx를
# 생성한다(참고 이미지의 "the Participants" 단일 코호트 구조를 그대로 따르지 않고 우리 연구의 2-코호트
# 구조를 유지). data/{gangnam,sinchon}_원본.xlsx에서 연령<20만 제외(스캐너 제한 없음, internal 1,088명 /
# external 925명)한 코호트를 사용하며, 원본 코호트에는 기존 4개(Siemens/GE/Philips) 외 Canon(Aquilion 계열)
# 스캐너도 포함되므로 VENDOR_PREFIXES에 Canon을 추가하고, GE 계열 중 매핑 누락되어 있던 Optima도 추가했다.
# 연속형은 Welch's t-test, 범주형은 chi-square test로 비교한다.
#
# 2026-08-27: table1_baseline_characteristics_cohort.py와 데이터 로딩/행 구성/표 이미지 렌더링 로직이 거의
# 그대로 중복되어(비교축만 Men/Women vs Internal/External) table1_common.py로 공통 로직을 뽑아냈다(사용자 요청).

import pandas as pd

from table1_common import (INTERNAL_XLSX, EXTERNAL_XLSX, OUTPUT_DIR, load_cohort, continuous_row, range_row,
                            categorical_rows, save_table)

COHORTS = [("internal", "Internal", INTERNAL_XLSX), ("external", "External", EXTERNAL_XLSX)]


# 코호트 1개(internal 또는 external)의 남성/여성 부분집합을 비교하는 표를 구성. 참고 이미지(Table 1:
# Baseline Characteristics of the Participants by Prevalent Diabetes)의 행 구성(연속형은 mean±SD,
# 이진 범주형은 n(%)+p-value 한 행)을 따르되, HTN/DM/CKD는 참고 이미지처럼 성별x질환 조합 컬럼을 만들지
# 않고 이 표 안에 각각 별도 행으로만 추가한다(사용자 확인 2026-08-24)
def build_table_by_sex(meta):
    men = meta[meta["PatientSex"].str.upper() == "M"]
    women = meta[meta["PatientSex"].str.upper() == "F"]
    vendor_labels = sorted(set(meta["Vendor"]), key=lambda v: ["Siemens", "GE", "Philips", "Canon"].index(v))
    rows = [
        continuous_row("Age, years, mean ± SD", men["PatientAge"], women["PatientAge"], "Men", "Women"),
        range_row("Age, years, range", men["PatientAge"], women["PatientAge"], "Men", "Women"),
        continuous_row("Height, cm, mean ± SD", men["Height"], women["Height"], "Men", "Women"),
        continuous_row("Weight, kg, mean ± SD", men["Weight"], women["Weight"], "Men", "Women"),
        continuous_row("BMI, kg/m², mean ± SD", men["BMI"], women["BMI"], "Men", "Women"),
        *categorical_rows("Hypertension", men["HTN"], women["HTN"], [1, 0], False, "Men", "Women"),
        *categorical_rows("Diabetes mellitus", men["DM"], women["DM"], [1, 0], False, "Men", "Women"),
        *categorical_rows("Chronic kidney disease", men["CKD"], women["CKD"], [1, 0], False, "Men", "Women"),
        *categorical_rows("CT scanner vendor", men["Vendor"], women["Vendor"], vendor_labels, True, "Men", "Women"),
    ]
    return pd.DataFrame(rows), len(men), len(women)


def main() -> None:
    for cohort_key, cohort_label, xlsx_path in COHORTS:
        meta = load_cohort(xlsx_path)
        table, n_men, n_women = build_table_by_sex(meta)
        print(f"=== {cohort_label} (n_men={n_men}, n_women={n_women}) ===")
        print(table[["Characteristic", "Men", "Women", "p-value"]].to_string(index=False))
        save_table(table, "Men", "Women", out_stem=f"table1_baseline_characteristics_{cohort_key}",
                   output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
