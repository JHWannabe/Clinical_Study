from __future__ import annotations

# outputs/table/table1_baseline_characteristics.png(Internal vs External 코호트 비교, 성별 층화 아님)를
# 만드는 스크립트. 원래 code/table1_baseline_characteristics.py에 있던 로직이었으나 그 파일이
# code/질병예측/table1_baseline_characteristics.py로 이름이 옮겨간 뒤 그 자리에서 성별 층화 버전(코호트별
# Men vs Women 비교, _internal/_external 접미사)으로 재설계되면서 Internal-vs-External 비교 로직 자체는
# 유실되고 결과물(outputs/table/table1_baseline_characteristics.{csv,png,xlsx})만 예전 버전 그대로
# 남아 있었다. 사용자 요청(2026-08-24: "outputs\table\table1_baseline_characteristics.png를 보강해",
# 레퍼런스 논문 Table 1 스타일 참고)에 따라 이 로직을 별도 파일로 복원하고, 레퍼런스 스타일(Anthropometry/
# Comorbidities/Imaging 같은 section 헤더 행, 방법론 각주)을 반영해 재구성했다. data 경로/AGE_CUTOFF는
# code/질병예측/table1_baseline_characteristics.py와 동일하게 맞춰(원본 xlsx, 연령<20 제외, 스캐너 제한
# 없음 -> internal 1,088명 / external 925명) 현재 코호트 정의와 어긋나지 않게 했다.
#
# 2026-08-27: table1_baseline_characteristics.py와 데이터 로딩/행 구성/표 이미지 렌더링 로직이 거의 그대로
# 중복되어(비교축만 Internal/External vs Men/Women) table1_common.py로 공통 로직을 뽑아냈다(사용자 요청).

import pandas as pd

from table1_common import (INTERNAL_XLSX, EXTERNAL_XLSX, OUTPUT_DIR, load_cohort, section_row, continuous_row,
                            range_row, categorical_rows, save_table)


# 레퍼런스 논문 Table 1(Anthropometry/Impedance analysis/CT-derived measures 같은 section 헤더로 변수를
# 구역화)을 참고해, 이 표도 Demographics/Anthropometry/Comorbidities/Imaging 4개 section으로 나눴다
def build_table(meta_int: pd.DataFrame, meta_ext: pd.DataFrame) -> pd.DataFrame:
    rows = [
        section_row("Demographics", "Internal", "External"),
        continuous_row("Age, years, mean ± SD", meta_int["PatientAge"], meta_ext["PatientAge"],
                        "Internal", "External"),
        range_row("Age, years, range", meta_int["PatientAge"], meta_ext["PatientAge"], "Internal", "External"),
        *categorical_rows("Sex", meta_int["PatientSex"].str.upper(), meta_ext["PatientSex"].str.upper(),
                           ["F", "M"], True, "Internal", "External"),

        section_row("Anthropometry", "Internal", "External"),
        continuous_row("Height, cm, mean ± SD", meta_int["Height"], meta_ext["Height"], "Internal", "External"),
        continuous_row("Weight, kg, mean ± SD", meta_int["Weight"], meta_ext["Weight"], "Internal", "External"),
        continuous_row("BMI, kg/m², mean ± SD", meta_int["BMI"], meta_ext["BMI"], "Internal", "External"),

        section_row("Comorbidities", "Internal", "External"),
        *categorical_rows("Hypertension", meta_int["HTN"], meta_ext["HTN"], [1, 0], False, "Internal", "External"),
        *categorical_rows("Diabetes mellitus", meta_int["DM"], meta_ext["DM"], [1, 0], False,
                           "Internal", "External"),
        *categorical_rows("Chronic kidney disease", meta_int["CKD"], meta_ext["CKD"], [1, 0], False,
                           "Internal", "External"),

        section_row("Imaging", "Internal", "External"),
        *categorical_rows("CT scanner vendor", meta_int["Vendor"], meta_ext["Vendor"],
                           ["Siemens", "GE", "Philips", "Canon"], True, "Internal", "External"),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    table = build_table(meta_int, meta_ext)
    print(table[["Characteristic", "Internal", "External", "p-value"]].to_string(index=False))
    save_table(table, "Internal", "External", out_stem="table1_baseline_characteristics", output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
