from __future__ import annotations

# metadata_cleaned(신장/체중 검증 완료 10,578명)와 aec_128_contrast(조영제 유 CT 시리즈 15,442행,
# 고유 12,544명)의 patientID 교집합을 사용자가 확인(9,914명, 2026-08-19). aec_128_contrast 쪽 행을
# 그 교집합으로 필터링해 새 시트에 저장한다(한 환자당 조영 시리즈가 여러 개면 해당 행 모두 유지).

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AEC_XLSX = PROJECT_ROOT / "data" / "aec_cropped.xlsx"

OUTPUT_SHEET = "aec_128_contrast_metaclean"


def main() -> None:
    xls = pd.ExcelFile(AEC_XLSX)
    meta_ids = set(pd.read_excel(xls, sheet_name="metadata_cleaned", usecols=["patientID"])["patientID"].astype(int))
    contrast = pd.read_excel(xls, sheet_name="aec_128_contrast")

    mask = contrast["PatientID"].astype(int).isin(meta_ids)
    result = contrast.loc[mask].reset_index(drop=True)

    print(f"aec_128_contrast {len(contrast)}행(고유 {contrast['PatientID'].nunique()}명) -> "
          f"metadata_cleaned 교집합 필터링 후 {len(result)}행(고유 {result['PatientID'].nunique()}명)")

    with pd.ExcelWriter(AEC_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        result.to_excel(writer, sheet_name=OUTPUT_SHEET, index=False)

    print(f"저장 완료: {AEC_XLSX} -> 시트 '{OUTPUT_SHEET}' ({len(result)}행 x {len(result.columns)}열)")


if __name__ == "__main__":
    main()
