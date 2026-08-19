from __future__ import annotations

# aec_cropped.xlsx의 aec_cropped/aec_128 시트에는 조영제 유무를 나타내는 명시적 컬럼이 없고,
# series_desc(예: "Pre Contrast 5.0 B30f", "With Contrast 3.0 B40f", "Portal 5.0 B30f") 문자열에
# CT 스캔 phase 명칭이 관례적으로 들어있을 뿐이다. CT 조영촬영 표준 용어상 Pre/Non/W/O(Without) =
# 조영 전(무조영), With/Post/Arterial/Portal/Delay(ed) = 조영 후(조영제 사용) phase이므로 이를
# 키워드로 판별한다. 두 시트 모두 동일한 PatientID+series_desc 조합(22867행)을 공유하지만 aec_1..N
# 컬럼 개수가 다르므로(cropped=가변길이, 128=고정 128포인트) 각각 별도 시트로 필터링해 저장한다.
# 사용자 확인 2026-08-19: "조영제 유인 데이터만 새로운 시트에 저장".

import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AEC_XLSX = PROJECT_ROOT / "data" / "aec_cropped.xlsx"

SOURCE_SHEETS = {
    "aec_cropped": "aec_cropped_contrast",
    "aec_128": "aec_128_contrast",
}

# 조영 전(precontrast)을 나타내는 키워드가 있으면 무조건 무조영으로 판정(다른 키워드보다 우선)
NO_CONTRAST_MARKERS = ("w/o", "without", "non contrast")


# series_desc 문자열 하나를 CT phase 명명 관례에 따라 'contrast'/'no_contrast'/'unknown'으로 분류
def classify_series_desc(desc: object) -> str:
    text = re.sub(r"\s+", " ", str(desc).strip().lower())
    if text.startswith("pre") or "(pre)" in text or any(marker in text for marker in NO_CONTRAST_MARKERS):
        return "no_contrast"
    if any(k in text for k in ("with", "post", "arterial", "artery", "portal", "port", "delay", "contrast")) or re.search(r"\bce\b", text):
        return "contrast"
    return "unknown"


def main() -> None:
    xls = pd.ExcelFile(AEC_XLSX)
    outputs: dict[str, pd.DataFrame] = {}

    for src_sheet, dst_sheet in SOURCE_SHEETS.items():
        df = pd.read_excel(xls, sheet_name=src_sheet)
        label = df["series_desc"].map(classify_series_desc)

        counts = label.value_counts()
        print(f"[{src_sheet}] 분류 결과: 조영제 유={counts.get('contrast', 0)}행, "
              f"무조영={counts.get('no_contrast', 0)}행, 판정불가={counts.get('unknown', 0)}행")
        if counts.get("unknown", 0):
            unknown_examples = df.loc[label == "unknown", "series_desc"].value_counts()
            print(f"  판정불가 series_desc 목록(제외됨): {dict(unknown_examples)}")

        outputs[dst_sheet] = df.loc[label == "contrast"].reset_index(drop=True)

    with pd.ExcelWriter(AEC_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for dst_sheet, result in outputs.items():
            result.to_excel(writer, sheet_name=dst_sheet, index=False)
            print(f"저장 완료: {AEC_XLSX} -> 시트 '{dst_sheet}' ({len(result)}행 x {len(result.columns)}열)")


if __name__ == "__main__":
    main()
