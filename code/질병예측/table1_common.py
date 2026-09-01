from __future__ import annotations

# table1_baseline_characteristics.py(코호트별 Men vs Women)와
# table1_baseline_characteristics_cohort.py(Internal vs External)가 데이터 로딩/행 구성/표 이미지 렌더링
# 로직을 거의 그대로 중복하고 있어(비교축만 다름) 공통 모듈로 분리했다(2026-08-27, 사용자 요청). 각 파일에
# 있던 개별 변경 이력 주석(스캐너 매핑 근거, cutoff 근거 등)은 이 파일 쪽으로 옮긴다.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949가 ±/–(en dash) 등을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "table"

INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_final_dataset.xlsx"
AGE_CUTOFF = 20

REQUIRED_COLS = ["PatientAge", "PatientSex", "Height", "Weight", "BMI", "HTN", "DM", "CKD", "Manufacturer"]

# 2026-08-27 sinchon_final_dataset.xlsx 재교체본에서 Height/Weight가 1/1, 10/10 등 placeholder성 값으로
# 손상되어 있음을 발견(BMI 10000/1000/2.08/100 등 비정상치)해 4명을 제외했었으나, 이후 파일이 다시
# 교체되며 3명(1858333/4036195/4371520)은 아예 사라졌고 남은 1명(2036751)은 Height=176.0/Weight=61.0/
# BMI=19.69로 완전히 정상값으로 복구된 것을 확인(2026-08-27 재확인) — 이 상수를 그대로 두면 이미 정상인
# 환자를 이유 없이 계속 제외하게 되어 비워둠. 향후 새로운 손상 사례가 생기면 여기 추가할 것
EXCLUDED_PATIENT_IDS: set[int] = set()

# CT 스캐너 모델명(Manufacturer 컬럼 원문) -> 제조사. m&m 초안 Materials에 기술된 내부 코호트 4개 모델
# (Siemens Sensation 64/SOMATOM Definition AS+, GE Revolution CT, Philips Ingenuity Core 128)과
# 외부 코호트에서 관측된 나머지 모델명(Revolution EVO/LightSpeed VCT/SOMATOM Definition Flash 등),
# 스캐너 제한 해제로 추가된 Canon(Aquilion 계열)까지 모두 접두어 기준으로 매핑한다. 매핑 안 되는 모델명이
# 있으면 KeyError 대신 명시적으로 실패시킨다
VENDOR_PREFIXES = {
    "Siemens": ("SOMATOM", "Sensation", "Definition", "Emotion", "Scope", "Spirit"),
    "GE": ("Revolution", "LightSpeed", "Discovery", "Optima", "Brivo"),
    "Philips": ("Ingenuity", "iCT", "Brilliance", "MX", "IQon"),
    "Canon": ("Aquilion", "Activion", "ECLOS", "Supria"),
}


# Manufacturer 원문 문자열을 접두어 매칭으로 제조사(Siemens/GE/Philips/Canon) 이름으로 변환
def classify_vendor(manufacturer: str) -> str:
    for vendor, prefixes in VENDOR_PREFIXES.items():
        if manufacturer.startswith(prefixes):
            return vendor
    raise ValueError(f"매핑되지 않은 CT 스캐너 모델명: {manufacturer!r}")


# 원본 metadata에서 연령<20만 제외(스캐너/벤더 제한 없음)한 뒤 Table 1에 필요한 컬럼 전체가 결측 없이
# 채워져 있는지 확인
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[~meta["PatientID"].isin(EXCLUDED_PATIENT_IDS)].reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    missing = meta[REQUIRED_COLS].isna().sum()
    assert missing.sum() == 0, f"{xlsx_path.name}: Table 1 필수 컬럼에 결측값 존재\n{missing[missing > 0]}"
    meta["Vendor"] = meta["Manufacturer"].map(classify_vendor)
    return meta


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


ROW_SECTION, ROW_DATA = "section", "data"


# section 헤더 행(연속형/범주형 변수를 구역화할 때만 사용, 성별 비교 표에서는 미사용)
def section_row(label: str, col_a: str, col_b: str) -> dict:
    return {"kind": ROW_SECTION, "Characteristic": label, col_a: "", col_b: "", "p-value": ""}


# 연속형 변수 1개를 두 그룹(a/b) 각각 mean±SD로 요약하고 Welch's t-test(등분산 가정 없음)로 비교
def continuous_row(label: str, vals_a: pd.Series, vals_b: pd.Series, col_a: str, col_b: str) -> dict:
    vals_a, vals_b = vals_a.astype(float), vals_b.astype(float)
    _, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    return {
        "kind": ROW_DATA, "Characteristic": label,
        col_a: f"{vals_a.mean():.1f} ± {vals_a.std(ddof=1):.1f}",
        col_b: f"{vals_b.mean():.1f} ± {vals_b.std(ddof=1):.1f}",
        "p-value": format_p(p),
    }


# 연속형 변수의 범위(min-max)만 참고용으로 별도 행에 표시(검정 대상 아님, p-value 없음)
def range_row(label: str, vals_a: pd.Series, vals_b: pd.Series, col_a: str, col_b: str) -> dict:
    return {
        "kind": ROW_DATA, "Characteristic": label,
        col_a: f"{int(vals_a.min())}–{int(vals_a.max())}",
        col_b: f"{int(vals_b.min())}–{int(vals_b.max())}",
        "p-value": "",
    }


# 범주형 변수 1개에 대해 (카테고리 x 그룹) 분할표로 chi-square test를 1회 수행한다. 카테고리가 2개인
# 이진 변수(예: HTN yes/no)는 양성 n(%)와 p-value를 한 행에 합쳐서 보여주고, 카테고리가 3개 이상인
# 변수(제조사)는 header 행(p-value만)과 카테고리별 서브 행(n(%)만)으로 나눈다
def categorical_rows(label: str, series_a: pd.Series, series_b: pd.Series, category_labels: list[str],
                      keep_all_categories: bool, col_a: str, col_b: str) -> list[dict]:
    counts_a = series_a.value_counts().reindex(category_labels, fill_value=0)
    counts_b = series_b.value_counts().reindex(category_labels, fill_value=0)
    n_a, n_b = len(series_a), len(series_b)

    contingency = np.array([counts_a.to_numpy(), counts_b.to_numpy()])
    _, p, _, _ = stats.chi2_contingency(contingency)

    def count_cell(counts: pd.Series, cat, n: int) -> str:
        return f"{counts[cat]} ({counts[cat] / n:.1%})"

    if not keep_all_categories:
        cat = category_labels[0]
        return [{"kind": ROW_DATA, "Characteristic": f"{label}, n (%)",
                  col_a: count_cell(counts_a, cat, n_a), col_b: count_cell(counts_b, cat, n_b),
                  "p-value": format_p(p)}]

    rows = [{"kind": ROW_DATA, "Characteristic": f"{label}, n (%)", col_a: "", col_b: "", "p-value": format_p(p)}]
    for cat in category_labels:
        rows.append({
            "kind": ROW_DATA, "Characteristic": f"— {cat}",
            col_a: count_cell(counts_a, cat, n_a),
            col_b: count_cell(counts_b, cat, n_b),
            "p-value": "",
        })
    return rows


# 표를 xlsx로 저장. "kind"==section인 행이 있어도 xlsx에는 "kind" 컬럼을 내보내지 않는다
def save_table(table: pd.DataFrame, col_a: str, col_b: str, out_stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_cols = ["Characteristic", col_a, col_b, "p-value"]
    table[export_cols].to_excel(output_dir / f"{out_stem}.xlsx", index=False)
    print(f"Saved table to {output_dir}")
