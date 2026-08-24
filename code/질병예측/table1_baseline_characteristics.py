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

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949가 ±/–(en dash) 등을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "table"

INTERNAL_XLSX = DATA_DIR / "gangnam_원본.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_원본.xlsx"
AGE_CUTOFF = 20
COHORTS = [("internal", "Internal", INTERNAL_XLSX), ("external", "External", EXTERNAL_XLSX)]

REQUIRED_COLS = ["PatientAge", "PatientSex", "Height", "Weight", "BMI", "HTN", "DM", "CKD", "Manufacturer"]

# CT 스캐너 모델명(Manufacturer 컬럼 원문) -> 제조사. m&m 초안 Materials에 기술된 내부 코호트 4개 모델
# (Siemens Sensation 64/SOMATOM Definition AS+, GE Revolution CT, Philips Ingenuity Core 128)과
# 외부 코호트에서 관측된 나머지 모델명(Revolution EVO/LightSpeed VCT/SOMATOM Definition Flash 등),
# 스캐너 제한 해제로 추가된 Canon(Aquilion 계열)까지 모두 접두어 기준으로 매핑한다. 매핑 안 되는 모델명이
# 있으면 KeyError 대신 명시적으로 실패시킨다
VENDOR_PREFIXES = {
    "Siemens": ("SOMATOM", "Sensation"),
    "GE": ("Revolution", "LightSpeed", "Discovery", "Optima"),
    "Philips": ("Ingenuity", "iCT"),
    "Canon": ("Aquilion",),
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
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    missing = meta[REQUIRED_COLS].isna().sum()
    assert missing.sum() == 0, f"{xlsx_path.name}: Table 1 필수 컬럼에 결측값 존재\n{missing[missing > 0]}"
    meta["Vendor"] = meta["Manufacturer"].map(classify_vendor)
    return meta


# 연속형 변수 1개(예: Age)를 Men/Women 각각 mean±SD로 요약하고 Welch's t-test(등분산 가정 없음)로 비교
def continuous_row(label: str, men_vals: pd.Series, women_vals: pd.Series) -> dict:
    men_vals, women_vals = men_vals.astype(float), women_vals.astype(float)
    _, p = stats.ttest_ind(men_vals, women_vals, equal_var=False)
    return {
        "Characteristic": label,
        "Men": f"{men_vals.mean():.1f} ± {men_vals.std(ddof=1):.1f}",
        "Women": f"{women_vals.mean():.1f} ± {women_vals.std(ddof=1):.1f}",
        "p-value": format_p(p),
    }


# 연속형 변수의 범위(min-max)만 참고용으로 별도 행에 표시(검정 대상 아님, p-value 없음)
def range_row(label: str, men_vals: pd.Series, women_vals: pd.Series) -> dict:
    return {
        "Characteristic": label,
        "Men": f"{int(men_vals.min())}–{int(men_vals.max())}",
        "Women": f"{int(women_vals.min())}–{int(women_vals.max())}",
        "p-value": "",
    }


# 범주형 변수 1개(예: HTN)에 대해 (카테고리 x 성별) 분할표로 chi-square test를 1회 수행한다.
# 카테고리가 2개인 이진 변수(예: HTN yes/no)는 양성 n(%)와 p-value를 한 행에 합쳐서 보여주고
# (docx Table 1의 Hypertension/Diabetes/CKD 행과 동일 형식), 카테고리가 3개 이상인 변수(제조사)는
# header 행(p-value만)과 카테고리별 서브 행(n(%)만)으로 나눈다
def categorical_rows(label: str, men_series: pd.Series, women_series: pd.Series, category_labels: list[str],
                      keep_all_categories: bool) -> list[dict]:
    counts_men = men_series.value_counts().reindex(category_labels, fill_value=0)
    counts_women = women_series.value_counts().reindex(category_labels, fill_value=0)
    n_men, n_women = len(men_series), len(women_series)

    contingency = np.array([counts_men.to_numpy(), counts_women.to_numpy()])
    _, p, _, _ = stats.chi2_contingency(contingency)

    def count_cell(counts: pd.Series, cat, n: int) -> str:
        return f"{counts[cat]} ({counts[cat] / n:.1%})"

    if not keep_all_categories:
        cat = category_labels[0]
        return [{"Characteristic": f"{label}, n (%)", "Men": count_cell(counts_men, cat, n_men),
                  "Women": count_cell(counts_women, cat, n_women), "p-value": format_p(p)}]

    rows = [{"Characteristic": f"{label}, n (%)", "Men": "", "Women": "", "p-value": format_p(p)}]
    for cat in category_labels:
        rows.append({
            "Characteristic": f"— {cat}",
            "Men": count_cell(counts_men, cat, n_men),
            "Women": count_cell(counts_women, cat, n_women),
            "p-value": "",
        })
    return rows


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# 코호트 1개(internal 또는 external)의 남성/여성 부분집합을 비교하는 표를 구성. 참고 이미지(Table 1:
# Baseline Characteristics of the Participants by Prevalent Diabetes)의 행 구성(연속형은 mean±SD,
# 이진 범주형은 n(%)+p-value 한 행)을 따르되, HTN/DM/CKD는 참고 이미지처럼 성별x질환 조합 컬럼을 만들지
# 않고 이 표 안에 각각 별도 행으로만 추가한다(사용자 확인 2026-08-24)
def build_table_by_sex(meta: pd.DataFrame) -> pd.DataFrame:
    men = meta[meta["PatientSex"].str.upper() == "M"]
    women = meta[meta["PatientSex"].str.upper() == "F"]
    vendor_labels = sorted(set(meta["Vendor"]), key=lambda v: ["Siemens", "GE", "Philips", "Canon"].index(v))
    rows = [
        continuous_row("Age, years, mean ± SD", men["PatientAge"], women["PatientAge"]),
        range_row("Age, years, range", men["PatientAge"], women["PatientAge"]),
        continuous_row("Height, cm, mean ± SD", men["Height"], women["Height"]),
        continuous_row("Weight, kg, mean ± SD", men["Weight"], women["Weight"]),
        continuous_row("BMI, kg/m², mean ± SD", men["BMI"], women["BMI"]),
        *categorical_rows("Hypertension", men["HTN"], women["HTN"], [1, 0], keep_all_categories=False),
        *categorical_rows("Diabetes mellitus", men["DM"], women["DM"], [1, 0], keep_all_categories=False),
        *categorical_rows("Chronic kidney disease", men["CKD"], women["CKD"], [1, 0], keep_all_categories=False),
        *categorical_rows("CT scanner vendor", men["Vendor"], women["Vendor"], vendor_labels,
                           keep_all_categories=True),
    ]
    return pd.DataFrame(rows), len(men), len(women)


# 표를 csv/xlsx로 저장하고, docx에 그대로 옮겨 붙일 수 있는 표 이미지도 함께 저장. 참고 이미지의 각주
# 스타일(검정 방법 명시)을 반영해 표 아래에 방법론 각주를 추가한다
def save_table(table: pd.DataFrame, n_men: int, n_women: int, cohort_key: str, cohort_label: str,
               output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table1_baseline_characteristics_{cohort_key}"
    table.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    table.to_excel(output_dir / f"{stem}.xlsx", index=False)
    print(f"Saved Table 1 ({cohort_label}) to {output_dir}")

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    col_labels = ["Characteristic", f"Men\n(n = {n_men:,})", f"Women\n(n = {n_women:,})", "p-value"]
    fig, ax = plt.subplots(figsize=(13, 1.6 + 0.6 * len(table)))
    ax.axis("off")
    tbl = ax.table(cellText=table.to_numpy(), colLabels=col_labels, colWidths=[0.4, 0.22, 0.22, 0.16],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1, 2.6)
    for (row_i, _col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=15)
            cell.set_facecolor("#161616")
        else:
            cell.set_facecolor("#f2f1ee" if row_i % 2 == 0 else "white")
    fig.suptitle(f"Table 1. {cohort_label} 코호트 환자 기저 특성 (성별 비교)", fontsize=22, fontweight="bold",
                 y=0.995)
    fig.text(0.02, 0.01,
              "연속형 변수는 mean ± SD로 표시하고 Welch's t-test로, 범주형 변수는 n (%)로 표시하고 "
              "chi-square test로 남/여를 비교했다.", fontsize=11, color="#3a3a3a", ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = output_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Table 1 image ({cohort_label}) to {out_path}")


def main() -> None:
    for cohort_key, cohort_label, xlsx_path in COHORTS:
        meta = load_cohort(xlsx_path)
        table, n_men, n_women = build_table_by_sex(meta)
        print(f"=== {cohort_label} (n_men={n_men}, n_women={n_women}) ===")
        print(table.to_string(index=False))
        save_table(table, n_men, n_women, cohort_key, cohort_label, OUTPUT_DIR)


if __name__ == "__main__":
    main()
