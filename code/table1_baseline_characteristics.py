from __future__ import annotations

# m&m 초안(docs/m&m 초안_한글.docx) Table 1(환자 기저 특성)을 만드는 스크립트가 이제까지 repo에 없어서
# (원본 xlsx로부터 ad hoc 계산 후 문서에 직접 기입된 상태였음) 신설. 사용자 확인 2026-08-19: "생성해"
# (docx 값이 원본과 일치하는지 확인해달라는 리뷰 요청에 이어, 재현 가능한 스크립트로 고정해달라는 요청).
# internal(gangnam)/external(sinchon) metadata를 그대로 로드해 연속형은 Welch's t-test, 범주형은
# chi-square test로 비교하며, docx Table 1과 동일한 행 구성(Age/Sex/Height/Weight/BMI/HTN/DM/CKD/
# CT scanner vendor)으로 출력한다. Sex와 CT scanner vendor는 카테고리 전체에 대해 검정을 1회만 수행하고
# p-value는 헤더 행에만 붙인다(기존 docx에서 Sex female/male 두 행에 p-value가 중복 표기되어 있던 것을
# 리뷰 중 발견, 이 스크립트에서는 처음부터 중복 없이 생성).

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 기본 cp949가 ±/–(en dash) 등을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "table"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"

REQUIRED_COLS = ["PatientAge", "PatientSex", "Height", "Weight", "BMI", "HTN", "DM", "CKD", "Manufacturer"]

# CT 스캐너 모델명(Manufacturer 컬럼 원문) -> 제조사. m&m 초안 Materials에 기술된 내부 코호트 4개 모델
# (Siemens Sensation 64/SOMATOM Definition AS+, GE Revolution CT, Philips Ingenuity Core 128)과
# 외부 코호트에서 관측된 나머지 모델명(Revolution EVO/LightSpeed VCT/SOMATOM Definition Flash 등)을
# 모두 접두어 기준으로 매핑한다. 매핑 안 되는 모델명이 있으면 KeyError 대신 명시적으로 실패시킨다
VENDOR_PREFIXES = {
    "Siemens": ("SOMATOM", "Sensation"),
    "GE": ("Revolution", "LightSpeed", "Discovery"),
    "Philips": ("Ingenuity", "iCT"),
}


# Manufacturer 원문 문자열을 접두어 매칭으로 제조사(Siemens/GE/Philips) 이름으로 변환
def classify_vendor(manufacturer: str) -> str:
    for vendor, prefixes in VENDOR_PREFIXES.items():
        if manufacturer.startswith(prefixes):
            return vendor
    raise ValueError(f"매핑되지 않은 CT 스캐너 모델명: {manufacturer!r}")


# metadata 시트를 로드하고 Vendor 컬럼을 추가, Table 1에 필요한 컬럼 전체가 결측 없이 채워져 있는지 확인
# (m&m 초안 Materials: "두 코호트 모두 ... 결측값은 없었다" 서술을 이 스크립트에서 매번 재검증)
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    missing = meta[REQUIRED_COLS].isna().sum()
    assert missing.sum() == 0, f"{xlsx_path.name}: Table 1 필수 컬럼에 결측값 존재\n{missing[missing > 0]}"
    meta["Vendor"] = meta["Manufacturer"].map(classify_vendor)
    return meta


# 연속형 변수 1개(예: Age)를 internal/external 각각 mean±SD로 요약하고 Welch's t-test(등분산 가정 없음)로 비교
def continuous_row(label: str, int_vals: pd.Series, ext_vals: pd.Series) -> dict:
    int_vals, ext_vals = int_vals.astype(float), ext_vals.astype(float)
    _, p = stats.ttest_ind(int_vals, ext_vals, equal_var=False)
    return {
        "Characteristic": label,
        "Internal": f"{int_vals.mean():.1f} ± {int_vals.std(ddof=1):.1f}",
        "External": f"{ext_vals.mean():.1f} ± {ext_vals.std(ddof=1):.1f}",
        "p-value": format_p(p),
    }


# 연속형 변수의 범위(min-max)만 참고용으로 별도 행에 표시(검정 대상 아님, p-value 없음)
def range_row(label: str, int_vals: pd.Series, ext_vals: pd.Series) -> dict:
    return {
        "Characteristic": label,
        "Internal": f"{int(int_vals.min())}–{int(int_vals.max())}",
        "External": f"{int(ext_vals.min())}–{int(ext_vals.max())}",
        "p-value": "",
    }


# 범주형 변수 1개(예: HTN)에 대해 (카테고리 x 코호트) 분할표로 chi-square test를 1회 수행한다.
# 카테고리가 2개인 이진 변수(예: HTN yes/no)는 양성 n(%)와 p-value를 한 행에 합쳐서 보여주고
# (docx Table 1의 Hypertension/Diabetes/CKD 행과 동일 형식), 카테고리가 3개 이상인 변수(성별, 제조사)는
# header 행(p-value만)과 카테고리별 서브 행(n(%)만)으로 나눈다 - 두 행 모두에 동일 p-value를 중복 표기했던
# 기존 docx Table 1의 Sex 행 오류를 이 스크립트에서는 애초에 재현하지 않는다
def categorical_rows(label: str, int_series: pd.Series, ext_series: pd.Series, category_labels: list[str],
                      keep_all_categories: bool) -> list[dict]:
    counts_int = int_series.value_counts().reindex(category_labels, fill_value=0)
    counts_ext = ext_series.value_counts().reindex(category_labels, fill_value=0)
    n_int, n_ext = len(int_series), len(ext_series)

    contingency = np.array([counts_int.to_numpy(), counts_ext.to_numpy()])
    _, p, _, _ = stats.chi2_contingency(contingency)

    def count_cell(counts: pd.Series, cat, n: int) -> str:
        return f"{counts[cat]} ({counts[cat] / n:.1%})"

    if not keep_all_categories:
        cat = category_labels[0]
        return [{"Characteristic": f"{label}, n (%)", "Internal": count_cell(counts_int, cat, n_int),
                  "External": count_cell(counts_ext, cat, n_ext), "p-value": format_p(p)}]

    rows = [{"Characteristic": f"{label}, n (%)", "Internal": "", "External": "", "p-value": format_p(p)}]
    for cat in category_labels:
        rows.append({
            "Characteristic": f"— {cat}",
            "Internal": count_cell(counts_int, cat, n_int),
            "External": count_cell(counts_ext, cat, n_ext),
            "p-value": "",
        })
    return rows


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# docx Table 1과 동일한 행 순서(Age/Sex/Height/Weight/BMI/HTN/DM/CKD/CT scanner vendor)로 표를 구성
def build_table(meta_int: pd.DataFrame, meta_ext: pd.DataFrame) -> pd.DataFrame:
    rows = [
        continuous_row("Age, years, mean ± SD", meta_int["PatientAge"], meta_ext["PatientAge"]),
        range_row("Age, years, range", meta_int["PatientAge"], meta_ext["PatientAge"]),
        *categorical_rows("Sex", meta_int["PatientSex"].str.upper(), meta_ext["PatientSex"].str.upper(),
                           ["F", "M"], keep_all_categories=True),
        continuous_row("Height, cm, mean ± SD", meta_int["Height"], meta_ext["Height"]),
        continuous_row("Weight, kg, mean ± SD", meta_int["Weight"], meta_ext["Weight"]),
        continuous_row("BMI, kg/m², mean ± SD", meta_int["BMI"], meta_ext["BMI"]),
        *categorical_rows("Hypertension", meta_int["HTN"], meta_ext["HTN"], [1, 0], keep_all_categories=False),
        *categorical_rows("Diabetes mellitus", meta_int["DM"], meta_ext["DM"], [1, 0], keep_all_categories=False),
        *categorical_rows("Chronic kidney disease", meta_int["CKD"], meta_ext["CKD"], [1, 0],
                           keep_all_categories=False),
        *categorical_rows("CT scanner vendor", meta_int["Vendor"], meta_ext["Vendor"],
                           ["Siemens", "GE", "Philips"], keep_all_categories=True),
    ]
    return pd.DataFrame(rows)


# 표를 csv/xlsx로 저장하고, docx에 그대로 옮겨 붙일 수 있는 표 이미지도 함께 저장
def save_table(table: pd.DataFrame, n_int: int, n_ext: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "table1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    table.to_excel(output_dir / "table1_baseline_characteristics.xlsx", index=False)
    print(f"Saved Table 1 to {output_dir}")

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    col_labels = ["Characteristic", f"Internal cohort\n(n = {n_int:,})", f"External cohort\n(n = {n_ext:,})",
                  "p-value"]
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.6 * len(table)))
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
    fig.suptitle("Table 1. 환자 기저 특성", fontsize=22, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "table1_baseline_characteristics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Table 1 image to {output_dir / 'table1_baseline_characteristics.png'}")


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    table = build_table(meta_int, meta_ext)
    print(table.to_string(index=False))
    save_table(table, len(meta_int), len(meta_ext), OUTPUT_DIR)


if __name__ == "__main__":
    main()
