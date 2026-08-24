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

REQUIRED_COLS = ["PatientAge", "PatientSex", "Height", "Weight", "BMI", "HTN", "DM", "CKD", "Manufacturer"]

# code/질병예측/table1_baseline_characteristics.py와 동일한 제조사 매핑(Canon/Optima 포함, 스캐너 제한 해제 반영)
VENDOR_PREFIXES = {
    "Siemens": ("SOMATOM", "Sensation"),
    "GE": ("Revolution", "LightSpeed", "Discovery", "Optima"),
    "Philips": ("Ingenuity", "iCT"),
    "Canon": ("Aquilion",),
}


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


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


ROW_SECTION, ROW_DATA = "section", "data"


def section_row(label: str) -> dict:
    return {"kind": ROW_SECTION, "Characteristic": label, "Internal": "", "External": "", "p-value": ""}


def continuous_row(label: str, int_vals: pd.Series, ext_vals: pd.Series) -> dict:
    int_vals, ext_vals = int_vals.astype(float), ext_vals.astype(float)
    _, p = stats.ttest_ind(int_vals, ext_vals, equal_var=False)
    return {
        "kind": ROW_DATA, "Characteristic": label,
        "Internal": f"{int_vals.mean():.1f} ± {int_vals.std(ddof=1):.1f}",
        "External": f"{ext_vals.mean():.1f} ± {ext_vals.std(ddof=1):.1f}",
        "p-value": format_p(p),
    }


def range_row(label: str, int_vals: pd.Series, ext_vals: pd.Series) -> dict:
    return {
        "kind": ROW_DATA, "Characteristic": label,
        "Internal": f"{int(int_vals.min())}–{int(int_vals.max())}",
        "External": f"{int(ext_vals.min())}–{int(ext_vals.max())}",
        "p-value": "",
    }


# 범주형 변수 1개에 대해 (카테고리 x 코호트) 분할표로 chi-square test를 1회 수행. 이진 변수는 양성
# n(%)+p-value를 한 행에, 3개 이상 카테고리는 header 행(p-value만) + 카테고리별 서브 행(n(%)만)으로 구성
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
        return [{"kind": ROW_DATA, "Characteristic": f"{label}, n (%)",
                  "Internal": count_cell(counts_int, cat, n_int),
                  "External": count_cell(counts_ext, cat, n_ext), "p-value": format_p(p)}]

    rows = [{"kind": ROW_DATA, "Characteristic": f"{label}, n (%)", "Internal": "", "External": "",
              "p-value": format_p(p)}]
    for cat in category_labels:
        rows.append({
            "kind": ROW_DATA, "Characteristic": f"— {cat}",
            "Internal": count_cell(counts_int, cat, n_int),
            "External": count_cell(counts_ext, cat, n_ext),
            "p-value": "",
        })
    return rows


# 레퍼런스 논문 Table 1(Anthropometry/Impedance analysis/CT-derived measures 같은 section 헤더로 변수를
# 구역화)을 참고해, 이 표도 Demographics/Anthropometry/Comorbidities/Imaging 4개 section으로 나눴다
def build_table(meta_int: pd.DataFrame, meta_ext: pd.DataFrame) -> pd.DataFrame:
    rows = [
        section_row("Demographics"),
        continuous_row("Age, years, mean ± SD", meta_int["PatientAge"], meta_ext["PatientAge"]),
        range_row("Age, years, range", meta_int["PatientAge"], meta_ext["PatientAge"]),
        *categorical_rows("Sex", meta_int["PatientSex"].str.upper(), meta_ext["PatientSex"].str.upper(),
                           ["F", "M"], keep_all_categories=True),

        section_row("Anthropometry"),
        continuous_row("Height, cm, mean ± SD", meta_int["Height"], meta_ext["Height"]),
        continuous_row("Weight, kg, mean ± SD", meta_int["Weight"], meta_ext["Weight"]),
        continuous_row("BMI, kg/m², mean ± SD", meta_int["BMI"], meta_ext["BMI"]),

        section_row("Comorbidities"),
        *categorical_rows("Hypertension", meta_int["HTN"], meta_ext["HTN"], [1, 0], keep_all_categories=False),
        *categorical_rows("Diabetes mellitus", meta_int["DM"], meta_ext["DM"], [1, 0], keep_all_categories=False),
        *categorical_rows("Chronic kidney disease", meta_int["CKD"], meta_ext["CKD"], [1, 0],
                           keep_all_categories=False),

        section_row("Imaging"),
        *categorical_rows("CT scanner vendor", meta_int["Vendor"], meta_ext["Vendor"],
                           ["Siemens", "GE", "Philips", "Canon"], keep_all_categories=True),
    ]
    return pd.DataFrame(rows)


# 표를 csv/xlsx/png로 저장. section 헤더 행은 레퍼런스 스타일(연한 배경 + bold, characteristic 칸만 표시)로
# 렌더링하고, 표 하단에 레퍼런스 Table 1의 Note.-- 각주 스타일을 반영한 방법론 설명을 추가한다
def save_table(table: pd.DataFrame, n_int: int, n_ext: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_cols = ["Characteristic", "Internal", "External", "p-value"]
    table[export_cols].to_csv(output_dir / "table1_baseline_characteristics.csv", index=False,
                               encoding="utf-8-sig")
    table[export_cols].to_excel(output_dir / "table1_baseline_characteristics.xlsx", index=False)
    print(f"Saved Table 1 to {output_dir}")

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    col_labels = ["Characteristic", f"Internal cohort\n(n = {n_int:,})", f"External cohort\n(n = {n_ext:,})",
                  "p-value"]
    fig, ax = plt.subplots(figsize=(15, 1.6 + 0.6 * len(table)))
    ax.axis("off")
    tbl = ax.table(cellText=table[export_cols].to_numpy(), colLabels=col_labels,
                    colWidths=[0.4, 0.22, 0.22, 0.16], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1, 2.6)
    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_text_props(weight="bold", color="white", fontsize=15)
            cell.set_facecolor("#161616")
            continue
        is_section = table.iloc[row_i - 1]["kind"] == ROW_SECTION
        if is_section:
            cell.set_facecolor("#cfd8e3")
            if col_i == 0:
                cell.set_text_props(weight="bold", ha="left")
        else:
            cell.set_facecolor("#f2f1ee" if row_i % 2 == 0 else "white")
            if col_i == 0:
                cell.set_text_props(ha="left")
    fig.suptitle("Table 1. Baseline Characteristics of the Participants by Cohort", fontsize=20,
                 fontweight="bold", y=0.995)
    fig.text(0.02, 0.01,
              "Note.—Continuous variables are expressed as mean ± SD and compared with Welch's "
              "t test; categorical variables are expressed as n (%) and compared with the chi-square test. "
              "BMI = body mass index, CKD = chronic kidney disease.",
              fontsize=11, color="#3a3a3a", ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = output_dir / "table1_baseline_characteristics.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Table 1 image to {out_path}")


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    table = build_table(meta_int, meta_ext)
    print(table[["Characteristic", "Internal", "External", "p-value"]].to_string(index=False))
    save_table(table, len(meta_int), len(meta_ext), OUTPUT_DIR)


if __name__ == "__main__":
    main()
