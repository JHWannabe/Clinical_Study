from __future__ import annotations

# code/질병예측/table1_baseline_characteristics.py를 베이스로, 스캐너/벤더 제한을 해제한 확장 코호트
# (data/{gangnam,sinchon}_원본.xlsx에서 연령<20만 제외, internal 1,088명 / external 925명)로 Table 1을
# 재생성. 원본 코호트에는 기존 4개(Siemens/GE/Philips) 외 Canon(Aquilion 계열) 스캐너도 포함되므로
# VENDOR_PREFIXES에 Canon을 추가하고, GE 계열 중 매핑 누락되어 있던 Optima도 추가한다.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "table_v2"

INTERNAL_XLSX = DATA_DIR / "gangnam_원본.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_원본.xlsx"
AGE_CUTOFF = 20

REQUIRED_COLS = ["PatientAge", "PatientSex", "Height", "Weight", "BMI", "HTN", "DM", "CKD", "Manufacturer"]

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


# 원본 metadata에서 연령<20만 제외(스캐너/벤더 제한 없음)한 확장 코호트를 로드
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    missing = meta[REQUIRED_COLS].isna().sum()
    assert missing.sum() == 0, f"{xlsx_path.name}: Table 1 필수 컬럼에 결측값 존재\n{missing[missing > 0]}"
    meta["Vendor"] = meta["Manufacturer"].map(classify_vendor)
    return meta


def continuous_row(label: str, int_vals: pd.Series, ext_vals: pd.Series) -> dict:
    int_vals, ext_vals = int_vals.astype(float), ext_vals.astype(float)
    _, p = stats.ttest_ind(int_vals, ext_vals, equal_var=False)
    return {
        "Characteristic": label,
        "Internal": f"{int_vals.mean():.1f} ± {int_vals.std(ddof=1):.1f}",
        "External": f"{ext_vals.mean():.1f} ± {ext_vals.std(ddof=1):.1f}",
        "p-value": format_p(p),
    }


def range_row(label: str, int_vals: pd.Series, ext_vals: pd.Series) -> dict:
    return {
        "Characteristic": label,
        "Internal": f"{int(int_vals.min())}–{int(int_vals.max())}",
        "External": f"{int(ext_vals.min())}–{int(ext_vals.max())}",
        "p-value": "",
    }


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


def build_table(meta_int: pd.DataFrame, meta_ext: pd.DataFrame) -> pd.DataFrame:
    vendor_labels = sorted(set(meta_int["Vendor"]) | set(meta_ext["Vendor"]),
                            key=lambda v: ["Siemens", "GE", "Philips", "Canon"].index(v))
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
        *categorical_rows("CT scanner vendor", meta_int["Vendor"], meta_ext["Vendor"], vendor_labels,
                           keep_all_categories=True),
    ]
    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, n_int: int, n_ext: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "table1_baseline_characteristics_v2.csv", index=False, encoding="utf-8-sig")
    table.to_excel(output_dir / "table1_baseline_characteristics_v2.xlsx", index=False)
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
    fig.suptitle("Table 1. 환자 기저 특성 (스캐너 제한 해제 코호트)", fontsize=22, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "table1_baseline_characteristics_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Table 1 image to {output_dir / 'table1_baseline_characteristics_v2.png'}")


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)
    table = build_table(meta_int, meta_ext)
    print(table.to_string(index=False))
    save_table(table, len(meta_int), len(meta_ext), OUTPUT_DIR)


if __name__ == "__main__":
    main()
