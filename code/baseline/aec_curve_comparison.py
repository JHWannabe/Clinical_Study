"""
data/g1090.xlsx 의 'aec_128' 시트(환자별 128-slice raw AEC 프로파일)를 이용해
성별 / 나이 / TAMA / BMI / Height / Weight 그룹 간 AEC "point curve"(슬라이스별
평균 곡선 + 신뢰구간 리본) 비교 그래프를 그린다.

- 각 환자 곡선은 자기 자신의 평균값으로 나눠 정규화한다 (patient-normalized AEC).
- 그룹별 정규화 곡선을 슬라이스 index(1~128)마다 평균 + 95% CI로 겹쳐 그린다.
- 연속형 변수(Age/TAMA/BMI/Height/Weight)는 중앙값 기준 상/하 2그룹으로 나눈다.
- 레거시 파이프라인(main_aec_full_derivation_pipeline_simplified.py 등)은
  재사용하지 않고 이 스크립트에서 새로 계산한다.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- palette --
COL_A = "#2a78d6"   # 그룹 1 (blue)
COL_B = "#1baf7a"   # 그룹 2 (aqua)
COL_C = "#eda100"   # 그룹 3 (yellow) - 3그룹 이상 비교용
COL_D = "#4a3aa7"   # 그룹 4 (violet)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DATA_PATH = os.path.join(os.path.dirname(__file__), "../..", "data", "g1090.xlsx")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../..", "outputs", "aec_curve_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def savefig(fig, name):
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved: {path}")


VENDOR_MAP = {
    "Sensation 64": "Siemens",
    "SOMATOM Definition AS+": "Siemens",
    "SOMATOM Definition Edge": "Siemens",
    "SOMATOM Definition": "Siemens",
    "SOMATOM Drive": "Siemens",
    "Revolution CT": "GE",
    "Optima CT660": "GE",
    "Ingenuity Core 128": "Philips",
    "Aquilion ONE": "Canon",
    "Aquilion": "Canon",
}

# 프로젝트 기존 low-SMI 임상 cutoff (main_aec_full_derivation_pipeline_simplified.py 정의값과 동일:
# 남성 <45.4, 여성 <34.4 cm^2/m^2). 값만 재사용하고 계산/파이프라인 로직은 새로 작성.
LOW_SMI_CUTOFF = {"M": 45.4, "F": 34.4}


def classify_contrast_phase(desc):
    d = str(desc).lower()
    if "w/o" in d or "without" in d or "pre contrast" in d:
        return "Non-contrast"
    if any(k in d for k in ["with contrast", "post", "portal", "contrast"]):
        return "Contrast"
    return "Non-contrast"


def load_data():
    meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    aec = pd.read_excel(DATA_PATH, sheet_name="aec_128")

    curves = aec[AEC_COLS].astype(float).to_numpy()
    patient_mean = curves.mean(axis=1, keepdims=True)
    norm_curves = curves / patient_mean  # patient-normalized AEC

    norm_df = pd.DataFrame(norm_curves, columns=AEC_COLS)
    norm_df.insert(0, "PatientID", aec["PatientID"].to_numpy())
    norm_df["z_range"] = aec["z_range"].values
    norm_df["n_slices_cropped"] = aec["n_slices_cropped"].values

    df = meta.merge(norm_df, on="PatientID", how="inner")

    df["AgeGroup2"] = np.where(df.PatientAge <= df.PatientAge.median(), "Low", "High")
    for col in ["TAMA", "BMI", "Height", "Weight", "z_range"]:
        med = df[col].median()
        df[f"{col}Group2"] = np.where(df[col] <= med, "Low", "High")

    cutoff = df["PatientSex"].map(LOW_SMI_CUTOFF)
    df["LowSMI"] = np.where(df["SMI"] < cutoff, "Low SMI", "Non-low SMI")

    df["IMATA_TAMA_ratio"] = df["IMATA"] / df["TAMA"]
    med = df["IMATA_TAMA_ratio"].median()
    df["MyosteatosisGroup2"] = np.where(df["IMATA_TAMA_ratio"] <= med, "Low", "High")

    df["Vendor"] = df["Manufacturer"].map(VENDOR_MAP)

    df["SliceThickness"] = df["z_range"] / df["n_slices_cropped"]
    med = df["SliceThickness"].median()
    df["SliceThicknessGroup2"] = np.where(df["SliceThickness"] <= med, "Low", "High")

    df["ContrastPhase"] = df["Series_Desc"].map(classify_contrast_phase)

    bmi_bins = [0, 18.5, 23, 25, 100]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMIGroup4"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=False)

    df["SexSMIGroup"] = df["PatientSex"] + " / " + df["LowSMI"]

    return df


def smooth(mat_mean, window=5):
    s = pd.Series(mat_mean)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def group_curve_stats(df, group_col, group_val):
    sub = df.loc[df[group_col] == group_val, AEC_COLS].to_numpy()
    mean = sub.mean(axis=0)
    sem = stats.sem(sub, axis=0)
    ci = 1.96 * sem
    return smooth(mean), smooth(ci), len(sub)


def plot_curve_comparison(ax, df, group_col, order, labels, colors, title):
    x = np.arange(1, N_SLICES + 1)
    for val, label, color in zip(order, labels, colors):
        mean, ci, n = group_curve_stats(df, group_col, val)
        ax.plot(x, mean, color=color, linewidth=2, label=f"{label} (n={n})")
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(1, N_SLICES)
    ax.set_xlabel("Slice index (1-128, resampled)")
    ax.set_ylabel("Patient-normalized AEC")
    ax.set_title(title, color=INK_PRIMARY, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="best")
    style_axes(ax)


def group_diff_note(df, group_col, order):
    """patient-mean AEC 기준: 2그룹이면 Mann-Whitney U, 3그룹 이상이면 Kruskal-Wallis."""
    samples = [df.loc[df[group_col] == v, AEC_COLS].to_numpy().mean(axis=1) for v in order]
    if len(samples) == 2:
        stat, p = stats.mannwhitneyu(*samples, alternative="two-sided")
        return f"Mann-Whitney p={p:.3g}"
    stat, p = stats.kruskal(*samples)
    return f"Kruskal-Wallis p={p:.3g}"


def main():
    df = load_data()
    print(f"merged patients: {len(df)}")

    specs = [
        ("PatientSex", ["M", "F"], ["Male", "Female"], "01_aec_curve_by_sex.png", "성별에 따른 AEC 곡선 비교"),
        ("AgeGroup2", ["Low", "High"], ["Age ≤ median", "Age > median"],
         "02_aec_curve_by_age.png", "나이(중앙값 분할)에 따른 AEC 곡선 비교"),
        ("TAMAGroup2", ["Low", "High"], ["TAMA ≤ median", "TAMA > median"],
         "03_aec_curve_by_tama.png", "TAMA(중앙값 분할)에 따른 AEC 곡선 비교"),
        ("BMIGroup2", ["Low", "High"], ["BMI ≤ median", "BMI > median"],
         "04_aec_curve_by_bmi.png", "BMI(중앙값 분할)에 따른 AEC 곡선 비교"),
        ("HeightGroup2", ["Low", "High"], ["Height ≤ median", "Height > median"],
         "05_aec_curve_by_height.png", "신장(중앙값 분할)에 따른 AEC 곡선 비교"),
        ("WeightGroup2", ["Low", "High"], ["Weight ≤ median", "Weight > median"],
         "06_aec_curve_by_weight.png", "체중(중앙값 분할)에 따른 AEC 곡선 비교"),
        ("LowSMI", ["Low SMI", "Non-low SMI"], ["Low SMI (sarcopenia)", "Non-low SMI"],
         "07_aec_curve_by_low_smi.png", "Low-SMI 임상 cutoff에 따른 AEC 곡선 비교"),
        ("MyosteatosisGroup2", ["Low", "High"], ["IMATA/TAMA ≤ median", "IMATA/TAMA > median"],
         "08_aec_curve_by_myosteatosis.png", "IMATA/TAMA 비율(근지방침윤)에 따른 AEC 곡선 비교"),
        ("z_rangeGroup2", ["Low", "High"], ["Scan length ≤ median", "Scan length > median"],
         "09_aec_curve_by_scan_length.png", "스캔 커버리지 길이(z_range)에 따른 AEC 곡선 비교"),
        ("ContrastPhase", ["Non-contrast", "Contrast"], ["Non-contrast", "Contrast"],
         "12_aec_curve_by_contrast.png", "조영제 사용 여부에 따른 AEC 곡선 비교"),
        ("SliceThicknessGroup2", ["Low", "High"], ["Slice thickness ≤ median", "Slice thickness > median"],
         "13_aec_curve_by_slice_thickness.png", "재구성 슬라이스 두께(중앙값 분할)에 따른 AEC 곡선 비교"),
    ]

    two_group_colors = [COL_A, COL_B]

    # 개별 그래프 (2그룹 비교)
    for group_col, order, labels, fname, title_kr in specs:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_curve_comparison(ax, df, group_col, order, labels, two_group_colors, title_kr)
        note = group_diff_note(df, group_col, order)
        ax.text(0.02, 0.02, f"patient-mean AEC {note}",
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        fig.tight_layout()
        savefig(fig, fname)

    # 통합 4x3 패널 (2그룹 비교만)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    for ax, (group_col, order, labels, _, title_kr) in zip(axes.flat, specs):
        plot_curve_comparison(ax, df, group_col, order, labels, two_group_colors, title_kr)
    for ax in axes.flat[len(specs):]:
        ax.axis("off")
    fig.suptitle("변수별 AEC point curve 비교 (환자 정규화, mean ± 95% CI)",
                 fontsize=14, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, "10_aec_curve_combined_panel.png")

    # 3그룹 이상 비교 (제조사, BMI 4구간, 성별x Low-SMI 조합)
    multi_specs = []

    vendor_df = df.dropna(subset=["Vendor"]).copy()
    vendor_counts = vendor_df["Vendor"].value_counts()
    vendor_order = vendor_counts[vendor_counts >= 30].index.tolist()
    vendor_df = vendor_df[vendor_df["Vendor"].isin(vendor_order)]
    multi_specs.append((vendor_df, "Vendor", vendor_order, vendor_order,
                         "11_aec_curve_by_vendor.png", "스캐너 제조사(Vendor)에 따른 AEC 곡선 비교"))

    bmi4_order = ["Underweight", "Normal", "Overweight", "Obese"]
    bmi4_df = df.dropna(subset=["BMIGroup4"]).copy()
    bmi4_df["BMIGroup4"] = bmi4_df["BMIGroup4"].astype(str)
    multi_specs.append((bmi4_df, "BMIGroup4", bmi4_order, bmi4_order,
                         "14_aec_curve_by_bmi4.png", "BMI 4구간(WHO 아시아 기준)에 따른 AEC 곡선 비교"))

    sexsmi_order = ["M / Low SMI", "M / Non-low SMI", "F / Low SMI", "F / Non-low SMI"]
    multi_specs.append((df, "SexSMIGroup", sexsmi_order, sexsmi_order,
                         "15_aec_curve_by_sex_x_lowsmi.png", "성별 x Low-SMI 조합에 따른 AEC 곡선 비교"))

    multi_colors = [COL_A, COL_B, COL_C, COL_D]
    for gdf, group_col, order, labels, fname, title_kr in multi_specs:
        colors = multi_colors[: len(order)]
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_curve_comparison(ax, gdf, group_col, order, labels, colors, title_kr)
        note = group_diff_note(gdf, group_col, order)
        ax.text(0.02, 0.02, f"patient-mean AEC {note}",
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        fig.tight_layout()
        savefig(fig, fname)

    # BMI 4구간 x Low-SMI 교차 패널: BMI 효과와 SMI 효과를 동시에 분리해서 확인
    bmi4_df["LowSMI"] = bmi4_df["LowSMI"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    smi_order = ["Low SMI", "Non-low SMI"]
    smi_labels = ["Low SMI", "Non-low SMI"]
    for ax, bmi_group in zip(axes.flat, bmi4_order):
        sub = bmi4_df[bmi4_df["BMIGroup4"] == bmi_group]
        plot_curve_comparison(ax, sub, "LowSMI", smi_order, smi_labels, two_group_colors,
                              f"BMI: {bmi_group} (n={len(sub)})")
    fig.suptitle("BMI 4구간 내에서 Low-SMI 효과 분리 (BMI x SMI 교차비교)",
                 fontsize=14, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "16_aec_curve_bmi4_x_lowsmi_facet.png")

    # SMI를 통제한 뒤(Non-low SMI만) BMI 4구간 효과가 남는지 재비교
    controlled_df = bmi4_df[bmi4_df["LowSMI"] == "Non-low SMI"]
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_curve_comparison(ax, controlled_df, "BMIGroup4", bmi4_order, bmi4_order,
                          multi_colors[: len(bmi4_order)],
                          "SMI 통제(Non-low SMI만) 후 BMI 4구간에 따른 AEC 곡선 비교")
    note = group_diff_note(controlled_df, "BMIGroup4", bmi4_order)
    ax.text(0.02, 0.02, f"patient-mean AEC {note}\n(Low-SMI 환자 제외, n={len(controlled_df)})",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout()
    savefig(fig, "17_aec_curve_bmi4_smi_controlled.png")

    print(f"\nAll figures saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
