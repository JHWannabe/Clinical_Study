from __future__ import annotations

# gangnam / sinchon / new10000 세 코호트의 데이터 분포(baseline characteristics)를 한 표로 만든다.
# 모집단은 모델이 실제로 쓴 것과 같게 맞춘다 - clinic4_logistic_regression.load_data를 그대로 써서
# clinic4(나이/신장/체중) 결측자를 동일하게 제외한 뒤 요약한다.
# 연속형은 mean ± SD, 범주형/유병률은 n (%). 코호트가 3개라 검정은 Kruskal-Wallis(연속형, 정규성 가정
# 없음)와 chi-square(범주형)를 쓴다. CKD는 new10000에 컬럼이 없어 두 코호트만으로 검정한다.
# 단위 표기는 붙이지 않는다 - 랩 데이터셋 규약(ct-dataset-conventions)의 컬럼 단위 항목이 아직 비어 있어
# 추측해서 쓰지 않는다. 규약이 채워지면 LABELS에 단위를 추가할 것.

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from scipy import stats

from clinic4_logistic_regression import CLINIC4_DIR, DATA_XLSX, EXTERNAL_COHORTS, load_data, save_sheet

sys.stdout.reconfigure(encoding="utf-8")

OUT_PNG = CLINIC4_DIR / "data_distribution.png"
SHEET = "distribution"

CONTINUOUS = ["PatientAge", "Height", "Weight", "BMI", "SMI", "VAT", "SAT", "IMATA", "NAMA", "LAMA", "TAMA"]
DISEASES = ["HTN", "DM", "CKD"]
# 그림에 넣을 4개(clinic4 입력 3개 + 대표 체성분 1개)
PLOT_COLS = ["PatientAge", "Height", "Weight", "BMI"]
LABELS = {"PatientAge": "Age", "Height": "Height", "Weight": "Weight", "BMI": "BMI", "SMI": "SMI"}
COHORT_TITLE = {"gangnam": "Gangnam", "sinchon": "Sinchon", "new10000": "New10000"}


def format_p(p: float) -> str:
    if np.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# 코호트별 값 목록으로 연속형 한 행(mean ± SD)을 만들고 Kruskal-Wallis로 비교. 컬럼이 없는 코호트는 "-"
def continuous_row(label: str, values: dict[str, pd.Series | None]) -> dict:
    present = [v.astype(float).dropna() for v in values.values() if v is not None]
    p = stats.kruskal(*present).pvalue if len(present) > 1 else float("nan")
    row = {"Characteristic": label}
    for cohort, v in values.items():
        row[COHORT_TITLE[cohort]] = "-" if v is None else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
    row["p-value"] = format_p(p)
    return row


# 코호트별 0/1 시리즈로 이진 변수 한 행(n (%))을 만들고 chi-square로 비교
def binary_row(label: str, values: dict[str, pd.Series | None]) -> dict:
    present = [v for v in values.values() if v is not None]
    table = np.array([[int((v == 1).sum()), int((v == 0).sum())] for v in present])
    p = stats.chi2_contingency(table)[1] if len(present) > 1 else float("nan")
    row = {"Characteristic": label}
    for cohort, v in values.items():
        row[COHORT_TITLE[cohort]] = "-" if v is None else f"{int((v == 1).sum())} ({(v == 1).mean():.1%})"
    row["p-value"] = format_p(p)
    return row


# 코호트별 주요 변수 분포를 히스토그램(밀도)으로 겹쳐 그린다 - 표만으로는 안 보이는 분포 모양/치우침 확인용
def save_distribution_plot(cohorts: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(PLOT_COLS), figsize=(12.2, 4.16))
    for ax, col in zip(axes, PLOT_COLS):
        for name, df in cohorts.items():
            if col in df.columns:
                ax.hist(df[col].astype(float).dropna(), bins=40, density=True, histtype="step",
                        linewidth=1.5, label=COHORT_TITLE[name])
        ax.set_title(LABELS.get(col, col))
        ax.set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    cohorts = {"gangnam": load_data(DATA_XLSX)}
    for cohort, path in EXTERNAL_COHORTS.items():
        cohorts[cohort] = load_data(path)

    def column(df: pd.DataFrame, col: str) -> pd.Series | None:
        return df[col] if col in df.columns else None

    rows = [{"Characteristic": "n", **{COHORT_TITLE[c]: str(len(df)) for c, df in cohorts.items()},
             "p-value": ""}]
    rows.append(binary_row("Sex (Male), n (%)",
                           {c: (df["PatientSex"].astype(str).str.upper() == "M").astype(int)
                            for c, df in cohorts.items()}))
    rows += [continuous_row(LABELS.get(col, col), {c: column(df, col) for c, df in cohorts.items()})
             for col in CONTINUOUS]
    rows += [binary_row(f"{d}, n (%)", {c: column(df, d) for c, df in cohorts.items()}) for d in DISEASES]

    table = pd.DataFrame(rows)
    save_sheet(table, "data_distribution.xlsx", SHEET)
    save_distribution_plot(cohorts)
    print(table.to_string(index=False))
    print(f"\nSaved sheet '{SHEET}' to data_distribution.xlsx and {OUT_PNG.name} in {CLINIC4_DIR}")

    # self-check: 모델이 쓴 모집단과 같은 n인지(load_data를 공유하므로 어긋나면 로딩 규칙이 갈라진 것)
    assert all(len(df) > 0 for df in cohorts.values()), "빈 코호트가 있음"
    assert table["Characteristic"].is_unique, "중복된 행 라벨"
    print("OK: 세 코호트 모두 로드됨 (모델과 동일한 load_data 기준)")


if __name__ == "__main__":
    main()
