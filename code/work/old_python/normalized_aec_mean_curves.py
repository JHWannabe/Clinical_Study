from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aec_conditional_value import DATA_DIR, aec_columns, matrix_from_sheet, resample_rows, row_norm


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "normalized_aec_mean_curves"
FILES = {
    "g1090": DATA_DIR / "g1090.xlsx",
    "sdata": DATA_DIR / "sdata.xlsx",
}


def load_curves(path: Path) -> dict:
    """엑셀 파일에서 저근감소증 라벨과, 정규화(자기 평균 대비 비율)된 a128/crop 곡선을 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    smi = tama / (height_m**2)
    low_smi = np.where(sex == "M", smi < 45.4, smi < 34.4)

    out = {"low_smi": low_smi.astype(bool)}
    for sheet in ["aec_128", "aec_cropped"]:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        mat = resample_rows(matrix_from_sheet(df), 128)
        out[sheet] = row_norm(mat)
    return out


def mean_ci(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """위치별 평균과 95% 신뢰구간(정규근사)을 계산."""
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)
    se = sd / np.sqrt(mat.shape[0])
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi


def build_summary(datasets: dict) -> pd.DataFrame:
    """코호트 x 곡선 x (저근감소증/비저근감소증) 조합별로 위치마다 평균·신뢰구간을 표로 정리."""
    rows = []
    for cohort, data in datasets.items():
        y = data["low_smi"]
        for sheet in ["aec_128", "aec_cropped"]:
            pos = np.linspace(0, 1, data[sheet].shape[1])
            for group_name, mask in [("low_smi", y), ("non_low_smi", ~y)]:
                mean, lo, hi = mean_ci(data[sheet][mask])
                for i, x in enumerate(pos):
                    rows.append(
                        {
                            "cohort": cohort,
                            "curve": sheet,
                            "group": group_name,
                            "position_0_to_1": x,
                            "point_index_1_to_128": i + 1,
                            "n": int(mask.sum()),
                            "mean_normalized_aec": mean[i],
                            "ci95_low": lo[i],
                            "ci95_high": hi[i],
                        }
                    )
    return pd.DataFrame(rows)


def pooled_dataset(datasets: dict) -> dict:
    """g1090과 sdata 두 코호트를 하나로 합친 풀링 데이터셋을 만듦."""
    pooled = {"low_smi": np.concatenate([d["low_smi"] for d in datasets.values()])}
    for sheet in ["aec_128", "aec_cropped"]:
        pooled[sheet] = np.vstack([d[sheet] for d in datasets.values()])
    return pooled


def plot_mean_curves(datasets: dict) -> None:
    """g1090/sdata/pooled x a128/crop 6개 패널로, 저근감소증군과 비저근감소증군의 평균 곡선(+신뢰구간)을 겹쳐 그려 PNG로 저장."""
    plot_sets = {**datasets, "pooled": pooled_dataset(datasets)}
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.4), sharex=True, sharey="row")
    colors = {"low_smi": "#C84630", "non_low_smi": "#2F6F73"}
    labels = {"low_smi": "Low SMI", "non_low_smi": "Non-low SMI"}
    sheets = [("aec_128", "AEC 128"), ("aec_cropped", "AEC cropped")]

    for col, (cohort, data) in enumerate(plot_sets.items()):
        y = data["low_smi"]
        for row, (sheet, sheet_label) in enumerate(sheets):
            ax = axes[row, col]
            x = np.linspace(0, 1, data[sheet].shape[1])
            for group_name, mask in [("low_smi", y), ("non_low_smi", ~y)]:
                mean, lo, hi = mean_ci(data[sheet][mask])
                ax.plot(x, mean, lw=2.2, color=colors[group_name], label=f"{labels[group_name]} (n={int(mask.sum())})")
                ax.fill_between(x, lo, hi, color=colors[group_name], alpha=0.16, linewidth=0)
            ax.axhline(1.0, color="#555555", lw=0.9, ls="--", alpha=0.7)
            ax.grid(alpha=0.22)
            ax.set_title(f"{cohort} - {sheet_label}", loc="left", fontsize=11, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Normalized z-axis position")
            if col == 0:
                ax.set_ylabel("AEC / patient mean AEC")
            ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle("Patient-normalized AEC mean curves by low SMI status", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "patient_normalized_aec_mean_curves.png", dpi=200)
    plt.close(fig)


def plot_difference_curves(datasets: dict) -> None:
    """코호트별로 저근감소증군 평균 곡선에서 비저근감소증군 평균 곡선을 뺀 차이 곡선을 그려 PNG로 저장 (코호트 간 패턴이 일관되는지 확인)."""
    plot_sets = {**datasets, "pooled": pooled_dataset(datasets)}
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
    colors = {"g1090": "#4C78A8", "sdata": "#F58518", "pooled": "#333333"}
    sheets = [("aec_128", "AEC 128"), ("aec_cropped", "AEC cropped")]

    for ax, (sheet, sheet_label) in zip(axes, sheets):
        for cohort, data in plot_sets.items():
            y = data["low_smi"]
            x = np.linspace(0, 1, data[sheet].shape[1])
            low_mean = np.nanmean(data[sheet][y], axis=0)
            non_mean = np.nanmean(data[sheet][~y], axis=0)
            ax.plot(x, low_mean - non_mean, lw=2.0, color=colors[cohort], label=cohort)
        ax.axhline(0.0, color="#555555", lw=0.9, ls="--", alpha=0.75)
        ax.grid(alpha=0.24)
        ax.set_title(f"{sheet_label}: Low SMI mean - non-low SMI mean", loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Difference in normalized AEC")
        ax.legend(frameon=False, ncol=3, loc="best")
    axes[-1].set_xlabel("Normalized z-axis position")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "patient_normalized_aec_low_minus_nonlow.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 순수하게 시각적으로, 저근감소증군과 비저근감소증군의
    평균 AEC 곡선 모양이 코호트 간에 일관되게 다른가?):

    1. g1090/sdata 각각 load_curves로 라벨과 정규화된 a128/crop 곡선을 읽는다.
    2. build_summary로 코호트x곡선x그룹별 위치별 평균/신뢰구간 표를 만들어 CSV로 저장.
    3. plot_mean_curves로 g1090/sdata/pooled 각각의 두 곡선에 대해 두 그룹 평균±95%CI를 겹쳐
       그린 6패널 그래프를 저장.
    4. plot_difference_curves로 각 코호트의 "저근감소증 평균 - 비저근감소증 평균" 차이 곡선을
       한 그래프에 겹쳐 그려, 코호트 간에 차이 패턴이 비슷한지 시각적으로 비교.
    5. 코호트별 표본수/이벤트수/유병률을 CSV로 저장하고 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_curves(path) for name, path in FILES.items()}
    summary = build_summary({**datasets, "pooled": pooled_dataset(datasets)})
    summary.to_csv(OUT_DIR / "patient_normalized_aec_mean_curves.csv", index=False)
    plot_mean_curves(datasets)
    plot_difference_curves(datasets)

    counts = []
    for cohort, data in {**datasets, "pooled": pooled_dataset(datasets)}.items():
        y = data["low_smi"]
        counts.append(
            {
                "cohort": cohort,
                "n": int(len(y)),
                "low_smi": int(y.sum()),
                "non_low_smi": int((~y).sum()),
                "low_smi_rate": float(y.mean()),
            }
        )
    pd.DataFrame(counts).to_csv(OUT_DIR / "cohort_counts.csv", index=False)
    print(pd.DataFrame(counts).to_string(index=False))
    print(OUT_DIR / "patient_normalized_aec_mean_curves.png")
    print(OUT_DIR / "patient_normalized_aec_low_minus_nonlow.png")
    print(OUT_DIR / "patient_normalized_aec_mean_curves.csv")


if __name__ == "__main__":
    main()
