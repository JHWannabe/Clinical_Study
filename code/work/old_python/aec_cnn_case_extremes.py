from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_cnn_case_extremes"
SCORE_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "0701"
    / "aec_image_cnn_preprocessing_sweep"
    / "cnn_preprocessing_scores.csv"
)
MODE = "mean_norm_no_smooth"
N_EXTREME = 12


def load_dataset(path: Path) -> dict:
    """엑셀 파일에서 정규화된 AEC_128 곡선과 라벨, 메타데이터를 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    norm = row_norm(raw)
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "norm": norm, "y": y}


def residual_and_derivative(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CNN 입력과 동일한 방식(양끝 연결 직선 대비 잔차, 1차 도함수)으로 곡선을 변환해 시각화용으로 재현."""
    z = np.linspace(0.0, 1.0, x.shape[1])
    line = x[:, [0]] * (1.0 - z) + x[:, [-1]] * z
    resid = x - line
    d1 = np.diff(x, axis=1)
    d1 = np.column_stack([d1[:, :1], d1])
    return resid, d1


def add_index(scores: pd.DataFrame) -> pd.DataFrame:
    """전처리방식x코호트 그룹 내에서 순서대로 0부터 매긴 row_index를 추가해 원본 데이터 행과 다시 연결할 수 있게 함."""
    out = scores.copy()
    out["row_index"] = out.groupby(["preprocessing", "cohort"]).cumcount()
    return out


def extreme_table(scores: pd.DataFrame, data_by_cohort: dict[str, dict]) -> pd.DataFrame:
    """이전에 저장된 CNN 점수 중, 코호트별로 점수가 가장 높은/낮은 12명씩을 뽑아 실제 라벨·임상변수와 함께 표로 정리 (CNN이 무엇을 근거로 판단하는지 사례 검토용)."""
    rows = []
    sub = scores[scores["preprocessing"].eq(MODE)].copy()
    for cohort in ["Gangnam", "Sinchon"]:
        cohort_scores = sub[sub["cohort"].eq(cohort)].copy()
        for group_name, selected in [
            ("cnn_high_low_smi_likely", cohort_scores.nlargest(N_EXTREME, "aec_image_cnn")),
            ("cnn_low_low_smi_unlikely", cohort_scores.nsmallest(N_EXTREME, "aec_image_cnn")),
        ]:
            meta = data_by_cohort[cohort]["meta"]
            for rank, (_, r) in enumerate(selected.iterrows(), start=1):
                i = int(r["row_index"])
                m = meta.iloc[i]
                rows.append(
                    {
                        "preprocessing": MODE,
                        "cohort": cohort,
                        "cnn_group": group_name,
                        "within_group_rank": rank,
                        "row_index": i,
                        "true_low_smi": int(r["y"]),
                        "cnn_score": float(r["aec_image_cnn"]),
                        "clinical_score": float(r["clinical"]),
                        "sex": str(m.get("PatientSex", "")),
                        "age": float(m.get("PatientAge", np.nan)),
                        "height": float(m.get("Height", np.nan)),
                        "weight": float(m.get("Weight", np.nan)),
                        "bmi": float(m.get("BMI", np.nan)),
                        "smi": float(m.get("SMI", np.nan)),
                        "manufacturer": str(m.get("Manufacturer", "")),
                    }
                )
    return pd.DataFrame(rows)


def group_indices(table: pd.DataFrame, cohort: str, group: str) -> np.ndarray:
    """코호트x그룹(CNN 고점/저점)에 해당하는 원본 데이터 행 인덱스들을 뽑아냄."""
    return table[(table["cohort"].eq(cohort)) & (table["cnn_group"].eq(group))]["row_index"].to_numpy(dtype=int)


def plot_group_contrast(table: pd.DataFrame, data_by_cohort: dict[str, dict], out_path: Path) -> None:
    """코호트x3채널(원곡선/잔차/도함수) 2x3 그리드에서, CNN 고점군 vs 저점군의 평균±표준오차 곡선을 겹쳐 그려 PNG로 저장."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True)
    xs = np.arange(1, 129)
    channels = ["normalized curve", "centerline residual", "first derivative"]
    colors = {"high": "#b2182b", "low": "#2166ac"}
    for row, cohort in enumerate(["Gangnam", "Sinchon"]):
        x = data_by_cohort[cohort]["norm"]
        resid, d1 = residual_and_derivative(x)
        mats = [x, resid, d1]
        high_idx = group_indices(table, cohort, "cnn_high_low_smi_likely")
        low_idx = group_indices(table, cohort, "cnn_low_low_smi_unlikely")
        for col, mat in enumerate(mats):
            ax = axes[row, col]
            for idx, label, color in [(high_idx, "CNN high", colors["high"]), (low_idx, "CNN low", colors["low"])]:
                mean = mat[idx].mean(axis=0)
                se = mat[idx].std(axis=0, ddof=1) / np.sqrt(len(idx))
                ax.plot(xs, mean, color=color, lw=2.1, label=label)
                ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)
            ax.axhline(0 if col > 0 else 1, color="0.65", lw=0.8, ls="--")
            ax.axvspan(45, 76, color="0.9", alpha=0.35, linewidth=0)
            ax.axvspan(85, 118, color="0.8", alpha=0.22, linewidth=0)
            ax.set_title(f"{cohort}: {channels[col]}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel("AEC value")
            ax.grid(alpha=0.18)
            if row == 0 and col == 2:
                ax.legend(frameon=False, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("craniocaudal AEC index: 1 inferior pubic margin -> 128 liver dome")
    fig.suptitle("What the mean-normalized image CNN scored as high vs low low-SMI likelihood", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_individual_extremes(table: pd.DataFrame, data_by_cohort: dict[str, dict], out_path: Path) -> None:
    """코호트x(CNN 고점/저점) 4개 패널에서 개별 환자 곡선을 실제 라벨 색상으로 겹쳐 그리는 스파게티 플롯을 PNG로 저장."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    xs = np.arange(1, 129)
    panels = [
        ("Gangnam", "cnn_high_low_smi_likely", "Gangnam CNN high"),
        ("Gangnam", "cnn_low_low_smi_unlikely", "Gangnam CNN low"),
        ("Sinchon", "cnn_high_low_smi_likely", "Sinchon CNN high"),
        ("Sinchon", "cnn_low_low_smi_unlikely", "Sinchon CNN low"),
    ]
    for ax, (cohort, group, title) in zip(axes, panels):
        d = data_by_cohort[cohort]
        sub = table[(table["cohort"].eq(cohort)) & (table["cnn_group"].eq(group))].copy()
        for _, r in sub.iterrows():
            i = int(r["row_index"])
            color = "#b2182b" if int(r["true_low_smi"]) else "#2166ac"
            ax.plot(xs, d["norm"][i], color=color, alpha=0.55, lw=1.1)
        ax.plot(xs, d["norm"][sub["row_index"].to_numpy(dtype=int)].mean(axis=0), color="black", lw=2.4)
        event_rate = sub["true_low_smi"].mean()
        ax.set_title(f"{title} | true low SMI {sub['true_low_smi'].sum()}/{len(sub)} ({event_rate:.0%})", fontsize=10, fontweight="bold")
        ax.axhline(1.0, color="0.65", lw=0.8, ls="--")
        ax.axvspan(45, 76, color="0.9", alpha=0.35, linewidth=0)
        ax.axvspan(85, 118, color="0.8", alpha=0.22, linewidth=0)
        ax.set_ylabel("norm AEC")
        ax.grid(alpha=0.18)
        ax.text(0.995, 0.08, "red=true low, blue=true non-low", transform=ax.transAxes, ha="right", fontsize=8, color="0.25")
    axes[-1].set_xlabel("craniocaudal AEC index: 1 inferior pubic margin -> 128 liver dome")
    fig.suptitle("Top and bottom 12 cases ranked by AEC image CNN score", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_case_grid(table: pd.DataFrame, data_by_cohort: dict[str, dict], cohort: str, group: str, out_path: Path) -> None:
    """한 코호트x그룹의 12개 개별 환자 곡선을 3x4 그리드로 각각 따로 그려 PNG로 저장 (사례별 상세 검토용)."""
    sub = table[(table["cohort"].eq(cohort)) & (table["cnn_group"].eq(group))].copy()
    d = data_by_cohort[cohort]
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.8), sharex=True, sharey=True)
    xs = np.arange(1, 129)
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        i = int(r["row_index"])
        color = "#b2182b" if int(r["true_low_smi"]) else "#2166ac"
        ax.plot(xs, d["norm"][i], color=color, lw=1.7)
        ax.axhline(1.0, color="0.7", lw=0.8, ls="--")
        ax.axvspan(45, 76, color="0.9", alpha=0.35, linewidth=0)
        ax.axvspan(85, 118, color="0.8", alpha=0.22, linewidth=0)
        ax.set_title(
            f"row {i} | y={int(r['true_low_smi'])} | CNN {r['cnn_score']:.2f}",
            fontsize=8,
            color=color,
        )
        ax.grid(alpha=0.15)
    fig.suptitle(f"{cohort} {group.replace('_', ' ')}: individual normalized AEC curves", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summarize_groups(table: pd.DataFrame) -> pd.DataFrame:
    """코호트x그룹별 표본수, 실제 저근감소증 발생률, CNN/임상 점수 평균, SMI/BMI 평균, 여성 비율을 요약."""
    return (
        table.groupby(["cohort", "cnn_group"])
        .agg(
            n=("row_index", "size"),
            true_low_smi_n=("true_low_smi", "sum"),
            true_low_smi_rate=("true_low_smi", "mean"),
            cnn_score_mean=("cnn_score", "mean"),
            clinical_score_mean=("clinical_score", "mean"),
            smi_mean=("smi", "mean"),
            bmi_mean=("bmi", "mean"),
            female_rate=("sex", lambda s: float(np.mean(s.astype(str).str.upper().eq("F")))),
        )
        .reset_index()
    )


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_image_cnn_preprocessing_sweep에서 CNN이 매긴 점수가
    가장 높은/낮은 환자들은 실제로 어떤 곡선 모양과 임상 특성을 가지고 있는가? — CNN 블랙박스를
    사례 기반으로 들여다보는 정성적 검토):

    1. g1090/sdata 원본 곡선을 로드하고, 이전에 저장된 CNN 점수 CSV(mean_norm_no_smooth 모드)를
       불러와 원본 행과 다시 연결한다.
    2. extreme_table로 코호트별 CNN 점수 상위/하위 12명씩을 뽑아, 실제 라벨·임상변수와 함께 표로 정리.
    3. plot_group_contrast로 고점군 vs 저점군의 평균 곡선(원곡선/잔차/도함수 3채널)을 비교하는 그래프,
       plot_individual_extremes로 개별 환자 곡선을 겹친 스파게티 플롯, plot_case_grid로 각 사례를
       따로 보여주는 그리드까지 3종류의 시각화를 PNG로 저장.
    4. 그룹별 요약통계(사건율, 점수평균, SMI/BMI, 여성비율)를 계산해 CSV로 저장.
    5. 그룹 요약과 상위/하위 표를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_by_cohort = {
        "Gangnam": load_dataset(DATA_DIR / "g1090.xlsx"),
        "Sinchon": load_dataset(DATA_DIR / "sdata.xlsx"),
    }
    scores = add_index(pd.read_csv(SCORE_PATH))
    table = extreme_table(scores, data_by_cohort)
    summary = summarize_groups(table)
    table.to_csv(OUT_DIR / "cnn_top_bottom_cases_mean_norm_no_smooth.csv", index=False)
    summary.to_csv(OUT_DIR / "cnn_top_bottom_group_summary.csv", index=False)
    plot_group_contrast(table, data_by_cohort, OUT_DIR / "cnn_high_vs_low_group_contrast.png")
    plot_individual_extremes(table, data_by_cohort, OUT_DIR / "cnn_top_bottom_individual_spaghetti.png")
    for cohort in ["Gangnam", "Sinchon"]:
        for group in ["cnn_high_low_smi_likely", "cnn_low_low_smi_unlikely"]:
            plot_case_grid(table, data_by_cohort, cohort, group, OUT_DIR / f"{cohort.lower()}_{group}_case_grid.png")

    print("\nGROUP SUMMARY")
    print(summary.to_string(index=False))
    print("\nTOP/BOTTOM TABLE")
    print(table[["cohort", "cnn_group", "within_group_rank", "row_index", "true_low_smi", "cnn_score", "clinical_score", "sex", "age", "bmi", "smi", "manufacturer"]].to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 CNN 점수 상위/하위 환자를 뽑아 곡선 형태와 임상 특성을 비교하는
    # 사례 기반 정성적 검토 파이프라인(표/그래프/요약통계 생성)이 수행된다.
    main()
