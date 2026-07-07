from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    SEED,
    aec_estimator,
    binary_metrics,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    make_folds,
    oof_and_external,
    threshold_youden,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec_threshold_tradeoff"


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 임상 점수의 "양성 판정 경계"를 어디로 잡느냐에 따라,
    여성 AEC 게이트를 켰을 때 민감도 손실과 특이도 이득이 어떻게 변하는가?):

    1. aec_conditional_value의 함수들로 train(g1090)/test(sdata)를 로드하고, 임상 단독 모델과
       AEC 단독(SVM) 모델을 학습해 각각의 표준화된 점수(c_z, a_z)를 얻는다.
    2. 실제 채택된 임상 Youden 임계값(selected_t)을 표시용으로 계산해두고, c_z 분포의 2~98
       분위수 구간을 220개 격자점(t 후보)으로 스캔한다.
    3. 각 t 후보마다: t를 "양성 경계"로 삼아 (a) 가우시안 가중치로 부드럽게 AEC를 얹는 게이트와
       (b) 경계 ±0.50 안에서만 AEC를 딱 잘라 얹는 게이트, 두 버전을 계산하고, 임상 단독 대비
       민감도 손실(sensitivity loss)/특이도 이득(specificity gain)/FP·TP 변화량을 기록한다.
    4. 220개 결과를 표로 만들어 CSV로 저장하고, 실제 채택된 임계값에 가장 가까운 행을 찾아
       별도 CSV로 저장한다.
    5. (a) 경계 위치에 따른 민감도손실/특이도이득 곡선과 임상/게이트 민감도·특이도 곡선을 함께
       그린 2단 그래프, (b) 민감도손실 대비 특이도이득의 트레이드오프 곡선(현재 임계값 위치 강조),
       두 개의 PNG를 저장한다.
    6. 저장된 파일 경로와 채택된 임계값에서의 핵심 지표를 콘솔에 출력한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]

    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    folds = make_folds(ytr, 5)
    clinical_oof, clinical_test = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)
    aec_oof, aec_test = oof_and_external(
        lambda seed: aec_estimator(train["aec"].shape[1], seed),
        train["aec"],
        ytr,
        test["aec"],
        folds,
    )

    c_z, c_te_z, _, _ = zfit_apply(clinical_oof, clinical_test)
    a_z, a_te_z, _, _ = zfit_apply(aec_oof, aec_test)
    female_tr = train["sex"] == "F"
    female_te = test["sex"] == "F"

    selected_t = float((threshold_youden(ytr, clinical_oof) - np.mean(clinical_oof)) / np.std(clinical_oof))
    grid = np.linspace(np.quantile(c_z, 0.02), np.quantile(c_z, 0.98), 220)

    rows = []
    for t in grid:
        boundary_tr = np.exp(-0.5 * ((c_z - t) / 0.75) ** 2)
        boundary_te = np.exp(-0.5 * ((c_te_z - t) / 0.75) ** 2)

        gaussian_train = c_z + 0.25 * female_tr * boundary_tr * a_z
        gaussian_test = c_te_z + 0.25 * female_te * boundary_te * a_te_z

        hard_train = c_z + 0.25 * female_tr * (np.abs(c_z - t) <= 0.50) * a_z
        hard_test = c_te_z + 0.25 * female_te * (np.abs(c_te_z - t) <= 0.50) * a_te_z

        clinical = binary_metrics(yte, c_te_z, t)
        gaussian = binary_metrics(yte, gaussian_test, t)
        hard = binary_metrics(yte, hard_test, t)

        train_clinical = binary_metrics(ytr, c_z, t)
        rows.append(
            {
                "clinical_threshold_z": float(t),
                "train_clinical_sensitivity": train_clinical["sensitivity"],
                "external_clinical_sensitivity": clinical["sensitivity"],
                "external_clinical_specificity": clinical["specificity"],
                "external_clinical_ppv": clinical["ppv"],
                "external_clinical_positive_n": clinical["tp"] + clinical["fp"],
                "gaussian_external_sensitivity": gaussian["sensitivity"],
                "gaussian_external_specificity": gaussian["specificity"],
                "gaussian_external_ppv": gaussian["ppv"],
                "gaussian_sensitivity_loss": clinical["sensitivity"] - gaussian["sensitivity"],
                "gaussian_specificity_gain": gaussian["specificity"] - clinical["specificity"],
                "gaussian_fp_change": gaussian["fp"] - clinical["fp"],
                "gaussian_tp_change": gaussian["tp"] - clinical["tp"],
                "hard_external_sensitivity": hard["sensitivity"],
                "hard_external_specificity": hard["specificity"],
                "hard_external_ppv": hard["ppv"],
                "hard_sensitivity_loss": clinical["sensitivity"] - hard["sensitivity"],
                "hard_specificity_gain": hard["specificity"] - clinical["specificity"],
                "hard_fp_change": hard["fp"] - clinical["fp"],
                "hard_tp_change": hard["tp"] - clinical["tp"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "clinical_threshold_tradeoff.csv", index=False)

    sel_idx = int(np.argmin(np.abs(df["clinical_threshold_z"] - selected_t)))
    selected_row = df.iloc[sel_idx].to_dict()
    pd.DataFrame([selected_row]).to_csv(OUT_DIR / "selected_threshold_tradeoff_row.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True)

    ax = axes[0]
    ax.plot(df["clinical_threshold_z"], df["gaussian_sensitivity_loss"], lw=2.4, color="#C84630", label="Sensitivity loss")
    ax.plot(df["clinical_threshold_z"], df["gaussian_specificity_gain"], lw=2.4, color="#2F6F73", label="Specificity gain")
    ax.axhline(0, color="#777777", lw=0.8)
    ax.axvline(selected_t, color="#333333", lw=1.4, ls="--", label="Current clinical Youden threshold")
    ax.set_ylabel("AEC gate - clinical tradeoff")
    ax.set_title("Gaussian female-boundary AEC gate")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(df["clinical_threshold_z"], df["external_clinical_sensitivity"], lw=2.0, color="#4C78A8", label="Clinical sensitivity")
    ax.plot(df["clinical_threshold_z"], df["external_clinical_specificity"], lw=2.0, color="#72B7B2", label="Clinical specificity")
    ax.plot(df["clinical_threshold_z"], df["gaussian_external_sensitivity"], lw=1.6, color="#C84630", ls="--", label="AEC-gated sensitivity")
    ax.plot(df["clinical_threshold_z"], df["gaussian_external_specificity"], lw=1.6, color="#2F6F73", ls="--", label="AEC-gated specificity")
    ax.axvline(selected_t, color="#333333", lw=1.4, ls="--")
    ax.set_xlabel("Clinical score threshold, standardized on g1090 OOF score")
    ax.set_ylabel("External sdata performance")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.text(
        0.01,
        0.01,
        "Sensitivity loss = clinical sensitivity - AEC-gated sensitivity. Specificity gain = AEC-gated specificity - clinical specificity. "
        "AEC adjustment: 0.25 x female x Gaussian boundary weight x AEC score; decision threshold kept equal to the clinical threshold.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 0.82, 1))
    fig.savefig(OUT_DIR / "clinical_threshold_sensitivity_loss_specificity_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(df["gaussian_sensitivity_loss"], df["gaussian_specificity_gain"], lw=2.0, color="#2F6F73")
    ax.scatter(
        [selected_row["gaussian_sensitivity_loss"]],
        [selected_row["gaussian_specificity_gain"]],
        s=55,
        color="#C84630",
        zorder=3,
        label="Current threshold",
    )
    ax.axhline(0, color="#777777", lw=0.8)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Sensitivity loss")
    ax.set_ylabel("Specificity gain")
    ax.set_title("Tradeoff curve across clinical score thresholds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_loss_vs_specificity_gain_curve.png", dpi=180)
    plt.close(fig)

    print("Saved:")
    print(OUT_DIR / "clinical_threshold_sensitivity_loss_specificity_gain.png")
    print(OUT_DIR / "sensitivity_loss_vs_specificity_gain_curve.png")
    print(OUT_DIR / "clinical_threshold_tradeoff.csv")
    print("Selected threshold row:")
    for key in [
        "clinical_threshold_z",
        "external_clinical_sensitivity",
        "external_clinical_specificity",
        "gaussian_external_sensitivity",
        "gaussian_external_specificity",
        "gaussian_sensitivity_loss",
        "gaussian_specificity_gain",
        "gaussian_fp_change",
        "gaussian_tp_change",
    ]:
        print(f"{key}: {selected_row[key]}")


if __name__ == "__main__":
    main()
