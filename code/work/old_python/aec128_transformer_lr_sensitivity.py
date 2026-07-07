from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec128_transformer_offset import Config, SEED, run_config, set_seed  # noqa: E402


BASE_OUT = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_transformer_offset"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_transformer_lr_sensitivity"


CONFIGS = [
    Config(
        name="tiny_pair100_lr0002",
        d_model=16,
        dropout=0.20,
        lr=2e-4,
        max_epochs=280,
        patience=70,
        lambda_pair=1.00,
        lambda_score=1e-3,
        lambda_alpha=1e-4,
        lambda_noise=0.0,
        noise_sd=0.0,
        alpha_init=0.75,
    ),
]


def add_delta_columns(perf: pd.DataFrame) -> pd.DataFrame:
    """설정x데이터셋별로 임상 단독과 결합모델의 AUC/로그손실/조건부페어AUC를 나란히 놓고 차이(delta)를 계산."""
    rows = []
    for (config, dataset), sub in perf.groupby(["config", "dataset"]):
        clinical = sub[sub["model_type"].eq("clinical")].iloc[0]
        combined = sub[sub["model_type"].eq("combined")].iloc[0]
        row = {
            "config": config,
            "dataset": dataset,
            "clinical_auc": float(clinical["auc"]),
            "combined_auc": float(combined["auc"]),
            "delta_auc": float(combined["auc"] - clinical["auc"]),
            "clinical_log_loss": float(clinical["log_loss"]),
            "combined_log_loss": float(combined["log_loss"]),
            "delta_log_loss": float(combined["log_loss"] - clinical["log_loss"]),
            "clinical_conditional_pair_auc": float(clinical["conditional_pair_auc"]),
            "combined_conditional_pair_auc": float(combined["conditional_pair_auc"]),
            "delta_conditional_pair_auc": float(combined["conditional_pair_auc"] - clinical["conditional_pair_auc"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_lr_comparison(current_hist: pd.DataFrame, slow_hist: pd.DataFrame) -> None:
    """학습률 8e-4(기존)와 2e-4(느림) 두 설정의 학습손실/검증로그손실/검증 delta AUC 곡선을 폴드 평균으로 겹쳐 그려 PNG로 저장."""
    current_hist = current_hist[current_hist["config"].eq("tiny_pair100_fast")].copy()
    slow_hist = slow_hist[slow_hist["config"].eq("tiny_pair100_lr0002")].copy()
    current_cv = current_hist[current_hist["fold"].astype(str).isin(["1", "2", "3", "4", "5"])].copy()
    slow_cv = slow_hist[slow_hist["fold"].astype(str).isin(["1", "2", "3", "4", "5"])].copy()

    def mean_curve(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.groupby("epoch", as_index=False)[col].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    panels = [
        ("train_loss", "Training total loss", "Loss"),
        ("valid_log_loss", "Validation log loss", "Log loss"),
        ("valid_delta_auc_vs_clinical", "Validation delta AUC", "AUC(combined) - AUC(clinical)"),
    ]
    for ax, (col, title, ylabel) in zip(axes, panels):
        a = mean_curve(current_cv, col)
        b = mean_curve(slow_cv, col)
        ax.plot(a["epoch"], a[col], lw=2.2, color="#E45756", label="lr=8e-4")
        ax.plot(b["epoch"], b[col], lw=2.2, color="#4C78A8", label="lr=2e-4")
        if col == "valid_delta_auc_vs_clinical":
            ax.axhline(0, color="#555555", lw=1.0, ls="--")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("AEC Transformer Learning Rate Sensitivity", x=0.01, y=1.02, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_transformer_lr_sensitivity_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec128_transformer_offset의 결과가 학습률(lr) 선택에
    민감한 것은 아닌가? — 더 느린 학습률로 다시 돌려 결과가 비슷한지 확인하는 강건성 점검):

    1. g1090/sdata를 로드하고, 학습률만 8e-4→2e-4로 낮추고 에폭/patience를 늘린 설정으로
       run_config(aec128_transformer_offset에서 재사용)를 실행한다.
    2. 새 설정의 성능·폴드·학습이력·부트스트랩·순열검정 결과를 각각 CSV로 저장.
    3. 기존(빠른 lr) 성능 CSV를 읽어와 새 결과와 합친 뒤, add_delta_columns로 두 lr 설정 모두에서
       "결합모델 - 임상 단독"의 AUC/로그손실/조건부페어AUC 개선폭을 비교하는 표를 만들어 CSV로 저장.
    4. 기존 학습이력과 새 학습이력을 함께 불러와, 두 학습률의 학습손실/검증손실/검증 delta AUC
       곡선을 겹쳐 그려 PNG로 저장 (학습률에 따라 수렴 양상이 달라지는지 확인).
    5. 비교 결과와 산출물 경로를 JSON으로 저장하고, delta 비교표와 부트스트랩 결과를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train = load_aec128(FILES["g1090"])
    test = load_aec128(FILES["sdata"])
    train["y"] = train["y"].astype(int)
    test["y"] = test["y"].astype(int)

    all_perf = []
    all_folds = []
    all_history = []
    all_boot = []
    all_perm = []
    summaries = []

    for cfg in CONFIGS:
        result = run_config(cfg, train, test)
        perf = result["performance"].copy()
        perf["config"] = cfg.name
        perf["model_type"] = np.where(perf["model"].str.contains("plus_aec"), "combined", "clinical")
        all_perf.append(perf)
        all_folds.append(result["folds"])
        all_history.append(result["history"])
        all_boot.append(result["bootstrap"])
        all_perm.append(result["permutation"])
        summaries.append(result["summary"])
        result["pred_oof"].to_csv(OUT_DIR / f"{cfg.name}_g1090_oof_predictions.csv", index=False)
        result["pred_external"].to_csv(OUT_DIR / f"{cfg.name}_sdata_external_predictions.csv", index=False)

    perf_df = pd.concat(all_perf, ignore_index=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    hist_df = pd.concat(all_history, ignore_index=True)
    boot_df = pd.concat(all_boot, ignore_index=True)
    perm_df = pd.concat(all_perm, ignore_index=True)

    perf_df.to_csv(OUT_DIR / "aec128_transformer_lr_sensitivity_performance.csv", index=False)
    fold_df.to_csv(OUT_DIR / "aec128_transformer_lr_sensitivity_fold_summary.csv", index=False)
    hist_df.to_csv(OUT_DIR / "aec128_transformer_lr_sensitivity_training_history.csv", index=False)
    boot_df.to_csv(OUT_DIR / "aec128_transformer_lr_sensitivity_bootstrap_delta_auc.csv", index=False)
    perm_df.to_csv(OUT_DIR / "aec128_transformer_lr_sensitivity_permutation_alignment.csv", index=False)

    baseline_perf = pd.read_csv(BASE_OUT / "aec128_transformer_performance.csv")
    comparison_perf = pd.concat([baseline_perf, perf_df], ignore_index=True)
    comparison_delta = add_delta_columns(comparison_perf)
    comparison_delta.to_csv(OUT_DIR / "aec128_transformer_lr_sensitivity_delta_summary.csv", index=False)

    baseline_hist = pd.read_csv(BASE_OUT / "aec128_transformer_training_history.csv")
    plot_lr_comparison(baseline_hist, hist_df)

    summary = {
        "seed": SEED,
        "configs": summaries,
        "comparison_delta_csv": str(OUT_DIR / "aec128_transformer_lr_sensitivity_delta_summary.csv"),
        "curve_plot": str(OUT_DIR / "aec128_transformer_lr_sensitivity_curves.png"),
    }
    (OUT_DIR / "aec128_transformer_lr_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Delta summary")
    print(comparison_delta)
    print("\nSlow LR bootstrap")
    print(boot_df)
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
