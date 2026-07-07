from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    auc_with_p,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
    make_single_deesc,
)
from aec_region_constrained_cnn_gate import (  # noqa: E402
    CONFIGS,
    REGIONS,
    crossfit_config,
    clinical_positive_weights,
    make_channels,
    standardize_channels_train_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_cnn_boundary_gate"

# These are the locked hand-crafted gate parameters, now applied to CNN branch
# risk features rather than exact window statistics.
BRANCH_WIDTH = np.array([0.70, 0.50, 0.70, 0.70], dtype=float)
BRANCH_LAMBDA = np.array([0.70, 0.70, 0.55, 0.55], dtype=float)


def logit_z_as_risk(train_prob: np.ndarray, test_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CNN branch 확률을 로짓 변환 후 훈련 기준 z-표준화하고 부호를 뒤집어, "값이 클수록 저SMI 위험이 크다"는 기존 boundary 게이트의 관례에 맞는 위험점수로 변환."""
    eps = 1e-5
    tg = np.log(np.clip(train_prob, eps, 1 - eps) / np.clip(1 - train_prob, eps, 1 - eps))
    ts = np.log(np.clip(test_prob, eps, 1 - eps) / np.clip(1 - test_prob, eps, 1 - eps))
    mu = tg.mean(axis=0, keepdims=True)
    sd = tg.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    # CNN branches were trained so higher probability means non-low morphology.
    # The boundary gate expects a higher feature value to increase low-SMI risk.
    return -((tg - mu) / sd), -((ts - mu) / sd)


def summarize_internal(rows: list[dict]) -> dict:
    """여러 운영점 결과 중 가장 나쁜 경우(최소 p값, 최대 민감도손실 등)를 뽑아 안전성 제약 판정용 요약통계로 압축."""
    return {
        "internal_min_p_loss": float(np.nanmin([r["sensitivity_loss_p_exact"] for r in rows])),
        "internal_max_sens_loss": float(np.nanmax([r["sensitivity_loss"] for r in rows])),
        "internal_min_spec_gain": float(np.nanmin([r["specificity_gain"] for r in rows])),
        "internal_mean_spec_gain": float(np.nanmean([r["specificity_gain"] for r in rows])),
        "internal_max_fisher_p": float(np.nanmax([r["deesc_event_fisher_p"] for r in rows])),
        "internal_min_deesc_n": int(np.nanmin([r["deesc_n"] for r in rows])),
        "internal_mean_event_rate": float(np.nanmean([r["deesc_event_rate"] for r in rows])),
    }


def evaluate_boundary_gate(
    config_name: str,
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    thresholds: dict[str, float],
    branch_risk_g: np.ndarray,
    branch_risk_s: np.ndarray,
    lambda_scale: float,
) -> pd.DataFrame:
    """각 CNN 구간(branch)의 위험점수를 손수 정한 폭/람다(BRANCH_WIDTH/BRANCH_LAMBDA, 람다는 lambda_scale로 배율 조정)로 개별 de-escalation 게이트에 넣어 투표시키고, 2개 이상 동의하면 강등하는 규칙을 내부/외부 x 모든 운영점에서 평가."""
    rows = []
    for dataset, d, clinical_z, branch_risk in [
        ("g1090_internal", g, c_g, branch_risk_g),
        ("sdata_external", s, c_s, branch_risk_s),
    ]:
        y = d["y"].astype(int)
        for op, _ in OPS:
            th = thresholds[op]
            cpos = clinical_z >= th
            votes = np.zeros((branch_risk.shape[1], len(y)), dtype=np.int8)
            for j in range(branch_risk.shape[1]):
                votes[j] = make_single_deesc(
                    clinical_z,
                    branch_risk[:, j],
                    th,
                    float(BRANCH_WIDTH[j]),
                    float(BRANCH_LAMBDA[j] * lambda_scale),
                ).astype(np.int8)
            deesc = cpos & (votes.sum(axis=0) >= 2)
            rows.append(
                deesc_metric_row(
                    dataset,
                    "cnn_boundary_2-of-4",
                    "broad_ROI_CNN_branches",
                    op,
                    y,
                    cpos,
                    deesc,
                )
                | {
                    "config": config_name,
                    "lambda_scale": float(lambda_scale),
                    "mean_votes_among_cpos": float(votes[:, cpos].sum(axis=0).mean()) if cpos.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def clinical_plus_auc(
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    aec_risk_g: np.ndarray,
    aec_risk_s: np.ndarray,
) -> tuple[float, float, float, float]:
    """임상점수와 CNN 구간 평균 위험점수를 함께 넣은 로지스틱 결합모델을 5-fold OOF로 학습해, 내부/외부 결합 AUC와 p값을 반환."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260701)
    oof = np.zeros(len(g["y"]), dtype=float)
    for fold_id, (tr, va) in enumerate(folds.split(np.zeros(len(g["y"])), g["y"])):
        model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260701 + fold_id)
        model.fit(np.column_stack([clinical_oof[tr], aec_risk_g[tr]]), g["y"][tr])
        oof[va] = model.decision_function(np.column_stack([clinical_oof[va], aec_risk_g[va]]))
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=20260799)
    model.fit(np.column_stack([clinical_oof, aec_risk_g]), g["y"])
    ext = model.decision_function(np.column_stack([clinical_ext, aec_risk_s]))
    ig_auc, ig_p = auc_with_p(g["y"], oof)
    es_auc, es_p = auc_with_p(s["y"], ext)
    return ig_auc, ig_p, es_auc, es_p


def auc_table(
    config_name: str,
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    branch_risk_g: np.ndarray,
    branch_risk_s: np.ndarray,
) -> pd.DataFrame:
    """임상단독/CNN구간위험점수평균단독/결합 모델의 내부·외부 AUC와 임상단독 대비 증분을 비교표로 만듦."""
    rows = []
    cg_auc, cg_p = auc_with_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_with_p(s["y"], clinical_ext)
    aec_risk_g = branch_risk_g.mean(axis=1)
    aec_risk_s = branch_risk_s.mean(axis=1)
    ag_auc, ag_p = auc_with_p(g["y"], aec_risk_g)
    as_auc, as_p = auc_with_p(s["y"], aec_risk_s)
    pg_auc, pg_p, ps_auc, ps_p = clinical_plus_auc(g, s, clinical_oof, clinical_ext, aec_risk_g, aec_risk_s)
    rows.extend(
        [
            {
                "config": config_name,
                "model": "clinical_only",
                "internal_auc": cg_auc,
                "internal_auc_p": cg_p,
                "external_auc": cs_auc,
                "external_auc_p": cs_p,
                "internal_delta_vs_clinical": 0.0,
                "external_delta_vs_clinical": 0.0,
            },
            {
                "config": config_name,
                "model": "cnn_branch_risk_aec_only",
                "internal_auc": ag_auc,
                "internal_auc_p": ag_p,
                "external_auc": as_auc,
                "external_auc_p": as_p,
                "internal_delta_vs_clinical": ag_auc - cg_auc,
                "external_delta_vs_clinical": as_auc - cs_auc,
            },
            {
                "config": config_name,
                "model": "clinical_plus_cnn_branch_risk",
                "internal_auc": pg_auc,
                "internal_auc_p": pg_p,
                "external_auc": ps_auc,
                "external_auc_p": ps_p,
                "internal_delta_vs_clinical": pg_auc - cg_auc,
                "external_delta_vs_clinical": ps_auc - cs_auc,
            },
        ]
    )
    return pd.DataFrame(rows)


def branch_summary(config_name: str, g: dict, s: dict, branch_risk_g: np.ndarray, branch_risk_s: np.ndarray) -> pd.DataFrame:
    """각 구간의 위험점수가 저SMI군과 비-저SMI군에서 평균적으로 얼마나 다른지 비교해 방향이 임상적으로 말이 되는지 확인하는 표를 만듦."""
    rows = []
    for dataset, d, x in [("g1090_internal", g, branch_risk_g), ("sdata_external", s, branch_risk_s)]:
        y = d["y"].astype(bool)
        for j, region in enumerate(REGIONS):
            rows.append(
                {
                    "config": config_name,
                    "dataset": dataset,
                    "region": region,
                    "low_mean_risk_z": float(x[y, j].mean()),
                    "nonlow_mean_risk_z": float(x[~y, j].mean()),
                    "diff_low_minus_nonlow": float(x[y, j].mean() - x[~y, j].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_result(best_detail: pd.DataFrame, best_branch: pd.DataFrame, out_path: Path) -> None:
    """운영점별 특이도이득/민감도손실, 강등군 사건율, 구간별 위험방향 차이를 3패널 그래프로 그려 PNG로 저장."""
    labels = [op for op, _ in OPS]
    colors = {"g1090_internal": "#2c7fb8", "sdata_external": "#d95f02"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    for dataset in ["g1090_internal", "sdata_external"]:
        sub = best_detail[best_detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", color=colors[dataset], label=f"{dataset} spec gain")
        axes[0].plot(x, sub["sensitivity_loss"] * 100, marker="x", color=colors[dataset], ls="--", label=f"{dataset} sens loss")
        axes[1].plot(x, sub["deesc_event_rate"] * 100, marker="o", color=colors[dataset], label=dataset)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Percentage points")
    axes[0].set_title("Conditional boundary gate", loc="left", fontweight="bold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].axhline(10, color="gray", lw=0.8, ls=":")
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Low SMI among de-escalated (%)")
    axes[1].set_title("De-escalated event rate", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    x = np.arange(len(REGIONS))
    width = 0.35
    for i, dataset in enumerate(["g1090_internal", "sdata_external"]):
        sub = best_branch[best_branch["dataset"].eq(dataset)]
        axes[2].bar(x + (i - 0.5) * width, sub["diff_low_minus_nonlow"], width=width, color=colors[dataset], alpha=0.8, label=dataset)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"R{i + 1}" for i in range(len(REGIONS))])
    axes[2].set_ylabel("Risk z: low minus non-low")
    axes[2].set_title("Branch direction", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_region_constrained_cnn_gate에서 얻은 CNN 구간별 위험점수를,
    "이산적 2-of-4 투표"가 아니라 손수 정한 boundary(폭/람다) 게이트에 넣어도 여전히 안전한
    de-escalation이 되는가? — CNN 특징 x 기존 조건부 boundary 게이트 방식의 결합):

    1. g1090/sdata를 로드하고, aec_region_constrained_cnn_gate의 CONFIGS별로 CNN을 다시 학습해
       구간별 branch 확률을 얻은 뒤 위험점수로 변환(logit_z_as_risk).
    2. 람다 배율(lambda_scales, 0.20~1.05)을 폭넓게 바꿔가며 각 설정 x 배율 조합의 2-of-4 boundary
       게이트를 내부+외부 x 모든 운영점에서 평가.
    3. 임상단독/CNN위험점수단독/결합 모델의 AUC 비교표와 구간별 위험방향 요약표도 계산.
    4. g1090 내부 데이터만으로 안전성 제약을 통과하며 점수가 가장 높은 (설정, 람다배율) 조합을 최종
       선택하고, 그 결과를 3패널 그래프로 시각화.
    5. 전체 상세/AUC/구간요약/학습로그/모델선택/최고결과를 CSV로, 실행 설정을 JSON으로 저장한 뒤
       콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
    sample_weight = clinical_positive_weights(c_g, thresholds)
    lambda_scales = np.round(np.arange(0.20, 1.051, 0.05), 2)

    all_auc = []
    all_detail = []
    all_branch = []
    all_logs = []
    score_cache = {}
    for cfg in CONFIGS:
        print(f"training {cfg.name}", flush=True)
        _, branch_g, _, branch_s, log_df = crossfit_config(cfg, xg, g["y"], sample_weight, xs)
        branch_risk_g, branch_risk_s = logit_z_as_risk(branch_g, branch_s)
        detail = pd.concat(
            [
                evaluate_boundary_gate(cfg.name, g, s, c_g, c_s, thresholds, branch_risk_g, branch_risk_s, float(scale))
                for scale in lambda_scales
            ],
            ignore_index=True,
        )
        auc = auc_table(cfg.name, g, s, clinical_oof, clinical_ext, branch_risk_g, branch_risk_s)
        branch = branch_summary(cfg.name, g, s, branch_risk_g, branch_risk_s)
        all_detail.append(detail)
        all_auc.append(auc)
        all_branch.append(branch)
        all_logs.append(log_df.assign(config=cfg.name))
        score_cache[cfg.name] = (detail, branch)

    detail_all = pd.concat(all_detail, ignore_index=True)
    auc_all = pd.concat(all_auc, ignore_index=True)
    branch_all = pd.concat(all_branch, ignore_index=True)
    log_all = pd.concat(all_logs, ignore_index=True)

    summary_rows = []
    for cfg in CONFIGS:
        for scale in lambda_scales:
            gi = detail_all[
                detail_all["config"].eq(cfg.name)
                & detail_all["dataset"].eq("g1090_internal")
                & np.isclose(detail_all["lambda_scale"], float(scale))
            ]
            ssum = summarize_internal(gi.to_dict("records"))
            survives = (
                ssum["internal_min_p_loss"] >= 0.05
                and ssum["internal_min_spec_gain"] > 0
                and ssum["internal_max_fisher_p"] < 0.05
                and ssum["internal_min_deesc_n"] >= 25
                and ssum["internal_max_sens_loss"] <= 0.08
            )
            score = (
                3.0 * ssum["internal_min_spec_gain"]
                + 1.3 * ssum["internal_mean_spec_gain"]
                - 0.9 * ssum["internal_max_sens_loss"]
                - 0.02 * ssum["internal_max_fisher_p"]
            )
            if not survives:
                score -= 10.0
            summary_rows.append(
                {
                    "config": cfg.name,
                    "lambda_scale": float(scale),
                    "survives_internal_constraints": survives,
                    "internal_selection_score": score,
                    **ssum,
                }
            )
    model_summary = pd.DataFrame(summary_rows).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)
    best_config = str(model_summary.iloc[0]["config"])
    best_scale = float(model_summary.iloc[0]["lambda_scale"])
    best_detail = detail_all[detail_all["config"].eq(best_config) & np.isclose(detail_all["lambda_scale"], best_scale)].copy()
    best_branch = branch_all[branch_all["config"].eq(best_config)].copy()

    detail_all.to_csv(OUT_DIR / "cnn_boundary_deescalation_details.csv", index=False)
    auc_all.to_csv(OUT_DIR / "cnn_boundary_auc_summary.csv", index=False)
    branch_all.to_csv(OUT_DIR / "cnn_boundary_branch_summary.csv", index=False)
    log_all.to_csv(OUT_DIR / "cnn_boundary_training_log.csv", index=False)
    model_summary.to_csv(OUT_DIR / "cnn_boundary_model_selection_summary.csv", index=False)
    best_detail.to_csv(OUT_DIR / "cnn_boundary_best_deescalation_details.csv", index=False)
    best_branch.to_csv(OUT_DIR / "cnn_boundary_best_branch_summary.csv", index=False)
    plot_result(best_detail, best_branch, OUT_DIR / "cnn_boundary_best_plot.png")
    with (OUT_DIR / "cnn_boundary_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "regions_1_indexed_inclusive": REGIONS,
                "branch_boundary_width": BRANCH_WIDTH.tolist(),
                "branch_boundary_lambda": BRANCH_LAMBDA.tolist(),
                "lambda_scales_searched_internal_only": lambda_scales.tolist(),
                "rule": "CNN branch risk features inserted into conditional boundary weighting; 2-of-4 votes de-escalate clinical-positive cases.",
                "best_config_by_internal_only": best_config,
                "best_lambda_scale_by_internal_only": best_scale,
            },
            f,
            indent=2,
        )

    print("\nMODEL SUMMARY")
    print(model_summary.to_string(index=False))
    print("\nAUC SUMMARY")
    print(auc_all.to_string(index=False))
    print("\nBEST CONDITIONAL BOUNDARY GATE")
    show = [
        "dataset",
        "operating_point",
        "clinical_sensitivity",
        "post_sensitivity",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "clinical_specificity",
        "post_specificity",
        "specificity_gain",
        "specificity_gain_p_exact",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
    ]
    print(best_detail[show].to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    # 데이터 로드 -> 구간제약 CNN 재학습으로 구간별 위험점수 산출 -> 여러 람다 배율에 대해 2-of-4
    # boundary 게이트 평가 -> 내부 기준 최적 (설정, 람다) 선택 및 AUC/구간요약과 함께 결과 저장 순으로
    # 실행된다.
    main()
