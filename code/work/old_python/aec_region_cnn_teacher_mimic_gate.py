from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    auc_with_p,
    build_candidate_bank,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
    make_single_deesc,
    risk_direction,
    standardize_train_test,
)
from aec_region_constrained_cnn_gate import (  # noqa: E402
    DEVICE,
    REGIONS,
    RegionConstrainedCnn,
    make_channels,
    standardize_channels_train_apply,
    stratified_folds,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_cnn_teacher_mimic_gate"
LOCK_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_lock_smoothed_deesc_gate"
BRANCH_WIDTH = np.array([0.70, 0.50, 0.70, 0.70], dtype=float)
BRANCH_LAMBDA = np.array([0.70, 0.70, 0.55, 0.55], dtype=float)
SEEDS = [20260701, 20260711]


def locked_targets(g: dict, s: dict, c_g: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """aec_lock_smoothed_deesc_gate가 이미 잠근(locked) 4개 특징 값을 읽어와, CNN이 "모방(mimic)"으로 학습할 정답(teacher) 타깃으로 재구성."""
    features = pd.read_csv(LOCK_DIR / "locked_gate_features.csv")["feature"].astype(str).tolist()
    fg = build_candidate_bank(g["norm"])
    fs = build_candidate_bank(s["norm"])
    xg_all, xs_all, names = standardize_train_test(fg, fs)
    name_to_idx = {name: i for i, name in enumerate(names)}
    idx = [name_to_idx[name] for name in features]
    xg = xg_all[:, idx]
    xs = xs_all[:, idx]
    direction = risk_direction(g["y"], c_g, xg)
    return xg * direction, xs * direction, features


def train_fold(
    x: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_ext: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """RegionConstrainedCnn을 분류가 아닌 회귀(smooth L1 손실)로 학습시켜, 손수 만든 4개 잠금 특징 값 자체를 예측하도록 훈련 — CNN이 "선생님(teacher) 특징"을 모방하게 하는 지식증류(distillation) 방식."""
    torch.manual_seed(seed)
    model = RegionConstrainedCnn(dropout=0.15).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-3)
    xt = torch.tensor(x[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(target[train_idx], dtype=torch.float32, device=DEVICE)
    xv = torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(target[val_idx], dtype=torch.float32, device=DEVICE)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    patience = 18
    rng = np.random.default_rng(seed)
    for epoch in range(160):
        order = rng.permutation(len(train_idx))
        model.train()
        for start in range(0, len(order), 96):
            batch = order[start : start + 96]
            opt.zero_grad(set_to_none=True)
            pred, _ = model(xt[batch])
            loss = F.smooth_l1_loss(pred, yt[batch])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_v, _ = model(xv)
            val_loss = float(F.smooth_l1_loss(pred_v, yv).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 18
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred, _ = model(torch.tensor(x[val_idx], dtype=torch.float32, device=DEVICE))
        ext_pred, _ = model(torch.tensor(x_ext, dtype=torch.float32, device=DEVICE))
    return (
        val_pred.detach().cpu().numpy(),
        ext_pred.detach().cpu().numpy(),
        np.asarray([best_loss], dtype=float),
        {"best_epoch": int(best_epoch), "best_val_loss": float(best_loss)},
    )


def crossfit_mimic(xg: np.ndarray, target_g: np.ndarray, y: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """여러 시드 x 5-fold로 train_fold를 반복해, 시드에 걸쳐 평균낸 내부 OOF/외부 모방(mimic) 예측과 학습 로그를 반환."""
    oof_accum = []
    ext_accum = []
    logs = []
    for seed in SEEDS:
        oof = np.zeros_like(target_g, dtype=float)
        exts = []
        for fold_id, (tr, va) in enumerate(stratified_folds(y, seed)):
            vp, ep, _, info = train_fold(xg, target_g, tr, va, xs, seed + fold_id * 101)
            oof[va] = vp
            exts.append(ep)
            logs.append({"seed": seed, "fold": fold_id, **info})
        oof_accum.append(oof)
        ext_accum.append(np.mean(exts, axis=0))
    return np.mean(oof_accum, axis=0), np.mean(ext_accum, axis=0), pd.DataFrame(logs)


def mimic_accuracy(pred_g: np.ndarray, pred_s: np.ndarray, target_g: np.ndarray, target_s: np.ndarray, features: list[str]) -> pd.DataFrame:
    """CNN이 예측한 값이 실제 잠긴(teacher) 특징 값과 각 구간별로 얼마나 상관되는지(상관계수, RMSE)를 내부/외부 각각 계산 — 모방 학습이 실제로 잘 됐는지 검증."""
    rows = []
    for dataset, pred, target in [("g1090_internal", pred_g, target_g), ("sdata_external", pred_s, target_s)]:
        for j, feature in enumerate(features):
            if np.std(pred[:, j]) > 1e-12 and np.std(target[:, j]) > 1e-12:
                corr = float(np.corrcoef(pred[:, j], target[:, j])[0, 1])
            else:
                corr = np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "region": list(REGIONS)[j],
                    "teacher_feature": feature,
                    "corr_pred_vs_teacher": corr,
                    "rmse": float(np.sqrt(np.mean((pred[:, j] - target[:, j]) ** 2))),
                    "pred_sd": float(pred[:, j].std()),
                    "teacher_sd": float(target[:, j].std()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_gate(
    label: str,
    g: dict,
    s: dict,
    c_g: np.ndarray,
    c_s: np.ndarray,
    thresholds: dict[str, float],
    feat_g: np.ndarray,
    feat_s: np.ndarray,
    lambda_scale: float = 1.0,
) -> pd.DataFrame:
    """주어진 4개 특징(정확한 teacher 특징이든 CNN 모방 예측이든)으로 2-of-4 boundary de-escalation 게이트를 내부/외부 x 모든 운영점에서 평가."""
    rows = []
    for dataset, d, clinical_z, feat in [("g1090_internal", g, c_g, feat_g), ("sdata_external", s, c_s, feat_s)]:
        y = d["y"].astype(int)
        for op, _ in OPS:
            th = thresholds[op]
            cpos = clinical_z >= th
            votes = np.zeros((feat.shape[1], len(y)), dtype=np.int8)
            for j in range(feat.shape[1]):
                votes[j] = make_single_deesc(
                    clinical_z,
                    feat[:, j],
                    th,
                    float(BRANCH_WIDTH[j]),
                    float(BRANCH_LAMBDA[j] * lambda_scale),
                ).astype(np.int8)
            deesc = cpos & (votes.sum(axis=0) >= 2)
            rows.append(
                deesc_metric_row(
                    dataset,
                    label,
                    "locked_four_features_or_CNN_mimic",
                    op,
                    y,
                    cpos,
                    deesc,
                )
                | {"lambda_scale": float(lambda_scale)}
            )
    return pd.DataFrame(rows)


def summarize_internal(rows: pd.DataFrame) -> dict:
    """여러 운영점 결과 중 가장 나쁜 경우를 뽑아 안전성 제약 판정용 요약통계로 압축."""
    return {
        "internal_min_p_loss": float(rows["sensitivity_loss_p_exact"].min(skipna=True)),
        "internal_max_sens_loss": float(rows["sensitivity_loss"].max(skipna=True)),
        "internal_min_spec_gain": float(rows["specificity_gain"].min(skipna=True)),
        "internal_mean_spec_gain": float(rows["specificity_gain"].mean(skipna=True)),
        "internal_max_fisher_p": float(rows["deesc_event_fisher_p"].max(skipna=True)),
        "internal_min_deesc_n": int(rows["deesc_n"].min(skipna=True)),
        "internal_mean_event_rate": float(rows["deesc_event_rate"].mean(skipna=True)),
    }


def select_mimic_scale(detail_all: pd.DataFrame) -> pd.DataFrame:
    """CNN 모방 특징을 쓸 때, 여러 람다 배율(lambda_scale) 후보 중 g1090 내부에서 안전성 제약을 통과하며 점수가 가장 높은 배율을 선택."""
    rows = []
    for scale in sorted(detail_all["lambda_scale"].unique()):
        gi = detail_all[
            detail_all["dataset"].eq("g1090_internal")
            & np.isclose(detail_all["lambda_scale"], float(scale))
        ]
        s = summarize_internal(gi)
        survives = (
            s["internal_min_p_loss"] >= 0.05
            and s["internal_min_spec_gain"] > 0
            and s["internal_max_fisher_p"] < 0.05
            and s["internal_min_deesc_n"] >= 25
            and s["internal_max_sens_loss"] <= 0.08
        )
        score = (
            3.0 * s["internal_min_spec_gain"]
            + 1.3 * s["internal_mean_spec_gain"]
            - 0.9 * s["internal_max_sens_loss"]
            - 0.02 * s["internal_max_fisher_p"]
        )
        if not survives:
            score -= 10.0
        rows.append({"lambda_scale": float(scale), "survives_internal_constraints": survives, "internal_selection_score": score, **s})
    return pd.DataFrame(rows).sort_values(["survives_internal_constraints", "internal_selection_score"], ascending=False)


def clinical_plus_auc(
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    aec_risk_g: np.ndarray,
    aec_risk_s: np.ndarray,
) -> tuple[float, float, float, float]:
    """임상점수와 AEC 특징 평균값을 함께 넣은 로지스틱 결합모델을 5-fold OOF로 학습해, 내부/외부 결합 AUC와 p값을 반환."""
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


def auc_summary(
    label: str,
    g: dict,
    s: dict,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    feat_g: np.ndarray,
    feat_s: np.ndarray,
) -> pd.DataFrame:
    """임상단독/AEC특징평균단독/결합 모델의 내부·외부 AUC와 임상단독 대비 증분을 비교표로 만듦."""
    rows = []
    cg_auc, cg_p = auc_with_p(g["y"], clinical_oof)
    cs_auc, cs_p = auc_with_p(s["y"], clinical_ext)
    risk_g = feat_g.mean(axis=1)
    risk_s = feat_s.mean(axis=1)
    ag_auc, ag_p = auc_with_p(g["y"], risk_g)
    as_auc, as_p = auc_with_p(s["y"], risk_s)
    pg_auc, pg_p, ps_auc, ps_p = clinical_plus_auc(g, s, clinical_oof, clinical_ext, risk_g, risk_s)
    rows.extend(
        [
            {
                "source": label,
                "model": "clinical_only",
                "internal_auc": cg_auc,
                "internal_auc_p": cg_p,
                "external_auc": cs_auc,
                "external_auc_p": cs_p,
                "internal_delta_vs_clinical": 0.0,
                "external_delta_vs_clinical": 0.0,
            },
            {
                "source": label,
                "model": "AEC_feature_mean_only",
                "internal_auc": ag_auc,
                "internal_auc_p": ag_p,
                "external_auc": as_auc,
                "external_auc_p": as_p,
                "internal_delta_vs_clinical": ag_auc - cg_auc,
                "external_delta_vs_clinical": as_auc - cs_auc,
            },
            {
                "source": label,
                "model": "clinical_plus_AEC_feature_mean",
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


def plot_result(detail: pd.DataFrame, acc: pd.DataFrame, out_path: Path) -> None:
    """정확한 teacher 특징 규칙과 CNN 모방 규칙의 특이도이득/민감도손실을 비교하고(2패널), 구간별 CNN 모방 정확도(외부 상관)를 막대그래프로(1패널) 그려 PNG로 저장."""
    labels = [op for op, _ in OPS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {"exact_teacher_features": "#2c7fb8", "cnn_teacher_mimic": "#d95f02"}
    for source in ["exact_teacher_features", "cnn_teacher_mimic"]:
        for dataset, ls in [("g1090_internal", "-"), ("sdata_external", "--")]:
            sub = detail[detail["rule"].eq(source) & detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
            x = np.arange(len(labels))
            axes[0].plot(x, sub["specificity_gain"] * 100, marker="o", ls=ls, color=colors[source], label=f"{source} {dataset}")
            axes[1].plot(x, sub["sensitivity_loss"] * 100, marker="x", ls=ls, color=colors[source], label=f"{source} {dataset}")
    for ax, title, ylab in [
        (axes[0], "Specificity gain", "Percentage points"),
        (axes[1], "Sensitivity loss", "Percentage points"),
    ]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    sub = acc[acc["dataset"].eq("sdata_external")]
    axes[2].bar(np.arange(len(sub)), sub["corr_pred_vs_teacher"], color="#756bb1")
    axes[2].set_xticks(np.arange(len(sub)))
    axes[2].set_xticklabels([f"R{i + 1}" for i in range(len(sub))])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("External corr")
    axes[2].set_title("CNN mimic accuracy", loc="left", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 이미 사전에 잠근(locked) 4개의 손수 만든 특징을, CNN이
    "선생님(teacher)"의 값을 그대로 예측하도록 지식증류 방식으로 학습시키면, CNN 예측값으로도
    똑같은 boundary de-escalation 게이트를 재현할 수 있는가? — 완전히 새로운 CNN 발견이 아니라
    "기존 잠금 규칙을 CNN으로 근사(모방)했을 때의 신뢰도" 검증):

    1. g1090/sdata를 로드하고, 잠긴 4개 특징 값(locked_targets)을 회귀 타깃으로 삼아 CNN(회귀)을
       여러 시드 x 5-fold로 학습(crossfit_mimic)해 내부 OOF/외부 모방 예측을 얻음.
    2. 정확한 teacher 특징 규칙(lambda_scale=1.0)과, CNN 모방 예측 x 여러 람다 배율 조합의 2-of-4
       boundary 게이트를 모두 평가.
    3. g1090 내부에서 안전성 제약을 통과하며 점수가 가장 높은 CNN 모방 람다 배율을 선택
       (select_mimic_scale)하고, 정확한 규칙 vs 최선의 CNN 모방 규칙을 나란히 비교.
    4. CNN이 각 구간의 teacher 특징 값을 얼마나 정확히 예측했는지(상관/RMSE)와, 임상단독/AEC단독/결합
       모델의 AUC 비교표도 계산.
    5. 결과를 그래프로 시각화하고, 상세/전체스케일/스케일선택/AUC/정확도/학습로그를 CSV로, 설정을
       JSON으로 저장한 뒤 콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    target_g, target_s, features = locked_targets(g, s, c_g)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))
    print("training teacher mimic", flush=True)
    pred_g, pred_s, log_df = crossfit_mimic(xg, target_g, g["y"], xs)

    lambda_scales = np.round(np.arange(0.20, 1.201, 0.05), 2)
    detail_exact = evaluate_gate("exact_teacher_features", g, s, c_g, c_s, thresholds, target_g, target_s, 1.0)
    detail_mimic_all = pd.concat(
        [
            evaluate_gate("cnn_teacher_mimic", g, s, c_g, c_s, thresholds, pred_g, pred_s, float(scale))
            for scale in lambda_scales
        ],
        ignore_index=True,
    )
    scale_summary = select_mimic_scale(detail_mimic_all)
    best_scale = float(scale_summary.iloc[0]["lambda_scale"])
    detail_mimic_best = detail_mimic_all[np.isclose(detail_mimic_all["lambda_scale"], best_scale)].copy()
    detail = pd.concat([detail_exact, detail_mimic_best], ignore_index=True)
    auc = pd.concat(
        [
            auc_summary("exact_teacher_features", g, s, clinical_oof, clinical_ext, target_g, target_s),
            auc_summary("cnn_teacher_mimic", g, s, clinical_oof, clinical_ext, pred_g, pred_s),
        ],
        ignore_index=True,
    )
    acc = mimic_accuracy(pred_g, pred_s, target_g, target_s, features)
    detail.to_csv(OUT_DIR / "teacher_mimic_deescalation_details.csv", index=False)
    detail_mimic_all.to_csv(OUT_DIR / "teacher_mimic_deescalation_all_scales.csv", index=False)
    scale_summary.to_csv(OUT_DIR / "teacher_mimic_scale_selection_summary.csv", index=False)
    auc.to_csv(OUT_DIR / "teacher_mimic_auc_summary.csv", index=False)
    acc.to_csv(OUT_DIR / "teacher_mimic_accuracy.csv", index=False)
    log_df.to_csv(OUT_DIR / "teacher_mimic_training_log.csv", index=False)
    plot_result(detail, acc, OUT_DIR / "teacher_mimic_plot.png")
    with (OUT_DIR / "teacher_mimic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing": "aec_128, gaussian smoothing sigma=1, patient-wise mean normalization",
                "teacher_features": features,
                "regions_1_indexed_inclusive": REGIONS,
                "mimic_lambda_scales_searched_internal_only": lambda_scales.tolist(),
                "best_mimic_lambda_scale_by_internal_only": best_scale,
                "rule": "Broad-ROI CNN regresses the four locked risk-oriented AEC features; final de-escalation remains conditional boundary 2-of-4.",
            },
            f,
            indent=2,
        )

    print("\nMIMIC ACCURACY")
    print(acc.to_string(index=False))
    print("\nAUC SUMMARY")
    print(auc.to_string(index=False))
    print("\nMIMIC SCALE SUMMARY")
    print(scale_summary.to_string(index=False))
    print("\nDE-ESCALATION")
    show = [
        "rule",
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
    print(detail[show].to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    # 데이터 로드 -> 잠긴 4개 특징을 타깃으로 CNN 회귀 학습(teacher mimic) -> 정확한 규칙 vs 여러
    # 람다 배율의 CNN 모방 규칙을 평가해 내부 기준 최적 배율 선택 -> 모방 정확도/AUC 비교 및 결과
    # 저장 순으로 실행된다.
    main()
