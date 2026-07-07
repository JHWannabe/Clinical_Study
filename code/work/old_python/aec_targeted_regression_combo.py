from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    adjusted_deesc_p,
    bootstrap_metrics,
    build_candidate_bank,
    clinical_scores,
    gate_metrics,
    load_aec128,
    standardize_train_test,
)
from aec_regression_combo_gate import (  # noqa: E402
    direct_clinical_plus_aec_auc,
    fit_oof_external_score,
    sensitivity_loss_exact_p,
    zfit,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_targeted_regression_combo"
SPEC70_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_specificity70_target_search"
PRIMARY_OPS = ["youden", "sens80", "sens85"]
ALL_OPS = ["youden", "sens80", "sens85", "sens90", "sens95"]


def feature_priority(row: pd.Series) -> float:
    """최소 최종특이도, 최소균형이득, 최대민감도손실, 최대 Fisher p값을 가중합해 특징의 우선순위 점수를 계산."""
    return (
        2.0 * row["primary_min_rule_specificity"]
        + 0.8 * row["primary_min_balanced_gain"]
        - 0.8 * row["primary_max_sens_loss"]
        - 0.01 * np.log10(np.clip(row["primary_max_fisher_p"], 1e-12, 1.0))
    )


def get_targeted_feature_sets(names: list[str]) -> dict[str, list[int]]:
    """aec_specificity70_target_search가 저장해둔 표에서 특이도70%+민감도손실7.5%이하 조건을
    만족하는 특징들만 우선순위로 정렬해, 상위 2~80개씩 묶은 여러 "타겟된" 특징 집합을 만들고,
    이전 실행에서 강했던 "초반 곡률" 특징군도 참고용 별도 집합으로 추가."""
    train = pd.read_csv(SPEC70_DIR / "specificity_target_train_candidates_all.csv")
    train = train[
        (train["primary_min_rule_specificity"] >= 0.70)
        & (train["primary_max_sens_loss"] <= 0.075)
        & (train["primary_min_balanced_gain"] > 0)
        & (train["train_primary_min_deesc_n"] >= 20)
        & (train["primary_max_fisher_p"] <= 0.05)
    ].copy()
    train["priority"] = train.apply(feature_priority, axis=1)
    unique = train.sort_values("priority", ascending=False).drop_duplicates("feature")
    name_to_idx = {name: i for i, name in enumerate(names)}
    ordered = [name_to_idx[f] for f in unique["feature"].tolist() if f in name_to_idx]
    sets = {}
    for k in [2, 3, 5, 8, 12, 20, 40, 80]:
        if len(ordered) >= k:
            sets[f"targeted_top{k}"] = ordered[:k]
    # The strongest external single-feature family from the previous run, kept as
    # a biologically interpretable sensitivity set. It is not used for strict selection.
    early_names = [
        "bank_norm__norm_curv_010_025_max",
        "bank_norm__norm_curv_010_021_max",
        "bank_norm__norm_curv_010_025_min",
        "bank_norm__norm_curv_013_016_sd",
        "bank_norm__log_slope_013_016_sd",
        "bank_norm__norm_slope_013_016_sd",
        "bank_norm__norm_curv_013_016_min",
        "midrange__visual_norm_slope_013_022_sd",
    ]
    early_idx = [name_to_idx[n] for n in early_names if n in name_to_idx]
    if early_idx:
        sets["early_curvature_family_reference"] = early_idx
    pd.DataFrame(
        [{"rank": i + 1, "feature": names[idx]} for i, idx in enumerate(ordered[:200])]
    ).to_csv(OUT_DIR / "targeted_feature_order_top200.csv", index=False)
    return sets


def eval_grid(y_g, y_s, c_g, c_s, a_g, a_s, thresholds, meta):
    """한 모델 점수에 대해 게이트폭 3종 x 람다 5종 x 5개 운영점 조합 전체에서 gate_metrics 성능을 계산."""
    rows = []
    for width in [0.35, 0.50, 0.70]:
        for lam in [0.25, 0.40, 0.55, 0.75, 1.00]:
            for dataset, y, c, a in [("g1090_oof", y_g, c_g, a_g), ("sdata_external", y_s, c_s, a_s)]:
                for op in ALL_OPS:
                    m = gate_metrics(y, c, a, thresholds[op]["clinical_z"], width, lam)
                    rows.append({**meta, "dataset": dataset, "width": width, "lambda": lam, "operating_point": op, **m})
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """모델설정별로 3개 주요 운영점의 지표를 하나의 요약 행으로 압축."""
    key = ["model_id", "feature_set", "n_features", "C", "class_weight", "width", "lambda", "dataset"]
    rows = []
    for keys, sub in df[df["operating_point"].isin(PRIMARY_OPS)].groupby(key, dropna=False):
        rows.append(
            {
                **dict(zip(key, keys)),
                "primary_min_rule_specificity": float(sub["rule_specificity"].min()),
                "primary_avg_rule_specificity": float(sub["rule_specificity"].mean()),
                "primary_avg_spec_gain": float(sub["specificity_gain"].mean()),
                "primary_avg_sens_loss": float(sub["sensitivity_loss"].mean()),
                "primary_max_sens_loss": float(sub["sensitivity_loss"].max()),
                "primary_min_balanced_gain": float(sub["balanced_gain"].min()),
                "primary_avg_deesc_prevalence": float(sub["deesc_prevalence"].mean()),
                "primary_max_fisher_p": float(sub["fisher_p"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_selected(eval_df: pd.DataFrame, row: pd.Series, label: str) -> None:
    """지정된 모델의 운영점별 (임상 vs 결합모델 특이도) 막대그래프와 (특이도이득 vs 민감도손실) 막대그래프를 나란히 그려 PNG로 저장."""
    ext = eval_df[
        (eval_df["model_id"].eq(row["model_id"]))
        & (eval_df["width"].eq(float(row["width"])))
        & (eval_df["lambda"].eq(float(row["lambda"])))
        & (eval_df["dataset"].eq("sdata_external"))
    ].copy()
    ext = ext.set_index("operating_point").loc[ALL_OPS].reset_index()
    x = np.arange(len(ext))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7))
    axes[0].bar(x - w / 2, ext["clinical_specificity"] * 100, w, color="#8DA0CB", label="Clinical")
    axes[0].bar(x + w / 2, ext["rule_specificity"] * 100, w, color="#4DAF4A", label="Clinical + targeted regression gate")
    axes[0].axhline(70, color="#333333", lw=1.1, ls="--", label="70%")
    axes[0].set_ylabel("Specificity (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ext["operating_point"].tolist())
    axes[0].grid(axis="y", alpha=0.24)
    axes[0].legend(frameon=False)
    axes[1].bar(x - w / 2, ext["specificity_gain"] * 100, w, color="#4DAF4A", label="Specificity gain")
    axes[1].bar(x + w / 2, ext["sensitivity_loss"] * 100, w, color="#D95F02", label="Sensitivity loss")
    axes[1].set_ylabel("Percentage points")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ext["operating_point"].tolist())
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend(frameon=False)
    fig.suptitle(f"{label}: {row['model_id']}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{label}_{row['model_id']}_tradeoff.png", dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_specificity70_target_search가 골라둔 "특이도70%
    달성 가능" 특징들만 모아 릿지 회귀로 결합하면, 단일 특징보다 더 안정적으로 특이도70%+낮은
    민감도손실을 external에서도 재현하는 결합모델을 만들 수 있는가?):

    1. g1090(train)/sdata(external)를 로드하고 임상점수·후보 특징뱅크·표준화값을 계산.
    2. get_targeted_feature_sets로, specificity70_target_search 결과표에서 조건(특이도≥70%,
       민감도손실≤7.5%, 균형이득>0, deesc 표본≥20, Fisher p≤0.05)을 만족하는 특징을 우선순위로
       정렬해 상위 2~80개씩 묶은 8개 "타겟된" 특징집합 + 참고용 "초반 곡률" 특징집합을 구성.
    3. 각 특징집합 x C값(0.03~1.00) x class_weight(None/balanced) 총 최대 64개 릿지 회귀 설정에
       대해, train에서 OOF 점수·external 점수를 학습(fit_oof_external_score)하고 표준화(zfit)한 뒤,
       게이트폭 3종 x 람다 5종 x 5개 운영점 전체 조합에서 gate_metrics 성능을 계산(eval_grid).
    4. 각 모델설정을 3개 주요 운영점 기준으로 요약(summarize)하고, train/외부 요약을 병합해
       "train_score"(train 성능 가중합) 기준으로 정렬.
    5. train 조건(특이도≥70%, 민감도손실≤7.5%, 균형이득>0)을 만족하는 최상위 모델을 "strict_train
       _selected"로, 더 느슨한 조건(민감도손실≤10%)에서 external 성능이 가장 좋은 모델을
       "external_reference"로 선택하고, 각각의 운영점별 트레이드오프 그래프를 저장.
    6. 두 선택 모델에 대해 민감도손실의 정확 이항검정 p값, 스캐너(제조사) 보정 하향조정 유의성
       (임상점수 포함/미포함), 부트스트랩 신뢰구간, 임상+AEC 결합 직접 AUC를 계산해 모두 CSV/JSON
       으로 저장하고 콘솔에 요약 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    y_g = g["y"].astype(int)
    y_s = s["y"].astype(int)
    feature_sets = get_targeted_feature_sets(names)

    rows = []
    score_store = {}
    mid = 0
    for set_name, idx_list in feature_sets.items():
        idx = np.asarray(idx_list, dtype=int)
        for c in [0.03, 0.10, 0.30, 1.00]:
            for class_weight in [None, "balanced"]:
                mid += 1
                model_id = f"targeted_ridge_{mid:03d}"
                oof, ext, _ = fit_oof_external_score(xg, y_g, xs, idx, c, class_weight)
                a_g, a_s = zfit(oof, ext)
                if roc_auc_score(y_g, a_g) < 0.5:
                    a_g = -a_g
                    a_s = -a_s
                meta = {
                    "model_id": model_id,
                    "feature_set": set_name,
                    "n_features": len(idx),
                    "C": c,
                    "class_weight": class_weight,
                }
                score_store[model_id] = {"g": a_g, "s": a_s, "meta": meta}
                rows.extend(eval_grid(y_g, y_s, c_g, c_s, a_g, a_s, thresholds, meta))

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(OUT_DIR / "targeted_regression_gate_all_models.csv", index=False)
    summary = summarize(eval_df)
    train = summary[summary["dataset"].eq("g1090_oof")].copy()
    ext = summary[summary["dataset"].eq("sdata_external")].copy()
    merged = train.merge(
        ext,
        on=["model_id", "feature_set", "n_features", "C", "class_weight", "width", "lambda"],
        suffixes=("_g1090", "_sdata"),
    )
    merged["train_score"] = (
        2.0 * merged["primary_min_rule_specificity_g1090"]
        + 0.8 * merged["primary_avg_rule_specificity_g1090"]
        + 0.8 * merged["primary_min_balanced_gain_g1090"]
        - 0.8 * merged["primary_avg_sens_loss_g1090"]
    )
    merged["external_score"] = (
        2.0 * merged["primary_min_rule_specificity_sdata"]
        + 0.8 * merged["primary_avg_rule_specificity_sdata"]
        + 0.8 * merged["primary_min_balanced_gain_sdata"]
        - 0.8 * merged["primary_avg_sens_loss_sdata"]
    )
    merged = merged.sort_values("train_score", ascending=False)
    merged.to_csv(OUT_DIR / "targeted_regression_primary_summary.csv", index=False)

    strict_pool = merged[
        (merged["primary_min_rule_specificity_g1090"] >= 0.70)
        & (merged["primary_max_sens_loss_g1090"] <= 0.075)
        & (merged["primary_min_balanced_gain_g1090"] > 0)
    ]
    strict = strict_pool.sort_values("train_score", ascending=False).iloc[0] if not strict_pool.empty else merged.iloc[0]
    ref_pool = merged[
        (merged["primary_min_rule_specificity_g1090"] >= 0.70)
        & (merged["primary_max_sens_loss_g1090"] <= 0.10)
        & (merged["primary_min_balanced_gain_g1090"] > 0)
    ]
    reference = (ref_pool if not ref_pool.empty else merged).sort_values("external_score", ascending=False).iloc[0]

    selected = []
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        plot_selected(eval_df, row, label)
        detail = eval_df[
            (eval_df["model_id"].eq(row["model_id"]))
            & (eval_df["width"].eq(float(row["width"])))
            & (eval_df["lambda"].eq(float(row["lambda"])))
        ].copy()
        detail.insert(0, "selection_type", label)
        selected.append(detail)
    selected_eval = pd.concat(selected, ignore_index=True)
    p_rows = []
    for _, r in selected_eval[selected_eval["dataset"].eq("sdata_external")].iterrows():
        p_rows.append(
            {
                "selection_type": r["selection_type"],
                "model_id": r["model_id"],
                "operating_point": r["operating_point"],
                "sensitivity_loss_exact_two_sided_p": sensitivity_loss_exact_p(int(r["tp_lost"])),
            }
        )
    selected_eval = selected_eval.merge(pd.DataFrame(p_rows), on=["selection_type", "model_id", "operating_point"], how="left")
    selected_eval.to_csv(OUT_DIR / "selected_targeted_regression_eval.csv", index=False)

    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(s["y"]))).astype(str).to_numpy()
    adj_rows = []
    boot_rows = []
    auc_rows = []
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        a_g = score_store[row["model_id"]]["g"]
        a_s = score_store[row["model_id"]]["s"]
        for include_clinical in [False, True]:
            adj = adjusted_deesc_p(y_s, c_s, a_s, scanner_s, thresholds, float(row["width"]), float(row["lambda"]), include_clinical=include_clinical)
            adj.insert(0, "selection_type", label)
            adj.insert(1, "model_id", row["model_id"])
            adj_rows.append(adj)
        boot = bootstrap_metrics(y_s, c_s, a_s, thresholds, float(row["width"]), float(row["lambda"]), n_boot=2000)
        boot.insert(0, "selection_type", label)
        boot.insert(1, "model_id", row["model_id"])
        boot_rows.append(boot)
        auc_rows.append({"selection_type": label, "model_id": row["model_id"], **direct_clinical_plus_aec_auc(y_g, y_s, c_g, c_s, a_g, a_s)})
    pd.concat(adj_rows, ignore_index=True).to_csv(OUT_DIR / "selected_targeted_regression_adjusted_pvalues.csv", index=False)
    pd.concat(boot_rows, ignore_index=True).to_csv(OUT_DIR / "selected_targeted_regression_bootstrap.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(OUT_DIR / "selected_targeted_regression_direct_auc.csv", index=False)
    (OUT_DIR / "targeted_regression_summary.json").write_text(
        json.dumps(
            {
                "n_feature_sets": len(feature_sets),
                "strict_train_selected": strict.to_dict(),
                "external_reference": reference.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Strict")
    print(strict.to_string())
    print("\nReference")
    print(reference.to_string())
    print("\nSelected external")
    print(selected_eval[selected_eval["dataset"].eq("sdata_external")].to_string(index=False))


if __name__ == "__main__":
    main()
