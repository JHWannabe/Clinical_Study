from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, make_folds  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    adjusted_deesc_p,
    bootstrap_metrics,
    build_candidate_bank,
    clinical_scores,
    counts,
    gate_metrics,
    load_aec128,
    standardize_train_test,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_regression_combo_gate"
SEED = 20260630
PRIMARY_OPS = ["youden", "sens80", "sens85"]
ALL_OPS = ["youden", "sens80", "sens85", "sens90", "sens95"]


def conditional_rank(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """임상점수만 넣은 귀무모델 기준 점수검정 통계량으로 모든 특징의 중요도를 계산해, 중요한 순서(내림차순)의 인덱스를 반환."""
    base = sm.add_constant(pd.DataFrame({"clinical_z": clinical_z}), has_constant="add")
    fit = sm.Logit(y.astype(int), base).fit(disp=False, maxiter=1000)
    p = np.asarray(fit.predict(base), dtype=float)
    resid = y.astype(float) - p
    w = p * (1 - p)
    base_np = np.column_stack([np.ones(len(y)), clinical_z])
    beta = np.linalg.pinv(base_np.T @ (w[:, None] * base_np)) @ (base_np.T @ (w[:, None] * x))
    x_res = x - base_np @ beta
    u = x_res.T @ resid
    info = np.sum(w[:, None] * x_res * x_res, axis=0)
    stat = np.where(info > 1e-12, (u * u) / info, 0.0)
    return np.argsort(np.nan_to_num(stat, nan=-np.inf))[::-1]


def univariate_rank(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """임상변수를 고려하지 않고, 라벨과의 단순 상관 크기만으로 특징 중요도 순서(내림차순) 인덱스를 반환."""
    yc = y.astype(float) - np.mean(y)
    score = np.abs(x.T @ yc)
    return np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]


def fit_oof_external_score(
    xg: np.ndarray,
    y: np.ndarray,
    xs: np.ndarray,
    idx: np.ndarray,
    c: float,
    class_weight: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """선택된 특징(idx)만으로 L2 로지스틱 회귀를 5-fold OOF로 학습해 train 점수와 외부 점수를 만들고, 폴드별+최종 계수도 함께 반환."""
    folds = make_folds(y.astype(int), 5)
    oof = np.zeros(len(y), dtype=float)
    all_idx = np.arange(len(y))
    coefs = []
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = LogisticRegression(
            penalty="l2",
            C=c,
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weight,
            random_state=SEED + fold_id,
        )
        model.fit(xg[tr_idx][:, idx], y[tr_idx])
        oof[val_idx] = model.decision_function(xg[val_idx][:, idx])
        coefs.append(model.coef_.ravel())
    final = LogisticRegression(
        penalty="l2",
        C=c,
        solver="lbfgs",
        max_iter=5000,
        class_weight=class_weight,
        random_state=SEED + 99,
    )
    final.fit(xg[:, idx], y)
    ext = final.decision_function(xs[:, idx])
    coefs.append(final.coef_.ravel())
    return oof, ext, np.vstack(coefs)


def zfit(train_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 train·test 점수를 함께 z-표준화."""
    mu = float(np.mean(train_score))
    sd = float(np.std(train_score))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (train_score - mu) / sd, (test_score - mu) / sd


def eval_gate_grid(
    y_g: np.ndarray,
    y_s: np.ndarray,
    c_g: np.ndarray,
    c_s: np.ndarray,
    a_g: np.ndarray,
    a_s: np.ndarray,
    thresholds: dict[str, dict],
    model_meta: dict,
) -> list[dict]:
    """한 모델 점수에 대해 게이트폭 3종 x 람다 4종 x 5개 운영점 조합 전체에서 gate_metrics 성능을 계산."""
    rows = []
    for width in [0.35, 0.50, 0.70]:
        for lam in [0.25, 0.40, 0.55, 0.75]:
            for dataset, y, clinical_z, aec_z in [
                ("g1090_oof", y_g, c_g, a_g),
                ("sdata_external", y_s, c_s, a_s),
            ]:
                for op in ALL_OPS:
                    m = gate_metrics(y, clinical_z, aec_z, thresholds[op]["clinical_z"], width, lam)
                    rows.append(
                        {
                            **model_meta,
                            "dataset": dataset,
                            "width": width,
                            "lambda": lam,
                            "operating_point": op,
                            **m,
                        }
                    )
    return rows


def summarize_primary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """모델설정(랭커/k/C/클래스가중치/폭/람다)별로 3개 주요 운영점의 지표를 하나의 요약 행으로 압축."""
    sub = df[(df["dataset"].eq(dataset)) & (df["operating_point"].isin(PRIMARY_OPS))]
    key_cols = ["model_id", "ranker", "k_features", "C", "class_weight", "width", "lambda"]
    rows = []
    for keys, g in sub.groupby(key_cols, dropna=False):
        rows.append(
            {
                **dict(zip(key_cols, keys)),
                f"{dataset}_primary_min_rule_specificity": float(g["rule_specificity"].min()),
                f"{dataset}_primary_avg_rule_specificity": float(g["rule_specificity"].mean()),
                f"{dataset}_primary_avg_spec_gain": float(g["specificity_gain"].mean()),
                f"{dataset}_primary_avg_sens_loss": float(g["sensitivity_loss"].mean()),
                f"{dataset}_primary_max_sens_loss": float(g["sensitivity_loss"].max()),
                f"{dataset}_primary_min_balanced_gain": float(g["balanced_gain"].min()),
                f"{dataset}_primary_avg_deesc_prevalence": float(g["deesc_prevalence"].mean()),
                f"{dataset}_primary_max_fisher_p": float(g["fisher_p"].max()),
            }
        )
    return pd.DataFrame(rows)


def selection_score(row: pd.Series) -> float:
    """g1090 OOF 기준 최소/평균 최종특이도, 최소균형이득, 평균민감도손실을 가중합해 train 선택 점수를 계산."""
    # Primary target is interpretable specificity improvement in the mid operating range.
    return (
        2.0 * row["g1090_oof_primary_min_rule_specificity"]
        + 0.8 * row["g1090_oof_primary_avg_rule_specificity"]
        + 0.8 * row["g1090_oof_primary_min_balanced_gain"]
        - 0.8 * row["g1090_oof_primary_avg_sens_loss"]
    )


def sensitivity_loss_exact_p(tp_lost: int) -> float:
    """놓친 진양성(tp_lost) 건수에 대한 이항분포 기반 양측 정확검정 p값을 계산 (모두 우연히 하향조정됐다는 귀무가설 하에서)."""
    if tp_lost <= 0:
        return 1.0
    return float(min(1.0, 2.0 * (0.5 ** int(tp_lost))))


def direct_clinical_plus_aec_auc(
    y_g: np.ndarray,
    y_s: np.ndarray,
    c_g: np.ndarray,
    c_s: np.ndarray,
    a_g: np.ndarray,
    a_s: np.ndarray,
) -> dict:
    """게이트 없이 임상점수+AEC조합점수를 그대로 로지스틱으로 합친 모델의 외부 AUC를 임상 단독/AEC 단독과 비교."""
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(np.column_stack([c_g, a_g]), y_g)
    full_s = model.decision_function(np.column_stack([c_s, a_s]))
    return {
        "clinical_auc_sdata": float(roc_auc_score(y_s, c_s)),
        "aec_combo_auc_sdata": float(roc_auc_score(y_s, a_s)),
        "clinical_plus_aec_score_auc_sdata": float(roc_auc_score(y_s, full_s)),
        "delta_auc_vs_clinical": float(roc_auc_score(y_s, full_s) - roc_auc_score(y_s, c_s)),
        "coef_clinical_z": float(model.coef_.ravel()[0]),
        "coef_aec_combo_z": float(model.coef_.ravel()[1]),
    }


def plot_candidate(eval_df: pd.DataFrame, model_id: str, width: float, lam: float, path: Path) -> None:
    """지정된 모델의 운영점별 (임상 vs 결합모델 특이도) 막대그래프와 (특이도이득 vs 민감도손실) 막대그래프를 나란히 그려 PNG로 저장."""
    ext = eval_df[
        (eval_df["model_id"].eq(model_id))
        & (eval_df["width"].eq(width))
        & (eval_df["lambda"].eq(lam))
        & (eval_df["dataset"].eq("sdata_external"))
    ].copy()
    ext = ext.set_index("operating_point").loc[ALL_OPS].reset_index()
    x = np.arange(len(ext))
    labels = ext["operating_point"].tolist()
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7))
    axes[0].bar(x - w / 2, ext["clinical_specificity"] * 100, w, color="#8DA0CB", label="Clinical")
    axes[0].bar(x + w / 2, ext["rule_specificity"] * 100, w, color="#4DAF4A", label="Clinical + regression AEC gate")
    axes[0].axhline(70, color="#333333", lw=1.1, ls="--", label="70%")
    axes[0].set_ylabel("Specificity (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.24)
    axes[0].legend(frameon=False)

    axes[1].bar(x - w / 2, ext["specificity_gain"] * 100, w, color="#4DAF4A", label="Specificity gain")
    axes[1].bar(x + w / 2, ext["sensitivity_loss"] * 100, w, color="#D95F02", label="Sensitivity loss")
    axes[1].set_ylabel("Percentage points")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend(frameon=False)
    fig.suptitle(f"Regression-combo AEC gate: {model_id}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 개별 특징 하나가 아니라, 특징 여러 개를 "정규화 로지스틱
    회귀로 결합"한 복합 점수를 게이트에 쓰면 단일 특징보다 나은가? — 랭킹 방법 2종 x 특징개수 5종
    x 정규화강도 4종 x 클래스가중치 2종 = 80개 회귀 조합 총점검):

    1. g1090/sdata를 로드하고 임상 단독 점수와 초대형 특징은행을 준비.
    2. conditional_rank(임상변수 통제한 점수검정)와 univariate_rank(단순 상관) 두 방식으로
       특징 중요도 순위를 매기고, 상위 k(16~256)개만 골라 L2 로지스틱 회귀로 결합한 80개 모델
       설정을 만든다.
    3. 각 모델을 5-fold OOF로 학습해 표준화된 결합 점수를 만들고, 게이트폭 3 x 람다 4 x 운영점 5
       조합에서 gate_metrics 성능을 모두 계산. 각 모델의 상위 20개 계수도 기록.
    4. 3개 주요 운영점 기준으로 train/외부 성능을 요약하고, train 선택점수로 전체 순위를 매긴다.
    5. "엄격한 train 선택"(특이도70%+민감도손실7.5%이하 조건, 외부 데이터 미관여)과 "외부 참고용"
       (조금 더 느슨한 조건 통과 후보 중 외부 성능 1위, 상한선 참고용) 두 후보를 뽑아 트레이드오프
       그래프를 저장.
    6. 선택된 후보들의 진양성 손실에 대한 정확 이항검정 p값을 계산하고, 스캐너 통제 하 하향조정
       유의성, 외부 부트스트랩, 게이트 없는 직접결합 AUC까지 모두 확인.
    7. 전체 요약을 JSON으로 저장하고, 선택된 후보·직접AUC·조정된 p값을 콘솔에 출력.
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

    ranks = {
        "conditional": conditional_rank(y_g, c_g, xg),
        "univariate": univariate_rank(y_g, xg),
    }
    configs = []
    model_counter = 0
    for ranker in ["conditional", "univariate"]:
        for k in [16, 32, 64, 128, 256]:
            for c in [0.03, 0.10, 0.30, 1.00]:
                for class_weight in [None, "balanced"]:
                    model_counter += 1
                    configs.append(
                        {
                            "model_id": f"ridge_{model_counter:03d}",
                            "ranker": ranker,
                            "k_features": k,
                            "C": c,
                            "class_weight": class_weight,
                            "idx": ranks[ranker][:k],
                        }
                    )

    all_rows = []
    score_store = {}
    coef_rows = []
    for cfg in configs:
        oof, ext, coef = fit_oof_external_score(xg, y_g, xs, cfg["idx"], cfg["C"], cfg["class_weight"])
        a_g, a_s = zfit(oof, ext)
        if roc_auc_score(y_g, a_g) < 0.5:
            a_g = -a_g
            a_s = -a_s
            coef = -coef
        model_meta = {k: v for k, v in cfg.items() if k != "idx"}
        score_store[cfg["model_id"]] = {"g": a_g, "s": a_s, "idx": cfg["idx"], "meta": model_meta}
        all_rows.extend(eval_gate_grid(y_g, y_s, c_g, c_s, a_g, a_s, thresholds, model_meta))
        final_coef = coef[-1]
        top_coef_order = np.argsort(np.abs(final_coef))[::-1][:20]
        for pos in top_coef_order:
            coef_rows.append(
                {
                    **model_meta,
                    "feature": names[int(cfg["idx"][pos])],
                    "coef": float(final_coef[pos]),
                    "abs_coef": float(abs(final_coef[pos])),
                }
            )

    eval_df = pd.DataFrame(all_rows)
    eval_df.to_csv(OUT_DIR / "regression_combo_gate_all_models.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT_DIR / "regression_combo_top_coefficients.csv", index=False)

    train_summary = summarize_primary(eval_df, "g1090_oof")
    ext_summary = summarize_primary(eval_df, "sdata_external")
    summary = train_summary.merge(
        ext_summary,
        on=["model_id", "ranker", "k_features", "C", "class_weight", "width", "lambda"],
        how="left",
    )
    summary["train_selection_score"] = summary.apply(selection_score, axis=1)
    summary = summary.sort_values("train_selection_score", ascending=False)
    summary.to_csv(OUT_DIR / "regression_combo_primary_summary.csv", index=False)

    # Strict selection: chosen without external labels.
    strict_candidates = summary[
        (summary["g1090_oof_primary_min_rule_specificity"] >= 0.70)
        & (summary["g1090_oof_primary_max_sens_loss"] <= 0.075)
        & (summary["g1090_oof_primary_min_balanced_gain"] > 0)
    ].copy()
    if strict_candidates.empty:
        strict = summary.iloc[0]
    else:
        strict = strict_candidates.sort_values("train_selection_score", ascending=False).iloc[0]

    # Reference: among train-acceptable models, which actually gives the best external 70%-specificity tradeoff.
    ref_pool = summary[
        (summary["g1090_oof_primary_min_rule_specificity"] >= 0.70)
        & (summary["g1090_oof_primary_max_sens_loss"] <= 0.10)
        & (summary["g1090_oof_primary_min_balanced_gain"] > 0)
    ].copy()
    if ref_pool.empty:
        ref_pool = summary.copy()
    ref_pool["external_reference_score"] = (
        2.0 * ref_pool["sdata_external_primary_min_rule_specificity"]
        + 0.8 * ref_pool["sdata_external_primary_avg_rule_specificity"]
        + 0.8 * ref_pool["sdata_external_primary_min_balanced_gain"]
        - 0.8 * ref_pool["sdata_external_primary_avg_sens_loss"]
    )
    reference = ref_pool.sort_values("external_reference_score", ascending=False).iloc[0]

    selected_rows = []
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        model_id = row["model_id"]
        width = float(row["width"])
        lam = float(row["lambda"])
        detail = eval_df[
            (eval_df["model_id"].eq(model_id))
            & (eval_df["width"].eq(width))
            & (eval_df["lambda"].eq(lam))
        ].copy()
        detail.insert(0, "selection_type", label)
        selected_rows.append(detail)
        plot_candidate(eval_df, model_id, width, lam, OUT_DIR / f"{label}_{model_id}_specificity_tradeoff.png")
    selected_eval = pd.concat(selected_rows, ignore_index=True)

    sens_p = []
    for _, row in selected_eval.iterrows():
        if row["dataset"] != "sdata_external":
            continue
        sens_p.append(
            {
                "selection_type": row["selection_type"],
                "model_id": row["model_id"],
                "operating_point": row["operating_point"],
                "tp_lost": int(row["tp_lost"]),
                "sensitivity_loss_exact_two_sided_p": sensitivity_loss_exact_p(int(row["tp_lost"])),
            }
        )
    sens_p_df = pd.DataFrame(sens_p)
    selected_eval = selected_eval.merge(sens_p_df, on=["selection_type", "model_id", "operating_point"], how="left")
    selected_eval.to_csv(OUT_DIR / "selected_regression_combo_gate_eval.csv", index=False)

    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(s["y"]))).astype(str).to_numpy()
    adj_rows = []
    boot_rows = []
    auc_rows = []
    seen = set()
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        model_id = row["model_id"]
        if model_id in seen:
            continue
        seen.add(model_id)
        a_g = score_store[model_id]["g"]
        a_s = score_store[model_id]["s"]
        width = float(row["width"])
        lam = float(row["lambda"])
        for include_clinical in [False, True]:
            adj = adjusted_deesc_p(y_s, c_s, a_s, scanner_s, thresholds, width, lam, include_clinical=include_clinical)
            adj.insert(0, "selection_type", label)
            adj.insert(1, "model_id", model_id)
            adj_rows.append(adj)
        boot = bootstrap_metrics(y_s, c_s, a_s, thresholds, width, lam, n_boot=2000)
        boot.insert(0, "selection_type", label)
        boot.insert(1, "model_id", model_id)
        boot_rows.append(boot)
        auc_rows.append({"selection_type": label, "model_id": model_id, **direct_clinical_plus_aec_auc(y_g, y_s, c_g, c_s, a_g, a_s)})

    adjusted = pd.concat(adj_rows, ignore_index=True)
    adjusted.to_csv(OUT_DIR / "selected_regression_combo_adjusted_pvalues.csv", index=False)
    boot = pd.concat(boot_rows, ignore_index=True)
    boot.to_csv(OUT_DIR / "selected_regression_combo_bootstrap.csv", index=False)
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(OUT_DIR / "selected_regression_combo_direct_auc.csv", index=False)

    report = {
        "n_features": len(names),
        "n_model_configs": len(configs),
        "strict_train_selected": strict.to_dict(),
        "external_reference": reference.to_dict(),
        "outputs": {
            "all_models": str(OUT_DIR / "regression_combo_gate_all_models.csv"),
            "primary_summary": str(OUT_DIR / "regression_combo_primary_summary.csv"),
            "selected_eval": str(OUT_DIR / "selected_regression_combo_gate_eval.csv"),
            "adjusted_pvalues": str(OUT_DIR / "selected_regression_combo_adjusted_pvalues.csv"),
            "bootstrap": str(OUT_DIR / "selected_regression_combo_bootstrap.csv"),
            "direct_auc": str(OUT_DIR / "selected_regression_combo_direct_auc.csv"),
        },
    }
    (OUT_DIR / "regression_combo_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Strict train-selected")
    print(strict.to_string())
    print("\nExternal reference")
    print(reference.to_string())
    print("\nSelected external eval")
    print(selected_eval[selected_eval["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nDirect AUC")
    print(auc_df.to_string(index=False))
    print("\nAdjusted p-values")
    print(adjusted.to_string(index=False))


if __name__ == "__main__":
    main()
