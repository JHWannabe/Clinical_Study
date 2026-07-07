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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, make_folds  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    adjusted_deesc_p,
    bootstrap_metrics,
    build_candidate_bank,
    clinical_scores,
    counts,
    load_aec128,
    standardize_train_test,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_deescalation_combo_rule"
SEED = 20260630
OPS = ["youden", "sens80", "sens85", "sens90", "sens95"]
PRIMARY_OPS = ["youden", "sens80", "sens85"]
MAX_TP_LOSS_GRID = [0.05, 0.075, 0.10]


def clinical_positive_feature_rank(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, th: float) -> np.ndarray:
    """임상양성군(clinical_z>=th) 내에서만 각 특징과 결과(y)의 상관 크기를 계산해, 특징을 중요도 내림차순으로 정렬한 인덱스를 반환."""
    mask = clinical_z >= th
    yy = y[mask].astype(float)
    xx = x[mask]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return np.arange(x.shape[1])
    yc = yy - yy.mean()
    score = np.abs(xx.T @ yc) / np.sqrt(np.sum(xx * xx, axis=0) + 1e-12)
    return np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]


def pooled_primary_feature_rank(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, thresholds: dict[str, dict]) -> np.ndarray:
    """3개 주요 운영점(youden/sens80/sens85) 각각의 clinical_positive_feature_rank 순위를 역순위 점수로 환산해 합산하고, 전체 운영점에서 고르게 상위권인 특징을 우선하는 통합 순위를 계산."""
    scores = np.zeros(x.shape[1], dtype=float)
    for op in PRIMARY_OPS:
        rank = clinical_positive_feature_rank(y, clinical_z, x, thresholds[op]["clinical_z"])
        inv = np.empty_like(rank, dtype=float)
        inv[rank] = np.arange(len(rank), dtype=float)
        scores += 1.0 / (1.0 + inv)
    return np.argsort(scores)[::-1]


def fit_cp_oof_external(
    y_g: np.ndarray,
    y_s: np.ndarray,
    c_g: np.ndarray,
    c_s: np.ndarray,
    xg: np.ndarray,
    xs: np.ndarray,
    idx: np.ndarray,
    th: float,
    c_param: float,
    class_weight: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF score for clinical-positive g1090 and external score for sdata.

    Higher score means higher low-SMI risk, so de-escalation uses low score.
    Scores for clinical-negative rows are still populated but ignored by metrics.
    """
    cp_g = c_g >= th
    cp_s = c_s >= th
    folds = make_folds(y_g.astype(int), 5)
    score_g = np.zeros(len(y_g), dtype=float)
    all_idx = np.arange(len(y_g))
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        tr_cp = tr_idx[cp_g[tr_idx]]
        if len(np.unique(y_g[tr_cp])) < 2 or len(tr_cp) < 20:
            tr_cp = tr_idx
        model = LogisticRegression(
            C=c_param,
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weight,
            random_state=SEED + fold_id,
        )
        model.fit(xg[tr_cp][:, idx], y_g[tr_cp])
        score_g[val_idx] = model.decision_function(xg[val_idx][:, idx])
    final_cp = np.flatnonzero(cp_g)
    if len(np.unique(y_g[final_cp])) < 2 or len(final_cp) < 20:
        final_cp = np.arange(len(y_g))
    final = LogisticRegression(
        C=c_param,
        solver="lbfgs",
        max_iter=5000,
        class_weight=class_weight,
        random_state=SEED + 99,
    )
    final.fit(xg[final_cp][:, idx], y_g[final_cp])
    score_s = final.decision_function(xs[:, idx])
    # Orient by clinical-positive OOF direction.
    if np.nanmean(score_g[cp_g & (y_g == 1)]) < np.nanmean(score_g[cp_g & (y_g == 0)]):
        score_g = -score_g
        score_s = -score_s
    return score_g, score_s


def choose_deesc_threshold(
    y: np.ndarray,
    clinical_z: np.ndarray,
    aec_risk: np.ndarray,
    th: float,
    max_tp_loss_rate: float,
) -> tuple[float, dict]:
    """임상양성군 내에서 aec_risk 점수의 가능한 모든 컷오프를 시도해, 놓치는 사건수가 max_tp_loss_rate 한도를 넘지 않는 범위 내에서 (특이도 이득 - 민감도 손실 + 제거된 위음성수 소량가중) 점수가 최대인 컷오프를 선택."""
    cp = clinical_z >= th
    clinical = counts(y, cp)
    n_events = int(np.sum(y == 1))
    max_tp_lost = int(np.floor(max_tp_loss_rate * n_events + 1e-12))
    best = None
    candidates = np.unique(aec_risk[cp])
    # De-escalate if aec_risk <= cutoff.
    for cutoff in candidates:
        deesc = cp & (aec_risk <= cutoff)
        tp_lost = int(np.sum(deesc & (y == 1)))
        if tp_lost > max_tp_lost:
            continue
        fp_removed = int(np.sum(deesc & (y == 0)))
        final = cp & ~deesc
        rule = counts(y, final)
        score = (
            2.0 * (rule["specificity"] - clinical["specificity"])
            - 1.0 * (clinical["sensitivity"] - rule["sensitivity"])
            + 0.002 * fp_removed
        )
        item = (score, fp_removed, -tp_lost, float(cutoff), rule, deesc)
        if best is None or item > best:
            best = item
    if best is None:
        cutoff = float(np.min(aec_risk[cp]) - 1e-9)
        deesc = cp & (aec_risk <= cutoff)
        rule = counts(y, cp & ~deesc)
        return cutoff, {"clinical": clinical, "rule": rule, "deesc": deesc}
    _, _, _, cutoff, rule, deesc = best
    return cutoff, {"clinical": clinical, "rule": rule, "deesc": deesc}


def eval_cutoff(y: np.ndarray, clinical_z: np.ndarray, aec_risk: np.ndarray, th: float, cutoff: float) -> dict:
    """주어진 임상임계값+AEC컷오프 조합으로 하향조정한 결과의 임상/최종 규칙 성능(특이도이득, 민감도손실, Fisher p, 이항검정 p 등)을 한 번에 계산."""
    cp = clinical_z >= th
    deesc = cp & (aec_risk <= cutoff)
    final = cp & ~deesc
    clinical = counts(y, cp)
    rule = counts(y, final)
    keep = final
    a = int(np.sum(keep & (y == 1)))
    b = int(np.sum(keep & (y == 0)))
    c = int(np.sum(deesc & (y == 1)))
    d = int(np.sum(deesc & (y == 0)))
    fisher_p = stats.fisher_exact([[a, b], [c, d]])[1] if (a + b) and (c + d) else np.nan
    sens_p = min(1.0, 2.0 * (0.5 ** c)) if c > 0 else 1.0
    return {
        **{f"clinical_{k}": v for k, v in clinical.items()},
        **{f"rule_{k}": v for k, v in rule.items()},
        "clinical_positive_n": int(cp.sum()),
        "clinical_positive_events": int(np.sum(cp & (y == 1))),
        "clinical_positive_prevalence": float(np.mean(y[cp])) if cp.any() else np.nan,
        "deesc_n": int(deesc.sum()),
        "deesc_events": c,
        "deesc_prevalence": c / (c + d) if c + d else np.nan,
        "fp_removed": d,
        "tp_lost": c,
        "specificity_gain": rule["specificity"] - clinical["specificity"],
        "sensitivity_loss": clinical["sensitivity"] - rule["sensitivity"],
        "balanced_gain": (rule["specificity"] - clinical["specificity"]) - (clinical["sensitivity"] - rule["sensitivity"]),
        "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
        "sensitivity_loss_exact_two_sided_p": sens_p,
    }


def summarize_primary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """지정된 데이터셋(g1090_oof 또는 sdata_external)에서 모델설정별로 3개 주요 운영점의 지표를 하나의 요약 행으로 압축."""
    key = ["model_id", "rank_scope", "k_features", "C", "class_weight", "max_tp_loss_train"]
    rows = []
    sub = df[(df["dataset"].eq(dataset)) & (df["operating_point"].isin(PRIMARY_OPS))]
    for keys, g in sub.groupby(key, dropna=False):
        rows.append(
            {
                **dict(zip(key, keys)),
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


def train_selection_score(row: pd.Series) -> float:
    """train(g1090 OOF) 요약 지표들을 가중합해 모델설정의 순위를 매기는 선택 점수를 계산."""
    return (
        2.0 * row["g1090_oof_primary_min_rule_specificity"]
        + 0.8 * row["g1090_oof_primary_avg_rule_specificity"]
        + 1.2 * row["g1090_oof_primary_min_balanced_gain"]
        - 1.0 * row["g1090_oof_primary_avg_sens_loss"]
        - 0.15 * row["g1090_oof_primary_avg_deesc_prevalence"]
    )


def plot_selected(df: pd.DataFrame, row: pd.Series, label: str) -> None:
    """지정된 모델의 운영점별 (임상 vs 결합규칙 특이도) 막대그래프와 (특이도이득 vs 민감도손실) 막대그래프를 나란히 그려 PNG로 저장."""
    ext = df[(df["dataset"].eq("sdata_external")) & (df["model_id"].eq(row["model_id"]))].copy()
    ext = ext.set_index("operating_point").loc[OPS].reset_index()
    x = np.arange(len(ext))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7))
    axes[0].bar(x - w / 2, ext["clinical_specificity"] * 100, w, color="#8DA0CB", label="Clinical")
    axes[0].bar(x + w / 2, ext["rule_specificity"] * 100, w, color="#4DAF4A", label="Clinical + de-escalation combo")
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


def bootstrap_fixed_cutoff(
    y: np.ndarray,
    clinical_z: np.ndarray,
    aec_risk: np.ndarray,
    thresholds: dict[str, dict],
    cutoffs: dict[str, float],
    n_boot: int = 2000,
) -> pd.DataFrame:
    """운영점별로 고정된 컷오프를 유지한 채 부트스트랩 재표본추출을 반복해, 특이도이득/민감도손실/균형이득/하향조정유병률의 평균과 95% 신뢰구간, 부호검정 확률을 계산."""
    rng = np.random.default_rng(SEED + 333)
    rows = []
    for op in OPS:
        th = thresholds[op]["clinical_z"]
        cutoff = cutoffs[op]
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            m = eval_cutoff(yy, clinical_z[idx], aec_risk[idx], th, cutoff)
            vals.append([m["specificity_gain"], m["sensitivity_loss"], m["balanced_gain"], m["deesc_prevalence"]])
        arr = np.asarray(vals)
        for j, metric in enumerate(["specificity_gain", "sensitivity_loss", "balanced_gain", "deesc_prevalence"]):
            x = arr[:, j]
            rows.append(
                {
                    "operating_point": op,
                    "metric": metric,
                    "mean": float(np.mean(x)),
                    "ci2.5": float(np.quantile(x, 0.025)),
                    "ci97.5": float(np.quantile(x, 0.975)),
                    "p_le_0": float(np.mean(x <= 0)),
                    "p_ge_0": float(np.mean(x >= 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 고정된 가우시안 게이트 대신, 임상양성군 안에서 로지스틱
    회귀로 학습한 "하향조정 위험 점수"와 운영점별 최적 컷오프를 결합하면, 5개 운영점(youden~sens95)
    모두에서 안정적으로 특이도를 높이면서 정해둔 최대 민감도손실 한도를 지키는 규칙을 찾을 수 있는가?):

    1. g1090(train)/sdata(external)를 로드하고 임상점수·후보 특징뱅크·표준화값을 계산.
    2. pooled_primary_feature_rank로 3개 주요 운영점에서 고르게 중요한 특징 순위를 만든다
       (rank_scope="pooled_primary" 한 가지만 사용).
    3. 상위 k개 특징(2~20개) x C값(0.10~1.00) x class_weight(None/balanced) x 허용 최대
       민감도손실(5%/7.5%/10%) 조합마다, 5개 운영점 각각에 대해 별도의 로지스틱 회귀 모델을
       임상양성군에서만 OOF로 학습하고(fit_cp_oof_external), choose_deesc_threshold로 그 운영점
       전용 하향조정 컷오프를 정한다.
    4. 모든 모델설정 x 5개 운영점 x 2개 데이터셋(g1090_oof/sdata_external)에 대해 eval_cutoff로
       성능을 계산하고, 3개 주요 운영점 기준으로 요약(summarize_primary)한 뒤 train_selection_score로
       정렬.
    5. train 조건(특이도≥70%, 민감도손실≤7.5%, 균형이득>0)을 만족하는 최상위 모델을 "strict_train
       _selected"로, 더 느슨한 조건(민감도손실≤10%)에서 external 성능이 가장 좋은 모델을
       "external_reference"로 선택하고, 각각의 트레이드오프 그래프를 저장.
    6. 두 선택 모델에 대해, 스캐너(제조사) 더미변수를 통제한 로지스틱회귀(LRT/Wald)로 하향조정
       변수의 유의성을 운영점별로 재확인하고(임상점수 포함/미포함), 부트스트랩으로 external 신뢰구간을
       재계산하며, 사용된 특징 목록도 저장. 모든 결과를 CSV/JSON으로 저장하고 콘솔에 요약 출력.
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

    rankers: dict[str, dict[str, np.ndarray]] = {"pooled_primary": {}}
    pooled_rank = pooled_primary_feature_rank(y_g, c_g, xg, thresholds)
    for op in OPS:
        rankers[op] = {op: clinical_positive_feature_rank(y_g, c_g, xg, thresholds[op]["clinical_z"])}
        rankers["pooled_primary"][op] = pooled_rank
    for op in OPS:
        rankers["pooled_primary"][op] = pooled_rank

    rows = []
    score_store = {}
    model_count = 0
    for rank_scope in ["pooled_primary"]:
        for k in [2, 3, 5, 8, 12, 20]:
            for c_param in [0.10, 0.30, 1.00]:
                for class_weight in [None, "balanced"]:
                    for max_loss in MAX_TP_LOSS_GRID:
                        model_count += 1
                        model_id = f"decombo_{model_count:04d}"
                        meta = {
                            "model_id": model_id,
                            "rank_scope": rank_scope,
                            "k_features": k,
                            "C": c_param,
                            "class_weight": class_weight,
                            "max_tp_loss_train": max_loss,
                        }
                        scores_g: dict[str, np.ndarray] = {}
                        scores_s: dict[str, np.ndarray] = {}
                        cutoffs: dict[str, float] = {}
                        used_features: dict[str, list[str]] = {}
                        valid = True
                        for op in OPS:
                            rank = rankers[rank_scope][op] if rank_scope == "pooled_primary" else rankers[rank_scope][rank_scope]
                            idx = rank[:k]
                            sg, ss = fit_cp_oof_external(
                                y_g,
                                y_s,
                                c_g,
                                c_s,
                                xg,
                                xs,
                                idx,
                                thresholds[op]["clinical_z"],
                                c_param,
                                class_weight,
                            )
                            scores_g[op] = sg
                            scores_s[op] = ss
                            used_features[op] = [names[i] for i in idx[: min(12, len(idx))]]
                            cutoff, _ = choose_deesc_threshold(y_g, c_g, sg, thresholds[op]["clinical_z"], max_loss)
                            cutoffs[op] = cutoff
                            if not np.isfinite(cutoff):
                                valid = False
                        if not valid:
                            continue
                        score_store[model_id] = {
                            "scores_g": scores_g,
                            "scores_s": scores_s,
                            "cutoffs": cutoffs,
                            "used_features": used_features,
                            "meta": meta,
                        }
                        for dataset, y, cvec, score_dict in [
                            ("g1090_oof", y_g, c_g, scores_g),
                            ("sdata_external", y_s, c_s, scores_s),
                        ]:
                            for op in OPS:
                                m = eval_cutoff(y, cvec, score_dict[op], thresholds[op]["clinical_z"], cutoffs[op])
                                rows.append({**meta, "dataset": dataset, "operating_point": op, "cutoff": cutoffs[op], **m})

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(OUT_DIR / "deescalation_combo_all_models.csv", index=False)
    train_sum = summarize_primary(eval_df, "g1090_oof")
    ext_sum = summarize_primary(eval_df, "sdata_external")
    summary = train_sum.merge(
        ext_sum,
        on=["model_id", "rank_scope", "k_features", "C", "class_weight", "max_tp_loss_train"],
        how="left",
    )
    summary["train_selection_score"] = summary.apply(train_selection_score, axis=1)
    summary = summary.sort_values("train_selection_score", ascending=False)
    summary.to_csv(OUT_DIR / "deescalation_combo_primary_summary.csv", index=False)

    strict_pool = summary[
        (summary["g1090_oof_primary_min_rule_specificity"] >= 0.70)
        & (summary["g1090_oof_primary_max_sens_loss"] <= 0.075)
        & (summary["g1090_oof_primary_min_balanced_gain"] > 0)
    ].copy()
    strict = strict_pool.sort_values("train_selection_score", ascending=False).iloc[0] if not strict_pool.empty else summary.iloc[0]
    ref_pool = summary[
        (summary["g1090_oof_primary_min_rule_specificity"] >= 0.70)
        & (summary["g1090_oof_primary_max_sens_loss"] <= 0.10)
        & (summary["g1090_oof_primary_min_balanced_gain"] > 0)
    ].copy()
    if ref_pool.empty:
        ref_pool = summary.copy()
    ref_pool["external_score"] = (
        2.0 * ref_pool["sdata_external_primary_min_rule_specificity"]
        + 0.8 * ref_pool["sdata_external_primary_avg_rule_specificity"]
        + 1.2 * ref_pool["sdata_external_primary_min_balanced_gain"]
        - 1.0 * ref_pool["sdata_external_primary_avg_sens_loss"]
        - 0.15 * ref_pool["sdata_external_primary_avg_deesc_prevalence"]
    )
    reference = ref_pool.sort_values("external_score", ascending=False).iloc[0]

    selected_rows = []
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        detail = eval_df[eval_df["model_id"].eq(row["model_id"])].copy()
        detail.insert(0, "selection_type", label)
        selected_rows.append(detail)
        plot_selected(eval_df, row, label)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected.to_csv(OUT_DIR / "selected_deescalation_combo_eval.csv", index=False)

    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(s["y"]))).astype(str).to_numpy()
    adjusted_rows = []
    boot_rows = []
    feature_rows = []
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        store = score_store[row["model_id"]]
        for op, feats in store["used_features"].items():
            for rank, feature in enumerate(feats, start=1):
                feature_rows.append({"selection_type": label, "model_id": row["model_id"], "operating_point": op, "rank": rank, "feature": feature})
        # adjusted_deesc_p expects one score and a common gate formula; not valid for
        # op-specific cutoffs. Implement equivalent adjusted tests directly.
        for op in OPS:
            th = thresholds[op]["clinical_z"]
            score_s = store["scores_s"][op]
            cutoff = store["cutoffs"][op]
            cp = c_s >= th
            deesc = cp & (score_s <= cutoff)
            yy = y_s[cp].astype(int)
            for include_clinical in [False, True]:
                base = pd.DataFrame()
                full = pd.DataFrame({"deesc": deesc[cp].astype(float)})
                if include_clinical:
                    base["clinical_z"] = c_s[cp]
                    full.insert(0, "clinical_z", c_s[cp])
                m = pd.Series(scanner_s[cp].astype(str))
                m = m.where(m.map(m.value_counts()) >= 20, "OTHER")
                dummies = pd.get_dummies(m, prefix="scanner", drop_first=True, dtype=float)
                base = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
                full = pd.concat([full.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
                adj_label = "scanner_plus_clinical" if include_clinical else "scanner_only"
                try:
                    fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=1000)
                    fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=1000)
                    lrt = 2 * (fit1.llf - fit0.llf)
                    adjusted_rows.append(
                        {
                            "selection_type": label,
                            "model_id": row["model_id"],
                            "operating_point": op,
                            "adjustment": adj_label,
                            "n_clinical_positive": int(cp.sum()),
                            "deesc_or": float(np.exp(fit1.params["deesc"])),
                            "deesc_coef": float(fit1.params["deesc"]),
                            "deesc_wald_p": float(fit1.pvalues["deesc"]),
                            "deesc_lrt_p": float(stats.chi2.sf(lrt, 1)),
                            "scanner_groups": int(m.nunique()),
                        }
                    )
                except Exception as exc:
                    adjusted_rows.append(
                        {
                            "selection_type": label,
                            "model_id": row["model_id"],
                            "operating_point": op,
                            "adjustment": adj_label,
                            "n_clinical_positive": int(cp.sum()),
                            "error": str(exc),
                        }
                    )
        # Bootstrap with op-specific scores and cutoffs.
        rng = np.random.default_rng(SEED + 444)
        for op in OPS:
            vals = []
            for _ in range(2000):
                idx = rng.integers(0, len(y_s), len(y_s))
                yy = y_s[idx]
                if len(np.unique(yy)) < 2:
                    continue
                m = eval_cutoff(
                    yy,
                    c_s[idx],
                    store["scores_s"][op][idx],
                    thresholds[op]["clinical_z"],
                    store["cutoffs"][op],
                )
                vals.append([m["specificity_gain"], m["sensitivity_loss"], m["balanced_gain"], m["deesc_prevalence"]])
            arr = np.asarray(vals)
            for j, metric in enumerate(["specificity_gain", "sensitivity_loss", "balanced_gain", "deesc_prevalence"]):
                x = arr[:, j]
                boot_rows.append(
                    {
                        "selection_type": label,
                        "model_id": row["model_id"],
                        "operating_point": op,
                        "metric": metric,
                        "mean": float(np.mean(x)),
                        "ci2.5": float(np.quantile(x, 0.025)),
                        "ci97.5": float(np.quantile(x, 0.975)),
                        "p_le_0": float(np.mean(x <= 0)),
                        "p_ge_0": float(np.mean(x >= 0)),
                    }
                )

    pd.DataFrame(adjusted_rows).to_csv(OUT_DIR / "selected_deescalation_combo_adjusted_pvalues.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "selected_deescalation_combo_bootstrap.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT_DIR / "selected_deescalation_combo_features.csv", index=False)
    (OUT_DIR / "deescalation_combo_summary.json").write_text(
        json.dumps(
            {
                "n_models": model_count,
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
    print(selected[selected["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nAdjusted")
    print(pd.DataFrame(adjusted_rows).to_string(index=False))


if __name__ == "__main__":
    main()
