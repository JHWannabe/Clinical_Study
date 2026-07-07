from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_deescalation_combo_rule import choose_deesc_threshold, eval_cutoff  # noqa: E402
from aec_midrange_feature_refit import build_candidate_bank, clinical_scores, load_aec128, standardize_train_test  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_pair_combo_deesc_search"
SPEC70_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_specificity70_target_search"
OPS = ["youden", "sens80", "sens85", "sens90", "sens95"]
PRIMARY_OPS = ["youden", "sens80", "sens85"]
MAX_LOSSES = [0.075, 0.10]
ALPHAS = [0.5, 1.0, 2.0]


def feature_priority(row: pd.Series) -> float:
    """최소 최종특이도, 최소균형이득, 최대민감도손실, 최대 Fisher p값을 가중합해 특징의 우선순위 점수를 계산."""
    return (
        2.0 * row["primary_min_rule_specificity"]
        + 0.8 * row["primary_min_balanced_gain"]
        - 0.8 * row["primary_max_sens_loss"]
        - 0.01 * np.log10(np.clip(row["primary_max_fisher_p"], 1e-12, 1.0))
    )


def candidate_features(names: list[str], n: int = 40) -> list[str]:
    """aec_specificity70_target_search 결과에서 특이도70%+민감도손실10%이하 조건을 만족하는 특징을 우선순위로 정렬하고, 이전에 강했던 필수 특징들을 앞에 고정한 뒤 상위 n개를 골라 특징 쌍 탐색 후보로 반환."""
    df = pd.read_csv(SPEC70_DIR / "specificity_target_train_candidates_all.csv")
    df = df[
        (df["primary_min_rule_specificity"] >= 0.70)
        & (df["primary_max_sens_loss"] <= 0.10)
        & (df["primary_min_balanced_gain"] > 0)
        & (df["primary_max_fisher_p"] <= 0.05)
    ].copy()
    df["priority"] = df.apply(feature_priority, axis=1)
    ordered = df.sort_values("priority", ascending=False).drop_duplicates("feature")["feature"].tolist()
    must_keep = [
        "bank_norm__norm_curv_010_025_max",
        "bank_norm__norm_curv_010_021_max",
        "bank_norm__norm_curv_013_016_sd",
        "bank_norm__log_slope_013_016_sd",
        "bank_norm__norm_slope_013_016_sd",
    ]
    out = []
    for f in must_keep + ordered:
        if f in names and f not in out:
            out.append(f)
        if len(out) >= n:
            break
    return out


def orient_features(y: np.ndarray, clinical_z: np.ndarray, x: np.ndarray, thresholds: dict[str, dict]) -> np.ndarray:
    """3개 주요 운영점의 임상양성군을 합친 표본에서 각 특징이 사건군과 비사건군 중 어느 쪽에서 더 높은지를 보고, 값이 클수록 위험이 커지도록 부호를 맞춘다."""
    mask = np.zeros(len(y), dtype=bool)
    for op in PRIMARY_OPS:
        mask |= clinical_z >= thresholds[op]["clinical_z"]
    dirs = np.ones(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        low = x[mask & (y == 1), j]
        non = x[mask & (y == 0), j]
        if len(low) and len(non) and np.nanmean(low) < np.nanmean(non):
            dirs[j] = -1.0
    return dirs


def metrics_for_combo(y, clinical_z, risk, thresholds, max_loss):
    """주어진 위험점수(단일 특징 또는 alpha로 결합한 특징쌍)에 대해 5개 운영점 각각에서 choose_deesc_threshold로 컷오프를 정하고 eval_cutoff로 성능을 계산."""
    rows = []
    cutoffs = {}
    for op in OPS:
        cutoff, _ = choose_deesc_threshold(y, clinical_z, risk, thresholds[op]["clinical_z"], max_loss)
        cutoffs[op] = cutoff
        m = eval_cutoff(y, clinical_z, risk, thresholds[op]["clinical_z"], cutoff)
        rows.append({"operating_point": op, "cutoff": cutoff, **m})
    return rows, cutoffs


def primary_summary(rows: list[dict]) -> dict:
    """metrics_for_combo가 만든 운영점별 행 목록 중 3개 주요 운영점만 골라 하나의 요약 딕셔너리로 압축."""
    p = [r for r in rows if r["operating_point"] in PRIMARY_OPS]
    return {
        "primary_min_rule_specificity": min(r["rule_specificity"] for r in p),
        "primary_avg_rule_specificity": float(np.mean([r["rule_specificity"] for r in p])),
        "primary_avg_spec_gain": float(np.mean([r["specificity_gain"] for r in p])),
        "primary_avg_sens_loss": float(np.mean([r["sensitivity_loss"] for r in p])),
        "primary_max_sens_loss": max(r["sensitivity_loss"] for r in p),
        "primary_min_balanced_gain": min(r["balanced_gain"] for r in p),
        "primary_avg_deesc_prevalence": float(np.mean([r["deesc_prevalence"] for r in p])),
        "primary_max_fisher_p": max(r["fisher_p"] for r in p),
    }


def score_train(row: pd.Series) -> float:
    """train 요약 지표들을 가중합해 후보(단일특징 또는 특징쌍) 설정의 순위를 매기는 선택 점수를 계산."""
    return (
        2.0 * row["primary_min_rule_specificity"]
        + 0.8 * row["primary_avg_rule_specificity"]
        + 1.2 * row["primary_min_balanced_gain"]
        - 1.0 * row["primary_avg_sens_loss"]
        - 0.15 * row["primary_avg_deesc_prevalence"]
    )


def adjusted_tests(y, clinical_z, risk, thresholds, cutoffs, scanner) -> pd.DataFrame:
    """운영점별로 스캐너(제조사) 더미변수를 통제한 로지스틱회귀(임상점수 포함/미포함)를 적합시켜, 하향조정 변수의 오즈비·Wald p값·LRT p값을 계산."""
    rows = []
    for op in OPS:
        th = thresholds[op]["clinical_z"]
        cp = clinical_z >= th
        deesc = cp & (risk <= cutoffs[op])
        yy = y[cp].astype(int)
        for include_clinical in [False, True]:
            base = pd.DataFrame()
            full = pd.DataFrame({"deesc": deesc[cp].astype(float)})
            if include_clinical:
                base["clinical_z"] = clinical_z[cp]
                full.insert(0, "clinical_z", clinical_z[cp])
            m = pd.Series(scanner[cp].astype(str))
            m = m.where(m.map(m.value_counts()) >= 20, "OTHER")
            dummies = pd.get_dummies(m, prefix="scanner", drop_first=True, dtype=float)
            base = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            full = pd.concat([full.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            try:
                fit0 = sm.Logit(yy, sm.add_constant(base, has_constant="add")).fit(disp=False, maxiter=1000)
                fit1 = sm.Logit(yy, sm.add_constant(full, has_constant="add")).fit(disp=False, maxiter=1000)
                lrt = 2 * (fit1.llf - fit0.llf)
                rows.append(
                    {
                        "operating_point": op,
                        "adjustment": "scanner_plus_clinical" if include_clinical else "scanner_only",
                        "n_clinical_positive": int(cp.sum()),
                        "deesc_or": float(np.exp(fit1.params["deesc"])),
                        "deesc_wald_p": float(fit1.pvalues["deesc"]),
                        "deesc_lrt_p": float(stats.chi2.sf(lrt, 1)),
                    }
                )
            except Exception as exc:
                rows.append({"operating_point": op, "adjustment": "scanner_plus_clinical" if include_clinical else "scanner_only", "error": str(exc)})
    return pd.DataFrame(rows)


def plot_selected(df: pd.DataFrame, path: Path, title: str) -> None:
    """지정된 모델의 운영점별 (임상 vs 결합쌍 특이도) 막대그래프와 (특이도이득 vs 민감도손실) 막대그래프를 나란히 그려 PNG로 저장."""
    ext = df[df["dataset"].eq("sdata_external")].copy()
    ext = ext.set_index("operating_point").loc[OPS].reset_index()
    x = np.arange(len(ext))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.7))
    axes[0].bar(x - w / 2, ext["clinical_specificity"] * 100, w, color="#8DA0CB", label="Clinical")
    axes[0].bar(x + w / 2, ext["rule_specificity"] * 100, w, color="#4DAF4A", label="Clinical + pair combo")
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
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 단일 특징을 쓰는 대신, 두 특징을 alpha 비율로 선형결합한
    "특징쌍" 위험점수를 쓰면 aec_specificity70_target_search가 찾은 단일특징 규칙보다 더 안정적으로
    특이도70%를 달성하면서 민감도손실을 낮게 유지할 수 있는가?):

    1. g1090(train)/sdata(external)를 로드하고 임상점수·후보 특징뱅크·표준화값을 계산한 뒤,
       orient_features로 모든 특징의 부호를 "값이 클수록 위험 증가"로 통일한다.
    2. candidate_features로 특이도70%+민감도손실10%이하 조건을 만족하는 상위 20개 특징을 뽑고,
       이 특징들의 모든 쌍(190개) x alpha(0.5/1.0/2.0) 조합 + 단일특징 대조군을 후보로 구성.
    3. 각 후보 x 허용 최대 민감도손실(7.5%/10%) 조합마다 metrics_for_combo로 train(g1090 OOF) 성능을
       계산하고 primary_summary로 요약한 뒤 score_train으로 정렬.
    4. train 조건(특이도≥70%, 민감도손실≤7.5%, 균형이득>0)을 만족하는 최상위 후보를 "strict_train
       _selected"로 선택.
    5. train 상위 150개 후보만 external(sdata)에서도 평가하고, 그중 external 성능이 가장 좋은
       후보를 "external_reference"로 선택한 뒤, 두 선택 후보의 트레이드오프 그래프를 저장.
    6. 두 선택 후보에 대해 스캐너(제조사) 더미변수를 통제한 로지스틱회귀로 하향조정 변수의 유의성을
       운영점별로 재확인(adjusted_tests)하고, 모든 결과를 CSV/JSON으로 저장, 콘솔에 요약 출력.
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
    dirs = orient_features(y_g, c_g, xg, thresholds)
    xg = xg * dirs[None, :]
    xs = xs * dirs[None, :]
    name_to_idx = {n: i for i, n in enumerate(names)}
    feats = candidate_features(names, n=20)
    pd.DataFrame({"rank": range(1, len(feats) + 1), "feature": feats}).to_csv(OUT_DIR / "pair_combo_candidate_features.csv", index=False)

    candidates = []
    # Singles as controls.
    for f in feats:
        candidates.append(("single", f, "", 0.0, name_to_idx[f], None))
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1 :]:
            for alpha in ALPHAS:
                candidates.append(("pair", f1, f2, alpha, name_to_idx[f1], name_to_idx[f2]))

    train_rows = []
    eval_cache = {}
    for cid, (kind, f1, f2, alpha, i, j) in enumerate(candidates, start=1):
        risk_g = xg[:, i] if j is None else xg[:, i] + alpha * xg[:, j]
        risk_s = xs[:, i] if j is None else xs[:, i] + alpha * xs[:, j]
        for max_loss in MAX_LOSSES:
            rows_g, cutoffs = metrics_for_combo(y_g, c_g, risk_g, thresholds, max_loss)
            summ = primary_summary(rows_g)
            model_id = f"paircombo_{cid:05d}_loss{max_loss:g}"
            row = {
                "model_id": model_id,
                "kind": kind,
                "feature1": f1,
                "feature2": f2,
                "alpha": alpha,
                "max_tp_loss_train": max_loss,
                **summ,
            }
            row["train_score"] = score_train(pd.Series(row))
            train_rows.append(row)
            eval_cache[model_id] = (risk_g, risk_s, cutoffs)
    train_df = pd.DataFrame(train_rows).sort_values("train_score", ascending=False)
    train_df.to_csv(OUT_DIR / "pair_combo_train_summary.csv", index=False)

    strict_pool = train_df[
        (train_df["primary_min_rule_specificity"] >= 0.70)
        & (train_df["primary_max_sens_loss"] <= 0.075)
        & (train_df["primary_min_balanced_gain"] > 0)
    ].copy()
    strict = strict_pool.sort_values("train_score", ascending=False).iloc[0] if not strict_pool.empty else train_df.iloc[0]

    # Evaluate top train candidates externally, then select an exploratory reference.
    top_train = train_df.head(150).copy()
    eval_rows = []
    for _, row in top_train.iterrows():
        risk_g, risk_s, cutoffs = eval_cache[row["model_id"]]
        for dataset, y, c, risk in [("g1090_oof", y_g, c_g, risk_g), ("sdata_external", y_s, c_s, risk_s)]:
            for op in OPS:
                m = eval_cutoff(y, c, risk, thresholds[op]["clinical_z"], cutoffs[op])
                eval_rows.append({**row.to_dict(), "dataset": dataset, "operating_point": op, "cutoff": cutoffs[op], **m})
    eval_top = pd.DataFrame(eval_rows)
    eval_top.to_csv(OUT_DIR / "pair_combo_top300_external_eval.csv", index=False)

    ext_summ_rows = []
    for model_id, sub in eval_top[(eval_top["dataset"].eq("sdata_external")) & (eval_top["operating_point"].isin(PRIMARY_OPS))].groupby("model_id"):
        p = sub
        ext_summ_rows.append(
            {
                "model_id": model_id,
                "ext_min_rule_specificity": float(p["rule_specificity"].min()),
                "ext_avg_rule_specificity": float(p["rule_specificity"].mean()),
                "ext_avg_spec_gain": float(p["specificity_gain"].mean()),
                "ext_avg_sens_loss": float(p["sensitivity_loss"].mean()),
                "ext_max_sens_loss": float(p["sensitivity_loss"].max()),
                "ext_min_balanced_gain": float(p["balanced_gain"].min()),
                "ext_avg_deesc_prevalence": float(p["deesc_prevalence"].mean()),
                "ext_max_fisher_p": float(p["fisher_p"].max()),
            }
        )
    ext_summ = pd.DataFrame(ext_summ_rows)
    merged = top_train.merge(ext_summ, on="model_id", how="left")
    merged["external_score"] = (
        2.0 * merged["ext_min_rule_specificity"]
        + 0.8 * merged["ext_avg_rule_specificity"]
        + 1.2 * merged["ext_min_balanced_gain"]
        - 1.0 * merged["ext_avg_sens_loss"]
        - 0.15 * merged["ext_avg_deesc_prevalence"]
    )
    merged.to_csv(OUT_DIR / "pair_combo_train_top300_external_summary.csv", index=False)
    reference = merged.sort_values("external_score", ascending=False).iloc[0]

    selected_rows = []
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        model_id = row["model_id"]
        if model_id in set(eval_top["model_id"]):
            detail = eval_top[eval_top["model_id"].eq(model_id)].copy()
        else:
            risk_g, risk_s, cutoffs = eval_cache[model_id]
            rows = []
            for dataset, y, c, risk in [("g1090_oof", y_g, c_g, risk_g), ("sdata_external", y_s, c_s, risk_s)]:
                for op in OPS:
                    m = eval_cutoff(y, c, risk, thresholds[op]["clinical_z"], cutoffs[op])
                    rows.append({**row.to_dict(), "dataset": dataset, "operating_point": op, "cutoff": cutoffs[op], **m})
            detail = pd.DataFrame(rows)
        detail.insert(0, "selection_type", label)
        selected_rows.append(detail)
        plot_selected(detail, OUT_DIR / f"{label}_{model_id}_tradeoff.png", f"{label}: {model_id}")
    selected = pd.concat(selected_rows, ignore_index=True)
    selected.to_csv(OUT_DIR / "selected_pair_combo_eval.csv", index=False)

    adj_rows = []
    scanner_s = s["meta"].get("Manufacturer", pd.Series(["UNKNOWN"] * len(y_s))).astype(str).to_numpy()
    for label, row in [("strict_train_selected", strict), ("external_reference", reference)]:
        risk_g, risk_s, cutoffs = eval_cache[row["model_id"]]
        adj = adjusted_tests(y_s, c_s, risk_s, thresholds, cutoffs, scanner_s)
        adj.insert(0, "selection_type", label)
        adj.insert(1, "model_id", row["model_id"])
        adj_rows.append(adj)
    pd.concat(adj_rows, ignore_index=True).to_csv(OUT_DIR / "selected_pair_combo_adjusted_pvalues.csv", index=False)
    (OUT_DIR / "pair_combo_summary.json").write_text(
        json.dumps({"strict_train_selected": strict.to_dict(), "external_reference": reference.to_dict(), "n_candidates": len(candidates)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Strict")
    print(strict.to_string())
    print("\nReference")
    print(reference.to_string())
    print("\nSelected external")
    print(selected[selected["dataset"].eq("sdata_external")].to_string(index=False))
    print("\nAdjusted")
    print(pd.concat(adj_rows, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
