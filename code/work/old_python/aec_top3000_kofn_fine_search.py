from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import (  # noqa: E402
    build_candidate_bank,
    clinical_scores,
    load_aec128,
    risk_direction,
    standardize_train_test,
)
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search"
RANKED_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_midrange_feature_refit" / "midrange_feature_search_train_ranked.csv"
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]
TOP_N = 3000
POOL_N = 34


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측 양성/음성과 실제 결과로부터 tp/fp/fn/tn, 정확도, 민감도, 특이도, 균형정확도를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(y),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": 0.5 * (sens + spec),
    }


def exact_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    if n == 0:
        return np.nan
    return float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def paired_pvalues(y: np.ndarray, clinical_pos: np.ndarray, final_pos: np.ndarray) -> dict:
    """임상단독 판정과 k-of-m 규칙 적용 후 최종판정을 비교해, 민감도손실/특이도이득/정확도변화 각각에 대한 대응 정확검정 p값을 계산."""
    yy = y.astype(bool)
    sens_loss = int(np.sum(yy & clinical_pos & ~final_pos))
    sens_gain = int(np.sum(yy & ~clinical_pos & final_pos))
    spec_gain = int(np.sum(~yy & clinical_pos & ~final_pos))
    spec_loss = int(np.sum(~yy & ~clinical_pos & final_pos))
    cc = clinical_pos == yy
    fc = final_pos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    return {
        "sensitivity_loss_p_exact": exact_p(sens_loss, sens_gain),
        "specificity_gain_p_exact": exact_p(spec_gain, spec_loss),
        "accuracy_delta_p_mcnemar": exact_p(acc_gain, acc_loss),
    }


def fisher_p(y: np.ndarray, final_pos: np.ndarray, deesc: np.ndarray) -> float:
    """최종 유지군과 하향조정군의 2x2 사건표에 대한 Fisher 정확검정 p값을 계산."""
    a = int(np.sum(y[final_pos] == 1))
    b = int(np.sum(y[final_pos] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    if not (a + b) or not (c + d):
        return np.nan
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def metric_row(
    dataset: str,
    rule: str,
    features: str,
    op: str,
    y: np.ndarray,
    clinical_pos: np.ndarray,
    deesc: np.ndarray,
) -> dict:
    """한 데이터셋·규칙·운영점 조합에 대해 임상단독 대비 최종판정의 민감도/특이도/균형정확도 변화와 하향조정군 사건비율·Fisher p값을 계산."""
    final_pos = clinical_pos & ~deesc
    base = counts(y, clinical_pos)
    post = counts(y, final_pos)
    return {
        "dataset": dataset,
        "rule": rule,
        "features": features,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "clinical_balanced_accuracy": base["balanced_accuracy"],
        "post_balanced_accuracy": post["balanced_accuracy"],
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_p(y, final_pos, deesc),
        **paired_pvalues(y, clinical_pos, final_pos),
    }


def summarize(rows: list[dict], selection_dataset: str) -> dict:
    """지정된 데이터셋(Gangnam 또는 Pooled)에서 5개 운영점에 걸친 metric_row 결과들을 모아, 모든 운영점에서 조건(민감도손실 p≥0.05, 특이도이득>0, Fisher p<0.05)을 만족하는지와 각 지표의 최솟값/최댓값/평균을 계산."""
    sub = [r for r in rows if r["dataset"] == selection_dataset]
    p_loss = np.asarray([r["sensitivity_loss_p_exact"] for r in sub], dtype=float)
    spec_gain = np.asarray([r["specificity_gain"] for r in sub], dtype=float)
    sens_loss = np.asarray([r["sensitivity_loss"] for r in sub], dtype=float)
    delta_ba = np.asarray([r["delta_balanced_accuracy"] for r in sub], dtype=float)
    fisher = np.asarray([r["deesc_event_fisher_p"] for r in sub], dtype=float)
    return {
        "selection_dataset": selection_dataset,
        "all_p_loss_ge_0_05": bool(np.nanmin(p_loss) >= 0.05),
        "all_spec_gain_positive": bool(np.nanmin(spec_gain) > 0),
        "all_fisher_lt_0_05": bool(np.nanmax(fisher) < 0.05),
        "min_p_loss": float(np.nanmin(p_loss)),
        "max_sens_loss": float(np.nanmax(sens_loss)),
        "min_spec_gain": float(np.nanmin(spec_gain)),
        "mean_spec_gain": float(np.nanmean(spec_gain)),
        "min_delta_ba": float(np.nanmin(delta_ba)),
        "mean_delta_ba": float(np.nanmean(delta_ba)),
        "max_fisher_p": float(np.nanmax(fisher)),
    }


def eval_rule(
    rule: str,
    features_label: str,
    selected: list[int],
    k: int,
    y_by: dict[str, np.ndarray],
    cpos_by: dict[tuple[str, str], np.ndarray],
    signal_by: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict], dict, dict]:
    """선택된 특징 부분집합(selected)에 대해 k-of-m 투표 규칙으로 하향조정 여부를 정하고, Gangnam/Sinchon/Pooled 3개 데이터셋 x 5개 운영점 전체 성능을 계산한 뒤 Gangnam·Pooled 기준 요약을 함께 반환."""
    rows = []
    for dataset in ["Gangnam internal OOF", "Sinchon external"]:
        for op, _ in OPS:
            sig = signal_by[(dataset, op)][selected]
            deesc = cpos_by[(dataset, op)] & (sig.sum(axis=0) >= k)
            rows.append(metric_row(dataset, rule, features_label, op, y_by[dataset], cpos_by[(dataset, op)], deesc))
    pooled_rows = []
    for op, _ in OPS:
        y = np.concatenate([y_by["Gangnam internal OOF"], y_by["Sinchon external"]])
        cpos = np.concatenate([cpos_by[("Gangnam internal OOF", op)], cpos_by[("Sinchon external", op)]])
        sig_g = signal_by[("Gangnam internal OOF", op)][selected]
        sig_s = signal_by[("Sinchon external", op)][selected]
        deesc = np.concatenate(
            [
                cpos_by[("Gangnam internal OOF", op)] & (sig_g.sum(axis=0) >= k),
                cpos_by[("Sinchon external", op)] & (sig_s.sum(axis=0) >= k),
            ]
        )
        pooled_rows.append(metric_row("Pooled Gangnam+Sinchon", rule, features_label, op, y, cpos, deesc))
    rows.extend(pooled_rows)
    return rows, summarize(rows, "Gangnam internal OOF"), summarize(rows, "Pooled Gangnam+Sinchon")


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_midrange_feature_refit이 train에서 순위매긴 상위
    3000개 "특징+폭+람다" 후보 중에서, 개별로 안전한 후보들만 골라 작은 풀을 만들고, 그 풀
    안에서 2~5개씩 묶은 k-of-m 합의규칙을 모두 탐색하면 Gangnam과 Gangnam+Sinchon 통합
    데이터 양쪽에서 안전하게 살아남는 규칙을 찾을 수 있는가?):

    1. g1090(Gangnam)/sdata(Sinchon)를 로드하고 임상점수·후보 특징뱅크·표준화값을 계산한 뒤,
       g1090 기준 5개 운영점(S80~S90)의 임상 임계값을 구한다.
    2. aec_midrange_feature_refit이 저장한 순위표에서 상위 3000개 후보를 읽어, 각 후보의
       가우시안 게이트 하향조정 신호를 두 데이터셋 x 5개 운영점 전체에 대해 미리 계산해둔다
       (signal_by_full, 대용량 불리언 행렬).
    3. 각 개별 후보(1-of-1 규칙)의 Gangnam·Pooled 성능을 요약(individual_summaries)하고, 두
       기준 모두 특이도이득>0, Fisher p<0.10/0.05를 만족하는 후보만 골라 pool_score로 정렬해
       중복특징 없이 상위 POOL_N(34)개의 다양한 탐색 풀을 구성.
    4. 이 풀에서 크기 2~5의 모든 부분집합 x 해당하는 k값(과반수 이상, OR형 1-of-m 제외) 조합에
       대해 eval_rule로 k-of-m 규칙 성능을 계산하고(Gangnam에서 안전한 규칙만 상세행 보관),
       Gangnam과 Pooled 양쪽에서 안전조건을 만족하는지(survives_both_selection_summaries)와
       score_both로 순위를 매겨 전체/생존 후보 표를 저장.
    5. 생존한 상위 25개 고유 규칙에 대해 상세 성능표를 별도로 저장하고, 상위 12개 규칙 요약을
       콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    y_by = {"Gangnam internal OOF": g["y"].astype(int), "Sinchon external": s["y"].astype(int)}
    c_g, c_s, _ = clinical_scores(g, s)
    c_by = {"Gangnam internal OOF": c_g, "Sinchon external": c_s}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], c_g, target) for op, target in OPS}

    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], c_g, xg)
    x_by = {"Gangnam internal OOF": xg * direction[None, :], "Sinchon external": xs * direction[None, :]}
    name_to_idx = {name: i for i, name in enumerate(names)}

    ranked = pd.read_csv(RANKED_PATH).head(TOP_N).copy()
    ranked = ranked[ranked["feature"].isin(name_to_idx)].copy()
    ranked = ranked.drop_duplicates(["feature", "width", "lambda"]).reset_index(drop=True)
    ranked["candidate_id"] = np.arange(len(ranked))
    ranked["feature_short"] = ranked["feature"].str.replace("bank_norm__", "", regex=False).str.replace("midrange__", "", regex=False)
    print(f"Evaluating {len(ranked)} candidates from top {TOP_N}")

    cpos_by: dict[tuple[str, str], np.ndarray] = {}
    signal_by_full: dict[tuple[str, str], np.ndarray] = {}
    for dataset in ["Gangnam internal OOF", "Sinchon external"]:
        n = len(y_by[dataset])
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c_by[dataset] >= th
            cpos_by[(dataset, op)] = cpos
            sig = np.zeros((len(ranked), n), dtype=np.int8)
            for i, r in ranked.iterrows():
                idx = name_to_idx[str(r["feature"])]
                width = float(r["width"])
                lam = float(r["lambda"])
                boundary = np.exp(-0.5 * ((c_by[dataset] - th) / width) ** 2)
                gate = c_by[dataset] + lam * boundary * x_by[dataset][:, idx]
                sig[i] = (cpos & (gate < th)).astype(np.int8)
            signal_by_full[(dataset, op)] = sig

    # Individual summaries for screening/pruning.
    individual_summaries = []
    individual_metric_rows = []
    for i, r in ranked.iterrows():
        label = str(r["feature_short"])
        rows, gangnam_sum, pooled_sum = eval_rule(
            "1-of-1",
            label,
            [int(i)],
            1,
            y_by,
            cpos_by,
            signal_by_full,
        )
        individual_metric_rows.extend(rows)
        individual_summaries.append({**r.to_dict(), **{f"gangnam_{k}": v for k, v in gangnam_sum.items()}})
        individual_summaries[-1].update({f"pooled_{k}": v for k, v in pooled_sum.items()})
    indiv = pd.DataFrame(individual_summaries)
    indiv.to_csv(OUT_DIR / "top3000_individual_screen_summary.csv", index=False)
    pd.DataFrame(individual_metric_rows).to_csv(OUT_DIR / "top3000_individual_fine_metrics.csv", index=False)

    # Build a compact but diverse pool from the top 3000.
    indiv["pool_score"] = (
        1.5 * indiv["gangnam_min_spec_gain"]
        + indiv["gangnam_mean_spec_gain"]
        + 1.5 * indiv["pooled_min_spec_gain"]
        + indiv["pooled_mean_spec_gain"]
        + 0.5 * indiv["gangnam_min_delta_ba"]
        + 0.5 * indiv["pooled_min_delta_ba"]
        - 0.4 * indiv["gangnam_max_sens_loss"]
        - 0.4 * indiv["pooled_max_sens_loss"]
    )
    near = indiv[
        (indiv["gangnam_min_spec_gain"] > 0.0)
        & (indiv["pooled_min_spec_gain"] > 0.0)
        & (indiv["gangnam_max_fisher_p"] < 0.10)
        & (indiv["pooled_max_fisher_p"] < 0.05)
    ].copy()
    # Keep one setting per feature to avoid consensus rules made of duplicate copies.
    near = near.sort_values("pool_score", ascending=False).drop_duplicates("feature")
    seeds = pd.concat(
        [
            near.sort_values("pool_score", ascending=False).head(POOL_N),
            near.sort_values("gangnam_min_spec_gain", ascending=False).head(12),
            near.sort_values("pooled_min_spec_gain", ascending=False).head(12),
        ],
        ignore_index=True,
    ).drop_duplicates("feature")
    pool = seeds.sort_values("pool_score", ascending=False).head(POOL_N).reset_index(drop=True)
    pool.to_csv(OUT_DIR / "search_pool_from_top3000.csv", index=False)
    pool_indices = pool["candidate_id"].astype(int).tolist()

    signal_by = {(dataset, op): sig[pool_indices] for (dataset, op), sig in signal_by_full.items()}
    rule_rows = []
    summary_rows = []
    combo_count = 0
    for m in range(2, min(5, len(pool_indices)) + 1):
        k_values = list(range((m + 1) // 2, m + 1))
        # Exclude OR-like 1-of-m for m > 1.
        k_values = [k for k in k_values if k >= 2 or m == 2]
        for subset in itertools.combinations(range(len(pool_indices)), m):
            feature_label = " + ".join(pool.iloc[list(subset)]["feature_short"].astype(str).tolist())
            for k in k_values:
                rule = f"{k}-of-{m}"
                rows, gangnam_sum, pooled_sum = eval_rule(
                    rule,
                    feature_label,
                    list(subset),
                    k,
                    y_by,
                    cpos_by,
                    signal_by,
                )
                combo_count += 1
                summary = {
                    "rule": rule,
                    "features": feature_label,
                    "m": m,
                    "k": k,
                    **{f"gangnam_{kk}": vv for kk, vv in gangnam_sum.items()},
                    **{f"pooled_{kk}": vv for kk, vv in pooled_sum.items()},
                }
                summary_rows.append(summary)
                # Keep detailed rows only for candidates that at least survive Gangnam,
                # then save final top rows later for compactness.
                if (
                    gangnam_sum["all_p_loss_ge_0_05"]
                    and gangnam_sum["all_spec_gain_positive"]
                    and gangnam_sum["all_fisher_lt_0_05"]
                ):
                    for rr in rows:
                        rr["combo_rule"] = rule
                        rr["combo_features"] = feature_label
                    rule_rows.extend(rows)
    print(f"Evaluated {combo_count} n-of-m rules from pool size {len(pool)}")
    summary = pd.DataFrame(summary_rows)
    summary["survives_gangnam"] = (
        summary["gangnam_all_p_loss_ge_0_05"]
        & summary["gangnam_all_spec_gain_positive"]
        & summary["gangnam_all_fisher_lt_0_05"]
    )
    summary["survives_pooled"] = (
        summary["pooled_all_p_loss_ge_0_05"]
        & summary["pooled_all_spec_gain_positive"]
        & summary["pooled_all_fisher_lt_0_05"]
    )
    summary["survives_both_selection_summaries"] = summary["survives_gangnam"] & summary["survives_pooled"]
    summary["score_both"] = (
        2.0 * np.minimum(summary["gangnam_min_spec_gain"], summary["pooled_min_spec_gain"])
        + 0.8 * np.minimum(summary["gangnam_mean_spec_gain"], summary["pooled_mean_spec_gain"])
        + 0.5 * np.minimum(summary["gangnam_min_delta_ba"], summary["pooled_min_delta_ba"])
        - 0.2 * np.maximum(summary["gangnam_max_sens_loss"], summary["pooled_max_sens_loss"])
    )
    summary = summary.sort_values(["survives_both_selection_summaries", "score_both"], ascending=False)
    summary.to_csv(OUT_DIR / "kofn_search_summary.csv", index=False)
    survived = summary[summary["survives_both_selection_summaries"]].copy()
    survived.to_csv(OUT_DIR / "kofn_surviving_candidates.csv", index=False)

    if not survived.empty:
        top = survived.head(25)[["rule", "features"]].drop_duplicates()
        detail_rows = []
        for _, rr in top.iterrows():
            selected = summary[(summary["rule"].eq(rr["rule"])) & (summary["features"].eq(rr["features"]))].iloc[0]
            feat_list = str(selected["features"]).split(" + ")
            pool_lookup = {short: i for i, short in enumerate(pool["feature_short"].astype(str))}
            subset = [pool_lookup[f] for f in feat_list]
            k = int(str(selected["rule"]).split("-of-")[0])
            rows, _, _ = eval_rule(str(selected["rule"]), str(selected["features"]), subset, k, y_by, cpos_by, signal_by)
            detail_rows.extend(rows)
        pd.DataFrame(detail_rows).to_csv(OUT_DIR / "top_surviving_candidate_details.csv", index=False)

    print("\nTop surviving candidates")
    print(
        survived.head(12)[
            [
                "rule",
                "features",
                "gangnam_min_p_loss",
                "gangnam_max_sens_loss",
                "gangnam_min_spec_gain",
                "gangnam_mean_spec_gain",
                "pooled_min_p_loss",
                "pooled_max_sens_loss",
                "pooled_min_spec_gain",
                "pooled_mean_spec_gain",
                "score_both",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
