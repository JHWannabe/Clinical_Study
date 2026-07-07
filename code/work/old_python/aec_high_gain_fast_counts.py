from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR  # noqa: E402
from aec_midrange_feature_refit import build_candidate_bank, clinical_scores, load_aec128, risk_direction, standardize_train_test  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_high_gain_both_cohort_search"
BOTH_SUMMARY_PATH = (
    Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search" / "top3000_individual_both_cohort_survival_summary.csv"
)
SCREEN_PATH = Path(__file__).resolve().parent / "outputs" / "0630" / "aec_top3000_kofn_fine_search" / "top3000_individual_screen_summary.csv"
OPS = [("S80", 0.80), ("S82.5", 0.825), ("S85", 0.85), ("S87.5", 0.875), ("S90", 0.90)]


def p_loss_from_lost_events(c: int) -> float:
    """놓친 사건수(c)만으로 정확 이항검정(부호검정) 양측 p값을 근사 계산(2^(1-c), 대용량 조합 탐색을 빠르게 하기 위한 간이 공식)."""
    if c <= 0:
        return 1.0
    return min(1.0, 2.0 ** (1 - c))


def exact_pair_p(a: int, b: int) -> float:
    """두 대응표본 이산화 카운트(a,b)에 대한 정확 이항검정(부호검정) 양측 p값을 계산."""
    n = a + b
    return np.nan if n == 0 else float(stats.binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue)


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    """예측 양성/음성과 실제 결과로부터 민감도, 특이도, 균형정확도, 정확도를 계산."""
    yy = y.astype(bool)
    pp = pred.astype(bool)
    tp = int(np.sum(yy & pp))
    fp = int(np.sum(~yy & pp))
    fn = int(np.sum(yy & ~pp))
    tn = int(np.sum(~yy & ~pp))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": 0.5 * (sens + spec), "accuracy": (tp + tn) / len(y)}


def detail_row(dataset: str, rule: str, features: str, op: str, y: np.ndarray, cpos: np.ndarray, deesc: np.ndarray) -> dict:
    """한 데이터셋·규칙·운영점 조합에 대해 임상단독 대비 최종판정의 성능 변화, 각종 근사/정확 p값, 하향조정군 사건비율·Fisher p값을 계산."""
    fpos = cpos & ~deesc
    base = counts(y, cpos)
    post = counts(y, fpos)
    yy = y.astype(bool)
    lost = int(np.sum(yy & deesc))
    removed = int(np.sum(~yy & deesc))
    cc = cpos == yy
    fc = fpos == yy
    acc_gain = int(np.sum(~cc & fc))
    acc_loss = int(np.sum(cc & ~fc))
    a = int(np.sum(y[fpos] == 1))
    b = int(np.sum(y[fpos] == 0))
    c = int(np.sum(y[deesc] == 1))
    d = int(np.sum(y[deesc] == 0))
    fisher_p = float(stats.fisher_exact([[a, b], [c, d]])[1]) if (a + b) and (c + d) else np.nan
    return {
        "dataset": dataset,
        "rule": rule,
        "features": features,
        "operating_point": op,
        "clinical_sensitivity": base["sensitivity"],
        "post_sensitivity": post["sensitivity"],
        "sensitivity_loss": base["sensitivity"] - post["sensitivity"],
        "sensitivity_loss_p_exact": p_loss_from_lost_events(lost),
        "clinical_specificity": base["specificity"],
        "post_specificity": post["specificity"],
        "specificity_gain": post["specificity"] - base["specificity"],
        "specificity_gain_p_exact": exact_pair_p(removed, 0),
        "delta_balanced_accuracy": post["balanced_accuracy"] - base["balanced_accuracy"],
        "clinical_accuracy": base["accuracy"],
        "post_accuracy": post["accuracy"],
        "delta_accuracy": post["accuracy"] - base["accuracy"],
        "accuracy_delta_p_mcnemar": exact_pair_p(acc_gain, acc_loss),
        "deesc_n": int(deesc.sum()),
        "deesc_events": int(y[deesc].sum()),
        "deesc_event_rate": float(y[deesc].mean()) if deesc.any() else np.nan,
        "deesc_event_fisher_p": fisher_p,
    }


def build_pool(n: int = 44) -> pd.DataFrame:
    """aec_top3000_kofn_fine_search가 저장한 "두 코호트 생존 요약"과 "개별 특징 스크리닝" 표를 합쳐, 두 코호트 모두 특이도이득이 확실하고 민감도손실이 작은 특징들만 골라 high_gain_score로 정렬한 뒤 상위 n개 탐색 풀을 구성(aec_high_gain_both_cohort_search보다 더 넓고 빠른 스크리닝용)."""
    both = pd.read_csv(BOTH_SUMMARY_PATH)
    screen = pd.read_csv(SCREEN_PATH)
    screen = screen.sort_values("train_selection_score", ascending=False).drop_duplicates("feature_short")
    cfg = screen[["feature_short", "feature", "width", "lambda"]]
    both = both.merge(cfg, left_on="feature", right_on="feature_short", how="inner", suffixes=("_metric", ""))
    both["min_both_spec"] = both[["g_min_spec_gain", "s_min_spec_gain"]].min(axis=1)
    both["mean_both_spec"] = both[["g_mean_spec_gain", "s_mean_spec_gain"]].min(axis=1)
    both["max_both_loss"] = both[["g_max_sens_loss", "s_max_sens_loss"]].max(axis=1)
    both["high_gain_score"] = 2.5 * both["min_both_spec"] + both["mean_both_spec"] - 0.20 * both["max_both_loss"]
    base = both[
        (both["g_min_spec_gain"] > 0.015)
        & (both["s_min_spec_gain"] > 0.015)
        & (both["g_max_fisher"] < 0.10)
        & (both["s_max_fisher"] < 0.10)
        & (both["max_both_loss"] <= 0.09)
    ].copy()
    pool = pd.concat(
        [
            base.sort_values("high_gain_score", ascending=False).head(n),
            base.sort_values("g_min_spec_gain", ascending=False).head(16),
            base.sort_values("s_min_spec_gain", ascending=False).head(16),
        ],
        ignore_index=True,
    ).drop_duplicates("feature_short")
    return pool.sort_values("high_gain_score", ascending=False).head(n).reset_index(drop=True)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_high_gain_both_cohort_search와 비슷한 목적이지만,
    더 큰 풀(44개)과 5개 조합까지 포함한 훨씬 많은 조합수를 감당하기 위해, 1차 스크리닝은
    빠른 근사 p값 공식으로 걸러내고 살아남은 후보만 정확 Fisher 검정으로 재확인하는 2단계
    방식으로 속도를 확보하려는 목적):

    1. build_pool(44)로 두 코호트 모두에서 특이도이득이 크고 안전한 44개 특징 탐색 풀을 구성.
    2. g1090(Gangnam)/sdata(Sinchon)를 로드해 임상점수·특징뱅크를 계산하고, 두 데이터셋 x 5개
       운영점(S80~S90) 전체에 대해 풀 특징들의 하향조정 신호를 미리 계산해둔다.
    3. 크기 1~4는 전체 44개에서, 크기 5는 상위 24개로 제한해 모든 조합 x k값을 나열하고, 각
       조합에 대해 근사 p값 공식(p_loss_from_lost_events)으로 빠르게 "1차 스크리닝 생존" 여부와
       score를 계산.
    4. 1차 생존자 중 score 상위 250개만 detail_row로 정확 Fisher p값을 다시 계산해, 두 코호트
       모두 Fisher p<0.05인 "정확 생존자"만 최종 확정.
    5. 전체 조합 요약, 정확 생존자, 정확 생존자의 상세 성능표를 CSV로 저장하고, 풀 크기·조합수·
       1차/정확 생존자 수와 상위 20개 정확 생존자를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool(44)
    pool.to_csv(OUT_DIR / "fast_count_pool44.csv", index=False)
    g = load_aec128(DATA_DIR / "g1090.xlsx")
    s = load_aec128(DATA_DIR / "sdata.xlsx")
    y_by = {"Gangnam": g["y"].astype(int), "Sinchon": s["y"].astype(int)}
    cg, cs, _ = clinical_scores(g, s)
    c_by = {"Gangnam": cg, "Sinchon": cs}
    thresholds = {op: threshold_for_min_sensitivity(g["y"], cg, target) for op, target in OPS}
    fg = build_candidate_bank(g)
    fs = build_candidate_bank(s)
    xg, xs, names = standardize_train_test(fg, fs)
    direction = risk_direction(g["y"], cg, xg)
    x_by = {"Gangnam": xg * direction[None, :], "Sinchon": xs * direction[None, :]}
    name_to_idx = {name: i for i, name in enumerate(names)}

    sig = {}
    context = {}
    for ds in ["Gangnam", "Sinchon"]:
        y = y_by[ds]
        y_bool = y.astype(bool)
        n_pos = int(y_bool.sum())
        n_neg = int((~y_bool).sum())
        for op, _ in OPS:
            th = thresholds[op]
            cpos = c_by[ds] >= th
            mat = np.zeros((len(pool), len(y)), dtype=np.int8)
            for i, r in pool.iterrows():
                idx = name_to_idx[str(r["feature"])]
                boundary = np.exp(-0.5 * ((c_by[ds] - th) / float(r["width"])) ** 2)
                gate = c_by[ds] + float(r["lambda"]) * boundary * x_by[ds][:, idx]
                mat[i] = (cpos & (gate < th)).astype(np.int8)
            sig[(ds, op)] = mat
            context[(ds, op)] = {"y": y, "y_bool": y_bool, "n_pos": n_pos, "n_neg": n_neg, "cpos": cpos}

    summaries = []
    combo_iter = []
    # Broad m<=4 over all 44; m=5 only over top 24 to keep runtime bounded.
    for m in [1, 2, 3, 4]:
        idxs = range(len(pool))
        k_values = [1] if m == 1 else [k for k in range((m + 1) // 2, m + 1) if k >= 2]
        for subset in itertools.combinations(idxs, m):
            for k in k_values:
                combo_iter.append((subset, k))
    for subset in itertools.combinations(range(min(24, len(pool))), 5):
        for k in [3, 4, 5]:
            combo_iter.append((subset, k))

    for subset, k in combo_iter:
        m = len(subset)
        rule = f"{k}-of-{m}"
        features = " + ".join(pool.iloc[list(subset)]["feature_short"].astype(str))
        rec = {"rule": rule, "features": features, "m": m, "k": k}
        survives = True
        score_parts = []
        for ds in ["Gangnam", "Sinchon"]:
            min_p_loss = 1.0
            max_sens_loss = 0.0
            min_spec_gain = 1.0
            mean_spec_gain_vals = []
            min_delta_ba = 1.0
            mean_delta_vals = []
            max_event_rate = 0.0
            for op, _ in OPS:
                ctx = context[(ds, op)]
                votes = sig[(ds, op)][list(subset)].sum(axis=0)
                deesc = ctx["cpos"] & (votes >= k)
                lost = int(np.sum(ctx["y_bool"] & deesc))
                removed = int(np.sum((~ctx["y_bool"]) & deesc))
                sens_loss = lost / ctx["n_pos"]
                spec_gain = removed / ctx["n_neg"]
                p_loss = p_loss_from_lost_events(lost)
                delta_ba = 0.5 * (spec_gain - sens_loss)
                event_rate = lost / int(deesc.sum()) if int(deesc.sum()) else np.nan
                min_p_loss = min(min_p_loss, p_loss)
                max_sens_loss = max(max_sens_loss, sens_loss)
                min_spec_gain = min(min_spec_gain, spec_gain)
                mean_spec_gain_vals.append(spec_gain)
                min_delta_ba = min(min_delta_ba, delta_ba)
                mean_delta_vals.append(delta_ba)
                if pd.notna(event_rate):
                    max_event_rate = max(max_event_rate, event_rate)
            rec[f"{ds}_min_p_loss"] = min_p_loss
            rec[f"{ds}_max_sens_loss"] = max_sens_loss
            rec[f"{ds}_min_spec_gain"] = min_spec_gain
            rec[f"{ds}_mean_spec_gain"] = float(np.mean(mean_spec_gain_vals))
            rec[f"{ds}_min_delta_ba"] = min_delta_ba
            rec[f"{ds}_mean_delta_ba"] = float(np.mean(mean_delta_vals))
            rec[f"{ds}_max_event_rate"] = max_event_rate
            if not (min_p_loss >= 0.05 and min_spec_gain > 0):
                survives = False
            score_parts.append((min_spec_gain, float(np.mean(mean_spec_gain_vals)), min_delta_ba, max_sens_loss))
        rec["survives_screen"] = survives
        rec["score"] = (
            2.4 * min(score_parts[0][0], score_parts[1][0])
            + min(score_parts[0][1], score_parts[1][1])
            + 0.6 * min(score_parts[0][2], score_parts[1][2])
            - 0.2 * max(score_parts[0][3], score_parts[1][3])
        )
        summaries.append(rec)

    summary = pd.DataFrame(summaries).sort_values(["survives_screen", "score"], ascending=False)
    # Compute exact Fisher/details for top screened survivors.
    details = []
    exact_survives = []
    for _, rec in summary[summary["survives_screen"]].head(250).iterrows():
        subset_names = str(rec["features"]).split(" + ")
        name_to_pool = {str(v): i for i, v in enumerate(pool["feature_short"])}
        subset = [name_to_pool[x] for x in subset_names]
        k = int(str(rec["rule"]).split("-of-")[0])
        rows = []
        for ds in ["Gangnam", "Sinchon"]:
            for op, _ in OPS:
                ctx = context[(ds, op)]
                deesc = ctx["cpos"] & (sig[(ds, op)][subset].sum(axis=0) >= k)
                rows.append(detail_row(ds, str(rec["rule"]), str(rec["features"]), op, ctx["y"], ctx["cpos"], deesc))
        g_max_f = max(r["deesc_event_fisher_p"] for r in rows if r["dataset"] == "Gangnam")
        s_max_f = max(r["deesc_event_fisher_p"] for r in rows if r["dataset"] == "Sinchon")
        if g_max_f < 0.05 and s_max_f < 0.05:
            exact_survives.append(rec.to_dict() | {"Gangnam_max_fisher": g_max_f, "Sinchon_max_fisher": s_max_f})
            details.extend(rows)
    exact = pd.DataFrame(exact_survives)
    if not exact.empty:
        exact = exact.sort_values("score", ascending=False)
    summary.to_csv(OUT_DIR / "fast_count_high_gain_summary.csv", index=False)
    exact.to_csv(OUT_DIR / "fast_count_high_gain_exact_survivors.csv", index=False)
    pd.DataFrame(details).to_csv(OUT_DIR / "fast_count_high_gain_exact_survivor_details.csv", index=False)
    print(f"pool={len(pool)} combos={len(summary)} screened_survivors={int(summary['survives_screen'].sum())} exact_survivors={len(exact)}")
    if not exact.empty:
        print(
            exact.head(20)[
                [
                    "rule",
                    "features",
                    "Gangnam_min_p_loss",
                    "Gangnam_max_sens_loss",
                    "Gangnam_min_spec_gain",
                    "Gangnam_mean_spec_gain",
                    "Sinchon_min_p_loss",
                    "Sinchon_max_sens_loss",
                    "Sinchon_min_spec_gain",
                    "Sinchon_mean_spec_gain",
                    "Gangnam_max_fisher",
                    "Sinchon_max_fisher",
                    "score",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
