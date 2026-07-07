from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_simple_gate_oracle_external_audit import (  # noqa: E402
    OUT_DIR as UNUSED_ORACLE_OUT,
    TOP_FOR_ADJUSTED,
    TOP_PAIR_FEATURES,
    adjusted_survives,
    auc_with_p,
    both_summary,
    clinical_plus_score_auc,
    evaluate_candidate,
    score_summary,
    unadjusted_survives,
)
from aec_simple_morphology_gate_search import (  # noqa: E402
    DATA_DIR,
    QUANTILES,
    build_simple_bank,
    clinical_scores,
    eval_score,
    load_dataset,
    lowrisk_orient,
    plot_result,
    standardize_train_test,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_residualized_simple_gate_audit"
TOP_SINGLE = 80
TOP_PAIR_FEATURES_LOCAL = 36
TOP_FOR_ADJUSTED_LOCAL = 500


def residualize_train_apply(xg: np.ndarray, xs: np.ndarray, c_g: np.ndarray, c_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """훈련셋에서 임상점수로 각 특징을 선형회귀한 뒤 그 잔차만 남기고(임상정보 제거) 다시 z-표준화 — 순수하게 임상과 독립적인 형태 정보만 남기기 위함."""
    bg = np.column_stack([np.ones(len(c_g)), c_g])
    bs = np.column_stack([np.ones(len(c_s)), c_s])
    beta = np.linalg.pinv(bg) @ xg
    rg = xg - bg @ beta
    rs = xs - bs @ beta
    mu = rg.mean(axis=0)
    sd = rg.std(axis=0)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return (rg - mu) / sd, (rs - mu) / sd


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_simple_gate_oracle_external_audit와 같은 오라클 탐색을,
    "임상점수와 상관된 부분을 미리 제거한(잔차화된)" 특징으로 반복해도 비슷한 저위험 신호가 남는가? —
    이 역시 사전 잠금 규칙이 아니라 존재 여부를 살펴보는 오라클/잔차화 감사임을 JSON에 명시):

    1. g1090/sdata를 로드하고 단순 특징뱅크를 만든 뒤, residualize_train_apply로 각 특징에서 임상점수
       설명분을 제거하고, 저위험 방향으로 정렬.
    2. 모든 단일 특징 x 분위수 컷오프를 내부+외부 양쪽으로 평가(제조사 조정 없이)하고, 상위 특징들로
       모든 쌍 조합도 마찬가지로 평가해 각각 CSV로 저장.
    3. 단일+쌍 후보 상위 500개에 대해 제조사 조정 LRT까지 포함한 최종 평가를 수행해 내부/외부 모두
       생존하는 최고 후보를 선택.
    4. 최고 후보의 운영점별 상세 결과, 특징 목록, 임상단독/AEC단독/결합 모델 AUC 비교표를 계산해 CSV로
       저장하고 그래프로 시각화.
    5. "임상 잔차화 + 오라클(외부데이터 참조) 감사"라는 경고 문구를 포함한 JSON 요약을 저장한 뒤 콘솔에
       결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_simple_bank(g["norm"])
    fs = build_simple_bank(s["norm"])
    xg, xs, names = standardize_train_test(fg, fs)
    xg_res, xs_res = residualize_train_apply(xg, xs, c_g, c_s)
    sign = lowrisk_orient(g["y"], c_g, xg_res)
    xg_low = xg_res * sign[None, :]
    xs_low = xs_res * sign[None, :]
    print(f"features={len(names)}", flush=True)

    single_rows = []
    for j, name in enumerate(names):
        if j and j % 200 == 0:
            print(f"single {j}/{len(names)}", flush=True)
        for q in QUANTILES:
            rows, cutoff = eval_score(name, xg_low[:, j], xs_low[:, j], g, s, c_g, c_s, thresholds, q, do_adjusted=False)
            summ = both_summary(rows)
            ok_g = unadjusted_survives(summ, "g1090_internal", min_spec=0.005, max_fisher=0.20)
            ok_s = unadjusted_survives(summ, "sdata_external", min_spec=0.005, max_fisher=0.20)
            single_rows.append(
                {
                    "label": name,
                    "subset": str(j),
                    "m": 1,
                    "q": q,
                    "cutoff": cutoff,
                    "unadjusted_both_survive": ok_g and ok_s,
                    "score": score_summary(summ),
                    **summ,
                }
            )
    single = pd.DataFrame(single_rows).sort_values(["unadjusted_both_survive", "score"], ascending=False)
    single.to_csv(OUT_DIR / "resid_single_unadjusted_summary.csv", index=False)

    top_features = []
    for subset in single["subset"].tolist():
        j = int(subset)
        if j not in top_features:
            top_features.append(j)
        if len(top_features) >= TOP_PAIR_FEATURES_LOCAL:
            break
    pair_rows = []
    for a_i, a in enumerate(top_features):
        print(f"pairs {a_i + 1}/{len(top_features)}", flush=True)
        for b in top_features[a_i + 1 :]:
            label = f"{names[a]} + {names[b]}"
            sg = xg_low[:, [a, b]].mean(axis=1)
            ss = xs_low[:, [a, b]].mean(axis=1)
            for q in QUANTILES:
                rows, cutoff = eval_score(label, sg, ss, g, s, c_g, c_s, thresholds, q, do_adjusted=False)
                summ = both_summary(rows)
                ok_g = unadjusted_survives(summ, "g1090_internal", min_spec=0.005, max_fisher=0.20)
                ok_s = unadjusted_survives(summ, "sdata_external", min_spec=0.005, max_fisher=0.20)
                pair_rows.append(
                    {
                        "label": label,
                        "subset": f"{a}|{b}",
                        "m": 2,
                        "q": q,
                        "cutoff": cutoff,
                        "unadjusted_both_survive": ok_g and ok_s,
                        "score": score_summary(summ),
                        **summ,
                    }
                )
    pair = pd.DataFrame(pair_rows).sort_values(["unadjusted_both_survive", "score"], ascending=False)
    pair.to_csv(OUT_DIR / "resid_pair_unadjusted_summary.csv", index=False)

    candidates = pd.concat([single.head(TOP_FOR_ADJUSTED_LOCAL), pair.head(TOP_FOR_ADJUSTED_LOCAL)], ignore_index=True)
    candidates = candidates.sort_values(["unadjusted_both_survive", "score"], ascending=False).head(TOP_FOR_ADJUSTED_LOCAL)
    adj_rows = []
    detail_rows = []
    for i, cand in candidates.reset_index(drop=True).iterrows():
        if i and i % 50 == 0:
            print(f"adjusted {i}/{len(candidates)}", flush=True)
        subset = tuple(int(v) for v in str(cand["subset"]).split("|"))
        sg = xg_low[:, list(subset)].mean(axis=1)
        ss = xs_low[:, list(subset)].mean(axis=1)
        rows, cutoff = eval_score(str(cand["label"]), sg, ss, g, s, c_g, c_s, thresholds, float(cand["q"]), do_adjusted=True)
        summ = both_summary(rows)
        ok_g = adjusted_survives(summ, "g1090_internal", min_spec=0.005)
        ok_s = adjusted_survives(summ, "sdata_external", min_spec=0.005)
        adj_rows.append(
            {
                **cand.to_dict(),
                "adjusted_internal_survive": ok_g,
                "adjusted_external_survive": ok_s,
                "adjusted_both_survive": ok_g and ok_s,
                "adjusted_score": score_summary(summ),
                **{f"adj_{k}": v for k, v in summ.items()},
            }
        )
        if ok_g or ok_s:
            for row in rows:
                rr = dict(row)
                rr["candidate_rank"] = i + 1
                rr["candidate_subset"] = str(cand["subset"])
                detail_rows.append(rr)
    adj = pd.DataFrame(adj_rows).sort_values(["adjusted_both_survive", "adjusted_external_survive", "adjusted_score"], ascending=False)
    adj.to_csv(OUT_DIR / "resid_adjusted_candidate_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "resid_adjusted_survivor_details.csv", index=False)

    locked = adj[adj["adjusted_both_survive"]].head(1)
    if locked.empty:
        locked = adj[adj["adjusted_external_survive"]].head(1)
    if locked.empty:
        locked = adj.head(1)
    best = locked.iloc[0]
    subset = tuple(int(v) for v in str(best["subset"]).split("|"))
    sg = xg_low[:, list(subset)].mean(axis=1)
    ss = xs_low[:, list(subset)].mean(axis=1)
    rows, cutoff = eval_score(str(best["label"]), sg, ss, g, s, c_g, c_s, thresholds, float(best["q"]), do_adjusted=True)
    details = pd.DataFrame(rows)
    details.to_csv(OUT_DIR / "resid_best_operating_point_details.csv", index=False)
    feature_rows = pd.DataFrame({"feature_index": list(subset), "feature": [names[i] for i in subset], "lowrisk_sign": sign[list(subset)]})
    feature_rows.to_csv(OUT_DIR / "resid_best_features.csv", index=False)

    combo_oof, combo_ext = clinical_plus_score_auc(g, s, clinical_oof, clinical_ext, -sg, -ss)
    auc_rows = []
    for model, ig, es in [
        ("clinical_only", clinical_oof, clinical_ext),
        ("resid_simple_aec_score_only", -sg, -ss),
        ("clinical_plus_resid_simple_aec_score", combo_oof, combo_ext),
    ]:
        ai, pi = auc_with_p(g["y"], ig)
        ae, pe = auc_with_p(s["y"], es)
        auc_rows.append({"model": model, "internal_auc": ai, "internal_auc_p": pi, "external_auc": ae, "external_auc_p": pe})
    auc_df = pd.DataFrame(auc_rows)
    auc_df["internal_delta_vs_clinical"] = auc_df["internal_auc"] - auc_df.loc[0, "internal_auc"]
    auc_df["external_delta_vs_clinical"] = auc_df["external_auc"] - auc_df.loc[0, "external_auc"]
    auc_df.to_csv(OUT_DIR / "resid_best_auc_summary.csv", index=False)
    plot_result(details, OUT_DIR / "resid_best_plot.png")

    (OUT_DIR / "resid_best_summary.json").write_text(
        json.dumps(
            {
                "warning": "Oracle/residualized audit: features residualized against internal clinical score; external data used to search for existence.",
                "best": best.to_dict(),
                "features": feature_rows.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("best")
    print(best.to_string())
    print("\nfeatures")
    print(feature_rows.to_string(index=False))
    print("\nauc")
    print(auc_df.to_string(index=False))
    print("\ndetails")
    show = [
        "dataset",
        "operating_point",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "specificity_gain",
        "specificity_gain_p_exact",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
        "adj_or",
        "adj_lrt_p",
    ]
    print(details[show].to_string(index=False))
    print("out_dir", OUT_DIR)


if __name__ == "__main__":
    # 임상점수와의 상관을 제거(잔차화)한 특징들에 대해 오라클 방식으로 저위험 신호가 남아있는지
    # 탐색하는 감사 파이프라인을 실행한다 — 이 역시 실제 잠금 규칙이 아니다.
    main()
