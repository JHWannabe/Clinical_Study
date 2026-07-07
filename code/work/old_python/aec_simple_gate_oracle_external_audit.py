from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_simple_morphology_gate_search import (  # noqa: E402
    DATA_DIR,
    OPS,
    OUT_DIR as BASE_OUT_DIR,
    QUANTILES,
    adjusted_p,
    auc_with_p,
    build_simple_bank,
    clinical_plus_score_auc,
    clinical_scores,
    eval_score,
    load_dataset,
    lowrisk_orient,
    plot_result,
    standardize_train_test,
    summarize,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_simple_gate_oracle_external_audit"
TOP_SINGLE = 80
TOP_PAIR_FEATURES = 36
TOP_FOR_ADJUSTED = 500


def both_summary(rows: list[dict]) -> dict:
    """내부(g1090)와 외부(sdata) 요약통계를 하나의 딕셔너리로 합침 — 이 오라클 감사에서는 외부 데이터도 탐색 단계에서 함께 봄."""
    gi = summarize(rows, "g1090_internal")
    se = summarize(rows, "sdata_external")
    return {**gi, **se}


def unadjusted_survives(s: dict, prefix: str, min_spec: float = 0.015, max_loss: float = 0.08, max_fisher: float = 0.10) -> bool:
    """제조사 조정 없이(1차 스크리닝용), 지정 데이터셋(prefix)에서 민감도손실 유의성/특이도이득/피셔p/강등인원 제약을 모두 만족하는지 판정."""
    return (
        s[f"{prefix}_min_p_loss"] >= 0.05
        and s[f"{prefix}_max_sens_loss"] <= max_loss
        and s[f"{prefix}_min_spec_gain"] >= min_spec
        and s[f"{prefix}_max_fisher_p"] <= max_fisher
        and s[f"{prefix}_min_deesc_n"] >= 20
    )


def adjusted_survives(s: dict, prefix: str, min_spec: float = 0.015, max_loss: float = 0.08) -> bool:
    """제조사 조정 LRT p값 제약까지 추가로 요구하는, 더 엄격한 최종 생존 판정."""
    return (
        s[f"{prefix}_min_p_loss"] >= 0.05
        and s[f"{prefix}_max_sens_loss"] <= max_loss
        and s[f"{prefix}_min_spec_gain"] >= min_spec
        and s[f"{prefix}_max_fisher_p"] < 0.05
        and s[f"{prefix}_max_adj_lrt_p"] < 0.05
        and s[f"{prefix}_min_deesc_n"] >= 20
    )


def score_summary(s: dict) -> float:
    """내부/외부 중 더 나쁜 쪽(min)을 기준으로 특이도이득/민감도손실/조정p값을 조합해 후보 순위를 매기는 점수를 계산."""
    min_spec = min(s["g1090_internal_min_spec_gain"], s["sdata_external_min_spec_gain"])
    mean_spec = min(s["g1090_internal_mean_spec_gain"], s["sdata_external_mean_spec_gain"])
    max_loss = max(s["g1090_internal_max_sens_loss"], s["sdata_external_max_sens_loss"])
    max_adj = max(
        float(np.nan_to_num(s.get("g1090_internal_max_adj_lrt_p", np.nan), nan=1.0)),
        float(np.nan_to_num(s.get("sdata_external_max_adj_lrt_p", np.nan), nan=1.0)),
    )
    return 3.0 * min_spec + mean_spec - 0.8 * max_loss - 0.03 * max_adj


def evaluate_candidate(label, subset, q, xg_low, xs_low, g, s, c_g, c_s, thresholds, do_adjusted):
    """특징 부분집합의 평균을 점수로 삼아 분위수 컷오프를 적용하고, 내부+외부 통합 요약과 원본 상세 행들을 반환."""
    sg = xg_low[:, list(subset)].mean(axis=1)
    ss = xs_low[:, list(subset)].mean(axis=1)
    rows, cutoff = eval_score(label, sg, ss, g, s, c_g, c_s, thresholds, q, do_adjusted=do_adjusted)
    summ = both_summary(rows)
    return rows, cutoff, summ, sg, ss


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: "만약 g1090뿐 아니라 sdata 외부 데이터까지 보면서 규칙을
    골랐다면" 특이도이득이 얼마나 더 좋아 보일 수 있는가? — 이것은 실제 배포용 잠금 규칙이 아니라,
    사전등록(lock) 프로토콜이 얼마나 보수적인지 가늠하기 위한 "오라클(사후참조) 감사"임을 명시):

    1. g1090/sdata를 로드하고 단순 특징뱅크를 만들어 저위험 방향으로 정렬.
    2. 모든 단일 특징 x 모든 분위수 컷오프 조합을 내부+외부 양쪽 기준으로 평가(제조사 조정 없이)해
       CSV로 저장하고, 상위 36개 특징으로 모든 쌍(pair) 조합도 마찬가지로 평가.
    3. 단일+쌍 후보 중 상위 500개에 대해서만 제조사 조정 LRT까지 포함한 최종 평가를 수행해, 내부와
       외부 모두에서 생존하는 후보를 우선으로 최고 후보 하나를 선택.
    4. 선택된 최고 후보의 운영점별 상세 결과, 특징 목록, 임상단독/AEC단독/결합 모델의 AUC 비교표를
       계산해 CSV로 저장하고 결과를 그래프로 시각화.
    5. "이것은 사전에 잠근 규칙이 아니라 외부 데이터 존재 여부를 살펴본 오라클 감사"라는 경고 문구를
       포함한 JSON 요약을 저장한 뒤 콘솔에 결과를 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    clinical_oof, clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    fg = build_simple_bank(g["norm"])
    fs = build_simple_bank(s["norm"])
    xg, xs, names = standardize_train_test(fg, fs)
    sign = lowrisk_orient(g["y"], c_g, xg)
    xg_low = xg * sign[None, :]
    xs_low = xs * sign[None, :]

    rows = []
    print(f"features={len(names)}", flush=True)
    for j, name in enumerate(names):
        if j and j % 200 == 0:
            print(f"single {j}/{len(names)}", flush=True)
        for q in QUANTILES:
            cand_rows, cutoff, summ, _, _ = evaluate_candidate(name, (j,), q, xg_low, xs_low, g, s, c_g, c_s, thresholds, False)
            ok_g = unadjusted_survives(summ, "g1090_internal", min_spec=0.01, max_fisher=0.20)
            ok_s = unadjusted_survives(summ, "sdata_external", min_spec=0.01, max_fisher=0.20)
            rows.append(
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
    single = pd.DataFrame(rows).sort_values(["unadjusted_both_survive", "score"], ascending=False)
    single.to_csv(OUT_DIR / "oracle_single_unadjusted_summary.csv", index=False)

    top_features = []
    for subset in single["subset"].tolist():
        j = int(subset)
        if j not in top_features:
            top_features.append(j)
        if len(top_features) >= TOP_PAIR_FEATURES:
            break
    pair_rows = []
    for a_i, a in enumerate(top_features):
        print(f"pairs {a_i + 1}/{len(top_features)}", flush=True)
        for b in top_features[a_i + 1 :]:
            label = f"{names[a]} + {names[b]}"
            for q in QUANTILES:
                cand_rows, cutoff, summ, _, _ = evaluate_candidate(label, (a, b), q, xg_low, xs_low, g, s, c_g, c_s, thresholds, False)
                ok_g = unadjusted_survives(summ, "g1090_internal", min_spec=0.01, max_fisher=0.20)
                ok_s = unadjusted_survives(summ, "sdata_external", min_spec=0.01, max_fisher=0.20)
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
    pair.to_csv(OUT_DIR / "oracle_pair_unadjusted_summary.csv", index=False)

    candidates = pd.concat([single.head(TOP_FOR_ADJUSTED), pair.head(TOP_FOR_ADJUSTED)], ignore_index=True)
    candidates = candidates.sort_values(["unadjusted_both_survive", "score"], ascending=False).head(TOP_FOR_ADJUSTED)
    adj_rows = []
    detail_rows = []
    for i, cand in candidates.reset_index(drop=True).iterrows():
        if i and i % 50 == 0:
            print(f"adjusted {i}/{len(candidates)}", flush=True)
        subset = tuple(int(v) for v in str(cand["subset"]).split("|"))
        cand_rows, cutoff, summ, sg, ss = evaluate_candidate(str(cand["label"]), subset, float(cand["q"]), xg_low, xs_low, g, s, c_g, c_s, thresholds, True)
        ok_g = adjusted_survives(summ, "g1090_internal", min_spec=0.01)
        ok_s = adjusted_survives(summ, "sdata_external", min_spec=0.01)
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
            for row in cand_rows:
                rr = dict(row)
                rr["candidate_rank"] = i + 1
                rr["candidate_subset"] = str(cand["subset"])
                detail_rows.append(rr)
    adj = pd.DataFrame(adj_rows).sort_values(["adjusted_both_survive", "adjusted_external_survive", "adjusted_score"], ascending=False)
    adj.to_csv(OUT_DIR / "oracle_adjusted_candidate_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "oracle_adjusted_survivor_details.csv", index=False)

    locked = adj[adj["adjusted_both_survive"]].head(1)
    if locked.empty:
        locked = adj[adj["adjusted_external_survive"]].head(1)
    if locked.empty:
        locked = adj.head(1)
    best = locked.iloc[0]
    subset = tuple(int(v) for v in str(best["subset"]).split("|"))
    best_rows, cutoff, summ, sg, ss = evaluate_candidate(str(best["label"]), subset, float(best["q"]), xg_low, xs_low, g, s, c_g, c_s, thresholds, True)
    details = pd.DataFrame(best_rows)
    details.to_csv(OUT_DIR / "oracle_best_operating_point_details.csv", index=False)
    feature_rows = pd.DataFrame({"feature_index": list(subset), "feature": [names[i] for i in subset], "lowrisk_sign": sign[list(subset)]})
    feature_rows.to_csv(OUT_DIR / "oracle_best_features.csv", index=False)

    combo_oof, combo_ext = clinical_plus_score_auc(g, s, clinical_oof, clinical_ext, -sg, -ss)
    auc_rows = []
    for model, ig, es in [
        ("clinical_only", clinical_oof, clinical_ext),
        ("oracle_simple_aec_score_only", -sg, -ss),
        ("clinical_plus_oracle_simple_aec_score", combo_oof, combo_ext),
    ]:
        ai, pi = auc_with_p(g["y"], ig)
        ae, pe = auc_with_p(s["y"], es)
        auc_rows.append({"model": model, "internal_auc": ai, "internal_auc_p": pi, "external_auc": ae, "external_auc_p": pe})
    auc_df = pd.DataFrame(auc_rows)
    auc_df["internal_delta_vs_clinical"] = auc_df["internal_auc"] - auc_df.loc[0, "internal_auc"]
    auc_df["external_delta_vs_clinical"] = auc_df["external_auc"] - auc_df.loc[0, "external_auc"]
    auc_df.to_csv(OUT_DIR / "oracle_best_auc_summary.csv", index=False)
    plot_result(details, OUT_DIR / "oracle_best_plot.png")

    (OUT_DIR / "oracle_best_summary.json").write_text(
        json.dumps(
            {
                "warning": "Oracle audit: external data was used to search for existence, not a locked publication rule.",
                "best": best.to_dict(),
                "best_adjusted_summary": summ,
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
    # 외부(sdata) 데이터까지 함께 참고해 단순 게이트 후보를 탐색하는 오라클(사후참조) 감사
    # 파이프라인을 실행한다 — 실제 배포용 잠금 규칙이 아니라 프로토콜의 보수성을 가늠하기 위한 것.
    main()
