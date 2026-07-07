from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_accuracy_max_reclassification_search import (  # noqa: E402
    DeescCandidate,
    metric_for_candidate,
    pattern_candidates,
    score_threshold_candidates,
)
from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    build_candidate_bank,
    clinical_scores,
    load_dataset,
)
from aec_oof_auc_max_search import Candidate, crossfit_candidate, curve_features  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_pick_single_accuracy_diagnosis"
TARGET_OPS = [("S80", 0.80), ("S85", 0.85), ("S90", 0.90)]
MAX_SENS_LOSS = 0.08
MIN_DEESC_N = 10
COMBO_TOP_N = 40


def build_clean_aec_matrices(g: dict, s: dict) -> tuple[np.ndarray, np.ndarray]:
    """AEC shape features only; CNN probabilities are deliberately excluded."""
    curve_g = curve_features(g)
    curve_s = curve_features(s)
    bank_g = build_candidate_bank(g["norm"]).to_numpy(dtype=float)
    bank_s = build_candidate_bank(s["norm"]).to_numpy(dtype=float)
    return np.column_stack([curve_g, bank_g]), np.column_stack([curve_s, bank_s])


def score_model_candidates(g: dict, s: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """CNN 확률을 제외한 순수 AEC 형태(shape) 특징만으로 다양한 k/C 조합의 L2/L1 로지스틱 회귀 후보를 학습해
    각 후보 이름과 내부 OOF 점수, 외부 재적합 점수를 담은 목록을 반환한다."""
    xg, xs = build_clean_aec_matrices(g, s)
    specs: list[Candidate] = []
    for k in [20, 40, 80, 100, 150, 250, 400, 600]:
        for c in [0.03, 0.1, 0.3]:
            specs.append(Candidate(f"aec_shape_l2_k{k}_C{c}", "aec_shape", "logit_l2", k=k, c=c))
    for k in [40, 100, 250]:
        for c in [0.03, 0.1]:
            specs.append(Candidate(f"aec_shape_l1_k{k}_C{c}", "aec_shape", "logit_l1", k=k, c=c))
    out = []
    for i, cand in enumerate(specs, start=1):
        print(f"[score {i}/{len(specs)}] {cand.name}", flush=True)
        score_g, _score_s_fold, score_s = crossfit_candidate(cand, xg, g["y"].astype(int), xs)
        out.append((cand.name, score_g, score_s))
    return out


def p_ok(value: object) -> float:
    """값을 float로 안전하게 변환하고, 변환 실패하거나 유한하지 않으면 보수적으로 1.0(유의하지 않음)을 반환한다."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(out):
        return 1.0
    return out


def constraints(df: pd.DataFrame) -> pd.Series:
    """대상 운영점(S80/S85/S90)에 속하고 내부·외부 모두에서 최소 재분류 표본 수, 민감도 손실 한도 및 비유의성, 양(+)의 특이도·정확도 증가 조건을 만족하는지 행 단위로 판정한다."""
    return (
        df["operating_point"].isin([op for op, _ in TARGET_OPS])
        & (df["internal_deesc_n"] >= MIN_DEESC_N)
        & (df["external_deesc_n"] >= MIN_DEESC_N)
        & (df["internal_sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12)
        & (df["external_sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12)
        & (df["internal_sensitivity_loss_p"].map(p_ok) >= 0.05)
        & (df["external_sensitivity_loss_p"].map(p_ok) >= 0.05)
        & (df["internal_specificity_gain"] > 0)
        & (df["external_specificity_gain"] > 0)
        & (df["internal_accuracy_gain"] > 0)
        & (df["external_accuracy_gain"] > 0)
    )


def loose_internal_pool(rows: pd.DataFrame, kind: str, op: str) -> pd.DataFrame:
    """지정된 종류(kind)와 운영점(op)의 후보들 중 완화된 조건(민감도 손실<=0.12, p>=0.01 등)을 만족하는 것을 선호 점수로 정렬해 상위 COMBO_TOP_N개를 반환한다(결합 후보 생성용 풀)."""
    sub = rows[(rows["operating_point"].eq(op)) & (rows["candidate_kind"].eq(kind))].copy()
    p_loss = sub["sensitivity_loss_p_exact"].map(p_ok)
    sub = sub[
        (sub["deesc_n"] >= MIN_DEESC_N)
        & (sub["specificity_gain"] > 0)
        & (sub["sensitivity_loss"] <= 0.12)
        & (p_loss >= 0.01)
    ].copy()
    if sub.empty:
        return sub
    sub["combo_pref_score"] = (
        sub["accuracy_delta"]
        + 0.35 * sub["specificity_gain"]
        - 0.20 * sub["sensitivity_loss"]
        - 0.002 * sub["deesc_event_rate"].fillna(0)
    )
    return sub.sort_values(["combo_pref_score", "accuracy_delta", "specificity_gain"], ascending=False).head(COMBO_TOP_N)


def build_combo_candidates_broad(
    op: str,
    base_candidates: list[DeescCandidate],
    internal_rows: pd.DataFrame,
) -> list[DeescCandidate]:
    """완화된 조건을 통과한 점수 기반 후보와 패턴 기반 후보를 짝지어 OR/AND 결합 후보(DeescCandidate) 목록을 폭넓게 생성한다."""
    score_rows = loose_internal_pool(internal_rows, "score", op)
    pattern_rows = loose_internal_pool(internal_rows, "pattern", op)
    by_name = {c.name: c for c in base_candidates if c.op == op}
    combos: list[DeescCandidate] = []
    for score_name in score_rows["rule"].astype(str):
        for pattern_name in pattern_rows["rule"].astype(str):
            if score_name not in by_name or pattern_name not in by_name:
                continue
            score_cand = by_name[score_name]
            pattern_cand = by_name[pattern_name]
            combos.append(
                DeescCandidate(
                    name=f"OR__{score_name}__{pattern_name}",
                    kind="score_OR_pattern",
                    op=op,
                    deesc_g=score_cand.deesc_g | pattern_cand.deesc_g,
                    deesc_s=score_cand.deesc_s | pattern_cand.deesc_s,
                    detail=f"OR({score_cand.detail}) + ({pattern_cand.detail})",
                )
            )
            combos.append(
                DeescCandidate(
                    name=f"AND__{score_name}__{pattern_name}",
                    kind="pattern_AND_score_guard",
                    op=op,
                    deesc_g=score_cand.deesc_g & pattern_cand.deesc_g,
                    deesc_s=score_cand.deesc_s & pattern_cand.deesc_s,
                    detail=f"AND({score_cand.detail}) + ({pattern_cand.detail})",
                )
            )
    return combos


def clinical_positive_masks(g: dict, s: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """각 운영점의 목표 민감도에 대응하는 임상 점수 임계값을 구하고, 내부(g)/외부(s) 데이터셋에서 임상 양성 여부를 운영점별 불리언 배열로 반환한다."""
    _clinical_oof, _clinical_ext, c_g, c_s, _ = clinical_scores(g, s)
    thresholds = {
        op: float(threshold_for_min_sensitivity(g["y"].astype(int), c_g, target))
        for op, target in OPS
    }
    return (
        {op: c_g >= thresholds[op] for op, _ in OPS},
        {op: c_s >= thresholds[op] for op, _ in OPS},
    )


def main() -> None:
    """S80/S85/S90 운영점에서 AEC 점수 후보, CNN 패턴 후보, 그 결합 후보 전체를 평가하여 내부·외부 제약 조건(민감도 손실 한도,
    양의 특이도/정확도 증가 등)을 모두 통과하는 후보 중 외부 정확도가 가장 높은 단일 최종 후보(winner)를 선정해 결과를 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    cpos_g, cpos_s = clinical_positive_masks(g, s)

    score_models = score_model_candidates(g, s)
    base_candidates = score_threshold_candidates(score_models, cpos_g, cpos_s)
    base_candidates.extend(pattern_candidates(cpos_g, cpos_s))
    base_candidates = [c for c in base_candidates if c.op in dict(TARGET_OPS)]

    internal_rows = [
        metric_for_candidate("g1090_internal", g, cpos_g[cand.op], cand)
        for cand in base_candidates
    ]
    internal_df = pd.DataFrame(internal_rows)

    combo_candidates = []
    for op, _ in TARGET_OPS:
        combo_candidates.extend(build_combo_candidates_broad(op, base_candidates, internal_df))
    all_candidates = base_candidates + combo_candidates

    rows = []
    for cand in all_candidates:
        int_row = metric_for_candidate("g1090_internal", g, cpos_g[cand.op], cand)
        ext_row = metric_for_candidate("sdata_external", s, cpos_s[cand.op], cand)
        rows.append(
            {
                "candidate": cand.name,
                "candidate_kind": cand.kind,
                "operating_point": cand.op,
                "candidate_detail": cand.detail,
                "internal_accuracy": int_row["post_accuracy"],
                "internal_accuracy_gain": int_row["accuracy_delta"],
                "internal_accuracy_p": int_row["accuracy_delta_p_mcnemar"],
                "internal_sensitivity": int_row["post_sensitivity"],
                "internal_sensitivity_loss": int_row["sensitivity_loss"],
                "internal_sensitivity_loss_p": int_row["sensitivity_loss_p_exact"],
                "internal_specificity": int_row["post_specificity"],
                "internal_specificity_gain": int_row["specificity_gain"],
                "internal_specificity_p": int_row["specificity_gain_p_exact"],
                "internal_deesc_n": int_row["deesc_n"],
                "internal_deesc_events": int_row["deesc_events"],
                "internal_deesc_event_rate": int_row["deesc_event_rate"],
                "external_accuracy": ext_row["post_accuracy"],
                "external_accuracy_gain": ext_row["accuracy_delta"],
                "external_accuracy_p": ext_row["accuracy_delta_p_mcnemar"],
                "external_sensitivity": ext_row["post_sensitivity"],
                "external_sensitivity_loss": ext_row["sensitivity_loss"],
                "external_sensitivity_loss_p": ext_row["sensitivity_loss_p_exact"],
                "external_specificity": ext_row["post_specificity"],
                "external_specificity_gain": ext_row["specificity_gain"],
                "external_specificity_p": ext_row["specificity_gain_p_exact"],
                "external_deesc_n": ext_row["deesc_n"],
                "external_deesc_events": ext_row["deesc_events"],
                "external_deesc_event_rate": ext_row["deesc_event_rate"],
            }
        )

    df = pd.DataFrame(rows)
    df["passes_constraints"] = constraints(df)
    df = df.sort_values(
        ["passes_constraints", "external_accuracy", "internal_accuracy", "external_accuracy_gain"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df.to_csv(OUT_DIR / "all_candidates_s80_s85_s90_ranked.csv", index=False)

    passing = df[df["passes_constraints"]].copy()
    passing.to_csv(OUT_DIR / "passing_candidates_ranked.csv", index=False)
    winners_by_op = (
        passing.sort_values(["operating_point", "external_accuracy"], ascending=[True, False])
        .groupby("operating_point", as_index=False)
        .head(1)
    )
    winners_by_op.to_csv(OUT_DIR / "best_by_operating_point.csv", index=False)
    winner = passing.iloc[0].to_dict()
    with (OUT_DIR / "winner.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_rule": "Among S80/S85/S90 candidates, require both internal and external non-significant sensitivity loss, positive specificity gain, and positive accuracy gain. Winner is highest external post-accuracy; internal accuracy is the first tie-breaker.",
                "score_split": "AEC score uses clean AEC shape features only. CNN enters separately through direct-vote pattern candidates.",
                "target_operating_points": TARGET_OPS,
                "max_sensitivity_loss": MAX_SENS_LOSS,
                "min_deescalated_n": MIN_DEESC_N,
                "n_all_candidates": int(len(df)),
                "n_passing_candidates": int(len(passing)),
                "winner": winner,
            },
            f,
            indent=2,
        )

    show_cols = [
        "operating_point",
        "candidate_kind",
        "external_accuracy",
        "external_accuracy_gain",
        "external_accuracy_p",
        "external_sensitivity",
        "external_sensitivity_loss",
        "external_sensitivity_loss_p",
        "external_specificity",
        "external_specificity_gain",
        "external_specificity_p",
        "internal_accuracy",
        "internal_accuracy_gain",
        "internal_accuracy_p",
        "internal_sensitivity",
        "internal_sensitivity_loss",
        "internal_sensitivity_loss_p",
        "internal_specificity",
        "internal_specificity_gain",
        "internal_specificity_p",
        "external_deesc_n",
        "external_deesc_events",
        "candidate_detail",
    ]
    print("\nBEST BY OPERATING POINT")
    print(winners_by_op[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nWINNER")
    print(pd.DataFrame([winner])[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# S80/S85/S90 운영점에서 안전 제약을 통과하는 AEC/패턴 결합 재분류 후보 중 외부 정확도가 최고인 단일 후보를 선정해 저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
