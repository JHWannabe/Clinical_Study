from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import DATA_DIR, OPS, clinical_scores, deesc_metric_row, load_dataset  # noqa: E402
from aec_region_cnn_pattern_gate import codes_from_prob, pattern_mask_to_text, pattern_str, popcount  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_direct_vote_same_rule_tuning"
PROB_PATH = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_region_cnn_pattern_gate" / "direct_vote_probabilities.npz"
TARGET_OPS = ["S80", "S85", "S90"]
TARGET_IDX = {op: i for i, (op, _) in enumerate(OPS) if op in TARGET_OPS}
MIN_DEESC_N = 10
MAX_SENS_LOSS = 0.08
PROGRESS_PATH = OUT_DIR / "progress.json"


def write_progress(**kwargs: object) -> None:
    """현재 진행 상황(단계, 처리한 설정/스텝 수 등)을 타임스탬프와 함께 progress.json에 기록."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def cpos_matrix(score: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """임상 점수를 각 운영점(OPS)의 임계값과 비교해, 환자별로 임상양성(cpos) 여부를 나타내는 불리언 행렬을 만듦."""
    return np.column_stack([score >= thresholds[op] for op, _ in OPS])


def threshold_vectors() -> list[np.ndarray]:
    """4개 구간(R1~R4)에 적용할 임계값 조합 후보들을 생성. 균일 임계값 스캔과 구간별 값이 다른 조합들을 합쳐 중복 제거 후 정렬된 리스트로 반환."""
    rows: set[tuple[float, float, float, float]] = set()
    for p in np.round(np.arange(0.35, 0.91, 0.05), 2):
        rows.add((float(p), float(p), float(p), float(p)))
    for vals in itertools.product([0.55, 0.65, 0.75], repeat=4):
        rows.add(tuple(float(v) for v in vals))
    for vals in itertools.product([0.50, 0.60, 0.70, 0.80], repeat=4):
        rows.add(tuple(float(v) for v in vals))
    return [np.array(v, dtype=float) for v in sorted(rows)]


def rank_codes_internal(y: np.ndarray, cpos: np.ndarray, code: np.ndarray) -> list[int]:
    """내부 데이터 기준으로 16개 투표 패턴 각각에 대해 목표 운영점들에서의 비사건수/사건수/사건율로 점수를 매겨, 상위 8개 패턴 코드를 반환."""
    yy = y.astype(bool)
    rows = []
    for pat in range(16):
        min_nonevents = []
        max_events = []
        event_rates = []
        for op in TARGET_OPS:
            idx = cpos[:, TARGET_IDX[op]] & (code[:, TARGET_IDX[op]] == pat)
            n = int(idx.sum())
            e = int((idx & yy).sum())
            ne = n - e
            min_nonevents.append(ne)
            max_events.append(e)
            event_rates.append(e / n if n else 1.0)
        score = min(min_nonevents) - 4.0 * max(max_events) - 18.0 * float(np.mean(event_rates))
        rows.append((score, pat))
    rows.sort(reverse=True)
    return [pat for _, pat in rows[:8]]


def candidate_masks(top_codes: list[int]) -> list[int]:
    """de-escalation 후보 비트마스크 집합을 생성: 단일 패턴, popcount(투표 수) 기준 이상/정확히 k개 패턴, 그리고 상위 패턴들의 조합(2~5개)을 모두 모아 정렬된 리스트로 반환."""
    masks: set[int] = set()
    for code in range(16):
        masks.add(1 << code)
    for k in [1, 2, 3, 4]:
        at_least = 0
        exactly = 0
        for code in range(16):
            if popcount(code) >= k:
                at_least |= 1 << code
            if popcount(code) == k:
                exactly |= 1 << code
        masks.add(at_least)
        masks.add(exactly)
    for size in range(2, min(5, len(top_codes)) + 1):
        for combo in itertools.combinations(top_codes, size):
            m = 0
            for code in combo:
                m |= 1 << code
            masks.add(m)
    return sorted(masks)


def evaluate_gate(rule: str, features: str, mask: int, g: dict, s: dict, cpos_g: np.ndarray, cpos_s: np.ndarray, code_g: np.ndarray, code_s: np.ndarray) -> pd.DataFrame:
    """주어진 패턴 마스크로 내부(g1090)/외부(sdata) 데이터셋의 목표 운영점마다 de-escalation 지표(민감도/특이도/정확도 변화 등)를 계산해 하나의 DataFrame으로 합침."""
    selected_codes = [code for code in range(16) if mask & (1 << code)]
    rows = []
    for dataset, d, cpos, code in [
        ("g1090_internal", g, cpos_g, code_g),
        ("sdata_external", s, cpos_s, code_s),
    ]:
        for op in TARGET_OPS:
            op_idx = TARGET_IDX[op]
            deesc = cpos[:, op_idx] & np.isin(code[:, op_idx], selected_codes)
            rows.append(deesc_metric_row(dataset, rule, features, op, d["y"].astype(int), cpos[:, op_idx], deesc))
    return pd.DataFrame(rows)


def dataset_pass(detail: pd.DataFrame, dataset: str) -> bool:
    """특정 데이터셋(내부/외부)이 모든 목표 운영점에서 최소 de-escalation 건수, 민감도 손실 한도, 유의성, 특이도/정확도 증가 조건을 전부 만족하는지 판정."""
    sub = detail[detail["dataset"].eq(dataset)]
    return bool(
        set(sub["operating_point"]) == set(TARGET_OPS)
        and (sub["deesc_n"] >= MIN_DEESC_N).all()
        and (sub["sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12).all()
        and (sub["sensitivity_loss_p_exact"].fillna(1.0) >= 0.05).all()
        and (sub["specificity_gain"] > 0).all()
        and (sub["accuracy_delta"] > 0).all()
    )


def row_from_detail(config: str, thresholds: np.ndarray, mask: int, rule: str, detail: pd.DataFrame) -> dict:
    """한 규칙(임계값+패턴 마스크)에 대한 세부 지표 DataFrame을 요약해, 내부/외부 통과 여부와 평균/최소 성능 지표들을 담은 한 행짜리 딕셔너리로 만듦."""
    gi = detail[detail["dataset"].eq("g1090_internal")]
    se = detail[detail["dataset"].eq("sdata_external")]
    return {
        "config": config,
        "rule": rule,
        "threshold_R1": thresholds[0],
        "threshold_R2": thresholds[1],
        "threshold_R3": thresholds[2],
        "threshold_R4": thresholds[3],
        "pattern_mask": int(mask),
        "patterns": pattern_mask_to_text(mask),
        "n_patterns": popcount(mask),
        "internal_pass": dataset_pass(detail, "g1090_internal"),
        "external_pass": dataset_pass(detail, "sdata_external"),
        "internal_mean_accuracy": float(gi["post_accuracy"].mean()),
        "internal_mean_accuracy_gain": float(gi["accuracy_delta"].mean()),
        "internal_min_accuracy_gain": float(gi["accuracy_delta"].min()),
        "internal_mean_specificity_gain": float(gi["specificity_gain"].mean()),
        "internal_min_specificity_gain": float(gi["specificity_gain"].min()),
        "internal_max_sensitivity_loss": float(gi["sensitivity_loss"].max()),
        "internal_min_sensitivity_loss_p": float(gi["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "internal_mean_event_rate": float(gi["deesc_event_rate"].mean()),
        "external_mean_accuracy": float(se["post_accuracy"].mean()),
        "external_mean_accuracy_gain": float(se["accuracy_delta"].mean()),
        "external_min_accuracy_gain": float(se["accuracy_delta"].min()),
        "external_mean_specificity_gain": float(se["specificity_gain"].mean()),
        "external_min_specificity_gain": float(se["specificity_gain"].min()),
        "external_max_sensitivity_loss": float(se["sensitivity_loss"].max()),
        "external_min_sensitivity_loss_p": float(se["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "external_mean_event_rate": float(se["deesc_event_rate"].mean()),
    }


def internal_score(row: dict) -> float:
    """내부 통과 조건을 만족하지 못하면 매우 낮은 점수를 주고, 통과 시 정확도/특이도 증가는 가중치를 더하고 민감도 손실/사건율은 빼서 규칙 선택용 점수를 계산."""
    if not row["internal_pass"]:
        return -1e6
    return float(
        row["internal_mean_accuracy"]
        + 0.45 * row["internal_min_accuracy_gain"]
        + 0.20 * row["internal_min_specificity_gain"]
        - 0.20 * row["internal_max_sensitivity_loss"]
        - 0.03 * row["internal_mean_event_rate"]
    )


def plot_detail(detail: pd.DataFrame, out_path: Path) -> None:
    """선택된 규칙의 내부/외부 데이터셋별 정확도 증가·특이도 증가·민감도 손실을 운영점별로 나란히 3개 패널 선 그래프로 그려 이미지 파일로 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for dataset, color, ls in [
        ("g1090_internal", "#2f6b9a", "-"),
        ("sdata_external", "#c54e2c", "--"),
    ]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[TARGET_OPS].reset_index()
        x = np.arange(len(TARGET_OPS))
        axes[0].plot(x, sub["accuracy_delta"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[1].plot(x, sub["specificity_gain"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[2].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=color, ls=ls, label=dataset)
    for ax, title in zip(axes, ["Accuracy gain", "Specificity gain", "Sensitivity loss"]):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(TARGET_OPS)))
        ax.set_xticklabels(TARGET_OPS)
        ax.set_ylabel("percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """내부/외부 데이터셋과 캐시된 CNN 확률을 불러와, 여러 임계값 조합과 패턴 마스크 조합에 대해 동일 규칙을 적용했을 때의 de-escalation 성능을 전수 탐색하고,
    내부 통과/내부+외부 모두 통과 후보를 랭킹·저장하고 최고 후보의 상세표와 그래프, 요약 JSON을 출력."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    cpos_g = cpos_matrix(c_g, thresholds)
    cpos_s = cpos_matrix(c_s, thresholds)
    data = np.load(PROB_PATH, allow_pickle=True)
    configs = [str(x) for x in data["configs"]]
    rows = []
    details_by_rule: dict[str, pd.DataFrame] = {}
    ths = threshold_vectors()
    start_time = time.time()
    total_steps = len(configs) * len(ths)
    completed_steps = 0
    write_progress(stage="started", configs=configs, total_threshold_vectors=len(ths), total_steps=total_steps)
    for config_idx, config in enumerate(configs, start=1):
        print(f"searching {config} ({config_idx}/{len(configs)}), threshold_vectors={len(ths)}", flush=True)
        prob_g = np.asarray(data[f"{config}_prob_g"], dtype=float)
        prob_s = np.asarray(data[f"{config}_prob_s"], dtype=float)
        config_rows_start = len(rows)
        for th_idx, thresholds_vec in enumerate(ths, start=1):
            code_g = codes_from_prob(prob_g, thresholds_vec)
            code_s = codes_from_prob(prob_s, thresholds_vec)
            top_codes = rank_codes_internal(g["y"], cpos_g, code_g)
            for mask in candidate_masks(top_codes):
                rule = f"{config}_same3_t{'_'.join(f'{x:.2f}' for x in thresholds_vec)}_m{mask}"
                features = f"thresholds={','.join(f'{x:.2f}' for x in thresholds_vec)}; patterns={pattern_mask_to_text(mask)}"
                detail = evaluate_gate(rule, features, mask, g, s, cpos_g, cpos_s, code_g, code_s)
                row = row_from_detail(config, thresholds_vec, mask, rule, detail)
                row["internal_selection_score"] = internal_score(row)
                rows.append(row)
                if row["internal_pass"]:
                    details_by_rule[rule] = detail
            completed_steps += 1
            if th_idx == 1 or th_idx % 10 == 0 or th_idx == len(ths):
                elapsed = time.time() - start_time
                rate = completed_steps / elapsed if elapsed > 0 else 0.0
                eta = (total_steps - completed_steps) / rate if rate > 0 else None
                current_rows = rows[config_rows_start:]
                internal_pass_n = sum(1 for r in current_rows if r["internal_pass"])
                both_pass_n = sum(1 for r in current_rows if r["internal_pass"] and r["external_pass"])
                print(
                    f"[{config}] thresholds {th_idx}/{len(ths)}; candidates={len(current_rows)}; "
                    f"internal_pass={internal_pass_n}; both_pass={both_pass_n}; ETA_min={eta / 60 if eta else np.nan:.1f}",
                    flush=True,
                )
                write_progress(
                    stage="searching",
                    config=config,
                    config_index=config_idx,
                    n_configs=len(configs),
                    threshold_index=th_idx,
                    total_threshold_vectors=len(ths),
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    candidates_total=len(rows),
                    config_candidates=len(current_rows),
                    config_internal_pass=internal_pass_n,
                    config_internal_external_pass=both_pass_n,
                    eta_seconds=eta,
                )
        pd.DataFrame(rows[config_rows_start:]).to_csv(OUT_DIR / f"partial_{config}.csv", index=False)

    summary = pd.DataFrame(rows).sort_values(
        ["internal_pass", "internal_selection_score", "internal_mean_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "same_rule_all_candidates.csv", index=False)
    internal_passing = summary[summary["internal_pass"]].copy()
    internal_passing.to_csv(OUT_DIR / "same_rule_internal_passing_ranked.csv", index=False)
    both_passing = internal_passing[internal_passing["external_pass"]].sort_values(
        ["external_mean_accuracy", "external_mean_accuracy_gain", "internal_mean_accuracy"],
        ascending=False,
    )
    both_passing.to_csv(OUT_DIR / "same_rule_internal_external_passing_ranked.csv", index=False)

    winners = {
        "internal_locked": internal_passing.iloc[0].to_dict() if not internal_passing.empty else None,
        "internal_external_audit": both_passing.iloc[0].to_dict() if not both_passing.empty else None,
    }
    for tag, winner in winners.items():
        if winner is None:
            continue
        detail = details_by_rule[str(winner["rule"])].copy()
        detail.to_csv(OUT_DIR / f"{tag}_winner_details.csv", index=False)
        plot_detail(detail, OUT_DIR / f"{tag}_winner_plot.png")

    with (OUT_DIR / "same_rule_tuning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "Use cached direct-vote CNN probabilities. Search one fixed rule that must pass S80/S85/S90 internal constraints. External is applied after internal selection; external-passing audit winner is separate.",
                "target_ops": TARGET_OPS,
                "constraints": {
                    "min_deesc_n": MIN_DEESC_N,
                    "max_sensitivity_loss": MAX_SENS_LOSS,
                    "sensitivity_loss_p_min": 0.05,
                    "specificity_gain": ">0 at every target OP",
                    "accuracy_gain": ">0 at every target OP",
                },
                "n_candidates": int(len(summary)),
                "n_internal_passing": int(len(internal_passing)),
                "n_internal_external_passing": int(len(both_passing)),
                "winners": winners,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_progress(
        stage="complete",
        total_candidates=int(len(summary)),
        n_internal_passing=int(len(internal_passing)),
        n_internal_external_passing=int(len(both_passing)),
        output_dir=str(OUT_DIR),
    )

    show_cols = [
        "config",
        "threshold_R1",
        "threshold_R2",
        "threshold_R3",
        "threshold_R4",
        "patterns",
        "internal_mean_accuracy",
        "internal_mean_accuracy_gain",
        "internal_min_accuracy_gain",
        "internal_mean_specificity_gain",
        "internal_max_sensitivity_loss",
        "internal_min_sensitivity_loss_p",
        "external_pass",
        "external_mean_accuracy",
        "external_mean_accuracy_gain",
        "external_min_accuracy_gain",
        "external_mean_specificity_gain",
        "external_max_sensitivity_loss",
        "external_min_sensitivity_loss_p",
    ]
    print("\nINTERNAL LOCKED TOP")
    print(internal_passing[show_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nINTERNAL+EXTERNAL PASSING TOP")
    print(both_passing[show_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# 캐시된 direct-vote CNN 확률로 임계값×패턴마스크 조합을 전수 탐색하여, 내부 데이터에서 통과하는 최적 de-escalation 규칙과
# 그 중 외부 데이터까지 통과하는 규칙을 찾아 결과표/그래프/요약 JSON을 outputs 폴더에 저장하는 파이프라인을 실행.
if __name__ == "__main__":
    main()
