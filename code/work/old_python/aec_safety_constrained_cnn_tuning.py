from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import (  # noqa: E402
    DATA_DIR,
    OPS,
    clinical_scores,
    deesc_metric_row,
    load_dataset,
)
from aec_region_constrained_cnn_gate import (  # noqa: E402
    CONFIGS as BASE_CONFIGS,
    DEVICE,
    REGIONS,
    TrainConfig,
    crossfit_config,
    make_channels,
    standardize_channels_train_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_safety_constrained_cnn_tuning"
TARGET_OPS = ["S80", "S85", "S90"]
TARGET_OP_INDEX = {op: i for i, (op, _) in enumerate(OPS) if op in TARGET_OPS}
MIN_DEESC_N = 10
MAX_SENS_LOSS = 0.08


CONFIGS = [
    *BASE_CONFIGS,
    TrainConfig("safety_guard_10", dropout=0.35, weight_decay=3.0e-3, lr=5.0e-4, low_weight=10.0, aux_weight=0.35, max_epochs=150, patience=18),
    TrainConfig("safety_guard_14", dropout=0.40, weight_decay=4.0e-3, lr=4.0e-4, low_weight=14.0, aux_weight=0.45, max_epochs=160, patience=20),
    TrainConfig("safety_soft_6", dropout=0.25, weight_decay=2.0e-3, lr=6.0e-4, low_weight=6.0, aux_weight=0.50, max_epochs=150, patience=18),
]


def pattern_str(code: int) -> str:
    """리전(REGIONS) 개수만큼 비트로 구성된 코드값을 각 리전의 양성(+)/음성(-) 여부를 나타내는 문자열로 변환한다."""
    return "".join("+" if code & (1 << j) else "-" for j in range(len(REGIONS)))


def pattern_mask_to_text(mask: int) -> str:
    """16가지 패턴 코드 중 mask 비트로 선택된 코드들을 pattern_str로 변환해 콤마로 이어붙인 텍스트를 만든다."""
    return ",".join(pattern_str(code) for code in range(16) if mask & (1 << code))


def popcount(x: int) -> int:
    """정수 x의 2진수 표현에서 1의 개수(비트 카운트)를 센다."""
    return int(bin(int(x)).count("1"))


def cpos_matrix(clinical_z: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """모든 운영점(OPS)에 대해 임상 점수가 해당 임계값 이상인지를 열로 쌓아 (표본 수 x 운영점 수) 형태의 임상 양성 행렬을 만든다."""
    return np.column_stack([clinical_z >= thresholds[op] for op, _ in OPS])


def codes_from_branch_prob(branch_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """각 리전(branch) 확률이 대응 임계값 이상인지 투표한 뒤, 이를 이진수로 합쳐 표본별 패턴 코드(0~15)를 계산한다."""
    votes = branch_prob >= thresholds[None, :]
    code = np.zeros(len(branch_prob), dtype=np.int16)
    for j in range(votes.shape[1]):
        code += votes[:, j].astype(np.int16) * (1 << j)
    return code


def threshold_vectors() -> list[np.ndarray]:
    """리전 4개에 대한 후보 임계값 벡터들(균일 임계값 격자 및 조합별 격자)을 생성해 중복 제거된 정렬 리스트로 반환한다."""
    rows: set[tuple[float, float, float, float]] = set()
    for p in np.round(np.arange(0.45, 0.91, 0.05), 2):
        rows.add((float(p), float(p), float(p), float(p)))
    for vals in itertools.product([0.55, 0.65, 0.75], repeat=4):
        rows.add(tuple(float(v) for v in vals))
    for vals in itertools.product([0.50, 0.60, 0.70, 0.80], repeat=4):
        rows.add(tuple(float(v) for v in vals))
    return [np.array(v, dtype=float) for v in sorted(rows)]


def fast_pattern_stats(y: np.ndarray, cpos: np.ndarray, code: np.ndarray) -> pd.DataFrame:
    """대상 운영점별, 패턴 코드(0~15)별로 임상 양성 표본 중 해당 코드에 속하는 표본 수/이벤트 수/비이벤트 수/이벤트 비율을 계산한 표를 만든다."""
    rows = []
    yy = y.astype(bool)
    for op in TARGET_OPS:
        op_idx = TARGET_OP_INDEX[op]
        cp = cpos[:, op_idx]
        for pat in range(16):
            idx = cp & (code == pat)
            rows.append(
                {
                    "operating_point": op,
                    "pattern_code": pat,
                    "n": int(idx.sum()),
                    "events": int(np.sum(idx & yy)),
                    "nonevents": int(np.sum(idx & ~yy)),
                    "event_rate": float(np.sum(idx & yy) / idx.sum()) if idx.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def ranked_single_codes(y: np.ndarray, cpos: np.ndarray, code: np.ndarray) -> list[int]:
    """각 패턴 코드를 비이벤트 최소값-이벤트 최대값-평균 이벤트율 기반 점수로 평가해, 안전한(이벤트가 적은) 순으로 상위 8개 코드를 반환한다."""
    stats = fast_pattern_stats(y, cpos, code)
    rows = []
    for pat in range(16):
        sub = stats[stats["pattern_code"].eq(pat)]
        min_nonevents = float(sub["nonevents"].min())
        max_events = float(sub["events"].max())
        mean_event_rate = float(sub["event_rate"].fillna(1.0).mean())
        score = min_nonevents - 4.0 * max_events - 20.0 * mean_event_rate
        rows.append((score, pat))
    rows.sort(reverse=True)
    return [pat for _, pat in rows[:8]]


def candidate_masks(top_codes: list[int]) -> list[int]:
    """단일 코드, '패턴 개수 k개 이상/정확히 k개' 마스크, 그리고 상위 코드(top_codes)들의 부분집합 조합 마스크를 모두 모아 후보 패턴 마스크 목록을 만든다."""
    masks: set[int] = set()
    for code in range(16):
        masks.add(1 << code)
    for k in [1, 2, 3, 4]:
        atleast = 0
        exactly = 0
        for code in range(16):
            if popcount(code) >= k:
                atleast |= 1 << code
            if popcount(code) == k:
                exactly |= 1 << code
        masks.add(atleast)
        masks.add(exactly)
    for size in range(2, min(5, len(top_codes)) + 1):
        for combo in itertools.combinations(top_codes, size):
            m = 0
            for code in combo:
                m |= 1 << code
            masks.add(m)
    return sorted(masks)


def evaluate_gate(
    rule: str,
    features: str,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
    code_g: np.ndarray,
    code_s: np.ndarray,
    mask: int,
) -> pd.DataFrame:
    """주어진 패턴 마스크로 선택된 코드에 해당하는 임상 양성 표본을 재분류(de-escalate) 대상으로 삼아, 내부/외부 데이터셋과 대상 운영점별 재분류 지표를 계산한 표를 만든다."""
    selected_codes = [code for code in range(16) if mask & (1 << code)]
    rows = []
    for dataset, d, cpos, code in [
        ("g1090_internal", g, cpos_g, code_g),
        ("sdata_external", s, cpos_s, code_s),
    ]:
        for op in TARGET_OPS:
            op_idx = TARGET_OP_INDEX[op]
            deesc = cpos[:, op_idx] & np.isin(code, selected_codes)
            rows.append(deesc_metric_row(dataset, rule, features, op, d["y"].astype(int), cpos[:, op_idx], deesc))
    return pd.DataFrame(rows)


def dataset_pass(detail: pd.DataFrame, dataset: str) -> bool:
    """지정된 데이터셋에서 모든 대상 운영점에 대해 최소 재분류 표본 수, 민감도 손실 한도 및 비유의성, 양(+)의 특이도/정확도 증가 조건을 동시에 만족하는지 확인한다."""
    sub = detail[detail["dataset"].eq(dataset)]
    if set(sub["operating_point"]) != set(TARGET_OPS):
        return False
    return bool(
        (sub["deesc_n"] >= MIN_DEESC_N).all()
        and (sub["sensitivity_loss"] <= MAX_SENS_LOSS + 1e-12).all()
        and (sub["sensitivity_loss_p_exact"].fillna(1.0) >= 0.05).all()
        and (sub["specificity_gain"] > 0).all()
        and (sub["accuracy_delta"] > 0).all()
    )


def summary_row(config: str, thresholds: np.ndarray, mask: int, detail: pd.DataFrame) -> dict:
    """CNN 설정, 임계값, 패턴 마스크와 그에 따른 내부/외부 통과 여부 및 정확도/특이도/민감도 손실 요약 통계를 하나의 딕셔너리(요약 행)로 만든다."""
    gi = detail[detail["dataset"].eq("g1090_internal")]
    se = detail[detail["dataset"].eq("sdata_external")]
    return {
        "config": config,
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
        "internal_mean_deesc_event_rate": float(gi["deesc_event_rate"].mean()),
        "external_mean_accuracy": float(se["post_accuracy"].mean()),
        "external_mean_accuracy_gain": float(se["accuracy_delta"].mean()),
        "external_min_accuracy_gain": float(se["accuracy_delta"].min()),
        "external_mean_specificity_gain": float(se["specificity_gain"].mean()),
        "external_min_specificity_gain": float(se["specificity_gain"].min()),
        "external_max_sensitivity_loss": float(se["sensitivity_loss"].max()),
        "external_min_sensitivity_loss_p": float(se["sensitivity_loss_p_exact"].fillna(1.0).min()),
        "external_mean_deesc_event_rate": float(se["deesc_event_rate"].mean()),
    }


def internal_selection_score(row: dict) -> float:
    """내부 제약 조건을 통과하지 못하면 매우 낮은 점수를 주고, 통과한 경우 평균 정확도와 최소 정확도/특이도 증가, 최대 민감도 손실, 평균 재분류 이벤트율을 가중합해 후보 선택 점수를 계산한다."""
    if not row["internal_pass"]:
        return -1e6
    return float(
        row["internal_mean_accuracy"]
        + 0.45 * row["internal_min_accuracy_gain"]
        + 0.20 * row["internal_min_specificity_gain"]
        - 0.20 * row["internal_max_sensitivity_loss"]
        - 0.03 * row["internal_mean_deesc_event_rate"]
    )


def search_config(
    config_name: str,
    branch_g: np.ndarray,
    branch_s: np.ndarray,
    g: dict,
    s: dict,
    cpos_g: np.ndarray,
    cpos_s: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """하나의 CNN 설정(config_name)에 대해 모든 임계값 벡터와 후보 패턴 마스크 조합을 순회하며 게이트를 평가하고,
    전체 요약 데이터프레임과 내부 통과 후보들의 상세 지표 표를 딕셔너리로 반환한다."""
    rows = []
    details: dict[str, pd.DataFrame] = {}
    ths = threshold_vectors()
    for idx, thresholds in enumerate(ths, start=1):
        code_g = codes_from_branch_prob(branch_g, thresholds)
        code_s = codes_from_branch_prob(branch_s, thresholds)
        top_codes = ranked_single_codes(g["y"], cpos_g, code_g)
        for mask in candidate_masks(top_codes):
            rule = f"{config_name}_t{'_'.join(f'{x:.2f}' for x in thresholds)}_m{mask}"
            features = f"thresholds={','.join(f'{x:.2f}' for x in thresholds)}; patterns={pattern_mask_to_text(mask)}"
            detail = evaluate_gate(rule, features, g, s, cpos_g, cpos_s, code_g, code_s, mask)
            row = summary_row(config_name, thresholds, mask, detail)
            row["rule"] = rule
            row["internal_selection_score"] = internal_selection_score(row)
            rows.append(row)
            if row["internal_pass"]:
                details[rule] = detail
    return pd.DataFrame(rows), details


def plot_winner(detail: pd.DataFrame, out_path: Path) -> None:
    """선택된 최종 규칙(winner)의 내부/외부 데이터셋에 대한 정확도 증가, 특이도 증가, 민감도 손실을 대상 운영점별로 3개 선그래프로 그려 저장한다."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    labels = TARGET_OPS
    for dataset, color, ls in [
        ("g1090_internal", "#2f6b9a", "-"),
        ("sdata_external", "#c54e2c", "--"),
    ]:
        sub = detail[detail["dataset"].eq(dataset)].set_index("operating_point").loc[labels].reset_index()
        x = np.arange(len(labels))
        axes[0].plot(x, sub["accuracy_delta"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[1].plot(x, sub["specificity_gain"] * 100, marker="o", color=color, ls=ls, label=dataset)
        axes[2].plot(x, sub["sensitivity_loss"] * 100, marker="o", color=color, ls=ls, label=dataset)
    for ax, title in zip(axes, ["Accuracy gain", "Specificity gain", "Sensitivity loss"]):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("percentage points")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    """여러 결과 인지형(outcome-aware) 리전 CNN 설정을 g1090 내부 데이터로 교차학습하고, 각 설정에 대해 고정 패턴 게이트를
    S80/S85/S90 안전 제약 조건 하에서 탐색, 내부 통과 최선 규칙과 내부+외부 모두 통과하는 감사용 규칙을 선정해 결과를 저장한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)
    cpos_g = cpos_matrix(c_g, thresholds)
    cpos_s = cpos_matrix(c_s, thresholds)
    xg, xs = standardize_channels_train_apply(make_channels(g["norm"]), make_channels(s["norm"]))

    all_rows = []
    all_logs = []
    selected_details: dict[str, pd.DataFrame] = {}
    for cfg in CONFIGS:
        print(f"training outcome CNN: {cfg.name}", flush=True)
        sample_weight = np.column_stack([(c_g >= thresholds[op]).astype(float) for op in TARGET_OPS]).mean(axis=1)
        sample_weight = (0.25 + sample_weight).astype(np.float32)
        _gate_g, branch_g, _gate_s, branch_s, log_df = crossfit_config(cfg, xg, g["y"].astype(int), sample_weight, xs)
        log_df["config"] = cfg.name
        all_logs.append(log_df)
        print(f"searching same-pattern gates: {cfg.name}", flush=True)
        summary, details = search_config(cfg.name, branch_g, branch_s, g, s, cpos_g, cpos_s)
        all_rows.append(summary)
        selected_details.update(details)

    summary_all = pd.concat(all_rows, ignore_index=True)
    summary_all = summary_all.sort_values(
        ["internal_pass", "internal_selection_score", "internal_mean_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_all.to_csv(OUT_DIR / "safety_cnn_all_candidate_summary.csv", index=False)
    pd.concat(all_logs, ignore_index=True).to_csv(OUT_DIR / "safety_cnn_training_log.csv", index=False)

    internal_passing = summary_all[summary_all["internal_pass"]].copy()
    internal_passing.to_csv(OUT_DIR / "safety_cnn_internal_passing_ranked.csv", index=False)
    both_passing = internal_passing[internal_passing["external_pass"]].sort_values(
        ["external_mean_accuracy", "external_mean_accuracy_gain", "internal_mean_accuracy"],
        ascending=False,
    )
    both_passing.to_csv(OUT_DIR / "safety_cnn_internal_external_passing_ranked.csv", index=False)

    winner_internal = internal_passing.iloc[0].to_dict() if not internal_passing.empty else None
    winner_both = both_passing.iloc[0].to_dict() if not both_passing.empty else None

    for tag, winner in [("internal_locked", winner_internal), ("internal_external_audit", winner_both)]:
        if winner is None:
            continue
        rule = str(winner["rule"])
        detail = selected_details[rule].copy()
        detail.to_csv(OUT_DIR / f"{tag}_winner_details.csv", index=False)
        plot_winner(detail, OUT_DIR / f"{tag}_winner_plot.png")

    with (OUT_DIR / "safety_cnn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_protocol": "Train outcome-aware region CNNs on g1090 internal OOF. Search one fixed pattern gate that must pass S80/S85/S90 internal constraints. External is applied after selection; an audit winner requiring external pass is also reported separately.",
                "target_ops": TARGET_OPS,
                "constraints": {
                    "min_deesc_n": MIN_DEESC_N,
                    "max_sensitivity_loss": MAX_SENS_LOSS,
                    "sensitivity_loss_p_min": 0.05,
                    "specificity_gain": ">0 at every target OP",
                    "accuracy_gain": ">0 at every target OP",
                },
                "regions_1_indexed_inclusive": REGIONS,
                "configs": [asdict(cfg) for cfg in CONFIGS],
                "winner_internal_locked": winner_internal,
                "winner_internal_external_audit": winner_both,
            },
            f,
            indent=2,
            ensure_ascii=False,
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
    print("\nINTERNAL-LOCKED TOP")
    print(internal_passing[show_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nINTERNAL+EXTERNAL PASSING TOP")
    if both_passing.empty:
        print("No same-rule outcome-CNN candidate passed all internal and external constraints.")
    else:
        print(both_passing[show_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nSaved to {OUT_DIR}", flush=True)


# 여러 CNN 설정을 학습해 S80/S85/S90 안전 제약을 만족하는 고정 패턴 게이트를 탐색하고, 내부 전용 및 내부+외부 통과 최종 규칙을 선정해 저장하는 파이프라인을 실행한다.
if __name__ == "__main__":
    main()
