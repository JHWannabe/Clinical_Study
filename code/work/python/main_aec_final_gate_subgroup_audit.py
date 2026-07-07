from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_lock_smoothed_deesc_gate import clinical_scores, deesc_metric_row, load_dataset  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent.parent / "data_cache"
MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "aec_new_region_cnn_surrogate_mimic_gate"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "aec_final_gate_subgroup_audit"
OPS = ["S80", "S85", "S90"]


def vendor_group(value: object) -> str:
    """CT 장비 제조사/모델명 문자열을 Siemens/Philips/GE/Canon-Toshiba/Other 다섯 개 제조사 그룹으로 분류."""
    s = str(value).upper()
    if any(token in s for token in ["SOMATOM", "SENSATION", "SIEMENS"]):
        return "Siemens"
    if any(token in s for token in ["INGENUITY", "ICT", "PHILIPS"]):
        return "Philips"
    if any(token in s for token in ["REVOLUTION", "LIGHTSPEED", "DISCOVERY", "OPTIMA", "GE"]):
        return "GE"
    if any(token in s for token in ["AQUILION", "CANON", "TOSHIBA"]):
        return "Canon/Toshiba"
    return "Other"


def codes_from_prob(prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """(환자 x 운영점 x 지역) 확률을 지역별 임계값과 비교해 0/1 투표로 만들고, 4개 투표를 하나의 4비트 패턴 코드(0~15)로 합침."""
    votes = prob >= thresholds[None, None, :]
    code = np.zeros(prob.shape[:2], dtype=np.int16)
    for j in range(prob.shape[-1]):
        code += votes[:, :, j].astype(np.int16) * (1 << j)
    return code


def selected_mask(mask: int, code: np.ndarray) -> np.ndarray:
    """4비트 코드 배열에서, 주어진 패턴 비트마스크에 포함된 패턴에 해당하는 환자만 True로 표시한 불리언 배열을 만듦."""
    selected = np.zeros_like(code, dtype=bool)
    for pat in range(16):
        if mask & (1 << pat):
            selected |= code == pat
    return selected


def subgroup_index(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """메타데이터로부터 전체/성별(남/녀)/제조사별 부분군 이름을 키로 하는 불리언 인덱스 딕셔너리를 만듦 (환자가 없는 제조사 그룹은 제외)."""
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    vendor = meta["Manufacturer"].map(vendor_group).astype(str).to_numpy()
    out = {
        "All": np.ones(len(meta), dtype=bool),
        "Male": sex == "M",
        "Female": sex == "F",
    }
    for name in ["Siemens", "Philips", "GE", "Canon/Toshiba", "Other"]:
        idx = vendor == name
        if idx.sum() > 0:
            out[f"Vendor:{name}"] = idx
    return out


def summarize_detail(detail: pd.DataFrame) -> dict[str, float]:
    """한 부분군의 운영점별 상세 지표 DataFrame을 평균/최소값 중심으로 요약해, 정확도/특이도/민감도손실/de-escalation 건수 등을 담은 딕셔너리로 만듦."""
    return {
        "n_rows": int(len(detail)),
        "mean_accuracy_gain": float(detail["accuracy_delta"].mean()),
        "min_accuracy_gain": float(detail["accuracy_delta"].min()),
        "mean_specificity_gain": float(detail["specificity_gain"].mean()),
        "min_specificity_gain": float(detail["specificity_gain"].min()),
        "max_sensitivity_loss": float(detail["sensitivity_loss"].max()),
        "min_sensitivity_loss_p": float(detail["sensitivity_loss_p_exact"].min()),
        "mean_deesc_n": float(detail["deesc_n"].mean()),
        "min_deesc_n": int(detail["deesc_n"].min()),
        "mean_deesc_event_rate": float(detail["deesc_event_rate"].mean()),
    }


def main() -> None:
    """내부+외부 모두 통과한 최종 de-escalation 게이트(surrogate_mimic_summary.json의 winner)를 불러와, 성별/제조사별 부분군마다
    (표본이 20명 이상이고 라벨이 두 클래스 모두 있는 경우) 운영점별 성능 지표를 재계산해, 부분군 상세표와 요약표를 저장하는 감사(audit) 스크립트."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g = load_dataset(DATA_DIR / "g1090.xlsx")
    s = load_dataset(DATA_DIR / "sdata.xlsx")
    _clinical_oof, _clinical_ext, c_g, c_s, thresholds = clinical_scores(g, s)

    with (MODEL_DIR / "surrogate_mimic_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    winner = summary["winners"]["internal_external_audit"]
    config = str(winner["config"])
    threshold_vec = np.array(
        [winner["threshold_R1"], winner["threshold_R2"], winner["threshold_R3"], winner["threshold_R4"]],
        dtype=float,
    )
    pattern_mask = int(winner["pattern_mask"])

    prob = np.load(MODEL_DIR / f"{config}_probabilities.npz", allow_pickle=True)
    code_g = codes_from_prob(np.asarray(prob["prob_g"], dtype=float), threshold_vec)
    code_s = codes_from_prob(np.asarray(prob["prob_s"], dtype=float), threshold_vec)
    pick_g = selected_mask(pattern_mask, code_g)
    pick_s = selected_mask(pattern_mask, code_s)

    subgroup_rows = []
    detail_rows = []

    for dataset, d, clinical_z, code_pick in [
        ("g1090_internal", g, c_g, pick_g),
        ("sdata_external", s, c_s, pick_s),
    ]:
        groups = subgroup_index(d["meta"])
        vendor = d["meta"]["Manufacturer"].map(vendor_group).astype(str).to_numpy()
        sex = d["meta"]["PatientSex"].astype(str).str.upper().to_numpy()
        for group_name, idx in groups.items():
            n = int(idx.sum())
            if n < 20:
                continue
            y = d["y"][idx].astype(int)
            if np.unique(y).size < 2:
                continue
            cscore = clinical_z[idx]
            pick = code_pick[idx]
            for op_idx, op in enumerate(OPS):
                cpos = cscore >= float(thresholds[op])
                deesc = cpos & pick[:, op_idx]
                row = deesc_metric_row(
                    dataset=dataset,
                    rule=winner["rule"],
                    features=f"{config}; subgroup={group_name}",
                    op=op,
                    y=y,
                    cpos=cpos,
                    deesc=deesc,
                )
                row["group"] = group_name
                row["group_n"] = n
                row["group_low_smi_n"] = int(y.sum())
                row["group_low_smi_rate"] = float(y.mean())
                row["male_n"] = int(np.sum(sex[idx] == "M"))
                row["female_n"] = int(np.sum(sex[idx] == "F"))
                row["vendor_unique_n"] = int(pd.Series(vendor[idx]).nunique())
                detail_rows.append(row)

        detail_df = pd.DataFrame([r for r in detail_rows if r["dataset"] == dataset])
        for group_name in detail_df["group"].unique():
            sub = detail_df[detail_df["group"].eq(group_name)].copy()
            subgroup_rows.append(
                {
                    "dataset": dataset,
                    "group": group_name,
                    "group_n": int(sub["group_n"].iloc[0]),
                    "group_low_smi_n": int(sub["group_low_smi_n"].iloc[0]),
                    "group_low_smi_rate": float(sub["group_low_smi_rate"].iloc[0]),
                    **summarize_detail(sub),
                }
            )

    detail_all = pd.DataFrame(detail_rows)
    summary_all = pd.DataFrame(subgroup_rows).sort_values(["dataset", "group"])
    detail_all.to_csv(OUT_DIR / "final_gate_subgroup_details.csv", index=False)
    summary_all.to_csv(OUT_DIR / "final_gate_subgroup_summary.csv", index=False)

    print("FINAL GATE")
    print(config, winner["rule"])
    print("thresholds", threshold_vec.tolist())
    print("patterns", winner["patterns"])
    print("\nSUBGROUP SUMMARY")
    print(summary_all.to_string(index=False))
    print("\nDETAIL")
    cols = [
        "dataset",
        "group",
        "group_n",
        "group_low_smi_n",
        "group_low_smi_rate",
        "operating_point",
        "clinical_sensitivity",
        "post_sensitivity",
        "sensitivity_loss",
        "sensitivity_loss_p_exact",
        "clinical_specificity",
        "post_specificity",
        "specificity_gain",
        "specificity_gain_p_exact",
        "clinical_accuracy",
        "post_accuracy",
        "accuracy_delta",
        "accuracy_delta_p_mcnemar",
        "deesc_n",
        "deesc_events",
        "deesc_event_rate",
        "deesc_event_fisher_p",
    ]
    print(detail_all[cols].to_string(index=False))
    print("out_dir", OUT_DIR)


# 최종 선정된 de-escalation 게이트를 불러와 성별/제조사 부분군별로 성능 지표를 재계산해, 특정 하위집단에서 성능이 나빠지지 않는지
# 점검하는 사후 감사(subgroup audit)를 실행하고 결과표를 outputs 폴더에 저장.
if __name__ == "__main__":
    main()
