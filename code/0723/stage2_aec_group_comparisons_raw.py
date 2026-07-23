from __future__ import annotations

# Stage-1 TP/FP/TN 그룹 간 AEC-128 곡선 비교를, patient-wise 정규화도 global z-score
# 정규화도 거치지 않은 순수 raw 값(aec_128 시트 원본 그대로) 기준으로 재확인한다.
# patient-wise 버전은 code/stage2_aec_group_comparisons.py, global z-score 버전은
# git 이력의 이 파일 이전 버전(aec_curve_comparison_global.py의 load_data와 동일한
# 정규화) 참고. stage2_model.py의 AecBranch 입력은 여전히 patient-normalized curve
# 그대로이며, 이 스크립트는 정규화 방식에 따라 그룹 간 곡선 차이의 결론이 달라지는지
# 보기 위한 진단용 비교다:
#   16: TP vs TN (raw) -- Stage-1이 "맞춘" 두 그룹 비교, AEC-128이 애초에 Low-SMI
#       신호를 담고 있는지 보는 가장 기본적인 sanity check
#   17: FP vs TN (raw) -- "FP가 AEC상 TN을 닮는가" 가설 검정
#   18: TP vs FP (raw, unmatched) -- Stage-1 screen-positive 안에서 실제 low-SMI(TP)와
#       오탐(FP)의 raw AEC 곡선이 다른지
#   19: TP vs FP (propensity-matched) -- clinic 공변량(성별/나이/신장/체중)을
#       Hungarian 최적할당으로 매칭해 confound를 설계로 통제한 뒤 재검정
#
# 이전에는 stage2_aec_tp_tn_comparison.py(18만 생성)와
# stage2_aec_tp_fp_matched_comparison.py(19만 생성)로 나뉘어 있었고, 16/17을
# 만들던 스크립트(구 stage2_aec_tp_fp_comparison.py 계열)는 git에 커밋되지 않아
# 소실된 상태였다 -- 네 비교 모두 stage2_dataset.py/aec_curve_comparison.py의
# 같은 파이프라인을 재사용하므로 이 파일 하나로 합치고, 각 코호트별로 흩어져
# 있던 개별 summary CSV(16/16_tp_vs_fp(중복)/17/18/19)도
# 16_19_stage2_group_comparison_summary.csv 하나로 합친다. (19의 covariate
# balance는 스키마가 달라 별도 파일로 유지.)
#
# 16/17/18/19 각각 원곡선(p)뿐 아니라 1차/2차 미분곡선(dp/d2p, slice축 np.gradient --
# code/0723/aec_two_step_ensemble.py의 Step-1 모델 입력 채널과 동일 정의)까지 세 변형을
# 모두 비교해, 원곡선에서 안 보이는 형태 차이가 미분에서 드러나는지 본다.
#
# Run: python code/stage2_aec_group_comparisons.py

import io
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
curve_mod = import_module("aec_curve_comparison_global")
stage2 = import_module("stage2_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "0_clinic-only_baseline" / "aec_curve_comparison_raw"

CLIN_COLS = stage2.CLIN_COLS  # ["sex_m", "age_std", "height_std", "weight_std"]
CALIPER_MULT = 0.2
SEED = 42

# p/dp/d2p: 정규화 없는 raw AEC-128 곡선(aec_128 시트 원본 그대로)과 그 1차/2차 미분곡선
# (slice축 np.gradient, central differences) -- 미분 정의 자체는 code/0723/
# aec_two_step_ensemble.py:compute_pdpd2p와 동일하나, 그 입력 p는 patient-normalized
# curve이고 여기서는 raw curve라는 점이 다르다.
CURVE_TYPE_META = {
    "p": {"suffix": "", "label_suffix": "", "ylabel": "Raw AEC", "ref_line": None},
    "dp": {"suffix": "_dp", "label_suffix": " (dp)", "ylabel": "d(AEC)/d(slice)", "ref_line": 0.0},
    "d2p": {"suffix": "_d2p", "label_suffix": " (d2p)", "ylabel": "d²(AEC)/d(slice)²", "ref_line": 0.0},
}


def load_aec_for_patients_raw(xlsx_path: Path, patient_ids: pd.Series) -> pd.DataFrame:
    # stage2.load_aec_for_patients와 merge/정렬 로직은 동일하지만, patient-mean 나눗셈도
    # 코호트 슬라이스별 z-score도 거치지 않은 xlsx의 aec_128 시트 원본 값을 그대로 쓴다.
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    aec_df = aec[["PatientID", *stage2.AEC_COLS]].astype({col: float for col in stage2.AEC_COLS})

    order = pd.DataFrame({"PatientID": patient_ids.to_numpy(), "__row__": np.arange(len(patient_ids))})
    merged = order.merge(aec_df, on="PatientID", how="left").sort_values("__row__").drop(columns="__row__")

    missing = int(merged[stage2.AEC_COLS].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} patients in {xlsx_path.name} have no matching aec_128 row")
    return merged.reset_index(drop=True)


def curve_variant(df: pd.DataFrame, curve_type: str) -> pd.DataFrame:
    """df has AEC_COLS holding raw (unnormalized) AEC (p) plus other columns
    (PatientID, group, ...). Returns a copy with AEC_COLS replaced by the requested
    derivative; "p" is returned unchanged."""
    if curve_type == "p":
        return df
    out = df.copy()
    p = df[curve_mod.AEC_COLS].to_numpy(dtype=float)
    dp = np.gradient(p, axis=1)
    mat = dp if curve_type == "dp" else np.gradient(dp, axis=1)
    out[curve_mod.AEC_COLS] = mat
    return out


def _run_curve_comparison(out_dir: Path, df: pd.DataFrame, order: list[str], labels: list[str],
                           colors: list[str], title: str, fig_name: str, comparison_label: str,
                           curve_type: str = "p") -> dict:
    meta = CURVE_TYPE_META[curve_type]
    df = curve_variant(df, curve_type)
    stem, ext = fig_name.rsplit(".", 1)
    fig_name = f"{stem}{meta['suffix']}.{ext}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    curve_mod.plot_curve_comparison(ax, df, "group", order, labels, colors, title + meta["label_suffix"],
                                     ylabel=meta["ylabel"], ref_line=meta["ref_line"])
    r = curve_mod.curve_diff_test(df, "group", order, labels)
    ax.text(0.02, 0.02, curve_mod.curve_diff_note(r),
            transform=ax.transAxes, fontsize=8, color=curve_mod.INK_MUTED, va="bottom")
    fig.tight_layout()
    curve_mod.savefig(fig, str(out_dir), fig_name)
    return {"figure": fig_name, "comparison": comparison_label + meta["label_suffix"],
            "curve_type": curve_type, **r}


# --------------------------------------------------------- propensity matching --
def fit_propensity(clin_df: pd.DataFrame, group: pd.Series) -> np.ndarray:
    x = clin_df[CLIN_COLS].to_numpy()
    y = (group.to_numpy() == "TP").astype(int)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=SEED)
    model.fit(x, y)
    return model.predict_proba(x)[:, 1]


def optimal_caliper_match(clin_df: pd.DataFrame, group: pd.Series, ps: np.ndarray) -> np.ndarray:
    # 소수그룹(TP) 전체와 다수그룹(FP) 사이의 총 logit(propensity) 거리를 최소화하는
    # 전역 최적 1:1 할당(Hungarian algorithm)을 구한 뒤, caliper를 넘는 쌍만 사후에 버린다.
    eps = 1e-6
    logit_ps = np.log(np.clip(ps, eps, 1 - eps) / np.clip(1 - ps, eps, 1 - eps))
    caliper = CALIPER_MULT * logit_ps.std(ddof=1)

    is_tp = (group.to_numpy() == "TP")
    tp_idx = np.where(is_tp)[0]
    fp_idx = np.where(~is_tp)[0]

    cost = np.abs(logit_ps[tp_idx][:, None] - logit_ps[fp_idx][None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = [(tp_idx[r], fp_idx[c]) for r, c in zip(row_ind, col_ind) if cost[r, c] <= caliper]

    matched_idx = np.array(sorted(i for pair in matched_pairs for i in pair))
    print(f"matched pairs: {len(matched_pairs)} / TP total {len(tp_idx)} (caliper={caliper:.4f} logit-units, optimal assignment)")
    return matched_idx


def smd(x: np.ndarray, treat_mask: np.ndarray) -> float:
    a, b = x[treat_mask], x[~treat_mask]
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0


def balance_table(clin_df: pd.DataFrame, group: pd.Series, matched_idx: np.ndarray) -> pd.DataFrame:
    treat_mask_all = (group.to_numpy() == "TP")
    treat_mask_matched = treat_mask_all[matched_idx]
    rows = []
    for col in CLIN_COLS:
        x_all = clin_df[col].to_numpy()
        x_matched = x_all[matched_idx]
        rows.append({
            "covariate": col,
            "smd_before": smd(x_all, treat_mask_all),
            "smd_after": smd(x_matched, treat_mask_matched),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- cohort --
def run_cohort(cohort: str, stage1_rows_all: pd.DataFrame, stage1_rows_pos: pd.DataFrame,
               stage2_input_clin: pd.DataFrame, data_xlsx: Path) -> None:
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    curve_types = ["p", "dp", "d2p"]

    # --- 16: TP vs TN (raw) ---
    tp_tn_rows = stage1_rows_all[stage1_rows_all["group"].isin(["TP", "TN"])].reset_index(drop=True)
    aec_tp_tn = load_aec_for_patients_raw(data_xlsx, tp_tn_rows["PatientID"])
    df_tp_tn = aec_tp_tn.merge(tp_tn_rows[["PatientID", "group"]], on="PatientID", how="inner")
    assert len(df_tp_tn) == len(tp_tn_rows), f"{cohort}: TP/TN merge dropped rows"
    for ct in curve_types:
        summary_rows.append(_run_curve_comparison(
            out_dir, df_tp_tn, ["TP", "TN"], ["TP (true low-SMI)", "TN (true negative)"],
            [curve_mod.COL_A, curve_mod.COL_C],
            f"Stage-1 ({cohort}): TP vs TN AEC 곡선 비교 (Stage-1이 맞춘 두 그룹)",
            "16_aec_curve_tp_vs_tn.png", "Stage-1 TP vs TN", curve_type=ct,
        ))

    # --- 17: FP vs TN (raw) ---
    fp_tn_rows = stage1_rows_all[stage1_rows_all["group"].isin(["FP", "TN"])].reset_index(drop=True)
    aec_fp_tn = load_aec_for_patients_raw(data_xlsx, fp_tn_rows["PatientID"])
    df_fp_tn = aec_fp_tn.merge(fp_tn_rows[["PatientID", "group"]], on="PatientID", how="inner")
    assert len(df_fp_tn) == len(fp_tn_rows), f"{cohort}: FP/TN merge dropped rows"
    for ct in curve_types:
        summary_rows.append(_run_curve_comparison(
            out_dir, df_fp_tn, ["FP", "TN"], ["FP (false positive)", "TN (true negative)"],
            [curve_mod.COL_B, curve_mod.COL_C],
            f"Stage-1 ({cohort}): FP vs TN AEC 곡선 비교",
            "17_aec_curve_fp_vs_tn.png", "Stage-1 FP vs TN", curve_type=ct,
        ))

    # --- 18: TP vs FP (raw, unmatched) ---
    aec_pos_raw = load_aec_for_patients_raw(data_xlsx, stage1_rows_pos["PatientID"])
    df_tp_fp = aec_pos_raw.merge(stage1_rows_pos[["PatientID", "group"]], on="PatientID", how="inner")
    assert len(df_tp_fp) == len(aec_pos_raw), f"{cohort}: TP/FP merge dropped rows"
    for ct in curve_types:
        summary_rows.append(_run_curve_comparison(
            out_dir, df_tp_fp, ["TP", "FP"], ["TP (true low-SMI)", "FP (false positive)"],
            [curve_mod.COL_A, curve_mod.COL_B],
            f"Stage-1 screen-positive ({cohort}): TP vs FP AEC 곡선 비교",
            "18_aec_curve_tp_vs_fp.png", "Stage-1 TP vs FP", curve_type=ct,
        ))

    # --- 19: TP vs FP (propensity-matched) ---
    group = stage1_rows_pos["group"].reset_index(drop=True)
    clin_df = stage2_input_clin.reset_index(drop=True)
    aec_df = aec_pos_raw.reset_index(drop=True)
    assert (clin_df["PatientID"].to_numpy() == aec_df["PatientID"].to_numpy()).all()

    ps = fit_propensity(clin_df, group)
    matched_idx = optimal_caliper_match(clin_df, group, ps)

    bal = balance_table(clin_df, group, matched_idx)
    bal_path = out_dir / "19_aec_curve_tp_vs_fp_matched_balance.csv"
    bal.to_csv(bal_path, index=False, encoding="utf-8-sig")
    print(f"[{cohort}] covariate balance (SMD):\n{bal.to_string(index=False)}")
    print(f"saved: {bal_path}")

    df_matched = aec_df.iloc[matched_idx].copy()
    df_matched["group"] = group.iloc[matched_idx].to_numpy()
    for ct in curve_types:
        r19 = _run_curve_comparison(
            out_dir, df_matched, ["TP", "FP"], ["TP (true low-SMI)", "FP (false positive)"],
            [curve_mod.COL_A, curve_mod.COL_B],
            f"Stage-1 screen-positive ({cohort}): TP vs FP AEC 곡선 비교 (propensity-matched)",
            "19_aec_curve_tp_vs_fp_matched.png", "Stage-1 TP vs FP (propensity-matched)", curve_type=ct,
        )
        r19["n_matched_pairs"] = len(matched_idx) // 2
        summary_rows.append(r19)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["significant_p<0.05"] = summary_df["p_value"] < 0.05
    summary_path = out_dir / "16_19_stage2_group_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"saved: {summary_path}")

    for row in summary_rows:
        print(f"[{cohort}] {row['comparison']}: {curve_mod.curve_diff_note(row)}")


def main():
    screen = stage2.fit_internal_screen()

    rows_all_int, rows_pos_int, clin_int, _ = stage2.build_stage2_inputs(screen)
    run_cohort("gangnam", rows_all_int, rows_pos_int, clin_int, stage2.INTERNAL_XLSX)

    rows_all_ext, rows_pos_ext, clin_ext, _ = stage2.build_stage2_inputs_external(screen)
    run_cohort("sinchon", rows_all_ext, rows_pos_ext, clin_ext, stage2.EXTERNAL_XLSX)


if __name__ == "__main__":
    main()
