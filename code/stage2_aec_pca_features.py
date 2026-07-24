from __future__ import annotations

# Stage-1 TP vs FP: 14개 hand-crafted curve-shape feature(stage2_aec_shape_features.py)에
# 이어, AEC-128 곡선을 PCA로 압축한 주성분 점수에서도 TP/FP를 가르는 신호가 있는지 확인한다.
# hand-crafted feature는 사람이 고른 정의(peak/trough/slope 등)에 좌우될 수 있지만, PCA는
# 곡선의 분산을 가장 잘 설명하는 방향을 데이터 기반으로 뽑으므로 별개의 독립적인 검증이 된다.
#
# PC1-5(내부 코호트 기준 누적 분산설명 96.8%)를 사용. PCA는 internal 전체 코호트(n=1090,
# Stage-1 TP/FN/FP/TN 전부)에 적합하고 external에는 frozen 적용(잔차화 회귀와 동일 관례,
# docs/1_aec_residual_construction_evidence.md 참고). raw/residualized 각각 별도로 PCA를
# 적합한다(raw는 patient-normalized 곡선 자체, residualized는 clinic 4변수 성분을 제거한
# 잔차 곡선 -- 서로 스케일과 중심이 다르므로 같은 PCA를 공유하지 않는다).
#
# raw / propensity-matched(comparison #19 표본) / clinic-residualized 세 방법을 나란히
# 비교하는 구조는 stage2_aec_shape_features.py와 동일. 각 PC 점수는 여전히 128개 슬라이스
# 전체로부터 계산되는 곡선 단위 요약량이므로 point-wise 검정이 아니다
# (feedback_aec_curve_wholistic 지침 유지).
#
# Run: python code/stage2_aec_pca_features.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
curve_mod = import_module("aec_curve_comparison")
stage2 = import_module("stage2_dataset")
group_mod = import_module("stage2_aec_group_comparisons")
shape_mod = import_module("stage2_aec_shape_features")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "0_clinic-only_baseline" / "aec_curve_comparison"

CLIN_COLS = stage2.CLIN_COLS  # ["sex_m", "age_std", "height_std", "weight_std"]
AEC_COLS = stage2.AEC_COLS

N_PCA = 5
PCA_COLS = [f"pc{i+1}" for i in range(N_PCA)]

FEATURE_TYPE = {c: "clinic" for c in CLIN_COLS}
FEATURE_TYPE.update({c: "aec_pca" for c in PCA_COLS})


def build_feature_table(clin_df: pd.DataFrame, pc_scores: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    out = clin_df[["PatientID"] + CLIN_COLS].reset_index(drop=True).copy()
    for i, col in enumerate(PCA_COLS):
        out[col] = pc_scores[:, i]
    out["group"] = np.asarray(group)
    return out


def compare_all_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    # reuse the exact same test + BH-FDR logic as the hand-crafted shape-feature analysis,
    # just pointed at FEATURE_TYPE/PCA_COLS defined in this module
    orig_feature_type = shape_mod.FEATURE_TYPE
    shape_mod.FEATURE_TYPE = FEATURE_TYPE
    try:
        return shape_mod.compare_all_features(df, features)
    finally:
        shape_mod.FEATURE_TYPE = orig_feature_type


def plot_pvalue_panel(ax, res: pd.DataFrame, title: str):
    res = res.sort_values("p_value", ascending=False)
    y = np.arange(len(res))
    colors = [curve_mod.COL_C if t == "clinic" else curve_mod.COL_D for t in res["feature_type"]]
    neglogp = -np.log10(res["p_value"].to_numpy())

    ax.barh(y, neglogp, color=colors, height=0.65, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(res["feature"])
    ax.axvline(-np.log10(0.05), color=curve_mod.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(-np.log10(0.05), len(res) - 0.3, " p=0.05", fontsize=7, color=curve_mod.INK_MUTED, va="top")

    for yi, is_sig, p in zip(y, res["significant_q<0.05"], res["p_value"]):
        if is_sig:
            ax.text(-np.log10(p) + 0.05, yi, "q<0.05", fontsize=7, color=curve_mod.INK_PRIMARY,
                    va="center", fontweight="bold")

    ax.set_xlabel("-log10(p)  [Mann-Whitney U / Fisher exact, TP vs FP]")
    ax.set_title(title, color=curve_mod.INK_PRIMARY, fontsize=10)
    curve_mod.style_axes(ax)
    ax.xaxis.grid(True, color=curve_mod.GRID, linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=curve_mod.COL_C, label="clinic"),
        plt.Rectangle((0, 0), 1, 1, color=curve_mod.COL_D, label="aec_pca"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=curve_mod.INK_SECONDARY, loc="lower right")


def run_cohort(cohort: str, stage1_rows_pos: pd.DataFrame, stage2_input_clin: pd.DataFrame,
               stage2_input_aec: pd.DataFrame, pca_raw: PCA, pca_resid: PCA,
               aec_reg) -> None:
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    group = stage1_rows_pos["group"].reset_index(drop=True)
    clin_df = stage2_input_clin.reset_index(drop=True)
    aec_df = stage2_input_aec.reset_index(drop=True)
    assert (clin_df["PatientID"].to_numpy() == aec_df["PatientID"].to_numpy()).all()
    aec_mat = aec_df[AEC_COLS].to_numpy()
    clin_x = clin_df[CLIN_COLS].to_numpy()

    # --- raw (unmatched), same population as comparison #16 ---
    pc_raw = pca_raw.transform(aec_mat)[:, :N_PCA]
    table_raw = build_feature_table(clin_df, pc_raw, group.to_numpy())
    res_raw = compare_all_features(table_raw, CLIN_COLS + PCA_COLS)
    res_raw.to_csv(out_dir / "23_pca_pvalue_comparison_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{cohort}] raw TP vs FP, PCA (n_TP={int((group=='TP').sum())}, n_FP={int((group=='FP').sum())})")
    print(res_raw[["feature", "feature_type", "test", "p_value", "q_value_fdr_bh", "significant_q<0.05"]]
          .to_string(index=False))

    # --- propensity-matched, same population as comparison #19 ---
    ps = group_mod.fit_propensity(clin_df, group)
    matched_idx = group_mod.optimal_caliper_match(clin_df, group, ps)
    pc_matched = pca_raw.transform(aec_mat[matched_idx])[:, :N_PCA]
    table_matched = build_feature_table(clin_df.iloc[matched_idx].reset_index(drop=True),
                                         pc_matched, group.iloc[matched_idx].to_numpy())
    res_matched = compare_all_features(table_matched, CLIN_COLS + PCA_COLS)
    res_matched.to_csv(out_dir / "24_pca_pvalue_comparison_matched.csv", index=False, encoding="utf-8-sig")
    n_tp_m = int((group.iloc[matched_idx] == "TP").sum())
    n_fp_m = int((group.iloc[matched_idx] == "FP").sum())
    print(f"\n[{cohort}] propensity-matched TP vs FP, PCA (n_TP={n_tp_m}, n_FP={n_fp_m})")
    print(res_matched[["feature", "feature_type", "test", "p_value", "q_value_fdr_bh", "significant_q<0.05"]]
          .to_string(index=False))

    # --- residualized ---
    aec_resid = shape_mod.apply_aec_residualizer(aec_reg, clin_x, aec_mat)
    pc_resid = pca_resid.transform(aec_resid)[:, :N_PCA]
    table_resid = build_feature_table(clin_df, pc_resid, group.to_numpy())
    res_resid = compare_all_features(table_resid, CLIN_COLS + PCA_COLS)
    res_resid.to_csv(out_dir / "25_pca_pvalue_comparison_residualized.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{cohort}] AEC-residualized TP vs FP, PCA (n_TP={int((group=='TP').sum())}, n_FP={int((group=='FP').sum())})")
    print(res_resid[["feature", "feature_type", "test", "p_value", "q_value_fdr_bh", "significant_q<0.05"]]
          .to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    plot_pvalue_panel(axes[0], res_raw, f"{cohort}: raw TP vs FP, PCA (n={len(table_raw)})")
    plot_pvalue_panel(axes[1], res_matched, f"{cohort}: propensity-matched TP vs FP, PCA (n={len(table_matched)})")
    plot_pvalue_panel(axes[2], res_resid, f"{cohort}: AEC clinic-residualized TP vs FP, PCA (n={len(table_resid)})")
    fig.suptitle(f"Clinic feature vs AEC PCA(PC1-{N_PCA}) feature: TP vs FP p-value 비교 (raw / matched / residualized)",
                 fontsize=13, color=curve_mod.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    curve_mod.savefig(fig, str(out_dir), "23_pca_pvalue_comparison.png")


def main():
    screen = stage2.fit_internal_screen()

    full_internal_aec = stage2.load_aec_for_patients(stage2.INTERNAL_XLSX, screen["meta"]["PatientID"])
    full_internal_mat = full_internal_aec[AEC_COLS].to_numpy()

    pca_raw = PCA(n_components=N_PCA).fit(full_internal_mat)
    print("PCA (raw) explained variance ratio:", np.round(pca_raw.explained_variance_ratio_, 4),
          "cumulative:", round(float(np.sum(pca_raw.explained_variance_ratio_)), 4))

    aec_reg = shape_mod.fit_aec_residualizer(screen["x"], full_internal_mat)
    full_internal_resid = shape_mod.apply_aec_residualizer(aec_reg, screen["x"], full_internal_mat)
    pca_resid = PCA(n_components=N_PCA).fit(full_internal_resid)
    print("PCA (residualized) explained variance ratio:", np.round(pca_resid.explained_variance_ratio_, 4),
          "cumulative:", round(float(np.sum(pca_resid.explained_variance_ratio_)), 4))

    _, rows_pos_int, clin_int, aec_int = stage2.build_stage2_inputs(screen)
    run_cohort("gangnam", rows_pos_int, clin_int, aec_int, pca_raw, pca_resid, aec_reg)

    _, rows_pos_ext, clin_ext, aec_ext = stage2.build_stage2_inputs_external(screen)
    run_cohort("sinchon", rows_pos_ext, clin_ext, aec_ext, pca_raw, pca_resid, aec_reg)


if __name__ == "__main__":
    main()
