from __future__ import annotations

# Stage-1 TP vs FP: 128-slice AEC 곡선 자체(comparison #16/#19, curve_diff_test의
# whole-curve RMSD permutation)는 유의하지 않았다. 곡선을 128개 점이 아니라 "형태"를
# 요약하는 소수의 curve-level feature(퍼짐, 비대칭도, 피크/트로프 위치, 구간별 기울기 등)로
# 압축한 뒤, 그 feature 각각에 대해 TP vs FP p-value를 비교한다. 각 feature는 여전히
# 128개 슬라이스 전체로부터 계산되는 곡선 단위 요약량이므로 point-wise 검정이 아니다
# (feedback_aec_curve_wholistic 지침 유지).
#
# 추가로 clinic feature(sex_m/age_std/height_std/weight_std, Stage-1 스크린에 실제로
# 쓰인 변수)를 같은 TP/FP 표본에 대해 동일한 방식으로 검정해, AEC-shape feature의
# p-value 양상이 clinic feature와 같은 패턴(예: 전부 유의/전부 비유의)을 보이는지,
# 아니면 clinic은 유의하고 AEC-shape는 여전히 비유의한 별개 패턴인지 나란히 비교한다.
# clinic+AEC-shape 전체를 하나의 feature 집합으로 묶어 BH-FDR도 함께 보정한다.
#
# 원본(raw, unmatched) TP/FP, propensity-matched TP/FP(comparison #19 표본, clinic
# 공변량을 매칭으로 통제), 그리고 residualized TP/FP(clinic 공변량을 매칭이 아니라
# AEC 곡선 쪽에서 회귀로 제거) 세 가지를 나란히 비교한다.
#   - matched: 표본을 clinic 공변량 기준으로 골라내 TP/FP 그룹 자체를 clinic이 비슷하게
#     맞춘다 (clinic feature의 p-value가 커져야 매칭이 의도대로 동작한 것).
#   - residualized: 표본은 그대로 두고, AEC-128 곡선에서 clinic 4변수(sex_m/age_std/
#     height_std/weight_std)로 설명되는 성분을 LinearRegression으로 빼낸 잔차 곡선에서
#     shape feature를 다시 뽑는다. 잔차화 회귀는 docs/1_aec_residual_construction_evidence.md와
#     동일한 관례로 internal 전체 코호트(n=1090, Stage-1 TP/FN/FP/TN 전부)에 적합하고,
#     external에는 frozen 적용한다(internal은 in-sample, external은 out-of-sample).
# 세 가지 모두에서 AEC-shape feature가 여전히 비유의라면, confound를 매칭으로 없애든
# AEC 쪽에서 회귀로 없애든 결론(TP/FP를 가르는 AEC 곡선 형태 신호가 없다)이 방법에
# 의존하지 않는다는 뜻이 된다.
#
# Run: python code/stage2_aec_shape_features.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
curve_mod = import_module("aec_curve_comparison")
stage2 = import_module("stage2_dataset")
group_mod = import_module("stage2_aec_group_comparisons")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "0_clinic-only_baseline" / "aec_curve_comparison"

CLIN_COLS = stage2.CLIN_COLS  # ["sex_m", "age_std", "height_std", "weight_std"]
AEC_COLS = stage2.AEC_COLS
N_SLICES = stage2.N_SLICES

SHAPE_FEATURE_COLS = [
    "std", "skew", "kurtosis",
    "peak_pos", "peak_val", "trough_pos", "trough_val", "range",
    "centroid", "early_late_diff", "slope_early", "slope_late",
    "total_abs_dev", "frac_above_center",
]

FEATURE_TYPE = {c: "clinic" for c in CLIN_COLS}
FEATURE_TYPE.update({c: "aec_shape" for c in SHAPE_FEATURE_COLS})


# ------------------------------------------------------------------ features --
def extract_shape_features(mat: np.ndarray, center: float = 1.0) -> pd.DataFrame:
    """mat: (n_patients, 128) 곡선(patient-normalized raw AEC, center=1.0 또는 clinic
    잔차화된 곡선, center=0.0) -> per-patient curve-shape feature table. Peak/trough/
    slope는 slice 노이즈를 줄이기 위해 곡선 전체를 가볍게 스무딩(rolling window=5,
    aec_curve_comparison.smooth와 동일)한 뒤 계산하고, std/skew/kurtosis/total_abs_dev/
    frac_above_center는 원곡선 값 분포를 그대로 쓴다. early_late_diff(비율이 아니라 차이)와
    centroid(원값이 아니라 |curve-center|^2 가중 위치)는 잔차 곡선처럼 center=0 근방에서
    부호가 바뀌거나 분모가 0에 가까워질 수 있는 raw-curve 전용 정의(비율/원값 가중)를 피해,
    raw(center=1)와 residual(center=0) 양쪽에서 동일한 공식으로 안전하게 계산되도록 한
    center-agnostic 정의다."""
    n, l = mat.shape
    idx = np.arange(1, l + 1)
    half = l // 2
    q = l // 4

    rows = []
    for raw in mat:
        sm = curve_mod.smooth(raw, window=5)

        peak_i = int(np.argmax(sm))
        trough_i = int(np.argmin(sm))
        slope_early = stats.linregress(idx[:half], sm[:half]).slope
        slope_late = stats.linregress(idx[half:], sm[half:]).slope
        energy = (raw - center) ** 2

        rows.append({
            "std": float(np.std(raw, ddof=1)),
            "skew": float(stats.skew(raw)),
            "kurtosis": float(stats.kurtosis(raw)),
            "peak_pos": (peak_i + 1) / l,
            "peak_val": float(sm[peak_i]),
            "trough_pos": (trough_i + 1) / l,
            "trough_val": float(sm[trough_i]),
            "range": float(sm[peak_i] - sm[trough_i]),
            "centroid": float(np.sum(idx * energy) / np.sum(energy)) / l,
            "early_late_diff": float(raw[:q].mean() - raw[-q:].mean()),
            "slope_early": float(slope_early),
            "slope_late": float(slope_late),
            "total_abs_dev": float(np.mean(np.abs(raw - center))),
            "frac_above_center": float(np.mean(raw > center)),
        })
    return pd.DataFrame(rows, columns=SHAPE_FEATURE_COLS)


def build_feature_table(clin_df: pd.DataFrame, aec_mat: np.ndarray, group: np.ndarray,
                         center: float = 1.0) -> pd.DataFrame:
    shape_feats = extract_shape_features(aec_mat, center=center)
    out = clin_df[["PatientID"] + CLIN_COLS].reset_index(drop=True).copy()
    for col in SHAPE_FEATURE_COLS:
        out[col] = shape_feats[col].to_numpy()
    out["group"] = np.asarray(group)
    return out


# ------------------------------------------------------------- residualizer --
def fit_aec_residualizer(clin_x: np.ndarray, aec_mat: np.ndarray) -> LinearRegression:
    """128개 slice를 clinic 4변수(CLIN_COLS)에 동시에(multi-output) 선형회귀 --
    docs/1_aec_residual_construction_evidence.md가 검증한 1_aec_residual_reclassify.py
    Step 2의 fit_aec_residualizer와 동일한 절차(LinearRegression, 절편 포함)."""
    reg = LinearRegression()
    reg.fit(clin_x, aec_mat)
    return reg


def apply_aec_residualizer(reg: LinearRegression, clin_x: np.ndarray, aec_mat: np.ndarray) -> np.ndarray:
    return aec_mat - reg.predict(clin_x)


# --------------------------------------------------------------------- tests --
def compare_feature(df: pd.DataFrame, feature: str, order=("TP", "FP")) -> dict:
    x = df.loc[df["group"] == order[0], feature].to_numpy()
    y = df.loc[df["group"] == order[1], feature].to_numpy()
    is_binary = set(np.unique(df[feature])) <= {0.0, 1.0}

    if is_binary:
        a, b = int(x.sum()), int(len(x) - x.sum())
        c, d = int(y.sum()), int(len(y) - y.sum())
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        return {
            "feature": feature, "feature_type": FEATURE_TYPE[feature], "test": "Fisher exact",
            f"{order[0]}_stat": a / len(x), f"{order[1]}_stat": c / len(y), "stat_label": "proportion=1",
            "effect": float(odds), "effect_name": "odds_ratio",
            "n_TP": len(x), "n_FP": len(y), "p_value": float(p),
        }

    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * u) / (len(x) * len(y))  # rank-biserial; >0 means order[0] tends larger
    return {
        "feature": feature, "feature_type": FEATURE_TYPE[feature], "test": "Mann-Whitney U",
        f"{order[0]}_stat": float(np.median(x)), f"{order[1]}_stat": float(np.median(y)), "stat_label": "median",
        "effect": float(r), "effect_name": "rank_biserial_r",
        "n_TP": len(x), "n_FP": len(y), "p_value": float(p),
    }


def compare_all_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [compare_feature(df, f) for f in features]
    res = pd.DataFrame(rows)
    reject, q, _, _ = multipletests(res["p_value"].to_numpy(), alpha=0.05, method="fdr_bh")
    res["q_value_fdr_bh"] = q
    res["significant_q<0.05"] = reject
    return res.sort_values("p_value").reset_index(drop=True)


# --------------------------------------------------------------------- plot ---
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
        plt.Rectangle((0, 0), 1, 1, color=curve_mod.COL_D, label="aec_shape"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=curve_mod.INK_SECONDARY, loc="lower right")


# ------------------------------------------------------------------- cohort ---
def run_cohort(cohort: str, stage1_rows_pos: pd.DataFrame, stage2_input_clin: pd.DataFrame,
               stage2_input_aec: pd.DataFrame, aec_reg: LinearRegression) -> None:
    out_dir = OUT_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    group = stage1_rows_pos["group"].reset_index(drop=True)
    clin_df = stage2_input_clin.reset_index(drop=True)
    aec_df = stage2_input_aec.reset_index(drop=True)
    assert (clin_df["PatientID"].to_numpy() == aec_df["PatientID"].to_numpy()).all()
    aec_mat = aec_df[AEC_COLS].to_numpy()
    clin_x = clin_df[CLIN_COLS].to_numpy()

    # --- raw (unmatched), same population as comparison #16 ---
    table_raw = build_feature_table(clin_df, aec_mat, group.to_numpy(), center=1.0)
    res_raw = compare_all_features(table_raw, CLIN_COLS + SHAPE_FEATURE_COLS)
    res_raw.to_csv(out_dir / "20_feature_pvalue_comparison_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{cohort}] raw TP vs FP (n_TP={int((group=='TP').sum())}, n_FP={int((group=='FP').sum())})")
    print(res_raw[["feature", "feature_type", "test", "p_value", "q_value_fdr_bh", "significant_q<0.05"]]
          .to_string(index=False))

    # --- propensity-matched, same population as comparison #19 (표본을 골라 clinic을 맞춤) ---
    ps = group_mod.fit_propensity(clin_df, group)
    matched_idx = group_mod.optimal_caliper_match(clin_df, group, ps)
    table_matched = build_feature_table(clin_df.iloc[matched_idx].reset_index(drop=True),
                                         aec_mat[matched_idx], group.iloc[matched_idx].to_numpy(), center=1.0)
    res_matched = compare_all_features(table_matched, CLIN_COLS + SHAPE_FEATURE_COLS)
    res_matched.to_csv(out_dir / "21_feature_pvalue_comparison_matched.csv", index=False, encoding="utf-8-sig")
    n_tp_m = int((group.iloc[matched_idx] == "TP").sum())
    n_fp_m = int((group.iloc[matched_idx] == "FP").sum())
    print(f"\n[{cohort}] propensity-matched TP vs FP (n_TP={n_tp_m}, n_FP={n_fp_m})")
    print(res_matched[["feature", "feature_type", "test", "p_value", "q_value_fdr_bh", "significant_q<0.05"]]
          .to_string(index=False))

    # --- residualized (표본은 그대로, AEC 곡선에서 clinic 성분을 회귀로 제거) ---
    aec_resid = apply_aec_residualizer(aec_reg, clin_x, aec_mat)
    table_resid = build_feature_table(clin_df, aec_resid, group.to_numpy(), center=0.0)
    res_resid = compare_all_features(table_resid, CLIN_COLS + SHAPE_FEATURE_COLS)
    res_resid.to_csv(out_dir / "22_feature_pvalue_comparison_residualized.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{cohort}] AEC-residualized TP vs FP (n_TP={int((group=='TP').sum())}, n_FP={int((group=='FP').sum())})")
    print(res_resid[["feature", "feature_type", "test", "p_value", "q_value_fdr_bh", "significant_q<0.05"]]
          .to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    plot_pvalue_panel(axes[0], res_raw, f"{cohort}: raw TP vs FP (n={len(table_raw)})")
    plot_pvalue_panel(axes[1], res_matched, f"{cohort}: propensity-matched TP vs FP (n={len(table_matched)})")
    plot_pvalue_panel(axes[2], res_resid, f"{cohort}: AEC clinic-residualized TP vs FP (n={len(table_resid)})")
    fig.suptitle("Clinic feature vs AEC curve-shape feature: TP vs FP p-value 비교 (raw / matched / residualized)",
                 fontsize=13, color=curve_mod.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    curve_mod.savefig(fig, str(out_dir), "20_feature_pvalue_comparison.png")


def main():
    screen = stage2.fit_internal_screen()

    # AEC residualizer: internal 전체 코호트(TP/FN/FP/TN 전부, n=1090)에 적합 --
    # docs/1_aec_residual_construction_evidence.md와 동일한 관례(internal fit, external frozen).
    full_internal_aec = stage2.load_aec_for_patients(stage2.INTERNAL_XLSX, screen["meta"]["PatientID"])
    aec_reg = fit_aec_residualizer(screen["x"], full_internal_aec[AEC_COLS].to_numpy())

    _, rows_pos_int, clin_int, aec_int = stage2.build_stage2_inputs(screen)
    run_cohort("gangnam", rows_pos_int, clin_int, aec_int, aec_reg)

    _, rows_pos_ext, clin_ext, aec_ext = stage2.build_stage2_inputs_external(screen)
    run_cohort("sinchon", rows_pos_ext, clin_ext, aec_ext, aec_reg)


if __name__ == "__main__":
    main()
