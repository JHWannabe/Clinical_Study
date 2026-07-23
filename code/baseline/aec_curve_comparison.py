"""
data/gangnam.xlsx, data/sinchon.xlsx 각각의 'aec_128' 시트(환자별 128-slice raw
AEC 프로파일)를 이용해 성별 / 나이 / TAMA / BMI / Height / Weight 그룹 간 AEC
"point curve"(슬라이스별 평균 곡선 + 신뢰구간 리본) 비교 그래프를 그린다.

- 강남(gangnam), 신촌(sinchon) 코호트를 각각 독립적으로 분석하여
  outputs/0_aec_curve_comparison/{gangnam,sinchon}/ 에 결과를 저장한다.
- 각 환자 곡선은 자기 자신의 평균값으로 나눠 정규화한다 (patient-normalized AEC).
- 그룹별 정규화 곡선을 슬라이스 index(1~128)마다 평균 + 95% CI로 겹쳐 그린다.
- 연속형 변수 중 TAMA/BMI는 남/여 각각의 median 기준 상/하 2그룹으로 나눈다
  (체격 지표라 성별에 따라 분포 자체가 다르므로 전체 median 하나로 나누면
  그룹이 성별과 뒤섞임). 그 외 연속형 변수(Age/Height/Weight)는 전체
  데이터셋의 median 기준 상/하 2그룹으로 나눈다.
- 레거시 파이프라인(main_aec_full_derivation_pipeline_simplified.py 등)은
  재사용하지 않고 이 스크립트에서 새로 계산한다.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- palette --
COL_A = "#2d2ad6"   # 그룹 1 (blue)
COL_B = "#af1b1b"   # 그룹 2 (red)
COL_C = "#eda100"   # 그룹 3 (yellow) - 3그룹 이상 비교용
COL_D = "#3aa74c"   # 그룹 4 (green) - 3그룹 이상 비교용
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "../..", "outputs", "0_clinic-only_baseline", "aec_curve_comparison")
COHORTS = ["gangnam", "sinchon"]

N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def savefig(fig, out_dir, name):
    fig.patch.set_facecolor(SURFACE)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved: {path}")


VENDOR_MAP = {
    "Sensation 64": "Siemens",
    "SOMATOM Definition AS+": "Siemens",
    "SOMATOM Definition Edge": "Siemens",
    "SOMATOM Definition": "Siemens",
    "SOMATOM Definition Flash": "Siemens",
    "SOMATOM Force": "Siemens",
    "SOMATOM Drive": "Siemens",
    "SOMATOM go.Top": "Siemens",
    "Revolution CT": "GE",
    "Revolution EVO": "GE",
    "Revolution Frontier": "GE",
    "Optima CT660": "GE",
    "LightSpeed VCT": "GE",
    "Discovery CT750 HD": "GE",
    "Ingenuity Core 128": "Philips",
    "iCT 256": "Philips",
    "Aquilion ONE": "Canon",
    "Aquilion": "Canon",
}

# 프로젝트 기존 low-SMI 임상 cutoff (main_aec_full_derivation_pipeline_simplified.py 정의값과 동일:
# 남성 <45.4, 여성 <34.4 cm^2/m^2). 값만 재사용하고 계산/파이프라인 로직은 새로 작성.
LOW_SMI_CUTOFF = {"M": 45.4, "F": 34.4}


def sex_median_group2(df, col):
    # 전체 median이 아니라 남/여 각각의 median으로 Low/High를 나눈다 (성별에 따라
    # 분포 자체가 다른 변수를 하나의 median으로 나누면 그룹이 성별과 뒤섞여 Simpson's
    # paradox식 교란이 생김). TAMA/BMI처럼 체격 지표라 성별 분포 차이가 큰 변수에만 사용.
    sex = df["PatientSex"].astype(str).str.upper()
    group = pd.Series(index=df.index, dtype=object)
    for s in sex.unique():
        mask = (sex == s).to_numpy()
        med = df.loc[mask, col].median()
        group.loc[mask] = np.where(df.loc[mask, col] <= med, "Low", "High")
    return group.to_numpy()


def overall_median_group2(df, col):
    # 전체 데이터셋 median 하나로 Low/High를 나눈다. TAMA/BMI를 제외한 나머지
    # 변수들은 성별 분리 기준을 요구하지 않아 단일 median 분할을 사용.
    med = df[col].median()
    return np.where(df[col] <= med, "Low", "High")


def load_data(data_path):
    meta = pd.read_excel(data_path, sheet_name="metadata")
    aec = pd.read_excel(data_path, sheet_name="aec_128")

    curves = aec[AEC_COLS].astype(float).to_numpy()
    patient_mean = curves.mean(axis=1, keepdims=True)
    norm_curves = curves / patient_mean  # patient-normalized AEC

    norm_df = pd.DataFrame(norm_curves, columns=AEC_COLS)
    norm_df.insert(0, "PatientID", aec["PatientID"].to_numpy())

    df = meta.merge(norm_df, on="PatientID", how="inner")

    df["AgeGroup2"] = overall_median_group2(df, "PatientAge")
    for col in ["TAMA", "BMI"]:
        df[f"{col}Group2"] = sex_median_group2(df, col)
    for col in ["Height", "Weight"]:
        df[f"{col}Group2"] = overall_median_group2(df, col)

    # SMI is recomputed from TAMA/Height (not read from the metadata sheet's own
    # "SMI" column) to match clinic-only_baseline.load_cohort() exactly -- the
    # ground-truth y actually used to train/evaluate the Stage-1/Stage-2 models.
    # The metadata SMI column disagrees substantially (internal Low-SMI n=291 vs
    # n=129 here; max abs SMI diff ~13.8), which silently made this script's
    # Low-SMI grouping inconsistent with the model's own labels.
    smi = df["TAMA"] / (df["Height"] / 100.0) ** 2
    cutoff = df["PatientSex"].map(LOW_SMI_CUTOFF)
    df["LowSMI"] = np.where(smi < cutoff, "Low SMI", "Non-low SMI")

    df["Vendor"] = df["Manufacturer"].map(VENDOR_MAP)

    bmi_bins = [0, 18.5, 23, 25, 100]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMIGroup4"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=False)

    df["SexSMIGroup"] = df["PatientSex"] + " / " + df["LowSMI"]

    return df


def smooth(mat_mean, window=5):
    s = pd.Series(mat_mean)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def group_curve_stats(df, group_col, group_val, cols=None):
    cols = list(cols) if cols is not None else AEC_COLS
    sub = df.loc[df[group_col] == group_val, cols].to_numpy()
    mean = sub.mean(axis=0)
    sem = stats.sem(sub, axis=0)
    ci = 1.96 * sem
    return smooth(mean), smooth(ci), len(sub)


def plot_curve_comparison(ax, df, group_col, order, labels, colors, title,
                           ylabel="Patient-normalized AEC", ref_line=1.0, cols=None):
    cols = list(cols) if cols is not None else AEC_COLS
    x = np.array([int(c.split("_")[1]) for c in cols])
    for val, label, color in zip(order, labels, colors):
        mean, ci, n = group_curve_stats(df, group_col, val, cols=cols)
        ax.plot(x, mean, color=color, linewidth=2, label=f"{label} (n={n})")
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    ax.axhline(ref_line, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Slice index (1-128, resampled)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="best")
    style_axes(ax)


def curve_diff_test(df, group_col, order, labels, n_perm=2000, seed=42, cols=None):
    """128 슬라이스를 개별 포인트로 따로따로 검정하지 않고, 곡선(또는 곡선의 부분구간)을
    하나의 벡터로 보고 그룹 간 차이를 정량화한다. cols를 지정하면 전체 128슬라이스 대신
    그 부분집합(예: peak_slice 주변 구간)을 하나의 sub-curve로 취급해 같은 방식으로
    검정한다 -- 슬라이스를 독립된 포인트로 쪼개서 보지 않는다는 원칙은 구간 단위에서도 유지.

    환자별로 자기 자신의 평균으로 정규화했기 때문에 patient-mean AEC(128슬라이스 평균)는
    모든 환자에서 항상 정확히 1.0이 되어(구성상 자명) 그룹 간 비교 대상이 될 수 없다.
    대신 두 그룹의 평균곡선(길이 len(cols) 벡터) 사이의 RMSD(root-mean-square deviation,
    구간 전체에 걸친 평균적 곡선 간 거리 - 2그룹 기준. 3그룹 이상은 슬라이스별
    그룹평균 간 분산을 구간 전체에 대해 평균한 값)를 하나의 전역 검정통계량으로 삼고,
    그룹 라벨을 섞는 permutation test로 그 거리가 우연 수준을 넘는지 검정한다.
    peak_slice/peak_deviation은 어디서 가장 크게 벌어지는지 보여주는 참고 정보일 뿐,
    검정 자체는 구간 전체(RMSD)를 기준으로 한다.
    """
    cols = list(cols) if cols is not None else AEC_COLS
    slice_nums = [int(c.split("_")[1]) for c in cols]
    sub = df[df[group_col].isin(order)]
    mat = sub[cols].to_numpy()
    labels_arr = sub[group_col].to_numpy()
    ns = [int((labels_arr == v).sum()) for v in order]

    def curve_stat(lab):
        means = np.stack([mat[lab == v].mean(axis=0) for v in order])
        deviation = (means[0] - means[1]) if len(order) == 2 else means.std(axis=0)
        rmsd = float(np.sqrt(np.mean(deviation ** 2)))
        return rmsd, deviation

    obs_stat, obs_deviation = curve_stat(labels_arr)
    peak_idx = int(np.argmax(np.abs(obs_deviation)))

    rng = np.random.default_rng(seed)
    perm_labels = labels_arr.copy()
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(perm_labels)
        perm_stats[i] = curve_stat(perm_labels)[0]
    p = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)

    direction = ""
    if len(order) == 2:
        direction = f"{labels[0]} {'>' if obs_deviation[peak_idx] > 0 else '<'} {labels[1]}"

    return {
        "group_col": group_col,
        "groups": "; ".join(f"{lab} (n={n})" for lab, n in zip(labels, ns)),
        "test": "whole-curve RMSD permutation",
        "curve_rmsd": obs_stat,
        "peak_slice": slice_nums[peak_idx],
        "peak_deviation": float(obs_deviation[peak_idx]),
        "direction": direction,
        "n_perm": n_perm,
        "p_value": float(p),
    }


def curve_diff_note(r):
    return (f"curve RMSD={r['curve_rmsd']:.4f}, perm p={r['p_value']:.3g} "
            f"(n_perm={r['n_perm']}; peak Δ={r['peak_deviation']:.3f} @ slice {r['peak_slice']})")


# 전체 곡선 permutation test(curve_diff_test)의 peak_slice를 중심으로 한 구간(sub-curve)만
# 잘라 같은 RMSD permutation test를 다시 돌린다. 128포인트를 임의로 등분(4등분/3등분)하는
# 대신, 이미 전체곡선 비교에서 그룹 차이가 가장 크게 나타난 지점을 기준으로 구간을 정의한다.
PEAK_WINDOW_HALFWIDTH = 10  # peak_slice 기준 좌우 슬라이스 수


def peak_window_cols(peak_slice, halfwidth=PEAK_WINDOW_HALFWIDTH):
    lo = max(1, peak_slice - halfwidth)
    hi = min(N_SLICES, peak_slice + halfwidth)
    return [f"aec_{i}" for i in range(lo, hi + 1)], lo, hi


# ------------------------------------------------------- propensity matching --
# 07(단순 median/cutoff 분할)은 Low-SMI 여부와 성별/나이/키/몸무게가 서로 얽혀 있을 때
# 그 얽힘을 통제하지 못한다. 07b는 같은 두 그룹을 stage2_dataset.CLIN_COLS와 동일한
# 정의(sex_m/age_std/height_std/weight_std)로 propensity score를 추정하고, TP vs FP
# 비교(code/0723/stage2_aec_group_comparisons.py)와 동일한 Hungarian 최적 1:1 caliper
# matching으로 공변량을 맞춘 뒤 재비교한다.
PSM_COVARIATES = ["sex_m", "age_std", "height_std", "weight_std"]
CALIPER_MULT = 0.2
PSM_SEED = 42


def build_psm_covariates(df):
    out = pd.DataFrame(index=df.index)
    out["sex_m"] = (df["PatientSex"].astype(str).str.upper() == "M").astype(float)
    for col, std_col in [("PatientAge", "age_std"), ("Height", "height_std"), ("Weight", "weight_std")]:
        x = df[col].astype(float)
        out[std_col] = (x - x.mean()) / x.std(ddof=1)
    return out


def fit_propensity(clin_df, group, treat_val):
    x = clin_df[PSM_COVARIATES].to_numpy()
    y = (group.to_numpy() == treat_val).astype(int)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=PSM_SEED)
    model.fit(x, y)
    return model.predict_proba(x)[:, 1]


def optimal_caliper_match(group, ps, treat_val):
    # treat_val 그룹 전체와 나머지 그룹 사이의 총 logit(propensity) 거리를 최소화하는 전역
    # 최적 1:1 할당(Hungarian algorithm)을 구한 뒤, caliper를 넘는 쌍만 사후에 버린다.
    eps = 1e-6
    logit_ps = np.log(np.clip(ps, eps, 1 - eps) / np.clip(1 - ps, eps, 1 - eps))
    caliper = CALIPER_MULT * logit_ps.std(ddof=1)

    is_treat = (group.to_numpy() == treat_val)
    treat_idx = np.where(is_treat)[0]
    control_idx = np.where(~is_treat)[0]

    cost = np.abs(logit_ps[treat_idx][:, None] - logit_ps[control_idx][None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = [(treat_idx[r], control_idx[c]) for r, c in zip(row_ind, col_ind) if cost[r, c] <= caliper]
    matched_idx = np.array(sorted(i for pair in matched_pairs for i in pair))
    print(f"  matched pairs: {len(matched_pairs)} / {treat_val} total {len(treat_idx)} "
          f"(caliper={caliper:.4f} logit-units, optimal assignment)")
    return matched_idx


def smd(x, treat_mask):
    a, b = x[treat_mask], x[~treat_mask]
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0


def psm_balance_table(clin_df, group, treat_val, matched_idx):
    treat_mask_all = (group.to_numpy() == treat_val)
    treat_mask_matched = treat_mask_all[matched_idx]
    rows = []
    for col in PSM_COVARIATES:
        x_all = clin_df[col].to_numpy()
        x_matched = x_all[matched_idx]
        rows.append({
            "covariate": col,
            "smd_before": smd(x_all, treat_mask_all),
            "smd_after": smd(x_matched, treat_mask_matched),
        })
    return pd.DataFrame(rows)


def run_cohort(cohort):
    data_path = os.path.join(DATA_DIR, f"{cohort}.xlsx")
    out_dir = os.path.join(OUT_ROOT, cohort)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== cohort: {cohort} ===")
    df = load_data(data_path)
    print(f"merged patients: {len(df)}")

    specs = [
        ("PatientSex", ["M", "F"], ["Male", "Female"], "01_aec_curve_by_sex.png", "성별에 따른 AEC 곡선 비교"),
        ("AgeGroup2", ["Low", "High"], ["Age ≤ median", "Age > median"],
         "02_aec_curve_by_age.png", "나이(median 분할)에 따른 AEC 곡선 비교"),
        ("TAMAGroup2", ["Low", "High"], ["TAMA ≤ 성별 median", "TAMA > 성별 median"],
         "03_aec_curve_by_tama.png", "TAMA(성별 median 분할)에 따른 AEC 곡선 비교"),
        ("HeightGroup2", ["Low", "High"], ["Height ≤ median", "Height > median"],
         "05_aec_curve_by_height.png", "신장(median 분할)에 따른 AEC 곡선 비교"),
        ("WeightGroup2", ["Low", "High"], ["Weight ≤ median", "Weight > median"],
         "06_aec_curve_by_weight.png", "체중(median 분할)에 따른 AEC 곡선 비교"),
        ("BMIGroup2", ["Low", "High"], ["BMI ≤ 성별 median", "BMI > 성별 median"],
         "04_aec_curve_by_bmi.png", "BMI(성별 median 분할)에 따른 AEC 곡선 비교"),
        ("LowSMI", ["Low SMI", "Non-low SMI"], ["Low SMI", "Non-low SMI"],
         "07_aec_curve_by_low_smi.png", "Low-SMI 임상 cutoff에 따른 AEC 곡선 비교"),
    ]

    two_group_colors = [COL_A, COL_B]
    summary_rows = []

    # 개별 그래프 (2그룹 비교)
    r_lowsmi_full = None
    for group_col, order, labels, fname, title_kr in specs:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plot_curve_comparison(ax, df, group_col, order, labels, two_group_colors, title_kr)
        r = curve_diff_test(df, group_col, order, labels)
        if group_col == "LowSMI":
            r_lowsmi_full = r
        summary_rows.append({"figure": fname, "comparison": title_kr, **r})
        ax.text(0.02, 0.02, curve_diff_note(r),
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        fig.tight_layout()
        savefig(fig, out_dir, fname)

    # 07b: Low-SMI, propensity score matching (성별/나이/키/몸무게 균형 후 재비교)
    clin_df = build_psm_covariates(df)
    smi_group = df["LowSMI"]
    ps = fit_propensity(clin_df, smi_group, "Low SMI")
    matched_idx = optimal_caliper_match(smi_group, ps, "Low SMI")

    bal = psm_balance_table(clin_df, smi_group, "Low SMI", matched_idx)
    bal_path = os.path.join(out_dir, "07b_aec_curve_by_low_smi_matched_balance.csv")
    bal.to_csv(bal_path, index=False, encoding="utf-8-sig")
    print(f"[{cohort}] Low-SMI PSM covariate balance (SMD):\n{bal.to_string(index=False)}")
    print(f"saved: {bal_path}")

    df_matched = df.iloc[matched_idx].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_curve_comparison(ax, df_matched, "LowSMI", ["Low SMI", "Non-low SMI"], ["Low SMI", "Non-low SMI"],
                          two_group_colors, "Low-SMI 임상 cutoff에 따른 AEC 곡선 비교 (propensity-matched)")
    r_psm = curve_diff_test(df_matched, "LowSMI", ["Low SMI", "Non-low SMI"], ["Low SMI", "Non-low SMI"])
    r_psm["n_matched_pairs"] = len(matched_idx) // 2
    summary_rows.append({"figure": "07b_aec_curve_by_low_smi_matched.png",
                          "comparison": "Low-SMI cutoff (propensity-matched)", **r_psm})
    ax.text(0.02, 0.02, f"{curve_diff_note(r_psm)}\n(matched pairs={r_psm['n_matched_pairs']})",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout()
    savefig(fig, out_dir, "07b_aec_curve_by_low_smi_matched.png")

    # 07c: Low-SMI(비매칭), 전체 곡선 permutation test의 peak_slice 주변 구간 확대 비교
    lowsmi_order = ["Low SMI", "Non-low SMI"]
    lowsmi_labels = ["Low SMI", "Non-low SMI"]
    win_cols, win_lo, win_hi = peak_window_cols(r_lowsmi_full["peak_slice"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_curve_comparison(ax, df, "LowSMI", lowsmi_order, lowsmi_labels, two_group_colors,
                          f"Low-SMI AEC 곡선 비교 - peak 구간 확대 (slice {win_lo}-{win_hi})",
                          cols=win_cols)
    r_win = curve_diff_test(df, "LowSMI", lowsmi_order, lowsmi_labels, cols=win_cols)
    summary_rows.append({"figure": "07c_aec_curve_by_low_smi_peak_window.png",
                          "comparison": f"Low-SMI cutoff, peak 구간 (slice {win_lo}-{win_hi})", **r_win})
    ax.text(0.02, 0.02,
            f"{curve_diff_note(r_win)}\n(전체 곡선 peak={r_lowsmi_full['peak_slice']} 기준 ±{PEAK_WINDOW_HALFWIDTH}슬라이스)",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout()
    savefig(fig, out_dir, "07c_aec_curve_by_low_smi_peak_window.png")

    # 07d: Low-SMI(propensity-matched), peak_slice 주변 구간 확대 비교
    win_cols_m, win_lo_m, win_hi_m = peak_window_cols(r_psm["peak_slice"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_curve_comparison(ax, df_matched, "LowSMI", lowsmi_order, lowsmi_labels, two_group_colors,
                          f"Low-SMI AEC 곡선 비교(propensity-matched) - peak 구간 확대 (slice {win_lo_m}-{win_hi_m})",
                          cols=win_cols_m)
    r_win_m = curve_diff_test(df_matched, "LowSMI", lowsmi_order, lowsmi_labels, cols=win_cols_m)
    r_win_m["n_matched_pairs"] = len(matched_idx) // 2
    summary_rows.append({"figure": "07d_aec_curve_by_low_smi_matched_peak_window.png",
                          "comparison": f"Low-SMI cutoff (propensity-matched), peak 구간 (slice {win_lo_m}-{win_hi_m})",
                          **r_win_m})
    ax.text(0.02, 0.02,
            f"{curve_diff_note(r_win_m)}\n(matched pairs={r_win_m['n_matched_pairs']}; "
            f"matched 전체곡선 peak={r_psm['peak_slice']} 기준 ±{PEAK_WINDOW_HALFWIDTH}슬라이스)",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout()
    savefig(fig, out_dir, "07d_aec_curve_by_low_smi_matched_peak_window.png")

    # 통합 3x3 패널 (2그룹 비교만)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for ax, (group_col, order, labels, _, title_kr) in zip(axes.flat, specs):
        plot_curve_comparison(ax, df, group_col, order, labels, two_group_colors, title_kr)
    for ax in axes.flat[len(specs):]:
        ax.axis("off")
    fig.suptitle("변수별 AEC point curve 비교 (환자 정규화, mean ± 95% CI)",
                 fontsize=14, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, out_dir, "10_aec_curve_combined_panel.png")

    # 3그룹 이상 비교 (제조사, BMI 4구간, 성별x Low-SMI 조합)
    multi_specs = []

    vendor_df = df.dropna(subset=["Vendor"]).copy()
    vendor_counts = vendor_df["Vendor"].value_counts()
    vendor_order = vendor_counts[vendor_counts >= 30].index.tolist()
    vendor_df = vendor_df[vendor_df["Vendor"].isin(vendor_order)]
    if len(vendor_order) >= 2:
        multi_specs.append((vendor_df, "Vendor", vendor_order, vendor_order,
                             "11_aec_curve_by_vendor.png", "스캐너 제조사(Vendor)에 따른 AEC 곡선 비교"))
    else:
        print(f"skip vendor comparison: only {len(vendor_order)} vendor group(s) with n>=30 ({vendor_order})")

    bmi4_order = ["Underweight", "Normal", "Overweight", "Obese"]
    bmi4_df = df.dropna(subset=["BMIGroup4"]).copy()
    bmi4_df["BMIGroup4"] = bmi4_df["BMIGroup4"].astype(str)
    multi_specs.append((bmi4_df, "BMIGroup4", bmi4_order, bmi4_order,
                         "12_aec_curve_by_bmi4.png", "BMI 4구간(WHO 아시아 기준)에 따른 AEC 곡선 비교"))

    sexsmi_order = ["M / Low SMI", "M / Non-low SMI", "F / Low SMI", "F / Non-low SMI"]
    multi_specs.append((df, "SexSMIGroup", sexsmi_order, sexsmi_order,
                         "13_aec_curve_by_sex_x_lowsmi.png", "성별 x Low-SMI 조합에 따른 AEC 곡선 비교"))

    multi_colors = [COL_A, COL_B, COL_C, COL_D]
    for gdf, group_col, order, labels, fname, title_kr in multi_specs:
        colors = multi_colors[: len(order)]
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_curve_comparison(ax, gdf, group_col, order, labels, colors, title_kr)
        r = curve_diff_test(gdf, group_col, order, labels)
        summary_rows.append({"figure": fname, "comparison": title_kr, **r})
        ax.text(0.02, 0.02, curve_diff_note(r),
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        fig.tight_layout()
        savefig(fig, out_dir, fname)

    # BMI 4구간 x Low-SMI 교차 패널: BMI 효과와 SMI 효과를 동시에 분리해서 확인
    bmi4_df["LowSMI"] = bmi4_df["LowSMI"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
    smi_order = ["Low SMI", "Non-low SMI"]
    smi_labels = ["Low SMI", "Non-low SMI"]
    for ax, bmi_group in zip(axes.flat, bmi4_order):
        sub = bmi4_df[bmi4_df["BMIGroup4"] == bmi_group]
        plot_curve_comparison(ax, sub, "LowSMI", smi_order, smi_labels, two_group_colors,
                              f"BMI: {bmi_group} (n={len(sub)})")
        if sub["LowSMI"].nunique() == 2:
            r = curve_diff_test(sub, "LowSMI", smi_order, smi_labels)
            summary_rows.append({
                "figure": "14_aec_curve_bmi4_x_lowsmi_facet.png",
                "comparison": f"BMI:{bmi_group} 내 Low-SMI vs Non-low SMI", **r,
            })
            ax.text(0.02, 0.02, curve_diff_note(r), transform=ax.transAxes, fontsize=8,
                    color=INK_MUTED, va="bottom")
    fig.suptitle("BMI 4구간 내에서 Low-SMI 효과 분리 (BMI x SMI 교차비교)",
                 fontsize=14, color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, out_dir, "14_aec_curve_bmi4_x_lowsmi_facet.png")

    # SMI를 통제한 뒤(Non-low SMI만) BMI 4구간 효과가 남는지 재비교
    controlled_df = bmi4_df[bmi4_df["LowSMI"] == "Non-low SMI"]
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_curve_comparison(ax, controlled_df, "BMIGroup4", bmi4_order, bmi4_order,
                          multi_colors[: len(bmi4_order)],
                          "SMI 통제(Non-low SMI만) 후 BMI 4구간에 따른 AEC 곡선 비교")
    r = curve_diff_test(controlled_df, "BMIGroup4", bmi4_order, bmi4_order)
    summary_rows.append({
        "figure": "15_aec_curve_bmi4_smi_controlled.png",
        "comparison": "SMI 통제 후 BMI 4구간", **r,
    })
    ax.text(0.02, 0.02, f"{curve_diff_note(r)}\n(Low-SMI 환자 제외, n={len(controlled_df)})",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
    fig.tight_layout()
    savefig(fig, out_dir, "15_aec_curve_bmi4_smi_controlled.png")

    summary_df = pd.DataFrame(summary_rows)
    summary_df["significant_p<0.05"] = summary_df["p_value"] < 0.05
    summary_path = os.path.join(out_dir, "00_group_diff_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"saved: {summary_path}")

    print(f"All figures saved under: {out_dir}")


def main():
    for cohort in COHORTS:
        run_cohort(cohort)


if __name__ == "__main__":
    main()
