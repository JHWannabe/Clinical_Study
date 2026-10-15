# landmark별 질환군 vs 정상군 raw 값 분포 선그래프(median + IQR 음영, Mann-Whitney U BH-FDR 표시): 행=질환, 열=VAT/SAT/TAMA/AEC
# 질환군 = 해당 질환 양성(동반질환 포함), 정상군 = HTN·DM·CKD 모두 없는 환자. OUT_DIR로 저장 폴더 지정
import os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / os.environ.get("OUT_DIR", "outputs/261012/0_data_distribution"); OUT.mkdir(parents=True, exist_ok=True)
DIS, MEAS = ["HTN", "DM", "CKD"], ["VAT", "SAT", "TAMA", "AEC"]
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center",
      "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
LBL = [a.replace("_center", "") for a in LM]
QC_ORDER = LM[1:]  # T10~pubis slice가 엄격히 증가하지 않는 환자는 제외 (liver_dome은 T10보다 뒤일 수 있어 제외)
star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# 코호트의 환자 메타 + landmark 위치의 VAT/SAT/TAMA/AEC 값을 한 표로 만든다
def load_cohort(cohort):
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    meta = meta[meta.PatientAge >= 20]  # 20세 미만 제외
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    key = lambda m: m.lower() if m == "AEC" else m
    curves = {m: pd.read_excel(path, sheet_name=f"{key(m)}_total").set_index("PatientID").filter(regex=rf"^{key(m)}_\d+$") for m in MEAS}
    meta = meta[meta.PatientID.isin(set.intersection(*[set(c.index) for c in curves.values()]))]
    meta = meta.dropna(subset=["PatientAge", "Height", "Weight"] + DIS).reset_index(drop=True)
    order = lm.loc[meta.PatientID, [f"{a}_slice" for a in QC_ORDER]].to_numpy()
    meta = meta[~(np.diff(order, axis=1) <= 0).any(axis=1)].reset_index(drop=True)
    feat = {f"{m}@{a}": [np.nan_to_num(curves[m].loc[p].to_numpy(float)[int(lm.loc[p, f"{a}_slice"]) - 1]) for p in meta.PatientID]
            for m in MEAS for a in LM}
    return pd.concat([meta, pd.DataFrame(feat)], axis=1)


# 질환 d의 비교 대상 환자: 질환군(동반질환 포함) + 정상군(세 질환 모두 없음)
def subset(df, d):
    return df[(df[d] == 1) | (df[DIS].sum(axis=1) == 0)]


# landmark 12곳마다 질환군 vs 정상군 Mann-Whitney U (BH-FDR 보정): (p, q, 질환군 median - 정상군 median)
def compare(pos, neg, cols):
    p = np.array([mannwhitneyu(pos[k], neg[k]).pvalue if pos[k].nunique() + neg[k].nunique() > 2 else 1. for k in cols])
    return p, false_discovery_control(p), (pos[cols].median() - neg[cols].median()).to_numpy()


# 코호트 하나의 분포 그림을 그리고 유의성 결과 행을 돌려준다
def plot_cohort(cohort, df):
    x = np.arange(len(LM)); rows = []
    fig, axs = plt.subplots(3, 4, figsize=(22, 12), sharex="col")
    for i, d in enumerate(DIS):
        s = subset(df, d)
        for j, m in enumerate(MEAS):
            ax, cols = axs[i, j], [f"{m}@{a}" for a in LM]
            for val, color, name in [(0, "#1f77b4", "정상군"), (1, "#d62728", "질환군")]:
                g = s[s[d] == val][cols]
                ax.plot(x, g.median(), "o-", color=color, ms=4, label=f"{name} (n={len(g)})")
                ax.fill_between(x, g.quantile(.25), g.quantile(.75), color=color, alpha=.2)
            p, q, diff = compare(s[s[d] == 1], s[s[d] == 0], cols)
            for xi, qi in zip(x, q):
                ax.annotate(star(qi), (xi, 1.01), xycoords=("data", "axes fraction"), ha="center", fontsize=7 if qi < .05 else 6, color="k" if qi < .05 else "gray")
            rows += [(cohort, d, m, a, *r) for a, r in zip(LM, zip(p, q, diff))]
            ax.legend(fontsize=7, frameon=False); ax.set_xticks(x); ax.set_xticklabels(LBL, rotation=60, ha="right")
            if i == 0: ax.set_title(m, fontsize=12, pad=18)
            if j == 0: ax.set_ylabel(d, fontsize=12)
    fig.suptitle(f"{cohort}: landmark별 질환군 vs 정상군(무질환) raw 값 분포 (선=median, 음영=IQR 25~75%)\n위 표시: Mann-Whitney U BH-FDR (*<0.05, **<0.01, ***<0.001, n.s.=비유의; 질환·측정별 landmark 12개 보정)", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / f"data_distribution_{cohort}.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    return rows


def main():
    rows = []
    for c in ["gangnam", "sinchon"]:
        df = load_cohort(c)
        print(f"[{c}] {len(df)}명 ({', '.join(f'{d}={int(df[d].sum())}' for d in DIS)})")
        rows += plot_cohort(c, df)
    res = pd.DataFrame(rows, columns=["cohort", "disease", "measure", "landmark", "p", "q_fdr", "median_diff"])
    res.to_excel(OUT / "data_distribution_significance.xlsx", sheet_name="mannwhitney", index=False)
    print(res[res.q_fdr < .05].groupby(["cohort", "disease", "measure"]).size().to_string())


if __name__ == "__main__":
    main()
