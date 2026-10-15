from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark별 질환군 vs 정상군 평균(±95% CI) 선그래프: 행=질환, 열=VAT/SAT/TAMA/AEC, 위 표시=Welch t-test BH-FDR (현재 데이터 기준, outputs/clinic4/261012 저장)
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from clinic4_landmark_vat_auc import LM, TISSUES, load_cohort
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
OUT = "outputs/clinic4/261012"
DIS, MEAS = ["HTN", "DM", "CKD"], TISSUES + ["AEC"]
LBL = [a.replace("_center", "") for a in LM]
star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

for c in ["gangnam", "sinchon"]:
    df = load_cohort(c)[0]
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    aec = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$")
    for a in LM:
        df[f"AEC@{a}"] = [aec.loc[p].to_numpy(float)[int(lm.loc[p, f"{a}_slice"]) - 1] for p in df.PatientID]
    fig, axs = plt.subplots(3, 4, figsize=(20, 11), sharex="col")
    for i, d in enumerate(DIS):
        y, n = df[df[d] == 1], df[df[d] == 0]
        for j, m in enumerate(MEAS):
            ax = axs[i, j]; x = np.arange(len(LM)); cols = [f"{m}@{a}" for a in LM]
            for g, col, nm in [(n, "#1f77b4", "정상군"), (y, "#d62728", "질환군")]:
                mu, ci = g[cols].mean().to_numpy(), 1.96 * g[cols].sem().to_numpy()
                ax.plot(x, mu, "o-", color=col, ms=4, label=f"{nm} (n={len(g)})"); ax.fill_between(x, mu - ci, mu + ci, color=col, alpha=.2)
            p = np.array([ttest_ind(y[k], n[k], equal_var=False).pvalue if y[k].std() + n[k].std() > 0 else 1. for k in cols])
            q = bh_fdr(p)
            for xi, qi in zip(x, q):
                ax.annotate(star(qi), (xi, 1.01), xycoords=("data", "axes fraction"), ha="center", fontsize=7 if qi < .05 else 6, color="k" if qi < .05 else "gray")
            ax.legend(fontsize=7, frameon=False); ax.set_xticks(x); ax.set_xticklabels(LBL, rotation=60, ha="right")
            if i == 0: ax.set_title(m, pad=18, fontsize=12)
            if j == 0: ax.set_ylabel(d, fontsize=12)
    fig.suptitle(f"{c}: landmark별 질환군 vs 정상군 평균 (±95% CI, raw 값)\n위 표시: Welch t-test BH-FDR (*<0.05, **<0.01, ***<0.001, n.s.=비유의; 질환·조직별 landmark 12개 보정)", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/landmark_group_means_{c}.png", dpi=135); plt.close()
