from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 질환 유무(Yes vs No)에 따라 landmark 단면 값(VAT/SAT/TAMA/AEC)이 얼마나 다른지: Cohen's d + Mann-Whitney p(BH-FDR), All/M/F 층화, 두 코호트 방향 일치 여부
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from clinic4_landmark_vat_auc import LM, TISSUES, load_cohort
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
OUT = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
MEASURES = TISSUES + ["AEC"]
LMN = {a: a.replace("_center", "") for a in LM}


# 환자 landmark slice의 AEC 값을 AEC@landmark 컬럼으로 추가
def add_aec(df, cohort):
    path = f"data/{cohort}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    aec = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$")
    for a in LM:
        df[f"AEC@{a}"] = [aec.loc[p].to_numpy(float)[int(lm.loc[p, f"{a}_slice"]) - 1] for p in df.PatientID]
    return df


rows = []
for c in ["gangnam", "sinchon"]:
    df = add_aec(load_cohort(c)[0], c)
    for sex in ["All", "M", "F"]:
        base = df if sex == "All" else df[df.PatientSex == sex]
        for d in DIS:
            y, n = base[base[d] == 1], base[base[d] == 0]
            for m in MEASURES:
                for a in LM:
                    if (m, a) == ("VAT", "inferior_pubic_margin"):
                        continue  # 치골 하단 VAT는 항상 0
                    u, v = y[f"{m}@{a}"], n[f"{m}@{a}"]
                    sp = np.sqrt(((len(u) - 1) * u.var() + (len(v) - 1) * v.var()) / (len(u) + len(v) - 2))
                    rows.append(dict(cohort=c, sex=sex, disease=d, measure=m, landmark=LMN[a], n_yes=len(u), n_no=len(v),
                                     mean_yes=u.mean(), mean_no=v.mean(), cohen_d=(u.mean() - v.mean()) / sp,
                                     p=mannwhitneyu(u, v).pvalue))
r = pd.DataFrame(rows)
r["p_fdr"] = r.groupby(["cohort", "sex", "disease"])["p"].transform(lambda p: bh_fdr(p.to_numpy()))

# 두 코호트 비교: 같은 (sex, disease, measure, landmark)에서 d 방향 일치 + 둘 다 FDR<0.05
w = r.pivot_table(index=["sex", "disease", "measure", "landmark"], columns="cohort", values=["cohen_d", "p_fdr"]).reset_index()
w.columns = ["_".join(x).strip("_") for x in w.columns]
w["consistent"] = (np.sign(w.cohen_d_gangnam) == np.sign(w.cohen_d_sinchon)) & (w.p_fdr_gangnam < .05) & (w.p_fdr_sinchon < .05)
w["min_abs_d"] = w[["cohen_d_gangnam", "cohen_d_sinchon"]].abs().min(axis=1)
w = w.sort_values(["sex", "disease", "min_abs_d"], ascending=[True, True, False])

with pd.ExcelWriter(f"{OUT}/landmark_disease_difference.xlsx") as xw:
    w[w.consistent].sort_values("min_abs_d", ascending=False).round(4).to_excel(xw, sheet_name="robust_top", index=False)
    w.round(4).to_excel(xw, sheet_name="both_cohorts", index=False)
    r.round(4).to_excel(xw, sheet_name="detail", index=False)

# 히트맵: 행=측정값@landmark, 열=성별|질환, 값=Cohen d (* = FDR<0.05), 코호트별 1장
for c in ["gangnam", "sinchon"]:
    s = r[r.cohort == c].assign(row=lambda x: x.measure + "@" + x.landmark, col=lambda x: x.sex + "|" + x.disease)
    cols = [f"{sx}|{d}" for sx in ["All", "M", "F"] for d in DIS]
    rowsl = [f"{m}@{LMN[a]}" for m in MEASURES for a in LM if (m, a) != ("VAT", "inferior_pubic_margin")]
    D = s.pivot(index="row", columns="col", values="cohen_d").loc[rowsl, cols]
    F = s.pivot(index="row", columns="col", values="p_fdr").loc[rowsl, cols]
    fig, ax = plt.subplots(figsize=(8, 16))
    im = ax.imshow(D, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D.iat[i, j]:.2f}" + ("*" if F.iat[i, j] < .05 else ""), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60); ax.set_yticks(range(len(rowsl))); ax.set_yticklabels(rowsl, fontsize=7)
    ax.set_title(f"{c}: Cohen d (Yes-No), * FDR<0.05"); plt.colorbar(im, shrink=.4); plt.tight_layout()
    plt.savefig(f"{OUT}/{c}_disease_difference_heatmap.png", dpi=130); plt.close()

top = w[w.consistent & (w.sex == "All")].sort_values("min_abs_d", ascending=False)
print(top.groupby("disease").head(6)[["disease", "measure", "landmark", "cohen_d_gangnam", "cohen_d_sinchon"]].round(2).to_string())
print(w[w.consistent].groupby(["sex", "disease"]).size())
