from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark 단면 값(VAT/SAT/TAMA/AEC, +1SD)당 질환 OR: 나이·성별·BMI 보정 로지스틱(All) + 성별 층화(나이·BMI 보정), BH-FDR은 코호트x층x질환 안
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from clinic4_landmark_vat_auc import LM, TISSUES, load_cohort
from delong_utils import bh_fdr

sys.stdout.reconfigure(encoding="utf-8")
OUT = "outputs/data_distribution/landmark_adjusted_or.xlsx"
DIS = ["HTN", "DM", "CKD"]
MEASURES = TISSUES + ["AEC"]


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
    df["male"] = (df.PatientSex == "M").astype(float)
    for sex, covs in [("All", ["PatientAge", "male", "BMI"]), ("M", ["PatientAge", "BMI"]), ("F", ["PatientAge", "BMI"])]:
        g = df if sex == "All" else df[df.PatientSex == sex]
        for m in MEASURES:
            for a in LM:
                if (m, a) == ("VAT", "inferior_pubic_margin"):
                    continue  # 치골 하단 VAT는 항상 0
                x = g[[f"{m}@{a}"]].apply(lambda s: (s - s.mean()) / s.std())
                X = sm.add_constant(pd.concat([x, g[covs]], axis=1).to_numpy(float))
                for d in DIS:
                    fit = sm.Logit(g[d].to_numpy(int), X).fit(disp=0, maxiter=200)
                    ci = np.exp(fit.conf_int()[1])
                    rows.append(dict(cohort=c, sex=sex, disease=d, measure=m, landmark=a.replace("_center", ""), n=len(g),
                                     OR_per_SD=np.exp(fit.params[1]), ci_low=ci[0], ci_high=ci[1], p=fit.pvalues[1]))
r = pd.DataFrame(rows)
r["p_fdr"] = r.groupby(["cohort", "sex", "disease"])["p"].transform(lambda p: bh_fdr(p.to_numpy()))
w = r.pivot_table(index=["sex", "disease", "measure", "landmark"], columns="cohort", values=["OR_per_SD", "ci_low", "ci_high", "p_fdr"]).reset_index()
w.columns = ["_".join(x).strip("_") for x in w.columns]
w["robust"] = ((np.log(w.OR_per_SD_gangnam) * np.log(w.OR_per_SD_sinchon)) > 0) & (w.p_fdr_gangnam < .05) & (w.p_fdr_sinchon < .05)
w["min_abs_logOR"] = np.minimum(np.log(w.OR_per_SD_gangnam).abs(), np.log(w.OR_per_SD_sinchon).abs())
with pd.ExcelWriter(OUT) as xw:
    w.sort_values(["sex", "disease", "min_abs_logOR"], ascending=[True, True, False]).round(4).to_excel(xw, sheet_name="both_cohorts", index=False)
    r.round(4).to_excel(xw, sheet_name="detail", index=False)
a = w[(w.sex == "All")]
print(a[a.robust].groupby("disease").size())
print(a.sort_values("min_abs_logOR", ascending=False).groupby("disease").head(4)[["disease", "measure", "landmark", "OR_per_SD_gangnam", "OR_per_SD_sinchon", "robust"]].round(2).to_string())
print(w[w.robust].groupby(["sex", "measure"]).size().unstack(fill_value=0))
