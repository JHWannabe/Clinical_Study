from __future__ import annotations

# clinic4 및 clinic4+구간 합산(VAT/SAT/TAMA/AEC) 로지스틱 회귀 계수: 코호트별·풀링, 연속형은 1 SD 단위 OR(95% CI, p) (단독 실행, outputs/261012 저장)
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/261012"
DIS, MEAS = ["HTN", "DM", "CKD"], ["VAT", "SAT", "TAMA", "AEC"]
import os
ALL_DIS = ["HTN", "DM", "CKD"]; SINGLE = os.environ.get("SINGLE_DIS")  # 설정하면 정상군(세 질환 모두 없음) vs 해당 질환 단독 양성만 사용 (결과는 OUT_DIR에 저장)
if SINGLE: DIS = [SINGLE]
if os.environ.get("OUT_DIR"): OUT = ROOT / os.environ["OUT_DIR"]
OUT.mkdir(parents=True, exist_ok=True)


# 정상군(세 질환 모두 없음) + 해당 질환 단독 양성 환자만 남김
def keep_rows(m):
    if not SINGLE: return m
    oth = [x for x in ALL_DIS if x != SINGLE]
    if os.environ.get("DEF_MODE") == "any": return m[(m[SINGLE] == 1) | (m[ALL_DIS].sum(axis=1) == 0)].reset_index(drop=True)  # 동반질환자도 질환군에 포함
    return m[((m[SINGLE] == 1) & (m[oth].sum(axis=1) == 0)) | (m[ALL_DIS].sum(axis=1) == 0)].reset_index(drop=True)

LM0, LM1 = "liver_dome", "inferior_pubic_margin"
SHEET = {m: f"{m.lower()}_total" if m == "AEC" else f"{m}_total" for m in MEAS}


# 코호트의 환자 메타 + liver dome~pubis 구간 합산({m}_sum)
def load(cohort: str) -> pd.DataFrame:
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    cur = {m: pd.read_excel(path, sheet_name=SHEET[m]).set_index("PatientID").filter(regex=rf"^{m.lower() if m == 'AEC' else m}_\d+$") for m in MEAS}
    meta = meta[meta.PatientID.isin(set.intersection(*[set(c.index) for c in cur.values()]) & set(lm.index))]
    meta = keep_rows(meta.dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + ALL_DIS).reset_index(drop=True))
    s0, s1 = lm.loc[meta.PatientID, f"{LM0}_slice"].to_numpy(int), lm.loc[meta.PatientID, f"{LM1}_slice"].to_numpy(int)
    for m in MEAS:
        v = np.nan_to_num(cur[m].loc[meta.PatientID].to_numpy(float)); meta[f"{m}_sum"] = [v[i, s0[i] - 1:s1[i]].sum() for i in range(len(v))]
    meta["sex(M=1)"] = (meta.PatientSex.astype(str).str.upper() == "M").astype(float); meta["cohort"] = cohort
    return meta.rename(columns={"PatientAge": "age", "Height": "height", "Weight": "weight"})


D = {c: load(c) for c in ["gangnam", "sinchon"]}; D["pooled"] = pd.concat(D.values(), ignore_index=True)
BASE = ["sex(M=1)", "age", "height", "weight"]; SUMS = [f"{m}_sum" for m in MEAS]
rows = []
for c, df in D.items():
    for name, cols in [("clinic4", BASE), ("clinic4+구간 합산", BASE + SUMS)]:
        X = df[cols].copy(); X[[k for k in cols if k != "sex(M=1)"]] = (X[[k for k in cols if k != "sex(M=1)"]] - X[[k for k in cols if k != "sex(M=1)"]].mean()) / X[[k for k in cols if k != "sex(M=1)"]].std()
        Xc = sm.add_constant(X); vf = {k: vif(Xc.to_numpy(float), i) for i, k in enumerate(Xc.columns)}
        for d in DIS:
            try: m = sm.Logit(df[d].to_numpy(int), Xc).fit(disp=0, maxiter=200); ci = m.conf_int()
            except Exception: continue  # 양성이 너무 적어 수렴하지 않는 경우 건너뜀
            for k in cols: rows.append(dict(cohort=c, model=name, disease=d, feature=k, coef=m.params[k], OR=np.exp(m.params[k]), ci_low=np.exp(ci.loc[k, 0]), ci_high=np.exp(ci.loc[k, 1]), p=m.pvalues[k], VIF=vf[k]))
res = pd.DataFrame(rows); res.to_excel(OUT / "clinic4_coefficients.xlsx", index=False)
for (c, name), g in res.groupby(["cohort", "model"], sort=False):
    print(f"\n[{c} | {name}] OR per 1 SD (95% CI) [p]; sex는 남성 vs 여성")
    print(g.assign(s=lambda x: x.OR.map("{:.2f}".format) + " (" + x.ci_low.map("{:.2f}".format) + "-" + x.ci_high.map("{:.2f}".format) + ") [" + x.p.map("{:.3f}".format) + "]").pivot(index="feature", columns="disease", values="s")[DIS].loc[BASE + (SUMS if "합산" in name else [])].to_string())
print("\nVIF(pooled, clinic4+구간 합산):", res[(res.cohort == "pooled") & (res.model == "clinic4+구간 합산") & (res.disease == "HTN")].set_index("feature").VIF.round(1).to_dict())
