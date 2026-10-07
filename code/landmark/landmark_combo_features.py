from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 곡선 모양이 아니라 "landmark별 지표의 조합"으로 질환과 연관된 feature가 나오는지 확인.
#  Tier 1 (문헌 근거 있음): 같은 landmark 안의 조직 간 비율 VAT/SAT, VAT/TAMA, TAMA/(VAT+SAT)
#     근거: VAT/SAT ratio (Acad Radiol 2026 doi:10.1016/j.acra.2026.09.035, Minerva Endocrinol 2025 doi:10.23736/S2724-6507.25.04366-0),
#           내장지방/근육 비 at L3 (Surg Today 2026 doi:10.1007/s00595-026-03487-7, 암 환자 예후)
#  Tier 2 (직접 근거 못 찾음, 탐색적): 서로 다른 landmark x 조직(AEC 포함) 로그 비 A@Li / B@Lj (A!=B)
#  (a) 단일 feature: +1SD당 OR(나이·성별·BMI 보정), BH-FDR은 Tier x 코호트 x 질환 안, 두 코호트 FDR<0.05 & 같은 방향이면 robust
#  (b) 조합 세트 모델: clinical(나이·성별·BMI) 위에 세트를 더한 로지스틱, Gangnam 5-fold CV AUC / Gangnam 학습 -> Sinchon 외부 AUC, DeLong p vs clinical
# 출력: outputs/data_distribution/landmark_combo_features.xlsx
import itertools
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import LM, load_cohort
from delong_utils import bh_fdr, delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
TIS = ["VAT", "SAT", "TAMA", "AEC"]
LMN = [a.replace("_center", "") for a in LM]
lr = lambda a, b: np.log((a + 1) / (b + 1))  # 기존 landmark 스크립트와 같은 로그 비율(0 방지)


# 코호트 로드: QC 통과 환자의 임상 변수 + 조직@landmark 단면 값(AEC 포함)
def load(c):
    df = load_cohort(c)[0].copy()
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[df.PatientID]
    aec = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$").loc[df.PatientID].to_numpy(float)
    for a, n in zip(LM, LMN):
        df[f"AEC@{n}"] = np.nan_to_num(aec[np.arange(len(df)), lm[f"{a}_slice"].to_numpy(int) - 1])
        for t in TIS[:3]:
            df[f"{t}@{n}"] = df[f"{t}@{a}"]
    return df


def tier1(df):  # 같은 landmark 안의 조직 비율
    f = {}
    for n in LMN:
        v, s, t = df[f"VAT@{n}"], df[f"SAT@{n}"], df[f"TAMA@{n}"]
        if n != "inferior_pubic_margin":
            f[f"VAT/SAT@{n}"] = lr(v, s)
            f[f"VAT/TAMA@{n}"] = lr(v, t)
            f[f"TAMA/(VAT+SAT)@{n}"] = lr(t, v + s)
    return pd.DataFrame(f)


def tier2(df):  # 서로 다른 landmark/조직의 조합 (같은 landmark 같은 쌍은 Tier 1과 중복이라 i!=j만)
    f = {}
    for (A, B) in itertools.permutations(TIS, 2):
        for i, j in itertools.product(range(len(LMN)), repeat=2):
            if i == j or "VAT@inferior_pubic_margin" in (f"{A}@{LMN[i]}", f"{B}@{LMN[j]}"):
                continue
            f[f"{A}@{LMN[i]}/{B}@{LMN[j]}"] = lr(df[f"{A}@{LMN[i]}"], df[f"{B}@{LMN[j]}"])
    return pd.DataFrame(f)


def fit_or(x, y, cov):
    if not np.isfinite(x).all() or x.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(np.column_stack([(x - x.mean()) / x.std(), cov]))
    try:
        f = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    ci = np.exp(f.conf_int()[1])
    return np.exp(f.params[1]), ci[0], ci[1], f.pvalues[1]


data = {c: load(c) for c in ["gangnam", "sinchon"]}
rows = []
for c, df in data.items():
    cov = np.column_stack([df.PatientAge, (df.PatientSex == "M").astype(float), df.BMI]).astype(float)
    for tier, F in (("tier1", tier1(df)), ("tier2", tier2(df))):
        for d in DIS:
            y = df[d].to_numpy(int)
            for name in F.columns:
                o, l, h, p = fit_or(F[name].to_numpy(float), y, cov)
                rows.append(dict(cohort=c, tier=tier, disease=d, feature=name, OR=o, ci_low=l, ci_high=h, p=p))
    print(c, "done", flush=True)
r = pd.DataFrame(rows)
r["p_fdr"] = r.groupby(["cohort", "tier", "disease"])["p"].transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
w = r.pivot_table(index=["tier", "disease", "feature"], columns="cohort", values=["OR", "ci_low", "ci_high", "p_fdr"]).reset_index()
w.columns = ["_".join(x).strip("_") for x in w.columns]
w["robust"] = (np.log(w.OR_gangnam) * np.log(w.OR_sinchon) > 0) & (w.p_fdr_gangnam < .05) & (w.p_fdr_sinchon < .05)
w["min_abs_logOR"] = np.minimum(np.log(w.OR_gangnam).abs(), np.log(w.OR_sinchon).abs())
w = w.sort_values(["tier", "disease", "min_abs_logOR"], ascending=[True, True, False])


# (b) 조합 세트 모델: Gangnam 5-fold CV AUC + Gangnam 학습 -> Sinchon 외부 AUC
def sets(df):
    t1 = tier1(df)
    base = lambda cols: df[cols]
    S = {
        "clinical": [],
        "VAT@L3": ["VAT@L3"],
        "VAT,SAT,TAMA@L3": [f"{t}@L3" for t in TIS[:3]],
        "VAT,SAT,TAMA@L1,L3,L5": [f"{t}@{a}" for t in TIS[:3] for a in ("L1", "L3", "L5")],
        "VAT,SAT,TAMA@all landmarks": [f"{t}@{a}" for t in TIS[:3] for a in LMN if (t, a) != ("VAT", "inferior_pubic_margin")],
        "Tier1 ratios@L3": ["VAT/SAT@L3", "VAT/TAMA@L3", "TAMA/(VAT+SAT)@L3"],
        "Tier1 ratios@all landmarks": list(t1.columns),
        "VAT,SAT,TAMA,AEC@L3 (AEC 탐색적)": [f"{t}@L3" for t in TIS],
    }
    out = {}
    for k, cols in S.items():
        parts = [df[["PatientAge", "BMI"]].assign(male=(df.PatientSex == "M").astype(float))]
        raw = df[[c_ for c_ in cols if c_ in df.columns]]
        extra = t1[[c_ for c_ in cols if c_ in t1.columns]]
        out[k] = np.column_stack([parts[0].to_numpy(float), raw.to_numpy(float), extra.to_numpy(float)])
    return out


def model(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


Sg, Ss = sets(data["gangnam"]), sets(data["sinchon"])
srows = []
for d in DIS:
    yg, ys = data["gangnam"][d].to_numpy(int), data["sinchon"][d].to_numpy(int)
    ext = {k: model(Sg[k], yg, Ss[k]) for k in Sg}
    for k in Sg:
        cv = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Sg[k], yg):
            cv.append(roc_auc_score(yg[te], model(Sg[k][tr], yg[tr], Sg[k][te])))
        dl = delong_paired_auc_test(ys, ext["clinical"], ext[k]) if k != "clinical" else {"diff": 0.0, "p_value": np.nan}
        srows.append(dict(disease=d, set=k, n_feat=Sg[k].shape[1], cv_auc_gangnam=np.mean(cv), ext_auc_sinchon=roc_auc_score(ys, ext[k]),
                          ext_delta=dl["diff"], delong_p_vs_clinical=dl["p_value"]))
s = pd.DataFrame(srows)

with pd.ExcelWriter(f"{D}/landmark_combo_features.xlsx") as xw:
    w[w.robust].round(4).to_excel(xw, sheet_name="robust", index=False)
    w.round(4).to_excel(xw, sheet_name="all", index=False)
    s.round(4).to_excel(xw, sheet_name="set_models", index=False)
pd.set_option("display.width", 250)
print(w.groupby(["tier", "disease"]).agg(n=("robust", "size"), robust=("robust", "sum")))
print(w[(w.tier == "tier1") & w.robust].groupby("disease").head(6)[["disease", "feature", "OR_gangnam", "OR_sinchon"]].round(2).to_string(index=False))
print(w[(w.tier == "tier2") & w.robust].groupby("disease").head(5)[["disease", "feature", "OR_gangnam", "OR_sinchon"]].round(2).to_string(index=False))
print(s.round(3).to_string(index=False))
