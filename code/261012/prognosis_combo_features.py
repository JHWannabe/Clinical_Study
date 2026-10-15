from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark_combo_features.py(단면)와 같은 case를 예후(5년 내 발병) 데이터에 적용: Tier1 조직 비율, Tier2 landmark x 조직 교차 비,
#  (a) 단일 feature: +1SD당 OR(나이·성별·BMI 보정), BH-FDR은 Tier x 질환 안, FDR<0.05면 robust (코호트 1개라 방향 일치 검증 없음)
#  (b) 조합 세트 모델: clinical(나이·성별·BMI) 위에 세트를 더한 로지스틱, 반복 5-fold CV AUC + DeLong p vs clinical
# 출력: outputs/261012/prognosis_combo_features.xlsx
import itertools, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from delong_utils import bh_fdr, delong_paired_auc_test

warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = _p.Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/261012"
PATH = ROOT / "data/prognosis_5y_landmark_filtered.xlsx"
DIS = {"DM": "dm", "HTN": "htn", "DL": "dl", "OSP": "osteoporosis", "MI": "mi", "STROKE": "stroke"}
TIS = ["VAT", "SAT", "TAMA", "AEC"]
SHEET = {m: "aec_total" if m == "AEC" else f"{m}_total" for m in TIS}
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
LMN = [a.replace("_center", "") for a in LM]
REPS = 10
lr = lambda a, b: np.log((a + 1) / (b + 1))  # 단면 스크립트와 같은 로그 비율(0 방지)

X_ = pd.ExcelFile(PATH)
lm = X_.parse("landmarks").set_index("PatientID")
cur = {m: X_.parse(SHEET[m]).set_index("PatientID").filter(regex=rf"^{m.lower() if m == 'AEC' else m}_\d+$") for m in TIS}


# 환자별 조직@landmark 단면 값(AEC 포함) 표
def tissue(ids):
    sl = lm.loc[ids, [f"{a}_slice" for a in LM]].to_numpy(int)
    f = {}
    for m in TIS:
        v = np.nan_to_num(cur[m].loc[ids].to_numpy(float))
        for n, j in zip(LMN, range(len(LM))): f[f"{m}@{n}"] = v[np.arange(len(v)), np.clip(sl[:, j] - 1, 0, v.shape[1] - 1)]
    return pd.DataFrame(f, index=ids)


def tier1(df):  # 같은 landmark 안의 조직 비율
    f = {}
    for n in LMN[:-1]:  # inferior_pubic_margin 제외
        v, s, t = df[f"VAT@{n}"], df[f"SAT@{n}"], df[f"TAMA@{n}"]
        f[f"VAT/SAT@{n}"], f[f"VAT/TAMA@{n}"], f[f"TAMA/(VAT+SAT)@{n}"] = lr(v, s), lr(v, t), lr(t, v + s)
    return pd.DataFrame(f)


def tier2(df):  # 서로 다른 landmark/조직의 조합
    f = {}
    for A, B in itertools.permutations(TIS, 2):
        for i, j in itertools.product(range(len(LMN)), repeat=2):
            if i != j: f[f"{A}@{LMN[i]}/{B}@{LMN[j]}"] = lr(df[f"{A}@{LMN[i]}"], df[f"{B}@{LMN[j]}"])
    return pd.DataFrame(f)


def fit_or(x, y, cov):
    if not np.isfinite(x).all() or x.std() == 0: return (np.nan,) * 4
    try: f = sm.Logit(y, sm.add_constant(np.column_stack([(x - x.mean()) / x.std(), cov]))).fit(disp=0, maxiter=100)
    except Exception: return (np.nan,) * 4
    ci = np.exp(f.conf_int()[1])
    return np.exp(f.params[1]), ci[0], ci[1], f.pvalues[1]


def sets(df, t1):
    S = {"clinical": [], "VAT@L3": ["VAT@L3"],
         "VAT,SAT,TAMA@L3": [f"{t}@L3" for t in TIS[:3]],
         "VAT,SAT,TAMA@L1,L3,L5": [f"{t}@{a}" for t in TIS[:3] for a in ("L1", "L3", "L5")],
         "VAT,SAT,TAMA@all landmarks": [f"{t}@{a}" for t in TIS[:3] for a in LMN if (t, a) != ("VAT", "inferior_pubic_margin")],
         "Tier1 ratios@L3": ["VAT/SAT@L3", "VAT/TAMA@L3", "TAMA/(VAT+SAT)@L3"],
         "Tier1 ratios@all landmarks": list(t1.columns),
         "VAT,SAT,TAMA,AEC@L3 (AEC 탐색적)": [f"{t}@L3" for t in TIS]}
    base = np.column_stack([df.PatientAge, df.BMI, (df.PatientSex.astype(str).str.upper() == "M").astype(float)]).astype(float)
    return {k: np.column_stack([base, df[[c for c in cols if c in df.columns]].to_numpy(float), t1[[c for c in cols if c in t1.columns]].to_numpy(float)]) for k, cols in S.items()}


def model(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


rows, srows, desc = [], [], []
for dn, sh in DIS.items():
    meta = X_.parse(f"patient_metadata_{sh}")
    meta = meta[(meta.hw_implausible == 0) & meta.PatientID.isin(lm.index) & meta.PatientID.isin(cur["AEC"].index)].dropna(subset=["PatientSex", "PatientAge", "BMI", "onset_within_5y"]).reset_index(drop=True)
    df = pd.concat([meta, tissue(pd.Index(meta.PatientID)).reset_index(drop=True)], axis=1)
    y = df.onset_within_5y.to_numpy(int); desc.append(dict(disease=dn, n=len(y), n_pos=int(y.sum()), n_neg=int((1 - y).sum())))
    cov = np.column_stack([df.PatientAge, (df.PatientSex.astype(str).str.upper() == "M").astype(float), df.BMI]).astype(float)
    t1 = tier1(df)
    for tier, F in (("tier1", t1), ("tier2", tier2(df))):
        for name in F.columns:
            o, l, h, p = fit_or(F[name].to_numpy(float), y, cov)
            rows.append(dict(tier=tier, disease=dn, feature=name, OR=o, ci_low=l, ci_high=h, p=p))
    S = sets(df, t1); oof0 = {}; aucs = {k: [] for k in S}
    for r in range(REPS):
        for k in S:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(S[k], y): oof[te] = model(S[k][tr], y[tr], S[k][te])
            aucs[k].append(roc_auc_score(y, oof))
            if r == 0: oof0[k] = oof
    for k in S:
        dl = delong_paired_auc_test(y, oof0["clinical"], oof0[k]) if k != "clinical" else {"p_value": np.nan}
        srows.append(dict(disease=dn, set=k, n_feat=S[k].shape[1], cv_auc=np.mean(aucs[k]), dAUC=np.mean(aucs[k]) - np.mean(aucs["clinical"]), delong_p_vs_clinical=dl["p_value"]))
    print(dn, {r["set"]: round(r["cv_auc"], 3) for r in srows if r["disease"] == dn}, flush=True)

w = pd.DataFrame(rows)
w["p_fdr"] = w.groupby(["tier", "disease"])["p"].transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
w["robust"] = w.p_fdr < .05
w = w.sort_values(["tier", "disease", "p"])
with pd.ExcelWriter(OUT / "prognosis_combo_features.xlsx") as xw:
    w[w.robust].round(4).to_excel(xw, sheet_name="robust", index=False)
    w.round(4).to_excel(xw, sheet_name="all", index=False)
    pd.DataFrame(srows).round(4).to_excel(xw, sheet_name="set_models", index=False)
    pd.DataFrame(desc).to_excel(xw, sheet_name="cohort", index=False)
pd.set_option("display.width", 250)
print(w.groupby(["tier", "disease"]).agg(n=("robust", "size"), robust=("robust", "sum")))
print(w[w.robust].groupby(["tier", "disease"]).head(4)[["tier", "disease", "feature", "OR", "p", "p_fdr"]].round(4).to_string(index=False))
print(pd.DataFrame(srows).round(3).to_string(index=False))
