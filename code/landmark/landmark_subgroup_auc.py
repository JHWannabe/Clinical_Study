from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 세부 그룹(부분군)별로 clinical(나이·성별·BMI) 대비 landmark 지표 세트를 더했을 때의 AUC 변화(ΔAUC)가 크게 다른 그룹이 있는지 탐색.
#  그룹(사전 지정): 성별 / 나이 3분위 / BMI 3분위 / 스캐너 / 성별x나이 / 성별xBMI (분위는 평가 코호트 안에서 계산)
#  모델: 로지스틱(clinical vs clinical+세트). 한 코호트 전체로 학습 -> 다른 코호트의 각 그룹에서 평가(Gangnam->Sinchon, Sinchon->Gangnam)
#  ΔAUC 95% CI: 그룹 내 환자 paired 부트스트랩(N_BOOT). 재현 = 두 방향 모두 CI가 0을 포함하지 않고 부호가 같음 (이름이 맞는 그룹만: 성별/나이·BMI 분위)
#  근거: 부분군 분석은 사전 지정, 전체 보고, 코호트 간 일관성으로 신뢰도 판단 (Wang NEJM 2007 doi:10.1056/NEJMsr077003, Sun BMJ 2010 doi:10.1136/bmj.c117)
#       성별 효과 차이(Framingham doi:10.1161/CIRCULATIONAHA.106.675355), 정상 BMI에서의 내장지방(Kidney Int Rep 2025 doi:10.1016/j.ekir.2025.103739)
# 출력: outputs/data_distribution/landmark_subgroup_auc.xlsx
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import LM, load_cohort

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
TIS = ["VAT", "SAT", "TAMA"]
LMN = [a.replace("_center", "") for a in LM]
N_BOOT, MIN_N, MIN_EV = 500, 150, 20
rng = np.random.default_rng(0)
lr_ = lambda a, b: np.log((a + 1) / (b + 1))


def load(c):
    df = load_cohort(c)[0].copy()
    lm = pd.read_excel(f"data/{c}_landmark_filtered.xlsx", sheet_name="landmarks").set_index("PatientID").loc[df.PatientID]
    vc = lm["manufacturer_model"].value_counts()
    df["scanner"] = lm["manufacturer_model"].where(lm["manufacturer_model"].map(vc) >= MIN_N, "other").to_numpy()
    for a, n in zip(LM, LMN):
        for t in TIS:
            df[f"{t}@{n}"] = df[f"{t}@{a}"]
    df["VAT/SAT@L3"] = lr_(df["VAT@L3"], df["SAT@L3"])
    df["VAT/TAMA@L3"] = lr_(df["VAT@L3"], df["TAMA@L3"])
    df["TAMA/(VAT+SAT)@L3"] = lr_(df["TAMA@L3"], df["VAT@L3"] + df["SAT@L3"])
    df["male"] = (df.PatientSex == "M").astype(float)
    return df


SETS = {"VAT@L3": ["VAT@L3"], "Tier1 ratios@L3": ["VAT/SAT@L3", "VAT/TAMA@L3", "TAMA/(VAT+SAT)@L3"], "VAT,SAT,TAMA@L3": [f"{t}@L3" for t in TIS]}
CLIN = ["PatientAge", "male", "BMI"]


def groups(df):  # 평가 코호트 안에서 그룹 마스크 정의 (이름: 마스크)
    g = {"all": np.ones(len(df), bool)}
    age_t = pd.qcut(df.PatientAge, 3, labels=["age_T1", "age_T2", "age_T3"])
    bmi_t = pd.qcut(df.BMI, 3, labels=["bmi_T1", "bmi_T2", "bmi_T3"])
    for s, nm in (("M", "male"), ("F", "female")):
        g[nm] = (df.PatientSex == s).to_numpy()
    for t in age_t.cat.categories:
        g[t] = (age_t == t).to_numpy()
    for t in bmi_t.cat.categories:
        g[t] = (bmi_t == t).to_numpy()
    for s, nm in (("M", "male"), ("F", "female")):
        for t in age_t.cat.categories:
            g[f"{nm}&{t}"] = ((df.PatientSex == s) & (age_t == t)).to_numpy()
        for t in bmi_t.cat.categories:
            g[f"{nm}&{t}"] = ((df.PatientSex == s) & (bmi_t == t)).to_numpy()
    for sc in df.scanner.unique():
        g[f"scanner:{sc}"] = (df.scanner == sc).to_numpy()
    return g


def fitpred(tr, te, cols, d):
    sc = StandardScaler().fit(tr[cols])
    m = LogisticRegression(max_iter=2000).fit(sc.transform(tr[cols]), tr[d])
    return m.predict_proba(sc.transform(te[cols]))[:, 1]


rows = []
data = {c: load(c) for c in ["gangnam", "sinchon"]}
for tr_c, te_c in (("gangnam", "sinchon"), ("sinchon", "gangnam")):
    tr, te = data[tr_c], data[te_c]
    grp = groups(te)
    for d in DIS:
        p0 = fitpred(tr, te, CLIN, d)
        y = te[d].to_numpy(int)
        for sname, cols in SETS.items():
            p1 = fitpred(tr, te, CLIN + cols, d)
            for gname, mask in grp.items():
                yy, a, b = y[mask], p0[mask], p1[mask]
                if mask.sum() < MIN_N or yy.sum() < MIN_EV or (1 - yy).sum() < MIN_EV:
                    continue
                delta = roc_auc_score(yy, b) - roc_auc_score(yy, a)
                bs = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(yy), len(yy))
                    if yy[i].min() == yy[i].max():
                        continue
                    bs.append(roc_auc_score(yy[i], b[i]) - roc_auc_score(yy[i], a[i]))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(train=tr_c, test=te_c, disease=d, set=sname, group=gname, n=int(mask.sum()), events=int(yy.sum()),
                                 auc_clinical=roc_auc_score(yy, a), auc_set=roc_auc_score(yy, b), delta=delta, ci_low=lo, ci_high=hi,
                                 ci_excl0=bool(lo > 0 or hi < 0)))
    print(tr_c, "->", te_c, "done", flush=True)

r = pd.DataFrame(rows)
# 재현: 이름이 같은 그룹이 두 방향 모두 존재, 부호 같고 둘 다 CI가 0 제외
a = r[r.train == "gangnam"].set_index(["disease", "set", "group"])
b = r[r.train == "sinchon"].set_index(["disease", "set", "group"])
j = a.join(b, lsuffix="_G2S", rsuffix="_S2G", how="inner")
j["replicated"] = j.ci_excl0_G2S & j.ci_excl0_S2G & (np.sign(j.delta_G2S) == np.sign(j.delta_S2G))
j = j.reset_index().sort_values("delta_G2S", key=lambda s: -s.abs())
with pd.ExcelWriter(f"{D}/landmark_subgroup_auc.xlsx") as xw:
    j.round(4).to_excel(xw, sheet_name="matched_groups", index=False)
    r.round(4).to_excel(xw, sheet_name="all_directions", index=False)
pd.set_option("display.width", 250)
print("matched groups:", len(j), "| 재현:", int(j.replicated.sum()))
cols = ["disease", "set", "group", "n_G2S", "events_G2S", "delta_G2S", "ci_low_G2S", "ci_high_G2S", "delta_S2G", "ci_low_S2G", "ci_high_S2G", "replicated"]
print(j[j.replicated][cols].round(3).to_string(index=False))
print("--- 방향별 |Δ| 상위(재현 무관) ---")
print(r.reindex(r.delta.abs().sort_values(ascending=False).index).head(15)[["train", "test", "disease", "set", "group", "n", "events", "auc_clinical", "auc_set", "delta", "ci_low", "ci_high"]].round(3).to_string(index=False))
print("--- 전체 그룹 'all' ---")
print(r[r.group == "all"][["train", "disease", "set", "delta", "ci_low", "ci_high"]].round(3).to_string(index=False))
