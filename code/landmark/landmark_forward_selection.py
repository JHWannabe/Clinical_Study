from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# liver_dome/pubis를 제외한 10개 landmark(T10~S1, femoral_head)에서 만들 수 있는 모든 feature 형태를 한 풀에 넣고 전진 선택으로 질환 AUC에 좋은 조합을 찾는다.
#  풀(조직 = VAT/SAT/TAMA/AEC):
#   point   : 조직@landmark 값                                 (40)
#   R       : 같은 landmark 안 조직 비율 VAT/SAT, VAT/TAMA, TAMA/(VAT+SAT)   (30)
#   L       : 같은 조직의 landmark 간 로그 비 t@Li/t@Lj (i<j)          (180)
#   X       : 서로 다른 조직 + 서로 다른 landmark 교차 로그 비 A@Li/B@Lj (A<B, i!=j)  (540)
#   range   : landmark~landmark 구간 평균 (i<j)                      (180)
#   rfrac   : 구간 평균 / 환자의 T10~femoral_head 평균                (180, patient-wise)
#   pw_mean : 조직@landmark / 환자의 T10~femoral_head 곡선 평균        (40, patient-wise)
#   pw_z    : (조직@landmark - 환자 곡선 평균) / 환자 곡선 SD          (40, patient-wise)
#  전진 선택: clinical(나이·성별·BMI) 위에 학습 코호트 5-fold CV AUC를 가장 올리는 feature를 하나씩 추가 (최대 MAXF개, 개선 < MIN_GAIN이면 중단).
#  선택은 학습 코호트 CV로만, 평가는 다른 코호트(양방향). 근거: 모델 구축 표본 성능은 과대평가되므로 CV/외부 검증 필요 (Steyerberg J Clin Epidemiol 2001
#  doi:10.1016/s0895-4356(01)00341-9), 보고는 TRIPOD (doi:10.1136/bmj.g7594). 대부분의 형태(L, X, range, AEC, patient-wise)는 질환 연관 문헌 근거를 못 찾음 -> 탐색적.
# 출력: outputs/data_distribution/landmark_forward_selection.xlsx
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import LM, load_cohort
from delong_utils import delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
TIS = ["VAT", "SAT", "TAMA", "AEC"]
EXCL = {"liver_dome", "inferior_pubic_margin"}
KEEP = [a for a in LM if a.replace("_center", "") not in EXCL]
LMN = [a.replace("_center", "") for a in KEEP]
MAXF, MIN_GAIN = 6, 0.002
lr_ = lambda a, b: np.log((a + 1) / (b + 1))
CLIN = ["PatientAge", "male", "BMI"]


# 코호트의 QC 통과 환자에 대해 모든 feature 풀을 만든다 -> (meta DataFrame, feature DataFrame, family 사전)
def build(c):
    meta = load_cohort(c)[0][["PatientID", "PatientSex", "PatientAge", "BMI"] + DIS].reset_index(drop=True)
    meta["male"] = (meta.PatientSex == "M").astype(float)
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[meta.PatientID]
    sl = np.stack([lm[f"{a}_slice"].to_numpy(int) for a in KEEP], axis=1)  # (n, 10), 1-indexed
    n = len(meta)
    cur = {}
    for t in TIS:
        pre = "aec" if t == "AEC" else t
        cur[t] = pd.read_excel(path, sheet_name=f"{pre}_total").set_index("PatientID").filter(regex=f"^{pre}_\\d+$").loc[meta.PatientID].to_numpy(float)
    pt = {t: np.stack([np.nan_to_num(cur[t][np.arange(n), sl[:, k] - 1]) for k in range(len(KEEP))], 1) for t in TIS}
    lo, hi = sl[:, 0], sl[:, -1]  # T10 ~ femoral_head 구간 (환자 정규화 기준)
    seg = {t: [cur[t][i, lo[i] - 1:hi[i]] for i in range(n)] for t in TIS}
    wmean = {t: np.array([np.nanmean(s) for s in seg[t]]) for t in TIS}
    wsd = {t: np.array([np.nanstd(s) for s in seg[t]]) for t in TIS}
    F, fam = {}, {}

    def add(name, v, f):
        F[name], fam[name] = np.asarray(v, float), f

    for t in TIS:
        for k, a in enumerate(LMN):
            add(f"{t}@{a}", pt[t][:, k], "point")
            add(f"pw_mean:{t}@{a}", pt[t][:, k] / np.where(wmean[t] > 0, wmean[t], np.nan), "pw_mean")
            add(f"pw_z:{t}@{a}", (pt[t][:, k] - wmean[t]) / np.where(wsd[t] > 0, wsd[t], np.nan), "pw_z")
        for i, j in itertools.combinations(range(len(KEEP)), 2):
            add(f"{t}@{LMN[i]}/{LMN[j]}", lr_(pt[t][:, i], pt[t][:, j]), "L")
            m = np.array([np.nanmean(cur[t][r, sl[r, i] - 1:sl[r, j]]) for r in range(n)])
            add(f"range:{t}[{LMN[i]}~{LMN[j]}]", m, "range")
            add(f"rfrac:{t}[{LMN[i]}~{LMN[j]}]", m / np.where(wmean[t] > 0, wmean[t], np.nan), "rfrac")
    for k, a in enumerate(LMN):
        v, s, tm = pt["VAT"][:, k], pt["SAT"][:, k], pt["TAMA"][:, k]
        add(f"R:VAT/SAT@{a}", lr_(v, s), "R")
        add(f"R:VAT/TAMA@{a}", lr_(v, tm), "R")
        add(f"R:TAMA/(VAT+SAT)@{a}", lr_(tm, v + s), "R")
    for (A, B) in itertools.combinations(TIS, 2):
        for i, j in itertools.permutations(range(len(KEEP)), 2):
            add(f"X:{A}@{LMN[i]}/{B}@{LMN[j]}", lr_(pt[A][:, i], pt[B][:, j]), "X")
    F = pd.DataFrame(F)
    F = F.replace([np.inf, -np.inf], np.nan)
    F = F.fillna(F.median())  # 분모가 0인 극소수 값은 중앙값으로
    F = F.loc[:, F.std() > 0]
    return meta, F, {k: v for k, v in fam.items() if k in F.columns}


def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


def cv_auc(X, y, splits):
    return np.mean([roc_auc_score(y[te], fitpred(X[tr], y[tr], X[te])) for tr, te in splits])


data = {c: build(c) for c in ["gangnam", "sinchon"]}
common = [c for c in data["gangnam"][1].columns if c in data["sinchon"][1].columns]  # 두 코호트에 모두 있는 feature만
print("pool size:", len(common), flush=True)
fam = data["gangnam"][2]
rows = []
for tr_c, te_c in (("gangnam", "sinchon"), ("sinchon", "gangnam")):
    mA, FA, _ = data[tr_c]
    mB, FB, _ = data[te_c]
    FAm, FBm = FA[common].to_numpy(float), FB[common].to_numpy(float)
    for d in DIS:
        yA, yB = mA[d].to_numpy(int), mB[d].to_numpy(int)
        splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(mA[CLIN], yA))
        CA, CB = mA[CLIN].to_numpy(float), mB[CLIN].to_numpy(float)
        p_clin = fitpred(CA, yA, CB)
        cur_cv, chosen = cv_auc(CA, yA, splits), []
        for step in range(1, MAXF + 1):
            best = (None, cur_cv)
            for j in range(len(common)):
                if j in chosen:
                    continue
                a = cv_auc(np.column_stack([CA, FAm[:, chosen + [j]]]), yA, splits)
                if a > best[1]:
                    best = (j, a)
            if best[0] is None or best[1] - cur_cv < MIN_GAIN:
                break
            chosen.append(best[0])
            cur_cv = best[1]
            XA, XB = np.column_stack([CA, FAm[:, chosen]]), np.column_stack([CB, FBm[:, chosen]])
            p = fitpred(XA, yA, XB)
            dl = delong_paired_auc_test(yB, p_clin, p)
            rows.append(dict(train=tr_c, test=te_c, disease=d, step=step, added=common[best[0]], family=fam[common[best[0]]], cv_auc=cur_cv,
                             clinical_ext=dl["auc_a"], ext_auc=dl["auc_b"], ext_delta=dl["diff"], delong_p=dl["p_value"]))
            print(tr_c, d, step, common[best[0]], round(cur_cv, 3), round(dl["auc_b"], 3), flush=True)
r = pd.DataFrame(rows)
final = r.sort_values("step").groupby(["train", "disease"]).tail(1)
with pd.ExcelWriter(f"{D}/landmark_forward_selection.xlsx") as xw:
    final.round(4).to_excel(xw, sheet_name="final", index=False)
    r.round(4).to_excel(xw, sheet_name="path", index=False)
    pd.Series(fam).value_counts().rename("n_features").to_frame().to_excel(xw, sheet_name="pool")
pd.set_option("display.width", 250)
print(r[["train", "disease", "step", "added", "family", "cv_auc", "clinical_ext", "ext_auc", "ext_delta", "delong_p"]].round(3).to_string(index=False))
print(r.groupby("family").size())
