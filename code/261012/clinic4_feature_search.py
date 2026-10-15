from __future__ import annotations

# clinic4 baseline 대비 AUC 개선 탐색(FPCA 제외): BMI/상호작용, landmark 체성분 raw, 환자 구간평균 정규화, 조직 간 비율을 풀링 5-fold CV(3회 반복, fold별 AUC 평균±표준편차, 검정은 Nadeau-Bengio 보정 t)로 비교 (단독 실행, outputs/261012 저장)
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from delong_utils import delong_paired_auc_test, bh_fdr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/261012"; OUT.mkdir(parents=True, exist_ok=True)
DIS, REPS = ["HTN", "DM", "CKD"], 3
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

MEAS = ["VAT", "SAT", "TAMA", "AEC"]
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center",
      "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
SHEET = {m: f"{m.lower()}_total" if m == "AEC" else f"{m}_total" for m in MEAS}


# 코호트의 환자 메타 + landmark slice 값({m}@{lm}) + 환자 구간(liver_dome~pubis) 평균({m}_mean)
def load(cohort: str) -> pd.DataFrame:
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    cur = {m: pd.read_excel(path, sheet_name=SHEET[m]).set_index("PatientID").filter(regex=rf"^{m.lower() if m == 'AEC' else m}_\d+$") for m in MEAS}
    meta = meta[meta.PatientID.isin(set.intersection(*[set(c.index) for c in cur.values()]) & set(lm.index))]
    meta = keep_rows(meta.dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + ALL_DIS).reset_index(drop=True))
    feat = {}
    for m in MEAS:
        v = np.nan_to_num(cur[m].loc[meta.PatientID].to_numpy(float)); sl = lm.loc[meta.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
        for j, a in enumerate(LM): feat[f"{m}@{a}"] = v[np.arange(len(v)), sl[:, j] - 1]
        feat[f"{m}_mean"] = [v[i, sl[i, 0] - 1:sl[i, -1]].mean() for i in range(len(v))]
    meta["sex"] = (meta.PatientSex.astype(str).str.upper() == "M").astype(float); meta["cohort"] = cohort
    return pd.concat([meta, pd.DataFrame(feat)], axis=1)


A = pd.concat([load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
col = lambda m, lms=LM: [f"{m}@{a}" for a in lms]
LMS3 = ["L1_center", "L3_center", "L5_center"]
bmi = A.Weight / (A.Height / 100) ** 2
C4 = np.column_stack([A.sex, A.PatientAge, A.Height, A.Weight])
inter = np.column_stack([bmi, A.sex * A.Weight, A.sex * A.Height, A.sex * A.PatientAge, A.PatientAge ** 2])
raw = lambda lms: A[[c for m in MEAS for c in col(m, lms)]].to_numpy(float)
norm = lambda lms: np.column_stack([A[c].to_numpy() / (A[f"{c.split('@')[0]}_mean"].to_numpy() + 1e-6) for m in MEAS for c in col(m, lms)])
means = A[[f"{m}_mean" for m in MEAS]].to_numpy(float)
ratio = np.column_stack([A[f"VAT@{a}"] / (A[f"SAT@{a}"] + 1e-6) for a in LM] + [A[f"TAMA@{a}"] / bmi for a in LM] + [A[f"VAT@{a}"] / (A[f"TAMA@{a}"] + 1e-6) for a in LM])
SETS = {"clinic4": C4, "c4+BMI/interaction": np.column_stack([C4, inter]),
        "c4+LM raw(L1,L3,L5)": np.column_stack([C4, raw(LMS3)]), "c4+LM raw(all12)": np.column_stack([C4, raw(LM)]),
        "c4+LM/mean(L1,L3,L5)": np.column_stack([C4, norm(LMS3), means]), "c4+LM/mean(all12)": np.column_stack([C4, norm(LM), means]),
        "c4+tissue ratios": np.column_stack([C4, ratio]), "c4+inter+ratios": np.column_stack([C4, inter, ratio]),
        "c4+inter+LM/mean(all12)+ratios": np.column_stack([C4, inter, norm(LM), means, ratio])}


# LR 하이퍼파라미터(L2 penalty의 C)를 학습 fold 내부 5-fold AUC로 고르는 모델 (nested: 평가 fold는 선택에 쓰지 않음)
def mk():
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                        {"logisticregression__C": np.logspace(-3, 1, 9)}, cv=5, scoring="roc_auc", n_jobs=-1)


# 풀링 5-fold CV를 REPS회 반복한 fold별 AUC 배열(REPS*5개; 층화: 라벨×코호트, 반복 k는 모든 feature set에서 같은 분할이라 fold가 짝지어짐)
def cv(X, y):
    r = []
    for k in range(REPS):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, y * 2 + (A.cohort == "sinchon").to_numpy()):
            r.append(roc_auc_score(y[te], mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return np.array(r)


# Nadeau-Bengio 보정 paired t 검정: fold별 AUC 차이(J=반복x5개)의 평균이 0인지 (fold 간 학습셋 중복에 의한 분산 과소추정 보정, 시험/학습 = 1/4)
def nb_test(diff):
    J, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else float(2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / J + .25) * v), J - 1))


rows = []
for d in DIS:
    y = A[d].to_numpy(int); base = None
    for name, X in SETS.items():
        r = cv(X, y)  # clinic4도 같은 nested 절차로 튜닝 (첫 set)
        if name == "clinic4": base = r
        rows.append(dict(disease=d, feature_set=name, n_feat=X.shape[1], AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(), win_folds=int((r > base).sum()),
                         p_cv=np.nan if name == "clinic4" else nb_test(r - base)))
        print(d, name, round(r.mean(), 3), flush=True)
res = pd.DataFrame(rows)
res["p_fdr_in_disease"] = res.groupby("disease").p_cv.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
res.to_excel(OUT / "clinic4_feature_search.xlsx", index=False)
best = res
for v in ["AUC", "AUC_sd", "dAUC", "p_cv", "p_fdr_in_disease"]:
    print(v); print(best.pivot(index="feature_set", columns="disease", values=v)[DIS].round(3).loc[list(SETS)])

# 코호트 간 외부검증: 한 코호트로 학습(C는 train 내부 5-fold로 선택)해 다른 코호트에서 clinic4 대비 AUC·DeLong 비교
ext = []
for tr, te in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
    a, b = (A.cohort == tr).to_numpy(), (A.cohort == te).to_numpy()
    for d in DIS:
        y = A[d].to_numpy(int); sc = {}
        for name, X in SETS.items():
            sc[name] = mk().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        for name in SETS:
            t = delong_paired_auc_test(y[b], sc["clinic4"], sc[name]) if name != "clinic4" else {"auc_a": roc_auc_score(y[b], sc[name]), "diff": 0, "p_value": np.nan}
            ext.append(dict(train=tr, test=te, disease=d, feature_set=name, AUC=roc_auc_score(y[b], sc[name]), dAUC=t["diff"], p_delong=t["p_value"]))
ext = pd.DataFrame(ext)
ext["p_fdr"] = ext.groupby(["train", "disease"]).p_delong.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
ext.to_excel(OUT / "clinic4_feature_search_external.xlsx", index=False)
for (tr, te), g in ext.groupby(["train", "test"]):
    print(f"\n[train {tr} -> test {te}]")
    for v in ["AUC", "dAUC", "p_fdr"]: print(v); print(g.pivot(index="feature_set", columns="disease", values=v)[DIS].round(3).loc[list(SETS)])
