from __future__ import annotations

# 비율 feature log((a+1)/(b+1))를 (같은 landmark·다른 조직 / 같은 조직·다른 landmark / 다른 조직·다른 landmark)로 나눠
# 일관성(두 코호트 모두 유의, 같은 방향)과 clinic4 추가 효과(풀링 nested CV, 강남↔신촌 외부검증)를 비교 (단독 실행, outputs/261012 저장)
import sys, warnings
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from delong_utils import delong_paired_auc_test, bh_fdr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/261012"
DIS, REPS, TOPK = ["HTN", "DM", "CKD"], 3, 3
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
ABB = [a.replace("_center", "").replace("liver_dome", "LD").replace("femoral_head", "FH").replace("inferior_pubic_margin", "PM") for a in LM]
SHEET = {m: f"{m.lower()}_total" if m == "AEC" else f"{m}_total" for m in MEAS}


# 코호트의 환자 메타 + landmark slice 값(조직×landmark)
def load(cohort: str):
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    cur = {m: pd.read_excel(path, sheet_name=SHEET[m]).set_index("PatientID").filter(regex=rf"^{m.lower() if m == 'AEC' else m}_\d+$") for m in MEAS}
    meta = meta[meta.PatientID.isin(set.intersection(*[set(c.index) for c in cur.values()]) & set(lm.index))]
    meta = keep_rows(meta.dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + ALL_DIS).reset_index(drop=True))
    sl = lm.loc[meta.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
    V = {m: np.nan_to_num(cur[m].loc[meta.PatientID].to_numpy(float))[np.arange(len(meta))[:, None], sl - 1] for m in MEAS}  # (n, 12)
    meta["sex"] = (meta.PatientSex.astype(str).str.upper() == "M").astype(float); meta["cohort"] = cohort
    return meta, V


parts = [load(c) for c in ["gangnam", "sinchon"]]
A = pd.concat([p[0] for p in parts], ignore_index=True)
V = {m: np.vstack([p[1][m] for p in parts]) for m in MEAS}
C4 = np.column_stack([A.sex, A.PatientAge, A.Height, A.Weight])
lg = lambda a, b: np.log((a + 1) / (b + 1))

# 비율 후보: (이름, 열). 같은 조직 쌍은 landmark 순서쌍 중 앞<뒤만(역수는 부호만 반대), 다른 조직 쌍은 (조직i<조직j) x 모든 landmark 순서쌍
TIERS = {"같은 landmark·다른 조직": [], "같은 조직·다른 landmark": [], "다른 조직·다른 landmark": []}
for a in range(12):
    for m1, m2 in combinations(MEAS, 2): TIERS["같은 landmark·다른 조직"].append((f"{m1}/{m2}@{ABB[a]}", lg(V[m1][:, a], V[m2][:, a])))
for m in MEAS:
    for a, b in combinations(range(12), 2): TIERS["같은 조직·다른 landmark"].append((f"{m}@{ABB[a]}/{ABB[b]}", lg(V[m][:, a], V[m][:, b])))
for m1, m2 in combinations(MEAS, 2):
    for a, b in permutations(range(12), 2): TIERS["다른 조직·다른 landmark"].append((f"{m1}@{ABB[a]}/{m2}@{ABB[b]}", lg(V[m1][:, a], V[m2][:, b])))
NAMES = {t: [n for n, _ in v] for t, v in TIERS.items()}; MAT = {t: np.column_stack([c for _, c in v]) for t, v in TIERS.items()}
MAT["전체 비율"] = np.column_stack(list(MAT.values())); NAMES["전체 비율"] = sum([NAMES[t] for t in TIERS], [])
print({t: m.shape[1] for t, m in MAT.items()}, flush=True)


# ---- 1) 일관성: 코호트별 Mann-Whitney BH-FDR<0.05이면서 두 코호트 방향이 같은 비율 수 ----
cons = []
for d in DIS:
    for t, M in MAT.items():
        st = {}
        for c in ["gangnam", "sinchon"]:
            i = (A.cohort == c).to_numpy(); y = A[d].to_numpy(int)[i]
            p = np.nan_to_num(mannwhitneyu(M[i][y == 1], M[i][y == 0], axis=0).pvalue, nan=1.0)
            st[c] = (bh_fdr(p) < .05, np.sign(M[i][y == 1].mean(0) - M[i][y == 0].mean(0)), p)
        both = st["gangnam"][0] & st["sinchon"][0] & (st["gangnam"][1] == st["sinchon"][1])
        best = np.argmin(np.maximum(st["gangnam"][2], st["sinchon"][2]) + (~both) * 10)
        cons.append(dict(disease=d, tier=t, n_cand=M.shape[1], n_sig_gangnam=int(st["gangnam"][0].sum()), n_sig_sinchon=int(st["sinchon"][0].sum()), n_consistent=int(both.sum()),
                         best_feature=NAMES[t][best] if both.any() else "-"))
cons = pd.DataFrame(cons); print(cons.to_string(), flush=True)


# ---- 2) clinic4 추가 효과: 학습 fold에서 단변량 AUC가 가장 벗어난 상위 TOPK 비율만 선택 (평가 데이터는 선택에 쓰지 않음) ----
def mk():
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), {"logisticregression__C": np.logspace(-3, 1, 9)}, cv=5, scoring="roc_auc", n_jobs=-1)


# 학습 fold 단변량 AUC(순위 기반)에서 |AUC-0.5|가 큰 상위 k개 열
def top_cols(Z, y, k=TOPK):
    r = rankdata(Z, axis=0); n1 = int(y.sum()); n0 = len(y) - n1
    auc = (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.argsort(-np.abs(auc - .5))[:k]


PICK = Counter()


def fit_predict(Z, y, tr, te, key=None):
    if Z is None: X = C4
    else:
        j = top_cols(Z[tr], y[tr]); X = np.column_stack([C4, Z[:, j]])
        if key: PICK.update((key, int(c)) for c in j)
    return mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


SETS = {"clinic4": None, **{f"+{t} 상위{TOPK}": t for t in MAT}}
# Nadeau-Bengio 보정 paired t 검정: fold별 AUC 차이(J=반복x5개)의 평균이 0인지 (fold 간 학습셋 중복에 의한 분산 과소추정 보정, 시험/학습 = 1/4)
def nb_test(diff):
    J, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else float(2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / J + .25) * v), J - 1))


rows, ext = [], []
for d in DIS:
    y = A[d].to_numpy(int); strat = y * 2 + (A.cohort == "sinchon").to_numpy(); base = None
    for name, t in SETS.items():
        Z = None if t is None else MAT[t]; r = []
        for k in range(REPS):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(C4, strat):
                r.append(roc_auc_score(y[te], fit_predict(Z, y, tr, te, (d, t) if k == 0 else None)))
        r = np.array(r)
        if t is None: base = r
        rows.append(dict(disease=d, feature_set=name, AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(), p_cv=np.nan if t is None else nb_test(r - base)))
        print(d, name, round(r.mean(), 3), flush=True)
    for tr_c, te_c in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        a, b = np.where((A.cohort == tr_c).to_numpy())[0], np.where((A.cohort == te_c).to_numpy())[0]
        sc = {n: fit_predict(None if t is None else MAT[t], y, a, b) for n, t in SETS.items()}
        for n in SETS:
            e = delong_paired_auc_test(y[b], sc["clinic4"], sc[n]) if n != "clinic4" else {"diff": 0, "p_value": np.nan}
            ext.append(dict(train=tr_c, test=te_c, disease=d, feature_set=n, AUC=roc_auc_score(y[b], sc[n]), dAUC=e["diff"], p_delong=e["p_value"]))
res = pd.DataFrame(rows); res["p_fdr"] = res.groupby("disease").p_cv.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
ext = pd.DataFrame(ext); ext["p_fdr"] = ext.groupby(["train", "disease"]).p_delong.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
pick = pd.DataFrame([dict(disease=d, tier=t, feature=NAMES[t][c], n_selected_folds=n) for (d, t), c, n in ((k[0], k[1], n) for k, n in PICK.items())])
pick = pick.sort_values(["disease", "tier", "n_selected_folds"], ascending=[True, True, False])
with pd.ExcelWriter(OUT / "clinic4_ratio_cross.xlsx") as w:
    cons.to_excel(w, sheet_name="consistency", index=False); res.to_excel(w, sheet_name="pooled_cv", index=False)
    ext.to_excel(w, sheet_name="external", index=False); pick.to_excel(w, sheet_name="selected", index=False)
fm = lambda g: g.assign(s=lambda x: x.AUC.round(3).astype(str) + " (" + x.dAUC.round(3).map("{:+.3f}".format) + ", q" + x.p_fdr.round(3).astype(str) + ")").pivot(index="feature_set", columns="disease", values="s")[DIS].loc[list(SETS)]
print("\n[pooled]"); print(fm(res).to_string())
for tr_c, g in ext.groupby("train"): print(f"\n[train {tr_c}]"); print(fm(g).to_string())
