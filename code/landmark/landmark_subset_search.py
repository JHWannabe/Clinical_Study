from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# liver_dome, inferior_pubic_margin을 제외한 10개 landmark(T10~S1, femoral_head)에서 질환 AUC에 가장 좋은 landmark 조합 탐색.
#  feature 형태 3가지: V = 조합 landmark들의 VAT/SAT/TAMA 값, R = 조합 landmark들의 같은 landmark 안 조직 비율(VAT/SAT, VAT/TAMA, TAMA/(VAT+SAT)),
#     L = 조합 landmark들 사이의 같은 조직 로그 비(조합 안 모든 쌍 i<j의 t@Li / t@Lj, t = VAT/SAT/TAMA; 크기 2 이상). L은 문헌 근거 못 찾음(탐색적)
#  실행: 인자 없으면 V,R,L 전부(landmark_subset_search.xlsx), 인자로 형태를 주면(예: L) 그 형태만 landmark_subset_search_L.xlsx로 저장
#  조합 크기 1~5 (모든 부분집합 638개). 모델 = clinical(나이·성별·BMI) + feature, 로지스틱.
#  선택은 "학습 코호트" 안의 repeated 5-fold CV AUC로만 하고, 선택된 조합을 다른 코호트에서 평가 (양방향: G->S, S->G) -> 선택 편향 없는 외부 AUC.
#  기준선: clinical / L3 단독 / L2+L3 / 10개 전체. 근거: L3가 전체 VAT·SAT·근육 부피를 가장 잘 대표 (Am J Clin Nutr 2015 doi:10.3945/ajcn.115.111203),
#         VAT는 L2-L3 부근 단면이 전체 부피 예측력 최고 (Eur J Clin Nutr 1996, PMID 8735309). 그 외 조합은 탐색적.
# 출력: outputs/data_distribution/landmark_subset_search.xlsx
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import LM, load_cohort
from delong_utils import delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
TIS = ["VAT", "SAT", "TAMA"]
EXCL = {"liver_dome", "inferior_pubic_margin"}
LMN = [a.replace("_center", "") for a in LM if a.replace("_center", "") not in EXCL]  # 10개
MAXK = 5
FT = sys.argv[1:] or ["V", "R", "L"]
SUF = "" if sys.argv[1:] == [] else "_" + "".join(FT)
lr_ = lambda a, b: np.log((a + 1) / (b + 1))


def load(c):
    df = load_cohort(c)[0].copy()
    for a in LM:
        n = a.replace("_center", "")
        if n in EXCL:
            continue
        v, s, t = df[f"VAT@{a}"], df[f"SAT@{a}"], df[f"TAMA@{a}"]
        df[f"V:VAT@{n}"], df[f"V:SAT@{n}"], df[f"V:TAMA@{n}"] = v, s, t
        df[f"R:VAT/SAT@{n}"], df[f"R:VAT/TAMA@{n}"], df[f"R:TAMA/(VAT+SAT)@{n}"] = lr_(v, s), lr_(v, t), lr_(t, v + s)
    for a, b in itertools.combinations(LMN, 2):  # 같은 조직의 landmark 간 로그 비
        for t in TIS:
            df[f"L:{t}@{a}/{b}"] = lr_(df[f"V:{t}@{a}"], df[f"V:{t}@{b}"])
    df["male"] = (df.PatientSex == "M").astype(float)
    return df


CLIN = ["PatientAge", "male", "BMI"]
def pair_cols(subset):  # 조합 안 landmark 쌍의 같은 조직 로그 비 컬럼명 (load에서 만든 L: 컬럼)
    return [f"L:{t}@{a}/{b}" for a, b in itertools.combinations(subset, 2) for t in TIS]


COLS = {"V": lambda subset: [f"V:{t}@{n}" for n in subset for t in TIS], "L": pair_cols,
        "R": lambda subset: [f"R:{r}@{n}" for n in subset for r in ("VAT/SAT", "VAT/TAMA", "TAMA/(VAT+SAT)")]}


def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]


def cv_auc(X, y, splits):
    return np.mean([roc_auc_score(y[te], fitpred(X[tr], y[tr], X[te])) for tr, te in splits])


data = {c: load(c) for c in ["gangnam", "sinchon"]}
subsets = [s for k in range(1, MAXK + 1) for s in itertools.combinations(LMN, k)]
print("subsets:", len(subsets), flush=True)
rows, picks = [], []
for tr_c, te_c in (("gangnam", "sinchon"), ("sinchon", "gangnam")):
    A, B = data[tr_c], data[te_c]
    for d in DIS:
        yA, yB = A[d].to_numpy(int), B[d].to_numpy(int)
        splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(A[CLIN], yA))
        base_cv = cv_auc(A[CLIN].to_numpy(float), yA, splits)
        p_clin = fitpred(A[CLIN].to_numpy(float), yA, B[CLIN].to_numpy(float))
        for ft in FT:
            res = []
            for sub in subsets:
                if ft == "L" and len(sub) < 2:
                    continue  # 비율은 landmark 2개 이상 필요
                cols = CLIN + COLS[ft](sub)
                XA, XB = A[cols].to_numpy(float), B[cols].to_numpy(float)
                res.append((sub, cv_auc(XA, yA, splits), roc_auc_score(yB, fitpred(XA, yA, XB))))
            r = pd.DataFrame(res, columns=["subset", "cv_auc_train", "ext_auc_test"])
            r["size"] = r.subset.map(len)
            r["subset"] = r.subset.map(lambda s: "+".join(s))
            best = r.sort_values("cv_auc_train", ascending=False).iloc[0]  # 학습 코호트 CV로만 선택
            sub_best = tuple(best.subset.split("+"))
            p_best = fitpred(A[CLIN + COLS[ft](sub_best)].to_numpy(float), yA, B[CLIN + COLS[ft](sub_best)].to_numpy(float))
            dl = delong_paired_auc_test(yB, p_clin, p_best)
            ext_of = lambda sub: roc_auc_score(yB, fitpred(A[CLIN + COLS[ft](sub)].to_numpy(float), yA, B[CLIN + COLS[ft](sub)].to_numpy(float)))  # 기준선 외부 AUC
            rho = spearmanr(r.cv_auc_train, r.ext_auc_test)[0]
            top20 = r.sort_values("cv_auc_train", ascending=False).head(20).subset.str.split("+").explode().value_counts()
            rows.append(dict(train=tr_c, test=te_c, disease=d, ftype=ft, clinical_cv=base_cv, clinical_ext=dl["auc_a"], best_subset=best.subset, best_size=int(best["size"]),
                             best_cv=best.cv_auc_train, best_ext=dl["auc_b"], ext_delta=dl["diff"], delong_p=dl["p_value"],
                             L3_ext=ext_of(("L3",)), L2L3_ext=ext_of(("L2", "L3")), all10_ext=ext_of(tuple(LMN)),
                             oracle_best_ext=r.ext_auc_test.max(), oracle_subset=r.sort_values("ext_auc_test", ascending=False).iloc[0].subset,
                             spearman_cv_vs_ext=rho, top20_landmarks=", ".join(f"{k}:{v}" for k, v in top20.items())))
            r.assign(train=tr_c, test=te_c, disease=d, ftype=ft).to_csv(f"{D}/_subset{SUF}_{tr_c}_{d}_{ft}.csv", index=False)
            print(tr_c, d, ft, best.subset, round(best.cv_auc_train, 3), round(dl["auc_b"], 3), flush=True)
out = pd.DataFrame(rows)
allr = pd.concat([pd.read_csv(f) for f in _p.Path(D).glob(f"_subset{SUF}_*.csv")])
with pd.ExcelWriter(f"{D}/landmark_subset_search{SUF}.xlsx") as xw:
    out.round(4).to_excel(xw, sheet_name="selected", index=False)
    allr.round(4).to_excel(xw, sheet_name="all_subsets", index=False)
for f in _p.Path(D).glob(f"_subset{SUF}_*.csv"):
    f.unlink()
pd.set_option("display.width", 250)
print(out[["train", "disease", "ftype", "best_subset", "best_cv", "clinical_ext", "best_ext", "ext_delta", "delong_p", "L3_ext", "L2L3_ext", "all10_ext", "oracle_best_ext", "spearman_cv_vs_ext"]].round(3).to_string(index=False))
