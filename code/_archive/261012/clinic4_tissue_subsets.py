from __future__ import annotations

# VAT·SAT·TAMA·AEC 4종을 모두 쓰지 않고 조직 부분집합(15가지)을 쓴 모델 비교 (SINGLE_DIS·DEF_MODE·OUT_DIR 환경변수로 질환별 실행, outputs/<OUT_DIR>/clinic4_tissue.xlsx)
#  표현 2가지: 구간 합산(조직당 1개), L1·L3·L5 평균 대비(조직당 약 4개) / 모델 2가지: clinic4에 추가, CT만(clinic4 없이)
#  평가: 풀링 5-fold CV(3회 반복, Nadeau-Bengio 보정 t)와 병원 간 외부검증(DeLong), 질환별 BH-FDR (조합이 많아 다중비교 부담이 크므로 q를 함께 본다)
# 데이터 적재·feature 조합·nested LR·검정 함수는 clinic4_feature_combos.py의 앞부분(분석 실행 전까지)을 그대로 재사용한다.
import warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
src = (HERE / "clinic4_feature_combos.py").read_text(encoding="utf-8").split("rows = []\nfor d in DIS:")[0]
g = {"__file__": str(HERE / "clinic4_feature_combos.py")}
exec(compile(src, "clinic4_feature_combos_head", "exec"), g)
A, C4, DIS, REPS, OUT, MEAS, LMS3 = g["A"], g["C4"], g["DIS"], g["REPS"], g["OUT"], g["MEAS"], g["LMS3"]
fit_predict, nb_test, bh_fdr, delong, auc, SKF = g["fit_predict"], g["nb_test"], g["bh_fdr"], g["delong_paired_auc_test"], g["roc_auc_score"], g["StratifiedKFold"]
tot, pw = g["tot"], g["pw"]
coh = A.cohort.to_numpy()
REP = {"구간 합산": lambda ms: tot(ms, "sum"), "L1·L3·L5 평균 대비": lambda ms: pw(ms, LMS3)}
SUBS = [s for r in range(1, 5) for s in combinations(MEAS, r)]
P, E = [], []
for d in DIS:
    y = A[d].to_numpy(int); sets = {("clinic4", "-", "-"): C4}
    for rep, f in REP.items():
        for sub in SUBS:
            Z = f(list(sub))
            sets[("clinic4+", rep, "+".join(sub))] = np.column_stack([C4, Z]); sets[("CT만", rep, "+".join(sub))] = Z
    folds = {k: [] for k in sets}
    for k_ in range(REPS):
        for tr, te in SKF(5, shuffle=True, random_state=k_).split(C4, y * 2 + (coh == "sinchon")):
            for key, X in sets.items(): folds[key].append(auc(y[te], fit_predict(X, "tissue", y, tr, te)))
    base = np.array(folds[("clinic4", "-", "-")])
    for key, X in sets.items():
        r = np.array(folds[key]); P.append(dict(disease=d, mode=key[0], rep=key[1], tissues=key[2], n_tissue=0 if key[2] == "-" else key[2].count("+") + 1, n_feat=X.shape[1], AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(), p=np.nan if key[0] == "clinic4" else nb_test(r - base)))
    for tr_c, te_c in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        a, b = np.where(coh == tr_c)[0], np.where(coh == te_c)[0]; sc = {key: fit_predict(X, "tissue", y, a, b) for key, X in sets.items()}
        for key in sets:
            t = delong(y[b], sc[("clinic4", "-", "-")], sc[key]) if key[0] != "clinic4" else {"diff": 0, "p_value": np.nan}
            E.append(dict(train=tr_c, test=te_c, disease=d, mode=key[0], rep=key[1], tissues=key[2], AUC=auc(y[b], sc[key]), dAUC=t["diff"], p=t["p_value"]))
    print(d, "완료", flush=True)
P, E = pd.DataFrame(P), pd.DataFrame(E)
P["q"] = P.groupby("disease").p.transform(lambda p: bh_fdr(p.fillna(1).to_numpy())); E["q"] = E.groupby(["train", "disease"]).p.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
with pd.ExcelWriter(OUT / "clinic4_tissue.xlsx") as w:
    P.to_excel(w, sheet_name="pooled_cv", index=False); E.to_excel(w, sheet_name="external", index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
for d in DIS:
    for m in ["clinic4+", "CT만"]:
        print(P[(P.disease == d) & (P["mode"] == m)].sort_values("AUC", ascending=False).head(4).round(3).to_string(index=False))
