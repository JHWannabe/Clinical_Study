from __future__ import annotations

# 성향점수 매칭(PSM)으로 정상군을 1:1 언더샘플링한 뒤 baseline(clinic4) vs CT 변수 추가/CT만 모델 비교 (SINGLE_DIS·OUT_DIR 환경변수로 질환별 실행, outputs/261012_single/<질환>/clinic4_psm.xlsx)
#  매칭: 병원 안에서 나이·성별·BMI 로지스틱 성향점수, logit 거리 최근접 1:1 비복원, caliper = 0.2 x SD(logit), 환자 순서를 seed마다 섞어 5회 반복
#  평가: 매칭된 자료의 풀링 5-fold CV(3회 반복, 5개 매칭 모두 합친 fold별 AUC) + 병원별 매칭 자료로 외부검증(매칭 5회 AUC 평균, DeLong p는 매칭 seed 0)
# 데이터 적재·feature 조합·nested LR·검정 함수는 clinic4_feature_combos.py의 앞부분(분석 실행 전까지)을 그대로 재사용한다.
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
src = (HERE / "clinic4_feature_combos.py").read_text(encoding="utf-8").split("rows = []\nfor d in DIS:")[0]
g = {"__file__": str(HERE / "clinic4_feature_combos.py")}
exec(compile(src, "clinic4_feature_combos_head", "exec"), g)
A, C4, SETS, DIS, REPS, OUT = g["A"], g["C4"], g["SETS"], g["DIS"], g["REPS"], g["OUT"]
fit_predict, nb_test, bh_fdr, delong, auc, SKF = g["fit_predict"], g["nb_test"], g["bh_fdr"], g["delong_paired_auc_test"], g["roc_auc_score"], g["StratifiedKFold"]
USE = ["clinic4", "(참고) +LM raw(L1,L3,L5)", "(참고) +LM/mean(L1,L3,L5)", "+분포 기반(질환별)", "+구간 합산(4종)", "CT만 구간 합산(4종)", "CT만 LM/mean(L1,L3,L5)"]
SEEDS = 5
coh = A.cohort.to_numpy()
COV = np.column_stack([A.PatientAge, A.sex, A.Weight / (A.Height / 100) ** 2]).astype(float)  # 매칭 변수: 나이, 성별, BMI


# 병원 안 성향점수 매칭: 질환군을 seed마다 섞어 가장 가까운(logit PS) 미사용 정상군을 caliper 안에서 1:1로 짝지음 -> 매칭된 행 인덱스
def match(y, seed):
    out = []
    for c in ["gangnam", "sinchon"]:
        i = np.where(coh == c)[0]; X = StandardScaler().fit_transform(COV[i]); p = LogisticRegression(max_iter=1000).fit(X, y[i]).predict_proba(X)[:, 1]
        lg = np.log(p / (1 - p)); cal = .2 * lg.std(); cases = np.where(y[i] == 1)[0]; free = set(np.where(y[i] == 0)[0])
        for a in np.random.default_rng(seed).permutation(cases):
            if not free: break
            f = np.array(sorted(free)); j = f[np.argmin(np.abs(lg[f] - lg[a]))]
            if abs(lg[j] - lg[a]) <= cal: out += [i[a], i[j]]; free.discard(j)
    return np.array(out)


# 표준화 평균차(SMD): 두 군 평균 차이 / 합동 SD
smd = lambda x, y: (x[y == 1].mean() - x[y == 0].mean()) / np.sqrt((x[y == 1].var() + x[y == 0].var()) / 2)
Xfull = lambda d, n: (lambda Z: C4 if Z is None else (Z if n.startswith("CT만") else np.column_stack([C4, Z])))(SETS[n](d))
bal, P, E = [], [], []
for d in DIS:
    y = A[d].to_numpy(int); sx = {n: Xfull(d, n) for n in USE}
    Ms = [match(y, s) for s in range(SEEDS)]
    for lab_, idx in [("매칭 전", np.arange(len(y))), ("매칭 후(seed0)", Ms[0])]:
        for j, nm in enumerate(["나이", "성별(남)", "BMI"]): bal.append(dict(disease=d, stage=lab_, variable=nm, SMD=smd(COV[idx, j], y[idx]), n=len(idx), n_pos=int(y[idx].sum())))
    if min(int(y[M].sum()) for M in Ms) < 10: print(d, "매칭된 양성이 10명 미만이라 성능 평가 생략", [int(y[M].sum()) for M in Ms]); continue
    folds = {n: [] for n in USE}
    for s_, M in enumerate(Ms):
        ym, cm = y[M], coh[M]
        for k in range(REPS):
            for tr, te in SKF(5, shuffle=True, random_state=k).split(M, ym * 2 + (cm == "sinchon")):
                for n in USE: folds[n].append(auc(ym[te], fit_predict(sx[n][M], n, ym, tr, te)))
    base = np.array(folds["clinic4"])
    for n in USE:
        r = np.array(folds[n]); P.append(dict(disease=d, feature_set=n, n_feat=sx[n].shape[1], n_matched=int(np.mean([len(M) for M in Ms])), n_pos=int(np.mean([y[M].sum() for M in Ms])), AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(), p=np.nan if n == "clinic4" else nb_test(r - base)))
    for tr_c, te_c in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        aucs = {n: [] for n in USE}; p0 = {}
        for s_, M in enumerate(Ms):
            a, b = M[coh[M] == tr_c], M[coh[M] == te_c]
            sc = {n: fit_predict(sx[n], n, y, a, b) for n in USE}
            for n in USE:
                aucs[n].append(auc(y[b], sc[n]))
                if s_ == 0: p0[n] = delong(y[b], sc["clinic4"], sc[n])["p_value"] if n != "clinic4" else np.nan
        for n in USE: E.append(dict(train=tr_c, test=te_c, disease=d, feature_set=n, AUC=np.mean(aucs[n]), dAUC=np.mean(aucs[n]) - np.mean(aucs["clinic4"]), p=p0[n]))
    print(d, "완료", flush=True)
P, E, B = pd.DataFrame(P), pd.DataFrame(E), pd.DataFrame(bal)
if len(P):
    P["q"] = bh_fdr(P.p.fillna(1).to_numpy()); E["q"] = E.groupby("train").p.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
with pd.ExcelWriter(OUT / "clinic4_psm.xlsx") as w:
    B.to_excel(w, sheet_name="balance", index=False); P.to_excel(w, sheet_name="pooled_cv", index=False); E.to_excel(w, sheet_name="external", index=False)
pd.set_option("display.width", 200); print(B.round(3).to_string()); print(P.round(3).to_string()); print(E.round(3).to_string())
