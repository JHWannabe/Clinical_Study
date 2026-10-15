from __future__ import annotations

# 결론 슬라이드의 '다음 단계' 3가지를 실행 (단독 실행, outputs/261012/clinic4_next_steps.xlsx 저장)
#  ① 병원별 표준화: 추가 feature 열을 코호트 안에서 z-score(라벨 미사용) 후 풀링 CV·외부검증
#  ② 성별 층화: 남/여를 따로 모델링(feature는 코호트x성별 안에서 z-score), 풀링 CV·외부검증
#  ③ 병원 효과 보정: 병원 더미를 넣은 풀링 CV, 병원 고정효과 + 장비(manufacturer_model) 군집 로버스트 로지스틱 OR와 feature x 병원 교호작용 p
#    (병원이 2곳뿐이라 병원 임의효과는 추정 불가 -> 고정효과, 장비 군집으로 상관 보정)
# 데이터 적재·feature 조합·모델(nested LR)·CV/검정 함수는 clinic4_feature_combos.py의 앞부분(분석 실행 전까지)을 그대로 재사용한다.
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
src = (HERE / "clinic4_feature_combos.py").read_text(encoding="utf-8").split("rows = []\nfor d in DIS:")[0]
g = {"__file__": str(HERE / "clinic4_feature_combos.py")}
exec(compile(src, "clinic4_feature_combos_head", "exec"), g)
A, C4, SETS, DIS, REPS, OUT = g["A"], g["C4"], g["SETS"], g["DIS"], g["REPS"], g["OUT"]
fit_predict, nb_test, bh_fdr, delong, auc, SKF, mk = g["fit_predict"], g["nb_test"], g["bh_fdr"], g["delong_paired_auc_test"], g["roc_auc_score"], g["StratifiedKFold"], g["mk"]
USE = ["clinic4", "(참고) +LM raw(L1,L3,L5)", "+분포 기반(질환별)", "+구간 합산(4종)"]
coh = A.cohort.to_numpy(); male = A.sex.to_numpy() == 1


# 열 단위 z-score를 그룹(코호트 등)별로 적용 (평균·SD는 해당 그룹 전체 환자에서 계산, 라벨 미사용)
def zgroup(Z, grp):
    Z = Z.astype(float).copy()
    for v in np.unique(grp):
        m = grp == v; sd = Z[m].std(0); Z[m] = (Z[m] - Z[m].mean(0)) / np.where(sd == 0, 1, sd)
    return Z


# feature set 행렬: clinic4(4열) + (선택) 추가 열 변환
def build(d, name, zf=None, idx=None):
    Z = SETS[name](d)
    if Z is not None and zf is not None: Z = zf(np.asarray(Z))
    X = C4 if Z is None else np.column_stack([C4, Z])
    return X if idx is None else X[idx]


# 풀링 CV: fold별 AUC 배열(반복 k는 모든 set에서 같은 분할). strat = 라벨 x 코호트
def pooled_cv(X, y, c, name):
    r = []
    for k in range(REPS):
        for tr, te in SKF(5, shuffle=True, random_state=k).split(X, y * 2 + (c == "sinchon")):
            r.append(auc(y[te], fit_predict(X, name, y, tr, te)))
    return np.array(r)


def run_pooled(label, sets, d, y, c, getX):
    out, base = [], None
    for name in sets:
        r = pooled_cv(getX(name), y, c, name)
        if name == "clinic4": base = r
        out.append(dict(setting=label, disease=d, feature_set=name, AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(), p=np.nan if name == "clinic4" else nb_test(r - base)))
    return out


def run_ext(label, sets, d, y, c, getX):
    out = []
    for tr_c, te_c in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        a, b = np.where(c == tr_c)[0], np.where(c == te_c)[0]
        if y[a].sum() < 5 or y[b].sum() < 5: continue
        sc = {n: fit_predict(getX(n), n, y, a, b) for n in sets}
        for n in sets:
            t = delong(y[b], sc["clinic4"], sc[n]) if n != "clinic4" else {"diff": 0, "p_value": np.nan}
            out.append(dict(setting=label, train=tr_c, test=te_c, disease=d, feature_set=n, AUC=auc(y[b], sc[n]), dAUC=t["diff"], p=t["p_value"]))
    return out


def fdr(df, by):
    df = df.copy(); df["q"] = df.groupby(by).p.transform(lambda p: bh_fdr(p.fillna(1).to_numpy())); return df


P, E = [], []
for d in DIS:
    y = A[d].to_numpy(int)
    # ① 병원별 표준화
    zc = lambda Z: zgroup(Z, coh)
    P += run_pooled("① 병원별 표준화", USE, d, y, coh, lambda n: build(d, n, zc))
    E += run_ext("① 병원별 표준화", USE, d, y, coh, lambda n: build(d, n, zc))
    # ② 성별 층화: 성별 열 제외(age, height, weight 3열), feature는 코호트x성별 안에서 z-score
    for sx, nm in [(True, "남"), (False, "여")]:
        idx = np.where(male == sx)[0]; ys, cs = y[idx], coh[idx]
        if ys.sum() < 10 or (ys == 0).sum() < 10: continue  # 양성이 너무 적은 성별은 제외
        zs = lambda Z: zgroup(Z, np.char.add(coh.astype(str), male.astype(str)))  # 전체 환자를 코호트x성별로 z-score한 뒤 해당 성별만 사용
        gx = lambda n, idx=idx: build(d, n, zs, idx)[:, 1:]
        # fit_predict는 4열 가정이 없고(선택형 제외) 행렬만 쓰므로 3열 clinic4로 평가
        P += [dict(r, setting=f"② 성별 층화({nm}, n={len(idx)}, 양성 {int(ys.sum())})") for r in run_pooled("", USE, d, ys, cs, gx)]
        E += [dict(r, setting=f"② 성별 층화({nm})") for r in run_ext("", USE, d, ys, cs, gx)]
    # ③ 병원 더미를 넣은 풀링 CV
    dum = (coh == "sinchon").astype(float)[:, None]
    P += run_pooled("③ 병원 더미 포함", USE, d, y, coh, lambda n: np.column_stack([build(d, n), dum]))
P = pd.DataFrame(P); E = pd.DataFrame(E)
P = fdr(P, ["setting", "disease"]); E = fdr(E, ["setting", "train", "disease"])

# ③ 병원 고정효과 + 장비 군집 로버스트 로지스틱: feature를 풀링 z-score로 하나씩, clinic4 + 병원 보정
lm = pd.concat([pd.read_excel(g["ROOT"] / f"data/{c}_landmark_filtered.xlsx", sheet_name="landmarks").set_index("PatientID").manufacturer_model for c in ["gangnam", "sinchon"]])
mm = lm.loc[A.PatientID].to_numpy(); vc = pd.Series(mm).value_counts(); sc_grp = np.where(pd.Series(mm).map(vc).to_numpy() >= 100, mm, "other")
grp = pd.factorize(pd.Series(coh).str.cat(pd.Series(sc_grp), sep="|"))[0]  # 병원 x 장비 군집
R = []
for d in DIS:
    y = A[d].to_numpy(int)
    for t in ["VAT_sum", "SAT_sum", "TAMA_sum", "AEC_sum"]:
        z = (A[t] - A[t].mean()) / A[t].std()
        X = sm.add_constant(np.column_stack([C4, (coh == "sinchon"), z, z * (coh == "sinchon")]).astype(float))
        X0 = X[:, :-1]  # 교호작용 제외 모델
        f0 = sm.Logit(y, X0).fit(disp=0, cov_type="cluster", cov_kwds={"groups": grp}); f1 = sm.Logit(y, X).fit(disp=0)
        j = X0.shape[1] - 1; ci = np.exp(f0.conf_int()[j])
        R.append(dict(disease=d, feature=t, OR_adj=np.exp(f0.params[j]), ci_lo=ci[0], ci_hi=ci[1], p_adj=f0.pvalues[j], p_interaction=f1.pvalues[-1],
                      OR_gangnam=np.exp(f1.params[j]), OR_sinchon=np.exp(f1.params[j] + f1.params[-1])))
R = pd.DataFrame(R)
with pd.ExcelWriter(OUT / "clinic4_next_steps.xlsx") as w:
    P.to_excel(w, sheet_name="pooled_cv", index=False); E.to_excel(w, sheet_name="external", index=False); R.to_excel(w, sheet_name="cohort_adjusted_or", index=False)
pd.set_option("display.width", 220, "display.max_rows", 200)
print(P.round(3).to_string()); print(E.round(3).to_string()); print(R.round(3).to_string())
