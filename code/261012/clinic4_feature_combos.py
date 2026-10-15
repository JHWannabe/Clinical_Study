from __future__ import annotations

# data_distribution 결과(질환군 vs 정상군 landmark 분포 차이)를 참고한 feature 조합을 clinic4에 추가해 풀링 nested 5-fold CV(fold별 AUC 평균±SD, Nadeau-Bengio 보정 t)와 강남↔신촌 외부검증으로 비교 (단독 실행, outputs/261012 저장)
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from delong_utils import delong_paired_auc_test, bh_fdr
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/261012"
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
        g = np.gradient(v, axis=1)  # slice 방향 기울기(중심차분)
        for j, a in enumerate(LM): feat[f"{m}@{a}"] = v[np.arange(len(v)), sl[:, j] - 1]; feat[f"{m}'@{a}"] = g[np.arange(len(v)), sl[:, j] - 1]
        feat[f"{m}_mean"] = [v[i, sl[i, 0] - 1:sl[i, -1]].mean() for i in range(len(v))]
        feat[f"{m}_sum"] = [v[i, sl[i, 0] - 1:sl[i, -1]].sum() for i in range(len(v))]  # liver dome~pubis 구간 합산(총량)
    meta["sex"] = (meta.PatientSex.astype(str).str.upper() == "M").astype(float); meta["cohort"] = cohort
    return pd.concat([meta, pd.DataFrame(feat)], axis=1)


A = pd.concat([load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
C4 = np.column_stack([A.sex, A.PatientAge, A.Height, A.Weight])
LMS3 = ["L1_center", "L3_center", "L5_center"]
raw = lambda ms, lms=LM: np.column_stack([A[f"{m}@{a}"] for m in ms for a in lms])  # landmark 값
pw = lambda ms, lms=LM: np.column_stack([A[f"{m}@{a}"] / (A[f"{m}_mean"] + 1e-6) for m in ms for a in lms] + [A[f"{m}_mean"] for m in ms])  # 환자 구간평균으로 나눈 곡선 모양 + 평균(진폭)
# 분포 분석에서 두 코호트 모두 일관되게 유의했던 조합: VAT raw(HTN/DM), SAT·AEC pw_mean(HTN/DM), AEC pw_mean(CKD 포함 3질환)
slope = lambda ms, lms=LM: np.column_stack([A[f"{m}'@{a}"] for m in ms for a in lms])  # 기울기 raw
slope_pw = lambda ms, lms=LM: np.column_stack([A[f"{m}'@{a}"] / (A[f"{m}_mean"] + 1e-6) for m in ms for a in lms])  # 기울기 / 환자 구간평균
guided = {"HTN": np.column_stack([raw(["VAT"]), pw(["SAT", "AEC"])]), "DM": np.column_stack([raw(["VAT"]), pw(["SAT", "AEC"])]), "CKD": pw(["AEC"])}
tot = lambda ms, k: np.column_stack([A[f"{m}_{k}"] for m in ms])  # 구간 합산/평균 4종
SETS = {"clinic4": lambda d: None,
        "+구간 합산(4종)": lambda d: tot(MEAS, "sum"), "+구간 평균(4종)": lambda d: tot(MEAS, "mean"),
        "+구간 합산 + 조직 비율": lambda d: np.column_stack([tot(MEAS, "sum"), A.VAT_sum / (A.SAT_sum + 1e-6), A.VAT_sum / (A.TAMA_sum + 1e-6), A.TAMA_sum / (A.Weight / 100)]),
        "+VAT raw(12)": lambda d: raw(["VAT"]), "+AEC pw(12)": lambda d: pw(["AEC"]), "+SAT pw(12)": lambda d: pw(["SAT"]),
        "+VAT raw+AEC pw": lambda d: np.column_stack([raw(["VAT"]), pw(["AEC"])]),
        "+VAT raw+SAT pw+AEC pw": lambda d: np.column_stack([raw(["VAT"]), pw(["SAT", "AEC"])]),
        "+VAT/SAT/AEC pw": lambda d: pw(["VAT", "SAT", "AEC"]),
        "+분포 기반(질환별)": lambda d: guided[d],
        "+분포 기반 + 4종 L1,L3,L5 raw": lambda d: np.column_stack([guided[d], raw(MEAS, LMS3)]),
        "+in-fold 유의 선택(raw+pw 4종)": lambda d: np.column_stack([raw(MEAS), pw(MEAS)]),
        "+기울기 raw(4종)": lambda d: slope(MEAS), "+기울기 pw(4종)": lambda d: slope_pw(MEAS),
        "+분포 기반 + 기울기 pw(VAT/SAT/AEC)": lambda d: np.column_stack([guided[d], slope_pw(["VAT", "SAT", "AEC"])]),
        "+in-fold 유의 선택(raw+pw+기울기 8종)": lambda d: np.column_stack([raw(MEAS), pw(MEAS), slope(MEAS), slope_pw(MEAS)]),
        "(참고) +LM raw(L1,L3,L5)": lambda d: raw(MEAS, LMS3), "(참고) +LM/mean(L1,L3,L5)": lambda d: pw(MEAS, LMS3),
        # CT만: clinic4(성별·나이·키·몸무게) 없이 영상 변수만 입력
        "CT만 구간 합산(4종)": lambda d: tot(MEAS, "sum"), "CT만 구간 평균(4종)": lambda d: tot(MEAS, "mean"),
        "CT만 LM raw(L1,L3,L5)": lambda d: raw(MEAS, LMS3), "CT만 LM/mean(L1,L3,L5)": lambda d: pw(MEAS, LMS3),
        "CT만 VAT/SAT/AEC pw(12)": lambda d: pw(["VAT", "SAT", "AEC"]), "CT만 분포 기반(질환별)": lambda d: guided[d],
        "CT만 raw+pw 4종 전체": lambda d: np.column_stack([raw(MEAS), pw(MEAS)])}
SELECT = {"+in-fold 유의 선택(raw+pw 4종)", "+in-fold 유의 선택(raw+pw+기울기 8종)"}  # 학습 fold의 Mann-Whitney BH-FDR<0.05 feature만 사용


# LR(L2)의 C를 학습 fold 내부 5-fold AUC로 고르는 모델
def mk():
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), {"logisticregression__C": np.logspace(-3, 1, 9)}, cv=5, scoring="roc_auc", n_jobs=-1)


# 학습 fold에서 질환군 vs 정상군 Mann-Whitney BH-FDR<0.05인 후보 열 마스크 (평가 fold 정보는 쓰지 않음)
def sig_mask(Z, y):
    p = np.nan_to_num(mannwhitneyu(Z[y == 1], Z[y == 0], axis=0).pvalue, nan=1.0)
    return bh_fdr(p) < .05


# train/test 인덱스로 feature 행렬을 만들어 학습·예측 (선택형이면 clinic4 4열 + 유의 후보 열)
def fit_predict(X, name, y, tr, te):
    if name in SELECT:
        Z = X[:, 4:]; m = sig_mask(Z[tr], y[tr]); X = np.column_stack([X[:, :4], Z[:, m]])
    return mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


# Youden 지수(민감도+특이도-1)가 최대인 cutoff와 그 cutoff에서의 (민감도, 특이도)
def youden(y, p):
    fpr, tpr, th = roc_curve(y, p); return th[np.argmax(tpr - fpr)]


def sens_spec(y, p, t):
    q = p >= t; return q[y == 1].mean(), (~q[y == 0]).mean()


# 풀링 5-fold CV를 REPS회 반복한 fold별 AUC 배열(REPS*5개, 반복 k는 모든 feature set에서 같은 분할)과 반복별 (민감도, 특이도)(OOF 예측 전체의 Youden cutoff)
def cv(X, name, y):
    r, ss = [], []
    for k in range(REPS):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, y * 2 + (A.cohort == "sinchon").to_numpy()):
            oof[te] = fit_predict(X, name, y, tr, te); r.append(roc_auc_score(y[te], oof[te]))
        ss.append(sens_spec(y, oof, youden(y, oof)))
    return np.array(r), np.mean(ss, axis=0)


# Nadeau-Bengio 보정 paired t 검정: fold별 AUC 차이(J=반복x5개)의 평균이 0인지 (fold 간 학습셋 중복에 의한 분산 과소추정 보정, 시험/학습 = 1/4)
def nb_test(diff):
    J, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else float(2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / J + .25) * v), J - 1))


rows = []
for d in DIS:
    y = A[d].to_numpy(int); base = None
    for name, f in SETS.items():
        Z = f(d); X = C4 if Z is None else (Z if name.startswith("CT만") else np.column_stack([C4, Z]))
        r, (se, sp) = cv(X, name, y)
        if name == "clinic4": base = r
        rows.append(dict(disease=d, feature_set=name, n_feat=X.shape[1], AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(),
                         p_cv=np.nan if name == "clinic4" else nb_test(r - base), Sens=se, Spec=sp))
        print(d, name, round(r.mean(), 3), flush=True)
res = pd.DataFrame(rows)
res["p_fdr"] = res.groupby("disease").p_cv.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
res.to_excel(OUT / "clinic4_feature_combos.xlsx", index=False)

# 코호트 간 외부검증 (선택형은 학습 코호트에서만 feature 선택)
ext = []
for tr_c, te_c in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
    a, b = np.where((A.cohort == tr_c).to_numpy())[0], np.where((A.cohort == te_c).to_numpy())[0]
    for d in DIS:
        y = A[d].to_numpy(int); sc, cut = {}, {}
        for name, f in SETS.items():
            Z = f(d); X = C4 if Z is None else (Z if name.startswith("CT만") else np.column_stack([C4, Z]))
            sc[name] = fit_predict(X, name, y, a, b)
            oof = np.zeros(len(y))  # 학습 코호트 5-fold OOF로 Youden cutoff 결정(시험 코호트 미사용)
            for t_, v_ in StratifiedKFold(5, shuffle=True, random_state=0).split(a, y[a]): oof[a[v_]] = fit_predict(X, name, y, a[t_], a[v_])
            cut[name] = youden(y[a], oof[a])
        for name in SETS:
            t = delong_paired_auc_test(y[b], sc["clinic4"], sc[name]) if name != "clinic4" else {"diff": 0, "p_value": np.nan}
            ext.append(dict(train=tr_c, test=te_c, disease=d, feature_set=name, AUC=roc_auc_score(y[b], sc[name]), dAUC=t["diff"], p_delong=t["p_value"], **dict(zip(["Sens", "Spec"], sens_spec(y[b], sc[name], cut[name])))))
ext = pd.DataFrame(ext)
ext["p_fdr"] = ext.groupby(["train", "disease"]).p_delong.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
ext.to_excel(OUT / "clinic4_feature_combos_external.xlsx", index=False)
fm = lambda g: g.assign(s=lambda x: x.AUC.round(3).astype(str) + " (" + x.dAUC.round(3).map("{:+.3f}".format) + ", q" + x.p_fdr.round(3).astype(str) + ")").pivot(index="feature_set", columns="disease", values="s")[DIS].loc[list(SETS)]
print("\n[pooled]"); print(fm(res).to_string())
for tr_c, g in ext.groupby("train"): print(f"\n[train {tr_c}]"); print(fm(g).to_string())
