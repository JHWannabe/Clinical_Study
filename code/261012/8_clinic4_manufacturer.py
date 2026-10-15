# 8단계: clinic4(+landmark 합/raw)에 CT 제조사(Siemens/GE/Philips/Canon/기타, manufacturer_model에서 분류) one-hot을 추가했을 때 AUC 변화
# 평가: 5-fold CV(풀링/강남/신촌, 3회 반복, Nadeau-Bengio 보정 t)와 강남↔신촌 외부검증(DeLong). 제조사 효과는 같은 영상 feature에서 제조사 유무(짝지은 fold)로도 비교
# 질환마다 질환군(동반질환 포함) vs 정상군(HTN·DM·CKD 모두 없음)만 사용 (0_data_distribution과 같은 정의). OUT_DIR로 저장 폴더 지정(그 아래 스크립트 이름 폴더에 저장)
import importlib.util, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from _sens import sens_spec, train_cutoff

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("s1", Path(__file__).with_name("1_clinic4_landmark_sum.py")); s1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
OUT = ROOT / os.environ.get("OUT_DIR", "outputs/261012") / Path(__file__).stem; OUT.mkdir(parents=True, exist_ok=True)
DIS, MEAS, LM, mk, nb_test, delong = s1.DIS, s1.MEAS, s1.LM, s1.mk, s1.nb_test, s1.delong_paired_auc_test
MFR = ["Siemens", "GE", "Philips", "Canon", "기타"]


# 장비 모델명 -> 제조사 (키워드 규칙)
def maker(m):
    m = str(m).lower()
    if any(k in m for k in ["somatom", "sensation", "definition", "emotion", "scope", "spirit", "perspective"]): return "Siemens"
    if any(k in m for k in ["revolution", "discovery", "lightspeed", "optima", "brightspeed", "brivo"]): return "GE"
    if any(k in m for k in ["ict", "ingenuity", "brilliance", "iqon", "mx 16"]): return "Philips"
    if any(k in m for k in ["aquilion", "alexion", "activion", "supria"]): return "Canon"
    return "기타"


# step1 load(합 4종 포함) + landmark 12곳 raw 값({m}@{lm}) + 제조사
def load(cohort):
    m = s1.load(cohort); path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID"); sl = lm.loc[m.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
    for t in MEAS:
        v = np.nan_to_num(pd.read_excel(path, sheet_name=s1.SHEET[t]).set_index("PatientID").filter(regex=rf"^{t.lower() if t == 'AEC' else t}_\d+$").loc[m.PatientID].to_numpy(float))
        for j, a in enumerate(LM): m[f"{t}@{a}"] = v[np.arange(len(m)), sl[:, j] - 1]
    m["maker"] = lm.loc[m.PatientID, "manufacturer_model"].map(maker).to_numpy()
    return m


def main():
    A = pd.concat([load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    print(pd.crosstab(A.cohort, A.maker)); pd.crosstab(A.cohort, A.maker).to_excel(OUT / "clinic4_step8_manufacturer_counts.xlsx")
    C4 = np.column_stack([A.sex, A.PatientAge, A.Height, A.Weight]); M = np.column_stack([(A.maker == k).astype(float) for k in MFR[:-1]])  # 기타가 기준
    SUM = np.column_stack([A[f"{t}_sum"] for t in MEAS]); RAW = np.column_stack([A[f"{t}@{a}"] for t in MEAS for a in LM])
    SETS = {"clinic4": C4, "+제조사": np.column_stack([C4, M]), "+합 4종": np.column_stack([C4, SUM]), "+합 4종 +제조사": np.column_stack([C4, SUM, M]),
            "+LM raw 전체 12곳": np.column_stack([C4, RAW]), "+LM raw 전체 12곳 +제조사": np.column_stack([C4, RAW, M])}
    PAIR = {"+제조사": "clinic4", "+합 4종 +제조사": "+합 4종", "+LM raw 전체 12곳 +제조사": "+LM raw 전체 12곳"}  # 제조사 유무 비교 쌍
    pd.DataFrame([(n, X.shape[1]) for n, X in SETS.items()], columns=["feature_set", "n_feat"]).to_excel(OUT / "clinic4_step8_features.xlsx", index=False)
    strat_c = (A.cohort == "sinchon").to_numpy()
    keep = {d: ((A[d] == 1) | (A[s1.ALL_DIS].sum(axis=1) == 0)).to_numpy() for d in DIS}
    subsets = {"pooled": np.ones(len(A), bool), "gangnam": (A.cohort == "gangnam").to_numpy(), "sinchon": strat_c}

    rows = []
    for ev, sub in subsets.items():
        for d in DIS:
            mm = sub & keep[d]; y = A[d].to_numpy(int)[mm]; r = {n: s1.cv(X[mm], y, y * 2 + strat_c[mm])[0] for n, X in SETS.items()}
            for n, X in SETS.items():
                ref = PAIR.get(n)
                rows.append(dict(eval=f"{ev} 5-fold CV", disease=d, feature_set=n, n=len(y), n_pos=y.sum(), n_feat=X.shape[1], AUC=r[n].mean(), Sens=r[n].ss[0], Spec=r[n].ss[1], AUC_sd=r[n].std(ddof=1),
                                 dAUC=(r[n] - r["clinic4"]).mean(), p_cv=np.nan if n == "clinic4" else nb_test(r[n] - r["clinic4"]),
                                 dAUC_vs_nomfr=(r[n] - r[ref]).mean() if ref else np.nan, p_vs_nomfr=nb_test(r[n] - r[ref]) if ref else np.nan))
    res = pd.DataFrame(rows)

    ext = []
    for a, b in [("gangnam", "sinchon"), ("sinchon", "gangnam")]:
        tr, te = (A.cohort == a).to_numpy(), (A.cohort == b).to_numpy()
        for d in DIS:
            y, tr_, te_ = A[d].to_numpy(int), tr & keep[d], te & keep[d]
            p = {n: s1.mk().fit(X[tr_], y[tr_]).predict_proba(X[te_])[:, 1] for n, X in SETS.items()}
            thr = {n: train_cutoff(s1.mk, X[tr_], y[tr_]) for n, X in SETS.items()}
            for n in SETS:
                ref = PAIR.get(n); t = delong(y[te_], p["clinic4"], p[n]) if n != "clinic4" else dict(diff=0, p_value=np.nan); t2 = delong(y[te_], p[ref], p[n]) if ref else dict(diff=np.nan, p_value=np.nan)
                ext.append(dict(train=a, test=b, disease=d, feature_set=n, AUC=roc_auc_score(y[te_], p[n]), **dict(zip(["Sens", "Spec"], sens_spec(y[te_], p[n], thr[n]))), dAUC=t["diff"], p_delong=t["p_value"], dAUC_vs_nomfr=t2["diff"], p_vs_nomfr=t2["p_value"]))
    ext = pd.DataFrame(ext)
    with pd.ExcelWriter(OUT / "clinic4_step8.xlsx") as xw: res.to_excel(xw, sheet_name="cv", index=False); ext.to_excel(xw, sheet_name="external", index=False)
    pd.set_option("display.width", 250)
    print(res[res["eval"] == "pooled 5-fold CV"].round(3).to_string()); print(ext.round(3).to_string())


if __name__ == "__main__":
    main()
