from __future__ import annotations

# 5년 내 발병(onset_within_5y) 예후 분류: clinic4 baseline 대비 landmark 체성분 feature 추가 효과를 1:1 언더샘플링 x 반복 5-fold CV로 비교 (단독 실행, outputs/261012/prognosis_auc.xlsx 저장)
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/261012"
PATH = ROOT / "data/prognosis_5y_landmark_filtered.xlsx"
DIS = {"DM": "dm", "HTN": "htn", "DL": "dl", "OSP": "osteoporosis", "MI": "mi", "STROKE": "stroke"}
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
SHEET = {m: "aec_total" if m == "AEC" else f"{m}_total" for m in MEAS}
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
REPS = 100
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))


# landmark 시트와 곡선 시트에서 환자별 landmark 값({m}@{lm}), 구간 평균/합산({m}_mean/_sum) 표
def features(ids: pd.Index, lm: pd.DataFrame, cur: dict) -> pd.DataFrame:
    feat = {}
    sl = lm.loc[ids, [f"{a}_slice" for a in LM]].to_numpy(int)
    for m in MEAS:
        v = np.nan_to_num(cur[m].loc[ids].to_numpy(float))
        for j, a in enumerate(LM): feat[f"{m}@{a}"] = v[np.arange(len(v)), np.clip(sl[:, j] - 1, 0, v.shape[1] - 1)]
        feat[f"{m}_mean"] = [v[i, sl[i, 0] - 1:sl[i, -1]].mean() for i in range(len(v))]
        feat[f"{m}_sum"] = [v[i, sl[i, 0] - 1:sl[i, -1]].sum() for i in range(len(v))]
    return pd.DataFrame(feat, index=ids)


X_ = pd.ExcelFile(PATH)
lm = X_.parse("landmarks").set_index("PatientID")
cur = {m: X_.parse(SHEET[m]).set_index("PatientID").filter(regex=rf"^{m.lower() if m == 'AEC' else m}_\d+$") for m in MEAS}
rows, desc = [], []
for dn, sh in DIS.items():
    meta = X_.parse(f"patient_metadata_{sh}")
    meta = meta[(meta.hw_implausible == 0) & meta.PatientID.isin(lm.index) & meta.PatientID.isin(cur["AEC"].index)].dropna(subset=["PatientSex", "PatientAge", "Height", "Weight", "onset_within_5y"]).reset_index(drop=True)
    F = features(pd.Index(meta.PatientID), lm, cur).reset_index(drop=True)
    for m in MEAS:
        for a in LM: F[f"{m}pw@{a}"] = F[f"{m}@{a}"] / (F[f"{m}_mean"] + 1e-6)
    lr = lambda a, b: np.log((a + 1) / (b + 1))  # landmark_combo_features.py와 같은 로그 비율
    for a in LM[:-1]:  # inferior_pubic_margin 제외
        F[f"VAT/SAT@{a}"], F[f"VAT/TAMA@{a}"], F[f"TAMA/(VAT+SAT)@{a}"] = lr(F[f"VAT@{a}"], F[f"SAT@{a}"]), lr(F[f"VAT@{a}"], F[f"TAMA@{a}"]), lr(F[f"TAMA@{a}"], F[f"VAT@{a}"] + F[f"SAT@{a}"])
    R = ["VAT/SAT", "VAT/TAMA", "TAMA/(VAT+SAT)"]
    C4 = np.column_stack([(meta.PatientSex.astype(str).str.upper() == "M").astype(float), meta.PatientAge, meta.Height, meta.Weight])
    tot = [f"{m}_{k}" for m in MEAS for k in ("mean", "sum")]
    sets = {"clinic4": C4, "+구간 합산·평균(8)": np.column_stack([C4, F[tot]]),
            "+VAT/SAT/AEC pw(36+)": np.column_stack([C4, F[[f"{m}pw@{a}" for m in ("VAT", "SAT", "AEC") for a in LM] + [f"{m}_mean" for m in ("VAT", "SAT", "AEC")]]]),
            "+LM raw(L1,L3,L5)": np.column_stack([C4, F[[f"{m}@{a}" for m in MEAS for a in ("L1_center", "L3_center", "L5_center")]]]),
            "+Tier1 비율@L3(3)": np.column_stack([C4, F[[f"{r}@L3_center" for r in R]]]),
            "+Tier1 비율@전체LM(33)": np.column_stack([C4, F[[f"{r}@{a}" for r in R for a in LM[:-1]]]]),
            "+raw+pw+합산(전체)": np.column_stack([C4, F[[f"{m}@{a}" for m in MEAS for a in LM] + [f"{m}pw@{a}" for m in MEAS for a in LM] + tot]])}
    y = meta.onset_within_5y.to_numpy(int); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; k = min(len(pos), len(neg))
    desc.append(dict(disease=dn, n=len(y), n_pos=len(pos), n_neg=len(neg), n_balanced=2 * k))
    res = {s: [] for s in sets}
    for sd in range(REPS):
        rng = np.random.default_rng(sd); idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)]
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(idx, y[idx]):
            for s, X in sets.items():
                res[s].append(roc_auc_score(y[idx][te], LR().fit(X[idx][tr], y[idx][tr]).predict_proba(X[idx][te])[:, 1]))
    b = np.array(res["clinic4"]).reshape(REPS, 5).mean(1)
    for s in sets:
        a = np.array(res[s]).reshape(REPS, 5).mean(1)
        rows.append(dict(disease=dn, feature_set=s, n_feat=sets[s].shape[1], AUC=a.mean(), AUC_sd_rep=a.std(ddof=1), dAUC=(a - b).mean(), win_rate=(a > b).mean()))
    print(dn, {r["feature_set"]: round(r["AUC"], 3) for r in rows if r["disease"] == dn}, flush=True)
with pd.ExcelWriter(OUT / "prognosis_auc.xlsx") as w:
    pd.DataFrame(rows).to_excel(w, sheet_name="auc", index=False); pd.DataFrame(desc).to_excel(w, sheet_name="cohort", index=False)
