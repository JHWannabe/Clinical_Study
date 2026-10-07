from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark_combo_features.py의 조합 세트(clinical + landmark 지표)를 로지스틱 대신 MLP로 평가하고 같은 프로토콜로 LR과 비교한다.
#  프로토콜: Gangnam 5-fold CV AUC / Gangnam 학습 -> Sinchon 외부 AUC. MLP 하이퍼파라미터는 학습 데이터 안의 inner 3-fold CV(AUC)로만 선택
#  (hidden {(16,), (32,16)} x alpha {0.01, 1}), 시드 고정. early_stopping은 끈다: 양성이 적은 DM(20%)/CKD(9%)에서 내부 검증 분할이
#  너무 작아 학습이 조기 중단되어 clinical만으로도 AUC가 0.49~0.58로 붕괴함을 확인(진단: early_stopping on -> CKD cv 0.46, off -> 0.76). DeLong p = 같은 Sinchon 환자에서 MLP vs LR AUC 비교.
#  근거: ML과 LR의 AUC 차이는 편향 위험이 낮은 비교에서 0.00 (J Clin Epidemiol 2019 doi:10.1016/j.jclinepi.2019.02.004) -> 가설은 "MLP도 LR 대비 이득 없음"
# 출력: outputs/data_distribution/landmark_mlp_sets.xlsx
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import LM, load_cohort
from delong_utils import delong_paired_auc_test

warnings.simplefilter("ignore", ConvergenceWarning)
sys.stdout.reconfigure(encoding="utf-8")
D = "outputs/data_distribution"
DIS = ["HTN", "DM", "CKD"]
TIS = ["VAT", "SAT", "TAMA", "AEC"]
LMN = [a.replace("_center", "") for a in LM]
lr_ = lambda a, b: np.log((a + 1) / (b + 1))
SEED = 0


# 코호트 로드: 임상 변수 + 조직@landmark 단면 값(AEC 포함) (landmark_combo_features.load와 같은 정의)
def load(c):
    df = load_cohort(c)[0].copy()
    path = f"data/{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[df.PatientID]
    aec = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$").loc[df.PatientID].to_numpy(float)
    for a, n in zip(LM, LMN):
        df[f"AEC@{n}"] = np.nan_to_num(aec[np.arange(len(df)), lm[f"{a}_slice"].to_numpy(int) - 1])
        for t in TIS[:3]:
            df[f"{t}@{n}"] = df[f"{t}@{a}"]
    return df


def tier1(df):  # 같은 landmark 안의 조직 비율
    f = {}
    for n in LMN:
        if n != "inferior_pubic_margin":
            v, s, t = df[f"VAT@{n}"], df[f"SAT@{n}"], df[f"TAMA@{n}"]
            f[f"VAT/SAT@{n}"], f[f"VAT/TAMA@{n}"], f[f"TAMA/(VAT+SAT)@{n}"] = lr_(v, s), lr_(v, t), lr_(t, v + s)
    return pd.DataFrame(f)


def sets(df):
    t1 = tier1(df)
    S = {"clinical": [], "VAT@L3": ["VAT@L3"], "VAT,SAT,TAMA@L3": [f"{t}@L3" for t in TIS[:3]],
         "VAT,SAT,TAMA@all landmarks": [f"{t}@{a}" for t in TIS[:3] for a in LMN if (t, a) != ("VAT", "inferior_pubic_margin")],
         "Tier1 ratios@L3": ["VAT/SAT@L3", "VAT/TAMA@L3", "TAMA/(VAT+SAT)@L3"], "Tier1 ratios@all landmarks": list(t1.columns),
         "VAT,SAT,TAMA,AEC@L3 (AEC 탐색적)": [f"{t}@L3" for t in TIS]}
    base = df[["PatientAge", "BMI"]].assign(male=(df.PatientSex == "M").astype(float)).to_numpy(float)
    return {k: np.column_stack([base, df[[c for c in cols if c in df.columns]].to_numpy(float), t1[[c for c in cols if c in t1.columns]].to_numpy(float)])
            for k, cols in S.items()}


def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def mlp_model():
    gs = GridSearchCV(MLPClassifier(max_iter=500, early_stopping=False, random_state=SEED),
                      {"hidden_layer_sizes": [(16,), (32, 16)], "alpha": [0.01, 1.0]}, scoring="roc_auc", cv=3)
    return make_pipeline(StandardScaler(), gs)


def pred(make, Xtr, ytr, Xte):
    return make().fit(Xtr, ytr).predict_proba(Xte)[:, 1]


Sg, Ss = sets(load("gangnam")), sets(load("sinchon"))
dg, ds = load("gangnam"), load("sinchon")
if __name__ == "__main__":  # import 시 전체 분석이 도는 것 방지
    rows = []
    for d in DIS:
        yg, ys = dg[d].to_numpy(int), ds[d].to_numpy(int)
        for k in Sg:
            cv = {"LR": [], "MLP": []}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Sg[k], yg):
                for nm, mk in (("LR", lr_model), ("MLP", mlp_model)):
                    cv[nm].append(roc_auc_score(yg[te], pred(mk, Sg[k][tr], yg[tr], Sg[k][te])))
            p_lr, p_mlp = pred(lr_model, Sg[k], yg, Ss[k]), pred(mlp_model, Sg[k], yg, Ss[k])
            dl = delong_paired_auc_test(ys, p_lr, p_mlp)
            rows.append(dict(disease=d, set=k, n_feat=Sg[k].shape[1], cv_LR=np.mean(cv["LR"]), cv_MLP=np.mean(cv["MLP"]), ext_LR=dl["auc_a"], ext_MLP=dl["auc_b"],
                             ext_MLP_minus_LR=dl["diff"], delong_p=dl["p_value"]))
            print(d, k, round(rows[-1]["cv_LR"], 3), round(rows[-1]["cv_MLP"], 3), round(dl["auc_a"], 3), round(dl["auc_b"], 3), flush=True)
    r = pd.DataFrame(rows)
    r.round(4).to_excel(f"{D}/landmark_mlp_sets.xlsx", index=False)
    print(r.round(3).to_string(index=False))
    print("MLP-LR 외부 AUC 평균 차이:", round(r.ext_MLP_minus_LR.mean(), 4), "| MLP가 유의하게 높은 세트:", int(((r.ext_MLP_minus_LR > 0) & (r.delong_p < .05)).sum()),
          "| 낮은 세트:", int(((r.ext_MLP_minus_LR < 0) & (r.delong_p < .05)).sum()))
