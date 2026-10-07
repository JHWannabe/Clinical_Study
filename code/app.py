from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parent.iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 실시간 모델 실험 페이지: input feature를 고르면 Gangnam hold-out AUC / Sinchon 외부 AUC / clinical 대비 DeLong p를 바로 보여준다
# 실행: streamlit run code/app.py (프로젝트 루트에서)
import io
import itertools
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch와 sklearn/lightgbm의 OpenMP 중복 로드 경고 회피 (Windows)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import contextlib
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from clinic4_landmark_vat_auc import DATA_DIR, DISEASES, LM, load_cohort
from delong_utils import bh_fdr, delong_paired_auc_test
from scipy.stats import mannwhitneyu

LMN = [a.replace("_center", "") for a in LM]
TIS = ["VAT", "SAT", "TAMA", "AEC"]
SEED = 111  # split seed 고정 (UI에서 변경 불가)
CLIN_BASE = ["Sex(M)", "PatientAge", "Height", "Weight"]  # baseline = 성별·나이·키·체중 (clinic4와 동일)
CLIN_ALL = ["Sex(M)", "PatientAge", "Height", "Weight"]  # BMI는 모델 입력에서 제외 (키·체중과 중복, 데이터 탭 설명 통계에는 표시)
# 질환별 기본 feature: Gangnam train으로 학습한 로지스틱(C=1)의 valid AUC만으로 forward selection(후보 약 330개, 최대 6개, 향상 0.005 미만이면 중단)한 결과.
# internal test와 external은 선택에 쓰지 않고, 선택된 feature를 고정한 뒤 한 번만 평가한다 (split seed 111, 정상 vs 질환 단독)
DEFAULT_FEATS = {
    "HTN": ["VAT@S1", "VAT@T11/VAT@L5", "AEC@L5/AEC@S1", "AEC@L3/AEC@inferior_pubic_margin", "TAMA/(VAT+SAT)@L2", "SAT@femoral_head/SAT@inferior_pubic_margin"],
    "DM": ["SAT@T10/SAT@T11", "TAMA@L4/TAMA@femoral_head", "AEC@T12/AEC@L1", "VAT@T10/VAT@T12"],
    "CKD": ["SAT@T11/SAT@L2", "SAT@S1/SAT@femoral_head", "SAT@T12/SAT@L3"],
}


t2_name = lambda t, a, b: f"{t}@{a}/{t}@{b}"
lr_ = lambda a, b: np.log((a + 1) / (b + 1))  # 비율 feature는 전부 log((a+1)/(b+1))


# 코호트 로드: 임상 변수 + 조직@landmark 단면 값(AEC 포함) + Tier1·Tier2 비율
@st.cache_data(show_spinner="데이터 로딩 중 (최초 1회)...")
def load(c: str) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_cohort(c)[0].copy()
    path = DATA_DIR / f"{c}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[df.PatientID]
    aec = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex=r"^aec_\d+$").loc[df.PatientID].to_numpy(float)
    assert df.PatientID.is_unique, "환자 중복: 행 단위 분할은 누수 위험"
    df["Sex(M)"] = (df.PatientSex.astype(str).str.upper() == "M").astype(float)
    for a, n in zip(LM, LMN):
        df[f"AEC@{n}"] = aec[np.arange(len(df)), lm[f"{a}_slice"].to_numpy(int) - 1]  # 결측은 NaN 유지, 파이프라인에서 train 중앙값으로 대치
        for t in TIS[:3]:
            df[f"{t}@{n}"] = df[f"{t}@{a}"]
    for n in LMN[:-1]:  # Tier1: 같은 landmark 안의 조직 비율 (VAT@pubis는 값이 없어 제외)
        v, s, t = df[f"VAT@{n}"], df[f"SAT@{n}"], df[f"TAMA@{n}"]
        df[f"VAT/SAT@{n}"], df[f"VAT/TAMA@{n}"], df[f"TAMA/(VAT+SAT)@{n}"] = lr_(v, s), lr_(v, t), lr_(t, v + s)
    t2 = {}  # Tier2: 같은 조직의 서로 다른 landmark 간 로그 비 (순서 무관이라 i<j만, VAT@pubis 제외)
    for t in TIS:
        for i, j in itertools.combinations(range(len(LMN)), 2):
            if t == "VAT" and LMN[-1] in (LMN[i], LMN[j]):
                continue
            t2[t2_name(t, LMN[i], LMN[j])] = lr_(df[f"{t}@{LMN[i]}"], df[f"{t}@{LMN[j]}"])
    return pd.concat([df, pd.DataFrame(t2, index=df.index)], axis=1)


# 비교 설계: 정상(HTN·DM·CKD 모두 없음) vs 선택 질환 "단독"(나머지 두 질환은 없음). 복합 질환자·다른 질환 단독은 제외
def apply_pop(df: pd.DataFrame, disease: str) -> pd.DataFrame:
    others = [k for k in DISEASES if k != disease]
    only = (df[disease] == 1) & (df[others] == 0).all(axis=1)
    normal = (df[DISEASES] == 0).all(axis=1)
    return df[only | normal].reset_index(drop=True)


# PyTorch MLP (sklearn 호환): hidden층 + BatchNorm + Dropout, AdamW, 내부 10% 검증 loss로 early stopping
class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(64, 32, 16), dropout=0.2, lr=1e-3, wd=1e-3, epochs=200, patience=15, seed=0):
        self.hidden, self.dropout, self.lr, self.wd, self.epochs, self.patience, self.seed = hidden, dropout, lr, wd, epochs, patience, seed

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        self.classes_ = np.array([0, 1])
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.1, random_state=self.seed, stratify=y)
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        self.net_ = nn.Sequential(*layers, nn.Linear(d, 1))
        opt, loss_f = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd), nn.BCEWithLogitsLoss()
        best, bad, state = 1e9, 0, None
        for _ in range(self.epochs):
            self.net_.train()
            for b in torch.randperm(len(i_tr)).split(64):
                if len(b) < 2:
                    continue
                opt.zero_grad(); loss_f(self.net_(Xt[i_tr][b]).squeeze(1), yt[i_tr][b]).backward(); opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = loss_f(self.net_(Xt[i_va]).squeeze(1), yt[i_va]).item()
            if v < best - 1e-4:
                best, bad, state = v, 0, {k: t.clone() for k, t in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(state)
        self.net_.eval()
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(X, dtype=torch.float32)).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4)


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=100, num_leaves=4, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


# 모델 종류: 선형 / 비선형(커널·거리·트리·부스팅) / 딥러닝. 로지스틱 회귀는 C=1 (질환별 기본 feature 선택에 사용).
# 나머지 모델의 하이퍼파라미터는 이전 탐색에서 Gangnam valid AUC로 고른 값(valid 양성이 적어 불안정). Elastic-net·PyTorch MLP는 튜닝하지 않은 고정값
MODELS = {
    "Logistic Regression": lambda: LogisticRegression(C=1.0, penalty="l2", max_iter=2000),
    "Logistic (Elastic-net)": lambda: LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5, max_iter=5000),
    "SVM (RBF)": lambda: SVC(probability=True, C=10.0, gamma=0.03, random_state=0),
    "k-NN": lambda: KNeighborsClassifier(40),
    "Naive Bayes": lambda: GaussianNB(var_smoothing=1e-9),
    "Random Forest": lambda: RandomForestClassifier(300, min_samples_leaf=20, max_depth=3, random_state=0, n_jobs=-1),
    "Extra Trees": lambda: ExtraTreesClassifier(300, min_samples_leaf=3, max_depth=6, random_state=0, n_jobs=-1),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=100, random_state=0),
    "XGBoost": _xgb,
    "LightGBM": _lgbm,
    "MLP (sklearn, 2층)": lambda: MLPClassifier((16,), alpha=1e-3, max_iter=800, early_stopping=True, random_state=0),
    "Deep MLP (PyTorch, 3층+BN+Dropout)": lambda: TorchMLP(),
}


# Gangnam 7/1/2 stratified split 인덱스 (질환 라벨 y 기준, seed 고정이면 baseline과 동일 분할)
def split(y: np.ndarray, seed: int):
    i_tv, i_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    i_tr, i_va = train_test_split(i_tv, test_size=0.1 / 0.8, random_state=seed, stratify=y[i_tv])
    return i_tr, i_va, i_te


# 모델 학습: Gangnam 7/1/2 stratified hold-out. train으로 학습 -> valid/test 예측, train+valid로 재학습한 모델로 Sinchon 예측
@st.cache_data(show_spinner="모델 학습 중...")
def run(feats: tuple, disease: str, model: str, seed: int):
    G, S = apply_pop(load("gangnam"), disease), apply_pop(load("sinchon"), disease)
    X, y = G[list(feats)].to_numpy(float), G[disease].to_numpy(int)
    i_tr, i_va, i_te = split(y, seed)
    i_tv = np.concatenate([i_tr, i_va])
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MODELS[model]())
    m = mk().fit(X[i_tr], y[i_tr])
    final = mk().fit(X[i_tv], y[i_tv])
    ext = final.predict_proba(S[list(feats)].to_numpy(float))[:, 1]
    coef = None
    if model == "Logistic Regression":
        coef = pd.DataFrame({"feature": feats, "coef(+1SD)": final[-1].coef_[0], "OR(+1SD)": np.exp(final[-1].coef_[0])})
    p = lambda i: m.predict_proba(X[i])[:, 1]
    return dict(y_va=y[i_va], p_va=p(i_va), y_te=y[i_te], p_te=p(i_te), y_ext=S[disease].to_numpy(int), p_ext=ext, pid_ext=S.PatientID.to_numpy(), coef=coef)


# feature 이름 목록을 모델 선택 탭 위젯(조직 단면 / Tier1 / Tier2)의 session_state 값으로 분류
def classify(sel):
    fp_, ft2, fr = {t: [] for t in TIS}, {t: [] for t in TIS}, []
    for c in sel:
        head = c.split("@")[0]
        if "/" in c and head in TIS:
            ft2[head].append(c)
        elif "/" in c:
            fr.append(c)
        else:
            fp_[head].append(c.split("@")[1])
    return {**fp_, **{f"t2_{t}": v for t, v in ft2.items()}, "ratios_sel": fr}


# 질환별 기본 모델(baseline + 기본 feature)의 odds ratio: 코호트별로 로지스틱 회귀를 전체 모집단에 적합. 연속형은 코호트 내 z-score(+1SD당 OR), Sex(M)은 남 vs 여
@st.cache_data(show_spinner="odds ratio 계산 중...")
def or_table(dz: str) -> pd.DataFrame:
    fs_ = list(CLIN_BASE) + DEFAULT_FEATS[dz]
    out = {}
    for tag, F in [("internal", load("gangnam")), ("external", load("sinchon"))]:
        d = apply_pop(F, dz)
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(d[fs_]), columns=fs_)
        for c in fs_:
            if c != "Sex(M)":
                X[c] = (X[c] - X[c].mean()) / X[c].std()
        try:
            r = sm.Logit(d[dz].to_numpy(int), sm.add_constant(X)).fit(disp=0, maxiter=200)
            ci = np.exp(r.conf_int())
            out[tag] = pd.DataFrame({"OR": np.exp(r.params), "lo": ci[0], "hi": ci[1], "p": r.pvalues}).drop(index="const")
        except Exception:  # 완전 분리 등으로 적합 실패
            out[tag] = pd.DataFrame({"OR": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}, index=fs_)
    T = out["internal"].join(out["external"], lsuffix="_int", rsuffix="_ext")
    T.insert(0, "질환", dz)
    T.index.name = "feature"
    return T.reset_index()


# 같은 환자에 대한 paired bootstrap으로 ΔAUC 95% CI (B=500)
def boot_delta(y, a, b, B=500, seed=0):
    rng, n, d = np.random.default_rng(seed), len(y), []
    while len(d) < B:
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return np.percentile(d, [2.5, 97.5])


# 임계값 선택 방식: 의료 연구에서 주로 쓰는 것만 (모두 Gangnam valid 예측 기준, test·외부 미사용). target은 목표 민감도/특이도 또는 직접 입력값
THR_MODES = ["Youden (sens+spec 최대)", "Closest to (0,1)", "Sensitivity ≥ 목표 (선별, rule-out)", "Specificity ≥ 목표 (확진, rule-in)", "직접 입력"]


def pick_thr(yv, pv, spec):
    mode, target = spec
    if mode == "직접 입력":
        return float(target)
    f, t, th = roc_curve(yv, pv)
    th = np.where(np.isinf(th), pv.max() + 1e-6, th)
    if mode.startswith("Youden"):
        i = np.argmax(t - f)
    elif mode.startswith("Closest"):
        i = np.argmin(f ** 2 + (1 - t) ** 2)
    elif mode.startswith("Sensitivity"):
        i = np.argmax(t >= target)  # sens가 목표 이상이 되는 가장 높은 임계값(특이도 최대)
    else:  # Specificity
        i = np.where(f <= 1 - target)[0][-1]  # spec이 목표 이상인 가장 낮은 임계값(민감도 최대)
    return float(th[i])


# confusion matrix 그림 + 지표. 행=실제, 열=예측
def cm_stats(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp_, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    dv = lambda a, b: a / b if b else np.nan
    return np.array([[tn, fp_], [fn, tp]]), {"Sens": dv(tp, tp + fn), "Spec": dv(tn, tn + fp_), "PPV": dv(tp, tp + fp_), "NPV": dv(tn, tn + fn), "Acc": dv(tp + tn, len(y))}


def cm_fig(m, title, scale="Blues"):
    pct = m / m.sum(axis=1, keepdims=True)
    fig = px.imshow(pct, x=["예측 -", "예측 +"], y=["실제 -", "실제 +"], zmin=0, zmax=1, color_continuous_scale=scale, text_auto=False, aspect="auto")
    fig.update_traces(text=[[f"{m[i, j]}<br>({pct[i, j]:.0%})" for j in range(2)] for i in range(2)], texttemplate="%{text}", textfont_size=15)
    fig.update_layout(title=title, height=300, coloraxis_showscale=False, margin=dict(t=40, b=10, l=10, r=10))
    return fig


fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
st.set_page_config(page_title="AEC·체성분 모델 실험", layout="wide")
st.title("AEC · 체성분 landmark 모델 실험")
st.caption("Gangnam 7/1/2 hold-out (train/valid/test) → Sinchon 외부검증. 모델 선택 탭에서 설정 후 [모델 실행]을 누르면 Output·비교 탭이 계산됩니다. 연구 요약은 code/research.html 참고.")
disease = st.radio("대상 질환 (비교 설계: 정상 = HTN·DM·CKD 모두 없음  vs  선택 질환 단독)", DISEASES, index=DISEASES.index("DM"), horizontal=True)
FG, FS = load("gangnam"), load("sinchon")
G, S = apply_pop(FG, disease), apply_pop(FS, disease)
st.success(f"**비교 설계**: 정상 vs **{disease} 단독**  —  Gangnam {len(G):,}명 (정상 {int((G[disease] == 0).sum())} / {disease} {int(G[disease].sum())}), "
           f"Sinchon {len(S):,}명 (정상 {int((S[disease] == 0).sum())} / {disease} {int(S[disease].sum())})")
t_sum, t_data, t_split, t_dist, t_model, t_out, t_base = st.tabs(["요약", "1. 데이터 로드", "2. 데이터 분할", "3. Data distribution", "4. 모델 선택", "5. Output", "6. Baseline 비교"])
auc = roc_auc_score

with t_sum:
    st.subheader("최종 요약: 정상 vs 질환 단독, 질환별 기본 조건 (로지스틱 회귀)")
    st.caption("baseline = 성별·나이·키·체중. 각 질환의 기본 feature를 baseline에 추가한 모델의 AUC가 baseline 대비 어떻게 달라지는지 internal test(Gangnam)와 external(Sinchon)에서 비교한 결과입니다. feature는 Gangnam train→valid AUC로만 선택했고(test·external 미사용) 고정 후 한 번만 평가했습니다. Δ는 paired bootstrap 95% CI(500회), p는 DeLong이며 Holm 보정은 질환 3개에 대한 것입니다. split seed 111 고정.")
    srows, fig_rows, pt, pe = [], [], [], []
    for dz in DISEASES:
        Gd, Sd = apply_pop(FG, dz), apply_pop(FS, dz)
        fs_ = tuple(CLIN_BASE) + tuple(DEFAULT_FEATS[dz])
        Q, Q0 = run(fs_, dz, "Logistic Regression", SEED), run(tuple(CLIN_BASE), dz, "Logistic Regression", SEED)
        dt, de = delong_paired_auc_test(Q["y_te"], Q0["p_te"], Q["p_te"]), delong_paired_auc_test(Q["y_ext"], Q0["p_ext"], Q["p_ext"])
        ct, ce = boot_delta(Q["y_te"], Q0["p_te"], Q["p_te"]), boot_delta(Q["y_ext"], Q0["p_ext"], Q["p_ext"])
        pt.append(dt["p_value"]); pe.append(de["p_value"])
        srows.append({"질환": dz, "internal(Gangnam) 정상/단독": f"{int((Gd[dz] == 0).sum())} / {int(Gd[dz].sum())}", "external(Sinchon) 정상/단독": f"{int((Sd[dz] == 0).sum())} / {int(Sd[dz].sum())}",
                      "internal test 양성": int(Q["y_te"].sum()), "추가 feature": len(DEFAULT_FEATS[dz]),
                      "internal test AUC (base→선택)": f"{auc(Q['y_te'], Q0['p_te']):.3f} → {auc(Q['y_te'], Q['p_te']):.3f}",
                      "internal test Δ [95% CI]": f"{dt['diff']:+.3f} [{ct[0]:+.3f}, {ct[1]:+.3f}]", "internal test p": fp(dt["p_value"]),
                      "external AUC (base→선택)": f"{auc(Q['y_ext'], Q0['p_ext']):.3f} → {auc(Q['y_ext'], Q['p_ext']):.3f}",
                      "external Δ [95% CI]": f"{de['diff']:+.3f} [{ce[0]:+.3f}, {ce[1]:+.3f}]", "external p": fp(de["p_value"]), "_dt": dt["diff"], "_de": de["diff"]})
        fig_rows += [(f"{dz} internal test", auc(Q["y_te"], Q0["p_te"]), auc(Q["y_te"], Q["p_te"])), (f"{dz} external", auc(Q["y_ext"], Q0["p_ext"]), auc(Q["y_ext"], Q["p_ext"]))]
    ht, he = multipletests(pt, method="holm")[1], multipletests(pe, method="holm")[1]  # 질환 3개에 대한 Holm 보정
    for r, a_, b_ in zip(srows, ht, he):
        r["internal p (Holm)"], r["external p (Holm)"] = fp(a_), fp(b_)
        r["둘 다 Δ>0 & Holm p<0.05"] = "예" if (r["_dt"] > 0 and a_ < 0.05 and r["_de"] > 0 and b_ < 0.05) else "아니오"
        del r["_dt"], r["_de"]
    ST = pd.DataFrame(srows)
    st.dataframe(ST, hide_index=True, width='stretch')
    fig = go.Figure()
    for lab, j, c_ in [("Baseline", 1, "#0072B2"), ("선택 모델", 2, "#D55E00")]:
        fig.add_bar(name=lab, x=[r[0] for r in fig_rows], y=[r[j] for r in fig_rows], text=[f"{r[j]:.3f}" for r in fig_rows], marker_color=c_)
    fig.update_layout(barmode="group", yaxis_range=[0, 1], height=360, margin=dict(t=20), yaxis_title="AUC")
    st.plotly_chart(fig, width='stretch')
    st.markdown("**질환별 기본 추가 feature**")
    st.dataframe(pd.DataFrame([{"질환": dz, "추가 feature": ", ".join(DEFAULT_FEATS[dz])} for dz in DISEASES]), hide_index=True, width='stretch')
    st.download_button("요약 표 CSV", ST.to_csv(index=False), "summary_default_conditions.csv", mime="text/csv")

    st.subheader("Odds ratio (기본 모델의 feature별, 정상 vs 질환 단독)")
    ORT = pd.concat([or_table(dz) for dz in DISEASES], ignore_index=True)
    ORT["방향 일치"] = np.where(np.sign(np.log(ORT["OR_int"])) == np.sign(np.log(ORT["OR_ext"])), "예", "아니오")
    fmt = lambda o, lo, hi: "-" if pd.isna(o) else f"{o:.2f} ({lo:.2f}-{hi:.2f})"
    show = pd.DataFrame({"질환": ORT["질환"], "feature": ORT["feature"],
                         "internal OR (95% CI)": [fmt(*r) for r in ORT[["OR_int", "lo_int", "hi_int"]].to_numpy()], "internal p": ORT["p_int"].map(lambda v: "-" if pd.isna(v) else fp(v)),
                         "external OR (95% CI)": [fmt(*r) for r in ORT[["OR_ext", "lo_ext", "hi_ext"]].to_numpy()], "external p": ORT["p_ext"].map(lambda v: "-" if pd.isna(v) else fp(v)),
                         "방향 일치": ORT["방향 일치"]})
    st.dataframe(show, hide_index=True, width='stretch')
    st.caption("각 코호트의 정상 + 질환 단독 환자 전체에 로지스틱 회귀를 적합(hold-out 분할과 무관). 연속형은 코호트 내 +1SD당 OR, Sex(M)은 남성 vs 여성. 모든 feature를 동시에 투입해 다른 변수 보정 후의 값이며, "
               "Tier2 비율끼리·키·체중은 상관이 커서 OR이 불안정할 수 있습니다. 로그 비율 feature는 +1SD 증가가 해당 두 값의 비 변화에 대응합니다. 다중비교 보정은 없습니다.")
    FD = st.radio("Forest plot 질환", DISEASES, index=DISEASES.index(disease), horizontal=True, key="forest_dis")
    F_ = ORT[ORT["질환"] == FD].iloc[::-1]
    fig = go.Figure()
    for tag, c_, off in [("int", "#0072B2", 0.12), ("ext", "#D55E00", -0.12)]:
        yy = np.arange(len(F_)) + off
        fig.add_scatter(x=F_[f"OR_{tag}"], y=yy, mode="markers", marker=dict(color=c_, size=9), name="internal" if tag == "int" else "external",
                        error_x=dict(type="data", symmetric=False, array=F_[f"hi_{tag}"] - F_[f"OR_{tag}"], arrayminus=F_[f"OR_{tag}"] - F_[f"lo_{tag}"], color=c_))
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    fig.update_layout(height=max(320, 42 * len(F_)), xaxis_type="log", xaxis_title="Odds ratio (log scale)", yaxis=dict(tickmode="array", tickvals=list(range(len(F_))), ticktext=list(F_["feature"])), margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    st.warning("**해석**: valid로만 feature를 고르는 절차로 바꾸자 세 질환 모두 baseline 대비 개선이 유의하지 않고 external에서는 Δ가 음수입니다(위 표). 이전에 test·external p로 feature를 고른 결과는 선택 과정이 만든 착시였습니다. "
               "valid 양성이 HTN 24·DM 10·CKD 2명뿐이라 valid 기반 선택 자체가 불안정하고(특히 CKD는 무의미), internal test 양성도 DM 19·CKD 4명으로 CI가 매우 넓습니다. "
               "현재 데이터·설계에서는 baseline(성별·나이·키·체중)에 landmark 지표를 추가한 AUC 개선을 지지하는 증거가 없다고 해석하는 것이 정직합니다.")

with t_data:
    st.markdown(f"위에서 고른 **{disease}** 기준으로, 정상군(세 질환 모두 없음)과 {disease} 단독군만 사용합니다. 다른 질환이 하나라도 겹친 환자는 모든 탭에서 제외됩니다.")
    rows = []
    for n, F in [("Gangnam (학습)", FG), ("Sinchon (외부)", FS)]:
        d = apply_pop(F, disease)
        rows.append({"코호트": n, "QC 후 전체": len(F), "정상": int((d[disease] == 0).sum()), f"{disease} 단독": int(d[disease].sum()), "제외(복합·타 질환)": len(F) - len(d),
                     "사용 인원": len(d), "여성(%)": round(100 * (1 - d["Sex(M)"].mean()), 1), "나이": f"{d.PatientAge.mean():.1f} ± {d.PatientAge.std():.1f}", "BMI": f"{d.BMI.mean():.1f} ± {d.BMI.std():.1f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("QC(landmark 순서 이상 제외) 후 인원 기준. 사용 가능한 feature: clinical 4개(성별·나이·키·체중), 조직 4종 × landmark 12곳, Tier1 비율 33개, Tier2 비율(같은 조직의 landmark 쌍) 253개.")
    cohort = st.radio("미리보기", ["Gangnam", "Sinchon"], horizontal=True)
    d = G if cohort == "Gangnam" else S
    show = st.multiselect("컬럼", [c for c in d.columns if c != "PatientID"], default=CLIN_BASE + DISEASES)  # clinic4(성별·나이·키·체중) + 질환 3개만 기본 표시
    st.dataframe(d[show].head(200), width='stretch')
    st.dataframe(d[show].describe().T.round(2), width='stretch')

with t_model:
    if st.session_state.get("_def_dis") != disease:  # 질환이 바뀌면 그 질환의 기본 feature·모델(로지스틱)로 초기화
        st.session_state.update(classify(DEFAULT_FEATS[disease]))
        st.session_state["model_sel"] = "Logistic Regression"
        st.session_state["_def_dis"] = disease
    if "_best" in st.session_state:  # [Best 모델·feature 선택]이 정한 모델과 feature를 위젯 생성 전에 적용
        _b = st.session_state.pop("_best")
        st.session_state["model_sel"] = _b["model"]
        st.session_state.update(_b["feat_state"])
    with st.form("cfg"):
        c1, c2 = st.columns(2)
        model = c1.selectbox("모델", list(MODELS), key="model_sel")
        seed = SEED
        base = c2.multiselect("Baseline (비교 기준) feature", CLIN_ALL, default=CLIN_BASE)
        st.markdown("**입력 feature** (Baseline feature는 항상 포함되고, 아래는 추가할 feature)")
        cols = st.columns(4)
        pick = {t: cols[i].multiselect(f"{t} @ landmark", LMN, key=t) for i, t in enumerate(TIS)}
        ratios = st.multiselect("Tier1 비율 (같은 landmark: VAT/SAT, VAT/TAMA, TAMA/(VAT+SAT))", [f"{k}@{n}" for n in LMN[:-1] for k in ["VAT/SAT", "VAT/TAMA", "TAMA/(VAT+SAT)"]], key="ratios_sel")
        st.markdown("<span style='font-size:0.875rem'>Tier2 비율 (같은 지표의 다른 landmark 간 로그 비, 예: VAT@L1/VAT@L3)</span>", unsafe_allow_html=True)  # 다른 위젯 라벨(0.875rem)과 같은 크기
        cols2 = st.columns(4)
        t2sel = [x for i, t in enumerate(TIS) for x in cols2[i].multiselect(
            f"{t} landmark 쌍", [t2_name(t, LMN[a], LMN[b]) for a, b in itertools.combinations(range(len(LMN)), 2) if not (t == "VAT" and LMN[-1] in (LMN[a], LMN[b]))], key=f"t2_{t}")]
        st.form_submit_button("모델 실행", type="primary")
    feats = tuple(list(base) + [f"{t}@{n}" for t in TIS for n in pick[t]] + ratios + t2sel)
    with st.expander(f"선택된 feature {len(feats)}개"):
        st.write(", ".join(feats) or "없음")
    # 현재 질환·feature·baseline에 대한 로지스틱 회귀 결과를 계산해 안내문을 만든다 (질환/선택이 바뀌면 같이 바뀜)
    n_g, n_pos_g = len(G), int(G[disease].sum()) if len(G) else 0
    if feats and base and n_g >= 100 and min(n_pos_g, n_g - n_pos_g) >= 20 and S[disease].nunique() == 2:
        QL, QL0 = run(feats, disease, "Logistic Regression", seed), run(tuple(base), disease, "Logistic Regression", seed)
        dte, dex = delong_paired_auc_test(QL["y_te"], QL0["p_te"], QL["p_te"]), delong_paired_auc_test(QL["y_ext"], QL0["p_ext"], QL["p_ext"])
        a_te0, a_te1, a_ex0, a_ex1 = auc(QL["y_te"], QL0["p_te"]), auc(QL["y_te"], QL["p_te"]), auc(QL["y_ext"], QL0["p_ext"]), auc(QL["y_ext"], QL["p_ext"])
        ok_te, ok_ex = dte["diff"] > 0 and dte["p_value"] < 0.05, dex["diff"] > 0 and dex["p_value"] < 0.05
        verdict = {(True, True): "test와 외부 **둘 다 개선이 DeLong 유의**합니다.", (False, True): "**Sinchon 외부에서만 개선이 DeLong 유의**하고 Gangnam test는 유의하지 않습니다.",
                   (True, False): "**Gangnam test에서만 개선이 DeLong 유의**하고 외부는 유의하지 않습니다.", (False, False): "test와 외부 모두 개선이 DeLong 유의하지 않습니다."}[(ok_te, ok_ex)]
        npv, npt = int(QL["y_va"].sum()), int(QL["y_te"].sum())
        txt = (f"**정상 vs {disease} 단독** · 현재 설정(로지스틱 회귀, baseline 포함 feature {len(feats)}개) vs baseline({', '.join(base)}): "
               f"Sinchon 외부 AUC {a_ex0:.3f}→{a_ex1:.3f} (p={fp(dex['p_value'])}), Gangnam test AUC {a_te0:.3f}→{a_te1:.3f} (p={fp(dte['p_value'])}). {verdict} "
               f"Gangnam {disease} 단독 {n_pos_g}명 중 valid 양성 {npv}명·test 양성 {npt}명")
        txt += "으로 표본이 작아 결과가 매우 불안정합니다. " if min(npv, npt) < 30 else "입니다. "
        if tuple(feats) == tuple(CLIN_BASE) + tuple(DEFAULT_FEATS[disease]) and list(base) == CLIN_BASE:
            txt += f"이 기본 feature는 {disease}에서 Gangnam valid AUC로만 골랐고 test·external은 선택에 쓰지 않았으므로 위 p는 선택에 오염되지 않았습니다(valid 양성이 적어 선택 자체는 불안정). "
        else:
            txt += "기본 조건에서 바꾼 설정이며, 결과를 보며 feature를 고르면 p가 낙관적이 됩니다(다중비교 보정 없음). "
        txt += "하이퍼파라미터는 DM(valid 양성 10명)에서 골라 불안정하며, 모델 간 비교는 [전체 모델 비교] 탭을 참고하세요. split seed는 111로 고정입니다."
        st.info(txt)
    bc1, bc2 = st.columns([1, 2])
    crit = "valid (권장)"  # 선택은 valid AUC로만 (test·external은 평가용으로 보존)
    bc2.caption("선택 기준: Gangnam valid AUC (test·external은 선택에 쓰지 않음)")
    if bc1.button("Best 모델·feature 선택", help="valid AUC로 forward selection(로지스틱, 최대 6개)해 feature를 고르고, 그 feature로 12개 모델을 학습해 valid AUC가 가장 높은 모델을 설정합니다."):
        if not base:
            st.warning("Baseline feature를 하나 이상 선택하세요.")
        else:
            ckey = {"valid (권장)": "valid AUC", "test": "test AUC", "Sinchon 외부": "Sinchon 외부 AUC"}[crit]
            AUCS = {"valid AUC": ("y_va", "p_va"), "test AUC": ("y_te", "p_te"), "Sinchon 외부 AUC": ("y_ext", "p_ext")}
            score = lambda fs, mn="Logistic Regression": roc_auc_score(*(run(tuple(fs), disease, mn, seed)[k] for k in AUCS[ckey]))
            singles = [f"{t}@{n}" for t in TIS for n in LMN if not (t == "VAT" and n == LMN[-1])]
            tier1 = [f"{k}@{n}" for n in LMN[:-1] for k in ["VAT/SAT", "VAT/TAMA", "TAMA/(VAT+SAT)"]]
            tier2 = [t2_name(t, LMN[a_], LMN[b_]) for t in TIS for a_, b_ in itertools.combinations(range(len(LMN)), 2) if not (t == "VAT" and LMN[-1] in (LMN[a_], LMN[b_]))]
            pool = [c for c in singles + tier1 + tier2 if c not in base]
            sel, cur, prog = [], score(base), st.progress(0.0)
            for step in range(6):
                prog.progress(step / 6, text=f"feature 탐색 {step + 1}/6 (후보 {len(pool)}개)...")
                v, c = max(((score(list(base) + sel + [c_]), c_) for c_ in pool if c_ not in sel), key=lambda x: x[0])
                if v - cur < 0.002:
                    break
                sel.append(c); cur = v
            res = []
            for i, mn in enumerate(MODELS):
                prog.progress(i / len(MODELS), text=f"{mn} 학습 중...")
                Q = run(tuple(list(base) + sel), disease, mn, seed)
                res.append({"모델": mn, "valid AUC": roc_auc_score(Q["y_va"], Q["p_va"]), "test AUC": roc_auc_score(Q["y_te"], Q["p_te"]), "Sinchon 외부 AUC": roc_auc_score(Q["y_ext"], Q["p_ext"])})
            prog.empty()
            R_ = pd.DataFrame(res).sort_values(ckey, ascending=False).reset_index(drop=True)
            st.session_state["best_rank"] = (R_, ckey, sel)
            st.session_state["_best"] = {"model": R_.loc[0, "모델"], "feat_state": classify(sel)}
            st.rerun()
    if "best_rank" in st.session_state:
        R_, ckey, sel = st.session_state["best_rank"]
        st.success(f"Best: **{R_.loc[0, '모델']}** ({ckey} {R_.loc[0, ckey]:.3f}), 추가 feature {len(sel)}개: {', '.join(sel) or '없음'} — 모델과 feature 선택에 설정되었습니다. 질환·baseline을 바꾸면 버튼을 다시 누르세요. valid AUC로 후보 수백 개를 탐색하므로 valid AUC는 크게 부풀려집니다(양성이 적을수록 심함). 선택에 쓰지 않은 internal test와 external AUC(Output·Baseline 탭)로 확인하세요.")
        st.dataframe(R_.round(3), hide_index=True, width='stretch')
    trials = st.session_state.setdefault("trials", set())
    trials.add((feats, disease, model, seed))
    st.caption(f"이번 세션에서 시도한 설정 {len(trials)}개. test·외부 AUC를 보고 feature/모델을 고르면 그 성능은 낙관적입니다 (선택은 valid AUC로, 외부 결과는 확정 후 한 번만 보고하세요).")

# 필터·선택 검증: 문제가 있으면 모든 결과 탭에 같은 안내를 띄우고 중단
npos = int(G[disease].sum()) if len(G) else 0
msg = None
if len(G) < 100 or len(S) < 50:
    msg = f"정상 vs {disease} 단독으로 좁히면 환자가 너무 적습니다 (Gangnam<100 또는 Sinchon<50). 다른 질환을 고르세요."
elif min(npos, len(G) - npos) < 20 or S[disease].nunique() < 2:
    msg = f"{disease} 단독 또는 정상이 Gangnam에서 20명 미만(또는 Sinchon이 한 클래스뿐)이라 분할·평가가 불안정합니다. 다른 질환을 고르세요."
elif not feats or not base:
    msg = "모델 선택 탭에서 feature와 baseline을 하나 이상 선택하세요."
if msg:
    for t in (t_split, t_dist, t_out, t_base):
        t.error(msg)
    st.stop()
if len(feats) > npos / 10:
    t_model.warning(f"feature {len(feats)}개가 양성 {npos}명 대비 많습니다 (양성 10명당 1개 권장). 과적합 가능성이 큽니다.")

with t_split:
    st.write(f"정상 vs **{disease} 단독** 데이터를 {disease} 비율을 유지(stratified)하며 split seed={seed}로 나눕니다. Gangnam은 train 70% / valid 10% / test 20%로 나눕니다. train으로 학습 → valid·test 평가, train+valid로 재학습한 모델을 Sinchon(전체)에 적용합니다.")
    i_tr, i_va, i_te = split(G[disease].to_numpy(int), seed)
    parts = {"Gangnam train": G.iloc[i_tr], "Gangnam valid": G.iloc[i_va], "Gangnam test": G.iloc[i_te], "Sinchon (외부 전체)": S}
    rows = [{"구분": k, "N": len(d), "비율(%)": round(100 * len(d) / len(G), 1) if k.startswith("Gangnam") else None, **{f"{k} 양성 (n, %)": f"{int(d[k].sum())} ({100 * d[k].mean():.1f}%)" for k in DISEASES}, "남성(%)": round(100 * d["Sex(M)"].mean(), 1), "나이": f"{d.PatientAge.mean():.1f} ± {d.PatientAge.std():.1f}",
             "BMI": f"{d.BMI.mean():.1f} ± {d.BMI.std():.1f}"} for k, d in parts.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    fig = go.Figure()
    for lab, c, col_ in [("음성", 0, "#0072B2"), ("양성", 1, "#D55E00")]:  # Data distribution 탭과 같은 파랑/주황 대비
        fig.add_bar(name=lab, x=list(parts), y=[int((d[disease] == c).sum()) for d in parts.values()], marker_color=col_)
    fig.update_layout(barmode="stack", height=360, yaxis_title="환자 수", margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    st.caption("valid는 보고용(튜닝 없음). split seed는 111로 고정되어 있습니다. test가 작아 다른 seed에서는 AUC가 크게 달라질 수 있습니다.")

with t_dist:
    num = [c for c in G.columns if c not in ("PatientID", "PatientSex", "Sex(M)", *DISEASES) and not c.endswith("_center") and pd.api.types.is_numeric_dtype(G[c])]
    c1, c2 = st.columns(2)
    col = c1.selectbox("변수", num, index=num.index(DEFAULT_FEATS[disease][0]) if DEFAULT_FEATS[disease][0] in num else 0)
    by = c2.radio("색 구분", ["질환 유무", "코호트"], horizontal=True)
    both = pd.concat([G.assign(코호트="Gangnam"), S.assign(코호트="Sinchon")], ignore_index=True)
    both["질환 유무"] = np.where(both[disease] == 1, f"{disease} 단독", "정상")
    cmap = {f"{disease} 단독": "#D55E00", "정상": "#0072B2", "Gangnam": "#0072B2", "Sinchon": "#D55E00"}  # 색약 안전한 주황/파랑 대비
    fig = px.histogram(both, x=col, color=by, facet_col="코호트" if by == "질환 유무" else None, barmode="overlay", histnorm="probability density", opacity=0.65, nbins=40, color_discrete_map=cmap)
    fig.update_layout(height=380, margin=dict(t=30))
    st.plotly_chart(fig, width='stretch')
    st.plotly_chart(px.box(both, x="코호트", y=col, color="질환 유무", color_discrete_map=cmap).update_layout(height=340, margin=dict(t=20)), width='stretch')
    st.dataframe(both.groupby(["코호트", "질환 유무"])[col].agg(["count", "mean", "std", "median"]).round(2), width='stretch')

    # 질환 유무(+/-) 간 차이 검정: Mann-Whitney U(양측) + Cohen d(양성-음성, 합동 SD)
    def diff_test(d, c):
        x1, x0 = d.loc[d[disease] == 1, c].dropna(), d.loc[d[disease] == 0, c].dropna()
        sp = np.sqrt(((len(x1) - 1) * x1.var() + (len(x0) - 1) * x0.var()) / (len(x1) + len(x0) - 2))
        return {"n+": len(x1), "n-": len(x0), "mean+": x1.mean(), "mean-": x0.mean(), "Cohen d": (x1.mean() - x0.mean()) / sp if sp > 0 else np.nan,
                "p": mannwhitneyu(x1, x0).pvalue if len(x1) and len(x0) else np.nan}

    st.subheader(f"{disease} 유무에 따른 차이 검정: {col}")
    tt = pd.DataFrame({n: diff_test(d, col) for n, d in [("Gangnam", G), ("Sinchon", S)]}).T
    tt["유의(p<0.05)"] = np.where(tt["p"] < 0.05, "예", "아니오")
    tt["p"] = tt["p"].map(fp)
    st.dataframe(tt.round(3), width='stretch')
    st.caption("Mann-Whitney U(양측). Cohen d>0은 질환 있는 쪽이 더 큼. 두 코호트 모두 유의하고 부호가 같으면 재현된 것으로 봅니다. 변수를 여러 개 보면 다중비교 보정이 필요합니다(아래 전체 변수 표는 BH-FDR 적용).")
    with st.expander("전체 변수 일괄 검정 (BH-FDR)"):
        allr = []
        for c in num:
            rg, rs = diff_test(G, c), diff_test(S, c)
            allr.append({"변수": c, "d Gangnam": rg["Cohen d"], "p Gangnam": rg["p"], "d Sinchon": rs["Cohen d"], "p Sinchon": rs["p"]})
        al = pd.DataFrame(allr)
        for k in ("Gangnam", "Sinchon"):
            al[f"FDR {k}"] = bh_fdr(al[f"p {k}"].fillna(1).to_numpy())
        al["두 코호트 일관"] = np.where((al["FDR Gangnam"] < 0.05) & (al["FDR Sinchon"] < 0.05) & (np.sign(al["d Gangnam"]) == np.sign(al["d Sinchon"])), "예", "-")
        al = al.sort_values(["두 코호트 일관", "FDR Gangnam"], ascending=[False, True])
        for k in ("p Gangnam", "p Sinchon", "FDR Gangnam", "FDR Sinchon"):
            al[k] = al[k].map(fp)
        st.dataframe(al.round(3), hide_index=True, width='stretch')
        st.caption(f"{disease} 유무 기준. FDR은 코호트별로 변수 전체({len(al)}개)에 대해 BH 보정. '일관' = 두 코호트 FDR<0.05 & 같은 방향.")
    st.caption(f"질환은 모델 선택 탭의 선택({disease})을 따릅니다. 값은 landmark 단면적(cm²)·AEC(mA)·로그 비율입니다.")

R, R0 = run(feats, disease, model, seed), run(tuple(base), disease, model, seed)  # baseline은 같은 모델·split
y, oof, ys, ext, coef = R["y_te"], R["p_te"], R["y_ext"], R["p_ext"], R["coef"]
y0, oof0, ext0 = R0["y_te"], R0["p_te"], R0["p_ext"]
d_int, d_ext = delong_paired_auc_test(y, oof0, oof), delong_paired_auc_test(ys, ext0, ext)

with t_out:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("feature 수", len(feats))
    c2.metric("Gangnam valid AUC", f"{auc(R['y_va'], R['p_va']):.3f}")
    c3.metric("Gangnam test AUC", f"{auc(y, oof):.3f}", f"{auc(y, oof) - auc(y0, oof0):+.3f} vs baseline")
    c4.metric("Sinchon 외부 AUC", f"{auc(ys, ext):.3f}", f"{auc(ys, ext) - auc(ys, ext0):+.3f} vs baseline")
    c5.metric("Sinchon Brier", f"{brier_score_loss(ys, ext):.3f}", f"{brier_score_loss(ys, ext) - brier_score_loss(ys, ext0):+.3f} vs baseline", delta_color="inverse",
              help="낮을수록 좋음(보정 정도). 코호트 간 유병률 차이로 커질 수 있음")
    st.caption(f"{disease} / {model}. Gangnam valid n={len(R['y_va'])}, test n={len(y)} (양성 {int(y.sum())}), Sinchon n={len(ys)} (양성 {int(ys.sum())}). split seed 111 고정, test가 작아 다른 seed에서는 AUC가 달라질 수 있음.")
    fig = go.Figure()
    for name, yy, p, c_ in [("Gangnam test", y, oof, "#0072B2"), ("Sinchon 외부", ys, ext, "#D55E00")]:
        f, t, _ = roc_curve(yy, p)
        fig.add_scatter(x=f, y=t, name=f"{name} {auc(yy, p):.3f}", line=dict(color=c_, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="gray", dash="dash"), showlegend=False)
    fig.update_layout(height=420, xaxis_title="1 - Specificity", yaxis_title="Sensitivity", margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    st.subheader("Confusion matrix")
    c1_, c2_ = st.columns([2, 1])
    mode = c1_.selectbox("임계값 선택 방식", THR_MODES, help="모든 방식은 Gangnam valid 예측으로 임계값을 정하고, 같은 값을 test와 외부에 적용합니다 (test·외부 결과로 정하지 않음).")
    if mode.startswith(("Sensitivity", "Specificity")):
        target = c2_.slider("목표 민감도/특이도", 0.5, 0.99, 0.9, 0.01)
    elif mode == "직접 입력":
        target = c2_.number_input("임계값", 0.0, 1.0, 0.5, 0.01)
    else:
        target = None
    thr_mode = (mode, target)
    thr = pick_thr(R["y_va"], R["p_va"], thr_mode)
    cc, rows_cm = st.columns(2), []
    for col_, (n, yy, pp, sc_) in zip(cc, [("Gangnam test", y, oof, "Blues"), ("Sinchon 외부", ys, ext, "Oranges")]):
        m, st_ = cm_stats(yy, pp, thr)
        col_.plotly_chart(cm_fig(m, f"{n} (임계값 {thr:.3f})", sc_), width='stretch')
        rows_cm.append({"평가": n, **st_})
    st.dataframe(pd.DataFrame(rows_cm).round(3), hide_index=True, width='stretch')
    st.caption("Sens=민감도, Spec=특이도, PPV/NPV=양성/음성 예측도. 임계값은 Gangnam valid에서 정했고 외부 코호트는 유병률이 달라 PPV/NPV가 달라집니다.")
    if coef is not None:
        st.subheader("계수 (표준화 입력, Gangnam train+valid 80%로 재학습한 외부검증용 모델)")
        st.dataframe(coef.round(3), width='stretch', hide_index=True)
    st.caption("valid/test AUC는 train(70%)으로 학습한 모델, 외부 AUC는 train+valid(80%)로 재학습한 모델의 결과입니다.")
    out = pd.DataFrame({"PatientID": R["pid_ext"], "y": ys, "pred_prob": ext})
    st.download_button("Sinchon 예측값 CSV", out.to_csv(index=False), f"pred_{disease}_{model.split()[0]}_s{seed}.csv", mime="text/csv")

with t_base:
    st.write(f"Baseline = {', '.join(base)} (동일 모델·split)")
    rows = []
    for n, yy, a_, b_, d_ in [("Gangnam test (20%)", y, oof0, oof, d_int), ("Sinchon 외부", ys, ext0, ext, d_ext)]:
        lo, hi = boot_delta(yy, a_, b_)
        rows.append({"평가": n, "Baseline AUC": f"{auc(yy, a_):.3f}", "선택 모델 AUC": f"{auc(yy, b_):.3f}",
                     "ΔAUC [95% bootstrap CI]": f"{auc(yy, b_) - auc(yy, a_):+.3f} [{lo:+.3f}, {hi:+.3f}]", "DeLong p": fp(d_["p_value"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("ΔAUC>0이고 CI가 0을 제외하며 p<0.05이면 baseline보다 높다고 볼 수 있습니다. 설정을 여러 번 바꿔 본 뒤의 값은 낙관적입니다.")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    cohorts = [("Gangnam test", y, oof0, oof), ("Sinchon 외부", ys, ext0, ext)]
    for lab, j, c_ in [("Baseline", 2, "#0072B2"), ("선택 모델", 3, "#D55E00")]:
        vals = [auc(co[1], co[j]) for co in cohorts]
        fig.add_bar(name=lab, x=[co[0] for co in cohorts], y=vals, text=[f"{v:.3f}" for v in vals], marker_color=c_)
    fig.update_layout(barmode="group", yaxis_range=[0, 1], height=380, margin=dict(t=20))
    c1.plotly_chart(fig, width='stretch')
    fig = go.Figure()
    for name, p, c_ in [("Baseline", ext0, "#0072B2"), ("선택 모델", ext, "#D55E00")]:
        f, t, _ = roc_curve(ys, p)
        fig.add_scatter(x=f, y=t, name=f"{name} {auc(ys, p):.3f}", line=dict(color=c_, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="gray", dash="dash"), showlegend=False)
    fig.update_layout(title="Sinchon 외부 ROC", height=380, margin=dict(t=40))
    c2.plotly_chart(fig, width='stretch')

    st.subheader("Confusion matrix: baseline vs 선택 모델")
    st.caption(f"임계값은 각 모델의 Gangnam valid 예측에서 정함 (Output 탭 선택: {thr_mode[0]}{'' if thr_mode[1] is None else f' {thr_mode[1]}'}). 같은 환자에 대한 비교.")
    for n, yy, p0, p1, key in [("Gangnam test", y, oof0, oof, "p_va"), ("Sinchon 외부", ys, ext0, ext, "p_va")]:
        t0, t1 = pick_thr(R0["y_va"], R0["p_va"], thr_mode), pick_thr(R["y_va"], R["p_va"], thr_mode)
        (m0, s0), (m1, s1) = cm_stats(yy, p0, t0), cm_stats(yy, p1, t1)
        a_, b_ = st.columns(2)
        a_.plotly_chart(cm_fig(m0, f"{n}: Baseline (임계값 {t0:.3f})", "Blues"), width='stretch')
        b_.plotly_chart(cm_fig(m1, f"{n}: 선택 모델 (임계값 {t1:.3f})", "Oranges"), width='stretch')
        st.dataframe(pd.DataFrame([{"모델": "Baseline", **s0}, {"모델": "선택 모델", **s1}, {"모델": "차이", **{k: s1[k] - s0[k] for k in s0}}]).round(3), hide_index=True, width='stretch')

    st.subheader("세 질환 한꺼번에 (같은 feature·모델)")
    rows = []
    with st.spinner("질환별 학습 중..."):
        for dz in DISEASES:  # 각 질환마다 정상 vs 해당 질환 단독
            Q, Q0 = run(feats, dz, model, seed), run(tuple(base), dz, model, seed)
            a, b, b0, c, e, e0 = Q["y_te"], Q["p_te"], Q0["p_te"], Q["y_ext"], Q["p_ext"], Q0["p_ext"]
            rows.append({"질환": dz, "test Base": auc(a, b0), "test 선택": auc(a, b), "외부 Base": auc(c, e0), "외부 선택": auc(c, e),
                         "외부 Δ": auc(c, e) - auc(c, e0), "p": delong_paired_auc_test(c, e0, e)["p_value"]})
    t3 = pd.DataFrame(rows)
    t3["p (Holm)"] = multipletests(t3["p"], method="holm")[1]
    t3["유의(Holm<0.05)"] = np.where(t3["p (Holm)"] < 0.05, "예", "-")
    t3[["p", "p (Holm)"]] = t3[["p", "p (Holm)"]].map(fp)
    num_cols = t3.select_dtypes("float").columns
    t3[num_cols] = t3[num_cols].round(3)
    st.dataframe(t3, hide_index=True, width='stretch')
    st.caption("각 질환은 정상 vs 해당 질환 단독 모집단에서 따로 학습·평가한 결과입니다. 외부 DeLong p를 질환 수(3)에 대해 Holm 보정했습니다. 피처·모델을 바꿔 본 횟수는 반영되지 않습니다.")
