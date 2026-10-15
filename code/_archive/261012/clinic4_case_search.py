from __future__ import annotations

# clinic4(성별·나이·키·몸무게) baseline에 landmark 체성분 feature를 case별로 하나씩 더해 AUC 변화를 비교 (단독 실행, outputs/261012_cases 저장)
# 환자군 = 정상군(HTN·DM·CKD 모두 없음) vs 해당 질환 양성(동반질환 포함, 261015 본문과 동일), 강남+신촌 풀링. 평가 = 풀링 5-fold CV 3회 반복(fold별 AUC 평균±SD, Nadeau-Bengio 보정 t) + 강남↔신촌 외부검증(DeLong)
# 모델 = StandardScaler + LR(C=1 고정; case가 수백 개라 C 탐색 생략, baseline도 같은 모델). q = 질환별 전체 case에 대한 BH-FDR
import itertools, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from delong_utils import bh_fdr, delong_paired_auc_test
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/261012_cases"; OUT.mkdir(parents=True, exist_ok=True)
DIS, REPS = ["HTN", "DM", "CKD"], 3
MEAS = ["VAT", "SAT", "TAMA", "AEC"]
LM = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center",
      "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
LN = [a.replace("_center", "") for a in LM]
SEG = [(0, 4), (4, 6), (6, 8), (8, 11)]  # 구간 = liver_dome~L1, L1~L3, L3~L5, L5~pubis (landmark 인덱스)
SEGN = ["liver~L1", "L1~L3", "L3~L5", "L5~pubis"]
SHEET = {m: f"{m.lower()}_total" if m == "AEC" else f"{m}_total" for m in MEAS}
lr_ = lambda a, b: np.log((a + 1) / (b + 1))  # 로그 비율(0 방지), 기존 landmark 스크립트와 동일


# 코호트 환자 메타 + 조직별 landmark 값 L[m] (n,12), 구간별 곡선 통계 {m: dict(mean,sum,peak,pos)} (n,4)
def load(cohort):
    path = ROOT / f"data/{cohort}_landmark_filtered.xlsx"
    meta = pd.read_excel(path, sheet_name="patient_metadata")
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID")
    cur = {m: pd.read_excel(path, sheet_name=SHEET[m]).set_index("PatientID").filter(regex=rf"^{m.lower() if m == 'AEC' else m}_\d+$") for m in MEAS}
    meta = meta[meta.PatientID.isin(set.intersection(*[set(c.index) for c in cur.values()]) & set(lm.index))]
    meta = meta.dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + DIS).reset_index(drop=True)
    sl = lm.loc[meta.PatientID, [f"{a}_slice" for a in LM]].to_numpy(int)
    n, L, S = len(meta), {}, {}
    for m in MEAS:
        v = np.nan_to_num(cur[m].loc[meta.PatientID].to_numpy(float))
        L[m] = np.stack([v[np.arange(n), sl[:, j] - 1] for j in range(12)], 1)
        st = {k: np.zeros((n, 4)) for k in ["mean", "sum", "peak", "pos"]}
        for i in range(n):
            for k, (a, b) in enumerate(SEG):
                lo, hi = sorted((sl[i, a] - 1, sl[i, b] - 1)); seg = v[i, lo:hi + 1]
                st["mean"][i, k], st["sum"][i, k], st["peak"][i, k] = seg.mean(), seg.sum(), seg.max()
                st["pos"][i, k] = seg.argmax() / max(len(seg) - 1, 1)  # 구간 안 peak의 상대 위치(0~1)
        S[m] = st
    meta["sex"] = (meta.PatientSex.astype(str).str.upper() == "M").astype(float); meta["cohort"] = cohort
    return meta, L, S, sl


parts = [load(c) for c in ["gangnam", "sinchon"]]
A = pd.concat([p[0] for p in parts], ignore_index=True)
L = {m: np.vstack([p[1][m] for p in parts]) for m in MEAS}
S = {m: {k: np.vstack([p[2][m][k] for p in parts]) for k in parts[0][2][m]} for m in MEAS}
SL = np.vstack([p[3] for p in parts])
C4 = np.column_stack([A.sex, A.PatientAge, A.Height, A.Weight])

# ---- case 정의: (그룹, case 이름, 추가 feature 행렬). 이름은 xlsx/ppt에 그대로 쓰인다 ----
CASES = []
add = lambda g, name, X: CASES.append((g, name, np.asarray(X, float).reshape(len(A), -1)))
PAIRS = list(itertools.combinations(MEAS, 2))
for m in MEAS: add("1. landmark 12점 값의 합", f"{m} 12점 합", L[m].sum(1))
for m in MEAS:
    for j, a in enumerate(LN): add("2. landmark 개별 값", f"{m}@{a}", L[m][:, j])
for j, a in enumerate(LN):
    for p, q in PAIRS: add("3. landmark별 체성분 비율", f"{p}/{q}@{a}", lr_(L[p][:, j], L[q][:, j]))
for m in MEAS:
    for i, j in itertools.combinations(range(12), 2): add("4. 체성분별 landmark 간 비율", f"{m}@{LN[i]}/{LN[j]}", lr_(L[m][:, i], L[m][:, j]))
for m in MEAS:
    for k, s in enumerate(SEGN): add("5. 구간별 peak 값", f"{m} peak [{s}]", S[m]["peak"][:, k])
for m in MEAS:
    for k, s in enumerate(SEGN): add("6. 구간별 평균", f"{m} 평균 [{s}]", S[m]["mean"][:, k])
for m in MEAS:
    for k, s in enumerate(SEGN): add("7. 구간별 합산", f"{m} 합산 [{s}]", S[m]["sum"][:, k])
for m in MEAS:
    for k, s in enumerate(SEGN): add("8. 구간별 peak 위치", f"{m} peak 위치 [{s}]", S[m]["pos"][:, k])
for k, s in enumerate(SEGN):
    for p, q in PAIRS: add("9. 구간 평균 체성분 비율", f"{p}/{q} 평균 [{s}]", lr_(S[p]["mean"][:, k], S[q]["mean"][:, k]))
for m in MEAS: add("10. 조직 1종의 12곳 전체", f"{m} 12곳 전체", L[m])
for j, a in enumerate(LN): add("11. landmark 1곳의 4종 전체", f"4종@{a}", np.column_stack([L[m][:, j] for m in MEAS]))
for m in MEAS: add("12. 조직 1종의 L1·L3·L5", f"{m}@L1,L3,L5", np.column_stack([L[m][:, LN.index(a)] for a in ["L1", "L3", "L5"]]))
add("12. 조직 1종의 L1·L3·L5", "4종@L1,L3,L5", np.column_stack([L[m][:, LN.index(a)] for m in MEAS for a in ["L1", "L3", "L5"]]))
for m in MEAS:
    for j in range(11):
        gap = np.maximum(SL[:, j + 1] - SL[:, j], 1)
        add("13. 인접 landmark 간 기울기", f"{m} {LN[j]}→{LN[j + 1]}", (L[m][:, j + 1] - L[m][:, j]) / gap)
for p, q in PAIRS: add("14. 12점 합의 체성분 비율", f"{p}/{q} 12점 합", lr_(L[p].sum(1), L[q].sum(1)))


def mk(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


# 풀링 5-fold CV를 REPS회 반복한 fold별 AUC (분할은 모든 case에서 동일)
def cv(X, y, strat):
    r = []
    for k in range(REPS):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=k).split(X, strat):
            r.append(roc_auc_score(y[te], mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return np.array(r)


# Nadeau-Bengio 보정 paired t (fold별 AUC 차이, 시험/학습 = 1/4)
def nb_test(diff):
    J, v = len(diff), np.var(diff, ddof=1)
    return 1.0 if v == 0 else float(2 * t_dist.sf(abs(diff.mean()) / np.sqrt((1 / J + .25) * v), J - 1))


rows, ext = [], []
for d in DIS:
    keep = ((A[d] == 1) | (A[DIS].sum(axis=1) == 0)).to_numpy()
    y = A[d].to_numpy(int)[keep]; coh = A.cohort.to_numpy()[keep]; strat = y * 2 + (coh == "sinchon")
    ia, ib = np.where(coh == "gangnam")[0], np.where(coh == "sinchon")[0]
    base = cv(C4[keep], y, strat)
    pb = {tr: mk().fit(C4[keep][a], y[a]).predict_proba(C4[keep][b])[:, 1] for tr, a, b in [("gangnam", ia, ib), ("sinchon", ib, ia)]}
    print(d, "n=", len(y), "pos=", y.sum(), "baseline AUC", round(base.mean(), 3), flush=True)
    rows.append(dict(disease=d, group="0. baseline", case="clinic4", n_feat=4, n=len(y), n_pos=int(y.sum()), AUC=base.mean(), AUC_sd=base.std(ddof=1), dAUC=0.0, p_cv=np.nan))
    for g, name, Z in CASES:
        X = np.column_stack([C4, Z])[keep]; r = cv(X, y, strat)
        rows.append(dict(disease=d, group=g, case=name, n_feat=X.shape[1], n=len(y), n_pos=int(y.sum()), AUC=r.mean(), AUC_sd=r.std(ddof=1), dAUC=(r - base).mean(), p_cv=nb_test(r - base)))
        for tr, a, b in [("gangnam", ia, ib), ("sinchon", ib, ia)]:
            p = mk().fit(X[a], y[a]).predict_proba(X[b])[:, 1]; t = delong_paired_auc_test(y[b], pb[tr], p)
            ext.append(dict(disease=d, group=g, case=name, train=tr, AUC=roc_auc_score(y[b], p), dAUC=t["diff"], p_delong=t["p_value"]))
    print(d, "done", flush=True)
res, ext = pd.DataFrame(rows), pd.DataFrame(ext)
res["q_fdr"] = res.groupby("disease").p_cv.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
ext["q_fdr"] = ext.groupby(["disease", "train"]).p_delong.transform(lambda p: bh_fdr(p.fillna(1).to_numpy()))
with pd.ExcelWriter(OUT / "clinic4_case_search.xlsx") as xw:
    res.to_excel(xw, sheet_name="pooled_cv", index=False); ext.to_excel(xw, sheet_name="external", index=False)
pd.set_option("display.width", 220)
print(res[res.group != "0. baseline"].sort_values(["disease", "dAUC"], ascending=[True, False]).groupby("disease").head(8)[["disease", "group", "case", "AUC", "dAUC", "p_cv", "q_fdr"]].round(4).to_string(index=False))
print(res.groupby(["group", "disease"]).dAUC.max().unstack().round(4))


# ---- heatmap: ΔAUC (clinic4 대비) 그룹 2(landmark x 조직), 3(landmark x 조직쌍), 4(조직별 landmark x landmark) ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
dv = res.set_index(["disease", "case"]).dAUC
def hm(ax, M, xl, yl, title, v):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=90, fontsize=8); ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl, fontsize=8); return im
for g, rows_, cols_, key, name in [("2", MEAS, LN, lambda r, c: f"{r}@{c}", "case_heatmap_g2"), ("3", [f"{p}/{q}" for p, q in PAIRS], LN, lambda r, c: f"{r}@{c}", "case_heatmap_g3")]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6 if g == "2" else 4.4), constrained_layout=True)
    for ax, d in zip(axs, DIS):
        M = np.array([[dv[(d, key(r, c))] for c in cols_] for r in rows_]); v = max(abs(M).max(), .01)
        im = hm(ax, M, cols_, rows_, d, v); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("clinic4 대비 ΔAUC (풀링 5-fold CV)"); fig.savefig(OUT / f"{name}.png", dpi=130); plt.close(fig)
for d in DIS:
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for ax, m in zip(axs, MEAS):
        M = np.full((12, 12), np.nan)
        for i, j in itertools.combinations(range(12), 2): M[i, j] = dv[(d, f"{m}@{LN[i]}/{LN[j]}")]
        v = max(np.nanmax(abs(M)), .01); im = hm(ax, M, LN, LN, m, v); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f"{d}: 조직별 landmark 간 비율(행/열) ΔAUC"); fig.savefig(OUT / f"case_heatmap_g4_{d}.png", dpi=110); plt.close(fig)
