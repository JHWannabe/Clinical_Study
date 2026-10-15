from __future__ import annotations

# 정상군(세 질환 모두 없음) vs 질환 단독군 결과(outputs/261012_single)로 덱에 쓰는 요약 그림 4종을 만든다 (단독 실행)
#  유의한 위치 수 heatmap(MW, OR), 유의 위치 matrix(raw/pw_mean, 기울기), baseline vs landmark raw(L1,L3,L5) 풀링 CV·외부검증 막대 그림
# 입력: outputs/261012_single/{data_distribution_significance, clinic4_feature_search(_external)}.xlsx
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
import os
OUT = Path(__file__).resolve().parents[2] / "outputs" / os.environ.get("OUT_ROOT", "261012_single")
DIS, MEAS_ = ["HTN", "DM", "CKD"], ["VAT", "SAT", "TAMA", "AEC"]
sig = pd.read_excel(OUT / "data_distribution_significance.xlsx", sheet_name="mannwhitney"); orr = pd.read_excel(OUT / "data_distribution_significance.xlsx", sheet_name="odds_ratio")
PRE = [("raw", "raw (landmark 값)"), ("pw_mean", "pw_mean (값 / 환자 구간평균)")]
rows_ = [(c, d) for c in ["gangnam", "sinchon"] for d in DIS]


# 유의한 landmark 수 heatmap (행 = 병원·질환, 열 = 조직, 숫자 = 12곳 중 유의한 위치 수)
def count_fig(tbl, cmap, fname):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (pre_, ttl) in zip(axs.ravel(), PRE):
        M = np.array([[((tbl.cohort == c) & (tbl.preprocess == pre_) & (tbl.disease == d) & (tbl.measure == m) & (tbl.q_fdr < .05)).sum() for m in MEAS_] for c, d in rows_])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=12)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > 7 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(MEAS_); ax.set_yticks(range(6)); ax.set_yticklabels([f"{'강남' if c == 'gangnam' else '신촌'} {d}" for c, d in rows_], fontsize=8)
        ax.axhline(2.5, color="white", lw=3); ax.set_title(ttl, fontsize=10)
    plt.tight_layout(); fig.savefig(OUT / fname, dpi=200); plt.close(fig)


count_fig(sig, "Blues", "data_distribution_summary.png"); count_fig(orr, "Oranges", "data_distribution_or_summary.png")

ABB = ["LD", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "FH", "PM"]
LMN = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
SG = sig.assign(sg=lambda x: (x.q_fdr < .05)).set_index(["preprocess", "disease", "measure", "landmark", "cohort"])
ARW = lambda v: "↑" if v > 0 else "↓"


# (전처리, 질환, 측정, landmark)의 상태: 두 코호트 모두 유의 'B'(방향이 같으면 화살표, 다르면 ±), 한쪽만 'G'/'S', 없으면 ''
def cell(pre_, d, m, lm):
    g, s_ = [SG.loc[(pre_, d, m, lm, c)] for c in ["gangnam", "sinchon"]]
    if g.sg and s_.sg: return "B", ARW(g.median_diff) if np.sign(g.median_diff) == np.sign(s_.median_diff) else "±"
    if g.sg: return "G", ARW(g.median_diff)
    if s_.sg: return "S", ARW(s_.median_diff)
    return "", ""


def matrix_fig(pre_list, fname):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    col = {"": "#FFFFFF", "B": "#2F6DB5", "G": "#BFD9F2", "S": "#F6C28B"}
    for ax, (pre_, ttl) in zip(axs, pre_list):
        labs = [(d, m) for d in DIS for m in MEAS_]
        for i, (d, m) in enumerate(labs):
            for j, lm in enumerate(LMN):
                k, ar = cell(pre_, d, m, lm)
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=col[k], ec="#CCCCCC", lw=.6))
                if ar: ax.text(j + .5, i + .5, ar, ha="center", va="center", fontsize=10, color="white" if k == "B" else "black")
        ax.set_xlim(0, 12); ax.set_ylim(12, 0); ax.set_aspect("equal"); ax.set_xticks(np.arange(12) + .5); ax.set_xticklabels(ABB, fontsize=8)
        ax.set_yticks(np.arange(12) + .5); ax.set_yticklabels([f"{d} {m}" for d, m in labs], fontsize=8); ax.xaxis.tick_top(); ax.set_title(ttl, fontsize=10, pad=22)
        for y in (4, 8): ax.axhline(y, color="k", lw=1.2)
        for sp in ax.spines.values(): sp.set_visible(False)
    axs[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=col[k], ec="#999") for k in "BGS"], labels=["두 코호트 모두 유의", "강남만 유의", "신촌만 유의"], loc="upper left", bbox_to_anchor=(1.0, .5), fontsize=8, frameon=False)
    fig.text(.5, .01, "화살표: 질환군 median이 정상군보다 높음(↑)/낮음(↓), ±는 두 코호트의 방향이 다름. LD=liver dome, FH=femoral head, PM=inferior pubic margin", ha="center", fontsize=8)
    plt.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(OUT / fname, dpi=200); plt.close(fig)


matrix_fig(PRE, "data_distribution_matrix_a.png")
matrix_fig([("d_raw", "기울기 d_raw"), ("d_pw_mean", "기울기 d_pw_mean (기울기 / 환자 구간평균)")], "data_distribution_matrix_b.png")

# baseline vs landmark raw(L1,L3,L5): 풀링 CV / 외부검증 AUC, * = baseline 대비 FDR q<0.05
fs = pd.read_excel(OUT / "clinic4_feature_search.xlsx"); ex = pd.read_excel(OUT / "clinic4_feature_search_external.xlsx")
best = fs.loc[fs.groupby(["disease", "feature_set"]).AUC.idxmax()]; REP = "c4+LM raw(L1,L3,L5)"
g = lambda v, s_, d: best[(best.disease == d) & (best.feature_set == s_)][v].iloc[0]
E = lambda tr, s_, d: ex[(ex.train == tr) & (ex.feature_set == s_) & (ex.disease == d)].iloc[0]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, d in zip(axs, DIS):
    a = [g("AUC", "clinic4", d), E("gangnam", "clinic4", d).AUC, E("sinchon", "clinic4", d).AUC]
    b = [g("AUC", REP, d), E("gangnam", REP, d).AUC, E("sinchon", REP, d).AUC]
    q = [g("p_fdr_in_disease", REP, d), E("gangnam", REP, d).p_fdr, E("sinchon", REP, d).p_fdr]
    x = np.arange(3)
    ax.bar(x - .2, a, .38, color="#9AA5B1", label="clinic4"); ax.bar(x + .2, b, .38, color="#2F6DB5", label="clinic4 + landmark raw (L1,L3,L5)")
    for xi, bi, qi in zip(x, b, q):
        if qi < .05: ax.text(xi + .2, bi + .005, "*", ha="center", fontsize=13)
    ax.set_title(d, fontsize=11); ax.set_xticks(x); ax.set_xticklabels(["합친 교차검증", "강남→신촌", "신촌→강남"], fontsize=8); ax.set_ylim(.4, .95); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("AUC"); axs[0].legend(fontsize=7.5, loc="upper left", frameon=False)
plt.tight_layout(); fig.savefig(OUT / "clinic4_feature_search.png", dpi=200); plt.close(fig)
print("saved figures")
