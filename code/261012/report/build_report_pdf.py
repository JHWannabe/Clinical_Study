from __future__ import annotations

# clinic4 baseline·feature 탐색·외부검증·CT만·성향점수 매칭·성별/병원 보정 결과 xlsx(outputs/261012_any)를 읽어 docs/보고서.pdf로 만든다 (단독 실행)
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, BaseDocTemplate, PageTemplate, Frame, NextPageTemplate

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / "outputs/261012_any"  # 정상군(세 질환 모두 없음) vs 해당 질환 보유(동반질환 포함) 결과
pdfmetrics.registerFont(TTFont("K", r"C:\Windows\Fonts\malgun.ttf")); pdfmetrics.registerFont(TTFont("KB", r"C:\Windows\Fonts\malgunbd.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("K", normal="K", bold="KB", italic="K", boldItalic="KB")
H1 = ParagraphStyle("h1", fontName="KB", fontSize=17, leading=24, spaceAfter=4)
H2 = ParagraphStyle("h2", fontName="KB", fontSize=12, spaceBefore=12, spaceAfter=5, keepWithNext=1)
B = ParagraphStyle("b", fontName="K", fontSize=9.5, leading=15, spaceAfter=4)
CAP = ParagraphStyle("cap", fontName="K", fontSize=8.5, leading=12, spaceAfter=6, textColor=colors.HexColor("#444444"))
CAPT = ParagraphStyle("capt", parent=CAP, fontName="KB", textColor=colors.black, keepWithNext=1, spaceAfter=3)
BOX = ParagraphStyle("box", fontName="K", fontSize=9.5, leading=15)
DIS = ["HTN", "DM", "CKD"]
MEAS_ = ["VAT", "SAT", "TAMA", "AEC"]
GREEN, RED, GREY = colors.HexColor("#D5ECD4"), colors.HexColor("#F6D3D0"), colors.HexColor("#E8EEF4")
base = pd.read_excel(OUT / "clinic4_baseline.xlsx")
fs = pd.read_excel(OUT / "clinic4_feature_search.xlsx")
ext = pd.read_excel(OUT / "clinic4_feature_search_external.xlsx")
fmt = lambda a, d, q, sd=None: ("%.3f±%.3f (%+.3f, q=%.3f)" % (a, sd, d, q)) if sd is not None else "%.3f (%+.3f, q=%.3f)" % (a, d, q)


# 표 생성: marks={(행,열): 색}으로 유의한 셀을 초록(개선)/빨강(악화)으로 칠한다
def tbl(rows, widths, marks=None, skip=0):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    fsz = 8.5  # 글자 크기: 가장 긴 셀이 열 너비(좌우 여백 12pt 제외)에 들어가도록 줄이되 6.5pt 아래로는 내리지 않는다 (skip = 열 너비 계산에서 뺄 맨 위 행 수)
    for r in rows[skip:]:
        for j, v in enumerate(r):
            if isinstance(v, str) and v:
                fsz = min(fsz, 8.5 * (widths[j] - 12) / stringWidth(v, "KB", 8.5))
    fsz = max(6.5, round(fsz * 20) // 2 / 10)
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "K", fsz), ("FONT", (0, 0), (-1, 0), "KB", fsz), ("BACKGROUND", (0, 0), (-1, 0), GREY),
          ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for (r, c), col in (marks or {}).items(): st += [("FONT", (c, r), (c, r), "KB", fsz), ("BACKGROUND", (c, r), (c, r), col)]
    t.setStyle(TableStyle(st)); t._fsz = fsz; return t


SETS = list(dict.fromkeys(fs.feature_set))
best = fs.loc[fs.groupby(["disease", "feature_set"]).AUC.idxmax()]
g = lambda v, s_, d: best[(best.disease == d) & (best.feature_set == s_)][v].iloc[0]
E = lambda tr, s_, d: ext[(ext.train == tr) & (ext.feature_set == s_) & (ext.disease == d)].iloc[0]
name = lambda s_: "clinic4 (baseline)" if s_ == "clinic4" else s_.replace("c4+", "+")
REP = "c4+LM raw(L1,L3,L5)"

# 그림 1: clinic4 vs 대표 구성(landmark raw L1,L3,L5)을 풀링 CV / 두 방향 외부검증에서 비교 (*=FDR q<0.05)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
groups = ["풀링 5-fold CV", "강남→신촌", "신촌→강남"]
for ax, d in zip(axs, DIS):
    a = [g("AUC", "clinic4", d), E("gangnam", "clinic4", d).AUC, E("sinchon", "clinic4", d).AUC]
    b = [g("AUC", REP, d), E("gangnam", REP, d).AUC, E("sinchon", REP, d).AUC]
    q = [g("p_fdr_in_disease", REP, d), E("gangnam", REP, d).p_fdr, E("sinchon", REP, d).p_fdr]
    x = np.arange(3)
    ax.bar(x - .2, a, .38, color="#9AA5B1", label="clinic4"); ax.bar(x + .2, b, .38, color="#2F6DB5", label="clinic4 + landmark raw (L1,L3,L5)")
    for xi, bi, qi in zip(x, b, q):
        if qi < .05: ax.text(xi + .2, bi + .005, "*", ha="center", fontsize=13)
    ax.set_title(d, fontsize=11); ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8); ax.set_ylim(.55, .85); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("AUC"); axs[0].legend(fontsize=7.5, loc="upper left", frameon=False)
plt.tight_layout(); fig.savefig(OUT / "clinic4_feature_search.png", dpi=200); plt.close(fig)

# 표 1: 대상 인원
lr = base[base.model == "LR"]
rows = [["코호트", "n"] + [f"{d} 양성 (%)" for d in DIS]]
for lab, e in [("강남", "gangnam 5-fold CV"), ("신촌", "sinchon 5-fold CV"), ("풀링(강남+신촌)", "pooled 5-fold CV")]:
    r = lr[lr["eval"] == e].set_index("disease")
    rows.append([lab, f"{r.n.iloc[0]:,}"] + [f"{r.loc[d, 'n_pos']:,} ({100 * r.loc[d, 'n_pos'] / r.loc[d, 'n']:.1f}%)" for d in DIS])
T1 = tbl(rows, [4.2 * cm, 2.2 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm])

# 표 2: baseline (LR / HGB)
rows = [["평가 방식", "HTN LR", "HTN HGB", "DM LR", "DM HGB", "CKD LR", "CKD HGB"]]
for lab, e in [("강남 5-fold CV", "gangnam 5-fold CV"), ("신촌 5-fold CV", "sinchon 5-fold CV"), ("풀링 5-fold CV", "pooled 5-fold CV"),
               ("외부: 강남 학습→신촌", "train gangnam -> test sinchon"), ("외부: 신촌 학습→강남", "train sinchon -> test gangnam")]:
    r = base[base["eval"] == e].set_index(["disease", "model"]).AUC
    sd_ = base[base["eval"] == e].set_index(["disease", "model"]).AUC_sd
    rows.append([lab] + [(f"{r[(d, m)]:.3f}±{sd_[(d, m)]:.3f}" if sd_[(d, m)] == sd_[(d, m)] else f"{r[(d, m)]:.3f}") for d in DIS for m in ["LR", "HGB"]])
T2 = tbl(rows, [4.0 * cm] + [2.2 * cm] * 6)

# 표 3: 풀링 CV feature 탐색 (초록=유의한 개선)
rows, mk = [["feature set (clinic4에 추가)"] + DIS], {}
for i, s_ in enumerate(SETS, 1):
    rows.append([name(s_)] + [("%.3f±%.3f" % (g("AUC", s_, d), g("AUC_sd", s_, d))) if s_ == "clinic4" else fmt(g("AUC", s_, d), g("dAUC", s_, d), g("p_fdr_in_disease", s_, d), g("AUC_sd", s_, d)) for d in DIS])
    for j, d in enumerate(DIS, 1):
        if s_ != "clinic4" and g("p_fdr_in_disease", s_, d) < .05: mk[(i, j)] = GREEN
T3 = tbl(rows, [4.4 * cm, 4.3 * cm, 4.3 * cm, 4.3 * cm], mk)

# 표 4/5: 외부검증 (초록=유의한 개선, 빨강=유의한 악화)
EXT = []
for tr in ["gangnam", "sinchon"]:
    rows, mk = [["feature set (clinic4에 추가)"] + DIS], {}
    for i, s_ in enumerate(SETS, 1):
        rows.append([name(s_)] + [("%.3f" % E(tr, s_, d).AUC) if s_ == "clinic4" else fmt(E(tr, s_, d).AUC, E(tr, s_, d).dAUC, E(tr, s_, d).p_fdr) for d in DIS])
        for j, d in enumerate(DIS, 1):
            e = E(tr, s_, d)
            if s_ != "clinic4" and e.p_fdr < .05: mk[(i, j)] = GREEN if e.dAUC > 0 else RED
    EXT.append(tbl(rows, [5.2 * cm, 3.9 * cm, 3.9 * cm, 3.9 * cm], mk))


# ---- 데이터 분포: 두 코호트의 landmark별 Mann-Whitney 유의 개수 heatmap (그림 1) ----
sig = pd.read_excel(OUT / "data_distribution_significance.xlsx")
PRE = [("raw", "raw (landmark 값)"), ("pw_mean", "pw_mean (값 / 환자 구간평균)")]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
rows_ = [(c, d) for c in ["gangnam", "sinchon"] for d in DIS]
for ax, (pre_, ttl) in zip(axs.ravel(), PRE):
    M = np.array([[((sig.cohort == c) & (sig.preprocess == pre_) & (sig.disease == d) & (sig.measure == m) & (sig.q_fdr < .05)).sum() for m in MEAS_] for c, d in rows_])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=12)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > 7 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(MEAS_); ax.set_yticks(range(6)); ax.set_yticklabels([f"{'강남' if c == 'gangnam' else '신촌'} {d}" for c, d in rows_], fontsize=8)
    ax.axhline(2.5, color="white", lw=3); ax.set_title(ttl, fontsize=10)
plt.tight_layout(); fig.savefig(OUT / "data_distribution_summary.png", dpi=200); plt.close(fig)

# OR(+1SD당, 나이·성별·BMI 보정 로지스틱) 유의 landmark 수 heatmap (그림 1b)
orr = pd.read_excel(OUT / "data_distribution_significance.xlsx", sheet_name="odds_ratio")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (pre_, ttl) in zip(axs.ravel(), PRE):
    M = np.array([[((orr.cohort == c) & (orr.preprocess == pre_) & (orr.disease == d) & (orr.measure == m) & (orr.q_fdr < .05)).sum() for m in MEAS_] for c, d in rows_])
    ax.imshow(M, cmap="Oranges", vmin=0, vmax=12)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > 7 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(MEAS_); ax.set_yticks(range(6)); ax.set_yticklabels([f"{'강남' if c == 'gangnam' else '신촌'} {d}" for c, d in rows_], fontsize=8)
    ax.axhline(2.5, color="white", lw=3); ax.set_title(ttl, fontsize=10)
plt.tight_layout(); fig.savefig(OUT / "data_distribution_or_summary.png", dpi=200); plt.close(fig)

# ---- 어느 landmark가 유의한지: 질환×측정 행, landmark 12열 matrix (진한 파랑=두 코호트 모두, 하늘=강남만, 주황=신촌만, 화살표=질환군이 정상군보다 높음/낮음) ----
ABB = ["LD", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "FH", "PM"]
LMN = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
SG = sig.assign(sg=lambda x: (x.q_fdr < .05)).set_index(["preprocess", "disease", "measure", "landmark", "cohort"])
ARW = lambda v: "↑" if v > 0 else "↓"


# (전처리, 질환, 측정, landmark)의 상태: 두 코호트 모두 유의면 'B'(방향이 같으면 화살표, 다르면 ±), 한쪽만이면 'G'/'S', 없으면 ''
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

# 표: 두 코호트 모두 유의한 landmark 목록 (질환×측정 12행 × 전처리 4열)
CP = ParagraphStyle("cp", fontName="K", fontSize=7, leading=9)
trows = [["질환 · 측정", "raw", "pw_mean"]]
for d in DIS:
    for m in MEAS_:
        r = [f"{d} {m}"]
        for pre_, _ in PRE:
            c = [(ABB[j], cell(pre_, d, m, lm)) for j, lm in enumerate(LMN)]
            r.append(Paragraph(" ".join(f"{a}{ar}" for a, (k, ar) in c if k == "B") or "-", CP))
        trows.append(r)
TL = tbl(trows, [2.6 * cm] + [7 * cm] * 2)
TL.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "LEFT"), ("LINEBELOW", (0, 4), (-1, 4), 1, colors.black), ("LINEBELOW", (0, 8), (-1, 8), 1, colors.black)]))

# ---- 분포 기반 feature 조합 결과 (표 6~8) ----
cb = pd.read_excel(OUT / "clinic4_feature_combos.xlsx"); cbe = pd.read_excel(OUT / "clinic4_feature_combos_external.xlsx")
cb, cbe =[x[~x.feature_set.str.contains("기울기")] for x in (cb, cbe)]  # 신호 없는 기울기 조합은 제외
CSETS = list(dict.fromkeys(cb.feature_set))
cname = lambda s_: "clinic4 (baseline)" if s_ == "clinic4" else s_


# 조합 결과표: df에서 (AUC, ΔAUC, q) 셀을 만들고 유의한 셀은 초록/빨강
def combo_tbl(df, qcol):
    rows, mk = [["feature set (clinic4에 추가)"] + DIS], {}
    for i, s_ in enumerate(CSETS, 1):
        r = df[df.feature_set == s_].set_index("disease")
        rows.append([cname(s_)] + [("%.3f" % r.loc[d, "AUC"]) if s_ == "clinic4" else fmt(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, qcol]) for d in DIS] if "AUC_sd" not in r else [cname(s_)] + [("%.3f±%.3f" % (r.loc[d, "AUC"], r.loc[d, "AUC_sd"])) if s_ == "clinic4" else fmt(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, qcol], r.loc[d, "AUC_sd"]) for d in DIS])
        for j, d in enumerate(DIS, 1):
            if s_ != "clinic4" and r.loc[d, qcol] < .05: mk[(i, j)] = GREEN if r.loc[d, "dAUC"] > 0 else RED
    return tbl(rows, [4.8 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm], mk)


T6 = combo_tbl(cb, "p_fdr"); T7 = combo_tbl(cbe[cbe.train == "gangnam"], "p_fdr"); T8 = combo_tbl(cbe[cbe.train == "sinchon"], "p_fdr")


# ---- 로지스틱 회귀 계수 (표 9, 10): clinic4 vs clinic4+구간 합산 ----
co = pd.read_excel(OUT / "clinic4_coefficients.xlsx")
CK = lambda c, md, d, f: co[(co.cohort == c) & (co.model == md) & (co.disease == d) & (co.feature == f)].iloc[0]
cs = lambda r: "%.2f (%.2f-%.2f)" % (r.OR, r.ci_low, r.ci_high)
FEATS = ["sex(M=1)", "age", "height", "weight", "VAT_sum", "SAT_sum", "TAMA_sum", "AEC_sum"]
rows, mk = [["feature"] + DIS * 2], {}
for i, f in enumerate(FEATS, 1):
    r_ = [f]
    for jj, md in enumerate(["clinic4", "clinic4+구간 합산"]):
        for kk, d in enumerate(DIS):
            if f.endswith("_sum") and jj == 0: r_.append("-"); continue
            r = CK("pooled", md, d, f); r_.append(cs(r))
            if r.p < .05: mk[(i, 1 + jj * 3 + kk)] = GREEN if r.OR > 1 else RED
    rows.append(r_)


# 위에 그룹 행(모델 또는 코호트)을 얹은 2단 헤더 표: 아래 행은 질환(HTN/DM/CKD)
def tbl2(rows, mk, g1, g2):
    t = tbl([["", g1, "", "", g2, "", ""]] + rows, [2.4 * cm] + [2.5 * cm] * 6, {(r + 1, c): v for (r, c), v in mk.items()}, skip=1)
    t.setStyle(TableStyle([("SPAN", (1, 0), (3, 0)), ("SPAN", (4, 0), (6, 0)), ("BACKGROUND", (0, 1), (-1, 1), GREY), ("FONT", (0, 1), (-1, 1), "KB", t._fsz)])); return t


T9 = tbl2(rows, mk, "clinic4", "clinic4 + 구간 합산")
rows, mk = [["feature"] + DIS * 2], {}
for i, f in enumerate(FEATS[:1] + FEATS[4:], 1):
    r_ = [f]
    for jj, c in enumerate(["gangnam", "sinchon"]):
        for kk, d in enumerate(DIS):
            r = CK(c, "clinic4+구간 합산", d, f); r_.append(cs(r))
            if r.p < .05: mk[(i, 1 + jj * 3 + kk)] = GREEN if r.OR > 1 else RED
    rows.append(r_)
T10 = tbl2(rows, mk, "강남", "신촌")

# ---- 비율 feature (표 R1~R4): 같은 landmark·다른 조직 / 같은 조직·다른 landmark / 다른 조직·다른 landmark ----
RX = "clinic4_ratio_cross.xlsx"
rc = pd.read_excel(OUT / RX, sheet_name="consistency"); rp = pd.read_excel(OUT / RX, sheet_name="pooled_cv"); re_ = pd.read_excel(OUT / RX, sheet_name="external")
RSETS = list(dict.fromkeys(rp.feature_set))
rows = [["비율 종류 (후보 수)"] + DIS]
for t in dict.fromkeys(rc.tier):
    r = rc[rc.tier == t].set_index("disease"); rows.append([f"{t} ({r.n_cand.iloc[0]})"] + [f"{r.loc[d, 'n_consistent']} / {r.loc[d, 'n_cand']}" for d in DIS])
TR1 = tbl(rows, [6 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm])


def ratio_tbl(df):
    rows, mk = [["feature set (clinic4에 추가)"] + DIS], {}
    for i, s_ in enumerate(RSETS, 1):
        r = df[df.feature_set == s_].set_index("disease")
        rows.append([s_ if s_ != "clinic4" else "clinic4 (baseline)"] + [("%.3f±%.3f" % (r.loc[d, "AUC"], r.loc[d, "AUC_sd"])) if (s_ == "clinic4" and "AUC_sd" in r) else ("%.3f" % r.loc[d, "AUC"]) if s_ == "clinic4" else fmt(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, "p_fdr"], r.loc[d, "AUC_sd"] if "AUC_sd" in r else None) for d in DIS])
        for j, d in enumerate(DIS, 1):
            if s_ != "clinic4" and r.loc[d, "p_fdr"] < .05: mk[(i, j)] = GREEN if r.loc[d, "dAUC"] > 0 else RED
    return tbl(rows, [5.4 * cm, 4.0 * cm, 4.0 * cm, 4.0 * cm], mk)


TR2 = ratio_tbl(rp); TR3 = ratio_tbl(re_[re_.train == "gangnam"]); TR4 = ratio_tbl(re_[re_.train == "sinchon"])

OC = lambda c, pre_, d, m: int(((orr.cohort == c) & (orr.preprocess == pre_) & (orr.disease == d) & (orr.measure == m) & (orr.q_fdr < .05)).sum())
W = 17.4 * cm
import re
# ---- 새 정의(정상군 = 세 질환 모두 없음, 질환군 = 해당 질환이 있는 모든 환자)의 추가 결과 읽기 ----
psm_b = pd.read_excel(OUT / "clinic4_psm.xlsx", sheet_name="balance"); psm_p = pd.read_excel(OUT / "clinic4_psm.xlsx", sheet_name="pooled_cv"); psm_e = pd.read_excel(OUT / "clinic4_psm.xlsx", sheet_name="external")
nsp = pd.read_excel(OUT / "clinic4_next_steps.xlsx", sheet_name="pooled_cv"); nse = pd.read_excel(OUT / "clinic4_next_steps.xlsx", sheet_name="external"); nsor = pd.read_excel(OUT / "clinic4_next_steps.xlsx", sheet_name="cohort_adjusted_or")
qf = lambda q: "q&lt;0.01" if q < .005 else f"q={q:.2f}"
CTS = [x for x in CSETS if x.startswith("CT만")]; NCT = [x for x in CSETS if not x.startswith("CT만")]
KOS = {"HTN": "HTN", "DM": "DM", "CKD": "CKD"}
lr = base[base.model == "LR"]

# 표 1: 코호트별 분석 대상 수(정상군 + 해당 질환군)와 질환군(양성) 수
rows = [["코호트"] + [f"{d}: 대상 n (질환군, %)" for d in DIS]]
for lab_, e in [("강남", "gangnam 5-fold CV"), ("신촌", "sinchon 5-fold CV"), ("풀링(강남+신촌)", "pooled 5-fold CV")]:
    r = lr[lr["eval"] == e].set_index("disease")
    rows.append([lab_] + [f"{r.loc[d, 'n']:,} ({r.loc[d, 'n_pos']:,}, {100 * r.loc[d, 'n_pos'] / r.loc[d, 'n']:.1f}%)" for d in DIS])
T1 = tbl(rows, [3.6 * cm, 4.6 * cm, 4.6 * cm, 4.6 * cm])
npos = {c: {d: int(lr[(lr["eval"] == f"{c} 5-fold CV") & (lr.disease == d)].n_pos.iloc[0]) for d in DIS} for c in ["gangnam", "sinchon"]}

# 결과에서 계산하는 본문 수치
c0 = {d: cb[(cb.feature_set == "clinic4") & (cb.disease == d)].iloc[0] for d in DIS}
bp = lambda d: cb[(cb.disease == d) & cb.feature_set.isin(NCT) & (cb.feature_set != "clinic4")]
bestp = {d: bp(d).loc[bp(d).AUC.idxmax()] for d in DIS}
nsig = {d: bp(d)[bp(d).p_fdr < .05] for d in DIS}
be = lambda d, tr, st: cbe[(cbe.train == tr) & (cbe.disease == d) & (cbe.feature_set == st)].iloc[0]
PT = {"HTN": "은", "DM": "은", "CKD": "는"}
pool_sent = "; ".join((f"{d}{PT[d]} {len(nsig[d])}개 구성에서 유의(최대 {nsig[d].dAUC.max():+.3f}, q≥{nsig[d].p_fdr.min():.3f})" if len(nsig[d]) else f"{d}{PT[d]} 유의한 구성 없음(최대 {bestp[d].dAUC:+.3f}, {qf(bestp[d].p_fdr)})") for d in DIS)
ext_sig = cbe[cbe.feature_set.isin(NCT) & (cbe.feature_set != "clinic4") & (cbe.p_fdr < .05)]
ext_sent = "; ".join(f"{'강남→신촌' if tr == 'gangnam' else '신촌→강남'} {d}: 개선 {int(((x.dAUC > 0)).sum())}개, 악화 {int(((x.dAUC < 0)).sum())}개 구성" for (tr, d), x in ext_sig.groupby(["train", "disease"]))
both_or = orr[orr.preprocess == "pw_mean"].pivot_table(index=["disease", "measure", "landmark"], columns="cohort", values=["OR", "q_fdr"])
both_or = both_or[(both_or[("q_fdr", "gangnam")] < .05) & (both_or[("q_fdr", "sinchon")] < .05) & ((both_or[("OR", "gangnam")] > 1) == (both_or[("OR", "sinchon")] > 1))]
bo = both_or.reset_index(); bo["lm"] = bo.landmark.str.replace("_center", "")
both_txt = ", ".join(f"{d} {m} {'·'.join(g.lm)}" for (d, m), g in bo.groupby(["disease", "measure"], sort=False))
mwb = lambda pre_, d, m: int((sig[(sig.preprocess == pre_) & (sig.disease == d) & (sig.measure == m) & (sig.q_fdr < .05)].groupby("landmark").cohort.nunique() == 2).sum())
top_mw = ", ".join(f"{d} {m}({'raw' if pre_ == 'raw' else 'pw_mean'} {n}곳)" for n, pre_, d, m in sorted([(mwb(pre_, d, m), pre_, d, m) for pre_ in ["raw", "pw_mean"] for d in DIS for m in MEAS_], reverse=True)[:6])
ctb = {d: cb[(cb.disease == d) & cb.feature_set.isin(CTS)].pipe(lambda g: g.loc[g.AUC.idxmax()]) for d in DIS}
pb0 = {d: psm_p[(psm_p.disease == d) & (psm_p.feature_set == "clinic4")].iloc[0] for d in DIS}
pb = {d: psm_p[(psm_p.disease == d) & (psm_p.feature_set != "clinic4")].pipe(lambda g: g.loc[g.AUC.idxmax()]) for d in DIS}
smd_b = psm_b[psm_b.stage == "매칭 전"].SMD.abs().max(); smd_a = psm_b[psm_b.stage != "매칭 전"].SMD.abs().max()
hgb_ok = bool((lambda p: (p.LR >= p.HGB - 1e-9).all())(base.pivot_table(index=["eval", "disease"], columns="model", values="AUC")))
ra = lambda d, t: nsor[(nsor.disease == d) & (nsor.feature == t)].iloc[0]
sb = lambda d, sx: nsp[nsp.setting.str.startswith(f"② 성별 층화({sx}") & (nsp.disease == d) & (nsp.feature_set == "clinic4")].AUC.iloc[0]
sex_q = nsp[nsp.setting.str.startswith("② ") & (nsp.feature_set != "clinic4")].q.min()
sig1 = nsp[nsp.setting.str.startswith("① ") & (nsp.feature_set != "clinic4") & (nsp.q < .05)]


# 표 11: CT 변수만 사용(기본 정보 없이) vs baseline
def ct_tbl():
    rows = [["질환", "baseline AUC", "CT 변수만 구성", "CT만 AUC (Δ, q)", "강남→신촌 AUC (q)", "신촌→강남 AUC (q)"]]; mk = {}
    for i, d in enumerate(DIS, 1):
        b = ctb[d].feature_set; r_ = [d, f"{c0[d].AUC:.3f}", b, f"{ctb[d].AUC:.3f} ({ctb[d].dAUC:+.3f}, {qf(ctb[d].p_fdr)})".replace("&lt;", "<")]
        for j, tr in enumerate(["gangnam", "sinchon"], 4):
            e1, e0 = be(d, tr, b), be(d, tr, "clinic4"); r_.append(f"{e0.AUC:.3f} → {e1.AUC:.3f} ({qf(e1.p_fdr)})".replace("&lt;", "<"))
            if e1.p_fdr < .05: mk[(i, j + 1)] = GREEN if e1.dAUC > 0 else RED
        if ctb[d].p_fdr < .05: mk[(i, 3)] = GREEN if ctb[d].dAUC > 0 else RED
        rows.append(r_)
    return tbl(rows, [1.4 * cm, 2.2 * cm, 4.4 * cm, 3.4 * cm, 3.0 * cm, 3.0 * cm], mk)


# 표 12: 성향점수 매칭 후 baseline vs best
def psm_tbl():
    rows = [["질환", "매칭 후 n (양성)", "baseline AUC", "best 구성", "best AUC (Δ, q)", "강남→신촌 AUC (q)", "신촌→강남 AUC (q)"]]; mk = {}
    for i, d in enumerate(DIS, 1):
        b0, b1 = pb0[d], pb[d]; r_ = [d, f"{int(b0.n_matched)} ({int(b0.n_pos)})", f"{b0.AUC:.3f}", b1.feature_set, f"{b1.AUC:.3f} ({b1.dAUC:+.3f}, {qf(b1.q)})".replace("&lt;", "<")]
        if b1.q < .05: mk[(i, 4)] = GREEN
        for j, tr in enumerate(["gangnam", "sinchon"], 5):
            e0 = psm_e[(psm_e.train == tr) & (psm_e.disease == d) & (psm_e.feature_set == "clinic4")].iloc[0]; e1 = psm_e[(psm_e.train == tr) & (psm_e.disease == d) & (psm_e.feature_set == b1.feature_set)].iloc[0]
            r_.append(f"{e0.AUC:.3f} → {e1.AUC:.3f} ({qf(e1.q)})".replace("&lt;", "<"))
            if e1.q < .05: mk[(i, j)] = GREEN if e1.dAUC > 0 else RED
        rows.append(r_)
    return tbl(rows, [1.2 * cm, 2.4 * cm, 1.9 * cm, 3.3 * cm, 3.0 * cm, 2.8 * cm, 2.8 * cm], mk)


# 표 13: 성별 분리 baseline vs best
def sex_tbl():
    rows = [["질환 · 성별", "정상군 / 질환군 (명)", "baseline AUC", "best 구성", "best AUC (Δ, q)", "강남→신촌 AUC (q)", "신촌→강남 AUC (q)"]]; mk = {}
    i = 0
    for d in DIS:
        for sx in ["남", "여"]:
            g_ = nsp[nsp.setting.str.startswith(f"② 성별 층화({sx}") & (nsp.disease == d)]
            if g_.empty: continue
            i += 1; b0 = g_[g_.feature_set == "clinic4"].iloc[0]; b1 = g_[g_.feature_set != "clinic4"].pipe(lambda x: x.loc[x.AUC.idxmax()])
            nn = re.search(r"n=(\d+), 양성 (\d+)", b0.setting)
            r_ = [f"{d} · {sx}", f"{int(nn.group(1)) - int(nn.group(2))} / {nn.group(2)}", f"{b0.AUC:.3f}", b1.feature_set, f"{b1.AUC:.3f} ({b1.dAUC:+.3f}, {qf(b1.q)})".replace("&lt;", "<")]
            for j, tr in enumerate(["gangnam", "sinchon"], 5):
                e = nse[(nse.setting == f"② 성별 층화({sx})") & (nse.train == tr) & (nse.disease == d)]
                if e.empty: r_.append("-"); continue
                e0 = e[e.feature_set == "clinic4"].iloc[0]; e1 = e[e.feature_set == b1.feature_set].iloc[0]; r_.append(f"{e0.AUC:.3f} → {e1.AUC:.3f} ({qf(e1.q)})".replace("&lt;", "<"))
                if e1.q < .05: mk[(i, j)] = GREEN if e1.dAUC > 0 else RED
            rows.append(r_)
    return tbl(rows, [1.8 * cm, 2.4 * cm, 1.9 * cm, 3.2 * cm, 2.9 * cm, 2.6 * cm, 2.6 * cm], mk)


# 표 14: 병원·장비 보정 OR
def adj_tbl():
    rows = [["질환", "feature", "병원 보정 OR (95% CI)", "p", "강남 OR", "신촌 OR", "교호작용 p"]]; mk = {}
    for i, r in enumerate(nsor.itertuples(), 1):
        rows.append([r.disease, r.feature.replace("_sum", " 합산"), f"{r.OR_adj:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f})", "%.3f" % r.p_adj, f"{r.OR_gangnam:.2f}", f"{r.OR_sinchon:.2f}", "%.3f" % r.p_interaction])
        if r.p_adj < .05: mk[(i, 3)] = GREEN
    return tbl(rows, [1.6 * cm, 2.6 * cm, 4.2 * cm, 1.8 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm], mk)


b_txt = ", ".join(f"{d} {c0[d].AUC:.3f}±{c0[d].AUC_sd:.3f}" for d in DIS)
summary = Table([[Paragraph(
    "<b>요약</b><br/>"
    f"① 정상군(HTN·DM·CKD 모두 없음) 대 질환군(해당 질환이 있는 모든 환자, 동반질환 포함)에서 sex·age·height·weight(clinic4)만으로 만든 baseline의 풀링 5-fold CV AUC(fold별 평균±SD)는 {b_txt}이다(LR, 하이퍼파라미터는 nested CV로 선택).<br/>"
    f"② 풀링 CV에서 landmark 체성분(VAT/SAT/TAMA/AEC)을 추가하면 {pool_sent}.<br/>"
    f"③ 강남↔신촌 외부검증에서 baseline 대비 유의한 변화(q&lt;0.05)는 {ext_sent or '없음'}이다. 풀링에서 유의했던 개선은 다른 병원에서 재현되지 않았다(DM 강남→신촌 {be('DM', 'gangnam', bestp['DM'].feature_set).dAUC:+.3f}, 신촌→강남 {be('DM', 'sinchon', bestp['DM'].feature_set).dAUC:+.3f}; 해당 구성 기준).<br/>"
    f"④ 두 코호트 모두 일관된 차이는 {top_mw} 등에서 크고, 나이·성별·BMI 보정 후 두 병원에서 모두 유의하고 방향이 같은 위치는 {len(both_or)}곳({both_txt})이다.<br/>"
    f"⑤ clinic4 없이 CT 변수만 쓰면 AUC는 HTN {ctb['HTN'].AUC:.3f}, DM {ctb['DM'].AUC:.3f}, CKD {ctb['CKD'].AUC:.3f}로 baseline과 비교해 HTN은 낮고({ctb['HTN'].dAUC:+.3f}) DM·CKD는 비슷하거나 약간 높으나 유의하지 않다. 나이·성별·BMI로 성향점수 매칭한 1:1 언더샘플링에서는 baseline이 무작위 수준({pb0['HTN'].AUC:.2f}, {pb0['DM'].AUC:.2f}, {pb0['CKD'].AUC:.2f})으로 내려가고 CT 변수가 풀링 CV에서 AUC를 올린다(HTN {pb['HTN'].AUC:.2f}, DM {pb['DM'].AUC:.2f}, CKD {pb['CKD'].AUC:.2f}; 모두 q&lt;0.05). 외부검증에서는 신촌→강남 방향에서만 재현된다.<br/>"
    "⑥ 따라서 landmark 체성분은 clinic4와 독립적인 연관을 보이지만(정보가 없다는 뜻은 아님), 병원 간에 일관되게 재현되는 성능 향상은 확인하지 못했다.", BOX)]], colWidths=[W])
summary.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .8, colors.HexColor("#2F6DB5")), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7FB")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
OR_IMG = lambda f: Image(str(OUT / f), width=W, height=W * 1440 / 2640)

S = [Paragraph("clinic4 baseline 및 체성분 feature 추가 효과", H1),
     Paragraph("HTN · DM · CKD 질환 유무 분류 / 강남·신촌 코호트 / 2026-10-14", CAP), Spacer(1, 4), summary,
     Paragraph("1. 목적", H2),
     Paragraph("(1) sex, age, height, weight(clinic4)만으로 HTN/DM/CKD 유무를 분류하는 baseline 성능을 확립하고, (2) 보유한 landmark 체성분 값을 추가해 AUC를 개선할 수 있는지, (3) 개선이 통계적으로 유의하며 다른 코호트에서도 유지되는지를 확인한다. FPCA 기반 feature는 제외했다. 비교 대상은 정상군(세 질환이 모두 없는 환자)과 질환군(해당 질환이 있는 모든 환자, 다른 질환을 동반한 경우 포함)이다.", B),
     Paragraph("2. 방법", H2),
     Paragraph("<b>데이터.</b> 강남·신촌 landmark_filtered.xlsx의 환자 메타데이터와 12개 해부학적 landmark(liver dome ~ inferior pubic margin)의 slice 위치에서 읽은 VAT(내장지방), SAT(피하지방), TAMA(근육 전체), AEC(몸이 두꺼울수록 CT가 더 세게 쏘는 방사선 세기로, 몸 크기·조성의 간접 지표) 값을 사용했다. 입력 변수나 라벨이 결측인 환자와 landmark 순서가 이상한 환자는 제외했다. 질환마다 정상군(세 질환 모두 없음)과 해당 질환이 있는 환자만 사용한다. '풀링'은 강남과 신촌 환자를 합친 것이며 코호트 정보는 모델 입력으로 쓰지 않았다.", B),
     Paragraph("<b>모델·평가.</b> Logistic Regression(LR)과 HistGradientBoosting(HGB)을 비교했고, feature 탐색은 LR(L2)로 수행했다. 하이퍼파라미터 C(0.001~10, 로그 간격 9개)는 모든 모델(clinic4 포함)에서 학습 fold 내부 5-fold AUC로 선택했고 평가 fold는 선택에 쓰지 않았다(nested CV). 5-fold CV를 3회 반복(fold 분할 seed 변경)해 얻은 <b>fold별 AUC 15개의 평균±표준편차</b>를 보고했으며, 층화는 질환 라벨×코호트로 했다. 외부검증은 한 코호트로 학습(C는 같은 방식으로 학습 코호트 내부에서 선택)해 다른 코호트에서 평가했다. 민감도·특이도는 Youden 지수가 최대인 기준점에서 구했고 외부검증의 기준점은 학습 코호트의 5-fold 예측으로 정했다.", B),
     Paragraph("<b>유의성.</b> 풀링 CV에서 clinic4 대비 AUC 차이는 같은 fold끼리 짝지은 fold별 AUC 차이에 Nadeau-Bengio 보정 paired t 검정으로 검정했고, 외부검증은 같은 환자에 대한 DeLong paired test로 검정했다. 두 경우 모두 질환별로 feature set 간 Benjamini-Hochberg FDR 보정 q값을 보고했다. 표의 괄호는 (ΔAUC, q)이며, <b>별표·초록 표시는 FDR 보정 후 q&lt;0.05</b>를 뜻한다. 같은 환자 데이터로 여러 조합을 시험했으므로 보정 전 p값만으로는 우연일 가능성을 배제할 수 없다.", B),
     Paragraph("3. 대상", H2), Paragraph("표 1. 코호트별 분석 대상 수(정상군 + 해당 질환군)와 질환군 수", CAPT), T1, Spacer(1, 6),
     Paragraph("4. 데이터 분포: 질환군 vs 정상군", H2),
     Paragraph("landmark별(12개) 값을 질환군과 정상군으로 나누어 Mann-Whitney U 검정(질환·측정별 landmark 12개 BH-FDR)을 했다. 비교한 값은 raw(landmark slice 값), pw_mean(값 / 환자 liver dome~pubis 구간 평균)이다. 이 비교는 성별·체격을 보정하지 않은 단변량 검정이므로 raw 값의 차이에는 체격 교란이 섞일 수 있고, pw_mean은 체격(진폭) 영향을 줄이고 곡선 모양을 본다.", B),
     Paragraph("그림 1. 질환군 vs 정상군 유의한 landmark 수 (12개 중, FDR q&lt;0.05)", CAPT),
     Image(str(OUT / "data_distribution_summary.png"), width=W, height=W * 3.8 / 10),
     Paragraph(f"두 코호트 모두 유의한 landmark 수가 많은 항목은 {top_mw} 순이다. TAMA는 병원 간 일관성이 낮고 CKD는 환자 수가 적어(강남 {npos['gangnam']['CKD']}명, 신촌 {npos['sinchon']['CKD']}명) 검정력이 제한된다.", CAP),
     Paragraph("<b>나이·성별·BMI 보정 후 Odds Ratio.</b> 위 검정은 보정이 없으므로, landmark 값(표준화, +1 SD)을 나이·성별·BMI와 함께 넣은 로지스틱 회귀의 OR을 landmark별로 구했다(질환·측정별 landmark 12개 BH-FDR). raw와 pw_mean 두 가지 값으로 계산했다.", B),
     Paragraph("그림 1b. 보정 OR이 유의한 landmark 수 (12개 중, FDR q&lt;0.05; 코호트별 개별 분석)", CAPT),
     Image(str(OUT / "data_distribution_or_summary.png"), width=W, height=W * 3.8 / 10),
     Paragraph(f"보정하면 단변량 검정의 신호가 줄었다. 두 병원 모두에서 유의하고 방향이 같은 위치는 pw_mean에서 {len(both_or)}곳({both_txt}; 모두 OR>1)이었고 raw에서는 {int(((orr.preprocess == 'raw') & (orr.q_fdr < .05)).groupby([orr.disease, orr.measure, orr.landmark]).sum().gt(1).sum())}곳이었다. 위치별 OR 곡선은 부록 그림 C1~C4, 수치는 data_distribution_significance.xlsx의 odds_ratio 시트에 있다. 이 결과는 보정 변수가 나이·성별·BMI뿐이고 landmark 12개 × 질환·측정 12개를 따로 검정했으므로 탐색적으로 해석해야 한다.", CAP),
     Paragraph("그림 1c. landmark별 보정 OR (+1 SD당, 95% CI) [raw]. 패널 = landmark, 행 = 질환·측정, 빨강 = 강남, 파랑 = 신촌, 채운 점 = FDR q&lt;0.05", CAPT), OR_IMG("data_distribution_or_by_landmark_raw.png"),
     Paragraph("그림 1d. landmark별 보정 OR [pw_mean]", CAPT), OR_IMG("data_distribution_or_by_landmark_pw_mean.png"),
     Paragraph("표 A. 두 코호트 모두 유의한 landmark (FDR q&lt;0.05). ↑ = 질환군이 정상군보다 높음, ↓ = 낮음, ± = 두 코호트의 방향이 다름. LD=liver dome, FH=femoral head, PM=inferior pubic margin, '-'=없음", CAPT), TL, Spacer(1, 4),
     Paragraph("전체 landmark별 분포는 부록 그림 A1~A4, 유의 위치 matrix는 부록 그림 B1에 있다.", CAP),
     Paragraph("5. clinic4 baseline", H2), Paragraph("표 2. clinic4(4개 변수)만 사용한 AUC", CAPT), T2,
     Paragraph(f"표 2는 기본 설정(LR C=1, HGB)의 값이며 표 3의 clinic4(C 튜닝)와 소수 셋째 자리에서 조금 다르다. {'모든 평가에서 LR이 HGB와 같거나 높았다. 변수가 4개뿐이어서 비선형 모델의 이점이 없으며, 이후 분석은 LR로 진행했다.' if hgb_ok else '평가에 따라 LR과 HGB의 우열이 다르며 이후 분석은 단순한 LR로 진행했다.'} 신촌 CKD가 가장 낮다.", CAP),
     Paragraph("6. landmark 체성분 추가 효과: 풀링 5-fold CV", H2),
     Paragraph("표 3. clinic4 대비 AUC (초록 = FDR q&lt;0.05로 유의하게 개선)", CAPT), T3,
     Paragraph("(표 3은 초기 feature 탐색 결과이다.) " + "; ".join(f"{d}: 최대 {best[(best.disease == d) & (best.feature_set != 'clinic4')].dAUC.max():+.3f}(q={best[(best.disease == d) & (best.feature_set != 'clinic4')].sort_values('dAUC').iloc[-1].p_fdr_in_disease:.3f})" for d in DIS) + ". clinic4와 추가 구성 모두 같은 nested 절차로 튜닝했으므로 하이퍼파라미터 선택에 따른 낙관 편향은 제거되었다.", CAP),
     Paragraph("7. 코호트 간 외부검증", H2),
     Paragraph("표 4. 강남으로 학습 → 신촌에서 평가 (초록 = 유의한 개선, 빨강 = 유의한 악화)", CAPT), EXT[0], Spacer(1, 6),
     Paragraph("표 5. 신촌으로 학습 → 강남에서 평가", CAPT), EXT[1], Spacer(1, 6),
     Paragraph("그림 2. clinic4 vs clinic4 + landmark raw(L1,L3,L5): 풀링 CV와 외부검증 (* = clinic4 대비 FDR q&lt;0.05)", CAPT),
     Image(str(OUT / "clinic4_feature_search.png"), width=W, height=W * 3.2 / 10),
     Paragraph("강남→신촌과 신촌→강남에서 landmark를 더한 구성의 변화 방향이 질환·방향마다 다르다. 자세한 수치는 표 4·5와 표 7·8을 참고한다.", CAP),
     Paragraph("8. 분포 분석 기반 feature 조합", H2),
     Paragraph("그림 1에서 두 코호트가 일관되게 유의했던 feature(VAT raw, SAT·AEC의 pw_mean 12개 landmark + 구간평균)를 단독 또는 조합해 clinic4에 더했다. '분포 기반(질환별)'은 HTN·DM에 VAT raw+SAT pw+AEC pw, CKD에 AEC pw만 쓰는 구성이다. 'in-fold 유의 선택'은 raw·pw_mean 4종(96개 후보) 중 <b>학습 fold(외부검증에서는 학습 코호트)에서만</b> Mann-Whitney BH-FDR&lt;0.05인 feature를 골라 쓰므로 평가 데이터가 선택에 관여하지 않는다. 다른 조합은 분포 그림을 전체 데이터에서 본 뒤 정했으므로 풀링 CV 결과는 약간 낙관적일 수 있다. 모델은 앞과 동일한 nested LR이다. 표에는 clinic4 없이 CT 변수만 입력한 'CT만' 구성도 함께 실었다.", B),
     Paragraph("표 6. 분포 기반 조합: 풀링 5-fold CV (초록 = 유의한 개선, 빨강 = 유의한 악화)", CAPT), T6, Spacer(1, 6),
     Paragraph("표 7. 강남으로 학습 → 신촌에서 평가 (초록 = 유의한 개선, 빨강 = 유의한 악화)", CAPT), T7, Spacer(1, 6),
     Paragraph("표 8. 신촌으로 학습 → 강남에서 평가", CAPT), T8,
     Paragraph(f"풀링 CV에서 clinic4에 구성을 더하면 {pool_sent}. 외부검증에서 유의한 변화는 {ext_sent or '없음'}이다. 분포가 코호트 간에 일관되어도 이를 입력으로 한 예측 개선이 코호트 간에 일관되지는 않았다.", CAP),
     Paragraph("9. 비율 feature: 조직·landmark 조합별 효과", H2),
     Paragraph("비율 feature log((a+1)/(b+1))를 만들 때 분자와 분모의 조직·landmark를 어떻게 고르는지에 따라 세 가지로 나눠 비교했다: (1) 같은 landmark에서 다른 조직(예: VAT/SAT@L1, 72개), (2) 같은 조직에서 다른 landmark(예: VAT@T11/PM, 264개), (3) 다른 조직·다른 landmark(예: VAT@T11/SAT@L4, 792개). 조직은 VAT, SAT, TAMA, AEC 4종이며 역수는 부호만 달라 한 방향만 만들었다. <b>일관성</b>은 코호트별 Mann-Whitney BH-FDR&lt;0.05이면서 두 코호트의 방향이 같은 비율 수이다. <b>clinic4 추가 효과</b>는 <b>학습 fold(외부검증은 학습 코호트)에서 단변량 AUC가 0.5에서 가장 먼 상위 3개</b>만 clinic4에 더해 평가 데이터가 선택에 관여하지 않게 했고, 모델과 검정은 앞 절과 같다.", B),
     Paragraph("표 R1. 두 코호트 모두 유의하고 방향이 같은 비율 수 / 후보 수 (보정 없는 단변량 검정)", CAPT), TR1, Spacer(1, 6),
     Paragraph("표 R2. 비율 상위 3개 추가: 풀링 5-fold CV (초록 = 유의한 개선)", CAPT), TR2, Spacer(1, 6),
     Paragraph("표 R3. 강남으로 학습 → 신촌에서 평가 (초록 = 유의한 개선, 빨강 = 유의한 악화)", CAPT), TR3, Spacer(1, 6),
     Paragraph("표 R4. 신촌으로 학습 → 강남에서 평가", CAPT), TR4,
     Paragraph("일관된 비율이 후보의 상당수를 차지하지만 이 검정은 나이·성별·체격을 보정하지 않아 VAT/TAMA처럼 체격과 연관된 비율에 교란이 섞이므로 개수 자체를 근거로 삼기 어렵다. " + "풀링 CV에서는 " + "; ".join(f"{d} {rp[(rp.disease == d) & (rp.feature_set != 'clinic4')].dAUC.min():+.3f}~{rp[(rp.disease == d) & (rp.feature_set != 'clinic4')].dAUC.max():+.3f}(최소 {qf(rp[(rp.disease == d) & (rp.feature_set != 'clinic4')].p_fdr.min())})" for d in DIS) + f"이고 외부검증에서 유의한 변화는 {len(re_[(re_.feature_set != 'clinic4') & (re_.p_fdr < .05)])}건이다. 조직과 landmark를 교차한 비율은 탐색 공간만 넓힐 뿐 일반화되는 개선을 주지 않는지 표 R2~R4로 확인할 수 있다. 선택된 상위 3개 feature는 xlsx의 selected 시트에 있다.", CAP),
     Paragraph("10. 체성분 구간 합산 값과 회귀 계수", H2),
     Paragraph("liver dome~inferior pubic margin 구간에서 VAT, SAT, TAMA, AEC의 slice 값을 합산한 총량(VAT_sum, SAT_sum, TAMA_sum, AEC_sum)을 clinic4에 더했다. 풀링 nested CV에서 AUC는 " + ", ".join(f"{d} {cb[(cb.feature_set == '+구간 합산(4종)') & (cb.disease == d)].iloc[0].AUC:.3f}±{cb[(cb.feature_set == '+구간 합산(4종)') & (cb.disease == d)].iloc[0].AUC_sd:.3f}({cb[(cb.feature_set == '+구간 합산(4종)') & (cb.disease == d)].iloc[0].dAUC:+.3f}, {qf(cb[(cb.feature_set == '+구간 합산(4종)') & (cb.disease == d)].iloc[0].p_fdr)})" for d in DIS) + "이다. 외부검증에서 신촌→강남은 " + ", ".join(f"{d} {be(d, 'sinchon', '+구간 합산(4종)').dAUC:+.3f}({qf(be(d, 'sinchon', '+구간 합산(4종)').p_fdr)})" for d in DIS) + ", 강남→신촌은 " + ", ".join(f"{d} {be(d, 'gangnam', '+구간 합산(4종)').dAUC:+.3f}({qf(be(d, 'gangnam', '+구간 합산(4종)').p_fdr)})" for d in DIS) + "이다.", B),
     Paragraph("표 9. 로지스틱 회귀 OR (95% CI), 풀링, 연속형은 1 SD당, sex는 남성 vs 여성 (초록 = p&lt;0.05로 위험 증가, 빨강 = 위험 감소)", CAPT), T9, Spacer(1, 6),
     Paragraph("표 10. 코호트별 OR (95% CI), clinic4 + 구간 합산 모델", CAPT), T10,
     Paragraph("합산 변수 중 풀링에서 유의한 것은 " + (", ".join(f"{r.disease} {r.feature.replace('_sum', '')}(OR {r.OR:.2f})" for r in co[(co.cohort == "pooled") & (co.model == "clinic4+구간 합산") & (co.p < .05) & co.feature.str.endswith("_sum")].itertuples()) or "없음") + "이다. 코호트별로는 " + ", ".join(f"{d} VAT_sum 강남 {co[(co.cohort == 'gangnam') & (co.model == 'clinic4+구간 합산') & (co.disease == d) & (co.feature == 'VAT_sum')].OR.iloc[0]:.2f} / 신촌 {co[(co.cohort == 'sinchon') & (co.model == 'clinic4+구간 합산') & (co.disease == d) & (co.feature == 'VAT_sum')].OR.iloc[0]:.2f}" for d in DIS) + f"로 병원마다 효과 크기가 다르다. sex의 HTN OR은 강남 {co[(co.cohort == 'gangnam') & (co.model == 'clinic4+구간 합산') & (co.disease == 'HTN') & (co.feature == 'sex(M=1)')].OR.iloc[0]:.2f}, 신촌 {co[(co.cohort == 'sinchon') & (co.model == 'clinic4+구간 합산') & (co.disease == 'HTN') & (co.feature == 'sex(M=1)')].OR.iloc[0]:.2f}로 방향이 반대여서 코호트 간 이질성을 보여준다. VIF는 최대 {pd.to_numeric(co.VIF, errors='coerce').max():.1f}로 공선성 문제는 크지 않았다.", CAP),
     Paragraph("11. clinic4 없이 CT 변수만 사용", H2),
     Paragraph("성별·나이·키·몸무게를 입력에서 빼고 landmark 체성분 값만으로 질환을 분류했다. 구성은 구간 합산·평균, landmark 값, pw_mean, 분포 기반, raw+pw 전체이며 표 11에는 풀링 CV AUC가 가장 높은 구성을 실었다(전체는 표 6~8과 부록).", B),
     Paragraph("표 11. clinic4 vs CT 변수만 사용 (q = baseline 대비 FDR 보정 p값)", CAPT), ct_tbl(),
     Paragraph(f"풀링 CV에서 CT만 구성의 AUC는 HTN {ctb['HTN'].AUC:.3f}({ctb['HTN'].dAUC:+.3f}), DM {ctb['DM'].AUC:.3f}({ctb['DM'].dAUC:+.3f}), CKD {ctb['CKD'].AUC:.3f}({ctb['CKD'].dAUC:+.3f})로 baseline보다 유의하게 높은 구성은 없다. 외부검증에서는 HTN과 DM에서 CT만 구성이 baseline보다 유의하게 낮아지는 경우가 많다(표 7·8). 즉 CT 변수만으로 기본 정보보다 유의하게 나은 성능은 얻지 못했다.", CAP),
     Paragraph("12. 성향점수 매칭 1:1 언더샘플링", H2),
     Paragraph(f"정상군이 질환군보다 많아 단순 무작위로 줄이는 대신, 같은 병원 안에서 나이·성별·BMI 로지스틱 성향점수가 가장 가까운 정상군을 1:1로 짝지었다(최근접, 비복원, caliper 0.2 SD of logit; 환자 순서를 섞어 5회 반복). 매칭 전 최대 표준화 평균차(SMD)는 {smd_b:.2f}였고 매칭 후 {smd_a:.2f} 이하로 균형이 맞았다(부록 표 P1). 매칭 자료에서 풀링 5-fold CV(3회 반복)와 병원별 매칭 자료로 외부검증을 수행했다(AUC는 매칭 5회 평균).", B),
     Paragraph("표 12. 성향점수 매칭 후 clinic4 vs 가장 좋은 구성 (q = FDR 보정 p값)", CAPT), psm_tbl(),
     Paragraph(f"매칭하면 clinic4의 구분 능력은 사라져 AUC가 무작위 수준({pb0['HTN'].AUC:.2f}, {pb0['DM'].AUC:.2f}, {pb0['CKD'].AUC:.2f})이 되고, CT 변수를 추가하면 풀링 CV에서 HTN {pb['HTN'].AUC:.3f}, DM {pb['DM'].AUC:.3f}, CKD {pb['CKD'].AUC:.3f}까지 오른다(모두 {qf(max(pb[d].q for d in DIS))} 이하). 즉 나이·성별·BMI가 같은 사람끼리는 CT 값이 질환 구분에 도움이 된다. 그러나 외부검증에서는 강남에서 학습한 모델은 재현되지 않고 신촌에서 학습한 모델에서만 DM·CKD가 크게, HTN이 소폭 재현된다.", CAP),
     Paragraph("13. 병원 표준화·성별 분리·병원 보정", H2),
     Paragraph(f"<b>① 병원별 표준화.</b> 병원마다 값의 수준이 달라(AEC 강남 약 300, 신촌 약 220) 추가 변수를 각 병원 안에서 평균 0, 표준편차 1로 맞춘 뒤 다시 평가했다. 풀링 CV에서 유의한 개선은 {', '.join(sorted(set(sig1.disease))) or '없음'}에서만 나타났다(최대 {sig1.dAUC.max() if len(sig1) else 0:+.3f}). <b>② 성별 분리.</b> 남녀를 따로 모델링하면(성별 변수 제외, 값은 병원×성별 안에서 표준화) baseline AUC가 남성에서 낮고(HTN {sb('HTN', '남'):.2f} 대 {sb('HTN', '여'):.2f}, DM {sb('DM', '남'):.2f} 대 {sb('DM', '여'):.2f}, CKD {sb('CKD', '남'):.2f} 대 {sb('CKD', '여'):.2f}) CT 변수 추가의 풀링 CV 개선은 어느 성별에서도 유의하지 않다(q≥{sex_q:.2f}). <b>③ 병원 보정.</b> 병원 고정효과와 장비(manufacturer_model) 군집 로버스트 표준오차를 쓴 로지스틱 회귀에서 나이·성별·키·몸무게를 보정해도 여러 합산 변수가 유의하지만(표 14) 병원마다 효과 크기가 달라 교호작용이 유의하다(예: HTN VAT 합산 강남 {ra('HTN', 'VAT_sum').OR_gangnam:.2f}, 신촌 {ra('HTN', 'VAT_sum').OR_sinchon:.2f}). 병원이 2곳이므로 임의효과 대신 고정효과를 사용했다.", B),
     Paragraph("표 13. 성별 분리: baseline vs 가장 좋은 구성 (q = FDR 보정 p값)", CAPT), sex_tbl(), Spacer(1, 6),
     Paragraph("표 14. 병원·장비 보정 OR (1 SD당; 교호작용 p = 효과가 병원마다 다른지)", CAPT), adj_tbl(),
     Paragraph("14. 결론 및 제언", H2),
     Paragraph(f"clinic4 baseline은 풀링 LR 기준 {b_txt}(fold별 평균±SD)이다. 풀링 CV에서 landmark 체성분은 {pool_sent}. 이 개선은 코호트 간 외부검증에서 재현되지 않았고 일부는 악화되었다. clinic4 없이 CT 변수만 쓰면 baseline보다 유의하게 나은 성능을 얻지 못하며, 나이·성별·BMI를 맞춘 성향점수 매칭에서는 풀링 CV에서 세 질환 모두 CT 변수가 AUC를 올리지만 외부검증은 신촌→강남 방향에서만 재현된다. 병원·장비를 보정해도 연관은 남지만 병원마다 크기가 달라, 현재 형태의 체성분 값만으로는 일반화 가능한 개선을 주장하기 어렵다(정보가 없다는 뜻은 아님). 다음 단계로 (1) 장비·촬영 조건 보정, (2) CKD 환자 수 확대, (3) 질환 정의(진단 기준) 확인을 제안한다.", B),
     Paragraph("15. 한계", H2),
     Paragraph(f"CKD는 환자 수가 적어(강남 {npos['gangnam']['CKD']}명, 신촌 {npos['sinchon']['CKD']}명) AUC의 불확실성이 크다. 질환군에 동반질환자를 포함하므로 질환 간 겹침이 있다. 풀링 CV의 검정은 fold 15개의 AUC 차이에 기반하므로 검정력이 낮고(보정 t), 외부검증의 DeLong 검정은 환자 단위로 수행했다. 다중비교 보정은 feature set 수에 대해서만 했으며 질환 간 보정은 포함하지 않았다. best 구성과 분포 기반 조합은 전체 데이터를 본 뒤 정한 것이어서(in-fold 선택 제외) 풀링 CV 결과가 낙관적일 수 있어 외부검증으로 확인했다. AUC 변화량의 신뢰구간은 보고하지 않았다. 단면 자료의 현재 유병 상태 분류이며 연관성이지 인과는 아니다. 하이퍼파라미터 탐색은 L2의 C만 했고 L1은 쓰지 않았다.", B),
     Paragraph("부록 표 P1. 성향점수 매칭 전후 공변량 균형(SMD)", CAPT),
     tbl([["질환", "변수", "SMD 매칭 전", "SMD 매칭 후(seed 0)"]] + [[r.disease, r.variable, "%+.3f" % r.SMD_pre, "%+.3f" % r.SMD_post] for r in psm_b.pivot_table(index=["disease", "variable"], columns="stage", values="SMD").rename(columns={"매칭 전": "SMD_pre", "매칭 후(seed0)": "SMD_post"}).reset_index().itertuples()], [3 * cm, 4 * cm, 4 * cm, 4.5 * cm]),
     NextPageTemplate("land"), PageBreak(), Paragraph("부록. landmark별 질환군 vs 정상군 분포와 보정 OR (A: 선 = median, 음영 = IQR, 위 표시 = Mann-Whitney BH-FDR)", H2)]
LW = 27 * cm
for k, (c, md, lab) in enumerate([("gangnam", "", "raw"), ("sinchon", "", "raw"), ("gangnam", "_pw_mean", "pw_mean"), ("sinchon", "_pw_mean", "pw_mean")], 1):
    S += [Paragraph(f"그림 A{k}. {'강남' if c == 'gangnam' else '신촌'} [{lab}]", CAPT), Image(str(OUT / f"data_distribution_{c}{md}.png"), width=LW, height=LW * 1620 / 2970)]
    if k < 4: S += [PageBreak()]
for k, (f, lab) in enumerate([("data_distribution_matrix_a.png", "raw, pw_mean")], 1):
    S += [PageBreak(), Paragraph(f"그림 B{k}. 유의한 landmark 위치 matrix [{lab}]", CAPT), Image(str(OUT / f), width=LW, height=LW * 5.4 / 14)]
for k, (c, md, lab) in enumerate([("gangnam", "", "raw"), ("sinchon", "", "raw"), ("gangnam", "_pw_mean", "pw_mean"), ("sinchon", "_pw_mean", "pw_mean")], 1):
    S += [PageBreak(), Paragraph(f"그림 C{k}. 보정 OR (+1 SD당, 95% CI, 나이·성별·BMI 보정) {'강남' if c == 'gangnam' else '신촌'} [{lab}]", CAPT), Image(str(OUT / f"data_distribution_or_{c}{md}.png"), width=LW, height=LW * 1620 / 2970)]
M = 1.6 * cm
doc = BaseDocTemplate(str(ROOT / "docs/보고서.pdf"), title="clinic4 baseline 및 체성분 feature 추가 효과")
pw_, ph_ = A4; lw_, lh_ = landscape(A4)
doc.addPageTemplates([PageTemplate("port", [Frame(1.8 * cm, M, pw_ - 3.6 * cm, ph_ - 2 * M, id="p")], pagesize=A4),
                      PageTemplate("land", [Frame(1.4 * cm, 1.2 * cm, lw_ - 2.8 * cm, lh_ - 2.4 * cm, id="l")], pagesize=landscape(A4))])
doc.build(S)
print("saved")
