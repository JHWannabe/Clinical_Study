from __future__ import annotations

# 논문 형식(IMRaD) 원고 -> docs/논문형식_초안.pdf. 본문 Table 3개 + Figure 2개(저널 통상 한도 5개 이내), 나머지는 Supplementary로 분리.
# 입력: outputs/261012/*.xlsx, png (build_report_pdf.py와 clinic4_*.py가 만든 결과). 단독 실행.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[3]; O = ROOT / "outputs/261012_any"  # 정상군(세 질환 모두 없음) vs 해당 질환 보유(동반질환 포함) 결과
pdfmetrics.registerFont(TTFont("K", r"C:\Windows\Fonts\malgun.ttf")); pdfmetrics.registerFont(TTFont("KB", r"C:\Windows\Fonts\malgunbd.ttf"))
registerFontFamily("K", normal="K", bold="KB", italic="K", boldItalic="KB")
TT = ParagraphStyle("tt", fontName="KB", fontSize=16, leading=22, spaceAfter=6, alignment=1)
AU = ParagraphStyle("au", fontName="K", fontSize=10, leading=14, alignment=1, spaceAfter=10, textColor=colors.HexColor("#444444"))
H1 = ParagraphStyle("h1", fontName="KB", fontSize=12, spaceBefore=12, spaceAfter=4, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="KB", fontSize=10, spaceBefore=6, spaceAfter=2, keepWithNext=1)
B = ParagraphStyle("b", fontName="K", fontSize=9.5, leading=15, spaceAfter=4, alignment=4)
CAP = ParagraphStyle("cap", fontName="K", fontSize=8.5, leading=12, spaceAfter=6, textColor=colors.HexColor("#333333"))
CAPT = ParagraphStyle("capt", parent=CAP, fontName="KB", textColor=colors.black, keepWithNext=1, spaceAfter=3)
REF = ParagraphStyle("ref", fontName="K", fontSize=8.5, leading=12, leftIndent=14, firstLineIndent=-14, spaceAfter=2)
DIS, MEAS = ["HTN", "DM", "CKD"], ["VAT", "SAT", "TAMA", "AEC"]
GREEN, RED, GREY = colors.HexColor("#D5ECD4"), colors.HexColor("#F6D3D0"), colors.HexColor("#E8EEF4")
W = 17.4 * cm
pw = lambda a, d, q, sd=None: (f"{a:.3f}±{sd:.3f} " if sd is not None else f"{a:.3f} ") + f"({d:+.3f}, q={q:.3f})"


# 표 생성: marks={(행,열): 색}
def tbl(rows, widths, marks=None, fs=8):
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "K", fs), ("FONT", (0, 0), (-1, 0), "KB", fs), ("BACKGROUND", (0, 0), (-1, 0), GREY), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for (r, c), col in (marks or {}).items(): st += [("FONT", (c, r), (c, r), "KB", fs), ("BACKGROUND", (c, r), (c, r), col)]
    t.setStyle(TableStyle(st)); return t


# ---- 데이터 ----
bl = pd.read_excel(O / "clinic4_baseline.xlsx"); cb = pd.read_excel(O / "clinic4_feature_combos.xlsx"); ce = pd.read_excel(O / "clinic4_feature_combos_external.xlsx")
rp = pd.read_excel(O / "clinic4_ratio_cross.xlsx", sheet_name="pooled_cv"); re_ = pd.read_excel(O / "clinic4_ratio_cross.xlsx", sheet_name="external")
orr = pd.read_excel(O / "data_distribution_significance.xlsx", sheet_name="odds_ratio")
co = pd.read_excel(O / "clinic4_coefficients.xlsx")
LMN = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center", "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
ABB = dict(zip(LMN, ["LD", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "FH", "PM"]))


# 코호트 특성(결측 제외 후): n, 나이, 남성 비율, 질환 양성
def cohort_row(c):
    m = pd.read_excel(ROOT / f"data/{c}_landmark_filtered.xlsx", sheet_name="patient_metadata").dropna(subset=["PatientSex", "PatientAge", "Height", "Weight"] + DIS)
    return m, [f"{len(m):,}", f"{m.PatientAge.mean():.1f}±{m.PatientAge.std():.1f}", f"{100 * (m.PatientSex.astype(str).str.upper() == 'M').mean():.1f}"] + [f"{int(m[d].sum())} ({100 * m[d].mean():.1f}%)" for d in DIS]


# ---- 표 1: 코호트 특성 + baseline ----
cm_ = {c: cohort_row(c) for c in ["gangnam", "sinchon"]}
allm = pd.concat([v[0] for v in cm_.values()])
tot = [f"{len(allm):,}", f"{allm.PatientAge.mean():.1f}±{allm.PatientAge.std():.1f}", f"{100 * (allm.PatientSex.astype(str).str.upper() == 'M').mean():.1f}"] + [f"{int(allm[d].sum())} ({100 * allm[d].mean():.1f}%)" for d in DIS]
lr = bl[bl.model == "LR"].set_index(["eval", "disease"])
base_cell = lambda e, d: (f"{lr.loc[(e, d), 'AUC']:.3f}±{lr.loc[(e, d), 'AUC_sd']:.3f}" if lr.loc[(e, d), "AUC_sd"] == lr.loc[(e, d), "AUC_sd"] else f"{lr.loc[(e, d), 'AUC']:.3f}")
rows = [["", "N", "나이(세)", "남성(%)", "HTN", "DM", "CKD"]]
for lab, k, e in [("강남", "gangnam", "gangnam 5-fold CV"), ("신촌", "sinchon", "sinchon 5-fold CV")]: rows.append([lab] + cm_[k][1])
rows.append(["전체"] + tot)
rows.append(["baseline AUC (LR, clinic4)", "", "", "", "", "", ""])
for lab, e in [("강남 5-fold CV", "gangnam 5-fold CV"), ("신촌 5-fold CV", "sinchon 5-fold CV"), ("풀링 5-fold CV", "pooled 5-fold CV"),
               ("외부: 강남→신촌", "train gangnam -> test sinchon"), ("외부: 신촌→강남", "train sinchon -> test gangnam")]:
    rows.append([lab, "", "", ""] + [base_cell(e, d) for d in DIS])
T1 = tbl(rows, [4.2 * cm, 1.7 * cm, 2.3 * cm, 1.8 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm])
T1.setStyle(TableStyle([("SPAN", (0, 4), (-1, 4)), ("BACKGROUND", (0, 4), (-1, 4), GREY), ("FONT", (0, 4), (-1, 4), "KB", 8)]))

# ---- 표 2: 풀링 CV + 외부검증 (선택 구성) ----
SETS2 = [("clinic4", "clinic4 (baseline)"), ("(참고) +LM raw(L1,L3,L5)", "+ landmark raw (L1·L3·L5)"), ("(참고) +LM/mean(L1,L3,L5)", "+ landmark/구간평균 (L1·L3·L5)"),
         ("+분포 기반(질환별)", "+ 분포 기반 조합"), ("+in-fold 유의 선택(raw+pw 4종)", "+ in-fold 유의 선택"), ("+구간 합산(4종)", "+ 구간 합산 4종")]
RS = "+다른 조직·다른 landmark 상위3"
rowsA = [["Feature set"] + DIS]; mkA = {}
for i, (k, lab) in enumerate(SETS2 + [(RS, "+ 비율 상위 3 (다른 조직·landmark)")], 1):
    src = rp if k == RS else cb; r = src[src.feature_set == k].set_index("disease")
    rowsA.append([lab] + [(f"{r.loc[d, 'AUC']:.3f}±{r.loc[d, 'AUC_sd']:.3f}" if k == "clinic4" else pw(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, "p_fdr"], r.loc[d, "AUC_sd"])) for d in DIS])
    for j, d in enumerate(DIS, 1):
        if k != "clinic4" and r.loc[d, "p_fdr"] < .05: mkA[(i, j)] = GREEN if r.loc[d, "dAUC"] > 0 else RED
TA = tbl(rowsA, [5.0 * cm, 4.1 * cm, 4.1 * cm, 4.1 * cm], mkA)
rowsB = [["Feature set", "G→S HTN", "G→S DM", "G→S CKD", "S→G HTN", "S→G DM", "S→G CKD"]]; mkB = {}
for i, (k, lab) in enumerate(SETS2 + [(RS, "+ 비율 상위 3 (다른 조직·landmark)")], 1):
    src = re_ if k == RS else ce; row = [lab]
    for jj, tr in enumerate(["gangnam", "sinchon"]):
        for kk, d in enumerate(DIS):
            r = src[(src.train == tr) & (src.feature_set == k) & (src.disease == d)].iloc[0]
            row.append(f"{r.AUC:.3f}" if k == "clinic4" else f"{r.dAUC:+.3f}")
            if k != "clinic4" and r.p_fdr < .05: mkB[(i, 1 + jj * 3 + kk)] = GREEN if r.dAUC > 0 else RED
    rowsB.append(row)
TB = tbl(rowsB, [4.8 * cm] + [2.1 * cm] * 6, mkB)

# ---- 표 3: 두 코호트 모두 유의하고 같은 방향인 landmark별 보정 OR (pw_mean) ----
g = orr[orr.preprocess == "pw_mean"].pivot_table(index=["disease", "measure", "landmark"], columns="cohort", values=["OR", "ci_lo", "ci_hi", "q_fdr"])
both = g[(g[("q_fdr", "gangnam")] < .05) & (g[("q_fdr", "sinchon")] < .05) & ((g[("OR", "gangnam")] > 1) == (g[("OR", "sinchon")] > 1))]
cell = lambda r, c: f"{r[('OR', c)]:.2f} ({r[('ci_lo', c)]:.2f}-{r[('ci_hi', c)]:.2f})"
rows3 = [["질환", "측정", "Landmark", "강남 OR (95% CI)", "신촌 OR (95% CI)"]] + [[d, m, ABB[a], cell(r, "gangnam"), cell(r, "sinchon")] for (d, m, a), r in both.iterrows()]
T3 = tbl(rows3, [2 * cm, 2 * cm, 2.4 * cm, 4.5 * cm, 4.5 * cm])
n_both = len(both)


# ---- 그림 1: 단변량/보정 유의 landmark 수 heatmap 두 장을 A/B 패널로 합성 ----
def fig1():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    for ax, f, t in zip(axs, ["data_distribution_summary.png", "data_distribution_or_summary.png"], ["A", "B"]):
        ax.imshow(plt.imread(O / f)); ax.axis("off"); ax.set_title(t, loc="left", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(O / "manuscript_fig1.png", dpi=170); plt.close(fig)


fig1()

# ---- 서론~결론 본문 ----
import re
qf = lambda q: "q&lt;0.01" if q < .005 else f"q={q:.2f}"
c4 = cb[cb.feature_set == "clinic4"].set_index("disease")
sx = lambda c: co[(co.cohort == c) & (co.model == "clinic4+구간 합산") & (co.disease == "HTN") & (co.feature == "sex(M=1)")].OR.iloc[0]
ext_ = lambda tr, k, d: ce[(ce.train == tr) & (ce.feature_set == k) & (ce.disease == d)].iloc[0]
psm_b = pd.read_excel(O / "clinic4_psm.xlsx", sheet_name="balance"); psm_p = pd.read_excel(O / "clinic4_psm.xlsx", sheet_name="pooled_cv"); psm_e = pd.read_excel(O / "clinic4_psm.xlsx", sheet_name="external")
nsp = pd.read_excel(O / "clinic4_next_steps.xlsx", sheet_name="pooled_cv"); nse = pd.read_excel(O / "clinic4_next_steps.xlsx", sheet_name="external"); nsor = pd.read_excel(O / "clinic4_next_steps.xlsx", sheet_name="cohort_adjusted_or")
CTS = [x for x in dict.fromkeys(cb.feature_set) if x.startswith("CT만")]; NCT = [x for x in dict.fromkeys(cb.feature_set) if not x.startswith("CT만") and "기울기" not in x]
npos = {c: {d: int(bl[(bl["eval"] == f"{c} 5-fold CV") & (bl.model == "LR") & (bl.disease == d)].n_pos.iloc[0]) for d in DIS} for c in ["gangnam", "sinchon"]}
nana = {c: {d: int(bl[(bl["eval"] == f"{c} 5-fold CV") & (bl.model == "LR") & (bl.disease == d)].n.iloc[0]) for d in DIS} for c in ["gangnam", "sinchon"]}
bp = lambda d: cb[(cb.disease == d) & cb.feature_set.isin(NCT) & (cb.feature_set != "clinic4")]
bestp = {d: bp(d).loc[bp(d).AUC.idxmax()] for d in DIS}; nsig = {d: bp(d)[bp(d).p_fdr < .05] for d in DIS}
PT = {"HTN": "은", "DM": "은", "CKD": "는"}
pool_sent = "; ".join((f"{d}{PT[d]} {len(nsig[d])}개 구성에서 유의(최대 +{nsig[d].dAUC.max():.3f}, 최소 q={nsig[d].p_fdr.min():.3f})" if len(nsig[d]) else f"{d}{PT[d]} 유의한 구성 없음(최대 {bestp[d].dAUC:+.3f}, {qf(bestp[d].p_fdr)})") for d in DIS)
ext_sig = ce[ce.feature_set.isin(NCT) & (ce.feature_set != "clinic4") & (ce.p_fdr < .05)]
ext_sent = "; ".join(f"{'강남→신촌' if tr == 'gangnam' else '신촌→강남'} {d}에서 개선 {int((x.dAUC > 0).sum())}개, 악화 {int((x.dAUC < 0).sum())}개 구성" for (tr, d), x in ext_sig.groupby(["train", "disease"]))
bo = both.reset_index(); bo["lm"] = bo.landmark.str.replace("_center", "")
both_txt = ", ".join(f"{d} {m} {'·'.join(g.lm)}" for (d, m), g in bo.groupby(["disease", "measure"], sort=False))
ctb = {d: cb[(cb.disease == d) & cb.feature_set.isin(CTS)].pipe(lambda g: g.loc[g.AUC.idxmax()]) for d in DIS}
pb0 = {d: psm_p[(psm_p.disease == d) & (psm_p.feature_set == "clinic4")].iloc[0] for d in DIS}
pb = {d: psm_p[(psm_p.disease == d) & (psm_p.feature_set != "clinic4")].pipe(lambda g: g.loc[g.AUC.idxmax()]) for d in DIS}
pe = lambda d, tr, st: psm_e[(psm_e.train == tr) & (psm_e.disease == d) & (psm_e.feature_set == st)].iloc[0]
smd_b = psm_b[psm_b.stage == "매칭 전"].SMD.abs().max(); smd_a = psm_b[psm_b.stage != "매칭 전"].SMD.abs().max()
sb = lambda d, s: nsp[nsp.setting.str.startswith(f"② 성별 층화({s}") & (nsp.disease == d) & (nsp.feature_set == "clinic4")].AUC.iloc[0]
sex_q = nsp[nsp.setting.str.startswith("② ") & (nsp.feature_set != "clinic4")].q.min()
ra = lambda d, t: nsor[(nsor.disease == d) & (nsor.feature == t)].iloc[0]
raw_both = int(((orr.preprocess == "raw") & (orr.q_fdr < .05)).groupby([orr.disease, orr.measure, orr.landmark]).sum().gt(1).sum())
mwb = lambda pre_, d, m: int((mw[(mw.preprocess == pre_) & (mw.disease == d) & (mw.measure == m) & (mw.q_fdr < .05)].groupby("landmark").cohort.nunique() == 2).sum())
mw = pd.read_excel(O / "data_distribution_significance.xlsx", sheet_name="mannwhitney")
top_mw = ", ".join(f"{d} {m}({'raw' if pre_ == 'raw' else '환자 평균 대비'} {n}곳)" for n, pre_, d, m in sorted([(mwb(pre_, d, m), pre_, d, m) for pre_ in ["raw", "pw_mean"] for d in DIS for m in ["VAT", "SAT", "TAMA", "AEC"]], reverse=True)[:5])
Nd = lambda d: f"{nana['gangnam'][d] + nana['sinchon'][d]:,}"


# Table 4(본문): clinic4 없이 CT만 + 성향점수 매칭 요약
def t4_tbl():
    rows = [["질환", "baseline AUC", "CT만 AUC (Δ, q)", "매칭 후 baseline", "매칭 후 best AUC (Δ, q)", "매칭 G→S (q)", "매칭 S→G (q)"]]; mk = {}
    for i, d in enumerate(DIS, 1):
        b1 = pb[d]; e_g, e_s, e0g, e0s = pe(d, "gangnam", b1.feature_set), pe(d, "sinchon", b1.feature_set), pe(d, "gangnam", "clinic4"), pe(d, "sinchon", "clinic4")
        rows.append([d, f"{c4.loc[d, 'AUC']:.3f}", f"{ctb[d].AUC:.3f} ({ctb[d].dAUC:+.3f}, {qf(ctb[d].p_fdr)})".replace("&lt;", "<"), f"{pb0[d].AUC:.3f}", f"{b1.AUC:.3f} ({b1.dAUC:+.3f}, {qf(b1.q)})".replace("&lt;", "<"),
                     f"{e0g.AUC:.3f}→{e_g.AUC:.3f} ({qf(e_g.q)})".replace("&lt;", "<"), f"{e0s.AUC:.3f}→{e_s.AUC:.3f} ({qf(e_s.q)})".replace("&lt;", "<")])
        if b1.q < .05: mk[(i, 4)] = GREEN
        if e_g.q < .05: mk[(i, 5)] = GREEN if e_g.dAUC > 0 else RED
        if e_s.q < .05: mk[(i, 6)] = GREEN if e_s.dAUC > 0 else RED
        if ctb[d].p_fdr < .05: mk[(i, 2)] = GREEN if ctb[d].dAUC > 0 else RED
    return tbl(rows, [1.2 * cm, 1.8 * cm, 3.2 * cm, 2.0 * cm, 3.2 * cm, 2.8 * cm, 2.8 * cm], mk, fs=7)


T4 = t4_tbl()
S = [Paragraph("CT 체성분 landmark 지표가 임상 변수 기반 HTN·DM·CKD 분류에 더하는 가치: 두 코호트 내·외부 검증", TT),
     Paragraph("[저자명] · [소속] · 교신저자: [이메일]<br/>(초안 — 저자·소속·참고문헌·윤리 문구는 저자가 보완)", AU),
     Paragraph("초록", H1),
     Paragraph("<b>배경.</b> 복부 CT에서 얻는 내장지방(VAT), 피하지방(SAT), 근육(TAMA), 자동노출제어 곡선(AEC)이 성별·나이·키·체중(clinic4)에 더해 고혈압(HTN), 당뇨(DM), 만성콩팥병(CKD) 분류를 개선하는지, 그 개선이 다른 병원에서도 유지되는지는 불분명하다.", B),
     Paragraph(f"<b>방법.</b> 두 병원(강남 {cm_['gangnam'][1][0]}명, 신촌 {cm_['sinchon'][1][0]}명)의 CT에서 12개 해부학적 landmark(간 윗부분~치골)별 VAT·SAT·TAMA·AEC를 추출했다. 질환마다 정상군(세 질환이 모두 없는 환자)과 해당 질환이 있는 환자(동반질환 포함)를 비교했다. 질환군과 정상군을 landmark별로 비교(Mann-Whitney U, 나이·성별·BMI 보정 로지스틱 OR)하고, clinic4 로지스틱 회귀에 체성분 feature를 더한 모델의 성능을 풀링 5-fold CV(3회 반복, fold별 AUC 평균±표준편차; Nadeau-Bengio 보정 t 검정)와 병원 간 외부검증(DeLong)으로 평가했다. clinic4 없이 CT 변수만 쓴 모델과, 나이·성별·BMI 성향점수 매칭으로 정상군을 1:1 언더샘플링한 평가를 추가했다. 다중비교는 Benjamini-Hochberg FDR로 보정했다.", B),
     Paragraph(f"<b>결과.</b> clinic4의 풀링 CV AUC는 HTN {c4.loc['HTN', 'AUC']:.3f}±{c4.loc['HTN', 'AUC_sd']:.3f}, DM {c4.loc['DM', 'AUC']:.3f}±{c4.loc['DM', 'AUC_sd']:.3f}, CKD {c4.loc['CKD', 'AUC']:.3f}±{c4.loc['CKD', 'AUC_sd']:.3f}이었다. 나이·성별·BMI 보정 후 두 병원 모두에서 유의하고 방향이 같은 landmark는 {n_both}개(HTN의 VAT·SAT, DM의 VAT·AEC; 모두 OR&gt;1)였다. 풀링 CV에서 체성분을 더하면 {pool_sent}. 이 개선은 병원 간 외부검증에서 재현되지 않았고(DM 최고 구성 기준 강남→신촌 {ext_('gangnam', bestp['DM'].feature_set, 'DM').dAUC:+.3f}, 신촌→강남 {ext_('sinchon', bestp['DM'].feature_set, 'DM').dAUC:+.3f}) HTN은 강남→신촌에서 대부분의 구성이 유의하게 악화되었다. CT 변수만으로는 HTN {ctb['HTN'].AUC:.3f}, DM {ctb['DM'].AUC:.3f}, CKD {ctb['CKD'].AUC:.3f}로 HTN은 clinic4보다 낮고 DM·CKD는 비슷하거나 약간 높았으나 유의하지 않았다. 성향점수 매칭 후 clinic4의 AUC는 무작위 수준({pb0['HTN'].AUC:.2f}, {pb0['DM'].AUC:.2f}, {pb0['CKD'].AUC:.2f})이었고 CT 변수를 더하면 풀링 CV에서 HTN {pb['HTN'].AUC:.2f}, DM {pb['DM'].AUC:.2f}, CKD {pb['CKD'].AUC:.2f}로 올랐으나(모두 q&lt;0.05) 외부검증에서는 신촌→강남에서만 재현되었다.", B),
     Paragraph("<b>결론.</b> 체성분 지표는 나이·성별·BMI를 맞춘 비교에서 clinic4와 독립적인 정보를 보였으나, 병원 간에 일관되게 재현되는 분류 성능 향상은 확인되지 않았다. 병원 간 표준화와 장비 보정이 필요하다.", B),
     Paragraph("<b>키워드:</b> 체성분, 복부 CT, 내장지방, 자동노출제어, 외부검증, 성향점수 매칭, 고혈압, 당뇨, 만성콩팥병", B),

     Paragraph("1. 서론", H1),
     Paragraph("고혈압, 당뇨, 만성콩팥병은 대사·심혈관 위험과 밀접하며, 복부 CT에서 정량하는 내장지방과 근육량은 체질량지수로는 포착하기 어려운 체성분 정보를 제공한다. 또한 CT 촬영 시 관전류를 환자 체형에 맞춰 조절하는 자동노출제어(AEC) 곡선은 몸의 크기와 조성에 대한 간접 지표로 쓸 수 있다. 그러나 이러한 영상 지표가 성별·나이·키·체중 같은 기본 임상 정보에 더해 질환 분류 성능을 실질적으로 높이는지, 그리고 한 병원에서 확인한 개선이 다른 병원에서도 유지되는지는 충분히 검증되지 않았다. 본 연구는 두 병원의 복부 CT에서 해부학적 landmark별 체성분 지표를 추출해 (1) 질환군과 정상군의 분포 차이와 나이·성별·BMI 보정 후 연관성을 살피고, (2) clinic4에 대한 분류 성능 개선을 풀링 교차검증과 병원 간 외부검증으로 평가했다.", B),

     Paragraph("2. 방법", H1),
     Paragraph("2.1 연구 대상 및 영상 지표", H2),
     Paragraph("강남과 신촌 두 병원의 복부 CT 코호트를 사용했다. 각 환자에서 12개 해부학적 landmark(liver dome, T10~L5, S1, femoral head, inferior pubic margin)의 slice 위치를 정하고, 그 위치의 VAT, SAT, TAMA와 AEC 값을 추출했다. 입력 변수(성별, 나이, 키, 체중)나 질환 라벨이 결측인 환자와 landmark 순서가 이상한 환자는 제외했다. 질환(HTN, DM, CKD)마다 정상군은 세 질환이 모두 없는 환자, 질환군은 해당 질환이 있는 모든 환자(다른 질환을 동반한 경우 포함)로 정의했다. 환자 자신의 liver dome~치골 구간 평균으로 나눈 값(pw_mean)을 체격 영향을 줄이는 정규화 값으로 함께 분석했다.", B),
     Paragraph("2.2 분포 및 연관성 분석", H2),
     Paragraph("landmark별로 질환군과 정상군을 Mann-Whitney U 검정으로 비교했다(질환·측정별 landmark 12개에 BH-FDR). 이어 landmark 값(코호트 내 표준화)을 나이·성별·BMI와 함께 넣은 로지스틱 회귀로 +1 SD당 odds ratio(OR)와 95% 신뢰구간을 구했다. 두 코호트 모두 FDR q&lt;0.05이고 방향이 같은 landmark를 일관된 신호로 정의했다.", B),
     Paragraph("2.3 예측 모델 및 평가", H2),
     Paragraph("기준 모델 clinic4는 성별, 나이, 키, 체중을 입력으로 하는 L2 정규화 로지스틱 회귀이며, 체성분 feature를 더한 모델과 비교했다. 정규화 강도 C는 학습 fold 내부 5-fold AUC로 선택했다(nested). 풀링(두 병원 합산) 5-fold CV를 fold 분할을 바꿔 3회 반복하고 fold별 AUC 15개의 평균±표준편차를 보고했다. clinic4 대비 AUC 차이는 같은 fold끼리 짝지은 차이에 Nadeau-Bengio 보정 paired t 검정을 적용했다. 외부검증은 한 병원으로 학습해 다른 병원에서 평가하고 DeLong paired test로 검정했다. 모든 검정의 다중비교는 질환별 BH-FDR로 보정했다. 분포 분석에서 정한 조합 외에, 학습 fold(외부검증은 학습 병원)에서만 단변량 기준으로 feature를 고르는 in-fold 선택을 사용해 평가 데이터가 선택에 관여하지 않게 했다. 조직·landmark를 교차한 비율 feature는 세 종류로 나눠 학습 fold에서 상위 3개를 선택해 평가했다(보충자료). 추가로 (1) 성별·나이·키·체중 없이 CT 변수만 입력한 모델, (2) 같은 병원 안에서 나이·성별·BMI 로지스틱 성향점수가 가장 가까운 정상군을 질환군과 1:1로 짝지은(최근접, 비복원, caliper 0.2 SD of logit, 5회 반복) 성향점수 매칭 언더샘플링 평가, (3) 병원별 표준화, 성별 분리, 병원 고정효과와 장비 군집 보정 로지스틱 회귀를 수행했다. 병원이 2곳이므로 임의효과 대신 고정효과를 사용했다.", B),

     Paragraph("3. 결과", H1),
     Paragraph("3.1 대상자 및 baseline 성능", H2),
     Paragraph(f"최종 분석 대상은 {len(allm):,}명(강남 {cm_['gangnam'][1][0]}명, 신촌 {cm_['sinchon'][1][0]}명)이었고 질환별 분석에는 정상군과 해당 질환군만 사용했다(HTN {Nd('HTN')}명, DM {Nd('DM')}명, CKD {Nd('CKD')}명; Table 1). clinic4의 풀링 CV AUC는 HTN {c4.loc['HTN', 'AUC']:.3f}, DM {c4.loc['DM', 'AUC']:.3f}, CKD {c4.loc['CKD', 'AUC']:.3f}였고 CKD는 양성이 적어(강남 {npos['gangnam']['CKD']}명, 신촌 {npos['sinchon']['CKD']}명) fold 간 변동이 컸다(SD {c4.loc['CKD', 'AUC_sd']:.3f}). 비선형 모델(HGB)은 LR보다 높지 않았다(Table S1).", B),
     Paragraph("3.2 질환군과 정상군의 landmark별 차이", H2),
     Paragraph(f"보정 없는 단변량 비교에서 두 코호트 모두 유의한 landmark가 많은 항목은 {top_mw} 순이었다(Figure 1A). 나이·성별·BMI를 보정하면 유의한 landmark 수가 줄었다(Figure 1B). 두 코호트에서 모두 유의하고 방향이 같은 landmark는 환자 평균 대비 값에서 "
               f"{n_both}개였고 모두 OR&gt;1이었다(Table 3): {both_txt}. raw 값에서 두 코호트가 동시에 유의하고 방향이 같은 landmark는 {raw_both}개였다. 위치별 전체 OR은 보충 Figure S1~S2에 있다.", B),
     Paragraph("3.3 체성분 feature 추가의 효과", H2),
     Paragraph(f"풀링 CV에서 체성분을 더하면 {pool_sent}(Table 2A, 보충 Table S2). 병원 간 외부검증에서는 baseline 대비 유의한 변화가 {ext_sent or '없었다'}(Table 2B, Figure 2). 분포 분석에서 일관된 feature를 모아 만든 조합도 병원 간 재현성을 확보하지 못했다. 조직·landmark를 교차한 비율은 풀링 CV에서 {', '.join(f'{d} {rp[(rp.disease == d) & (rp.feature_set != chr(99) + chr(108) + chr(105) + chr(110) + chr(105) + chr(99) + chr(52))].dAUC.min():+.3f}~{rp[(rp.disease == d) & (rp.feature_set != chr(99) + chr(108) + chr(105) + chr(110) + chr(105) + chr(99) + chr(52))].dAUC.max():+.3f}' for d in DIS)} 변화였고(보충 Table S3), 다른 조직·다른 landmark 비율이 같은 landmark 비율보다 낫다는 근거는 없었다. 구간 합산 4종과 회귀 계수는 보충 Table S4~S5에 있다.", B),
     Paragraph("3.4 clinic4 없이 CT 변수만 사용", H2),
     Paragraph(f"성별·나이·키·체중 없이 CT 변수만 입력하면 풀링 CV AUC가 가장 높은 구성에서 HTN {ctb['HTN'].AUC:.3f}({ctb['HTN'].dAUC:+.3f}), DM {ctb['DM'].AUC:.3f}({ctb['DM'].dAUC:+.3f}), CKD {ctb['CKD'].AUC:.3f}({ctb['CKD'].dAUC:+.3f})로 clinic4보다 유의하게 높은 구성은 없었다(Table 4). 외부검증에서는 HTN과 DM에서 CT만 구성이 clinic4보다 유의하게 낮아지는 경우가 많았다(보충 Table S2). 따라서 CT 변수만으로 기본 임상 정보보다 유의하게 나은 성능은 얻지 못했다.", B),
     Paragraph("3.5 성향점수 매칭 언더샘플링과 추가 분석", H2),
     Paragraph(f"정상군을 같은 병원의 나이·성별·BMI가 가까운 환자로 1:1 매칭하자 표준화 평균차가 {smd_b:.2f}에서 {smd_a:.2f} 이하로 줄었고(보충 Table S6) clinic4의 AUC는 무작위 수준({pb0['HTN'].AUC:.2f}, {pb0['DM'].AUC:.2f}, {pb0['CKD'].AUC:.2f})이 되었다. 이때 CT 변수를 더하면 풀링 CV에서 HTN {pb['HTN'].AUC:.3f}, DM {pb['DM'].AUC:.3f}, CKD {pb['CKD'].AUC:.3f}까지 올랐다(모두 {qf(max(pb[d].q for d in DIS))} 이하; Table 4). 외부검증에서는 강남에서 학습한 모델의 개선은 유의하지 않았고 신촌에서 학습한 모델에서 DM·CKD가 크게, HTN이 소폭 재현되었다. 성별로 나누면 baseline AUC가 남성에서 낮았고(HTN {sb('HTN', '남'):.2f} 대 {sb('HTN', '여'):.2f}, DM {sb('DM', '남'):.2f} 대 {sb('DM', '여'):.2f}, CKD {sb('CKD', '남'):.2f} 대 {sb('CKD', '여'):.2f}) CT 변수 추가의 풀링 CV 개선은 어느 성별에서도 유의하지 않았다(q≥{sex_q:.2f}; 보충 Table S7). 병원과 장비를 보정한 로지스틱 회귀에서도 여러 합산 변수가 유의했으나(예: HTN VAT 합산 OR {ra('HTN', 'VAT_sum').OR_adj:.2f}) 병원마다 효과 크기가 달랐다(강남 {ra('HTN', 'VAT_sum').OR_gangnam:.2f}, 신촌 {ra('HTN', 'VAT_sum').OR_sinchon:.2f}; 보충 Table S8).", B),

     Paragraph("4. 고찰", H1),
     Paragraph(f"내장지방과 AEC 기반 지표는 질환군과 뚜렷이 달랐고 나이·성별·BMI를 보정한 뒤에도 두 병원에서 일관된 신호가 {n_both}개 landmark(상복부 VAT·SAT와 DM의 AEC)에 남았다. 나이·성별·BMI를 맞춘 성향점수 매칭에서는 풀링 CV에서 CT 변수가 세 질환 모두 AUC를 올려, 체성분이 임상 변수와 독립적인 정보를 일부 가짐을 시사한다. 그러나 이 개선은 병원 간 외부검증에서 한 방향(신촌 학습→강남 평가)에서만 재현되었고 clinic4에 직접 더한 모델에서는 DM의 풀링 CV 개선이 다른 병원에서 재현되지 않았다. 병원 간 분포 차이(AEC 수준 자체가 강남 약 300, 신촌 약 220; 성별 효과의 방향까지 달랐다: HTN에서 sex OR 강남 {sx('gangnam'):.2f}, 신촌 {sx('sinchon'):.2f})와 병원마다 다른 효과 크기가 크게 작용했을 가능성을 시사한다. 분포를 본 뒤 정한 조합과 best 구성은 낙관적일 수 있어 in-fold 선택과 외부검증으로 점검했다.", B),
     Paragraph("<b>제한점.</b> CKD 양성이 적다. 질환군에 동반질환자를 포함해 질환 간 겹침이 있다. 질환 정의, CT 장비·촬영 조건, 병원별 환자 구성은 충분히 분석하지 않았다. AUC 변화량의 신뢰구간은 보고하지 않았다. 풀링 CV의 검정은 fold 15개에 기반해 검정력이 낮다. 단면 자료의 현재 유병 상태 분류이며 연관성이지 인과는 아니다.", B),

     Paragraph("5. 결론", H1),
     Paragraph("landmark 체성분 지표는 clinic4와 독립적인 연관을 보였으나, clinic4에 더해도 병원 간에 일관되게 재현되는 질환 분류 개선을 보이지 않았다. 장비·촬영 조건 보정, 성별 층화 분석, CKD 환자 수 확대가 후속 과제이다.", B),

     Paragraph("참고문헌 (방법론 문헌만 수록, 임상 문헌은 저자가 추가)", H1)]
for r in ["Mann HB, Whitney DR. On a test of whether one of two random variables is stochastically larger than the other. Ann Math Stat. 1947;18:50-60.",
          "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44:837-45.",
          "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B. 1995;57:289-300.",
          "Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52:239-81.",
          "Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. BMJ. 2015;350:g7594."]:
    S.append(Paragraph(r, REF))

# ---- 본문 Table / Figure: 결과 섹션 안에서 처음 언급되는 문단 바로 뒤에 배치 (저널 통상: 합계 5개 이내) ----
BLK_T1 = [KeepTogether([Paragraph("Table 1. 코호트 특성과 baseline(clinic4) 성능", CAPT), T1,
                        Paragraph("질환 칸은 코호트 내 양성 수(%)이며 질환별 분석에는 정상군(세 질환 모두 없음)과 해당 질환군만 사용한다. baseline AUC는 LR(C=1) 기준이며 CV 행은 fold별 평균±SD(5회 반복), 외부검증 행은 단일 값이다.", CAP)]), Spacer(1, 6)]
BLK_F1 = [KeepTogether([Paragraph("Figure 1. 질환군과 정상군이 다른 landmark 수 (12개 중, FDR q&lt;0.05)", CAPT), Image(str(O / "manuscript_fig1.png"), width=W * .8, height=W * .8 * .8),
                        Paragraph("(A) 보정 없는 Mann-Whitney U 검정, (B) 나이·성별·BMI 보정 로지스틱 OR. 열 = 측정(VAT, SAT, TAMA, AEC), 행 = 코호트·질환, 왼쪽 = raw, 오른쪽 = 환자 평균 대비(pw_mean).", CAP)]), Spacer(1, 6)]
BLK_T3 = [KeepTogether([Paragraph("Table 3. 두 병원 모두 유의하고 방향이 같은 landmark별 보정 OR (환자 평균 대비 값, +1 SD당, 나이·성별·BMI 보정)", CAPT), T3,
                        Paragraph("두 코호트 모두 FDR q&lt;0.05. LD = liver dome, FH = femoral head, PM = inferior pubic margin. raw 값에서 두 코호트가 동시에 유의하고 방향이 같은 landmark는 " + str(raw_both) + "개였다.", CAP)]), Spacer(1, 6)]
BLK_T2 = [KeepTogether([Paragraph("Table 2. clinic4에 체성분 feature를 더한 성능", CAPT), Paragraph("A. 풀링 5-fold CV: AUC 평균±SD (clinic4 대비 차이, FDR q; Nadeau-Bengio 보정 t). 초록 = q&lt;0.05 개선", CAP), TA]), Spacer(1, 6),
          KeepTogether([Paragraph("B. 병원 간 외부검증: AUC(clinic4 행) 또는 clinic4 대비 AUC 차이. G→S = 강남 학습·신촌 평가, S→G = 반대. 초록/빨강 = DeLong FDR q&lt;0.05 개선/악화", CAP), TB]), Spacer(1, 6)]
BLK_F2 = [KeepTogether([Paragraph("Figure 2. clinic4와 clinic4 + landmark raw(L1·L3·L5)의 풀링 CV와 병원 간 외부검증 AUC", CAPT), Image(str(O / "clinic4_feature_search.png"), width=W * .9, height=W * .9 * .32),
                        Paragraph("* = clinic4 대비 FDR q&lt;0.05.", CAP)]), Spacer(1, 6)]


# 문단 시작 문구로 위치를 찾아 그 문단 바로 뒤에 블록 삽입
def insert_after(items, start, blocks):
    i = next(k for k, x in enumerate(items) if isinstance(x, Paragraph) and x.text.startswith(start))
    items[i + 1:i + 1] = blocks


insert_after(S, "최종 분석 대상은", BLK_T1)
insert_after(S, "보정 없는 단변량 비교에서", BLK_F1 + BLK_T3)
insert_after(S, "풀링 CV에서 체성분을 더하면", BLK_T2 + BLK_F2)
BLK_T4 = [KeepTogether([Paragraph("Table 4. clinic4 없이 CT 변수만 사용한 모델과 성향점수 매칭 후 성능", CAPT), T4,
                        Paragraph("CT만 = CT 변수만 입력한 구성 중 풀링 CV AUC 최고(Δ, q는 clinic4 대비). 매칭 = 나이·성별·BMI 성향점수 1:1 언더샘플링 후 clinic4 vs 가장 좋은 구성(매칭 5회 평균). G→S = 강남 학습·신촌 평가, S→G = 반대. 초록/빨강 = FDR q&lt;0.05 개선/악화.", CAP)]), Spacer(1, 6)]
insert_after(S, "정상군을 같은 병원의 나이·성별·BMI가", BLK_T4)

# ---- Supplementary ----
LW = 17.0 * cm
gen = lambda df, cols, fs=7: tbl([cols] + df[cols].astype(str).values.tolist(), [LW / len(cols)] * len(cols), fs=fs)
S += [PageBreak(), Paragraph("보충자료 (Supplementary)", H1)]
# S1 baseline LR/HGB
rows = [["평가 방식"] + [f"{d} {m}" for d in DIS for m in ["LR", "HGB"]]]
for lab, e in [("강남 5-fold CV", "gangnam 5-fold CV"), ("신촌 5-fold CV", "sinchon 5-fold CV"), ("풀링 5-fold CV", "pooled 5-fold CV"), ("강남→신촌", "train gangnam -> test sinchon"), ("신촌→강남", "train sinchon -> test gangnam")]:
    x = bl[bl["eval"] == e].set_index(["disease", "model"])
    rows.append([lab] + [(f"{x.loc[(d, m), 'AUC']:.3f}±{x.loc[(d, m), 'AUC_sd']:.3f}" if x.loc[(d, m), "AUC_sd"] == x.loc[(d, m), "AUC_sd"] else f"{x.loc[(d, m), 'AUC']:.3f}") for d in DIS for m in ["LR", "HGB"]])
S += [Paragraph("Table S1. clinic4 baseline: LR vs HGB (CV는 fold별 평균±SD)", CAPT), tbl(rows, [3.4 * cm] + [2.3 * cm] * 6), Spacer(1, 10)]
# S2 전체 조합
SETS = list(dict.fromkeys(cb.feature_set))
for ttl, df in [("Table S2a. 전체 feature 조합: 풀링 5-fold CV (AUC 평균±SD (ΔAUC, q))", cb)]:
    rows = [["feature set"] + DIS]; mk = {}
    for i, s_ in enumerate(SETS, 1):
        r = df[df.feature_set == s_].set_index("disease")
        rows.append([s_] + [(f"{r.loc[d, 'AUC']:.3f}±{r.loc[d, 'AUC_sd']:.3f}" if s_ == "clinic4" else pw(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, "p_fdr"], r.loc[d, "AUC_sd"])) for d in DIS])
        for j, d in enumerate(DIS, 1):
            if s_ != "clinic4" and r.loc[d, "p_fdr"] < .05: mk[(i, j)] = GREEN
    S += [Paragraph(ttl, CAPT), tbl(rows, [5.4 * cm, 3.9 * cm, 3.9 * cm, 3.9 * cm], mk, fs=7), Spacer(1, 8)]
for tr, nm in [("gangnam", "강남 학습 → 신촌 평가"), ("sinchon", "신촌 학습 → 강남 평가")]:
    rows = [["feature set"] + DIS]; mk = {}
    for i, s_ in enumerate(SETS, 1):
        r = ce[(ce.train == tr) & (ce.feature_set == s_)].set_index("disease")
        rows.append([s_] + [(f"{r.loc[d, 'AUC']:.3f}" if s_ == "clinic4" else pw(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, "p_fdr"])) for d in DIS])
        for j, d in enumerate(DIS, 1):
            if s_ != "clinic4" and r.loc[d, "p_fdr"] < .05: mk[(i, j)] = GREEN if r.loc[d, "dAUC"] > 0 else RED
    S += [Paragraph(f"Table S2{'b' if tr == 'gangnam' else 'c'}. 전체 feature 조합: 외부검증 ({nm}; 초록 = 개선, 빨강 = 악화)", CAPT), tbl(rows, [5.4 * cm, 3.9 * cm, 3.9 * cm, 3.9 * cm], mk, fs=7), Spacer(1, 8)]
# S3 비율
S.append(PageBreak())
rows = [["비율 종류 (상위 3 추가)"] + DIS]; mk = {}
RSETS = list(dict.fromkeys(rp.feature_set))
for i, s_ in enumerate(RSETS, 1):
    r = rp[rp.feature_set == s_].set_index("disease")
    rows.append([s_] + [(f"{r.loc[d, 'AUC']:.3f}±{r.loc[d, 'AUC_sd']:.3f}" if s_ == "clinic4" else pw(r.loc[d, "AUC"], r.loc[d, "dAUC"], r.loc[d, "p_fdr"], r.loc[d, "AUC_sd"])) for d in DIS])
    for j, d in enumerate(DIS, 1):
        if s_ != "clinic4" and r.loc[d, "p_fdr"] < .05: mk[(i, j)] = GREEN
S += [Paragraph("Table S3a. 비율 feature(같은 landmark·다른 조직 / 같은 조직·다른 landmark / 다른 조직·다른 landmark): 풀링 5-fold CV", CAPT), tbl(rows, [5.4 * cm, 3.9 * cm, 3.9 * cm, 3.9 * cm], mk, fs=7), Spacer(1, 8)]
rows = [["비율 종류", "G→S HTN", "G→S DM", "G→S CKD", "S→G HTN", "S→G DM", "S→G CKD"]]; mk = {}
for i, s_ in enumerate(RSETS, 1):
    row = [s_]
    for jj, tr in enumerate(["gangnam", "sinchon"]):
        for kk, d in enumerate(DIS):
            r = re_[(re_.train == tr) & (re_.feature_set == s_) & (re_.disease == d)].iloc[0]
            row.append(f"{r.AUC:.3f}" if s_ == "clinic4" else f"{r.dAUC:+.3f}")
            if s_ != "clinic4" and r.p_fdr < .05: mk[(i, 1 + jj * 3 + kk)] = GREEN if r.dAUC > 0 else RED
    rows.append(row)
S += [Paragraph("Table S3b. 비율 feature: 외부검증 (clinic4 행 = AUC, 그 외 = ΔAUC)", CAPT), tbl(rows, [4.8 * cm] + [2.1 * cm] * 6, mk, fs=7), Spacer(1, 10)]
# S4 회귀 계수(풀링)
rows = [["feature"] + [f"{m} {d}" for m in ["clinic4", "+구간합산"] for d in DIS]]; mk = {}
for i, f in enumerate(["sex(M=1)", "age", "height", "weight", "VAT_sum", "SAT_sum", "TAMA_sum", "AEC_sum"], 1):
    row = [f]
    for jj, md in enumerate(["clinic4", "clinic4+구간 합산"]):
        for kk, d in enumerate(DIS):
            x = co[(co.cohort == "pooled") & (co.model == md) & (co.disease == d) & (co.feature == f)]
            if x.empty: row.append("-"); continue
            x = x.iloc[0]; row.append(f"{x.OR:.2f} ({x.ci_low:.2f}-{x.ci_high:.2f})")
            if x.p < .05: mk[(i, 1 + jj * 3 + kk)] = GREEN if x.OR > 1 else RED
    rows.append(row)
S += [Paragraph("Table S4. 로지스틱 회귀 OR (풀링, 연속형 +1 SD당, sex = 남성 vs 여성; 초록 = p&lt;0.05 위험 증가, 빨강 = 감소)", CAPT), tbl(rows, [1.9 * cm] + [2.55 * cm] * 6, mk, fs=6.5), Spacer(1, 10)]
rows = [["feature"] + [f"{c} {d}" for c in ["강남", "신촌"] for d in DIS]]; mk = {}
for i, f in enumerate(["sex(M=1)", "VAT_sum", "SAT_sum", "TAMA_sum", "AEC_sum"], 1):
    row = [f]
    for jj, c in enumerate(["gangnam", "sinchon"]):
        for kk, d in enumerate(DIS):
            x = co[(co.cohort == c) & (co.model == "clinic4+구간 합산") & (co.disease == d) & (co.feature == f)].iloc[0]
            row.append(f"{x.OR:.2f} ({x.ci_low:.2f}-{x.ci_high:.2f})")
            if x.p < .05: mk[(i, 1 + jj * 3 + kk)] = GREEN if x.OR > 1 else RED
    rows.append(row)
S += [Paragraph("Table S5. 코호트별 OR (clinic4 + 구간 합산 모델)", CAPT), tbl(rows, [1.9 * cm] + [2.55 * cm] * 6, mk, fs=6.5)]
# S6 성향점수 매칭 균형, S7 성별 분리, S8 병원·장비 보정 OR
bp_ = psm_b.pivot_table(index=["disease", "variable"], columns="stage", values="SMD").reset_index(); bn_ = psm_b.groupby(["disease", "stage"])[["n", "n_pos"]].first()
rows = [["질환", "변수", "SMD 매칭 전", "SMD 매칭 후(seed 0)", "매칭 전 n (양성)", "매칭 후 n (양성)"]] + [[r.disease, r.variable, f"{r._3:+.3f}", f"{r._4:+.3f}", f"{bn_.loc[(r.disease, '매칭 전')].n}({bn_.loc[(r.disease, '매칭 전')].n_pos})", f"{bn_.loc[(r.disease, '매칭 후(seed0)')].n}({bn_.loc[(r.disease, '매칭 후(seed0)')].n_pos})"] for r in bp_.itertuples()]
S += [PageBreak(), Paragraph("Table S6. 성향점수 매칭 전후 공변량 균형(표준화 평균차, SMD; 병원 안에서 나이·성별·BMI 매칭)", CAPT), tbl(rows, [1.8 * cm, 2.4 * cm, 2.8 * cm, 3.4 * cm, 3.0 * cm, 3.0 * cm], fs=7), Spacer(1, 10)]
rows = [["질환 · 성별", "정상군 / 질환군 (명)", "baseline AUC", "best 구성", "best AUC (Δ, q)", "강남→신촌 (q)", "신촌→강남 (q)"]]; mk = {}; i = 0
for d in DIS:
    for sx_ in ["남", "여"]:
        g_ = nsp[nsp.setting.str.startswith(f"② 성별 층화({sx_}") & (nsp.disease == d)]
        if g_.empty: continue
        i += 1; b0 = g_[g_.feature_set == "clinic4"].iloc[0]; b1 = g_[g_.feature_set != "clinic4"].pipe(lambda x: x.loc[x.AUC.idxmax()]); nn = re.search(r"n=(\d+), 양성 (\d+)", b0.setting)
        r_ = [f"{d} · {sx_}", f"{int(nn.group(1)) - int(nn.group(2))} / {nn.group(2)}", f"{b0.AUC:.3f}", b1.feature_set, f"{b1.AUC:.3f} ({b1.dAUC:+.3f}, {qf(b1.q)})".replace("&lt;", "<")]
        for j, tr in enumerate(["gangnam", "sinchon"], 5):
            e = nse[(nse.setting == f"② 성별 층화({sx_})") & (nse.train == tr) & (nse.disease == d)]
            if e.empty: r_.append("-"); continue
            e0 = e[e.feature_set == "clinic4"].iloc[0]; e1 = e[e.feature_set == b1.feature_set].iloc[0]; r_.append(f"{e0.AUC:.3f}→{e1.AUC:.3f} ({qf(e1.q)})".replace("&lt;", "<"))
            if e1.q < .05: mk[(i, j)] = GREEN if e1.dAUC > 0 else RED
        rows.append(r_)
S += [Paragraph("Table S7. 성별 분리: baseline vs 가장 좋은 구성 (정상군 = 세 질환 모두 없음, 질환군 = 해당 질환; 성별 안에서 구성, q = FDR 보정 p값)", CAPT), tbl(rows, [1.7 * cm, 2.5 * cm, 1.8 * cm, 3.0 * cm, 3.0 * cm, 2.7 * cm, 2.7 * cm], mk, fs=6.5), Spacer(1, 10)]
rows = [["질환", "feature", "병원 보정 OR (95% CI)", "p", "강남 OR", "신촌 OR", "교호작용 p"]] + [[r.disease, r.feature.replace("_sum", " 합산"), f"{r.OR_adj:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f})", "%.3f" % r.p_adj, f"{r.OR_gangnam:.2f}", f"{r.OR_sinchon:.2f}", "%.3f" % r.p_interaction] for r in nsor.itertuples()]
S += [Paragraph("Table S8. 병원 고정효과 + 장비 군집 로버스트 로지스틱 OR (나이·성별·키·몸무게 포함, 1 SD당; 교호작용 p = 효과가 병원마다 다른지)", CAPT), tbl(rows, [1.6 * cm, 2.6 * cm, 4.2 * cm, 1.8 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm], fs=7)]
# 그림
for f, t in [("data_distribution_or_by_landmark_pw_mean.png", "Figure S1. landmark별 보정 OR (pw_mean; 채운 점 = FDR q&lt;0.05)"), ("data_distribution_or_by_landmark_raw.png", "Figure S2. landmark별 보정 OR (raw)"),
             ("data_distribution_gangnam.png", "Figure S3. 강남: landmark별 질환군 vs 정상군 분포 (raw; 선 = median, 음영 = IQR)"), ("data_distribution_sinchon.png", "Figure S4. 신촌: landmark별 분포 (raw)"),
             ("data_distribution_gangnam_pw_mean.png", "Figure S5. 강남: landmark별 분포 (pw_mean)"), ("data_distribution_sinchon_pw_mean.png", "Figure S6. 신촌: landmark별 분포 (pw_mean)"),
             ("data_distribution_matrix_a.png", "Figure S7. 두 코호트 모두 유의한 landmark 위치 (raw, pw_mean)")]:
    from PIL import Image as PI
    w_, h_ = PI.open(O / f).size
    S += [PageBreak(), Paragraph(t, CAPT), Image(str(O / f), width=W, height=W * h_ / w_)]

doc = SimpleDocTemplate(str(ROOT / "docs/논문형식_초안.pdf"), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="CT 체성분 landmark 지표의 질환 분류 기여: 원고 초안")
doc.build(S)
print("saved", n_both)
