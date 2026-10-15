from __future__ import annotations

# outputs/261012/prognosis_auc.xlsx(prognosis_auc.py 결과)를 읽어 docs/보고서_예후.pdf로 만든다 (단독 실행)
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
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[3]; OUT = ROOT / "outputs/261012"
pdfmetrics.registerFont(TTFont("K", r"C:\Windows\Fonts\malgun.ttf")); pdfmetrics.registerFont(TTFont("KB", r"C:\Windows\Fonts\malgunbd.ttf"))
H1 = ParagraphStyle("h1", fontName="KB", fontSize=17, leading=24, spaceAfter=4)
H2 = ParagraphStyle("h2", fontName="KB", fontSize=12, spaceBefore=12, spaceAfter=5, keepWithNext=1)
B = ParagraphStyle("b", fontName="K", fontSize=9.5, leading=15, spaceAfter=4)
CAP = ParagraphStyle("cap", fontName="K", fontSize=8.5, leading=12, spaceAfter=6, textColor=colors.HexColor("#444444"))
GREEN, RED, GREY = colors.HexColor("#D5ECD4"), colors.HexColor("#F6D3D0"), colors.HexColor("#E8EEF4")
R = pd.read_excel(OUT / "prognosis_auc.xlsx", sheet_name="auc"); C = pd.read_excel(OUT / "prognosis_auc.xlsx", sheet_name="cohort")
DIS = list(C.disease); SETS = list(dict.fromkeys(R.feature_set))
g = lambda d, s, c: R[(R.disease == d) & (R.feature_set == s)][c].iloc[0]


# 표 생성: marks={(행,열): 색}
def tbl(rows, widths, marks=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "K", 8.5), ("FONT", (0, 0), (-1, 0), "KB", 8.5), ("BACKGROUND", (0, 0), (-1, 0), GREY),
          ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for (r, c), col in (marks or {}).items(): st.append(("BACKGROUND", (c, r), (c, r), col))
    t.setStyle(TableStyle(st)); return t


# 그림: 질환별 feature set의 AUC (clinic4 baseline 기준선 대비)
fig, ax = plt.subplots(1, len(DIS), figsize=(13, 3.4), sharey=True)
for a, d in zip(ax, DIS):
    v = [g(d, s, "AUC") for s in SETS]; e = [g(d, s, "AUC_sd_rep") for s in SETS]
    a.bar(range(len(SETS)), v, yerr=e, color=["#888"] + ["#4C78A8"] * (len(SETS) - 1), capsize=2); a.axhline(.5, color="r", lw=.8, ls="--")
    a.set_title(d); a.set_xticks(range(len(SETS))); a.set_xticklabels(["c4"] + [s.split("(")[0] for s in SETS[1:]], rotation=60, ha="right", fontsize=7); a.set_ylim(.3, .7)
ax[0].set_ylabel("AUC (100회 평균)"); plt.tight_layout(); plt.savefig(OUT / "prognosis_auc.png", dpi=170); plt.close()

better = [(d, s) for d in DIS if d != "MI" for s in SETS[1:] if g(d, s, "dAUC") >= .02 and g(d, s, "win_rate") >= .9]
worse = [(d, s) for d in DIS if d != "MI" for s in SETS[1:] if g(d, s, "dAUC") <= -.02 and g(d, s, "win_rate") <= .1]
fmt = lambda L: ", ".join(f"{d} {s}({g(d, s, 'dAUC'):+.3f})" for d, s in L) or "없음"
S = []
S += [Paragraph("5년 내 발병 예후: clinic4 + landmark 체성분 feature 추가 효과", H1),
      Paragraph("261012 · data/prognosis_5y_landmark_filtered.xlsx · 1:1 언더샘플링 100회 반복 5-fold CV", CAP),
      Paragraph("요약", H2),
      Paragraph(f"MI를 뺀 5개 질환에서 clinic4 baseline AUC는 {min(g(d,'clinic4','AUC') for d in DIS if d != 'MI'):.2f}~{max(g(d,'clinic4','AUC') for d in DIS if d != 'MI'):.2f}로 우연 수준에 가깝다. feature를 더해 ΔAUC가 +0.02 이상이고 반복의 90% 이상에서 baseline을 이긴 조합은 <b>{fmt(better)}</b>뿐이다. 개선 폭은 DL +0.03, STROKE +0.07 정도이고 STROKE는 235명이라 불안정하다. "
                f"반대로 일관되게 나빠진 조합은 <b>{fmt(worse)}</b>이다. 나머지 질환·조합은 변화가 없거나 방향이 일정하지 않다. 즉 5년 내 발병 시기를 구분하는 데 landmark 체성분이 안정적으로 기여한다고 말하기는 어렵다. "
                "MI는 54명(양성 25, 균형화 후 50명)이라 AUC가 0.5 아래로 내려가며 해석에서 제외했다.", B),
      Paragraph("1. 방법", H2),
      Paragraph("대상: prognosis_5y 코호트에서 키·체중 비현실값(hw_implausible)을 제외한 환자. 라벨 onset_within_5y는 CT 이후 5년(1,825일) 이내 발병=1이며, 이 코호트는 결국 발병한 환자만 포함하므로 '질환 유무'가 아니라 '발병 시기가 빠른가'를 맞히는 문제다. "
                "질환마다 소수 클래스 수만큼 다수 클래스를 무작위 1:1 언더샘플링하고(100회 반복, seed별) 층화 5-fold CV(LR, C=0.01, 표준화)를 수행해 반복별 fold 평균 AUC를 다시 100회 평균했다. "
                "clinic4=성별·나이·키·체중. 추가 feature: 구간(liver dome~치골) 평균·합산, VAT/SAT/AEC를 환자 구간평균으로 나눈 곡선 모양(pw), L1/L3/L5 landmark 값, 전체.", B),
      Paragraph("2. 코호트", H2),
      tbl([["질환", "전체 n", "5년 내 발병(양성)", "이후 발병(음성)", "균형화 후 n"]] + [[r.disease, r.n, r.n_pos, r.n_neg, r.n_balanced] for r in C.itertuples()], [3 * cm, 3 * cm, 4 * cm, 4 * cm, 3.5 * cm]),
      Paragraph("표 1. 질환별 코호트. 균형화 후 n = 2 × min(양성, 음성).", CAP),
      Paragraph("3. 결과", H2)]
rows = [["feature set (n_feat)"] + DIS]; marks = {}
for i, s in enumerate(SETS, 1):
    row = [f"{s} ({int(g(DIS[0], s, 'n_feat'))}~)" if s != "clinic4" else "clinic4 (baseline)"]
    for j, d in enumerate(DIS, 1):
        row.append("%.3f" % g(d, s, "AUC") + ("" if s == "clinic4" else " (%+.3f)" % g(d, s, "dAUC")))
        if s != "clinic4" and g(d, s, "dAUC") >= .02 and g(d, s, "win_rate") >= .9: marks[(i, j)] = GREEN
        if s != "clinic4" and g(d, s, "dAUC") <= -.02 and g(d, s, "win_rate") <= .1: marks[(i, j)] = RED
    rows.append(row)
S += [tbl(rows, [5.2 * cm] + [2 * cm] * len(DIS), marks),
      Paragraph("표 2. 100회 평균 AUC(괄호: baseline 대비 ΔAUC). 초록=ΔAUC≥+0.02이고 반복의 90% 이상에서 baseline보다 높음, 빨강=ΔAUC≤-0.02이고 90% 이상에서 낮음. 반복은 같은 환자를 재사용하므로 독립 검정이 아니며 p-value는 계산하지 않았다.", CAP),
      Image(str(OUT / "prognosis_auc.png"), width=17 * cm, height=17 * cm * 3.4 / 13),
      Paragraph("그림 1. 질환별 AUC(오차막대=반복 간 SD, 점선=0.5).", CAP),
      Paragraph("4. 해석과 한계", H2),
      Paragraph("• baseline이 0.5에 가까운 것은 이 코호트의 라벨 구조 때문이다. 전원이 결국 발병하는 환자라 성별·나이·체격만으로 발병 시기를 구분하기 어렵다. 질환 유무(단면) 분류의 0.7~0.8과 직접 비교할 수 없다.", B),
      Paragraph("• feature를 늘릴수록(전체 세트) 표본이 작은 질환(OSP 364명, MI 54명, STROKE 235명)에서 오히려 AUC가 내려가거나 흔들린다. 과적합 때문으로 보인다.", B),
      Paragraph("• 라벨 정의: CT 시점 이전 또는 당일에 발병한 환자(event_days_from_ct ≤ 0)가 HTN 1명, MI 1명, STROKE 7명 포함되어 있다. 제외하지 않았으며 STROKE(235명)에서는 영향이 있을 수 있다. DL 2명은 1,826일(5년 + 1일)에 발병했는데 양성으로 분류되어 있다.", B),
      Paragraph("• 외부검증은 하지 않았다. 이 코호트의 환자가 gangnam/sinchon 단면 코호트와 거의 겹치지 않아(약 50명, 2%) 코호트 간 검증이 불가능하다.", B)]
SimpleDocTemplate(str(ROOT / "docs/보고서_예후.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm, title="5년 예후 보고서").build(S)
print("saved docs/보고서_예후.pdf"); print("better:", better); print("worse:", worse)
