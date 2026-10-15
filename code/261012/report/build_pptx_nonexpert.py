# 비전공자용 발표 덱 생성: outputs/261012의 step 0~7 결과(질환군 vs 무질환 정상군 기준)와 data_distribution 결과만 읽어 docs/261015.pptx를 만든다.
# 템플릿 = docs/261002_연구세미나발표자료.pptx (표지·제목·핵심문장·배지·번호 도형, 종료 표지). 템플릿의 기존 슬라이드는 새 슬라이드를 만든 뒤 전부 제거한다.
# 표는 사용자가 확정한 표 규격(위치 0.56"/1.07", 총높이 5,124,636 EMU, 12pt, 중앙정렬), 종료 표지는 항상 Appendix 앞. 슬라이드마다 발표 노트(쉬운 설명)를 단다.
import copy, io, re, sys
from lxml import etree
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
TEMPLATE, OUT = ROOT / "docs/261002_연구세미나발표자료.pptx", ROOT / "docs/261015.pptx"
O = ROOT / "outputs/261012"
FIG = ROOT / "outputs/261012/nonexpert_figs"; FIG.mkdir(parents=True, exist_ok=True)
REPORT_DIR = ROOT.parent / "2026-1_Study/연구코드/data/강남/landmark/landmark_report"  # 환자별 landmark 확인 이미지(12곳 단면 + 분할 결과)
DIS = ["HTN", "DM", "CKD"]
DISK = {"HTN": "HTN", "DM": "DM", "CKD": "CKD"}  # 약어를 그대로 쓰고 뜻은 용어 슬라이드에 한 번 설명
NAVY, TEXT = RGBColor(0x1F, 0x3B, 0x63), RGBColor(0x33, 0x3F, 0x4D)
TBL_STYLE = "{C083E6E3-FA7D-4D7B-A595-EF9225AFEA82}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
STEPS = {1: "1_clinic4_landmark_sum", 2: "2_clinic4_landmark_raw", 3: "3_clinic4_landmark_peak", 4: "4_clinic4_tissue_ratio", 5: "5_clinic4_segment_stats",
         6: "6_clinic4_single_tissue", 7: "7_clinic4_landmark_topk"}
STEPE = {1: "sum of 12 landmark values", 2: "raw values by landmark", 3: "curve peak features", 4: "inter-tissue ratios", 5: "segment mean and sum", 6: "single-tissue features", 7: "top-k landmark selection by coefficient"}
STEPN = {1: "12곳 값의 합", 2: "위치별 원래 값", 3: "곡선의 꼭대기(peak)", 4: "조직 간 비율", 5: "구간 평균·합", 6: "조직 하나씩", 7: "coefficient 순서대로"}
LMK = {"liver_dome": "Liver dome", "T10": "T10", "T11": "T11", "T12": "T12", "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4",
       "L5": "L5", "S1": "S1", "femoral_head": "Femoral head", "inferior_pubic_margin": "Inferior pubic margin"}
EVN = {"gangnam 5-fold CV": "강남 5-fold CV", "sinchon 5-fold CV": "신촌 5-fold CV", "pooled 5-fold CV": "두 병원 합친 5-fold CV",
       "train gangnam -> test sinchon": "강남 train → 신촌 test", "train sinchon -> test gangnam": "신촌 train → 강남 test"}
pstr = lambda p: "<0.001" if p < .001 else f"{p:.3f}"
pcell = lambda p: "p<0.001" if p < .001 else f"p={p:.3f}"
# 셀에 사방 테두리 박스 (p<0.05 강조용)
def box_cell(cell, color="C00000", w=28575):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("lnL", "lnR", "lnT", "lnB"):
        ln = etree.SubElement(tcPr, f"{A}{tag}", w=str(w), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, f"{A}solidFill"); etree.SubElement(sf, f"{A}srgbClr", val=color)
        etree.SubElement(ln, f"{A}prstDash", val="solid")


dstr = lambda d, p: f"{d:+.3f} (p{'<0.001' if p < .001 else '=' + format(p, '.3f')})"


# 한 단계의 풀링 5-fold CV 결과를 표 행으로: 방법 | 입력 수 | 질환별 "AUC (ΔAUC, p)" (기준 대비)
def step_rows(r):
    r = r[r["eval"] == "pooled 5-fold CV"]; rows = []
    for fs in r.feature_set.drop_duplicates():
        g = r[r.feature_set == fs].set_index("disease")
        cells = [f"{g.loc[d, 'AUC']:.3f} ± {g.loc[d, 'AUC_sd']:.3f}" if fs == "clinic4" else f"{g.loc[d, 'AUC']:.3f} ± {g.loc[d, 'AUC_sd']:.3f} ({g.loc[d, 'dAUC']:+.3f}, {pcell(g.loc[d, 'p_cv'])})" for d in DIS]
        rows.append([fs.replace("clinic4", "clinic4 (기준)").replace("LM raw ", "").replace("+", "+ ", 1) if fs != "clinic4" else "clinic4 (기준)", str(int(g.n_feat.iloc[0]))] + cells)
    return rows


STEP_HDR, STEP_W = ["방법", "입력 수"] + DIS, [2.2, .7, 3.1, 3.1, 3.1]


# 한 단계의 내부 CV(cv)와 외부검증(ext) 결과
def load_step(i):
    x = O / STEPS[i] / f"clinic4_step{i}.xlsx"
    if i == 1:
        return pd.read_excel(x, sheet_name="cv"), pd.read_excel(x, sheet_name="external")
    return pd.read_excel(x), pd.read_excel(str(x).replace(".xlsx", "_external.xlsx"))


# 단계·질환별 pooled CV에서 ΔAUC가 가장 큰 feature set과 그 외부검증 결과 (선택 기준은 내부 CV뿐)
def best_rows():
    rows = []
    for i in STEPS:
        r, e = load_step(i)
        r0 = r[(r.feature_set == "clinic4") & (r["eval"] == "pooled 5-fold CV")].set_index("disease").AUC_sd
        r = r[(r.feature_set != "clinic4") & (r["eval"] == "pooled 5-fold CV")]
        for d in DIS:
            g = r[r.disease == d].sort_values("dAUC", ascending=False).iloc[0]
            ee = e[(e.disease == d) & (e.feature_set == g.feature_set)].set_index("train")
            rows.append(dict(step=i, disease=d, name=g.feature_set, n_feat=int(g.n_feat), base=g.AUC - g.dAUC, auc=g.AUC, auc_sd=g.AUC_sd, base_sd=r0.loc[d], d=g.dAUC, p=g.p_cv,
                             gs=ee.loc["gangnam", "dAUC"], gs_p=ee.loc["gangnam", "p_delong"], sg=ee.loc["sinchon", "dAUC"], sg_p=ee.loc["sinchon", "p_delong"]))
    return pd.DataFrame(rows)


# ---- 그림 ----
# 12 landmark 위치와 각 위치에서 읽는 4가지 값을 보여주는 도식
def fig_landmarks():
    fig, ax = plt.subplots(figsize=(11, 4.6)); y = np.arange(12)[::-1]
    ax.plot([0, 0], [-.4, 11.4], color="#9aa5b1", lw=10, solid_capstyle="round", zorder=1)
    for yy, (k, v) in zip(y, LMK.items()):
        ax.scatter([0], [yy], s=170, color="#4C78A8", zorder=3); ax.text(.35, yy, f"{v}", va="center", fontsize=12)
    ax.text(-.35, 11.4, "머리 쪽", ha="right", va="center", fontsize=11, color="#555"); ax.text(-.35, -.4, "다리 쪽", ha="right", va="center", fontsize=11, color="#555")
    for k, (t, c, d) in enumerate([("VAT", "#E4572E", "내장지방"), ("SAT", "#F3A712", "피하지방"), ("TAMA", "#2E86AB", "근육"), ("AEC", "#76B041", "자동노출제어 곡선")]):
        ax.add_patch(plt.Rectangle((3.4 + 0 * k, 8.6 - 2.1 * k), 3.8, 1.7, color=c, alpha=.25)); ax.text(3.6, 9.45 - 2.1 * k, f"{t}  {d}", fontsize=13, va="center", fontweight="bold")
    ax.annotate("", xy=(3.3, 7), xytext=(1.6, 7), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
    ax.text(3.4, .9, "높이 12곳 × 값 4가지\n= 환자 한 명당 48개 숫자", fontsize=13, va="center", fontweight="bold")
    ax.set(xlim=(-2.2, 7.4), ylim=(-1, 12)); ax.axis("off"); fig.tight_layout(); fig.savefig(FIG / "landmarks.png", dpi=200, facecolor="white"); plt.close(fig)


# 평가 방식(같은 병원 안 5-fold, 병원 간 시험) 도식
def fig_eval():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    for k in range(5):
        for j in range(5):
            ax.add_patch(plt.Rectangle((j * 1.05, (4 - k) * 0.9), 1, .7, color="#E4572E" if j == k else "#9db4cf", alpha=.9))
        ax.text(-.15, (4 - k) * .9 + .35, f"{k + 1}회", ha="right", va="center", fontsize=11)
    ax.text(2.6, 4.75, "5-fold 교차검증 (같은 병원 안)", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.6, -.55, "빨강 = test fold, 파랑 = train fold", ha="center", fontsize=11, color="#444"); ax.set(xlim=(-1, 5.5), ylim=(-.9, 5.1)); ax.axis("off")
    ax = axs[1]
    for x, name in [(0, "강남 병원"), (3.2, "신촌 병원")]:
        ax.add_patch(plt.Rectangle((x, 1.6), 2, 1.1, color="#9db4cf")); ax.text(x + 1, 2.15, name, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(3.15, 2.45), xytext=(2.05, 2.45), arrowprops=dict(arrowstyle="->", lw=2.2)); ax.annotate("", xy=(2.05, 1.85), xytext=(3.15, 1.85), arrowprops=dict(arrowstyle="->", lw=2.2))
    ax.text(2.6, 3.4, "외부검증: 한 병원에서 train한 모델을\n다른 병원 환자로 test", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.6, .7, "화살표 방향 = train → test", ha="center", fontsize=11, color="#444"); ax.set(xlim=(-.3, 5.5), ylim=(0, 4.3)); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "eval.png", dpi=200, facecolor="white"); plt.close(fig)


# 위치(landmark) 하나씩 넣었을 때 질환별 ΔAUC (pooled CV, 별표 = p<0.05)
def fig_landmark_bars():
    r, _ = load_step(2); r = r[(r["eval"] == "pooled 5-fold CV") & r.feature_set.str.startswith("+LM raw") & ~r.feature_set.str.contains("전체")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, d in zip(axs, DIS):
        g = r[r.disease == d].set_index(r[r.disease == d].feature_set.str.replace("+LM raw ", "", regex=False)).loc[list(LMK)]
        ys = np.arange(len(g))[::-1]
        ax.barh(ys, g.dAUC, color=np.where(g.p_cv < .05, "#E4572E", "#9db4cf")); ax.axvline(0, color="#555", lw=1)
        for y, v, p in zip(ys, g.dAUC, g.p_cv):
            if p < .05: ax.text(v + .001, y, "*", va="center", fontsize=16, fontweight="bold")
        lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi + (hi - lo) * .08)  # 오른쪽에 여백을 둬 별표가 그래프 안에 들어오게
        ax.set(title=DISK[d], xlabel="ΔAUC (기준 대비)"); ax.set_yticks(ys); ax.set_yticklabels([LMK[k] for k in g.index]); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "landmark_bars.png", dpi=200, facecolor="white"); plt.close(fig)


# coefficient가 큰 위치부터 1곳씩 늘렸을 때 pooled CV AUC
def fig_topk():
    r, _ = load_step(7); r = r[r["eval"] == "pooled 5-fold CV"]
    names = ["clinic4"] + [f"+상위 {k}곳" for k in range(1, 7)] + ["+전체 12곳"]; xs = [0, 1, 2, 3, 4, 5, 6, 7]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for d, c in zip(DIS, ["#4C78A8", "#E4572E", "#76B041"]):
        v = r[r.disease == d].set_index("feature_set").loc[names, "AUC"].to_numpy()
        ax.plot(xs, v, "o-", color=c, lw=2.2, label=DISK[d]); ax.text(7.12, v[-1], f"{v[-1]:.3f}", color=c, va="center", fontsize=11)
        ax.text(-.12, v[0] + {"HTN": .0013, "DM": -.0013, "CKD": 0}[d], f"{v[0]:.3f}", color=c, va="center", ha="right", fontsize=11)  # 기준(clinic4) AUC
        p = r[r.disease == d].set_index("feature_set").loc[names, "p_cv"].to_numpy()  # clinic4 대비 p<0.05인 점에 별표 (clinic4 자신은 제외)
        for x, y, pv in zip(xs[1:], v[1:], p[1:]):
            if pv < .05: ax.text(x, y + .0007, "*", color=c, ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(["기준\n(추가 없음)"] + [f"{k}곳" for k in range(1, 7)] + ["12곳\n전부"]); ax.set_xlim(-.9, 7.8)
    ax.set_ylim(.744, .8165); ax.set_ylabel("AUC (두 병원 합친 5-fold CV의 test fold)"); ax.grid(axis="y", alpha=.3); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "topk.png", dpi=200, facecolor="white"); plt.close(fig)


# 4개 구간별 조직 평균값: 정상군 vs 질환군 (중앙값, 막대 = IQR, 별표 = Mann-Whitney BH-FDR q<0.05)
def fig_segments():
    import importlib.util; sys.path.insert(0, str(ROOT / "code/261012"))
    from scipy.stats import mannwhitneyu
    sp = importlib.util.spec_from_file_location("seg5", ROOT / "code/261012/5_clinic4_segment_stats.py"); m5 = importlib.util.module_from_spec(sp); sp.loader.exec_module(m5)
    A = pd.concat([m5.load(c) for c in ["gangnam", "sinchon"]], ignore_index=True)
    unit = {"VAT": "VAT (cm²)", "SAT": "SAT (cm²)", "TAMA": "TAMA (cm²)", "AEC": "AEC (mA)"}
    stats = {}
    for d in DIS:
        use = A[(A[d] == 1) | (A[DIS].sum(axis=1) == 0)]
        for t in m5.MEAS:
            for k in range(4):
                a, b = use[use[d] == 0][f"{t}_mean{k}"], use[use[d] == 1][f"{t}_mean{k}"]
                stats[(d, t, k)] = (a.quantile([.25, .5, .75]).to_numpy(), b.quantile([.25, .5, .75]).to_numpy(), mannwhitneyu(a, b).pvalue)
    ps = np.array([v[2] for v in stats.values()]); o = np.argsort(ps); q = np.empty_like(ps); q[o] = np.minimum.accumulate((ps[o] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
    qd = dict(zip(stats, q))
    fig, axs = plt.subplots(3, 4, figsize=(12.5, 6.4))
    for i, d in enumerate(DIS):
        for j, t in enumerate(m5.MEAS):
            ax = axs[i, j]
            for k in range(4):
                a, b, _ = stats[(d, t, k)]
                for off, v, c in ((-.17, a, "#4C78A8"), (.17, b, "#E4572E")):
                    ax.errorbar(k + off, v[1], yerr=[[v[1] - v[0]], [v[2] - v[1]]], fmt="o", color=c, capsize=3, ms=5)
                if qd[(d, t, k)] < .05: ax.text(k, max(a[2], b[2]), "*", ha="center", va="bottom", fontsize=16, fontweight="bold")
            ax.set_xticks(range(4)); ax.set_xticklabels([f"구간{k + 1}" for k in range(4)], fontsize=9); ax.grid(axis="y", alpha=.3); ax.margins(y=.25)
            if i == 0: ax.set_title(unit[t], fontsize=12, fontweight="bold")
            if j == 0: ax.set_ylabel(d, fontsize=12, fontweight="bold")
    axs[0, 0].plot([], [], "o", color="#4C78A8", label="정상군"); axs[0, 0].plot([], [], "o", color="#E4572E", label="질환군"); fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, fontsize=10, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(FIG / "segments.png", dpi=200, facecolor="white"); plt.close(fig)


# best model(36개 입력)의 구간 feature 계수: 질환 x (평균, 합) 별로 조직 x 구간 OR per SD 히트맵 (색 = log OR, 숫자 = OR)
def fig_coef():
    c = pd.read_excel(O / "5c_clinic4_segment_coef/clinic4_step5c_coef.xlsx").set_index(["disease", "feature"])
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 5.2)); v = np.log(2.0)
    for j, d in enumerate(DIS):
        for i, st in enumerate(("mean", "sum")):
            ax = axs[i, j]; m = np.array([[c.loc[(d, f"{t}_{st}{k}"), "OR"] for k in range(4)] for t in ["VAT", "SAT", "TAMA", "AEC"]])
            ax.imshow(np.log(m), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            for a in range(4):
                for b in range(4): ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=12, color="white" if abs(np.log(m[a, b])) > .5 else "black")
            ax.set_xticks(range(4)); ax.set_xticklabels([f"구간{k + 1}" for k in range(4)], fontsize=10); ax.set_yticks(range(4)); ax.set_yticklabels(["VAT", "SAT", "TAMA", "AEC"], fontsize=11)
            ax.set_title(f"{d}: 구간 {'평균' if st == 'mean' else '합'}", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "coef.png", dpi=200, facecolor="white"); plt.close(fig)


# 환자별 landmark 확인 이미지 샘플 2장: 제목의 PatientID 문구만 흰색으로 가려 저장 (정렬한 목록의 첫 번째와 가운데 파일)
def make_samples():
    fs = sorted(REPORT_DIR.glob("*.png")); out = []
    for k, p in enumerate([fs[0], fs[len(fs) // 2]]):
        im = Image.open(p).convert("RGB"); w, h = im.size
        im.paste((255, 255, 255), (int(w * .30), int(h * .012), int(w * .70), int(h * .034)))  # 'PatientID ...' 줄만 가림(범례는 유지)
        out.append(FIG / f"landmark_report_sample{k + 1}.png"); im.save(out[-1])
    return out, len(fs)


# 덱 조립기: 제목·핵심문장·배지·번호 도형은 템플릿 슬라이드에서 복제
# 슬라이드 제목 키워드 -> 근거 논문 (발표 노트 맨 아래에 추가). PubMed 확인 항목은 PMID 포함
R_SLICE = ["Mourtzakis M et al. Appl Physiol Nutr Metab 2008;33:997-1006 (L3 CT 단면이 전신 지방·근육을 예측). https://doi.org/10.1139/H08-075 (PMID 18923576)",
           "Shen W et al. J Appl Physiol 2004;97:2333-8 (복부 한 단면으로 전신 근육·지방량 추정). https://doi.org/10.1152/japplphysiol.00744.2004 (PMID 15310748)"]
R_VAT = ["Fox CS et al. Circulation 2007;116:39-48 (Framingham: 내장지방 VAT가 고혈압·당뇨·대사증후군과 강하게 연관). https://doi.org/10.1161/CIRCULATIONAHA.106.675355 (PMID 17576866)"]
R_FDR = ["Benjamini Y, Hochberg Y. J R Stat Soc B 1995;57:289-300 (BH-FDR 다중비교 보정). https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"]
R_DELONG = ["DeLong ER et al. Biometrics 1988;44:837-45 (같은 환자에서 두 AUC 비교, DeLong 검정). https://doi.org/10.2307/2531595 (PMID 3203132)"]
R_NB = ["Nadeau C, Bengio Y. Mach Learn 2003;52:239-81 (반복 교차검증 fold 간 중복을 보정한 t 검정). https://doi.org/10.1023/A:1024068626366"]
REFS = {"Background": R_SLICE + R_VAT, "Features": R_SLICE, "Evaluation design": R_NB + R_DELONG, "Disease vs. normal": R_VAT + R_FDR,
        "number of significant": R_FDR, "Adding body composition": R_NB + R_DELONG, "External validation": R_DELONG + R_FDR, "Appendix A1": R_VAT}

class Deck:
    def __init__(self):
        self.prs = Presentation(TEMPLATE); self.n_orig = len(self.prs.slides); self.notes = []
        shp = {s.name: s for s in self.prs.slides[7].shapes}
        self.rule, self.badge, self.num = (copy.deepcopy(shp[n]._element) for n in ("Rectangle 3", "Rounded Rectangle 4", "슬라이드 번호 개체 틀 3"))
        self.cover_boxes = [copy.deepcopy(s._element) for s in self.prs.slides[0].shapes]
        self.end_blob = next(sh for sh in self.prs.slides[19].shapes if sh.shape_type == 13).image.blob

    def end_cover(self, note):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6]); self.notes.append(note)
        s.shapes.add_picture(io.BytesIO(self.end_blob), 0, 0, self.prs.slide_width, self.prs.slide_height)

    def cover(self, title_lines, date, note):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[0]); self.notes.append(note)
        for el in self.cover_boxes: s.shapes._spTree.append(copy.deepcopy(el))
        boxes = list(s.shapes)
        for p, t in zip(boxes[0].text_frame.paragraphs, title_lines): p.runs[0].text = t
        boxes[1].text_frame.paragraphs[0].runs[0].text = date

    # 제목 + 핵심 문장(lines가 비면 문장 상자 제거) + 구분선/배지/번호
    def page(self, title, badge, lines, note, text_h=931520, size=18):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[1]); self.notes.append(note)
        t = s.shapes.title; t.left, t.top, t.width, t.height = 508000, 365126, 11176000, 532342
        r = t.text_frame.paragraphs[0].add_run(); r.text, r.font.size, r.font.bold, r.font.color.rgb = title, Pt(26), True, NAVY
        body = s.placeholders[1]
        if lines:
            body.left, body.top, body.width, body.height = 508000, 1050000, 11176000, text_h
            for k, line in enumerate(lines):
                p = body.text_frame.paragraphs[0] if k == 0 else body.text_frame.add_paragraph()
                r = p.add_run(); r.text, r.font.size, r.font.bold, r.font.color.rgb = line, Pt(size), False, TEXT
        else:
            body._element.getparent().remove(body._element)
        for el in (self.rule, self.badge, self.num): s.shapes._spTree.append(copy.deepcopy(el))
        for sh in s.shapes:
            if sh.name == "Rounded Rectangle 4": sh.text_frame.paragraphs[0].runs[0].text = badge
        return s

    # 그림을 영역 안에 비율 유지로 가운데 맞춰 넣기
    def picture(self, s, path, top, max_h, max_w=11176000, left=508000):
        w, h = Image.open(path).size; max_h = min(max_h, 6280000 - top); k = min(max_w / w, max_h / h); pw, ph = int(w * k), int(h * k)  # 아래 장식 띠(약 6.3M EMU) 위까지만
        s.shapes.add_picture(str(path), Emu(left + (max_w - pw) // 2), Emu(top), Emu(pw), Emu(ph))

    # 표 규격(확정): 위치 (0.56", 1.07"), 12.22" 폭, 총높이 5,124,636 EMU, 12pt, 가로·세로 중앙, 헤더만 굵게
    def table(self, s, header, rows, col_w=None, merge_col0=False, top=978408, height=5124636):
        nr, nc, width = len(rows) + 1, len(header), 11176000
        tbl = s.shapes.add_table(nr, nc, Emu(508000), Emu(top), Emu(width), Emu(height)).table
        tbl._tbl.tblPr.set("firstRow", "1"); tbl._tbl.tblPr.set("bandRow", "1"); tbl._tbl.tblPr.find(f"{A}tableStyleId").text = TBL_STYLE
        col_w = col_w or [1] * nc
        for j, w in enumerate(col_w): tbl.columns[j].width = int(width * w / sum(col_w))
        for i in range(nr): tbl.rows[i].height = height // nr
        if merge_col0:  # 병합을 먼저, 값은 시작 셀에만
            i = 1
            while i < nr:
                j = i
                while j + 1 < nr and rows[j][0] == rows[i - 1][0]: j += 1
                if j > i: tbl.cell(i, 0).merge(tbl.cell(j, 0))
                i = j + 1
        for i in range(nr):
            vals = header if i == 0 else rows[i - 1]
            for j in range(nc):
                cell = tbl.cell(i, j)
                if cell.is_spanned: continue
                lines = vals[j].split("\n"); cell.text_frame.text = lines[0]  # 줄바꿈이 있으면 문단을 나눠 같은 서식을 적용
                for extra in lines[1:]: cell.text_frame.add_paragraph().add_run().text = extra
                cell.vertical_anchor = 3
                for p in cell.text_frame.paragraphs:
                    p.alignment = 2; p.runs[0].font.size, p.runs[0].font.bold = Pt(12), i == 0

    def save(self):
        new = list(self.prs.slides)[self.n_orig:]; assert len(new) == len(self.notes), (len(new), len(self.notes))
        for sl, text in zip(new, self.notes):
            t = sl.shapes.title.text if sl.shapes.title is not None else ""
            rs = list(dict.fromkeys(r for k, v in REFS.items() if k in t for r in v))
            sl.notes_slide.notes_text_frame.text = text + ('\n\n[근거 논문]\n' + '\n'.join(f"- {r}" for r in rs) if rs else "")
        ids = self.prs.slides._sldIdLst
        for sld in list(ids)[:self.n_orig]:  # 템플릿 원본 슬라이드 제거(rel도 끊어 고아 파트가 남지 않게)
            ids.remove(sld); self.prs.part.drop_rel(sld.rId)
        self.prs.save(OUT); print(f"Saved {OUT} ({len(self.prs.slides)} slides)")


def main():
    fig_landmark_bars(); fig_topk(); fig_segments(); fig_coef(); samples, n_report = make_samples()
    best = best_rows(); base = pd.read_excel(O / "0_baseline/clinic4_baseline.xlsx")
    b5 = best[best.step == 5].set_index("disease")  # 단계 5(구간 평균·합)가 세 질환 모두에서 pooled CV 최고
    top = best.sort_values("d", ascending=False).groupby("disease").head(1).set_index("disease")  # 질환별 전체 최고
    pool = base[base["eval"] == "pooled 5-fold CV"].set_index("disease")
    n_g, n_s = base[base["eval"] == "gangnam 5-fold CV"].set_index("disease"), base[base["eval"] == "sinchon 5-fold CV"].set_index("disease")
    D = Deck()

    # ---------------- 본문 ----------------
    D.cover(["Detecting HTN, DM, and CKD with CT Body Composition", ""], "2026.10.15", "오늘 발표는 CT 사진에서 얻는 몸 구성 정보(내장지방, 근육 등)가 질환을 가려내는 데 얼마나 도움이 되는지 확인한 결과입니다.")
    D.page("Summary", "Summary", [
        "목적: 기본 정보(나이·성별·키·몸무게)에 CT 몸 구성(VAT·SAT·TAMA·AEC)을 추가했을 때 HTN·DM·CKD 예측 성능 변화 평가",
        "자료: 강남·신촌 환자, 질환군 vs 정상군, 두 병원 합친 5-fold CV(3회 반복)",
        f"기본 정보만: AUC HTN {pool.loc['HTN', 'AUC']:.2f}, DM {pool.loc['DM', 'AUC']:.2f}, CKD {pool.loc['CKD', 'AUC']:.2f}",
        f"CT 몸 구성 추가: DM {b5.loc['DM', 'auc']:.2f} (ΔAUC +{b5.loc['DM', 'd']:.3f}), CKD {b5.loc['CKD', 'auc']:.2f} (+{b5.loc['CKD', 'd']:.3f}) 상승, HTN +{b5.loc['HTN', 'd']:.3f} (p={b5.loc['HTN', 'p']:.3f})",
        "높이별: CKD는 Femoral head·T11에서 상승폭 최대, 상위 3곳 추가부터 유의 (DM 5곳, HTN 없음)",
        "한계: CKD 환자 수 적음, 다중비교 보정 미실시, 단면 자료로 인과 해석 불가"],
        "기본 정보만으로 AUC 0.75~0.77입니다. CT 몸 구성을 더하면 두 병원을 합친 5-fold 기준으로 당뇨와 콩팥병의 AUC가 오르고, 고혈압은 유의한 변화가 없습니다. 콩팥병은 환자 수가 적고 여러 방법 중 best를 고른 값이라 실제보다 높게 나왔을 수 있습니다.", text_h=4600000, size=20)
    s = D.page("Landmark localization: visual quality check", "Methods", [
        "Filtering criteria (landmark_filtered)",
        "Height·Weight·BMI와 HTN·DM·CKD 라벨 결측 없음",
        "키·몸무게가 비현실적인 환자 제외",
        "Segmentation 성공 (seg_status = ok)",
        "Landmark slice가 T10 → 치골 순으로 단조 증가 (liver_dome 제외)",
        "AEC flat 제외 (CV 0.05 미만 · 범위 10 mA 미만 · R² 0.95 이상)",
        "Disease label: CT 촬영월 ≥ 첫 진단월 (기저질환 ICD-10)"],
        f"landmark_filtered 데이터셋은 아래 조건을 모두 만족한 환자만 남긴 것입니다(연구 코드 merge_metadata.py의 _filter_sheets). (1) Height·Weight·BMI와 HTN·DM·CKD 라벨에 결측이 없을 것, (2) 키·몸무게 입력 오류로 표시된 환자 제외, (3) TotalSegmentator segmentation 성공, (4) T10에서 치골까지 landmark slice 번호가 엄격히 증가(liver_dome은 T10과의 상하 위치가 달라 검사하지 않음), (5) AEC 신호가 flat이 아닐 것(CV 0.05 미만, 범위 10 mA 미만, 직선 R² 0.95 이상이면 flat으로 제외). 질환 라벨은 기저질환.xlsx의 ICD-10 코드로, CT 촬영월이 첫 진단월과 같거나 늦으면 양성입니다. 오른쪽은 환자별 확인용 이미지 샘플로(환자 번호는 가렸습니다), 이런 이미지를 강남 {n_report:,}명분 만들어 landmark 위치와 분할 영역을 눈으로 확인했습니다. 색 범례의 NAMA, LAMA, IMATA는 근육 관련 영역 구분입니다.", text_h=4800000, size=16)
    body = s.placeholders[1]; body.width = 6300000  # 왼쪽에 조건 목록, 오른쪽에 확인용 이미지 샘플
    for k, p in enumerate(body.text_frame.paragraphs):
        for r in p.runs: r.font.size, r.font.bold = (Pt(18) if k == 0 else Pt(15)), k == 0
    D.picture(s, samples[0], 1100000, 5100000, max_w=4600000, left=6900000)
    pc = pd.read_excel(O / "previous_vs_current_counts.xlsx")  # 질환 라벨 변경 전후 환자 수 (previous = 라벨 변경 전 landmark_filtered, current = 현재)
    ITEM = {"Total patients": "전체 환자", "HTN": "HTN", "DM": "DM", "CKD": "CKD", "Normal (no HTN/DM/CKD)": "정상군 (세 질환 없음)"}
    rows_pc = [[{"gangnam": "강남", "sinchon": "신촌"}[r.cohort], ITEM[r.item], f"{r.previous:,}", f"{r.current:,}", f"{r.current - r.previous:+,}", "–" if pd.isna(r.relabeled_1to0) else f"{int(r.relabeled_1to0):,}"] for r in pc.itertuples()]
    s = D.page("Disease label change: patient counts", "Methods", [
        "질환 라벨을 CT 촬영월 ≥ 첫 진단월(기저질환 ICD-10) 기준으로 변경, 20세 미만 제외",
        "질환군 감소, 정상군 증가: 진단 전 촬영 환자가 질환군(1)에서 정상군(0)으로 이동 (0 → 1 이동은 없음)"],
        "라벨 변경 전후 환자 수 비교입니다. 질환 라벨을 CT 촬영월이 첫 진단월과 같거나 늦은 경우에만 양성으로 바꿨습니다. 그 결과 질환군은 줄고 정상군(세 질환이 모두 없는 사람)은 늘었습니다. 질환군에서 정상군으로 바뀐 환자 수는 마지막 열에 있고, 반대로 바뀐 환자는 없습니다. 전체 환자 수의 감소는 20세 미만 환자를 제외했기 때문입니다.", text_h=1100000, size=16)
    D.table(s, ["병원", "항목", "변경 전", "변경 후", "증감", "재라벨 (1→0)"], rows_pc, col_w=[1, 2.6, 1.3, 1.3, 1.3, 1.6], merge_col0=True, top=2300000, height=3900000)
    s = D.page("Data Distribution", "Distribution", [], "두 병원 환자를 사용했습니다. 질환마다 '질환군'과 '정상군(세 질환이 모두 없는 사람)'만 비교했고, 다른 질환을 가진 환자는 비교에서 뺐습니다. 만성콩팥병 환자는 강남 45명, 신촌 63명으로 적은 편입니다.")
    rows = []
    for c, tb, key in [("강남", n_g, "gangnam"), ("신촌", n_s, "sinchon")]:
        for d in DIS:
            npos, n = int(tb.loc[d, "n_pos"]), int(tb.loc[d, "n"])
            rows.append([c, DISK[d], f"{npos}", f"{n - npos:,}", f"{n:,}", f"{100 * npos / n:.1f}"])
    D.table(s, ["병원", "질환", "질환군(명)", "정상군(명)", "비교한 전체(명)", "질환 비율(%)\n질환군 / 비교한 전체"], rows, col_w=[1, 1, 1.3, 1.3, 1.5, 2.2], merge_col0=True)
    s = D.page("Disease vs. normal: Gangnam", "Distribution", [
        "선 = 중앙값, 색칠한 띠 = 환자의 가운데 절반이 분포한 범위. 위 별표 = Mann-Whitney U, BH-FDR q 0.05 미만",
        "VAT: 세 질환 모두, 거의 모든 높이에서 질환군이 높음"],
        "행은 질환, 열은 4가지 값입니다. 파란 선은 정상군, 빨간 선은 질환군입니다. VAT는 두 선이 벌어지고 별표가 많고, TAMA는 거의 겹칩니다. 신촌 병원 그림은 다음 슬라이드에 있습니다.", text_h=1100000, size=16)
    D.picture(s, O / "0_data_distribution/data_distribution_gangnam.png", 2200000, 4500000)
    # 신촌 분포 + 두 병원 비교 (유의한 높이 수는 data_distribution_significance.xlsx의 q<0.05 개수)
    sig = pd.read_excel(O / "0_data_distribution/data_distribution_significance.xlsx"); sig["sig"] = sig.q_fdr < .05
    cnt = sig.groupby(["disease", "measure", "cohort"]).sig.sum().unstack("cohort")
    rg = lambda v, fmt: f"{fmt(v.min())}~{fmt(v.max())}"
    vd = sig[sig.measure == "VAT"].groupby(["disease", "cohort"]).median_diff.mean().unstack("cohort")
    s = D.page("Disease vs. normal: Sinchon", "Distribution", [
        "강남과 동일한 방식 (선 = 중앙값, 띠 = 가운데 절반, 별표 = BH-FDR q 0.05 미만)",
        "VAT: 신촌도 세 질환 모두, 거의 모든 높이에서 질환군이 높음. TAMA: 두 군이 거의 겹침"],
        "신촌 결과입니다. VAT는 강남과 같은 방향이나 질환군과 정상군의 차이는 강남보다 작습니다. TAMA는 강남에서 모든 높이에서 차이가 났으나 신촌에서는 별표가 없습니다.", text_h=1100000, size=16)
    D.picture(s, O / "0_data_distribution/data_distribution_sinchon.png", 2200000, 4500000)
    s = D.page("Baseline model (clinic4)", "Results", [
        "모델: clinic4 (Age·Sex·Height·Weight 4개 입력) 로지스틱 회귀, 규제 강도는 train fold 안에서 선택",
        "평가: 두 병원 합친 5-fold CV (3회 반복), 값은 mean ± SD (AUC는 fold별, Sens·Spec은 반복별)",
        "Sens·Spec: 민감도 90% 이상을 만족하는 가장 큰 cutoff 기준"], f"기준 모델의 점수입니다(두 병원 합친 5-fold CV, 3회 반복). 기본 정보만으로 AUC {pool.AUC.min():.2f}~{pool.AUC.max():.2f}이고, 이후 비교는 이 값 대비 변화량(ΔAUC)입니다. 값은 mean ± SD입니다(AUC는 fold별, Sens·Spec은 반복별 SD). Sens·Spec은 민감도 90% 이상을 만족하는 가장 큰 cutoff에서의 값입니다.", text_h=1500000, size=18)
    D.table(s, ["두 병원 합친 5-fold CV"] + DIS, [["AUC ± SD"] + [f"{pool.loc[d, 'AUC']:.3f} ± {pool.loc[d, 'AUC_sd']:.3f}" for d in DIS],
                                              ["Sensitivity"] + [f"{pool.loc[d, 'Sens']:.3f} ± {pool.loc[d, 'Sens_sd']:.3f}" for d in DIS], ["Specificity"] + [f"{pool.loc[d, 'Spec']:.3f} ± {pool.loc[d, 'Spec_sd']:.3f}" for d in DIS],
                                              ["환자 수 (질환군)"] + [f"{int(pool.loc[d, 'n']):,} ({int(pool.loc[d, 'n_pos']):,})" for d in DIS]], col_w=[2.4, 1.6, 1.6, 1.6], top=2700000, height=2800000)
    s = D.page("Cases explored: best ΔAUC per case", "Results", [], "기본 정보에 CT 몸 구성을 추가하는 방법 7가지를 비교했습니다. 표의 값은 방법별로 질환마다 가장 높았던 조합의 ΔAUC(기준 clinic4 대비)와 p값이고, 두 병원 합친 5-fold CV(3회 반복) 기준입니다. 테두리 박스는 p<0.05입니다. 5번(구간 평균·합)이 세 질환 모두에서 가장 높았습니다.")
    D.table(s, ["Case", "HTN", "DM", "CKD"], [[f"{i}. {STEPN[i]}"] + [dstr(g.d.iloc[0], g.p.iloc[0]) for d in DIS for g in [best[(best.step == i) & (best.disease == d)]]] for i in STEPS], col_w=[3, 2.4, 2.4, 2.4])
    tb8 = [sh for sh in s.shapes if sh.has_table][0].table  # p<0.05 셀 테두리 박스
    for ri in range(1, len(tb8.rows)):
        for ci in range(1, 4):
            c = tb8.cell(ri, ci); m = re.search(r"p([<=])([\d.]+)", c.text)
            if m and (m.group(1) == "<" or float(m.group(2)) < .05): box_cell(c)
    s = D.page("Segment features: normal vs. disease", "Results", ["4개 구간: ① Liver dome·T10·T11·T12, ② L1·L2, ③ L3·L4, ④ L5·S1·Femoral head·Pubic margin (구간 안 단면의 평균값, 한 구간에만 포함)", "점 = 중앙값, 막대 = 사분위 범위 (Q1~Q3), * = Mann-Whitney BH-FDR q<0.05"],
               "VAT·SAT·TAMA·AEC의 구간 평균값을 정상군(파랑)과 질환군(빨강)으로 나눠 비교한 그림입니다. 두 병원을 합쳤고, 질환마다 질환군과 세 질환이 모두 없는 정상군만 사용했습니다. 이 구간별 평균과 합이 best model의 입력(32개)입니다.", text_h=900000, size=14)
    D.picture(s, FIG / "segments.png", 2050000, 4650000)
    cf = pd.read_excel(O / "5c_clinic4_segment_coef/clinic4_step5c_coef.xlsx").set_index(["disease", "feature"]).OR
    base_or = lambda f: "·".join(f"{cf.loc[(d, f)]:.2f}" for d in DIS)
    s = D.page("Final best model: coefficients", "Results", [
        "best model (clinic4 + 4종 구간 평균·합, 36개 입력)의 계수",
        "OR per SD (표준화 계수), 두 병원 합친 5-fold CV 15개 train fold 모델의 평균, 1보다 크면 질환 가능성 증가 (HTN·DM·CKD 순)",
        f"기본 정보: Age {base_or('PatientAge')}, Weight {base_or('Weight')}, Height {base_or('Height')}, Sex {base_or('sex')}"],
        "36개 입력 모델의 계수입니다. 위 줄은 구간 평균, 아래 줄은 구간 합이고, 구간 ①~④는 ① Liver dome·T10·T11·T12, ② L1·L2, ③ L3·L4, ④ L5·S1·Femoral head·Pubic margin입니다. 구간 평균과 합은 서로 상관이 커서 개별 계수를 그대로 해석하기는 어렵습니다. AEC 구간④ 평균은 세 질환에서 모두 OR이 1보다 작고 15개 모델에서 부호가 같았습니다.", text_h=1250000, size=14)
    D.picture(s, FIG / "coef.png", 2300000, 4400000)
    s = D.page("Final best model: segment mean and sum", "Results", ["+ body comp. = 기본 정보 4개 + VAT·SAT·TAMA·AEC 4종 × 4개 구간(① Liver dome·T10·T11·T12, ② L1·L2, ③ L3·L4, ④ L5·S1·Femoral head·Pubic margin)의 평균·합 32개 (12곳 raw 값 전체가 아님)", "ΔAUC = 기준(clinic4) 대비 AUC 변화, 두 병원 합친 5-fold CV"], "ΔAUC는 기준 모델 대비 AUC 변화입니다. 단계 5(몸을 4구간으로 나눠 구간 평균과 합을 더한 방법)가 세 질환에서 두 병원 합친 5-fold CV test 점수가 가장 높았습니다. DM과 CKD는 AUC가 오르고 p값이 작지만, 여러 방법 중 best를 고른 값이라 실제보다 높게 나왔을 수 있습니다.", text_h=1500000, size=18)
    rows = [[DISK[d], f"{b5.loc[d, 'n_feat']}", f"{b5.loc[d, 'base']:.3f} ± {b5.loc[d, 'base_sd']:.3f}", f"{b5.loc[d, 'auc']:.3f} ± {b5.loc[d, 'auc_sd']:.3f}", dstr(b5.loc[d, 'd'], b5.loc[d, 'p'])] for d in DIS]
    D.table(s, ["Disease", "# features", "Baseline\nAUC", "AUC\n(+ body comp.)", "ΔAUC\n(두 병원 합친 5-fold CV)"], rows, col_w=[1, 1, 2, 2, 2.4], top=2700000, height=3000000)
    r5 = load_step(5)[0]; r5 = r5[r5["eval"] == "pooled 5-fold CV"].set_index(["disease", "feature_set"])  # best model과 baseline의 Sens90 기준 Sens·Spec
    pm = lambda row, c: f"{row[c]:.3f} ± {row[c + '_sd']:.3f}"
    sp = {d: (r5.loc[(d, "clinic4")], r5.loc[(d, b5.loc[d, "name"])]) for d in DIS}
    s = D.page("Final best model: specificity at 90% sensitivity", "Results", [
        "민감도 90% 이상을 만족하는 가장 큰 cutoff (Sens90)에서 Spec 비교, 값은 mean ± SD (반복 간 SD)",
        "Sens는 두 모델 모두 cutoff로 0.90대에 맞춰 baseline 값만 표기 (best model과 차이 0.001 이하), Spec 향상: " + ", ".join(f"{d} {sp[d][1].Spec - sp[d][0].Spec:+.3f}" for d in DIS)],
        "민감도를 90% 이상으로 맞춘 상태에서 정상군을 정상으로 맞추는 비율(특이도)이 얼마나 오르는지 비교한 표입니다. 같은 cutoff 기준으로 baseline(기본 정보만)과 최종 best model을 나란히 두었고, 두 병원 합친 5-fold CV(3회 반복) 기준입니다.", text_h=1500000, size=18)
    D.table(s, ["Disease", "Sens\n(Baseline)", "Spec\n(Baseline)", "Spec\n(Best model)", "ΔSpec"],
            [[d, pm(sp[d][0], "Sens"), pm(sp[d][0], "Spec"), pm(sp[d][1], "Spec"), f"{sp[d][1].Spec - sp[d][0].Spec:+.3f}"] for d in DIS], col_w=[1, 2.2, 2.2, 2.2, 1.4], top=2700000, height=3000000)
    D.page("Limitations", "Limitations", [
        "CKD 환자 수 적음 (강남 45명, 신촌 63명), 점수 변동 큼 (best model AUC SD 0.052)",
        "수백 개 조합 비교, 다중비교 보정 미실시로 우연히 높게 나온 결과 포함 가능",
        "best case 선택과 성능 평가에 같은 CV 사용, AUC가 실제보다 높게 나왔을 가능성",
        "독립된 외부 test set 없이 두 병원 합친 5-fold CV만 평가",
        "병원 간 차이: TAMA의 질환군-정상군 차이는 강남에서만 유의, 신촌은 유의한 높이 없음",
        "정상군은 HTN·DM·CKD가 모두 없는 환자로 한정, 질환 라벨은 CT 촬영월 ≥ 첫 진단월 기준",
        "단면 자료로 인과 해석 불가"],
        "CKD는 환자 수가 적고, 방법을 많이 비교했으며 best를 같은 CV로 고른 값이라 AUC가 실제보다 높게 나왔을 수 있습니다. 외부 test set이 없고 병원 간 차이가 섞여 있습니다. 정상군은 세 질환이 모두 없는 환자만 사용했습니다.", text_h=4800000, size=18)
    D.end_cover("본문은 여기까지입니다. 이후는 질문에 답하기 위한 부록입니다.")

    # ---------------- 부록 ----------------
    orr = pd.read_excel(O / "0_baseline/clinic4_baseline_coef_OR.xlsx"); orr = orr[orr.cohort == "pooled"].set_index(["feature", "disease"])
    FN = {"sex": "성별 (남=1)", "PatientAge": "나이", "Height": "키", "Weight": "몸무게"}
    s = D.page("Appendix A1. Baseline model coefficients (OR)", "Appendix", [],
               "OR = 변수가 1 SD 커질 때의 odds ratio입니다(1보다 크면 질환 가능성 증가). 두 병원 합친 자료에서 5-fold test fold마다 학습한 15개 모델의 평균이고, 괄호는 그 15개 결과의 가운데 95% 범위입니다. 예를 들어 나이의 OR이 2.5이면, 나이가 표준편차 1만큼 높을 때 질환 가능성이 약 2.5배가 된다는 뜻입니다.")
    D.table(s, ["변수"] + DIS, [[FN[k]] + [f"{orr.loc[(k, d), 'OR_per_SD']:.2f} ({orr.loc[(k, d), 'OR_lo']:.2f}~{orr.loc[(k, d), 'OR_hi']:.2f})" for d in DIS] for k in FN], col_w=[2, 2.4, 2.4, 2.4])
    TXT = {
        1: ("12곳 값의 합을 더하기", "조직마다 12곳의 값을 더해 '그 조직의 대략적인 총량'으로 사용 (4종을 한꺼번에 넣은 경우 포함)"),
        2: ("위치별 원래 값 더하기", "12곳 중 한 곳씩, 그리고 전부를 더해 어느 높이에 정보가 있는지 확인 (L3는 CT 체성분 연구의 표준 단면)"),
        3: ("곡선의 꼭대기(peak) 더하기", "몸을 따라 그린 지방·근육 곡선의 가장 높은 값(그 환자에서 조직이 가장 많은 곳)과 그 위치, 4구간별 꼭대기 값"),
        4: ("조직 간 비율 더하기", "내장지방/피하지방, 근육/BMI, 내장지방/근육 비율을 12곳마다 계산 (체격이나 지방 분포의 균형을 반영)"),
        5: ("구간 평균·합 더하기", "몸을 4구간(① Liver dome·T10·T11·T12, ② L1·L2, ③ L3·L4, ④ L5·S1·Femoral head·Pubic margin)으로 나눠 구간 평균과 합을 사용. 한 점보다 잡음에 덜 민감"),
        6: ("조직 하나씩 더하기", "VAT·SAT·TAMA·AEC 중 한 조직의 12곳 값만 더해, 어느 조직이 도움이 되는지 분리"),
        7: ("coefficient 순으로 늘려 더하기", "12곳 전체 모델의 coefficient가 큰 높이 순으로 1~6곳만 더함 (순서는 train 데이터 안에서만 계산해 test 정보가 새지 않게 함)")}
    for i in STEPS:
        s = D.page(f"Appendix B{i}. Step {i}: {STEPE[i]}", "Appendix", [],
                   f"추가한 정보: {TXT[i][1]}. 표는 두 병원 합친 5-fold CV(3회 반복)의 AUC(mean ± SD)이고, 괄호는 기본 정보만 쓴 모델(clinic4) 대비 ΔAUC와 p값(Nadeau-Bengio 보정 t)입니다.")
        D.table(s, STEP_HDR, step_rows(load_step(i)[0]), col_w=STEP_W)
    # 슬라이드 13 본문 수치: 결과 xlsx에서 계산
    lm2 = load_step(2)[0].rename(columns={"eval": "ev"}); lm2 = lm2[(lm2.ev == "pooled 5-fold CV") & lm2.feature_set.str.contains("LM raw") & ~lm2.feature_set.str.contains("전체") & (lm2.p_cv < .05)]
    ckd_sig = lm2[lm2.disease == "CKD"].sort_values("dAUC", ascending=False); ckd_top = [LMK[x.replace("+LM raw ", "")] for x in ckd_sig.feature_set[:2]]
    n_htn_sig = int((lm2.disease == "HTN").sum())
    s = D.page("Appendix C1. Landmark-wise contribution (ΔAUC)", "Appendix", [
        "12곳 중 한 곳씩만 더했을 때의 ΔAUC (기준 모델 대비 AUC 변화) (두 병원 합친 5-fold CV 기준, 빨강 = p값 0.05 미만)",
        f"CKD는 {'·'.join(ckd_top)}에서 상승폭 최대 (유의한 높이 {len(ckd_sig)}곳), HTN은 유의한 높이 {n_htn_sig}곳"],
        "막대가 오른쪽으로 길수록 해당 높이를 추가했을 때 AUC가 더 올랐습니다. 빨간 막대만 유의합니다(p<0.05). 탐색적 비교이므로 특정 높이가 가장 좋다고 단정할 수 없습니다.", text_h=1100000, size=16)
    D.picture(s, FIG / "landmark_bars.png", 2250000, 4400000)
    # 상위 k곳 추가 시 풀링 5-fold CV AUC (슬라이드 14)
    t7 = load_step(7)[0]; t7 = t7[t7["eval"] == "pooled 5-fold CV"]
    def k_line(d):
        g = t7[t7.disease == d].set_index("feature_set"); first = next((n for n in g.index[1:] if g.loc[n, "p_cv"] < .05), None)
        return f"{d}: 기준 {g.loc['clinic4', 'AUC']:.3f} → 상위 3곳 {g.loc['+상위 3곳', 'AUC']:.3f} → 12곳 {g.loc['+전체 12곳', 'AUC']:.3f} (기준보다 유의하게 오르는 최소 구성: {first.replace('+', '') if first else '없음'})"
    s = D.page("Appendix C2. Performance vs. number of landmarks", "Appendix", [
        "12곳 전체 모델에서 coefficient가 큰 높이부터 1곳씩 추가하며 test했습니다 (coefficient 순서는 train 데이터 안에서만 계산, 두 병원 합친 5-fold CV)",
        k_line("DM"), k_line("CKD"), k_line("HTN")],
        "coefficient가 큰 높이부터 순서대로 1곳, 2곳, 3곳 늘려 갔을 때의 점수입니다. 순서를 정할 때 test용 환자를 쓰지 않도록, 매번 train 데이터 안에서만 순서를 계산했습니다. 그래프는 두 병원을 합친 5-fold CV를 3회 반복한 test fold 15개의 평균이고, 별표는 기준(clinic4)보다 유의하게 높은 경우(Nadeau-Bengio 보정 t)입니다.", text_h=1700000, size=15)
    D.picture(s, FIG / "topk.png", 2800000, 3450000)
    # landmark coefficient 순위 (슬라이드 15): 풀링 5-fold의 train fold 15개에서 순위
    sel = pd.read_excel(O / STEPS[7] / "clinic4_step7_selection.xlsx"); sel = sel[sel["eval"] == "pooled 5-fold CV"].set_index(["landmark", "disease"])
    s = D.page("Appendix C3. Landmark coefficient ranking", "Appendix", [], "12곳 전체 모델에서 4조직 계수 크기(|표준화 계수| 합)가 큰 순서대로 높이마다 순위를 매겼습니다. 표의 숫자는 두 병원 합친 5-fold CV(3회 반복)의 train fold 15개에서 낸 평균 순위(1이 가장 큼)이고, 괄호는 상위 3위 안에 든 비율입니다. 순위는 train 데이터 안에서만 계산했습니다. 평균 순위가 가장 높은 높이는 " + ", ".join(f"{d} {LMK[sel.xs(d, level='disease').mean_rank.idxmin()]}" for d in DIS) + "입니다.")
    D.table(s, ["Landmark"] + [f"{d}\n평균 순위 (상위 3위 비율)" for d in DIS], [[LMK[a]] + [f"{sel.loc[(a, d), 'mean_rank']:.1f} ({sel.loc[(a, d), 'top3_freq'] * 100:.0f}%)" for d in DIS] for a in LMK], col_w=[2, 2.4, 2.4, 2.4])
    # 조직 일부만 더했을 때 (슬라이드 16, 추가 연구)
    s9 = pd.read_excel(O / "9_clinic4_tissue_subsets/clinic4_step9.xlsx")
    sub_rows = [[r[0].replace(" 12곳", "").replace("+ ", "+"), r[1]] + r[2:] for r in step_rows(s9)]
    for ttl, ns, nt in [("two tissues", ("28",), "지금까지는 12곳마다 VAT·SAT·TAMA·AEC 4종을 모두 넣었습니다. 여기서는 4종 중 일부 조직만 12곳에 넣었을 때(1종 4가지, 2종 6가지, 3종 4가지, 4종 1가지) 점수가 어떻게 달라지는지 확인했습니다. 두 병원 합친 5-fold CV 기준이고, 괄호는 기준(clinic4) 대비 ΔAUC와 p값입니다. 조합이 많아 다중비교 보정은 하지 않았으므로 탐색적 결과입니다. 질환별로 ΔAUC가 가장 큰 조합은 " + ", ".join(f"{d} {g.loc[g.dAUC.idxmax(), 'feature_set'].replace('+', '', 1).replace(' 12곳', '')}(+{g.dAUC.max():.3f})" for d, g in ((d, s9[(s9['eval'] == 'pooled 5-fold CV') & (s9.disease == d)]) for d in DIS)) + "입니다. 1종 4가지 결과는 Appendix B6과 같습니다."), ("three or four tissues", ("40", "52"), "3종 조합 4가지와 4종 전체(12곳 전체 raw 값, 입력 52개)입니다. 기준과 표 읽는 법은 앞 슬라이드와 같습니다.")]:
        s = D.page(f"Appendix C{4 + ['two tissues', 'three or four tissues'].index(ttl)}. Additional study: {ttl}", "Appendix", [], nt)
        D.table(s, ["넣은 조직 (12곳)", "입력 수"] + DIS, [sub_rows[0]] + [r for r in sub_rows[1:] if r[1] in ns], col_w=[2.0, .7, 3.15, 3.15, 3.15])
    D.save()


if __name__ == "__main__":
    main()
