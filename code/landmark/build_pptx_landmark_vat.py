from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark 체성분 AUC 연구 결과(outputs/clinic4/landmark/landmark_*.xlsx)를 docs/261003_landmark_VAT_AUC.pptx로 생성.
# 데이터셋: 2026-10-06 버전(gangnam 1,443 / sinchon 1,145명, 조직 = VAT, SAT, TAMA), 내부 평가 = gangnam 7/1/2 hold-out test, 외부 = sinchon.
# 템플릿 = docs/261002_연구세미나발표자료.pptx (표지 layout, 제목 26pt/1F3B63, 핵심문장 placeholder, 구분선, 섹션 배지, 슬라이드 번호
# 도형을 템플릿 슬라이드에서 복제). 템플릿의 기존 28장은 새 슬라이드를 만든 뒤 전부 제거한다. 종료 표지는 항상 Appendix 앞.
# 표의 "# Feat" = clinic4 4개(성별/나이/신장/체중)를 포함한 모델 입력 feature 총개수. 같은 값을 각 xlsx summary 시트에도 n_features로 저장한다.
# 슬라이드 노트(비전공자용 설명)는 landmark_vat_notes.NOTES (슬라이드 순서와 개수가 같아야 함).

import copy
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

import clinic4_landmark_level_auc as lvl
import clinic4_landmark_muscle_region_auc as reg
import clinic4_landmark_vat_auc as vat
from landmark_vat_notes import NOTES
from clinic4_logistic_regression import CLINIC4_DIR, PROJECT_ROOT, format_floats, save_sheet

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TEMPLATE = PROJECT_ROOT / "docs" / "261002_연구세미나발표자료.pptx"
OUT = PROJECT_ROOT / "docs" / "261006_landmark_VAT_AUC.pptx"  # 데이터셋 갱신 + 7/1/2 hold-out + 성별 층화 버전(이전 261003 덱은 그대로 둠)
LMDIR = PROJECT_ROOT / "outputs" / "clinic4" / "landmark"
FIG = LMDIR / "landmark_or_heatmap.png"
DIS = ["HTN", "DM", "CKD"]
NAVY, TEXT = RGBColor(0x1F, 0x3B, 0x63), RGBColor(0x33, 0x3F, 0x4D)
TBL_STYLE = "{C083E6E3-FA7D-4D7B-A595-EF9225AFEA82}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


# 모델명 -> 입력 feature 총개수(clinic4 4개 포함, baseline 이후는 VAT 합 1개 포함)
def n_features(model_sets: dict[str, list[str]]) -> dict[str, int]:
    return {m: 4 if m == "baseline" else 5 + len(e) for m, e in model_sets.items()}


NFEAT = {**n_features(vat.model_sets()), **n_features(lvl.model_sets()), **n_features(reg.MODELS)}


def read(xlsx: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(LMDIR / xlsx, sheet_name=sheet)
    for c in df.columns:
        if c not in ("disease", "model", "cohort", "term", "landmark", "tissue", "feature", "ref", "fold_aucs", "best_penalty", "selected_all_data",
                     "sex", "PatientSex"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# summary 시트를 읽고 n_features 열을 보장(없으면 모델 정의에서 계산해 시트에도 저장). NFEAT에 모델별 개수를 모은다
def with_nfeat(xlsx: str) -> pd.DataFrame:
    s = read(xlsx, "summary")
    if "n_features" not in s:
        s["n_features"] = s["model"].map(NFEAT)
        cols = list(s.columns)
        cols.insert(cols.index("model") + 1, cols.pop(cols.index("n_features")))
        save_sheet(format_floats(s[cols]), f"landmark/{xlsx}", "summary")
    NFEAT.update(dict(zip(s.model, s.n_features.astype(int))))
    return s


def auc_rows(summary: pd.DataFrame, models: dict[str, str]) -> list[list[str]]:
    piv = summary.pivot(index="model", columns="disease", values=["internal_auc", "external_auc"])
    return [[label, str(NFEAT[m])] + [f"{piv.loc[m, (k, d)]:.3f}" for d in DIS for k in ("internal_auc", "external_auc")]
            for m, label in models.items()]


AUC_HEADER = ["Model", "# Feat"] + [f"{d} {k}" for d in DIS for k in ("Int", "Ext")]


# 단일 landmark x 조직 OR 히트맵(질환별 패널). 두 코호트 모두 p<0.05 & 같은 방향이면 굵은 테두리
def or_heatmap(ors: pd.DataFrame) -> None:
    lms, tis = list(dict.fromkeys(ors["landmark"])), list(dict.fromkeys(ors["tissue"]))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.0), sharey=True)
    for ax, d in zip(axes, DIS):
        g = ors[ors.disease == d]
        mat, sig = np.full((len(lms), len(tis) * 2), np.nan), np.zeros((len(lms), len(tis) * 2), dtype=bool)
        for i, a in enumerate(lms):
            for k, t in enumerate(tis):
                rr = g[(g.landmark == a) & (g.tissue == t)].set_index("cohort")
                if len(rr) < 2:
                    continue
                both = (rr.p_value < 0.05).all() and np.sign(np.log(rr.OR_per_SD)).nunique() == 1
                for c, coh in enumerate(("gangnam", "sinchon")):
                    mat[i, 2 * k + c], sig[i, 2 * k + c] = np.log(rr.loc[coh, "OR_per_SD"]), both
        ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            if not np.isnan(v):
                ax.text(j, i, f"{np.exp(v):.2f}", ha="center", va="center", fontsize=8, fontweight="bold" if sig[i, j] else "normal")
                if sig[i, j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.6))
        ax.set_xticks(range(len(tis) * 2), [f"{t}-{'Gangnam' if c == 0 else 'Sinchon'}" for t in tis for c in (0, 1)], fontsize=8, rotation=90)
        ax.set_yticks(range(len(lms)), [a.replace("_center", "").replace("_margin", "") for a in lms], fontsize=9)
        ax.set_title(d, fontsize=13)
    fig.suptitle("단일 anchor 체성분(+1SD)당 OR  |  굵은 테두리 = Gangnam·Sinchon 모두 p<0.05 & 같은 방향", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation(TEMPLATE)
        self.n_orig = len(self.prs.slides)
        tpl = self.prs.slides[7]  # Table 1 슬라이드: 구분선/배지/번호 도형 원본
        shp = {s.name: s for s in tpl.shapes}
        self.rule, self.badge, self.num = (copy.deepcopy(shp[n]._element) for n in ("Rectangle 3", "Rounded Rectangle 4", "슬라이드 번호 개체 틀 3"))
        self.cover_boxes = [copy.deepcopy(s._element) for s in self.prs.slides[0].shapes]
        self.end_blob = next(sh for sh in self.prs.slides[19].shapes if sh.shape_type == 13).image.blob  # 템플릿 20번 = 종료 표지

    # 종료 표지(항상 Appendix 앞): 템플릿의 종료 슬라이드 그림을 전체 화면으로
    def end_cover(self) -> None:
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        s.shapes.add_picture(io.BytesIO(self.end_blob), 0, 0, self.prs.slide_width, self.prs.slide_height)

    def cover(self, title_lines: list[str], date: str) -> None:
        s = self.prs.slides.add_slide(self.prs.slide_layouts[0])
        for el in self.cover_boxes:
            s.shapes._spTree.append(copy.deepcopy(el))
        boxes = list(s.shapes)
        paras = boxes[0].text_frame.paragraphs
        for p, t in zip(paras, title_lines):
            p.runs[0].text = t
        boxes[1].text_frame.paragraphs[0].runs[0].text = date

    # 제목 + 핵심 문장(placeholder) + 구분선/배지/번호. lines는 핵심 문장 목록(단락), 반환: 슬라이드
    def page(self, title: str, badge: str, lines: list[str], text_h: int = 931520, size: int = 18) -> object:
        s = self.prs.slides.add_slide(self.prs.slide_layouts[1])
        s.shapes.title.left, s.shapes.title.top, s.shapes.title.width, s.shapes.title.height = 508000, 365126, 11176000, 532342
        r = s.shapes.title.text_frame.paragraphs[0].add_run()
        r.text, r.font.size, r.font.bold, r.font.color.rgb = title, Pt(26), True, NAVY
        body = s.placeholders[1]
        body.left, body.top, body.width, body.height = 508000, 1050000, 11176000, text_h
        for k, line in enumerate(lines):
            p = body.text_frame.paragraphs[0] if k == 0 else body.text_frame.add_paragraph()
            r = p.add_run()
            r.text, r.font.size, r.font.bold, r.font.color.rgb = line, Pt(size), False, TEXT
        for el in (self.rule, self.badge, self.num):
            s.shapes._spTree.append(copy.deepcopy(el))
        for sh in s.shapes:
            if sh.name == "Rounded Rectangle 4":
                sh.text_frame.paragraphs[0].runs[0].text = badge
        return s

    def table(self, s, header: list[str], rows: list[list[str]], top: int, height: int, col_w: list[float],
              merge_col0: bool = False, hs: int = 14, bs: int = 12) -> None:
        nr, nc, width = len(rows) + 1, len(header), 11176000
        gf = s.shapes.add_table(nr, nc, Emu(508000), Emu(top), Emu(width), Emu(height))
        tbl = gf.table
        tbl._tbl.tblPr.set("firstRow", "1")
        tbl._tbl.tblPr.set("bandRow", "1")
        tbl._tbl.tblPr.find(f"{A}tableStyleId").text = TBL_STYLE
        for j, w in enumerate(col_w):
            tbl.columns[j].width = int(width * w / sum(col_w))
        for i in range(nr):
            tbl.rows[i].height = height // nr
        if merge_col0:  # 병합을 먼저, 값은 시작 셀에만 쓴다
            i = 1
            while i < nr:
                j = i
                while j + 1 < nr and rows[j][0] == rows[i - 1][0]:
                    j += 1
                if j > i:
                    tbl.cell(i, 0).merge(tbl.cell(j, 0))
                i = j + 1
        for i in range(nr):
            vals = header if i == 0 else rows[i - 1]
            for j in range(nc):
                cell = tbl.cell(i, j)
                if cell.is_spanned:
                    continue
                cell.text_frame.text = vals[j]
                p = cell.text_frame.paragraphs[0]
                p.alignment = 2  # PP_ALIGN.CENTER
                cell.vertical_anchor = 3  # MSO_ANCHOR.MIDDLE
                r = p.runs[0]
                r.font.size, r.font.bold = Pt(hs if i == 0 else bs), i == 0

    def save(self, notes: list[str]) -> None:
        new = list(self.prs.slides)[self.n_orig:]
        assert len(new) == len(notes), (len(new), len(notes))
        for sl, text in zip(new, notes):
            sl.notes_slide.notes_text_frame.text = text
        ids = self.prs.slides._sldIdLst
        for sld in list(ids)[:self.n_orig]:  # 템플릿 원본 슬라이드 제거(rel도 같이 끊어 고아 파트가 남지 않게)
            ids.remove(sld)
            self.prs.part.drop_rel(sld.rId)
        self.prs.save(OUT)
        print(f"Saved {OUT} ({len(self.prs.slides)} slides)")


def main() -> None:
    s_vat, s_lvl = with_nfeat("landmark_vat_auc.xlsx"), with_nfeat("landmark_level_auc.xlsx")
    with_nfeat("landmark_muscle_region_auc.xlsx")
    s_sub, s_rat = with_nfeat("landmark_anchor_subset_auc.xlsx"), with_nfeat("landmark_anchor_ratio_auc.xlsx")
    or_vat, or_single = read("landmark_vat_auc.xlsx", "odds_ratio"), read("landmark_level_auc.xlsx", "odds_ratio_single")
    perf = read("landmark_anchor_k_sweep.xlsx", "perf_by_k")
    ors_k, rank_k = read("landmark_anchor_k_sweep.xlsx", "or_by_k"), read("landmark_anchor_k_sweep.xlsx", "ranking_gangnam")
    s_sex = read("landmark_sex_stratified.xlsx", "summary")
    or_heatmap(or_single)

    d = Deck()
    d.cover(["Anchor-Level Body Composition Features", "for Chronic Disease Prediction"], "2026. 10. 06")

    d.page("Study Design", "METHODS", [
        "코호트(2026-10-06 데이터셋): gangnam 1,436명 (HTN 469 / DM 291 / CKD 135), sinchon 1,129명 (HTN 492 / DM 338 / CKD 171). landmark 순서 이상 환자(gangnam 7, sinchon 16명) 제외",
        "baseline = clinic4 (성별, 나이, 신장, 체중) — feature 4개.  Model1 = baseline + liver_dome ~ pubis 구간 VAT 단면적 합 — feature 5개",
        "Model2~ = Model1 + 12개 anchor(liver_dome, T10~T12, L1~L5, S1, femoral_head, pubis)의 체성분 단면 값(cm²). 조직은 VAT, SAT, TAMA(근육 전체) 3종, VAT@pubis는 항상 0이라 제외 → 후보 35개",
        "내부 평가: gangnam 1:1 언더샘플링 후 train / valid / test = 7 / 1 / 2 hold-out (valid로 GridSearchCV 튜닝, test AUC 보고). 외부 평가: train+valid로 재학습한 모델을 sinchon(1:1 균형)에 적용",
        "test가 작아(HTN 188 / DM 117 / CKD 54명) 내부 AUC의 95% CI 폭이 큼 (Model1: HTN 0.67~0.82, DM 0.53~0.73, CKD 0.73~0.93). 모델 간 비교는 paired DeLong(BH-FDR)",
        "연관성: 전체 코호트 로지스틱 회귀 +1SD당 OR (clinic4 + VAT 합 보정). 단면 데이터라 HR/RR은 산출 불가",
    ], text_h=4500000, size=16)

    # feature 추출 확인: landmark_report 샘플 환자 1명(VAT/SAT 값 일치, TAMA = 리포트 NAMA+LAMA+IMATA 합과 12 anchor 모두 일치)
    s = d.page("Feature Extraction: Sample Patient Check", "METHODS", [
        "CT를 자동 분할해 조직(SAT, VAT, 근육)을 구분 — 리포트는 근육을 NAMA/LAMA/IMATA로 나눠 표시, 분석 데이터는 합(TAMA)을 사용",
        "12개 anchor 위치의 단면에서 조직별 면적(cm²)을 추출 → 이 값이 분석의 feature",
        "내장지방 합(Model1) = liver_dome ~ pubis 구간 모든 slice의 VAT 면적 합",
        "확인: 샘플 환자(ID 864318)의 리포트와 분석 데이터를 대조 → VAT·SAT 24개 값과 slice 번호가 일치, TAMA는 12개 anchor 모두 리포트의 NAMA+LAMA+IMATA 합과 일치 (소수 둘째 자리까지)",
        "예) L3 (slice 73): VAT 139.27, SAT 181.30, TAMA 79.76 (= 53.05 + 23.16 + 3.55) cm²",
        "색: 노랑 SAT, 빨강 VAT, 파랑 NAMA, 하늘 LAMA, 초록 IMATA",
    ], text_h=5300000, size=16)
    body = s.placeholders[1]
    body.left, body.top, body.width, body.height = 6350000, 1050000, 5334000, 5300000
    s.shapes.add_picture(str(LMDIR / "landmark_report_sample_864318.png"), Emu(508000), Emu(1000000), height=Emu(5500000))

    s = d.page("Table 1. Anchor Values Add Little AUC", "RESULTS",
               ["12 anchor 값을 더해도 Model1 대비 일관된 향상이 없음: HTN은 외부 +0.005 이내, DM은 내부만 +0.01~0.07(test 117명, 노이즈)이고 외부는 ±0.015 이내, CKD는 내부·외부가 엇갈림. FDR 후 유의 모델 없음",
                "Int = gangnam hold-out test AUC, Ext = sinchon 외부 AUC, # Feat = clinic4 4개 포함 입력 feature 총개수"], text_h=1250000)
    models = {"baseline": "baseline (clinic4)", "Model1": "Model1 (+VAT 합)",
              "+VAT@all12": "+VAT @12 anchors", "+SAT@all12": "+SAT @12 anchors", "+TAMA@all12": "+TAMA @12 anchors",
              "+VAT+SAT@all12": "+VAT,SAT @12 anchors", "+all3@all12": "+VAT,SAT,TAMA @12 anchors"}
    d.table(s, AUC_HEADER, auc_rows(s_lvl, models), 2450000, 3500000, [3.4, 1, 1, 1, 1, 1, 1, 1])

    s = d.page("Table 2. Associations Replicate Mainly for DM", "RESULTS",
               ["두 코호트에서 모두 유의하고 방향이 같은 단일 값 5개: DM의 SAT@femoral_head(0.70/0.67)·TAMA@femoral_head(0.73/0.69)·SAT@pubis·SAT@L4, CKD의 SAT@S1(0.71/0.69). HTN은 없음",
                "OR = +1SD당, clinic4 + VAT 합 보정, anchor를 하나씩 투입 (35개 값은 서로 상관이 커서 동시 투입 OR은 불안정)"], text_h=1250000)
    rows = []
    for dis in DIS:
        for label, src, flt in (("VAT 합 (liver~pubis)", or_vat, lambda x: (x.model == "Model1") & (x.term == "VAT_sum")),
                                ("SAT @ femoral_head", or_single, lambda x: (x.landmark == "femoral_head_center") & (x.tissue == "SAT")),
                                ("TAMA @ femoral_head", or_single, lambda x: (x.landmark == "femoral_head_center") & (x.tissue == "TAMA"))):
            cells = []
            for coh in ("gangnam", "sinchon"):
                r = src[(src.disease == dis) & (src.cohort == coh) & flt(src)].iloc[0]
                cells += [f"{r.OR_per_SD:.2f} ({r.ci_low:.2f}-{r.ci_high:.2f})", f"{r.p_value:.3f}"]
            rows.append([dis, label] + cells)
    d.table(s, ["Disease", "Feature", "Gangnam OR (95% CI)", "p", "Sinchon OR (95% CI)", "p"], rows, 2450000, 3500000,
            [1, 2.2, 2.2, 0.8, 2.2, 0.8], merge_col0=True)

    s = d.page("Figure 1. OR Profile by Anchor and Tissue", "RESULTS",
               ["하부 anchor(L4, femoral_head, pubis)의 SAT·TAMA가 질환과 반대 방향(보호)으로 두 코호트에서 일치하는 값이 DM에 집중 (굵은 테두리 5쌍)"], text_h=600000)
    s.shapes.add_picture(str(FIG), Emu(1000000), Emu(1750000), width=Emu(10200000))

    # anchor 값 개수 k 스윕(1~35): AUC 표/곡선, k별 OR
    s = d.page("Table 3. AUC by Number of Selected Values (k)", "RESULTS",
               ["후보 35개 중 상위 k개를 추가해도 향상이 일관되지 않음: HTN 내부 0~+0.02·외부 ±0.005, DM 내부 k≥8에서 +0.05(test 117명) 외부 −0.01~+0.02, CKD 내부는 −0.03~+0.04로 출렁임(test 54명)",
                "순위 = clinic4+VAT 합 보정 후 feature별 |z| (train+valid 안에서만 계산, test 미사용), 모든 k를 GridSearchCV(24조합, valid 단일 fold)로 튜닝"], text_h=1250000)
    ks = [1, 3, 5, 10, 15, 20, 30, 35]
    piv = perf.pivot(index="k", columns="disease", values=["internal_auc", "external_auc"])
    base = s_vat[s_vat.model.isin(["baseline", "Model1"])].pivot(index="model", columns="disease", values=["internal_auc", "external_auc"])
    rows = [[lab, str(n)] + [f"{base.loc[m, (c, dis)]:.3f}" for dis in DIS for c in ("internal_auc", "external_auc")]
            for m, lab, n in (("baseline", "baseline (clinic4)", 4), ("Model1", "Model1 (+VAT 합)", 5))]
    rows += [[f"Model1 + top {k}", str(5 + k)] + [f"{piv.loc[k, (c, dis)]:.3f}" for dis in DIS for c in ("internal_auc", "external_auc")] for k in ks]
    d.table(s, AUC_HEADER, rows, 2450000, 3700000, [3.4, 1, 1, 1, 1, 1, 1, 1], hs=13, bs=12)

    s = d.page("Figure 2. AUC vs Number of Selected Values", "RESULTS",
               ["HTN은 내부 0.75~0.78, 외부 0.71로 거의 평평, DM은 k≥8에서 내부가 올라가나 외부는 평평, CKD 내부는 0.81~0.88로 크게 흔들림 (점선 baseline, 파선 Model1)"], text_h=600000)
    s.shapes.add_picture(str(LMDIR / "anchor_k_curve.png"), Emu(508000), Emu(2000000), width=Emu(11176000))

    s = d.page("Table 4. Odds Ratios Become Unstable as k Grows", "RESULTS",
               ["gangnam 순위 상위 feature의 OR: k=1(단독)에서는 DM의 SAT@femoral_head가 두 코호트에서 일치(0.70/0.67)하나 k=10, 35에서는 방향이 뒤집힘(sinchon 1.45~1.49). * p<0.05",
                "두 코호트에서 재현(p<0.05 & 같은 방향)되는 feature는 k별 0~2개이고 대부분 0~1개"], text_h=1250000)
    rows = []

    def cell(dis, k, coh, f):
        t = ors_k[(ors_k.disease == dis) & (ors_k.k == k) & (ors_k.cohort == coh) & (ors_k.term == f)]
        return "-" if t.empty else f"{t.OR_per_SD.iloc[0]:.2f}" + ("*" if t.p_value.iloc[0] < 0.05 else "")

    for dis in DIS:
        for f in rank_k[rank_k.disease == dis].feature.head(3):
            rows.append([dis, f.replace("_center", "").replace("inferior_pubic_margin", "pubis")] +
                        [cell(dis, k, c, f) for k in (1, 10, 35) for c in ("gangnam", "sinchon")])
    d.table(s, ["Disease", "Feature (rank 1-3)", "k=1 Gangnam", "k=1 Sinchon", "k=10 Gangnam", "k=10 Sinchon", "k=35 Gangnam", "k=35 Sinchon"],
            rows, 2450000, 3600000, [1, 2.6, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], merge_col0=True, hs=13)

    s = d.page("Table 5. Anchor Ratio Features", "RESULTS",
               ["anchor 간(anchor/L3)·anchor 내(VAT/SAT, 근육 점유율) 비율 55개도 향상이 작고 비일관: 내부 최대 +0.03(DM, test 117명), 외부 최대 +0.014. FDR 후 유의 모델 없음",
                "비율 하나씩의 OR은 DM에서 여러 개가 두 코호트 일치: TAMA 점유율@pubis 1.38/1.37, VAT/SAT@L4 1.27/1.31 (sinchon FDR<0.05). 비율은 log((a+1)/(b+1))"], text_h=1250000)
    models = {"baseline": "baseline (clinic4)", "Model1": "Model1 (+VAT 합)",
              "+VAT/SAT @anchors": "+VAT/SAT @12 anchors", "+TAMA/(TAMA+fat) @anchors": "+TAMA/(TAMA+fat) @12 anchors",
              "+tissue ratios @all anchors": "+조직 비율 2종 @12 anchors", "+all3 anchor/L3": "+VAT,SAT,TAMA anchor/L3",
              "+all ratios": "+모든 비율 (55개)"}
    d.table(s, AUC_HEADER, auc_rows(s_rat, models), 2450000, 3500000, [3.4, 1, 1, 1, 1, 1, 1, 1], hs=13)

    s = d.page("Table 6. A Priori Anchor Subsets", "RESULTS",
               ["해부학적 부위로 미리 정한 묶음도 일관된 향상이 없음. FDR 후 유의한 변화는 악화뿐(DM 외부 TAMA@mid −0.050, p_fdr 0.006). 최대 향상은 +0.02대(하부 척추 VAT+SAT의 DM 외부 +0.023, 대표 4곳 VAT+SAT의 CKD 외부 +0.024; 모두 p_fdr>0.08)",
                "이전 데이터(754/443명)에서 보였던 골반-DM 향상(외부 +0.05)은 이번에 재현되지 않음 (VAT+SAT@pelvis DM 외부 0.689 = Model1)"], text_h=1250000)
    models = {"baseline": "baseline (clinic4)", "Model1": "Model1 (+VAT 합)",
              "VAT+SAT@pelvis(femoral,pubis)": "+VAT,SAT @pelvis (2 anchors)", "all3@pelvis(femoral,pubis)": "+VAT,SAT,TAMA @pelvis",
              "VAT+SAT@lowspine(L4,L5,S1)": "+VAT,SAT @L4,L5,S1", "all3@lowspine(L4,L5,S1)": "+VAT,SAT,TAMA @L4,L5,S1",
              "VAT+SAT@rep4(T12,L3,S1,femoral)": "+VAT,SAT @T12,L3,S1,femoral", "all3@rep4(T12,L3,S1,femoral)": "+VAT,SAT,TAMA @T12,L3,S1,femoral"}
    d.table(s, AUC_HEADER, auc_rows(s_sub, models), 2450000, 3600000, [3.8, 1, 1, 1, 1, 1, 1, 1], hs=13)

    # 성별 층화(남/여 각각 따로 학습·평가, 성별 컬럼 제외)
    s = d.page("Table 7. Sex-Stratified Comparison", "RESULTS",
               ["성별로 나눠도 anchor 값의 향상은 없음(DeLong 유의 향상 0건). baseline 성능이 성별로 다름: HTN은 여성 0.82/0.77, 남성 0.66/0.69. OR 성별 이질성은 216개 중 10개(우연 수준), FDR<0.10 없음",
                "각 성별로 따로 학습·평가(성별 변수 제외). test가 매우 작음: 남 HTN 103 / DM 62 / CKD 36명, 여 HTN 85 / DM 55 / CKD 18명 → AUC의 95% CI 폭 ±0.1~0.25"], text_h=1250000)
    models = {"baseline": "baseline", "Model1": "Model1", "+VAT+SAT@all12": "+VAT,SAT @12 anchors", "+all3@all12": "+VAT,SAT,TAMA @12 anchors",
              "VAT+SAT@pelvis(femoral,pubis)": "+VAT,SAT @pelvis"}
    piv = s_sex.pivot_table(index=["sex", "model"], columns="disease", values=["internal_auc", "external_auc"])
    rows = []
    for sx, lab in (("M", "남성 (M)"), ("F", "여성 (F)")):
        for m, ml in models.items():
            rows.append([lab, ml] + [f"{piv.loc[(sx, m), (c, dis)]:.3f}" for dis in DIS for c in ("internal_auc", "external_auc")])
    d.table(s, ["Sex", "Model"] + [f"{dis} {k}" for dis in DIS for k in ("Int", "Ext")], rows, 2450000, 3800000, [1.2, 3.0, 1, 1, 1, 1, 1, 1],
            merge_col0=True, hs=13, bs=12)

    d.page("Conclusion", "CONCLUSION", [
        "정확도: Model1(clinic4 + VAT 합) 위에 anchor 값(12 anchor, k 스윕), 비율, FPCA, 부위 묶음 어느 방식으로 더해도 일관된 AUC 향상이 없음 — FDR 후 유의한 향상 0건, 외부 향상은 대체로 +0.02 이내",
        "VAT 합(Model1) 자체: baseline 대비 HTN 내부 +0.022 / 외부 +0.011, CKD 외부 +0.036(sinchon DeLong p=0.004, FDR 후 비유의), DM 내부는 −0.037(test 117명, 노이즈)",
        "연관성(OR): 두 코호트 일치 단일 값 5개 — DM의 femoral_head·pubis·L4 SAT와 femoral_head TAMA, CKD의 S1 SAT. DM 비율(TAMA 점유율@pubis 1.38/1.37 등)도 일치. HTN은 일치하는 값이 없고 VAT 합도 gangnam(1.79)에서 뚜렷하나 sinchon은 1.16(p=0.052). k가 커질수록 다변량 OR은 불안정(공선성)",
        "성별 층화: 남·여 따로 평가해도 향상 없음. baseline 성능이 성별로 다르고(HTN 여성 0.82 vs 남성 0.66) OR 성별 이질성은 우연 수준(10/216, FDR<0.10 없음)",
        "데이터·방법 주의: 7/1/2 hold-out이라 내부 test가 작아(HTN 188 / DM 117 / CKD 54명) 내부 AUC가 ±0.05~0.13 흔들림. 이전 데이터(754/443명)에서 보인 DM 골반 향상, CKD 향상은 이번에 재현되지 않음",
        "한계: 단일 seed 분할, 단면 연구(예후 불가), anchor 단면은 slice 1장이라 노이즈가 큼, liver_dome이 T10보다 뒤인 환자 다수(gangnam 46%, sinchon 13%). 다음: seed 반복 hold-out으로 안정성 확인, 체격(BMI) 보정",
    ], text_h=4500000, size=16)

    d.end_cover()  # 종료 표지는 항상 Appendix 바로 앞

    # Appendix: 전체 모델 표
    lvl_models = [m for m in dict.fromkeys(s_lvl.model) if m not in ("baseline", "Model1")]
    vat_models = [m for m in dict.fromkeys(s_vat.model) if m not in ("baseline", "Model1")]
    rat_models = [m for m in dict.fromkeys(s_rat.model) if m not in ("baseline", "Model1")]
    sub_models = [m for m in dict.fromkeys(s_sub.model) if m not in ("baseline", "Model1")]
    chunks = [(f"S{n}. Anchor Value Models ({n}/2)", s_lvl, lvl_models[k:k + 8], "L3 단독 / L1,L3,L5 / 12 anchors를 Model1에 추가. ")
              for n, k in enumerate(range(0, len(lvl_models), 8), 1)]
    chunks += [("S3. Ratio Feature Models", s_vat, vat_models[:7], "VAT/SAT, VAT/TAMA 등 전체 구간 합 비율과 landmark 비율. "),
               ("S4. FPCA Feature Models", s_vat, vat_models[7:], "조직별 곡선을 FPCA(PC 3개)로 요약. "),
               ("S5. Anchor Ratio Models", s_rat, rat_models, "anchor/L3, anchor 내 조직 비율. ")]
    chunks += [(f"S{6 + n}. A Priori Anchor Subset Models ({n + 1}/2)", s_sub, sub_models[k:k + 9],
                "위치 구간 부분집합 + nested 상위 k 선택(nested는 train+valid 안에서 선택). ") for n, k in enumerate(range(0, len(sub_models), 9))]
    for title, src, ms, note in chunks:
        s = d.page(title, "APPENDIX", ["Model1에 feature를 추가한 전체 AUC (Int = gangnam hold-out test, Ext = sinchon)",
                                       note + "# Feat = clinic4 4개 포함 입력 feature 총개수"], text_h=900000)
        d.table(s, AUC_HEADER, auc_rows(src, {m: m for m in ("baseline", "Model1", *ms)}), 1980000, 4300000,
                [3.8, 1, 1, 1, 1, 1, 1, 1], hs=13, bs=11)
    s = d.page("S8. AUC vs k: Ratio Pool vs Value Pool", "APPENDIX",
               ["비율 후보 55개(진한 선)와 값 후보 35개(옅은 선)에서 train+valid 안 순위 top-k를 추가한 AUC. 점선=baseline, 파선=Model1"], text_h=700000)
    s.shapes.add_picture(str(LMDIR / "anchor_ratio_k_curve.png"), Emu(508000), Emu(1900000), width=Emu(11176000))
    s = d.page("S9. Sex-Stratified ΔAUC vs Model1", "APPENDIX",
               ["대표 모델의 Model1 대비 AUC 변화를 남성(파랑)·여성(주황)으로 비교. 위: 내부(gangnam test), 아래: 외부(sinchon). 성별별 test가 작아 막대 크기에 비해 불확실성이 큼"], text_h=900000)
    s.shapes.add_picture(str(LMDIR / "sex_stratified_delta_auc.png"), Emu(1000000), Emu(1900000), width=Emu(10200000))
    d.save(NOTES)


if __name__ == "__main__":
    main()
