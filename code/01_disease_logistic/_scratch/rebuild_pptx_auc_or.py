from __future__ import annotations
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu, Pt, Inches
from pptx.enum.text import PP_ALIGN

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS = PROJECT_ROOT / "docs"
PPTX_PATH = DOCS / "260907.pptx"
IMG_DIR = PROJECT_ROOT / "outputs" / "0907" / "clinic_compare" / "dm"

prs = Presentation(str(PPTX_PATH))
layout_title = prs.slide_masters[0].slide_layouts[0]   # 제목 슬라이드
layout_content = prs.slide_masters[0].slide_layouts[1]  # 제목 및 내용 (title + 1 object placeholder)
layout_two = prs.slide_masters[0].slide_layouts[3]       # 콘텐츠 2개 (title + 2 object placeholders)

# ---------- Slide 1: 제목 슬라이드 텍스트 갱신 ----------
title_slide = prs.slides[0]
for shape in title_slide.shapes:
    if shape.name == "TextBox 1":
        tf = shape.text_frame
        p = tf.paragraphs[0]
        for run in list(p.runs)[1:]:
            run.text = ""
        p.runs[0].text = "Clinic Baseline AUC · Odds Ratio 재정리 (DM 예시)"
    elif shape.name == "TextBox 2":
        shape.text_frame.paragraphs[0].runs[0].text = "2026. 09. 08"

# ---------- 슬라이드 추가 헬퍼 ----------

def add_title_only(title: str):
    slide = prs.slides.add_slide(layout_content)
    slide.shapes.title.text_frame.text = title
    ph = slide.placeholders[1]
    ph._element.getparent().remove(ph._element)
    return slide


def add_bullets(title: str, bullets: list[str], notes: str = "") -> None:
    slide = prs.slides.add_slide(layout_content)
    slide.shapes.title.text_frame.text = title
    body = slide.placeholders[1].text_frame
    body.clear()
    body.word_wrap = True
    for i, text in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = text
        p.level = 0
        for run in p.runs:
            run.font.size = Pt(20)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes


def add_table(title: str, header: list[str], rows: list[list[str]], notes: str = "",
              col_widths: list[float] | None = None, footnote: str | None = None):
    slide = prs.slides.add_slide(layout_content)
    slide.shapes.title.text_frame.text = title
    ph = slide.placeholders[1]
    left, top, width, height = ph.left, ph.top, ph.width, ph.height
    if footnote:
        height = int(height * 0.88)
    ph._element.getparent().remove(ph._element)

    n_rows, n_cols = len(rows) + 1, len(header)
    graphic_frame = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = graphic_frame.table
    if col_widths:
        total = sum(col_widths)
        for c, w in zip(table.columns, col_widths):
            c.width = int(width * w / total)

    for c, text in enumerate(header):
        cell = table.cell(0, c)
        cell.text = text
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(14)

    for r, row in enumerate(rows, start=1):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(text)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size = Pt(13)

    if footnote:
        tb = slide.shapes.add_textbox(left, top + height, width, int(prs.slide_height * 0.06))
        tb.text_frame.text = footnote
        for run in tb.text_frame.paragraphs[0].runs:
            run.font.size = Pt(11)
            run.font.italic = True

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_table_and_picture(title: str, header: list[str], rows: list[list[str]], image_path: Path,
                           col_widths: list[float] | None = None, notes: str = ""):
    slide = prs.slides.add_slide(layout_two)
    slide.shapes.title.text_frame.text = title
    ph_left, ph_right = slide.placeholders[1], slide.placeholders[2]
    l_left, l_top, l_width, l_height = ph_left.left, ph_left.top, ph_left.width, ph_left.height
    r_left, r_top, r_width, r_height = ph_right.left, ph_right.top, ph_right.width, ph_right.height
    ph_left._element.getparent().remove(ph_left._element)
    ph_right._element.getparent().remove(ph_right._element)

    n_rows, n_cols = len(rows) + 1, len(header)
    graphic_frame = slide.shapes.add_table(n_rows, n_cols, l_left, l_top, l_width, l_height)
    table = graphic_frame.table
    if col_widths:
        total = sum(col_widths)
        for c, w in zip(table.columns, col_widths):
            c.width = int(l_width * w / total)
    for c, text in enumerate(header):
        cell = table.cell(0, c)
        cell.text = text
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(12)
    for r, row in enumerate(rows, start=1):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(text)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size = Pt(11)

    pic = slide.shapes.add_picture(str(image_path), r_left, r_top, height=r_height)
    if pic.width > r_width:
        ratio = r_width / pic.width
        pic.width, pic.height = r_width, int(pic.height * ratio)
    pic.left = r_left + (r_width - pic.width) // 2
    pic.top = r_top + (r_height - pic.height) // 2

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def add_picture_slide(title: str, image_path: Path, bullets: list[str] | None = None, notes: str = ""):
    slide = prs.slides.add_slide(layout_content)
    slide.shapes.title.text_frame.text = title
    ph = slide.placeholders[1]
    left, top, width, height = ph.left, ph.top, ph.width, ph.height
    if bullets:
        height = int(height * 0.68)
    ph._element.getparent().remove(ph._element)

    pic = slide.shapes.add_picture(str(image_path), left, top, height=height)
    if pic.width > width:
        ratio = width / pic.width
        pic.width, pic.height = width, int(pic.height * ratio)
    pic.left = left + (width - pic.width) // 2
    pic.top = top

    if bullets:
        tb_top = pic.top + pic.height + Emu(91440)
        tb = slide.shapes.add_textbox(left, tb_top, width, height := (top + int(ph.height) - tb_top))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, text in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = text
            for run in p.runs:
                run.font.size = Pt(16)

    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


# ---------- Slide 2: 목차 ----------
add_bullets("목차", [
    "Research Summary",
    "실험설계: Baseline 6종 · AEC Feature 2종",
    "결과 ① Baseline별 AUC 요약 (HTN/DM/CKD)",
    "결과 ② DM AUC — Baseline vs AEC 추가",
    "결과 ③ DM AUC 개선 유의성 (DeLong)",
    "결과 ④ DM Odds Ratio — Baseline 계수",
    "결과 ⑤ DM Odds Ratio — AEC 항 요약",
    "부록: 기타 강건성 체크 (CKD·AEC-total·t-SNE)",
    "Conclusion & Next Plans",
])

# ---------- Slide 3: Research Summary ----------
add_bullets("Research Summary", [
    "Baseline 6종 × AEC feature 2종(FPCA(3)/Upper-Lower ratio) × 3질환 × 2코호트 = 72개 DeLong 검정",
    "DM 예시: baseline AUC 0.66~0.73(external), AEC 추가 시 ΔAUC +0.008~+0.027",
    "DM external 개별유의 1/12(clinic9+FPCA, p=0.006) — Bonferroni(α=0.05/72≈0.0007) 미달",
    "Odds Ratio: AEC Upper/Lower ratio는 일관되게 위험증가(OR 1.51~1.67), FPCA PC2는 일관되게 보호적(OR 0.58~0.63)",
    "다중비교 보정 후 AEC 추가효과는 전 질환·전 baseline에서 강건하지 않음(0/72)",
], notes=(
    "AUC는 모델이 질병 있는 사람과 없는 사람을 얼마나 잘 구별하는지를 0.5(무작위)~1.0(완벽)로 나타낸 수치이고, "
    "Odds Ratio(승산비)는 특정 변수가 1단위 증가할 때 질병 발생 승산(오즈)이 몇 배가 되는지를 나타냅니다(1보다 크면 위험증가, 작으면 보호적). "
    "이번 발표는 질병 예시를 당뇨병(DM) 하나로 통일해서 보여드립니다. "
    "핵심은 \'AEC 곡선 정보를 추가하면 AUC가 약간 올라가고 Odds Ratio 방향도 일관되지만, "
    "여러 번 반복 검정한 것을 감안한 보정(Bonferroni)을 거치면 통계적으로 강건한 개선이라고 단언하기는 어렵다\'는 것입니다."
))

# ---------- Slide 4: 실험설계 - Baseline 6종 정의 ----------
add_table("실험설계 ① Baseline 6종 정의",
    ["Baseline", "구성 변수", "비고"],
    [
        ["clinic4", "Age, Sex, Height, Weight", "기본 baseline"],
        ["clinic6 (scanner)", "clinic4 + kVp, Vendor(정수라벨)", "스캐너 요인 추가"],
        ["clinic7", "clinic4 + VAT, SAT, SMI", "체성분 3종 추가"],
        ["clinic6 (VSR)", "clinic4 + VAT/SAT ratio, SMI", "체성분을 비율 1개로 압축"],
        ["clinic9 (all)", "clinic4 + kVp, Vendor, VAT, SAT, SMI", "전부 결합 (4+5=9)"],
        ["clinic8 (all ratio)", "clinic4 + kVp, Vendor, VAT/SAT ratio, SMI", "전부 결합, 체성분은 비율 (4+4=8)"],
    ],
    col_widths=[2.2, 4.5, 2.5],
)

# ---------- Slide 5: 실험설계 - AEC Feature·검증 원칙 ----------
add_bullets("실험설계 ② AEC Feature · 검증 원칙", [
    "FPCA(3): AEC-128 곡선 → internal PCA → n_components 고정 3 (elbow는 참고진단값)",
    "Upper/Lower ratio: 앞 64 slice(liver측) 평균 / 뒤 64 slice(hip측) 평균",
    "Hyperparameter: internal 5-fold CV grid search(C × L1/L2/ElasticNet) → 질환·모델별 best 고정",
    "Validation discipline: internal = 모델 선택/탐색 전용, external = 동결 모델 1회 최종평가만",
    "이 원칙은 AUC와 Odds Ratio 둘 다에 동일하게 적용(external 계수는 보고하지 않고 internal full-fit 계수만 사용)",
])

# ---------- Slide 6: 결과① Baseline별 AUC 요약 ----------
add_table("결과 ① Baseline별 AUC 요약",
    ["Baseline", "HTN (int/ext)", "DM (int/ext)", "CKD (int/ext)"],
    [
        ["clinic4", "0.794 / 0.719", "0.722 / 0.666", "0.768 / 0.616"],
        ["clinic6 (scanner)", "0.793 / 0.712", "0.721 / 0.659", "0.772 / 0.614"],
        ["clinic7", "0.818 / 0.735", "0.726 / 0.681", "0.784 / 0.625"],
        ["clinic6 (VSR)", "0.812 / 0.743", "0.730 / 0.694", "0.784 / 0.625"],
        ["clinic9 (all)", "0.817 / 0.731", "0.723 / 0.667", "0.786 / 0.628"],
        ["clinic8 (all ratio)", "0.811 / 0.737", "0.728 / 0.685", "0.788 / 0.629"],
    ],
    footnote="값은 baseline 단독(AEC 미포함) 모델의 AUC, internal은 5-fold CV OOF, external은 동결평가",
    notes=(
        "AEC를 전혀 넣지 않은 6가지 baseline 조합만으로 예측했을 때의 성능(AUC)입니다. "
        "고혈압(HTN)이 세 질환 중 가장 예측이 잘 되고(0.79~0.82), 당뇨병(DM)은 중간(0.72~0.73/0.66~0.69), "
        "만성콩팥병(CKD)이 가장 어렵습니다(0.61~0.63). 체지방이나 근육량 지표를 더한 baseline이 대체로 소폭 더 좋아서 "
        "이후 슬라이드의 AEC 추가효과를 해석할 때 '체성분 정보가 이미 baseline에 들어있으면 AEC가 줄 수 있는 추가정보가 적어진다'는 맥락으로 참고할 수 있습니다."
    ),
)

# ---------- Slide 7: 결과② DM AUC — Baseline vs AEC 추가 ----------
add_table_and_picture("결과 ② DM AUC — Baseline vs AEC 추가",
    ["Baseline", "Baseline\n(int/ext)", "+FPCA(3)\n(int/ext)", "+Up/Low ratio\n(int/ext)"],
    [
        ["clinic4", "0.722/0.665", "0.740/0.681", "0.737/0.677"],
        ["clinic6(scanner)", "0.721/0.659", "0.742/0.671", "0.739/0.670"],
        ["clinic7", "0.726/0.681", "0.741/0.696", "0.739/0.691"],
        ["clinic6(VSR)", "0.730/0.694", "0.743/0.706", "0.741/0.702"],
        ["clinic9(all)", "0.723/0.667", "0.739/0.695", "0.739/0.688"],
        ["clinic8(all ratio)", "0.728/0.685", "0.742/0.702", "0.741/0.695"],
    ],
    image_path=IMG_DIR / "dm_roc_curve_aec_compare.png",
    col_widths=[2.4, 1.7, 1.7, 1.9],
    notes=(
        "당뇨병(DM)을 예시로, 6가지 baseline에 AEC를 FPCA 또는 Upper/Lower 비율로 추가했을 때 AUC 변화입니다. "
        "모든 baseline에서 둘 다 baseline보다 약간 증가(internal +0.01~0.02, external +0.008~0.027)하는 일관된 경향을 보입니다. "
        "오른쪽 ROC 곡선(clinic4 예시)을 보면 baseline과 AEC 추가 모델 곡선이 거의 겹칠 정도로 비슷해 차이가 크지 않음을 시각적으로도 확인할 수 있습니다."
    ),
)

# ---------- Slide 8: 결과③ DM AUC 개선 유의성 (DeLong) ----------
add_table("결과 ③ DM AUC 개선 유의성 (DeLong, external)",
    ["Baseline", "ΔAUC (+FPCA)", "p (+FPCA)", "ΔAUC (+Up/Low)", "p (+Up/Low)"],
    [
        ["clinic4", "+0.0154", "0.072", "+0.0116", "0.226"],
        ["clinic6(scanner)", "+0.0128", "0.167", "+0.0115", "0.266"],
        ["clinic7", "+0.0152", "0.039*", "+0.0103", "0.185"],
        ["clinic6(VSR)", "+0.0118", "0.119", "+0.0078", "0.320"],
        ["clinic9(all)", "+0.0273", "0.006*", "+0.0206", "0.046*"],
        ["clinic8(all ratio)", "+0.0165", "0.080", "+0.0092", "0.327"],
    ],
    col_widths=[2.4, 1.8, 1.4, 1.8, 1.4],
    footnote="* p<0.05(개별 검정). Bonferroni 임계값 α=0.05/72≈0.0007 — 3건 모두 미달(다중비교 보정 미생존)",
    notes=(
        "DM external 코호트에서 AEC 추가의 DeLong 검정 결과입니다. clinic9(all)+FPCA에서만 p=0.006으로 개별적으로 유의하고, "
        "clinic9+Upper/Lower도 p=0.046으로 간신히 유의하지만, 이를 전체 72번 검정을 감안한 Bonferroni 기준(약 0.0007)으로 보면 셀 다 미달합니다. "
        "즉, DM도 다른 질환과 마찬가지로 AEC 추가가 통계적으로 확실한 개선으로는 입증되지 않았습니다."
    ),
)

# ---------- Slide 9: 결과④ DM Odds Ratio — Baseline(clinic4) ----------
add_picture_slide("결과 ④ DM Odds Ratio — Baseline(clinic4)",
    IMG_DIR / "dm_or_forest.png",
    bullets=[
        "Age/Sex/Weight ↑ → DM 위험 ↑(OR>1), Height ↑ → 위험 ↓(OR<1) — 임상적 방향과 일치",
        "AEC 추가(FPCA/Upper-Lower ratio) 후에도 clinic4 4개 변수의 방향성은 유지되고 크기만 소폭 감소(다중공선성 반영)",
    ],
    notes=(
        "clinic4(baseline)만으로 만든 당뇨병 예측 모델의 승산비(Odds Ratio) 그림입니다. 점이 1보다 오른쪽에 있을수록(예: 나이, 체중) "
        "해당 변수가 클수록 당뇨병 위험이 높아진다는 뜻이고, 왼쪽에 있으면(예: 키) 반대로 위험이 낮아집니다. "
        "세 모델(baseline만 / FPCA 추가 / Upper-Lower 비율 추가) 모두에서 이 방향이 또같이 유지되는 것을 확인할 수 있습니다."
    ),
)

# ---------- Slide 10: 결과⑤ DM Odds Ratio — AEC 항 요약 ----------
add_table("결과 ⑤ DM Odds Ratio — AEC 항 요약 (internal full-fit)",
    ["Baseline", "OR(Up/Low ratio)", "OR(FPCA PC1)", "OR(FPCA PC2)", "OR(FPCA PC3)"],
    [
        ["clinic4", "1.560", "1.093", "0.611", "1.148"],
        ["clinic6(scanner)", "1.667", "1.061", "0.579", "1.145"],
        ["clinic7", "1.547", "1.111", "0.627", "1.090"],
        ["clinic6(VSR)", "1.513", "1.126", "0.614", "1.111"],
        ["clinic9(all)", "1.664", "1.137", "0.576", "1.080"],
        ["clinic8(all ratio)", "1.590", "1.153", "0.584", "1.106"],
    ],
    col_widths=[2.4, 2.0, 1.8, 1.8, 1.8],
    footnote="OR>1: 위험 증가 방향, OR<1: 보호적 방향 (패널화(L1/ElasticNet) 계수라 CI는 미산출)",
    notes=(
        "AEC 관련 항들의 승산비를 6가지 baseline에 걸쳐 모은 표입니다. Upper/Lower 비율은 baseline을 무엇으로 하든 항상 1보다 크게(1.51~1.67) "
        "나와 위험을 높이는 방향이고, FPCA의 세 번째 주성분(PC2)만은 항상 1보다 작게(0.58~0.63) 나와 보호적 방향입니다. "
        "방향은 일관되지만 앞 슬라이드의 DeLong 검정에서 보았듯 통계적 유의성은 다중비교를 넘지 못해, '방향은 있지만 확실하지는 않다'로 요약됩니다."
    ),
)

# ---------- Slide 11: 부록 - 기타 강건성 체크 요약 ----------
add_bullets("부록: 기타 강건성 체크 요약", [
    "CKD external + Upper/Lower ratio: baseline 6종 전부 명목유의(ΔAUC +0.015~+0.023) — 유일한 baseline-불변 신호(HTN/DM엔 없음)",
    "AEC-total(비크롭) vs AEC-128: 72건 중 3건만 산발적 유의, 방향 비일관 → 크롭·리샘플링 정보손실 근거 없음",
    "t-SNE(raw/patient-wise): HTN/DM/CKD 군집 분리 신호 없음 — FPCA(선형) 결론과 일치",
], notes=(
    "AUC/Odds Ratio 외에 함께 진행한 3가지 보조 검증을 간단히 요약한 슬라이드입니다. "
    "만성콩팥병(CKD) 외부코호트에서만 Upper/Lower 비율이 baseline과 무관하게 항상 유의하게 개선되는 특이한 패턴이 있고, "
    "크롭하지 않은 전체 스캔 구간과 비교해도 성능이 통계적으로 동등해 크롭·리샘플링 과정에서 정보가 손실되지 않았음을 확인했습니다. "
    "비선형 시각화(t-SNE)로도 질병군이 특별히 분리되어 뭉치지 않아, 선형 방법(FPCA)으로 내렸던 기존 결론이 재확인되었습니다."
))

# ---------- Slide 12: Conclusion & Next Plans ----------
add_bullets("Conclusion & Next Plans", [
    "Conclusion",
    "DM 기준: AEC 추가 시 AUC +0.01~0.03 상승 경향, 다만 Bonferroni 보정 후에는 강건한 개선이 아님(external 개별유의 1/12, 다중비교 생존 0/12)",
    "Odds Ratio로 보면 AEC Upper/Lower ratio는 항상 위험증가 방향(OR>1), FPCA PC2는 항상 보호적 방향(OR<1) — 방향은 일관되나 통계적 유의성은 약함",
    "Baseline을 어떻게 확장해도(스캐너/체성분/비율) 위 결론은 바뀌지 않음(baseline-invariant)",
    "Next Plans",
    "CKD external Upper/Lower ratio 신호의 기전적 해석 추가 검토",
    "비열등성(세그멘테이션 대체) 프레이밍과의 정합성 재검토",
    "후속 후보: 연속형 검사수치(ΔR²), 사망 예후(follow-up cohort)로 평가축 확장",
])

prs.save(str(PPTX_PATH))
print("saved, total slides:", len(prs.slides))
