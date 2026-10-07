from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# landmark 단면 값(VAT/SAT/TAMA/AEC)의 질환(HTN/DM/CKD) 유무 차이 + 나이·성별·BMI 보정 OR -> docs/261006_landmark_disease_difference.pptx
# 입력: outputs/data_distribution/landmark_disease_difference.xlsx, landmark_adjusted_or.xlsx (landmark/ 하위 스크립트가 생성), 분포 boxplot png.
# 덱 뼈대(Deck)는 build_pptx_landmark_vat 재사용 (템플릿 표지/제목/배지, 종료 표지는 Appendix 앞).
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx.util import Emu, Pt

import build_pptx_landmark_vat as base
from clinic4_landmark_vat_auc import LM, load_cohort

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
D = base.PROJECT_ROOT / "outputs" / "data_distribution"
base.OUT = base.PROJECT_ROOT / "docs" / "261006_landmark_disease_difference.pptx"
DIS, MEAS = ["HTN", "DM", "CKD"], ["VAT", "SAT", "TAMA", "AEC"]
LMN = [a.replace("_center", "") for a in LM]
COH = {"gangnam": "Gangnam", "sinchon": "Sinchon"}

diff = pd.read_excel(D / "landmark_disease_difference.xlsx", sheet_name="detail")
adj = pd.read_excel(D / "landmark_adjusted_or.xlsx", sheet_name="detail")
feat = pd.read_excel(D / "landmark_feature_cases.xlsx", sheet_name="all")  # landmark_feature_cases.py 결과
rob = feat[feat.robust]
pw = pd.read_excel(D / "landmark_patientwise_preprocessing.xlsx", sheet_name="summary")  # landmark_patientwise_preprocessing.py 결과
scd = pd.read_excel(D / "landmark_scanner_check.xlsx", sheet_name="scanner_distribution")  # landmark_scanner_check.py 결과
scr = pd.read_excel(D / "landmark_scanner_check.xlsx", sheet_name="robust_pos_count")
pc1 = pd.read_excel(D / "landmark_pc1_adjusted.xlsx")  # landmark_pc1_adjusted.py 결과
seg = pd.read_excel(D / "landmark_segment_mean.xlsx", sheet_name="both_cohorts")  # landmark_segment_mean.py 결과
combo = pd.read_excel(D / "landmark_combo_features.xlsx", sheet_name="all")  # landmark_combo_features.py 결과
cset = pd.read_excel(D / "landmark_combo_features.xlsx", sheet_name="set_models")
mlp = pd.read_excel(D / "landmark_mlp_sets.xlsx")  # landmark_mlp_sets.py 결과 (early_stopping 끈 설정)
sga = pd.read_excel(D / "landmark_subgroup_auc.xlsx", sheet_name="all_directions")  # landmark_subgroup_auc.py 결과
sgm = pd.read_excel(D / "landmark_subgroup_auc.xlsx", sheet_name="matched_groups")
sub = pd.read_excel(D / "landmark_subset_search.xlsx", sheet_name="selected")  # landmark_subset_search.py 결과
fsel = pd.read_excel(D / "landmark_forward_selection.xlsx", sheet_name="final")  # landmark_forward_selection.py 결과
fpath = pd.read_excel(D / "landmark_forward_selection.xlsx", sheet_name="path")
subL = pd.read_excel(D / "landmark_subset_search_L.xlsx", sheet_name="selected")  # landmark_subset_search.py L 결과
PWV = ["raw", "pw_mean", "pw_z", "pw_minmax", "pw_L3"]


# 측정값 x 코호트(열) 대 landmark(행) 히트맵, 질환별 패널. 굵은 테두리 = FDR<0.05
def heatmap(df, val, fname, title, fmt, vlim):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6), sharey=True)
    for ax, d in zip(axes, DIS):
        cols = [(m, c) for m in MEAS for c in COH]
        mat, sg = np.full((len(LMN), len(cols)), np.nan), np.zeros((len(LMN), len(cols)), bool)
        for j, (m, c) in enumerate(cols):
            g = df[(df.disease == d) & (df.measure == m) & (df.cohort == c)].set_index("landmark")
            for i, a in enumerate(LMN):
                if a in g.index:
                    mat[i, j], sg[i, j] = val(g.loc[a]), g.loc[a, "p_fdr"] < .05
        ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            if not np.isnan(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7, fontweight="bold" if sg[i, j] else "normal")
                if sg[i, j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.4))
        ax.set_xticks(range(len(cols)), [f"{m}-{'G' if c == 'gangnam' else 'S'}" for m, c in cols], fontsize=8, rotation=90)
        ax.set_yticks(range(len(LMN)), [a.replace("_margin", "") for a in LMN], fontsize=9)
        ax.set_title(d, fontsize=13)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(D / fname, dpi=170)
    plt.close(fig)


heatmap(diff[diff.sex == "All"], lambda r: r.cohen_d, "ppt_unadjusted_d_heatmap.png",
        "질환 Yes-No 차이 Cohen d (미보정)  |  굵은 테두리 = FDR<0.05  |  G=Gangnam, S=Sinchon", lambda v: f"{v:.2f}", 1.0)
heatmap(adj[adj.sex == "All"], lambda r: np.log(r.OR_per_SD), "ppt_adjusted_or_heatmap.png",
        "+1SD당 OR (나이·성별·BMI 보정)  |  굵은 테두리 = FDR<0.05  |  G=Gangnam, S=Sinchon", lambda v: f"{np.exp(v):.2f}", 0.7)


# 셀 문자열: 값 + FDR<0.05면 별표
def cell(df, col, sex, d, c, m, a):
    r = df[(df.sex == sex) & (df.disease == d) & (df.cohort == c) & (df.measure == m) & (df.landmark == a)].iloc[0]
    return f"{r[col]:.2f}{'*' if r.p_fdr < .05 else ''}"


def cohort_table():
    rows = {}
    for c in COH:
        df = load_cohort(c)[0]
        rows[c] = [f"{len(df):,}", f"{(df.PatientSex == 'F').mean() * 100:.1f}%", f"{df.PatientAge.mean():.1f} ± {df.PatientAge.std():.1f}",
                   f"{df.BMI.mean():.1f} ± {df.BMI.std():.1f}"] + [f"{int(df[d].sum())} ({df[d].mean() * 100:.1f}%)" for d in DIS]
    names = ["N (QC 후)", "여성 비율", "나이 (mean ± SD)", "BMI (mean ± SD)", "HTN", "DM", "CKD"]
    return [[n, rows["gangnam"][i], rows["sinchon"][i]] for i, n in enumerate(names)]


CASES = [("point", ["point"]), ("lm_ratio", ["lm_ratio"]), ("range_mean", ["range_mean"]), ("range_frac", ["range_frac"]),
         ("조직 비율", ["tissue_ratio_pt", "range_ratio"])]


# 질환 x 조직(SAT/TAMA/AEC) x case별 "두 코호트 일관" feature 수 / 전체 수
def case_count_rows():
    rows = []
    for d in DIS:
        for t in ["SAT", "TAMA", "AEC"]:
            g = feat[(feat.disease == d) & (feat.tissue == t)]
            rows.append([d, t] + [f"{int(g[g.case.isin(cs)].robust.sum())} / {int(g.case.isin(cs).sum())}" for _, cs in CASES])
    return rows


# 질환 x 조직별 best 일관 feature (min |log OR| 최대)
def best_rows():
    rows = []
    for d in DIS:
        for t in ["SAT", "TAMA", "AEC"]:
            g = rob[(rob.disease == d) & (rob.tissue == t)].sort_values("min_abs_logOR", ascending=False)
            if g.empty:
                rows.append([d, t, "-", "일관된 feature 없음", "-", "-"])
            else:
                r = g.iloc[0]
                rows.append([d, t, r.case, r.feature, f"{r.OR_gangnam:.2f}", f"{r.OR_sinchon:.2f}"])
    return rows


# 측정값 x 질환 행, 전처리 열의 표: col="n_robust_pos"(일관 위치 수 / 128) 또는 "ext_auc_gain"(외부 AUC 변화)
def pw_rows(col):
    rows = []
    for m in ["VAT", "SAT", "TAMA", "AEC"]:
        for d in DIS:
            g = pw[(pw.measure == m) & (pw.disease == d)].set_index("variant")
            rows.append([m, d] + [f"{int(g.loc[v, col])}" if col == "n_robust_pos" else f"{g.loc[v, col]:+.3f}" for v in PWV])
    return rows


# 장비별 환자 수·여성 비율·질환 유병률
def scanner_dist_rows():
    return [[COH[r.cohort], r.scanner, str(r.n), f"{r.female * 100:.0f}%", f"{r.HTN_prev * 100:.0f}%", f"{r.DM_prev * 100:.0f}%", f"{r.CKD_prev * 100:.0f}%"] for r in scd.itertuples()]


# 측정값 x 질환 행, 전처리 열: "보정 전 → 장비 보정 후 / 최다 장비 단독" 일관 위치 수
def scanner_rows():
    rows = []
    for m in ["VAT", "SAT", "TAMA", "AEC"]:
        for d in DIS:
            g = scr[(scr.measure == m) & (scr.disease == d)].set_index("variant")
            rows.append([m, d] + [f"{int(g.loc[v, 'base'])} → {int(g.loc[v, 'scanner_adj'])} / {int(g.loc[v, 'top_scanner_only'])}" for v in ["pw_mean", "pw_z", "pw_L3"]])
    return rows


# 분석 단계별 근거 (PubMed에서 확인한 문헌만. 직접 근거가 없는 단계는 "없음(탐색적)"으로 명시)
RATIONALE = [["질환 유무와 VAT/SAT 연관, 성별 층화", "Framingham(Circulation 2007), Jackson Heart(JCEM 2010), Sci Rep 2026", "있음"],
             ["나이·성별·BMI 보정", "Framingham: BMI·허리둘레 보정 후에도 VAT 연관 유지", "있음"],
             ["같은 landmark 조직 비율(VAT/SAT, VAT/근육)", "Matsuzawa 1995(V/S), Acad Radiol 2026·Minerva 2025(VAT/SAT, 대사 위험), Surg Today 2026(L3 VAT/psoas, 암 예후)", "있음 (근육 비는 암 환자)"],
             ["다른 landmark·조직 교차 조합(AEC 포함)", "직접 근거 못 찾음", "없음 (탐색적)"],
             ["L3 등 단일 slice 사용", "L3 = 근육 기준 표준(Eur J Radiol 2021, Clin Nutr 2019), T4도 유사(JCM 2023)", "부분 (근육 중심)"],
             ["체격 정규화", "SMI/VFI를 키² 또는 BSA로 나눔(JCM 2025). 환자 곡선 평균/L3 값으로 나누는 방식은 아님", "부분"],
             ["Patient-wise 곡선 정규화 (mean, z, min-max, L3)", "직접 근거 못 찾음. 체격 정규화 개념과 L3 기준 개념만 간접 연결", "없음 (탐색적)"],
             ["부분군 분석 (사전 지정, 전체 보고, 코호트 간 재현)", "Wang NEJM 2007(보고 지침), Sun BMJ 2010(신뢰도 기준). 초록 미제공이라 제목과 일반 내용에 근거", "부분"],
             ["MLP vs LR 모델 비교", "J Clin Epidemiol 2019: 편향 위험이 낮은 비교에서 ML과 LR의 AUC 차이 0.00", "있음"],
             ["landmark 단일 단면 대표성(L3, L2~L3)", "L3가 전체 VAT·SAT·근육 부피를 가장 잘 대표(Am J Clin Nutr 2015, MRI), VAT는 L2~L3 단면이 예측력 최고(Eur J Clin Nutr 1996)", "부분 (건강인·부피 대표성)"],
             ["전진 선택, CV와 외부 검증 분리", "Steyerberg 2001: 구축 표본 성능은 과대평가되므로 CV/bootstrap 필요, TRIPOD: 보고 기준. 선택 대상 형태(교차 비율, 구간 평균, AEC 등)는 문헌 근거 없음", "방법 있음 / feature 없음"],
             ["스캐너 보정", "스캐너·촬영 조건이 CT 특징에 영향(Sci Rep 2024), ComBat 보정(Front Oncol 2026). radiomics 대상", "부분"],
             ["AEC를 질환 지표로 해석", "AEC를 질환 지표로 다룬 문헌 못 찾음 (TCM 문헌은 선량·체격 중심)", "없음 (탐색적)"],
             ["통계: BH-FDR, Cohen d, Mann-Whitney", "표준 통계 방법. PubMed 검색으로는 원 논문을 확인하지 못함 (고전 통계 문헌)", "확인 못 함"]]
REFS = ["Fox CS et al. Circulation 2007. doi:10.1161/CIRCULATIONAHA.106.675355", "Liu J et al. JCEM 2010. doi:10.1210/jc.2010-1378",
        "Matsuzawa Y et al. Obes Res 1995. doi:10.1002/j.1550-8528.1995.tb00481.x", "Chung GE et al. Sci Rep 2026. doi:10.1038/s41598-025-30244-6",
        "Fan J et al. Acad Radiol 2023. doi:10.1016/j.acra.2023.01.033", "Jin K et al. JCSM 2026. doi:10.1002/jcsm.70166",
        "van Heusden HC et al. Eur J Radiol 2021. doi:10.1016/j.ejrad.2021.109879", "Rollins KE et al. Clin Nutr 2019. doi:10.1016/j.clnu.2019.10.003",
        "Belger E et al. J Clin Med 2023. doi:10.3390/jcm12072520", "Özdemir M et al. J Clin Med 2025. doi:10.3390/jcm14227915",
        "Zhang Y et al. Acad Radiol 2026. doi:10.1016/j.acra.2026.09.035", "Ko S et al. Minerva Endocrinol 2025. doi:10.23736/S2724-6507.25.04366-0",
        "Sakai M et al. Surg Today 2026. doi:10.1007/s00595-026-03487-7", "Christodoulou E et al. J Clin Epidemiol 2019. doi:10.1016/j.jclinepi.2019.02.004", "Wang R et al. N Engl J Med 2007. doi:10.1056/NEJMsr077003",
        "Sun X et al. BMJ 2010. doi:10.1136/bmj.c117", "Paschen C et al. Kidney Int Rep 2025. doi:10.1016/j.ekir.2025.103739", "Schweitzer L et al. Am J Clin Nutr 2015. doi:10.3945/ajcn.115.111203", "Armellini F et al. Eur J Clin Nutr 1996. PMID 8735309", "Steyerberg EW et al. J Clin Epidemiol 2001. doi:10.1016/s0895-4356(01)00341-9", "Collins GS et al. BMJ 2015 (TRIPOD). doi:10.1136/bmj.g7594", "Levi R et al. Sci Rep 2024. doi:10.1038/s41598-024-68158-4", "Zhang X et al. Front Oncol 2026. doi:10.3389/fonc.2026.1896994"]


def pfmt(x):
    return "<0.001" if x < .001 else f"{x:.3f}"


# 측정값 x 질환 행: 코호트별 "OR (p)" 보정 / 장비 보정 (PC1 점수, 부호 = 치골 쪽 상대값이 클수록 +)
def pc1_rows():
    rows = []
    for m in ["AEC", "VAT", "SAT", "TAMA"]:
        for d in DIS:
            r = [m, d]
            for c in COH:
                g = pc1[(pc1.cohort == c) & (pc1.measure == m) & (pc1.disease == d)].iloc[0]
                r += [f"{g.OR_adj:.2f} ({pfmt(g.p_adj)})", f"{g.OR_adj_scanner:.2f} ({pfmt(g.p_adj_scanner)})"]
            rows.append(r)
    return rows


# 구간 평균(raw)과 환자 평균 대비 구간 평균(pw_mean)의 장비 보정 OR "Gangnam / Sinchon" (* = 두 코호트 FDR<0.05 & 같은 방향)
def seg_rows():
    rows = []
    for sg in ["liver_dome~L1", "L1~L3", "L3~L5", "L5~inferior_pubic_margin"]:
        for m, d in [("AEC", "HTN"), ("AEC", "DM"), ("VAT", "HTN")]:
            if m == "VAT" and sg.startswith("L5"):
                continue  # 치골 쪽 VAT는 거의 0이라 분석 제외
            r = [sg.replace("inferior_pubic_margin", "pubis"), f"{m} / {d}"]
            for v in ["raw", "pw_mean"]:
                g = seg[(seg.measure == m) & (seg.segment == sg) & (seg.variant == v) & (seg.disease == d)].iloc[0]
                r.append(f"{g.OR_scanner_gangnam:.2f} / {g.OR_scanner_sinchon:.2f}{'*' if g.both_scanner else ''}")
            rows.append(r)
    return rows


# Tier x 질환별 일관 feature 수와 best feature (Tier1 = 같은 landmark 조직 비율, Tier2 = landmark/조직 교차 조합, 탐색적)
def combo_rows():
    rows = []
    for tier, nm in [("tier1", "Tier 1 (근거 있음)"), ("tier2", "Tier 2 (탐색적)")]:
        for d in DIS:
            g = combo[(combo.tier == tier) & (combo.disease == d)]
            rb = g[g.robust].sort_values("min_abs_logOR", ascending=False)
            best = "-" if rb.empty else f"{rb.iloc[0].feature} ({rb.iloc[0].OR_gangnam:.2f} / {rb.iloc[0].OR_sinchon:.2f})"
            rows.append([nm, d, f"{len(g)}", f"{int(g.robust.sum())}", best])
    return rows


KEYSETS = ["clinical", "VAT@L3", "VAT,SAT,TAMA@L3", "Tier1 ratios@L3", "Tier1 ratios@all landmarks", "VAT,SAT,TAMA@all landmarks"]


# 세트 모델: Gangnam CV AUC, Sinchon 외부 AUC, clinical 대비 변화(DeLong p)
def set_rows():
    rows = []
    for d in DIS:
        for k in KEYSETS:
            g = cset[(cset.disease == d) & (cset.set == k)].iloc[0]
            dl = "-" if k == "clinical" else f"{g.ext_delta:+.3f} (p={g.delong_p_vs_clinical:.3f})"
            rows.append([d, k, str(int(g.n_feat)), f"{g.cv_auc_gangnam:.3f}", f"{g.ext_auc_sinchon:.3f}", dl])
    return rows


# MLP vs LR: 핵심 세트의 Gangnam CV AUC, Sinchon 외부 AUC, MLP-LR 차이(DeLong p)
def mlp_rows():
    keep = ["clinical", "VAT@L3", "Tier1 ratios@L3", "VAT,SAT,TAMA@all landmarks"]
    rows = []
    for d in DIS:
        for k in keep:
            g = mlp[(mlp.disease == d) & (mlp.set == k)].iloc[0]
            rows.append([d, k, f"{g.cv_LR:.3f} / {g.cv_MLP:.3f}", f"{g.ext_LR:.3f} / {g.ext_MLP:.3f}", f"{g.ext_MLP_minus_LR:+.3f} (p={g.delong_p:.3f})"])
    return rows


# 부분군: 전체(all) 그룹의 ΔAUC (방향별 95% CI)
def sg_all_rows():
    rows = []
    for d in DIS:
        for k in ["VAT@L3", "Tier1 ratios@L3", "VAT,SAT,TAMA@L3"]:
            r = [d, k]
            for tr in ["gangnam", "sinchon"]:
                g = sga[(sga.train == tr) & (sga.disease == d) & (sga.set == k) & (sga.group == "all")].iloc[0]
                r.append(f"{g.delta:+.3f} [{g.ci_low:+.3f}, {g.ci_high:+.3f}]")
            rows.append(r)
    return rows


# 부분군: 한 방향에서 |ΔAUC|가 큰 매칭 그룹과 반대 방향 값 (재현 여부 확인용)
def sg_top_rows(n=8):
    j = sgm[sgm.group != "all"].copy()
    j["mx"] = j[["delta_G2S", "delta_S2G"]].abs().max(axis=1)
    return [[r.disease, r.set, r.group, f"{r.delta_G2S:+.3f} [{r.ci_low_G2S:+.2f}, {r.ci_high_G2S:+.2f}]", f"{r.delta_S2G:+.3f} [{r.ci_low_S2G:+.2f}, {r.ci_high_S2G:+.2f}]"]
            for r in j.sort_values("mx", ascending=False).head(n).itertuples()]


# landmark 조합 탐색: 학습 코호트 CV로 고른 조합, 다른 코호트 외부 AUC (clinical / 선택 조합 / L3 단독 / 10개 전체)
def subset_rows():
    return [[r.disease, "G→S" if r.train == "gangnam" else "S→G", {"V": "값", "R": "비율"}[r.ftype], r.best_subset.replace("inferior_pubic_margin", "pubis"),
             f"{r.clinical_ext:.3f}", f"{r.best_ext:.3f} (p={r.delong_p:.2f})", f"{r.L3_ext:.3f}", f"{r.all10_ext:.3f}"]
            for r in sub.sort_values(["disease", "train", "ftype"], key=lambda c: c.map({"HTN": 0, "DM": 1, "CKD": 2, "gangnam": 0, "sinchon": 1, "V": 0, "R": 1}) if c.name != "x" else c).itertuples()]


# 전체 feature 풀 전진 선택: 방향·질환별 선택된 feature 수, 학습 CV AUC, 외부 AUC (clinical / 선택 모델), 선택된 feature 형태
def fsel_rows():
    rows = []
    for r in fsel.sort_values(["disease", "train"], key=lambda c: c.map({"HTN": 0, "DM": 1, "CKD": 2, "gangnam": 0, "sinchon": 1}).fillna(c) if c.name in ("disease", "train") else c).itertuples():
        fams = ", ".join(fpath[(fpath.train == r.train) & (fpath.disease == r.disease)].sort_values("step").family)
        rows.append([r.disease, "G→S" if r.train == "gangnam" else "S→G", str(r.step), f"{r.cv_auc:.3f}", f"{r.clinical_ext:.3f}", f"{r.ext_auc:.3f} (p={r.delong_p:.2f})", fams])
    return rows


LIT = [["Framingham (n=3,001)", "VAT가 SAT보다 HTN·DM·MetS와 강하게 연관, 여성에서 더 강함", "VAT > SAT, 여성에서 효과 큼: 일치"],
       ["Jackson Heart (n=2,477)", "VAT 1SD당 OR: 여성 HTN 1.62, DM 1.82. BMI 보정 후에도 유의", "보정 OR HTN 1.3~1.8: 크기 유사"],
       ["Sci Rep 2026, 한국 (n=14,105)", "VAT 증가 시 CKD 위험 증가 (남 1.25, 여 1.18), 여성 SAT는 역상관", "CKD VAT 방향 일치, 보정 후 비유의"],
       ["Acad Radiol 2023 (T2DM)", "여성에서 VAT index가 DKD 위험인자", "여성 CKD VAT 효과가 더 큼: 방향 일치"],
       ["Sci Rep 2026 / Framingham", "SAT는 VAT보다 약하게 연관, 여성 SAT는 CKD와 역상관 가능", "SAT 단독은 약함, VAT 대비 비율(SAT/(SAT+VAT))은 OR 0.6"],
       ["JCSM 2026 메타분석 (CKD)", "CKD에서 당뇨는 근감소 위험(OR 1.96), 높은 BMI는 보호", "TAMA는 단면 값 대신 상대 비율에서만 일관(DM·CKD)"],
       ["AEC (문헌 없음)", "AEC를 질환 지표로 다룬 연구는 찾지 못함", "체격·장비 교란 가능, 해석 주의"]]

from pptx_notes_disease_difference import NOTES  # 비전공자용 슬라이드 노트 (슬라이드 순서와 개수가 같아야 함)


def main() -> None:
    k = base.Deck()
    k.cover(["Landmark 단면 체성분의", "질환 유무 차이 분석"], "2026.10.06")
    k.page("연구 목적", "Introduction", ["VAT·SAT·TAMA·AEC의 landmark 단면 값 중 HTN/DM/CKD 유무를 뚜렷이 구분하는 위치·조직을 찾는다",
                                      "선행연구(Framingham, Jackson Heart 등)와 방향·크기를 비교한다"], text_h=1700000)
    k.page("데이터와 분석 방법", "Methods", ["Gangnam 1,443명 / Sinchon 1,145명 (QC 후 1,436 / 1,129명), landmark 12곳 × 조직 4종",
                                         "1) 미보정 Cohen d, Mann-Whitney, BH-FDR   2) 나이·성별·BMI 보정 로지스틱 OR(+1SD)",
                                         "두 코호트 모두 FDR<0.05 & 같은 방향이면 일관된 결과"], text_h=1700000, size=16)
    s = k.page("분석 단계별 문헌 근거", "Methods", ["PubMed에서 확인한 문헌 기준. 근거가 없는 단계는 탐색적 분석으로 표시"], text_h=600000, size=14)
    k.table(s, ["분석 단계", "근거 (PubMed 확인)", "근거 수준"], RATIONALE, 1700000, 5100000, [3, 6.5, 1.8], bs=9)
    s = k.page("코호트 특성", "Table 1", [], text_h=200000)
    k.table(s, ["Variable", "Gangnam", "Sinchon"], cohort_table(), 1100000, 4700000, [3, 3, 3])
    s = k.page("질환 유무 차이: 미보정 Cohen d", "Table 2", ["VAT가 전 질환에서 가장 크다. * = FDR<0.05"], text_h=500000, size=16)
    pairs = [("VAT", "L1"), ("VAT", "L3"), ("SAT", "L3"), ("TAMA", "L3"), ("AEC", "L3")]
    rows = [[d, COH[c]] + [cell(diff, "cohen_d", "All", d, c, m, a) for m, a in pairs] for d in DIS for c in COH]
    k.table(s, ["Disease", "Cohort", "VAT@L1", "VAT@L3", "SAT@L3", "TAMA@L3", "AEC@L3"], rows, 1700000, 4300000, [1.4, 1.6, 1.4, 1.4, 1.4, 1.4, 1.4], merge_col0=True)
    s = k.page("질환 유무 차이: 전체 landmark 개요", "Figure 1", [], text_h=200000)
    s.shapes.add_picture(str(D / "ppt_unadjusted_d_heatmap.png"), Emu(508000), Emu(900000), width=Emu(11176000))
    s = k.page("나이·성별·BMI 보정 후 OR (+1SD)", "Table 3", ["보정 후 일관된 결과는 HTN의 VAT뿐이다. * = FDR<0.05"], text_h=500000, size=16)
    pairs = [("VAT", "L1"), ("VAT", "L2"), ("VAT", "L3"), ("SAT", "L3"), ("TAMA", "L3"), ("AEC", "L3")]
    rows = [[d, COH[c]] + [cell(adj, "OR_per_SD", "All", d, c, m, a) for m, a in pairs] for d in DIS for c in COH]
    k.table(s, ["Disease", "Cohort", "VAT@L1", "VAT@L2", "VAT@L3", "SAT@L3", "TAMA@L3", "AEC@L3"], rows, 1700000, 4300000, [1.3, 1.5] + [1.3] * 6, merge_col0=True)
    s = k.page("보정 OR 개요", "Figure 2", [], text_h=200000)
    s.shapes.add_picture(str(D / "ppt_adjusted_or_heatmap.png"), Emu(508000), Emu(900000), width=Emu(11176000))
    s = k.page("성별 층화 보정 OR: VAT@L2", "Table 4", ["남성은 코호트 간 차이가 크다. * = FDR<0.05"], text_h=500000, size=16)
    rows = [[d, sx, cell(adj, "OR_per_SD", sx, d, "gangnam", "VAT", "L2"), cell(adj, "OR_per_SD", sx, d, "sinchon", "VAT", "L2")] for d in DIS for sx in ("M", "F")]
    k.table(s, ["Disease", "Sex", "Gangnam", "Sinchon"], rows, 1700000, 4000000, [2, 2, 2.5, 2.5], merge_col0=True)
    s = k.page("SAT·TAMA·AEC 여러 case 분석: 일관된 feature 수", "Table 5", ["두 코호트 모두 FDR<0.05 & 같은 방향인 feature 수 / 전체 (보정 OR). 조직 비율 = TAMA/SAT, TAMA/(SAT+VAT), SAT/(SAT+VAT)"], text_h=800000, size=14)
    k.table(s, ["Disease", "Tissue", "Landmark 단면", "Landmark 간 비율", "구간 평균", "구간/전체 비율", "조직 비율"], case_count_rows(), 1900000, 4300000, [1.2, 1.2, 1.6, 1.8, 1.5, 1.8, 1.5], merge_col0=True)
    s = k.page("SAT·TAMA·AEC: 질환별 best 일관 feature", "Table 6", ["SAT/(SAT+VAT)는 VAT 비중의 역수라 VAT 정보가 들어 있다. AEC 비율은 체격·장비 영향 가능"], text_h=800000, size=14)
    k.table(s, ["Disease", "Tissue", "Case", "Feature", "OR Gangnam", "OR Sinchon"], best_rows(), 1900000, 4300000, [1, 1, 1.5, 4.2, 1.3, 1.3], merge_col0=True, bs=11)
    s = k.page("정규화 위치별 OR 곡선", "Figure 3", [], text_h=200000)
    s.shapes.add_picture(str(D / "landmark_pos_or_curves.png"), Emu(508000), Emu(900000), height=Emu(5600000))
    s = k.page("Patient-wise 전처리 비교: 일관된 위치 수", "Table 7", ["liver_dome~pubis 128 위치 중 두 코호트 FDR<0.05 & 같은 방향인 위치 수. 환자 자신의 값으로 정규화 (pw_mean=환자 평균으로 나눔, pw_z=환자 내 z-score, pw_minmax=환자 내 0~1, pw_L3=L3 값으로 나눔)"], text_h=1100000, size=13)
    k.table(s, ["Measure", "Disease", "raw", "pw_mean", "pw_z", "pw_minmax", "pw_L3"], pw_rows("n_robust_pos"), 2300000, 4000000, [1.3, 1.3, 1, 1.2, 1, 1.3, 1.1], merge_col0=True, bs=11)
    s = k.page("Patient-wise 전처리 비교: 외부 AUC 변화", "Table 8", ["Gangnam에서 p가 가장 작은 위치 1개를 골라 clinical(나이·성별·BMI)에 더한 모델의 Sinchon AUC 변화 (선택은 Gangnam만 사용)"], text_h=800000, size=14)
    k.table(s, ["Measure", "Disease", "raw", "pw_mean", "pw_z", "pw_minmax", "pw_L3"], pw_rows("ext_auc_gain"), 1900000, 4300000, [1.3, 1.3, 1, 1.2, 1, 1.3, 1.1], merge_col0=True, bs=11)
    s = k.page("Patient-wise 전처리별 위치 OR 곡선: AEC", "Figure 4", [], text_h=200000)
    s.shapes.add_picture(str(D / "landmark_patientwise_or_curves_AEC.png"), Emu(2200000), Emu(900000), height=Emu(5800000))
    for i, c in enumerate(COH):  # patient-wise(환자 평균 대비) AEC 곡선: 질환 x 성별
        s = k.page(f"Patient-wise AEC 곡선: 질환 x 성별 ({COH[c]})", f"Figure {5 + i}", [], text_h=200000)
        s.shapes.add_picture(str(D / "figures" / f"{c}_AEC_pwmean_curve_by_disease_sex.png"), Emu(508000), Emu(1300000), width=Emu(11176000))
    s = k.page("스캐너 분포와 질환 유병률", "Table 10", ["장비에 따라 질환 유병률이 달라 AEC 신호의 교란 가능성이 있다 (예: Gangnam Revolution CT는 여성 78%, HTN 20%)"], text_h=800000, size=14)
    k.table(s, ["Cohort", "Scanner", "N", "여성", "HTN", "DM", "CKD"], scanner_dist_rows(), 1900000, 4300000, [1.3, 3, 0.9, 1, 1, 1, 1], merge_col0=True, bs=11)
    s = k.page("스캐너 보정 후에도 patient-wise 신호가 남는가", "Table 11", ["일관된 위치 수: 보정 전 → 장비 더미 보정 후 / 최다 장비 1종만 (Gangnam Sensation 64, Sinchon iCT 256)"], text_h=800000, size=14)
    k.table(s, ["Measure", "Disease", "pw_mean", "pw_z", "pw_L3"], scanner_rows(), 1900000, 4300000, [1.3, 1.3, 2, 2, 2], merge_col0=True, bs=11)
    s = k.page("Patient-wise 곡선 전체(PC1) 비교: 보정 후", "Table 12", ["곡선 전체를 PC1 점수 1개로 요약해 질환과 비교 (+1SD당 OR, 괄호 = p). 나이·성별·BMI 보정 / +스캐너 보정. AEC는 두 코호트 모두 HTN·DM에서 유지, CKD는 Sinchon에서 유지"], text_h=900000, size=13)
    k.table(s, ["Measure", "Disease", "Gangnam 보정", "Gangnam +장비", "Sinchon 보정", "Sinchon +장비"], pc1_rows(), 2100000, 4300000, [1.2, 1.2, 2, 2, 2, 2], merge_col0=True, bs=11)
    s = k.page("곡선 요약을 구간 평균으로: raw vs patient-wise", "Table 13", ["구간 평균(raw)에서는 AEC 신호가 없고, 환자 평균 대비 구간 평균(pw_mean)에서만 두 코호트 모두 유의 (장비 보정 OR). 상복부는 상대 전류가 높고 하복부는 낮다. SAT·TAMA는 어떤 구간·전처리에서도 없음"], text_h=900000, size=13)
    k.table(s, ["구간", "측정값 / 질환", "raw 구간 평균", "pw_mean (구간/환자 전체 평균)"], seg_rows(), 2100000, 4300000, [2, 2, 2.5, 3], merge_col0=True, bs=11)
    s = k.page("landmark 지표 조합 feature", "Table 14", ["Tier 1 = 같은 landmark 안의 조직 비율 (VAT/SAT, VAT/TAMA, TAMA/(VAT+SAT)), Tier 2 = 다른 landmark·조직(AEC 포함) 교차 로그 비. 일관 = 두 코호트 FDR<0.05 & 같은 방향 (OR Gangnam / Sinchon)"], text_h=1000000, size=13)
    k.table(s, ["Tier", "Disease", "전체 feature", "일관", "best feature (OR)"], combo_rows(), 2200000, 4100000, [2, 1.2, 1.4, 1, 5], merge_col0=True, bs=11)
    s = k.page("조합 세트 모델: 외부 AUC", "Table 15", ["clinical(나이·성별·BMI)에 landmark 지표 세트를 더한 로지스틱. Gangnam 5-fold CV AUC, Gangnam 학습 → Sinchon 외부 AUC, DeLong p는 clinical 대비"], text_h=800000, size=13)
    k.table(s, ["Disease", "Set", "# Feat", "CV AUC (G)", "Ext AUC (S)", "Δ vs clinical"], set_rows(), 1900000, 4500000, [1.2, 4, 0.9, 1.4, 1.4, 2.4], merge_col0=True, bs=10)
    s = k.page("모델 비교: MLP vs 로지스틱 회귀", "Table 17", ["clinical에 landmark 지표 세트를 더한 모델. 값은 LR / MLP. MLP는 inner CV로 하이퍼파라미터 선택. 전체 21개 세트 평균 외부 AUC 차이(MLP−LR) -0.001, 개별 차이 모두 |0.012| 이하"], text_h=900000, size=13)
    k.table(s, ["Disease", "Set", "Gangnam CV AUC (LR / MLP)", "Sinchon 외부 AUC (LR / MLP)", "MLP − LR (DeLong)"], mlp_rows(), 2100000, 4300000, [1.2, 3.6, 2.4, 2.4, 2.6], merge_col0=True, bs=11)
    s = k.page("부분군 분석: 전체 그룹의 ΔAUC", "Table 18", ["clinical 대비 landmark 지표 세트의 ΔAUC (한 코호트로 학습, 다른 코호트에서 평가, 95% 부트스트랩 CI)"], text_h=700000, size=14)
    k.table(s, ["Disease", "Set", "Gangnam→Sinchon", "Sinchon→Gangnam"], sg_all_rows(), 1800000, 4500000, [1.2, 3, 3, 3], merge_col0=True, bs=11)
    s = k.page("부분군 분석: ΔAUC가 큰 그룹은 재현되지 않는다", "Table 19", ["그룹 = 성별·나이/BMI 3분위·성별×분위 (사전 지정). 매칭 그룹 162개 중 두 방향 모두 CI가 0을 제외하고 부호가 같은 그룹은 0개. 두 방향 ΔAUC 부호 일치 59%, 상관 0.12"], text_h=1000000, size=13)
    k.table(s, ["Disease", "Set", "Group", "Δ Gangnam→Sinchon [95% CI]", "Δ Sinchon→Gangnam [95% CI]"], sg_top_rows(), 2300000, 4000000, [1.2, 2.6, 2.4, 3, 3], bs=11)
    s = k.page("최적 landmark 조합 탐색 (liver dome, 치골 제외)", "Table 21", ["10개 landmark의 1~5개 조합(638개)을 학습 코호트 CV AUC로 선택하고 다른 코호트에서 평가. 값 = VAT·SAT·TAMA 값, 비율 = VAT/SAT·VAT/TAMA·TAMA/(VAT+SAT). 외부 AUC (선택 조합은 DeLong p vs clinical)"], text_h=1000000, size=12)
    k.table(s, ["Disease", "방향", "형태", "선택된 조합", "clinical", "선택 조합", "L3 단독", "10개 전체"], subset_rows(), 2100000, 4300000, [1, 0.8, 0.8, 3.6, 1.1, 2.0, 1.2, 1.3], merge_col0=True, bs=10)
    s = k.page("전체 feature 풀 전진 선택", "Table 23", ["풀 1,226개 = 단면 값, 같은 landmark 비율, landmark 간 같은 조직 비율, 교차 비율, 구간 평균, 구간/전체, AEC, patient-wise(pw_mean, pw_z). 학습 코호트 CV로 최대 6개 선택, 다른 코호트 외부 AUC. L(landmark 간 같은 조직 비율) 단독 조합 탐색도 clinical 대비 유의 개선 1/6"], text_h=1100000, size=12)
    k.table(s, ["Disease", "방향", "# 선택", "학습 CV AUC", "clinical 외부", "선택 모델 외부 (DeLong p)", "선택된 feature 형태 (순서)"], fsel_rows(), 2300000, 4100000, [1, 0.8, 0.9, 1.3, 1.3, 2.3, 4.2], merge_col0=True, bs=10)
    s = k.page("선행연구와의 비교 (PubMed)", "Table 24", [], text_h=200000)
    k.table(s, ["Study", "Key finding", "본 연구와의 관계"], LIT, 1100000, 4800000, [2.6, 5, 3.4])
    k.page("Conclusion", "Conclusion", ["VAT가 질환 유무와 가장 뚜렷이 연관되고, 위치는 T11~L2 부근이 크다 (SAT·TAMA·AEC는 작음)",
                                       "나이·성별·BMI 보정 후에도 두 코호트에서 일관된 것은 HTN의 VAT뿐이다",
                                       "DM은 Gangnam에서만, CKD는 둘 다 비유의: BMI 교란 또는 표본 수 한계",
                                       "환자별 정규화(patient-wise)를 하면 AEC(pw_mean: HTN 111, DM 94, CKD 47 위치)와 SAT(pw_z: CKD)에서 일관된 신호가 나오지만 외부 AUC 변화는 ≤0.023",
                                       "AEC patient-wise 곡선(PC1)은 두 코호트 모두 HTN·DM에서 나이·성별·BMI·장비 보정 후에도 유의 (OR 0.51~0.74), CKD는 Sinchon에서만 유의. 위치별 FDR 기준에서는 CKD 신호가 장비 보정 후 사라졌다",
                                       "SAT·TAMA·AEC는 단면 값 단독으로는 약하고, 비율/구간 case에서 일관된 신호가 나온다 (SAT/(SAT+VAT)는 VAT 비중이라 독립 정보는 아님)",
                                       "같은 landmark의 VAT/SAT 비가 HTN(25개 landmark 조합)·DM에서 두 코호트 일관되게 유의하나, 세트 모델의 외부 AUC 향상은 ≤ +0.026 수준으로 제한적이다",
                                       "MLP도 LR과 같은 수준(평균 외부 AUC 차이 -0.001)이며, 성별·나이·BMI·스캐너 부분군에서 ΔAUC가 큰 그룹은 두 코호트에서 재현되지 않았다 (0/162)",
                                       "liver dome·치골을 뺀 10개 landmark 조합 탐색에서 안정적인 최적 조합은 없고(L3 단독 대비 외부 AUC +0.002, clinical 대비 유의 개선 0/12), L3 단독이 충분하다는 문헌과 일치한다",
                                       "모든 feature 형태를 넣은 전진 선택(1,226개 풀)도 학습 CV만 오르고 외부 AUC 개선은 없다(평균 -0.006, 유의 개선 0/6). 안정적인 최적 조합은 확인되지 않았다",
                                       "patient-wise 정규화와 AEC 해석은 직접 문헌 근거가 없는 탐색적 분석이므로 해석에 주의가 필요하다",
                                       "문헌과 방향은 일치하나 문헌은 부피/VFA 기준이라 단면 비교에는 한계가 있다"], text_h=5000000, size=14)
    s = k.page("참고문헌", "References", [], text_h=200000)
    tb = s.shapes.add_textbox(Emu(508000), Emu(1100000), Emu(11176000), Emu(5400000)).text_frame
    tb.word_wrap = True
    for i, r_ in enumerate(REFS):
        para = tb.paragraphs[0] if i == 0 else tb.add_paragraph()
        run = para.add_run()
        run.text, run.font.size = f"{i + 1}. {r_}", Pt(13)
    k.end_cover()
    for c in COH:
        s = k.page(f"Appendix. 질환별 landmark VAT 분포 ({COH[c]})", "Appendix", [], text_h=200000)
        s.shapes.add_picture(str(D / f"{c}_VAT_by_landmark.png"), Emu(508000), Emu(1100000), width=Emu(11176000))
    for m in ["VAT", "SAT", "TAMA"]:
        s = k.page(f"Appendix. Patient-wise 전처리별 위치 OR 곡선: {m}", "Appendix", [], text_h=200000)
        s.shapes.add_picture(str(D / f"landmark_patientwise_or_curves_{m}.png"), Emu(2200000), Emu(900000), height=Emu(5800000))
    for c in COH:
        for m in ["VAT", "SAT", "TAMA"]:
            s = k.page(f"Appendix. Patient-wise {m} 곡선: 질환 x 성별 ({COH[c]})", "Appendix", [], text_h=200000)
            s.shapes.add_picture(str(D / "figures" / f"{c}_{m}_pwmean_curve_by_disease_sex.png"), Emu(508000), Emu(1300000), width=Emu(11176000))
    for c in COH:  # outputs/clinic4/figures/aec_curve_by_disease_sex.png 형식의 질환 x 성별 128구간 곡선
        for m in ["VAT", "SAT", "TAMA", "AEC", "TAMA_over_SAT"]:
            s = k.page(f"Appendix. {m.replace('_over_', '/')} 곡선: 질환 x 성별 ({COH[c]})", "Appendix", [], text_h=200000)
            s.shapes.add_picture(str(D / "figures" / f"{c}_{m}_curve_by_disease_sex.png"), Emu(508000), Emu(1300000), width=Emu(11176000))
    s = k.page("Appendix. 성별 층화 보정 OR 상세 (VAT)", "Appendix", [], text_h=200000)
    rows = [[d, sx] + [f"{cell(adj, 'OR_per_SD', sx, d, 'gangnam', 'VAT', a)} / {cell(adj, 'OR_per_SD', sx, d, 'sinchon', 'VAT', a)}" for a in ("T12", "L1", "L2", "L3")]
            for d in DIS for sx in ("M", "F")]
    k.table(s, ["Disease", "Sex", "T12 (G/S)", "L1 (G/S)", "L2 (G/S)", "L3 (G/S)"], rows, 1100000, 4000000, [1.5, 1, 2.2, 2.2, 2.2, 2.2], merge_col0=True)
    for sl in list(k.prs.slides)[k.n_orig:]:  # 본문 텍스트가 비어 있는 슬라이드(표/그림만)는 빈 본문 placeholder 삭제
        for sh in list(sl.placeholders):
            if sh.placeholder_format.idx == 1 and sh.has_text_frame and not sh.text_frame.text.strip():
                sh._element.getparent().remove(sh._element)
    box_l, box_t, box_w, box_h = 508000, 1050000, 11176000, 5350000  # 제목 밑줄(top 930000) 아래, 슬라이드 번호 위
    for sl in list(k.prs.slides)[k.n_orig:]:  # 그림은 밑줄 아래 영역 안에 비율 유지로 맞추고 가운데 정렬 (전체 화면 종료 표지는 제외)
        for sh in sl.shapes:
            if sh.shape_type == 13 and sh.width < k.prs.slide_width:
                r = min(box_w / sh.width, box_h / sh.height)
                sh.width, sh.height = int(sh.width * r), int(sh.height * r)
                sh.left, sh.top = box_l + (box_w - sh.width) // 2, box_t
    k.save(NOTES)


main()
