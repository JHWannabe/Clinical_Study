from __future__ import annotations

# 연구세미나 발표자료용 추가 그림 4종을 outputs/clinic4/seminar_figs/에 생성한다.
#  - study_design.png : 연구 설계 개요(코호트 -> 5-fold CV -> 외부검증)
#  - fpca_elbow.png   : 곡선별 설명분산(scree)과 선택한 PC 수(Kneedle elbow)
#  - fpca_recovered_bodycomp.png : Best 모델 곡선별 FPCA 계수를 slice 축으로 환산
#  - forest_or.png    : Baseline vs Best 임상변수 Odds Ratio(95% CI)

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import clinic4_5fold_cv as m
from clinic4_aec_bodycomp_logistic import curve_cols, load_data_with_curves
from clinic4_logistic_regression import DATA_XLSX, DISEASES, PROJECT_ROOT, SEED

sys.stdout.reconfigure(encoding="utf-8")
OUT = PROJECT_ROOT / "outputs" / "clinic4" / "seminar_figs"
OUT.mkdir(parents=True, exist_ok=True)
NAME = {"HTN": "고혈압", "DM": "당뇨병", "CKD": "만성신장질환"}
C_BASE, C_BEST = "#7f8c9a", "#c0392b"
N_PC = 20  # 원 분석의 FPCA_COMPONENT_CANDIDATES_MAX와 같은 값
ELBOW_K = {"AEC": 3, "VAT": 2, "SAT": 2, "LAMA": 2, "NAMA": 2, "IMATA": 3}  # 곡선별 선택값(Kneedle elbow)


def box(ax, x, y, w, h, text, fc, size=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="#34495e", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#34495e"))


def study_design() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.6)); ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.2, 2.9, 2.6, 1.3, "Internal\n강남 1,260명", "#dfe9f5", 12, True)
    box(ax, 0.2, 0.4, 2.6, 1.3, "External\n신촌 1,123명", "#fbe5d6", 12, True)
    box(ax, 3.5, 2.9, 3.0, 1.3, "5-fold CV (환자 단위)\n튜닝 + OOF 예측", "#eef3e2", 11)
    box(ax, 7.1, 2.9, 2.2, 1.3, "Youden\ncutoff", "#eef3e2", 11)
    box(ax, 3.5, 0.4, 3.0, 1.3, "강남 전체로 재학습\n(고정 모델)", "#eef3e2", 11)
    box(ax, 7.1, 0.4, 2.2, 1.3, "cutoff 동일 적용", "#eef3e2", 11)
    box(ax, 9.9, 1.65, 2.1, 1.3, "AUC·Sens·Spec\nDeLong 검정", "#f5e1e1", 11, True)
    arrow(ax, 2.8, 3.55, 3.5, 3.55); arrow(ax, 6.5, 3.55, 7.1, 3.55); arrow(ax, 2.8, 1.05, 3.5, 1.05); arrow(ax, 6.5, 1.05, 7.1, 1.05)
    arrow(ax, 9.3, 3.4, 9.9, 2.6); arrow(ax, 9.3, 1.2, 9.9, 2.0); arrow(ax, 8.2, 2.9, 8.2, 1.7)
    ax.text(5.0, 2.3, "Input: Baseline(성별·나이·키·몸무게)  vs  +AEC·체성분 FPCA 점수", ha="center", fontsize=11, color="#555")
    fig.tight_layout(); fig.savefig(OUT / "study_design.png", dpi=170); plt.close(fig)


# 원 분석(git 7bc04c3 clinic all compare aec_total.py select_fpca_n_by_elbow)과 같은 방식: 성분별 설명분산(scree)을 [0,1]로
# 정규화하고, 첫 점-마지막 점 직선까지의 수직거리가 최대인 PC를 elbow로 본다(후보 최대 20개)
def kneedle_elbow(ev: np.ndarray) -> int:
    x = np.arange(1, len(ev) + 1, dtype=float); xn = (x - x.min()) / (x.max() - x.min()); yn = (ev - ev.min()) / (ev.max() - ev.min())
    pts = np.column_stack([xn, yn]); p1, p2 = pts[0], pts[-1]; v = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.linalg.norm((pts - p1) - np.outer((pts - p1) @ v, v), axis=1)
    return int(np.argmax(dist)) + 1


def fpca_elbow() -> None:
    df = load_data_with_curves(DATA_XLSX)
    curves = {"AEC": "aec", "VAT": "VFA", "SAT": "SFA", "LAMA": "LAMA", "NAMA": "NAMA", "IMATA": "IMATA"}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 5.4), sharey=True)
    for ax, (nm, pre) in zip(axes.ravel(), curves.items()):
        ev = PCA(N_PC, random_state=SEED).fit(df[curve_cols(pre)].to_numpy(float)).explained_variance_ratio_ * 100
        k = np.arange(1, N_PC + 1); sel = kneedle_elbow(ev)
        print(f"{nm}: Kneedle elbow={sel} (모델 사용 k={ELBOW_K[nm]}){'' if sel == ELBOW_K[nm] else '  <- 불일치'}")
        ax.plot(k, ev, "o-", color="#34495e", lw=2, ms=4)
        ax.plot([1, N_PC], [ev[0], ev[-1]], ls="--", color="#2e86c1", lw=1.6, label=f"PC1-PC{N_PC} 직선")
        ax.plot(sel, ev[sel - 1], "o", ms=14, mfc="none", mec=C_BEST, mew=2.5)
        ax.axvline(sel, color=C_BEST, ls="--", lw=1)
        ax.set_title(f"{nm}: elbow k={sel} (누적 {ev[:sel].sum():.1f}%)", fontsize=11)
        ax.set_xticks([1, 2, 3, 5, 10, 15, 20]); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        if nm == "AEC": ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes[1]: ax.set_xlabel("PC 번호")
    for ax in axes[:, 0]: ax.set_ylabel("설명분산 (%)")
    fig.suptitle(f"FPCA 성분 수 선택: scree 곡선(PC1~PC{N_PC})의 elbow(Kneedle)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fpca_elbow.png", dpi=170); plt.close(fig)


def forest() -> None:
    terms = ["Sex (M)", "Age", "Height", "Weight"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), sharex=True)
    fits = {}
    for name, f in (("baseline", m.build_baseline), ("best", m.build_best)):
        x, df, _, _ = f()
        for d in DISEASES:
            r = sm.Logit(df[d].to_numpy(int), sm.add_constant(x)).fit(disp=0)
            fits[(d, name)] = (np.exp(r.params[1:5]), np.exp(r.conf_int()[1:5]))
    for ax, d in zip(axes, DISEASES):
        for off, name, col, lab in ((-0.17, "baseline", C_BASE, "Baseline"), (0.17, "best", C_BEST, "+AEC+체성분")):
            orr, ci = fits[(d, name)]; yy = np.arange(4) + off
            ax.errorbar(orr, yy, xerr=[orr - ci[:, 0], ci[:, 1] - orr], fmt="o", color=col, capsize=3, label=lab)
        ax.axvline(1, color="k", lw=1, ls="--"); ax.set_xscale("log"); ax.set_yticks(range(4)); ax.set_yticklabels(terms)
        ax.invert_yaxis(); ax.set_title(NAME[d]); ax.set_xlabel("Odds Ratio (95% CI, log)"); ax.grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "forest_or.png", dpi=170); plt.close(fig)


# Best 모델(14개 점수)의 체성분 곡선별 FPCA 계수를 128-slice 축으로 환산(Figure 2와 같은 방식: fold별 PCA·scaler를 학습 fold로만 fit,
# 규제 없는 Logit, 곡선 slice 표준편차를 곱해 "+1SD당 logit 변화"로 정규화, 5-fold 평균 ± SD)
def fpca_recovered_bodycomp() -> None:
    from sklearn.preprocessing import StandardScaler
    from clinic4_aec_bodycomp_logistic import BODYCOMP_CURVES, CLINICAL_BASE_COLS, AEC_FPCA_N, AEC_PREFIX
    curves = [("AEC", AEC_PREFIX, AEC_FPCA_N)] + [(n.upper(), pre, k) for n, (_, pre, k) in BODYCOMP_CURVES.items()]
    df = load_data_with_curves(DATA_XLSX); raw = {n: df[curve_cols(pre)].to_numpy(float) for n, pre, _ in curves}
    sex = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float); base = df[CLINICAL_BASE_COLS].to_numpy(float)
    rec = {d: {n: [] for n, _, _ in curves} for d in DISEASES}
    for d in DISEASES:
        y = df[d].to_numpy(int)
        for tr, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(base, y):
            pcas = {n: PCA(k, random_state=SEED).fit(raw[n][tr]) for n, _, k in curves}
            A = np.hstack([base[tr]] + [pcas[n].transform(raw[n][tr]) for n, _, _ in curves]); sc = StandardScaler().fit(A)
            beta = sm.Logit(y[tr], sm.add_constant(np.column_stack([sex[tr], sc.transform(A)]))).fit(disp=0).params
            off = 3
            for n, _, k in curves:
                w = np.zeros(128)
                for j in range(k): w += beta[2 + off + j] / sc.scale_[off + j] * pcas[n].components_[j]
                rec[d][n].append(w * raw[n][tr].std(0, ddof=1)); off += k
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 5.6), sharex=True)
    cols = {"HTN": "#1f77b4", "DM": "#e07b22", "CKD": "#2a9d3b"}
    for ax, (n, _, _) in zip(axes.ravel()[1:], curves[1:]):
        for d in DISEASES:
            r = np.array(rec[d][n]); mu, sd = r.mean(0), r.std(0, ddof=1); sl = np.arange(1, 129)
            ax.fill_between(sl, mu - sd, mu + sd, color=cols[d], alpha=0.12, lw=0); ax.plot(sl, mu, color=cols[d], lw=2, label=d)
        ax.axhline(0, color="gray", ls="--", lw=1); ax.set_title(n); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax0 = axes.ravel()[0]
    for d in DISEASES:
        r = np.array(rec[d]["AEC"]); mu, sd = r.mean(0), r.std(0, ddof=1); sl = np.arange(1, 129)
        ax0.fill_between(sl, mu - sd, mu + sd, color=cols[d], alpha=0.12, lw=0); ax0.plot(sl, mu, color=cols[d], lw=2, label=NAME[d])
    ax0.axhline(0, color="gray", ls="--", lw=1); ax0.set_title("AEC"); ax0.grid(alpha=0.25); ax0.spines[["top", "right"]].set_visible(False); ax0.legend(frameon=False, fontsize=9)
    for ax in axes[1]: ax.set_xlabel("slice (1=간 → 128=치골)")
    for ax in axes[:, 0]: ax.set_ylabel("+1SD당 logit 변화")
    fig.suptitle("Best 모델 곡선별 위치 기여도 (5-fold 평균, 띠 = ±1SD)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fpca_recovered_bodycomp.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    study_design(); fpca_elbow(); forest(); fpca_recovered_bodycomp()
    print("saved to", OUT)
