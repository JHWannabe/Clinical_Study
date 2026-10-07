from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# data/{gangnam,sinchon}_landmark_filtered.xlsx(landmark QC 통과 환자)로 outputs/clinic4/figures의 그림 11장을 다시 만든다.
# 곡선 = liver_dome~inferior_pubic_margin을 128구간으로 리샘플한 AEC/VAT/SAT/TAMA (final_dataset의 LAMA/NAMA/IMATA는 이 파일에 없어 TAMA로 통합).
# 모델: baseline(성별+나이/신장/체중) vs best(+ AEC·VAT·SAT·TAMA FPCA 점수, 성분 수 = 강남 곡선의 Kneedle elbow).
# 평가: 강남 전체(언더샘플링 없음)를 환자 단위 stratified 5-fold로 튜닝+OOF, 강남 전체로 재학습한 고정 모델을 신촌에 적용.
# PCA/scaler는 학습 fold로만 fit(누수 없음). 예측·요약은 outputs/clinic4/landmark/clinic4_figures_5fold.xlsx에 저장.

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from clinic4_landmark_vat_auc import END, N_POINTS, START, load_cohort
from clinic4_logistic_regression import CLINICAL_COLS, DATA_DIR, DISEASES, PARAM_GRID, PROJECT_ROOT, SEED, save_sheet
from delong_utils import delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")
OUT = PROJECT_ROOT / "outputs" / "clinic4" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
XLSX = "landmark/clinic4_figures_5fold.xlsx"
CURVES = ["AEC", "VAT", "SAT", "TAMA"]
N_PC = 20  # elbow 탐색용 PC 후보 수
NAME = {"HTN": "고혈압", "DM": "당뇨병", "CKD": "만성신장질환"}
SEXNAME = {"F": "여성", "M": "남성"}
COLSEX = {"F": "#c0392b", "M": "#1f3b63"}
C_BASE, C_BEST = "#7f8c9a", "#c0392b"
Z = np.linspace(0, 1, N_POINTS)
# ΔAUC 그림의 비교 모델: 이름 -> 추가할 곡선
MODELS = {"baseline": [], "+AEC": ["AEC"], "+VAT": ["VAT"], "+SAT": ["SAT"], "+TAMA": ["TAMA"],
          "+체성분(VAT,SAT,TAMA)": ["VAT", "SAT", "TAMA"], "+AEC +체성분": CURVES}


# 코호트 하나: (patient meta df, {AEC/VAT/SAT/TAMA: (n,128)}). AEC는 load_cohort와 같은 환자 순서/리샘플 방식으로 추가
def load(cohort: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df, curves = load_cohort(cohort)
    path = DATA_DIR / f"{cohort}_landmark_filtered.xlsx"
    lm = pd.read_excel(path, sheet_name="landmarks").set_index("PatientID").loc[df["PatientID"]]
    a = pd.read_excel(path, sheet_name="aec_total").set_index("PatientID").filter(regex="^aec_\\d+$").loc[df["PatientID"]].to_numpy(float)
    s, e = lm[f"{START}_slice"].to_numpy(int), lm[f"{END}_slice"].to_numpy(int)
    curves["AEC"] = np.stack([np.interp(np.linspace(s[i], e[i], N_POINTS), np.arange(1, a.shape[1] + 1), np.nan_to_num(a[i])) for i in range(len(df))])
    return df, curves


def take(d, idx):
    return d[0].iloc[idx], {t: v[idx] for t, v in d[1].items()}


# 첫 점-마지막 점 직선까지의 수직거리가 최대인 PC 번호(scree의 Kneedle elbow)
def kneedle_elbow(ev: np.ndarray) -> int:
    x = np.arange(1, len(ev) + 1, dtype=float); xn = (x - x.min()) / (x.max() - x.min()); yn = (ev - ev.min()) / (ev.max() - ev.min())
    pts = np.column_stack([xn, yn]); p1, p2 = pts[0], pts[-1]; v = (p2 - p1) / np.linalg.norm(p2 - p1)
    return int(np.argmax(np.linalg.norm((pts - p1) - np.outer((pts - p1) @ v, v), axis=1))) + 1


# fit 데이터로 PCA(곡선별 K개)·scaler를 학습해 fit과 others의 입력 행렬을 만든다: [성별, 표준화(나이/신장/체중, FPCA 점수...)]
def make_x(tissues, fit, others):
    pcas = {t: PCA(K[t], random_state=SEED).fit(fit[1][t]) for t in tissues}

    def raw(d):
        sex = (d[0]["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
        return sex, np.hstack([d[0][CLINICAL_COLS].to_numpy(float)] + [pcas[t].transform(d[1][t]) for t in tissues])
    sex, a = raw(fit); sc = StandardScaler().fit(a)
    xs = [np.column_stack([sex, sc.transform(a)])] + [np.column_stack([s, sc.transform(r)]) for s, r in map(raw, others)]
    return xs, pcas, sc


# 질환 하나 x 모델 하나: 5-fold 평균 AUC 최대 하이퍼파라미터 -> OOF 예측, 강남 전체 재학습 -> 신촌 예측
def run(disease, model, tissues, g, s):
    y, ye = g[0][disease].to_numpy(int), s[0][disease].to_numpy(int)
    folds = [(tr, te) + tuple(make_x(tissues, take(g, tr), [take(g, te)])[0]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y)]
    fold_auc = lambda p: [roc_auc_score(y[te], LogisticRegression(max_iter=2000, **p).fit(a, y[tr]).predict_proba(b)[:, 1]) for tr, te, a, b in folds]
    best = max(ParameterGrid(PARAM_GRID), key=lambda p: np.mean(fold_auc(p)))  # ponytail: 5-fold 평균 AUC 최대(덱의 --search와 동일), nested CV 아님
    oof = np.empty(len(y))
    for tr, te, a, b in folds:
        oof[te] = LogisticRegression(max_iter=2000, **best).fit(a, y[tr]).predict_proba(b)[:, 1]
    xg, xs = make_x(tissues, g, [s])[0]
    ext = LogisticRegression(max_iter=2000, **best).fit(xg, y).predict_proba(xs)[:, 1]
    row = {"disease": disease, "model": model, "n_gangnam": len(y), "n_sinchon": len(ye), "internal_mean": np.mean(fold_auc(best)), "oof_auc": roc_auc_score(y, oof),
           "external_auc": roc_auc_score(ye, ext), "brier_gangnam_oof": brier_score_loss(y, oof), "brier_sinchon_frozen": brier_score_loss(ye, ext), **{f"best_{k}": str(v) for k, v in best.items()}}
    pred = pd.concat([pd.DataFrame({"disease": disease, "model": model, "cohort": "gangnam", "patient_id": g[0]["PatientID"].to_numpy(), "y": y, "score": oof}),
                      pd.DataFrame({"disease": disease, "model": model, "cohort": "sinchon", "patient_id": s[0]["PatientID"].to_numpy(), "y": ye, "score": ext})])
    print(f"[{disease}/{model}] OOF={row['oof_auc']:.3f} ext={row['external_auc']:.3f} params={best}", flush=True)
    return row, pred


def band(ax, X, color, label, ls="-"):
    m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
    ax.fill_between(Z, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, lw=0)
    ax.plot(Z, m, color=color, lw=2.4, ls=ls, label=label)


def pstr(a, b):
    p = mannwhitneyu(a.mean(1), b.mean(1)).pvalue
    return ("p<0.001" if p < 0.001 else f"p={p:.3f}") + (" (*)" if p < 0.05 else " (ns)")


def data_distribution(g, s):
    cols = [("PatientAge", "Age"), ("Height", "Height"), ("Weight", "Weight"), ("BMI", "BMI")]
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 4.16))
    for ax, (c, lab) in zip(axes, cols):
        for name, d in (("Gangnam", g[0]), ("Sinchon", s[0])):
            ax.hist(d[c].astype(float).dropna(), bins=40, density=True, histtype="step", linewidth=1.5, label=name)
        ax.set_title(lab); ax.set_ylabel("Density")
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "data_distribution.png", dpi=150); plt.close(fig)


def aec_curves(g):
    df, X = g[0], g[1]["AEC"]
    sex = df["PatientSex"].astype(str).str.upper().to_numpy()
    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    for sx in "FM":
        band(ax, X[sex == sx], COLSEX[sx], f"{SEXNAME[sx]} (n={(sex == sx).sum()})")
    ax.set(xlabel="z축: liver dome → 치골 (128구간, 정규화)", ylabel="평균 AEC 튜브 전류 (mA)", title="성별 AEC 평균 곡선 (강남 코호트, 평균 ± 95% CI)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "aec_curve_by_sex.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)
    for ax, d in zip(axes, DISEASES):
        y = df[d].to_numpy(int) == 1
        for sx in "FM":
            for pos, ls in ((False, "--"), (True, "-")):
                band(ax, X[(sex == sx) & (y == pos)], COLSEX[sx], f"{SEXNAME[sx]} {'양성' if pos else '음성'}", ls)
        ax.set_title(f"{d}\n" + "  ".join(f"{SEXNAME[sx]} {pstr(X[(sex == sx) & y], X[(sex == sx) & ~y])}" for sx in "FM"), fontsize=10)
        ax.set_xlabel("z축: liver dome → 치골 (128구간)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("평균 AEC 튜브 전류 (mA)"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("질환 유무에 따른 AEC 곡선 (강남 코호트, 평균 ± 95% CI; 실선=양성, 점선=음성)")
    fig.tight_layout(); fig.savefig(OUT / "aec_curve_by_disease_sex.png", dpi=190); plt.close(fig)


def fpca_concept(g):
    X = g[1]["AEC"]; pca = PCA(3, random_state=SEED).fit(X); sd = np.sqrt(pca.explained_variance_); rng = np.random.default_rng(SEED)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    for i in rng.choice(len(X), 18, replace=False):
        a.plot(Z, X[i], color="#a9bddc", lw=1.2)
    a.plot(Z, pca.mean_, color="#1f3b63", lw=3, label=f"평균 곡선 (n={len(X):,})")
    a.set(title="원본 곡선들 (강남 실제 AEC, 128구간)", xlabel="liver dome -> 치골 (0~1 정규화)", ylabel="AEC 튜브 전류 (mA)")
    b.plot(Z, pca.mean_, color="#1f3b63", lw=3, label="평균 곡선")
    for k, c in enumerate(["#e07b22", "#2a9d6b", "#a83a5a"]):
        b.plot(Z, pca.mean_ + 2 * sd[k] * pca.components_[k], color=c, ls="--", lw=2, label=f"평균 + 2SD x PC{k + 1} (설명분산 {pca.explained_variance_ratio_[k] * 100:.1f}%)")
    b.set(title="FPCA 모드(PC1~PC3)", xlabel="liver dome -> 치골 (0~1 정규화)")
    for ax in (a, b):
        ax.legend(frameon=False, fontsize=9, loc="upper left" if ax is a else "lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("실제 강남 코호트 AEC — 곡선을 몇 개의 대표 모양 패턴으로 압축")
    fig.tight_layout(); fig.savefig(OUT / "fpca_concept_real.png", dpi=150); plt.close(fig)


def box(ax, x, y, w, h, text, fc, size=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="#34495e", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#34495e"))


def study_design(g, s):
    fig, ax = plt.subplots(figsize=(12.2, 4.6)); ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.2, 2.9, 2.6, 1.3, f"Internal\n강남 {len(g[0]):,}명", "#dfe9f5", 12, True)
    box(ax, 0.2, 0.4, 2.6, 1.3, f"External\n신촌 {len(s[0]):,}명", "#fbe5d6", 12, True)
    box(ax, 3.5, 2.9, 3.0, 1.3, "5-fold CV (환자 단위)\n튜닝 + OOF 예측", "#eef3e2", 11)
    box(ax, 7.1, 2.9, 2.2, 1.3, "OOF AUC\nBrier", "#eef3e2", 11)
    box(ax, 3.5, 0.4, 3.0, 1.3, "강남 전체로 재학습\n(고정 모델)", "#eef3e2", 11)
    box(ax, 7.1, 0.4, 2.2, 1.3, "신촌에 동일 적용", "#eef3e2", 11)
    box(ax, 9.9, 1.65, 2.1, 1.3, "AUC·Brier\nDeLong 검정", "#f5e1e1", 11, True)
    arrow(ax, 2.8, 3.55, 3.5, 3.55); arrow(ax, 6.5, 3.55, 7.1, 3.55); arrow(ax, 2.8, 1.05, 3.5, 1.05); arrow(ax, 6.5, 1.05, 7.1, 1.05)
    arrow(ax, 9.3, 3.4, 9.9, 2.6); arrow(ax, 9.3, 1.2, 9.9, 2.0); arrow(ax, 8.2, 2.9, 8.2, 1.7)
    ax.text(5.0, 2.3, "Input: Baseline(성별·나이·키·몸무게)  vs  +AEC·체성분(VAT·SAT·TAMA) FPCA 점수 (liver dome~치골 128구간)", ha="center", fontsize=10, color="#555")
    fig.tight_layout(); fig.savefig(OUT / "study_design.png", dpi=170); plt.close(fig)


def fpca_elbow(g):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4), sharey=True)
    for ax, t in zip(axes.ravel(), CURVES):
        ev = PCA(N_PC, random_state=SEED).fit(g[1][t]).explained_variance_ratio_ * 100; sel = K[t]
        ax.plot(range(1, N_PC + 1), ev, "o-", color="#34495e", lw=2, ms=4)
        ax.plot([1, N_PC], [ev[0], ev[-1]], ls="--", color="#2e86c1", lw=1.6, label=f"PC1-PC{N_PC} 직선")
        ax.plot(sel, ev[sel - 1], "o", ms=14, mfc="none", mec=C_BEST, mew=2.5); ax.axvline(sel, color=C_BEST, ls="--", lw=1)
        ax.set_title(f"{t}: elbow k={sel} (누적 {ev[:sel].sum():.1f}%)", fontsize=11)
        ax.set_xticks([1, 2, 3, 5, 10, 15, 20]); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        if t == "AEC": ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes[1]: ax.set_xlabel("PC 번호")
    for ax in axes[:, 0]: ax.set_ylabel("설명분산 (%)")
    fig.suptitle(f"FPCA 성분 수 선택: scree 곡선(PC1~PC{N_PC})의 elbow(Kneedle)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fpca_elbow.png", dpi=170); plt.close(fig)


def forest(g):
    terms = ["Sex (M)", "Age", "Height", "Weight"]; fits = {}
    for name, tissues in (("baseline", []), ("best", CURVES)):
        x = make_x(tissues, g, [])[0][0]
        for d in DISEASES:
            r = sm.Logit(g[0][d].to_numpy(int), sm.add_constant(x)).fit(disp=0)
            fits[(d, name)] = (np.exp(r.params[1:5]), np.exp(r.conf_int()[1:5]))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), sharex=True)
    for ax, d in zip(axes, DISEASES):
        for off, name, col, lab in ((-0.17, "baseline", C_BASE, "Baseline"), (0.17, "best", C_BEST, "+AEC+체성분")):
            orr, ci = fits[(d, name)]; yy = np.arange(4) + off
            ax.errorbar(orr, yy, xerr=[orr - ci[:, 0], ci[:, 1] - orr], fmt="o", color=col, capsize=3, label=lab)
        ax.axvline(1, color="k", lw=1, ls="--"); ax.set_xscale("log"); ax.set_yticks(range(4)); ax.set_yticklabels(terms)
        ax.invert_yaxis(); ax.set_title(NAME[d]); ax.set_xlabel("Odds Ratio (95% CI, log)"); ax.grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "forest_or.png", dpi=170); plt.close(fig)


# Best 모델 곡선별 FPCA 계수를 128-slice 축으로 환산(+1SD당 logit 변화, fold별 PCA·scaler는 학습 fold로만 fit, 규제 없는 Logit, 5-fold 평균 ± SD)
def fpca_recovered(g):
    rec = {d: {t: [] for t in CURVES} for d in DISEASES}
    for d in DISEASES:
        y = g[0][d].to_numpy(int)
        for tr, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, y):
            sub = take(g, tr); x, pcas, sc = make_x(CURVES, sub, [])
            beta = sm.Logit(y[tr], sm.add_constant(x[0])).fit(disp=0).params; off = len(CLINICAL_COLS)
            for t in CURVES:
                w = np.zeros(N_POINTS)
                for j in range(K[t]): w += beta[2 + off + j] / sc.scale_[off + j] * pcas[t].components_[j]
                rec[d][t].append(w * sub[1][t].std(0, ddof=1)); off += K[t]
    cols = {"HTN": "#1f77b4", "DM": "#e07b22", "CKD": "#2a9d3b"}; sl = np.arange(1, N_POINTS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.6), sharex=True)
    for ax, t in zip(axes.ravel(), CURVES):
        for d in DISEASES:
            r = np.array(rec[d][t]); mu, sd = r.mean(0), r.std(0, ddof=1)
            ax.fill_between(sl, mu - sd, mu + sd, color=cols[d], alpha=0.12, lw=0); ax.plot(sl, mu, color=cols[d], lw=2, label=NAME[d] if t == "AEC" else d)
        ax.axhline(0, color="gray", ls="--", lw=1); ax.set_title(t); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]: ax.set_xlabel("slice (1=liver dome → 128=치골)")
    for ax in axes[:, 0]: ax.set_ylabel("+1SD당 logit 변화")
    fig.suptitle("Best 모델 곡선별 위치 기여도 (5-fold 평균, 띠 = ±1SD)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "fpca_recovered_bodycomp.png", dpi=170); plt.close(fig)


def calibration(pred, summ):
    files = {"baseline": ("calibration_plot_baseline.png", "Baseline"), "best": ("calibration_plot_best.png", "Best (+AEC+체성분)")}
    for key, (fn, label) in files.items():
        model = "baseline" if key == "baseline" else "+AEC +체성분"
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, d in zip(axes, DISEASES):
            row = summ[(summ.disease == d) & (summ.model == model)].iloc[0]
            for cohort, (color, ls, bcol) in {"gangnam": ("tab:blue", "-", "brier_gangnam_oof"), "sinchon": ("tab:orange", "--", "brier_sinchon_frozen")}.items():
                p = pred[(pred.disease == d) & (pred.model == model) & (pred.cohort == cohort)]
                grp = pd.DataFrame({"y": p.y, "s": p.score, "b": pd.qcut(p.score, 5, duplicates="drop")}).groupby("b", observed=True).agg(pr=("s", "mean"), ob=("y", "mean"))
                ax.plot(grp.pr, grp.ob, marker="o", ls=ls, color=color, label=f"{cohort} (Brier={row[bcol]:.3f})")
            ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1, label="Perfect calibration")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed frequency", title=f"{NAME[d]}({d})"); ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Calibration Plot — {label}, Internal=5-fold OOF / External=full-refit, 5 quantile bins")
        fig.tight_layout(); fig.savefig(OUT / fn, dpi=150); plt.close(fig)


def delta_auc(pred, summ):
    models = [m for m in MODELS if m != "baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); w = 0.35
    for ax, d in zip(axes, DISEASES):
        for off, cohort, color in ((-w / 2, "gangnam", "tab:blue"), (w / 2, "sinchon", "tab:orange")):
            g = pred[(pred.disease == d) & (pred.cohort == cohort)]; piv = g.pivot(index="patient_id", columns="model", values="score")
            y = g.drop_duplicates("patient_id").set_index("patient_id")["y"].loc[piv.index].to_numpy(int)
            for i, m in enumerate(models):
                r = delong_paired_auc_test(y, piv["baseline"].to_numpy(), piv[m].to_numpy()); delta = r["auc_b"] - r["auc_a"]
                ax.bar(i + off, delta, width=w, color=color, label=cohort if i == 0 else None)
                if r["p_value"] < 0.05: ax.annotate("*", (i + off, delta), ha="center", va="bottom" if delta >= 0 else "top", fontsize=13)
        ax.axhline(0, color="black", lw=1); ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_title(d); ax.set_ylabel("Δ AUC vs Baseline"); ax.margins(y=0.22); ax.legend(fontsize=9, loc="upper left")
    fig.text(0.5, -0.02, "* p<0.05 vs Baseline (paired DeLong), Internal=5-fold CV OOF / External=full-refit", ha="center", fontsize=10, color="dimgray")
    fig.tight_layout(); fig.savefig(OUT / "delta_auc_vs_baseline.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    global K
    g, s = load("gangnam"), load("sinchon")
    K = {t: kneedle_elbow(PCA(N_PC, random_state=SEED).fit(g[1][t]).explained_variance_ratio_) for t in CURVES}  # 강남 곡선으로만 결정
    print("FPCA 성분 수(Kneedle elbow):", K)
    data_distribution(g, s); aec_curves(g); fpca_concept(g); study_design(g, s); fpca_elbow(g); forest(g); fpca_recovered(g)
    rows, preds = [], []
    for name, tissues in MODELS.items():
        for d in DISEASES:
            r, p = run(d, name, tissues, g, s); rows.append(r); preds.append(p)
    summ, pred = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    save_sheet(pred, XLSX, "predictions"); save_sheet(summ, XLSX, "summary")
    calibration(pred, summ); delta_auc(pred, summ)
    print(summ[["disease", "model", "internal_mean", "oof_auc", "external_auc"]].round(3).to_string(index=False)); print("saved to", OUT)


if __name__ == "__main__":
    main()
