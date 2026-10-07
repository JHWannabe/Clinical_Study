from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# 세미나 Introduction용 AEC 그림을 강남 실제 데이터로 생성한다(candle/box plot 대신 128구간 곡선).
#  - fpca_concept_real.png: 원본 AEC 곡선 + FPCA 모드(PC1~PC3) (Key Concept 2)
#  - aec_curve_by_sex.png, aec_curve_by_disease_sex.png: 성별/질환x성별 평균 곡선 ± 95% CI
# p값은 환자별 평균 AEC의 Mann-Whitney. 인자 없이 실행하면 셋 다 생성.

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA

from clinic4_logistic_regression import DATA_XLSX, DISEASES, PROJECT_ROOT, SEED
from clinic4_aec_bodycomp_logistic import AEC_FPCA_N, AEC_PREFIX, curve_cols, load_data_with_curves

sys.stdout.reconfigure(encoding="utf-8")
OUT = PROJECT_ROOT / "outputs" / "clinic4" / "figures"

COL = {"F": "#c0392b", "M": "#1f3b63"}
NAME = {"F": "여성", "M": "남성"}
Z = np.linspace(0, 1, 128)


# 평균 곡선 + 95% CI 밴드
def band(ax, X, color, label, ls="-"):
    m, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
    ax.fill_between(Z, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, lw=0)
    ax.plot(Z, m, color=color, lw=2.4, ls=ls, label=label)


def pstr(a, b):
    p = mannwhitneyu(a.mean(1), b.mean(1)).pvalue
    return ("p<0.001" if p < 0.001 else f"p={p:.3f}") + (" (*)" if p < 0.05 else " (ns)")


def curves_by_group() -> None:
    df = load_data_with_curves(DATA_XLSX)
    X = df[curve_cols(AEC_PREFIX)].to_numpy(float)
    sex = df["PatientSex"].astype(str).str.upper().to_numpy()

    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    for s in "FM":
        band(ax, X[sex == s], COL[s], f"{NAME[s]} (n={(sex == s).sum()})")
    ax.set(xlabel="z축: 간 → 치골 (128구간, 정규화)", ylabel="평균 AEC 튜브 전류 (mA)",
           title="성별 AEC-128 평균 곡선 (강남 코호트, 평균 ± 95% CI)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "aec_curve_by_sex.png", dpi=190)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)
    for ax, d in zip(axes, DISEASES):
        y = df[d].to_numpy().astype(int) == 1
        for s in "FM":
            for pos, ls in ((False, "--"), (True, "-")):
                m = (sex == s) & (y == pos)
                band(ax, X[m], COL[s], f"{NAME[s]} {'양성' if pos else '음성'}", ls)
        ps = "  ".join(f"{NAME[s]} {pstr(X[(sex == s) & y], X[(sex == s) & ~y])}" for s in "FM")
        ax.set_title(f"{d}\n{ps}", fontsize=10)
        ax.set_xlabel("z축: 간 → 치골 (128구간)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("평균 AEC 튜브 전류 (mA)"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("질환 유무에 따른 AEC-128 곡선 (강남 코호트, 평균 ± 95% CI; 실선=양성, 점선=음성)")
    fig.tight_layout(); fig.savefig(OUT / "aec_curve_by_disease_sex.png", dpi=190)
    print([(d, int(df[d].sum())) for d in DISEASES])


def fpca_concept() -> None:
    df = load_data_with_curves(DATA_XLSX)
    X = df[curve_cols(AEC_PREFIX)].to_numpy(float)
    pca = PCA(n_components=AEC_FPCA_N, random_state=SEED).fit(X)
    z = np.linspace(0, 1, X.shape[1])
    sd = np.sqrt(pca.explained_variance_)
    rng = np.random.default_rng(SEED)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    for i in rng.choice(len(X), 18, replace=False):
        a.plot(z, X[i], color="#a9bddc", lw=1.2)
    a.plot(z, pca.mean_, color="#1f3b63", lw=3, label=f"평균 곡선 (n={len(X):,})")
    a.set(title="원본 곡선들 (강남 실제 AEC, 128구간)", xlabel="간 -> 치골 (0~1 정규화)", ylabel="AEC 튜브 전류 (mA)")
    b.plot(z, pca.mean_, color="#1f3b63", lw=3, label="평균 곡선")
    for k, c in enumerate(["#e07b22", "#2a9d6b", "#a83a5a"]):
        b.plot(z, pca.mean_ + 2 * sd[k] * pca.components_[k], color=c, ls="--", lw=2,
               label=f"평균 + 2SD x PC{k+1} (설명분산 {pca.explained_variance_ratio_[k]*100:.1f}%)")
    b.set(title="FPCA 모드(PC1~PC3)", xlabel="간 -> 치골 (0~1 정규화)")
    for ax in (a, b):
        ax.legend(frameon=False, fontsize=9, loc="upper left" if ax is a else "lower right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("실제 강남 코호트 AEC — 곡선을 몇 개의 대표 모양 패턴으로 압축")
    fig.tight_layout()
    fig.savefig(OUT / "fpca_concept_real.png", dpi=150)
    print("saved fpca_concept_real.png", pca.explained_variance_ratio_)


if __name__ == "__main__":
    fpca_concept()
    curves_by_group()
