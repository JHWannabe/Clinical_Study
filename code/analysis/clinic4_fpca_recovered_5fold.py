from __future__ import annotations
import sys as _s, pathlib as _p; _s.path[:0] = [str(d) for d in _p.Path(__file__).resolve().parents[1].iterdir() if d.is_dir()]  # code/ 하위 폴더를 import 경로에 추가

# Figure 1(AEC FPCA 계수 역변환) 재생성. 학습 방식은 기존 그림과 동일(강남 전체 5-fold, fold마다 PCA(3)+정규화 없는
# Logit 적합 후 128-slice 축으로 역변환해 fold 평균±SD)하되, 왼쪽에 평균 AEC 곡선을 붙이고 양/음 영역에 해석을
# 직접 써 넣어 그림만 봐도 "무엇을 보는 그림인지" 읽히게 한다. 수식은 clinic4_aec_logistic_regression.recover_curve_coefficients 참고.

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import clinic4_aec_logistic_regression as a
from clinic4_logistic_regression import DATA_XLSX, DISEASES, PROJECT_ROOT, SEED

sys.stdout.reconfigure(encoding="utf-8")
OUT = PROJECT_ROOT / "outputs" / "clinic4" / "aec" / "fpca_recovered_curve_annotated.png"
COLORS = {"HTN": "#1f77b4", "DM": "#e07b22", "CKD": "#2a9d3b"}
NAMES = {"HTN": "고혈압", "DM": "당뇨병", "CKD": "만성신장질환"}
SLICES = np.arange(1, 129)


# 질병 하나에 대해 fold별 slice 기여도(5 x 128)를 반환
def recovered_per_fold(df, curve: np.ndarray, disease: str) -> np.ndarray:
    y = df[disease].to_numpy(int)
    cols = a.CLINICAL_BASE_COLS + a.FPCA_COLS
    out = []
    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(curve, y):
        pca = PCA(a.AEC_FPCA_N, random_state=SEED).fit(curve[tr])
        d = df.iloc[tr].copy()
        score = pca.transform(curve[tr])
        for i in range(a.AEC_FPCA_N):
            d[f"aec_fpca_pc{i + 1}"] = score[:, i]
        x, scaler = a.clinical_matrix(d, a.FPCA_COLS)
        beta = sm.Logit(y[tr], sm.add_constant(x)).fit(disp=0).params[1:]
        w = np.zeros(a.N_SLICES)
        for i, c in enumerate(a.FPCA_COLS):
            w += beta[1 + cols.index(c)] / scaler.scale_[cols.index(c)] * pca.components_[i]
        out.append(w * curve[tr].std(axis=0, ddof=1))
    return np.array(out)


def main() -> None:
    df = a.load_data_with_aec(DATA_XLSX)
    curve = df[a.AEC_COLS].to_numpy(float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.3), gridspec_kw={"width_ratios": [1, 2.1]})
    m, s = curve.mean(0), curve.std(0, ddof=1)
    ax0.fill_between(SLICES, m - s, m + s, color="#9aa5b1", alpha=0.3, lw=0)
    ax0.plot(SLICES, m, color="#333", lw=2)
    ax0.set(title="① 입력: 환자 AEC 곡선 (평균 ± 1SD)", ylabel="AEC 튜브 전류 (mA)")

    for d in DISEASES:
        r = recovered_per_fold(df, curve, d)
        mu, sd = r.mean(0), r.std(0, ddof=1)
        ax1.fill_between(SLICES, mu - sd, mu + sd, color=COLORS[d], alpha=0.13, lw=0)
        ax1.plot(SLICES, mu, color=COLORS[d], lw=2.2, label=NAMES[d])
    ax1.axhline(0, color="gray", lw=1, ls="--")
    lo, hi = -0.03, 0.03; ax1.set_ylim(lo, hi)
    ax1.axhspan(0, hi, color="#d9534f", alpha=0.05)
    ax1.axhspan(lo, 0, color="#5b8def", alpha=0.05)
    ax1.text(0.99, 0.96, "위쪽(+): 이 위치 AEC가 높을수록 질환 확률 ↑", transform=ax1.transAxes, ha="right", va="top",
             color="#b03a2e", fontsize=10)
    ax1.text(0.99, 0.04, "아래쪽(-): 이 위치 AEC가 높을수록 질환 확률 ↓", transform=ax1.transAxes, ha="right", va="bottom",
             color="#2e5cb8", fontsize=10)
    ax1.set(title="② 결과: 위치별 질환 확률(logit) 기여도 (5-fold 평균, 띠 = ±1SD)",
            ylabel="그 위치 AEC가 +1SD 클 때 logit 변화")
    ax1.legend(loc="lower left", frameon=False)

    for ax in (ax0, ax1):
        ax.set_xlabel("스캔 위치 (slice 1 = 간 쪽 → 128 = 치골 쪽)")
        ax.set_xlim(1, 128)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
