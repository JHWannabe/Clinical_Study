from __future__ import annotations

# 세그멘테이션 없이 얻는 AEC-128(clinic4 + mean mAs + AEC FPCA) 모델이 세그멘테이션 기반 VAT/SAT 모델(clinic4 +
# VAT + SAT)에 비열등(non-inferior)한지 검정(2026-09-03, 사용자 확인: 연구 목표는 "AEC 가치 입증", 우월성이 아니라
# 대체 가능성 프레임). 입력은 step_disease_logistic.py가 저장한 outputs/01_disease_logistic/logistic/predictions.csv
# (같은 코호트·같은 fold의 OOF/external 예측)를 patient_id로 정렬해 그대로 사용 — 재학습 없음.
#
# 통계: paired DeLong으로 Δ = AUC(AEC 모델) − AUC(VAT/SAT 모델)와 SE를 구하고,
#   H0: Δ ≤ −δ  vs  H1: Δ > −δ  (one-sided z = (Δ + δ)/SE)  — δ는 사전 지정 비열등 마진.
#   주 마진 δ=0.02(AUC 비열등성 연구에서 관례적으로 쓰는 0.02~0.05 중 가장 보수적 값), 민감도로 0.01/0.015/0.03/0.05.
#   비열등 판정 = 양측 95% CI 하한 > −δ (one-sided α=0.025와 동치). 6개 검정(3질환×2코호트)은 BH-FDR도 병기.
# 주의: 이 스크립트를 작성한 시점에 Δ 점추정치는 이미 확인된 상태였음(2026-09-03 세션) — 논문에서는 마진을
# 문헌 근거로 정당화하고 이 사실을 투명하게 기술할 것. 마진을 결과 보고 고르지 않도록 후보를 전부 보고한다.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGISTIC_DIR = PROJECT_ROOT / "outputs" / "01_disease_logistic" / "logistic"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "01_disease_logistic" / "noninferiority"

FEATURES = ["HTN", "DM", "CKD"]
COHORTS = ["internal", "external"]
REFERENCE_MODEL = "clinic4_vatsat"          # 세그멘테이션 기반 기준 모델
TEST_MODEL = "clinic4_meanmAs_aec"          # 세그멘테이션 없는 AEC 모델
CONTEXT_MODEL = "clinic4"                   # 참고용 바닥선
PRIMARY_MARGIN = 0.02
MARGINS = [0.01, 0.015, 0.02, 0.03, 0.05]
ALPHA_ONE_SIDED = 0.025

MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_vatsat": "clinic4 + VAT + SAT (segmentation)",
    "clinic4_meanmAs_aec": "clinic4 + mean mAs + AEC (segmentation-free)",
}


# --- DeLong (step_disease_logistic.py와 동일 구현, 코드베이스 관례상 중복 허용) ---
def _delong_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_covariance(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    n_neg = scores.shape[1] - n_pos
    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    k = scores.shape[0]
    tx = np.vstack([_delong_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_delong_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_delong_midrank(scores[r]) for r in range(k)])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


# Δ = AUC(b) − AUC(a), SE, 양측 p (paired DeLong)
def delong_paired(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(var)) if var > 0 else float("nan")
    z = diff / se if se > 0 else float("nan")
    p_two = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "se": se, "z": z, "p_two_sided": p_two}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def aligned_scores(pred: pd.DataFrame, feat: str, cohort: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sub = pred[(pred["feature"] == feat) & (pred["cohort"] == cohort)]
    ref = sub[sub["model"] == REFERENCE_MODEL].set_index("patient_id")
    ids = ref.index.to_numpy()
    y = ref["y"].to_numpy().astype(int)
    scores = {}
    for m in (REFERENCE_MODEL, TEST_MODEL, CONTEXT_MODEL):
        s = sub[sub["model"] == m].set_index("patient_id").reindex(ids)
        assert s["score"].notna().all(), f"{feat}/{cohort}/{m}: patient_id 정렬 실패"
        assert (s["y"].to_numpy().astype(int) == y).all(), f"{feat}/{cohort}/{m}: y 불일치"
        scores[m] = s["score"].to_numpy()
    return y, scores


def plot_forest(res: pd.DataFrame, out_path: Path) -> None:
    rows = [(f, c) for f in FEATURES for c in COHORTS]
    fig, ax = plt.subplots(figsize=(14, 9))
    ypos = np.arange(len(rows))[::-1]
    for yi, (f, c) in zip(ypos, rows):
        r = res[(res["feature"] == f) & (res["cohort"] == c)].iloc[0]
        color = "#1b6ba8" if r["noninferior_primary"] else "#c0392b"
        ax.errorbar(r["auc_diff"], yi, xerr=[[r["auc_diff"] - r["ci95_lower"]], [r["ci95_upper"] - r["auc_diff"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=3, capsize=8, markersize=12)
    ax.axvline(0, color="gray", linestyle="-", linewidth=1.5)
    ax.axvline(-PRIMARY_MARGIN, color="#c0392b", linestyle="--", linewidth=2.5, label=f"non-inferiority margin -δ (δ={PRIMARY_MARGIN})")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{f} / {c}" for f, c in rows], fontsize=24)
    ax.set_xlabel("ΔAUC = AEC(segmentation-free) - VAT/SAT(segmentation), 95% CI", fontsize=24)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlim(-0.05, 0.05)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=18, loc="lower right", frameon=False)
    ax.set_title("Non-inferiority of AEC model vs segmentation-based VAT/SAT model", fontsize=24, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forest plot to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(LOGISTIC_DIR / "predictions.csv")
    missing = {REFERENCE_MODEL, TEST_MODEL, CONTEXT_MODEL} - set(pred["model"].unique())
    assert not missing, f"predictions.csv에 없는 모델: {missing}"

    rows, margin_rows = [], []
    for feat in FEATURES:
        for cohort in COHORTS:
            y, s = aligned_scores(pred, feat, cohort)
            d = delong_paired(y, s[REFERENCE_MODEL], s[TEST_MODEL])
            ci_lo, ci_hi = d["diff"] - 1.96 * d["se"], d["diff"] + 1.96 * d["se"]
            ctx = delong_paired(y, s[CONTEXT_MODEL], s[TEST_MODEL])
            row = {
                "feature": feat, "cohort": cohort, "n": int(len(y)), "n_pos": int(y.sum()),
                "auc_clinic4": float(roc_auc_score(y, s[CONTEXT_MODEL])),
                "auc_reference_vatsat": d["auc_a"], "auc_test_aec": d["auc_b"],
                "auc_diff": d["diff"], "se": d["se"], "ci95_lower": ci_lo, "ci95_upper": ci_hi,
                "p_two_sided_superiority": d["p_two_sided"],
                "aec_vs_clinic4_diff": ctx["diff"], "aec_vs_clinic4_p": ctx["p_two_sided"],
            }
            for delta in MARGINS:
                z_ni = (d["diff"] + delta) / d["se"]
                p_ni = float(stats.norm.sf(z_ni))  # one-sided: H0 Δ ≤ −δ
                row[f"p_ni_delta{delta:g}"] = p_ni
                row[f"noninferior_delta{delta:g}"] = bool(ci_lo > -delta)
                margin_rows.append({"feature": feat, "cohort": cohort, "delta": delta, "z_ni": z_ni, "p_ni_one_sided": p_ni,
                                    "noninferior": bool(ci_lo > -delta)})
            row["p_ni_primary"] = row[f"p_ni_delta{PRIMARY_MARGIN:g}"]
            row["noninferior_primary"] = row[f"noninferior_delta{PRIMARY_MARGIN:g}"]
            rows.append(row)
            print(f"[{feat}/{cohort}] VAT/SAT={d['auc_a']:.4f} AEC={d['auc_b']:.4f} Δ={d['diff']:+.4f} "
                  f"95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] p_NI(δ={PRIMARY_MARGIN})={row['p_ni_primary']:.4f} "
                  f"-> {'NON-INFERIOR' if row['noninferior_primary'] else 'not shown'}")

    res = pd.DataFrame(rows)
    res["q_ni_primary_bh"] = bh_fdr(res["p_ni_primary"].to_numpy())
    res.to_csv(OUTPUT_DIR / "noninferiority_summary.csv", index=False)

    mr = pd.DataFrame(margin_rows)
    mr.to_csv(OUTPUT_DIR / "noninferiority_margin_sensitivity.csv", index=False)
    pivot = mr.pivot_table(index=["feature", "cohort"], columns="delta", values="noninferior", aggfunc="first")
    pivot.to_csv(OUTPUT_DIR / "noninferiority_margin_pivot.csv")

    n_ni = int(res["noninferior_primary"].sum())
    print(f"\nδ={PRIMARY_MARGIN}: {n_ni}/{len(res)} 비열등, BH-FDR q<0.05: {int((res['q_ni_primary_bh'] < 0.05).sum())}/{len(res)}")
    print("마진별 비열등 건수:", {float(k): int(v.sum()) for k, v in pivot.items()})
    print(f"Saved to {OUTPUT_DIR}")
    plot_forest(res, OUTPUT_DIR / "noninferiority_forest.png")


if __name__ == "__main__":
    main()
