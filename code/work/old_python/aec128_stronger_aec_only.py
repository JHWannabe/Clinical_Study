from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec128_common_shape_feature import FILES, load_aec128  # noqa: E402
from aec128_cylindrical_features import smooth_log_profile  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_stronger_aec_only"
DEEP_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_deep_feature_mining"
SEED = 20260629
RNG = np.random.default_rng(SEED)
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
K_GRID = [1, 3, 5, 10, 20, 40]


def metric_row(name: str, y: np.ndarray, score: np.ndarray) -> dict:
    """모델 이름과 점수로부터 AUC/AP/로그손실/Brier를 계산해 한 행으로 정리."""
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))
    return {
        "model": name,
        "auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int = 3000) -> dict:
    """점수의 AUC를 부트스트랩 재표본추출로 반복 계산해 평균과 95% 신뢰구간을 추정."""
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        rows.append(roc_auc_score(yy, score[idx]))
    arr = np.asarray(rows)
    return {
        "mean": float(arr.mean()),
        "ci2.5": float(np.quantile(arr, 0.025)),
        "ci97.5": float(np.quantile(arr, 0.975)),
    }


def load_deep_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """딥 특징 채굴 결과 CSV를 읽어, 매칭필터(atlas_) 특징을 제외한 순수 비지도 모양 특징만 남기고, 양쪽 코호트 모두 결측이 적고 분산이 있는 컬럼만 걸러냄."""
    tr = pd.read_csv(DEEP_DIR / "g1090_aec128_deep_features_patient_level.csv")
    te = pd.read_csv(DEEP_DIR / "sdata_aec128_deep_features_patient_level.csv")
    # Keep only fixed unsupervised shape features for train-only model selection.
    keep = [
        c
        for c in tr.columns
        if c != "low_smi"
        and not c.startswith("atlas_")
        and c in te.columns
        and pd.api.types.is_numeric_dtype(tr[c])
    ]
    tr = tr[keep].copy()
    te = te[keep].copy()
    valid = []
    for c in keep:
        a = pd.to_numeric(tr[c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(te[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(a).mean() > 0.95 and np.isfinite(b).mean() > 0.95 and np.nanstd(a) > 1e-9:
            valid.append(c)
    return tr[valid], te[valid]


def raw_profile_feature_tables(datasets: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """로그 프로파일 자체(128)+1차 도함수(128)+2차 도함수(128)+DCT 계수(20)를 이어붙인 400여 차원 원시 특징 행렬을 만듦."""
    p_tr = smooth_log_profile(datasets["g1090"]["x"])
    p_te = smooth_log_profile(datasets["sdata"]["x"])
    d1_tr = np.gradient(p_tr, axis=1)
    d1_te = np.gradient(p_te, axis=1)
    d2_tr = np.gradient(d1_tr, axis=1)
    d2_te = np.gradient(d1_te, axis=1)
    c_tr = dct(p_tr, type=2, norm="ortho", axis=1)[:, 1:21]
    c_te = dct(p_te, type=2, norm="ortho", axis=1)[:, 1:21]
    xtr = np.column_stack([p_tr, d1_tr, d2_tr, c_tr])
    xte = np.column_stack([p_te, d1_te, d2_te, c_te])
    names = (
        [f"logp_{i:03d}" for i in range(1, 129)]
        + [f"d1_{i:03d}" for i in range(1, 129)]
        + [f"d2_{i:03d}" for i in range(1, 129)]
        + [f"dct_{i:02d}" for i in range(1, 21)]
    )
    return xtr, xte, names


def finite_fill_train_apply(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train 평균(결측 제외)으로 train·test 양쪽의 결측값을 채움."""
    mu = np.nanmean(xtr, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    return np.where(np.isfinite(xtr), xtr, mu), np.where(np.isfinite(xte), xte, mu)


def orient_auc(feature: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """특징의 AUC가 0.5 미만이면 부호를 뒤집어(sign=-1) AUC를 0.5 이상으로 맞춰 반환 (방향에 상관없이 판별력만 비교하기 위함)."""
    auc = roc_auc_score(y, feature)
    sign = 1
    if auc < 0.5:
        auc = 1.0 - auc
        sign = -1
    return float(auc), sign


def select_top_features(x: np.ndarray, y: np.ndarray, names: list[str], k: int) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """모든 특징의 방향보정 AUC를 계산해 0.5에서 가장 먼(=판별력이 가장 큰) 상위 k개를 선택하고, 전체 랭킹 표도 함께 반환."""
    rows = []
    for j, name in enumerate(names):
        col = x[:, j]
        if np.nanstd(col) < 1e-10:
            continue
        try:
            auc, sign = orient_auc(col, y)
        except ValueError:
            continue
        rows.append({"idx": j, "feature": name, "oriented_auc": auc, "sign": sign, "abs_auc_distance": abs(auc - 0.5)})
    df = pd.DataFrame(rows).sort_values(["abs_auc_distance", "oriented_auc"], ascending=False)
    sub = df.head(min(k, len(df)))
    return sub["idx"].to_numpy(dtype=int), sub["sign"].to_numpy(dtype=float), sub["feature"].tolist(), df


def apply_oriented_standardized(xtr: np.ndarray, xva: np.ndarray, idx: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """선택된 특징 idx만 골라 방향(sign)을 맞추고, train 기준으로 표준화해 train·검증 데이터에 함께 적용."""
    a = xtr[:, idx] * sign[None, :]
    b = xva[:, idx] * sign[None, :]
    mu = np.mean(a, axis=0)
    sd = np.std(a, axis=0)
    sd[sd < 1e-8] = 1.0
    return (a - mu) / sd, (b - mu) / sd


def choose_k_for_average(x: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[int, pd.DataFrame]:
    """4-fold 교차검증으로 "상위 k개 특징의 단순 평균"을 점수로 썼을 때, 여러 k 후보 중 최적값을 AUC 기준으로 선택."""
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    rows = []
    for k in K_GRID:
        score = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in skf.split(x, y):
            idx, sign, _, _ = select_top_features(x[tr_idx], y[tr_idx], names, k)
            a, b = apply_oriented_standardized(x[tr_idx], x[va_idx], idx, sign)
            score[va_idx] = b.mean(axis=1)
        rows.append({"k": k, "cv_auc": float(roc_auc_score(y, score)), "cv_ap": float(average_precision_score(y, score))})
    df = pd.DataFrame(rows)
    best = df.sort_values(["cv_auc", "cv_ap"], ascending=False).iloc[0]
    return int(best["k"]), df


def nested_topk_average(x: np.ndarray, y: np.ndarray, xte: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """외부 폴드마다 내부에서 k와 특징을 다시 선택하는 중첩교차검증으로 "상위 k개 평균" 모델의 OOF/외부 점수를 만들고, 폴드별 선택 내역을 메타데이터로 기록."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 2)
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(outer.split(x, y), start=1):
        k, cv = choose_k_for_average(x[tr_idx], y[tr_idx], names)
        idx, sign, selected, _ = select_top_features(x[tr_idx], y[tr_idx], names, k)
        _, b = apply_oriented_standardized(x[tr_idx], x[va_idx], idx, sign)
        oof[va_idx] = b.mean(axis=1)
        fold_rows.append({"fold": fold, "selected_k": k, "features": selected})
    final_k, final_cv = choose_k_for_average(x, y, names)
    idx, sign, selected, ranking = select_top_features(x, y, names, final_k)
    a, b = apply_oriented_standardized(x, xte, idx, sign)
    ext = b.mean(axis=1)
    meta = {
        "final_k": final_k,
        "final_selected_features": selected,
        "folds": fold_rows,
        "final_k_cv": final_cv.to_dict(orient="records"),
        "final_feature_ranking_top50": ranking.head(50).to_dict(orient="records"),
    }
    return oof, ext, meta


def choose_ridge_params(x: np.ndarray, y: np.ndarray, names: list[str], topk_grid: list[int]) -> tuple[int, float, pd.DataFrame]:
    """4-fold 교차검증으로 "상위 k개 특징 + 로지스틱 회귀" 조합에서 k와 정규화 강도 C의 최적 쌍을 AUC 기준으로 함께 탐색."""
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 3)
    rows = []
    for k in topk_grid:
        for c in C_GRID:
            score = np.zeros(len(y), dtype=float)
            for tr_idx, va_idx in skf.split(x, y):
                idx, sign, _, _ = select_top_features(x[tr_idx], y[tr_idx], names, k)
                a, b = apply_oriented_standardized(x[tr_idx], x[va_idx], idx, sign)
                model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED)
                model.fit(a, y[tr_idx])
                score[va_idx] = model.decision_function(b)
            rows.append(
                {
                    "k": k,
                    "C": c,
                    "cv_auc": float(roc_auc_score(y, score)),
                    "cv_ap": float(average_precision_score(y, score)),
                }
            )
    df = pd.DataFrame(rows)
    best = df.sort_values(["cv_auc", "cv_ap"], ascending=False).iloc[0]
    return int(best["k"]), float(best["C"]), df


def nested_topk_ridge(
    x: np.ndarray,
    y: np.ndarray,
    xte: np.ndarray,
    names: list[str],
    topk_grid: list[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """외부 폴드마다 내부에서 k·C·특징을 다시 선택하는 중첩교차검증으로 "상위 k개+로지스틱" 모델의 OOF/외부 점수를 만들고, 폴드별 선택 내역을 메타데이터로 기록."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 4)
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(outer.split(x, y), start=1):
        k, c, cv = choose_ridge_params(x[tr_idx], y[tr_idx], names, topk_grid)
        idx, sign, selected, _ = select_top_features(x[tr_idx], y[tr_idx], names, k)
        a, b = apply_oriented_standardized(x[tr_idx], x[va_idx], idx, sign)
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED)
        model.fit(a, y[tr_idx])
        oof[va_idx] = model.decision_function(b)
        fold_rows.append({"fold": fold, "selected_k": k, "selected_C": c, "features": selected})
    final_k, final_c, final_cv = choose_ridge_params(x, y, names, topk_grid)
    idx, sign, selected, ranking = select_top_features(x, y, names, final_k)
    a, b = apply_oriented_standardized(x, xte, idx, sign)
    model = LogisticRegression(C=final_c, solver="lbfgs", max_iter=5000, random_state=SEED)
    model.fit(a, y)
    ext = model.decision_function(b)
    meta = {
        "final_k": final_k,
        "final_C": final_c,
        "final_selected_features": selected,
        "folds": fold_rows,
        "final_param_cv": final_cv.to_dict(orient="records"),
        "final_feature_ranking_top50": ranking.head(50).to_dict(orient="records"),
    }
    return oof, ext, meta


def fixed_known_scores(deep_tr: pd.DataFrame, deep_te: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """이전 스크립트들에서 이미 정의된 4개의 고정(사전 선택 없는) 특징을 벤치마크 점수로 그대로 가져옴."""
    cols = {
        "known_late_edge": "haar_haar_l5_b12_right_minus_left",
        "known_regional_rebound": "haar_haar_l2_b02_right_minus_left",
        "known_cyl_rebound_mass": "cyl_cyl_late_positive_plus_mid_negative",
        "known_visual_rebound_height": "visual_aec128_rebound_height_peak_minus_valley",
    }
    out = {}
    for name, col in cols.items():
        out[name] = (
            pd.to_numeric(deep_tr[col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(deep_te[col], errors="coerce").to_numpy(dtype=float),
        )
    return out


def plot_scores(results: dict, ytr: np.ndarray, yte: np.ndarray) -> None:
    """대표 4개 후보 점수의 train/외부 분포 히스토그램을 그려 PNG로 저장."""
    selected = ["deep_topk_average", "deep_topk_ridge", "raw_profile_topk_ridge", "known_late_edge"]
    fig, axes = plt.subplots(len(selected), 2, figsize=(11.6, 10.5), sharey=False)
    for row, name in enumerate(selected):
        for col, (split, y) in enumerate([("g1090_oof", ytr), ("sdata_external", yte)]):
            ax = axes[row, col]
            score = results[name][split]
            ax.hist(score[y == 0], bins=36, density=True, color="#2F6F73", alpha=0.55, label="Non-low SMI")
            ax.hist(score[y == 1], bins=24, density=True, color="#C84630", alpha=0.58, label="Low SMI")
            ax.set_title(f"{split}: {name}", loc="left", fontsize=9.5, fontweight="bold")
            ax.grid(alpha=0.22)
            if row == len(selected) - 1:
                ax.set_xlabel("AEC-only score")
            if col == 0:
                ax.set_ylabel("Density")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec128_stronger_aec_only_score_distributions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 손으로 고른 소수 특징 대신, "top-k 자동 선택" 방식으로
    수백 개 특징 중 가장 판별력 있는 것들을 train 안에서만 골라 쓰면 AEC 단독 모델이 더 강해지는가?):

    1. g1090/sdata를 로드하고, 딥 특징 채굴 결과(수백 개 비지도 특징)와, 로그 프로파일+도함수+DCT로
       이루어진 원시 400여 차원 특징 행렬을 각각 준비한다.
    2. 3가지 "자동 특징선택" 전략을 중첩교차검증으로 평가:
       - deep_topk_average: 딥 특징 중 상위 k개를 골라 단순 평균한 점수 (k도 내부 CV로 선택)
       - deep_topk_ridge: 딥 특징 중 상위 k개를 골라 로지스틱 회귀로 결합 (k, C 모두 내부 CV로 선택)
       - raw_profile_topk_ridge: 원시 400여 차원 특징에서 상위 k개를 골라 로지스틱으로 결합
       각 전략은 외부 폴드에서 선택 과정 자체를 반복하는 중첩 CV라 train 내에서는 과최적화가
       통제되지만, 그래도 "많은 후보 중에서 고른다"는 점은 동일하다.
    3. 비교 벤치마크로, 이전 스크립트들이 이미 정의해둔 4개의 "고정(선택 없는)" 특징 점수도 그대로 가져온다.
    4. 모든 후보의 train OOF/외부 AUC·AP·로그손실·Brier를 표로 만들고, 외부 AUC는 부트스트랩으로
       신뢰구간까지 추정해 CSV로 저장.
    5. 각 후보의 환자별 점수를 CSV로 저장하고, 선택된 특징 목록·k·C 등 메타데이터와 함께 전체
       결과를 JSON으로 저장.
    6. 대표 후보들의 점수 분포를 히스토그램으로 그려 저장하고, 성능 요약과 선택 메타데이터를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_aec128(path) for name, path in FILES.items()}
    ytr = datasets["g1090"]["y"].astype(int)
    yte = datasets["sdata"]["y"].astype(int)

    deep_tr_df, deep_te_df = load_deep_feature_tables()
    deep_xtr, deep_xte = finite_fill_train_apply(deep_tr_df.to_numpy(dtype=float), deep_te_df.to_numpy(dtype=float))
    deep_names = list(deep_tr_df.columns)
    raw_xtr, raw_xte, raw_names = raw_profile_feature_tables(datasets)
    raw_xtr, raw_xte = finite_fill_train_apply(raw_xtr, raw_xte)

    results: dict[str, dict] = {}
    metas: dict[str, dict] = {}

    oof, ext, meta = nested_topk_average(deep_xtr, ytr, deep_xte, deep_names)
    results["deep_topk_average"] = {"g1090_oof": oof, "sdata_external": ext}
    metas["deep_topk_average"] = meta

    oof, ext, meta = nested_topk_ridge(deep_xtr, ytr, deep_xte, deep_names, topk_grid=[3, 5, 10, 20, 40])
    results["deep_topk_ridge"] = {"g1090_oof": oof, "sdata_external": ext}
    metas["deep_topk_ridge"] = meta

    oof, ext, meta = nested_topk_ridge(raw_xtr, ytr, raw_xte, raw_names, topk_grid=[5, 10, 20, 40, 80])
    results["raw_profile_topk_ridge"] = {"g1090_oof": oof, "sdata_external": ext}
    metas["raw_profile_topk_ridge"] = meta

    # Benchmarks: fixed known interpretable features, no train selection except sign is left as originally defined.
    raw_scores = fixed_known_scores(pd.read_csv(DEEP_DIR / "g1090_aec128_deep_features_patient_level.csv"), pd.read_csv(DEEP_DIR / "sdata_aec128_deep_features_patient_level.csv"))
    for name, (tr_score, te_score) in raw_scores.items():
        results[name] = {"g1090_oof": tr_score, "sdata_external": te_score}
        metas[name] = {"definition": "fixed precomputed interpretable feature"}

    rows = []
    for name, scores in results.items():
        rows.append({"candidate": name, "split": "g1090_oof", **metric_row(name, ytr, scores["g1090_oof"])})
        rows.append({"candidate": name, "split": "sdata_external", **metric_row(name, yte, scores["sdata_external"])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "aec128_stronger_aec_only_metrics.csv", index=False)

    external_boot = []
    for name, scores in results.items():
        external_boot.append({"candidate": name, **bootstrap_auc(yte, scores["sdata_external"])})
    boot_df = pd.DataFrame(external_boot)
    boot_df.to_csv(OUT_DIR / "aec128_stronger_aec_only_external_auc_bootstrap.csv", index=False)

    for name, scores in results.items():
        pd.DataFrame({"low_smi": ytr, "aec_only_score": scores["g1090_oof"]}).to_csv(OUT_DIR / f"{name}_g1090_oof_scores.csv", index=False)
        pd.DataFrame({"low_smi": yte, "aec_only_score": scores["sdata_external"]}).to_csv(OUT_DIR / f"{name}_sdata_external_scores.csv", index=False)

    with open(OUT_DIR / "aec128_stronger_aec_only_summary.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics.to_dict(orient="records"), "bootstrap": external_boot, "metadata": metas}, f, ensure_ascii=False, indent=2)

    plot_scores(results, ytr, yte)

    print(metrics.sort_values(["split", "auc"], ascending=[True, False]).to_string(index=False))
    print("\nExternal AUC bootstrap")
    print(boot_df.sort_values("mean", ascending=False).to_string(index=False))
    print("\nSelected feature metadata")
    for key in ["deep_topk_average", "deep_topk_ridge", "raw_profile_topk_ridge"]:
        print(key, json.dumps({k: metas[key][k] for k in metas[key] if k.startswith("final_")}, ensure_ascii=False, indent=2)[:3000])
    print(OUT_DIR / "aec128_stronger_aec_only_metrics.csv")
    print(OUT_DIR / "aec128_stronger_aec_only_score_distributions.png")


if __name__ == "__main__":
    main()
