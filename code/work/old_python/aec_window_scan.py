from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    oof_and_external,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec_window_scan"
SEED = 20260629


def make_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """클래스 비율을 유지하며 데이터를 k개의 교차검증 폴드 인덱스로 분할."""
    rng = np.random.default_rng(SEED)
    folds = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % k].append(int(ix))
    return [np.array(sorted(f), dtype=int) for f in folds]


def make_estimator(n_features: int, c: float = 0.2) -> Pipeline:
    """결측대체→표준화→상위 32개 특징 선택→선형 SVM으로 이어지는 소규모 윈도우 전용 분류 파이프라인 생성."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=min(32, n_features))),
            ("svm", LinearSVC(C=c, class_weight="balanced", max_iter=10000, random_state=SEED)),
        ]
    )


def score_estimator(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """학습된 파이프라인의 decision_function 점수를 반환."""
    return model.decision_function(x)


def window_oof_external(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """폴드별로 학습해 train의 out-of-fold 점수를 만들고, 전체 train으로 재학습한 모델로 외부 데이터 점수도 함께 반환."""
    oof = np.zeros(len(ytr), dtype=float)
    all_idx = np.arange(len(ytr))
    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = make_estimator(xtr.shape[1])
        model.fit(xtr[tr_idx], ytr[tr_idx])
        oof[val_idx] = score_estimator(model, xtr[val_idx])
    final = make_estimator(xtr.shape[1])
    final.fit(xtr, ytr)
    return oof, score_estimator(final, xte)


def zscore_train_apply(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차로 train·test를 함께 z-표준화."""
    mu = float(np.mean(tr))
    sd = float(np.std(tr))
    if sd == 0 or not np.isfinite(sd):
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def lrt_add_score(y: np.ndarray, clinical_score: np.ndarray, add_score: np.ndarray) -> tuple[float, float, float]:
    """임상점수만 넣은 모델과 윈도우 점수까지 넣은 모델의 우도비검정(LRT) 카이제곱/p값/계수를 계산 (실패 시 NaN)."""
    c = (clinical_score - clinical_score.mean()) / (clinical_score.std() or 1.0)
    a = (add_score - add_score.mean()) / (add_score.std() or 1.0)
    x0 = sm.add_constant(c, has_constant="add")
    x1 = sm.add_constant(np.column_stack([c, a]), has_constant="add")
    try:
        m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
        m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
        chi2 = 2 * (m1.llf - m0.llf)
        return float(chi2), float(stats.chi2.sf(chi2, 1)), float(m1.params[2])
    except Exception:
        return np.nan, np.nan, np.nan


def stack_auc(ytr: np.ndarray, yte: np.ndarray, c_tr: np.ndarray, c_te: np.ndarray, a_tr: np.ndarray, a_te: np.ndarray) -> tuple[float, float]:
    """임상점수와 윈도우 AEC점수를 표준화해 로지스틱으로 결합(스택)한 뒤, train/외부 AUC를 계산."""
    ctr, cte = zscore_train_apply(c_tr, c_te)
    atr, ate = zscore_train_apply(a_tr, a_te)
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
    model.fit(np.column_stack([ctr, atr]), ytr)
    tr_score = model.decision_function(np.column_stack([ctr, atr]))
    te_score = model.decision_function(np.column_stack([cte, ate]))
    return float(roc_auc_score(ytr, tr_score)), float(roc_auc_score(yte, te_score))


def windows() -> list[tuple[str, int, int, np.ndarray]]:
    """a128/crop 곡선 각각에서 폭 8~64, 겹치는 슬라이딩 윈도우 구간들과, a128-crop 같은 위치를 짝지은 윈도우까지 모두 나열해 (곡선명, 시작, 끝, 인덱스) 목록을 생성."""
    out = []
    for curve, offset in [("a128", 0), ("crop", 128)]:
        for width in [8, 16, 24, 32, 48, 64]:
            step = max(4, width // 2)
            for start in range(0, 128 - width + 1, step):
                idx = np.arange(offset + start, offset + start + width)
                out.append((curve, start + 1, start + width, idx))
    for start in range(0, 128 - 16 + 1, 8):
        idx = np.r_[np.arange(start, start + 16), np.arange(128 + start, 128 + start + 16)]
        out.append(("paired_a128_crop", start + 1, start + 16, idx))
    return out


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: AEC 곡선의 어느 위치/폭 구간이 임상변수 대비 추가
    정보를 주는 후보인지, 슬라이딩 윈도우로 총점검):

    1. train(g1090)/test(sdata)를 로드하고 임상 단독 모델의 train OOF/외부 AUC를 기준선으로 계산.
    2. windows()로 만든 수백 개의 (곡선, 시작-끝 위치, 폭) 윈도우 후보 각각에 대해:
       - 그 윈도우 구간만 잘라 SVM 파이프라인(window_oof_external)으로 학습해 train OOF/외부 AUC·AP를 구하고,
       - lrt_add_score로 임상점수에 그 윈도우 점수를 추가했을 때의 우도비검정(LRT) 결과를,
       - stack_auc로 임상점수+윈도우점수를 로지스틱으로 결합했을 때 임상 단독 대비 AUC 개선폭을 계산.
    3. 모든 윈도우 결과를 표로 모아, train과 외부 양쪽에서 AUC 개선이 있고 방향(계수 부호)이
       일치하는(direction_concordant) 윈도우에 가산점을 주는 rank_score로 정렬한다.
    4. 전체 결과와 상위 25개 윈도우를 각각 CSV로 저장하고, "이건 탐색적 스캔이므로 최종 검증 전에
       윈도우를 g1090 안에서 미리 고정해야 한다"는 주의사항과 함께 요약 JSON을 저장한다.
    5. 폭 16 윈도우에 대해 위치별 delta AUC(train vs 외부)를 a128/crop 곡선별로 그려 PNG로 저장한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]
    folds = make_folds(ytr, 5)

    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    clinical_oof, clinical_test = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)
    clinical_auc_train = roc_auc_score(ytr, clinical_oof)
    clinical_auc_external = roc_auc_score(yte, clinical_test)

    rows = []
    for curve, start, end, idx in windows():
        xtr = train["aec"][:, idx]
        xte = test["aec"][:, idx]
        a_oof, a_test = window_oof_external(xtr, ytr, xte, folds)
        train_auc = roc_auc_score(ytr, a_oof)
        external_auc = roc_auc_score(yte, a_test)
        train_ap = average_precision_score(ytr, a_oof)
        external_ap = average_precision_score(yte, a_test)
        train_chi2, train_p, train_beta = lrt_add_score(ytr, clinical_oof, a_oof)
        ext_chi2, ext_p, ext_beta = lrt_add_score(yte, clinical_test, a_test)
        stack_train_auc, stack_external_auc = stack_auc(ytr, yte, clinical_oof, clinical_test, a_oof, a_test)
        rows.append(
            {
                "curve": curve,
                "start": start,
                "end": end,
                "width": end - start + 1,
                "n_features": len(idx),
                "train_aec_auc": train_auc,
                "external_aec_auc": external_auc,
                "train_aec_ap": train_ap,
                "external_aec_ap": external_ap,
                "train_lrt_chi2_add_to_clinical": train_chi2,
                "train_lrt_p_add_to_clinical": train_p,
                "train_beta_add_to_clinical": train_beta,
                "external_lrt_chi2_add_to_clinical": ext_chi2,
                "external_lrt_p_add_to_clinical": ext_p,
                "external_beta_add_to_clinical": ext_beta,
                "stack_train_auc": stack_train_auc,
                "stack_external_auc": stack_external_auc,
                "stack_train_delta_auc_vs_clinical": stack_train_auc - clinical_auc_train,
                "stack_external_delta_auc_vs_clinical": stack_external_auc - clinical_auc_external,
            }
        )
    df = pd.DataFrame(rows)
    df["direction_concordant"] = np.sign(df["train_beta_add_to_clinical"]) == np.sign(df["external_beta_add_to_clinical"])
    df["rank_score"] = (
        df["stack_train_delta_auc_vs_clinical"].clip(lower=-0.01)
        + df["stack_external_delta_auc_vs_clinical"].clip(lower=-0.02)
        + (df["direction_concordant"].astype(float) * 0.01)
    )
    df = df.sort_values("rank_score", ascending=False)
    df.to_csv(OUT_DIR / "aec_window_scan_results.csv", index=False)

    top = df.head(25).copy()
    top.to_csv(OUT_DIR / "aec_window_scan_top25.csv", index=False)

    summary = {
        "clinical_auc_train_oof": float(clinical_auc_train),
        "clinical_auc_external": float(clinical_auc_external),
        "n_windows": int(len(df)),
        "top_windows": top.to_dict(orient="records"),
        "notes": "Discovery scan only. Window choice must be locked in g1090 or nested CV before using sdata as final validation.",
    }
    with open(OUT_DIR / "aec_window_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_df = df[df["curve"].isin(["a128", "crop"])].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, curve in zip(axes, ["a128", "crop"]):
        sub = plot_df[(plot_df["curve"] == curve) & (plot_df["width"] == 16)].sort_values("start")
        ax.plot(sub["start"], sub["stack_train_delta_auc_vs_clinical"], marker="o", lw=1.8, label="g1090 OOF")
        ax.plot(sub["start"], sub["stack_external_delta_auc_vs_clinical"], marker="o", lw=1.8, label="sdata external")
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_title(f"{curve} 16-position window: clinical + window-AEC stack delta AUC")
        ax.set_ylabel("Delta AUC vs clinical")
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Window start position")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "window16_delta_auc_by_position.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
