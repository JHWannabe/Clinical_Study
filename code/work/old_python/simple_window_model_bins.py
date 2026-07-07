from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import (  # noqa: E402
    DATA_DIR,
    binary_metrics,
    clinical_estimator,
    clinical_matrix,
    load_dataset,
    oof_and_external,
    threshold_youden,
    zfit_apply,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "simple_window_model"
SEED = 20260629


def candidate_windows() -> list[tuple[str, str, int, int]]:
    """a128/crop 곡선에서 폭 8~64의 겹치는 슬라이딩 윈도우 후보들을 (라벨, 곡선, 시작, 끝)으로 나열."""
    out = []
    for curve in ["a128", "crop"]:
        for width in [8, 16, 24, 32, 48, 64]:
            step = max(4, width // 2)
            for start0 in range(0, 128 - width + 1, step):
                out.append((f"{curve}_{start0+1}_{start0+width}", curve, start0 + 1, start0 + width))
    return out


def window_matrix(aec: np.ndarray, curve: str, start: int, end: int) -> np.ndarray:
    """결합 AEC 배열에서 지정한 곡선(a128/crop)의 start~end 구간만 잘라냄."""
    offset = 0 if curve == "a128" else 128
    return aec[:, offset + start - 1 : offset + end]


def shape_features(aec: np.ndarray, curve: str, start: int, end: int) -> np.ndarray:
    """윈도우 구간에서 평균/표준편차/최소·최대/기울기/전반부-후반부 차이/최댓값·최솟값 위치 등 12개 모양 특징을 계산."""
    mat = window_matrix(aec, curve, start, end)
    width = mat.shape[1]
    pos = np.linspace(-1.0, 1.0, width)
    denom = float(np.sum(pos**2)) or 1.0
    mean = mat.mean(axis=1)
    centered = mat - mean[:, None]
    slope = centered @ pos / denom
    half = max(1, width // 2)
    early = mat[:, :half].mean(axis=1)
    late = mat[:, half:].mean(axis=1)
    max_pos = np.argmax(mat, axis=1) / max(1, width - 1)
    min_pos = np.argmin(mat, axis=1) / max(1, width - 1)
    feats = np.column_stack(
        [
            mean,
            mat.std(axis=1),
            mat.min(axis=1),
            mat.max(axis=1),
            mat.max(axis=1) - mat.min(axis=1),
            slope,
            early,
            late,
            late - early,
            max_pos,
            min_pos,
            max_pos - min_pos,
        ]
    )
    return feats


def make_aec_model() -> Pipeline:
    """표준화 후 클래스 균형 로지스틱 회귀로 이어지는 윈도우 특징 전용 파이프라인 생성."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )


def aec_oof_external(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """폴드별로 학습해 train의 out-of-fold 점수를 만들고, 전체 train으로 재학습한 모델로 외부 점수도 반환."""
    oof = np.zeros(len(ytr), dtype=float)
    all_idx = np.arange(len(ytr))
    for val_idx in folds:
        tr_idx = np.setdiff1d(all_idx, val_idx)
        model = make_aec_model()
        model.fit(xtr[tr_idx], ytr[tr_idx])
        oof[val_idx] = model.decision_function(xtr[val_idx])
    final = make_aec_model()
    final.fit(xtr, ytr)
    return oof, final.decision_function(xte)


def lrt_add_score(y: np.ndarray, clinical_score: np.ndarray, add_score: np.ndarray) -> dict:
    """임상점수만 넣은 모델과 추가 점수까지 넣은 모델의 우도비검정(LRT) 카이제곱/p값/계수/오즈비를 계산."""
    c = (clinical_score - clinical_score.mean()) / (clinical_score.std() or 1.0)
    a = (add_score - add_score.mean()) / (add_score.std() or 1.0)
    x0 = sm.add_constant(c, has_constant="add")
    x1 = sm.add_constant(np.column_stack([c, a]), has_constant="add")
    m0 = sm.Logit(y, x0).fit(disp=False, maxiter=1000)
    m1 = sm.Logit(y, x1).fit(disp=False, maxiter=1000)
    chi2 = 2 * (m1.llf - m0.llf)
    return {
        "chi2": float(chi2),
        "p": float(stats.chi2.sf(chi2, 1)),
        "beta": float(m1.params[2]),
        "or_per_sd": float(np.exp(m1.params[2])),
    }


def stack_scores(ytr: np.ndarray, clinical_oof: np.ndarray, clinical_ext: np.ndarray, aec_oof: np.ndarray, aec_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """임상점수와 윈도우 AEC점수를 표준화한 뒤 로지스틱으로 결합(스택)해 train/외부 점수를 반환."""
    c_oof_z, c_ext_z, _, _ = zfit_apply(clinical_oof, clinical_ext)
    a_oof_z, a_ext_z, _, _ = zfit_apply(aec_oof, aec_ext)
    stack = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
    stack.fit(np.column_stack([c_oof_z, a_oof_z]), ytr)
    return (
        stack.decision_function(np.column_stack([c_oof_z, a_oof_z])),
        stack.decision_function(np.column_stack([c_ext_z, a_ext_z])),
    )


def evaluate_window(
    label: str,
    curve: str,
    start: int,
    end: int,
    train: dict,
    test: dict,
    ytr: np.ndarray,
    yte: np.ndarray,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    folds: list[np.ndarray],
) -> dict:
    """한 윈도우 후보에 대해 모양 특징 추출→OOF/외부 점수→임상점수와 스택→우도비검정까지 수행해 결과 딕셔너리로 반환."""
    xtr = shape_features(train["aec"], curve, start, end)
    xte = shape_features(test["aec"], curve, start, end)
    aec_oof, aec_ext = aec_oof_external(xtr, ytr, xte, folds)
    stack_oof, stack_ext = stack_scores(ytr, clinical_oof, clinical_ext, aec_oof, aec_ext)
    return {
        "window": label,
        "curve": curve,
        "start": start,
        "end": end,
        "width": end - start + 1,
        "train_aec_auc": float(roc_auc_score(ytr, aec_oof)),
        "external_aec_auc": float(roc_auc_score(yte, aec_ext)),
        "train_aec_ap": float(average_precision_score(ytr, aec_oof)),
        "external_aec_ap": float(average_precision_score(yte, aec_ext)),
        "train_lrt": lrt_add_score(ytr, clinical_oof, aec_oof),
        "external_lrt": lrt_add_score(yte, clinical_ext, aec_ext),
        "stack_train_auc": float(roc_auc_score(ytr, stack_oof)),
        "stack_external_auc": float(roc_auc_score(yte, stack_ext)),
        "stack_train_ap": float(average_precision_score(ytr, stack_oof)),
        "stack_external_ap": float(average_precision_score(yte, stack_ext)),
        "aec_oof": aec_oof,
        "aec_ext": aec_ext,
        "stack_oof": stack_oof,
        "stack_ext": stack_ext,
    }


def bin_accuracy_table(
    y: np.ndarray,
    clinical_z: np.ndarray,
    clinical_z_threshold: float,
    clinical_pred: np.ndarray,
    stack_pred: np.ndarray,
) -> pd.DataFrame:
    """임상점수와 임계값의 거리(margin)를 6개 구간으로 나눠, 구간별로 임상 단독 모델과 AEC 스택 모델의 정확도·FP·FN을 비교."""
    margin = clinical_z - clinical_z_threshold
    bins = [-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf]
    labels = ["<-1.0", "-1.0 to -0.5", "-0.5 to 0", "0 to 0.5", "0.5 to 1.0", ">=1.0"]
    cat = pd.cut(margin, bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        mask = cat == label
        n = int(mask.sum())
        if n == 0:
            continue
        yy = y[mask]
        cp = clinical_pred[mask]
        sp = stack_pred[mask]
        rows.append(
            {
                "clinical_score_margin_bin": label,
                "n": n,
                "events": int(yy.sum()),
                "event_rate": float(yy.mean()),
                "clinical_positive_n": int(cp.sum()),
                "aec_stack_positive_n": int(sp.sum()),
                "clinical_accuracy": float(np.mean(cp == yy)),
                "aec_stack_accuracy": float(np.mean(sp == yy)),
                "accuracy_delta": float(np.mean(sp == yy) - np.mean(cp == yy)),
                "clinical_fp": int(np.sum((cp == 1) & (yy == 0))),
                "aec_stack_fp": int(np.sum((sp == 1) & (yy == 0))),
                "clinical_fn": int(np.sum((cp == 0) & (yy == 1))),
                "aec_stack_fn": int(np.sum((sp == 0) & (yy == 1))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 손으로 설계한 12개 모양 특징만으로도, train 안에서
    "가장 좋아 보이는" AEC 윈도우를 골라 임상변수에 보탤 만한 것이 나오는가?):

    1. train(g1090)/test(sdata)를 로드하고 임상 단독 모델의 OOF/외부 점수와 AUC를 구한다.
    2. candidate_windows()로 만든 수백 개의 윈도우 후보 각각에 대해 evaluate_window로:
       - shape_features(12개 모양 특징: 평균/표준편차/기울기/전후반 차이/피크 위치 등)를 뽑고,
       - 로지스틱 회귀로 OOF/외부 점수를 만들고, 임상점수와 스택하고, LRT를 수행한다.
    3. 오직 train(OOF) 우도비 카이제곱이 가장 큰 윈도우 하나만 "선택"한다 (외부 데이터는 선택
       과정에 전혀 관여하지 않음 — train-only 윈도우 선택 규칙).
    4. 전체 스캔 결과와 상위 25개를 CSV로 저장.
    5. 선택된 윈도우의 스택 모델로 외부 데이터에서 임상 단독 대비 성능(AUC/AP/혼동행렬)을 비교하고,
       임상점수와 임계값의 거리 구간(margin bin)별로 임상 단독 vs 스택 모델의 정확도를
       bin_accuracy_table로 비교해 CSV로 저장 + 구간별 정확도 그래프를 PNG로 저장.
    6. 선택 규칙, 선택된 윈도우 정보, 모델 요약, 구간별 정확도표, 그리고 "이건 train 안에서
       많은 후보를 스캔한 것이므로 외부 데이터를 최종 검증으로 쓰려면 중첩선택이나 사전 지정
       윈도우가 필요하다"는 주의사항을 모두 JSON으로 저장한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]
    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    folds = [va for _, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(xclin_tr, ytr)]
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)
    clinical_auc = float(roc_auc_score(yte, clinical_ext))

    scan_rows = []
    best_payload = None
    for label, curve, start, end in candidate_windows():
        res = evaluate_window(label, curve, start, end, train, test, ytr, yte, clinical_oof, clinical_ext, folds)
        row = {k: v for k, v in res.items() if k not in {"aec_oof", "aec_ext", "stack_oof", "stack_ext", "train_lrt", "external_lrt"}}
        row.update(
            {
                "train_lrt_chi2": res["train_lrt"]["chi2"],
                "train_lrt_p": res["train_lrt"]["p"],
                "train_lrt_beta": res["train_lrt"]["beta"],
                "external_lrt_chi2": res["external_lrt"]["chi2"],
                "external_lrt_p": res["external_lrt"]["p"],
                "external_lrt_beta": res["external_lrt"]["beta"],
                "stack_external_delta_auc_vs_clinical": res["stack_external_auc"] - clinical_auc,
            }
        )
        scan_rows.append(row)
        if best_payload is None or row["train_lrt_chi2"] > best_payload["row"]["train_lrt_chi2"]:
            best_payload = {"row": row, "res": res}

    scan = pd.DataFrame(scan_rows).sort_values("train_lrt_chi2", ascending=False)
    scan.to_csv(OUT_DIR / "train_only_window_selection_scan.csv", index=False)
    scan.head(25).to_csv(OUT_DIR / "train_only_window_selection_top25.csv", index=False)

    selected = best_payload["res"]
    selected_row = best_payload["row"]
    clinical_th = threshold_youden(ytr, clinical_oof)
    stack_th = threshold_youden(ytr, selected["stack_oof"])
    clinical_pred = clinical_ext >= clinical_th
    stack_pred = selected["stack_ext"] >= stack_th
    clinical_z_train, clinical_z_ext, _, _ = zfit_apply(clinical_oof, clinical_ext)
    clinical_z_threshold = (clinical_th - clinical_oof.mean()) / clinical_oof.std()
    bins = bin_accuracy_table(yte, clinical_z_ext, clinical_z_threshold, clinical_pred, stack_pred)
    bins.to_csv(OUT_DIR / "external_clinical_score_bin_accuracy.csv", index=False)

    model_summary = pd.DataFrame(
        [
            {
                "model": "clinical_only",
                "external_auc": clinical_auc,
                "external_ap": float(average_precision_score(yte, clinical_ext)),
                **{f"external_{k}": v for k, v in binary_metrics(yte, clinical_ext, clinical_th).items()},
            },
            {
                "model": f"clinical_plus_simple_window_{selected['window']}",
                "external_auc": selected["stack_external_auc"],
                "external_ap": selected["stack_external_ap"],
                "external_delta_auc_vs_clinical": selected["stack_external_auc"] - clinical_auc,
                **{f"external_{k}": v for k, v in binary_metrics(yte, selected["stack_ext"], stack_th).items()},
            },
        ]
    )
    model_summary.to_csv(OUT_DIR / "selected_simple_window_model_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(bins))
    ax.plot(x, bins["clinical_accuracy"], marker="o", lw=2, label="Clinical-only")
    ax.plot(x, bins["aec_stack_accuracy"], marker="o", lw=2, label="Clinical + AEC window")
    ax.set_xticks(x)
    ax.set_xticklabels(bins["clinical_score_margin_bin"], rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Clinical score margin from train Youden threshold")
    ax.set_ylabel("External sdata accuracy")
    ax.set_title("Accuracy by clinical score interval")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "external_accuracy_by_clinical_score_bin.png", dpi=180)
    plt.close(fig)

    result = {
        "selection_rule": "Window selected using g1090 OOF largest likelihood-ratio chi-square for AEC score added to clinical score.",
        "selected_window": selected_row,
        "clinical_external_auc": clinical_auc,
        "model_summary": model_summary.to_dict(orient="records"),
        "external_clinical_score_bin_accuracy": bins.to_dict(orient="records"),
        "caveat": "Window selection scans many candidates on g1090; final inference still needs nested selection or a locked prespecified window before treating sdata as final validation.",
    }
    with open(OUT_DIR / "simple_window_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
