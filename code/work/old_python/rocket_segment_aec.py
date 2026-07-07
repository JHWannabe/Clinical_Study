from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
from rocket_aec_incremental import (  # noqa: E402
    RocketTransformer,
    bootstrap_delta,
    cross_val_scores,
    fit_predict_external,
    logit_lrt,
    score_metrics,
    tune_c,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "rocket_segment_aec"
SEED = 20260629


def segment_series(aec: np.ndarray, start: int, end: int, curve: str = "a128") -> np.ndarray:
    """Return shape n x 1 x width. Positions are 1-based inclusive."""
    offset = 0 if curve == "a128" else 128
    mat = aec[:, offset + start - 1 : offset + end]
    return mat[:, None, :].astype(np.float32)


def stack_scores(
    ytr: np.ndarray,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    aec_oof: np.ndarray,
    aec_ext: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """임상점수와 세그먼트 ROCKET 점수를 표준화한 뒤 로지스틱으로 결합(스택)해 train/외부 점수를 반환."""
    c_oof_z, c_ext_z, _, _ = zfit_apply(clinical_oof, clinical_ext)
    a_oof_z, a_ext_z, _, _ = zfit_apply(aec_oof, aec_ext)
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
    model.fit(np.column_stack([c_oof_z, a_oof_z]), ytr)
    return (
        model.decision_function(np.column_stack([c_oof_z, a_oof_z])),
        model.decision_function(np.column_stack([c_ext_z, a_ext_z])),
    )


def run_segment(
    name: str,
    train_aec: np.ndarray,
    test_aec: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    clinical_oof: np.ndarray,
    clinical_ext: np.ndarray,
    curve: str,
    start: int,
    end: int,
    n_kernels: int = 1500,
) -> dict:
    """지정한 (곡선, 시작-끝) 구간만 잘라 ROCKET 특징을 뽑고, C 튜닝→OOF/외부 예측→임상점수와의 스택→
    조건부 LRT→임상양성군 내 위험 10분위표까지 한 세그먼트에 대한 전체 분석을 수행해 결과 딕셔너리로 반환."""
    xtr_series = segment_series(train_aec, start, end, curve)
    xte_series = segment_series(test_aec, start, end, curve)

    rocket = RocketTransformer(n_kernels=n_kernels, seed=SEED + start + end)
    xtr = rocket.fit_transform(xtr_series)
    xte = rocket.transform(xte_series)

    c_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    best_c, cv = tune_c(xtr, ytr, c_grid, f"rocket_{name}")
    cv.to_csv(OUT_DIR / f"{name}_cv_C.csv", index=False)

    aec_oof, aec_oof_prob = cross_val_scores(xtr, ytr, best_c)
    aec_ext, aec_ext_prob = fit_predict_external(xtr, ytr, xte, best_c)
    stack_oof, stack_ext = stack_scores(ytr, clinical_oof, clinical_ext, aec_oof, aec_ext)

    clinical_th = threshold_youden(ytr, clinical_oof)
    stack_th = threshold_youden(ytr, stack_oof)

    cond_train = logit_lrt(ytr, clinical_oof, aec_oof)
    cond_external = logit_lrt(yte, clinical_ext, aec_ext)
    delta_stack = bootstrap_delta(yte, clinical_ext, stack_ext, n_boot=1500)

    decile_rows = []
    clinical_pos = clinical_ext >= clinical_th
    df = pd.DataFrame({"y": yte[clinical_pos], "aec_score": aec_ext[clinical_pos]})
    df["decile"] = pd.qcut(df["aec_score"], 10, labels=False, duplicates="drop") + 1
    for decile, g in df.groupby("decile", observed=True):
        decile_rows.append(
            {
                "segment": name,
                "decile": int(decile),
                "n": int(len(g)),
                "events": int(g["y"].sum()),
                "prevalence": float(g["y"].mean()),
            }
        )

    return {
        "segment": name,
        "curve": curve,
        "start": start,
        "end": end,
        "width": end - start + 1,
        "n_kernels": n_kernels,
        "n_rocket_features": int(xtr.shape[1]),
        "best_C": best_c,
        "rocket_aec_only": {
            "train_oof": score_metrics(ytr, aec_oof, aec_oof_prob),
            "external": score_metrics(yte, aec_ext, aec_ext_prob),
        },
        "clinical_plus_segment_rocket_stack": {
            "train_oof_auc": float(roc_auc_score(ytr, stack_oof)),
            "train_oof_average_precision": float(average_precision_score(ytr, stack_oof)),
            "external_auc": float(roc_auc_score(yte, stack_ext)),
            "external_average_precision": float(average_precision_score(yte, stack_ext)),
            "external_binary_at_train_youden": binary_metrics(yte, stack_ext, stack_th),
            "bootstrap_delta_vs_clinical": delta_stack,
        },
        "conditional_lrt": {
            "train_oof": cond_train,
            "external": cond_external,
        },
        "clinical_positive_deciles": decile_rows,
    }


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 이전 window scan에서 유망해 보였던 특정 AEC 구간들만
    골라, ROCKET 특징으로 더 깊게 파봤을 때도 임상변수 대비 이득이 유지되는가?):

    1. train(g1090)/test(sdata)를 로드하고 임상 단독 모델의 OOF/외부 점수를 구한다.
    2. 사전에 정해둔 6개 후보 구간(a128의 65-96/65-104/73-104/81-112/65-128, crop의 33-64)에
       대해 run_segment를 반복 실행한다. run_segment는 각 구간마다:
       - 해당 구간만 잘라 RocketTransformer로 1500개 커널 특징을 뽑고,
       - tune_c로 정규화 강도 C를 교차검증으로 고르고,
       - out-of-fold/외부 점수를 구해 임상점수와 스택하고,
       - 우도비검정(LRT)과 부트스트랩 델타 AUC, 임상양성군 내 위험 10분위표까지 계산한다.
    3. 6개 구간의 결과를 한 표로 모아 "임상 단독 대비 외부 delta AUC"가 큰 순으로 정렬해 CSV로 저장.
    4. 각 구간의 임상양성군 10분위 위험표를 모아 별도 CSV로 저장.
    5. "이 구간들은 이전 스캔 결과에 이끌려(motivated) 고른 것이므로, 최종 검증 전에는 train 내
       중첩 선택으로 미리 고정되어야 한다"는 주의사항과 함께 전체 결과를 JSON으로 저장.
    6. 구간별 외부 delta AUC를 막대그래프로 그려 PNG로 저장하고, 요약 표와 저장 경로를 출력한다.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    ytr = train["y"]
    yte = test["y"]

    xclin_tr, xclin_te, _ = clinical_matrix(train["meta"], test["meta"])
    folds = [va for _, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(xclin_tr, ytr)]
    clinical_oof, clinical_ext = oof_and_external(lambda seed: clinical_estimator(), xclin_tr, ytr, xclin_te, folds)

    candidates = [
        ("a128_65_96", "a128", 65, 96),
        ("a128_65_104", "a128", 65, 104),
        ("a128_73_104", "a128", 73, 104),
        ("a128_81_112", "a128", 81, 112),
        ("a128_65_128", "a128", 65, 128),
        ("crop_33_64", "crop", 33, 64),
    ]

    results = []
    for name, curve, start, end in candidates:
        print(f"Running {name}", flush=True)
        results.append(run_segment(name, train["aec"], test["aec"], ytr, yte, clinical_oof, clinical_ext, curve, start, end))

    rows = []
    deciles = []
    clinical_auc = float(roc_auc_score(yte, clinical_ext))
    for r in results:
        rows.append(
            {
                "segment": r["segment"],
                "curve": r["curve"],
                "start": r["start"],
                "end": r["end"],
                "width": r["width"],
                "best_C": r["best_C"],
                "aec_train_auc": r["rocket_aec_only"]["train_oof"]["auc"],
                "aec_external_auc": r["rocket_aec_only"]["external"]["auc"],
                "stack_train_auc": r["clinical_plus_segment_rocket_stack"]["train_oof_auc"],
                "stack_external_auc": r["clinical_plus_segment_rocket_stack"]["external_auc"],
                "stack_external_delta_auc_vs_clinical": r["clinical_plus_segment_rocket_stack"]["external_auc"] - clinical_auc,
                "external_lrt_p": r["conditional_lrt"]["external"]["lrt_p"],
                "external_lrt_beta": r["conditional_lrt"]["external"]["add_score_beta"],
                "train_lrt_p": r["conditional_lrt"]["train_oof"]["lrt_p"],
                "train_lrt_beta": r["conditional_lrt"]["train_oof"]["add_score_beta"],
                "delta_auc_boot_mean": r["clinical_plus_segment_rocket_stack"]["bootstrap_delta_vs_clinical"]["delta_auc"]["mean"],
                "delta_auc_boot_ci2.5": r["clinical_plus_segment_rocket_stack"]["bootstrap_delta_vs_clinical"]["delta_auc"]["ci2.5"],
                "delta_auc_boot_ci97.5": r["clinical_plus_segment_rocket_stack"]["bootstrap_delta_vs_clinical"]["delta_auc"]["ci97.5"],
            }
        )
        deciles.extend(r["clinical_positive_deciles"])

    summary_df = pd.DataFrame(rows).sort_values("stack_external_delta_auc_vs_clinical", ascending=False)
    summary_df.to_csv(OUT_DIR / "rocket_segment_summary_table.csv", index=False)
    pd.DataFrame(deciles).to_csv(OUT_DIR / "rocket_segment_clinical_positive_deciles.csv", index=False)

    result = {
        "clinical_external_auc": clinical_auc,
        "interpretation": "Exploratory: segments were motivated by prior window scan; must be locked by train-only nested selection before final validation.",
        "segments": results,
        "summary_table": summary_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "rocket_segment_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["stack_external_delta_auc_vs_clinical"], color="#2F6F73")
    ax.axhline(0, color="#777777", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["segment"], rotation=35, ha="right")
    ax.set_ylabel("External delta AUC vs clinical")
    ax.set_title("ROCKET on selected AEC segments")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rocket_segment_delta_auc.png", dpi=180)
    plt.close(fig)

    print(summary_df.to_string(index=False))
    print(OUT_DIR / "rocket_segment_summary.json")


if __name__ == "__main__":
    main()
