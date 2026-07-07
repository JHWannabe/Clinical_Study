from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aec_conditional_value import DATA_DIR, binary_metrics, load_dataset  # noqa: E402
from aec_offset_score import (  # noqa: E402
    bootstrap_delta,
    clinical_raw,
    crossfit_offset_score,
    lrt_score_test,
    metric_row,
    sigmoid,
    train_external_offset_score,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_feature_combo_models"
FEATURE_DIR = Path(__file__).resolve().parent / "outputs" / "0629" / "aec128_deep_feature_mining"
SEED = 20260629
LAMBDAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


FEATURE_SETS = {
    "edge_only": [
        "haar_haar_l5_b12_right_minus_left",
    ],
    "regional_rebound_only": [
        "haar_haar_l2_b02_right_minus_left",
    ],
    "cyl_visual_rebound": [
        "cyl_cyl_late_positive_plus_mid_negative",
        "visual_aec128_rebound_height_peak_minus_valley",
        "visual_aec128_visual_rebound_score",
    ],
    "compact_shape_combo": [
        "haar_haar_l5_b12_right_minus_left",
        "haar_haar_l2_b02_right_minus_left",
        "cyl_cyl_late_positive_plus_mid_negative",
        "cyl_cyl_mean_upstroke_78_110",
        "visual_aec128_rebound_height_peak_minus_valley",
    ],
    "atlas_plus_shape_combo": [
        "atlas_log_mid_late_score",
        "haar_haar_l5_b12_right_minus_left",
        "cyl_cyl_late_positive_plus_mid_negative",
        "visual_aec128_rebound_height_peak_minus_valley",
    ],
}


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """목표 민감도(target) 이상을 유지하는 가장 높은(엄격한) 임계값을 찾음 (해당하는 값이 없으면 분위수로 근사)."""
    best = None
    for th in np.unique(score):
        pred = score >= th
        tp = np.sum(pred & (y == 1))
        fn = np.sum(~pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        tn = np.sum(~pred & (y == 0))
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        row = (float(th), sens, spec)
        if sens >= target and (best is None or th > best[0]):
            best = row
    if best is None:
        return float(np.quantile(score[y == 1], 1 - target))
    return float(best[0])


def threshold_youden_local(y: np.ndarray, score: np.ndarray) -> float:
    """Youden index(민감도+특이도-1)를 최대화하는 임계값을 후보 점수들 중에서 탐색."""
    best_th = float(np.min(score))
    best_j = -np.inf
    for th in np.unique(score):
        m = binary_metrics(y, score, float(th))
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_j = j
            best_th = float(th)
    return best_th


def load_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """run_aec128_deep_feature_mining.py가 만들어둔 환자별 특징 CSV(g1090/sdata)를 읽어옴."""
    tr = pd.read_csv(FEATURE_DIR / "g1090_aec128_deep_features_patient_level.csv")
    te = pd.read_csv(FEATURE_DIR / "sdata_aec128_deep_features_patient_level.csv")
    return tr, te


def standardize_apply(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train의 평균/표준편차(결측 제외)로 train·test를 함께 표준화하고, 결측은 train 평균으로 대체."""
    mu = np.nanmean(xtr, axis=0)
    sd = np.nanstd(xtr, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    xtr2 = np.where(np.isfinite(xtr), xtr, mu)
    xte2 = np.where(np.isfinite(xte), xte, mu)
    return (xtr2 - mu) / sd, (xte2 - mu) / sd


def choose_c_for_aec_only(x: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    """5-fold 교차검증으로 여러 정규화 강도 C 후보 중 로그손실 기준 최적값을 선택 (AEC 특징 조합 단독 모델용)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for c in C_GRID:
        score = np.zeros(len(y), dtype=float)
        prob = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in skf.split(x, y):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logit", LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED)),
                ]
            )
            pipe.fit(x[tr_idx], y[tr_idx])
            score[va_idx] = pipe.decision_function(x[va_idx])
            prob[va_idx] = pipe.predict_proba(x[va_idx])[:, 1]
        rows.append(
            {
                "C": c,
                "cv_auc": float(roc_auc_score(y, score)),
                "cv_average_precision": float(average_precision_score(y, score)),
                "cv_log_loss": float(log_loss(y, prob)),
                "cv_brier": float(brier_score_loss(y, prob)),
            }
        )
    df = pd.DataFrame(rows)
    best = df.sort_values(["cv_log_loss", "cv_brier"], ascending=True).iloc[0]
    return float(best["C"]), df


def crossfit_aec_only(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    """고정된 C값으로 5-fold 교차검증을 돌려 train 전체에 대한 out-of-fold 점수를 만듦."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 10)
    score = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(x, y):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED)),
            ]
        )
        pipe.fit(x[tr_idx], y[tr_idx])
        score[va_idx] = pipe.decision_function(x[va_idx])
    return score


def fit_external_aec_only(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, c: float) -> np.ndarray:
    """train 전체로 모델을 학습해 외부(test) 데이터에 대한 점수를 예측."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=SEED)),
        ]
    )
    pipe.fit(xtr, ytr)
    return pipe.decision_function(xte)


def deescalation_summary(
    y: np.ndarray,
    clinical_score: np.ndarray,
    aec_score: np.ndarray,
    clinical_th: float,
    aec_th: float,
) -> dict:
    """임상 양성군을 AEC 임계값 기준으로 "유지"와 "하향조정"으로 나눠, 각 그룹 사건율과 Fisher 정확검정 결과를 계산."""
    clinical_pos = clinical_score >= clinical_th
    aec_deesc = aec_score < aec_th
    kept = clinical_pos & ~aec_deesc
    deesc = clinical_pos & aec_deesc
    table = np.array(
        [
            [int(np.sum(y[kept])), int(np.sum(kept) - np.sum(y[kept]))],
            [int(np.sum(y[deesc])), int(np.sum(deesc) - np.sum(y[deesc]))],
        ]
    )
    fisher_or, fisher_p = stats.fisher_exact(table)

    def row(mask: np.ndarray) -> dict:
        n = int(np.sum(mask))
        e = int(np.sum(y[mask]))
        return {"n": n, "events": e, "event_rate": float(e / n) if n else np.nan}

    return {
        "aec_threshold": float(aec_th),
        "clinical_positive_n": int(np.sum(clinical_pos)),
        "clinical_positive_events": int(np.sum(y[clinical_pos])),
        "clinical_positive_event_rate": float(np.mean(y[clinical_pos])) if np.any(clinical_pos) else np.nan,
        "clinical_positive_aec_kept": row(kept),
        "clinical_positive_aec_deescalated": row(deesc),
        "fisher_or_kept_vs_deescalated": float(fisher_or),
        "fisher_p": float(fisher_p),
    }


def evaluate_feature_set(
    name: str,
    cols: list[str],
    ftr: pd.DataFrame,
    fte: pd.DataFrame,
    train: dict,
    test: dict,
) -> dict:
    """한 특징 조합(cols)에 대해 (a) 특징만 쓰는 단독 로지스틱 모델의 OOF/외부 성능, (b) 임상 오프셋
    릿지로 결합한 모델의 성능·조건부 LRT·부트스트랩 델타, (c) 임상 90% 민감도 양성군 내에서 AEC
    하위 10%를 "하향조정"으로 보는 재분류 분석까지 모두 수행해 결과 딕셔너리로 반환."""
    ytr = train["y"].astype(int)
    yte = test["y"].astype(int)
    xtr = ftr[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xte = fte[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xtr_s, xte_s = standardize_apply(xtr, xte)

    best_c, cv_df = choose_c_for_aec_only(xtr_s, ytr)
    aec_oof = crossfit_aec_only(xtr_s, ytr, best_c)
    aec_ext = fit_external_aec_only(xtr_s, ytr, xte_s, best_c)

    clinical_xtr = clinical_raw(train["meta"])
    clinical_xte = clinical_raw(test["meta"])
    clinical_oof, aec_offset_oof, combined_oof, fold_df = crossfit_offset_score(xtr, ytr, clinical_xtr, LAMBDAS)
    clinical_tr_full, clinical_ext, aec_offset_tr, aec_offset_ext, best_lam, lambda_cv = train_external_offset_score(
        xtr,
        ytr,
        clinical_xtr,
        xte,
        clinical_xte,
        LAMBDAS,
    )
    combined_ext = clinical_ext + aec_offset_ext

    clinical_th = threshold_youden_local(ytr, clinical_oof)
    combined_th = threshold_youden_local(ytr, combined_oof)
    clinical_90_th = threshold_for_min_sensitivity(ytr, clinical_oof, 0.90)
    aec_lowrisk_th = np.quantile(aec_oof[clinical_oof >= clinical_90_th], 0.10)

    result = {
        "feature_set": name,
        "features": cols,
        "n_features": len(cols),
        "aec_only": {
            "selected_C": best_c,
            "train_oof": metric_row(f"{name}_aec_only_oof", ytr, aec_oof),
            "external": metric_row(f"{name}_aec_only_external", yte, aec_ext),
        },
        "clinical_plus_aec_offset": {
            "selected_lambda": best_lam,
            "outer_fold_lambdas": fold_df.to_dict(orient="records"),
            "train_oof_clinical": metric_row("clinical_oof", ytr, clinical_oof, clinical_th),
            "train_oof_combined": metric_row(f"{name}_combined_oof", ytr, combined_oof, combined_th),
            "external_clinical": metric_row("clinical_external", yte, clinical_ext, clinical_th),
            "external_combined": metric_row(f"{name}_combined_external", yte, combined_ext, combined_th),
            "train_lrt_aec_added_to_clinical": lrt_score_test(ytr, clinical_oof, aec_offset_oof),
            "external_lrt_aec_added_to_clinical": lrt_score_test(yte, clinical_ext, aec_offset_ext),
            "external_delta_combined_vs_clinical": bootstrap_delta(yte, clinical_ext, combined_ext, n_boot=2500),
        },
        "deescalation_at_clinical90": {
            "clinical90_threshold": float(clinical_90_th),
            "aec_lowrisk_rule": "Among g1090 clinical-positive patients at 90% clinical sensitivity, AEC-low is bottom 10% of AEC combo score.",
            "aec_lowrisk_threshold": float(aec_lowrisk_th),
            "train": deescalation_summary(ytr, clinical_oof, aec_oof, clinical_90_th, aec_lowrisk_th),
            "external": deescalation_summary(yte, clinical_ext, aec_ext, clinical_90_th, aec_lowrisk_th),
        },
    }
    cv_df.to_csv(OUT_DIR / f"{name}_aec_only_c_grid.csv", index=False)
    lambda_cv.to_csv(OUT_DIR / f"{name}_offset_lambda_grid.csv", index=False)
    fold_df.to_csv(OUT_DIR / f"{name}_offset_outer_lambdas.csv", index=False)
    return result


def flatten_results(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """여러 특징 조합의 결과 딕셔너리들을 성능표/하향조정표/LRT표 세 개의 평평한 DataFrame으로 펼침."""
    perf_rows = []
    deesc_rows = []
    lrt_rows = []
    for r in results:
        fs = r["feature_set"]
        perf_rows.append({"feature_set": fs, "block": "aec_only_train_oof", **r["aec_only"]["train_oof"]})
        perf_rows.append({"feature_set": fs, "block": "aec_only_external", **r["aec_only"]["external"]})
        perf_rows.append({"feature_set": fs, "block": "clinical_external", **r["clinical_plus_aec_offset"]["external_clinical"]})
        perf_rows.append({"feature_set": fs, "block": "combined_external", **r["clinical_plus_aec_offset"]["external_combined"]})
        perf_rows.append({"feature_set": fs, "block": "clinical_train_oof", **r["clinical_plus_aec_offset"]["train_oof_clinical"]})
        perf_rows.append({"feature_set": fs, "block": "combined_train_oof", **r["clinical_plus_aec_offset"]["train_oof_combined"]})
        for split in ["train", "external"]:
            deesc = r["deescalation_at_clinical90"][split]
            for group_name in ["clinical_positive_aec_kept", "clinical_positive_aec_deescalated"]:
                deesc_rows.append(
                    {
                        "feature_set": fs,
                        "split": split,
                        "group": group_name,
                        **deesc[group_name],
                        "clinical_positive_n": deesc["clinical_positive_n"],
                        "clinical_positive_event_rate": deesc["clinical_positive_event_rate"],
                        "fisher_p": deesc["fisher_p"],
                    }
                )
        lrt_rows.append({"feature_set": fs, "split": "train", **r["clinical_plus_aec_offset"]["train_lrt_aec_added_to_clinical"]})
        lrt_rows.append({"feature_set": fs, "split": "external", **r["clinical_plus_aec_offset"]["external_lrt_aec_added_to_clinical"]})
    return pd.DataFrame(perf_rows), pd.DataFrame(deesc_rows), pd.DataFrame(lrt_rows)


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: 앞서 발견한 개별 특징들을 몇 개씩 묶은 "조합"으로 쓰면
    단일 특징보다 더 나은 임상 활용 모델이 되는가?):

    1. run_aec128_deep_feature_mining이 만든 환자별 특징 CSV를 로드하고, g1090/sdata 원본
       데이터셋도 로드한다.
    2. 미리 정의된 5개 특징 조합(edge_only, regional_rebound_only, cyl_visual_rebound,
       compact_shape_combo, atlas_plus_shape_combo)마다 evaluate_feature_set을 실행:
       - 그 조합만으로 로지스틱 회귀(C 교차검증)를 학습해 AEC 단독 성능을 구하고,
       - 임상 변수를 오프셋으로 고정한 릿지 결합 모델의 성능과 조건부 LRT를 구하고,
       - 임상 90% 민감도 양성군에서 AEC 조합 점수 하위 10%를 "하향조정"으로 재분류했을 때
         유지군 vs 하향조정군의 사건율 차이(Fisher 검정)를 계산한다.
    3. 5개 조합의 결과를 성능표/하향조정표/LRT표로 펼쳐 각각 CSV로 저장.
    4. 특징 조합 정의와 전체 결과를 JSON으로 저장하고, 외부 성능 요약과 하향조정 결과를 콘솔에 출력.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ftr, fte = load_feature_tables()
    train = load_dataset(DATA_DIR / "g1090.xlsx")
    test = load_dataset(DATA_DIR / "sdata.xlsx")
    results = []
    for name, cols in FEATURE_SETS.items():
        missing = [c for c in cols if c not in ftr.columns or c not in fte.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        print(f"Running {name}: {len(cols)} features")
        results.append(evaluate_feature_set(name, cols, ftr, fte, train, test))

    perf_df, deesc_df, lrt_df = flatten_results(results)
    perf_df.to_csv(OUT_DIR / "aec128_feature_combo_performance.csv", index=False)
    deesc_df.to_csv(OUT_DIR / "aec128_feature_combo_deescalation_clinical90.csv", index=False)
    lrt_df.to_csv(OUT_DIR / "aec128_feature_combo_lrt.csv", index=False)
    with open(OUT_DIR / "aec128_feature_combo_summary.json", "w", encoding="utf-8") as f:
        json.dump({"feature_sets": FEATURE_SETS, "results": results}, f, ensure_ascii=False, indent=2)

    summary = perf_df[
        (perf_df["block"].isin(["aec_only_external", "clinical_external", "combined_external"]))
    ][["feature_set", "block", "auc", "average_precision", "log_loss", "brier", "sensitivity", "specificity", "ppv", "npv"]]
    print(summary.to_string(index=False))
    print("\nDe-escalation external")
    print(
        deesc_df[(deesc_df["split"] == "external")].to_string(index=False)
    )
    print(OUT_DIR / "aec128_feature_combo_performance.csv")
    print(OUT_DIR / "aec128_feature_combo_deescalation_clinical90.csv")


if __name__ == "__main__":
    main()
