from __future__ import annotations

# data/gangnam_final_dataset.xlsx(internal)의 metadata 시트로 clinic4(성별/나이/신장/체중) baseline
# logistic regression을 질병 3종(HTN/DM/CKD)에 대해 각각 독립적으로 만든다. internal은 질병별 양성
# prevalence가 낮아(class imbalance) 음성군을 양성군 수만큼 랜덤 언더샘플링해 1:1로 맞춘 뒤([[new10000_auc_ceiling]]:
# 언더샘플링도 AUC엔 큰 효과 없음을 확인했지만 성능 확인 목적으로 채택), train:valid:test=7:1:2로
# stratified split한다. C는 valid AUC를 최대화하는 값으로 고르고, threshold는 valid ROC에서 Youden's J를
# 최대화하는 지점으로 정한다. acc/sens/spec/auc는 test에서만 산출(held-out). coefficient/p-value는
# 정규화가 없는 MLE라야 Wald 검정이 성립하므로 statsmodels Logit(unregularized)을 train+valid에 적합해 산출한다.
# 외부검증에 적용할 frozen 모델은 train+valid(test 미포함)로 재학습하고(scaler는 언더샘플링 전 전체 internal에서
# fit), threshold도 internal valid Youden 값을 그대로 씀. AUC는 bootstrap 95% CI를 낸다. 외부검증 코호트는 2개:
# sinchon(HTN/DM/CKD 모두 존재), new10000(CKD 컬럼 자체가 없어 HTN/DM만 - [[new10000_auc_ceiling]] 참고)

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트(없으면 그래프 한글 라벨이 깨짐)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1002" / "clinic4_baseline"

METADATA_SHEET = "metadata"
INTERNAL_NAME = "internal"
INTERNAL_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_COHORTS = {  # name -> xlsx path. 코호트별로 없는 질병 컬럼은 자동으로 skip됨
    "external_sinchon": DATA_DIR / "sinchon_final_dataset.xlsx",
    "external_new10000": DATA_DIR / "new10000_final_dataset.xlsx",
}
N_BOOT = 2000
SEED = 20260709
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]  # clinic4의 성별 제외 나머지 3개(표준화 대상)
C_GRID = np.logspace(-4, 4, 17)  # AUC 최대화 대상 하이퍼파라미터 grid
DISEASES = ["HTN", "DM", "CKD"]
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
COHORT_COLORS = {INTERNAL_NAME: "#0072B2", "external_sinchon": "#E69F00", "external_new10000": "#009E73"}  # Okabe-Ito


# metadata 시트를 로드하고 성별이 M/F가 아니거나 clinic4 입력(나이/키/체중)이 결측인 행을 제외
def load_cohort(path: Path) -> pd.DataFrame:
    meta = pd.read_excel(path, sheet_name=METADATA_SHEET, engine="openpyxl").reset_index(drop=True)
    valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"])
    valid_clinic = meta[CLINICAL_BASE_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    mask = valid_sex & valid_clinic
    print(f"[{path.stem}] 성별 이상/clinic4 결측 제외: {(~mask).sum()}/{len(mask)}명")
    return meta[mask].reset_index(drop=True)


# 성별(M=1/F=0) + 표준화된 나이/신장/체중(+extra_cols가 있으면 그 스칼라 컬럼도 같이 표준화)으로 입력
# 행렬을 구성. scaler는 internal에서 fit한 것을 external에는 frozen으로 넘겨받아 transform만 함(재학습 없음).
# extra_cols는 clinic4_aec_compare.py에서 AEC Upper/Lower ratio 같은 스칼라 파생변수를 추가할 때 재사용
def clinical_matrix(meta: pd.DataFrame, extra_cols: list[str] | None = None,
                     scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[CLINICAL_BASE_COLS + (extra_cols or [])].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    sex_m = (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


# AUC의 bootstrap 95% CI(환자 단위 복원추출, external frozen 평가에만 적용)
def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for idx in rng.integers(0, n, size=(n_boot, n)):
        if len(np.unique(y[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


# OOF ROC에서 Youden's J(sensitivity+specificity-1)를 최대화하는 threshold
def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


# 확률 점수와 고정 threshold로 sensitivity/specificity/accuracy 산출
def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
             "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
             "accuracy": float((tp + tn) / len(y))}


# 정규화 없는 statsmodels Logit으로 coefficient/odds ratio/Wald p-value 산출(Wald 검정은 MLE 전제라 규제항 없는 모델 필요)
# extra_terms는 x의 clinic4(5열) 뒤에 추가로 붙은 열들의 이름(예: AEC 관련 파생변수)
def coefficient_table(x: np.ndarray, y: np.ndarray, extra_terms: list[str] | None = None) -> pd.DataFrame:
    model = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    terms = ["intercept", "sex_M", "age", "height", "weight"] + (extra_terms or [])
    return pd.DataFrame({
        "term": terms,
        "coefficient": model.params,
        "std_err": model.bse,
        "p_value": model.pvalues,
        "odds_ratio": np.exp(model.params),
    })


# 음성군(y==0)에서 양성군 수만큼 랜덤 언더샘플링한 행 인덱스(양성 전부 + 샘플된 음성) - internal의
# class imbalance를 1:1로 맞추기 위함([[new10000_auc_ceiling]]: AUC 자체엔 효과가 작다고 확인됨)
def balanced_index(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = rng.choice(np.flatnonzero(y == 0), size=len(pos_idx), replace=False)
    return np.sort(np.concatenate([pos_idx, neg_idx]))


# 질병 하나에 대해 internal을 음성군 언더샘플링으로 1:1 맞춘 뒤 train:valid:test=7:1:2로 stratified
# split한다. valid AUC로 C를 고르고 valid Youden으로 threshold를 정한 뒤, test에서 held-out 성능을
# 산출(long-format 1행). 외부검증용 frozen 모델은 train+valid(test 미포함)로 재학습해 각 external
# 코호트(해당 질병 컬럼이 있는 것만)에 적용
def run_disease(disease: str, x_int: np.ndarray, meta_int: pd.DataFrame,
                 externals: dict[str, tuple[np.ndarray, pd.DataFrame]],
                 extra_terms: list[str] | None = None
                 ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    y_full = meta_int[disease].to_numpy(dtype=int)
    bal_idx = balanced_index(y_full, seed=SEED)
    x_bal, y = x_int[bal_idx], y_full[bal_idx]
    patient_id_bal = meta_int["PatientID"].to_numpy()[bal_idx]

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=TEST_RATIO, random_state=SEED, stratify=y)
    idx_train, idx_valid = train_test_split(idx_trainval, test_size=VALID_RATIO / (TRAIN_RATIO + VALID_RATIO),
                                             random_state=SEED, stratify=y[idx_trainval])

    best_c, best_valid_auc = C_GRID[0], -1.0
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=2000).fit(x_bal[idx_train], y[idx_train])
        valid_auc = roc_auc_score(y[idx_valid], model.predict_proba(x_bal[idx_valid])[:, 1])
        if valid_auc > best_valid_auc:
            best_c, best_valid_auc = c, valid_auc

    train_model = LogisticRegression(C=best_c, max_iter=2000).fit(x_bal[idx_train], y[idx_train])
    valid_proba = train_model.predict_proba(x_bal[idx_valid])[:, 1]
    threshold = youden_threshold(y[idx_valid], valid_proba)

    test_proba = train_model.predict_proba(x_bal[idx_test])[:, 1]
    test_auc = float(roc_auc_score(y[idx_test], test_proba))
    cls_stats = classification_stats(y[idx_test], test_proba, threshold)

    rows = [{"disease": disease, "cohort": INTERNAL_NAME, "n": int(len(idx_test)),
             "n_pos": int(y[idx_test].sum()), "prevalence": float(y[idx_test].mean()), "best_C": float(best_c),
             "auc_mean": test_auc, "auc_std": float("nan"), "auc_ci_lower": float("nan"),
             "auc_ci_upper": float("nan"), "threshold": threshold, **cls_stats}]
    print(f"[{disease}] {INTERNAL_NAME}(test) n={len(idx_test)} n_pos={int(y[idx_test].sum())} best_C={best_c:.4g} "
          f"AUC={test_auc:.3f} Acc={cls_stats['accuracy']:.3f} "
          f"Se={cls_stats['sensitivity']:.3f} Sp={cls_stats['specificity']:.3f}")

    pred_rows = [pd.DataFrame({"disease": disease, "cohort": INTERNAL_NAME,
                                "patient_id": patient_id_bal[idx_test],
                                "y": y[idx_test], "score": test_proba, "threshold": threshold})]

    # 외부검증용 frozen 모델은 train+valid(test 미포함)로 재학습(이 모델이 external에 적용되는 최종 모델)
    final_model = LogisticRegression(C=best_c, max_iter=2000).fit(x_bal[idx_trainval], y[idx_trainval])

    for cohort_name, (x_ext, meta_ext) in externals.items():
        if disease not in meta_ext.columns:
            print(f"[{disease}] {cohort_name} SKIP: 해당 질병 컬럼 없음")
            continue
        y_ext = meta_ext[disease].to_numpy(dtype=int)
        ext_proba = final_model.predict_proba(x_ext)[:, 1]
        ext_auc = float(roc_auc_score(y_ext, ext_proba))
        ci_lo, ci_hi = bootstrap_auc_ci(y_ext, ext_proba)
        ext_cls_stats = classification_stats(y_ext, ext_proba, threshold)

        rows.append({"disease": disease, "cohort": cohort_name, "n": int(len(y_ext)), "n_pos": int(y_ext.sum()),
                     "prevalence": float(y_ext.mean()), "best_C": float(best_c), "auc_mean": ext_auc,
                     "auc_std": float("nan"), "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi,
                     "threshold": threshold, **ext_cls_stats})
        print(f"[{disease}] {cohort_name} n={len(y_ext)} n_pos={int(y_ext.sum())} "
              f"AUC={ext_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] Acc={ext_cls_stats['accuracy']:.3f} "
              f"Se={ext_cls_stats['sensitivity']:.3f} Sp={ext_cls_stats['specificity']:.3f}")

        pred_rows.append(pd.DataFrame({"disease": disease, "cohort": cohort_name,
                                        "patient_id": meta_ext["PatientID"].to_numpy(),
                                        "y": y_ext, "score": ext_proba, "threshold": threshold}))

    # coefficient/p-value는 정규화 없는 MLE(Wald 검정 전제) - final_model과 같은 train+valid로 적합
    coef_df = coefficient_table(x_bal[idx_trainval], y[idx_trainval], extra_terms=extra_terms)
    coef_df.insert(0, "disease", disease)

    predictions = pd.concat(pred_rows, ignore_index=True)
    return rows, coef_df, predictions


# predictions(코호트별 y/score)와 performance_summary(auc_mean)로 질병별 ROC curve 1장(질병당 subplot
# 1개, 코호트별 선 하나씩)을 그려 저장. internal은 test held-out AUC, external은 frozen 평가 AUC
def save_roc_curve_plot(predictions: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(5 * len(DISEASES), 5))
    for ax, disease in zip(axes, DISEASES):
        pred_d = predictions[predictions["disease"] == disease]
        summary_d = summary[summary["disease"] == disease].set_index("cohort")
        for cohort in pred_d["cohort"].unique():
            pred_c = pred_d[pred_d["cohort"] == cohort]
            fpr, tpr, _ = roc_curve(pred_c["y"].to_numpy(), pred_c["score"].to_numpy())
            auc = summary_d.loc[cohort, "auc_mean"]
            ax.plot(fpr, tpr, color=COHORT_COLORS.get(cohort), label=f"{cohort} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(disease)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_int = load_cohort(INTERNAL_XLSX)
    x_int, scaler = clinical_matrix(meta_int)

    externals: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name, path in EXTERNAL_COHORTS.items():
        meta_ext = load_cohort(path)
        x_ext, _ = clinical_matrix(meta_ext, scaler=scaler)
        externals[name] = (x_ext, meta_ext)

    # 질병마다 독립된 모델 1개씩, 총 3개(HTN/DM/CKD 모델 파라미터·undersampling·split이 서로 완전히 분리됨 -
    # run_disease 호출마다 그 질병의 y로 balanced_index를 새로 뽑고 C/threshold도 그 질병 것만으로 정함)
    rows_htn, coef_htn, pred_htn = run_disease("HTN", x_int, meta_int, externals)
    rows_dm, coef_dm, pred_dm = run_disease("DM", x_int, meta_int, externals)
    rows_ckd, coef_ckd, pred_ckd = run_disease("CKD", x_int, meta_int, externals)

    summary_rows = rows_htn + rows_dm + rows_ckd
    coef_tables = [coef_htn, coef_dm, coef_ckd]
    pred_tables = [pred_htn, pred_dm, pred_ckd]

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.concat(pred_tables, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    pd.concat(coef_tables, ignore_index=True).round(6).to_csv(OUTPUT_DIR / "coefficients.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    save_roc_curve_plot(predictions_df, summary_df, OUTPUT_DIR / "roc_curve_all.png")
    print(f"Saved performance_summary.csv, coefficients.csv, predictions.csv, roc_curve_all.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
