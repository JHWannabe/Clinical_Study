from __future__ import annotations

# gangnam_final_dataset.xlsx의 clinic4(성별/나이/신장/체중)만으로 HTN/DM/CKD 각각 독립된 로지스틱
# 회귀 모델을 만든다. 질병마다 양성 100% + 음성은 양성 개수만큼 무작위 언더샘플링(1:1 balanced)한 뒤,
# 그 질병 라벨로 stratify한 patient-level train:valid:test=7:1:2 split을 따로 뽑는다(prevalence가
# 질병마다 달라 split을 공유하면 다른 질병의 stratify가 깨짐). 외부검증(frozen 모델, scaler는 gangnam에서
# fit)은 원본 prevalence 그대로의 sinchon/new10000에 적용하고, 해당 코호트에 없는 질병 컬럼(new10000의
# CKD)은 자동으로 skip한다.

import sys
import time
from pathlib import Path
from typing import cast
from zipfile import BadZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트(없으면 그래프 한글 라벨이 깨짐)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_XLSX = DATA_DIR / "gangnam_final_dataset.xlsx"
EXTERNAL_COHORTS = {  # name -> xlsx path. 코호트에 없는 질병 컬럼은 자동 skip
    "sinchon": DATA_DIR / "sinchon_final_dataset.xlsx",
    "new10000": DATA_DIR / "new10000_final_dataset.xlsx",
}
CLINIC4_DIR = PROJECT_ROOT / "outputs" / "clinic4"

CLINICAL_COLS = ["PatientAge", "Height", "Weight"]  # clinic4의 성별 제외 나머지 3개(표준화 대상)
DISEASES = ["HTN", "DM", "CKD"]
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
SEED = 20260709
PARAM_GRID = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"], "solver": ["liblinear"],
              "class_weight": [None, "balanced"]}


# 소수점 3자리로 표현하되 그보다 작은 값(0.001 미만)은 3자리에서 0으로 뭉개지므로 지수 표기로 전환
def f3(x: float | np.floating) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.3e}" if x != 0 and abs(x) < 1e-3 else f"{x:.3f}"


# DataFrame의 float 컬럼(정수/문자열 컬럼 제외)에 f3 포맷을 적용한 사본 반환(csv 저장/출력용)
def format_floats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].map(f3)
    return out


# 모델(baseline/aec_vat/...)마다 같은 파일명(coefficients.csv 등)으로 흩어져 저장되던 것을 CLINIC4_DIR
# 루트의 xlsx 한 파일 + 모델별 시트로 모은다. 이미 있는 파일이면 해당 시트만 교체하고 나머지는 보존.
# ponytail: 이 폴더가 OneDrive 동기화 대상이라 직후 재오픈 시 간헐적으로 BadZipFile이 남(동기화 락 추정) -
# 짧은 재시도로 우회. 더 잦아지면 로컬 임시경로에서 쓰고 마지막에 복사하는 방식으로 바꿀 것
def save_sheet(df: pd.DataFrame, xlsx_name: str, sheet_name: str, retries: int = 5) -> None:
    xlsx_path = CLINIC4_DIR / xlsx_name
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            mode = "a" if xlsx_path.exists() else "w"
            kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
            with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode=mode, **kwargs) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            return
        except BadZipFile:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    valid = df[CLINICAL_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if (~valid).any():
        print(f"[{path.stem}] clinic4 결측 {(~valid).sum()}/{len(valid)}명 제외")
    return df[valid].reset_index(drop=True)


# 성별(M=1/F=0) + 표준화된 나이/신장/체중으로 입력 행렬 구성. scaler는 gangnam에서 fit한 것을
# 외부 코호트에는 frozen으로 넘겨받아 transform만 함(재학습 없음)
def clinical_matrix(df: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    sex_m = (df["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float)
    rest = df[CLINICAL_COLS].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


DEFAULT_TERMS = ["intercept", "sex_M", "age", "height", "weight"]  # x 컬럼 순서(clinical_matrix)와 일치해야 함


# 정규화 없는 statsmodels Logit으로 coefficient/odds ratio/Wald p-value 산출(Wald 검정은 MLE 전제라
# sklearn LogisticRegression의 L2 정규화 계수로는 성립하지 않음) - final_model과 같은 train+valid로 적합.
# terms는 x 컬럼(=clinical_matrix가 만든 순서)에 맞는 이름 목록(입력 변수를 늘린 파생 스크립트가 오버라이드)
def coefficient_table(disease: str, x: np.ndarray, y: np.ndarray, terms: list[str] = DEFAULT_TERMS) -> pd.DataFrame:
    model = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    return pd.DataFrame({"disease": disease, "term": terms, "coefficient": model.params,
                        "std_err": model.bse, "p_value": model.pvalues, "odds_ratio": np.exp(model.params)})


# 예측변수(sex_M/age/height/weight)간 다중공선성 체크. 질병(y)과 무관하게 x 자체의 성질이라 gangnam
# 전체 표본 1개로 산출(질병별 idx_trainval은 행만 다르고 열은 동일해 결과 차이가 미미함)
def vif_table(x: np.ndarray, terms: list[str] = DEFAULT_TERMS) -> pd.DataFrame:
    x_const = sm.add_constant(x)
    vif = [variance_inflation_factor(x_const, i) for i in range(x_const.shape[1])]
    return pd.DataFrame({"term": terms, "VIF": vif})


# 양성 인덱스 100% + 음성은 양성 개수만큼 rng로 무작위 비복원추출한 인덱스(1:1 balanced) 반환
def balanced_idx(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = rng.choice(np.flatnonzero(y == 0), size=len(pos_idx), replace=False)
    return np.concatenate([pos_idx, neg_idx])


# ROC 위에서 Youden index(J = sensitivity + specificity - 1)가 최대인 지점의 확률 cutoff.
# sklearn predict()의 0.5 고정 cutoff는 balance=False(원본 prevalence) 코호트에서 예측 확률이 전부 0.5
# 아래에 몰려 전원 음성으로 찍히고 sensitivity가 0에 수렴하므로 쓰지 않는다
def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, score)
    return float(thr[1 + int(np.argmax((tpr - fpr)[1:]))])  # thr[0]은 sklearn이 넣는 inf(전원 음성) 지점이라 제외


# 주어진 cutoff로 confusion matrix를 만들어 sensitivity/specificity 계산
def sens_spec(y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    tn, fp, fn, tp = confusion_matrix(y, (score >= threshold).astype(int), labels=[0, 1]).ravel()
    return tp / (tp + fn), tn / (tn + fp)


# 질병 하나에 대해 그 질병 라벨로 stratify한 7:1:2 split → LogisticRegression 학습 → valid/test AUC 산출.
# 외부검증용 frozen 모델은 train+valid(test 미포함)로 재학습해 해당 질병 컬럼이 있는 external 코호트에 적용.
# ROC curve plot용으로 cohort별 (y, score)도 함께 반환
def run_disease(disease: str, x: np.ndarray, df: pd.DataFrame,
                 externals: dict[str, tuple[np.ndarray, pd.DataFrame]], terms: list[str] = DEFAULT_TERMS,
                 balance: bool = True) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    # balance=True: 양성 100% + 음성은 양성 개수만큼 무작위 언더샘플링(1:1 balanced) → 이하 split/학습은 이 부분집합으로 진행
    # balance=False: 원본 prevalence 그대로 사용(undersampling 없음)
    rng = np.random.default_rng(SEED)
    y_full = df[disease].to_numpy(dtype=int)
    pid_full = df["PatientID"].to_numpy()
    if balance:
        idx_full = balanced_idx(y_full, rng)
        x, y, pid = x[idx_full], y_full[idx_full], pid_full[idx_full]
    else:
        x, y, pid = x, y_full, pid_full

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=TEST_RATIO, random_state=SEED, stratify=y)
    idx_train, idx_valid = train_test_split(
        idx_trainval, test_size=VALID_RATIO / (TRAIN_RATIO + VALID_RATIO), random_state=SEED, stratify=y[idx_trainval]
    )

    # valid를 하나짜리 CV fold로 넘겨 GridSearchCV가 기존 train/valid split 그대로 하이퍼파라미터를 고르게 함
    # (k-fold를 새로 만들면 소수 질환은 fold가 너무 작아져 불안정 - new10000_auc_ceiling 메모 참고)
    fold = np.where(np.isin(idx_trainval, idx_valid), 0, -1)
    grid = GridSearchCV(LogisticRegression(max_iter=2000), PARAM_GRID, scoring="roc_auc",
                        cv=PredefinedSplit(fold)).fit(x[idx_trainval], y[idx_trainval])
    best_params = grid.best_params_

    model = LogisticRegression(max_iter=2000, **best_params).fit(x[idx_train], y[idx_train])
    valid_score = model.predict_proba(x[idx_valid])[:, 1]
    valid_auc = roc_auc_score(y[idx_valid], valid_score)
    test_score = model.predict_proba(x[idx_test])[:, 1]
    test_auc = roc_auc_score(y[idx_test], test_score)

    # cutoff는 test를 보지 않고 valid(hyperparameter 고를 때 쓴 held-out)에서 고정
    threshold = youden_threshold(y[idx_valid], valid_score)
    test_sens, test_spec = sens_spec(y[idx_test], test_score, threshold)
    print(f"[{disease}] gangnam n_train={len(idx_train)} n_valid={len(idx_valid)} n_test={len(idx_test)} "
        f"valid_prev={f3(y[idx_valid].mean())} test_prev={f3(y[idx_test].mean())} "
        f"best_params={best_params} valid_AUC={f3(valid_auc)} test_AUC={f3(test_auc)} "
        f"thr={f3(threshold)} sens={f3(test_sens)} spec={f3(test_spec)}")
    rows = [{"disease": disease, "cohort": "gangnam", "n": len(idx_test), "prevalence": float(y[idx_test].mean()),
            "prevalence_train": float(y[idx_train].mean()), "prevalence_valid": float(y[idx_valid].mean()),
            "valid_auc": float(valid_auc), "auc": float(test_auc),
            "sensitivity": float(test_sens), "specificity": float(test_spec), "threshold": float(threshold),
            "best_C": best_params["C"], "best_penalty": best_params["penalty"],
            "best_class_weight": best_params["class_weight"]}]
    pred_rows = [pd.DataFrame({"disease": disease, "cohort": "gangnam", "patient_id": pid[idx_test],
                               "y": y[idx_test], "score": test_score})]

    final_model = LogisticRegression(max_iter=2000, **best_params).fit(x[idx_trainval], y[idx_trainval])
    # frozen 모델의 cutoff도 개발 코호트(gangnam train+valid)에서 한 번 고정해 외부 코호트에 그대로 적용
    # (코호트마다 다시 고르면 그 코호트 정답을 보고 튜닝하는 셈이라 낙관적 편향)
    ext_threshold = youden_threshold(y[idx_trainval], final_model.predict_proba(x[idx_trainval])[:, 1])
    for cohort, (x_ext, df_ext) in externals.items():
        if disease not in df_ext.columns:
            print(f"[{disease}] {cohort} SKIP: 해당 질병 컬럼 없음")
            continue
        # gangnam과 동일하게 balance 여부를 맞춰서 평가(True면 1:1 balanced 부분집합, False면 원본 그대로)
        y_ext_full = df_ext[disease].to_numpy(dtype=int)
        pid_ext_full = df_ext["PatientID"].to_numpy()
        if balance:
            ext_idx = balanced_idx(y_ext_full, rng)
            x_ext_b, y_ext, pid_ext = x_ext[ext_idx], y_ext_full[ext_idx], pid_ext_full[ext_idx]
        else:
            x_ext_b, y_ext, pid_ext = x_ext, y_ext_full, pid_ext_full
        ext_score = final_model.predict_proba(x_ext_b)[:, 1]
        ext_auc = roc_auc_score(y_ext, ext_score)
        ext_sens, ext_spec = sens_spec(y_ext, ext_score, ext_threshold)
        print(f"[{disease}] {cohort} n={len(y_ext)} prevalence={f3(y_ext.mean())} AUC={f3(ext_auc)} "
            f"thr={f3(ext_threshold)} sens={f3(ext_sens)} spec={f3(ext_spec)}")
        rows.append({"disease": disease, "cohort": cohort, "n": len(y_ext), "prevalence": float(y_ext.mean()),
                    "prevalence_train": float("nan"), "prevalence_valid": float("nan"), "valid_auc": float("nan"),
                    "auc": float(ext_auc), "sensitivity": float(ext_sens), "specificity": float(ext_spec),
                    "threshold": float(ext_threshold)})
        pred_rows.append(pd.DataFrame({"disease": disease, "cohort": cohort, "patient_id": pid_ext,
                                       "y": y_ext, "score": ext_score}))

    coef_df = coefficient_table(disease, x[idx_trainval], y[idx_trainval], terms=terms)
    return rows, coef_df, pd.concat(pred_rows, ignore_index=True)


# predictions(cohort별 y/score)와 summary(auc)로 질병별 ROC curve 1장(질병당 subplot, 코호트별 선)을 그려 저장
def save_roc_curve_plot(predictions: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(5 * len(DISEASES), 5))
    for ax, disease in zip(axes, DISEASES):
        pred_d = predictions[predictions["disease"] == disease]
        summary_d = summary[summary["disease"] == disease].set_index("cohort")
        for cohort in pred_d["cohort"].unique():
            pred_c = pred_d[pred_d["cohort"] == cohort]
            fpr, tpr, _ = roc_curve(pred_c["y"].to_numpy(), pred_c["score"].to_numpy())
            auc = cast(float, summary_d.loc[cohort, "auc"])
            ax.plot(fpr, tpr, label=f"{cohort} (AUC={f3(auc)})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(disease)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


UNBALANCED_SUFFIX = "_unbalanced"  # 언더샘플링 없이(원본 유병률) 돌린 결과 시트에 붙는 꼬리표


# 질병 3개를 학습/외부검증하고 결과를 clinic4 루트 xlsx의 sheet 하나 + ROC 그림으로 저장한다.
# balance=False면 언더샘플링 없이 원본 유병률로 돌리고 시트명에 _unbalanced를 붙여 따로 남긴다.
# 파생 스크립트(aec/체성분)들이 전부 이 꼬리 블록을 복붙하고 있어 한 군데로 모았다.
def run_and_save(sheet: str, x: np.ndarray, df: pd.DataFrame,
                 externals: dict[str, tuple[np.ndarray, pd.DataFrame]], terms: list[str] = DEFAULT_TERMS,
                 balance: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not balance:
        sheet += UNBALANCED_SUFFIX
    print(f"=== {sheet} (balance={balance}) ===")
    results = [run_disease(disease, x, df, externals, terms=terms, balance=balance) for disease in DISEASES]
    rows = [row for disease_rows, _, _ in results for row in disease_rows]
    coef_df = pd.concat([coef for _, coef, _ in results], ignore_index=True)
    predictions = pd.concat([pred for _, _, pred in results], ignore_index=True)
    summary = pd.DataFrame(rows)

    vif_df = vif_table(x, terms=terms)
    print(format_floats(vif_df).to_string(index=False))

    save_sheet(format_floats(summary), "performance_summary.xlsx", sheet)
    save_sheet(format_floats(coef_df), "coefficients.xlsx", sheet)
    save_sheet(format_floats(vif_df), "vif.xlsx", sheet)
    save_sheet(predictions, "predictions.xlsx", sheet)
    # roc_curve_*.png는 AEC 포함 여부로 폴더를 나눠 저장(roc_curve/aec, roc_curve/no_aec)
    roc_dir = CLINIC4_DIR / "roc_curve" / ("aec" if "aec" in sheet.lower() else "no_aec")
    roc_dir.mkdir(parents=True, exist_ok=True)
    roc_path = roc_dir / f"roc_curve_{sheet}.png"
    save_roc_curve_plot(predictions, summary, roc_path)
    print(f"Saved sheet '{sheet}' to performance_summary.xlsx, coefficients.xlsx, vif.xlsx, predictions.xlsx, "
          f"{roc_path.relative_to(CLINIC4_DIR)}")

    # self-check: cutoff가 전원 음성으로 찍으면 sensitivity가 0으로 붕괴(0.5 고정 cutoff 시절의 회귀 방지)
    assert (summary["sensitivity"] > 0).all(), f"[{sheet}] sensitivity=0 - cutoff가 전원 음성으로 찍고 있음"
    if balance:
        # self-check: 1:1 언더샘플링 후 stratified split이 train/valid/test 모두 50% prevalence를 보존하는지
        gangnam = summary[summary["cohort"] == "gangnam"]
        for _, row in gangnam.iterrows():
            for prev in (row["prevalence_train"], row["prevalence_valid"], row["prevalence"]):
                assert abs(prev - 0.5) < 0.05, f"{row['disease']} split not stratified (balanced prevalence drifted)"
        print(f"OK: [{sheet}] stratified split preserves 1:1 balanced prevalence within 5pp")
    return summary, coef_df


def main(balance: bool = True) -> None:
    df = load_data(DATA_XLSX)
    x, scaler = clinical_matrix(df)

    externals: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for cohort, path in EXTERNAL_COHORTS.items():
        df_ext = load_data(path)
        x_ext, _ = clinical_matrix(df_ext, scaler=scaler)
        externals[cohort] = (x_ext, df_ext)

    run_and_save("baseline", x, df, externals, balance=balance)


if __name__ == "__main__":
    main(balance=True)
    main(balance=False)
