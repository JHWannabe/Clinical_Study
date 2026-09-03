from __future__ import annotations

# data/aec_cropped.xlsx의 metadata_cleaned 시트(verify_clean_aec_metadata.py가 생성, 10,578명)를 써서
# clinic4(성별/나이/신장/체중)만으로 질병 6종(당뇨병/고혈압/이상지질혈증/골다공증/심근경색/뇌졸중) 유무를
# 예측하는 logistic regression. 질병예측/step3_clinic_aec_disease_logistic.py와 달리 이 추적관찰 코호트는
# 단일 소스(internal/external 분리 없음)이므로 FPCA/DeLong/스캐너 로직 없이 clinic4 단일 모델만 internal
# 5-fold OOF로 학습/평가한다. 사용자 확인(2026-08-19): aec_128 데이터가 있는 환자만 필터링해서 진행 -
# aec_128_contrast 시트(조영 CT 시리즈별 128포인트 raw, PatientID 기준 중복 가능)에 존재하는 환자로 한정한다
# (filter_aec128_contrast_by_metadata.py와 동일한 교집합 로직, 9,914명).

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "04_followup_cohort" / "clinic4"

AEC_XLSX = DATA_DIR / "aec_cropped.xlsx"
METADATA_SHEET = "metadata_cleaned"
AEC_CONTRAST_SHEET = "aec_128_contrast"
N_FOLDS = 5
SEED = 20260709
CLINICAL_BASE_COLS = ["나이", "신장", "체중"]  # clinic4의 성별 제외 나머지 3개(표준화 대상)
MIN_POSITIVES = 2  # 이 미만이면 ROC/logistic 자체가 정의되지 않아 해당 질병을 skip

# 질병 6종(metadata_cleaned에 이미 0/1로 존재, 통합 문서.xlsx 진단 시트 재계산값) -> 파일명에 쓸 slug
DISEASES: dict[str, str] = {
    "당뇨병_여부": "dm",
    "고혈압_여부": "htn",
    "이상지질혈증_여부": "dyslipidemia",
    "골다공증_여부": "osteoporosis",
    "심근경색_여부": "mi",
    "뇌졸중_여부": "stroke",
}


# aec_cropped.xlsx의 metadata_cleaned 시트를 로드하고, 성별이 M/F가 아니거나 clinic4 입력이 결측인 행,
# aec_128_contrast 시트(조영 CT 128포인트 raw)에 없는 환자를 제외
def load_cohort() -> pd.DataFrame:
    meta = pd.read_excel(AEC_XLSX, sheet_name=METADATA_SHEET, engine="openpyxl").reset_index(drop=True)
    valid_sex = meta["성별"].astype(str).str.upper().isin(["M", "F"])
    valid_clinic = meta[CLINICAL_BASE_COLS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    mask = valid_sex & valid_clinic
    print(f"clinic4 입력 결측/성별 이상 제외: {(~mask).sum()}/{len(mask)}명")
    meta = meta[mask].reset_index(drop=True)

    aec_ids = set(pd.read_excel(AEC_XLSX, sheet_name=AEC_CONTRAST_SHEET, usecols=["PatientID"],
                                 engine="openpyxl")["PatientID"].astype(int))
    mask_aec = meta["patientID"].astype(int).isin(aec_ids)
    print(f"aec_128 데이터 없는 환자 제외: {(~mask_aec).sum()}/{len(mask_aec)}명 "
          f"-> {AEC_CONTRAST_SHEET} 교집합 {int(mask_aec.sum())}명")
    return meta[mask_aec].reset_index(drop=True)


# 성별(M=1/F=0) + 표준화된 나이/신장/체중으로 clinic4 입력 행렬을 구성
def clinical_matrix(meta: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    rest = meta[CLINICAL_BASE_COLS].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    sex_m = (meta["성별"].astype(str).str.upper().to_numpy() == "M").astype(float)
    return np.column_stack([sex_m, scaler.transform(rest)]), scaler


# 질병 유병 여부 클래스 균형에 맞춘 StratifiedKFold fold 수(최소 2, 최대 N_FOLDS)
def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


# OOF ROC에서 Youden's J(sensitivity+specificity-1)를 최대화하는 threshold
def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


# 확률 점수와 고정 threshold로 sensitivity/specificity/accuracy 산출
def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


# 질병별 핵심 통계를 한 줄로 출력
def _log(disease: str, s: dict) -> None:
    print(f"[{disease}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
          f"AUC={s['auc']:.3f} Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} Acc={s['accuracy']:.3f}")


# 기존 파일에서 다른 스크립트가 쓴 시트는 보존한 채, 이 스크립트가 소유한 시트만 추가/교체 저장
def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


# 질병 6개의 OOF ROC curve를 한 그림에 겹쳐 그림
def plot_roc_all(curves: dict[str, dict[str, np.ndarray]], stats_by_disease: dict[str, dict],
                  out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))

    fig, ax = plt.subplots(figsize=(9, 9))
    for (disease, c), color in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(c["y"], c["score"])
        s = stats_by_disease[disease]
        ax.plot(fpr, tpr, color=color, linewidth=2.4, label=f"{disease} AUC={s['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_title("clinic4(성별/나이/신장/체중) 질병 예측 ROC (추적관찰 코호트, internal OOF)",
                 fontsize=18, fontweight="bold", color=INK_PRIMARY)
    ax.set_xlabel("1 - Specificity", fontsize=15)
    ax.set_ylabel("Sensitivity", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, loc="lower right", frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


# 질병 6개의 AUC를 막대그래프로 비교
def plot_auc_summary(summary: pd.DataFrame, out_path: Path) -> None:
    INK_PRIMARY = "#161616"
    diseases = [d for d in DISEASES if d in summary["disease"].unique()]
    slugs = [DISEASES[d] for d in diseases]
    rows = summary.set_index("disease").reindex(diseases)
    x = np.arange(len(diseases))

    fig, ax = plt.subplots(figsize=(3 * len(diseases) + 2, 6))
    ax.bar(x, rows["auc"], width=0.6, color="#2a78d6")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, fontsize=16, rotation=0)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("clinic4 질병 예측 AUC (추적관찰 코호트, internal OOF)", fontsize=18,
                 fontweight="bold", color=INK_PRIMARY)
    ax.set_ylabel("AUC", fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


# clinic4(성별/나이/신장/체중)로 질병 6종 유무를 각각 독립적인 logistic regression으로 예측한다. 코호트가
# 단일 소스라 internal 5-fold OOF로만 학습/평가하며(external 없음), Youden threshold로 Se/Sp/Acc를 함께 산출
def run(meta: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_diseases: list[str] = []
    skipped = []
    for disease in DISEASES:
        y_all = pd.to_numeric(meta[disease], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        n_pos = int(y_all[mask].sum())
        n_neg = int(mask.sum() - n_pos)
        print(f"[{disease}] n_pos={n_pos}/{mask.sum()}")
        if min(n_pos, n_neg) < MIN_POSITIVES:
            msg = (f"[{disease}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만이라 logistic regression/ROC 산출이 불가함 "
                   f"(pos={n_pos} neg={n_neg})")
            print(msg)
            skipped.append({"disease": disease, "reason": msg, "n_pos": n_pos, "n_neg": n_neg})
            continue
        valid_diseases.append(disease)

    x_all, _ = clinical_matrix(meta)

    summary_rows = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    stats_by_disease: dict[str, dict] = {}
    predictions_rows = []

    for disease in valid_diseases:
        slug = DISEASES[disease]
        y_full = pd.to_numeric(meta[disease], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_full)
        x = x_all[mask]
        y = y_full[mask].astype(int)

        cv = StratifiedKFold(n_splits=n_splits_for(y), shuffle=True, random_state=SEED)
        oof_proba = cross_val_predict(LogisticRegression(max_iter=2000), x, y, cv=cv, method="predict_proba")[:, 1]
        model = LogisticRegression(max_iter=2000).fit(x, y)

        threshold = youden_threshold(y, oof_proba)
        auc = float(roc_auc_score(y, oof_proba))
        cls_stats = classification_stats(y, oof_proba, threshold)
        s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
             "auc": auc, "threshold": threshold, **cls_stats}
        _log(disease, s)

        summary_rows.append({"disease": disease, **s})
        curves[disease] = {"y": y, "score": oof_proba}
        stats_by_disease[disease] = s

        predictions_rows.append(pd.DataFrame({
            "disease": disease, "patient_id": meta.loc[mask, "patientID"].to_numpy(),
            "y": y, "score": oof_proba, "threshold": threshold,
        }))

        coef_df = pd.DataFrame({
            "term": ["sex_M", "age", "height", "weight", "intercept"],
            "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)]),
        })
        coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
        write_sheets(output_dir / f"{slug}_logistic_coefficients.xlsx", {"clinic4": coef_df.round(4)})

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_diseases.csv", index=False)
        print(f"Saved skipped disease log to {output_dir / 'skipped_diseases.csv'}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    predictions = pd.concat(predictions_rows, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved per-patient predictions to {output_dir / 'predictions.csv'}")

    if not summary.empty:
        plot_roc_all(curves, stats_by_disease, output_dir / "roc_curve_all_diseases.png")
        plot_auc_summary(summary, output_dir / "logistic_regression_auc_summary.png")


def main() -> None:
    meta = load_cohort()
    run(meta, OUTPUT_DIR)


if __name__ == "__main__":
    main()
