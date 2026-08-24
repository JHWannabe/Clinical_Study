from __future__ import annotations

# HTN/DM/CKD 진단 이진값(metadata의 HTN/DM/CKD 컬럼, 이미 0/1)을 clinic4 baseline과 AEC-128 기반 feature로
# 예측하는 logistic regression 스크립트. 2026-08-24 기준 code/질병예측와 code/질병예측_v2 두 폴더를 하나로
# 통합하며 이 파일은 v2(재설계본) 내용으로 교체됐다(사용자 확인: "통합시켜, 최대한 v2의 내용으로 진행").
# 피드백(2026-08-21) 반영사항:
#   1) 코호트: data/{gangnam,sinchon}_원본.xlsx에서 연령<20만 제외(스캐너/벤더 제한 해제, kVp는 원본 전체가
#      100kVp뿐이라 해제 불가 - internal 1,088명 / external 925명).
#   2) 비교 모델을 두 계열로 재구성(사용자 확인 2026-08-21; Family B에 clinic4 baseline 추가는 2026-08-24):
#        Family A: clinic4 vs clinic4+mean_mAs vs clinic4+mean_mAs+AEC
#        Family B: clinic4 vs clinic4+VAT+SAT vs clinic4+VAT+SAT+AEC
#      mean_mAs는 AEC-128 128포인트 곡선(hip-to-liver 구간 관전류)의 환자별 평균(사용자 확인: 기존 mAs
#      메타데이터 컬럼은 분포가 이봉형이라 정의가 불명확해 제외). VAT/SAT는 비율이 아니라 절대값
#      VAT(내장지방)_SUM/SAT(피하지방)_SUM 두 변수를 그대로 사용(사용자 확인: "VAT + SAT"로 명시).
#   3) AEC 점수는 질환/arm별로 5개 후보 중 매번 새로 고르던 기존 방식 대신, "Prespecified AEC Score"
#      섹션명에 맞춰 internal 코호트 곡선의 elbow(Kneedle)로 한 번만 정한 FPCA PC1-k를 모든 모델·질환에
#      동일하게 적용(사용자 확인 2026-08-21: "하나의 고정 AEC 점수로 단순화"). 질환별 candidate 재탐색
#      단계(select_best_shape_model_per_feature)는 이 버전에서 완전히 제거했다.
# 그 외 5-fold stratified OOF/외부 frozen 평가, paired DeLong test, Youden's J 고정 threshold, 스캐너별
# 서브그룹(step_disease_scanner.py로 분리)은 기존 설계를 그대로 유지한다.

import sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step_disease_logistic"

INTERNAL_XLSX = DATA_DIR / "gangnam_원본.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon_원본.xlsx"
AGE_CUTOFF = 20
N_FOLDS = 5
SEED = 20260709
N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]

VAT_COL = "VAT(내장지방)_SUM"
SAT_COL = "SAT(피하지방)_SUM"
MEAN_MAS_COL = "mean_mAs"
CLINICAL_BASE_COLS = ["PatientAge", "Height", "Weight"]

FPCA_COMPONENT_CANDIDATES_MAX = 20
MIN_POSITIVES = 2

FEATURES: dict[str, str] = {"HTN": "htn", "DM": "dm", "CKD": "ckd"}

MODEL_ORDER = ["clinic4", "clinic4_meanmAs", "clinic4_meanmAs_aec", "clinic4_vatsat", "clinic4_vatsat_aec"]
MODEL_EXTRA_COLS = {
    "clinic4": [],
    "clinic4_meanmAs": [MEAN_MAS_COL],
    "clinic4_meanmAs_aec": [MEAN_MAS_COL],
    "clinic4_vatsat": [VAT_COL, SAT_COL],
    "clinic4_vatsat_aec": [VAT_COL, SAT_COL],
}
MODEL_USES_AEC = {"clinic4": False, "clinic4_meanmAs": False, "clinic4_meanmAs_aec": True,
                   "clinic4_vatsat": False, "clinic4_vatsat_aec": True}
# DeLong로 비교할 (baseline, 확장모델) 쌍 4개: mean_mAs 단독효과, mean_mAs 위에 AEC 추가효과,
# VAT+SAT 단독효과, VAT+SAT 위에 AEC 추가효과
DELONG_PAIRS = [
    ("clinic4", "clinic4_meanmAs"),
    ("clinic4_meanmAs", "clinic4_meanmAs_aec"),
    ("clinic4", "clinic4_vatsat"),
    ("clinic4_vatsat", "clinic4_vatsat_aec"),
]
MODEL_LABELS = {
    "clinic4": "clinic4",
    "clinic4_meanmAs": "clinic4 + mean mAs",
    "clinic4_meanmAs_aec": "clinic4 + mean mAs + AEC",
    "clinic4_vatsat": "clinic4 + VAT + SAT",
    "clinic4_vatsat_aec": "clinic4 + VAT + SAT + AEC",
}
# 그래프를 나눌 두 계열: A) clinic4 -> +mean mAs -> +mean mAs+AEC, B) clinic4 -> +VAT+SAT -> +VAT+SAT+AEC
# (ROC curve와 AUC 막대그래프 모두 두 계열 다 clinic4 baseline부터 3-way 비교로 통일 - 사용자 확인 2026-08-24)
FAMILIES = {
    "meanmAs": ["clinic4", "clinic4_meanmAs", "clinic4_meanmAs_aec"],
    "vatsat": ["clinic4", "clinic4_vatsat", "clinic4_vatsat_aec"],
}


# 원본 metadata에서 연령<20만 제외(스캐너/벤더 제한 없음)한 뒤 aec_128 원시곡선을 병합하고, 128포인트 평균인
# mean_mAs를 새로 계산해 둔다(사용자 확인: 기존 mAs 컬럼 대신 AEC-128 곡선 평균을 "mean mAs"로 정의)
def load_cohort(xlsx_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(xlsx_path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    meta = meta[meta["PatientAge"] >= AGE_CUTOFF].reset_index(drop=True)
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    merged = meta.merge(aec[["PatientID"] + AEC_COLS], on="PatientID", how="inner")
    assert len(merged) == len(meta), f"{xlsx_path.name}: metadata/aec_128 merge dropped rows"
    merged[MEAN_MAS_COL] = merged[AEC_COLS].astype(float).mean(axis=1)
    return merged


# scree curve(개별 explained variance ratio)를 정규화된 축에서의 chord-거리로 평가해 elbow를 찾는다
# (Satopaa et al. 2011 Kneedle). 이 값이 이 스크립트에서 유일하게 채택하는 AEC 점수의 component 수이며,
# 질환/모델과 무관하게 internal 코호트 곡선 하나로만 한 번 결정된다("Prespecified AEC Score")
def select_fpca_n_by_elbow(aec_int_raw: np.ndarray) -> tuple[int, pd.Series]:
    max_components = min(FPCA_COMPONENT_CANDIDATES_MAX, aec_int_raw.shape[0], aec_int_raw.shape[1])
    pca = PCA(n_components=max_components, random_state=SEED).fit(aec_int_raw)
    cum_var = pd.Series(np.cumsum(pca.explained_variance_ratio_), index=range(1, max_components + 1))

    scree = cum_var.diff().fillna(cum_var.iloc[0])
    x, y = scree.index.to_numpy(dtype=float), scree.to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
    dist = np.array([np.linalg.norm((pt - p1) - np.dot(pt - p1, line_vec) * line_vec)
                      for pt in np.column_stack([xn, yn])])
    elbow_n = int(np.argmax(dist)) + 1

    print(f"[FPCA] n_components별 누적 explained variance ratio:\n{cum_var.round(4)}")
    print(f"[FPCA] elbow(Kneedle) n_components = {elbow_n} (누적분산비율={cum_var[elbow_n]:.4f}) — 이 값을 "
          f"모든 모델/질환에 공통으로 쓰는 prespecified AEC score(FPCA PC1-{elbow_n})로 채택")
    return elbow_n, cum_var


def save_cum_var_excel(cum_var: pd.Series, best_n: int, out_path: Path) -> None:
    df = pd.DataFrame({"n_components": cum_var.index, "cumulative_variance_ratio": cum_var.values})
    df["selected_prespecified_n"] = df["n_components"] == best_n
    df.to_excel(out_path, index=False)
    print(f"Saved FPCA cumulative variance ratio to {out_path}")


# clinic4(age/height/weight+sex) + extra_cols(모델별 mean_mAs 또는 VAT/SAT) + (aec_extra가 있으면) AEC 점수를
# 각각 별도 StandardScaler(clinic용/AEC용)로 표준화해 결합. scaler는 internal에서 fit해 external에 frozen 적용
def build_matrix(meta: pd.DataFrame, extra_cols: list[str], aec_extra: np.ndarray | None = None,
                  scaler: StandardScaler | None = None, aec_scaler: StandardScaler | None = None
                  ) -> tuple[np.ndarray, StandardScaler, StandardScaler | None]:
    cols = CLINICAL_BASE_COLS + extra_cols
    rest = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(rest)
    scaled = scaler.transform(rest)
    clinic = np.column_stack([(meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float), scaled])
    if aec_extra is None:
        return clinic, scaler, None
    if aec_scaler is None:
        aec_scaler = StandardScaler().fit(aec_extra)
    x = np.column_stack([clinic, aec_scaler.transform(aec_extra)])
    return x, scaler, aec_scaler


def n_splits_for(y: np.ndarray) -> int:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return max(2, min(N_FOLDS, n_pos, n_neg))


# AEC를 포함하는 모델(clinic4_meanmAs_aec/clinic4_vatsat_aec)의 internal OOF 확률. PCA(고유함수 추정)를
# 검증 fold를 제외한 학습 fold에서만 fit해 곡선 정보 누수를 막는다(다른 클리닉 feature는 fold와 무관하게
# 이미 계산돼 있어 누수가 아님)
def fpca_oof_proba(meta: pd.DataFrame, aec_raw: np.ndarray, extra_cols: list[str], y: np.ndarray, n_fpca: int,
                    cv: StratifiedKFold) -> np.ndarray:
    oof = np.empty(len(y))
    for train_idx, test_idx in cv.split(aec_raw, y):
        pca = PCA(n_components=n_fpca, random_state=SEED).fit(aec_raw[train_idx])
        fpca_train, fpca_test = pca.transform(aec_raw[train_idx]), pca.transform(aec_raw[test_idx])

        x_train, scaler, aec_scaler = build_matrix(meta.iloc[train_idx], extra_cols, fpca_train)
        x_test, _, _ = build_matrix(meta.iloc[test_idx], extra_cols, fpca_test, scaler, aec_scaler)

        model = LogisticRegression(max_iter=2000).fit(x_train, y[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


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


def delong_paired_auc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[1] - aucs[0])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"),
                "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def classification_stats(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / len(y)
    return {"sensitivity": float(sens), "specificity": float(spec), "accuracy": float(acc)}


def write_sheets(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    kwargs: dict[str, Any] = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved sheet(s) {list(sheets)} to {path}")


def plot_roc(feat: str, curves: dict[str, dict[str, np.ndarray]], stats_by_model: dict[str, dict[str, dict]],
             out_path: Path, model_list: list[str]) -> None:
    colors = {"clinic4": "#898781", "clinic4_meanmAs": "#2a78d6", "clinic4_meanmAs_aec": "#1baf7a",
              "clinic4_vatsat": "#a35ad1", "clinic4_vatsat_aec": "#e2622e"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, cohort in zip(axes, ["internal", "external"]):
        y = curves[cohort]["y"]
        for model_name in model_list:
            score = curves[cohort][model_name]
            fpr, tpr, _ = roc_curve(y, score)
            s = stats_by_model[model_name][cohort]
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=1.8,
                     label=f"{MODEL_LABELS[model_name]} AUC={s['auc']:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{feat} ({cohort})", fontsize=16, fontweight="bold", color="#161616")
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to {out_path}")


def run(meta_int: pd.DataFrame, meta_ext: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aec_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[AEC_COLS].astype(float).to_numpy()

    n_fpca, cum_var = select_fpca_n_by_elbow(aec_int_raw)
    save_cum_var_excel(cum_var, n_fpca, output_dir / "fpca_cumulative_variance.xlsx")
    pca_full = PCA(n_components=n_fpca, random_state=SEED).fit(aec_int_raw)
    fpca_int_full, fpca_ext_full = pca_full.transform(aec_int_raw), pca_full.transform(aec_ext_raw)
    print(f"[FPCA] explained variance ratio (PC1-{n_fpca}): {pca_full.explained_variance_ratio_.round(4)}")

    fpca_cols = [f"FPCA{i}" for i in range(1, n_fpca + 1)]
    df_fpca_int = pd.DataFrame(fpca_int_full, columns=fpca_cols)
    df_fpca_int.insert(0, "PatientID", meta_int["PatientID"].to_numpy())
    df_fpca_ext = pd.DataFrame(fpca_ext_full, columns=fpca_cols)
    df_fpca_ext.insert(0, "PatientID", meta_ext["PatientID"].to_numpy())
    write_sheets(output_dir / "fpca_scores.xlsx", {"internal": df_fpca_int, "external": df_fpca_ext})

    valid_features: list[str] = []
    skipped = []
    for feat in FEATURES:
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_val_int, mask_val_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        n_pos_int, n_neg_int = int(y_int_all[mask_val_int].sum()), int(mask_val_int.sum() - y_int_all[mask_val_int].sum())
        n_pos_ext, n_neg_ext = int(y_ext_all[mask_val_ext].sum()), int(mask_val_ext.sum() - y_ext_all[mask_val_ext].sum())
        print(f"[{feat}] internal n_pos={n_pos_int}/{mask_val_int.sum()} external n_pos={n_pos_ext}/{mask_val_ext.sum()}")
        if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < MIN_POSITIVES:
            msg = f"[{feat}] SKIP: 한쪽 클래스가 {MIN_POSITIVES}명 미만"
            print(msg)
            skipped.append({"feature": feat, "reason": msg})
            continue
        valid_features.append(feat)

    summary_rows, delong_rows, predictions_rows = [], [], []

    for feat in valid_features:
        slug = FEATURES[feat]
        y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
        y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
        mask_int, mask_ext = np.isfinite(y_int_all), np.isfinite(y_ext_all)
        y_int, y_ext = y_int_all[mask_int].astype(int), y_ext_all[mask_ext].astype(int)

        meta_int_m, meta_ext_m = meta_int.loc[mask_int].reset_index(drop=True), meta_ext.loc[mask_ext].reset_index(drop=True)
        aec_int_m, aec_ext_m = aec_int_raw[mask_int], aec_ext_raw[mask_ext]
        fpca_int_m, fpca_ext_m = fpca_int_full[mask_int], fpca_ext_full[mask_ext]

        cv = StratifiedKFold(n_splits=n_splits_for(y_int), shuffle=True, random_state=SEED)

        feat_dir = output_dir / slug
        feat_dir.mkdir(parents=True, exist_ok=True)

        stats_by_model: dict[str, dict[str, dict]] = {m: {} for m in MODEL_ORDER}
        scores_by_model: dict[str, dict[str, np.ndarray]] = {m: {} for m in MODEL_ORDER}
        threshold_by_model: dict[str, float] = {}
        coef_sheets = {}

        for model_name in MODEL_ORDER:
            extra_cols = MODEL_EXTRA_COLS[model_name]
            uses_aec = MODEL_USES_AEC[model_name]

            if uses_aec:
                oof_proba = fpca_oof_proba(meta_int_m, aec_int_m, extra_cols, y_int, n_fpca, cv)
                x_int_full, scaler, aec_scaler = build_matrix(meta_int_m, extra_cols, fpca_int_m)
                x_ext_full, _, _ = build_matrix(meta_ext_m, extra_cols, fpca_ext_m, scaler, aec_scaler)
            else:
                x_int_full, scaler, _ = build_matrix(meta_int_m, extra_cols)
                x_ext_full, _, _ = build_matrix(meta_ext_m, extra_cols, scaler=scaler)
                oof_proba = cross_val_predict(LogisticRegression(max_iter=2000), x_int_full, y_int, cv=cv,
                                               method="predict_proba")[:, 1]

            model = LogisticRegression(max_iter=2000).fit(x_int_full, y_int)
            ext_proba = model.predict_proba(x_ext_full)[:, 1]

            threshold = youden_threshold(y_int, oof_proba)
            threshold_by_model[model_name] = threshold

            for cohort, meta_c, y, score in (("internal", meta_int_m, y_int, oof_proba),
                                              ("external", meta_ext_m, y_ext, ext_proba)):
                auc = float(roc_auc_score(y, score))
                cls_stats = classification_stats(y, score, threshold)
                s = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "auc": auc,
                     "threshold": threshold, **cls_stats}
                print(f"[{feat} / {model_name} / {cohort}] n={s['n']} n_pos={s['n_pos']} ({s['prevalence']:.1%}) "
                      f"AUC={s['auc']:.3f} Se={s['sensitivity']:.3f} Sp={s['specificity']:.3f} Acc={s['accuracy']:.3f}")
                summary_rows.append({"feature": feat, "model": model_name, "cohort": cohort, **s})
                stats_by_model[model_name][cohort] = s
                scores_by_model[model_name][cohort] = score

                predictions_rows.append(pd.DataFrame({
                    "feature": feat, "model": model_name, "cohort": cohort,
                    "patient_id": meta_c["PatientID"].to_numpy(), "manufacturer": meta_c["Manufacturer"].astype(str).to_numpy(),
                    "y": y, "score": score, "threshold": threshold,
                }))

            n_extra_aec = n_fpca if uses_aec else 0
            input_cols = ["sex_M", "age", "height", "weight"] + extra_cols + [f"fpca_pc{i}" for i in range(1, n_extra_aec + 1)]
            coef_df = pd.DataFrame({"term": input_cols + ["intercept"],
                                     "coefficient": np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_)])})
            coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
            coef_sheets[model_name] = coef_df.round(4)

        for baseline, extended in DELONG_PAIRS:
            for cohort, y in (("internal", y_int), ("external", y_ext)):
                d = delong_paired_auc_test(y, scores_by_model[baseline][cohort], scores_by_model[extended][cohort])
                print(f"[{feat} / {cohort}] DeLong {baseline} vs {extended}: AUC diff={d['diff']:+.4f} "
                      f"z={d['z']:.3f} p={d['p_value']:.4f}")
                delong_rows.append({"feature": feat, "cohort": cohort, "baseline_model": baseline,
                                     "extended_model": extended, "auc_baseline": d["auc_a"],
                                     "auc_extended": d["auc_b"], "auc_diff": d["diff"], "z": d["z"],
                                     "p_value": d["p_value"]})

        write_sheets(feat_dir / f"{slug}_logistic_coefficients.xlsx", coef_sheets)

        curves = {
            "internal": {"y": y_int, **{m: scores_by_model[m]["internal"] for m in MODEL_ORDER}},
            "external": {"y": y_ext, **{m: scores_by_model[m]["external"] for m in MODEL_ORDER}},
        }
        for family_name, model_list in FAMILIES.items():
            plot_roc(feat, curves, stats_by_model, feat_dir / f"{slug}_roc_curve_{family_name}.png", model_list)

    if skipped:
        pd.DataFrame(skipped).to_csv(output_dir / "skipped_features.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "logistic_regression_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'logistic_regression_summary.csv'}")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_auc_comparison.csv", index=False)
    print(f"Saved DeLong comparison to {output_dir / 'delong_auc_comparison.csv'}")

    predictions = pd.concat(predictions_rows, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved per-patient predictions to {output_dir / 'predictions.csv'}")

    for family_name, model_list in FAMILIES.items():
        plot_auc_summary(summary, output_dir / f"logistic_regression_auc_summary_{family_name}.png", model_list)
    build_and_save_delta_table(summary, delong_df, output_dir)


def plot_auc_summary(summary: pd.DataFrame, out_path: Path, model_list: list[str]) -> None:
    colors = {"clinic4": "#898781", "clinic4_meanmAs": "#2a78d6", "clinic4_meanmAs_aec": "#1baf7a",
              "clinic4_vatsat": "#a35ad1", "clinic4_vatsat_aec": "#e2622e"}
    features = [f for f in FEATURES if f in summary["feature"].unique()]
    slugs = [FEATURES[f] for f in features]
    x = np.arange(len(features))
    width = 0.8 / len(model_list)

    fig, axes = plt.subplots(1, 2, figsize=(8 * len(features) / 2 + 4, 6))
    for ax, cohort in zip(axes, ["internal", "external"]):
        sub = summary[summary["cohort"] == cohort]
        for i, model_name in enumerate(model_list):
            rows = sub[sub["model"] == model_name].set_index("feature").reindex(features)
            offset = (i - (len(model_list) - 1) / 2) * width
            ax.bar(x + offset, rows["auc"], width, label=MODEL_LABELS[model_name], color=colors[model_name])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(slugs, fontsize=18)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(cohort, fontsize=18, fontweight="bold", color="#161616")
        ax.set_ylabel("AUC", fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(alpha=0.3, axis="y")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15), fontsize=14,
               frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary plot to {out_path}")


def build_and_save_delta_table(summary: pd.DataFrame, delong: pd.DataFrame, output_dir: Path) -> None:
    pivot = summary.pivot(index=["feature", "cohort"], columns="model", values="auc")
    pivot.columns = [f"auc_{m}" for m in pivot.columns]
    pivot = pivot.reset_index()

    merged = pivot
    for baseline, extended in DELONG_PAIRS:
        d = delong[(delong["baseline_model"] == baseline) & (delong["extended_model"] == extended)][
            ["feature", "cohort", "auc_diff", "z", "p_value"]].copy()
        colname = f"{extended}_minus_{baseline}"
        d = d.rename(columns={"auc_diff": f"delta_auc_{colname}", "z": f"delong_z_{colname}",
                               "p_value": f"delong_p_{colname}"})
        merged = merged.merge(d, on=["feature", "cohort"], how="left")

    feature_order = list(FEATURES.keys())
    merged["feature"] = pd.Categorical(merged["feature"], categories=feature_order, ordered=True)
    merged["cohort"] = pd.Categorical(merged["cohort"], categories=["internal", "external"], ordered=True)
    merged = merged.sort_values(["feature", "cohort"]).reset_index(drop=True)

    merged.round(4).to_csv(output_dir / "auc_delta_summary.csv", index=False)
    merged.round(4).to_excel(output_dir / "auc_delta_summary.xlsx", index=False)
    print(f"Saved AUC/delta summary table to {output_dir / 'auc_delta_summary.csv'}")


def main() -> None:
    meta_int, meta_ext = load_cohort(INTERNAL_XLSX), load_cohort(EXTERNAL_XLSX)

    required_cols = CLINICAL_BASE_COLS + [VAT_COL, SAT_COL, MEAN_MAS_COL]

    def valid_rows(meta: pd.DataFrame) -> np.ndarray:
        vals = meta[required_cols].apply(pd.to_numeric, errors="coerce")
        mask = vals.notna().all(axis=1).to_numpy()
        valid_sex = meta["PatientSex"].astype(str).str.upper().isin(["M", "F"]).to_numpy()
        return mask & valid_sex

    mask_int, mask_ext = valid_rows(meta_int), valid_rows(meta_ext)
    print(f"Clinical input 결측 제외: internal {(~mask_int).sum()}/{len(mask_int)}, "
          f"external {(~mask_ext).sum()}/{len(mask_ext)}")
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"Final cohort: internal n={len(meta_int)}, external n={len(meta_ext)}")

    run(meta_int, meta_ext, OUTPUT_DIR)


if __name__ == "__main__":
    main()
