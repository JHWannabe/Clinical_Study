from __future__ import annotations

# baseline(clinic4_baseline_logistic.py)에 체성분 곡선(AEC/VAT/SAT/IMATA/NAMA/LAMA/TAMA, liver->pubis
# 128구간, code/0910 CURVE_SOURCES와 동일)을 추가 input으로 넣었을 때 성능이 어떻게 바뀌는지 FPCA(elbow n,
# 곡선별 고정값은 0910 CURVE_FPCA_N_ELBOW와 동일)와 Upper/Lower ratio 두 표현으로 각 곡선마다 비교한다.
# 방법론(음성군 undersampling으로 1:1 맞춘 뒤 train:valid:test=7:1:2 stratified split, 하이퍼파라미터 튜닝/
# Youden threshold/coefficient·p-value/외부검증/DeLong)은 baseline과 전부 동일한 기준(2026-09-22 baseline이
# 5-fold CV에서 이 방식으로 바뀌면서 여기도 맞춤). 원래 AEC만 다루는 clinic4_aec_compare.py를 먼저 만들었다가,
# 다른 체성분도 비교해달라는 요청으로 이 파일을 curve 일반화 버전으로 새로 만들면서 로직이 거의 그대로 겹쳤음 -
# 사용자 요청(2026-09-22: "겹치는 코드내용이 있으면 최신 파일로 유지하고 통합파일을 만들어")에 따라
# clinic4_aec_compare.py를 삭제하고 AEC를 CURVE_SOURCES에 도로 포함시켜 이 파일 하나로 7개 곡선을 전부
# 처리한다(출력도 이 폴더 하나로 통합).

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트(없으면 그래프 한글 라벨이 깨짐)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from clinic4_baseline_logistic import (
    C_GRID, DISEASES, EXTERNAL_COHORTS, INTERNAL_XLSX, SEED, TEST_RATIO, TRAIN_RATIO, VALID_RATIO,
    balanced_index, bootstrap_auc_ci, classification_stats, clinical_matrix, coefficient_table, load_cohort,
    run_disease, youden_threshold,
)
from delong_utils import bh_fdr, delong_paired_auc_test

sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 cp949가 한글을 인코딩 못 해 print에서 죽는 것 방지

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "1002" / "clinic4_baseline"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1002" / "clinic4_bodycomp_compare"
N_SLICES = 128

# code/0910 CURVE_SOURCES/CURVE_FPCA_N_ELBOW와 동일(elbow로 검증된 고정값 그대로 재사용)
CURVE_SOURCES: dict[str, tuple[str, str]] = {
    "AEC": ("aec_128", "aec"), "VAT": ("VFA_128", "VFA"), "SAT": ("SFA_128", "SFA"),
    "IMATA": ("IMATA_128", "IMATA"), "NAMA": ("NAMA_128", "NAMA"), "LAMA": ("LAMA_128", "LAMA"),
    "TAMA": ("TAMA_128", "TAMA"),
}
CURVE_FPCA_N_ELBOW: dict[str, int] = {"AEC": 3, "VAT": 2, "SAT": 2, "IMATA": 3, "NAMA": 2, "LAMA": 2, "TAMA": 2}
CURVE_COLORS = dict(zip(CURVE_SOURCES, plt.cm.tab10.colors))  # 곡선별 ROC 비교 plot 색상


def curve_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, N_SLICES + 1)]


# baseline의 load_cohort(성별/clinic4 결측 필터)에 CURVE_SOURCES 전체 곡선을 한 번에 병합하고 곡선별
# Upper/Lower ratio를 계산. 곡선 없는 환자는 inner join으로 자연 제외됨
def load_cohort_with_curves(path: Path) -> pd.DataFrame:
    meta = load_cohort(path)
    n0 = len(meta)
    for key, (sheet, prefix) in CURVE_SOURCES.items():
        curve = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        meta = meta.merge(curve[["PatientID"] + curve_cols(prefix)], on="PatientID", how="inner")
    meta = meta.reset_index(drop=True)
    print(f"[{path.stem}] 체성분 곡선({'/'.join(CURVE_SOURCES)}) 없는 환자 제외: {n0 - len(meta)}/{n0}명")

    half = N_SLICES // 2
    for key, (_sheet, prefix) in CURVE_SOURCES.items():
        cols = curve_cols(prefix)
        upper_mean = meta[cols[:half]].astype(float).mean(axis=1)
        lower_mean = meta[cols[half:]].astype(float).mean(axis=1)
        meta[f"{key}_uplow_ratio"] = upper_mean / lower_mean
    return meta


# clinic4 + 체성분곡선 FPCA. baseline(run_disease)과 동일하게 음성군 undersampling(1:1) 후
# train:valid:test=7:1:2 stratified split. PCA/clinic scaler/curve scaler는 train에서만 fit(누수 방지)
# 해 valid AUC로 C를 고르고 valid Youden으로 threshold, test는 held-out 성능. 외부검증용 frozen
# 모델(PCA 포함)은 train+valid(test 미포함)로 재적합
def run_disease_fpca(disease: str, curve_key: str, n_fpca: int, meta_int: pd.DataFrame, curve_int: np.ndarray,
                      externals: dict[str, tuple[np.ndarray, pd.DataFrame]]
                      ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    y_full = meta_int[disease].to_numpy(dtype=int)
    bal_idx = balanced_index(y_full, seed=SEED)
    meta_bal = meta_int.iloc[bal_idx].reset_index(drop=True)
    curve_bal, y = curve_int[bal_idx], y_full[bal_idx]
    patient_id_bal = meta_int["PatientID"].to_numpy()[bal_idx]
    tag = f"{curve_key} FPCA"

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=TEST_RATIO, random_state=SEED, stratify=y)
    idx_train, idx_valid = train_test_split(idx_trainval, test_size=VALID_RATIO / (TRAIN_RATIO + VALID_RATIO),
                                             random_state=SEED, stratify=y[idx_trainval])

    pca_train = PCA(n_components=n_fpca, random_state=SEED).fit(curve_bal[idx_train])
    _, clin_scaler_train = clinical_matrix(meta_bal.iloc[idx_train])
    curve_scaler_train = StandardScaler().fit(pca_train.transform(curve_bal[idx_train]))

    def make_x(row_idx: np.ndarray) -> np.ndarray:
        x_clin, _ = clinical_matrix(meta_bal.iloc[row_idx], scaler=clin_scaler_train)
        fpca = curve_scaler_train.transform(pca_train.transform(curve_bal[row_idx]))
        return np.column_stack([x_clin, fpca])

    x_train, x_valid, x_test = make_x(idx_train), make_x(idx_valid), make_x(idx_test)

    best_c, best_valid_auc = C_GRID[0], -1.0
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=2000).fit(x_train, y[idx_train])
        valid_auc = roc_auc_score(y[idx_valid], model.predict_proba(x_valid)[:, 1])
        if valid_auc > best_valid_auc:
            best_c, best_valid_auc = c, valid_auc

    train_model = LogisticRegression(C=best_c, max_iter=2000).fit(x_train, y[idx_train])
    valid_proba = train_model.predict_proba(x_valid)[:, 1]
    threshold = youden_threshold(y[idx_valid], valid_proba)

    test_proba = train_model.predict_proba(x_test)[:, 1]
    test_auc = float(roc_auc_score(y[idx_test], test_proba))
    cls_stats = classification_stats(y[idx_test], test_proba, threshold)

    rows = [{"disease": disease, "cohort": "internal", "n": int(len(idx_test)), "n_pos": int(y[idx_test].sum()),
             "prevalence": float(y[idx_test].mean()), "best_C": float(best_c), "auc_mean": test_auc,
             "auc_std": float("nan"), "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan"),
             "threshold": threshold, **cls_stats}]
    print(f"[{disease}] internal(test, {tag}) n={len(idx_test)} n_pos={int(y[idx_test].sum())} best_C={best_c:.4g} "
          f"AUC={test_auc:.3f} Acc={cls_stats['accuracy']:.3f} "
          f"Se={cls_stats['sensitivity']:.3f} Sp={cls_stats['specificity']:.3f}")

    pred_rows = [pd.DataFrame({"disease": disease, "cohort": "internal",
                                "patient_id": patient_id_bal[idx_test],
                                "y": y[idx_test], "score": test_proba, "threshold": threshold})]

    # 외부검증용 frozen 모델(PCA/scaler 포함)은 train+valid(test 미포함)로 재적합
    pca_final = PCA(n_components=n_fpca, random_state=SEED).fit(curve_bal[idx_trainval])
    x_trainval_clin, scaler_final = clinical_matrix(meta_bal.iloc[idx_trainval])
    curve_scaler_final = StandardScaler().fit(pca_final.transform(curve_bal[idx_trainval]))
    x_trainval_full = np.column_stack(
        [x_trainval_clin, curve_scaler_final.transform(pca_final.transform(curve_bal[idx_trainval]))])
    final_model = LogisticRegression(C=best_c, max_iter=2000).fit(x_trainval_full, y[idx_trainval])

    for cohort_name, (curve_ext, meta_ext) in externals.items():
        if disease not in meta_ext.columns:
            print(f"[{disease}] {cohort_name}({tag}) SKIP: 해당 질병 컬럼 없음")
            continue
        y_ext = meta_ext[disease].to_numpy(dtype=int)
        fpca_ext = pca_final.transform(curve_ext)
        x_ext, _ = clinical_matrix(meta_ext, scaler=scaler_final)
        x_ext = np.column_stack([x_ext, curve_scaler_final.transform(fpca_ext)])
        ext_proba = final_model.predict_proba(x_ext)[:, 1]
        ext_auc = float(roc_auc_score(y_ext, ext_proba))
        ci_lo, ci_hi = bootstrap_auc_ci(y_ext, ext_proba)
        ext_cls_stats = classification_stats(y_ext, ext_proba, threshold)

        rows.append({"disease": disease, "cohort": cohort_name, "n": int(len(y_ext)), "n_pos": int(y_ext.sum()),
                     "prevalence": float(y_ext.mean()), "best_C": float(best_c), "auc_mean": ext_auc,
                     "auc_std": float("nan"), "auc_ci_lower": ci_lo, "auc_ci_upper": ci_hi,
                     "threshold": threshold, **ext_cls_stats})
        print(f"[{disease}] {cohort_name}({tag}) n={len(y_ext)} n_pos={int(y_ext.sum())} "
              f"AUC={ext_auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] Acc={ext_cls_stats['accuracy']:.3f} "
              f"Se={ext_cls_stats['sensitivity']:.3f} Sp={ext_cls_stats['specificity']:.3f}")

        pred_rows.append(pd.DataFrame({"disease": disease, "cohort": cohort_name,
                                        "patient_id": meta_ext["PatientID"].to_numpy(),
                                        "y": y_ext, "score": ext_proba, "threshold": threshold}))

    # coefficient/p-value는 정규화 없는 MLE(Wald 검정 전제) - final_model과 같은 train+valid로 적합
    extra_terms = [f"{curve_key.lower()}_fpca_pc{i}" for i in range(1, n_fpca + 1)]
    coef_df = coefficient_table(x_trainval_full, y[idx_trainval], extra_terms=extra_terms)
    coef_df.insert(0, "disease", disease)

    predictions = pd.concat(pred_rows, ignore_index=True)
    return rows, coef_df, predictions


# baseline predictions.csv와 모델 predictions를 patient_id로 inner join해(코호트별 환자 집합이 다를 수
# 있음) paired DeLong test를 돌림(clinic4_aec_compare.py의 delong_vs_baseline과 동일 로직)
def delong_vs_baseline(baseline_pred: pd.DataFrame, model_pred: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    m_only = model_pred[model_pred["model"] == model_name]
    for disease in DISEASES:
        for cohort in sorted(m_only["cohort"].unique()):
            b = baseline_pred[(baseline_pred["disease"] == disease) & (baseline_pred["cohort"] == cohort)]
            m = m_only[(m_only["disease"] == disease) & (m_only["cohort"] == cohort)]
            if b.empty or m.empty:
                continue
            merged = b.merge(m, on="patient_id", suffixes=("_base", "_model"))
            if merged.empty:
                continue
            y = merged["y_base"].to_numpy(dtype=int)
            res = delong_paired_auc_test(y, merged["score_base"].to_numpy(), merged["score_model"].to_numpy())
            rows.append({"disease": disease, "cohort": cohort, "model": model_name, "n": int(len(merged)),
                         "auc_baseline": res["auc_a"], "auc_model": res["auc_b"], "auc_diff": res["diff"],
                         "z": res["z"], "p_value": res["p_value"]})
    return rows


COHORT_ROWS = ["internal"] + list(EXTERNAL_COHORTS)  # ROC 비교 plot의 행 순서(internal -> external들)
N_TOP_MODELS = 5  # 질병당 표기할 모델 수(ratio/FPCA 14개 후보 전부 중 internal AUC 상위 5개)


# 곡선 7개 x 변형 2개(ratio/FPCA) = 14개 후보 전부의 internal(test) AUC를 구해 질병별 상위 N_TOP_MODELS개만
# baseline과 함께 코호트(internal/external)별로 ROC curve 비교. 질병마다 별도 PNG로 저장(한 이미지에 다
# 몰아두면 보기 불편하다는 피드백에 따라 분리). external로 고르면 정보 누수이므로 선정은 항상 internal 기준.
# 코호트에 없는 질병은 subplot을 비움(예: new10000은 CKD 컬럼 없음)
def save_roc_curve_compare_plots(baseline_pred: pd.DataFrame, predictions_all: pd.DataFrame,
                                  summary_all: pd.DataFrame, out_dir: Path) -> None:
    for disease in DISEASES:
        candidates = []
        for key in CURVE_SOURCES:
            for variant, model in (("ratio", f"clinic4_{key.lower()}_uplow_ratio"),
                                    ("fpca", f"clinic4_{key.lower()}_fpca")):
                row = summary_all[(summary_all["disease"] == disease) & (summary_all["cohort"] == "internal") &
                                   (summary_all["model"] == model)]
                if not row.empty:
                    candidates.append((float(row["auc_mean"].iloc[0]), model, key, variant))
        candidates.sort(key=lambda t: -t[0])
        top_models = [(model, key, variant) for _auc, model, key, variant in candidates[:N_TOP_MODELS]]

        fig, axes = plt.subplots(1, len(COHORT_ROWS), figsize=(6 * len(COHORT_ROWS), 5), squeeze=False)
        for ax, cohort in zip(axes[0], COHORT_ROWS):
            b = baseline_pred[(baseline_pred["disease"] == disease) & (baseline_pred["cohort"] == cohort)]
            if b.empty:
                ax.axis("off")
                continue
            fpr, tpr, _ = roc_curve(b["y"].to_numpy(), b["score"].to_numpy())
            b_auc = roc_auc_score(b["y"].to_numpy(), b["score"].to_numpy())
            n_total, n_pos = len(b), int(b["y"].sum())
            lines = [(b_auc, "baseline", fpr, tpr, "black", "--", 2)]

            for model, key, variant in top_models:
                p = predictions_all[(predictions_all["disease"] == disease) &
                                     (predictions_all["cohort"] == cohort) &
                                     (predictions_all["model"] == model)]
                if p.empty:
                    continue
                fpr, tpr, _ = roc_curve(p["y"].to_numpy(), p["score"].to_numpy())
                auc = float(roc_auc_score(p["y"].to_numpy(), p["score"].to_numpy()))
                linestyle = "-" if variant == "fpca" else "-."  # 같은 곡선의 ratio/FPCA가 둘 다 top5에 들 수 있어 구분
                lines.append((auc, model, fpr, tpr, CURVE_COLORS[key], linestyle, 1.5))

            # 범례를 AUC 높은 순으로 정렬(plot 순서 = legend 순서)
            for auc, label, fpr, tpr, color, linestyle, linewidth in sorted(lines, key=lambda t: -t[0]):
                ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth,
                         label=f"{label} (AUC={auc:.3f}) [{n_total}/{n_pos}]")

            ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
            ax.set_xlabel("1 - Specificity")
            ax.set_ylabel("Sensitivity")
            ax.set_title(cohort)
            ax.legend(fontsize=7, loc="lower right")
        fig.suptitle(disease)
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_curve_compare_{disease}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_int = load_cohort_with_curves(INTERNAL_XLSX)
    externals_meta = {name: load_cohort_with_curves(path) for name, path in EXTERNAL_COHORTS.items()}

    all_summary, all_coef, all_pred = [], [], []

    for key, (_sheet, prefix) in CURVE_SOURCES.items():
        cols = curve_cols(prefix)
        curve_int = meta_int[cols].to_numpy(dtype=float)
        ratio_col = f"{key}_uplow_ratio"
        model_ratio = f"clinic4_{key.lower()}_uplow_ratio"
        model_fpca = f"clinic4_{key.lower()}_fpca"

        # ---- + Upper/Lower ratio (baseline의 run_disease를 extra_cols로 재사용) ----
        x_int_ratio, ratio_scaler = clinical_matrix(meta_int, extra_cols=[ratio_col])
        externals_ratio = {name: (clinical_matrix(meta_ext, extra_cols=[ratio_col], scaler=ratio_scaler)[0],
                                   meta_ext)
                            for name, meta_ext in externals_meta.items()}
        for disease in DISEASES:
            rows, coef_df, predictions = run_disease(disease, x_int_ratio, meta_int, externals_ratio,
                                                       extra_terms=[f"{key.lower()}_uplow_ratio"])
            for r in rows:
                r["model"] = model_ratio
            coef_df.insert(0, "model", model_ratio)
            predictions.insert(0, "model", model_ratio)
            all_summary.extend(rows)
            all_coef.append(coef_df)
            all_pred.append(predictions)

        # ---- + FPCA(elbow n) ----
        externals_curve = {name: (meta_ext[cols].to_numpy(dtype=float), meta_ext)
                            for name, meta_ext in externals_meta.items()}
        for disease in DISEASES:
            rows, coef_df, predictions = run_disease_fpca(disease, key, CURVE_FPCA_N_ELBOW[key], meta_int,
                                                            curve_int, externals_curve)
            for r in rows:
                r["model"] = model_fpca
            coef_df.insert(0, "model", model_fpca)
            predictions.insert(0, "model", model_fpca)
            all_summary.extend(rows)
            all_coef.append(coef_df)
            all_pred.append(predictions)

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    pd.concat(all_coef, ignore_index=True).round(6).to_csv(OUTPUT_DIR / "coefficients.csv", index=False)
    predictions_all = pd.concat(all_pred, ignore_index=True)
    predictions_all.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"Saved performance_summary.csv, coefficients.csv, predictions.csv to {OUTPUT_DIR}")

    baseline_pred = pd.read_csv(BASELINE_DIR / "predictions.csv")
    delong_rows = []
    for key in CURVE_SOURCES:
        delong_rows += delong_vs_baseline(baseline_pred, predictions_all, f"clinic4_{key.lower()}_uplow_ratio")
        delong_rows += delong_vs_baseline(baseline_pred, predictions_all, f"clinic4_{key.lower()}_fpca")
    delong_df = pd.DataFrame(delong_rows)
    delong_df["q_value_bh"] = bh_fdr(delong_df["p_value"].to_numpy())
    delong_df.round(6).to_csv(OUTPUT_DIR / "delong_comparison.csv", index=False)
    print(f"Saved delong_comparison.csv to {OUTPUT_DIR}")

    save_roc_curve_compare_plots(baseline_pred, predictions_all, summary_df, OUTPUT_DIR)
    print(f"Saved roc_curve_compare_{{disease}}.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
