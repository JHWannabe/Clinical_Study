from __future__ import annotations

# step4(진단) 부속 스크립트: step2(회귀)는 FPCA n_components를 internal CV로 매번 동적 선택하는데
# total scope 결과가 n=12로 나온다(outputs/step1_fpca, step12 슬라이드). 반면 step3(이진분류)는
# N_FPCA_COMPONENTS=3이 코드 상수로 고정되어 있고, 그 주석은 "step2와 동일"이라 되어 있지만 실제로는
# 다르다(회귀=12, 분류=3). 이 스크립트는 이 불일치가 실제 결과(R²/AUC)에 얼마나 영향을 주는지 total
# scope에서 n=3 vs n=12로 회귀·분류 양쪽을 직접 재계산해 비교한다. 정식 파이프라인(step2/step3)의
# output dir은 건드리지 않고 outputs/step5_fpca_n3_vs_n12/total/에만 결과를 남기는 진단 스크립트다.
#
# 모델 재선택(reselection)이 아니라 진단 목적이므로, internal CV로 고른 조합을 external로 1회만
# 검증한다는 원칙([[feedback_internal_external_validation_discipline]])은 지키되, 이번엔 internal과
# external 결과를 둘 다 나란히 "보고"만 한다(어느 n을 채택할지 이 스크립트가 결정하지 않음).
#
# n=12 대비 n=3의 델타(=n12-n3)를 9개 feature별로 그린 비교 차트(plot_delta)까지 이 스크립트에서
# 함께 생성한다. 분류 패널은 DeLong p<0.05인 막대에만 유의 표시(*)를 붙인다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

import step2_clinic_aec_linear as s2
import step3_clinic_aec_logistic as s3

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step5_fpca_n3_vs_n12"
N_VALUES = [3, 12]
REG_FEATURES = {**s2.FEATURES, **s2.ABS_FEATURES}  # 6 ratio + 3 abs = 9

FEATURE_LABELS = {
    "VAT_TotalFat_ratio": "VAT/TotalFat",
    "VAT_SAT_ratio": "VAT/SAT",
    "VAT_TAMA_ratio": "VAT/TAMA",
    "TotalFat_TAMA_ratio": "TotalFat/TAMA",
    "SAT_TotalFat_ratio": "SAT/TotalFat",
    "SAT_TAMA_ratio": "SAT/TAMA",
    "VAT(내장지방)_SUM": "VAT",
    "SAT(피하지방)_SUM": "SAT",
    "Total Fat_SUM": "Total Fat",
}
FEATURE_ORDER = list(FEATURE_LABELS.keys())


# main()의 clinical NA 필터링과 동일한 로직(step2/step3 공통)
def valid_clinical_rows(meta: pd.DataFrame) -> np.ndarray:
    vals = meta[["PatientAge", "Height", "Weight"]].apply(pd.to_numeric, errors="coerce")
    return vals.notna().all(axis=1).to_numpy()


# ---------- 회귀(step2 로직 재사용, shape_all 후보를 n=3/n=12로 직접 평가) ----------
def regression_comparison(meta_int: pd.DataFrame, meta_ext: pd.DataFrame) -> pd.DataFrame:
    cv = KFold(n_splits=s2.N_FOLDS, shuffle=True, random_state=s2.SEED)
    aec_int_raw = meta_int[s2.AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[s2.AEC_COLS].astype(float).to_numpy()
    shape_int, shape_ext = s2.shape_features(aec_int_raw), s2.shape_features(aec_ext_raw)
    shape_int_stack = np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]])
    shape_ext_stack = np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"]])

    rows = []
    for n in N_VALUES:
        fpca_int, fpca_ext, _ = s2.fpca_scores(aec_int_raw, aec_ext_raw, n)
        shape_all_int = np.column_stack([shape_int_stack, fpca_int])
        shape_all_ext = np.column_stack([shape_ext_stack, fpca_ext])
        x_int_all, scaler = s2.clinical_matrix(meta_int, shape_all_int, include_sex=True)
        x_ext_all, _ = s2.clinical_matrix(meta_ext, shape_all_ext, scaler, include_sex=True)

        for feat in REG_FEATURES:
            y_int_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
            y_ext_all = pd.to_numeric(meta_ext[feat], errors="coerce").to_numpy(dtype=float)
            mask_int = np.isfinite(y_int_all)
            mask_ext = np.isfinite(y_ext_all)
            y_int, y_ext = y_int_all[mask_int], y_ext_all[mask_ext]

            oof = s2._fpca_oof_predict(meta_int, aec_int_raw, shape_int_stack, y_int_all, mask_int, n, True, cv)
            model = LinearRegression().fit(x_int_all[mask_int], y_int)
            pred_ext = model.predict(x_ext_all[mask_ext])

            rows.append({
                "feature": feat, "n_fpca": n,
                "r2_internal": r2_score(y_int, oof), "r2_external": r2_score(y_ext, pred_ext),
            })
            print(f"[REG n={n}] {feat}: internal R2={rows[-1]['r2_internal']:.4f} "
                  f"external R2={rows[-1]['r2_external']:.4f}")
    return pd.DataFrame(rows)


# ---------- 분류(step3 로직 재사용, n을 명시적으로 threading — 모듈 상수 monkeypatch 의존 안 함) ----------
# select_best_shape_model을 n_fpca 명시적 인자로 재구현(원본은 fpca_scores() 기본값에 의존해
# N_FPCA_COMPONENTS monkeypatch가 최종 계수 산출에는 반영 안 되는 함정이 있어 직접 재구현함)
def select_best_shape_model_fixed_n(meta_int, aec_int_raw, aec_ext_raw, include_sex, target_features, cv, n_fpca):
    shape_int, shape_ext = s3.shape_features(aec_int_raw), s3.shape_features(aec_ext_raw)
    fpca_int, fpca_ext = s3.fpca_scores(aec_int_raw, aec_ext_raw, n_fpca)
    shape_int_stack = np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"]])

    candidates = {
        "aec_sd": (shape_int["sd"].reshape(-1, 1), shape_ext["sd"].reshape(-1, 1)),
        "aec_skew": (shape_int["skew"].reshape(-1, 1), shape_ext["skew"].reshape(-1, 1)),
        "aec_uplow_ratio": (shape_int["upper_lower_ratio"].reshape(-1, 1), shape_ext["upper_lower_ratio"].reshape(-1, 1)),
        "aec_fpca": (fpca_int, fpca_ext),
        "aec_shape_all": (
            np.column_stack([shape_int["sd"], shape_int["skew"], shape_int["upper_lower_ratio"], fpca_int]),
            np.column_stack([shape_ext["sd"], shape_ext["skew"], shape_ext["upper_lower_ratio"], fpca_ext]),
        ),
    }
    mean_r2 = {}
    for name, (aec_int, _) in candidates.items():
        fpca_kind = s3.FPCA_CANDIDATE_KINDS.get(name)
        x_int_all = None
        if fpca_kind is None:
            x_int_all, _, _ = s3.clinical_matrix(meta_int, aec_int, include_sex=include_sex)
        r2s = []
        for feat in target_features:
            y_all = pd.to_numeric(meta_int[feat], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y_all)
            y = y_all[mask]
            if fpca_kind is None:
                oof = cross_val_predict(LinearRegression(), x_int_all[mask], y, cv=cv)
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof = s3._fpca_oof_predict(meta_int, aec_int_raw, shape_extra, y_all, mask, n_fpca, include_sex, cv)
            r2s.append(r2_score(y, oof))
        mean_r2[name] = float(np.mean(r2s))

    best_name = max(mean_r2, key=lambda k: mean_r2[k])
    print(f"[CLS n={n_fpca}] 후보별 internal OOF 평균 R^2: {({k: round(v, 4) for k, v in mean_r2.items()})}")
    print(f"[CLS n={n_fpca}] 선택된 조합 = {best_name}")
    return best_name, candidates[best_name][0], candidates[best_name][1], shape_int_stack


def classification_comparison(meta_int: pd.DataFrame, meta_ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sex_int = meta_int["PatientSex"].astype(str).str.upper().to_numpy()
    sex_ext = meta_ext["PatientSex"].astype(str).str.upper().to_numpy()
    aec_int_raw = meta_int[s3.AEC_COLS].astype(float).to_numpy()
    aec_ext_raw = meta_ext[s3.AEC_COLS].astype(float).to_numpy()
    cv_reg = KFold(n_splits=s3.N_FOLDS, shuffle=True, random_state=s3.SEED)

    rows = []
    best_names = {}
    scores_by_n: dict[int, dict[str, dict[str, np.ndarray]]] = {n: {} for n in N_VALUES}
    y_by_feat: dict[str, dict[str, np.ndarray]] = {}

    for n in N_VALUES:
        best_name, aec_int_best, aec_ext_best, shape_int_stack = select_best_shape_model_fixed_n(
            meta_int, aec_int_raw, aec_ext_raw, True, list(s3.FEATURES.keys()), cv_reg, n)
        best_names[n] = best_name
        fpca_kind = s3.FPCA_CANDIDATE_KINDS.get(best_name)

        x_int, clinic_scaler, aec_scaler = s3.clinical_matrix(meta_int, aec_int_best, include_sex=True)
        x_ext, _, _ = s3.clinical_matrix(meta_ext, aec_ext_best, clinic_scaler, aec_scaler, include_sex=True)

        for feat, (slug, direction) in s3.FEATURES.items():
            values_int = s3.label_source_values(meta_int, feat)
            values_ext = s3.label_source_values(meta_ext, feat)
            mask_val_int = np.isfinite(values_int)
            mask_val_ext = np.isfinite(values_ext)
            cutoffs = s3.sex_specific_cutoffs(values_int[mask_val_int], sex_int[mask_val_int], direction)
            y_int_all = s3.apply_cutoff_label(values_int, sex_int, cutoffs, direction)
            y_ext_all = s3.apply_cutoff_label(values_ext, sex_ext, cutoffs, direction)

            n_pos_int = int(y_int_all[mask_val_int].sum())
            n_neg_int = int(mask_val_int.sum() - n_pos_int)
            n_pos_ext = int(y_ext_all[mask_val_ext].sum())
            n_neg_ext = int(mask_val_ext.sum() - n_pos_ext)
            if min(n_pos_int, n_neg_int, n_pos_ext, n_neg_ext) < s3.MIN_POSITIVES:
                print(f"[CLS n={n}] SKIP {feat}: 클래스 수 부족")
                continue

            mask_int, mask_ext = mask_val_int, mask_val_ext
            y_int, y_ext = y_int_all[mask_int], y_ext_all[mask_ext]
            y_by_feat[feat] = {"internal": y_int, "external": y_ext}

            n_splits = max(2, min(s3.N_FOLDS, n_pos_int, n_neg_int))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s3.SEED)

            x_int_f, x_ext_f = x_int[mask_int], x_ext[mask_ext]
            if fpca_kind is None:
                oof_proba = cross_val_predict(LogisticRegression(max_iter=2000), x_int_f, y_int,
                                               cv=cv, method="predict_proba")[:, 1]
            else:
                shape_extra = shape_int_stack if fpca_kind == "shape_all" else None
                oof_proba = s3._fpca_oof_proba_predict(meta_int, aec_int_raw, shape_extra, y_int_all, mask_int,
                                                         n, True, cv)
            model = LogisticRegression(max_iter=2000).fit(x_int_f, y_int)
            ext_proba = model.predict_proba(x_ext_f)[:, 1]

            scores_by_n[n][feat] = {"internal": oof_proba, "external": ext_proba}
            auc_int = roc_auc_score(y_int, oof_proba)
            auc_ext = roc_auc_score(y_ext, ext_proba)
            rows.append({"feature": feat, "n_fpca": n, "best_candidate": best_name,
                         "auc_internal": auc_int, "auc_external": auc_ext})
            print(f"[CLS n={n}] {feat} (best={best_name}): internal AUC={auc_int:.4f} external AUC={auc_ext:.4f}")

    # n=3 vs n=12를 같은 환자 집합(mask 동일 - cutoff이 값/방향에만 의존, n에 무관)에서 DeLong paired test로 비교
    delong_rows = []
    for feat in y_by_feat:
        if feat not in scores_by_n[3] or feat not in scores_by_n[12]:
            continue
        for cohort in ("internal", "external"):
            y = y_by_feat[feat][cohort]
            score_n3 = scores_by_n[3][feat][cohort]
            score_n12 = scores_by_n[12][feat][cohort]
            d = s3.delong_paired_auc_test(y, score_n3, score_n12)
            delong_rows.append({"feature": feat, "cohort": cohort, **d})

    print(f"\nbest_candidate by n: {best_names}")
    return pd.DataFrame(rows), pd.DataFrame(delong_rows)


# ---------- n=3 vs n=12 델타를 9개 feature별로 그리는 비교 차트 ----------
def plot_delta(reg_df: pd.DataFrame, cls_df: pd.DataFrame, delong_df: pd.DataFrame) -> None:
    reg_p = reg_df.pivot(index="feature", columns="n_fpca", values=["r2_internal", "r2_external"]).loc[FEATURE_ORDER]
    reg_d_int = reg_p[("r2_internal", 12)] - reg_p[("r2_internal", 3)]
    reg_d_ext = reg_p[("r2_external", 12)] - reg_p[("r2_external", 3)]

    cls_p = cls_df.pivot(index="feature", columns="n_fpca", values=["auc_internal", "auc_external"]).loc[FEATURE_ORDER]
    cls_d_int = cls_p[("auc_internal", 12)] - cls_p[("auc_internal", 3)]
    cls_d_ext = cls_p[("auc_external", 12)] - cls_p[("auc_external", 3)]

    p_by_feat_cohort = delong_df.set_index(["feature", "cohort"])["p_value"]

    labels = [FEATURE_LABELS[f] for f in FEATURE_ORDER]
    x = np.arange(len(FEATURE_ORDER))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(34, 13))

    ax = axes[0]
    ax.bar(x - width / 2, reg_d_int.values, width, label="internal", color="#2a78d6")
    ax.bar(x + width / 2, reg_d_ext.values, width, label="external", color="#e2622e")
    ax.axhline(0, color="#161616", linewidth=1.5)
    ax.set_title("회귀 R² 델타 (n=12 - n=3, shape_all)", fontsize=34, fontweight="bold", color="#161616", pad=20)
    ax.set_ylabel("ΔR²", fontsize=30)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=24)
    ax.tick_params(axis="y", labelsize=26)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=26, loc="upper left")

    ax = axes[1]
    bars_int = ax.bar(x - width / 2, cls_d_int.values, width, label="internal", color="#2a78d6")
    bars_ext = ax.bar(x + width / 2, cls_d_ext.values, width, label="external", color="#e2622e")
    ax.axhline(0, color="#161616", linewidth=1.5)
    ax.set_title("분류 AUC 델타 (n=12 - n=3, clinic4_aec_best)", fontsize=34, fontweight="bold", color="#161616",
                 pad=20)
    ax.set_ylabel("ΔAUC", fontsize=30)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=24)
    ax.tick_params(axis="y", labelsize=26)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=26, loc="upper left")

    for bars, cohort in ((bars_int, "internal"), (bars_ext, "external")):
        for bar, feat in zip(bars, FEATURE_ORDER):
            p = p_by_feat_cohort.get((feat, cohort), np.nan)
            if pd.notna(p) and p < 0.05:
                height = bar.get_height()
                ax.annotate("*", (bar.get_x() + bar.get_width() / 2, height), ha="center",
                            va="bottom" if height >= 0 else "top", fontsize=32, color="#161616")

    fig.text(0.99, 0.01, "* DeLong p<0.05 (n=3 vs n=12, paired)", ha="right", fontsize=20, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path = OUTPUT_DIR / "fpca_n3_vs_n12_delta.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_int, meta_ext = s2.load_cohort(s2.INTERNAL_XLSX), s2.load_cohort(s2.EXTERNAL_XLSX)
    mask_int, mask_ext = valid_clinical_rows(meta_int), valid_clinical_rows(meta_ext)
    meta_int = meta_int[mask_int].reset_index(drop=True)
    meta_ext = meta_ext[mask_ext].reset_index(drop=True)
    print(f"internal n={len(meta_int)}, external n={len(meta_ext)}")

    reg_df = regression_comparison(meta_int, meta_ext)
    reg_df.to_csv(OUTPUT_DIR / "regression_n3_vs_n12.csv", index=False)
    reg_pivot = reg_df.pivot(index="feature", columns="n_fpca", values=["r2_internal", "r2_external"])
    print("\n=== 회귀(shape_all 모델) n=3 vs n=12 ===")
    print(reg_pivot.round(4))

    cls_df, delong_df = classification_comparison(meta_int, meta_ext)
    cls_df.to_csv(OUTPUT_DIR / "classification_n3_vs_n12.csv", index=False)
    delong_df.to_csv(OUTPUT_DIR / "classification_n3_vs_n12_delong.csv", index=False)
    cls_pivot = cls_df.pivot(index="feature", columns="n_fpca", values=["auc_internal", "auc_external"])
    print("\n=== 분류(clinic4_aec_best) n=3 vs n=12 ===")
    print(cls_pivot.round(4))
    print("\n=== DeLong paired test (n=3 vs n=12, 같은 환자) ===")
    print(delong_df.round(4).to_string())

    plot_delta(reg_df, cls_df, delong_df)


if __name__ == "__main__":
    main()
