from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# 0904의 12개 clinic*compare*.py가 전부 동일한 LogisticRegression 파이프라인(모델별 internal full-fit +
# 별도 external frozen 평가) 구조라 SHAP 계산/저장 로직을 공통 모듈로 뺐다(사용자 요청 2026-09-08: "SHAP로도
# 분석해줘", 전체 12개 스크립트 적용 확인). LinearExplainer는 로지스틱 회귀 계수를 그대로 설명하므로
# feature_perturbation 방식과 무관하게 정확하고(선형모델 SHAP는 근사가 아니라 폐형식), background/masker를
# internal 코호트로 고정해 OR forest plot과 동일하게 "internal full-fit 모델을 그대로 설명"한다. external은
# 그 frozen 모델·마스커에 external X를 넣어 설명값만 새로 계산할 뿐 모델을 재학습하지 않는다
# ([[feedback_internal_external_validation_discipline]] 준수)


def run_shap_analysis(model, x_int: np.ndarray, x_ext: np.ndarray, feature_names: list[str],
                       patient_ids_int: np.ndarray, patient_ids_ext: np.ndarray,
                       y_int: np.ndarray, y_ext: np.ndarray, feat_dir: Path, slug: str,
                       model_name: str, feat_label: str, model_label: str,
                       label_fs: float, tick_fs: float) -> None:
    masker = shap.maskers.Independent(x_int)
    explainer = shap.LinearExplainer(model, masker)

    for cohort, x, pids, y in (("internal", x_int, patient_ids_int, y_int),
                                ("external", x_ext, patient_ids_ext, y_ext)):
        exp = explainer(x)
        exp.feature_names = list(feature_names)

        shap_df = pd.DataFrame(exp.values, columns=feature_names)
        shap_df.insert(0, "patient_id", pids)
        shap_df.insert(1, "y", y)
        shap_df["base_value"] = exp.base_values
        shap_df.to_csv(feat_dir / f"{slug}_shap_values_{model_name}_{cohort}.csv", index=False)

        title = f"{feat_label} / {model_label} SHAP feature importance ({cohort})"
        shap.plots.bar(exp, show=False)
        ax = plt.gca()
        ax.set_title(title, fontsize=label_fs, fontweight="bold", color="#161616")
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        plt.tight_layout()
        plt.savefig(feat_dir / f"{slug}_shap_bar_{model_name}_{cohort}.png", dpi=200, bbox_inches="tight")
        plt.close()

        shap.plots.beeswarm(exp, show=False)
        ax = plt.gca()
        ax.set_title(title.replace("feature importance", "beeswarm"), fontsize=label_fs,
                     fontweight="bold", color="#161616")
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        plt.tight_layout()
        plt.savefig(feat_dir / f"{slug}_shap_beeswarm_{model_name}_{cohort}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"[{feat_label} / {model_name}] Saved SHAP bar/beeswarm/csv (internal+external) to {feat_dir}")
