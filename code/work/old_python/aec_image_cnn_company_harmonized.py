from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aec_image_cnn_exploratory as cnn  # noqa: E402
from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402
from aec_vendor_neutral_preprocessing_audit import company_from_manufacturer  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_image_cnn_company_harmonized"
WORK_DATA_DIR = Path(__file__).resolve().parent / "data_cache"
SIGMA = 1.0


def load_raw_dataset(path: Path) -> dict:
    """엑셀에서 원시 AEC_128 곡선, 정규화 버전, 라벨, 제조사(회사) 범주를 함께 읽어옴."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    company = meta["Manufacturer"].map(company_from_manufacturer).to_numpy()
    return {"meta": meta, "raw": raw, "norm": row_norm(raw), "y": y, "company": company}


def rank_rows(x: np.ndarray) -> np.ndarray:
    """각 행의 값을 순위(0~1로 정규화)로 바꿔 절대값 스케일에 무관한 표현으로 변환."""
    ranked = np.vstack([stats.rankdata(row, method="average") for row in x])
    ranked = (ranked - 1.0) / (x.shape[1] - 1.0)
    return ranked - ranked.mean(axis=1, keepdims=True)


def company_template_harmonize_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
    company_train: np.ndarray,
    company_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """train에서 회사별 평균 템플릿과 전체 평균 템플릿을 만들어, 각 환자 곡선의 회사 평균을 전체 평균으로 치환(harmonize)."""
    keep = company_train != "Other"
    global_template = x_train[keep].mean(axis=0)
    templates = {
        label: x_train[company_train == label].mean(axis=0)
        for label in np.unique(company_train[keep])
    }

    def apply(x: np.ndarray, company: np.ndarray) -> np.ndarray:
        """각 행에서 그 행의 회사 템플릿을 빼고 전체 템플릿을 더해 회사 차이를 보정."""
        out = np.empty_like(x)
        for i, label in enumerate(company):
            template = templates.get(label, global_template)
            out[i] = x[i] - template + global_template
        return out

    return apply(x_train, company_train), apply(x_test, company_test)


def base_transform(raw: np.ndarray, mode: str) -> np.ndarray:
    """평활화된 원시곡선을 모드에 따라 평균정규화/로그중심화/순위모양 중 하나로 변환 (회사 보정 이전 단계)."""
    smooth_raw = ndimage.gaussian_filter1d(raw, sigma=SIGMA, axis=1, mode="nearest")
    x = row_norm(smooth_raw)
    if mode == "smooth_mean_norm_company_harmonized":
        return x
    if mode == "smooth_log_centered_company_harmonized":
        lx = np.log(np.clip(x, 1e-6, None))
        return lx - lx.mean(axis=1, keepdims=True)
    if mode == "smooth_rank_shape_company_harmonized":
        return rank_rows(x)
    raise ValueError(mode)


def evaluate_mode(mode: str, g: dict, s: dict, cg: np.ndarray, cs: np.ndarray) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """한 전처리 모드에 대해 회사별 템플릿 보정을 적용한 뒤, aec_image_cnn_exploratory의 CNN
    파이프라인(이미지화→학습→스택→AUC→하향조정)을 그대로 실행."""
    cfg = cnn.CnnConfig(max_epochs=80, patience=12)
    xg0 = base_transform(g["raw"], mode)
    xs0 = base_transform(s["raw"], mode)
    xg_signal, xs_signal = company_template_harmonize_train(xg0, xs0, g["company"], s["company"])
    img_g, img_s, _ = cnn.make_images(xg_signal, xs_signal, cfg)
    cnn_oof, cnn_ext, train_log = cnn.crossfit_cnn(img_g, g["y"].astype(int), img_s, cfg)
    train_log.insert(0, "preprocessing", mode)

    yg = g["y"].astype(int)
    ys = s["y"].astype(int)
    stack = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=cnn.SEEDS[0])
    stack.fit(np.column_stack([cg, cnn_oof]), yg)
    fusion_oof = stack.decision_function(np.column_stack([cg, cnn_oof]))
    fusion_ext = stack.decision_function(np.column_stack([cs, cnn_ext]))

    clinical_th = threshold_youden(yg, cg)
    cnn_th = threshold_youden(yg, cnn_oof)
    fusion_th = threshold_youden(yg, fusion_oof)
    auc_rows = []
    for row in [
        cnn.model_metrics("Gangnam internal OOF", "clinical", yg, cg, clinical_th),
        cnn.model_metrics("Sinchon external", "clinical", ys, cs, clinical_th),
        cnn.model_metrics("Gangnam internal OOF", "aec_image_cnn", yg, cnn_oof, cnn_th),
        cnn.model_metrics("Sinchon external", "aec_image_cnn", ys, cnn_ext, cnn_th),
        {
            **cnn.model_metrics("Gangnam internal OOF", "clinical_plus_aec_image_cnn", yg, fusion_oof, fusion_th),
            **cnn.bootstrap_auc_delta(yg, cg, fusion_oof, seed=cnn.SEEDS[0] + 30),
        },
        {
            **cnn.model_metrics("Sinchon external", "clinical_plus_aec_image_cnn", ys, fusion_ext, fusion_th),
            **cnn.bootstrap_auc_delta(ys, cs, fusion_ext, seed=cnn.SEEDS[0] + 31),
        },
    ]:
        auc_rows.append({"preprocessing": mode, **row})

    deesc_rows = []
    for op, target in cnn.OPS:
        cth = threshold_for_min_sensitivity(yg, cg, target)
        cpos_g = cg >= cth
        cpos_s = cs >= cth
        ath, train_row = cnn.choose_deesc_threshold(yg, cpos_g, cnn_oof, op)
        deesc_rows.append({"preprocessing": mode, **train_row})
        deesc_rows.append({"preprocessing": mode, **cnn.deesc_row("Sinchon external", op, ys, cpos_s, cnn_ext, ath)})
    return auc_rows, deesc_rows, train_log


def summarize_deesc(details: pd.DataFrame) -> pd.DataFrame:
    """전처리 모드x데이터셋별로 여러 민감도 목표에 걸친 최소/평균 하향조정 지표를 요약."""
    return (
        details.groupby(["preprocessing", "dataset"])
        .agg(
            min_p_loss=("sensitivity_loss_p_exact", "min"),
            max_sens_loss=("sensitivity_loss", "max"),
            min_spec_gain=("specificity_gain", "min"),
            mean_spec_gain=("specificity_gain", "mean"),
            min_delta_ba=("delta_balanced_accuracy", "min"),
            mean_delta_ba=("delta_balanced_accuracy", "mean"),
            max_fisher_p=("deesc_event_fisher_p", "max"),
            min_deesc_event_rate=("deesc_event_rate", "min"),
            max_deesc_event_rate=("deesc_event_rate", "max"),
            mean_deesc_event_rate=("deesc_event_rate", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    """
    이 스크립트의 핵심 실행 흐름 (질문: aec_vendor_neutral_preprocessing_audit에서 만든 "회사별
    템플릿 보정"을 실제로 이미지 CNN 파이프라인에 적용하면, 장비 흔적을 지운 채로도 CNN 성능이
    유지되는가?):

    1. g1090/sdata의 원시 곡선과 제조사 범주를 로드하고 임상점수를 준비한다.
    2. 3가지 전처리 모드(평균정규화/로그중심화/순위모양, 모두 "회사 보정" 버전)마다 evaluate_mode를
       실행: base_transform으로 기본 표현을 만들고, company_template_harmonize_train으로 회사별
       평균을 전체 평균으로 치환한 뒤, CNN 이미지화→학습→임상점수 스택→AUC 비교→하향조정 분석까지
       aec_image_cnn_exploratory의 파이프라인을 그대로 수행.
    3. 3개 모드의 AUC 결과, 하향조정 상세·요약, 학습로그를 각각 모아 CSV로 저장.
    4. 모드별 AUC/하향조정 요약을 콘솔에 출력해, 회사 보정을 걸어도 성능이 유지되는지 확인.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_path = WORK_DATA_DIR / "g1090.xlsx" if (WORK_DATA_DIR / "g1090.xlsx").exists() else DATA_DIR / "g1090.xlsx"
    s_path = WORK_DATA_DIR / "sdata.xlsx" if (WORK_DATA_DIR / "sdata.xlsx").exists() else DATA_DIR / "sdata.xlsx"
    g = load_raw_dataset(g_path)
    s = load_raw_dataset(s_path)
    cg, cs, _ = clinical_scores(g, s)
    modes = [
        "smooth_mean_norm_company_harmonized",
        "smooth_log_centered_company_harmonized",
        "smooth_rank_shape_company_harmonized",
    ]
    auc_rows: list[dict] = []
    deesc_rows: list[dict] = []
    logs = []
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        a, d, log = evaluate_mode(mode, g, s, cg, cs)
        auc_rows.extend(a)
        deesc_rows.extend(d)
        logs.append(log)
        print(pd.DataFrame(a)[["dataset", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot"]].to_string(index=False))

    auc = pd.DataFrame(auc_rows)
    deesc = pd.DataFrame(deesc_rows)
    summary = summarize_deesc(deesc)
    auc.to_csv(OUT_DIR / "company_harmonized_cnn_auc_metrics.csv", index=False)
    deesc.to_csv(OUT_DIR / "company_harmonized_cnn_deescalation_details.csv", index=False)
    summary.to_csv(OUT_DIR / "company_harmonized_cnn_deescalation_summary.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(OUT_DIR / "company_harmonized_cnn_training_log.csv", index=False)

    print("\nAUC SUMMARY")
    print(auc[["preprocessing", "dataset", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot"]].to_string(index=False))
    print("\nDE-ESCALATION SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 회사(제조사) 템플릿 보정을 적용한 3가지 전처리 모드마다 이미지 CNN
    # 파이프라인을 재실행해 AUC와 하향조정 지표가 유지되는지 확인하는 흐름이 수행된다.
    main()
