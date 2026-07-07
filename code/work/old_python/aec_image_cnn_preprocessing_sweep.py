from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aec_image_cnn_exploratory as cnn  # noqa: E402
from aec_conditional_value import DATA_DIR, matrix_from_sheet, row_norm, threshold_youden  # noqa: E402
from aec_midrange_feature_refit import clinical_scores  # noqa: E402
from aec_universal_boundary_gate import threshold_for_min_sensitivity  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "outputs" / "0701" / "aec_image_cnn_preprocessing_sweep"


def load_raw_dataset(path: Path) -> dict:
    """엑셀에서 원시 AEC_128 행렬(정규화 전)과 정규화 버전, 라벨을 함께 읽어옴 (여러 전처리 모드를 이후 적용하기 위해 원시값도 보관)."""
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl")
    raw = matrix_from_sheet(pd.read_excel(path, sheet_name="aec_128", engine="openpyxl"))
    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    smi = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float) / (
        pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    ) ** 2
    y = np.where(sex == "M", smi < 45.4, smi < 34.4).astype(int)
    return {"meta": meta, "raw": raw, "norm": row_norm(raw), "y": y}


def patient_z(x: np.ndarray) -> np.ndarray:
    """환자 평균 나눗셈 정규화 대신, 환자별 z-표준화(평균0/표준편차1)로 정규화하는 대안 방식."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x - mu) / sd


def transform_signal(raw: np.ndarray, mode: str) -> np.ndarray:
    """지정된 전처리 모드(평균정규화/환자z, 평활화 유무·강도)에 따라 원시 곡선을 변환."""
    if mode == "mean_norm_no_smooth":
        return row_norm(raw)
    if mode == "mean_norm_smooth_after_sigma0p75":
        return ndimage.gaussian_filter1d(row_norm(raw), sigma=0.75, axis=1, mode="nearest")
    if mode == "mean_norm_smooth_after_sigma1p5":
        return ndimage.gaussian_filter1d(row_norm(raw), sigma=1.5, axis=1, mode="nearest")
    if mode == "patient_z_no_smooth":
        return patient_z(raw)
    if mode == "patient_z_smooth_after_sigma1":
        return ndimage.gaussian_filter1d(patient_z(raw), sigma=1.0, axis=1, mode="nearest")
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_mode(mode: str, g: dict, s: dict, cg: np.ndarray, cs: np.ndarray) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """한 전처리 모드에 대해 aec_image_cnn_exploratory의 CNN 파이프라인 전체(이미지화→학습→스택→
    AUC 지표→하향조정 분석)를 재사용해 실행하고, AUC 결과·하향조정 결과·점수·학습로그를 반환."""
    cfg = cnn.CnnConfig(max_epochs=80, patience=12)
    xg_signal = transform_signal(g["raw"], mode)
    xs_signal = transform_signal(s["raw"], mode)
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
            **cnn.bootstrap_auc_delta(yg, cg, fusion_oof, seed=cnn.SEEDS[0] + 10),
        },
        {
            **cnn.model_metrics("Sinchon external", "clinical_plus_aec_image_cnn", ys, fusion_ext, fusion_th),
            **cnn.bootstrap_auc_delta(ys, cs, fusion_ext, seed=cnn.SEEDS[0] + 11),
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

    score_g = pd.DataFrame(
        {
            "preprocessing": mode,
            "cohort": "Gangnam",
            "y": yg,
            "clinical": cg,
            "aec_image_cnn": cnn_oof,
            "clinical_plus_aec_image_cnn": fusion_oof,
        }
    )
    score_s = pd.DataFrame(
        {
            "preprocessing": mode,
            "cohort": "Sinchon",
            "y": ys,
            "clinical": cs,
            "aec_image_cnn": cnn_ext,
            "clinical_plus_aec_image_cnn": fusion_ext,
        }
    )
    return auc_rows, deesc_rows, pd.concat([score_g, score_s], ignore_index=True), train_log


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
    이 스크립트의 핵심 실행 흐름 (질문: 이미지 CNN 결과가 "정규화 방식(평균 나눗셈 vs z표준화)"과
    "평활화 강도"라는 전처리 선택에 얼마나 민감한가? — aec_image_cnn_exploratory의 강건성 점검):

    1. g1090/sdata의 원시(정규화 전) AEC_128 곡선과 임상점수를 로드한다.
    2. 5가지 전처리 모드(평균정규화 평활화없음/약한평활화/강한평활화, 환자z 평활화없음/평활화)
       각각에 대해 evaluate_mode로 aec_image_cnn_exploratory의 CNN 파이프라인 전체를 재실행:
       이미지화→CNN 학습→임상점수와 스택→AUC 비교→민감도 목표별 하향조정 분석까지 동일하게 수행.
    3. 5개 모드의 AUC 결과, 하향조정 상세결과·요약, 환자별 점수, 학습로그를 각각 모아 CSV로 저장.
    4. 모드별 AUC 요약과 하향조정 요약을 콘솔에 출력해, 전처리 선택에 따라 결론이 바뀌는지 확인.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = load_raw_dataset(DATA_DIR / "g1090.xlsx")
    s = load_raw_dataset(DATA_DIR / "sdata.xlsx")
    cg, cs, _ = clinical_scores(g, s)
    modes = [
        "mean_norm_no_smooth",
        "mean_norm_smooth_after_sigma0p75",
        "mean_norm_smooth_after_sigma1p5",
        "patient_z_no_smooth",
        "patient_z_smooth_after_sigma1",
    ]

    auc_rows: list[dict] = []
    deesc_rows: list[dict] = []
    scores = []
    logs = []
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        a, d, sc, log = evaluate_mode(mode, g, s, cg, cs)
        auc_rows.extend(a)
        deesc_rows.extend(d)
        scores.append(sc)
        logs.append(log)
        print(pd.DataFrame(a)[["dataset", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot"]].to_string(index=False))

    auc = pd.DataFrame(auc_rows)
    deesc = pd.DataFrame(deesc_rows)
    summary = summarize_deesc(deesc)
    auc.to_csv(OUT_DIR / "cnn_preprocessing_auc_metrics.csv", index=False)
    deesc.to_csv(OUT_DIR / "cnn_preprocessing_deescalation_details.csv", index=False)
    summary.to_csv(OUT_DIR / "cnn_preprocessing_deescalation_summary.csv", index=False)
    pd.concat(scores, ignore_index=True).to_csv(OUT_DIR / "cnn_preprocessing_scores.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(OUT_DIR / "cnn_preprocessing_training_log.csv", index=False)

    print("\nAUC SUMMARY")
    print(auc[["preprocessing", "dataset", "model", "auc", "auc_p_mannwhitney", "delta_auc", "delta_auc_p_boot"]].to_string(index=False))
    print("\nDE-ESCALATION SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    # 스크립트 실행 시 5가지 전처리 모드(정규화 방식 x 평활화 강도)마다 이미지 CNN 파이프라인 전체를
    # 재실행해 AUC와 하향조정 지표를 비교하는 전처리 민감도 스윕이 수행된다.
    main()
