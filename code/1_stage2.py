from __future__ import annotations

# Stage-1 screen (clinic-only_baseline.py) output prep for Stage 2.
#
# Refits the clinical-only LR on the internal cohort (5-fold OOF), picks the
# threshold at Sensitivity>=90%, and saves the internal patients the screen
# calls Positive, split into TP (actual low-SMI) and FP (actual non-low-SMI)
# rows, for downstream Stage-2 work.
#
# Run: python code/1_stage2.py

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "1_stage2"

INTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"
EXTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"

N_SLICES = 128
AEC_COLS = [f"aec_{i}" for i in range(1, N_SLICES + 1)]
CLIN_COLS = ["sex_m", "age_std", "height_std", "weight_std"]


def load_aec_for_patients(xlsx_path: Path, patient_ids: pd.Series) -> pd.DataFrame:
    # Patient-normalized AEC-128 curves (same convention as aec_curve_comparison.py /
    # the deleted 1_aec_residual_reclassify.py), restricted to the given PatientIDs.
    aec = pd.read_excel(xlsx_path, sheet_name="aec_128", engine="openpyxl")
    curves_raw = aec[AEC_COLS].astype(float).to_numpy()
    patient_mean = curves_raw.mean(axis=1, keepdims=True)
    norm_curves = curves_raw / patient_mean

    aec_df = pd.DataFrame(norm_curves, columns=AEC_COLS)
    aec_df.insert(0, "PatientID", aec["PatientID"].to_numpy())

    order = pd.DataFrame({"PatientID": patient_ids.to_numpy(), "__row__": np.arange(len(patient_ids))})
    merged = order.merge(aec_df, on="PatientID", how="left").sort_values("__row__").drop(columns="__row__")

    missing = int(merged[AEC_COLS].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} patients in {xlsx_path.name} have no matching aec_128 row")
    return merged.reset_index(drop=True)


def _stage1_positive_rows(xlsx_path: Path, meta: pd.DataFrame, y: np.ndarray, score: np.ndarray, th: float,
                           x_std: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Shared by internal and external: builds the full TP/FN/FP/TN group table, then
    # splits out the TP/FP screen-positive rows and attaches the row-aligned clinical
    # features / AEC-128 curves for those patients (Stage 2's inputs). FN/TN rows
    # (screen-negative) never reach Stage 2 -- they're returned in stage1_rows_all so
    # the final pipeline confusion matrix can still account for them.
    rows = baseline.build_group_rows(meta, y, score, th)
    mask = rows["group"].isin(["TP", "FP"]).to_numpy()
    stage1_rows = rows[mask].reset_index(drop=True)
    print(f"Stage-1 rows ({xlsx_path.name}): TP={int((stage1_rows['group'] == 'TP').sum())}, FP={int((stage1_rows['group'] == 'FP').sum())}")

    aec = load_aec_for_patients(xlsx_path, stage1_rows["PatientID"])
    print(f"Loaded AEC-128 curves for Stage-1 cohort: {aec.shape}")

    x_df = pd.DataFrame(x_std, columns=CLIN_COLS)
    x_df.insert(0, "PatientID", meta["PatientID"].to_numpy())
    stage2_input_clin = x_df[mask].reset_index(drop=True)

    assert (stage2_input_clin["PatientID"].to_numpy() == aec["PatientID"].to_numpy()).all()
    stage2_input_aec = aec
    print(f"stage2_input_clin shape: {stage2_input_clin.shape}, stage2_input_aec shape: {stage2_input_aec.shape}")

    return rows, stage1_rows, stage2_input_clin, stage2_input_aec


def fit_internal_screen() -> dict:
    # Fits the clinical-only LR screen on internal only: standardizer (med/mu/sd),
    # 5-fold OOF (for the S90 threshold), and a model refit on the full internal
    # cohort -- the frozen artifacts external Stage-1 screening reuses.
    meta, y = baseline.load_cohort(INTERNAL_XLSX)
    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)
    oof = baseline.oof_scores(x, y)
    th = baseline.threshold_for_sensitivity(y, oof, baseline.TARGET_SENSITIVITY)
    model = baseline.fit_baseline_model(x, y)
    print(f"[internal, S{int(baseline.TARGET_SENSITIVITY * 100)}] threshold={th:.4f}")
    return {"meta": meta, "y": y, "x": x, "oof": oof, "th": th, "med": med, "mu": mu, "sd": sd, "model": model}


def build_stage2_inputs(screen: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Internal Stage-2 inputs: (stage1_rows_all, stage1_rows_pos, stage2_input_clin,
    # stage2_input_aec). stage1_rows_all covers all four TP/FN/FP/TN groups (needed
    # for the final pipeline confusion matrix); the rest are TP/FP-only and row-aligned
    # so Stage 2 can feed them into a model as two branches. Pass an already-fit
    # `screen` (fit_internal_screen()) to avoid refitting the clinical LR when the
    # caller also needs it for external transfer.
    if screen is None:
        screen = fit_internal_screen()
    return _stage1_positive_rows(INTERNAL_XLSX, screen["meta"], screen["y"], screen["oof"], screen["th"], screen["x"])


def build_stage2_inputs_external(screen: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # External Stage-2 inputs: applies the FROZEN internal standardizer, model, and
    # threshold (from fit_internal_screen()) to sinchon.xlsx, so the external screen
    # is the internal model transferred, not refit on external data. Same return shape
    # as build_stage2_inputs (stage1_rows_all, stage1_rows_pos, clin, aec).
    meta_ext, y_ext = baseline.load_cohort(EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), screen["med"], screen["mu"], screen["sd"])
    score_ext = screen["model"].predict_proba(x_ext)[:, 1]
    return _stage1_positive_rows(EXTERNAL_XLSX, meta_ext, y_ext, score_ext, screen["th"], x_ext)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    screen = fit_internal_screen()
    build_stage2_inputs(screen)
    build_stage2_inputs_external(screen)


if __name__ == "__main__":
    main()
