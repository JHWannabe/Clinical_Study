from __future__ import annotations

# Cohort-swapped rerun of code/stage2_dataset.py: the original fits the Stage-1
# clinical-only screen on internal=gangnam.xlsx and transfers to external=sinchon.xlsx.
# This module swaps the roles -- internal=sinchon.xlsx (fit + OOF threshold),
# external=gangnam.xlsx (frozen-model transfer) -- reusing stage2_dataset.py's own
# generic helpers (_stage1_positive_rows, load_aec_for_patients, CLIN_COLS, AEC_COLS)
# which take an xlsx_path parameter and don't hardcode the cohort assignment.

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
baseline = importlib.import_module("clinic-only_baseline")
stage2 = importlib.import_module("stage2_dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

INTERNAL_XLSX = DATA_DIR / "sinchon.xlsx"
EXTERNAL_XLSX = DATA_DIR / "gangnam.xlsx"

N_SLICES = stage2.N_SLICES
AEC_COLS = stage2.AEC_COLS
CLIN_COLS = stage2.CLIN_COLS

load_aec_for_patients = stage2.load_aec_for_patients


def fit_internal_screen() -> dict:
    # Same as stage2_dataset.fit_internal_screen but fit on internal=sinchon.
    meta, y = baseline.load_cohort(INTERNAL_XLSX)
    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)
    oof = baseline.oof_scores(x, y)
    th = baseline.threshold_for_sensitivity(y, oof, baseline.TARGET_SENSITIVITY)
    model = baseline.fit_baseline_model(x, y)
    print(f"[internal=sinchon, S{int(baseline.TARGET_SENSITIVITY * 100)}] threshold={th:.4f}")
    return {"meta": meta, "y": y, "x": x, "oof": oof, "th": th, "med": med, "mu": mu, "sd": sd, "model": model}


def build_stage2_inputs(screen: dict | None = None) -> tuple:
    if screen is None:
        screen = fit_internal_screen()
    return stage2._stage1_positive_rows(INTERNAL_XLSX, screen["meta"], screen["y"], screen["oof"], screen["th"], screen["x"])


def build_stage2_inputs_external(screen: dict) -> tuple:
    meta_ext, y_ext = baseline.load_cohort(EXTERNAL_XLSX)
    x_ext = baseline.apply_clinical_standardizer(baseline.raw_clinical_matrix(meta_ext), screen["med"], screen["mu"], screen["sd"])
    score_ext = screen["model"].decision_function(x_ext)
    return stage2._stage1_positive_rows(EXTERNAL_XLSX, meta_ext, y_ext, score_ext, screen["th"], x_ext)
