from __future__ import annotations

# Combined branch ablation: does ClinicalBranch need to be an MLP, and does AecBranch
# need to be a plain global-avg-pooled 1D-CNN? Both share one data-loading pass; results
# go to one .xlsx with each family on its own sheet.
#
# Clinical variants (AecBranch held at stage2_model's default "convpool"), all three
# trained fresh here (mlp is no longer read from a cached reference -- see 2026-07-22
# note below):
#   - mlp:        ClinicalBranch's original 2-layer MLP (Linear(4,32)+ReLU+Dropout+
#                 Linear(32,16)+ReLU)
#   - linear:     ClinicalBranch with the hidden layer/activation stripped -- single
#                 Linear(4,16), still jointly trained, just no nonlinearity
#   - frozen_lr:  no learned clinical branch at all -- z_clin IS the frozen Stage-1 LR
#                 score itself (stage1_rows["score"]), standardized the same way as
#                 stage2_model.py's own default (see fit_score_standardizer)
#
# NOTE (2026-07-22): this script used to skip retraining "mlp" and instead relabel
# outputs/stage2_model/clinical_vs_aec_assisted_summary.csv as the "mlp" row. That file
# reflects whatever CLIN_BRANCH_VARIANT is stage2_model.py's *current* default -- which
# had since become "frozen_lr" -- so the old "mlp" row was silently showing frozen_lr's
# numbers twice. Fixed by actually training "mlp" here instead of reusing a reference.
# Also fixed: build_clin_tensor's frozen_lr case fed the raw (unstandardized) Stage-1
# score, and run_aec_variant fit the score standardizer separately per cohort instead of
# fitting on internal and freezing for external -- both now match stage2_model.py's
# fit_score_standardizer/_to_tensors pattern.
#
# AEC variants (clinical branch held at stage2_model.CLIN_BRANCH_VARIANT, "frozen_lr",
# the clinical ablation's winner):
#   - convpool:        current stage2_model.py default (reused from its existing
#                      2026-07-21 run, not retrained here)
#   - convpool_avgmax: same conv stack, but pools with both AdaptiveAvgPool1d AND
#                      AdaptiveMaxPool1d (concatenated) -- does the curve's peak
#                      (value/sharpness) carry information the average level discards?
#   - handcrafted:     no CNN at all -- mean/std/slope/AUC/peak-value/peak-location/
#                      curvature descriptors of the whole curve through a single Linear
#                      layer, mirroring the clinical branch's "linear" variant but for AEC
#   - convflat:        same conv stack as convpool, but skips pooling and flattens the
#                      full (16 channels x 128 slices) feature map into the fc layer --
#                      the CNN's own directly-extracted per-position features, unreduced
#
# All variants share the same OOF/frozen-external evaluation protocol as
# stage2_model.py main() -- only the branch under test differs.
#
# Run: python code/stage2_model_branch_ablation.py

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2 = import_module("stage2_dataset")
m = import_module("stage2_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_model_branch_ablation"
REFERENCE_DIR = PROJECT_ROOT / "outputs" / "stage2_model"

CLIN_VARIANTS = ["mlp", "linear", "frozen_lr"]
AEC_VARIANTS = ["convpool_avgmax", "handcrafted", "convflat"]

SUMMARY_COLS = ["variant", "cohort", "n_params", "n", "event", "auc",
                "sens_clin", "sens_aec", "sens_p", "spec_clin", "spec_aec", "spec_p",
                "acc_clin", "acc_aec", "acc_p", "net_nri"]


def build_clin_tensor(variant: str, stage2_input_clin: pd.DataFrame, stage1_rows_pos: pd.DataFrame,
                       score_standardizer: tuple[float, float] | None = None) -> torch.Tensor:
    if variant == "frozen_lr":
        score = stage1_rows_pos["score"].to_numpy(dtype=np.float32)
        mean, std = score_standardizer if score_standardizer is not None else m.fit_score_standardizer(score)
        return torch.tensor((score - mean) / std, dtype=torch.float32).unsqueeze(1)
    return torch.tensor(stage2_input_clin[m.CLIN_COLS].to_numpy(dtype=np.float32))


def _evaluate_variant(screen: dict, data_int: tuple, data_ext: tuple,
                       x_clin_int: torch.Tensor, x_aec_int: torch.Tensor,
                       x_clin_ext: torch.Tensor, x_aec_ext: torch.Tensor,
                       oof: np.ndarray, model) -> list[dict]:
    stage1_rows_all_int, stage1_rows_int, _, _ = data_int
    stage1_rows_all_ext, stage1_rows_ext, _, _ = data_ext

    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)

    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    th = m.choose_stage2_threshold(y_all_int, pos_mask_int, oof, stage1_only_int["sens"], stage1_only_int["spec"])

    score_ext = model.predict_proba(x_clin_ext, x_aec_ext)

    pos_mask_ext = stage1_rows_all_ext["group"].isin(["TP", "FP"]).to_numpy()
    y_all_ext, _, pred_all_ext = m.final_pipeline_labels(stage1_rows_all_ext, score_ext, th)
    pred_all_int = m.combine_predictions(pos_mask_int, oof, th)

    stage1_only_ext = baseline.evaluate("external / stage-1 only", y_all_ext, pos_mask_ext, screen["th"])
    result_final_int = baseline.evaluate("internal", y_all_int, pred_all_int, th)
    result_final_ext = baseline.evaluate("external", y_all_ext, pred_all_ext, th)

    auc_int = baseline.auc_significance_stats(y_int, oof)
    auc_ext = baseline.auc_significance_stats(y_ext, score_ext)

    return [
        m.build_clinical_vs_aec_row("internal", y_all_int, pos_mask_int, pred_all_int, stage1_only_int, result_final_int, auc_int["auc"]),
        m.build_clinical_vs_aec_row("external", y_all_ext, pos_mask_ext, pred_all_ext, stage1_only_ext, result_final_ext, auc_ext["auc"]),
    ]


def run_clin_variant(variant: str, screen: dict, data_int: tuple, data_ext: tuple) -> dict:
    print(f"\n=== clinical variant: {variant} ===")
    _, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = data_int
    _, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = data_ext

    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    # Fit once on internal, freeze for external -- same frozen-transfer pattern as
    # stage2_model.py main() (fit_score_standardizer). Harmless no-op for mlp/linear,
    # which ignore this argument in build_clin_tensor.
    score_standardizer = m.fit_score_standardizer(stage1_rows_int["score"].to_numpy(dtype=np.float32))
    x_clin_int = build_clin_tensor(variant, stage2_input_clin_int, stage1_rows_int, score_standardizer)
    x_aec_int = torch.tensor(stage2_input_aec_int[m.AEC_COLS].to_numpy(dtype=np.float32))
    x_clin_ext = build_clin_tensor(variant, stage2_input_clin_ext, stage1_rows_ext, score_standardizer)
    x_aec_ext = torch.tensor(stage2_input_aec_ext[m.AEC_COLS].to_numpy(dtype=np.float32))

    oof, _ = m.oof_scores(x_clin_int, x_aec_int, y_int, clin_variant=variant)
    model, _ = m.fit_final_model(x_clin_int, x_aec_int, y_int, clin_variant=variant)

    rows = _evaluate_variant(screen, data_int, data_ext, x_clin_int, x_aec_int, x_clin_ext, x_aec_ext, oof, model)
    for r in rows:
        r["variant"] = variant
    return {"rows": rows, "n_params": sum(p.numel() for p in model.models[0].clin_branch.parameters())}


def run_aec_variant(variant: str, screen: dict, data_int: tuple, data_ext: tuple) -> dict:
    print(f"\n=== aec variant: {variant} ===")
    _, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = data_int
    _, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = data_ext

    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    # Clinical branch is held at the default "frozen_lr" here, so _to_tensors needs the
    # same internal-fit/external-frozen score standardizer as stage2_model.py main() --
    # fitting it separately per cohort (the old behavior) violates frozen-transfer.
    score_standardizer = m.fit_score_standardizer(stage1_rows_int["score"].to_numpy(dtype=np.float32))
    x_clin_int, x_aec_int = m._to_tensors(stage2_input_clin_int, stage2_input_aec_int, stage1_rows_int,
                                           score_standardizer=score_standardizer)
    x_clin_ext, x_aec_ext = m._to_tensors(stage2_input_clin_ext, stage2_input_aec_ext, stage1_rows_ext,
                                           score_standardizer=score_standardizer)

    oof, _ = m.oof_scores(x_clin_int, x_aec_int, y_int, aec_variant=variant)
    model, _ = m.fit_final_model(x_clin_int, x_aec_int, y_int, aec_variant=variant)

    rows = _evaluate_variant(screen, data_int, data_ext, x_clin_int, x_aec_int, x_clin_ext, x_aec_ext, oof, model)
    for r in rows:
        r["variant"] = variant
    return {"rows": rows, "n_params": sum(p.numel() for p in model.models[0].aec_branch.parameters())}


def load_reference(variant_label: str) -> list[dict]:
    df = pd.read_csv(REFERENCE_DIR / "clinical_vs_aec_assisted_summary.csv")
    df["variant"] = variant_label
    return df.to_dict("records")


def build_summary(rows: list[dict], n_params_by_variant: dict, n_params_col: str) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    summary["n_params"] = summary["variant"].map(n_params_by_variant)
    return summary[SUMMARY_COLS].rename(columns={"n_params": n_params_col})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    screen = stage2.fit_internal_screen()
    data_int = stage2.build_stage2_inputs(screen)
    data_ext = stage2.build_stage2_inputs_external(screen)

    # --- clinical branch ablation ---
    clin_rows: list[dict] = []
    n_params_clin: dict[str, int] = {}
    for variant in CLIN_VARIANTS:
        result = run_clin_variant(variant, screen, data_int, data_ext)
        clin_rows.extend(result["rows"])
        n_params_clin[variant] = result["n_params"]
    clin_summary = build_summary(clin_rows, n_params_clin, "clin_branch_n_params")

    # --- aec branch ablation ---
    aec_rows = load_reference("convpool")
    n_params_aec = {"convpool": None}
    for variant in AEC_VARIANTS:
        result = run_aec_variant(variant, screen, data_int, data_ext)
        aec_rows.extend(result["rows"])
        n_params_aec[variant] = result["n_params"]
    aec_summary = build_summary(aec_rows, n_params_aec, "aec_branch_n_params")

    out_path = OUTPUT_DIR / "branch_variant_comparison.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        clin_summary.to_excel(writer, sheet_name="clinical_branch", index=False)
        aec_summary.to_excel(writer, sheet_name="aec_branch", index=False)
    print(f"\nsaved: {out_path}")
    print("\n[clinical_branch]")
    print(clin_summary.to_string(index=False))
    print("\n[aec_branch]")
    print(aec_summary.to_string(index=False))


if __name__ == "__main__":
    main()
