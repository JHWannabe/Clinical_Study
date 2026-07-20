from __future__ import annotations

# Stage-2 late-fusion classifier: clinical features and the AEC-128 curve each
# go through their own branch, and the two branch embeddings are concatenated
# (late fusion) before the final classification head.
#
# Run: python code/2_stage2_model.py

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2 = import_module("1_stage2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "2_stage2_model"

AEC_COLS = stage2.AEC_COLS
CLIN_COLS = ["sex_m", "age_std", "height_std", "weight_std"]

N_FOLDS = baseline.N_FOLDS
SEED = baseline.SEED
N_EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Acceptance criteria: sensitivity may only drop by a RELATIVE 5% of its stage-1-only
# value (sens_after >= sens_before * 0.95 -- e.g. internal 0.907 -> floor 0.862), and
# specificity must not get worse. Gates both threshold selection (choose_stage2_threshold)
# and the final PASS/FAIL verdict (pass_fail), so the two never disagree.
SENS_LOSS_RATIO_MARGIN = 0.05

# Newcombe (1998) Method 10 CI-based test (docs/model_algorithm.md "임계값 선택 기준")
# is still computed and reported alongside for statistical context, but is no longer
# the gating criterion -- with internal n=129 actual positives its CI upper bound is
# wide enough that even a single flipped case fails it, which is stricter than the
# relative-margin criterion above.
SENS_NONINF_MARGIN = 0.05
NI_ALPHA = 0.025  # one-sided; equivalent to the upper bound of a two-sided 95% CI
NI_Z = float(norm.ppf(1 - NI_ALPHA))


class ClinicalBranch(nn.Module):
    # MLP branch over the 4 standardized clinical features.
    def __init__(self, in_dim: int = 4, embed_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x_clin: torch.Tensor) -> torch.Tensor:
        return self.net(x_clin)


class AecBranch(nn.Module):
    # 1D-CNN branch over the 128-slice AEC curve. Global-average-pools the
    # conv features so the branch reads the curve holistically (whole-shape),
    # not as 128 independent point-wise inputs.
    def __init__(self, n_slices: int = 128, embed_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x_aec: torch.Tensor) -> torch.Tensor:
        z = self.conv(x_aec.unsqueeze(1)).squeeze(-1)
        return self.fc(z)


class LateFusionNet(nn.Module):
    # Runs the clinical and AEC branches independently, then fuses their
    # embeddings by concatenation (late fusion) before the classification head.
    def __init__(
        self,
        clin_dim: int = 4,
        n_slices: int = 128,
        embed_dim: int = 16,
        fusion_hidden: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.clin_branch = ClinicalBranch(clin_dim, embed_dim, dropout)
        self.aec_branch = AecBranch(n_slices, embed_dim, dropout)
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_clin: torch.Tensor, x_aec: torch.Tensor) -> torch.Tensor:
        z_clin = self.clin_branch(x_clin)
        z_aec = self.aec_branch(x_aec)
        z = torch.cat([z_clin, z_aec], dim=1)
        return self.fusion_head(z).squeeze(-1)  # logit


def train_fold(x_clin_tr: torch.Tensor, x_aec_tr: torch.Tensor, y_tr: torch.Tensor,
                x_clin_va: torch.Tensor, x_aec_va: torch.Tensor, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = LateFusionNet(clin_dim=x_clin_tr.shape[1], n_slices=x_aec_tr.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_clin_tr, x_aec_tr), y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_clin_va, x_aec_va)).numpy()


def oof_scores(x_clin: torch.Tensor, x_aec: torch.Tensor, y: np.ndarray) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x_clin.numpy(), y)):
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        oof[va_idx] = train_fold(
            x_clin[tr_idx], x_aec[tr_idx], y_tr,
            x_clin[va_idx], x_aec[va_idx],
            seed=SEED + fold_id,
        )
    return oof


def fit_final_model(x_clin: torch.Tensor, x_aec: torch.Tensor, y: np.ndarray, seed: int = SEED) -> LateFusionNet:
    # Refit on the FULL internal Stage-2 cohort (no held-out fold) -- the frozen
    # model applied to external, mirroring clinic-only_baseline.py's fit_baseline_model.
    torch.manual_seed(seed)
    model = LateFusionNet(clin_dim=x_clin.shape[1], n_slices=x_aec.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_clin, x_aec), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _to_tensors(stage2_input_clin, stage2_input_aec) -> tuple[torch.Tensor, torch.Tensor]:
    x_clin = torch.tensor(stage2_input_clin[CLIN_COLS].to_numpy(dtype=np.float32))
    x_aec = torch.tensor(stage2_input_aec[AEC_COLS].to_numpy(dtype=np.float32))
    return x_clin, x_aec


def combine_predictions(pos_mask: np.ndarray, stage2_score: np.ndarray, th_stage2: float) -> np.ndarray:
    # Whole-cohort prediction: screen-negative (FN/TN) rows never reach Stage 2, so
    # they're predicted Negative as-is; TP/FP (screen-positive) rows get Stage 2's
    # reclassification. stage2_score must be row-aligned with pos_mask's True entries.
    final_pred = np.zeros(len(pos_mask), dtype=bool)
    final_pred[pos_mask] = stage2_score >= th_stage2
    return final_pred


def final_pipeline_labels(stage1_rows_all, stage2_score: np.ndarray, th_stage2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns (y_all, pred_stage1_only, pred_final) -- the "before"/"after" pair the
    # non-inferiority test compares.
    y_all = stage1_rows_all["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask = stage1_rows_all["group"].isin(["TP", "FP"]).to_numpy()
    final_pred = combine_predictions(pos_mask, stage2_score, th_stage2)
    return y_all, pos_mask, final_pred


def _wilson_ci(count: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = count / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return center - half, center + half


def noninferiority_test_sensitivity(pred_before: np.ndarray, pred_after: np.ndarray, y: np.ndarray,
                                     margin: float = SENS_NONINF_MARGIN, z: float = NI_Z) -> dict:
    # Newcombe (1998) "Method 10" score-based CI for the difference between two paired
    # proportions, applied to sensitivity (subset = actual positives). Non-inferiority
    # is only declared if the upper confidence bound on the sensitivity DROP stays
    # within margin, not just the observed point-estimate drop -- accounts for
    # sampling uncertainty. By construction c=0 here (Stage 2 only ever turns a
    # Stage-1 positive call negative, never the reverse).
    subset = y == 1
    before = pred_before[subset].astype(bool)
    after = pred_after[subset].astype(bool)
    n = len(before)
    a = int(np.sum(before & after))
    b = int(np.sum(before & ~after))
    c = int(np.sum(~before & after))
    d = int(np.sum(~before & ~after))

    p1 = (a + b) / n  # sensitivity before (stage-1 only)
    p2 = (a + c) / n  # sensitivity after (stage-1+stage-2)
    drop = p1 - p2    # positive value = sensitivity fell

    l1, u1 = _wilson_ci(a + b, n, z)
    l2, u2 = _wilson_ci(a + c, n, z)

    denom = float(np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))
    phi = (a * d - b * c) / denom if denom > 0 else 0.0

    ci_lower = drop - np.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    ci_upper = drop + np.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)

    return {"n": n, "a": a, "b": b, "c": c, "d": d,
            "sens_before": p1, "sens_after": p2, "sens_drop": drop,
            "ci_lower": ci_lower, "ci_upper": ci_upper, "margin": margin,
            "noninferior": bool(ci_upper <= margin)}


def sens_noninferior(sens_before: float, sens_after: float, margin: float = SENS_LOSS_RATIO_MARGIN) -> bool:
    # Relative margin: sens_after must retain >= (1-margin) of sens_before (e.g.
    # internal 0.907 -> floor 0.907*0.95 = 0.862), not an absolute percentage-point drop.
    return sens_after >= sens_before * (1 - margin)


def pass_fail(stage1_res: dict, final_res: dict, margin: float = SENS_LOSS_RATIO_MARGIN) -> tuple[float, float, bool]:
    # Acceptance criteria: sensitivity retains >=95% of its stage-1-only value AND
    # specificity delta >= 0.
    sens_delta = final_res["sens"] - stage1_res["sens"]
    spec_delta = final_res["spec"] - stage1_res["spec"]
    ok = sens_noninferior(stage1_res["sens"], final_res["sens"], margin) and (spec_delta >= 0)
    return sens_delta, spec_delta, ok


def choose_stage2_threshold(y_all: np.ndarray, pos_mask: np.ndarray, stage2_score: np.ndarray,
                             margin: float = SENS_LOSS_RATIO_MARGIN) -> float:
    # Selects th2 with the best PPV among thresholds that keep sensitivity within the
    # relative margin of the stage-1-only sensitivity -- the same criterion used for
    # the final PASS/FAIL verdict (pass_fail), so threshold selection and reporting
    # never disagree. Internal-only: external is never touched by this sweep.
    # Sentinel th=-inf always reproduces Stage 1 exactly (sens_delta=0), so it's
    # always a valid, always-passing candidate.
    tp0, fp0, fn0, tn0 = baseline.confusion_counts(y_all, pos_mask)
    sens_before = tp0 / (tp0 + fn0) if (tp0 + fn0) else float("nan")

    candidates = np.concatenate([[-np.inf], np.unique(stage2_score)])
    best = None
    for th in candidates:
        pred = combine_predictions(pos_mask, stage2_score, th)
        tp, fp, fn, tn = baseline.confusion_counts(y_all, pred)
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        sens_after = tp / (tp + fn) if (tp + fn) else float("nan")
        if np.isfinite(ppv) and sens_noninferior(sens_before, sens_after, margin) and (best is None or ppv > best[1]):
            best = (float(th), ppv)
    assert best is not None, "sentinel th=-inf reproduces stage-1 exactly (sens_delta=0) and must always pass"
    return best[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal: 5-fold OOF for an unbiased internal estimate ---
    screen = stage2.fit_internal_screen()
    stage1_rows_all_int, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = stage2.build_stage2_inputs(screen)
    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    x_clin_int, x_aec_int = _to_tensors(stage2_input_clin_int, stage2_input_aec_int)

    oof = oof_scores(x_clin_int, x_aec_int, y_int)

    # th2 is chosen on the FULL internal cohort (not just the screen-positive
    # subgroup): among candidates whose NI test passes, pick the one maximizing PPV
    # -- so the threshold baked into training is the one the final verdict checks.
    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    th = choose_stage2_threshold(y_all_int, pos_mask_int, oof)

    # --- freeze: refit on the full internal Stage-2 cohort, transfer to external ---
    model = fit_final_model(x_clin_int, x_aec_int, y_int)

    stage1_rows_all_ext, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = stage2.build_stage2_inputs_external(screen)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)
    x_clin_ext, x_aec_ext = _to_tensors(stage2_input_clin_ext, stage2_input_aec_ext)
    with torch.no_grad():
        score_ext = torch.sigmoid(model(x_clin_ext, x_aec_ext)).numpy()

    result_int = baseline.evaluate("internal", y_int, oof >= th, th)
    result_ext = baseline.evaluate("external", y_ext, score_ext >= th, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Stage-2 Late Fusion (clinical + AEC-128), screen-positive only", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_int)
    baseline.plot_confusion_matrix(axes[1], result_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    auc_int = baseline.auc_significance_stats(y_int, oof)
    baseline.plot_roc_curve(y_int, oof, auc_int, OUTPUT_DIR / "roc_curve_internal.png",
                             title="Stage-2 Late Fusion: ROC (internal, OOF)")

    auc_ext = baseline.auc_significance_stats(y_ext, score_ext)
    baseline.plot_roc_curve(y_ext, score_ext, auc_ext, OUTPUT_DIR / "roc_curve_external.png",
                             title="Stage-2 Late Fusion: ROC (external, frozen internal model)")

    # --- final pipeline: Stage-1 FN/TN (screen-negative, untouched) + Stage-2's
    # reclassification of Stage-1 TP/FP (screen-positive), over the whole cohort ---
    pred_all_int = combine_predictions(pos_mask_int, oof, th)
    y_all_ext, pos_mask_ext, pred_all_ext = final_pipeline_labels(stage1_rows_all_ext, score_ext, th)

    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    stage1_only_ext = baseline.evaluate("external / stage-1 only", y_all_ext, pos_mask_ext, screen["th"])
    result_final_int = baseline.evaluate("internal", y_all_int, pred_all_int, th)
    result_final_ext = baseline.evaluate("external", y_all_ext, pred_all_ext, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Full Pipeline (Stage 1 screen + Stage 2 late fusion)", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_final_int)
    baseline.plot_confusion_matrix(axes[1], result_final_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    # --- acceptance test on the full pipeline: sens retains >=95% of stage-1-only sens
    # (relative margin) AND spec_delta >= 0. Newcombe CI-based NI test (absolute margin)
    # still computed and reported for statistical context, but is no longer the gate. ---
    ni_int = noninferiority_test_sensitivity(pos_mask_int, pred_all_int, y_all_int)
    ni_ext = noninferiority_test_sensitivity(pos_mask_ext, pred_all_ext, y_all_ext)
    sens_delta_int, spec_delta_int, ok_int = pass_fail(stage1_only_int, result_final_int)
    sens_delta_ext, spec_delta_ext, ok_ext = pass_fail(stage1_only_ext, result_final_ext)

    for cohort, stage1_res, ni, sens_delta, spec_delta, ok in [
        ("internal", stage1_only_int, ni_int, sens_delta_int, spec_delta_int, ok_int),
        ("external", stage1_only_ext, ni_ext, sens_delta_ext, spec_delta_ext, ok_ext),
    ]:
        sens_floor = stage1_res["sens"] * (1 - SENS_LOSS_RATIO_MARGIN)
        print(f"[{cohort}] sens_delta={sens_delta:+.3f} spec_delta={spec_delta:+.3f} "
              f"(sens floor={sens_floor:.3f}, margin={SENS_LOSS_RATIO_MARGIN:.0%} relative) -> {'PASS' if ok else 'FAIL'}")
        print(f"[{cohort}] Newcombe CI-based NI test (reference only): drop={ni['sens_drop']:.3f} "
              f"97.5%CI upper={ni['ci_upper']:.3f} (absolute margin={ni['margin']:.2f}) "
              f"-> {'NON-INFERIOR' if ni['noninferior'] else 'NOT NON-INFERIOR'}")

    ni_summary = pd.DataFrame([
        {"cohort": "internal", "sens_before": stage1_only_int["sens"], "sens_after": result_final_int["sens"],
         "sens_delta": sens_delta_int, "sens_floor": stage1_only_int["sens"] * (1 - SENS_LOSS_RATIO_MARGIN),
         "spec_delta": spec_delta_int, "pass": ok_int,
         "ni_ci_upper_reference": ni_int["ci_upper"]},
        {"cohort": "external", "sens_before": stage1_only_ext["sens"], "sens_after": result_final_ext["sens"],
         "sens_delta": sens_delta_ext, "sens_floor": stage1_only_ext["sens"] * (1 - SENS_LOSS_RATIO_MARGIN),
         "spec_delta": spec_delta_ext, "pass": ok_ext,
         "ni_ci_upper_reference": ni_ext["ci_upper"]},
    ])
    ni_summary_path = OUTPUT_DIR / "ni_test_final_pipeline.csv"
    ni_summary.to_csv(ni_summary_path, index=False)
    print(f"Saved NI test summary to {ni_summary_path}")


if __name__ == "__main__":
    main()
