from __future__ import annotations

# Stage-2 late-fusion classifier: clinical features and the AEC-128 curve each
# go through their own branch, and the two branch embeddings are concatenated
# (late fusion) before the final classification head.
#
# Run: python code/stage2_model.py

import copy
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2 = import_module("stage2_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_model"

AEC_COLS = stage2.AEC_COLS
CLIN_COLS = ["sex_m", "age_std", "height_std", "weight_std"]

N_FOLDS = baseline.N_FOLDS
SEED = baseline.SEED
N_EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
# ReduceLROnPlateau on the training loss: halves LR after PLATEAU_PATIENCE epochs
# without improvement, to damp the late-training loss oscillation from pos_weight's
# amplified minority-class gradient. Patience is kept below EARLY_STOP_PATIENCE so
# LR has a chance to drop before early stopping fires, and MIN_DELTA is shared with
# the early-stop criterion so a lower LR doesn't make "improvement" easier to miss.
LR_PLATEAU_PATIENCE = 12
LR_PLATEAU_FACTOR = 0.5

# NI test acceptance criteria vs. stage-1-only: sensitivity may only drop by a
# RELATIVE 5% of its stage-1-only value (sens_after >= sens_before * 0.95), and
# specificity must not get worse (spec_delta >= 0).
SENS_LOSS_RATIO_MARGIN = 0.05

# Early stopping: halt once the monitored signal stops improving by at least
# MIN_DELTA for PATIENCE consecutive epochs, then restore the best-so-far weights.
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-4

# Seed ensemble size: each fold (and the final refit) trains this many
# independently-initialized models and averages their sigmoid outputs, to damp
# the seed-to-seed variance visible in loss_curve.png (e.g. fold 3 vs. the rest)
# without needing more data or a longer per-model training run.
N_ENSEMBLE_SEEDS = 5


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


class LateFusionEnsemble:
    # N_ENSEMBLE_SEEDS frozen LateFusionNets averaged at the probability level --
    # the frozen artifact fit_final_model hands to external scoring.
    def __init__(self, models: list[LateFusionNet]) -> None:
        self.models = models

    def predict_proba(self, x_clin: torch.Tensor, x_aec: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = [torch.sigmoid(m(x_clin, x_aec)).numpy() for m in self.models]
        return np.mean(probs, axis=0)


def _make_criterion(y_tr: torch.Tensor) -> nn.Module:
    # pos_weight = n_neg/n_pos (recomputed per call, e.g. differs slightly across
    # folds and for the full-cohort refit) counters the ~21% TP prevalence within
    # the screen-positive cohort so the minority (TP) class isn't underweighted.
    n_pos = float(y_tr.sum().item())
    n_neg = float(y_tr.shape[0]) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _train_with_early_stopping(model: LateFusionNet, x_clin_tr: torch.Tensor, x_aec_tr: torch.Tensor,
                                y_tr: torch.Tensor) -> list[float]:
    # Training loss is the monitored signal (tried inner-split validation AUC
    # instead -- with only ~68 rows / ~14 events per inner split, epoch-count
    # decisions were too noisy across folds and made OOF AUC worse, not better;
    # reverted). Stops once loss stops improving by MIN_DELTA for PATIENCE epochs,
    # then restores the best-loss weights.
    criterion = _make_criterion(y_tr)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_PLATEAU_FACTOR, patience=LR_PLATEAU_PATIENCE
    )

    model.train()
    loss_history: list[float] = []
    best_loss = float("inf")
    best_state = None
    patience_ctr = 0
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_clin_tr, x_aec_tr), y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        loss_val = loss.item()
        loss_history.append(loss_val)
        scheduler.step(loss_val)

        if loss_val < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return loss_history


def train_fold(x_clin_tr: torch.Tensor, x_aec_tr: torch.Tensor, y_tr: torch.Tensor,
                x_clin_va: torch.Tensor, x_aec_va: torch.Tensor, seed: int) -> tuple[np.ndarray, list[float]]:
    # Trains N_ENSEMBLE_SEEDS independently-initialized models on the same
    # outer-train fold and averages their sigmoid outputs on the outer-va fold --
    # only the first seed's loss history is kept (for the convergence plot; the
    # other seeds converge similarly, see loss_curve.png).
    preds = []
    loss_history: list[float] = []
    for i in range(N_ENSEMBLE_SEEDS):
        seed_i = seed * 100 + i
        torch.manual_seed(seed_i)
        model = LateFusionNet(clin_dim=x_clin_tr.shape[1], n_slices=x_aec_tr.shape[1])
        lh = _train_with_early_stopping(model, x_clin_tr, x_aec_tr, y_tr)
        if i == 0:
            loss_history = lh

        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(x_clin_va, x_aec_va)).numpy())

    return np.mean(preds, axis=0), loss_history


def oof_scores(x_clin: torch.Tensor, x_aec: torch.Tensor, y: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    oof = np.zeros(len(y), dtype=float)
    fold_loss_histories: list[list[float]] = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x_clin.numpy(), y)):
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        oof[va_idx], loss_history = train_fold(
            x_clin[tr_idx], x_aec[tr_idx], y_tr,
            x_clin[va_idx], x_aec[va_idx],
            seed=SEED + fold_id,
        )
        fold_loss_histories.append(loss_history)
    return oof, fold_loss_histories


def fit_final_model(x_clin: torch.Tensor, x_aec: torch.Tensor, y: np.ndarray,
                     seed: int = SEED) -> tuple[LateFusionEnsemble, list[float]]:
    # Refit on the FULL internal Stage-2 cohort (no held-out fold) -- an
    # N_ENSEMBLE_SEEDS-model ensemble, mirroring train_fold, is the frozen
    # artifact applied to external (mirrors clinic-only_baseline.py's
    # fit_baseline_model, generalized from one model to an averaged ensemble).
    y_t = torch.tensor(y, dtype=torch.float32)
    models: list[LateFusionNet] = []
    loss_history: list[float] = []
    for i in range(N_ENSEMBLE_SEEDS):
        seed_i = seed * 100 + i
        torch.manual_seed(seed_i)
        model = LateFusionNet(clin_dim=x_clin.shape[1], n_slices=x_aec.shape[1])
        lh = _train_with_early_stopping(model, x_clin, x_aec, y_t)
        if i == 0:
            loss_history = lh
        model.eval()
        models.append(model)
    return LateFusionEnsemble(models), loss_history


def plot_loss_curves(fold_loss_histories: list[list[float]], final_loss_history: list[float], out_path: Path) -> None:
    # Confirms training convergence: BCE loss vs. epoch for each of the 5 OOF folds
    # plus the final refit on the full internal cohort. Early-stopped runs are
    # shorter than N_EPOCHS, so each curve is plotted over its own epoch count.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for fold_id, loss_history in enumerate(fold_loss_histories):
        ax.plot(np.arange(1, len(loss_history) + 1), loss_history, linewidth=1, alpha=0.6, label=f"fold {fold_id + 1}")
    ax.plot(np.arange(1, len(final_loss_history) + 1), final_loss_history, linewidth=2.5, color="black", label="final refit (full cohort)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Stage-2 Late Fusion: training loss vs. epoch", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved loss curve to {out_path}")


def plot_stage1_vs_full_pipeline_roc(rows: list[dict], out_path: Path) -> None:
    # Overlays Stage-1-only vs. full-pipeline ROC curves per cohort (on the whole
    # cohort, not just the screen-positive subgroup) so the AUC gain/loss from
    # adding Stage 2 is visible directly on the same axes. Each row: {"label", "y",
    # "stage1_score", "stage1_auc", "full_score", "full_auc"} (auc entries are
    # baseline.auc_significance_stats dicts).
    fig, axes = plt.subplots(1, len(rows), figsize=(6.5 * len(rows), 6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        fpr1, tpr1, _ = roc_curve(r["y"], r["stage1_score"])
        fpr2, tpr2, _ = roc_curve(r["y"], r["full_score"])
        ax.plot(fpr1, tpr1, color="#9a9a9a", linewidth=2,
                 label=f"Stage 1 only (AUC={r['stage1_auc']['auc']:.3f})")
        ax.plot(fpr2, tpr2, color="#2a78d6", linewidth=2.5,
                 label=f"Full pipeline (AUC={r['full_auc']['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("1 - Specificity (FPR)")
        ax.set_ylabel("Sensitivity (TPR)")
        delong_p = r.get("delong_p")
        title = r["label"]
        if delong_p is not None:
            p_str = "p<0.001" if delong_p < 0.001 else f"p={delong_p:.3f}"
            title = f"{r['label']}  (DeLong {p_str})"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Stage 1 (clinical-only) vs. Full Pipeline (Stage 1 + Stage 2): ROC comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Stage-1 vs full-pipeline ROC comparison to {out_path}")


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
    # acceptance test compares.
    y_all = stage1_rows_all["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask = stage1_rows_all["group"].isin(["TP", "FP"]).to_numpy()
    final_pred = combine_predictions(pos_mask, stage2_score, th_stage2)
    return y_all, pos_mask, final_pred


def combine_full_pipeline_score(stage1_score: np.ndarray, pos_mask: np.ndarray, stage2_score: np.ndarray,
                                 th_stage1: float) -> np.ndarray:
    # Whole-cohort continuous score comparable 1:1 with the Stage-1-only score
    # (so their AUCs can be plotted on the same axes): screen-negative rows
    # (never reach Stage 2) keep their Stage-1 score, which is always < th_stage1
    # by construction; screen-positive rows get th_stage1 + Stage-2's score,
    # guaranteeing every screen-positive row ranks above every screen-negative
    # row -- matching the pipeline's actual hard decision (screen-negative is
    # always predicted Negative) -- while each stage's own within-group ranking
    # is preserved. stage2_score must be row-aligned with pos_mask's True entries.
    full_score = stage1_score.copy()
    full_score[pos_mask] = th_stage1 + stage2_score
    return full_score


def _delong_midrank(x: np.ndarray) -> np.ndarray:
    # Midranks (ties get the average rank) used by the fast DeLong algorithm.
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
    # scores: (n_scores, n_samples) with columns ordered positives-then-negatives.
    # Returns (AUC per score row, covariance matrix of those AUCs) via the fast
    # DeLong et al. (1988) / Sun & Xu (2014) algorithm -- the standard way to test
    # whether two AUCs computed on the SAME (correlated) samples differ, as opposed
    # to each AUC's independent vs-0.5 significance test.
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
    # Two-sided DeLong test for whether AUC(score_a) differs from AUC(score_b) on
    # the SAME patients/outcome (e.g. Stage-1 score vs. full-pipeline score) --
    # a paired test, not two independent vs-0.5 tests, since the two scores share
    # every sample.
    order = np.argsort(-y)
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _delong_covariance(scores, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if not (var > 0):
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": float(z), "p_value": p}


def choose_stage2_threshold(y_all: np.ndarray, pos_mask: np.ndarray, stage2_score: np.ndarray,
                             sens_before: float, spec_before: float,
                             margin: float = SENS_LOSS_RATIO_MARGIN) -> float:
    # Selects th2 maximizing specificity among thresholds that pass the NI test vs.
    # stage-1-only (sens retains >=(1-margin) of sens_before AND spec does not get
    # worse) over the whole cohort -- ties broken by higher sensitivity. th=-inf
    # (no Stage-2 reclassification, i.e. stage-1-only itself) always trivially
    # satisfies the NI test, so a valid candidate always exists. Internal-only:
    # external is never touched by this sweep.
    sens_floor = sens_before * (1 - margin)
    candidates = np.concatenate([[-np.inf], np.unique(stage2_score)])
    best = None
    for th in candidates:
        pred = combine_predictions(pos_mask, stage2_score, th)
        tp, fp, fn, tn = baseline.confusion_counts(y_all, pred)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        if not (np.isfinite(sens) and np.isfinite(spec)):
            continue
        if sens < sens_floor or spec < spec_before:
            continue
        if best is None or spec > best[1] or (spec == best[1] and sens > best[2]):
            best = (float(th), spec, sens)
    assert best is not None
    return best[0]


def sens_noninferior(sens_before: float, sens_after: float, margin: float = SENS_LOSS_RATIO_MARGIN) -> bool:
    # Relative margin: sens_after must retain >= (1-margin) of sens_before (e.g.
    # 0.907 -> floor 0.907*0.95 = 0.862), not an absolute percentage-point drop.
    return sens_after >= sens_before * (1 - margin)


def ni_pass_fail(sens_before: float, sens_after: float, spec_before: float, spec_after: float,
                  margin: float = SENS_LOSS_RATIO_MARGIN) -> bool:
    # NI test vs. stage-1-only: sens retains >=95% of its stage-1-only value
    # (relative margin) AND spec does not get worse (spec_delta >= 0).
    return sens_noninferior(sens_before, sens_after, margin) and (spec_after - spec_before >= 0)


def exact_mcnemar_p(gain_n: int, loss_n: int) -> float:
    # Exact binomial test on the discordant pairs (gain vs. loss under H0: p=0.5) --
    # only screen-positive rows can be reclassified and only downward (positive ->
    # negative, see combine_predictions), so gain/loss counts fully summarize the
    # before/after transitions relevant to each metric.
    n = gain_n + loss_n
    if n == 0:
        return float("nan")
    return float(stats.binomtest(min(gain_n, loss_n), n, 0.5, alternative="two-sided").pvalue)


def accuracy(result: dict) -> float:
    tp, fn, fp, tn = result["matrix"][0, 0], result["matrix"][0, 1], result["matrix"][1, 0], result["matrix"][1, 1]
    return float((tp + tn) / (tp + fn + fp + tn))


TABLE_HEADER_BG = "#1c1c1c"
TABLE_HEADER_FG = "#ffffff"
TABLE_HEADER_SUB = "#b9b8b3"
TABLE_BAND_BG = "#f6f6f4"
TABLE_GRID = "#d9d8d3"
TABLE_DIVIDER = "#2a2a2a"
TABLE_GOOD = "#1a7a4c"
TABLE_BAD = "#c0392b"
TABLE_NRI_BG = "#d9e8fb"
TABLE_NRI_FG = "#1553b6"
TABLE_TEXT = "#161616"
TABLE_MUTED = "#4d4c48"
TABLE_SUBTEXT = "#6b6a66"


def build_clinical_vs_aec_row(cohort: str, y_all: np.ndarray, pos_mask: np.ndarray, pred_all: np.ndarray,
                               stage1_only: dict, result_final: dict, auc: float) -> dict:
    # tp_lost/fp_removed are the only two transition types possible: reclassification
    # only ever moves screen-positive rows from predicted-positive to
    # predicted-negative (see combine_predictions), never the other way.
    tp_lost = int(np.sum((y_all == 1) & pos_mask & ~pred_all))
    fp_removed = int(np.sum((y_all == 0) & pos_mask & ~pred_all))
    return {
        "cohort": cohort, "n": int(len(y_all)), "event": int(y_all.sum()), "auc": auc,
        "sens_clin": stage1_only["sens"], "sens_aec": result_final["sens"], "sens_p": exact_mcnemar_p(0, tp_lost),
        "spec_clin": stage1_only["spec"], "spec_aec": result_final["spec"], "spec_p": exact_mcnemar_p(fp_removed, 0),
        "acc_clin": accuracy(stage1_only), "acc_aec": accuracy(result_final), "acc_p": exact_mcnemar_p(fp_removed, tp_lost),
        "net_nri": fp_removed - tp_lost,
    }


def plot_clinical_vs_aec_table(rows: list[dict], out_path: Path, title: str) -> None:
    # Each row shows a cohort's stage-1-only ("Clinical only") vs full-pipeline
    # ("AEC-assisted") sens/spec/acc, McNemar p-values, and Net NRI.
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [("sens", "Sensitivity"), ("spec", "Specificity"), ("acc", "Accuracy")]
    row_h, header_h, footer_h = 1.0, 1.7, 0.55
    block_h = len(metrics) * row_h
    total_h = header_h + len(rows) * block_h + footer_h

    col = {"cohort": (0.00, 0.15), "n": (0.15, 0.205), "event": (0.205, 0.26),
           "metric": (0.26, 0.40), "clin": (0.40, 0.62), "aec": (0.62, 0.90), "nri": (0.90, 1.00)}
    cx = lambda key: (col[key][0] + col[key][1]) / 2

    fig, ax = plt.subplots(figsize=(13.5, total_h * 0.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    header_bottom = total_h - header_h
    ax.add_patch(Rectangle((0, header_bottom), 1, header_h, facecolor=TABLE_HEADER_BG, edgecolor="none", zorder=1))
    header_main_y = header_bottom + header_h * 0.68
    header_sub_y = header_bottom + header_h * 0.28
    for key, label in [("cohort", "코호트"), ("n", "N"), ("event", "Event"), ("metric", "지표")]:
        ax.text(cx(key), header_bottom + header_h / 2, label, ha="center", va="center",
                color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("clin"), header_main_y, "Clinical only", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("clin"), header_sub_y, "sens / spec / acc", ha="center", va="center",
            color=TABLE_HEADER_SUB, fontsize=9.5)
    ax.text(cx("aec"), header_main_y, "AEC-assisted", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")
    ax.text(cx("aec"), header_sub_y, "sens / spec / acc (p)", ha="center", va="center",
            color=TABLE_HEADER_SUB, fontsize=9.5)
    ax.text(cx("nri"), header_bottom + header_h / 2, "Net\nNRI", ha="center", va="center",
            color=TABLE_HEADER_FG, fontsize=13, fontweight="bold")

    def pfmt(p: float) -> str:
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    y_cursor = header_bottom
    for gi, r in enumerate(rows):
        block_top = y_cursor
        block_bottom = y_cursor - block_h
        if gi % 2 == 0:
            ax.add_patch(Rectangle((0, block_bottom), 1, block_h,
                                    facecolor=TABLE_BAND_BG, edgecolor="none", zorder=0))

        mid_y = (block_top + block_bottom) / 2
        ax.text(cx("cohort"), mid_y + 0.12, r["cohort"], ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=TABLE_TEXT)
        ax.text(cx("cohort"), mid_y - 0.22, f"AUC {r['auc']:.3f}", ha="center", va="center",
                fontsize=9.5, color=TABLE_GOOD)
        ax.text(cx("n"), mid_y, f"{r['n']}", ha="center", va="center", fontsize=12, color=TABLE_TEXT)
        ax.text(cx("event"), mid_y, f"{r['event']}", ha="center", va="center", fontsize=12, color=TABLE_TEXT)

        nri = r["net_nri"]
        box_w, box_h = 0.07, 0.9
        ax.add_patch(FancyBboxPatch((cx("nri") - box_w / 2, mid_y - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     linewidth=0, facecolor=TABLE_NRI_BG, zorder=2))
        ax.text(cx("nri"), mid_y, f"{nri:+d}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=TABLE_NRI_FG, zorder=3)

        for mi, (mkey, mlabel) in enumerate(metrics):
            row_top = block_top - mi * row_h
            row_bottom = row_top - row_h
            row_mid = (row_top + row_bottom) / 2

            ax.text(cx("metric"), row_mid, mlabel, ha="center", va="center", fontsize=11.5, color=TABLE_TEXT)

            clin_val, aec_val, p_val = r[f"{mkey}_clin"], r[f"{mkey}_aec"], r[f"{mkey}_p"]
            delta = aec_val - clin_val
            dcolor = TABLE_GOOD if delta >= 0 else TABLE_BAD

            ax.text(cx("clin"), row_mid, f"{clin_val:.3f}", ha="center", va="center",
                    fontsize=12, color=TABLE_MUTED)
            aec_x0, aec_x1 = col["aec"]
            ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.28, row_mid, f"{aec_val:.3f}",
                    ha="center", va="center", fontsize=12, color=TABLE_TEXT)
            ax.text(aec_x0 + (aec_x1 - aec_x0) * 0.72, row_mid, f"({delta:+.3f}) {pfmt(p_val)}",
                    ha="center", va="center", fontsize=9.5, color=dcolor)

            ax.plot([col["metric"][0], 1], [row_bottom, row_bottom], color=TABLE_GRID,
                    linewidth=0.8, zorder=1)

        y_cursor = block_bottom
        ax.plot([0, 1], [block_bottom, block_bottom], color=TABLE_DIVIDER, linewidth=1.4, zorder=2)

    footnote = "* p < 0.05 (유의)    n.s. p ≥ 0.05 (비유의)    Net NRI: AEC 추가 시 순 재분류 개선 환자 수"
    ax.text(0.0, footer_h * 0.4, footnote, ha="left", va="center", fontsize=9, color=TABLE_SUBTEXT)

    fig.suptitle(title, x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=TABLE_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved clinical-vs-AEC table to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- internal: 5-fold OOF for an unbiased internal estimate ---
    screen = stage2.fit_internal_screen()
    stage1_rows_all_int, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = stage2.build_stage2_inputs(screen)
    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    x_clin_int, x_aec_int = _to_tensors(stage2_input_clin_int, stage2_input_aec_int)

    oof, fold_loss_histories = oof_scores(x_clin_int, x_aec_int, y_int)

    # th2 is chosen on the FULL internal cohort (not just the screen-positive
    # subgroup): among thresholds passing the NI test vs. stage-1-only, pick the
    # one maximizing specificity -- so the threshold baked into training is the
    # one used for evaluation.
    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    th = choose_stage2_threshold(y_all_int, pos_mask_int, oof, stage1_only_int["sens"], stage1_only_int["spec"])

    # --- freeze: refit on the full internal Stage-2 cohort, transfer to external ---
    model, final_loss_history = fit_final_model(x_clin_int, x_aec_int, y_int)
    plot_loss_curves(fold_loss_histories, final_loss_history, OUTPUT_DIR / "loss_curve.png")

    stage1_rows_all_ext, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = stage2.build_stage2_inputs_external(screen)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)
    x_clin_ext, x_aec_ext = _to_tensors(stage2_input_clin_ext, stage2_input_aec_ext)
    score_ext = model.predict_proba(x_clin_ext, x_aec_ext)

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

    # --- Stage-1 vs. full-pipeline AUC: a whole-cohort continuous score for the
    # full pipeline, built to be directly comparable to Stage-1's own AUC (see
    # combine_full_pipeline_score). stage1_rows_all_*["score"] is each cohort's
    # Stage-1 score for every patient (build_group_rows keeps it, row-aligned with
    # y_all_*/pos_mask_*). ---
    stage1_score_int = stage1_rows_all_int["score"].to_numpy()
    stage1_score_ext = stage1_rows_all_ext["score"].to_numpy()
    full_score_int = combine_full_pipeline_score(stage1_score_int, pos_mask_int, oof, screen["th"])
    full_score_ext = combine_full_pipeline_score(stage1_score_ext, pos_mask_ext, score_ext, screen["th"])

    auc_stage1_int = baseline.auc_significance_stats(y_all_int, stage1_score_int)
    auc_full_int = baseline.auc_significance_stats(y_all_int, full_score_int)
    auc_stage1_ext = baseline.auc_significance_stats(y_all_ext, stage1_score_ext)
    auc_full_ext = baseline.auc_significance_stats(y_all_ext, full_score_ext)

    # DeLong test: Stage-1 and full-pipeline scores are evaluated on the SAME
    # patients, so their AUCs are correlated -- an independent vs-0.5 comparison
    # (auc_significance_stats) can't tell whether the two differ from each other.
    delong_int = delong_paired_auc_test(y_all_int.astype(float), stage1_score_int, full_score_int)
    delong_ext = delong_paired_auc_test(y_all_ext.astype(float), stage1_score_ext, full_score_ext)
    print(f"[internal] Stage-1 AUC={auc_stage1_int['auc']:.3f} [{auc_stage1_int['ci_lower']:.3f}, {auc_stage1_int['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full_int['auc']:.3f} [{auc_full_int['ci_lower']:.3f}, {auc_full_int['ci_upper']:.3f}]  "
          f"DeLong diff={delong_int['diff']:+.4f} p={delong_int['p_value']:.4f}")
    print(f"[external] Stage-1 AUC={auc_stage1_ext['auc']:.3f} [{auc_stage1_ext['ci_lower']:.3f}, {auc_stage1_ext['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full_ext['auc']:.3f} [{auc_full_ext['ci_lower']:.3f}, {auc_full_ext['ci_upper']:.3f}]  "
          f"DeLong diff={delong_ext['diff']:+.4f} p={delong_ext['p_value']:.4f}")

    plot_stage1_vs_full_pipeline_roc([
        {"label": "internal", "y": y_all_int, "stage1_score": stage1_score_int, "stage1_auc": auc_stage1_int,
         "full_score": full_score_int, "full_auc": auc_full_int, "delong_p": delong_int["p_value"]},
        {"label": "external", "y": y_all_ext, "stage1_score": stage1_score_ext, "stage1_auc": auc_stage1_ext,
         "full_score": full_score_ext, "full_auc": auc_full_ext, "delong_p": delong_ext["p_value"]},
    ], OUTPUT_DIR / "roc_comparison_stage1_vs_full_pipeline.png")

    pd.DataFrame([
        {"cohort": "internal", "stage1_auc": auc_stage1_int["auc"], "stage1_ci_lower": auc_stage1_int["ci_lower"],
         "stage1_ci_upper": auc_stage1_int["ci_upper"], "full_pipeline_auc": auc_full_int["auc"],
         "full_pipeline_ci_lower": auc_full_int["ci_lower"], "full_pipeline_ci_upper": auc_full_int["ci_upper"],
         "auc_diff": delong_int["diff"], "delong_z": delong_int["z"], "delong_p_value": delong_int["p_value"],
         "significant_p05": bool(np.isfinite(delong_int["p_value"]) and delong_int["p_value"] < 0.05)},
        {"cohort": "external", "stage1_auc": auc_stage1_ext["auc"], "stage1_ci_lower": auc_stage1_ext["ci_lower"],
         "stage1_ci_upper": auc_stage1_ext["ci_upper"], "full_pipeline_auc": auc_full_ext["auc"],
         "full_pipeline_ci_lower": auc_full_ext["ci_lower"], "full_pipeline_ci_upper": auc_full_ext["ci_upper"],
         "auc_diff": delong_ext["diff"], "delong_z": delong_ext["z"], "delong_p_value": delong_ext["p_value"],
         "significant_p05": bool(np.isfinite(delong_ext["p_value"]) and delong_ext["p_value"] < 0.05)},
    ]).to_csv(OUTPUT_DIR / "stage1_vs_full_pipeline_auc.csv", index=False)

    stage1_only_ext = baseline.evaluate("external / stage-1 only", y_all_ext, pos_mask_ext, screen["th"])
    result_final_int = baseline.evaluate("internal", y_all_int, pred_all_int, th)
    result_final_ext = baseline.evaluate("external", y_all_ext, pred_all_ext, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Stage 1 only (Clinical-only S90 screen)", fontsize=13, fontweight="bold")
    for ax, result, cohort_label in [(axes[0], stage1_only_int, "internal"), (axes[1], stage1_only_ext, "external")]:
        matrix = result["matrix"]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            label = [["TP", "FN"], ["FP", "TN"]][i][j]
            ax.text(j, i, f"{label}\n{matrix[i, j]}", ha="center", va="center", fontsize=13,
                    color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Positive", "Actual Negative"])
        ax.set_title(f"{cohort_label}\n(threshold={result['th']:.3f})", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage1_only.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Full Pipeline (Stage 1 screen + Stage 2 late fusion)", fontsize=13, fontweight="bold")
    baseline.plot_confusion_matrix(axes[0], result_final_int)
    baseline.plot_confusion_matrix(axes[1], result_final_ext)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    # --- full-pipeline sens/spec: stage-1-only (before) vs. stage-1+stage-2 (after) ---
    sens_delta_int = result_final_int["sens"] - stage1_only_int["sens"]
    spec_delta_int = result_final_int["spec"] - stage1_only_int["spec"]
    sens_delta_ext = result_final_ext["sens"] - stage1_only_ext["sens"]
    spec_delta_ext = result_final_ext["spec"] - stage1_only_ext["spec"]

    # --- NI test vs. stage-1-only: sens retains >=95% of its stage-1-only value
    # (relative margin) AND spec_delta >= 0 ---
    ok_int = ni_pass_fail(stage1_only_int["sens"], result_final_int["sens"], stage1_only_int["spec"], result_final_int["spec"])
    ok_ext = ni_pass_fail(stage1_only_ext["sens"], result_final_ext["sens"], stage1_only_ext["spec"], result_final_ext["spec"])

    for cohort, sens_delta, spec_delta, stage1_res, ok in [
        ("internal", sens_delta_int, spec_delta_int, stage1_only_int, ok_int),
        ("external", sens_delta_ext, spec_delta_ext, stage1_only_ext, ok_ext),
    ]:
        sens_floor = stage1_res["sens"] * (1 - SENS_LOSS_RATIO_MARGIN)
        print(f"[{cohort}] sens_delta={sens_delta:+.3f} spec_delta={spec_delta:+.3f} "
              f"(sens floor={sens_floor:.3f}, margin={SENS_LOSS_RATIO_MARGIN:.0%} relative) -> {'PASS' if ok else 'FAIL'}")

    pipeline_summary = pd.DataFrame([
        {"cohort": "internal", "sens_before": stage1_only_int["sens"], "sens_after": result_final_int["sens"],
         "sens_delta": sens_delta_int, "sens_floor": stage1_only_int["sens"] * (1 - SENS_LOSS_RATIO_MARGIN),
         "spec_before": stage1_only_int["spec"], "spec_after": result_final_int["spec"],
         "spec_delta": spec_delta_int, "pass": ok_int},
        {"cohort": "external", "sens_before": stage1_only_ext["sens"], "sens_after": result_final_ext["sens"],
         "sens_delta": sens_delta_ext, "sens_floor": stage1_only_ext["sens"] * (1 - SENS_LOSS_RATIO_MARGIN),
         "spec_before": stage1_only_ext["spec"], "spec_after": result_final_ext["spec"],
         "spec_delta": spec_delta_ext, "pass": ok_ext},
    ])
    pipeline_summary_path = OUTPUT_DIR / "final_pipeline_summary.csv"
    pipeline_summary.to_csv(pipeline_summary_path, index=False)
    print(f"Saved final pipeline summary to {pipeline_summary_path}")

    # --- clinical-vs-AEC-assisted summary table image ---
    table_rows = [
        build_clinical_vs_aec_row("internal", y_all_int, pos_mask_int, pred_all_int, stage1_only_int, result_final_int, auc_int["auc"]),
        build_clinical_vs_aec_row("external", y_all_ext, pos_mask_ext, pred_all_ext, stage1_only_ext, result_final_ext, auc_ext["auc"]),
    ]
    plot_clinical_vs_aec_table(
        table_rows, OUTPUT_DIR / "clinical_vs_aec_assisted_table.png",
        "clinical-only vs. AEC-assisted(Late Fusion) 성능 비교 (Stage-1 vs Stage-1+Stage-2)",
    )

    # --- NRI / McNemar as the PRIMARY significance test for Stage 2's effect.
    # Whole-curve AUC/DeLong is known to be underpowered for detecting a marker's
    # incremental value (Pepe et al. 2004; Pencina et al. 2008) -- Stage 2 only
    # moves specific screen-positive patients across the decision boundary, which
    # a reclassification test (McNemar on the discordant pairs, summarized as Net
    # NRI) is built to detect directly, whereas it gets diluted across the whole
    # ROC curve in an AUC comparison. AUC/DeLong (stage1_vs_full_pipeline_auc.csv)
    # is kept as a secondary, whole-curve discrimination check. ---
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "clinical_vs_aec_assisted_summary.csv", index=False)
    print("\n=== PRIMARY significance test: NRI / McNemar (reclassification) ===")
    for r in table_rows:
        print(f"[{r['cohort']}] Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
              f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
              f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})  "
              f"acc: {r['acc_clin']:.3f}->{r['acc_aec']:.3f} (p={r['acc_p']:.4f})")
    print("=== Secondary: whole-curve AUC / DeLong ===")
    print(f"[internal] DeLong p={delong_int['p_value']:.4f}   [external] DeLong p={delong_ext['p_value']:.4f}")


if __name__ == "__main__":
    main()
