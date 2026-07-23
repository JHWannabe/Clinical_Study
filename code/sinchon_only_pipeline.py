from __future__ import annotations

# Single-cohort (Sinchon-only) Stage-1 -> Stage-2 pipeline, evaluated entirely via
# 5-fold out-of-fold (OOF) predictions -- unlike the gangnam(internal)/sinchon(external)
# cross-cohort transfer used elsewhere, this script only touches sinchon.xlsx and never
# holds out a separate external cohort. Stage 1 (clinical-only LR) is scored OOF over
# the whole Sinchon cohort (5-fold CV); only the patients its OOF score calls Positive
# feed Stage 2, which -- unlike stage2_model.py's joint end-to-end late fusion -- is a
# two-step stack: (1) a standalone 1D-CNN (AecCNN) takes only the AEC-128 curve and
# outputs a low-SMI logit, scored OOF (5-fold CV) over the screen-positive subset;
# (2) that AEC-CNN OOF score is added as a 5th feature alongside the 4 standardized
# clinical variables (sex_m, age_std, height_std, weight_std) into a final classifier,
# itself scored OOF (another 5-fold CV) for the final Stage-2 prediction. grid_search_stage2
# tunes the AEC-CNN's hyperparameters and picks this final classifier's model type
# (logistic regression / random forest / gradient boosting / SVM-RBF) + hyperparameters
# by OOF AUC, and main() reruns the winning configuration for the reported result.
#
# Reuses baseline (clinic-only_baseline.py) for Stage-1 LR + eval/plot utilities,
# stage2_dataset._stage1_positive_rows for building the screen-positive clinic+AEC rows
# (works for any meta/y/score/th over sinchon.xlsx, not just the internal cohort it was
# originally written for), and stage2_model for threshold selection / NRI-McNemar
# summary utilities (these are generic over any Stage-2 score, not specific to its
# late-fusion model).
#
# Run: python code/sinchon_only_pipeline.py

import copy
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2_dataset = import_module("stage2_dataset")
stage2_model = import_module("stage2_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sinchon_only_pipeline"

SINCHON_XLSX = DATA_DIR / "sinchon.xlsx"
CLIN_COLS = stage2_dataset.CLIN_COLS
AEC_COLS = stage2_dataset.AEC_COLS


def plot_confusion_matrix_custom(ax: Axes, result: dict, label: str) -> None:
    # Same drawing as baseline.plot_confusion_matrix, but with an explicit label
    # instead of that function's hardcoded internal/external cohort-name check --
    # this script only ever has one cohort (Sinchon, 5-fold OOF).
    matrix = result["matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center", fontsize=13,
                    color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Positive", "Actual Negative"])
    ax.set_title(f"{label}\n(threshold={result['th']:.3f})", fontsize=11, fontweight="bold")


def plot_fold_loss_curves(fold_loss_histories: list[list[float]], out_path: Path, title: str) -> None:
    # Convergence check across OOF folds -- no final full-cohort refit here (unlike
    # stage2_model.plot_loss_curves), since this script has no external cohort to
    # apply one to.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for fold_id, loss_history in enumerate(fold_loss_histories):
        ax.plot(np.arange(1, len(loss_history) + 1), loss_history, linewidth=1.2, alpha=0.8, label=f"fold {fold_id + 1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved loss curve to {out_path}")


class AecCNN(nn.Module):
    # Standalone 1D-CNN over the 128-slice AEC curve: same conv trunk as
    # stage2_model.AecBranch(variant="convpool") (global-avg-pooled so the curve is
    # read holistically, not point-wise), but ending in a classification head
    # (Linear(embed_dim, 1)) instead of stopping at an embedding for fusion -- this
    # model is trained standalone against the low-SMI label, and only its scalar
    # output feeds the final Stage-2 classifier below (a stacked model, not
    # stage2_model.py's joint end-to-end late fusion).
    #
    # Two additions vs. AecBranch, both needed because this net has to carry the
    # whole classification signal alone (in late fusion the frozen Stage-1 score
    # does much of the work): (1) BatchNorm1d after each conv -- full-batch Adam on
    # the raw curve (patient-normalized to mean~1, see load_aec_for_patients) was
    # oscillating around the constant-prediction loss for the whole training run
    # (loss_curve_aec_cnn.png never trended down) instead of converging; (2) the
    # output layer's bias is initialized to the training set's logit(prior) instead
    # of 0, so the model starts near the base rate rather than a maximally-uncertain
    # 0-logit and only needs to learn a *deviation* from it.
    def __init__(self, n_slices: int = 128, embed_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def init_output_bias(self, prior: float) -> None:
        logit = float(np.log(prior / (1 - prior)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, logit)

    def forward(self, x_aec: torch.Tensor) -> torch.Tensor:
        feat = self.conv((x_aec - 1.0).unsqueeze(1))  # center: curves are patient-normalized to mean~1
        pooled = self.avg_pool(feat).squeeze(-1)
        return self.head(pooled).squeeze(-1)  # logit


def _train_aec_cnn(model: AecCNN, x_aec_tr: torch.Tensor, y_tr: torch.Tensor,
                    lr: float = stage2_model.LR, weight_decay: float = stage2_model.WEIGHT_DECAY) -> list[float]:
    # Same training recipe (optimizer, LR-plateau schedule, grad clipping, early
    # stopping) as stage2_model._train_with_early_stopping, reused via its shared
    # hyperparameters and pos_weight criterion -- just called with a single (AEC-only)
    # input instead of (x_clin, x_aec). lr/weight_decay are exposed (rather than always
    # reading the stage2_model constants) so grid_search_stage2 can sweep them.
    criterion = stage2_model._make_criterion(y_tr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=stage2_model.LR_PLATEAU_FACTOR, patience=stage2_model.LR_PLATEAU_PATIENCE
    )

    model.train()
    loss_history: list[float] = []
    best_loss = float("inf")
    best_state = None
    patience_ctr = 0
    for _ in range(stage2_model.N_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_aec_tr), y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), stage2_model.GRAD_CLIP_NORM)
        optimizer.step()
        loss_val = loss.item()
        loss_history.append(loss_val)
        scheduler.step(loss_val)

        if loss_val < best_loss - stage2_model.EARLY_STOP_MIN_DELTA:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= stage2_model.EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return loss_history


def _train_aec_cnn_fold(x_aec_tr: torch.Tensor, y_tr: torch.Tensor, x_aec_va: torch.Tensor, seed: int,
                         embed_dim: int = 16, dropout: float = 0.2, lr: float = stage2_model.LR,
                         weight_decay: float = stage2_model.WEIGHT_DECAY,
                         n_ensemble_seeds: int = stage2_model.N_ENSEMBLE_SEEDS) -> tuple[np.ndarray, list[float]]:
    # n_ensemble_seeds independently-initialized AecCNNs on the same fold, sigmoid
    # outputs averaged -- mirrors stage2_model.train_fold's seed-ensemble. Defaults to
    # the full N_ENSEMBLE_SEEDS; grid_search_stage2 passes a smaller value to keep the
    # search itself cheap.
    preds = []
    loss_history: list[float] = []
    for i in range(n_ensemble_seeds):
        seed_i = seed * 100 + i
        torch.manual_seed(seed_i)
        model = AecCNN(n_slices=x_aec_tr.shape[1], embed_dim=embed_dim, dropout=dropout)
        model.init_output_bias(float(y_tr.mean().item()))
        lh = _train_aec_cnn(model, x_aec_tr, y_tr, lr=lr, weight_decay=weight_decay)
        if i == 0:
            loss_history = lh
        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(x_aec_va)).numpy())
    return np.mean(preds, axis=0), loss_history


def aec_cnn_oof_scores(x_aec: torch.Tensor, y: np.ndarray, embed_dim: int = 16, dropout: float = 0.2,
                        lr: float = stage2_model.LR, weight_decay: float = stage2_model.WEIGHT_DECAY,
                        n_ensemble_seeds: int = stage2_model.N_ENSEMBLE_SEEDS) -> tuple[np.ndarray, list[list[float]]]:
    # 5-fold OOF probability from AEC alone -- this is the feature the final Stage-2
    # LR consumes, not a prediction reported on its own.
    oof = np.zeros(len(y), dtype=float)
    fold_loss_histories: list[list[float]] = []
    skf = StratifiedKFold(n_splits=stage2_model.N_FOLDS, shuffle=True, random_state=stage2_model.SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x_aec.numpy(), y)):
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        oof[va_idx], loss_history = _train_aec_cnn_fold(
            x_aec[tr_idx], y_tr, x_aec[va_idx], seed=stage2_model.SEED + fold_id,
            embed_dim=embed_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, n_ensemble_seeds=n_ensemble_seeds)
        fold_loss_histories.append(loss_history)
    return oof, fold_loss_histories


def _make_final_model(name: str, params: dict, seed: int):
    # Candidate final-stage classifiers for the 5-feature (4 clinical + AEC-CNN score)
    # input -- logreg is stage2_model.py-style linear, the rest are added so the grid
    # search can check whether a nonlinear/tree-based model reads this small (5-dim)
    # feature set better than a linear one.
    if name == "logreg":
        return LogisticRegression(C=params["C"], solver="lbfgs", max_iter=5000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                       random_state=seed, n_jobs=-1)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                           learning_rate=params["learning_rate"], random_state=seed)
    if name == "svm_rbf":
        return SVC(C=params["C"], gamma=params["gamma"], kernel="rbf", probability=True, random_state=seed)
    raise ValueError(f"unknown final model: {name}")


def final_model_oof_scores(name: str, params: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 5-fold OOF probability for any FINAL_MODEL_GRID candidate. Always predict_proba
    # (never decision_function) so every candidate's OOF score sits on the same [0,1]
    # scale regardless of model type -- random_forest/gradient_boosting don't implement
    # decision_function, and combine_full_pipeline_score downstream assumes a
    # non-negative score (see its call site in main()), which a raw logit could violate.
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=baseline.N_FOLDS, shuffle=True, random_state=baseline.SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x, y)):
        model = _make_final_model(name, params, seed=baseline.SEED + fold_id)
        model.fit(x[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(x[va_idx])[:, 1]
    return oof


AEC_CNN_GRID = {"lr": [1e-3, 5e-4], "dropout": [0.2, 0.4], "embed_dim": [16, 32]}
FINAL_MODEL_GRID: dict[str, list[dict]] = {
    "logreg": [{"C": c} for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]],
    "random_forest": [
        {"n_estimators": 200, "max_depth": 3},
        {"n_estimators": 200, "max_depth": 5},
        {"n_estimators": 500, "max_depth": None},
    ],
    "gradient_boosting": [
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    ],
    "svm_rbf": [
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
    ],
}
GRID_SEARCH_ENSEMBLE_SEEDS = 1  # cheaper than the reported run's N_ENSEMBLE_SEEDS=5 -- only for ranking configs


def grid_search_stage2(x_aec_t: torch.Tensor, y2: np.ndarray, x_clin_4: np.ndarray, out_dir: Path) -> dict:
    # Two-phase (coordinate) search, not one joint grid over every knob -- that keeps
    # the expensive part (CNN training) to len(AEC_CNN_GRID) combos instead of
    # combos*len(all final-model hyperparameter combos). Phase 1 ranks AEC-CNN configs
    # by the CNN's OWN OOF AUC (decoupled from the final classifier); phase 2 fixes the
    # winning CNN config and compares final-classifier model TYPES (logreg / random
    # forest / gradient boosting / SVM-RBF), not just logreg's C, over the resulting
    # 5-feature matrix (cheap regardless of model type -- sklearn fits on n~490, 5
    # features). Scored with plain roc_auc_score (no bootstrap CI) since this only
    # needs to RANK configs; main() reruns the winner at full N_ENSEMBLE_SEEDS with the
    # full auc_significance_stats for the reported result.
    print("\n=== Grid search (phase 1): AEC-CNN hyperparameters (own OOF AUC, "
          f"{GRID_SEARCH_ENSEMBLE_SEEDS}-seed search ensemble) ===")
    trials = []
    best_cnn_trial = None
    for lr in AEC_CNN_GRID["lr"]:
        for dropout in AEC_CNN_GRID["dropout"]:
            for embed_dim in AEC_CNN_GRID["embed_dim"]:
                oof, _ = aec_cnn_oof_scores(x_aec_t, y2, embed_dim=embed_dim, dropout=dropout, lr=lr,
                                             n_ensemble_seeds=GRID_SEARCH_ENSEMBLE_SEEDS)
                auc = roc_auc_score(y2, oof)
                trial = {"phase": 1, "lr": lr, "dropout": dropout, "embed_dim": embed_dim,
                          "final_model": None, "final_params": None, "auc": auc}
                trials.append(trial)
                print(f"  lr={lr:.0e} dropout={dropout} embed_dim={embed_dim} -> AEC-CNN OOF AUC={auc:.4f}")
                if best_cnn_trial is None or auc > best_cnn_trial["auc"]:
                    best_cnn_trial = trial
    best_cnn = {"lr": best_cnn_trial["lr"], "dropout": best_cnn_trial["dropout"], "embed_dim": best_cnn_trial["embed_dim"]}
    print(f"Best AEC-CNN config: {best_cnn} (OOF AUC={best_cnn_trial['auc']:.4f})")

    print("\n=== Grid search (phase 2): final classifier model + hyperparameters (fixed AEC-CNN config) ===")
    oof_cnn, _ = aec_cnn_oof_scores(x_aec_t, y2, n_ensemble_seeds=GRID_SEARCH_ENSEMBLE_SEEDS, **best_cnn)
    aec_mean, aec_std = stage2_model.fit_score_standardizer(oof_cnn)
    x_final = np.column_stack([x_clin_4, (oof_cnn - aec_mean) / aec_std])
    best_final_trial = None
    for name, param_grid in FINAL_MODEL_GRID.items():
        for params in param_grid:
            oof_final = final_model_oof_scores(name, params, x_final, y2)
            auc = roc_auc_score(y2, oof_final)
            trial = {"phase": 2, **best_cnn, "final_model": name, "final_params": params, "auc": auc}
            trials.append(trial)
            print(f"  {name} {params} -> Stage-2 OOF AUC={auc:.4f}")
            if best_final_trial is None or auc > best_final_trial["auc"]:
                best_final_trial = trial
    print(f"Best final model: {best_final_trial['final_model']} {best_final_trial['final_params']} "
          f"(OOF AUC={best_final_trial['auc']:.4f})")

    pd.DataFrame(trials).to_csv(out_dir / "grid_search_stage2.csv", index=False)
    return {**best_cnn, "final_model": best_final_trial["final_model"], "final_params": best_final_trial["final_params"]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: clinical-only LR, 5-fold OOF over the whole Sinchon cohort ---
    meta, y = baseline.load_cohort(SINCHON_XLSX)
    print(f"Sinchon cohort: n={len(y)} (event={int(y.sum())})")

    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)
    oof1 = baseline.oof_scores(x, y)
    th = baseline.threshold_for_sensitivity(y, oof1, baseline.TARGET_SENSITIVITY)
    print(f"[Sinchon, S{int(baseline.TARGET_SENSITIVITY * 100)}, 5-fold OOF] threshold={th:.4f}")

    stage1_only = baseline.evaluate("sinchon (5-fold OOF)", y, oof1 >= th, th)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    plot_confusion_matrix_custom(ax, stage1_only, "Stage 1 only (Sinchon, 5-fold OOF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage1_only.png", dpi=220)
    plt.close(fig)

    auc1 = baseline.auc_significance_stats(y, oof1)
    baseline.plot_roc_curve(y, oof1, auc1, OUTPUT_DIR / "roc_curve_stage1.png",
                             title="Stage 1 (Clinical-only LR): ROC (Sinchon, 5-fold OOF)")

    # --- Stage-2 inputs: screen-positive (TP/FP) rows only, clinic + AEC-128, from
    # Stage-1's OOF score. _stage1_positive_rows only needs an xlsx path (for AEC lookup)
    # plus meta/y/score/th/x_std, so it works here exactly as it does for
    # stage2_dataset's internal cohort. ---
    stage1_rows_all, stage1_rows_pos, stage2_clin, stage2_aec = \
        stage2_dataset._stage1_positive_rows(SINCHON_XLSX, meta, y, oof1, th, x)

    y2 = (stage1_rows_pos["group"] == "TP").to_numpy().astype(int)
    x_aec_t = torch.tensor(stage2_aec[AEC_COLS].to_numpy(dtype=np.float32))
    x_clin_4 = stage2_clin[CLIN_COLS].to_numpy(dtype=np.float64)

    # --- Grid search over the AEC-CNN's hyperparameters and the final classifier's
    # model type + hyperparameters (logreg/random_forest/gradient_boosting/svm_rbf),
    # ranked by OOF AUC (see grid_search_stage2's docstring for the two-phase design).
    best_hp = grid_search_stage2(x_aec_t, y2, x_clin_4, OUTPUT_DIR)
    print(f"\n=== Using grid-search-selected hyperparameters for the reported Stage-2 run: {best_hp} ===")

    # --- Stage 2, step 1: standalone AEC-only 1D-CNN (grid-search-selected hyperparameters,
    # full N_ENSEMBLE_SEEDS), scored 5-fold OOF over the screen-positive subset -- its OOF
    # probability is used only as a feature below, not reported as a Stage-2 result on its own. ---
    aec_cnn_oof, fold_loss_histories = aec_cnn_oof_scores(
        x_aec_t, y2, embed_dim=best_hp["embed_dim"], dropout=best_hp["dropout"], lr=best_hp["lr"])
    plot_fold_loss_curves(fold_loss_histories, OUTPUT_DIR / "loss_curve_aec_cnn.png",
                           title="AEC-CNN (Stage-2 feature extractor): training loss vs. epoch (Sinchon, 5-fold OOF)")

    # --- Stage 2, step 2: final classifier (grid-search-selected model type +
    # hyperparameters) on 4 standardized clinical features + the AEC-CNN's OOF score
    # (5 features total), itself scored 5-fold OOF (final_model_oof_scores -- always a
    # [0,1] probability regardless of model type, see that function's docstring). th2
    # is chosen (NI test vs. stage-1-only) on this OOF score, same selection rule as
    # stage2_model.py. The AEC-CNN's raw sigmoid output is standardized first -- same
    # reasoning as stage2_model.fit_score_standardizer for the frozen Stage-1 LR score:
    # on its own scale it sits off the clinical features' mean-0/std-1 scale, which
    # would leave tree-based/linear models alike (over)weighting one feature group. ---
    aec_mean, aec_std = stage2_model.fit_score_standardizer(aec_cnn_oof)
    aec_cnn_oof_std = (aec_cnn_oof - aec_mean) / aec_std
    x_final = np.column_stack([x_clin_4, aec_cnn_oof_std])
    oof2 = final_model_oof_scores(best_hp["final_model"], best_hp["final_params"], x_final, y2)

    y_all = stage1_rows_all["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask = stage1_rows_all["group"].isin(["TP", "FP"]).to_numpy()
    th2 = stage2_model.choose_stage2_threshold(y_all, pos_mask, oof2, stage1_only["sens"], stage1_only["spec"])

    result2 = baseline.evaluate("sinchon (5-fold OOF)", y2, oof2 >= th2, th2)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    plot_confusion_matrix_custom(ax, result2, "Stage 2 only (Sinchon, 5-fold OOF)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_stage2_only.png", dpi=220)
    plt.close(fig)

    auc2 = baseline.auc_significance_stats(y2, oof2)
    baseline.plot_roc_curve(y2, oof2, auc2, OUTPUT_DIR / "roc_curve_stage2.png",
                             title=f"Stage 2 (AEC-CNN feature + {best_hp['final_model']}): ROC (Sinchon, 5-fold OOF)")

    # --- Final pipeline (whole cohort, OOF): Stage-1 FN/TN (screen-negative, untouched)
    # + Stage-2's OOF reclassification of Stage-1 TP/FP (screen-positive). y_all/pos_mask
    # are the same masks final_pipeline_labels would derive from stage1_rows_all. ---
    pred_all = stage2_model.combine_predictions(pos_mask, oof2, th2)
    result_final = baseline.evaluate("sinchon (5-fold OOF)", y_all, pred_all, th)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Sinchon (5-fold OOF): Stage 1 only vs. Full Pipeline (Stage 1 + Stage 2)", fontsize=13, fontweight="bold")
    plot_confusion_matrix_custom(axes[0], stage1_only, "Stage 1 only")
    plot_confusion_matrix_custom(axes[1], result_final, "Full pipeline")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_pipeline.png", dpi=220)
    plt.close(fig)

    # --- Stage-1-only vs. full-pipeline AUC, whole cohort, on a directly comparable
    # continuous score (see stage2_model.combine_full_pipeline_score). That function
    # assumes stage2_score >= 0 so every screen-positive row's th_stage1+stage2_score
    # still ranks above every screen-negative row's raw (< th_stage1) Stage-1 score --
    # oof2 is always a [0,1] predict_proba output (final_model_oof_scores), which
    # satisfies that directly, unlike a raw (possibly negative) decision_function logit. ---
    stage1_score_all = stage1_rows_all["score"].to_numpy()
    full_score = stage2_model.combine_full_pipeline_score(stage1_score_all, pos_mask, oof2, th)
    auc_stage1_all = baseline.auc_significance_stats(y_all, stage1_score_all)
    auc_full = baseline.auc_significance_stats(y_all, full_score)
    delong = stage2_model.delong_paired_auc_test(y_all.astype(float), stage1_score_all, full_score)
    print(f"[sinchon] Stage-1 AUC={auc_stage1_all['auc']:.3f} "
          f"[{auc_stage1_all['ci_lower']:.3f}, {auc_stage1_all['ci_upper']:.3f}]  "
          f"Full-pipeline AUC={auc_full['auc']:.3f} [{auc_full['ci_lower']:.3f}, {auc_full['ci_upper']:.3f}]  "
          f"DeLong diff={delong['diff']:+.4f} p={delong['p_value']:.4f}")

    stage2_model.plot_stage1_vs_full_pipeline_roc([
        {"label": "sinchon (5-fold OOF)", "y": y_all, "stage1_score": stage1_score_all, "stage1_auc": auc_stage1_all,
         "full_score": full_score, "full_auc": auc_full, "delong_p": delong["p_value"]},
    ], OUTPUT_DIR / "roc_comparison_stage1_vs_full_pipeline.png")

    # --- NI test vs. stage-1-only (sens retains >=95% of its stage-1-only value AND
    # spec doesn't get worse), and the NRI/McNemar summary table (PRIMARY significance
    # test for Stage 2's effect -- see stage2_model.py's docstring for the rationale) ---
    ok = stage2_model.ni_pass_fail(stage1_only["sens"], result_final["sens"], stage1_only["spec"], result_final["spec"])
    sens_floor = stage1_only["sens"] * (1 - stage2_model.SENS_LOSS_RATIO_MARGIN)
    print(f"[sinchon] sens: {stage1_only['sens']:.3f}->{result_final['sens']:.3f}  "
          f"spec: {stage1_only['spec']:.3f}->{result_final['spec']:.3f}  "
          f"(sens floor={sens_floor:.3f}, margin={stage2_model.SENS_LOSS_RATIO_MARGIN:.0%} relative) -> "
          f"{'PASS' if ok else 'FAIL'}")

    pd.DataFrame([{
        "cohort": "sinchon (5-fold OOF)", "sens_before": stage1_only["sens"], "sens_after": result_final["sens"],
        "sens_floor": sens_floor, "spec_before": stage1_only["spec"], "spec_after": result_final["spec"],
        "pass": ok,
    }]).to_csv(OUTPUT_DIR / "final_pipeline_summary.csv", index=False)

    table_row = stage2_model.build_clinical_vs_aec_row(
        "sinchon (5-fold OOF)", y_all, pos_mask, pred_all, stage1_only, result_final, auc2["auc"])
    stage2_model.plot_clinical_vs_aec_table(
        [table_row], OUTPUT_DIR / "clinical_vs_aec_assisted_table.png",
        f"clinical-only vs. AEC-assisted(AEC-CNN + {best_hp['final_model']}) 성능 비교 (Sinchon, 5-fold OOF)")
    pd.DataFrame([table_row]).to_csv(OUTPUT_DIR / "clinical_vs_aec_assisted_summary.csv", index=False)

    print("\n=== PRIMARY significance test: NRI / McNemar (reclassification), Sinchon 5-fold OOF ===")
    r = table_row
    print(f"Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
          f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
          f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})  "
          f"acc: {r['acc_clin']:.3f}->{r['acc_aec']:.3f} (p={r['acc_p']:.4f})")
    print("=== Secondary: whole-curve AUC / DeLong ===")
    print(f"[sinchon] DeLong p={delong['p_value']:.4f}")


if __name__ == "__main__":
    main()
