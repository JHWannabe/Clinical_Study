from __future__ import annotations

# Ablation: does the AEC-128 branch contribute anything to Stage-2 Net NRI, or is the
# reported improvement (+71 internal / +34 external) entirely explained by z_clin
# (frozen Stage-1 LR score) standardization alone (see stage2_model.fit_score_standardizer)?
#
# Motivation: docs/260724_Results_of_Residual_Algorithm.pptx Slide 6 shows that once
# BMI confound is controlled (propensity matching), raw AEC-128 shows no reliable
# TP-vs-FP separation (gangnam p=0.983). Slide 9 shows Net NRI going 0 -> +34 (external)
# purely from standardizing the frozen clinical score, with no AEC change. This script
# strips the AEC branch out entirely (ClinOnlyNet: z_clin -> tiny head -> logit) and
# reruns the exact same OOF / threshold-selection / Net-NRI protocol as stage2_model.py,
# so the two Net NRI numbers are directly comparable apples-to-apples.
#
# Run: python code/stage2_model_no_aec_ablation.py

import copy
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline"))
baseline = import_module("clinic-only_baseline")
stage2 = import_module("stage2_dataset")
s2model = import_module("stage2_model")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2_model_no_aec_ablation"

N_FOLDS = baseline.N_FOLDS
SEED = baseline.SEED
N_ENSEMBLE_SEEDS = s2model.N_ENSEMBLE_SEEDS


class ClinOnlyNet(nn.Module):
    # Same clinical path as LateFusionNet's clin_variant="frozen_lr" (z_clin IS the
    # standardized frozen Stage-1 LR score, 1-dim), but with the AEC branch removed
    # entirely -- fusion_head collapses to a tiny head over z_clin alone.
    def __init__(self, clin_dim: int = 1, fusion_hidden: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(clin_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_clin: torch.Tensor) -> torch.Tensor:
        return self.head(x_clin).squeeze(-1)


class ClinOnlyEnsemble:
    def __init__(self, models: list[ClinOnlyNet]) -> None:
        self.models = models

    def predict_proba(self, x_clin: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = [torch.sigmoid(m(x_clin)).numpy() for m in self.models]
        return np.mean(probs, axis=0)


def _train_with_early_stopping(model: ClinOnlyNet, x_clin_tr: torch.Tensor, y_tr: torch.Tensor) -> None:
    criterion = s2model._make_criterion(y_tr)
    optimizer = torch.optim.Adam(model.parameters(), lr=s2model.LR, weight_decay=s2model.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=s2model.LR_PLATEAU_FACTOR, patience=s2model.LR_PLATEAU_PATIENCE
    )
    model.train()
    best_loss = float("inf")
    best_state = None
    patience_ctr = 0
    for _ in range(s2model.N_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(x_clin_tr), y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), s2model.GRAD_CLIP_NORM)
        optimizer.step()
        loss_val = loss.item()
        scheduler.step(loss_val)
        if loss_val < best_loss - s2model.EARLY_STOP_MIN_DELTA:
            best_loss = loss_val
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= s2model.EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def train_fold(x_clin_tr: torch.Tensor, y_tr: torch.Tensor, x_clin_va: torch.Tensor, seed: int) -> np.ndarray:
    preds = []
    for i in range(N_ENSEMBLE_SEEDS):
        torch.manual_seed(seed * 100 + i)
        model = ClinOnlyNet(clin_dim=x_clin_tr.shape[1])
        _train_with_early_stopping(model, x_clin_tr, y_tr)
        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(x_clin_va)).numpy())
    return np.mean(preds, axis=0)


def oof_scores(x_clin: torch.Tensor, y: np.ndarray) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x_clin.numpy(), y)):
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        oof[va_idx] = train_fold(x_clin[tr_idx], y_tr, x_clin[va_idx], seed=SEED + fold_id)
    return oof


def fit_final_model(x_clin: torch.Tensor, y: np.ndarray, seed: int = SEED) -> ClinOnlyEnsemble:
    y_t = torch.tensor(y, dtype=torch.float32)
    models = []
    for i in range(N_ENSEMBLE_SEEDS):
        torch.manual_seed(seed * 100 + i)
        model = ClinOnlyNet(clin_dim=x_clin.shape[1])
        _train_with_early_stopping(model, x_clin, y_t)
        model.eval()
        models.append(model)
    return ClinOnlyEnsemble(models)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    screen = stage2.fit_internal_screen()
    stage1_rows_all_int, stage1_rows_int, stage2_input_clin_int, stage2_input_aec_int = stage2.build_stage2_inputs(screen)
    y_int = (stage1_rows_int["group"] == "TP").to_numpy().astype(int)
    score_standardizer = s2model.fit_score_standardizer(stage1_rows_int["score"].to_numpy(dtype=np.float32))
    x_clin_int, _ = s2model._to_tensors(stage2_input_clin_int, stage2_input_aec_int, stage1_rows_int,
                                         score_standardizer=score_standardizer)

    oof = oof_scores(x_clin_int, y_int)

    y_all_int = stage1_rows_all_int["group"].isin(["TP", "FN"]).to_numpy().astype(int)
    pos_mask_int = stage1_rows_all_int["group"].isin(["TP", "FP"]).to_numpy()
    stage1_only_int = baseline.evaluate("internal / stage-1 only", y_all_int, pos_mask_int, screen["th"])
    th = s2model.choose_stage2_threshold(y_all_int, pos_mask_int, oof, stage1_only_int["sens"], stage1_only_int["spec"])

    model = fit_final_model(x_clin_int, y_int)

    stage1_rows_all_ext, stage1_rows_ext, stage2_input_clin_ext, stage2_input_aec_ext = stage2.build_stage2_inputs_external(screen)
    y_ext = (stage1_rows_ext["group"] == "TP").to_numpy().astype(int)
    x_clin_ext, _ = s2model._to_tensors(stage2_input_clin_ext, stage2_input_aec_ext, stage1_rows_ext,
                                         score_standardizer=score_standardizer)
    score_ext = model.predict_proba(x_clin_ext)

    pred_all_int = s2model.combine_predictions(pos_mask_int, oof, th)
    y_all_ext, pos_mask_ext, pred_all_ext = s2model.final_pipeline_labels(stage1_rows_all_ext, score_ext, th)

    stage1_only_ext = baseline.evaluate("external / stage-1 only", y_all_ext, pos_mask_ext, screen["th"])
    result_final_int = baseline.evaluate("internal", y_all_int, pred_all_int, th)
    result_final_ext = baseline.evaluate("external", y_all_ext, pred_all_ext, th)

    auc_int = baseline.auc_significance_stats(y_int, oof)
    auc_ext = baseline.auc_significance_stats(y_ext, score_ext)

    table_rows = [
        s2model.build_clinical_vs_aec_row("internal", y_all_int, pos_mask_int, pred_all_int, stage1_only_int, result_final_int, auc_int["auc"]),
        s2model.build_clinical_vs_aec_row("external", y_all_ext, pos_mask_ext, pred_all_ext, stage1_only_ext, result_final_ext, auc_ext["auc"]),
    ]
    pd.DataFrame(table_rows).to_csv(OUTPUT_DIR / "clinical_vs_zclin_only_summary.csv", index=False)

    print("=== z_clin-ONLY ablation (AEC branch removed entirely) ===")
    for r in table_rows:
        print(f"[{r['cohort']}] AUC={r['auc']:.3f}  Net NRI={r['net_nri']:+d} (n={r['n']}, event={r['event']})  "
              f"sens: {r['sens_clin']:.3f}->{r['sens_aec']:.3f} (p={r['sens_p']:.4f})  "
              f"spec: {r['spec_clin']:.3f}->{r['spec_aec']:.3f} (p={r['spec_p']:.4f})")
    print("\n(compare against stage2_model.py's full late-fusion Net NRI: internal +71 / external +34)")


if __name__ == "__main__":
    main()
