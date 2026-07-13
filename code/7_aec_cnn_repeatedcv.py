from __future__ import annotations

# Stage-2 1D-CNN variant: repeated stratified 5-fold OOF instead of a single split.
#
# Reuses 3_aec_cnn_reclassify.py entirely and overrides exactly one thing:
# `stage2_oof_scores_cnn`. docs/residual_reclassify_cnn_algorithm.md section 6
# flags that this pipeline's external spec_delta swung from +0.014 to +0.038 (~3x)
# just from changing N_SEEDS -- evidence that a lot of the reported variance comes
# from picking th2 off a *single* StratifiedKFold(5, shuffle, seed=SEED) split, not
# from the CNN itself. This variant repeats that whole 5-fold OOF procedure
# N_REPEATS times with different fold-shuffle seeds and averages each patient's
# logit across repeats before choose_stage2_threshold ever sees it -- the same
# "repeated k-fold" trick used for stabilizing CV estimates generally, applied here
# to the threshold-selection signal specifically.
#
# Cost: N_REPEATS x as many models trained as the base script. Model/architecture,
# side features, augmentation, and the sweep/threshold/evaluation/plotting code are
# all untouched, so any change in outcome should be attributable to OOF-averaging
# alone.
#
# Run: python code/7_aec_cnn_repeatedcv.py

import importlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("3_aec_cnn_reclassify")  # sets KMP_DUPLICATE_LIB_OK before its own torch import

import numpy as np
from sklearn.model_selection import StratifiedKFold

setattr(base, "OUTPUT_DIR", base.PROJECT_ROOT / "outputs" / "7_aec_cnn_repeatedcv")

# sens_delta on this internal cohort (~129 screen-positive) is quantized to whole
# patients: losing 0, 1, or 2 of them maps to a non-inferiority CI-upper/margin ratio
# of 0.47 / 0.74 / 0.98 respectively -- essentially fixed regardless of which variant
# produced the flip (see stage2_cnn_sweep_ranking.csv across scripts 3/5/6/7/8, all
# land on exactly these three ratios). 3_aec_cnn_reclassify.py's sweep-selection
# safety filter (SAFE_MARGIN_FRAC=0.8) treats the 2-patient/0.98 tier as too close to
# the boundary to trust, which is exactly where this variant's two side-feature
# configs sit at N_REPEATS=5 -- so both get excluded and the curve-only config (no
# side features, tiny effect) is selected by default. Raising N_REPEATS is the one
# lever this file controls that's actually on-thesis: if the 2nd flipped patient at
# N_REPEATS=5 is a single-split noise artifact rather than a signal the CNN
# consistently finds, more repeats should average it away and drop that config to
# the 1-patient/0.74 tier, where it'd clear the safety filter with its spec_delta
# gain intact. If it's a genuine, reproducible flip, more repeats won't move it.
N_REPEATS = 10


def stage2_oof_scores_cnn_repeated(curve: np.ndarray, side: np.ndarray, y: np.ndarray,
                                    pos_mask: np.ndarray) -> np.ndarray:
    all_scores = np.full((N_REPEATS, len(y)), np.nan)
    for rep in range(N_REPEATS):
        repeat_seed = base.SEED + rep * 1000  # distinct fold-shuffle per repeat
        scores = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=base.N_FOLDS, shuffle=True, random_state=repeat_seed)
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(curve, y)):
            tr_pos = tr_idx[pos_mask[tr_idx]]
            va_pos = va_idx[pos_mask[va_idx]]
            if len(va_pos) == 0 or len(np.unique(y[tr_pos])) < 2:
                continue
            models = base.train_cnn_ensemble(curve[tr_pos], side[tr_pos], y[tr_pos], repeat_seed + fold_id)
            scores[va_pos] = base.predict_ensemble(models, curve[va_pos], side[va_pos])
        all_scores[rep] = scores
    # nanmean: a patient missing from a given repeat's screen-positive val fold (rare,
    # only if a fold's positive count degenerates) just contributes fewer repeats
    # rather than propagating a NaN through the whole average. A patient who is
    # screen-negative (pos_mask False) in every repeat is legitimately all-NaN --
    # that's the expected "mean of empty slice" case, not a bug, so it's silenced.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(all_scores, axis=0)


setattr(base, "stage2_oof_scores_cnn", stage2_oof_scores_cnn_repeated)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
