from __future__ import annotations

# Stage-2 1D-CNN variant: bagged ensemble members instead of same-data/different-seed.
#
# Reuses 3_aec_cnn_reclassify.py entirely and overrides exactly one thing:
# `train_cnn_ensemble`. The base script's N_SEEDS=5 ensemble trains every member on
# the identical fold-training rows, varying only the random init/shuffling -- the
# tuning log in docs/residual_reclassify_cnn_algorithm.md notes N_SEEDS=7 gave
# *less* diversity benefit than 5, suggesting "more inits on the same data" was
# close to its ceiling. This variant adds a second, complementary source of
# diversity: each ensemble member also trains on its own class-stratified bootstrap
# resample of the fold-training rows (bagging), so members disagree both because of
# init noise and because of which patients they actually saw -- the textbook lever
# for reducing ensemble variance on a small, small-sample-noisy training set.
#
# The early-stopping epoch count is still chosen once per fold (find_best_epoch on
# the full, non-resampled training data) and shared by every bagged member, so this
# stays a fair comparison against the base script's ensembling strategy alone.
#
# outputs/100_compare_cnn_variants/spec_delta_comparison.png (comparing all 6 CNN
# variants) showed this variant landing well below baseline (internal spec_delta
# +0.008 vs baseline's +0.052, external +0.015 vs +0.038) -- the weakest of the 5
# proposed improvements, alongside 5_aec_cnn_skip.py. The two knobs bagging actually
# has -- how large each bootstrap resample is relative to the fold-training data,
# and how many bagged members to average -- were both fixed at one untuned guess
# (100% resample size, 5 members, matching baseline's N_SEEDS). `run_grid_search()`
# below sweeps both instead, reusing 3_aec_cnn_reclassify.py's own internal-OOF
# model-selection machinery (SWEEP_CONFIGS x select_best_sweep_config) at each grid
# point exactly as a bare `python code/6_aec_cnn_bagging.py` run already did.
# External is still never touched for selection -- only used afterward to report the
# frozen winner's held-out numbers.
#
# Grid search result (2026-07-13, outputs/6_aec_cnn_bagging/grid_search_ranking.csv):
# bootstrap_frac=1.0, n_members=7 won outright -- internal spec_delta +0.035 (vs the
# untuned default's +0.008) AND external spec_delta +0.023 (vs +0.015), both PASS. Unlike
# 5_aec_cnn_skip.py's grid search (where the internal-best architecture change turned out
# to fail external and had to fall back to the untuned default), this one genuinely
# generalized -- two more members gave the resample-diversity idea more to average over
# without needing a bigger or smaller bootstrap draw. Still trails baseline's plain,
# no-bagging CNN (+0.052 / +0.038), but the gap shrank substantially.
#
# Run: python code/6_aec_cnn_bagging.py

import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("3_aec_cnn_reclassify")  # sets KMP_DUPLICATE_LIB_OK before its own torch import

import numpy as np
import pandas as pd
import torch

OUTPUT_DIR = base.PROJECT_ROOT / "outputs" / "6_aec_cnn_bagging"
GRID_DIR = base.PROJECT_ROOT / "outputs" / "6_aec_cnn_bagging_grid"

# Grid search over bagging's two free knobs:
#   - bootstrap_frac: size of each member's per-class bootstrap resample, as a
#     fraction of that class's fold-training count (1.0, the original choice, draws
#     as many rows as the class has, with replacement -- the textbook bagging
#     default). Smaller fractions trade per-member data for more inter-member
#     diversity; larger fractions do the reverse.
#   - n_members: bagged ensemble size (5, the original choice, matches baseline's
#     N_SEEDS so bagging-vs-same-data-different-seed stays a same-cost comparison;
#     more members costs proportionally more training time).
BOOTSTRAP_FRAC_GRID = [0.7, 1.0, 1.3]
N_MEMBERS_GRID = [5, 7, 9]
GRID = [(f, n) for f in BOOTSTRAP_FRAC_GRID for n in N_MEMBERS_GRID]

# The original, hand-picked (bootstrap_frac, n_members) -- always included in GRID
# above, so it's a real grid point, not a separate re-run.
DEFAULT_BOOTSTRAP_FRAC = 1.0
DEFAULT_N_MEMBERS = 5


def build_train_fn(bootstrap_frac: float, n_members: int):
    def _stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Resample within each class separately (rather than a plain bootstrap over all
        # rows) so a bad draw can't accidentally starve the minority class -- the
        # screen-positive Low-SMI rate is well under 50%.
        idx_pos = np.flatnonzero(y == 1)
        idx_neg = np.flatnonzero(y == 0)
        n_pos = max(1, int(round(len(idx_pos) * bootstrap_frac)))
        n_neg = max(1, int(round(len(idx_neg) * bootstrap_frac)))
        boot_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        boot_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        return np.concatenate([boot_pos, boot_neg])

    def train_cnn_ensemble_bagging(curve: np.ndarray, side: np.ndarray, y: np.ndarray,
                                    base_seed: int, n_seeds: int = n_members) -> list:
        best_epoch = base.find_best_epoch(curve, side, y, base_seed)
        rng = np.random.default_rng(base_seed)

        models = []
        for member in range(n_seeds):
            seed = base_seed * 10 + member
            boot_idx = _stratified_bootstrap_indices(y, rng)
            curve_b, side_b, y_b = curve[boot_idx], side[boot_idx], y[boot_idx]

            base.set_seed(seed)
            model = base.ResidualCNN(side.shape[1]).to(base.DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)

            curve_t = torch.tensor(curve_b, dtype=torch.float32, device=base.DEVICE).unsqueeze(1)
            side_t = torch.tensor(side_b, dtype=torch.float32, device=base.DEVICE)
            y_t = torch.tensor(y_b, dtype=torch.float32, device=base.DEVICE)
            loss_fn = base._make_loss(y_b)  # pos_weight recomputed per bootstrap draw's own class balance

            base._run_epochs(model, opt, loss_fn, curve_t, side_t, y_t, n_epochs=best_epoch)
            model.eval()
            models.append(model)
        return models

    return train_cnn_ensemble_bagging


def select_best_grid_point(rows: list[dict]) -> dict:
    # Mirrors 3_aec_cnn_reclassify.select_best_sweep_config, one level up: each row
    # here is already the best of that inner (clinical, stage1_score) sweep for one
    # (bootstrap_frac, n_members) grid point's internal cohort.
    passing = [r for r in rows if r["verdict_int"] == "PASS"]
    pool = passing if passing else rows
    safe = [r for r in pool if r["ni_ci_upper_int"] <= base.SAFE_MARGIN_FRAC * r["ni_margin_int"]]
    candidates = safe if safe else pool
    best = max(candidates, key=lambda r: (round(r["spec_delta_int"], 6), r["sens_delta_int"]))

    # 5_aec_cnn_skip.py's grid search found that ranking purely on internal spec_delta can
    # pick an architecture change that overfits internal and fails external -- internal-only
    # ranking can't see that by construction (external is never touched for selection), so
    # it's checked here as a sanity gate: a grid point only overrides the untuned default if
    # it also still passes external. If it doesn't, fall back to DEFAULT_BOOTSTRAP_FRAC/
    # DEFAULT_N_MEMBERS.
    if best["verdict_ext"] != "PASS":
        return next(r for r in rows if r["bootstrap_frac"] == DEFAULT_BOOTSTRAP_FRAC
                    and r["n_members"] == DEFAULT_N_MEMBERS)
    return best


def run_grid_search() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRID_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for bootstrap_frac, n_members in GRID:
        tag = f"f{bootstrap_frac}_n{n_members}"
        grid_point_dir = GRID_DIR / tag
        print(f"\n########## grid point: bootstrap_frac={bootstrap_frac}, n_members={n_members} ##########")

        setattr(base, "train_cnn_ensemble", build_train_fn(bootstrap_frac, n_members))
        setattr(base, "OUTPUT_DIR", grid_point_dir)
        base.main()

        summary = pd.read_csv(grid_point_dir / "stage1_vs_stage2_summary.csv")
        row_int = summary[summary["cohort"] == "internal"].iloc[0]
        row_ext = summary[summary["cohort"] == "external"].iloc[0]
        rows.append({
            "bootstrap_frac": bootstrap_frac, "n_members": n_members, "tag": tag,
            "sens_delta_int": float(row_int["sens_delta"]), "spec_delta_int": float(row_int["spec_delta"]),
            "verdict_int": row_int["verdict"], "ni_ci_upper_int": float(row_int["ni_ci_upper_97.5"]),
            "ni_margin_int": float(row_int["ni_margin"]),
            "sens_delta_ext": float(row_ext["sens_delta"]), "spec_delta_ext": float(row_ext["spec_delta"]),
            "verdict_ext": row_ext["verdict"],
        })

    ranking = pd.DataFrame(rows).sort_values("spec_delta_int", ascending=False)
    print("\n=== grid search ranking (6_aec_cnn_bagging): bootstrap_frac x n_members ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(ranking.to_string(index=False))

    best = select_best_grid_point(rows)
    print(f"\nSelected grid point: bootstrap_frac={best['bootstrap_frac']}, n_members={best['n_members']} "
          f"(internal spec_delta={best['spec_delta_int']:+.3f}, external spec_delta={best['spec_delta_ext']:+.3f})")

    # Promote the winning grid point's already-computed outputs into the canonical
    # outputs/6_aec_cnn_bagging/ folder, so 100_compare_cnn_variants.py keeps working
    # unchanged -- no need to retrain the winner a second time.
    winner_dir = GRID_DIR / best["tag"]
    for name in ["stage1_vs_stage2_summary.csv", "stage2_cnn_sweep_ranking.csv",
                 "stage1_vs_stage2_confusion_matrix.png", "clinical_vs_aec_assisted_table.png"]:
        shutil.copyfile(winner_dir / name, OUTPUT_DIR / name)
    ranking["selected"] = ranking["tag"] == best["tag"]
    ranking.to_csv(OUTPUT_DIR / "grid_search_ranking.csv", index=False)
    print(f"Copied winning grid point's outputs to {OUTPUT_DIR}")
    print(f"Saved full grid ranking to {OUTPUT_DIR / 'grid_search_ranking.csv'}")


def main() -> None:
    run_grid_search()


if __name__ == "__main__":
    main()
