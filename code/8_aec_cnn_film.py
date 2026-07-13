from __future__ import annotations

# Stage-2 1D-CNN variant: FiLM conditioning instead of late concat.
#
# Reuses 3_aec_cnn_reclassify.py entirely and overrides exactly one thing: the
# `ResidualCNN` class. The base model computes a curve embedding and clinical/
# stage-1 side features completely independently, joining them only at the very
# last linear layer (concat -> Linear -> ReLU -> Dropout -> Linear). That layout
# can only learn additive effects of "curve shape" and "patient profile" on the
# logit -- it has no way to let a patient's clinical profile change *how the curve
# embedding itself gets read*.
#
# FiLM (Feature-wise Linear Modulation) replaces that late concat: the side
# features are projected to a per-channel (gamma, beta) pair via one small linear
# layer, which then scales/shifts the curve embedding *before* the head:
#   z' = (1 + gamma(side)) * z + beta(side)
# The film layer is zero-initialized, so at the start of training z' == z (an
# identity transform) and the model only learns to use side-conditioning if it
# actually helps -- this keeps early training as stable as the base concat model.
#
# Everything else -- data loading, side-feature construction, early-stopping,
# N_SEEDS ensembling, augmentation, fold/sweep/threshold/evaluation/plotting -- is
# untouched.
#
# First version left gamma/beta as a fully unconstrained affine projection of the
# side features. Both side-feature configs (clinical-only, clinical+stage1) learned
# to use that freedom identically aggressively: sens_delta=-0.0155 in both, pinning
# the internal non-inferiority CI at 97.8% of the allowed margin -- basically
# guaranteed to tip over on a fresh external sample (spec_delta ended up +0.084 but
# NOT NON-INFERIOR externally: 3_aec_cnn_reclassify.py's sweep-selection safety
# filter now excludes any config that thin on margin, which for FiLM meant falling
# back to the *no side feature* config, where FiLM's own film layer isn't even
# instantiated -- a config that ends up not testing FiLM at all, and which
# reclassified zero external patients (the same degenerate failure 5_aec_cnn_skip.py
# hit before its own fix). So this version bounds gamma/beta through tanh
# (FILM_SCALE below) so the side features can meaningfully condition the curve
# embedding without being able to swing predictions as hard as before -- the model
# should land on a smaller, safer sens_delta instead of always saturating near -0.0155.
#
# Run: python code/8_aec_cnn_film.py

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("3_aec_cnn_reclassify")  # sets KMP_DUPLICATE_LIB_OK before its own torch import

import torch
import torch.nn as nn

setattr(base, "OUTPUT_DIR", base.PROJECT_ROOT / "outputs" / "8_aec_cnn_film")

# Caps how far (1+gamma)*z+beta can drift from the identity transform z'=z:
# gamma in (-FILM_SCALE, FILM_SCALE) so the multiplier stays in a bounded range
# around 1, and beta in (-FILM_SCALE, FILM_SCALE) so the additive shift stays the
# same order of magnitude as z itself (z is a post-BatchNorm/ReLU/GAP embedding,
# so O(1)-scale). Bounded via tanh, which also keeps gradients well-behaved instead
# of letting a few large-side-feature patients dominate the film layer's update.
FILM_SCALE = 1.0


class ResidualCNNFiLM(nn.Module):
    def __init__(self, n_side_features: int) -> None:
        super().__init__()
        self.n_side = n_side_features
        self.conv = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=9, padding=4), nn.BatchNorm1d(6), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(6, base.CONV_EMBED_DIM, kernel_size=5, padding=2), nn.BatchNorm1d(base.CONV_EMBED_DIM), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base.CONV_EMBED_DIM, base.CONV_EMBED_DIM, kernel_size=3, padding=1), nn.BatchNorm1d(base.CONV_EMBED_DIM), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        if n_side_features > 0:
            self.film = nn.Linear(n_side_features, base.CONV_EMBED_DIM * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)  # start as identity: gamma=beta=0 -> z' = z
        self.head = nn.Sequential(
            nn.Linear(base.CONV_EMBED_DIM, base.CONV_EMBED_DIM), nn.ReLU(), nn.Dropout(base.DROPOUT),
            nn.Linear(base.CONV_EMBED_DIM, 1),
        )

    def forward(self, curve: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        z = self.conv(curve).squeeze(-1)
        if self.n_side > 0:
            gamma_raw, beta_raw = self.film(side).chunk(2, dim=1)
            gamma = FILM_SCALE * torch.tanh(gamma_raw)
            beta = FILM_SCALE * torch.tanh(beta_raw)
            z = (1 + gamma) * z + beta
        return self.head(z).squeeze(-1)


setattr(base, "ResidualCNN", ResidualCNNFiLM)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
