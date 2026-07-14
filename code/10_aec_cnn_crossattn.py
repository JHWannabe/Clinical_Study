from __future__ import annotations

# Stage-2 1D-CNN variant: cross-attention fusion instead of late concat / FiLM.
#
# Reuses 3_aec_cnn_reclassify.py entirely and overrides exactly one thing: the
# `ResidualCNN` class. Baseline reduces the whole curve to one embedding via
# global average pooling *before* it ever sees the side features (clinical /
# stage-1 score) -- so every curve position contributes to that embedding with
# the same fixed weight, for every patient. 8_aec_cnn_film.py already showed
# that letting side features condition the curve embedding (FiLM) is worth
# trying; cross-attention is a stronger version of that idea: instead of one
# global scale/shift applied uniformly across the curve, the side features
# build a *query* that attends over the individual curve positions (one
# key/value token per position, taken from block3's output before pooling) and
# picks out a weighted combination of *specific* positions -- i.e. a patient's
# clinical profile can change *which part of the residual curve* the model
# reads, not just how the whole-curve summary gets rescaled. This is the same
# query/key/value mechanism as Transformer cross-attention (Vaswani et al.,
# 2017) and multimodal fusion models that attend one modality over another
# (e.g. Lu et al., 2019, ViLBERT), just with "modality A" = one side-feature
# vector per patient and "modality B" = the 32-position curve-embedding
# sequence.
#
# Concretely: block1->block2->block3 (identical conv stack to baseline, minus
# the final AdaptiveAvgPool1d) leaves a (B, CONV_EMBED_DIM, 32) sequence. When
# side features exist, a linear layer projects them to a single query token
# (B, 1, CONV_EMBED_DIM); nn.MultiheadAttention attends that query over the 32
# curve tokens (used as both key and value) to produce one context vector per
# patient. When no side features are configured (the "curve only" sweep
# entry), there is nothing to build a query from, so this falls back to
# baseline's plain global-average-pool path unchanged.
#
# Everything else -- data loading, side-feature construction, early-stopping,
# N_SEEDS ensembling, augmentation, fold/sweep/threshold/evaluation/plotting -- is
# untouched.
#
# First attempt fed the attention context straight into the head (concat with
# side features), replacing the pooled embedding outright whenever side
# features existed. All three side-conditioned sweep configs landed on the
# exact same sens_delta=-0.0155 that 8_aec_cnn_film.py's first (unbounded)
# attempt also hit, which pinned their non-inferiority CI at 97.8% of the
# allowed margin (ni_ci_upper=0.049 vs margin=0.05) -- outside
# 3_aec_cnn_reclassify.py's SAFE_MARGIN_FRAC=0.8 safety filter (0.04). With
# every side-using config rejected as unsafe, select_best_sweep_config fell
# back to the only "safe" one left: curve-only (no side features at all),
# which is a near no-op (internal spec_delta +0.001) and reclassified zero
# external patients (spec_delta=0.0 -> FAIL). So the first version never
# actually got to test whether cross-attention helps -- the safety filter
# just always steered it back to the do-nothing config.
#
# Fix: same identity-at-init strategy 8_aec_cnn_film.py's fixed version (and
# 5_aec_cnn_skip.py's ReZero attempt) used. Instead of replacing the pooled
# embedding, the attention context is mixed in through a learnable per-channel
# gate initialized to zero: `z = pooled + gate * context`. At init gate=0, so
# z == pooled exactly -- training starts from baseline's own known-good
# GAP embedding and only pulls in attended, side-conditioned context if doing
# so actually earns it, instead of committing to the side-conditioned
# representation from epoch 1 the way the first attempt did.
#
# With the gate, the sweep no longer collapses to the no-side-feature
# fallback: (clinical=True, stage1_score=False) is now safe on its own
# (ni_ci_upper=0.037 vs margin=0.05) and gets selected, giving internal
# spec_delta +0.015 / external +0.032, both PASS/NON-INFERIOR -- a real,
# non-degenerate result (vs. the unbounded version's spec_delta~0/FAIL), but
# still below baseline's plain-concat CNN (+0.052/+0.038). Same conclusion as
# 8_aec_cnn_film.py's bounded fix: identity-at-init makes the side-conditioned
# idea usable rather than self-defeating, but neither FiLM nor cross-attention
# has yet beaten baseline's much simpler late-concat fusion on this cohort
# size -- see docs/cnn_variant_comparison.md.
#
# Run: python code/10_aec_cnn_crossattn.py

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("3_aec_cnn_reclassify")  # sets KMP_DUPLICATE_LIB_OK before its own torch import

import torch
import torch.nn as nn

setattr(base, "OUTPUT_DIR", base.PROJECT_ROOT / "outputs" / "10_aec_cnn_crossattn")

# CONV_EMBED_DIM (10) must be divisible by NUM_HEADS for nn.MultiheadAttention.
NUM_HEADS = 2


class ResidualCNNCrossAttn(nn.Module):
    def __init__(self, n_side_features: int) -> None:
        super().__init__()
        self.n_side = n_side_features
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=9, padding=4), nn.BatchNorm1d(6), nn.ReLU(), nn.MaxPool1d(2),
        )  # 128 -> 64
        self.block2 = nn.Sequential(
            nn.Conv1d(6, base.CONV_EMBED_DIM, kernel_size=5, padding=2), nn.BatchNorm1d(base.CONV_EMBED_DIM), nn.ReLU(), nn.MaxPool1d(2),
        )  # 64 -> 32, channels -> CONV_EMBED_DIM
        self.block3 = nn.Sequential(
            nn.Conv1d(base.CONV_EMBED_DIM, base.CONV_EMBED_DIM, kernel_size=3, padding=1), nn.BatchNorm1d(base.CONV_EMBED_DIM), nn.ReLU(),
        )  # 32 -> 32; left as a sequence (no pooling) so attention can pick individual positions

        self.pool = nn.AdaptiveAvgPool1d(1)  # baseline's own GAP -- always computed, the gate's identity target
        if n_side_features > 0:
            self.query_proj = nn.Linear(n_side_features, base.CONV_EMBED_DIM)
            self.attn = nn.MultiheadAttention(embed_dim=base.CONV_EMBED_DIM, num_heads=NUM_HEADS,
                                               dropout=base.DROPOUT, batch_first=True)
            self.gate = nn.Parameter(torch.zeros(base.CONV_EMBED_DIM))  # 0-init: z == pooled at the start of training
            head_in = base.CONV_EMBED_DIM + n_side_features
        else:
            head_in = base.CONV_EMBED_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, base.CONV_EMBED_DIM), nn.ReLU(), nn.Dropout(base.DROPOUT),
            nn.Linear(base.CONV_EMBED_DIM, 1),
        )

    def forward(self, curve: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        h = self.block3(self.block2(self.block1(curve)))  # (B, CONV_EMBED_DIM, 32)
        pooled = self.pool(h).squeeze(-1)  # (B, CONV_EMBED_DIM)
        if self.n_side > 0:
            tokens = h.transpose(1, 2)  # (B, 32, CONV_EMBED_DIM) -- key/value, one token per curve position
            query = self.query_proj(side).unsqueeze(1)  # (B, 1, CONV_EMBED_DIM) -- one query per patient
            context, _ = self.attn(query, tokens, tokens, need_weights=False)
            z = pooled + self.gate * context.squeeze(1)
            z = torch.cat([z, side], dim=1)
        else:
            z = pooled
        return self.head(z).squeeze(-1)


setattr(base, "ResidualCNN", ResidualCNNCrossAttn)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
