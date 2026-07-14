from __future__ import annotations

# Stage-2 1D-CNN variant: replace the repo's small user-defined ResidualCNN with
# FCN (Fully Convolutional Network), the most widely used "off-the-shelf" 1D-CNN
# architecture for time-series/1D-signal classification (Wang, Yan & Oates, 2017,
# "Time Series Classification from Scratch with Deep Neural Networks: A Strong
# Baseline", https://arxiv.org/abs/1611.06455) -- the standard 1D-CNN comparison
# point in the UCR/UEA time-series-classification benchmark and in many
# biomedical-signal papers (ECG/EEG/PPG classification, etc), as opposed to the
# hand-designed tiny net baseline uses.
#
# What's different from 3_aec_cnn_reclassify.py's ResidualCNN:
#   - 3 conv blocks at the paper's standard channel widths/kernel sizes
#     (128ch/k=8 -> 256ch/k=5 -> 128ch/k=3), each Conv1d -> BatchNorm1d -> ReLU,
#     with 'same' padding so the 128-point curve length is preserved through all
#     three blocks (baseline instead shrinks the curve with two MaxPool1d(2) and
#     uses much narrower 6/10/10 channels).
#   - Global average pooling over the last block's 128 channels, concatenated with
#     side features (clinical / stage-1 score) and passed through the exact same
#     2-layer head (Linear -> ReLU -> Dropout -> Linear) baseline uses.
# Nothing else changes: training loop, N_SEEDS ensembling, early stopping,
# augmentation, fold/sweep/threshold selection, evaluation, and plotting are all
# reused unmodified via monkeypatching (see docs/cnn_variant_comparison.md section 1).
#
# Caveat worth flagging up front: FCN's standard width (128/256/128 channels) gives
# this model ~330k parameters -- roughly 150x baseline's ResidualCNN (~2k params)
# -- trained on only ~560 screen-positive internal patients. The paper's channel
# widths were tuned on UCR benchmark datasets with thousands of rows each; on a
# cohort this small, "more commonly used elsewhere" does not automatically mean
# "better suited to this sample size". That's exactly the empirical question this
# script exists to answer rather than assume, same as the other variants compared
# in docs/cnn_variant_comparison.md.
#
# Run: python code/9_aec_cnn_fcn.py

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("3_aec_cnn_reclassify")  # sets KMP_DUPLICATE_LIB_OK before its own torch import

import torch
import torch.nn as nn

setattr(base, "OUTPUT_DIR", base.PROJECT_ROOT / "outputs" / "9_aec_cnn_fcn")

FCN_CHANNELS = (128, 256, 128)
FCN_KERNELS = (8, 5, 3)


class FCN1D(nn.Module):
    # Standard time-series FCN encoder (Wang et al., 2017) feeding baseline's own
    # [curve embedding | side features] -> linear head, so only the conv
    # architecture itself differs from ResidualCNN.
    def __init__(self, n_side_features: int) -> None:
        super().__init__()
        c1, c2, c3 = FCN_CHANNELS
        k1, k2, k3 = FCN_KERNELS
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k1, padding="same"), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding="same"), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k3, padding="same"), nn.BatchNorm1d(c3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(c3 + n_side_features, c3), nn.ReLU(), nn.Dropout(base.DROPOUT),
            nn.Linear(c3, 1),
        )

    def forward(self, curve: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        z = self.conv(curve).squeeze(-1)  # (B, c3)
        if side.shape[1] > 0:
            z = torch.cat([z, side], dim=1)
        return self.head(z).squeeze(-1)


setattr(base, "ResidualCNN", FCN1D)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
