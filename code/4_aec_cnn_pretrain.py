from __future__ import annotations

# Stage-2 1D-CNN variant: self-supervised encoder pretraining.
#
# Reuses 3_aec_cnn_reclassify.py entirely (data loading, ResidualCNN architecture,
# fold/sweep/threshold/evaluation/plotting) and overrides exactly one thing: how the
# conv encoder's weights are initialized before supervised fine-tuning.
#
# Motivation (see docs/residual_reclassify_cnn_algorithm.md 4.6): the CNN only ever
# trains on the screen-positive subset (~500-600 internal patients), which is the
# real ceiling on what the small conv encoder can learn from scratch each fold. But
# the *unlabeled* residual curve shape is available for the FULL internal cohort
# (n=1090) before Stage-1 screening even happens -- curve_mu/curve_sd in the base
# script is already fit on that full set. This variant spends that extra unlabeled
# signal: before the 5-fold supervised loop starts, it pretrains a denoising
# autoencoder (same conv encoder + a throwaway linear decoder) on ALL internal
# patients' standardized residual curves, then every fold/seed's ResidualCNN starts
# its conv weights from that pretrained encoder instead of random init.
#
# This mirrors exactly the "unsupervised representation learning on more data, then
# fine-tune on the small labeled subset" pattern -- cheap here because the encoder
# is tiny (~700 params) and the pretext task (reconstruct a denoised curve) needs no
# labels, so it carries none of the label-leakage risk a supervised pretraining step
# would.
#
# Run: python code/4_aec_cnn_pretrain.py

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("3_aec_cnn_reclassify")  # sets KMP_DUPLICATE_LIB_OK before its own torch import

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

setattr(base, "OUTPUT_DIR", base.PROJECT_ROOT / "outputs" / "4_aec_cnn_pretrain")

AE_MAX_EPOCHS = 100
AE_PATIENCE = 10
AE_VAL_FRAC = 0.2

# Cache: pretraining is expensive-ish and its outcome doesn't depend on the sweep
# config (side features) or fold, only on the full internal curve set, which is the
# same array on every call -- so pretrain once and reuse the frozen conv weights for
# every fold/seed/sweep-config/final-freeze model built afterward.
_PRETRAIN_STATE: dict[str, dict] = {}


class ResidualAutoencoder(nn.Module):
    # Same conv encoder as base.ResidualCNN.conv, plus a small linear decoder that
    # is discarded after pretraining -- only self.conv's weights get reused.
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=9, padding=4), nn.BatchNorm1d(6), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(6, base.CONV_EMBED_DIM, kernel_size=5, padding=2), nn.BatchNorm1d(base.CONV_EMBED_DIM), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base.CONV_EMBED_DIM, base.CONV_EMBED_DIM, kernel_size=3, padding=1), nn.BatchNorm1d(base.CONV_EMBED_DIM), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(base.CONV_EMBED_DIM, 64), nn.ReLU(),
            nn.Linear(64, base.N_SLICES),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        z = self.conv(curve).squeeze(-1)
        return self.decoder(z)


def pretrain_encoder(curve_all: np.ndarray, seed: int) -> dict:
    idx = np.arange(len(curve_all))
    tr_idx, va_idx = train_test_split(idx, test_size=AE_VAL_FRAC, random_state=seed)

    base.set_seed(seed)
    ae = ResidualAutoencoder().to(base.DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    curve_tr = torch.tensor(curve_all[tr_idx], dtype=torch.float32, device=base.DEVICE).unsqueeze(1)
    curve_va = torch.tensor(curve_all[va_idx], dtype=torch.float32, device=base.DEVICE).unsqueeze(1)

    best_loss, best_state, patience_left = float("inf"), None, AE_PATIENCE
    n = curve_tr.shape[0]
    for _epoch in range(AE_MAX_EPOCHS):
        ae.train()
        perm = torch.randperm(n, device=base.DEVICE)
        for start in range(0, n, base.BATCH_SIZE):
            batch_idx = perm[start:start + base.BATCH_SIZE]
            clean = curve_tr[batch_idx]
            noised = clean + torch.randn_like(clean) * base.NOISE_STD
            opt.zero_grad()
            recon = ae(noised)
            loss = loss_fn(recon, clean.squeeze(1))
            loss.backward()
            opt.step()

        ae.eval()
        with torch.no_grad():
            noised_va = curve_va + torch.randn_like(curve_va) * base.NOISE_STD
            val_loss = float(loss_fn(ae(noised_va), curve_va.squeeze(1)).item())
        ae.train()
        if val_loss < best_loss - 1e-5:
            best_loss, patience_left = val_loss, AE_PATIENCE
            best_state = {k: v.detach().clone() for k, v in ae.conv.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best_state is not None
    print(f"[pretrain] best AE val MSE={best_loss:.4f} (seed={seed})")
    return best_state


def train_cnn_ensemble_pretrained(curve: np.ndarray, side: np.ndarray, y: np.ndarray,
                                   base_seed: int, n_seeds: int = base.N_SEEDS) -> list:
    conv_state = _PRETRAIN_STATE["conv"]
    best_epoch = base.find_best_epoch(curve, side, y, base_seed)
    curve_t = torch.tensor(curve, dtype=torch.float32, device=base.DEVICE).unsqueeze(1)
    side_t = torch.tensor(side, dtype=torch.float32, device=base.DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=base.DEVICE)
    loss_fn = base._make_loss(y)

    models = []
    for member in range(n_seeds):
        seed = base_seed * 10 + member
        base.set_seed(seed)
        model = base.ResidualCNN(side.shape[1]).to(base.DEVICE)
        model.conv.load_state_dict(conv_state)  # init from the pretrained encoder instead of from scratch
        opt = torch.optim.Adam(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
        base._run_epochs(model, opt, loss_fn, curve_t, side_t, y_t, n_epochs=best_epoch)
        model.eval()
        models.append(model)
    return models


_orig_stage2_oof_scores_cnn = base.stage2_oof_scores_cnn  # capture before monkeypatching below


def stage2_oof_scores_cnn_pretrained(curve: np.ndarray, side: np.ndarray, y: np.ndarray,
                                      pos_mask: np.ndarray) -> np.ndarray:
    if "conv" not in _PRETRAIN_STATE:
        # `curve` here is curve_std_int -- the FULL internal cohort (not just
        # screen-positive), computed once in base.main() before pos_mask is applied.
        _PRETRAIN_STATE["conv"] = pretrain_encoder(curve, base.SEED)
    return _orig_stage2_oof_scores_cnn(curve, side, y, pos_mask)


setattr(base, "train_cnn_ensemble", train_cnn_ensemble_pretrained)
setattr(base, "stage2_oof_scores_cnn", stage2_oof_scores_cnn_pretrained)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
