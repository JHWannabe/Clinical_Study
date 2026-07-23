from __future__ import annotations

# Standalone architecture comparison: AecCNN (hand-designed, code/sinchon_only_pipeline.py)
# vs. ResNet1D (open-source 1D-CNN benchmark, see ResNet1D docstring below) as the AEC-128
# curve -> low-SMI classifier, both scored 5-fold OOF over the same Sinchon screen-positive
# subset under the IDENTICAL training recipe (sp._train_aec_cnn: same optimizer, LR-plateau
# schedule, grad clipping, early stopping, pos_weight criterion, seed-ensemble). Neither
# model's hyperparameters are grid-searched here (AecCNN uses the pipeline's untuned
# defaults, ResNet1D uses the original paper's defaults) -- the point is to compare the two
# ARCHITECTURES under matched training conditions, not to find the best config of either.
#
# Reuses sinchon_only_pipeline.py (Stage-1 LR + screen-positive subset extraction, AecCNN,
# the generic per-model training loop, and the loss-curve plot) instead of duplicating that
# setup.
#
# Run: python code/sinchon_aec_cnn_vs_resnet1d.py

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sp = import_module("sinchon_only_pipeline")
baseline = sp.baseline
stage2_dataset = sp.stage2_dataset
stage2_model = sp.stage2_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sinchon_aec_cnn_vs_resnet1d"


class ResNet1D(nn.Module):
    # Faithful port of the ResNet baseline from Wang, Yan & Oates, "Time Series
    # Classification from Scratch with Deep Neural Networks: A Strong Baseline"
    # (IJCNN 2017) -- the standard open-source 1D-CNN benchmark for time-series
    # classification (as implemented in hfawaz/dl-4-tsc, classifiers/resnet.py). Three
    # residual blocks of [Conv(k=8)-Conv(k=5)-Conv(k=3)] (feature maps 64/128/128), each
    # with a BN-only shortcut when channels match and a 1x1-conv+BN shortcut when they
    # don't, then global-average-pool + linear head. Only the input (single-channel
    # 128-slice AEC curve) and output (single logit instead of softmax over classes) are
    # adapted -- the block structure/kernel sizes/feature-map counts are unmodified from
    # the paper, so this is a like-for-like "open-source architecture" comparison against
    # the hand-designed AecCNN, not a redesign.
    def __init__(self, n_slices: int = 128, n_feature_maps: int = 64) -> None:
        super().__init__()

        def conv_bn(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
            # TF/Keras "same" padding (asymmetric for even kernel_size, e.g. the paper's
            # kernel_size=8 blocks) -- PyTorch's symmetric Conv1d(padding=k//2) would grow
            # the length by 1 on every even-kernel conv instead of preserving it.
            total_pad = kernel_size - 1
            left, right = total_pad // 2, total_pad - total_pad // 2
            return nn.Sequential(
                nn.ZeroPad1d((left, right)),
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=0),
                nn.BatchNorm1d(out_ch),
            )

        def make_block(in_ch: int, out_ch: int) -> tuple[nn.Sequential, nn.Module]:
            convs = nn.Sequential(
                conv_bn(in_ch, out_ch, 8), nn.ReLU(),
                conv_bn(out_ch, out_ch, 5), nn.ReLU(),
                conv_bn(out_ch, out_ch, 3),
            )
            shortcut = nn.BatchNorm1d(out_ch) if in_ch == out_ch else conv_bn(in_ch, out_ch, 1)
            return convs, shortcut

        self.block1_convs, self.block1_shortcut = make_block(1, n_feature_maps)
        self.block2_convs, self.block2_shortcut = make_block(n_feature_maps, n_feature_maps * 2)
        self.block3_convs, self.block3_shortcut = make_block(n_feature_maps * 2, n_feature_maps * 2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_feature_maps * 2, 1)

    def init_output_bias(self, prior: float) -> None:
        logit = float(np.log(prior / (1 - prior)))
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, logit)

    def forward(self, x_aec: torch.Tensor) -> torch.Tensor:
        x = (x_aec - 1.0).unsqueeze(1)  # center: curves are patient-normalized to mean~1, same convention as AecCNN
        x = torch.relu(self.block1_convs(x) + self.block1_shortcut(x))
        x = torch.relu(self.block2_convs(x) + self.block2_shortcut(x))
        x = torch.relu(self.block3_convs(x) + self.block3_shortcut(x))
        pooled = self.avg_pool(x).squeeze(-1)
        return self.head(pooled).squeeze(-1)  # logit


def _oof_scores_generic(model_ctor, model_name: str, x_aec: torch.Tensor, y: np.ndarray,
                         lr: float = stage2_model.LR, weight_decay: float = stage2_model.WEIGHT_DECAY,
                         n_ensemble_seeds: int = stage2_model.N_ENSEMBLE_SEEDS) -> tuple[np.ndarray, list[list[float]]]:
    # Same 5-fold OOF + seed-ensemble structure as sp.aec_cnn_oof_scores, generalized to
    # any model_ctor() -> nn.Module with (forward(x_aec) -> logit, init_output_bias) --
    # lets AecCNN and ResNet1D share the exact same OOF/training loop for a fair comparison.
    oof = np.zeros(len(y), dtype=float)
    fold_loss_histories: list[list[float]] = []
    skf = StratifiedKFold(n_splits=stage2_model.N_FOLDS, shuffle=True, random_state=stage2_model.SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(x_aec.numpy(), y)):
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        preds = []
        loss_history: list[float] = []
        for i in range(n_ensemble_seeds):
            torch.manual_seed((stage2_model.SEED + fold_id) * 100 + i)
            model = model_ctor()
            model.init_output_bias(float(y_tr.mean().item()))
            lh = sp._train_aec_cnn(model, x_aec[tr_idx], y_tr, lr=lr, weight_decay=weight_decay)
            if i == 0:
                loss_history = lh
            model.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(model(x_aec[va_idx])).numpy())
        oof[va_idx] = np.mean(preds, axis=0)
        fold_loss_histories.append(loss_history)
        print(f"  [{model_name}] fold {fold_id + 1}/{stage2_model.N_FOLDS} done")
    return oof, fold_loss_histories


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta, y = baseline.load_cohort(sp.SINCHON_XLSX)
    x_raw = baseline.raw_clinical_matrix(meta)
    med, mu, sd = baseline.fit_clinical_standardizer(x_raw)
    x = baseline.apply_clinical_standardizer(x_raw, med, mu, sd)
    oof1 = baseline.oof_scores(x, y)
    th = baseline.threshold_for_sensitivity(y, oof1, baseline.TARGET_SENSITIVITY)

    _, stage1_rows_pos, _, stage2_aec = stage2_dataset._stage1_positive_rows(sp.SINCHON_XLSX, meta, y, oof1, th, x)
    y2 = (stage1_rows_pos["group"] == "TP").to_numpy().astype(int)
    x_aec_t = torch.tensor(stage2_aec[sp.AEC_COLS].to_numpy(dtype=np.float32))
    n_slices = x_aec_t.shape[1]
    print(f"Screen-positive subset (AEC-only classifier target): n={len(y2)} (event={int(y2.sum())})")

    print("\n=== AecCNN (custom, hand-designed) ===")
    aec_cnn_oof, aec_cnn_loss = _oof_scores_generic(
        lambda: sp.AecCNN(n_slices=n_slices, embed_dim=16, dropout=0.2), "AecCNN", x_aec_t, y2)
    n_params_aec_cnn = sum(p.numel() for p in sp.AecCNN(n_slices=n_slices).parameters())

    print("\n=== ResNet1D (open-source: Wang et al. 2017 / dl-4-tsc baseline) ===")
    resnet_oof, resnet_loss = _oof_scores_generic(
        lambda: ResNet1D(n_slices=n_slices, n_feature_maps=64), "ResNet1D", x_aec_t, y2)
    n_params_resnet = sum(p.numel() for p in ResNet1D(n_slices=n_slices).parameters())

    auc_aec_cnn = baseline.auc_significance_stats(y2, aec_cnn_oof)
    auc_resnet = baseline.auc_significance_stats(y2, resnet_oof)
    delong = stage2_model.delong_paired_auc_test(y2.astype(float), aec_cnn_oof, resnet_oof)

    print(f"\n[AEC-only, 5-fold OOF] AecCNN   AUC={auc_aec_cnn['auc']:.3f} "
          f"[{auc_aec_cnn['ci_lower']:.3f}, {auc_aec_cnn['ci_upper']:.3f}]  (n_params={n_params_aec_cnn})")
    print(f"[AEC-only, 5-fold OOF] ResNet1D AUC={auc_resnet['auc']:.3f} "
          f"[{auc_resnet['ci_lower']:.3f}, {auc_resnet['ci_upper']:.3f}]  (n_params={n_params_resnet})")
    print(f"DeLong paired test (AecCNN vs. ResNet1D): diff={delong['diff']:+.4f} p={delong['p_value']:.4f}")

    baseline.plot_roc_curve_dual([
        (y2, aec_cnn_oof, auc_aec_cnn, f"AecCNN (custom, {n_params_aec_cnn:,} params)"),
        (y2, resnet_oof, auc_resnet, f"ResNet1D (open-source, {n_params_resnet:,} params)"),
    ], OUTPUT_DIR / "roc_aec_cnn_vs_resnet1d.png")

    sp.plot_fold_loss_curves(aec_cnn_loss, OUTPUT_DIR / "loss_curve_aec_cnn.png",
                              title="AecCNN: training loss vs. epoch (Sinchon screen-positive, 5-fold OOF)")
    sp.plot_fold_loss_curves(resnet_loss, OUTPUT_DIR / "loss_curve_resnet1d.png",
                              title="ResNet1D: training loss vs. epoch (Sinchon screen-positive, 5-fold OOF)")

    pd.DataFrame([
        {"model": "AecCNN", "n_params": n_params_aec_cnn, "auc": auc_aec_cnn["auc"],
         "ci_lower": auc_aec_cnn["ci_lower"], "ci_upper": auc_aec_cnn["ci_upper"],
         "delong_diff_vs_other": None, "delong_p_vs_other": None},
        {"model": "ResNet1D", "n_params": n_params_resnet, "auc": auc_resnet["auc"],
         "ci_lower": auc_resnet["ci_lower"], "ci_upper": auc_resnet["ci_upper"],
         "delong_diff_vs_other": delong["diff"], "delong_p_vs_other": delong["p_value"]},
    ]).to_csv(OUTPUT_DIR / "aec_cnn_vs_resnet1d_summary.csv", index=False)
    print(f"\nSaved summary to {OUTPUT_DIR / 'aec_cnn_vs_resnet1d_summary.csv'}")


if __name__ == "__main__":
    main()
