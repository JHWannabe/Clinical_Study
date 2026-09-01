from __future__ import annotations

# "모델이 너무 빨리 학습되는데 depth를 늘리면 개선되지 않을까?" 질문에 대한 진단 전용 스크립트.
# 기존 CurveEncoderGAP(conv 3층 고정)보다 훨씬 깊은 residual 1D CNN(CurveEncoderResNet, stem+ResBlock1D
# n개=conv 1+2n층)으로 concat fusion을 다시 만들어 internal 5-fold CV에서만 HTN/DM/CKD 3개 질환을 평가한다.
# external은 전혀 건드리지 않는다([[feedback_internal_external_validation_discipline]]) — 이건 "이 설정이
# 최종 후보다"가 아니라 "depth를 늘리면 internal 신호가 실제로 느는가"를 확인하는 순수 진단이다. 여기서
# 뚜렷한 개선(예: delta_vs_clinic4가 기존 concat/gated/attnpool/crossattn의 +0.003~+0.015보다 확실히 커짐)이
# 안 보이면, depth 부족이 원인이 아니라는 뜻이므로 이 방향은 접는다
# ([[project_aec_fusion_htn_dm_ckd_comparison]] — 4종 fusion이 이미 이 범위에 수렴했음).

from pathlib import Path

from aec_fusion_common import (
    BACKBONE, ClinicEncoderMLP, CurveEncoderResNet, ENSEMBLE_SIZE, FEATURES, INTERNAL_XLSX,
    AEC_COLS, N_FOLDS, SEED, PROJECT_ROOT, clinical_matrix, evaluate_config, load_cohort,
)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "depth_ablation"
DEPTHS = [1, 2, 3, 4]  # ResBlock1D 개수 -> conv 층수 = 1(stem) + 2*n_blocks = 3/5/7/9
WIDTH = 128            # 기존 backbone 최종 채널(128,256,128)과 같은 규모
KERNEL_SIZE = 7        # 기존 backbone kernel_sizes(11,9,7)의 중간값
HEAD_HIDDEN = 32       # aec_fusion_concat.py stage2 우승 설정 재사용


class DepthConcatCNN(nn.Module):
    def __init__(self, n_clinic: int, width: int, kernel_size: int, n_blocks: int,
                 embed_dim: int, dropout: float, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderResNet(width, kernel_size, n_blocks, embed_dim, dropout)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return DepthConcatCNN(
        n_clinic=n_clinic, width=WIDTH, kernel_size=KERNEL_SIZE, n_blocks=config["n_blocks"],
        embed_dim=config["embed_dim"], dropout=config["dropout"], head_hidden=HEAD_HIDDEN,
    )


def main() -> None:
    meta_int = load_cohort(INTERNAL_XLSX)
    curve_int_raw = meta_int[AEC_COLS].astype(float).to_numpy()
    clinic_int, _ = clinical_matrix(meta_int, None)
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(curve_int_raw))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature in FEATURES:
        y_int = meta_int[feature].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        clinic4_oof = np.full(len(y_int), np.nan)
        for tr_idx, va_idx in folds:
            clf = LogisticRegression(max_iter=2000).fit(clinic_int[tr_idx], y_int[tr_idx])
            clinic4_oof[va_idx] = clf.predict_proba(clinic_int[va_idx])[:, 1]
        clinic4_auc = float(roc_auc_score(y_int, clinic4_oof))
        print(f"[baseline/{feature}] clinic4 internal OOF AUC={clinic4_auc:.4f}")
        rows.append({"feature": feature, "n_blocks": 0, "n_conv_layers": 0, "model": "clinic4",
                     "internal_oof_auc": clinic4_auc, "delta_vs_clinic4": 0.0})

        for n_blocks in DEPTHS:
            cfg = {**BACKBONE, "n_blocks": n_blocks}
            auc, _ = evaluate_config(
                curve_int_raw, clinic_int, y_int, folds, cfg, ensemble_size=ENSEMBLE_SIZE, build_model_fn=build_model,
            )
            n_layers = 1 + 2 * n_blocks
            delta = auc - clinic4_auc
            print(f"[depth/{feature}] n_blocks={n_blocks} (conv층={n_layers}) internal_OOF_AUC={auc:.4f} (delta={delta:+.4f})")
            rows.append({"feature": feature, "n_blocks": n_blocks, "n_conv_layers": n_layers, "model": "depth_resnet",
                         "internal_oof_auc": auc, "delta_vs_clinic4": delta})

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "depth_ablation_internal.csv", index=False)
    print(out.to_string(index=False))
    print(f"Saved to {OUTPUT_DIR / 'depth_ablation_internal.csv'}")


if __name__ == "__main__":
    main()
