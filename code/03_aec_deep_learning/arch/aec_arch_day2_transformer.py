from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# reorg bootstrap: aec_fusion_common은 ../fusion/에 있음(code/03_aec_deep_learning 재편, 2026-09-03)
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "fusion"))

# Day2 — Transformer self-attention curve encoder. 128포인트 곡선에 위치 임베딩을 더해 self-attention
# layer들에 넣고 mean pooling으로 요약. CNN GAP·RNN과 달리 시점간 관계를 전역적으로(거리 제약 없이)
# 학습한다. clinic4 인코더/head/fusion(concat)은 backbone과 동일. docs/aec_architecture_rotation_plan.md 참고.

from aec_fusion_common import BACKBONE, ClinicEncoderMLP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "arch" / "day2_transformer"
FUSION_NAME = "transformer"
N_SLICES = 128
TRANSFORMER_DROPOUT = 0.1  # attention/FFN 내부 dropout(head dropout=config["dropout"]와 별개, 고정)
SEARCH_SPACE = {
    "d_model": [16, 32],
    "nhead": [2, 4],
    "num_layers": [1, 2],
    "head_hidden": [16, 32],
}


class CurveEncoderTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, embed_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, N_SLICES, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=TRANSFORMER_DROPOUT, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.encoder = nn.Sequential(nn.Linear(d_model, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve.unsqueeze(-1)  # (B, 128, 1)
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)  # (B, 128, d_model)
        pooled = x.mean(dim=1)
        return self.encoder(pooled)


class ConcatFusionTransformer(nn.Module):
    def __init__(self, n_clinic: int, d_model: int, nhead: int, num_layers: int,
                 dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderTransformer(d_model, nhead, num_layers, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionTransformer(
        n_clinic=n_clinic, d_model=config["d_model"], nhead=config["nhead"], num_layers=config["num_layers"],
        dropout=config["dropout"], embed_dim=config["embed_dim"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
