from __future__ import annotations

# Attention pooling — curve 인코더의 GAP(균등 평균)을 슬라이스별 학습된 α(t) 가중합으로 교체
# (DeepSpiro SpiroEncoder 방식). fusion 지점은 concat 그대로 두고 curve 인코더 내부만 바꿔, GAP→attention
# pooling 교체 자체의 효과만 분리해서 본다. attn_hidden(attention MLP 은닉 크기)과 head_hidden을 탐색.
# [[project_step6_multimodal_fusion_references]] 비교표의 "Attention pooling" 행.

from aec_fusion_common import ClinicEncoderMLP, CurveEncoderAttnPool, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fusion_attnpool"
FUSION_NAME = "attnpool"
SEARCH_SPACE = {"attn_hidden": [8, 16, 32], "head_hidden": [16, 32, 64]}


class AttnPoolFusionCNN(nn.Module):
    def __init__(self, n_clinic: int, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 dropout: float, embed_dim: int, attn_hidden: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderAttnPool(channels, kernel_sizes, embed_dim, attn_hidden)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return AttnPoolFusionCNN(
        n_clinic=n_clinic, channels=config["channels"], kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"], embed_dim=config["embed_dim"],
        attn_hidden=config["attn_hidden"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
