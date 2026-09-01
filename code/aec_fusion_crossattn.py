from __future__ import annotations

# Cross-attention — curve_emb·clinic_emb를 각각 1-token 시퀀스로 보고 nn.MultiheadAttention으로
# 서로를 query/key/value 삼아 attend한다(curve→clinic, clinic→curve 양방향, CBAM/ViT류 joint fusion의
# 축소판). embed_dim=8(backbone 고정)을 나누어떨어지는 num_heads만 탐색 가능. 파라미터가 가장 많이
# 늘어나는 방식이라 이 코호트(internal n≈1,168) 규모에서 과적합 위험이 가장 크다는 게 사전 판단
# ([[project_step6_multimodal_fusion_references]] 비교표 "Cross-attention" 행).

from aec_fusion_common import ClinicEncoderMLP, CurveEncoderGAP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fusion_crossattn"
FUSION_NAME = "crossattn"
SEARCH_SPACE = {"num_heads": [1, 2, 4], "head_hidden": [16, 32, 64]}  # embed_dim=8의 약수만


class CrossAttnFusionCNN(nn.Module):
    def __init__(self, n_clinic: int, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 dropout: float, embed_dim: int, num_heads: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderGAP(channels, kernel_sizes, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.curve_attends_clinic = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.clinic_attends_curve = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve).unsqueeze(1)     # (B, 1, embed_dim)
        t = self.clinic_encoder(clinic).unsqueeze(1)    # (B, 1, embed_dim)
        c_att, _ = self.curve_attends_clinic(query=c, key=t, value=t)
        t_att, _ = self.clinic_attends_curve(query=t, key=c, value=c)
        fused = torch.cat([c_att.squeeze(1), t_att.squeeze(1)], dim=1)
        return self.head(fused)


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return CrossAttnFusionCNN(
        n_clinic=n_clinic, channels=config["channels"], kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"], embed_dim=config["embed_dim"],
        num_heads=config["num_heads"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
