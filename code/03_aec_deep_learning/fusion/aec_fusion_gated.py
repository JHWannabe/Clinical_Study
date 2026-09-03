from __future__ import annotations

# Gated fusion — concat 대신 학습된 게이트 g=σ(MLP[curve_emb,clinic_emb])로
# g·curve_emb+(1-g)·clinic_emb를 계산해 두 모달 중 어느 쪽을 얼마나 신뢰할지 모델이 입력별로 정하게
# 한다. fusion 블록만 aec_fusion_concat.py와 다르고 curve/clinic 인코더는 동일 backbone.
# gate_hidden(게이트 MLP 은닉 크기)과 head_hidden을 그리드 탐색한다.
# [[project_step6_multimodal_fusion_references]] 비교표의 "Gated fusion" 행.

from aec_fusion_common import ClinicEncoderMLP, CurveEncoderGAP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "gated"
FUSION_NAME = "gated"
SEARCH_SPACE = {"gate_hidden": [8, 16, 32], "head_hidden": [16, 32, 64]}


class GatedFusionCNN(nn.Module):
    def __init__(self, n_clinic: int, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 dropout: float, embed_dim: int, gate_hidden: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderGAP(channels, kernel_sizes, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, embed_dim), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        g = self.gate(torch.cat([c, t], dim=1))
        fused = g * c + (1.0 - g) * t
        return self.head(fused)


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return GatedFusionCNN(
        n_clinic=n_clinic, channels=config["channels"], kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"], embed_dim=config["embed_dim"],
        gate_hidden=config["gate_hidden"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
