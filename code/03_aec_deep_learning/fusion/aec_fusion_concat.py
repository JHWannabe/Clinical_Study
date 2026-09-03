from __future__ import annotations

# Concat fusion (aec_deep_learning.py의 CurveClinicCNN과 동일 메커니즘) — curve_emb·clinic_emb를
# 단순히 이어붙여 head에 넣는 baseline. gated/attnpool/crossattn 세 대안과 동일 backbone
# (aec_fusion_common.BACKBONE)으로 비교하기 위한 기준점. fusion 자체에는 바꿀 하이퍼파라미터가 없으므로
# head_hidden(fusion 이후 MLP head의 은닉 크기)만 작은 그리드로 탐색한다.
# [[project_step6_multimodal_fusion_references]] 비교표의 "Concat(현재)" 행.

from aec_fusion_common import ClinicEncoderMLP, CurveEncoderGAP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "03_aec_deep_learning" / "fusion" / "concat"
FUSION_NAME = "concat"
SEARCH_SPACE = {"head_hidden": [16, 32, 64]}


class ConcatFusionCNN(nn.Module):
    def __init__(self, n_clinic: int, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderGAP(channels, kernel_sizes, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionCNN(
        n_clinic=n_clinic, channels=config["channels"], kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"], embed_dim=config["embed_dim"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
