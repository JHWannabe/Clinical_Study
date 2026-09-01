from __future__ import annotations

# Day1 — BiLSTM/GRU curve encoder. 128포인트 곡선을 순환 구조(양방향)로 처리해 CNN GAP과 달리
# 시점간 순서 의존성을 명시적으로 모델링한다. clinic4 인코더/head/fusion(concat)은 기존 backbone과
# 동일하게 두고 curve 인코더만 교체([[project_step6_multimodal_fusion_references]] Day1).
# docs/aec_architecture_rotation_plan.md 참고.

from aec_fusion_common import BACKBONE, ClinicEncoderMLP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arch_day1_rnn"
FUSION_NAME = "rnn"
SEARCH_SPACE = {
    "rnn_type": ["gru", "lstm"],
    "hidden_size": [16, 32],
    "num_layers": [1, 2],
    "head_hidden": [16, 32],
}


# 양방향 GRU/LSTM으로 128-step curve를 인코딩, 마지막 layer의 forward+backward hidden state를 concat
class CurveEncoderRNN(nn.Module):
    def __init__(self, rnn_type: str, hidden_size: int, num_layers: int, dropout: float, embed_dim: int):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.encoder = nn.Sequential(nn.Linear(hidden_size * 2, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve.unsqueeze(-1)  # (B, 128, 1)
        out, h = self.rnn(x)
        h = h[0] if self.rnn_type == "lstm" else h  # LSTM은 (h_n, c_n) 튜플
        h = h.view(self.num_layers, 2, -1, self.hidden_size)[-1]  # 마지막 layer만: (2, B, hidden)
        pooled = torch.cat([h[0], h[1]], dim=-1)  # (B, hidden*2)
        return self.encoder(pooled)


class ConcatFusionRNN(nn.Module):
    def __init__(self, n_clinic: int, rnn_type: str, hidden_size: int, num_layers: int,
                 dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderRNN(rnn_type, hidden_size, num_layers, dropout, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionRNN(
        n_clinic=n_clinic, rnn_type=config["rnn_type"], hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], dropout=config["dropout"], embed_dim=config["embed_dim"],
        head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
