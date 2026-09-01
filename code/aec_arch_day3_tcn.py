from __future__ import annotations

# Day3 — TCN(dilated causal convolution) curve encoder. dilation을 1,2,4,8...로 지수적으로 늘려
# 적은 층수로도 넓은 receptive field를 확보(Bai et al. 2018 TCN 구조). CNN GAP(고정 kernel, dilation=1)·
# depth ablation(단순 층수 증가)과 달리 receptive field 확장 방식 자체가 다르다.
# docs/aec_architecture_rotation_plan.md 참고.

from aec_fusion_common import BACKBONE, ClinicEncoderMLP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arch_day3_tcn"
FUSION_NAME = "tcn"
HEAD_HIDDEN = 32  # depth ablation과 동일 고정값
SEARCH_SPACE = {
    "n_levels": [3, 4],
    "width": [16, 32],
    "kernel_size": [3, 5],
}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.relu(self.chomp1(self.conv1(x))))
        out = self.drop(self.relu(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CurveEncoderTCN(nn.Module):
    def __init__(self, width: int, kernel_size: int, n_levels: int, dropout: float, embed_dim: int):
        super().__init__()
        blocks = []
        in_ch = 1
        for i in range(n_levels):
            blocks.append(TemporalBlock(in_ch, width, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = width
        self.network = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(nn.Linear(width, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve.unsqueeze(1)  # (B, 1, 128)
        x = self.network(x)
        x = self.gap(x).squeeze(-1)
        return self.encoder(x)


class ConcatFusionTCN(nn.Module):
    def __init__(self, n_clinic: int, width: int, kernel_size: int, n_levels: int,
                 dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderTCN(width, kernel_size, n_levels, dropout, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionTCN(
        n_clinic=n_clinic, width=config["width"], kernel_size=config["kernel_size"], n_levels=config["n_levels"],
        dropout=config["dropout"], embed_dim=config["embed_dim"], head_hidden=HEAD_HIDDEN,
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
