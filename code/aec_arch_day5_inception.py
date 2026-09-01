from __future__ import annotations

# Day5 — Multi-scale Inception-1D curve encoder. kernel 3/7/15/31을 병렬 branch로 두고 채널축 concat —
# 기존 concat/depth/TCN ablation은 순차 층 크기·dilation만 바꿨고 병렬 multi-scale은 시도 안 함.
# 서로 다른 시간 스케일(국소 변화 vs 넓은 구간 추세)을 한 층에서 동시에 포착. docs/aec_architecture_rotation_plan.md 참고.

from aec_fusion_common import BACKBONE, ClinicEncoderMLP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arch_day5_inception"
FUSION_NAME = "inception"
KERNEL_SIZES = (3, 7, 15, 31)
SEARCH_SPACE = {
    "branch_channels": [8, 16],
    "n_blocks": [1, 2],
    "head_hidden": [16, 32],
}


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int, kernel_sizes: tuple[int, ...]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, branch_ch, k, padding=k // 2), nn.BatchNorm1d(branch_ch), nn.ReLU())
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.branches], dim=1)


class CurveEncoderInception(nn.Module):
    def __init__(self, branch_ch: int, kernel_sizes: tuple[int, ...], n_blocks: int, embed_dim: int):
        super().__init__()
        out_ch = branch_ch * len(kernel_sizes)
        blocks, in_ch = [], 1
        for _ in range(n_blocks):
            blocks.append(InceptionBlock1D(in_ch, branch_ch, kernel_sizes))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(nn.Linear(out_ch, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve.unsqueeze(1)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.encoder(x)


class ConcatFusionInception(nn.Module):
    def __init__(self, n_clinic: int, branch_channels: int, n_blocks: int,
                 dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderInception(branch_channels, KERNEL_SIZES, n_blocks, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionInception(
        n_clinic=n_clinic, branch_channels=config["branch_channels"], n_blocks=config["n_blocks"],
        dropout=config["dropout"], embed_dim=config["embed_dim"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
