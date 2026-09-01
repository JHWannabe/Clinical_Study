from __future__ import annotations

# Day4 — SE-CNN(Squeeze-and-Excitation) curve encoder. 기존 CNN GAP 인코더(conv 3층)에 각 층마다
# 채널별 중요도를 학습하는 SE block을 추가 — GAP은 전 구간 동일 가중 평균인 반면 SE는 z축 어느 구간
# (채널)이 중요한지 게이팅으로 자동 조절한다. attnpool(시점별 attention)과 달리 채널 축 재조정이라는
# 점에서 메커니즘이 다르다. docs/aec_architecture_rotation_plan.md 참고.

from aec_fusion_common import BACKBONE, ClinicEncoderMLP, PROJECT_ROOT, run_fusion_pipeline

import torch
import torch.nn as nn

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "arch_day4_secnn"
FUSION_NAME = "secnn"
CHANNELS = BACKBONE["channels"]
KERNEL_SIZES = BACKBONE["kernel_sizes"]
SEARCH_SPACE = {
    "se_reduction": [2, 4, 8],
    "head_hidden": [16, 32],
}


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).squeeze(-1)
        w = self.fc(s).unsqueeze(-1)
        return x * w


class CurveEncoderSECNN(nn.Module):
    def __init__(self, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 se_reduction: int, embed_dim: int):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, c1, k1, padding=k1 // 2), nn.BatchNorm1d(c1), nn.ReLU())
        self.se1 = SEBlock1D(c1, se_reduction)
        self.conv2 = nn.Sequential(nn.Conv1d(c1, c2, k2, padding=k2 // 2), nn.BatchNorm1d(c2), nn.ReLU())
        self.se2 = SEBlock1D(c2, se_reduction)
        self.conv3 = nn.Sequential(nn.Conv1d(c2, c3, k3, padding=k3 // 2), nn.BatchNorm1d(c3), nn.ReLU())
        self.se3 = SEBlock1D(c3, se_reduction)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(nn.Linear(c3, embed_dim), nn.ReLU())

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        x = curve.unsqueeze(1)
        x = self.se1(self.conv1(x))
        x = self.se2(self.conv2(x))
        x = self.se3(self.conv3(x))
        x = self.gap(x).squeeze(-1)
        return self.encoder(x)


class ConcatFusionSECNN(nn.Module):
    def __init__(self, n_clinic: int, channels: tuple[int, int, int], kernel_sizes: tuple[int, int, int],
                 se_reduction: int, dropout: float, embed_dim: int, head_hidden: int):
        super().__init__()
        self.curve_encoder = CurveEncoderSECNN(channels, kernel_sizes, se_reduction, embed_dim)
        self.clinic_encoder = ClinicEncoderMLP(n_clinic, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
        )

    def forward(self, curve: torch.Tensor, clinic: torch.Tensor) -> torch.Tensor:
        c = self.curve_encoder(curve)
        t = self.clinic_encoder(clinic)
        return self.head(torch.cat([c, t], dim=1))


def build_model(config: dict, n_clinic: int) -> nn.Module:
    return ConcatFusionSECNN(
        n_clinic=n_clinic, channels=CHANNELS, kernel_sizes=KERNEL_SIZES, se_reduction=config["se_reduction"],
        dropout=config["dropout"], embed_dim=config["embed_dim"], head_hidden=config["head_hidden"],
    )


def main() -> None:
    run_fusion_pipeline(
        fusion_name=FUSION_NAME, output_dir=OUTPUT_DIR, search_space=SEARCH_SPACE,
        build_model_fn=build_model, n_refine_top_k=3,
    )


if __name__ == "__main__":
    main()
