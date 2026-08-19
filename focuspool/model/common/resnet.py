"""Small ResNet backbone helpers shared by focus encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

GROUPNORM_DIVISOR = 16


@dataclass(frozen=True)
class ResNetStageShapes:
    stage_channels: int
    stage_grid_h: int
    stage_grid_w: int


def build_resnet18_backbone(
    *,
    pretrained_imagenet: bool = True,
) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
    backbone = resnet18(weights=weights)
    matches = [
        key.split(".")
        for key, module in backbone.named_modules(remove_duplicate=True)
        if isinstance(module, nn.BatchNorm2d)
    ]
    for *parent, key in matches:
        parent_module = backbone
        if parent:
            parent_module = backbone.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            bn = parent_module[int(key)]
            channels = bn.num_features
            parent_module[int(key)] = nn.GroupNorm(
                max(1, channels // GROUPNORM_DIVISOR),
                channels,
            )
        else:
            bn = getattr(parent_module, key)
            channels = bn.num_features
            setattr(
                parent_module,
                key,
                nn.GroupNorm(max(1, channels // GROUPNORM_DIVISOR), channels),
            )

    assert not any(isinstance(module, nn.BatchNorm2d) for module in backbone.modules())
    return backbone


def build_resnet18_stages(
    *,
    pretrained_imagenet: bool = True,
) -> nn.Module:
    backbone = build_resnet18_backbone(
        pretrained_imagenet=pretrained_imagenet,
    )
    backbone.stem = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
    )
    return backbone


def get_resnet18_stage_modules(backbone: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    return (
        ("stem", backbone.stem),
        ("l1", backbone.layer1),
        ("l2", backbone.layer2),
        ("l3", backbone.layer3),
        ("l4", backbone.layer4),
    )


def probe_resnet18_stage_shapes(
    backbone: nn.Module,
    *,
    input_res: int,
    pooling_stage: str,
    input_channels: int = 3,
) -> ResNetStageShapes:
    param = next(backbone.parameters())
    probe = torch.zeros(
        1,
        int(input_channels),
        int(input_res),
        int(input_res),
        device=param.device,
        dtype=param.dtype,
    )
    with torch.no_grad():
        for name, stage in get_resnet18_stage_modules(backbone):
            probe = stage(probe)
            if name == pooling_stage:
                return ResNetStageShapes(
                    stage_channels=probe.shape[1],
                    stage_grid_h=probe.shape[2],
                    stage_grid_w=probe.shape[3],
                )

    raise ValueError(f"Unsupported pooling_stage: {pooling_stage}")
