from __future__ import annotations

import math

import torch


def get_normalized_grid_coordinates(grid_h: int, grid_w: int) -> torch.Tensor:
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    y_coords = torch.linspace(-1.0, 1.0, grid_h)
    x_coords = torch.linspace(-1.0, 1.0, grid_w)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def get_freq_position_embedding(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    dim = int(dim)
    num_bands = max(1, math.ceil(dim / 4))
    freqs = torch.exp(torch.linspace(0.0, math.log(16.0), num_bands))
    coords = get_normalized_grid_coordinates(grid_h, grid_w).view(
        grid_h,
        grid_w,
        2,
    )
    angles = coords[..., :, None] * freqs
    enc = torch.cat([angles.sin(), angles.cos()], dim=-2).flatten(-2)
    return enc[..., :dim].view(-1, dim)
