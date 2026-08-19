from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn

from focuspool.model.common.position_encoding import (
    get_freq_position_embedding,
    get_normalized_grid_coordinates,
)
from focuspool.model.common.resnet import (
    build_resnet18_stages,
    get_resnet18_stage_modules,
    probe_resnet18_stage_shapes,
)

QUERY_COND_DIMS = {"eef": 3, "gripper": 1}
QUERY_COND_KEYS = {"eef": "eef_pos", "gripper": "gripper_opening"}
POOLING_INTERNAL_INIT_STD = 0.02


@dataclass
class PoolOutput:
    """Pooling output consumed by StagePooledResNet."""

    pool_map: Optional[torch.Tensor]
    ctx: torch.Tensor
    keypoints: Optional[torch.Tensor] = None


class StagePooledResNet(nn.Module):
    """ResNet-18 encoder that pools features at a selected stage."""

    POOLING_METHODS = {"average_pool", "focus_refine", "spatial_softmax"}
    POOL_STAGES = {"l1", "l2", "l3", "l4"}

    def __init__(
        self,
        *,
        input_res: int,
        feat_dim: int,
        pretrained: bool,
        pooling: Optional[dict | str] = None,
    ):
        super().__init__()
        pooling_cfg = _normalize_pooling_config(pooling)
        pooling_stage = str(pooling_cfg.get("stage", "l2"))
        if pooling_stage not in self.POOL_STAGES:
            raise ValueError(
                f"Unsupported pooling_stage: {pooling_stage!r}. "
                f"Expected one of {sorted(self.POOL_STAGES)}"
            )
        pooling_method = str(pooling_cfg.get("method", "average_pool"))
        if pooling_method not in self.POOLING_METHODS:
            raise ValueError(
                f"Unsupported pooling: {pooling_method!r}. "
                f"Expected one of {sorted(self.POOLING_METHODS)}"
            )
        query_cond = tuple(pooling_cfg.get("query_cond", ("eef", "gripper")))

        self.backbone = build_resnet18_stages(pretrained_imagenet=bool(pretrained))
        self.stages = get_resnet18_stage_modules(self.backbone)
        self.input_res = int(input_res)
        self.feat_dim = int(feat_dim)
        self.pretrained = bool(pretrained)
        self.pooling_stage = pooling_stage
        self.pooling = pooling_method
        self.freeze_image_rep_before_pool = _as_bool(
            pooling_cfg.get("freeze_image_rep_before_pool", False)
        )
        if self.freeze_image_rep_before_pool:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        if "avg_pool_residual" in pooling_cfg:
            raise ValueError(
                "avg_pool_residual has been removed from StagePooledResNet."
            )

        shapes = probe_resnet18_stage_shapes(
            self.backbone,
            input_res=self.input_res,
            pooling_stage=self.pooling_stage,
        )

        self.pool = _make_pooling(
            pooling=self.pooling,
            in_channels=shapes.stage_channels,
            pooling_stage=self.pooling_stage,
            grid_h=shapes.stage_grid_h,
            grid_w=shapes.stage_grid_w,
            query_cond=query_cond,
            kwargs=pooling_cfg.get("kwargs"),
        )
        self.stage_grid_h = shapes.stage_grid_h
        self.stage_grid_w = shapes.stage_grid_w
        self.out_proj = nn.Linear(int(self.pool.ctx_dim), self.feat_dim)
        _init_stage_pooled_module(self.out_proj)

    def forward(
        self,
        image: torch.Tensor,
        *,
        composer_in: dict,
        prop_noise: float = 0.0,
        return_pool_map: bool = False,
    ):
        feat = self._forward_backbone(image)
        pool_out = self.pool(feat, composer_in, prop_noise)
        out = self.out_proj(pool_out.ctx)
        if not return_pool_map:
            return out
        return out, pool_out.pool_map, pool_out.keypoints

    def _forward_backbone(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        pooled_stage_feat = None
        for name, stage in self.stages:
            x = stage(x)
            if name == self.pooling_stage:
                pooled_stage_feat = x
                return pooled_stage_feat
        if pooled_stage_feat is None:
            raise RuntimeError(f"Pooling stage {self.pooling_stage!r} was not produced")
        return pooled_stage_feat

    def get_config(self) -> dict:
        """Return the compact runtime-summary block for the pooling encoder."""
        return {
            "Backbone": {
                "Summary": (
                    f"ResNet-18, {self.pooling_stage.upper()} "
                    f"{self.stage_grid_h}x{self.stage_grid_w}"
                ),
                "Frozen": (
                    "Enabled" if self.freeze_image_rep_before_pool else "Disabled"
                ),
                "Pooling": _format_pooling_runtime_config(
                    self.pooling,
                    self.pool,
                    output_dim=self.feat_dim,
                ),
            },
        }


class AveragePool(nn.Module):
    """Average-pool a selected ResNet stage feature map."""

    def __init__(
        self,
        *,
        in_channels: int,
        pooling_stage: str,
        grid_h: int,
        grid_w: int,
        query_cond: tuple[str, ...] = (),
    ):
        super().__init__()
        self.ctx_dim = int(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        _ = pooling_stage, grid_h, grid_w, query_cond

    def forward(
        self, feat: torch.Tensor, composer_in: dict, prop_noise: float = 0.0
    ) -> PoolOutput:
        _ = composer_in, prop_noise
        return PoolOutput(
            pool_map=None,
            ctx=self.pool(feat).flatten(1),
            keypoints=None,
        )


class QueryComposer(nn.Module):
    """Build a pooling query from a learned token and proprioception."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        query_cond: tuple[str, ...] = ("eef", "gripper"),
    ):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        query_cond = tuple(query_cond)
        try:
            proprio_dim = sum(QUERY_COND_DIMS[name] for name in query_cond)
        except KeyError as exc:
            raise ValueError(f"Unsupported query condition: {exc.args[0]!r}") from exc

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query_cond = query_cond
        hidden_dim = 2 * self.head_dim

        self.query_token = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        nn.init.normal_(self.query_token, std=POOLING_INTERNAL_INIT_STD)
        self.proprio_to_film: Optional[nn.Sequential] = None
        if proprio_dim > 0:
            self.proprio_to_film = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * self.head_dim),
                nn.Tanh(),
            )

    def forward(
        self,
        composer_in: dict,
        batch_size: int,
        noise: float = 0.0,
    ) -> torch.Tensor:
        proprio = _select_query_proprio(composer_in, self.query_cond)
        if proprio is None:
            token = self.query_token.unsqueeze(0).expand(int(batch_size), -1, -1)
            return token.reshape(int(batch_size), self.dim)

        proprio = proprio.unsqueeze(1).expand(-1, self.num_heads, -1)
        if noise > 0.0 and self.training:
            proprio = proprio + torch.randn_like(proprio) * noise

        assert self.proprio_to_film is not None
        token = self.query_token.unsqueeze(0).expand(proprio.shape[0], -1, -1)
        gamma, beta = self.proprio_to_film(proprio).chunk(2, dim=-1)
        query = (1.0 + gamma) * token + beta
        return query.reshape(proprio.shape[0], self.dim)


class FocusRefine(nn.Module):
    """Pool a feature map with iterative query-to-spatial-token attention."""

    def __init__(
        self,
        *,
        in_channels: int,
        pooling_stage: str,
        grid_h: int,
        grid_w: int,
        query_cond: tuple[str, ...],
        iters: int = 3,
        heads: int = 4,
        head_dim: int = 128,
    ):
        super().__init__()
        in_channels = int(in_channels)
        iters = int(iters)
        heads = int(heads)
        head_dim = int(head_dim)
        dim = head_dim * heads
        if iters < 1:
            raise ValueError("focus_refine.iters must be >= 1")
        if heads <= 0:
            raise ValueError("focus_refine.heads must be > 0")
        if head_dim <= 0:
            raise ValueError("focus_refine.head_dim must be > 0")

        self.dim = dim
        self.num_heads = heads
        self.iters = iters
        self.head_dim = head_dim
        self.ctx_dim = self.head_dim
        _ = pooling_stage

        self.query_builder = QueryComposer(
            dim=self.dim,
            num_heads=self.num_heads,
            query_cond=query_cond,
        )
        self.pos_dim = head_dim
        pos_enc = get_freq_position_embedding(grid_h, grid_w, self.pos_dim)
        self.register_buffer("pos_enc", pos_enc, persistent=False)
        coords = get_normalized_grid_coordinates(grid_h, grid_w)
        self.register_buffer("coord_tokens", coords, persistent=False)

        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.ctx_norm = nn.LayerNorm(self.head_dim)

        self.to_k = nn.Linear(in_channels + self.pos_dim, self.dim)
        self.to_v = nn.Linear(in_channels + self.pos_dim, self.dim)

        self.to_query = nn.Linear(self.dim, self.dim)
        self.ctx_to_film = nn.Sequential(
            nn.Linear(self.head_dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, 2 * self.dim),
            nn.Tanh(),
        )
        self.apply(_init_stage_pooled_module)

    def forward(
        self,
        feat: torch.Tensor,
        composer_in: dict,
        prop_noise: float = 0.0,
    ) -> PoolOutput:
        B, _, H, W = feat.shape
        query = self.query_builder(composer_in, B, prop_noise)[:, None]
        Nt = H * W
        Nh = self.num_heads
        d_head = self.head_dim

        feat_tokens = feat.flatten(2).transpose(1, 2)
        pos_tokens = self.pos_enc.to(device=feat.device, dtype=feat.dtype)
        pos_tokens = pos_tokens.unsqueeze(0).expand(B, -1, -1)

        value_tokens = torch.cat([feat_tokens, pos_tokens], dim=-1)
        k = self.key_norm(self.to_k(value_tokens))
        v = self.to_v(value_tokens)

        k = k.view(B, Nt, Nh, d_head).permute(0, 2, 1, 3)
        v = v.view(B, Nt, Nh, d_head).permute(0, 2, 1, 3)
        k_t = k.transpose(-2, -1)

        for step_idx in range(self.iters):
            q = self.to_query(self.query_norm(query))
            q = q.view(B, query.shape[1], Nh, d_head).permute(0, 2, 1, 3)
            scores = torch.matmul(q, k_t) / math.sqrt(d_head)
            attn = torch.softmax(scores, dim=-1)
            ctx_heads = torch.matmul(attn, v)
            pool_ctx = ctx_heads.mean(dim=(1, 2))

            if step_idx < self.iters - 1:
                norm_ctx = self.ctx_norm(pool_ctx).unsqueeze(1)
                gamma, beta = self.ctx_to_film(norm_ctx).chunk(2, dim=-1)
                query = (1.0 + gamma) * query + beta

        pool_map = attn.reshape(B, Nh * query.shape[1], H, W)
        keypoints = torch.matmul(
            pool_map.detach().flatten(2),
            self.coord_tokens.to(device=feat.device, dtype=feat.dtype),
        )
        return PoolOutput(
            pool_map=pool_map,
            ctx=pool_ctx,
            keypoints=keypoints.detach(),
        )


class SpatialSoftmax(nn.Module):
    """Pool a feature map as Robomimic-style expected 2D keypoints."""

    def __init__(
        self,
        *,
        in_channels: int,
        pooling_stage: str,
        grid_h: int,
        grid_w: int,
        query_cond: tuple[str, ...],
        num_points: int = 32,
    ):
        super().__init__()
        in_channels = int(in_channels)
        num_points = int(num_points)
        if num_points < 1:
            raise ValueError("spatial_softmax.num_points must be >= 1")
        self.query_cond = tuple(query_cond)
        self.num_points = num_points
        self.ctx_dim = 2 * self.num_points
        _ = pooling_stage
        try:
            self.proprio_dim = sum(QUERY_COND_DIMS[name] for name in self.query_cond)
        except KeyError as exc:
            raise ValueError(f"Unsupported query condition: {exc.args[0]!r}") from exc
        self.register_buffer(
            "coord_tokens",
            get_normalized_grid_coordinates(grid_h, grid_w),
            persistent=False,
        )
        self.score_proj = nn.Conv2d(
            in_channels + self.proprio_dim,
            self.num_points,
            kernel_size=1,
        )
        self.apply(_init_stage_pooled_module)

    def forward(
        self, feat: torch.Tensor, composer_in: dict, prop_noise: float = 0.0
    ) -> PoolOutput:
        B, _, H, W = feat.shape
        score_in = feat
        proprio = _select_query_proprio(composer_in, self.query_cond)
        if proprio is not None:
            proprio = proprio.to(feat)
            if prop_noise > 0.0 and self.training:
                proprio = proprio + torch.randn_like(proprio) * prop_noise
            proprio_map = proprio[..., None, None]
            proprio_map = proprio_map.expand(B, self.proprio_dim, H, W)
            score_in = torch.cat([feat, proprio_map], dim=1)

        logits = self.score_proj(score_in)
        prob = torch.softmax(logits.flatten(-2), dim=-1).view(B, self.num_points, H, W)
        coords = torch.matmul(
            prob.flatten(2),
            self.coord_tokens.to(device=feat.device, dtype=feat.dtype),
        )
        ctx = coords.flatten(1)
        pool_map = prob.mean(dim=1, keepdim=True).detach()
        return PoolOutput(
            pool_map=pool_map,
            ctx=ctx,
            keypoints=coords.detach(),
        )


def _make_pooling(
    *,
    pooling: str,
    in_channels: int,
    pooling_stage: str,
    grid_h: int,
    grid_w: int,
    query_cond: tuple[str, ...],
    kwargs: Optional[dict],
) -> nn.Module:
    pooling_classes = {
        "average_pool": AveragePool,
        "focus_refine": FocusRefine,
        "spatial_softmax": SpatialSoftmax,
    }
    try:
        pooling_cls = pooling_classes[pooling]
    except KeyError as exc:
        raise ValueError(f"Unsupported pooling: {pooling!r}") from exc

    return pooling_cls(
        in_channels=in_channels,
        pooling_stage=pooling_stage,
        grid_h=grid_h,
        grid_w=grid_w,
        query_cond=query_cond,
        **dict(kwargs or {}),
    )


def _normalize_pooling_config(pooling: Optional[dict | str]) -> dict:
    if pooling is None:
        return {}
    if isinstance(pooling, str):
        return {"method": pooling}
    return dict(pooling)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _format_pooling_runtime_config(
    pooling: str,
    pool: nn.Module,
    *,
    output_dim: int,
) -> str:
    method = pooling
    if isinstance(pool, AveragePool):
        return f"{method}, Out={output_dim}"
    if isinstance(pool, FocusRefine):
        return (
            f"{method}, {pool.iters} Iters, "
            f"{pool.num_heads}x{pool.head_dim} Heads, Out={output_dim}"
        )
    if isinstance(pool, SpatialSoftmax):
        return f"{method}, {pool.num_points} Points, Out={output_dim}"
    return f"{method}, Out={output_dim}"


def _select_query_proprio(
    composer_in: dict,
    query_cond: tuple[str, ...],
) -> Optional[torch.Tensor]:
    proprio = []
    for name in query_cond:
        if name not in QUERY_COND_KEYS:
            raise ValueError(f"Unsupported query condition: {name!r}")
        key = QUERY_COND_KEYS[name]
        try:
            proprio.append(composer_in[key])
        except KeyError as exc:
            raise KeyError(
                f"composer_in is missing {key!r} required by query condition {name!r}"
            ) from exc
    if not proprio:
        return None
    return proprio[0] if len(proprio) == 1 else torch.cat(proprio, dim=-1)


def _init_stage_pooled_module(m: nn.Module) -> None:
    # pass
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight, std=POOLING_INTERNAL_INIT_STD)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
