from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf

from focuspool.model.common.augmentation import BackgroundOverlay, CropRandomizer
from focuspool.model.common.normalizer import MultiRobotLinearNormalizer
from focuspool.model.encoder.obs_input import ObsInputProcessor
from focuspool.model.stage_pooled_resnet import StagePooledResNet
from focuspool.util.image_ops import resize_image
from focuspool.util.visualization import visualize


VIZ_KEYPOINT_TOP_P = 0.7
VIZ_KEYPOINT_CELL_RADIUS_FRAC = 0.15


class ViewPrep(nn.Module):
    """Resize then train-random/eval-center crop a normalized image batch."""

    def __init__(self, *, in_res: int, out_res: int):
        super().__init__()
        self.in_res = int(in_res)
        self.out_res = int(out_res)
        self.crop = None
        if self.out_res < self.in_res:
            self.crop = CropRandomizer(
                input_shape=(3, self.in_res, self.in_res),
                crop_size=self.out_res,
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.in_res, self.in_res):
            image = resize_image(image, self.in_res)
        if self.crop is not None:
            return self.crop(image)
        return image

    def inspect(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.in_res, self.in_res):
            image = resize_image(image, self.in_res)
        if self.crop is not None:
            return ttf.center_crop(image, [self.out_res, self.out_res])
        return image


class StagePooledObsEncoder(ObsInputProcessor):
    """Observation encoder using stage-pooled ResNet features for each view."""

    def __init__(
        self,
        *,
        feat_dim: int = 128,
        n_hidden: int = 256,
        prop_feat_dim: int = 64,
        log_freq: int = 1000,
        pretrained: bool = True,
        pooling: Optional[dict | str] = None,
        prop_noise: float = 0.01,
        enable_eih: bool = True,
        view: Optional[dict] = None,
        overlay: Optional[dict] = None,
    ):
        view_cfg = {
            "in_res": 144,
            "out_res": 128,
            **(view or {}),
        }
        view_cfg["in_res"] = int(view_cfg["in_res"])
        view_cfg["out_res"] = int(view_cfg["out_res"])
        if view_cfg["in_res"] <= 0 or view_cfg["out_res"] <= 0:
            raise ValueError("view resolutions must be positive")
        if view_cfg["out_res"] > view_cfg["in_res"]:
            raise ValueError("view.out_res must be <= in_res")

        enable_eih = bool(enable_eih)
        super().__init__(input_res=None, enable_eih=enable_eih)

        self.normalizer = MultiRobotLinearNormalizer()
        self.view_cfg = view_cfg
        self.enable_eih = enable_eih
        self.log_freq = int(log_freq)
        self.prop_noise = float(prop_noise)
        self.count = 0
        self.overlay_step = 0
        self.viz_dir = None
        self._viz_request_split = None
        self._viz_request_step = None
        self.feat_dim = int(feat_dim)
        self.prop_feat_dim = int(prop_feat_dim)
        self.n_hidden = int(n_hidden)
        self.last_video_boxes = []
        self.last_visualization_grid = None
        self.current_epoch = 0

        self.agentview_prep = ViewPrep(
            in_res=view_cfg["in_res"],
            out_res=view_cfg["out_res"],
        )
        self.eih_prep = (
            ViewPrep(in_res=view_cfg["in_res"], out_res=view_cfg["out_res"])
            if self.enable_eih
            else None
        )
        self.overlay = BackgroundOverlay(overlay, cache_res=view_cfg["out_res"])

        view_kwargs = {
            "input_res": view_cfg["out_res"],
            "feat_dim": self.feat_dim,
            "pretrained": bool(pretrained),
            "pooling": pooling,
        }
        self.agentview_encoder = StagePooledResNet(**view_kwargs)
        self.eih_encoder = None
        if self.enable_eih:
            self.eih_encoder = StagePooledResNet(**view_kwargs)

        prop_dim = 11
        view_dim = self.feat_dim + (self.feat_dim if self.enable_eih else 0)
        self.prop_proj = nn.Linear(prop_dim, self.prop_feat_dim)
        self.fuse_proj = nn.Linear(view_dim + self.prop_feat_dim, self.n_hidden)

    def initialize_internal(
        self,
        dataset_size: int,
        device: torch.device,
        viz_dir: Optional[str] = None,
    ):
        _ = dataset_size, device
        self.viz_dir = viz_dir

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def request_visualization(
        self,
        *,
        split: str = "val",
        step: Optional[int] = None,
    ) -> None:
        self._viz_request_split = str(split)
        self._viz_request_step = None if step is None else int(step)
        self.last_visualization_grid = None

    def forward(
        self,
        obs,
        obs_index=None,
        attention_targets=None,
        return_aux: bool = False,
        attention_sigma_grid: float = 2.0,
        attention_view_sample_probs: Optional[dict] = None,
    ):
        _ = obs_index
        self.last_video_boxes = []
        enc_in = self.obs_to_input(obs, self.normalizer, resize=False)
        prop_noise = self.prop_noise if self.training else 0.0
        train_viz = (
            self.training and self.log_freq > 0 and self.count % self.log_freq == 0
        )
        requested_viz = self._viz_request_split is not None
        do_viz = train_viz or requested_viz
        collect_pool_outputs = (
            do_viz or not self.training or attention_targets is not None
        )
        viz_split = self._viz_request_split if requested_viz else "train"
        viz_step = (
            self.count
            if self._viz_request_step is None
            else self._viz_request_step
        )

        overlay_step = self.overlay_step
        agent_image = self.overlay(
            self.agentview_prep(enc_in.agentview),
            step=overlay_step,
        )
        agent_out = self.agentview_encoder(
            agent_image,
            composer_in=enc_in.composer_in,
            prop_noise=prop_noise,
            return_pool_map=collect_pool_outputs,
        )
        if collect_pool_outputs:
            agent_feat, agent_pool_map, agent_keypoints = agent_out
        else:
            agent_feat = agent_out
            agent_pool_map = None
            agent_keypoints = None

        feats = [agent_feat]
        eih_image = None
        eih_pool_map = None
        eih_keypoints = None
        if self.enable_eih:
            if enc_in.eye_in_hand is None:
                raise ValueError("eye-in-hand image is required")
            assert self.eih_prep is not None and self.eih_encoder is not None
            eih_image = self.overlay(
                self.eih_prep(enc_in.eye_in_hand),
                step=overlay_step,
            )
            eih_out = self.eih_encoder(
                eih_image,
                composer_in=enc_in.composer_in,
                prop_noise=prop_noise,
                return_pool_map=collect_pool_outputs,
            )
            if collect_pool_outputs:
                eih_feat, eih_pool_map, eih_keypoints = eih_out
            else:
                eih_feat = eih_out
            feats.append(eih_feat)

        if collect_pool_outputs:
            self._record_video_points(
                T=enc_in.T,
                agent_keypoints=agent_keypoints,
                eih_keypoints=eih_keypoints,
            )

        if do_viz:
            agent_viz_pool_map = _last_timestep(agent_pool_map, enc_in.T)
            views = [
                [
                    _last_timestep(agent_image, enc_in.T),
                    _pool_map_for_viz(agent_viz_pool_map),
                    None,
                    _pool_keypoints_for_debug_viz(
                        agent_viz_pool_map,
                        image_res=int(self.view_cfg["out_res"]),
                    ),
                ]
            ]
            if eih_image is not None:
                eih_viz_pool_map = _last_timestep(eih_pool_map, enc_in.T)
                views.append(
                    [
                        _last_timestep(eih_image, enc_in.T),
                        _pool_map_for_viz(eih_viz_pool_map),
                        None,
                        _pool_keypoints_for_debug_viz(
                            eih_viz_pool_map,
                            image_res=int(self.view_cfg["out_res"]),
                        ),
                    ]
                )
            save_root = (
                self.viz_dir if self.viz_dir is not None else "./visualization_temp"
            )
            grid = visualize(
                views,
                temporal_dim=1,
                num_viz=8,
                padding=4,
                mask_alpha=0.45,
                mask_color=(0.0, 0.75, 1.0),
                save_dir=os.path.join(save_root, viz_split),
                step=viz_step,
            )
            self.last_visualization_grid = grid.detach().cpu()
            self._viz_request_split = None
            self._viz_request_step = None

        feats.append(self.prop_proj(enc_in.proprio))
        if (
            isinstance(attention_targets, dict)
            and "xy_px" in attention_targets
            and "valid" in attention_targets
        ):
            attention_targets = {"agentview": attention_targets}
        attention_targets = dict(attention_targets or {})
        if "eye_in_hand" in attention_targets and not self.enable_eih:
            raise ValueError(
                "RVT2 attention target includes eye_in_hand, but enable_eih=False."
            )

        agent_target = attention_targets.get("agentview")
        eih_target = attention_targets.get("eye_in_hand")
        zero_loss = agent_feat.sum() * 0.0
        agent_loss = (
            _rvt2_attention_loss(
                agent_pool_map,
                agent_target,
                sigma_grid=attention_sigma_grid,
                sample_prob=_attention_sample_prob_for_view(
                    attention_view_sample_probs,
                    "agentview",
                ),
            )
            if agent_target is not None
            else zero_loss
        )

        eih_loss = agent_loss * 0.0
        if eih_target is not None:
            eih_loss = _rvt2_attention_loss(
                eih_pool_map,
                eih_target,
                sigma_grid=attention_sigma_grid,
                sample_prob=_attention_sample_prob_for_view(
                    attention_view_sample_probs,
                    "eye_in_hand",
                ),
            )

        agent_valid = (
            agent_target["valid"]
            .to(device=agent_loss.device, dtype=torch.float32)
            .reshape(-1)
            .mean()
            if agent_target is not None
            else agent_loss * 0.0
        )
        eih_valid = (
            eih_target["valid"]
            .to(device=eih_loss.device, dtype=torch.float32)
            .reshape(-1)
            .mean()
            if eih_target is not None
            else eih_loss * 0.0
        )
        aux = {
            "rvt2_attention_loss": (
                agent_loss + eih_loss
            )
            / max(1, int(agent_target is not None) + int(eih_target is not None)),
            "agentview_attention_loss": agent_loss,
            "eye_in_hand_attention_loss": eih_loss,
            "agentview_valid_frac": agent_valid,
            "eye_in_hand_valid_frac": eih_valid,
        }
        feature = self.fuse_proj(torch.cat(feats, dim=-1))
        self.count += 1
        if self.training:
            self.overlay_step += 1
        return (feature, aux) if return_aux else feature

    def get_config(self) -> dict:
        def overlay_config() -> dict:
            cfg = self.overlay.get_config()
            if cfg["Status"] != "Enabled":
                return "Disabled"
            alpha_lo, alpha_hi = cfg["Alpha"]
            schedule = cfg.get("Schedule")
            if schedule is not None:
                steps = _format_ksteps(schedule["Steps"])
                return (
                    f"Prob {schedule['Start Prob']:.2f}->{schedule['End Prob']:.2f} "
                    f"Over {steps}; Alpha {alpha_lo:.2f}-{alpha_hi:.2f}"
                )
            return (
                f"Prob={cfg['Prob']:.2f}, "
                f"Alpha={alpha_lo:.2f}-{alpha_hi:.2f}"
            )

        return {
            "Encoder": {
                "Eye-in-Hand": "Enabled" if self.enable_eih else "Disabled",
                "Overlay": overlay_config(),
                "Crop": (
                    f"{self.view_cfg['in_res']} -> {self.view_cfg['out_res']}"
                ),
                **self.agentview_encoder.get_config(),
            },
        }

    def _record_video_points(
        self,
        *,
        T: int,
        agent_keypoints: Optional[torch.Tensor],
        eih_keypoints: Optional[torch.Tensor],
    ) -> None:
        """Store latest-step pooled keypoints for rollout video overlays."""
        out = []
        for view, keypoints in (
            ("agentview", agent_keypoints),
            ("eye_in_hand", eih_keypoints),
        ):
            if keypoints is None:
                continue
            B = keypoints.shape[0] // int(T)
            last_points = keypoints.view(B, int(T), *keypoints.shape[1:])[:, -1]
            num_source_points = int(last_points.shape[1])
            last_points = _append_mean_keypoint(last_points)
            item = {
                "source": "stage_pooled",
                "view": view,
                "source_size": int(self.view_cfg["in_res"]),
                "points_px": _pool_keypoints_to_input_px(
                    last_points,
                    in_res=int(self.view_cfg["in_res"]),
                    out_res=int(self.view_cfg["out_res"]),
                ),
            }
            encoder = (
                self.agentview_encoder
                if view == "agentview"
                else self.eih_encoder
            )
            point_radius = _encoder_keypoint_radius_px(
                encoder,
                image_res=int(self.view_cfg["out_res"]),
            )
            if point_radius is not None:
                item["point_radius_px"] = point_radius
            if int(last_points.shape[1]) > num_source_points:
                item["mean_point_index"] = int(last_points.shape[1] - 1)
            out.append(item)
        self.last_video_boxes = out


def _format_ksteps(steps: int) -> str:
    steps = int(steps)
    if steps >= 1000 and steps % 1000 == 0:
        return f"{steps // 1000}k steps"
    return f"{steps} steps"


def _rvt2_attention_loss(
    pool_map: Optional[torch.Tensor],
    targets: Optional[dict],
    *,
    sigma_grid: float,
    sample_prob: float = 1.0,
) -> torch.Tensor:
    if targets is None:
        if pool_map is None:
            return torch.zeros(())
        return pool_map.sum() * 0.0

    xy_px = targets["xy_px"]
    valid = targets["valid"]
    if pool_map is None:
        raise ValueError(
            "RVT2 attention regularization requires a pooling method that returns "
            "an attention map; use focus_refine."
        )
    if torch.is_grad_enabled() and not pool_map.requires_grad:
        raise ValueError(
            "RVT2 attention regularization requires a differentiable attention map; "
            "use focus_refine."
        )

    pool = torch.nan_to_num(pool_map, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    pool = pool.mean(dim=1)
    B, H, W = pool.shape

    xy_px = xy_px.to(device=pool.device, dtype=pool.dtype).reshape(B, 2)
    valid = valid.to(device=pool.device, dtype=torch.bool).reshape(B)
    source_size = targets["source_size"]
    if torch.is_tensor(source_size):
        source_size = float(source_size.reshape(-1)[0].item())
    else:
        source_size = float(source_size)
    if source_size <= 1:
        return pool.sum() * 0.0

    x_px = xy_px[:, 0]
    y_px = xy_px[:, 1]
    valid = (
        valid
        & torch.isfinite(x_px)
        & torch.isfinite(y_px)
        & (x_px >= 0)
        & (y_px >= 0)
        & (x_px < source_size)
        & (y_px < source_size)
    )
    sample_prob = float(sample_prob)
    if not 0.0 <= sample_prob <= 1.0:
        raise ValueError(f"sample_prob must be in [0, 1], got {sample_prob}")
    if sample_prob < 1.0:
        valid = valid & (torch.rand(B, device=pool.device) < sample_prob)
    if not bool(valid.any()):
        return pool.sum() * 0.0

    pool = pool[valid]
    gx = x_px[valid] / (source_size - 1.0) * float(W - 1)
    gy = y_px[valid] / (source_size - 1.0) * float(H - 1)

    grid_y = torch.arange(H, device=pool.device, dtype=pool.dtype)
    grid_x = torch.arange(W, device=pool.device, dtype=pool.dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    dist2 = (xx.unsqueeze(0) - gx[:, None, None]).square() + (
        yy.unsqueeze(0) - gy[:, None, None]
    ).square()
    sigma = max(float(sigma_grid), 1e-6)
    target = torch.exp(-0.5 * dist2 / (sigma * sigma)).flatten(1)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    prob = pool.flatten(1)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(target * prob.clamp_min(1e-12).log()).sum(dim=1).mean()


def _attention_sample_prob_for_view(
    attention_view_sample_probs: Optional[dict],
    view: str,
) -> float:
    if attention_view_sample_probs is None:
        return 1.0
    if view in attention_view_sample_probs:
        return float(attention_view_sample_probs[view])
    return 1.0


def _pool_map_for_viz(pool_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if pool_map is None:
        return None
    pool_map = pool_map.detach()
    pool_map = torch.nan_to_num(pool_map, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if pool_map.shape[1] == 1:
        return pool_map
    return pool_map.mean(dim=1, keepdim=True)


def _last_timestep(x: Optional[torch.Tensor], T: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    T = int(T)
    if T <= 1:
        return x
    if x.shape[0] % T != 0:
        raise ValueError(f"cannot select last timestep from {tuple(x.shape)} with T={T}")
    return x.reshape(x.shape[0] // T, T, *x.shape[1:])[:, -1]


def _pool_keypoints_for_debug_viz(
    pool_map: Optional[torch.Tensor],
    *,
    image_res: int,
) -> Optional[torch.Tensor]:
    if pool_map is None:
        return None
    keypoints = _top_p_pool_keypoints(pool_map, top_p=VIZ_KEYPOINT_TOP_P)
    if keypoints is None:
        return None
    mean_keypoint = _top_p_pool_keypoints(
        _pool_map_for_viz(pool_map),
        top_p=VIZ_KEYPOINT_TOP_P,
    )
    if mean_keypoint is None:
        viz_keypoints = keypoints
    else:
        viz_keypoints = torch.cat([keypoints, mean_keypoint], dim=1)
    num_source_points = int(keypoints.shape[1])
    flags = viz_keypoints.new_zeros(*viz_keypoints.shape[:-1], 1)
    if int(viz_keypoints.shape[1]) > num_source_points:
        flags[..., -1, :] = 1.0
    radius = _pool_map_keypoint_radius_px(
        pool_map,
        image_res=int(image_res),
    )
    radii = viz_keypoints.new_full((*viz_keypoints.shape[:-1], 1), float(radius))
    return torch.cat([viz_keypoints, flags, radii], dim=-1)


def _top_p_pool_keypoints(
    pool_map: Optional[torch.Tensor],
    *,
    top_p: float,
) -> Optional[torch.Tensor]:
    if pool_map is None:
        return None

    pool_map = pool_map.detach()
    pool_map = torch.nan_to_num(pool_map, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    B, K, H, W = pool_map.shape
    prob = pool_map.flatten(2)
    prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    sorted_prob, sorted_idx = prob.sort(dim=-1, descending=True)
    keep = sorted_prob.cumsum(dim=-1) <= float(top_p)
    keep[..., 0] = True
    truncated = torch.zeros_like(prob)
    truncated.scatter_(-1, sorted_idx, sorted_prob * keep.to(sorted_prob.dtype))
    truncated = truncated / truncated.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    coords = _normalized_grid_coordinates(
        H,
        W,
        device=pool_map.device,
        dtype=pool_map.dtype,
    )
    return torch.matmul(truncated, coords).reshape(B, K, 2)


def _append_mean_keypoint(
    keypoints: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if keypoints is None or keypoints.shape[1] <= 1:
        return keypoints
    mean_point = keypoints[..., :2].mean(dim=1, keepdim=True)
    return torch.cat([keypoints[..., :2], mean_point], dim=1)


def _encoder_keypoint_radius_px(
    encoder: Optional[nn.Module],
    *,
    image_res: int,
) -> Optional[int]:
    if encoder is None:
        return None
    grid_h = getattr(encoder, "stage_grid_h", None)
    grid_w = getattr(encoder, "stage_grid_w", None)
    if grid_h is None or grid_w is None:
        return None
    return _grid_keypoint_radius_px(
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        image_res=int(image_res),
    )


def _pool_map_keypoint_radius_px(pool_map: torch.Tensor, *, image_res: int) -> int:
    return _grid_keypoint_radius_px(
        grid_h=int(pool_map.shape[-2]),
        grid_w=int(pool_map.shape[-1]),
        image_res=int(image_res),
    )


def _grid_keypoint_radius_px(*, grid_h: int, grid_w: int, image_res: int) -> int:
    cell = float(image_res) / float(max(1, min(int(grid_h), int(grid_w))))
    return max(1, int(round(VIZ_KEYPOINT_CELL_RADIUS_FRAC * cell)))


def _normalized_grid_coordinates(
    H: int,
    W: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, int(H), device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, int(W), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(int(H) * int(W), 2)


def _pool_keypoints_to_input_px(
    keypoints: torch.Tensor,
    *,
    in_res: int,
    out_res: int,
) -> torch.Tensor:
    points = keypoints[..., :2].detach().float().clone()
    points[..., 0] = (points[..., 0] + 1.0) * 0.5 * float(out_res - 1)
    points[..., 1] = (points[..., 1] + 1.0) * 0.5 * float(out_res - 1)
    crop_offset = max(int(in_res) - int(out_res), 0) / 2.0
    points = points + points.new_tensor([crop_offset, crop_offset])
    return points
