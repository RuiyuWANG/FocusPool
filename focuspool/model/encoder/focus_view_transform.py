"""Per-view RVT2 focus transforms used inside policy observation encoders."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from focuspool.model.common.normalizer import MultiRobotLinearNormalizer
from focuspool.model.rvt2_heatmap import RVT2Heatmap
from focuspool.focus_toolbox.prediction import VisualFocusPrediction
from focuspool.model.common.augmentation import BackgroundOverlay, CropRandomizer
from focuspool.focus_toolbox.geometry import crop_with_box
from focuspool.util.formatting import pretty_print_nested

VALID_FOCUS_MODES = {
    "pass_through",
    "rvt2_heatmap_crop",
    "random_overlay",
    "disabled",
}
RVT2_HEATMAP_MODES = {"rvt2_heatmap_crop"}
NO_FOCUS_MODES = {"pass_through", "random_overlay"}


class FocusViewTransform(nn.Module):
    """Apply RVT2Heatmap crop/pass-through pipelines for enabled camera views."""

    def __init__(
        self,
        config: Dict[str, Any],  # focus_view_transform.* YAML block
        *,
        verbose: bool = False,
    ):
        super().__init__()

        config = dict(config)
        self.cfg = config
        self.verbose = bool(verbose)

        self.vit_in = int(config.get("vit_in"))
        self.low_res = int(config.get("low_res"))
        self.out_res = int(config.get("out_res"))

        overlay_cfg = dict(config.get("overlay", {}))
        self.overlay_prob = float(overlay_cfg.get("prob", 0.0))
        self.overlay_noise_std = float(overlay_cfg.get("noise_std", 0.0))
        self.overlay_alpha_min = float(overlay_cfg.get("alpha_min", 0.3))
        self.overlay_alpha_max = float(overlay_cfg.get("alpha_max", 0.8))
        self.overlay_warmup_steps = int(overlay_cfg.get("warmup_steps", 0))
        self.overlay_background_path = overlay_cfg.get("background_path")

        crop_cfg = dict(config.get("crop", {}))
        self.center_jitter = float(crop_cfg.get("center_jitter", 0.0))
        self.scale_jitter = float(crop_cfg.get("scale_jitter", 0.0))
        self.box_margin_px = int(crop_cfg.get("box_margin_px", 8))
        self.rvt2_heatmap_cfg = dict(config.get("rvt2_heatmap", {}))

        # View setup
        views_cfg = config["views"]  # dict: view -> {mode: ...}

        self.views = []
        self.view_modes = {}

        for v, c in views_cfg.items():
            if c is None:
                mode = "pass_through"
            elif isinstance(c, str):
                mode = c
            else:
                mode = str(c.get("mode", "pass_through"))
            if mode not in VALID_FOCUS_MODES:
                raise ValueError(
                    f"FocusViewTransform: invalid mode '{mode}' for view '{v}'"
                )
            # Disabled views are dropped from runtime processing.
            if mode == "disabled":
                continue
            self.views.append(v)
            self.view_modes[v] = mode
        if len(self.views) == 0:
            raise ValueError(
                "FocusViewTransform: no enabled views in focus_view_transform.views"
            )
        assert "agentview" in self.views, "FocusViewTransform requires 'agentview'"
        self.enable_eih = "eye_in_hand" in self.views
        self.rvt2_heatmap_modes = RVT2_HEATMAP_MODES
        self.uses_rvt2_heatmap = any(
            self.view_modes[v] in self.rvt2_heatmap_modes for v in self.views
        )
        self.uses_random_overlay = any(
            mode == "random_overlay" for mode in self.view_modes.values()
        )
        self.uses_overlay = self.overlay_prob > 0.0 and self.uses_random_overlay

        self.normalizer = MultiRobotLinearNormalizer()
        self.patch_size = 16
        self.grid_res = self.vit_in // self.patch_size

        self.rvt2_heatmap: Optional[RVT2Heatmap] = None
        if self.uses_rvt2_heatmap:
            for view, mode in self.view_modes.items():
                if mode in self.rvt2_heatmap_modes and view != "agentview":
                    raise ValueError(
                        "RVT2Heatmap backend currently supports agentview only"
                    )
            self.rvt2_heatmap = RVT2Heatmap(
                checkpoint=self.rvt2_heatmap_cfg.get("checkpoint"),
                vit_in=self.vit_in,
            )
            self.patch_size = self.rvt2_heatmap.patch_size
            self.grid_res = self.rvt2_heatmap.grid_res

        # Cache only focus outputs consumed by the configured modes.
        self.cached_views = [
            v for v in self.views if self.view_modes[v] not in NO_FOCUS_MODES
        ]
        self.view_to_box_idx = {v: i for i, v in enumerate(self.cached_views)}

        # Augmentation helpers
        self.background_overlay = None
        if self.uses_overlay:
            if not self.overlay_background_path:
                raise ValueError(
                    "focus_view_transform.overlay.background_path is required "
                    "when overlay probability is positive."
                )
            alpha_mid = (self.overlay_alpha_min + self.overlay_alpha_max) / 2.0
            self.background_overlay = BackgroundOverlay(
                {
                    "prob": self.overlay_prob,
                    "alpha": [alpha_mid, alpha_mid],
                    "background_path": self.overlay_background_path,
                },
                cache_res=self.vit_in,
            )

        self.crop_randomizer = CropRandomizer(
            input_shape=(3, self.low_res, self.low_res),
            crop_size=self.out_res,
        )

        # Boxes are uint8 to match the previous compact cache representation.
        self.buffer_valid: Optional[torch.Tensor] = None  # [N] uint8
        self.box_buffer: Optional[torch.Tensor] = None  # [N, cached_views, 4] uint8

        self.counter = 0

        if verbose:
            config = self.get_config()
            pretty_print_nested(config, title="FocusViewTransform")

    def initialize_buffer(self, buffer_size: int, device: torch.device):
        """Initialize atomic cache used for repeated training observations."""
        self.buffer_valid = torch.zeros(
            (buffer_size,), dtype=torch.uint8, device=device
        )
        self.box_buffer = torch.zeros(
            (buffer_size, len(self.cached_views), 4), dtype=torch.uint8, device=device
        )

        if self.verbose:
            print("[FocusViewTransform] buffer initialized")
            print(f"  valid: {tuple(self.buffer_valid.shape)}")
            print(f"  boxes: {tuple(self.box_buffer.shape)}")

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer) -> None:
        """Set observation normalizer used by encoder preprocessing."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def retrieve_from_buffer(
        self, obs_index: torch.Tensor
    ) -> Optional[Dict[str, VisualFocusPrediction]]:
        """Return cached focus for all enabled views, or None on cache miss."""
        if self.buffer_valid is None or self.box_buffer is None:
            return None

        obs_index = obs_index.to(torch.int64)

        # Atomic validity: any zero flag means miss for the full sample.
        flags = self.buffer_valid[obs_index]  # [N] uint8
        if (flags == 0).any():
            return None

        out: Dict[str, VisualFocusPrediction] = {}
        for view in self.cached_views:
            box_u8 = self.box_buffer[obs_index, self.view_to_box_idx[view]]
            box_px = (box_u8.float() / 255.0) * float(self.vit_in - 1)  # [N,4]
            out[view] = VisualFocusPrediction(
                box_px=box_px,
                mask_grid=None,
                source="rvt2_heatmap",
            )

        return out

    def fill_buffer(
        self,
        *,
        obs_index: torch.Tensor,  # [N] int64
        payloads: Dict[str, VisualFocusPrediction],  # must contain cached views
    ):
        """Write all per-view payloads and then atomically mark entries valid."""
        if self.buffer_valid is None or self.box_buffer is None:
            return

        obs_index = obs_index.to(torch.int64)
        assert set(payloads.keys()) == set(
            self.cached_views
        ), "payloads must include all cached focus views"

        for view, val in payloads.items():
            box_px = val.box_px

            assert box_px.shape[-1] == 4 and box_px.dim() == 2

            box01 = (box_px / float(self.vit_in - 1)).clamp(0.0, 1.0)
            box_u8 = (box01 * 255.0).round().to(torch.uint8)  # [N,4]
            self.box_buffer[obs_index, self.view_to_box_idx[view]] = box_u8

        # Atomic mark valid after all payload writes complete.
        self.buffer_valid[obs_index] = 1

    @torch.no_grad()
    def infer_all_visual_focus(
        self,
        *,
        images_vit_by_view: Dict[str, torch.Tensor],  # view -> [N,3,vit_in,vit_in]
        composer_in: dict,
        obs_index: Optional[torch.Tensor],  # [N]
    ) -> Dict[str, Optional[VisualFocusPrediction]]:
        """Infer visual focus for all views, with optional atomic cache reuse."""

        # Cache hit
        if self.training and (obs_index is not None):
            cached = self.retrieve_from_buffer(obs_index)
            if cached is not None:
                return {
                    v: (None if self.view_modes[v] in NO_FOCUS_MODES else cached[v])
                    for v in self.views
                }

        # Compute focus
        payloads: Dict[str, VisualFocusPrediction] = {}
        out: Dict[str, Optional[VisualFocusPrediction]] = {}
        for view in self.views:
            if self.view_modes[view] in NO_FOCUS_MODES:
                out[view] = None
                continue

            if self.view_modes[view] in self.rvt2_heatmap_modes:
                if self.rvt2_heatmap is None:
                    raise RuntimeError("RVT2Heatmap is not initialized")
                sf = self.rvt2_heatmap.predict_visual_focus(
                    image=images_vit_by_view[view],
                    composer_in=composer_in,
                    view_name=view,
                )
                payloads[view] = sf
                out[view] = sf
                continue

            raise RuntimeError(
                f"Mode {self.view_modes[view]!r} does not have a release backend"
            )

        # Fill cache atomically
        if (
            self.training
            and (obs_index is not None)
            and (self.buffer_valid is not None)
        ):
            self.fill_buffer(obs_index=obs_index, payloads=payloads)

        return out

    def process_view(
        self,
        *,
        view: str,
        image_vit: torch.Tensor,  # [N,3,vit_in,vit_in] for crop/mask ops
        visual_focus: Optional[VisualFocusPrediction],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Apply mode-specific focus processing for a single view."""
        mode = self.view_modes[view]
        default_box_px = torch.tensor(
            [[0.0, 0.0, float(self.vit_in - 1), float(self.vit_in - 1)]],
            device=image_vit.device,
        ).expand(image_vit.shape[0], -1)

        if mode == "pass_through":
            crop = self.lowres_crop(image_vit)
            return {
                "image": crop,
                "box_px": default_box_px,
                "visual_focus": None,
            }

        if mode == "random_overlay":
            image_aug = self.overlay(image_vit)
            crop = self.lowres_crop(image_aug)
            return {
                "image": crop,
                "box_px": default_box_px,
                "visual_focus": None,
            }

        assert (
            visual_focus is not None
        ), "visual_focus required when mode is rvt2_heatmap_crop"
        box_px = visual_focus.box_px

        if mode == "rvt2_heatmap_crop":
            crop, _, box_px = self.box_crop(image_vit, None, box_px)
            return {"image": crop, "box_px": box_px, "visual_focus": visual_focus}

        raise RuntimeError(f"Unsupported focus mode: {mode}")

    def lowres_crop(self, image) -> torch.Tensor:
        """Resize to low-res and apply random crop augmentation."""
        H, W = image.shape[-2:]
        res = self.low_res
        if H != res or W != res:
            image = F.interpolate(image, size=(res, res), mode="bilinear")
        return self.crop_randomizer(image)

    def box_crop(
        self,
        image: torch.Tensor,
        mask_px: Optional[torch.Tensor],
        box_px: torch.Tensor,
    ) -> torch.Tensor:
        """Crop image/mask around `box_px` with optional jitter and margin."""
        center_jitter = self.center_jitter if self.training else 0.0
        scale_jitter = self.scale_jitter if self.training else 0.0
        crop, mask, box_px = crop_with_box(
            image=image,
            box=box_px,
            mask=mask_px,
            output_size=(self.out_res, self.out_res),
            center_jitter=center_jitter,
            scale_jitter=scale_jitter,
            margin=self.box_margin_px,
        )
        return crop, mask, box_px

    def overlay(self, image) -> torch.Tensor:
        """Apply optional texture/background overlay."""
        if not self.training or self.background_overlay is None:
            return image

        mask_prob = self.overlay_prob

        # Linearly warm up overlay probability.
        if self.overlay_warmup_steps > 0:
            mask_prob = mask_prob * min(1.0, self.counter / self.overlay_warmup_steps)

        alpha = (self.overlay_alpha_min + self.overlay_alpha_max) / 2
        return self.background_overlay(
            image,
            alpha=alpha,
            prob=mask_prob,
        )

    def forward(
        self,
        view_imgs: Dict[str, torch.Tensor],
        composer_in: dict,
        obs_index: Optional[torch.Tensor] = None,  # [N]
    ) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """Process all enabled views and return focus-conditioned images/boxes."""

        assert set(view_imgs.keys()) == set(
            self.views
        ), f"Expected views {self.views}, got {list(view_imgs.keys())}"

        # Resize enabled views to the focus backend input resolution once.
        images_vit_by_view = {}
        for view in self.views:
            x = view_imgs[view]
            if x.shape[-2:] != (self.vit_in, self.vit_in):
                x = F.interpolate(
                    x,
                    size=(self.vit_in, self.vit_in),
                    mode="bilinear",
                    align_corners=False,
                )
            images_vit_by_view[view] = x

        # Infer visual focus (with cache when available).
        focus_by_view = self.infer_all_visual_focus(
            images_vit_by_view=images_vit_by_view,
            composer_in=composer_in,
            obs_index=obs_index,
        )

        # Apply per-view mode pipeline.
        ret: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for view in self.views:
            ret[view] = self.process_view(
                view=view,
                image_vit=images_vit_by_view[view],
                visual_focus=focus_by_view[view],
            )

        if self.training:
            self.counter += 1

        return ret

    def get_config(self) -> dict:
        """Return focus transform and model config summary for logging/debugging."""
        view_names = {"agentview": "Agent View", "eye_in_hand": "Eye-in-Hand"}
        mode_names = {"rvt2_heatmap_crop": "RVT2 Heatmap Crop"}
        config = {
            "Focus Transform": {
                view_names.get(view, view.replace("_", " ").title()): mode_names.get(
                    self.view_modes[view],
                    self.view_modes[view].replace("_", " ").title(),
                )
                for view in self.views
            },
        }

        if self.uses_overlay:
            config["Focus Transform"]["Overlay"] = (
                f"{self.overlay_prob:.2f} prob, warmup {self.overlay_warmup_steps}, "
                f"alpha {self.overlay_alpha_min:.2f}-{self.overlay_alpha_max:.2f}"
            )
        else:
            config["Focus Transform"]["Overlay"] = "Disabled"

        crop_modes = {"rvt2_heatmap_crop"}
        if any(mode in crop_modes for mode in self.view_modes.values()):
            crop = (
                f"center {100.0 * self.center_jitter:.1f}%, "
                f"scale {100.0 * self.scale_jitter:.1f}%"
            )
            if self.box_margin_px:
                crop += f", margin {self.box_margin_px}px"
            config["Focus Transform"]["Crop"] = crop
        else:
            config["Focus Transform"]["Crop"] = "Disabled"

        if self.uses_rvt2_heatmap and self.rvt2_heatmap is not None:
            config["RVT2 Heatmap"] = {
                "Status": "Enabled",
                "Checkpoint": self.rvt2_heatmap_cfg.get("checkpoint"),
                "Zoom": f"{self.rvt2_heatmap.zoom:.1f}x",
            }
        else:
            config["Visual Focus"] = "Disabled"
        return config
