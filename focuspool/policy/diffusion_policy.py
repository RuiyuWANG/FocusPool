"""Generic diffusion policy that consumes a policy observation encoder."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from einops import reduce

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from focuspool.model.common.normalizer import MultiRobotLinearNormalizer
from focuspool.model.diffusion.conditional_unet1d import ConditionalUnet1D
from focuspool.model.diffusion.mask_generator import LowdimMaskGenerator
from focuspool.policy.base_image_policy import BaseImagePolicy

from focuspool.dataset.action import validate_action_rep


class DiffusionPolicy(BaseImagePolicy):
    """Action diffusion policy conditioned on encoded observations."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        obs_encoder: Optional[dict] = None,
        action_rep="absolute",
        enc_n_hidden=128,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        attention_prior=None,
        # extra kwargs passed to scheduler.step during sampling
        **kwargs,
    ):
        super().__init__()

        # Parse action/observation shapes.
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action shape, got {action_shape}")
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        if obs_encoder is None:
            raise ValueError("DiffusionPolicy requires an obs_encoder config or module")
        if not isinstance(obs_encoder, nn.Module):
            obs_encoder = hydra.utils.instantiate(obs_encoder)

        # Build diffusion model.
        obs_feature_dim = enc_n_hidden * n_obs_steps
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        # Action-only normalizer for policy targets/predictions.
        # Observation normalization is handled by the policy observation encoder.
        self.local_normalizer = MultiRobotLinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.attention_prior = (
            dict(attention_prior)
            if attention_prior is not None
            else {}
        )
        self.attention_prior_enabled = _strict_bool(
            self.attention_prior.get("enabled", False),
            "attention_prior.enabled",
        )
        self.attention_prior_start_weight = float(
            self.attention_prior.get(
                "start_weight",
                self.attention_prior.get("weight", 0.0),
            )
        )
        self.attention_prior_end_weight = float(
            self.attention_prior.get("end_weight", self.attention_prior_start_weight)
        )
        self.attention_prior_decay_steps = int(
            self.attention_prior.get("decay_steps", 0) or 0
        )
        self.attention_prior_sigma_grid = float(
            self.attention_prior.get("sigma_grid", 2.0)
        )
        self.attention_prior_camera = str(
            self.attention_prior.get("camera", "agentview")
        )
        self.attention_prior_view_sample_probs = self._parse_attention_prior_view_sample_probs()
        self.last_attention_prior_metrics = {}

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        diff_num_params = sum(p.numel() for p in model.parameters()) / 1e6
        enc_num_params = sum(p.numel() for p in self.obs_encoder.parameters()) / 1e6
        action_rep = validate_action_rep(action_rep)
        if "views" in self.attention_prior:
            attention_prior_camera_summary = "Cameras=" + ",".join(
                f"{view}:{dict(view_cfg or {}).get('camera', view)}"
                for view, view_cfg in dict(self.attention_prior["views"]).items()
            )
            attention_prior_sample_summary = "Sample Prob=" + ",".join(
                f"{view}:{prob:g}"
                for view, prob in self.attention_prior_view_sample_probs.items()
            )
        else:
            attention_prior_camera_summary = f"Camera={self.attention_prior_camera}"
            attention_prior_sample_summary = (
                f"Sample Prob={self.attention_prior_view_sample_probs['agentview']:g}"
            )

        self.runtime_config = {
            "Training": {
                "Action Chunk Rep": action_rep,
                "Temporal Window": _format_temporal_window(
                    obs_steps=n_obs_steps,
                    pred_steps=horizon,
                    exec_steps=n_action_steps,
                ),
                "Model Params": f"{diff_num_params:.1f} M",
                "Encoder Params": f"{enc_num_params:.1f} M",
                "Attn Prior": (
                    "Soft Next-Keypoint Target, "
                    f"{attention_prior_camera_summary}, "
                    f"Weight={_format_attention_prior_schedule(self)}, "
                    f"Sigma={self.attention_prior_sigma_grid:g}, "
                    f"{attention_prior_sample_summary}"
                    if self.attention_prior_enabled
                    else "Disabled (Attn Prior=False)"
                ),
            },
            **self.obs_encoder.get_config(),
        }

    def get_runtime_config(self) -> Dict:
        """Return the runtime configuration block shown at startup."""
        return self.runtime_config

    def _parse_attention_prior_view_sample_probs(self) -> Dict[str, float]:
        if "views" in self.attention_prior:
            sample_probs = {}
            for view, view_cfg in dict(self.attention_prior["views"]).items():
                view = str(view)
                prob = float(dict(view_cfg or {}).get("sample_prob", 1.0))
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(
                        f"attention_prior.views.{view}.sample_prob must be in [0, 1]."
                    )
                sample_probs[view] = prob
            if not sample_probs:
                raise ValueError("attention_prior.views must not be empty")
            return sample_probs

        prob = float(self.attention_prior.get("sample_prob", 1.0))
        if not 0.0 <= prob <= 1.0:
            raise ValueError("attention_prior.sample_prob must be in [0, 1].")
        return {"agentview": prob}

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # kwargs forwarded to scheduler.step
        **kwargs,
    ):
        """Run reverse diffusion with hard conditioning."""
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Set diffusion steps.
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # Enforce conditioning before model call.
            trajectory[condition_mask] = condition_data[condition_mask]

            # Predict residual / sample.
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # One reverse step: x_t -> x_{t-1}.
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # Re-apply conditioning at the end.
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def set_normalizer(self, normalizer: MultiRobotLinearNormalizer):
        """Set action normalizer state for this policy."""
        self.local_normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def compute_loss(
        self,
        batch,
        epoch_idx=None,
        global_step=None,
    ):
        """Compute diffusion training loss for one batch."""
        obs = batch["obs"]
        if "robot_id" not in obs:
            raise KeyError("batch['obs'] must include robot_id")
        # robot_id is [B, T], action is [B, H, Da]; use first step for normalization.
        robot_id = obs["robot_id"][:, 0:1]
        nactions = self.local_normalizer.normalize_action(batch["action"], robot_id)
        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory

        local_cond = None
        obs_index = batch.get("obs_index", None)

        # Flatten obs indices for encoder cache interface.
        if obs_index is not None:
            obs_index = obs_index.reshape(-1)

        aux = {}
        if self.attention_prior_enabled:
            attention_targets = batch.get("rvt2_attention_target", None)
            if attention_targets is None:
                raise KeyError(
                    "RVT2 attention regularization is enabled, but the batch is "
                    "missing rvt2_attention_target."
                )
            obs_feature, aux = self.obs_encoder(
                obs=obs,
                obs_index=obs_index,
                attention_targets=attention_targets,
                return_aux=True,
                attention_sigma_grid=self.attention_prior_sigma_grid,
                attention_view_sample_probs=self.attention_prior_view_sample_probs,
            )
        else:
            obs_feature = self.obs_encoder(obs=obs, obs_index=obs_index)
        # Reshape back to [B, Do].
        global_cond = obs_feature.reshape(batch_size, -1)
        # Generate inpainting mask.
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample forward-process noise.
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample random timestep for each sample.
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Forward diffusion: add noise at timestep t.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Compute loss only on non-conditioned entries.
        loss_mask = ~condition_mask

        # Enforce conditioning before prediction.
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict residual.
        model = self.model
        pred = model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        if self.attention_prior_enabled:
            weight = self.attention_prior_weight(global_step)
            loss = loss + weight * aux["rvt2_attention_loss"]
            self.last_attention_prior_metrics = {
                "attention_prior_weight": float(weight),
                **{
                    key: float(value.detach().cpu().item())
                    for key, value in aux.items()
                    if key.endswith("_attention_loss") or key.endswith("_valid_frac")
                },
            }
        else:
            self.last_attention_prior_metrics = {}
        return loss

    def get_last_attention_prior_metrics(self) -> dict:
        return dict(self.last_attention_prior_metrics)

    def attention_prior_weight(self, global_step=None) -> float:
        if self.attention_prior_decay_steps <= 0:
            return self.attention_prior_start_weight
        step = 0 if global_step is None else max(int(global_step), 0)
        progress = min(float(step) / float(self.attention_prior_decay_steps), 1.0)
        return (
            self.attention_prior_start_weight
            + progress
            * (self.attention_prior_end_weight - self.attention_prior_start_weight)
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Sample actions from observations via conditional diffusion."""
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action inference is not implemented")
        if "robot_id" not in obs_dict:
            raise KeyError("obs_dict must include robot_id")
        device = self.device

        # robot_id is [B, T], action is [B, H, Da]; use first step for normalization.
        robot_id = obs_dict["robot_id"][:, 0:1]

        value = next(iter(obs_dict.values()))
        B, _ = value.shape[:2]

        T = self.horizon
        Da = self.action_dim

        dtype = self.dtype

        local_cond = None
        global_cond = None

        obs_feature = self.obs_encoder(obs_dict)

        global_cond = obs_feature.reshape(B, -1)
        # Empty action tensor + mask (no hard conditioning here).
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run reverse diffusion sampling.
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # Unnormalize predictions.
        naction_pred = nsample[..., :Da]
        action_pred = self.local_normalizer.unnormalize_action(naction_pred, robot_id)

        action = action_pred[:, : self.n_action_steps]

        result = {"action": action, "action_pred": action_pred}
        return result


def _format_temporal_window(*, obs_steps: int, pred_steps: int, exec_steps: int) -> str:
    return (
        f"{int(obs_steps)} obs steps, "
        f"{int(pred_steps)} pred steps, "
        f"{int(exec_steps)} exec steps"
    )


def _format_attention_prior_schedule(policy: DiffusionPolicy) -> str:
    start = policy.attention_prior_start_weight
    end = policy.attention_prior_end_weight
    steps = policy.attention_prior_decay_steps
    if steps <= 0 or start == end:
        return f"{start:g}"
    return f"{start:g}->{end:g} over {steps} steps"


def _strict_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value
