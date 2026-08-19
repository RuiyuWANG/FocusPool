"""Shared base interface for image-conditioned policies."""

from typing import Dict
import torch
import torch.nn as nn

from focuspool.model.common.normalizer import LinearNormalizer


class BaseImagePolicy(nn.Module):
    """Abstract policy interface used by training and inference policies."""

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run policy inference.

        Args:
            obs_dict: Observation dictionary with shape `[B, To, ...]` tensors.
        Returns:
            Dict containing at least `action` with shape `[B, Ta, Da]`.
        """
        raise NotImplementedError()

    def reset(self):
        """Reset state for stateful policies."""
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set data normalizer used by the policy."""
        raise NotImplementedError()
