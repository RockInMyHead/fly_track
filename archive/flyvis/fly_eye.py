"""Project Cartesian video frames onto the fly hexagonal retina."""

from __future__ import annotations

import numpy as np
import torch

from flyvis.datasets.rendering import BoxEye

from .device import best_device


class FlyEyeRenderer:
    """Render grayscale frames to photoreceptor activations."""

    def __init__(self, extent: int = 15, kernel_size: int = 13, device: str | None = None):
        # BoxEye conv weights are CPU-only in FlyVis — always render on CPU
        self.device = "cpu"
        self.receptors = BoxEye(extent=extent, kernel_size=kernel_size)

    @property
    def n_hexals(self) -> int:
        return self.receptors.hexals

    def render(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: (T, H, W) float32 in [0, 1]

        Returns:
            lum: (1, T, 1, n_hexals) tensor on device
        """
        # BoxEye expects (samples, frames, height, width)
        tensor = torch.tensor(frames, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)  # (1, T, H, W)
        rendered = self.receptors(tensor)  # (1, T, 1, n_hexals)
        return rendered
