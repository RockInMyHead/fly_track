"""Pick best available torch device for FlyVis network inference."""

from __future__ import annotations

import os

import torch


def best_device() -> str:
    forced = os.environ.get("FLYVO_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    # FlyVis BoxEye rendering is CPU-only; MPS for DMN is optional and often flaky
    if os.environ.get("FLYVO_USE_MPS", "").lower() in ("1", "true", "yes"):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"
