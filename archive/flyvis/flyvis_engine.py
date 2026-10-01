"""Run FlyVis pretrained network on rendered video."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flyvis.utils.activity_utils import LayerActivity

from .config import FlyVOConfig
from .device import best_device


@dataclass
class FlyVisOutput:
    """Per-frame FlyVis simulation results."""

    flow: np.ndarray  # (T, 2, n_hexals) optic flow from decoder
    t4: dict[str, np.ndarray]  # cell type -> (T, n_columns)
    t5: dict[str, np.ndarray]
    dt: float
    timestamps: np.ndarray


def _init_datamate_thread_context() -> None:
    """datamate uses threading.local — worker threads need defaults initialized."""
    from datamate.context import context, set_root_dir

    for key, val in (
        ("enforce_config_match", True),
        ("check_size_on_init", False),
        ("verbosity_level", 1),
        ("delete_if_exists", False),
    ):
        if not hasattr(context, key):
            setattr(context, key, val)

    root = os.environ.get("FLYVIS_ROOT_DIR")
    if root and not hasattr(context, "root_dir"):
        set_root_dir(Path(root))


class FlyVisEngine:
    """Wrap pretrained FlyVis network + flow decoder."""

    def __init__(self, config: FlyVOConfig | None = None):
        self.config = config or FlyVOConfig()
        self.device = best_device()
        self._network = None
        self._decoder = None
        self._network_view = None

    def _ensure_loaded(self) -> None:
        if self._network is not None:
            return

        _init_datamate_thread_context()

        import flyvis
        from flyvis import NetworkView

        self._network_view = NetworkView(self.config.network_path)
        self._network = self._network_view.init_network()
        self._network.eval()
        decoders = self._network_view.init_decoder()
        self._decoder = decoders["flow"] if isinstance(decoders, dict) else decoders
        self._decoder.eval()
        try:
            self._network = self._network.to(self.device)
            self._decoder = self._decoder.to(self.device)
        except Exception:
            # FlyVis ops may not support MPS/CUDA on this machine — fall back to CPU
            self.device = "cpu"
            self._network = self._network.to(self.device)
            self._decoder = self._decoder.to(self.device)

    def simulate(self, lum: torch.Tensor, timestamps: np.ndarray) -> FlyVisOutput:
        """
        Args:
            lum: (1, T, 1, n_hexals) photoreceptor input
            timestamps: (T,) seconds
        """
        self._ensure_loaded()
        assert self._network is not None and self._decoder is not None

        dt = 1.0 / self.config.sim_hz
        lum = lum.to(self.device)

        with torch.no_grad():
            # fade_in_state expects (batch, 1, hexals)
            first_frame = lum[:, 0:1].squeeze(2)  # (1, 1, n_hexals)
            stationary = self._network.fade_in_state(
                self.config.fade_in_seconds, dt, first_frame
            )
            responses = self._network.simulate(
                lum, dt, initial_state=stationary, as_layer_activity=True
            )
            flow_pred = self._decoder(responses.activity)

        # responses: LayerActivity; flow_pred: (1, T, 2, n_hexals)
        flow = flow_pred.squeeze(0).cpu().numpy()

        t4, t5 = self._extract_motion_cells(responses, lum.shape[1])

        return FlyVisOutput(
            flow=flow,
            t4=t4,
            t5=t5,
            dt=dt,
            timestamps=timestamps[: flow.shape[0]],
        )

    def simulate_batch(
        self,
        lum: torch.Tensor,
        timestamps: np.ndarray,
        initial_state=None,
    ) -> tuple[FlyVisOutput, object | None]:
        """Simulate a batch; return output and final network state for streaming."""
        self._ensure_loaded()
        assert self._network is not None and self._decoder is not None

        dt = 1.0 / self.config.sim_hz
        lum = lum.to(self.device)

        with torch.no_grad():
            first_frame = lum[:, 0:1].squeeze(2)
            if initial_state is None:
                initial_state = self._network.fade_in_state(
                    self.config.fade_in_seconds, dt, first_frame
                )
            else:
                # Short warm-start at chunk boundary (avoids extra as_states pass)
                initial_state = self._network.fade_in_state(0.15, dt, first_frame)
            responses = self._network.simulate(
                lum, dt, initial_state=initial_state, as_layer_activity=True
            )
            flow_pred = self._decoder(responses.activity)
            final_state = initial_state  # unused; kept for API compatibility

        flow = flow_pred.squeeze(0).cpu().numpy()
        t4, t5 = self._extract_motion_cells(responses, lum.shape[1])

        return (
            FlyVisOutput(
                flow=flow,
                t4=t4,
                t5=t5,
                dt=dt,
                timestamps=timestamps[: flow.shape[0]],
            ),
            final_state,
        )

    def _extract_motion_cells(
        self, responses: LayerActivity, n_frames: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        t4: dict[str, np.ndarray] = {}
        t5: dict[str, np.ndarray] = {}

        central = responses.central
        for cell_type in self.config.t4_types:
            try:
                t4[cell_type] = self._central_column_trace(central[cell_type], n_frames)
            except (ValueError, KeyError):
                pass

        for cell_type in self.config.t5_types:
            try:
                t5[cell_type] = self._central_column_trace(central[cell_type], n_frames)
            except (ValueError, KeyError):
                pass

        return t4, t5

    @staticmethod
    def _central_column_trace(activity, n_frames: int) -> np.ndarray:
        """Average central-column activity -> (T,) trace."""
        if isinstance(activity, torch.Tensor):
            arr = activity.detach().cpu().numpy()
        else:
            arr = np.asarray(activity)
        # Shapes: (batch, frames), (batch, frames, n_columns), or (frames,)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 2:
            # (batch, frames) central column — take batch 0
            arr = arr[0]
        arr = arr[:n_frames]
        return arr.astype(np.float32)
