"""EPG/PEN-inspired ring attractor heading integration (SEISMIC-compatible design)."""

from __future__ import annotations

import numpy as np


class RingAttractorCompass:
    """
    Integrate angular velocity into a stable heading estimate.

    Biologically inspired by Drosophila EPG/PEN ring attractor (Nature 2017).
    P0 uses a lightweight bump-attractor; swap with SEISMIC weights in P1.
    """

    def __init__(self, n_neurons: int = 8, gain: float = 1.0):
        self.n = n_neurons
        self.gain = gain
        self.preferred_angles = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
        self.bump = np.zeros(n_neurons, dtype=np.float64)
        # Initial heading = π/2 (forward = +Y on trajectory plot)
        start_idx = n_neurons // 4
        self.bump[start_idx] = 1.0
        self._heading = np.pi / 2

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def heading_deg(self) -> float:
        return float(np.degrees(self._heading) % 360)

    def set_heading_deg(self, degrees: float) -> None:
        """Set compass bump to a world heading (0=east, 90=north)."""
        self._heading = float(np.radians(degrees) % (2 * np.pi))
        self.bump.fill(0.0)
        idx = int(round(self._heading / (2 * np.pi) * self.n)) % self.n
        self.bump[idx] = 1.0
        left, right = (idx - 1) % self.n, (idx + 1) % self.n
        self.bump[left] = 0.25
        self.bump[right] = 0.25
        self.bump /= self.bump.sum()

    def nudge_heading_deg(self, delta_deg: float) -> None:
        """Instant heading step — for sharp camera turns (no rate smoothing)."""
        self.set_heading_deg(self.heading_deg + delta_deg)

    def step(self, yaw_rate: float, dt: float) -> float:
        """Integrate yaw rate and update ring bump."""
        delta = self.gain * yaw_rate * dt
        self._heading = (self._heading + delta) % (2 * np.pi)

        # Shift bump on ring (PEN-like velocity input)
        shift = delta / (2 * np.pi) * self.n
        self.bump = self._shift_bump(self.bump, shift)

        # Local recurrence keeps bump stable
        self.bump = self._recurrent_smooth(self.bump)
        self.bump = np.clip(self.bump, 0, None)
        total = self.bump.sum()
        if total > 0:
            self.bump /= total

        return self._heading

    def integrate(self, yaw_rates: np.ndarray, dt: float) -> np.ndarray:
        headings = np.zeros(len(yaw_rates), dtype=np.float32)
        for i, omega in enumerate(yaw_rates):
            headings[i] = self.step(float(omega), dt)
        return headings

    def _shift_bump(self, bump: np.ndarray, shift: float) -> np.ndarray:
        n = len(bump)
        idx = np.arange(n)
        src = (idx - shift) % n
        lo = np.floor(src).astype(int) % n
        hi = (lo + 1) % n
        w = src - np.floor(src)
        return (1 - w) * bump[lo] + w * bump[hi]

    @staticmethod
    def _recurrent_smooth(bump: np.ndarray, w: float = 0.25) -> np.ndarray:
        n = len(bump)
        out = bump.copy()
        for i in range(n):
            left = bump[(i - 1) % n]
            right = bump[(i + 1) % n]
            out[i] = (1 - 2 * w) * bump[i] + w * (left + right)
        return out
