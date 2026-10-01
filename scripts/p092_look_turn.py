#!/usr/bin/env python3
"""P09.2 — the frozen look-or-turn rule, as one reusable piece.

P09.1 established that the signal the tracker already reads can tell a glance from a turn, and
that it does so only when the window is taken from the start of the swing rather than from a
fixed span ending at the decision. Measured on five clips, reading it the tracker's way gives
55 percent and reading it this way gives 65, with the improvement present on every one of the
five and VID00005 and VID00009 taking no part in the construction.

This module is that rule, frozen. Nothing here is fitted and nothing here may be changed to suit
a route: the swing-finding is P07.4's, the window is the swing plus a fixed 1.5 s, the statistic
is net over total travel, and the threshold is 0.610.

Two signals, and why the split is kept
---------------------------------------
It is worth being explicit, because using the wrong one would be a different measurement wearing
the same name. P07.4 finds the swings from `yaw_gated` in P07.1's trajectory — that is the
signal it smoothed and crossed, so those are the swing boundaries the labels belong to. The
ratio itself was measured on `yaw_signal_deadband` from P07, the signal the tracker reads. The
two are close but not the same, and P09.1 verified its numbers on exactly this combination, so
the module reproduces the combination rather than tidying it.

When it can be computed at all
------------------------------
The return happens *after* the junction. A decision taken at the node therefore cannot be
filtered at the instant it is taken; the rule becomes available a moment later, once the swing
has ended. That is not a defect to work around — it is the structure P08.6 already has, where a
junction is provisionally taken and revisited on what follows. The rule returns `unknown` when
the window cannot be closed inside the recording, and the caller is expected to fall back rather
than guess.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P07 = ROOT / "output/p07"
P071 = ROOT / "output/p071"

# Frozen from P09.1. The threshold was measured there on the five clips, not chosen here.
LOOK_TURN_THRESHOLD = 0.610
LOOK_MAX = 0.45          # P07.4's own boundary for calling a pair a glance
TURN_MIN = 0.70          # and for calling it a turn
MIN_SWING = 0.15         # swings smaller than this are noise
MERGE_GAP_S = 0.20       # a short run of zeros inside one swing is bridged
SMOOTH_S = 0.30          # P07.4's smoothing, reused unchanged
EVAL_W = 1.5             # seconds after the swing end; the window P09.1 validated


def box(x: np.ndarray, n: int) -> np.ndarray:
    return x if n <= 1 else np.convolve(x, np.ones(n) / n, mode="same")


def load_ratio_signal(video: str) -> tuple[np.ndarray, np.ndarray]:
    """The signal the tracker reads. The ratio is measured on this one."""
    rows = list(csv.DictReader((P07 / f"yaw_signal_{video}.csv").open(encoding="utf-8")))
    t = np.array([float(r["t"]) for r in rows])
    y = np.array([float(r["yaw_signal_deadband"]) for r in rows])
    return t, y


def load_swing_source(video: str) -> tuple[np.ndarray, np.ndarray]:
    """The signal P07.4 crossed to define its swings. The boundaries belong to this one."""
    rows = list(csv.DictReader((P071 / f"trajectory_{video}.csv").open(encoding="utf-8")))
    t = np.array([float(r["t"]) for r in rows])
    y = np.array([float(r["yaw_gated"]) for r in rows])
    return t, y


def find_swings(t: np.ndarray, yaw: np.ndarray, cam: np.ndarray,
                min_swing: float = MIN_SWING,
                merge_gap_s: float = MERGE_GAP_S) -> list[dict]:
    """Monotone runs of the accumulated signal, delimited by zero crossings of the rate.

    This is P07.4's `swings` function, reproduced rather than imported so the rule stands on its
    own and can be applied to a clip P07.4 was never run on. `p092_replay.py` checks it against
    P07.4's own output for all five clips, so a mistake in the copy shows up rather than hiding.

    A run of zeros ends a swing rather than extending it. That matters more than it sounds: the
    deadband leaves the signal at exactly zero for most frames, so absorbing zeros into the
    current run would stretch a swing across every pause and merge separate moves into one.
    """
    out = []
    n = len(yaw)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.02
    gap = max(int(round(merge_gap_s / max(dt, 1e-9))), 0)
    i = 0
    while i < n:
        if yaw[i] == 0:
            i += 1
            continue
        j = i
        s = np.sign(yaw[i])
        while j + 1 < n:
            if yaw[j + 1] != 0 and np.sign(yaw[j + 1]) == s:
                j += 1
                continue
            k = j + 1
            while k < n and yaw[k] == 0:
                k += 1
            if k < n and (k - (j + 1)) <= gap and np.sign(yaw[k]) == s:
                j = k
                continue
            break
        c0, c1 = float(cam[i]), float(cam[j])
        size = abs(c1 - c0)
        if size >= min_swing:
            out.append({"t0": float(t[i]), "t1": float(t[j]), "i0": i, "i1": j,
                        "delta": c1 - c0, "size": size,
                        "direction": "LEFT" if c1 > c0 else "RIGHT"})
        i = j + 1
    return out


def classify_pairs(sw: list[dict], look_max: float = LOOK_MAX,
                   turn_min: float = TURN_MIN) -> list[dict]:
    """P07.4's labelling, reproduced so the swings carry the same boundaries and labels.

    Adjacent swings are always opposite in sign. A pair whose halves cancel is a glance, a pair
    where one half dominates is a turn, and anything in between is left inconclusive. A swing
    with no opposite neighbour is a turn if it is large.
    """
    for s in sw:
        s["kind"] = "INCONCLUSIVE"
        s["paired_with"] = None
    k = 0
    while k < len(sw) - 1:
        a, b = sw[k], sw[k + 1]
        if np.sign(a["delta"]) == np.sign(b["delta"]):
            k += 1
            continue
        ratio = abs(a["delta"] + b["delta"]) / max(a["size"] + b["size"], 1e-9)
        if ratio <= look_max:
            a["kind"] = b["kind"] = "LOOK"
        elif ratio >= turn_min:
            a["kind"] = b["kind"] = "TURN"
        else:
            k += 1
            continue
        a["paired_with"] = b["t0"]
        b["paired_with"] = a["t0"]
        k += 2
    for s in sw:
        if s["kind"] == "INCONCLUSIVE" and s["size"] >= 3 * MIN_SWING:
            s["kind"] = "TURN"
    return sw


class LookTurn:
    """The rule for one clip, answering queries about particular moments."""

    def __init__(self, video: str):
        self.video = video
        st, sy = load_swing_source(video)
        dt = float(np.median(np.diff(st))) if len(st) > 1 else 0.02
        n_sm = max(int(round(SMOOTH_S / max(dt, 1e-9))), 1)
        smoothed = box(sy, n_sm)
        self.swing = classify_pairs(find_swings(st, smoothed, np.cumsum(smoothed) * dt))
        # the ratio is measured on this one, and it is a different recording of the same thing
        self.rt, self.ry = load_ratio_signal(video)
        self.rcam = np.cumsum(self.ry) * float(np.median(np.diff(self.rt)))

    def window_for(self, at: float) -> tuple[float, float, str] | None:
        """The window the rule uses for a moment: swing start to swing end plus the fixed span.

        None when no swing contains the moment, which is the honest answer for a junction
        reached during a pause — there is no swing to judge, so there is nothing to revoke and
        nothing to confirm.
        """
        for s in self.swing:
            if s["t0"] <= at <= s["t1"]:
                return s["t0"], s["t1"] + EVAL_W, s["kind"]
        return None

    def score(self, at: float) -> dict:
        """The look-or-turn reading at one moment, with everything needed to read it back."""
        w = self.window_for(at)
        empty = {"class": "LOOK_TURN_UNKNOWN", "score": None, "pair_start": None,
                 "pair_end": None, "p074_kind": ""}
        if w is None:
            return {**empty, "why": "нет колебания в этот момент"}
        t0, end, kind = w
        m = (self.rt >= t0) & (self.rt <= end)
        if m.sum() < 3:
            return {**empty, "pair_start": t0, "pair_end": end, "p074_kind": kind,
                    "why": "слишком мало отсчётов в окне"}
        seg = self.rcam[m]
        total = float(np.abs(np.diff(seg)).sum())
        if total < 1e-9:
            return {**empty, "pair_start": t0, "pair_end": end, "p074_kind": kind,
                    "why": "полный ход в окне ноль"}
        sc = abs(float(seg[-1] - seg[0])) / total
        return {"class": "LOOK" if sc < LOOK_TURN_THRESHOLD else "TURN",
                "score": float(sc), "pair_start": float(t0), "pair_end": float(end),
                "p074_kind": kind,
                "why": f"{sc:.3f} {'<' if sc < LOOK_TURN_THRESHOLD else '>='} "
                       f"{LOOK_TURN_THRESHOLD}"}
