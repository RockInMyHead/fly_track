"""Fixed segment windows for P0.1R+ validation (not derived from Reichardt).

Segment windows must be chosen via blind frame review before running the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class VideoRole(str, Enum):
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"
    NEGATIVE_CONTROL = "negative_control"


@dataclass(frozen=True)
class SegmentProfile:
    pre: tuple[float, float]
    turn: tuple[float, float]
    post: tuple[float, float]
    turn_sign: float = -1.0  # L-turn → negative yaw; R-turn → +1.0
    role: VideoRole = VideoRole.HOLDOUT
    expected_turn: str = "LEFT"  # LEFT | RIGHT — for human audit only
    notes: str = ""


PROFILES: dict[str, SegmentProfile] = {
    # Blind frame review + robust SNR on center_band (calibration clip).
    "VID00001_51-70": SegmentProfile(
        pre=(0.5, 8.0),
        turn=(14.0, 16.0),
        post=(17.0, 18.9),
        turn_sign=-1.0,
        role=VideoRole.CALIBRATION,
        expected_turn="LEFT",
        notes="Calibration LEFT; turn confirmed t≈14–16 s via frames.",
    ),
    # Complex-path negative control — not a hold-out gate requirement.
    "VID00001_100-130": SegmentProfile(
        pre=(0.5, 2.0),
        turn=(4.0, 6.0),
        post=(8.0, 12.0),
        turn_sign=-1.0,
        role=VideoRole.NEGATIVE_CONTROL,
        expected_turn="LEFT",
        notes=(
            "COMPLEX_PATH_NEGATIVE_CONTROL: translation + continuous arc + blur. "
            "SNR may be high but sign/post template fails — documents encoder limits."
        ),
    ),
}


# Blind frame review 2025-09-17 — decisions recorded BEFORE encoder runs.
BLIND_REVIEW_REJECTED: dict[str, str] = {
    "VID00001_130-160": (
        "REJECT hold-out: 0–9 s continuous forward follow-cam in corridor; "
        "10–18 s gradual left navigation with person + mixed translation/yaw; "
        "18–28 s factory floor continuous motion. No isolated FWD→YAW→FWD event."
    ),
    "VID00001_160-190": (
        "REJECT hold-out: 0–4 s near-static facing person (not forward segment); "
        "6–12 s horizontal blur (yaw-like) but occluded by person + mixed motion; "
        "14–28 s lateral workspace navigation. No clean three-phase template."
    ),
}


def profile_for_video(path: str | Path) -> SegmentProfile | None:
    key = Path(path).stem
    return PROFILES.get(key)


def profiles_by_role(role: VideoRole) -> dict[str, SegmentProfile]:
    return {k: v for k, v in PROFILES.items() if v.role == role}
