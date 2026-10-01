"""Pipeline configuration — MaleCNS v1.0 via flybrain."""

from dataclasses import dataclass, field


@dataclass
class FlyVOConfig:
    """Configuration for MaleCNS visual odometry."""

    # Brain simulation step (s). flybrain default = 20 ms → 50 Hz.
    brain_dt: float = 0.020

    # Downscale long HD videos before visual encoding.
    max_frame_size: int = 480

    # Process long videos in chunks (seconds) to limit memory use.
    chunk_seconds: float = 120.0

    # Visual encoder frame size (width, height) before eye_drive mapping.
    encode_width: int = 96
    encode_height: int = 72

    # Visual drive strength into photoreceptors + inject targets.
    visual_gain: float = 0.9

    # flybrain device: cpu, cuda, or auto.
    device: str = "auto"

    # Path to MaleCNS brain files (weights.npz + brain.npz).
    # Default: $FLY_DATA or ./data/malecns
    data_dir: str = ""

    # Egomotion smoothing (exponential moving average alpha).
    motion_smooth_alpha: float = 0.25

    # Descending-neuron readout tuning.
    steer_window: int = 25  # brain steps (~0.5 s at 20 ms)
    steer_margin: float = 2.0  # spike count diff L/R to turn
    forward_gain: float = 0.04  # DNg100 spikes → forward speed
    turn_gain: float = 0.06  # DNa02 asymmetry → yaw rate
    lateral_gain: float = 0.02  # residual lateral from steer imbalance
    speed_cap: float = 2.0
    yaw_deadzone: float = 0.01
    descending_min_activity: float = 1.5  # mean DN spikes/step in window before DN readout

    # P0 observability: disable ORB/pathway fallback to test pure MaleCNS response.
    use_pathway_fallback: bool = True

    # Pathway fallback (when descending neurons stay quiet — typical for video inject).
    # Calibrated on VID00001_51-70: motion≈0.036, VP≈1135, looming≈72, pan_shift std≈0.01
    pathway_forward_gain: float = 18.0  # motion energy → vx
    pathway_loom_gain: float = 0.012  # looming spikes/step → vx boost
    pathway_yaw_gain: float = 4.0  # pan_shift → yaw_rate (small turns)
    pathway_rot_gain: float = 1.35  # rotation_deg → direct heading nudge (sharp turns)
    rotation_snap_deg: float = 0.25  # |rotation_deg| above this → instant heading step
    corner_snap_deg: float = 90.0  # instant L-turn when ref-burst confirms corner
    corner_ref_deg: float = 10.0  # |ref rotation| threshold to trigger corner snap
    corner_min_time_s: float = 8.0  # ignore early ORB ref glitches
    corner_pan_min: float = 0.008  # L-turns have visible pan; gentle arcs do not
    corner_step_min: float = 0.85  # step ORB must agree with ref (360px ≈0.96)
    arc_min_time_s: float = 18.0  # gentle curves at clip end (130-160)
    arc_gain_scale: float = 0.55  # partial turn, not 90° snap
    pathway_motion_ref: float = 0.036  # typical motion energy for this camera

    # Ring attractor heading integration.
    n_epg_neurons: int = 8
    heading_gain: float = 1.0

    # Path integration.
    velocity_gain: float = 1.0

    # Pathways tracked for visualization / probing.
    pathway_types: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "optic_lobe": ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "Tm", "Mi", "L1", "L2"),
            "t4": ("T4a", "T4b", "T4c", "T4d"),
            "t5": ("T5a", "T5b", "T5c", "T5d"),
            "visual_projection": ("visual_projection", "visual_projection_neuron"),
            "looming": ("LC4", "LPLC2"),
            "steering": ("DNa02",),
            "forward": ("DNg100",),
            "backward": ("MDN",),
            "escape": ("DNp01",),
            "descending": ("descending_neuron",),
        }
    )

    # Output columns for trajectory CSV.
    output_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "timestamp",
            "x",
            "y",
            "heading_deg",
            "vx_body",
            "vy_body",
            "yaw_rate",
            "speed",
            "confidence",
        )
    )

    @property
    def sim_hz(self) -> float:
        return 1.0 / self.brain_dt
