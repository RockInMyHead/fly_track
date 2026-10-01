"""P0.1I pooled-signal injection — SUPERSEDED, kept only for reproducibility.

`VideoVisualEncoder` no longer uses this. It is retained so the P0.1I gate and its
algebraic invariants stay reproducible against the code that produced them.

Why it was replaced
-------------------
It computed one scalar (`yaw_frozen`), decided LEFT/RIGHT in Python, and injected
that decision into T4/T5. The fly's visual system was not deriving direction from
the image at all — it was being handed the answer. It also drove all four T4 and
T4 subtypes with the yaw signal, including T4c/T4d which are vertical (pitch)
detectors.

The active path is `fly_vo/local_motion_inject.py`.
"""

from __future__ import annotations

from .optic_flow import FlowDiagnostics, tanh_drive

YAW_WEIGHT_T4 = 0.55
YAW_WEIGHT_T5 = 0.50
SCENE_EXPANSION_WEIGHT = 0.30
SCENE_HEMIFIELD_FLOW_WEIGHT = 0.15

T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")


def compute_inject_strengths(
    flow: FlowDiagnostics,
    left_loom: float,
    right_loom: float,
    gain: float,
    flow_inject_gain: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """P0.1I mapping — returns (strengths_by_group, diagnostics).

    yaw < 0 = LEFT, yaw > 0 = RIGHT; both T4 and T5 lateralise to the same side.
    Scene terms (expansion, hemifield flow) enter strictly common-mode, so they
    can never change the sign of the L-R differential.
    """
    yaw = tanh_drive(flow_inject_gain, flow.yaw_signal)
    left_yaw = max(0.0, -yaw)
    right_yaw = max(0.0, yaw)

    exp = tanh_drive(flow_inject_gain, flow.expansion_signal)
    flow_l = tanh_drive(flow_inject_gain, flow.flow_L)
    flow_r = tanh_drive(flow_inject_gain, flow.flow_R)

    yaw_l_t4 = gain * YAW_WEIGHT_T4 * left_yaw
    yaw_r_t4 = gain * YAW_WEIGHT_T4 * right_yaw
    yaw_l_t5 = gain * YAW_WEIGHT_T5 * left_yaw
    yaw_r_t5 = gain * YAW_WEIGHT_T5 * right_yaw

    scene_common = gain * (
        SCENE_EXPANSION_WEIGHT * abs(exp)
        + SCENE_HEMIFIELD_FLOW_WEIGHT * 0.5 * (abs(flow_l) + abs(flow_r))
    )

    strengths: dict[str, float] = {}
    strengths["LC4_L"] = gain * (0.5 * left_loom + 0.3 * left_yaw + 0.2 * flow.frame_diff_mean)
    strengths["LPLC2_L"] = strengths["LC4_L"] * 0.85
    strengths["LC4_R"] = gain * (0.5 * right_loom + 0.3 * right_yaw + 0.2 * flow.frame_diff_mean)
    strengths["LPLC2_R"] = strengths["LC4_R"] * 0.85

    for side, yaw_side in (("L", yaw_l_t4), ("R", yaw_r_t4)):
        val = yaw_side + scene_common
        for t in T4_TYPES:
            strengths[f"{t}_{side}"] = val

    for side, yaw_side in (("L", yaw_l_t5), ("R", yaw_r_t5)):
        val = yaw_side + scene_common
        for t in T5_TYPES:
            strengths[f"{t}_{side}"] = val

    diagnostics = {
        "yaw_tanh": yaw,
        "left_yaw": left_yaw,
        "right_yaw": right_yaw,
        "scene_common": scene_common,
        "t4_lr": yaw_l_t4 - yaw_r_t4,
        "t5_lr": yaw_l_t5 - yaw_r_t5,
        "combined_lr": (yaw_l_t4 + yaw_l_t5) - (yaw_r_t4 + yaw_r_t5),
    }
    return strengths, diagnostics
