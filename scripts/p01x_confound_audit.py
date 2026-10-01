#!/usr/bin/env python3
"""
P0.1X audit — is the "29/32 direction-selective" result real, or scene identity?

The P0.1X statistic is

    D = mean rate over the camera-RIGHT windows  -  rate on the camera-LEFT window

and it came out with |z| up to 420, with 100% agreement between the two
camera-RIGHT windows and 29 of 32 types flipping under mirroring. Numbers that
large are a warning, not a result. This script checks the obvious confound.

The confound
------------
The three windows are not three views of one scene. They are three different
moments of a walk through a workshop:

    left_65    63.5 - 66.5     \
    right_67   66.0 - 68.5     /  overlapping, same physical turn
    left_108  106.0 - 109.5        a different place, 40 s later

So `D` mostly measures "the scene at t~65 s versus the scene at t~107 s": different
walls, different lighting, different texture. A cell that responds to anything
global in the image will show a large `D` with a perfectly stable sign, and every
cell will point the same way — which is exactly what was observed.

Two checks
----------
1. How alike are the two camera-RIGHT windows to each other, and how unlike the
   camera-LEFT window? If the split follows the scene and not the direction, that
   shows up directly.

2. A statistic that does not compare across scenes at all: within a single window,
   compare the response to the original clip against the response to the mirrored
   clip. Same scene, same moment, only the direction of image motion reversed.
   That is the only mirror contrast in this design that is not also a scene change.

Nothing is tuned. It reads the already-produced rates and the footage.

Usage:
    PYTHONPATH=. python scripts/p01x_confound_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
REPORT = ROOT / "output/p01x_confirmed/report.json"
EVENTS = {"left_65": (63.5, "RIGHT"), "right_67": (66.0, "RIGHT"), "left_108": (106.0, "LEFT")}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
ENCODE = (192, 108)


def scene_stats(start: float) -> dict:
    """Global image statistics of a window: nothing to do with direction."""
    lum, contrast, texture, motion = [], [], [], []
    lk = LKMotionField(grid_w=16, grid_h=8)
    t0 = max(0.0, start - PRE_ROLL_S)
    for _t, frame in iter_video_at_brain_hz(str(VIDEO), t0, start + EVENT_DUR_S, 0.02):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        lum.append(float(g.mean()))
        contrast.append(float(g.std()))
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        texture.append(float(np.mean(np.sqrt(gx * gx + gy * gy))))
        h = g.shape[0]
        dxf, _dy, _d = lk.process(np.ascontiguousarray(g[int(0.3 * h): int(0.7 * h)]))
        motion.append(float(np.abs(dxf).mean()))
    return {
        "mean_luminance": float(np.median(lum)),
        "contrast": float(np.median(contrast)),
        "texture_energy": float(np.median(texture)),
        "local_motion": float(np.median(motion)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="output/p01x_confound")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1X audit — is the result direction, or scene identity?\n")

    # ---------------------------------------------------------------- check 1
    print("=== 1. how different are the three windows as scenes? ===")
    stats = {e: scene_stats(s) for e, (s, _c) in EVENTS.items()}
    print(f"  {'window':10s} {'camera':7s} {'luminance':>10s} {'contrast':>9s} "
          f"{'texture':>9s} {'local motion':>13s}")
    for e, (_s, cam) in EVENTS.items():
        v = stats[e]
        print(f"  {e:10s} {cam:7s} {v['mean_luminance']:10.3f} {v['contrast']:9.3f} "
              f"{v['texture_energy']:9.3f} {v['local_motion']:13.3f}")

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a) + abs(b), 1e-9) * 2

    print(f"\n  relative differences (0 = identical, 1 = maximally different):")
    r65 = stats["left_65"]
    r67 = stats["right_67"]
    l108 = stats["left_108"]
    for key in ("mean_luminance", "contrast", "texture_energy", "local_motion"):
        same = rel(r65[key], r67[key])            # the two camera-RIGHT windows
        cross = (rel(r65[key], l108[key]) + rel(r67[key], l108[key])) / 2
        print(f"    {key:16s} between the two camera-RIGHT windows {same:5.2f}   "
              f"between camera-RIGHT and camera-LEFT {cross:5.2f}")

    # ---------------------------------------------------------------- check 2
    print("\n=== 2. within-event mirror contrast (same scene, direction reversed) ===")
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    per_type = rep["per_type"]
    print(f"  {'type':10s} {'sd':2s} | "
          f"{'left_65 contrast':>17s} {'right_67 contrast':>18s} {'left_108 contrast':>18s} | "
          f"{'consistent':>11s}")
    rows = []
    for r in per_type:
        if r["is_injected"]:
            continue
        line = {}
        for e in EVENTS:
            o = r[f"rate_orig_{e}"]
            m = r[f"rate_mirror_{e}"]
            line[e] = (o - m) / max(abs(o) + abs(m), 1e-9)
        # Direction-sensitive expectation: the two windows where the camera turns the
        # SAME way must agree with each other, and the window where it turns the other
        # way must have the OPPOSITE sign. Requiring all three to match (a first-pass
        # bug) would have demanded the mirror contrast ignore direction altogether.
        same_dir = bool(np.sign(line["left_65"]) == np.sign(line["right_67"]))
        opposite = bool(np.sign(line["left_108"]) == -np.sign(line["left_65"]))
        agree = bool(same_dir and opposite)
        rows.append({"type": r["cell_type"], "side": r["side"],
                     **{k: float(v) for k, v in line.items()},
                     "two_right_agree": same_dir, "left_is_opposite": opposite,
                     "direction_pattern": agree})
        print(f"  {r['cell_type'][:10]:10s} {r['side']:2s} | "
              f"{line['left_65']:+17.3f} {line['right_67']:+18.3f} {line['left_108']:+18.3f} | "
              f"{('yes' if agree else 'no'):>11s}")

    n_agree = sum(1 for r in rows if r["direction_pattern"])
    # contrast magnitude: how strongly does mirroring change the response at all?
    strengths = [np.mean([abs(r["left_65"]), abs(r["right_67"]), abs(r["left_108"])])
                 for r in rows]
    print(f"\n  downstream types: {len(rows)}")
    print(f"  same-direction windows agree AND the opposite-direction window disagrees: "
          f"{n_agree}/{len(rows)}")
    print(f"  mean |mirror contrast| across the three windows: {np.mean(strengths):.3f}")
    print(f"  window alignment: left_65/right_67 sit on the same turn and share the scene,")
    print(f"  so they count as ONE independent location, not two. The pattern therefore")
    print(f"  rests on two distinct locations: t~65 s and t~107 s.")

    # ---------------------------------------------------------------- verdict
    scene_confound = all(
        rel(r65[k], l108[k]) > rel(r65[k], r67[k]) for k in
        ("mean_luminance", "contrast", "texture_energy")
    )
    n_down = len(rows)
    if scene_confound and n_agree >= 0.9 * n_down:
        verdict = (
            "CROSS-WINDOW RESULT IS CONFOUNDED, WITHIN-EVENT RESULT SURVIVES. The two "
            "camera-RIGHT windows are the same physical turn and share the scene, and the "
            "camera-LEFT window is a different place: luminance, contrast and texture all "
            "differ far more across the direction split than within it. So the huge |z| of the "
            "cross-window D and the 100% agreement are the signature of a scene difference, "
            "not evidence of direction. The within-event mirror contrast is not a scene change "
            "of that kind, and it follows the direction pattern for "
            f"{n_agree}/{n_down} downstream types: the two same-direction windows agree with "
            "each other and the opposite-direction window disagrees. That is a real signature, "
            "but it rests on TWO distinct locations (t~65 s and t~107 s), not three, because "
            "left_65 and right_67 overlap. Two locations is a weak footing for a claim about "
            "the connectome."
        )
    elif scene_confound:
        verdict = (
            "CONFOUNDED AND NO CONSISTENT WITHIN-EVENT PATTERN. The direction split coincides "
            "with a scene change, so the cross-window statistic cannot be trusted; and the "
            f"within-event mirror contrast does not follow the direction pattern either "
            f"({n_agree}/{n_down} downstream types). Nothing here establishes direction "
            "selectivity on the confirmed turns."
        )
    else:
        verdict = (
            "NO DOMINANT SCENE CONFOUND: the camera-RIGHT and camera-LEFT windows are about as "
            "similar to each other as the two camera-RIGHT windows are, so the cross-window "
            f"statistic is not obviously explained by scene identity ({n_agree}/{n_down} "
            "downstream types also follow the direction pattern within a single window)."
        )
    print(f"\n=== VERDICT ===\n{verdict}")

    (out / "report.json").write_text(json.dumps(
        {"scene_stats": stats,
         "within_event_mirror": rows,
         "scene_comparison": {
             "two_right_windows": {
                 k: rel(r65[k], r67[k]) for k in
                 ("mean_luminance", "contrast", "texture_energy", "local_motion")
             },
             "right_vs_left": {
                 k: (rel(r65[k], l108[k]) + rel(r67[k], l108[k])) / 2 for k in
                 ("mean_luminance", "contrast", "texture_energy", "local_motion")
             },
         },
         "independent_locations": 2,
         "note": "left_65 and right_67 overlap and share the scene, so they count as one location",
         "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
