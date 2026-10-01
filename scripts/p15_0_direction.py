#!/usr/bin/env python3
"""P15.0 — can a synthetic map carry a direction into MaleCNS at all?

No camera. No graph. Four clips, same texture, same seed:

    LEFT      the picture translates left
    RIGHT     the picture translates right
    STRAIGHT  the picture translates up (forward in an egocentric view)
    EMPTY     the picture does not move

A mirror of LEFT must match RIGHT. Outward expansion is not used.

What is read, in order:

    visual encoder  →  signed local motion field, yaw_frozen (logged, not injected)
    injector        →  T4a/T5a versus T4b/T5b  (opponent_index)
    brain           →  spikes of those four types

A yaw turns both eyes the same way, so T4*_L versus T4*_R is not the direction.
The direction is the a-family against the b-family.

    PYTHONPATH=. .venv/bin/python scripts/p15_0_direction.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

OUT = ROOT / "output/p15"
SEED = 64
N_FRAMES = 12
SHIFT_PX = 4


def translating_clip(direction: str, n: int = N_FRAMES, shift: int = SHIFT_PX,
                     w: int = 96, h: int = 72, seed: int = 1) -> list[np.ndarray]:
    """One texture, slid. LEFT and RIGHT are opposite slides of the same pixels."""
    rng = np.random.default_rng(seed)
    pad = n * shift + 8
    noise = rng.integers(0, 256, (h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    # a few hard edges, so a correlator has something to lock onto
    noise[pad // 2 :: 7, :] = 255
    noise[:, pad // 2 :: 11] = 0
    frames = []
    for i in range(n):
        if direction == "LEFT":
            dx, dy = -i * shift, 0
        elif direction == "RIGHT":
            dx, dy = i * shift, 0
        elif direction == "STRAIGHT":
            dx, dy = 0, -i * shift
        else:
            dx, dy = 0, 0
        x0 = pad + dx
        y0 = pad + dy
        crop = noise[y0:y0 + h, x0:x0 + w]
        frames.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    return frames


def mirror(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.flip(fr, 1) for fr in frames]


def play(engine, encoder, frames: list[np.ndarray]) -> dict:
    """Step the encoder and the brain once per frame. The first frame has no flow."""
    encoder.reset()
    engine.reset(SEED)
    groups = {
        "T4a": np.concatenate([encoder.t4_t5_groups["T4a_L"], encoder.t4_t5_groups["T4a_R"]]),
        "T4b": np.concatenate([encoder.t4_t5_groups["T4b_L"], encoder.t4_t5_groups["T4b_R"]]),
        "T5a": np.concatenate([encoder.t4_t5_groups["T5a_L"], encoder.t4_t5_groups["T5a_R"]]),
        "T5b": np.concatenate([encoder.t4_t5_groups["T5b_L"], encoder.t4_t5_groups["T5b_R"]]),
    }
    spikes = {k: 0.0 for k in groups}
    # signed_flow_field is not copied into the metrics dict. The pooled yaw and
    # the opponent index are. The first frame has no previous image, so it is dropped.
    yaws, opps, diffs, flow_l, flow_r = [], [], [], [], []
    n = 0
    for fr in frames:
        _eye, _inj, met = encoder.encode_frame(fr)
        if float(met.get("frame_diff_mean") or 0.0) <= 0.0 and n == 0 and not yaws:
            continue
        n += 1
        yaws.append(float(met.get("yaw_frozen") or 0.0))
        opps.append(float(met.get("local_opponent_index") or 0.0))
        diffs.append(float(met.get("frame_diff_mean") or 0.0))
        flow_l.append(float(met.get("flow_L") or 0.0))
        flow_r.append(float(met.get("flow_R") or 0.0))
        fired = set(np.asarray(
            engine.step(n * engine.config.brain_dt, eye_drive=_eye, inject=_inj).fired
        ).tolist())
        for name, idx in groups.items():
            spikes[name] += sum(1 for i in idx if int(i) in fired)
    if n:
        spikes = {k: v / n for k, v in spikes.items()}
    a = spikes["T4a"] + spikes["T5a"]
    b = spikes["T4b"] + spikes["T5b"]
    return {
        "flow_mean": float(np.mean(yaws)) if yaws else 0.0,
        "flow_L": float(np.mean(flow_l)) if flow_l else 0.0,
        "flow_R": float(np.mean(flow_r)) if flow_r else 0.0,
        "frame_diff": float(np.mean(diffs)) if diffs else 0.0,
        "yaw_frozen": float(np.mean(yaws)) if yaws else 0.0,
        "opponent_index": float(np.mean(opps)) if opps else 0.0,
        "T4a": round(spikes["T4a"], 4),
        "T4b": round(spikes["T4b"], 4),
        "T5a": round(spikes["T5a"], 4),
        "T5b": round(spikes["T5b"], 4),
        "t4t5_opponent": round((a - b) / (a + b), 4) if (a + b) > 0 else 0.0,
        "n_flow_frames": n,
    }


def sign_of(x: float, eps: float = 1e-4) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    maps = OUT / "local_maps"
    maps.mkdir(exist_ok=True)

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    encoder = VideoVisualEncoder(engine)

    clips = {
        "LEFT": translating_clip("LEFT", w=cfg.encode_width, h=cfg.encode_height),
        "RIGHT": translating_clip("RIGHT", w=cfg.encode_width, h=cfg.encode_height),
        "STRAIGHT": translating_clip("STRAIGHT", w=cfg.encode_width, h=cfg.encode_height),
        "EMPTY": translating_clip("EMPTY", w=cfg.encode_width, h=cfg.encode_height),
        "MIRROR_LEFT": mirror(translating_clip("LEFT", w=cfg.encode_width, h=cfg.encode_height)),
    }
    for name in ("LEFT", "RIGHT", "STRAIGHT"):
        cv2.imwrite(str(maps / f"p15_0_{name.lower()}.png"), clips[name][N_FRAMES // 2])

    print("P15.0 — направление в самой карте, без камеры")
    rows = {}
    for name, frames in clips.items():
        rows[name] = play(engine, encoder, frames)
        r = rows[name]
        print(f"  {name:<12} flow {r['flow_mean']:+.4f}  yaw {r['yaw_frozen']:+.4f}  "
              f"opponent {r['opponent_index']:+.3f}  T4/T5 {r['t4t5_opponent']:+.3f}  "
              f"(a {r['T4a']+r['T5a']:.2f} / b {r['T4b']+r['T5b']:.2f})")

    flow_ok = sign_of(rows["LEFT"]["flow_mean"]) == -sign_of(rows["RIGHT"]["flow_mean"]) \
        and sign_of(rows["LEFT"]["flow_mean"]) != 0
    mirror_flow = sign_of(rows["MIRROR_LEFT"]["flow_mean"]) == sign_of(rows["RIGHT"]["flow_mean"]) \
        and sign_of(rows["RIGHT"]["flow_mean"]) != 0
    cells_ok = sign_of(rows["LEFT"]["t4t5_opponent"]) == -sign_of(rows["RIGHT"]["t4t5_opponent"]) \
        and sign_of(rows["LEFT"]["t4t5_opponent"]) != 0
    mirror_cells = sign_of(rows["MIRROR_LEFT"]["t4t5_opponent"]) == sign_of(rows["RIGHT"]["t4t5_opponent"]) \
        and sign_of(rows["RIGHT"]["t4t5_opponent"]) != 0
    inject_ok = sign_of(rows["LEFT"]["opponent_index"]) == -sign_of(rows["RIGHT"]["opponent_index"]) \
        and sign_of(rows["LEFT"]["opponent_index"]) != 0

    if cells_ok and mirror_cells:
        verdict = "DIRECTION_REACHES_T4_T5"
        next_step = "можно возвращать камеру и α 0.1 / 0.25"
    elif flow_ok or inject_ok:
        verdict = "FIELD_HAS_DIRECTION_CELLS_DO_NOT"
        next_step = "полный P15 не запускать: поле видит сторону, T4/T5 — нет"
    else:
        verdict = "NO_DIRECTION"
        next_step = "полный P15 не запускать: картинка не несёт LEFT/RIGHT"

    doc = {
        "phase": "P15.0",
        "camera_used": False,
        "motion": "чистый перенос текстуры, не разъезжание наружу",
        "shift_px": SHIFT_PX,
        "n_frames": N_FRAMES,
        "seed": SEED,
        "why_not_left_minus_right_eye": (
            "поворот камеры двигает оба глаза одинаково; направление — это T4a/T5a против T4b/T5b"
        ),
        "rows": rows,
        "checks": {
            "flow_left_opposite_right": flow_ok,
            "mirror_flow_matches_right": mirror_flow,
            "inject_opponent_flips": inject_ok,
            "t4t5_opponent_flips": cells_ok,
            "mirror_t4t5_matches_right": mirror_cells,
        },
        "verdict": verdict,
        "next_step": next_step,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "P15_0_direction.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  вердикт: {verdict}")
    print(f"  {next_step}")
    print(f"  записано: {OUT / 'P15_0_direction.json'}")
    return 0 if verdict == "DIRECTION_REACHES_T4_T5" else 2


if __name__ == "__main__":
    raise SystemExit(main())
