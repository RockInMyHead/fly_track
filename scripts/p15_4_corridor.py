#!/usr/bin/env python3
"""P15.4 — a passage is a corridor with two edges, not a point on the midline.

δ = 15° is fixed before the run. It is not chosen by trying 5°, 10°, 20°, 30°.

    heading θ  →  edges at θ−15° and θ+15°

Both edges slide outward along their own ray, the way a wall moves when the fly
goes forward. Straight ahead is then the pair −15° and +15°, which sits inside
the eyes instead of on the blind seam between them.

The three single corridors are the templates. All 7 nonempty combinations are
shown and ranked against the sum of those templates. A combination counts only
when it leads the nearest wrong set by more than 0.05. Rank 1 with a gap of
0.002 does not pass.

No camera. FINAL V1 is not touched.

    PYTHONPATH=. .venv/bin/python scripts/p15_4_corridor.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_3_motion2d as P  # noqa: E402
import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

OUT = ROOT / "output/p15"
DELTA = M.CORRIDOR_DELTA_DEG
HEADING = {"LEFT": -90.0, "STRAIGHT": 0.0, "RIGHT": 90.0}
NAMES = ("LEFT", "STRAIGHT", "RIGHT")
# Same bar as P15.2. Written here before the run. 0.002 is not a pass.
MARGIN = 0.05
# Weaker wall must still be a quarter of the vertical mass. Fixed before the run.
WALL_SHARE = 0.25


def combos() -> list[tuple[str, ...]]:
    out = []
    for k in range(1, len(NAMES) + 1):
        out.extend(itertools.combinations(NAMES, k))
    return out


def hemifield(vec: np.ndarray, centres: np.ndarray) -> tuple[float, float]:
    return float(vec[centres < 0].sum()), float(vec[centres > 0].sum())


def shape_report(name: str, h: np.ndarray, v: np.ndarray, centres: np.ndarray) -> dict:
    hl, hr = hemifield(h, centres)
    vl, vr = hemifield(v, centres)
    dx, dy = float(h.sum()), float(v.sum())
    vmass = abs(vl) + abs(vr)
    two_walls = (
        vmass > 0.0
        and vl * vr > 0.0
        and abs(vl) >= WALL_SHARE * vmass
        and abs(vr) >= WALL_SHARE * vmass
    )
    if name == "STRAIGHT":
        ok = abs(dy) > abs(dx) and two_walls
    elif name == "LEFT":
        ok = abs(dx) > abs(dy) and abs(hl) > abs(hr)
    else:
        ok = abs(dx) > abs(dy) and abs(hr) > abs(hl)
    return {
        "class": name,
        "dx_brain": round(dx, 4),
        "dy_brain": round(dy, 4),
        "horizontal_left": round(hl, 4),
        "horizontal_right": round(hr, 4),
        "vertical_left": round(vl, 4),
        "vertical_right": round(vr, 4),
        "two_walls_same_sign": two_walls,
        "shape_ok": bool(ok),
        "edges_deg": M.corridor_edges(HEADING[name], DELTA),
    }


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if DELTA != 15.0:
        raise SystemExit(f"δ was fixed at 15°, script has {DELTA}")
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    enc = VideoVisualEncoder(engine)
    groups = {
        "a": P.family_bins(enc.motion_injector, "a"),
        "b": P.family_bins(enc.motion_injector, "b"),
        "c": P.family_bins(enc.vertical_injector, "c"),
        "d": P.family_bins(enc.vertical_injector, "d"),
    }
    centres = np.asarray(enc.motion_injector.bin_azimuth, dtype=float)
    w, h = cfg.encode_width, cfg.encode_height
    print(f"P15.4 — коридор, δ = {DELTA:.0f}° зафиксирован, камеры нет")

    stored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    previews = []
    for name in NAMES:
        frames = M.corridor_clip([HEADING[name]], w, h, delta=DELTA)
        previews.append(frames[len(frames) // 2])
        stored[name] = P.play(engine, enc, frames, groups)
        dx, dy = float(stored[name][0].sum()), float(stored[name][1].sum())
        print(f"  {name:8s}  края {M.corridor_edges(HEADING[name], DELTA)}  "
              f"dx {dx:+8.2f}  dy {dy:+8.2f}")

    templates = {name: P.pack(*stored[name]) for name in NAMES}
    shapes = [shape_report(name, *stored[name], centres) for name in NAMES]
    signs_oppose = bool(
        shapes[0]["dx_brain"] * shapes[2]["dx_brain"] < 0.0
    )
    for row in shapes:
        print(f"  форма {row['class']:8s}  "
              f"H {row['horizontal_left']:+.1f}/{row['horizontal_right']:+.1f}  "
              f"V {row['vertical_left']:+.1f}/{row['vertical_right']:+.1f}  "
              f"{'ok' if row['shape_ok'] else 'мимо'}")

    rows = []
    n_sep = 0
    for combo in combos():
        if len(combo) == 1:
            obs = templates[combo[0]]
        else:
            frames = M.corridor_clip([HEADING[n] for n in combo], w, h, delta=DELTA)
            obs = P.pack(*P.play(engine, enc, frames, groups))
        ranked = P.rank_subsets(obs, templates)
        truth = list(combo)
        rank = next(i for i, r in enumerate(ranked, start=1) if r["classes"] == truth)
        truth_corr = next(r["corr"] for r in ranked if r["classes"] == truth)
        rival = next(r for r in ranked if r["classes"] != truth)
        margin = round(truth_corr - rival["corr"], 4)
        separated = rank == 1 and margin > MARGIN
        n_sep += int(separated)
        rows.append({
            "shown": list(combo),
            "edges_deg": [e for n in combo for e in M.corridor_edges(HEADING[n], DELTA)],
            "rank_of_truth": rank,
            "n_candidates": len(ranked),
            "truth_corr": truth_corr,
            "rival": rival,
            "margin_over_next_different": margin,
            "separated": separated,
            "top3": ranked[:3],
        })
        print(f"  {'+'.join(combo):28s}  место {rank}/7  "
              f"отрыв {margin:+.3f}  {'отделен' if separated else 'слит'}  "
              f"соперник {'+'.join(rival['classes'])}")

    shape_ok = all(r["shape_ok"] for r in shapes) and signs_oppose
    forks_ok = n_sep == len(rows)
    verdict = "CORRIDORS_READABLE" if shape_ok and forks_ok else "CORRIDORS_NOT_SEPARATED"
    doc = {
        "phase": "P15.4",
        "final_v1_touched": False,
        "camera_used": False,
        "no_learned_weights": True,
        "delta_deg": DELTA,
        "delta_fixed_before_run": True,
        "delta_not_searched": True,
        "margin_required": MARGIN,
        "wall_share_required": WALL_SHARE,
        "motion": "каждый край едет наружу вдоль своего луча, как одиночный маркер P15.3",
        "texture": "случайное пятно, без решётки периода 2",
        "shapes": shapes,
        "left_opposes_right": signs_oppose,
        "combinations": rows,
        "separated": n_sep,
        "n_combinations": len(rows),
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = OUT / "P15_4_corridor.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, frame, name in zip(axes, previews, NAMES):
            ax.imshow(frame[:, :, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{name}  {M.corridor_edges(HEADING[name], DELTA)}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / "p15_4_corridors.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.6))
        labels = ["+".join(r["shown"]) for r in rows]
        margins = [r["margin_over_next_different"] for r in rows]
        colors = ["#2a7" if r["separated"] else "#c45" for r in rows]
        ax.barh(labels, margins, color=colors)
        ax.axvline(MARGIN, color="k", lw=0.7, ls="--")
        ax.set_xlabel("отрыв от ближайшего чужого набора")
        ax.set_title(f"P15.4  δ={DELTA:.0f}°  {verdict}")
        fig.tight_layout()
        fig.savefig(OUT / "p15_4_margins.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  рисунок не построен: {exc}")

    print()
    print(f"  форма {'ok' if shape_ok else 'мимо'}, левый против правого: {signs_oppose}")
    print(f"  отделены {n_sep}/{len(rows)} при отрыве > {MARGIN}")
    print(f"  вердикт: {verdict}")
    print(f"  записано: {path}")
    return 0 if verdict == "CORRIDORS_READABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
