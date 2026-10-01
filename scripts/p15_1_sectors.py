#!/usr/bin/env python3
"""P15.1 — read the map as a direction mask, not as one yaw number.

Each open passage is its own moving marker at its own angle. T4/T5 spikes are
counted in the azimuth bins those cells already have. No weight is fitted:
an edge at angle θ is compared with the bin at azimuth sin(θ).

Map only, first:

    A correct fork
    B mirror
    C a different fork
    D empty

The mirror must flip the azimuth pattern. The other fork must not match it.
Only then is the same vector read with the camera mixed in at α = 0.1 and 0.25.

FINAL V1 is not touched.

    PYTHONPATH=. .venv/bin/python scripts/p15_1_sectors.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_ab_test as AB  # noqa: E402
import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402
from p08_graph import Graph  # noqa: E402

OUT = ROOT / "output/p15"
SEED = 64
SECTORS = (-90, -45, 0, 45, 90)


def azimuth_bins(encoder) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Cell ids per azimuth bin, a-family and b-family, plus the bin centres."""
    inj = encoder.motion_injector
    n = inj.n_az_bins
    a = [np.array([], dtype=np.int64) for _ in range(n)]
    b = [np.array([], dtype=np.int64) for _ in range(n)]
    for _key, (fam, _pref, bins) in inj.groups.items():
        dest = a if fam == "a" else b
        for i, cells in enumerate(bins):
            if len(cells) == 0:
                continue
            dest[i] = cells if len(dest[i]) == 0 else np.concatenate([dest[i], cells])
    return a, b, inj.bin_azimuth.copy()


def bin_of_angle(deg: float, n_bins: int) -> int:
    """Passage angle → horizontal azimuth. −90° is the left edge, +90° the right."""
    u = math.sin(math.radians(deg))
    return int(np.clip(round((u + 1.0) * 0.5 * (n_bins - 1)), 0, n_bins - 1))


def play(engine, encoder, frames, a_bins, b_bins) -> np.ndarray:
    """Opponent (a − b) spike rate in each azimuth bin. The first frame has no flow."""
    encoder.reset()
    engine.reset(SEED)
    n_bins = len(a_bins)
    acc = np.zeros(n_bins, dtype=float)
    used = 0
    for fr in frames:
        eye, inj, met = encoder.encode_frame(fr)
        if float(met.get("frame_diff_mean") or 0.0) <= 0.0 and used == 0:
            continue
        used += 1
        fired = set(np.asarray(
            engine.step(used * engine.config.brain_dt, eye_drive=eye, inject=inj).fired
        ).tolist())
        for i in range(n_bins):
            na = sum(1 for c in a_bins[i] if int(c) in fired)
            nb = sum(1 for c in b_bins[i] if int(c) in fired)
            acc[i] += na - nb
    if used:
        acc /= used
    return acc


def play_mix(engine, cam_enc, map_enc, cam_frames, map_frames, alpha, a_bins, b_bins) -> np.ndarray:
    engine.reset(SEED)
    cam_enc.reset()
    map_enc.reset()
    n_bins = len(a_bins)
    acc = np.zeros(n_bins, dtype=float)
    n = max(len(map_frames), 2)
    for i in range(1, n):
        eye_c, inj_c, _ = cam_enc.encode_frame(cam_frames[i % len(cam_frames)])
        eye_m, inj_m, _ = map_enc.encode_frame(map_frames[i])
        eye, inj = M.mix_drives((eye_c, inj_c), (eye_m, inj_m), alpha)
        fired = set(np.asarray(
            engine.step(i * engine.config.brain_dt, eye_drive=eye, inject=inj).fired
        ).tolist())
        for b in range(n_bins):
            na = sum(1 for c in a_bins[b] if int(c) in fired)
            nb = sum(1 for c in b_bins[b] if int(c) in fired)
            acc[b] += na - nb
    return acc / max(n - 1, 1)


def coarse(vec: np.ndarray, centres: np.ndarray) -> dict[str, float]:
    """Five fixed sectors. A bin belongs to the nearest of −90…+90 degrees."""
    # centres are azimuth −1..+1; angle whose sine is that azimuth
    ang = np.degrees(np.arcsin(np.clip(centres, -1, 1)))
    out = {}
    for s in SECTORS:
        w = np.exp(-0.5 * ((ang - s) / 22.0) ** 2)
        out[f"{s:+d}"] = round(float(np.sum(w * vec) / max(w.sum(), 1e-9)), 4)
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def support(vec: np.ndarray, exits: list[dict]) -> list[dict]:
    """How much the bin at sin(θ) agrees with that passage. No fitted weight."""
    n = len(vec)
    rows = []
    for e in exits:
        i = bin_of_angle(e["deg"], n)
        horizontal = math.sin(math.radians(e["deg"]))
        rows.append({
            "edge": e["edge"], "deg": round(e["deg"], 1), "bin": i,
            "activity": round(float(vec[i]), 4),
            "horizontal": round(horizontal, 3),
            "sign_agrees": int(np.sign(vec[i]) == np.sign(horizontal)) if abs(horizontal) > 0.25 else "",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--video", default="VID00010")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    maps = OUT / "local_maps"
    maps.mkdir(exist_ok=True)

    g = Graph.load(ROOT / "data/p08/graph.json")
    events = AB.junctions(g, a.limit)
    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    enc = VideoVisualEncoder(engine)
    cam_enc = VideoVisualEncoder(engine)
    map_enc = VideoVisualEncoder(engine)
    a_bins, b_bins, centres = azimuth_bins(enc)
    w, h = cfg.encode_width, cfg.encode_height

    video = None
    for cand in (ROOT / "data/p01r" / f"{a.video}.AVI",
                 Path("/Volumes/NO NAME/DCIM") / f"{a.video}.AVI"):
        if cand.exists():
            video = cand
            break

    print("P15.1 — вектор по азимуту, не один yaw")
    print(f"  развилок {len(events)}, бинов {len(centres)}")

    map_rows = []
    vectors = {}
    for i, ev in enumerate(events):
        wrong = events[(i + 1) % len(events)]["exits"]
        kinds = {
            "A_correct": M.sector_clip(ev["exits"], w, h, kind="correct", seed=i + 1),
            "B_mirror": M.sector_clip(ev["exits"], w, h, kind="mirror", seed=i + 1),
            "C_wrong": M.sector_clip(wrong, w, h, kind="correct", seed=i + 7),
            "D_empty": M.sector_clip([], w, h, kind="empty"),
        }
        if i == 0:
            cv2.imwrite(str(maps / "p15_1_correct.png"), kinds["A_correct"][4])
            cv2.imwrite(str(maps / "p15_1_mirror.png"), kinds["B_mirror"][4])
        got = {k: play(engine, enc, fr, a_bins, b_bins) for k, fr in kinds.items()}
        vectors[ev["node"]] = {k: v.tolist() for k, v in got.items()}
        flip_b = got["B_mirror"][::-1]
        abs_a = np.abs(got["A_correct"])
        row = {
            "node": ev["node"],
            "exits": [round(e["deg"], 1) for e in ev["exits"]],
            "sectors_correct": coarse(got["A_correct"], centres),
            "sectors_mirror": coarse(got["B_mirror"], centres),
            "corr_mirror": round(corr(got["A_correct"], got["B_mirror"]), 3),
            "corr_mirror_flipped": round(corr(got["A_correct"], flip_b), 3),
            "corr_abs_mirror_flipped": round(corr(abs_a, np.abs(flip_b)), 3),
            "corr_abs_wrong": round(corr(abs_a, np.abs(got["C_wrong"])), 3),
            "corr_wrong": round(corr(got["A_correct"], got["C_wrong"]), 3),
            "energy_correct": round(float(np.mean(np.abs(got["A_correct"]))), 4),
            "energy_empty": round(float(np.mean(np.abs(got["D_empty"]))), 4),
            "support": support(got["A_correct"], ev["exits"]),
        }
        map_rows.append(row)
        print(f"  {ev['node']:<6} |зерк.перевёрнутое| {row['corr_abs_mirror_flipped']:+.2f}  "
              f"|чужая| {row['corr_abs_wrong']:+.2f}  "
              f"энергия {row['energy_correct']:.2f} / пусто {row['energy_empty']:.2f}")

    def mean_key(k):
        return float(np.mean([r[k] for r in map_rows]))

    mirror_flips = mean_key("corr_abs_mirror_flipped") > mean_key("corr_mirror") + 0.15
    wrong_differs = mean_key("corr_abs_wrong") < mean_key("corr_abs_mirror_flipped") - 0.15
    empty_quiet = mean_key("energy_empty") < 0.35 * max(mean_key("energy_correct"), 1e-6)
    spatial_ok = bool(mirror_flips and wrong_differs and empty_quiet)

    # camera mixed in only as the second reading, same vector, not a pooled yaw
    cam_rows = []
    if video is not None:
        print("  камера + карта, тот же вектор")
        for i, ev in enumerate(events):
            cam = AB.grab_pair(video, 40.0 + 12.0 * i)
            if not cam:
                continue
            base_map = M.sector_clip(ev["exits"], w, h, kind="correct", seed=i + 1)
            for alpha in (0.1, 0.25):
                vec = play_mix(engine, cam_enc, map_enc, cam, base_map, alpha, a_bins, b_bins)
                cam_only = play_mix(engine, cam_enc, map_enc, cam,
                                    M.sector_clip([], w, h, kind="empty"), 0.0, a_bins, b_bins)
                cam_rows.append({
                    "node": ev["node"], "alpha": alpha,
                    "sectors": coarse(vec, centres),
                    "corr_with_map_only": round(corr(vec, np.asarray(vectors[ev["node"]]["A_correct"])), 3),
                    "delta_energy": round(float(np.mean(np.abs(vec - cam_only))), 4),
                    "support": support(vec, ev["exits"]),
                })

    verdict = "SPATIAL_MASK" if spatial_ok else "STILL_NOT_A_FORK_MASK"
    doc = {
        "phase": "P15.1",
        "final_v1_touched": False,
        "no_learned_weights": True,
        "angle_to_azimuth": "bin = sin(угол прохода); −90° слева, +90° справа",
        "vector": "T4a+T5a минус T4b+T5b в каждом азимутальном бине",
        "map_only": map_rows,
        "means": {
            "corr_mirror": round(mean_key("corr_mirror"), 3),
            "corr_mirror_flipped": round(mean_key("corr_mirror_flipped"), 3),
            "corr_abs_mirror_flipped": round(mean_key("corr_abs_mirror_flipped"), 3),
            "corr_abs_wrong": round(mean_key("corr_abs_wrong"), 3),
            "corr_wrong": round(mean_key("corr_wrong"), 3),
            "energy_correct": round(mean_key("energy_correct"), 4),
            "energy_empty": round(mean_key("energy_empty"), 4),
        },
        "checks": {
            "mirror_matches_flipped_pattern": mirror_flips,
            "wrong_fork_less_alike_than_flipped_mirror": wrong_differs,
            "empty_quieter_than_correct": empty_quiet,
        },
        "camera_mix": cam_rows,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "P15_1_sectors.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if map_rows:
            node = map_rows[0]["node"]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            xs = np.arange(len(centres))
            for key, color in (("A_correct", "#1f77ff"), ("B_mirror", "#d62728"),
                               ("C_wrong", "#ff7f0e"), ("D_empty", "#888")):
                ax.plot(xs, vectors[node][key], color=color, lw=1.4, label=key)
            ax.set_xlabel("азимут, слева направо")
            ax.set_ylabel("T4/T5 a−b")
            ax.set_title(f"P15.1 — {node}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT / "p15_1_summary.png", dpi=120)
            plt.close(fig)
    except Exception as exc:
        print(f"  рисунок не построен: {exc}")

    print()
    print(f"  вердикт: {verdict}")
    print(f"  |зеркало, перевёрнутое| {doc['means']['corr_abs_mirror_flipped']:+.2f}, "
          f"|чужая| {doc['means']['corr_abs_wrong']:+.2f}")
    print(f"  записано: {OUT / 'P15_1_sectors.json'}")
    return 0 if spatial_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
