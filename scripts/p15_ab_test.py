#!/usr/bin/env python3
"""P15.A — does a local map, seen through the eye, change MaleCNS at a junction?

Four pictures, same seed, same camera frame:

    A  camera only          (alpha = 0; must match today's encoder)
    B  camera + correct map
    C  camera + mirrored map
    D  camera + empty map
    E  camera + a different junction's map

Alphas are fixed in advance: 0, 0.1, 0.25. None is chosen because a route looked better.
The graph is not allowed to vote for an edge in code.

    PYTHONPATH=. .venv/bin/python scripts/p15_ab_test.py --limit 8 --steps 15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402
from p08_graph import Graph  # noqa: E402

OUT = ROOT / "output/p15"
ALPHAS = (0.0, 0.1, 0.25)
KINDS = ("correct", "mirror", "empty", "wrong")
PREFERRED = ("J35", "T49", "T50")
SEED = 64


def junctions(g: Graph, limit: int) -> list[dict]:
    """Forks taken from the graph alone. No human route is read."""
    found = []
    seen = set()

    def consider(node: str) -> None:
        if node in seen or node not in g.nodes:
            return
        for in_edge in g.edges_at(node):
            exits = M.forward_exits(g, node, in_edge)
            if len(exits) < 2:
                continue
            seen.add(node)
            found.append({"node": node, "in_edge": in_edge, "exits": exits})
            return

    for name in PREFERRED:
        consider(name)
    for node in sorted(g.nodes):
        if len(found) >= limit:
            break
        consider(node)
    return found[:limit]


def family(counts: dict[str, float], fam: str) -> float:
    return sum(v for k, v in counts.items()
               if k.startswith(("T4" + fam, "T5" + fam)))


def opponent_balance(counts: dict[str, float]) -> float:
    """(a − b) / (a + b). This is the direction channel: a yaw drives both eyes alike."""
    a, b = family(counts, "a"), family(counts, "b")
    den = a + b
    return 0.0 if den <= 0 else (a - b) / den


def side_balance(counts: dict[str, float]) -> float:
    """(right − left) / (right + left). Kept as a control; it is not the yaw channel."""
    left = right = 0.0
    for name, n in counts.items():
        if not name.startswith(("T4", "T5")):
            continue
        if name.endswith("_L"):
            left += n
        elif name.endswith("_R"):
            right += n
    den = left + right
    if den <= 0:
        return 0.0
    return (right - left) / den


def map_hint(exits: list[dict]) -> float:
    """Mean sin(angle): positive when the open ways sit to the right."""
    if not exits:
        return 0.0
    return float(np.mean([math.sin(math.radians(e["deg"])) for e in exits]))


def grab_pair(video: Path | None, t_s: float) -> list | None:
    if video is None or not video.exists():
        return None
    frames = []
    for extra in (0.0, 0.04):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t_s + extra:.3f}",
               "-i", str(video), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not r.stdout:
            return None
        arr = np.frombuffer(r.stdout, dtype=np.uint8)
        fr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if fr is None:
            return None
        frames.append(fr)
    return frames


def run_condition(engine, cam_enc, map_enc, groups, cam_frames, map_frames,
                  alpha: float) -> dict[str, float]:
    """One encoder for the camera, another for the map. Their drives are added.

    A single encoder cannot see both clips: it would mix the two motions into one flow.
    """
    engine.reset(SEED)
    cam_enc.reset()
    map_enc.reset()
    acc = {name: 0.0 for name in groups}
    n = max(len(map_frames), 2)
    for i in range(n):
        eye_c, inj_c, _ = cam_enc.encode_frame(cam_frames[i % len(cam_frames)])
        if alpha == 0.0:
            eye, inj = eye_c, list(inj_c)
        else:
            eye_m, inj_m, _ = map_enc.encode_frame(map_frames[min(i, len(map_frames) - 1)])
            eye, inj = M.mix_drives((eye_c, inj_c), (eye_m, inj_m), alpha)
        result = engine.step(i * engine.config.brain_dt, eye_drive=eye, inject=inj)
        got = M.spikes_by_group(result.fired, groups)
        for name, k in got.items():
            acc[name] += k
    return {k: v / n for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--video", default="VID00010")
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    a = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    maps = OUT / "local_maps"
    maps.mkdir(exist_ok=True)

    g = Graph.load(a.graph)
    events = junctions(g, a.limit)
    if len(events) < 2:
        print("нужно хотя бы две развилки в графе")
        return 1

    video = None
    for cand in (ROOT / "data/p01r" / f"{a.video}.AVI",
                 Path("/Volumes/NO NAME/DCIM") / f"{a.video}.AVI"):
        if cand.exists():
            video = cand
            break

    print("P15.A — карта как зрительный вход (FINAL V1 не используется)")
    print(f"  развилок {len(events)}, шагов мозга {a.steps}, seed {SEED}")
    print(f"  видео: {video if video else 'нет — только карта против пустого кадра'}")

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    cam_enc = VideoVisualEncoder(engine)
    map_enc = VideoVisualEncoder(engine)
    groups = {**cam_enc.t4_t5_groups, **cam_enc.loom_groups}
    w, h = cfg.encode_width, cfg.encode_height

    # one grey camera clip when the file is missing, so A/B still compares map vs no map
    blank_cam = [np.full((h, w, 3), 40, np.uint8) for _ in range(2)]

    rows = []
    for i, ev in enumerate(events):
        wrong = events[(i + 1) % len(events)]
        cam_frames = grab_pair(video, 30.0 + 15.0 * i) or blank_cam
        hint = map_hint(ev["exits"])
        if i < 3:
            clip = M.map_clip(ev["exits"], w, h, "correct")
            cv2.imwrite(str(maps / f"event_{i:03d}_map.png"), clip[1])
            cv2.imwrite(str(maps / f"event_{i:03d}_camera.png"), cam_frames[-1])

        for alpha in ALPHAS:
            for kind in KINDS:
                if alpha == 0.0 and kind != "correct":
                    # alpha 0 ignores the picture; one row is the camera-only control
                    continue
                if kind == "wrong":
                    exits = wrong["exits"]
                elif kind == "empty":
                    exits = []
                else:
                    exits = ev["exits"]
                mode = "A_camera" if alpha == 0.0 else {
                    "correct": "B_correct", "mirror": "C_mirror",
                    "empty": "D_empty", "wrong": "E_wrong",
                }[kind]
                counts = run_condition(
                    engine, cam_enc, map_enc, groups, cam_frames,
                    M.map_clip(exits, w, h, "empty" if kind == "empty" else kind),
                    0.0 if alpha == 0.0 else alpha,
                )
                rows.append({
                    "event": i, "node": ev["node"], "in_edge": ev["in_edge"],
                    "n_exits": len(ev["exits"]),
                    "exits": ";".join(f"{e['edge']}@{e['deg']:.0f}" for e in ev["exits"]),
                    "map_hint": round(hint, 4),
                    "alpha": alpha, "kind": kind if alpha else "none", "mode": mode,
                    "side_index": round(side_balance(counts), 4),
                    "t4t5_opponent": round(opponent_balance(counts), 4),
                    "T4T5_L": round(sum(v for k, v in counts.items()
                                        if k.startswith(("T4", "T5")) and k.endswith("_L")), 3),
                    "T4T5_R": round(sum(v for k, v in counts.items()
                                        if k.startswith(("T4", "T5")) and k.endswith("_R")), 3),
                    "LC4": round(counts.get("LC4_L", 0) + counts.get("LC4_R", 0), 3),
                    "LPLC2": round(counts.get("LPLC2_L", 0) + counts.get("LPLC2_R", 0), 3),
                })
        print(f"  {ev['node']}  выходов {len(ev['exits'])}  hint {hint:+.2f}")

    fields = list(rows[0].keys())
    with (OUT / "responses.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=fields)
        wri.writeheader()
        wri.writerows(rows)

    # comparison: at each alpha>0, |side_index - map_hint sign agreement| vs camera-only
    summary = []
    by_event = {}
    for r in rows:
        by_event.setdefault(r["event"], []).append(r)
    for ev_i, group in by_event.items():
        base = next(r for r in group if r["mode"] == "A_camera")
        hint = base["map_hint"]
        for r in group:
            if r["mode"] == "A_camera":
                continue
            summary.append({
                "event": ev_i, "node": r["node"], "mode": r["mode"], "alpha": r["alpha"],
                "map_hint": hint,
                "opponent_camera": base["t4t5_opponent"],
                "opponent_here": r["t4t5_opponent"],
                "delta_opponent": round(r["t4t5_opponent"] - base["t4t5_opponent"], 4),
                "side_camera": base["side_index"],
                "side_here": r["side_index"],
                "delta_side": round(r["side_index"] - base["side_index"], 4),
            })
    with (OUT / "ab_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(summary[0].keys()) if summary else ["event"])
        wri.writeheader()
        wri.writerows(summary)
    with (OUT / "controls.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=fields)
        wri.writeheader()
        wri.writerows([r for r in rows if r["mode"] != "A_camera"])

    def mean_abs_delta(mode: str, alpha: float) -> float:
        xs = [abs(s["delta_opponent"]) for s in summary
              if s["mode"] == mode and s["alpha"] == alpha]
        return float(np.mean(xs)) if xs else float("nan")

    verdict_bits = {}
    for alpha in (0.1, 0.25):
        verdict_bits[str(alpha)] = {
            "mean_abs_delta_correct": mean_abs_delta("B_correct", alpha),
            "mean_abs_delta_mirror": mean_abs_delta("C_mirror", alpha),
            "mean_abs_delta_empty": mean_abs_delta("D_empty", alpha),
            "mean_abs_delta_wrong": mean_abs_delta("E_wrong", alpha),
        }

    report = {
        "phase": "P15.A",
        "final_v1_touched": False,
        "question": "меняет ли локальная карта, прошедшая через зрительный вход, side_index T4/T5",
        "alphas_fixed_in_advance": list(ALPHAS),
        "alpha_not_chosen_by_trajectory": True,
        "seed": SEED,
        "steps": a.steps,
        "n_events": len(events),
        "video": str(video) if video else None,
        "camera_was_real": video is not None,
        "events": [{"node": e["node"], "in_edge": e["in_edge"],
                    "exits": e["exits"]} for e in events],
        "effect_vs_camera": verdict_bits,
        "reading": ("если correct заметно двигает side_index, а mirror двигает его в другую "
                    "сторону относительно correct — карта доходит до мозга. если correct ≈ "
                    "mirror ≈ empty ≈ wrong — карта ничего не даёт. alpha здесь не выбирается."),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        labels, vals = [], []
        for alpha in (0.1, 0.25):
            for mode, short in (("B_correct", "прав."), ("C_mirror", "зерк."),
                                ("D_empty", "пуст."), ("E_wrong", "чужая")):
                labels.append(f"{short}\nα={alpha}")
                vals.append(mean_abs_delta(mode, alpha))
        ax.bar(range(len(vals)), vals, color="#3d6b8f")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("|Δ opponent T4a/T5a − T4b/T5b|")
        ax.set_title("P15.A — сдвиг направления T4/T5 от локальной карты")
        fig.tight_layout()
        fig.savefig(OUT / "summary.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  summary.png не построен: {exc}")

    print(f"  записано: {OUT}")
    print(f"  эффект: {json.dumps(verdict_bits, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
