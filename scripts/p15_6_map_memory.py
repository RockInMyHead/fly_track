#!/usr/bin/env python3
"""P15.6 — show the map, then the camera. Do not add them.

The pulse is 200 ms of corridor, fixed beforehand. The camera then runs alone.
The brain is read at 0, 200, 500 and 1000 ms after the handoff. Those four
times are fixed too. Nothing here is chosen because a curve looked better.

0 ms is the end of the pulse, before any camera frame. The later three times
are after the camera has been the only image. The eye's motion estimate is
restarted at the cut so the first camera frame is not a cut against the map.
The nervous system is not restarted.

The question is whether the network, apart from the cells the current frame
drives directly, still matches the pulse a few hundred milliseconds later.

FINAL V1 is not modified.

    PYTHONPATH=. .venv/bin/python scripts/p15_6_map_memory.py
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

import p15_3_motion2d as P  # noqa: E402
import p15_map_vision as M  # noqa: E402
from fly_vo.config import FlyVOConfig  # noqa: E402
from fly_vo.malecns_engine import MaleCNSEngine  # noqa: E402
from fly_vo.visual_encoder import VideoVisualEncoder  # noqa: E402

OUT = ROOT / "output/p15"
VIDEO = ROOT / "webapp/media/VID00010_fixed.mp4"
DT = 0.02
MAP_MS = 200
MAP_STEPS = int(round(MAP_MS / (DT * 1000)))
DELAYS_MS = (0, 200, 500, 1000)
CAM_STEPS = int(round(DELAYS_MS[-1] / (DT * 1000)))
# Same noticeable-gap bar as P15.4. A trace also has to keep 5% of the
# pulse's own network difference, or a noisy correlation is not memory.
GAP = 0.05
RETAIN = 0.05
PAIRS = (("LEFT", "STRAIGHT"), ("STRAIGHT", "RIGHT"), ("LEFT", "RIGHT"))
HEADING = {"LEFT": -90.0, "STRAIGHT": 0.0, "RIGHT": 90.0}
FLIP = {"LEFT": "RIGHT", "RIGHT": "LEFT", "STRAIGHT": "STRAIGHT"}


def mirror_pair(pair: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(n for n in ("LEFT", "STRAIGHT", "RIGHT") if n in {FLIP[x] for x in pair})


def wrong_pair(pair: tuple[str, ...]) -> tuple[str, ...]:
    i = PAIRS.index(pair)
    return PAIRS[(i + 1) % len(PAIRS)]


def headings(pair: tuple[str, ...]) -> list[float]:
    return [HEADING[name] for name in pair]


def flip_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Left-right mirror of the picture. Motion reverses with it."""
    return [np.ascontiguousarray(fr[:, ::-1]) for fr in frames]


def grab_camera(video: Path, n: int, t_s: float) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"не открывается видео {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < n:
        raise RuntimeError(f"камера дала {len(frames)} кадров, нужно {n}")
    return frames


def input_cells(engine, enc, groups) -> set[int]:
    """Cells the current frame drives directly. They are not the memory."""
    cells = set()
    for fam in groups.values():
        for bucket in fam:
            cells.update(bucket)
    for idx in enc.loom_groups.values():
        cells.update(int(i) for i in np.asarray(idx).tolist())
    visual = np.asarray(engine.brain.visual)
    if visual.dtype == bool:
        cells.update(int(i) for i in np.flatnonzero(visual).tolist())
    else:
        cells.update(int(i) for i in visual.tolist())
    return cells


def mem_vec(fired, n: int, driven: set[int]) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    for i in np.asarray(fired).tolist():
        i = int(i)
        if i not in driven:
            v[i] = 1.0
    return v


def t4_vec(fired, groups) -> np.ndarray:
    got = set(int(i) for i in np.asarray(fired).tolist())
    n = len(groups["a"])
    h = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    for i in range(n):
        h[i] = len(groups["a"][i] & got) - len(groups["b"][i] & got)
        v[i] = len(groups["c"][i] & got) - len(groups["d"][i] & got)
    return P.pack(h, v)


def run_sequence(engine, enc, groups, driven, map_frames, cam_frames) -> dict[int, dict]:
    """Map pulse, then camera. Brain is not reset at the cut. Frontend is."""
    engine.reset(64)
    enc.reset()
    n = engine.n_neurons
    acc = np.zeros(n, dtype=np.float32)
    last_t4 = np.zeros(0, dtype=float)
    t = 0.0
    for fr in map_frames:
        t += DT
        eye, inj, _ = enc.encode_frame(fr, vertical=True)
        fired = engine.step(t, eye_drive=eye, inject=inj).fired
        acc += mem_vec(fired, n, driven)
        last_t4 = t4_vec(fired, groups)
    if len(map_frames):
        acc /= len(map_frames)
    reads = {0: {"mem": acc, "t4": last_t4, "n_mem": int(np.count_nonzero(acc))}}
    enc.reset()
    for i, fr in enumerate(cam_frames):
        t += DT
        eye, inj, _ = enc.encode_frame(fr, vertical=True)
        fired = engine.step(t, eye_drive=eye, inject=inj).fired
        elapsed = int(round((i + 1) * DT * 1000))
        if elapsed in DELAYS_MS and elapsed != 0:
            mem = mem_vec(fired, n, driven)
            reads[elapsed] = {
                "mem": mem,
                "t4": t4_vec(fired, groups),
                "n_mem": int(mem.sum()),
            }
    return reads


def gap_of(scores: dict[str, float]) -> float:
    rival = max(scores["mirror"], scores["wrong"], scores["empty"])
    return round(scores["correct"] - rival, 4)


def holds(scores: dict[str, float], retained: float) -> bool:
    return (
        scores["correct"] > scores["mirror"]
        and scores["correct"] > scores["wrong"]
        and scores["correct"] > scores["empty"]
        and gap_of(scores) >= GAP
        and retained >= RETAIN
    )


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if MAP_STEPS != 10 or CAM_STEPS != 50:
        raise SystemExit(f"шаги съехали: карта {MAP_STEPS}, камера {CAM_STEPS}")
    if not VIDEO.exists():
        print(f"нет камеры: {VIDEO}")
        return 1

    cfg = FlyVOConfig()
    engine = MaleCNSEngine(cfg)
    enc = VideoVisualEncoder(engine)
    groups = {
        "a": P.family_bins(enc.motion_injector, "a"),
        "b": P.family_bins(enc.motion_injector, "b"),
        "c": P.family_bins(enc.vertical_injector, "c"),
        "d": P.family_bins(enc.vertical_injector, "d"),
    }
    driven = input_cells(engine, enc, groups)
    w, h = cfg.encode_width, cfg.encode_height
    camera = grab_camera(VIDEO, CAM_STEPS, 20.0)
    blank = M.corridor_clip([], w, h, n=MAP_STEPS)
    print("P15.6 — импульс карты, затем камера. Сложения нет.")
    print(f"  импульс {MAP_MS} мс, отсчёты {list(DELAYS_MS)} мс, FINAL V1 не менялся")
    print(f"  клеток вне прямого входа: {engine.n_neurons - len(driven)}")

    by_delay: dict[int, list] = {ms: [] for ms in DELAYS_MS}
    for pair in PAIRS:
        pictures = {
            "correct": M.corridor_clip(headings(pair), w, h, n=MAP_STEPS),
            "mirror": flip_frames(M.corridor_clip(headings(pair), w, h, n=MAP_STEPS)),
            "wrong": M.corridor_clip(headings(wrong_pair(pair)), w, h, n=MAP_STEPS),
            "empty": blank,
            "camera": [],
        }
        ran = {
            kind: run_sequence(engine, enc, groups, driven, frames, camera)
            for kind, frames in pictures.items()
        }
        sig = ran["correct"][0]["mem"] - ran["empty"][0]["mem"]
        sig_energy = float(np.mean(np.abs(sig)))
        t4_template = ran["correct"][0]["t4"]
        print(f"  {'+'.join(pair):22s}  сеть на импульсе |Δ| {sig_energy:.5f}")
        for ms in DELAYS_MS:
            base = ran["empty"][ms]["mem"]
            scores = {}
            t4_scores = {}
            for kind in ("correct", "mirror", "wrong", "empty"):
                scores[kind] = round(P.corr(ran[kind][ms]["mem"] - base, sig), 4)
                t4_scores[kind] = round(P.corr(ran[kind][ms]["t4"], t4_template), 4)
            retained = 0.0 if sig_energy <= 0 else float(
                np.mean(np.abs(ran["correct"][ms]["mem"] - base)) / sig_energy
            )
            same_camera = round(P.corr(ran["correct"][ms]["mem"], ran["camera"][ms]["mem"]), 4)
            ok = holds(scores, retained)
            by_delay[ms].append({
                "pair": list(pair),
                "mirror_pair": list(mirror_pair(pair)),
                "wrong_pair": list(wrong_pair(pair)),
                "network_corr": scores,
                "network_gap": gap_of(scores),
                "retained_fraction": round(retained, 4),
                "t4_corr": t4_scores,
                "t4_gap": gap_of(t4_scores),
                "corr_with_camera_only": same_camera,
                "n_mem_correct": ran["correct"][ms]["n_mem"],
                "holds": ok,
            })
            print(f"    {ms:5d} мс  сеть {gap_of(scores):+.3f}  "
                  f"остаток {retained:.3f}  T4/T5 {gap_of(t4_scores):+.3f}  "
                  f"{'держится' if ok else 'нет'}")

    late = [ms for ms in (200, 500, 1000) if all(row["holds"] for row in by_delay[ms])]
    verdict = "MAP_MEMORY_HOLDS" if late else "VISUAL_MAP_MEMORY_NOT_FOUND"
    doc = {
        "phase": "P15.6",
        "final_v1_touched": False,
        "mixing": "none — map pulse, then camera frames, drives are never added",
        "map_pulse_ms": MAP_MS,
        "delays_ms": list(DELAYS_MS),
        "delays_fixed_before_run": True,
        "gap_required": GAP,
        "retain_required": RETAIN,
        "frontend_reset_at_cut": True,
        "brain_not_reset_at_cut": True,
        "memory_cells": "все, кроме фоторецепторов, T4/T5 и LC4/LPLC2 — их задаёт текущий кадр",
        "pairs": [list(p) for p in PAIRS],
        "triples_used": False,
        "video": str(VIDEO.relative_to(ROOT)),
        "by_delay": {str(ms): by_delay[ms] for ms in DELAYS_MS},
        "holds_at_ms": late,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = OUT / "P15_6_map_memory.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for pair in PAIRS:
            ys = []
            for ms in DELAYS_MS:
                row = next(r for r in by_delay[ms] if tuple(r["pair"]) == pair)
                ys.append(row["network_gap"])
            ax.plot(DELAYS_MS, ys, marker="o", label="+".join(pair))
        ax.axhline(GAP, color="k", lw=0.7, ls="--")
        ax.set_xlabel("мс после конца импульса")
        ax.set_ylabel("отрыв правильной карты от зеркала, чужой и пустой")
        ax.set_title(verdict)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "p15_6_map_memory.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  рисунок не построен: {exc}")

    print()
    print(f"  держится на {late if late else 'ни одном отсчёте после камеры'}")
    print(f"  вердикт: {verdict}")
    print(f"  записано: {path}")
    return 0 if verdict == "MAP_MEMORY_HOLDS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
