#!/usr/bin/env python3
"""P13.1B — build the blind STOP/MOVE review, by the corrected design.

WHY THIS PASS EXISTS
--------------------
`p13_1_gate` reported 107 metres of invented walking on the held-out chunks reduced to 5. That
number is close to guaranteed by construction: STOP is defined as `frame_motion < 12` and the gate
zeroes the speed exactly when `frame_motion < 12`, so of course the walker stops there. It shows the
code does what it claims. It does not show that `frame_motion < 12` means "the person is standing"
rather than "walking smoothly down an unvarying corridor".

That question needs a measurement sharing nothing with the gate: a person watching the clip. This
script picks the clips, blinds them, and hands them over.

THE DESIGN, AND WHY IT IS NOT THE OBVIOUS ONE
---------------------------------------------
The first attempt required five seconds in which *every* sample of the motion signal sat below the
threshold. Three held-out chunks produced no such window at all — in VID00008 the longest below-
threshold stretch is 4.0 seconds, in VID00013 it is 3.5, in VID00017 it is 1.0 — and the sample came
out 15 STOP against 30 MOVE instead of 30 and 30.

The fix is to separate *selecting* a moment from *showing* one:

    selecting   a clean stretch on one side of the threshold, at least 2.5 seconds long. Its centre
                is the moment that will be judged. Nothing is shown yet.
    showing     five seconds centred on that moment, for every clip, whatever its class.

Two consequences, and both matter. The reviewer cannot tell the class from the length of the clip,
because every clip is five seconds. And the moment judged is genuinely inside a clean stretch,
because that is how it was chosen, while the clip around it may contain movement — which is honest,
since a person standing for two seconds is still a person standing.

The marker is not burned into the video. A stamp appearing at the instant being judged could be read
as a cue, and the page already knows where the centre is: the clip is always five seconds and the
moment is always the middle. The page pauses there on first play and marks it on the timeline.

A SECOND CHANGE FROM THE FIRST ATTEMPT
--------------------------------------
Thirty STOP clips, not sixty. Sixty clean stretches exist across the held-out chunks, and thirty of
them are drawn at random, spread over the recordings that have any. VID00017 has none — its longest
below-threshold stretch is one second — so the sample is drawn from the other eight.

Usage:
    PYTHONPATH=. python scripts/p13_1b_build.py
    PYTHONPATH=. python scripts/p13_1b_build.py --per-class 30 --seed 20260924
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.p13_stillness import frame_motion  # noqa: E402

OUT = ROOT / "output/p13"
CLIPS = OUT / "p13_1b_clips"
MEDIA = ROOT / "webapp/media"
DATA = ROOT / "data/p13"
FREEZE = DATA / "FROZEN_STILLNESS_GATE.json"
MANIFEST = OUT / "p13_1b_manifest.json"
KEY = DATA / "p13_1b_key.json"

MIN_RUN_S = 2.5      # how long a clean stretch must be to contribute a moment
CLIP_S = 5.0         # what the reviewer is shown, for every clip


def motion_series(video: str) -> tuple[np.ndarray, float] | None:
    """The indicator, from the cache the gate wrote, or measured the same way if absent."""
    cache = OUT / "motion" / f"{video}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["mot"], float(d["dt"])
    prev = MEDIA / f"{video}_fixed.mp4"
    if not prev.exists():
        return None
    mot, dt = frame_motion(prev)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, mot=mot, dt=np.float32(dt))
    return mot, dt


def clean_runs(mot: np.ndarray, dt: float, th: float, below: bool,
               duration: float | None = None) -> list[dict]:
    """Stretches on one side of the threshold, long enough to contain a moment worth judging.

    Returns the centre of each stretch, its length, and how far inside or outside the threshold its
    mean sits. `below` picks the class: True for the STOP side, False for the MOVE side.

    `duration` keeps only the stretches whose centre is at least half a clip away from either end of
    the recording. Without it, a stretch near the start or the end produces a clip that cannot be
    centred on its own moment, and the shell shifts — one of the sixty came out with the moment at
    1.0 s instead of 2.5. The reviewer is told the moment is the middle of the clip, so it has to be
    the middle of the clip.
    """
    idx, cur = [], []
    for i, x in enumerate(mot):
        if (x < th) if below else (x >= th):
            cur.append(i)
        else:
            if cur:
                idx.append(cur)
                cur = []
    if cur:
        idx.append(cur)

    half = CLIP_S / 2.0
    out = []
    for run in idx:
        length = len(run) * dt
        if length < MIN_RUN_S:
            continue
        center = (run[0] + run[-1]) / 2.0 * dt
        if duration is not None and not (half <= center <= duration - half):
            continue
        seg = mot[run]
        mean = float(seg.mean())
        out.append({
            "center": center,
            "run_s": round(length, 2),
            "margin": round(mean - th, 2),          # negative means inside the STOP side
            "extreme": round(float(seg.max() if below else seg.min()), 2),
        })
    return out


def spread(items: list[dict], k: int, rng: random.Random, gap: float) -> list[dict]:
    """Up to k items at random, no two closer than `gap` seconds within one recording."""
    order = items[:]
    rng.shuffle(order)
    taken = []
    for w in order:
        if all(abs(w["center"] - x["center"]) >= gap for x in taken):
            taken.append(w)
            if len(taken) == k:
                break
    return taken


def draw_class(pools: dict[str, list[dict]], k: int, rng: random.Random,
               gap: float) -> list[dict]:
    """k moments for one class, spread across the recordings that can supply them.

    A plain draw over the pooled candidates could land all thirty in one corridor and end up
    measuring that corridor instead of the rule. Each recording contributes as evenly as it can;
    a recording with nothing to give is skipped rather than counted as an empty slot.
    """
    names = [v for v in sorted(pools) if pools[v]]
    if not names:
        return []
    per = [k // len(names)] * len(names)
    for i in range(k % len(names)):
        per[i] += 1

    out: list[dict] = []
    for v, want in zip(names, per):
        for w in spread(pools[v], want, rng, gap):
            out.append({**w, "video": v})
    short = k - len(out)
    if short > 0:
        rest = []
        for v in names:
            rest += [{**w, "video": v} for w in pools[v]]
        rng.shuffle(rest)
        for w in rest:
            if all(not (w["video"] == x["video"] and abs(w["center"] - x["center"]) < gap)
                   for x in out):
                out.append(w)
                short -= 1
                if short == 0:
                    break
    rng.shuffle(out)
    return out


def cut(video: str, center: float, dest: Path, duration: float) -> bool:
    """Five seconds centred on the moment, and exactly five — 150 frames, not 149.

    Trimming by time gave clips of 4.9667 s for twenty-six of the sixty, because a seek lands on a
    frame boundary and the last frame can fall outside the window. One frame is invisible to a
    reviewer, and the 11-against-15 split across the classes is consistent with chance, but a
    difference in clip length that could in principle carry the answer is worth removing rather than
    arguing about. Asking for a frame count makes every clip the same length by construction.
    """
    half = CLIP_S / 2.0
    t0 = max(0.0, min(center - half, duration - CLIP_S))
    frames = int(round(CLIP_S * 30))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}",
           "-i", str(MEDIA / f"{video}_fixed.mp4"), "-frames:v", str(frames),
           "-vf", "scale=640:360", "-r", "30",
           "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        return False
    # where the moment sits inside the clip, so the page can pause there exactly
    dest.with_suffix(".json").write_text(json.dumps({
        "clip_t0": round(t0, 3), "moment_in_clip": round(center - t0, 3),
    }), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--force", action="store_true",
                    help="пересобрать поверх прежнего отбора (дизайн изменён)")
    a = ap.parse_args()

    if MANIFEST.exists() and not a.force:
        print(f"{MANIFEST} уже существует. Пересборка только с --force,")
        print("  и только если дизайн изменён осознанно.")
        return 1

    fr = json.loads(FREEZE.read_text(encoding="utf-8"))
    th = float(fr["threshold"])
    held = fr["held_out"]

    print("=" * 104)
    print("P13.1B — слепая ручная проверка STOP/MOVE")
    print("=" * 104)
    print(f"  порог {th} (заморожен {fr['frozen_at']})")
    print(f"  куски: только отложенные — {len(held)}")
    print(f"  кандидат: чистый участок ≥{MIN_RUN_S} с, оценивается его ЦЕНТР")
    print(f"  показывается: {CLIP_S:.0f} с вокруг центра, ОДИНАКОВО для обоих классов")
    print(f"  окон на класс: {a.per_class}, разнос ≥{a.gap:.0f} с")
    print()

    ready = [v for v in held if (MEDIA / f"{v}_fixed.mp4").exists()]
    if len(ready) < len(held):
        print(f"  СТОП: готово видео {len(ready)} из {len(held)} — отбор по неполному набору")
        print(f"  ждут: {[v for v in held if v not in ready]}")
        return 1

    rng = random.Random(a.seed)
    pools = {"STOP": {}, "MOVE": {}}
    durations: dict[str, float] = {}
    for v in ready:
        mo = motion_series(v)
        if mo is None:
            print(f"  {v}: признак не посчитался")
            continue
        mot, dt = mo
        durations[v] = len(mot) * dt
        pools["STOP"][v] = clean_runs(mot, dt, th, below=True, duration=durations[v])
        pools["MOVE"][v] = clean_runs(mot, dt, th, below=False, duration=durations[v])
        print(f"  {v}: участков STOP {len(pools['STOP'][v]):>3}, "
              f"MOVE {len(pools['MOVE'][v]):>4}  "
              f"(длит. {durations[v]:.0f} с)")

    picked = []
    for cls in ("STOP", "MOVE"):
        got = draw_class(pools[cls], a.per_class, rng, a.gap)
        by = dict(sorted(Counter(w["video"] for w in got).items()))
        print(f"  отобрано {cls}: {len(got)}  по роликам {by}")
        picked += [(cls, w) for w in got]

    if not picked:
        print("  ни одного момента не отобрано")
        return 1

    # interleave, so a run of one class cannot give the answer away by position
    rng.shuffle(picked)

    CLIPS.mkdir(parents=True, exist_ok=True)
    for f in list(CLIPS.glob("*.mp4")) + list(CLIPS.glob("*.json")):
        f.unlink()

    items, key = [], []
    for i, (cls, w) in enumerate(picked, 1):
        wid = f"w{i:03d}"
        dest = CLIPS / f"{wid}.mp4"
        if not cut(w["video"], w["center"], dest, durations[w["video"]]):
            print(f"  {wid} ({w['video']} t={w['center']:.0f}): не нарезался, пропуск")
            continue
        meta = json.loads(dest.with_suffix(".json").read_text(encoding="utf-8"))
        items.append({"id": wid, "file": f"{wid}.mp4", **meta})
        key.append({"id": wid, "video": w["video"], "gate": cls,
                    "center": round(w["center"], 2), "run_s": w["run_s"],
                    "margin": w["margin"], "extreme": w["extreme"],
                    "clip_t0": meta["clip_t0"], "moment_in_clip": meta["moment_in_clip"]})

    MANIFEST.write_text(json.dumps({
        "phase": "P13.1B — слепая ручка STOP/MOVE",
        "design": "кандидат — чистый участок ≥2.5 с на одной стороне порога; оценивается его "
                  "центр; показывается 5 с вокруг центра, одинаково для обоих классов",
        "why_not_five_seconds_of_clean": (
            "первый вариант требовал 5 с подряд ниже порога; в VID00008, VID00013 и VID00017 таких "
            "окон нет вообще (самые длинные участки 4.0, 3.5 и 1.0 с), и выборка вышла 15 STOP "
            "против 30 MOVE"
        ),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": th, "clip_s": CLIP_S, "min_run_s": MIN_RUN_S, "seed": a.seed,
        "held_out_used": ready,
        "per_class": {c: sum(1 for k in key if k["gate"] == c) for c in ("STOP", "MOVE")},
        "clips": items,
        "instructions_to_reviewer": (
            "смотреть клип; отмеченный момент — его середина, страница останавливается там. "
            "Ответить: в этот момент человек идёт или стоит"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    KEY.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps({
        "note": "ключ к слепому отбору P13.1B. Странице не отдаётся.",
        "threshold": th, "clip_s": CLIP_S, "seed": a.seed, "items": key,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    n_stop = sum(1 for k in key if k["gate"] == "STOP")
    n_move = len(key) - n_stop
    print()
    print(f"  нарезано клипов: {len(items)} (STOP {n_stop}, MOVE {n_move})")
    print(f"  манифест: {MANIFEST}")
    print(f"  ключ:     {KEY}")
    print()
    print("  Дальше: scripts/p13_1b_server.py и разметить вслепую.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
