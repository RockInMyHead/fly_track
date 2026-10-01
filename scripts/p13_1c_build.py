#!/usr/bin/env python3
"""P13.1C — build the stratified sample that tests the threshold of 28.

The composition is fixed in data/p13/FROZEN_THRESHOLD_28.json before anything is built: forty clips
whose moment falls between 12 and 28, ten below 12, ten at 28 and above. Only the middle band can
change behaviour — those moments are MOVE under the old threshold and STOP under the new one — so
most of the sample is spent there. The outer ten are controls: the bottom must stay safe, the top
must not start freezing.

Every window is a new episode. Windows already reviewed in P13.1B are excluded by a fifteen-second
margin, so no clip asks the reviewer to judge a moment twice.

The clips carry nothing about their band. `p13_1b_server.py` serves them with the same page as
before, which shows no motion value, no class and no recording, and every clip is exactly five
seconds with the judged moment in the middle.

Usage:
    PYTHONPATH=. python scripts/p13_1c_build.py
    PYTHONPATH=. python scripts/p13_1c_build.py --force
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

from scripts.p13_1b_build import cut, motion_series  # noqa: E402

OUT = ROOT / "output/p13"
CLIPS = OUT / "p13_1c_clips"
MEDIA = ROOT / "webapp/media"
DATA = ROOT / "data/p13"
FREEZE = DATA / "FROZEN_THRESHOLD_28.json"
MANIFEST = OUT / "p13_1c_manifest.json"
KEY = DATA / "p13_1c_key.json"
OLD_KEY = DATA / "p13_1b_key.json"


def runs_in(mot: np.ndarray, dt: float, lo: float, hi: float, min_s: float) -> list[dict]:
    """Clean stretches wholly inside [lo, hi), with the mean motion of each.

    Wholly inside, not merely centred there: a window whose samples straddle the band edge would
    change class partway through, and the point of the band is that the threshold decides it one
    way or the other.
    """
    need = max(int(round(min_s / dt)), 2)
    runs, cur = [], []
    for i, x in enumerate(mot):
        if lo <= x < hi:
            cur.append(i)
        else:
            if len(cur) >= need:
                runs.append(cur)
            cur = []
    if len(cur) >= need:
        runs.append(cur)
    return [{
        "center": (r[0] + r[-1]) / 2.0 * dt,
        "run_s": round(len(r) * dt, 2),
        "motion": round(float(mot[r].mean()), 2),
    } for r in runs]


def build_band(video: str, mot: np.ndarray, dt: float, lo: float, hi: float,
               min_s: float, duration: float, taken: list[float], gap: float,
               rng: random.Random) -> list[dict]:
    """Candidate moments for one band on one recording, away from everything already taken."""
    half = 2.5
    out = []
    for w in runs_in(mot, dt, lo, hi, min_s):
        c = w["center"]
        if not (half <= c <= duration - half):
            continue
        if any(abs(c - t) < gap for t in taken):
            continue
        out.append(w)
    rng.shuffle(out)
    keep, chosen = [], []
    for w in out:
        if all(abs(w["center"] - c) >= gap for c in chosen):
            keep.append(w)
            chosen.append(w["center"])
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if MANIFEST.exists() and not a.force:
        print(f"{MANIFEST} уже существует — набор уже собран, повтор запрещён")
        return 1

    fr = json.loads(FREEZE.read_text(encoding="utf-8"))
    old = json.loads(OLD_KEY.read_text(encoding="utf-8"))["items"]
    held = json.loads((DATA / "FROZEN_STILLNESS_GATE.json").read_text(
        encoding="utf-8"))["held_out"]
    rules = fr["sample_rules"]
    gap = float(rules["separation_from_reviewed_s"])
    min_s = float(rules["clean_run_s"])
    bands = fr["bands"]

    print("=" * 102)
    print("P13.1C — набор для проверки порога 28")
    print("=" * 102)
    print(f"  порог заморожен: {fr['threshold']}")
    print(f"  состав: {bands['middle']['n']} из 12–28, {bands['low']['n']} из <12, "
          f"{bands['high']['n']} из ≥28")
    print(f"  чистый участок ≥{min_s} с, разнос ≥{gap:.0f} с от уже показанных и между новыми")
    print()

    old_by: dict[str, list[float]] = {}
    for i in old:
        old_by.setdefault(i["video"], []).append(i["center"])
    print(f"  уже показано человеку: {len(old)} окон — все исключаются")

    rng = random.Random(a.seed)
    pools = {k: {} for k in bands}
    durations: dict[str, float] = {}
    taken: dict[str, list[float]] = {v: list(old_by.get(v, [])) for v in held}

    for v in held:
        mo = motion_series(v)
        if mo is None:
            print(f"  {v}: признак не посчитался")
            continue
        mot, dt = mo
        durations[v] = len(mot) * dt
        for name, b in bands.items():
            lo, hi = float(b["range"][0]), float(b["range"][1])
            got = build_band(v, mot, dt, lo, hi, min_s, durations[v],
                             taken[v], gap, rng)
            pools[name][v] = got
            taken[v] += [w["center"] for w in got]
        print(f"  {v}: 12–28 {len(pools['middle'][v]):>3}, <12 {len(pools['low'][v]):>3}, "
              f"≥28 {len(pools['high'][v]):>3}")

    picked = []
    for name, b in bands.items():
        want = int(b["n"])
        pool = pools[name]
        names = [v for v in sorted(pool) if pool[v]]
        if not names:
            print(f"  {name}: нет кандидатов")
            continue
        per = [want // len(names)] * len(names)
        for i in range(want % len(names)):
            per[i] += 1
        got = []
        for v, k in zip(names, per):
            for w in pool[v][:k]:
                got.append({**w, "video": v})
        short = want - len(got)
        if short > 0:
            rest = [{**w, "video": v} for v in names for w in pool[v] if w not in got]
            rng.shuffle(rest)
            got += rest[:short]
            short = want - len(got)
        if short:
            print(f"  {name}: набралось {len(got)} из {want}, не хватает {short}")
        print(f"  отобрано {name}: {len(got)} "
              f"по роликам {dict(sorted(Counter(w['video'] for w in got).items()))}")
        picked += [(name, w) for w in got]

    if not picked:
        print("  ничего не отобрано")
        return 1

    rng.shuffle(picked)

    CLIPS.mkdir(parents=True, exist_ok=True)
    for f in list(CLIPS.glob("*.mp4")) + list(CLIPS.glob("*.json")):
        f.unlink()

    items, key = [], []
    for i, (band, w) in enumerate(picked, 1):
        wid = f"c{i:03d}"
        dest = CLIPS / f"{wid}.mp4"
        if not cut(w["video"], w["center"], dest, durations[w["video"]]):
            print(f"  {wid} ({w['video']} t={w['center']:.0f}): не нарезался")
            continue
        meta = json.loads(dest.with_suffix(".json").read_text(encoding="utf-8"))
        items.append({"id": wid, "file": f"{wid}.mp4", **meta})
        key.append({"id": wid, "video": w["video"], "band": band,
                    "center": round(w["center"], 2), "run_s": w["run_s"],
                    "motion": w["motion"],
                    "clip_t0": meta["clip_t0"], "moment_in_clip": meta["moment_in_clip"]})

    MANIFEST.write_text(json.dumps({
        "phase": "P13.1C — набор для проверки порога 28",
        "frozen_at": fr["frozen_at"], "threshold": fr["threshold"],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_s": rules["clip_s"], "seed": a.seed,
        "held_out_used": held,
        "per_band": {k: sum(1 for x in key if x["band"] == k) for k in bands},
        "clips": items,
        "instructions_to_reviewer": (
            "смотреть клип; отмеченный момент — середина, страница останавливается там. "
            "Ответить: в этот момент человек идёт или стоит"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    KEY.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps({
        "note": "ключ к набору P13.1C. Странице не отдаётся.",
        "threshold": fr["threshold"], "clip_s": rules["clip_s"], "items": key,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"  нарезано клипов: {len(items)}")
    print(f"  по полосам: {dict(Counter(x['band'] for x in key))}")
    print(f"  манифест: {MANIFEST}")
    print(f"  ключ:     {KEY}")
    print()
    print("  Дальше: страница разметки на этих файлах, вслепую.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
