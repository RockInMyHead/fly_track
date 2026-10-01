#!/usr/bin/env python3
"""
Registry of videos the frozen P07 chain can be applied to.

Both `p07_fly_readout.py` and `p07_trajectory.py` had the two known clips written into
them. Adding a third means a new brain recording, and editing two frozen scripts to accept
it would risk changing what "frozen" means. Instead the list lives here, and either script
can be pointed at a new clip by writing one line into the registry file.

Each entry gives the brain recording for that clip and the target metadata that goes with
it. Both are produced by the brain run, not chosen here.

Usage from another script:
    from p07_videos import videos
    for name, paths in videos().items():
        ...

From the shell, to add a clip before running the frozen chain:
    PYTHONPATH=. python scripts/p07_videos.py --add VID00003 \\
        --trace output/p073_ground_truth/spike_trace.npz \\
        --targets output/p073_ground_truth/targets.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/p07_videos.json"

# the two clips everything so far was built on
BASE = {
    "VID00001": {"trace": "output/p053_threshold/spike_trace.npz",
                 "targets": "output/p053_threshold/targets.csv"},
    "VID00002": {"trace": "output/p06_neurons/spike_trace.npz",
                 "targets": "output/p06_neurons/targets.csv"},
}


def videos() -> dict[str, dict[str, Path]]:
    """All videos the chain knows about, base clips plus anything registered."""
    out = {k: {"trace": ROOT / v["trace"], "targets": ROOT / v["targets"]}
           for k, v in BASE.items()}
    if REGISTRY.exists():
        try:
            extra = json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  реестр {REGISTRY} не прочитан: {e}")
            extra = {}
        for name, v in extra.items():
            trace = ROOT / v["trace"]
            targets = ROOT / v["targets"]
            if trace.exists() and targets.exists():
                out[name] = {"trace": trace, "targets": targets}
            else:
                # an entry whose recording is missing must not silently join the loop,
                # or every frozen script would start failing on a stale name
                print(f"  {name} в реестре, но записи нет: {trace}")
    return out


# Which labelling set belongs to which clip. This is stated rather than inferred, because
# inferring it from the file name got it wrong: `review_set_v2.json` is the second round of
# the FIRST clip, not the second clip, and a name-based rule silently attributed its 33
# turns to VID00002. Every window of that set carries a timestamp from the first clip's
# timeline, so evaluating them against the second clip sampled arbitrary moments and made
# the validation numbers meaningless. All three scripts in the chain import this mapping.
LABEL_SETS = [
    ("review_set.json", "turn_verdicts.json", "VID00001"),
    ("review_set_v2.json", "turn_verdicts_v2.json", "VID00001"),
    ("review_set_vid2.json", "turn_verdicts_vid2.json", "VID00002"),
    ("review_set_vid5.json", "turn_verdicts_vid5.json", "VID00005"),
]


def labelled_turns(video: str | None = None) -> list[dict]:
    """Human-confirmed turns, tagged with the clip they belong to."""
    out = []
    for setname, vername, clip in LABEL_SETS:
        if video is not None and clip != video:
            continue
        setp = ROOT / "data/p01r" / setname
        verp = ROOT / "data/p01r" / vername
        if not setp.exists() or not verp.exists():
            continue
        rs = json.loads(setp.read_text(encoding="utf-8"))["candidates"]
        vd = json.loads(verp.read_text(encoding="utf-8"))["verdicts"]
        for x in rs:
            if x.get("repeat_of"):
                continue
            v = vd.get(x["id"])
            if not v:
                continue
            truth = x["camera_direction"] if v["verdict"] == "correct" \
                else (v.get("actual_direction") or "REJECT")
            if truth in ("LEFT", "RIGHT"):
                out.append({"id": x["id"], "video": clip, "t0": x["t0"],
                            "t1": x["t1"], "kind": truth})
    return sorted(out, key=lambda e: e["t0"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", metavar="NAME")
    ap.add_argument("--trace")
    ap.add_argument("--targets")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.add:
        print("известные видео:")
        for name, v in videos().items():
            print(f"  {name:>12s}  {v['trace'].relative_to(ROOT)}")
        return

    if not args.trace or not args.targets:
        print("нужны --trace и --targets")
        return
    trace = Path(args.trace)
    targets = Path(args.targets)
    if not trace.exists() or not targets.exists():
        print(f"файлы не найдены: {trace.exists()=} {targets.exists()=}")
        return

    doc = {}
    if REGISTRY.exists():
        doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
    doc[args.add] = {
        "trace": str(trace if trace.is_absolute() else ROOT / trace),
        "targets": str(targets if targets.is_absolute() else ROOT / targets),
    }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"добавлено {args.add} -> {REGISTRY}")
    print("теперь замороженные скрипты подхватят это видео без правок")


if __name__ == "__main__":
    main()
