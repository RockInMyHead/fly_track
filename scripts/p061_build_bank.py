#!/usr/bin/env python3
"""
P06.1 — labelling bank for the second video.

Same design as the second round of the first clip: candidates proposed by the frozen
classifier, labels assigned by eye, controls mixed in, and repeats to measure labelling
noise. What makes this clip worth the effort is that the scenery is different: the frozen
cells are tested on a corridor they have never seen.

Turn candidates are taken balanced, and controls are spread across the clip so the
reviewer cannot tell where the turns are.

Usage:
    PYTHONPATH=. python scripts/p061_build_bank.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "data/p06/windows_v1.json"
OUT_JSON = ROOT / "data/p01r/review_set_vid2.json"
OUT_DIR = ROOT / "data/p01r"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-turns", type=int, default=36, help="must be even")
    ap.add_argument("--n-controls", type=int, default=24)
    ap.add_argument("--n-repeats", type=int, default=6)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--windows", default=None, help="candidate file to draw from")
    ap.add_argument("--out", default=None, help="where to write the bank")
    args = ap.parse_args()

    global WINDOWS, OUT_JSON
    if args.windows:
        WINDOWS = (Path(args.windows) if str(args.windows).startswith("/")
                   else ROOT / args.windows)
    if args.out:
        OUT_JSON = (Path(args.out) if str(args.out).startswith("/")
                    else ROOT / args.out)
    if not WINDOWS.exists():
        print(f"нет файла кандидатов: {WINDOWS}")
        return

    rng = np.random.default_rng(args.seed)
    ws = json.loads(WINDOWS.read_text(encoding="utf-8"))["windows"]
    turns = [w for w in ws if w["kind"] in ("LEFT", "RIGHT")]
    ctrl = [w for w in ws if w["kind"] in ("FWD", "STATIC")]

    lefts = sorted([w for w in turns if w["kind"] == "LEFT"], key=lambda z: z["t0"])
    rights = sorted([w for w in turns if w["kind"] == "RIGHT"], key=lambda z: z["t0"])
    per_side = args.n_turns // 2

    def spread(items: list[dict], k: int) -> list[dict]:
        """Evenly spaced pick, so the sample covers the whole clip."""
        if len(items) <= k:
            return items
        idx = np.linspace(0, len(items) - 1, k).round().astype(int)
        return [items[i] for i in idx]

    sel_turns = spread(lefts, per_side) + spread(rights, per_side)
    sel_ctrl = spread(sorted(ctrl, key=lambda z: z["t0"]), args.n_controls)

    print("P06.1 — банк разметки для второго видео")
    print(f"  кандидатов в ролике: {len(turns)} поворотов "
          f"({dict(Counter(w['kind'] for w in turns))}), {len(ctrl)} FWD/STATIC")
    print(f"  отобрано: {len(sel_turns)} поворотов "
          f"({sum(1 for w in sel_turns if w['kind'] == 'LEFT')} LEFT, "
          f"{sum(1 for w in sel_turns if w['kind'] == 'RIGHT')} RIGHT), "
          f"{len(sel_ctrl)} контролей")

    n_rep = args.n_repeats // 2
    rep_src = sel_turns[:n_rep] + sel_ctrl[:n_rep]

    def make(x: dict, cid: str) -> dict:
        kind = x["kind"]
        return {
            "kind": kind, "t0": x["t0"], "t1": x["t1"],
            "duration_s": x.get("duration_s", round(x["t1"] - x["t0"], 2)),
            "a": x.get("a", 0.0), "divergence": x.get("divergence", 0.0),
            "residual": x.get("residual", 0.0),
            "activity": x.get("activity", 0.0),
            "activity_vertical": x.get("activity_vertical", 0.0),
            "camera_direction": kind,
            "confidence": x.get("confidence", "medium"),
            "id": cid,
            "expected": (f"камера повернула {kind}" if kind in ("LEFT", "RIGHT")
                         else f"камера {kind.lower()}"),
        }

    items = [make(x, f"v{i:03d}") for i, x in enumerate(sel_turns)]
    items += [make(x, f"w{j:03d}") for j, x in enumerate(sel_ctrl)]
    for k, x in enumerate(rep_src):
        rep = make(x, f"q{k:03d}")
        src = next(it["id"] for it in items
                   if abs(it["t0"] - x["t0"]) < 1e-6 and it["kind"] == x["kind"])
        rep["repeat_of"] = src
        items.append(rep)

    # shuffle, then push repeats away from their twins
    order = rng.permutation(len(items))
    shuffled = [items[i] for i in order]
    final, taken = [], set()
    for i, it in enumerate(shuffled):
        if "repeat_of" in it:
            j = next(k for k, o in enumerate(shuffled) if o["id"] == it["repeat_of"])
            if abs(i - j) < 12:
                continue
        final.append(it)
        taken.add(it["id"])
    for it in shuffled:
        if it["id"] not in taken:
            final.append(it)

    counts = Counter(it["kind"] for it in final)
    doc = {
        "purpose": "Второе видео, другой маршрут. Замороженные нейроны проверяются на "
                   "сценах, которых не было при отборе. Кандидаты предложены "
                   "классификатором, метки ставятся глазами.",
        "source": str(WINDOWS),
        "video": "VID00002.AVI",
        "n": len(final), "counts": dict(counts),
        "n_turns_candidates": len(sel_turns), "n_controls": len(sel_ctrl),
        "n_repeats": len(rep_src),
        "repeat_ids": {it["id"]: it["repeat_of"] for it in final if "repeat_of" in it},
        "candidates": final,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    pos = {it["id"]: i for i, it in enumerate(final)}
    bad = sum(1 for it in final if "repeat_of" in it
              and abs(pos[it["id"]] - pos[it["repeat_of"]]) < 12)
    print(f"  повторов: {len(rep_src)}, слишком близко к близнецу: {bad}")
    print(f"  окно просмотра: {min(it['t0'] for it in final):.0f} .. "
          f"{max(it['t1'] for it in final):.0f} с")
    print(f"  состав: {dict(counts)}")
    print(f"\nWrote {OUT_JSON}")
    print(f"\n  запуск:  REVIEW_TAG=vid2 REVIEW_VIDEO=VID00002.AVI ./webapp/run.sh")


if __name__ == "__main__":
    main()
