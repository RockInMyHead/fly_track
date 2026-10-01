#!/usr/bin/env python3
"""
P07.3 — labelling bank for VID00009.

Two things have to be recorded by hand for this clip, and they come from two tabs:

    the route shape   drawn on the trajectory tab: place a point as the clip plays, so
                      the canvas ends up holding the footprint of the walk
    the turns         marked on the turns tab: whether each proposed window really is a
                      turn, and if so which way

The proposed windows come with a caveat that has to be stated rather than hidden. The
classifier's rotation threshold is absolute, `TURN_MIN = 1.0` in units of the uniform
component, and this clip's motion is about three times larger than the first clip's:

    clip        median |a|    share of frames above 1.0    candidates
    VID00001        0.545                     35.8 percent          70
    VID00009        1.632                     62.3 percent         242

So the 242 proposals are not 242 turns. They are what an absolute threshold returns when
the scale changes, and a large share of them will be rejected. That is fine and even
useful, since rejecting them measures how much the threshold over-fires, but it means the
bank must be sampled rather than taken whole, and the caveat must travel with the data.

Usage:
    PYTHONPATH=. python scripts/p073_build_bank.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "data/p06/windows_vid9.json"
OUT_JSON = ROOT / "data/p01r/review_set_vid9.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-turns", type=int, default=48)
    ap.add_argument("--n-controls", type=int, default=24)
    ap.add_argument("--n-repeats", type=int, default=6)
    ap.add_argument("--seed", type=int, default=909)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ws = json.loads(WINDOWS.read_text(encoding="utf-8"))["windows"]
    turns = sorted([w for w in ws if w["kind"] in ("LEFT", "RIGHT")],
                   key=lambda z: z["t0"])
    ctrl = sorted([w for w in ws if w["kind"] in ("FWD", "STATIC")],
                  key=lambda z: z["t0"])

    def spread(items: list[dict], k: int) -> list[dict]:
        """Evenly spaced pick, so the whole clip is covered rather than one stretch."""
        if len(items) <= k:
            return items
        idx = np.linspace(0, len(items) - 1, k).round().astype(int)
        return [items[i] for i in idx]

    # keep the left/right balance of the proposals, since the clip's own mix is what the
    # sampler should preserve
    lefts = [w for w in turns if w["kind"] == "LEFT"]
    rights = [w for w in turns if w["kind"] == "RIGHT"]
    share = len(lefts) / max(len(turns), 1)
    n_left = int(round(args.n_turns * share))
    sel_turns = spread(lefts, n_left) + spread(rights, args.n_turns - n_left)
    sel_turns.sort(key=lambda z: z["t0"])
    sel_ctrl = spread(ctrl, args.n_controls)

    print("P07.3 — банк разметки для VID00009")
    print(f"  предложено классификатором: {len(turns)} поворотов "
          f"({dict(Counter(w['kind'] for w in turns))}), {len(ctrl)} FWD/STATIC")
    print(f"  ОГОВОРКА: порог поворота абсолютный, у этого ролика движение втрое "
          f"больше,")
    print(f"  поэтому предложения перегружены. Банк берёт выборку, а не всё.")
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

    items = [make(x, f"n{i:03d}") for i, x in enumerate(sel_turns)]
    items += [make(x, f"m{j:03d}") for j, x in enumerate(sel_ctrl)]
    for k, x in enumerate(rep_src):
        rep = make(x, f"r{k:03d}")
        src = next(it["id"] for it in items
                   if abs(it["t0"] - x["t0"]) < 1e-6 and it["kind"] == x["kind"])
        rep["repeat_of"] = src
        items.append(rep)

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
        "purpose": "VID00009. Отдельный ролик, тот же цех. Нужен для проверки формы "
                   "маршрута: разметка поворотов здесь, форма маршрута — на вкладке "
                   "траектории.",
        "source": str(WINDOWS),
        "video": "VID00009.AVI",
        "caveat": "порог TURN_MIN абсолютный; при движении втрое больше первого ролика "
                  "предложений втрое больше, и большая их часть — ложные. Ожидаемо "
                  "отклонять много.",
        "n": len(final), "counts": dict(counts),
        "n_proposed_turns": len(turns), "n_sampled_turns": len(sel_turns),
        "n_controls": len(sel_ctrl), "n_repeats": len(rep_src),
        "repeat_ids": {it["id"]: it["repeat_of"] for it in final if "repeat_of" in it},
        "candidates": final,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    pos = {it["id"]: i for i, it in enumerate(final)}
    bad = sum(1 for it in final if "repeat_of" in it
              and abs(pos[it["id"]] - pos[it["repeat_of"]]) < 12)
    print(f"  повторов: {len(rep_src)}, слишком близко к близнецу: {bad}")
    print(f"  окно просмотра: {min(it['t0'] for it in final):.0f} .. "
          f"{max(it['t1'] for it in final):.0f} с")
    print(f"  состав: {dict(counts)}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
