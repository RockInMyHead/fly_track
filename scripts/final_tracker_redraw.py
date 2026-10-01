#!/usr/bin/env python3
"""Перерисовать trajectory.png из уже сохранённых CSV (без пересчёта трекера)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_tracker as FT  # noqa: E402
from p08_graph import Graph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="*", help="VID00010 …; по умолчанию все с trajectory.csv")
    a = ap.parse_args()
    base = ROOT / "output/final_tracker"
    clips = a.clip or sorted(d.name for d in base.iterdir()
                               if d.is_dir() and (d / "trajectory.csv").exists())
    g = Graph.load(ROOT / "data/p08/graph.json")
    n = 0
    for clip in clips:
        d = base / clip
        rows = list(csv.DictReader((d / "trajectory.csv").open(encoding="utf-8")))
        decs = list(csv.DictReader((d / "decisions.csv").open())) if (d / "decisions.csv").exists() else []
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        start = rep.get("start") or {}
        edge = start.get("edge") or rows[0]["edge"]
        frm = start.get("from") or g.edges[edge]["from"]
        # minimal best stub for draw title
        class B:
            history = [r["edge"] for r in rows[::max(1, len(rows) // 200)]]
        res = {"rows": rows, "decisions": decs, "best": B()}
        if FT.draw(d, g, res):
            print(f"  {clip}: trajectory.png")
            n += 1
        else:
            print(f"  {clip}: не нарисован")
    print(f"готово: {n}/{len(clips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
