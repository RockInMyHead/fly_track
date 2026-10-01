#!/usr/bin/env python3
"""FINAL TRACKER V1 — the smoke test on the five known camera recordings.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------
The spec is explicit: the five known clips are a check that the code works, not a set to tune on.
So this runs each clip from the start that the earlier chaining used, collects only structural
facts, and refuses to report anything that could be read as a quality score. There are no drawn
routes for these clips, so a quality score would be invented.

What it checks, per the spec:

    the code runs end to end
    no teleports
    STOP really stops
    every transition is a legal move on the graph
    the route's shape is a walk rather than a loop

That last one was added after the first run: with the wrong walking speed the route went round a
five-edge ring fifteen times, which is what the smoke test exists to catch. The check is reported
rather than acted on, because the spec forbids novelty in the routing.

Usage:
    PYTHONPATH=. python scripts/final_tracker_smoke.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/final_tracker"
CLIPS = ["VID00003", "VID00004", "VID00005", "VID00009", "VID00010"]


def start_of(clip: str) -> tuple[str, str] | None:
    p = ROOT / f"output/p13/runs/{clip}/run_start.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("edge") or not d.get("from"):
        return None
    return d["edge"], d["from"]


def main() -> int:
    print("=" * 104)
    print("FINAL TRACKER V1 — ПРОВЕРКА НА ПЯТИ ИЗВЕСТНЫХ КУСКАХ")
    print("=" * 104)
    print("  назначение: код работает, нет телепортаций, STOP останавливает, переходы законны,")
    print("              маршрут по форме прогулка, а не кольцо. Качество здесь НЕ измеряется:")
    print("              размеченных маршрутов для этих кусков нет, и число было бы выдумано.")
    print()

    rows = []
    for clip in CLIPS:
        st = start_of(clip)
        if st is None:
            print(f"  {clip}: нет старта в output/p13/runs/{clip}/run_start.json — пропуск")
            continue
        edge, frm = st
        cmd = [sys.executable, "scripts/final_tracker.py", "--video", clip,
               "--graph", "data/p08/graph.json", "--start-edge", edge, "--start-from", frm,
               "--quiet"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
        rep = OUT / clip / "report.json"
        if r.returncode != 0 or not rep.exists():
            print(f"  {clip}: КОД {r.returncode}")
            print((r.stdout or "")[-400:])
            print((r.stderr or "")[-400:])
            rows.append({"clip": clip, "ok": False, "returncode": r.returncode})
            continue
        d = json.loads(rep.read_text(encoding="utf-8"))
        s = d["stats"]
        rows.append({
            "clip": clip, "ok": True,
            "edges": s["edges_traversed"], "distinct": s["distinct_edges"],
            "meters": s["route_meters"], "decisions": s["decisions"],
            "status": s["decisions_by_status"],
            "stop_fraction": s["stop_fraction"],
            "teleports": s["self_checks"]["teleports"],
            "moved_while_stopped": s["self_checks"]["moved_while_stopped"],
            "illegal_edges": s["self_checks"]["illegal_edges"],
            "max_edge_repeats": s["loops"]["max_traversals_of_one_edge"],
            "distinct_over_total": s["loops"]["distinct_over_total"],
            "start": {"edge": edge, "from": frm},
        })

    print(f"  {'кусок':<10}{'рёбер':>7}{'разл.':>7}{'м':>8}{'развилок':>10}"
          f"{'телеп.':>8}{'STOP-движ':>11}{'незак.':>8}{'макс.повт':>11}{'разл/всего':>12}")
    print("  " + "-" * 94)
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['clip']:<10}  НЕ ПРОШЁЛ")
            continue
        print(f"  {r['clip']:<10}{r['edges']:>7}{r['distinct']:>7}{r['meters']:>8.0f}"
              f"{r['decisions']:>10}{r['teleports']:>8}{r['moved_while_stopped']:>11}"
              f"{r['illegal_edges']:>8}{r['max_edge_repeats']:>11}"
              f"{r['distinct_over_total']:>12.2f}")

    ok = [r for r in rows if r.get("ok")]
    print()
    bad = [r for r in ok if r["teleports"] or r["moved_while_stopped"] or r["illegal_edges"]]
    print(f"  кусков прошло: {len(ok)} из {len(CLIPS)}")
    if bad:
        print(f"  СТРУКТУРНЫЕ НАРУШЕНИЯ: {[r['clip'] for r in bad]}")
    else:
        print("  структурных нарушений нет ни на одном куске: телепортов 0, движений в STOP 0,")
        print("  незаконных переходов 0")
    loops = [r["clip"] for r in ok if r["distinct_over_total"] < 0.35]
    if loops:
        print()
        print(f"  ВНИМАНИЕ к форме маршрута: на {loops} различных рёбер меньше 35% от всех,")
        print(f"  то есть маршрут возвращается по пройденному. Это дефект маршрутизации,")
        print(f"  и он остаётся открытым.")
    else:
        print()
        print("  форма маршрута: на всех кусках различных рёбер больше 35% от пройденных")

    doc = {"phase": "FINAL TRACKER V1 — проверка на пяти кусках",
           "purpose": "структурная проверка кода; качество маршрута здесь не измеряется",
           "quality_not_measured_because": "размеченных маршрутов для этих кусков нет",
           "clips": rows,
           "structural_violations": [r["clip"] for r in bad],
           "route_shape_warning": loops}
    (OUT / "SMOKE.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  записано: {OUT/'SMOKE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
