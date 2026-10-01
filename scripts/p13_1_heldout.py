#!/usr/bin/env python3
"""P13.1 — the held-out test, run once, after the gate was frozen.

Nine chunks were set aside before the rule was written down: VID00007, VID00008 and VID00011 to
VID00017. They are the test, and the rule that governs them is the one already fixed in
data/p13/FROZEN_STILLNESS_GATE.json. This script exists so that the test cannot be quietly repeated
with different settings:

    it refuses to run      if any of the frozen files has changed
    it waits               for each chunk's video to exist rather than skipping it, because
                           skipping would silently test fewer than nine chunks
    it runs once           a lock file is written on completion; a second run is refused, so the
                           first result is the result

It does not modify the rule, and it reports rather than concludes: the numbers it prints are the
ones to read, not a verdict about whether the phase succeeded.

Usage:
    PYTHONPATH=. python scripts/p13_1_heldout.py
    PYTHONPATH=. python scripts/p13_1_heldout.py --no-wait     # test only what is ready now
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p08_graph as G  # noqa: E402
from scripts.p13_1_freeze import DEST as FREEZE, sha256  # noqa: E402

OUT = ROOT / "output/p13"
MEDIA = ROOT / "webapp/media"
LOCK = OUT / "P13_1_HELDOUT_RESULT.json"


def verify_freeze() -> tuple[bool, list[str]]:
    if not FREEZE.exists():
        return False, [f"нет файла заморозки {FREEZE}"]
    d = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = []
    for rel, m in d["manifest"].items():
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel} — пропал")
        elif sha256(p) != m["sha256"]:
            bad.append(f"{rel} — изменён")
    return (not bad), bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--wait-min", type=float, default=240.0)
    ap.add_argument("--poll-s", type=float, default=60.0)
    a = ap.parse_args()

    print("=" * 100)
    print("P13.1 — ПРОВЕРКА НА ОТЛОЖЕННЫХ КУСКАХ")
    print("=" * 100)

    ok, bad = verify_freeze()
    if not ok:
        print("  ЗАМОРОЗКА НАРУШЕНА, прогон запрещён:")
        for b in bad:
            print(f"    {b}")
        return 1
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    held = frozen["held_out"]
    print(f"  заморозка цела, порог {frozen['threshold']}")
    print(f"  отложенных кусков: {len(held)}")

    if LOCK.exists():
        print()
        print(f"  УЖЕ ПРОГНАНО: {LOCK}")
        print("  Первый результат — основной результат проверки. Повтор не выполняется.")
        print("  Если он всё же нужен, удалите файл сами, понимая, что это уже не та проверка.")
        return 1

    todo = [v for v in held if (MEDIA / f"{v}_fixed.mp4").exists()]
    if not todo:
        if a.no_wait:
            print("  ни одного готового видео, а --no-wait задан: нечего проверять")
            return 1
        print()
        print("  видео ещё не перекодированы, ждём. Это ожидание, а не ошибка:")
        print(f"    {', '.join(v for v in held if v not in todo)}")
        print()

    t0 = time.time()
    while len(todo) < len(held):
        missing = [v for v in held if v not in todo]
        if a.no_wait:
            break
        if time.time() - t0 > a.wait_min * 60:
            print(f"  ждём уже {a.wait_min:.0f} мин, остаются {missing}")
            break
        time.sleep(a.poll_s)
        todo = [v for v in held if (MEDIA / f"{v}_fixed.mp4").exists()]

    print(f"  готовы к проверке: {len(todo)} из {len(held)}: {', '.join(todo)}")
    still = [v for v in held if v not in todo]
    if still:
        print(f"  ещё не готовы (в результат не войдут): {', '.join(still)}")
    print()

    # the one run, using the gate exactly as frozen
    import scripts.p13_1_gate as gate

    g = G.Graph.load(ROOT / "data/p08/graph.json")
    rows = []
    print(f"  {'кусок':<10}{'стоп,%':>7}{'старый':>8}{'новый':>8}{'убрано':>8}"
          f"{'разв.в стоп':>12}{'разв.всего':>11}{'сдвиг t':>9}")
    print("  " + "-" * 80)
    for v in todo:
        r = gate.analyse(v, g, float(frozen["threshold"]))
        if r is None:
            print(f"  {v}: не посчиталось")
            continue
        rows.append(r)
        sh = r["time_shift_median_s"]
        print(f"  {v:<10}{100*r['stop_fraction']:>6.0f}%{r['old_m_in_stop']:>8.0f}"
              f"{r['new_m_in_stop']:>8.0f}{r['false_m_removed']:>8.0f}"
              f"{str(r['old_decisions_in_stop'])+'→'+str(r['new_decisions_in_stop']):>12}"
              f"{str(r['old_decisions'])+'→'+str(r['new_decisions']):>11}"
              f"{(f'{sh:+.0f}с' if sh is not None else '—'):>9}")
    print()

    # the same run at the robustness threshold, reported and not chosen between
    rob = []
    for v in todo:
        r = gate.analyse(v, g, float(frozen["robustness_threshold"]))
        if r:
            rob.append(r)

    o = sum(r["old_m_in_stop"] for r in rows)
    n = sum(r["new_m_in_stop"] for r in rows)
    nr = sum(r["new_m_in_stop"] for r in rob)
    print(f"  ИТОГО на отложенных: ложных метров было {o:.0f}, осталось {n:.0f}, "
          f"убрано {o-n:.0f} ({100*(o-n)/max(o,1):.0f}%)")
    print(f"  при пороге {frozen['robustness_threshold']} осталось бы {nr:.0f} м")
    print()

    dev = frozen["developed_on_result"]
    print("  СРАВНЕНИЕ С КУСКАМИ, НА КОТОРЫХ ПРАВИЛО СТРОИЛОСЬ:")
    print(f"    на семи:      {dev['false_metres_in_stop_before']} → "
          f"{dev['false_metres_in_stop_after']} м "
          f"({100*(1-dev['false_metres_in_stop_after']/dev['false_metres_in_stop_before']):.0f}% убрано)")
    if o:
        print(f"    на девяти:    {o:.0f} → {n:.0f} м ({100*(o-n)/o:.0f}% убрано)")
        print("    Столбцы читаются как есть. Совпадение или расхождение — результат, а не")
        print("    повод что-то подправить.")
    print()

    doc = {
        "phase": "P13.1 — проверка на отложенных кусках",
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_at": frozen["frozen_at"], "threshold": frozen["threshold"],
        "robustness_threshold": frozen["robustness_threshold"],
        "freeze_verified": True,
        "nothing_fitted_here": True,
        "n_tested": len(rows), "n_held_out": len(held),
        "not_ready": still,
        "totals": {"false_m_before": o, "false_m_after": n,
                   "removed_fraction": (o - n) / o if o else None,
                   "robustness_after": nr},
        "developed_on": {"false_m_before": dev["false_metres_in_stop_before"],
                         "false_m_after": dev["false_metres_in_stop_after"]},
        "chunks": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "rule": frozen["held_out_rule"],
    }
    LOCK.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")
    print(f"  записано: {LOCK}")
    print()
    print("  После просмотра этого результата ничего не менять и не перепрогонять.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
