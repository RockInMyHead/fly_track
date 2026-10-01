#!/usr/bin/env python3
"""P13.1 — freeze the stop-gate before the held-out chunks are looked at.

The rule was built on seven chunks and is now fixed: the indicator, the threshold, the logic and
the code that implements them. The nine remaining chunks — VID00007, VID00008 and VID00011 to
VID00017 — have not been measured yet, and the point of writing this file now is that they cannot
influence what they are about to test.

What is frozen:

    indicator   the frame-difference measure of scripts/p13_stillness.py, imported rather than
                reimplemented so that one function serves both the audit and the gate
    threshold   12, the value the audit used
    logic       video_motion < threshold -> progress speed 0; otherwise unchanged
    code        hashes of the gate, the audit and the tracker it calls

The second threshold, 14, is carried as a robustness check and is explicitly not available for
choosing between: if the result were good at one and bad at the other, that would be a finding to
report, not a setting to pick.

Usage:
    PYTHONPATH=. python scripts/p13_1_freeze.py
    PYTHONPATH=. python scripts/p13_1_freeze.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p13"
DEST = ROOT / "data/p13/FROZEN_STILLNESS_GATE.json"

CODE = (
    "scripts/p13_1_gate.py",             # the gate itself
    "scripts/p13_stillness.py",          # the indicator it imports
    "scripts/p08_graph_tracker.py",      # track(), load_signals(), relative_speed()
    "scripts/p08_graph.py",              # the graph
    "scripts/p084b_geometry.py",         # the decision rule called inside track()
    "data/p08/graph.json",
    "output/p13/CHAIN_ANCHOR.json",      # the order and the start of the walk
)

DEVELOPED_ON = ["VID00002", "VID00003", "VID00004", "VID00005",
                "VID00006", "VID00009", "VID00010"]
HELD_OUT = ["VID00007", "VID00008", "VID00011", "VID00012", "VID00013",
            "VID00014", "VID00015", "VID00016", "VID00017"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="только проверить, что код не менялся после заморозки")
    a = ap.parse_args()

    if a.verify:
        if not DEST.exists():
            print(f"нет файла заморозки {DEST}")
            return 1
        d = json.loads(DEST.read_text(encoding="utf-8"))
        bad = []
        for rel, m in d["manifest"].items():
            p = ROOT / rel
            if not p.exists():
                bad.append(f"{rel} — пропал")
            elif sha256(p) != m["sha256"]:
                bad.append(f"{rel} — изменён (было {m['sha256'][:16]})")
        if bad:
            print("ЗАМОРОЗКА НАРУШЕНА:")
            for b in bad:
                print(f"  {b}")
            return 1
        print(f"заморозка цела: {len(d['manifest'])} файлов совпадают")
        print(f"  порог {d['threshold']}, заморожено {d['frozen_at']}")
        return 0

    manifest = {}
    for rel in CODE:
        p = ROOT / rel
        if not p.exists():
            print(f"  СТОП: нет файла {rel}")
            return 1
        manifest[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    doc = {
        "phase": "P13.1 — STOP-гейт",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "why_frozen_now": (
            "правило построено на семи кусках и зафиксировано до измерения девяти остальных; "
            "иначе проверочные куски повлияли бы на то, что их же проверяет"
        ),
        "indicator": {
            "what": "кадровая разница: среднее абсолютное различие между соседними кадрами "
                    "в градациях серого",
            "how": "ffmpeg -i <mp4> -vf fps=2,scale=200:-2 -q:v 5 <кадры>, затем MAD "
                   "соседних кадров; 2 Гц, ширина 200 px",
            "implemented_in": "scripts/p13_stillness.py, функция frame_motion",
            "imported_not_reimplemented": (
                "гейт вызывает эту же функцию, а не свою копию: две реализации, совпадающие "
                "сегодня, разойдутся завтра"
            ),
        },
        "threshold": 12.0,
        "robustness_threshold": 14.0,
        "threshold_note": (
            "12 — значение аудита. 14 сообщается как проверка устойчивости и НЕ может быть "
            "использовано для выбора: если результат хорош при одном и плох при другом, это "
            "находка для отчёта, а не настройка"
        ),
        "logic": {
            "if_video_motion_below_threshold": "progress_speed = 0",
            "else": "progress_speed = текущий speed без изменений",
            "not_used": "net_displacement в гейте не участвует",
            "why_not_net_displacement": (
                "его класс NO_NET_DISPLACEMENT включает человека, который реально шёл и вернулся; "
                "гейт по нему заморозил бы движение во время настоящей ходьбы"
            ),
        },
        "not_claimed": (
            "P13.1 исправляет «стою, а трекер идёт». Он НЕ исправляет «иду медленно или быстро — "
            "сколько метров в секунду»: величина скорости движения по-прежнему не измерена"
        ),
        "does_not_touch": {
            "P12": "слепой результат VID00010 не пересчитывается задним числом; он остаётся "
                   "историческим результатом замороженной системы",
            "tracker_code": "p08_graph_tracker не изменён: гейт подставляет другой массив "
                            "скорости в ту же функцию track()",
        },
        "developed_on": DEVELOPED_ON,
        "held_out": HELD_OUT,
        "developed_on_result": {
            "chunks": 7,
            "false_metres_in_stop_before": 410,
            "false_metres_in_stop_after": 9,
            "removed_fraction": 0.98,
            "robustness_after_at_14": 11,
            "windows_check_VID00002": {
                "window_s": [840, 1020],
                "still_fraction": 0.70,
                "before_m": 84,
                "before_m_while_still": 66,
                "after_m": 19,
                "after_m_while_still": 0,
                "reading": "66 м ложного продвижения в окне убрано полностью; оставшиеся 19 м "
                           "приходятся на те 30% окна, когда человек действительно шёл",
            },
        },
        "held_out_rule": (
            "после просмотра результата на проверочных кусках ничего не менять и не "
            "перепрогонять с другими параметрами; первый результат — основной"
        ),
        "manifest": manifest,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 92)
    print("P13.1 — ГЕЙТ ЗАМОРОЖЕН")
    print("=" * 92)
    print(f"  признак:  {doc['indicator']['how']}")
    print(f"  порог:    {doc['threshold']} (устойчивость проверяется при "
          f"{doc['robustness_threshold']})")
    print(f"  логика:   {doc['logic']['if_video_motion_below_threshold']} / "
          f"{doc['logic']['else']}")
    print(f"  на семи:  ложных метров {doc['developed_on_result']['false_metres_in_stop_before']}"
          f" → {doc['developed_on_result']['false_metres_in_stop_after']}")
    print()
    print(f"  проверочные, ещё не измерены: {', '.join(HELD_OUT)}")
    print()
    for rel, m in manifest.items():
        print(f"  {m['sha256'][:16]}  {m['bytes']:>9}  {rel}")
    print()
    print(f"  записано: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
