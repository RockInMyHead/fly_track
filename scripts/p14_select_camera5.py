#!/usr/bin/env python3
"""P14.1 — choose two more camera recordings by provenance alone, before any labelling.

WHY A RULE RATHER THAN A CHOICE
-------------------------------
The set is about to grow from three recordings to five, and the result of that growth is the thing
under test. If the two additions were picked by hand, a sceptic could say they were picked because
of what they contained. So the rule is written down first and then applied mechanically, and the
rule uses only properties of the files and of what has already been computed from them — never
whether a clip is interesting, balanced, or easy.

THE RULE
--------
A recording is eligible when all of these hold, and each is a provenance property:

    the file is the camera's own, and its real frame count was measured, not declared
    it is a full-length chunk (at least 30,000 real frames), so it is comparable to the rest
    no trace of it in the pipeline was produced from a non-camera file
    it is not already part of the camera set

Then: take the two eligible recordings with the smallest identifier. Ordering by identifier is a
proxy for recording time and is fixed in advance; nothing about the content enters.

Usage:
    PYTHONPATH=. python scripts/p14_select_camera5.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/p14/MANIFEST.json"
ALREADY = ("VID00005", "VID00009", "VID00010")
MIN_REAL_FRAMES = 30000
TAKE = 2


def pipeline_traces(clip: str) -> dict:
    return {
        "P09_dn": (ROOT / f"output/p09/trace_{clip}.npz").exists(),
        "P10_input_early": (ROOT / f"output/p10/brain_{clip}.npz").exists(),
        "P11B_cells": (ROOT / f"output/p11b/cells_{clip}.npz").exists(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data/p14/SELECTION_CAMERA5.json"))
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contaminated = {"VID00001", "VID00002", "VID00006"}

    print("=" * 104)
    print("P14.1 — ВЫБОР ДВУХ КУСКОВ ПО ПРОИСХОЖДЕНИЮ (сделано до разметки)")
    print("=" * 104)
    print(f"  правило: камерный файл, реальные кадры посчитаны, >= {MIN_REAL_FRAMES} кадров,")
    print(f"           следы в конвейере только с камеры, не входят в текущий набор")
    print(f"  затем: два наименьших идентификатора среди прошедших")
    print()
    print(f"  {'кусок':<10}{'кадров':>8}{'camera':>8}{'3 следа':>9}{'в наборе':>10}"
          f"{'чистота':>10}   вердикт")
    print("  " + "-" * 96)

    rows = []
    for vid, rec in sorted(man["videos"].items()):
        cam = rec.get("camera") or {}
        real = cam.get("real_frames")
        tr = pipeline_traces(vid)
        n_tr = sum(tr.values())
        in_set = vid in ALREADY
        dirty = vid in contaminated
        reasons = []
        if not cam:
            reasons.append("нет камерного файла")
        if real is None:
            reasons.append("кадры не посчитаны")
        elif real < MIN_REAL_FRAMES:
            reasons.append(f"короткий ({real})")
        if dirty:
            reasons.append("следы в конвейере не с камеры")
        if in_set:
            reasons.append("уже в наборе")
        ok = not reasons
        rows.append({"video": vid, "real_frames": real, "traces_present": tr,
                     "already": in_set, "contaminated": dirty, "eligible": ok,
                     "rejected_because": reasons})
        verdict = "ГОДЕН" if ok else "; ".join(reasons)
        print(f"  {vid:<10}{str(real):>8}{'да' if cam else 'нет':>8}{n_tr:>9}"
              f"{'да' if in_set else 'нет':>10}{'чистый' if not dirty else 'ГРЯЗНЫЙ':>10}   {verdict}")

    eligible = [r for r in rows if r["eligible"]]
    eligible.sort(key=lambda r: r["video"])
    chosen = eligible[:TAKE]
    print()
    print(f"  годных всего: {len(eligible)} — {[r['video'] for r in eligible]}")
    print(f"  ВЫБРАНО (два наименьших идентификатора): {[r['video'] for r in chosen]}")
    print()
    if len(chosen) < TAKE:
        print(f"  СТОП: годных меньше {TAKE}")
        return 1

    doc = {
        "phase": "P14.1 — CAMERA5, выбор кусков",
        "when": "сделано ДО получения новых человеческих меток",
        "rule": {
            "camera_file": "файл камеры, реальное число кадров измерено (не объявлено)",
            "full_length": f"не меньше {MIN_REAL_FRAMES} реальных кадров",
            "no_mixed_traces": "ни один след этого куска в конвейере не получен из не-камерного файла",
            "not_in_set": f"не входит в текущий камерный набор {list(ALREADY)}",
            "then": f"взять {TAKE} годных с наименьшими идентификаторами",
            "what_the_rule_ignores": [
                "есть ли в куске повороты",
                "сбалансированы ли категории",
                "хорошо ли работает мозг на этом куске",
                "любые метрики и AUC",
            ],
        },
        "contaminated_clips_excluded": {
            "VID00001": "следы из ~/Downloads",
            "VID00002": "следы в конвейере (P09/P10) сняты с файла ~/Downloads, а не с камеры",
            "VID00006": "уровни P09/P10 покрывают 250 с старого клипа",
        },
        "chosen": [r["video"] for r in chosen],
        "eligible": [r["video"] for r in eligible],
        "per_clip": rows,
        "camera_set_after": list(ALREADY) + [r["video"] for r in chosen],
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
