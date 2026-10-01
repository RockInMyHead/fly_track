#!/usr/bin/env python3
"""P12.4 — the blind run on a recording that was not used anywhere before.

The order of operations is the whole point of this file, and it is enforced rather than described:

    1. load the recording, do not look at it
    2. walk both routes, save both results
    3. only then may a human annotate anything

So the script does not annotate, does not plot, and does not print a verdict. It walks, it writes
what it walked to a file, and it locks itself. A second invocation refuses to run: the first result
is the result, and a run made after the footage has been seen would not be a test of anything.

WHAT IT REFUSES TO DO
---------------------
    run at all          if any file in data/p12/FROZEN_BEFORE_VID00010.json has changed. The
                        message names the file, so a post-hoc edit cannot pass unnoticed.
    run twice           a lock file is written on success. Removing it is possible but has to be
                        deliberate, and the freeze file records that it was there.
    pick parameters     tau, the readers, the graph, the walk and the threshold are read from the
                        freeze. Nothing is fitted here.

WHAT IT NEEDS FOR A NEW RECORDING
---------------------------------
The pipeline in front of the readers has to have been run on the new video first, and this script
says which pieces are missing and which script makes each one:

    output/p10/brain_{V}.npz      the twenty early populations      scripts/p10_record.py
    output/p09/trace_{V}.npz      the sixty-eight descending cells  scripts/p09_real_holdout.py
    output/p07/yaw_signal_{V}.csv the tracker's own yaw             scripts/p07_trajectory.py
    output/p07/forward_signal_{V}.csv
    output/p071/trajectory_{V}.csv
    output/p084c/run/             the tracker's own decisions       scripts/p08_graph_tracker.py

Usage:
    PYTHONPATH=. python scripts/p12_blind.py --video VID00010
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FREEZE = ROOT / "data/p12/FROZEN_BEFORE_VID00010.json"

# The chain that turns a raw recording into the inputs the readers need, in the order it has to be
# run. Two of these steps are easy to get wrong and both were wrong in the first version of this
# file: `output/p071/trajectory_*.csv` is written by p072_trajectory.py and not by p07_trajectory.py,
# and that script needs output/p072/events_*.csv from the event detector first. A recipe that names
# the wrong script fails loudly; the one below also names the registration step, which fails
# quietly, so it is the one worth stating.
CHAIN = (
    ("data/p01r/{v}.AVI", "источник: положить ролик сюда"),
    ("data/p07_videos.json", "зарегистрировать ролик: scripts/p07_videos.py --add {v} "
                             "--trace … --targets … (нужен трейс в стиле P06)"),
    ("output/p07/yaw_signal_{v}.csv", "scripts/p07_trajectory.py — yaw и forward"),
    ("output/p07/forward_signal_{v}.csv", "scripts/p07_trajectory.py"),
    ("output/p072/events_{v}.csv", "scripts/p072_event_detector.py — события поворотов"),
    ("output/p071/trajectory_{v}.csv", "scripts/p072_trajectory.py — траектория"),
    ("output/p09/trace_{v}.npz", "scripts/p09_real_holdout.py --video {v} --record — 68 клеток"),
    ("output/p10/brain_{v}.npz", "scripts/p10_record.py --video {v} — вход и ранние популяции"),
    ("<run>/{decisions,edge_sequence}.csv", "scripts/p08_graph_tracker.py --video {v} --out <run>; "
                                            "каталог ОБЯЗАТЕЛЬНО свой, не output/p084c/run"),
)

# only the ones that must exist for the blind run to be possible at all
NEEDS = {
    "output/p10/brain_{v}.npz": "scripts/p10_record.py — вход и ранние зрительные популяции",
    "output/p09/trace_{v}.npz": "scripts/p09_real_holdout.py --record — 68 нисходящих клеток",
    "output/p07/yaw_signal_{v}.csv": "scripts/p07_trajectory.py — yaw трекера",
    "output/p07/forward_signal_{v}.csv": "scripts/p07_trajectory.py — forward трекера",
    "output/p071/trajectory_{v}.csv": "scripts/p072_trajectory.py — траектория "
                                      "(после p072_event_detector.py)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(freeze: dict) -> list[str]:
    """Every file in the manifest must still hash the same. Returns the changed ones."""
    bad = []
    for rel, m in freeze["manifest"].items():
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel} — ПРОПАЛ")
        elif sha256(p) != m["sha256"]:
            bad.append(f"{rel} — ИЗМЕНЁН (было {m['sha256'][:16]})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="VID00010")
    ap.add_argument("--run", default=None,
                    help="каталог прогона трекера ДЛЯ ЭТОГО ролика: decisions.csv, "
                         "edge_sequence.csv, report.json. Обязателен и без значения по "
                         "умолчанию: раньше здесь стоял output/p084c/run, то есть прогон "
                         "VID00006, и слепой прогон молча взял бы чужие решения")
    ap.add_argument("--out", default=str(ROOT / "output/p12"))
    ap.add_argument("--check-only", action="store_true",
                    help="только проверить заморозку и наличие входов, ничего не запускать")
    args = ap.parse_args()
    v = args.video
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"P12.4 — СЛЕПОЙ ПРОГОН НА {v}")
    print("=" * 100)
    if not FREEZE.exists():
        print(f"  СТОП: нет файла заморозки {FREEZE}")
        return 1
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))

    print(f"  заморожено: {freeze['frozen_at']}, tau={freeze['tau']}, "
          f"min_cons={freeze['min_cons']}")
    print()

    # ---- 1. the freeze must hold ----
    bad = verify(freeze)
    print("─── ПРОВЕРКА ЗАМОРОЗКИ ───")
    if bad:
        print(f"  НАРУШЕНА: {len(bad)} файлов изменились после заморозки")
        for b in bad:
            print(f"    {b}")
        print()
        print("  Прогон запрещён. Первый результат должен быть получен на неизменённой цепочке;")
        print("  изменение после заморозки делает сравнение с P08.6 бессмысленным.")
        return 1
    print(f"  {len(freeze['manifest'])} файлов совпадают с заморозкой")
    print()

    # ---- 2. is a run already recorded? ----
    lock = out / f"BLIND_{v}_RESULT.json"
    if lock.exists() and not args.check_only:
        print("─── УЖЕ ПРОГНАНО ───")
        print(f"  {lock} существует. Первый результат — основной результат проверки.")
        print("  Повторный прогон не выполняется. Если он всё же нужен, удалите файл сами,")
        print("  понимая, что после просмотра ролика это уже не слепая проверка.")
        return 1
    print()

    # ---- 3. are the inputs there? ----
    print("─── ВХОДЫ ДЛЯ НОВОГО РОЛИКА ───")
    missing = []
    for tmpl, how in NEEDS.items():
        rel = tmpl.format(v=v)
        p = ROOT / rel
        ok = p.exists()
        print(f"  {rel:<44}{'есть' if ok else 'НЕТ'}")
        if not ok:
            missing.append((rel, how))

    print()
    if missing:
        print("  СТОП: сначала нужно прогнать цепочку на новом ролике, ничего не размечая.")
        print("  Цепочка целиком, по порядку:")
        for tmpl, how in CHAIN:
            print(f"    {tmpl.format(v=v) if '{v}' in tmpl else tmpl}")
            print(f"      {how}")
        print()
        print("  Сейчас не хватает:")
        for rel, how in missing:
            print(f"    {rel}")
            print(f"      {how}")
        print()
        print("  После появления всех входов запустите этот скрипт снова, указав СВОЙ каталог "
              "прогона:")
        print(f"    PYTHONPATH=. python scripts/p12_blind.py --video {v} "
              f"--run output/p12/blind_run_{v}")
        print("  Он ничего не обучaет: читатели, tau и граф берутся из заморозки.")
        return 1
    print("  все входы на месте")
    print()

    if args.check_only:
        print("  --check-only: заморозка цела, входы на месте, прогон не выполнен")
        return 0

    # ---- 4. the blind walk ----
    import p084b_geometry as geo
    import p08_graph as G
    import p086_multi as M
    import scripts.p12_channels as P
    import scripts.p12_run as R

    tau = float(freeze["tau"])
    g = G.Graph.load(ROOT / "data/p08/graph.json")
    mpp = float(g.meters_per_pixel)

    # readers come from the freeze, not from a fresh fit: pasted into the same shape fit_channel
    # returns, so the same scoring code applies
    def reader(spec: dict) -> dict:
        return {"ok": True, "k": spec["k"], "n_groups": spec["n_groups"], "nf": spec["nf"],
                "cell_idx": spec["cell_idx"], "cells": spec["cells"],
                "feat_idx": spec["feat_idx"], "feat": spec["feat"], "sign": spec["sign"],
                "w": spec["w"], "b": spec["b"], "cons": [1.0] * len(spec["cell_idx"]),
                "dev": [0.0] * len(spec["cell_idx"])}

    fold = {k: reader(x) for k, x in freeze["readers_for_new_recording"].items()
            if k != "classes"}

    # the tracker's own decisions on the new video must already exist, in their OWN directory.
    # This used to have a default pointing at output/p084c/run, which is the VID00006 walk; a blind
    # run for VID00010 would then have read VID00006's decisions and reported them as VID00010's.
    # There is no default now, and the directory is checked to belong to the requested recording.
    if not args.run:
        print("  СТОП: не указан --run. Нужен каталог прогона трекера ИМЕННО для этого ролика:")
        print(f"    PYTHONPATH=. python scripts/p08_graph_tracker.py --video {v} "
              f"--out output/p12/blind_run_{v}")
        print("  Каталог output/p084c/run содержит прогон VID00006 и для слепой проверки "
              "не годится.")
        return 1
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    dec_path = run_dir / "decisions.csv"
    if not dec_path.exists():
        print(f"  СТОП: нет {dec_path} — прогон трекера на новом ролике не сделан")
        print(f"    PYTHONPATH=. python scripts/p08_graph_tracker.py --video {v} --out {args.run}")
        return 1
    rep_path = run_dir / "report.json"
    if not rep_path.exists():
        print(f"  СТОП: нет {rep_path} — каталог прогона неполный")
        return 1
    # report.json records which recording the walk belongs to, which is the only reliable way to
    # tell one run directory from another: decisions.csv carries no video column, so a check based
    # on it would silently pass on the wrong directory.
    rep_video = json.loads(rep_path.read_text(encoding="utf-8")).get("video")
    if rep_video != v:
        print(f"  СТОП: {rep_path} описывает прогон {rep_video}, а не {v}.")
        print("  Именно от этого защищает обязательный --run без значения по умолчанию:")
        print("  прогон VID00006 лежит в output/p084c/run и для слепой проверки не годится.")
        print(f"    PYTHONPATH=. python scripts/p08_graph_tracker.py --video {v} "
              f"--out output/p12/blind_run_{v}")
        return 1

    yaw_csv = ROOT / f"output/p07/yaw_signal_{v}.csv"
    traj_csv = ROOT / f"output/p071/trajectory_{v}.csv"
    rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    pace = float(rep["params"]["pix_per_sec"])
    sig = M.Signal(yaw_csv, traj_csv, pace)
    decisions = list(csv.DictReader(dec_path.open(encoding="utf-8")))

    print("─── СЛЕПОЙ ПРОГОН: P08.6 И P12 ───")
    print(f"  точек решения {sum(1 for d in decisions if d['reason'] != 'ONLY_OPTION')}")
    print()

    rows = []
    for d in decisions:
        node, arrival, t = d["node"], d["current_edge"], float(d["time"])
        if d["reason"] == "ONLY_OPTION" or arrival not in g.edges:
            continue
        ways = [c for c in g.classify_candidates(node, arrival, allow_back=True)]
        forward = [c for c in ways if not geo.is_reversal(c["deg"])]
        if len(forward) < 2:
            continue
        c = R.read_at(v, fold, t, tau)
        r = {"t": t, "node": node, "arrival": arrival, "committed_p086": d["chosen_edge"],
             "yaw": c["yaw"], "route_change": c["route_change"],
             "net_displacement": c["net_displacement"],
             "forward_edges": [{"edge": x["edge"], "deg": float(x["deg"]),
                                "to": x["to"]} for x in forward]}
        rows.append(r)

    # P12's picking rule, identical to p12_run's, applied to the frozen readers
    for r in rows:
        fwd = r["forward_edges"]
        pick = r["committed_p086"]
        notes = []
        rc, nd, yaw = r["route_change"], r["net_displacement"], r["yaw"]
        if rc.get("class") == "SAME" and fwd:
            straight = min(fwd, key=lambda x: abs(x["deg"]))
            if pick != straight["edge"]:
                notes.append("route_change=SAME: повёрнуто на прямое")
            pick = straight["edge"]
        elif rc.get("class") == "DIFFERENT" and yaw.get("class"):
            same = [x for x in fwd if R.side_of(x["deg"]) == yaw["class"]]
            if same:
                best = max(same, key=lambda x: abs(x["deg"]))
                if pick != best["edge"]:
                    notes.append("route_change=DIFFERENT + yaw: выбрана названная сторона")
                pick = best["edge"]
        if nd.get("class") == "NO_NET":
            notes.append("net=NO_NET: продвижение остановлено")
        r["pick_p12"] = pick
        r["changed"] = pick != r["committed_p086"]
        r["notes"] = notes

    print(f"  {'t':>8}{'узел':<8}{'P08.6':<16}{'P12':<16}{'yaw':<7}{'route_change':<14}"
          f"{'net':<10}изменилось")
    for r in rows:
        mark = "ДА" if r["changed"] else ""
        print(f"  {r['t']:>8.1f}{r['node']:<8}{r['committed_p086']:<16}{r['pick_p12']:<16}"
              f"{r['yaw'].get('class') or '—':<7}"
              f"{r['route_change'].get('class') or 'UNKNOWN':<14}"
              f"{r['net_displacement'].get('class') or 'UNKNOWN':<10}{mark}")
    print()

    doc = {
        "phase": f"P12.4 — слепой прогон на {v}",
        "video": v,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_at": freeze["frozen_at"],
        "tau": tau,
        "freeze_verified": True,
        "nothing_fitted_here": True,
        "n_decisions": len(rows),
        "changed_by_p12": sum(1 for r in rows if r["changed"]),
        "human_annotation_must_happen_after_this": True,
        "annotation_instructions": freeze["rules"]["after_the_video_appears"][3],
        "decisions": rows,
        "ledger_to_fill_after_annotation": {
            "правильных развилок P08.6": None, "правильных развилок P12": None,
            "ошибочных развилок": None, "неразрешённых развилок": None,
            "ложных заходов в тупик": None, "ложных разворотов": None,
            "ошибок camera_yaw": None, "ошибок route_change": None,
            "ошибок net_displacement": None,
            "P08.6 correct": None, "P12 correct": None,
        },
    }
    lock.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")
    print(f"  записано: {lock}")
    print()
    print("  Ролик НЕ размечался и не показывался. Результат зафиксирован до разметки.")
    print("  Следующий шаг по порядку: человеческая разметка развилок VID00010, и только потом")
    print("  заполнение летописи. После этого ничего не менять и не перепрогонять.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
