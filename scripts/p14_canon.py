#!/usr/bin/env python3
"""P14 — one measurement per (clip, source), checked for internal coherence.

WHAT A MEASUREMENT IS
---------------------
A level comparison is only meaningful if every level of a clip was read from the same file. The
project had no way to state or check that, and it failed twice in the same way: the name
`VID00002` meant the camera's file to the route and a downloaded file to the page and the neural
recordings, and `VID00006`'s neural levels cover 250 seconds of an old clip while its route covers
1228 seconds of the camera's. So the unit here is not the clip but the *measurement*: a clip, a
declared source, the levels read from it, and the set that source belongs to.

A measurement is coherent when every level covers the same span and names the same source. What the
checks can and cannot see is stated per check, because the two failures above are not symmetric:

    the route of VID00002 and its levels agree on duration to the second while being different
    recordings, so a duration check passes and cannot be trusted alone

    the levels of VID00006 and its route disagree by 978 seconds, so a duration check catches it

Hence the source is compared as well, by the file's recorded size where the source still exists, and
by the identity tests already run where it does not. Where a source has been deleted the record says
so instead of implying verification that did not happen.

WHY THE SETS MATTER MORE THAN THE TOTALS
----------------------------------------
Three sets exist among the five clips P10 was built on:

    downloads20   VID00001, VID00002 — 1080p files from ~/Downloads, about 1228 s each
    legacy250     VID00006         — a 250 s clip that no longer exists
    camera        VID00005, VID00009 — the camera's own files

An experiment may draw from one set. Its result quoted over all five is a mixture, and a number
that holds only in the mixture is not a property of the pipeline.

Usage:
    PYTHONPATH=. python scripts/p14_canon.py
    PYTHONPATH=. python scripts/p14_canon.py --no-runs      # только реестр и проверки
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/p14"
DUR_TOL_S = 2.0

# Which set each clip's recordings belong to, and what says so. Declared, with its evidence, rather
# than inferred: inference from names is what produced the original mix-up.
SETS = {
    "downloads20": {
        "clips": ["VID00001", "VID00002"],
        "resolution": "1920x1080",
        "evidence": ("data/p01r/VID00001.AVI is a symlink to febf7cf1-…_VID00001.AVI in "
                     "~/Downloads; VID00002 was matched to dae9fd91-…_VID00002.AVI by frame "
                     "identity at +0.996 against the camera's +0.405 background"),
    },
    "legacy250": {
        "clips": ["VID00006"],
        "resolution": "unknown (file deleted)",
        "evidence": ("the P09 and P10 recordings for VID00006 end at 250 s while the camera's file "
                     "of that name holds 1228 s; the 250 s clip that produced them was removed"),
    },
    "camera": {
        "clips": ["VID00005", "VID00009", "VID00010"],
        "resolution": "1280x720",
        "evidence": ("copied from /Volumes/NO NAME/DCIM by scripts/p13_queue.py; the recorded byte "
                     "sizes in output/p13/sources/*.json equal the camera files' sizes"),
    },
}


def level_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import numpy as np
        z = np.load(path)
        return float(np.asarray(z["t"]).astype(float)[-1])
    except Exception:
        return None


def route_seconds(clip: str) -> float | None:
    p = ROOT / f"output/p13/runs/{clip}/graph_trajectory.csv"
    if not p.exists():
        return None
    with p.open() as fh:
        fh.readline()
        rows = [ln for ln in fh if ln.strip()]
    return float(rows[-1].split(",")[0]) if rows else None


def yaw_seconds(clip: str) -> float | None:
    """Length of the trace the yaw chain actually reads for this clip.

    Read through the registry rather than from a fixed path: `data/p07_videos.json` overrides the
    built-in default, and the overrides point at camera recordings from 24 September. Assuming the
    directory `output/p06_neurons` belonged to VID00006 was wrong — it belonged to nobody by then,
    and the manifest now records it as an orphan.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from p07_videos import videos as _videos
        spec = _videos().get(clip)
    except Exception:
        return None
    if not spec:
        return None
    return level_seconds(spec["trace"])


def recorded_source_size(clip: str) -> int | None:
    p = ROOT / f"output/p13/sources/{clip}.json"
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["bytes"])
    except Exception:
        return None


def camera_size(clip: str) -> int | None:
    p = Path("/Volumes/NO NAME/DCIM") / f"{clip}.AVI"
    return p.stat().st_size if p.exists() else None


def measurements() -> dict:
    frozen = json.loads((ROOT / "data/p10/FROZEN_P10.json").read_text(encoding="utf-8"))
    labelled = {}
    for l in frozen["labels"]:
        labelled.setdefault(l["clip"], []).append(l["time"])

    out = {}
    for name, spec in SETS.items():
        for clip in spec["clips"]:
            lv = {
                "P09_dn": level_seconds(ROOT / f"output/p09/trace_{clip}.npz"),
                "P10_input_early": level_seconds(ROOT / f"output/p10/brain_{clip}.npz"),
                "yaw_chain": yaw_seconds(clip),
                "route": route_seconds(clip),
            }
            cells = ROOT / f"output/p11b/cells_{clip}.npz"
            if cells.exists():
                lv["P11B_candidates"] = level_seconds(cells)
            ev = sorted(labelled.get(clip, []))
            rec = {
                "clip": clip, "set": name, "declared_source_resolution": spec["resolution"],
                "levels_seconds": lv,
                "events": len(ev),
                "event_times_s": ev,
                "source_bytes_recorded": recorded_source_size(clip),
                "source_bytes_camera_now": camera_size(clip),
            }
            out[f"{clip}__{name}"] = rec

    # the second measurement of VID00002: the same clip, a different source, recorded by hand
    cam = ROOT / "output/p13/camera_traces/trace_VID00002_from_camera.npz"
    if cam.exists():
        out["VID00002__camera_recheck"] = {
            "clip": "VID00002", "set": "camera",
            "declared_source_resolution": "1280x720",
            "levels_seconds": {"P09_dn": level_seconds(cam)},
            "events": 0,
            "note": ("посчитано при разборе путаницы файлов: один и тот же клип, другой исходник. "
                     "меток для него нет, поэтому в диагностику не входит — но именно он и "
                     "VID00002__downloads20 показывают, что клип и измерение это не одно и то же"),
        }
    return out


def check(rec: dict) -> list[dict]:
    """Violations of coherence, each stating what it can and cannot establish."""
    bad = []
    lv = {k: v for k, v in rec["levels_seconds"].items() if v is not None}
    if not lv:
        return bad
    rounded = {k: round(v) for k, v in lv.items()}

    # the route against the neural levels: catches the VID00006 case outright
    if "route" in lv and len(lv) > 1:
        others = {k: v for k, v in lv.items() if k != "route"}
        ref = next(iter(others.values()))
        if abs(lv["route"] - ref) > DUR_TOL_S:
            bad.append({
                "kind": "длительность",
                "severity": "нарушение",
                "detail": (f"маршрут покрывает {lv['route']:.0f} с, нейронные уровни "
                           f"{rounded} — разница {abs(lv['route'] - ref):.0f} с"),
                "catches": "да: провал по длительности виден",
            })

    # the yaw chain against the levels it is compared with. This is the level that was missing from
    # the first version of this check, and it is the one that carries VID00006: its yaw and its
    # route both cover the camera's 1228 s while its P09/P10 levels cover 250 s of an old clip.
    if "yaw_chain" in lv and "P09_dn" in lv:
        if abs(lv["yaw_chain"] - lv["P09_dn"]) > DUR_TOL_S:
            bad.append({
                "kind": "yaw против уровней",
                "severity": "нарушение",
                "detail": (f"yaw-цепочка читает след на {lv['yaw_chain']:.0f} с, а уровни P09/P10 "
                           f"— на {lv['P09_dn']:.0f} с: одна и та же величина сравнивается между "
                           f"двумя разными записями"),
                "catches": ("да, если yaw включён в проверку — в первой версии его не было, и "
                            "нарушение осталось незамеченным"),
            })

    # the neural levels against each other
    vals = list(lv.values())
    if max(vals) - min(vals) > DUR_TOL_S:
        bad.append({
            "kind": "длительность внутри уровней",
            "severity": "нарушение",
            "detail": f"уровни расходятся: {rounded}",
            "catches": "да",
        })

    # the route's source against the levels' source, where a size was recorded.
    # The third element used to be a literal None, so `cam is not None` was never true and this
    # warning could not fire at all — a check that always passes, which is how the mix-up survived
    # in the first place.
    rb = rec.get("source_bytes_recorded")
    cam = rec.get("source_bytes_camera_now")
    if rec["set"] != "camera" and rb is not None and cam is not None and rb == cam:
        bad.append({
            "kind": "источник",
            "severity": "нарушение",
            "detail": ("маршрут читал файл камеры (записанный размер совпал с камерой), а "
                       f"нейронные уровни — файл из набора {rec['set']}; длительности при этом "
                       "совпадают, поэтому проверка по длительности пропускает это"),
            "catches": ("размер поймал маршрут, но не уровни: чтобы отличить уровни, нужно сравнить "
                        "кадры — как это и было сделано для VID00002 (+0.996 против фона)"),
        })
    if rb is None:
        bad.append({
            "kind": "источник",
            "severity": "не проверено",
            "detail": ("размер прочитанного файла не записан и файла-источника уже нет: проверить "
                       "источник нечем"),
            "catches": "нет",
        })
    return bad


def run_pair(exclude: list[str], tag: str, extra: list[str] | None = None) -> dict:
    """Run the two diagnostics restricted to one set, and read back their headline numbers."""
    ex = ",".join(exclude)
    log = {}
    p10_out = OUT / f"run_{tag}"
    p11_out = OUT / f"run_{tag}_p11a"
    cmds = [
        [sys.executable, "scripts/p10_levels.py", "--perm", "300", "--exclude", ex,
         "--out", str(p10_out)],
        [sys.executable, "scripts/p11a_route_change.py", "--perm", "200", "--exclude", ex,
         "--least", "2", "--out", str(p11_out)],
    ]
    for c in cmds:
        env = {"PYTHONPATH": ".", "PATH": "/usr/bin:/bin:/usr/local/bin"}
        r = subprocess.run(c, cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=3600)
        log[Path(c[1]).stem] = {"returncode": r.returncode,
                                "tail": (r.stdout or "")[-600:] + (r.stderr or "")[-300:]}
    got = {"log": log}
    lr = p10_out / "levels_report.json"
    if lr.exists():
        d = json.loads(lr.read_text(encoding="utf-8"))
        got["p10"] = {lvl: {t: {"auc": r.get("auc_aggregate"), "p": r.get("p_aggregate")}
                            for t, r in v.items() if isinstance(r, dict) and "auc_aggregate" in r}
                      for lvl, v in d["levels"].items()}
    pr = p11_out / "report.json"
    if pr.exists():
        d = json.loads(pr.read_text(encoding="utf-8"))
        got["p11a"] = {
            "n_events": d.get("n_events"),
            "dn68_logreg": d["sources"]["dn68"]["logreg"].get("auc_oof"),
            "yaw18_logreg": d["sources"].get("dn_yaw18", {}).get("logreg", {}).get("auc_oof"),
            "yaw_csv": d["sources"].get("yaw_csv", {}).get("auc_oof"),
            "null_p": d.get("null", {}).get("p"),
        }
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-runs", action="store_true", help="только реестр и проверки")
    ap.add_argument("--out", default=str(OUT / "CANON.json"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("P14 — КАНОНИЗАЦИЯ: ИЗМЕРЕНИЯ И ПРОВЕРКА СВЯЗНОСТИ")
    print("=" * 100)

    ms = measurements()
    print(f"  измерений: {len(ms)}")
    print()
    print(f"  {'измерение':<28}{'набор':<14}{'P09':>8}{'P10':>8}{'маршрут':>9}{'событий':>9}")
    print("  " + "-" * 74)
    viol = {}
    for k, rec in ms.items():
        lv = rec["levels_seconds"]
        f = lambda x: "—" if x is None else f"{x:.0f}"          # noqa: E731
        print(f"  {k:<28}{rec['set']:<14}{f(lv.get('P09_dn')):>8}"
              f"{f(lv.get('P10_input_early')):>8}{f(lv.get('route')):>9}"
              f"{rec['events']:>9}")
        v = check(rec)
        if v:
            viol[k] = v

    print()
    print("=" * 100)
    print("  НАРУШЕНИЯ СВЯЗНОСТИ")
    print("=" * 100)
    if not viol:
        print("  нет")
    for k, vs in viol.items():
        print(f"  {k}")
        for v in vs:
            print(f"    [{v['severity']}] {v['kind']}: {v['detail']}")
            print(f"        ловится проверкой: {v['catches']}")

    runs = {}
    if not a.no_runs:
        print()
        print("=" * 100)
        print("  ПРОГОНЫ ПО НАБОРАМ (только диагностика, без подбора клеток и порогов)")
        print("=" * 100)
        all_five = {"VID00001", "VID00002", "VID00005", "VID00006", "VID00009"}
        for tag, spec in SETS.items():
            inside = set(spec["clips"]) & all_five
            if len(inside) < 2:
                print(f"  {tag}: кусков в P10 — {len(inside)}, прогон невозможен "
                      f"(нужно хотя бы два для проверки на отложенном)")
                continue
            exclude = sorted(all_five - inside)
            print(f"\n  {tag}: куски {sorted(inside)}, исключаю {exclude}")
            got = run_pair(exclude, tag)
            runs[tag] = got
            if "p11a" in got:
                p = got["p11a"]
                print(f"    P11.A: событий {p['n_events']}, 68 DN {p['dn68_logreg']}, "
                      f"yaw18 {p['yaw18_logreg']}, yaw_csv {p['yaw_csv']}, p(нуль) {p['null_p']}")
            if "p10" in got:
                for lvl in ("input", "early", "dn"):
                    r = got["p10"].get(lvl, {})
                    for task in ("SAME_vs_DIFFERENT", "MOVING_vs_NO_NET"):
                        if task in r:
                            print(f"    P10 {lvl:<6} {task:<18} "
                                  f"AUC {r[task]['auc']}  p {r[task]['p']}")

    doc = {
        "phase": "P14 — канонизация данных",
        "why": ("уровни нельзя сравнивать, если прочитаны из разных файлов; проект не мог это "
                "ни высказать, ни проверить, и ошибся так дважды"),
        "unit": "измерение = клип + объявленный источник + уровни, прочитанные из него",
        "sets": SETS,
        "measurements": ms,
        "violations": viol,
        "checks_can_and_cannot": {
            "duration": ("ловит VID00006 (250 против 1228 с), НЕ ловит VID00002 (1228 против 1228 с "
                         "при разных записях)"),
            "source_size": ("ловит подмену, когда размеры расходятся; у VID00002 маршрут совпал с "
                            "камерой по размеру, поэтому размер подтверждает маршрут, а не уровни"),
            "identity_test": ("нужен там, где длительность и размер совпадают: только сравнение "
                              "кадров отличило dae9fd91-… от камерного VID00002"),
        },
        "runs": runs,
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"\n  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
