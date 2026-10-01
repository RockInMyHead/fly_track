#!/usr/bin/env python3
"""P12.4 — freeze everything before VID00010 exists, so the blind run is blind.

WHY THIS HAS TO BE DONE BEFORE THE VIDEO ARRIVES
-----------------------------------------------
The five recordings have been used in P09, P10, P11 and P12. Any further adjustment made after
seeing a sixth one would be an adjustment made with knowledge of the answer. So the whole chain is
frozen now: the two route walks, the three readers, the confidence threshold, the graph, and the
readers themselves as fitted numbers rather than as a procedure that will run again later.

That last point is the one that would otherwise be missed. The readers in output/p12/channels.json
were fitted leave-one-recording-out, five times, and each one never saw its own recording. A sixth
recording is a recording that *none* of them saw — but none of them was fitted to be applied to it
either: each was fitted on four. The reader that belongs on a new recording is the one fitted on
all five old ones, and it does not exist yet. Fitting it after VID00010 has been looked at, or even
after it has been loaded, is the leak this file exists to prevent. It is fitted here, written down
as numbers, and hashed.

WHAT IS FROZEN
--------------
    the readers        cells, features, directions, weights, bias — as numbers, per channel
    tau                0.10, fixed now and not to be revisited after the video is seen
    the code           P08.6, the geometry rule, the graph tracker, the three P12 scripts
    the graph          data/p08/graph.json
    the front end      visual_encoder and config, which produce the drive the readers read
    the labels         P10's frozen set and the review that produced it (old data, used for
                       fitting and for nothing else after this)

A manifest of sha256 hashes is written next to it. scripts/p12_blind.py refuses to run if any of
them has changed, and says which file changed, so a post-hoc edit cannot pass unnoticed.

Usage:
    PYTHONPATH=. python scripts/p12_freeze.py
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.p12_channels as P  # noqa: E402

TAU = 0.10          # fixed now; the phase's own rule is that this is never revisited
MIN_CONS = 0.70     # the same value the per-recording readers were built with

CODE = (
    "scripts/p086_multi.py",            # P08.6, the multi-hypothesis walk
    "scripts/p084b_geometry.py",        # the decision rule P08.6 calls
    "scripts/p08_graph.py",             # graph and candidates
    "scripts/p08_graph_tracker.py",
    "scripts/p12_channels.py",
    "scripts/p12_replay.py",
    "scripts/p12_run.py",
    "scripts/p12_freeze.py",
    "scripts/p12_blind.py",
    "fly_vo/visual_encoder.py",
    "fly_vo/config.py",
    "fly_vo/local_motion_inject.py",
    "fly_vo/malecns_engine.py",
    # the recorders matter as much as the readers: they decide how the brain is driven for the new
    # recording, so changing one of them between now and the blind run would change what the
    # readers see without changing a single reader
    "scripts/p10_record.py",            # brain input + early visual populations
    "scripts/p09_real_holdout.py",      # the 68 descending cells
    "scripts/p07_trajectory.py",        # yaw and forward
    "fly_vo/brain_clock.py",            # the clock both recorders walk the video on
)

DATA = (
    "data/p08/graph.json",
    "output/p12/channels.json",          # the five per-recording readers, for reference
    "output/p086/report.json",           # the junction list and the P08.6 picks
    "data/p10/FROZEN_P10.json",
    "data/p095/review_set_v2.json",
    "data/p095/human_labels_v2.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_for_new_recording() -> dict:
    """Both readers fitted on all five recordings — the ones that belong on a sixth.

    Nothing here has seen VID00010, because it does not exist. This is the only fitting step in the
    whole P12.4 procedure, and it happens before the video arrives so that it cannot be influenced
    by it.
    """
    ev, _lab = P.review()
    clips = np.array([e["clip"] for e in ev])
    P._CLIPS = clips
    videos = sorted(set(clips.tolist()))
    nf = len(P.FEATS)

    by_video: dict[str, list[tuple[int, float]]] = {}
    for i, e in enumerate(ev):
        by_video.setdefault(e["clip"], []).append((i, float(e["time"])))

    F: dict[str, np.ndarray] = {}
    for kind in ("dn", "early"):
        rows: list = [None] * len(ev)
        for v, items in by_video.items():
            Fv, _names, _nc = P.source_features(kind, v, [t for _i, t in items])
            for (i, _t), row in zip(items, Fv):
                rows[i] = row
        F[kind] = np.stack(rows)

    uniq_dn = list(P.dn(videos[0])[2])
    uniq_ea = list(P.early_series(videos[0])[2])
    nc_dn = np.ones(len(uniq_dn))
    nc_ea = P.early_series(videos[0])[3]

    y_rc = np.where(np.array([e["old_label"] for e in ev]) == "TURN", 1, 0)
    keep_rc = np.isin([e["old_label"] for e in ev], ["LOOK", "TURN"])
    y_nd = np.where(np.array([e["old_label"] for e in ev]) == "NO_LOCOMOTION", 1, 0)

    allr = np.arange(len(ev))
    rc = P.fit_channel(F["dn"], y_rc, allr[keep_rc], uniq_dn, nc_dn, min_cons=MIN_CONS)
    nd = P.fit_channel(F["early"], y_nd, allr, uniq_ea, nc_ea, min_cons=MIN_CONS)

    def plain(ch: dict, label: str) -> dict:
        if not ch.get("ok"):
            raise RuntimeError(f"{label}: не обучилось — {ch.get('why')}")
        return {"source": label, "n_groups": ch["n_groups"], "nf": ch["nf"], "k": ch["k"],
                "cell_idx": [int(x) for x in ch["cell_idx"]],
                "cells": [str(x) for x in ch["cells"]],
                "feat_idx": [int(x) for x in ch["feat_idx"]],
                "feat": [str(x) for x in ch["feat"]],
                "sign": [float(x) for x in ch["sign"]],
                "w": [float(x) for x in ch["w"]], "b": float(ch["b"])}

    return {"route_change": plain(rc, "68 нисходящих клеток"),
            "net_displacement": plain(nd, "20 ранних зрительных популяций"),
            "classes": {"route_change": {"neg": "SAME", "pos": "DIFFERENT"},
                        "net_displacement": {"neg": "MOVE", "pos": "NO_NET"}}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data/p12"))
    ap.add_argument("--force", action="store_true",
                    help="перезаписать (нельзя после появления нового ролика)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "FROZEN_BEFORE_VID00010.json"
    if dest.exists() and not args.force:
        print(f"{dest} уже существует — повторная заморозка запрещена")
        return 1

    print("=" * 100)
    print("P12.4 — ЗАМОРОЗКА ПЕРЕД ПОЯВЛЕНИЕМ VID00010")
    print("=" * 100)
    print()

    print("─── ОБУЧЕНИЕ ЧИТАТЕЛЕЙ ДЛЯ НОВОГО РОЛИКА (на всех пяти старых) ───")
    readers = fit_for_new_recording()
    for name, r in readers.items():
        if name == "classes":
            continue
        print(f"  {name}: групп {r['n_groups']}, выбрано {r['k']} пар")
        for c, f, s in zip(r["cells"], r["feat"], r["sign"]):
            print(f"      {c:<16}{f:<14}{'+' if s > 0 else '-'}")
    print()

    manifest = {}
    missing = []
    for rel in CODE + DATA:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        manifest[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    if missing:
        print(f"  СТОП: нет файлов: {missing}")
        return 1
    print(f"─── ХЭШИ: {len(manifest)} файлов ───")
    for rel, m in manifest.items():
        print(f"  {m['sha256'][:16]}  {m['bytes']:>10}  {rel}")
    print()

    doc = {
        "phase": "P12.4 — слепая проверка на новом видео",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_before": "VID00010",
        "why": (
            "Пять записей использованы в P09-P12. Любая правка после знакомства с шестой была бы "
            "правкой со знанием ответа, поэтому вся цепочка фиксируется сейчас, а читатели — "
            "числами, а не процедурой, которая запустится позже."
        ),
        "rules": {
            "frozen": ["P08.6 без изменений", "P12 без изменений", "tau = 0.10",
                       "camera_yaw — текущая версия", "route_change — текущая версия",
                       "net_displacement — текущая версия",
                       "все клетки, признаки, веса и пороги", "граф маршрутизации"],
            "after_the_video_appears": [
                "ШАГ 1. Положить VID00010 в проект. НЕ смотреть и НЕ размечать вручную.",
                "ШАГ 2. Вслепую прогнать P08.6 и P12 tau=0.10, сохранить оба результата "
                "ДО разметки.",
                "ШАГ 3. Зафиксировать рёбра, развилки, решения, сигналы каналов, "
                "неоднозначные участки, тупики, возвраты.",
                "ШАГ 4. Только после этого — человеческая разметка VID00010.",
                "ШАГ 5. Сравнить P08.6 и P12 по восьми показателям.",
            ],
            "the_rule_that_matters": (
                "Первый результат — основной результат проверки. После просмотра VID00010 ничего "
                "не менять и не перепрогонять с новыми параметрами."
            ),
            "even_a_strange_result_counts": (
                "даже если в середине подготовки обнаружится странный результат — ничего не "
                "чинить. Если код выполнил то, что записано в заморозке, странный результат тоже "
                "является результатом слепой проверки."
            ),
            "a_new_version_is_a_new_phase": (
                "если после VID00010 появится новая версия, это уже P13, и проверять её надо на "
                "VID00011 или следующем новом видео, а не на том же ролике"
            ),
            "if_a_second_recording_is_available": (
                "VID00011 желателен: один ролик может содержать всего несколько полезных "
                "развилок, два дают существенно более убедительный ответ"
            ),
        },
        "chain_before_the_blind_run": {
            "note": (
                "порядок обязателен. Два шага легко перепутать, и оба были перепутаны в первой "
                "версии: p071/trajectory пишется p072_trajectory.py, а не p07_trajectory.py, и "
                "ему нужен p072/events от детектора. Шаг регистрации падает тихо, поэтому он "
                "назван отдельно."
            ),
            "steps": [{"artifact": a, "how": h} for a, h in (
                ("data/p01r/{v}.AVI", "положить ролик в проект"),
                ("data/p07_videos.json", "зарегистрировать: scripts/p07_videos.py --add {v} "
                                         "--trace … --targets …; нужен трейс в стиле P06"),
                ("output/p07/yaw_signal_{v}.csv", "scripts/p07_trajectory.py"),
                ("output/p07/forward_signal_{v}.csv", "scripts/p07_trajectory.py"),
                ("output/p072/events_{v}.csv", "scripts/p072_event_detector.py"),
                ("output/p071/trajectory_{v}.csv", "scripts/p072_trajectory.py"),
                ("output/p09/trace_{v}.npz", "scripts/p09_real_holdout.py --video {v} --record"),
                ("output/p10/brain_{v}.npz", "scripts/p10_record.py --video {v}"),
                ("<run>/decisions.csv и report.json", "scripts/p08_graph_tracker.py --video {v} "
                 "--out output/p12/blind_run_{v}"),
                ("output/p12/BLIND_{v}_RESULT.json", "scripts/p12_blind.py --video {v} "
                 "--run output/p12/blind_run_{v}"),
            )],
            "the_run_directory_must_be_yours": (
                "output/p084c/run содержит прогон VID00006. Первая версия p12_blind.py читала "
                "именно его по умолчанию, то есть слепой прогон для нового ролика вернул бы "
                "решения VID00006. Теперь --run обязателен, а report.json сверяется по имени "
                "ролика."
            ),
        },
        "tau": TAU,
        "min_cons": MIN_CONS,
        "tau_note": (
            "0.10 выбран ПОСЛЕ просмотра старых данных и потому не является доказательством "
            "улучшения на них. Он замораживается здесь именно для честной проверки на новом "
            "ролике. На новых данных τ не пересматривается."
        ),
        "readers_for_new_recording": readers,
        "readers_note": (
            "это единственная обучающая операция во всей процедуре P12.4. Оба читателя обучены "
            "на всех пяти старых роликах; VID00010 не существует, и ничто в них не могло его "
            "увидеть"
        ),
        "manifest": manifest,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  записано: {dest}")
    print()
    print("  Этот файл — граница. Всё, что после него, либо воспроизводит его, либо")
    print("  считается нарушением заморозки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
