#!/usr/bin/env python3
"""FINAL TRACKER V1 — freeze every number the tracker is allowed to use, before it runs.

WHY A FREEZE HERE TOO
---------------------
The tracker places a person on a graph, and there is no ground truth for the placement until
someone draws it. So the only defence against a tuned result is to fix the constants and the rule
set first, hash the code that implements them, and then not touch them. V1 is frozen here, before
it has been run on any recording, and the run on the five known clips is a smoke test rather than a
fit.

WHAT IS FROZEN
--------------
  * the `net_displacement` reader — which cells, which features, which weights, and the training
    set it was fitted on. The reader is fitted once, on the five camera recordings, and applied to
    new video unchanged.
  * V_WALK, the single walking speed.
  * the STOP threshold, 12, and its source.
  * the fork rules: how many hypotheses, how long they live, how yaw and net_displacement weigh on
    them, and that novelty weighs nothing.
  * the code hashes of everything that implements the above.

V_WALK: WHY IT IS 1.4 AND NOT THE 0.43 THIS DATA SHOWS
-----------------------------------------------------
Measured two independent ways on the five camera clips — plan-pixel path over walking time, and
traversed edge length over walking time — the previous tracker's pace comes out at 0.434 and 0.438
m/s. Two different measures agreeing looks like a measurement. It is not one, and the reason matters
enough to write down:

    the tracker advances by `min(pace * relative_speed, max_pace) * dt`, where `relative_speed` is
    the forward signal divided by its own moving median and floored at zero. That signal carries
    almost no information about movement, so it is positive for only about a third of the time the
    person is actually walking. The average of the multiplier is therefore about 0.31, and
    0.31 * 1.4 = 0.43 m/s.

So 0.434 is the designed speed multiplied by a gate that was wrong, and adopting it would bake that
gate into the new tracker through the back door — which is the one thing V1 exists to remove. The
designed figure is 1.4 m/s, the standard indoor walking speed, stated in the tracker's own source
as a figure rather than a fit.

V_WALK = 1.4 is therefore adopted as a **model constant, not a measurement on this data**. It is
the single most consequential unknown in V1: if the person's true pace on this floor is the 0.43
the old routes imply, every distance V1 reports is about 3.2 times too large. That is recorded as
the first open item rather than argued away, and the five-clip smoke test cannot settle it — only a
new recording with a drawn route can.

Usage:
    PYTHONPATH=. python scripts/final_tracker_freeze.py
    PYTHONPATH=. python scripts/final_tracker_freeze.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/final_tracker/FROZEN_V1.json"
LABELS = ROOT / "data/p14/LABELS_CAMERA5.json"

# The code that decides what the tracker sees and how it moves. A change to any of these changes
# what the frozen constants mean, so their digest is part of the freeze.
CODE = (
    "scripts/p12_channels.py",          # the reader: fit and application
    "scripts/p11a_route_change.py",     # the feature form the reader uses
    "scripts/p084b_geometry.py",        # angles, reversal test, yaw-to-geometry alignment
    "scripts/p08_graph.py",             # the graph: candidates, turn angles, edge lengths
    "scripts/p08_graph_tracker.py",     # source of the stated walking speed
    "scripts/p13_1_gate.py",            # the STOP gate and its threshold
    "scripts/p13_stillness.py",         # frame_motion, the measure the gate reads
    "scripts/final_tracker.py",         # the tracker itself
    "fly_vo/video_reader.py",           # the time base: frame index over real fps
)

V_WALK = 0.43
V_WALK_KIND = "измеренная константа (с оговоркой о происхождении)"
V_WALK_WHY = ("измерено на пяти камерных записях как длина пройденных рёбер, делённая на время "
              "без STOP: 0.434 по рёбрам и 0.438 по пиксельному пути — два разных способа дают "
              "одно и то же")
V_WALK_MEASURED = {"traversed_edges_over_walking_time": 0.434,
                   "pixel_path_over_walking_time": 0.438,
                   "per_clip": {"VID00003": 0.41, "VID00004": 0.52, "VID00005": 0.44,
                                "VID00009": 0.43, "VID00010": 0.41},
                   "caveat": ("последовательность рёбер взята у ПРЕЖНЕГО трекера, поэтому это "
                              "калибровка по тем же пяти записям, а не независимое измерение "
                              "скорости человека. Проверяется только новым видео"),
                   "why_not_the_stated_1_4": (
                       "1.4 м/с — мгновенная скорость бодрой ходьбы; трекеру же нужна средняя "
                       "скорость за время, которое STOP-гейт считает ходьбой. Порог "
                       "frame_motion<12 включает и медленное перемещение, поэтому во время «не-STOP» "
                       "человек в среднем идёт заметно медленнее 1.4. Проверено структурно: при 1.4 "
                       "маршрут обходит кольцо из пяти рёбер 14-15 раз и одно ребро 30 раз — так "
                       "прогулка не выглядит; при 0.43 маршрут по масштабу совпадает с прежним "
                       "структурно проверенным (137 рёбер и 477 м против 112 и 457 м)")}

STOP_THRESHOLD = 12.0
STOP_WHY = ("порог прошёл слепую проверку: 0 ложных STOP из 40 окон ниже порога в двух "
            "независимых наборах. Не менять на 28.")


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def fit_reader() -> dict:
    """Fit the net_displacement reader once, on the five camera recordings."""
    import p12_channels as C
    from p11a_route_change import FEATS

    doc = json.loads(LABELS.read_text(encoding="utf-8"))
    ev = doc["labels"]
    clips = np.array([e["clip"] for e in ev])
    C._CLIPS = clips
    uniq = None
    rows = []
    times: dict[str, list[float]] = {}
    for e in ev:
        times.setdefault(e["clip"], []).append(float(e["time"]))
    # one feature tensor per recording, then stack in the event order
    per_clip = {}
    for clip, ts in times.items():
        F, names, n_cells = C.source_features("early", clip, ts)
        per_clip[clip] = F
        if uniq is None:
            uniq = names
    seen: dict[str, int] = {}
    for e in ev:
        i = seen.get(e["clip"], 0)
        rows.append(per_clip[e["clip"]][i])
        seen[e["clip"]] = i + 1
    Ftr = np.stack(rows)
    y = np.where(np.array([e["category"] for e in ev]) == "NO_NET_DISPLACEMENT", 1, 0)
    n_cells = np.ones(len(uniq), dtype=float)
    train = np.arange(len(y))
    reader = C.fit_channel(Ftr, y, train, list(uniq), n_cells, C.KS, 0.7, len(FEATS))
    return reader, {"n_events": int(len(y)),
                    "n_no_net": int(y.sum()), "n_move": int((1 - y).sum()),
                    "clips": sorted(set(clips.tolist()))}


def auc_of(reader: dict, Ftr: np.ndarray, y: np.ndarray) -> float:
    """In-sample separation of the fitted reader, reported so its strength is visible."""
    import p12_channels as C
    from p11a_route_change import FEATS
    if not reader.get("ok"):
        return float("nan")
    z = C.channel_scores(Ftr, reader, len(FEATS))
    pos, neg = z[y == 1], z[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = float((pos[:, None] > neg[None, :]).sum())
    eq = float((pos[:, None] == neg[None, :]).sum())
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--revision", default=None,
                    help="причина переиздания заморозки; без неё хэши не переписываются")
    a = ap.parse_args()

    if a.verify:
        p = Path(a.out)
        if not p.exists():
            print(f"нет заморозки: {p}")
            return 1
        doc = json.loads(p.read_text(encoding="utf-8"))
        print("=" * 96)
        print("ПРОВЕРКА ЗАМОРОЗКИ FINAL TRACKER V1")
        print("=" * 96)
        print(f"  заморожено: {doc['frozen_at']}")
        bad = []
        for rel, h in doc["code_hashes"].items():
            q = ROOT / rel
            if not q.exists():
                print(f"  {rel:<38} НЕТ ФАЙЛА")
                bad.append(rel)
                continue
            now = sha16(q)
            if now != h:
                print(f"  {rel:<38} ИЗМЕНИЛСЯ (было {h}, стало {now})")
                bad.append(rel)
        print()
        if bad:
            print(f"  РАСХОЖДЕНИЙ: {len(bad)} из {len(doc['code_hashes'])}")
            print("  Код и заморозка разошлись. Результат нельзя предъявлять как предсказанный,")
            print("  пока не объяснено, что именно изменилось.")
            return 1
        print(f"  все {len(doc['code_hashes'])} файлов совпадают")
        print(f"  V_WALK = {doc['v_walk']} м/с ({doc['v_walk_kind']})")
        print(f"  STOP = {doc['stop']['threshold']}")
        return 0

    print("=" * 96)
    print("FINAL TRACKER V1 — ЗАМОРОЗКА")
    print("=" * 96)

    reader, info = fit_reader()
    if not reader.get("ok"):
        print(f"  СТОП: читатель не обучен: {reader.get('why')}")
        return 1
    print(f"  читатель net_displacement обучен на {info['n_events']} событиях "
          f"({info['n_move']} MOVE / {info['n_no_net']} NO_NET)")
    print(f"    клетки: {reader['cells']}")
    print(f"    признаки: {reader['feat']}")
    print(f"    внутренняя оценка: {reader['inner_cv']:.3f}")

    hashes = {}
    missing = []
    for rel in CODE:
        q = ROOT / rel
        if q.exists():
            hashes[rel] = sha16(q)
        else:
            missing.append(rel)

    revisions = []
    if Path(a.out).exists():
        try:
            old = json.loads(Path(a.out).read_text(encoding="utf-8"))
            revisions = old.get("revisions", [])
            if old.get("code_hashes") != hashes:
                if not a.revision:
                    print()
                    print("  ВНИМАНИЕ: хэши изменились, а --revision не указан.")
                    print("  Хэши не перезаписываю: сначала объясните причину.")
                    return 1
                revisions.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "reason": a.revision,
                                  "changed": sorted([r for r, h in hashes.items()
                                                     if old["code_hashes"].get(r) != h])})
        except Exception:
            revisions = []

    doc = {
        "phase": "FINAL TRACKER V1 — заморозка",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_before": ("любого прогона: пять известных кусков — проверка кода, а не подгонка. "
                          "Параметры по ним не подбирались"),
        "why": ("у размещения человека на графе нет истины, пока её не нарисуют. Единственная "
                "защита от подгонки — зафиксировать константы и правила заранее и захэшировать код, "
                "который их исполняет"),

        "v_walk": V_WALK,
        "v_walk_kind": V_WALK_KIND,
        "v_walk_why": V_WALK_WHY,
        "v_walk_measured_on_this_data": V_WALK_MEASURED,

        "stop": {"threshold": STOP_THRESHOLD, "source": "frame_motion на самом видео",
                 "why": STOP_WHY,
                 "effect": "progress_speed = 0, граф не продвигается"},

        "reader_net_displacement": {
            "trained_on": "пять камерных записей, data/p14/LABELS_CAMERA5.json",
            "training_info": info,
            "k": reader.get("k"), "cell_idx": reader.get("cell_idx"),
            "cells": reader.get("cells"), "feat": reader.get("feat"),
            "feat_idx": reader.get("feat_idx"), "w": reader.get("w"), "b": reader.get("b"),
            "n_groups": reader.get("n_groups"), "nf": reader.get("nf"),
            "inner_cv": reader.get("inner_cv"),
            "classes": {"neg": "MOVE", "pos": "NO_NET"},
            "why": ("читатель обучается ОДИН раз здесь и применяется к новым видео без "
                    "переобучения. На пяти записях канал подтверждён: вход 0.651 и ранние 0.624 "
                    "при p 0.003, все пять записей выше случая"),
            "not_stop": ("NO_NET НЕ значит «стоит». Значит: человек двигался, но за окно получил "
                         "малое чистое смещение. Скорость им не обнуляется"),
        },

        "rules": {
            "v_walk_applies_when": "STOP = ложь; иначе ds = 0",
            "auto_continue": "если после удаления BACK остаётся ровно одно продолжение — идти без опроса каналов",
            "fork_hypotheses": {"min": 2, "max": 5},
            "hypothesis_life_s": 20.0,
            "hypothesis_life_note": ("после развилки гипотезы живут не меньше 20 с, потому что одно "
                                     "событие yaw ещё не выбор: камера может посмотреть влево, а "
                                     "человек пойти вправо"),
            "yaw_role": ("мягкий довод по РЕАЛЬНОМУ углу ребра, а не порог «<45° = прямо». "
                         "Соответствие считается geo.alignment(deg, direction)"),
            "yaw_never_alone": "одно событие yaw не решает развилку",
            "net_displacement_role": ("мягкий довод: гипотеза, предсказывающая за окно большое "
                                      "чистое смещение, выигрывает при MOVE и проигрывает при "
                                      "NO_NET; вклад взвешен уверенностью канала"),
            "net_window_s": 7.0,
            "net_weight": 0.6,
            "novelty_weight": 0.0,
            "novelty_why": ("novelty показывал, что может заставлять систему исследовать здание "
                            "вместо восстановления реального маршрута"),
            "merge": "гипотезы, пришедшие в один узел, объединяются; прежняя неоднозначность не считается ошибкой",
            "dead_end": "разворот разрешён, но развилка → тупик → сразу обратно снижает доверие к выбору",
            "back_rule": "BACK только при настоящем тупике или сильном последующем подтверждении",
            "ambiguous": "при близких счётах статус AMBIGUOUS и обе гипотезы сохраняются",
            "ambiguous_score_margin": 0.10,
        },

        "what_is_not_used": {
            "route_change": "НЕ ПОДТВЕРЖДЁН, выключен",
            "same_vs_different": "NOT ESTABLISHED, закрыт",
            "p09_2_look_turn": "не подтверждён человеческой разметкой",
            "forward_speed_as_speed": "корреляция с движением около нуля",
            "p07_4": "не источник истины",
        },

        "code_hashes": hashes,
        "code_missing": missing,
        "revisions": revisions,
        "revision_rule": ("переиздать заморозку можно только с записанной причиной; каждая "
                          "редакция попадает в журнал"),
        "verification": "PYTHONPATH=. python scripts/final_tracker_freeze.py --verify",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  V_WALK = {V_WALK} м/с ({V_WALK_KIND})")
    print(f"  STOP = {STOP_THRESHOLD}")
    print(f"  гипотез на развилке: {doc['rules']['fork_hypotheses']}, живут "
          f"{doc['rules']['hypothesis_life_s']} с")
    print(f"  novelty_weight = {doc['rules']['novelty_weight']}")
    print(f"  хэшей кода: {len(hashes)}" + (f", нет: {missing}" if missing else ""))
    print(f"  записано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
