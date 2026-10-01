#!/usr/bin/env python3
"""
P08 — the two mandatory cases, and the rules around them, driven by synthetic yaw.

Why these tests exist and why they are synthetic
------------------------------------------------
Stages 23 and 24 of the P08 description name two cases that must work:

    23  mid-corridor glance     the walker looks right and comes back in the middle of a
                                corridor, and the route must not change
    24  turn at a junction      the walker arrives at a junction, turns right for real,
                                and the right-hand passage must be chosen

Checking them on the recording needs a graph of the actual room, which only a person who
knows the room can draw. The rules themselves, though, do not depend on the room: they
depend on the graph and the yaw trace. So both are driven here by a synthetic trace where
the yaw the fly would report is placed exactly where the test needs it.

The graph is the junction from the stage description with one passage added downward, so
that a right turn has somewhere to lead:

            C   (left)
            |
    A ------B------ D   (straight on)
            |
            F   (right)

This is a test of the routing rules, not of the yaw signal. Whether the yaw signal itself
carries the direction is P07's question and was answered there.

Usage:
    PYTHONPATH=. python scripts/p08_logic_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402
from p08_graph_tracker import pix_per_sec_for, track  # noqa: E402

DT = 0.02           # 50 Hz, the rate the brain recording runs at
T_END = 60.0

# Failures that are already in the tree and are not caused by this phase. They are kept in
# the suite as known rather than deleted, so they stay visible, and reported separately so
# they cannot be mistaken for something P08.4B broke.
KNOWN_PREEXISTING: dict[str, str] = {
    # Empty as of P08.4C. The symmetric-junction case used to fail here: the revision window
    # counted the passage the walker had just entered as already walked, so it looked spent
    # against an unused alternative and the deferred decision flipped, reporting the flip with
    # 0.93 confidence. The tracker now reads the visit history as it stood before the current
    # traversal, and the case passes. Kept as an empty mapping so the next such finding has a
    # place to go rather than being deleted from the suite.
}

NODES = {"A": (0.10, 0.50), "B": (0.40, 0.50), "C": (0.40, 0.20),
         "D": (0.75, 0.50), "F": (0.40, 0.80)}
EDGES = [("A_B", "A", "B"), ("B_C", "B", "C"), ("B_D", "B", "D"),
         ("B_F", "B", "F")]


def junction() -> Graph:
    return Graph({
        "img_w": 1600, "img_h": 1000,
        "nodes": [{"id": k, "x": v[0], "y": v[1]} for k, v in NODES.items()],
        "edges": [{"id": i, "from": a, "to": b, "bidirectional": True}
                  for i, a, b in EDGES],
        "start": {"type": "edge", "edge": "A_B", "from": "A", "to": "B"},
    })


def corridor() -> Graph:
    """A straight run A—B—D with no alternative: the no-choice case of stages 14-15."""
    return Graph({
        "img_w": 1600, "img_h": 1000,
        "nodes": [{"id": "A", "x": 0.10, "y": 0.50},
                  {"id": "B", "x": 0.40, "y": 0.50},
                  {"id": "D", "x": 0.75, "y": 0.50}],
        "edges": [{"id": "A_B", "from": "A", "to": "B", "bidirectional": True},
                  {"id": "B_D", "from": "B", "to": "D", "bidirectional": True}],
        "start": {"type": "edge", "edge": "A_B", "from": "A", "to": "B"},
    })


def blank(t_end: float = T_END) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, t_end, DT)
    return t, np.zeros_like(t), np.ones_like(t)


def no_left_graph() -> Graph:
    """A junction with a straight-on passage and a right turn, and no left at all.

    Used for the case where the signal names a side that does not exist. Geometry has a
    clear answer there — carry on — so the decision should not be deferred. That is
    different from a genuine tie, which is what `defer_graph` produces.
    """
    return Graph({
        "img_w": 1600, "img_h": 1000,
        "nodes": [{"id": "A", "x": 0.10, "y": 0.50},
                  {"id": "B", "x": 0.40, "y": 0.50},
                  {"id": "D", "x": 0.75, "y": 0.50},
                  {"id": "F", "x": 0.40, "y": 0.80}],
        "edges": [{"id": "A_B", "from": "A", "to": "B", "bidirectional": True},
                  {"id": "B_D", "from": "B", "to": "D", "bidirectional": True},
                  {"id": "B_F", "from": "B", "to": "F", "bidirectional": True}],
        "start": {"type": "edge", "edge": "A_B", "from": "A", "to": "B"},
    })


def defer_graph() -> Graph:
    """A junction with two passages set symmetrically about straight on.

    Arriving along A_B the onward passages are twenty-five degrees to the left and
    twenty-five to the right, so geometry scores them identically and neither is preferred.
    That is a genuine ambiguity, and it is the case stage 10 is about.

    An earlier version of this test used a junction where the signal named a side that did
    not exist. Under the P08.1 rules that is no longer ambiguous: when the named side is
    absent, geometry has an unambiguous answer and going straight is simply correct, so
    that case now checks something different.
    """
    return Graph({
        "img_w": 1600, "img_h": 1000,
        "nodes": [{"id": "A", "x": 0.10, "y": 0.50},
                  {"id": "B", "x": 0.40, "y": 0.50},
                  {"id": "L", "x": 0.80, "y": 0.36},
                  {"id": "R", "x": 0.80, "y": 0.64}],
        "edges": [{"id": "A_B", "from": "A", "to": "B", "bidirectional": True},
                  {"id": "B_L", "from": "B", "to": "L", "bidirectional": True},
                  {"id": "B_R", "from": "B", "to": "R", "bidirectional": True}],
        "start": {"type": "edge", "edge": "A_B", "from": "A", "to": "B"},
    })


def burst(t: np.ndarray, yaw: np.ndarray, t0: float, dur: float, rate: float) -> None:
    """A stretch of turning at `rate` Hz, in place, so the shape of the trace is explicit.

    The sign follows the documented convention of the yaw signal, the one stated in
    `p07_fly_readout.py`: positive means the camera turned LEFT. This test was first written
    with the opposite reading, which is what the tracker used, so it had been confirming the
    inversion rather than catching it.
    """
    m = (t >= t0) & (t < t0 + dur)
    yaw[m] = rate


def edges_of(res: dict) -> list[str]:
    return [s["edge"] for s in res["edge_sequence"]]


def reasons(res: dict) -> list[str]:
    return [d["reason"] for d in res["decisions"]]


def main() -> None:
    fails: list[str] = []
    g = junction()
    gd = g.length_px("A_B")
    travel = gd / pix_per_sec_for(g)[0]
    print("P08 — обязательные случаи на синтетическом сигнале поворота")
    print(f"  узел B: B_C {(g.turn('A_B','B','B_C'))['side']}, "
          f"B_D {(g.turn('A_B','B','B_D'))['side']}, "
          f"B_F {(g.turn('A_B','B','B_F'))['side']}, "
          f"A_B {(g.turn('A_B','B','A_B'))['side']}")
    print(f"  ребро A_B {gd:.0f} px → проход {travel:.1f} с "
          f"при {pix_per_sec_for(g)[0]:.1f} px/с")

    print(f"  зона узла: последние {travel * 0.2:.1f} с; окно интегрирования 2.0 с")
    print()

    # ---- stage 23 ---------------------------------------------------------
    print("=== ЭТАП 23: ВЗГЛЯД ВПРАВО ПОСРЕДИ КОРИДОРА ===")
    print("  человек идёт A → B, на середине смотрит вправо и возвращает камеру")
    t, yaw, rel = blank(travel + 3.0)
    # a full out-and-back glance, well before the junction: out at 5.5 s, back at 6.5 s
    burst(t, yaw, 5.5, 0.5, +1.6)
    burst(t, yaw, 6.0, 0.5, -1.6)
    res = track(g, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл в окне на подходе: {d['yaw_integral']:+.3f} (порог 0.40)")
    print(f"  направление: '{d['male_direction'] or 'нет'}'")
    print(f"  последовательность рёбер: {edges_of(res)}")
    print(f"  решение в B: выбрано {d['chosen_edge']}, причина {d['reason']}")
    if d["reason"] not in ("STRAIGHT", "ONLY_OPTION"):
        fails.append("взгляд посреди коридора повлиял на решение в B")
    if d["chosen_edge"] == "B_F":
        fails.append("взгляд вправо увёл маршрут в правое ребро B_F")
    if d["male_direction"]:
        fails.append("взгляд посреди коридора приписал направление")
    print("  ожидалось: взгляд не меняет маршрут, направления нет")
    print()

    # ---- stage 24 ---------------------------------------------------------
    print("=== ЭТАП 24: НАСТОЯЩИЙ ПОВОРОТ ВПРАВО НА РАЗВИЛКЕ ===")
    print("  человек доходит до B и реально поворачивает вправо")
    print("  (по документированному соглашению поворот ВПРАВО это отрицательный интеграл)")
    t, yaw, rel = blank(travel + 3.0)
    burst(t, yaw, travel - 1.2, 1.6, -2.0)      # right turn, negative by convention
    res = track(g, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл в окне: {d['yaw_integral']:+.3f}")
    print(f"  мужское направление: {d['male_direction']}")
    print(f"  кандидаты и стороны: {d['candidate_edges']} / {d['candidate_sides']}")
    print(f"  вероятности: {d['probabilities']}")
    print(f"  выбрано: {d['chosen_edge']}, причина {d['reason']}, "
          f"уверенность {d['confidence']}")
    print(f"  последовательность рёбер: {edges_of(res)}")
    if d["chosen_edge"] != "B_F":
        fails.append(f"на развилке с поворотом вправо выбрано {d['chosen_edge']}, не B_F")
    if d["reason"] != "MALE_RIGHT":
        fails.append(f"причина решения {d['reason']}, ожидалось MALE_RIGHT")
    print("  ожидалось: выбран B_F, причина MALE_RIGHT")
    print()

    # ---- the mirror -------------------------------------------------------
    print("=== ТО ЖЕ ВЛЕВО ===")
    t, yaw, rel = blank(travel + 3.0)
    burst(t, yaw, travel - 1.2, 1.6, +2.0)
    res = track(g, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл {d['yaw_integral']:+.3f}, направление {d['male_direction']}, "
          f"выбрано {d['chosen_edge']} ({d['reason']})")
    if d["chosen_edge"] != "B_C":
        fails.append(f"на левый поворот выбрано {d['chosen_edge']}, не B_C")
    print()

    # ---- stages 14-15 -----------------------------------------------------
    print("=== ЭТАПЫ 14-15: ГДЕ ВЫБОРА НЕТ, МУХУ НЕ СПРАШИВАЮТ ===")
    t, yaw, rel = blank(travel + 3.0)
    burst(t, yaw, travel - 1.2, 1.6, -2.5)      # a strong signal that must be ignored
    res = track(corridor(), t, yaw, rel)
    d = res["decisions"][0]
    print(f"  коридор A—B—D, приходим в B при сильном сигнале {d['yaw_integral']:+.3f}")
    print(f"  кандидатов {len(d['candidate_edges'].split('|'))}, "
          f"выбрано {d['chosen_edge']}, причина {d['reason']}")
    print(f"  MaleCNS спрошен: {'да' if d['male_direction'] else 'нет'}")
    if d["reason"] != "ONLY_OPTION":
        fails.append(f"в коридоре причина {d['reason']}, ожидалось ONLY_OPTION")
    if d["male_direction"]:
        fails.append("в коридоре MaleCNS был спрошен, хотя выбора нет")
    print()

    # ---- stage 13 ---------------------------------------------------------
    print("=== ЭТАП 13: ОБЫЧНЫЙ ВЗГЛЯД НАЗАД НЕ РАЗВОРАЧИВАЕТ МАРШРУТ ===")
    t, yaw, rel = blank(travel + 3.0)
    # a brief left-right wiggle near the junction, but far below the reversal floor
    burst(t, yaw, travel - 1.0, 0.3, -1.5)
    burst(t, yaw, travel - 0.7, 0.3, +1.5)
    res = track(g, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл {d['yaw_integral']:+.3f}, порог разворота 1.20")
    print(f"  кандидаты: {d['candidate_edges']} (пройденное ребро "
          f"{'предложено' if 'A_B' in d['candidate_edges'] else 'НЕ предложено'})")
    print(f"  выбрано {d['chosen_edge']} ({d['reason']})")
    if "A_B" in d["candidate_edges"]:
        fails.append("пройденное ребро предложено при слабом сигнале")
    if d["chosen_edge"] == "A_B":
        fails.append("маршрут развернулся на слабом сигнале")
    print()

    # ---- dead end ---------------------------------------------------------
    print("=== ТУПИК: РАЗВОРОТ ОБЯЗАТЕЛЕН ===")
    t, yaw, rel = blank(20.0)
    res = track(g, t, yaw, rel,
                start={"type": "edge", "edge": "B_C", "from": "B", "to": "C"})
    d = res["decisions"][0]
    print(f"  идём B → C, в C выхода нет")
    print(f"  выбрано {d['chosen_edge']} ({d['reason']})")
    print(f"  последовательность: {edges_of(res)}")
    if d["reason"] != "BACK" or d["chosen_edge"] != "B_C":
        fails.append(f"в тупике выбрано {d['chosen_edge']} ({d['reason']}), ожидалось B_C/BACK")
    print()

    # ---- deferred ---------------------------------------------------------
    print("=== ЭТАП 10: ОТЛОЖЕННОЕ РЕШЕНИЕ ПРИ НЕУВЕРЕННОСТИ ===")
    print("  развилка с двумя проходами, симметричными относительно прямого:")
    print("  геометрия оценивает их одинаково, и выбрать между ними нечем")
    dg = defer_graph()
    t, yaw, rel = blank(40.0)
    res = track(dg, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл {d['yaw_integral']:+.3f}, направление {d['male_direction'] or 'нет'}")
    print(f"  кандидаты: {d['candidate_edges']} / {d['candidate_sides']}")
    print(f"  геометрия: {d['geometry_scores']}")
    print(f"  итоговые оценки: {d['final_scores']}")
    print(f"  вероятности: {d['probabilities']}")
    print(f"  выбрано {d['chosen_edge']}, причина {d['reason']}, "
          f"уверенность {d['confidence']}")
    reasons_all = reasons(res)
    print(f"  все причины: {reasons_all}")
    if d["reason"] not in ("DELAYED_DECISION", "RECOVERY"):
        fails.append(f"при симметричных проходах причина {d['reason']}, "
                     f"ожидалось DELAYED_DECISION или RECOVERY")
    if d["confidence"] > 0.75:
        fails.append("при симметричных проходах выдана высокая уверенность")
    print("  ожидалось: решение отложено, уверенность понижена")
    print()

    # ---- the named side does not exist ------------------------------------
    print("=== СИГНАЛ НАЗЫВАЕТ СТОРОНУ, КОТОРОЙ НЕТ ===")
    print("  раньше это считалось неоднозначностью. Теперь нет: если названной")
    print("  стороны нет, у геометрии есть однозначный ответ — идти дальше")
    nl = no_left_graph()
    t, yaw, rel = blank(travel + 3.0)
    burst(t, yaw, travel - 1.2, 2.0, +2.0)      # LEFT, and there is no left passage
    res = track(nl, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл {d['yaw_integral']:+.3f} ({d['male_direction']}), "
          f"кандидаты {d['candidate_edges']} / {d['candidate_sides']}")
    print(f"  геометрия {d['geometry_scores']}, итог {d['final_scores']}")
    print(f"  выбрано {d['chosen_edge']} ({d['reason']}), уверенность {d['confidence']}")
    if "LEFT" not in d["candidate_sides"].split("|"):
        if d["reason"] == "DELAYED_DECISION":
            fails.append("названной стороны нет и геометрия однозначна, "
                         "но решение отложено")
        if d["chosen_edge"] != "B_D":
            fails.append(f"названной стороны нет, выбрано {d['chosen_edge']}, ожидалось B_D")
    print()

    # ---- rules 3 and 4: a weak signal must not choose backwards -----------
    print("=== ПРАВИЛА 3-4: СЛАБЫЙ СИГНАЛ НЕ ВЫБИРАЕТ НАЗАД ===")
    print("  узел как J6: прямо-направо проход и второй почти строго назад")
    j6 = Graph({
        "img_w": 1600, "img_h": 1000,
        "nodes": [{"id": "J5", "x": 0.40, "y": 0.52},
                  {"id": "J6", "x": 0.44, "y": 0.50},
                  {"id": "M2", "x": 0.30, "y": 0.42},   # behind and to the left
                  {"id": "T12", "x": 0.44, "y": 0.47}],  # about ninety degrees right
        "edges": [{"id": "J5__J6", "from": "J5", "to": "J6", "bidirectional": True},
                  {"id": "M2__J6", "from": "M2", "to": "J6", "bidirectional": True},
                  {"id": "T12__J6", "from": "T12", "to": "J6", "bidirectional": True}],
        "start": {"type": "at_node", "node": "J6", "from_node": "J5"},
    })
    t, yaw, rel = blank(40.0)
    burst(t, yaw, 1.0, 0.3, -1.2)   # a weak wobble, well below the floor
    res = track(j6, t, yaw, rel)
    d = res["decisions"][0]
    print(f"  интеграл {d['yaw_integral']:+.3f}, порог {d['yaw_floor']}")
    print(f"  кандидаты {d['candidate_edges']} / {d['candidate_sides']}")
    print(f"  стороны: {d['candidate_sides']}")
    print(f"  геометрия: {d['geometry_scores']}  итог: {d['final_scores']}")
    print(f"  выбрано {d['chosen_edge']} ({d['reason']})")
    if "BACK" in d["candidate_sides"].split("|"):
        fails.append("при слабом сигнале назад-ведущий проход остался в кандидатах")
    if d["chosen_edge"] == "M2__J6":
        fails.append("при слабом сигнале выбран проход назад")
    print("  ожидалось: назад-ведущий проход убран, выбран боковой/прямой")
    print()

    print("=== ПРАВИЛА 6-7: СИЛЬНЫЙ СИГНАЛ ВЫБИРАЕТ, СЛАБЫЙ НЕТ ===")
    print("  настоящая развилка: прямо, налево и направо. Слабый сигнал должен")
    print("  оставить выбор геометрии, сильный — повернуть")
    for label, rate in (("слабый ", -0.10), ("средний", -1.0), ("сильный", -3.0)):
        t, yaw, rel = blank(travel + 3.0)
        burst(t, yaw, travel - 1.2, 2.0, rate)
        d = track(g, t, yaw, rel)["decisions"][0]
        print(f"  {label} I={d['yaw_integral']:+6.2f} порог {d['yaw_floor']:.2f} → "
              f"{d['male_direction'] or 'нет направления':>4s}  "
              f"итог {d['final_scores']:>22s}  выбрано {d['chosen_edge']} "
              f"({d['reason']})")
    print("  ожидалось: при слабом сигнале прямо, при сильном — направо")
    t, yaw, rel = blank(travel + 3.0)
    burst(t, yaw, travel - 1.2, 2.0, -0.10)
    weak = track(g, t, yaw, rel)["decisions"][0]
    t, yaw, rel = blank(travel + 3.0)
    burst(t, yaw, travel - 1.2, 2.0, -3.0)
    strong = track(g, t, yaw, rel)["decisions"][0]
    if weak["male_direction"]:
        fails.append("слабый сигнал всё же назвал направление")
    if weak["chosen_edge"] != "B_D":
        fails.append(f"слабый сигнал увёл в {weak['chosen_edge']}, ожидалось B_D")
    if strong["chosen_edge"] != "B_F" or strong["reason"] != "MALE_RIGHT":
        fails.append(f"сильный сигнал дал {strong['chosen_edge']} "
                     f"({strong['reason']}), ожидалось B_F / MALE_RIGHT")
    print()

    print("=== ИТОГ ===")
    known = [f for f in fails if f in KNOWN_PREEXISTING]
    fresh = [f for f in fails if f not in KNOWN_PREEXISTING]
    for f in fresh:
        print(f"  НЕ ПРОШЛО: {f}")
    for f in known:
        print(f"  ИЗВЕСТНОЕ, ДО P08.4B: {f}")
        print(f"    {KNOWN_PREEXISTING[f]}")
    if fresh:
        sys.exit(1)
    if known:
        print(f"  новых провалов нет; {len(known)} известный провал оставлен как есть")
    print("  взгляд посреди коридора маршрут не меняет")
    print("  настоящий поворот на развилке выбирает нужное ребро")
    print("  где выбора нет — MaleCNS не спрашивается")


if __name__ == "__main__":
    main()
