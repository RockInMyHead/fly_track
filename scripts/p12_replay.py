#!/usr/bin/env python3
"""P12 step 2 — offline replay: what the three channels do to the route decisions, without retuning.

P08.6 already builds several hypotheses at a junction and scores them against the fly's own turns.
P12 does not replace that. It adds three separate observations, each with its own confidence, and
lets them modify the ranking in ways that are named and reported one by one rather than summed into
a single score. The forbidden move is a weighted total of the three; the allowed one is three
distinguishable effects on the same list of hypotheses.

    camera_yaw         LEFT / RIGHT / SILENT        no effect on its own — see below
    route_change       SAME / DIFFERENT / UNKNOWN   does the body's course change?
    net_displacement   MOVE / NO_NET / UNKNOWN      was there net progress?

HOW EACH ONE ACTS
-----------------
    route_change = SAME        the camera may have swung, but the body's course did not change, so
                               a hypothesis that requires a large turn is the wrong reading of it.
                               Routes that carry straight on are preferred.
    route_change = DIFFERENT   the course did change, so a hypothesis that turns is credible.
    yaw + route_change = DIFFERENT  a turn in the direction the camera named becomes a real
                               candidate.
    yaw + route_change = SAME  the camera named a direction and the body did not follow. This is
                               the case the whole phase exists for: the yaw is a glance, and it is
                               not allowed to steer the route.
    net_displacement = NO_NET  the fly moved but made no net progress, so a hypothesis that walks
                               a long way is not supported. Shorter, returning routes are.
    any channel UNKNOWN        contributes nothing. The junction is left as ambiguous as P08.6
                               left it, which is the honest outcome when the evidence is thin.

Every term is reported per hypothesis, so a reader can see which channel moved a decision and by
how much. Weights are fixed and printed; the report also shows the outcome at zero weight for each
channel, which is what P08.6 gives.

Usage:
    PYTHONPATH=. python scripts/p12_replay.py [--w 0.25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "output/p12"
TURN_FOR_FULL = 90.0      # the turn, in degrees, that counts as "fully a turn"
DIST_FOR_FULL = 10.0      # metres of walking that count as "fully far"


def side_of(deg: float) -> str:
    return "LEFT" if deg > 0 else "RIGHT"


def terms_for(hyp: dict, rc: dict, nd: dict, yaw: dict, w: float) -> dict:
    """The three named contributions to one hypothesis. None of them is folded into another."""
    t_rc = t_nd = t_yaw = 0.0
    why = []

    deg = float(hyp["deg"])
    turn_frac = min(abs(deg) / TURN_FOR_FULL, 1.0)

    if rc.get("class") == "SAME":
        t_rc = -w * turn_frac
        if t_rc:
            why.append("route_change=SAME: поворот не поддержан")
    elif rc.get("class") == "DIFFERENT":
        t_rc = +w * turn_frac
        if t_rc:
            why.append("route_change=DIFFERENT: поворот поддержан")

    if nd.get("class") == "NO_NET":
        dist = float(hyp.get("distance_m", 0.0))
        t_nd = -w * min(dist / DIST_FOR_FULL, 2.0)
        if t_nd:
            why.append("net=NO_NET: далёкий маршрут не поддержан")

    # yaw only acts through route_change: a named direction with a body that did not follow is a
    # glance, and a glance must not steer. That is the case this phase was built for.
    if yaw.get("class") and rc.get("class") == "DIFFERENT":
        if side_of(deg) == yaw["class"]:
            t_yaw = +w
            why.append(f"yaw={yaw['class']} и курс изменился: совпадение")
        else:
            t_yaw = -w
            why.append(f"yaw={yaw['class']}, а поворот в {side_of(deg)}: расхождение")

    return {"route_change": t_rc, "net_displacement": t_nd, "yaw_with_route": t_yaw,
            "total": t_rc + t_nd + t_yaw, "why": why}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w", type=float, default=0.25, help="вес каждого канала")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--p086", default=str(ROOT / "output/p086/report.json"))
    ap.add_argument("--channels", default=None)
    args = ap.parse_args()
    out = Path(args.out)

    ch = json.loads((args.channels or (out / "channels.json")).read_text(encoding="utf-8"))
    p86 = json.loads(Path(args.p086).read_text(encoding="utf-8"))
    W = args.w

    print("=" * 108)
    print("P12 шаг 2 — OFFLINE REPLAY: что три канала делают с решениями маршрута")
    print("=" * 108)
    print(f"  вес канала {W}; слагаемые НЕ складываются в один score, а выводятся отдельно")
    print(f"  развилки и гипотезы берутся из P08.6 без единого изменения")
    print()

    rows = []
    print("─── РЕШЕНИЯ ПО РАЗВИЛКАМ ───")
    print(f"  {'узел':<6}{'P08.6 выбр.':<16}{'P12 выбр.':<16}{'истина':<16}"
          f"{'каналы':<34}{'итог'}")
    for j in p86["junctions"]:
        node = j["node"]
        c = ch["junctions"].get(node)
        if c is None:
            continue
        yaw, rc, nd = c["camera_yaw"], c["route_change"], c["net_displacement"]
        hyps = j.get("hypotheses", [])
        if not hyps:
            continue
        for h in hyps:
            h["terms"] = terms_for(h, rc, nd, yaw, W)
            h["with_p12"] = float(h["score"]) + h["terms"]["total"]
        base = max(hyps, key=lambda h: float(h["score"]))
        new = max(hyps, key=lambda h: h["with_p12"])
        truth = j.get("truth_edge", "")
        chans = (f"yaw={yaw.get('class') or '—'} "
                 f"rc={rc.get('class') or '—'} nd={nd.get('class') or '—'}")
        mark = ""
        if base["edge"] != new["edge"]:
            mark = "  <- ИЗМЕНИЛОСЬ"
        if truth:
            mark += f"  [{('верно' if new['edge']==truth else 'НЕВЕРНО')}]"
        print(f"  {node:<6}{base['edge']:<16}{new['edge']:<16}{truth or '—':<16}"
              f"{chans:<34}{mark}")
        rows.append({"node": node, "t": j["t"], "yaw": yaw, "route_change": rc,
                     "net_displacement": nd, "pick_p086": base["edge"],
                     "pick_p12": new["edge"], "truth": truth,
                     "changed": base["edge"] != new["edge"],
                     "correct_p086": bool(truth and base["edge"] == truth),
                     "correct_p12": bool(truth and new["edge"] == truth),
                     "hypotheses": [{"edge": h["edge"], "deg": h["deg"],
                                     "score_p086": h["score"], "terms": h["terms"],
                                     "score_p12": round(h["with_p12"], 4),
                                     "distance_m": h.get("distance_m")} for h in hyps]})
    print()

    # ---- the two junctions the phase names ----
    print("─── ДВЕ РАЗВИЛКИ, НАЗВАННЫЕ В ЗАДАНИИ ───")
    for node in ("J35", "T50"):
        r = next((x for x in rows if x["node"] == node), None)
        if not r:
            print(f"  {node}: нет данных")
            continue
        print(f"  {node} (t={r['t']:.1f})")
        print(f"    каналы: yaw={r['yaw'].get('class') or 'молчит'} "
              f"({r['yaw'].get('confidence',0):.2f}), "
              f"route_change={r['route_change'].get('class') or 'UNKNOWN'} "
              f"({r['route_change'].get('confidence',0):.2f}), "
              f"net={r['net_displacement'].get('class') or 'UNKNOWN'} "
              f"({r['net_displacement'].get('confidence',0):.2f})")
        print(f"    P08.6 выбрал {r['pick_p086']}, P12 выбирает {r['pick_p12']}, "
              f"истина {r['truth'] or '—'}")
        for h in r["hypotheses"]:
            t = h["terms"]
            print(f"      {h['edge']:<16} deg={h['deg']:>7.1f}  P08.6={h['score_p086']:>7.3f}  "
                  f"rc={t['route_change']:+.3f} nd={t['net_displacement']:+.3f} "
                  f"yaw={t['yaw_with_route']:+.3f}  P12={h['score_p12']:>7.3f}")
            for w_ in t["why"]:
                print(f"          {w_}")
        print()

    # ---- how the outcome depends on the weight ----
    print("─── ЧУВСТВИТЕЛЬНОСТЬ К ВЕСУ (сколько решений меняется) ───")
    print(f"  {'вес':>6}{'изменилось':>12}{'верных P08.6':>14}{'верных P12':>12}")
    for w in (0.0, 0.1, 0.25, 0.5, 1.0):
        n_ch = n_c08 = n_c12 = 0
        n_tot = 0
        for j in p86["junctions"]:
            c = ch["junctions"].get(j["node"])
            hyps = j.get("hypotheses", [])
            if c is None or not hyps:
                continue
            yaw, rc, nd = c["camera_yaw"], c["route_change"], c["net_displacement"]
            sc = [(float(h["score"]) + terms_for(h, rc, nd, yaw, w)["total"], h) for h in hyps]
            base = max(hyps, key=lambda h: float(h["score"]))
            new = max(sc, key=lambda x: x[0])[1]
            truth = j.get("truth_edge", "")
            if not truth:
                continue
            n_tot += 1
            n_ch += int(base["edge"] != new["edge"])
            n_c08 += int(base["edge"] == truth)
            n_c12 += int(new["edge"] == truth)
        print(f"  {w:>6.2f}{n_ch:>12}{n_c08:>10}/{n_tot}{n_c12:>10}/{n_tot}")
    print()

    (out / "replay.json").write_text(json.dumps({
        "phase": "P12 шаг 2 — offline replay",
        "weight": W, "junctions": rows,
        "note": ("слагаемые каналов выводятся отдельно; нигде не считается их сумма как "
                 "единственный score маршрута"),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"  записано: {out/'replay.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
