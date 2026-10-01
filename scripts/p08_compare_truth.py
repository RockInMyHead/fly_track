#!/usr/bin/env python3
"""
P08.2 — the route a person walked against the route the system built.

Nothing here tunes anything. The parameters of P08 are frozen, and this script only measures
what they produced. If the answer is bad, it is reported as bad; changing a constant to make
it look better is a different step and is not taken here.

The two routes
--------------
The person's route is a list of passages clicked in the order the walker entered them,
usually with the moment of entry read from the video. The system's route is
`output/p08/edge_sequence.csv`, produced by the frozen tracker.

Both are sequences of passages, so they can be compared directly, which a comparison of
coordinates could not do: the plan coordinates of the drawn route and the integrated
position of the system's route live in different spaces, but a passage is a passage.

What is counted
---------------
    how many passages match, aligned by longest common subsequence rather than by position,
    so one missed passage does not make everything after it count as wrong
    how many passages the person walked that the system never took
    how many the system took that the person never walked
    at every junction where the person chose, whether the system chose the same passage
    the first junction where it did not, and the reason the system gave there

The per-junction comparison is the one that matters most, because it does not depend on the
two routes staying in step: each choice is judged where it was made.

Usage:
    PYTHONPATH=. python scripts/p08_compare_truth.py
    PYTHONPATH=. python scripts/p08_compare_truth.py --limit 180
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import Graph  # noqa: E402

DEFAULT_TRUTH = ROOT / "data/p08/truth_route.json"
P08_OUT = ROOT / "output/p08"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def load_truth(path: Path | str = DEFAULT_TRUTH) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if doc.get("edges"):
        return doc
    # an earlier version of the editor stored node names; convert when possible so that a
    # drawing made before the change is not lost
    nodes = doc.get("nodes") or []
    if len(nodes) >= 2:
        return {"edges": [{"edge": None, "from": a, "to": b, "t": None}
                          for a, b in zip(nodes, nodes[1:])],
                "converted_from_nodes": True}
    return doc


def lcs(a: list, b: list) -> list[tuple[int, int]]:
    """Longest common subsequence as index pairs, so alignment is by content not position.

    A full table is built rather than rolling rows. The earlier rolling version had the
    backtracking indices off by one — `table[i][j-1]` indexed one row past the end — and
    with a few hundred passages each the table costs a megabyte, which is not worth saving
    at the price of getting the alignment wrong.
    """
    n, m = len(a), len(b)
    if not n or not m:
        return []
    # dp[i][j] = length of the longest common subsequence of a[:i] and b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev = dp[i], dp[i - 1]
        ai = a[i - 1]
        for j in range(1, m + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(out))


def first_divergence(truth_e: list[str], p08_e: list[str]) -> int:
    k = 0
    while k < min(len(truth_e), len(p08_e)) and truth_e[k] == p08_e[k]:
        k += 1
    return k


def junction_choices(g: Graph, truth: list[dict], p08: list[dict],
                     decisions: list[dict]) -> dict:
    """Every place the person chose, and what the system chose when it arrived there.

    A choice is a change of passage at a node: truth[i-1] ends at a node and truth[i]
    starts there. Where the two consecutive passages are the same edge walked back, there
    was no choice to make, and those are counted separately rather than as successes.
    """
    rows = []
    for i in range(1, len(truth)):
        a, b = truth[i - 1], truth[i]
        node = a["to"]
        if b["from"] != node:
            rows.append({"at": node, "truth_next": b["edge"], "status": "разрыв в разметке",
                         "p08_next": None})
            continue
        if b["edge"] == a["edge"]:
            rows.append({"at": node, "truth_next": b["edge"], "status": "разворот",
                         "p08_next": None})
            continue

        # what the system did the first time it stood at this node having arrived by `a`
        got = None
        for d in decisions:
            if d["node"] != node:
                continue
            t_dec = float(d["time"])
            t_truth = a.get("t")
            if t_truth is not None and abs(t_dec - float(t_truth)) > 45.0:
                continue
            got = d
            break
        if got is None:
            rows.append({"at": node, "truth_next": b["edge"], "status": "система сюда не пришла",
                         "p08_next": None})
            continue

        ok = got["chosen_edge"] == b["edge"]
        # reasons that name what decided it: the fly, the geometry, or the memory
        why = got["reason"]
        if why == "DELAYED_DECISION":
            why = "DELAYED_DECISION"
        try:
            sig = float(got.get("signal_strength") or 0)
            flo = float(got.get("yaw_floor") or 1)
        except ValueError:
            sig, flo = 0.0, 1.0
        driver = ("MaleCNS" if why.startswith("MALE") else
                  "BACK" if why == "BACK" else
                  "только один выход" if why == "ONLY_OPTION" else
                  "геометрия" if why in ("STRAIGHT", "GEOMETRY", "NO_MATCH") else
                  "отложено" if why in ("DELAYED_DECISION", "RECOVERY") else why)
        if "novelty" in (got.get("novelty_scores") or "") and not ok:
            pass
        rows.append({
            "at": node, "truth_next": b["edge"], "p08_next": got["chosen_edge"],
            "ok": bool(ok), "status": "верно" if ok else "ошибка",
            "reason": got["reason"], "driver": driver,
            "signal": sig, "floor": flo,
            "arrival_edge": got.get("current_edge", ""),
            "chosen_side": None, "truth_side": None,
            "time": float(got["time"]),
            "candidate_edges": got.get("candidate_edges", ""),
            "candidate_angles": got.get("candidate_angles", ""),
            "candidate_sides": got.get("candidate_sides", ""),
            "geometry_scores": got.get("geometry_scores", ""),
            "novelty_scores": got.get("novelty_scores", ""),
            "visit_counts": got.get("visit_counts", ""),
            "back_penalties": got.get("back_penalties", ""),
            "final_scores": got.get("final_scores", ""),
            "male_direction": got.get("male_direction", ""),
        })
    return rows


def analyse(g: Graph, truth: list[dict], seq: list[dict], dec: list[dict],
            limit: float | None = None) -> tuple[dict, list[str]]:
    """The comparison and the sentences describing it, shared by the CLI and the page."""
    lines: list[str] = []

    def say(t=""):
        lines.append(t)

    truth = [dict(r) for r in truth]
    truth_e = [r["edge"] for r in truth]
    if limit is not None:
        seq = [x for x in seq if float(x["time_start"]) <= limit]
        dec = [x for x in dec if float(x["time"]) <= limit]
    p08_e = [x["edge"] for x in seq]
    ttime = [r.get("t") for r in truth if r.get("t") is not None]

    say(f"эталон: {len(truth_e)} рёбер"
        + (f", времена входа записаны ({len(ttime)} из {len(truth)})" if ttime
           else ", времён входа нет"))
    say(f"P08:    {len(p08_e)} рёбер" + (f" (первые {limit:.0f} с)" if limit else ""))
    say()

    pairs = lcs(truth_e, p08_e)
    matched = len(pairs)
    truth_idx = {i for i, _ in pairs}
    p08_idx = {j for _, j in pairs}
    missed = [truth_e[i] for i in range(len(truth_e)) if i not in truth_idx]
    extra = [p08_e[j] for j in range(len(p08_e)) if j not in p08_idx]
    k = first_divergence(truth_e, p08_e)

    say("СОВПАДЕНИЕ ПО РЁБРАМ")
    say(f"  совпало {matched} из {len(truth_e)} рёбер эталона "
        f"({matched / max(len(truth_e), 1) * 100:.0f}%)")
    say(f"  пропущено системой: {len(missed)}" + (f"  {missed[:8]}" if missed else ""))
    say(f"  лишних у системы: {len(extra)}" + (f"  {extra[:8]}" if extra else ""))
    say()

    say("ПЕРВОЕ РАСХОЖДЕНИЕ")
    if k >= min(len(truth_e), len(p08_e)):
        say("  последовательности совпадают на всей длине эталона")
    else:
        say(f"  позиция {k}: эталон {truth_e[k]}, система {p08_e[k]}")
        if k == 0:
            say("  расхождение с первого ребра: вероятно, не совпало направление")
    say()

    jc = junction_choices(g, truth, seq, dec)
    n_ok = sum(1 for r in jc if r.get("ok"))
    n_choice = sum(1 for r in jc if r["status"] in ("верно", "ошибка"))
    n_none = sum(1 for r in jc if r["status"] == "система сюда не пришла")
    n_rev = sum(1 for r in jc if r["status"] == "разворот")
    say("ВЫБОР НА РАЗВИЛКАХ")
    say(f"  разворотов в эталоне: {n_rev} (выбора нет)")
    say(f"  мест, где человек выбирал: {n_choice}")
    say(f"  верно выбрано: {n_ok}"
        + (f" ({n_ok / n_choice * 100:.0f}%)" if n_choice else ""))
    if n_none:
        say(f"  мест, куда система не дошла: {n_none}")
    say()

    wrong = [r for r in jc if r["status"] == "ошибка"]
    if wrong:
        # Every passage on offer at the first wrong junction, with everything that went into
        # the choice. This is the whole point of the stage: not that the system went wrong,
        # but which term made it go wrong.
        f0 = wrong[0]
        say("ВСЕ ВАРИАНТЫ НА ПЕРВОЙ ОШИБКЕ")
        say(f"  узел {f0['at']}, пришёл по {f0.get('arrival_edge') or '—'}, "
            f"время {f0['time']:.1f} с")
        cands = [c for c in (f0.get("candidate_edges") or "").split("|") if c]
        angs = [a for a in (f0.get("candidate_angles") or "").split("|") if a]
        sides = [x for x in (f0.get("candidate_sides") or "").split("|") if x]
        geos = [x for x in (f0.get("geometry_scores") or "").split("|") if x]
        novs = [x for x in (f0.get("novelty_scores") or "").split("|") if x]
        vis = [x for x in (f0.get("visit_counts") or "").split("|") if x]
        back = [x for x in (f0.get("back_penalties") or "").split("|") if x]
        fins = [x for x in (f0.get("final_scores") or "").split("|") if x]

        def at(lst, i):
            return lst[i] if i < len(lst) else "—"

        say(f"  {'ребро':>16s} {'угол':>6s} {'сторона':>9s} {'геом.':>7s} "
            f"{'новизна':>8s} {'хожено':>7s} {'штраф':>7s} {'итог':>7s}  кем выбран")
        for i, e in enumerate(cands):
            who = []
            if e == f0["truth_next"]:
                who.append("ЧЕЛОВЕК")
            if e == f0["p08_next"]:
                who.append("P08")
            say(f"  {e:>16s} {at(angs, i):>6s} {at(sides, i):>9s} {at(geos, i):>7s} "
                f"{at(novs, i):>8s} {at(vis, i):>7s} {at(back, i):>7s} "
                f"{at(fins, i):>7s}  {', '.join(who)}")
        say()
        # what carried the wrong choice: the largest term in its final score
        try:
            i_wrong = cands.index(f0["p08_next"])
            i_true = cands.index(f0["truth_next"])
            gw, gt = float(at(geos, i_wrong)), float(at(geos, i_true))
            nw, nt = float(at(novs, i_wrong)), float(at(novs, i_true))
            say("ПОЧЕМУ ВЫИГРАЛО НЕПРАВИЛЬНОЕ")
            say(f"  геометрия: у P08 {gw:.3f}, у человека {gt:.3f}"
                + ("   ← тут P08 имел преимущество" if gw > gt + 0.02 else
                   "   ← преимущества не дала" if abs(gw - gt) <= 0.02 else
                   "   ← человек имел преимущество"))
            say(f"  новизна:   у P08 {nw:.3f}, у человека {nt:.3f}"
                + ("   ← тут P08 имел преимущество" if nw > nt + 0.02 else
                   "   ← преимущества не дала" if abs(nw - nt) <= 0.02 else
                   "   ← человек имел преимущество"))
            say(f"  сигнал: {float(f0['signal']):.3f} при пороге {float(f0['floor']):.3f}"
                + (f", направление {f0.get('male_direction') or '—'}"
                   if f0.get("male_direction") is not None else "")
                + ("   ← сигнал молчал" if float(f0["signal"]) < float(f0["floor"])
                   else "   ← сигнал был выше порога"))
            say(f"  решило: {f0['driver']} (причина {f0['reason']})")
        except (ValueError, IndexError):
            pass
        say()
        say("ОШИБКИ НА РАЗВИЛКАХ")
        for r in wrong[:15]:
            say(f"  {r['at']}: истина {r['truth_next']}, P08 {r['p08_next']}, "
                f"решило «{r['driver']}», сигнал {r['signal']:.3f} при пороге {r['floor']:.3f}")
        tally: dict[str, int] = {}
        for r in wrong:
            tally[r["driver"]] = tally.get(r["driver"], 0) + 1
        say("  что решало в ошибочных случаях:")
        for k2, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            say(f"    {k2}: {v}")
        f = wrong[0]
        say()
        say("ПЕРВАЯ ОШИБКА ПОДРОБНО")
        say(f"  узел {f['at']}, время {f['time']:.1f} с")
        say(f"  человек пошёл в {f['truth_next']}, система в {f['p08_next']}")
        say(f"  причина: {f['reason']} ({f['driver']})")
        say(f"  сигнал {f['signal']:.3f} при пороге {f['floor']:.3f}")
        say(f"  кандидаты:       {f['candidate_edges']}")
        say(f"  геометрия:       {f['geometry_scores']}")
        say(f"  новизна:         {f['novelty_scores']}")
        say(f"  сколько хожено:  {f['visit_counts']}")
        say(f"  итоговые оценки: {f['final_scores']}")
    else:
        say("ОШИБОК НА РАЗВИЛКАХ НЕТ")
        say("  на всех развилках, где человек выбирал и куда система дошла,")
        say("  выбран тот же проход")

    rep = {
        "frozen": True,
        "truth_edges": len(truth_e), "p08_edges": len(p08_e),
        "has_times": bool(ttime),
        "matched": matched, "missed": missed, "extra": extra,
        "first_divergence": k,
        "junctions_total": n_choice, "junctions_correct": n_ok,
        "junctions_never_reached": n_none, "u_turns": n_rev,
        "junction_rows": jc,
        "truth_sequence": truth_e, "p08_sequence": p08_e,
        "ok": True,
    }
    return rep, lines


def draw_comparison(g: Graph, truth: list[dict], seq: list[dict], out: Path,
                    junction_rows: list[dict] | None = None) -> None:
    """The plan with both routes on it, which is what makes a wrong choice visible.

    The two are drawn on the same axes here, unlike the P07 comparison, because both are
    expressed in the plan's own coordinates: the drawn route interpolates along the graph
    and the system's route does the same, so the two lines mean the same kind of thing and
    can be laid over each other. Where they separate is where the routes disagree.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_path = ROOT / "data/p08" / str(g.doc.get("image") or "plan.png")
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    for ax, (title, show_truth, show_p08) in zip(axes, (
            ("эталон (нарисовано человеком)", True, False),
            ("P08 (замороженные параметры)", True, True))):
        if img_path.exists():
            ax.imshow(plt.imread(img_path), alpha=0.5)
        for e in g.edges.values():
            a, b = g.pos(e["from"]), g.pos(e["to"])
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#cbd5e1", lw=0.5, alpha=0.5, zorder=1)
        if show_truth:
            xs, ys = [], []
            for r in truth:
                a, b = g.pos(r["from"]), g.pos(r["to"])
                xs += [a[0], b[0]]
                ys += [a[1], b[1]]
            ax.plot(xs, ys, color="#2563eb", lw=3.0, alpha=0.9, zorder=3,
                    label="эталон")
        if show_p08:
            xs, ys = [], []
            for x in seq:
                a, b = g.pos(x["from"]), g.pos(x["to"])
                xs += [a[0], b[0]]
                ys += [a[1], b[1]]
            ax.plot(xs, ys, color="#dc2626", lw=1.6, alpha=0.8, zorder=2,
                    label="P08")
        # where the two routes disagree: the first one large, the rest small, so the eye goes
        # to the first wrong choice rather than to the seventh
        if show_p08 and junction_rows:
            bad = [r for r in junction_rows if r.get("status") == "ошибка"]
            for k, r in enumerate(bad):
                n = g.nodes.get(r["at"])
                if not n:
                    continue
                x, y = g.pos(r["at"])
                ax.plot([x], [y], "o", ms=22 if k == 0 else 9,
                        mec="black", mew=2 if k == 0 else 1,
                        mfc="#f59e0b" if k == 0 else "#ef4444", zorder=7)
                if k == 0:
                    ax.annotate("первая ошибка", (x, y), textcoords="offset points",
                                xytext=(18, 18), fontsize=13, zorder=8,
                                bbox=dict(fc="#fef3c7", ec="#b45309"))
        if show_truth and show_p08:
            ax.legend(loc="lower left", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("P08.2 — эталонный маршрут и маршрут системы. "
                 "Синее нарисовано человеком, красное построено по видео",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "truth_vs_p08.png", dpi=105)
    plt.close(fig)
    print(f"  рисунок: {out}/truth_vs_p08.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(DEFAULT_TRUTH))
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--out", default=str(P08_OUT))
    ap.add_argument("--limit", type=float, default=None,
                    help="only compare the first this many seconds")
    args = ap.parse_args()

    out = Path(args.out)
    g = Graph.load(args.graph)
    doc = load_truth(args.truth)

    print("P08.2 — нарисованный маршрут против маршрута P08")
    print("  параметры P08 заморожены, ничего не подбирается")
    print()

    truth = doc.get("edges") or []
    if len(truth) < 2:
        print("  нарисованный маршрут пуст или слишком короткий.")
        print(f"  ожидается {args.truth} со списком рёбер.")
        print("  размечайте в закладке ТРЕКИНГ: смотрите видео и кликайте ребро")
        print("  в момент, когда человек в него входит.")
        return

    broken = []
    for i, r in enumerate(truth):
        if not r.get("edge"):
            for eid in g.edges_at(r["from"]):
                if g.other(eid, r["from"]) == r["to"]:
                    r["edge"] = eid
                    break
        if not r.get("edge"):
            broken.append(i)
    missing = [r for r in truth if not g.edges.get(r["edge"] or "")]
    if broken or missing:
        print(f"  в разметке есть рёбра, которых нет в графе: "
              f"{[r.get('edge') for r in missing][:5]} — сравнение невозможно")
        return

    seq = read_csv(out / "edge_sequence.csv")
    dec = read_csv(out / "decisions.csv")
    if not seq:
        print("  нет edge_sequence.csv — сначала прогоните трекер")
        return

    rep, lines = analyse(g, truth, seq, dec, args.limit)
    for line in lines:
        print(f"  {line}" if line else "")

    (out / "truth_comparison.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_comparison(g, truth, seq, out, rep.get("junction_rows") or [])
    with (out / "truth_edge_sequence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order", "edge", "from_node", "to_node",
                                          "time_start", "time_end"])
        w.writeheader()
        for i, r in enumerate(truth):
            t = r.get("t")
            w.writerow({"order": i + 1, "edge": r["edge"], "from_node": r["from"],
                        "to_node": r["to"],
                        "time_start": "" if t is None else f"{float(t):.2f}",
                        "time_end": ""})
    with (out / "truth_edges.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i", "edge", "from", "to", "t"])
        w.writeheader()
        for i, r in enumerate(truth):
            w.writerow({"i": i, "edge": r["edge"], "from": r["from"], "to": r["to"],
                        "t": "" if r.get("t") is None else f"{float(r['t']):.2f}"})
    print("\nWrote " + str(out) + "/truth_comparison.json and truth_edges.csv")


if __name__ == "__main__":
    main()
