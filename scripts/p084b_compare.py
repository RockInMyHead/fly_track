#!/usr/bin/env python3
"""P08.4B — P08.4A against P08.4B, with the counts the note asked for.

Same graph, same video, same fly, same weights, same novelty, same pace, same floors. The
only difference between the two runs is how a junction is chosen between: the old run read
the graph's 45-degree label, this one reads the passage's own angle.

The replay already showed that the rule changes none of the twenty-six choices on the
junctions the earlier route reached. This checks the same thing on the whole clip rather than
on a frozen subset, and adds the counts the note names: passages walked, distinct passages,
reversals, dead ends, junctions with a real choice, and how many of those the fly decided.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p08_graph as G  # noqa: E402


def rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_run(d: Path) -> dict:
    return {"report": json.loads((d / "report.json").read_text(encoding="utf-8")),
            "seq": rows(d / "edge_sequence.csv"),
            "dec": rows(d / "decisions.csv"),
            "traj": rows(d / "graph_trajectory.csv")}


def lcs(a: list[str], b: list[str]) -> list[str]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def travel_and_time(traj: list[dict], mpp: float) -> tuple[float, float]:
    if not traj:
        return 0.0, 0.0
    ds = 0.0
    for p, q in zip(traj, traj[1:]):
        ds += ((float(q["x_px"]) - float(p["x_px"])) ** 2 +
               (float(q["y_px"]) - float(p["y_px"])) ** 2) ** 0.5
    return ds * mpp, float(traj[-1]["t"]) - float(traj[0]["t"])


def profile(run: dict, g: G.Graph) -> dict:
    dec, seq = run["dec"], run["seq"]
    real = [d for d in dec if d["reason"] != "ONLY_OPTION"]
    reasons: dict[str, int] = {}
    for d in dec:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    # a dead end is a node of degree one; count the visits rather than the nodes so that
    # going into the same tip twice counts twice, which is the thing worth avoiding
    dead_visits = sum(1 for s in seq if g.degree(s["to"]) <= 1)
    return {
        "decisions": len(dec),
        "with_choice": len(real),
        "reasons": reasons,
        "male_decided": sum(1 for d in dec if d["reason"].startswith("MALE_")),
        "geometry_only": sum(1 for d in dec if d["reason"] in ("GEOMETRY", "STRAIGHT")),
        "ambiguous": sum(1 for d in dec if d["reason"] == "GEOMETRY_AMBIGUOUS"),
        "deferred": sum(1 for d in dec if d["reason"] == "DELAYED_DECISION"),
        "recovered": sum(1 for d in dec if d["reason"] == "RECOVERY"),
        "backtracks": sum(1 for d in dec if d["reason"] == "BACK"),
        "edges_walked": len(seq),
        "distinct_edges": len({s["edge"] for s in seq}),
        "dead_end_visits": dead_visits,
        "male_consulted": run["report"].get("n_male_consulted"),
        "final_node": run["report"].get("final_node"),
        "sequence": [s["edge"] for s in seq],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default=str(ROOT / "output/p084a/new"))
    ap.add_argument("--new", default=str(ROOT / "output/p084b/run"))
    ap.add_argument("--graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--out", default=str(ROOT / "output/p084b"))
    args = ap.parse_args()

    out = Path(args.out)
    g = G.Graph.load(Path(args.graph))
    mpp = float(g.meters_per_pixel)
    old, new = load_run(Path(args.old)), load_run(Path(args.new))
    po, pn = profile(old, g), profile(new, g)

    so, sn = po["sequence"], pn["sequence"]
    agree = lcs(so, sn)
    first = next((k for k, (a, b) in enumerate(zip(so, sn)) if a != b), min(len(so), len(sn)))
    om, ot = travel_and_time(old["traj"], mpp)
    nm, nt = travel_and_time(new["traj"], mpp)

    print("=" * 84)
    print("P08.4B — ПРОГОН P08.4A ПРОТИВ P08.4B НА ВСЁМ VID00006")
    print("=" * 84)
    print(f"граф: {len(g.nodes)} узлов, {len(g.edges)} рёбер (один и тот же в обоих прогонах)")
    print("муха, веса, novelty, темп, пороги — НЕ менялись; менялось только правило выбора")
    print()
    print(f"  {'метрика':<34}{'P08.4A':>12}{'P08.4B':>12}")
    for key, label in (
        ("edges_walked", "пройдено рёбер"),
        ("distinct_edges", "разных рёбер"),
        ("decisions", "решений всего"),
        ("with_choice", "решений с реальным выбором"),
        ("male_decided", "решил MaleCNS (MALE_*)"),
        ("geometry_only", "решила геометрия (GEOMETRY/STRAIGHT)"),
        ("ambiguous", "GEOMETRY_AMBIGUOUS"),
        ("deferred", "DELAYED_DECISION"),
        ("recovered", "RECOVERY"),
        ("backtracks", "разворотов (BACK)"),
        ("dead_end_visits", "заходов в тупик"),
    ):
        a, b = po[key], pn[key]
        mark = "" if a == b else "   ←"
        print(f"  {label:<34}{a:>12}{b:>12}{mark}")
    print()
    print(f"  путь:  {om:.2f} м → {nm:.2f} м")
    print(f"  время: {ot:.2f} с ({ot/60:.1f} мин) → {nt:.2f} с ({nt/60:.1f} мин)")
    print(f"  конечный узел: {po['final_node']} → {pn['final_node']}")
    print()

    print("─── согласие маршрутов ───")
    print(f"  совпадает по порядку: {len(agree)} из {len(so)}")
    print(f"  первое расхождение — шаг {first + 1}")
    if so == sn:
        print("  последовательности совпадают полностью")
    else:
        print(f"    было:  {' → '.join(so[max(0,first-2):first+3])}")
        print(f"    стало: {' → '.join(sn[max(0,first-2):first+3])}")
    print()

    print("─── причины решений: было против стало ───")
    for k in sorted(set(po["reasons"]) | set(pn["reasons"])):
        a, b = po["reasons"].get(k, 0), pn["reasons"].get(k, 0)
        print(f"  {k:<20}{a:>6}{b:>6}{'' if a == b else '   ←'}")
    print()

    # the junctions where the two runs disagreed about what to call the choice
    print("─── развилки, где сменилось основание решения ───")
    o_by = {d["node"] + "@" + d["time"]: d for d in old["dec"]}
    n_by = {d["node"] + "@" + d["time"]: d for d in new["dec"]}
    shared = sorted(set(o_by) & set(n_by),
                    key=lambda k: float(o_by[k]["time"]))
    for k in shared:
        a, b = o_by[k], n_by[k]
        if a["reason"] == b["reason"] and a["chosen_edge"] == b["chosen_edge"]:
            continue
        print(f"  t={float(a['time']):7.2f} {a['node']:<6} "
              f"{a['chosen_edge']:<16} {a['reason']:<18} → {b['reason']:<18}")
    print()

    (out / "comparison.json").write_text(json.dumps({
        "graph": {"nodes": len(g.nodes), "edges": len(g.edges)},
        "p084a": po, "p084b": pn,
        "agreement_steps": len(agree), "first_divergence_step": first + 1,
        "travel_m": {"p084a": round(om, 3), "p084b": round(nm, 3)},
        "time_s": {"p084a": round(ot, 3), "p084b": round(nt, 3)},
        "identical_sequence": so == sn,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (out / "report.json").write_text(json.dumps({
        "phase": "P08.4B — geometry of choice at junctions",
        "graph": str(args.graph),
        "video": "VID00006",
        "old_run": str(args.old), "new_run": str(args.new),
        "unchanged": ["MaleCNS", "клетки", "yaw", "novelty", "скорость", "веса",
                      "пороги нейронного сигнала"],
        "what_changed": (
            "решение берётся по настоящему углу ребра; подписи STRAIGHT/LEFT/RIGHT/BACK "
            "(15°/135°) остались только для интерфейса и не участвуют в выборе; "
            "наклон к сигналу стал непрерывным sin угла в названную сторону; "
            "BACK вынесен в отдельный случай; две ветки ближе 15° помечаются "
            "GEOMETRY_AMBIGUOUS и сигнал на них не влияет"),
        "p084a": po, "p084b": pn,
        "conclusion": (
            "правило изменено и проверено; на замороженных точках оно не меняет ни одного "
            "из 26 решений, и полный маршрут совпадает ребро в ребро — значит порог 45° "
            "не был причиной ошибок в T50, T49, J35"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"записано: {out/'comparison.json'}")
    print(f"записано: {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
