#!/usr/bin/env python3
"""P08.4A — what the map repair changed in the route, and nothing else.

The tracker was not touched: same video, same MaleCNS, same weights, same novelty, same pace,
same thresholds. Only the graph differs between the two runs, so every difference below is
the map's doing and can be attributed to it.

Reads the run made before the repair and the run made after it, and reports the six things
the audit asked about: the old sequence, the new sequence, the same decision points, dead ends,
turns back, and choices that lead nowhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seq_of(rs: list[dict]) -> list[str]:
    return [r["edge"] for r in rs]


def lcs(a: list[str], b: list[str]) -> list[str]:
    """The longest run of passages the two routes agree on, in order."""
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


def first_divergence(a: list[str], b: list[str]) -> int:
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


def degree_map(graph: dict) -> dict[str, int]:
    deg: dict[str, int] = {}
    for e in graph["edges"]:
        for n in (e["from"], e["to"]):
            deg[n] = deg.get(n, 0) + 1
    return deg


def edge_ends(graph: dict) -> dict[str, tuple[str, str]]:
    return {e["id"]: (e["from"], e["to"]) for e in graph["edges"]}


def travel_and_time(traj: list[dict], mpp: float) -> tuple[float, float]:
    """Distance walked and wall-clock span from the trajectory the tracker drew."""
    if not traj:
        return 0.0, 0.0
    ds = 0.0
    for p, q in zip(traj, traj[1:]):
        ds += ((float(q["x_px"]) - float(p["x_px"])) ** 2 +
               (float(q["y_px"]) - float(p["y_px"])) ** 2) ** 0.5
    return ds * mpp, float(traj[-1]["t"]) - float(traj[0]["t"])


def read_run(d: Path, prefix: str = "") -> dict:
    rep = json.loads((d / f"{prefix}report.json").read_text(encoding="utf-8"))
    return {
        "dir": d,
        "report": rep,
        "seq": rows(d / f"{prefix}edge_sequence.csv"),
        "dec": rows(d / f"{prefix}decisions.csv"),
        "traj": rows(d / f"{prefix}graph_trajectory.csv"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default=str(ROOT / "output/p084a"))
    ap.add_argument("--new", default=str(ROOT / "output/p084a/new"))
    ap.add_argument("--old-graph", default=str(ROOT / "data/p08/graph_before_p084a.json"))
    ap.add_argument("--new-graph", default=str(ROOT / "data/p08/graph.json"))
    ap.add_argument("--json", default=str(ROOT / "output/p084a/comparison.json"))
    args = ap.parse_args()

    old = read_run(Path(args.old), prefix="OLD_")
    new = read_run(Path(args.new))

    go = json.loads(Path(args.old_graph).read_text(encoding="utf-8"))
    gn = json.loads(Path(args.new_graph).read_text(encoding="utf-8"))
    mpp = gn["meters_per_pixel"]
    dego, degn = degree_map(go), degree_map(gn)
    ends = edge_ends(gn)

    so, sn = seq_of(old["seq"]), seq_of(new["seq"])
    agree = lcs(so, sn)

    print("=" * 78)
    print("P08.4A — сравнение до и после починки графа")
    print("=" * 78)
    print(f"граф: узлов {len(go['nodes'])} → {len(gn['nodes'])}, "
          f"рёбер {len(go['edges'])} → {len(gn['edges'])}")
    print(f"видео, муха, веса, novelty, темп, пороги — НЕ менялись")
    print()

    print("─── 1. последовательность пройденных рёбер ───")
    print(f"  было рёбер: {len(so)}, стало: {len(sn)}, "
          f"совпадает по порядку: {len(agree)}")
    k = first_divergence(so, sn)
    print(f"  первое расхождение — на шаге {k + 1}:")
    print(f"    было:  ... {' → '.join(so[max(0,k-2):k+3])}")
    print(f"    стало: ... {' → '.join(sn[max(0,k-2):k+3])}")
    print()
    print(f"  было целиком: {' → '.join(so)}")
    print()
    print(f"  стало целиком: {' → '.join(sn)}")
    print()

    print("─── 2. точки решений ───")
    print(f"  решений всего: было {len(old['dec'])}, стало {len(new['dec'])}")
    witho = [r for r in old["dec"] if r["reason"] != "ONLY_OPTION"]
    withn = [r for r in new["dec"] if r["reason"] != "ONLY_OPTION"]
    print(f"  решений с реальным выбором: было {len(witho)}, стало {len(withn)}")
    def reasons(rs):
        c: dict[str, int] = {}
        for r in rs:
            c[r["reason"]] = c.get(r["reason"], 0) + 1
        return c
    ro, rn = reasons(old["dec"]), reasons(new["dec"])
    print("  причины:")
    for key in sorted(set(ro) | set(rn)):
        print(f"    {key:<18} было {ro.get(key,0):>3}   стало {rn.get(key,0):>3}")
    print()

    print("─── 3. те же самые узлы ───")
    by_o = {r["node"]: r for r in old["dec"]}
    by_n = {r["node"]: r for r in new["dec"]}
    print(f"  {'узел':<7} {'было':<24} {'стало':<24}")
    for node in sorted(set(by_o) | set(by_n)):
        o, n = by_o.get(node), by_n.get(node)
        fo = f"{o['reason']:<16} {o['chosen_edge']}" if o else "— решения нет —"
        fn = f"{n['reason']:<16} {n['chosen_edge']}" if n else "— решения нет —"
        mark = "" if (o and n and o["chosen_edge"] == n["chosen_edge"]
                      and o["reason"] == n["reason"]) else "   ←"
        print(f"  {node:<7} {fo:<24} {fn:<24}{mark}")
    print()

    def dead_ends(dec: list[dict], ends: dict, deg: dict) -> list[dict]:
        """Choices whose far end has nothing beyond it but the way back."""
        out = []
        for r in dec:
            if r["reason"] == "ONLY_OPTION":
                continue
            e = r["chosen_edge"]
            if e not in ends:
                continue
            a, b = ends[e]
            far = b if r["node"] == a else a
            if deg.get(far, 0) <= 1:
                out.append({"t": r["time"], "node": r["node"], "edge": e, "dead_end": far})
        return out

    do = dead_ends(old["dec"], edge_ends(go), dego)
    dn = dead_ends(new["dec"], ends, degn)
    print("─── 4. выбор в тупик (за выбранным ребром ничего нет) ───")
    print(f"  было: {len(do)}")
    for d in do:
        print(f"    t={d['t']:>7} узел {d['node']:<5} → {d['edge']:<14} тупик {d['dead_end']}")
    print(f"  стало: {len(dn)}")
    for d in dn:
        print(f"    t={d['t']:>7} узел {d['node']:<5} → {d['edge']:<14} тупик {d['dead_end']}")
    print()

    def backtracks(dec: list[dict]) -> list[dict]:
        return [r for r in dec if r["reason"] == "BACK"]

    bo, bn = backtracks(old["dec"]), backtracks(new["dec"])
    print("─── 5. развороты назад (решение BACK) ───")
    print(f"  было: {len(bo)}   стало: {len(bn)}")
    for r in bo:
        print(f"    было  t={r['time']:>7} узел {r['node']:<5} → {r['chosen_edge']}")
    for r in bn:
        print(f"    стало t={r['time']:>7} узел {r['node']:<5} → {r['chosen_edge']}")
    print()

    # how often the route revisits an edge it has already walked
    def revisits(dec: list[dict]) -> int:
        seen: set[str] = set()
        c = 0
        for r in dec:
            if r["chosen_edge"] in seen:
                c += 1
            seen.add(r["chosen_edge"])
        return c

    print("─── 6. повторные проходы по уже пройденному ребру ───")
    print(f"  было: {revisits(old['dec'])}   стало: {revisits(new['dec'])}")
    print()

    do_m, do_t = travel_and_time(old["traj"], mpp)
    dn_m, dn_t = travel_and_time(new["traj"], mpp)
    print("─── 7. пройденный путь и время ───")
    print(f"  путь:  было {do_m:.2f} м, стало {dn_m:.2f} м")
    print(f"  время: было {do_t:.2f} с ({do_t/60:.1f} мин), стало {dn_t:.2f} с ({dn_t/60:.1f} мин)")
    print(f"  средняя скорость: было {do_m/do_t:.3f} м/с, стало {dn_m/dn_t:.3f} м/с")
    print()
    fo, fn_ = old["report"].get("final_node"), new["report"].get("final_node")
    print(f"  конечный узел: было {fo}, стало {fn_}")
    print(f"  предсказанная средняя скорость в отчёте: "
          f"{old['report'].get('predicted_mean_mps', 0):.3f} → "
          f"{new['report'].get('predicted_mean_mps', 0):.3f} м/с")

    Path(args.json).write_text(json.dumps({
        "old": {"nodes": len(go["nodes"]), "edges": len(go["edges"]),
                "sequence": so, "decisions": len(old["dec"]),
                "with_choice": len(witho), "dead_ends": do, "backtracks": bo,
                "travel_m": round(do_m, 3), "time_s": round(do_t, 3),
                "final_node": fo},
        "new": {"nodes": len(gn["nodes"]), "edges": len(gn["edges"]),
                "sequence": sn, "decisions": len(new["dec"]),
                "with_choice": len(withn), "dead_ends": dn, "backtracks": bn,
                "travel_m": round(dn_m, 3), "time_s": round(dn_t, 3),
                "final_node": fn_},
        "agreement_steps": len(agree), "first_divergence_step": k + 1,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nзаписано: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
