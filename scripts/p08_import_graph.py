#!/usr/bin/env python3
"""
P08 — import a topology graph into the format the tracker and the editor read.

The incoming file is `trackai.floorplan_graph.v1`: nodes keyed by long generated ids, and
edges carrying their own length in metres, a width, and an `enabled` flag. Three things
have to happen before it can be used here.

Disabled edges are dropped. The file marks five as off, and the validation block confirms
they are the only ones; keeping them would offer the tracker passages that are known to be
blocked. Nodes that end up with no edges are dropped too, because a node that cannot be
reached is a place the walker can never be.

Identifiers are shortened. The originals are generated and long — `turn__edge_00002__p001`
— and the editor draws every label on the plan, so a readable id matters. The new id keeps
the kind: `J` for a junction, `T` for a corner inside a corridor, `E` for an endpoint,
`M` for a node added by hand. The original id is kept in the file next to each node, so
nothing is lost and any result can be traced back.

Coordinates are kept in plan pixels and divided by the plan size on load, which is what the
editor does. The scale comes from the file rather than being assumed: 0.049629 metres per
pixel, and the per-edge `length_meters` is carried through so the route can be reported in
metres rather than in arbitrary units.

Usage:
    PYTHONPATH=. python scripts/p08_import_graph.py \
        --src "/Users/artem/Downloads/kerama-floorplan-graph-package"
    PYTHONPATH=. python scripts/p08_import_graph.py --src ... --keep-start A_B
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/p08"

KIND_LETTER = {"junction": "J", "turn": "T", "endpoint": "E", "manual": "M"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="the package directory holding graph/ and drawing/ or editor/")
    ap.add_argument("--graph", default="production.topology-graph.v1.json")
    ap.add_argument("--keep-start", default=None,
                    help="edge id in the NEW naming that the walk begins on")
    args = ap.parse_args()

    src = Path(args.src)
    gpath = src / "graph" / args.graph
    if not gpath.exists():
        raise SystemExit(f"нет файла графа {gpath}")
    doc = json.loads(gpath.read_text(encoding="utf-8"))

    # the plan: the editor copy is the same image as the drawing, either will do
    plan_src = None
    for cand in (src / "editor/plan.png", src / "drawing/kerama-marazzi-2025.png"):
        if cand.exists():
            plan_src = cand
            break
    if plan_src is None:
        raise SystemExit("не нашёл план ни в editor/, ни в drawing/")

    DEST.mkdir(parents=True, exist_ok=True)

    W = float(doc.get("width") or 0)
    H = float(doc.get("height") or 0)
    mpp = doc.get("meters_per_pixel")
    mins = doc.get("source", {}).get("minimum_edge_meters")

    print("P08 — импорт графа")
    print(f"  источник: {gpath.name}")
    print(f"  схема: {doc.get('schema_version')}, карта {doc.get('map_id')}")
    print(f"  план {W:.0f}x{H:.0f} px, масштаб {mpp:.6f} м/пиксель, "
          f"минимальное ребро {mins} м")

    raw_nodes = doc["nodes"]
    raw_edges = doc["edges"]
    n_off_e = sum(1 for e in raw_edges if not e.get("enabled", True))
    n_off_n = sum(1 for n in raw_nodes if not n.get("enabled", True))
    print(f"  в файле: узлов {len(raw_nodes)} (отключено {n_off_n}), "
          f"рёбер {len(raw_edges)} (отключено {n_off_e})")

    keep_edges = [e for e in raw_edges if e.get("enabled", True)]

    # Merge duplicates. The source lists four node pairs twice, each copy with its own id
    # but identical endpoints and identical length: E11-J56 at 4.66 m, J17-M13 at 1.24 m,
    # J21-J28 at 3.03 m, J40-T57 at 1.68 m. Two passages between the same pair of nodes
    # cannot be told apart by the tracker, and leaving both in creates a choice where the
    # building has none, which the fly is then asked about and which lets the route
    # rebound between the two nodes indefinitely. One copy is kept and the merge is
    # reported, so the change to the supplied graph is visible rather than silent.
    by_pair: dict[tuple[str, str], dict] = {}
    merged: list[tuple[str, str]] = []
    for e in sorted(keep_edges, key=lambda x: x["id"]):
        key = tuple(sorted((e["from"], e["to"])))
        if key in by_pair:
            merged.append((by_pair[key]["id"], e["id"]))
            continue
        by_pair[key] = e
    keep_edges = list(by_pair.values())
    if merged:
        print(f"  слито дублирующихся рёбер: {len(merged)}")
        for a, b in merged:
            print(f"    {a} и {b} — одна и та же пара узлов, оставлено {a}")

    used = set()
    for e in keep_edges:
        used.add(e["from"])
        used.add(e["to"])
    keep_nodes = [n for n in raw_nodes if n.get("enabled", True) and n["id"] in used]
    dropped = len(raw_nodes) - len(keep_nodes)
    print(f"  оставлено: узлов {len(keep_nodes)}, рёбер {len(keep_edges)}")
    print(f"  отброшено узлов без рёбер: {dropped}")

    # short ids, deterministic, with the kind in the letter
    counters: dict[str, int] = {}
    new_id: dict[str, str] = {}
    for n in sorted(keep_nodes, key=lambda x: x["id"]):
        letter = KIND_LETTER.get(n.get("kind"), "X")
        counters[letter] = counters.get(letter, 0) + 1
        new_id[n["id"]] = f"{letter}{counters[letter]}"

    # short ids for edges, from the two endpoints, unique by suffix if repeated
    edge_new: dict[str, str] = {}
    seen: dict[str, int] = {}
    for e in sorted(keep_edges, key=lambda x: x["id"]):
        base = f"{new_id[e['from']]}__{new_id[e['to']]}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        edge_new[e["id"]] = base if k == 0 else f"{base}__{k + 1}"

    out_nodes = []
    for n in sorted(keep_nodes, key=lambda x: new_id[x["id"]]):
        out_nodes.append({
            "id": new_id[n["id"]],
            "x": round(float(n["x"]) / W, 6),
            "y": round(float(n["y"]) / H, 6),
            "orig_id": n["id"],
            "kind": n.get("kind"),
            "degree": n.get("degree"),
        })

    out_edges = []
    lengths_m = []
    for e in sorted(keep_edges, key=lambda x: edge_new[x["id"]]):
        length_m = e.get("length_meters")
        if length_m is not None:
            lengths_m.append(float(length_m))
        out_edges.append({
            "id": edge_new[e["id"]],
            "from": new_id[e["from"]],
            "to": new_id[e["to"]],
            "bidirectional": bool(e.get("bidirectional", True)),
            "length_meters": round(float(length_m), 4) if length_m is not None else None,
            "width_median_m": e.get("median_width_meters"),
            "orig_id": e["id"],
        })

    start = None
    if args.keep_start:
        match = [e for e in out_edges if e["id"] == args.keep_start]
        if not match:
            raise SystemExit(f"ребра {args.keep_start} нет среди импортированных")
        start = {"type": "edge", "edge": match[0]["id"],
                 "from": match[0]["from"], "to": match[0]["to"]}

    out = {
        "schema": "p08.graph.v1",
        "source_file": gpath.name,
        "source_schema": doc.get("schema_version"),
        "map_id": doc.get("map_id"),
        "image": "plan.png",
        "img_w": W,
        "img_h": H,
        "meters_per_pixel": mpp,
        "minimum_edge_meters": mins,
        "known_distance": None,
        "start": start,
        "nodes": out_nodes,
        "edges": out_edges,
    }

    (DEST / "graph.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    shutil.copyfile(plan_src, DEST / "plan.png")

    # the untouched original, kept beside the converted file
    shutil.copyfile(gpath, DEST / "graph_source.json")

    kinds: dict[str, int] = {}
    for n in out_nodes:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    total_m = sum(lengths_m)

    print()
    print(f"  записано: {DEST}/graph.json и {DEST}/plan.png")
    print(f"  оригинал сохранён: {DEST}/graph_source.json")
    print()
    print(f"  узлы по видам: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))

    # degrees are recomputed from the kept edges, not read from the file: merging the
    # duplicates changes them, and the file's own degree field still counts both copies
    deg: dict[int, int] = {}
    deg_of: dict[str, int] = {}
    for n in out_nodes:
        d = sum(1 for e in out_edges if e["from"] == n["id"] or e["to"] == n["id"])
        deg_of[n["id"]] = d
        deg[d] = deg.get(d, 0) + 1

    print(f"  узлы по степени (пересчитано): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(deg.items(), reverse=True)))
    print(f"  суммарная длина всех рёбер: {total_m:.0f} м")
    print(f"  старт: {start['edge'] + ' ' + start['from'] + ' → ' + start['to'] if start else 'не задан'}")
    print()
    n_choice = sum(1 for d in deg_of.values() if d >= 3)
    print(f"  узлов с развилкой (степень >= 3): {n_choice} — "
          f"только там спрашивается MaleCNS")
    print(f"  узлов проходных (степень 2): {deg.get(2, 0)} — проходятся автоматически")
    print(f"  концов и тупиков (степень 1): {deg.get(1, 0)}")
    if deg.get(0):
        print(f"  изолированных (степень 0): {deg[0]} — сюда попасть нельзя")


if __name__ == "__main__":
    main()
