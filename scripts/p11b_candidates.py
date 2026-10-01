#!/usr/bin/env python3
"""P11.B step 1 — which cells sit downstream of the early visual cells, and how strongly.

P10 established a division of labour between the two questions and it is worth restating, because
everything here serves only the second one:

    SAME / DIFFERENT     present in the sixty-eight recorded descending cells. P11.A read it out
                         at 0.764 and found the loss was in the pooling. Done; not touched here.
    MOVE / NO_NET        present in the brain input (0.705) and in the early visual cells (0.695),
                         gone by the time the sixty-eight descending cells are recorded (0.560).

So the search here is narrow: follow the connectome outward from the cells the visual encoder
actually drives, and find where the second distinction still exists. Nothing about direction is
measured in this step, and SAME/DIFFERENT is not used at all.

ORIENTATION
-----------
The stored arrays group entries by *presynaptic* cell: `indptr[j]..indptr[j+1]` lists the cells
that cell j sends to, and `indices` holds those targets. Read as a SciPy CSR matrix this makes
W[i, k] the weight of the edge from i to k, so a row holds a cell's targets and a column holds its
inputs. That is the transpose of the usual convention and it was worth checking rather than
assuming: walking the rows of `W.T` instead collects a cell's *inputs*, which puts the lamina and
medulla cells that feed T4 at the top of a list meant to be about what lies past it. The check that
settled it: the cells that send to T4a are Mi1, Tm3, Mi9, Mi4 and C3, which are T4a's known inputs,
and the cells T4a sends to include LPT31 and LLPC1, which are lobula plate tangential cells.

WHAT IS MEASURED PER CELL TYPE
------------------------------
    depth            fewest hops from any early visual cell
    n_cells          how many cells of the type the connectome has
    direct_from_eye  total synaptic weight arriving straight from the early visual cells, summed
                     over the type
    frac             for a single cell, that weight as a share of all its input. A cell that
                     takes almost all of its drive from the eye is a relay; one that takes a
                     hundredth is integrating something else as well.

The encoder injects into four of the ten early types: T4a, T4b, T5a, T5b and the loom channels
LC4 and LPLC2. T4c, T4d, T5c and T5d are the vertical detectors and receive nothing, so both
definitions of "early source" are reported — the full ten and the six that are actually driven.

Usage:
    PYTHONPATH=. python scripts/p11b_candidates.py [--top 80] [--depth 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/p11b"

EARLY_ALL = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "LC4", "LPLC2")
EARLY_DRIVEN = ("T4a", "T4b", "T5a", "T5b", "LC4", "LPLC2")


def load_brain():
    from fly_vo.config import FlyVOConfig
    from fly_vo.malecns_engine import MaleCNSEngine
    eng = MaleCNSEngine(FlyVOConfig())
    return eng.brain


def early_indices(brain, types) -> np.ndarray:
    idx = np.unique(np.concatenate([brain.cells([t]) for t in types]))
    return idx.astype(np.int64)


def bfs_down(W: sparse.csr_matrix, src: np.ndarray, max_depth: int) -> np.ndarray:
    """Fewest outward hops from `src`; -1 where unreached. BFS, not a depth-3 ball.

    W is the matrix whose row i lists the targets of i, so this walks forward.
    """
    n = W.shape[0]
    depth = np.full(n, -1, dtype=np.int8)
    depth[src] = 0
    frontier = src
    for d in range(1, max_depth + 1):
        nxt = np.unique(W[frontier].indices)
        nxt = nxt[depth[nxt] < 0]
        if nxt.size == 0:
            break
        depth[nxt] = d
        frontier = nxt
    return depth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=80, help="сколько типов отобрать")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    brain = load_brain()
    n = brain.n
    indptr = np.asarray(brain.indptr)
    indices = np.asarray(brain.indices)
    weights = np.asarray(brain.weights, dtype=np.float64)
    ctype = np.asarray(brain.cell_type)
    side = np.asarray(brain.side)
    sclass = np.asarray(brain.superclass)

    print("=" * 100)
    print("P11.B шаг 1 — кандидаты после ранних зрительных клеток, по connectome")
    print("=" * 100)
    print(f"  клеток {n}, связей {len(indices)}, глубина поиска {args.depth}")
    print()

    W = sparse.csr_matrix((weights, indices, indptr), shape=(n, n))
    # row i = targets of i, column i = inputs to i (see ORIENTATION above).
    # Weights are signed, so a "share of the input" is only meaningful in absolute terms; the
    # model itself does the same, counting in-degree with np.abs(w) in flybrain.build.
    abs_w = np.abs(weights)
    Wabs = sparse.csr_matrix((abs_w, indices, indptr), shape=(n, n))
    total_in = np.asarray(Wabs.sum(axis=0)).ravel()
    n_in_edges = np.asarray((Wabs > 0).sum(axis=0)).ravel()
    print("  ориентация проверена на известных цепях: в T4a входят Mi1/Tm3, из T4a выходят "
          "LPT31/LLPC1")
    print(f"  связей: {W.nnz}; доля входа считается по |весу| (веса знаковые)")
    print()

    src_all = early_indices(brain, EARLY_ALL)
    src_driven = early_indices(brain, EARLY_DRIVEN)
    print(f"  ранних зрительных клеток всего: {len(src_all)}  ({', '.join(EARLY_ALL)})")
    print(f"  из них реально получают вход от encoder: {len(src_driven)} "
          f"({', '.join(EARLY_DRIVEN)})")
    print()

    # direct synaptic drive from the early cells into each cell downstream of them
    def direct_from(src: np.ndarray) -> np.ndarray:
        """Sum over the source rows: for every target, the weight arriving straight from the eye."""
        return np.asarray(Wabs[src].sum(axis=0)).ravel()

    d_all = direct_from(src_all)
    d_driven = direct_from(src_driven)

    depth_all = bfs_down(Wabs, src_all, args.depth)
    depth_driven = bfs_down(Wabs, src_driven, args.depth)
    for label, dep in (("все 10 типов", depth_all), ("6 реально ведомых", depth_driven)):
        cnt = Counter(dep[dep >= 0].tolist())
        print(f"  достижимо от {label}: "
              + ", ".join(f"глубина {k}: {cnt[k]}" for k in sorted(cnt) if k > 0))
    print()

    print("  ПРОВЕРКА НАПРАВЛЕНИЯ: клетки, которые КОРМЯТ T4 (Mi1, Tm3, Mi9, Mi4), не должны")
    print("  оказаться его целями — иначе обход идёт вверх:")
    for t in ("Mi1", "Tm3", "Mi9", "Mi4"):
        c = brain.cells([t])
        if len(c) == 0:
            continue
        dep = depth_driven[c]
        reach = int((dep >= 0).sum())
        print(f"    {t:<6} достижимо вниз: {reach} из {len(c)}"
              f"   {'OK' if reach == 0 else 'ВНИМАНИЕ: часть достижима — значит есть петли'}")
    print("  ЦЕЛИ T4a ПЕРВОГО ШАГА (должны быть лобула-пластинчатые и немного возвратных):")
    t4 = brain.cells(["T4a"])
    nxt = np.unique(Wabs[t4].indices)
    print(f"    {Counter(ctype[nxt].tolist()).most_common(8)}")
    print()

    # ---- aggregate per cell type ----
    rows = []
    types = np.unique(ctype)
    for t in types:
        m = ctype == t
        if not m.any():
            continue
        n_cells = int(m.sum())
        w_all = float(d_all[m].sum())
        w_dr = float(d_driven[m].sum())
        if w_all <= 0 and w_dr <= 0:
            continue
        tot = total_in[m]
        with_in = tot > 0
        frac = np.zeros(n_cells)
        frac[with_in] = d_all[m][with_in] / np.maximum(tot[with_in], 1e-12)
        depths = depth_all[m]
        depths = depths[depths >= 0]
        rows.append({
            "cell_type": str(t),
            "superclass": str(Counter(sclass[m]).most_common(1)[0][0]),
            "sides": sorted(set(side[m].tolist())),
            "n_cells": n_cells,
            "direct_from_early": w_all,
            "direct_from_driven": w_dr,
            "n_with_any_input": int(with_in.sum()),
            "median_frac_from_early": float(np.median(frac[with_in])) if with_in.any() else 0.0,
            "min_depth": int(depths.min()) if depths.size else -1,
            "n_at_depth1": int((depths == 1).sum()),
            "n_at_depth2": int((depths == 2).sum()),
            "n_at_depth3": int((depths == 3).sum()),
        })
    print(f"  типов клеток с прямой связью от ранних: {len(rows)}")
    print()

    # exclude the early visual types themselves
    is_early = np.array([r["cell_type"] in EARLY_ALL for r in rows])
    cand = [r for r, e in zip(rows, is_early) if not e]
    cand.sort(key=lambda r: -r["direct_from_driven"])
    print(f"  после исключения самих ранних типов: {len(cand)}")
    print()

    print("─── ТОП-30 ПО СУММАРНОМУ ВХОДУ ОТ РЕАЛЬНО ВЕДОМЫХ КЛЕТОК ───")
    print(f"  {'тип':<16}{'класс':<19}{'клеток':>7}{'глуб':>5}{'доля':>7}"
          f"{'гл.1':>6}{'гр.2':>6}{'гр.3':>6}")
    for r in cand[:30]:
        print(f"  {r['cell_type'][:15]:<16}{r['superclass'][:18]:<19}{r['n_cells']:>7}"
              f"{r['min_depth']:>5}{r['median_frac_from_early']:>7.2f}"
              f"{r['n_at_depth1']:>6}{r['n_at_depth2']:>6}{r['n_at_depth3']:>6}")
    print()

    chosen = cand[:args.top]
    by_class = Counter(r["superclass"] for r in chosen)
    by_depth = Counter(r["min_depth"] for r in chosen)
    print(f"─── ОТОБРАНО {len(chosen)} ТИПОВ ───")
    print(f"  по классам: {dict(by_class.most_common())}")
    print(f"  по минимальной глубине: {dict(sorted(by_depth.items()))}")
    print(f"  всего клеток в отобранных типах: {sum(r['n_cells'] for r in chosen)}")
    print()

    known = {"DNp17", "DNa07", "DNp26", "DNp20", "MDN", "DNa02", "DNg100"}
    hit = [r["cell_type"] for r in chosen if r["cell_type"] in known]
    print(f"  среди отобранных есть уже записанные ранее: {hit if hit else 'нет'}")
    print()

    np.savez_compressed(out / "connectome_cache.npz",
                        depth_all=depth_all, depth_driven=depth_driven,
                        d_all=d_all.astype(np.float32), d_driven=d_driven.astype(np.float32),
                        total_in=total_in.astype(np.float32),
                        src_all=src_all, src_driven=src_driven)
    (out / "candidates.json").write_text(json.dumps({
        "phase": "P11.B шаг 1 — кандидаты по connectome",
        "question_this_serves": "MOVE против NO_NET_DISPLACEMENT. SAME/DIFFERENT здесь не "
                                "используется.",
        "orientation": "CSR хранит [пост, пре]; обход вниз идёт по транспонированной",
        "early_all": list(EARLY_ALL),
        "early_driven_by_encoder": list(EARLY_DRIVEN),
        "max_depth": args.depth,
        "n_types_with_direct_link": len(rows),
        "n_candidates": len(cand),
        "n_chosen": len(chosen),
        "chosen_cells_total": int(sum(r["n_cells"] for r in chosen)),
        "chosen": chosen,
        "all_types": cand,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  записано: {out/'candidates.json'}, {out/'connectome_cache.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
