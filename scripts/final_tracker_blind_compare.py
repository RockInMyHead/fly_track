#!/usr/bin/env python3
"""Сравнение слепого прогона FINAL V1 с человеческой истиной (после разметки VID00018).

Главная метрика: доля времени на правильном ребре.

    PYTHONPATH=. .venv/bin/python scripts/final_tracker_blind_compare.py \\
        --truth data/final_tracker/VID00018_truth.json

Формат truth (пример):

{
  "video": "VID00018",
  "edge_timeline": [
    {"t0": 0.0, "t1": 32.5, "edge": "J5__J6"},
    ...
  ],
  "junctions": [
    {"time": 12.0, "node": "J35", "incoming_edge": "…", "chosen_edge": "…"},
    ...
  ]
}
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


def edge_at(timeline: list[dict], t: float) -> str | None:
    for seg in timeline:
        if float(seg["t0"]) <= t < float(seg["t1"]):
            return seg["edge"]
    if timeline and t >= float(timeline[-1]["t0"]):
        return timeline[-1]["edge"]
    return None


def compress_edges(edges: list[str]) -> list[str]:
    out = []
    for e in edges:
        if not out or out[-1] != e:
            out.append(e)
    return out


def lcs(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--run", default=None, help="каталог BLIND, по умолчанию из truth")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    truth_path = Path(a.truth)
    if not truth_path.is_absolute():
        truth_path = ROOT / truth_path
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    v = truth["video"]
    run_dir = Path(a.run) if a.run else ROOT / f"output/final_tracker/{v}_BLIND"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    traj_path = run_dir / "trajectory.csv"
    dec_path = run_dir / "decisions.csv"
    if not traj_path.exists():
        print(f"нет {traj_path}")
        return 1

    g = Graph.load(ROOT / "data/p08/graph.json")
    rows = list(csv.DictReader(traj_path.open(encoding="utf-8")))
    decs = list(csv.DictReader(dec_path.open())) if dec_path.exists() else []
    timeline = truth.get("edge_timeline") or []
    if not timeline:
        print("truth: нужен edge_timeline")
        return 1

    on_correct = 0
    total = 0
    first_wrong_t = None
    for r in rows:
        t = float(r["time"])
        pred = r["edge"]
        gold = edge_at(timeline, t)
        if gold is None:
            continue
        total += 1
        if pred == gold:
            on_correct += 1
        elif first_wrong_t is None:
            first_wrong_t = t

    frac_time = on_correct / max(total, 1)

    pred_seq = compress_edges([r["edge"] for r in rows])
    gold_seq = compress_edges([seg["edge"] for seg in timeline])
    lcs_len = lcs(pred_seq, gold_seq)
    frac_edges_lcs = lcs_len / max(len(gold_seq), 1)

    junctions = truth.get("junctions") or []
    j_total = len(junctions)
    j_ok = 0
    for j in junctions:
        t = float(j["time"])
        node = j["node"]
        want = j["chosen_edge"]
        hit = None
        for d in decs:
            if abs(float(d["time"]) - t) > 1.5:
                continue
            if d.get("node") == node:
                hit = d.get("chosen_edge")
                break
        if hit == want:
            j_ok += 1

    amb = [d for d in decs if d.get("decision_status") == "AMBIGUOUS"]
    dead = [d for d in decs if d.get("decision_status") == "DEAD_END_RETURN"]
    merged = [d for d in decs if d.get("decision_status") == "MERGED_LATER"]

    end_pred = rows[-1] if rows else {}
    end_gold = timeline[-1]
    end_node_pred = g.other(end_pred["edge"], g.edges[end_pred["edge"]]["from"])
    if end_pred.get("edge"):
        pe = end_pred["edge"]
        prog = float(end_pred.get("progress_m") or 0)
        L = g.length_m(pe) or 1e-9
        if prog > L * 0.5:
            end_node_pred = g.other(pe, g.edges[pe]["from"]) if prog < L else g.edges[pe]["to"]
    end_edge_err = 0.0
    if end_gold.get("edge") != end_pred.get("edge"):
        end_edge_err = 1.0

    pred_m = float(json.loads((run_dir / "report.json").read_text())["stats"]["route_meters"])
    gold_m = sum((g.length_m(seg["edge"]) or 0) for seg in timeline)
    len_err = abs(pred_m - gold_m)

    report = {
        "video": v,
        "truth": str(truth_path),
        "run_dir": str(run_dir),
        "primary_metric": {
            "fraction_time_on_correct_edge": round(frac_time, 4),
            "seconds_on_correct": on_correct,
            "seconds_scored": total,
        },
        "junctions": {
            "correct": j_ok,
            "total": j_total,
            "fraction": round(j_ok / max(j_total, 1), 4),
        },
        "edges_lcs": {
            "lcs_length": lcs_len,
            "truth_distinct_edges": len(gold_seq),
            "predicted_distinct_edges": len(pred_seq),
            "fraction_of_truth": round(frac_edges_lcs, 4),
        },
        "first_error_time_s": first_wrong_t,
        "false_dead_end_returns": len(dead),
        "ambiguous_decisions": len(amb),
        "merged_later": len(merged),
        "endpoint": {"predicted_edge": end_pred.get("edge"), "truth_edge": end_gold.get("edge"),
                     "edge_match": end_gold.get("edge") == end_pred.get("edge")},
        "route_length": {"predicted_m": pred_m, "truth_m": round(gold_m, 1),
                         "abs_error_m": round(len_err, 1)},
    }

    out = Path(a.out) if a.out else run_dir / "COMPARE.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"СРАВНЕНИЕ {v}")
    print("=" * 80)
    print(f"  ГЛАВНАЯ: время на правильном ребре = {frac_time:.1%} ({on_correct}/{total} с)")
    print(f"  развилки: {j_ok}/{j_total}  рёбра (LCS): {lcs_len}/{len(gold_seq)}")
    print(f"  первая ошибка: t={first_wrong_t}")
    print(f"  AMBIGUOUS={len(amb)}  DEAD_END_RETURN={len(dead)}  длина ±{len_err:.0f} м")
    print(f"  записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
