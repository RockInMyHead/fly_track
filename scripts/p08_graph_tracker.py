#!/usr/bin/env python3
"""
P08 — the route as a walk over the graph, instead of a line integrated from yaw.

The change this stage makes
---------------------------
P07 integrated yaw into a heading and let every event bend the route. That is wrong in
one specific and very visible way: a glance to the side in the middle of a corridor
becomes a permanent turn. Stages 4 and 5 of the P08 description replace the free heading
with a graph, and the rule becomes:

    while the walker is inside an edge, yaw cannot change the route at all.
    a route change is only possible near a node, and only onto an edge that exists there.

So yaw is not used as a direction any more. It is used as a choice between the two or
three passages that are physically present, and only at the moment one of them has to be
chosen. Where there is no choice, MaleCNS is not consulted at all (stages 14 and 15).

Progress along an edge
----------------------
Not free integration, and not metres either. A single constant `pix_per_sec` says how
many plan pixels the walker covers per second, modulated by the clip's own speed signal
so that a pause slows the approach and does not let the clock run on. The speed signal is
zero for about two thirds of frames, so it is floored at a fraction of the clip's median
moving speed rather than used raw, which would freeze progress whenever the walker stands
still. `pix_per_sec` is the one free constant of P08, fitted on VID00001 only, and it
cannot change the *order* of chosen edges except through the timing of arrivals. The
reason it cannot: at an arrival the choice is between edges that exist at that node.

What yaw is read from
---------------------
The frozen P07 signal, `output/p07/yaw_signal_<clip>.csv`. The integral over the approach
is taken from `yaw_signal_deadband` rather than from `yaw_gated`: `yaw_gated` is a detector
output and is zero about 92 percent of the time, so its integral over a window measures
the detector rather than the turn. The other two columns are carried along as diagnostics
so the choice can be checked.

Usage:
    PYTHONPATH=. python scripts/p08_graph_tracker.py
    PYTHONPATH=. python scripts/p08_graph_tracker.py --video VID00002
    PYTHONPATH=. python scripts/p08_graph_tracker.py --t0 60 --t1 240 --pix-per-sec 45
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from p08_graph import BACK_MIN_DEG, Graph  # noqa: E402

# P08.4B: the choice is made on the real angle of each passage, not on which side of 45
# degrees it falls. The rule lives in its own module so the replay and this run cannot drift
# apart — both call it, so any difference between them is a difference of input.
import p084b_geometry as geo  # noqa: E402
# P09.2: the look-or-turn filter. It can only withdraw a direction, never supply one.
import p092_look_turn as lookturn  # noqa: E402

P07 = ROOT / "output/p07"
P071 = ROOT / "output/p071"
OUT = ROOT / "output/p08"
GRAPH_PATH = ROOT / "data/p08/graph.json"

# ---- parameters, all stated here and never tuned per clip afterwards ---------
NODE_ZONE = 0.0         # stage 12's zone. Zero disables the gate, which is the default:
                        # the route can only change at a node anyway, and gating the yaw
                        # window to the last fifth of a 3.3 m edge left too little signal
                        # to decide with. Set to 0.8 to restore the literal stage 12 rule.
APPROACH_S = 2.0        # stage 16: the integral is taken over the last this many seconds
SETTLE_S = 1.2          # yaw keeps accumulating this long after the node is reached
# The floor for a usable integral. Integrating a non-turning signal over a window does not
# give zero, and how far from zero it lands grows with the window: measured on every
# sliding window of both clips, the 95th percentile of |integral| is
#     0.125 / 0.173 / 0.214 / 0.252 / 0.331 / 0.403 / 0.511   at   0.4 / 0.6 / 0.8 / 1 / 1.5 / 2 / 3 s
# which fits 0.091 + 0.147·W on VID00001 (R² 0.98) and 0.077 + 0.117·W on VID00002 (R² 0.99).
# The floor is therefore a line in the window length rather than a constant, and the
# coefficients come from the noisier clip so that the same rule is strict on both:
FLOOR_A = 0.09
FLOOR_B = 0.15
# The multiplier on that floor. 1.0 is the literal reading of the noise level and leaves
# the fly naming a direction at only 14 junctions in a twenty minute walk, contradicting
# the graph on 71 percent of them. 0.5 is the operating point chosen by the person running
# the experiment after seeing the trade in `--sensitivity`: 44 junctions named, 34 percent
# contradicting, and 76 of the 205 passages visited rather than 22. It is a choice, not a
# measurement, and it is stated in the report as such.
FLOOR_SCALE = 0.50
# The sign convention of the yaw signal, which is the one thing in this file that was
# wrong for the whole of P08 and made every directional decision backwards.
#
# `p07_fly_readout.py` states it explicitly, under a heading called "Sign convention":
#
#     yaw > 0   the camera turned LEFT
#
# and the frozen scripts agree with it: `p07_fly_readout` scores a RIGHT turn as expecting a
# negative value, `np.sign(vals) == np.where(isR, -1, 1)`. The signal itself is not in doubt —
# measured against the hand-confirmed turns on both labelled clips it is right on 91 of 91,
# which is the figure the docstring of `p074` quotes.
#
# The tracker read it the other way round. `"LEFT" if integral < 0 else "RIGHT"` is the
# inverse, so every junction where the fly was consulted was told the opposite of what the
# fly said. It was invisible in the outputs because the geometry usually decided anyway, and
# where it did not the result looked merely unmotivated rather than wrong.
#
# Kept as a named mapping so that the convention is stated once and cannot drift again.
YAW_SIGN_LEFT = 1       # sign of the integral that means the camera turned left

# Geometry is the base: how nearly a passage continues the current direction.
# 0.5 * (1 + cos(deg)) runs from 1 at straight on to 0 at a reversal.
MALE_BONUS = 1.50       # added to a passage on the side the signal names, scaled by how
                        # far the integral stands above the floor: a signal barely clearing
                        # the floor should not override geometry, a decisive one should
BACK_PENALTY = 0.08     # multiplier for a passage that leads backwards, so geometry alone
                        # puts it far below any way of carrying on
BACK_FACTOR = 3.0       # a reversal, or a turn reported where the graph has none, needs
                        # this many times the floor before it is believed

# ---- memory. A person walking a building does not go round the same ring for eleven
# minutes, but a rule that only looks at angles does exactly that: the straightest way out
# of a small loop leads back round it, and nothing remembers having been there. On the
# twenty minute walk this was not a subtle effect. From 566 s to 1224 s — 54 percent of the
# clip — the route went round one ring of dead ends, passing the same seven passages 38 to
# 40 times each, and never once took the side exit at M27 that would have left.
#
# Two things are therefore remembered. How often a passage has been walked, which makes an
# unused one more attractive as the count grows, and when it was last walked, which makes a
# passage just travelled in the opposite direction very unattractive. The second is what
# stops an out-and-back into a dead end from being repeated: the way back in is right there
# and geometrically straight, so nothing but memory prevents taking it again.
RETURN_S = 60.0         # a passage walked within this many seconds is treated as spent
RECENT_PENALTY = 0.12   # multiplier applied to such a passage
NOVELTY_DECAY = 0.7     # how fast preference for an unvisited passage fades with each walk
MALE_MARGIN = 0.25      # top two candidates closer than this: decision deferred
REVISE_FRAC = 0.15      # a deferred decision may still change over this part of the edge
# How clear a majority a deferred decision needs before it is revised. This used to be a
# bare `> 0.5`, which is not a majority at all but an infinitesimal asymmetry: on a junction
# with two ways on mirrored about straight on, the scores are equal to six decimal places and
# any nudge at all moved the leader over half. Found in P08.4B, when the fly's promotion
# began to vary continuously with the angle — a passage twelve degrees off straight now earns
# a little from a RIGHT signal where the old label gave it nothing — and a junction that is
# genuinely fifty-fifty started reporting RECOVERY with 0.93 confidence. The bar is set at the
# same 0.75 the tests call "high confidence", which is a three-to-one score ratio.
RECOVER_MARGIN = 0.75
# Progress stops when the video shows no movement. The floor used to be 0.30 of the median
# moving speed, which meant a walker standing still still advanced at 0.66 m/s, and a long
# pause turned into distance walked. With the floor at zero the pace is rescaled by the mean
# of the relative speed (see `normalise_pace`), so the average speed over the clip is
# unchanged while the distribution follows the video: fast when the walker walks, nothing
# when the walker stands.
SPEED_FLOOR = 0.0

# The fastest the walker is allowed to move, whatever the forward signal says. Rescaling the
# pace by the mean of the relative speed makes the clip average equal WALK_MPS, and on a clip
# that is idle for much of its length that pushes the peak up instead: VID00006 has a mean
# relative speed of 0.40, so its pace at full signal came out at 3.46 m/s, which is running,
# and a junction decision then covered 4.2 m of walking. Capping the instantaneous speed
# keeps the peaks plausible; the clip average is then an outcome rather than a target, and it
# is reported.
MAX_MPS = 2.0
ENTRY_MAX = 0.95        # a passage is never entered already finished. See `move_onto`: the
                        # overshoot from a junction decision can exceed a short passage, and
                        # without this cap such a passage gets zero duration and is skipped
SUB_RESOLUTION_S = 0.25  # a pass shorter than this is marked `pass_masked`: it was walked
                         # while a junction decision was underway, so its duration is not a
                         # measurement. The value is a reading convenience, not a threshold
                         # the tracker acts on

# Speed. With a stub plan and no scale this had to be an arbitrary number of plan pixels per
# second. With the real graph the plan carries metres per pixel, so it is no longer a free
# parameter at all: it is an ordinary walking speed divided by the scale.
#     1.4 m/s  is the standard figure for walking indoors on the level
#     / 0.049629 m per pixel  =  28.2 plan pixels per second
# The figure is stated, not fitted, and the tracker prints the metres per second it implies.
WALK_MPS = 1.4
PIX_PER_SEC_FALLBACK = 28.2     # used only when the graph carries no scale



def load_signals(video: str, yaw_column: str):
    yrows = list(csv.DictReader((P07 / f"yaw_signal_{video}.csv").open()))
    t = np.array([float(r["t"]) for r in yrows])
    yaw = np.array([float(r[yaw_column]) for r in yrows])
    alt = {}
    for c in ("yaw_signal", "yaw_signal_deadband", "yaw_pair_diagnostic"):
        if c in yrows[0]:
            alt[c] = np.array([float(r[c]) for r in yrows])

    frows = list(csv.DictReader((P071 / f"trajectory_{video}.csv").open()))
    tf = np.array([float(r["t"]) for r in frows])
    speed = np.array([float(r["speed"]) for r in frows])
    n = min(len(t), len(tf))
    return t[:n], yaw[:n], tf[:n], speed[:n], {k: v[:n] for k, v in alt.items()}


def relative_speed(speed: np.ndarray) -> np.ndarray:
    """Speed normalised to the clip's own moving median, and floored so it never stops."""
    moving = speed[speed > 0]
    ref = float(np.median(moving)) if len(moving) else 1.0
    rel = speed / max(ref, 1e-9)
    return np.maximum(rel, SPEED_FLOOR)


def normalise_pace(rel: np.ndarray, pix_per_sec: float) -> tuple[float, float]:
    """The pace, returned unchanged.

    This used to divide the pace by the mean of `rel`, so that the average speed over a clip
    came out equal to the stated walking speed. That was a mistake, and it is worth recording
    why. `relative_speed` already divides by the median speed of the frames that show
    movement, so `rel = 1` already means an ordinary walking frame, and the stated speed
    applies to it directly. Dividing again by the mean — which on these clips is about 0.4,
    because the walker is still for most of the time — inflated the pace by two and a half
    times. An ordinary walking frame then advanced at 3.46 m/s on VID00006 and 3.36 m/s on
    VID00001, and a junction decision covered 1.7 m of walking, which is what made a 0.63 m
    passage come out with a duration of 0.02 s.

    The average speed over a clip is therefore an outcome, not a target: it comes to about
    half the walking speed, because the walker is still for about half the time. That is
    reported rather than corrected away.

    Kept as a function so the mean is still computed and shown in the report.
    """
    m = float(np.mean(rel)) if len(rel) else 1.0
    return pix_per_sec, m


def yaw_floor(window_s: float, scale: float = 1.0) -> float:
    """How large |integral| has to be before a direction is believed, given the window.

    `scale` moves the floor against the measured noise level. It exists so that the
    consequence of believing weaker evidence can be stated rather than assumed: at 1.0 the
    floor is the 95th percentile of a non-turning window, so a direction is named on about
    one junction in twenty by chance alone; lower values admit more of the signal and more
    of the noise, and the trade is reported rather than chosen silently.
    """
    return scale * (FLOOR_A + FLOOR_B * max(float(window_s), 0.0))


def geometry_score(deg: float) -> float:
    """Superseded by `p084b_geometry.geometry_score`; kept as a name for older callers.

    The formula is unchanged — this phase did not reshape the score, it stopped the decision
    from reading a label instead of the number. The implementation now lives in the rule
    module so that the tracker, the frozen replay and the graph audit all use one copy.
    """
    return geo.geometry_score(deg)


def is_back(deg: float) -> bool:
    return geo.is_reversal(deg)


def novelty(edge: str, visits: dict, last_visit: dict, tnow: float,
            return_s: float = RETURN_S) -> float:
    """How attractive a passage is, given what has already been walked.

    An unused passage scores one. A used one loses value in two separate ways that multiply,
    because they answer different questions and one of them alone is not enough:

      how often it has been walked      1 / (1 + NOVELTY_DECAY * visits)
      whether it was walked just now    times RECENT_PENALTY, inside RECENT_S

    The multiplication is the correction. With a flat recency penalty — the same number for
    any recently walked passage — the two things collapse into one value and the count is
    lost, which is what the visit counts in decisions.csv showed happening at node A: the
    exit A_J38 stayed at one walk while the dead-end A_B climbed to eight, and both were
    scored 0.120, so geometry sent the route back into the dead end nine times running. With
    the two multiplied, a passage walked once keeps 0.588 of its value against 0.152 for one
    walked eight times, and the less-used way out wins.

    The consequences of the two are deliberately different. Recency is a strong, short-lived
    penalty: it is what stops an out-and-back into a dead end being repeated immediately.
    Frequency is a mild, permanent one: it never excludes a passage, it only makes the
    walker prefer one it has not tried.
    """
    n = visits.get(edge, 0)
    if n == 0:
        return 1.0
    by_count = 1.0 / (1.0 + NOVELTY_DECAY * n)
    if tnow - last_visit.get(edge, -1e9) <= return_s:
        return by_count * RECENT_PENALTY
    return by_count


def score_candidates(cands: list[dict], integral: float, floor: float,
                     visits: dict | None = None, last_visit: dict | None = None,
                     tnow: float = 0.0) -> tuple[list[dict], str]:
    """Superseded by `p084b_geometry.score_candidates`, which the decision now calls.

    What changed is one thing, and it is the whole of P08.4B: the promotion for a passage the
    signal named used to depend on the passage's *label* — `c["side"] == male` — so a passage
    26 degrees off, labelled STRAIGHT by the 45 degree boundary, was never promoted by a
    RIGHT signal even though it is plainly a right turn. It is now a continuous function of
    the passage's own angle: `sin` of the deflection into the direction named, positive on
    the side the fly named, negative on the other side, and zero at straight on.

    Kept as a thin adapter so that anything importing it goes on working; the tracker itself
    no longer calls it.
    """
    res = geo.score_candidates(cands, integral, floor, visits, last_visit, tnow,
                               male_bonus=MALE_BONUS, back_factor=BACK_FACTOR,
                               back_penalty_min=BACK_PENALTY, decay=NOVELTY_DECAY,
                               recent_penalty=RECENT_PENALTY, return_s=RETURN_S,
                               yaw_sign_left=YAW_SIGN_LEFT)
    out = [dict(s, male_bonus=s["male_score"], is_back=geo.is_reversal(s["deg"]))
           for s in res["scored"]]
    return out, res["male_direction"]


def pix_per_sec_for(g: Graph, walk_mps: float | None = None) -> tuple[float, str]:
    """Plan pixels per second, from a walking speed and the plan's own scale.

    Returns the value and a one-line account of where it came from, so that a run always
    states whether the speed was derived from the plan or fell back to a default.
    """
    v = WALK_MPS if walk_mps is None else walk_mps
    if g.meters_per_pixel and g.meters_per_pixel > 0:
        return v / float(g.meters_per_pixel), (
            f"{v:.2f} м/с ÷ {float(g.meters_per_pixel):.6f} м/пиксель")
    return PIX_PER_SEC_FALLBACK, (
        f"{v:.2f} м/с, но в графе нет масштаба — взято значение по умолчанию")


def track(g: Graph, t: np.ndarray, yaw: np.ndarray, speed_rel: np.ndarray,
          pix_per_sec: float = None, start: dict | None = None,
          node_zone: float = NODE_ZONE, floor_scale: float = FLOOR_SCALE,
          look_turn=None, unknown_revokes: bool = True) -> dict:
    """Walk the graph under the rules above. Returns decisions, edges and a per-frame path.

    Written as a plain function over arrays so that it can be driven by a synthetic yaw
    trace as easily as by a recording, which is how the two mandatory cases are checked.
    """
    if pix_per_sec is None:
        # the same derivation the command line uses, so a sweep and a single run agree
        pix_per_sec, _ = pix_per_sec_for(g)
        pix_per_sec, _ = normalise_pace(speed_rel, pix_per_sec)
    start = start or g.start
    if not start:
        raise ValueError("граф не задаёт старт: укажите start в graph.json или --start-edge")

    if start.get("type") == "at_node" or "from_node" in start:
        # Standing at a node and having arrived from a named neighbour. This is how a person
        # describes a start in a building — "at J6, coming from J5" — and it is more precise
        # than naming an edge, because the direction of travel is what fixes which way is
        # straight, left and right at the first junction. The edge to the neighbour is used
        # only to define that heading; the walker begins at the node, so the first thing
        # that happens is the choice there.
        node = start["node"]
        came = start.get("from_node") or start.get("from")
        cand = [e for e in g.edges_at(node) if g.other(e, node) == came]
        if not cand:
            raise ValueError(f"в графе нет ребра {came}—{node}")
        edge = cand[0]
        frm, to = came, node
        at_node_start = True
    elif start.get("type") == "node" or "node" in start:
        node = start["node"]
        at = g.edges_at(node)
        if not at:
            raise ValueError(f"из узла {node} не выходит ни одного ребра")
        edge = at[0]
        frm, to = node, g.other(edge, node)
        at_node_start = False
    else:
        edge = start["edge"]
        frm = start.get("from") or g.edges[edge]["from"]
        to = start.get("to") or g.other(edge, frm)
        if not g.can_travel(edge, frm):
            raise ValueError(f"по ребру {edge} нельзя идти из {frm}")
        at_node_start = False

    decisions: list[dict] = []
    seq: list[dict] = []
    path: list[dict] = []
    t_edge_start = float(t[0])
    warnings: list[str] = []

    carry_lost = 0.0        # distance that could not be carried, in plan pixels
    visits: dict[str, int] = {edge: 1}
    last_visit: dict[str, float] = {edge: float(t[0])}
    # Every time each passage was entered, in order. `visits` and `last_visit` are the summary
    # the novelty score reads, but the revision window needs to look at the history *without*
    # the traversal that is happening right now, and a summary cannot be un-summed. See the
    # revision block below for what went wrong without this.
    visit_times: dict[str, list[float]] = {edge: [float(t[0])]}
    progress = 0.0
    state = "ON_EDGE"
    carry_px = 0.0                  # overshoot past the previous node, in plan pixels
    zone_t: float | None = None
    buf: list[tuple[float, float]] = []   # (time, yaw*dt) inside the window
    w_start = 0.0                   # left edge of the window, frozen at the node
    t_node = 0.0                    # when the node was reached
    settle_left = 0.0
    pending: dict | None = None

    if at_node_start:
        # start already at the junction, so the first frame is the decision there
        progress = 1.0
        t_node = float(t[0])
        w_start = t_node - APPROACH_S

    def window_integral(tnow: float) -> tuple[float, float]:
        """Sum of yaw over the window, and how long the window actually is.

        The window is anchored at the node, not at the present: it runs from
        `t_node - APPROACH_S` to the moment of the decision. This is the whole point of the
        settle period. A turn begins at the junction, so a window that ended at the node
        would contain the straight approach and almost none of the turn — measured with
        such a window, only 8 percent of junctions produced an integral above the floor and
        MaleCNS was consulted 25 times in twenty minutes. Carrying the window past the node
        puts the turn itself inside it.

        The length is returned because the floor depends on it.
        """
        if not buf:
            return 0.0, 0.0
        vals = [c for tt, c in buf if tt >= w_start]
        if not vals:
            return 0.0, 0.0
        first = min(tt for tt, _ in buf if tt >= w_start)
        return float(sum(vals)), float(min(tnow - first, APPROACH_S + SETTLE_S))

    # plan pixels covered in one step at this frame, with the instantaneous speed capped
    max_pace = (MAX_MPS / float(g.meters_per_pixel)
                if g.meters_per_pixel else float("inf"))
    pace_used = min(pix_per_sec, max_pace)

    def advance(i: int) -> float:
        """Plan pixels walked during one step of this frame.

        The pace is the nominal one, rescaled so the clip average equals the walking speed,
        but it is capped so that a high forward reading cannot make the walker run.
        """
        return min(pix_per_sec * float(speed_rel[i]), max_pace)

    def push_yaw(tnow: float, y: float, dt: float) -> None:
        buf.append((tnow, y * dt))
        while buf and buf[0][0] < w_start:
            buf.pop(0)

    def log_decision(tnow, node, cands, integral, chosen, probs, reason, conf, male_dir,
                     window_s=APPROACH_S, detail=None, floor_used=None,
                     arrival_edge=None, back_rule=None, ambiguous=0):
        detail = detail or {}
        row = {
            "time": round(float(tnow), 3),
            # the passage we arrived along, NOT the one being entered. It used to be written
            # after the move had already happened, so this column always repeated the chosen
            # edge and the arrival had to be inferred from the sequence.
            "current_edge": arrival_edge if arrival_edge is not None else edge,
            "progress": round(float(min(progress, 1.0)), 4),
            "node": node,
            "candidate_edges": "|".join(c["edge"] for c in cands),
            "candidate_angles": "|".join(f"{c['deg']:.1f}" for c in cands),
            "candidate_sides": "|".join(c["side"] for c in cands),
            "yaw_integral": round(float(integral), 4),
            "yaw_window_s": round(float(window_s), 3),
            # the floor that was actually applied, not a fresh recomputation: the
            # multiplier belongs in this number and leaving it out made the log disagree
            # with the decision it describes
            "yaw_floor": round(float(floor_used if floor_used is not None
                                     else yaw_floor(window_s)), 4),
            # how far the integral stands above the floor, 0 at the floor and 1 at the
            # level that would justify a reversal. This is the number that decides how loud
            # the fly is allowed to be, so it belongs in the log next to the integral.
            "yaw_strength": round(float(detail.get("yaw_strength", 0.0)), 4),
            "male_direction": male_dir or "",
            "chosen_edge": chosen,
            "confidence": round(float(conf), 3),
            "reason": reason,
            # why the passage behind was or was not on the table, so a reversal can be read
            # back the same way every other decision can
            "back_rule": back_rule or "",
            "ambiguous": int(ambiguous),
            # P09.2
            "yaw_direction": (detail or {}).get("yaw_direction", male_dir or ""),
            "look_turn_score": (detail or {}).get("_lt", {}).get("score", ""),
            "look_turn_class": (detail or {}).get("_lt", {}).get("class", ""),
            "pair_start": (detail or {}).get("_lt", {}).get("pair_start", ""),
            "pair_end": (detail or {}).get("_lt", {}).get("pair_end", ""),
            "male_used_for_route": (int(bool((detail or {}).get("_lt", {}).get("male_used")))
                                    if (detail or {}).get("_lt") else ""),
            "male_route": (detail or {}).get("_lt", {}).get("route", ""),
            "probabilities": "|".join(f"{k}:{v:.2f}" for k, v in (probs or {}).items()),
        }
        for k in ("signal_strength", "angle_degrees", "geometry_scores", "back_penalties",
                  "male_scores", "final_scores", "novelty_scores", "visit_counts",
                  "alignments"):
            row[k] = (detail or {}).get(k, "")
        decisions.append(row)

    def close_edge(t_end):
        dur = float(t_end) - t_edge_start
        # A pass shorter than the time a junction decision masks cannot be timed on its own:
        # the walker covered it while the decision was being made. The passage is still in
        # the route, in the right place, because the order is what matters; only its
        # duration is uninformative, and this column says so rather than leaving a
        # suspiciously small number to be read as a measurement.
        masked = 1 if dur < SUB_RESOLUTION_S else 0
        seq.append({"time_start": round(t_edge_start, 3),
                    "time_end": round(float(t_end), 3),
                    "edge": edge, "from": frm, "to": to,
                    "duration": round(dur, 3),
                    "pass_masked": masked,
                    "length_m": (round(g.length_m(edge), 3)
                                 if g.length_m(edge) is not None else "")})

    def move_onto(new_edge, new_from, new_to, reason, tnow, cands, integral, probs,
                  conf, male_dir, defer, window_s=APPROACH_S, detail=None, floor=None,
                  back_rule=None, ambiguous=0):
        nonlocal edge, frm, to, progress, state, zone_t, settle_left, pending, t_edge_start
        nonlocal carry_px, carry_lost
        arrived_by = edge    # remembered before it is overwritten, for the log
        # the previous passage ends at the node, not at the moment of the decision
        close_edge(float(t_node) if t_node else tnow)
        edge, frm, to = new_edge, new_from, new_to
        # The overshoot past the previous node is the distance already walked on the new
        # passage. It is carried, because discarding it makes the route shorter than the
        # clock and the speed imply — every node would cost the walker whatever it had gone
        # beyond the node.
        #
        # It is however capped, because a passage shorter than that overshoot would
        # otherwise be entered already finished: the edge is then logged with a duration of
        # zero and effectively skipped. That happened on VID00006 to T12__J6 at t=1.20 s: a
        # 0.63 m passage, entered with the overshoot from a junction decision that covers
        # about 1.7 m of walking on that clip. Capping rather than dropping keeps the passage
        # in the route; the distance that does not fit is recorded, so the loss is visible
        # instead of silent.
        new_len = max(g.length_px(new_edge), 1e-9)
        entry = max(carry_px, 0.0) / new_len
        if entry > ENTRY_MAX:
            carry_lost += (entry - ENTRY_MAX) * new_len
            entry = ENTRY_MAX
        progress = entry
        carry_px = 0.0
        state = "ON_EDGE"
        zone_t = None
        buf.clear()
        settle_left = 0.0
        # the passage begins at the node, not at the moment the decision was taken: the
        # walker keeps walking through the settle period, and dating the new edge from the
        # decision would shift every boundary in the sequence by that much
        t_edge_start = float(t_node) if t_node else float(tnow)
        pending = ({"node": new_from, "cands": cands, "integral": integral,
                    "edge": new_edge, "time": t_edge_start, "window_s": window_s,
                    # how much of the sequence exists now, and when this passage began:
                    # both are needed to undo exactly this excursion if the decision is
                    # revised, and to know how far the walker has already come
                    "seq_len": len(seq), "t_start": t_edge_start,
                    # the ways on and the passage behind, kept so the revision re-runs the
                    # same rule on the same two lists rather than rebuilding them
                    "ways_on": detail.get("_ways_on") if detail else None,
                    "behind": detail.get("_behind") if detail else None,
                    "back_rule": back_rule, "ambiguous": ambiguous}
                   if defer else None)
        # entering a passage counts as walking it; the count and the time of the last walk
        # are what the novelty score reads at the next junction
        visits[new_edge] = visits.get(new_edge, 0) + 1
        last_visit[new_edge] = float(tnow)
        visit_times.setdefault(new_edge, []).append(float(tnow))
        log_decision(tnow, new_from, cands, integral, new_edge, probs,
                     reason, conf, male_dir, window_s,
                     detail, floor, arrived_by, back_rule, ambiguous)

    for i in range(len(t) - 1):
        tnow = float(t[i])
        dt = float(t[i + 1] - t[i])
        if dt <= 0:
            continue

        elen = max(g.length_px(edge), 1e-9)

        if state == "ON_EDGE":
            progress += dt * advance(i) / elen
            if progress >= 1.0:
                # the node has been reached; the window is anchored here and keeps filling
                # through the settle period so that the turn itself lands inside it
                state = "DECIDING"
                t_node = tnow
                w_start = t_node - APPROACH_S
                settle_left = SETTLE_S
            else:
                push_yaw(tnow, yaw[i], dt)

            if pending is not None:
                if progress < REVISE_FRAC:
                    # The choice could not be made on angles alone — two ways on were within
                    # AMBIGUOUS_DEG of each other — so the walker keeps listening while it
                    # is still at the start of the new edge. This is rule 7's "use what comes
                    # next": the signal is the only thing that can separate passages the map
                    # cannot, and a moment later it has more of the turn in it. The ranking is
                    # recomputed from the running total on the same two lists of passages, and
                    # the floor is halved because the window is now shorter and the noise
                    # floor scales with the window.
                    pending["integral"] += yaw[i] * dt
                    # The passage we are standing on was counted as walked when we entered
                    # it, a fraction of a second ago, by this very decision. Scoring the
                    # revision with that count in place asks whether to stay on a passage
                    # that now looks spent, against one that looks unused, which is not a
                    # question about the fly at all: it is the novelty penalty arguing for
                    # its own reversal. It made a deferred decision flip to its alternative
                    # and then report the flip as a 0.93-confidence recovery, on a junction
                    # where the two ways on were mirror images and both scores were equal to
                    # six decimal places. Measured on VID00006 it never fired — no RECOVERY
                    # in any run — but on the synthetic symmetric junction it fired every
                    # time. The history is therefore read as it stood *before* this
                    # traversal: the count loses the entry just made, and the last-visit
                    # time falls back to the previous one, or to nothing if this is the
                    # first time the passage has been walked at all.
                    hist = visit_times.get(edge, [])
                    rev_visits = dict(visits)
                    rev_last = dict(last_visit)
                    earlier = hist[:-1]
                    rev_visits[edge] = len(earlier)
                    if earlier:
                        rev_last[edge] = earlier[-1]
                    else:
                        rev_last.pop(edge, None)
                    if pending.get("ways_on") is not None:
                        res2 = geo.decide(
                            pending["ways_on"], pending["behind"], pending["integral"],
                            yaw_floor(pending.get("window_s", APPROACH_S),
                                      floor_scale) * 0.5,
                            rev_visits, rev_last, tnow, back_factor=BACK_FACTOR,
                            male_bonus=MALE_BONUS, back_penalty_min=BACK_PENALTY,
                            decay=NOVELTY_DECAY, recent_penalty=RECENT_PENALTY,
                            return_s=RETURN_S)
                        scored2 = res2["scored"]
                        probs2 = res2["probs"]
                        male2 = res2["male_direction"]
                    else:
                        scored2, male2 = [], {}
                        probs2 = {}
                    top = max(probs2, key=lambda k: probs2[k]) if probs2 else edge
                    if top != edge and probs2.get(top, 0.0) >= RECOVER_MARGIN:
                        decisions[-1]["reason"] = "RECOVERY"
                        decisions[-1]["chosen_edge"] = top
                        decisions[-1]["yaw_integral"] = round(float(pending["integral"]), 4)
                        decisions[-1]["male_direction"] = male2
                        decisions[-1]["probabilities"] = "|".join(
                            f"{k}:{v:.2f}" for k, v in probs2.items())
                        decisions[-1]["confidence"] = round(probs2[top], 3)
                        # A sequence entry is appended only when an edge is left, so the
                        # entry for the abandoned edge does not exist yet: the entries that
                        # do exist are the ones for edges completed during the excursion.
                        # Popping the last one therefore removed the wrong record. The count
                        # taken when the decision was made is used instead, which discards
                        # exactly the excursion and nothing else.
                        del seq[pending["seq_len"]:]
                        edge = top
                        frm = pending["node"]
                        to = g.other(top, frm)
                        t_edge_start = float(pending["time"])
                        # the walker has been moving through the settle period and the
                        # revision, so it is already partway along the passage it is really
                        # on; starting it at zero would lose that stretch
                        new_len = max(g.length_px(top), 1e-9)
                        moved = (tnow - pending["t_start"]) * pix_per_sec * \
                            float(speed_rel[i])
                        progress = min(max(moved / new_len, 0.0), 0.95)
                        carry_px = 0.0
                        zone_t = None
                        buf.clear()
                        pending = None
                else:
                    pending = None

        elif state == "DECIDING":
            # the window keeps filling through the settle period, so a turn that lands
            # just after the node is still counted
            push_yaw(tnow, yaw[i], dt)
            progress += dt * advance(i) / elen
            # remember how far past the node we have gone; it is carried onto the next
            # passage rather than discarded
            carry_px = max((progress - 1.0) * elen, 0.0)
            settle_left -= dt
            if settle_left > 0:
                continue

            I, win_s = window_integral(tnow)
            floor = yaw_floor(win_s, floor_scale)

            # Every passage at this node, with its real deflection from the current heading.
            # The way on and the way back are separated by the angle itself rather than by a
            # label, and a reversal is offered to the decision only when there is a reason to
            # believe one — see `p084b_geometry.decide`, which is also what the frozen replay
            # calls, so this run and that replay cannot disagree about the rule.
            all_at_node = g.classify_candidates(to, edge, allow_back=True)
            ways_on = [c for c in all_at_node
                       if c["edge"] != edge and not geo.is_reversal(c["deg"])]
            behind = [c for c in all_at_node if geo.is_reversal(c["deg"])]
            if all(c["edge"] != edge for c in behind):
                # the passage just walked is the reversal that matters; an edge pointing the
                # same way by another route is offered beside it, since either could be taken
                behind = [c for c in all_at_node if c["edge"] == edge] + behind

            if not ways_on and not behind:
                warnings.append(f"{tnow:.1f} с: узел {to} вообще без выхода")
                progress = 0.99
                state = "ON_EDGE"
                continue

            # The third reason a reversal may be considered — "a return from a dead end" — is
            # off, and deliberately. At a real dead end there is no way on and the clause
            # above has already returned the walker; every broader reading of it puts the
            # passage behind back among the candidates at junctions that have good ways on,
            # which is the reversal-at-a-junction defect the P08.1 rules removed. The
            # parameter exists so the claim can be measured; `p084b_geometry.decide` records
            # the reasoning in full.
            res = geo.decide(ways_on, behind, I, floor, visits, last_visit, tnow,
                             back_factor=BACK_FACTOR,
                             back_from_dead_end=False,
                             male_bonus=MALE_BONUS, back_penalty_min=BACK_PENALTY,
                             decay=NOVELTY_DECAY, recent_penalty=RECENT_PENALTY,
                             return_s=RETURN_S, defer_margin=MALE_MARGIN)

            # ---- P09.2: is this a turn, or only a glance? ------------------------------------------
            # The direction stays MaleCNS's business. This filter answers a different question - whether the
            # rotation the fly reported was the body turning or the head looking - and its only power is to
            # *withdraw* a direction, never to supply one or to flip one. A glance leaves and comes back, so
            # the direction is evidence about the head; the route is then chosen without it, which at J35 is
            # what keeps the walker out of the dead end.
            #
            # Note where the evidence comes from in time: the window runs to the end of the swing, which is
            # just after the junction. So this is applied on what follows the node, not at the instant the
            # node is reached. That is the structure P08.6 already has, and `unknown` therefore means
            # "not yet" rather than "no".
            # The direction is remembered before anything can withdraw it. A revoked direction
            # must stay visible in the log: what the fly said is a fact about the fly, and a
            # reader has to be able to tell "it said nothing" from "it said LEFT and that LEFT
            # was taken away".
            yaw_direction_said = res["male_direction"]
            lt_info = {"score": "", "class": "", "pair_start": "", "pair_end": "",
                       "male_used": bool(res["male_direction"]), "route": ""}
            if look_turn is not None:
                if res["male_direction"]:
                    sc = look_turn.score(tnow)
                    lt_info.update({"score": sc["score"] if sc["score"] is not None else "",
                                    "class": sc["class"],
                                    "pair_start": sc["pair_start"] if sc["pair_start"] is not None else "",
                                    "pair_end": sc["pair_end"] if sc["pair_end"] is not None else ""})
                    if sc["class"] == "TURN":
                        lt_info["male_used"] = True
                        lt_info["route"] = "MALE_TURN"
                    elif sc["class"] == "LOOK":
                        lt_info["male_used"] = False
                        lt_info["route"] = "MALE_LOOK_REVOKED"
                    else:
                        lt_info["male_used"] = not unknown_revokes
                        lt_info["route"] = "LOOK_TURN_UNKNOWN"
                    if not lt_info["male_used"]:
                        # retake the decision with the direction off the table entirely: the integral is
                        # zero, so no passage is promoted, and geometry and memory decide alone
                        res = geo.decide(ways_on, behind, 0.0, floor, visits, last_visit, tnow,
                                         back_factor=BACK_FACTOR, back_from_dead_end=False,
                                         male_bonus=MALE_BONUS, back_penalty_min=BACK_PENALTY,
                                         decay=NOVELTY_DECAY, recent_penalty=RECENT_PENALTY,
                                         return_s=RETURN_S, defer_margin=MALE_MARGIN)
                else:
                    lt_info["route"] = "MALE_SILENT"

            if res["chosen"] is None:
                warnings.append(f"{tnow:.1f} с: узел {to} вообще без выхода")
                progress = 0.99
                state = "ON_EDGE"
                continue

            scored = res["scored"]
            chosen = res["chosen"]["edge"]
            by = {s["edge"]: s for s in scored}
            # the log keeps the graph's own candidate shape, so every reader of decisions.csv
            # goes on working, while the numbers behind the choice travel beside it
            cands = [{"edge": s["edge"], "to": s["to"], "deg": s["deg"], "side": s["display"]}
                     for s in scored]
            detail = {
                "signal_strength": round(float(abs(I)), 4),
                "yaw_strength": round(float(res["ramp"]), 4),
                "angle_degrees": "|".join(f"{s['deg']:.1f}" for s in scored),
                "geometry_scores": "|".join(f"{s['geometry_score']:.3f}" for s in scored),
                "back_penalties": "|".join(f"{s['back_penalty']:.3f}" for s in scored),
                "male_scores": "|".join(f"{s['male_score']:.3f}" for s in scored),
                "alignments": "|".join(f"{s['alignment']:.3f}" for s in scored),
                "final_scores": "|".join(f"{s['final_score']:.3f}" for s in scored),
                "novelty_scores": "|".join(f"{s['novelty']:.3f}" for s in scored),
                "visit_counts": "|".join(str(s["visits"]) for s in scored),
                "deferred": int(res["deferred"]),
                # the two lists the decision ran on, so a deferred decision can be re-run on
                # exactly the same inputs rather than rebuilding them from the graph
                "_ways_on": ways_on, "_behind": behind,
            }
            # when the filter had something to say about the direction, its verdict is the
            # reason, so that a revoked direction is visible in the log rather than hidden
            final_reason = res["reason"]
            if lt_info["route"] in ("MALE_TURN", "MALE_LOOK_REVOKED", "LOOK_TURN_UNKNOWN"):
                final_reason = lt_info["route"]
            detail["_lt"] = lt_info
            detail["yaw_direction"] = yaw_direction_said
            move_onto(chosen, to, by[chosen]["to"], final_reason, tnow, cands, I,
                      res["probs"], res["confidence"], res["male_direction"],
                      res["ambiguous"] or res["deferred"], win_s, detail, floor,
                      res["back_rule"], res["ambiguous"])
            continue


        a = np.array(g.pos(frm), dtype=float)
        b = np.array(g.pos(to), dtype=float)
        p = min(max(progress, 0.0), 1.0)
        xy = a + (b - a) * p
        path.append({"t": tnow, "edge": edge, "from": frm, "to": to,
                     "progress": round(p, 4),
                     "x_norm": round(float(xy[0] / g.img_w), 6),
                     "y_norm": round(float(xy[1] / g.img_h), 6),
                     "x_px": round(float(xy[0]), 2), "y_px": round(float(xy[1]), 2)})

    close_edge(t[-1])
    return {"decisions": decisions, "edge_sequence": seq, "path": path,
            "warnings": warnings, "final_edge": edge, "final_node": to,
            "final_progress": progress, "carry_lost_px": carry_lost}


def sensitivity(g: Graph, t: np.ndarray, yaw: np.ndarray, rel: np.ndarray,
                pps: float, args, out: Path) -> None:
    """What believing weaker evidence costs.

    The floor sits at the 95th percentile of a non-turning window, so at full strength a
    direction is named on roughly one junction in twenty by chance. Halving it admits more
    real turns and more noise together, and there is no ground truth here to say which is
    better, so both sides of the trade are printed instead of one being chosen.
    """
    print("=== ЧУВСТВИТЕЛЬНОСТЬ К ПОРОГУ ===")
    print("  порог = множитель × измеренный уровень шума (p95 шума = 1.00)")
    print("  'названо' — решения, где сигнал назвал сторону")
    print("  'против' — из них те, где сторона названа, а прохода на ней нет")
    print()
    print(f"  {'множ.':>6s} {'порог, p50':>11s} {'названо':>8s} {'из решений':>11s} "
          f"{'MALE_*':>7s} {'против':>7s} {'доля против':>12s} {'разных рёбер':>13s} "
          f"{'BACK':>5s}")
    rows = []
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.0):
        res = track(g, t, yaw, rel, pix_per_sec=pps, node_zone=args.node_zone,
                    floor_scale=scale)
        dec = res["decisions"]
        choice = [d for d in dec
                  if len([x for x in d["candidate_edges"].split("|") if x]) >= 2]
        named = [d for d in dec if d["male_direction"]]
        contra = sum(1 for d in named
                     if d["male_direction"] not in d["candidate_sides"].split("|"))
        male = sum(1 for d in dec if d["reason"].startswith("MALE"))
        fl = np.median([d["yaw_floor"] for d in choice]) if choice else 0.0
        row = {"floor_scale": scale, "floor_median": float(fl),
               "named": len(named), "n_choice": len(choice),
               "male_decisions": male, "contradictions": contra,
               "contradiction_rate": contra / max(len(named), 1),
               "distinct_edges": len({s["edge"] for s in res["edge_sequence"]}),
               "n_back": sum(1 for d in dec if d["reason"] == "BACK")}
        rows.append(row)
        print(f"  {scale:6.2f} {fl:11.3f} {len(named):8d} {len(choice):11d} "
              f"{male:7d} {contra:7d} {row['contradiction_rate'] * 100:11.0f}% "
              f"{row['distinct_edges']:13d} {row['n_back']:5d}")
    print()
    print(f"  для справки: рёбер в графе {len(g.edges)}, развилок {len(g.choice_nodes())}")
    print()
    print("  при множителе 1.00 сигнал называют редко; вниз по таблице он начинает")
    print("  называть сторону чаще, но доля противоречий при этом не растёт до единицы,")
    print("  а держится заметно ниже случайного уровня около 50 процентов.")
    print("  выбор между строками — не вопрос арифметики: он зависит от того, что")
    print("  важнее, не выдумывать повороты или не терять настоящие.")
    with (out / "floor_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}/floor_sensitivity.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID00001")
    ap.add_argument("--graph", default=str(GRAPH_PATH))
    ap.add_argument("--pix-per-sec", type=float, default=None,
                   help="override; by default derived from --walk-mps and the plan scale")
    ap.add_argument("--walk-mps", type=float, default=WALK_MPS,
                   help="walking speed in metres per second, used with the plan scale")
    ap.add_argument("--floor-scale", type=float, default=FLOOR_SCALE,
                   help="множитель к измеренному шумовому порогу; 1.0 = уровень p95 шума")
    ap.add_argument("--sensitivity", action="store_true",
                   help="прогнать набор множителей и вывести таблицу вместо одного прогона")
    ap.add_argument("--pace", choices=("moving", "as-is"), default="moving",
                   help="moving: --walk-mps applies to a frame that shows movement, "
                        "which is what relative_speed normalises to")
    ap.add_argument("--node-zone", type=float, default=NODE_ZONE)
    ap.add_argument("--yaw-column", default="yaw_signal_deadband")
    # P09.2. Off by default so that earlier runs stay reproducible; when on, a direction that
    # the filter calls a glance is withdrawn before the route is chosen.
    ap.add_argument("--look-turn", action="store_true",
                    help="P09.2: снять направление, если колебание оказалось взглядом")
    ap.add_argument("--unknown-keeps-direction", action="store_true",
                    help="при LOOK_TURN_UNKNOWN всё же использовать направление; по умолчанию "
                         "не использовать, как требует P09.2")
    ap.add_argument("--t0", type=float, default=None, help="restrict to a segment")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--start-edge", default=None,
                   help="edge to begin on, when graph.json carries no start")
    ap.add_argument("--start-from", default=None,
                   help="node to travel from; defaults to the edge's own `from`")
    ap.add_argument("--start-node", default=None,
                   help="node to begin at; with --start-from it means standing at that "
                        "node having come from that neighbour")
    ap.add_argument("--start-at", default=None,
                   help="node the walker is standing at, paired with --start-from")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = Graph.load(args.graph)

    print("P08 — маршрут как проход по графу")
    print(f"  граф: {args.graph}")
    print(f"  узлов {len(g.nodes)}, рёбер {len(g.edges)}, план {g.img_w:.0f}x{g.img_h:.0f}")
    probs = g.validate()
    for p in probs:
        print(f"  замечание к графу: {p}")
    if not probs:
        print("  граф без замечаний")
    s = g.start
    if args.start_at:
        if not args.start_from:
            raise SystemExit("--start-at требует --start-from: нужно указать, откуда пришли")
        g.start = {"type": "at_node", "node": args.start_at, "from_node": args.start_from}
        s = g.start
    elif args.start_node:
        g.start = {"type": "node", "node": args.start_node}
        s = g.start
    if args.start_edge:
        e = g.edges.get(args.start_edge)
        if not e:
            raise SystemExit(f"ребра {args.start_edge} нет в графе")
        frm = args.start_from or e["from"]
        g.start = {"type": "edge", "edge": args.start_edge, "from": frm,
                   "to": g.other(args.start_edge, frm), "hypothesis": True}
        s = g.start
    if s:
        if s.get("type") == "at_node" or ("from_node" in s and "node" in s):
            desc = (f"в узле {s['node']}, пришёл со стороны "
                    f"{s.get('from_node') or s.get('from')}")
        elif s.get("type") == "node" or "node" in s and "edge" not in s:
            desc = f"узел {s.get('node')}"
        else:
            desc = f"{s.get('from')} → {s.get('to')} по {s.get('edge')}"
        print(f"  старт: {desc}"
              + ("  ← задан ключом, не из файла"
                 if (args.start_edge or args.start_node or args.start_at) else "")
              + ("  ← помечен как гипотеза, не подтверждён"
                 if s.get("hypothesis") else ""))
        if s.get("note"):
            print(f"         {s['note']}")
    else:
        print("  старт не задан: укажите --start-edge или внесите start в graph.json")
    print()

    t, yaw, tf, speed, alt = load_signals(args.video, args.yaw_column)
    if args.t0 is not None:
        m = t >= args.t0
        t, yaw = t[m], yaw[m]
        speed = np.interp(t, tf, speed)
        alt = {k: np.interp(t, tf, v) for k, v in alt.items()}
    if args.t1 is not None:
        m = t <= args.t1
        t, yaw = t[m], yaw[m]
        speed = speed[m]
        alt = {k: v[m] for k, v in alt.items()}
    rel = relative_speed(speed)
    pps, how = (pix_per_sec_for(g, args.walk_mps) if args.pix_per_sec is None
                else (args.pix_per_sec, "задано ключом --pix-per-sec"))
    pps, m_rel = normalise_pace(rel, pps)
    pace_note = ""
    print(f"  {args.video}: {t[0]:.1f}..{t[-1]:.1f} с ({len(t)} кадров)")
    print(f"  сигнал поворота: {args.yaw_column}, ненулевых {np.mean(yaw != 0) * 100:.0f}%")
    print(f"  скорость: относительная в диапазоне {rel.min():.2f}..{rel.max():.2f}, "
          f"среднее {np.mean(rel):.2f} — это доля времени, когда видео показывает движение")
    print(f"  темп продвижения {pps:.1f} план-пикселей в секунду ({how}{pace_note})")
    if g.meters_per_pixel:
        # the average speed is the pace times the mean of the relative speed, always:
        # with --pace mean the division by that mean has already cancelled it out
        nom = pps * float(g.meters_per_pixel)
        print(f"  кадр движения {nom:.2f} м/с, потолок {MAX_MPS:.2f} м/с")
        print(f"  ожидаемая средняя за клип {nom * float(np.mean(rel)):.2f} м/с "
              f"(видео показывает движение {np.mean(rel) * 100:.0f}% времени)")
        # a junction decision masks this much walking, so any passage shorter than it
        # cannot be timed on its own
        print(f"  предел разрешения: за {SETTLE_S:.1f} с решения проходится до "
              f"{MAX_MPS * SETTLE_S:.1f} м, рёбра короче этого по времени не измерить")
    print()

    if args.sensitivity:
        return sensitivity(g, t, yaw, rel, pps, args, out)

    look_turn = None
    if args.look_turn:
        try:
            look_turn = lookturn.LookTurn(args.video)
            print(f"  P09.2 включён: порог {lookturn.LOOK_TURN_THRESHOLD}, "
                  f"колебаний разобрано {len(look_turn.swing)}")
            print(f"  при LOOK_TURN_UNKNOWN направление "
                  f"{'сохраняется' if args.unknown_keeps_direction else 'снимается'}")
        except Exception as e:
            print(f"  P09.2 недоступен ({e}); правило не применяется")
    res = track(g, t, yaw, rel, pix_per_sec=pps, node_zone=args.node_zone,
                floor_scale=args.floor_scale, look_turn=look_turn,
                unknown_revokes=not args.unknown_keeps_direction)

    print("=== ПОСЛЕДОВАТЕЛЬНОСТЬ РЁБЕР ===")
    for s in res["edge_sequence"]:
        length = g.length_m(s["edge"])
        print(f"  {s['time_start']:8.1f} → {s['time_end']:8.1f} с  "
              f"{s['from']} → {s['to']}  ({s['edge']}, {s['duration']:.1f} с"
              + (f", {length:.1f} м)" if length else ")"))
    total_m = g.route_meters([s["edge"] for s in res["edge_sequence"]])
    print()
    masked = [s for s in res["edge_sequence"] if s.get("pass_masked")]
    print(f"  всего рёбер {len(res['edge_sequence'])}, "
          f"разных {len({s['edge'] for s in res['edge_sequence']})} из {len(g.edges)}")
    print(f"  проходов короче {SUB_RESOLUTION_S:.2f} с (время не измерено): {len(masked)}"
          + (f" — {', '.join(s['edge'] for s in masked[:4])}" if masked else ""))
    if total_m:
        dur = t[-1] - t[0]
        print(f"  пройдено {total_m:.0f} м за {dur:.0f} с → {total_m / dur:.2f} м/с")
    print()
    print(f"=== РЕШЕНИЯ ({len(res['decisions'])}) ===")
    print(f"  {'время':>8s} {'узел':>5s} {'накопл. yaw':>12s} {'male':>6s} "
          f"{'кандидаты':>16s} {'выбрано':>8s} {'причина':>17s} {'увер.':>6s}")
    for d in res["decisions"]:
        print(f"  {d['time']:8.1f} {d['node']:>5s} {d['yaw_integral']:12.3f} "
              f"{d['male_direction'] or '—':>6s} {d['candidate_edges']:>16s} "
              f"{d['chosen_edge']:>8s} {d['reason']:>17s} {d['confidence']:5.2f}")
    print()

    reasons = {}
    for d in res["decisions"]:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    print("=== СВОДКА ===")
    for k in ("ONLY_OPTION", "GEOMETRY", "STRAIGHT", "MALE_LEFT", "MALE_RIGHT", "BACK",
              "DELAYED_DECISION", "RECOVERY"):
        if k in reasons:
            print(f"  {k:>17s}: {reasons[k]}")
    n_choice = sum(1 for d in res["decisions"] if len(d["candidate_edges"].split("|")) > 1)
    print(f"  решений всего {len(res['decisions'])}, из них с реальным выбором {n_choice}")
    print(f"  MaleCNS спрашивался {sum(1 for d in res['decisions'] if d['male_direction'])} раз")
    for w in res["warnings"][:5]:
        print(f"  предупреждение: {w}")

    with (out / "decisions.csv").open("w", newline="") as f:
        if res["decisions"]:
            w = csv.DictWriter(f, fieldnames=list(res["decisions"][0].keys()))
            w.writeheader()
            w.writerows(res["decisions"])
    with (out / "edge_sequence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_start", "time_end", "edge", "from", "to",
                                          "duration", "pass_masked", "length_m"])
        w.writeheader()
        w.writerows(res["edge_sequence"])
    with (out / "graph_trajectory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["path"][0].keys()))
        w.writeheader()
        w.writerows(res["path"])

    report = {
        "video": args.video, "graph": args.graph,
        "speed_cap_mps": MAX_MPS,
        "pace_nominal_mps": pps * float(g.meters_per_pixel) if g.meters_per_pixel else None,
        "rel_mean_is_moving_fraction": float(np.mean(rel)),
        "predicted_mean_mps": (pps * float(np.mean(rel)) * float(g.meters_per_pixel)
                               if g.meters_per_pixel else None),
        "pace_after_cap_mps": min(pps, (MAX_MPS / float(g.meters_per_pixel))
                                 if g.meters_per_pixel else pps)
                              * float(g.meters_per_pixel) if g.meters_per_pixel else None,
        "carry_truncated_m": (res.get("carry_lost_px", 0.0) *
                              float(g.meters_per_pixel)) if g.meters_per_pixel else None,
        "params": {"pix_per_sec": pps, "walk_mps": args.walk_mps, "pace": args.pace,
                   "floor_scale": args.floor_scale,
                   "speed_source": how, "node_zone": args.node_zone,
                   "settle_s": SETTLE_S, "floor_a": FLOOR_A, "floor_b": FLOOR_B,
                   "male_margin": MALE_MARGIN, "revise_frac": REVISE_FRAC,
                   "speed_floor": SPEED_FLOOR, "yaw_column": args.yaw_column},
        "graph_problems": probs,
        "n_decisions": len(res["decisions"]),
        "n_with_choice": n_choice,
        "n_masked_passes": len([s for s in res["edge_sequence"]
                                if s.get("pass_masked")]),
        "n_male_consulted": sum(1 for d in res["decisions"] if d["male_direction"]),
        "reasons": reasons,
        "edge_sequence": res["edge_sequence"],
        "final_edge": res["final_edge"], "final_node": res["final_node"],
        "warnings": res["warnings"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
