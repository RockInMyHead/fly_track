import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Minus, Plus } from "lucide-react";
import type { Graph, Place, RunState, TrajRow } from "@/lib/api";
import { VERSION_INFO } from "@/lib/api";
import { Button } from "./ui";

type View = { s: number; tx: number; ty: number };

export type PlanMode = "start" | "direction" | "view";

function rowAt(rows: TrajRow[], t: number): TrajRow | null {
  if (!rows.length) return null;
  let lo = 0;
  let hi = rows.length - 1;
  if (t <= rows[0].t) return rows[0];
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (rows[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return rows[lo];
}

export default function PlanView({
  graph,
  planUrl,
  place,
  toward,
  mode,
  runs,
  time,
  onPlanClick,
}: {
  graph: Graph | null;
  planUrl: string;
  place: Place | null;
  toward: string | null;
  mode: PlanMode;
  runs: RunState[];
  time: number | null;
  onPlanClick: (x: number, y: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [view, setView] = useState<View | null>(null);
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null);

  useEffect(() => {
    const im = new Image();
    im.onload = () => setImg(im);
    im.src = planUrl;
  }, [planUrl]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const nodes = useMemo(() => {
    const m = new Map<string, { x: number; y: number }>();
    if (graph) for (const n of graph.nodes) m.set(n.id, { x: n.x * graph.img_w, y: n.y * graph.img_h });
    return m;
  }, [graph]);

  const fit = useCallback(
    (pts?: { x: number; y: number }[]) => {
      const all = pts && pts.length > 1 ? pts : [...nodes.values()];
      if (!all.length) return;
      const xs = all.map((p) => p.x);
      const ys = all.map((p) => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = Math.max(maxX - minX, maxY - minY) * 0.08 + 40;
      const bw = maxX - minX + pad * 2;
      const bh = maxY - minY + pad * 2;
      const s = Math.min(size.w / bw, size.h / bh);
      setView({ s, tx: (size.w - (maxX + minX) * s) / 2, ty: (size.h - (maxY + minY) * s) / 2 });
    },
    [nodes, size],
  );

  useEffect(() => {
    if (!view && nodes.size && size.w > 10) fit();
  }, [nodes, size, view, fit]);

  const runsKey = runs.map((r) => `${r.version}:${r.rows.length}`).join(",");
  useEffect(() => {
    if (!runs.length) return;
    const pts = runs.flatMap((r) => r.rows.map((p) => ({ x: p.x, y: p.y })));
    fit(pts);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runsKey]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !view) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(size.w * dpr);
    cv.height = Math.round(size.h * dpr);
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0b1424";
    ctx.fillRect(0, 0, cv.width, cv.height);
    const { s, tx, ty } = view;
    ctx.setTransform(s * dpr, 0, 0, s * dpr, tx * dpr, ty * dpr);
    const px = (n: number) => n / s;

    if (img && graph) {
      ctx.globalAlpha = 0.9;
      ctx.drawImage(img, 0, 0, graph.img_w, graph.img_h);
      ctx.globalAlpha = 1;
      const placing = mode === "start" || mode === "direction";
      ctx.fillStyle = placing ? "rgba(7, 17, 31, 0.38)" : "rgba(7, 17, 31, 0.62)";
      ctx.fillRect(0, 0, graph.img_w, graph.img_h);

      if (placing && graph.edges?.length) {
        ctx.strokeStyle = "rgba(74, 222, 128, 0.72)";
        ctx.lineWidth = px(6);
        ctx.lineJoin = "round";
        for (const e of graph.edges) {
          const a = nodes.get(e.from);
          const b = nodes.get(e.to);
          if (!a || !b) continue;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    ctx.lineCap = "round";
    const ordered = [...runs].sort((a, b) => (a.version === "v5" ? 1 : b.version === "v5" ? -1 : 0));
    for (const r of ordered) {
      if (r.rows.length < 2) continue;
      ctx.strokeStyle = VERSION_INFO[r.version].color;
      ctx.lineWidth = px(r.version === "v5" ? 5 : 3);
      ctx.lineJoin = "round";
      ctx.globalAlpha = r.version === "v5" || runs.length === 1 ? 1 : 0.75;
      ctx.beginPath();
      ctx.moveTo(r.rows[0].x, r.rows[0].y);
      for (const p of r.rows) ctx.lineTo(p.x, p.y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (place) {
      const end = toward && mode === "view" ? place.ends.find((e) => e.id === toward) : null;
      if (end) {
        const dx = end.x - place.x;
        const dy = end.y - place.y;
        const len = Math.hypot(dx, dy) || 1;
        const L = px(46);
        const ux = dx / len, uy = dy / len;
        const hx = place.x + ux * L, hy = place.y + uy * L;
        ctx.strokeStyle = "#fb923c";
        ctx.fillStyle = "#fb923c";
        ctx.lineWidth = px(4);
        ctx.beginPath();
        ctx.moveTo(place.x, place.y);
        ctx.lineTo(hx, hy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(hx + ux * px(12), hy + uy * px(12));
        ctx.lineTo(hx - uy * px(8), hy + ux * px(8));
        ctx.lineTo(hx + uy * px(8), hy - ux * px(8));
        ctx.closePath();
        ctx.fill();
      }
      ctx.fillStyle = "#fb923c";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = px(3);
      ctx.beginPath();
      ctx.arc(place.x, place.y, px(9), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    if (time !== null) {
      for (const r of ordered) {
        const p = rowAt(r.rows, time);
        if (!p) continue;
        ctx.fillStyle = VERSION_INFO[r.version].color;
        ctx.strokeStyle = p.stop ? "#fb923c" : "#0b1424";
        ctx.lineWidth = px(p.stop ? 4 : 2.5);
        ctx.beginPath();
        ctx.arc(p.x, p.y, px(r.version === "v5" ? 10 : 7), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }, [view, size, img, graph, nodes, runs, place, toward, mode, time]);

  const toWorld = (cx: number, cy: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const v = view!;
    return { x: (cx - rect.left - v.tx) / v.s, y: (cy - rect.top - v.ty) / v.s };
  };

  const zoomAt = (factor: number, cx?: number, cy?: number) => {
    if (!view) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const ox = cx === undefined ? size.w / 2 : cx - rect.left;
    const oy = cy === undefined ? size.h / 2 : cy - rect.top;
    const s = Math.max(0.02, Math.min(8, view.s * factor));
    const k = s / view.s;
    setView({ s, tx: ox - (ox - view.tx) * k, ty: oy - (oy - view.ty) * k });
  };

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden rounded-b-lg">
      <canvas
        ref={canvasRef}
        className={mode === "view" ? "cursor-grab active:cursor-grabbing" : "cursor-crosshair"}
        style={{ width: size.w, height: size.h, display: "block" }}
        onWheel={(e) => zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX, e.clientY)}
        onPointerDown={(e) => {
          if (!view) return;
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d || !view) return;
          const dx = e.clientX - d.x;
          const dy = e.clientY - d.y;
          if (!d.moved && Math.hypot(dx, dy) < 4) return;
          d.moved = true;
          setView({ ...view, tx: d.tx + dx, ty: d.ty + dy });
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (d && !d.moved && view && mode !== "view") {
            const w = toWorld(e.clientX, e.clientY);
            onPlanClick(w.x, w.y);
          }
        }}
      />
      <div className="absolute right-3 top-3 flex flex-col gap-1.5">
        <Button variant="secondary" size="icon" onClick={() => zoomAt(1.3)} title="Приблизить">
          <Plus className="h-4 w-4" />
        </Button>
        <Button variant="secondary" size="icon" onClick={() => zoomAt(1 / 1.3)} title="Отдалить">
          <Minus className="h-4 w-4" />
        </Button>
        <Button variant="secondary" size="icon" onClick={() => fit()} title="Вся карта">
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
