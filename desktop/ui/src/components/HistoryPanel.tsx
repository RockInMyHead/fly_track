import { History, Loader2, X } from "lucide-react";
import type { Clip } from "@/lib/api";
import { cn, formatDuration } from "@/lib/utils";
import { Badge, Button } from "./ui";

export function clipBadge(c: Clip) {
  if (c.status === "error") return <Badge tone="destructive">ошибка</Badge>;
  if (c.status === "queued" || c.status === "running")
    return (
      <Badge tone="primary">
        <Loader2 className="h-3 w-3 animate-spin" /> обработка
      </Badge>
    );
  if (c.track_status === "running" || c.track_status === "queued")
    return (
      <Badge tone="warning">
        <Loader2 className="h-3 w-3 animate-spin" /> маршрут
      </Badge>
    );
  if (c.track_status === "waiting") return <Badge tone="warning">ждёт {c.chain_from}</Badge>;
  if (c.track_status === "error") return <Badge tone="destructive">маршрут: ошибка</Badge>;
  if (c.runs?.v5) return <Badge tone="success">{c.runs.v5.meters} м</Badge>;
  if (Object.keys(c.runs || {}).length) return <Badge tone="success">есть маршрут</Badge>;
  if (c.tracker_ready) return <Badge>готов к старту</Badge>;
  return <Badge>нет данных</Badge>;
}

export default function HistoryPanel({
  open,
  clips,
  active,
  onPick,
  onClose,
}: {
  open: boolean;
  clips: Clip[];
  active: string | null;
  onPick: (id: string) => void;
  onClose: () => void;
}) {
  return (
    <div
      className={cn(
        "fixed inset-y-0 right-0 z-40 flex w-[420px] flex-col border-l bg-card shadow-2xl transition-transform duration-300",
        open ? "translate-x-0" : "translate-x-full",
      )}
    >
      <div className="flex items-center justify-between border-b px-5 py-4">
        <div className="flex items-center gap-2 font-semibold">
          <History className="h-4 w-4 text-primary" /> История
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex-1 overflow-auto p-3">
        {!clips.length && <div className="p-6 text-center text-sm text-muted-foreground">История пока пуста</div>}
        {clips.map((c) => (
          <button
            key={c.id}
            onClick={() => onPick(c.id)}
            className={cn(
              "mb-2 w-full rounded-lg border p-3 text-left transition-colors hover:bg-muted/50",
              active === c.id && "border-primary/60 bg-primary/5",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="font-semibold">{c.id}</div>
              {clipBadge(c)}
            </div>
            <div className="mt-1 truncate text-xs text-muted-foreground">
              {c.title?.replace(`${c.id} — `, "") || c.source_name || ""}
              {c.duration_s ? ` · ${formatDuration(c.duration_s)}` : ""}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
