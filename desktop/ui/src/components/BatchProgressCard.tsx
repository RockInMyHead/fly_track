import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import type { Clip, ImportStatus } from "@/lib/api";
import { cn, formatBytes, formatDuration } from "@/lib/utils";
import { Badge, Button, Card, CardHeader, Progress } from "./ui";

function PhaseDot({ state }: { state: "done" | "active" | "todo" | "error" }) {
  if (state === "done") return <CheckCircle2 className="h-4 w-4 shrink-0 text-success" />;
  if (state === "active") return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />;
  if (state === "error") return <XCircle className="h-4 w-4 shrink-0 text-destructive" />;
  return <Circle className="h-4 w-4 shrink-0 text-muted-foreground/45" />;
}

function procState(c: Clip): "done" | "active" | "queued" | "error" | "todo" {
  if (c.status === "error") return "error";
  if (c.status === "running" || c.status === "queued") return c.status === "running" ? "active" : "queued";
  if (c.status === "done") return "done";
  return "todo";
}

function trackState(c: Clip): "done" | "active" | "waiting" | "queued" | "error" | "none" {
  const t = c.track_status;
  if (t === "running") return "active";
  if (t === "queued") return "queued";
  if (t === "waiting") return "waiting";
  if (t === "error") return "error";
  if (t === "done") return "done";
  return "none";
}

function trackLabel(c: Clip): string {
  const t = trackState(c);
  if (t === "done") return "маршрут готов";
  if (t === "active") return "считаем маршрут…";
  if (t === "queued") return "маршрут в очереди";
  if (t === "waiting") return `ждёт ${c.chain_from ?? "предшественника"}`;
  if (t === "error") return "маршрут: ошибка";
  if (c.status !== "done") return "после анализа";
  return "нет старта";
}

function procLabel(c: Clip): string {
  const s = procState(c);
  if (s === "done") return "анализ готов";
  if (s === "active") return c.step_title ? c.step_title : "обрабатываем…";
  if (s === "queued") return "в очереди на анализ";
  if (s === "error") return c.error || "ошибка анализа";
  return "—";
}

export default function BatchProgressCard({
  imp,
  clips,
  batchIds,
  filesTotal,
  onOpenClip,
  onDismiss,
}: {
  imp: ImportStatus;
  clips: Clip[];
  batchIds: string[];
  filesTotal: number;
  onOpenClip: (id: string) => void;
  onDismiss: () => void;
}) {
  const copying = imp.state === "running";
  const copyDone = !copying && imp.files_done >= imp.files_total && imp.files_total > 0;
  const copyPct = imp.total_bytes ? (imp.done_bytes / imp.total_bytes) * 100 : imp.files_total ? (imp.files_done / imp.files_total) * 100 : 0;

  const batchClips = batchIds.map((id) => clips.find((c) => c.id === id)).filter(Boolean) as Clip[];
  const procDone = batchClips.filter((c) => c.status === "done").length;
  const procActive = batchClips.some((c) => c.status === "running" || c.status === "queued");
  const procPct = batchClips.length ? (procDone / batchClips.length) * 100 : 0;

  const trackDone = batchClips.filter((c) => c.track_status === "done").length;
  const trackActive = batchClips.some((c) =>
    ["running", "queued", "waiting"].includes(c.track_status ?? ""),
  );
  const trackPct = batchClips.length ? (trackDone / batchClips.length) * 100 : 0;

  const copyPhase: "done" | "active" | "todo" = copying ? "active" : copyDone ? "done" : imp.files_done ? "active" : "todo";
  const procPhase: "done" | "active" | "todo" =
    procDone === batchClips.length && batchClips.length > 0 && !procActive ? "done" : procActive || procDone ? "active" : "todo";
  const trackPhase: "done" | "active" | "todo" =
    trackDone === batchClips.length && batchClips.length > 0 && !trackActive ? "done" : trackActive || trackDone ? "active" : "todo";

  const remainingCopy = Math.max(0, (filesTotal || imp.files_total) - imp.files_done);

  return (
    <Card>
      <CardHeader
        title="Загрузка с камеры"
        icon={<Loader2 className={cn("h-4 w-4", copying && "animate-spin")} />}
        right={
          !copying && batchClips.length > 0 ? (
            <Button size="sm" variant="ghost" onClick={onDismiss}>
              К ролику
            </Button>
          ) : undefined
        }
      />
      <div className="space-y-6 p-5">
        <div className="grid grid-cols-3 gap-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {(
            [
              ["Копирование", copyPhase],
              ["Анализ", procPhase],
              ["Маршрут", trackPhase],
            ] as const
          ).map(([label, state]) => (
            <div key={label} className={cn("flex flex-col items-center gap-1 text-center", state === "active" && "text-primary")}>
              <PhaseDot state={state} />
              <span>{label}</span>
            </div>
          ))}
        </div>

        <section className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">1. Копирование с карты памяти</span>
            <span className="tabular-nums text-muted-foreground">
              {imp.files_done}/{filesTotal || imp.files_total || "—"} файлов
            </span>
          </div>
          <Progress value={copyPct} tone={copyDone ? "success" : "primary"} />
          <div className="flex flex-wrap justify-between gap-1 text-xs text-muted-foreground">
            <span className="truncate">
              {copying && imp.file ? (
                <>
                  <span className="font-medium text-foreground">{imp.file}</span> — копируем и сверяем сумму
                </>
              ) : copyDone ? (
                "Все файлы скопированы и проверены"
              ) : (
                "Ожидание…"
              )}
            </span>
            {imp.total_bytes > 0 && (
              <span className="tabular-nums">
                {formatBytes(imp.done_bytes)} / {formatBytes(imp.total_bytes)}
              </span>
            )}
          </div>
          {remainingCopy > 0 && copying && (
            <div className="text-xs text-muted-foreground">В очереди на копирование: ещё {remainingCopy} файл(ов)</div>
          )}
        </section>

        <section className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">2. Анализ видео</span>
            <span className="tabular-nums text-muted-foreground">
              {procDone}/{batchClips.length || "—"} готово
            </span>
          </div>
          <Progress value={procPct} tone={procPhase === "done" ? "success" : "accent"} />
          <p className="text-xs text-muted-foreground">По одному ролику: кадры, brain, yaw — можно параллельно с копированием.</p>
        </section>

        <section className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">3. Маршрут V1–V5</span>
            <span className="tabular-nums text-muted-foreground">
              {trackDone}/{batchClips.length || "—"} готово
            </span>
          </div>
          <Progress value={trackPct} tone={trackPhase === "done" ? "success" : "accent"} />
          <p className="text-xs text-muted-foreground">Цепочка: следующий ролик ждёт конца маршрута предыдущего (V3).</p>
        </section>

        {batchClips.length > 0 && (
          <div className="max-h-52 overflow-y-auto rounded-md border">
            {batchClips.map((c) => {
              const ps = procState(c);
              const ts = trackState(c);
              return (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => onOpenClip(c.id)}
                  className="flex w-full items-start gap-2 border-b px-3 py-2.5 text-left text-xs last:border-b-0 hover:bg-muted/40"
                >
                  <div className="mt-0.5 w-4 shrink-0">
                    {ps === "active" || ts === "active" ? (
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    ) : ps === "error" || ts === "error" ? (
                      <XCircle className="h-4 w-4 text-destructive" />
                    ) : ps === "done" && ts === "done" ? (
                      <CheckCircle2 className="h-4 w-4 text-success" />
                    ) : (
                      <Circle className="h-4 w-4 text-muted-foreground/40" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-semibold text-foreground">{c.id}</span>
                      {c.source_name && <span className="truncate text-muted-foreground">{c.source_name}</span>}
                      {c.duration_s != null && (
                        <Badge tone="muted">{formatDuration(c.duration_s)}</Badge>
                      )}
                    </div>
                    <div className="mt-0.5 text-muted-foreground">{procLabel(c)}</div>
                    <div className={cn("mt-0.5", ts === "error" && "text-destructive")}>{trackLabel(c)}</div>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {imp.state === "error" && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">{imp.error || "Копирование прервано"}</div>
        )}
      </div>
    </Card>
  );
}
