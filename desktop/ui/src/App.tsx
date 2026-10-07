import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Activity,
  Camera,
  CheckCircle2,
  Circle,
  Compass,
  FileVideo,
  History,
  Loader2,
  MapPinned,
  Navigation,
  RotateCcw,
  Route,
  RefreshCw,
  Upload,
  XCircle,
} from "lucide-react";
import {
  api,
  desktop,
  VERSION_INFO,
  VERSIONS,
  type Clip,
  type ClipsResponse,
  type Graph,
  type Place,
  type RunState,
  type StartPoint,
  type Version,
} from "@/lib/api";
import { cn, formatBytes, formatDuration } from "@/lib/utils";
import PlanView, { type PlanMode } from "./components/PlanView";
import CameraDialog from "./components/CameraDialog";
import HistoryPanel, { clipBadge } from "./components/HistoryPanel";
import BatchProgressCard from "./components/BatchProgressCard";
import DiagnosticsDialog from "./components/DiagnosticsDialog";
import { Badge, Button, Card, CardHeader, Progress } from "./components/ui";

function StepIcon({ state }: { state: "done" | "active" | "todo" | "error" }) {
  if (state === "done") return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (state === "active") return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
  if (state === "error") return <XCircle className="h-4 w-4 text-destructive" />;
  return <Circle className="h-4 w-4 text-muted-foreground/50" />;
}

export default function App() {
  const [graph, setGraph] = useState<Graph | null>(null);
  const [planUrl, setPlanUrl] = useState("/api/p08/plan");
  const [data, setData] = useState<ClipsResponse | null>(null);
  const [backendDown, setBackendDown] = useState(false);

  const [place, setPlace] = useState<Place | null>(null);
  const [toward, setToward] = useState<string | null>(null);
  const [mode, setMode] = useState<PlanMode>("start");

  const [activeId, setActiveId] = useState<string | null>(null);
  const [awaitingImport, setAwaitingImport] = useState(false);
  const [batchIds, setBatchIds] = useState<string[]>([]);
  const [batchFilesTotal, setBatchFilesTotal] = useState(0);
  const [batchDismissed, setBatchDismissed] = useState(false);
  const [runs, setRuns] = useState<RunState[]>([]);
  const [hidden, setHidden] = useState<Set<Version>>(new Set());
  const [videoUrl, setVideoUrl] = useState("");
  const [time, setTime] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const [cameraOpen, setCameraOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [diagOpen, setDiagOpen] = useState(false);

  useEffect(() => {
    api.graph().then((g) => {
      setGraph(g.graph);
      setPlanUrl(g.plan_url);
    }).catch((e) => toast.error(`План не загрузился: ${e.message}`));
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await api.clips();
        if (alive) {
          setData(r);
          setBackendDown(false);
        }
      } catch {
        if (alive) setBackendDown(true);
      }
    };
    void tick();
    const id = setInterval(tick, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const deepLinked = useRef(false);
  useEffect(() => {
    if (deepLinked.current || !data) return;
    deepLinked.current = true;
    const id = new URLSearchParams(window.location.search).get("clip");
    if (id && data.clips.some((c) => c.id === id)) void openClip(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const imp = data?.import;
  useEffect(() => {
    if (!data?.import?.imported?.length || batchIds.length) return;
    if (data.import.state === "running" || data.import.state === "done") {
      setBatchIds(data.import.imported);
      setBatchFilesTotal(data.import.files_total);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.import?.state, data?.import?.imported?.length]);

  useEffect(() => {
    if (!imp) return;
    if (imp.imported?.length) {
      setBatchIds((prev) => {
        const next = [...prev];
        for (const id of imp.imported) if (!next.includes(id)) next.push(id);
        return next;
      });
    }
    if (imp.files_total) setBatchFilesTotal(imp.files_total);
    if (awaitingImport && imp.state === "done") setAwaitingImport(false);
    if (awaitingImport && imp.state === "error") {
      setAwaitingImport(false);
      toast.error(imp.error || "Копирование остановилось");
    }
  }, [awaitingImport, imp]);

  const clip: Clip | null = useMemo(() => data?.clips.find((c) => c.id === activeId) ?? null, [data, activeId]);

  const clipStartKey = clip?.start ? `${clip.id}|${clip.start.x}|${clip.start.y}|${clip.start.toward}` : "";
  useEffect(() => {
    const s = clip?.start;
    if (!s) return;
    let alive = true;
    api
      .place(s.x, s.y, s.toward)
      .then((p) => {
        if (!alive) return;
        setPlace(p);
        setToward(s.toward);
        setMode("view");
      })
      .catch((e) => {
        toast.error((e as Error).message || "Старт из записи ролика не подошёл к плану");
        setPlace(null);
        setToward(null);
        setMode("start");
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipStartKey]);

  const runsKey = clip ? `${clip.id}|${clip.track_status}|${JSON.stringify(clip.runs)}` : "";
  useEffect(() => {
    if (!clip) {
      setRuns([]);
      setVideoUrl("");
      return;
    }
    let alive = true;
    const vers = VERSIONS.filter((v) => clip.runs?.[v]);
    if (clip.track_status === "running" || clip.track_status === "queued" || clip.track_status === "waiting") {
      setRuns([]);
    } else {
      Promise.all(vers.map((v) => api.run(clip.id, v).catch(() => null))).then((rs) => {
        if (alive) setRuns(rs.filter(Boolean) as RunState[]);
      });
    }
    if (clip.has_video || clip.tracker_ready) {
      api.videoUrl(clip.id).then((u) => alive && setVideoUrl(u)).catch(() => {});
    }
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runsKey, clip?.has_video]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const loop = () => {
      setTime(v.currentTime);
      if (!v.paused) raf = requestAnimationFrame(loop);
    };
    const onPlay = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    };
    const onSeek = () => setTime(v.currentTime);
    v.addEventListener("play", onPlay);
    v.addEventListener("seeked", onSeek);
    v.addEventListener("loadedmetadata", onSeek);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("seeked", onSeek);
      v.removeEventListener("loadedmetadata", onSeek);
    };
  }, [videoUrl]);

  const start: StartPoint | null = place && toward ? { x: place.x, y: place.y, toward } : null;

  const onPlanClick = useCallback(
    async (x: number, y: number) => {
      try {
        if (mode === "start" || !place) {
          const p = await api.place(x, y);
          setPlace(p);
          setToward(null);
          setMode("direction");
          toast.success("Старт на проходе — укажите направление");
          return;
        }
        if (mode === "direction") {
          const p = await api.placeDirection(place, x, y);
          setPlace(p);
          setToward(p.toward ?? null);
          setMode("view");
          toast.success("Направление задано");
        }
      } catch (e) {
        toast.error((e as Error).message);
      }
    },
    [mode, place],
  );

  const resetStart = () => {
    setPlace(null);
    setToward(null);
    setMode("start");
  };

  const importPaths = async (paths: string[]) => {
    if (!start) return;
    try {
      setBatchIds([]);
      setBatchFilesTotal(paths.length);
      setBatchDismissed(false);
      setActiveId(null);
      setRuns([]);
      await api.importFiles(paths, start);
      setAwaitingImport(true);
      setHistoryOpen(false);
      toast.success("Копируем видео — прогресс в блоке «Загрузка с камеры»");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const pickFile = async () => {
    if (!desktop) return;
    const paths = await desktop.pickVideos();
    if (paths.length) await importPaths(paths);
  };

  const rebuild = async () => {
    if (!clip || !start) return;
    try {
      await api.track(clip.id, start);
      toast.success(`Строим маршрут ${clip.id} с нового старта`);
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const retryTrack = async () => {
    if (!clip) return;
    try {
      await api.retryTrack(clip.id);
      toast.success(`Снова строим маршрут ${clip.id}`);
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const openClip = async (id: string) => {
    setActiveId(id);
    setBatchDismissed(true);
    setHistoryOpen(false);
    setTime(null);
    const c = data?.clips.find((x) => x.id === id);
    if (c?.start) {
      try {
        const p = await api.place(c.start.x, c.start.y, c.start.toward);
        setPlace(p);
        setToward(c.start.toward);
        setMode("view");
      } catch (e) {
        toast.error((e as Error).message || "Старт из записи ролика не подошёл к плану");
        setPlace(null);
        setToward(null);
        setMode("start");
      }
    }
  };

  const importing = imp?.state === "running";
  const importPct = imp && imp.total_bytes ? (imp.done_bytes / imp.total_bytes) * 100 : 0;
  const batchPanelOpen =
    !batchDismissed && (importing || awaitingImport || batchIds.length > 0) && !!imp;
  const processing = clip && (clip.status === "queued" || clip.status === "running");
  const tracking = clip && (clip.track_status === "waiting" || clip.track_status === "queued" || clip.track_status === "running");
  const shownRuns = runs.filter((r) => !hidden.has(r.version));
  const startChanged =
    clip && start && clip.tracker_ready && !tracking &&
    (!clip.start || clip.start.toward !== start.toward || Math.hypot(clip.start.x - start.x, clip.start.y - start.y) > 1);

  const stepsDone = clip?.steps.filter((s) => s.done).length ?? 0;
  const procPct = clip ? ((stepsDone + (clip.step_progress ?? 0)) / clip.steps.length) * 100 : 0;

  const hint =
    mode === "start"
      ? "Кликните примерно где человек стоял — точку перенесёт к ближайшему проходу"
      : mode === "direction"
        ? "Кликните куда угодно на плане в сторону хода — проход выберется сам"
        : start
          ? "Старт и направление заданы"
          : "";

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b bg-card/60 px-6 py-3 backdrop-blur">
        <div className="flex items-center gap-3">
          <img src="/ui/favicon.png" alt="" className="h-9 w-9 rounded-lg" />
          <div>
            <div className="text-lg font-bold leading-tight">Fly Track</div>
            <div className="text-xs text-muted-foreground">Маршрут человека по видео с камеры на плане цеха</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {backendDown && <Badge tone="destructive">локальный сервер не отвечает</Badge>}
          {data?.tracker?.state === "running" && (
            <Badge tone="warning">
              <Loader2 className="h-3 w-3 animate-spin" /> трекер: {data.tracker.clip}
            </Badge>
          )}
          <Button variant="outline" onClick={() => setHistoryOpen(true)}>
            <History className="h-4 w-4" /> История
          </Button>
          <Button variant="outline" onClick={() => setDiagOpen(true)}>
            <Activity className="h-4 w-4" /> Проверка системы
          </Button>
        </div>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-[440px_1fr] gap-4 p-4">
        <div className="flex min-h-0 flex-col gap-4 overflow-auto pr-1">
          <Card>
            <CardHeader title="Видео с камеры" icon={<Camera className="h-4 w-4" />} />
            <div className="space-y-4 p-5">
              <p className="text-sm leading-relaxed text-muted-foreground">
                Сначала укажите старт и направление на плане. Затем выберите файл на диске или загрузите видео с экшен-камеры — Fly Track
                проанализирует его и покажет маршрут. Если выбрать несколько роликов, они идут по порядку имён, и каждый
                следующий начинается там, где закончился предыдущий.
              </p>
              <div className={cn("grid gap-2", desktop ? "grid-cols-2" : "grid-cols-1")}>
                {desktop && (
                  <Button variant="secondary" size="lg" disabled={!start || importing} onClick={() => void pickFile()}>
                    <FileVideo className="h-5 w-5" /> Выбрать файл
                  </Button>
                )}
                <Button size="lg" disabled={!start || importing} onClick={() => setCameraOpen(true)}>
                  <Upload className="h-5 w-5" /> Загрузить с камеры
                </Button>
              </div>
              {!start && (
                <div className="flex items-start gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
                  <MapPinned className="mt-0.5 h-4 w-4 shrink-0" />
                  Перед загрузкой задайте стартовую точку и направление движения на плане.
                </div>
              )}
            </div>
          </Card>

          {batchPanelOpen && imp && (
            <BatchProgressCard
              imp={imp}
              clips={data?.clips ?? []}
              batchIds={batchIds}
              filesTotal={batchFilesTotal}
              onOpenClip={(id) => void openClip(id)}
              onDismiss={() => {
                if (batchIds[0]) void openClip(batchIds[0]);
                else setBatchDismissed(true);
              }}
            />
          )}

          {!batchPanelOpen && clip && (
            <Card>
              <CardHeader
                title={clip.id}
                icon={<Route className="h-4 w-4" />}
                right={clipBadge(clip)}
              />
              <div className="space-y-5 p-5">
                <div className="-mt-1 text-xs text-muted-foreground">{clip.title}{clip.duration_s ? ` · ${formatDuration(clip.duration_s)}` : ""}</div>

                {clip.status === "done" && (
                  <div className="flex items-center gap-2.5 text-sm">
                    <StepIcon state="done" />
                    <span className="font-medium">Обработка видео</span>
                    <span className="ml-auto text-xs text-muted-foreground">анализ завершён</span>
                  </div>
                )}

                {clip && clip.status !== "external" && clip.status !== "done" && (
                  <div>
                    <div className="mb-2 flex justify-between text-sm">
                      <span className="font-medium">Обработка видео</span>
                      <span className="tabular-nums text-muted-foreground">{Math.round(procPct)}%</span>
                    </div>
                    <Progress value={procPct} />
                    <div className="mt-3 space-y-1.5">
                      {clip.steps.map((s) => {
                        const st = s.done ? "done" : clip.step === s.id ? (clip.status === "error" ? "error" : "active") : "todo";
                        return (
                          <div key={s.id} className="flex items-center gap-2.5 text-xs">
                            <StepIcon state={st} />
                            <span className={cn(st === "todo" && "text-muted-foreground")}>{s.title}</span>
                            {st === "active" && clip.step_progress != null && (
                              <span className="ml-auto tabular-nums text-muted-foreground">{Math.round(clip.step_progress * 100)}%</span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    {clip.status === "error" && (
                      <div className="mt-3 rounded-md bg-destructive/10 p-3 text-xs text-destructive">
                        {clip.error}
                        <Button size="sm" variant="destructive" className="mt-2" onClick={() => void api.retry(clip.id)}>
                          <RefreshCw className="h-3.5 w-3.5" /> Повторить
                        </Button>
                      </div>
                    )}
                  </div>
                )}

                {clip && (clip.track_status || clip.status === "done") && (
                  <div className="flex items-center gap-2.5 text-sm">
                    <StepIcon
                      state={
                        tracking ? "active" : clip.track_status === "error" ? "error" : runs.length ? "done" : "todo"
                      }
                    />
                    <span className="font-medium">Маршрут V1–V5</span>
                    <span className="ml-auto text-xs text-muted-foreground">
                      {clip.track_status === "waiting" && `ждёт конца маршрута ${clip.chain_from}`}
                      {clip.track_status === "queued" && (processing ? "после обработки" : "в очереди")}
                      {clip.track_status === "running" && "считаем…"}
                      {clip.track_status === "error" && "ошибка"}
                      {!clip.track_status && !runs.length && "задайте старт и нажмите «Построить»"}
                    </span>
                  </div>
                )}
                {clip && (clip.chain_from || clip.start?.from_clip) && (
                  <div className="text-xs text-muted-foreground">
                    Старт: продолжение {clip.start?.from_clip ?? clip.chain_from} — с конца его маршрута V3 и в том же направлении
                  </div>
                )}
                {clip?.track_status === "error" && (
                  <div className="rounded-md bg-destructive/10 p-3 text-xs text-destructive">
                    {clip.track_error}
                    <Button size="sm" variant="destructive" className="mt-2" onClick={() => void retryTrack()}>
                      <RefreshCw className="h-3.5 w-3.5" /> Повторить маршрут
                    </Button>
                  </div>
                )}

                {clip && (startChanged || (clip.tracker_ready && !runs.length && !tracking && start)) && (
                  <Button className="w-full" variant="accent" onClick={() => void rebuild()}>
                    <Navigation className="h-4 w-4" /> Построить маршрут с этого старта
                  </Button>
                )}
              </div>
            </Card>
          )}

          {runs.length > 0 && (
            <Card>
              <CardHeader title="Результат" icon={<Compass className="h-4 w-4" />} />
              <div className="divide-y">
                {runs.map((r) => {
                  const s = clip?.runs?.[r.version];
                  const off = hidden.has(r.version);
                  return (
                    <label key={r.version} className="flex cursor-pointer items-center gap-3 px-5 py-2.5 text-sm hover:bg-muted/30">
                      <input
                        type="checkbox"
                        className="h-4 w-4"
                        checked={!off}
                        onChange={() => {
                          const next = new Set(hidden);
                          if (off) next.delete(r.version);
                          else next.add(r.version);
                          setHidden(next);
                        }}
                      />
                      <span className="h-3 w-3 rounded-full" style={{ background: VERSION_INFO[r.version].color }} />
                      <span className={cn("flex-1", r.version === "v5" && "font-semibold")}>{VERSION_INFO[r.version].title}</span>
                      <span className="tabular-nums">{s?.meters ?? "—"} м</span>
                      <span className="w-16 text-right text-xs tabular-nums text-muted-foreground">
                        {s?.stop_fraction != null ? `стоит ${Math.round(s.stop_fraction * 100)}%` : ""}
                      </span>
                    </label>
                  );
                })}
              </div>
            </Card>
          )}

          {videoUrl && (
            <Card className="overflow-hidden">
              <video ref={videoRef} src={videoUrl} controls className="aspect-video w-full bg-black" />
              <div className="px-4 py-2 text-xs text-muted-foreground">
                Точки на плане идут вместе с видео. Оранжевая обводка — человек стоит.
              </div>
            </Card>
          )}
        </div>

        <Card className="flex min-h-0 flex-col">
          <CardHeader
            title="План цеха"
            icon={<MapPinned className="h-4 w-4" />}
            right={
              <div className="flex items-center gap-2">
                <span className="mr-2 text-xs text-muted-foreground">{hint}</span>
                <Button
                  size="sm"
                  variant={mode === "start" ? "default" : "outline"}
                  onClick={() => {
                    setToward(null);
                    setMode("start");
                  }}
                >
                  <MapPinned className="h-3.5 w-3.5" /> Старт
                </Button>
                <Button size="sm" variant={mode === "direction" ? "default" : "outline"} disabled={!place} onClick={() => { setToward(null); setMode("direction"); }}>
                  <Navigation className="h-3.5 w-3.5" /> Направление
                </Button>
                <Button size="sm" variant="ghost" onClick={resetStart}>
                  <RotateCcw className="h-3.5 w-3.5" /> Сбросить
                </Button>
              </div>
            }
          />
          <div className="relative min-h-0 flex-1">
            <PlanView graph={graph} planUrl={planUrl} place={place} toward={toward} mode={mode} runs={shownRuns} time={time} onPlanClick={(x, y) => void onPlanClick(x, y)} />
            {shownRuns.length > 0 && (
              <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border bg-card/90 px-3 py-2 text-xs backdrop-blur">
                {shownRuns.map((r) => (
                  <div key={r.version} className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: VERSION_INFO[r.version].color }} />
                    {r.version.toUpperCase()} · {clip?.runs?.[r.version]?.meters ?? "—"} м
                  </div>
                ))}
                {time !== null && <div className="mt-1 text-muted-foreground">t = {time.toFixed(1)} с</div>}
              </div>
            )}
          </div>
        </Card>
      </main>

      <CameraDialog open={cameraOpen} onClose={() => setCameraOpen(false)} onImport={importPaths} />
      <HistoryPanel open={historyOpen} clips={data?.clips ?? []} active={activeId} onPick={(id) => void openClip(id)} onClose={() => setHistoryOpen(false)} />
      <DiagnosticsDialog open={diagOpen} onClose={() => setDiagOpen(false)} />
    </div>
  );
}
