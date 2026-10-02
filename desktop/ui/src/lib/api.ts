export type Version = "v1" | "v2" | "v3" | "v4" | "v5";
export const VERSIONS: Version[] = ["v5", "v4", "v3", "v2", "v1"];
export const VERSION_INFO: Record<Version, { color: string; title: string }> = {
  v5: { color: "#facc15", title: "V5 — маршрут V1, стояние по видео и шагам" },
  v4: { color: "#f43f5e", title: "V4 — развилки V1 + стояние" },
  v3: { color: "#22c55e", title: "V3 — стояние по видео и шагам" },
  v2: { color: "#38bdf8", title: "V2 — стоит по пикселям" },
  v1: { color: "#c084fc", title: "V1 — первая замороженная" },
};

export type Step = { id: string; title: string; done: boolean };
export type StartPoint = { x: number; y: number; toward: string; from_clip?: string };
export type RunSummary = { meters: number | null; stop_fraction: number | null };

export type Clip = {
  id: string;
  title: string;
  status: "queued" | "running" | "done" | "error" | "external";
  step: string | null;
  step_title: string;
  step_progress: number | null;
  step_index?: number;
  steps: Step[];
  error?: string;
  log?: string;
  duration_s?: number;
  has_video: boolean;
  tracker_ready: boolean;
  source_name?: string;
  imported_at?: number;
  start?: StartPoint;
  track_status?: "waiting" | "queued" | "running" | "done" | "error";
  track_error?: string;
  chain_from?: string | null;
  runs: Partial<Record<Version, RunSummary>>;
};

export type ImportStatus = {
  state: "idle" | "running" | "done" | "error";
  file: string;
  done_bytes: number;
  total_bytes: number;
  files_done: number;
  files_total: number;
  error: string;
  imported: string[];
};

export type ClipsResponse = {
  clips: Clip[];
  import: ImportStatus;
  tracker: { state: string; clip: string | null; error: string };
};

export type GraphNode = { id: string; x: number; y: number };
export type GraphEdge = { id: string; from: string; to: string };
export type Graph = { nodes: GraphNode[]; edges: GraphEdge[]; img_w: number; img_h: number; meters_per_pixel: number };

export type Place = {
  ok: boolean;
  error?: string;
  edge: string;
  x: number;
  y: number;
  ends: { id: string; x: number; y: number }[];
  toward?: string;
  start_from?: string;
};

export type TrajRow = { t: number; x: number; y: number; stop: boolean };
export type RunState = {
  version: Version;
  rows: TrajRow[];
  duration: number;
  video_url: string;
  start_xy?: { x: number; y: number };
};

export type CameraFile = {
  path: string;
  name: string;
  size: number;
  mtime_text: string;
  imported_as?: string | null;
};
export type Camera = { root: string; label: string; files: CameraFile[] };

export type Check = { id: string; title: string; ok: boolean; message: string };

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `Ошибка ${res.status}`);
  }
  return data as T;
}

export const api = {
  clips: () => request<ClipsResponse>("/api/app/clips"),
  graph: () => request<{ graph: Graph; plan_url: string }>("/api/app/graph"),
  place: (x: number, y: number, toward?: string) => request<Place>("/api/start/place", { x, y, toward }),
  scanCamera: () => request<{ cameras: Camera[] }>("/api/camera/scan"),
  importFiles: (paths: string[], start: StartPoint) => request<{ ok: true }>("/api/camera/import", { paths, start }),
  track: (clip: string, start: StartPoint) => request<{ ok: true }>("/api/app/track", { clip, ...start }),
  retry: (clip: string) => request<{ ok: true }>("/api/app/retry", { clip }),
  diagnostics: () => request<{ ok: boolean; checks: Check[]; workspace: string }>("/api/app/diagnostics"),
  async run(clip: string, ver: Version): Promise<RunState | null> {
    const j = await request<any>(`/api/final/state?clip=${clip}&ver=${ver}`);
    if (j.version !== ver || !j.trajectory?.length) return null;
    return {
      version: ver,
      rows: j.trajectory.map((r: any) => ({
        t: Number(r.time),
        x: Number(r.x),
        y: Number(r.y),
        stop: r.is_stop === "1" || r.is_stop === 1 || r.is_stop === true,
      })),
      duration: Number(j.duration || 0),
      video_url: j.video_url || "",
      start_xy: j.start_xy,
    };
  },
  async videoUrl(clip: string): Promise<string> {
    const j = await request<any>(`/api/final/state?clip=${clip}`);
    return j.video_url || "";
  },
};

type DesktopBridge = {
  isDesktop: true;
  version: string;
  pickVideos: () => Promise<string[]>;
  openDataFolder: () => Promise<void>;
  openLogs: () => Promise<void>;
};

export const desktop: DesktopBridge | null = (window as any).flytrack ?? null;
