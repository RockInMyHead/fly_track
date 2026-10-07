import { useEffect, useState } from "react";
import { Camera as CameraIcon, HardDrive, Loader2, RefreshCw } from "lucide-react";
import { api, type Camera } from "@/lib/api";
import { formatBytes } from "@/lib/utils";
import { Badge, Button, Dialog } from "./ui";

export default function CameraDialog({
  open,
  onClose,
  onImport,
}: {
  open: boolean;
  onClose: () => void;
  onImport: (paths: string[]) => Promise<void>;
}) {
  const [cams, setCams] = useState<Camera[] | null>(null);
  const [error, setError] = useState("");
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);

  const scan = async () => {
    setCams(null);
    setError("");
    try {
      const r = await api.scanCamera();
      setCams(r.cameras);
      setPicked(new Set(r.cameras.flatMap((c) => c.files.map((f) => f.path))));
    } catch (e) {
      setError(String((e as Error).message));
      setCams([]);
    }
  };

  useEffect(() => {
    if (open) void scan();
  }, [open]);

  const files = cams?.flatMap((c) => c.files) ?? [];
  const total = files.filter((f) => picked.has(f.path)).reduce((s, f) => s + f.size, 0);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      wide
      title={
        <span className="flex items-center gap-2">
          <CameraIcon className="h-4 w-4 text-primary" /> Видео с экшен-камеры
        </span>
      }
      description="Подключите камеру по USB — она появится как диск. Файлы только копируются, на камере ничего не меняется."
      footer={
        <>
          <Button variant="outline" onClick={() => void scan()} disabled={cams === null}>
            <RefreshCw className="h-4 w-4" /> Искать снова
          </Button>
          <Button
            disabled={!picked.size || busy}
            onClick={async () => {
              setBusy(true);
              try {
                await onImport([...picked]);
                onClose();
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy && <Loader2 className="h-4 w-4 animate-spin" />}
            Загрузить {picked.size ? `${picked.size} · ${formatBytes(total)}` : ""}
          </Button>
        </>
      }
    >
      {cams === null && (
        <div className="flex items-center gap-3 py-10 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Ищем подключённую камеру…
        </div>
      )}
      {error && <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
      {cams && !cams.length && !error && (
        <div className="flex flex-col items-center gap-3 py-10 text-center text-muted-foreground">
          <HardDrive className="h-10 w-10 opacity-50" />
          <div className="font-medium text-foreground">Камера не найдена</div>
          <div className="max-w-sm text-sm">
            Включите камеру, подключите кабель и дождитесь, пока она появится в «Этом компьютере». Нужна папка DCIM с файлами AVI.
          </div>
        </div>
      )}
      {cams?.map((cam) => (
        <div key={cam.root} className="mb-4">
          <div className="mb-2 flex items-center gap-2 text-sm text-muted-foreground">
            <HardDrive className="h-4 w-4" /> {cam.label} · {cam.files.length} AVI
          </div>
          <div className="overflow-hidden rounded-md border">
            {cam.files.map((f) => (
              <label key={f.path} className="flex cursor-pointer items-center gap-3 border-b px-3 py-2.5 text-sm last:border-b-0 hover:bg-muted/40">
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-[hsl(199_89%_56%)]"
                  checked={picked.has(f.path)}
                  onChange={(e) => {
                    const next = new Set(picked);
                    if (e.target.checked) next.add(f.path);
                    else next.delete(f.path);
                    setPicked(next);
                  }}
                />
                <span className="flex-1 font-medium">{f.name}</span>
                <span className="w-36 text-muted-foreground">{f.mtime_text}</span>
                <span className="w-20 text-right tabular-nums text-muted-foreground">{formatBytes(f.size)}</span>
                <span className="w-28 text-right">
                  {f.imported_as ? (
                    <Badge tone="warning">перезагрузка · {f.imported_as}</Badge>
                  ) : (
                    <Badge tone="primary">новый</Badge>
                  )}
                </span>
              </label>
            ))}
          </div>
        </div>
      ))}
    </Dialog>
  );
}
