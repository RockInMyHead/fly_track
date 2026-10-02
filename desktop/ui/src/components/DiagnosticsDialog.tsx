import { useEffect, useState } from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { api, desktop, type Check } from "@/lib/api";
import { Button, Dialog } from "./ui";

export default function DiagnosticsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [workspace, setWorkspace] = useState("");
  const [error, setError] = useState("");

  const run = async () => {
    setChecks(null);
    setError("");
    try {
      const r = await api.diagnostics();
      setChecks(r.checks);
      setWorkspace(r.workspace);
    } catch (e) {
      setError(String((e as Error).message));
    }
  };

  useEffect(() => {
    if (open) void run();
  }, [open]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Проверка системы"
      description="Видео → кадры → мозг мухи → трекер → маршрут на плане"
      footer={
        <>
          {desktop && (
            <Button variant="outline" onClick={() => void desktop?.openLogs()}>
              Открыть журналы
            </Button>
          )}
          <Button variant="secondary" onClick={() => void run()} disabled={checks === null && !error}>
            Проверить ещё раз
          </Button>
          <Button onClick={onClose}>Закрыть</Button>
        </>
      }
    >
      {checks === null && !error && (
        <div className="flex items-center gap-3 py-6 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Выполняем проверки…
        </div>
      )}
      {error && <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">Локальный сервер не отвечает: {error}</div>}
      <div className="space-y-2">
        {checks?.map((c) => (
          <div key={c.id} className="flex items-start gap-3 rounded-md border px-3 py-2.5">
            {c.ok ? (
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
            ) : (
              <XCircle className={`mt-0.5 h-4 w-4 shrink-0 ${c.id === "camera" ? "text-muted-foreground" : "text-destructive"}`} />
            )}
            <div className="min-w-0">
              <div className="text-sm font-medium">{c.title}</div>
              <div className="break-all text-xs text-muted-foreground">{c.message}</div>
            </div>
          </div>
        ))}
      </div>
      {workspace && <div className="mt-4 break-all text-xs text-muted-foreground">Данные: {workspace}</div>}
    </Dialog>
  );
}
