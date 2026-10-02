"""Fly Track desktop app: a native window over the local tracker server.

Layout of an installed copy (Windows):

    FlyTrack/
        python/        embedded CPython with all packages
        ffmpeg/        ffmpeg.exe, ffprobe.exe
        app/           this repository's code + shared data; writable (videos, outputs)
            desktop/fly_track_app.py   <- this file

The same file runs from a source checkout (python desktop/fly_track_app.py).

    --headless --port N    serve without a window (CI and local smoke tests)
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent
INSTALL = PROJECT.parent
LOCK_PORT = 47615
TITLE = "Fly Track"


def setup_env() -> None:
    for ff in (INSTALL / "ffmpeg", INSTALL / "ffmpeg" / "bin"):
        if (ff / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")).exists():
            os.environ["PATH"] = str(ff) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["FLY_APP"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("FLY_DATA", str(PROJECT / "data" / "malecns"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT / "output" / "numba_cache"))
    for p in (PROJECT / "webapp", PROJECT):
        sys.path.insert(0, str(p))
    os.chdir(PROJECT)
    if sys.stdout is None or sys.stderr is None:
        log = PROJECT / "output" / "app.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = fh
    print(f"\n=== {TITLE} {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


def message(text: str) -> None:
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, TITLE, 0x40)
    else:
        print(text, file=sys.stderr)


def single_instance() -> socket.socket | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
        s.listen(1)
        return s
    except OSError:
        s.close()
        return None


def check_ffmpeg() -> str | None:
    import shutil
    missing = [n for n in ("ffmpeg", "ffprobe") if not shutil.which(n)]
    return f"не найдены: {', '.join(missing)}" if missing else None


def set_window_icon(window) -> None:
    icon = APP_DIR / "assets" / "fly_track.ico"
    if os.name != "nt" or not icon.exists():
        return
    try:
        import clr  # noqa: F401  (pythonnet, comes with pywebview on Windows)
        from System.Drawing import Icon

        def apply():
            window.native.Icon = Icon(str(icon))
        window.native.Invoke(__import__("System").Action(apply))
    except Exception as e:
        print(f"иконка окна не установлена: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()

    setup_env()
    lock = None if a.headless else single_instance()
    if lock is None and not a.headless:
        message("Fly Track уже запущен — посмотрите на панели задач.")
        return 0

    problem = check_ffmpeg()
    if problem:
        message(f"Не хватает ffmpeg ({problem}). Переустановите Fly Track.")

    import server  # webapp/server.py
    import app_backend

    srv = server.create_server("127.0.0.1", a.port)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True, name="http").start()
    url = f"http://127.0.0.1:{port}/app"
    print(f"сервер {url}", flush=True)

    try:
        if a.headless:
            print(f"READY {url}", flush=True)
            while True:
                time.sleep(3600)
        if os.name == "nt":
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FlyTrack.App")
        import webview
        window = webview.create_window(TITLE, url, width=1440, height=920, min_size=(1100, 700),
                                       confirm_close=True, background_color="#0b0d12")
        webview.start(set_window_icon, (window,), localization={
            "global.quitConfirmation": "Закрыть Fly Track? Обработка роликов продолжится при следующем запуске.",
            "global.ok": "Закрыть", "global.cancel": "Отмена",
        })
    except KeyboardInterrupt:
        pass
    finally:
        app_backend.stop_children()
        srv.shutdown()
        if lock:
            lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
