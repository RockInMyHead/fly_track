#!/usr/bin/env python3
"""End-to-end check of an assembled Fly Track: fake camera -> import -> pipeline -> V1..V5.

    python desktop/smoke_test.py --app DIR [--python EXE] [--seconds 12]

DIR is an app/ folder made by build_bundle.py (or an installed {app}\\app). A synthetic
MJPEG AVI is written into a temporary DCIM folder, the app is started headless with
FLY_CAMERA_ROOTS pointing there, and everything goes through the same HTTP API as the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def call(base: str, path: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"), strict=False)
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode("utf-8"), strict=False)


def wait(what: str, fn, limit_s: float, every: float = 5):
    t0 = time.time()
    last = None
    while time.time() - t0 < limit_s:
        ok, msg = fn()
        if msg != last:
            print(f"  [{time.time() - t0:5.0f} s] {what}: {msg}", flush=True)
            last = msg
        if ok is not None:
            return ok
        time.sleep(every)
    sys.exit(f"FAIL: {what} не закончилось за {limit_s:.0f} с")


def make_camera(root: Path, seconds: int) -> Path:
    avi = root / "DCIM" / "100MEDIA" / "VID00001.AVI"
    avi.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=1280x720:rate=30:duration={seconds}",
                    "-c:v", "mjpeg", "-q:v", "5", str(avi)], check=True)
    return avi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seconds", type=int, default=12)
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    app = Path(a.app).resolve()
    if os.sep in a.python or (os.altsep and os.altsep in a.python):
        a.python = os.path.abspath(a.python)
    base = f"http://127.0.0.1:{a.port}"

    tmp = Path(tempfile.mkdtemp(prefix="flytrack_cam_"))
    make_camera(tmp, a.seconds)
    env = dict(os.environ, FLY_CAMERA_ROOTS=str(tmp))
    log = open(app / "output" / "smoke_server.log", "w", encoding="utf-8")
    srv = subprocess.Popen([a.python, "-X", "utf8", str(app / "desktop" / "fly_track_app.py"),
                            "--headless", "--port", str(a.port)],
                           cwd=str(app), env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        def up():
            if srv.poll() is not None:
                sys.exit(f"FAIL: приложение завершилось с кодом {srv.returncode}")
            try:
                return (True, "ok") if call(base, "/api/app/clips").get("ok") else (None, "нет ответа")
            except OSError as e:
                return None, type(e).__name__
        wait("запуск", up, 120, 1)

        scan = call(base, "/api/camera/scan")
        files = [f for c in scan.get("cameras", []) for f in c["files"]]
        names = [f["name"] for f in files]
        print(f"  камера: {names}")
        if names != ["VID00001.AVI"]:
            sys.exit(f"FAIL: ожидался один VID00001.AVI, найдено {names}")
        g = json.loads((app / "data" / "p08" / "graph.json").read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in g["nodes"]}
        e = g["edges"][0]
        p, q = nodes[e["from"]], nodes[e["to"]]
        start = {"x": (p["x"] + q["x"]) / 2 * g["img_w"], "y": (p["y"] + q["y"]) / 2 * g["img_h"],
                 "toward": e["to"]}
        r = call(base, "/api/camera/import", {"paths": [files[0]["path"]], "start": start})
        if not r.get("ok"):
            sys.exit(f"FAIL: импорт не начался: {r}")

        def imported():
            s = call(base, "/api/camera/import")
            st = s.get("state")
            if st == "error":
                sys.exit(f"FAIL: импорт: {s}")
            return (True if st == "done" else None), f"{st} {s.get('files_done')}/{s.get('files_total')}"
        wait("копирование", imported, 120, 1)

        got = call(base, "/api/camera/import").get("imported") or []
        if len(got) != 1:
            sys.exit(f"FAIL: ожидался один новый ролик, получено {got}")
        clip = got[0]

        def processed():
            c = next(x for x in call(base, "/api/app/clips")["clips"] if x["id"] == clip)
            if c.get("status") == "error":
                sys.exit(f"FAIL: шаг {c.get('step')}: {c.get('error')}\n{c.get('log', '')[-2000:]}")
            return (True if c.get("status") in ("done", "ready") else None), \
                f"{c.get('status')} {c.get('step') or ''}"
        wait(f"обработка {clip}", processed, 1800)

        def trackers():
            c = next(x for x in call(base, "/api/app/clips")["clips"] if x["id"] == clip)
            st = c.get("track_status")
            if st == "error":
                log = call(base, "/api/start/status").get("log", "")
                sys.exit(f"FAIL: трекеры: {c.get('track_error')}\n{log[-3000:]}")
            return (True if st == "done" else None), str(st)
        wait("маршрут V1–V5 (сам после обработки)", trackers, 1800)

        bad = []
        for v in ("v1", "v2", "v3", "v4", "v5"):
            st = call(base, f"/api/final/state?clip={clip}&ver={v}")
            n = len(st.get("trajectory") or [])
            print(f"  {v}: {n} точек траектории")
            if n < a.seconds - 2:
                bad.append(v)
        if bad:
            sys.exit(f"FAIL: нет траектории у {bad}")
        print("SMOKE OK")
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(15)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
