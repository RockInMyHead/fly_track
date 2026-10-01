#!/usr/bin/env python3
"""Build interactive dashboard: 3D fly + video + trajectory + brain activity."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_vo.brain_recorder import BrainRecorder
from fly_vo.config import FlyVOConfig


def downsample_trajectory(
    csv_path: Path,
    step: int = 40,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> dict:
    df = pd.read_csv(csv_path)
    if start_seconds is not None:
        df = df[df.timestamp >= start_seconds - 1e-6]
    if end_seconds is not None:
        df = df[df.timestamp <= end_seconds + 1e-6]
    if df.empty:
        raise ValueError(
            f"No trajectory points in [{start_seconds}, {end_seconds}] from {csv_path}"
        )
    df = df.iloc[::step].reset_index(drop=True)
    t0 = float(df.timestamp.iloc[0])
    t1 = float(df.timestamp.iloc[-1])
    return {
        "start_seconds": t0,
        "duration": t1 - t0,
        "timestamps": df.timestamp.tolist(),
        "x": df.x.tolist(),
        "y": df.y.tolist(),
        "heading": df.heading_deg.tolist(),
        "speed": df.speed.tolist(),
    }


def trajectory_span(csv_path: Path) -> tuple[float, float]:
    df = pd.read_csv(csv_path)
    t0 = float(df.timestamp.iloc[0])
    t1 = float(df.timestamp.iloc[-1])
    return t0, t1 - t0


def auto_brain_stride(duration_seconds: float, sim_hz: float = 50.0, target_frames: int = 1200) -> int:
    """Keep brain JSON/HTML size manageable on long segments."""
    total = max(1, int(duration_seconds * sim_hz))
    return max(3, int(np.ceil(total / target_frames)))


def export_video_clip(src: Path, dst: Path, start: float, duration: float) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 640 / max(w, h))
    out_w, out_h = int(w * scale), int(h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_w, out_h))
    n = int(duration * fps)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        writer.write(frame)
        if i and i % 900 == 0:
            print(f"  video export: {i}/{n} frames")
    cap.release()
    writer.release()


def encode_h264(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def export_video_ffmpeg(src: Path, dst: Path, start: float, duration: float, width: int = 640) -> None:
    """Stream-export a segment with ffmpeg (works for long clips without OOM)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(src),
            "-t",
            str(duration),
            "-vf",
            f"scale='min({width},iw)':-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )


def brain_from_trajectory(
    csv_path: Path,
    start_seconds: float,
    end_seconds: float,
    stride: int = 25,
    n_epg: int = 8,
) -> dict:
    """Build dashboard brain frames from trajectory CSV (no FlyVis re-run)."""
    df = pd.read_csv(csv_path)
    df = df[(df.timestamp >= start_seconds) & (df.timestamp <= end_seconds)]
    df = df.iloc[::stride].reset_index(drop=True)
    preferred = np.linspace(0, 2 * np.pi, n_epg, endpoint=False)

    frames = []
    for row in df.itertuples(index=False):
        heading_rad = np.radians(row.heading_deg)
        delta = (preferred - heading_rad + np.pi) % (2 * np.pi) - np.pi
        bump = np.exp(-0.5 * (delta / 0.55) ** 2)
        bump = (bump / (bump.sum() + 1e-9)).tolist()

        vx, vy = float(row.vx_body), float(row.vy_body)
        fwd, bwd = max(0.0, vx), max(0.0, -vx)
        lat_r, lat_l = max(0.0, vy), max(0.0, -vy)
        speed = float(row.speed)

        frames.append(
            {
                "t": float(row.timestamp),
                "t4": {"T4a": bwd, "T4b": fwd, "T4c": lat_r, "T4d": lat_l},
                "t5": {"T5a": bwd * 0.85, "T5b": fwd * 0.85, "T5c": lat_r * 0.85, "T5d": lat_l * 0.85},
                "epg": bump,
                "heading": float(row.heading_deg),
                "vx": vx,
                "vy": vy,
                "yaw": float(row.yaw_rate),
                "flow_mag": [speed * 0.08] * 721,
            }
        )

    return {"start_seconds": start_seconds, "sim_hz": 50.0, "frames": frames}


def brain_to_json(recording) -> dict:
    return {
        "start_seconds": recording.start_seconds,
        "sim_hz": recording.sim_hz,
        "frames": [
            {
                "t": s.timestamp,
                "t4": s.t4,
                "t5": s.t5,
                "epg": s.epg_bump,
                "heading": s.heading_deg,
                "vx": s.vx,
                "vy": s.vy,
                "yaw": s.yaw_rate,
                "flow_mag": [
                    float(np.hypot(fx, fy))
                    for fx, fy in zip(s.flow_fx, s.flow_fy)
                ],
            }
            for s in recording.snapshots
        ],
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>Fly Visual Odometry — Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Pro Text', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
  header { padding: 16px 24px; border-bottom: 1px solid #1e293b; }
  header h1 { font-size: 18px; font-weight: 600; }
  header p { font-size: 12px; color: #64748b; margin-top: 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto auto; gap: 12px; padding: 12px; height: calc(100vh - 70px); }
  .panel { background: #161b26; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; overflow: hidden; display: flex; flex-direction: column; }
  .panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 8px; }
  .panel-body { flex: 1; min-height: 0; position: relative; }
  canvas, video { width: 100%; height: 100%; display: block; border-radius: 4px; }
  #fly3d { width: 100%; height: 100%; }
  .controls { grid-column: 1 / -1; display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: #161b26; border: 1px solid #1e293b; border-radius: 8px; }
  .controls input[type=range] { flex: 1; accent-color: #3b82f6; }
  .controls span { font-size: 12px; color: #94a3b8; min-width: 80px; }
  .controls button { background: #1e40af; color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .legend { display: flex; gap: 12px; font-size: 10px; color: #64748b; margin-top: 6px; flex-wrap: wrap; }
  .legend span::before { content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
  .lg-t4::before { background: #22c55e; }
  .lg-t5::before { background: #f97316; }
  .lg-epg::before { background: #a855f7; }
  .lg-traj::before { background: #3b82f6; }
</style>
</head>
<body>
<header>
  <h1>🪰 Fly Visual Odometry — Brain Dashboard</h1>
  <p>VID00001.AVI · connectome-constrained visual system · T4/T5 · EPG ring attractor · path integration</p>
</header>
<div class="grid">
  <div class="panel">
    <h2>3D Fly Model (heading sync)</h2>
    <div class="panel-body"><div id="fly3d"></div></div>
  </div>
  <div class="panel">
    <h2>Source Video</h2>
    <div class="panel-body"><video id="vid" src="PREVIEW_MP4" muted playsinline></video></div>
  </div>
  <div class="panel">
    <h2>Trajectory (segment)</h2>
    <div class="panel-body"><canvas id="traj"></canvas></div>
    <div class="legend"><span class="lg-traj">path</span><span style="color:#22c55e">● start</span><span style="color:#ef4444">● current</span></div>
  </div>
  <div class="panel">
    <h2>Brain Activity — T4/T5 + EPG Compass</h2>
    <div class="panel-body"><canvas id="brain"></canvas></div>
    <div class="legend"><span class="lg-t4">T4 ON-motion</span><span class="lg-t5">T5 OFF-motion</span><span class="lg-epg">EPG bump</span></div>
  </div>
  <div class="controls">
    <button id="playBtn">▶ Play</button>
    <input type="range" id="scrub" min="0" max="1000" value="0"/>
    <span id="timeLabel">0:00</span>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const TRAJ = __TRAJ__;
const TRAJ_FULL = __TRAJ_FULL__;
const BRAIN = __BRAIN__;
const BRAIN_T0 = BRAIN.start_seconds;
const BRAIN_DUR = BRAIN.frames.length > 1 ? BRAIN.frames[BRAIN.frames.length-1].t - BRAIN.frames[0].t : 30;
const TRAJ_T0 = TRAJ.start_seconds;
const TRAJ_T1 = TRAJ_T0 + TRAJ.duration;

// --- Three.js fly ---
const container = document.getElementById('fly3d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1117);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0.5, 3.5);
const renderer = new THREE.WebGLRenderer({ antialias: true });
container.appendChild(renderer.domElement);

const fly = new THREE.Group();
// body
const bodyGeo = new THREE.SphereGeometry(0.35, 16, 12);
bodyGeo.scale(1, 0.7, 1.4);
const bodyMat = new THREE.MeshPhongMaterial({ color: 0x3d2b1f });
fly.add(new THREE.Mesh(bodyGeo, bodyMat));
// head
const headGeo = new THREE.SphereGeometry(0.22, 16, 12);
const head = new THREE.Mesh(headGeo, new THREE.MeshPhongMaterial({ color: 0x2a1f15 }));
head.position.set(0, 0.05, 0.55);
fly.add(head);
// compound eyes (hex-ish)
const eyeGeo = new THREE.SphereGeometry(0.12, 8, 8);
const eyeMat = new THREE.MeshPhongMaterial({ color: 0x991b1b, emissive: 0x330000 });
[-1,1].forEach(s => {
  const eye = new THREE.Mesh(eyeGeo, eyeMat);
  eye.position.set(s*0.18, 0.08, 0.65);
  fly.add(eye);
});
// wings
const wingGeo = new THREE.PlaneGeometry(0.7, 0.35);
const wingMat = new THREE.MeshPhongMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
[-1,1].forEach(s => {
  const wing = new THREE.Mesh(wingGeo, wingMat);
  wing.position.set(s*0.45, 0.1, 0);
  wing.rotation.y = s * 0.4;
  fly.add(wing);
});
scene.add(fly);
scene.add(new THREE.AmbientLight(0x404060, 1.5));
const dl = new THREE.DirectionalLight(0xffffff, 1);
dl.position.set(2, 3, 4);
scene.add(dl);

function resize3d() {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize3d();
window.addEventListener('resize', resize3d);

// --- Canvas helpers ---
function setupCanvas(id) {
  const c = document.getElementById(id);
  const ctx = c.getContext('2d');
  function resize() {
    const p = c.parentElement;
    c.width = p.clientWidth * devicePixelRatio;
    c.height = p.clientHeight * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);
  return { c, ctx, resize };
}

const traj = setupCanvas('traj');
const brain = setupCanvas('brain');
const video = document.getElementById('vid');
const scrub = document.getElementById('scrub');
const timeLabel = document.getElementById('timeLabel');
const playBtn = document.getElementById('playBtn');

let animId = null;
let playing = false;

function fmtTime(s) {
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return m + ':' + String(sec).padStart(2,'0');
}

function drawPath(ctx, xs, ys, sx, sy, color, width, alpha) {
  if (xs.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.beginPath();
  xs.forEach((x, i) => { i === 0 ? ctx.moveTo(sx(x), sy(ys[i])) : ctx.lineTo(sx(x), sy(ys[i])); });
  ctx.stroke();
  ctx.restore();
}

function drawTrajectory(tGlobal) {
  const { c, ctx } = traj;
  const w = c.width/devicePixelRatio, h = c.height/devicePixelRatio;
  ctx.fillStyle = '#0f1117'; ctx.fillRect(0,0,w,h);

  const allX = TRAJ_FULL ? TRAJ_FULL.x.concat(TRAJ.x) : TRAJ.x;
  const allY = TRAJ_FULL ? TRAJ_FULL.y.concat(TRAJ.y) : TRAJ.y;
  const pad = 30;
  const xmin = Math.min(...allX), xmax = Math.max(...allX);
  const ymin = Math.min(...allY), ymax = Math.max(...allY);
  const sx = (v) => pad + (v - xmin)/(xmax-xmin+1e-6)*(w-2*pad);
  const sy = (v) => h - pad - (v - ymin)/(ymax-ymin+1e-6)*(h-2*pad);

  if (TRAJ_FULL && TRAJ_FULL.x.length > 1) {
    drawPath(ctx, TRAJ_FULL.x, TRAJ_FULL.y, sx, sy, '#1e293b', 1, 0.55);
  }
  drawPath(ctx, TRAJ.x, TRAJ.y, sx, sy, '#3b82f6', 2, 1);

  const tClamped = Math.max(TRAJ_T0, Math.min(tGlobal, TRAJ_T1));
  let idx = 0;
  for (let i = 0; i < TRAJ.timestamps.length; i++) {
    if (TRAJ.timestamps[i] <= tClamped) idx = i;
  }
  ctx.fillStyle = '#22c55e'; ctx.beginPath(); ctx.arc(sx(TRAJ.x[0]), sy(TRAJ.y[0]), 5, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(sx(TRAJ.x[idx]), sy(TRAJ.y[idx]), 6, 0, Math.PI*2); ctx.fill();

  ctx.fillStyle = '#64748b'; ctx.font = '11px system-ui';
  const tLocal = Math.max(0, tClamped - TRAJ_T0);
  ctx.fillText('t = ' + fmtTime(tLocal) + ' / ' + fmtTime(TRAJ.duration), pad, 18);
}

function drawBrain(tLocal, fi) {
  const { c, ctx } = brain;
  const w = c.width/devicePixelRatio, h = c.height/devicePixelRatio;
  ctx.fillStyle = '#0f1117'; ctx.fillRect(0,0,w,h);

  const f = BRAIN.frames[fi];

  // T4/T5 bars
  const types = ['a','b','c','d'];
  const labels = types.map(t => 'T4'+t);
  const barW = (w - 60) / 8;
  const maxH = h * 0.35;
  types.forEach((t, i) => {
    const v4 = f.t4['T4'+t] || 0;
    const v5 = f.t5['T5'+t] || 0;
    const x0 = 40 + i * barW * 2;
    ctx.fillStyle = '#22c55e'; ctx.fillRect(x0, h*0.45 - v4*maxH, barW*0.8, v4*maxH);
    ctx.fillStyle = '#f97316'; ctx.fillRect(x0+barW, h*0.45 - v5*maxH, barW*0.8, v5*maxH);
    ctx.fillStyle = '#64748b'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText('T4'+t, x0+barW*0.4, h*0.45+12);
    ctx.fillText('T5'+t, x0+barW*1.4, h*0.45+12);
  });

  // EPG ring
  const cx = w*0.75, cy = h*0.22, R = Math.min(w,h)*0.12;
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI*2); ctx.stroke();
  f.epg.forEach((v, i) => {
    const angle = (i / f.epg.length) * Math.PI*2 - Math.PI/2;
    const len = v * R * 1.8;
    ctx.strokeStyle = `hsl(${270 + v*60}, 70%, 60%)`;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle)*len, cy + Math.sin(angle)*len); ctx.stroke();
  });
  ctx.fillStyle = '#a855f7'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  ctx.fillText('EPG θ=' + f.heading.toFixed(0) + '°', cx, cy + R + 16);

  // Optic flow magnitude (721 hex → radial plot)
  const fx0 = 40, fy0 = h*0.78, fR = Math.min(w,h)*0.15;
  const nFlow = f.flow_mag.length;
  for (let i = 0; i < nFlow; i++) {
    const angle = (i/nFlow)*Math.PI*2;
    const r = fR * (0.2 + f.flow_mag[i]*3);
    ctx.fillStyle = `rgba(59,130,246,${0.3 + f.flow_mag[i]*2})`;
    ctx.beginPath();
    ctx.arc(fx0 + Math.cos(angle)*r*0.5, fy0 + Math.sin(angle)*r*0.5, 1.5, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.fillStyle = '#64748b'; ctx.textAlign = 'left';
  ctx.fillText('Optic flow field (721 hexals)', 20, h*0.68);
  ctx.fillText('vx='+f.vx.toFixed(3)+' vy='+f.vy.toFixed(3)+' ω='+f.yaw.toFixed(3), 20, h-10);
}

function updateFrame(tLocal) {
  const tGlobal = BRAIN_T0 + tLocal;
  drawTrajectory(tGlobal);
  let fi = 0;
  for (let i = 0; i < BRAIN.frames.length; i++) {
    if (BRAIN.frames[i].t <= tGlobal) fi = i;
  }
  drawBrain(tLocal, fi);
  fly.rotation.y = (BRAIN.frames[fi]?.heading || 0) * Math.PI/180;
  fly.position.y = Math.sin(tLocal*8)*0.03;
  renderer.render(scene, camera);
  timeLabel.textContent = fmtTime(tGlobal);
  scrub.value = Math.round(tLocal / BRAIN_DUR * 1000);
}

scrub.addEventListener('input', () => {
  const tLocal = scrub.value / 1000 * BRAIN_DUR;
  video.currentTime = tLocal;
  updateFrame(tLocal);
});

video.addEventListener('timeupdate', () => updateFrame(video.currentTime));

playBtn.addEventListener('click', () => {
  if (playing) { video.pause(); playBtn.textContent = '▶ Play'; playing = false; }
  else { video.play(); playBtn.textContent = '⏸ Pause'; playing = true; }
});

function animate() {
  if (playing && !video.paused) updateFrame(video.currentTime);
  animId = requestAnimationFrame(animate);
}
updateFrame(0);
animate();
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("-o", type=Path, default=Path("output/vid00001_full"))
    p.add_argument("--brain-start", type=float, default=300.0)
    p.add_argument("--brain-duration", type=float, default=30.0)
    p.add_argument(
        "--show-full-trajectory",
        action="store_true",
        help="Draw full trajectory in background when clip is shorter than CSV",
    )
    p.add_argument(
        "--match-trajectory",
        action="store_true",
        help="Use full trajectory CSV span for video + sync (not just --brain-duration)",
    )
    p.add_argument("--brain-stride", type=int, default=0, help="0 = auto from duration")
    args = p.parse_args()

    args.o.mkdir(parents=True, exist_ok=True)

    if args.match_trajectory:
        traj_t0, traj_dur = trajectory_span(args.trajectory)
        args.brain_start = traj_t0
        args.brain_duration = traj_dur
        print(f"Match trajectory: {traj_t0:.0f}s – {traj_t0 + traj_dur:.0f}s ({traj_dur:.0f}s)")

    stride = args.brain_stride or auto_brain_stride(args.brain_duration)
    clip_end = args.brain_start + args.brain_duration
    h264_path = args.o / "preview_h264.mp4"

    use_trajectory_brain = args.match_trajectory or args.brain_duration > 120
    if use_trajectory_brain:
        print(
            f"Using trajectory brain + ffmpeg video ({args.brain_duration:.0f}s, stride={stride})..."
        )
        brain_data = brain_from_trajectory(
            args.trajectory, args.brain_start, clip_end, stride=stride
        )
        print(f"Exporting full video via ffmpeg ({args.brain_duration:.0f}s)...")
        export_video_ffmpeg(args.video, h264_path, args.brain_start, args.brain_duration)
    else:
        print(f"Recording brain activity ({args.brain_duration:.0f}s, stride={stride})...")
        recorder = BrainRecorder(FlyVOConfig())
        recording = recorder.record(
            str(args.video),
            start_seconds=args.brain_start,
            duration_seconds=args.brain_duration,
            stride=stride,
        )
        brain_data = brain_to_json(recording)
        print(f"Exporting video clip ({args.brain_duration:.0f}s)...")
        clip_path = args.o / "preview.mp4"
        export_video_clip(args.video, clip_path, args.brain_start, args.brain_duration)
        print("Encoding H.264 preview...")
        encode_h264(clip_path, h264_path)

    print("Building trajectory data...")
    traj_data = downsample_trajectory(
        args.trajectory,
        start_seconds=args.brain_start,
        end_seconds=clip_end,
    )
    traj_full_data = None
    if args.show_full_trajectory and not args.match_trajectory:
        full_df = pd.read_csv(args.trajectory)
        span = full_df.timestamp.iloc[-1] - full_df.timestamp.iloc[0]
        if span > traj_data["duration"] + 1.0:
            traj_full_data = downsample_trajectory(args.trajectory, step=120)

    html = HTML_TEMPLATE.replace("__TRAJ__", json.dumps(traj_data))
    html = html.replace("__TRAJ_FULL__", json.dumps(traj_full_data))
    html = html.replace("__BRAIN__", json.dumps(brain_data))
    html = html.replace("PREVIEW_MP4", "preview_h264.mp4")

    out_html = args.o / "dashboard.html"
    out_html.write_text(html, encoding="utf-8")

    data_json = args.o / "brain_data.json"
    data_json.write_text(json.dumps({"trajectory": traj_data, "brain": brain_data}, indent=2))

    print(f"Dashboard: {out_html}")
    print(f"Video: {h264_path} ({args.brain_duration:.0f}s)")
    print(f"Open: http://127.0.0.1:8766/dashboard.html")


if __name__ == "__main__":
    main()
