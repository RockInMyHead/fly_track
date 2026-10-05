// Local Python backend: workspace sync, start, stop.
//
// Installed layout (resources/runtime, read-only):
//   python/    portable CPython with numpy, scipy, numba, opencv, flybrain
//   ffmpeg/    ffmpeg.exe, ffprobe.exe
//   malecns/   brain.npz, weights.npz (FLY_DATA)
//   app/       code + shared seed data; copied to the writable workspace on each new build
//
// The workspace (userData/workspace) holds camera videos, brain records and routes, so
// reinstalling or updating the app never touches them.

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { app } = require('electron');

const IS_WIN = process.platform === 'win32';

function runtimeDir() {
  return path.join(process.resourcesPath, 'runtime');
}

function layout() {
  if (app.isPackaged) {
    const rt = runtimeDir();
    return {
      mode: 'packaged',
      runtime: rt,
      python: process.env.FLY_PYTHON || path.join(rt, 'python', IS_WIN ? 'python.exe' : 'bin/python3'),
      ffmpegDir: path.join(rt, 'ffmpeg'),
      brainDir: path.join(rt, 'malecns'),
      appSource: path.join(rt, 'app'),
      workspace: process.env.FLY_WORKSPACE || path.join(app.getPath('userData'), 'workspace'),
    };
  }
  const project = process.env.FLY_PROJECT || path.resolve(__dirname, '..', '..');
  return {
    mode: 'source',
    runtime: null,
    python: process.env.FLY_PYTHON || path.join(project, '.venv', IS_WIN ? 'Scripts/python.exe' : 'bin/python'),
    ffmpegDir: null,
    brainDir: null,
    appSource: null,
    workspace: project,
  };
}

function logsDir() {
  const dir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function buildStamp(appSource) {
  try {
    return fs.readFileSync(path.join(appSource, 'BUILD.txt'), 'utf8').trim();
  } catch {
    return `version ${app.getVersion()}`;
  }
}

function syncWorkspace(L, log) {
  if (!L.appSource) return;
  const stampFile = path.join(L.workspace, '.fly_track_build');
  const want = buildStamp(L.appSource);
  let have = '';
  try {
    have = fs.readFileSync(stampFile, 'utf8').trim();
  } catch {
    /* first start */
  }
  if (have === want && fs.existsSync(path.join(L.workspace, 'desktop', 'fly_track_app.py'))) return;
  log(`обновляю рабочую папку: ${have || 'пусто'} -> ${want}`);
  fs.mkdirSync(L.workspace, { recursive: true });
  fs.cpSync(L.appSource, L.workspace, { recursive: true, force: true });
  fs.writeFileSync(stampFile, want);
  if (IS_WIN) clearReadOnly(L.workspace);
}

function clearReadOnly(root) {
  const walk = (p) => {
    let st;
    try {
      st = fs.statSync(p);
    } catch {
      return;
    }
    try {
      fs.chmodSync(p, st.isDirectory() ? 0o755 : 0o666);
    } catch {
      /* ignore */
    }
    if (st.isDirectory()) {
      for (const name of fs.readdirSync(p)) walk(path.join(p, name));
    }
  };
  walk(root);
}

function childEnv(L) {
  const env = { ...process.env };
  if (L.ffmpegDir) env.PATH = `${L.ffmpegDir}${path.delimiter}${env.PATH || ''}`;
  if (!IS_WIN) env.PATH = `/opt/homebrew/bin:/usr/local/bin:${env.PATH || ''}`;
  if (L.brainDir) env.FLY_DATA = L.brainDir;
  env.FLY_APP = '1';
  env.PYTHONUTF8 = '1';
  env.PYTHONIOENCODING = 'utf-8';
  env.PYTHONNOUSERSITE = '1';
  env.MPLBACKEND = 'Agg';
  env.NUMBA_CACHE_DIR = path.join(L.workspace, 'output', 'numba_cache');
  delete env.PYTHONHOME;
  delete env.PYTHONPATH;
  return env;
}

function createBackend({ onExit } = {}) {
  const L = layout();
  const logFile = path.join(logsDir(), 'backend.log');
  const out = fs.createWriteStream(logFile, { flags: 'a' });
  const log = (line) => out.write(`[${new Date().toISOString()}] ${line}\n`);
  let child = null;
  let stopping = false;

  const start = () =>
    new Promise((resolve, reject) => {
      try {
        syncWorkspace(L, log);
      } catch (e) {
        reject(new Error(`не удалось подготовить рабочую папку ${L.workspace}: ${e.message}`));
        return;
      }
      if (!fs.existsSync(L.python)) {
        reject(new Error(`не найден Python: ${L.python}`));
        return;
      }
      const script = path.join(L.workspace, 'desktop', 'fly_track_app.py');
      log(`старт: ${L.python} ${script} (рабочая папка ${L.workspace})`);
      child = spawn(L.python, ['-X', 'utf8', script, '--headless', '--port', '0'], {
        cwd: L.workspace,
        env: childEnv(L),
        windowsHide: true,
        detached: !IS_WIN,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let done = false;
      const timer = setTimeout(() => {
        if (!done) {
          done = true;
          reject(new Error('локальный сервер не ответил за 120 с'));
        }
      }, 120000);
      let buf = '';
      const onData = (chunk) => {
        const text = chunk.toString('utf8');
        out.write(text);
        buf += text;
        const m = buf.match(/READY (http:\/\/127\.0\.0\.1:\d+)/);
        if (m && !done) {
          done = true;
          clearTimeout(timer);
          resolve(m[1]);
        }
        if (buf.length > 20000) buf = buf.slice(-2000);
      };
      child.stdout.on('data', onData);
      child.stderr.on('data', onData);
      child.on('error', (e) => {
        log(`ошибка запуска: ${e.message}`);
        if (!done) {
          done = true;
          clearTimeout(timer);
          reject(e);
        }
      });
      child.on('exit', (code, signal) => {
        log(`сервер завершился: code=${code} signal=${signal}`);
        const wasReady = done;
        if (!done) {
          done = true;
          clearTimeout(timer);
          reject(new Error(`локальный сервер завершился при запуске (код ${code}). Журнал: ${logFile}`));
        }
        child = null;
        if (wasReady && !stopping && onExit) onExit(code);
      });
    });

  const stop = () => {
    stopping = true;
    if (!child) return;
    const pid = child.pid;
    try {
      if (IS_WIN) spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true });
      else process.kill(-pid, 'SIGTERM');
    } catch {
      try {
        child.kill();
      } catch {
        /* already gone */
      }
    }
  };

  return { start, stop, layout: L, logFile, logsDir: logsDir() };
}

module.exports = { createBackend };
