/** Синхронная панель «муха → влево / прямо / вправо» под камерным видео. */
(function () {
  const RU = { LEFT: 'ВЛЕВО', RIGHT: 'ВПРАВО', STRAIGHT: 'ПРЯМО', SILENT: 'ТИХО' };

  function sampleSeries(series, t) {
    if (!series || !series.length) return null;
    let lo = 0, hi = series.length - 1;
    if (t <= +series[0].t) return series[0];
    if (t >= +series[hi].t) return series[hi];
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (+series[mid].t <= t) lo = mid; else hi = mid;
    }
    return Math.abs(t - +series[lo].t) <= Math.abs(+series[hi].t - t) ? series[lo] : series[hi];
  }

  function bind(opts) {
    const canvas = document.getElementById(opts.canvasId);
    const label = document.getElementById(opts.labelId);
    const video = document.getElementById(opts.videoId);
    const onTime = typeof opts.onTime === 'function' ? opts.onTime : null;
    const getRows = typeof opts.getRows === 'function' ? opts.getRows : () => [];
    const getStand = typeof opts.getStand === 'function' ? opts.getStand : null;
    const blind = !!opts.blind;
    if (!canvas || !label || !video) return { load: async () => {}, draw: () => {} };

    let motion = [];
    let methodNote = '';
    let shownDir = 'STRAIGHT';
    let shownUntil = 0;

    async function load(clip) {
      motion = [];
      const r = await fetch('/api/fly/motion?clip=' + encodeURIComponent(clip));
      const j = await r.json();
      if (j.ok) {
        motion = j.samples || [];
        methodNote = j.note || '';
      }
      draw(video.currentTime || 0);
    }

    function pickDir(m) {
      if (!m) return 'STRAIGHT';
      return m.dir === 'LEFT' || m.dir === 'RIGHT' ? m.dir : 'STRAIGHT';
    }

    function rowAt(rows, t) {
      if (!rows.length) return null;
      let i = rows.findIndex(r => +r.time >= t);
      if (i < 0) i = rows.length - 1;
      if (i > 0 && +rows[i].time - t > t - +rows[i - 1].time) i--;
      return rows[i];
    }

    function pill(ctx, x, y, text, col) {
      ctx.font = '600 12px -apple-system, sans-serif';
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.roundRect(x, y - 13, tw + 16, 19, 9);
      ctx.fill();
      ctx.fillStyle = '#0b0d12';
      ctx.textAlign = 'left';
      ctx.fillText(text, x + 8, y + 1);
      return tw + 16;
    }

    function drawGait(ctx, w, top, t) {
      const rows = getRows() || [];
      const cur = rowAt(rows, t);
      ctx.textAlign = 'left';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillStyle = '#8a94a8';
      if (blind) {
        ctx.fillText('Слепая разметка: решения трекера и мухи скрыты', 12, top + 12);
        drawHuman(ctx, w, top + 36, t, +video.duration || (rows.length ? +rows[rows.length - 1].time : 1));
        return;
      }
      if (!rows.length) {
        ctx.fillText('Стоит / идёт: появится после прогона трекера', 12, top + 14);
        return;
      }
      const stopped = cur && (cur.is_stop === '1' || cur.is_stop === 1);
      ctx.fillText('Кадр (так решил трекер):', 12, top + 12);
      pill(ctx, 160, top + 12, stopped ? 'СТОИТ' : 'ИДЁТ', stopped ? '#f59e0b' : '#22c55e');
      ctx.fillStyle = '#8a94a8';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillText('Муха (MOVE / NO_NET):', 12, top + 36);
      const net = cur ? cur.net_displacement : '';
      const conf = cur ? +cur.net_displacement_confidence : 0;
      const flyMove = net === 'MOVE';
      const pw = pill(ctx, 160, top + 36, net ? (flyMove ? 'ПЕРЕМЕЩЕНИЕ' : 'НА МЕСТЕ') : '—',
        !net ? '#3a4150' : flyMove ? '#22c55e' : '#f59e0b');
      ctx.fillStyle = conf < 0.25 ? '#ef4444' : '#8a94a8';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillText(`уверенность ${conf.toFixed(2)}${conf < 0.25 ? ' — почти наугад' : ''}`, 168 + pw, top + 36);

      const x0 = 44, sw = w - x0 - 10;
      const tEnd = +rows[rows.length - 1].time || 1;
      const sy1 = top + 50, sy2 = top + 68, sh = 13;
      ctx.fillStyle = '#8a94a8';
      ctx.fillText('кадр', 10, sy1 + 10);
      ctx.fillText('муха', 10, sy2 + 10);
      const cell = Math.max(1, sw / rows.length);
      rows.forEach((r) => {
        const x = x0 + (+r.time / tEnd) * sw;
        const st = r.is_stop === '1' || r.is_stop === 1;
        ctx.globalAlpha = 1;
        ctx.fillStyle = st ? '#f59e0b' : '#22c55e';
        ctx.fillRect(x, sy1, cell + 0.5, sh);
        const c = Math.max(0, Math.min(1, +r.net_displacement_confidence || 0));
        ctx.globalAlpha = 0.15 + 0.85 * c;
        ctx.fillStyle = r.net_displacement === 'MOVE' ? '#22c55e' : '#f59e0b';
        ctx.fillRect(x, sy2, cell + 0.5, sh);
      });
      ctx.globalAlpha = 1;
      let yBottom = sy2 + sh;
      if (getStand) yBottom = drawHuman(ctx, w, top + 86, t, tEnd, true);
      playhead(ctx, x0 + Math.min(1, t / tEnd) * sw, sy1 - 3, yBottom + 3);
    }

    function playhead(ctx, px, y0, y1) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, y0);
      ctx.lineTo(px, y1);
      ctx.stroke();
    }

    function drawHuman(ctx, w, sy3, t, tEnd, noHead) {
      const x0 = 44, sw = w - x0 - 10, sh = 13;
      const st = (getStand && getStand()) || {};
      ctx.font = '11px -apple-system, sans-serif';
      ctx.fillStyle = '#8a94a8';
      ctx.fillText('человек', 2, sy3 + 10);
      const toX = s => x0 + Math.min(1, Math.max(0, s / tEnd)) * sw;
      ctx.fillStyle = '#1a1f2b';
      ctx.fillRect(x0, sy3, sw, sh);
      const rev = +st.reviewed || 0;
      ctx.fillStyle = '#22c55e';
      ctx.fillRect(x0, sy3, toX(rev) - x0, sh);
      ctx.fillStyle = '#f59e0b';
      for (const iv of st.stand || []) ctx.fillRect(toX(iv[0]), sy3, Math.max(2, toX(iv[1]) - toX(iv[0])), sh);
      if (st.open != null) {
        ctx.globalAlpha = 0.6;
        ctx.fillRect(toX(st.open), sy3, Math.max(2, toX(t) - toX(st.open)), sh);
        ctx.globalAlpha = 1;
      }
      if (!noHead) playhead(ctx, toX(t), sy3 - 3, sy3 + sh + 3);
      return sy3 + sh;
    }

    function draw(t) {
      const wrap = canvas.parentElement;
      const w = wrap.clientWidth || 320;
      const h = Math.max(getStand ? 240 : 220, wrap.clientHeight || 230);
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#050608';
      ctx.fillRect(0, 0, w, h);
      const yawH = 112;

      const m = sampleSeries(motion, t);
      let dir = pickDir(m);
      // лёгкий гистерезис, чтобы при дребезге на границе порога не мигало
      if (dir !== shownDir && t >= shownUntil) {
        shownDir = dir;
        shownUntil = t + 0.35;
      } else if (t >= shownUntil) {
        shownDir = dir;
      }
      dir = shownDir;

      const cols = [
        { k: 'LEFT', x: w * 0.2, col: '#3fb950' },
        { k: 'STRAIGHT', x: w * 0.5, col: '#58a6ff' },
        { k: 'RIGHT', x: w * 0.8, col: '#c678dd' },
      ];
      const active = dir === 'LEFT' ? 'LEFT' : dir === 'RIGHT' ? 'RIGHT' : 'STRAIGHT';

      ctx.font = '600 13px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      for (const c of cols) {
        const on = c.k === active;
        ctx.fillStyle = on ? c.col : '#3a4150';
        ctx.globalAlpha = on ? 1 : 0.45;
        ctx.fillText(RU[c.k], c.x, 18);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = on ? c.col : '#252b3a';
        ctx.lineWidth = on ? 3 : 1;
        ctx.beginPath();
        ctx.arc(c.x, yawH * 0.6, on ? 34 : 22, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.save();
      ctx.translate(w * 0.5, yawH * 0.6);
      ctx.scale(0.75, 0.75);
      let rot = 0;
      if (active === 'LEFT') rot = -Math.PI / 2;
      if (active === 'RIGHT') rot = Math.PI / 2;
      ctx.rotate(rot);
      ctx.fillStyle = cols.find(c => c.k === active).col;
      ctx.beginPath();
      ctx.moveTo(0, -36);
      ctx.lineTo(22, 28);
      ctx.lineTo(0, 18);
      ctx.lineTo(-22, 28);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      ctx.strokeStyle = '#252b3a';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, yawH + 2);
      ctx.lineTo(w, yawH + 2);
      ctx.stroke();
      drawGait(ctx, w, yawH + 10, t);

      const yaw = m ? +m.yaw : 0;
      const strength = m && m.strength != null ? +m.strength : 0;
      label.innerHTML = `<b>${RU[active] || RU.SILENT}</b> · t=${t.toFixed(1)} с`
        + (m ? ` · интеграл=${yaw.toFixed(3)}, уверенность=${strength.toFixed(3)}` : '')
        + `<br><span style="color:#8a94a8">${methodNote || 'как в FINAL TRACKER (p12 yaw_at, шаг 0.25 с)'}</span>`;
    }

    function pulse(t, source) {
      draw(t);
      if (source === 'video' && onTime) onTime(t, source);
    }
    video.addEventListener('timeupdate', () => pulse(video.currentTime, 'video'));
    video.addEventListener('seeked', () => pulse(video.currentTime, 'video'));

    return { load, draw: (t) => draw(t) };
  }

  window.FlyMotionPanel = { bind, sampleSeries, RU };
})();
