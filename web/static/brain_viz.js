/** Fly retina + brain activity — polished live visualization. */

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function viridis(t) {
  t = clamp(t, 0, 1);
  const r = Math.round(255 * clamp(0.15 + t * 0.85, 0, 1) * (t < 0.5 ? t * 2 : 1));
  const g = Math.round(255 * clamp(t * 1.1, 0, 1) * 0.85);
  const b = Math.round(255 * clamp(0.55 - t * 0.5 + (t > 0.6 ? (t - 0.6) * 0.8 : 0), 0, 1));
  return `rgb(${r},${g},${b})`;
}

function plasma(t) {
  t = clamp(t, 0, 1);
  const r = Math.round(255 * clamp(0.05 + t * 1.2, 0, 1));
  const g = Math.round(255 * clamp(t * t * 0.9, 0, 1));
  const b = Math.round(255 * clamp(0.5 + t * 0.5 - t * t * 0.3, 0, 1));
  return `rgb(${r},${g},${b})`;
}

function drawPanelBg(ctx, x, y, w, h, title) {
  ctx.fillStyle = '#111827';
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#64748b';
  ctx.font = '600 9px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText(title.toUpperCase(), x + 10, y + 16);
}

function hexScale(hexX, hexY, w, h, pad) {
  const cx = w / 2;
  const cy = h / 2;
  const maxR = Math.max(
    ...hexX.map(v => Math.abs(v)),
    ...hexY.map(v => Math.abs(v)),
    1e-6,
  );
  const scale = (Math.min(w, h) - 2 * pad) / (2 * maxR);
  return { cx, cy, scale, maxR };
}

function drawHexField(ctx, x, y, w, h, hexX, hexY, values, opts) {
  const pad = opts?.pad ?? 18;
  const colorFn = opts?.colorFn ?? viridis;
  const label = opts?.label ?? '';
  const emptyText = opts?.emptyText ?? 'Ожидание…';

  drawPanelBg(ctx, x, y, w, h, label);

  if (!values?.length || !hexX?.length) {
    ctx.fillStyle = '#475569';
    ctx.font = '12px system-ui';
    ctx.fillText(emptyText, x + 14, y + h / 2);
    return;
  }

  const ix = x + pad;
  const iy = y + pad + 8;
  const iw = w - 2 * pad;
  const ih = h - 2 * pad - 8;
  const { cx, cy, scale } = hexScale(hexX, hexY, iw, ih, 4);

  let vmin = Infinity;
  let vmax = -Infinity;
  for (const v of values) {
    vmin = Math.min(vmin, v);
    vmax = Math.max(vmax, v);
  }
  const rng = vmax - vmin + 1e-6;

  const dotR = clamp(Math.sqrt(iw * ih / values.length) * 0.09, 2.2, 4.5);
  for (let i = 0; i < values.length; i++) {
    const t = (values[i] - vmin) / rng;
    const px = ix + cx + hexX[i] * scale;
    const py = iy + cy - hexY[i] * scale;
    ctx.fillStyle = colorFn(t);
    ctx.beginPath();
    ctx.arc(px, py, dotR, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(ix + cx, iy + cy, scale, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = '#475569';
  ctx.font = '10px system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(`${vmin.toFixed(2)} – ${vmax.toFixed(2)}`, x + w - 10, y + h - 8);
  ctx.textAlign = 'left';
}

function drawMotionBars(ctx, x, y, w, h, t4, t5) {
  drawPanelBg(ctx, x, y, w, h, 'T4 / T5 · cardinal motion');

  const cols = [
    { key: 'a', t4: 'T4a', t5: 'T5a', label: 'back', sub: '180°' },
    { key: 'b', t4: 'T4b', t5: 'T5b', label: 'forward', sub: '0°' },
    { key: 'c', t4: 'T4c', t5: 'T5c', label: 'right', sub: '90°' },
    { key: 'd', t4: 'T4d', t5: 'T5d', label: 'left', sub: '270°' },
  ];

  const top = y + 28;
  const bottom = y + h - 22;
  const midY = (top + bottom) / 2;
  const chartH = (bottom - top) * 0.42;

  const vals = cols.flatMap(c => [t4[c.t4] || 0, t5[c.t5] || 0]);
  const maxAbs = Math.max(0.06, ...vals.map(v => Math.abs(v)));

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + 12, midY);
  ctx.lineTo(x + w - 12, midY);
  ctx.stroke();

  const groupW = (w - 36) / cols.length;
  const barW = groupW * 0.22;

  cols.forEach((c, i) => {
    const gx = x + 18 + i * groupW + groupW / 2;
    const v4 = t4[c.t4] || 0;
    const v5 = t5[c.t5] || 0;

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.strokeRect(gx - barW * 0.6, midY - chartH, barW * 1.8, chartH * 2);

    for (const [v, color, off] of [[v4, '#34d399', -barW * 0.55], [v5, '#fb923c', barW * 0.35]]) {
      const bh = (v / maxAbs) * chartH;
      ctx.fillStyle = color;
      if (Math.abs(bh) < 0.5) continue;
      if (bh >= 0) {
        ctx.fillRect(gx + off, midY - bh, barW, bh);
      } else {
        ctx.fillRect(gx + off, midY, barW, -bh);
      }
    }

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(c.label, gx + barW * 0.1, bottom + 4);
    ctx.fillStyle = '#475569';
    ctx.font = '9px system-ui';
    ctx.fillText(c.sub, gx + barW * 0.1, bottom + 14);
  });

  ctx.textAlign = 'left';
  ctx.fillStyle = '#64748b';
  ctx.font = '9px system-ui';
  ctx.fillText('● T4 ON', x + 12, y + h - 8);
  ctx.fillStyle = '#fb923c';
  ctx.fillText('● T5 OFF', x + 52, y + h - 8);
}

let smoothHeadingDeg = null;
window.resetBrainSmoothing = () => { smoothHeadingDeg = null; };

function drawEpgRing(ctx, x, y, w, h, epg, headingDeg, turning) {
  drawPanelBg(ctx, x, y, w, h, 'EPG · integrated heading');

  if (!epg?.length) return;

  if (headingDeg != null) {
    smoothHeadingDeg = smoothHeadingDeg == null
      ? headingDeg
      : smoothHeadingDeg * 0.88 + headingDeg * 0.12;
  }
  const showHeading = smoothHeadingDeg ?? headingDeg ?? 0;

  const cx = x + w / 2;
  const cy = y + h / 2 + 4;
  const R = Math.min(w, h) * 0.32;
  const n = epg.length;

  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (const r of [R * 0.35, R]) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  for (let i = 0; i < n; i++) {
    const a0 = (i / n) * Math.PI * 2 - Math.PI / 2;
    const a1 = ((i + 0.82) / n) * Math.PI * 2 - Math.PI / 2;
    const v = epg[i];
    const r0 = R * 0.38;
    const r1 = r0 + v * R * 0.95;
    const hue = 270 + v * 50;
    ctx.beginPath();
    ctx.arc(cx, cy, r1, a0, a1);
    ctx.arc(cx, cy, r0, a1, a0, true);
    ctx.closePath();
    ctx.fillStyle = `hsla(${hue}, 75%, 62%, ${0.35 + v * 0.65})`;
    ctx.fill();
  }

  const hd = showHeading * Math.PI / 180;
  ctx.strokeStyle = turning ? '#fbbf24' : '#e879f9';
  ctx.lineWidth = turning ? 3 : 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.cos(hd - Math.PI / 2) * R * 0.92, cy + Math.sin(hd - Math.PI / 2) * R * 0.92);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#e879f9';
  ctx.fill();

  ctx.fillStyle = '#c084fc';
  ctx.font = '600 11px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(`${showHeading.toFixed(0)}°`, cx, cy + R + 18);
  ctx.fillStyle = turning ? '#fbbf24' : '#64748b';
  ctx.font = '9px system-ui';
  ctx.fillText(turning ? 'turning' : 'straight · yaw gated', cx, cy + R + 30);
  ctx.textAlign = 'left';
}

function drawMetricsStrip(ctx, x, y, w, h, frame) {
  drawPanelBg(ctx, x, y, w, h, 'Ego-motion decode');

  const items = [
    { label: 'vx', value: (frame.vx_body || 0).toFixed(3), color: '#34d399' },
    { label: 'vy', value: (frame.vy_body || 0).toFixed(3), color: '#60a5fa' },
    { label: frame.turning ? 'ω turn' : 'ω (0)', value: (frame.yaw_rate || 0).toFixed(3), color: frame.turning ? '#fbbf24' : '#64748b' },
    { label: 'speed', value: (frame.speed || 0).toFixed(3), color: '#fbbf24' },
    { label: 'conf', value: (frame.confidence || 0).toFixed(2), color: '#94a3b8' },
  ];

  const slot = w / items.length;
  items.forEach((it, i) => {
    const sx = x + i * slot + slot / 2;
    ctx.textAlign = 'center';
    ctx.fillStyle = '#64748b';
    ctx.font = '9px system-ui';
    ctx.fillText(it.label, sx, y + 30);
    ctx.fillStyle = it.color;
    ctx.font = '600 13px ui-monospace, monospace';
    ctx.fillText(it.value, sx, y + 48);
  });
  ctx.textAlign = 'left';
}

function drawBarChart(ctx, x, y, w, h, data, title, colorFn) {
  drawPanelBg(ctx, x, y, w, h, title);
  const entries = Object.entries(data || {});
  if (!entries.length) {
    ctx.fillStyle = '#475569';
    ctx.font = '11px system-ui';
    ctx.fillText('нет данных', x + 12, y + h / 2);
    return;
  }
  entries.sort((a, b) => b[1] - a[1]);
  const maxV = Math.max(...entries.map(([, v]) => v), 1);
  const barH = Math.min(18, (h - 40) / entries.length - 4);
  const left = x + 10;
  const chartW = w - 20;
  entries.forEach(([label, val], i) => {
    const by = y + 28 + i * (barH + 4);
    const bw = (val / maxV) * (chartW - 70);
    ctx.fillStyle = '#334155';
    ctx.fillRect(left + 68, by, chartW - 68, barH);
    ctx.fillStyle = colorFn(Math.min(1, val / maxV));
    ctx.fillRect(left + 68, by, bw, barH);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText(label.slice(0, 14), left, by + barH - 2);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(String(Math.round(val)), left + 68 + bw + 4, by + barH - 2);
  });
}

function drawEyeStrip(ctx, x, y, w, h, values, title) {
  drawPanelBg(ctx, x, y, w, h, title);
  if (!values?.length) {
    ctx.fillStyle = '#475569';
    ctx.font = '11px system-ui';
    ctx.fillText('ожидание кадра…', x + 12, y + h / 2);
    return;
  }
  const ix = x + 10;
  const iy = y + 28;
  const iw = w - 20;
  const ih = h - 40;
  const n = values.length;
  const cellW = iw / n;
  let vmin = Infinity, vmax = -Infinity;
  for (const v of values) { vmin = Math.min(vmin, v); vmax = Math.max(vmax, v); }
  const rng = vmax - vmin + 1e-6;
  for (let i = 0; i < n; i++) {
    const t = (values[i] - vmin) / rng;
    ctx.fillStyle = viridis(t);
    ctx.fillRect(ix + i * cellW, iy, Math.max(1, cellW - 0.5), ih);
  }
}

function drawBrainActivity(ctx, w, h, frame, hexX, hexY) {
  ctx.fillStyle = '#0a0c10';
  ctx.fillRect(0, 0, w, h);

  const isMaleCNS = frame?.pathways || frame?.descending;
  const isFlyVis = frame?.t4;

  if (!isMaleCNS && !isFlyVis) {
    ctx.fillStyle = '#475569';
    ctx.font = '13px system-ui';
    ctx.fillText('Ожидание MaleCNS…', 16, h / 2);
    return;
  }

  const pad = 8;
  const gap = 8;
  const metricsH = 58;
  const mainH = h - metricsH - gap - pad * 2;
  const x0 = pad;
  const y0 = pad;

  if (isMaleCNS) {
    const col1w = w * 0.38;
    const col2w = w * 0.32;
    const col3w = w - col1w - col2w - gap * 2 - pad * 2;
    drawBarChart(ctx, x0, y0, col1w, mainH, frame.pathways, 'Pathways · spikes/step', viridis);
    drawBarChart(ctx, x0 + col1w + gap, y0, col2w, mainH, frame.descending, 'Descending · L/R', plasma);
    drawEyeStrip(ctx, x0 + col1w + col2w + gap * 2, y0, col3w, mainH, frame.eye_preview, `Network · ${frame.n_spikes || 0} active`);
  } else {
    const col1w = w * 0.34;
    const col2w = w * 0.26;
    const col3w = w - col1w - col2w - gap * 2 - pad * 2;
    drawMotionBars(ctx, x0, y0, col1w, mainH, frame.t4, frame.t5 || {});
    drawEpgRing(ctx, x0 + col1w + gap, y0, col2w, mainH, frame.epg || [], frame.heading, frame.turning);
    drawHexField(ctx, x0 + col1w + col2w + gap * 2, y0, col3w, mainH, hexX, hexY, frame.flow_mag,
      { colorFn: plasma, label: 'Optic flow · magnitude', emptyText: 'нет flow' });
  }

  drawMetricsStrip(ctx, pad, y0 + mainH + gap, w - pad * 2, metricsH, frame);
}

/** Full-panel retina view (legacy API for index.html). */
function drawHexFieldLegacy(ctx, w, h, hexX, hexY, values, label) {
  ctx.fillStyle = '#0a0c10';
  ctx.fillRect(0, 0, w, h);
  if (values?.length && (!hexX?.length || hexX.length !== values.length)) {
    drawEyeStrip(ctx, 8, 8, w - 16, h - 16, values, label || 'Retina · photoreceptor drive');
    return;
  }
  drawHexField(ctx, 8, 8, w - 16, h - 16, hexX, hexY, values, {
    colorFn: viridis,
    label: label || 'Retina · photoreceptor drive',
    emptyText: 'Ожидание кадра…',
  });
}
