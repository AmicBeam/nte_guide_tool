#!/usr/bin/env python3
"""Standalone RL training dashboard. Does not attach to the card-game Flask app."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.status_board import list_run_snapshots, resolve_run, snapshot_run

PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL 训练监控</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font: 15px/1.45 system-ui, sans-serif; background: #10141c; color: #e8edf5; }
    main { max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    .muted { color: #9aa6b8; }
    header { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; justify-content: space-between; margin-bottom: 20px; }
    select { background: #1b2230; color: #e8edf5; border: 1px solid #334056; border-radius: 8px; padding: 8px 10px; min-width: 280px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .card { background: #18202e; border: 1px solid #2a3548; border-radius: 12px; padding: 12px 14px; }
    .label { font-size: 12px; color: #9aa6b8; }
    .value { font-size: 22px; font-variant-numeric: tabular-nums; margin-top: 4px; }
    .bar { height: 8px; background: #2a3548; border-radius: 99px; overflow: hidden; margin: 8px 0 18px; }
    .bar > span { display: block; height: 100%; background: #6ea8ff; }
    canvas { width: 100%; height: 120px; background: #121924; border-radius: 10px; }
    pre { background: #121924; border-radius: 10px; padding: 12px; overflow: auto; max-height: 280px; font-size: 12px; }
    .ok { color: #7ddea0; }
    .bad { color: #ff8b8b; }
    .warn { color: #ffd37a; }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>RL 训练监控</h1>
      <p class="muted">独立页面，不挂在大厅。每 2 秒刷新 artifacts 目录。</p>
    </div>
    <label>运行
      <select id="run"></select>
    </label>
  </header>
  <p id="meta" class="muted">读取中…</p>
  <div class="bar"><span id="timebar" style="width:0%"></span></div>
  <div class="grid" id="cards"></div>
  <div class="grid">
    <div class="card"><div class="label">loss</div><canvas id="loss" width="480" height="120"></canvas></div>
    <div class="card"><div class="label">终局奖励</div><canvas id="reward" width="480" height="120"></canvas></div>
  </div>
  <div class="card"><div class="label">train.log</div><pre id="log"></pre></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
function fmt(n, d=1) {
  if (n == null || n === '') return '—';
  const x = Number(n);
  if (!Number.isFinite(x)) return String(n);
  return x.toFixed(d).replace(/\.0+$/, '');
}
function spark(canvas, values, color) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!values.length) return;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = values.length === 1 ? w / 2 : i * (w - 8) / (values.length - 1) + 4;
    const y = h - 6 - ((v - min) / span) * (h - 12);
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
}
async function loadRuns(selected) {
  const rows = await (await fetch('/api/runs')).json();
  const sel = $('run');
  const names = rows.runs.map(r => r.name);
  if (JSON.stringify(names) !== JSON.stringify([...sel.options].map(o => o.value))) {
    sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join('') || '<option value="">暂无训练目录</option>';
    if (selected && names.includes(selected)) sel.value = selected;
  }
  return rows.runs;
}
function cards(run) {
  const live = run.alive ? '<span class="ok">运行中</span>' : '<span class="bad">已结束</span>';
  const ok = run.ok === false ? '<span class="bad">失败</span>' : live;
  const ep = run.episodes || {};
  const items = [
    ['状态', ok + ' · ' + (run.phase || '—')],
    ['更新', run.n_updates],
    ['环境步', run.num_timesteps],
    ['引擎步', run.raw_steps],
    ['对局', run.games_started],
    ['loss', fmt(run.loss, 3)],
    ['环境步/秒', fmt(run.env_steps_per_sec, 1)],
    ['已用时间', fmt(run.elapsed_seconds, 0) + ' / ' + fmt(run.max_seconds, 0) + 's'],
    ['胜/负/截断', (ep.wins||0) + '/' + (ep.losses||0) + '/' + (ep.truncated||0)],
    ['胜率', fmt((ep.win_rate||0)*100, 1) + '%'],
  ];
  $('cards').innerHTML = items.map(([k,v]) => `<div class="card"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
}
async function tick() {
  const runs = await loadRuns($('run').value);
  const name = $('run').value;
  if (!name) { $('meta').textContent = 'artifacts 下没有 status.json / train.log 目录。'; return; }
  const run = await (await fetch('/api/run?name=' + encodeURIComponent(name))).json();
  if (run.error) { $('meta').textContent = run.error; return; }
  const live = run.alive ? '进程 ' + run.pid : '进程未在运行';
  $('meta').textContent = [run.deck_id || '未记录牌组', run.device || '', (run.n_envs||'?') + ' env × ' + (run.n_steps||'?') + ' steps', live].filter(Boolean).join(' · ');
  $('timebar').style.width = ((run.time_progress || 0) * 100) + '%';
  cards(run);
  spark($('loss'), run.loss_series || [], '#6ea8ff');
  spark($('reward'), run.reward_series || [], '#7ddea0');
  $('log').textContent = (run.log || []).join('\n') || '暂无日志';
}
$('run').addEventListener('change', tick);
setInterval(tick, 2000);
tick();
</script>
</body>
</html>
'''


class Handler(BaseHTTPRequestHandler):
    root: Path

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self._send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            return
        if parsed.path == '/api/runs':
            payload = {'root': str(self.root), 'runs': list_run_snapshots(self.root)}
            self._send(200, json.dumps(payload, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')
            return
        if parsed.path == '/api/run':
            name = (parse_qs(parsed.query).get('name') or [''])[0]
            path = resolve_run(self.root, name)
            if path is None:
                self._send(404, json.dumps({'error': '未找到该训练目录'}).encode('utf-8'), 'application/json; charset=utf-8')
                return
            self._send(200, json.dumps(snapshot_run(path), ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')
            return
        self._send(404, b'not found', 'text/plain; charset=utf-8')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Standalone RL training dashboard (not the card-game site).')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--root', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args(argv)
    Handler.root = args.root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'rl-monitor http://127.0.0.1:{args.port}/  (LAN {args.host}:{args.port}) root={Handler.root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
