"""Read training artifact dirs for a standalone status page. No Flask, no GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _tail_text(path: Path, max_bytes: int = 65536) -> str:
    if not path.is_file():
        return ''
    with path.open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read().decode('utf-8', errors='replace')
    if size > max_bytes:
        data = data.split('\n', 1)[-1]
    return data


def _tail_jsonl(path: Path, *, max_bytes: int = 262144, limit: int = 120) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _tail_text(path, max_bytes)
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows[-limit:]


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and ((path / 'status.json').is_file() or (path / 'train.log').is_file())


def list_run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    runs = [path for path in root.iterdir() if is_run_dir(path)]
    runs.sort(key=lambda path: (path / 'status.json').stat().st_mtime if (path / 'status.json').is_file() else path.stat().st_mtime, reverse=True)
    return runs


def resolve_run(root: Path, name: str) -> Path | None:
    if not name or name in {'.', '..'} or '/' in name or '\\' in name:
        return None
    path = (root / name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if is_run_dir(path) else None


def _episode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = losses = draws = truncated = 0
    shaping = 0.0
    counted = 0
    for row in rows:
        if row.get('event') not in (None, 'episode') and 'terminal' not in row:
            continue
        counted += 1
        if row.get('truncated'):
            truncated += 1
        terminal = float(row.get('terminal') or 0)
        if terminal > 0:
            wins += 1
        elif terminal < 0:
            losses += 1
        else:
            draws += 1
        shaping += float(row.get('shaping') or 0)
    return {
        'count': counted,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'truncated': truncated,
        'win_rate': round(wins / counted, 4) if counted else 0.0,
        'mean_shaping': round(shaping / counted, 4) if counted else 0.0,
    }


def snapshot_run(path: Path) -> dict[str, Any]:
    status = _read_json(path / 'status.json') or {}
    config = _read_json(path / 'config.json') or {}
    pid = 0
    pid_path = path / 'train.pid'
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding='utf-8').strip() or 0)
        except ValueError:
            pid = 0
    metrics = _tail_jsonl(path / 'metrics.jsonl')
    rollouts = [row for row in metrics if row.get('event') == 'rollout_end']
    episodes = _tail_jsonl(path / 'episodes.jsonl', max_bytes=524288, limit=400)
    log_lines = [line for line in _tail_text(path / 'train.log').splitlines() if line.strip()][-40:]
    elapsed = float(status.get('elapsed_seconds') or 0)
    max_seconds = float(status.get('max_seconds') or config.get('max_seconds') or 0)
    return {
        'name': path.name,
        'path': str(path),
        'alive': pid_alive(pid),
        'pid': pid,
        'phase': status.get('phase') or config.get('budget', {}).get('phase') or 'unknown',
        'ok': status.get('ok', True),
        'error': status.get('error'),
        'elapsed_seconds': elapsed,
        'max_seconds': max_seconds,
        'time_progress': round(min(1.0, elapsed / max_seconds), 4) if max_seconds else 0.0,
        'num_timesteps': int(status.get('num_timesteps') or 0),
        'n_updates': int(status.get('n_updates') or 0),
        'raw_steps': int(status.get('raw_steps') or 0),
        'games_started': int(status.get('games_started') or 0),
        'loss': status.get('loss'),
        'env_steps_per_sec': rollouts[-1].get('env_steps_per_sec') if rollouts else None,
        'raw_steps_per_sec': rollouts[-1].get('raw_steps_per_sec') if rollouts else None,
        'deck_id': config.get('deck_id') or status.get('training_deck_id'),
        'n_envs': config.get('n_envs'),
        'n_steps': config.get('n_steps'),
        'device': next((row.get('device') for row in metrics if row.get('device')), None),
        'episodes': _episode_summary(episodes),
        'loss_series': [float(row['train/loss']) for row in rollouts if 'train/loss' in row][-80:],
        'reward_series': [float(row['terminal']) for row in episodes if 'terminal' in row][-80:],
        'log': log_lines,
    }


def list_run_snapshots(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in list_run_dirs(root):
        status = _read_json(path / 'status.json') or {}
        pid = 0
        pid_path = path / 'train.pid'
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text(encoding='utf-8').strip() or 0)
            except ValueError:
                pid = 0
        rows.append({
            'name': path.name,
            'phase': status.get('phase') or 'unknown',
            'ok': status.get('ok', True),
            'alive': pid_alive(pid),
            'n_updates': int(status.get('n_updates') or 0),
            'elapsed_seconds': float(status.get('elapsed_seconds') or 0),
            'loss': status.get('loss'),
        })
    return rows
