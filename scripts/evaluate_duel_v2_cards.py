#!/usr/bin/env python3
"""V2 card-effect selfplay coverage CLI. No weight training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.modules.card_game.rl.card_effect_eval import (
    HARD_MAX_GAMES,
    HARD_MAX_RAW_STEPS,
    HARD_MAX_SECONDS,
    LABEL,
    PER_GAME_ACTIONS,
    CardEffectEvaluator,
    atomic_write,
    check_payload,
    find_manifest,
    verify_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='evaluate_duel_v2_cards.py',
        description='Stochastic legal-action selfplay coverage baseline; not RL training.',
    )
    parser.add_argument('--check', action='store_true', help='Print caps/label; no new_game or steps')
    parser.add_argument('--run', action='store_true', help='Foreground supervisor + guarded worker')
    parser.add_argument('--launch', action='store_true', help='Detach supervisor and print pid')
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-raw-steps', type=int, default=HARD_MAX_RAW_STEPS)
    parser.add_argument('--max-games', type=int, default=HARD_MAX_GAMES)
    parser.add_argument('--max-seconds', type=int, default=HARD_MAX_SECONDS)
    parser.add_argument('--per-game-actions', type=int, default=PER_GAME_ACTIONS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--apply-timeout', type=float, default=None)
    return parser


def _require_budget(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.max_raw_steps <= HARD_MAX_RAW_STEPS:
        parser.error(f'--max-raw-steps must be 1..{HARD_MAX_RAW_STEPS}')
    if not 1 <= args.max_games <= HARD_MAX_GAMES:
        parser.error(f'--max-games must be 1..{HARD_MAX_GAMES}')
    if not 1 <= args.max_seconds <= HARD_MAX_SECONDS:
        parser.error(f'--max-seconds must be 1..{HARD_MAX_SECONDS}')
    if not 1 <= args.per_game_actions <= PER_GAME_ACTIONS:
        parser.error(f'--per-game-actions must be 1..{PER_GAME_ACTIONS}')


def _prepare_output(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    if args.output is None:
        parser.error('--output is required')
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f'output dir must be new, not clobber: {output}')
    output.mkdir(parents=True)
    return output


def _maybe_manifest(args: argparse.Namespace) -> None:
    path = args.manifest or find_manifest()
    if path is None:
        return
    verify_manifest(path)


def _flags(args: argparse.Namespace, output: Path) -> list[str]:
    flags = [
        '--output', str(output),
        '--max-raw-steps', str(args.max_raw_steps),
        '--max-games', str(args.max_games),
        '--max-seconds', str(args.max_seconds),
        '--per-game-actions', str(args.per_game_actions),
        '--seed', str(args.seed),
    ]
    if args.manifest:
        flags += ['--manifest', str(args.manifest)]
    if args.apply_timeout is not None:
        flags += ['--apply-timeout', str(args.apply_timeout)]
    return flags


def launch_detached(args: argparse.Namespace, output: Path, *, popen=subprocess.Popen, executable=None) -> int:
    executable = executable or sys.executable
    cmd = [executable, str(Path(__file__).resolve()), '--supervise', *_flags(args, output)]
    kwargs: dict = {
        'stdin': subprocess.DEVNULL, 'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL,
        'close_fds': True,
    }
    if os.name == 'nt':
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        flags |= getattr(subprocess, 'DETACHED_PROCESS', 0)
        flags |= getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        kwargs['creationflags'] = flags
    else:
        kwargs['start_new_session'] = True
    proc = popen(cmd, **kwargs)
    atomic_write(output / 'config.json', {
        'budget': {
            'max_raw_steps': args.max_raw_steps, 'max_games': args.max_games,
            'max_seconds': args.max_seconds, 'per_game_actions': args.per_game_actions,
        },
        'pid': proc.pid, 'supervisor_pid': proc.pid, 'model': 0, 'training_updates': 0, 'label': LABEL,
    })
    (output / 'supervisor.pid').write_text(str(proc.pid), encoding='utf-8')
    print(proc.pid)
    return proc.pid


def write_timeout(output: Path, args: argparse.Namespace, pid: int) -> None:
    payload = {
        'ok': False, 'phase': 'failed', 'pid': pid, 'raw_steps': 0, 'games_started': 0,
        'model': 0, 'training_updates': 0, 'label': LABEL,
        'stopped_reason': 'timeout', 'error': 'supervisor wall-clock timeout',
        'timeouts': 1, 'exceptions': 1,
        'budget': {
            'max_raw_steps': args.max_raw_steps, 'max_games': args.max_games,
            'max_seconds': args.max_seconds, 'per_game_actions': args.per_game_actions,
        },
    }
    summary_path = output / 'summary.json'
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding='utf-8'))
            if existing.get('phase') in ('finished', 'failed') and existing.get('ok') is True:
                return
            if isinstance(existing, dict):
                payload['raw_steps'] = existing.get('raw_steps', 0)
                payload['games_started'] = existing.get('games_started', 0)
        except Exception:
            pass
    atomic_write(output / 'status.json', payload)
    atomic_write(summary_path, payload)


def supervise(args: argparse.Namespace, output: Path, *, popen=subprocess.Popen, executable=None, wait=None) -> int:
    executable = executable or sys.executable
    stdout = (output / 'worker.stdout.log').open('w', encoding='utf-8')
    stderr = (output / 'worker.stderr.log').open('w', encoding='utf-8')
    cmd = [executable, str(Path(__file__).resolve()), '--worker', *_flags(args, output)]
    proc = popen(cmd, stdout=stdout, stderr=stderr)
    atomic_write(output / 'config.json', {
        'budget': {
            'max_raw_steps': args.max_raw_steps, 'max_games': args.max_games,
            'max_seconds': args.max_seconds, 'per_game_actions': args.per_game_actions,
        },
        'pid': proc.pid, 'supervisor_pid': os.getpid(), 'model': 0, 'training_updates': 0, 'label': LABEL,
    })
    (output / 'supervisor.pid').write_text(str(os.getpid()), encoding='utf-8')
    (output / 'worker.pid').write_text(str(proc.pid), encoding='utf-8')
    try:
        if wait:
            wait(proc, args.max_seconds)
        else:
            proc.wait(timeout=args.max_seconds)
        return int(proc.returncode or 0)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        write_timeout(output, args, proc.pid)
        return 1
    finally:
        stdout.close()
        stderr.close()


def run_worker(args: argparse.Namespace, output: Path) -> int:
    result = CardEffectEvaluator(
        output,
        max_raw_steps=args.max_raw_steps,
        max_games=args.max_games,
        max_seconds=args.max_seconds,
        per_game_actions=args.per_game_actions,
        base_seed=args.seed,
        apply_timeout=args.apply_timeout if args.apply_timeout is not None else args.max_seconds,
    ).run()
    return 1 if result.get('error') else 0


def main(argv=None, *, popen=subprocess.Popen, executable=None, wait=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    modes = [args.check, args.run, args.launch, args.supervise, args.worker]
    if sum(bool(flag) for flag in modes) != 1:
        parser.error('exactly one of --check --run --launch is required')
    if args.check:
        print(__import__('json').dumps(check_payload(), ensure_ascii=False, indent=2))
        return 0
    _require_budget(parser, args)
    if args.worker or args.supervise:
        if args.output is None:
            parser.error('--output is required')
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        _maybe_manifest(args)
        if args.worker:
            return run_worker(args, output)
        return supervise(args, output, popen=popen, executable=executable, wait=wait)
    output = _prepare_output(parser, args)
    _maybe_manifest(args)
    if args.launch:
        launch_detached(args, output, popen=popen, executable=executable)
        return 0
    return supervise(args, output, popen=popen, executable=executable, wait=wait)


if __name__ == '__main__':
    raise SystemExit(main())
