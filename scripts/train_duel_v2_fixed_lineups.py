#!/usr/bin/env python3
"""Freeze and launch five fixed-build policies with a hard wall-clock deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'


def freeze(out, source):
    """Copy only code, the catalog, referenced artwork and the three numeric sources."""
    files = list((ROOT / 'app').rglob('*.py'))
    files += [ROOT / 'app/modules/card_game/content/duel_v2/catalog.json', Path(__file__),
              ROOT / 'app/modules/card_game/rl/experiment_dispositions.json']
    catalog = json.loads(files[-3].read_text(encoding='utf-8'))
    for character in catalog['characters']:
        for key in ('avatar', 'portrait'):
            value = character.get(key, '')
            if value.startswith('/static/'):
                p = ROOT / 'app' / value.lstrip('/')
                if p.is_file(): files.append(p)
    manifest = {}
    for p in sorted(set(files)):
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT): raise ValueError('Out-of-scope source')
        target = out / 'frozen' / p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target); manifest[target.relative_to(out).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    for key in ('starter', 'weave-rush', 'quick-rush'):
        for ext in ('npz', 'json'):
            p = source / f'{key}.{ext}'
            if p.is_symlink(): raise ValueError('Symlink model')
            target = out / 'source' / p.name; target.parent.mkdir(exist_ok=True)
            shutil.copy2(p, target); manifest[target.relative_to(out).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'app/modules/card_game/engine/ai/models')
    parser.add_argument('--learner-build', type=Path, help='Explicit fixed Zhenhong build JSON')
    parser.add_argument('--learner-source', type=Path, help='Prior fixed_ten_v1 Zhenhong numeric directory')
    parser.add_argument('--zhenhong-passive-reward', type=float, default=0.2, help='Extra reward per actual Zhenhong passive trigger')
    parser.add_argument('--seconds', type=int, default=5400)
    parser.add_argument('--hard-deadline', type=float, help='Preserve an already authorized absolute cutoff')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed-ledger', type=Path, default=ROOT / 'artifacts/rl-seed-ledger.json')
    parser.add_argument('--full-cycle', action='store_true', help='Five-team build search and paired acceptance workflow')
    parser.add_argument('--warm-models', type=Path, help='Five same-contract research model pairs')
    parser.add_argument('--confirmation-seeds', type=int, default=512)
    parser.add_argument('--adaptation-steps', type=int, default=32)
    parser.add_argument('--search-candidates', type=int, default=16)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); out = args.output.resolve()
    from app.modules.card_game.rl.fixed_lineup import write, builds, identity
    if args.supervise or args.worker:
        config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
        if args.worker:
            try:
                for rel, digest in json.loads((out / 'source-manifest.json').read_text()).items():
                    p = (out / rel).resolve()
                    if not p.is_relative_to(out) or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                        raise ValueError('Frozen source integrity failure')
                if identity() != config['rule_hash']: raise ValueError('Rule identity mismatch')
                if config.get('full_cycle'):
                    from app.modules.card_game.rl.five_full_cycle import run
                else:
                    from app.modules.card_game.rl.fixed_lineup_training import run
                run(out, config)
            except Exception as exc:
                write(out / 'status.json', dict(phase='failed', error=repr(exc), finished=time.time()))
                raise
            return
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--output', str(out), '--worker'],
                                cwd=ROOT, start_new_session=os.name != 'nt')
        guard = dict(phase='running', worker_pid=proc.pid, hard_deadline=config['hard_deadline'])
        write(out / 'guard.json', guard)
        end = time.monotonic() + max(0, config['hard_deadline'] - time.time())
        try:
            while proc.poll() is None:
                remaining = min(end - time.monotonic(), config['hard_deadline'] - time.time())
                if remaining <= 0:
                    if os.name == 'nt': subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], check=False)
                    else: os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15); guard['phase'] = 'hard_deadline'; break
                try: proc.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired: pass
            if guard['phase'] == 'running': guard['phase'] = 'finished' if proc.returncode == 0 else 'failed'
            guard.update(exit_code=proc.returncode, finished=time.time()); write(out / 'guard.json', guard)
        finally:
            if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return
    if not 1 <= args.workers <= 8 or not (60 if args.smoke else 600) <= args.seconds <= 21600:
        parser.error('Invalid bounded budget/workers')
    import math
    if not math.isfinite(args.zhenhong_passive_reward) or args.zhenhong_passive_reward < 0:
        parser.error('Passive reward must be finite and nonnegative')
    if not 32 <= args.confirmation_seeds <= 4096 or not 1 <= args.adaptation_steps <= 128 or not 2 <= args.search_candidates <= 32:
        parser.error('Invalid full-cycle workload')
    if args.full_cycle and not args.warm_models:
        parser.error('Full five-team cycle requires explicit warm models')
    from app.modules.card_game.rl.build_acceptance import check_source, claim_seeds
    check_source(args.source)
    if bool(args.learner_build) != bool(args.learner_source):
        parser.error('Provide both learner build and learner source')
    custom = None
    if args.learner_build:
        from app.modules.card_game.content.duel_v2 import validate_deck
        custom = validate_deck(json.loads(args.learner_build.read_text(encoding='utf-8')))
        if custom['character_ids'] != ['zhenhong', 'zero', 'iloy', 'yi']:
            raise ValueError('Expected the explicit four-person Zhenhong lineup')
        custom['id'] = 'zhenhong'
    config = dict(kind='five_fixed_lineups_v1', rule_hash=identity(), builds=builds(args.source), device=args.device,
                  workers=args.workers, smoke=args.smoke, evaluation_pairs=1 if args.smoke else 32,
                  algorithm='outcome_ppo_with_zhenhong_passive', zhenhong_passive_reward=args.zhenhong_passive_reward, seconds=args.seconds, automatic_serving_approval=False)
    config.update(full_cycle=args.full_cycle, confirmation_seeds=32 if args.smoke else args.confirmation_seeds,
                  adaptation_steps=args.adaptation_steps, search_candidates=args.search_candidates)
    if custom:
        if not args.full_cycle: config['learner_keys'] = ['zhenhong']
        config['builds']['zhenhong'] = custom
    if not args.run:
        print(json.dumps(dict(training_started=False, **config), ensure_ascii=False)); return
    if out.exists(): raise ValueError('Output must be new; no duplicate launch')
    out.mkdir(parents=True)
    manifest = freeze(out, args.source.resolve())
    if custom:
        from app.modules.card_game.rl.build_acceptance import check_source
        check_source(args.learner_source)
        (out / 'learner-source').mkdir()
        for ext in ('npz', 'json'):
            p = args.learner_source / f'zhenhong.{ext}'
            if p.is_symlink(): raise ValueError('Symlink learner source')
            dest = out / 'learner-source' / p.name; shutil.copy2(p, dest)
            manifest[dest.relative_to(out).as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
    if args.warm_models:
        check_source(args.warm_models)
        (out / 'warm-models').mkdir()
        for key in ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'midrange'):
            for ext in ('npz', 'json'):
                p = args.warm_models / f'{key}.{ext}'
                if p.is_symlink(): raise ValueError('Symlink warm model')
                dest = out / 'warm-models' / p.name; shutil.copy2(p, dest)
                manifest[dest.relative_to(out).as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
    write(out / 'source-manifest.json', manifest)
    from app.modules.card_game.rl.fixed_lineup_training import prepare
    config['builds'] = prepare(out / 'source', out)
    if args.warm_models:
        from app.modules.card_game.rl.fixed_lineup_training import migrate_fixed
        for key in ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'midrange'):
            metadata = json.loads((out / 'warm-models' / f'{key}.json').read_text(encoding='utf-8'))
            build = custom if key == 'zhenhong' and custom else metadata['build']
            migrate_fixed(out / 'warm-models', key, out / 'initial', build)
            config['builds'][key] = build
    if custom:
        from app.modules.card_game.rl.fixed_lineup_training import migrate_fixed
        migrate_fixed(out / 'learner-source', 'zhenhong', out / 'initial', custom)
        config['builds']['zhenhong'] = custom
    config['seeds'] = claim_seeds(args.seed_ledger, str(out))
    now = time.time()
    seconds = min(args.seconds, args.hard_deadline - now) if args.hard_deadline else args.seconds
    if seconds < 60: raise ValueError('Insufficient time before original hard deadline')
    config.update(created=now, seconds=seconds, train_until=now + seconds * 5 / 6,
                  evaluate_until=now + seconds - min(60, seconds / 20), hard_deadline=now + seconds)
    if args.full_cycle:
        from app.modules.card_game.rl.build_acceptance import phase_deadlines
        config['deadlines'] = phase_deadlines(now, seconds)
        # Five lineups need more search time; paired adaptation remains equal-step.
        config['deadlines'].update(search=now+seconds*.55,
            baseline_adaptation=now+seconds*.625, candidate_adaptation=now+seconds*.70)
        config['kind'] = 'five_full_cycle_v1'
        config['train_until'] = config['deadlines']['candidate_adaptation']
    write(out / 'config.json', config)
    frozen_script = out / 'frozen/scripts' / Path(__file__).name
    opts = dict(cwd=out / 'frozen', stdin=subprocess.DEVNULL, close_fds=True)
    if os.name == 'nt': opts['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | 0x01000000
    else: opts['start_new_session'] = True
    with (out / 'console.log').open('xb') as log:
        proc = subprocess.Popen([sys.executable, '-X', 'utf8', str(frozen_script), '--output', str(out), '--supervise'],
                                stdout=log, stderr=subprocess.STDOUT, **opts)
    for _ in range(100):
        if (out / 'guard.json').exists(): break
        if proc.poll() is not None: raise RuntimeError('Supervisor exited before acknowledgement')
        time.sleep(.1)
    if not (out / 'guard.json').exists(): raise RuntimeError('Supervisor did not acknowledge startup')
    receipt = dict(supervisor_pid=proc.pid, output=str(out), started=now, train_until=config['train_until'], hard_deadline=config['hard_deadline'])
    write(out / 'launch.json', receipt); print(json.dumps(receipt))


if __name__ == '__main__':
    main()
