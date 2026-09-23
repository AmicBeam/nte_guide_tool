#!/usr/bin/env python3
"""Detach the supervised five-preset training process on Windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'scripts/train_duel_v2_current_round.py'
BREAKAWAY = 0x01000000
DETACHED = 0x00000008
NEW_GROUP = 0x00000200


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--roster-source', type=Path, required=True)
    p.add_argument('--murk-source', type=Path, required=True)
    p.add_argument('--source-rule-hashes', type=Path, required=True)
    p.add_argument('--seed-reservation', type=Path, required=True)
    p.add_argument('--hard-deadline', required=True)
    p.add_argument('--matrix-games', type=int, required=True)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        p.error('Output already exists')
    for source in (a.python, a.roster_source, a.murk_source, a.source_rule_hashes, a.seed_reservation):
        if not source.exists():
            p.error(f'Missing source: {source}')
    deadline = datetime.fromisoformat(a.hard_deadline)
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        p.error('Future timezone-aware deadline required')
    command = [str(a.python), '-X', 'utf8', str(TRAINER), '--output', str(out),
               '--workers', str(a.workers), '--device', 'cuda',
               '--roster-source', str(a.roster_source.resolve()),
               '--murk-source', str(a.murk_source.resolve()),
               '--source-rule-hashes', str(a.source_rule_hashes.resolve()),
               '--seed-reservation', str(a.seed_reservation.resolve()),
               '--hard-deadline', deadline.isoformat(),
               '--matrix-games', str(a.matrix_games), '--run']
    stdout_path = ROOT / 'cross-formal.stdout.log'
    stderr_path = ROOT / 'cross-formal.stderr.log'
    if stdout_path.exists() or stderr_path.exists():
        p.error('Launcher log already exists')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True,
                creationflags=BREAKAWAY | DETACHED | NEW_GROUP if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt')
        except OSError:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True,
                creationflags=DETACHED | NEW_GROUP if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt')
    launch = dict(pid=child.pid, command=command, output=str(out),
        hard_deadline=deadline.isoformat(), launched_at=datetime.now(timezone.utc).isoformat(),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trainer_sha256=hashlib.sha256(TRAINER.read_bytes()).hexdigest(),
        stdout=str(stdout_path), stderr=str(stderr_path))
    (ROOT / 'cross-formal-launch.json').write_text(json.dumps(launch, ensure_ascii=False, indent=2))
    print(json.dumps(dict(pid=child.pid, output=str(out), hard_deadline=deadline.isoformat())), flush=True)


if __name__ == '__main__':
    main()
