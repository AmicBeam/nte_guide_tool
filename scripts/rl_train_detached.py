#!/usr/bin/env python3
"""Start train_duel_v2.py outside the parent Job Object so a 1-hour run is not killed."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'scripts' / 'train_duel_v2.py'
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Detach a V2 RL training process on Windows.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('train_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    extra = list(args.train_args)
    if extra and extra[0] == '--':
        extra = extra[1:]
    if '--train' not in extra:
        extra = ['--train', *extra]
    command = [args.python, '-X', 'utf8', str(TRAINER), *extra]
    flags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    if os.name == 'nt':
        flags |= CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS
    try:
        proc = subprocess.Popen(
            command, cwd=str(ROOT), close_fds=True,
            creationflags=flags if os.name == 'nt' else 0,
        )
    except OSError:
        proc = subprocess.Popen(
            command, cwd=str(ROOT), close_fds=True,
            creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW if os.name == 'nt' else 0,
            start_new_session=True,
        )
    print(f'detached pid={proc.pid} cmd={" ".join(command)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
