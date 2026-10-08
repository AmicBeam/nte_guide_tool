#!/usr/bin/env python3
"""Time one formal Gumbel batch and a few raw games. Does not train or reserve seeds."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from app.modules.card_game.rl.fixed_lineup import identity
    from app.modules.card_game.rl.information_search import search_game
    from app.modules.card_game.rl.ten_search_runtime import migrate_source
    source = Path(sys.argv[1])
    builds = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    out = Path(sys.argv[3])
    workers = int(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    prepared = out / 'prepared'
    if not (prepared / 'zhenhong.npz').exists():
        migrate_source(source, prepared, builds)
    deadline = time.time() + 1800
    jobs = []
    keys = list(builds)
    for index, key in enumerate(keys):
        foe = keys[(index + 1) % len(keys)]
        jobs.append(dict(
            seed=910000 + index, first='a' if index % 2 == 0 else 'b', deadline=deadline,
            decks={'a': builds[key], 'b': builds[foe]},
            policies={'a': (str(prepared), key), 'b': (str(prepared), foe)},
            runtime='ten', simulations=32, search_algorithm='gumbel', gumbel_candidates=16,
            training=True, collect=True, train_sides=['a', 'b'],
        ))
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        results = list(pool.map(search_game, jobs))
    elapsed = time.time() - started
    raw_jobs = [dict(job, raw_sides=['a', 'b'], training=False, collect=False, seed=job['seed'] + 50) for job in jobs[:4]]
    raw_started = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        raw = list(pool.map(search_game, raw_jobs))
    raw_elapsed = time.time() - raw_started
    report = dict(
        rule_hash=identity(), search_algorithm='gumbel', gumbel_candidates=16, simulations=32,
        workers=workers, complete=all(row['complete'] for row in results),
        measured_at=time.time(), search_games=len(results), search_seconds=elapsed,
        search_decisions=sum(row.get('decisions', 0) for row in results),
        raw_games=len(raw), raw_seconds=raw_elapsed,
        raw_complete=all(row['complete'] for row in raw),
    )
    (out / 'batch-timing.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
