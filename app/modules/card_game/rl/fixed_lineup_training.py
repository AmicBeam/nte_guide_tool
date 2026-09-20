"""Bounded fixed-opponent PPO using complete official-engine games only."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .fixed_lineup import (KEYS, FOES, FixedModel, all_features, CAND_DIM, builds,
                           encode, expand_source, identity, save_weights, write)


@lru_cache(maxsize=16)
def model(directory, key):
    return FixedModel(directory, key)


def play(job):
    from ..engine.duel_v2 import new_game, observe, apply_action, acting_side
    state = new_game(seed=job['seed'], first_side=job['first'], decks=job['decks'])
    policies = {s: model(*spec) for s, spec in job['policies'].items()}
    rng = np.random.default_rng(job['seed']); trajectory = []; action_counts = Counter()
    for step in range(600):
        if state['phase'] == 'finished': break
        if time.time() >= job['deadline']: return dict(complete=False, reason='deadline', trajectory=[])
        side = acting_side(state); view = observe(state, side, include_previews=False)
        actions = [e['action'] for e in view['legal_actions'] if e['action']['type'] != 'concede']
        x, c = encode(view, actions); scores, value = policies[side].scores_value(x, c)
        prob = np.exp(scores.astype(np.float64) - scores.max()); prob /= prob.sum()
        index = int(rng.choice(len(actions), p=prob)) if job.get('train') and side == 'a' else int(scores.argmax())
        if job.get('train') and side == 'a':
            trajectory.append((x, c, index, float(np.log(max(prob[index], 1e-20))), value))
        if side == 'a': action_counts[actions[index]['type']] += 1
        state = apply_action(state, side, actions[index]); state['events'] = []; state.pop('_public_board', None)
    done = state['phase'] == 'finished'
    return dict(complete=done, reason=None if done else 'action_limit', winner=state.get('winner'),
                reward=(10 if state.get('winner') == 'a' else -10 if state.get('winner') == 'b' else 0) if done else None,
                trajectory=trajectory if done else [], steps=step + 1, turn=state['turn'], action_counts=dict(action_counts))


def append(root, name, row):
    with (root / name).open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')


def prepare(source, root):
    """Explicit offline rule migration and zero-column expansion, no approval inherited."""
    target_builds = builds(source); expanded = {}
    for key in FOES:
        weights, manifest = expand_source(source, key); expanded[key] = (weights, manifest)
        origin = dict(kind='frozen_old_numeric_expanded', source_sha256=manifest['sha256'],
                      source_rule_hash=manifest['rule_hash'], original_weights_preserved=True,
                      new_columns_zero=True, approval_inherited=False)
        save_weights(weights, root / 'reference', key, manifest['build'], origin)
    sources = dict(starter='starter', **{'weave-rush': 'weave-rush', 'quick-rush': 'quick-rush'},
                   zhenhong='starter', midrange='weave-rush')
    for key, source_key in sources.items():
        weights, manifest = expanded[source_key]
        save_weights(weights, root / 'initial', key, target_builds[key], dict(
            kind='fixed_build_warm_start', source_key=source_key, source_sha256=manifest['sha256'],
            source_rule_hash=manifest['rule_hash'], new_columns_zero=True, approval_inherited=False))
    return target_builds


def job_for(key, foe, seed, first, directory, root, deadline, *, train=False):
    learner = FixedModel(directory, key); opponent = FixedModel(root / 'reference', foe)
    return dict(key=key, foe=foe, seed=seed, first=first, deadline=deadline, train=train,
                decks={'a': learner.serving_deck, 'b': opponent.serving_deck},
                policies={'a': (str(directory), key), 'b': (str(root / 'reference'), foe)})


def evaluate(pool, root, directory, seed, deadline, pairs):
    # Complete every team/opponent/seat cell for a seed before the next seed.
    rows = []
    for i in range(pairs):
        if time.time() >= deadline: break
        jobs = [job_for(key, foe, seed + i, first, directory, root, deadline)
                for key in KEYS for foe in FOES for first in ('a', 'b')]
        for job, result in zip(jobs, pool.map(play, jobs)):
            row = {k: job[k] for k in ('key', 'foe', 'seed', 'first')}
            row.update({k: v for k, v in result.items() if k != 'trajectory'})
            row['variant'] = Path(directory).name; rows.append(row)
            append(root, 'evaluation-games.jsonl', row)
        if not all(r['complete'] for r in rows[-len(jobs):]): break
    return rows


def summary(rows, pairs):
    groups = {}
    for key in KEYS:
        cells = {}
        for foe in FOES:
            for first in ('a', 'b'):
                group = [r for r in rows if r['key'] == key and r['foe'] == foe and r['first'] == first and r['complete']]
                cells[f'{foe}/{first}'] = dict(n=len(group), wins=sum(r['winner'] == 'a' for r in group),
                                               draws=sum(r['winner'] not in ('a', 'b') for r in group))
        complete = all(c['n'] == pairs for c in cells.values())
        groups[key] = dict(complete=complete, cells=cells,
                          equal_cell_win_rate=float(np.mean([c['wins'] / c['n'] for c in cells.values()])) if all(c['n'] for c in cells.values()) else None)
    return groups


def run(root, config):
    import torch
    from .gpu_duel.ppo import CompactScorer
    from .league_learning import ordinary_update
    torch.set_num_threads(1); torch.manual_seed(config['seeds']['foundation'])
    root = Path(root); nets = {}; opts = {}; origins = {}; target_builds = config['builds']
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'reference').iterdir()}
    for key in KEYS:
        policy = FixedModel(root / 'initial', key)
        net = CompactScorer(len(all_features()), CAND_DIM, policy.hidden).to(config['device'])
        net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in policy.weights.items()})
        nets[key] = net; opts[key] = torch.optim.Adam(net.parameters(), lr=3e-5); origins[key] = policy.manifest['origin']
    totals = {key: dict(games=0, samples=0, optimizer_steps=0) for key in KEYS}
    current = root / 'initial'; cycle = 0
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool:
        while time.time() < config['train_until'] and not (config.get('smoke') and cycle >= 1):
            write(root / 'status.json', dict(phase='training', cycle=cycle, totals=totals, train_until=config['train_until'], hard_deadline=config['hard_deadline']))
            # The five learners receive exactly the same six opponent/seat cells.
            jobs = [job_for(key, foe, config['seeds']['foundation'] + cycle * 100 + fi, first,
                            current, root, config['train_until'], train=True)
                    for key in KEYS for fi, foe in enumerate(FOES) for first in ('a', 'b')]
            results = list(pool.map(play, jobs)); batches = {key: [] for key in KEYS}
            for job, result in zip(jobs, results):
                append(root, 'training-games.jsonl', dict(cycle=cycle, **{k: job[k] for k in ('key', 'foe', 'seed', 'first')},
                       **{k: v for k, v in result.items() if k != 'trajectory'}))
                if result['complete']: batches[job['key']].append(result)
            # Partial batches are recorded but never update only some teams.
            if not all(r['complete'] for r in results) or time.time() >= config['train_until']: break
            for key in KEYS:
                update = ordinary_update(nets[key], opts[key], batches[key], config['seeds']['foundation'] + cycle)
                if update.get('updates') != 1: raise ValueError('Missing terminal trajectory update')
                totals[key]['games'] += len(batches[key]); totals[key]['samples'] += update['samples']; totals[key]['optimizer_steps'] += 1
                append(root, 'losses.jsonl', dict(cycle=cycle, key=key, **update))
            cycle += 1; current = root / 'generations' / f'cycle-{cycle:05}'
            for key in KEYS:
                save_weights({k: v.detach().cpu().numpy() for k, v in nets[key].state_dict().items()},
                             current, key, target_builds[key], origins[key])
            write(root / 'checkpoint.json', dict(cycle=cycle, current=str(current), totals=totals))
        latest = root / 'latest'
        import shutil
        latest.mkdir(exist_ok=True)
        for p in current.iterdir(): shutil.copy2(p, latest / p.name)
        training = dict(cycles=cycle, totals=totals, finished=time.time(), rule_hash=identity(),
                        algorithm='terminal_outcome_ppo', fixed_builds=True, frozen_opponents=True)
        write(root / 'training-summary.json', training)
        if cycle == 0: raise RuntimeError('No complete training cycle')
        write(root / 'status.json', dict(phase='evaluating', training=training, hard_deadline=config['hard_deadline']))
        pairs = config['evaluation_pairs']; all_rows = {}
        # Interleaving variants prevents a slow baseline from consuming all candidate time.
        for i in range(pairs):
            if time.time() >= config['evaluate_until']: break
            for variant in ('initial', 'latest'):
                rows = evaluate(pool, root, root / variant, config['seeds']['audit'] + i, config['evaluate_until'], 1)
                all_rows.setdefault(variant, []).extend(rows)
        if frozen != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'reference').iterdir()}:
            raise ValueError('Frozen opponent identity changed')
        reports = {v: summary(all_rows.get(v, []), pairs) for v in ('initial', 'latest')}
        complete = all(g['complete'] for groups in reports.values() for g in groups.values())
        write(root / 'summary.json', dict(training=training, evaluation=reports, complete=complete,
              evaluation_pairs=pairs, automatic_serving_approval=False, quality_approved=False,
              note='Fixed endpoint descriptive short-run evaluation; no automatic promotion or balance claim.'))
        write(root / 'status.json', dict(phase='finished', complete=complete, finished=time.time(), totals=totals))
