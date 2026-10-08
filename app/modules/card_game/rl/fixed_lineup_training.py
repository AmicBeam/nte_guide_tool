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


def passive_count(state, side):
    return int(state['sides'][side]['characters'].get('zhenhong', {}).get('surplus_passive_triggers', 0))


def passive_returns(outcome, total, before_decisions, coefficient):
    if not np.isfinite(coefficient) or coefficient < 0 or any(n > total or n < 0 for n in before_decisions):
        raise ValueError('Invalid passive reward trajectory')
    return [outcome + coefficient * (total - before) for before in before_decisions]


def track_immune_window(before, after, action, actor, metrics, risks):
    """Public consequences only; diagnostics never change actions or rewards."""
    from ..engine.duel_v2.state import damage_immune
    team = before['sides']['a']; next_team = after['sides']['a']
    if 'zhenhong' not in team['characters'] or not damage_immune(before, 'a', 'zhenhong'):
        return
    metrics['decisions_in_immune_window'] += int(actor == 'a')
    old_front, front = team.get('front'), next_team.get('front')
    labels = []
    down = [cid for cid, h in team['characters'].items() if cid != 'zhenhong' and h['hp'] > 0
            and next_team['characters'][cid]['hp'] <= 0]
    if down:
        metrics['allies_down_while_immune'] += len(down); labels.append('allies_down_while_immune')
    loss = max(0, team['hp'] - next_team['hp'])
    if labels and len(risks) < 16:
        risks.append(dict(turn=before['turn'], actor=actor, action=action, labels=labels,
                          before_front=old_front, after_front=front, down=down, player_hp_loss=loss))


def play(job):
    if time.time() >= job['deadline']:
        return dict(complete=False, reason='deadline', trajectory=[])
    from ..engine.duel_v2 import new_game, observe, apply_action, acting_side
    state = new_game(seed=job['seed'], first_side=job['first'], decks=job['decks'])
    policies = {s: model(*spec) for s, spec in job['policies'].items()}
    rng = np.random.default_rng(job['seed']); trajectory = []; action_counts = Counter()
    immune_metrics = Counter(); immune_risks = []
    passive_at_decision = []
    bonus = float(job.get('zhenhong_passive_reward', 0.2)) if job.get('train') and job.get('key') == 'zhenhong' else 0.0
    if not np.isfinite(bonus) or bonus < 0: raise ValueError('Invalid passive reward')
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
            passive_at_decision.append(passive_count(state, 'a'))
        if side == 'a': action_counts[actions[index]['type']] += 1
        after = apply_action(state, side, actions[index])
        track_immune_window(state, after, actions[index], side, immune_metrics, immune_risks)
        state = after; state['events'] = []; state.pop('_public_board', None)
    done = state['phase'] == 'finished'
    count = passive_count(state, 'a')
    outcome = 10 if state.get('winner') == 'a' else -10 if state.get('winner') == 'b' else 0
    reward = outcome + bonus * count
    returns = passive_returns(outcome, count, passive_at_decision, bonus) if done else []
    return dict(complete=done, reason=None if done else 'action_limit', winner=state.get('winner'),
                reward=reward if done else None, outcome_reward=outcome if done else None,
                passive_triggers=count, passive_reward=bonus * count if done else None, trajectory_returns=returns,
                trajectory=trajectory if done else [], steps=step + 1, turn=state['turn'], action_counts=dict(action_counts),
                immune_window=dict(immune_metrics), immune_risks=immune_risks)


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


def evaluate(pool, root, directory, seed, deadline, pairs, keys=KEYS):
    # Complete every team/opponent/seat cell for a seed before the next seed.
    rows = []
    for i in range(pairs):
        if time.time() >= deadline: break
        jobs = [job_for(key, foe, seed + i, first, directory, root, deadline)
                for key in keys for foe in FOES for first in ('a', 'b')]
        for job, result in zip(jobs, pool.map(play, jobs)):
            row = {k: job[k] for k in ('key', 'foe', 'seed', 'first')}
            row.update({k: v for k, v in result.items() if k not in ('trajectory', 'trajectory_returns')})
            row['variant'] = Path(directory).name; rows.append(row)
            append(root, 'evaluation-games.jsonl', row)
        if not all(r['complete'] for r in rows[-len(jobs):]): break
    return rows


def summary(rows, pairs, keys=KEYS):
    groups = {}
    for key in keys:
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
    keys = tuple(config.get('learner_keys', KEYS))
    if not keys or not set(keys) <= set(KEYS) or len(keys) != len(set(keys)):
        raise ValueError('Invalid learner keys')
    torch.set_num_threads(1); torch.manual_seed(config['seeds']['foundation'])
    root = Path(root); nets = {}; opts = {}; origins = {}; target_builds = config['builds']
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'reference').iterdir()}
    for key in keys:
        policy = FixedModel(root / 'initial', key)
        net = CompactScorer(len(all_features()), CAND_DIM, policy.hidden).to(config['device'])
        net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in policy.weights.items()})
        nets[key] = net; opts[key] = torch.optim.Adam(net.parameters(), lr=3e-5)
        origins[key] = dict(policy.manifest.get('origin') or {}, training_reward=dict(
            outcome_win=10, outcome_loss=-10, passive_per_trigger=config.get('zhenhong_passive_reward', 0.2) if key == 'zhenhong' else 0,
            scope='learner_only', returns='future_triggers_only'))
    totals = {key: dict(games=0, samples=0, optimizer_steps=0) for key in keys}
    current = root / 'initial'; cycle = 0
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=get_context('spawn')) as pool:
        while time.time() < config['train_until'] and not (config.get('smoke') and cycle >= 1):
            write(root / 'status.json', dict(phase='training', cycle=cycle, totals=totals, train_until=config['train_until'], hard_deadline=config['hard_deadline']))
            # The five learners receive exactly the same six opponent/seat cells.
            jobs = [job_for(key, foe, config['seeds']['foundation'] + cycle * 100 + fi, first,
                            current, root, config['train_until'], train=True)
                    for key in keys for fi, foe in enumerate(FOES) for first in ('a', 'b')]
            for job in jobs: job['zhenhong_passive_reward'] = config.get('zhenhong_passive_reward', 0.2)
            results = list(pool.map(play, jobs)); batches = {key: [] for key in keys}
            for job, result in zip(jobs, results):
                append(root, 'training-games.jsonl', dict(cycle=cycle, **{k: job[k] for k in ('key', 'foe', 'seed', 'first')},
                       **{k: v for k, v in result.items() if k not in ('trajectory', 'trajectory_returns')}))
                if result['complete']: batches[job['key']].append(result)
            # Partial batches are recorded but never update only some teams.
            if not all(r['complete'] for r in results) or time.time() >= config['train_until']: break
            for key in keys:
                update = ordinary_update(nets[key], opts[key], batches[key], config['seeds']['foundation'] + cycle)
                if update.get('updates') != 1: raise ValueError('Missing terminal trajectory update')
                totals[key]['games'] += len(batches[key]); totals[key]['samples'] += update['samples']; totals[key]['optimizer_steps'] += 1
                append(root, 'losses.jsonl', dict(cycle=cycle, key=key, **update))
            cycle += 1; current = root / 'generations' / f'cycle-{cycle:05}'
            for key in keys:
                save_weights({k: v.detach().cpu().numpy() for k, v in nets[key].state_dict().items()},
                             current, key, target_builds[key], origins[key])
            write(root / 'checkpoint.json', dict(cycle=cycle, current=str(current), totals=totals))
        latest = root / 'latest'
        import shutil
        latest.mkdir(exist_ok=True)
        for p in current.iterdir(): shutil.copy2(p, latest / p.name)
        training = dict(cycles=cycle, totals=totals, finished=time.time(), rule_hash=identity(),
                        algorithm='outcome_ppo_with_zhenhong_passive', zhenhong_passive_reward=config.get('zhenhong_passive_reward', 0.2), fixed_builds=True, frozen_opponents=True)
        write(root / 'training-summary.json', training)
        if cycle == 0: raise RuntimeError('No complete training cycle')
        write(root / 'status.json', dict(phase='evaluating', training=training, hard_deadline=config['hard_deadline']))
        pairs = config['evaluation_pairs']; all_rows = {}
        # Interleaving variants prevents a slow baseline from consuming all candidate time.
        for i in range(pairs):
            if time.time() >= config['evaluate_until']: break
            for variant in ('initial', 'latest'):
                rows = evaluate(pool, root, root / variant, config['seeds']['audit'] + i, config['evaluate_until'], 1, keys)
                all_rows.setdefault(variant, []).extend(rows)
        if frozen != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'reference').iterdir()}:
            raise ValueError('Frozen opponent identity changed')
        reports = {v: summary(all_rows.get(v, []), pairs, keys) for v in ('initial', 'latest')}
        complete = all(g['complete'] for groups in reports.values() for g in groups.values())
        write(root / 'summary.json', dict(training=training, evaluation=reports, complete=complete,
              evaluation_pairs=pairs, automatic_serving_approval=False, quality_approved=False,
              note='Fixed endpoint descriptive short-run evaluation; no automatic promotion or balance claim.'))
        write(root / 'status.json', dict(phase='finished', complete=complete, finished=time.time(), totals=totals))


def migrate_fixed(source, key, destination, build=None):
    """Explicit research migration across rule changes; retain provenance, not approval."""
    from .fixed_lineup import SCHEMA, candidate_names, tensor_shapes
    from .build_acceptance import check_source
    from .league_schema import build_hash
    check_source(source)
    source = Path(source)
    metadata = json.loads((source / f'{key}.json').read_text(encoding='utf-8'))
    path = source / f'{key}.npz'
    if metadata['schema'] != SCHEMA or metadata['deck'] != key:
        raise ValueError('Unsupported warm-start schema')
    if metadata['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Source checksum mismatch')
    if metadata['build_sha256'] != build_hash(metadata['build']):
        raise ValueError('Source build mismatch')
    features = metadata['features']; target_features = all_features()
    prior = [n for n in target_features if not n.startswith('fixed:damage_immune:')]
    if features not in (prior, target_features) or metadata['candidates'] != candidate_names():
        raise ValueError('Unknown feature migration')
    hidden = metadata['hidden']
    if type(hidden) != int or not 1 <= hidden <= 256: raise ValueError('Invalid width')
    expected = tensor_shapes(hidden); expected['state_net.0.weight'] = (hidden, len(features))
    with np.load(path, allow_pickle=False) as data: weights = {n: data[n].copy() for n in expected}
    for name, shape in expected.items():
        if weights[name].shape != shape or not np.isfinite(weights[name]).all(): raise ValueError('Invalid tensor')
    before = weights['state_net.0.weight']; after = np.zeros((hidden, len(target_features)), np.float32)
    index = {n: i for i, n in enumerate(target_features)}
    for i, name in enumerate(features): after[:, index[name]] = before[:, i]
    weights['state_net.0.weight'] = after
    save_weights(weights, destination, key, build or metadata['build'], dict(
        kind='explicit_fixed_research_migration', source_sha256=metadata['sha256'],
        source_rule_hash=metadata['rule_hash'], source_build_sha256=metadata['build_sha256'],
        approval_inherited=False, new_features_zero=True))
