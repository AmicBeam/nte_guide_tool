"""Bounded complete-episode replay and private, information-set reanalysis."""
from copy import deepcopy
from pathlib import Path
import gzip
import json
import time
import numpy as np


def dump_episode(path, job, game):
    if not game['complete'] or game['decisions'] != len(game['episode_actions']):
        raise ValueError('Only complete replayable episodes')
    rows = [r for r in game['rows'] + game['value_rows'] if r['side'] in job['train_sides']]
    data = dict(job=job, model_versions=game.get('model_versions',{}), actions=game['episode_actions'], rows=rows, winner=game['winner'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, default=encode)
    return data


def load_episode(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    for r in data['rows']:
        for k in ('x', 'c', 'pi'):
            if k in r:
                r[k] = np.asarray(r[k], dtype=np.float32)
    return data


class EpisodePool:
    def __init__(self, limit=128):
        self.limit = limit
        self.episodes = []

    def add(self, identity, data):
        if any(e['id'] == identity for e in self.episodes):
            raise ValueError('Duplicate episode')
        rows = deepcopy(data['rows'])
        if not rows or any(r['z'] not in (-1, 0, 1) for r in rows):
            raise ValueError('Missing terminal rows')
        self.episodes.append(dict(id=identity, rows=rows, used=0, priority=np.ones(len(rows)).tolist()))
        if len(self.episodes) > self.limit:
            # Preserve up to one quarter rare winning episodes; never synthesize wins.
            protected = {e['id'] for e in self.episodes if e['rows'][0]['z'] == 1}
            protected = set(list(e['id'] for e in self.episodes if e['id'] in protected)[-max(1, self.limit//4):])
            i = next(i for i, e in enumerate(self.episodes) if e['id'] not in protected)
            self.episodes.pop(i)

    def sample(self, rng, count, prioritized):
        # Uniform-over-episode then uniform-over-decision is the reference p0.
        eligible = [e for e in self.episodes if e['used'] < 16*len(e['rows'])]
        if not eligible:
            return [], [], {}
        refs = [(e, i) for e in eligible for i in range(len(e['rows']))]
        p0 = np.array([1/len(eligible)/len(e['rows']) for e, i in refs])
        recent = eligible[-min(4,len(eligible)):]
        recent_ids = {e['id'] for e in recent}
        fresh = np.array([1/len(recent)/len(e['rows']) if e['id'] in recent_ids else 0 for e, i in refs])
        priority = np.array([max(.05,min(5.,e['priority'][i])) for e, i in refs])*p0
        priority /= priority.sum()
        q = .5*fresh + .25*p0 + .25*priority if prioritized else p0
        indexes = rng.choice(len(refs), size=count, p=q)
        rows = []
        chosen = []
        weights = []
        for index in indexes:
            e, i = refs[index]
            weight = min(4., float(p0[index]/q[index]))
            rows.append({**e['rows'][i], 'sample_weight':weight})
            chosen.append((e['id'],i))
            weights.append(weight)
            e['used'] += 1
        return rows, chosen, dict(effective_samples=float(sum(weights)**2/sum(w*w for w in weights)),
            unique_episodes=len({x[0] for x in chosen}), max_weight=max(weights), probabilities=q[indexes].tolist())

    def priorities(self, refs, errors):
        lookup = {e['id']:e for e in self.episodes}
        for (identity,i), error in zip(refs, errors):
            lookup[identity]['priority'][i] = float(error)+.05

    def replace_labels(self, identity, refreshed):
        episode = next(e for e in self.episodes if e['id']==identity)
        lookup = {(r['side'],r['step']):r for r in refreshed}
        for i, row in enumerate(episode['rows']):
            new = lookup.get((row['side'],row['step']))
            if new is None or 'pi' not in row:
                continue
            if row['z'] != new['z'] or not np.array_equal(row['x'],new['x']) or not np.array_equal(row['c'],new['c']):
                raise ValueError('Reanalysis changed observation or outcome')
            old = row['pi'].copy()
            episode['rows'][i] = {**new,'original_pi':row.get('original_pi',old)}
            episode['priority'][i] += float(np.abs(old-new['pi']).sum())


def reanalyse(job):
    from . import outcome_runtime as runtime
    from .information_search import search
    from .league_rollout import clean
    from ..engine.duel_v2 import new_game, apply_action
    data = load_episode(job['episode'])
    original = data['job']
    state = new_game(seed=original['seed'], first_side=original['first'], decks=original['decks'])
    policies = {side:runtime.model(*spec) for side,spec in job['policies'].items()}
    targets = {r['step']:r for r in data['rows'] if 'pi' in r and r['step'] in job['steps']}
    refreshed = []
    for step, entry in enumerate(data['actions']):
        if time.time() >= job['deadline']:
            break
        row = targets.get(step)
        if row is not None:
            result = search(state,row['side'],policies,seed=job['seed']+step*104729,
                            simulations=job['simulations'],deadline=job['deadline'],runtime=runtime)
            if not result['complete']:
                break
            if not np.array_equal(row['x'],result['x']) or not np.array_equal(row['c'],result['c']):
                raise ValueError('Episode reconstruction mismatch')
            refreshed.append({**row,'pi':result['pi'],'reanalysis_model':policies[row['side']].version})
        if len(refreshed)==len(targets):
            break
        state=clean(apply_action(state,entry['side'],entry['action']))
    return dict(episode_id=job['episode_id'],rows=refreshed,requested=len(targets),complete=len(refreshed)==len(targets))
