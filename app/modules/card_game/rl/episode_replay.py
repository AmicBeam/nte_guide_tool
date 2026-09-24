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


def _row_id(identity, row, index):
    kind = 'pi' if 'pi' in row else 'v'
    return '%s:%s:%s:%s:%s' % (identity, row.get('side', ''), row.get('step', index), kind, index)


def _valid_policy_row(row):
    if 'pi' not in row or 'c' not in row or row.get('z') not in (-1, 0, 1):
        return False
    if row.get('reward_to_go') is not None or row.get('setup_reward', 0):
        return False
    policy = np.asarray(row['pi'])
    if policy.ndim != 1 or len(policy) != len(row['c']):
        return False
    return bool(np.isfinite(policy).all() and np.all(policy >= 0) and np.isclose(policy.sum(), 1))


class EpisodePool:
    def __init__(self, limit=128, max_row_reuse=8):
        if type(max_row_reuse) is not int or max_row_reuse < 1:
            raise ValueError('Invalid max_row_reuse')
        self.limit = limit
        self.max_row_reuse = max_row_reuse
        self.episodes = []
        self.row_uses = {}
        self.fresh_ids = set()
        self.first_use = 0
        self.repeated_use = 0
        self.batch_first_use = 0
        self.batch_repeated_use = 0

    def begin_fresh_batch(self):
        self.fresh_ids = set()
        self.batch_first_use = 0
        self.batch_repeated_use = 0

    def add(self, identity, data):
        if any(e['id'] == identity for e in self.episodes):
            raise ValueError('Duplicate episode')
        rows = deepcopy(data['rows'])
        if not rows or any(r['z'] not in (-1, 0, 1) for r in rows):
            raise ValueError('Missing terminal rows')
        for index, row in enumerate(rows):
            row['row_id'] = _row_id(identity, row, index)
            if _valid_policy_row(row):
                self.fresh_ids.add(row['row_id'])
        self.episodes.append(dict(id=identity, rows=rows, used=0, priority=np.ones(len(rows)).tolist()))
        if len(self.episodes) > self.limit:
            # Preserve up to one quarter rare winning episodes; never synthesize wins.
            # A mirror episode contains both seats. Retention must not depend on
            # which seat produced the first row (usually the first player).
            protected = {e['id'] for e in self.episodes if any(r['z'] == 1 for r in e['rows'])}
            protected = set(list(e['id'] for e in self.episodes if e['id'] in protected)[-max(1, self.limit//4):])
            i = next(i for i, e in enumerate(self.episodes) if e['id'] not in protected)
            self._forget(self.episodes.pop(i))

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
            episode['rows'][i] = {**new,'original_pi':row.get('original_pi',old),'row_id':row.get('row_id') or _row_id(identity, row, i)}
            episode['priority'][i] += float(np.abs(old-new['pi']).sum())

    def coverage_report(self):
        fresh = sorted(self.fresh_ids)
        covered = [row_id for row_id in fresh if self.row_uses.get(row_id, 0) > 0]
        return dict(
            fresh_policy_rows=len(fresh),
            unique_row_ids=covered,
            unique_covered=len(covered),
            first_use_coverage=(len(covered) / len(fresh) if fresh else 1.0),
            first_use=self.batch_first_use,
            repeated_use=self.batch_repeated_use,
            total_first_use=self.first_use,
            total_repeated_use=self.repeated_use,
        )

    def export_state(self):
        return dict(
            limit=self.limit, max_row_reuse=self.max_row_reuse, episodes=self.episodes,
            row_uses=dict(self.row_uses), fresh_ids=sorted(self.fresh_ids),
            first_use=self.first_use, repeated_use=self.repeated_use,
            batch_first_use=self.batch_first_use, batch_repeated_use=self.batch_repeated_use,
        )

    @classmethod
    def from_state(cls, state):
        if isinstance(state, list):
            pool = cls()
            pool.episodes = state
            return pool
        pool = cls(limit=state.get('limit', 128), max_row_reuse=state.get('max_row_reuse', 8))
        pool.episodes = state['episodes']
        pool.row_uses = dict(state.get('row_uses') or {})
        pool.fresh_ids = set(state.get('fresh_ids') or ())
        pool.first_use = int(state.get('first_use') or 0)
        pool.repeated_use = int(state.get('repeated_use') or 0)
        pool.batch_first_use = int(state.get('batch_first_use') or 0)
        pool.batch_repeated_use = int(state.get('batch_repeated_use') or 0)
        return pool

    def policy_pass_batches(self, rng, *, passes, batch_size=128, deadline=None, include_old_replay=True):
        """Yield policy-only minibatches. Fresh unused rows first; repeats are not new."""
        if passes not in (1, 4, 8):
            raise ValueError('policy_passes must be 1, 4 or 8')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Invalid policy batch size')
        yield from self._iter_refs(self._policy_refs(fresh=True, unused_only=True), rng, batch_size, deadline, 'fresh')
        leftover = self._policy_refs(fresh=True, unused_only=True)
        if leftover:
            return
        for _ in range(1, passes):
            yield from self._iter_refs(self._policy_refs(fresh=True, unused_only=False), rng, batch_size, deadline, 'fresh_repeat')
        if include_old_replay and self.fresh_ids and (deadline is None or time.time() < deadline):
            old = self._policy_refs(fresh=False, unused_only=False)
            rng.shuffle(old)
            yield from self._iter_refs(old[:batch_size], rng, batch_size, deadline, 'old_replay')

    def _policy_refs(self, *, fresh, unused_only):
        refs = []
        for episode in self.episodes:
            for index, row in enumerate(episode['rows']):
                if not _valid_policy_row(row):
                    continue
                row_id = row.get('row_id') or _row_id(episode['id'], row, index)
                if fresh and row_id not in self.fresh_ids:
                    continue
                if not fresh and row_id in self.fresh_ids:
                    continue
                uses = self.row_uses.get(row_id, 0)
                if unused_only and uses:
                    continue
                if uses >= self.max_row_reuse:
                    continue
                refs.append((episode, index, row_id))
        return refs

    def _iter_refs(self, refs, rng, batch_size, deadline, source):
        refs = list(refs)
        if not refs:
            return
        rng.shuffle(refs)
        for start in range(0, len(refs), batch_size):
            if deadline is not None and time.time() >= deadline:
                return
            rows, chosen, stats = self._take(refs[start:start + batch_size], source)
            if rows:
                yield rows, chosen, stats

    def _take(self, refs, source):
        rows = []
        chosen = []
        row_ids = []
        first_use = 0
        repeated_use = 0
        for episode, index, row_id in refs:
            previous = self.row_uses.get(row_id, 0)
            if previous >= self.max_row_reuse:
                continue
            if previous == 0:
                first_use += 1
                self.first_use += 1
                if row_id in self.fresh_ids:
                    self.batch_first_use += 1
            else:
                repeated_use += 1
                self.repeated_use += 1
                if row_id in self.fresh_ids:
                    self.batch_repeated_use += 1
            self.row_uses[row_id] = previous + 1
            episode['used'] += 1
            rows.append({**episode['rows'][index], 'sample_weight': 1.})
            chosen.append((episode['id'], index))
            row_ids.append(row_id)
        return rows, chosen, dict(
            source=source, row_ids=row_ids, first_use=first_use, repeated_use=repeated_use,
            unique_rows=sum(1 for count in self.row_uses.values() if count > 0),
            unique_row_ids=sorted(row_id for row_id, count in self.row_uses.items() if count > 0),
        )

    def _forget(self, episode):
        for index, row in enumerate(episode['rows']):
            row_id = row.get('row_id') or _row_id(episode['id'], row, index)
            self.fresh_ids.discard(row_id)
            self.row_uses.pop(row_id, None)


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
