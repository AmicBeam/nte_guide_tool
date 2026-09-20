"""Offline ten-character fixed-build policies. Never authorize website serving."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import sys
import numpy as np
from . import league_schema as old
from .league_policy import LeagueModel, shapes
from .league_observation import encode_league

SCHEMA = 'fixed_ten_v1'
SEATS = (*old.SEATS, 'zhenhong', 'yi')
SEAT_INDEX = {v: i for i, v in enumerate(SEATS)}
CARD_IDS = (*old.CARD_IDS, *(f'{p}{i:02}' for p in ('R', 'I') for i in range(1, 9)), 'RF01')
CARD_INDEX = {v: i for i, v in enumerate(CARD_IDS)}
CAND_DIM = 11 + len(CARD_IDS) + len(old.OPTION_IDS) + 4
KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'midrange')
FOES = old.PRESETS
EXTRA_SIDE = ('delay_left', 'delay_by', 'delay_slow', 'delay_turns', 'skip_normal_attack',
              'genesis_this_turn', 'genesis_player_hits')
EXTRA_CHAR = ('beast_fangs', 'intimidation', 'extra_attacks', 'attack', 'timed_attack_1', 'timed_attack_2', 'damage_immune')


def feature_names():
    names = ['turn', 'first', 'escalation', 'phase', 'choice_kind', 'resolving', 'choice_count']
    names += [f'{k}:{s}' for k in old.SIDE_FIELDS for s in (0, 1)]
    names += [f'{k}:{s}:{c}' for k in old.CHAR_FIELDS for s in (0, 1) for c in SEATS]
    names += [f'cards:{z}:{c}' for z in range(5) for c in CARD_IDS]
    names += [f'marks:{d}:{c}' for d in (-1, 0, 1) for c in CARD_IDS]
    return names


def all_features():
    return feature_names() + [f'fixed:{k}:{s}' for k in EXTRA_SIDE for s in (0, 1)] + [
        f'fixed:{k}:{s}:{c}' for k in EXTRA_CHAR for s in (0, 1) for c in SEATS]


def candidate_names(cards=CARD_IDS):
    return [f'base:{i}' for i in range(11)] + [f'mulligan:{c}' for c in cards] + [
        f'option:{o}' for o in old.OPTION_IDS] + ['marked', 'delta', 'set_attack', 'instant']


@lru_cache(maxsize=1)
def identity():
    digest = sha256(old.rule_hash().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def encode(view, actions):
    x, c = encode_league(view, actions, contract=sys.modules[__name__])
    side = view['viewer_side']; order = (side, 'b' if side == 'a' else 'a')
    sides = []; heroes = []
    for s in order:
        team = view['sides'][s]; meta = view['policy_state']['sides'][s]
        delay = (team.get('front_debuff') or {}).get('delay') or {}
        sides.append(dict(delay_left=delay.get('left', 0), delay_by=-1 if not delay else int(delay.get('by') != s),
                          delay_slow=delay.get('slow', 0) if delay.get('slow_turn') == view['turn'] else 0,
                          delay_turns=meta.get('delay_turns') or 0, skip_normal_attack=bool(meta.get('skip_normal_attack')),
                          genesis_this_turn=meta.get('genesis_turn') == view['turn'],
                          genesis_player_hits=meta.get('genesis_player_hits') or 0))
        indexed = {h['id']: h for h in team['characters']}; rows = []
        for cid in SEATS:
            h = indexed.get(cid, {}); flags = (meta['characters'].get(cid) or {}).get('flags') or {}
            row = {k: h.get(k, 0) for k in EXTRA_CHAR}
            for left in (1, 2):
                row[f'timed_attack_{left}'] = sum(e['amount'] for e in flags.get('timed_attack') or [] if e['left'] == left)
            rows.append(row)
        heroes.append(rows)
    extras = [row[k] / 10 for k in EXTRA_SIDE for row in sides]
    extras += [row[k] / 10 for k in EXTRA_CHAR for group in heroes for row in group]
    result = np.concatenate((x, np.asarray(extras, dtype=np.float32)))
    if result.shape != (len(all_features()),) or not np.isfinite(result).all():
        raise ValueError('Invalid fixed observation')
    return result, c


def builds(source):
    """Keep old two serving builds; new teams reuse current fixed preset kits."""
    from ..content.duel_v2 import STARTER_DECKS, CARDS, validate_deck
    from copy import deepcopy
    presets = {d['id']: d for d in STARTER_DECKS}
    result = {}
    for key in ('starter', 'weave-rush'):
        manifest = json.loads((Path(source) / f'{key}.json').read_text(encoding='utf-8'))
        result[key] = validate_deck(manifest['build'])
    for key in ('quick-rush', 'zhenhong'):
        result[key] = validate_deck(deepcopy(presets[key]))
    entries = [('starter', 'nanali'), ('weave-rush', 'baicang'), ('starter', 'zero'), ('starter', 'iloy')]
    result['midrange'] = validate_deck(dict(id='midrange', name='娜白零伊',
        character_ids=[c for _, c in entries], card_ids=[card for preset, cid in entries
            for card in presets[preset]['card_ids'] if CARDS[card]['character_id'] == cid]))
    for build in result.values():
        if not set(build['character_ids']) <= set(SEATS): raise ValueError('Unsupported team')
        if any(CARDS[c].get('training_excluded') for c in build['card_ids']): raise ValueError('Excluded card')
    return result


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'); temp.replace(path)


def tensor_shapes(hidden):
    result = shapes(hidden)
    result['state_net.0.weight'] = (hidden, len(all_features()))
    result['cand_net.0.weight'] = (hidden, CAND_DIM)
    return result


class FixedModel(LeagueModel):
    def __init__(self, directory, key):
        root = Path(directory); m = json.loads((root / f'{key}.json').read_text(encoding='utf-8'))
        p = root / f'{key}.npz'; digest = sha256(p.read_bytes()).hexdigest()
        if (m['schema'] != SCHEMA or m['rule_hash'] != identity() or m['sha256'] != digest
                or m['features'] != all_features() or m['candidates'] != candidate_names() or m['deck'] != key):
            raise ValueError('Fixed policy identity mismatch')
        self.hidden = m['hidden']
        if type(self.hidden) != int or not 1 <= self.hidden <= 256: raise ValueError('Invalid width')
        with np.load(p, allow_pickle=False) as z: self.weights = {n: z[n].copy() for n in tensor_shapes(self.hidden)}
        for n, shape in tensor_shapes(self.hidden).items():
            if self.weights[n].shape != shape or not np.isfinite(self.weights[n]).all(): raise ValueError('Invalid tensor')
            self.weights[n].flags.writeable = False
        self.serving_deck = m['build']; self.manifest = m; self.version = digest
        if m['build_sha256'] != old.build_hash(self.serving_deck): raise ValueError('Build mismatch')
        if not set(self.serving_deck['character_ids']) <= set(SEATS): raise ValueError('Unsupported model build')

    def select_public_action(self, view, actions):
        return int(self.scores(*encode(view, actions)).argmax())

    def validate_human(self, build):
        raise ValueError('Research policy has no website authorization')


def save_weights(weights, directory, key, build, origin):
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    tmp = root / f'{key}.tmp.npz'; np.savez_compressed(tmp, **weights); p = root / f'{key}.npz'; tmp.replace(p)
    write(root / f'{key}.json', dict(schema=SCHEMA, deck=key, rule_hash=identity(), sha256=sha256(p.read_bytes()).hexdigest(),
          features=all_features(), candidates=candidate_names(), hidden=len(weights['state_net.0.bias']),
          build=build, build_sha256=old.build_hash(build), origin=origin, validation={'approved': False}, automatic_serving_approval=False))


def expand_source(directory, key):
    root = Path(directory); m = json.loads((root / f'{key}.json').read_text(encoding='utf-8'))
    p = root / f'{key}.npz'
    from .build_acceptance import check_source
    check_source(root)
    if (m['schema'] != old.SCHEMA or m['deck'] != key or m['features'] != old.feature_names()
            or m['seat_ids'] != list(old.SEATS) or m['card_ids'] != list(old.CARD_IDS)
            or m['cand_dim'] != old.CAND_DIM or m['sha256'] != sha256(p.read_bytes()).hexdigest()
            or m['build_sha256'] != old.build_hash(m['build'])): raise ValueError('Invalid source identity')
    if type(m['hidden']) != int or not 1 <= m['hidden'] <= 256: raise ValueError('Invalid source width')
    with np.load(p, allow_pickle=False) as z: source = {n: z[n].copy() for n in shapes(m['hidden'])}
    for n, shape in shapes(m['hidden']).items():
        if source[n].shape != shape or not np.isfinite(source[n]).all(): raise ValueError('Invalid source tensor')
    target = {n: v.copy() for n, v in source.items()}
    for tensor, before, after in [('state_net.0.weight', old.feature_names(), all_features()),
                                   ('cand_net.0.weight', candidate_names(old.CARD_IDS), candidate_names())]:
        target[tensor] = np.zeros((m['hidden'], len(after)), dtype=np.float32)
        indexes = {name: i for i, name in enumerate(after)}
        for i, name in enumerate(before): target[tensor][:, indexes[name]] = source[tensor][:, i]
    return target, m
