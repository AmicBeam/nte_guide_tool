"""Four-seat observation for 浊燃预组. Does not change the ten-seat tensors."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import numpy as np
from .league_schema import CHAR_FIELDS, OPTION_IDS, SIDE_FIELDS, rule_hash
from .league_observation import encode_league

SCHEMA = 'murk_grounded_wdl_v1'
SEATS = ('anhunqu', 'canhong', 'zaowu', 'adler', 'summon')
SEAT_INDEX = {v: i for i, v in enumerate(SEATS)}
CARD_IDS = tuple(f'A{i:02}' for i in range(1, 9)) + ('AF01',) + tuple(
    f'{prefix}{i:02}' for prefix in 'CSD' for i in range(1, 9))
CARD_INDEX = {v: i for i, v in enumerate(CARD_IDS)}
CAND_DIM = 11 + len(CARD_IDS) + len(OPTION_IDS) + 4
DOT_KINDS = ('nightmare', 'etch', 'venom', 'guard')
GENERATED = ('AF01', 'A03')
KEY = 'murk'


def feature_names():
    names = ['turn', 'first', 'escalation', 'phase', 'choice_kind', 'resolving', 'choice_count']
    names += [f'{k}:{s}' for k in SIDE_FIELDS for s in (0, 1)]
    names += [f'{k}:{s}:{c}' for k in CHAR_FIELDS for s in (0, 1) for c in SEATS]
    names += [f'cards:{z}:{c}' for z in range(5) for c in CARD_IDS]
    names += [f'marks:{d}:{c}' for d in (-1, 0, 1) for c in CARD_IDS]
    return names


def extra_names():
    names = [f'murk:{k}:{s}' for s in (0, 1) for k in ('burn_stacks', *DOT_KINDS)]
    names += [f'murk:mirage:{s}' for s in (0, 1)]
    names += [f'murk:dot_attack_used:{s}' for s in (0, 1)]
    return names


def all_features():
    return feature_names() + extra_names()


def candidate_names(cards=CARD_IDS):
    return [f'base:{i}' for i in range(11)] + [f'mulligan:{c}' for c in cards] + [
        f'option:{o}' for o in OPTION_IDS] + ['marked', 'delta', 'set_attack', 'instant']


def identity():
    digest = sha256(rule_hash().encode())
    root = Path(__file__).resolve().parent
    for name in ('murk_lineup.py', 'murk_grounded.py', 'murk_runtime.py'):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _rewrite(value):
    return 'summon' if isinstance(value, str) and value.startswith('summon_') else value


def _adapt(view, actions):
    view = deepcopy(view)
    actions = deepcopy(list(actions))
    for side in ('a', 'b'):
        team = view['sides'][side]
        chars = list(team.get('characters') or [])
        summons = [h for h in chars if h.get('summoned') or str(h.get('id', '')).startswith('summon_')]
        chars = [h for h in chars if h not in summons]
        if summons:
            chosen = next((h for h in summons if team.get('front') == h.get('id')), summons[-1])
            old = chosen['id']
            chosen = dict(chosen)
            chosen['id'] = 'summon'
            chars.append(chosen)
            meta = view['policy_state']['sides'][side]['characters']
            if old in meta:
                meta['summon'] = meta.pop(old)
            for other in summons:
                meta.pop(other.get('id'), None)
        for key in ('front', 'last_front'):
            team[key] = _rewrite(team.get(key))
        team['characters'] = chars
    for action in actions:
        if action.get('character_id'):
            action['character_id'] = _rewrite(action['character_id'])
        target = action.get('target_id')
        if isinstance(target, str) and ':' in target:
            side, cid = target.split(':', 1)
            action['target_id'] = f'{side}:{_rewrite(cid)}'
    return view, actions


def _zone_extra(team):
    zones = team.get('front_debuff') or {}
    burn = zones.get('burn') or {}
    dots = zones.get('dots') or {}
    values = [int(burn.get('stacks') or 0) / 10]
    values += [int((dots.get(kind) or {}).get('stacks') or 0) / 10 for kind in DOT_KINDS]
    return values


def _mirage(team):
    for hero in team.get('characters') or []:
        if hero.get('id') != 'canhong':
            continue
        if any(effect.get('definition_id') == 'canhong.mirage' for effect in hero.get('effects') or []):
            return 1.0
    return 0.0


def _dot_attack_used(team, turn):
    for hero in team.get('characters') or []:
        if hero.get('id') == 'canhong':
            return float((team.get('used') or {}).get(
                f"canhong_dot_attack_turn:{hero['entity_id']}") == turn)
    return 0.0


def encode(view, actions):
    view, actions = _adapt(view, actions)
    state, candidates = encode_league(view, actions, contract=sys.modules[__name__])
    viewer = view['viewer_side']
    order = (viewer, 'b' if viewer == 'a' else 'a')
    extra = [value for side in order for value in _zone_extra(view['sides'][side])]
    extra += [_mirage(view['sides'][side]) for side in order]
    extra += [_dot_attack_used(view['sides'][side], view['turn']) for side in order]
    result = np.concatenate((state, np.asarray(extra, dtype=np.float32)))
    if result.shape != (len(all_features()),) or not np.isfinite(result).all():
        raise ValueError('Invalid murk observation')
    if len(actions) and (candidates.shape != (len(actions), CAND_DIM) or not np.isfinite(candidates).all()):
        raise ValueError('Invalid murk candidates')
    return result, candidates


def deck():
    from copy import deepcopy as copy_deck
    from ..content.duel_v2 import STARTER_DECKS, validate_deck
    return validate_deck(copy_deck(next(item for item in STARTER_DECKS if item['id'] == KEY)))
