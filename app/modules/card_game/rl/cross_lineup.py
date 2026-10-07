"""Union observation for five-preset cross play; old schemas stay unchanged."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import sys
import numpy as np
from . import league_schema as old
from .league_policy import LeagueModel, shapes
from .league_observation import encode_league

SCHEMA = 'cross_five_grounded_wdl_v1'
from . import murk_lineup as murk
SEATS = (*old.SEATS, 'zhenhong', 'yi', *murk.SEATS)
SEAT_INDEX = {v: i for i, v in enumerate(SEATS)}
CARD_IDS = (*old.CARD_IDS, *(f'{p}{i:02}' for p in ('R', 'I') for i in range(1, 9)), 'RF01', *murk.CARD_IDS)
CARD_INDEX = {v: i for i, v in enumerate(CARD_IDS)}
CANDIDATE_CONTEXT = ('copy', 'expires_in', 'derived_override', 'antique_investment')
CAND_DIM = 11 + 2*len(CARD_IDS) + len(old.OPTION_IDS) + 4 + len(CANDIDATE_CONTEXT)
KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
FOES = old.PRESETS
EXTRA_SIDE = ('delay_left', 'delay_by', 'delay_slow', 'delay_turns', 'skip_normal_attack',
              'genesis_this_turn', 'genesis_player_hits')
EXTRA_CHAR = ('beast_fangs', 'intimidation', 'extra_attacks', 'attack', 'timed_attack_1', 'timed_attack_2', 'damage_immune', 'damage_limit')


def feature_names():
    names = ['turn', 'first', 'escalation', 'phase', 'choice_kind', 'resolving', 'choice_count']
    names += [f'{k}:{s}' for k in old.SIDE_FIELDS for s in (0, 1)]
    names += [f'{k}:{s}:{c}' for k in old.CHAR_FIELDS for s in (0, 1) for c in SEATS]
    names += [f'cards:{z}:{c}' for z in range(5) for c in CARD_IDS]
    names += [f'marks:{d}:{c}' for d in (-1, 0, 1) for c in CARD_IDS]
    return names


def all_features():
    return feature_names() + [f'fixed:{k}:{s}' for k in EXTRA_SIDE for s in (0, 1)] + [
        f'fixed:{k}:{s}:{c}' for k in EXTRA_CHAR for s in (0, 1) for c in SEATS] + murk.extra_names() + [
        f'substituted_hand:{cid}' for cid in CARD_IDS]


def candidate_names(cards=CARD_IDS):
    return [f'base:{i}' for i in range(11)] + [f'mulligan:{c}' for c in cards] + [
        f'option:{o}' for o in old.OPTION_IDS] + ['marked', 'delta', 'set_attack', 'instant'] + list(CANDIDATE_CONTEXT) + [
        f'physical:{cid}' for cid in cards]


@lru_cache(maxsize=1)
def identity():
    digest = sha256(old.rule_hash().encode())
    for name in ('cross_lineup.py', 'cross_grounded.py', 'cross_runtime.py', 'murk_lineup.py', 'information_search.py', 'search_policy.py'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def encode(view, actions):
    view, actions = murk._adapt(view, actions)
    x, c = encode_league(view, actions, contract=sys.modules[__name__])
    own = view['sides'][view['viewer_side']]
    hand = {card['instance_id']: card for card in own['hand']}
    choices = {item['id']: item['card'] for item in (view.get('pending_choice') or {}).get('choices', [])}
    from app.modules.card_game.content.duel_v2 import CARDS
    context_start=CAND_DIM-len(CARD_IDS)-len(CANDIDATE_CONTEXT)
    physical_start=CAND_DIM-len(CARD_IDS)
    for index, action in enumerate(actions):
        item = hand.get(action.get('card_id'), {}) if action['type'] == 'play_card' else choices.get(action.get('choice_id'), {})
        if item:
            # Only the acting player's visible card semantics. Never add an
            # entity/position ID to let the network memorize Gumbel noise.
            c[index, context_start:physical_start] = [bool(item.get('copy')),
                max(0, int(item.get('expires_turn') or 0) - own['turn_count']) / 10,
                bool(item.get('derived')) != bool(CARDS[item['card_id']].get('derived')),
                float(item.get('antique_investment') or 0)]
            face=item.get('hand_face') or {}
            if face.get('card_id') and face['card_id']!=item['card_id']:
                c[index,physical_start+CARD_INDEX[face['card_id']]]=1
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
            row = {k: h.get(k) or 0 for k in EXTRA_CHAR}
            for left in (1, 2):
                row[f'timed_attack_{left}'] = sum(e['amount'] for e in flags.get('timed_attack') or [] if e['left'] == left)
            rows.append(row)
        heroes.append(rows)
    extras = [row[k] / 10 for k in EXTRA_SIDE for row in sides]
    extras += [row[k] / 10 for k in EXTRA_CHAR for group in heroes for row in group]
    extras += [v for side in order for v in murk._zone_extra(view['sides'][side])]
    extras += [murk._mirage(view['sides'][side]) for side in order]
    extras += [murk._dot_attack_used(view['sides'][side], view['turn']) for side in order]
    physical_counts=np.zeros(len(CARD_IDS),dtype=np.float32)
    for card in own['hand']:
        face=card.get('hand_face') or {}
        if face.get('card_id') and face['card_id']!=card['card_id']:
            physical_counts[CARD_INDEX[face['card_id']]]+=.5
    extras += physical_counts.tolist()
    result = np.concatenate((x, np.asarray(extras, dtype=np.float32)))
    if result.shape != (len(all_features()),) or not np.isfinite(result).all():
        raise ValueError('Invalid fixed observation')
    return result, c
