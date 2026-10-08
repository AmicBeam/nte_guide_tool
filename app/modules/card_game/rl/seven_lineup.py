"""Union observation for seven-preset cross play; old schemas stay unchanged."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import sys
import numpy as np
from . import league_schema as old
from .league_policy import LeagueModel, shapes
from .league_observation import encode_league

SCHEMA = 'cross_seven_grounded_wdl_v1'
from . import murk_lineup as murk
SEATS = (*old.SEATS, 'zhenhong', 'yi', *murk.SEATS, 'heiyu', 'dafutier', 'lingke')
SEAT_INDEX = {v: i for i, v in enumerate(SEATS)}
CARD_IDS = (*old.CARD_IDS, *(f'{p}{i:02}' for p in ('R', 'I') for i in range(1, 9)), 'RF01', *murk.CARD_IDS)
from ..content.duel_v2 import CARDS, PUBLIC_CHARACTER_IDS
CARD_IDS += tuple(sorted(cid for cid,card in CARDS.items() if card['character_id'] in PUBLIC_CHARACTER_IDS and cid not in CARD_IDS))
CARD_INDEX = {v: i for i, v in enumerate(CARD_IDS)}
CANDIDATE_CONTEXT = ('copy', 'expires_in', 'derived_override', 'antique_investment')
CAND_DIM = 11 + 2*len(CARD_IDS) + len(old.OPTION_IDS) + 4 + len(CANDIDATE_CONTEXT)
KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk', 'collapse', 'lingke')
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


def _old_all_features():
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
    for name in ('seven_lineup.py', 'seven_grounded.py', 'seven_runtime.py', 'seven_policy.py', 'murk_lineup.py', 'information_search.py', 'search_policy.py'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def _encode_old_fields(view, actions):
    view, actions = murk._adapt(view, actions)
    x, c = encode_league(view, actions, contract=sys.modules[__name__])
    own = view['sides'][view['viewer_side']]
    hand = {card['instance_id']: card for card in own['hand']}
    choices = {item['id']: item['card'] for item in (view.get('pending_choice') or {}).get('choices', [])}
    from app.modules.card_game.content.duel_v2 import CARDS
    context_start=OLD_CAND_DIM-len(CARD_IDS)-len(CANDIDATE_CONTEXT)
    physical_start=OLD_CAND_DIM-len(CARD_IDS)
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
    if result.shape != (len(_old_all_features()),) or not np.isfinite(result).all():
        raise ValueError('Invalid fixed observation')
    return result, c


ACTION_TYPES = ('end_turn', 'attack', 'ultimate', 'play_card', 'choose', 'mulligan',
                'choose_deck_order', 'exit_ultimate', 'choose_cards')
ORDER_SLOTS = 4
OLD_CAND_DIM = CAND_DIM
NEW_SIDE = ('max_hp', 'insight_damage', 'insight_played', 'support_count', 'heiyu_divinations')
NEW_CHAR = ('collapse_max', 'chain_count', 'chain_actor', 'insight_used')
ATTRIBUTES = ('光', '暗', '灵', '魂', '相', '咒')
CHOICES = ('inspect_top','discard','enemy_hand','kit','kit_multi','heiyu_divination','hand_deck_placement','hand_redraw')
ORDER_FIELDS = tuple(f'order:{zone}:{slot}:{cid}' for zone in ('top','bottom')
                     for slot in range(ORDER_SLOTS) for cid in CARD_IDS)
SELECT_FIELDS = tuple(f'select:{cid}' for cid in CARD_IDS)
EXTRA_CANDIDATES = ORDER_FIELDS + SELECT_FIELDS + ('additional_ap_cost',)
CAND_DIM = OLD_CAND_DIM + len(EXTRA_CANDIDATES)
_old_candidate_names = candidate_names


def candidate_names(cards=CARD_IDS):
    return _old_candidate_names(cards) + list(EXTRA_CANDIDATES)


def all_features():
    return _old_all_features() + [f'seven:{k}:{s}' for s in (0,1) for k in NEW_SIDE] + [
        f'seven:{k}:{s}:{cid}' for s in (0,1) for cid in SEATS for k in NEW_CHAR] + [
        f'seven:resistance:{s}:{attr}' for s in (0,1) for attr in ATTRIBUTES] + [
        f'seven:resistance:{s}:{cid}:{attr}' for s in (0,1) for cid in SEATS for attr in ATTRIBUTES] + [
        f'seven:choice:{kind}' for kind in CHOICES] + [
        f'seven:choice_card:{cid}' for cid in CARD_IDS]


def encode(view, actions):
    # New actions have their own categorical IDs and ordered/card semantics.
    # A temporary end-turn row only obtains the unchanged legacy field block.
    from copy import deepcopy
    actions = list(actions)
    old_actions = [a if a['type'] in ACTION_TYPES[:6] else {'type':'end_turn'} for a in actions]
    adapted = deepcopy(view)
    x, c = _encode_old_fields(adapted, old_actions)
    # The shared packer allocates the expanded width; physical columns in the
    # legacy encoder must still refer to OLD_CAND_DIM, not new order columns.
    c = c[:,:OLD_CAND_DIM]
    extra = np.zeros((len(actions),len(EXTRA_CANDIDATES)),dtype=np.float32)
    own = view['sides'][view['viewer_side']]
    cards = {item['instance_id']:item for item in own['hand']}
    pending = view.get('pending_choice') or {}
    choices = {item['id']:item['card'] for item in pending.get('choices',[])}
    for i,a in enumerate(actions):
        if a['type'] not in ACTION_TYPES: raise ValueError('Unencoded seven-team action: '+a['type'])
        c[i,0] = ACTION_TYPES.index(a['type'])
        if a['type']=='exit_ultimate':c[i,1]=SEAT_INDEX[a['character_id']]
        if a['type']=='choose_deck_order':
            zones = (a['card_ids'], a.get('bottom_card_ids', [a['bottom_card_id']] if a.get('bottom_card_id') else []))
            for z,ids in enumerate(zones):
                if len(ids)>ORDER_SLOTS: raise ValueError('Deck order exceeds declared four-card contract')
                for slot,instance in enumerate(ids):
                    card=choices[instance]
                    extra[i,(z*ORDER_SLOTS+slot)*len(CARD_IDS)+CARD_INDEX[card['card_id']]]=1
        if a['type']=='choose_cards':
            for instance in a['card_ids']:
                card=choices.get(instance) or cards[instance]
                extra[i,len(ORDER_FIELDS)+CARD_INDEX[card['card_id']]]+=.5
        if a['type']=='play_card':extra[i,-1]=float(cards[a['card_id']].get('additional_ap_cost') or 0)/10
    side=view['viewer_side'];order=(side,'b' if side=='a' else 'a')
    fields=[]
    for s in order:
        team=view['sides'][s]
        fields.extend(float((view['policy_state']['sides'][s].get(k) if k=='heiyu_divinations' else team.get(k)) or 0)/10 for k in NEW_SIDE)
    for s in order:
        team=view['sides'][s];indexed={h['id']:h for h in team['characters']}
        for cid in SEATS:
            h=indexed.get(cid,{})
            chain=h.get('lingke_chain') or {}
            fields.extend([float(h.get('collapse_max') or 0)/10,float(chain.get('count') or 0)/10,
                           (SEAT_INDEX[chain['character_id']]+1)/20 if chain.get('character_id') else 0,
                           float((view['policy_state']['sides'][s]['characters'].get(cid) or {}).get('insight_used_turn')==view['turn'])])
    for s in order:
        values=view['policy_state']['sides'][s].get('resistances') or {}
        fields.extend(float(values.get(attribute) or 0)/10 for attribute in ATTRIBUTES)
    for s in order:
        heroes=view['policy_state']['sides'][s]['characters']
        for cid in SEATS:
            values=(heroes.get(cid) or {}).get('resistances') or {}
            fields.extend(float(values.get(attribute) or 0)/10 for attribute in ATTRIBUTES)
    fields += [float(pending.get('kind')==kind) for kind in CHOICES]
    counts=np.zeros(len(CARD_IDS),dtype=np.float32)
    for item in choices.values():counts[CARD_INDEX[item['card_id']]]+=.5
    fields += counts.tolist()
    result=np.concatenate((x,np.asarray(fields,dtype=np.float32)))
    if result.shape!=(len(all_features()),) or not np.isfinite(result).all():raise ValueError('Invalid seven-team observation')
    return result,np.concatenate((c,extra),axis=1)
