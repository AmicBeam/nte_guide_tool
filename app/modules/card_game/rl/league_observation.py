"""Matching NumPy and tensor observations from explicit public mechanics only."""
from .league_schema import *


def encode_base(view, actions, *, schema=None, contract=None):
    import numpy as np
    from . import league_schema
    contract = contract or league_schema
    SCHEMA, SEATS = contract.SCHEMA, contract.SEATS
    CARD_IDS, CARD_INDEX = contract.CARD_IDS, contract.CARD_INDEX
    SEAT_INDEX = contract.SEAT_INDEX
    schema = SCHEMA if schema is None else schema
    from app.modules.card_game.content.duel_v2 import CARDS
    from .league_schema import ACT_END, ACT_ATTACK, ACT_PLAY, ACT_CHOOSE, ACT_ULTIMATE
    if schema != SCHEMA:
        raise ValueError('Unsupported observation schema')
    viewer = view['viewer_side']
    order = (viewer, 'b' if viewer == 'a' else 'a')
    public = view['policy_state']
    new = schema == SCHEMA
    side_fields, char_fields = (SIDE_FIELDS, CHAR_FIELDS) if new else (LEGACY_SIDE, LEGACY_CHAR)
    values = [view['turn']/30, float(viewer == view['first_side'])]
    if new:
        values.append(float(public['escalation_enabled']))
        pending = view.get('pending_choice') or {}
        resolving = view.get('resolving_card') or {}
        values.extend([{'mulligan':0,'playing':1,'choice':2,'finished':3}[view['phase']]/10,
                       {'inspect_top':1,'discard':2,'enemy_hand':3}.get(pending.get('kind'),-1)/10,
                       CARD_INDEX[resolving['card_id']]/len(OLD_CARDS) if pending and resolving else -1/len(OLD_CARDS),
                       len(pending.get('choices', []))/10])
    side_rows, char_rows = {}, {}
    for side in order:
        team = view['sides'][side]
        extra = public['sides'][side]
        zones = team.get('front_debuff') or {}
        side_rows[side] = {
            **{key: team.get(key, 0) for key in ('hp','shield','ap','turn_count','fatigue')},
            'front': SEAT_INDEX.get(team.get('front'), -1),
            'last_front': SEAT_INDEX.get(team.get('last_front'), -1),
            'normal_atk': bool(team.get('normal_attack_available')),
            'ultimate_ok': bool(team.get('ultimate_available')),
            'ultimate_remaining': team['ultimate_remaining'],
            'used_instant': bool((team.get('used') or {}).get('instant')),
            'extra_ap': extra.get('extra_ap') or 0,
            'extra_genesis': bool(extra.get('extra_genesis_pending')),
            'extra_genesis_actor': SEAT_INDEX.get(extra.get('extra_genesis_actor'), -1),
            'surplus': bool(extra.get('surplus')),
            'genesis_damage_bonus': int(extra.get('genesis_damage_bonus') or 0),
            'harmony_damage': bool(extra.get('harmony_damage')),
            'nanali_seed': extra.get('nanali_family') or 0,
            'nanali_seed_played': extra.get('nanali_family_played') or 0,
            'hand_n': team['hand_count'], 'deck_n': team['deck_count'],
            'discard_n': len(team['discard']), 'weave': bool(zones.get('weave')),
            'burn_left': (zones.get('burn') or {}).get('left', 0),
            'burn_infinite': bool((zones.get('burn') or {}).get('infinite')),
            'burn_by': (-1 if not zones.get('burn') else int(zones['burn'].get('by') != side)),
            'delay_end': (zones.get('delay') or {}).get('end', 0),
            'light_resistance': extra.get('light_resistance', 0),
            'dark_left': (zones.get('star') or {}).get('left',0),
            'dark_by': -1 if not zones.get('star') else int(zones['star'].get('by') != side),
        }
        pending = view.get('pending_choice') or {}
        if not new and pending.get('kind')=='inspect_top' and pending.get('side')==side:
            side_rows[side]['deck_n'] -= len(pending.get('choices',[]))
        indexed = {h['id']: h for h in team['characters']}
        if set(indexed) - set(SEATS):
            raise ValueError('Model only supports public characters')
        char_rows[side] = []
        for cid in SEATS:
            row = {key: 0 for key in char_fields}
            row.update(ch_shape=-1, ch_order=-1, ch_collapse_by=-1)
            h = indexed.get(cid)
            if h:
                meta = extra['characters'][cid]
                flags = meta['flags']
                for key, source in (('hp','hp'),('max_hp','max_hp'),('shield','shield'),
                    ('growth','growth'),('harmony','harmony'),('energy','energy'),
                    ('energy_max','energy_max'),('harmony_max','harmony_max'),
                    ('down','down_turns'),('awakened','awakened'),('ult_turns','ultimate_turns')):
                    row['ch_'+key] = h.get(source, 0)
                for key in ('atk_buff','next_bonus','next_shield','next_followup','pact',
                            'energy_next','allied_hurt','pending_atk','share_overflow','hp_floor','collapse_count'):
                    row['ch_'+key] = flags.get(key) or 0
                row.update(ch_present=1, ch_base_atk=meta['base_attack'],
                           ch_base_max=meta['base_max_hp'], ch_order=meta['order'],
                           ch_shape=CARD_INDEX[h['shape_id']] if h.get('shape_id') else -1,
                           ch_harmonized=meta['harmonized'])
                collapse = flags.get('collapse') or {}
                row.update(ch_jingu=h.get('jingu',0), ch_jingu_loan=meta.get('jingu_loan',False), ch_ultimate_used=meta.get('ultimate_used_turn')==view['turn'], ch_attacked=meta.get('attacked_this_turn',False), ch_light_resistance=meta.get('light_resistance',0))
                row['ch_collapse_until'] = collapse.get('until', 0)
                row['ch_collapse_by'] = -1 if not collapse else int(collapse['side'] != side)
                if not new:
                    # Legacy packer stores absolute side IDs (not consumed by its schema).
                    row['ch_collapse_by'] = -1 if not collapse else 'ab'.index(collapse['side'])
            char_rows[side].append(row)
    for key in side_fields:
        values.extend(float(side_rows[s][key])/10 for s in order)
    for key in char_fields:
        values.extend(float(h[key])/10 for s in order for h in char_rows[s])
    for cards in (view['sides'][viewer]['hand'], *(view['sides'][s]['discard'] for s in order)):
        counts = [0.0]*len(CARD_IDS)
        for card in cards:
            counts[CARD_INDEX[card['card_id']]] += .5
        values.extend(counts)
    if new:
        for side in (order[1],viewer):
            counts = [0.0]*len(CARD_IDS)
            for card in view['sides'][side]['hand']:
                if not card.get('hidden') and (side != viewer or card.get('revealed')):
                    counts[CARD_INDEX[card['card_id']]] += .5
            values.extend(counts)
    hand = {c['instance_id']: c for c in view['sides'][viewer]['hand']}
    choices = {c['id']:c['card'] for c in (view.get('pending_choice') or {}).get('choices', [])}
    candidates = []
    for a in actions:
        kind = {'end_turn':ACT_END,'attack':ACT_ATTACK,'ultimate':ACT_ULTIMATE,
                'play_card':ACT_PLAY,'choose':ACT_CHOOSE}[a['type']]
        actor = SEAT_INDEX[a['character_id']] if a['type'] in ('attack','ultimate') else -1
        card_kind, target_side, target_seat = -1, -1, -1
        card = {}
        revealed = False
        if a['type'] in ('play_card','choose'):
            item = hand[a['card_id']] if a['type']=='play_card' else choices[a['choice_id']]
            card_kind = CARD_INDEX[item['card_id']]
            card = CARDS[item['card_id']]
            revealed = bool(item.get('revealed'))
            if a['type']=='play_card':
                actor = SEAT_INDEX[item['character_id']]
        if a.get('target_id'):
            ts, cid = a['target_id'].split(':', 1)
            target_side = int(ts != viewer)
            target_seat = -1 if cid=='player' else SEAT_INDEX[cid]
        candidates.append([kind,actor,card_kind,target_side,target_seat,
                           int(card.get('cost') or 0),int(card.get('attack') or 0),
                           int(card.get('shield') or 0),int(card.get('hp') or 0),int(bool(card.get('instant')))])
        if new:candidates[-1].append(int(revealed))
    state = np.asarray(values, dtype=np.float32)
    cand = np.asarray(candidates, dtype=np.float32).reshape(-1, 11 if new else 10)
    if not np.isfinite(state).all() or not np.isfinite(cand).all():
        raise ValueError('Non-finite model observation')
    return state, cand



def encode_league(view, actions, *, contract=None):
    """Only sanitized observation and legal actions; no raw snapshot accepted."""
    import numpy as np
    from . import league_schema
    contract = contract or league_schema
    CARD_IDS, CARD_INDEX = contract.CARD_IDS, contract.CARD_INDEX
    CAND_DIM = contract.CAND_DIM
    own=view['sides'][view['viewer_side']]['hand']
    hand={c['instance_id']:c for c in own}
    playing=[{'type':'end_turn'} if a['type']=='mulligan' else a for a in actions]
    state,base=encode_base(view,playing,contract=contract)
    extra=np.zeros((len(actions),CAND_DIM-11),dtype=np.float32)
    marks=np.zeros((3,len(CARD_IDS)),dtype=np.float32)
    for card in own:
        if 'jingu_mark' in card:marks[int(card['jingu_mark']['delta'])+1,CARD_INDEX[card['card_id']]]+=.5
    for i,a in enumerate(actions):
        if a['type']=='mulligan':
            base[i]=[ACT_MULLIGAN,-1,-1,-1,-1,0,0,0,0,0,0]
            for identifier in a['card_ids']:extra[i,CARD_INDEX[hand[identifier]['card_id']]]+=.5
        item=hand.get(a.get('card_id'),{})
        opt=a.get('option_id')
        if opt is not None:
            if opt not in OPTION_IDS:raise ValueError('Unknown play option '+opt)
            extra[i,len(CARD_IDS)+OPTION_IDS.index(opt)]=1
        k=len(CARD_IDS)+len(OPTION_IDS)
        mark=item.get('jingu_mark')
        extra[i,k:k+4]=[float(mark is not None),0 if mark is None else mark['delta'],float(item.get('attack_mode')=='set'),float(item.get('instant',False))]
    state=np.concatenate((state,marks.reshape(-1)))
    assert state.shape==(len(contract.feature_names()),)
    return state,np.concatenate((base,extra),axis=1)
