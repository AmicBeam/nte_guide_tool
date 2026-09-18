"""Eight-character Python-rule league. Independent of the six-seat CUDA ABI."""
from hashlib import sha256
from pathlib import Path

SCHEMA = 'official_public_v5'
PRESETS = ('starter', 'weave-rush', 'quick-rush')
SEATS = ('nanali', 'iloy', 'zero', 'jiuyuan', 'bohe', 'baicang', 'xiaozhi', 'haiyue')
SEAT_INDEX = {v:i for i,v in enumerate(SEATS)}
# Keep all old numeric identities; append new cards instead of inserting them.
OLD_CARDS = tuple([f'N{i:02}' for i in range(1,9)] + ['NF01'] + [f'{p}{i:02}' for p in 'ZJYMB' for i in range(1,9)])
CARD_IDS = OLD_CARDS + tuple(f'{p}{i:02}' for p in 'QU' for i in range(1,9))
CARD_INDEX = {v:i for i,v in enumerate(CARD_IDS)}
N_SEATS = len(SEATS)
LEGACY_SIDE = ('hp','shield','ap','front','last_front','turn_count','normal_atk','ultimate_ok','used_instant','extra_genesis','extra_genesis_actor','nanali_seed','nanali_seed_played','hand_n','deck_n','discard_n','weave')
LEGACY_CHAR = ('ch_present','ch_hp','ch_max_hp','ch_base_atk','ch_growth','ch_atk_buff','ch_shield','ch_harmony','ch_energy','ch_down','ch_awakened','ch_ult_turns','ch_shape','ch_next_bonus','ch_next_shield','ch_next_followup','ch_pact','ch_harmonized','ch_collapse_count','ch_collapse_until','ch_allied_hurt','ch_energy_next')
OLD_SIDE = (*LEGACY_SIDE,'fatigue','extra_ap','surplus','ultimate_remaining','burn_left','burn_infinite','delay_end','harmony_damage','burn_by','genesis_damage_bonus')
OLD_CHAR = (*LEGACY_CHAR,'ch_base_max','ch_energy_max','ch_harmony_max','ch_pending_atk','ch_share_overflow','ch_hp_floor','ch_order','ch_collapse_by')
SIDE_FIELDS = (*OLD_SIDE,'light_resistance','dark_left','dark_by')
CHAR_FIELDS = (*OLD_CHAR,'ch_jingu','ch_jingu_loan','ch_ultimate_used','ch_attacked','ch_light_resistance')
ACT_END, ACT_ATTACK, ACT_PLAY, ACT_ULTIMATE, ACT_MULLIGAN, ACT_CHOOSE = range(6)
OPTION_IDS = ('base','boost','release_a','release_b','release_ba')
CAND_DIM = 11 + len(CARD_IDS) + len(OPTION_IDS) + 4

def feature_names(*, old=False):
    seats=SEATS[:6] if old else SEATS;cards=OLD_CARDS if old else CARD_IDS
    side=OLD_SIDE if old else SIDE_FIELDS;chars=OLD_CHAR if old else CHAR_FIELDS
    names=['turn','first','escalation','phase','choice_kind','resolving','choice_count']
    names += [f'{k}:{s}' for k in side for s in (0,1)]
    names += [f'{k}:{s}:{c}' for k in chars for s in (0,1) for c in seats]
    names += [f'cards:{z}:{c}' for z in range(5) for c in cards]
    if not old:names += [f'marks:{d}:{c}' for d in (-1,0,1) for c in cards]
    return names

def rule_hash():
    root=Path(__file__).resolve().parents[1];digest=sha256(SCHEMA.encode())
    paths=[* (root/'engine/duel_v2').rglob('*.py'),* (root/'content/duel_v2').rglob('*.py'),root/'content/duel_v2/catalog.json',Path(__file__),root/'rl/league_observation.py']
    for p in sorted(set(paths)):
        digest.update(p.relative_to(root).as_posix().encode());digest.update(p.read_bytes())
    return digest.hexdigest()

def deck(key):
    from copy import deepcopy
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS,validate_deck
    if key not in PRESETS:raise ValueError('Unknown league strategy')
    return validate_deck(deepcopy(next(d for d in STARTER_DECKS if d['id']==key)))

def build_hash(build):
    import json
    from app.modules.card_game.content.duel_v2 import validate_deck
    b=validate_deck(build)
    return sha256(json.dumps({k:b[k] for k in ('character_ids','card_ids')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
