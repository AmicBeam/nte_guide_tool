"""Front occupants which disappear on removal rather than joining the bench."""
from .entities import CharacterEntity, allocate_entity_id
from .state import hero, log, move_out


def summon_front(state, side, *, name, attack, hp, portrait='', avatar=''):
    team = state['sides'][side]
    if team['front']:
        move_out(state, side, team['front'])
    eid = allocate_entity_id(state)
    cid = 'summon_' + eid
    team['characters'][cid] = CharacterEntity(dict(
        id=cid, name=name, summoned=True, attribute='', faction='', portrait=portrait, avatar=avatar,
        base_attack=attack, base_max_hp=hp, hp=hp, max_hp=hp, shield=0, harmony=0, energy=0,
        awakened=False, ultimate_turns=0, growth=0, family=0, gaze=0, dream=0, shape=None,
        down_turns=0, flags={}, passive='倒地或离开战斗区时消失；回合开始时不返回备战区。',
        awakened_passive=''), entity_id=eid, side=side)
    team['order'].append(cid)
    team['front'] = cid
    team['last_front'] = None
    log(state, f'在战斗区召唤{name}。', 'enter', side=side, actor=f'{side}:{cid}', target=f'{side}:{cid}')
    return cid


def dismiss(state, side, cid):
    team = state['sides'][side]
    h = hero(state, side, cid)
    h.update(hp=0, shield=0, disappeared=True)
    if team['front'] == cid:
        team['front'] = None
    team['last_front'] = None
    team['order'] = [c for c in team['order'] if c != cid]
    foe = 'b' if side == 'a' else 'a'
    if state['sides'][foe].get('seal_normal_attack') == cid:
        state['sides'][foe].pop('seal_normal_attack', None)
    # Keep the identity until the current simultaneous damage operation finishes.
    log(state, f"{h['name']}消失。", 'move', side=side, actor=f'{side}:{cid}', target=f'{side}:{cid}')
