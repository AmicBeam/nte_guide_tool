"""Pure immutable public commands and sequential turn/choice orchestration."""
from copy import deepcopy
from itertools import combinations
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS, STARTER_DECK, validate_deck
from app.modules.card_game.content.duel_v2.effects import EFFECTS, TARGET_POLICIES
from .context import EffectContext
from app.modules.card_game.engine.events import GameEvent
from .state import (ORDER, SIDES, add_hand, alive, card_instance, damage, discard,
                    draw, energy, finished, hero, log, name, new_character,
                    other, shuffle)


def new_game(seed=0, decks=None, first_side=None, skip_mulligan=False):
    if type(seed) is not int or first_side not in (None, 'a', 'b'):
        raise ValueError('Invalid seed or first side')
    if type(skip_mulligan) is not bool:
        raise ValueError('skip_mulligan must be boolean')
    if decks is not None and (not isinstance(decks, dict) or set(decks) != set(SIDES)):
        raise ValueError('Both decks are required')
    selected = {s: validate_deck(deepcopy(decks[s] if decks else STARTER_DECK)) for s in SIDES}
    first_side = first_side or SIDES[Random(seed).randrange(2)]
    from .presentation import capture_public_board
    state = {'rules_version': 'duel_v2', 'design_version': 'V2.4', 'version': 0,
             'phase': 'mulligan', 'active_side': first_side, 'first_side': first_side,
             'turn': 0, 'winner': None, 'reason': None, 'rng': seed, 'next_instance': 0,
             'event_seq': 0, 'events': [], 'pending_choice': None, 'operation': None,
             'mulligan_done': [], 'current_action_id': 'setup', 'current_action_version': 0,
             'combat_group': 0, 'sides': {}}
    for side in SIDES:
        state['sides'][side] = {
            'name': name(state, side), 'hp': 30, 'shield': 0 if side == first_side else 5,
            'ap': 0, 'normal_attack_available': False, 'front': None, 'last_front': None,
            'characters': {cid: new_character(cid) for cid in ORDER},
            'hand': [], 'deck': [], 'discard': [], 'removed': [], 'records': [],
            'turn_count': 0, 'fatigue': 0, 'used': {}, 'harmonized': {}, 'extra_flower': False,
        }
        team = state['sides'][side]
        team['deck'] = [card_instance(state, cid, side) for cid in selected[side]['card_ids']]
        shuffle(state, team['deck'])
        draw(state, side, 5)
    if skip_mulligan:
        state['mulligan_done'] = list(SIDES)
        begin_turn(state, first_side)
    state['_public_board'] = capture_public_board(state)
    return state


def acting_side(state):
    if finished(state):
        return None
    if state['phase'] == 'mulligan':
        return next((s for s in SIDES if s not in state['mulligan_done']), None)
    if state['phase'] == 'choice':
        return state['pending_choice']['side']
    return state['active_side']


def targets(state, side, card):
    policy = TARGET_POLICIES.get(card['card_id'])
    team, enemy = state['sides'][side], state['sides'][other(side)]
    if policy == 'other_living_ally':
        return [{'target_id': f'{side}:{cid}'} for cid in ORDER
                if cid != card['character_id'] and alive(state, side, cid)]
    if policy == 'enemy_front':
        return [{}] if enemy['front'] else []
    if policy == 'enemy_hand':
        return [{}] if enemy['hand'] else []
    if policy == 'own_nonterminal_tactic_discard':
        return [{'target_id': c['instance_id']} for c in team['discard']
                if c['character_id'] == card['character_id'] and c['type'] == 'tactic'
                and not c['terminal'] and not c.get('copy')]
    return [{}]


def legal_actions(state, side=None):
    side = side or acting_side(state)
    if side not in SIDES or finished(state):
        return []
    result = [{'type': 'concede'}]
    phase = state['phase']
    if phase == 'mulligan':
        if side in state['mulligan_done']:
            return result
        ids = [c['instance_id'] for c in state['sides'][side]['hand']]
        return result + [{'type': 'mulligan', 'card_ids': list(group)}
                         for size in range(min(3, len(ids)) + 1) for group in combinations(ids, size)]
    if side != acting_side(state):
        return result
    if phase == 'choice':
        return result + [{'type': 'choose', 'choice_id': c['instance_id']}
                         for c in state['pending_choice']['cards']]
    if phase != 'playing':
        return result
    team = state['sides'][side]
    result.append({'type': 'end_turn'})
    for cid in ORDER:
        h = hero(state, side, cid)
        if h['hp'] <= 0:
            continue
        if team['ap'] >= 1 and team['normal_attack_available']:
            result.append({'type': 'attack', 'character_id': cid})
        if team['ap'] >= 1 and h['energy'] == 4 and not h['awakened']:
            result.append({'type': 'awaken', 'character_id': cid})
    for card in team['hand']:
        if alive(state, side, card['character_id']):
            if team['ap'] >= card['cost']:
                result.extend({'type': 'play_card', 'card_id': card['instance_id'], **target}
                              for target in targets(state, side, card))
        elif team['ap'] >= 1 and not team['used'].get('cycle'):
            result.append({'type': 'cycle', 'card_id': card['instance_id']})
    return result


def begin_turn(state, side):
    state['phase'], state['active_side'] = 'playing', side
    state['turn'] += 1
    team = state['sides'][side]
    team['turn_count'] += 1
    team['used'] = {}
    state['sides'][other(side)]['used'].pop('Z03', None)
    for cid in ORDER:
        h = hero(state, side, cid)
        h['shield'] = 0
        if h['down_turns']:
            h['down_turns'] -= 1
            if not h['down_turns']:
                h['hp'] = h['max_hp']
                log(state, f"{h['name']}在备战区恢复。", 'revive', side=side, source=cid,
                    actor=f'{side}:{cid}', target=f'{side}:{cid}',
                    before={'hp': 0, 'down_turns': 1}, after={'hp': h['hp'], 'down_turns': 0})
    zero = hero(state, side, 'zero')
    if zero['hp'] > 0 and zero['awakened'] and team['front'] != 'zero':
        zero['harmony'] = min(2, zero['harmony'] + 1)
    log(state, f"{name(state, side)}的第 {team['turn_count']} 个回合开始。", 'turn', side=side,
        actor=f'{side}:player')
    draw(state, side)
    if not finished(state):
        team['ap'] = 1 if state['turn'] == 1 else 2
        team['normal_attack_available'] = True


def end_turn(state, side):
    team = state['sides'][side]
    for cid in ORDER:
        h = hero(state, side, cid)
        if h['hp'] > 0 and h['flags'].get('burn'):
            h['flags']['burn'] -= 1
            damage(state, side, cid, 1, kind='burn')
        if finished(state):
            return
        if h['hp'] > 0 and h['flags'].get('star', float('inf')) <= team['turn_count']:
            h['flags'].pop('star', None)
            damage(state, side, cid, 3, kind='star')
        if finished(state):
            return
        if h['flags'].get('slow', {}).get('end', float('inf')) <= team['turn_count']:
            h['flags'].pop('slow', None)
        h['flags'].pop('return_after', None)
    for card in list(team['hand']):
        if card.get('copy') and card['expires_turn'] <= team['turn_count']:
            team['hand'].remove(card)
            discard(state, side, card)
    team.update(ap=0, normal_attack_available=False, extra_flower=False)
    begin_turn(state, other(side))


def operation(state, side, cid, card=None, target_id=None):
    xun = hero(state, side, 'xun')
    op = {'side': side, 'actor': cid, 'card': card, 'target_id': target_id,
          'participants': [], 'harmony': None, 'first_bonus': 0, 'redeem_bonus': 0,
          'record_candidate': xun['hp'] > 0 and state['sides'][side]['front'] != 'xun'}
    state['operation'] = op
    return EffectContext(state, side, cid, op)


def finish_operation(state):
    op = state['operation']
    if not op:
        return
    if finished(state):
        state['pending_choice'] = None
        return
    side, cid, card = op['side'], op['actor'], op['card']
    team = state['sides'][side]
    c = EffectContext(state, side, cid, op)
    if card:
        if card.get('copy'):
            c.on_shape(GameEvent.V2_COPY_FINISHED)
        if finished(state):
            return
        discard(state, side, card)
        xun = hero(state, side, 'xun')
        if (op['record_candidate'] and xun['hp'] > 0 and team['front'] != 'xun'
                and cid != 'xun' and not card.get('copy') and card['cost'] == 1
                and card['type'] in ('battle', 'tactic') and not card['terminal']
                and not team['used'].get('record')):
            team['used']['record'] = True
            record = {key: card[key] for key in ('card_id', 'name', 'type', 'character_id')}
            records = [r for r in team['records'] if r['character_id'] != cid]
            team['records'] = (records + [record])[-(2 if xun['awakened'] else 1):]
            from .presentation import public_card
            log(state, f"浔记录了{hero(state, side, cid)['name']}的「{card['name']}」。",
                'record', side=side, source='xun', actor=f'{side}:xun',
                target=f'{side}:{cid}', card=public_card(card))
    for ps, pc in op['participants']:
        if alive(state, ps, pc):
            h = hero(state, ps, pc)
            old_harmony, old_energy = h['harmony'], h['energy']
            h['harmony'] = min(2, h['harmony'] + (2 if pc == 'zero' else 1))
            energy(state, ps, pc)
            dh, de = h['harmony'] - old_harmony, h['energy'] - old_energy
            log(state, f"{h['name']}战斗积累：自己的环合 +{dh}、能量 +{de}。", 'resource',
                side=ps, source=pc, actor=f'{ps}:{pc}', target=f'{ps}:{pc}',
                harmony_delta=dh, energy_delta=de,
                before={'harmony': old_harmony, 'energy': old_energy},
                after={'harmony': h['harmony'], 'energy': h['energy']})
    state['phase'] = 'playing'
    state['operation'], state['pending_choice'] = None, None


def choose(state, side, choice_id):
    pending = state['pending_choice']
    selected = next(c for c in pending['cards'] if c['instance_id'] == choice_id)
    team = state['sides'][side]
    kind = pending['kind']
    if kind == 'discard':
        team['hand'].remove(selected)
        discard(state, side, selected)
    elif kind == 'inspect_top':
        add_hand(state, side, selected)
        team['deck'].extend(c for c in pending['cards'] if c['instance_id'] != choice_id)
    elif kind == 'enemy_hand':
        enemy = other(side)
        state['sides'][enemy]['hand'].remove(selected)
        from .presentation import public_card
        log(state, f"「{selected['name']}」被选中并公开。", 'reveal', side=side,
            actor=f'{side}:player', card=public_card(selected),
            target=f'{enemy}:player')
        discard(state, enemy, selected, bottom=True)
    finish_operation(state)


def apply_action(state, side, action):
    if side not in SIDES or not isinstance(action, dict):
        raise ValueError('无效的操作格式。')
    candidate = deepcopy(action)
    if candidate.get('type') == 'mulligan' and isinstance(candidate.get('card_ids'), list):
        ids = candidate['card_ids']
        if any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('换牌选择无效。')
        order = {c['instance_id']: i for i, c in enumerate(state['sides'][side]['hand'])}
        candidate['card_ids'] = sorted(ids, key=lambda x: order.get(x, 999))
    if candidate not in legal_actions(state, side):
        raise ValueError('当前不能执行此操作：请检查回合、行动力、角色状态和目标。')
    from .presentation import bump_public_patch, capture_public_board
    result = deepcopy(state)
    result['_public_board'] = capture_public_board(result)
    result['version'] += 1
    result['current_action_version'] = result['version']
    result['current_action_id'] = f"v{result['version']}"
    result['combat_group'] = 0
    team = result['sides'][side]
    kind = candidate['type']
    if kind == 'concede':
        result.update(phase='finished', winner=other(side), reason='concede', pending_choice=None)
        log(result, f"{name(result, side)}认输。", 'finish', side=side, actor=f'{side}:player')
    elif kind == 'mulligan':
        removed = [c for c in team['hand'] if c['instance_id'] in candidate['card_ids']]
        team['hand'] = [c for c in team['hand'] if c['instance_id'] not in candidate['card_ids']]
        draw(result, side, len(removed))
        team['deck'].extend(removed)
        shuffle(result, team['deck'])
        result['mulligan_done'].append(side)
        if len(result['mulligan_done']) == 2:
            begin_turn(result, result['first_side'])
    elif kind == 'choose':
        choose(result, side, candidate['choice_id'])
    elif kind == 'end_turn':
        end_turn(result, side)
    elif kind == 'awaken':
        team['ap'] -= 1
        h = hero(result, side, candidate['character_id'])
        h.update(energy=0, awakened=True)
        log(result, f"{h['name']}消耗自己的满槽能量与 1 行动力，主动觉醒。",
            'awaken', side=side, source=h['id'], actor=f'{side}:{h["id"]}',
            target=f'{side}:{h["id"]}',
            before={'energy': 4, 'awakened': False, 'ap': team['ap'] + 1},
            after={'energy': 0, 'awakened': True, 'ap': team['ap']})
    elif kind == 'attack':
        team['ap'] -= 1
        team['normal_attack_available'] = False
        operation(result, side, candidate['character_id']).sortie()
        finish_operation(result)
    else:
        card = next(c for c in team['hand'] if c['instance_id'] == candidate['card_id'])
        team['hand'].remove(card)
        if kind == 'cycle':
            team['ap'] -= 1
            team['used']['cycle'] = True
            discard(result, side, card)
            draw(result, side)
        else:
            team['ap'] -= card['cost']
            from .presentation import public_card
            log(result, f"{hero(result, side, card['character_id'])['name']}使用「{card['name']}」。",
                'play', side=side, source=card['character_id'],
                actor=f"{side}:{card['character_id']}", card=public_card(card))
            c = operation(result, side, card['character_id'], card, candidate.get('target_id'))
            EFFECTS[card['effect_id']](c)
            if result['phase'] != 'choice':
                finish_operation(result)
    bump_public_patch(result)
    return result
