"""Pure immutable public commands and sequential turn/choice orchestration."""
from copy import deepcopy
from itertools import combinations
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, validate_deck

from app.modules.card_game.content.duel_v2.registry import KITS, fire
from .context import EffectContext
from app.modules.card_game.engine.events import GameEvent
from .state import (SIDES, add_hand, alive, card_instance, clear_harmony_source, collapsed, damage, discard,
                    draw, energy, finished, front_debuff, hero, knockdowns, log, move_out, name,
                    new_character, other, shield, shuffle, team_order)

from .escalation import begin_escalation, clamp_resources, energy_cap, harmony_cap, ultimate_remaining


def new_game(seed=0, decks=None, first_side=None, skip_mulligan=False, scenario=None, escalation=True):
    if scenario:
        from .tutorial import new_tutorial_game
        return new_tutorial_game(scenario, seed=seed)
    if type(seed) is not int or first_side not in (None, 'a', 'b'):
        raise ValueError('Invalid seed or first side')
    if type(escalation) is not bool:
        raise ValueError('escalation must be boolean')
    if type(skip_mulligan) is not bool:
        raise ValueError('skip_mulligan must be boolean')
    if decks is not None and (not isinstance(decks, dict) or set(decks) != set(SIDES)):
        raise ValueError('Both decks are required')
    selected = {s: validate_deck(deepcopy(decks[s] if decks else STARTER_DECK)) for s in SIDES}
    first_side = first_side or SIDES[Random(seed).randrange(2)]
    from .presentation import capture_public_board
    state = {'rules_version': 'duel_v2', 'design_version': 'V2.5', 'version': 0,
             'phase': 'mulligan', 'active_side': first_side, 'first_side': first_side,
             'turn': 0, 'winner': None, 'reason': None, 'rng': seed, 'next_instance': 0,
             'event_seq': 0, 'events': [], 'pending_choice': None, 'operation': None,
             'mulligan_done': [], 'current_action_id': 'setup', 'current_action_version': 0,
             'combat_group': 0, 'sides': {}, 'escalation_enabled': bool(escalation)}
    for side in SIDES:
        order = list(selected[side]['character_ids'])
        state['sides'][side] = {
            'name': name(state, side), 'hp': 30, 'shield': 0 if side == first_side else 5,
            'ap': 0, 'normal_attack_available': False, 'ultimate_available': False,
            'front': None, 'last_front': None,
            'order': order,
            'characters': {cid: new_character(cid) for cid in order},
            'hand': [], 'deck': [], 'discard': [], 'removed': [], 'records': [],
            'turn_count': 0, 'fatigue': 0, 'used': {}, 'harmonized': {},
            'extra_flower': False, 'extra_delay': False, 'front_debuff': {},
        }
        team = state['sides'][side]
        team['deck'] = [card_instance(state, cid, side) for cid in selected[side]['card_ids']]
        shuffle(state, team['deck'])
    second = other(first_side)
    log(state, f"{name(state, first_side)}获得先手。" + (f"{name(state, second)}因后手获得 5 点开局护盾。" if second else ''),
        'initiative', side=first_side, actor=f'{first_side}:player', first_side=first_side)
    for side in SIDES:
        draw(state, side, 5)
        for cid in state['sides'][side]['order']:
            fire(EffectContext(state, side, cid, {'side': side}), 'on_game_start', actor_only=True)
    if skip_mulligan:
        state['mulligan_done'] = list(SIDES)
        begin_turn(state, first_side)
    state['_public_board'] = capture_public_board(state)
    return state


def kit_makes_form_instant(state, side, card):
    if card.get('type') != 'form':
        return False
    team = state['sides'][side]
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h['hp'] <= 0:
            continue
        kit = KITS.get(cid)
        if h.get('awakened') and getattr(kit, 'ultimate_forms_instant', False):
            return True
    owner = card.get('character_id')
    kit = KITS.get(owner)
    if getattr(kit, 'shaped_forms_instant', False):
        holder = team['characters'].get(owner)
        if holder and holder['hp'] > 0 and holder.get('shape'):
            return True
    return False


def kit_makes_card_instant(state, side, card):
    kit = KITS.get(card.get('character_id'))
    fn = getattr(kit, 'makes_instant', None)
    return bool(fn and fn(state, side, card))


def card_plays_instant(state, side, card):
    return bool(card.get('instant')) or kit_makes_form_instant(state, side, card) or kit_makes_card_instant(state, side, card)


def card_is_free_instant(state, side, card):
    team = state['sides'][side]
    return bool(card_plays_instant(state, side, card) and not team['used'].get('instant'))


def can_afford_card(state, side, card):
    cost = int((card or {}).get('cost') or 0)
    return card_is_free_instant(state, side, card) or int(state['sides'][side].get('ap') or 0) >= cost


def resolve_play(state, side, card, target_id=None, *, pay=True):
    from app.modules.card_game.content.duel_v2.characters.common import scale_battle_frame
    from .presentation import public_card
    team = state['sides'][side]
    form_bonus = kit_makes_form_instant(state, side, card)
    if pay:
        if card_is_free_instant(state, side, card):
            team['used']['instant'] = True
        else:
            team['ap'] -= int(card.get('cost') or 0)
    log(state, f"{hero(state, side, card['character_id'])['name']}使用「{card['name']}」。",
        'play', side=side, source=card['character_id'],
        actor=f"{side}:{card['character_id']}", card=public_card(card))
    c = operation(state, side, card['character_id'], card, target_id)
    if card.get('type') == 'battle' and alive(state, side, card['character_id']):
        attack = int(card.get('attack') or 0)
        value = int(card.get('shield') or 0)
        attack, value = scale_battle_frame(c, attack, value)
        if value:
            shield(state, side, card['character_id'], value)
        if attack:
            if state.get('_response_lock'):
                flags = hero(state, side, card['character_id']).setdefault('flags', {})
                flags['counter_bonus'] = int(flags.get('counter_bonus') or 0) + attack
            else:
                c.operation['card_bonus'] = attack
    from app.modules.card_game.content.duel_v2.effects import EFFECTS
    EFFECTS[card['effect_id']](c)
    if form_bonus and not finished(state) and alive(state, side, card['character_id']):
        draw(state, side, 1)
    if state['phase'] != 'choice':
        finish_operation(state)


def trigger_responses(state, defending_side, kind, target_id=None):
    """Auto-play the first affordable response matching this timing. Kinds come from the card."""
    if state.get('_response_lock') or finished(state):
        return
    team = state['sides'][defending_side]
    for card in list(team['hand']):
        if card.get('response') != kind:
            continue
        owner = card['character_id']
        if not alive(state, defending_side, owner) or collapsed(hero(state, defending_side, owner)):
            continue
        if kind == 'ally_attacked' and (not target_id or target_id == owner):
            continue
        if kind == 'self_attacked' and target_id != owner:
            continue
        if kind == 'self_lethal' and target_id != owner:
            continue
        if not can_afford_card(state, defending_side, card):
            continue
        team['hand'].remove(card)
        parent_op = state.get('operation')
        state['_response_lock'] = True
        try:
            resolve_play(state, defending_side, card, pay=True)
        finally:
            state['_response_lock'] = False
            state['operation'] = parent_op
        return


def acting_side(state):
    if finished(state):
        return None
    if state['phase'] == 'mulligan':
        return next((s for s in SIDES if s not in state['mulligan_done']), None)
    if state['phase'] == 'choice':
        return state['pending_choice']['side']
    return state['active_side']


def targets(state, side, card):
    from app.modules.card_game.content.duel_v2.effects import TARGET_POLICIES
    policy = TARGET_POLICIES.get(card['card_id'])
    team, enemy = state['sides'][side], state['sides'][other(side)]
    if policy == 'living_ally':
        return [{'target_id': f'{side}:{cid}'} for cid in team_order(state, side)
                if alive(state, side, cid)]
    if policy == 'other_living_ally':
        return [{'target_id': f'{side}:{cid}'} for cid in team_order(state, side)
                if cid != card['character_id'] and alive(state, side, cid)]
    if policy == 'downed_ally':
        return [{'target_id': f'{side}:{cid}'} for cid in team_order(state, side)
                if hero(state, side, cid)['down_turns'] > 0]
    if policy == 'downed_enemy':
        foe = other(side)
        return [{'target_id': f'{foe}:{cid}'} for cid in team_order(state, foe)
                if hero(state, foe, cid)['down_turns'] > 0]
    if policy == 'living_enemy':
        foe = other(side)
        return [{'target_id': f'{foe}:{cid}'} for cid in team_order(state, foe)
                if alive(state, foe, cid)]
    if policy == 'living_role':
        result = [{'target_id': f'{side}:player'}]
        result.extend({'target_id': f'{side}:{cid}'} for cid in team_order(state, side)
                      if alive(state, side, cid))
        return result
    if policy == 'injured_role':
        result = []
        if team['hp'] < 30:
            result.append({'target_id': f'{side}:player'})
        for cid in team_order(state, side):
            h = hero(state, side, cid)
            if alive(state, side, cid) and h['hp'] < h['max_hp']:
                result.append({'target_id': f'{side}:{cid}'})
        return result
    if policy == 'enemy_front':
        return [{}] if enemy['front'] else []
    if policy == 'ally_front':
        return [{}] if team['front'] else []
    if policy == 'enemy_hand':
        return [{}] if enemy['hand'] else []
    if policy == 'own_tactic_discard':
        return [{'target_id': c['instance_id']} for c in team['discard']
                if c['character_id'] == card['character_id'] and c['type'] == 'tactic'
                and not c.get('copy')]
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
        result = result + [{'type': 'mulligan', 'card_ids': list(group)}
                           for size in range(min(3, len(ids)) + 1) for group in combinations(ids, size)]
        from .tutorial import filter_legal_actions, is_tutorial
        if is_tutorial(state):
            return filter_legal_actions(state, side, result)
        return result
    if side != acting_side(state):
        return result
    if phase == 'choice':
        return result + [{'type': 'choose', 'choice_id': c['instance_id']}
                         for c in state['pending_choice']['cards']]
    if phase != 'playing':
        return result
    team = state['sides'][side]
    result.append({'type': 'end_turn'})
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h['hp'] <= 0:
            continue
        if team['ap'] >= 1 and team['normal_attack_available'] and not collapsed(h):
            result.append({'type': 'attack', 'character_id': cid})
        if ultimate_remaining(state, side) and h['energy'] >= energy_cap(state, h):
            result.append({'type': 'ultimate', 'character_id': cid})
    for card in team['hand']:
        owner = card['character_id']
        if not alive(state, side, owner) and not card.get('playable_downed'):
            continue
        if collapsed(hero(state, side, owner)):
            continue
        if card.get('require') == 'harmony_damage' and not team.get('harmony_damage'):
            continue
        if card.get('require') == 'surplus' and not team.get('surplus'):
            continue
        if card.get('require') == 'own_burn':
            burn = front_debuff(state, other(side)).get('burn') or {}
            if burn.get('by') != side:
                continue
        if can_afford_card(state, side, card):
            result.extend({'type': 'play_card', 'card_id': card['instance_id'], **target}
                          for target in targets(state, side, card))
    from .tutorial import filter_legal_actions, is_tutorial
    if is_tutorial(state):
        return filter_legal_actions(state, side, result)
    return result


def ultimate_spec(cid):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    spec = (lookup_character(cid) or {}).get('ultimate') or {}
    kind = spec.get('kind') or 'status'
    if kind not in ('status', 'instant'):
        kind = 'status'
    turns = int(spec.get('turns') or 2)
    return {'kind': kind, 'turns': max(1, turns)}


def start_ultimate(state, side, cid, *, spent_ap=False):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    h = hero(state, side, cid)
    spec = ultimate_spec(cid)
    team = state['sides'][side]
    before = {'energy': h['energy'], 'awakened': h.get('awakened', False),
              'ultimate_turns': h.get('ultimate_turns', 0), 'ap': team['ap'] + (1 if spent_ap else 0)}
    h['energy'] = 0
    if spec['kind'] == 'instant':
        h['awakened'] = False
        h['ultimate_turns'] = 0
        text = f"{h['name']}消耗 {energy_cap(state, h)} 点能量，发动终结。"
    else:
        h['awakened'] = True
        h['ultimate_turns'] = spec['turns']
        text = f"{h['name']}消耗 {energy_cap(state, h)} 点能量，发动终结（持续 {spec['turns']} 个己方回合）。"
    log(state, text, 'ultimate', side=side, source=h['id'], actor=f'{side}:{h["id"]}',
        target=f'{side}:{h["id"]}',
        description=(lookup_character(cid) or {}).get('awakened_passive') or '',
        before=before,
        after={'energy': h['energy'], 'awakened': h['awakened'], 'ultimate_turns': h['ultimate_turns'], 'ap': team['ap']})
    fire(EffectContext(state, side, cid, {'side': side}), 'on_ultimate', actor_only=True)
    if getattr(KITS.get(cid), 'ultimate_sortie', False) and alive(state, side, cid):
        operation(state, side, cid).sortie()
        if state['phase'] != 'choice':
            finish_operation(state)


def end_ultimate(state, side, cid):
    h = hero(state, side, cid)
    if not h.get('awakened') and not h.get('ultimate_turns'):
        return
    h['awakened'] = False
    h['ultimate_turns'] = 0
    fire(EffectContext(state, side, cid, {'side': side}), 'on_ultimate_end',
         actor_only=True, include_down=True)
    if finished(state):
        return
    log(state, f"{h['name']}的终结结束。", 'ultimate', side=side, source=h['id'],
        actor=f'{side}:{h["id"]}', target=f'{side}:{h["id"]}',
        present=False,
        after={'awakened': False, 'ultimate_turns': 0})


def tick_ultimates(state, side):
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h.get('ultimate_turns', 0) <= 0:
            continue
        h['ultimate_turns'] -= 1
        if h['ultimate_turns'] <= 0:
            end_ultimate(state, side, cid)
        if finished(state):
            return


def begin_turn(state, side):
    state['phase'], state['active_side'] = 'playing', side
    state['turn'] += 1
    team = state['sides'][side]
    team['turn_count'] += 1
    team['used'] = {}
    team['harmony_damage'] = False
    team['surplus'] = False
    clamp_resources(state)
    # Announce the turn before any start-of-turn board changes are presented.
    log(state, f"{name(state, side)}的第 {team['turn_count']} 个回合开始。", 'turn', side=side,
        actor=f'{side}:player')
    for ps in SIDES:
        for cid in team_order(state, ps):
            h = hero(state, ps, cid)
            col = (h.get('flags') or {}).get('collapse')
            if col and col.get('side') == side and col.get('until', 0) <= team['turn_count']:
                h['flags'].pop('collapse', None)
                log(state, f"{h['name']}的倾陷结束。", 'collapse', side=ps, source=cid,
                    actor=f'{ps}:{cid}', target=f'{ps}:{cid}')
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        h['shield'] = 0
        if h['down_turns']:
            clear_harmony_source(state, side, cid)
            h['down_turns'] -= 1
            if not h['down_turns']:
                h['hp'] = h['max_hp']
                log(state, f"{h['name']}在备战区恢复。", 'revive', side=side, source=cid,
                    actor=f'{side}:{cid}', target=f'{side}:{cid}',
                    before={'hp': 0, 'down_turns': 1}, after={'hp': h['hp'], 'down_turns': 0})
    begin_escalation(state, side)
    carrier = team_order(state, side)[0]
    fire(EffectContext(state, side, carrier, {'side': side}), 'on_turn_begin')
    if finished(state):
        return
    if team.pop('extra_genesis_pending', False):
        from .combat import resolve_genesis
        actor = team.pop('extra_genesis_actor', None)
        if actor and alive(state, side, actor):
            EffectContext(state, side, actor, {'side': side}).history(
                '回合开始，结算「错误的门」留下的追加创生。')
            resolve_genesis(EffectContext(state, side, actor, {'side': side, 'actor': actor}),
                            arm_delay=False)
    other_side = other(side)
    other_carrier = team_order(state, other_side)[0]
    fire(EffectContext(state, other_side, other_carrier, {'side': other_side}), 'on_enemy_turn_begin')
    if finished(state):
        return
    tick_front_debuff(state, side)
    if finished(state):
        return
    tick_ultimates(state, side)
    if finished(state):
        return
    front = team.get('front')
    if front and not collapsed(hero(state, side, front)):
        move_out(state, side, front)
    from .tutorial import tutorial_flags
    flags = tutorial_flags(state)
    if not flags.get('disable_draw'):
        draw(state, side)
    if not finished(state):
        extra = int(team.pop('extra_ap', 0) or 0)
        if flags.get('ap_per_turn') is not None:
            team['ap'] = int(flags['ap_per_turn']) + extra
        else:
            team['ap'] = (1 if state['turn'] == 1 else 2) + extra
        if extra:
            log(state, f'回合开始，结算延后效果：额外获得 {extra} 点行动力。',
                'effect', side=side, actor=f'{side}:player', target=f'{side}:player', present=False)
        team['normal_attack_available'] = True
        team['ultimates_used'] = 0
        team['ultimate_available'] = not flags.get('disable_ultimate')


def tick_front_debuff(state, side):
    """The affected battle zone ticks before its front returns and normal draw."""
    zone = front_debuff(state, side)
    for kind in ('burn', 'star', 'delay'):
        status = zone.get(kind)
        if not status or status.get('expires_side'):
            continue
        # Older snapshots stored an absolute end-turn index.
        left = int(status.get('left', max(1, int(status.get('end', 0)) - state['sides'][side]['turn_count'] + 1)))
        amount = 1 if kind == 'burn' else (2 if kind == 'star' and left <= 1 else 0)
        if amount:
            by = status.get('by') or other(side)
            target = state['sides'][side]['front'] or 'player'
            damage(state, side, target, amount, kind=kind, source=f'{by}:{kind}')
            if finished(state):
                return
            fire(EffectContext(state, by, team_order(state, by)[0], {'side': by}),
                 'on_dot', target_id=target, kind=kind)
            if finished(state):
                return
        if not status.get('infinite'):
            status['left'] = max(0, left - 1)
            if status['left'] == 0:
                zone.pop(kind, None)
        log(state, '战斗区状态倒计时更新。', 'countdown', side=side,
            actor=f'{side}:front', target=f'{side}:front')


def expire_turn_zone_states(state, side):
    for owner in SIDES:
        zone = front_debuff(state, owner)
        for kind in ('delay', 'weave', 'stain'):
            status = zone.get(kind)
            if status and status.get('expires_side', status.get('by') if kind != 'delay' else None) == side:
                zone.pop(kind, None)
                log(state, '本回合战斗区状态结束。', 'countdown', side=owner,
                    actor=f'{owner}:front', target=f'{owner}:front')


def end_turn(state, side):
    team = state['sides'][side]
    opponent = other(side)
    opp_ctx = EffectContext(state, opponent, team_order(state, opponent)[0], {'side': opponent})
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h.get('flags', {}).get('temp_revive') and h['hp'] > 0:
            h['hp'] = 0
            h['flags'].pop('temp_revive', None)
    knockdowns(state)
    if finished(state):
        return
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h['hp'] > 0 and h['flags'].get('nightmare'):
            h['flags']['nightmare'] -= 1
            if not h['flags']['nightmare']:
                h['flags'].pop('nightmare', None)
            by = (h.get('flags') or {}).get('nightmare_by')
            damage(state, side, cid, 1, kind='nightmare',
                   source=f'{by}:nightmare' if by else None)
            fire(opp_ctx, 'on_dot', target_id=cid, kind='nightmare')
        if finished(state):
            return
        h['flags'].pop('return_after', None)
    fire(EffectContext(state, side, team_order(state, side)[0], {'side': side}), 'on_turn_end')
    if finished(state):
        return
    for card in list(team['hand']):
        if card.get('copy') and card['expires_turn'] <= team['turn_count']:
            team['hand'].remove(card)
            discard(state, side, card)
    team.update(normal_attack_available=False, ultimate_available=False,
                extra_flower=False, extra_delay=False)
    expire_turn_zone_states(state, side)
    if finished(state):
        return
    begin_turn(state, other(side))


def operation(state, side, cid, card=None, target_id=None):
    op = {'side': side, 'actor': cid, 'card': card, 'target_id': target_id,
          'participants': [], 'harmony': None, 'first_bonus': 0, 'redeem_bonus': 0,
          'record_candidate': False}
    state['operation'] = op
    context = EffectContext(state, side, cid, op)
    fire(context, 'on_operation_start')
    return context


def finish_operation(state):
    op = state['operation']
    if not op:
        return
    if finished(state):
        state['pending_choice'] = None
        return
    side, cid, card = op['side'], op['actor'], op['card']
    c = EffectContext(state, side, cid, op)
    if card:
        if card.get('copy'):
            c.on_shape(GameEvent.V2_COPY_FINISHED)
        if finished(state):
            return
        discard(state, side, card)
        fire(c, 'after_card', actor_id=cid, card=card)
    attackers = [(ps, pc) for ps, pc in (op.get('participants') or []) if alive(state, ps, pc)]
    from .tutorial import tutorial_flags
    flags = tutorial_flags(state)
    gain_energy = not flags.get('disable_energy_gain')
    gain_harmony = not flags.get('disable_harmony_gain')
    if attackers and (gain_energy or gain_harmony):
        share_side = attackers[0][0]
        before = {}
        for ally_id in team_order(state, share_side):
            if not alive(state, share_side, ally_id):
                continue
            h = hero(state, share_side, ally_id)
            before[ally_id] = {'harmony': h['harmony'], 'energy': h['energy']}
            if gain_energy:
                energy(state, share_side, ally_id, 1)
        for ps, pc in attackers:
            h = hero(state, ps, pc)
            extra = 0
            kit = KITS.get(pc)
            hook = getattr(kit, 'combat_harmony_extra', None) if kit else None
            if hook and gain_harmony:
                extra = int(hook(EffectContext(state, ps, pc, op)) or 0)
            snap = before.get(pc) or {'harmony': 0, 'energy': 0}
            if gain_harmony:
                h['harmony'] = min(harmony_cap(state), h['harmony'] + 1 + extra)
            if gain_energy:
                energy(state, ps, pc, 1)
            dh, de = h['harmony'] - snap['harmony'], h['energy'] - snap['energy']
            if gain_harmony and gain_energy:
                text = f"{h['name']}战斗积累：自己的环合 +{dh}、能量 +{de}；全队存活异能者分享能量 +1。"
            elif gain_harmony:
                text = f"{h['name']}战斗积累：自己的环合 +{dh}。"
            else:
                text = f"{h['name']}战斗积累：自己的能量 +{de}；全队存活异能者分享能量 +1。"
            log(state, text, 'resource', side=ps, source=pc, actor=f'{ps}:{pc}', target=f'{ps}:{pc}',
                harmony_delta=dh, energy_delta=de,
                before=snap, after={'harmony': h['harmony'], 'energy': h['energy']})
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
    from .tutorial import after_player_action, apply_tutorial_ack, is_tutorial
    if is_tutorial(state) and action.get('type') == 'tutorial_ack':
        return apply_tutorial_ack(state, side)
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
        log(result, f"{name(result, side)}认输。", 'finish', side=side, actor=f'{side}:player',
            winner=result['winner'], reason=result.get('reason'))
    elif kind == 'mulligan':
        removed = [c for c in team['hand'] if c['instance_id'] in candidate['card_ids']]
        team['hand'] = [c for c in team['hand'] if c['instance_id'] not in candidate['card_ids']]
        swapped = len(removed)
        log(result,
            f"{name(result, side)}替换 {swapped} 张起手牌。" if swapped else f"{name(result, side)}保留全部起手牌。",
            'mulligan', side=side, actor=f'{side}:player', amount=swapped,
            card_ids=[card['instance_id'] for card in removed])
        draw(result, side, swapped)
        team['deck'].extend(removed)
        shuffle(result, team['deck'])
        result['mulligan_done'].append(side)
        if len(result['mulligan_done']) == 2:
            begin_turn(result, result['first_side'])
    elif kind == 'choose':
        choose(result, side, candidate['choice_id'])
    elif kind == 'end_turn':
        end_turn(result, side)
    elif kind == 'ultimate':
        team['ultimates_used'] = int(team.get('ultimates_used', 0)) + 1
        team['ultimate_available'] = bool(ultimate_remaining(result, side))
        start_ultimate(result, side, candidate['character_id'], spent_ap=False)
    elif kind == 'attack':
        team['ap'] -= 1
        team['normal_attack_available'] = False
        operation(result, side, candidate['character_id']).sortie()
        finish_operation(result)
    else:
        card = next(c for c in team['hand'] if c['instance_id'] == candidate['card_id'])
        team['hand'].remove(card)
        resolve_play(result, side, card, candidate.get('target_id'), pay=True)
    if is_tutorial(result) and side == 'a' and candidate.get('type') != 'concede':
        after_player_action(result, candidate)
        from .tutorial import run_scripted_opponent
        result = run_scripted_opponent(result, apply_action)
    bump_public_patch(result)
    return result
