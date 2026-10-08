"""Pure immutable public commands and sequential turn/choice orchestration."""
from copy import deepcopy
from .entities import PlayerEntity, allocate_entity_id, clone_state, hydrate_entities, normalize_entity_action
from itertools import combinations
from random import Random

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, STARTER_DECK, validate_deck

from app.modules.card_game.content.duel_v2.registry import KITS, fire, decide, tally
from .context import EffectContext
from app.modules.card_game.engine.events import GameEvent
from .state import (SIDES, add_hand, alive, card_instance, clear_harmony_source, collapsed, damage, discard,
                    draw, effective_card, energy, finished, front_debuff, hero, knockdowns, log, move_out, name,
                    new_character, other, shield, shuffle, team_order, upgrade_harmony_residue)

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
             'combat_group': 0, 'sides': {}, 'escalation_enabled': bool(escalation),
             'harmony_residue_version': 1}
    for side in SIDES:
        order = list(selected[side]['character_ids'])
        state['sides'][side] = PlayerEntity({
            'name': name(state, side), 'hp': 30, 'shield': 0 if side == first_side else 5,
            'ap': 0, 'normal_attack_available': False, 'ultimate_available': False,
            'front': None, 'last_front': None,
            'order': order,
            'characters': {cid: new_character(cid, entity_id=allocate_entity_id(state), side=side) for cid in order},
            'hand': [], 'deck': [], 'discard': [], 'removed': [], 'records': [],
            'turn_count': 0, 'fatigue': 0, 'used': {}, 'harmonized': {},
            'extra_flower': False, 'extra_delay': False, 'front_debuff': {},
        }, entity_id=allocate_entity_id(state), side=side)
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
    for cid in (card.get('character_id'), card.get('original_character_id')):
        kit = KITS.get(cid)
        fn = getattr(kit, 'makes_instant', None)
        if fn and fn(state, side, card):
            return True
    return False


def card_plays_instant(state, side, card):
    return bool(card.get('instant')) or kit_makes_form_instant(state, side, card) or kit_makes_card_instant(state, side, card)


def card_is_free_instant(state, side, card):
    team = state['sides'][side]
    # A zero-AP card never spends the shared instant opportunity, even when granted instant.
    return bool(card_cost(state, side, card) > 0
                and card_plays_instant(state, side, card) and not team['used'].get('instant'))


def card_cost(state, side, card):
    limit = card.get('spend_all_ap')
    base = min(int(state['sides'][side]['ap']), int(limit)) if limit else int(card.get('cost') or 0)
    if card.get('free_if_surplus') and state['sides'][side].get('surplus'):
        base = 0
    from .modifiers import evaluate
    return max(0, int(evaluate(state, side, hero(state, side, card['character_id']),
                              'card.cost', base, query={'card': card})['value']))


def can_afford_card(state, side, card):
    if card.get('spend_all_ap') and state['sides'][side]['ap'] <= 0:
        return False
    if not decide(EffectContext(state, side, card['character_id'], {'side': side}),
                  'can_play', True, actor_only=True, card=card):
        return False
    cost = card_cost(state, side, card)
    return card_is_free_instant(state, side, card) or int(state['sides'][side].get('ap') or 0) >= cost


def card_play_options(state, side, card):
    options = card.get('play_options') or []
    if not options:
        return [{}]
    owner = hero(state, side, card['character_id'])
    return [{'option_id': option['id']} for option in options
            if all(int(owner.get(resource, 0)) >= int(amount)
                   for resource, amount in option.get('costs', {}).items())]


def resolve_play(state, side, card, target_id=None, *, pay=True, option_id=None):
    from app.modules.card_game.content.duel_v2.characters.common import scale_battle_frame
    from .presentation import public_card
    team = state['sides'][side]
    options = card.get('play_options') or []
    # Automatic releases/responses always resolve the leftmost option.
    selected_id = option_id if pay and not state.get('_response_lock') else None
    option = next((item for item in options if item['id'] == selected_id), options[0] if options else {})
    if option and options and {'option_id': option['id']} not in card_play_options(state, side, card):
        raise ValueError('所选效果的资源不足。')
    if option.get('costs'):
        owner = hero(state, side, card['character_id'])
        for resource, amount in option['costs'].items():
            owner[resource] = int(owner.get(resource, 0)) - int(amount)
        log(state, f"{owner['name']}选择「{option['label']}」，消耗{option.get('cost_label', '额外资源')}。",
            'effect', side=side, actor=f"{side}:{card['character_id']}", present=False,
            private_side=side, private_hand=[public_card(held) for held in team['hand']])
    form_bonus = kit_makes_form_instant(state, side, card)
    ap_paid = 0
    if pay:
        if card_is_free_instant(state, side, card):
            team['used']['instant'] = True
        else:
            ap_paid = card_cost(state, side, card)
            team['ap'] -= ap_paid
    log(state, f"{hero(state, side, card['character_id'])['name']}使用「{card['name']}」。",
        'play', side=side, source=card['character_id'],
        actor=f"{side}:{card['character_id']}", card=public_card(card, state, side))
    c = operation(state, side, card['character_id'], card, target_id)
    c.operation['option_id'] = option.get('id')
    c.operation['ap_paid'] = ap_paid
    if card.get('type') == 'battle' and alive(state, side, card['character_id']):
        attack = int(card.get('attack') or 0)
        value = int(card.get('shield') or 0)
        attack, value = scale_battle_frame(c, attack, value)
        if value:
            shield(state, side, card['character_id'], value)
        if attack or card.get('attack_mode') == 'set':
            if state.get('_response_intercept'):
                flags = hero(state, side, card['character_id']).setdefault('flags', {})
                if card.get('attack_mode') == 'set':
                    flags['counter_set_attack'] = attack
                else:
                    flags['counter_bonus'] = int(flags.get('counter_bonus') or 0) + attack
            elif card.get('attack_mode') == 'set':
                c.operation['card_set_attack'] = attack
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
        if state.get('_response_lock') and kind == 'self_downed' and not finished(state):
            state.setdefault('_pending_down_responses', []).append((defending_side, target_id))
        return
    team = state['sides'][defending_side]
    for physical in list(team['hand']):
        card = effective_card(state, defending_side, physical)
        if card.get('response') != kind:
            continue
        owner = card['character_id']
        down_response = kind == 'self_downed' and target_id == owner
        if (not alive(state, defending_side, owner) and not down_response) or collapsed(hero(state, defending_side, owner)):
            continue
        if kind == 'ally_attacked' and (not target_id or target_id == owner):
            continue
        if kind == 'faction_front_attacked' and (team['front'] != target_id or
                hero(state, defending_side, target_id).get('faction') != card.get('response_faction')):
            continue
        if kind == 'self_attacked' and target_id != owner:
            continue
        if kind in ('self_lethal', 'self_downed') and target_id != owner:
            continue
        if kind == 'enemy_turn_end_empty' and state['sides'][other(defending_side)]['front']:
            continue
        if not can_afford_card(state, defending_side, card):
            continue
        team['hand'].remove(physical)
        parent_op = state.get('operation')
        state['_response_lock'] = True
        state['_response_intercept'] = kind in ('self_attacked', 'ally_attacked', 'self_lethal')
        try:
            response_target = f'{defending_side}:{target_id}' if kind == 'ally_attacked' and target_id else None
            resolve_play(state, defending_side, card, response_target, pay=True)
        finally:
            state['_response_lock'] = False
            state.pop('_response_intercept', None)
            state['operation'] = parent_op
        for down_side, down_id in state.pop('_pending_down_responses', []):
            trigger_responses(state, down_side, 'self_downed', target_id=down_id)
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
    owner = card.get('character_id')
    if owner in state['sides'][side]['characters'] and card.get('type') == 'battle':
        aimed = decide(EffectContext(state, side, owner, {'side': side}),
                       'battle_targets', None, actor_only=True, card=card)
        if isinstance(aimed, list):
            return [{'target_id': target} for target in aimed]
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
    if policy == 'enemy_bench':
        foe = other(side)
        return [{'target_id': f'{foe}:{cid}'} for cid in team_order(state, foe)
                if alive(state, foe, cid) and enemy['front'] != cid]
    if policy in ('living_enemy', 'living_enemy_esper'):
        foe = other(side)
        return [{'target_id': f'{foe}:{cid}'} for cid in team_order(state, foe)
                if alive(state, foe, cid)
                and (policy != 'living_enemy_esper' or not hero(state, foe, cid).get('summoned'))]
    if policy == 'living_enemy_role':
        foe = other(side)
        return [{'target_id': f'{foe}:player'}] + [
            {'target_id': f'{foe}:{cid}'} for cid in team_order(state, foe)
            if alive(state, foe, cid)]
    if policy == 'any_living_esper':
        return [{'target_id': f'{ps}:{cid}'} for ps in SIDES
                for cid in team_order(state, ps)
                if alive(state, ps, cid) and not hero(state, ps, cid).get('summoned')]
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
    if policy == 'enemy_not_in_lineup':
        foe = other(side)
        ours = set(team_order(state, side))
        return [{'target_id': f'{foe}:{cid}'} for cid in team_order(state, foe)
                if cid not in ours and alive(state, foe, cid) and not hero(state, foe, cid).get('summoned')]
    if policy == 'any_summon':
        found = []
        for ps in SIDES:
            for cid in team_order(state, ps):
                h = hero(state, ps, cid)
                if h.get('summoned') and alive(state, ps, cid):
                    found.append({'target_id': f'{ps}:{cid}'})
        return found
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


def normal_attack_payment(state, side, cid):
    """Pure payment offer shared by legality, execution and previews."""
    team = state['sides'][side]
    if team.get('seal_normal_attack'):
        return None
    default = {'ap': 1, 'normal': True} if team['normal_attack_available'] else None
    if hero(state, side, cid).get('summoned'):
        return None
    offer = decide(EffectContext(state, side, cid, {'side': side, 'actor': cid}),
                   'normal_attack_payment', default, actor_only=True)
    if isinstance(offer, dict) and hero(state, side, cid).get('flags', {}).get('free_normal_attack_turn') == state['turn']:
        offer = {**offer, 'ap': 0}
    return offer if isinstance(offer, dict) else None


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
    if phase == 'choice' and state['pending_choice']['kind'] == 'hand_redraw':
        ids = [c['instance_id'] for c in state['pending_choice']['cards']]
        return result + [{'type': 'choose_cards', 'card_ids': list(group)}
                         for size in range(min(3, len(ids)) + 1) for group in combinations(ids, size)]
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
        payment = normal_attack_payment(state, side, cid)
        if payment is not None and team['ap'] >= payment['ap'] and not collapsed(h):
            aimed = decide(EffectContext(state, side, cid, {'side': side}),
                           'normal_attack_targets', None, actor_only=True)
            if isinstance(aimed, list) and aimed:
                result.extend({'type': 'attack', 'character_id': cid, 'target_id': target}
                              for target in aimed)
            else:
                result.append({'type': 'attack', 'character_id': cid})
        if (ultimate_remaining(state, side) and h['energy'] >= energy_cap(state, h)
                and h.get('ultimate_used_turn') != state['turn'] and not h.get('summoned')
                and decide(EffectContext(state, side, cid, {'side': side}),
                           'can_ultimate', True, actor_only=True)):
            result.append({'type': 'ultimate', 'character_id': cid})
    for physical in team['hand']:
        card = effective_card(state, side, physical)
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
            result.extend({'type': 'play_card', 'card_id': card['instance_id'], **target, **option}
                          for target in targets(state, side, card)
                          for option in card_play_options(state, side, card))
    from .tutorial import filter_legal_actions, is_tutorial
    if is_tutorial(state):
        return filter_legal_actions(state, side, result)
    return result


def ultimate_spec(cid):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    try:
        spec = (lookup_character(cid) or {}).get('ultimate') or {}
    except KeyError:  # Runtime summons have no catalog ultimate.
        spec = {}
    kind = spec.get('kind') or 'status'
    if kind not in ('status', 'instant'):
        kind = 'status'
    turns = int(spec.get('turns') or 2)
    return {'kind': kind, 'turns': max(1, turns), 'expires': spec.get('expires', 'turn_start')}


def start_ultimate(state, side, cid, *, spent_ap=False):
    from app.modules.card_game.content.duel_v2.tutorial import lookup_character
    h = hero(state, side, cid)
    spec = ultimate_spec(cid)
    team = state['sides'][side]
    before = {'energy': h['energy'], 'awakened': h.get('awakened', False),
              'ultimate_turns': h.get('ultimate_turns', 0), 'ap': team['ap'] + (1 if spent_ap else 0)}
    h['energy'] = 0
    h['ultimate_used_turn'] = state['turn']
    if spec['kind'] == 'instant':
        h['awakened'] = False
        h['ultimate_turns'] = 0
        text = f"{h['name']}消耗 {energy_cap(state, h)} 点{h.get('resource_name', '能量')}，发动终结。"
    else:
        h['awakened'] = True
        spec['turns'] = int(decide(EffectContext(state, side, cid, {'side': side}),
                                   'ultimate_duration', spec['turns'], actor_only=True))
        h['ultimate_turns'] = spec['turns']
        from .effect_lifecycle import schedule_ultimate
        schedule_ultimate(state, side, h, spec['turns'], spec['expires'])
        duration = '持续至本回合结束' if spec['expires'] == 'turn_end' else f"持续 {spec['turns']} 个己方回合"
        text = f"{h['name']}消耗 {energy_cap(state, h)} 点{h.get('resource_name', '能量')}，发动终结（{duration}）。"
    log(state, text, 'ultimate', side=side, source=h['id'], actor=f'{side}:{h["id"]}',
        target=f'{side}:{h["id"]}',
        description=(lookup_character(cid) or {}).get('awakened_passive') or '',
        before=before,
        after={'energy': h['energy'], 'awakened': h['awakened'], 'ultimate_turns': h['ultimate_turns'], 'ap': team['ap']})
    fire(EffectContext(state, side, cid, {'side': side}), 'on_ultimate', actor_only=True)
    fire(EffectContext(state, side, cid, {'side': side}), 'on_ally_ultimate', actor_id=cid)
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
    from .modifiers import clear
    clear(h, 'ultimate_end')
    fire(EffectContext(state, side, cid, {'side': side}), 'on_ultimate_end',
         actor_only=True, include_down=True)
    if finished(state):
        return
    log(state, f"{h['name']}的终结结束。", 'ultimate', side=side, source=h['id'],
        actor=f'{side}:{h["id"]}', target=f'{side}:{h["id"]}',
        present=False,
        after={'awakened': False, 'ultimate_turns': 0})




def begin_turn(state, side):
    from .effect_lifecycle import prepare_effects, expire_effects
    prepare_effects(state)
    state['phase'], state['active_side'] = 'playing', side
    state['turn'] += 1
    team = state['sides'][side]
    team['turn_count'] += 1
    # Announce the turn before any start-of-turn board changes are presented.
    log(state, f"{name(state, side)}的第 {team['turn_count']} 个回合开始。", 'turn', side=side,
        actor=f'{side}:player')
    expire_effects(state, side, 'turn_start')
    if finished(state):
        return
    team.pop('life_locked', None)
    team['used'] = {}
    for ps in SIDES:
        state['sides'][ps]['attacked_this_turn'] = []
    upgrade_harmony_residue(state)
    team['harmony_damage'] = False
    team['genesis_damage_bonus'] = 0
    team['surplus'] = False
    clamp_resources(state)
    # Player shields, including the second player's opening shield, expire here.
    team['shield'] = 0
    from .continuous import halve_dots
    halve_dots(state, side)
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
    fire(EffectContext(state, side, carrier, {'side': side}), 'on_turn_clock', include_down=True)
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
    from .deferred import drain
    drain(state)
    tick_front_debuff(state, side)
    if finished(state):
        return
    from .state import record_delay_turn
    for ps in SIDES:
        record_delay_turn(state, ps)
    # Zone clocks see attachments before their own countdown is reduced.
    fire(EffectContext(state, side, carrier, {'side': side}), GameEvent.V2_TURN_COUNTDOWN, include_down=True)
    if finished(state):
        return
    front = team.get('front')
    if (front and not collapsed(hero(state, side, front)) and not hero(state, side, front).get('summoned')
            and decide(EffectContext(state, side, front, {'side': side}), 'natural_return', True, actor_only=True)):
        move_out(state, side, front)
    from .tutorial import tutorial_flags
    flags = tutorial_flags(state)
    if not flags.get('disable_draw'):
        inspect = decide(EffectContext(state, side, carrier, {'side': side}), 'turn_draw_inspect', None)
        if inspect and team['deck']:
            c = operation(state, side, inspect['character_id'])
            c.inspect_top_choice(inspect['count'])
            if state['phase'] != 'choice':
                finish_operation(state)
            else:
                state['turn_start_pending'] = side
        else:
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
        team['normal_attack_available'] = not bool(team.pop('skip_normal_attack', False))
        team['ultimates_used'] = 0
        team['ultimate_available'] = not flags.get('disable_ultimate')
        from .temporal import save_turn_start
        save_turn_start(state, side)


def tick_front_debuff(state, side):
    """The affected battle zone ticks before its front returns and normal draw."""
    zone = front_debuff(state, side)
    for kind in ('star', 'delay'):
        status = zone.get(kind)
        if not status or status.get('expires_side') or (kind == 'delay' and status.get('tick_side')):
            continue
        # Older snapshots stored an absolute end-turn index.
        left = int(status.get('left', max(1, int(status.get('end', 0)) - state['sides'][side]['turn_count'] + 1)))
        amount = 2 if kind == 'star' and left <= 1 else 0
        if amount:
            by = status.get('by') or other(side)
            target = state['sides'][side]['front'] or 'player'
            dot_context = EffectContext(state, by, team_order(state, by)[0], {'side': by})
            amount += tally(dot_context, 'dot_damage_bonus', kind=kind)
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

    for target_side in SIDES:
        zone = front_debuff(state, target_side)
        status = zone.get('delay')
        if not status or status.get('tick_side') != side:
            continue
        ctx = EffectContext(state, side, team_order(state, side)[0], {'side': side})
        paused = decide(ctx, 'pause_delay', False, include_down=True)
        if not paused:
            status['left'] = max(0, int(status['left']) - 1)
            if not status['left']:
                zone.pop('delay', None)
        log(state, '延滞倒计时保持。' if paused else '延滞倒计时更新。', 'countdown',
            side=target_side, actor=f'{target_side}:front', target=f'{target_side}:front')


def expire_turn_zone_states(state, side):
    for owner in SIDES:
        zone = front_debuff(state, owner)
        for kind in ('delay', 'weave', 'stain'):
            status = zone.get(kind)
            if status and status.get('expires_side', status.get('by') if kind != 'delay' else None) == side:
                zone.pop(kind, None)
                log(state, '本回合战斗区状态结束。', 'countdown', side=owner,
                    actor=f'{owner}:front', target=f'{owner}:front')


def _expire_collapse(state, side):
    team = state['sides'][side]
    for ps in SIDES:
        for cid in team_order(state, ps):
            h = hero(state, ps, cid)
            col = (h.get('flags') or {}).get('collapse')
            if col and col.get('side') == side and col.get('until', 0) <= team['turn_count']:
                h['flags'].pop('collapse', None)
                log(state, f"{h['name']}的倾陷结束。", 'collapse', side=ps, source=cid,
                    actor=f'{ps}:{cid}', target=f'{ps}:{cid}')


def end_turn(state, side):
    team = state['sides'][side]
    opponent = other(side)
    for cid in team_order(state, side):
        h = hero(state, side, cid)
        if h.get('flags', {}).get('temp_revive') and h['hp'] > 0:
            h['hp'] = 0
            h['flags'].pop('temp_revive', None)
    knockdowns(state)
    if finished(state):
        return
    for cid in team_order(state, side):
        hero(state, side, cid).get('flags', {}).pop('return_after', None)
    fire(EffectContext(state, side, team_order(state, side)[0], {'side': side}), 'on_turn_end')
    if finished(state):
        return
    from .continuous import tick_owner_end
    tick_owner_end(state, side)
    if finished(state):
        return
    _expire_collapse(state, side)
    for h in team['characters'].values():
        (h.get('flags') or {}).pop('mimic', None)
    for ps in SIDES:
        hand = state['sides'][ps]['hand']
        for card in list(hand):
            if card.get('ephemeral_turn') is not None and card['ephemeral_turn'] <= state['turn']:
                hand.remove(card)
                discard(state, ps, card)
    if finished(state):
        return
    trigger_responses(state, opponent, 'enemy_turn_end_empty')
    if finished(state):
        return
    for card in list(team['hand']):
        if card.get('copy') and card['expires_turn'] <= team['turn_count']:
            team['hand'].remove(card)
            discard(state, side, card)
    team.update(normal_attack_available=False, ultimate_available=False,
                extra_flower=False, extra_delay=False, genesis_damage_bonus=0)
    for h in team['characters'].values():
        h['flags'].pop('free_normal_attack_turn', None)
    expire_turn_zone_states(state, side)
    from .effect_lifecycle import prepare_effects, expire_effects
    prepare_effects(state)
    expire_effects(state, side, 'turn_end')
    # Old snapshots did not record an expiry for the one-shot resistance table.
    from .modifiers import entities
    for _, holder in entities(state):
        holder.pop('next_damage_resistance', None)
    if finished(state):
        return
    begin_turn(state, other(side))


def operation(state, side, cid, card=None, target_id=None):
    op = {'side': side, 'actor': cid, 'card': card, 'target_id': target_id,
          'participants': [], 'harmony': None, 'first_bonus': 0, 'redeem_bonus': 0,
          'record_candidate': False, 'card_mark': deepcopy((card or {}).get('jingu_mark'))}
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
        if card.get('retain'):
            add_hand(state, side, card)
        else:
            discard(state, side, card)
        fire(c, 'after_card', actor_id=cid, card=card)
        if op.get('card_mark'):
            if state.get('_response_intercept') and card.get('type') == 'battle':
                state.setdefault('_deferred_card_marks', []).append(deepcopy(op))
            else:
                fire(c, 'after_marked_card', include_down=True)
    attackers = [(ps, pc) for ps, pc in (op.get('participants') or []) if alive(state, ps, pc)]
    from .tutorial import tutorial_flags
    flags = tutorial_flags(state)
    gain_energy = not flags.get('disable_energy_gain')
    gain_harmony = not flags.get('disable_harmony_gain') and not op.get('skip_combat_harmony')
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
    fire(c, GameEvent.V2_OPERATION_RESOURCES_RESOLVED, actor_only=True)
    from .modifiers import clear, entities
    for _, holder in entities(state):
        clear(holder, 'operation_end')
    state['phase'] = 'playing'
    state['operation'], state['pending_choice'] = None, None
    from .deferred import drain
    drain(state)


def settle_deferred_card_marks(state):
    """Counter cards finish only after their simultaneous battle damage."""
    for op in state.pop('_deferred_card_marks', []):
        fire(EffectContext(state, op['side'], op['actor'], op),
             'after_marked_card', include_down=True)


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
    state['phase'], state['pending_choice'] = 'playing', None
    if kind == 'kit':
        op = state['operation']
        fire(EffectContext(state, side, op['actor'], op), 'on_choice', actor_only=True, selected=selected)
    if state['phase'] != 'choice':
        finish_operation(state)
        if state.get('turn_start_pending'):
            from .temporal import save_turn_start
            save_turn_start(state, state.pop('turn_start_pending'))


def redraw_hand(state, side, card_ids):
    from .presentation import public_card
    team = state['sides'][side]
    selected = [c for c in team['hand'] if c['instance_id'] in card_ids]
    team['hand'] = [c for c in team['hand'] if c['instance_id'] not in card_ids]
    state['pending_choice'] = None
    state['phase'] = 'playing'
    log(state, f"{name(state, side)}调度 {len(selected)} 张手牌。", 'mulligan', side=side,
        amount=len(selected), card_ids=card_ids, private_side=side,
        private_hand=[public_card(c) for c in team['hand']])
    draw(state, side, len(selected))
    team['deck'].extend(selected)
    if selected:
        shuffle(state, team['deck'])
    finish_operation(state)


def apply_action(state, side, action):
    if side not in SIDES or not isinstance(action, dict):
        raise ValueError('无效的操作格式。')
    from .tutorial import after_player_action, apply_tutorial_ack, is_tutorial
    if is_tutorial(state) and action.get('type') == 'tutorial_ack':
        return apply_tutorial_ack(state, side)
    state = clone_state(state)
    candidate = normalize_entity_action(state, side, action)
    if candidate.get('type') in ('mulligan', 'choose_cards') and isinstance(candidate.get('card_ids'), list):
        ids = candidate['card_ids']
        if any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('换牌选择无效。')
        order = {c['instance_id']: i for i, c in enumerate(state['sides'][side]['hand'])}
        candidate['card_ids'] = sorted(ids, key=lambda x: order.get(x, 999))
    if candidate not in legal_actions(state, side):
        raise ValueError('当前不能执行此操作：请检查回合、行动力、角色状态和目标。')
    from .presentation import bump_public_patch, capture_public_board
    result = state
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
    elif kind == 'choose_cards':
        redraw_hand(result, side, candidate['card_ids'])
    elif kind == 'choose':
        choose(result, side, candidate['choice_id'])
    elif kind == 'end_turn':
        end_turn(result, side)
    elif kind == 'ultimate':
        team['ultimates_used'] = int(team.get('ultimates_used', 0)) + 1
        team['ultimate_available'] = bool(ultimate_remaining(result, side))
        start_ultimate(result, side, candidate['character_id'], spent_ap=False)
    elif kind == 'attack':
        cid = candidate['character_id']
        payment = normal_attack_payment(result, side, cid)
        team['ap'] -= payment['ap']
        if payment.get('normal'):
            team['normal_attack_available'] = False
        c = operation(result, side, cid)
        if candidate.get('target_id'):
            c.operation['target_id'] = candidate['target_id']
        c.operation['normal_attack'] = True
        fire(c, 'pay_normal_attack', actor_only=True, payment=payment)
        c.sortie(bonus=payment.get('bonus', 0))
        finish_operation(result)
    else:
        card = next(c for c in team['hand'] if c['instance_id'] == candidate['card_id'])
        team['hand'].remove(card)
        resolve_play(result, side, effective_card(result, side, card), candidate.get('target_id'), pay=True, option_id=candidate.get('option_id'))
    from .deferred import drain
    drain(result)
    if is_tutorial(result) and side == 'a' and candidate.get('type') != 'concede':
        after_player_action(result, candidate)
        from .tutorial import run_scripted_opponent
        result = run_scripted_opponent(result, apply_action)
    bump_public_patch(result)
    return result
