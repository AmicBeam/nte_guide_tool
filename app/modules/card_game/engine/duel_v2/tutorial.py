"""Tutorial overlay: scripted legal actions, opponent scripts, and guidance."""
from copy import deepcopy
from .entities import PlayerEntity, allocate_entity_id, clone_state, hydrate_entities

from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS, load_scenario
from .presentation import bump_public_patch, capture_public_board
from .state import SIDES, card_instance, name, new_character, other, shuffle


def is_tutorial(state):
    flags = state.get('flags') if isinstance(state, dict) else None
    return bool(isinstance(flags, dict) and flags.get('tutorial'))


def tutorial_flags(state):
    flags = (state.get('flags') or {}).get('tutorial') or {}
    return flags if isinstance(flags, dict) else {}


def current_step(state):
    flags = tutorial_flags(state)
    steps = flags.get('steps') or []
    index = int(flags.get('step_index') or 0)
    if 0 <= index < len(steps):
        return steps[index]
    return None


def new_tutorial_game(scenario_id, seed=0):
    scenario = load_scenario(scenario_id)
    seed = int(scenario.get('seed') or seed or 0)
    first_side = scenario.get('first_side') or 'a'
    starting_turn = int(scenario.get('starting_turn', 1))
    opening_side = first_side if starting_turn % 2 else other(first_side)
    flags = deepcopy(scenario.get('flags') or {})
    order_a = list(scenario.get('order_a') or [])
    order_b = list(scenario.get('order_b') or [])
    player_hp = int(flags.get('player_hp') or 30)
    player_shield = int(flags.get('player_shield') or 0)
    state = {
        'rules_version': 'duel_v2',
        'design_version': 'V2.5',
        'version': 0,
        'phase': 'playing',
        'active_side': opening_side,
        'first_side': first_side,
        'turn': starting_turn - 1,
        'escalation_enabled': bool(flags.get('escalation_enabled')),
        'winner': None,
        'reason': None,
        'rng': seed,
        'next_instance': 0,
        'event_seq': 0,
        'events': [],
        'pending_choice': None,
        'operation': None,
        'mulligan_done': [],
        'current_action_id': 'setup',
        'current_action_version': 0,
        'combat_group': 0,
        'sides': {},
        'flags': {
            'tutorial': {
                **flags,
                'scenario': scenario['id'],
                'title': scenario.get('title') or '',
                'level_index': int(scenario.get('level_index') or 1),
                'level_count': len(CAMPAIGN_LEVELS),
                'step_index': 0,
                'steps': deepcopy(scenario.get('steps') or []),
                'hide': list(flags.get('hide') or []),
            },
        },
    }
    for side, order in (('a', order_a), ('b', order_b)):
        state['sides'][side] = PlayerEntity({
            'name': name(state, side),
            'hp': player_hp,
            'shield': player_shield,
            'ap': 0,
            'normal_attack_available': False,
            'ultimate_available': False,
            'front': scenario.get(f'front_{side}'),
            'last_front': None,
            'order': list(order),
            'characters': {cid: new_character(cid, entity_id=allocate_entity_id(state), side=side) for cid in order},
            'hand': [],
            'deck': [],
            'discard': [],
            'removed': [],
            'records': [],
            'turn_count': (starting_turn - 1) // 2 + int(starting_turn % 2 == 0 and side == first_side),
            'fatigue': 0,
            'used': {},
            'harmonized': {},
            'extra_flower': False,
            'extra_delay': False,
            'front_debuff': {},
        }, entity_id=allocate_entity_id(state), side=side)
        for cid, stats in (scenario.get('stats') or {}).items():
            hero = state['sides'][side]['characters'].get(cid)
            if not hero or not isinstance(stats, dict):
                continue
            if 'attack' in stats:
                hero['attack'] = int(stats['attack'])
                hero['base_attack'] = int(stats['attack'])
            if 'hp' in stats:
                hero['hp'] = int(stats['hp'])
            if 'max_hp' in stats:
                hero['max_hp'] = int(stats['max_hp'])
                hero['base_max_hp'] = int(stats['max_hp'])
            if 'energy' in stats:
                hero['energy'] = int(stats['energy'])
            if 'harmony' in stats:
                hero['harmony'] = int(stats['harmony'])
        for card_id in scenario.get(f'hand_{side}') or []:
            state['sides'][side]['hand'].append(card_instance(state, card_id, side))
        for card_id in scenario.get(f'deck_{side}') or []:
            state['sides'][side]['deck'].append(card_instance(state, card_id, side))
        if not flags.get('fixed_rng'):
            shuffle(state, state['sides'][side]['deck'])
        opening_draw = int(flags.get('opening_draw') or 0)
        if opening_draw:
            from .state import draw
            draw(state, side, opening_draw)
    from .flow import begin_turn
    for side, extra in (scenario.get('extra_ap') or {}).items():
        if side in state['sides']:
            state['sides'][side]['extra_ap'] = int(extra)
    skip_mulligan = flags.get('skip_mulligan', True)
    if skip_mulligan:
        state['mulligan_done'] = list(SIDES)
        begin_turn(state, opening_side)
        opening = list(scenario.get('opening_b') or [])
        if opening_side == 'b' and opening:
            from .flow import apply_action as apply_opening
            for action in opening:
                state = apply_opening(state, 'b', action)
    else:
        state['phase'] = 'mulligan'
        state['mulligan_done'] = ['b']
    state['_public_board'] = capture_public_board(state)
    return state


def _same_action(action, spec):
    if not isinstance(action, dict) or not isinstance(spec, dict):
        return False
    return all(action.get(key) == value for key, value in spec.items())


def filter_legal_actions(state, side, actions):
    if not is_tutorial(state):
        return actions
    step = current_step(state)
    if step is None:
        return []
    if side != 'a':
        return [item for item in actions if item.get('type') != 'concede']
    allowed = list(step.get('allowed') or [])
    result = []
    for spec in allowed:
        if spec.get('type') == 'tutorial_ack':
            result.append({'type': 'tutorial_ack'})
            continue
        result.extend(item for item in actions if _same_action(item, spec))
    return result


def apply_tutorial_ack(state, side):
    step = current_step(state)
    if side != 'a' or not step or not step.get('ack_required'):
        raise ValueError('当前不能执行此操作：请检查回合、行动力、角色状态和目标。')
    result = clone_state(state)
    result['_public_board'] = capture_public_board(result)
    result['version'] += 1
    result['current_action_version'] = result['version']
    result['current_action_id'] = f"v{result['version']}"
    result['combat_group'] = 0
    if step.get('finish_level'):
        _complete_level(result)
    else:
        _advance_step(result)
    bump_public_patch(result)
    return result


def after_player_action(state, action):
    if not is_tutorial(state) or not isinstance(action, dict):
        return state
    step = current_step(state)
    if step is None:
        return state
    allowed = list(step.get('allowed') or [])
    if not any(_same_action(action, spec) for spec in allowed if spec.get('type') != 'tutorial_ack'):
        return state
    if step.get('finish_level'):
        _complete_level(state)
    else:
        _advance_step(state)
    return state


def _advance_step(state):
    flags = tutorial_flags(state)
    flags['step_index'] = int(flags.get('step_index') or 0) + 1
    state['flags']['tutorial'] = flags


def _complete_level(state):
    from .state import log
    if state.get('phase') == 'finished':
        return
    state['phase'] = 'finished'
    state['winner'] = 'a'
    state['reason'] = 'tutorial_complete'
    log(state, '本关完成。', 'system', side='a', actor='a:player')


def scripted_actions(state):
    flags = tutorial_flags(state)
    index = int(flags.get('step_index') or 0)
    if int(flags.get('scripted_for') or -1) == index:
        return []
    steps = flags.get('steps') or []
    if index <= 0 or index > len(steps):
        return []
    previous = steps[index - 1]
    return list(previous.get('opponent_then') or [])


def run_scripted_opponent(state, apply_action):
    from .flow import acting_side
    if not is_tutorial(state) or state.get('phase') == 'finished':
        return state
    flags = tutorial_flags(state)
    pending = list(scripted_actions(state))
    flags['scripted_for'] = int(flags.get('step_index') or 0)
    state.setdefault('flags', {})['tutorial'] = flags
    for action in pending:
        if acting_side(state) != 'b' or state.get('phase') == 'finished':
            break
        state = apply_action(state, 'b', action)
    return state


def project_tutorial(state, side):
    if not is_tutorial(state):
        return None
    flags = tutorial_flags(state)
    step = current_step(state)
    hide = list(flags.get('hide') or [])
    if flags.get('disable_hand'):
        for key in ('hand', 'deck', 'mulligan'):
            if key not in hide:
                hide.append(key)
    payload = {
        'enabled': True,
        'campaign_id': 'duel_v2',
        'scenario': flags.get('scenario'),
        'title': flags.get('title') or '',
        'level_index': int(flags.get('level_index') or 1),
        'level_count': int(flags.get('level_count') or 12),
        'step_id': (step or {}).get('id'),
        'ack_required': bool((step or {}).get('ack_required')),
        'can_skip_step': False,
        'can_skip_tutorial': False,
        'can_retry': True,
        'character_slots': int(flags.get('character_slots') or 2),
        'hide': hide,
        'mask': bool((step or {}).get('mask', True)),
        'placement': (step or {}).get('placement') or 'bottom-right',
        'spotlights': list((step or {}).get('spotlights') or []),
        'drag_arrow': deepcopy((step or {}).get('drag_arrow')),
        'click_cue': (step or {}).get('click_cue'),
        'cue': (step or {}).get('cue') or '',
        'modal': deepcopy((step or {}).get('modal')),
        'allowed_actions': deepcopy((step or {}).get('allowed') or []),
        'blocked_reason': (step or {}).get('blocked_reason') or '',
        'viewer_side': side,
    }
    return payload


def strip_hidden_fields(observe_payload, tutorial):
    if not tutorial:
        return observe_payload
    hide = set(tutorial.get('hide') or [])
    for side in SIDES:
        team = observe_payload.get('sides', {}).get(side) or {}
        if 'hand' in hide:
            team['hand'] = []
            team['hand_count'] = 0
        if 'deck' in hide:
            team['deck_count'] = 0
            team['discard'] = []
        characters = []
        for hero in team.get('characters') or []:
            item = dict(hero)
            if 'passive' in hide:
                item.pop('passive', None)
                item.pop('awakened_passive', None)
            if 'harmony' in hide:
                item.pop('harmony', None)
                item.pop('harmony_max', None)
                item.pop('harmony_ready', None)
                item.pop('harmony_source', None)
            if 'energy' in hide:
                item.pop('energy', None)
                item.pop('energy_max', None)
                item.pop('awakened', None)
                item.pop('ultimate_turns', None)
            if 'form' in hide:
                item['shape'] = None
                item['shape_id'] = None
                item['shape_description'] = None
            if 'ultimate' in hide:
                item.pop('awakened', None)
                item.pop('ultimate_turns', None)
            if tutorial.get('scenario') != 'tutorial_l11_pierce' or item.get('id') != 'tutorial_bohe':
                if 'passive' in hide:
                    item.pop('passive', None)
            characters.append(item)
        team['characters'] = characters
        if 'harmony' in hide:
            team.pop('harmony_available', None)
            team.pop('front_debuff', None)
        if 'ultimate' in hide:
            team['ultimate_available'] = False
        observe_payload['sides'][side] = team
    return observe_payload
