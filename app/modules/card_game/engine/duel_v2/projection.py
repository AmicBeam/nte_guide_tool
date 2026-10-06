"""Viewer-specific projection. Never expose decks, RNG, or private choices."""
from .modifiers import compatibility_flags, public_effects
from copy import deepcopy
from .entities import clone_state, entity_action, needs_hydration

from .effect_markers import character_effect_markers, team_effect_markers
from app.modules.card_game.content.duel_v2 import CARDS
from .flow import acting_side, apply_action, card_plays_instant, card_is_free_instant, legal_actions, normal_attack_payment, ultimate_spec, card_cost
from .presentation import (interaction_for_action, presentation_bundle, public_card,
                           sanitize_public_event)
from .combat import harmony_marks
from .state import SIDES, attack_value, collapsed, damage_immune, damage_limit, effective_card, hero, name, other, team_order


def _card(state, side, instance_id):
    return effective_card(state, side, next(c for c in state['sides'][side]['hand'] if c['instance_id'] == instance_id))


TYPE_ORDER = {'tactic': 0, 'battle': 1, 'form': 2}


def _sort_hand(cards, character_ids):
    index = {cid: i for i, cid in enumerate(character_ids)}
    return sorted(cards, key=lambda card: (
        index.get(card.get('hand_face', card).get('character_id'), len(index)),
        TYPE_ORDER.get(card.get('hand_face', card).get('type'), 9),
    ))


def action_label(state, side, action):
    kind = action['type']
    if kind == 'attack':
        return f"{hero(state, side, action['character_id'])['name']}出击"
    if kind == 'ultimate':
        return f"终结{hero(state, side, action['character_id'])['name']}"
    if kind == 'play_card':
        text = f"使用「{_card(state, side, action['card_id'])['name']}」"
        option = next((o for o in (_card(state, side, action['card_id']).get('play_options') or []) if o['id'] == action.get('option_id')), None)
        if option:
            text += ' · ' + option['label']
        target = action.get('target_id')
        if target and ':' in target:
            ts, cid = target.split(':', 1)
            text += f" → {name(state, ts, cid)}"
        elif target:
            recovered = next(c for c in state['sides'][side]['discard'] if c['instance_id'] == target)
            text += f" → 回收「{recovered['name']}」"
        return text
    if kind == 'cycle':
        return f"弃一抽一：「{_card(state, side, action['card_id'])['name']}」"
    if kind == 'mulligan':
        return f"替换 {len(action['card_ids'])} 张起手牌" if action['card_ids'] else '保留全部起手牌'
    if kind == 'choose_cards':
        return f"调度 {len(action['card_ids'])} 张手牌"
    if kind == 'choose':
        card = next(c for c in state['pending_choice']['cards'] if c['instance_id'] == action['choice_id'])
        return f"选择「{card['name']}」"
    return {'end_turn': '结束回合', 'concede': '认输'}.get(kind, kind)


def preview(state, side, action):
    kind = action['type']
    cost = 1 if kind in ('attack', 'cycle') else 0
    card = _card(state, side, action['card_id']) if kind == 'play_card' else None
    if card:
        cost = 0 if card_is_free_instant(state, side, card) else card_cost(state, side, card)
    payment = normal_attack_payment(state, side, action['character_id']) if kind == 'attack' else None
    if payment is not None:
        cost = payment['ap']
    value = {'ap_cost': cost}
    selected_option = next((o for o in ((card or {}).get('play_options') or []) if o['id'] == action.get('option_id')), None)
    if selected_option:
        value['所选效果'] = selected_option['description']
        if selected_option.get('cost_label'): value['额外消耗'] = selected_option['cost_label']
    if payment and payment.get('resource'):
        value['额外消耗'] = '3 点金谷' if payment['resource'] == 'jingu' else '本回合瞬发机会'
    if kind != 'attack' and not (card and card['type'] == 'battle'):
        return value
    cid = action.get('character_id') or card['character_id']
    # Only combat is previewed. Replace private opposing hands and unknown decks;
    # public counts still drive shield, fatigue and hand-limit behavior.
    sandbox = deepcopy(state)
    def unknown_card(ps, zone, i):
        return {'instance_id': f'preview-{ps}-{zone}-{i}', 'card_id': 'unknown',
                'character_id': 'unknown', 'name': '未公开牌', 'copy': False,
                'type': 'tactic', 'cost': 999}
    for ps in SIDES:
        team = sandbox['sides'][ps]
        team['deck'] = [unknown_card(ps, 'deck', i) for i in range(len(team['deck']))]
        if ps != side:
            revealed = set(team.get('revealed_ids') or [])
            # Keep public hand counts and known cards, but never simulate a
            # response based on an unrevealed opposing card or private gold mark.
            team['hand'] = [card if card.get('copy') or card['instance_id'] in revealed
                            else unknown_card(ps, 'hand', i)
                            for i, card in enumerate(team['hand'])]
            for card in team['hand']:
                card.pop('jingu_mark', None)
    before = sandbox['event_seq']
    projected = apply_action(sandbox, side, action)
    events = [e for e in projected['events'] if e['seq'] > before]
    attacks = [e for e in events if e['type'] == 'attack']
    harmony = next((e for e in events if e['type'] == 'harmony'), None)
    previous = state['sides'][side]['front']
    payer = previous or state['sides'][side].get('last_front')
    enters = any(e['type'] == 'enter' and e.get('actor') == f'{side}:{cid}' for e in events)
    value.update(will_switch=enters and previous != cid, harmony=harmony['text'] if harmony else '不触发',
                 入场=hero(state, side, cid)['name'] if enters else '不入场',
                 替换=hero(state, side, previous)['name'] if enters and previous and previous != cid else '无')
    if harmony and payer and payer != cid:
        value['消耗环合'] = hero(state, side, payer)['name']
    elif previous != cid:
        value['消耗环合'] = '无'
    if attacks:
        target_side, target_id = attacks[0]['target'].split(':', 1)
        value.update(attack=attacks[0]['amount'], counter=attacks[0]['counter'],
                     攻击目标=name(state, target_side, target_id),
                     攻击次数=len(attacks))
    flowers = sum(e['amount'] for e in events if e['type'] == 'flower')
    if flowers:
        value['创生花伤害'] = flowers
    value['自身生命变化'] = hero(projected, side, cid)['hp'] - hero(state, side, cid)['hp']
    value['对手玩家生命变化'] = projected['sides'][other(side)]['hp'] - state['sides'][other(side)]['hp']
    return value


def observe(state, side, after_seq=None, *, include_previews=True):
    from .escalation import public_progress, energy_cap, harmony_cap, ultimate_remaining
    if side not in SIDES:
        raise ValueError('Unknown viewer')
    if needs_hydration(state):
        state = clone_state(state)
    actions = legal_actions(state, side)
    playable = {a['card_id'] for a in actions if a['type'] == 'play_card'}
    result = {key: deepcopy(state[key]) for key in ('rules_version', 'version', 'phase', 'active_side', 'first_side', 'turn', 'winner', 'reason')}
    result.update(escalation=public_progress(state), viewer_side=side, is_my_turn=(side == acting_side(state) or
                  state['phase'] == 'mulligan' and side not in state['mulligan_done']), sides={},
                  legal_actions=[{'action': deepcopy(a), 'entity_action': entity_action(state, side, a), 'label': action_label(state, side, a),
                                  'preview': preview(state, side, a) if include_previews else {},
                                  'interaction': interaction_for_action(state, side, a) if include_previews else {}} for a in actions],
                  events=[sanitize_public_event(event, side) for event in deepcopy(state['events'][-200:])] if include_previews else [],
                  logs=[], presentation=presentation_bundle(state, side, after_seq=after_seq) if include_previews else {})
    for event in result['events']:
        result['logs'].append(event['text'])
    for ps in SIDES:
        source = state['sides'][ps]
        marks = harmony_marks(state, ps)
        team = {k: deepcopy(source[k]) for k in ('entity_id', 'name', 'hp', 'shield', 'ap', 'normal_attack_available',
                'ultimate_available',
                'front', 'last_front', 'turn_count', 'fatigue', 'records', 'used', 'harmonized',
                'extra_flower', 'front_debuff') if k in source}
        team['life_locked'] = bool(source.get('life_locked'))
        team['genesis_player_hits'] = int(source.get('genesis_player_hits', 0))
        team['genesis_turn'] = source.get('genesis_turn')
        if ps == side and source.get('deck_visible_turn') == state['turn'] and state['active_side'] == ps:
            team['visible_deck'] = [public_card(c) for c in source['deck']]
        team['ultimate_remaining'] = ultimate_remaining(state, ps)
        team['harmony_available'] = marks['available']
        team['effect_markers'] = team_effect_markers(source)
        chars = []
        for cid in team_order(state, ps):
            h = source['characters'][cid]
            flags = compatibility_flags(h, state, ps)
            c = {k: deepcopy(h[k]) for k in ('entity_id', 'id', 'name', 'attribute', 'hp', 'max_hp', 'shield', 'harmony',
                 'energy', 'beast_fangs', 'intimidation', 'extra_attacks', 'protagonist_aura', 'jingu', 'arid', 'truth_keys', 'resource_name', 'faction', 'summoned', 'awakened', 'ultimate_turns', 'growth', 'family', 'gaze', 'dream', 'down_turns',
                 'avatar', 'portrait', 'passive', 'awakened_passive', 'flavor') if k in h}
            from app.modules.card_game.content.duel_v2.tutorial import lookup_card
            form = lookup_card(h['shape']) if h['shape'] else None
            from .equipment import stat_layers, public_equipment
            c['stat_layers'] = stat_layers(h)
            c['equipment'] = public_equipment(h)
            c['effects'] = public_effects(h)
            c['attack_breakdown'] = attack_value(h, state, ps, explain=True)
            c.update(ultimate_expires=ultimate_spec(cid)['expires'], attack=c['attack_breakdown']['value'], shape=(form or {}).get('name') if form else None,
                     shape_id=h['shape'], shape_description=(form or {}).get('description') if form else None,
                     harmony_max=harmony_cap(state), energy_max=energy_cap(state, h), gaze=h.get('gaze', 0), dream=h.get('dream', 0),
                     next_attack_bonus=flags.get('next_attack', 0),
                     return_after_sortie=flags.get('return_after', False),
                     brand=flags.get('brand', 0),
                     nightmare=flags.get('nightmare', 0),
                     collapse_count=flags.get('collapse_count', 0),
                     collapse_max=int(flags.get('collapse_max') or 5),
                     collapse=bool(flags.get('collapse')),
                     pact=bool(flags.get('pact')),
                     atk_debuff=int(flags.get('atk_debuff') or 0),
                     atk_buff=int(flags.get('atk_buff') or 0),
                     form_attack=int(flags.get('form_attack') or 0),
                     harmony_source=marks['payer'] == cid,
                     harmony_ready=cid in marks['ready'])
            c['effect_markers'] = character_effect_markers(h, state, ps)
            c['damage_immune'] = damage_immune(state, ps, cid)
            c['damage_limit'] = damage_limit(state, ps, cid)
            chars.append(c)
        team['characters'] = chars
        team['deck_count'] = len(source['deck'])
        # Cards being inspected remain counted as deck cards until disposition.
        pending = state['pending_choice']
        if pending and pending['side'] == ps and pending['kind'] == 'inspect_top':
            team['deck_count'] += len(pending['cards'])
        team['hand_count'] = len(source['hand'])
        revealed = set(source.get('revealed_ids') or [])
        built = []
        for physical in source['hand']:
            raw = effective_card(state, ps, physical)
            if ps == side or raw.get('copy') or raw.get('instance_id') in revealed:
                card = public_card(raw, state, ps)
                if ps == side and card_plays_instant(state, ps, raw):
                    card['instant'] = True
                if raw.get('instance_id') in revealed:
                    card['revealed'] = True
            else:
                card = {'hidden': True}
            if ps == side and raw['instance_id'] not in playable:
                if state['phase'] != 'playing' or state['active_side'] != side:
                    reason = '等待行动阶段'
                elif ((not source['characters'][raw['character_id']]['hp']
                       or source['characters'][raw['character_id']].get('down_turns'))
                      and not raw.get('playable_downed')):
                    reason = '所属角色倒地'
                elif collapsed(source['characters'][raw['character_id']]):
                    reason = '所属角色倾陷'
                elif source['ap'] < raw['cost']:
                    reason = f"还差 {raw['cost'] - source['ap']} 点行动力"
                else:
                    reason = '没有合法目标'
                card['unavailable_reason'] = reason
            built.append(card)
        if ps == side:
            built = _sort_hand(built, team_order(state, ps))
        team['hand'] = built
        team['discard'] = [public_card(c, state, ps) for c in source['discard']]
        result['sides'][ps] = team
    pending = state['pending_choice']
    result['pending_choice'] = None
    if pending:
        result['pending_choice'] = {'side': pending['side'], 'kind': pending['kind'],
            'prompt': pending['prompt'] if pending['side'] == side else '对手正在选择',
            'max_select': pending.get('max_select') if pending['side'] == side else None,
            'choices': [{'id': c['instance_id'], 'label': c['name'], 'card': public_card(c, state, ('b' if pending['side']=='a' else 'a') if pending['kind']=='enemy_hand' else pending['side'])}
                        for c in pending['cards']] if pending['side'] == side else []}
    op = state['operation']
    result['resolving_card'] = public_card(op['card'], state, op['side']) if op and op['card'] else None
    from .tutorial import is_tutorial, project_tutorial, strip_hidden_fields
    if is_tutorial(state):
        tutorial = project_tutorial(state, side)
        result['tutorial'] = tutorial
        strip_hidden_fields(result, tutorial)
    else:
        from .policy_state import public_policy_state
        result['policy_state'] = public_policy_state(state)
    return result


def choose_action(state, side):
    """Small deterministic rule opponent, restricted to its public observation."""
    observation = observe(state, side)
    held = {c.get('instance_id'): c for c in observation['sides'][side]['hand']}
    entries = [e for e in observation['legal_actions']
               if not held.get(e['action'].get('card_id'), {}).get('training_excluded')]
    if not entries:
        raise ValueError('No legal action')
    for kind in ('mulligan', 'choose', 'choose_cards'):
        choice = next((e for e in entries if e['action']['type'] == kind), None)
        if choice:
            return choice['action']
    cards = {c['instance_id']: c for c in observation['sides'][side]['hand']}
    def score(entry):
        a, p = entry['action'], entry['preview']
        kind = a['type']
        if kind == 'concede':
            return -1000
        if kind == 'end_turn':
            return -100
        if kind == 'ultimate':
            return 6
        if kind == 'cycle':
            return 1
        if kind == 'attack' or kind == 'play_card' and cards[a['card_id']]['type'] == 'battle':
            return p.get('attack', 0) * p.get('攻击次数', 1) + p.get('创生花伤害', 0) - p.get('counter', 0) * .4 + 3
        return 2
    return deepcopy(max(entries, key=score)['action'])
