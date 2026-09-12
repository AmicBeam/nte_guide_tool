"""Viewer-specific projection. Never expose decks, RNG, or private choices."""
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS
from .flow import acting_side, apply_action, legal_actions
from .presentation import (interaction_for_action, presentation_bundle, public_card,
                           sanitize_public_event)
from .state import ORDER, SIDES, attack_value, hero, name, other


def _card(state, side, instance_id):
    return next(c for c in state['sides'][side]['hand'] if c['instance_id'] == instance_id)


def action_label(state, side, action):
    kind = action['type']
    if kind == 'attack':
        return f"{hero(state, side, action['character_id'])['name']}出击"
    if kind == 'awaken':
        return f"觉醒{hero(state, side, action['character_id'])['name']}"
    if kind == 'play_card':
        text = f"使用「{_card(state, side, action['card_id'])['name']}」"
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
    if kind == 'choose':
        card = next(c for c in state['pending_choice']['cards'] if c['instance_id'] == action['choice_id'])
        return f"选择「{card['name']}」"
    return {'end_turn': '结束回合', 'concede': '认输'}.get(kind, kind)


def preview(state, side, action):
    kind = action['type']
    cost = 1 if kind in ('attack', 'awaken', 'cycle') else 0
    card = _card(state, side, action['card_id']) if kind == 'play_card' else None
    if card:
        cost = card['cost']
    value = {'ap_cost': cost}
    if kind != 'attack' and not (card and card['type'] == 'battle'):
        return value
    cid = action.get('character_id') or card['character_id']
    # Only combat is previewed. Replace all hidden deck cards before simulation;
    # shield, fatigue and hand-limit behavior still works, without future card data.
    sandbox = deepcopy(state)
    for ps in SIDES:
        sandbox['sides'][ps]['deck'] = [
            {'instance_id': f'preview-{ps}-{i}', 'card_id': 'unknown', 'character_id': 'unknown',
             'name': '未公开牌', 'copy': False}
            for i in range(len(sandbox['sides'][ps]['deck']))]
    before = sandbox['event_seq']
    projected = apply_action(sandbox, side, action)
    events = [e for e in projected['events'] if e['seq'] > before]
    attacks = [e for e in events if e['type'] == 'attack']
    harmony = next((e for e in events if e['type'] == 'harmony'), None)
    previous = state['sides'][side]['front']
    value.update(will_switch=previous != cid, harmony=harmony['text'] if harmony else '不触发',
                 入场=hero(state, side, cid)['name'],
                 替换=hero(state, side, previous)['name'] if previous and previous != cid else '无')
    if attacks:
        target_side, target_id = attacks[0]['target'].split(':', 1)
        value.update(attack=attacks[0]['amount'], counter=attacks[0]['counter'],
                     攻击目标=name(state, target_side, None if target_id == 'player' else target_id),
                     攻击次数=len(attacks))
    flowers = sum(e['amount'] for e in events if e['type'] == 'flower')
    if flowers:
        value['创生花伤害'] = flowers
    value['自身生命变化'] = hero(projected, side, cid)['hp'] - hero(state, side, cid)['hp']
    value['对手玩家生命变化'] = projected['sides'][other(side)]['hp'] - state['sides'][other(side)]['hp']
    return value


def observe(state, side, after_seq=None):
    if side not in SIDES:
        raise ValueError('Unknown viewer')
    actions = legal_actions(state, side)
    playable = {a['card_id'] for a in actions if a['type'] == 'play_card'}
    result = {key: deepcopy(state[key]) for key in ('rules_version', 'version', 'phase', 'active_side', 'turn', 'winner', 'reason')}
    result.update(viewer_side=side, is_my_turn=(side == acting_side(state) or
                  state['phase'] == 'mulligan' and side not in state['mulligan_done']), sides={},
                  legal_actions=[{'action': deepcopy(a), 'label': action_label(state, side, a),
                                  'preview': preview(state, side, a),
                                  'interaction': interaction_for_action(state, side, a)} for a in actions],
                  events=[sanitize_public_event(event, side) for event in deepcopy(state['events'][-200:])],
                  logs=[], presentation=presentation_bundle(state, side, after_seq=after_seq))
    for event in result['events']:
        result['logs'].append(event['text'])
    for ps in SIDES:
        source = state['sides'][ps]
        team = {k: deepcopy(source[k]) for k in ('name', 'hp', 'shield', 'ap', 'normal_attack_available',
                'front', 'last_front', 'turn_count', 'fatigue', 'records', 'used', 'harmonized', 'extra_flower')}
        chars = []
        for cid in ORDER:
            h = source['characters'][cid]
            c = {k: deepcopy(h[k]) for k in ('id', 'name', 'attribute', 'hp', 'max_hp', 'shield', 'harmony',
                 'energy', 'awakened', 'growth', 'down_turns', 'avatar', 'portrait', 'passive', 'awakened_passive')}
            c.update(attack=attack_value(h), shape=CARDS[h['shape']]['name'] if h['shape'] else None,
                     shape_id=h['shape'], shape_description=CARDS[h['shape']]['description'] if h['shape'] else None,
                     harmony_max=2, energy_max=4,
                     next_attack_bonus=h['flags'].get('next_attack', 0),
                     return_after_sortie=h['flags'].get('return_after', False),
                     slow=deepcopy(h['flags'].get('slow')), burn=h['flags'].get('burn', 0),
                     star=h['flags'].get('star'))
            chars.append(c)
        team['characters'] = chars
        team['deck_count'] = len(source['deck'])
        # Cards being inspected remain counted as deck cards until disposition.
        pending = state['pending_choice']
        if pending and pending['side'] == ps and pending['kind'] == 'inspect_top':
            team['deck_count'] += len(pending['cards'])
        team['hand_count'] = len(source['hand'])
        team['hand'] = []
        for raw in source['hand']:
            card = public_card(raw) if ps == side or raw.get('copy') else {'hidden': True}
            if ps == side and raw['instance_id'] not in playable:
                if state['phase'] != 'playing' or state['active_side'] != side:
                    reason = '等待行动阶段'
                elif not source['characters'][raw['character_id']]['hp']:
                    reason = '所属角色倒地'
                elif source['ap'] < raw['cost']:
                    reason = f"还差 {raw['cost'] - source['ap']} 点行动力"
                else:
                    reason = '没有合法目标'
                card['unavailable_reason'] = reason
            team['hand'].append(card)
        team['discard'] = [public_card(c) for c in source['discard']]
        result['sides'][ps] = team
    pending = state['pending_choice']
    result['pending_choice'] = None
    if pending:
        result['pending_choice'] = {'side': pending['side'], 'kind': pending['kind'],
            'prompt': pending['prompt'] if pending['side'] == side else '对手正在选择',
            'choices': [{'id': c['instance_id'], 'label': c['name'], 'card': public_card(c)}
                        for c in pending['cards']] if pending['side'] == side else []}
    op = state['operation']
    result['resolving_card'] = public_card(op['card']) if op and op['card'] else None
    return result


def choose_action(state, side):
    """Small deterministic rule opponent, restricted to its public observation."""
    observation = observe(state, side)
    entries = observation['legal_actions']
    if not entries:
        raise ValueError('No legal action')
    for kind in ('mulligan', 'choose'):
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
        if kind == 'awaken':
            return 6
        if kind == 'cycle':
            return 1
        if kind == 'attack' or kind == 'play_card' and cards[a['card_id']]['type'] == 'battle':
            return p.get('attack', 0) * p.get('攻击次数', 1) + p.get('创生花伤害', 0) - p.get('counter', 0) * .4 + 3
        return 2
    return deepcopy(max(entries, key=score)['action'])
