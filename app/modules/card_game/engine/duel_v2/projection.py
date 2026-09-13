"""Viewer-specific projection. Never expose decks, RNG, or private choices."""
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS
from .flow import acting_side, apply_action, card_plays_instant, legal_actions
from .presentation import (interaction_for_action, presentation_bundle, public_card,
                           sanitize_public_event)
from .combat import harmony_marks
from .state import SIDES, attack_value, collapsed, energy_max as character_energy_max, hero, name, other, team_order


def _card(state, side, instance_id):
    return next(c for c in state['sides'][side]['hand'] if c['instance_id'] == instance_id)


TYPE_ORDER = {'tactic': 0, 'battle': 1, 'form': 2}


def _sort_hand(cards, character_ids):
    index = {cid: i for i, cid in enumerate(character_ids)}
    return sorted(cards, key=lambda card: (
        index.get(card.get('character_id'), len(index)),
        TYPE_ORDER.get(card.get('type'), 9),
    ))


def action_label(state, side, action):
    kind = action['type']
    if kind == 'attack':
        return f"{hero(state, side, action['character_id'])['name']}出击"
    if kind == 'ultimate':
        return f"终结{hero(state, side, action['character_id'])['name']}"
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
    cost = 1 if kind in ('attack', 'cycle') else 0
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
    payer = previous or state['sides'][side].get('last_front')
    value.update(will_switch=previous != cid, harmony=harmony['text'] if harmony else '不触发',
                 入场=hero(state, side, cid)['name'],
                 替换=hero(state, side, previous)['name'] if previous and previous != cid else '无')
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


def observe(state, side, after_seq=None):
    if side not in SIDES:
        raise ValueError('Unknown viewer')
    actions = legal_actions(state, side)
    playable = {a['card_id'] for a in actions if a['type'] == 'play_card'}
    result = {key: deepcopy(state[key]) for key in ('rules_version', 'version', 'phase', 'active_side', 'first_side', 'turn', 'winner', 'reason')}
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
        marks = harmony_marks(state, ps)
        team = {k: deepcopy(source[k]) for k in ('name', 'hp', 'shield', 'ap', 'normal_attack_available',
                'ultimate_available',
                'front', 'last_front', 'turn_count', 'fatigue', 'records', 'used', 'harmonized',
                'extra_flower', 'front_debuff') if k in source}
        team['harmony_available'] = marks['available']
        chars = []
        for cid in team_order(state, ps):
            h = source['characters'][cid]
            flags = h.get('flags') or {}
            c = {k: deepcopy(h[k]) for k in ('id', 'name', 'attribute', 'hp', 'max_hp', 'shield', 'harmony',
                 'energy', 'awakened', 'ultimate_turns', 'growth', 'family', 'gaze', 'dream', 'down_turns',
                 'avatar', 'portrait', 'passive', 'awakened_passive') if k in h}
            c.update(attack=attack_value(h), shape=CARDS[h['shape']]['name'] if h['shape'] else None,
                     shape_id=h['shape'], shape_description=CARDS[h['shape']]['description'] if h['shape'] else None,
                     harmony_max=2, energy_max=character_energy_max(h), gaze=h.get('gaze', 0), dream=h.get('dream', 0),
                     next_attack_bonus=flags.get('next_attack', 0),
                     return_after_sortie=flags.get('return_after', False),
                     brand=flags.get('brand', 0),
                     nightmare=flags.get('nightmare', 0),
                     collapse_count=flags.get('collapse_count', 0), collapse_max=5,
                     collapse=bool(flags.get('collapse')),
                     pact=bool(flags.get('pact')),
                     atk_debuff=int(flags.get('atk_debuff') or 0),
                     atk_buff=int(flags.get('atk_buff') or 0),
                     form_attack=int(flags.get('form_attack') or 0),
                     harmony_source=marks['payer'] == cid,
                     harmony_ready=cid in marks['ready'])
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
        for raw in source['hand']:
            if ps == side or raw.get('copy') or raw.get('instance_id') in revealed:
                card = public_card(raw)
                if ps == side and card_plays_instant(state, ps, raw):
                    card['instant'] = True
                if raw.get('instance_id') in revealed and ps != side:
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
        if kind == 'ultimate':
            return 6
        if kind == 'cycle':
            return 1
        if kind == 'attack' or kind == 'play_card' and cards[a['card_id']]['type'] == 'battle':
            return p.get('attack', 0) * p.get('攻击次数', 1) + p.get('创生花伤害', 0) - p.get('counter', 0) * .4 + 3
        return 2
    return deepcopy(max(entries, key=score)['action'])
