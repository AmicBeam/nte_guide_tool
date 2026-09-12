"""Sortie, harmony, simultaneous damage, and character combat passives."""
from app.modules.card_game.engine.events import GameEvent
from .state import (alive, attack_value, card_instance, add_hand, consume_once, damage,
                    finished, front_target, hero, knockdowns, log, move_out, name,
                    other, shield, victory)

PAIRS = {frozenset(('光', '灵')): '创生', frozenset(('光', '相')): '延滞',
         frozenset(('暗', '咒')): '浊燃', frozenset(('暗', '魂')): '黯星'}


def harmony_kind(state, side, cid):
    team = state['sides'][side]
    if team['front'] == cid or hero(state, side, cid)['harmony'] < 2:
        return None
    previous = team['front'] or team['last_front']
    if not previous or previous == cid:
        return None
    kind = PAIRS.get(frozenset((hero(state, side, previous)['attribute'], hero(state, side, cid)['attribute'])))
    if kind != '创生' and not state['sides'][other(side)]['front']:
        return None
    return kind


def enter(c):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    team = s['sides'][side]
    if team['front'] == cid:
        return
    kind = harmony_kind(s, side, cid)
    if team['front']:
        move_out(s, side, team['front'])
    team['front'] = cid
    log(s, f"{c.character['name']}进入战斗区。", 'enter', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}')
    if not kind:
        return
    c.character['harmony'] -= 2
    c.character['flags']['has_harmonized'] = True
    # This once-per-game fact must survive death.
    team['harmonized'][cid] = True
    op['harmony'] = kind
    log(s, f"{c.character['name']}消耗自己的 2 点环合值，触发{kind}。", 'harmony', side=side, source=cid,
        actor=f'{side}:{cid}', target=f'{side}:{cid}',
        before={'harmony': c.character['harmony'] + 2}, after={'harmony': c.character['harmony']})
    enemy, target = front_target(s, side)
    if kind == '创生':
        shield(s, side, cid, 2)
        op['first_bonus'] += 1
        j = hero(s, side, 'jiuyuan')
        flowers = (2 if j['awakened'] else 1) if j['hp'] > 0 else 0
        flowers += int(team['extra_flower'])
        team['extra_flower'] = False
        for _ in range(flowers):
            if finished(s) or not alive(s, side, cid):
                break
            target = front_target(s, side)
            from .presentation import next_group_id
            flower_group = next_group_id(s)
            damage(s, *target, 1, source=f'{side}:{cid}', kind='flower', group_id=flower_group)
            if not finished(s) and alive(s, side, 'jiuyuan'):
                c.for_character(side, 'jiuyuan').on_shape(GameEvent.V2_FLOWER_RESOLVED, target=target, entering=cid)
    elif kind == '延滞':
        hero(s, enemy, target)['flags']['slow'] = {'amount': 2, 'end': s['sides'][enemy]['turn_count'] + 1}
    elif kind == '浊燃':
        hero(s, enemy, target)['flags']['burn'] = 2
    elif kind == '黯星':
        hero(s, enemy, target)['flags']['star'] = s['sides'][enemy]['turn_count'] + 1
    if not finished(s) and alive(s, side, cid):
        c.on_shape(GameEvent.V2_HARMONY_RESOLVED, operation=op)


def redeem(c):
    s, side, op, card = c.state, c.side, c.operation, c.card
    team = s['sides'][side]
    if (c.cid != 'xun' or not card or card.get('copy') or not card['terminal']
            or card['type'] != 'battle' or card['character_id'] != 'xun'
            or not team['records'] or team['used'].get('redeem')
            or finished(s) or team['front'] != 'xun' or not alive(s, side, 'xun')):
        return
    records = team['records']
    team['records'] = []
    team['used']['redeem'] = True
    op['redeem_bonus'] = 2 if c.character['awakened'] else 1
    log(s, f"浔兑现 {len(records)} 条记录，本次出击所有攻击伤害 +{op['redeem_bonus']}。",
        'redeem', side=side, source='xun', actor=f'{side}:xun', amount=len(records))
    for record in records:
        add_hand(s, side, card_instance(s, record['card_id'], side, copy=True))
    c.on_shape(GameEvent.V2_RECORD_REDEEMED, operation=op, count=len(records))


def _slow(h):
    return h['flags'].pop('slow', {}).get('amount', 0)


def attack(c, spec, index):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    enemy, target_id = front_target(s, side)
    h = c.character
    target = hero(s, enemy, target_id) if target_id else s['sides'][enemy]
    bonus = spec.get('bonus', 0)
    if not target_id and h['awakened'] and 'awake_player_bonus' in spec:
        bonus = spec['awake_player_bonus']
    low = bool(target_id and target['hp'] <= 3)
    if low:
        bonus += spec.get('low_hp_bonus', 0)
    bonus += op.get('redeem_bonus', 0)
    if index == 0:
        bonus += op['first_bonus']
    bonus += h['flags'].pop('next_attack', 0)
    attack_amount = max(0, attack_value(h) + bonus - _slow(h))
    no_counter = spec.get('no_counter') or op.get('no_counter') or (spec.get('no_counter_harmony') and op['harmony'])
    counter = max(0, attack_value(target) - _slow(target)) if target_id else 0
    if no_counter:
        counter = 0
    bypass = bool(spec.get('bypass'))
    penetrate = (spec.get('penetrate_harmony') and op['harmony']) or (spec.get('penetrate_low_hp') and low)
    overflow = max(0, attack_amount - (0 if bypass else target['shield']) - target['hp']) if target_id and penetrate else 0
    participants = [(side, cid)] + ([(enemy, target_id)] if target_id else [])
    for participant in participants:
        pair = list(participant)
        if pair not in op['participants']:
            op['participants'].append(pair)
    from .presentation import next_group_id
    group_id = next_group_id(s)
    hit_index = index
    log(s, f"{h['name']}攻击{name(s, enemy, target_id)}：攻击 {attack_amount}，反击 {counter}。",
        'attack', side=side, source=cid, actor=f'{side}:{cid}',
        target=f'{enemy}:{target_id or "player"}', amount=attack_amount, attack=attack_amount,
        counter=counter, group_id=group_id, hit_index=hit_index)
    target_loss = damage(s, enemy, target_id, attack_amount, bypass=bypass, source=f'{side}:{cid}',
                         kind='combat', settle=False, group_id=group_id)
    self_loss = damage(s, side, cid, counter, source=f'{enemy}:{target_id}',
                       kind='combat', settle=False, group_id=group_id) if target_id else 0
    if overflow:
        damage(s, enemy, None, overflow, source=f'{side}:{cid}', kind='penetration',
               settle=False, group_id=group_id)
    knockdowns(s)
    victory(s)
    if finished(s):
        return
    # Attack-after shape effects occur per hit, after simultaneous death handling.
    for ps, pc, loss in [(side, cid, self_loss), (enemy, target_id, target_loss)]:
        if pc and loss > 0 and alive(s, ps, pc) and not finished(s):
            c.for_character(ps, pc).on_shape(GameEvent.V2_COMBAT_HP_LOST)


def sortie(c, **spec):
    s, side, cid, op = c.state, c.side, c.cid, c.operation
    if finished(s) or not alive(s, side, cid):
        return
    enter(c)
    redeem(c)
    for index in range(spec.get('hits', 1)):
        if finished(s) or not alive(s, side, cid) or s['sides'][side]['front'] != cid:
            break
        attack(c, spec, index)
    if finished(s) or not alive(s, side, cid):
        return
    if spec.get('shield_after_shape') and c.character['shape']:
        c.shield(spec['shield_after_shape'])
    if s['sides'][side]['front'] == cid:
        if cid == 'nanali' and consume_once(s, side, 'growth'):
            cap = 4 if c.character['awakened'] else 2
            if c.character['growth'] < cap:
                c.character['growth'] += 1
                log(s, f"娜娜莉获得 1 点成长，之后攻击为 {attack_value(c.character)}。",
                    'growth', side=side, source=cid, actor=f'{side}:{cid}',
                    before={'growth': c.character['growth'] - 1, 'attack': attack_value(c.character) - 1},
                    after={'growth': c.character['growth'], 'attack': attack_value(c.character)})
                c.on_shape(GameEvent.V2_GROWTH_GAINED)
        c.on_shape(GameEvent.V2_SORTIE_FINISHED)
        if not finished(s) and alive(s, side, cid) and c.character['flags'].pop('return_after', False):
            move_out(s, side, cid)
            c.draw(1)
