"""Attack layers shared by ordinary, card and deferred attacks.

Panel -> sortie frame -> card set -> hit additions -> zone -> zero floor.
No damage, targeting, resource gains or persistent mutation occur here.
"""
from .modifiers import contribution, evaluate
from .state import attack_value, is_player, is_esper, zone_attack_penalty


def attack_result(c, spec, index, target, enemy, target_id):
    h, op = c.character, c.operation
    entries = []
    def add(key, value, layer=0, op='add'):
        if value or op == 'set':
            entries.append(contribution(key, 'attack.hit', value, layer=layer, op=op,
                                       source_entity_id=c.entity_id, label=key))
    add('角色面板', attack_value(h, c.state, c.side, include_zone=False), -100, 'set')
    add('本次出击', int(op.get('sortie_attack_bonus', 0)), -90)
    if 'card_set_attack' in op:
        add('战斗牌覆写', op['card_set_attack'], -80, 'set')
    bonus = spec.get('bonus', 0)
    if is_player(target_id) and h['awakened'] and 'awake_player_bonus' in spec:
        bonus = spec['awake_player_bonus']
    add('本次攻击', bonus)
    low = bool(is_esper(target_id) and target['hp'] <= spec.get('low_hp_max', 3))
    if low:
        add('目标生命条件', spec.get('low_hp_bonus', 0))
    add('兑现', op.get('redeem_bonus', 0))
    add('战斗牌加攻', int(op.get('card_bonus') or 0))
    if index == 0:
        add('战斗准备加成', op['first_bonus'], -90)
    if spec.get('harmony_bonus') and op.get('harmony'):
        add('入场环合加攻', spec['harmony_bonus'])
    add('战斗区减攻', -zone_attack_penalty(c.state, c.side, c.cid), 1000)
    query = dict(attacker_id=c.cid, target=target, target_id=target_id,
                 spec=spec, index=index, operation=op)
    result = evaluate(c.state, c.side, h, 'attack.hit', 0, query=query, extra=entries)
    result['value'] = max(0, int(result['value']))
    result['low_target'] = low
    return result
