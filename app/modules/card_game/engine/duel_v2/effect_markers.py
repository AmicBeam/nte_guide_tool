"""Public pending-effect labels shared by live observations and replay patches."""


def _marker(key, name, timing, description, clears_on_down=True, kind="buff"):
    return dict(id=key, name=name, timing=timing, description=description,
                clears_on_down=clears_on_down, kind=kind)


def character_effect_markers(character, state=None, side=None):
    if character.get('down_turns') or character.get('hp', 0) <= 0:
        return []
    from .modifiers import compatibility_flags
    flags = compatibility_flags(character, state, side)
    markers = []
    if flags.get('pact'):
        markers.append(_marker('pact', '枚约', '九原终结清算时',
                               '九原终结对该目标额外造成 2 点伤害，并消耗枚约', kind='debuff'))
    for attribute, value in (character.get('next_damage_resistance') or {}).items():
        markers.append(_marker('next_resistance', attribute + '抗性', '下次受到该属性伤害',
                               f'{attribute}属性伤害抗性 {value:+d}', kind='debuff' if value < 0 else 'buff'))
    from .modifiers import all_effects, remaining
    for effect in all_effects(character):
        if not effect.get('show_marker') or effect.get('visibility') != 'public':
            continue
        left = remaining(effect, state, side)
        timing = (('本回合下次受到对应属性伤害' if (effect.get('expiry') or {}).get('phase') == 'turn_end' else '下次受到对应属性伤害') if effect.get('consume_on') == 'damage' else
                  '下次主动攻击' if effect.get('consume_on') == 'attack' else
                  f'剩余 {left} 个己方回合' if left is not None else '指定时点到期')
        amount = effect['value']
        markers.append(_marker(effect['effect_id'], effect['label'], timing,
                               f"攻击 {amount:+d}" if effect['property'].startswith('attack.') and isinstance(amount, int)
                               else effect['label'], 'down' in effect['clear_on'],
                               kind='debuff' if amount < 0 else 'buff'))
    if flags.get('free_normal_attack_turn'):
        markers.append(_marker('free_normal_attack', '普通出击免费', '本回合',
                               '普通出击不消耗行动力，仍需普通出击机会'))
    if flags.get('energy_next'):
        markers.append(_marker('energy_next', '终结 · 回能', '下个己方回合开始',
                               '己方存活异能者各获得 1 点能量'))
    if flags.get('pending_atk'):
        markers.append(_marker('pending_atk', '第一直觉', '下个己方回合开始',
                               f"攻击 +{flags['pending_atk']}"))
    if flags.get('next_bonus') or flags.get('next_shield'):
        markers.append(_marker('next_battle', '策略特性：可激怒', '下次战斗',
                               f"攻击 +{flags.get('next_bonus', 0)}、护盾 +{flags.get('next_shield', 0)}"))
    if flags.get('next_followup'):
        markers.append(_marker('next_followup', '小弟遍天下', '本回合下次战斗',
                               f"额外造成 {flags['next_followup']} 点追击伤害"))

    if flags.get('return_after'):
        markers.append(_marker('return_after', '忆昔告远', '本回合下次出击后',
                               '仍存活且在战斗区时返回备战区，然后抽 1 张牌'))
    return markers


def team_effect_markers(team):
    from .modifiers import all_effects
    markers = [_marker('next_resistance', attribute + '抗性', '玩家下次受到该属性伤害',
                       f'{attribute}属性伤害抗性 {value:+d}', False)
               for attribute, value in (team.get('next_damage_resistance') or {}).items()]
    for e in all_effects(team):
        if e['property'] == 'damage.resistance' and e.get('visibility') == 'public':
            markers.append(_marker(e['effect_id'], e['label'],
                                   '本回合下次受到对应属性伤害' if (e.get('expiry') or {}).get('phase') == 'turn_end' else '下次受到对应属性伤害',
                                   e['label'], False, kind='debuff' if e['value'] < 0 else 'buff'))
    if team.get('empty_draw_wins'):
        markers.append(_marker('empty_draw_wins', '热爱异象的观察员', '无牌可抽时',
                               '无牌可抽时改为获胜；埃德嘉倒地后仍保留', False))
    if team.get('skip_normal_attack'):
        markers.append(_marker('skip_normal_attack', '普通出击减少', '下个己方回合', '普通出击机会 -1', False))
    if not team.get('extra_genesis_pending'):
        return markers
    actor = (team.get('characters') or {}).get(team.get('extra_genesis_actor')) or {}
    return markers + [_marker('extra_genesis', '错误的门', '下个己方回合开始',
                    '伊洛伊倒地后保留；原环合角色' + actor.get('name', '') + '须存活，再触发一次创生', False)]
