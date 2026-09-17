"""Public pending-effect labels shared by live observations and replay patches."""


def _marker(key, name, timing, description, clears_on_down=True):
    return dict(id=key, name=name, timing=timing, description=description,
                clears_on_down=clears_on_down)


def character_effect_markers(character):
    if character.get('down_turns') or character.get('hp', 0) <= 0:
        return []
    flags = character.get('flags') or {}
    markers = []
    for attribute, value in (character.get('next_damage_resistance') or {}).items():
        markers.append(_marker('next_resistance', attribute + '抗性', '下次受到该属性伤害',
                               f'{attribute}属性伤害抗性 {value:+d}'))
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
    if flags.get('next_attack'):
        markers.append(_marker('next_attack', '攻击强化', '下次主动攻击',
                               f"攻击 +{flags['next_attack']}"))
    if flags.get('return_after'):
        markers.append(_marker('return_after', '忆昔告远', '本回合下次出击后',
                               '仍存活且在战斗区时返回备战区，然后抽 1 张牌'))
    return markers


def team_effect_markers(team):
    markers = [_marker('next_resistance', attribute + '抗性', '玩家下次受到该属性伤害',
                       f'{attribute}属性伤害抗性 {value:+d}', False)
               for attribute, value in (team.get('next_damage_resistance') or {}).items()]
    if not team.get('extra_genesis_pending'):
        return markers
    actor = (team.get('characters') or {}).get(team.get('extra_genesis_actor')) or {}
    return markers + [_marker('extra_genesis', '错误的门', '下个己方回合开始',
                    '伊洛伊倒地后保留；原环合角色' + actor.get('name', '') + '须存活，再触发一次创生', False)]
