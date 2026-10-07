"""Yi's delay attachments. Durations and damage are user tabletop parameters."""
from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    alive, damage, finished, front_debuff, hero, other, team_order,
)


def _fangs(c, turns):
    c.character['beast_fangs'] = max(int(c.character.get('beast_fangs', 0)), turns)
    c.history(f'生成「兽牙影刺」，持续 {turns} 个己方回合。')


def _battle(c):
    _fangs(c, 4)
    c.sortie()


def _long(c):
    _fangs(c, 4)
    c.draw(1)


def _free(c):
    _fangs(c, 2)
    c.draw(1)


def _skip(c):
    c.state['sides'][other(c.side)]['skip_normal_attack'] = True
    c.history('对方下个回合的普通出击机会 -1。')


def _weaken(c):
    foe, cid = c.operation['target_id'].split(':', 1)
    count = int(c.state['sides'][c.side].get('delay_turns', 0))
    if not count or finished(c.state) or not alive(c.state, foe, cid):
        return
    target = hero(c.state, foe, cid)
    from app.modules.card_game.engine.duel_v2.modifiers import add_effect, deadline
    add_effect(target, 'yi.attack_debuff', 'attack.panel', -count, source_entity_id=c.entity_id,
               source_key='I06', label='无谓形貌的立身',
               expiry=deadline(c.state, foe, 2, 'turn_start'))
    c.history(f"{target['name']}攻击 -{count}，持续 2 个对方回合。", target_side=foe, target_id=cid)


def _weapon_damage(c):
    if c.character.get('shape') != 'I07' or not c.character.get('beast_fangs'):
        return
    foe = other(c.side)
    target = c.state['sides'][foe]['front']
    if target and not hero(c.state, foe, target).get('summoned'):
        damage(c.state, foe, target, 1, source=f'{c.side}:{c.cid}', kind='attachment',
               damage_class='attachment', reason='「潜影追击」触发')


@register
class Yi:
    id = 'yi'
    text_name = '翳'
    effects = {'I01': _battle, 'I02': _battle, 'I03': _long, 'I04': _free,
               'I05': _skip, 'I06': _weaken, 'I07': set_shape, 'I08': set_shape}
    target_policies = {'I06': 'living_enemy_esper'}

    @staticmethod
    def on_ultimate(c):
        c.character['beast_fangs'] = int(c.character.get('beast_fangs', 0)) + 1
        c.history(f"「兽牙影刺」持续时间 +1 个己方回合（剩余 {c.character['beast_fangs']} 个己方回合）。")

    @staticmethod
    def attachment_damage_bonus(c):
        return 1 if c.character['awakened'] else 0

    @staticmethod
    def pause_delay(c, current=False, **_data):
        return bool(current or c.character.get('beast_fangs', 0))

    @staticmethod
    def on_turn_countdown(c):
        if c.character.get('beast_fangs', 0):
            c.character['beast_fangs'] -= 1
            c.history(f"「兽牙影刺」剩余 {c.character['beast_fangs']} 个己方回合。")

    on_turn_begin = staticmethod(_weapon_damage)
    on_enemy_turn_begin = staticmethod(_weapon_damage)

    @staticmethod
    def after_hits(c, **_data):
        status = front_debuff(c.state, other(c.side)).get('delay')
        # User tabletop tuning (2026-09-21), not Everness combat values.
        if status and status.get('by') == c.side and int(status.get('left', 0)) < 2:
            status['left'] = min(2, int(status.get('left', 0)) + 1)
            c.history(f"翳的攻击延长延滞至 {status['left']} 个己方回合。")

    @staticmethod
    def can_play(c, current=True, card=None, **_data):
        if card and card.get('card_id') == 'I05':
            return bool(current and front_debuff(c.state, other(c.side)).get('delay'))
        return current

    @staticmethod
    def makes_instant(state, side, card):
        h = hero(state, side, 'yi')
        return bool(h['hp'] > 0 and card.get('card_id') == 'I08'
                    and front_debuff(state, other(side)).get('delay'))
