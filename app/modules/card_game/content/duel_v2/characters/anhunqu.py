"""安魂曲：噩梦与失谐强化。数值是桌游试玩参数。"""
from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import KITS, register
from app.modules.card_game.engine.duel_v2.continuous import apply_dot
from app.modules.card_game.engine.duel_v2.state import (
    add_hand, alive, card_instance, damage, front_debuff, hero, other,
)


def _stacks(c):
    foe = other(c.side)
    dot = (front_debuff(c.state, foe).get('dots') or {}).get('nightmare') or {}
    return int(dot.get('stacks') or 0) if dot.get('by') == c.side else 0


def _a01(c):
    side, cid = c.operation['target_id'].split(':', 1)
    c.character.setdefault('flags', {})['mimic'] = {'id': cid}
    name = hero(c.state, side, cid)['name']
    c.history(f'「不加班之宽赦」：本回合安魂曲的异能视为与{name}相同。')
    kit = KITS.get(cid)
    hook = getattr(kit, 'on_turn_begin', None)
    if hook is None:
        return
    awakened = c.character.get('awakened')
    c.character['awakened'] = False
    try:
        hook(c)
    finally:
        c.character['awakened'] = awakened


def _a02(c):
    stacks = _stacks(c)
    side, cid = c.operation['target_id'].split(':', 1)
    if stacks:
        damage(c.state, side, cid, stacks, source=f'{c.side}:{c.cid}',
               reason='「闹钟响彻四方」')
    if stacks >= 5:
        damage(c.state, other(c.side), 'player', 3, source=f'{c.side}:{c.cid}',
               reason='「噩梦」不少于 5 层')


def _a03(c):
    from app.modules.card_game.engine.duel_v2.continuous import trigger_dot
    foe = other(c.side)
    dots = front_debuff(c.state, foe).get('dots') or {}
    dot = dots.get('nightmare')
    if dot and dot.get('by') == c.side and int(dot.get('stacks') or 0):
        trigger_dot(c.state, foe, 'nightmare', dot, reason='「冰淇淋在烈阳中融化」')
        dots.pop('nightmare', None)
        c.history('触发并移除全部「噩梦」。')
    c.draw(1)


def _a04(c):
    foe = other(c.side)
    apply_dot(c.state, foe, 'nightmare', by=c.side, source_cid=c.cid, stacks=5)
    if _stacks(c) >= 10:
        add_hand(c.state, c.side, card_instance(c.state, 'AF01', c.side))
        c.history('「噩梦」达到 10 层，获得「茄汁金属乐」。')


def _a05(c):
    target = tuple(c.operation['target_id'].split(':', 1))
    on_bench = c.state['sides'][c.side]['front'] != c.cid
    c.sortie(target=target, no_counter=on_bench, stay_bench=on_bench)


def _a06(c):
    from app.modules.card_game.engine.duel_v2.combat import apply_enter_harmony, enter
    from app.modules.card_game.engine.duel_v2.escalation import harmony_cap
    team = c.state['sides'][c.side]
    front = team.get('front')
    payer = hero(c.state, c.side, front) if front and front != c.cid else None
    if payer and payer.get('attribute') in ('魂', '咒') and int(payer.get('harmony') or 0) < harmony_cap(c.state):
        c.operation['skip_nightmare'] = True
        kind = '浊燃' if payer['attribute'] == '咒' else '黯星'
        enter(c, resolve_harmony=False)
        apply_enter_harmony(c, front, kind, pay=False)
        c.sortie()
        return
    c.sortie()


def _af01(c):
    card = card_instance(c.state, 'A03', c.side)
    card['ephemeral_turn'] = c.state['turn']
    add_hand(c.state, c.side, card)
    c.history('获得「冰淇淋在烈阳中融化」，回合结束时弃置。')
    c.sortie()


def _a07(c):
    set_shape(c)
    if alive(c.state, c.side, c.cid):
        apply_dot(c.state, other(c.side), 'nightmare', by=c.side, source_cid=c.cid, stacks=2)


def _a07_turn_begin(c):
    apply_dot(c.state, other(c.side), 'nightmare', by=c.side, source_cid=c.cid, stacks=2)
    c.history('己方回合开始，「广告拍摄中！」施加 2 层「噩梦」。')


@register
class Anhunqu:
    id = 'anhunqu'
    text_name = '安魂曲'
    effects = {
        'A01': _a01, 'A02': _a02, 'A03': _a03, 'A04': _a04,
        'A05': _a05, 'A06': _a06, 'A07': _a07, 'A08': set_shape, 'AF01': _af01,
    }
    target_policies = {
        'A01': 'enemy_not_in_lineup', 'A02': 'any_living_esper', 'A05': 'living_enemy_role',
    }
    weapon_hooks = {'A07': {'on_turn_begin': _a07_turn_begin}}

    @staticmethod
    def on_ultimate(c):
        apply_dot(c.state, other(c.side), 'nightmare', by=c.side, source_cid=c.cid, stacks=5)
        c.history('发动终结：施加 5 层「噩梦」。')

    @staticmethod
    def after_hits(c, **_data):
        if c.operation.get('skip_nightmare') or not alive(c.state, c.side, c.cid):
            return
        apply_dot(c.state, other(c.side), 'nightmare', by=c.side, source_cid=c.cid, stacks=1)

    @staticmethod
    def on_dissonance(c, enemy=None, target_id=None, **_data):
        if not enemy:
            return
        front = c.state['sides'][enemy].get('front')
        if front and alive(c.state, enemy, front):
            damage(c.state, enemy, front, 1, source=f'{c.side}:{c.cid}', reason='【失谐】强化')
