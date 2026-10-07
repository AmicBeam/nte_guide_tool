"""早雾：持续伤害增幅与鬼郎丸。数值是桌游试玩参数。"""
from random import Random

from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.continuous import kind_count
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, contribution, deadline
from app.modules.card_game.engine.duel_v2.state import (
    alive, damage, discard, energy, energy_max, force_collapse, front_debuff, hero, name, other, team_order,
)
from app.modules.card_game.engine.duel_v2.summons import summon_front


FLAVOR = '鬼郎丸似乎吃了什么不太妙的东西，需要拍打一下！'
ITEMS = ('vinyl', 'collar', 'liquid')
LABELS = {'vinyl': '黑胶唱片', 'collar': '好狗狗的项圈', 'liquid': '气味奇怪的液体'}


def _gulang(c):
    for cid in team_order(c.state, c.side):
        h = hero(c.state, c.side, cid)
        if h.get('summoned') and h.get('name') == '鬼郎丸' and h.get('hp', 0) > 0:
            return cid
    return None


def _rng(state):
    rng = Random(state['rng'])
    state['rng'] = rng.getrandbits(63)
    return rng


def summon_gulang(c, item):
    kinds = kind_count(c.state, c.side)
    cid = summon_front(c.state, c.side, name='鬼郎丸', attack=kinds, hp=4,
                       portrait='/static/images/characters/portrait/鬼郎丸.webp',
                       avatar='/static/images/characters/portrait/鬼郎丸.webp')
    h = hero(c.state, c.side, cid)
    h['flags']['gulang_item'] = item
    h['flavor'] = FLAVOR
    h['passive'] = f'吞噬了{LABELS[item]}。受到伤害时触发。倒地或离开战斗区时消失。'
    c.history(f'召唤鬼郎丸，其吞噬了{LABELS[item]}。')
    return cid


def _s01(c):
    summon_gulang(c, 'vinyl')


def _s02(c):
    summon_gulang(c, 'collar')


def _s03(c):
    summon_gulang(c, 'liquid')


def _s04(c):
    side, cid = c.operation['target_id'].split(':', 1)
    damage(c.state, side, cid, 3, source=f'{c.side}:{c.cid}', reason='「祈愿性的依归」')


def _s05(c):
    cid = _gulang(c)
    if not cid:
        return
    damage(c.state, c.side, cid, 2, source=f'{c.side}:{c.cid}', reason='「失觉性的崩落」')
    foe = other(c.side)
    for enemy_id in list(team_order(c.state, foe)):
        h = hero(c.state, foe, enemy_id)
        if h.get('summoned') or h.get('hp', 0) <= 0:
            continue
        damage(c.state, foe, enemy_id, 2, source=f'{c.side}:{c.cid}', reason='「失觉性的崩落」')


def _s06(c):
    cid = _gulang(c)
    if not cid:
        return
    c.state['sides'][other(c.side)]['seal_normal_attack'] = cid
    c.history('鬼郎丸在场：对方不能普通出击。')


def _maybe_recall(c):
    if c.character.get('shape') != 'S07' or kind_count(c.state, c.side) < 4:
        return
    if not alive(c.state, c.side, c.cid):
        return
    item = _rng(c.state).choice(ITEMS)
    cid = summon_gulang(c, item)
    damage(c.state, c.side, cid, 3, source=f'{c.side}:{c.cid}', reason='「小鬼头大胃口」')


def _s07(c):
    set_shape(c)
    _maybe_recall(c)


@register
class Zaowu:
    id = 'zaowu'
    text_name = '早雾'
    effects = {
        'S01': _s01, 'S02': _s02, 'S03': _s03, 'S04': _s04,
        'S05': _s05, 'S06': _s06, 'S07': _s07, 'S08': set_shape,
    }
    target_policies = {'S04': 'any_summon'}

    @staticmethod
    def can_play(c, current=True, card=None, **_data):
        if card and card.get('card_id') in ('S05', 'S06') and not _gulang(c):
            return False
        return None

    @staticmethod
    def on_turn_begin(c):
        _maybe_recall(c)

    @staticmethod
    def on_ultimate(c):
        bonus = 3 if c.character.get('shape') == 'S08' else 2
        for cid in team_order(c.state, c.side):
            h = hero(c.state, c.side, cid)
            if cid == c.cid or h.get('summoned') or h.get('hp', 0) <= 0:
                continue
            add_effect(h, 'zaowu.ultimate_atk', 'attack.panel', bonus, stacking='replace',
                       source_entity_id=c.entity_id, source_key='S-ultimate', label='饕餮宴',
                       expiry=deadline(c.state, c.side, 2, 'turn_start'))
        c.history(f'发动终结：其他己方异能者攻击 +{bonus}，持续 2 个己方回合。')

    @staticmethod
    def on_ally_summon_hurt(c, summon_id=None, amount=0, **_data):
        if not summon_id or amount <= 0:
            return
        h = hero(c.state, c.side, summon_id)
        if h.get('name') != '鬼郎丸':
            return
        item = (h.get('flags') or {}).get('gulang_item')
        foe = other(c.side)
        enemies = [cid for cid in team_order(c.state, foe)
                   if not hero(c.state, foe, cid).get('summoned') and hero(c.state, foe, cid).get('hp', 0) > 0]
        if item == 'vinyl':
            hand = c.state['sides'][foe]['hand']
            if hand:
                card = hand.pop(_rng(c.state).randrange(len(hand)))
                discard(c.state, foe, card)
                c.history(f"鬼郎丸受到伤害：{name(c.state, foe)}随机弃置手牌「{card['name']}」。",
                          target_side=foe, target_id='player', present=True)
        elif item == 'collar' and enemies:
            cid = _rng(c.state).choice(enemies)
            force_collapse(c.state, foe, cid, by=c.side)
            c.history(f"鬼郎丸受到伤害：{hero(c.state, foe, cid)['name']}进入倾陷。")
        elif item == 'liquid' and enemies:
            cid = _rng(c.state).choice(enemies)
            damage(c.state, foe, cid, amount, source=f'{c.side}:{summon_id}', reason='鬼郎丸')

    @staticmethod
    def on_summon_kill(c, **_data):
        if c.character.get('shape') == 'S08' and alive(c.state, c.side, c.cid):
            missing = energy_max(c.character) - int(c.character.get('energy') or 0)
            if missing:
                energy(c.state, c.side, c.cid, missing)
                c.history('鬼郎丸消灭对方异能者，「好狗狗走四方」触发：早雾回满能量。')
