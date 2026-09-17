"""小吱：2026-09-18 用户指定的桌游试玩规则，非原作数值。"""
from random import Random

from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    alive, card_instance, finished, hero, is_player, other, shuffle,
)


def change_jingu(c, amount, reason):
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    before = int(c.character.get('jingu', 0))
    c.character['jingu'] = before + amount
    c.history(f'{reason}：金谷 {amount:+d}（现有 {before + amount} 点）。',
              before={'jingu': before}, after={'jingu': before + amount})


def spend(c):
    if finished(c.state) or not alive(c.state, c.side, c.cid) or c.character.get('jingu', 0) < 3:
        return False
    change_jingu(c, -3, '小吱消耗金谷')
    return True


def release(c, card_id):
    """A generated battle resolution shares the parent's once-per-operation resources.

    It is not a hand play, creates no discard/copy/record, and cannot activate
    the active-play-only bonus of Q01. The chain is strictly Q03 -> Q02 -> Q01.
    """
    from app.modules.card_game.content.duel_v2 import CARDS
    from app.modules.card_game.engine.duel_v2.context import EffectContext
    card = dict(CARDS[card_id], card_id=card_id, character_id=c.cid)
    child = dict(c.operation, card=card, card_bonus=card['attack'], released=True,
                 first_bonus=0, redeem_bonus=0, harmony=None, support=False, last_attack_target=None)
    ctx = EffectContext(c.state, c.side, c.cid, child)
    ctx.history(f"小吱额外释放「{card['name']}」。")
    ctx.shield(card.get('shield', 0))
    Xiaozhi.effects[card_id](ctx)


def _q01(c):
    bonus = 0
    if not c.operation.get('released') and not c.state.get('_response_intercept') and spend(c):
        bonus = 3
    c.sortie(bonus=bonus)


def _q02(c):
    c.sortie()
    target_ref = c.operation.get('last_attack_target')
    if not target_ref or finished(c.state):
        return
    enemy, target_id = target_ref
    target = c.state['sides'][enemy] if is_player(target_id) else hero(c.state, enemy, target_id)
    if target['hp'] > 0:
        target.setdefault('next_damage_resistance', {})['光'] = -2
        c.history('命中后，使目标下次受到的光属性伤害抗性 -2。', target_side=enemy, target_id=target_id)
    if spend(c):
        release(c, 'Q01')


def _q03(c):
    c.sortie()
    if spend(c):
        release(c, 'Q02')


def _q04(c):
    if c.state.get('_response_intercept') and c.character.get('shape') == 'Q07':
        flags = c.character.setdefault('flags', {})
        flags['counter_bonus'] = int(flags.get('counter_bonus', 0)) + max(0, c.character.get('jingu', 0)) // 2
    c.sortie()


def _q05(c):
    change_jingu(c, 1, '小吱打出战术牌')
    # Unlike Xun's temporary copies, this is a permanent same-name deck card.
    # A temporary copy may not itself produce another copy.
    if not c.card.get('copy') and not finished(c.state):
        card = card_instance(c.state, 'Q05', c.side)
        c.state['sides'][c.side]['deck'].append(card)
        shuffle(c.state, c.state['sides'][c.side]['deck'])
        c.history(f"将一张「{card['name']}」的复制洗入牌库。")
    c.draw(1)


def _q06(c):
    from app.modules.card_game.engine.duel_v2.escalation import energy_cap
    c.energy(energy_cap(c.state, c.character), reason='本回合已触发盈蓄')


@register
class Xiaozhi:
    id = 'xiaozhi'
    text_name = '小吱'
    effects = {'Q01': _q01, 'Q02': _q02, 'Q03': _q03, 'Q04': _q04,
               'Q05': _q05, 'Q06': _q06, 'Q07': set_shape, 'Q08': set_shape}

    @staticmethod
    def on_game_start(c):
        c.character['jingu'] = 0

    @staticmethod
    def on_turn_begin(c):
        change_jingu(c, 1, '己方回合开始，小吱触发异能')

    @staticmethod
    def on_damage_dealt(c, target_side=None, target_id=None, amount=0, **_):
        if target_side == other(c.side) and is_player(target_id) and amount > 0:
            change_jingu(c, 1, '小吱对对方玩家造成伤害，触发异能')

    @staticmethod
    def on_ultimate(c):
        # Refreshing a live ultimate settles its old loan before lending again.
        if c.character.get('jingu_loan'):
            change_jingu(c, -3, '小吱结清上次终结的金谷')
        c.character['jingu_loan'] = True
        change_jingu(c, 3, '小吱发动终结')

    @staticmethod
    def on_ultimate_end(c):
        if c.character.pop('jingu_loan', False):
            change_jingu(c, -3, '小吱终结结束')

    @staticmethod
    def on_knockdown(c):
        c.character['jingu'] = 0
        c.character.pop('jingu_loan', None)

    @staticmethod
    def after_card(c, actor_id=None, card=None):
        if actor_id == c.cid and card.get('type') == 'battle' and c.character.get('awakened'):
            rng = Random(c.state['rng'])
            amount = rng.choice((-1, 0, 1))
            c.state['rng'] = rng.getrandbits(63)
            change_jingu(c, amount, '小吱终结：战斗牌结算')

    @staticmethod
    def attack_bonus(c, attacker_id=None, **_):
        if attacker_id == c.cid and c.character.get('shape') == 'Q07' and c.card and c.card['type'] == 'battle':
            return max(0, int(c.character.get('jingu', 0))) // 2
        return 0

    @staticmethod
    def normal_attack_payment(c, current=None):
        if current is None and c.character.get('shape') == 'Q08' and c.character.get('jingu', 0) >= 3:
            return {'ap': 0, 'normal': False, 'resource': 'jingu', 'bonus': 2}
        return None

    @staticmethod
    def pay_normal_attack(c, payment):
        if payment.get('resource') == 'jingu':
            change_jingu(c, -3, '小吱武备额外出击')
