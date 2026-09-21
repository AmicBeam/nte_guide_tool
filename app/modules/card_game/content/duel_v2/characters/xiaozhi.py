"""小吱：2026-09-18 用户指定的桌游试玩规则，非原作数值。"""
from random import Random

from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    alive, card_instance, finished, hero, is_player, other, shuffle,
)


def change_jingu(c, amount, reason, *, allow_downed=False):
    if finished(c.state) or (not allow_downed and not alive(c.state, c.side, c.cid)):
        return
    before = int(c.character.get('jingu', 0))
    c.character['jingu'] = before + amount
    c.history(f'{reason}：金谷 {amount:+d}（现有 {before + amount} 点）。',
              before={'jingu': before}, after={'jingu': before + amount})


def mark_card(c, card):
    rng = Random(c.state['rng'])
    card['jingu_mark'] = {'delta': rng.choice((-1, 0, 1)), 'source_entity_id': c.entity_id}
    c.state['rng'] = rng.getrandbits(63)


def publish_marks(c):
    from app.modules.card_game.engine.duel_v2.presentation import public_card
    from app.modules.card_game.engine.duel_v2.state import effective_card
    from app.modules.card_game.engine.duel_v2.state import log
    log(c.state, '小吱终结更新手牌金谷标注。', 'hand_marks', side=c.side,
        actor=f'{c.side}:{c.cid}', present=False, private_side=c.side,
        private_hand=[public_card(effective_card(c.state, c.side, card)) for card in c.state['sides'][c.side]['hand']])


def refresh_marks(c):
    for card in c.state['sides'][c.side]['hand']:
        mark_card(c, card)
    publish_marks(c)


def release(c, card_id):
    """A generated battle resolution shares the parent's once-per-operation resources.

    It is not a hand play, creates no discard/copy/record, and cannot activate
    the paid options. Every release uses the leftmost (base) option.
    """
    from app.modules.card_game.content.duel_v2 import CARDS
    from app.modules.card_game.engine.duel_v2.context import EffectContext
    card = dict(CARDS[card_id], card_id=card_id, character_id=c.cid)
    child = dict(c.operation, card=card, card_bonus=0, card_set_attack=card['attack'], released=True,
                 first_bonus=0, redeem_bonus=0, harmony=None, support=False, last_attack_target=None, option_id='base')
    ctx = EffectContext(c.state, c.side, c.cid, child)
    ctx.history(f"小吱额外释放「{card['name']}」。")
    ctx.shield(card.get('shield', 0))
    Xiaozhi.effects[card_id](ctx)


def _q01(c):
    c.sortie(bonus=3 if c.operation.get('option_id') == 'boost' else 0)


def _q02(c):
    c.sortie()
    target_ref = c.operation.get('last_attack_target')
    if not target_ref or finished(c.state):
        return
    enemy, target_id = target_ref
    target = c.state['sides'][enemy] if is_player(target_id) else hero(c.state, enemy, target_id)
    if target['hp'] > 0:
        from app.modules.card_game.engine.duel_v2.modifiers import deadline
        target.get('next_damage_resistance', {}).pop('光', None)
        c.apply_effect('xiaozhi.light_resistance', 'damage.resistance', -2, target=target,
                       stacking='replace', stack_key='resistance:光', label='光抗性 -2',
                       conditions=[{'id': 'damage_attribute', 'args': {'attribute': '光'}}],
                       consume_on='damage', expiry=deadline(c.state, c.state['active_side'], 0, 'turn_end'))
        c.history('命中后，使目标本回合下次受到的光属性伤害抗性 -2。', target_side=enemy, target_id=target_id)
    if (c.operation.get('option_id') == 'release_a' and alive(c.state, c.side, c.cid)
            and not finished(c.state)):
        release(c, 'Q01')


def _q03(c):
    c.sortie()
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    selected = c.operation.get('option_id')
    if selected in ('release_b', 'release_ba'):
        release(c, 'Q02')
    if selected == 'release_ba' and not finished(c.state) and alive(c.state, c.side, c.cid):
        release(c, 'Q01')


def _q04(c):
    if c.state.get('_response_intercept') and c.character.get('shape') == 'Q08':
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
               'Q05': _q05, 'Q06': _q06, 'Q08': set_shape, 'Q07': set_shape}

    @staticmethod
    def on_game_start(c):
        c.character['jingu'] = 0

    @staticmethod
    def on_turn_begin(c):
        change_jingu(c, 1, '己方回合开始，小吱触发异能')
        if c.character.get('awakened'):
            refresh_marks(c)

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
        refresh_marks(c)

    @staticmethod
    def on_ultimate_end(c):
        for card in c.state['sides'][c.side]['hand']:
            card.pop('jingu_mark', None)
        publish_marks(c)
        if c.character.pop('jingu_loan', False):
            change_jingu(c, -3, '小吱终结结束', allow_downed=True)

    @staticmethod
    def on_hand_added(c, card):
        if c.character.get('awakened'):
            mark_card(c, card)
            publish_marks(c)

    @staticmethod
    def after_marked_card(c):
        mark = c.operation.get('card_mark') or {}
        if mark.get('source_entity_id') != c.entity_id:
            return
        change_jingu(c, int(mark['delta']), '手牌金谷标注结算', allow_downed=True)

    @staticmethod
    def _attack_bonus(c, attacker_id=None, **_):
        if attacker_id == c.cid and c.character.get('shape') == 'Q08' and c.card and c.card['type'] == 'battle':
            return max(0, int(c.character.get('jingu', 0))) // 2
        return 0

    @staticmethod
    def normal_attack_payment(c, current=None):
        if current is None and c.character.get('shape') == 'Q07' and c.character.get('jingu', 0) >= 3:
            return {'ap': 0, 'normal': False, 'resource': 'jingu', 'bonus': 2}
        return None

    @staticmethod
    def pay_normal_attack(c, payment):
        if payment.get('resource') == 'jingu':
            change_jingu(c, -3, '小吱武备额外出击')

    @staticmethod
    def aura_effects(c, property, query):
        if property != 'attack.hit':
            return
        from app.modules.card_game.engine.duel_v2.modifiers import contribution
        amount = Xiaozhi._attack_bonus(c, **{k: v for k, v in query.items() if k != 'operation'})
        if amount:
            yield contribution('xiaozhi.attack_bonus', property, amount, kind='aura', scope='allies',
                               source_entity_id=c.entity_id, source_key=c.weapon_source('Q08'), label='思考喵')
