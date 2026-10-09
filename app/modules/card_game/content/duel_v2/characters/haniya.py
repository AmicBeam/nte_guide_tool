"""Haniya: Everness 1020 identity/names; all rules are user-authored (2026-09-20)."""
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import alive, damage, finished, hero, other, shuffle
from .common import set_shape


def gain_aura(c, amount):
    if finished(c.state) or not alive(c.state, c.side, c.cid):
        return
    h = c.character
    before = int(h.get('protagonist_aura', 0))
    h['protagonist_aura'] = min(6, before + amount * (2 if h.get('awakened') else 1))
    gained = h['protagonist_aura'] - before
    if gained:
        c.history(f'哈尼娅获得 {gained} 点主角光环。')


def _v01(c):
    gain_aura(c, 1)
    c.state['sides'][c.side]['normal_attack_available'] = True
    c.history('重置己方普通出击次数。')


def _v02(c):
    side, cid = c.operation['target_id'].split(':', 1)
    hero(c.state, side, cid)['flags']['free_normal_attack_turn'] = c.state['turn']
    c.history(f"{hero(c.state, side, cid)['name']}本回合普通出击不消耗行动力。", target_id=cid)
    gain_aura(c, 2)


def _v03(c):
    side, cid = c.operation['target_id'].split(':', 1)
    # Same lifetime as Iloy's attack gain: survives turns/leaving/equipment;
    # the common knockdown cleanup removes the attached modifier.
    c.add_atk_debuff(side, cid, 2)
    gain_aura(c, 2)


def _v05(c):
    count = int(c.character.get('protagonist_aura', 0))
    c.character['protagonist_aura'] = 0
    if count:
        c.history(f'哈尼娅消耗 {count} 点主角光环，随机分配 {count * 2} 点伤害。')
    enemy = other(c.side)
    for _ in range(count * 2):
        if finished(c.state):
            return
        targets = [cid for cid, h in c.state['sides'][enemy]['characters'].items() if h['hp'] > 0]
        targets.append('player')
        shuffle(c.state, targets)
        damage(c.state, enemy, targets[0], 1, source=f'{c.side}:{c.cid}')


def _apply_attack_aura(c, actor_id, response):
    op, h = c.operation, c.character
    battle = bool(c.card and c.card.get('type') == 'battle')
    amount = int(h.get('protagonist_aura', 0))
    if not amount:
        return
    free = h.get('shape') == 'V06' and op.get('harmony') == '黯星'
    if not free:
        h['protagonist_aura'] = 0
    # Applied with the battle frame; set-attack cards overwrite this bonus.
    if response:
        flags = hero(c.state, c.side, actor_id)['flags']
        if not (battle and c.card.get('attack_mode') == 'set') and 'counter_set_attack' not in flags:
            flags['counter_bonus'] = int(flags.get('counter_bonus', 0)) + amount
    else:
        op['sortie_attack_bonus'] = amount
    overwritten = battle and c.card.get('attack_mode') == 'set'
    result = '本次攻击由战斗牌覆写' if overwritten else f'本次战斗攻击 +{amount}'
    cost = '主角光环保留' if free else f'消耗 {amount} 点主角光环'
    c.history(f'哈尼娅的异能触发：{cost}，{result}。', target_id=actor_id)


@register
class Haniya:
    id = 'haniya'
    text_name = '哈尼娅'
    effects = {'V01': _v01, 'V02': _v02, 'V03': _v03,
               'V04': lambda c: gain_aura(c, 2 * int(c.state['sides'][c.side].get('dark_star_count', 0))),
               'V05': _v05, 'V06': set_shape, 'V07': set_shape, 'V08': set_shape}
    target_policies = {'V02': 'living_ally', 'V03': 'living_enemy'}

    @staticmethod
    def card_description(c, current, card):
        if card.get('card_id') == 'V04':
            count = int(c.state['sides'][c.side].get('dark_star_count', 0))
            return current.removesuffix('（当前为0）') + f'（当前为{count}）'
        return current

    @staticmethod
    def on_game_start(c):
        c.character['protagonist_aura'] = 0

    @staticmethod
    def on_turn_begin(c):
        gain_aura(c, 1)

    @staticmethod
    def on_card_used(c, actor_id, card):
        if card.get('type') == 'battle' and c.character.get('shape') == 'V08':
            _apply_attack_aura(c, actor_id, bool(c.state.get('_response_intercept')))

    @staticmethod
    def on_sortie_ready(c, actor_id, response):
        if c.operation.get('normal_attack'):
            _apply_attack_aura(c, actor_id, response)

    @staticmethod
    def on_ally_attack_resolved(c, actor_id, player_damage):
        if (c.character.get('shape') == 'V07' and player_damage > 0
                and not hero(c.state, c.side, actor_id).get('summoned')):
            c.history('「青春注脚」触发：己方异能者的攻击对对方玩家造成伤害，抽 1 张牌。')
            c.draw(1)
