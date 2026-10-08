from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register


def _z01(c):
    from app.modules.card_game.engine.duel_v2.state import heal, hero
    cid = c.state['sides'][c.side]['front']
    if cid:
        h = hero(c.state, c.side, cid)
        heal(c.state, c.side, cid, max(0, h['max_hp'] - h['hp']))


def _z02(c):
    from app.modules.card_game.engine.duel_v2.state import add_hand, finished
    if finished(c.state):
        return
    team = c.state['sides'][c.side]
    taken = 0
    index = 0
    while taken < 2 and index < len(team['deck']):
        card = team['deck'][index]
        if card.get('type') == 'form' and not card.get('copy'):
            team['deck'].pop(index)
            add_hand(c.state, c.side, card, reveal=True)
            taken += 1
            continue
        index += 1


def _z03(c):
    c.sortie()


def _z04(c):
    c.sortie()


def _z05(c):
    c.sortie()


def _weapon_attack(c):
    from app.modules.card_game.engine.duel_v2.state import add_atk_buff, hero, log
    front = c.state['sides'][c.side]['front']
    if front:
        target = hero(c.state, c.side, front)
        add_atk_buff(target, 1, source_entity_id=c.entity_id, source_key=c.effect_source_key, label='倾世之雨')
        log(c.state, f"「倾世之雨」使{target['name']}攻击 +1。", 'effect',
            side=c.side, source=c.cid, actor=f'{c.side}:{c.cid}', target=f'{c.side}:{front}', amount=1)


def _weapon_heal(c):
    from app.modules.card_game.engine.duel_v2.state import heal
    front = c.state['sides'][c.side]['front']
    if front:
        heal(c.state, c.side, front, 2, reason='回合结束，「休息日」触发')


def _weapon_ap(c):
    team = c.state['sides'][c.side]
    team['ap'] = int(team.get('ap') or 0) + 1
    c.history('回合结束，「诓定解放」触发：获得 1 点行动力。')


@register
class Zero:
    id = 'zero'
    text_name = '零'
    shaped_forms_instant = True
    ultimate_forms_instant = True
    effects = {
        'Z01': _z01, 'Z02': _z02, 'Z03': _z03, 'Z04': _z04,
        'Z05': _z05, 'Z06': set_shape, 'Z07': set_shape, 'Z08': set_shape,
    }
    target_policies = {'Z01': 'ally_front'}

    @staticmethod
    def after_card(c, *, actor_id, card):
        if actor_id != c.cid or not card or card.get('type') != 'battle':
            return
        if c.character['hp'] <= 0:
            return
        c.grant_harmony(c.cid, 2, reason='零打出战斗牌，触发异能')

    weapon_hooks = {
        'Z08': {'on_turn_end': _weapon_attack},
        'Z07': {'on_turn_end': _weapon_heal},
        'Z06': {'on_turn_end': _weapon_ap},
    }
