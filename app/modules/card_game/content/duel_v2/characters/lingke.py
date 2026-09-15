from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _k01(c):
    c.draw(1)
    c.energy()


def _k03(c):
    c.sortie(harmony_bonus=1)


def _k04(c):
    c.hit_front(2 if c.state['sides'][c.side].get('sync_ever') else 1)


def _k06(c):
    side, cid = c.operation['target_id'].split(':', 1)
    c.grant_harmony(cid, 1)


def _k07(c):
    switched = c.state['sides'][c.side]['front'] != c.cid
    c.sortie(bonus=1)
    if switched:
        c.draw(1)


def _k08(c):
    c.sortie(hits=2, no_counter_harmony=True)


def _k02_shape(c, **_data):
    if c.is_enemy_turn() and c.once('K08'):
        c.shield(1, reason='对方回合战斗受伤，「远行者之声」触发')


def _k05_shape(c, **data):
    if data.get('operation', {}).get('harmony') == '同频':
        data['operation']['first_bonus'] += 1
        c.history('触发同频，「宇宙的频段」触发：灵可本次第一次攻击伤害 +1。')


@register
class Lingke:
    id = 'lingke'
    text_name = '灵可'
    effects = {
        'K01': _k01, 'K02': _k07, 'K03': _k03, 'K04': _k04,
        'K05': _k08, 'K06': _k06, 'K07': set_shape, 'K08': set_shape,
    }
    target_policies = {'K04': 'enemy_front', 'K06': 'other_living_ally'}
    shape_events = {
        ('K08', GameEvent.V2_COMBAT_HP_LOST): _k02_shape,
        ('K07', GameEvent.V2_HARMONY_RESOLVED): _k05_shape,
    }

    @staticmethod
    def on_enemy_turn_begin(c):
        c.state['sides'][c.side]['used'].pop('K08', None)

    @staticmethod
    def harmony_kind(c, current=None, **_data):
        if current or c.state['sides'][c.side]['used'].get('sync'):
            return None
        return '同频'

    @staticmethod
    def on_sync(c, entering=None, **_data):
        from app.modules.card_game.engine.duel_v2.state import alive, finished, shield
        team = c.state['sides'][c.side]
        team['sync_ever'] = True
        amount = 2 if c.character['awakened'] else 1
        if entering and alive(c.state, c.side, entering) and not finished(c.state):
            shield(c.state, c.side, entering, amount)
        c.hit_front(amount)
