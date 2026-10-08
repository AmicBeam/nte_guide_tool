from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _h01(c):
    c.energy()
    c.extra_delay()


def _h03(c):
    c.sortie(low_hp_bonus=2, low_hp_max=4)


def _h05(c):
    c.draw(1)
    c.heal(1)


def _h06(c):
    c.inspect_top_choice(2)


def _h07(c):
    c.energy()
    side, cid = c.operation['target_id'].split(':', 1)
    c.grant_harmony(cid, 1)


def _h08(c):
    c.sortie(bonus=2, bypass=True)


def _h02_shape(c, **data):
    if c.once('H08') and data.get('target'):
        c.followup(data['target'], 1, reason='「焰魂狂飙」触发')


def _h04_shape(c, **data):
    if data.get('operation', {}).get('harmony') == '延滞':
        data['operation']['no_counter'] = True
        c.history('触发延滞，「铁骑速递」触发：哈索尔本次出击不受反击。')


@register
class Hathor:
    id = 'hathor'
    text_name = '哈索尔'
    effects = {
        'H01': _h01, 'H02': _h07, 'H03': _h03, 'H04': _h08,
        'H05': _h05, 'H06': _h06, 'H07': set_shape, 'H08': set_shape,
    }
    target_policies = {'H02': 'other_living_ally'}
    shape_events = {
        ('H08', GameEvent.V2_FLOWER_RESOLVED): _h02_shape,
        ('H07', GameEvent.V2_HARMONY_RESOLVED): _h04_shape,
    }

    @staticmethod
    def delay_duration(c, current=1, **_data):
        return 4 if c.character.get('awakened') else 3

    @staticmethod
    def delay_slow(c, current=1, **_data):
        return 0

    @staticmethod
    def delay_hits(c) -> int:
        return 0

    @staticmethod
    def on_delay_hit(c, target=None, **_data):
        if c.character.get('shape') == 'H02':
            c.on_shape(GameEvent.V2_FLOWER_RESOLVED, target=target, entering=c.cid)
