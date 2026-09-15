from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _r01(c):
    c.shield(1)
    c.energy()


def _r03(c):
    c.sortie(bonus=1)


def _r05(c):
    c.heal(2)
    if c.state['sides'][c.side]['front'] == c.cid:
        c.shield(2)


def _r06(c):
    bonus = 2 if c.character.get('gaze', 0) >= 2 else 0
    c.sortie(bonus=bonus)


def _r07(c):
    c.sortie(bonus=2)
    if c.character['awakened']:
        c.add_gaze(1)


def _r08(c):
    c.sortie(bonus=1, bypass=True, awake_player_bonus=3)


def _r02_bypass(c, current=False, attacker_id=None, target=None, target_id=None, **_data):
    if current or attacker_id != c.cid or c.character.get('shape') != 'R08' or not target_id:
        return None
    if target.get('shield'):
        return True
    return None


def _r04_shape(c, **_data):
    if c.once('R07'):
        c.shield(1, reason='获得凝视，「梦的边缘」触发')


@register
class Zhenhong:
    id = 'zhenhong'
    text_name = '真红'
    effects = {
        'R01': _r01, 'R02': _r07, 'R03': _r03, 'R04': _r08,
        'R05': _r05, 'R06': _r06, 'R07': set_shape, 'R08': set_shape,
    }
    shape_events = {('R07', GameEvent.V2_GROWTH_GAINED): _r04_shape}

    @staticmethod
    def attack_bypass(c, current=False, attacker_id=None, target=None, target_id=None, **data):
        return _r02_bypass(c, current, attacker_id, target, target_id, **data)

    @staticmethod
    def after_sortie(c):
        from app.modules.card_game.engine.duel_v2.state import consume_once
        if not consume_once(c.state, c.side, 'gaze'):
            return
        gained = c.add_gaze(1)
        if gained:
            from app.modules.card_game.engine.events import GameEvent
            c.on_shape(GameEvent.V2_GROWTH_GAINED)

    @staticmethod
    def on_turn_begin(c):
        team = c.state['sides'][c.side]
        if c.character['awakened'] and c.character['hp'] > 0 and team['front'] != c.cid:
            c.add_gaze(1)

    @staticmethod
    def on_ultimate_end(c):
        c.character['gaze'] = min(int(c.character.get('gaze') or 0), 2)
