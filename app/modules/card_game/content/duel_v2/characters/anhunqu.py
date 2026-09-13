from app.modules.card_game.content.duel_v2.characters.common import grant_nightmare, set_shape, settle_nightmare_on
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _front_hero(c):
    from app.modules.card_game.engine.duel_v2.state import front_target, hero, is_esper
    enemy, target_id = front_target(c.state, c.side)
    if not is_esper(target_id):
        return None, None, None
    return enemy, target_id, hero(c.state, enemy, target_id)


def _a01(c):
    enemy, target_id, target = _front_hero(c)
    if target is None:
        return
    grant_nightmare(target, 1)
    c.energy()


def _a03(c):
    _, _, target = _front_hero(c)
    bonus = 1 if target and (target.get('flags') or {}).get('nightmare') else 0
    c.sortie(bonus=bonus)
    _, _, after = _front_hero(c)
    if after:
        take = settle_nightmare_on(after, 1)
        if take:
            from app.modules.card_game.engine.duel_v2.state import front_target
            c.followup(front_target(c.state, c.side), take)


def _a05(c):
    c.hit_front(1)
    _, _, target = _front_hero(c)
    if target and (target.get('flags') or {}).get('nightmare'):
        c.heal(1)


def _a06(c):
    c.grant_harmony(c.cid, 1)
    if c.state['sides'][c.side]['front'] != c.cid:
        c.energy()


def _a07(c):
    _, _, target = _front_hero(c)
    bypass = bool(target and (target.get('flags') or {}).get('nightmare'))
    c.sortie(bonus=2, bypass=bypass)


def _a08(c):
    c.sortie(bonus=1)
    _, _, target = _front_hero(c)
    if target:
        take = settle_nightmare_on(target, None)
        if take:
            from app.modules.card_game.engine.duel_v2.state import front_target
            c.followup(front_target(c.state, c.side), take)


def _a02_shape(c, **data):
    if data.get('operation', {}).get('harmony'):
        data['operation']['first_bonus'] += 1


@register
class Anhunqu:
    id = 'anhunqu'
    text_name = '安魂曲'
    effects = {
        'A01': _a01, 'A02': _a07, 'A03': _a03, 'A04': _a08,
        'A05': _a05, 'A06': _a06, 'A07': set_shape, 'A08': set_shape,
    }
    target_policies = {'A01': 'enemy_front', 'A05': 'enemy_front'}
    shape_events = {('A08', GameEvent.V2_HARMONY_RESOLVED): _a02_shape}

    @staticmethod
    def on_burn_applied(c, enemy=None, target_id=None, **_data):
        from app.modules.card_game.engine.duel_v2.state import hero, is_esper
        if not is_esper(target_id):
            return
        target = hero(c.state, enemy, target_id)
        layers = 3 if c.character['awakened'] else 2
        grant_nightmare(target, layers)

    @staticmethod
    def after_sortie(c):
        if not c.character['awakened']:
            return
        _, _, target = _front_hero(c)
        if target and (target.get('flags') or {}).get('nightmare'):
            take = settle_nightmare_on(target, 1)
            if take:
                from app.modules.card_game.engine.duel_v2.state import front_target
                c.followup(front_target(c.state, c.side), take)

    @staticmethod
    def on_dot(c, target_id=None, kind=None, **_data):
        if c.character.get('shape') != 'A07' or kind not in ('burn', 'nightmare'):
            return
        from app.modules.card_game.engine.duel_v2.state import front_target
        _, front = front_target(c.state, c.side)
        if target_id == front and c.once('A07'):
            c.heal(1)

    @staticmethod
    def on_enemy_turn_begin(c):
        c.state['sides'][c.side]['used'].pop('A07', None)
