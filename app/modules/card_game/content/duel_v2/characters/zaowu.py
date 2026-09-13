from app.modules.card_game.content.duel_v2.characters.common import negative_kinds, set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _front_hero(c):
    from app.modules.card_game.engine.duel_v2.state import front_target, hero, is_esper
    enemy, target_id = front_target(c.state, c.side)
    if not is_esper(target_id):
        return None, None, None
    return enemy, target_id, hero(c.state, enemy, target_id)


def _s01(c):
    c.draw(1)
    enemy, _, target = _front_hero(c)
    if target and negative_kinds(target, c.state, enemy):
        c.energy()


def _s03(c):
    enemy, _, target = _front_hero(c)
    bonus = 1 if target and negative_kinds(target, c.state, enemy) else 0
    c.sortie(bonus=bonus)


def _s04(c):
    c.team_next_attack()


def _s06(c):
    enemy, _, target = _front_hero(c)
    c.hit_front(2)
    if target and negative_kinds(target, c.state, enemy):
        c.draw(1)


def _s07(c):
    c.energy()
    for h in c.state['sides'][c.side]['characters'].values():
        if h['hp'] > 0:
            h['flags']['next_attack'] = max(h['flags'].get('next_attack', 0), 1)


def _s08(c):
    enemy, _, target = _front_hero(c)
    extra = len(negative_kinds(target, c.state, enemy)) if target else 0
    c.sortie(bonus=1 + extra)


def _s02_shape(c, **_data):
    if not c.once('S08'):
        return
    enemy, _, target = _front_hero(c)
    if target and negative_kinds(target, c.state, enemy):
        c.heal(1)


def _s05_shape(c, **_data):
    if not c.once('S07'):
        return
    from app.modules.card_game.engine.duel_v2.state import front_target
    enemy, _, target = _front_hero(c)
    if target and negative_kinds(target, c.state, enemy):
        c.followup(front_target(c.state, c.side), 1)


@register
class Zaowu:
    id = 'zaowu'
    text_name = '早雾'
    effects = {
        'S01': _s01, 'S02': _s07, 'S03': _s03, 'S04': _s04,
        'S05': _s08, 'S06': _s06, 'S07': set_shape, 'S08': set_shape,
    }
    target_policies = {'S06': 'enemy_front'}
    shape_events = {
        ('S08', GameEvent.V2_SORTIE_FINISHED): _s02_shape,
        ('S07', GameEvent.V2_SORTIE_FINISHED): _s05_shape,
    }

    @staticmethod
    def attack_bonus(c, target=None, target_id=None, **_data):
        from app.modules.card_game.engine.duel_v2.state import is_esper, other
        if not is_esper(target_id):
            return 0
        cap = 3 if c.character['awakened'] else 2
        return min(cap, len(negative_kinds(target, c.state, other(c.side))))
