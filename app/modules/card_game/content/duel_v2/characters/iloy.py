from app.modules.card_game.content.duel_v2.characters.common import (
    draw_owned_card, revive_character, set_shape,
)
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import damage, energy, front_target, heal, hero, shield


def _mark_healed(c, side, cid):
    if side != c.side or cid in (None, 'player'):
        return
    flags = hero(c.state, side, cid).setdefault('flags', {})
    flags['atk_buff'] = int(flags.get('atk_buff') or 0) + 1


def _y01(c):
    downed = sum(1 for item in c.state['sides'][c.side]['characters'].values()
                 if item.get('down_turns'))
    if downed:
        side, cid = front_target(c.state, c.side)
        damage(c.state, side, cid, 3 * downed, source=f'{c.side}:{c.cid}')


def _y02(c):
    _side, cid = c.operation['target_id'].split(':', 1)
    draw_owned_card(c, cid)


def _y03(c):
    side, cid = c.operation['target_id'].split(':', 1)
    heal(c.state, side, cid, 5)
    _mark_healed(c, side, cid)


def _y04(c):
    flags = c.character.setdefault('flags', {})
    flags['next_bonus'] = int(flags.get('next_bonus') or 0) + 1
    flags['next_shield'] = int(flags.get('next_shield') or 0) + 2


def _y05(c):
    side, cid = c.operation['target_id'].split(':', 1)
    revive_character(c, side, cid, temporary=True)


def _y06(c):
    side, cid = c.operation['target_id'].split(':', 1)
    revive_character(c, side, cid)


@register
class Iloy:
    id = 'iloy'
    text_name = '伊洛伊'
    effects = {
        'Y01': _y01, 'Y02': _y02, 'Y03': _y03, 'Y04': _y04,
        'Y05': _y05, 'Y06': _y06, 'Y07': set_shape, 'Y08': set_shape,
    }
    target_policies = {
        'Y02': 'living_ally',
        'Y03': 'living_role',
        'Y05': 'downed_ally',
        'Y06': 'downed_ally',
    }

    @staticmethod
    def before_hits(c):
        flags = c.character.setdefault('flags', {})
        c.operation['first_bonus'] += int(flags.pop('next_bonus', 0) or 0)
        extra_shield = int(flags.pop('next_shield', 0) or 0)
        if extra_shield:
            shield(c.state, c.side, c.cid, extra_shield)

    @staticmethod
    def on_ultimate(c):
        c.character.setdefault('flags', {})['energy_next'] = True
        if c.character.get('shape') != 'Y07':
            return
        cid = c.state['sides'][c.side].get('front')
        if cid:
            heal(c.state, c.side, cid, 3)
            _mark_healed(c, c.side, cid)

    @staticmethod
    def on_turn_begin(c):
        if not c.character.get('flags', {}).pop('energy_next', False):
            return
        for item in c.state['sides'][c.side]['characters'].values():
            if item['hp'] > 0:
                energy(c.state, c.side, item['id'], 1)

    @staticmethod
    def on_ally_down(c, downed=None, **_data):
        if c.character.get('shape') != 'Y08':
            return
        side, cid = front_target(c.state, c.side)
        damage(c.state, side, cid, 2, source=f'{c.side}:{c.cid}', settle=False)

    @staticmethod
    def on_ultimate_end(c):
        c.character.get('flags', {}).pop('energy_next', None)

