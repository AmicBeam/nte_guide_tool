"""Hathor awaits a user-authored rework; only base card behavior remains."""
from .common import set_shape
from app.modules.card_game.content.duel_v2.registry import register


def _empty(c):
    pass


def _battle(c):
    c.sortie()


@register
class Hathor:
    id = 'hathor'
    text_name = '哈索尔'
    effects = {'H01': _empty, 'H02': _empty, 'H03': _battle, 'H04': _battle,
               'H05': _empty, 'H06': _empty, 'H07': set_shape, 'H08': set_shape}

    @staticmethod
    def card_description(c, current, card):
        return '' if card.get('card_id') in Hathor.effects else current
