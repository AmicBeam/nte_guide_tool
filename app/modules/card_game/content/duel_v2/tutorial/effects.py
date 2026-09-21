"""Register tutorial-only card effects and isolated teaching kits."""
from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.effects import EFFECTS, TARGET_POLICIES
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import damage, heal, other


def _noop(c):
    return


def _sortie(c):
    c.sortie()


def _heal(c):
    target = (c.operation or {}).get('target_id') or ''
    cid = target.split(':', 1)[-1]
    if cid:
        heal(c.state, c.side, cid, 2)


def _instant(c):
    damage(c.state, other(c.side), 'player', 1, source=f'{c.side}:{c.cid}', kind='effect')


EFFECTS.update({
    'tutorial_z03': _sortie,
    'tutorial_m_strike': _sortie,
    'tutorial_z_heal': _heal,
    'tutorial_z08': set_shape,
    'tutorial_z_instant': _instant,
    'tutorial_m_response': _sortie,
})
TARGET_POLICIES.update({
    'tutorial_z_heal': 'other_living_ally',
})


@register
class TutorialProtagonist:
    id = 'tutorial_protagonist'
    text_name = '主角'

    @staticmethod
    def after_card(c, *, actor_id, card, **_data):
        if actor_id != c.cid or not card or card.get('type') != 'battle':
            return
        if c.character['hp'] <= 0:
            return
        c.character['harmony'] = 2


@register
class TutorialBohe:
    id = 'tutorial_bohe'
    text_name = '薄荷'

    @staticmethod
    def combat_overflow(c, current=0, attacker_id=None, defender_id=None, raw_overflow=0, **_data):
        if attacker_id == c.cid:
            return int(raw_overflow or 0)
        return None


@register
class TutorialHaiyue:
    """Only the ranged ultimate; independent of the production character kit."""
    id = 'tutorial_haiyue'
    text_name = '海月'

    @staticmethod
    def sortie_from_bench(c, current=False):
        return bool(c.character.get('awakened') and c.state['sides'][c.side]['front'] != c.cid)
