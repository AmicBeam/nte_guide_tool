from app.modules.card_game.content.duel_v2.characters.common import (
    deal_combat_followup, on_first_turn_seed, on_rout_seed, perm_plus,
    revive_self, set_shape,
)
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _n01(c):
    c.heal(max(0, c.character['max_hp'] - c.character['hp']))


def _nf01(c):
    perm_plus(c)
    c.grant_harmony(c.cid, 1, reason='家族壮大')
    team = c.state['sides'][c.side]
    key = Nanali.seed_play_key
    team[key] = int(team.get(key) or 0) + 1


def _n02(c):
    c.sortie()


def _n03(c):
    c.sortie()


def _n04(c):
    team = c.state['sides'][c.side]
    team['genesis_damage_bonus'] = int(team.get('genesis_damage_bonus') or 0) + 1
    c.history('「不是闯祸精」生效：本回合己方创生伤害 +1。')
    c.sortie()


def _n05(c):
    from app.modules.card_game.engine.duel_v2.state import damage
    side, cid = c.operation['target_id'].split(':', 1)
    amount = 4 + (2 if c.character.get('shape') == 'N08' else 0)
    damage(c.state, side, cid, amount, source=f'{c.side}:{c.cid}', kind='followup',
           reason='「小弟遍天下」的追击伤害')


def _n06(c):
    revive_self(c)


@register
class Nanali:
    id = 'nanali'
    text_name = '娜娜莉'
    seed_card = 'NF01'
    seed_key = 'nanali_family'
    seed_play_key = 'nanali_family_played'
    followup_shape = 'N08'
    scale_shape = 'N07'
    payoff_card = 'N03'
    effects = {
        'N01': _n01, 'N02': _n02, 'N03': _n03, 'N04': _n04,
        'N05': _n05, 'N06': _n06, 'N07': set_shape, 'N08': set_shape,
        'NF01': _nf01,
    }
    target_policies = {'N05': 'living_enemy'}
    on_turn_begin = staticmethod(on_first_turn_seed)
    on_rout = staticmethod(on_rout_seed)

    @staticmethod
    def makes_instant(state, side, card):
        if card.get('card_id') != 'NF01':
            return False
        return int(state.get('turn') or 0) >= 5

    @staticmethod
    def after_sortie(c):
        extra = 0
        if c.card and c.card.get('card_id') == 'N03':
            extra = 0
        deal_combat_followup(c)

    @staticmethod
    def on_ultimate_end(c):
        return
