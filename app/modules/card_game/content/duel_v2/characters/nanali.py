from app.modules.card_game.content.duel_v2.characters.common import (
    deal_combat_followup, on_first_turn_seed, perm_plus,
    revive_self, set_shape,
)
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent


def _n01(c):
    c.heal(max(0, c.character['max_hp'] - c.character['hp']))


def _nf01(c):
    perm_plus(c)
    team = c.state['sides'][c.side]
    key = Nanali.seed_play_key
    team[key] = int(team.get(key) or 0) + 1


def _n02(c):
    c.sortie()


def _n03(c):
    c.sortie()


def _n04(c):
    c.sortie()


def _n05(c):
    from app.modules.card_game.engine.duel_v2.state import damage
    side, cid = c.operation['target_id'].split(':', 1)
    amount = 3 + (2 if c.character.get('shape') == 'N08' else 0)
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
    @staticmethod
    def on_harmony_triggered(c, kind=None, entering=None, payer=None, **_data):
        if kind == '创生' and c.cid in (entering, payer):
            from app.modules.card_game.content.duel_v2.characters.common import grant_seed
            grant_seed(c, reason='娜娜莉参与触发【创生】')

    @staticmethod
    def combat_followup_ultimate(c, current=0):
        from app.modules.card_game.engine.duel_v2.state import front_debuff, other
        has_genesis = bool(c.state['sides'][c.side].get('genesis_layers'))
        has_weave = bool(front_debuff(c.state, other(c.side)).get('weave'))
        return 2 if c.character.get('awakened') and (has_genesis or has_weave) else 0

    @staticmethod
    def on_counter_resolved(c, target=None):
        amount = Nanali.combat_followup_ultimate(c)
        if amount and target:
            if c.character.get('shape') == 'N08':
                amount += 2
            c.followup(target, amount, reason='娜娜莉终结的追击伤害')

    @staticmethod
    def card_description(c, current, card):
        if card.get('card_id') == 'N04':
            count = int(c.state['sides'][c.side].get('genesis_weave_damage', 0)) // 10
            return current.removesuffix('（当前为0）') + f'（当前为{count}）'
        if card.get('card_id') == 'N07':
            count = int(c.state['sides'][c.side].get(Nanali.seed_play_key) or 0)
            return current.removesuffix('（当前为0）') + f'（当前为{count}）'
        return current

    @staticmethod
    def battle_frame(c, attack, shield):
        if c.card and c.card.get('card_id') == 'N04':
            attack += int(c.state['sides'][c.side].get('genesis_weave_damage', 0)) // 10
        return attack, shield

    @staticmethod
    def makes_card_free(state, side, card):
        return card.get('card_id') == 'NF01' and int(state.get('turn') or 0) >= 11

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
