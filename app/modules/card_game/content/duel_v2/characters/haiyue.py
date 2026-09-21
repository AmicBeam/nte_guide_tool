"""海月：2026-09-18 用户指定的桌游试玩规则，非原作数值。"""
from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import add_atk_buff, move_out, other


def teammate_attacked(state, side, cid):
    own = state['sides'][side]['characters'][cid]['entity_id']
    return any(entity_id != own for entity_id in state['sides'][side].get('attacked_this_turn', []))


def _u03(c):
    c.sortie(target=(other(c.side), 'player'))


def _u04(c):
    c.sortie(target=tuple(c.operation['target_id'].split(':', 1)), no_counter=True)


@register
class Haiyue:
    id = 'haiyue'
    text_name = '海月'
    effects = {'U01': lambda c: c.sortie(), 'U02': lambda c: c.sortie(),
               'U03': _u03, 'U04': _u04, 'U05': set_shape, 'U08': set_shape,
               'U07': set_shape, 'U06': set_shape}
    target_policies = {'U04': 'enemy_bench'}

    @staticmethod
    def makes_instant(state, side, card):
        if card['card_id'] == 'U01':
            return state['active_side'] != side
        return card['card_id'] in ('U02', 'U07') and teammate_attacked(state, side, card['character_id'])

    @staticmethod
    def sortie_from_bench(c, current=False):
        return bool(c.character.get('awakened') and c.state['sides'][c.side]['front'] != c.cid)

    @staticmethod
    def after_sortie(c):
        c.operation['return_after_resources'] = c.character['entity_id']

    @staticmethod
    def after_operation_resources(c):
        if c.operation.get('return_after_resources') == c.character['entity_id']:
            move_out(c.state, c.side, c.cid)

    @staticmethod
    def before_attack(c):
        if c.character.get('shape') == 'U08':
            add_atk_buff(c.character, 1, source_entity_id=c.entity_id, source_key=c.weapon_source('U08'), label='银河暂留')
            c.history('海月攻击前，武备触发：攻击 +1。')

    @staticmethod
    def dot_damage_bonus(c, kind=None):
        return 1 if kind == 'star' and c.character.get('shape') == 'U05' else 0

    @staticmethod
    def normal_attack_payment(c, current=None):
        team = c.state['sides'][c.side]
        if (c.character.get('shape') == 'U06' and not team['used'].get('instant')
                and teammate_attacked(c.state, c.side, c.cid)):
            return {'ap': 0, 'normal': False, 'resource': 'instant'}
        return None

    @staticmethod
    def pay_normal_attack(c, payment):
        if payment.get('resource') == 'instant':
            c.state['sides'][c.side]['used']['instant'] = True
            c.history('海月消耗本回合瞬发机会，发动普通出击。')
