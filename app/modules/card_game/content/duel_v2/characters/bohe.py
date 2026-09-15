from app.modules.card_game.content.duel_v2 import CHARACTERS
from app.modules.card_game.content.duel_v2.characters.common import perm_plus, set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import (
    add_atk_buff, alive, attack_value, collapsed, finished,
)


def _m01(c):
    perm_plus(c, atk=1, hp=0)
    c.sortie()


def _m02(c):
    from app.modules.card_game.engine.duel_v2.state import team_order
    for cid in team_order(c.state, c.side):
        if alive(c.state, c.side, cid):
            add_atk_buff(c.state['sides'][c.side]['characters'][cid], 1)


def _m03(c):
    c.sortie(ignore_shield=True)


def _m04(c):
    c.character.setdefault('flags', {})['share_overflow'] = True
    c.sortie()


def _m05(c):
    c.character.setdefault('flags', {})['pending_atk'] = int(
        (c.character.get('flags') or {}).get('pending_atk') or 0) + 2


def _m06(c):
    printed = int(CHARACTERS[c.cid]['attack'])
    extra = max(0, int(c.character.get('base_attack') or printed) - printed)
    c.hit_front(2 + extra)


@register
class Bohe:
    id = 'bohe'
    text_name = '薄荷'
    effects = {
        'M01': _m01, 'M02': _m02, 'M03': _m03, 'M04': _m04,
        'M05': _m05, 'M06': _m06, 'M07': set_shape, 'M08': set_shape,
    }
    target_policies = {
        'M06': 'enemy_front',
    }

    @staticmethod
    def combat_overflow(c, current=0, attacker_id=None, defender_id=None, raw_overflow=0, **_data):
        raw = int(raw_overflow or 0)
        if attacker_id == c.cid:
            return raw
        flags = c.character.setdefault('flags', {})
        if defender_id == c.cid and flags.get('share_overflow'):
            flags.pop('share_overflow', None)
            return raw
        return None

    @staticmethod
    def outgoing_attack(c, current=0, attacker_id=None, **_data):
        if attacker_id != c.cid:
            return None
        if c.character.get('shape') == 'M07' and c.state.get('active_side') != c.side:
            return 0
        return None

    @staticmethod
    def incoming_damage(c, current=0, kind=None, target_id=None, source=None, **_data):
        if target_id != c.cid or not c.character.get('awakened'):
            return None
        if kind in ('combat', 'followup', 'penetration'):
            return None
        src_side = None
        if isinstance(source, str) and ':' in source:
            src_side = source.split(':', 1)[0]
        if src_side == c.side:
            return None
        if src_side or kind in ('burn', 'star', 'nightmare', 'overlay', 'genesis', 'delay', 'sync'):
            return 0
        return None

    @staticmethod
    def after_hits(c, **_data):
        (c.character.get('flags') or {}).pop('share_overflow', None)

    @staticmethod
    def on_turn_begin(c):
        flags = c.character.setdefault('flags', {})
        flags.pop('share_overflow', None)
        pending = int(flags.pop('pending_atk', 0) or 0)
        if pending:
            before = attack_value(c.character)
            add_atk_buff(c.character, pending)
            c.history(f'回合开始，「第一直觉」触发：薄荷攻击 +{pending}。',
                      before={'attack': before}, after={'attack': attack_value(c.character)})
        if c.character.get('shape') != 'M08':
            return
        if finished(c.state) or not alive(c.state, c.side, c.cid) or collapsed(c.character):
            return
        from app.modules.card_game.engine.duel_v2.flow import finish_operation, operation
        c.history('回合开始，「开始净空」触发：薄荷自动出击。')
        parent = c.state.get('operation')
        operation(c.state, c.side, c.cid).sortie()
        if c.state['phase'] != 'choice' and not finished(c.state):
            finish_operation(c.state)
        c.state['operation'] = parent

    @staticmethod
    def on_enemy_turn_begin(c):
        (c.character.get('flags') or {}).pop('share_overflow', None)
