from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.state import PLAYER_ID, alive, damage, finished, front_target, hero, is_esper, other


def _j01(c):
    c.inspect_top_choice(3)


def _j02(c):
    from app.modules.card_game.engine.duel_v2.state import finished, front_target
    side, cid = front_target(c.state, c.side)
    had_pact = False
    hit_player = not is_esper(cid)
    if is_esper(cid):
        # User balance rule (2026-09-15): check before damage can clear the pact.
        had_pact = bool(hero(c.state, side, cid).get('flags', {}).get('pact'))
    damage(c.state, side, cid, 1, source=f'{c.side}:{c.cid}')
    if is_esper(cid):
        c.attach_pact(side, cid)
    if (had_pact or hit_player) and not finished(c.state):
        c.draw(1)


def _j03(c):
    side, cid = c.operation['target_id'].split(':', 1)
    hero(c.state, side, cid)['down_turns'] = 3


def _j04(c):
    side, cid = c.operation['target_id'].split(':', 1)
    damage(c.state, side, cid, 2, source=f'{c.side}:{c.cid}')
    c.attach_pact(side, cid)


def _j05(c):
    triggered = c.state['sides'][c.side].get('genesis_turn') == c.state['turn']
    damage(c.state, other(c.side), PLAYER_ID, 3 if triggered else 2, source=f'{c.side}:{c.cid}')


def _j06(c):
    # Lock the number of hits when played; these are tactic damage, not genesis.
    count = int(c.state['sides'][c.side].get('genesis_player_hits', 0))
    for _ in range(count):
        if finished(c.state):
            break
        damage(c.state, *front_target(c.state, c.side), 1, source=f'{c.side}:{c.cid}')


@register
class Jiuyuan:
    id = 'jiuyuan'
    text_name = '九原'
    effects = {
        'J01': _j01, 'J02': _j02, 'J03': _j03, 'J04': _j04,
        'J05': _j05, 'J06': _j06, 'J07': set_shape, 'J08': set_shape,
    }
    target_policies = {
        'J03': 'downed_enemy',
        'J04': 'living_enemy',
    }

    @staticmethod
    def genesis_bead_stacks(c, current=1, **_data):
        return 2

    @staticmethod
    def card_description(c, current, card):
        if card['card_id'] == 'J06':
            count = int(c.state['sides'][c.side].get('genesis_player_hits', 0))
            return f'{current}（X为{count}）'
        return current

    @staticmethod
    def on_ultimate(c):
        c.history('发动终结：每回合首次创生伤害对全体对方角色结算，并清算枚约，持续 2 个己方回合。')

    @staticmethod
    def on_genesis_segment(c, **_data):
        c.operation.pop('jiuyuan_genesis_sweep', None)

    @staticmethod
    def genesis_sweep_targets(c, current=None, **_data):
        if (not c.character.get('awakened')
                or c.character.get('genesis_sweep_turn') == c.state['turn']):
            return None
        c.character['genesis_sweep_turn'] = c.state['turn']
        c.operation['jiuyuan_genesis_sweep'] = c.entity_id
        foe = other(c.side)
        return [(foe, cid) for cid in c.state['sides'][foe]['characters'] if alive(c.state, foe, cid)] + [(foe, PLAYER_ID)]

    @staticmethod
    def genesis_hit_amount(c, current=1, target_side=None, target_id=None, **_data):
        if c.operation.get('jiuyuan_genesis_sweep') != c.entity_id or not is_esper(target_id):
            return None
        flags = hero(c.state, target_side, target_id).setdefault('flags', {})
        if not flags.pop('pact', None):
            return None
        from app.modules.card_game.content.duel_v2.registry import tally
        c.history('九原的终结消耗枚约，本次伤害 +2。', target_side=target_side, target_id=target_id)
        if c.character.get('shape') == 'J07':
            c.energy(1, reason='「使命必达」触发')
        return int(current) + 2 + tally(c, 'attachment_damage_bonus')

    @staticmethod
    def on_genesis_damage(c, target=None, damage_dealt=False, **_data):
        if not target:
            return
        side, cid = target
        if is_esper(cid) and not alive(c.state, side, cid):
            return
        if (damage_dealt and is_esper(cid)
                and c.operation.get('jiuyuan_genesis_sweep') != c.entity_id):
            c.attach_pact(side, cid)
        if c.character.get('shape') == 'J08':
            damage(c.state, side, cid, 1, source=f'{c.side}:{c.cid}', reason='创生造成伤害，「自律时间」触发')
