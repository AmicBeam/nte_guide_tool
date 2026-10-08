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
    def genesis_extra(c) -> int:
        return 1 if c.character['hp'] > 0 else 0

    @staticmethod
    def card_description(c, current, card):
        if card['card_id'] == 'J06':
            count = int(c.state['sides'][c.side].get('genesis_player_hits', 0))
            return f'{current}（X为{count}）'
        return current

    @staticmethod
    def on_ultimate(c):
        foe = other(c.side)
        for cid in list(c.state['sides'][foe]['characters']):
            if not alive(c.state, foe, cid):
                continue
            amount = 1
            flags = hero(c.state, foe, cid).setdefault('flags', {})
            if flags.get('pact'):
                from app.modules.card_game.content.duel_v2.registry import tally
                amount += 2 + tally(c, 'attachment_damage_bonus')
                flags.pop('pact', None)
                c.history(f"九原的终结清算{hero(c.state, foe, cid)['name']}的枚约，本次伤害 +2。",
                          target_side=foe, target_id=cid)
                if c.character.get('shape') == 'J07':
                    c.energy(1, reason='「使命必达」触发')
            damage(c.state, foe, cid, amount, source=f'{c.side}:{c.cid}')

    @staticmethod
    def on_genesis_damage(c, target=None, damage_dealt=False, **_data):
        if not target:
            return
        side, cid = target
        if is_esper(cid) and not alive(c.state, side, cid):
            return
        if damage_dealt and is_esper(cid):
            c.attach_pact(side, cid)
        if c.character.get('shape') == 'J08':
            damage(c.state, side, cid, 1, source=f'{c.side}:{c.cid}', reason='创生造成伤害，「现实避难所」触发')
