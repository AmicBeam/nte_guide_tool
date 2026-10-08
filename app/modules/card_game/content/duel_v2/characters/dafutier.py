"""达芙蒂尔：用户指定的洞察、倾陷与失谐试玩规则。"""
from random import Random

from app.modules.card_game.content.duel_v2.characters.common import set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.events import GameEvent
from app.modules.card_game.engine.duel_v2.state import (
    SIDES, add_collapse, add_hand, alive, card_instance, collapsed, damage,
    energy_max, finished, force_collapse, front_target, hero, other, team_order,
)


def _insight_damage(c):
    return 2 + int(c.state['sides'][c.side].get('insight_bonus', 0))


def _support_count(c):
    team = c.state['sides'][c.side]
    return min(2, int(team.get('support_count', 0))) if team.get('support_turn') == c.state['turn'] else 0


def _eye(c):
    # Entry may itself release a support, which belongs to this turn's count.
    from app.modules.card_game.engine.duel_v2.combat import enter
    enter(c)
    count = _support_count(c)
    c.operation['card_bonus'] = int(c.operation.get('card_bonus', 0)) + count
    c.operation['collapse_bonus'] = count
    c.sortie()


def _target_collapse(c, amount):
    side, cid = c.operation['target_id'].split(':', 1)
    add_collapse(c.state, side, cid, amount, by=c.side)


def _creations(c):
    _target_collapse(c, 2)


def _lake(c):
    _target_collapse(c, 4)


def _hidden(c):
    foe = other(c.side)
    for cid in list(team_order(c.state, foe)):
        if not hero(c.state, foe, cid).get('summoned'):
            add_collapse(c.state, foe, cid, 2, by=c.side)
    team = c.state['sides'][c.side]
    team['insight_bonus'] = int(team.get('insight_bonus', 0)) + 1
    c.history(f"「于隐匿中」：本局「洞察」的伤害 +1（当前为 {_insight_damage(c)}）。")


def _truth(c):
    team = c.state['sides'][c.side]
    amount, repeats = _insight_damage(c), 1 + int(team.get('insight_played', 0))
    foe = other(c.side)
    for _ in range(repeats):
        if finished(c.state):
            break
        targets = ['player'] + [cid for cid in team_order(c.state, foe) if alive(c.state, foe, cid)]
        rng = Random(c.state['rng'])
        target = rng.choice(targets)
        c.state['rng'] = rng.getrandbits(63)
        damage(c.state, foe, target, amount, source=f'{c.side}:{c.cid}', reason='「完美真相」')


def _finale(c):
    from app.modules.card_game.engine.duel_v2.continuous import living_espers
    rng = Random(c.state['rng'])
    targets = []
    for side in SIDES:
        candidates = living_espers(c.state, side)
        if candidates:
            targets.append((side, rng.choice(candidates)))
    c.state['rng'] = rng.getrandbits(63)
    for side, cid in targets:
        force_collapse(c.state, side, cid, by=c.side)


def _insight(c):
    side, cid = c.operation['target_id'].split(':', 1)
    if alive(c.state, side, cid) and collapsed(hero(c.state, side, cid)):
        damage(c.state, side, cid, _insight_damage(c), source=f'{c.side}:{c.cid}', reason='「洞察」')


def _black_weapon(c, attacker_side, attacker_id, **_data):
    if attacker_side == other(c.side) and not hero(c.state, attacker_side, attacker_id).get('summoned'):
        add_collapse(c.state, attacker_side, attacker_id, 3, by=c.side)


@register
class Dafutier:
    id = 'dafutier'
    text_name = '达芙蒂尔'
    effects = {'F01': _eye, 'F02': set_shape, 'F03': _creations, 'F04': _hidden,
               'F05': _truth, 'F06': _lake, 'F07': _finale, 'F08': set_shape,
               'FF01': _insight}
    target_policies = {'F03': 'any_living_esper', 'F06': 'living_enemy_esper',
                       'FF01': 'collapsed_enemy_esper'}
    weapon_hooks = {'F08': {GameEvent.V2_OPPONENT_SORTIE_FINISHED: _black_weapon}}

    @staticmethod
    def on_collapse_entered(c, target_side, target_id, **_data):
        if hero(c.state, target_side, target_id).get('summoned'):
            return
        if target_side == other(c.side) and c.character.get('insight_used_turn') != c.state['turn']:
            c.character['insight_used_turn'] = c.state['turn']
            card = card_instance(c.state, 'FF01', c.side)
            card['character_id'] = c.cid
            add_hand(c.state, c.side, card, reason='使对方异能者倾陷，达芙蒂尔的异能触发')
        # This card has an unconditional in-zone trigger, independently of the
        # once-per-turn passive. The resolving instance is in neither zone.
        if c.cid != Dafutier.id:
            return
        team = c.state['sides'][c.side]
        for zone in ('deck', 'discard'):
            card = next((item for item in team[zone] if item.get('card_id') == 'F03'
                         and not item.get('copy')), None)
            if card is not None:
                team[zone].remove(card)
                add_hand(c.state, c.side, card, reveal=True, reason='使异能者倾陷，「造物之众」触发')
                break

    @staticmethod
    def on_dissonance(c, enemy=None, **_data):
        if enemy != other(c.side):
            return
        layers = int(c.character.get('dissonance_layers', 0))
        if layers >= 2:
            return
        c.character['dissonance_layers'] = layers + 1
        for cid in list(team_order(c.state, enemy)):
            target = hero(c.state, enemy, cid)
            if target.get('summoned'):
                continue
            target.setdefault('collapse_reductions', {})[c.entity_id] = layers + 1
        c.history(f"【失谐】强化：所有对方异能者的倾陷值上限永久 -1（已叠加 {layers + 1}/2 层）。")
        for cid in list(team_order(c.state, enemy)):
            if not hero(c.state, enemy, cid).get('summoned'):
                add_collapse(c.state, enemy, cid, 0, by=c.side)

    @staticmethod
    def on_card_used(c, actor_id, card, **_data):
        if actor_id == c.cid and card.get('card_id') == 'FF01':
            team = c.state['sides'][c.side]
            team['insight_played'] = int(team.get('insight_played', 0)) + 1

    @staticmethod
    def after_hits(c, target_id, enemy, **_data):
        if ((c.card or {}).get('card_id') == 'F01' and alive(c.state, enemy, target_id)
                and not hero(c.state, enemy, target_id).get('summoned')):
            add_collapse(c.state, enemy, target_id, c.operation.get('collapse_bonus', 0), by=c.side)

    @staticmethod
    def on_support_resolved(c, actor_id, **_data):
        if actor_id != c.cid or not c.character.get('awakened'):
            return
        damage(c.state, *front_target(c.state, c.side), 2,
               source=f'{c.side}:{c.cid}', reason='达芙蒂尔终结的援护追加伤害')
        if c.character.get('shape') == 'F02':
            c.energy(energy_max(c.character), reason='终结追加伤害触发「机械游行」')

    @staticmethod
    def card_description(c, current, card):
        cid = card.get('card_id')
        if cid == 'F01':
            return f'{current}（当前为 {_support_count(c)}）'
        if cid == 'FF01':
            return f'不消耗行动力。对 1 名处于倾陷的对方异能者造成 {_insight_damage(c)} 点伤害。'
        if cid == 'F05':
            count = int(c.state['sides'][c.side].get('insight_played', 0))
            return f'{current}（当前伤害为 {_insight_damage(c)}，共 {1 + count} 次）'
        return current
