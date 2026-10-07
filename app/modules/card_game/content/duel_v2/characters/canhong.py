"""残虹：浊燃叠层、幻境与持续伤害。数值是桌游试玩参数。"""
from app.modules.card_game.content.duel_v2.characters.common import revive_character, set_shape
from app.modules.card_game.content.duel_v2.registry import register
from app.modules.card_game.engine.duel_v2.continuous import (
    add_stack_all, apply_dot, living_espers, total_stacks,
)
from app.modules.card_game.engine.duel_v2.modifiers import add_effect, deadline
from app.modules.card_game.engine.duel_v2.state import (
    add_hand, alive, card_instance, damage, discard, force_collapse,
    front_target, hero, is_esper, other,
)


def _mirage(c):
    return any(effect.get('definition_id') == 'canhong.mirage' for effect in c.character.get('effects') or [])


def _enter_mirage(c):
    add_effect(c.character, 'canhong.mirage', 'status.mirage', 1, stacking='replace',
               source_entity_id=c.entity_id, source_key='mirage', label='幻境',
               expiry=deadline(c.state, c.side, 2, 'turn_start'), clear_on=('down',))
    c.history('残虹进入幻境，持续 2 个己方回合。')


def _targets(c):
    if c.character.get('shape') != 'C08' or total_stacks(c.state, c.side) < 5:
        return None
    foe = other(c.side)
    return [f'{foe}:{cid}' for cid in living_espers(c.state, foe)]


def _c01(c):
    c.sortie()
    if total_stacks(c.state, c.side) >= 5 and alive(c.state, c.side, c.cid):
        missing = c.character['max_hp'] - c.character['hp']
        if missing:
            c.heal(missing, reason='「瞳中深渊」')


def _c02(c):
    if total_stacks(c.state, c.side) >= 10:
        c.draw(1)
    foe = other(c.side)
    before = set(living_espers(c.state, foe))
    c.sortie()
    if before - set(living_espers(c.state, foe)):
        add_hand(c.state, c.side, card_instance(c.state, 'C02', c.side))
        c.history('击倒对方异能者，获得 1 张「渊底之吻」。')


def _c03(c):
    c.sortie()
    team = c.state['sides'][c.side]
    if team.get('canhong_collapse_used'):
        return
    target = (c.operation or {}).get('last_attack_target') or []
    if len(target) == 2 and is_esper(target[1]) and alive(c.state, target[0], target[1]):
        force_collapse(c.state, target[0], target[1], by=c.side)
        team['canhong_collapse_used'] = True
        c.history(f"「吻痕窥梦」：{hero(c.state, target[0], target[1])['name']}进入倾陷。")


def _c04(c):
    steps = min(4, total_stacks(c.state, c.side) // 5)
    if steps:
        c.shield(steps)
    c.sortie(bonus=steps)


def _c05(c):
    mirage = _mirage(c)
    c.sortie(no_counter=mirage)
    if mirage and alive(c.state, c.side, c.cid):
        c.energy(1, reason='幻境，「花开见血」')


def _c06(c):
    set_shape(c)


def _c07(c):
    revive_character(c, c.side, c.cid)
    set_shape(c)


def _play_c07(c):
    team = c.state['sides'][c.side]
    card = next((item for item in team['hand'] if item.get('card_id') == 'C07'), None)
    if card is not None:
        team['hand'].remove(card)
    else:
        for index, item in enumerate(list(team['deck'])):
            if item.get('card_id') == 'C07':
                card = team['deck'].pop(index)
                break
    if card is None:
        return
    c.operation['card'] = card
    # The downed-ally context was built before this card existed on the operation.
    _c07(c.for_character(c.side, c.cid))
    discard(c.state, c.side, card)
    c.history('持续伤害不少于 20 层，自动使用「惑心谲影」。')


@register
class Canhong:
    id = 'canhong'
    text_name = '残虹'
    effects = {
        'C01': _c01, 'C02': _c02, 'C03': _c03, 'C04': _c04, 'C05': _c05,
        'C06': _c06, 'C07': _c07, 'C08': set_shape,
    }

    @staticmethod
    def burn_extra_stack(c, current=0, **_data):
        if not alive(c.state, c.side, c.cid):
            return 0
        return 1

    @staticmethod
    def on_game_start(c):
        # User-authored tabletop rule (2026-10-07), granted only during setup.
        c.grant_harmony(c.cid, 2, reason='对局开始，残虹的异能')

    @staticmethod
    def on_ultimate(c):
        apply_dot(c.state, other(c.side), 'venom', by=c.side, source_cid=c.cid, stacks=5)
        c.history('发动终结：施加 5 层「鸩火」。')

    @staticmethod
    def after_card(c, card=None, actor_id=None, **_data):
        if actor_id != c.cid or not card or card.get('type') != 'battle':
            return
        if alive(c.state, c.side, c.cid):
            _enter_mirage(c)

    @staticmethod
    def after_hits(c, target_loss=0, **_data):
        if not _mirage(c) or not alive(c.state, c.side, c.cid):
            return
        foe = other(c.side)
        apply_dot(c.state, foe, 'etch', by=c.side, source_cid=c.cid, stacks=1)
        if target_loss > 0:
            add_stack_all(c.state, foe, c.side, 1)

    @staticmethod
    def on_damage_dealt(c, target_side=None, amount=0, damage_kind=None, **_data):
        # Combat grows the just-applied 蚀心 in after_hits; other damage uses this timing.
        if (damage_kind != 'combat' and amount > 0 and target_side == other(c.side)
                and _mirage(c) and alive(c.state, c.side, c.cid)):
            add_stack_all(c.state, target_side, c.side, 1)

    @staticmethod
    def on_replaced(c):
        if c.character.get('shape') != 'C06':
            return
        side, cid = front_target(c.state, c.side)
        damage(c.state, side, cid, 1, source=f'{c.side}:{c.cid}', reason='「血染双瞳」')

    @staticmethod
    def on_ally_down(c, downed=None, **_data):
        if downed == c.cid and total_stacks(c.state, c.side) >= 20:
            _play_c07(c)

    @staticmethod
    def hp_floor(c, current=0, target_id=None, **_data):
        card = (c.operation or {}).get('card') or {}
        if target_id == c.cid and card.get('card_id') == 'C01' and total_stacks(c.state, c.side) >= 5:
            return 1
        return None

    @staticmethod
    def battle_targets(c, card=None, **_data):
        if not card or card.get('type') != 'battle' or card.get('character_id') != 'canhong':
            return None
        return _targets(c)

    @staticmethod
    def normal_attack_targets(c, current=None, **_data):
        return _targets(c)

    @staticmethod
    def combat_overflow(c, current=0, raw_overflow=0, attacker_id=None, **_data):
        if attacker_id == c.cid and c.character.get('shape') == 'C08' and total_stacks(c.state, c.side) >= 5:
            return raw_overflow
        return None
