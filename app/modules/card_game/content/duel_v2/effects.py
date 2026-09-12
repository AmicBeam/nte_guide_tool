"""The 32 card effects, expressed through an explicit engine-supplied context.

This module imports only event constants, never engine services. A copied card receives Xun as its caster;
no card text replacement or inherited original-character passive is involved.
"""

from app.modules.card_game.engine.events import GameEvent


def n01(c):
    c.draw(1)
    c.energy()


def shape(c):
    c.set_shape()


def n03(c):
    c.sortie(bonus=1, shield_after_shape=1)


def n04(c):
    c.sortie(no_counter=True)


def n06(c):
    c.heal_target(2, front_shield=2)


def n07(c):
    c.sortie(bonus=2, penetrate_harmony=True)


def n08(c):
    c.shield(3)
    c.sortie(bonus=1, awake_player_bonus=3)


def z01(c):
    c.hit_front(2, bypass=True)


def z02(c):
    c.draw(2)
    c.discard_choice()


def z04(c):
    c.clear_front_shield()
    c.energy()


def z06(c):
    c.sortie(bonus=1, bypass=True)


def z07(c):
    c.reveal_top_owned()


def z08(c):
    c.team_next_attack()


def j01(c):
    wounded = c.front_wounded()
    c.hit_front(2)
    if wounded:
        c.heal(1)


def j03(c):
    c.inspect_top_choice(2)


def j04(c):
    c.sortie(low_hp_bonus=2)


def j06(c):
    c.inspect_enemy_hand()


def j07(c):
    c.energy()
    c.extra_flower()


def j08(c):
    c.sortie(bonus=2, penetrate_low_hp=True)


def x01(c):
    c.heal(2)
    c.energy()


def x03(c):
    c.sortie(no_counter_harmony=True)


def x04(c):
    c.recover_target()


def x06(c):
    c.sortie(bonus=2, bypass=True)


def x07(c):
    c.return_after_sortie()


def x08(c):
    c.sortie(hits=2, no_counter=True)


EFFECTS = {
    'N01': n01, 'N02': shape, 'N03': n03, 'N04': n04,
    'N05': shape, 'N06': n06, 'N07': n07, 'N08': n08,
    'Z01': z01, 'Z02': z02, 'Z03': shape, 'Z04': z04,
    'Z05': shape, 'Z06': z06, 'Z07': z07, 'Z08': z08,
    'J01': j01, 'J02': shape, 'J03': j03, 'J04': j04,
    'J05': shape, 'J06': j06, 'J07': j07, 'J08': j08,
    'X01': x01, 'X02': shape, 'X03': x03, 'X04': x04,
    'X05': shape, 'X06': x06, 'X07': x07, 'X08': x08,
}

# These target policies are used by both validation and legal-action enumeration.
TARGET_POLICIES = {
    'N06': 'other_living_ally', 'Z01': 'enemy_front', 'Z04': 'enemy_front',
    'J01': 'enemy_front', 'J06': 'enemy_hand', 'X04': 'own_nonterminal_tactic_discard',
}
BOARD_TARGET_POLICIES = {'other_living_ally', 'enemy_front'}
CHOICE_TARGET_POLICIES = {'enemy_hand', 'own_nonterminal_tactic_discard'}


def interaction_kind(card):
    """UI drop/target hint derived from card type and declared target policy."""
    if card.get('type') == 'form':
        return 'form'
    policy = TARGET_POLICIES.get(card.get('card_id') or card.get('id'))
    if policy in BOARD_TARGET_POLICIES:
        return 'target'
    if policy in CHOICE_TARGET_POLICIES:
        return 'choice'
    return 'cast'

def shape_attack_bonus(shape_id):
    return 1 if shape_id == 'N05' else 0


def shape_event(c, event, **data):
    """Handle only the caster's current shape, with shared per-turn usage keys."""
    shape_id = c.character['shape']
    if shape_id == 'N02' and event == GameEvent.V2_GROWTH_GAINED:
        c.shield(1)
    elif shape_id == 'N05' and event == GameEvent.V2_SORTIE_FINISHED and c.once('N05'):
        c.heal(1)
    elif shape_id == 'Z03' and event == GameEvent.V2_COMBAT_HP_LOST and c.is_enemy_turn() and c.once('Z03'):
        c.draw(1)
    elif shape_id == 'Z05' and event == GameEvent.V2_HARMONY_RESOLVED:
        data['operation']['first_bonus'] += 1
    elif shape_id == 'J02' and event == GameEvent.V2_FLOWER_RESOLVED and c.once('J02'):
        c.followup(data['target'], 1)
    elif shape_id == 'J05' and event == GameEvent.V2_FLOWER_RESOLVED and c.once('J05'):
        c.heal_character(data['entering'], 1)
    elif shape_id == 'X02' and event == GameEvent.V2_COPY_FINISHED and c.once('X02'):
        c.heal(2)
    elif shape_id == 'X05' and event == GameEvent.V2_RECORD_REDEEMED:
        data['operation']['no_counter'] = True
        if data['count'] == 2:
            data['operation']['first_bonus'] += 1
