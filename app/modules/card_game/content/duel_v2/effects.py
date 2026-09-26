"""V2 card API used by the engine. Character kits own the implementations."""
from app.modules.card_game.content.duel_v2.characters import EFFECTS, TARGET_POLICIES
from app.modules.card_game.content.duel_v2.registry import shape_attack_bonus, shape_event

BOARD_TARGET_POLICIES = {
    'living_ally', 'other_living_ally', 'enemy_front', 'ally_front',
    'enemy_bench', 'downed_ally', 'downed_enemy', 'living_enemy', 'living_role', 'injured_role',
    'living_enemy_role', 'any_living_esper', 'enemy_not_in_lineup', 'any_summon',
}
CHOICE_TARGET_POLICIES = {'enemy_hand', 'own_tactic_discard'}


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


__all__ = [
    'BOARD_TARGET_POLICIES',
    'CHOICE_TARGET_POLICIES',
    'EFFECTS',
    'TARGET_POLICIES',
    'interaction_kind',
    'shape_attack_bonus',
    'shape_event',
]
