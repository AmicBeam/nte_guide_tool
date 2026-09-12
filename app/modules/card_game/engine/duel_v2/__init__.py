"""Four-character alternating-turn duel, independent of the legacy engine."""
from .flow import acting_side, apply_action, legal_actions, new_game
from .projection import choose_action, observe

__all__ = ['new_game', 'acting_side', 'legal_actions', 'apply_action', 'observe', 'choose_action']
