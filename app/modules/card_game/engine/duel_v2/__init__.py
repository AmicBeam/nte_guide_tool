"""Four-character alternating-turn duel, independent of the legacy engine."""

__all__ = ['new_game', 'acting_side', 'legal_actions', 'apply_action', 'observe', 'choose_action']


def __getattr__(name):
    if name in ('new_game', 'acting_side', 'legal_actions', 'apply_action'):
        from .flow import acting_side, apply_action, legal_actions, new_game
        values = {
            'new_game': new_game,
            'acting_side': acting_side,
            'legal_actions': legal_actions,
            'apply_action': apply_action,
        }
        return values[name]
    if name in ('observe', 'choose_action'):
        from .projection import choose_action, observe
        values = {'observe': observe, 'choose_action': choose_action}
        return values[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
