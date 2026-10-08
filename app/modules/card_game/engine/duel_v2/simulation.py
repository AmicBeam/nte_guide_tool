"""Offline hypothetical transitions using the exact rules, without replay patches.

Never use this entry point for persisted rooms or public replay collection.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_collect_patches = ContextVar('duel_v2_collect_presentation_patches', default=True)


def collecting_patches():
    return _collect_patches.get()


@contextmanager
def _without_patches():
    token = _collect_patches.set(False)
    try:
        yield
    finally:
        _collect_patches.reset(token)


def simulate_action(state, side, action):
    from .flow import apply_action
    # The public-board cache is derivable and is never a rule input. Remove it
    # before the rule engine clones state, without changing the caller's object.
    source = {k: v for k, v in state.items() if k != '_public_board'}
    with _without_patches():
        result = apply_action(source, side, action)
    result.pop('_public_board', None)
    return result
