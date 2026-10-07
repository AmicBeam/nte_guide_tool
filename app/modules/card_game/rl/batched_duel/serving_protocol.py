"""Observer-safe serving search metadata and exact one-action public probes."""


def root_metadata(state,side):
    from ...engine.ai.advanced_search import _public_terminal_allowed
    phase=state['phase'];pending=state.get('pending_choice') or {}
    supported=phase in ('playing','mulligan')
    if phase=='choice':
        supported=(pending.get('side')==side and
            pending.get('kind') in ('inspect_top','discard','enemy_hand','hand_redraw') and
            (not state.get('operation') or state['operation'].get('side')==side))
    return dict(version=int(state['version']),phase=phase,search_supported=supported,
                public_terminal_allowed=phase=='playing' and _public_terminal_allowed(state,side))


def public_terminal_probe(state,side,action):
    from ...engine.duel_v2 import apply_action
    after=apply_action(state,side,action)
    return (after['phase']=='finished' and after.get('winner')==side and
            after['rng']==state['rng'] and
            not any(event['type'] in ('draw','gain') for event in after['events']))
