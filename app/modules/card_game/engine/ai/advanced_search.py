"""Bounded observer-only planning for frozen serving policies. Never trains."""
from copy import deepcopy
from hashlib import sha256
import time

SEARCH_SECONDS = 3.0
SEARCH_SIMULATIONS = 32


def _runtime(model):
    from .recovery_model import SCHEMAS
    schema = getattr(model, 'schema', None)
    if schema in SCHEMAS:
        from app.modules.card_game.rl import recovery_runtime
        return recovery_runtime
    if schema == 'fixed_ten_v1':
        from app.modules.card_game.rl import ten_search_runtime
        return ten_search_runtime
    if schema == 'cross_five_grounded_wdl_v1':
        from app.modules.card_game.rl import cross_runtime
        return cross_runtime
    if schema == 'official_public_v5':
        from app.modules.card_game.rl import league_rollout
        return league_rollout
    return None


def _public_terminal_allowed(state, side):
    """Decline a proof if an unknown opponent response could change it.

    Only public roster/life and hand count are consulted. Never inspect hidden
    identities to conclude that a response is absent.
    """
    from app.modules.card_game.content.duel_v2 import CARDS
    foe = state['sides']['b' if side == 'a' else 'a']
    if not foe['hand']:
        return True
    return not any(card.get('response') and card['character_id'] in foe['characters']
                   and foe['characters'][card['character_id']]['hp'] > 0
                   for card in CARDS.values())


def choose(state, side, model, fallback):
    """Return the search action; fallback only when no usable search exists.

    The frozen bot is also the observation-only opponent rollout surrogate;
    this is not a claim to know the human's policy. Hidden worlds use the
    existing information-set sampler. Root proof checks and tree simulations
    share one budget of at most 32, and all work shares the 3-second deadline.
    """
    runtime = _runtime(model)
    if runtime is None:
        return fallback
    from app.modules.card_game.rl.information_search import sample_world, search
    from app.modules.card_game.engine.duel_v2 import apply_action
    start = time.monotonic()
    end = start + SEARCH_SECONDS
    seed = int.from_bytes(sha256(f"serving-search-v1:{state['version']}:{side}:{model.version}".encode()).digest(), 'big')
    remaining = SEARCH_SIMULATIONS
    try:
        actions, _ = runtime.decision(state, side)
        if len(actions) == 1:
            return deepcopy(actions[0])
        if state['phase'] == 'playing' and _public_terminal_allowed(state, side):
            world = sample_world(state, side, seed, runtime=runtime)
            world['events'] = []
            world.pop('_public_board', None)
            # Ordinary attacks first prevent a low policy prior from hiding a
            # public one-action win. No character-specific ranking or reward.
            ordered = sorted(actions, key=lambda a: a['type'] != 'attack')
            proof_checks = 0
            for action in ordered:
                if proof_checks >= 8 or remaining <= 1 or time.monotonic() >= end:
                    break
                if action['type'] not in ('attack', 'play_card', 'ultimate'):
                    continue
                after = apply_action(world, side, action)
                remaining -= 1
                proof_checks += 1
                # A proof must not depend on an unobserved draw/shuffle or RNG.
                if (after['phase'] == 'finished' and after.get('winner') == side
                        and after['rng'] == world['rng']
                        and not any(e['type'] in ('draw', 'gain') for e in after['events'])):
                    return deepcopy(action)
        if remaining <= 0 or time.monotonic() >= end:
            return fallback
        from .recovery_model import SCHEMAS
        options = dict(algorithm='gumbel')
        if getattr(model, 'schema', None) in SCHEMAS:
            from app.modules.card_game.rl.recovery_search import search
            options = dict(q_scale='natural_wdl', teacher_mode='network')
        result = search(state, side, {side: model, 'a' if side == 'b' else 'b': model},
                        seed=seed, simulations=remaining, **options,
                        gumbel_candidates=16, terminal_horizon=3, noise=False,
                        runtime=runtime, deadline=time.time() + max(0., end - time.monotonic()))
        # An interrupted search still has useful completed simulations. Never
        # replace their selected action with the original model argmax.
        if result.get('simulations', 0) > 0:
            selected = result['actions'][result['search_choice']]
            if selected in actions:
                return deepcopy(selected)
    except (ValueError, KeyError, TypeError, NotImplementedError):
        # Unsupported information-set roots (e.g. legacy private choices) keep
        # the exact pre-existing legal serving policy; never consult dark cards.
        return fallback
    return fallback
