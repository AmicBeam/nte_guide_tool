"""CPU Oracle Backend for Duel V2 batched search equivalence testing.

Provides ground-truth CPU rule execution by delegating to the official Duel V2
Python rule engine and public information-set sampler.
This is strictly a CPU reference implementation and does NOT claim GPU, CUDA,
or native array acceleration.
"""
from typing import Any, List, Optional, Tuple
import numpy as np


class CPUOracleBackend:
    """Explicit CPU oracle backend wrapping official Duel V2 rules and samplers."""

    def __init__(self, *, runtime: Optional[Any] = None, fast_simulation: bool = True) -> None:
        self.runtime = runtime
        self.fast_simulation = bool(fast_simulation)

    backend_kind = 'python_oracle'
    supports_gpu = False

    def serving_root_metadata(self,state,side):
        from ..batched_duel.serving_protocol import root_metadata
        return root_metadata(state,side)

    def public_terminal_probe(self,state,side,action):
        from ..batched_duel.serving_protocol import public_terminal_probe
        return public_terminal_probe(state,side,action)

    def value_observation(self, state, viewer):
        from app.modules.card_game.engine.duel_v2 import observe
        view = observe(state, viewer, include_previews=False)
        if self.runtime is not None and hasattr(self.runtime, 'encode'):
            return self.runtime.encode(view, [])
        from app.modules.card_game.rl.league_observation import encode_league
        return encode_league(view, [])

    def acting_side(self, state: Any) -> Optional[str]:
        from app.modules.card_game.engine.duel_v2 import acting_side
        return acting_side(state)

    def finished(self, state: Any) -> bool:
        if isinstance(state, dict):
            return state.get('phase') == 'finished'
        return getattr(state, 'phase', None) == 'finished'

    def terminal_value(self, state: Any, viewer: str) -> float:
        winner = state.get('winner') if isinstance(state, dict) else getattr(state, 'winner', None)
        if winner not in ('a', 'b'):
            return 0.0
        return 1.0 if winner == viewer else -1.0

    def decision(self, state: Any, side: str) -> Tuple[List[Any], Tuple[np.ndarray, np.ndarray]]:
        if self.runtime is not None and hasattr(self.runtime, 'decision'):
            return self.runtime.decision(state, side)
        from app.modules.card_game.rl import league_rollout
        return league_rollout.decision(state, side)

    def sample_world(self, state: Any, viewer: str, seed: int) -> Any:
        from app.modules.card_game.rl import information_search
        return information_search.sample_world(state, viewer, seed, runtime=self.runtime)

    def step(self, state: Any, side: str, action: Any) -> Any:
        if self.fast_simulation:
            from app.modules.card_game.engine.duel_v2.simulation import simulate_action
            return simulate_action(state, side, action)
        from app.modules.card_game.engine.duel_v2 import apply_action
        return apply_action(state, side, action)

    def clean(self, state: Any) -> Any:
        from app.modules.card_game.rl import league_rollout
        return league_rollout.clean(state)
