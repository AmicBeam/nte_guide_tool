"""Bind only the frozen four-character V2 engine, never the old cover-card AI."""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from typing import Any

from .contracts import DESIGN_VERSION, Json, Outcome, RULES_VERSION


class V2Backend:
    rules_version = RULES_VERSION
    design_version = DESIGN_VERSION

    def __init__(self, engine: Any = None, *, copy_on_apply: bool = True) -> None:
        # Injection supports bounded synthetic contract tests before engine delivery.
        self._engine = engine
        self.copy_on_apply = bool(copy_on_apply)

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = import_module('app.modules.card_game.engine.duel_v2')
        return self._engine

    def new_game(self, *, seed: int = 0, **options: Any) -> Json:
        return deepcopy(self.engine.new_game(seed=seed, **deepcopy(options)))

    def current_player(self, state: Json) -> str | None:
        return self.engine.acting_side(state)

    def observe(self, state: Json, side: str) -> Json:
        projection = self.engine.observe(state, side)
        if projection.get('rules_version') != self.rules_version:
            raise ValueError('RL requires a duel_v2 projection')
        if projection.get('viewer_side') != side:
            raise ValueError('Projection viewer does not match requested seat')
        excluded = {c['instance_id'] for c in projection.get('sides', {}).get(side, {}).get('hand', [])
                    if c.get('training_excluded')}
        projection['legal_actions'] = [entry for entry in projection['legal_actions']
                                       if entry['action'].get('card_id') not in excluded]
        return deepcopy(projection)

    def legal_actions(self, state: Json, side: str) -> list[Json]:
        return [deepcopy(item['action']) for item in self.observe(state, side)['legal_actions']]

    def apply_action(self, state: Json, side: str, action: Json) -> Json:
        if action.get('type') == 'play_card' and any(
                card.get('training_excluded') and card.get('instance_id') == action.get('card_id')
                for card in state.get('sides', {}).get(side, {}).get('hand', [])):
            raise ValueError('此牌不参与人机训练')
        incoming = deepcopy(state) if self.copy_on_apply else state
        payload = deepcopy(action) if self.copy_on_apply else action
        result = self.engine.apply_action(incoming, side, payload)
        if result is incoming:
            result = deepcopy(result)
        return result

    def outcome(self, state: Json) -> Outcome:
        return Outcome(state['phase'] == 'finished', state.get('winner'), str(state.get('reason') or ''))
