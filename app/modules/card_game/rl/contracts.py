"""Pure-memory contracts; policies must never receive internal game state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

Json = dict[str, Any]


RULES_VERSION = 'duel_v2'
DESIGN_VERSION = 'V2.4'
ACTION_CAPACITY = 256
SIDES = ('a', 'b')
PHASES = ('mulligan', 'playing', 'choice', 'finished')
WINNERS = ('a', 'b', 'draw')
ACTION_TYPES = (
    'mulligan',
    'attack',
    'play_card',
    'ultimate',
    'cycle',
    'choose',
    'end_turn',
    'concede',
)


@dataclass(frozen=True)
class Outcome:
    terminated: bool
    winner: str | None = None
    reason: str = ''

    def __post_init__(self) -> None:
        if self.terminated and self.winner not in ('a', 'b', 'draw'):
            raise ValueError('Finished game must identify winner a/b/draw')
        if not self.terminated and self.winner is not None:
            raise ValueError('Unfinished game cannot have a winner')


class Backend(Protocol):
    def new_game(self, *, seed: int = 0, **options: Any) -> Json: ...
    def current_player(self, state: Json) -> str | None: ...
    def observe(self, state: Json, side: str) -> Json: ...
    def apply_action(self, state: Json, side: str, action: Json) -> Json: ...
    def outcome(self, state: Json) -> Outcome: ...


class Encoder(Protocol):
    version: str
    observation_dim: int
    action_dim: int

    def encode_observation(self, observation: Json) -> Sequence[float]: ...
    def encode_action(self, observation: Json, action: Json) -> Sequence[float]: ...


class FrozenPolicy(Protocol):
    def __call__(self, observation: Json, legal_actions: tuple[Json, ...]) -> int:
        """Return one candidate index using only the supplied visible data."""
        ...
