"""Explicit compiled starter-deck backend. Not imported by the website or default trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.modules.card_game.content.duel_v2 import STARTER_DECK
from app.modules.card_game.rl.gpu_duel.catalog import assert_gpu_deck
from app.modules.card_game.rl.rule_ir.completeness import (
    COMPILED_DECK_ID, CompletenessError, compiled_rule_hash, support_manifest,
    validate_compiled_deck,
)
from app.modules.card_game.rl.rule_ir.compile_engine import compile_cuda_engine, compile_native_engine
from app.modules.card_game.rl.rule_ir.layout import ENGINE_NAME, ENGINE_VERSION, ROW_WIDTH
from app.modules.card_game.rl.rule_ir.pack_row import pack_gpu_row, pack_python_row, unpack_gpu_row


class CompiledEngineError(RuntimeError):
    pass


def _public_reset_flags(escalation=True, mulligan=False) -> int:
    """Pack public reset options into the existing escalation integer.

    bit0: escalation enabled; bit1: optional opening mulligan.
    """
    return (1 if escalation else 0) | (2 if mulligan else 0)


def _assert_compiled_deck(deck: dict[str, Any]) -> dict[str, Any]:
    from app.modules.card_game.content.duel_v2 import STARTER_DECKS
    payload = assert_gpu_deck(deck)
    for candidate in STARTER_DECKS:
        if candidate['id'] in ('starter', 'weave-rush') and payload['character_ids'] == candidate['character_ids'] and payload['card_ids'] == candidate['card_ids']:
            return candidate
    raise CompletenessError('compiled backend requires an exact supported preset')


class CompiledStarterBackend:
    """Owns a compiled CPU or CUDA engine. Callers must construct this explicitly."""

    def __init__(self, directory: str | Path | None = None, *, backend: str = 'native', deck_id='starter'):
        if backend not in ('native', 'cuda'):
            raise ValueError(f'unsupported compiled backend {backend!r}')
        self.deck_id = deck_id
        self.manifest = validate_compiled_deck(deck_id)
        self.backend = backend
        self.rule_hash = compiled_rule_hash(deck_id)
        self.engine_name = ENGINE_NAME
        self.engine_version = ENGINE_VERSION
        self.row_width = ROW_WIDTH
        self.stats = {'steps': 0, 'rows': 0, 'errors': 0, 'fallback_rows': 0}
        if backend == 'native':
            if directory is None:
                raise ValueError('native compiled backend requires an explicit cache directory')
            self.launch = compile_native_engine(Path(directory) / deck_id, deck_id=deck_id)
        else:
            self.launch = compile_cuda_engine(deck_id=deck_id)
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        closer = getattr(self.launch, 'close', None)
        if closer:
            closer()
        self.closed = True

    def identity(self) -> dict[str, Any]:
        return {
            'backend': self.backend,
            'engine_name': self.engine_name,
            'engine_version': self.engine_version,
            'rule_hash': self.rule_hash,
            'row_width': self.row_width,
            'fallback_rows': self.stats['fallback_rows'],
            **support_manifest(self.deck_id),
            'supports_mulligan': True,
        }

    @property
    def supports_mulligan(self) -> bool:
        return True

    def step_lists(self, rows: list[list[int]], actions: list[int]) -> list[list[int]]:
        if self.closed:
            raise CompiledEngineError('compiled backend is closed')
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        launch_lists = getattr(self.launch, 'launch_lists', None)
        if launch_lists is None:
            raise CompiledEngineError('compiled backend has no list launcher; no Python/tensor fallback')
        out = launch_lists(rows, actions)
        self.stats['steps'] += 1
        self.stats['rows'] += len(out)
        errors = sum(1 for row in out if row[OFFSETS['error']])
        if errors:
            self.stats['errors'] += errors
            raise CompiledEngineError(
                f'compiled engine hit {errors} unsupported/error rows; no Python/tensor fallback'
            )
        return out

    def step_rows(self, rows, actions):
        if self.closed:
            raise CompiledEngineError('compiled backend is closed')
        import torch
        if not torch.is_tensor(rows):
            return self.step_lists(rows, list(actions))
        if actions.dtype != torch.int32:
            actions = actions.to(torch.int32)
        self.launch(rows, actions)
        self.stats['steps'] += 1
        self.stats['rows'] += int(rows.shape[0])
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        errors = int((rows[:, OFFSETS['error']] != 0).sum())
        if errors:
            self.stats['errors'] += errors
            raise CompiledEngineError(
                f'compiled engine hit {errors} unsupported/error rows; no Python/tensor fallback'
            )
        return rows

    def step_python(self, states: list[dict[str, Any]], actions: list[int]) -> list[list[int]]:
        packed = [pack_python_row(state) for state in states]
        return self.step_lists(packed, list(actions))

    def reset_lists(self, seeds: list[int]) -> list[list[int]]:
        if self.closed:
            raise CompiledEngineError('compiled backend is closed')
        reset_lists = getattr(self.launch, 'reset_lists', None)
        if reset_lists is None:
            raise CompiledEngineError('compiled backend has no compiled reset; no Python/tensor fallback')
        from app.modules.card_game.rl.rule_ir.pack_row import empty_row
        rows = [empty_row() for _ in seeds]
        out = reset_lists(rows, list(seeds))
        self.stats['steps'] += 1
        self.stats['rows'] += len(out)
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        errors = sum(1 for row in out if row[OFFSETS['error']])
        if errors:
            self.stats['errors'] += errors
            raise CompiledEngineError(
                f'compiled reset hit {errors} unsupported/error rows; no Python/tensor fallback'
            )
        return out

    def _state_rows(self, state):
        from app.modules.card_game.rl.rule_ir.pack_row import bind_gpu_rows
        wanted = 'cuda' if self.backend == 'cuda' else 'cpu'
        if state.device.type != wanted:
            raise ValueError(f'{self.backend} executor requires {wanted} resident state')
        return bind_gpu_rows(state)

    def step_gpu_state(self, state, actions) -> None:
        import torch
        rows = self._state_rows(state)
        actions = torch.as_tensor(actions, dtype=torch.int32, device=state.device).contiguous()
        self.step_rows(rows, actions)

    def reset_gpu_state(self, state, seeds) -> None:
        import torch
        if self.closed:
            raise CompiledEngineError('compiled backend is closed')
        rows = self._state_rows(state)
        seeds = torch.as_tensor(seeds, dtype=torch.int32, device=state.device).contiguous()
        if seeds.ndim != 1 or seeds.shape[0] != state.n:
            raise ValueError('Reset seeds must have one entry per environment')
        self.launch.reset(rows, seeds)
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        if bool((rows[:, OFFSETS['error']] != 0).any()):
            raise CompiledEngineError('Compiled reset failed')

    def reset_public_gpu_state(self, state, seeds, deck_rows_a, deck_rows_b, escalation=True, mulligan=False) -> None:
        """Reset compiled rows from public custom four-person decks.

        seeds: int32 [n]; a negative seed preserves that row.
        deck_rows_a/b: int32 [n,36] with the first 4 values as seat indices in
        actual team order and the remaining 32 as card-kind indices. All tensors
        must already live on the compiled executor device.
        escalation=True enables global-turn 6/13/20 white-heat rules; False keeps
        the legacy old-model path with one ultimate and original resource caps.
        mulligan=False preserves the keep-all playing-start reset. True starts in
        PHASE_MULLIGAN with 5 original cards per side and no first-turn draw.
        """
        import torch
        if self.closed:
            raise CompiledEngineError('compiled backend is closed')
        public_reset = getattr(self.launch, 'public_reset', None)
        if public_reset is None:
            raise CompiledEngineError('compiled backend has no public reset; no Python/tensor fallback')
        rows = self._state_rows(state)
        seeds = torch.as_tensor(seeds, dtype=torch.int32, device=state.device).contiguous()
        deck_rows_a = torch.as_tensor(deck_rows_a, dtype=torch.int32, device=state.device).contiguous()
        deck_rows_b = torch.as_tensor(deck_rows_b, dtype=torch.int32, device=state.device).contiguous()
        if seeds.ndim != 1 or seeds.shape[0] != state.n:
            raise ValueError('Reset seeds must have one entry per environment')
        if deck_rows_a.shape != (state.n, 36) or deck_rows_b.shape != (state.n, 36):
            raise ValueError('public deck rows must be int32 [n,36]')
        # torch.device('cuda') is an alias; allocated tensors report cuda:<index>.
        if any(t.device != rows.device for t in (seeds, deck_rows_a, deck_rows_b)):
            raise ValueError('public reset tensors must already live on the compiled device')
        public_reset(rows, seeds, deck_rows_a, deck_rows_b, _public_reset_flags(escalation, mulligan))
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        if bool((rows[:, OFFSETS['error']] != 0).any()):
            raise CompiledEngineError('Compiled public reset failed')

    def reset_public_lists(self, seeds, deck_rows_a, deck_rows_b, escalation=True, mulligan=False):
        if self.closed:
            raise CompiledEngineError('compiled backend is closed')
        public_reset_lists = getattr(self.launch, 'public_reset_lists', None)
        if public_reset_lists is None:
            raise CompiledEngineError('compiled backend has no public reset; no Python/tensor fallback')
        from app.modules.card_game.rl.rule_ir.pack_row import empty_row
        rows = [empty_row() for _ in seeds]
        out = public_reset_lists(rows, list(seeds), list(deck_rows_a), list(deck_rows_b), _public_reset_flags(escalation, mulligan))
        from app.modules.card_game.rl.rule_ir.layout import OFFSETS
        errors = sum(1 for row in out if row[OFFSETS['error']])
        if errors:
            raise CompiledEngineError(
                f'compiled public reset hit {errors} unsupported/error rows; no Python/tensor fallback'
            )
        return out
