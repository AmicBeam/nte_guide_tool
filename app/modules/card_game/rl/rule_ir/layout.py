"""Packed integer layout for the compiled starter engine.

The compiled kernel mutates one contiguous int32 row. Catalog tables are
compile-time constants, not part of the mutable row. Extra pending-play and
last-hit fields exist only so a choice or knockdown can span a step without a
Python rule fallback.
"""
from __future__ import annotations

from app.modules.card_game.rl.gpu_duel.catalog import (
    CHOICE_SLOTS, DECK_SLOTS, DISCARD_SLOTS, HAND_SLOTS, MAX_LEGAL, N_SEATS, SIDES,
)

ENGINE_NAME = 'starter_engine'
ENGINE_VERSION = 'compiled_public_v2'

SIDE_SCALARS = (
    'hp', 'shield', 'ap', 'front', 'last_front', 'turn_count', 'fatigue',
    'normal_atk', 'ultimate_ok', 'used_instant', 'extra_flower', 'extra_ap',
    'extra_genesis', 'extra_genesis_actor', 'surplus', 'nanali_seed',
    'nanali_seed_played', 'burn_left', 'delay_end', 'hand_n', 'deck_n',
    'discard_n', 'removed_n', 'weave', 'ultimates_used', 'burn_by',
    'burn_infinite', 'harmony_damage',
)
CHAR_FIELDS = (
    'ch_hp', 'ch_max_hp', 'ch_base_max', 'ch_shield', 'ch_base_atk', 'ch_growth',
    'ch_harmony', 'ch_energy', 'ch_energy_max', 'ch_down', 'ch_awakened',
    'ch_ult_turns', 'ch_shape', 'ch_atk_buff', 'ch_next_bonus', 'ch_next_shield',
    'ch_next_followup', 'ch_pact', 'ch_energy_next', 'ch_harmonized', 'ch_present',
    'ch_allied_hurt', 'ch_pending_atk', 'ch_share_overflow',
    'ch_collapse_count', 'ch_collapse_by', 'ch_collapse_until',
    'ch_hp_floor', 'ch_order', 'ch_ultimate_used_turn',
)
ROW_SCALARS = (
    'phase', 'active', 'first', 'turn', 'winner', 'reason', 'rng', 'next_instance',
    'pending_kind', 'pending_side', 'pending_count',
    'pending_play_kind', 'pending_play_inst', 'error', 'action',
    'escalation_enabled',
)
GPU_STATE_SCALARS = (
    'phase', 'active', 'first', 'turn', 'winner', 'reason', 'rng', 'next_instance',
    'pending_kind', 'pending_side', 'pending_count', 'escalation_enabled',
)
LEGAL_FIELDS = ('legal_type', 'legal_actor', 'legal_hand', 'legal_tside', 'legal_tseat')


def _build_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0

    def place(name: str, width: int) -> None:
        nonlocal cursor
        offsets[name] = cursor
        cursor += width

    for name in ROW_SCALARS:
        place(name, 1)
    place('pending_card', CHOICE_SLOTS)
    place('pending_inst', CHOICE_SLOTS)
    for name in SIDE_SCALARS:
        place(name, SIDES)
    place('hand_kind', SIDES * HAND_SLOTS)
    place('hand_inst', SIDES * HAND_SLOTS)
    place('deck_kind', SIDES * DECK_SLOTS)
    place('deck_inst', SIDES * DECK_SLOTS)
    place('discard_kind', SIDES * DISCARD_SLOTS)
    place('discard_inst', SIDES * DISCARD_SLOTS)
    place('hand_revealed', SIDES * HAND_SLOTS)
    for name in CHAR_FIELDS:
        place(name, SIDES * N_SEATS)
    place('last_hit_side', SIDES * N_SEATS)
    place('last_hit_seat', SIDES * N_SEATS)
    for name in LEGAL_FIELDS:
        place(name, MAX_LEGAL)
    place('legal_n', 1)
    offsets['_width'] = cursor
    return offsets


OFFSETS = _build_offsets()
ROW_WIDTH = OFFSETS['_width']


def c_defines() -> str:
    lines = [
        f'#define ROW_WIDTH {ROW_WIDTH}',
        f'#define N_SIDES {SIDES}',
        f'#define N_SEATS {N_SEATS}',
        f'#define HAND_SLOTS {HAND_SLOTS}',
        f'#define DECK_SLOTS {DECK_SLOTS}',
        f'#define DISCARD_SLOTS {DISCARD_SLOTS}',
        f'#define CHOICE_SLOTS {CHOICE_SLOTS}',
        f'#define MAX_LEGAL {MAX_LEGAL}',
    ]
    for name, off in OFFSETS.items():
        if name.startswith('_'):
            continue
        lines.append(f'#define OFF_{name.upper()} {off}')
    return '\n'.join(lines)
