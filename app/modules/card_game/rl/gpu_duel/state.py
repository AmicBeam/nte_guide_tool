"""Fixed-length integer GPU duel state. One row is one independent game."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .catalog import (
    CHOICE_SLOTS, DECK_SLOTS, DISCARD_SLOTS, HAND_SLOTS, MAX_LEGAL, N_SEATS,
    REMOVED_SLOTS, SIDES, card_table,
)


def _i32(n: int, *shape: int, device) -> torch.Tensor:
    return torch.zeros((n, *shape), dtype=torch.int32, device=device)


@dataclass
class GpuState:
    n: int
    device: torch.device
    phase: torch.Tensor
    active: torch.Tensor
    first: torch.Tensor
    turn: torch.Tensor
    winner: torch.Tensor
    reason: torch.Tensor
    rng: torch.Tensor
    next_instance: torch.Tensor
    pending_kind: torch.Tensor
    pending_side: torch.Tensor
    pending_count: torch.Tensor
    pending_card: torch.Tensor
    pending_inst: torch.Tensor
    pending_play_kind: torch.Tensor
    pending_play_inst: torch.Tensor
    escalation_enabled: torch.Tensor
    hp: torch.Tensor
    shield: torch.Tensor
    ap: torch.Tensor
    front: torch.Tensor
    last_front: torch.Tensor
    turn_count: torch.Tensor
    fatigue: torch.Tensor
    normal_atk: torch.Tensor
    ultimate_ok: torch.Tensor
    used_instant: torch.Tensor
    extra_flower: torch.Tensor
    extra_ap: torch.Tensor
    extra_genesis: torch.Tensor
    extra_genesis_actor: torch.Tensor
    surplus: torch.Tensor
    nanali_seed: torch.Tensor
    nanali_seed_played: torch.Tensor
    burn_left: torch.Tensor
    delay_end: torch.Tensor
    ultimates_used: torch.Tensor
    burn_by: torch.Tensor
    burn_infinite: torch.Tensor
    harmony_damage: torch.Tensor
    hand_revealed: torch.Tensor
    hand_n: torch.Tensor
    deck_n: torch.Tensor
    discard_n: torch.Tensor
    removed_n: torch.Tensor
    hand_kind: torch.Tensor
    hand_inst: torch.Tensor
    deck_kind: torch.Tensor
    deck_inst: torch.Tensor
    discard_kind: torch.Tensor
    discard_inst: torch.Tensor
    ch_hp: torch.Tensor
    ch_max_hp: torch.Tensor
    ch_base_max: torch.Tensor
    ch_shield: torch.Tensor
    ch_base_atk: torch.Tensor
    ch_growth: torch.Tensor
    ch_harmony: torch.Tensor
    ch_energy: torch.Tensor
    ch_energy_max: torch.Tensor
    ch_down: torch.Tensor
    ch_awakened: torch.Tensor
    ch_ultimate_used_turn: torch.Tensor
    ch_ult_turns: torch.Tensor
    ch_shape: torch.Tensor
    ch_atk_buff: torch.Tensor
    ch_next_bonus: torch.Tensor
    ch_next_shield: torch.Tensor
    ch_next_followup: torch.Tensor
    ch_pact: torch.Tensor
    ch_energy_next: torch.Tensor
    ch_harmonized: torch.Tensor
    ch_present: torch.Tensor
    ch_allied_hurt: torch.Tensor
    ch_pending_atk: torch.Tensor
    ch_share_overflow: torch.Tensor
    ch_collapse_count: torch.Tensor
    ch_collapse_by: torch.Tensor
    ch_collapse_until: torch.Tensor
    ch_hp_floor: torch.Tensor
    ch_order: torch.Tensor
    weave: torch.Tensor
    catalog_cost: torch.Tensor
    catalog_atk: torch.Tensor
    catalog_shield: torch.Tensor
    catalog_hp: torch.Tensor
    catalog_type: torch.Tensor
    catalog_owner: torch.Tensor
    catalog_response: torch.Tensor
    catalog_instant: torch.Tensor
    catalog_derived: torch.Tensor
    catalog_policy: torch.Tensor
    catalog_require: torch.Tensor
    char_attr: torch.Tensor
    char_base_atk: torch.Tensor
    char_base_hp: torch.Tensor
    char_energy_max: torch.Tensor
    char_ult_kind: torch.Tensor
    char_ult_turns: torch.Tensor
    legal_type: torch.Tensor
    legal_actor: torch.Tensor
    legal_hand: torch.Tensor
    legal_tside: torch.Tensor
    legal_tseat: torch.Tensor
    legal_n: torch.Tensor

    def to(self, device) -> 'GpuState':
        device = torch.device(device)
        payload = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = value.to(device) if torch.is_tensor(value) else value
        payload['device'] = device
        return GpuState(**payload)


def empty_state(n: int, device: str | torch.device = 'cpu') -> GpuState:
    device = torch.device(device)
    table = card_table()
    n_cards = len(table['cost'])
    def t(*shape):
        return _i32(n, *shape, device=device)
    def cat(values):
        return torch.tensor(values, dtype=torch.int32, device=device)
    return GpuState(
        n=n, device=device,
        phase=t(), active=t(), first=t(), turn=t(),
        winner=t() - 1, reason=t(), rng=t(), next_instance=t(),
        pending_kind=t() - 1, pending_side=t() - 1, pending_count=t(),
        pending_card=t(CHOICE_SLOTS) - 1, pending_inst=t(CHOICE_SLOTS),
        pending_play_kind=t() - 1, pending_play_inst=t(),
        escalation_enabled=t() + 1,
        hp=t(SIDES), shield=t(SIDES), ap=t(SIDES),
        front=t(SIDES) - 1, last_front=t(SIDES) - 1,
        turn_count=t(SIDES), fatigue=t(SIDES),
        normal_atk=t(SIDES), ultimate_ok=t(SIDES),
        used_instant=t(SIDES), extra_flower=t(SIDES), extra_ap=t(SIDES),
        extra_genesis=t(SIDES), extra_genesis_actor=t(SIDES) - 1, surplus=t(SIDES),
        nanali_seed=t(SIDES), nanali_seed_played=t(SIDES),
        burn_left=t(SIDES), delay_end=t(SIDES),
        ultimates_used=t(SIDES), burn_by=t(SIDES) - 1, burn_infinite=t(SIDES),
        harmony_damage=t(SIDES),
        hand_n=t(SIDES), deck_n=t(SIDES), discard_n=t(SIDES), removed_n=t(SIDES),
        hand_revealed=t(SIDES, HAND_SLOTS),
        hand_kind=t(SIDES, HAND_SLOTS) - 1, hand_inst=t(SIDES, HAND_SLOTS),
        deck_kind=t(SIDES, DECK_SLOTS) - 1, deck_inst=t(SIDES, DECK_SLOTS),
        discard_kind=t(SIDES, DISCARD_SLOTS) - 1, discard_inst=t(SIDES, DISCARD_SLOTS),
        ch_hp=t(SIDES, N_SEATS), ch_max_hp=t(SIDES, N_SEATS), ch_base_max=t(SIDES, N_SEATS),
        ch_shield=t(SIDES, N_SEATS), ch_base_atk=t(SIDES, N_SEATS), ch_growth=t(SIDES, N_SEATS),
        ch_harmony=t(SIDES, N_SEATS), ch_energy=t(SIDES, N_SEATS), ch_energy_max=t(SIDES, N_SEATS),
        ch_down=t(SIDES, N_SEATS), ch_awakened=t(SIDES, N_SEATS), ch_ult_turns=t(SIDES, N_SEATS),
        ch_ultimate_used_turn=t(SIDES, N_SEATS) - 1, ch_shape=t(SIDES, N_SEATS) - 1, ch_atk_buff=t(SIDES, N_SEATS),
        ch_next_bonus=t(SIDES, N_SEATS), ch_next_shield=t(SIDES, N_SEATS),
        ch_next_followup=t(SIDES, N_SEATS), ch_pact=t(SIDES, N_SEATS),
        ch_energy_next=t(SIDES, N_SEATS), ch_harmonized=t(SIDES, N_SEATS),
        ch_present=t(SIDES, N_SEATS), ch_allied_hurt=t(SIDES, N_SEATS),
        ch_pending_atk=t(SIDES, N_SEATS), ch_share_overflow=t(SIDES, N_SEATS),
        ch_collapse_count=t(SIDES, N_SEATS), ch_collapse_by=t(SIDES, N_SEATS) - 1,
        ch_collapse_until=t(SIDES, N_SEATS),
        ch_hp_floor=t(SIDES, N_SEATS), ch_order=t(SIDES, N_SEATS),
        weave=t(SIDES),
        catalog_cost=cat(table['cost']), catalog_atk=cat(table['atk']),
        catalog_shield=cat(table['shield']), catalog_hp=cat(table['hp']),
        catalog_type=cat(table['type']), catalog_owner=cat(table['owner']),
        catalog_response=cat(table['response']), catalog_instant=cat(table['instant']),
        catalog_derived=cat(table['derived']), catalog_policy=cat(table['policy']),
        catalog_require=cat(table['require']),
        char_attr=cat(table['attr']), char_base_atk=cat(table['base_atk']),
        char_base_hp=cat(table['base_hp']), char_energy_max=cat(table['energy_max']),
        char_ult_kind=cat(table['ult_kind']), char_ult_turns=cat(table['ult_turns']),
        legal_type=t(MAX_LEGAL) - 1, legal_actor=t(MAX_LEGAL) - 1,
        legal_hand=t(MAX_LEGAL) - 1, legal_tside=t(MAX_LEGAL) - 1,
        legal_tseat=t(MAX_LEGAL) - 1, legal_n=t(),
    )


def other_side(side: torch.Tensor) -> torch.Tensor:
    return 1 - side
