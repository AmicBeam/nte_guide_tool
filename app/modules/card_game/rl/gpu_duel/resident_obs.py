"""Vectorized public-state encoding for resident training and Python evaluation.

No Python JSON construction, deck contents, opponent hand identities, or RNG
enter the policy. Checkpoint metadata identifies this training-only schema.
"""
from __future__ import annotations

import torch

from .catalog import CARD_IDS, HAND_SLOTS, MAX_LEGAL
from .observe import compact_observe

from .resident_schema import SCHEMA, SIDE_FIELDS, CHAR_FIELDS


def resident_observe(s):
    b = torch.arange(s.n, device=s.device)
    viewer = s.active.long()
    sides = torch.stack((viewer, 1 - viewer), dim=1)
    batch = b[:, None]
    pieces = [(s.turn.float() / 30).unsqueeze(1), (viewer == s.first).float().unsqueeze(1)]
    for name in SIDE_FIELDS:
        pieces.append(getattr(s, name)[batch, sides].float() / 10)
    for name in CHAR_FIELDS:
        pieces.append(getattr(s, name)[batch, sides].flatten(1).float() / 10)
    # Complete own hand, including cards currently unusable; discard counts public.
    own = s.hand_kind[b, viewer]
    hand_counts = torch.zeros(s.n, len(CARD_IDS), device=s.device)
    hand_counts.scatter_add_(1, own.clamp(min=0).long(), (own >= 0).float())
    pieces.append(hand_counts / 2)
    for side in (viewer, 1 - viewer):
        kinds = s.discard_kind[b, side]
        counts = torch.zeros_like(hand_counts)
        valid = (torch.arange(kinds.shape[1], device=s.device)[None] < s.discard_n[b, side, None]) & (kinds >= 0)
        counts.scatter_add_(1, kinds.clamp(min=0).long(), valid.float())
        pieces.append(counts / 2)
    state = torch.cat(pieces, dim=1)
    hs = s.legal_hand.clamp(min=0, max=HAND_SLOTS - 1).long()
    kinds = own.gather(1, hs)
    kinds = torch.where(s.legal_hand >= 0, kinds, -1)
    # A choice selects a pending card, not a hand card.
    from .catalog import ACT_CHOOSE, CHOICE_SLOTS
    choice_kinds = s.pending_card.gather(1, s.legal_hand.clamp(min=0, max=CHOICE_SLOTS - 1).long())
    kinds = torch.where(s.legal_type == ACT_CHOOSE, choice_kinds, kinds)
    idx = kinds.clamp(min=0).long()
    valid_card = kinds >= 0
    def attr(name):
        return torch.where(valid_card, getattr(s, name)[idx], 0).float()
    target_side = torch.where(s.legal_tside < 0, -1, (s.legal_tside != viewer[:, None]).int())
    cand = torch.stack((s.legal_type.float(), s.legal_actor.float(), kinds.float(),
                        target_side.float(), s.legal_tseat.float(),
                        attr('catalog_cost'), attr('catalog_atk'), attr('catalog_shield'),
                        attr('catalog_hp'), attr('catalog_instant')), dim=-1)
    return state, cand, s.legal_type >= 0
