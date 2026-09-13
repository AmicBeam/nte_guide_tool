"""Candidate scorer with the same generic+card split as CandidateScoringPolicy."""
from __future__ import annotations

import math

import torch
from torch import nn

from app.modules.card_game.rl.encoding import (
    ACTION_DIM, ACTION_LAYOUT, CARD_IDS, CARD_IDENTITY_SLICES, PUBLIC_CARD_ID_OFFSET,
)


class GpuCandidatePolicy(nn.Module):
    """Website-encoder compatible scorer. Weights load without SB3."""

    def __init__(self, obs_dim: int, cand_dim: int, hidden: int = 256, n_cards: int | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.cand_dim = cand_dim
        self.hidden = hidden
        n_cards = 1 + (n_cards if n_cards is not None else len(CARD_IDS))
        self.state_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.generic_encoder = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.generic_score_proj = nn.Linear(hidden, 1)
        self.card_embedding = nn.Embedding(n_cards, hidden, padding_idx=0)
        self.card_encoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.card_score_proj = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def _generic(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.shape[-1] != ACTION_DIM:
            return candidates
        generic = candidates.clone()
        for region in CARD_IDENTITY_SLICES:
            generic[..., region] = 0
        return generic

    def _card_indices(self, candidates: torch.Tensor) -> torch.Tensor:
        batch, slots, dim = candidates.shape
        if dim != ACTION_DIM:
            return torch.zeros(batch, slots, dtype=torch.long, device=candidates.device)
        card_one_hot = candidates[..., ACTION_LAYOUT['action_card_id']]
        index = card_one_hot.argmax(-1) + 1
        found = card_one_hot.sum(-1) > 0
        for key in ('played_card', 'choice_card'):
            block = ACTION_LAYOUT[key]
            visible = candidates[..., block.start] > 0.5
            start = block.start + PUBLIC_CARD_ID_OFFSET
            block_one_hot = candidates[..., start:start + len(CARD_IDS)]
            alt = block_one_hot.argmax(-1) + 1
            use = (~found) & visible & (block_one_hot.sum(-1) > 0)
            index = torch.where(use, alt, index)
            found = found | use
        max_index = self.card_embedding.num_embeddings - 1
        return torch.where(found, index.clamp(max=max_index), torch.zeros_like(index))

    def forward(self, state, cand, mask):
        state_h = self.state_net(state)
        batch, slots, _dim = cand.shape
        generic = self._generic(cand)
        generic_h = self.generic_encoder(generic.reshape(batch * slots, -1)).reshape(batch, slots, -1)
        scale = math.sqrt(generic_h.shape[-1])
        logits = (generic_h * state_h.unsqueeze(1)).sum(-1) / scale
        logits = logits + self.generic_score_proj(generic_h).squeeze(-1)
        card_h = self.card_encoder(self.card_embedding(self._card_indices(cand)))
        logits = logits + (card_h * state_h.unsqueeze(1)).sum(-1) / scale
        logits = logits + self.card_score_proj(card_h).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        occupied = cand.abs().sum(-1, keepdim=True) > 0
        pooled = (generic_h * occupied).sum(1) / occupied.sum(1).clamp(min=1.0)
        values = self.value_head(torch.cat([state_h, pooled], dim=-1)).squeeze(-1)
        return logits, values
