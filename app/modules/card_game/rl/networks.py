"""GPU candidate scorer for MaskablePPO. Imported only from the training loop."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from .encoding import (
    ACTION_DIM,
    ACTION_LAYOUT,
    ACTION_TYPES,
    CARD_IDS,
    CARD_IDENTITY_SLICES,
    PUBLIC_CARD_ID_OFFSET,
    SHARED_ACTION_TYPES,
)
from .transfer import similar_card_sources


class StateOnlyExtractor(BaseFeaturesExtractor):
    """Keeps SB3's extractor contract; candidate scoring lives in the policy."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space['state'].shape[0])
        self.net = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        return self.net(observations['state'])


class CandidateScoringPolicy(MaskableMultiInputActorCriticPolicy):
    """Score generic operations and card identity separately.

    logit = generic(state, type/cost/target seats) + card(state, catalog id)
    Adapting a new deck can freeze the generic branch and only train card embeddings.
    """

    def __init__(self, *args, hidden_dim: int = 256, **kwargs):
        kwargs.setdefault('features_extractor_class', StateOnlyExtractor)
        kwargs.setdefault('features_extractor_kwargs', {'features_dim': hidden_dim})
        kwargs.setdefault('net_arch', dict(pi=[], vf=[]))
        kwargs.setdefault('share_features_extractor', True)
        self._hidden_dim = hidden_dim
        super().__init__(*args, **kwargs)
        self._transfer = {'teacher': None, 'kl_coef': 0.0, 'ent_coef': 0.01, 'overlap_card_indexes': ()}

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        hidden = self._hidden_dim
        cand_dim = int(self.observation_space['candidates'].shape[1])
        self.generic_encoder = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(self.device)
        self.generic_score_proj = nn.Linear(hidden, 1, bias=True).to(self.device)
        card_rows = 1 + (len(CARD_IDS) if cand_dim == ACTION_DIM else 0)
        self.card_embedding = nn.Embedding(card_rows, hidden, padding_idx=0).to(self.device)
        self.card_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(self.device)
        self.card_score_proj = nn.Linear(hidden, 1, bias=True).to(self.device)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def card_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.card_embedding, self.card_encoder, self.card_score_proj):
            params.extend(module.parameters())
        return params

    def generic_parameters(self) -> list[nn.Parameter]:
        card_ids = {id(param) for param in self.card_parameters()}
        return [param for param in self.parameters() if id(param) not in card_ids]

    def freeze_generic(self) -> None:
        for param in self.generic_parameters():
            param.requires_grad = False

    def attach_teacher(self, teacher: nn.Module | None, *, kl_coef: float, ent_coef: float,
                       overlap_card_ids: Sequence[str] = ()) -> None:
        indexes = []
        for card_id in overlap_card_ids:
            if card_id in CARD_IDS:
                indexes.append(CARD_IDS.index(card_id) + 1)
        self._transfer = {
            'teacher': teacher,
            'kl_coef': float(kl_coef),
            'ent_coef': float(ent_coef),
            'overlap_card_indexes': tuple(indexes),
        }

    def init_new_card_embeddings(
        self,
        previous_card_ids: Iterable[str],
        next_card_ids: Iterable[str],
    ) -> dict[str, list[str]]:
        sources = similar_card_sources(list(previous_card_ids), list(next_card_ids))
        initialized: list[str] = []
        with th.no_grad():
            weight = self.card_embedding.weight
            for card_id, similar in sources.items():
                dest = CARD_IDS.index(card_id) + 1
                if dest >= weight.shape[0] or not similar:
                    continue
                source = [CARD_IDS.index(item) + 1 for item in similar]
                weight[dest].copy_(weight[source].mean(dim=0))
                initialized.append(card_id)
        return {'initialized': initialized, 'new_cards': list(sources)}

    def _generic_candidates(self, candidates: th.Tensor) -> th.Tensor:
        if candidates.shape[-1] != ACTION_DIM:
            return candidates
        generic = candidates.clone()
        for region in CARD_IDENTITY_SLICES:
            generic[..., region] = 0
        return generic

    def _card_indices(self, candidates: th.Tensor) -> th.Tensor:
        batch, slots, dim = candidates.shape
        if dim != ACTION_DIM:
            return th.zeros(batch, slots, dtype=th.long, device=candidates.device)
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
            index = th.where(use, alt, index)
            found = found | use
        max_index = self.card_embedding.num_embeddings - 1
        index = th.where(found, index.clamp(max=max_index), th.zeros_like(index))
        return index

    def _shared_candidate_mask(self, candidates: th.Tensor, action_masks) -> th.Tensor:
        occupied = candidates.abs().sum(-1) > 0
        if action_masks is not None:
            masks = action_masks if isinstance(action_masks, th.Tensor) else th.as_tensor(
                action_masks, device=candidates.device)
            masks = masks.to(dtype=th.bool, device=candidates.device)
            if masks.shape != occupied.shape:
                masks = masks.reshape(occupied.shape)
            occupied = occupied & masks
        if candidates.shape[-1] != ACTION_DIM:
            return occupied
        types = candidates[..., ACTION_LAYOUT['action_type']]
        shared = th.zeros_like(occupied)
        for offset, action_type in enumerate(ACTION_TYPES):
            if action_type in SHARED_ACTION_TYPES:
                shared = shared | (types[..., offset] > 0.5)
        overlap = self._transfer.get('overlap_card_indexes') or ()
        if overlap:
            card_index = self._card_indices(candidates)
            overlap_mask = th.zeros(
                self.card_embedding.num_embeddings, dtype=th.bool, device=candidates.device)
            for item in overlap:
                if 0 < int(item) < overlap_mask.numel():
                    overlap_mask[int(item)] = True
            shared = shared | overlap_mask[card_index]
        return occupied & shared

    def _shared_action_kl(self, obs: PyTorchObs, action_masks, teacher: nn.Module) -> th.Tensor:
        student_logits, _values = self._logits_values(obs)
        with th.no_grad():
            teacher_logits, _teacher_values = teacher._logits_values(obs)
        shared = self._shared_candidate_mask(obs['candidates'], action_masks)
        valid = shared.any(dim=-1)
        fill = th.finfo(student_logits.dtype).min
        student_logp = th.log_softmax(student_logits.masked_fill(~shared, fill), dim=-1)
        teacher_logp = th.log_softmax(teacher_logits.masked_fill(~shared, fill), dim=-1)
        student_p = student_logp.exp()
        kl = (student_p * (student_logp - teacher_logp)).sum(dim=-1)
        return th.where(valid, kl.clamp(min=0.0), th.zeros_like(kl))

    def _logits_values(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        state_h = self.extract_features(obs, self.pi_features_extractor)
        candidates = obs['candidates']
        batch, slots, _dim = candidates.shape
        generic = self._generic_candidates(candidates)
        generic_h = self.generic_encoder(generic.reshape(batch * slots, -1)).reshape(batch, slots, -1)
        scale = math.sqrt(generic_h.shape[-1])
        logits = (generic_h * state_h.unsqueeze(1)).sum(-1) / scale
        logits = logits + self.generic_score_proj(generic_h).squeeze(-1)
        card_h = self.card_encoder(self.card_embedding(self._card_indices(candidates)))
        logits = logits + (card_h * state_h.unsqueeze(1)).sum(-1) / scale
        logits = logits + self.card_score_proj(card_h).squeeze(-1)
        occupied = candidates.abs().sum(-1, keepdim=True) > 0
        pooled = (generic_h * occupied).sum(1) / occupied.sum(1).clamp(min=1.0)
        values = self.value_head(th.cat([state_h, pooled], dim=-1))
        return logits, values

    def forward(self, obs: PyTorchObs, deterministic: bool = False, action_masks=None):
        logits, values = self._logits_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor, action_masks=None):
        logits, values = self._logits_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        entropy = distribution.entropy()
        teacher = self._transfer.get('teacher')
        kl_coef = float(self._transfer.get('kl_coef') or 0.0)
        if teacher is not None and kl_coef > 0:
            kl = self._shared_action_kl(obs, action_masks, teacher)
            ent_coef = max(float(self._transfer.get('ent_coef') or 0.01), 1e-8)
            scale = kl_coef / ent_coef
            if entropy is None:
                entropy = -distribution.log_prob(actions) - scale * kl
            elif entropy.ndim == 0:
                entropy = entropy - scale * kl.mean()
            else:
                entropy = entropy - scale * kl
        return values, distribution.log_prob(actions), entropy

    def get_distribution(self, obs: PyTorchObs, action_masks: np.ndarray | None = None):
        logits, _values = self._logits_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        _logits, values = self._logits_values(obs)
        return values

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data['hidden_dim'] = self._hidden_dim
        return data
