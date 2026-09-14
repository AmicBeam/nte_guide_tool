"""Evaluate a GPU-trained encoder policy on the Python engine oracle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.modules.card_game.rl.budget import TrainBudget
from app.modules.card_game.rl.encoding import ACTION_DIM, OBSERVATION_DIM, V2Encoder
from app.modules.card_game.rl.evaluate import _play_match
from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
from app.modules.card_game.rl.transfer import default_training_deck

from .catalog import MAX_LEGAL
from .policy import GpuCandidatePolicy


def load_gpu_policy(path: str | Path, device: str = 'cpu') -> GpuCandidatePolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim = int(ckpt.get('state_dim') or OBSERVATION_DIM)
    cand_dim = int(ckpt.get('cand_dim') or ACTION_DIM)
    hidden = int(ckpt.get('hidden') or 256)
    net = GpuCandidatePolicy(obs_dim, cand_dim, hidden=hidden)
    net.load_state_dict(ckpt['model'])
    net.to(device).eval()
    return net


class GpuWeightPolicy:
    def __init__(self, net: GpuCandidatePolicy, device: str = 'cpu'):
        self.net = net
        self.device = device
        self.encoder = V2Encoder()

    def __call__(self, observation: dict[str, Any], legal_actions: tuple[dict[str, Any], ...]) -> int:
        state = torch.tensor([self.encoder.encode_observation(observation)], dtype=torch.float32, device=self.device)
        cand = torch.zeros(1, MAX_LEGAL, ACTION_DIM, dtype=torch.float32, device=self.device)
        mask = torch.zeros(1, MAX_LEGAL, dtype=torch.bool, device=self.device)
        for index, action in enumerate(legal_actions):
            if index >= MAX_LEGAL:
                break
            if action.get('type') == 'concede':
                continue
            cand[0, index] = torch.tensor(self.encoder.encode_action(observation, action), dtype=torch.float32)
            mask[0, index] = True
        if not bool(mask.any()):
            return 0
        with torch.no_grad():
            logits, _values = self.net(state, cand, mask)
        return int(logits[0].argmax().item())


def eval_gpu_policy_on_python(path: str | Path, *, n_games: int = 4, device: str = 'cpu',
                              deck: dict[str, Any] | None = None) -> dict[str, Any]:
    net = load_gpu_policy(path, device=device)
    learner = GpuWeightPolicy(net, device=device)
    opponent = VisibleEngineRulePolicy()
    deck = deck or default_training_deck()
    budget = TrainBudget(max_raw_steps=50_000, max_games=n_games + 2, max_seconds=600)
    games = []
    wins = losses = draws = truncated = errors = 0
    for index in range(n_games):
        seat = 'a' if index % 2 == 0 else 'b'
        match = _play_match(budget, learner, opponent, seed=20_260_914 + index, learning_side=seat, deck=deck)
        games.append({'seed': match.get('seed'), 'seat': seat, 'winner': match.get('winner'),
                      'terminated': match.get('terminated'), 'truncated': match.get('truncated'),
                      'error': match.get('error')})
        if match.get('error'):
            errors += 1
        elif match.get('truncated'):
            truncated += 1
        elif match.get('winner') == seat:
            wins += 1
        elif match.get('winner') in ('a', 'b'):
            losses += 1
        else:
            draws += 1
    return {
        'ok': errors == 0,
        'n_games': n_games,
        'wins': wins, 'losses': losses, 'draws': draws,
        'truncated': truncated, 'errors': errors,
        'games': games,
        'note': 'Python-engine eval of a GPU-trained encoder policy. Small sample, not a balance claim.',
    }
