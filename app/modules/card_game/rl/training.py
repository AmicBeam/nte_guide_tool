"""Single-env MaskablePPO training against a frozen public rule opponent."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .adapter import DuelAdapter
from .backend import V2Backend
from .budget import BudgetStop, CountingBackend, TrainBudget
from .checkpoint import write_training_checkpoint
from .encoding import V2Encoder
import gymnasium as gym

from .gym_compat import make_gym_env
from .capability import training_matchup
from .policies import RuleFrozenPolicy, VisibleEngineRulePolicy
from .transfer import default_training_deck

from .training_config import MAX_N_ENVS, DEFAULT_N_ENVS_CAP, default_n_envs


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')
        handle.flush()


def _device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def make_rule_adapter(budget: TrainBudget, *, learning_side: str = 'a',
                      max_game_actions: int = 400, max_opponent_actions: int = 80,
                      fast: bool = False, opponent=None) -> DuelAdapter:
    backend = CountingBackend(V2Backend(copy_on_apply=not fast), budget)
    if opponent is None:
        opponent = RuleFrozenPolicy() if fast else VisibleEngineRulePolicy()
    adapter = DuelAdapter(
        backend, V2Encoder(), learning_side=learning_side, opponent=opponent,
        max_game_actions=max_game_actions, max_opponent_actions=max_opponent_actions,
        strict_isolation=not fast, allow_concede=False,
    )
    return adapter


def build_training_env_spawn(rank, raw, games, lock, start, config: dict[str, Any]):
    config = dict(config)
    config['rank'] = rank
    config['shared'] = {'raw': raw, 'games': games, 'lock': lock, 'start': start}
    return build_training_env(config)


class SeededTrainingWrapper(gym.Wrapper):
    """Module-level gym wrapper so ProcessVecEnv can pickle action_masks."""

    def __init__(self, env, *, start_seed: int, config: dict[str, Any], budget: TrainBudget):
        super().__init__(env)
        self.next_seed = start_seed
        self.episodes_finished = 0
        self._dead = False
        self._last_obs = None
        self._config = config
        self._budget = budget

    def action_masks(self):
        if self._dead:
            import numpy as np
            return np.zeros(self.action_space.n, dtype=bool)
        return self.env.action_masks()

    def _dead_payload(self, reason: str):
        import numpy as np
        if self._last_obs is None:
            sample = self.observation_space.sample()
            self._last_obs = {key: np.zeros_like(value) for key, value in sample.items()}
        return self._last_obs, {'budget_stop': True, 'truncation_reason': reason, 'bootstrap_allowed': False}

    def reset(self, *, seed=None, options=None):
        if self._dead:
            return self._dead_payload(self._budget.remaining_reason() or 'exhausted')
        last_error = None
        config = self._config
        for _ in range(16):
            seed_used = self.next_seed if seed is None else seed
            self.next_seed = int(seed_used) + int(config.get('n_envs', 1))
            reset_options = dict(options or {})
            if 'decks' not in reset_options:
                learner = config.get('deck') or default_training_deck()
                reset_options['decks'] = training_matchup(
                    int(seed_used),
                    learning_side=config.get('learning_side', 'a'),
                    learner_deck=learner,
                    opponent_deck=config.get('opponent_deck'),
                    sample_public=bool(config.get('sample_public')),
                )
            try:
                observation, info = self.env.reset(seed=seed_used, options=reset_options)
            except (RuntimeError, BudgetStop) as exc:
                last_error = exc
                if isinstance(exc, BudgetStop) or 'budget' in str(exc).lower():
                    self._dead = True
                    return self._dead_payload(getattr(exc, 'reason', None) or 'exhausted')
                continue
            info = dict(info)
            info['match_seed'] = seed_used
            self._last_obs = observation
            return observation, info
        raise RuntimeError(f'Reset could not reach a live learner decision: {last_error}')

    def step(self, action):
        if self._dead:
            observation, info = self._dead_payload(self._budget.remaining_reason() or 'exhausted')
            return observation, 0.0, False, True, info
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
        except BudgetStop as exc:
            self._dead = True
            observation, info = self._dead_payload(exc.reason)
            return observation, 0.0, False, True, info
        self._last_obs = observation
        if terminated or truncated:
            self.episodes_finished += 1
            info = dict(info)
            info['episodes_finished'] = self.episodes_finished
        return observation, reward, terminated, truncated, info


def _action_masks(env):
    return env.action_masks()


def build_training_env(config: dict[str, Any], budget: TrainBudget | None = None):
    """Picklable env factory for DummyVecEnv / Windows spawn SubprocVecEnv."""
    import sys
    root = str(Path(__file__).resolve().parents[4])
    if root not in sys.path:
        sys.path.insert(0, root)
    from sb3_contrib.common.wrappers import ActionMasker

    if budget is None:
        budget = TrainBudget(
            max_raw_steps=config['max_raw_steps'], max_games=config['max_games'],
            max_seconds=config['max_seconds'], reserve_raw_steps=config['reserve_raw_steps'],
            reserve_games=config['reserve_games'], reserve_seconds=config['reserve_seconds'],
        )
        shared = config.get('shared')
        if shared is not None:
            budget.attach_shared(shared)
    opponent = None
    if config.get('opponent_policy') == 'visible':
        opponent = VisibleEngineRulePolicy()
    adapter = make_rule_adapter(
        budget, learning_side=config.get('learning_side', 'a'),
        max_game_actions=config.get('max_game_actions', 400),
        max_opponent_actions=config.get('max_opponent_actions', 80),
        fast=config.get('opponent_policy') != 'visible',
        opponent=opponent,
    )
    inner = make_gym_env(adapter)
    start_seed = int(config.get('start_seed', 0)) + int(config.get('rank', 0)) * 100_003
    wrapped = SeededTrainingWrapper(inner, start_seed=start_seed, config=config, budget=budget)
    return ActionMasker(wrapped, _action_masks)


def train_maskable_ppo(output: Path, budget: TrainBudget, *, start_seed: int = 0,
                       n_steps: int = 128, batch_size: int = 256, n_epochs: int = 2,
                       learning_rate: float = 3e-4, resume_from: Path | None = None,
                       n_envs: int | None = None, hidden_dim: int = 256,
                       deck: dict[str, Any] | None = None,
                       freeze_generic: bool = False, generic_lr_mult: float = 1.0,
                       kl_coef: float = 0.0,
                       opponent_deck: dict[str, Any] | None = None,
                       sample_public: bool = False,
                       log: Callable[[str], None] | None = None) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .budget import make_shared_counters
    from .checkpoint import load_training_checkpoint
    from .networks import CandidateScoringPolicy
    from .transfer import apply_transfer, unique_card_ids

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = log or print
    metrics_path = output / 'metrics.jsonl'
    device = _device()

    class _Callback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.n_rollouts = 0
            self.last_update = -1
            self.last_checkpoint = 0
            self._episodes_path = output / 'episodes.jsonl'

        def _on_training_start(self):
            log('phase=sampling training_start')
            append_jsonl(metrics_path, {'event': 'training_start', **budget.snapshot(), 'device': device})
            self._episodes_path = output / 'episodes.jsonl'
            atomic_write(output / 'status.json', {
                'ok': True, 'phase': 'sampling', **budget.snapshot(),
                'num_timesteps': 0, 'n_updates': 0,
            })

        def _on_rollout_start(self):
            self.n_rollouts += 1
            gpu = ''
            if torch.cuda.is_available():
                gpu = f' gpu_mem_mb={torch.cuda.memory_allocated() / 1024 / 1024:.1f}'
            log(f'phase=sampling rollout={self.n_rollouts} env_steps={int(self.model.num_timesteps)}'
                f' raw_steps={budget.raw_steps} games={budget.games}{gpu}')
            return True

        def _on_step(self):
            for info in self.locals.get('infos') or []:
                score = info.get('episode_score') if isinstance(info, dict) else None
                if score:
                    append_jsonl(self._episodes_path, {'event': 'episode', **score})
            updates = int(getattr(self.model, '_n_updates', 0) or 0)
            if self.n_calls == 1 or self.n_calls % 8 == 0:
                atomic_write(output / 'status.json', {
                    'ok': True, 'phase': 'sampling', **budget.snapshot(),
                    'num_timesteps': int(self.model.num_timesteps), 'n_updates': updates,
                })
            reason = budget.remaining_reason() or ('max_games' if budget.games_exhausted() else None)
            if reason:
                log(f'phase=stop reason={reason} env_steps={int(self.model.num_timesteps)}'
                    f' n_updates={updates}')
                return False
            return True

        def _on_rollout_end(self):
            updates = int(getattr(self.model, '_n_updates', 0) or 0)
            row = {
                'event': 'rollout_end', **budget.snapshot(),
                'num_timesteps': int(self.model.num_timesteps),
                'n_updates': updates, 'n_rollouts': self.n_rollouts,
            }
            logger = getattr(self.model, 'logger', None)
            name_to_value = getattr(logger, 'name_to_value', None) or {}
            for key in ('train/loss', 'train/policy_gradient_loss', 'train/value_loss',
                        'train/entropy_loss', 'rollout/ep_rew_mean', 'train/approx_kl'):
                if key in name_to_value:
                    row[key] = float(name_to_value[key])
            elapsed = max(budget.elapsed(), 1e-6)
            row['env_steps_per_sec'] = round(int(self.model.num_timesteps) / elapsed, 4)
            row['raw_steps_per_sec'] = round(budget.raw_steps / elapsed, 4)
            append_jsonl(metrics_path, row)
            if updates > self.last_update:
                self.last_update = updates
                log(
                    f'phase=update n_updates={updates} env_steps={int(self.model.num_timesteps)}'
                    f' raw_steps={budget.raw_steps} games={budget.games}'
                    f' elapsed_s={round(elapsed, 2)} loss={row.get("train/loss", "")}'
                )
                atomic_write(output / 'status.json', {
                    'ok': True, 'phase': 'update', **budget.snapshot(),
                    'num_timesteps': int(self.model.num_timesteps), 'n_updates': updates,
                    'loss': row.get('train/loss'),
                })
                if updates - self.last_checkpoint >= 10:
                    write_training_checkpoint(output, self.model, {
                        'num_timesteps': int(self.model.num_timesteps),
                        'n_updates': updates,
                    })
                    self.last_checkpoint = updates
                    log(f'phase=checkpoint n_updates={updates} path={output / "model.zip"}')
            return True

    n_envs = max(1, min(MAX_N_ENVS, int(n_envs if n_envs is not None else default_n_envs())))
    rollout = n_steps * n_envs
    if batch_size > rollout:
        batch_size = rollout
    training_deck = deck or default_training_deck()
    env_config = {
        'max_raw_steps': budget.max_raw_steps, 'max_games': budget.max_games,
        'max_seconds': budget.max_seconds, 'reserve_raw_steps': budget.reserve_raw_steps,
        'reserve_games': budget.reserve_games, 'reserve_seconds': budget.reserve_seconds,
        'start_seed': start_seed, 'n_envs': n_envs, 'max_game_actions': 400,
        'max_opponent_actions': 80, 'learning_side': 'a',
        'deck': training_deck,
        'opponent_deck': opponent_deck or training_deck,
        'sample_public': bool(sample_public),
        'opponent_policy': 'rule',
    }
    if n_envs > 1:
        shared = make_shared_counters()
        budget.attach_shared(shared)
        from functools import partial
        env_fns = [
            partial(
                build_training_env_spawn, rank, shared['raw'], shared['games'],
                shared['lock'], shared['start'], env_config,
            )
            for rank in range(n_envs)
        ]
        try:
            from .process_vec_env import ProcessVecEnv
            vec = ProcessVecEnv(env_fns)
            log(f'phase=sampling vec=process n_envs={n_envs}')
        except Exception as exc:  # noqa: BLE001
            log(f'phase=warn process_vec_failed {type(exc).__name__}: {exc}')
            if os.name == 'nt':
                from .budget import make_thread_counters
                from .thread_vec_env import ThreadedVecEnv
                shared = make_thread_counters()
                budget.attach_shared(shared)
                thread_fns = [
                    lambda rank=rank: build_training_env({**env_config, 'rank': rank}, budget=budget)
                    for rank in range(n_envs)
                ]
                vec = ThreadedVecEnv(thread_fns)
                log(f'phase=sampling vec=thread n_envs={n_envs}')
            else:
                try:
                    vec = SubprocVecEnv(env_fns, start_method='spawn')
                    log(f'phase=sampling vec=subproc n_envs={n_envs}')
                except Exception as subproc_exc:  # noqa: BLE001
                    log(f'phase=warn subproc_failed {type(subproc_exc).__name__}: {subproc_exc}; falling back to 1 env')
                    n_envs = 1
                    rollout = n_steps * n_envs
                    if batch_size > rollout:
                        batch_size = rollout
                    budget.attach_shared(None)
                    vec = DummyVecEnv([lambda: build_training_env({**env_config, 'rank': 0}, budget=budget)])
    else:
        vec = DummyVecEnv([lambda: build_training_env({**env_config, 'rank': 0}, budget=budget)])
    callback = _Callback()
    transfer_info: dict[str, Any] = {
        'freeze_generic': False, 'generic_lr_mult': generic_lr_mult, 'kl_coef': 0.0,
        'initialized': [], 'new_cards': [], 'overlap_cards': [],
    }
    if resume_from:
        model, resume_meta = load_training_checkpoint(resume_from, env=vec, device=device)
        log(f'phase=resume loaded={resume_from} timesteps={model.num_timesteps}')
        teacher = None
        if kl_coef > 0:
            teacher, _teacher_meta = load_training_checkpoint(resume_from, env=vec, device=device)
            teacher.policy.eval()
            teacher = teacher.policy
        previous_cards = list(resume_meta.get('training_deck_card_ids') or [])
        transfer_info = apply_transfer(
            model, freeze_generic=freeze_generic, generic_lr_mult=generic_lr_mult,
            learning_rate=learning_rate, kl_coef=kl_coef, teacher=teacher,
            previous_card_ids=previous_cards, next_card_ids=unique_card_ids(training_deck),
        )
        log(
            f'phase=transfer freeze_generic={transfer_info["freeze_generic"]}'
            f' kl_coef={transfer_info["kl_coef"]} new_cards={",".join(transfer_info["new_cards"]) or "-"}'
            f' initialized={",".join(transfer_info["initialized"]) or "-"}'
        )
    else:
        if freeze_generic or kl_coef > 0:
            raise ValueError('--freeze-generic and --kl-coef require --resume')
        model = MaskablePPO(
            CandidateScoringPolicy, vec, verbose=0, device=device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            learning_rate=learning_rate, gamma=1.0, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(hidden_dim=hidden_dim),
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        param_device = next(model.policy.parameters()).device
        log(f'phase=sampling device={device} n_envs={n_envs} n_steps={n_steps} '
            f'batch_size={batch_size} param_device={param_device} pid={os.getpid()}')
    else:
        log(f'phase=sampling device={device} n_envs={n_envs} n_steps={n_steps} pid={os.getpid()}')
    model.learn(total_timesteps=10_000_000, callback=callback, progress_bar=False,
                reset_num_timesteps=resume_from is None)
    elapsed = max(budget.elapsed(), 1e-6)
    n_updates = int(getattr(model, '_n_updates', 0) or 0)
    stats = {
        'num_timesteps': int(model.num_timesteps),
        'n_updates': n_updates,
        'raw_steps': budget.raw_steps,
        'games_started': budget.games,
        'elapsed_seconds': round(elapsed, 4),
        'env_steps_per_sec': round(model.num_timesteps / elapsed, 4),
        'raw_steps_per_sec': round(budget.raw_steps / elapsed, 4),
        'device': device,
        'stopped_reason': budget.remaining_reason() or 'learn_returned',
        'n_steps': n_steps, 'batch_size': batch_size, 'n_epochs': n_epochs, 'n_envs': n_envs,
        'policy': 'CandidateScoringPolicy',
        'training_deck_id': training_deck.get('id'),
        'training_deck_card_ids': unique_card_ids(training_deck),
        'opponent_deck': opponent_deck or training_deck,
        'sample_public_opponent': bool(sample_public),
        'transfer': transfer_info,
    }
    write_training_checkpoint(output, model, stats)
    atomic_write(output / 'status.json', {'ok': n_updates > 0, 'phase': 'trained', **stats, **budget.snapshot()})
    log(f'phase=trained n_updates={n_updates} env_steps={model.num_timesteps} raw_steps={budget.raw_steps}')
    vec.close()
    if n_updates <= 0:
        raise RuntimeError('Training finished without an optimizer update')
    return stats
