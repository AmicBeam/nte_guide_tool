"""Synchronous PPO on compact GPU observations. Not encoder-compatible yet."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..episode_return import cap_episode_reward
from .catalog import MAX_LEGAL
from .env import GpuDuelEnv
from .observe import compact_observe
from .rules import rebuild_legal


class CompactScorer(nn.Module):
    def __init__(self, state_dim: int, cand_dim: int, hidden: int = 256):
        super().__init__()
        self.state_net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.cand_net = nn.Sequential(nn.Linear(cand_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state, cand, mask):
        sh = self.state_net(state)
        ch = self.cand_net(cand)
        scale = math.sqrt(ch.shape[-1])
        logits = (ch * sh.unsqueeze(1)).sum(-1) / scale + self.score(ch).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        values = self.value(sh).squeeze(-1)
        return logits, values


def _sample(logits):
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action, dist.log_prob(action), dist.entropy()


def _terminal_reward(env: GpuDuelEnv, prev_hp_foe=None, prev_down_foe=None) -> torch.Tensor:
    """Outcome only; damage, knockdowns and all unfinished steps award zero."""
    done, winner = env.outcome()
    learner = env.learner
    term = torch.zeros(env.n, device=env.device, dtype=torch.float32)
    term = torch.where(done & (winner == learner), torch.full_like(term, 10.0), term)
    term = torch.where(done & (winner == 1 - learner), torch.full_like(term, -10.0), term)
    return cap_episode_reward(env, term)


def ppo_update(net: CompactScorer, opt, batch, clip=0.2, epochs=2, minibatch=1024, vf_coef=0.5, ent_coef=0.01, stop_check=None):
    state, cand, mask, action, old_logp, old_value, ret, adv = batch
    n = state.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    optimizer_steps = 0
    diagnostics = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=state.device)
        for start in range(0, n, minibatch):
            if stop_check:
                stop_check()
            idx = perm[start:start + minibatch]
            logits, values = net(state[idx], cand[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(action[idx])
            ratio = (logp - old_logp[idx]).exp()
            surr = torch.minimum(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx])
            loss = -surr.mean() + vf_coef * (values - ret[idx]).pow(2).mean() - ent_coef * dist.entropy().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite PPO loss')
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            optimizer_steps += 1
            with torch.no_grad():
                logratio = logp - old_logp[idx]
                diagnostics.append(torch.stack((loss.detach(), dist.entropy().mean(),
                    ((logratio.exp() - 1) - logratio).mean(),
                    ((ratio - 1).abs() > clip).float().mean(),
                    (values - ret[idx]).pow(2).mean())))
    stats = torch.stack(diagnostics).mean(0).tolist()
    variance = ret.var(unbiased=False)
    explained = 1 - (ret - old_value).var(unbiased=False) / variance.clamp(min=1e-8)
    return dict(optimizer_steps=optimizer_steps, last_loss=float(loss.detach()),
                loss_mean=stats[0], entropy=stats[1], approx_kl=stats[2],
                clip_fraction=stats[3], value_loss=stats[4], explained_variance=float(explained))


@torch.no_grad()
def collect_rollout(env: GpuDuelEnv, net: CompactScorer, horizon: int, learner_side: int = 0,
                    observe_fn=None, stop_check=None):
    observe = observe_fn or compact_observe
    env.reset_finished()
    env.prepare_decision()
    states, cands, masks, actions, logps, values, rewards, dones = [], [], [], [], [], [], [], []
    winners = []
    net_device = next(net.parameters()).device
    for _ in range(horizon):
        if stop_check:
            stop_check()
        if bool((env.outcome()[0] | (env.state.active != env.learner)).any()):
            raise RuntimeError('Rollout requires a live learner decision in every row')
        state, cand, mask = observe(env.state)
        net_device = next(net.parameters()).device
        state = state.to(net_device)
        cand = cand.to(net_device)
        mask = mask.to(net_device)
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            raise RuntimeError('Live learner decision has no legal actions')
        logits, value = net(state, cand, mask)
        action, logp, _ent = _sample(logits)
        env_action = action.to(env.device)
        env.step_learner(env_action)
        reward = _terminal_reward(env).to(net_device)
        done, winner = env.outcome()
        states.append(state)
        cands.append(cand)
        masks.append(mask)
        actions.append(action)
        logps.append(logp.detach())
        values.append(value.detach())
        rewards.append(reward.detach())
        dones.append(done.to(device=net_device, dtype=reward.dtype).clone())
        winners.append(winner.to(net_device).clone())
        # Outcome and reward belong to the old episode. Only now start a new one.
        env.reset_finished()
    state_bt = torch.stack(states, dim=1)
    cand_bt = torch.stack(cands, dim=1)
    mask_bt = torch.stack(masks, dim=1)
    b, t = state_bt.shape[0], state_bt.shape[1]
    return {
        'state': state_bt.reshape(b * t, -1),
        'cand': cand_bt.reshape(b * t, cand_bt.shape[2], cand_bt.shape[3]),
        'mask': mask_bt.reshape(b * t, mask_bt.shape[2]),
        'action': torch.stack(actions, dim=1).reshape(-1),
        'logp': torch.stack(logps, dim=1).reshape(-1),
        'value': torch.stack(values, dim=1),
        'reward': torch.stack(rewards, dim=1),
        'done': torch.stack(dones, dim=1),
        'winner': torch.stack(winners, dim=1),
        'last_value': net(*[item.to(net_device) if torch.is_tensor(item) else item for item in observe(env.state)])[1].detach(),
    }


def gae(rollout, gamma=1.0, lam=0.95):
    reward = rollout['reward']
    done = rollout['done']
    value = rollout['value']
    last = rollout['last_value']
    b, t = reward.shape
    adv = torch.zeros_like(reward)
    lastgaelam = torch.zeros(b, device=reward.device)
    for step in reversed(range(t)):
        nxt = last if step == t - 1 else value[:, step + 1]
        nonterm = 1.0 - done[:, step]
        delta = reward[:, step] + gamma * nxt * nonterm - value[:, step]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[:, step] = lastgaelam
    ret = adv + value
    return ret.reshape(-1), adv.reshape(-1)


def rollout_to_batch(rollout):
    ret, adv = gae(rollout)
    return (
        rollout['state'], rollout['cand'], rollout['mask'], rollout['action'],
        rollout['logp'], rollout['value'].reshape(-1), ret, adv,
    )
