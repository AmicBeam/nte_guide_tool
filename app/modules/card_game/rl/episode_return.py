"""Per-game upper bound on emitted rewards; bookkeeping stays on env.device."""
REWARD_MODE = "outcome_only_v1"
EPISODE_RETURN_CAP = 10.0


def validate_reward_resume(checkpoint):
    """Old shaped checkpoints can warm-start, but cannot resume this objective."""
    if checkpoint.get("reward_mode") != REWARD_MODE:
        raise ValueError("Resume reward objective mismatch; use --warm-start for a fresh optimizer")


def reset_episode_return(env, mask=None):
    """Clear only games being reset, never clear an unfinished rollout row."""
    import torch
    total = getattr(env, 'episode_return', None)
    if total is None or mask is None:
        env.episode_return = torch.zeros(env.n, dtype=torch.float32, device=env.device)
    else:
        total.masked_fill_(mask, 0.0)


def cap_episode_reward(env, reward):
    """Keep cumulative return <= 10 without clipping any negative reward.

    Applied to the original reward, including terminal rewards. A win after
    +3 emits +7; a loss still emits -10, leaving that game's total at -7.
    The caller records the terminal reward before resetting the game's total.
    """
    import torch
    if getattr(env, 'episode_return', None) is None:
        reset_episode_return(env)
    total = env.episode_return
    emitted = torch.minimum(reward, EPISODE_RETURN_CAP - total)
    total.add_(emitted).clamp_(max=EPISODE_RETURN_CAP)
    return emitted
