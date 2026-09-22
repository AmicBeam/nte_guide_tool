"""Offline WDL extension. Pure terminal value, fixed observer, numeric inference."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
from .fixed_lineup import FixedModel, all_features, candidate_names, tensor_shapes, identity, CAND_DIM, encode, write
from .league_policy import LeagueModel
from .league_schema import build_hash
from .ten_search_runtime import decision, determinize, passive_count

SCHEMA = 'fixed_ten_wdl_v1'


class OutcomeModel(LeagueModel):
    def __init__(self, directory, key):
        root = Path(directory)
        m = json.loads((root / f'{key}.json').read_text(encoding='utf-8'))
        p = root / f'{key}.npz'
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if (m['schema'] != SCHEMA or m['rule_hash'] != identity() or m['sha256'] != digest
                or m['deck'] != key or m['features'] != all_features() or m['candidates'] != candidate_names()
                or m['wdl_order'] != ['loss', 'draw', 'win'] or m['value_context'] != ['viewer_is_actor']):
            raise ValueError('Outcome model identity mismatch')
        self.hidden = m['hidden']
        if type(self.hidden) != int or not 1 <= self.hidden <= 256:
            raise ValueError('Invalid width')
        expected = {**tensor_shapes(self.hidden), 'wdl.weight': (3, self.hidden + 1), 'wdl.bias': (3,)}
        with np.load(p, allow_pickle=False) as z:
            if set(z.files) != set(expected):
                raise ValueError('Outcome tensor set mismatch')
            self.weights = {n: z[n].copy() for n in expected}
        for n, shape in expected.items():
            if self.weights[n].shape != shape or not np.isfinite(self.weights[n]).all():
                raise ValueError('Invalid outcome tensor')
            self.weights[n].flags.writeable = False
        self.serving_deck = m['build']
        if build_hash(self.serving_deck) != m['build_sha256']:
            raise ValueError('Outcome build mismatch')
        self.manifest = m
        self.version = digest

    def scores_only(self, x, candidates):
        # Preserve the existing policy arithmetic exactly; WDL historically uses
        # matmul, so do not silently replace its reduction order with einsum.
        def linear(v,n):
            return np.einsum('...j,ij->...i',v,self.weights[n+'.weight'],optimize=False)+self.weights[n+'.bias']
        h=np.maximum(linear(x,'state_net.0'),0);h=np.maximum(linear(h,'state_net.2'),0)
        c=np.maximum(linear(candidates,'cand_net.0'),0);c=np.maximum(linear(c,'cand_net.2'),0)
        scores=np.sum(c*h,-1)/np.sqrt(np.float32(self.hidden))+linear(c,'score').reshape(-1)
        if not np.isfinite(scores).all():raise ValueError('Non-finite policy scores')
        return scores

    def wdl(self, x, is_actor):
        w = self.weights
        h = np.maximum(w['state_net.0.weight'] @ x + w['state_net.0.bias'], 0)
        h = np.maximum(w['state_net.2.weight'] @ h + w['state_net.2.bias'], 0)
        logits = w['wdl.weight'] @ np.append(h, np.float32(is_actor)) + w['wdl.bias']
        p = np.exp(logits - logits.max())
        return p / p.sum()


@lru_cache(maxsize=24)
def _load(directory, key, modified):
    manifest = json.loads((Path(directory) / f'{key}.json').read_text(encoding='utf-8'))
    return OutcomeModel(directory, key) if manifest['schema'] == SCHEMA else FixedModel(directory, key)


def model(directory, key):
    return _load(str(directory), key, (Path(directory) / f'{key}.json').stat().st_mtime_ns)


def value_for(state, viewer, policies):
    """None means legacy fallback. Never consult another player's private value."""
    policy = policies[viewer]
    if not isinstance(policy, OutcomeModel):
        return None
    from ..engine.duel_v2 import observe, acting_side
    x, _ = encode(observe(state, viewer, include_previews=False), [])
    p = policy.wdl(x, acting_side(state) == viewer)
    return float(p[2] - p[0])


def predict_encoded(policy,x,c):
    """Reuse the caller's viewer encoding; avoid the unused legacy value head."""
    if isinstance(policy,OutcomeModel):
        scores=policy.scores_only(x,c);p=policy.wdl(x,True)
        return scores,float(p[2]-p[0])
    scores,raw=policy.scores_value(x,c)
    return scores,float(np.tanh(raw/10))


def action_scores(policy,x,c):
    return policy.scores_only(x,c) if isinstance(policy,OutcomeModel) else policy.scores_value(x,c)[0]


def create_network(source, key, device):
    import torch
    from .gpu_duel.ppo import CompactScorer
    policy = model(source, key)
    net = CompactScorer(len(all_features()), CAND_DIM, policy.hidden)
    net.wdl = torch.nn.Linear(policy.hidden + 1, 3)
    with torch.no_grad():
        net.wdl.weight.zero_()
        net.wdl.bias.zero_()
    net.load_state_dict({**net.state_dict(), **{k: torch.from_numpy(v.copy()) for k, v in policy.weights.items()}})
    # Old shaped head is preserved solely for compatibility, never optimized.
    for p in net.value.parameters():
        p.requires_grad_(False)
    return net.to(device)


def export(net, directory, key, build, origin):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f'{key}.npz'
    temp = root / f'{key}.tmp.npz'
    np.savez_compressed(temp, **{k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
    temp.replace(p)
    write(root / f'{key}.json', dict(schema=SCHEMA, rule_hash=identity(), deck=key,
        features=all_features(), candidates=candidate_names(), hidden=net.score.in_features,
        wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'],
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), build=build, build_sha256=build_hash(build),
        origin=origin, reward_mode='terminal_wdl_only', validation={'approved': False}, automatic_serving_approval=False))


def update(net, opt, rows, *, warmup=False):
    import torch
    from .league_learning import tensors
    if not rows:
        raise ValueError('Empty WDL batch')
    if any(r['z'] not in (-1, 0, 1) or r.get('reward_to_go') is not None or r.get('setup_reward', 0) for r in rows):
        raise ValueError('Only pure terminal labels permitted')
    device = next(net.parameters()).device
    x = torch.as_tensor(np.stack([r['x'] for r in rows]), device=device)
    h = net.state_net(x)
    if warmup:
        h = h.detach()
    context = torch.tensor([[r.get('value_actor', 1)] for r in rows], dtype=h.dtype, device=device)
    logits = net.wdl(torch.cat((h, context), -1))
    z = torch.tensor([int(r['z']) + 1 for r in rows], device=device)
    weights = torch.tensor([r.get('sample_weight', 1.) for r in rows], dtype=h.dtype, device=device)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid sampling weights')
    value_losses = torch.nn.functional.cross_entropy(logits, z, reduction='none')
    value_loss = (weights * value_losses).mean()
    selected = [i for i, r in enumerate(rows) if 'pi' in r] if not warmup else []
    policy_loss = logits.sum() * 0
    if selected:
        xx, c, mask = tensors([(rows[i]['x'], rows[i]['c']) for i in selected], device)
        policy, _ = net(xx, c, mask)
        target = torch.zeros_like(policy)
        for j, i in enumerate(selected):
            p = np.asarray(rows[i]['pi'])
            if len(p) != len(rows[i]['c']) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
                raise ValueError('Invalid policy label')
            target[j, :len(p)] = torch.as_tensor(p, device=device)
        losses = -(target * policy.log_softmax(-1).masked_fill(~mask, 0)).sum(-1)
        policy_loss = (weights[selected] * losses).mean()
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise ValueError('Non-finite WDL loss')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                samples=len(rows), policy_samples=len(selected), priority=value_losses.detach().cpu().numpy().tolist())
