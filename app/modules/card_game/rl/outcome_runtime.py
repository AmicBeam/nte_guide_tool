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
from . import grounded_candidates as grounded


class OutcomeModel(LeagueModel):
    def __init__(self, directory, key):
        root = Path(directory)
        m = json.loads((root / f'{key}.json').read_text(encoding='utf-8'))
        p = root / f'{key}.npz'
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        self.grounded = m['schema'] == grounded.SCHEMA
        if (m['schema'] not in (SCHEMA,grounded.SCHEMA) or m['rule_hash'] != identity() or m['sha256'] != digest
                or m['deck'] != key or m['features'] != all_features() or m['candidates'] != candidate_names()
                or m['wdl_order'] != ['loss', 'draw', 'win'] or m['value_context'] != ['viewer_is_actor']):
            raise ValueError('Outcome model identity mismatch')
        if self.grounded and m.get('candidate_transform_sha256') != grounded.fingerprint():
            raise ValueError('Grounded candidate transform mismatch')
        self.hidden = m['hidden']
        if type(self.hidden) != int or not 1 <= self.hidden <= 256:
            raise ValueError('Invalid width')
        expected = {**tensor_shapes(self.hidden), 'wdl.weight': (3, self.hidden + 1), 'wdl.bias': (3,)}
        if self.grounded:expected['cand_net.0.weight']=(self.hidden,grounded.DIM)
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
        if getattr(self,'grounded',False):candidates=grounded.transform(x,candidates)
        h=np.maximum(linear(x,'state_net.0'),0);h=np.maximum(linear(h,'state_net.2'),0)
        c=np.maximum(linear(candidates,'cand_net.0'),0);c=np.maximum(linear(c,'cand_net.2'),0)
        scores=np.sum(c*h,-1)/np.sqrt(np.float32(self.hidden))+linear(c,'score').reshape(-1)
        if not np.isfinite(scores).all():raise ValueError('Non-finite policy scores')
        return scores

    def scores_value(self,x,candidates):
        if not getattr(self,'grounded',False):return super().scores_value(x,candidates)
        p=self.wdl(x,True)
        return self.scores_only(x,candidates),float(p[2]-p[0])

    def scores(self,x,candidates):
        # Raw-policy evaluation does not need a value head or actor context.
        return self.scores_only(x,candidates)

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
    return OutcomeModel(directory, key) if manifest['schema'] in (SCHEMA,grounded.SCHEMA) else FixedModel(directory, key)


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


DEFAULT_RANKING_CONFIDENCE = 0.45
MAX_RANKING_COEFFICIENT = 1.0


def ranking_auxiliary_loss(policy_logits, mask, rows, selected, *, coefficient=0.,
                           confidence=DEFAULT_RANKING_CONFIDENCE, weights=None):
    """Pairwise ranking of a confident Gumbel top action versus every other legal action.

    Coefficient 0 is a true no-op: no extra graph, no extra optimizer term.
    The comparison is identity-based, so a target attack or end_turn is not
    pulled toward play, and two different cards remain competitors.
    """
    import torch
    if coefficient == 0:
        return policy_logits.new_zeros(()), dict(
            ranking_coefficient=0., ranking_confidence=float(confidence),
            ranking_samples=0, ranking_excluded=0)
    if not np.isfinite(coefficient) or not 0 < coefficient <= MAX_RANKING_COEFFICIENT:
        raise ValueError('Invalid ranking coefficient')
    if not np.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('Invalid ranking confidence')
    if not selected:
        return policy_logits.new_zeros(()), dict(
            ranking_coefficient=float(coefficient), ranking_confidence=float(confidence),
            ranking_samples=0, ranking_excluded=0)
    device = policy_logits.device
    included = []
    excluded = 0
    for local, index in enumerate(selected):
        row = rows[index]
        if row.get('truncated') or row.get('complete') is False:
            excluded += 1
            continue
        target = np.asarray(row['pi'], dtype=np.float64)
        top = int(np.argmax(target))
        legal_count = int(mask[local, :len(target)].sum().item()) if len(target) else 0
        if not bool(mask[local, top]) or float(target[top]) < confidence or legal_count < 2:
            excluded += 1
            continue
        included.append((local, top))
    if not included:
        return policy_logits.new_zeros(()), dict(
            ranking_coefficient=float(coefficient), ranking_confidence=float(confidence),
            ranking_samples=0, ranking_excluded=excluded)
    rows_i = torch.tensor([item[0] for item in included], device=device)
    tops = torch.tensor([item[1] for item in included], device=device)
    chosen = policy_logits[rows_i, tops]
    rival = policy_logits.masked_fill(~mask, torch.finfo(policy_logits.dtype).min)
    chosen_mask = mask.clone()
    chosen_mask[rows_i, tops] = False
    strongest = rival.masked_fill(~chosen_mask, torch.finfo(policy_logits.dtype).min)[rows_i].max(-1).values
    sample_weights = torch.ones(len(included), dtype=policy_logits.dtype, device=device)
    if weights is not None:
        sample_weights = torch.stack([weights[item[0]] for item in included])
    loss = (sample_weights * torch.nn.functional.softplus(strongest - chosen)).sum() / sample_weights.sum()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite ranking loss')
    return coefficient * loss, dict(
        ranking_coefficient=float(coefficient), ranking_confidence=float(confidence),
        ranking_samples=len(included), ranking_excluded=excluded,
        ranking_loss=float(loss.detach()))


def create_network(source, key, device):
    import torch
    from .gpu_duel.ppo import CompactScorer
    policy = model(source, key)
    net = grounded.network(policy.hidden) if getattr(policy,'grounded',False) else CompactScorer(len(all_features()), CAND_DIM, policy.hidden)
    if not getattr(policy,'grounded',False):net.wdl = torch.nn.Linear(policy.hidden + 1, 3)
    with torch.no_grad():
        net.wdl.weight.zero_()
        net.wdl.bias.zero_()
    net.load_state_dict({**net.state_dict(), **{k: torch.from_numpy(v.copy()) for k, v in policy.weights.items()}})
    # Old shaped head is preserved solely for compatibility, never optimized.
    for p in net.value.parameters():
        p.requires_grad_(False)
    return net.to(device)


def create_grounded_network(source,key,device,*,categorical=True):
    """Retain the learned state/value representation; reset ordinal action head."""
    import torch
    policy=model(source,key)
    if not isinstance(policy,OutcomeModel):raise ValueError('WDL source required')
    if categorical:net=grounded.network(policy.hidden)
    else:
        from .gpu_duel.ppo import CompactScorer
        net=CompactScorer(len(all_features()),CAND_DIM,policy.hidden)
        net.wdl=torch.nn.Linear(policy.hidden+1,3)
    weights=net.state_dict()
    for name,value in policy.weights.items():
        if name.startswith(('state_net.','wdl.','value.')):weights[name]=torch.from_numpy(value.copy())
    net.load_state_dict(weights)
    # Small, finite initial action variation without copying identity extrapolation.
    with torch.no_grad():
        net.cand_net[2].weight.mul_(.01);net.cand_net[2].bias.mul_(.01)
        net.score.weight.zero_();net.score.bias.zero_()
    for p in net.value.parameters():p.requires_grad_(False)
    return net.to(device)


def export(net, directory, key, build, origin):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f'{key}.npz'
    temp = root / f'{key}.tmp.npz'
    np.savez_compressed(temp, **{k: v.detach().cpu().numpy() for k, v in net.state_dict().items()})
    temp.replace(p)
    write(root / f'{key}.json', dict(schema=grounded.SCHEMA if getattr(net,'grounded',False) else SCHEMA,
        candidate_transform_sha256=grounded.fingerprint() if getattr(net,'grounded',False) else None,rule_hash=identity(), deck=key,
        features=all_features(), candidates=candidate_names(), hidden=net.score.in_features,
        wdl_order=['loss', 'draw', 'win'], value_context=['viewer_is_actor'],
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), build=build, build_sha256=build_hash(build),
        origin=origin, reward_mode='terminal_wdl_only', validation={'approved': False}, automatic_serving_approval=False))


def update(net, opt, rows, *, warmup=False, value_only=False, self_imitation=0.,
           ranking_coefficient=0., ranking_confidence=DEFAULT_RANKING_CONFIDENCE):
    import torch
    from .league_learning import tensors
    if not rows:
        raise ValueError('Empty WDL batch')
    if not np.isfinite(self_imitation) or not 0<=self_imitation<=1:
        raise ValueError('Invalid self-imitation coefficient')
    if any(r['z'] not in (-1, 0, 1) or r.get('reward_to_go') is not None or r.get('setup_reward', 0) for r in rows):
        raise ValueError('Only pure terminal labels permitted')
    device = next(net.parameters()).device
    x = torch.as_tensor(np.stack([r['x'] for r in rows]), device=device)
    # Recovery schemas have an independent terminal-value tower. Legacy
    # schemas retain the original shared trunk for historical reproduction.
    h = getattr(net, 'value_net', net.state_net)(x)
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
    selected = [i for i, r in enumerate(rows) if 'pi' in r] if not (warmup or value_only) else []
    policy_loss = logits.sum() * 0
    imitation_loss = logits.sum() * 0;imitation_value_loss=logits.sum()*0;imitation_samples=0
    ranking_loss = logits.new_zeros(())
    ranking_stats = dict(ranking_coefficient=0., ranking_confidence=float(ranking_confidence),
                         ranking_samples=0, ranking_excluded=0)
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
        if ranking_coefficient:
            ranking_loss, ranking_stats = ranking_auxiliary_loss(
                policy, mask, rows, selected, coefficient=ranking_coefficient,
                confidence=ranking_confidence, weights=weights[selected])
        if self_imitation:
            chosen=[]
            for i in selected:
                a=rows[i].get('selected')
                if not isinstance(a,(int,np.integer)) or not 0<=a<len(rows[i]['c']):
                    raise ValueError('Self-imitation requires the actual recorded action index')
                if rows[i].get('value_actor',1)!=1:raise ValueError('Policy row must belong to actor')
                chosen.append(int(a))
            probability=logits[selected].softmax(-1)
            value=probability[:,2]-probability[:,0]
            returns=torch.tensor([rows[i]['z'] for i in selected],dtype=value.dtype,device=device)
            advantage=(returns-value).clamp_min(0)
            chosen=torch.tensor(chosen,device=device)
            logp=policy.log_softmax(-1)[torch.arange(len(selected),device=device),chosen]
            imitation_loss=-(weights[selected]*advantage.detach()*logp).mean()
            imitation_value_loss=(weights[selected]*advantage.square()*.5).mean()
            imitation_samples=int((advantage>0).sum().detach())
    # Oh et al. (ICML 2018), equations 2–3, added to the existing Gumbel/WDL
    # objective. Returns are real terminal outcomes, never mechanism bonuses.
    loss = policy_loss + value_loss + ranking_loss + self_imitation*(imitation_loss+.01*imitation_value_loss)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite WDL loss')
    policy_gradient_norms = {}
    prefixes = ('cand_net', 'score', 'state_net')
    owned = [(name, parameter) for name, parameter in net.named_parameters()
             if parameter.requires_grad and name.startswith(prefixes)]
    joint = all(any(name.startswith(prefix) for name, _ in owned) for prefix in prefixes)
    if selected and joint:
        # Measure the policy objective alone. A joint backward would let WDL
        # gradients in state_net mask a disconnected policy path.
        grads = torch.autograd.grad(policy_loss, [parameter for _, parameter in owned],
                                    retain_graph=True, allow_unused=True)
        totals = {prefix: 0.0 for prefix in prefixes}
        for (name, _), gradient in zip(owned, grads):
            value = 0.0 if gradient is None else float(gradient.detach().norm())
            policy_gradient_norms[name] = value
            for prefix in prefixes:
                if name.startswith(prefix):
                    totals[prefix] += value * value
                    break
        policy_gradient_norms.update({prefix: float(np.sqrt(total)) for prefix, total in totals.items()})
        if any(not np.isfinite(totals[prefix]) for prefix in prefixes):
            raise ValueError('Non-finite policy gradient')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                self_imitation_loss=float(imitation_loss.detach()),self_imitation_value_loss=float(imitation_value_loss.detach()),
                self_imitation_coefficient=self_imitation,self_imitation_samples=imitation_samples,
                samples=len(rows), policy_samples=len(selected), value_only=bool(value_only),
                policy_gradient_norms=policy_gradient_norms,
                unique_row_ids=[r.get('row_id') for r in rows if r.get('row_id') and 'pi' in r],
                ranking_loss=float(ranking_loss.detach()),
                ranking_unscaled_loss=ranking_stats.get('ranking_loss', 0.),
                ranking_coefficient=ranking_stats.get('ranking_coefficient', 0.),
                ranking_confidence=ranking_stats.get('ranking_confidence', float(ranking_confidence)),
                ranking_samples=ranking_stats.get('ranking_samples', 0),
                ranking_excluded=ranking_stats.get('ranking_excluded', 0),
                priority=value_losses.detach().cpu().numpy().tolist())
