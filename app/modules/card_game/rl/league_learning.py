"""PPO with terminal or explicit reward-to-go targets; Torch imports are lazy."""
import numpy as np
from .league_preferences import validate_pairs

def tensors(rows,device):
    import torch
    n=len(rows);width=max(len(r[1]) for r in rows);d=rows[0][1].shape[1]
    state=np.stack([r[0] for r in rows]);cand=np.zeros((n,width,d),dtype=np.float32);mask=np.zeros((n,width),dtype=bool)
    for i,r in enumerate(rows):cand[i,:len(r[1])]=r[1];mask[i,:len(r[1])]=True
    return tuple(torch.as_tensor(x,device=device) for x in (state,cand,mask))

def ordinary_rows(games):
    """Use per-decision reward-to-go when supplied, preserving legacy terminal targets."""
    rows = []
    for game in games:
        if not game['complete']: continue
        trajectory = game['trajectory']
        returns = game.get('trajectory_returns')
        if returns is None: returns = [game['reward']] * len(trajectory)
        if len(returns) != len(trajectory) or not np.isfinite(returns).all():
            raise ValueError('Invalid trajectory returns')
        rows.extend((*entry, float(target)) for entry, target in zip(trajectory, returns))
    return rows


def ordinary_update(net,opt,games,seed):
    import torch
    rows = ordinary_rows(games)
    if not rows:return {'updates':0,'samples':0}
    rng=np.random.default_rng(seed)
    if len(rows)>512:rows=[rows[i] for i in rng.choice(len(rows),512,replace=False)]
    # One batch is exactly one optimizer step, regardless of trajectory length.
    dev=next(net.parameters()).device;x,c,m=tensors(rows,dev)
    action=torch.tensor([r[2] for r in rows],device=dev);oldlogp=torch.tensor([r[3] for r in rows],device=dev)
    target=torch.tensor([r[5] for r in rows],device=dev);oldvalue=torch.tensor([r[4] for r in rows],device=dev)
    logits,value=net(x,c,m);dist=torch.distributions.Categorical(logits=logits)
    ratio=(dist.log_prob(action)-oldlogp).clamp(-20,20).exp();adv=((target-oldvalue)/10).clamp(-2,2)
    loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*((value-target)/10).square().mean()-.01*dist.entropy().mean()
    if not torch.isfinite(loss):raise ValueError('Non-finite PPO update')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5,error_if_nonfinite=True);opt.step()
    return {'updates':1,'optimizer_steps':1,'samples':len(rows),'loss':float(loss.detach())}


def branch_update(net,opt,x,c,returns,*,anchor_logits=None,all_candidates=None,selected_indices=None,auxiliary=()):
    import torch
    q=np.asarray(returns,dtype=np.float32)
    auxiliary=validate_pairs(auxiliary,q)
    if len(q)<2 or (float(np.ptp(q))<=1e-6 and not auxiliary):return {'updates':0,'reason':'equal_terminal_returns'}
    dev=next(net.parameters()).device;xx,cc,mm=tensors([(x,c)],dev)
    target=torch.softmax(torch.as_tensor(q,device=dev)/4,0)
    logits,value=net(xx,cc,mm)
    # Rank only unequal terminal estimates; equal candidates are not forced uniform.
    best=np.flatnonzero(q>=q.max()-1e-6)
    pairs=[(int(i),j,float(q[i]-q[j])/20) for i in best for j in range(len(q)) if q[i]-q[j]>1e-6]
    weights=torch.tensor([w for _,_,w in pairs],device=dev)
    loss=logits.sum()*0
    if pairs:
        margins=torch.stack([logits[0,j]-logits[0,i] for i,j,_ in pairs])
        loss=(weights*torch.nn.functional.softplus(margins)).sum()/weights.sum()
        loss=loss+.25*((value[0]-(target*torch.as_tensor(q,device=dev)).sum())/10).square()
    # Keep auxiliary weights absolute: normalizing by their sum cancels weakness.
    for reason in ('certain_lethal','winning_speed','losing_delay'):
        group=[(i,j,w) for i,j,w,k in auxiliary if k==reason]
        if group:
            loss=loss+torch.stack([w*torch.nn.functional.softplus(logits[0,j]-logits[0,i]) for i,j,w in group]).mean()
    kl=torch.tensor(0.,device=dev)
    anchor_wrong=False
    if anchor_logits is not None:
        indices=list(range(len(q))) if selected_indices is None else list(selected_indices)
        if len(indices)!=len(q) or len(set(indices))!=len(indices):raise ValueError('Invalid evaluated candidate indices')
        preferred=int(np.argmax(anchor_logits))
        anchor_wrong=preferred in indices and (float(q.max()-q[indices.index(preferred)])>=2.5 or any(j==indices.index(preferred) for i,j,w,reason in auxiliary))
    if anchor_logits is not None and not anchor_wrong:
        ax,ac,am=tensors([(x,all_candidates)],dev)
        current,_=net(ax,ac,am)
        reference=torch.softmax(torch.as_tensor(anchor_logits,device=dev),0)
        kl=(reference*(reference.clamp(min=1e-20).log()-current[0].log_softmax(0))).sum()
        loss=loss+.05*kl
    if not torch.isfinite(loss):raise ValueError('Non-finite branch update')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5,error_if_nonfinite=True);opt.step()
    return {'updates':1,'loss':float(loss.detach()),'auxiliary_pairs':len(auxiliary),'min_return':float(q.min()),'max_return':float(q.max()),'anchor_kl':float(kl.detach()),'anchor_disabled_for_error':anchor_wrong,'optimizer_steps':1}
