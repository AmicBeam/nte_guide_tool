"""Outcome-only PPO and terminal branch distillation; all Torch imports are lazy."""
import numpy as np

def tensors(rows,device):
    import torch
    n=len(rows);width=max(len(r[1]) for r in rows);d=rows[0][1].shape[1]
    state=np.stack([r[0] for r in rows]);cand=np.zeros((n,width,d),dtype=np.float32);mask=np.zeros((n,width),dtype=bool)
    for i,r in enumerate(rows):cand[i,:len(r[1])]=r[1];mask[i,:len(r[1])]=True
    return tuple(torch.as_tensor(x,device=device) for x in (state,cand,mask))

def ordinary_update(net,opt,games,seed):
    import torch
    rows=[(*entry,float(g['reward'])) for g in games if g['complete'] for entry in g['trajectory']]
    if not rows:return {'updates':0,'samples':0}
    rng=np.random.default_rng(seed)
    if len(rows)>512:rows=[rows[i] for i in rng.choice(len(rows),512,replace=False)]
    dev=next(net.parameters()).device;losses=[]
    for _ in range(2):
        for start in range(0,len(rows),64):
            batch=rows[start:start+64];x,c,m=tensors(batch,dev)
            action=torch.tensor([r[2] for r in batch],device=dev);oldlogp=torch.tensor([r[3] for r in batch],device=dev)
            target=torch.tensor([r[5] for r in batch],device=dev);oldvalue=torch.tensor([r[4] for r in batch],device=dev)
            logits,value=net(x,c,m);dist=torch.distributions.Categorical(logits=logits)
            ratio=(dist.log_prob(action)-oldlogp).clamp(-20,20).exp();adv=((target-oldvalue)/10).clamp(-2,2)
            loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*((value-target)/10).square().mean()-.01*dist.entropy().mean()
            if not torch.isfinite(loss):raise ValueError('Non-finite PPO update')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step();losses.append(float(loss.detach()))
    return {'updates':1,'optimizer_steps':len(losses),'samples':len(rows),'loss':sum(losses)/len(losses)}

def branch_update(net,opt,x,c,returns,*,anchor_logits=None,all_candidates=None):
    import torch
    q=np.asarray(returns,dtype=np.float32)
    if len(q)<2 or float(np.ptp(q))<1e-6:return {'updates':0,'reason':'equal_terminal_returns'}
    dev=next(net.parameters()).device;xx,cc,mm=tensors([(x,c)],dev)
    target=torch.softmax(torch.as_tensor(q,device=dev)/4,0)
    logits,value=net(xx,cc,mm);loss=-(target*logits[0].log_softmax(0)).sum()+.25*((value[0]-(target*torch.as_tensor(q,device=dev)).sum())/10).square()
    kl=torch.tensor(0.,device=dev)
    if anchor_logits is not None:
        ax,ac,am=tensors([(x,all_candidates)],dev)
        current,_=net(ax,ac,am)
        reference=torch.softmax(torch.as_tensor(anchor_logits,device=dev),0)
        kl=(reference*(reference.clamp(min=1e-20).log()-current[0].log_softmax(0))).sum()
        loss=loss+.05*kl
    if not torch.isfinite(loss):raise ValueError('Non-finite branch update')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step()
    return {'updates':1,'loss':float(loss.detach()),'min_return':float(q.min()),'max_return':float(q.max()),'anchor_kl':float(kl.detach())}
