"""Exact episode-weighted metrics; CUDA only batches visible value queries."""
import numpy as np


def compute(games,policies,*,device='cuda'):
    import torch
    from .seven_runtime import restore
    from scripts.check_duel_v2_value_calibration import probability_metrics
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('CUDA metrics required; no CPU fallback')
    if torch.backends.cuda.matmul.allow_tf32:raise ValueError('TF32 must be disabled for numerical checks')
    result={}
    for key,policy in policies.items():
        net=restore(policy_directory(policy),key,device)[0] if hasattr(policy,'directory') else None
        if net is None:raise ValueError('Versioned numerical policy directory missing')
        rows=[];labels=[];weights=[];count=0
        for game in games:
            own=[r for r in game['value_rows'] if r['key']==key]
            if not own:continue
            count+=1;rows.extend(own);labels.extend(int(r['z'])+1 for r in own);weights.extend([1/len(own)]*len(own))
        probabilities=[]
        with torch.no_grad():
            for start in range(0,len(rows),256):
                batch=rows[start:start+256]
                x=torch.as_tensor(np.asarray([r['x'] for r in batch]),device=device)
                actors=torch.as_tensor([[r['value_actor']] for r in batch],dtype=x.dtype,device=device)
                h=net.value_net(x);p=net.wdl(torch.cat((h,actors),-1)).softmax(-1).cpu().numpy()
                probabilities.extend(p)
        result[key]=dict(**probability_metrics(probabilities,labels,weights),normal_games=count)
    return result


def policy_directory(policy):return policy.directory
