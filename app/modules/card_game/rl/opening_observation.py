"""Joint play/mulligan schema: own hand and public lineups only, no hidden cards."""
from .public_schema import CAND_DIM
from .gpu_duel.catalog import CARD_IDS, CARD_INDEX, ACT_MULLIGAN

OPENING_SCHEMA = 'resident_public_v3'
OPENING_CAND_DIM = CAND_DIM + len(CARD_IDS)


def encode_opening(view, actions):
    import numpy as np
    from .public_observation import encode_public
    playing=[{'type':'end_turn'} if a['type']=='mulligan' else a for a in actions]
    state,base=encode_public(view,playing)
    extra=np.zeros((len(actions),len(CARD_IDS)),dtype=np.float32)
    hand={c['instance_id']:c['card_id'] for c in view['sides'][view['viewer_side']]['hand']}
    for i,action in enumerate(actions):
        if action['type']!='mulligan':continue
        base[i]=[ACT_MULLIGAN,-1,-1,-1,-1,0,0,0,0,0,0]
        ids=action.get('card_ids',[])
        if len(ids)>3 or len(set(ids))!=len(ids):raise ValueError('Invalid mulligan subset')
        for identifier in ids:
            extra[i,CARD_INDEX[hand[identifier]]]+=.5
    return state,np.concatenate((base,extra),axis=1)


def opening_observe(state):
    import torch
    from .public_observation import public_observe
    values,base,mask=public_observe(state)
    mulligan=state.legal_type==ACT_MULLIGAN
    fixed=torch.tensor([ACT_MULLIGAN,-1,-1,-1,-1,0,0,0,0,0,0],dtype=base.dtype,device=base.device)
    base=torch.where(mulligan[:,:,None],fixed,base)
    extra=torch.zeros(*mask.shape,len(CARD_IDS),device=base.device,dtype=base.dtype)
    batch=torch.arange(state.n,device=base.device)
    own=state.hand_kind[batch,state.active.long()]
    for slot in range(5):
        kind=own[:,slot]
        selected=mulligan & ((state.legal_hand>>slot)&1).bool() & (kind[:,None]>=0)
        indices=kind.clamp(min=0).long()[:,None,None].expand(-1,mask.shape[1],1)
        extra.scatter_add_(2,indices,selected[:,:,None].float()*.5)
    return values,torch.cat((base,extra),dim=-1),mask
