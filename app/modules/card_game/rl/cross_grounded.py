"""Categorical actions with actor/target public features; no rules or rewards.

Raw fixed-lineup encoding stays stable for old opponents. Only the new model
transforms its inputs, identically in NumPy inference and Torch optimization.
"""
import hashlib
from pathlib import Path
import numpy as np
from .cross_lineup import all_features, SEATS, CARD_IDS, CAND_DIM, EXTRA_CHAR
from .league_schema import CHAR_FIELDS, ACT_ATTACK, ACT_PLAY
from ..content.duel_v2 import CARDS

SCHEMA = 'cross_five_grounded_wdl_v1'
FIELDS = tuple(k for k in CHAR_FIELDS if k not in ('ch_shape','ch_order')) + tuple('fixed:'+k for k in EXTRA_CHAR)
INDEX = np.asarray([[[all_features().index(f'{k}:{s}:{c}') for k in FIELDS] for c in SEATS] for s in (0,1)])
CATEGORIES = ((0,6),(1,len(SEATS)+1),(2,len(CARD_IDS)+1),(3,3),(4,len(SEATS)+1))
DIM = sum(n for _,n in CATEGORIES) + CAND_DIM-5 + 2*len(FIELDS)
FRONT_INDEX = all_features().index('front:1')
BATTLE_CARDS = np.asarray([False]+[CARDS[cid]['type']=='battle' for cid in CARD_IDS])


def context_targets_numpy(x,c):
    """Explicit target, otherwise current visible front for automatic combat.

    This is pre-action context, not a prediction of the final resolved target.
    """
    targets=c[:,4].astype(np.int64).copy();sides=c[:,3].astype(np.int64).copy()
    combat=(c[:,0]==ACT_ATTACK)|((c[:,0]==ACT_PLAY)&BATTLE_CARDS[c[:,2].astype(np.int64)+1])
    implicit=combat&(targets<0)&(sides<0)
    targets[implicit]=int(np.rint(x[FRONT_INDEX]*10));sides[implicit]=1
    return sides,targets


def context_targets_tensor(x,c):
    import torch
    targets=c[...,4].long();sides=c[...,3].long()
    battle=torch.as_tensor(BATTLE_CARDS,device=c.device)[c[...,2].long()+1]
    combat=(c[...,0]==ACT_ATTACK)|((c[...,0]==ACT_PLAY)&battle)
    implicit=combat&(targets<0)&(sides<0)
    front=torch.round(x[:,FRONT_INDEX]*10).long()[:,None].expand_as(targets)
    return torch.where(implicit,torch.ones_like(sides),sides),torch.where(implicit,front,targets)


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def transform(x,c):
    """Single observation and arbitrary legal candidate count, including zero."""
    if len(c) == 0:
        return np.zeros((0, DIM), dtype=np.float32)
    blocks=[]
    for col,n in CATEGORIES:
        ids=c[:,col].astype(np.int64)+(col!=0)
        if ((ids<0)|(ids>=n)).any():raise ValueError('Unknown categorical action')
        blocks.append(np.eye(n,dtype=np.float32)[ids])
    numeric=c[:,5:].copy();numeric[:,:4]/=10
    blocks.append(numeric)
    actors=c[:,1].astype(np.int64);sides,targets=context_targets_numpy(x,c)
    heroes=x[INDEX]
    blocks.append(heroes[0,np.maximum(actors,0)]*(actors>=0)[:,None])
    blocks.append(heroes[np.maximum(sides,0),np.maximum(targets,0)]*((targets>=0)&(sides>=0))[:,None])
    return np.concatenate(blocks,axis=-1).astype(np.float32)


def network(hidden):
    import torch
    from .gpu_duel.ppo import CompactScorer

    class GroundedScorer(CompactScorer):
        grounded=True

        def __init__(self):
            super().__init__(len(all_features()),DIM,hidden)
            self.wdl=torch.nn.Linear(hidden+1,3)
            self.register_buffer('actor_indices',torch.tensor(INDEX),persistent=False)

        def candidate_features(self,x,c):
            blocks=[]
            for col,n in CATEGORIES:
                ids=c[...,col].long()+(col!=0)
                # Padding uses all-zero raw candidates and is masked after scoring.
                blocks.append(torch.nn.functional.one_hot(ids,n).to(x.dtype))
            numeric=c[...,5:].clone();numeric[...,:4]/=10
            blocks.append(numeric)
            heroes=x[:,self.actor_indices]
            batch=torch.arange(len(x),device=x.device)[:,None]
            actors=c[...,1].long();sides,targets=context_targets_tensor(x,c)
            blocks.append(heroes[batch,0,actors.clamp_min(0)]*(actors>=0).unsqueeze(-1))
            blocks.append(heroes[batch,sides.clamp_min(0),targets.clamp_min(0)]*((targets>=0)&(sides>=0)).unsqueeze(-1))
            return torch.cat(blocks,-1)

        def forward(self,x,c,mask):
            return super().forward(x,self.candidate_features(x,c),mask)

    return GroundedScorer()
