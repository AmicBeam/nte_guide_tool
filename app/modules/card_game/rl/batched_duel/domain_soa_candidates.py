"""Device seven-model candidate encoding from explicit public numeric tables.

The caller must supply already projected visible cards and action descriptors.
This module neither generates legal actions nor projects raw private state.
"""
from dataclasses import dataclass
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.league_schema import OPTION_IDS
from app.modules.card_game.content.duel_v2 import CARDS

CARD_FIELDS = ('kind', 'owner', 'revealed', 'marked', 'delta', 'set_attack',
               'instant', 'copy', 'expires_turn', 'derived', 'antique_investment',
               'physical_kind', 'additional_ap_cost')
ACTION_FIELDS = ('kind', 'actor', 'card', 'target_side', 'target_seat', 'option')
INVALID = 1
supports_full_duel_rules = False


@dataclass(frozen=True)
class PreparedCandidates:
    owner: object
    sources: object
    buffers: object
    features: object
    valid: object
    errors: object
    action_start: int
    action_width: int


class DomainSoACandidates:
    """Stable device buffers; all runtime reads remain on the selected device."""
    supports_full_duel_rules = False
    legal_actions_available = False
    gpu_qualified = False

    def __init__(self, *, device='cuda', allow_cpu_oracle=False):
        import torch
        self.t = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no candidate CPU fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
        elif self.device.type != 'cpu' or not allow_cpu_oracle:
            raise RuntimeError('CPU mode is an explicit test oracle only')
        self.catalog = torch.tensor([[int(CARDS[c].get(k) or 0) for k in
                                      ('cost', 'attack', 'shield', 'hp', 'instant', 'derived')]
                                     for c in seven.CARD_IDS], dtype=torch.float64, device=self.device)
        self.closed = False

    def prepare(self, *, cards, actions, selected, top, bottom, counts, viewers,
                turn_counts, active, prior_errors, action_start=0, action_width=None):
        t = self.t
        if self.closed:
            raise RuntimeError('Candidate encoder closed')
        if self.device.type == 'cuda' and t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare candidate buffers outside capture')
        if not isinstance(cards, t.Tensor) or cards.ndim != 3:
            raise ValueError('Public cards must be a rank-three numeric table')
        B, K, F = cards.shape
        if F != len(CARD_FIELDS) or not 1 <= B <= 400 or not 1 <= K <= 1024:
            raise ValueError('Public card capacity exceeded')
        if not isinstance(actions, t.Tensor) or actions.ndim != 3 or actions.shape[0] != B:
            raise ValueError('Resident action descriptors required')
        A = actions.shape[1]
        if not 1 <= A <= 800 or actions.shape[2] != len(ACTION_FIELDS):
            raise ValueError('Action capacity exceeded')
        width=A-action_start if action_width is None else action_width
        if type(action_start) is not int or type(width) is not int or not 0<=action_start<A or not 1<=width<=A-action_start:
            raise ValueError('Invalid bounded candidate window')
        specs = {'cards': (cards, t.float64, (B,K,F)),
                 'actions': (actions, t.int64, (B,A,len(ACTION_FIELDS))),
                 'counts': (counts,t.int64,(B,)), 'viewers': (viewers,t.int64,(B,)),
                 'turn_counts': (turn_counts,t.int64,(B,)), 'active': (active,t.bool,(B,)),
                 'prior_errors': (prior_errors,t.int32,(B,))}
        for name,value,limit in (('selected',selected,64),('top',top,4),('bottom',bottom,4)):
            if not isinstance(value,t.Tensor) or value.ndim!=3 or value.shape[:2]!=(B,A) or not 1<=value.shape[2]<=limit:
                raise ValueError('Bounded '+name+' card-index lists required')
            specs[name]=(value,t.int64,tuple(value.shape))
        for name,(value,dtype,shape) in specs.items():
            if not isinstance(value,t.Tensor) or value.device!=self.device or value.dtype!=dtype or tuple(value.shape)!=shape:
                raise ValueError('Invalid resident '+name)
        sources={k:v[0] for k,v in specs.items()}
        buffers={k:v.clone(memory_format=t.contiguous_format) for k,v in sources.items()}
        return PreparedCandidates(self,sources,buffers,
            t.empty((B,width,seven.CAND_DIM),dtype=t.float32,device=self.device),
            t.empty((B,width),dtype=t.bool,device=self.device),t.empty(B,dtype=t.int32,device=self.device),action_start,width)

    def launch(self,p):
        t=self.t
        if self.closed or not isinstance(p,PreparedCandidates) or p.owner is not self:
            raise ValueError('Closed or foreign candidate buffers')
        for name,value in p.buffers.items():
            value.copy_(p.sources[name])
        f=dict(p.buffers);cards=f['cards'];total=f['actions'].shape[1]
        for name in ('actions','selected','top','bottom'):f[name]=f[name][:,p.action_start:p.action_start+p.action_width]
        a=f['actions'];B,A=a.shape[:2];K=cards.shape[1];N=len(seven.CARD_IDS)
        live=((t.arange(A,device=self.device)[None,:]+p.action_start)<f['counts'][:,None])&f['active'][:,None]
        kind,actor,ref,ts,seat,opt=a.unbind(-1)
        play=kind==3;choose=kind==4;item_on=play|choose
        mul=kind==5;order=kind==6;select=kind==8
        errors=((f['counts']<0)|(f['counts']>total)|(f['viewers']<0)|(f['viewers']>1)|(f['turn_counts']<0))
        bad=(kind<0)|(kind>=len(seven.ACTION_TYPES))|(actor< -1)|(actor>=len(seven.SEATS))|(ts< -1)|(ts>1)|(seat< -1)|(seat>=len(seven.SEATS))|(opt< -1)|(opt>=len(OPTION_IDS))
        bad|=((kind==1)|(kind==2)|(kind==7))&(actor<0)
        bad|=(ts==-1)&(seat!=-1)
        # All references are table-local public entries, never persistent IDs.
        def read(indices):
            return cards.gather(1,indices.clamp(0,K-1).reshape(B,-1,1).expand(-1,-1,len(CARD_FIELDS))).reshape(tuple(indices.shape)+(len(CARD_FIELDS),))
        def check_card(values,indices):
            k=values[...,0];owner=values[...,1];physical=values[...,11]
            failure=(indices<0)|(indices>=K)|~t.isfinite(values).all(-1)
            failure|=(k<0)|(k>=N)|(k!=k.trunc())|(owner<0)|(owner>=len(seven.SEATS))|(owner!=owner.trunc())
            failure|=(physical< -1)|(physical>=N)|(physical!=physical.trunc())
            for j in (2,3,5,6,7,9):failure|=(values[...,j]!=0)&(values[...,j]!=1)
            failure|=(values[...,4]!=values[...,4].trunc())|(values[...,4]< -1)|(values[...,4]>1)
            failure|=values[...,8]!=values[...,8].trunc()
            return failure
        item=read(ref);bad|=item_on&check_card(item,ref)
        item=t.where(item_on[...,None],item,t.zeros_like(item))
        ck=item[...,0].long().clamp(0,N-1);cat=self.catalog[ck]
        out=p.features
        out.zero_()
        out[...,0]=kind.float()
        out[...,1]=t.where(play,item[...,1],t.where((kind==1)|(kind==2)|(kind==7),actor,-1)).float()
        out[...,2]=t.where(item_on,ck,-1).float()
        out[...,3]=t.where(ts>=0,(ts!=f['viewers'][:,None]).long(),-1).float()
        out[...,4]=seat.float()
        # New-only action kinds obtain an unchanged legacy end-turn row.
        legacy=kind<6
        out[...,3:5]=t.where(legacy[...,None],out[...,3:5],t.full_like(out[...,3:5],-1))
        out[...,5:10]=t.where(item_on[...,None],cat[...,:5],0).float()
        out[...,10]=t.where(item_on,item[...,2],0).float()
        base=11+N;mark=base+len(OPTION_IDS);ctx=mark+4;physical=ctx+len(seven.CANDIDATE_CONTEXT)
        options=t.nn.functional.one_hot(opt.clamp(0,len(OPTION_IDS)-1),len(OPTION_IDS)).float()*(opt>=0)[...,None]*legacy[...,None]
        out[...,base:mark]=options
        out[...,mark:mark+4]=t.stack((item[...,3],t.where(item[...,3]!=0,item[...,4],0),item[...,5],item[...,6]),-1).float()*play[...,None]
        out[...,ctx:physical]=t.stack((item[...,7],((item[...,8]-f['turn_counts'][:,None]).clamp(min=0)/10),
                                      (item[...,9]!=cat[...,5]).double(),item[...,10]),-1).float()*item_on[...,None]
        face=item[...,11].long();has_face=item_on&(face>=0)&(face!=ck)
        out[...,physical:physical+N]=t.nn.functional.one_hot(face.clamp(0,N-1),N).float()*has_face[...,None]
        extra=seven.OLD_CAND_DIM
        for name,enabled,start,ordered in (('selected',mul,11,False),('top',order,extra,True),
                                          ('bottom',order,extra+4*N,True),('selected',select,extra+8*N,False)):
            indices=f[name];values=read(indices);present=indices>=0
            bad|=(enabled[...,None]&present&check_card(values,indices)).any(-1)
            # -1 padding must be a suffix; no silent holes or repeated cards.
            bad|=enabled&((indices< -1).any(-1)|((~present).cummax(-1).values&present).any(-1))
            sorted_refs=indices.sort(-1).values
            bad|=enabled&((sorted_refs[...,1:]==sorted_refs[...,:-1])&(sorted_refs[...,1:]>=0)).any(-1)
            kk=values[...,0].long().clamp(0,N-1)
            weight=(present&enabled[...,None]).float()
            if ordered:
                width=indices.shape[-1]*N
                block=out[...,start:start+width].reshape(B,A,indices.shape[-1],N)
                block.scatter_(-1,kk[...,None],weight[...,None])
            else:
                block=t.zeros((B,A,N),dtype=t.float32,device=self.device)
                block.scatter_add_(-1,kk,weight*.5)
                out[...,start:start+N]=block
        # Top and bottom are disjoint physical cards, even if faces match.
        cross=(f['top'][...,None]==f['bottom'][...,None,:])&(f['top'][...,None]>=0)&(f['bottom'][...,None,:]>=0)
        bad|=order&cross.any(-1).any(-1)
        out[...,-1]=(item[...,12]/10).float()*play
        bad|=~t.isfinite(out).all(-1)
        errors|=(live&bad).any(1)
        errors=errors.int()*INVALID
        errors=t.where(f['active'],errors,t.zeros_like(errors))
        errors=t.where(f['prior_errors']!=0,f['prior_errors'],errors)
        p.errors.copy_(errors)
        p.valid.copy_(live&(errors==0)[:,None])
        out.masked_fill_(~p.valid[...,None],0)
        return {'features':p.features,'valid':p.valid,'errors':p.errors,
                'candidate_names':tuple(seven.candidate_names()),'legal_actions_available':False,
                'public_projection_available':False,'full_duel_qualified':False,
                'action_start':p.action_start,'action_width':p.action_width,'descriptor_capacity':total}

    def encode(self,**kwargs):return self.launch(self.prepare(**kwargs))
    def close(self):self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
