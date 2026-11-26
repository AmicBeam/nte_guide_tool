"""Ordered device target projection for explicit public card queries.

Descriptors are (kind, absolute side, stored character/card index). NONE is an
actual [{}] target, not an empty result. This is neither payment validation nor
complete legal-action generation or physical execution.
"""
from dataclasses import dataclass
import numpy as np
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.content.duel_v2 import CARDS,CHARACTERS
from app.modules.card_game.content.duel_v2.effects import TARGET_POLICIES
from .domain_soa_schema import (DomainSoALayout,array_specs,catalog_fingerprint,CARD_VOCAB,CHARACTER_VOCAB,
    CHARACTER_INT_FIELDS,CHARACTER_BOOL_FIELDS,CHARACTER_STR_FIELDS,CARD_STR_FIELDS,CARD_BOOL_FIELDS,FLAG_INT_FIELDS)

NONE,PLAYER,CHARACTER,DISCARD=0,1,2,3
INVALID,UNSUPPORTED=1,2
POLICIES=(None,'living_ally','ally_esper','other_living_ally','downed_ally','other_downed_ally',
    'downed_enemy','enemy_bench','living_enemy','living_enemy_esper','collapsed_enemy_esper',
    'living_enemy_role','any_living_esper','living_role','injured_role','enemy_not_in_lineup',
    'any_summon','enemy_front','ally_front','enemy_hand','own_tactic_discard')
CI={k:i for i,k in enumerate(CHARACTER_INT_FIELDS)};CB={k:i for i,k in enumerate(CHARACTER_BOOL_FIELDS)}
CS={k:i for i,k in enumerate(CHARACTER_STR_FIELDS)};KS={k:i for i,k in enumerate(CARD_STR_FIELDS)}
KB={k:i for i,k in enumerate(CARD_BOOL_FIELDS)};FI={k:i for i,k in enumerate(FLAG_INT_FIELDS)}
supports_full_duel_rules=False


@dataclass(frozen=True)
class PreparedTargets:
    owner: object
    sources: object
    fields: object
    inputs: object
    buffers: object
    targets: object
    counts: object
    errors: object


class DomainSoATargets:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False):
        import torch
        self.t=torch;self.layout=layout or DomainSoALayout();self.device=torch.device(device);self.closed=False
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no target CPU fallback')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        elif self.device.type!='cpu'or allow_cpu_oracle is not True:raise ValueError('Explicit CPU oracle required')
        self.specs=array_specs(self.layout)
        self.policies=torch.tensor([POLICIES.index(TARGET_POLICIES.get(c))if TARGET_POLICIES.get(c)in POLICIES else -1 for c in seven.CARD_IDS],device=self.device)
        nc=max(CHARACTER_VOCAB.values())+1;nk=max(CARD_VOCAB.values())+1
        self.char_seat=torch.full((nc,),-1,dtype=torch.int64,device=self.device)
        self.char_defaults=torch.zeros(nc,len(CI),dtype=torch.int64,device=self.device)
        for cid,i in CHARACTER_VOCAB.items():
            self.char_seat[i]=seven.SEAT_INDEX[cid]
            for k,j in CI.items():
                v=CHARACTERS[cid].get(k);self.char_defaults[i,j]=v if type(v)is int else 0
        self.card_owner=torch.full((nk,),-1,dtype=torch.int64,device=self.device)
        self.card_tactic=torch.zeros(nk,dtype=torch.bool,device=self.device)
        for cid,i in CARD_VOCAB.items():
            self.card_owner[i]=seven.SEAT_INDEX[CARDS[cid]['character_id']];self.card_tactic[i]=CARDS[cid]['type']=='tactic'
        self.digest=torch.tensor(list(catalog_fingerprint()),dtype=torch.uint8,device=self.device)
        self.labels={s:torch.tensor(list(s.encode()),dtype=torch.uint8,device=self.device)for s in (*seven.SEATS,'tactic','C08')}

    def prepare(self,fields,sides,card_kinds,owners,is_battle,active,prior_errors,normal_attacks=None):
        t=self.t
        if self.closed:raise RuntimeError('Target projector closed')
        if self.device.type=='cuda'and t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare target buffers outside capture')
        if not isinstance(sides,t.Tensor)or sides.ndim!=2:raise ValueError('Rank-two query sides required')
        B,N=sides.shape
        if not 1<=B<=min(400,self.layout.game_capacity)or not 1<=N<=800:raise ValueError('Target query capacity exceeded')
        if normal_attacks is None:normal_attacks=t.zeros_like(active)
        inputs=dict(sides=sides,card_kinds=card_kinds,owners=owners,is_battle=is_battle,active=active,prior_errors=prior_errors,normal_attacks=normal_attacks)
        for (name,v),dtype in zip(inputs.items(),(t.int64,t.int64,t.int64,t.bool,t.bool,t.int32,t.bool)):
            if not isinstance(v,t.Tensor)or v.device!=self.device or v.dtype!=dtype or tuple(v.shape)!=(B,N):raise ValueError('Resident '+name+' contract mismatch')
        cols=fields if isinstance(fields,dict)else getattr(fields,'arrays',None)
        if cols is None:raise TypeError('Named resident target fields required')
        sources={};dtype={'bool':t.bool,'uint8':t.uint8,'int8':t.int8,'int16':t.int16,'int32':t.int32,'int64':t.int64,'float64':t.float64}
        for name,(kind,tail)in self.specs.items():
            v=cols.get(name)
            if not isinstance(v,t.Tensor)or v.device!=self.device or v.dtype!=dtype[str(np.dtype(kind))]or tuple(v.shape)!=(B,)+tuple(tail):raise ValueError('Resident typed target field required: '+name)
            sources[name]=v
        T=1+2*(self.layout.max_characters+1)+self.layout.max_zone
        return PreparedTargets(self,sources,{k:v.clone(memory_format=t.contiguous_format)for k,v in sources.items()},inputs,
            {k:v.clone(memory_format=t.contiguous_format)for k,v in inputs.items()},t.empty((B,N,T,3),dtype=t.int64,device=self.device),
            t.empty((B,N),dtype=t.int32,device=self.device),t.empty((B,N),dtype=t.int32,device=self.device))

    def _eq(self,f,slots,text):
        t=self.t;literal=self.labels[text];shape=slots.shape;B=shape[0]
        safe=slots.long().clamp(0,self.layout.max_string_slots-1)
        lengths=f['string_length'].gather(1,safe.flatten(1)).reshape(shape).long()
        offsets=f['string_offset'].gather(1,safe.flatten(1)).reshape(shape).long()
        bad=(slots<0)|(slots>=self.layout.max_string_slots)|(slots>=f['string_count'].reshape((B,)+(1,)*(len(shape)-1)))|(lengths<0)|(offsets<0)|((offsets+lengths)>f['string_bytes_used'].reshape((B,)+(1,)*(len(shape)-1)))|((offsets+lengths)>self.layout.max_string_bytes)
        pos=offsets[...,None]+t.arange(len(text),device=self.device)
        raw=f['string_bytes'].gather(1,pos.clamp(0,self.layout.max_string_bytes-1).flatten(1)).reshape(tuple(shape)+(len(text),))
        return (raw==literal).all(-1)&(lengths==len(text))&~bad,bad

    def launch(self,p):
        if self.closed or not isinstance(p,PreparedTargets)or p.owner is not self:raise ValueError('Closed or foreign target buffers')
        t=self.t
        for k,v in p.fields.items():v.copy_(p.sources[k])
        for k,v in p.buffers.items():v.copy_(p.inputs[k])
        f=p.fields;q=p.buffers;B,N=q['sides'].shape;C=self.layout.max_characters;Z=self.layout.max_zone;K=self.layout.max_cards
        rows=t.arange(B,device=self.device);side=q['sides'].clamp(0,1);kind=q['card_kinds'].clamp(0,len(seven.CARD_IDS)-1);owner=q['owners'];pol=self.policies[kind]
        bad=(q['sides']<0)|(q['sides']>1)|(q['card_kinds']<0)|(q['card_kinds']>=len(seven.CARD_IDS))|(owner<0)|(owner>=len(seven.SEATS))
        normal=q['normal_attacks']
        unsupported=((pol<0)&~normal)|(owner==seven.SEAT_INDEX['summon'])
        stored=f['player_character_count'].long();declared=f['player_order_count'].long();pos=t.arange(C,device=self.device).reshape(1,1,C)
        count=t.where(declared==0,stored,declared)
        ordered=t.where((declared==0)[...,None],pos,f['player_order_slot'].long())
        valid=pos<count[...,None];safe=ordered.clamp(0,C-1)
        def order(x):return x.gather(2,safe)
        tm=f['character_template'].long();ctm=tm.clamp(0,len(self.char_seat)-1)
        actual_seat=order(self.char_seat[ctm]);hp=order(f['character_hp']);summoned=order((f['character_bool_presence'][...,CB['summoned']]==2)&f['character_bool_value'][...,CB['summoned']])
        def cint(k):
            j=CI[k];pr=f['character_int_presence'][...,j]
            return order(t.where(pr==3,self.char_defaults[ctm,j],t.where(pr==2,f['character_int_value'][...,j],0)))
        down=cint('down_turns');maxhp=cint('max_hp');alive=valid&(hp>0)
        holders=t.arange(2,device=self.device).reshape(1,2,1)*(C+1)+1+safe
        collapse=(f['flag_collapse_presence'].gather(1,holders.flatten(1)).reshape(B,2,C)==2)
        rawbad=((stored<1)|(stored>C)|(count<0)|(count>C)).any(-1)
        rawbad|=(valid&((ordered<0)|(ordered>=stored[...,None])|~order(f['character_valid']))).flatten(1).any(-1)
        rawbad|=(f['catalog_digest']!=self.digest[None,:]).any(-1)|~f['row_valid']|(f['phase']<0)|(f['phase']>3)
        rawbad|=((f['zone_count']<0)|(f['zone_count']>Z)).flatten(1).any(-1)
        rawbad|=(f['string_count']<0)|(f['string_count']>self.layout.max_string_slots)|(f['string_bytes_used']<0)|(f['string_bytes_used']>self.layout.max_string_bytes)
        bad|=rawbad[:,None]
        # Fixed absolute-side pool: player then team_order, a before b.
        R=2*(C+1);rs=t.arange(2,device=self.device).repeat_interleave(C+1)
        isplayer=(t.arange(R,device=self.device)%(C+1)==0)
        def role(x,player_value=0):
            return t.cat((t.full((B,2,1),player_value,dtype=x.dtype,device=self.device),x),-1).flatten(1)
        rolevalid=role(valid,True);ralive=role(alive,True)&~isplayer[None,:]
        rsummon=role(summoned,False);rdown=role(down);rhp=role(hp);rmax=role(maxhp);rcollapse=role(collapse,False);rseat=role(actual_seat,-1)
        own=(rs[None,None,:]==side[...,None]);enemy=~own;chars=(~isplayer)[None,None,:]
        live=ralive[:,None,:];es=enemy&live&~rsummon[:,None,:]
        # Canhong's battle-target override uses the actual owner's passive kit.
        actor_match=(self.char_seat[ctm][:,None,:,:]==owner[:,:,None,None])&f['character_valid'][:,None,:,:]&(t.arange(2,device=self.device)[None,None,:,None]==side[...,None,None])
        actor_n=actor_match.flatten(2).sum(-1);actor_slot=(actor_match.long()*t.arange(C,device=self.device).reshape(1,1,1,C)).sum((2,3)).clamp(0,C-1)
        actor_hp=f['character_hp'][rows[:,None],side,actor_slot]
        ah=side*(C+1)+1+actor_slot
        mimic=f['flag_mimic_template'].gather(1,ah).long()
        wrapped=(f['flag_mimic_presence'].gather(1,ah)==2)&(f['flag_mimic_id_presence'].gather(1,ah)==2)&(mimic>0)
        kit=t.where(wrapped,mimic,tm[rows[:,None],side,actor_slot])
        shape_slot=f['character_str_slot'][rows[:,None],side,actor_slot,CS['shape']]
        shaped,shapebad=self._eq(f,shape_slot,'C08');shape_on=f['character_str_presence'][rows[:,None],side,actor_slot,CS['shape']]==2
        burn=f['zone_status_presence'][:,:,1]==2;mask=f['zone_status_i32'][:,:,1,7];burn_by=f['zone_status_side'][:,:,1,0].long()
        stack_slot=((mask&1)!=0).long()
        burn_stacks=t.where((mask&(1<<2))!=0,f['zone_status_i32'][:,:,1].gather(-1,stack_slot[...,None]).squeeze(-1),0);burn_stacks=t.where(burn_stacks==0,1,burn_stacks)
        totals=[]
        for s in range(2):
            value=t.where(burn&((mask&(1<<1))!=0)&(burn_by==s),burn_stacks,0).sum(1)
            value+=t.where(f['zone_dot_valid']&(f['zone_dot_by']==s),f['zone_dot_stacks'],0).sum((1,2))
            totals.append(value)
        total=t.stack(totals,1).gather(1,side)
        ready=(actor_n==1)&(actor_hp>0)&(kit==CHARACTER_VOCAB['canhong'])&(f['phase'][:,None]!=3)&shape_on&shaped&(total>=5)
        override=~normal&q['is_battle']&(owner==seven.SEAT_INDEX['canhong'])&ready
        normal_override=normal&ready&es.any(-1)
        uses_callback=q['is_battle']|normal
        bad|=uses_callback&(actor_n>1)
        legacy_mimic=t.where(f['flag_int_presence'][:,:,FI['mimic']]>=2,f['flag_int_value'][:,:,FI['mimic']],0).gather(1,ah)
        unsupported|=uses_callback&(actor_n==1)&(legacy_mimic!=0)
        bad|=uses_callback&(actor_n==1)&shape_on&shapebad
        keep=t.zeros((B,N,1+R+Z),dtype=t.bool,device=self.device)
        def match(name):return pol==POLICIES.index(name)
        conditions={
            'living_ally':own&live,'ally_esper':own&chars&rolevalid[:,None,:]&~rsummon[:,None,:],
            'other_living_ally':own&live&(rseat[:,None,:]!=owner[...,None]),
            'downed_ally':own&chars&rolevalid[:,None,:]&(rdown[:,None,:]>0),
            'other_downed_ally':own&chars&rolevalid[:,None,:]&(rdown[:,None,:]>0)&~rsummon[:,None,:]&(rseat[:,None,:]!=owner[...,None]),
            'downed_enemy':enemy&chars&rolevalid[:,None,:]&(rdown[:,None,:]>0),
            'living_enemy':enemy&live,'living_enemy_esper':es,'collapsed_enemy_esper':es&rcollapse[:,None,:],
            'living_enemy_role':enemy&(live|isplayer[None,None,:]),
            'any_living_esper':live&~rsummon[:,None,:],
            'living_role':own&(live|isplayer[None,None,:]),
            'injured_role':own&live&(rhp[:,None,:]<rmax[:,None,:]),
            'any_summon':live&rsummon[:,None,:]}
        # Injured player precedes the ordered characters on the same side.
        player_injured=(f['player_hp']<f['player_max_hp'])[:,rs]
        conditions['injured_role']|=own&isplayer[None,None,:]&player_injured[:,None,:]
        front=f['player_front_slot'][:,rs];front_on=(f['player_front_presence'][:,rs]==2)
        stored_roles=role(ordered,-1)
        conditions['enemy_bench']=enemy&live&(~front_on[:,None,:]|(stored_roles[:,None,:]!=front[:,None,:]))
        own_tm=order(tm)[rows[:,None],side];own_valid=valid[rows[:,None],side]
        member=(role(order(tm),-1)[:,None,:,None]==own_tm[:,:,None,:])&own_valid[:,:,None,:]
        conditions['enemy_not_in_lineup']=es&~member.any(-1)
        roles=t.zeros((B,N,R),dtype=t.bool,device=self.device)
        for name,allowed in conditions.items():roles|=match(name)[...,None]&allowed
        roles=t.where((override|normal_override)[...,None],es,roles&~normal[...,None])
        keep[:,:,1:1+R]=roles
        no_target=match(None)
        no_target|=match('enemy_front')&(f['player_front_presence'][rows[:,None],1-side]==2)
        no_target|=match('ally_front')&(f['player_front_presence'][rows[:,None],side]==2)
        no_target|=match('enemy_hand')&(f['zone_count'][rows[:,None],1-side,0]>0)
        keep[:,:,0]=t.where(normal,~normal_override,no_target&~override)
        # Ordered own tactic discard uses physical card-instance slots.
        refs=f['zone_card_index'][rows[:,None],side,2].long();safe_refs=refs.clamp(0,K-1)
        discarded=t.arange(Z,device=self.device)[None,None,:]<f['zone_count'][rows[:,None],side,2,None]
        kt=f['card_template'].long().clamp(0,len(self.card_owner)-1)
        author=self.card_owner[kt];tactic=self.card_tactic[kt]
        pr=f['card_str_presence'][...,KS['character_id']];slots=f['card_str_slot'][...,KS['character_id']]
        decoded=t.full_like(author,-1)
        for name,index in seven.SEAT_INDEX.items():
            eq,_=self._eq(f,slots,name);decoded=t.where(eq,t.full_like(decoded,index),decoded)
        author=t.where(pr==2,decoded,t.where(pr==3,author,-1))
        tactic_eq,_=self._eq(f,f['card_str_slot'][...,KS['type']],'tactic')
        tp=f['card_str_presence'][...,KS['type']];tactic=t.where(tp==2,tactic_eq,(tp==3)&tactic)
        copied=(f['card_bool_presence'][...,KB['copy']]==2)&f['card_bool_value'][...,KB['copy']]
        dkeep=discarded&f['card_valid'][rows[:,None,None],safe_refs]&(author[rows[:,None,None],safe_refs]==owner[...,None])&tactic[rows[:,None,None],safe_refs]&~copied[rows[:,None,None],safe_refs]
        keep[:,:,1+R:]=match('own_tactic_discard')[...,None]&dkeep&~override[...,None]&~normal[...,None]
        bad|=match('own_tactic_discard')&(discarded&((refs<0)|(refs>=K))).any(-1)
        errors=bad.int()*INVALID|unsupported.int()*UNSUPPORTED
        errors=t.where(q['active'],errors,t.zeros_like(errors));errors=t.where(q['prior_errors']!=0,q['prior_errors'],errors)
        errors=t.where(f['row_error'][:,None]!=0,f['row_error'][:,None],errors)
        good=q['active']&(errors==0);keep&=good[...,None]
        counts=keep.sum(-1).int();T=keep.shape[-1];ordinal=t.arange(T,device=self.device)
        permutation=t.where(keep,ordinal,ordinal+T).argsort(dim=-1,stable=True)
        raw=p.targets;raw.fill_(-1);raw[:,:,0,0]=NONE
        raw[:,:,1:1+R,0]=t.where(isplayer,PLAYER,CHARACTER)[None,None,:]
        raw[:,:,1:1+R,1]=rs[None,None,:];raw[:,:,1:1+R,2]=stored_roles[:,None,:]
        raw[:,:,1+R:,0]=DISCARD;raw[:,:,1+R:,1]=side[...,None];raw[:,:,1+R:,2]=refs
        packed=raw.gather(2,permutation[...,None].expand(-1,-1,-1,3))
        raw.copy_(t.where((ordinal[None,None,:]<counts[...,None])[...,None],packed,t.full_like(packed,-1)))
        p.counts.copy_(counts);p.errors.copy_(errors)
        return {'targets':raw,'counts':p.counts,'errors':p.errors,'full_duel_qualified':False,'legal_actions_available':False}

    def project(self,fields,sides,card_kinds,owners,is_battle,active,prior_errors,normal_attacks=None):return self.launch(self.prepare(fields,sides,card_kinds,owners,is_battle,active,prior_errors,normal_attacks))
    def close(self):self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
