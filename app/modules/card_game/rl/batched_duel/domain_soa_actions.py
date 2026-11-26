"""Phase-complete read-only device action enumeration for accepted domain states.

The network list excludes concede. No physical transition, sampling or H3 is
implemented or qualified by this module.
"""
from dataclasses import dataclass
from .domain_soa_playing_actions import DomainPlayingActions,INVALID,UNSUPPORTED,OVERFLOW
from .domain_soa_choices import DomainChoiceActions
from .domain_soa_schema import CHOICE_KINDS,CHOICE_INT_KEYS,CHOICE_BOOL_KEYS,CARD_VOCAB,CHARACTER_VOCAB

supports_full_duel_rules=False


@dataclass(frozen=True)
class PreparedActions:
    owner: object
    playing: object
    choices: object
    selected: object
    top: object
    bottom: object
    bottom_plural: object


class DomainSoAActions(DomainPlayingActions):
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False,max_actions=800):
        super().__init__(layout,device=device,allow_cpu_oracle=allow_cpu_oracle,max_actions=max_actions)
        self.S=max(4,self.layout.max_choice_cards)
        if self.S>64:raise ValueError('Choice selection capacity exceeds direct contract')
        self.W=min(64,max(self.layout.max_choice_cards,self.layout.max_zone))
        self.choices=DomainChoiceActions(device=self.device,allow_cpu_reference=allow_cpu_oracle,
                                        max_actions=min(800,max_actions+1),selection_width=self.S)

    def prepare(self,fields,viewers,active):
        t=self.t;parent=super().prepare(fields,viewers,active);B=viewers.shape[0];A=self.max_actions
        cb=self.choices.prepare(t.zeros((B,13),dtype=t.int32,device=self.device),t.full((B,self.W),-1,dtype=t.int32,device=self.device))
        return PreparedActions(self,parent,cb,t.empty((B,A,self.S),dtype=t.int64,device=self.device),
            t.empty((B,A,4),dtype=t.int64,device=self.device),t.empty((B,A,4),dtype=t.int64,device=self.device),
            t.empty((B,A),dtype=t.bool,device=self.device))

    def launch(self,p):
        if self.closed or not isinstance(p,PreparedActions)or p.owner is not self:raise ValueError('Closed or foreign complete-action buffers')
        t=self.t;out=super().launch(p.playing);pp=p.playing.public;b=pp.observation;f=b.fields;B=b.viewers.shape[0];A=self.max_actions;K=self.layout.max_cards;C=self.C
        rows=t.arange(B,device=self.device);own=b.viewers.clamp(0,1);phase=f['phase'];mul=phase==0;choice=phase==2;stage=mul|choice
        cs=f['choice_side'].long().clamp(0,1);is_actor=cs==own
        done=((t.arange(2,device=self.device)[None,:]<f['mulligan_count'][:,None])&(f['mulligan_sides'].long()==own[:,None])).any(-1)
        actor=t.where(mul,~done,is_actor)
        own_hand=f['zone_count'][rows,own,0].int();n=t.where(mul,own_hand,f['choice_card_count'].int())
        cp=f['choice_int_presence'];cv=f['choice_int_value'];bp=f['choice_bool_presence'];bv=f['choice_bool_value']
        def integer(k):i=CHOICE_INT_KEYS.index(k);return t.where(cp[:,i]==2,cv[:,i],0)
        def boolean(k):i=CHOICE_BOOL_KEYS.index(k);return (bp[:,i]==2)&bv[:,i]
        kind=t.where(mul,CHOICE_KINDS.index('hand_redraw'),f['choice_kind'].int())
        deck=choice&(boolean('deck_order')|(kind==CHOICE_KINDS.index('heiyu_divination')))
        minimum=integer('min_bottom');maximum=integer('max_bottom');hasmax=cp[:,CHOICE_INT_KEYS.index('max_bottom')]!=0
        multi=boolean('multi_bottom');placement=boolean('hand_placement')
        # Resolve the actor_only/include_down deck-order callback, not an
        # unconditional heiyu-kind shortcut. Mimicked Heiyu may also resolve it.
        tm=f['character_template'].long();opside=f['op_side'].long().clamp(0,1);opslot=f['op_actor_slot'].long()
        optm=tm[rows,opside,opslot.clamp(0,C-1)]
        cross=(tm[rows,cs]==optm[:,None])&f['character_valid'][rows,cs]
        cross_slot=t.where(cross.any(-1),(cross.long()*t.arange(C,device=self.device)).sum(-1),-1)
        fallback=t.where(opside==cs,opslot,cross_slot)
        first=t.where(f['player_order_count'][rows,cs]>0,f['player_order_slot'][rows,cs,0].long(),0)
        fallback=t.where(f['operation_present']&(opslot>=0),fallback,first)
        actor_slot=t.where(f['choice_actor_presence']==2,f['choice_actor_slot'].long(),fallback)
        valid_actor=(actor_slot>=0)&(actor_slot<f['player_character_count'][rows,cs])
        safe=actor_slot.clamp(0,C-1);holder=cs*(C+1)+1+safe
        wrapped=(f['flag_mimic_presence'][rows,holder]==2)&(f['flag_mimic_id_presence'][rows,holder]==2)&(f['flag_mimic_template'][rows,holder]>0)
        kit=t.where(wrapped,f['flag_mimic_template'][rows,holder].long(),tm[rows,cs,safe])
        resolve=choice&deck&valid_actor&(tm[rows,cs,safe]>0)&(kit==CHARACTER_VOCAB['heiyu'])&(kind==CHOICE_KINDS.index('heiyu_divination'))
        opref=f['op_card_index'].long();opcard=f['card_template'][rows,opref.clamp(0,K-1)].long()
        minimum=t.where(resolve,(f['operation_present']&(opref>=0)&(opcard==CARD_VOCAB['PF01'])).int(),minimum)
        maximum=t.where(resolve,n,maximum);hasmax|=resolve;multi|=resolve
        bad=choice&(~f['pending_choice_present']|(f['choice_side']<0)|(f['choice_side']>1))
        bad|=choice&actor&(kind==CHOICE_KINDS.index('kit_multi'))&(cp[:,CHOICE_INT_KEYS.index('max_select')]!=2)
        bad|=choice&actor&deck&~placement&hasmax&~resolve&(cp[:,CHOICE_INT_KEYS.index('max_bottom')]!=2)
        bad|=mul&((f['mulligan_count']<0)|(f['mulligan_count']>2))
        inputs=p.choices.inputs
        inputs.copy_(t.stack((kind,t.where(bad,-1,n),integer('max_select'),minimum,maximum,hasmax.int(),multi.int(),placement.int(),deck.int(),
                              t.ones_like(n),actor.int(),(phase==3).int(),(stage&b.active).int()),1))
        refs=p.choices.references;refs.fill_(-1)
        hand=t.full_like(refs,-1);ch=t.full_like(refs,-1)
        h=min(self.W,pp.hand_refs.shape[1]);z=min(self.W,pp.choice_refs.shape[1])
        hand[:,:h].copy_(pp.hand_refs[:,:h].int());ch[:,:z].copy_(pp.choice_refs[:,:z].int())
        refs.copy_(t.where(mul[:,None],hand,ch))
        cb=self.choices.launch(p.choices);tokens=cb.tokens[:,1:];ordinals=cb.selections[:,1:];width=tokens.shape[1]
        counts=(cb.counts.long()-1).clamp(min=0)
        errors=t.where(stage,pp.errors|cb.errors,out['errors'])
        errors=t.where(stage&bad&b.active,errors|INVALID,errors)
        errors|=(stage&(counts>A)).int()*OVERFLOW
        # Frozen seven-model order fields have four slots in each direction.
        wide=stage&((tokens[...,0]==6)&((tokens[...,2]>4)|(tokens[...,3]>4))).any(-1)
        errors|=wide.int()*UNSUPPORTED
        errors=t.where(pp.errors!=0,pp.errors,errors)
        valid=(t.arange(width,device=self.device)[None,:]<counts[:,None])&(errors==0)[:,None]&b.active[:,None]&stage[:,None]
        selected=refs.long().gather(1,ordinals.long().clamp(0,self.W-1).flatten(1)).reshape(B,width,self.S)
        selected=t.where(ordinals>=0,selected,-1)
        code=t.where(mul[:,None]&(tokens[...,0]==8),5,tokens[...,0].long())
        choice_actions=t.full((B,width,6),-1,dtype=t.int64,device=self.device);choice_actions[...,0]=code
        choice_actions[...,2]=t.where(code==4,tokens[...,1].long(),-1)
        choice_payload=t.full((B,width,7),-1,dtype=t.int64,device=self.device);choice_payload[...,0]=code;choice_payload[...,3]=0
        choice_payload[...,2]=t.where(code==4,tokens[...,1].long()-K,-1)
        parent=p.playing;mask=stage&b.active
        parent.actions[:,:width].copy_(t.where(mask[:,None,None],t.where(valid[...,None],choice_actions,-1),parent.actions[:,:width]))
        parent.payloads[:,:width].copy_(t.where(mask[:,None,None],t.where(valid[...,None],choice_payload,-1),parent.payloads[:,:width]))
        if width<A:
            parent.actions[:,width:].masked_fill_(mask[:,None,None],-1);parent.payloads[:,width:].masked_fill_(mask[:,None,None],-1)
        parent.counts.copy_(t.where(stage,t.where((errors==0)&b.active,counts,0),parent.counts));parent.errors.copy_(errors)
        p.selected.fill_(-1);p.top.fill_(-1);p.bottom.fill_(-1);p.bottom_plural.zero_()
        p.selected[:,:width].copy_(t.where((valid&((code==5)|(code==8)))[...,None],selected,-1))
        pos=t.arange(4,device=self.device)[None,None,:]
        top=selected[:,:,:4];bottom=selected.gather(2,(pos+tokens[...,2,None]).long().clamp(0,self.S-1))
        p.top[:,:width].copy_(t.where((valid&(code==6))[...,None]&(pos<tokens[...,2,None]),top,-1))
        p.bottom[:,:width].copy_(t.where((valid&(code==6))[...,None]&(pos<tokens[...,3,None]),bottom,-1))
        p.bottom_plural[:,:width].copy_(valid&(code==6)&(tokens[...,4]!=0))
        out.update(selected=p.selected,top=p.top,bottom=p.bottom,bottom_plural=p.bottom_plural,
                   other_choosing_phases_available=True,full_duel_qualified=False)
        return out

    def close(self):
        if not self.closed:self.choices.close();super().close()
