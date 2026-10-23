"""Device playing-phase action composition, excluding the concede action.

Other choosing phases require their separate enumerators. This read-only
composer does not execute rules, enable sampling/H3 or qualify full games.
"""
from dataclasses import dataclass
from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl.league_schema import OPTION_IDS
from app.modules.card_game.content.duel_v2 import CARDS
from .domain_soa_public_cards import DomainPublicCards
from .domain_soa_rule_queries import DomainRuleQueries
from .domain_soa_targets import DomainSoATargets,NONE,PLAYER,CHARACTER,DISCARD
from .domain_soa_observation import CI,CB,CS,PI,PB,PS,FI
from .domain_soa_schema import CHARACTER_VOCAB

INVALID,UNSUPPORTED,OVERFLOW=1,2,4
supports_full_duel_rules=False


@dataclass(frozen=True)
class PreparedPlayingActions:
    owner: object
    public: object
    rules: object
    rule_inputs: object
    target_queries: object
    target_inputs: object
    actions: object
    payloads: object
    counts: object
    errors: object


class DomainPlayingActions:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False,max_actions=800):
        self.public=DomainPublicCards(layout,device=device,allow_cpu_oracle=allow_cpu_oracle)
        self.t=self.public.t;self.device=self.public.device;self.layout=self.public.layout;self.closed=False
        if type(max_actions)is not int or not 1<=max_actions<=800:raise ValueError('Bounded complete action capacity required')
        self.max_actions=max_actions;self.H=min(10,self.layout.max_zone);self.C=self.layout.max_characters;self.T=2*self.C+1
        self.rules=DomainRuleQueries(self.layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
        self.targets=DomainSoATargets(self.layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
        t=self.t;O=max(1,max(len(CARDS[c].get('play_options')or[])for c in seven.CARD_IDS));self.O=O
        option_ids=[];costs=[]
        for cid in seven.CARD_IDS:
            ids=[];amount=[]
            for opt in CARDS[cid].get('play_options')or[]:
                if set((opt.get('costs')or{})) - {'jingu'}:raise RuntimeError('Unimplemented public option resource')
                ids.append(OPTION_IDS.index(opt['id']));amount.append(int((opt.get('costs')or{}).get('jingu',0)))
            option_ids.append(ids+[-1]*(O-len(ids)));costs.append(amount+[0]*(O-len(ids)))
        self.option_ids=t.tensor(option_ids,dtype=t.int64,device=self.device);self.option_costs=t.tensor(costs,dtype=t.int64,device=self.device)
        for text in ('U06','Q07','鬼郎丸'):
            self.public.observation.literals[text]=t.tensor(list(text.encode()),dtype=t.uint8,device=self.device)

    def prepare(self,fields,viewers,active):
        t=self.t
        if self.closed:raise RuntimeError('Playing composer closed')
        pub=self.public.prepare(fields,viewers,active);B=viewers.shape[0];N=self.C+self.H
        ri={'subjects':t.zeros((B,N),dtype=t.int64,device=self.device),
            'properties':t.cat((t.full((B,self.C),6,dtype=t.int32,device=self.device),t.full((B,self.H),5,dtype=t.int32,device=self.device)),1),
            'bases':t.zeros((B,N),dtype=t.float64,device=self.device),'base_present':t.ones((B,N),dtype=t.bool,device=self.device),
            'active':t.zeros((B,N),dtype=t.bool,device=self.device)}
        rule=self.rules.prepare(pub.observation.fields,**ri)
        ti={'sides':t.zeros((B,N),dtype=t.int64,device=self.device),'card_kinds':t.zeros((B,N),dtype=t.int64,device=self.device),
            'owners':t.zeros((B,N),dtype=t.int64,device=self.device),'is_battle':t.zeros((B,N),dtype=t.bool,device=self.device),
            'active':t.zeros((B,N),dtype=t.bool,device=self.device),'prior_errors':t.zeros((B,N),dtype=t.int32,device=self.device),
            'normal_attacks':t.cat((t.ones((B,self.C),dtype=t.bool,device=self.device),t.zeros((B,self.H),dtype=t.bool,device=self.device)),1)}
        target=self.targets.prepare(pub.observation.fields,**ti)
        return PreparedPlayingActions(self,pub,rule,ri,target,ti,
            t.empty((B,self.max_actions,6),dtype=t.int64,device=self.device),t.empty((B,self.max_actions,7),dtype=t.int64,device=self.device),t.empty(B,dtype=t.int64,device=self.device),
            t.empty(B,dtype=t.int32,device=self.device))

    def launch(self,p):
        if self.closed or not isinstance(p,PreparedPlayingActions)or p.owner is not self:raise ValueError('Closed or foreign playing buffers')
        t=self.t;pub=self.public.launch(p.public,_include_rules=True);b=p.public.observation;f=b.fields;B=b.viewers.shape[0];C,H,T,O=self.C,self.H,self.T,self.O
        rows=t.arange(B,device=self.device);own=b.viewers.clamp(0,1);cp=t.arange(C,device=self.device)[None,:];hp=t.arange(H,device=self.device)[None,:]
        prule={k:v[:,:H]for k,v in pub['_hand_rules'].items()};hand_refs=pub['hand_refs'][:,:H]
        def ownv(v):return v[rows,own]
        def cint(k):
            j=CI[k];pr=f['character_int_presence'][...,j];tm=f['character_template'].long().clamp(0,self.public.observation.char_catalog.shape[0]-1)
            return ownv(t.where(pr==3,self.public.observation.char_catalog[tm,j],t.where(pr==2,f['character_int_value'][...,j],0)))
        def pb(k):j=PB[k];return ownv((f['player_bool_presence'][...,j]==2)&f['player_bool_value'][...,j])
        def pi(k):j=PI[k];return ownv(t.where(f['player_int_presence'][...,j]==2,f['player_int_value'][...,j],0))
        acting=f['active_side'].long()==own;playing=(f['phase']==1)&acting&b.active
        errors=pub['errors'].clone()
        # Other phases with no network action can be answered exactly; actors
        # choosing or taking a mulligan must use the separate complete stages.
        choice_actor=f['pending_choice_present']&(f['choice_side'].long()==own)
        unsupported=b.active&(((f['phase']==0))|((f['phase']==2)&choice_actor))
        errors|=unsupported.int()*UNSUPPORTED
        hand_count=ownv(f['zone_count'][:,:,0]);errors|=(playing&(hand_count>H)).int()*OVERFLOW
        hpraw=ownv(f['character_hp']);summoned=ownv((f['character_bool_presence'][...,CB['summoned']]==2)&f['character_bool_value'][...,CB['summoned']])
        awake=ownv((f['character_bool_presence'][...,CB['awakened']]==2)&f['character_bool_value'][...,CB['awakened']])
        stored=ownv(f['player_character_count']).long();declared=ownv(f['player_order_count']).long()
        order=t.where(declared[:,None]==0,cp,ownv(f['player_order_slot']).long());on=cp<t.where(declared==0,stored,declared)[:,None]
        os=order.clamp(0,C-1);selected=t.zeros((B,C),dtype=t.int32,device=self.device);selected.scatter_add_(1,os,on.int());selected=selected>0
        tm=ownv(f['character_template']).long();seat=self.public.observation.partial.catalog.character_to_seat[tm.clamp(0,self.public.observation.partial.catalog.character_to_seat.shape[0]-1)].long()
        chvalid=ownv(f['character_valid']);alive=chvalid&(hpraw>0)
        holder=own[:,None]*(C+1)+1+cp
        collapse=f['flag_collapse_presence'].gather(1,holder)==2
        mimic=f['flag_mimic_template'].gather(1,holder).long();wrapped=(f['flag_mimic_presence'].gather(1,holder)==2)&(f['flag_mimic_id_presence'].gather(1,holder)==2)&(mimic>0)
        kit=t.where(wrapped,mimic,tm)
        match,mbad=self.public.observation._eq(f,f['player_used_key_slot'],'instant')
        uv=t.where(f['player_used_value_kind']==1,f['player_used_int'],f['player_used_value'].int())
        used=ownv(((t.arange(self.layout.max_used,device=self.device)[None,None,:]<f['player_used_count'][...,None])&match&(uv!=0)).any(-1))
        turn=f['turn'];ap=ownv(f['player_ap'])
        shape_slot=ownv(f['character_str_slot'][...,CS['shape']]);shape_pr=ownv(f['character_str_presence'][...,CS['shape']])
        u06,_=self.public.observation._eq(f,shape_slot,'U06');q07,_=self.public.observation._eq(f,shape_slot,'Q07')
        u06&=shape_pr==2;q07&=shape_pr==2
        other_attacked=(ownv(f['player_attacked_this_turn'])[:,None,:]&(cp[:,None,:]!=cp[:,:,None])).any(-1)
        offered=pb('normal_attack_available')[:,None].expand(-1,C).clone();price=t.ones((B,C),dtype=t.int64,device=self.device)
        hy=alive&(kit==CHARACTER_VOCAB['haiyue'])&u06&~used[:,None]&other_attacked
        xz=alive&(kit==CHARACTER_VOCAB['xiaozhi'])&q07&~offered&(cint('jingu')>=3)
        offered|=hy|xz;price=t.where(hy|xz,0,price)
        ri=p.rule_inputs;ri['subjects'][:,:C].copy_(holder);ri['bases'][:,:C].zero_()
        hand_slot=prule['owner_slot'].clamp(0,C-1);hand_valid=(hp<hand_count[:,None])&(hand_refs>=0)
        ri['subjects'][:,C:].copy_(own[:,None]*(C+1)+1+hand_slot)
        base=t.where(prule['spend_all_ap']!=0,t.minimum(ap[:,None].long(),prule['spend_all_ap']),prule['cost'])
        ri['bases'][:,C:].copy_(base.double())
        ri['active'][:,:C].copy_(playing[:,None]&selected&alive&~summoned)
        ri['active'][:,C:].copy_(playing[:,None]&hand_valid)
        qr=self.rules.launch(p.rules);errors|=qr['errors'].amax(-1)
        instant=(qr['values'][:,:C]!=0)&~used[:,None];offered|=instant;price=t.where(instant,0,price)
        free=(f['flag_int_presence'][:,:,FI['free_normal_attack_turn']].gather(1,holder)>=2)&(f['flag_int_value'][:,:,FI['free_normal_attack_turn']].gather(1,holder)==turn[:,None])
        price=t.where(free&offered,0,price)
        seal_slot=ownv(f['player_str_slot'][...,PS['seal_normal_attack']]).long().clamp(0,self.layout.max_string_slots-1)
        seal=(ownv(f['player_str_presence'][...,PS['seal_normal_attack']])==2)&(f['string_length'].gather(1,seal_slot[:,None]).squeeze(-1)>0)
        attack=playing[:,None]&selected&alive&~summoned&~collapse&offered&(ap[:,None]>=price)&~seal[:,None]
        remaining=t.where(pb('ultimate_available'),(1+(f['escalation_enabled']&(turn>=6)).int()-pi('ultimates_used')).clamp(min=0),0)
        exit_ult=alive&~summoned&(tm==CHARACTER_VOCAB['heiyu'])&(kit==CHARACTER_VOCAB['heiyu'])&awake&~wrapped
        used_ult=(ownv(f['character_int_presence'][...,CI['ultimate_used_turn']])>=2)&(cint('ultimate_used_turn')==turn[:,None])
        cap=pub['_character_queries'][rows,own,:,3]
        ultimate=alive&~summoned&(remaining[:,None]>0)&(ownv(f['character_energy'])>=cap)&~used_ult&~exit_ult
        ultimate=(ultimate|exit_ult)&selected&playing[:,None]
        cost=qr['values'][:,C:].trunc().clamp(min=0).long()
        ordinary=t.where((cost>0)&prule['plays_instant']&~used[:,None],0,cost)
        surcharge=pub['cards'].gather(1,hand_refs.clamp(0,pub['cards'].shape[1]-1)[...,None].expand(-1,-1,13))[...,12].long()
        payment=ordinary+surcharge
        owner_hp=hpraw.gather(1,hand_slot);owner_collapse=collapse.gather(1,hand_slot);owner_kit=kit.gather(1,hand_slot)
        req=prule['require'];burnmask=f['zone_status_i32'][:,:,1,7]
        enemy=1-own
        own_burn=(f['zone_status_presence'][rows,enemy,1]==2)&((burnmask[rows,enemy]&(1<<1))!=0)&(f['zone_status_side'][rows,enemy,1,0].long()==own)
        require_ok=((req!=1)|pb('harmony_damage')[:,None])&((req!=2)|pb('surplus')[:,None])&((req!=3)|own_burn[:,None])
        delay=(f['zone_status_presence'][:,:,0]==2)&(f['zone_status_i32'][:,:,0,7]!=0)
        enemy_delay=delay[rows,1-own]
        ghost,gbad=self.public.observation._eq(f,f['character_str_slot'][...,CS['name']],'鬼郎丸')
        gulang=(ownv(ghost)&(ownv(f['character_str_presence'][...,CS['name']])==2)&selected&alive&summoned).any(-1)
        can_play=~((owner_hp>0)&(owner_kit==CHARACTER_VOCAB['yi'])&(prule['kind']==seven.CARD_INDEX['I05'])&~enemy_delay[:,None])
        can_play&=~((owner_hp>0)&(owner_kit==CHARACTER_VOCAB['zaowu'])&(prule['kind']==seven.CARD_INDEX['S06'])&~gulang[:,None])
        play=playing[:,None]&hand_valid&((owner_hp>0)|prule['playable_downed'])&~owner_collapse&require_ok&can_play&(ap[:,None]>=payment)&((prule['spend_all_ap']==0)|(ap[:,None]>0))
        opt=self.option_ids[prule['kind'].clamp(0,len(seven.CARD_IDS)-1)];opt_cost=self.option_costs[prule['kind'].clamp(0,len(seven.CARD_IDS)-1)]
        valid_options=(opt>=0)&(cint('jingu').gather(1,hand_slot)[...,None]>=opt_cost)
        no_options=~prule['options'];valid_options=t.where(no_options[...,None],t.arange(O,device=self.device)==0,valid_options)
        opt=t.where(no_options[...,None],-1,opt)
        ti=p.target_inputs;ti['sides'].copy_(own[:,None].expand(-1,C+H));ti['card_kinds'][:,:C].zero_();ti['card_kinds'][:,C:].copy_(prule['kind'])
        ti['owners'][:,:C].copy_(seat);ti['owners'][:,C:].copy_(prule['owner']);ti['is_battle'][:,:C].zero_();ti['is_battle'][:,C:].copy_(prule['is_battle'])
        ti['active'][:,:C].copy_(attack);ti['active'][:,C:].copy_(play&valid_options.any(-1));ti['prior_errors'].copy_(errors[:,None].expand(-1,C+H))
        target=self.targets.launch(p.target_queries);errors|=target['errors'].amax(-1)
        tc=target['counts'];desc=target['targets'][:,:,:T]
        errors|=((ti['active']&(tc>T)).any(-1)).int()*OVERFLOW
        errors|=((desc[...,0]==DISCARD)&(t.arange(T,device=self.device)[None,None,:]<tc[...,None])&ti['active'][...,None]).any((1,2)).int()*UNSUPPORTED
        # Translate stored targets to the frozen model seat, keeping physical
        # target slots in the separate target-query receipt for future execution.
        target_side=desc[...,1].clamp(0,1);target_slot=desc[...,2].clamp(0,C-1)
        tm_all=f['character_template'].long().clamp(0,self.public.observation.partial.catalog.character_to_seat.shape[0]-1)
        target_seat=self.public.observation.partial.catalog.character_to_seat[tm_all[rows[:,None,None],target_side,target_slot]].long()
        all_summoned=(f['character_bool_presence'][...,CB['summoned']]==2)&f['character_bool_value'][...,CB['summoned']]
        target_seat=t.where(all_summoned[rows[:,None,None],target_side,target_slot],seven.SEAT_INDEX['summon'],target_seat)
        ts=t.where(desc[...,0]==NONE,-1,desc[...,1]);target_seat=t.where(desc[...,0]==CHARACTER,target_seat,-1)
        header=t.full((B,C,T+1,6),-1,dtype=t.int64,device=self.device);header[:,:,:T,0]=1;header[:,:,:T,1]=seat[...,None]
        header[:,:,:T,3]=ts[:,:C];header[:,:,:T,4]=target_seat[:,:C]
        header[:,:,T,0]=t.where(exit_ult,7,2);header[:,:,T,1]=seat
        headmask=t.cat((attack[...,None]&(t.arange(T,device=self.device)[None,None,:]<tc[:,:C,None]),ultimate[...,None]),-1)
        header_payload=t.full((B,C,T+1,7),-1,dtype=t.int64,device=self.device)
        header_payload[...,0]=header[...,0];header_payload[...,1]=cp[...,None]
        header_payload[:,:,:T,3:6]=desc[:,:C];header_payload[:,:,T,3]=NONE
        header_payload=header_payload.gather(1,os[...,None,None].expand(-1,-1,T+1,7))
        header=header.gather(1,os[...,None,None].expand(-1,-1,T+1,6));headmask=headmask.gather(1,os[...,None].expand(-1,-1,T+1))&on[...,None]
        cards=t.full((B,H,T,O,6),-1,dtype=t.int64,device=self.device);cards[...,0]=3;cards[...,2]=hand_refs[:,:,None,None]
        cards[...,3]=ts[:,C:,:,None];cards[...,4]=target_seat[:,C:,:,None];cards[...,5]=opt[:,:,None,:]
        cardmask=play[:,:,None,None]&(t.arange(T,device=self.device)[None,None,:,None]<tc[:,C:,None,None])&valid_options[:,:,None,:]
        card_payload=t.full((B,H,T,O,7),-1,dtype=t.int64,device=self.device)
        card_payload[...,0]=3;card_payload[...,2]=hand_refs[:,:,None,None];card_payload[...,3:6]=desc[:,C:,:,None,:]
        card_payload[...,6]=opt[:,:,None,:]
        end_payload=t.full((B,1,7),-1,dtype=t.int64,device=self.device);end_payload[:,:,0]=0;end_payload[:,:,3]=NONE
        raw_payload=t.cat((end_payload,header_payload.flatten(1,2),card_payload.flatten(1,3)),1)
        end=t.full((B,1,6),-1,dtype=t.int64,device=self.device);end[:,:,0]=0
        raw=t.cat((end,header.flatten(1,2),cards.flatten(1,3)),1);keep=t.cat((playing[:,None],headmask.flatten(1),cardmask.flatten(1)),1)
        count=keep.sum(-1);errors|=(count>self.max_actions).int()*OVERFLOW
        errors=t.where(pub['errors']!=0,pub['errors'],errors)
        keep&=(errors==0)[:,None];count=keep.sum(-1)
        index=t.arange(raw.shape[1],device=self.device)[None,:];perm=t.where(keep,index,index+raw.shape[1]).argsort(dim=-1,stable=True)[:,:self.max_actions]
        selected_actions=raw.gather(1,perm[...,None].expand(-1,-1,6));width=selected_actions.shape[1]
        p.actions.fill_(-1);p.actions[:,:width].copy_(t.where((t.arange(width,device=self.device)[None,:]<count[:,None])[...,None],selected_actions,-1))
        selected_payload=raw_payload.gather(1,perm[...,None].expand(-1,-1,7))
        p.payloads.fill_(-1);p.payloads[:,:width].copy_(t.where((t.arange(width,device=self.device)[None,:]<count[:,None])[...,None],selected_payload,-1))
        p.counts.copy_(count);p.errors.copy_(errors)
        return {'actions':p.actions,'payloads':p.payloads,'counts':p.counts,'errors':p.errors,'public_cards':pub['cards'],'observation':pub['observation'],'state_versions':f['version'],
                'full_duel_qualified':False,'other_choosing_phases_available':False,'physical_rules_available':False}

    def project(self,fields,viewers,active):return self.launch(self.prepare(fields,viewers,active))
    def close(self):
        if not self.closed:self.public.close();self.rules.close();self.targets.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
