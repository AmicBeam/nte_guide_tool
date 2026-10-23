"""Seven-model observation assembly from named device fields and public queries.

All declared columns are computed. This is not a legal-action encoder, physical
rule executor, information-set qualification or a complete-game backend.
"""
from dataclasses import dataclass
from collections.abc import Mapping

from app.modules.card_game.rl import seven_lineup as seven
from app.modules.card_game.rl import league_schema as league
from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.catalog import CHARACTER_ORDER
from .domain_soa_schema import (
    DomainSoALayout, array_specs, CHARACTER_INT_FIELDS, CHARACTER_BOOL_FIELDS,
    CHARACTER_STR_FIELDS, CARD_BOOL_FIELDS, CARD_STR_FIELDS, CHARACTER_VOCAB,
    CARD_VOCAB, PLAYER_INT_FIELDS, PLAYER_BOOL_FIELDS, PLAYER_STR_FIELDS,
    FLAG_INT_FIELDS, CHOICE_KINDS, CHOICE_BOOL_KEYS, DOT_KINDS,
)
from .domain_soa_projection import DomainSoAProjection
from .domain_soa_character_queries import DomainCharacterQueries, OUTPUT_NAMES

supports_full_duel_rules = False
INVALID = 1
UNSUPPORTED = 2
FEATURE_NAMES = tuple(seven.all_features())
CI={k:i for i,k in enumerate(CHARACTER_INT_FIELDS)}
CB={k:i for i,k in enumerate(CHARACTER_BOOL_FIELDS)}
CS={k:i for i,k in enumerate(CHARACTER_STR_FIELDS)}
PI={k:i for i,k in enumerate(PLAYER_INT_FIELDS)}
PB={k:i for i,k in enumerate(PLAYER_BOOL_FIELDS)}
PS={k:i for i,k in enumerate(PLAYER_STR_FIELDS)}
KB={k:i for i,k in enumerate(CARD_BOOL_FIELDS)}
KS={k:i for i,k in enumerate(CARD_STR_FIELDS)}
FI={k:i for i,k in enumerate(FLAG_INT_FIELDS)}
QV={k:i for i,k in enumerate(OUTPUT_NAMES)}
ZONE_KEYS=(('left','tick_side','slow','slow_turn','by','expires_side'),
           ('left','by','stacks','source','applied_turn','infinite'),
           ('left','by','source_cid','renewals'),('by','expires_side'),('by','expires_side','ends_left'))
ZONE_INTS={'left','slow','slow_turn','stacks','applied_turn','renewals','ends_left'}
ZONE_SIDES={'tick_side','by','expires_side'}


@dataclass(frozen=True)
class PreparedObservation:
    owner: object
    sources: object
    fields: object
    viewer_source: object
    viewers: object
    active_source: object
    active: object
    queries: object
    features: object
    errors: object


class DomainSoAObservation:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False):
        import torch
        self.t=torch;self.layout=layout or DomainSoALayout();self.device=torch.device(device)
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no observation CPU fallback')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self.partial=DomainSoAProjection(self.layout,device=self.device,max_actions=1,allow_cpu_oracle=allow_cpu_oracle)
        self.queries=DomainCharacterQueries(self.layout,device=self.device,allow_cpu_oracle=allow_cpu_oracle)
        self.specs=array_specs(self.layout);self.closed=False
        self.literals={}
        for text in ('instant','canhong.mirage','canhong_dot_attack_turn:e'):
            self.literals[text]=torch.tensor(list(text.encode('utf8')),dtype=torch.uint8,device=self.device)
        self.lookups={}
        for key,labels in (('seats',seven.SEATS),('cards',seven.CARD_IDS)):
            width=max(map(len,labels))
            if width>8 or any(not x.isascii()for x in labels):raise ValueError('Frozen ASCII identity table exceeds 64 bits')
            self.lookups[key]=(width,torch.tensor([int.from_bytes(x.encode(),'little')for x in labels],dtype=torch.int64,device=self.device),
                               torch.tensor(list(map(len,labels)),dtype=torch.int64,device=self.device),
                               torch.tensor([1<<(8*j)for j in range(width)],dtype=torch.int64,device=self.device))
        n=max(CARD_VOCAB.values())+1
        self.card_seat=torch.full((n,),-1,dtype=torch.int64,device=self.device)
        for cid,i in CARD_VOCAB.items():self.card_seat[i]=seven.SEAT_INDEX[CARDS[cid]['character_id']]
        self.char_catalog=self.partial.catalog.character_int_defaults
        self.seq_digits=torch.arange(10,device=self.device)
        self.decimal_powers=torch.tensor([10**i for i in range(10)],dtype=torch.int64,device=self.device)

    def prepare(self,fields,viewers,active):
        t=self.t
        if self.closed:raise RuntimeError('Observation backend closed')
        if self.device.type=='cuda'and t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare observation buffers outside capture')
        cols=fields if isinstance(fields,Mapping)else getattr(fields,'arrays',None)
        if not isinstance(cols,Mapping):raise TypeError('Named device field mapping required')
        if not isinstance(viewers,t.Tensor)or viewers.device!=self.device or viewers.dtype!=t.int64 or viewers.ndim!=1:raise ValueError('Resident int64 viewers required')
        rows=viewers.shape[0]
        if not 1<=rows<=self.layout.game_capacity:raise ValueError('Observation batch bound exceeded')
        if not isinstance(active,t.Tensor)or active.device!=self.device or active.dtype!=t.bool or active.shape!=(rows,):raise ValueError('Resident bool active mask required')
        sources={};packed={}
        for name in self.specs:
            value=self.partial._column(cols,name,rows);sources[name]=value
            packed[name]=value.clone(memory_format=t.contiguous_format)
        vi=viewers.clone(memory_format=t.contiguous_format);ac=active.clone(memory_format=t.contiguous_format)
        return PreparedObservation(self,sources,packed,viewers,vi,active,ac,self.queries.prepare(packed,ac),
                                   t.empty((rows,len(FEATURE_NAMES)),dtype=t.float32,device=self.device),
                                   t.empty(rows,dtype=t.int32,device=self.device))

    def _bytes(self,f,slots,width):
        t=self.t;rows=slots.shape[0];tail=tuple(slots.shape[1:])
        safe=slots.long().clamp(0,self.layout.max_string_slots-1)
        offsets=f['string_offset'].gather(1,safe.flatten(1)).reshape((rows,)+tail).long()
        lengths=f['string_length'].gather(1,safe.flatten(1)).reshape((rows,)+tail).long()
        bad=(slots<0)|(slots>=f['string_count'].reshape((rows,)+(1,)*len(tail)))|(offsets<0)|(lengths<0)|((offsets+lengths)>self.layout.max_string_bytes)|((offsets+lengths)>f['string_bytes_used'].reshape((rows,)+(1,)*len(tail)))
        pos=offsets[...,None]+t.arange(width,device=self.device)
        raw=f['string_bytes'].gather(1,pos.clamp(0,self.layout.max_string_bytes-1).flatten(1)).reshape((rows,)+tail+(width,))
        raw=t.where(t.arange(width,device=self.device)<lengths[...,None],raw,t.zeros_like(raw))
        return raw,lengths,bad

    def _eq(self,f,slots,text):
        literal=self.literals[text];raw,lengths,bad=self._bytes(f,slots,literal.shape[0])
        return (raw==literal).all(-1)&(lengths==literal.shape[0])&~bad,bad

    def _lookup(self,f,slots,kind):
        width,codes,lens,powers=self.lookups[kind]
        raw,lengths,bad=self._bytes(f,slots,width)
        code=(raw.long()*powers).sum(-1)
        match=(code[...,None]==codes)&(lengths[...,None]==lens)&~bad[...,None]
        index=match.int().argmax(-1)
        return self.t.where(match.any(-1),index,self.t.full_like(index,-1)),bad,lengths

    def _ratio(self,v,d):return (v.to(self.t.float64)/d).to(self.t.float32)

    def launch(self,b,*,_include_private=False):
        if type(_include_private)is not bool:raise TypeError('Internal projection flag must be a static bool')
        t=self.t
        if self.closed:raise RuntimeError('Observation backend closed')
        if not isinstance(b,PreparedObservation)or b.owner is not self:raise ValueError('Foreign observation buffers')
        for name,value in b.fields.items():value.copy_(b.sources[name])
        b.viewers.copy_(b.viewer_source);b.active.copy_(b.active_source)
        f=b.fields;rows=int(b.viewers.shape[0]);C=self.layout.max_characters;K=self.layout.max_cards;M=len(seven.SEATS)
        q=self.queries.launch(b.queries)
        prefix=self.partial.project(f,b.viewers,b.active,observation_only=True)
        errors=q['errors']|prefix.errors
        errors=errors|((b.viewers<0)|(b.viewers>1)|(f['phase']<0)|(f['phase']>3)|(f['active_side']<0)|(f['active_side']>1)|(f['first_side']<0)|(f['first_side']>1)).int()*INVALID
        batch=t.arange(rows,device=self.device);own=b.viewers.clamp(0,1);sides=t.stack((own,1-own),1)
        def g(x):return self.partial._gather_sides(x,sides)
        def pi(k):return t.where(f['player_int_presence'][:,:,PI[k]]==2,f['player_int_value'][:,:,PI[k]],0)
        def pb(k):return (f['player_bool_presence'][:,:,PB[k]]==2)&f['player_bool_value'][:,:,PB[k]]
        pos=t.arange(C,device=self.device)[None,None,:]
        stored=f['player_character_count'].long();declared=f['player_order_count'].long()
        order=t.where((declared==0)[:,:,None],pos,f['player_order_slot'].long())
        order_n=t.where(declared==0,stored,declared)
        valid_order=pos<order_n[:,:,None]
        selected=(t.nn.functional.one_hot(order.clamp(0,C-1),C).bool()&valid_order[:,:,:,None]).any(2)
        live=f['character_valid']&selected
        tm=f['character_template'].long();safe_tm=tm.clamp(0,self.partial.catalog.character_to_seat.shape[0]-1)
        cbp=f['character_bool_presence'];cbv=f['character_bool_value']
        summoned=(cbp[...,CB['summoned']]==2)&cbv[...,CB['summoned']]
        seats=t.where(summoned,t.full_like(tm,seven.SEAT_INDEX['summon']),self.partial.catalog.character_to_seat[safe_tm].long())
        hit=t.nn.functional.one_hot(seats.clamp(0,M-1),M).bool()&live[...,None]
        present=hit.any(2)
        def sc(v,default=0):
            out=(v.to(t.float64)[...,None]*hit).sum(2)
            return t.where(present,out,t.full_like(out,default))
        def cint(k):
            i=CI[k];p=f['character_int_presence'][...,i]
            return t.where(p==3,self.char_catalog[safe_tm,i],t.where(p==2,f['character_int_value'][...,i],0))
        def has_cint(k):return f['character_int_presence'][...,CI[k]]>=2
        hp=f['character_hp'];awake=(cbp[...,CB['awakened']]==2)&cbv[...,CB['awakened']]
        hi=t.arange(2*(C+1),device=self.device).reshape(2,C+1)[:,1:]
        fp=f['flag_int_presence'][:,hi];fv=f['flag_int_value'][:,hi]
        def flag(k):return t.where(fp[...,FI[k]]>=2,fv[...,FI[k]],0)
        qv={k:q['values'][...,i]for i,k in enumerate(OUTPUT_NAMES)}
        # Shapes and physical authors are decoded only through frozen public IDs.
        shape,shape_bad,shape_len=self._lookup(f,f['character_str_slot'][...,CS['shape']],'cards')
        shape=t.where(f['character_str_presence'][...,CS['shape']]==2,shape,-1)
        errors=errors|((live&(f['character_str_presence'][...,CS['shape']]==2)&(shape_bad|(shape<0))).flatten(1).any(1)).int()*INVALID
        raw_kind=self.partial.catalog.card_to_seven[f['card_template'].long().clamp(0,self.partial.catalog.card_to_seven.shape[0]-1)].long()
        raw_owner=self.card_seat[f['card_template'].long().clamp(0,self.card_seat.shape[0]-1)]
        owner_override,owner_bad,_=self._lookup(f,f['card_str_slot'][...,KS['character_id']],'seats')
        override=f['card_str_presence'][...,KS['character_id']]==2
        owner=t.where(override,owner_override,raw_owner)
        errors=errors|((f['card_valid']&override&(owner_bad|(owner<0))).any(1)).int()*INVALID
        raw_owner=owner.clone()
        effective=raw_kind.clone()
        transformed=t.zeros_like(effective)
        # Dispatch order is the formal registry's fixed public character order.
        used_active=t.arange(self.layout.max_used,device=self.device)[None,None,:]<f['player_used_count'][:,:,None]
        instant_match,instant_bad=self._eq(f,f['player_used_key_slot'],'instant')
        used_num=t.where(f['player_used_value_kind']==1,f['player_used_int'],f['player_used_value'].int())
        instant=(used_active&instant_match&(used_num!=0)).any(2)
        errors=errors|((used_active&instant_bad).flatten(1).any(1)).int()*INVALID
        card_side=f['card_side'].long().clamp(0,1)
        for cid in CHARACTER_ORDER:
            if cid not in CHARACTER_VOCAB:continue
            role=(tm==CHARACTER_VOCAB[cid])&f['character_valid']&(hp>0)&(f['phase'][:,None,None]!=3)
            actor_seat=seven.SEAT_INDEX[cid]
            wrapped=(f['flag_mimic_presence'][:,hi]==2)&(f['flag_mimic_id_presence'][:,hi]==2)&(f['flag_mimic_template'][:,hi]>0)
            kit=t.where(wrapped,f['flag_mimic_template'][:,hi].long(),tm)
            rf=(role&(kit==CHARACTER_VOCAB['zhenhong'])&awake&~wrapped).any(2)
            rf_card=rf.gather(1,card_side)
            effective=t.where(rf_card,t.full_like(effective,seven.CARD_INDEX['RF01']),effective)
            owner=t.where(rf_card,t.full_like(owner,seven.SEAT_INDEX['zhenhong']),owner)
            transformed=t.where(rf_card,t.ones_like(transformed),transformed)
            lk=(role&(kit==CHARACTER_VOCAB['lingke'])&(shape==seven.CARD_INDEX['K03'])).any(2)&(f['player_ap']==0)&~instant
            lk_card=lk.gather(1,card_side)&(owner==actor_seat)
            effective=t.where(lk_card,t.full_like(effective,seven.CARD_INDEX['K03']),effective)
            owner=t.where(lk_card,t.full_like(owner,actor_seat),owner)
            transformed=t.where(lk_card,t.full_like(transformed,2),transformed)
        # Public hand visibility is physical-identity based, before face changes.
        rev_valid=(t.arange(K,device=self.device)[None,None,:]<f['player_revealed_count'][:,:,None])&(f['player_revealed_presence'][:,:,None]==2)
        revealed=((f['player_revealed_instance'][:,:,:,None]==f['card_instance_slot'][:,None,None,:])&
                  (f['player_revealed_side'][:,:,:,None]==f['card_side'][:,None,None,:])&rev_valid[:,:,:,None]).any(2)
        copy=(f['card_bool_presence'][...,KB['copy']]==2)&f['card_bool_value'][...,KB['copy']]
        zpos=t.arange(self.layout.max_zone,device=self.device)[None,:]
        def zone(side,which,kinds,extra=None):
            indices=f['zone_card_index'][batch,side,which].long().clamp(0,K-1)
            valid=zpos<f['zone_count'][batch,side,which,None]
            valid=valid&f['card_valid'].gather(1,indices)
            if extra is not None:valid=valid&extra.gather(1,indices)
            return indices,kinds.gather(1,indices),valid
        def bag(kinds,valid):
            return (t.nn.functional.one_hot(kinds.clamp(0,len(seven.CARD_IDS)-1),len(seven.CARD_IDS)).long()*valid[...,None]).sum(1)
        hand_idx,hand_kind,hand_valid=zone(own,0,effective)
        _,od,ov=zone(own,2,raw_kind);_,ed,ev=zone(1-own,2,raw_kind)
        _,eh,ehv=zone(1-own,0,effective,copy|revealed[batch,1-own])
        _,rh,rhv=zone(own,0,effective,revealed[batch,own])
        cards=[bag(hand_kind,hand_valid),bag(od,ov),bag(ed,ev),bag(eh,ehv),bag(rh,rhv)]
        delta=f['card_jingu_delta'].gather(1,hand_idx)
        marked=(f['card_jingu_mark_presence'].gather(1,hand_idx)==2)&hand_valid
        errors=errors|((marked&((delta< -1)|(delta>1))).any(1)).int()*INVALID
        marks=[bag(hand_kind,marked&(delta==d))for d in (-1,0,1)]
        physical=raw_kind.gather(1,hand_idx)
        substituted=bag(physical,hand_valid&(physical!=hand_kind))
        # Compact, explicit status layouts retain key presence through their mask.
        def zone_get(z,key):
            if key not in ZONE_KEYS[z]:return t.zeros((rows,2),dtype=t.int32,device=self.device)
            keys=ZONE_KEYS[z];bit=keys.index(key);mask=f['zone_status_i32'][:,:,z,7]
            has=(f['zone_status_presence'][:,:,z]==2)&((mask&(1<<bit))!=0)
            if key in ZONE_INTS:arr=f['zone_status_i32'][:,:,z,list(k for k in keys if k in ZONE_INTS).index(key)]
            elif key in ZONE_SIDES:arr=f['zone_status_side'][:,:,z,list(k for k in keys if k in ZONE_SIDES).index(key)]
            elif key=='infinite':arr=f['zone_status_bool'][:,:,z,0]
            else:raise ValueError('Non-numeric status requested')
            return t.where(has,arr,t.zeros_like(arr))
        def zone_has(z,key):
            if key not in ZONE_KEYS[z]:return t.zeros((rows,2),dtype=t.bool,device=self.device)
            return (f['zone_status_presence'][:,:,z]==2)&((f['zone_status_i32'][:,:,z,7]&(1<<ZONE_KEYS[z].index(key)))!=0)
        def zone_truth(z):return (f['zone_status_presence'][:,:,z]==2)&(f['zone_status_i32'][:,:,z,7]!=0)
        def zone_by(z):return t.where(zone_truth(z),t.where(zone_has(z,'by'),(zone_get(z,'by')!=t.arange(2,device=self.device)).int(),1),-1)
        extra_actor,extra_bad,_=self._lookup(f,f['player_str_slot'][:,:,PS['extra_genesis_actor']],'seats')
        extra_actor=t.where(f['player_str_presence'][:,:,PS['extra_genesis_actor']]==2,extra_actor,-1)
        pending=f['pending_choice_present'];kind=f['choice_kind'].long();cside=f['choice_side'].long().clamp(0,1)
        inspect=pending&((kind==CHOICE_KINDS.index('inspect_top'))|(kind==CHOICE_KINDS.index('heiyu_divination'))|
                        ((f['choice_bool_presence'][:,CHOICE_BOOL_KEYS.index('deck_cards')]==2)&f['choice_bool_value'][:,CHOICE_BOOL_KEYS.index('deck_cards')]))
        deck=f['zone_count'][:,:,1]+t.nn.functional.one_hot(cside,2).int()*(inspect*f['choice_card_count'])[:,None]
        remaining=t.where(pb('ultimate_available'),((1+(f['escalation_enabled']&(f['turn']>=6)).int())[:,None]-pi('ultimates_used')).clamp(min=0),0)
        side_values={
            'hp':f['player_hp'],'shield':f['player_shield'],'ap':f['player_ap'],
                        'turn_count':pi('turn_count'),'normal_atk':pb('normal_attack_available'),'ultimate_ok':pb('ultimate_available'),
            'used_instant':instant,'extra_genesis':pb('extra_genesis_pending'),'extra_genesis_actor':extra_actor,
            'nanali_seed':pi('nanali_family'),'nanali_seed_played':pi('nanali_family_played'),
            'hand_n':f['zone_count'][:,:,0],'deck_n':deck,'discard_n':f['zone_count'][:,:,2],
            'weave':zone_truth(3),'fatigue':pi('fatigue'),'extra_ap':pi('extra_ap'),'surplus':pb('surplus'),
            'ultimate_remaining':remaining,'burn_left':zone_get(1,'left'),'burn_infinite':zone_get(1,'infinite'),
            'delay_end':zone_get(0,'end'),'harmony_damage':pb('harmony_damage'),'burn_by':zone_by(1),
            'genesis_damage_bonus':pi('genesis_damage_bonus'),'light_resistance':q['player_resistances'][:,:,0],
            'dark_left':zone_get(2,'left'),'dark_by':zone_by(2),
        }
        # Front references in the prefix are already viewer-relative.
        relative_side={k:g(v)for k,v in side_values.items()}
        relative_side['front']=prefix.named['player_front_seat'];relative_side['last_front']=prefix.named['player_last_front_seat']
        cvalues={'ch_present':sc(live),'ch_hp':sc(hp),'ch_max_hp':sc(cint('max_hp')),
                 'ch_base_atk':sc(qv['base_attack']),'ch_growth':sc(cint('growth')),'ch_atk_buff':sc(qv['atk_buff']),
                 'ch_shield':sc(f['character_shield']),'ch_harmony':sc(cint('harmony')),'ch_energy':sc(f['character_energy']),
                 'ch_down':sc(cint('down_turns')),'ch_awakened':sc(awake),'ch_ult_turns':sc(cint('ultimate_turns')),
                 'ch_shape':sc(shape,-1),'ch_harmonized':sc(f['player_harmonized']),
                 'ch_collapse_count':sc(flag('collapse_count')),
                 'ch_collapse_until':sc(t.where(f['flag_collapse_presence'][:,hi]==2,f['flag_collapse_until'][:,hi],0)),
                 'ch_base_max':sc(qv['base_max_hp']),'ch_energy_max':sc(qv['energy_max']),
                 'ch_harmony_max':sc(qv['harmony_max']),
                 'ch_collapse_by':sc(t.where(f['flag_collapse_presence'][:,hi]==2,(f['flag_collapse_side'][:,hi]!=t.arange(2,device=self.device)[None,:,None]).int(),-1),-1),
                 'ch_jingu':sc(cint('jingu')),'ch_jingu_loan':sc((cbp[...,CB['jingu_loan']]==2)&cbv[...,CB['jingu_loan']]),
                 'ch_ultimate_used':sc(has_cint('ultimate_used_turn')&(cint('ultimate_used_turn')==f['turn'][:,None,None])),
                 'ch_attacked':sc(f['player_attacked_this_turn']),'ch_light_resistance':sc(qv['resistance_光'])}
        for name in ('next_bonus','next_shield','next_followup','pact','allied_hurt','energy_next','pending_atk','share_overflow','hp_floor'):
            cvalues['ch_'+name]=sc(flag(name))
        relative_char={k:g(v)for k,v in cvalues.items()};relative_char['ch_order']=prefix.named['char_order']
        fixed_s={'delay_left':zone_get(0,'left'),'delay_by':zone_by(0),
                 'delay_slow':t.where(zone_has(0,'slow_turn')&(zone_get(0,'slow_turn')==f['turn'][:,None]),zone_get(0,'slow'),0),
                 'delay_turns':pi('delay_turns'),'skip_normal_attack':pb('skip_normal_attack'),
                 'genesis_this_turn':(f['player_int_presence'][:,:,PI['genesis_turn']]>=2)&(pi('genesis_turn')==f['turn'][:,None]),
                 'genesis_player_hits':pi('genesis_player_hits')}
        fixed_c={k:sc(cint(k))for k in ('beast_fangs','intimidation','extra_attacks')}
        for key in ('attack','timed_attack_1','timed_attack_2','damage_immune','damage_limit'):fixed_c[key]=sc(qv[key])
        dots=[]
        for kind_name in DOT_KINDS:
            i=DOT_KINDS.index(kind_name)
            # Schema preserves the explicit kind tag; entries need not be positional.
            dots.append(t.where(f['zone_dot_valid']&(f['zone_dot_kind']==i),f['zone_dot_stacks'],0).sum(2))
        murk_zone=t.stack((zone_get(1,'stacks'),*dots),dim=-1)
        defmatch,bad_def=self._eq(f,f['effect_definition_slot'],'canhong.mirage')
        effect_active=t.arange(self.layout.max_effects,device=self.device)[None,None,:]<f['holder_effect_count'][:,:,None]
        mirage_h=(defmatch&effect_active&f['effect_valid']&(f['effect_visibility']==0)).any(2)[:,hi]
        mirage=(mirage_h&live&(tm==CHARACTER_VOCAB['canhong'])).any(2)
        # Resolve the actual persistent entity key used by the canhong clock.
        prefix_bytes=self.literals['canhong_dot_attack_turn:e'];W=int(prefix_bytes.shape[0])
        raw,strlen,key_bad=self._bytes(f,f['player_used_key_slot'],W+10)
        n=strlen-W;digits=raw[...,W:].long()-48;okdigits=((digits>=0)&(digits<=9))|(self.seq_digits>=n[...,None])
        powers=self.decimal_powers[(n[...,None]-1-self.seq_digits).clamp(0,9)]
        number=(digits.clamp(0,9)*powers*(self.seq_digits<n[...,None])).sum(-1)
        key_ok=(raw[...,:W]==prefix_bytes).all(-1)&(n>0)&(n<=10)&okdigits.all(-1)&((n==1)|(digits[...,0]!=0))&~key_bad
        canhong_eid=t.where(live&(tm==CHARACTER_VOCAB['canhong']),f['character_entity_slot'],0).sum(2)
        dot_used=(used_active&key_ok&(number==canhong_eid[:,:,None])&(used_num==f['turn'][:,None,None])).any(2)&(canhong_eid>0)
        support=t.where((f['player_int_presence'][:,:,PI['support_turn']]>=2)&(pi('support_turn')==f['turn'][:,None]),pi('support_count'),0)
        seven_s=t.stack((f['player_max_hp'],2+pi('insight_bonus'),pi('insight_played'),support,pi('heiyu_divinations')),dim=-1)
        chain=(f['character_lingke_presence']==2)
        chain_actor,chain_bad,chain_len=self._lookup(f,f['character_lingke_id_slot'],'seats')
        errors=errors|((live&chain&(chain_bad|((chain_len>0)&(chain_actor<0)))).flatten(1).any(1)).int()*INVALID
        mimic=f['flag_mimic_template'][:,hi].long().clamp(0,self.partial.catalog.character_to_seat.shape[0]-1)
        mimic_actor=self.partial.catalog.character_to_seat[mimic].long()
        mimic_on=(f['flag_mimic_presence'][:,hi]==2)&(f['flag_mimic_id_presence'][:,hi]==2)&(f['flag_mimic_template'][:,hi]>0)
        seven_c=t.stack((self._ratio(sc(qv['collapse_max']),10),self._ratio(sc(t.where(chain,f['character_lingke_count'],0)),10),
                         self._ratio(sc(t.where(chain&(chain_len>0),chain_actor+1,0)),20),
                         sc(has_cint('insight_used_turn')&(cint('insight_used_turn')==f['turn'][:,None,None])).float(),
                         self._ratio(sc(t.where(mimic_on,mimic_actor+1,0)),20)),dim=-1)
        # The frozen seven tail uses original public IDs; generated summon IDs
        # have no stable 'summon' entry there (the legacy block does adapt them).
        tail_seat=(t.arange(M,device=self.device)!=seven.SEAT_INDEX['summon'])
        seven_c=seven_c*tail_seat[None,None,:,None]
        char_resist=t.stack([g(sc(qv['resistance_'+a]))for a in seven.ATTRIBUTES],dim=-1)*tail_seat[None,None,:,None]
        choices=t.stack([pending&(kind==CHOICE_KINDS.index(name))for name in seven.CHOICES],dim=-1)
        chidx=f['choice_card_index'].long().clamp(0,K-1)
        chvalid=(t.arange(self.layout.max_choice_cards,device=self.device)[None,:]<f['choice_card_count'][:,None])&pending[:,None]&(f['choice_side'].long()==own)[:,None]
        choicebag=bag(raw_kind.gather(1,chidx),chvalid)
        header=t.stack((self._ratio(f['turn'],30),(own==f['first_side']).float(),f['escalation_enabled'].float(),
                        self._ratio(f['phase'],10),self._ratio(prefix.named['choice_kind'],10),
                        self._ratio(prefix.named['resolving'],len(league.OLD_CARDS)),self._ratio(prefix.named['choice_count'],10)),dim=-1)
        parts=[header,
               self._ratio(t.stack([relative_side[k]for k in league.SIDE_FIELDS],dim=1).flatten(1),10),
               self._ratio(t.stack([relative_char[k]for k in league.CHAR_FIELDS],dim=1).flatten(1),10),
               *[self._ratio(v,2)for v in cards],self._ratio(t.stack(marks,1).flatten(1),2),
               self._ratio(t.stack([g(fixed_s[k])for k in seven.EXTRA_SIDE],dim=1).flatten(1),10),
               self._ratio(t.stack([g(fixed_c[k])for k in seven.EXTRA_CHAR],dim=1).flatten(1),10),
               self._ratio(g(murk_zone).flatten(1),10),g(mirage).float(),g(dot_used).float(),self._ratio(substituted,2),
               self._ratio(g(seven_s).flatten(1),10),g(seven_c).flatten(1),self._ratio(g(q['player_resistances']).flatten(1),10),
               self._ratio(char_resist.flatten(1),10),choices.float(),self._ratio(choicebag,2)]
        features=t.cat(parts,dim=1)
        if features.shape!=(rows,len(FEATURE_NAMES)):raise RuntimeError('Full observation layout mismatch')
        errors=errors|(~t.isfinite(features).all(1)).int()*INVALID
        errors=t.where(b.active,errors,f['row_error'])
        errors=t.where(f['row_error']!=0,f['row_error'],errors)
        b.errors.copy_(errors)
        b.features.copy_(t.where((b.active&(errors==0))[:,None],features,t.zeros_like(features)))
        result={'features':b.features,'errors':b.errors,'feature_names':FEATURE_NAMES,
                'full_duel_qualified':False,'action_features_available':False}
        if _include_private:
            result['_private_card_fields']={'raw_kind':raw_kind,'raw_owner':raw_owner,
                'kind':effective,'owner':owner,'transformed':transformed,'revealed':revealed,
                'queries':q,'selected_characters':selected}
        return result

    def observe(self,fields,viewers,active):return self.launch(self.prepare(fields,viewers,active))
    def close(self):
        if not self.closed:self.queries.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
