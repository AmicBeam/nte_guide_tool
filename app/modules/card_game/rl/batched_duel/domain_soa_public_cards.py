"""Public candidate card tables projected directly from named device state.

Only the observer's hand and its visible pending choices are exported. This
projection does not generate legal actions, execute cards or qualify H3.
"""
from dataclasses import dataclass
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.rl import seven_lineup as seven
from .domain_soa_schema import CARD_VOCAB,CARD_INT_FIELDS,CARD_BOOL_FIELDS,CARD_STR_FIELDS,CHARACTER_VOCAB
from .domain_soa_observation import DomainSoAObservation,CI,CB,CS
from .domain_soa_candidates import CARD_FIELDS

KI={k:i for i,k in enumerate(CARD_INT_FIELDS)}
KB={k:i for i,k in enumerate(CARD_BOOL_FIELDS)}
KS={k:i for i,k in enumerate(CARD_STR_FIELDS)}
supports_full_duel_rules=False


@dataclass(frozen=True)
class PreparedPublicCards:
    owner: object
    observation: object
    cards: object
    valid: object
    hand_refs: object
    choice_refs: object
    errors: object
    rule_fields: object


class DomainPublicCards:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False):
        self.observation=DomainSoAObservation(layout,device=device,allow_cpu_oracle=allow_cpu_oracle)
        self.t=self.observation.t;self.device=self.observation.device;self.layout=self.observation.layout;self.closed=False
        t=self.t
        for text in ('form','set','battle','harmony_damage','surplus','own_burn'):
            self.observation.literals[text]=t.tensor(list(text.encode()),dtype=t.uint8,device=self.device)
        self.catalog_bool=t.tensor([[bool(CARDS[c].get(k))for k in CARD_BOOL_FIELDS]for c in seven.CARD_IDS],device=self.device)
        self.catalog_int=t.tensor([[int(CARDS[c].get(k)or 0)for k in CARD_INT_FIELDS]for c in seven.CARD_IDS],dtype=t.int64,device=self.device)
        self.catalog_form=t.tensor([CARDS[c].get('type')=='form'for c in seven.CARD_IDS],device=self.device)
        self.catalog_battle=t.tensor([CARDS[c].get('type')=='battle'for c in seven.CARD_IDS],device=self.device)
        self.catalog_require=t.tensor([('harmony_damage','surplus','own_burn').index(CARDS[c]['require'])+1 if CARDS[c].get('require')in ('harmony_damage','surplus','own_burn')else 0 for c in seven.CARD_IDS],dtype=t.int64,device=self.device)
        self.catalog_options=t.tensor([bool(CARDS[c].get('play_options'))for c in seven.CARD_IDS],device=self.device)
        self.catalog_set=t.tensor([CARDS[c].get('attack_mode')=='set'for c in seven.CARD_IDS],device=self.device)

    def prepare(self,fields,viewers,active):
        if self.closed:raise RuntimeError('Public card projector closed')
        t=self.t;p=self.observation.prepare(fields,viewers,active);B=viewers.shape[0];K=self.layout.max_cards
        return PreparedPublicCards(self,p,t.empty((B,2*K,len(CARD_FIELDS)),dtype=t.float64,device=self.device),
            t.empty((B,2*K),dtype=t.bool,device=self.device),
            t.empty((B,self.layout.max_zone),dtype=t.int64,device=self.device),
            t.empty((B,self.layout.max_choice_cards),dtype=t.int64,device=self.device),
            t.empty(B,dtype=t.int32,device=self.device),
            {key:t.empty((B,self.layout.max_zone),dtype=dtype,device=self.device)for key,dtype in
             [('kind',t.int64),('owner',t.int64),('owner_slot',t.int64),('is_battle',t.bool),('plays_instant',t.bool),
              ('cost',t.int64),('spend_all_ap',t.int64),('require',t.int64),('playable_downed',t.bool),('options',t.bool)]})

    def launch(self,p,*,_include_rules=False):
        if type(_include_rules)is not bool:raise TypeError('Internal rules flag must be a static bool')
        if self.closed or not isinstance(p,PreparedPublicCards)or p.owner is not self:raise ValueError('Closed or foreign public card buffers')
        t=self.t;out=self.observation.launch(p.observation,_include_private=True);b=p.observation;f=b.fields;meta=out['_private_card_fields']
        B=f['row_error'].shape[0];K=self.layout.max_cards;C=self.layout.max_characters
        rows=t.arange(B,device=self.device);own=b.viewers.clamp(0,1)
        raw=meta['raw_kind'];effective=meta['kind'];owner=meta['owner'];raw_owner=meta['raw_owner'];transformed=meta['transformed']!=0
        errors=out['errors'].clone()
        def kbool(key):
            j=KB[key];pr=f['card_bool_presence'][...,j]
            return t.where(pr==3,self.catalog_bool[raw,j],(pr==2)&f['card_bool_value'][...,j])
        def kint(key):
            j=KI[key];pr=f['card_int_presence'][...,j]
            return t.where(pr==3,self.catalog_int[raw,j],t.where(pr==2,f['card_int_value'][...,j],0))
        def kstring(key,literal,catalog):
            j=KS[key];pr=f['card_str_presence'][...,j]
            eq,bad=self.observation._eq(f,f['card_str_slot'][...,j],literal)
            return t.where(pr==3,catalog,(pr==2)&eq),bad&(pr==2)
        isform,formbad=kstring('type','form',self.catalog_form[raw])
        is_set,setbad=kstring('attack_mode','set',self.catalog_set[raw])
        isform=t.where(transformed,self.catalog_form[effective],isform)
        isform=t.where(meta['transformed']==2,t.zeros_like(isform),isform)
        is_set=t.where(transformed,self.catalog_set[effective],is_set)
        instant=t.where(transformed,self.catalog_bool[effective,KB['instant']],kbool('instant'))
        instant=instant|(meta['transformed']==2)
        derived=t.where(transformed,self.catalog_bool[effective,KB['derived']],kbool('derived'))
        extra_ap=t.where(transformed,0,kint('additional_ap_cost'))
        cbp=f['character_bool_presence'];cbv=f['character_bool_value'];tm=f['character_template'].long()
        selected=meta['selected_characters']&f['character_valid']
        living=selected&(f['character_hp']>0)
        zero=tm==CHARACTER_VOCAB['zero']
        zero_awake=(living&zero&(cbp[...,CB['awakened']]==2)&cbv[...,CB['awakened']]).any(-1)
        shape_exists=(f['character_str_presence'][...,CS['shape']]==2)&(f['string_length'].gather(1,f['character_str_slot'][...,CS['shape']].flatten(1).long().clamp(0,self.layout.max_string_slots-1)).reshape(B,2,C)>0)
        zero_shaped=(f['character_valid']&(f['character_hp']>0)&zero&shape_exists).any(-1)
        side=f['card_side'].long().clamp(0,1)
        # Public card owner is a seat, while effect subjects use stored entities.
        seats=self.observation.partial.catalog.character_to_seat[tm.clamp(0,self.observation.partial.catalog.character_to_seat.shape[0]-1)].long()
        hits=(seats[:,None,:,:]==owner[:,:,None,None])&f['character_valid'][:,None,:,:]
        hits=hits&(t.arange(2,device=self.device)[None,None,:,None]==side[:,:,None,None])
        found=hits.flatten(2).any(-1)
        holder=(hits.long()*t.arange(2*C,device=self.device).reshape(1,1,2,C)).sum((2,3))
        owner_slot=holder.remainder(C)
        forms=isform&(zero_awake.gather(1,side)|((owner==seven.SEAT_INDEX['zero'])&zero_shaped.gather(1,side)))
        original,origbad,origlen=self.observation._lookup(f,f['card_str_slot'][...,KS['original_character_id']],'seats')
        origbad=(f['card_str_presence'][...,KS['original_character_id']]==2)&(origlen>0)&(origbad|(original<0))
        original=t.where(f['card_str_presence'][...,KS['original_character_id']]==2,original,-1)
        nanali=(owner==seven.SEAT_INDEX['nanali'])|(original==seven.SEAT_INDEX['nanali'])
        haiyue=(owner==seven.SEAT_INDEX['haiyue'])|(original==seven.SEAT_INDEX['haiyue'])
        yi=(owner==seven.SEAT_INDEX['yi'])|(original==seven.SEAT_INDEX['yi'])
        nf=(effective==seven.CARD_INDEX['NF01'])&(f['turn'][:,None]>=5)&nanali
        attacked=f['player_attacked_this_turn'][rows[:,None],side]
        other_attacked=(attacked&(t.arange(C,device=self.device)[None,None,:]!=owner_slot[...,None])).any(-1)
        hy=haiyue&(((effective==seven.CARD_INDEX['U01'])&(f['active_side'][:,None]!=side))|
            (((effective==seven.CARD_INDEX['U02'])|(effective==seven.CARD_INDEX['U07']))&other_attacked))
        yi_alive=(f['character_valid']&(tm==CHARACTER_VOCAB['yi'])&(f['character_hp']>0)).any(-1)
        delay=(f['zone_status_presence'][:,:,0]==2)&(f['zone_status_i32'][:,:,0,7]!=0)
        yi_instant=yi&(effective==seven.CARD_INDEX['I08'])&yi_alive.gather(1,side)&delay.gather(1,1-side)
        base=instant|forms|nf|hy|yi_instant
        values=meta['queries']['values']
        false_col=values[rows[:,None],side,owner_slot,21]!=0
        true_col=values[rows[:,None],side,owner_slot,22]!=0
        plays_instant=t.where(base,true_col,false_col)
        projected_instant=instant|plays_instant
        phy=f['card_physical_index'].long().clamp(0,K-1)
        raw_face=t.where(f['card_is_view'],raw.gather(1,phy),-1)
        hand_face=t.where(transformed,raw,raw_face)
        mark=f['card_jingu_mark_presence']==2;delta=t.where(mark,f['card_jingu_delta'],0)
        # Antique investment is not in the accepted raw schema; such source
        # states are rejected by the strict codec, never silently projected.
        zero_num=t.zeros_like(raw)
        def table(kind,author,revealed,setmode,inst,der,face,surcharge):
            return t.stack((kind,author,revealed,mark,delta,setmode,inst,kbool('copy'),kint('expires_turn'),der,
                            zero_num,face,surcharge),-1).double()
        hand=table(effective,owner,meta['revealed'][rows,own],is_set,projected_instant,derived,hand_face,extra_ap)
        raw_set,_=kstring('attack_mode','set',self.catalog_set[raw])
        choices=table(raw,raw_owner,t.zeros_like(mark),raw_set,kbool('instant'),kbool('derived'),raw_face,kint('additional_ap_cost'))
        # Retain table-local identity but mask every non-visible entry.
        hand_indices=f['zone_card_index'][rows,own,0].long();hpos=t.arange(self.layout.max_zone,device=self.device)[None,:]
        hvalid=hpos<f['zone_count'][rows,own,0,None]
        safe_hand=hand_indices.clamp(0,K-1)
        hmask=t.zeros((B,K),dtype=t.int32,device=self.device);hmask.scatter_add_(1,safe_hand,hvalid.int());hmask=hmask>0
        choice_indices=f['choice_card_index'].long();cpos=t.arange(self.layout.max_choice_cards,device=self.device)[None,:]
        cvalid=(cpos<f['choice_card_count'][:,None])&f['pending_choice_present'][:,None]&(f['choice_side'].long()==own)[:,None]
        safe_choice=choice_indices.clamp(0,K-1)
        cmask=t.zeros((B,K),dtype=t.int32,device=self.device);cmask.scatter_add_(1,safe_choice,cvalid.int());cmask=cmask>0
        bad=(hvalid&((hand_indices<0)|(hand_indices>=K))).any(-1)|(cvalid&((choice_indices<0)|(choice_indices>=K))).any(-1)
        bad|=(hmask&(~found|formbad|setbad|origbad)).any(-1)
        bad|=(hmask&~f['card_valid']).any(-1)|(cmask&~f['card_valid']).any(-1)
        errors=errors|bad.int()
        valid=t.cat((hmask,cmask),-1)&b.active[:,None]&(errors==0)[:,None]
        p.errors.copy_(errors);p.valid.copy_(valid)
        table_all=t.cat((hand,choices),1)
        p.cards.copy_(t.where(valid[...,None],table_all,t.zeros_like(table_all)))
        p.cards[...,0].masked_fill_(~valid,-1)
        p.hand_refs.copy_(t.where(hvalid&b.active[:,None]&(errors==0)[:,None],safe_hand,-1))
        p.choice_refs.copy_(t.where(cvalid&b.active[:,None]&(errors==0)[:,None],safe_choice+K,-1))
        result={'cards':p.cards,'valid':p.valid,'hand_refs':p.hand_refs,'choice_refs':p.choice_refs,'errors':p.errors,
                'card_fields':CARD_FIELDS,'observation':out['features'],'legal_actions_available':False,'full_duel_qualified':False}
        if _include_rules:
            battle,_=kstring('type','battle',self.catalog_battle[raw])
            battle=t.where(transformed,self.catalog_battle[effective],battle)&(meta['transformed']!=2)
            cost=t.where(transformed,self.catalog_int[effective,KI['cost']],kint('cost'))
            cost=t.where(meta['transformed']==2,1,cost)
            limit=t.where(transformed,self.catalog_int[effective,KI['spend_all_ap']],kint('spend_all_ap'))
            req=t.where(f['card_str_presence'][...,KS['require']]==3,self.catalog_require[raw],0)
            for i,text in enumerate(('harmony_damage','surplus','own_burn')):
                eq,_=self.observation._eq(f,f['card_str_slot'][...,KS['require']],text)
                req=t.where((f['card_str_presence'][...,KS['require']]==2)&eq,i+1,req)
            req=t.where(transformed,self.catalog_require[effective],req)
            play_down=t.where(transformed,self.catalog_bool[effective,KB['playable_downed']],kbool('playable_downed'))
            options=t.where(transformed,self.catalog_options[effective],self.catalog_options[raw]&(f['card_play_options_presence']==3))
            full=dict(kind=effective,owner=owner,owner_slot=owner_slot,is_battle=battle,plays_instant=plays_instant,
                      cost=cost,spend_all_ap=limit,require=req,playable_downed=play_down,options=options)
            for name,dest in p.rule_fields.items():
                value=full[name].gather(1,safe_hand)
                dest.copy_(t.where(hvalid&b.active[:,None]&(errors==0)[:,None],value,t.zeros_like(value)))
            result['_hand_rules']=p.rule_fields
            result['_character_queries']=meta['queries']['values']
        return result

    def project(self,fields,viewers,active):return self.launch(self.prepare(fields,viewers,active))
    def close(self):
        if not self.closed:self.observation.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
