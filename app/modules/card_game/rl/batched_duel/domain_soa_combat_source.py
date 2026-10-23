"""Direct C++ combat source for the no-VM domain executor.

Emits DFN implementations of damage, knockdowns, sortie, enter, harmony and
continuous damage clocks. Card/kit dispatch and turn lifecycle remain other
workers. Common field accessors are the shared ABI, not host rule masks.
"""

supports_full_duel_rules = False

DK_DAMAGE = 0
DK_COMBAT = 1
DK_GENESIS = 2
DK_FLOWER = 3
DK_DELAY = 4
DK_BURN = 5
DK_STAR = 6
DK_OVERLAY = 7
DK_SYNC = 8
DK_PENETRATION = 9
DK_FOLLOWUP = 10

HK_NONE = 0
HK_GENESIS = 1
HK_DELAY = 2
HK_BURN = 3
HK_STAR = 4
HK_WEAVE = 5
HK_STAIN = 6
HK_SYNC = 7

# Matches domain_soa_character_queries.ATTRIBUTES (1-based).
ATTR_LIGHT = 1
ATTR_DARK = 2
ATTR_SPIRIT = 3
ATTR_SOUL = 4
ATTR_PHASE = 5
ATTR_HEX = 6

HARMONY_PY = {
    "\u521b\u751f": HK_GENESIS,
    "\u5ef6\u6ede": HK_DELAY,
    "\u6d4a\u71c3": HK_BURN,
    "\u9eef\u661f": HK_STAR,
    "\u8986\u7eb9": HK_WEAVE,
    "\u6d78\u67d3": HK_STAIN,
    "\u540c\u9891": HK_SYNC,
}
ATTR_PY = {
    "\u5149": ATTR_LIGHT,
    "\u6697": ATTR_DARK,
    "\u7075": ATTR_SPIRIT,
    "\u9b42": ATTR_SOUL,
    "\u76f8": ATTR_PHASE,
    "\u5492": ATTR_HEX,
}
KIND_PY = {
    "damage": DK_DAMAGE,
    "combat": DK_COMBAT,
    "genesis": DK_GENESIS,
    "flower": DK_FLOWER,
    "delay": DK_DELAY,
    "burn": DK_BURN,
    "star": DK_STAR,
    "overlay": DK_OVERLAY,
    "sync": DK_SYNC,
    "penetration": DK_PENETRATION,
}

_SOURCE = r'''
#ifndef DFN
#error combat source requires DFN from the shared ABI header
#endif

enum CombatDamageKind {
 DK_DAMAGE=0, DK_COMBAT=1, DK_GENESIS=2, DK_FLOWER=3, DK_DELAY=4,
 DK_BURN=5, DK_STAR=6, DK_OVERLAY=7, DK_SYNC=8, DK_PENETRATION=9, DK_FOLLOWUP=10
};
enum CombatHarmonyKind {
 HK_NONE=0, HK_GENESIS=1, HK_DELAY=2, HK_BURN=3, HK_STAR=4,
 HK_WEAVE=5, HK_STAIN=6, HK_SYNC=7
};
enum CombatAttr {
 ATTR_NONE=0, ATTR_LIGHT=1, ATTR_DARK=2, ATTR_SPIRIT=3, ATTR_SOUL=4,
 ATTR_PHASE=5, ATTR_HEX=6
};
enum CombatClass { DC_INFER=0, DC_ESPER=1, DC_HARMONY=2, DC_ATTACHMENT=3 };
enum CombatDotKind { DOT_NIGHTMARE=0, DOT_ETCH=1, DOT_VENOM=2, DOT_GUARD=3 };
#ifndef COMBAT_D
#define COMBAT_D 4
#endif

#ifndef NSI
#define NSI 9
#define NSB 12
#endif
#ifndef DCOMBAT_SPEC_DEFINED
#define DCOMBAT_SPEC_DEFINED
struct DCombatSpec {
 DAttack base;
 int stay_bench, front_only, additional_attack, bypass;
 int penetrate_harmony, penetrate_low_hp, penetrate_brand;
 int no_combat_resources, support_allows_counter, no_counter_harmony;
 int sync_attack, shield_after_shape, attribute, has_target;
 DR target;
 unsigned char int_presence[NSI];
 unsigned char bool_presence[NSB];
 unsigned char has_attribute;
};
#endif
DFN void d_set_pending_sortie(DX*,const DCombatSpec*);
DFN int d_load_pending_sortie(DX*,DCombatSpec*);
DFN void d_consume_applied(DX*,const char*,const int*,int);
DFN void d_record_delay_turn(DX*,int);
DFN void d_apply_dark_star(DX*,int);
DFN void d_clear_harmony_source(DX*,int,int);

struct CombatOpSnap {
 int side, actor, card, first_bonus, redeem_bonus;
 unsigned char harmony_presence, target_presence, last_presence;
 int harmony_slot, target_slot, last_slot;
 unsigned char flag_presence[NOPBOOL];
 unsigned char flag_value[NOPBOOL];
 unsigned char int_presence[NOPINT];
 int int_value[NOPINT];
 short participant_count;
 signed char part_side[PART];
 short part_slot[PART];
 short choice_n;
 int choice_i[CH];
 unsigned char pending_p, stay, bonus_p;
 int bonus;
 unsigned char sortie_int_p[NSI];
 int sortie_int_v[NSI];
 unsigned char sortie_bool_p[NSB];
 unsigned char sortie_bool_v[NSB];
 unsigned char sortie_target_p, sortie_attr_p;
 signed char sortie_target_side, sortie_attr;
 short sortie_target_actor;
};

DFN int combat_ok(DX* c){return !d_bad(c) && !d_finished(c);}
DFN int combat_esper(DR r){return r.slot>=0;}
DFN DX combat_ctx(DX* c,int side,int actor){DX n=*c;n.side=side;n.actor=actor;n.damage_source_slot=-1;return n;}
DFN DEvt combat_evt(void){
 DEvt e{}; e.actor=e.card=e.attacker_side=e.attacker=e.defender_side=e.defender=e.target_side=e.target=-1;
 e.amount=e.loss=e.absorbed=e.raw_overflow=e.kind=e.attribute=e.additional=e.index=e.response=e.entering=0;
 return e;
}
DFN int combat_turn(DX* c){return c->s->turn[0];}
DFN int combat_active(DX* c){return c->s->active_side[0];}
DFN int combat_phase(DX* c){return c->s->phase[0];}
DFN int combat_last_front(DX* c,int side){
 if(side<0||side>1){d_fail(c,1);return -1;}
 return c->s->player_last_front_presence[side]==2?c->s->player_last_front_slot[side]:-1;
}
DFN void combat_set_last_front(DX* c,int side,int slot){
 if(side<0||side>1){d_fail(c,1);return;}
 c->s->player_last_front_presence[side]=slot>=0?2:1;
 c->s->player_last_front_slot[side]=slot>=0?slot:0;
}
DFN int combat_attr(DX* c,int side,int ch){
 if(ch<0)return 0;
 if(d_cstr_is(c,side,ch,"attribute","\xe5\x85\x89"))return ATTR_LIGHT;
 if(d_cstr_is(c,side,ch,"attribute","\xe6\x9a\x97"))return ATTR_DARK;
 if(d_cstr_is(c,side,ch,"attribute","\xe7\x81\xb5"))return ATTR_SPIRIT;
 if(d_cstr_is(c,side,ch,"attribute","\xe9\xad\x82"))return ATTR_SOUL;
 if(d_cstr_is(c,side,ch,"attribute","\xe7\x9b\xb8"))return ATTR_PHASE;
 if(d_cstr_is(c,side,ch,"attribute","\xe5\x92\x92"))return ATTR_HEX;
 return 0;
}
DFN int combat_source_attr(DX* c,DR source){
 if(source.side<0||source.slot<0)return 0;
 return combat_attr(c,source.side,source.slot);
}
DFN int combat_damage_class(int kind,int dclass){
 if(dclass==DC_HARMONY||dclass==DC_ATTACHMENT)return dclass;
 if(kind==DK_GENESIS||kind==DK_FLOWER||kind==DK_DELAY||kind==DK_BURN||kind==DK_STAR||kind==DK_OVERLAY||kind==DK_SYNC)return DC_HARMONY;
 return DC_ESPER;
}
DFN int combat_harmony_pair(int a,int b){
 int x=a<b?a:b,y=a<b?b:a;
 if(x==ATTR_LIGHT&&y==ATTR_SPIRIT)return HK_GENESIS;
 if(x==ATTR_LIGHT&&y==ATTR_PHASE)return HK_DELAY;
 if(x==ATTR_DARK&&y==ATTR_HEX)return HK_BURN;
 if(x==ATTR_DARK&&y==ATTR_SOUL)return HK_STAR;
 if(x==ATTR_SPIRIT&&y==ATTR_HEX)return HK_WEAVE;
 if(x==ATTR_SOUL&&y==ATTR_PHASE)return HK_STAIN;
 return HK_NONE;
}
DFN int combat_hp(DX* c,DR r){return combat_esper(r)?d_ci(c,r.side,r.slot,"hp"):d_pi(c,r.side,"hp");}
DFN int combat_shield(DX* c,DR r){return combat_esper(r)?d_ci(c,r.side,r.slot,"shield"):d_pi(c,r.side,"shield");}
DFN void combat_set_hp(DX* c,DR r,int v){if(combat_esper(r))d_set_ci(c,r.side,r.slot,"hp",v);else d_set_pi(c,r.side,"hp",v);}
DFN void combat_set_shield(DX* c,DR r,int v){if(combat_esper(r))d_set_ci(c,r.side,r.slot,"shield",v);else d_set_pi(c,r.side,"shield",v);}
DFN int combat_eval_bool(DX* c,DR r,const char* prop,int base,const DEvt* ev){
 return d_evaluate(c,r,prop,(double)base,ev)!=0.0;
}
DFN int combat_next_group(DX* c){
 if(c->s->combat_group[0]==2147483647){d_fail(c,2);return 0;}
 c->s->combat_group[0]++; return c->s->combat_group[0];
}
DFN void combat_set_last_hit(DX*c,int side,int ch,DR source){int slot=c->damage_source_slot>=0?c->damage_source_slot:d_ref_target_slot(c,source);int at=(side*C+ch)*NS+CS_last_hit_source;c->s->character_str_presence[at]=2;c->s->character_str_slot[at]=slot;}
DFN DR combat_parse_ref(DX*c,int slot){if(slot<0||slot>=c->s->string_count[0])return DR{-1,-2};int n=c->s->string_length[slot],at=c->s->string_offset[slot];if(n<3||n>125)return DR{-1,-2};int side=c->s->string_bytes[at]=='a'?0:c->s->string_bytes[at]=='b'?1:-1;if(side<0||c->s->string_bytes[at+1]!=':')return DR{-1,-2};char name[128];for(int j=2;j<n;j++)name[j-2]=c->s->string_bytes[at+j];name[n-2]=0;if(d_streq(name,"player"))return DR{side,-1};return DR{side,d_hero(c,side,name)};}
DFN DR combat_last_hit(DX* c,int side,int ch){
 int i=(side*C+ch)*NS+CS_last_hit_source;
 if(c->s->character_str_presence[i]!=2) return DR{-1,-2};
 return combat_parse_ref(c,c->s->character_str_slot[i]);
}
DFN int combat_op_flag(DX* c,const char* key){return d_op_int(c,key,0);}
DFN void combat_set_op_flag(DX* c,const char* key,int v){d_set_op_int(c,key,v,1);}
DFN int combat_support(DX* c){return combat_op_flag(c,"support");}

DFN void combat_save_op(DX* c,CombatOpSnap* o){
 DS* f=c->s; o->side=f->op_side[0]; o->actor=f->op_actor_slot[0]; o->card=f->op_card_index[0];
 o->first_bonus=f->op_first_bonus[0]; o->redeem_bonus=f->op_redeem_bonus[0];
 o->harmony_presence=f->op_harmony_presence[0]; o->harmony_slot=f->op_harmony_slot[0];
 o->target_presence=f->op_target_presence[0]; o->target_slot=f->op_target_slot[0];
 o->last_presence=f->op_last_attack_target_presence[0]; o->last_slot=f->op_last_attack_target_slot[0];
 for(int i=0;i<NOPBOOL;i++){o->flag_presence[i]=f->op_flag_presence[i]; o->flag_value[i]=f->op_flag_value[i];}
 for(int i=0;i<NOPINT;i++){o->int_presence[i]=f->op_int_presence[i]; o->int_value[i]=f->op_int_value[i];}
 int n=f->op_participant_count[0]; if(n<0)n=0; if(n>PART){d_fail(c,2);n=PART;} o->participant_count=(short)n;
 for(int i=0;i<n;i++){o->part_side[i]=f->op_participant_side[i]; o->part_slot[i]=f->op_participant_slot[i];}
 int cn=f->op_choice_card_count[0]; if(cn<0)cn=0; if(cn>CH){d_fail(c,2);cn=CH;} o->choice_n=(short)cn;
 for(int i=0;i<cn;i++) o->choice_i[i]=f->op_choice_card_index[i];
 o->pending_p=f->op_pending_sortie_presence[0]; o->stay=f->op_pending_stay_bench[0];
 o->bonus_p=f->op_pending_bonus_presence[0]; o->bonus=f->op_pending_bonus[0];
 for(int i=0;i<NSI;i++){o->sortie_int_p[i]=f->op_sortie_int_presence[i]; o->sortie_int_v[i]=f->op_sortie_int_value[i];}
 for(int i=0;i<NSB;i++){o->sortie_bool_p[i]=f->op_sortie_bool_presence[i]; o->sortie_bool_v[i]=f->op_sortie_bool_value[i];}
 o->sortie_target_p=f->op_sortie_target_presence[0]; o->sortie_target_side=f->op_sortie_target_side[0]; o->sortie_target_actor=f->op_sortie_target_actor[0];
 o->sortie_attr_p=f->op_sortie_attribute_presence[0]; o->sortie_attr=f->op_sortie_attribute[0];
}
DFN void combat_restore_op(DX* c,const CombatOpSnap* o){
 DS* f=c->s; f->op_side[0]=(signed char)o->side; f->op_actor_slot[0]=(short)o->actor; f->op_card_index[0]=o->card;
 f->op_first_bonus[0]=o->first_bonus; f->op_redeem_bonus[0]=o->redeem_bonus;
 f->op_harmony_presence[0]=o->harmony_presence; f->op_harmony_slot[0]=o->harmony_slot;
 f->op_target_presence[0]=o->target_presence; f->op_target_slot[0]=o->target_slot;
 f->op_last_attack_target_presence[0]=o->last_presence; f->op_last_attack_target_slot[0]=o->last_slot;
 for(int i=0;i<NOPBOOL;i++){f->op_flag_presence[i]=o->flag_presence[i]; f->op_flag_value[i]=o->flag_value[i];}
 for(int i=0;i<NOPINT;i++){f->op_int_presence[i]=o->int_presence[i]; f->op_int_value[i]=o->int_value[i];}
 f->op_participant_count[0]=o->participant_count;
 int n=o->participant_count; for(int i=0;i<n;i++){f->op_participant_side[i]=o->part_side[i]; f->op_participant_slot[i]=o->part_slot[i];}
 f->op_choice_card_count[0]=o->choice_n; for(int i=0;i<o->choice_n;i++) f->op_choice_card_index[i]=o->choice_i[i];
 f->op_pending_sortie_presence[0]=o->pending_p; f->op_pending_stay_bench[0]=o->stay;
 f->op_pending_bonus_presence[0]=o->bonus_p; f->op_pending_bonus[0]=o->bonus;
 for(int i=0;i<NSI;i++){f->op_sortie_int_presence[i]=o->sortie_int_p[i]; f->op_sortie_int_value[i]=o->sortie_int_v[i];}
 for(int i=0;i<NSB;i++){f->op_sortie_bool_presence[i]=o->sortie_bool_p[i]; f->op_sortie_bool_value[i]=o->sortie_bool_v[i];}
 f->op_sortie_target_presence[0]=o->sortie_target_p; f->op_sortie_target_side[0]=o->sortie_target_side; f->op_sortie_target_actor[0]=o->sortie_target_actor;
 f->op_sortie_attribute_presence[0]=o->sortie_attr_p; f->op_sortie_attribute[0]=o->sortie_attr;
}
DFN void combat_append_participant(DX* c,int side,int actor){
 DS* f=c->s; int n=f->op_participant_count[0];
 for(int i=0;i<n;i++) if(f->op_participant_side[i]==side && f->op_participant_slot[i]==actor) return;
 if(n>=PART){d_fail(c,2);return;}
 f->op_participant_side[n]=(signed char)side; f->op_participant_slot[n]=(short)actor; f->op_participant_count[0]=(short)(n+1);
}
DFN void combat_mark_attacked(DX* c,int side,int actor){
 DS* f=c->s; int n=f->player_attacked_count[side];
 int eid=d_entity(c,DR{side,actor});
 for(int i=0;i<n && i<C;i++) if(f->player_attacked_slot[side*C+i]==actor) return;
 if(n>=C){d_fail(c,2);return;}
 f->player_attacked_this_turn_presence[side]=2;
 f->player_attacked_this_turn[side*C+actor]=1;
 f->player_attacked_slot[side*C+n]=(short)actor;
 f->player_attacked_count[side]=(short)(n+1);
 (void)eid;
}
DFN void combat_victory(DX* c){
 if(d_finished(c)||d_bad(c))return;
 int da=d_pi(c,0,"hp")<=0, db=d_pi(c,1,"hp")<=0;
 if(!da&&!db)return;
 if(da&&db) d_finish(c,-1,"player_hp");
 else d_finish(c,da?1:0,"player_hp");
}
DFN void combat_clear_harmony_source(DX* c,int side,int ch){
 d_clear_harmony_source(c,side,ch);
}
DFN void combat_clear_holder_flags(DX* c,int side,int ch){
 int h=d_holder(c,side,ch); DS* f=c->s;
 for(int i=0;i<NF;i++){ f->flag_int_presence[h*NF+i]=0; f->flag_int_value[h*NF+i]=0; }
 for(int i=0;i<FB;i++){ f->flag_bool_presence[h*FB+i]=0; f->flag_bool_value[h*FB+i]=0; }
 for(int i=0;i<FS;i++){ f->flag_str_presence[h*FS+i]=0; f->flag_str_slot[h*FS+i]=0; }
 f->flag_collapse_presence[h]=0; f->flag_collapse_side[h]=0; f->flag_collapse_until[h]=0;
 f->flag_mimic_presence[h]=0; f->flag_mimic_id_presence[h]=0; f->flag_mimic_template[h]=0;
 f->flag_expiring_presence[h]=0; f->flag_timed_count[h]=0;
}
DFN void combat_dismiss_summon(DX* c,int side,int ch){
 if(!combat_ok(c))return;
 d_set_ci(c,side,ch,"hp",0); d_set_ci(c,side,ch,"shield",0); d_set_cb(c,side,ch,"disappeared",1);
 if(d_front(c,side)==ch) d_set_front(c,side,-1);
 combat_set_last_front(c,side,-1);
 int n=c->s->player_order_count[side], w=0;
 for(int i=0;i<n;i++){ int sl=c->s->player_order_slot[side*C+i]; if(sl!=ch) c->s->player_order_slot[side*C+w++]=(short)sl; }
 c->s->player_order_count[side]=(short)w;
 d_clear_seal_if_summon(c,1-side,ch);
 d_event(c);
}
DFN int combat_collapse_max(DX* c,int side,int ch){
 int base=d_flag(c,side,ch,"collapse_max"); if(base<=0) base=5;
 int h=side*C+ch; int n=c->s->character_reductions_count[h]; if(n<0)n=0; if(n>2*C)n=2*C;
 int sum=0; for(int i=0;i<n;i++) sum+=c->s->character_reductions_value[h*(2*C)+i];
 int v=base-sum; return v<1?1:v;
}
DFN void combat_force_collapse(DX* c,DR target,int by){
 if(!combat_ok(c)||!combat_esper(target)||!d_alive(c,target.side,target.slot))return;
 if(d_collapsed(c,target.side,target.slot))return;
 int applier=by>=0?by:d_enemy(target.side);
 d_set_flag(c,target.side,target.slot,"collapse_count",0,0);
 int h=d_holder(c,target.side,target.slot);
 c->s->flag_collapse_presence[h]=2;
 c->s->flag_collapse_side[h]=(signed char)applier;
 c->s->flag_collapse_until[h]=d_pi(c,applier,"turn_count")+1;
 d_event(c);
 int carrier=-1;
 for(int i=0;i<d_order_count(c,applier);i++){
  int sl=d_order_slot(c,applier,i);
  if(!d_cb(c,applier,sl,"summoned")){carrier=sl;break;}
 }
 if(carrier>=0){
  DX ctx=combat_ctx(c,applier,carrier);
  DEvt ev=combat_evt(); ev.target_side=target.side; ev.target=target.slot;
  d_fire(&ctx,"on_collapse_entered",&ev,0,0);
 }
}
DFN void d_add_collapse(DX* c,DR target,int amount,int by){
 if(!combat_ok(c)||!combat_esper(target)||!d_alive(c,target.side,target.slot))return;
 if(amount<0)amount=0;
 int before=d_flag(c,target.side,target.slot,"collapse_count");
 d_set_flag(c,target.side,target.slot,"collapse_count",before+amount,0);
 if(amount) d_event(c);
 int limit=combat_collapse_max(c,target.side,target.slot);
 if(d_flag(c,target.side,target.slot,"collapse_count")>=limit && !d_collapsed(c,target.side,target.slot))
  combat_force_collapse(c,target,by);
}

DFN int combat_round_half_up_div(long long num,long long den){
 if(den<=0)return 0;
 long long q=num/den, r=num%den;
 if(r<0){r+=den;q--;}
 if(r*2>=den) return (int)(q+1);
 return (int)q;
}
DFN int combat_reduce_max_hp(DX* c,DR target,int amount,DR source,int settle){
 (void)source;
 if(!combat_ok(c))return 0;
 if(!combat_esper(target)){
  if(d_pb(c,target.side,"life_locked"))return 0;
 }
 int hp=combat_hp(c,target); if(hp<=0)return 0;
 if(amount<0)amount=0;
 int maxhp=d_max_hp(c,target.side,target.slot);
 int cut=0;
 for(int i=0;i<amount;i++){
  if(maxhp<=0)break;
  int neu=maxhp-1;
  hp=combat_round_half_up_div((long long)hp*neu,maxhp);
  if(hp>neu)hp=neu;
  maxhp=neu; cut++;
  if(hp<=0||maxhp<=0){hp=0;break;}
 }
 combat_set_hp(c,target,hp);
 if(combat_esper(target)) d_set_ci(c,target.side,target.slot,"max_hp",maxhp);
 else d_set_pi(c,target.side,"max_hp",maxhp);
 d_event(c);
 if(settle){ d_knockdowns(c); combat_victory(c);}
 return cut;
}

DFN int combat_pop_next_resistance(DX* c,DR target,int attribute){
 return d_pop_next_resistance(c,target,attribute);
}

DFN int combat_damage(DX* c,DR target,int amount,DR source,int kind,int bypass,int settle,int attribute,int additional,int counter_immunity,int applied,int dclass);
DFN int d_damage(DX* c,DR target,int amount,DR source,int kind,int bypass,int settle,int attribute,int additional,int counter_immunity){
 return combat_damage(c,target,amount,source,kind,bypass,settle,attribute,additional,counter_immunity,0,DC_INFER);
}

DFN int combat_damage(DX* c,DR target,int amount,DR source,int kind,int bypass,int settle,int attribute,int additional,int counter_immunity,int applied,int dclass){
 if(d_finished(c)||d_bad(c))return 0;
 int esper=combat_esper(target);
 if(esper && combat_hp(c,target)<=0)return 0;
 DEvt ev=combat_evt();
 ev.kind=kind; ev.attribute=attribute; ev.additional=additional;
 ev.attacker_side=source.side; ev.attacker=source.slot;
 ev.target_side=target.side; ev.target=target.slot; ev.amount=amount;
 /* 1. damage.amount modifiers */
 amount=(int)d_evaluate(c,target,"damage.amount",(double)amount,&ev);
 if(amount<0)amount=0;
 if(d_bad(c)||d_finished(c))return 0;
 /* 2. additional-attack bonus unless the caller already applied it */
 if(!applied && source.side>=0 && source.slot>=0){
  DX src=combat_ctx(c,source.side,source.slot);
  additional=d_decide(&src,"damage_is_additional",additional,&ev,1,0)!=0;
  if(additional){
   DEvt evb=ev; evb.target_side=target.side; evb.target=target.slot;
   amount+=d_decide(&src,"additional_attack_damage_bonus",0,&evb,0,0);
  }
 }
 if(dclass==DC_ATTACHMENT && source.side>=0){
  int carrier=d_order_slot(c,source.side,0);
  if(carrier>=0){ DX ctx=combat_ctx(c,source.side,carrier); amount+=d_tally(&ctx,"attachment_damage_bonus",&ev); }
 }
 if(esper){
  DX defn=combat_ctx(c,target.side,target.slot);
  amount=d_decide(&defn,"incoming_damage",amount,&ev,0,0);
  if(amount<0)amount=0;
 }
 if(d_bad(c)||d_finished(c))return 0;
 int klass=combat_damage_class(kind,dclass);
 int attrv=attribute?attribute:combat_source_attr(c,source);
 ev.attribute=attrv;
 /* 3. immunity: rule.damage_immune, combat immune, support counter immunity */
 int immune=0;
 if(esper && d_alive(c,target.side,target.slot) && combat_eval_bool(c,target,"rule.damage_immune",0,&ev)) immune=1;
 if(kind==DK_COMBAT && combat_eval_bool(c,target,"rule.combat_immune",0,&ev)) immune=1;
 if(counter_immunity==1 && kind==DK_COMBAT) immune=1;
 if(immune) amount=0;
 /* 4. stain: soul/phase combat vs current front +2 */
 if(!immune && kind==DK_COMBAT && (attrv==ATTR_SOUL||attrv==ATTR_PHASE) && d_zone_has(c,target.side,"stain")){
  int front=d_front(c,target.side);
  if((esper && front==target.slot) || (!esper && front<0)) amount+=2;
 }
 /* 5. resistance evaluate then one-shot next_damage_resistance, then consume */
 if(amount>0){
  DEvt q=ev; q.amount=amount; q.attribute=attrv; q.kind=kind;
  int resist=(int)d_evaluate(c,target,"damage.resistance",0.0,&q);
  int popped=combat_pop_next_resistance(c,target,attrv);
  amount-=resist+popped; if(amount<0)amount=0;
  int ids[MAXQ]; int ni=0, herr=0, subject=d_holder(c,target.side,target.slot);
  for(int h=0;h<H;h++) if(exists(c->s,h)){
   int count=c->s->holder_effect_count[h]; if(count<0||count>E){d_fail(c,1);break;}
   for(int j=0;j<count;j++){
    int e=h*E+j;
    if(c->s->effect_consume_on_presence[e]!=2)continue;
    if(!d_eq(c,c->s->effect_consume_on_slot[e],"damage"))continue;
    if(!d_eq(c,c->s->effect_property_slot[e],"damage.resistance"))continue;
    if(!d_applies(c->s,e,h,subject,attrv,&herr))continue;
    if(ni<MAXQ) ids[ni++]=c->s->effect_id_slot[e];
   }
  }
  if(herr) d_fail(c,herr);
  d_consume_applied(c,"damage",ids,ni);
 }
 /* 6. public per-event cap */
 if(esper){
  double lim=d_evaluate(c,target,"damage.limit",-1.0,&ev);
  if(lim>=0.0 && amount>(int)lim) amount=(int)lim;
 }
 int before_hp=combat_hp(c,target), before_sh=combat_shield(c,target);
 int absorbed=bypass?0:(before_sh<amount?before_sh:amount);
 combat_set_shield(c,target,before_sh-absorbed);
 int loss=amount-absorbed;
 int redirect=-2, redirect_loss=0;
 /* 7. shield-break overflow redirect */
 if(esper && loss>0 && absorbed>0){
  DX defn=combat_ctx(c,target.side,target.slot);
  redirect=d_decide_redirect_overflow(&defn,target,loss,absorbed);
  if(redirect==target.slot || redirect<0 || !d_alive(c,target.side,redirect)) redirect=-2;
  else { redirect_loss=loss; amount=absorbed; loss=0; }
 }
 /* 8. lethal response then hp_floor */
 int floorv=0;
 if(esper){
  if(before_hp-loss<=0) d_trigger_responses(c,target.side,"self_lethal",target.slot,source.slot);
  if(d_bad(c)||d_finished(c)) return 0;
  DX defn=combat_ctx(c,target.side,target.slot);
  DEvt fev=ev; fev.target=target.slot;
  floorv=d_decide(&defn,"hp_floor",0,&fev,0,0);
 }
 if(!esper && d_pb(c,target.side,"life_locked")) loss=0;
 int next=before_hp-loss; if(next<floorv) next=floorv;
 combat_set_hp(c,target,next);
 /* 9. genesis-on-player history */
 if(kind==DK_GENESIS && !esper && source.side>=0 && source.side!=target.side && before_hp>next){
  d_set_pi(c,source.side,"genesis_player_hits",d_pi(c,source.side,"genesis_player_hits")+1);
 }
 if(esper && (source.side>=0||source.slot>=-1)) combat_set_last_hit(c,target.side,target.slot,source);
 d_event(c);
 if(c->trace && kind==DK_FLOWER) c->trace->genesis += (double)amount;
 if(klass==DC_HARMONY && source.side>=0) d_set_pb(c,source.side,"harmony_damage",1);
 ev.loss=loss; ev.absorbed=absorbed; ev.amount=amount;
 if(loss && combat_ok(c)){
  if(esper){
   DX defn=combat_ctx(c,target.side,target.slot);
   DEvt tev=ev; tev.amount=loss;
   d_fire(&defn,"on_damage_taken",&tev,1,0);
   if(d_cb(c,target.side,target.slot,"summoned")){
    int carrier=d_order_slot(c,target.side,0);
    if(carrier>=0){ DX ctx=combat_ctx(c,target.side,carrier); d_fire(&ctx,"on_ally_summon_hurt",&tev,0,0); }
   }
  }
  if(source.side>=0 && source.slot>=0){
   DX src=combat_ctx(c,source.side,source.slot);
   DEvt de=ev; de.amount=loss; de.target_side=target.side; de.target=target.slot; de.kind=kind;
   d_fire(&src,"on_damage_dealt",&de,1,0);
  }
 }
 if(d_bad(c))return 0;
 if(redirect_loss && redirect>=0 && combat_ok(c)){
  combat_damage(c,DR{target.side,redirect},redirect_loss,source,kind,0,settle,attribute,additional,0,1,dclass);
  return loss;
 }
 if(settle){ d_knockdowns(c); combat_victory(c); }
 return loss;
}

DFN int combat_knockdown_pass(DX* c){
 int downed_side[16], downed_slot[16], nd=0;
 for(int side=0; side<2; side++){
  int n=d_order_count(c,side);
  for(int i=0;i<n;i++){
   int ch=d_order_slot(c,side,i);
   if(d_ci(c,side,ch,"hp")>0 || d_ci(c,side,ch,"down_turns")>0) continue;
   if(d_cb(c,side,ch,"summoned")){ combat_dismiss_summon(c,side,ch); continue; }
   if(nd>=16){d_fail(c,2);return 0;}
   downed_side[nd]=side; downed_slot[nd]=ch; nd++;
   d_set_ci(c,side,ch,"hp",0);
   combat_clear_harmony_source(c,side,ch);
   DR killer=combat_last_hit(c,side,ch);
   d_del_cstr(c,side,ch,"last_hit_source");
   d_move_out(c,side,ch,0);
   int kept_h=d_ci(c,side,ch,"harmony"), kept_e=d_ci(c,side,ch,"energy");
   DX self=combat_ctx(c,side,ch);
   DEvt ev=combat_evt(); ev.actor=ch;
   d_fire(&self,"on_knockdown",&ev,1,1);
   if(killer.side>=0 && killer.slot>=0 && killer.side!=side && d_alive(c,killer.side,killer.slot)){
    DX kx=combat_ctx(c,killer.side,killer.slot);
    DEvt ke=combat_evt(); ke.actor=ch;
    d_fire(&kx,"on_rout",&ke,1,0);
   }
   if(killer.side>=0 && killer.slot>=0 && d_cb(c,killer.side,killer.slot,"summoned")){
    int carrier=killer.slot;
    for(int j=0;j<d_order_count(c,killer.side);j++){
     int sl=d_order_slot(c,killer.side,j);
     if(!d_cb(c,killer.side,sl,"summoned")){carrier=sl;break;}
    }
    DX cx=combat_ctx(c,killer.side,carrier);
    DEvt ke=combat_evt(); ke.actor=killer.slot; ke.target=ch;
    d_fire(&cx,"on_summon_kill",&ke,0,0);
   }
   if(d_cb(c,side,ch,"awakened") || d_ci(c,side,ch,"ultimate_turns")) d_end_ultimate(c,side,ch);
   double qout[NO]={0}; int qerr=0; character_query(c->s,side,ch,qout,&qerr); if(qerr)d_fail(c,qerr);
   int base_max=(int)qout[OUT_base_max_hp];
   d_remove_weapon(c,side,ch);
   if(!d_cstr_is(c,side,ch,"shape","")) d_remove_source_shape(c,d_entity(c,DR{side,ch}));
   DX cl=combat_ctx(c,side,ch);
   d_clear_effects(&cl,"down");
   d_clear_next_resistance(c,DR{side,ch});
   d_set_ci(c,side,ch,"down_turns",3);
   d_set_ci(c,side,ch,"harmony",kept_h);
   d_set_ci(c,side,ch,"energy",kept_e);
   d_set_ci(c,side,ch,"shield",0);
   d_set_ci(c,side,ch,"growth",0);
   d_set_ci(c,side,ch,"gaze",0);
   d_set_ci(c,side,ch,"dream",0);
   d_set_cstr(c,side,ch,"shape",nullptr);
   d_set_cb(c,side,ch,"awakened",0);
   d_set_ci(c,side,ch,"ultimate_turns",0);
   d_set_ci(c,side,ch,"max_hp",base_max);
   combat_clear_holder_flags(c,side,ch);
   d_event(c);
   if(!combat_ok(c) && !d_finished(c)) return nd;
  }
 }
 combat_victory(c);
 for(int i=0;i<nd;i++){
  int side=downed_side[i], ch=downed_slot[i];
  DX self=combat_ctx(c,side,ch);
  DEvt ev=combat_evt(); ev.actor=ch;
  d_fire(&self,"on_ally_down",&ev,0,1);
  if(d_finished(c)||d_bad(c)) break;
  d_trigger_responses(c,side,"self_downed",ch,-1);
 }
 return nd;
}
DFN void d_knockdowns(DX* c){
 while(combat_knockdown_pass(c)) {}
}
DFN int combat_harmony_previous(DX* c,int side,int ch){
 int front=d_front(c,side);
 int prev=front>=0?front:((c->s->harmony_residue_version[0]==1)?combat_last_front(c,side):-1);
 if(prev<0||prev==ch)return -1;
 return prev;
}
DFN int combat_harmony_payer(DX* c,int side,int ch){
 int prev=combat_harmony_previous(c,side,ch);
 if(prev<0||!d_alive(c,side,prev))return -1;
 if(d_collapsed(c,side,prev))return -1;
 if(d_ci(c,side,prev,"harmony")<d_harmony_cap(c))return -1;
 return prev;
}
DFN int combat_harmony_eligible(DX* c,int side,int ch){
 if(combat_active(c)!=side)return 0;
 if(d_front(c,side)==ch)return 0;
 return combat_harmony_payer(c,side,ch)>=0;
}
DFN int d_harmony_kind(DX* c){
 if(!combat_harmony_eligible(c,c->side,c->actor))return HK_NONE;
 int prev=combat_harmony_previous(c,c->side,c->actor);
 if(prev<0)return HK_NONE;
 return combat_harmony_pair(combat_attr(c,c->side,prev),combat_attr(c,c->side,c->actor));
}
DFN int combat_pending_harmony(DX* c,int* payer){
 int side=c->side, ch=c->actor;
 *payer=combat_harmony_payer(c,side,ch);
 int kind=d_harmony_kind(c);
 if(combat_harmony_eligible(c,side,ch)){
  DEvt ev=combat_evt(); ev.actor=combat_harmony_previous(c,side,ch);
  kind=d_decide(c,"harmony_kind",kind,&ev,0,0);
  if(kind<0)kind=0;
 }
 if(!kind||*payer<0){ *payer=-1; return HK_NONE; }
 return kind;
}
DFN void combat_record_delay_turn(DX* c,int side){
 int enemy=d_enemy(side);
 if(!d_zone_has(c,enemy,"delay"))return;
 if(d_pi(c,side,"delay_counted_turn")==combat_turn(c))return;
 d_set_pi(c,side,"delay_counted_turn",combat_turn(c));
 d_set_pi(c,side,"delay_turns",d_pi(c,side,"delay_turns")+1);
}
DFN void combat_notify_support(DX* c,int exclude_entity){
 int turn=combat_turn(c);
 if(d_pi(c,c->side,"support_turn")!=turn){ d_set_pi(c,c->side,"support_turn",turn); d_set_pi(c,c->side,"support_count",0); }
 d_set_pi(c,c->side,"support_count",d_pi(c,c->side,"support_count")+1);
 DEvt ev=combat_evt(); ev.actor=c->actor; ev.card=exclude_entity;
 d_fire(c,"on_support",&ev,0,0);
}
DFN void combat_apply_burn(DX* c,int enemy,int by,int source){
 int had_star=d_zone_has(c,enemy,"star");
 DX ctx=combat_ctx(c,by,source);
 int gained=0;
 if(d_zone_has(c,enemy,"burn") && d_zone_int(c,enemy,"burn","by",-1)==by){
  if(!d_zone_int(c,enemy,"burn","infinite",0)) d_zone_set(c,enemy,"burn","left",2);
  int before=d_zone_int(c,enemy,"burn","stacks",1);
  int extra=d_decide(&ctx,"burn_extra_stack",0,nullptr,0,0);
  if(extra && before<3){ int neu=before+extra; if(neu>3)neu=3; d_zone_set(c,enemy,"burn","stacks",neu); gained=neu-before; }
  d_zone_source(c,enemy,"burn",by,source);
  d_event(c);
 } else {
  d_zone_set(c,enemy,"burn","left",2);
  d_zone_set(c,enemy,"burn","by",by);
  d_zone_set(c,enemy,"burn","stacks",1);
  d_zone_set(c,enemy,"burn","applied_turn",combat_turn(c));
  d_zone_source(c,enemy,"burn",by,source);
  gained=1; d_event(c);
 }
 DEvt ev=combat_evt(); ev.target_side=enemy; ev.target=d_front(c,enemy);
 d_fire(&ctx,"on_burn_applied",&ev,0,0);
 if(gained){ DEvt ge=ev; ge.amount=gained; d_fire(&ctx,"on_burn_stacks",&ge,0,0); }
 d_fire(&ctx,"on_continuous_applied",&ev,0,0);
 if(had_star){
  int front=d_front(c,enemy);
  if(front>=0 && d_alive(c,enemy,front)) d_add_collapse(c,DR{enemy,front},1,by);
  d_event(c);
  d_fire(&ctx,"on_dissonance",&ev,0,0);
 }
}
DFN void combat_apply_dark_star(DX* c){
 int side=c->side, ch=c->actor; DR ft=d_front_target(c); int enemy=ft.side;
 d_set_pi(c,side,"dark_star_count",d_pi(c,side,"dark_star_count")+1);
 int had_burn=d_zone_has(c,enemy,"burn");
 int renew=d_decide(c,"dark_star_renewals",0,nullptr,0,0);
 d_zone_set(c,enemy,"star","left",2);
 d_zone_set(c,enemy,"star","by",side);
 d_zone_set(c,enemy,"star","renewals",renew);
 d_zone_source(c,enemy,"star",side,ch);
 d_event(c);
 if(had_burn){
  if(ft.slot>=0 && d_alive(c,enemy,ft.slot)) d_add_collapse(c,ft,1,side);
  d_event(c);
  DEvt ev=combat_evt(); ev.target_side=enemy; ev.target=ft.slot;
  d_fire(c,"on_dissonance",&ev,0,0);
 }
}
DFN void d_resolve_genesis(DX* c,int arm_delay){
 int side=c->side, ch=c->actor;
 if(!combat_ok(c)||!d_alive(c,side,ch))return;
 d_set_pi(c,side,"genesis_turn",combat_turn(c));
 int repeats=1+d_tally(c,"genesis_extra",nullptr);
 int surplus=0;
 for(int i=0;i<repeats;i++){
  if(!combat_ok(c)||!d_alive(c,side,ch))break;
  DR tgt=d_front_target(c);
  int delayed=d_zone_has(c,tgt.side,"delay");
  int sh0=combat_shield(c,tgt);
  combat_next_group(c);int loss=combat_damage(c,tgt,1+d_pi(c,side,"genesis_damage_bonus"),DR{side,ch},DK_GENESIS,0,1,0,0,0,0,DC_HARMONY);
  int dealt=loss>0 || combat_shield(c,tgt)<sh0;
  if(delayed && d_alive(c,side,ch) && combat_ok(c)){
   surplus=1; d_set_pb(c,side,"surplus",1); d_set_pb(c,side,"surplus_ever",1);
   int rec=d_front(c,side);
   if(rec>=0 && d_alive(c,side,rec)) d_energy(c,side,rec,1);
   d_event(c);
  }
  DEvt ev=combat_evt(); ev.target_side=tgt.side; ev.target=tgt.slot; ev.amount=dealt; ev.entering=ch;
  d_fire(c,"on_genesis_damage",&ev,0,0);
 }
 if(surplus){ DEvt ev=combat_evt(); ev.entering=ch; d_fire(c,"on_surplus",&ev,0,0); }
 int flowers=d_tally(c,"genesis_flowers",nullptr)+(d_pb(c,side,"extra_flower")?1:0);
 d_set_pb(c,side,"extra_flower",0);
 for(int i=0;i<flowers;i++){
  if(!combat_ok(c)||!d_alive(c,side,ch))break;
  DR tgt=d_front_target(c);
  combat_next_group(c);combat_damage(c,tgt,1,DR{side,ch},DK_FLOWER,0,1,0,0,0,0,DC_HARMONY);
  DEvt ev=combat_evt(); ev.target_side=tgt.side; ev.target=tgt.slot; ev.entering=ch;
  d_fire(c,"on_flower",&ev,0,0);
 }
 if(arm_delay){ DEvt ev=combat_evt(); ev.entering=ch; d_fire(c,"after_genesis",&ev,0,0); }
}
DFN void d_resolve_harmony(DX* c,int kind){
 if(!kind||!combat_ok(c))return;
 int side=c->side, ch=c->actor; DR ft=d_front_target(c); int enemy=ft.side;
 if(kind==HK_GENESIS) d_resolve_genesis(c,1);
 else if(kind==HK_DELAY){
  int duration=d_decide(c,"delay_duration",1,nullptr,0,0); if(duration<=0)duration=1;
  int slow=d_decide(c,"delay_slow",1,nullptr,0,0);
  int old_left=d_zone_has(c,enemy,"delay")?d_zone_int(c,enemy,"delay","left",0):0;
  int left=old_left>duration?old_left:duration;
  int slow_turn=d_zone_has(c,enemy,"delay")?d_zone_int(c,enemy,"delay","slow_turn",combat_turn(c)):combat_turn(c);
  d_zone_set(c,enemy,"delay","left",left);
  d_zone_set(c,enemy,"delay","tick_side",side);
  d_zone_set(c,enemy,"delay","slow",slow);
  d_zone_set(c,enemy,"delay","slow_turn",slow_turn);
  d_zone_set(c,enemy,"delay","by",side);
  d_event(c);
  combat_record_delay_turn(c,side);
  int hits=d_tally(c,"delay_hits",nullptr)+(d_pb(c,side,"extra_delay")?1:0);
  d_set_pb(c,side,"extra_delay",0);
  for(int i=0;i<hits;i++){
   if(d_finished(c)||d_bad(c))break;
   DR tgt=d_front_target(c);
   combat_damage(c,tgt,1,DR{side,ch},DK_DELAY,0,1,0,0,0,0,DC_HARMONY);
   DEvt ev=combat_evt(); ev.target_side=tgt.side; ev.target=tgt.slot;
   d_fire(c,"on_delay_hit",&ev,0,0);
  }
 } else if(kind==HK_BURN) combat_apply_burn(c,enemy,side,ch);
 else if(kind==HK_STAR) combat_apply_dark_star(c);
 else if(kind==HK_WEAVE){ d_zone_set(c,enemy,"weave","by",side); d_zone_set(c,enemy,"weave","expires_side",combat_active(c)); d_event(c); }
 else if(kind==HK_STAIN){
  int ends=d_decide(c,"stain_duration",1,nullptr,0,0); if(ends<1)ends=1;
  d_zone_set(c,enemy,"stain","by",side); d_zone_set(c,enemy,"stain","expires_side",combat_active(c)); d_zone_set(c,enemy,"stain","ends_left",ends);
  d_event(c);
 } else if(kind==HK_SYNC){
  d_set_used(c,side,"sync",1,1);
  DEvt ev=combat_evt(); ev.entering=ch; d_fire(c,"on_sync",&ev,0,0);
 }
}
DFN void combat_apply_enter_harmony(DX* c,int payer,int kind,int as_support,int pay){
 if(!kind||(pay&&payer<0))return;
 int side=c->side, ch=c->actor;
 if(payer>=0 && d_collapsed(c,side,payer))return;
 int before=payer>=0?d_ci(c,side,payer,"harmony"):0;
 if(pay){
  d_set_ci(c,side,payer, "harmony", before-d_harmony_cap(c));
  if(combat_last_front(c,side)==payer) combat_set_last_front(c,side,-1);
 }
 d_set_flag(c,side,ch,"has_harmonized",1,1);
 c->s->player_harmonized[side*C+ch]=1;
 c->s->op_harmony_presence[0]=2; c->s->op_harmony_slot[0]=d_intern(c,kind==1?"\xe5\x88\x9b\xe7\x94\x9f":kind==2?"\xe5\xbb\xb6\xe6\xbb\x9e":kind==3?"\xe6\xb5\x8a\xe7\x87\x83":kind==4?"\xe9\xbb\xaf\xe6\x98\x9f":kind==5?"\xe8\xa6\x86\xe7\xba\xb9":kind==6?"\xe6\xb5\xb8\xe6\x9f\x93":"\xe5\x90\x8c\xe9\xa2\x91");
 if(as_support){
  combat_set_op_flag(c,"support",1);
  int notified=combat_op_flag(c,"support_notified");d_del_op_int(c,"support_notified");if(!notified)combat_notify_support(c,-1);
 }
 d_event(c); (void)before;
 d_resolve_harmony(c,kind);
 if(combat_ok(c)&&d_alive(c,side,ch)) d_on_shape(c,"v2_harmony_resolved",nullptr);
}
DFN void d_enter(DX* c,int as_support,int resolve_harmony){
 int side=c->side, ch=c->actor;
 if(!combat_ok(c)||!d_alive(c,side,ch))return;
 if(d_front(c,side)==ch)return;
 int payer=-1, kind=combat_pending_harmony(c,&payer);
 int cur=d_front(c,side);
 if(cur>=0){ d_move_out(c,side,cur,1); if(d_front(c,side)>=0)return; }
 combat_set_last_front(c,side,-1);
 d_set_front(c,side,ch);
 d_event(c);
 if(resolve_harmony) combat_apply_enter_harmony(c,payer,kind,as_support,1);
}
DFN void combat_intercept_enter(DX* c){
 int side=c->side, ch=c->actor;
 if(!combat_ok(c)||!d_alive(c,side,ch))return;
 if(d_front(c,side)==ch)return;
 int payer=-1, kind=combat_pending_harmony(c,&payer);
 int cur=d_front(c,side);
 if(cur>=0){ d_move_out(c,side,cur,0); if(d_front(c,side)>=0)return; }
 combat_set_last_front(c,side,-1);
 d_set_front(c,side,ch);
 d_event(c);
 combat_apply_enter_harmony(c,payer,kind,0,1);
}

DFN int combat_attack_stopped(DX* c){
 int stopped=d_finished(c)||d_bad(c)||!d_alive(c,c->side,c->actor);
 if(!stopped && d_collapsed(c,c->side,c->actor)){ d_event(c); stopped=1; }
 if(stopped){ d_settle_deferred_card_marks(c); d_clear_effects(c,"battle"); }
 return stopped;
}
DFN int combat_attack_result(DX*c,const DCombatSpec*spec,int index,DR target,int*low){int side=c->side,ch=c->actor;Candidate entries[12];int n=0;entries[n++]=Candidate{(double)d_attack_value(c,side,ch,0),-100,2};int value=d_op_int(c,"sortie_attack_bonus",0);if(value)entries[n++]=Candidate{(double)value,-90,0};if(c->s->op_int_presence[2]==2)entries[n++]=Candidate{(double)d_op_int(c,"card_set_attack",0),-80,2};int bonus=spec->base.bonus;if(target.slot<0&&d_cb(c,side,ch,"awakened")&&spec->base.has_awake_player_bonus)bonus=spec->base.awake_player_bonus;if(bonus)entries[n++]=Candidate{(double)bonus,0,0};int maximum=spec->int_presence[3]==2?spec->base.low_hp_max:3;*low=target.slot>=0&&combat_hp(c,target)<=maximum;if(*low&&spec->base.low_hp_bonus)entries[n++]=Candidate{(double)spec->base.low_hp_bonus,0,0};if(c->s->op_redeem_bonus[0])entries[n++]=Candidate{(double)c->s->op_redeem_bonus[0],0,0};value=d_op_int(c,"card_bonus",0);if(value)entries[n++]=Candidate{(double)value,0,0};if(spec->base.additional_damage_bonus)entries[n++]=Candidate{(double)spec->base.additional_damage_bonus,0,0};if(index==0&&c->s->op_first_bonus[0])entries[n++]=Candidate{(double)c->s->op_first_bonus[0],-90,0};if(spec->base.harmony_bonus&&c->s->op_harmony_presence[0]==2&&c->s->string_length[c->s->op_harmony_slot[0]]>0)entries[n++]=Candidate{(double)spec->base.harmony_bonus,0,0};int zone=d_zone_penalty(c,side,ch);if(zone)entries[n++]=Candidate{(double)(-zone),1000,0};DEvt ev=combat_evt();ev.attacker_side=side;ev.attacker=ch;ev.target_side=target.side;ev.target=target.slot;ev.index=index;int result=(int)d_evaluate_entries(c,DR{side,ch},"attack.hit",0,&ev,entries,n);d_consume_applied(c,DR{side,ch},"attack.hit","attack",&ev);return result>0?result:0;}
DFN int combat_attack(DX* c,DCombatSpec spec,int index){
 if(combat_attack_stopped(c))return 0;
 int side=c->side, ch=c->actor;
 DR target=spec.has_target?spec.target:d_front_target(c);
 int enemy=target.side, tid=target.slot;
 if(spec.front_only && tid<0)return 0;
 if(tid>=0 && !d_alive(c,enemy,tid))return 0;
 d_fire(c,"before_attack",nullptr,1,0);
 if(combat_attack_stopped(c))return 0;
 if(index==0 && tid>=0){
  if(d_front(c,enemy)==tid){
   d_trigger_responses(c,enemy,"front_attacked",tid,ch);
   if(combat_attack_stopped(c))return 0;
  }
  if(!spec.has_target){
   d_trigger_responses(c,enemy,"faction_front_attacked",tid,-1);
   if(combat_attack_stopped(c))return 0;
   tid=d_front(c,enemy); target.slot=tid;
  }
  if(tid>=0){
   d_trigger_responses(c,enemy,"self_attacked",tid,-1);
   if(combat_attack_stopped(c))return 0;
   d_trigger_responses(c,enemy,"ally_attacked",tid,-1);
  }
  if(combat_attack_stopped(c))return 0;
  if(!spec.has_target){ target=d_front_target(c); enemy=target.side; tid=target.slot; }
 }
 if(spec.front_only && tid<0)return 0;
 int carrier=tid>=0?tid:d_order_slot(c,enemy,0);
 if(carrier>=0){
  DX ox=combat_ctx(c,enemy,carrier);
  DEvt ev=combat_evt(); ev.attacker_side=side; ev.attacker=ch; ev.target=tid;
  d_fire(&ox,"on_opponent_attack",&ev,0,0);
 }
 if(combat_attack_stopped(c))return 0;
 int attribute=spec.attribute?spec.attribute:combat_attr(c,side,ch);
 DEvt aev=combat_evt(); aev.attacker=ch; aev.target_side=enemy; aev.target=tid; aev.attribute=attribute;
 aev.additional=spec.additional_attack||combat_op_flag(c,"additional_attack");aev.support=combat_support(c);aev.sync_attack=spec.sync_attack;
 int additional=d_decide(c,"attack_is_additional",aev.additional,&aev,0,0)!=0;
 additional=d_decide(c,"damage_is_additional",additional,&aev,1,0)!=0;
 aev.additional=additional;
 d_fire(c,"before_attack_damage",&aev,0,0);
 if(combat_attack_stopped(c))return 0;
 combat_mark_attacked(c,side,ch);
 d_migrate_legacy(c);
 DCombatSpec work=spec; work.additional_attack=additional;
 work.base.additional_damage_bonus=additional?d_decide(c,"additional_attack_damage_bonus",0,&aev,0,0):0;
 int low=0; int attack_amount=combat_attack_result(c,&work,index,target,&low);
 d_set_op_int(c,"last_attack_amount",attack_amount,0);
 int extra_counter=0, counter_base=0;
 if(tid>=0){
  counter_base=d_attack_value(c,enemy,tid,0);
  if(d_flag(c,enemy,tid,"counter_set_attack")||c->s->flag_int_presence[d_holder(c,enemy,tid)*NF+FI_counter_set_attack]>=2){
   counter_base=d_flag(c,enemy,tid,"counter_set_attack"); d_del_flag(c,enemy,tid,"counter_set_attack");
  }
  extra_counter=d_flag(c,enemy,tid,"counter_bonus"); d_del_flag(c,enemy,tid,"counter_bonus");
 }
 int support_immunity=combat_support(c)&&!spec.support_allows_counter;
 int no_counter=spec.base.no_counter||combat_op_flag(c,"no_counter")||support_immunity
  ||(spec.no_counter_harmony&&c->s->op_harmony_presence[0]==2&&c->s->op_harmony_slot[0])
  ||(tid>=0&&d_collapsed(c,enemy,tid));
 int counter=0;
 if(tid>=0 && !no_counter){
  counter=counter_base+extra_counter-d_zone_penalty(c,enemy,tid); if(counter<0)counter=0;
 }
 int counter_additional=0;
 if(tid>=0 && !no_counter){
  DX defn=combat_ctx(c,enemy,tid);
  DEvt cev=combat_evt(); cev.target_side=side; cev.target=ch; cev.kind=DK_COMBAT;
  counter_additional=d_decide(&defn,"damage_is_additional",0,&cev,1,0)!=0;
  if(counter_additional){
   int add=d_decide(&defn,"additional_attack_damage_bonus",0,&cev,0,0);
   counter+=add; if(counter<0)counter=0;
  }
 }
 int bypass=spec.bypass;
 DEvt bev=combat_evt(); bev.attacker=ch; bev.target=tid;
 bypass=d_decide(c,"attack_bypass",bypass,&bev,0,0)!=0;
 int cut_max=bypass && tid>=0;
 int penetrate=(spec.penetrate_harmony&&c->s->op_harmony_slot[0])||(spec.penetrate_low_hp&&low);
 if(spec.penetrate_brand && tid>=0 && d_flag(c,enemy,tid,"brand")>=spec.penetrate_brand) penetrate=1;
 int overflow=0, raw_overflow=0;
 if(tid>=0 && !cut_max && !combat_eval_bool(c,DR{enemy,tid},"rule.damage_immune",0,nullptr)){
  int blocked=spec.base.ignore_shield?0:combat_shield(c,DR{enemy,tid});
  raw_overflow=attack_amount-blocked-combat_hp(c,DR{enemy,tid}); if(raw_overflow<0)raw_overflow=0;
  overflow=penetrate?raw_overflow:0;
  DEvt oev=combat_evt(); oev.attacker=ch; oev.defender_side=enemy; oev.defender=tid; oev.raw_overflow=raw_overflow; oev.index=index;
  oev.attacker_side=side; oev.effect_source_entity=d_entity(c,DR{side,ch}); oev.holder_entity=tid>=0?d_entity(c,DR{enemy,tid}):d_entity(c,DR{enemy,-1});
  overflow=d_decide(c,"combat_overflow",overflow,&oev,0,0);
  DX defn=combat_ctx(c,enemy,tid);
  overflow=d_decide(&defn,"combat_overflow",overflow,&oev,0,0);
  /* tutorial overflow lock is outside public-17 SoA coverage */
 }
 int weave_n=0; int ws[2],wa[2],wt_s[2],wt_a[2],wattr[2];
 if((attribute==ATTR_SPIRIT||attribute==ATTR_HEX) && d_zone_has(c,enemy,"weave")){
  ws[weave_n]=side; wa[weave_n]=ch; wt_s[weave_n]=enemy; wt_a[weave_n]=tid; wattr[weave_n]=attribute; weave_n++;
 }
 if(tid>=0 && !no_counter){
  int tattr=combat_attr(c,enemy,tid);
  if((tattr==ATTR_SPIRIT||tattr==ATTR_HEX) && d_zone_has(c,side,"weave")){
   ws[weave_n]=enemy; wa[weave_n]=tid; wt_s[weave_n]=side; wt_a[weave_n]=ch; wattr[weave_n]=tattr; weave_n++;
  }
 }
 if(!spec.no_combat_resources) combat_append_participant(c,side,ch);
 combat_next_group(c);
 d_event(c);
 if(c->trace){
  if(!c->trace->has_attack){ c->trace->attack=(double)attack_amount; c->trace->counter=(double)counter; c->trace->has_attack=1; }
  c->trace->attack_count += 1.0;
 }
 int target_loss=0, self_loss=0, player_damage=0;
 if(cut_max) combat_reduce_max_hp(c,DR{enemy,tid},attack_amount,DR{side,ch},0);
 else target_loss=combat_damage(c,DR{enemy,tid},attack_amount,DR{side,ch},DK_COMBAT,spec.base.ignore_shield,0,attribute,additional,0,1,DC_INFER);
 if(tid>=0) self_loss=combat_damage(c,DR{side,ch},counter,DR{enemy,tid},DK_COMBAT,0,0,0,counter_additional,support_immunity?1:0,1,DC_INFER);
 if(tid<0) player_damage=target_loss;
 if(overflow) player_damage+=combat_damage(c,DR{enemy,-1},overflow,DR{side,ch},DK_PENETRATION,0,0,attribute,additional,0,1,DC_INFER);
 if(combat_support(c) && tid>=0 && d_front(c,enemy)==tid && d_alive(c,enemy,tid)) d_add_collapse(c,DR{enemy,tid},1,side);
 c->s->op_last_attack_target_presence[0]=2; c->s->op_last_attack_target_slot[0]=enemy*256+(tid<0?255:tid);
 DEvt hev=combat_evt(); hev.loss=target_loss; hev.amount=self_loss; hev.target=tid; hev.target_side=enemy;
 d_fire(c,"after_hits",&hev,1,1);
 d_knockdowns(c); combat_victory(c);
 if(d_finished(c)||d_bad(c)){ d_settle_deferred_card_marks(c); d_clear_effects(c,"battle"); return 1; }
 DEvt ares=combat_evt(); ares.actor=ch; ares.amount=player_damage;
 d_fire(c,"on_ally_attack_resolved",&ares,0,0);
 if(d_finished(c)||d_bad(c)){ d_settle_deferred_card_marks(c); d_clear_effects(c,"battle"); return 1; }
 for(int i=0;i<weave_n;i++){
  if(!d_alive(c,ws[i],wa[i]))continue;
  if(wt_a[i]>=0 && (!d_alive(c,wt_s[i],wt_a[i]) || d_front(c,wt_s[i])!=wt_a[i])) continue;
  DX src=combat_ctx(c,ws[i],wa[i]);
  int amt=d_decide(&src,"weave_damage",2,nullptr,0,0);
  combat_next_group(c);combat_damage(c,DR{wt_s[i],wt_a[i]},amt,DR{ws[i],wa[i]},DK_OVERLAY,0,1,wattr[i],1,0,0,DC_HARMONY);
  if(d_finished(c)||d_bad(c)){ d_settle_deferred_card_marks(c); d_clear_effects(c,"battle"); return 1; }
 }
 if(tid>=0 && self_loss>0 && d_alive(c,side,ch) && combat_ok(c)){ DX sx=combat_ctx(c,side,ch); d_on_shape(&sx,"v2_combat_hp_lost",nullptr); }
 if(tid>=0 && target_loss>0 && d_alive(c,enemy,tid) && combat_ok(c)){ DX dx=combat_ctx(c,enemy,tid); d_on_shape(&dx,"v2_combat_hp_lost",nullptr); }
 if(combat_support(c) && !combat_op_flag(c,"support_resolved")){
  combat_set_op_flag(c,"support_resolved",1);
  DEvt ev=combat_evt(); ev.actor=ch; d_fire(c,"on_support_resolved",&ev,1,0);
 }
 d_settle_deferred_card_marks(c); d_clear_effects(c,"battle");
 return 1;
}
DFN void combat_notify_opponent_sortie(DX* c){
 if(!combat_ok(c)||!d_alive(c,c->side,c->actor))return;
 int enemy=d_enemy(c->side); int carrier=d_order_slot(c,enemy,0);
 if(carrier<0)return;
 DX ox=combat_ctx(c,enemy,carrier);
 DEvt ev=combat_evt(); ev.attacker_side=c->side; ev.attacker=c->actor;
 d_fire(&ox,"on_opponent_sortie_finished",&ev,0,0);
}
DFN void combat_finish_sortie(DX* c,DCombatSpec spec,int stay_bench){
 if(combat_attack_stopped(c))return;
 d_fire(c,"before_hits",nullptr,1,0);
 int attacked=0, hits=spec.base.hits>0?spec.base.hits:1;
 for(int index=0;index<hits;index++){
  if(d_finished(c)||!d_alive(c,c->side,c->actor)||(!stay_bench && d_front(c,c->side)!=c->actor)) break;
  if(!combat_attack(c,spec,index)) break;
  attacked=1;
 }
 if(!attacked||d_finished(c)||!d_alive(c,c->side,c->actor))return;
 if(spec.shield_after_shape && !d_cstr_is(c,c->side,c->actor,"shape",""))
  d_shield(c,DR{c->side,c->actor},spec.shield_after_shape);
 if(stay_bench || d_front(c,c->side)==c->actor){
  d_fire(c,"after_sortie",nullptr,1,0);
  d_on_shape(c,"v2_sortie_finished",nullptr);
  d_fire(c,"ally_sortie",nullptr,0,0);
  if(combat_ok(c)&&d_alive(c,c->side,c->actor)&&d_flag(c,c->side,c->actor,"return_after")){
   d_del_flag(c,c->side,c->actor,"return_after");
   d_move_out(c,c->side,c->actor,0);
   d_draw(c,c->side,1,-1);
  }
 }
 combat_notify_opponent_sortie(c);
}
DFN DCombatSpec combat_spec_from_attack(DX* c,const DAttack* in){
 DCombatSpec spec{}; DAttack z{}; z.bonus=z.ignore_shield=z.no_counter=z.no_attack=z.hits=z.return_after=z.draw_after=0;
 z.low_hp_bonus=0; z.low_hp_max=3; z.awake_player_bonus=0; z.has_awake_player_bonus=0; z.harmony_bonus=0; z.additional_damage_bonus=0;
 spec.base=in?*in:z; if(spec.base.hits<=0) spec.base.hits=1; if(!spec.base.low_hp_max) spec.base.low_hp_max=3;
 spec.stay_bench=spec.front_only=spec.additional_attack=spec.bypass=0;
 spec.penetrate_harmony=spec.penetrate_low_hp=spec.penetrate_brand=0;
 spec.no_combat_resources=spec.support_allows_counter=spec.no_counter_harmony=0;
 spec.sync_attack=spec.shield_after_shape=spec.attribute=spec.has_target=0;
 for(int i=0;i<NSI;i++) spec.int_presence[i]=0; for(int i=0;i<NSB;i++) spec.bool_presence[i]=0; spec.has_attribute=0;
 spec.target=d_op_target(c,"target_id");
 if(spec.target.side>=0) spec.has_target=1;
 spec.additional_attack=combat_op_flag(c,"additional_attack");
 spec.stay_bench=0;
 if(in){
  spec.int_presence[0]=2; spec.int_presence[1]=2; spec.int_presence[2]=2; spec.int_presence[3]=2;
  spec.int_presence[5]=2; spec.int_presence[6]=2; if(in->has_awake_player_bonus) spec.int_presence[4]=2;
  spec.bool_presence[0]=2; spec.bool_presence[1]=2;
 }
 if(spec.additional_attack) spec.bool_presence[3]=2;
 if(spec.stay_bench) spec.bool_presence[2]=2;
 if(spec.has_target) spec.bool_presence[5]=spec.bool_presence[5];
 return spec;
}
DFN void combat_run_sortie(DX* c,DCombatSpec spec){
 if(combat_attack_stopped(c))return;
 if(d_op_int(c,"sortie_attack_bonus",-2147483647)==-2147483647) d_set_op_int(c,"sortie_attack_bonus",0,0);
 if(c->s->response_intercept_presence[0]==2 && c->s->response_intercept[0]){
  combat_intercept_enter(c);
  if(combat_ok(c)&&d_alive(c,c->side,c->actor)){ DEvt ev=combat_evt(); ev.actor=c->actor; ev.response=1; d_fire(c,"on_sortie_ready",&ev,0,0); }
  return;
 }
 spec.stay_bench=spec.stay_bench||d_decide(c,"sortie_from_bench",0,nullptr,1,0)!=0;
 if(spec.stay_bench) spec.bool_presence[2]=2;
 if(spec.stay_bench) spec.base.no_counter=1; else d_enter(c,1,1);
 if(combat_attack_stopped(c))return;
 if(combat_ok(c)&&d_alive(c,c->side,c->actor)){ DEvt ev=combat_evt(); ev.actor=c->actor; ev.response=0; d_fire(c,"on_sortie_ready",&ev,0,0); }
 if(combat_attack_stopped(c))return;
 if(combat_phase(c)==2){
  d_set_pending_sortie(c,&spec);
  if(spec.has_target) d_set_op_target(c,"target_id",spec.target);
  return;
 }
 combat_finish_sortie(c,spec,spec.stay_bench);
}
DFN void d_sortie(DX* c,const DAttack* in){ combat_run_sortie(c,combat_spec_from_attack(c,in)); }
DFN void combat_sortie_ex(DX* c,const DCombatSpec* spec){
 if(!spec){ d_sortie(c,nullptr); return; }
 DCombatSpec copy=*spec; combat_run_sortie(c,copy);
}
DFN void d_resume_sortie(DX* c){
 if(c->s->op_pending_sortie_presence[0]!=2)return;
 DCombatSpec spec{}; if(!d_load_pending_sortie(c,&spec)) return;
 int stay=spec.stay_bench;
 c->s->op_pending_sortie_presence[0]=0;
 combat_finish_sortie(c,spec,stay);
}
DFN void combat_additional_attack(DX* c,int attribute,DR target,int has_target,int sync_attack,int support_actor,int front_only){
 if(combat_attack_stopped(c))return;
 if(front_only && d_front(c,d_enemy(c->side))<0)return;
 int borrowed=-1;
 if(sync_attack && support_actor>=0 && support_actor!=c->actor && d_alive(c,c->side,support_actor)
    && d_entity(c,DR{c->side,support_actor})!=d_entity(c,DR{c->side,c->actor})) borrowed=support_actor;
 CombatOpSnap parent; combat_save_op(c,&parent);
 c->card=-1;c->target=DR{-1,-2};c->option_slot=-1;c->effect_source_slot=-1;d_open_operation(c);
 c->s->op_side[0]=(signed char)c->side; c->s->op_actor_slot[0]=(short)c->actor; c->s->op_card_index[0]=-1;
 c->s->op_participant_count[0]=0; c->s->op_harmony_presence[0]=0;
 c->s->op_first_bonus[0]=0; c->s->op_redeem_bonus[0]=0; d_set_op_int(c,"sortie_attack_bonus",0,0);
 if(sync_attack) combat_set_op_flag(c,"support",1);
 DCombatSpec spec=combat_spec_from_attack(c,nullptr);
 spec.attribute=attribute; spec.has_target=has_target; spec.target=target;
 spec.base.no_counter=1; spec.no_combat_resources=1; spec.additional_attack=1; spec.sync_attack=sync_attack; spec.front_only=front_only;
 int exclude=borrowed>=0?d_entity(c,DR{c->side,borrowed}):-1;
 if(sync_attack) combat_notify_support(c,exclude);
 int attacked=combat_attack(c,spec,0);
 if(attacked && borrowed>=0 && d_alive(c,c->side,borrowed)){
  DX bx=combat_ctx(c,c->side,borrowed);
  DEvt ev=combat_evt(); ev.actor=borrowed;
  d_fire(&bx,"on_support",&ev,1,0);
  d_fire(&bx,"on_support_resolved",&ev,1,0);
 }
 if(attacked) combat_notify_opponent_sortie(c);
 combat_restore_op(c,&parent);
}

DFN void combat_rose_followup(DX* c,DR source,int target_side,int cid,int dealt);
DFN int combat_boost_continuous(DX* c,int target_side,int by,int amount){
 if(amount<=0)return amount;
 if(amount<=0)return amount;
 int zaowu=d_hero(c,by,"zaowu");
 if(zaowu<0||!d_alive(c,by,zaowu))return amount;
 if(!d_zone_has(c,target_side,"burn"))return amount;
 int kinds=1;
 for(int i=0;i<COMBAT_D;i++) if(c->s->zone_dot_valid[target_side*COMBAT_D+i] && c->s->zone_dot_stacks[target_side*COMBAT_D+i]>0) kinds++;
 return amount+(amount/4)*kinds;
}
DFN int combat_living(DX* c,int side,int* out,int cap,int include_player){
 int n=0; if(include_player){ if(n<cap) out[n++]=-1; }
 for(int i=0;i<d_order_count(c,side);i++){
  int ch=d_order_slot(c,side,i);
  if((!include_player&&d_cb(c,side,ch,"summoned"))||d_ci(c,side,ch,"hp")<=0) continue;
  if(n<cap) out[n++]=ch;
 }
 return n;
}
DFN void combat_distribute(DX* c,int target_side,int amount,DR source,int by,int ignore_shield,int dclass,int include_roles,int rose){
 if(!combat_ok(c))return;
 if(amount<0)amount=0;
 amount=combat_boost_continuous(c,target_side,by,amount);
 int ids[16]; int n=combat_living(c,target_side,ids,16,include_roles);
 if(amount<=0||n<=0)return;
 int buckets[16]; for(int i=0;i<n;i++) buckets[i]=0;
 bmt_state_t random;d_random_begin(c,&random,0);for(int i=0;i<amount;i++){int pick=d_random_index(c,&random,n);if(pick<0||pick>=n){d_fail(c,1);return;}buckets[pick]++;}d_random_end(c,&random);
 combat_next_group(c);
 for(int i=0;i<n;i++){
  if(!buckets[i]||!combat_ok(c))continue;
  int dealt=combat_damage(c,DR{target_side,ids[i]},buckets[i],source,DK_DAMAGE,ignore_shield,0,0,0,0,0,dclass);
  if(rose&&combat_ok(c))combat_rose_followup(c,source,target_side,ids[i],buckets[i]);
  if(d_finished(c)||d_bad(c))return;
 }
 d_knockdowns(c); combat_victory(c);
}
DFN void combat_rose_followup(DX* c,DR source,int target_side,int cid,int dealt){
 if(source.slot<0||!d_cstr_is(c,source.side,source.slot,"shape","A08")||dealt<=0)return;
 if(combat_hp(c,DR{target_side,cid})>0) combat_reduce_max_hp(c,DR{target_side,cid},dealt,source,0);
 if(combat_ok(c)&&d_alive(c,source.side,source.slot)) d_heal(c,DR{source.side,source.slot},dealt,source);
}
DFN void combat_deal_one(DX* c,int target_side,int amount,DR source,int by,int ignore_shield){
 if(!combat_ok(c))return;
 amount=combat_boost_continuous(c,target_side,by,amount<0?0:amount);
 int ids[16]; int n=combat_living(c,target_side,ids,16,0);
 if(amount<=0||n<=0)return;
 int pick=d_randbelow(c,n); if(pick<0||pick>=n){d_fail(c,1);return;}
 combat_next_group(c);combat_damage(c,DR{target_side,ids[pick]},amount,source,DK_DAMAGE,ignore_shield,1,0,0,0,0,DC_INFER);
}
DFN DR combat_dot_source(DX* c,int side,int i,int by){
 int slot=c->s->zone_dot_source_slot[side*COMBAT_D+i];
 DR r=combat_parse_ref(c,slot); if(r.side<0){ r.side=by; r.slot=-1; } return r;
}
DFN void combat_trigger_dot(DX* c,int target_side,int i){
 if(!c->s->zone_dot_valid[target_side*COMBAT_D+i])return;
 int kind=c->s->zone_dot_kind[target_side*COMBAT_D+i];
 if(kind==DOT_ETCH||kind==DOT_VENOM)return;
 int stacks=c->s->zone_dot_stacks[target_side*COMBAT_D+i];
 int by=c->s->zone_dot_by[target_side*COMBAT_D+i];
 DR source=combat_dot_source(c,target_side,i,by);
 if(kind==DOT_GUARD) combat_deal_one(c,target_side,1,source,by,1);
 else combat_distribute(c,target_side,stacks,source,by,0,DC_INFER,kind==DOT_NIGHTMARE,kind==DOT_NIGHTMARE);
}
DFN void combat_tick_owner_end(DX* c,int side){
 if(!combat_ok(c))return;
 for(int target_side=0; target_side<2; target_side++){
  if(d_zone_has(c,target_side,"burn") && d_zone_int(c,target_side,"burn","by",-1)==side){
   int stacks=d_zone_int(c,target_side,"burn","stacks",1);
   int source_slot=d_zone_int(c,target_side,"burn","source",-1);DR source=source_slot>=0?combat_parse_ref(c,source_slot):DR{side,-2};
   combat_distribute(c,target_side,stacks,source,side,0,DC_HARMONY,0,0);
   if(!d_zone_int(c,target_side,"burn","infinite",0)){
    int left=d_zone_int(c,target_side,"burn","left",1)-1;
    if(left<=0) d_zone_remove(c,target_side,"burn"); else d_zone_set(c,target_side,"burn","left",left);
   }
   if(combat_ok(c)) d_event(c);
   if(d_finished(c)||d_bad(c))return;
  }
  for(int i=0;i<COMBAT_D;i++){
   if(!c->s->zone_dot_valid[target_side*COMBAT_D+i]||c->s->zone_dot_by[target_side*COMBAT_D+i]!=side) continue;
   combat_trigger_dot(c,target_side,i);
   if(d_finished(c)||d_bad(c))return;
   if(c->s->zone_dot_left_presence[target_side*COMBAT_D+i]==2){
    int left=c->s->zone_dot_left[target_side*COMBAT_D+i]-1;
    c->s->zone_dot_left[target_side*COMBAT_D+i]=left;
    if(left<=0){ c->s->zone_dot_valid[target_side*COMBAT_D+i]=0; d_event(c); }
   }
  }
 }
}
DFN void combat_halve_dots(DX* c,int side){
 for(int target_side=0; target_side<2; target_side++){
  for(int i=0;i<COMBAT_D;i++){
   if(!c->s->zone_dot_valid[target_side*COMBAT_D+i]) continue;
   if(c->s->zone_dot_by[target_side*COMBAT_D+i]!=side) continue;
   int kind=c->s->zone_dot_kind[target_side*COMBAT_D+i];
   if(kind!=DOT_NIGHTMARE) continue;
   if(c->s->zone_dot_applied_turn_presence[target_side*COMBAT_D+i]==2 && c->s->zone_dot_applied_turn[target_side*COMBAT_D+i]==combat_turn(c)) continue;
   int stacks=c->s->zone_dot_stacks[target_side*COMBAT_D+i]/2;
   c->s->zone_dot_stacks[target_side*COMBAT_D+i]=stacks;
   if(stacks<=0){ c->s->zone_dot_valid[target_side*COMBAT_D+i]=0; d_event(c); } else d_event(c);
  }
 }
}
'''


def emit_source(layout=None):
    from .domain_soa_schema import DomainSoALayout
    from .domain_soa_lifecycle_helpers import operation_snapshot_source
    source=_SOURCE
    start=source.index('struct CombatOpSnap {')
    stop=source.index('DFN int combat_ok',start)
    snapshot=operation_snapshot_source(layout or DomainSoALayout()).replace('DOpSnap','CombatOpSnap').replace('life_save_op','combat_save_op').replace('life_restore_op','combat_restore_op')
    source=source[:start]+snapshot+source[stop:]
    start=source.index('DFN void combat_save_op',source.index('DFN int combat_ok'))
    stop=source.index('DFN void combat_append_participant',start)
    return source[:start]+source[stop:]
