"""Direct C++ lifecycle source for turn, operation, response and choice flow."""
from .domain_soa_schema import CHOICE_KINDS

supports_full_duel_rules = False
CHOICE_INDEX = {name: i for i, name in enumerate(CHOICE_KINDS)}


def emit_source(layout=None):
    from .domain_soa_lifecycle_helpers import operation_snapshot_source
    from .domain_soa_schema import DomainSoALayout
    core=_LIFE_CORE
    begin=core.index("struct DOpSnap {");end=core.index("DFN void life_clear_op",begin)
    core=core[:begin]+operation_snapshot_source(layout or DomainSoALayout())+core[end:]
    text = "\n".join((core, _LIFE_OP, _LIFE_PLAY, _LIFE_TURN, _LIFE_CHOOSE))
    if any(ord(ch) > 127 for ch in text):
        raise ValueError("lifecycle source must stay ASCII")
    return text


_LIFE_CORE = r'''
DFN int life_stop(DX* x){return d_bad(x)||d_finished(x);}
DFN int life_phase(DX* x){return x->s->phase[0];}
DFN void life_set_phase(DX* x,int p){x->s->phase[0]=(short)p;}
DFN int life_lock(DX* x){return x->s->response_lock_presence[0]==2&&x->s->response_lock[0];}
DFN void life_set_lock(DX* x,int v){x->s->response_lock_presence[0]=2;x->s->response_lock[0]=v?1:0;}
DFN int life_intercept(DX* x){return x->s->response_intercept_presence[0]==2&&x->s->response_intercept[0];}
DFN void life_set_intercept(DX* x,int v){x->s->response_intercept_presence[0]=v?2:0;x->s->response_intercept[0]=v?1:0;}
DFN int life_draining(DX* x){return x->s->draining_presence[0]==2&&x->s->draining[0];}
DFN void life_set_draining(DX* x,int v){x->s->draining_presence[0]=v?2:0;x->s->draining[0]=v?1:0;}
DFN int life_op_present(DX* x){return x->s->operation_present[0]!=0;}
DFN int life_choice_present(DX* x){return x->s->pending_choice_present[0]!=0;}
DFN int life_carrier(DX* x,int side){int n=d_order_count(x,side);return n>0?d_order_slot(x,side,0):-1;}
DFN DR life_player(int side){return DR{side,-1};}
DFN DR life_char(int side,int slot){return DR{side,slot};}
DFN int life_hand_n(DX* x,int side){return x->s->zone_count[side*4+0];}
DFN int life_hand_at(DX* x,int side,int i){return x->s->zone_card_index[(side*4+0)*Z+i];}
DFN int life_deck_n(DX* x,int side){return x->s->zone_count[side*4+1];}
DFN int life_deck_at(DX* x,int side,int i){return x->s->zone_card_index[(side*4+1)*Z+i];}
DFN int life_discard_n(DX* x,int side){return x->s->zone_count[side*4+2];}
DFN int life_discard_at(DX* x,int side,int i){return x->s->zone_card_index[(side*4+2)*Z+i];}
DFN int life_card_str_is(DX* x,int card,const char* key,const char* value){
 if(card<0||card>=K||!x->s->card_valid[card])return 0;
 int field=-1;
 if(d_streq(key,"type"))field=KS_type;
 else if(d_streq(key,"response"))field=KS_response;
 else if(d_streq(key,"response_faction"))field=KS_response_faction;
 else if(d_streq(key,"character_id"))field=KS_character_id;
 else if(d_streq(key,"attack_mode"))field=KS_attack_mode;
 else return d_card_str_is(x,card,key,value);
 int p=x->s->card_str_presence[card*KS+field];
 if(p==2)return d_eq(x,x->s->card_str_slot[card*KS+field],value);
 if(p==3)return d_card_str_is(x,card,key,value);
 return 0;
}
DFN int life_card_has_int(DX* x,int card,int field){
 return card>=0&&card<K&&x->s->card_valid[card]&&x->s->card_int_presence[card*KI+field]>=2;
}
DFN int life_owner_slot(DX*x,int side,int card){DX own=*x;own.side=side;return d_card_owner(&own,card);}

DFN void life_clear_choice(DX* x){
 x->s->pending_choice_present[0]=0; x->s->choice_card_count[0]=0;
 x->s->choice_continuation_presence[0]=0; x->s->choice_actor_presence[0]=0;
}
struct DOpSnap {
 unsigned char present; signed char side; short actor; int card;
 unsigned char target_p; int target; unsigned char harmony_p; int harmony;
 int first_bonus,redeem_bonus; unsigned char record;
 unsigned char flag_p[16]; unsigned char flag_v[16];
 unsigned char int_p[16]; int int_v[16];
 unsigned char pending_p,stay,bonus_p; int bonus;
 unsigned char str_p; int str_s; unsigned char option_p; int option_s;
 unsigned char last_p; int last_s; unsigned char mark_p; int mark_d,mark_src;
 short part_n; signed char part_side[8]; short part_slot[8];
 short choice_n; int choice_i[32];
 int dx_side,dx_actor,dx_card; DR dx_target;
};
DFN void life_save_op(DX* x,DOpSnap* o){
 DS* s=x->s; o->present=s->operation_present[0];
 o->side=s->op_side[0]; o->actor=s->op_actor_slot[0]; o->card=s->op_card_index[0];
 o->target_p=s->op_target_presence[0]; o->target=s->op_target_slot[0];
 o->harmony_p=s->op_harmony_presence[0]; o->harmony=s->op_harmony_slot[0];
 o->first_bonus=s->op_first_bonus[0]; o->redeem_bonus=s->op_redeem_bonus[0]; o->record=s->op_record_candidate[0];
 for(int i=0;i<NOPBOOL&&i<16;i++){o->flag_p[i]=s->op_flag_presence[i];o->flag_v[i]=s->op_flag_value[i];}
 for(int i=0;i<NOPINT&&i<16;i++){o->int_p[i]=s->op_int_presence[i];o->int_v[i]=s->op_int_value[i];}
 o->pending_p=s->op_pending_sortie_presence[0]; o->stay=s->op_pending_stay_bench[0];
 o->bonus_p=s->op_pending_bonus_presence[0]; o->bonus=s->op_pending_bonus[0];
 o->str_p=s->op_str_presence[0]; o->str_s=s->op_str_slot[0];
 o->option_p=s->op_option_id_presence[0]; o->option_s=s->op_option_id_slot[0];
 o->last_p=s->op_last_attack_target_presence[0]; o->last_s=s->op_last_attack_target_slot[0];
 o->mark_p=s->op_card_mark_presence[0]; o->mark_d=s->op_card_mark_delta[0]; o->mark_src=s->op_card_mark_source_slot[0];
 o->part_n=s->op_participant_count[0]; if(o->part_n>8)o->part_n=8;
 for(int i=0;i<o->part_n;i++){o->part_side[i]=s->op_participant_side[i];o->part_slot[i]=s->op_participant_slot[i];}
 o->choice_n=s->op_choice_card_count[0]; if(o->choice_n>32)o->choice_n=32;
 for(int i=0;i<o->choice_n;i++)o->choice_i[i]=s->op_choice_card_index[i];
 o->dx_side=x->side; o->dx_actor=x->actor; o->dx_card=x->card; o->dx_target=x->target;
}
DFN void life_restore_op(DX* x,const DOpSnap* o){
 DS* s=x->s; s->operation_present[0]=o->present;
 s->op_side[0]=o->side; s->op_actor_slot[0]=o->actor; s->op_card_index[0]=o->card;
 s->op_target_presence[0]=o->target_p; s->op_target_slot[0]=o->target;
 s->op_harmony_presence[0]=o->harmony_p; s->op_harmony_slot[0]=o->harmony;
 s->op_first_bonus[0]=o->first_bonus; s->op_redeem_bonus[0]=o->redeem_bonus; s->op_record_candidate[0]=o->record;
 for(int i=0;i<NOPBOOL&&i<16;i++){s->op_flag_presence[i]=o->flag_p[i];s->op_flag_value[i]=o->flag_v[i];}
 for(int i=0;i<NOPINT&&i<16;i++){s->op_int_presence[i]=o->int_p[i];s->op_int_value[i]=o->int_v[i];}
 s->op_pending_sortie_presence[0]=o->pending_p; s->op_pending_stay_bench[0]=o->stay;
 s->op_pending_bonus_presence[0]=o->bonus_p; s->op_pending_bonus[0]=o->bonus;
 s->op_str_presence[0]=o->str_p; s->op_str_slot[0]=o->str_s;
 s->op_option_id_presence[0]=o->option_p; s->op_option_id_slot[0]=o->option_s;
 s->op_last_attack_target_presence[0]=o->last_p; s->op_last_attack_target_slot[0]=o->last_s;
 s->op_card_mark_presence[0]=o->mark_p; s->op_card_mark_delta[0]=o->mark_d; s->op_card_mark_source_slot[0]=o->mark_src;
 s->op_participant_count[0]=o->part_n;
 for(int i=0;i<o->part_n;i++){s->op_participant_side[i]=o->part_side[i];s->op_participant_slot[i]=o->part_slot[i];}
 s->op_choice_card_count[0]=o->choice_n;
 for(int i=0;i<o->choice_n;i++)s->op_choice_card_index[i]=o->choice_i[i];
 x->side=o->dx_side; x->actor=o->dx_actor; x->card=o->dx_card; x->target=o->dx_target;
}
DFN void life_clear_op(DX* x){
 DS* s=x->s; s->operation_present[0]=0; s->op_card_index[0]=-1; s->op_target_presence[0]=0;
 s->op_pending_sortie_presence[0]=0; s->op_participant_count[0]=0; s->op_choice_card_count[0]=0; s->op_card_mark_presence[0]=0;
 for(int i=0;i<NOPBOOL;i++){s->op_flag_presence[i]=0;s->op_flag_value[i]=0;}
 for(int i=0;i<NOPINT;i++){s->op_int_presence[i]=0;s->op_int_value[i]=0;}
 life_clear_choice(x); if(life_phase(x)==2)life_set_phase(x,1);
}
'''
_LIFE_OP = r'''
DFN int life_form_instant(DX* x,int side,int card){
 if(!life_card_str_is(x,card,"type","form"))return 0;
 int n=d_order_count(x,side);
 for(int i=0;i<n;i++){
  int ch=d_order_slot(x,side,i);
  if(d_alive(x,side,ch)&&d_cb(x,side,ch,"awakened")&&d_template(x,side,ch)==ID_zero)return 1;
 }
 int owner=life_owner_slot(x,side,card);
 return owner>=0&&d_template(x,side,owner)==ID_zero&&d_alive(x,side,owner)&&d_shape_present(x,side,owner);
}
DFN int life_card_cost(DX* x,int side,int actor,int card){
 int limit=d_card_int(x,card,"spend_all_ap"); int ap=d_pi(x,side,"ap");
 int base=limit>0?(ap<limit?ap:limit):d_card_int(x,card,"cost");
 if(d_card_bool(x,card,"free_if_surplus")&&d_pb(x,side,"surplus"))base=0;
 int v=(int)d_evaluate(x,life_char(side,actor),"card.cost",(double)base,nullptr);
 return v<0?0:v;
}
DFN int life_plays_instant(DX* x,int side,int actor,int card){
 int base=d_card_bool(x,card,"instant")||life_form_instant(x,side,card)||d_card_makes_instant(x,side,actor,card);
 return (int)d_evaluate(x,life_char(side,actor),"card.instant",(double)base,nullptr)!=0;
}
DFN int life_free_instant(DX* x,int side,int actor,int card){
 return life_card_cost(x,side,actor,card)>0&&life_plays_instant(x,side,actor,card)&&!d_used(x,side,"instant");
}
DFN int life_ap_payment(DX* x,int side,int actor,int card){
 int ordinary=life_free_instant(x,side,actor,card)?0:life_card_cost(x,side,actor,card);
 return ordinary+d_card_int(x,card,"additional_ap_cost");
}
DFN int life_can_afford(DX*x,int side,int actor,int card){DX own=*x;own.side=side;own.actor=actor;own.card=card;if(d_card_int(&own,card,"spend_all_ap")>0&&d_pi(&own,side,"ap")<=0)return 0;int ok=d_decide(&own,"can_play",1,nullptr,1);return ok&&d_pi(&own,side,"ap")>=life_ap_payment(&own,side,actor,card);}
DFN void life_pay_ap(DX* x,int side,int actor,int card,int* paid){
 *paid=life_ap_payment(x,side,actor,card);if(life_free_instant(x,side,actor,card))d_set_used(x,side,"instant",1,1); d_set_pi(x,side,"ap",d_pi(x,side,"ap")-*paid);
 d_consume_applied(x,life_char(side,actor),"card.instant","card_use",nullptr);
}
DFN void life_pay_option(DX* x,int response){
 if(response||x->s->op_option_id_presence[0]!=2)return;
 if(d_eq(x,x->s->op_option_id_slot[0],"base"))return;
 int j=d_ci(x,x->side,x->actor,"jingu"); if(j<3){d_fail(x,1);return;} d_set_ci(x,x->side,x->actor,"jingu",j-3);
}
DFN void life_zero_evt(DEvt* ev){
 ev->actor=-1; ev->card=-1; ev->attacker_side=-1; ev->attacker=-1; ev->defender_side=-1; ev->defender=-1;
 ev->target_side=-1; ev->target=-1; ev->amount=0; ev->loss=0; ev->absorbed=0; ev->raw_overflow=0;
 ev->kind=0; ev->attribute=0; ev->additional=0; ev->index=0; ev->response=0; ev->entering=0;
}
DFN void life_queue_down(DX* x,int side,int target){
 int n=x->s->down_response_count[0]; if(n<0||n>=DRMAX){d_fail(x,2);return;}
 x->s->down_response_side[n]=(signed char)side; x->s->down_response_target_slot[n]=x->s->character_id_slot[side*C+target];
 x->s->down_response_count[0]=(short)(n+1);
}
DFN void life_flush_down(DX*x){int n=x->s->down_response_count[0],sides[DRMAX],targets[DRMAX];if(n<0||n>DRMAX){d_fail(x,1);return;}for(int j=0;j<n;j++){sides[j]=x->s->down_response_side[j];targets[j]=x->s->down_response_target_slot[j];}x->s->down_response_count[0]=0;for(int j=0;j<n;j++){int actor=-1;for(int a=0;a<x->s->player_character_count[sides[j]];a++)if(qslots(x->s,targets[j],x->s->character_id_slot[sides[j]*C+a],x->s->row_error))actor=a;if(actor>=0)d_trigger_responses(x,sides[j],"self_downed",actor,-1);}}
DFN void life_resume_pending_sortie(DX* x){
 if(life_stop(x)||x->s->op_pending_sortie_presence[0]!=2)return; d_resume_sortie(x);
}
DFN void life_resume_turn_start(DX* x){
 if(x->s->turn_start_pending_presence[0]!=2)return;
 int side=x->s->turn_start_pending_side[0]; x->s->turn_start_pending_presence[0]=0; d_save_turn_start(x,side);
}
DFN void life_mark_turn_start(DX* x,int side){
 x->s->turn_start_pending_presence[0]=2; x->s->turn_start_pending_side[0]=(signed char)side;
}
DFN void d_drain(DX* x){
 if(life_draining(x)||life_lock(x)||life_op_present(x))return;
 if(d_bad(x))return;
 if(d_finished(x)){x->s->deferred_count[0]=0;x->s->deferred_presence[0]=2;return;}
 life_set_draining(x,1);
 int had=x->s->deferred_presence[0]==2;
 while(x->s->deferred_presence[0]&&x->s->deferred_count[0]>0&&!life_stop(x)&&life_phase(x)!=2){
  had=1;
  int n=x->s->deferred_count[0]; int side=x->s->deferred_side[0]; int actor=x->s->deferred_actor_slot[0];
  int entity=x->s->deferred_entity_slot[0]; int kind_slot=x->s->deferred_kind_slot[0];int deferred_card=x->s->deferred_card_presence[0]==2?x->s->deferred_card_index[0]:-1;int deferred_attribute=x->s->deferred_attribute_presence[0]==2?x->s->deferred_attribute[0]+1:0;int deferred_support=x->s->deferred_support_presence[0]==2?x->s->deferred_support_actor[0]:-1;int deferred_support_entity=x->s->deferred_support_entity[0];
  for(int i=0;i<n-1;i++){
   x->s->deferred_side[i]=x->s->deferred_side[i+1]; x->s->deferred_actor_slot[i]=x->s->deferred_actor_slot[i+1];
   x->s->deferred_entity_slot[i]=x->s->deferred_entity_slot[i+1]; x->s->deferred_kind_slot[i]=x->s->deferred_kind_slot[i+1];x->s->deferred_card_presence[i]=x->s->deferred_card_presence[i+1];x->s->deferred_card_index[i]=x->s->deferred_card_index[i+1];x->s->deferred_attribute_presence[i]=x->s->deferred_attribute_presence[i+1];x->s->deferred_attribute[i]=x->s->deferred_attribute[i+1];x->s->deferred_support_presence[i]=x->s->deferred_support_presence[i+1];x->s->deferred_support_actor[i]=x->s->deferred_support_actor[i+1];x->s->deferred_support_entity[i]=x->s->deferred_support_entity[i+1];
  }
  x->s->deferred_count[0]=(short)(n-1);
  if(!d_alive(x,side,actor)||d_entity(x,life_char(side,actor))!=entity)continue;
  x->side=side; x->actor=actor; x->card=-1; x->target=DR{side,-2};
  d_open_operation(x); d_set_op_int(x,"additional_attack",1,1);
  DEvt ev; life_zero_evt(&ev); ev.actor=actor;ev.amount=kind_slot;ev.deferred_kind_slot=kind_slot;ev.deferred_attribute=deferred_attribute;ev.card=deferred_card;ev.deferred_support_actor=deferred_support>=0&&deferred_support<C&&d_entity(x,DR{side,deferred_support})==deferred_support_entity?deferred_support:-1;ev.deferred_support_entity=deferred_support_entity;
  d_fire(x,"on_deferred_action",&ev,1,0);
  if(life_phase(x)!=2&&!life_stop(x))d_finish_operation(x);
 }
 life_set_draining(x,0);
 if(d_finished(x)||(had&&x->s->deferred_count[0]<=0)){x->s->deferred_count[0]=0;x->s->deferred_presence[0]=2;}
}
DFN void life_share_resources(DX* x){
 int n=x->s->op_participant_count[0]; if(n<=0)return;
 int skip=d_op_int(x,"skip_combat_harmony",0); int gain_energy=1; int gain_harmony=!skip;
 int live_side[8]; int live_slot[8]; int live_n=0;
 for(int i=0;i<n&&live_n<8;i++){
  int ps=x->s->op_participant_side[i]; int pc=x->s->op_participant_slot[i];
  if(d_alive(x,ps,pc)){live_side[live_n]=ps; live_slot[live_n]=pc; live_n++;}
 }
 if(!live_n||(!gain_energy&&!gain_harmony))return;
 int share=live_side[0]; int order_n=d_order_count(x,share);
 for(int i=0;i<order_n;i++){int ally=d_order_slot(x,share,i); if(d_alive(x,share,ally)&&gain_energy)d_energy(x,share,ally,1);}
 for(int i=0;i<live_n;i++){
  int ps=live_side[i], pc=live_slot[i]; int before_h=d_ci(x,ps,pc,"harmony");
  int saved=x->side, saved_a=x->actor; x->side=ps; x->actor=pc;
  int extra=gain_harmony?d_decide(x,"combat_harmony_extra",0,nullptr,1,0):0;
  x->side=saved; x->actor=saved_a;
  if(gain_harmony){int cap=d_harmony_cap(x); int next=before_h+1+extra; if(next>cap)next=cap; d_set_ci(x,ps,pc,"harmony",next);}
  if(gain_energy)d_energy(x,ps,pc,1); d_event(x);
 }
}
DFN void d_finish_operation(DX* x){
 if(!life_op_present(x))return;
 if(d_finished(x)){life_clear_choice(x);return;}
 if(life_stop(x))return;
 int side=x->s->op_side[0]; int actor=x->s->op_actor_slot[0]; int card=x->s->op_card_index[0];
 x->side=side; x->actor=actor; x->card=card; int release=d_op_int(x,"release_only",0);
 if(card>=0){
  if(d_card_bool(x,card,"copy"))d_on_shape(x,"v2_copy_finished",nullptr);
  if(life_stop(x))return;
  if(d_card_bool(x,card,"retain"))d_add_hand(x,side,card,0); else d_discard(x,side,card,0);
  if(!release){DEvt ev; life_zero_evt(&ev); ev.actor=actor; ev.card=card; d_fire(x,"after_card",&ev,0,0);}
  if(x->s->op_card_mark_presence[0]==2&&!release){
   if(life_intercept(x)&&life_card_str_is(x,card,"type","battle"))d_defer_marked_card(x);
   else d_fire(x,"after_marked_card",nullptr,0,1);
  }
 }
 if(life_stop(x))return;
 life_share_resources(x); d_fire(x,"after_operation_resources",nullptr,1,0);
 d_migrate_legacy(x);for(int holder=0;holder<H;holder++)if(exists(x->s,holder)){d_consume_holder(x,holder,"operation_end",1);if(holder_char(holder)<0)x->s->player_effects_presence[holder_side(holder)]=2;else x->s->character_effects_presence[holder_side(holder)*C+holder_char(holder)]=2;} life_set_phase(x,1); life_clear_op(x); d_drain(x);
}
DFN void d_settle_deferred_card_marks(DX*x){d_settle_mark_queue(x);}

'''
_LIFE_PLAY = r'''
DFN void life_resolve_battle_frame(DX* x,int side,int actor,int card){
 if(!life_card_str_is(x,card,"type","battle")||!d_alive(x,side,actor))return;
 int attack=d_card_int(x,card,"attack"); int shield=d_card_int(x,card,"shield");
 d_scale_battle_frame(x,&attack,&shield);
 if(shield)d_shield(x,life_char(side,actor),shield);
 int is_set=life_card_str_is(x,card,"attack_mode","set");
 if(attack||is_set){
  if(life_intercept(x)){
   if(is_set)d_set_flag(x,side,actor,"counter_set_attack",attack,0);
   else d_set_flag(x,side,actor,"counter_bonus",d_flag(x,side,actor,"counter_bonus")+attack,0);
  }else if(is_set)d_set_op_int(x,"card_set_attack",attack,0);
  else d_set_op_int(x,"card_bonus",attack,0);
 }
}
DFN void d_play_card(DX* x,int response,int release_only){
 if(life_stop(x))return;
 int side=x->side; int card=x->card;
 if(card<0||card>=K||!x->s->card_valid[card]){d_fail(x,1);return;}
 int actor=life_owner_slot(x,side,card); if(actor<0){d_fail(x,1);return;} x->actor=actor;
 int form_bonus=life_form_instant(x,side,card); int paid=0;
 d_play_option(x,response,release_only);if(life_stop(x))return;if(!release_only){life_pay_ap(x,side,actor,card,&paid);if(life_stop(x))return;}
 d_open_operation(x);x->s->op_option_id_presence[0]=x->option_slot>=0?2:1;x->s->op_option_id_slot[0]=x->option_slot>=0?x->option_slot:0;d_set_op_int(x,"ap_paid",paid,0); if(release_only)d_set_op_int(x,"release_only",1,1); d_event(x);
 if(!release_only){
  DEvt ev; life_zero_evt(&ev); ev.actor=actor; ev.card=card; ev.target_side=x->target.side; ev.target=x->target.slot; ev.amount=paid; ev.response=response;
  d_fire(x,"on_card_used",&ev,0,0); if(form_bonus&&!life_stop(x))d_draw(x,side,1,-1);
 }
 if(life_stop(x))return; d_fire(x,"on_card_released",nullptr,0,0); if(life_stop(x))return;
 life_resolve_battle_frame(x,side,actor,card); d_card_effect(x);
 if(life_phase(x)!=2&&!life_stop(x))d_finish_operation(x);
}
DFN int life_response_ok(DX* x,int defending,const char* kind,int target,int attacker,int card,int owner){
 if(!life_card_str_is(x,card,"response",kind))return 0;
 int down=d_streq(kind,"self_downed")&&target==owner;
 if(!d_alive(x,defending,owner)&&!down)return 0;
 if(d_collapsed(x,defending,owner))return 0;
 if(d_streq(kind,"ally_attacked")&&(target<0||target==owner))return 0;
 if(d_streq(kind,"front_attacked")&&(d_front(x,defending)!=target||attacker<0))return 0;
 if(d_streq(kind,"faction_front_attacked")){
  if(d_front(x,defending)!=target)return 0;
  if(!d_response_faction_matches(x,card,defending,target))return 0;
 }
 if(d_streq(kind,"self_attacked")&&target!=owner)return 0;
 if((d_streq(kind,"self_lethal")||d_streq(kind,"self_downed"))&&target!=owner)return 0;
 if(d_streq(kind,"enemy_turn_end_empty")&&d_front(x,d_enemy(defending))>=0)return 0;
 return life_can_afford(x,defending,owner,card);
}
DFN void d_trigger_responses(DX*x,int side,const char*kind,int target,int attacker){if(d_finished(x)||d_bad(x))return;if(life_lock(x)){if(d_streq(kind,"self_downed"))life_queue_down(x,side,target);return;}int n=life_hand_n(x,side),physical[10];if(n<0||n>10){d_fail(x,1);return;}for(int j=0;j<n;j++)physical[j]=life_hand_at(x,side,j);for(int j=0;j<n;j++){DX own=*x;own.side=side;own.actor=life_carrier(x,side);own.card=physical[j];int card=d_effective_card(&own,side,physical[j]);own.card=card;int actor=d_card_owner(&own,card);if(actor<0)continue;own.actor=actor;if(!life_response_ok(&own,side,kind,target,attacker,card,actor)){if(card!=physical[j]&&card>=0)own.s->card_valid[card]=0;continue;}DOpSnap parent;life_save_op(x,&parent);d_remove_zone_card(x,side,0,physical[j]);life_set_lock(x,1);life_set_intercept(x,d_streq(kind,"self_attacked")||d_streq(kind,"ally_attacked")||d_streq(kind,"self_lethal"));own.depth=x->depth+1;own.option_slot=-1;own.target=DR{-1,-2};if(d_streq(kind,"ally_attacked")&&target>=0)own.target=DR{side,target};if(d_streq(kind,"front_attacked"))own.target=DR{1-side,attacker};d_play_card(&own,1,0);life_set_lock(x,0);life_set_intercept(x,0);life_restore_op(x,&parent);life_flush_down(x);return;}}

'''
_LIFE_TURN = r'''
DFN void life_end_ultimate(DX* x,int side,int actor){
 if(!d_cb(x,side,actor,"awakened")&&d_ci(x,side,actor,"ultimate_turns")==0)return;
 d_set_cb(x,side,actor,"awakened",0); d_set_ci(x,side,actor,"ultimate_turns",0);
 int saved_s=x->side,saved_a=x->actor; x->side=side; x->actor=actor;
 d_clear_effects(x,"ultimate_end"); d_fire(x,"on_ultimate_end",nullptr,1,1);
 x->side=saved_s; x->actor=saved_a; if(!life_stop(x))d_event(x);
}
DFN void life_expire_collapse(DX* x,int side){
 int until_turn=d_pi(x,side,"turn_count");
 for(int ps=0;ps<2;ps++){
  int n=d_order_count(x,ps);
  for(int i=0;i<n;i++){
   int ch=d_order_slot(x,ps,i); int holder=ps*(C+1)+1+ch;
   if(x->s->flag_collapse_presence[holder]==2&&x->s->flag_collapse_side[holder]==side&&x->s->flag_collapse_until[holder]<=until_turn){
    x->s->flag_collapse_presence[holder]=0; d_event(x);
   }
  }
 }
}
DFN void life_expire_ephemeral(DX* x){
 int turn=x->s->turn[0];
 for(int ps=0;ps<2;ps++){
  int hn=life_hand_n(x,ps); int cards[96]; int n=hn; if(n>96)n=96;
  for(int i=0;i<n;i++)cards[i]=life_hand_at(x,ps,i);
  for(int i=0;i<n;i++){
   int card=cards[i];
   if(life_card_has_int(x,card,KI_ephemeral_turn)&&d_card_int(x,card,"ephemeral_turn")<=turn){
    d_remove_zone_card(x,ps,0,card); d_discard(x,ps,card,0); if(life_stop(x))return;
   }
  }
 }
}
DFN void life_expire_copies(DX* x,int side){
 int turns=d_pi(x,side,"turn_count"); int hn=life_hand_n(x,side); int cards[96]; int n=hn; if(n>96)n=96;
 for(int i=0;i<n;i++)cards[i]=life_hand_at(x,side,i);
 for(int i=0;i<n;i++){
  int card=cards[i];
  if(d_card_bool(x,card,"copy")&&life_card_has_int(x,card,KI_expires_turn)&&d_card_int(x,card,"expires_turn")<=turns){
   d_remove_zone_card(x,side,0,card); d_discard(x,side,card,0); if(life_stop(x))return;
  }
 }
}
DFN void life_tick_star_delay(DX* x,int side){
 const char* kinds[2]={"star","delay"};
 for(int k=0;k<2;k++){
  const char* kind=kinds[k];
  if(!d_zone_has(x,side,kind))continue;
  if(d_zone_int(x,side,kind,"expires_side",-1)>=0)continue;
  if(k==1&&d_zone_int(x,side,kind,"tick_side",-1)>=0)continue;
  int left=d_zone_int(x,side,kind,"left",1); int amount=(k==0&&left<=1)?2:0;
  if(amount){
   int by=d_zone_int(x,side,kind,"by",d_enemy(side)); int tgt=d_front(x,side);
   DR dest=tgt>=0?life_char(side,tgt):life_player(side);
   int saved_s=x->side,saved_a=x->actor; x->side=by; x->actor=life_carrier(x,by);
   DEvt bev; life_zero_evt(&bev); bev.kind=k?4:6; amount+=d_tally(x,"dot_damage_bonus",&bev);
   char source[16];source[0]=(char)('a'+by);source[1]=':';const char*name=k?"delay":"star";for(int j=0;;j++){source[j+2]=name[j];if(!name[j])break;}DX damage=*x;damage.damage_source_slot=d_intern(x,source);d_damage(&damage,dest,amount,life_player(by),k?4:6,0,1,0,0,0);
   if(!life_stop(x)){DEvt ev; life_zero_evt(&ev); ev.actor=x->actor; ev.attacker_side=by; ev.attacker=x->actor; ev.defender_side=side; ev.defender=tgt; ev.target_side=side; ev.target=tgt; ev.amount=amount; ev.kind=k?4:6; d_fire(x,"on_dot",&ev,0,0);}
   x->side=saved_s; x->actor=saved_a; if(life_stop(x))return;
  }
  if(!d_zone_int(x,side,kind,"infinite",0)){
   left=left-1; if(left<0)left=0; d_zone_set(x,side,kind,"left",left);
   if(left==0){
    int renewals=d_zone_int(x,side,kind,"renewals",0); int by=d_zone_int(x,side,kind,"by",d_enemy(side));int source_slot=k==0?d_zone_int(x,side,kind,"source_cid",-1):-1;int source_actor=life_carrier(x,by);if(source_slot>=0){for(int j=0;j<x->s->player_character_count[by];j++)if(qslots(x->s,source_slot,x->s->character_id_slot[by*C+j],x->s->row_error))source_actor=j;}
    d_zone_remove(x,side,kind);
    if(k==0&&renewals>0&&!life_stop(x)){int saved_s=x->side,saved_a=x->actor; x->side=by; x->actor=source_actor; d_apply_dark_star(x,renewals-1); x->side=saved_s; x->actor=saved_a;}
   }
  }
  d_event(x);
 }
 for(int owner=0;owner<2;owner++){
  if(!d_zone_has(x,owner,"delay")||d_zone_int(x,owner,"delay","tick_side",-3)!=side)continue;
  int saved_s=x->side,saved_a=x->actor; x->side=side; x->actor=life_carrier(x,side);
  int paused=d_decide(x,"pause_delay",0,nullptr,0,1); x->side=saved_s; x->actor=saved_a;
  if(!paused){int left=d_zone_int(x,owner,"delay","left",1)-1; if(left<0)left=0; d_zone_set(x,owner,"delay","left",left); if(!left)d_zone_remove(x,owner,"delay");}
  d_event(x);
 }
}
DFN void life_expire_turn_zones(DX* x,int side){
 const char* kinds[3]={"delay","weave","stain"};
 for(int owner=0;owner<2;owner++)for(int k=0;k<3;k++){
  const char* kind=kinds[k]; if(!d_zone_has(x,owner,kind))continue;
  if(k==2){int ends=d_zone_int(x,owner,kind,"ends_left",0); if(ends>1){d_zone_set(x,owner,kind,"ends_left",ends-1); d_zone_set(x,owner,kind,"expires_side",d_enemy(side)); continue;}}
  int exp=d_zone_int(x,owner,kind,"expires_side",k==0?-3:d_zone_int(x,owner,kind,"by",-3));
  if(exp==side){d_zone_remove(x,owner,kind); d_event(x);}
 }
}
DFN void life_revive_downed(DX* x,int side){
 int n=d_order_count(x,side);
 for(int i=0;i<n;i++){
  int ch=d_order_slot(x,side,i); x->s->character_shield[side*C+ch]=0;
  int down=d_ci(x,side,ch,"down_turns");
  if(down>0){d_clear_harmony_source(x,side,ch); down-=1; d_set_ci(x,side,ch,"down_turns",down); if(!down){d_set_ci(x,side,ch,"hp",d_max_hp(x,side,ch)); d_event(x);}}
 }
}
DFN void life_clamp(DX* x){
 if(!x->s->escalation_enabled[0])return;
 for(int ps=0;ps<2;ps++){
  int n=d_order_count(x,ps);
  for(int i=0;i<n;i++){
   int ch=d_order_slot(x,ps,i);
   int e=d_ci(x,ps,ch,"energy"); int cap=d_energy_cap(x,ps,ch); if(e>cap)d_set_ci(x,ps,ch,"energy",cap);
   int h=d_ci(x,ps,ch,"harmony"); int hc=d_harmony_cap(x); if(h>hc)d_set_ci(x,ps,ch,"harmony",hc);
  }
 }
}
DFN void life_begin_escalation(DX* x,int side){
 if(!(x->s->escalation_enabled[0]&&x->s->turn[0]>=6))return;
 int n=d_order_count(x,side);
 for(int i=0;i<n;i++){
  int ch=d_order_slot(x,side,i); if(!d_can_gain_energy(x,side,ch))continue;
  int before=d_ci(x,side,ch,"energy"); d_energy(x,side,ch,1); if(d_ci(x,side,ch,"energy")>before)d_event(x);
 }
 if(!(x->s->escalation_enabled[0]&&x->s->turn[0]>=20))return;
 int dn=life_discard_n(x,side); int cand[96]; int cn=0;
 for(int i=0;i<dn&&cn<96;i++){
  int card=life_discard_at(x,side,i);
  if(card<0||card>=K||!x->s->card_valid[card])continue;
  int id=x->s->card_template[card];
  if(id<=0||d_card_bool(x,card,"copy")||d_card_bool(x,card,"derived")||d_card_default_bool(id,KB_derived))continue;
  cand[cn++]=card;
 }
 if(!cn)return;
 if(cn>1)d_shuffle_refs(x,cand,cn);
 if(life_stop(x))return;
 int card=cand[0]; d_remove_zone_card(x,side,2,card); d_add_hand(x,side,card,1);
}
DFN int life_turn_ap(DX* x){if(x->s->turn[0]==1)return 1; if(x->s->escalation_enabled[0]&&x->s->turn[0]>=20)return 3; return 2;}
DFN void life_clear_used(DX* x,int side){x->s->player_used_count[side]=0;}
DFN void life_inspect_open(DX* x,int side,int count){
 int dn=life_deck_n(x,side); if(count>dn)count=dn; if(count<=0)return;
 int taken[32]; int n=count; if(n>32)n=32;
 for(int i=0;i<n;i++)taken[i]=life_deck_at(x,side,i);
 for(int i=0;i<n;i++)d_remove_zone_card(x,side,1,taken[i]);
 if(n==1)d_add_hand(x,side,taken[0],0);
 else {
  d_choice(x,"inspect_top",n); x->s->choice_card_count[0]=(short)n;
  for(int i=0;i<n;i++)x->s->choice_card_index[i]=taken[i];
  x->s->choice_side[0]=(signed char)side; x->s->choice_kind[0]=1;
 }
}
DFN void life_after_draw_choice(DX* x,int side){
 if(life_phase(x)==2){
  if(!life_op_present(x)){int actor=x->s->choice_actor_presence[0]==2?x->s->choice_actor_slot[0]:life_carrier(x,side); x->side=side; x->actor=actor; x->card=-1; x->target=DR{side,-2}; d_open_operation(x);}
  life_mark_turn_start(x,side);
 }
}
DFN void d_begin_turn(DX* x,int side){
 if(life_stop(x))return; if(side<0||side>1){d_fail(x,1);return;}
 d_prepare_effects(x); life_set_phase(x,1); x->s->active_side[0]=(signed char)side; x->s->turn[0]+=1;
 d_set_pi(x,side,"turn_count",d_pi(x,side,"turn_count")+1); d_event(x);
 d_expire_effects(x,side,"turn_start"); if(life_stop(x))return;
 d_del_pb(x,side,"life_locked"); life_clear_used(x,side);
 for(int ps=0;ps<2;ps++){x->s->player_attacked_count[ps]=0;x->s->player_attacked_this_turn_presence[ps]=2; int cn=x->s->player_character_count[ps]; for(int c=0;c<cn&&c<C;c++)x->s->player_attacked_this_turn[ps*C+c]=0;}
 d_upgrade_harmony_residue(x); d_set_pb(x,side,"harmony_damage",0); d_set_pi(x,side,"genesis_damage_bonus",0); d_set_pb(x,side,"surplus",0);
 life_clamp(x); d_set_pi(x,side,"shield",0); d_halve_dots(x,side); life_revive_downed(x,side); life_begin_escalation(x,side);
 int carrier=life_carrier(x,side); x->side=side; x->actor=carrier; x->card=-1; x->target=DR{side,-2};
 d_fire(x,"on_turn_clock",nullptr,0,1); d_fire(x,"on_turn_begin",nullptr,0,0); if(life_stop(x))return;
 if(d_pb(x,side,"extra_genesis_pending")){
  d_del_pb(x,side,"extra_genesis_pending"); int actor=d_extra_genesis_actor(x,side);d_del_pstr(x,side,"extra_genesis_actor");if(actor>=0&&d_alive(x,side,actor))d_event(x);
  if(actor>=0&&d_alive(x,side,actor)){x->actor=actor; d_resolve_genesis(x,0);}
 }
 int other=d_enemy(side); int saved_s=x->side,saved_a=x->actor; x->side=other; x->actor=life_carrier(x,other);
 d_fire(x,"on_enemy_turn_begin",nullptr,0,0); x->side=saved_s; x->actor=saved_a; if(life_stop(x))return;
 d_drain(x); life_tick_star_delay(x,side); if(life_stop(x))return;
 for(int ps=0;ps<2;ps++)d_record_delay_turn(x,ps);
 x->side=side; x->actor=carrier; d_fire(x,"on_turn_countdown",nullptr,0,1); if(life_stop(x))return;
 int front=d_front(x,side);
 if(front>=0&&!d_collapsed(x,side,front)&&!d_cb(x,side,front,"summoned")){x->actor=front; if(d_decide(x,"natural_return",1,nullptr,1,0))d_move_out(x,side,front,0);}
 x->actor=carrier; int inspect_actor=0,inspect_count=0;
 if(d_turn_draw_inspect(x,&inspect_actor,&inspect_count)&&life_deck_n(x,side)>0){
  x->actor=inspect_actor; d_open_operation(x); life_inspect_open(x,side,inspect_count);
  if(life_phase(x)!=2)d_finish_operation(x); else life_mark_turn_start(x,side);
 }else{d_draw(x,side,1,-1); d_fire(x,"after_turn_draw",nullptr,0,0); life_after_draw_choice(x,side);}
 if(life_stop(x))return;
 int extra=d_pi(x,side,"extra_ap"); d_del_pi(x,side,"extra_ap"); d_set_pi(x,side,"ap",life_turn_ap(x)+extra); if(extra)d_event(x);
 int skip=d_pb(x,side,"skip_normal_attack"); d_del_pb(x,side,"skip_normal_attack");
 d_set_pb(x,side,"normal_attack_available",skip?0:1); d_set_pi(x,side,"ultimates_used",0); d_set_pb(x,side,"ultimate_available",1);
 if(life_phase(x)==1)d_save_turn_start(x,side);
}
DFN void d_end_turn(DX* x){
 if(life_stop(x))return; int side=x->s->active_side[0]; int n=d_order_count(x,side);
 for(int i=0;i<n;i++){int ch=d_order_slot(x,side,i); if(d_flag(x,side,ch,"temp_revive")&&d_ci(x,side,ch,"hp")>0){d_set_ci(x,side,ch,"hp",0); d_del_flag(x,side,ch,"temp_revive");}}
 d_knockdowns(x); if(life_stop(x))return;
 n=d_order_count(x,side); for(int i=0;i<n;i++)d_del_flag(x,side,d_order_slot(x,side,i),"return_after");
 x->side=side; x->actor=life_carrier(x,side); x->card=-1; x->target=DR{side,-2};
 d_fire(x,"on_turn_end",nullptr,0,0); if(life_stop(x))return;
 d_tick_owner_end(x,side); if(life_stop(x))return; life_expire_collapse(x,side);
 n=d_order_count(x,side);
 for(int i=0;i<n;i++){int ch=d_order_slot(x,side,i); int holder=side*(C+1)+1+ch; x->s->flag_mimic_presence[holder]=0; x->s->flag_mimic_id_presence[holder]=0;}
 life_expire_ephemeral(x); if(life_stop(x))return;
 d_trigger_responses(x,d_enemy(side),"enemy_turn_end_empty",-1,-1); if(life_stop(x))return;
 life_expire_copies(x,side);
 d_set_pb(x,side,"normal_attack_available",0); d_set_pb(x,side,"ultimate_available",0);
 d_set_pb(x,side,"extra_flower",0); d_set_pb(x,side,"extra_delay",0); d_set_pi(x,side,"genesis_damage_bonus",0);
 n=d_order_count(x,side); for(int i=0;i<n;i++)d_del_flag(x,side,d_order_slot(x,side,i),"free_normal_attack_turn");
 life_expire_turn_zones(x,side); d_prepare_effects(x); d_expire_effects(x,side,"turn_end"); d_clear_next_damage_resistance(x);
 if(life_stop(x))return; d_begin_turn(x,d_enemy(side));
}
'''
_LIFE_CHOOSE = r'''
DFN int life_choice_index_of(DX* x,int card){int n=x->s->choice_card_count[0]; for(int i=0;i<n;i++)if(x->s->choice_card_index[i]==card)return i; return -1;}
DFN int life_in_choice(DX* x,int card){return life_choice_index_of(x,card)>=0;}
DFN int life_hand_has(DX* x,int side,int card){int n=life_hand_n(x,side); for(int i=0;i<n;i++)if(life_hand_at(x,side,i)==card)return 1; return 0;}
DFN void life_finish_choice_op(DX* x){
 if(life_phase(x)!=2&&!life_stop(x)){life_resume_pending_sortie(x); if(life_phase(x)!=2&&!life_stop(x))d_finish_operation(x); life_resume_turn_start(x);}
}
DFN void life_choose_discard(DX* x,int side,const int* selected,int n){
 if(n!=1){d_fail(x,1);return;} int card=selected[0]; if(!life_in_choice(x,card)){d_fail(x,1);return;}
 d_remove_zone_card(x,side,0,card); d_discard(x,side,card,0);
}
DFN void life_choose_inspect_top(DX* x,int side,const int* selected,int n){
 if(n!=1){d_fail(x,1);return;} int card=selected[0]; if(!life_in_choice(x,card)){d_fail(x,1);return;}
 d_add_hand(x,side,card,0); int cn=x->s->choice_card_count[0];
 for(int i=0;i<cn;i++){
  int other=x->s->choice_card_index[i]; if(other==card)continue;
  int dn=x->s->zone_count[side*4+1]; if(dn>=Z){d_fail(x,2);return;}
  x->s->zone_card_index[(side*4+1)*Z+dn]=other; x->s->zone_count[side*4+1]=(short)(dn+1);
 }
}
DFN void life_choose_enemy_hand(DX* x,int side,const int* selected,int n){
 if(n!=1){d_fail(x,1);return;} int card=selected[0]; int enemy=d_enemy(side);
 d_remove_zone_card(x,enemy,0,card); d_event(x); d_discard(x,enemy,card,1);
}
DFN void life_choose_kit(DX* x,int side,const int* selected,int n){
 if(n!=1){d_fail(x,1);return;} int card=selected[0]; if(!life_in_choice(x,card)){d_fail(x,1);return;}
 int cn=x->s->choice_card_count[0]; if(cn>CH)cn=CH; x->s->op_choice_card_count[0]=(short)cn;
 for(int i=0;i<cn;i++)x->s->op_choice_card_index[i]=x->s->choice_card_index[i];
 DEvt ev; life_zero_evt(&ev); ev.actor=x->s->op_actor_slot[0]; ev.card=card; ev.amount=card; ev.kind=3;
 d_fire(x,"on_choice",&ev,1,0); (void)side;
}
DFN void life_choose_kit_multi(DX* x,int side,const int* selected,int n){
 int maxs=x->s->choice_int_presence[0]==2?x->s->choice_int_value[0]:-1;
 if(n!=maxs){d_fail(x,1);return;} for(int i=0;i<n;i++)if(!life_in_choice(x,selected[i])){d_fail(x,1);return;}
 x->s->op_choice_card_count[0]=(short)n; for(int i=0;i<n&&i<CH;i++)x->s->op_choice_card_index[i]=selected[i];
 d_fire(x,"on_choice_cards",nullptr,1,0); (void)side;
}
DFN void life_choose_redraw(DX* x,int side,const int* selected,int n){
 if(n<0||n>3){d_fail(x,1);return;} int taken[8];
 for(int i=0;i<n;i++){taken[i]=selected[i]; d_remove_zone_card(x,side,0,taken[i]);}
 d_event(x); d_draw(x,side,n,-1);
 for(int i=0;i<n;i++){int dn=x->s->zone_count[side*4+1]; if(dn>=Z){d_fail(x,2);return;} x->s->zone_card_index[(side*4+1)*Z+dn]=taken[i]; x->s->zone_count[side*4+1]=(short)(dn+1);}
 if(n>0)d_shuffle_zone(x,side,1);
}
DFN void life_choose_inspect_deck(DX* x,int side,const int* selected,int n){
 if(n!=1){d_fail(x,1);return;} life_choose_kit(x,side,selected,n);
}
DFN void life_choose_order(DX* x,int side,const int* top,int ntop,const int* bottom,int nbottom,int bottom_plural,int kind){
 int cn=x->s->choice_card_count[0];
 int hand_place=x->s->choice_bool_presence[1]==2&&x->s->choice_bool_value[1];
 int min_bottom=x->s->choice_int_presence[3]==2?x->s->choice_int_value[3]:0;
 int max_bottom=x->s->choice_int_presence[2]==2?x->s->choice_int_value[2]:cn;
 if(nbottom<min_bottom||nbottom>max_bottom){d_fail(x,1);return;}
 int chosen_n=ntop+nbottom;
 if(hand_place){
  if(chosen_n!=1){d_fail(x,1);return;} int card=ntop?top[0]:bottom[0];
  if(!life_hand_has(x,side,card)){d_fail(x,1);return;} d_remove_zone_card(x,side,0,card);
 }else{
  if(chosen_n!=cn){d_fail(x,1);return;} int seen[64]; for(int i=0;i<cn&&i<64;i++)seen[i]=0;
  for(int i=0;i<ntop;i++){int idx=life_choice_index_of(x,top[i]); if(idx<0||(idx<64&&seen[idx])){d_fail(x,1);return;} if(idx<64)seen[idx]=1;}
  for(int i=0;i<nbottom;i++){int idx=life_choice_index_of(x,bottom[i]); if(idx<0||(idx<64&&seen[idx])){d_fail(x,1);return;} if(idx<64)seen[idx]=1;}
 }
 for(int i=ntop-1;i>=0;i--){
  int card=top[i]; int dn=x->s->zone_count[side*4+1]; if(dn>=Z){d_fail(x,2);return;}
  for(int p=dn;p>0;p--)x->s->zone_card_index[(side*4+1)*Z+p]=x->s->zone_card_index[(side*4+1)*Z+p-1];
  x->s->zone_card_index[(side*4+1)*Z+0]=card; x->s->zone_count[side*4+1]=(short)(dn+1);
 }
 for(int i=0;i<nbottom;i++){
  int card=bottom[i]; int dn=x->s->zone_count[side*4+1]; if(dn>=Z){d_fail(x,2);return;}
  x->s->zone_card_index[(side*4+1)*Z+dn]=card; x->s->zone_count[side*4+1]=(short)(dn+1);
 }
 int execute=1; if(x->s->choice_bool_presence[2]==2)execute=x->s->choice_bool_value[2];
 int actor=x->s->choice_actor_presence[0]==2?x->s->choice_actor_slot[0]:(life_op_present(x)?x->s->op_actor_slot[0]:life_carrier(x,side));
 x->side=side; x->actor=actor;
 if(x->s->choice_continuation_presence[0]==2){
  int slot=x->s->choice_continuation_slot[0];
  if(d_eq(x,slot,"heiyu.divination"))d_continue_choice(x,"heiyu.divination",bottom,nbottom,execute);
  else d_continue_choice(x,"unknown",bottom,nbottom,execute);
 }else{
  d_fire(x,"on_deck_reorder",nullptr,1,0);
  if(life_phase(x)!=2&&(kind==9||d_op_int(x,"heiyu_after_placement",0)))life_resume_pending_sortie(x);
 }
 (void)bottom_plural;
}
DFN void d_choose(DX* x,int kind,const int* selected,int n,const int* top,int ntop,const int* bottom,int nbottom,int bottom_plural){
 if(life_stop(x))return;
 if(!life_choice_present(x)||life_phase(x)!=2){d_fail(x,1);return;}
 int side=x->s->choice_side[0]; int pending=x->s->choice_kind[0]; if(kind<0)kind=pending;
 if(kind!=pending){d_fail(x,1);return;} x->side=side;life_set_phase(x,1);x->s->pending_choice_present[0]=0;x->choice_selected=n?selected[0]:-1;
 if(life_op_present(x)){x->actor=x->s->op_actor_slot[0]; x->card=x->s->op_card_index[0];}
 if(kind==0)life_choose_discard(x,side,selected,n);
 else if(kind==1)life_choose_inspect_top(x,side,selected,n);
 else if(kind==2)life_choose_enemy_hand(x,side,selected,n);
 else if(kind==3)life_choose_kit(x,side,selected,n);
 else if(kind==4)life_choose_kit_multi(x,side,selected,n);
 else if(kind==5)life_choose_redraw(x,side,selected,n);
 else if(kind==7)life_choose_inspect_deck(x,side,selected,n);
 else if(kind==6||kind==8||kind==9)life_choose_order(x,side,top,ntop,bottom,nbottom,bottom_plural,kind);
 else {d_fail(x,1);return;}
 if(life_stop(x))return;if(life_choice_present(x)&&life_phase(x)==2)return;life_clear_choice(x);life_finish_choice_op(x);
}
'''
