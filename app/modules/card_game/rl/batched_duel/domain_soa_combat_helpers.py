"""Named combat helper device functions for the no-VM domain executor.

Direct C++ only. Pending sortie uses typed SoA columns; legacy maps are not
reintroduced. Wrappers keep card/lifecycle call sites on concrete d_* names.
"""
from . import domain_soa_schema as schema

supports_full_duel_rules = False

_NSI = len(schema.SORTIE_INT_KEYS)
_NSB = len(schema.SORTIE_BOOL_KEYS)


def _c_strings(names):
    return ''.join('if(d_streq(key,"'+n+'"))return '+str(i)+';' for i,n in enumerate(names))


def emit_source(layout=None):
    nsi, nsb = _NSI, _NSB
    seal = schema.PLAYER_STR_FIELDS.index('seal_normal_attack')
    n_pstr = len(schema.PLAYER_STR_FIELDS)
    text = '#ifndef NSI\n#define NSI '+str(nsi)+'\n#define NSB '+str(nsb)+'\n#endif\n'
    text += '#ifndef DCOMBAT_SPEC_DEFINED\n#define DCOMBAT_SPEC_DEFINED\n'
    text += (
        'struct DCombatSpec {\n'
        ' DAttack base;\n'
        ' int stay_bench, front_only, additional_attack, bypass;\n'
        ' int penetrate_harmony, penetrate_low_hp, penetrate_brand;\n'
        ' int no_combat_resources, support_allows_counter, no_counter_harmony;\n'
        ' int sync_attack, shield_after_shape, attribute, has_target;\n'
        ' DR target;\n'
        ' unsigned char int_presence[NSI];\n'
        ' unsigned char bool_presence[NSB];\n'
        ' unsigned char has_attribute;\n'
        '};\n#endif\n'
    )
    text += r'''
DFN void combat_spec_clear_presence(DCombatSpec* spec){
 for(int i=0;i<NSI;i++)spec->int_presence[i]=0;
 for(int i=0;i<NSB;i++)spec->bool_presence[i]=0;
 spec->has_attribute=0; spec->has_target=0;
}
DFN int combat_sortie_int_key(const char* key){'''+_c_strings(schema.SORTIE_INT_KEYS)+r'''return -1;}
DFN int combat_sortie_bool_key(const char* key){'''+_c_strings(schema.SORTIE_BOOL_KEYS)+r'''return -1;}
'''
    text += r'''
DFN void combat_spec_set_int(DCombatSpec* spec,const char* key,int value){
 int i=combat_sortie_int_key(key); if(i<0)return; spec->int_presence[i]=2;
 if(i==0)spec->base.bonus=value; else if(i==1)spec->base.hits=value; else if(i==2)spec->base.low_hp_bonus=value;
 else if(i==3)spec->base.low_hp_max=value; else if(i==4){spec->base.awake_player_bonus=value;spec->base.has_awake_player_bonus=1;}
 else if(i==5)spec->base.harmony_bonus=value; else if(i==6)spec->base.additional_damage_bonus=value;
 else if(i==7)spec->penetrate_brand=value; else if(i==8)spec->shield_after_shape=value;
}
DFN void combat_spec_set_bool(DCombatSpec* spec,const char* key,int value){
 int i=combat_sortie_bool_key(key); if(i<0)return; spec->bool_presence[i]=2; value=value!=0;
 if(i==0)spec->base.ignore_shield=value; else if(i==1)spec->base.no_counter=value; else if(i==2)spec->stay_bench=value;
 else if(i==3)spec->additional_attack=value; else if(i==4)spec->bypass=value; else if(i==5)spec->front_only=value;
 else if(i==6)spec->no_combat_resources=value; else if(i==7)spec->no_counter_harmony=value;
 else if(i==8)spec->support_allows_counter=value; else if(i==9)spec->sync_attack=value;
 else if(i==10)spec->penetrate_harmony=value; else if(i==11)spec->penetrate_low_hp=value;
}
'''
    text += r'''
DFN int d_pop_next_resistance(DX* c,DR target,int attribute){(void)c;(void)target;(void)attribute;return 0;}
DFN void d_clear_next_resistance(DX* c,DR target){(void)c;(void)target;}
DFN void d_consume_applied(DX* c,const char* event,const int* ids,int n){
 if(!event||n<0){d_fail(c,1);return;}
 if(!n||!ids)return;
 DS* f=c->s;
 for(int holder=0;holder<H;holder++){
  if(!exists(f,holder))continue;
  int count=f->holder_effect_count[holder]; if(count<0||count>E){d_fail(c,1);return;}
  for(int j=count-1;j>=0;j--){
   int e=holder*E+j;
   if(f->effect_consume_on_presence[e]!=2)continue;
   if(!d_eq(c,f->effect_consume_on_slot[e],event))continue;
   int eid=f->effect_id_slot[e], hit=0;
   for(int k=0;k<n;k++) if(ids[k]==eid){hit=1;break;}
   if(hit) d_effect_remove(c,holder,j);
  }
 }
}
DFN int d_decide_redirect_overflow(DX* c,DR target,int loss,int absorbed){
 DEvt ev{}; ev.actor=ev.card=ev.attacker_side=ev.attacker=ev.defender_side=ev.defender=-1;
 ev.target_side=target.side; ev.target=target.slot; ev.amount=0; ev.loss=loss; ev.absorbed=absorbed;
 ev.raw_overflow=0; ev.kind=0; ev.attribute=0; ev.additional=0; ev.index=0; ev.response=0; ev.entering=0;
 ev.holder_entity=d_entity(c,target); ev.effect_source_entity=ev.holder_entity;
 return d_decide(c,"redirect_overflow",-2,&ev,0,0);
}
DFN void d_remove_source_shape(DX* c,int entity){
 if(!entity)return;
 DR r=d_entity_ref(c,entity); if(d_bad(c)||r.slot<0)return;
 int h=r.side*C+r.slot, p=c->s->character_str_presence[h*NS+CS_shape];
 if(p!=2)return;
 d_remove_effect_source(c,entity,c->s->character_str_slot[h*NS+CS_shape]);
}
'''
    text += (
        'DFN void d_clear_seal_if_summon(DX* c,int foe,int ch){\n'
        ' if(foe<0||foe>1){d_fail(c,1);return;}\n'
        ' int side=1-foe; if(!d_valid_char(c,side,ch))return;\n'
        ' if(!d_cb(c,side,ch,"summoned"))return;\n'
        ' int at=foe*'+str(n_pstr)+'+'+str(seal)+';\n'
        ' if(c->s->player_str_presence[at]!=2)return;\n'
        ' if(c->s->player_str_slot[at]==c->s->character_id_slot[side*C+ch]){\n'
        '  c->s->player_str_presence[at]=0; c->s->player_str_slot[at]=0;\n'
        ' }\n'
        '}\n'
    )
    text += r'''
DFN void d_upgrade_harmony_residue(DX* c){
 if(c->s->harmony_residue_version[0]==1)return;
 for(int side=0;side<2;side++){ c->s->player_last_front_presence[side]=1; c->s->player_last_front_slot[side]=0; }
 c->s->harmony_residue_version[0]=1;
}
DFN void d_clear_harmony_source(DX* c,int side,int ch){
 if(side<0||side>1){d_fail(c,1);return;}
 if(c->s->player_last_front_presence[side]==2 && c->s->player_last_front_slot[side]==ch){
  c->s->player_last_front_presence[side]=1; c->s->player_last_front_slot[side]=0;
 }
}
DFN void d_record_delay_turn(DX* c,int side){
 if(side<0||side>1){d_fail(c,1);return;}
 int enemy=d_enemy(side);
 if(!d_zone_has(c,enemy,"delay"))return;
 if(d_pi(c,side,"delay_counted_turn")==c->s->turn[0])return;
 d_set_pi(c,side,"delay_counted_turn",c->s->turn[0]);
 d_set_pi(c,side,"delay_turns",d_pi(c,side,"delay_turns")+1);
}
'''
    text += r'''
DFN void combat_write_sortie_columns(DX* c,const DCombatSpec* spec){
 DS* f=c->s;
 f->op_pending_sortie_presence[0]=2;
 f->op_pending_stay_bench[0]=(unsigned char)(spec->stay_bench?1:0);
 if(spec->int_presence[0]==2){ f->op_pending_bonus_presence[0]=2; f->op_pending_bonus[0]=spec->base.bonus; }
 else { f->op_pending_bonus_presence[0]=0; f->op_pending_bonus[0]=0; }
 for(int i=0;i<NSI;i++){
  f->op_sortie_int_presence[i]=spec->int_presence[i];
  int v=0;
  if(i==0)v=spec->base.bonus; else if(i==1)v=spec->base.hits; else if(i==2)v=spec->base.low_hp_bonus;
  else if(i==3)v=spec->base.low_hp_max; else if(i==4)v=spec->base.awake_player_bonus; else if(i==5)v=spec->base.harmony_bonus;
  else if(i==6)v=spec->base.additional_damage_bonus; else if(i==7)v=spec->penetrate_brand; else v=spec->shield_after_shape;
  f->op_sortie_int_value[i]=spec->int_presence[i]==2?v:0;
 }
 for(int i=0;i<NSB;i++){
  f->op_sortie_bool_presence[i]=spec->bool_presence[i];
  int v=0;
  if(i==0)v=spec->base.ignore_shield; else if(i==1)v=spec->base.no_counter; else if(i==2)v=spec->stay_bench;
  else if(i==3)v=spec->additional_attack; else if(i==4)v=spec->bypass; else if(i==5)v=spec->front_only;
  else if(i==6)v=spec->no_combat_resources; else if(i==7)v=spec->no_counter_harmony; else if(i==8)v=spec->support_allows_counter;
  else if(i==9)v=spec->sync_attack; else if(i==10)v=spec->penetrate_harmony; else v=spec->penetrate_low_hp;
  f->op_sortie_bool_value[i]=(unsigned char)(spec->bool_presence[i]==2&&v);
 }
 if(spec->has_attribute){ f->op_sortie_attribute_presence[0]=2; f->op_sortie_attribute[0]=(signed char)spec->attribute; }
 else { f->op_sortie_attribute_presence[0]=0; f->op_sortie_attribute[0]=0; }
 if(spec->has_target){
  f->op_sortie_target_presence[0]=2; f->op_sortie_target_side[0]=(signed char)spec->target.side;
  f->op_sortie_target_actor[0]=(short)spec->target.slot;
 } else { f->op_sortie_target_presence[0]=0; f->op_sortie_target_side[0]=0; f->op_sortie_target_actor[0]=0; }
}
DFN int combat_read_sortie_columns(DX* c,DCombatSpec* spec){
 DS* f=c->s; if(f->op_pending_sortie_presence[0]!=2)return 0;
 DAttack z{}; spec->base=z; combat_spec_clear_presence(spec);
 spec->stay_bench=f->op_pending_stay_bench[0]!=0;
 spec->front_only=spec->additional_attack=spec->bypass=0;
 spec->penetrate_harmony=spec->penetrate_low_hp=spec->penetrate_brand=0;
 spec->no_combat_resources=spec->support_allows_counter=spec->no_counter_harmony=0;
 spec->sync_attack=spec->shield_after_shape=spec->attribute=0;
 spec->target=DR{-1,-2};
 for(int i=0;i<NSI;i++){
  spec->int_presence[i]=f->op_sortie_int_presence[i];
  if(spec->int_presence[i]!=2)continue;
  int v=f->op_sortie_int_value[i];
  if(i==0)spec->base.bonus=v; else if(i==1)spec->base.hits=v; else if(i==2)spec->base.low_hp_bonus=v;
  else if(i==3)spec->base.low_hp_max=v; else if(i==4){spec->base.awake_player_bonus=v;spec->base.has_awake_player_bonus=1;}
  else if(i==5)spec->base.harmony_bonus=v; else if(i==6)spec->base.additional_damage_bonus=v;
  else if(i==7)spec->penetrate_brand=v; else spec->shield_after_shape=v;
 }
 for(int i=0;i<NSB;i++){
  spec->bool_presence[i]=f->op_sortie_bool_presence[i];
  if(spec->bool_presence[i]!=2)continue;
  int v=f->op_sortie_bool_value[i]!=0;
  if(i==0)spec->base.ignore_shield=v; else if(i==1)spec->base.no_counter=v; else if(i==2)spec->stay_bench=v;
  else if(i==3)spec->additional_attack=v; else if(i==4)spec->bypass=v; else if(i==5)spec->front_only=v;
  else if(i==6)spec->no_combat_resources=v; else if(i==7)spec->no_counter_harmony=v; else if(i==8)spec->support_allows_counter=v;
  else if(i==9)spec->sync_attack=v; else if(i==10)spec->penetrate_harmony=v; else spec->penetrate_low_hp=v;
 }
 if(f->op_sortie_attribute_presence[0]==2){ spec->has_attribute=1; spec->attribute=f->op_sortie_attribute[0]; }
 if(f->op_sortie_target_presence[0]==2){
  spec->has_target=1; spec->target.side=f->op_sortie_target_side[0]; spec->target.slot=f->op_sortie_target_actor[0];
 }
 if(spec->int_presence[0]!=2 && f->op_pending_bonus_presence[0]==2){
  spec->int_presence[0]=2; spec->base.bonus=f->op_pending_bonus[0];
 }
 if(spec->base.hits<=0) spec->base.hits=1;
 if(!spec->base.low_hp_max) spec->base.low_hp_max=3;
 return 1;
}
DFN void d_set_pending_sortie(DX* c,const DCombatSpec* spec){ if(!spec){d_fail(c,1);return;} combat_write_sortie_columns(c,spec); }
DFN void d_set_pending_sortie(DX* c,const DAttack* in){
 DCombatSpec spec{}; combat_spec_clear_presence(&spec); spec.target=DR{-1,-2};
 if(in){
  spec.base=*in;
  spec.int_presence[0]=2; spec.int_presence[1]=2; spec.int_presence[2]=2; spec.int_presence[3]=2;
  spec.int_presence[5]=2; spec.int_presence[6]=2;
  if(in->has_awake_player_bonus) spec.int_presence[4]=2;
  spec.bool_presence[0]=2; spec.bool_presence[1]=2;
 }
 combat_write_sortie_columns(c,&spec);
}
DFN int d_load_pending_sortie(DX* c,DCombatSpec* spec){ if(!spec){d_fail(c,1);return 0;} return combat_read_sortie_columns(c,spec); }
DFN int d_load_pending_sortie(DX* c,DAttack* out){
 if(!out){d_fail(c,1);return 0;} DCombatSpec spec{}; if(!combat_read_sortie_columns(c,&spec))return 0; *out=spec.base; return 1;
}
'''
    text += r'''
DFN void combat_zero_effect(DEffect* e){
 e->id=e->property=e->op=e->scope=e->stacking=e->stack_key=""; e->value=0; e->layer=0;
 e->expiry_side=-1; e->expiry_turn=0; e->expiry_phase=e->consume_on=e->clear_on="";
 e->condition=e->condition_key=e->condition_string=""; e->condition_number=0;
 e->label=e->source_key=e->clear_on2=e->kind=e->visibility=nullptr; e->source_entity=-1;
 e->max_stacks=0; e->show_marker=-1; e->value_kind=-1;
}
DFN void combat_migrate_holder(DX* c,int side,int ch){
 int h=d_holder(c,side,ch); DS* f=c->s; DR r{side,ch}; int src=d_entity(c,r);
 int atk=d_flag(c,side,ch,"atk_buff"), next=d_flag(c,side,ch,"next_attack");
 int form=d_flag(c,side,ch,"form_attack"), deb=d_flag(c,side,ch,"atk_debuff");
 int immune=d_flag(c,side,ch,"damage_immunity_until");
 int temp=f->flag_expiring_presence[h]==2?f->flag_expiring_amount[h]:0;
 int until=f->flag_expiring_presence[h]==2?f->flag_expiring_until[h]:0;
 int net=atk-temp;
 DEffect e; combat_zero_effect(&e); e.op="add"; e.scope="self"; e.stacking="independent"; e.source_entity=src; e.clear_on="down";
 if(net){ e.id="legacy.atk_buff"; e.property="attack.panel"; e.value=(double)net; e.label="atk_buff"; e.source_key="atk_buff"; d_effect(c,r,&e); }
 if(temp){ e.id="legacy.atk_buff_expiring"; e.property="attack.panel"; e.value=(double)temp; e.label="atk_buff"; e.source_key="atk_buff_expiring"; e.show_marker=0; e.expiry_side=side; e.expiry_turn=until; e.expiry_phase="turn_start"; d_effect(c,r,&e); e.expiry_phase=""; e.show_marker=-1; }
 if(deb){ e.id="legacy.atk_debuff"; e.property="attack.panel"; e.value=(double)(-deb); e.label="atk_debuff"; e.source_key="atk_debuff"; d_effect(c,r,&e); }
 if(form){ e.id="legacy.form_attack"; e.property="attack.panel"; e.value=(double)form; e.label="form_attack"; e.source_key="form_attack"; d_effect(c,r,&e); }
 int tn=f->flag_timed_count[h]; if(tn<0||tn>T){d_fail(c,1);return;}
 for(int j=0;j<tn;j++){
  int left=f->flag_timed_left[h*T+j], amount=f->flag_timed_amount[h*T+j];
  if(left<=0||!amount)continue;
  e.id="legacy.timed_attack"; e.property="attack.panel"; e.value=(double)amount; e.source_key="timed_attack";
  e.label=amount>0?"atk_buff":"atk_debuff"; e.expiry_side=side; e.expiry_turn=d_pi(c,side,"turn_count")+left; e.expiry_phase="turn_start";
  d_effect(c,r,&e); e.expiry_phase="";
 }
 if(next){
  e.id="attack.next"; e.property="attack.hit"; e.value=(double)next; e.label="next_attack"; e.source_key="next_attack";
  e.stack_key="next_attack"; e.consume_on="attack"; e.expiry_phase=""; d_effect(c,r,&e); e.consume_on=""; e.stack_key="";
 }
 if(immune){
  e.id="legacy.damage_immunity"; e.property="rule.damage_immune"; e.value=1; e.op="allow"; e.label="immune";
  e.source_key="damage_immunity_until"; e.show_marker=0; e.expiry_side=side; e.expiry_turn=immune; e.expiry_phase="turn_start";
  d_effect(c,r,&e);
 }
 d_del_flag(c,side,ch,"atk_buff"); d_del_flag(c,side,ch,"next_attack"); d_del_flag(c,side,ch,"form_attack");
 d_del_flag(c,side,ch,"atk_debuff"); d_del_flag(c,side,ch,"damage_immunity_until");
 f->flag_expiring_presence[h]=0; f->flag_expiring_amount[h]=0; f->flag_expiring_until[h]=0;
 f->flag_timed_count[h]=0; for(int j=0;j<T;j++){ f->flag_timed_amount[h*T+j]=0; f->flag_timed_left[h*T+j]=0; }
}
DFN void d_migrate_legacy(DX* c){
 for(int side=0;side<2;side++){
  combat_migrate_holder(c,side,-1);
  int n=c->s->player_character_count[side];
  for(int ch=0;ch<n;ch++) if(c->s->character_valid[side*C+ch]) combat_migrate_holder(c,side,ch);
 }
}
'''
    text += r'''
#ifndef COMBAT_D
#define COMBAT_D 4
#endif
DFN int combat_dot_kind_id(const char* k){
 if(d_streq(k,"nightmare"))return 0; if(d_streq(k,"etch"))return 1; if(d_streq(k,"venom"))return 2; if(d_streq(k,"guard"))return 3; return -1;
}
DFN int combat_dot_max(int kind){ return kind==3?1:10; }
DFN int d_burn_present(DX* c){ for(int s=0;s<2;s++) if(d_zone_has(c,s,"burn")) return 1; return 0; }
DFN int d_total_stacks(DX* c,int by){
 int total=0;
 for(int s=0;s<2;s++){
  if(d_zone_has(c,s,"burn") && d_zone_int(c,s,"burn","by",-1)==by) total+=d_zone_int(c,s,"burn","stacks",1);
  for(int i=0;i<COMBAT_D;i++) if(c->s->zone_dot_valid[s*COMBAT_D+i] && c->s->zone_dot_by[s*COMBAT_D+i]==by) total+=c->s->zone_dot_stacks[s*COMBAT_D+i];
 }
 return total;
}
DFN void d_add_stack_all(DX* c,int target_side,int by,int amount){
 if(amount==0)return;
 if(d_zone_has(c,target_side,"burn") && d_zone_int(c,target_side,"burn","by",-1)==by){
  int before=d_zone_int(c,target_side,"burn","stacks",1); int neu=before+amount; if(neu>3)neu=3;
  if(neu>before){ d_zone_set(c,target_side,"burn","stacks",neu); DEvt ev{}; ev.target_side=target_side; ev.amount=neu-before; d_fire(c,"on_burn_stacks",&ev,0,0); d_fire(c,"on_continuous_applied",&ev,0,0); }
 }
 for(int i=0;i<COMBAT_D;i++){
  int at=target_side*COMBAT_D+i; if(!c->s->zone_dot_valid[at]||c->s->zone_dot_by[at]!=by)continue;
  int before=c->s->zone_dot_stacks[at], cap=combat_dot_max(c->s->zone_dot_kind[at]); int neu=before+amount; if(neu>cap)neu=cap;
  if(neu>before){ c->s->zone_dot_stacks[at]=neu; DEvt ev{}; ev.target_side=target_side; d_fire(c,"on_continuous_applied",&ev,0,0); }
 }
}
DFN int d_apply_dot(DX* c,int target_side,const char* kind,int by,int source,int stacks,int duration){
 int k=combat_dot_kind_id(kind); if(k<0||target_side<0||target_side>1){d_fail(c,1);return 0;}
 d_compact_dots(c,target_side);if(stacks<0)stacks=0; int cap=combat_dot_max(k), empty=-1, found=-1,old=-1;
 for(int i=0;i<COMBAT_D;i++){
  int at=target_side*COMBAT_D+i;
  if(!c->s->zone_dot_valid[at]){ if(empty<0)empty=i; continue; }
  if(c->s->zone_dot_kind[at]==k){if(c->s->zone_dot_by[at]==by)found=i;else old=i;}
 }
 if(found<0&&old>=0)empty=old;c->s->zone_dots_presence[target_side]=1;int unique=k==3;int source_slot=d_ref_target_slot(c,DR{by,source});
 if(found>=0 && unique){
  int at=target_side*COMBAT_D+found;
  if(duration>=0){ c->s->zone_dot_left_presence[at]=2; c->s->zone_dot_left[at]=duration; }
  c->s->zone_dot_source_slot[at]=source_slot;
  DEvt ev{}; ev.target_side=target_side; d_event(c);d_fire(c,"on_continuous_applied",&ev,0,0); return 0;
 }
 int gained=0;
 if(found>=0){
  int at=target_side*COMBAT_D+found; int before=c->s->zone_dot_stacks[at]; int neu=before+stacks; if(neu>cap)neu=cap;
  c->s->zone_dot_stacks[at]=neu; gained=neu-before;
  c->s->zone_dot_source_slot[at]=source_slot;
  if(duration>=0){ c->s->zone_dot_left_presence[at]=2; c->s->zone_dot_left[at]=duration; }
 } else {
  if(empty<0){d_fail(c,2);return 0;}
  int at=target_side*COMBAT_D+empty; gained=stacks<cap?stacks:cap;
  c->s->zone_dot_valid[at]=1; c->s->zone_dot_kind[at]=(signed char)k; c->s->zone_dot_stacks[at]=gained;
  c->s->zone_dot_by[at]=(signed char)by; c->s->zone_dot_source_slot[at]=source_slot;
  if(duration>=0){ c->s->zone_dot_left_presence[at]=2; c->s->zone_dot_left[at]=duration; } else {c->s->zone_dot_left_presence[at]=1;c->s->zone_dot_left[at]=0;}
  c->s->zone_dot_applied_turn_presence[at]=2; c->s->zone_dot_applied_turn[at]=c->s->turn[0];
 }
 if(gained||unique){ DEvt ev{}; ev.target_side=target_side; d_event(c);d_fire(c,"on_continuous_applied",&ev,0,0); }
 return gained;
}
DFN int combat_attr_from_name(const char* name){
 if(!name)return 0;
 if(d_streq(name,"light")||d_streq(name,"\xe5\x85\x89"))return 1;
 if(d_streq(name,"dark")||d_streq(name,"\xe6\x9a\x97"))return 2;
 if(d_streq(name,"spirit")||d_streq(name,"\xe7\x81\xb5"))return 3;
 if(d_streq(name,"soul")||d_streq(name,"\xe9\xad\x82"))return 4;
 if(d_streq(name,"phase")||d_streq(name,"\xe7\x9b\xb8"))return 5;
 if(d_streq(name,"curse")||d_streq(name,"hex")||d_streq(name,"\xe5\x92\x92"))return 6;
 return 0;
}
DFN void d_halve_dots(DX* c,int side){ combat_halve_dots(c,side); }
DFN void d_tick_owner_end(DX* c,int side){ combat_tick_owner_end(c,side); }
DFN void d_trigger_dot(DX* c,int target_side,const char* kind){
 int k=combat_dot_kind_id(kind); if(k<0){d_fail(c,4);return;}
 for(int i=0;i<COMBAT_D;i++) if(c->s->zone_dot_valid[target_side*COMBAT_D+i] && c->s->zone_dot_kind[target_side*COMBAT_D+i]==k){ combat_trigger_dot(c,target_side,i); return; }
}
DFN void d_additional_attack(DX* c,const char* attribute,DR target,int sync_attack,int support_actor,int front_only){
 int attr=combat_attr_from_name(attribute); int has=target.side>=0 && target.slot>=-1;
 combat_additional_attack(c,attr,target,has,sync_attack,support_actor,front_only);
}
DFN void d_apply_enter_harmony(DX* c,int payer,int kind,int as_support){ combat_apply_enter_harmony(c,payer,kind,as_support,1); }
DFN void d_force_collapse(DX* c,int side,int slot,int by){ combat_force_collapse(c,DR{side,slot},by); }
DFN void d_apply_dark_star(DX* c,int renewals){
 int side=c->side, ch=c->actor; DR ft=d_front_target(c); int enemy=ft.side;
 d_set_pi(c,side,"dark_star_count",d_pi(c,side,"dark_star_count")+1);
 int had_burn=d_zone_has(c,enemy,"burn");
 int renew=renewals>=0?renewals:d_decide(c,"dark_star_renewals",0,nullptr,0,0);
 d_zone_set(c,enemy,"star","left",2); d_zone_set(c,enemy,"star","by",side); d_zone_set(c,enemy,"star","renewals",renew);
 d_zone_source(c,enemy,"star",side,ch); d_event(c);
 if(had_burn){
  if(ft.slot>=0 && d_alive(c,enemy,ft.slot)) d_add_collapse(c,ft,1,side);
  d_event(c); DEvt ev{}; ev.target_side=enemy; ev.target=ft.slot; d_fire(c,"on_dissonance",&ev,0,0);
 }
}
'''
    fields=[name for name in schema.array_specs(layout or schema.DomainSoALayout()) if name.startswith('zone_dot_')]
    compact='DFN void d_compact_dots(DX*c,int side){int write=0;for(int read=0;read<COMBAT_D;read++){int src=side*COMBAT_D+read;if(!c->s->zone_dot_valid[src])continue;int dst=side*COMBAT_D+write;'
    for name in fields:compact+=f'c->s->{name}[dst]=c->s->{name}[src];'
    compact+='write++;}for(int j=write;j<COMBAT_D;j++){int at=side*COMBAT_D+j;'
    for name in fields:compact+=f'c->s->{name}[at]=0;'
    compact+='}}\n'
    return compact+text
