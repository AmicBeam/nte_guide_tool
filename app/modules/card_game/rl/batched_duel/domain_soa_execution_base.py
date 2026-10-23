"""Typed direct-C++ execution ABI shared by the no-VM physical handlers.

This builds named row accessors and domain primitives. No arbitrary objects,
Python opcodes, per-root callbacks, or interpreter state are represented.
"""
import re
from math import prod
import numpy as np
from . import domain_soa_schema as schema
from .domain_soa_character_queries import _source as query_source
from .random_source import emit_random_source

supports_full_duel_rules=False

_TYPES={'bool':'unsigned char','uint8':'unsigned char','int8':'signed char','int16':'short','int32':'int','int64':'long long','float64':'double'}

_DECLARATIONS=r'''
typedef QRow DS;
struct DR {int side;int slot;};
struct DTrace {double attack,attack_count,genesis,counter;int has_attack;};
struct DX {DS*s;int side,actor,card;DR target;int depth;int effect_source_slot=-1;int choice_selected=-1;const int*choice_bottom=nullptr;int choice_bottom_count=0;DTrace*trace=nullptr;int option_slot=-1,target_card=-1,choice_execute=1,damage_source_slot=-1;};
struct DAttack {int bonus,ignore_shield,no_counter,no_attack,hits,return_after,draw_after,low_hp_bonus,low_hp_max,awake_player_bonus,has_awake_player_bonus,harmony_bonus,additional_damage_bonus;};
struct DEvt {int actor,card,attacker_side,attacker,defender_side,defender,target_side,target,amount,loss,absorbed,raw_overflow,kind,attribute,additional,index,response,entering;int effect_definition_slot=-1,effect_source_entity=0,holder_entity=0,deferred_kind_slot=-1,deferred_attribute=0;int support=0,sync_attack=0,deferred_support_actor=-1,deferred_support_entity=0;};
struct DEffect {const char*id;const char*property;double value;const char*op;int layer;const char*scope;const char*stacking;const char*stack_key;int expiry_side,expiry_turn;const char*expiry_phase;const char*consume_on;const char*clear_on;const char*condition;const char*condition_key;const char*condition_string;double condition_number;const char*label=nullptr;const char*source_key=nullptr;int source_entity=-1;const char*clear_on2=nullptr;int max_stacks=0;int show_marker=-1;int value_kind=-1;const char*kind=nullptr;const char*visibility=nullptr;const char*clear_on3=nullptr;int clear_count=-1;};
DFN int d_pi(DX*,int,const char*);DFN void d_set_pi(DX*,int,const char*,int);DFN int d_pb(DX*,int,const char*);DFN void d_set_pb(DX*,int,const char*,int);
DFN int d_ci(DX*,int,int,const char*);DFN void d_set_ci(DX*,int,int,const char*,int);DFN int d_cb(DX*,int,int,const char*);DFN void d_set_cb(DX*,int,int,const char*,int);
DFN int d_flag(DX*,int,int,const char*);DFN void d_set_flag(DX*,int,int,const char*,int,int is_bool=0);DFN void d_del_flag(DX*,int,int,const char*);
DFN int d_cstr_is(DX*,int,int,const char*,const char*);DFN void d_set_cstr(DX*,int,int,const char*,const char*);DFN void d_del_cstr(DX*,int,int,const char*);
DFN int d_card_int(DX*,int,const char*);DFN int d_card_bool(DX*,int,const char*);DFN int d_card_is(DX*,int,const char*);DFN int d_card_owner(DX*,int);
DFN int d_hero(DX*,int,const char*);DFN int d_template(DX*,int,int);DFN int d_alive(DX*,int,int);DFN int d_finished(DX*);DFN int d_collapsed(DX*,int,int);
DFN int d_front(DX*,int);DFN void d_set_front(DX*,int,int);DFN int d_order_count(DX*,int);DFN int d_order_slot(DX*,int,int);DFN int d_enemy(int);DFN int d_entity(DX*,DR);DFN DR d_front_target(DX*);
DFN int d_energy_cap(DX*,int,int);DFN int d_harmony_cap(DX*);DFN int d_attack_value(DX*,int,int,int include_zone=1);DFN int d_permanent_attack(DX*,int,int);DFN int d_max_hp(DX*,int,int);DFN int d_zone_penalty(DX*,int,int);DFN int d_can_gain_energy(DX*,int,int);DFN int d_has_shape(DX*,const char*);
DFN void d_event(DX*);DFN void d_finish(DX*,int,const char*);DFN void d_fail(DX*,int);DFN int d_bad(DX*);
DFN void d_draw(DX*,int,int,int type=-1);DFN void d_add_hand(DX*,int,int,int reveal=0);DFN void d_discard(DX*,int,int,int bottom=0);DFN int d_make_card(DX*,int,const char*,int owner=-1);DFN int d_copy_card(DX*,int,int,int);
DFN void d_remove_zone_card(DX*,int,int,int);DFN void d_shuffle_zone(DX*,int,int);DFN int d_randbelow(DX*,int);
DFN void d_energy(DX*,int,int,int);DFN void d_shield(DX*,DR,int);DFN void d_heal(DX*,DR,int,DR);DFN void d_grow(DX*,int,int,int,int);DFN void d_move_out(DX*,int,int,int reason=0);DFN void d_set_shape(DX*);DFN void d_revive(DX*,int,int);DFN void d_draw_owned(DX*,int,int reveal=1);DFN void d_next_attack(DX*,DR,int);
DFN void d_effect(DX*,DR,const DEffect*);DFN void d_clear_effect_id(DX*,DR,const char*);DFN double d_evaluate(DX*,DR,const char*,double,const DEvt*event=nullptr);DFN void d_consume(DX*,const char*);
DFN double d_evaluate_entries(DX*,DR,const char*,double,const DEvt*,const Candidate*,int);DFN void d_consume_applied(DX*,DR,const char*,const char*,const DEvt*);
DFN void d_schedule_ultimate(DX*,int,int,int,const char*);DFN void d_end_ultimate(DX*,int,int);DFN void d_migrate_legacy(DX*);
DFN void d_clear_effects(DX*,const char*);DFN void d_expire_effects(DX*,int,const char*);DFN void d_prepare_effects(DX*);
DFN int d_zone_has(DX*,int,const char*);DFN int d_zone_int(DX*,int,const char*,const char*,int fallback=0);DFN void d_zone_set(DX*,int,const char*,const char*,int);DFN void d_zone_remove(DX*,int,const char*);DFN void d_zone_source(DX*,int,const char*,int,int);
DFN int d_op_int(DX*,const char*,int fallback=0);DFN void d_set_op_int(DX*,const char*,int,int is_bool=0);DFN void d_del_op_int(DX*,const char*);DFN DR d_op_target(DX*,const char*key="target_id");DFN void d_set_op_target(DX*,const char*,DR);
DFN void d_open_operation(DX*);DFN void d_set_pending_sortie(DX*,const DAttack*);DFN void d_defer(DX*,const char*,int,int amount=0);DFN void d_choice(DX*,const char*,int count=0);DFN int d_summon(DX*,const char*);
DFN void d_card_effect(DX*);DFN void d_fire(DX*,const char*,const DEvt*event=nullptr,int actor_only=0,int include_down=0);DFN int d_decide(DX*,const char*,int,const DEvt*event=nullptr,int actor_only=0,int include_down=0);DFN int d_tally(DX*,const char*,const DEvt*event=nullptr);DFN void d_on_shape(DX*,const char*,const DEvt*event=nullptr);
DFN int d_damage(DX*,DR,int,DR,int kind=0,int bypass=0,int settle=1,int attribute=0,int additional=0,int counter_immunity=0);DFN void d_knockdowns(DX*);DFN void d_sortie(DX*,const DAttack*spec=nullptr);DFN void d_enter(DX*,int as_support=1,int resolve_harmony=1);DFN int d_harmony_kind(DX*);DFN void d_resolve_harmony(DX*,int);DFN void d_resolve_genesis(DX*,int arm_delay=1);DFN void d_add_collapse(DX*,DR,int,int);
DFN void d_begin_turn(DX*,int);DFN void d_end_turn(DX*);DFN void d_finish_operation(DX*);DFN void d_drain(DX*);DFN void d_play_card(DX*,int response=0,int release_only=0);DFN void d_trigger_responses(DX*,int,const char*,int target=-1,int attacker=-1);DFN void d_choose(DX*,int,const int*,int,const int*,int,const int*,int,int);
DFN int d_used(DX*,int,const char*);DFN void d_set_used(DX*,int,const char*,int,int is_bool=1);DFN int d_once(DX*,const char*);
DFN int d_intern(DX*,const char*);DFN int d_eq(DX*,int,const char*);DFN int d_streq(const char*,const char*);
'''


def _literal(text):return '"'+''.join('\\x%02x'%b for b in text.encode('utf8'))+'"'


def emit_header(layout):
    text,_=query_source(layout)
    text=text[:text.index('QFN void run_query(')]
    specs=schema.array_specs(layout)
    members=''.join(_TYPES[np.dtype(dtype).name]+'* '+name+';\n'for name,(dtype,shape)in specs.items())
    text=re.sub(r'struct QRow\{.*?\};',lambda _: 'struct QRow{'+members+'int choice_serial;int error_site;};',text,count=1,flags=re.S)
    text+='\n#if defined(__CUDACC__) || defined(__CUDACC_RTC__)\n#define DFN __device__\n#define BMT_DEVICE __device__\n#else\n#define DFN inline\n#define BMT_DEVICE\n#endif\n'+emit_random_source()
    values=dict(K=layout.max_cards,Z=layout.max_zone,CH=layout.max_choice_cards,CL=layout.max_clear_on,DF=layout.max_deferred,DRMAX=layout.max_down_responses,COMBAT_D=layout.max_dots,PART=layout.max_participants,
        KB=len(schema.CARD_BOOL_FIELDS),KI=len(schema.CARD_INT_FIELDS),KS=len(schema.CARD_STR_FIELDS),PB=len(schema.PLAYER_BOOL_FIELDS),PS=len(schema.PLAYER_STR_FIELDS),FB=len(schema.FLAG_BOOL_FIELDS),FS=len(schema.FLAG_STR_FIELDS),NU=layout.max_used,
        NOPINT=len(schema.OP_INT_KEYS),NOPBOOL=len(schema.OP_FLAG_KEYS),NTCARD=max(schema.CARD_VOCAB.values())+1)
    text+=''.join('#define '+k+' '+str(v)+'\n'for k,v in values.items())
    for prefix,names in [('PB',schema.PLAYER_BOOL_FIELDS),('CB',schema.CHARACTER_BOOL_FIELDS),('KB',schema.CARD_BOOL_FIELDS),('KI',schema.CARD_INT_FIELDS),('KS',schema.CARD_STR_FIELDS),('FB',schema.FLAG_BOOL_FIELDS),('FS',schema.FLAG_STR_FIELDS)]:
        text+=''.join('#define '+prefix+'_'+name+' '+str(i)+'\n'for i,name in enumerate(names))
    text+=''.join('#define CARD_'+cid+' '+str(v)+'\n'for cid,v in schema.CARD_VOCAB.items())
    return text+_DECLARATIONS


_SIMPLE=r'''
DFN int d_streq(const char*a,const char*b){if(!a||!b)return a==b;int n=0;while(a[n]&&a[n]==b[n])n++;return a[n]==b[n];}
DFN void d_fail(DX*c,int error){if(c->s->row_error[0]==0)c->s->row_error[0]=error;}
DFN void d_note_site(DX*c,int site){if(c->s->row_error[0]==0)c->s->error_site=site;}
#define d_fail(c,error) (d_note_site(c,__LINE__),d_fail(c,error))
DFN int d_bad(DX*c){return c->s->row_error[0]!=0;}
DFN int d_finished(DX*c){return c->s->phase[0]==3;}
DFN int d_enemy(int side){return 1-side;}
DFN int d_valid_side(DX*c,int s){if(s<0||s>1){d_fail(c,1);return 0;}return 1;}
DFN int d_valid_char(DX*c,int s,int ch){if(!d_valid_side(c,s)||ch<0||ch>=c->s->player_character_count[s]||ch>=C||!c->s->character_valid[s*C+ch]){d_fail(c,1);return 0;}return 1;}
DFN int d_holder(DX*c,int s,int ch){if(!d_valid_side(c,s))return 0;if(ch>=0&&!d_valid_char(c,s,ch))return 0;return s*(C+1)+1+ch;}
DFN int d_eq(DX*c,int slot,const char*value){if(!value)return 0;int err=0;int result=qeq(c->s,slot,value,&err);if(err)d_fail(c,err);return result;}
DFN int d_intern(DX*c,const char*value){
 if(!value){d_fail(c,1);return 0;}DS*f=c->s;int n=0;while(value[n]){n++;if(n>SB){d_fail(c,2);return 0;}}
 if(f->string_count[0]<0||f->string_count[0]>SS||f->string_bytes_used[0]<0||f->string_bytes_used[0]>SB){d_fail(c,1);return 0;}
 for(int j=0;j<f->string_count[0];j++)if(d_eq(c,j,value))return j;
 if(d_bad(c))return 0;int slot=f->string_count[0],at=f->string_bytes_used[0];if(slot>=SS||at+n>SB){d_fail(c,2);return 0;}
 f->string_count[0]++;f->string_bytes_used[0]+=n;f->string_offset[slot]=at;f->string_length[slot]=n;for(int j=0;j<n;j++)f->string_bytes[at+j]=(unsigned char)value[j];return slot;
}
DFN void d_event(DX*c){if(c->s->event_seq[0]>=2147483647)d_fail(c,2);else c->s->event_seq[0]++;}
DFN void d_finish(DX*c,int winner,const char*reason){if(d_finished(c)||d_bad(c))return;if(winner< -1||winner>1){d_fail(c,1);return;}int slot=d_intern(c,reason);if(d_bad(c))return;c->s->phase[0]=3;c->s->winner_presence[0]=2;c->s->winner[0]=winner<0?2:winner;c->s->reason_presence[0]=2;c->s->reason_slot[0]=slot;d_event(c);}
DFN int d_template(DX*c,int side,int ch){return d_valid_char(c,side,ch)?c->s->character_template[side*C+ch]:0;}
DFN int d_alive(DX*c,int side,int ch){return d_valid_char(c,side,ch)&&c->s->character_hp[side*C+ch]>0;}
DFN int d_collapsed(DX*c,int side,int ch){if(!d_valid_char(c,side,ch))return 0;return c->s->flag_collapse_presence[side*(C+1)+1+ch]==2;}
DFN int d_entity(DX*c,DR r){if(!d_valid_side(c,r.side))return 0;if(r.slot<0)return c->s->player_entity_slot[r.side];return d_valid_char(c,r.side,r.slot)?c->s->character_entity_slot[r.side*C+r.slot]:0;}
DFN int d_front(DX*c,int side){if(!d_valid_side(c,side))return -1;return c->s->player_front_presence[side]==2?c->s->player_front_slot[side]:-1;}
DFN void d_set_front(DX*c,int side,int ch){if(!d_valid_side(c,side))return;if(ch>=0&&!d_valid_char(c,side,ch))return;c->s->player_front_presence[side]=ch>=0?2:1;c->s->player_front_slot[side]=ch>=0?ch:0;}
DFN int d_order_count(DX*c,int side){if(!d_valid_side(c,side))return 0;return c->s->player_order_count[side]?c->s->player_order_count[side]:c->s->player_character_count[side];}
DFN int d_order_slot(DX*c,int side,int i){int n=d_order_count(c,side);if(i<0||i>=n){d_fail(c,1);return -1;}return c->s->player_order_count[side]?c->s->player_order_slot[side*C+i]:i;}
DFN DR d_front_target(DX*c){int side=1-c->side;return DR{side,d_front(c,side)};}
DFN int d_harmony_cap(DX*c){return 2-(c->s->escalation_enabled[0]&&c->s->turn[0]>=13);}
DFN int d_can_gain_energy(DX*c,int side,int ch){if(!d_alive(c,side,ch)||d_cb(c,side,ch,"summoned"))return 0;DX x=*c;x.side=side;x.actor=ch;x.card=-1;return d_evaluate(&x,DR{side,ch},"rule.gain_energy",d_decide(&x,"can_gain_energy",1,nullptr,1))!=0;}
DFN int d_energy_cap(DX*c,int side,int ch){if(!d_valid_char(c,side,ch))return 0;int id=d_template(c,side,ch);int cap=id?cenergy[id]:((attr(c->s,d_holder(c,side,ch),c->s->row_error)==2||attr(c->s,d_holder(c,side,ch),c->s->row_error)==4||attr(c->s,d_holder(c,side,ch),c->s->row_error)==6)?6:5);if(!id||cruntime_energy[id]){int i=(side*C+ch)*NC+CI_energy_max_override;if(c->s->character_int_presence[i]==2)cap=c->s->character_int_value[i];}return cap-(c->s->escalation_enabled[0]&&c->s->turn[0]>=13);}
DFN void d_energy(DX*c,int side,int ch,int n){if(d_finished(c)||!d_can_gain_energy(c,side,ch))return;long long value=(long long)c->s->character_energy[side*C+ch]+n;int cap=d_energy_cap(c,side,ch);if(value>cap)value=cap;if(value< -2147483648LL){d_fail(c,2);return;}c->s->character_energy[side*C+ch]=(int)value;}
DFN int d_max_hp(DX*c,int side,int ch){return ch<0?c->s->player_max_hp[side]:d_ci(c,side,ch,"max_hp");}
DFN int d_permanent_attack(DX*c,int side,int ch){if(!d_valid_char(c,side,ch))return 0;double out[NO]={0};int err=0;character_query(c->s,side,ch,out,&err);if(err)d_fail(c,err);return (int)out[OUT_base_attack];}
DFN int d_zone_penalty(DX*c,int side,int ch){if(d_front(c,side)!=ch)return 0;return d_zone_int(c,side,"delay","slow_turn",c->s->turn[0])==c->s->turn[0]?d_zone_int(c,side,"delay","slow",0):0;}
DFN int d_attack_value(DX*c,int side,int ch,int include_zone){if(!d_valid_char(c,side,ch))return 0;double out[NO]={0};int err=0;unsigned char saved=c->s->zone_status_presence[side*5];if(!include_zone)c->s->zone_status_presence[side*5]=0;character_query(c->s,side,ch,out,&err);c->s->zone_status_presence[side*5]=saved;if(err)d_fail(c,err);return (int)out[OUT_attack];}
DFN int d_has_shape(DX*c,const char*id){return d_cstr_is(c,c->side,c->actor,"shape",id);}
'''


def emit_accessors():
    text=_SIMPLE
    def named(names,template):return ''.join('if(d_streq(key,'+_literal(k)+')){'+template(i,k)+'}'for i,k in enumerate(names))
    text+='DFN int d_pi(DX*c,int side,const char*key){if(!d_valid_side(c,side))return 0;DS*f=c->s;'
    for k,field in [('hp','player_hp'),('shield','player_shield'),('ap','player_ap'),('max_hp','player_max_hp')]:text+='if(d_streq(key,"'+k+'"))return f->'+field+'[side];'
    text+=named(schema.PLAYER_INT_FIELDS,lambda i,k:f'return f->player_int_presence[side*NP+{i}]>=2?f->player_int_value[side*NP+{i}]:0;')+'d_fail(c,4);return 0;}\n'
    text+='DFN void d_set_pi(DX*c,int side,const char*key,int value){if(!d_valid_side(c,side))return;DS*f=c->s;'
    for k,field in [('hp','player_hp'),('shield','player_shield'),('ap','player_ap'),('max_hp','player_max_hp')]:text+='if(d_streq(key,"'+k+'")){f->'+field+'[side]=value;return;}'
    text+=named(schema.PLAYER_INT_FIELDS,lambda i,k:f'f->player_int_presence[side*NP+{i}]={2};f->player_int_value[side*NP+{i}]=value;return;')+'d_fail(c,4);}\n'
    for getset in ('get','set'):
        head='DFN int d_pb(DX*c,int side,const char*key){if(!d_valid_side(c,side))return 0;'if getset=='get'else'DFN void d_set_pb(DX*c,int side,const char*key,int value){if(!d_valid_side(c,side))return;'
        text+=head+'DS*f=c->s;'+named(schema.PLAYER_BOOL_FIELDS,lambda i,k: f'return f->player_bool_presence[side*PB+{i}]==2&&f->player_bool_value[side*PB+{i}];'if getset=='get'else f'f->player_bool_presence[side*PB+{i}]=2;f->player_bool_value[side*PB+{i}]=value!=0;return;')+('d_fail(c,4);return 0;}\n'if getset=='get'else'd_fail(c,4);}\n')
    text+='DFN int d_ci(DX*c,int side,int ch,const char*key){if(!d_valid_char(c,side,ch))return 0;DS*f=c->s;int h=side*C+ch;'
    for key,field in [('hp','character_hp'),('shield','character_shield'),('energy','character_energy')]:text+='if(d_streq(key,"'+key+'"))return f->'+field+'[h];'
    text+=named(schema.CHARACTER_INT_FIELDS,lambda i,k:f'int p=f->character_int_presence[h*NC+{i}];return p==2?f->character_int_value[h*NC+{i}]:p==3?cdefs[f->character_template[h]*NC+{i}]:0;')+'d_fail(c,4);return 0;}\n'
    text+='DFN void d_set_ci(DX*c,int side,int ch,const char*key,int value){if(!d_valid_char(c,side,ch))return;DS*f=c->s;int h=side*C+ch;'
    for key,field in [('hp','character_hp'),('shield','character_shield'),('energy','character_energy')]:text+='if(d_streq(key,"'+key+'")){f->'+field+'[h]=value;f->'+field+'_presence[h]=2;return;}'
    text+=named(schema.CHARACTER_INT_FIELDS,lambda i,k:f'f->character_int_presence[h*NC+{i}]=2;f->character_int_value[h*NC+{i}]=value;return;')+'d_fail(c,4);}\n'
    for setting in (False,True):
        text+=('DFN void d_set_cb(DX*c,int side,int ch,const char*key,int value){if(!d_valid_char(c,side,ch))return;'if setting else'DFN int d_cb(DX*c,int side,int ch,const char*key){if(!d_valid_char(c,side,ch))return 0;')+'DS*f=c->s;int h=side*C+ch;'
        text+=named(schema.CHARACTER_BOOL_FIELDS,lambda i,k: f'f->character_bool_presence[h*NB+{i}]=2;f->character_bool_value[h*NB+{i}]=value!=0;return;'if setting else f'return f->character_bool_presence[h*NB+{i}]==2&&f->character_bool_value[h*NB+{i}];')+('d_fail(c,4);}\n'if setting else'd_fail(c,4);return 0;}\n')
    for mode in ('get','set','del'):
        signature={'get':'DFN int d_flag(DX*c,int side,int ch,const char*key)', 'set':'DFN void d_set_flag(DX*c,int side,int ch,const char*key,int value,int is_bool)','del':'DFN void d_del_flag(DX*c,int side,int ch,const char*key)'}[mode]
        text+=signature+'{int h=d_holder(c,side,ch);DS*f=c->s;'
        for kind,names,pref,dim in [('int',schema.FLAG_INT_FIELDS,'flag_int','NF'),('bool',schema.FLAG_BOOL_FIELDS,'flag_bool','FB')]:
            if mode=='get':text+=named(names,lambda i,k:f'return f->{pref}_presence[h*{dim}+{i}]>=2?f->{pref}_value[h*{dim}+{i}]:0;')
            else:text+=named(names,lambda i,k:f'f->{pref}_presence[h*{dim}+{i}]='+str(2 if mode=='set'else 0)+f';f->{pref}_value[h*{dim}+{i}]='+('value'if mode=='set'and kind=='int'else'value!=0'if mode=='set'else'0')+';return;')
        text+=('d_fail(c,4);return 0;}\n'if mode=='get'else'd_fail(c,4);}\n')
    text+='DFN void d_del_cb(DX*c,int side,int actor,const char*key){if(!d_valid_char(c,side,actor))return;'
    for i,key in enumerate(schema.CHARACTER_BOOL_FIELDS):text+=f'if(d_streq(key,"{key}")){{c->s->character_bool_presence[(side*C+actor)*NB+{i}]=0;return;}}'
    text+='d_fail(c,4); }\n'
    return text


def emit_player_deletions():
    text=''
    for typ,keys,dim in [('pi',schema.PLAYER_INT_FIELDS,'NP'),('pb',schema.PLAYER_BOOL_FIELDS,'PB'),('pstr',schema.PLAYER_STR_FIELDS,'PS')]:
        prefix={'pi':'int','pb':'bool','pstr':'str'}[typ]
        text+=f'DFN void d_del_{typ}(DX*c,int side,const char*key){{if(!d_valid_side(c,side))return;'
        for i,key in enumerate(keys):
            text+=f'if(d_streq(key,"{key}")){{c->s->player_{prefix}_presence[side*{dim}+{i}]=0;return;}}'
        text+='d_fail(c,4); }\n'
    return text


def emit_strings_and_cards():
    from app.modules.card_game.content.duel_v2 import CHARACTERS
    from app.modules.card_game.content.duel_v2.tutorial import lookup_card
    specs=schema.array_specs(schema.DomainSoALayout())
    n=max(schema.CARD_VOCAB.values())+1
    cards={i:lookup_card(cid)for cid,i in schema.CARD_VOCAB.items()}
    text='DFN int d_card_template(DX*c,int card){if(card<=-2)return -card-2;if(card<0||card>=K||!c->s->card_valid[card]){d_fail(c,1);return 0;}return c->s->card_template[card];}\n'
    text+='DFN const char*d_card_name(int id){switch(id){'+''.join('case '+str(v)+':return "'+cid+'";'for cid,v in schema.CARD_VOCAB.items())+'default:return "";}}\n'
    text+='DFN const char*d_character_name(int id){switch(id){'+''.join('case '+str(v)+':return "'+cid+'";'for cid,v in schema.CHARACTER_VOCAB.items())+'default:return "";}}\n'
    text+='DFN int d_card_id(const char*id){'+''.join('if(d_streq(id,"'+cid+'"))return '+str(v)+';'for cid,v in schema.CARD_VOCAB.items())+'return 0;}\n'
    for typ,keys in [('int',schema.CARD_INT_FIELDS),('bool',schema.CARD_BOOL_FIELDS),('str',schema.CARD_STR_FIELDS)]:
        ret='const char*'if typ=='str'else'int'
        text+='DFN '+ret+' d_card_default_'+typ+'(int id,int key){switch(id){'
        for i,card in cards.items():
            text+='case '+str(i)+':switch(key){'
            for k,name in enumerate(keys):
                v=card.get(name)
                if v is not None:
                    val=_literal(v)if typ=='str'else str(int(v))
                    text+='case '+str(k)+':return '+val+';'
            text+='default:return '+('nullptr'if typ=='str'else'0')+';}'
        text+='default:return '+('nullptr'if typ=='str'else'0')+';}}\n'
    text+='DFN int d_hero(DX*c,int side,const char*id){if(!d_valid_side(c,side))return -1;for(int j=0;j<c->s->player_character_count[side];j++)if(d_eq(c,c->s->character_id_slot[side*C+j],id))return j;return -1;}\n'
    text+='DFN int d_cstr_is(DX*c,int side,int ch,const char*key,const char*value){if(!d_valid_char(c,side,ch))return 0;DS*f=c->s;int h=side*C+ch;'
    for i,key in enumerate(schema.CHARACTER_STR_FIELDS):
        text+='if(d_streq(key,"'+key+'")){int p=f->character_str_presence[h*NS+'+str(i)+'];if(p==2)return d_eq(c,f->character_str_slot[h*NS+'+str(i)+'],value);if(p!=3)return !value;switch(f->character_template[h]){'
        for cid,t in schema.CHARACTER_VOCAB.items():
            value=CHARACTERS[cid].get(key)
            if isinstance(value,str):text+='case '+str(t)+':return d_streq(value,'+_literal(value)+');'
        text+='default:return !value;}}'
    text+='d_fail(c,4);return 0;}\n'
    for mode in ('set','del'):
        text+=('DFN void d_set_cstr(DX*c,int side,int ch,const char*key,const char*value)'if mode=='set'else'DFN void d_del_cstr(DX*c,int side,int ch,const char*key)')+'{if(!d_valid_char(c,side,ch))return;DS*f=c->s;int h=side*C+ch;'
        for i,key in enumerate(schema.CHARACTER_STR_FIELDS):
            value='value?d_intern(c,value):0'if mode=='set'else'0';p='value?2:1'if mode=='set'else'0'
            text+='if(d_streq(key,"'+key+'")){f->character_str_presence[h*NS+'+str(i)+']='+p+';f->character_str_slot[h*NS+'+str(i)+']='+value+';return;}'
        text+='d_fail(c,4);}\n'
    for typ,keys,dim in [('int',schema.CARD_INT_FIELDS,'KI'),('bool',schema.CARD_BOOL_FIELDS,'KB')]:
        text+='DFN int d_card_'+typ+'(DX*c,int card,const char*key){int id=d_card_template(c,card);if(d_bad(c))return 0;int k=-1;'
        text+=''.join('if(d_streq(key,"'+name+'"))k='+str(i)+';'for i,name in enumerate(keys))
        text+=('if(d_streq(key,"free_if_surplus"))return 0;'if typ=='bool'else'')
        text+='if(k<0){d_fail(c,4);return 0;}if(card<=-2)return d_card_default_'+typ+'(id,k);int p=c->s->card_'+typ+'_presence[card*'+dim+'+k];return p==2?c->s->card_'+typ+'_value[card*'+dim+'+k]:p==3?d_card_default_'+typ+'(id,k):0;}\n'
    text+='''DFN int d_card_is(DX*c,int card,const char*id){if(card==-1)return 0;return d_card_template(c,card)==d_card_id(id);}
DFN int d_card_owner(DX*c,int card){if(card<=-2)return c->actor;if(card<0||card>=K){d_fail(c,1);return -1;}int id=d_card_template(c,card),p=c->s->card_str_presence[card*KS+KS_character_id];for(int j=0;j<c->s->player_character_count[c->side];j++){if(p==2){int err=0;if(qslots(c->s,c->s->card_str_slot[card*KS+KS_character_id],c->s->character_id_slot[c->side*C+j],&err))return j;if(err)d_fail(c,err);}else if(p==3){const char*name=d_card_default_str(id,KS_character_id);if(d_eq(c,c->s->character_id_slot[c->side*C+j],name))return j;}}return -1;}
DFN void d_uint_text(char*out,int value){int j=0;if(value<0){*out++='-';value=-value;}char rev[12];do{rev[j++]=(char)('0'+value%10);value/=10;}while(value);for(int k=0;k<j;k++)out[k]=rev[j-1-k];out[j]=0;}
DFN int d_identity_text(DX*c,int id,const char*suffix,int serial){char buf[128];buf[0]='e';d_uint_text(buf+1,id);int n=0;while(buf[n])n++;for(int j=0;suffix[j]&&n<110;j++)buf[n++]=suffix[j];d_uint_text(buf+n,serial);return d_intern(c,buf);}
DFN int d_used(DX*c,int side,const char*key){if(!d_valid_side(c,side))return 0;int n=c->s->player_used_count[side];if(n<0||n>NU){d_fail(c,1);return 0;}for(int j=0;j<n;j++){int at=side*NU+j;if(d_eq(c,c->s->player_used_key_slot[at],key))return c->s->player_used_value_kind[at]==0?c->s->player_used_value[at]:c->s->player_used_int[at];}return 0;}
DFN void d_set_used(DX*c,int side,const char*key,int value,int is_bool){if(!d_valid_side(c,side))return;int n=c->s->player_used_count[side];if(n<0||n>NU){d_fail(c,1);return;}int at=-1;for(int j=0;j<n;j++)if(d_eq(c,c->s->player_used_key_slot[side*NU+j],key))at=j;if(at<0){if(n==NU){d_fail(c,2);return;}at=n;c->s->player_used_count[side]++;}at+=side*NU;c->s->player_used_key_slot[at]=d_intern(c,key);c->s->player_used_value_kind[at]=is_bool?0:1;c->s->player_used_value[at]=value!=0;c->s->player_used_int[at]=is_bool?0:value;}
DFN int d_once(DX*c,const char*key){if(d_used(c,c->side,key))return 0;d_set_used(c,c->side,key,1);return !d_bad(c);}
DFN int d_randbelow(DX*c,int n){if(n<=0||c->s->rng[0]<0){d_fail(c,1);return 0;}uint64_t seed=(uint64_t)c->s->rng[0],value=0;uint32_t words[2]={(uint32_t)seed,(uint32_t)(seed>>32)};bmt_state_t state;bmt_seed_words(&state,words,words[1]?2:1);int bits=bmt_bit_length_u64(n),ok=0;for(int j=0;j<4096;j++){bmt_getbits64(&state,bits,&value);if(value<(uint64_t)n){ok=1;break;}}if(!ok){d_fail(c,2);return 0;}int result=(int)value;bmt_getbits64(&state,63,&value);c->s->rng[0]=(long long)value;return result;}
'''
    # Static catalog values are selected by template, not by an instruction VM.
    text+='DFN int d_make_card(DX*c,int side,const char*name,int owner){int id=d_card_id(name);if(!id||!d_valid_side(c,side)){d_fail(c,1);return -1;}DS*f=c->s;int ref=-1;for(int j=0;j<K;j++)if(!f->card_valid[j]){ref=j;break;}if(ref<0||f->next_entity[0]>=2147483647||f->next_instance[0]>=2147483647){d_fail(c,2);return -1;}'
    for field,(dtype,shape)in specs.items():
        if field.startswith('card_'):
            width=prod(shape[1:]);text+='for(int j=0;j<'+str(width)+';j++)f->'+field+'[ref*'+str(width)+'+j]=0;'
    text+='f->card_valid[ref]=1;f->card_template[ref]=id;f->card_side[ref]=side;f->card_entity_slot[ref]=++f->next_entity[0];f->card_instance_slot[ref]=++f->next_instance[0];switch(id){'
    for i,card in cards.items():
        text+='case '+str(i)+':'
        for typ,keys,dim in [('int',schema.CARD_INT_FIELDS,'KI'),('bool',schema.CARD_BOOL_FIELDS,'KB'),('str',schema.CARD_STR_FIELDS,'KS')]:
            for j,key in enumerate(keys):
                if key in card:text+='f->card_'+typ+'_presence[ref*'+dim+'+'+str(j)+']='+('1'if card[key]is None else'3')+';'
        for key,field in [('source','card_source_presence'),('play_options','card_play_options_presence')]:
            if key in card:text+='f->'+field+'[ref]=3;'
        text+='break;'
    text+='}f->card_bool_presence[ref*KB+KB_copy]=2;f->card_bool_value[ref*KB+KB_copy]=0;if(owner>=0){if(!d_valid_char(c,side,owner))return -1;const char*original=d_card_default_str(id,KS_character_id);int err=0;if(!qeq(f,f->character_id_slot[side*C+owner],original,&err)){f->card_str_presence[ref*KS+KS_original_character_id]=2;f->card_str_slot[ref*KS+KS_original_character_id]=d_intern(c,original);f->card_str_presence[ref*KS+KS_character_id]=2;f->card_str_slot[ref*KS+KS_character_id]=f->character_id_slot[side*C+owner];}if(err)d_fail(c,err);}return ref;}\n'
    text+='''DFN void d_remove_zone_card(DX*c,int side,int zone,int card){if(!d_valid_side(c,side)||zone<0||zone>3){d_fail(c,1);return;}DS*f=c->s;int n=f->zone_count[side*4+zone],at=-1;if(n<0||n>Z){d_fail(c,1);return;}for(int j=0;j<n;j++)if(f->zone_card_index[(side*4+zone)*Z+j]==card){at=j;break;}if(at<0){d_fail(c,1);return;}for(int j=at;j<n-1;j++)f->zone_card_index[(side*4+zone)*Z+j]=f->zone_card_index[(side*4+zone)*Z+j+1];f->zone_card_index[(side*4+zone)*Z+n-1]=0;f->zone_count[side*4+zone]--;}
DFN void d_zone_append(DX*c,int side,int zone,int card){if(!d_valid_side(c,side)||zone<0||zone>3||card<0||card>=K||!c->s->card_valid[card]){d_fail(c,1);return;}int n=c->s->zone_count[side*4+zone];if(n<0||n>=Z){d_fail(c,2);return;}c->s->zone_card_index[(side*4+zone)*Z+n]=card;c->s->zone_count[side*4+zone]++;}
DFN void d_discard(DX*c,int side,int card,int bottom){if(card<=-2)return;if(card<0||card>=K){d_fail(c,1);return;}DS*f=c->s;if(f->card_is_view[card]){card=f->card_physical_index[card];if(card<0||card>=K||f->card_is_view[card]){d_fail(c,1);return;}}f->card_jingu_mark_presence[card]=0;f->card_jingu_delta[card]=0;f->card_jingu_source_slot[card]=0;if(d_card_bool(c,card,"copy")){d_zone_append(c,side,3,card);d_event(c);}else if(bottom)d_zone_append(c,side,1,card);else{d_zone_append(c,side,2,card);d_event(c);}}
DFN void d_add_hand(DX*c,int side,int card,int reveal){if(d_finished(c)||d_bad(c))return;if(c->s->zone_count[side*4]>=10){d_discard(c,side,card);return;}d_zone_append(c,side,0,card);if(d_bad(c))return;DX own=*c;own.side=side;own.actor=d_order_slot(c,side,0);own.card=-1;DEvt ev={};ev.card=card;d_fire(&own,"on_hand_added",&ev);if(reveal||d_card_bool(c,card,"copy")||d_card_bool(c,card,"derived"))d_event(c);}
DFN int d_deck_count(DX*c,int side){return d_valid_side(c,side)?c->s->zone_count[side*4+1]:0;}
DFN void d_insert_deck(DX*c,int side,int card,int at){int n=d_deck_count(c,side);if(n>=Z||at<0||at>n){d_fail(c,2);return;}for(int j=n;j>at;j--)c->s->zone_card_index[(side*4+1)*Z+j]=c->s->zone_card_index[(side*4+1)*Z+j-1];c->s->zone_card_index[(side*4+1)*Z+at]=card;c->s->zone_count[side*4+1]++;}
DFN void d_shuffle_zone(DX*c,int side,int zone){if(!d_valid_side(c,side)||zone<0||zone>3){d_fail(c,1);return;}int n=c->s->zone_count[side*4+zone];if(n<0||n>Z||c->s->rng[0]<0){d_fail(c,1);return;}uint64_t seed=(uint64_t)c->s->rng[0],value=0;uint32_t words[2]={(uint32_t)seed,(uint32_t)(seed>>32)};bmt_state_t state;bmt_seed_words(&state,words,words[1]?2:1);for(int j=n-1;j>0;j--){int bits=bmt_bit_length_u64(j+1),ok=0;for(int k=0;k<4096;k++){bmt_getbits64(&state,bits,&value);if(value<(uint64_t)(j+1)){ok=1;break;}}if(!ok){d_fail(c,2);return;}int*a=c->s->zone_card_index+(side*4+zone)*Z;int old=a[j];a[j]=a[(int)value];a[(int)value]=old;}bmt_getbits64(&state,63,&value);c->s->rng[0]=(long long)value;}
'''
    return text
