"""Direct numeric public character queries; no VM or host rule fallback.

Only immutable character queries are implemented here, not card execution,
complete observations, legal actions, H3 or complete games. CPU execution is an
explicit compiler oracle. Unknown conditions and unsupported inputs fail closed.
"""
from dataclasses import dataclass
from math import prod
from pathlib import Path
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np

from .domain_soa_schema import (
    CHARACTER_VOCAB, CHARACTER_INT_FIELDS, CHARACTER_BOOL_FIELDS,
    CHARACTER_STR_FIELDS, PLAYER_INT_FIELDS, FLAG_INT_FIELDS,
    EFFECT_CONDITION_IDS, EFFECT_CONDITION_ARGS, DomainSoALayout,
    array_specs, catalog_fingerprint,
)

supports_full_duel_rules = False
INVALID, UNSUPPORTED, NUMERIC_BOUND = 1, 2, 3
ATTRIBUTES = ('光','暗','灵','魂','相','咒')
OUTPUT_NAMES = ('attack','base_attack','base_max_hp','energy_max','harmony_max',
                'collapse_max','damage_immune','damage_limit','damage_limit_present',
                'atk_buff','atk_debuff','form_attack','next_attack',
                'timed_attack_1','timed_attack_2', *('resistance_'+a for a in ATTRIBUTES), 'card_instant_false', 'card_instant_true')

# This is domain rule code: named field reads, bounded declarative effects and
# static current auras. There are no instruction frames or bytecode dispatches.
_BODY = r'''
QFN int qfinite(double v){return v==v && v<=1.7976931348623157e308 && v>=-1.7976931348623157e308;}
QFN double qabs(double v){return v<0?-v:v;}
QFN int qeq(const QRow* f,int slot,const char* text,int* err){
 if(slot<0||slot>=f->string_count[0]||slot>=SS){*err=1;return 0;}
 int n=f->string_length[slot],p=f->string_offset[slot];
 if(n<0||n>SB||p<0||p>SB||p+n>f->string_bytes_used[0]||p+n>SB){*err=1;return 0;}
 int j=0;for(;text[j];j++){if(j>=n||f->string_bytes[p+j]!=(unsigned char)text[j])return 0;}
 return j==n;
}
QFN int qslots(const QRow* f,int a,int b,int*err){
 if(a<0||b<0)return a==b;
 if(a>=f->string_count[0]||b>=f->string_count[0]||a>=SS||b>=SS){*err=1;return 0;}
 int n=f->string_length[a],m=f->string_length[b],p=f->string_offset[a],q=f->string_offset[b];
 if(n<0||m<0||n>SB||m>SB||p<0||q<0||p>SB||q>SB||p+n>f->string_bytes_used[0]||q+m>f->string_bytes_used[0]||p+n>SB||q+m>SB){*err=1;return 0;}
 if(n!=m)return 0;for(int j=0;j<n;j++)if(f->string_bytes[p+j]!=f->string_bytes[q+j])return 0;return 1;
}
QFN int holder_side(int h){return h/(C+1);}
QFN int holder_char(int h){return h%(C+1)-1;}
QFN int exists(const QRow*f,int h){int s=holder_side(h),c=holder_char(h);return c<0||c<f->player_character_count[s];}
QFN int heid(const QRow*f,int h){int s=holder_side(h),c=holder_char(h);return c<0?f->player_entity_slot[s]:f->character_entity_slot[s*C+c];}
QFN int hhp(const QRow*f,int h){int s=holder_side(h),c=holder_char(h);return c<0?f->player_hp[s]:f->character_hp[s*C+c];}
QFN int tmpl(const QRow*f,int h,int*err){int c=holder_char(h);if(c<0)return 0;int v=f->character_template[holder_side(h)*C+c];if(v<0||v>=NT){*err=1;return 0;}return v;}
QFN int ci(const QRow*f,int h,int k,int*err){
 int c=holder_char(h);if(c<0)return 0;int i=(holder_side(h)*C+c)*NC+k,p=f->character_int_presence[i];
 if(p==2)return f->character_int_value[i];if(p==3)return cdefs[tmpl(f,h,err)*NC+k];if(p<0||p>3)*err=1;return 0;
}
QFN int cb(const QRow*f,int h,int k){int c=holder_char(h);if(c<0)return 0;int i=(holder_side(h)*C+c)*NB+k;return f->character_bool_presence[i]==2&&f->character_bool_value[i];}
QFN int pi(const QRow*f,int s,int k){return f->player_int_presence[s*NP+k]==2?f->player_int_value[s*NP+k]:0;}
QFN int flag(const QRow*f,int h,int k){int i=h*NF+k;return f->flag_int_presence[i]>=2?f->flag_int_value[i]:0;}
QFN int shape_slot(const QRow*f,int h){int c=holder_char(h);if(c<0)return -1;int i=(holder_side(h)*C+c)*NS+CS_shape;return f->character_str_presence[i]==2?f->character_str_slot[i]:-1;}
QFN int shape(const QRow*f,int h,int*err){int s=shape_slot(f,h);if(s<0)return 0;SHAPE_CASES *err=2;return 0;}
QFN int attr(const QRow*f,int h,int*err){
 int c=holder_char(h);if(c<0)return 0;int v=tmpl(f,h,err);int i=(holder_side(h)*C+c)*NS+CS_attribute;
 if(f->character_str_presence[i]==3)return cattr[v];
 if(f->character_str_presence[i]==0||f->character_str_presence[i]==1)return 0;
 if(f->character_str_presence[i]!=2){*err=1;return 0;}int slot=f->character_str_slot[i];ATTR_CASES return 0;
}
QFN int source(const QRow*f,int e,int holder,int*err){
 if(f->effect_source_entity_presence[e]!=2)return holder;
 int eid=f->effect_source_entity_slot[e];if(eid<=0){*err=1;return -1;}
 if(eid==heid(f,holder))return holder;
 for(int h=0;h<H;h++)if(exists(f,h)&&heid(f,h)==eid)return h;return -1;
}
QFN double argnum(const QRow*f,int at,int*err){
 int k=f->effect_condition_arg_kind[at];double v=0;
 if(f->effect_condition_arg_presence[at]!=2){*err=2;return 0;}
 if(k==0||k==2)v=f->effect_condition_arg_int[at];else if(k==1)v=f->effect_condition_arg_float[at];else *err=2;
 if(!qfinite(v))*err=3;return v;
}
QFN double resource(const QRow*f,int h,int slot,int*err){
 if(h<0)return 0;int c=holder_char(h),s=holder_side(h);
 if(qeq(f,slot,"hp",err))return hhp(f,h);
 if(qeq(f,slot,"shield",err))return c<0?f->player_shield[s]:f->character_shield[s*C+c];
 if(qeq(f,slot,"energy",err))return c<0?0:f->character_energy[s*C+c];
 RESOURCE_CASES *err=2;return 0;
}
QFN int conditions(const QRow*f,int e,int holder,int subject,int src,int*err){
 int n=f->effect_conditions_count[e];if(n<0||n>Q){*err=1;return 0;}
 for(int j=0;j<n;j++){
  int at=e*Q+j,id=f->effect_condition_id[at],ab=at*NA;
  if(id<0||id>=NCOND){*err=2;return 0;}
  int yes=0;
  if(id==COND_source_front){
   if(src>=0&&holder_char(src)>=0){int s=holder_side(src),c=f->player_front_slot[s];yes=f->player_front_presence[s]==2&&c>=0&&c<C&&f->character_entity_slot[s*C+c]==heid(f,src);}
  }else if(id==COND_source_alive)yes=src>=0&&hhp(f,src)>0;
  else if(id==COND_source_awakened)yes=src>=0&&cb(f,src,CB_awakened);
  else if(id==COND_source_shape){
   int p=f->effect_condition_arg_presence[ab+ARG_card_id];
   yes=src>=0&&(p<=1?shape_slot(f,src)<0:(p==2&&f->effect_condition_arg_kind[ab+ARG_card_id]==3&&qslots(f,shape_slot(f,src),f->effect_condition_arg_string[ab+ARG_card_id],err)));
  }else if(id==COND_enemy_turn)yes=f->active_side[0]!=holder_side(holder);
  else if(id==COND_resource_at_least){
   int x=ab+ARG_resource;if(src<0)yes=0;
   else if(f->effect_condition_arg_presence[x]!=2||f->effect_condition_arg_kind[x]!=3){*err=2;return 0;}
   else yes=resource(f,src,f->effect_condition_arg_string[x],err)>=argnum(f,ab+ARG_amount,err);
  }else if(id==COND_damage_attribute){
   // Observation queries have no damage attribute. None equals a missing argument.
   yes=f->effect_condition_arg_presence[ab+ARG_attribute]<=1;
  }else if(id==COND_card_owner_other)yes=src>=0&&heid(f,src)!=heid(f,subject);
  else if(id==COND_attribute_is){
   int x=ab+ARG_value,c=holder_char(subject),si=(holder_side(subject)*C+c)*NS+CS_attribute;
   int p=c<0?0:f->character_str_presence[si];
   if(f->effect_condition_arg_presence[x]<=1)yes=p==0||p==1;
   else if(f->effect_condition_arg_presence[x]==2&&f->effect_condition_arg_kind[x]==3){
    int slot=f->effect_condition_arg_string[x];if(p==2)yes=qslots(f,f->character_str_slot[si],slot,err);
    else if(p==3){int a=attr(f,subject,err);ATTRIBUTE_ARGUMENT_CASES}
   }
  }
  if(!yes)return 0;
 }
 return 1;
}
QFN int applies(const QRow*f,int e,int h,int subject,int*err){
 int s=holder_side(h),ss=holder_side(subject),expiry=f->effect_expiry_presence[e];
 if(expiry==2&&f->effect_expiry_turn_presence[e]==2){
  int es=f->effect_expiry_side_presence[e]==2?f->effect_expiry_side[e]:s;
  if(es<0||es>1){*err=1;return 0;}int now=pi(f,es,PI_turn_count),until=f->effect_expiry_turn[e];
  if(until<now||(f->effect_expiry_phase[e]==0&&until==now))return 0;
 }
 int same=heid(f,h)==heid(f,subject),scope=f->effect_scope[e];
 if(scope<0||scope>4){*err=1;return 0;}
 if(!(scope==0?same:scope==1?s==ss:scope==2?s==ss&&!same:scope==3?s!=ss:1))return 0;
 int src=source(f,e,h,err);
 if(f->effect_kind[e]==1&&(src<0||hhp(f,src)<=0))return 0;
 return conditions(f,e,h,subject,src,err);
}
QFN int remaining(const QRow*f,int e,int h,int*err){
 if(f->effect_expiry_remaining_presence[e]==2)return f->effect_expiry_remaining[e];
 if(f->effect_expiry_remaining_presence[e]==1)return -1;
 if(f->effect_expiry_turn_presence[e]==2){int s=f->effect_expiry_side_presence[e]==2?f->effect_expiry_side[e]:holder_side(h);if(s<0||s>1){*err=1;return -1;}long long n=(long long)f->effect_expiry_turn[e]-pi(f,s,PI_turn_count);if(n>2147483647LL){*err=3;return -1;}return n>0?(int)n:0;}return -1;
}
QFN double eval_value(const QRow*f,int e,int*err){int k=f->effect_value_kind[e];double v=k==1?f->effect_value_float[e]:(double)f->effect_value_int[e];if(k<0||k>2||!qfinite(v)){*err=1;return 0;}return v;}
struct Candidate{double value;int layer,op;};
QFN int rank_op(int op){return op==2?0:op==0?1:op==1?2:op;}
QFN void push(Candidate* a,int*n,double v,int layer,int op,int*err){
 if(*n>=MAXQ||op<0||op>4||!qfinite(v)){*err=1;return;}int j=*n,r=rank_op(op);
 while(j>0&&(a[j-1].layer>layer||(a[j-1].layer==layer&&rank_op(a[j-1].op)>r))){a[j]=a[j-1];j--;}
 a[j].value=v;a[j].layer=layer;a[j].op=op;(*n)++;
}
QFN double reduce(Candidate*a,int n,double base,int*present,int*err){
 double v=base;
 for(int i=0;i<n;i++){
  double x=a[i].value;int op=a[i].op;
  if(op==2){v=x;*present=1;}else if(op==0||op==1){if(!*present){*err=2;return 0;}v=op==0?v+x:v*x;}
  else if(op==3){v=(v!=0)||x!=0;*present=1;}else if(x!=0){v=0;*present=1;}
  if(!qfinite(v)||qabs(v)>2147483647.){*err=3;return 0;}
 }
 return v;
}
QFN int legacy_expired(const QRow*f,int h){return f->flag_expiring_until[h]<=pi(f,holder_side(h),PI_turn_count);}
QFN double query(const QRow*f,int subject,int property,double base,int*present,int*err){
 Candidate a[MAXQ];int n=0;
 for(int h=0;h<H;h++)if(exists(f,h)){
  int count=f->holder_effect_count[h];if(count<0||count>E){*err=1;return 0;}
  for(int j=0;j<count;j++){
   int e=h*E+j;if(!f->effect_valid[e]){*err=1;return 0;}
   int match=0,slot=f->effect_property_slot[e];
   if(property==0)match=qeq(f,slot,"attack.panel",err);else if(property==1)match=qeq(f,slot,"attack.override",err);
   else if(property==2)match=qeq(f,slot,"rule.damage_immune",err);else if(property==3)match=qeq(f,slot,"damage.limit",err);else if(property==4)match=qeq(f,slot,"card.instant",err);else if(property==5)match=qeq(f,slot,"card.cost",err);else if(property==6)match=qeq(f,slot,"rule.normal_attack_instant",err);else match=qeq(f,slot,"marker.lingke_taunt",err);
   if(match&&applies(f,e,h,subject,err))push(a,&n,eval_value(f,e,err),f->effect_layer[e],f->effect_op[e],err);
  }
  if(heid(f,h)==heid(f,subject)){
   if(property==0){
    int temporary=f->flag_expiring_presence[h]==2?f->flag_expiring_amount[h]:0;
    double amount=(double)flag(f,h,FI_atk_buff)-temporary;if(amount)push(a,&n,amount,0,0,err);
    if(temporary&&!legacy_expired(f,h))push(a,&n,temporary,0,0,err);
    if(flag(f,h,FI_atk_debuff))push(a,&n,-(double)flag(f,h,FI_atk_debuff),0,0,err);
    if(flag(f,h,FI_form_attack))push(a,&n,flag(f,h,FI_form_attack),0,0,err);
    int tn=f->flag_timed_count[h];if(tn<0||tn>T){*err=1;return 0;}
    for(int j=0;j<tn;j++)if(f->flag_timed_left[h*T+j]>0&&f->flag_timed_amount[h*T+j])push(a,&n,f->flag_timed_amount[h*T+j],0,0,err);
   }else if(property==2){int until=flag(f,h,FI_damage_immunity_until);if(until&&until>pi(f,holder_side(h),PI_turn_count))push(a,&n,1,0,3,err);}
  }
 }
 // Current public aura providers relevant to these four immutable properties.
 for(int s=0;s<2;s++)for(int c=0;c<f->player_character_count[s];c++){
  int h=s*(C+1)+1+c;if(hhp(f,h)<=0)continue;int id=tmpl(f,h,err),sh=shape(f,h,err);
  if(property==0&&id==ID_adler&&sh==SH_D07&&s==holder_side(subject)&&heid(f,h)!=heid(f,subject)&&attr(f,subject,err)==6)push(a,&n,1,0,0,err);
  if(property==1&&id==ID_bohe&&s==holder_side(subject)&&heid(f,h)==heid(f,subject)&&f->active_side[0]!=s&&card_enemy_present[sh])push(a,&n,card_enemy_value[sh],0,2,err);
 }
 return reduce(a,n,base,present,err);
}
QFN void compatibility(const QRow*f,int h,double*out,int*err){
 int count=f->holder_effect_count[h];if(count<0||count>E){*err=1;return;}
 for(int j=0;j<count;j++){
  int e=h*E+j;if(!f->effect_valid[e]){*err=1;return;}
  if(f->effect_visibility[e]!=0)continue;double v=eval_value(f,e,err);
  if(qeq(f,f->effect_property_slot[e],"damage.resistance",err)){
   int n=f->effect_conditions_count[e];if(n<0||n>Q){*err=1;return;}
   for(int k=0;k<n;k++){int at=e*Q+k;if(f->effect_condition_id[at]!=COND_damage_attribute)continue;
    int x=at*NA+ARG_attribute;if(f->effect_condition_arg_presence[x]==2&&f->effect_condition_arg_kind[x]==3){int slot=f->effect_condition_arg_string[x];RESISTANCE_CASES}break;
   }
  }
  if(f->effect_scope[e]!=0||f->effect_op[e]!=0)continue;
  if(qeq(f,f->effect_property_slot[e],"attack.hit",err)&&f->effect_consume_on_presence[e]==2&&qeq(f,f->effect_consume_on_slot[e],"attack",err))out[OUT_next_attack]+=v;
  else if(qeq(f,f->effect_property_slot[e],"attack.panel",err)){
   if(qeq(f,f->effect_definition_slot[e],"legacy.timed_attack",err)||qeq(f,f->effect_definition_slot[e],"yi.attack_debuff",err)){int r=remaining(f,e,h,err);if(r==1)out[OUT_timed_attack_1]+=v;else if(r==2)out[OUT_timed_attack_2]+=v;}
   else if(qeq(f,f->effect_definition_slot[e],"legacy.form_attack",err))out[OUT_form_attack]+=v;
   else if(v<0)out[OUT_atk_debuff]-=v;else out[OUT_atk_buff]+=v;
  }
 }
 int temporary=f->flag_expiring_presence[h]==2?f->flag_expiring_amount[h]:0;
 double normal=(double)flag(f,h,FI_atk_buff)-temporary;
 if(normal<0)out[OUT_atk_debuff]-=normal;else out[OUT_atk_buff]+=normal;
 if(temporary<0)out[OUT_atk_debuff]-=temporary;else out[OUT_atk_buff]+=temporary;
 normal=-(double)flag(f,h,FI_atk_debuff);if(normal<0)out[OUT_atk_debuff]-=normal;else out[OUT_atk_buff]+=normal;
 out[OUT_form_attack]+=flag(f,h,FI_form_attack);out[OUT_next_attack]+=flag(f,h,FI_next_attack);
 int tn=f->flag_timed_count[h];if(tn<0||tn>T){*err=1;return;}
 for(int j=0;j<tn;j++){int left=f->flag_timed_left[h*T+j];double amount=f->flag_timed_amount[h*T+j];if(left==1)out[OUT_timed_attack_1]+=amount;else if(left==2)out[OUT_timed_attack_2]+=amount;}
}
QFN void character_query(const QRow*f,int side,int c,double*out,int*err){
 int h=side*(C+1)+1+c,ch=side*C+c,id=tmpl(f,h,err),sh=shape(f,h,err);
 if(flag(f,h,FI_mimic)){*err=2;return;}
 long long printed=f->character_base_attack_presence[ch]==2?f->character_base_attack[ch]:(id?cattack[id]:ci(f,h,CI_base_attack,err));
 long long printed_hp=f->character_base_max_hp_presence[ch]==2?f->character_base_max_hp[ch]:(id?chp[id]:(f->character_int_presence[ch*NC+CI_base_max_hp]>=2?ci(f,h,CI_base_max_hp,err):ci(f,h,CI_max_hp,err)));
 long long growth=f->character_growth_attack_presence[ch]==2?f->character_growth_attack[ch]:0;
 long long growth_hp=f->character_growth_max_hp_presence[ch]==2?f->character_growth_max_hp[ch]:0;
 if(f->character_int_presence[ch*NC+CI_base_attack]>=2)growth=ci(f,h,CI_base_attack,err)-printed;
 if(f->character_int_presence[ch*NC+CI_base_max_hp]>=2)growth_hp=ci(f,h,CI_base_max_hp,err)-printed_hp;
 out[OUT_base_attack]=printed+growth;out[OUT_base_max_hp]=printed_hp+growth_hp;
 double base;
 if(card_form[sh]&&card_attack[sh]>0&&card_hp[sh]>0)base=card_attack[sh]+(growth>0?growth:0);
 else base=printed+growth+ci(f,h,CI_growth,err)+ci(f,h,CI_gaze,err)+card_bonus[sh];
 int present=1;base=query(f,h,0,base,&present,err);base=query(f,h,1,base,&present,err);
 int panel_id=id;
 if(f->flag_mimic_presence[h]==2&&f->flag_mimic_id_presence[h]==2&&f->flag_mimic_template[h]>0){
  panel_id=f->flag_mimic_template[h];if(panel_id>=NT){*err=1;return;}
 }
 if(panel_id==ID_adler&&f->phase[0]!=3)base=f->character_shield[ch];
 else if(f->player_front_presence[side]==2&&f->player_front_slot[side]==c&&f->zone_status_presence[side*5]==2){
  int mask=f->zone_status_i32[(side*5)*8+7];
  int at=0,slow=0,slow_turn=f->turn[0];
  // delay keys: left,end,by,tick_side,slow,slow_turn,expires_side,infinite
  DELAY_READ
  if(slow_turn==f->turn[0])base-=slow;
 }
 if(!qfinite(base)||qabs(base)>2147483647.){*err=3;return;}
 int attack=(int)base;out[OUT_attack]=attack>0?attack:0;
 int attribute=attr(f,h,err);long long energy=id?cenergy[id]:((attribute==2||attribute==4||attribute==6)?6:5);
 if((!id||cruntime_energy[id])&&f->character_int_presence[ch*NC+CI_energy_max_override]==2)energy=ci(f,h,CI_energy_max_override,err);
 int cut=f->escalation_enabled[0]&&f->turn[0]>=13;out[OUT_energy_max]=energy-cut;out[OUT_harmony_max]=2-cut;
 long long limit=flag(f,h,FI_collapse_max);if(!limit)limit=5;
 int rn=f->character_reductions_count[ch];if(rn<0||rn>2*C){*err=1;return;}
 for(int j=0;j<rn;j++)limit-=f->character_reductions_value[ch*(2*C)+j];if(limit>2147483647LL){*err=3;return;}out[OUT_collapse_max]=limit>1?limit:1;
 if(hhp(f,h)>0){present=1;out[OUT_damage_immune]=query(f,h,2,0,&present,err)!=0;present=0;out[OUT_damage_limit]=query(f,h,3,0,&present,err);out[OUT_damage_limit_present]=present;if(present){int n=(int)out[OUT_damage_limit];out[OUT_damage_limit]=n>0?n:0;}}
 present=1;out[OUT_card_instant_false]=query(f,h,4,0,&present,err)!=0;
 present=1;out[OUT_card_instant_true]=query(f,h,4,1,&present,err)!=0;
 compatibility(f,h,out,err);
}
'''


def _source(layout, *, separate_query_frame=False):
    from app.modules.card_game.content.duel_v2 import CHARACTERS, CARDS
    from app.modules.card_game.content.duel_v2.registry import shape_attack_bonus
    from app.modules.card_game.engine.duel_v2.state import energy_max
    from .domain_soa_schema import CARD_VOCAB
    specs = array_specs(layout)
    body = _BODY
    card_names = [None] * (max(CARD_VOCAB.values()) + 1)
    for name, index in CARD_VOCAB.items():
        card_names[index] = name
    def cstr(text):
        return '"'+''.join('\\x%02x'%b for b in text.encode('utf-8'))+'"'
    replacements = {
        'SHAPE_CASES': ''.join('if(qeq(f,s,"%s",err))return %d;' % (name,i) for i,name in enumerate(card_names) if name),
        'ATTR_CASES': ''.join('if(qeq(f,slot,%s,err))return %d;' % (cstr(a),i+1) for i,a in enumerate(ATTRIBUTES)),
        'ATTRIBUTE_ARGUMENT_CASES': ''.join('if(a==%d&&qeq(f,slot,%s,err))yes=1;' % (i+1,cstr(a)) for i,a in enumerate(ATTRIBUTES)),
        'RESISTANCE_CASES': ''.join('if(qeq(f,slot,%s,err))out[%d]+=v;' % (cstr(a),15+i) for i,a in enumerate(ATTRIBUTES)),
    }
    resource = []
    for key in CHARACTER_INT_FIELDS:
        resource.append('if(c>=0&&qeq(f,slot,"%s",err))return ci(f,h,CI_%s,err);' % (key,key))
    for key in PLAYER_INT_FIELDS:
        resource.append('if(c<0&&qeq(f,slot,"%s",err))return pi(f,s,PI_%s);' % (key,key))
    replacements['RESOURCE_CASES'] = ''.join(resource)
    # Match the exact current delay status field order used by the domain codec.
    from .domain_soa_codec import _encode_front_debuff
    # The codec import is compilation-time host metadata, never a device fallback.
    import inspect
    text = inspect.getsource(_encode_front_debuff)
    match = re.search(r'delay_keys\s*=\s*\(([^\n]+)\)', text)
    keys = tuple(re.findall(r'["\']([a-z_]+)["\']', match.group(1))) if match else ()
    if not keys or 'slow' not in keys:
        raise RuntimeError('Explicit delay field contract required')
    delay = []
    for bit, key in enumerate(keys):
        if key in ('left','slow','slow_turn','stacks','applied_turn','renewals','ends_left'):
            target = ('slow=' if key == 'slow' else 'slow_turn=' if key == 'slow_turn' else '')
            delay.append('if(mask&(1<<%d)){%sf->zone_status_i32[(side*5)*8+at];at++;}' % (bit,target))
    replacements['DELAY_READ'] = ''.join(delay)
    for key,value in replacements.items():
        body = body.replace(key,value)
    names = sorted(set(re.findall(r'f->([a-z_0-9]+)',body)) | {'row_error','catalog_digest','row_valid','character_valid'})
    missing = set(names)-set(specs)
    if missing:
        raise RuntimeError('Query field contract missing: '+','.join(sorted(missing)))
    ctypes_names = {'bool':'unsigned char','uint8':'unsigned char','int8':'signed char','int16':'short','int32':'int','int64':'long long','float64':'double'}
    meta = []
    def define(key,value):meta.append('#define %s %d\n' % (key,value))
    C,E,T,Q=layout.max_characters,layout.max_effects,layout.max_timed_attack,layout.max_conditions
    for key,value in dict(C=C,H=2*(C+1),E=E,T=T,Q=Q,NC=len(CHARACTER_INT_FIELDS),NB=len(CHARACTER_BOOL_FIELDS),NS=len(CHARACTER_STR_FIELDS),NP=len(PLAYER_INT_FIELDS),NF=len(FLAG_INT_FIELDS),NA=len(EFFECT_CONDITION_ARGS),NCOND=len(EFFECT_CONDITION_IDS),NT=max(CHARACTER_VOCAB.values())+1,SS=layout.max_string_slots,SB=layout.max_string_bytes,NO=len(OUTPUT_NAMES),MAXQ=2*(C+1)*(E+T+8)+2*C).items():define(key,value)
    for prefix,items in (('CI',CHARACTER_INT_FIELDS),('CB',CHARACTER_BOOL_FIELDS),('CS',CHARACTER_STR_FIELDS),('PI',PLAYER_INT_FIELDS),('FI',FLAG_INT_FIELDS),('COND',EFFECT_CONDITION_IDS),('ARG',EFFECT_CONDITION_ARGS),('OUT',OUTPUT_NAMES)):
        for i,name in enumerate(items):
            if name.isascii():define(prefix+'_'+name,i)
    for name,i in CHARACTER_VOCAB.items():define('ID_'+name,i)
    define('SH_D07', CARD_VOCAB['D07'])
    tables = []
    def table(name,values):tables.append('QTAB int %s[]={%s};\n' % (name,','.join(map(str,values))))
    n=max(CHARACTER_VOCAB.values())+1
    byid={i:CHARACTERS[cid]for cid,i in CHARACTER_VOCAB.items()}
    table('cdefs',[int(byid.get(i,{}).get(k) or 0) if type(byid.get(i,{}).get(k)) is int else 0 for i in range(n) for k in CHARACTER_INT_FIELDS])
    table('cattack',[int(byid.get(i,{}).get('attack')or 0)for i in range(n)])
    table('chp',[int(byid.get(i,{}).get('max_hp')or 0)for i in range(n)])
    table('cattr',[ATTRIBUTES.index(byid[i]['attribute'])+1 if i in byid else 0 for i in range(n)])
    table('cenergy',[energy_max(dict(byid[i],id=next(cid for cid,x in CHARACTER_VOCAB.items()if x==i))) if i in byid else 0 for i in range(n)])
    table('cruntime_energy',[int('energy_max_override'not in byid.get(i,{}))for i in range(n)])
    for field in ('attack','hp'):
        table('card_'+field,[int((CARDS.get(name)or{}).get(field)or 0)for name in card_names])
    table('card_bonus',[shape_attack_bonus(name)if name else 0 for name in card_names])
    table('card_form',[int((CARDS.get(name)or{}).get('type')=='form')for name in card_names])
    table('card_enemy_present',[int('enemy_turn_attack'in(CARDS.get(name)or{}))for name in card_names])
    table('card_enemy_value',[int((CARDS.get(name)or{}).get('enemy_turn_attack')or 0)for name in card_names])
    declarations=''.join('const %s* %s;\n' % (ctypes_names[str(np.dtype(specs[name][0]))],name)for name in names)
    params=','.join('const %s* p_%s'%(ctypes_names[str(np.dtype(specs[name][0]))],name)for name in names)
    initializers=''.join('f.%s=p_%s+row*%d;'%(name,name,prod(specs[name][1]))for name in names)
    header='''\n#if defined(__CUDACC__) || defined(__CUDACC_RTC__)\n#define QFN __device__ __forceinline__\n#define QTAB __device__ __constant__\n#else\n#define QFN inline\n#define QTAB static const\n#endif\n'''
    common='''QFN void run_query('''+params+''',const unsigned char*active,double*values,int*errors,int index){
 int row=index/(2*(C+1)),ph=index%(2*(C+1)),side=ph/(C+1),c=ph%(C+1)-1,ch=side*C+c;double*out=values+index*NO;
 for(int k=0;k<NO;k++)out[k]=0;int err=0;errors[index]=0;if(!active[row])return;QRow f;'''+initializers+'''
 if(f.row_error[0])return;
 if(!f.row_valid[0]){errors[index]=1;return;}
 for(int s=0;s<2;s++)if(f.player_character_count[s]<1||f.player_character_count[s]>C){errors[index]=1;return;}
 if(f.string_count[0]<0||f.string_count[0]>SS||f.string_bytes_used[0]<0||f.string_bytes_used[0]>SB){errors[index]=1;return;}
 if(c<0)compatibility(&f,side*(C+1),out,&err);
 else {if(c>=f.player_character_count[side]){if(f.character_valid[ch])errors[index]=1;return;}if(!f.character_valid[ch]){errors[index]=1;return;}
 character_query(&f,side,c,out,&err);}for(int k=0;k<NO;k++)if(!qfinite(out[k])||qabs(out[k])>2147483647.)err=3;errors[index]=err;
}
'''
    args=','.join('p_'+name for name in names)
    tail='''\n#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void domain_character_queries('''+params+''',const unsigned char*active,double*values,int*errors,int rows){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*2*(C+1))run_query('''+args+''',active,values,errors,i);}
#else
extern "C" void domain_character_queries('''+params+''',const unsigned char*active,double*values,int*errors,int rows){for(int i=0;i<rows*2*(C+1);i++)run_query('''+args+''',active,values,errors,i);}
#endif
'''
    if separate_query_frame:
        # A character asks several properties serially. Force-inlining query
        # duplicates its MAXQ array in one frame on CUDA; one callee reuses it.
        header+='\n#if defined(__CUDACC__) || defined(__CUDACC_RTC__)\n#define QQUERY __device__ __noinline__\n#else\n#define QQUERY inline\n#endif\n'
        body=body.replace('QFN double query(', 'QQUERY double query(')
    return header+''.join(meta)+''.join(tables)+'struct QRow{'+declarations+'};\n'+body+common+tail, tuple(names)


@dataclass(frozen=True)
class PreparedCharacterQueries:
    owner: object
    sources: object
    fields: object
    active: object
    active_input: object
    values: object
    character_errors: object
    errors: object


class DomainCharacterQueries:
    supports_full_duel_rules = False

    def __init__(self, layout=None, *, device='cuda', allow_cpu_oracle=False):
        import torch
        self.t=torch;self.layout=layout or DomainSoALayout();self.device=torch.device(device)
        if not isinstance(self.layout,DomainSoALayout) or not (4<=self.layout.max_characters<=16 and self.layout.max_effects<=32 and self.layout.max_timed_attack<=16 and self.layout.max_conditions<=8):
            raise ValueError('Bounded current character query layout required')
        self.module=None;self.library=None;self.temporary=None;self.closed=False
        source,self.names=_source(self.layout,separate_query_frame=True);self.specs=array_specs(self.layout)
        if self.device.type=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; no query CPU fallback')
            if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
            from .cuda_runtime import CudaModule
            self.module=CudaModule(source,{'domain_character_queries':['uint8*']*len(self.names)+['uint8*','float64*','int32*','int32']},device=str(self.device))
        elif self.device.type=='cpu' and allow_cpu_oracle is True:
            compiler=shutil.which('clang++')or shutil.which('g++')
            if compiler is None:raise RuntimeError('Explicit compiler oracle unavailable')
            self.temporary=tempfile.TemporaryDirectory(prefix='nte-character-query-');root=Path(self.temporary.name)
            path=root/'query.cpp';path.write_text(source);library=root/('query.dylib'if sys.platform=='darwin'else'query.so')
            result=subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',str(path),'-o',str(library)],capture_output=True,text=True,timeout=60)
            if result.returncode:self.close();raise RuntimeError(result.stderr)
            self.library=ctypes.CDLL(str(library));self.kernel=self.library.domain_character_queries
            self.kernel.argtypes=[ctypes.c_void_p]*(len(self.names)+3)+[ctypes.c_int32];self.kernel.restype=None
        else:raise ValueError('CUDA or explicit CPU compiler oracle required')
        self.digest=torch.tensor(list(catalog_fingerprint()),dtype=torch.uint8,device=self.device)

    def prepare(self, fields, active, *, _borrow_fields=False):
        t=self.t
        if type(_borrow_fields)is not bool:raise TypeError('Static internal borrow flag required')
        if self.closed:raise RuntimeError('Query backend closed')
        if self.device.type=='cuda' and t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare stable query buffers outside capture')
        cols=fields if isinstance(fields,dict)else getattr(fields,'arrays',None)
        if cols is None:raise TypeError('Named resident SoA columns required')
        if not isinstance(active,t.Tensor)or active.device!=self.device or active.dtype!=t.bool or active.ndim!=1:raise ValueError('Resident bool active mask required')
        rows=active.shape[0]
        if not 1<=rows<=self.layout.game_capacity:raise ValueError('Query row bound exceeded')
        dtype={'bool':t.bool,'uint8':t.uint8,'int8':t.int8,'int16':t.int16,'int32':t.int32,'int64':t.int64,'float64':t.float64}
        sources={}
        for name in self.names:
            value=cols.get(name);kind,tail=self.specs[name]
            if not isinstance(value,t.Tensor)or value.device!=self.device or value.dtype!=dtype[str(np.dtype(kind))]or tuple(value.shape)!=(rows,)+tuple(tail):raise ValueError('Resident typed query field required: '+name)
            sources[name]=value
        # The enclosing observation owns a stable private snapshot. Its query
        # kernels are read-only and finish before that snapshot is reused.
        if _borrow_fields and any(not value.is_contiguous()for value in sources.values()):raise ValueError('Borrowed query columns must be contiguous')
        packed=dict(sources)if _borrow_fields else {name:value.clone(memory_format=t.contiguous_format)for name,value in sources.items()}
        C=self.layout.max_characters
        return PreparedCharacterQueries(self,sources,packed,active,active.clone(memory_format=t.contiguous_format),
            t.empty((rows,2,C+1,len(OUTPUT_NAMES)),dtype=t.float64,device=self.device),
            t.empty((rows,2,C+1),dtype=t.int32,device=self.device),t.empty(rows,dtype=t.int32,device=self.device))

    def launch(self, batch):
        t=self.t
        if self.closed:raise RuntimeError('Query backend closed')
        if not isinstance(batch,PreparedCharacterQueries)or batch.owner is not self:raise ValueError('Foreign query buffers')
        for name in self.names:
            if batch.fields[name]is not batch.sources[name]:batch.fields[name].copy_(batch.sources[name])
        batch.active_input.copy_(batch.active)
        args=[batch.fields[name].view(t.uint8)for name in self.names]+[batch.active_input.view(t.uint8),batch.values,batch.character_errors]
        rows=int(batch.active.shape[0])
        if self.module is not None:
            self.module.launch('domain_character_queries',args+[rows],grid=((rows*2*(self.layout.max_characters+1)+63)//64,),block=(64,),capture_safe=True)
        else:self.kernel(*[ctypes.c_void_p(x.data_ptr())for x in args],ctypes.c_int32(rows))
        inherited=batch.fields['row_error'];error=batch.character_errors.amax((1,2))
        bad=(batch.fields['catalog_digest']!=self.digest[None,:]).any(1)&batch.active_input
        error=t.where(bad,INVALID,error)
        batch.errors.copy_(t.where(inherited!=0,inherited,error))
        batch.values.masked_fill_(((batch.errors!=0)|~batch.active_input)[:,None,None,None],0)
        return {'values':batch.values[:,:,1:,:],'player_resistances':batch.values[:,:,0,15:21],
                'errors':batch.errors,'names':OUTPUT_NAMES,'attributes':ATTRIBUTES}

    def query(self, fields, active):return self.launch(self.prepare(fields,active))

    def close(self):
        if self.closed:return
        if self.module is not None:self.module.close();self.module=None
        if self.library is not None:
            import _ctypes
            (_ctypes.FreeLibrary if os.name=='nt'else _ctypes.dlclose)(self.library._handle);self.library=None
        if self.temporary is not None:self.temporary.cleanup();self.temporary=None
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
