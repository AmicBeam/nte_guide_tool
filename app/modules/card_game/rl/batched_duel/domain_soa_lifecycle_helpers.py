"""Direct typed choice, ultimate, deferred and operation snapshot helpers."""
from math import prod
import numpy as np
from . import domain_soa_schema as schema
from .domain_soa_execution_base import _TYPES,_literal
supports_full_duel_rules=False


def operation_snapshot_source(layout):
    fields={name:spec for name,spec in schema.array_specs(layout).items()if name.startswith('op_')or name=='operation_present'}
    text='struct DOpSnap {DX context;'
    for name,(dtype,shape)in fields.items():text+=f'{_TYPES[np.dtype(dtype).name]} {name}[{prod(shape)}];'
    text+='};\nDFN void life_save_op(DX*c,DOpSnap*out){out->context=*c;'
    for name,(dtype,shape)in fields.items():text+=f'for(int j=0;j<{prod(shape)};j++)out->{name}[j]=c->s->{name}[j];'
    text+='}\nDFN void life_restore_op(DX*c,const DOpSnap*in){DS*f=c->s;*c=in->context;'
    for name,(dtype,shape)in fields.items():text+=f'for(int j=0;j<{prod(shape)};j++)f->{name}[j]=in->{name}[j];'
    text+='}\n'
    return text


def emit_source(layout=None):
    layout=layout or schema.DomainSoALayout()
    from app.modules.card_game.content.duel_v2 import CHARACTERS
    from app.modules.card_game.content.duel_v2.registry import KITS,_ensure_kits
    _ensure_kits()
    if any(getattr(KITS[cid],'turn_draw_inspect',None) for cid in schema.CHARACTER_VOCAB):
        raise RuntimeError('Unported public turn-draw inspection provider')
    text=f'#define NCHI {len(schema.CHOICE_INT_KEYS)}\n#define NCHB {len(schema.CHOICE_BOOL_KEYS)}\n#define NCHS {len(schema.CHOICE_STR_KEYS)}\n'
    text+='DFN const char*d_character_faction(int id){switch(id){'+''.join(f'case {i}:return {_literal(CHARACTERS[cid].get("faction", ""))};' for cid,i in schema.CHARACTER_VOCAB.items())+'default:return "";}}\n'
    text+='DFN int d_ultimate_sortie(int id){switch(id){'+''.join(f'case {i}:return {int(bool(getattr(KITS[cid],"ultimate_sortie",False)))};' for cid,i in schema.CHARACTER_VOCAB.items())+'default:return 0;}}\n'
    choice=_CHOICES.replace('CHOICE_KIND_CASES',''.join(f'if(d_streq(kind,"{name}"))index={i};' for i,name in enumerate(schema.CHOICE_KINDS)))
    choice=choice.replace('CHOICE_INT_CASES',''.join(f'if(d_streq(key,"{name}")){{c->s->choice_int_presence[{i}]=2;c->s->choice_int_value[{i}]=value;return;}}' for i,name in enumerate(schema.CHOICE_INT_KEYS)))
    choice=choice.replace('CHOICE_BOOL_CASES',''.join(f'if(d_streq(key,"{name}")){{c->s->choice_bool_presence[{i}]=2;c->s->choice_bool_value[{i}]=value!=0;return;}}' for i,name in enumerate(schema.CHOICE_BOOL_KEYS)))
    prompts=dict(PROMPT_DISCARD='选择 1 张手牌弃置',PROMPT_ENEMY='选择对手的 1 张手牌，公开并置于其牌库底',PROMPT_REDRAW='选择至多 3 张手牌调度',PROMPT_PLACE='将 1 张手牌置于牌库顶或底',PROMPT_INSPECT='检视牌库顶：选择 1 张加入手牌，其余置于牌库底',PROMPT_KIT='选择 1 张加入手牌，其余洗回牌库')
    for name,value in prompts.items():choice=choice.replace(name,_literal(value))
    text+=_BASIC+choice
    text+='DFN void d_divination_prompt(char*out,int count,int min_bottom){const char*prefix='+_literal('检视牌库顶 ')+';int n=0;while(prefix[n]){out[n]=prefix[n];n++;}d_uint_text(out+n,count);while(out[n])n++;const char*suffix=min_bottom?'+_literal(' 张牌：至少置底 1 张，其余自由排序。')+':'+_literal(' 张牌：可将任意数量置底，其余自由排序。')+';for(int j=0;;j++){out[n+j]=suffix[j];if(!suffix[j])break;}}\n'

    text+=_summon_source(layout)+_mark_queue_source(layout)
    return text


_BASIC=r'''
DFN int d_shape_present(DX*c,int side,int actor){int at=(side*C+actor)*NS+CS_shape;return c->s->character_str_presence[at]==2&&c->s->string_length[c->s->character_str_slot[at]]>0;}
DFN void d_clear_next_damage_resistance(DX*c){(void)c;/* Legacy map is rejected at strict ingress; live resistance is stored effects. */}
DFN void d_save_turn_start(DX*c,int side){(void)c;(void)side;/* Current public17 has no temporal snapshot-enabled kit; ingress rejects that state. */}
DFN int d_extra_genesis_actor(DX*c,int side){int at=side*PS;if(c->s->player_str_presence[at]!=2)return -1;int slot=c->s->player_str_slot[at];for(int j=0;j<c->s->player_character_count[side];j++)if(qslots(c->s,slot,c->s->character_id_slot[side*C+j],c->s->row_error))return j;return -1;}
DFN int d_response_faction_matches(DX*c,int card,int side,int actor){if(!d_valid_char(c,side,actor))return 0;int at=(side*C+actor)*NS+CS_faction,p=c->s->character_str_presence[at];if(p==3){int id=d_template(c,side,actor);const char*name=d_character_faction(id);return d_card_str_is(c,card,"response_faction",name);}if(p==2){int id=d_card_template(c,card),k=KS_response_faction,cp=c->s->card_str_presence[card*KS+k];if(cp==2)return qslots(c->s,c->s->character_str_slot[card*KS+k],c->s->character_str_slot[at],c->s->row_error);if(cp==3)return d_eq(c,c->s->character_str_slot[at],d_card_default_str(id,k));}return 0;}
DFN void d_end_ultimate(DX*c,int side,int actor){if(!d_valid_char(c,side,actor))return;if(!d_cb(c,side,actor,"awakened")&&!d_ci(c,side,actor,"ultimate_turns"))return;d_set_cb(c,side,actor,"awakened",0);d_set_ci(c,side,actor,"ultimate_turns",0);DX own=*c;own.side=side;own.actor=actor;own.card=-1;own.effect_source_slot=-1;d_consume_holder(&own,d_holder(c,side,actor),"ultimate_end",1);d_fire(&own,"on_ultimate_end",nullptr,1,1);if(!d_finished(c))d_event(c);}
DFN void d_end_ultimate(DX*c,int actor){d_end_ultimate(c,c->side,actor);}
DFN void d_start_ultimate(DX*c,int side,int actor){DX own=*c;own.side=side;own.actor=actor;own.card=-1;int id=d_template(c,side,actor);d_set_ci(c,side,actor,"energy",0);d_set_ci(c,side,actor,"ultimate_used_turn",c->s->turn[0]);int instant=d_ultimate_instant(id);d_set_cb(c,side,actor,"awakened",!instant);int turns=instant?0:d_decide(&own,"ultimate_duration",d_ultimate_turns(id),nullptr,1);d_set_ci(c,side,actor,"ultimate_turns",turns);if(!instant)d_schedule_ultimate(&own,side,actor,turns,d_ultimate_phase(id));d_event(c);d_fire(&own,"on_ultimate",nullptr,1);DEvt ev={};ev.actor=actor;d_fire(&own,"on_ally_ultimate",&ev);if(d_ultimate_sortie(id)&&d_alive(c,side,actor)){d_open_operation(&own);d_sortie(&own);if(c->s->phase[0]!=2)d_finish_operation(&own);}}
DFN void d_revive(DX*c,int side,int actor){if(!d_valid_char(c,side,actor))return;if(d_ci(c,side,actor,"hp")>0&&!d_ci(c,side,actor,"down_turns"))return;d_clear_harmony_source(c,side,actor);d_set_ci(c,side,actor,"down_turns",0);int hp=d_max_hp(c,side,actor);d_set_ci(c,side,actor,"hp",hp>0?hp:1);d_event(c);}
DFN void d_defer(DX*c,const char*kind,int actor,int amount){DS*f=c->s;int n=f->deferred_count[0];if(n<0||n>=DF||!d_valid_char(c,c->side,actor)){d_fail(c,2);return;}f->deferred_presence[0]=2;f->deferred_side[n]=c->side;f->deferred_actor_slot[n]=actor;f->deferred_entity_slot[n]=d_entity(c,DR{c->side,actor});f->deferred_kind_slot[n]=d_intern(c,kind);f->deferred_card_presence[n]=d_streq(kind,"heiyu_release_after_attack")?2:0;f->deferred_card_index[n]=amount;f->deferred_attribute_presence[n]=0;f->deferred_support_presence[n]=0;f->deferred_count[0]++;}
DFN int d_turn_draw_inspect(DX*c,int*actor,int*count){(void)c;(void)actor;(void)count;return 0;}
DFN void d_scale_battle_frame(DX*c,int*attack,int*shield){if(d_template(c,c->side,c->actor)==ID_zhenhong&&d_card_is(c,c->card,"R06")){int n=d_ci(c,c->side,c->actor,"surplus_passive_triggers");*attack+=n;*shield+=n;}int nanali=d_hero(c,c->side,"nanali");if(nanali>=0&&d_alive(c,c->side,nanali)&&d_cstr_is(c,c->side,nanali,"shape","N07")&&d_card_str_is(c,c->card,"type","battle")&&d_card_owner(c,c->card)==nanali){int n=d_pi(c,c->side,"nanali_family_played");*attack+=n;*shield+=n;}}
'''

_CHOICES=r'''
DFN void d_reset_choice(DX*c,const char*kind,const char*prompt){DS*f=c->s;f->choice_serial++;int index=-1;CHOICE_KIND_CASES if(index<0){d_fail(c,4);return;}f->pending_choice_present[0]=1;f->phase[0]=2;f->choice_kind[0]=index;f->choice_side[0]=c->side;f->choice_prompt_slot[0]=d_intern(c,prompt);f->choice_actor_presence[0]=0;f->choice_continuation_presence[0]=0;for(int j=0;j<NCHI;j++){f->choice_int_presence[j]=0;f->choice_int_value[j]=0;}for(int j=0;j<NCHB;j++){f->choice_bool_presence[j]=0;f->choice_bool_value[j]=0;}for(int j=0;j<NCHS;j++){f->choice_str_presence[j]=0;f->choice_str_slot[j]=0;}}
DFN void d_choice_int(DX*c,const char*key,int value){CHOICE_INT_CASES d_fail(c,4);}
DFN void d_choice_bool(DX*c,const char*key,int value){CHOICE_BOOL_CASES d_fail(c,4);}
DFN void d_choice(DX*c,const char*kind,int count){if(d_finished(c)||d_bad(c))return;DS*f=c->s;if(d_streq(kind,"discard")||d_streq(kind,"enemy_hand")||d_streq(kind,"hand_redraw")||d_streq(kind,"hand_deck_placement")){int side=d_streq(kind,"enemy_hand")?1-c->side:c->side;int n=f->zone_count[side*4];if(!n)return;if(n>CH){d_fail(c,2);return;}f->choice_card_count[0]=n;for(int j=0;j<n;j++)f->choice_card_index[j]=f->zone_card_index[side*4*Z+j];}
 else {if(count<=0)return;if(count>CH){d_fail(c,2);return;}f->choice_card_count[0]=count;}
 const char*prompt=d_streq(kind,"discard")?PROMPT_DISCARD:d_streq(kind,"enemy_hand")?PROMPT_ENEMY:d_streq(kind,"hand_redraw")?PROMPT_REDRAW:d_streq(kind,"hand_deck_placement")?PROMPT_PLACE:d_streq(kind,"inspect_top")?PROMPT_INSPECT:PROMPT_KIT;
 d_reset_choice(c,kind,prompt);if(d_streq(kind,"hand_deck_placement")){d_choice_bool(c,"deck_order",1);d_choice_bool(c,"hand_placement",1);d_choice_bool(c,"multi_bottom",1);d_choice_int(c,"min_bottom",0);d_choice_int(c,"max_bottom",1);d_choice_bool(c,"execute_divination",0);f->choice_actor_presence[0]=2;f->choice_actor_side[0]=c->side;f->choice_actor_slot[0]=c->actor;}}
DFN void d_begin_divination(DX*c,int requested,int min_bottom,const char*continuation){DS*f=c->s;int count=f->zone_count[c->side*4+1];if(count>requested)count=requested;if(count<=0){kit_heiyu_resume_inspection(c);return;}if(count>CH){d_fail(c,2);return;}f->choice_card_count[0]=count;for(int j=0;j<count;j++){int card=f->zone_card_index[(c->side*4+1)*Z];f->choice_card_index[j]=card;d_remove_zone_card(c,c->side,1,card);}char prompt[256];d_divination_prompt(prompt,count,min_bottom);d_reset_choice(c,"heiyu_divination",prompt);d_choice_bool(c,"deck_cards",1);d_choice_bool(c,"deck_order",1);d_choice_bool(c,"execute_divination",1);d_choice_bool(c,"multi_bottom",1);d_choice_int(c,"snapshot_n",requested);d_choice_int(c,"min_bottom",min_bottom);d_choice_int(c,"max_bottom",count);f->choice_actor_presence[0]=2;f->choice_actor_side[0]=c->side;f->choice_actor_slot[0]=c->actor;f->choice_continuation_presence[0]=2;f->choice_continuation_slot[0]=d_intern(c,continuation);}
DFN void d_continue_choice(DX*c,const char*name,const int*bottoms,int count,int execute){if(!d_streq(name,"heiyu.divination")){d_fail(c,4);return;}const int*saved=c->choice_bottom;int n=c->choice_bottom_count;c->choice_bottom=bottoms;c->choice_bottom_count=count;int old_execute=c->choice_execute;c->choice_execute=execute;kit_continue_heiyu_divination(c);c->choice_bottom=saved;c->choice_bottom_count=n;c->choice_execute=old_execute;}
DFN void d_lingke_finish_kit_choice(DX*c){int selected=c->choice_selected;if(selected<0){d_fail(c,1);return;}int n=c->s->op_choice_card_count[0];d_add_hand(c,c->side,selected);for(int j=0;j<n;j++){int card=c->s->op_choice_card_index[j];if(card!=selected)d_insert_deck(c,c->side,card,d_deck_count(c,c->side));}c->s->op_choice_card_count[0]=0;d_shuffle_zone(c,c->side,1);}
DFN void d_lingke_offer_owned(DX*c,int owner,int count){DS*f=c->s;int unique[K],n=0,seen[NTCARD]={0};int deck=d_deck_count(c,c->side);for(int j=0;j<deck;j++){int card=f->zone_card_index[(c->side*4+1)*Z+j],id=d_card_template(c,card);if(d_card_owner(c,card)==owner&&!seen[id]){unique[n++]=card;seen[id]=1;}}if(!n)return;d_shuffle_refs(c,unique,n);if(n>count)n=count;for(int j=0;j<n;j++){f->choice_card_index[j]=unique[j];d_remove_zone_card(c,c->side,1,unique[j]);}f->choice_card_count[0]=n;d_reset_choice(c,"kit",PROMPT_KIT);d_choice_bool(c,"deck_cards",1);}
'''


def _summon_source(layout):
    from app.modules.card_game.content.duel_v2.characters.zaowu import FLAVOR,_gulang_passive,ITEMS
    specs=schema.array_specs(layout)
    text='DFN int d_summon(DX*c,const char*kind){int item=-1;'+''.join(f'if(d_streq(kind,"{kind}"))item={i};'for i,kind in enumerate(ITEMS))+'if(item<0){d_fail(c,4);return -1;}DS*f=c->s;int side=c->side,slot=f->player_character_count[side];if(slot<0||slot>=C||f->next_entity[0]>=2147483647){d_fail(c,2);return -1;}int front=d_front(c,side);if(front>=0)d_move_out(c,side,front);int h=side*C+slot;'
    for name,(dtype,shape)in specs.items():
        if name.startswith('character_'):
            width=prod(shape[2:]);text+=f'for(int j=0;j<{width};j++)f->{name}[h*{width}+j]=0;'
    text+='int entity=++f->next_entity[0];char id[32];const char*prefix="summon_e";int n=0;while(prefix[n]){id[n]=prefix[n];n++;}d_uint_text(id+n,entity);f->character_valid[h]=1;f->character_entity_slot[h]=entity;f->character_template[h]=0;f->character_id_slot[h]=d_intern(c,id);f->player_character_count[side]++;int order=f->player_order_count[side];if(order>=C){d_fail(c,2);return -1;}f->player_order_slot[side*C+order]=slot;f->player_order_count[side]++;f->character_flags_presence[h]=2;'
    for k,v in (('hp','4'),('max_hp','4'),('shield','0'),('energy','0'),('base_attack','d_kind_count(c,side)'),('base_max_hp','4'),('harmony','0'),('ultimate_turns','0'),('growth','0'),('family','0'),('gaze','0'),('dream','0'),('down_turns','0')):text+=f'd_set_ci(c,side,slot,"{k}",{v});'
    text+='d_set_cb(c,side,slot,"summoned",1);d_set_cb(c,side,slot,"awakened",0);'
    for k,v in [('name','鬼郎丸'),('attribute',''),('faction',''),('portrait','/static/images/characters/portrait/鬼郎丸.webp'),('avatar','/static/images/characters/portrait/鬼郎丸.webp'),('awakened_passive',''),('flavor',FLAVOR)]:text+=f'd_set_cstr(c,side,slot,"{k}",{_literal(v)});'
    text+='d_set_cstr(c,side,slot,"shape",nullptr);const char*passive=nullptr;switch(item){'+''.join(f'case {i}:passive={_literal(_gulang_passive(kind))};break;'for i,kind in enumerate(ITEMS))+'}d_set_cstr(c,side,slot,"passive",passive);int holder=d_holder(c,side,slot);f->flag_str_presence[holder*FS+FS_gulang_item]=2;f->flag_str_slot[holder*FS+FS_gulang_item]=d_intern(c,kind);d_set_front(c,side,slot);f->player_last_front_presence[side]=1;f->player_last_front_slot[side]=0;d_event(c);d_event(c);return slot;}\n'
    return text


def _mark_queue_source(layout):
    specs=schema.array_specs(layout)
    ops={name:spec for name,spec in specs.items()if name.startswith('op_')or name=='operation_present'}
    text=f'#define MARKS {layout.max_deferred_marks}\n'
    text+='DFN int d_clone_mark_card(DX*c,int source,int*map){if(source<0)return source;if(source>=K||!c->s->card_valid[source]){d_fail(c,1);return -1;}if(map[source]>=0)return map[source];int dst=d_alloc_card_row(c);if(dst<0)return -1;map[source]=dst;DS*f=c->s;'
    for name,(dtype,shape)in specs.items():
        if name.startswith('card_'):
            width=prod(shape[1:]);text+=f'for(int j=0;j<{width};j++)f->{name}[dst*{width}+j]=f->{name}[source*{width}+j];'
    text+='f->card_is_snapshot[dst]=1;if(f->card_is_view[source]){int physical=f->card_physical_index[source];if(physical<0||physical>=K||f->card_is_view[physical]){d_fail(c,1);return -1;}f->card_physical_index[dst]=d_clone_mark_card(c,physical,map);}return dst;}\n'
    text+='DFN void d_defer_marked_card(DX*c){DS*f=c->s;int m=f->mark_queue_count[0];if(m<0||m>=MARKS){d_fail(c,2);return;}'
    for name,(dtype,shape)in ops.items():text+=f'for(int j=0;j<{prod(shape)};j++)f->mark_{name}[m*{prod(shape)}+j]=f->{name}[j];'
    text+='int map[K];for(int j=0;j<K;j++)map[j]=-1;f->mark_op_card_index[m]=d_clone_mark_card(c,f->op_card_index[0],map);for(int j=0;j<f->op_choice_card_count[0];j++)f->mark_op_choice_card_index[m*CH+j]=d_clone_mark_card(c,f->op_choice_card_index[j],map);if(d_bad(c))return;f->mark_queue_presence[0]=2;f->mark_queue_count[0]++;}\n'
    text+='DFN void d_settle_mark_queue(DX*c){DS*f=c->s;int n=f->mark_queue_count[0];if(f->mark_queue_presence[0]!=2)return;if(n<0||n>MARKS){d_fail(c,1);return;}DOpSnap parent,queue[MARKS];life_save_op(c,&parent);for(int m=0;m<n;m++){queue[m].context=*c;'
    for name,(dtype,shape)in ops.items():text+=f'for(int j=0;j<{prod(shape)};j++)queue[m].{name}[j]=f->mark_{name}[m*{prod(shape)}+j];'
    text+='queue[m].context.side=f->mark_op_side[m];queue[m].context.actor=f->mark_op_actor_slot[m];queue[m].context.card=f->mark_op_card_index[m];}f->mark_queue_presence[0]=0;f->mark_queue_count[0]=0;for(int m=0;m<n&&!d_finished(c)&&!d_bad(c);m++){life_restore_op(c,&queue[m]);d_fire(c,"after_marked_card",nullptr,0,1);}life_restore_op(c,&parent);}\n'
    return text

_BASIC+=r'''
DFN int d_kind_count(DX*c,int by){int total=0;for(int side=0;side<2;side++){if(d_zone_has(c,side,"burn")&&d_zone_int(c,side,"burn","by",-1)==by)total++;for(int j=0;j<COMBAT_D;j++)if(c->s->zone_dot_valid[side*COMBAT_D+j]&&c->s->zone_dot_by[side*COMBAT_D+j]==by&&c->s->zone_dot_stacks[side*COMBAT_D+j]>0)total++;}return total;}
DFN void d_set_seal_normal_attack(DX*c,int side,int actor){int at=side*PS+PS_SEAL;c->s->player_str_presence[at]=2;c->s->player_str_slot[at]=c->s->character_id_slot[c->side*C+actor];}
DFN int d_lingke_chain_bump(DX*c,int owner,int actor){int h=c->side*C+owner,entity=d_entity(c,DR{c->side,actor});int count=c->s->character_lingke_presence[h]==2&&c->s->character_lingke_entity_slot[h]==entity?c->s->character_lingke_count[h]+1:1;c->s->character_lingke_presence[h]=2;c->s->character_lingke_entity_slot[h]=entity;c->s->character_lingke_id_slot[h]=c->s->character_id_slot[c->side*C+actor];c->s->character_lingke_count[h]=count%3;return count;}
DFN void d_shuffle_refs(DX*c,int*refs,int n){if(n<0||n>K||c->s->rng[0]<0){d_fail(c,1);return;}uint64_t seed=(uint64_t)c->s->rng[0],value=0;uint32_t words[2]={(uint32_t)seed,(uint32_t)(seed>>32)};bmt_state_t state;bmt_seed_words(&state,words,words[1]?2:1);for(int j=n-1;j>0;j--){int bits=bmt_bit_length_u64(j+1),ok=0;for(int k=0;k<4096;k++){bmt_getbits64(&state,bits,&value);if(value<(uint64_t)(j+1)){ok=1;break;}}if(!ok){d_fail(c,2);return;}int old=refs[j];refs[j]=refs[(int)value];refs[(int)value]=old;}bmt_getbits64(&state,63,&value);c->s->rng[0]=(long long)value;}
'''
_BASIC=_BASIC.replace('PS_SEAL',str(schema.PLAYER_STR_FIELDS.index('seal_normal_attack')))

_BASIC+=r'''
DFN void d_resolve_release_card_ex(DX*c,int card,DR target,int preserve_parent){if(d_finished(c)||d_bad(c)||card<0||card>=K){if(card<0)d_fail(c,1);return;}DOpSnap parent;life_save_op(c,&parent);DS*f=c->s;int old_p=f->draining_presence[0],old_v=f->draining[0];d_detach_card(c,c->side,card);f->draining_presence[0]=2;f->draining[0]=1;DX own=*c;own.card=card;own.target=target;own.option_slot=-1;int owner=d_card_owner(&own,card);if(owner==c->actor){int id=d_card_template(&own,card);const char*name=d_card_default_str(id,KS_character_id);if(name&&d_eq(c,f->character_id_slot[c->side*C+c->actor],name))d_restore_heiyu_spec(&own,card);}d_play_card(&own,0,1);if(preserve_parent)life_restore_op(c,&parent);f->draining_presence[0]=old_p;f->draining[0]=old_v;}
DFN void d_resolve_release_card_target(DX*c,int card,DR target){d_resolve_release_card_ex(c,card,target,1);}
DFN void d_resolve_release_card(DX*c,int card){d_resolve_release_card_ex(c,card,DR{-1,-2},1);}
DFN void d_resolve_release_named(DX*c,const char*card_id){int card=d_make_card(c,c->side,card_id,c->actor);if(card>=0)d_resolve_release_card(c,card);}
'''

_BASIC+=r'''
DFN int d_card_trait(DX*c,int card,const char*owner){return d_card_str_is(c,card,"character_id",owner)||d_card_str_is(c,card,"original_character_id",owner);}
DFN int d_card_makes_instant(DX*c,int side,int actor,int card){if(d_card_trait(c,card,"nanali")&&d_card_is(c,card,"NF01")&&c->s->turn[0]>=5)return 1;if(d_card_trait(c,card,"haiyue")){if(d_card_is(c,card,"U01")&&c->s->active_side[0]!=side)return 1;if((d_card_is(c,card,"U02")||d_card_is(c,card,"U07"))&&d_teammate_attacked(c,side,actor))return 1;}if(d_card_trait(c,card,"yi")&&d_card_is(c,card,"I08")){int a=d_hero(c,side,"yi");if(a<0){d_fail(c,1);return 0;}if(d_alive(c,side,a)&&d_zone_has(c,1-side,"delay"))return 1;}return 0;}
DFN void d_play_option(DX*c,int response,int release){int id=d_card_template(c,c->card),chosen=-1;const char*names[3]={"base",nullptr,nullptr};int costs[3]={0,0,0};int n=0;if(id==CARD_Q01){n=2;names[1]="boost";costs[1]=3;}else if(id==CARD_Q02){n=2;names[1]="release_a";costs[1]=3;}else if(id==CARD_Q03){n=3;names[1]="release_b";costs[1]=3;names[2]="release_ba";costs[2]=6;}if(!n){c->option_slot=-1;return;}chosen=0;if(!response&&!release&&c->option_slot>=0)for(int j=0;j<n;j++)if(d_eq(c,c->option_slot,names[j]))chosen=j;c->option_slot=d_intern(c,names[chosen]);if(!release&&costs[chosen]){int amount=d_ci(c,c->side,c->actor,"jingu");if(amount<costs[chosen]){d_fail(c,1);return;}d_set_ci(c,c->side,c->actor,"jingu",amount-costs[chosen]);d_event(c);}}
'''

_BASIC+=r'''
DFN void kit_xiaozhi_release(DX*c,const char*card_id){if(d_finished(c)||d_bad(c))return;int id=d_card_id(card_id);if(!id){d_fail(c,1);return;}DOpSnap parent;life_save_op(c,&parent);DS*f=c->s;int virtual_card=-id-2;c->card=virtual_card;f->op_card_index[0]=virtual_card;d_set_op_int(c,"card_bonus",0);d_set_op_int(c,"card_set_attack",d_card_int(c,virtual_card,"attack"));f->op_first_bonus[0]=0;f->op_redeem_bonus[0]=0;f->op_harmony_presence[0]=1;f->op_harmony_slot[0]=0;d_set_op_int(c,"support",0,1);f->op_last_attack_target_presence[0]=1;f->op_option_id_presence[0]=2;f->op_option_id_slot[0]=d_intern(c,"base");d_event(c);d_shield(c,DR{c->side,c->actor},d_card_int(c,virtual_card,"shield"));d_card_effect(c);int count=f->op_participant_count[0],sides[PART],actors[PART];if(count<0||count>PART){d_fail(c,2);return;}for(int j=0;j<count;j++){sides[j]=f->op_participant_side[j];actors[j]=f->op_participant_slot[j];}life_restore_op(c,&parent);f->op_participant_count[0]=count;for(int j=0;j<count;j++){f->op_participant_side[j]=sides[j];f->op_participant_slot[j]=actors[j];}}
'''

_BASIC+='DFN const char*d_attribute_text(int code){switch(code){'+''.join(f'case {i+1}:return {_literal(a)};'for i,a in enumerate(schema.SORTIE_ATTRIBUTES))+'default:return "";}}\n'

_BASIC+=r'''
DFN void d_attach_pact(DX*c,DR target){if(d_finished(c)||target.slot<0||!d_alive(c,target.side,target.slot))return;int before=d_flag(c,target.side,target.slot,"pact");d_set_flag(c,target.side,target.slot,"pact",1,1);if(!before)d_event(c);}
'''
