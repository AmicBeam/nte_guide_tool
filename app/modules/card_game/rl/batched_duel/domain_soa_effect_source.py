"""Direct typed modifier writes and reductions for physical rule execution."""
from math import prod
from . import domain_soa_schema as schema


def emit_source(layout):
    specs=schema.array_specs(layout)
    text='DFN void d_effect_copy(DS*f,int dst,int src){'
    for name,(dtype,shape)in specs.items():
        if name.startswith('effect_'):
            n=prod(shape[2:]);text+=f'for(int j=0;j<{n};j++)f->{name}[dst*{n}+j]=f->{name}[src*{n}+j];'
    text+='}\nDFN void d_effect_zero(DS*f,int at){'
    for name,(dtype,shape)in specs.items():
        if name.startswith('effect_'):
            n=prod(shape[2:]);text+=f'for(int j=0;j<{n};j++)f->{name}[at*{n}+j]=0;'
    text+='}\n'
    from .domain_soa_execution_base import _TYPES
    import numpy as np
    text+='struct DEffectSnapshot {'
    keep=('effect_definition_slot','effect_value_kind','effect_source_entity_presence','effect_source_entity_slot','effect_source_key_slot','effect_label_slot','effect_visibility','effect_conditions_presence','effect_conditions_count')
    fields=[name for name in specs if name in keep or name.startswith('effect_condition_')]
    for name in fields:
        dtype,shape=specs[name];text+=f'{_TYPES[np.dtype(dtype).name]} {name}[{prod(shape[2:])}];'
    text+='};\nDFN void d_capture_effect(DS*f,int e,DEffectSnapshot*out){'
    for name in fields:
        width=prod(specs[name][1][2:]);text+=f'for(int j=0;j<{width};j++)out->{name}[j]=f->{name}[e*{width}+j];'
    text+='}\nDFN void d_restore_max_metadata(DS*f,int e,const DEffectSnapshot*in){'
    for name in fields:
        if name=='effect_value_kind':continue
        width=prod(specs[name][1][2:]);text+=f'for(int j=0;j<{width};j++)f->{name}[e*{width}+j]=in->{name}[j];'
    text+='}\n'
    text+='''DFN void d_effect_remove(DX*c,int holder,int j){int n=c->s->holder_effect_count[holder];if(j<0||j>=n||n>E){d_fail(c,1);return;}for(int k=j;k<n-1;k++)d_effect_copy(c->s,holder*E+k,holder*E+k+1);d_effect_zero(c->s,holder*E+n-1);c->s->holder_effect_count[holder]--;}
DFN int d_effect_enum(DX*c,const char*value,int type){
'''
    for typ,values in enumerate((schema.EFFECT_OPS,schema.EFFECT_SCOPES,schema.EFFECT_KINDS,schema.EFFECT_VISIBILITY,schema.EFFECT_PHASES,schema.EFFECT_CLEAR_EVENTS,schema.EFFECT_CONDITION_IDS,schema.EFFECT_CONDITION_ARGS)):
        text+=f'if(type=={typ}){{'+''.join(f'if(d_streq(value,"{v}"))return {i};'for i,v in enumerate(values))+'}'
    text+='d_fail(c,4);return 0;}\n'
    text+='''DFN int d_effect_has_clear(DX*c,int e,const char*boundary){for(int j=0;j<c->s->effect_clear_count[e];j++){int k=c->s->effect_clear_event[e*CL+j];'''+''.join(f'if(k=={i}&&d_streq(boundary,"{v}"))return 1;'for i,v in enumerate(schema.EFFECT_CLEAR_EVENTS))+'''}return 0;}
DFN void d_clear_effect_id(DX*c,DR r,const char*id){int h=d_holder(c,r.side,r.slot);for(int j=c->s->holder_effect_count[h]-1;j>=0;j--)if(d_eq(c,c->s->effect_definition_slot[h*E+j],id))d_effect_remove(c,h,j);}
DFN int d_has_effect_id(DX*c,int side,int actor,const char*id){int h=d_holder(c,side,actor);for(int j=0;j<c->s->holder_effect_count[h];j++)if(d_eq(c,c->s->effect_definition_slot[h*E+j],id))return 1;return 0;}
DFN void d_effect(DX*c,DR r,const DEffect*d){
 if(d_bad(c))return;DS*f=c->s;int h=d_holder(c,r.side,r.slot);int op=d_effect_enum(c,d->op?d->op:"add",0),scope=d_effect_enum(c,d->scope?d->scope:"self",1),kind=d_effect_enum(c,d->kind?d->kind:"persistent",2);if(d->stacking&&!d_streq(d->stacking,"independent")&&!d_streq(d->stacking,"replace")&&!d_streq(d->stacking,"refresh")&&!d_streq(d->stacking,"max")){d_fail(c,1);return;}
 int id=d_intern(c,d->id),prop=d_intern(c,d->property),src=d->source_entity>=0?d->source_entity:d_entity(c,DR{c->side,c->actor});
 const char*source_key=c->card!=-1?d_card_name(d_card_template(c,c->card)):"ability";int key=d->source_key?d_intern(c,d->source_key):c->effect_source_slot>=0?c->effect_source_slot:d_intern(c,source_key);
 int stack=d->stack_key&&d->stack_key[0]?d_intern(c,d->stack_key):-1;int n=f->holder_effect_count[h],matches[E],nm=0;if(n<0||n>E){d_fail(c,1);return;}
 for(int j=0;j<n;j++){int e=h*E+j;int same=f->effect_property_slot[e]==prop&&f->effect_scope[e]==scope&&f->effect_op[e]==op&&f->effect_kind[e]==kind;
 if(stack>=0)same=same&&f->effect_stack_key_presence[e]==2&&f->effect_stack_key_slot[e]==stack;
 else same=same&&f->effect_definition_slot[e]==id&&f->effect_source_entity_presence[e]==2&&f->effect_source_entity_slot[e]==src&&f->effect_source_key_slot[e]==key;
 if(same)matches[nm++]=j;}
 double value=d->value;int effect_id=0,old_kind=-1,restore_max=0;DEffectSnapshot previous={};
 if(nm&&d->stacking&&!d_streq(d->stacking,"independent")){int e=h*E+matches[nm-1];effect_id=f->effect_id_slot[e];if(d_streq(d->stacking,"refresh")||(d_streq(d->stacking,"max")&&eval_value(f,e,f->row_error)>value)){value=eval_value(f,e,f->row_error);old_kind=f->effect_value_kind[e];if(d_streq(d->stacking,"max")){d_capture_effect(f,e,&previous);restore_max=1;}}for(int k=nm-1;k>=0;k--)d_effect_remove(c,h,matches[k]);}
 else {int serial=r.slot<0?d_pi(c,r.side,"effect_seq"):d_ci(c,r.side,r.slot,"effect_seq");if(serial>=2147483647){d_fail(c,2);return;}serial++;if(r.slot<0)d_set_pi(c,r.side,"effect_seq",serial);else d_set_ci(c,r.side,r.slot,"effect_seq",serial);effect_id=d_identity_text(c,d_entity(c,r),":effect:",serial);}
 if(d->max_stacks>0&&(!d->stacking||d_streq(d->stacking,"independent"))&&nm>=d->max_stacks){int remove=nm-d->max_stacks+1;for(int j=remove-1;j>=0;j--)d_effect_remove(c,h,matches[j]);}n=f->holder_effect_count[h];if(n>=E){d_fail(c,2);return;}int e=h*E+n;d_effect_zero(f,e);f->holder_effect_count[h]++;if(r.slot<0)f->player_effects_presence[r.side]=2;else f->character_effects_presence[r.side*C+r.slot]=2;
 f->effect_valid[e]=1;f->effect_id_slot[e]=effect_id;f->effect_definition_slot[e]=id;f->effect_property_slot[e]=prop;f->effect_op[e]=op;
 if(!qfinite(value)||value>2147483647.||value< -2147483648.){d_fail(c,2);return;}f->effect_value_kind[e]=old_kind>=0?old_kind:d->value_kind>=0?d->value_kind:value==(int)value?0:1;f->effect_value_int[e]=(int)value;f->effect_value_float[e]=value;
 f->effect_source_entity_presence[e]=2;f->effect_source_entity_slot[e]=src;f->effect_source_key_slot[e]=key;f->effect_scope[e]=scope;f->effect_kind[e]=kind;f->effect_layer[e]=d->layer;
 f->effect_label_slot[e]=d->label&&d->label[0]?d_intern(c,d->label):id;f->effect_visibility[e]=d_effect_enum(c,d->visibility?d->visibility:"public",3);f->effect_show_marker_presence[e]=2;f->effect_show_marker[e]=d->show_marker>=0?d->show_marker:(d->expiry_phase&&d->expiry_phase[0])||(d->consume_on&&d->consume_on[0]);
 if(stack>=0){f->effect_stack_key_presence[e]=2;f->effect_stack_key_slot[e]=stack;}
 f->effect_clear_presence[e]=2;f->effect_clear_count[e]=d->clear_count>=0?d->clear_count:kind==1&&!d->clear_on?0:1;f->effect_clear_event[e*CL]=d_effect_enum(c,d->clear_on&&d->clear_on[0]?d->clear_on:"down",5);
 if(d->clear_on2){if(CL<2){d_fail(c,2);return;}f->effect_clear_count[e]=2;f->effect_clear_event[e*CL+1]=d_effect_enum(c,d->clear_on2,5);}
 if(d->clear_on3){if(CL<3){d_fail(c,2);return;}f->effect_clear_count[e]=3;f->effect_clear_event[e*CL+2]=d_effect_enum(c,d->clear_on3,5);}
 if(d->max_stacks>0){f->effect_stack_limit_presence[e]=2;f->effect_stack_limit[e]=d->max_stacks;}
 if(d->consume_on&&d->consume_on[0]){f->effect_consume_on_presence[e]=2;f->effect_consume_on_slot[e]=d_intern(c,d->consume_on);}
 if(d->expiry_phase&&d->expiry_phase[0]){f->effect_expiry_presence[e]=2;f->effect_expiry_side_presence[e]=2;f->effect_expiry_side[e]=d->expiry_side;f->effect_expiry_turn_presence[e]=2;f->effect_expiry_turn[e]=d->expiry_turn;f->effect_expiry_phase[e]=d_effect_enum(c,d->expiry_phase,4);}
 if(d->condition&&d->condition[0]){f->effect_conditions_presence[e]=2;f->effect_conditions_count[e]=1;int at=e*Q;f->effect_condition_id[at]=d_effect_enum(c,d->condition,6);f->effect_condition_args_presence[at]=d->condition_key&&d->condition_key[0]?2:0;
 if(d->condition_key&&d->condition_key[0]){int a=at*NA+d_effect_enum(c,d->condition_key,7);f->effect_condition_arg_presence[a]=2;if(d->condition_string){f->effect_condition_arg_kind[a]=3;f->effect_condition_arg_string[a]=d_intern(c,d->condition_string);}else{f->effect_condition_arg_kind[a]=d->condition_number==(int)d->condition_number?0:1;f->effect_condition_arg_int[a]=(int)d->condition_number;f->effect_condition_arg_float[a]=d->condition_number;}}}
 if(restore_max)d_restore_max_metadata(f,e,&previous);
}
DFN void d_consume_holder(DX*c,int h,const char*boundary,int clear){for(int j=c->s->holder_effect_count[h]-1;j>=0;j--){int e=h*E+j;int yes=clear?d_effect_has_clear(c,e,boundary):c->s->effect_consume_on_presence[e]==2&&d_eq(c,c->s->effect_consume_on_slot[e],boundary);if(yes)d_effect_remove(c,h,j);}}
DFN void d_consume(DX*c,const char*boundary){d_consume_holder(c,d_holder(c,c->side,c->actor),boundary,0);}
DFN void d_clear_effects(DX*c,const char*boundary){d_consume_holder(c,d_holder(c,c->side,c->actor),boundary,1);}
'''
    return text


def emit_evaluate_source(layout):
    from .domain_soa_execution_workspace import use_query_workspace
    from .domain_soa_character_queries import _source, ATTRIBUTES
    from .domain_soa_execution_base import _literal
    source,_=_source(layout)
    cond=source[source.index('QFN int conditions('):source.index('QFN int remaining(')]
    cond=cond.replace('int*err){','int damage_attribute,int*err){')
    cond=cond.replace('conditions(f,e,h,subject,src,err)','d_conditions(f,e,h,subject,src,damage_attribute,err)')
    cond=cond.replace('QFN int conditions(','QFN int d_conditions(').replace('QFN int applies(','QFN int d_applies(')
    old='yes=f->effect_condition_arg_presence[ab+ARG_attribute]<=1;'
    cond=cond.replace(old,'yes=damage_attribute<=0&&f->effect_condition_arg_presence[ab+ARG_attribute]<=1;'+''.join(f'if(damage_attribute=={i+1}&&f->effect_condition_arg_presence[ab+ARG_attribute]==2&&f->effect_condition_arg_kind[ab+ARG_attribute]==3&&qeq(f,f->effect_condition_arg_string[ab+ARG_attribute],{_literal(a)},err))yes=1;'for i,a in enumerate(ATTRIBUTES)))
    body=source[source.index('QFN double query('):source.index('QFN void compatibility(')]
    body=use_query_workspace(body)
    body=body.replace('QFN double query(const QRow*f,int subject,int property,double base,int*present,int*err)', 'QFN double d_query(const QRow*f,int subject,int property,const char*name,double base,int damage_attribute,int attacker,int battle,const Candidate*extra,int nextra,int*present,int*err)')
    start=body.index('if(property==0)match=');end=body.index('\n   if(match',start)
    body=body[:start]+'match=qeq(f,slot,name,err);'+body[end:]
    body=body.replace('applies(f,e,h,subject,err)','d_applies(f,e,h,subject,damage_attribute,err)')
    body=body.replace('}else if(property==2){int until=', '}else if(property==8){int amount=flag(f,h,FI_next_attack);if(amount)push(a,&n,amount,0,0,err);}else if(property==2){int until=')
    aura='  if(property==9&&id==ID_zhenhong&&heid(f,h)==heid(f,subject)&&(cb(f,h,CB_awakened)||f->player_front_presence[s]!=2||f->player_front_slot[s]!=c))push(a,&n,1,0,4,err);\n  if(property==8&&id==ID_xiaozhi&&s==holder_side(subject)&&sh==CARD_Q08&&attacker==c&&battle){int j=ci(f,h,CI_jingu,err);if(j>0)push(a,&n,j/2,0,0,err);}\n'
    body=body.replace('if(property==0&&id==ID_adler',aura+'  if(property==0&&id==ID_adler')
    body=body.replace('return reduce(a,n,base,present,err);','for(int j=0;j<nextra;j++)push(a,&n,extra[j].value,extra[j].layer,extra[j].op,err);return reduce(a,n,base,present,err);')
    body+='DFN double d_evaluate_entries(DX*c,DR r,const char*name,double base,const DEvt*event,const Candidate*extra,int nextra){int present=1,err=0,property=-1;'
    names=('attack.panel','attack.override','rule.damage_immune','damage.limit','card.instant','card.cost','rule.normal_attack_instant','marker.lingke_taunt','attack.hit','rule.gain_energy')
    body+=''.join(f'if(d_streq(name,"{n}"))property={i};'for i,n in enumerate(names))
    body+='int battle=0;if(c->card!=-1){int id=d_card_template(c,c->card);battle=d_streq(d_card_default_str(id,KS_type),"battle");if(c->card>=0&&c->s->card_str_presence[c->card*KS+KS_type]==2)battle=d_eq(c,c->s->card_str_slot[c->card*KS+KS_type],"battle");}double result=d_query(c->s,d_holder(c,r.side,r.slot),property,name,base,event?event->attribute:0,event?event->attacker:-1,battle,extra,nextra,&present,&err);if(err)d_fail(c,err);return result;}\n'
    body+='DFN double d_evaluate(DX*c,DR r,const char*name,double base,const DEvt*event){return d_evaluate_entries(c,r,name,base,event,nullptr,0);}'
    body+='''DFN void d_consume_applied(DX*c,DR target,const char*property,const char*boundary,const DEvt*event){int subject=d_holder(c,target.side,target.slot),err=0;for(int h=0;h<H;h++)if(exists(c->s,h))for(int j=c->s->holder_effect_count[h]-1;j>=0;j--){int e=h*E+j;if(d_eq(c,c->s->effect_property_slot[e],property)&&c->s->effect_consume_on_presence[e]==2&&d_eq(c,c->s->effect_consume_on_slot[e],boundary)&&d_applies(c->s,e,h,subject,event?event->attribute:0,&err))d_effect_remove(c,h,j);}if(err)d_fail(c,err);}'''
    return cond+body


def emit_clock_source():
    from .domain_soa_execution_base import _literal
    from app.modules.card_game.engine.duel_v2.flow import ultimate_spec
    text='DFN int d_ultimate_turns(int id){switch(id){'+''.join(f'case {value}:return {ultimate_spec(cid)["turns"]};'for cid,value in schema.CHARACTER_VOCAB.items())+'default:return 2;}}\n'
    text+='DFN const char*d_ultimate_phase(int id){switch(id){'+''.join(f'case {value}:return "{ultimate_spec(cid)["expires"]}";'for cid,value in schema.CHARACTER_VOCAB.items())+'default:return "turn_start";}}\n'
    text+='DFN int d_ultimate_instant(int id){switch(id){'+''.join(f'case {value}:return {int(ultimate_spec(cid)["kind"]=="instant")};'for cid,value in schema.CHARACTER_VOCAB.items())+'default:return 0;}}\n'
    text+='''DFN void d_schedule_ultimate(DX*c,int side,int actor,int turns,const char*phase){DX own=*c;own.side=side;own.actor=actor;own.card=-1;own.effect_source_slot=-1;d_clear_effect_id(&own,DR{side,actor},"status.ultimate");DEffect e={};e.id="status.ultimate";e.property="status.ultimate";e.value=1;e.op="add";e.scope="self";e.stacking="independent";e.source_key="ultimate";e.label='''+_literal('终结')+''';e.show_marker=0;e.expiry_side=side;e.expiry_turn=d_pi(&own,side,"turn_count")+(d_streq(phase,"turn_end")?0:turns);e.expiry_phase=phase;e.clear_on="down";e.clear_on2="ultimate_end";d_effect(&own,DR{side,actor},&e);}
DFN void d_prepare_effects(DX*c){d_migrate_legacy(c);DS*f=c->s;for(int h=0;h<H;h++)if(exists(f,h)){int owner=holder_side(h);for(int j=0;j<f->holder_effect_count[h];j++){int e=h*E+j;if(f->effect_expiry_presence[e]==2&&f->effect_expiry_remaining_presence[e]==2){int side=f->effect_expiry_side_presence[e]==2?f->effect_expiry_side[e]:owner;if(side<0||side>1){d_fail(c,1);return;}long long until=(long long)d_pi(c,side,"turn_count")+f->effect_expiry_remaining[e];if(until>2147483647||until< -2147483648LL){d_fail(c,2);return;}f->effect_expiry_side_presence[e]=2;f->effect_expiry_side[e]=side;f->effect_expiry_turn_presence[e]=2;f->effect_expiry_turn[e]=(int)until;f->effect_expiry_remaining_presence[e]=0;f->effect_expiry_remaining[e]=0;}}}
 for(int side=0;side<2;side++)for(int actor=0;actor<f->player_character_count[side];actor++)if(d_cb(c,side,actor,"awakened")&&d_ci(c,side,actor,"ultimate_turns")>0&&!d_has_effect_id(c,side,actor,"status.ultimate"))d_schedule_ultimate(c,side,actor,d_ci(c,side,actor,"ultimate_turns"),d_ultimate_phase(d_template(c,side,actor)));}
struct DExpired {int holder,definition,source,visibility;};
DFN void d_expire_effects(DX*c,int side,const char*phase){DS*f=c->s;int phase_id=d_effect_enum(c,phase,4),count=0;DExpired due[H*E];for(int h=0;h<H;h++)if(exists(f,h)){int j=0;while(j<f->holder_effect_count[h]){int e=h*E+j,clock=f->effect_expiry_side_presence[e]==2?f->effect_expiry_side[e]:holder_side(h);if(f->effect_expiry_presence[e]==2&&f->effect_expiry_phase[e]==phase_id&&clock==side&&f->effect_expiry_turn_presence[e]==2&&f->effect_expiry_turn[e]<=d_pi(c,side,"turn_count")){due[count++]=DExpired{h,f->effect_definition_slot[e],f->effect_source_entity_presence[e]==2?f->effect_source_entity_slot[e]:0,f->effect_visibility[e]};d_effect_remove(c,h,j);}else j++;}if(holder_char(h)<0)f->player_effects_presence[holder_side(h)]=2;else f->character_effects_presence[holder_side(h)*C+holder_char(h)]=2;}
 for(int j=0;j<count;j++)if(d_eq(c,due[j].definition,"status.ultimate"))d_end_ultimate(c,holder_side(due[j].holder),holder_char(due[j].holder));
 for(int h=0;h<H;h++)if(exists(f,h)&&holder_char(h)>=0)for(int j=0;j<f->holder_effect_count[h];j++){int e=h*E+j;if(d_eq(c,f->effect_definition_slot[e],"status.ultimate")&&f->effect_expiry_phase[e]==0){int clock=f->effect_expiry_side_presence[e]==2?f->effect_expiry_side[e]:holder_side(h);int turns=f->effect_expiry_turn[e]-d_pi(c,clock,"turn_count");d_set_ci(c,holder_side(h),holder_char(h),"ultimate_turns",turns>0?turns:0);}}
 for(int j=0;j<count&&!d_finished(c)&&!d_bad(c);j++){DEvt ev={};ev.effect_definition_slot=due[j].definition;ev.effect_source_entity=due[j].source;ev.holder_entity=heid(f,due[j].holder);for(int observer=0;observer<2;observer++){DX own=*c;own.side=observer;own.actor=d_order_slot(c,observer,0);own.card=-1;d_fire(&own,"on_effect_expired",&ev);}d_drain(c);if(!d_finished(c)&&due[j].visibility==0)d_event(c);}}
DFN void d_next_attack(DX*c,DR target,int amount){DEffect e={};e.id="attack.next";e.property="attack.hit";e.value=amount;e.op="add";e.scope="self";e.stacking="max";e.stack_key="next_attack";e.label='''+_literal('攻击强化')+''';e.consume_on="attack";e.source_key=c->card!=-1?d_card_name(d_card_template(c,c->card)):d_character_name(d_template(c,c->side,c->actor));d_effect(c,target,&e);}
'''
    return text
