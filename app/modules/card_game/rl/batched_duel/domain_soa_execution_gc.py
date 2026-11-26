"""Direct collection of unreachable cards and strings at action boundaries.

References are enumerated by the domain schema. No generic objects or frames are
traced, and collection never executes or approximates a gameplay rule.
"""
from math import prod
from . import domain_soa_schema as schema


def emit_source(layout):
    specs=schema.array_specs(layout)
    text='''DFN void d_gc_card_mark(DX*c,int ref,int*mark){if(ref<0||ref>=K||!c->s->card_valid[ref]){d_fail(c,1);return;}mark[ref]=1;if(c->s->card_is_view[ref]){int p=c->s->card_physical_index[ref];if(p<0||p>=K||!c->s->card_valid[p]||c->s->card_is_view[p]){d_fail(c,1);return;}mark[p]=1;}}
DFN void d_collect_cards(DX*c){DS*f=c->s;int mark[K]={0};for(int side=0;side<2;side++)for(int z=0;z<4;z++){int n=f->zone_count[side*4+z];if(n<0||n>Z){d_fail(c,1);return;}for(int j=0;j<n;j++)d_gc_card_mark(c,f->zone_card_index[(side*4+z)*Z+j],mark);}if(f->pending_choice_present[0])for(int j=0;j<f->choice_card_count[0];j++)d_gc_card_mark(c,f->choice_card_index[j],mark);if(f->operation_present[0]){if(f->op_card_index[0]>=0)d_gc_card_mark(c,f->op_card_index[0],mark);for(int j=0;j<f->op_choice_card_count[0];j++)d_gc_card_mark(c,f->op_choice_card_index[j],mark);}if(f->mark_queue_presence[0]==2)for(int m=0;m<f->mark_queue_count[0];m++)if(f->mark_operation_present[m]){if(f->mark_op_card_index[m]>=0)d_gc_card_mark(c,f->mark_op_card_index[m],mark);for(int j=0;j<f->mark_op_choice_card_count[m];j++)d_gc_card_mark(c,f->mark_op_choice_card_index[m*CH+j],mark);}if(f->deferred_presence[0]==2)for(int j=0;j<f->deferred_count[0];j++)if(f->deferred_card_presence[j]==2)d_gc_card_mark(c,f->deferred_card_index[j],mark);if(d_bad(c))return;for(int ref=0;ref<K;ref++)if(!mark[ref]){'''
    for name,(dtype,shape)in specs.items():
        if name.startswith('card_'):
            width=prod(shape[1:]);text+=f'for(int j=0;j<{width};j++)f->{name}[ref*{width}+j]=0;'
    text+='}}\n'
    # Mark and rewrite only explicitly named string references.
    refs=[]
    def add(field,index,live):refs.append((field,index,live))
    for field in ('rules_version_slot','design_version_slot','current_action_id_slot'):add(field,'0','1')
    add('reason_slot','0','f->reason_presence[0]==2')
    add('player_name_slot','s','1')
    for field,pref,width in [('player_str_slot','player_str_presence',len(schema.PLAYER_STR_FIELDS)),('character_str_slot','character_str_presence',len(schema.CHARACTER_STR_FIELDS)),('card_str_slot','card_str_presence',len(schema.CARD_STR_FIELDS)),('flag_str_slot','flag_str_presence',len(schema.FLAG_STR_FIELDS))]:
        count=2 if field.startswith('player_')else 2*layout.max_characters if field.startswith('character_')else layout.max_cards if field.startswith('card_')else layout.holder_count
        valid='1'if field.startswith('player_')else'f->character_valid[s]'if field.startswith('character_')else'f->card_valid[s]'if field.startswith('card_')else'exists(f,s)'
        for i in range(width):add(field,f's*{width}+{i}',f'{valid}&&f->{pref}[s*{width}+{i}]==2')
    add('player_used_key_slot','s*NU+j','j<f->player_used_count[s]')
    add('character_id_slot','s','f->character_valid[s]')
    add('character_equipment_id_slot','s','f->character_valid[s]&&f->character_equipment_presence[s]==2')
    add('character_lingke_id_slot','s','f->character_valid[s]&&f->character_lingke_presence[s]==2')
    for field in ('effect_id_slot','effect_definition_slot','effect_property_slot','effect_source_key_slot','effect_label_slot'):add(field,'s','f->effect_valid[s]')
    for field,presence in [('effect_stack_key_slot','effect_stack_key_presence'),('effect_consume_on_slot','effect_consume_on_presence')]:add(field,'s',f'f->effect_valid[s]&&f->{presence}[s]==2')
    add('effect_condition_arg_string','s','f->effect_valid[s/(Q*NA)]&&f->effect_condition_arg_presence[s]==2&&f->effect_condition_arg_kind[s]==3')
    add('zone_dot_source_slot','s','f->zone_dot_valid[s]')
    add('choice_prompt_slot','0','f->pending_choice_present[0]')
    add('choice_continuation_slot','0','f->pending_choice_present[0]&&f->choice_continuation_presence[0]==2')
    for j in range(len(schema.CHOICE_STR_KEYS)):add('choice_str_slot',str(j),f'f->pending_choice_present[0]&&f->choice_str_presence[{j}]==2')
    for field,presence in [('op_target_slot','op_target_presence'),('op_harmony_slot','op_harmony_presence'),('op_option_id_slot','op_option_id_presence')]:
        add(field,'0',f'f->operation_present[0]&&f->{presence}[0]==2')
        add('mark_'+field,'s',f'f->mark_queue_presence[0]==2&&s<f->mark_queue_count[0]&&f->mark_operation_present[s]&&f->mark_{presence}[s]==2')
    for j in range(len(schema.OP_STR_KEYS)):
        add('op_str_slot',str(j),f'f->operation_present[0]&&f->op_str_presence[{j}]==2')
        add('mark_op_str_slot',f's*{len(schema.OP_STR_KEYS)}+{j}',f'f->mark_queue_presence[0]==2&&s<f->mark_queue_count[0]&&f->mark_operation_present[s]&&f->mark_op_str_presence[s*{len(schema.OP_STR_KEYS)}+{j}]==2')
    add('deferred_kind_slot','s','f->deferred_presence[0]==2&&s<f->deferred_count[0]')
    add('down_response_target_slot','s','s<f->down_response_count[0]')
    def stmt(field,index,live):
        return f'if({live})d_gc_string_ref(c,f->{field}+{index},mark,map,pass);'
    text+='''DFN void d_gc_string_ref(DX*c,int*ref,int*mark,int*map,int pass){int slot=*ref;if(slot<0||slot>=c->s->string_count[0]||slot>=SS){d_fail(c,1);return;}if(pass==0)mark[slot]=1;else {if(map[slot]<0){d_fail(c,1);return;}*ref=map[slot];}}
DFN void d_collect_strings(DX*c){DS*f=c->s;int mark[SS]={0},map[SS];for(int j=0;j<SS;j++)map[j]=-1;int old_count=f->string_count[0];if(old_count<0||old_count>SS){d_fail(c,1);return;}for(int pass=0;pass<2;pass++){
'''
    for field,index,live in refs:
        if field=='player_used_key_slot':text+='for(int s=0;s<2;s++)for(int j=0;j<NU;j++){'+stmt(field,index,live)+'}\n';continue
        if field.startswith('player_'):count=2
        elif field.startswith('character_'):count=2*layout.max_characters
        elif field.startswith('card_'):count=layout.max_cards
        elif field.startswith('flag_'):count=layout.holder_count
        elif field=='effect_condition_arg_string':count=prod(specs[field][1])
        elif field.startswith('effect_'):count=layout.holder_count*layout.max_effects
        elif field=='zone_dot_source_slot':count=2*layout.max_dots
        elif field.startswith('mark_'):count=layout.max_deferred_marks
        elif field=='deferred_kind_slot':count=layout.max_deferred
        elif field=='down_response_target_slot':count=layout.max_down_responses
        else:count=None
        text+=('for(int s=0;s<'+str(count)+';s++){'+stmt(field,index,live)+'}'if count else stmt(field,index,live))+'\n'
    text+='''for(int side=0;side<2;side++)for(int z=0;z<5;z++)if(f->zone_status_presence[side*5+z]==2){int mask=f->zone_status_i32[(side*5+z)*8+7],n=d_zone_offset(z,mask,32,3);if(n>2){d_fail(c,1);return;}for(int j=0;j<n;j++)d_gc_string_ref(c,f->zone_status_slot+(side*5+z)*2+j,mark,map,pass);}
 if(d_bad(c))return;if(pass==0){int used=0,count=0,previous_end=0;for(int j=0;j<old_count;j++){int at=f->string_offset[j],n=f->string_length[j];if(at<previous_end||n<0||at<0||at+n>f->string_bytes_used[0]||at+n>SB){d_fail(c,1);return;}previous_end=at+n;if(mark[j]){map[j]=count;for(int k=0;k<n;k++)f->string_bytes[used+k]=f->string_bytes[at+k];f->string_offset[count]=used;f->string_length[count]=n;used+=n;count++;}}for(int j=count;j<old_count;j++){f->string_offset[j]=0;f->string_length[j]=0;}for(int j=used;j<f->string_bytes_used[0];j++)f->string_bytes[j]=0;f->string_bytes_used[0]=used;}
 }
 int count=0;for(int j=0;j<old_count;j++)if(map[j]>=0)count++;f->string_count[0]=count;}
'''
    return text
