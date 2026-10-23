"""Named physical mutations shared by direct domain execution handlers."""
from math import prod
from . import domain_soa_schema as schema

_STATUS=(('left','tick_side','slow','slow_turn','by','expires_side'),('left','by','stacks','source','applied_turn','infinite'),('left','by','source_cid','renewals'),('by','expires_side'),('by','expires_side','ends_left'))
_INTS={'left','slow','slow_turn','stacks','applied_turn','renewals','ends_left'}
_SIDES={'tick_side','by','expires_side'}

def emit_zone_source():
    text='DFN int d_zone_kind(const char*k){'+''.join(f'if(d_streq(k,"{k}"))return {i};'for i,k in enumerate(schema.ZONE_STATUS_KEYS))+'return -1;}\n'
    text+='DFN int d_zone_has(DX*c,int side,const char*kind){int z=d_zone_kind(kind);if(!d_valid_side(c,side)||z<0){d_fail(c,1);return 0;}return c->s->zone_status_presence[side*5+z]==2;}\n'
    text+='DFN int d_zone_key(int z,const char*key,int*type,int*ordinal){switch(z){'
    for z,keys in enumerate(_STATUS):
        counts=[0,0,0,0];text+=f'case {z}:'
        for bit,key in enumerate(keys):
            typ=0 if key in _INTS else 1 if key in _SIDES else 2 if key=='infinite' else 3
            text+=f'if(d_streq(key,"{key}")){{*type={typ};*ordinal={counts[typ]};return {bit};}}';counts[typ]+=1
        text+='break;'
    text+='}return -1;}\n'
    # Values are compacted by type among PRESENT keys, never fixed field ordinals.
    text+='DFN int d_zone_offset(int z,int mask,int bit,int type){int n=0,t=0,o=0;'
    for z,keys in enumerate(_STATUS):
        text+=f'if(z=={z}){{'
        for b,key in enumerate(keys):
            typ=0 if key in _INTS else 1 if key in _SIDES else 2 if key=='infinite' else 3
            text+=f'if({b}<bit&&type=={typ}&&(mask&(1<<{b})))n++;'
        text+='}'
    text+='return n;}\n'
    text+='''DFN int d_zone_int(DX*c,int side,const char*kind,const char*key,int fallback){int z=d_zone_kind(kind),type=0,ord=0;if(!d_valid_side(c,side)||z<0){d_fail(c,1);return fallback;}if(c->s->zone_status_presence[side*5+z]!=2)return fallback;int bit=d_zone_key(z,key,&type,&ord);if(bit<0){if(d_streq(key,"tick_side")||d_streq(key,"expires_side")||d_streq(key,"infinite")||d_streq(key,"renewals"))return fallback;d_fail(c,4);return fallback;}int h=side*5+z,mask=c->s->zone_status_i32[h*8+7];if(!(mask&(1<<bit)))return fallback;int at=d_zone_offset(z,mask,bit,type);if(type==0)return c->s->zone_status_i32[h*8+at];if(type==1)return c->s->zone_status_side[h*4+at];if(type==2)return c->s->zone_status_bool[h*2+at];return c->s->zone_status_slot[h*2+at];}
DFN void d_zone_set(DX*c,int side,const char*kind,const char*key,int value){int z=d_zone_kind(kind),type=0,ord=0;if(!d_valid_side(c,side)||z<0){d_fail(c,1);return;}int bit=d_zone_key(z,key,&type,&ord);if(bit<0){d_fail(c,4);return;}int h=side*5+z;int mask=c->s->zone_status_presence[h]==2?c->s->zone_status_i32[h*8+7]:0;int at=d_zone_offset(z,mask,bit,type);if(!(mask&(1<<bit))){int total=d_zone_offset(z,mask,32,type);for(int j=total;j>at;j--){if(type==0)c->s->zone_status_i32[h*8+j]=c->s->zone_status_i32[h*8+j-1];else if(type==1)c->s->zone_status_side[h*4+j]=c->s->zone_status_side[h*4+j-1];else if(type==2)c->s->zone_status_bool[h*2+j]=c->s->zone_status_bool[h*2+j-1];else c->s->zone_status_slot[h*2+j]=c->s->zone_status_slot[h*2+j-1];}}c->s->zone_status_presence[h]=2;c->s->zone_status_i32[h*8+7]=mask|(1<<bit);if(type==0)c->s->zone_status_i32[h*8+at]=value;else if(type==1)c->s->zone_status_side[h*4+at]=value;else if(type==2)c->s->zone_status_bool[h*2+at]=value!=0;else c->s->zone_status_slot[h*2+at]=value;}
DFN void d_zone_remove(DX*c,int side,const char*kind){int z=d_zone_kind(kind);if(!d_valid_side(c,side)||z<0){d_fail(c,1);return;}int h=side*5+z;c->s->zone_status_presence[h]=0;for(int j=0;j<8;j++)c->s->zone_status_i32[h*8+j]=0;for(int j=0;j<4;j++)c->s->zone_status_side[h*4+j]=0;for(int j=0;j<2;j++){c->s->zone_status_bool[h*2+j]=0;c->s->zone_status_slot[h*2+j]=0;}}
DFN void d_zone_source(DX*c,int side,const char*kind,int source_side,int actor){d_zone_set(c,side,kind,"by",source_side);if(d_streq(kind,"star"))d_zone_set(c,side,kind,"source_cid",c->s->character_id_slot[source_side*C+actor]);else if(d_streq(kind,"burn")){char b[128];b[0]=(char)('a'+source_side);b[1]=':';int slot=c->s->character_id_slot[source_side*C+actor],n=c->s->string_length[slot],at=c->s->string_offset[slot];if(n>120){d_fail(c,2);return;}for(int j=0;j<n;j++)b[j+2]=c->s->string_bytes[at+j];b[n+2]=0;d_zone_set(c,side,kind,"source",d_intern(c,b));}}
'''
    text=text.replace('DFN int d_zone_has(', 'DFN int d_status_has(').replace('DFN int d_zone_int(', 'DFN int d_status_int(').replace('DFN void d_zone_set(', 'DFN void d_status_set(').replace('DFN void d_zone_remove(', 'DFN void d_status_remove(')
    text+='''DFN int d_domain_dot_kind(const char*kind){if(d_streq(kind,"nightmare"))return 0;if(d_streq(kind,"etch"))return 1;if(d_streq(kind,"venom"))return 2;if(d_streq(kind,"guard"))return 3;return -1;}
DFN int d_domain_dot_at(DX*c,int side,int kind){if(!d_valid_side(c,side))return -1;for(int j=0;j<COMBAT_D;j++)if(c->s->zone_dot_valid[side*COMBAT_D+j]&&c->s->zone_dot_kind[side*COMBAT_D+j]==kind)return side*COMBAT_D+j;return -1;}
DFN int d_zone_has(DX*c,int side,const char*kind){int dot=d_domain_dot_kind(kind);return dot>=0?d_domain_dot_at(c,side,dot)>=0:d_status_has(c,side,kind);}
DFN int d_zone_int(DX*c,int side,const char*kind,const char*key,int fallback){int dot=d_domain_dot_kind(kind);if(dot<0)return d_status_int(c,side,kind,key,fallback);int at=d_domain_dot_at(c,side,dot);if(at<0)return fallback;if(d_streq(key,"by"))return c->s->zone_dot_by[at];if(d_streq(key,"stacks"))return c->s->zone_dot_stacks[at];if(d_streq(key,"source"))return c->s->zone_dot_source_slot[at];if(d_streq(key,"left"))return c->s->zone_dot_left_presence[at]==2?c->s->zone_dot_left[at]:fallback;if(d_streq(key,"applied_turn"))return c->s->zone_dot_applied_turn_presence[at]==2?c->s->zone_dot_applied_turn[at]:fallback;d_fail(c,4);return fallback;}
DFN void d_zone_remove(DX*c,int side,const char*kind){int dot=d_domain_dot_kind(kind);if(dot<0){d_status_remove(c,side,kind);return;}int at=d_domain_dot_at(c,side,dot);if(at>=0)c->s->zone_dot_valid[at]=0;}
DFN void d_zone_set(DX*c,int side,const char*kind,const char*key,int value){int dot=d_domain_dot_kind(kind);if(dot<0){d_status_set(c,side,kind,key,value);return;}int at=d_domain_dot_at(c,side,dot);if(at<0){d_fail(c,4);return;}if(d_streq(key,"stacks"))c->s->zone_dot_stacks[at]=value;else if(d_streq(key,"by"))c->s->zone_dot_by[at]=value;else if(d_streq(key,"left")){c->s->zone_dot_left_presence[at]=2;c->s->zone_dot_left[at]=value;}else d_fail(c,4);}
'''
    return text


def emit_operation_source(layout=None):
    text='DFN int d_op_int(DX*c,const char*key,int fallback){DS*f=c->s;if(d_streq(key,"pending_sortie"))return f->op_pending_sortie_presence[0]==2;'
    for key,col in [('first_bonus','op_first_bonus'),('redeem_bonus','op_redeem_bonus'),('record_candidate','op_record_candidate')]:text+=f'if(d_streq(key,"{key}"))return f->{col}[0];'
    for pref,keys in [('int',schema.OP_INT_KEYS),('flag',schema.OP_FLAG_KEYS)]:
        for i,key in enumerate(keys):text+=f'if(d_streq(key,"{key}"))return f->op_{pref}_presence[{i}]==2?f->op_{pref}_value[{i}]:fallback;'
    text+='if(d_streq(key,"heiyu_after_divination_code")){if(f->op_str_presence[0]!=2)return fallback;return d_eq(c,f->op_str_slot[0],"draw")?1:d_eq(c,f->op_str_slot[0],"sortie")?2:0;}if(d_streq(key,"return_after_resources"))return f->op_return_entity_presence[0]==2?f->op_return_entity[0]:fallback;d_fail(c,4);return fallback;}\n'
    for mode in ('set','del'):
        text+=('DFN void d_set_op_int(DX*c,const char*key,int value,int is_bool)'if mode=='set'else'DFN void d_del_op_int(DX*c,const char*key)')+'{DS*f=c->s;'
        for key,col in [('first_bonus','op_first_bonus'),('redeem_bonus','op_redeem_bonus'),('record_candidate','op_record_candidate')]:text+=f'if(d_streq(key,"{key}")){{f->{col}[0]='+('value'if mode=='set'else'0')+';return;}'
        for pref,keys in [('int',schema.OP_INT_KEYS),('flag',schema.OP_FLAG_KEYS)]:
            for i,key in enumerate(keys):text+=f'if(d_streq(key,"{key}")){{f->op_{pref}_presence[{i}]='+('2'if mode=='set'else'0')+f';f->op_{pref}_value[{i}]='+('value'if mode=='set'else'0')+';return;}'
        text+='if(d_streq(key,"heiyu_after_divination_code")){f->op_str_presence[0]='+('2'if mode=='set'else'0')+';f->op_str_slot[0]='+('d_intern(c,value==1?"draw":"sortie")'if mode=='set'else'0')+';return;}if(d_streq(key,"return_after_resources")){f->op_return_entity_presence[0]='+('2'if mode=='set'else'0')+';f->op_return_entity[0]='+('value'if mode=='set'else'0')+';return;}d_fail(c,4);}\n'
    text+='''DFN DR d_entity_ref(DX*c,int eid){for(int s=0;s<2;s++){if(c->s->player_entity_slot[s]==eid)return DR{s,-1};for(int j=0;j<c->s->player_character_count[s];j++)if(c->s->character_entity_slot[s*C+j]==eid)return DR{s,j};}d_fail(c,1);return DR{-1,-2};}
DFN DR d_target_slot_ref(DX*c,int slot){DS*f=c->s;for(int s=0;s<2;s++){char prefix[3]={(char)('a'+s),':',0};int n=f->string_length[slot],at=f->string_offset[slot];if(n<2||f->string_bytes[at]!=prefix[0]||f->string_bytes[at+1]!=':')continue;char id[128];if(n>125){d_fail(c,2);return DR{-1,-2};}for(int j=2;j<n;j++)id[j-2]=f->string_bytes[at+j];id[n-2]=0;if(d_streq(id,"player"))return DR{s,-1};int ch=d_hero(c,s,id);if(ch>=0)return DR{s,ch};}d_fail(c,1);return DR{-1,-2};}
DFN int d_ref_target_slot(DX*c,DR r){char text[128];text[0]=(char)('a'+r.side);text[1]=':';if(r.slot<0){const char*p="player";for(int j=0;j<7;j++)text[j+2]=p[j];}else{int slot=c->s->character_id_slot[r.side*C+r.slot],n=c->s->string_length[slot],at=c->s->string_offset[slot];if(n>120){d_fail(c,2);return 0;}for(int j=0;j<n;j++)text[j+2]=c->s->string_bytes[at+j];text[n+2]=0;}return d_intern(c,text);}
DFN DR d_op_target(DX*c,const char*key){int p=0,slot=0;if(d_streq(key,"target_id")){p=c->s->op_target_presence[0];slot=c->s->op_target_slot[0];}else if(d_streq(key,"last_attack_target")){p=c->s->op_last_attack_target_presence[0];slot=c->s->op_last_attack_target_slot[0];if(p==2)return DR{slot/256,slot%256==255?-1:slot%256};}else{d_fail(c,4);return DR{-1,-2};}return p==2?d_target_slot_ref(c,slot):DR{-1,-2};}
DFN void d_set_op_target(DX*c,const char*key,DR r){int slot=r.slot>=-1?d_ref_target_slot(c,r):0;int p=r.slot>=-1?2:0;if(d_streq(key,"target_id")){c->s->op_target_presence[0]=p;c->s->op_target_slot[0]=slot;}else if(d_streq(key,"last_attack_target")){c->s->op_last_attack_target_presence[0]=p;c->s->op_last_attack_target_slot[0]=r.side*256+(r.slot<0?255:r.slot);}else d_fail(c,4);}

'''
    text+='DFN void d_reset_operation(DX*c){DS*f=c->s;'
    for name,(dtype,shape)in schema.array_specs(layout or schema.DomainSoALayout()).items():
        if name.startswith('op_'):
            text+=f'for(int j=0;j<{prod(shape)};j++)f->{name}[j]=0;'
    text+='''f->operation_present[0]=1;f->op_side[0]=c->side;f->op_actor_slot[0]=c->actor;f->op_card_index[0]=c->card;f->op_target_presence[0]=1;f->op_harmony_presence[0]=1;f->op_card_mark_presence[0]=1;if(c->target.side>=0&&c->target.slot>=-1)d_set_op_target(c,"target_id",c->target);if(c->option_slot>=0){f->op_option_id_presence[0]=2;f->op_option_id_slot[0]=c->option_slot;}if(c->card>=0&&c->card<K&&f->card_jingu_mark_presence[c->card]==2){f->op_card_mark_presence[0]=2;f->op_card_mark_delta[0]=f->card_jingu_delta[c->card];f->op_card_mark_source_slot[0]=f->card_jingu_source_slot[c->card];}}
DFN void d_open_operation(DX*c){d_reset_operation(c);d_fire(c,"on_operation_start");}
'''
    return text
