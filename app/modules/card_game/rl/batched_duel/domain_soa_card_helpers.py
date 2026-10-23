"""Direct named SoA helpers for public-17 card identity, rewrite and release."""
from . import domain_soa_schema as schema
from app.modules.card_game.content.duel_v2.tutorial import lookup_card

supports_full_duel_rules = False


def _lit(text):
    return '"' + ''.join('\\x%02x' % b for b in text.encode('utf8')) + '"'


_LIGHT = ('\u5149', '\u7075', '\u76f8')
_DARK = ('\u6697', '\u9b42', '\u5492')
_ATTR = ('\u5149', '\u6697', '\u7075', '\u9b42', '\u76f8', '\u5492')
_DEFERRED = ('surplus', 'heiyu_release_after_attack', 'lingke_sync', 'genesis')


def emit_source(layout=None):
    layout=layout or schema.DomainSoALayout()
    parts = []
    def add(text):
        parts.append(text)
    add(_identity_source())
    add(_mark_source())
    add(_lookup_source())
    add(_rewrite_source())
    add(_zone_source())
    add(_extra_source(layout))
    return ''.join(parts)


def _identity_source():
    named = ''.join('if(d_streq(id,"%s"))return %d;' % item for item in schema.CHARACTER_VOCAB.items())
    return (
        'DFN int d_turn(DX* c){return c&&c->s?c->s->turn[0]:0;}\n'
        'DFN int d_character_template_named(const char* id){' + named + 'return 0;}\n'
        'DFN int d_physical_card(DX* c,int card){'
        'if(card<=-2)return card;'
        'if(card<0||card>=K||!c->s->card_valid[card]){d_fail(c,1);return -1;}'
        'if(!c->s->card_is_view[card])return card;'
        'int physical=c->s->card_physical_index[card];'
        'if(physical<0||physical>=K||!c->s->card_valid[physical]||c->s->card_is_view[physical]||physical==card){d_fail(c,1);return -1;}'
        'return physical;}\n'
        'DFN int d_card_is_unknown(DX* c,int card){'
        'if(card<0)return 0;int id=d_card_template(c,card);if(d_bad(c))return 0;if(id==0)return 1;'
        'int p=c->s->card_str_presence[card*KS+KS_id];'
        'if(p==2)return d_eq(c,c->s->card_str_slot[card*KS+KS_id],"unknown");'
        'return d_streq(d_card_default_str(id,KS_id),"unknown");}\n'
        'DFN int d_ap(DX* c,int side){return d_pi(c,side,"ap");}\n'
        'DFN void d_add_ap(DX* c,int side,int amount){'
        'if(d_finished(c)||d_bad(c)||!d_valid_side(c,side))return;'
        'long long value=(long long)d_pi(c,side,"ap")+amount;'
        'if(value>2147483647LL||value<-2147483648LL){d_fail(c,2);return;}'
        'd_set_pi(c,side,"ap",(int)value);}\n'
        'DFN int d_mimic_template(DX* c,int side,int actor){'
        'int h=d_holder(c,side,actor);if(d_bad(c))return 0;'
        'if(c->s->flag_mimic_presence[h]!=2||c->s->flag_mimic_id_presence[h]!=2)return 0;'
        'return c->s->flag_mimic_template[h];}\n'
        'DFN void d_set_mimic_template(DX* c,int side,int actor,int tmpl){'
        'int h=d_holder(c,side,actor);if(d_bad(c))return;'
        'if(tmpl<=0){c->s->flag_mimic_presence[h]=0;c->s->flag_mimic_id_presence[h]=0;c->s->flag_mimic_template[h]=0;return;}'
        'c->s->flag_mimic_presence[h]=2;c->s->flag_mimic_id_presence[h]=2;c->s->flag_mimic_template[h]=(short)tmpl;}\n'
        'DFN int d_weapon_matches(DX* c,int actor,const char* card_id){'
        'if(!d_valid_char(c,c->side,actor)||!card_id)return 0;'
        'int want=d_card_id(card_id);if(!want)return 0;'
        'int h=c->side*C+actor;'
        'if(c->s->character_equipment_presence[h]!=2)return 1;'
        'return c->s->character_equipment_card[h]==want;}\n'
    )


def _mark_source():
    return (
        'DFN int d_card_jingu(DX* c,int card,int* delta,int* source_entity){'
        'int physical=d_physical_card(c,card);if(physical<0||d_bad(c))return 0;'
        'if(c->s->card_jingu_mark_presence[physical]!=2)return 0;'
        'if(delta)*delta=c->s->card_jingu_delta[physical];'
        'if(source_entity)*source_entity=c->s->card_jingu_source_slot[physical];return 1;}\n'
        'DFN void d_set_card_jingu(DX* c,int card,int delta,int source_entity){'
        'int physical=d_physical_card(c,card);if(physical<0||d_bad(c))return;'
        'c->s->card_jingu_mark_presence[physical]=2;'
        'c->s->card_jingu_delta[physical]=delta;'
        'c->s->card_jingu_source_slot[physical]=source_entity;}\n'
        'DFN void d_clear_card_jingu(DX* c,int card){'
        'int physical=d_physical_card(c,card);if(physical<0||d_bad(c))return;'
        'c->s->card_jingu_mark_presence[physical]=0;'
        'c->s->card_jingu_delta[physical]=0;'
        'c->s->card_jingu_source_slot[physical]=0;}\n'
        'DFN int d_has_effect_id(DX* c,DR target,const char* id){'
        'int h=d_holder(c,target.side,target.slot);if(d_bad(c)||!id)return 0;'
        'int n=c->s->holder_effect_count[h],j;if(n<0||n>E){d_fail(c,1);return 0;}'
        'for(j=0;j<n;++j)if(d_eq(c,c->s->effect_definition_slot[h*E+j],id))return 1;return 0;}\n'
        'DFN void d_set_collapse_reduction(DX* c,int side,int actor,int source_entity,int layers){'
        'if(!d_valid_char(c,side,actor))return;int h=side*C+actor;int n=c->s->character_reductions_count[h];'
        'int cap=2*C,at=-1,j;if(n<0||n>cap){d_fail(c,1);return;}'
        'for(j=0;j<n;++j)if(c->s->character_reductions_entity[h*cap+j]==source_entity)at=j;'
        'if(at<0){if(n==cap){d_fail(c,2);return;}at=n;c->s->character_reductions_count[h]=(short)(n+1);}'
        'c->s->character_reductions_presence[h]=2;'
        'c->s->character_reductions_entity[h*cap+at]=source_entity;'
        'c->s->character_reductions_value[h*cap+at]=layers;}\n'
        'DFN void d_clear_side_shields(DX* c,int side){'
        'if(!d_valid_side(c,side)||d_finished(c))return;if(d_pi(c,side,"shield")>0)d_event(c);d_set_pi(c,side,"shield",0);'
        'int n=d_order_count(c,side),i,slot;for(i=0;i<n;++i){slot=d_order_slot(c,side,i);if(d_ci(c,side,slot,"shield")>0)d_event(c);d_set_ci(c,side,slot,"shield",0);}}\n'
    )


def _lookup_source():
    attr_is = ''.join('if(d_cstr_is(c,side,actor,"attribute",%s))return 1;' % _lit(n) for n in _LIGHT)
    attr_dark = ''.join('if(d_cstr_is(c,side,actor,"attribute",%s))return 1;' % _lit(n) for n in _DARK)
    attr_of = ''.join('if(d_cstr_is(c,side,actor,"attribute",%s))return %s;' % (_lit(n), _lit(n)) for n in _ATTR)
    attr_code = ''.join('if(code==%d)return %s;' % (i + 1, _lit(n)) for i, n in enumerate(_ATTR))
    deferred = ''.join('if(d_eq(c,ev->deferred_kind_slot,"%s"))return "%s";' % (n, n) for n in _DEFERRED)
    owner_eq = ''.join('if(d_eq(c,slot,"%s"))return %d;' % item for item in schema.CHARACTER_VOCAB.items())
    return (
        'DFN int d_zone_count_hand(DX* c,int side){return d_valid_side(c,side)?c->s->zone_count[side*4]:0;}\n'
        'DFN int d_zone_card_at(DX* c,int side,int zone,int index){'
        'if(!d_valid_side(c,side)||zone<0||zone>3){d_fail(c,1);return -1;}'
        'int n=c->s->zone_count[side*4+zone];if(index<0||index>=n||n>Z){d_fail(c,1);return -1;}'
        'return c->s->zone_card_index[(side*4+zone)*Z+index];}\n'
        'DFN int d_attacked_this_turn(DX* c,int actor){'
        'if(!d_valid_side(c,c->side)||actor<0||actor>=C)return 0;'
        'if(c->s->player_attacked_this_turn_presence[c->side]!=2)return 0;'
        'return c->s->player_attacked_this_turn[c->side*C+actor]!=0;}\n'
        'DFN int d_teammate_attacked(DX* c,int side,int actor){'
        'if(!d_valid_side(c,side)||c->s->player_attacked_this_turn_presence[side]!=2)return 0;'
        'int n=d_order_count(c,side),i,slot;for(i=0;i<n;++i){slot=d_order_slot(c,side,i);'
        'if(slot!=actor&&c->s->player_attacked_this_turn[side*C+slot])return 1;}return 0;}\n'
        'DFN int d_used_instant(DX* c,int side){return d_used(c,side,"instant");}\n'
        'DFN void d_set_used_instant(DX* c,int side,int value){d_set_used(c,side,"instant",value,1);}\n'
        'DFN int d_passives_disabled(DX* c){(void)c;return 0;}\n'
        'DFN int d_response_intercept(DX* c){return c->s->response_intercept_presence[0]==2&&c->s->response_intercept[0]!=0;}\n'
        'DFN int d_op_option_is(DX* c,const char* option_id){'
        'if(!option_id||c->s->op_option_id_presence[0]!=2)return 0;return d_eq(c,c->s->op_option_id_slot[0],option_id);}\n'
        'DFN int d_attribute_is_light_family(DX* c,int side,int actor){' + attr_is + 'return 0;}\n'
        'DFN int d_attribute_is_dark_family(DX* c,int side,int actor){' + attr_dark + 'return 0;}\n'
        'DFN const char* d_attribute_of(DX* c,int side,int actor){' + attr_of + 'return "";}\n'
        'DFN const char* d_deferred_kind(DX* c,const DEvt* ev){'
        'if(!ev||ev->deferred_kind_slot<0)return "";' + deferred + 'return "";}\n'
        'DFN const char* d_deferred_attribute(DX* c,const DEvt* ev){(void)c;if(!ev)return "";int code=ev->deferred_attribute;' + attr_code + 'return "";}\n'
        'DFN int d_choice_selected_card(DX* c){return c->choice_selected;}\n'
        'DFN const int* d_choice_bottoms(DX* c,int* n){if(n)*n=c->choice_bottom_count;return c->choice_bottom;}\n'
        'DFN int d_printed_attack(DX* c,int side,int actor){'
        'if(!d_valid_char(c,side,actor))return 0;int h=side*C+actor;'
        'int id=d_template(c,side,actor);return id?cattack[id]:d_ci(c,side,actor,"base_attack");}\n'
        'DFN int d_card_owner_template(DX* c,int card){'
        'int id=d_card_template(c,card);if(d_bad(c)||!id)return 0;'
        'if(card>=0&&card<K&&c->s->card_str_presence[card*KS+KS_character_id]==2){'
        'int slot=c->s->card_str_slot[card*KS+KS_character_id],side,j;'
        'for(side=0;side<2;++side){int n=c->s->player_character_count[side];'
        'for(j=0;j<n;++j){int err=0;if(qslots(c->s,slot,c->s->character_id_slot[side*C+j],&err))return d_template(c,side,j);if(err)d_fail(c,err);}}'
        + owner_eq + 'return 0;}'
        'return d_character_template_named(d_card_default_str(id,KS_character_id));}\n'
    )


def _face_switch():
    cards = {i: lookup_card(cid) for cid, i in schema.CARD_VOCAB.items()}
    out = []
    for i, card in cards.items():
        body = []
        for typ, keys, dim in (('int', schema.CARD_INT_FIELDS, 'KI'), ('bool', schema.CARD_BOOL_FIELDS, 'KB'), ('str', schema.CARD_STR_FIELDS, 'KS')):
            for j, key in enumerate(keys):
                if key in card:
                    body.append('f->card_%s_presence[ref*%s+%d]=%s;' % (typ, dim, j, '1' if card[key] is None else '3'))
        for key, field in (('source', 'card_source_presence'), ('play_options', 'card_play_options_presence')):
            if key in card:
                body.append('f->%s[ref]=3;' % field)
        out.append('case %d:%sbreak;' % (i, ''.join(body)))
    return ''.join(out)


def _rewrite_source():
    ki = ''.join('if(d_streq(key,"%s"))k=%d;' % (name, i) for i, name in enumerate(schema.CARD_INT_FIELDS))
    return (
        'DFN int d_alloc_card_row(DX* c){int j;for(j=0;j<K;++j)if(!c->s->card_valid[j])return j;d_fail(c,2);return -1;}\n'
        'DFN void d_zero_card_row(DX* c,int ref){DS* f=c->s;int j;'
        'for(j=0;j<KI;++j){f->card_int_presence[ref*KI+j]=0;f->card_int_value[ref*KI+j]=0;}'
        'for(j=0;j<KB;++j){f->card_bool_presence[ref*KB+j]=0;f->card_bool_value[ref*KB+j]=0;}'
        'for(j=0;j<KS;++j){f->card_str_presence[ref*KS+j]=0;f->card_str_slot[ref*KS+j]=0;}'
        'f->card_source_presence[ref]=0;f->card_play_options_presence[ref]=0;'
        'f->card_jingu_mark_presence[ref]=0;f->card_jingu_delta[ref]=0;f->card_jingu_source_slot[ref]=0;}\n'
        'DFN void d_apply_catalog_face(DX* c,int ref,int id){DS* f=c->s;int j;'
        'for(j=0;j<KI;++j){f->card_int_presence[ref*KI+j]=0;f->card_int_value[ref*KI+j]=0;}'
        'for(j=0;j<KB;++j){if(j==KB_copy)continue;f->card_bool_presence[ref*KB+j]=0;f->card_bool_value[ref*KB+j]=0;}'
        'for(j=0;j<KS;++j){if(j==KS_character_id||j==KS_original_character_id)continue;f->card_str_presence[ref*KS+j]=0;f->card_str_slot[ref*KS+j]=0;}'
        'f->card_template[ref]=(short)id;switch(id){' + _face_switch() + 'default:d_fail(c,1);return;}}\n'
        'DFN void d_copy_card_identity(DX* c,int dst,int src){DS* f=c->s;'
        'f->card_entity_slot[dst]=f->card_entity_slot[src];'
        'f->card_instance_slot[dst]=f->card_instance_slot[src];'
        'f->card_side[dst]=f->card_side[src];'
        'f->card_jingu_mark_presence[dst]=f->card_jingu_mark_presence[src];'
        'f->card_jingu_delta[dst]=f->card_jingu_delta[src];'
        'f->card_jingu_source_slot[dst]=f->card_jingu_source_slot[src];'
        'f->card_bool_presence[dst*KB+KB_copy]=f->card_bool_presence[src*KB+KB_copy];'
        'f->card_bool_value[dst*KB+KB_copy]=f->card_bool_value[src*KB+KB_copy];'
        'f->card_int_presence[dst*KI+KI_expires_turn]=f->card_int_presence[src*KI+KI_expires_turn];'
        'f->card_int_value[dst*KI+KI_expires_turn]=f->card_int_value[src*KI+KI_expires_turn];'
        'f->card_str_presence[dst*KS+KS_original_character_id]=f->card_str_presence[src*KS+KS_original_character_id];'
        'f->card_str_slot[dst*KS+KS_original_character_id]=f->card_str_slot[src*KS+KS_original_character_id];'
        'f->card_str_presence[dst*KS+KS_character_id]=f->card_str_presence[src*KS+KS_character_id];'
        'f->card_str_slot[dst*KS+KS_character_id]=f->card_str_slot[src*KS+KS_character_id];}\n'
        'DFN void d_retarget_hand(DX* c,int side,int from_card,int to_card){'
        'int n=c->s->zone_count[side*4],i;for(i=0;i<n;++i)if(c->s->zone_card_index[side*4*Z+i]==from_card)c->s->zone_card_index[side*4*Z+i]=to_card;}\n'
        'DFN int d_ensure_hand_view(DX* c,int card){'
        'int physical=d_physical_card(c,card);if(physical<0||d_bad(c))return -1;'
        'if(card>=0&&c->s->card_is_view[card]){c->card=card;return card;}'
        'int view=d_alloc_card_row(c);if(view<0)return -1;d_zero_card_row(c,view);'
        'c->s->card_valid[view]=1;c->s->card_is_view[view]=1;c->s->card_physical_index[view]=physical;'
        'd_copy_card_identity(c,view,physical);d_apply_catalog_face(c,view,c->s->card_template[physical]);'
        'int side=c->s->card_side[physical]>=0?c->s->card_side[physical]:c->side;'
        'c->card=view;return view;}\n'
        'DFN void d_rewrite_hand_as(DX* c,int card,const char* card_id){'
        'if(d_card_is_unknown(c,card))return;int id=d_card_id(card_id);if(!id){d_fail(c,1);return;}'
        'int view=d_ensure_hand_view(c,card);if(view<0||d_bad(c))return;'
        'int physical=c->s->card_physical_index[view];d_copy_card_identity(c,view,physical);d_apply_catalog_face(c,view,id);}\n'
        'DFN void d_rewrite_k03_recovery(DX* c,int card){'
        'if(d_card_is_unknown(c,card))return;int view=d_ensure_hand_view(c,card);if(view<0||d_bad(c))return;'
        'int physical=c->s->card_physical_index[view];'
        'int name_p=c->s->card_str_presence[physical*KS+KS_name];int name_s=c->s->card_str_slot[physical*KS+KS_name];'
        'd_copy_card_identity(c,view,physical);d_apply_catalog_face(c,view,d_card_id("K03"));'
        'if(name_p==2){c->s->card_str_presence[view*KS+KS_name]=2;c->s->card_str_slot[view*KS+KS_name]=name_s;}'
        'else if(name_p==3){c->s->card_str_presence[view*KS+KS_name]=2;c->s->card_str_slot[view*KS+KS_name]=d_intern(c,d_card_default_str(d_card_template(c,physical),KS_name));}'
        'c->s->card_int_presence[view*KI+KI_cost]=2;c->s->card_int_value[view*KI+KI_cost]=1;'
        'c->s->card_bool_presence[view*KB+KB_terminal]=2;c->s->card_bool_value[view*KB+KB_terminal]=0;'
        'c->s->card_bool_presence[view*KB+KB_instant]=2;c->s->card_bool_value[view*KB+KB_instant]=1;'
        'c->s->card_bool_presence[view*KB+KB__k03_recovery]=2;c->s->card_bool_value[view*KB+KB__k03_recovery]=1;'
        'c->s->card_str_presence[view*KS+KS_type]=2;c->s->card_str_slot[view*KS+KS_type]=d_intern(c,"tactic");'
        'c->s->card_str_presence[view*KS+KS_effect_id]=2;c->s->card_str_slot[view*KS+KS_effect_id]=d_intern(c,"K03");}\n'
        'DFN void d_set_card_int(DX* c,int card,const char* key,int value){'
        'if(card<=-2||card<0||card>=K||!c->s->card_valid[card]){d_fail(c,1);return;}'
        'int k=-1;' + ki + 'if(k<0){d_fail(c,4);return;}'
        'c->s->card_int_presence[card*KI+k]=2;c->s->card_int_value[card*KI+k]=value;}\n'
    )


def _zone_source():
    return (
        'DFN void d_detach_card(DX*c,int side,int card){if(card<0||card>=K){d_fail(c,1);return;}int entity=c->s->card_entity_slot[card];for(int zone=0;zone<3;zone++){int j=0;while(j<c->s->zone_count[side*4+zone]){int ref=c->s->zone_card_index[(side*4+zone)*Z+j];if(c->s->card_entity_slot[ref]==entity)d_remove_zone_card(c,side,zone,ref);else j++;}}}\n'
        'DFN void d_draw_named_from_zones(DX* c,int side,const char* card_id,int reveal){'
        'if(d_finished(c)||!d_valid_side(c,side))return;int zone,i;'
        'for(zone=1;zone<3;++zone){int n=c->s->zone_count[side*4+zone];'
        'for(i=0;i<n;++i){int card=c->s->zone_card_index[(side*4+zone)*Z+i];'
        'if(!d_card_is(c,card,card_id)||d_card_bool(c,card,"copy"))continue;d_remove_zone_card(c,side,zone,card);d_add_hand(c,side,card,reveal);return;}}}\n'
        'DFN void d_advance_owned_to_top(DX* c,int side,int actor){'
        'int n=d_deck_count(c,side),i,hits=0,owned[128];if(n>128){d_fail(c,2);return;}'
        'for(i=0;i<n;++i){int card=d_zone_card_at(c,side,1,i);if(d_card_owner(c,card)==actor)owned[hits++]=i;}'
        'if(!hits)return;int pick=owned[d_random_before(c,hits)];int card=d_zone_card_at(c,side,1,pick);'
        'd_remove_zone_card(c,side,1,card);d_insert_deck(c,side,card,0);d_event(c);}\n'
        'DFN void d_inspect_top(DX* c,int side,int count){'
        'if(d_finished(c)||d_bad(c)||!d_valid_side(c,side)||count<=0)return;'
        'int n=d_deck_count(c,side),take=count<n?count:n,i;if(take<=0)return;'
        'if(take==1){int card=d_zone_card_at(c,side,1,0);d_remove_zone_card(c,side,1,card);d_add_hand(c,side,card,0);return;}'
        'if(take>CH){d_fail(c,2);return;}c->s->choice_card_count[0]=(short)take;'
        'for(i=0;i<take;++i){int card=d_zone_card_at(c,side,1,0);d_remove_zone_card(c,side,1,card);c->s->choice_card_index[i]=card;}'
        'c->s->pending_choice_present[0]=1;d_choice(c,"inspect_top",take);}\n'
        'DFN void d_draw_matching(DX* c,int side,int count,const char* type,int reveal){'
        '(void)reveal;int code=-1;if(d_streq(type,"tactic"))code=0;else if(d_streq(type,"battle"))code=1;else if(d_streq(type,"form"))code=2;else{d_fail(c,1);return;}'
        'd_draw(c,side,count,code);}\n'
        'DFN int d_redraw_own_hand(DX* c,int side,int actor){'
        'int n=d_zone_count_hand(c,side),i,removed=0;'
        'for(i=n-1;i>=0;--i){int card=d_zone_card_at(c,side,0,i);if(d_card_owner(c,card)!=actor)continue;'
        'd_remove_zone_card(c,side,0,card);d_discard(c,side,card,0);++removed;}return removed;}\n'
    )


def _extra_source(layout):
    from math import prod
    from .domain_soa_execution_base import _TYPES
    from .domain_soa_kit_source import PUBLIC_KIT_ORDER
    specs=schema.array_specs(layout)
    text='DFN void d_copy_card_values(DX*c,int dst,int src){DS*f=c->s;'
    for name,(dtype,shape)in specs.items():
        if name.startswith('card_'):
            width=prod(shape[1:]);text+=f'for(int j=0;j<{width};j++)f->{name}[dst*{width}+j]=f->{name}[src*{width}+j];'
    text+='}\nDFN void d_restore_heiyu_spec(DX*c,int card){if(d_card_is_unknown(c,card))return;int id=d_card_template(c,card),dst=d_alloc_card_row(c);if(dst<0)return;d_copy_card_values(c,dst,card);DS*f=c->s;c->card=dst;'
    for key in ('instant',):text+=f'f->card_bool_presence[dst*KB+KB_{key}]=0;f->card_bool_value[dst*KB+KB_{key}]=0;'
    for key in ('attack','shield','hp'):text+=f'f->card_int_presence[dst*KI+KI_{key}]=0;f->card_int_value[dst*KI+KI_{key}]=0;'
    text+='switch(id){'
    for cid,i in schema.CARD_VOCAB.items():
        card=lookup_card(cid);text+=f'case {i}:'
        for key in ('name','type','description','cost','effect_id','instant','attack','shield','hp'):
            if key not in card:continue
            typ='str'if key in schema.CARD_STR_FIELDS else'bool'if key in schema.CARD_BOOL_FIELDS else'int'
            dim={'str':'KS','bool':'KB','int':'KI'}[typ];idx=getattr(schema,'CARD_'+typ.upper()+'_FIELDS').index(key)
            text+=f'f->card_{typ}_presence[dst*{dim}+{idx}]=3;'
        text+='break;'
    text+='}}\n'
    text+='DFN int d_effective_card(DX*c,int side,int original){if(d_card_is_unknown(c,original))return original;DX own=*c;own.side=side;own.card=original;'
    for cid in PUBLIC_KIT_ORDER:
        text+='{int actor=d_hero(&own,side,"'+cid+'");if(actor>=0&&d_alive(&own,side,actor)){own.actor=actor;int kit=d_mimic_template(&own,side,actor);int wrapped=kit>0;if(!kit)kit=d_template(&own,side,actor);int awoke=d_cb(&own,side,actor,"awakened");if(wrapped)d_set_cb(&own,side,actor,"awakened",0);'
        text+='if(kit==ID_zhenhong&&d_cb(&own,side,actor,"awakened"))d_rewrite_hand_as(&own,own.card,"RF01");'
        text+='else if(kit==ID_lingke&&d_cstr_is(&own,side,actor,"shape","K03")&&d_card_owner(&own,own.card)==actor&&d_pi(&own,side,"ap")==0&&!d_used(&own,side,"instant"))d_rewrite_k03_recovery(&own,own.card);'
        text+='else if(kit==ID_heiyu&&d_card_owner(&own,own.card)==actor){int id=d_card_template(&own,own.card);const char*owner=d_card_default_str(id,KS_character_id);if(owner&&d_eq(&own,own.s->character_id_slot[side*C+actor],owner))d_restore_heiyu_spec(&own,own.card);}'
        text+='if(wrapped)d_set_cb(&own,side,actor,"awakened",awoke);}}'
    text+='return own.card;}\n'
    return text
