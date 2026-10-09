"""Seven-runtime CUDA intern layout and extra public-field projection.

Recovery intern names/indices stay frozen in gpu_rule_projection. This module
only compiles when GraphProjectionKernel(runtime='seven') is constructed.
Seats, cards and action types are interned as contiguous dedicated ranges so
unknown roster/action codes cannot alias historical field names.
"""
from __future__ import annotations

from app.modules.card_game.rl.batched_duel.gpu_schema_contract import runtime_contract
from app.modules.card_game.rl.batched_duel.gpu_seven_candidates import (
    MAX_MULLIGAN_SELECTIONS,
    ORDER_SLOTS,
)
from app.modules.card_game.rl.batched_duel.gpu_seven_observation import (
    ATTRIBUTES,
    CHOICES,
    MAX_HAND_CARDS,
    MAX_PRESENT,
    MAX_TIMED,
    MAX_ZONE_CARDS,
    NEW_CHAR,
    NEW_SIDE,
)
from app.modules.card_game.rl.seven_lineup import ACTION_TYPES

SEVEN_EXTRA_FIELDS = (
    'insight_damage', 'insight_played', 'support_count', 'heiyu_divinations',
    'collapse_max', 'lingke_chain', 'insight_used_turn', 'mimic', 'resistances',
    'kit', 'kit_multi', 'heiyu_divination', 'hand_deck_placement',
    'additional_ap_cost', 'bottom_card_ids', 'bottom_card_id',
    'choose_deck_order', 'exit_ultimate', 'choose_cards', 'count',
) + tuple(ATTRIBUTES)


def intern_layout_seven():
    from .gpu_rule_projection import FIELD_NAMES, OPTION_IDS_TUPLE
    contract = runtime_contract('seven')
    names = (
        list(contract['seats'])
        + list(contract['card_ids'])
        + list(OPTION_IDS_TUPLE)
        + list(contract['action_types'])
        + list(FIELD_NAMES)
        + list(SEVEN_EXTRA_FIELDS)
    )
    unique = []
    index = {}
    for name in names:
        if name not in index:
            index[name] = len(unique)
            unique.append(name)
    seats = contract['seats']
    cards = contract['card_ids']
    actions = contract['action_types']
    seat_start = index[seats[0]]
    for offset, seat in enumerate(seats):
        if index[seat] != seat_start + offset:
            raise RuntimeError('Seven intern seats are not a contiguous dedicated range')
    card_start = index[cards[0]]
    for offset, card in enumerate(cards):
        if index[card] != card_start + offset:
            raise RuntimeError('Seven intern cards are not a contiguous dedicated range')
    action_start = index[actions[0]]
    for offset, action in enumerate(actions):
        if index[action] != action_start + offset:
            raise RuntimeError('Seven intern action types are not a contiguous dedicated range')
    return unique, index


def _ident(name):
    return 'F_' + ''.join(ch if ch.isalnum() else '_' for ch in name)


def _ascii_ident(name):
    if not name or any(ord(ch) > 127 for ch in name):
        raise ValueError('ASCII intern identifier required: %r' % (name,))
    return _ident(name)


def _replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError('Seven projection splice %s matched %s time(s)' % (label, count))
    return source.replace(old, new, 1)


def emit_seven_projection_source():
    from .gpu_rule_projection import FIELD_NAMES, OPTION_IDS_TUPLE, _PROJECTION_SOURCE
    from .vm_heap import emit_vm_heap_source
    names, index = intern_layout_seven()
    contract = runtime_contract('seven')
    seats = contract['seats']
    cards = contract['card_ids']
    actions = contract['action_types']
    constants = []
    for name in FIELD_NAMES:
        constants.append('const int %s=%d;' % (_ascii_ident(name), index[name]))
    for name in SEVEN_EXTRA_FIELDS:
        if any(ord(ch) > 127 for ch in name):
            continue
        constants.append('const int %s=%d;' % (_ascii_ident(name), index[name]))
    for attr_i, attr in enumerate(ATTRIBUTES):
        constants.append('const int F_ATTR_%d=%d;' % (attr_i, index[attr]))
    constants.append('const int SEAT_START=%d,N_SEATS=%d,CARD_START=%d,N_CARDS=%d;' % (
        index[seats[0]], len(seats), index[cards[0]], len(cards),
    ))
    constants.append('const int OPTION_START=%d,N_OPTIONS=%d,ACTION_START=%d,N_ACTIONS=%d,SUMMON_SEAT=%d;' % (
        index[OPTION_IDS_TUPLE[0]], len(OPTION_IDS_TUPLE), index[actions[0]], len(actions), list(seats).index('summon'),
    ))
    constants.append('const int MAX_PRESENT=%d,MAX_TIMED=%d,MAX_ZONE_CARDS=%d,MAX_HAND_CARDS=%d,MAX_MULLIGAN=%d;' % (
        MAX_PRESENT, MAX_TIMED, MAX_ZONE_CARDS, MAX_HAND_CARDS, MAX_MULLIGAN_SELECTIONS,
    ))
    constants.append('const int CODE_END=%d,CODE_ATK=%d,CODE_ULT=%d,CODE_PLAY=%d,CODE_CHOOSE=%d,CODE_MULL=%d,CODE_ORDER=%d,CODE_EXIT=%d,CODE_CARDS=%d;' % (
        ACTION_TYPES.index('end_turn'), ACTION_TYPES.index('attack'), ACTION_TYPES.index('ultimate'),
        ACTION_TYPES.index('play_card'), ACTION_TYPES.index('choose'), ACTION_TYPES.index('mulligan'),
        ACTION_TYPES.index('choose_deck_order'), ACTION_TYPES.index('exit_ultimate'), ACTION_TYPES.index('choose_cards'),
    ))
    constants.append('const int N_NEW_SIDE=%d,N_NEW_CHAR=%d,N_ATTR=%d,N_CHOICE=%d,ORDER_SLOTS=%d;' % (
        len(NEW_SIDE), len(NEW_CHAR), len(ATTRIBUTES), len(CHOICES), ORDER_SLOTS,
    ))
    constants.append('const int NEW_SIDE_IDS[%d]={%s};' % (
        len(NEW_SIDE), ','.join(_ascii_ident(name) for name in NEW_SIDE),
    ))
    constants.append('const int NEW_SIDE_FROM_EXTRA[%d]={%s};' % (
        len(NEW_SIDE), ','.join('1' if name == 'heiyu_divinations' else '0' for name in NEW_SIDE),
    ))
    constants.append('const int ATTR_IDS[%d]={%s};' % (
        len(ATTRIBUTES), ','.join('F_ATTR_%d' % i for i in range(len(ATTRIBUTES))),
    ))
    constants.append('const int CHOICE_IDS[%d]={%s};' % (
        len(CHOICES), ','.join(_ascii_ident(name) for name in CHOICES),
    ))
    extra_obs = r'''
'''.lstrip('\n') + r'''
__device__ inline int fill_seven_observation(
 Heap* h,int view,int row,const unsigned char* names,const int* off,const int* len,
 float* seven_side,float* seven_char,float* side_resist,float* char_resist,
 float* choice_kind,int* choice_cards,int* status){
 int viewer=dict_get_intern(h,view,names,off,len,F_viewer_side);
 int viewer_b=intern_eq(h,viewer,names,off,len,F_b);
 int sides=dict_get_intern(h,view,names,off,len,F_sides);
 int publicv=dict_get_intern(h,view,names,off,len,F_policy_state);
 int public_sides=dict_get_intern(h,publicv,names,off,len,F_sides);
 int side_keys[2]={viewer_b?F_b:F_a, viewer_b?F_a:F_b};
 double turn=0; as_double(h,dict_get_intern(h,view,names,off,len,F_turn),&turn);
 for(int s=0;s<2;s++){
  int team=dict_get_intern(h,sides,names,off,len,side_keys[s]);
  int extra=dict_get_intern(h,public_sides,names,off,len,side_keys[s]);
  for(int k=0;k<N_NEW_SIDE;k++){
   int src=NEW_SIDE_FROM_EXTRA[k]?extra:team;
   double v=0; as_double(h,dict_get_intern(h,src,names,off,len,NEW_SIDE_IDS[k]),&v);
   seven_side[(row*2+s)*N_NEW_SIDE+k]=(float)v;
  }
  int resists=dict_get_intern(h,extra,names,off,len,F_resistances);
  for(int a=0;a<N_ATTR;a++){
   double v=0; as_double(h,dict_get_intern(h,resists,names,off,len,ATTR_IDS[a]),&v);
   side_resist[(row*2+s)*N_ATTR+a]=(float)v;
  }
  int chars=dict_get_intern(h,team,names,off,len,F_characters);
  int nchars=list_count(h,chars); if(nchars<0) nchars=0;
  int meta_chars=dict_get_intern(h,extra,names,off,len,F_characters);
  for(int seat=0;seat<N_SEATS;seat++){
   int base=((row*2+s)*N_SEATS+seat);
   seven_char[base*5+0]=0; seven_char[base*5+1]=0; seven_char[base*5+2]=-1; seven_char[base*5+3]=0; seven_char[base*5+4]=-1;
   for(int a=0;a<N_ATTR;a++) char_resist[base*N_ATTR+a]=0;
   int hero=NOT_FOUND;
   for(int i=0;i<nchars;i++){
    int cand=list_item(h,chars,i);
    if(intern_eq(h,dict_get_intern(h,cand,names,off,len,F_id),names,off,len,SEAT_START+seat)){ hero=cand; break; }
   }
   if(hero==NOT_FOUND) continue;
   double cm=0,cc=0; as_double(h,dict_get_intern(h,hero,names,off,len,F_collapse_max),&cm);
   int chain=dict_get_intern(h,hero,names,off,len,F_lingke_chain);
   as_double(h,dict_get_intern(h,chain,names,off,len,F_count),&cc);
   int chain_seat=-1;
   int chain_id=dict_get_intern(h,chain,names,off,len,F_character_id);
   if(chain_id!=NOT_FOUND && !is_none(h,chain_id)){
    chain_seat=seat_lookup(h,chain_id,names,off,len);
    if(chain_seat==-2){ *status=P_ROSTER; return 0; }
   }
   int hid=dict_get_intern(h,hero,names,off,len,F_id);
   int meta=dict_get_key(h,meta_chars,hid);
   double used=0; as_double(h,dict_get_intern(h,meta,names,off,len,F_insight_used_turn),&used);
   int flags=dict_get_intern(h,meta,names,off,len,F_flags);
   int mimic=dict_get_intern(h,flags,names,off,len,F_mimic);
   int mimic_seat=-1;
   int mimic_id=dict_get_intern(h,mimic,names,off,len,F_id);
   if(mimic_id!=NOT_FOUND && !is_none(h,mimic_id)){
    mimic_seat=seat_lookup(h,mimic_id,names,off,len);
    if(mimic_seat==-2){ *status=P_ROSTER; return 0; }
   }
   seven_char[base*5+0]=(float)cm; seven_char[base*5+1]=(float)cc;
   seven_char[base*5+2]=(chain_seat>=0)?(float)chain_seat:-1.f;
   seven_char[base*5+3]=(used==turn)?1.f:0.f;
   seven_char[base*5+4]=(mimic_seat>=0)?(float)mimic_seat:-1.f;
   int cres=dict_get_intern(h,meta,names,off,len,F_resistances);
   for(int a=0;a<N_ATTR;a++){
    double v=0; as_double(h,dict_get_intern(h,cres,names,off,len,ATTR_IDS[a]),&v);
    char_resist[base*N_ATTR+a]=(float)v;
   }
  }
 }
 int pending=dict_get_intern(h,view,names,off,len,F_pending_choice);
 int kind=dict_get_intern(h,pending,names,off,len,F_kind);
 if(nonempty(h,kind)){
  int matched=0;
  for(int k=0;k<N_CHOICE;k++){
   int hit=intern_eq(h,kind,names,off,len,CHOICE_IDS[k]);
   choice_kind[row*N_CHOICE+k]=hit?1.f:0.f;
   if(hit) matched=1;
  }
  if(!matched){ *status=P_VIEW; return 0; }
 }
 int choices=dict_get_intern(h,pending,names,off,len,F_choices);
 int nc=list_count(h,choices); if(nc<0) nc=0;
 if(nc>MAX_ZONE_CARDS){ *status=P_CAP; return 0; }
 for(int i=0;i<nc;i++){
  int entry=list_item(h,choices,i);
  int card=dict_get_intern(h,entry,names,off,len,F_card_id);
  if(card==NOT_FOUND){
   NodeView en; if(get_node(h,entry,&en)&&en.tag==TAG_DICT){
    for(int k=0;k<en.count;k++){
     int key=0,val=0; get_edge(h,&en,k,&key,&val);
     NodeView kn; if(get_node(h,key,&kn)&&kn.tag==TAG_STR&&kn.count==4){
      const unsigned char* p=get_payload_ptr(h,&kn);
      if(p && vm_memcmp(p,(const unsigned char*)"card",4)==0){ card=val; break; }
     }
    }
   }
  }
  if(card==NOT_FOUND||is_none(h,card)) continue;
  int cid=dict_get_intern(h,card,names,off,len,F_card_id);
  int idx=card_lookup(h,cid==NOT_FOUND?card:cid,names,off,len);
  if(idx<0){ *status=P_CARD; return 0; }
  choice_cards[row*MAX_ZONE_CARDS+i]=idx;
 }
 return 1;
}
__device__ inline int lookup_choice_card(Heap* h,int choices,int inst,const unsigned char* names,const int* off,const int* len,int* status){
 int nc=list_count(h,choices); if(nc<0) nc=0;
 for(int c=0;c<nc;c++){
  int entry=list_item(h,choices,c);
  int eid=dict_get_intern(h,entry,names,off,len,F_id);
  if(eid==NOT_FOUND||!key_equal(h,eid,inst)) continue;
  int card=dict_get_intern(h,entry,names,off,len,F_card_id);
  if(card==NOT_FOUND){
   NodeView en; if(get_node(h,entry,&en)&&en.tag==TAG_DICT){
    for(int k=0;k<en.count;k++){
     int key=0,val=0; get_edge(h,&en,k,&key,&val);
     NodeView kn; if(get_node(h,key,&kn)&&kn.tag==TAG_STR&&kn.count==4){
      const unsigned char* p=get_payload_ptr(h,&kn);
      if(p && vm_memcmp(p,(const unsigned char*)"card",4)==0){ card=val; break; }
     }
    }
   }
  }
  if(card==NOT_FOUND||is_none(h,card)){ *status=P_OBJ; return -1; }
  int cid=dict_get_intern(h,card,names,off,len,F_card_id);
  int idx=card_lookup(h,cid==NOT_FOUND?card:cid,names,off,len);
  if(idx<0){ *status=P_CARD; return -1; }
  return idx;
 }
*status=P_OBJ; return -1;
}
'''
    source = _PROJECTION_SOURCE
    source = _replace_once(
        source,
        '__device__ inline int fill_observation(',
        extra_obs + '__device__ inline int fill_observation(',
        'seven helper insertion',
    )
    source = _replace_once(
        source,
        '__device__ inline int action_type_code(Heap* h,int ref,const unsigned char* names,const int* off,const int* len){\n'
        ' if(intern_eq(h,ref,names,off,len,F_concede)) return -2;\n'
        ' for(int i=0;i<N_ACTIONS;i++) if(intern_eq(h,ref,names,off,len,ACTION_START+i)) return i;\n'
        ' return -1;\n'
        '}',
        '__device__ inline int action_type_code(Heap* h,int ref,const unsigned char* names,const int* off,const int* len){\n'
        ' if(intern_eq(h,ref,names,off,len,F_concede)) return -2;\n'
        ' if(intern_eq(h,ref,names,off,len,F_end_turn)) return CODE_END;\n'
        ' if(intern_eq(h,ref,names,off,len,F_attack)) return CODE_ATK;\n'
        ' if(intern_eq(h,ref,names,off,len,F_ultimate)) return CODE_ULT;\n'
        ' if(intern_eq(h,ref,names,off,len,F_play_card)) return CODE_PLAY;\n'
        ' if(intern_eq(h,ref,names,off,len,F_choose)) return CODE_CHOOSE;\n'
        ' if(intern_eq(h,ref,names,off,len,F_mulligan)) return CODE_MULL;\n'
        ' if(intern_eq(h,ref,names,off,len,F_choose_deck_order)) return CODE_ORDER;\n'
        ' if(intern_eq(h,ref,names,off,len,F_exit_ultimate)) return CODE_EXIT;\n'
        ' if(intern_eq(h,ref,names,off,len,F_choose_cards)) return CODE_CARDS;\n'
        ' return -1;\n'
        '}',
        'action_type_code',
    )
    source = _replace_once(source, 'if(code==2){', 'if(code==CODE_PLAY){', 'play_card code')
    source = _replace_once(source, '}else if(code==5){', '}else if(code==CODE_CHOOSE){', 'choose code')
    source = _replace_once(source, '}else if(code==1||code==3){', '}else if(code==CODE_ATK||code==CODE_ULT){', 'attack/ultimate code')
    source = _replace_once(source, '}else if(code==4){', '}else if(code==CODE_MULL){', 'mulligan code')
    extra_branch = (
        'else if(code==CODE_EXIT){\n'
        '   int ch=dict_get_intern(h,action,names,off,len,F_character_id);\n'
        '   actor_idx=seat_lookup(h,ch,names,off,len);\n'
        '   if(actor_idx<0){ *status=P_ROSTER; return 0; }\n'
        '  }else if(code==CODE_ORDER){\n'
        '   int pending=dict_get_intern(h,view,names,off,len,F_pending_choice);\n'
        '   int choices=dict_get_intern(h,pending,names,off,len,F_choices);\n'
        '   int top=dict_get_intern(h,action,names,off,len,F_card_ids);\n'
        '   int bottom=dict_get_intern(h,action,names,off,len,F_bottom_card_ids);\n'
        '   if(bottom==NOT_FOUND||is_none(h,bottom)){\n'
        '    int one=dict_get_intern(h,action,names,off,len,F_bottom_card_id);\n'
        '    if(one!=NOT_FOUND && !is_none(h,one)){\n'
        '     int idx=lookup_choice_card(h,choices,one,names,off,len,status);\n'
        '     if(idx<0) return 0;\n'
        '     order_cards[((row*max_candidates+kept)*2+1)*ORDER_SLOTS+0]=idx;\n'
        '    }\n'
        '   }\n'
        '   int zones[2]={top,bottom};\n'
        '   for(int z=0;z<2;z++){\n'
        '    int ids=zones[z]; if(ids==NOT_FOUND||is_none(h,ids)) continue;\n'
        '    int ns=list_count(h,ids); if(ns<0) ns=0;\n'
        '    if(ns>ORDER_SLOTS){ *status=P_CAP; return 0; }\n'
        '    for(int slot=0;slot<ns;slot++){\n'
        '     int inst=list_item(h,ids,slot);\n'
        '     int idx=lookup_choice_card(h,choices,inst,names,off,len,status);\n'
        '     if(idx<0) return 0;\n'
        '     order_cards[((row*max_candidates+kept)*2+z)*ORDER_SLOTS+slot]=idx;\n'
        '    }\n'
        '   }\n'
        '  }else if(code==CODE_CARDS){\n'
        '   int ids=dict_get_intern(h,action,names,off,len,F_card_ids);\n'
        '   int ns=list_count(h,ids); if(ns<0) ns=0;\n'
        '   if(ns>MAX_MULLIGAN){ *status=P_CAP; return 0; }\n'
        '   int pending=dict_get_intern(h,view,names,off,len,F_pending_choice);\n'
        '   int choices=dict_get_intern(h,pending,names,off,len,F_choices);\n'
        '   int nh=list_count(h,hand); if(nh<0) nh=0;\n'
        '   for(int sel=0;sel<ns;sel++){\n'
        '    int inst=list_item(h,ids,sel); int idx=-1;\n'
        '    int saved=*status; *status=P_OK;\n'
        '    idx=lookup_choice_card(h,choices,inst,names,off,len,status);\n'
        '    if(idx<0){\n'
        '     *status=saved; int item=NOT_FOUND;\n'
        '     for(int c=0;c<nh;c++){\n'
        '      int card=list_item(h,hand,c);\n'
        '      int iid=dict_get_intern(h,card,names,off,len,F_instance_id);\n'
        '      if(iid!=NOT_FOUND && key_equal(h,iid,inst)){ item=card; break; }\n'
        '     }\n'
        '     if(item==NOT_FOUND||is_none(h,item)){ *status=P_OBJ; return 0; }\n'
        '     int cid=dict_get_intern(h,item,names,off,len,F_card_id);\n'
        '     idx=card_lookup(h,cid==NOT_FOUND?item:cid,names,off,len);\n'
        '     if(idx<0){ *status=P_CARD; return 0; }\n'
        '    }\n'
        '    select_cards[(row*max_candidates+kept)*MAX_MULLIGAN+sel]=idx;\n'
        '   }\n'
        '  }else if(code!=CODE_END){\n'
        '   *status=P_ACTION; return 0;\n'
        '  }\n'
        '  if(code==CODE_PLAY){\n'
        '   int inst=dict_get_intern(h,action,names,off,len,F_card_id);\n'
        '   int item=NOT_FOUND; int nh=list_count(h,hand); if(nh<0) nh=0;\n'
        '   for(int c=0;c<nh;c++){\n'
        '    int card=list_item(h,hand,c);\n'
        '    int iid=dict_get_intern(h,card,names,off,len,F_instance_id);\n'
        '    if(iid!=NOT_FOUND && key_equal(h,iid,inst)){ item=card; break; }\n'
        '   }\n'
        '   double ap=0; as_double(h,dict_get_intern(h,item,names,off,len,F_additional_ap_cost),&ap);\n'
        '   extra_ap[row*max_candidates+kept]=(float)ap;\n'
        '  }\n'
        '  '
    )
    source = _replace_once(
        source,
        '  int_tokens[base+1]=actor_idx; int_tokens[base+2]=card_kind; int_tokens[base+3]=tside;',
        extra_branch + 'int_tokens[base+1]=actor_idx; int_tokens[base+2]=card_kind; int_tokens[base+3]=tside;',
        'extra action branches',
    )
    source = _replace_once(
        source,
        'item_flags[ibase]=0; item_flags[ibase+1]=0;',
        'item_flags[ibase]=0; item_flags[ibase+1]=0;\n'
        '  extra_ap[row*max_candidates+kept]=0.f;\n'
        '  for(int z=0;z<2;z++) for(int slot=0;slot<ORDER_SLOTS;slot++) order_cards[((row*max_candidates+kept)*2+z)*ORDER_SLOTS+slot]=-1;\n'
        '  for(int sel=0;sel<MAX_MULLIGAN;sel++) select_cards[(row*max_candidates+kept)*MAX_MULLIGAN+sel]=-1;',
        'extra candidate init',
    )
    source = _replace_once(
        source,
        ' Heap* h,int view,int actions,int row,int max_candidates,\n'
        ' const unsigned char* names,const int* off,const int* len,\n'
        ' int* int_tokens,float* float_tokens,float* item_flags,float* turn_counts,\n'
        ' int* mulligan_cards,int* cand_counts,int* cand_mask,int* action_root,int* status){',
        ' Heap* h,int view,int actions,int row,int max_candidates,\n'
        ' const unsigned char* names,const int* off,const int* len,\n'
        ' int* int_tokens,float* float_tokens,float* item_flags,float* turn_counts,\n'
        ' int* mulligan_cards,int* cand_counts,int* cand_mask,int* action_root,int* order_cards,int* select_cards,float* extra_ap,int* status){',
        'fill_candidates signature',
    )
    source = _replace_once(
        source,
        'if(!fill_candidates(&h,view,legal,row,max_candidates,names,off,len,int_tokens,float_tokens,item_flags,turn_counts,\n'
        '  mulligan_cards,cand_counts,cand_mask,action_roots,status+row)) return;',
        'if(!fill_candidates(&h,view,legal,row,max_candidates,names,off,len,int_tokens,float_tokens,item_flags,turn_counts,\n'
        '  mulligan_cards,cand_counts,cand_mask,action_roots,order_cards,select_cards,extra_ap,status+row)) return;\n'
        ' if(!fill_seven_observation(&h,view,row,names,off,len,seven_side,seven_char,side_resist,char_resist,choice_kind,choice_cards,status+row)) return;',
        'decision fill_candidates call',
    )
    source = _replace_once(
        source,
        'int* int_tokens,float* float_tokens,float* item_flags,float* turn_counts,\n'
        ' int* mulligan_cards,int* cand_counts,int* cand_mask,int* action_roots,int* status,\n'
        ' int rows,int maxn,int maxe,int maxp,int max_candidates){',
        'int* int_tokens,float* float_tokens,float* item_flags,float* turn_counts,\n'
        ' int* mulligan_cards,int* cand_counts,int* cand_mask,int* action_roots,\n'
        ' int* order_cards,int* select_cards,float* extra_ap,\n'
        ' float* seven_side,float* seven_char,float* side_resist,float* char_resist,float* choice_kind,int* choice_cards,int* status,\n'
        ' int rows,int maxn,int maxe,int maxp,int max_candidates){',
        'decision kernel signature',
    )
    source = _replace_once(
        source,
        'if(!fill_observation(&h,views[row],row,names,off,len,global_tokens,side_int,side_float,char_present,char_seats,\n'
        '  char_int,char_float,char_timed,zone_cards,mark_cards,mark_deltas,mark_valid,substituted_cards,substituted_valid,\n'
        '  obs_mask,obs_counts,status+row)) return;',
        'if(!fill_observation(&h,views[row],row,names,off,len,global_tokens,side_int,side_float,char_present,char_seats,\n'
        '  char_int,char_float,char_timed,zone_cards,mark_cards,mark_deltas,mark_valid,substituted_cards,substituted_valid,\n'
        '  obs_mask,obs_counts,status+row)) return;\n'
        ' if(!fill_seven_observation(&h,views[row],row,names,off,len,seven_side,seven_char,side_resist,char_resist,choice_kind,choice_cards,status+row)) return;',
        'observation fill_seven call',
    )
    source = _replace_once(
        source,
        'int* obs_mask,int* obs_counts,int* status,\n'
        ' int rows,int maxn,int maxe,int maxp){',
        'int* obs_mask,int* obs_counts,float* seven_side,float* seven_char,float* side_resist,float* char_resist,float* choice_kind,int* choice_cards,int* status,\n'
        ' int rows,int maxn,int maxe,int maxp){',
        'observation kernel signature',
    )
    if 'fill_seven_observation' not in source or 'CODE_PLAY' not in source or 'order_cards' not in source:
        raise RuntimeError('Seven projection source lost required extra kernels')
    return emit_vm_heap_source() + '\n' + '\n'.join(constants) + '\n' + source


SEVEN_DECISION_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*',
    'int32*', 'int32*', 'uint8*',
    'int32', 'int32', 'int32', 'int32*',
    'uint8*', 'int32*', 'int32*', 'int32',
    'float32*', 'int32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'float32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'int32*', 'int32*', 'int32*',
    'int32*', 'int32*',
    'int32*', 'float32*', 'float32*', 'float32*',
    'int32*', 'int32*', 'int32*', 'int32*',
    'int32*', 'int32*', 'float32*',
    'float32*', 'float32*', 'float32*', 'float32*', 'float32*', 'int32*', 'int32*',
    'int32', 'int32', 'int32', 'int32', 'int32',
)
SEVEN_OBS_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*',
    'int32*', 'int32*', 'uint8*',
    'int32', 'int32', 'int32', 'int32*',
    'uint8*', 'int32*', 'int32*', 'int32',
    'float32*', 'int32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'float32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'int32*', 'int32*', 'int32*',
    'int32*', 'int32*',
    'float32*', 'float32*', 'float32*', 'float32*', 'float32*', 'int32*',
    'int32*',
    'int32', 'int32', 'int32', 'int32',
)


def zero_seven_observation_extras(torch, n, device):
    n_seats = len(runtime_contract('seven')['seats'])
    seven_char = torch.zeros((n, 2, n_seats, 5), dtype=torch.float32, device=device)
    seven_char[..., 2] = -1
    seven_char[..., 4] = -1
    return {
        'seven_side': torch.zeros((n, 2, len(NEW_SIDE)), dtype=torch.float32, device=device),
        'seven_char': seven_char,
        'side_resist': torch.zeros((n, 2, len(ATTRIBUTES)), dtype=torch.float32, device=device),
        'char_resist': torch.zeros((n, 2, n_seats, len(ATTRIBUTES)), dtype=torch.float32, device=device),
        'choice_kind': torch.zeros((n, len(CHOICES)), dtype=torch.float32, device=device),
        'choice_cards': torch.full((n, MAX_ZONE_CARDS), -1, dtype=torch.int32, device=device),
    }


def zero_seven_candidate_extras(torch, n, max_candidates, device):
    return {
        'order_cards': torch.full((n, max_candidates, 2, ORDER_SLOTS), -1, dtype=torch.int32, device=device),
        'select_cards': torch.full((n, max_candidates, MAX_MULLIGAN_SELECTIONS), -1, dtype=torch.int32, device=device),
        'extra_ap': torch.zeros((n, max_candidates), dtype=torch.float32, device=device),
    }
