"""CUDA graph-to-token projection for GPU-rule public views and legal actions.

Ports gpu_observation/gpu_candidates host extractors onto device graphs, then
uses their existing device encoders. Production must not call the host token
extractors or the NumPy encoder. Unrepresented recovery-roster characters are
rejected rather than truncated; generic public views may still contain every
current catalogue role.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gpu_candidates import (
    MAX_MULLIGAN_SELECTIONS,
    OPTION_IDS_TUPLE,
    RawCandidateTokens,
    encode_tokens,
)
from .gpu_observation import (
    CARD_IDS,
    MAX_HAND_CARDS,
    MAX_PRESENT,
    MAX_TIMED,
    MAX_ZONE_CARDS,
    RawObservationTokens,
    SEATS,
    encode_observation_tokens,
)
from .vm_heap import emit_vm_heap_source


class GraphProjectionError(RuntimeError):
    """Fail-closed error for CUDA public-view / legality token projection."""


STATUS_OK = 0
STATUS_UNSUPPORTED_ROSTER = 1
STATUS_UNKNOWN_CARD = 2
STATUS_CAPACITY = 3
STATUS_INVALID_VIEW = 4
STATUS_EMPTY_POLICY = 5
STATUS_UNKNOWN_ACTION = 6
STATUS_UNKNOWN_OPTION = 7
STATUS_MISSING_OBJECT = 8
STATUS_UNKNOWN_TARGET = 9
STATUS_INVALID_OUTPUT = 10

_STATUS_NAMES = {
    STATUS_OK: 'ok',
    STATUS_UNSUPPORTED_ROSTER: 'unsupported_roster',
    STATUS_UNKNOWN_CARD: 'unknown_card',
    STATUS_CAPACITY: 'capacity',
    STATUS_INVALID_VIEW: 'invalid_view',
    STATUS_EMPTY_POLICY: 'empty_policy',
    STATUS_UNKNOWN_ACTION: 'unknown_action',
    STATUS_UNKNOWN_OPTION: 'unknown_option',
    STATUS_MISSING_OBJECT: 'missing_object',
    STATUS_UNKNOWN_TARGET: 'unknown_target',
    STATUS_INVALID_OUTPUT: 'invalid_output',
}

FIELD_NAMES = (
    'viewer_side', 'first_side', 'policy_state', 'turn', 'phase', 'pending_choice',
    'resolving_card', 'kind', 'choices', 'card_id', 'sides', 'front_debuff',
    'characters', 'summoned', 'id', 'front', 'last_front', 'extra_genesis_actor',
    'burn', 'star', 'delay', 'by', 'used', 'effects', 'definition_id', 'hp', 'shield',
    'ap', 'turn_count', 'fatigue', 'normal_attack_available', 'ultimate_available',
    'instant', 'ultimate_remaining', 'hand_count', 'deck_count', 'discard', 'weave',
    'left', 'infinite', 'end', 'light_resistance', 'extra_ap', 'extra_genesis_pending',
    'surplus', 'genesis_damage_bonus', 'harmony_damage', 'nanali_family',
    'nanali_family_played', 'slow', 'slow_turn', 'delay_turns', 'skip_normal_attack',
    'genesis_turn', 'genesis_player_hits', 'stacks', 'dots', 'nightmare', 'etch',
    'venom', 'guard', 'shape_id', 'flags', 'collapse', 'until', 'side', 'atk_buff',
    'next_bonus', 'next_shield', 'next_followup', 'pact', 'energy_next', 'allied_hurt',
    'pending_atk', 'share_overflow', 'hp_floor', 'collapse_count', 'base_attack',
    'base_max_hp', 'order', 'harmonized', 'jingu_loan', 'ultimate_used_turn',
    'attacked_this_turn', 'timed_attack', 'amount', 'entity_id', 'energy', 'down_turns',
    'awakened', 'ultimate_turns', 'growth', 'harmony', 'max_hp', 'energy_max',
    'harmony_max', 'jingu', 'beast_fangs', 'intimidation', 'extra_attacks', 'attack',
    'damage_immune', 'damage_limit', 'hand', 'hidden', 'revealed', 'jingu_mark',
    'delta', 'hand_face', 'type', 'character_id', 'target_id', 'option_id', 'choice_id',
    'card_ids', 'copy', 'expires_turn', 'derived', 'antique_investment', 'attack_mode',
    'instance_id', 'action', 'escalation_enabled', 'inspect_top', 'enemy_hand',
    'hand_redraw', 'mulligan', 'playing', 'choice', 'finished', 'a', 'b', 'player',
    'set', 'canhong', 'canhong.mirage', 'end_turn', 'play_card', 'ultimate', 'choose',
    'concede', 'legal_actions', 'canhong_dot_attack_turn:',
)
ACTION_TYPE_NAMES = ('end_turn', 'attack', 'play_card', 'ultimate', 'mulligan', 'choose')


def intern_layout(runtime=None):
    if runtime is not None:
        from .gpu_schema_contract import SEVEN_RUNTIME, normalize_runtime
        name = normalize_runtime(runtime)
        if name == SEVEN_RUNTIME:
            from .gpu_seven_projection import intern_layout_seven
            return intern_layout_seven()
    names = list(FIELD_NAMES) + list(SEATS) + list(CARD_IDS) + list(OPTION_IDS_TUPLE) + list(ACTION_TYPE_NAMES)
    unique = []
    index = {}
    for name in names:
        if name not in index:
            index[name] = len(unique)
            unique.append(name)
    return unique, index


INTERN_NAMES, INTERN_INDEX = intern_layout()
SEAT_START = INTERN_INDEX[SEATS[0]]
CARD_START = INTERN_INDEX[CARD_IDS[0]]
OPTION_START = INTERN_INDEX[OPTION_IDS_TUPLE[0]]
ACTION_START = INTERN_INDEX[ACTION_TYPE_NAMES[0]]
SUMMON_SEAT = SEATS.index('summon')


def _ident(name):
    return 'F_' + ''.join(ch if ch.isalnum() else '_' for ch in name)


def _intern_buffers(names):
    payload = bytearray(); offsets = []; lengths = []
    for name in names:
        raw = name.encode('utf-8')
        offsets.append(len(payload)); lengths.append(len(raw)); payload.extend(raw)
    return (
        np.frombuffer(bytes(payload), dtype=np.uint8).copy(),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
    )


def emit_projection_source():
    constants = '\n'.join('const int %s=%d;' % (_ident(name), INTERN_INDEX[name]) for name in FIELD_NAMES)
    constants += '\nconst int SEAT_START=%d,N_SEATS=%d,CARD_START=%d,N_CARDS=%d;' % (
        SEAT_START, len(SEATS), CARD_START, len(CARD_IDS),
    )
    constants += '\nconst int OPTION_START=%d,N_OPTIONS=%d,ACTION_START=%d,N_ACTIONS=%d,SUMMON_SEAT=%d;' % (
        OPTION_START, len(OPTION_IDS_TUPLE), ACTION_START, len(ACTION_TYPE_NAMES), SUMMON_SEAT,
    )
    constants += '\nconst int MAX_PRESENT=%d,MAX_TIMED=%d,MAX_ZONE_CARDS=%d,MAX_HAND_CARDS=%d,MAX_MULLIGAN=%d;' % (
        MAX_PRESENT, MAX_TIMED, MAX_ZONE_CARDS, MAX_HAND_CARDS, MAX_MULLIGAN_SELECTIONS,
    )
    return emit_vm_heap_source() + '\n' + constants + _PROJECTION_SOURCE


_PROJECTION_SOURCE = r'''
using namespace duel_vm_heap;
enum ProjStatus {
 P_OK=0,P_ROSTER=1,P_CARD=2,P_CAP=3,P_VIEW=4,P_EMPTY=5,P_ACTION=6,P_OPTION=7,P_OBJ=8,P_TARGET=9,P_OUT=10
};
const int NOT_FOUND=(-2147483647-1);
__device__ inline int intern_eq(Heap* h,int ref,const unsigned char* names,const int* off,const int* len,int ident){
 NodeView node; if(!get_node(h,ref,&node)||node.tag!=TAG_STR) return 0;
 if(node.count!=len[ident]) return 0;
 const unsigned char* p=get_payload_ptr(h,&node); if(!p) return 0;
 return vm_memcmp(p,names+off[ident],len[ident])==0;
}
__device__ inline int dict_get_intern(Heap* h,int dict,const unsigned char* names,const int* off,const int* len,int ident){
 if(dict==NOT_FOUND) return NOT_FOUND;
 NodeView node; if(!get_node(h,dict,&node)) return NOT_FOUND;
 if(node.tag!=TAG_DICT && node.tag<TAG_PLAYER_ENTITY) return NOT_FOUND;
 for(int i=0;i<node.count;i++){
  int k=0,v=0; if(!get_edge(h,&node,i,&k,&v)) return NOT_FOUND;
  if(intern_eq(h,k,names,off,len,ident)) return v;
 }
 return NOT_FOUND;
}
__device__ inline int is_none(Heap* h,int ref){
 if(ref==NOT_FOUND) return 1;
 NodeView node; if(!get_node(h,ref,&node)) return 0;
 return node.tag==TAG_NONE;
}
__device__ inline int nonempty(Heap* h,int ref){
 if(ref==NOT_FOUND||is_none(h,ref)) return 0;
 NodeView node; if(!get_node(h,ref,&node)) return 0;
 if(node.tag==TAG_DICT||node.tag==TAG_LIST||node.tag==TAG_TUPLE) return node.count>0;
 return 1;
}
__device__ inline int as_bool(Heap* h,int ref){
 if(ref==NOT_FOUND||is_none(h,ref)) return 0;
 NodeView node; if(!get_node(h,ref,&node)) return 0;
 if(node.tag==TAG_BOOL) return node.extra!=0;
 if(node.tag==TAG_INT){ long long v=0; if(read_int64(h,ref,&v)!=0) return 0; return v!=0; }
 if(node.tag==TAG_FLOAT){ double v=0; if(read_float64(h,ref,&v)!=0) return 0; return v!=0; }
 if(node.tag==TAG_LIST||node.tag==TAG_TUPLE||node.tag==TAG_DICT) return node.count!=0;
 return 1;
}
__device__ inline int as_double(Heap* h,int ref,double* out){
 if(ref==NOT_FOUND||is_none(h,ref)){ *out=0; return 1; }
 NodeView node; if(!get_node(h,ref,&node)) return 0;
 if(node.tag==TAG_BOOL){ *out=(double)node.extra; return 1; }
 if(node.tag==TAG_INT){ long long v=0; if(read_int64(h,ref,&v)!=0) return 0; *out=(double)v; return 1; }
 if(node.tag==TAG_FLOAT) return read_float64(h,ref,out)==0;
 return 0;
}
__device__ inline int lookup_range(Heap* h,int ref,const unsigned char* names,const int* off,const int* len,int start,int count){
 if(ref==NOT_FOUND||is_none(h,ref)) return -1;
 for(int i=0;i<count;i++) if(intern_eq(h,ref,names,off,len,start+i)) return i;
 return -2;
}
__device__ inline int seat_lookup(Heap* h,int ref,const unsigned char* names,const int* off,const int* len){
 if(ref==NOT_FOUND||is_none(h,ref)) return -1;
 NodeView node; if(!get_node(h,ref,&node)||node.tag!=TAG_STR) return -2;
 const unsigned char* p=get_payload_ptr(h,&node); if(!p) return -2;
 if(node.count>=7 && vm_memcmp(p,(const unsigned char*)"summon_",7)==0) return SUMMON_SEAT;
 int idx=lookup_range(h,ref,names,off,len,SEAT_START,N_SEATS);
 return idx;
}
__device__ inline int card_lookup(Heap* h,int ref,const unsigned char* names,const int* off,const int* len){
 return lookup_range(h,ref,names,off,len,CARD_START,N_CARDS);
}
__device__ inline int list_count(Heap* h,int ref){
 if(ref==NOT_FOUND||is_none(h,ref)) return 0;
 NodeView node; if(!get_node(h,ref,&node)) return -1;
 if(node.tag!=TAG_LIST && node.tag!=TAG_TUPLE) return -1;
 return node.count;
}
__device__ inline int list_item(Heap* h,int ref,int index){
 NodeView node; if(!get_node(h,ref,&node)) return NOT_FOUND;
 if((node.tag!=TAG_LIST && node.tag!=TAG_TUPLE) || index<0 || index>=node.count) return NOT_FOUND;
 int k=0,v=0; if(!get_edge(h,&node,index,&k,&v)) return NOT_FOUND;
 return k;
}
__device__ inline int dict_get_key(Heap* h,int dict,int key){
 if(dict==NOT_FOUND||key==NOT_FOUND) return NOT_FOUND;
 NodeView node; if(!get_node(h,dict,&node)) return NOT_FOUND;
 if(node.tag!=TAG_DICT && node.tag<TAG_PLAYER_ENTITY) return NOT_FOUND;
 for(int i=0;i<node.count;i++){
  int k=0,v=0; if(!get_edge(h,&node,i,&k,&v)) return NOT_FOUND;
  if(key_equal(h,k,key)) return v;
 }
 return NOT_FOUND;
}
__device__ inline int parse_target(Heap* h,int ref,int viewer_b,const unsigned char* names,const int* off,const int* len,int* tside,int* tseat,int* status){
 *tside=-1; *tseat=-1;
 if(ref==NOT_FOUND||is_none(h,ref)) return 1;
 NodeView node; if(!get_node(h,ref,&node)||node.tag!=TAG_STR){ *status=P_TARGET; return 0; }
 const unsigned char* p=get_payload_ptr(h,&node); if(!p){ *status=P_TARGET; return 0; }
 int colon=-1; for(int i=0;i<node.count;i++) if(p[i]==':'){ colon=i; break; }
 if(colon!=1||(p[0]!='a'&&p[0]!='b')){ *status=P_TARGET; return 0; }
 int target_b=p[0]=='b'; *tside=(target_b==viewer_b)?0:1;
 int cid_len=node.count-2; const unsigned char* cid=p+2;
 if(cid_len==6 && vm_memcmp(cid,(const unsigned char*)"player",6)==0){ *tseat=-1; return 1; }
 if(cid_len>=7 && vm_memcmp(cid,(const unsigned char*)"summon_",7)==0){ *tseat=SUMMON_SEAT; return 1; }
 for(int i=0;i<N_SEATS;i++){
  if(len[SEAT_START+i]==cid_len && vm_memcmp(names+off[SEAT_START+i],cid,cid_len)==0){ *tseat=i; return 1; }
 }
 *status=P_TARGET; return 0;
}
__device__ inline int used_dot_turn(Heap* h,int used,int eid,const unsigned char* names,const int* off,const int* len,double* out){
 *out=-999.0;
 if(used==NOT_FOUND||eid==NOT_FOUND||is_none(h,used)||is_none(h,eid)) return 1;
 NodeView un,en; if(!get_node(h,used,&un)||!get_node(h,eid,&en)||en.tag!=TAG_STR) return 1;
 if(un.tag!=TAG_DICT) return 1;
 const unsigned char* ep=get_payload_ptr(h,&en); if(!ep) return 1;
 int pref=len[F_canhong_dot_attack_turn_];
 for(int i=0;i<un.count;i++){
  int k=0,v=0; if(!get_edge(h,&un,i,&k,&v)) continue;
  NodeView kn; if(!get_node(h,k,&kn)||kn.tag!=TAG_STR) continue;
  if(kn.count!=pref+en.count) continue;
  const unsigned char* kp=get_payload_ptr(h,&kn); if(!kp) continue;
  if(vm_memcmp(kp,names+off[F_canhong_dot_attack_turn_],pref)==0 && vm_memcmp(kp+pref,ep,en.count)==0){
   return as_double(h,v,out);
  }
 }
 return 1;
}
__device__ inline int has_effect(Heap* h,int effects,const unsigned char* names,const int* off,const int* len,int ident){
 int n=list_count(h,effects); if(n<0) n=0;
 for(int i=0;i<n;i++){
  int eff=list_item(h,effects,i);
  if(intern_eq(h,dict_get_intern(h,eff,names,off,len,F_definition_id),names,off,len,ident)) return 1;
 }
 return 0;
}
__device__ inline int fill_observation(
 Heap* h,int view,int row,const unsigned char* names,const int* off,const int* len,
 float* global_tokens,int* side_int,float* side_float,int* char_present,int* char_seats,
 int* char_int,float* char_float,float* char_timed,int* zone_cards,int* mark_cards,
 int* mark_deltas,int* mark_valid,int* substituted_cards,int* substituted_valid,
 int* mask,int* counts,int* status){
 int viewer=dict_get_intern(h,view,names,off,len,F_viewer_side);
 int first=dict_get_intern(h,view,names,off,len,F_first_side);
 int viewer_b=intern_eq(h,viewer,names,off,len,F_b);
 int viewer_a=intern_eq(h,viewer,names,off,len,F_a);
 int first_b=intern_eq(h,first,names,off,len,F_b);
 int first_a=intern_eq(h,first,names,off,len,F_a);
 if(!(viewer_a||viewer_b)||!(first_a||first_b)){ *status=P_VIEW; return 0; }
 int publicv=dict_get_intern(h,view,names,off,len,F_policy_state);
 if(!nonempty(h,publicv)){ *status=P_VIEW; return 0; }
 mask[row]=1; counts[row]=1;
 double turn=0; as_double(h,dict_get_intern(h,view,names,off,len,F_turn),&turn);
 int phase=dict_get_intern(h,view,names,off,len,F_phase);
 float phase_val=-1.f;
 if(intern_eq(h,phase,names,off,len,F_mulligan)) phase_val=0.f;
 else if(intern_eq(h,phase,names,off,len,F_playing)) phase_val=1.f;
 else if(intern_eq(h,phase,names,off,len,F_choice)) phase_val=2.f;
 else if(intern_eq(h,phase,names,off,len,F_finished)) phase_val=3.f;
 else { *status=P_VIEW; return 0; }
 int pending=dict_get_intern(h,view,names,off,len,F_pending_choice);
 int resolving=dict_get_intern(h,view,names,off,len,F_resolving_card);
 float pending_kind=-1.f, choices_count=0.f;
 if(nonempty(h,pending)){
  int kind=dict_get_intern(h,pending,names,off,len,F_kind);
  if(intern_eq(h,kind,names,off,len,F_inspect_top)) pending_kind=1.f;
  else if(intern_eq(h,kind,names,off,len,F_discard)) pending_kind=2.f;
  else if(intern_eq(h,kind,names,off,len,F_enemy_hand)) pending_kind=3.f;
  int nch=list_count(h,dict_get_intern(h,pending,names,off,len,F_choices));
  if(nch<0) nch=0; choices_count=(float)nch;
 }
 float resolving_idx=-1.f, has_pr=0.f;
 if(nonempty(h,pending)&&nonempty(h,resolving)){
  int rcid=dict_get_intern(h,resolving,names,off,len,F_card_id);
  if(!is_none(h,rcid)&&rcid!=NOT_FOUND){
   int idx=card_lookup(h,rcid,names,off,len);
   if(idx==-2){ *status=P_CARD; return 0; }
   if(idx>=0){ resolving_idx=(float)idx; has_pr=1.f; }
  }
 }
 float* gt=global_tokens+row*9;
 gt[0]=(float)turn; gt[1]=viewer_b?1.f:0.f; gt[2]=first_b?1.f:0.f;
 gt[3]=as_bool(h,dict_get_intern(h,publicv,names,off,len,F_escalation_enabled))?1.f:0.f;
 gt[4]=phase_val; gt[5]=pending_kind; gt[6]=choices_count; gt[7]=resolving_idx; gt[8]=has_pr;
 int sides=dict_get_intern(h,view,names,off,len,F_sides);
 int public_sides=dict_get_intern(h,publicv,names,off,len,F_sides);
 int side_keys[2]={viewer_b?F_b:F_a, viewer_b?F_a:F_b};
 int own_team=NOT_FOUND, opp_team=NOT_FOUND;
 for(int s=0;s<2;s++){
  int team=dict_get_intern(h,sides,names,off,len,side_keys[s]);
  int extra=dict_get_intern(h,public_sides,names,off,len,side_keys[s]);
  if(team==NOT_FOUND){ *status=P_VIEW; return 0; }
  if(s==0) own_team=team; else opp_team=team;
  int zones=dict_get_intern(h,team,names,off,len,F_front_debuff);
  int chars=dict_get_intern(h,team,names,off,len,F_characters);
  int nchars=list_count(h,chars); if(nchars<0) nchars=0;
  int summons[16], nsum=0, nonsum[16], nnonsum=0;
  int front=dict_get_intern(h,team,names,off,len,F_front);
  for(int i=0;i<nchars && i<16;i++){
   int hero=list_item(h,chars,i);
   int hid=dict_get_intern(h,hero,names,off,len,F_id);
   int is_sum=as_bool(h,dict_get_intern(h,hero,names,off,len,F_summoned));
   NodeView idn;
   if(get_node(h,hid,&idn)&&idn.tag==TAG_STR){
    const unsigned char* p=get_payload_ptr(h,&idn);
    if(p && idn.count>=7 && vm_memcmp(p,(const unsigned char*)"summon_",7)==0) is_sum=1;
   }
   if(is_sum){ if(nsum<16) summons[nsum++]=i; }
   else if(nnonsum<16) nonsum[nnonsum++]=i;
  }
  int chosen=-1;
  if(nsum){
   chosen=summons[nsum-1];
   for(int i=0;i<nsum;i++){
    int hero=list_item(h,chars,summons[i]);
    int hid=dict_get_intern(h,hero,names,off,len,F_id);
    if(hid!=NOT_FOUND && key_equal(h,hid,front)){ chosen=summons[i]; break; }
   }
  }
  int active_hero[8], active_lookup[8], nactive=0;
  for(int i=0;i<nnonsum;i++){
   if(nactive>=MAX_PRESENT){ *status=P_CAP; return 0; }
   int hero=list_item(h,chars,nonsum[i]);
   active_hero[nactive]=hero;
   active_lookup[nactive]=dict_get_intern(h,hero,names,off,len,F_id);
   nactive++;
  }
  if(chosen>=0){
   if(nactive>=MAX_PRESENT){ *status=P_CAP; return 0; }
   int hero=list_item(h,chars,chosen);
   active_hero[nactive]=hero;
   active_lookup[nactive]=dict_get_intern(h,hero,names,off,len,F_id);
   nactive++;
  }
  int last_front=dict_get_intern(h,team,names,off,len,F_last_front);
  int front_seat=seat_lookup(h,front,names,off,len);
  int last_seat=seat_lookup(h,last_front,names,off,len);
  if(front_seat==-2||last_seat==-2){ *status=P_ROSTER; return 0; }
  int burn=dict_get_intern(h,zones,names,off,len,F_burn);
  int star=dict_get_intern(h,zones,names,off,len,F_star);
  int delay=dict_get_intern(h,zones,names,off,len,F_delay);
  int burn_by=-1, dark_by=-1, delay_by=-1;
  if(nonempty(h,burn)){
   int by=dict_get_intern(h,burn,names,off,len,F_by);
   burn_by=intern_eq(h,by,names,off,len,F_a)?0:1;
  }
  if(nonempty(h,star)){
   int by=dict_get_intern(h,star,names,off,len,F_by);
   dark_by=intern_eq(h,by,names,off,len,F_a)?0:1;
  }
  if(nonempty(h,delay)){
   int by=dict_get_intern(h,delay,names,off,len,F_by);
   delay_by=intern_eq(h,by,names,off,len,F_a)?0:1;
  }
  int extra_actor=dict_get_intern(h,extra,names,off,len,F_extra_genesis_actor);
  int extra_seat=seat_lookup(h,extra_actor,names,off,len);
  if(extra_seat==-2){ *status=P_ROSTER; return 0; }
  int* si=side_int+((row*2+s)*6);
  si[0]=front_seat; si[1]=last_seat; si[2]=burn_by; si[3]=dark_by; si[4]=extra_seat; si[5]=delay_by;
  int dots=dict_get_intern(h,zones,names,off,len,F_dots);
  double mirage=0, dot_turn=-999.0;
  int used=dict_get_intern(h,team,names,off,len,F_used);
  for(int p=0;p<nactive;p++){
   int hid=dict_get_intern(h,active_hero[p],names,off,len,F_id);
   if(intern_eq(h,hid,names,off,len,F_canhong)){
    int eid=dict_get_intern(h,active_hero[p],names,off,len,F_entity_id);
    used_dot_turn(h,used,eid,names,off,len,&dot_turn);
    if(has_effect(h,dict_get_intern(h,active_hero[p],names,off,len,F_effects),names,off,len,F_canhong_mirage)) mirage=1.0;
   }
  }
  double tmp=0;
  float* sf=side_float+((row*2+s)*39);
  as_double(h,dict_get_intern(h,team,names,off,len,F_hp),&tmp); sf[0]=(float)tmp;
  as_double(h,dict_get_intern(h,team,names,off,len,F_shield),&tmp); sf[1]=(float)tmp;
  as_double(h,dict_get_intern(h,team,names,off,len,F_ap),&tmp); sf[2]=(float)tmp;
  as_double(h,dict_get_intern(h,team,names,off,len,F_turn_count),&tmp); sf[3]=(float)tmp;
  as_double(h,dict_get_intern(h,team,names,off,len,F_fatigue),&tmp); sf[4]=(float)tmp;
  sf[5]=as_bool(h,dict_get_intern(h,team,names,off,len,F_normal_attack_available))?1.f:0.f;
  sf[6]=as_bool(h,dict_get_intern(h,team,names,off,len,F_ultimate_available))?1.f:0.f;
  sf[7]=as_bool(h,dict_get_key(h,used,dict_get_intern(h,used,names,off,len,F_instant)))?1.f:0.f;
  // used instant is keyed by intern 'instant' string present in the intern table, not a graph key node.
  int instant_key_used=NOT_FOUND;
  if(used!=NOT_FOUND && !is_none(h,used)){
   NodeView un; if(get_node(h,used,&un) && un.tag==TAG_DICT){
    for(int i=0;i<un.count;i++){
     int k=0,v=0; get_edge(h,&un,i,&k,&v);
     if(intern_eq(h,k,names,off,len,F_instant)){ instant_key_used=v; break; }
    }
   }
  }
  sf[7]=as_bool(h,instant_key_used)?1.f:0.f;
  as_double(h,dict_get_intern(h,team,names,off,len,F_ultimate_remaining),&tmp); sf[8]=(float)tmp;
  as_double(h,dict_get_intern(h,team,names,off,len,F_hand_count),&tmp); sf[9]=(float)tmp;
  as_double(h,dict_get_intern(h,team,names,off,len,F_deck_count),&tmp); sf[10]=(float)tmp;
  int ndisc=list_count(h,dict_get_intern(h,team,names,off,len,F_discard)); if(ndisc<0) ndisc=0; sf[11]=(float)ndisc;
  sf[12]=as_bool(h,dict_get_intern(h,zones,names,off,len,F_weave))?1.f:0.f;
  as_double(h,dict_get_intern(h,burn,names,off,len,F_left),&tmp); sf[13]=(float)tmp;
  sf[14]=as_bool(h,dict_get_intern(h,burn,names,off,len,F_infinite))?1.f:0.f;
  as_double(h,dict_get_intern(h,delay,names,off,len,F_end),&tmp); sf[15]=(float)tmp;
  as_double(h,dict_get_intern(h,extra,names,off,len,F_light_resistance),&tmp); sf[16]=(float)tmp;
  as_double(h,dict_get_intern(h,star,names,off,len,F_left),&tmp); sf[17]=(float)tmp;
  as_double(h,dict_get_intern(h,extra,names,off,len,F_extra_ap),&tmp); sf[18]=(float)tmp;
  sf[19]=as_bool(h,dict_get_intern(h,extra,names,off,len,F_extra_genesis_pending))?1.f:0.f;
  sf[20]=as_bool(h,dict_get_intern(h,extra,names,off,len,F_surplus))?1.f:0.f;
  as_double(h,dict_get_intern(h,extra,names,off,len,F_genesis_damage_bonus),&tmp); sf[21]=(float)tmp;
  sf[22]=as_bool(h,dict_get_intern(h,extra,names,off,len,F_harmony_damage))?1.f:0.f;
  as_double(h,dict_get_intern(h,extra,names,off,len,F_nanali_family),&tmp); sf[23]=(float)tmp;
  as_double(h,dict_get_intern(h,extra,names,off,len,F_nanali_family_played),&tmp); sf[24]=(float)tmp;
  as_double(h,dict_get_intern(h,delay,names,off,len,F_left),&tmp); sf[25]=(float)tmp;
  as_double(h,dict_get_intern(h,delay,names,off,len,F_slow),&tmp); sf[26]=(float)tmp;
  int slow_turn=dict_get_intern(h,delay,names,off,len,F_slow_turn);
  if(slow_turn==NOT_FOUND||is_none(h,slow_turn)) sf[27]=-999.f; else { as_double(h,slow_turn,&tmp); sf[27]=(float)tmp; }
  as_double(h,dict_get_intern(h,extra,names,off,len,F_delay_turns),&tmp); sf[28]=(float)tmp;
  sf[29]=as_bool(h,dict_get_intern(h,extra,names,off,len,F_skip_normal_attack))?1.f:0.f;
  int genesis_turn=dict_get_intern(h,extra,names,off,len,F_genesis_turn);
  if(genesis_turn==NOT_FOUND||is_none(h,genesis_turn)) sf[30]=-999.f; else { as_double(h,genesis_turn,&tmp); sf[30]=(float)tmp; }
  as_double(h,dict_get_intern(h,extra,names,off,len,F_genesis_player_hits),&tmp); sf[31]=(float)tmp;
  as_double(h,dict_get_intern(h,burn,names,off,len,F_stacks),&tmp); sf[32]=(float)tmp;
  int nightmare=dict_get_intern(h,dots,names,off,len,F_nightmare);
  int etch=dict_get_intern(h,dots,names,off,len,F_etch);
  int venom=dict_get_intern(h,dots,names,off,len,F_venom);
  int guard=dict_get_intern(h,dots,names,off,len,F_guard);
  as_double(h,dict_get_intern(h,nightmare,names,off,len,F_stacks),&tmp); sf[33]=(float)tmp;
  as_double(h,dict_get_intern(h,etch,names,off,len,F_stacks),&tmp); sf[34]=(float)tmp;
  as_double(h,dict_get_intern(h,venom,names,off,len,F_stacks),&tmp); sf[35]=(float)tmp;
  as_double(h,dict_get_intern(h,guard,names,off,len,F_stacks),&tmp); sf[36]=(float)tmp;
  sf[37]=(float)mirage; sf[38]=(float)dot_turn;
  int meta_chars=dict_get_intern(h,extra,names,off,len,F_characters);
  for(int p=0;p<nactive;p++){
   int hero=active_hero[p];
   int hid=dict_get_intern(h,hero,names,off,len,F_id);
   int seat=(chosen>=0 && p==nactive-1 && nsum)?SUMMON_SEAT:seat_lookup(h,hid,names,off,len);
   if(nsum && p==nactive-1){
    NodeView idn;
    int is_sum=as_bool(h,dict_get_intern(h,hero,names,off,len,F_summoned));
    if(get_node(h,hid,&idn)&&idn.tag==TAG_STR){
     const unsigned char* rp=get_payload_ptr(h,&idn);
     if(rp && idn.count>=7 && vm_memcmp(rp,(const unsigned char*)"summon_",7)==0) is_sum=1;
    }
    if(is_sum) seat=SUMMON_SEAT;
   }
   if(seat<0){ *status=P_ROSTER; return 0; }
   int base=((row*2+s)*MAX_PRESENT+p);
   char_present[base]=1; char_seats[base]=seat;
   int shape=dict_get_intern(h,hero,names,off,len,F_shape_id);
   int shape_idx=-1;
   if(shape!=NOT_FOUND && !is_none(h,shape)){
    shape_idx=card_lookup(h,shape,names,off,len);
    if(shape_idx==-2){ *status=P_CARD; return 0; }
   }
   int meta_hero=dict_get_key(h,meta_chars,active_lookup[p]);
   int flags=dict_get_intern(h,meta_hero,names,off,len,F_flags);
   int collapse=dict_get_intern(h,flags,names,off,len,F_collapse);
   int collapse_by=-1; double collapse_until=0;
   if(nonempty(h,collapse)){
    int cs=dict_get_intern(h,collapse,names,off,len,F_side);
    if(intern_eq(h,cs,names,off,len,F_a)) collapse_by=0;
    else if(intern_eq(h,cs,names,off,len,F_b)) collapse_by=1;
    else { *status=P_VIEW; return 0; }
    as_double(h,dict_get_intern(h,collapse,names,off,len,F_until),&collapse_until);
   }
   char_int[base*2+0]=shape_idx; char_int[base*2+1]=collapse_by;
   float* cf=char_float+base*38;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_hp),&tmp); cf[0]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_max_hp),&tmp); cf[1]=(float)tmp;
   as_double(h,dict_get_intern(h,meta_hero,names,off,len,F_base_attack),&tmp); cf[2]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_growth),&tmp); cf[3]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_atk_buff),&tmp); cf[4]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_shield),&tmp); cf[5]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_harmony),&tmp); cf[6]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_energy),&tmp); cf[7]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_down_turns),&tmp); cf[8]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_awakened),&tmp); cf[9]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_ultimate_turns),&tmp); cf[10]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_next_bonus),&tmp); cf[11]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_next_shield),&tmp); cf[12]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_next_followup),&tmp); cf[13]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_pact),&tmp); cf[14]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_energy_next),&tmp); cf[15]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_allied_hurt),&tmp); cf[16]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_pending_atk),&tmp); cf[17]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_share_overflow),&tmp); cf[18]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_hp_floor),&tmp); cf[19]=(float)tmp;
   as_double(h,dict_get_intern(h,flags,names,off,len,F_collapse_count),&tmp); cf[20]=(float)tmp;
   as_double(h,dict_get_intern(h,meta_hero,names,off,len,F_base_max_hp),&tmp); cf[21]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_energy_max),&tmp); cf[22]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_harmony_max),&tmp); cf[23]=(float)tmp;
   int order=dict_get_intern(h,meta_hero,names,off,len,F_order);
   if(order==NOT_FOUND||is_none(h,order)) cf[24]=-1.f; else { as_double(h,order,&tmp); cf[24]=(float)tmp; }
   as_double(h,dict_get_intern(h,meta_hero,names,off,len,F_harmonized),&tmp); cf[25]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_jingu),&tmp); cf[26]=(float)tmp;
   cf[27]=as_bool(h,dict_get_intern(h,meta_hero,names,off,len,F_jingu_loan))?1.f:0.f;
   int ult_turn=dict_get_intern(h,meta_hero,names,off,len,F_ultimate_used_turn);
   if(ult_turn==NOT_FOUND||is_none(h,ult_turn)) cf[28]=-999.f; else { as_double(h,ult_turn,&tmp); cf[28]=(float)tmp; }
   cf[29]=as_bool(h,dict_get_intern(h,meta_hero,names,off,len,F_attacked_this_turn))?1.f:0.f;
   as_double(h,dict_get_intern(h,meta_hero,names,off,len,F_light_resistance),&tmp); cf[30]=(float)tmp;
   cf[31]=(float)collapse_until;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_beast_fangs),&tmp); cf[32]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_intimidation),&tmp); cf[33]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_extra_attacks),&tmp); cf[34]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_attack),&tmp); cf[35]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_damage_immune),&tmp); cf[36]=(float)tmp;
   as_double(h,dict_get_intern(h,hero,names,off,len,F_damage_limit),&tmp); cf[37]=(float)tmp;
   int timed=dict_get_intern(h,flags,names,off,len,F_timed_attack);
   int nt=list_count(h,timed); if(nt<0) nt=0;
   if(nt>MAX_TIMED){ *status=P_CAP; return 0; }
   for(int t=0;t<nt;t++){
    int entry=list_item(h,timed,t);
    float* ct=char_timed+(((row*2+s)*MAX_PRESENT+p)*MAX_TIMED+t)*2;
    as_double(h,dict_get_intern(h,entry,names,off,len,F_amount),&tmp); ct[0]=(float)tmp;
    as_double(h,dict_get_intern(h,entry,names,off,len,F_left),&tmp); ct[1]=(float)tmp;
   }
  }
 }
 int z0=dict_get_intern(h,own_team,names,off,len,F_hand);
 int z1=dict_get_intern(h,own_team,names,off,len,F_discard);
 int z2=dict_get_intern(h,opp_team,names,off,len,F_discard);
 int z3=dict_get_intern(h,opp_team,names,off,len,F_hand);
 int zones_src[5]={z0,z1,z2,z3,z0};
 int revealed_only[5]={0,0,0,1,1};
 int hidden_skip[5]={0,0,0,1,0};
 for(int z=0;z<5;z++){
  int n=list_count(h,zones_src[z]); if(n<0) n=0;
  int written=0;
  for(int i=0;i<n;i++){
   int card=list_item(h,zones_src[z],i);
   if(hidden_skip[z] && as_bool(h,dict_get_intern(h,card,names,off,len,F_hidden))) continue;
   if(revealed_only[z]==1 && z==4){
    if(as_bool(h,dict_get_intern(h,card,names,off,len,F_hidden))) continue;
    if(!as_bool(h,dict_get_intern(h,card,names,off,len,F_revealed))) continue;
   }
   if(written>=MAX_ZONE_CARDS){ *status=P_CAP; return 0; }
   int cid=dict_get_intern(h,card,names,off,len,F_card_id);
   int idx=card_lookup(h,cid,names,off,len);
   if(idx<0){ *status=P_CARD; return 0; }
   zone_cards[(row*5+z)*MAX_ZONE_CARDS+written]=idx;
   written++;
  }
 }
 int nh=list_count(h,z0); if(nh<0) nh=0;
 if(nh>MAX_HAND_CARDS){ *status=P_CAP; return 0; }
 for(int i=0;i<nh;i++){
  int card=list_item(h,z0,i);
  int cid=dict_get_intern(h,card,names,off,len,F_card_id);
  int idx=card_lookup(h,cid,names,off,len);
  if(idx<0){ *status=P_CARD; return 0; }
  int mark=dict_get_intern(h,card,names,off,len,F_jingu_mark);
  if(mark!=NOT_FOUND){
   double delta=0; as_double(h,dict_get_intern(h,mark,names,off,len,F_delta),&delta);
   int di=(int)delta; if(di<-1||di>1){ *status=P_VIEW; return 0; }
   mark_cards[row*MAX_HAND_CARDS+i]=idx; mark_deltas[row*MAX_HAND_CARDS+i]=di; mark_valid[row*MAX_HAND_CARDS+i]=1;
  }
  int face=dict_get_intern(h,card,names,off,len,F_hand_face);
  int face_cid=dict_get_intern(h,face,names,off,len,F_card_id);
  if(face_cid!=NOT_FOUND && !is_none(h,face_cid) && !intern_eq(h,face_cid,names,off,len,F_card_id)){
   int same=0;
   NodeView a,b;
   if(get_node(h,face_cid,&a)&&get_node(h,cid,&b)&&a.tag==TAG_STR&&b.tag==TAG_STR&&a.count==b.count){
    const unsigned char* pa=get_payload_ptr(h,&a); const unsigned char* pb=get_payload_ptr(h,&b);
    if(pa&&pb&&vm_memcmp(pa,pb,a.count)==0) same=1;
   }
   if(!same){
    int fidx=card_lookup(h,face_cid,names,off,len);
    if(fidx<0){ *status=P_CARD; return 0; }
    substituted_cards[row*MAX_HAND_CARDS+i]=fidx; substituted_valid[row*MAX_HAND_CARDS+i]=1;
   }
  }
 }
 return 1;
}
__device__ inline int action_type_code(Heap* h,int ref,const unsigned char* names,const int* off,const int* len){
 if(intern_eq(h,ref,names,off,len,F_concede)) return -2;
 for(int i=0;i<N_ACTIONS;i++) if(intern_eq(h,ref,names,off,len,ACTION_START+i)) return i;
 return -1;
}
__device__ inline int fill_candidates(
 Heap* h,int view,int actions,int row,int max_candidates,
 const unsigned char* names,const int* off,const int* len,
 int* int_tokens,float* float_tokens,float* item_flags,float* turn_counts,
 int* mulligan_cards,int* cand_counts,int* cand_mask,int* action_root,int* status){
 int viewer=dict_get_intern(h,view,names,off,len,F_viewer_side);
 int viewer_b=intern_eq(h,viewer,names,off,len,F_b);
 int sides=dict_get_intern(h,view,names,off,len,F_sides);
 int own=dict_get_intern(h,sides,names,off,len,viewer_b?F_b:F_a);
 double turn=0; as_double(h,dict_get_intern(h,own,names,off,len,F_turn_count),&turn);
 turn_counts[row]=(float)turn;
 int n=list_count(h,actions); if(n<0){ *status=P_OUT; return 0; }
 int filtered=make_list(h); if(filtered<0||h->error){ *status=P_CAP; return 0; }
 int kept=0;
 for(int i=0;i<n;i++){
  int action=list_item(h,actions,i);
  int typ=dict_get_intern(h,action,names,off,len,F_type);
  int code=action_type_code(h,typ,names,off,len);
  if(code==-2) continue;
  if(code<0){ *status=P_ACTION; return 0; }
  if(kept>=max_candidates){ *status=P_CAP; return 0; }
  if(list_append(h,filtered,action)<0||h->error){ *status=P_CAP; return 0; }
  int base=(row*max_candidates+kept)*7;
  int fbase=(row*max_candidates+kept)*8;
  int ibase=(row*max_candidates+kept)*2;
  int mbase=(row*max_candidates+kept)*MAX_MULLIGAN;
  for(int k=0;k<7;k++) int_tokens[base+k]=-1;
  for(int k=0;k<8;k++) float_tokens[fbase+k]=0;
  item_flags[ibase]=0; item_flags[ibase+1]=0;
  int_tokens[base+0]=code;
  cand_mask[row*max_candidates+kept]=1;
  int actor_idx=-1, card_kind=-1, tside=-1, tseat=-1, option_idx=-1, physical=-1;
  float revealed=0, marked=0, mark_delta=0, set_attack=0, item_instant=0, is_copy=0, raw_expires=0, antique=0, has_item=0, item_derived=0;
  int opt=dict_get_intern(h,action,names,off,len,F_option_id);
  if(opt!=NOT_FOUND && !is_none(h,opt)){
   option_idx=lookup_range(h,opt,names,off,len,OPTION_START,N_OPTIONS);
   if(option_idx<0){ *status=P_OPTION; return 0; }
  }
  int hand=dict_get_intern(h,own,names,off,len,F_hand);
  if(code==2){
   int inst=dict_get_intern(h,action,names,off,len,F_card_id);
   int item=NOT_FOUND; int nh=list_count(h,hand); if(nh<0) nh=0;
   for(int c=0;c<nh;c++){
    int card=list_item(h,hand,c);
    int iid=dict_get_intern(h,card,names,off,len,F_instance_id);
    if(iid!=NOT_FOUND && key_equal(h,iid,inst)){ item=card; break; }
   }
   if(item==NOT_FOUND){ *status=P_OBJ; return 0; }
   int ch=dict_get_intern(h,item,names,off,len,F_character_id);
   actor_idx=seat_lookup(h,ch,names,off,len);
   if(actor_idx<0){ *status=P_ROSTER; return 0; }
   int cid=dict_get_intern(h,item,names,off,len,F_card_id);
   card_kind=card_lookup(h,cid,names,off,len);
   if(card_kind<0){ *status=P_CARD; return 0; }
   if(!parse_target(h,dict_get_intern(h,action,names,off,len,F_target_id),viewer_b,names,off,len,&tside,&tseat,status)) return 0;
   revealed=as_bool(h,dict_get_intern(h,item,names,off,len,F_revealed))?1.f:0.f;
   int mark=dict_get_intern(h,item,names,off,len,F_jingu_mark);
   if(mark!=NOT_FOUND){
    marked=1.f; double d=0; as_double(h,dict_get_intern(h,mark,names,off,len,F_delta),&d); mark_delta=(float)d;
   }
   set_attack=intern_eq(h,dict_get_intern(h,item,names,off,len,F_attack_mode),names,off,len,F_set)?1.f:0.f;
   item_instant=as_bool(h,dict_get_intern(h,item,names,off,len,F_instant))?1.f:0.f;
   is_copy=as_bool(h,dict_get_intern(h,item,names,off,len,F_copy))?1.f:0.f;
   double ex=0; as_double(h,dict_get_intern(h,item,names,off,len,F_expires_turn),&ex); raw_expires=(float)((int)ex);
   item_derived=as_bool(h,dict_get_intern(h,item,names,off,len,F_derived))?1.f:0.f;
   double an=0; as_double(h,dict_get_intern(h,item,names,off,len,F_antique_investment),&an); antique=(float)an;
   has_item=1.f;
   int face=dict_get_intern(h,item,names,off,len,F_hand_face);
   int face_cid=dict_get_intern(h,face,names,off,len,F_card_id);
   if(face_cid!=NOT_FOUND && !is_none(h,face_cid)){
    NodeView a,b; int same=0;
    if(get_node(h,face_cid,&a)&&get_node(h,cid,&b)&&a.tag==TAG_STR&&b.tag==TAG_STR&&a.count==b.count){
     const unsigned char* pa=get_payload_ptr(h,&a); const unsigned char* pb=get_payload_ptr(h,&b);
     if(pa&&pb&&vm_memcmp(pa,pb,a.count)==0) same=1;
    }
    if(!same){
     physical=card_lookup(h,face_cid,names,off,len);
     if(physical<0){ *status=P_CARD; return 0; }
    }
   }
  }else if(code==5){
   int chid=dict_get_intern(h,action,names,off,len,F_choice_id);
   int pending=dict_get_intern(h,view,names,off,len,F_pending_choice);
   int choices=dict_get_intern(h,pending,names,off,len,F_choices);
   int item=NOT_FOUND; int nc=list_count(h,choices); if(nc<0) nc=0;
   for(int c=0;c<nc;c++){
    int entry=list_item(h,choices,c);
    int eid=dict_get_intern(h,entry,names,off,len,F_id);
    if(eid!=NOT_FOUND && key_equal(h,eid,chid)){ item=dict_get_intern(h,entry,names,off,len,F_card_id); break; }
   }
   // pending choice stores item in entry['card']
   if(item==NOT_FOUND){
    for(int c=0;c<nc;c++){
     int entry=list_item(h,choices,c);
     int eid=dict_get_intern(h,entry,names,off,len,F_id);
     if(eid!=NOT_FOUND && key_equal(h,eid,chid)){
      int card=NOT_FOUND;
      NodeView en; if(get_node(h,entry,&en) && en.tag==TAG_DICT){
       for(int k=0;k<en.count;k++){
        int key=0,val=0; get_edge(h,&en,k,&key,&val);
        NodeView kn; if(get_node(h,key,&kn)&&kn.tag==TAG_STR&&kn.count==4){
         const unsigned char* p=get_payload_ptr(h,&kn);
         if(p && vm_memcmp(p,(const unsigned char*)"card",4)==0){ card=val; break; }
        }
       }
      }
      item=card; break;
     }
    }
   }
   if(item==NOT_FOUND||is_none(h,item)){ *status=P_OBJ; return 0; }
   int cid=dict_get_intern(h,item,names,off,len,F_card_id);
   if(cid==NOT_FOUND) cid=item;
   card_kind=card_lookup(h,cid,names,off,len);
   if(card_kind<0){
    NodeView it; if(get_node(h,item,&it)&&it.tag==TAG_STR) card_kind=card_lookup(h,item,names,off,len);
   }
   if(card_kind<0){ *status=P_CARD; return 0; }
   revealed=as_bool(h,dict_get_intern(h,item,names,off,len,F_revealed))?1.f:0.f;
   is_copy=as_bool(h,dict_get_intern(h,item,names,off,len,F_copy))?1.f:0.f;
   double ex=0; as_double(h,dict_get_intern(h,item,names,off,len,F_expires_turn),&ex); raw_expires=(float)((int)ex);
   item_derived=as_bool(h,dict_get_intern(h,item,names,off,len,F_derived))?1.f:0.f;
   double an=0; as_double(h,dict_get_intern(h,item,names,off,len,F_antique_investment),&an); antique=(float)an;
   has_item=1.f;
  }else if(code==1||code==3){
   int ch=dict_get_intern(h,action,names,off,len,F_character_id);
   actor_idx=seat_lookup(h,ch,names,off,len);
   if(actor_idx<0){ *status=P_ROSTER; return 0; }
   if(!parse_target(h,dict_get_intern(h,action,names,off,len,F_target_id),viewer_b,names,off,len,&tside,&tseat,status)) return 0;
  }else if(code==4){
   int ids=dict_get_intern(h,action,names,off,len,F_card_ids);
   int ns=list_count(h,ids); if(ns<0) ns=0;
   if(ns>MAX_MULLIGAN){ *status=P_CAP; return 0; }
   int nh=list_count(h,hand); if(nh<0) nh=0;
   for(int sel=0;sel<ns;sel++){
    int inst=list_item(h,ids,sel);
    int item=NOT_FOUND;
    for(int c=0;c<nh;c++){
     int card=list_item(h,hand,c);
     int iid=dict_get_intern(h,card,names,off,len,F_instance_id);
     if(iid!=NOT_FOUND && key_equal(h,iid,inst)){ item=card; break; }
    }
    if(item==NOT_FOUND){ *status=P_OBJ; return 0; }
    int cid=dict_get_intern(h,item,names,off,len,F_card_id);
    int idx=card_lookup(h,cid,names,off,len);
    if(idx<0){ *status=P_CARD; return 0; }
    mulligan_cards[mbase+sel]=idx;
   }
  }
  int_tokens[base+1]=actor_idx; int_tokens[base+2]=card_kind; int_tokens[base+3]=tside;
  int_tokens[base+4]=tseat; int_tokens[base+5]=option_idx; int_tokens[base+6]=physical;
  float_tokens[fbase+0]=revealed; float_tokens[fbase+1]=marked; float_tokens[fbase+2]=mark_delta;
  float_tokens[fbase+3]=set_attack; float_tokens[fbase+4]=item_instant; float_tokens[fbase+5]=is_copy;
  float_tokens[fbase+6]=raw_expires; float_tokens[fbase+7]=antique;
  item_flags[ibase]=has_item; item_flags[ibase+1]=item_derived;
  kept++;
 }
 if(kept<1){ *status=P_EMPTY; return 0; }
 cand_counts[row]=kept; action_root[row]=filtered;
 return 1;
}
extern "C" __global__ void project_public_decision(
 int* nodes,int* edges,unsigned char* payload,int* counts,
 const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* outputs,
 const unsigned char* names,const int* off,const int* len,int n_names,
 float* global_tokens,int* side_int,float* side_float,int* char_present,int* char_seats,
 int* char_int,float* char_float,float* char_timed,int* zone_cards,int* mark_cards,
 int* mark_deltas,int* mark_valid,int* substituted_cards,int* substituted_valid,
 int* obs_mask,int* obs_counts,
 int* int_tokens,float* float_tokens,float* item_flags,float* turn_counts,
 int* mulligan_cards,int* cand_counts,int* cand_mask,int* action_roots,int* status,
 int rows,int maxn,int maxe,int maxp,int max_candidates){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(row>=rows) return;
 Heap h; init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,payload+(long long)row*maxp,
  counts+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 status[row]=P_OK; action_roots[row]=-1;
 int out=outputs[row];
 NodeView node; if(!get_node(&h,out,&node)||node.tag!=TAG_TUPLE||node.count!=2){ status[row]=P_OUT; return; }
 int legal=0,view=0,dummy=0; if(!get_edge(&h,&node,0,&legal,&dummy)||!get_edge(&h,&node,1,&view,&dummy)){ status[row]=P_OUT; return; }
 if(!fill_observation(&h,view,row,names,off,len,global_tokens,side_int,side_float,char_present,char_seats,
  char_int,char_float,char_timed,zone_cards,mark_cards,mark_deltas,mark_valid,substituted_cards,substituted_valid,
  obs_mask,obs_counts,status+row)) return;
 if(!fill_candidates(&h,view,legal,row,max_candidates,names,off,len,int_tokens,float_tokens,item_flags,turn_counts,
  mulligan_cards,cand_counts,cand_mask,action_roots,status+row)) return;
 if(h.error) status[row]=P_CAP;
}
extern "C" __global__ void project_public_observation(
 int* nodes,int* edges,unsigned char* payload,int* counts,
 const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* views,
 const unsigned char* names,const int* off,const int* len,int n_names,
 float* global_tokens,int* side_int,float* side_float,int* char_present,int* char_seats,
 int* char_int,float* char_float,float* char_timed,int* zone_cards,int* mark_cards,
 int* mark_deltas,int* mark_valid,int* substituted_cards,int* substituted_valid,
 int* obs_mask,int* obs_counts,int* status,
 int rows,int maxn,int maxe,int maxp){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x); if(row>=rows) return;
 Heap h; init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,payload+(long long)row*maxp,
  counts+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 status[row]=P_OK;
 if(!fill_observation(&h,views[row],row,names,off,len,global_tokens,side_int,side_float,char_present,char_seats,
  char_int,char_float,char_timed,zone_cards,mark_cards,mark_deltas,mark_valid,substituted_cards,substituted_valid,
  obs_mask,obs_counts,status+row)) return;
 if(h.error) status[row]=P_CAP;
}
'''

_DECISION_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*',
    'int32*', 'int32*', 'uint8*',
    'int32', 'int32', 'int32', 'int32*',
    'uint8*', 'int32*', 'int32*', 'int32',
    'float32*', 'int32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'float32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'int32*', 'int32*', 'int32*',
    'int32*', 'int32*',
    'int32*', 'float32*', 'float32*', 'float32*',
    'int32*', 'int32*', 'int32*', 'int32*', 'int32*',
    'int32', 'int32', 'int32', 'int32', 'int32',
)
_OBS_SPEC = (
    'int32*', 'int32*', 'uint8*', 'int32*',
    'int32*', 'int32*', 'uint8*',
    'int32', 'int32', 'int32', 'int32*',
    'uint8*', 'int32*', 'int32*', 'int32',
    'float32*', 'int32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'float32*', 'float32*', 'int32*', 'int32*',
    'int32*', 'int32*', 'int32*', 'int32*',
    'int32*', 'int32*',
    'int32*',
    'int32', 'int32', 'int32', 'int32',
)


def _zero_obs_tensors(torch, n, device):
    return {
        'global_tokens': torch.zeros((n, 9), dtype=torch.float32, device=device),
        'side_int': torch.full((n, 2, 6), -1, dtype=torch.int32, device=device),
        'side_float': torch.zeros((n, 2, 39), dtype=torch.float32, device=device),
        'char_present': torch.zeros((n, 2, MAX_PRESENT), dtype=torch.int32, device=device),
        'char_seats': torch.full((n, 2, MAX_PRESENT), -1, dtype=torch.int32, device=device),
        'char_int': torch.full((n, 2, MAX_PRESENT, 2), -1, dtype=torch.int32, device=device),
        'char_float': torch.zeros((n, 2, MAX_PRESENT, 38), dtype=torch.float32, device=device),
        'char_timed': torch.zeros((n, 2, MAX_PRESENT, MAX_TIMED, 2), dtype=torch.float32, device=device),
        'zone_cards': torch.full((n, 5, MAX_ZONE_CARDS), -1, dtype=torch.int32, device=device),
        'mark_cards': torch.full((n, MAX_HAND_CARDS), -1, dtype=torch.int32, device=device),
        'mark_deltas': torch.zeros((n, MAX_HAND_CARDS), dtype=torch.int32, device=device),
        'mark_valid': torch.zeros((n, MAX_HAND_CARDS), dtype=torch.int32, device=device),
        'substituted_cards': torch.full((n, MAX_HAND_CARDS), -1, dtype=torch.int32, device=device),
        'substituted_valid': torch.zeros((n, MAX_HAND_CARDS), dtype=torch.int32, device=device),
        'mask': torch.zeros(n, dtype=torch.int32, device=device),
        'counts': torch.zeros(n, dtype=torch.int32, device=device),
    }


def _to_observation_tokens(arrays, torch):
    return RawObservationTokens(
        global_tokens=arrays['global_tokens'],
        side_int_tokens=arrays['side_int'].to(dtype=torch.int64),
        side_float_tokens=arrays['side_float'],
        char_present=arrays['char_present'].to(dtype=torch.bool),
        char_seats=arrays['char_seats'].to(dtype=torch.int64),
        char_int_tokens=arrays['char_int'].to(dtype=torch.int64),
        char_float_tokens=arrays['char_float'],
        char_timed_attacks=arrays['char_timed'],
        zone_cards=arrays['zone_cards'].to(dtype=torch.int64),
        mark_cards=arrays['mark_cards'].to(dtype=torch.int64),
        mark_deltas=arrays['mark_deltas'].to(dtype=torch.int64),
        mark_valid=arrays['mark_valid'].to(dtype=torch.bool),
        substituted_cards=arrays['substituted_cards'].to(dtype=torch.int64),
        substituted_valid=arrays['substituted_valid'].to(dtype=torch.bool),
        mask=arrays['mask'].to(dtype=torch.bool),
        counts=arrays['counts'].to(dtype=torch.int64),
    )


@dataclass(frozen=True)
class ProjectedDecision:
    observations: object
    candidates: object
    action_roots: object
    arena: object
    errors: object = None
    action_refs: object = None
    action_types: object = None


def _masked_projection_source(source):
    decision=source[source.index('extern "C" __global__ void project_public_decision('):source.index('extern "C" __global__ void project_public_observation(')]
    observation=source[source.index('extern "C" __global__ void project_public_observation('):]
    device=[]
    for name,kernel in (("decision",decision),("observation",observation)):
        kernel=kernel.replace("void project_public_"+name+"(","void project_public_"+name+"_device(",1)
        marker="int max_candidates){" if name=="decision" else "int maxp){"
        if kernel.count(marker)!=1:raise ValueError("Unrecognized frozen projection signature")
        kernel=kernel.replace(marker,marker[:-2]+",const unsigned char* active){",1)
        kernel=kernel.replace("if(row>=rows) return;","if(row>=rows) return; if(!active[row]){status[row]=0;return;}",1)
        device.append(kernel)
    return source+"\n"+"\n".join(device)


class GraphProjectionKernel:
    """CUDA public-view/legality projector. Does not call host extractors."""

    uses_host_extractors = False
    uses_numpy_encoder = False

    def __init__(self, device='cuda', max_candidates=128, runtime=None):
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required for graph projection; no CPU fallback')
        if type(max_candidates) is not int or isinstance(max_candidates, bool) or not 1 <= max_candidates <= 16384:
            raise ValueError('Positive bounded candidate capacity required')
        from .cuda_runtime import CudaModule
        from .gpu_schema_contract import SEVEN_RUNTIME, normalize_runtime, runtime_metadata
        runtime = normalize_runtime(runtime)
        meta = runtime_metadata(runtime)
        self.torch = torch
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.max_candidates = max_candidates
        self.runtime = runtime
        self.obs_dim = meta['obs_dim']
        self.cand_dim = meta['cand_dim']
        self.schema = meta['schema']
        self.observation_schema = meta['observation_schema']
        self.schema_sha = meta['schema_sha']
        self.candidate_schema_sha = meta['candidate_schema_sha']
        self.seats = tuple(meta['seats'])
        self.card_ids = tuple(meta['card_ids'])
        self.action_types = tuple(meta['action_types'])
        self.keys = tuple(meta['keys'])
        self.prefix_obs_dim = meta['prefix_obs_dim']
        self.old_cand_dim = meta['old_cand_dim']
        if runtime == SEVEN_RUNTIME:
            from .gpu_seven_projection import (
                SEVEN_DECISION_SPEC,
                SEVEN_OBS_SPEC,
                emit_seven_projection_source,
                intern_layout_seven,
            )
            intern_names, _intern_index = intern_layout_seven()
            source = emit_seven_projection_source()
            specs = {'project_public_decision': SEVEN_DECISION_SPEC, 'project_public_observation': SEVEN_OBS_SPEC}
        else:
            intern_names = INTERN_NAMES
            source = emit_projection_source()
            specs = {'project_public_decision': _DECISION_SPEC, 'project_public_observation': _OBS_SPEC}
        self._intern_names = intern_names
        specs.update({name+"_device":list(signature)+["uint8*"] for name,signature in list(specs.items())})
        source = _masked_projection_source(source)
        self.module = CudaModule(
            source,
            specs,
            device=str(self.device),
        )
        payload, offsets, lengths = _intern_buffers(intern_names)
        self._names = torch.tensor(payload, dtype=torch.uint8, device=self.device)
        self._off = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        self._len = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        if self.runtime == 'seven':
            # Static card configuration is uploaded before any graph capture.
            from .gpu_seven_candidates import get_card_template_table
            get_card_template_table(self.device)
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.module.close(); self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _check(self, status):
        codes = status.cpu().numpy()
        bad = np.flatnonzero(codes)
        if len(bad):
            code = int(codes[int(bad[0])])
            raise GraphProjectionError(
                'GPU graph projection failed row=%s status=%s'
                % (int(bad[0]), _STATUS_NAMES.get(code, str(code)))
            )

    def project_decision(self, call, *, active=None):
        """Project public_legal_decision GPU outputs to encoder tokens on device."""
        if self.closed:
            raise RuntimeError('Graph projection kernel closed')
        device_path=active is not None
        if device_path and self.runtime != "seven":raise ValueError("Device projection requires frozen seven schema")
        if not device_path:call.check()
        arena = call.arena
        vm = call.executor
        rows = arena.rows
        torch = self.torch
        obs = _zero_obs_tensors(torch, rows, self.device)
        m = self.max_candidates
        int_tokens = torch.full((rows, m, 7), -1, dtype=torch.int32, device=self.device)
        float_tokens = torch.zeros((rows, m, 8), dtype=torch.float32, device=self.device)
        item_flags = torch.zeros((rows, m, 2), dtype=torch.float32, device=self.device)
        turn_counts = torch.zeros(rows, dtype=torch.float32, device=self.device)
        mulligan = torch.full((rows, m, MAX_MULLIGAN_SELECTIONS), -1, dtype=torch.int32, device=self.device)
        cand_counts = torch.zeros(rows, dtype=torch.int32, device=self.device)
        cand_mask = torch.zeros((rows, m), dtype=torch.int32, device=self.device)
        action_roots = torch.full((rows,), -1, dtype=torch.int32, device=self.device)
        status = torch.zeros(rows, dtype=torch.int32, device=self.device)
        if device_path:
            self._validate_device_mask(active, rows)
            active=active & (call.statuses[:,0]==0).to(torch.uint8)
        buffers = vm._buffers
        args = [arena.nodes, arena.edges, arena.payload, arena.counts,
             buffers['nodes'], buffers['edges'], buffers['payload'],
             len(vm.image.nodes), len(vm.image.edges), len(vm.image.payload), call.outputs,
             self._names, self._off, self._len, len(self._intern_names),
             obs['global_tokens'], obs['side_int'], obs['side_float'], obs['char_present'], obs['char_seats'],
             obs['char_int'], obs['char_float'], obs['char_timed'], obs['zone_cards'], obs['mark_cards'],
             obs['mark_deltas'], obs['mark_valid'], obs['substituted_cards'], obs['substituted_valid'],
             obs['mask'], obs['counts'],
             int_tokens, float_tokens, item_flags, turn_counts, mulligan, cand_counts, cand_mask,
             action_roots]
        extras = None
        cand_extras = None
        if self.runtime == 'seven':
            from .gpu_seven_projection import zero_seven_candidate_extras, zero_seven_observation_extras
            extras = zero_seven_observation_extras(torch, rows, self.device)
            cand_extras = zero_seven_candidate_extras(torch, rows, m, self.device)
            args.extend([
                cand_extras['order_cards'], cand_extras['select_cards'], cand_extras['extra_ap'],
                extras['seven_side'], extras['seven_char'], extras['side_resist'], extras['char_resist'],
                extras['choice_kind'], extras['choice_cards'],
            ])
        args.extend([status, rows, arena.layout.max_nodes, arena.layout.max_edges, arena.layout.max_payload_bytes, m])
        event = self.module.launch(
            'project_public_decision_device' if device_path else 'project_public_decision',
            args+[active] if device_path else args,
            grid=(rows + 127) // 128, block=128, capture_safe=device_path,
        )
        if not device_path:
            event.synchronize()
            self._check(status)
        if self.runtime == 'seven':
            from .gpu_seven_candidates import RawCandidateTokens as SevenCandidateTokens
            from .gpu_seven_candidates import encode_tokens as encode_seven_tokens
            from .gpu_seven_observation import encode_observation_tokens as encode_seven_observation
            observation_tokens = _to_seven_observation_tokens(obs, extras, torch)
            candidate_tokens = SevenCandidateTokens(
                int_tokens=int_tokens.to(dtype=torch.int64),
                float_tokens=float_tokens,
                item_flags=item_flags,
                turn_counts=turn_counts,
                mulligan_cards=mulligan.to(dtype=torch.int64),
                counts=cand_counts.to(dtype=torch.int64),
                mask=cand_mask.to(dtype=torch.bool),
                order_cards=cand_extras['order_cards'].to(dtype=torch.int64),
                select_cards=cand_extras['select_cards'].to(dtype=torch.int64),
                extra_ap=cand_extras['extra_ap'],
            )
            if device_path:
                from .gpu_seven_observation import encode_observation_tokens_device
                from .gpu_seven_candidates import encode_tokens_device
                observations,observation_errors=encode_observation_tokens_device(observation_tokens,device=self.device)
                candidates,candidate_errors=encode_tokens_device(candidate_tokens,device=self.device)
            else:
                observations = encode_seven_observation(observation_tokens, device=self.device)
                candidates = encode_seven_tokens(candidate_tokens, device=self.device)
        else:
            observation_tokens = _to_observation_tokens(obs, torch)
            candidate_tokens = RawCandidateTokens(
                int_tokens=int_tokens.to(dtype=torch.int64),
                float_tokens=float_tokens,
                item_flags=item_flags,
                turn_counts=turn_counts,
                mulligan_cards=mulligan.to(dtype=torch.int64),
                counts=cand_counts.to(dtype=torch.int64),
                mask=cand_mask.to(dtype=torch.bool),
            )
            observations = encode_observation_tokens(observation_tokens, device=self.device)
            candidates = encode_tokens(candidate_tokens, device=self.device)
        if device_path:
            roots=action_roots.clamp(0,arena.layout.max_nodes-1).long()
            indices=torch.arange(rows,device=self.device)
            node=arena.nodes[indices,roots]
            positions=torch.arange(m,device=self.device)[None,:]
            offsets=node[:,1].long()[:,None]+positions
            valid=candidates.mask & active[:,None].bool()
            malformed=(node[:,0]!=5)|(node[:,2]!=candidates.counts)|(node[:,1]<0)|(node[:,1]+node[:,2]>arena.counts[:,1])
            errors=status | call.statuses[:,0] | observation_errors | candidate_errors | (malformed & active.bool()).int()*8
            refs=arena.edges[indices[:,None],offsets.clamp(0,arena.layout.max_edges-1),0]
            refs=torch.where(valid,refs,torch.full_like(refs,-1))
            return ProjectedDecision(observations,candidates,action_roots,arena,errors,refs,int_tokens[:,:,0].long())
        return ProjectedDecision(observations, candidates, action_roots, arena)

    def project_observation(self, call, view_refs, *, active=None):
        """Project a generic public view graph to observation tokens on device."""
        if self.closed:
            raise RuntimeError('Graph projection kernel closed')
        device_path=active is not None
        if device_path and self.runtime != "seven":raise ValueError("Device projection requires frozen seven schema")
        if not device_path:call.check()
        arena = call.arena
        vm = call.executor
        rows = arena.rows
        torch = self.torch
        if (not isinstance(view_refs, torch.Tensor) or view_refs.device != self.device
                or view_refs.dtype != torch.int32 or view_refs.shape != (rows,)
                or not view_refs.is_contiguous()):
            raise ValueError('Contiguous CUDA int32 view refs required')
        obs = _zero_obs_tensors(torch, rows, self.device)
        status = torch.zeros(rows, dtype=torch.int32, device=self.device)
        if device_path:
            self._validate_device_mask(active, rows)
            active=active & (call.statuses[:,0]==0).to(torch.uint8)
        buffers = vm._buffers
        args = [arena.nodes, arena.edges, arena.payload, arena.counts,
             buffers['nodes'], buffers['edges'], buffers['payload'],
             len(vm.image.nodes), len(vm.image.edges), len(vm.image.payload), view_refs,
             self._names, self._off, self._len, len(self._intern_names),
             obs['global_tokens'], obs['side_int'], obs['side_float'], obs['char_present'], obs['char_seats'],
             obs['char_int'], obs['char_float'], obs['char_timed'], obs['zone_cards'], obs['mark_cards'],
             obs['mark_deltas'], obs['mark_valid'], obs['substituted_cards'], obs['substituted_valid'],
             obs['mask'], obs['counts']]
        extras = None
        if self.runtime == 'seven':
            from .gpu_seven_projection import zero_seven_observation_extras
            extras = zero_seven_observation_extras(torch, rows, self.device)
            args.extend([
                extras['seven_side'], extras['seven_char'], extras['side_resist'], extras['char_resist'],
                extras['choice_kind'], extras['choice_cards'],
            ])
        args.extend([status, rows, arena.layout.max_nodes, arena.layout.max_edges, arena.layout.max_payload_bytes])
        event = self.module.launch(
            'project_public_observation_device' if device_path else 'project_public_observation',
            args+[active] if device_path else args,
            grid=(rows + 127) // 128, block=128, capture_safe=device_path,
        )
        if not device_path:
            event.synchronize()
            self._check(status)
        if self.runtime == 'seven':
            from .gpu_seven_observation import encode_observation_tokens as encode_seven_observation
            tokens=_to_seven_observation_tokens(obs,extras,torch)
            if device_path:
                from .gpu_seven_observation import encode_observation_tokens_device
                encoded,errors=encode_observation_tokens_device(tokens,device=self.device)
                return encoded,status | call.statuses[:,0] | errors
            return encode_seven_observation(tokens, device=self.device)
        return encode_observation_tokens(_to_observation_tokens(obs, torch), device=self.device)

    def _validate_device_mask(self,active,rows):
        t=self.torch
        if (not isinstance(active,t.Tensor) or active.dtype!=t.uint8 or active.device!=self.device
                or active.shape!=(rows,) or not active.is_contiguous()):
            raise ValueError('Exact contiguous uint8[B] device activity required')


def _to_seven_observation_tokens(arrays, extras, torch):
    from .gpu_seven_observation import RawObservationTokens as SevenObservationTokens
    return SevenObservationTokens(
        global_tokens=arrays['global_tokens'],
        side_int_tokens=arrays['side_int'].to(dtype=torch.int64),
        side_float_tokens=arrays['side_float'],
        char_present=arrays['char_present'].to(dtype=torch.bool),
        char_seats=arrays['char_seats'].to(dtype=torch.int64),
        char_int_tokens=arrays['char_int'].to(dtype=torch.int64),
        char_float_tokens=arrays['char_float'],
        char_timed_attacks=arrays['char_timed'],
        zone_cards=arrays['zone_cards'].to(dtype=torch.int64),
        mark_cards=arrays['mark_cards'].to(dtype=torch.int64),
        mark_deltas=arrays['mark_deltas'].to(dtype=torch.int64),
        mark_valid=arrays['mark_valid'].to(dtype=torch.bool),
        substituted_cards=arrays['substituted_cards'].to(dtype=torch.int64),
        substituted_valid=arrays['substituted_valid'].to(dtype=torch.bool),
        seven_side=extras['seven_side'],
        seven_char=extras['seven_char'],
        side_resist=extras['side_resist'],
        char_resist=extras['char_resist'],
        choice_kind=extras['choice_kind'],
        choice_cards=extras['choice_cards'].to(dtype=torch.int64),
        mask=arrays['mask'].to(dtype=torch.bool),
        counts=arrays['counts'].to(dtype=torch.int64),
    )
