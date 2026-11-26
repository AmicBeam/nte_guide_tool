"""GPU mulligan and hand-redraw on named public-17 domain SoA tensors.

This stage consumes selections already checked against the full legal-action table.
It returns only the listed field deltas, not an independently complete state.

Selected cards leave the hand in hand order, replacements are drawn from the
deck front, then the selected cards are appended and shuffled with the formal
CPython RNG. This is not a VM, object-graph interpreter, or complete duel
engine. supports_full_duel_rules stays False.

begin_turn, finish_operation, mutating on_hand_added, empty-draw finish and
hand-overflow discard are typed continuations or closed rejects emitted before
those hooks run. player_character_count is stored identity occupancy;
player_order_slot maps lineup seats onto that storage; scalar actor/front
values are storage slots.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .domain_soa_schema import (
    catalog_fingerprint,
    CHARACTER_BOOL_FIELDS,
    CHARACTER_VOCAB,
    CHOICE_KINDS,
    GAME_PHASES,
    PRESENCE_NULL,
    ZONES,
    supports_full_duel_rules as _SCHEMA_FULL,
)
from .random_source import emit_random_source

supports_full_duel_rules = False
if _SCHEMA_FULL:
    raise RuntimeError('domain_soa_schema must not claim full duel coverage')

KIND_MULLIGAN, KIND_REDRAW = 0, 1
HAND, DECK = ZONES.index('hand'), ZONES.index('deck')
PHASE_MULLIGAN = GAME_PHASES.index('mulligan')
PHASE_CHOICE = GAME_PHASES.index('choice')
PHASE_PLAYING = GAME_PHASES.index('playing')
REDRAW_KIND = CHOICE_KINDS.index('hand_redraw')
AWAKENED = CHARACTER_BOOL_FIELDS.index('awakened')
XIAOZHI = CHARACTER_VOCAB['xiaozhi']
MAX_SELECT, HAND_LIMIT, MAX_ZONE_STACK = 3, 10, 96

INVALID_INPUT, INVALID_ID, DUPLICATE_SELECTION = 1, 2, 3
CAPACITY, RNG_EXHAUSTED, UNKNOWN_HOOK, EMPTY_DRAW = 4, 5, 6, 7
CONT_NONE, CONT_BEGIN_TURN, CONT_ON_HAND_ADDED, CONT_FINISH_OPERATION = 0, 1, 2, 3

REQUIRED = (
    'catalog_digest', 'row_error', 'zone_count', 'zone_card_index', 'rng', 'mulligan_count', 'mulligan_sides',
    'version', 'current_action_version', 'combat_group', 'event_seq', 'phase',
    'pending_choice_present', 'row_valid', 'first_side', 'choice_kind', 'choice_side',
    'card_instance_slot', 'player_character_count', 'player_order_count',
    'player_order_slot', 'player_front_presence', 'player_front_slot',
    'character_valid', 'character_template', 'character_hp',
    'character_bool_presence', 'character_bool_value',
)
UNRESOLVED_HOOKS = (
    'begin_turn and nested turn-start hooks after the second mulligan',
    'on_hand_added when awakened xiaozhi or an unknown kit would mutate a draw',
    'finish_operation after hand_redraw zone/RNG replacement',
    'empty-draw finish/victory and hand-overflow discard',
    'presentation logs and current_action_id string intern',
    'mimic/weapon on_hand_added, concede, tutorial, legal-action enumeration',
)
UNRESOLVED_FIELDS = (
    'current_action_id_slot',
    'events / _public_board',
    'turn-start clocks mutated only by begin_turn',
    'choice payload fields other than pending_choice_present',
    'player_front is storage-referenced and left unchanged by this stage',
)


def coverage():
    return MappingProxyType({
        'supports_full_duel_rules': False,
        'seven_official_presets': True,
        'full_state_transition': False,
        'legal_action_validation': False,
        'stage_only': True,
        'implemented': (
            'opening mulligan on named domain SoA tensors',
            'hand_redraw compact/draw/return/shuffle',
            'selected cards removed in hand order',
            'replacements drawn before selected cards return',
            'formal CPython RNG shuffle',
            'mulligan_done append in completion order',
            'second mulligan emits begin_turn continuation without running it',
            'player_character_count storage identities, player_order_slot lineup map',
        ),
        'unresolved_hooks': UNRESOLVED_HOOKS,
        'unresolved_state_fields': UNRESOLVED_FIELDS,
    })


def turns_source():
    prefix = r'''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define SOA_FN __device__
#else
#define SOA_FN
#endif
#define BMT_DEVICE SOA_FN
#define SOA_TURN_ZONE 96
#define SOA_TURN_SELECT 3
'''
    body = r'''
SOA_FN int rng_below(bmt_state_t* rng, unsigned long long n, int max_draws, unsigned long long* out){
 if(n==0ULL)return 2;
 int k=bmt_bit_length_u64(n);
 for(int a=0;a<max_draws;a++){uint64_t r=0;bmt_getbits64(rng,k,&r);if(r<n){*out=r;return 0;}}
 return 2;
}
SOA_FN int rng_shuffle(bmt_state_t* rng, int* a, int n, int max_draws){
 for(int i=n-1;i>0;i--){
  unsigned long long r=0;int st=rng_below(rng,(unsigned long long)i+1ULL,max_draws,&r);if(st)return st;
  int j=(int)r,t=a[i];a[i]=a[j];a[j]=t;
 }
 return 0;
}
SOA_FN void soa_turn_row(
 const short* zone_count,const int* zone_index,const int* card_inst,const long long* rng,
 const short* phase,const signed char* mcount,const signed char* msides,
 const int* version,const int* action_version,const int* combat_group,const int* event_seq,
 const unsigned char* pending,const short* choice_kind,const signed char* choice_side,
 const signed char* first_side,const unsigned char* row_valid,
 const short* stored_n,const short* order_n,const short* order_slot,
 const unsigned char* cvalid,const short* ctmpl,const int* chp,
 const unsigned char* cbool_p,const unsigned char* cbool_v,
 const int* side_in,const int* select_count,const int* select_ids,const int* kind,const unsigned char* active,
 short* o_zone_count,int* o_zone_index,long long* o_rng,signed char* o_mcount,signed char* o_msides,
 int* o_version,int* o_action_version,int* o_combat,int* o_event,short* o_phase,unsigned char* o_pending,
 int* errors,int* cont_kind,int* cont_side,int* cont_actor,const int* cfg,int row){
 int rows=cfg[0],Z=cfg[1],K=cfg[2],C=cfg[3],n_cbool=cfg[4],max_select=cfg[5],max_draws=cfg[6];
 int xiaozhi=cfg[7],awakened=cfg[8],redraw_kind=cfg[9];
 int s,z,p,i,j,side,sel_n,hand_n,deck_n,err=0,cont=0,actor=-1,cs,kind_v,ids[SOA_TURN_SELECT],pos[SOA_TURN_SELECT];
 int hand[SOA_TURN_ZONE],deck[SOA_TURN_ZONE],kept[SOA_TURN_ZONE],taken[SOA_TURN_ZONE],drawn[SOA_TURN_ZONE],out_deck[SOA_TURN_ZONE];
 int kept_n=0,taken_n=0,drawn_n=0,rest_n=0,new_hand_n=0,new_deck_n=0,need_shuffle=0,hook=0,slot,base,aw_p,aw_v;
 long long next_rng; bmt_state_t rng_state; uint32_t words[2]; uint64_t bits=0; unsigned long long raw;
 for(s=0;s<2;s++)for(z=0;z<4;z++){
  o_zone_count[(row*2+s)*4+z]=zone_count[(row*2+s)*4+z];
  for(p=0;p<Z;p++)o_zone_index[((row*2+s)*4+z)*Z+p]=zone_index[((row*2+s)*4+z)*Z+p];
 }
 o_rng[row]=rng[row];o_mcount[row]=mcount[row];o_msides[row*2]=msides[row*2];o_msides[row*2+1]=msides[row*2+1];
 o_version[row]=version[row];o_action_version[row]=action_version[row];o_combat[row]=combat_group[row];o_event[row]=event_seq[row];
 o_phase[row]=phase[row];o_pending[row]=pending[row];
 errors[row]=0;cont_kind[row]=0;cont_side[row]=0;cont_actor[row]=-1;
 if(!active[row])return;
 side=side_in[row];sel_n=select_count[row];kind_v=kind[row];
 if(!row_valid[row]||side<0||side>1||sel_n<0||sel_n>max_select||sel_n>SOA_TURN_SELECT||Z<10||Z>SOA_TURN_ZONE||K<1||C<1||max_select<1||max_select>SOA_TURN_SELECT){errors[row]=1;return;}
 if(kind_v!=0&&kind_v!=1){errors[row]=1;return;}
 hand_n=(int)zone_count[(row*2+side)*4+0];deck_n=(int)zone_count[(row*2+side)*4+1];
 if(hand_n<0||hand_n>Z||deck_n<0||deck_n>Z){errors[row]=1;return;}
 if(kind_v==0){
  if((int)phase[row]!=0||(int)mcount[row]<0||(int)mcount[row]>1){errors[row]=1;return;}
  if(((int)mcount[row]>0&&(int)msides[row*2]==side)||((int)mcount[row]>1&&(int)msides[row*2+1]==side)){errors[row]=1;return;}
 }else{
  if((int)phase[row]!=2||!pending[row]||(int)choice_kind[row]!=redraw_kind||(int)choice_side[row]!=side){errors[row]=1;return;}
 }
 for(i=0;i<sel_n;i++){
  ids[i]=select_ids[row*max_select+i];pos[i]=-1;
  if(ids[i]<1){err=2;break;}
  for(j=0;j<i;j++)if(ids[j]==ids[i]){err=3;break;}
  if(err)break;
  for(p=0;p<hand_n;p++){
   int ci=zone_index[((row*2+side)*4+0)*Z+p];
   if(ci<0||ci>=K){err=2;break;}
   if(card_inst[row*K+ci]==ids[i]){pos[i]=p;break;}
  }
  if(err)break;
  if(pos[i]<0){err=2;break;}
 }
 if(err){errors[row]=err;return;}
 for(i=0;i<sel_n;i++)for(j=i+1;j<sel_n;j++)if(pos[j]<pos[i]){
  int tp=pos[i],ti=ids[i];pos[i]=pos[j];ids[i]=ids[j];pos[j]=tp;ids[j]=ti;
 }
 for(p=0;p<hand_n;p++){
  int marked=0;for(i=0;i<sel_n;i++)if(pos[i]==p)marked=1;
  int ci=zone_index[((row*2+side)*4+0)*Z+p];
  if(ci<0||ci>=K){errors[row]=2;return;}
  if(card_inst[row*K+ci]<1){errors[row]=2;return;}
  for(j=0;j<p;j++)if(hand[j]==ci){errors[row]=2;return;}
  hand[p]=ci;if(marked)taken[taken_n++]=ci;else kept[kept_n++]=ci;
 }
 for(p=0;p<deck_n;p++){
  int ci=zone_index[((row*2+side)*4+1)*Z+p];
  if(ci<0||ci>=K){errors[row]=2;return;}
  if(card_inst[row*K+ci]<1){errors[row]=2;return;}
  for(j=0;j<hand_n;j++)if(hand[j]==ci){errors[row]=2;return;}
  for(j=0;j<p;j++)if(deck[j]==ci){errors[row]=2;return;}
  deck[p]=ci;
 }
 if(sel_n>0){
  int occ=(int)stored_n[row*2+side];
  if(occ<0||occ>C){errors[row]=1;return;}
  for(slot=0;slot<occ;slot++){
   base=(row*2+side)*C+slot;
   if(!cvalid[base]||chp[base]<=0)continue;
   aw_p=(int)cbool_p[base*n_cbool+awakened];aw_v=(int)cbool_v[base*n_cbool+awakened];
   if((int)ctmpl[base]==xiaozhi&&aw_p>1&&aw_v){hook=1;if(actor<0)actor=slot;}
  }
  if(hook){
   errors[row]=6;cont_kind[row]=2;cont_side[row]=side;cont_actor[row]=actor;return;
  }
 }
 if(sel_n>deck_n){errors[row]=7;return;}
 if(kept_n+sel_n>10||deck_n-sel_n+sel_n>Z){errors[row]=4;return;}
 for(i=0;i<sel_n;i++)drawn[drawn_n++]=deck[i];
 for(p=sel_n;p<deck_n;p++)out_deck[rest_n++]=deck[p];
 for(i=0;i<taken_n;i++)out_deck[rest_n+i]=taken[i];
 new_hand_n=kept_n+drawn_n;new_deck_n=rest_n+taken_n;
 need_shuffle=(kind_v==0)||(sel_n>0);
 next_rng=rng[row];
 if(need_shuffle){
  if(rng[row]<0){errors[row]=1;return;}
  raw=(unsigned long long)rng[row];words[0]=(uint32_t)raw;words[1]=(uint32_t)(raw>>32);
  bmt_seed_words(&rng_state,words,words[1]?2:1);
  cs=rng_shuffle(&rng_state,out_deck,new_deck_n,max_draws);
  if(cs){errors[row]=5;return;}
  bits=0;bmt_getbits64(&rng_state,63,&bits);next_rng=(long long)bits;
 }
 for(p=0;p<new_hand_n;p++)o_zone_index[((row*2+side)*4+0)*Z+p]=(p<kept_n?kept[p]:drawn[p-kept_n]);
 for(p=new_hand_n;p<hand_n;p++)o_zone_index[((row*2+side)*4+0)*Z+p]=0;
 for(p=0;p<new_deck_n;p++)o_zone_index[((row*2+side)*4+1)*Z+p]=out_deck[p];
 for(p=new_deck_n;p<deck_n;p++)o_zone_index[((row*2+side)*4+1)*Z+p]=0;
 o_zone_count[(row*2+side)*4+0]=(short)new_hand_n;
 o_zone_count[(row*2+side)*4+1]=(short)new_deck_n;
 o_rng[row]=next_rng;
 o_version[row]=version[row]+1;o_action_version[row]=o_version[row];o_combat[row]=0;
 o_event[row]=event_seq[row]+1+sel_n;
 if(kind_v==0){
  o_mcount[row]=(signed char)((int)mcount[row]+1);
  o_msides[row*2+((int)mcount[row])]= (signed char)side;
  if((int)o_mcount[row]==2){cont=1;cont_side[row]=(int)first_side[row];}
  else cont_side[row]=side;
 }else{
  o_phase[row]=(short)1;o_pending[row]=0;cont=3;cont_side[row]=side;
 }
 int actor_side=(cont==1)?(int)first_side[row]:side;
 if(actor_side<0||actor_side>1){errors[row]=1;return;}
 if((int)order_n[row*2+actor_side]>0)actor=(int)order_slot[(row*2+actor_side)*C+0];
 if(cont==1||cont==3)cont_actor[row]=actor;
 errors[row]=0;cont_kind[row]=cont;
}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void domain_soa_turns(
 const short* zone_count,const int* zone_index,const int* card_inst,const long long* rng,
 const short* phase,const signed char* mcount,const signed char* msides,
 const int* version,const int* action_version,const int* combat_group,const int* event_seq,
 const unsigned char* pending,const short* choice_kind,const signed char* choice_side,
 const signed char* first_side,const unsigned char* row_valid,
 const short* stored_n,const short* order_n,const short* order_slot,
 const unsigned char* cvalid,const short* ctmpl,const int* chp,
 const unsigned char* cbool_p,const unsigned char* cbool_v,
 const int* side_in,const int* select_count,const int* select_ids,const int* kind,const unsigned char* active,
 short* o_zone_count,int* o_zone_index,long long* o_rng,signed char* o_mcount,signed char* o_msides,
 int* o_version,int* o_action_version,int* o_combat,int* o_event,short* o_phase,unsigned char* o_pending,
 int* errors,int* cont_kind,int* cont_side,int* cont_actor,const int* cfg){
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);
 if(row<cfg[0])soa_turn_row(zone_count,zone_index,card_inst,rng,phase,mcount,msides,version,action_version,combat_group,event_seq,pending,choice_kind,choice_side,first_side,row_valid,stored_n,order_n,order_slot,cvalid,ctmpl,chp,cbool_p,cbool_v,side_in,select_count,select_ids,kind,active,o_zone_count,o_zone_index,o_rng,o_mcount,o_msides,o_version,o_action_version,o_combat,o_event,o_phase,o_pending,errors,cont_kind,cont_side,cont_actor,cfg,row);
}
#else
extern "C" void domain_soa_turns(
 const short* zone_count,const int* zone_index,const int* card_inst,const long long* rng,
 const short* phase,const signed char* mcount,const signed char* msides,
 const int* version,const int* action_version,const int* combat_group,const int* event_seq,
 const unsigned char* pending,const short* choice_kind,const signed char* choice_side,
 const signed char* first_side,const unsigned char* row_valid,
 const short* stored_n,const short* order_n,const short* order_slot,
 const unsigned char* cvalid,const short* ctmpl,const int* chp,
 const unsigned char* cbool_p,const unsigned char* cbool_v,
 const int* side_in,const int* select_count,const int* select_ids,const int* kind,const unsigned char* active,
 short* o_zone_count,int* o_zone_index,long long* o_rng,signed char* o_mcount,signed char* o_msides,
 int* o_version,int* o_action_version,int* o_combat,int* o_event,short* o_phase,unsigned char* o_pending,
 int* errors,int* cont_kind,int* cont_side,int* cont_actor,const int* cfg){
 for(int row=0;row<cfg[0];row++)soa_turn_row(zone_count,zone_index,card_inst,rng,phase,mcount,msides,version,action_version,combat_group,event_seq,pending,choice_kind,choice_side,first_side,row_valid,stored_n,order_n,order_slot,cvalid,ctmpl,chp,cbool_p,cbool_v,side_in,select_count,select_ids,kind,active,o_zone_count,o_zone_index,o_rng,o_mcount,o_msides,o_version,o_action_version,o_combat,o_event,o_phase,o_pending,errors,cont_kind,cont_side,cont_actor,cfg,row);
}
#endif
'''
    return prefix + emit_random_source() + body


@dataclass(frozen=True)
class TurnFields:
    zone_count: object
    zone_card_index: object
    rng: object
    mulligan_count: object
    mulligan_sides: object
    version: object
    current_action_version: object
    combat_group: object
    event_seq: object
    phase: object
    pending_choice_present: object
    errors: object
    continuation_kind: object
    continuation_side: object
    continuation_actor: object


@dataclass(frozen=True)
class PreparedTurns:
    owner: object
    inputs: tuple
    outputs: tuple
    cfg: object
    result: TurnFields
    sources: tuple
    inherited_error: object
    catalog_source: object
    catalog_expected: object


def _columns(fields):
    if isinstance(fields, Mapping):
        return fields
    arrays = getattr(fields, 'arrays', None)
    if isinstance(arrays, Mapping):
        return arrays
    raise TypeError('Named domain SoA tensors required')


def _as_bytes(value):
    import torch
    if value.dtype == torch.bool:
        return value.view(torch.uint8)
    return value


class DomainSoATurns:
    supports_full_duel_rules = False
    gpu_qualified = False

    def __init__(self, *, device='cuda', allow_cpu_oracle=False, max_draws=4096,
                 max_select=MAX_SELECT):
        import torch
        if type(max_draws) is not int or not 1 <= max_draws <= 1048576:
            raise ValueError('Bounded RNG rejection budget required')
        if type(max_select) is not int or not 1 <= max_select <= MAX_SELECT:
            raise ValueError('Opening mulligan selects at most three cards')
        self.t = torch
        self.device = torch.device(device)
        self.max_draws = max_draws
        self.max_select = max_select
        self._temporary = None
        self._module = None
        self._library = None
        self.closed = False
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no implicit CPU turns fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            from .cuda_runtime import CudaModule
            sig = (['int16*', 'int32*', 'int32*', 'int64*', 'int16*', 'int8*', 'int8*',
                    'int32*', 'int32*', 'int32*', 'int32*', 'uint8*', 'int16*', 'int8*',
                    'int8*', 'uint8*', 'int16*', 'int16*', 'int16*', 'uint8*', 'int16*',
                    'int32*', 'uint8*', 'uint8*', 'int32*', 'int32*', 'int32*', 'int32*',
                    'uint8*', 'int16*', 'int32*', 'int64*', 'int8*', 'int8*', 'int32*',
                    'int32*', 'int32*', 'int32*', 'int16*', 'uint8*', 'int32*', 'int32*',
                    'int32*', 'int32*', 'int32*'])
            self._module = CudaModule(turns_source(), {'domain_soa_turns': [x if x not in ('int16*', 'int8*') else 'uint8*' for x in sig]}, device=str(self.device))
        elif self.device.type == 'cpu' and allow_cpu_oracle is True:
            compiler = shutil.which('clang++') or shutil.which('g++')
            if compiler is None:
                raise RuntimeError('Native CPU SoA turns require a C++ compiler')
            self._temporary = tempfile.TemporaryDirectory(prefix='nte-soa-turns-')
            root = Path(self._temporary.name)
            source = root / 'turns.cpp'
            import sys
            library = root / ('turns.dll' if os.name == 'nt' else
                              'turns.dylib' if sys.platform == 'darwin' else 'turns.so')
            source.write_text(turns_source(), encoding='utf-8')
            try:
                compiled = subprocess.run(
                    [compiler, '-std=c++17', '-shared', '-fPIC', '-O2', str(source), '-o', str(library)],
                    capture_output=True, text=True, timeout=60)
            except BaseException:
                self.close()
                raise
            if compiled.returncode:
                self.close()
                raise RuntimeError('Native SoA turns compilation failed: ' + compiled.stderr)
            self._library = ctypes.CDLL(str(library))
            self._kernel = self._library.domain_soa_turns
            self._kernel.argtypes = [ctypes.c_void_p] * 45
            self._kernel.restype = None
        else:
            raise ValueError('CUDA or explicitly enabled CPU oracle required')

    def prepare(self, fields, side, select_count, select_ids, kind, active):
        import torch
        if self.closed:
            raise RuntimeError('SoA turns backend is closed')
        if self.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare turn buffers before CUDA capture')
        cols = _columns(fields)
        missing = [name for name in REQUIRED if name not in cols]
        if missing:
            raise ValueError('Complete named schema required; missing ' + ','.join(missing))
        phase = cols['phase']
        if (not isinstance(phase, torch.Tensor) or phase.device != self.device
                or phase.dtype != torch.int16 or phase.ndim != 1):
            raise ValueError('Resident named tensor required: phase')
        rows = int(phase.shape[0])
        if rows < 1:
            raise ValueError('Nonempty bounded batch required')
        zone = cols['zone_card_index']
        counts = cols['zone_count']
        if (not isinstance(zone, torch.Tensor) or zone.dtype != torch.int32 or zone.ndim != 4
                or zone.shape[0] != rows or tuple(zone.shape[1:3]) != (2, 4)
                or zone.device != self.device):
            raise ValueError('Named int32 zone_card_index[B,2,4,Z] required')
        width = int(zone.shape[3])
        if not HAND_LIMIT <= width <= MAX_ZONE_STACK:
            raise ValueError('Zone width must cover the formal hand limit and stack bound')
        if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int16
                or counts.shape != (rows, 2, 4) or counts.device != self.device):
            raise ValueError('Named int16 zone_count[B,2,4] required')
        stored = cols['player_character_count']
        order_n = cols['player_order_count']
        order_slot = cols['player_order_slot']
        chars = cols['character_valid']
        if chars.ndim != 3 or chars.shape[0] != rows or chars.shape[1] != 2:
            raise ValueError('Named character storage [B,2,C] required')
        cap_c = int(chars.shape[2])
        if (stored.dtype != torch.int16 or stored.shape != (rows, 2)
                or order_n.dtype != torch.int16 or order_n.shape != (rows, 2)
                or order_slot.dtype != torch.int16 or order_slot.shape != (rows, 2, cap_c)
                or stored.device != self.device or order_n.device != self.device
                or order_slot.device != self.device):
            raise ValueError('Storage occupancy and lineup slot map required')
        inst = cols['card_instance_slot']
        if inst.ndim != 2 or inst.shape[0] != rows or inst.dtype != torch.int32:
            raise ValueError('Named card_instance_slot[B,K] required')
        cap_k = int(inst.shape[1])
        n_cbool = int(cols['character_bool_presence'].shape[-1])
        for name, dtype in (
                ('rng', torch.int64), ('mulligan_count', torch.int8),
                ('version', torch.int32), ('current_action_version', torch.int32),
                ('combat_group', torch.int32), ('event_seq', torch.int32),
                ('first_side', torch.int8), ('choice_kind', torch.int16),
                ('choice_side', torch.int8)):
            value = cols[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype or value.shape != (rows,)
                    or value.device != self.device):
                raise ValueError('Resident named tensor required: ' + name)
        msides = cols['mulligan_sides']
        if (msides.dtype != torch.int8 or msides.shape != (rows, 2)
                or msides.device != self.device):
            raise ValueError('Named mulligan_sides[B,2] required')
        for name in ('pending_choice_present', 'row_valid', 'character_valid',
                     'character_bool_value'):
            value = cols[name]
            if (not isinstance(value, torch.Tensor) or value.device != self.device
                   ):
                raise ValueError('Resident named tensor required: ' + name)
        if (not isinstance(side, torch.Tensor) or side.dtype != torch.int32 or side.shape != (rows,)
                or side.device != self.device or not side.is_contiguous()):
            raise ValueError('Resident int32 side[B] required')
        if (not isinstance(select_count, torch.Tensor) or select_count.dtype != torch.int32
                or select_count.shape != (rows,) or select_count.device != self.device
                or not select_count.is_contiguous()):
            raise ValueError('Resident int32 select_count[B] required')
        if (not isinstance(select_ids, torch.Tensor) or select_ids.dtype != torch.int32
                or select_ids.ndim != 2 or select_ids.shape[0] != rows
                or int(select_ids.shape[1]) != self.max_select or select_ids.device != self.device
                or not select_ids.is_contiguous()):
            raise ValueError('Resident int32 select_ids[B,<=3] required')
        if (not isinstance(kind, torch.Tensor) or kind.dtype != torch.int32 or kind.shape != (rows,)
                or kind.device != self.device or not kind.is_contiguous()):
            raise ValueError('Resident int32 kind[B] required')
        if (not isinstance(active, torch.Tensor) or active.dtype != torch.bool or active.shape != (rows,)
                or active.device != self.device or not active.is_contiguous()):
            raise ValueError('Resident bool active[B] required')
        # C pointers are untyped: validate every read before allocating launch buffers.
        shapes = {
            'catalog_digest': (torch.uint8, (rows, 32)),
            'row_error': (torch.int32, (rows,)),
            'pending_choice_present': (torch.bool, (rows,)),
            'row_valid': (torch.bool, (rows,)),
            'character_valid': (torch.bool, (rows, 2, cap_c)),
            'character_template': (torch.int16, (rows, 2, cap_c)),
            'character_hp': (torch.int32, (rows, 2, cap_c)),
            'character_bool_presence': (torch.uint8, (rows, 2, cap_c, len(CHARACTER_BOOL_FIELDS))),
            'character_bool_value': (torch.bool, (rows, 2, cap_c, len(CHARACTER_BOOL_FIELDS))),
            'card_instance_slot': (torch.int32, (rows, cap_k)),
        }
        for name, (dtype, shape) in shapes.items():
            value = cols[name]
            if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape or value.device != self.device:
                raise ValueError('Resident named tensor required: ' + name)
        result = TurnFields(
            zone_count=torch.empty_like(counts),
            zone_card_index=torch.empty_like(zone),
            rng=torch.empty_like(cols['rng']),
            mulligan_count=torch.empty_like(cols['mulligan_count']),
            mulligan_sides=torch.empty_like(msides),
            version=torch.empty_like(cols['version']),
            current_action_version=torch.empty_like(cols['current_action_version']),
            combat_group=torch.empty_like(cols['combat_group']),
            event_seq=torch.empty_like(cols['event_seq']),
            phase=torch.empty_like(phase),
            pending_choice_present=torch.empty_like(cols['pending_choice_present']),
            errors=torch.empty(rows, dtype=torch.int32, device=self.device),
            continuation_kind=torch.empty(rows, dtype=torch.int32, device=self.device),
            continuation_side=torch.empty(rows, dtype=torch.int32, device=self.device),
            continuation_actor=torch.empty(rows, dtype=torch.int32, device=self.device),
        )
        cfg = torch.tensor(
            [rows, width, cap_k, cap_c, n_cbool, self.max_select, self.max_draws,
             XIAOZHI, AWAKENED, REDRAW_KIND], dtype=torch.int32, device=self.device)
        inputs = (
            counts, zone, inst, cols['rng'], phase, cols['mulligan_count'], msides,
            cols['version'], cols['current_action_version'], cols['combat_group'], cols['event_seq'],
            _as_bytes(cols['pending_choice_present']), cols['choice_kind'], cols['choice_side'],
            cols['first_side'], _as_bytes(cols['row_valid']), stored, order_n, order_slot,
            _as_bytes(chars), cols['character_template'], cols['character_hp'],
            cols['character_bool_presence'], _as_bytes(cols['character_bool_value']),
            side, select_count, select_ids, kind, _as_bytes(active),
        )
        outputs = (
            result.zone_count, result.zone_card_index, result.rng, result.mulligan_count,
            result.mulligan_sides, result.version, result.current_action_version,
            result.combat_group, result.event_seq, result.phase,
            _as_bytes(result.pending_choice_present), result.errors, result.continuation_kind,
            result.continuation_side, result.continuation_actor,
        )
        sources = inputs
        inputs = tuple(item.clone(memory_format=torch.contiguous_format) for item in sources)
        inherited_error = cols['row_error']
        return PreparedTurns(self, inputs, outputs, cfg, result, sources, inherited_error, cols['catalog_digest'], torch.tensor(list(catalog_fingerprint()), dtype=torch.uint8, device=self.device))

    def launch(self, batch):
        if self.closed:
            raise RuntimeError('SoA turns backend is closed')
        if not isinstance(batch, PreparedTurns) or batch.owner is not self:
            raise ValueError('Prepared buffers must belong to this turns backend')
        for source, target in zip(batch.sources, batch.inputs):
            target.copy_(source)
        import torch
        args = batch.inputs + batch.outputs + (batch.cfg,)
        if self._module is not None:
            self._module.launch(
                'domain_soa_turns', [x.view(torch.uint8) if x.dtype in (torch.int16, torch.int8) else x for x in args],
                grid=((int(batch.result.errors.shape[0]) + 63) // 64,), block=(64,),
                capture_safe=True)
        else:
            self._kernel(*(ctypes.c_void_p(item.data_ptr()) for item in args))
        # Failed rows are transactional, including upstream errors. Do not silently
        # turn a previously failed state into a valid transition.
        import torch
        result = batch.result
        result.errors.copy_(torch.where(batch.inherited_error != 0, batch.inherited_error, result.errors))
        stale = (batch.catalog_source != batch.catalog_expected[None, :]).any(1)
        active = batch.sources[-1] != 0
        result.errors.copy_(torch.where(stale & active & (result.errors == 0), INVALID_INPUT, result.errors))
        failed = result.errors != 0
        inherited = batch.inherited_error != 0
        result.continuation_kind.masked_fill_(inherited, CONT_NONE)
        result.continuation_side.masked_fill_(inherited, 0)
        result.continuation_actor.masked_fill_(inherited, -1)
        original = (batch.inputs[0], batch.inputs[1], batch.inputs[3], batch.inputs[5],
                    batch.inputs[6], batch.inputs[7], batch.inputs[8], batch.inputs[9],
                    batch.inputs[10], batch.inputs[4], batch.inputs[11])
        for target, source in zip(batch.outputs[:11], original):
            mask = failed.reshape((failed.shape[0],) + (1,) * (target.ndim - 1))
            target.copy_(torch.where(mask, source, target))
        return result

    def apply(self, fields, side, select_count, select_ids, kind, active):
        return self.launch(self.prepare(fields, side, select_count, select_ids, kind, active))

    def close(self):
        if self.closed:
            return
        if self._module is not None:
            self._module.close()
            self._module = None
        if self._library is not None:
            import _ctypes
            (_ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose)(self._library._handle)
            self._library = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
