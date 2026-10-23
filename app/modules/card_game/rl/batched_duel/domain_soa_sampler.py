"""Direct CUDA public-count determinization on named domain SoA arrays.

This is the seven_sampler allocation / recovery / generation path, not a toy
shuffle and not a VM. Incomplete schema fields fail closed. CPU is an explicit
oracle compile of the same source. supports_full_duel_rules stays False.
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

from app.modules.card_game.content.duel_v2.tutorial import lookup_card
from app.modules.card_game.rl import seven_lineup
from .domain_soa_schema import (
    CARD_BOOL_FIELDS,
    CARD_INT_FIELDS,
    CARD_STR_FIELDS,
    CARD_VOCAB,
    CHARACTER_BOOL_FIELDS,
    CHARACTER_VOCAB,
    CHOICE_KINDS,
    DomainSoALayout,
    PLAYER_INT_FIELDS,
    PRESENCE_ABSENT,
    PRESENCE_CATALOG,
    PRESENCE_NULL,
    array_specs,
    catalog_fingerprint,
    supports_full_duel_rules as _SCHEMA_FULL,
)
from .random_source import emit_random_source

supports_full_duel_rules = False
if _SCHEMA_FULL:
    raise RuntimeError('domain_soa_schema must not claim full duel coverage')

INVALID_INPUT = 1
RNG_EXHAUSTED = 2
UNSUPPORTED_ROOT = 3
INCONSISTENT_FAMILY = 4
NEGATIVE_SURPLUS = 5
NO_PUBLIC_ALLOC = 6
NO_HAND_ALLOC = 7
UNENCODED_HERO = 8
ILLEGAL_ORIGINAL = 9
IMPOSSIBLE_ORIGINAL = 10
POOL_SIZE = 11
GENERATED_EXCEED_HAND = 12
MISSING_FIELD = 13
CAPACITY = 14
PARTICLE_CHANGED = 15
SCHEMA_INCOMPLETE = 16
PUBLIC_GUARD_UNAVAILABLE = 17  # Reserved legacy code; the default now runs the complete checker.

PHASE_PLAYING, PHASE_CHOICE = 1, 2
ZONE_HAND, ZONE_DECK, ZONE_DISCARD = 0, 1, 2
KINT_EPH, KINT_EXPIRE = CARD_INT_FIELDS.index('ephemeral_turn'), CARD_INT_FIELDS.index('expires_turn')
KBOOL_COPY, KBOOL_DERIVED = CARD_BOOL_FIELDS.index('copy'), CARD_BOOL_FIELDS.index('derived')
KSTR_CHAR, KSTR_ORIG = CARD_STR_FIELDS.index('character_id'), CARD_STR_FIELDS.index('original_character_id')
CBOOL_AWAKENED, CBOOL_SUMMONED = CHARACTER_BOOL_FIELDS.index('awakened'), CHARACTER_BOOL_FIELDS.index('summoned')
PINT_FAMILY = PLAYER_INT_FIELDS.index('nanali_family')
N_KINT, N_KBOOL, N_KSTR = len(CARD_INT_FIELDS), len(CARD_BOOL_FIELDS), len(CARD_STR_FIELDS)
N_CBOOL, N_PINT = len(CHARACTER_BOOL_FIELDS), len(PLAYER_INT_FIELDS)
VOCAB_CARDS = max(CARD_VOCAB.values())
VOCAB_CHARS = max(CHARACTER_VOCAB.values())
KIND_Q05, KIND_C02, KIND_AF01, KIND_A03E, KIND_PF01, KIND_FF01 = range(6)
SUPPORTED_CHOICE = frozenset(
    CHOICE_KINDS.index(kind) for kind in
    ('discard', 'inspect_top', 'enemy_hand', 'kit', 'kit_multi', 'hand_redraw', 'heiyu_divination', 'hand_deck_placement')
)
WRITABLE = (
    'zone_count', 'zone_card_index',
    'card_valid', 'card_entity_slot', 'card_instance_slot', 'card_template', 'card_side',
    'card_int_presence', 'card_int_value', 'card_bool_presence', 'card_bool_value',
    'card_str_presence', 'card_str_slot', 'card_source_presence', 'card_play_options_presence',
    'card_jingu_mark_presence', 'card_jingu_delta', 'card_jingu_source_slot',
    'next_instance', 'next_entity', 'rng', 'row_error',
    'string_count', 'string_bytes_used', 'string_offset', 'string_length', 'string_bytes',
)
REQUIRED = (
    'phase','turn','active_side','first_side','escalation_enabled',
    'pending_choice_present','choice_kind','choice_side','choice_card_count',
    'choice_card_index','operation_present','op_card_index','player_valid',
    'player_entity_slot','player_character_count','player_int_presence',
    'player_int_value','character_valid','character_template',
    'character_entity_slot','character_hp','character_bool_presence',
    'character_bool_value','zone_count','player_revealed_count',
    'player_revealed_side','player_revealed_instance','card_is_view','card_physical_index',
)
MISSING_STAGES = (
    'nested, invalid, or identity-mismatched effective physical-card views are refused',
    'seeds are nonnegative int64; arbitrary Python ints are not represented',
    'Random._randbelow is bounded by max_draws instead of an unbounded host loop',
    'card-table holes are not compacted into codec registration order',
    'complete physical duel execution and H3 are not provided by guarded sampling',
)
_TORCH_DTYPES = {
    'bool': 'bool', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32',
    'int64': 'int64', 'uint8': 'uint8', 'float64': 'float64',
}


def coverage():
    return MappingProxyType({
        'supports_full_duel_rules': False,
        'oracle': 'app.modules.card_game.rl.seven_sampler',
        'full_information_set_guard': True,
        'full_sampling_qualified': False,
        'physical_rules_available': False,
        'h3_available': False,
        'implemented': (
            'playing and schema-supported choice roots',
            'public-count originals/specials/surplus/hidden family',
            'generation allocation including recovery/deck-generator retry',
            'kit reconstruction, mandatory generated-hand placement, heiyu/lingke exception',
            'A03 public ephemeral assignment, xiaozhi jingu, own-deck JSON sort+shuffle',
            'entity/instance reissue, formal CPython MT19937 seed/shuffle/choice/randrange',
            'revealed enemy hand preserves original identity, order and metadata',
            'visible own-deck order skips both canonical sort and shuffle RNG',
            'visible effective-card views retain their physical row and shared aliases',
            'default full 3479 observation and all original 1576 candidate-feature checks; mismatch rollback',
        ),
        'missing_stages': MISSING_STAGES,
        'schema_fields_rejected': (
            'complete named schema required; missing or unknown columns are rejected',
        ),
    })


@dataclass(frozen=True)
class FrozenSamplerCatalog:
    device: object
    tables: dict
    scalars: dict
    fingerprint: bytes


def prepare_sampler_catalog(device, torch_module=None):
    t = torch_module
    if t is None:
        import torch as t
    device = t.device(device)
    vc, vh = VOCAB_CARDS + 1, VOCAB_CHARS + 1
    derived = t.zeros(vc, dtype=t.uint8, device=device)
    owner = t.zeros(vc, dtype=t.int32, device=device)
    cid_len = t.zeros(vc, dtype=t.int32, device=device)
    cid_bytes = t.zeros((vc, 8), dtype=t.uint8, device=device)
    hid_len = t.zeros(vh, dtype=t.int32, device=device)
    hid_bytes = t.zeros((vh, 16), dtype=t.uint8, device=device)
    seats = t.full((vh,), -1, dtype=t.int32, device=device)
    kit_n = t.zeros(vh, dtype=t.int32, device=device)
    kit = t.zeros((vh, 8), dtype=t.int32, device=device)
    int_p = t.zeros((vc, N_KINT), dtype=t.uint8, device=device)
    int_v = t.zeros((vc, N_KINT), dtype=t.int32, device=device)
    bool_p = t.zeros((vc, N_KBOOL), dtype=t.uint8, device=device)
    bool_v = t.zeros((vc, N_KBOOL), dtype=t.uint8, device=device)
    str_p = t.zeros((vc, N_KSTR), dtype=t.uint8, device=device)
    src_p = t.zeros(vc, dtype=t.uint8, device=device)
    play_p = t.zeros(vc, dtype=t.uint8, device=device)
    for name, vocab in CHARACTER_VOCAB.items():
        seats[vocab] = seven_lineup.SEAT_INDEX[name]
        blob = name.encode('ascii')
        hid_len[vocab] = len(blob)
        hid_bytes[vocab, :len(blob)] = t.tensor(list(blob), dtype=t.uint8, device=device)
    kits = {cid: [] for cid in CHARACTER_VOCAB}
    for card_id in seven_lineup.CARD_IDS:
        spec = lookup_card(card_id)
        vocab = CARD_VOCAB[card_id]
        blob = card_id.encode('ascii')
        cid_len[vocab] = len(blob)
        cid_bytes[vocab, :len(blob)] = t.tensor(list(blob), dtype=t.uint8, device=device)
        owner[vocab] = CHARACTER_VOCAB[spec['character_id']]
        derived[vocab] = 1 if spec.get('derived') else 0
        if not spec.get('derived'):
            kits[spec['character_id']].append(vocab)
        for i, key in enumerate(CARD_INT_FIELDS):
            if key not in spec:
                int_p[vocab, i] = PRESENCE_ABSENT
            elif spec[key] is None:
                int_p[vocab, i] = PRESENCE_NULL
            else:
                int_p[vocab, i] = PRESENCE_CATALOG
                int_v[vocab, i] = int(spec[key])
        for i, key in enumerate(CARD_BOOL_FIELDS):
            if key not in spec:
                bool_p[vocab, i] = PRESENCE_ABSENT
            elif spec[key] is None:
                bool_p[vocab, i] = PRESENCE_NULL
            else:
                bool_p[vocab, i] = PRESENCE_CATALOG
                bool_v[vocab, i] = 1 if spec[key] else 0
        for i, key in enumerate(CARD_STR_FIELDS):
            if key not in spec:
                str_p[vocab, i] = PRESENCE_ABSENT
            elif spec[key] is None:
                str_p[vocab, i] = PRESENCE_NULL
            else:
                str_p[vocab, i] = PRESENCE_CATALOG
        src_p[vocab] = PRESENCE_CATALOG if 'source' in spec else PRESENCE_ABSENT
        play_p[vocab] = PRESENCE_CATALOG if 'play_options' in spec else PRESENCE_ABSENT
    for cid, cards in kits.items():
        vocab = CHARACTER_VOCAB[cid]
        if len(cards) > 8:
            raise RuntimeError('kit wider than 8 original cards')
        kit_n[vocab] = len(cards)
        if cards:
            kit[vocab, :len(cards)] = t.tensor(cards, dtype=t.int32, device=device)
    tables = dict(
        card_derived=derived, card_owner=owner, card_id_len=cid_len, card_id_bytes=cid_bytes.reshape(-1),
        char_id_len=hid_len, char_id_bytes=hid_bytes.reshape(-1), char_seat=seats, kit_n=kit_n,
        kit=kit.reshape(-1), cat_int_p=int_p.reshape(-1), cat_int_v=int_v.reshape(-1),
        cat_bool_p=bool_p.reshape(-1), cat_bool_v=bool_v.reshape(-1), cat_str_p=str_p.reshape(-1),
        cat_src=src_p, cat_play=play_p,
    )
    scalars = dict(
        nf01=CARD_VOCAB['NF01'], a03=CARD_VOCAB['A03'], af01=CARD_VOCAB['AF01'],
        ff01=CARD_VOCAB['FF01'], q05=CARD_VOCAB['Q05'], c02=CARD_VOCAB['C02'],
        a04=CARD_VOCAB['A04'], pf01=CARD_VOCAB['PF01'],
        xiaozhi=CHARACTER_VOCAB['xiaozhi'], canhong=CHARACTER_VOCAB['canhong'],
        anhunqu=CHARACTER_VOCAB['anhunqu'], heiyu=CHARACTER_VOCAB['heiyu'],
        dafutier=CHARACTER_VOCAB['dafutier'], lingke=CHARACTER_VOCAB['lingke'],
    )
    return FrozenSamplerCatalog(device=device, tables=tables, scalars=scalars, fingerprint=catalog_fingerprint())


def sampler_source(seed_words=1):
    if seed_words not in (1,5):raise ValueError("Sampler seed ABI must be scalar63 or full320")
    prefix = r'''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define SOA_FN __device__
#else
#define SOA_FN
#endif
#define BMT_DEVICE SOA_FN
'''
    body = r'''
SOA_FN int i8(const unsigned char* a,int i){return (int)(signed char)a[i];}
SOA_FN int u8(const unsigned char* a,int i){return (int)a[i];}
SOA_FN int i16(const unsigned char* a,int i){return (int)(short)(a[(size_t)i*2]|((unsigned)a[(size_t)i*2+1]<<8));}
SOA_FN void s8(unsigned char* a,int i,int v){a[i]=(unsigned char)v;}
SOA_FN void s16(unsigned char* a,int i,int v){unsigned short x=(unsigned short)(short)v;a[(size_t)i*2]=(unsigned char)x;a[(size_t)i*2+1]=(unsigned char)(x>>8);}
SOA_FN int rng_below(bmt_state_t* rng,unsigned long long n,int max_draws,unsigned long long* out){
 if(n==0ULL)return 2;
 int k=bmt_bit_length_u64(n);
 for(int a=0;a<max_draws;a++){uint64_t r=0;bmt_getbits64(rng,k,&r);if(r<n){*out=r;return 0;}}
 return 2;
}
SOA_FN int rng_choice(bmt_state_t* rng,int n,int max_draws,int* err){
 unsigned long long r=0;if(n<=0){*err=1;return 0;}
 int st=rng_below(rng,(unsigned long long)n,max_draws,&r);if(st){*err=st;return 0;}
 return (int)r;
}
SOA_FN int rng_shuffle(bmt_state_t* rng,int* a,int n,int max_draws){
 for(int i=n-1;i>0;i--){
  unsigned long long r=0;int st=rng_below(rng,(unsigned long long)i+1ULL,max_draws,&r);if(st)return st;
  int j=(int)r,t=a[i];a[i]=a[j];a[j]=t;
 }
 return 0;
}
SOA_FN int append_c(char* b,int n,int cap,const char* s){
 int i=0;while(s[i]){if(n>=cap-1)return -1;b[n++]=s[i++];}b[n]=0;return n;
}
SOA_FN int append_n(char* b,int n,int cap,const unsigned char* s,int m){
 int i=0;while(i<m){if(n>=cap-1)return -1;b[n++]=(char)s[i++];}b[n]=0;return n;
}
SOA_FN int append_i(char* b,int n,int cap,int v){
 char tmp[16];int m=0;unsigned u;if(v<0){if(n>=cap-1)return -1;b[n++]='-';u=(unsigned)(-(long long)v);}else u=(unsigned)v;
 if(u==0)tmp[m++]='0';else{char r[16];int k=0;while(u){r[k++]=(char)('0'+u%10);u/=10;}while(k)tmp[m++]=r[--k];}
 tmp[m]=0;return append_c(b,n,cap,tmp);
}
SOA_FN int intern_ascii(unsigned char* bytes,int* used,int* count,int* offset,int* length,int max_slots,int max_bytes,int row,int slots,int nbytes,const unsigned char* s,int n){
 int i,k;for(i=0;i<count[row];i++){int off=offset[row*slots+i],ln=length[row*slots+i];if(ln!=n)continue;int ok=1;for(k=0;k<n;k++)if(bytes[row*nbytes+off+k]!=s[k]){ok=0;break;}if(ok)return i;}
 if(count[row]>=max_slots||used[row]+n>max_bytes)return -1;
 i=count[row];offset[row*slots+i]=used[row];length[row*slots+i]=n;
 for(k=0;k<n;k++)bytes[row*nbytes+used[row]+k]=s[k];used[row]+=n;count[row]+=1;return i;
}
SOA_FN int needed_n(int family,int copies_q,int copies_c,int known_q,int known_c,int disc_c02,int af,int a3,int ff,int deck_gen){
 int n=family+af+a3+ff,extra;
 if(!deck_gen){extra=copies_q-known_q;if(extra<0)extra=0;n+=extra;}
 extra=copies_c-known_c+(disc_c02>0?1:0);if(extra<0)extra=0;n+=extra;return n;
}
SOA_FN int feasible_n(int family,int copies_q,int copies_c,int known_q,int known_c,int disc_c02,int af,int a3,int ff,int pf,int lim_q,int lim_c,int lim_pf,int remaining,int hand_n,int deck_gen,int place){
 int need,free=0,left;if(place)return 1;
 need=needed_n(family,copies_q,copies_c,known_q,known_c,disc_c02,af,a3,ff,deck_gen);
 if(copies_q||lim_q){if(deck_gen)free+=lim_q;else{int x=known_q-copies_q;if(x<0)x=0;free+=(lim_q<x?lim_q:x);}}
 if(copies_c||lim_c){int x=known_c-(disc_c02>0?1:0)-copies_c;if(x<0)x=0;free+=(lim_c<x?lim_c:x);}
 if(deck_gen)free+=lim_pf;left=remaining-free;if(left<0)left=0;return need+left<=hand_n;
}
'''
    body += r'''
SOA_FN void write_new_card(unsigned char* cvalid,int* entity,int* inst,unsigned char* tmpl,unsigned char* sideb,
 unsigned char* ip,int* iv,unsigned char* bp,unsigned char* bv,unsigned char* sp,int* ss,
 unsigned char* src,unsigned char* play,unsigned char* jp,int* jd,int* js,
 int row,int K,int slot,int tmpl_v,int side,int ent,int ins,
 const unsigned char* cip,const int* civ,const unsigned char* cbp,const unsigned char* cbv,const unsigned char* csp,
 const unsigned char* csrc,const unsigned char* cplay,int n_kint,int n_kbool,int n_kstr,int vc){
 int i,base=row*K+slot,tb=tmpl_v;
 s8(cvalid,base,1);entity[base]=ent;inst[base]=ins;s16(tmpl,base,tmpl_v);s8(sideb,base,side);
 for(i=0;i<n_kint;i++){s8(ip,base*n_kint+i,u8(cip,tb*n_kint+i));iv[base*n_kint+i]=civ[tb*n_kint+i];}
 for(i=0;i<n_kbool;i++){s8(bp,base*n_kbool+i,u8(cbp,tb*n_kbool+i));s8(bv,base*n_kbool+i,u8(cbv,tb*n_kbool+i));}
 s8(bp,base*n_kbool+0,2);s8(bv,base*n_kbool+0,0);
 for(i=0;i<n_kstr;i++)s8(sp,base*n_kstr+i,u8(csp,tb*n_kstr+i));
 ss[base*n_kstr+0]=0;s8(src,base,u8(csrc,tb));s8(play,base,u8(cplay,tb));s8(jp,base,0);jd[base]=0;js[base]=0;
}
'''
    body += r'''
SOA_FN int dump_own_key(char* buf,int cap,int tmpl,int copy_p,int copy_v,int eph_p,int eph_v,int exp_p,int exp_v,
 int jing_p,int jing_d,int jing_s,int ch_p,const unsigned char* ch_s,int ch_n,int orig_p,const unsigned char* orig_s,int orig_n,
 const unsigned char* cid_bytes,const int* cid_len){
 int n=0,comma=0;
 n=append_c(buf,n,cap,"[\"");if(n<0)return -1;
 n=append_n(buf,n,cap,cid_bytes+tmpl*8,cid_len[tmpl]);if(n<0)return -1;
 n=append_c(buf,n,cap,"\", {");if(n<0)return -1;
 if(ch_p){n=append_c(buf,n,cap,"\"character_id\": ");if(n<0)return -1;
  if(ch_p==1)n=append_c(buf,n,cap,"null");else{n=append_c(buf,n,cap,"\"");n=append_n(buf,n,cap,ch_s,ch_n);n=append_c(buf,n,cap,"\"");}if(n<0)return -1;comma=1;}
 if(copy_p){if(comma){n=append_c(buf,n,cap,", ");if(n<0)return -1;}n=append_c(buf,n,cap,"\"copy\": ");if(n<0)return -1;
  n=append_c(buf,n,cap,copy_p==1?"null":(copy_v?"true":"false"));if(n<0)return -1;comma=1;}
 if(eph_p){if(comma){n=append_c(buf,n,cap,", ");if(n<0)return -1;}n=append_c(buf,n,cap,"\"ephemeral_turn\": ");if(n<0)return -1;
  if(eph_p==1)n=append_c(buf,n,cap,"null");else n=append_i(buf,n,cap,eph_v);if(n<0)return -1;comma=1;}
 if(exp_p){if(comma){n=append_c(buf,n,cap,", ");if(n<0)return -1;}n=append_c(buf,n,cap,"\"expires_turn\": ");if(n<0)return -1;
  if(exp_p==1)n=append_c(buf,n,cap,"null");else n=append_i(buf,n,cap,exp_v);if(n<0)return -1;comma=1;}
 if(jing_p){if(comma){n=append_c(buf,n,cap,", ");if(n<0)return -1;}n=append_c(buf,n,cap,"\"jingu_mark\": ");if(n<0)return -1;
  if(jing_p==1)n=append_c(buf,n,cap,"null");else{n=append_c(buf,n,cap,"{\"delta\": ");n=append_i(buf,n,cap,jing_d);
   n=append_c(buf,n,cap,", \"source_entity_id\": \"e");n=append_i(buf,n,cap,jing_s);n=append_c(buf,n,cap,"\"}");}if(n<0)return -1;comma=1;}
 if(orig_p){if(comma){n=append_c(buf,n,cap,", ");if(n<0)return -1;}n=append_c(buf,n,cap,"\"original_character_id\": ");if(n<0)return -1;
  if(orig_p==1)n=append_c(buf,n,cap,"null");else{n=append_c(buf,n,cap,"\"");n=append_n(buf,n,cap,orig_s,orig_n);n=append_c(buf,n,cap,"\"");}if(n<0)return -1;}
 n=append_c(buf,n,cap,"}]");return n;
}
SOA_FN int cmp_key(const char* a,const char* b){int i=0;while(a[i]&&a[i]==b[i])i++;return (unsigned char)a[i]-(unsigned char)b[i];}
'''
    body += r'''
SOA_FN int has_hero(const unsigned char* cvalid,const unsigned char* ctmpl,const unsigned char* cb_p,const unsigned char* cb_v,
 int row,int side,int C,int n_cbool,int want,int skip_summon){
 int slot;for(slot=0;slot<C;slot++){
  int b=((row*2+side)*C+slot);
  if(!u8(cvalid,b))continue;
  if(skip_summon && u8(cb_p,b*n_cbool+1)==2 && u8(cb_v,b*n_cbool+1))continue;
  if(i16(ctmpl,b)==want)return 1;
 }
 return 0;
}
SOA_FN int opt_int(const unsigned char* p,const int* v,int idx,int field,int n_fields){
 int code=u8(p,idx*n_fields+field);if(code!=2)return 0;return v[idx*n_fields+field];
}
SOA_FN int opt_bool(const unsigned char* p,const unsigned char* v,int idx,int field,int n_fields){
 return u8(p,idx*n_fields+field)==2 && u8(v,idx*n_fields+field);
}
SOA_FN void soa_sample_row(
 int row,int C,int K,int Z,int CH,int n_pint,int n_cbool,int n_kint,int n_kbool,int n_kstr,
 int max_draws,int max_slots,int max_bytes,int SW,int vc,
 const int* sp,int* scratch,const long long* seeds,const long long* viewers,const int* active,int* errors,
 const unsigned char* phase,const int* turn,const unsigned char* active_side,const unsigned char* first_side,
 const unsigned char* escalation,const unsigned char* pending,const unsigned char* choice_kind,const unsigned char* choice_side,
 const unsigned char* choice_n,const int* choice_idx,const unsigned char* operation,
 const unsigned char* pvalid,const int* pent,const unsigned char* porder,const unsigned char* pint_p,const int* pint_v,
 const unsigned char* cvalid,const unsigned char* ctmpl,const int* cent,const int* chp,const unsigned char* cbool_p,const unsigned char* cbool_v,
 const unsigned char* zcount,int* zindex,
 unsigned char* kvalid,int* kentity,int* kinst,unsigned char* ktmpl,unsigned char* kside,
 unsigned char* kint_p,int* kint_v,unsigned char* kbool_p,unsigned char* kbool_v,unsigned char* kstr_p,int* kstr_s,
 unsigned char* ksrc,unsigned char* kplay,unsigned char* kj_p,int* kj_d,int* kj_s,
 int* next_inst,int* next_ent,long long* rng_out,int* str_count,int* str_used,int* str_off,int* str_len,unsigned char* str_bytes,
 const unsigned char* card_derived,const int* card_owner,const int* cid_len,const unsigned char* cid_bytes,
 const int* hid_len,const unsigned char* hid_bytes,const int* char_seat,const int* kit_n,const int* kit,
 const unsigned char* cat_int_p,const int* cat_int_v,const unsigned char* cat_bool_p,const unsigned char* cat_bool_v,
 const unsigned char* cat_str_p,const unsigned char* cat_src,const unsigned char* cat_play,const int* operation_card,
 const unsigned char* revealed_count,const unsigned char* revealed_side,const int* revealed_instance,
 const unsigned char* card_views,const int* physical_cards,const unsigned char* protected_cards){
 if(errors[row]!=0||!active[row])return;
 int viewer=(int)viewers[row],foe,phase_v=i16(phase,row),choosing,kind=-1,err=0,i,s,z,pos,slot,P=2*Z;
 int nf01=sp[0],a03=sp[1],af01=sp[2],ff01=sp[3],q05=sp[4],c02=sp[5],a04=sp[6],pf01=sp[7];
 int xiaozhi=sp[8],canhong=sp[9],anhunqu=sp[10],heiyu=sp[11],dafutier=sp[12],lingke=sp[13];
 int *pool_cid=scratch+row*SW,*pool_eph=pool_cid+P,*idx=pool_eph+P,*keep=idx+P,*own_slot=keep+P,*orig=own_slot+Z,*specc=orig+vc;
 int hand_n,deck_n,keep_n=0,family,hidden_family,surplus,budget=0,place,has_x,has_c,has_a,has_h,has_d,has_l;
 int copies_q=0,copies_c=0,lim[6],alloc[6],alloc_order[4],alloc_n=0,temporary=0,remaining=0,extended=0,pool_n=0,turn_v=turn[row];
 int known_q,known_c,disc_c02,disc_af01,disc_a03,disc_a04,pub_a03,spec_sum,a_known,nposs,t,left;
 int kinds[6],nk=0,free_n=0,hid_n,own_n,next_i,next_e,xiaozhi_hp=0,xiaozhi_aw=0,xiaozhi_ent=0,own_visible=0;
 bmt_state_t rng,saved;unsigned long long raw,draw;uint32_t words[SOA_SEED_WORDS*2];char ka[192],kb[192];
 if(viewer!=0&&viewer!=1){errors[row]=1;return;}
 #if SOA_SEED_WORDS==1
 if(seeds[row]<0){errors[row]=1;return;}
 #endif
 foe=1-viewer;choosing=phase_v==2;
 for(s=0;s<2;s++)for(z=0;z<4;z++){
  int count=i16(zcount,(row*2+s)*4+z);
  if(count<0||count>Z){errors[row]=1;return;}
  for(pos=0;pos<count;pos++){
   int ci=zindex[((row*2+s)*4+z)*Z+pos];
   if(ci<0||ci>=K||!u8(kvalid,row*K+ci)){errors[row]=1;return;}
   if(u8(card_views,row*K+ci)){errors[row]=3;return;}
   if(z==2||(s==viewer&&(z==0||z==1))){
    int tmpl=i16(ktmpl,row*K+ci);if(tmpl<1||tmpl>=vc){errors[row]=1;return;}
   }
  }
 }
 if(choosing){
  if(!u8(pending,row)||i8(choice_side,row)!=viewer){errors[row]=3;return;}
  kind=i16(choice_kind,row);
  int cn=i16(choice_n,row);if(cn<0||cn>CH){errors[row]=1;return;}
  if(!SUPPORTED_ROOT_CHOICE){errors[row]=3;return;}
 }else if(phase_v!=1||u8(pending,row)||u8(operation,row)){errors[row]=3;return;}
 if(!u8(pvalid,row*2+foe)||!u8(pvalid,row*2+viewer)){errors[row]=1;return;}
 hand_n=i16(zcount,(row*2+foe)*4+0);deck_n=i16(zcount,(row*2+foe)*4+1);
 if(hand_n<0||deck_n<0||hand_n>Z||deck_n>Z){errors[row]=1;return;}
 {
  int rn=i16(revealed_count,row*2+foe),cn=choosing&&kind==2?i16(choice_n,row):0;
  if(rn<0||rn>K||cn<0||cn>CH){errors[row]=1;return;}
  for(pos=0;pos<hand_n;pos++){
   slot=zindex[((row*2+foe)*4+0)*Z+pos];if(slot<0||slot>=K||!u8(kvalid,row*K+slot)){errors[row]=1;return;}
   int inst=kinst[row*K+slot],hit=0;
   for(i=0;i<rn;i++)if(i8(revealed_side,(row*2+foe)*K+i)==foe&&revealed_instance[(row*2+foe)*K+i]==inst)hit=1;
   for(i=0;i<cn;i++){int ci=choice_idx[row*CH+i];if(ci>=0&&ci<K&&kinst[row*K+ci]==inst)hit=1;}
   if(hit){int known=i16(ktmpl,row*K+slot);if(known<1||known>=vc){errors[row]=1;return;}keep[keep_n++]=slot;}
  }
 }
 hidden_family=0;family=opt_int(pint_p,pint_v,row*2+foe,8,n_pint);
 for(z=2;z<=2;z++)for(pos=0;pos<i16(zcount,(row*2+foe)*4+z);pos++){
  slot=zindex[((row*2+foe)*4+z)*Z+pos];if(slot<0||slot>=K){errors[row]=1;return;}
  if(i16(ktmpl,row*K+slot)==nf01)hidden_family++;
 }
 for(i=0;i<keep_n;i++)if(i16(ktmpl,row*K+keep[i])==nf01)hidden_family++;
 hidden_family=family-hidden_family;
 if(hidden_family<0||hidden_family>hand_n-keep_n+deck_n){errors[row]=4;return;}
 surplus=i16(zcount,(row*2+foe)*4+2)+hand_n+deck_n-family-32;
 if(surplus<0){errors[row]=5;return;}
 has_x=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,xiaozhi,0);
 has_c=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,canhong,0);
 has_a=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,anhunqu,0);
 has_h=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,heiyu,0)||(has_a&&has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,viewer,C,n_cbool,heiyu,0));
 has_d=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,dafutier,0)||(has_a&&has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,viewer,C,n_cbool,dafutier,0));
 has_l=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,lingke,0);
 place=has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,heiyu,0)||has_l;
 if(u8(escalation,row)&&turn_v>=20){int first=i8(first_side,row)==foe?21:20;budget=turn_v<first?0:1+(turn_v-first)/2;}
 for(i=0;i<vc;i++){orig[i]=0;specc[i]=0;}
 disc_c02=0;disc_af01=0;disc_a03=0;disc_a04=0;pub_a03=0;spec_sum=0;
 for(pos=0;pos<i16(zcount,(row*2+foe)*4+2);pos++){
  slot=zindex[((row*2+foe)*4+2)*Z+pos];int tmpl=i16(ktmpl,row*K+slot),der=u8(card_derived,tmpl)||opt_bool(kbool_p,kbool_v,row*K+slot,3,n_kbool);
  if(tmpl==c02)disc_c02++;if(tmpl==af01)disc_af01++;if(tmpl==a03)disc_a03++;if(tmpl==a04)disc_a04++;
  if(tmpl==nf01)continue;if(der){specc[tmpl]++;spec_sum++;}else orig[tmpl]++;if(tmpl==a03)pub_a03++;
 }
 for(i=0;i<keep_n;i++){
  slot=keep[i];int tmpl=i16(ktmpl,row*K+slot),der=u8(card_derived,tmpl)||opt_bool(kbool_p,kbool_v,row*K+slot,3,n_kbool);
  if(tmpl==nf01)continue;if(der){specc[tmpl]++;spec_sum++;}else orig[tmpl]++;if(tmpl==a03)pub_a03++;
 }
 for(int seed_word=0;seed_word<SOA_SEED_WORDS;seed_word++){raw=(unsigned long long)seeds[row*SOA_SEED_WORDS+seed_word];words[seed_word*2]=(uint32_t)raw;words[seed_word*2+1]=(uint32_t)(raw>>32);}
 int word_count=SOA_SEED_WORDS*2;while(word_count>1&&words[word_count-1]==0)word_count--;bmt_seed_words(&rng,words,word_count);saved=rng;
 {int pass,ok_alloc=0,retry=budget||has_x||has_h;
  for(pass=0;pass<2&&!ok_alloc;pass++){
   int recovered=pass?budget:0,deck_gen=pass?1:0,tmax=(orig[a03]<disc_af01?orig[a03]:disc_af01)+1,poss_t[129],nposs=0;
   if(pass)rng=saved;
   if(tmax>129){errors[row]=14;return;}
   for(t=0;t<tmax;t++){
    int known_a03=orig[a03]-t,rem,lq=0,lc=0,laf=0,la3=0,lpf=0,lff=0,cq=0,cc=0,count_x=0,count_c=0;
    if(known_a03>2)continue;
    a_known=0;for(i=1;i<vc;i++)if(card_owner[i]==anhunqu)a_known+= (i==a03?known_a03:orig[i]);
    if(a_known>8)continue;
    rem=surplus-spec_sum-t;
    if(has_x){for(i=1;i<vc;i++)if(card_owner[i]==xiaozhi&&i!=q05)count_x+=orig[i];count_x+= (q05==a03?known_a03:orig[q05]);
     cq=orig[q05]-2;if(cq<count_x-8)cq=count_x-8;if(cq<0)cq=0;rem-=cq;lq=surplus-cq;}
    if(has_c){for(i=1;i<vc;i++)if(card_owner[i]==canhong)count_c+= (i==a03?known_a03:orig[i]);
     cc=orig[c02]-2;if(cc<count_c-8)cc=count_c-8;if(cc<0)cc=0;rem-=cc;lc=disc_c02+recovered-cc;}
    if(has_a){laf=disc_a04+recovered-specc[af01];la3=i8(active_side,row)==foe?(disc_af01-t):0;}
    if(has_h)lpf=surplus-specc[pf01];if(lpf<0)lpf=0;
    if(has_d){lff=surplus-specc[ff01];if(lff<0)lff=0;}
    if(lq<0||lc<0||laf<0||la3<0||lpf<0||lff<0)continue;
    {int sum=0;if(has_x)sum+=lq;if(has_c)sum+=lc;if(has_a)sum+=laf+la3;if(has_h)sum+=lpf;if(has_d)sum+=lff;
     if(rem<0||rem>sum)continue;
     if(!feasible_n(hidden_family,cq,cc,orig[q05]-(q05==a03?t:0),orig[c02],disc_c02,0,0,0,0,lq,lc,lpf,rem,hand_n-keep_n,deck_gen,place))continue;
     poss_t[nposs++]=t;}
   }
   if(!nposs){if(pass==0&&retry)continue;errors[row]=6;return;}
   t=poss_t[rng_choice(&rng,nposs,max_draws,&err)];if(err){errors[row]=err;return;}
   {int known_a03=orig[a03]-t,rem,lq=0,lc=0,laf=0,la3=0,lpf=0,lff=0,cq=0,cc=0,count_x=0,count_c=0,aq,ac,aaf,aa3,apf,aff,gfail=0;
    rem=surplus-spec_sum-t;
    if(has_x){for(i=1;i<vc;i++)if(card_owner[i]==xiaozhi&&i!=q05)count_x+=orig[i];count_x+=orig[q05];
     cq=orig[q05]-2;if(cq<count_x-8)cq=count_x-8;if(cq<0)cq=0;rem-=cq;lq=surplus-cq;}
    if(has_c){for(i=1;i<vc;i++)if(card_owner[i]==canhong)count_c+=orig[i];
     cc=orig[c02]-2;if(cc<count_c-8)cc=count_c-8;if(cc<0)cc=0;rem-=cc;lc=disc_c02+(pass?budget:0)-cc;}
    if(has_a){laf=disc_a04+(pass?budget:0)-specc[af01];la3=i8(active_side,row)==foe?(disc_af01-t):0;}
    if(has_h)lpf=surplus-specc[pf01];if(lpf<0)lpf=0;if(has_d){lff=surplus-specc[ff01];if(lff<0)lff=0;}
    aq=ac=aaf=aa3=apf=aff=0;alloc_n=0;nk=0;if(has_x)kinds[nk++]=0;if(has_c)kinds[nk++]=1;if(has_a){kinds[nk++]=2;kinds[nk++]=3;}if(has_h)kinds[nk++]=4;if(has_d)kinds[nk++]=5;
    for(left=rem;left>0;left--){
     int chn=0,chc[6],knd,trial_q,trial_c,trial_aq,trial_ac,trial_aaf,trial_aa3,trial_apf,trial_aff,tlq,tlc,tlaf,tla3,tlpf,tlff;
     for(i=0;i<nk;i++){
      knd=kinds[i];tlq=lq;tlc=lc;tlaf=laf;tla3=la3;tlpf=lpf;tlff=lff;trial_q=cq;trial_c=cc;trial_aq=aq;trial_ac=ac;trial_aaf=aaf;trial_aa3=aa3;trial_apf=apf;trial_aff=aff;
      if(knd==0){if(!tlq)continue;tlq--;trial_q++;}else if(knd==1){if(!tlc)continue;tlc--;trial_c++;}else if(knd==2){if(!tlaf)continue;tlaf--;trial_aaf++;}else if(knd==3){if(!tla3)continue;tla3--;trial_aa3++;}else if(knd==4){if(!tlpf)continue;tlpf--;trial_apf++;}else {if(!tlff)continue;tlff--;trial_aff++;}
      if(feasible_n(hidden_family,trial_q,trial_c,orig[q05],orig[c02],disc_c02,trial_aaf,trial_aa3,trial_aff,trial_apf,tlq,tlc,tlpf,left-1,hand_n-keep_n,pass?1:0,place))chc[chn++]=knd;
     }
     if(!chn){gfail=1;break;}
     knd=chc[rng_choice(&rng,chn,max_draws,&err)];if(err){errors[row]=err;return;}
     if((knd==2&&aaf==0)||(knd==3&&aa3==0)||(knd==4&&apf==0)||(knd==5&&aff==0))alloc_order[alloc_n++]=knd;
     if(knd==0){lq--;cq++;}else if(knd==1){lc--;cc++;}else if(knd==2){laf--;aaf++;}else if(knd==3){la3--;aa3++;}else if(knd==4){lpf--;apf++;}else {lff--;aff++;}
    }
    if(gfail){if(pass==0&&retry)continue;errors[row]=7;return;}
    copies_q=cq;copies_c=cc;alloc[0]=0;alloc[1]=0;alloc[2]=aaf;alloc[3]=aa3;alloc[4]=apf;alloc[5]=aff;temporary=t;remaining=0;extended=pass==1;ok_alloc=1;
   }
  }
 }
 orig[a03]-=temporary;
 {int a03n=0;for(pos=0;pos<i16(zcount,(row*2+foe)*4+2);pos++){slot=zindex[((row*2+foe)*4+2)*Z+pos];if(i16(ktmpl,row*K+slot)==a03)idx[a03n++]=slot;}
  for(i=0;i<keep_n;i++)if(i16(ktmpl,row*K+keep[i])==a03)idx[a03n++]=keep[i];
  if((err=rng_shuffle(&rng,idx,a03n,max_draws))){errors[row]=err;return;}
  for(i=0;i<a03n;i++){slot=idx[i];int base=row*K+slot;if(i<temporary){s8(kint_p,base*n_kint+6,2);kint_v[base*n_kint+6]=turn_v;}else s8(kint_p,base*n_kint+6,0);}
 }
 pool_n=0;
 for(s=0;s<i16(porder,row*2+foe);s++){
  int b=((row*2+foe)*C+s),hero,n,bonus=0,forced,opt_n=0,counts[8],caps[8];
  if(!u8(cvalid,b))continue;
  if(opt_bool(cbool_p,cbool_v,b,1,n_cbool))continue;
  hero=i16(ctmpl,b);if(hero<=0||hero>=N_CHAR_TEMPLATES||char_seat[hero]<0){errors[row]=8;return;}
  if(card_owner[q05]==hero)bonus+=copies_q;if(card_owner[c02]==hero)bonus+=copies_c;
  n=8+bonus;for(i=0;i<kit_n[hero];i++){int cid=kit[hero*8+i];counts[i]=orig[cid];caps[i]=2+(cid==q05?copies_q:0)+(cid==c02?copies_c:0);n-=counts[i];if(counts[i]>caps[i]){errors[row]=9;return;}}
  if(n<0){errors[row]=9;return;}
  for(i=0;i<kit_n[hero];i++){int cid=kit[hero*8+i];if(cid!=q05&&cid!=c02)continue;forced= (cid==q05?copies_q:copies_c)-counts[i]+(cid==c02&&disc_c02>0?1:0);if(forced<0)forced=0;
   for(pos=0;pos<forced;pos++){if(pool_n>=P){errors[row]=14;return;}pool_cid[pool_n]=cid;pool_eph[pool_n]=0;pool_n++;}counts[i]+=forced;n-=forced;}
  for(i=0;i<kit_n[hero];i++){int cid=kit[hero*8+i],leftc=caps[i]-counts[i];for(pos=0;pos<leftc;pos++){if(opt_n>=P){errors[row]=14;return;}idx[opt_n++]=cid;}}
  if((err=rng_shuffle(&rng,idx,opt_n,max_draws))){errors[row]=err;return;}
  if(n<0||n>opt_n){errors[row]=10;return;}
  for(i=0;i<n;i++){if(pool_n>=P){errors[row]=14;return;}pool_cid[pool_n]=idx[i];pool_eph[pool_n]=0;pool_n++;}
 }
 for(i=0;i<hidden_family;i++){if(pool_n>=P){errors[row]=14;return;}pool_cid[pool_n]=nf01;pool_eph[pool_n]=0;pool_n++;}
 for(int order=0;order<alloc_n;order++){
  int kind=alloc_order[order],cid=kind==2?af01:kind==3?a03:kind==4?pf01:ff01;
  for(i=0;i<alloc[kind];i++){
   if(pool_n>=P){errors[row]=14;return;}
   pool_cid[pool_n]=cid;pool_eph[pool_n]=kind==3;pool_n++;
  }
 }
 if(pool_n!=hand_n-keep_n+deck_n){errors[row]=11;return;}
 for(i=0;i<pool_n;i++)idx[i]=i;if((err=rng_shuffle(&rng,idx,pool_n,max_draws))){errors[row]=err;return;}
 {int tmpc[256],tmpe[256];if(pool_n>256){errors[row]=14;return;}for(i=0;i<pool_n;i++){tmpc[i]=pool_cid[idx[i]];tmpe[i]=pool_eph[idx[i]];}for(i=0;i<pool_n;i++){pool_cid[i]=tmpc[i];pool_eph[i]=tmpe[i];}}
 hid_n=hand_n-keep_n;
 if(!place){
  int mand=0,rest=0;for(i=0;i<pool_n;i++){int cid=pool_cid[i],eph=pool_eph[i];if(cid==nf01||cid==af01||cid==ff01||eph){idx[mand++]=i;}else idx[96+rest++]=i;}
  {int mc[256],me[256],rc[256],re[256],mn=0,rn=0;
   for(i=0;i<mand;i++){mc[mn]=pool_cid[idx[i]];me[mn]=pool_eph[idx[i]];mn++;}
   for(i=0;i<rest;i++){rc[rn]=pool_cid[idx[96+i]];re[rn]=pool_eph[idx[96+i]];rn++;}
   if(copies_c){int need=copies_c-orig[c02]+(disc_c02>0?1:0);if(need<0)need=0;for(t=0;t<need;t++){int hit=-1;for(i=0;i<rn;i++)if(rc[i]==c02){hit=i;break;}if(hit<0){errors[row]=11;return;}mc[mn]=rc[hit];me[mn]=re[hit];mn++;for(i=hit;i<rn-1;i++){rc[i]=rc[i+1];re[i]=re[i+1];}rn--;}}
   if(copies_q&&!extended){int need=copies_q-orig[q05];if(need<0)need=0;for(t=0;t<need;t++){int hit=-1;for(i=0;i<rn;i++)if(rc[i]==q05){hit=i;break;}if(hit<0){errors[row]=11;return;}mc[mn]=rc[hit];me[mn]=re[hit];mn++;for(i=hit;i<rn-1;i++){rc[i]=rc[i+1];re[i]=re[i+1];}rn--;}}
   if(mn>hid_n){errors[row]=12;return;}
   pool_n=0;for(i=0;i<mn;i++){pool_cid[pool_n]=mc[i];pool_eph[pool_n]=me[i];pool_n++;}
   for(i=0;i<rn;i++){pool_cid[pool_n]=rc[i];pool_eph[pool_n]=re[i];pool_n++;}
  }
 }else if(hid_n<0){errors[row]=12;return;}
 for(i=0;i<hid_n;i++)idx[i]=i;if((err=rng_shuffle(&rng,idx,hid_n,max_draws))){errors[row]=err;return;}
 {int tmpc[128],tmpe[128];for(i=0;i<hid_n;i++){tmpc[i]=pool_cid[idx[i]];tmpe[i]=pool_eph[idx[i]];}for(i=0;i<hid_n;i++){pool_cid[i]=tmpc[i];pool_eph[i]=tmpe[i];}}
 own_n=i16(zcount,(row*2+viewer)*4+1);
 own_visible=(i8(active_side,row)==viewer&&u8(pint_p,(row*2+viewer)*n_pint+VISIBLE_DECK_INDEX)==2&&pint_v[(row*2+viewer)*n_pint+VISIBLE_DECK_INDEX]==turn_v);
 for(pos=0;pos<own_n;pos++)own_slot[pos]=zindex[((row*2+viewer)*4+1)*Z+pos];
 for(i=1;!own_visible&&i<own_n;i++){
  int key=own_slot[i],j=i-1;
  while(j>=0){
   int oa=own_slot[j],ob=key,ba=row*K+oa,bb=row*K+ob,ta=i16(ktmpl,ba),tb=i16(ktmpl,bb),na,nb,pa,pb;
   const unsigned char *sa,*sb;int lena,lenb;
   pa=u8(kstr_p,ba*n_kstr+5);if(pa==3){int ow=card_owner[ta];sa=hid_bytes+ow*16;lena=hid_len[ow];pa=2;}else if(pa==2){int sl=kstr_s[ba*n_kstr+5];sa=str_bytes+row*max_bytes+str_off[row*max_slots+sl];lena=str_len[row*max_slots+sl];}else{sa=0;lena=0;}
   pb=u8(kstr_p,bb*n_kstr+5);if(pb==3){int ow=card_owner[tb];sb=hid_bytes+ow*16;lenb=hid_len[ow];pb=2;}else if(pb==2){int sl=kstr_s[bb*n_kstr+5];sb=str_bytes+row*max_bytes+str_off[row*max_slots+sl];lenb=str_len[row*max_slots+sl];}else{sb=0;lenb=0;}
   na=dump_own_key(ka,192,ta,u8(kbool_p,ba*n_kbool+0),u8(kbool_v,ba*n_kbool+0),u8(kint_p,ba*n_kint+6),kint_v[ba*n_kint+6],u8(kint_p,ba*n_kint+5),kint_v[ba*n_kint+5],u8(kj_p,ba),kj_d[ba],kj_s[ba],pa,sa,lena,u8(kstr_p,ba*n_kstr+6),0,0,cid_bytes,cid_len);
   nb=dump_own_key(kb,192,tb,u8(kbool_p,bb*n_kbool+0),u8(kbool_v,bb*n_kbool+0),u8(kint_p,bb*n_kint+6),kint_v[bb*n_kint+6],u8(kint_p,bb*n_kint+5),kint_v[bb*n_kint+5],u8(kj_p,bb),kj_d[bb],kj_s[bb],pb,sb,lenb,u8(kstr_p,bb*n_kstr+6),0,0,cid_bytes,cid_len);
   if(na<0||nb<0){errors[row]=16;return;}
   if(cmp_key(ka,kb)<=0)break;own_slot[j+1]=own_slot[j];j--;
  }
  own_slot[j+1]=key;
 }
 if(!own_visible&&(err=rng_shuffle(&rng,own_slot,own_n,max_draws))){errors[row]=err;return;}
 {int* otmpl=specc+vc;for(i=0;i<own_n;i++){int src=own_slot[i],b=row*K+src;otmpl[i]=i16(ktmpl,b);otmpl[Z+i]=u8(kbool_p,b*n_kbool+0);otmpl[2*Z+i]=u8(kbool_v,b*n_kbool+0);otmpl[3*Z+i]=u8(kint_p,b*n_kint+6);otmpl[4*Z+i]=kint_v[b*n_kint+6];otmpl[5*Z+i]=u8(kint_p,b*n_kint+5);otmpl[6*Z+i]=kint_v[b*n_kint+5];otmpl[7*Z+i]=u8(kj_p,b);otmpl[8*Z+i]=kj_d[b];otmpl[9*Z+i]=kj_s[b];otmpl[10*Z+i]=u8(kstr_p,b*n_kstr+5);otmpl[11*Z+i]=kstr_s[b*n_kstr+5];otmpl[12*Z+i]=u8(kstr_p,b*n_kstr+6);otmpl[13*Z+i]=kstr_s[b*n_kstr+6];}}
 next_i=0;next_e=0;
 for(s=0;s<2;s++){
  if(pent[row*2+s]>next_e)next_e=pent[row*2+s];
  for(pos=0;pos<C;pos++){int b=((row*2+s)*C+pos);if(u8(cvalid,b)&&cent[b]>next_e)next_e=cent[b];}
  for(pos=0;pos<i16(zcount,(row*2+s)*4+2);pos++){slot=zindex[((row*2+s)*4+2)*Z+pos];if(kentity[row*K+slot]>next_e)next_e=kentity[row*K+slot];if(kinst[row*K+slot]>next_i)next_i=kinst[row*K+slot];}
 }
 for(pos=0;pos<i16(zcount,(row*2+viewer)*4+0);pos++){slot=zindex[((row*2+viewer)*4+0)*Z+pos];if(kentity[row*K+slot]>next_e)next_e=kentity[row*K+slot];if(kinst[row*K+slot]>next_i)next_i=kinst[row*K+slot];}
 for(i=0;i<keep_n;i++){slot=keep[i];if(kentity[row*K+slot]>next_e)next_e=kentity[row*K+slot];if(kinst[row*K+slot]>next_i)next_i=kinst[row*K+slot];}
 if(choosing){int cn=i16(choice_n,row);for(i=0;i<cn;i++){slot=choice_idx[row*CH+i];if(slot>=0&&slot<K){if(kentity[row*K+slot]>next_e)next_e=kentity[row*K+slot];if(kinst[row*K+slot]>next_i)next_i=kinst[row*K+slot];}}}
 if(choosing&&u8(operation,row)&&operation_card[row]>=0){
  int ci=operation_card[row];if(ci>=K||!u8(kvalid,row*K+ci)){errors[row]=1;return;}
  if(kentity[row*K+ci]>next_e)next_e=kentity[row*K+ci];
  if(kinst[row*K+ci]>next_i)next_i=kinst[row*K+ci];
 }
 {int used[512],un=0,need=hid_n+deck_n+own_n,assign=0,hand_slots[128],deck_slots[128],own_new[128];
  if(K>512||need>P){errors[row]=14;return;}
  for(i=0;i<K;i++)used[i]=0;
  for(s=0;s<2;s++)for(z=0;z<4;z++)for(pos=0;pos<i16(zcount,(row*2+s)*4+z);pos++){slot=zindex[((row*2+s)*4+z)*Z+pos];if(slot>=0&&slot<K)used[slot]=1;}
  for(pos=0;pos<hand_n;pos++){slot=zindex[((row*2+foe)*4+0)*Z+pos];int kept=0;for(i=0;i<keep_n;i++)if(keep[i]==slot)kept=1;if(!kept)used[slot]=0;}
  for(pos=0;pos<deck_n;pos++){slot=zindex[((row*2+foe)*4+1)*Z+pos];used[slot]=0;}
  for(pos=0;pos<own_n;pos++)used[own_slot[pos]]=0;
  if(choosing){
   int cn=i16(choice_n,row);
   for(i=0;i<cn;i++){
    int ci=choice_idx[row*CH+i];if(ci<0||ci>=K||!u8(kvalid,row*K+ci)){errors[row]=1;return;}
    used[ci]=1;
   }
   if(u8(operation,row)&&operation_card[row]>=0)used[operation_card[row]]=1;
  }
  for(i=0;i<K;i++)if(u8(protected_cards,row*K+i)){if(!u8(kvalid,row*K+i)){errors[row]=1;return;}used[i]=1;if(kentity[row*K+i]>next_e)next_e=kentity[row*K+i];if(kinst[row*K+i]>next_i)next_i=kinst[row*K+i];}
  for(i=0;i<K;i++)if(used[i]&&u8(card_views,row*K+i)){
   int pi=physical_cards[row*K+i];
   if(pi<0||pi>=K||pi==i||!u8(kvalid,row*K+pi)||u8(card_views,row*K+pi)
      ||kentity[row*K+pi]!=kentity[row*K+i]||kinst[row*K+pi]!=kinst[row*K+i]||i8(kside,row*K+pi)!=i8(kside,row*K+i)){errors[row]=3;return;}
   used[pi]=1;
  }
  if(next_i>2147483647-need||next_e>2147483647-need){errors[row]=14;return;}
  free_n=0;for(i=0;i<K;i++)if(!used[i]&&free_n<need)idx[free_n++]=i;
  if(free_n<need){errors[row]=14;return;}
  for(i=0;i<hid_n;i++){
   slot=idx[assign++];int cid=pool_cid[i],eph=pool_eph[i],ow,rewrite;
   next_i++;next_e++;write_new_card(kvalid,kentity,kinst,ktmpl,kside,kint_p,kint_v,kbool_p,kbool_v,kstr_p,kstr_s,ksrc,kplay,kj_p,kj_d,kj_s,row,K,slot,cid,foe,next_e,next_i,cat_int_p,cat_int_v,cat_bool_p,cat_bool_v,cat_str_p,cat_src,cat_play,n_kint,n_kbool,n_kstr,vc);
   ow=card_owner[cid];rewrite=!has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,ow,0)&&has_a;
   if(rewrite){int sl=intern_ascii(str_bytes,str_used,str_count,str_off,str_len,max_slots,max_bytes,row,max_slots,max_bytes,hid_bytes+ow*16,hid_len[ow]);int al=intern_ascii(str_bytes,str_used,str_count,str_off,str_len,max_slots,max_bytes,row,max_slots,max_bytes,hid_bytes+anhunqu*16,hid_len[anhunqu]);
    if(sl<0||al<0){errors[row]=16;return;}s8(kstr_p,(row*K+slot)*n_kstr+6,2);kstr_s[(row*K+slot)*n_kstr+6]=sl;s8(kstr_p,(row*K+slot)*n_kstr+5,2);kstr_s[(row*K+slot)*n_kstr+5]=al;}
   if(eph){s8(kint_p,(row*K+slot)*n_kint+6,2);kint_v[(row*K+slot)*n_kint+6]=turn_v;}
   hand_slots[i]=slot;
  }
  for(i=0;i<deck_n;i++){
   slot=idx[assign++];int cid=pool_cid[hid_n+i],eph=pool_eph[hid_n+i],ow,rewrite;
   next_i++;next_e++;write_new_card(kvalid,kentity,kinst,ktmpl,kside,kint_p,kint_v,kbool_p,kbool_v,kstr_p,kstr_s,ksrc,kplay,kj_p,kj_d,kj_s,row,K,slot,cid,foe,next_e,next_i,cat_int_p,cat_int_v,cat_bool_p,cat_bool_v,cat_str_p,cat_src,cat_play,n_kint,n_kbool,n_kstr,vc);
   ow=card_owner[cid];rewrite=!has_hero(cvalid,ctmpl,cbool_p,cbool_v,row,foe,C,n_cbool,ow,0)&&has_a;
   if(rewrite){int sl=intern_ascii(str_bytes,str_used,str_count,str_off,str_len,max_slots,max_bytes,row,max_slots,max_bytes,hid_bytes+ow*16,hid_len[ow]);int al=intern_ascii(str_bytes,str_used,str_count,str_off,str_len,max_slots,max_bytes,row,max_slots,max_bytes,hid_bytes+anhunqu*16,hid_len[anhunqu]);
    if(sl<0||al<0){errors[row]=16;return;}s8(kstr_p,(row*K+slot)*n_kstr+6,2);kstr_s[(row*K+slot)*n_kstr+6]=sl;s8(kstr_p,(row*K+slot)*n_kstr+5,2);kstr_s[(row*K+slot)*n_kstr+5]=al;}
   if(eph){s8(kint_p,(row*K+slot)*n_kint+6,2);kint_v[(row*K+slot)*n_kint+6]=turn_v;}
   deck_slots[i]=slot;
  }
  for(i=0;i<own_n;i++){
   int src=own_slot[i],cid=i16(ktmpl,row*K+src),base;
   int* otmpl=specc+vc;int cid2=otmpl[i];
   slot=idx[assign++];next_i++;next_e++;
   write_new_card(kvalid,kentity,kinst,ktmpl,kside,kint_p,kint_v,kbool_p,kbool_v,kstr_p,kstr_s,ksrc,kplay,kj_p,kj_d,kj_s,row,K,slot,cid2,viewer,next_e,next_i,cat_int_p,cat_int_v,cat_bool_p,cat_bool_v,cat_str_p,cat_src,cat_play,n_kint,n_kbool,n_kstr,vc);
   base=row*K+slot;
   s8(kbool_p,base*n_kbool+0,otmpl[Z+i]);s8(kbool_v,base*n_kbool+0,otmpl[2*Z+i]);
   s8(kint_p,base*n_kint+6,otmpl[3*Z+i]);kint_v[base*n_kint+6]=otmpl[4*Z+i];
   s8(kint_p,base*n_kint+5,otmpl[5*Z+i]);kint_v[base*n_kint+5]=otmpl[6*Z+i];
   s8(kj_p,base,otmpl[7*Z+i]);kj_d[base]=otmpl[8*Z+i];kj_s[base]=otmpl[9*Z+i];
   s8(kstr_p,base*n_kstr+5,otmpl[10*Z+i]);kstr_s[base*n_kstr+5]=otmpl[11*Z+i];
   s8(kstr_p,base*n_kstr+6,otmpl[12*Z+i]);kstr_s[base*n_kstr+6]=otmpl[13*Z+i];
   own_new[i]=slot;
  }
  for(i=0;i<keep_n;i++)zindex[((row*2+foe)*4+0)*Z+i]=keep[i];
  for(i=0;i<hid_n;i++)zindex[((row*2+foe)*4+0)*Z+keep_n+i]=hand_slots[i];
  for(i=0;i<deck_n;i++)zindex[((row*2+foe)*4+1)*Z+i]=deck_slots[i];
  for(i=0;i<own_n;i++)zindex[((row*2+viewer)*4+1)*Z+i]=own_new[i];
  for(s=0;s<C;s++){int b=((row*2+foe)*C+s);if(u8(cvalid,b)&&i16(ctmpl,b)==xiaozhi){xiaozhi_hp=chp[b];xiaozhi_aw=opt_bool(cbool_p,cbool_v,b,0,n_cbool);xiaozhi_ent=cent[b];}}
  for(i=0;i<hid_n;i++){slot=hand_slots[i];s8(kj_p,row*K+slot,0);
   if(xiaozhi_hp>0&&xiaozhi_aw){int d=rng_choice(&rng,3,max_draws,&err);if(err){errors[row]=err;return;}s8(kj_p,row*K+slot,2);kj_d[row*K+slot]=d-1;kj_s[row*K+slot]=xiaozhi_ent;}}
 }
 {unsigned long long r=0;int st=rng_below(&rng,1073741823ULL,max_draws,&r);if(st){errors[row]=st;return;}rng_out[row]=(long long)(1ULL+r);next_inst[row]=next_i;next_ent[row]=next_e;}
}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void domain_soa_sample(
#else
extern "C" void domain_soa_sample(
#endif
 int rows,int C,int K,int Z,int CH,int n_pint,int n_cbool,int n_kint,int n_kbool,int n_kstr,int max_draws,int max_slots,int max_bytes,int SW,int vc,
 const int* sp,int* scratch,const long long* seeds,const long long* viewers,const int* active,int* errors,
 const unsigned char* phase,const int* turn,const unsigned char* active_side,const unsigned char* first_side,
 const unsigned char* escalation,const unsigned char* pending,const unsigned char* choice_kind,const unsigned char* choice_side,
 const unsigned char* choice_n,const int* choice_idx,const unsigned char* operation,
 const unsigned char* pvalid,const int* pent,const unsigned char* porder,const unsigned char* pint_p,const int* pint_v,
 const unsigned char* cvalid,const unsigned char* ctmpl,const int* cent,const int* chp,const unsigned char* cbool_p,const unsigned char* cbool_v,
 const unsigned char* zcount,int* zindex,
 unsigned char* kvalid,int* kentity,int* kinst,unsigned char* ktmpl,unsigned char* kside,
 unsigned char* kint_p,int* kint_v,unsigned char* kbool_p,unsigned char* kbool_v,unsigned char* kstr_p,int* kstr_s,
 unsigned char* ksrc,unsigned char* kplay,unsigned char* kj_p,int* kj_d,int* kj_s,
 int* next_inst,int* next_ent,long long* rng_out,int* str_count,int* str_used,int* str_off,int* str_len,unsigned char* str_bytes,
 const unsigned char* card_derived,const int* card_owner,const int* cid_len,const unsigned char* cid_bytes,
 const int* hid_len,const unsigned char* hid_bytes,const int* char_seat,const int* kit_n,const int* kit,
 const unsigned char* cat_int_p,const int* cat_int_v,const unsigned char* cat_bool_p,const unsigned char* cat_bool_v,
 const unsigned char* cat_str_p,const unsigned char* cat_src,const unsigned char* cat_play,const int* operation_card,
 const unsigned char* revealed_count,const unsigned char* revealed_side,const int* revealed_instance,
 const unsigned char* card_views,const int* physical_cards,const unsigned char* protected_cards){
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(row>=rows)return;
 soa_sample_row(row,C,K,Z,CH,n_pint,n_cbool,n_kint,n_kbool,n_kstr,max_draws,max_slots,max_bytes,SW,vc,sp,scratch,seeds,viewers,active,errors,phase,turn,active_side,first_side,escalation,pending,choice_kind,choice_side,choice_n,choice_idx,operation,pvalid,pent,porder,pint_p,pint_v,cvalid,ctmpl,cent,chp,cbool_p,cbool_v,zcount,zindex,kvalid,kentity,kinst,ktmpl,kside,kint_p,kint_v,kbool_p,kbool_v,kstr_p,kstr_s,ksrc,kplay,kj_p,kj_d,kj_s,next_inst,next_ent,rng_out,str_count,str_used,str_off,str_len,str_bytes,card_derived,card_owner,cid_len,cid_bytes,hid_len,hid_bytes,char_seat,kit_n,kit,cat_int_p,cat_int_v,cat_bool_p,cat_bool_v,cat_str_p,cat_src,cat_play,operation_card,revealed_count,revealed_side,revealed_instance,card_views,physical_cards,protected_cards);
#else
 int row;for(row=0;row<rows;row++)soa_sample_row(row,C,K,Z,CH,n_pint,n_cbool,n_kint,n_kbool,n_kstr,max_draws,max_slots,max_bytes,SW,vc,sp,scratch,seeds,viewers,active,errors,phase,turn,active_side,first_side,escalation,pending,choice_kind,choice_side,choice_n,choice_idx,operation,pvalid,pent,porder,pint_p,pint_v,cvalid,ctmpl,cent,chp,cbool_p,cbool_v,zcount,zindex,kvalid,kentity,kinst,ktmpl,kside,kint_p,kint_v,kbool_p,kbool_v,kstr_p,kstr_s,ksrc,kplay,kj_p,kj_d,kj_s,next_inst,next_ent,rng_out,str_count,str_used,str_off,str_len,str_bytes,card_derived,card_owner,cid_len,cid_bytes,hid_len,hid_bytes,char_seat,kit_n,kit,cat_int_p,cat_int_v,cat_bool_p,cat_bool_v,cat_str_p,cat_src,cat_play,operation_card,revealed_count,revealed_side,revealed_instance,card_views,physical_cards,protected_cards);
#endif
}
'''
    return prefix + '\n#define SOA_SEED_WORDS '+str(seed_words)+'\n#define N_CHAR_TEMPLATES '+str(VOCAB_CHARS+1)+'\n' + emit_random_source() + body.replace('VISIBLE_DECK_INDEX',str(PLAYER_INT_FIELDS.index('deck_visible_turn'))).replace('SUPPORTED_ROOT_CHOICE', '('+'||'.join('kind=='+str(v)for v in sorted(SUPPORTED_CHOICE))+')')


_PTR_TYPES = (
    'int32*', 'int32*', 'int64*', 'int64*', 'int32*', 'int32*',
    'uint8*', 'int32*', 'uint8*', 'uint8*', 'uint8*', 'uint8*', 'uint8*', 'uint8*',
    'uint8*', 'int32*', 'uint8*', 'uint8*', 'int32*', 'uint8*', 'uint8*', 'int32*',
    'uint8*', 'uint8*', 'int32*', 'int32*', 'uint8*', 'uint8*', 'uint8*', 'int32*',
    'uint8*', 'int32*', 'int32*', 'uint8*', 'uint8*', 'uint8*', 'int32*', 'uint8*',
    'uint8*', 'uint8*', 'int32*', 'uint8*', 'uint8*', 'uint8*', 'int32*', 'int32*',
    'int32*', 'int32*', 'int64*', 'int32*', 'int32*', 'int32*', 'int32*', 'uint8*',
    'uint8*', 'int32*', 'int32*', 'uint8*', 'int32*', 'uint8*', 'int32*', 'int32*',
    'int32*', 'uint8*', 'int32*', 'uint8*', 'uint8*', 'uint8*', 'uint8*', 'uint8*',
)+('int32*','uint8*','uint8*','int32*','uint8*','int32*','uint8*')
_SCALAR_N = 15


@dataclass(frozen=True)
class PreparedSample:
    owner: object
    fields: dict
    outputs: dict
    viewers: object
    seeds: object
    active: object
    pointers: tuple
    sources: dict
    scratch: object
    information: object = None
    information_active: object = None
    extra_protected: object = None


def cpu_oracle(layout=None, **kwargs):
    kwargs.pop('device', None)
    kwargs.pop('allow_cpu_oracle', None)
    return DomainSoASampler(layout=layout, device='cpu', allow_cpu_oracle=True, **kwargs)


class DomainSoASampler:
    supports_full_duel_rules = False
    gpu_qualified = False

    def __init__(self, layout=None, catalog=None, *, device='cuda', allow_cpu_oracle=False,
                 max_draws=4096, guard=True, seed_words=1):
        import torch
        if layout is None:
            layout = DomainSoALayout()
        if not isinstance(layout, DomainSoALayout):
            raise TypeError('DomainSoALayout required')
        if type(max_draws) is not int or not 1 <= max_draws <= 1048576:
            raise ValueError('Bounded RNG rejection budget required')
        self.t = torch
        self.layout = layout
        self.max_draws = max_draws
        if type(seed_words)is not int or seed_words not in (1,5):raise ValueError("Explicit scalar63 or full320 seed ABI required")
        self.seed_words=seed_words
        if type(guard)is not bool:raise TypeError('Explicit guard selection required')
        if layout.max_zone>128 or layout.max_cards>512:
            raise ValueError('Direct sampler supports max_zone<=128 and max_cards<=512')
        self.guard = guard
        self.device = torch.device(device)
        self.specs = array_specs(layout)
        self.scratch_width = 24 * layout.max_zone + 2 * (VOCAB_CARDS + 1) + 64
        self._temporary = None
        self._module = None
        self._library = None
        self.closed = False
        self._information_guard = None
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('Actual CUDA required; no CPU sampler fallback')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            from .cuda_runtime import CudaModule
            spec = ['int32'] * _SCALAR_N + list(_PTR_TYPES)
            self._module = CudaModule(sampler_source(self.seed_words), {'domain_soa_sample': spec}, device=str(self.device))
        elif self.device.type == 'cpu' and allow_cpu_oracle is True:
            compiler = shutil.which('clang++') or shutil.which('g++')
            if compiler is None:
                raise RuntimeError('Native CPU SoA sampler requires a C++ compiler')
            self._temporary = tempfile.TemporaryDirectory(prefix='nte-soa-sampler-')
            root = Path(self._temporary.name)
            source = root / 'sampler.cpp'
            import sys
            library = root / ('sampler.dll' if os.name == 'nt' else
                              'sampler.dylib' if sys.platform == 'darwin' else 'sampler.so')
            source.write_text(sampler_source(self.seed_words), encoding='utf-8')
            try:
                result = subprocess.run(
                    [compiler, '-std=c++17', '-shared', '-fPIC', '-O2', str(source), '-o', str(library)],
                    capture_output=True, text=True, timeout=120)
            except BaseException:
                self.close()
                raise
            if result.returncode:
                self.close()
                raise RuntimeError('Native SoA sampler compilation failed: ' + result.stderr)
            self._library = ctypes.CDLL(str(library))
            self._kernel = self._library.domain_soa_sample
            self._kernel.argtypes = [ctypes.c_int] * _SCALAR_N + [ctypes.c_void_p] * len(_PTR_TYPES)
            self._kernel.restype = None
        else:
            raise ValueError('CUDA or explicitly enabled CPU oracle required')
        if catalog is None:
            catalog = prepare_sampler_catalog(self.device, torch)
        if not isinstance(catalog, FrozenSamplerCatalog) or catalog.device != self.device:
            raise ValueError('Frozen sampler catalog must already reside on the sampler device')
        self.catalog = catalog
        self._catalog_digest=torch.tensor(list(catalog.fingerprint),dtype=torch.uint8,device=self.device)
        self._specials = torch.tensor(
            [catalog.scalars[name] for name in
             ('nf01', 'a03', 'af01', 'ff01', 'q05', 'c02', 'a04', 'pf01',
              'xiaozhi', 'canhong', 'anhunqu', 'heiyu', 'dafutier', 'lingke')],
            dtype=torch.int32, device=self.device)
        if self.guard:
            from .domain_soa_information_guard import DomainInformationGuard
            try:
                self._information_guard=DomainInformationGuard(layout,device=self.device,
                    allow_cpu_oracle=allow_cpu_oracle,max_actions=800,window=16)
            except BaseException:
                self.close()
                raise

    def _torch_dtype(self, np_dtype):
        import numpy as np
        name = np.dtype(np_dtype).name.replace('_', '')
        if name == 'bool':
            return self.t.bool
        return getattr(self.t, _TORCH_DTYPES[name])

    def _fields(self, fields):
        if hasattr(fields, 'arrays'):
            fields = fields.arrays
        if not isinstance(fields, Mapping):
            raise TypeError('Named domain SoA field mapping required')
        fields=dict(fields)
        missing = [name for name in self.specs if name not in fields]
        if missing:
            raise TypeError('Missing domain SoA fields: ' + ','.join(missing))
        if set(fields)!=set(self.specs):
            raise TypeError('Unknown domain SoA columns')
        return fields

    def _column(self, fields, name, rows):
        dtype, tail = self.specs[name]
        value = fields[name]
        shape = (rows,) + tuple(tail)
        if (not isinstance(value, self.t.Tensor) or value.device != self.device
                or value.dtype != self._torch_dtype(dtype) or tuple(value.shape) != shape):
            raise ValueError('Resident ' + name + ' required')
        return value

    def _bytes(self, tensor):
        if tensor.dtype == self.t.bool:
            return tensor.view(self.t.uint8)
        if tensor.dtype in (self.t.int8, self.t.int16):
            return tensor.view(self.t.uint8)
        return tensor

    def prepare(self, fields, viewers, seeds, active):
        t = self.t
        if self.closed:
            raise RuntimeError('SoA sampler is closed')
        if self.device.type == 'cuda' and t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare sampler buffers before CUDA capture')
        fields = self._fields(fields)
        if (not isinstance(viewers, t.Tensor) or viewers.device != self.device
                or viewers.dtype != t.int64 or viewers.ndim != 1 or not viewers.is_contiguous()):
            raise ValueError('Resident int64 viewers required')
        rows = int(viewers.shape[0])
        if not 1 <= rows <= self.layout.game_capacity:
            raise ValueError('Nonempty viewer batch within layout.game_capacity required')
        if (not isinstance(seeds, t.Tensor) or seeds.device != self.device or seeds.dtype != t.int64
                or seeds.shape != ((rows,)if self.seed_words==1 else(rows,5)) or not seeds.is_contiguous()):
            raise ValueError('Resident seed words matching the bound seed ABI required')
        if (not isinstance(active, t.Tensor) or active.device != self.device or active.dtype != t.int32
                or active.shape != (rows,) or not active.is_contiguous()):
            raise ValueError('Resident int32 active mask required')
        sources = {name:self._column(fields,name,rows) for name in self.specs}
        packed = {name: sources[name].clone(memory_format=t.contiguous_format) for name in REQUIRED}
        outputs = {name:value.clone(memory_format=t.contiguous_format) for name,value in sources.items()}
        scratch = t.zeros((rows, self.scratch_width), dtype=t.int32, device=self.device)
        errors = outputs['row_error']
        L = self.layout
        cat = self.catalog.tables
        extra_protected=t.zeros((rows,self.layout.max_cards),dtype=t.bool,device=self.device)
        pointers = (
            self._specials, scratch, seeds, viewers, active, errors,
            self._bytes(packed['phase']), packed['turn'], self._bytes(packed['active_side']),
            self._bytes(packed['first_side']), self._bytes(packed['escalation_enabled']),
            self._bytes(packed['pending_choice_present']), self._bytes(packed['choice_kind']),
            self._bytes(packed['choice_side']), self._bytes(packed['choice_card_count']),
            packed['choice_card_index'], self._bytes(packed['operation_present']),
            self._bytes(packed['player_valid']), packed['player_entity_slot'],
            self._bytes(packed['player_character_count']), packed['player_int_presence'],
            packed['player_int_value'], self._bytes(packed['character_valid']),
            self._bytes(packed['character_template']), packed['character_entity_slot'],
            packed['character_hp'], packed['character_bool_presence'],
            self._bytes(packed['character_bool_value']), self._bytes(packed['zone_count']),
            outputs['zone_card_index'], self._bytes(outputs['card_valid']),
            outputs['card_entity_slot'], outputs['card_instance_slot'],
            self._bytes(outputs['card_template']), self._bytes(outputs['card_side']),
            outputs['card_int_presence'], outputs['card_int_value'],
            outputs['card_bool_presence'], self._bytes(outputs['card_bool_value']),
            outputs['card_str_presence'], outputs['card_str_slot'],
            outputs['card_source_presence'], outputs['card_play_options_presence'],
            outputs['card_jingu_mark_presence'], outputs['card_jingu_delta'],
            outputs['card_jingu_source_slot'], outputs['next_instance'], outputs['next_entity'],
            outputs['rng'], outputs['string_count'], outputs['string_bytes_used'],
            outputs['string_offset'], outputs['string_length'], outputs['string_bytes'],
            cat['card_derived'], cat['card_owner'], cat['card_id_len'], cat['card_id_bytes'],
            cat['char_id_len'], cat['char_id_bytes'], cat['char_seat'], cat['kit_n'], cat['kit'],
            cat['cat_int_p'], cat['cat_int_v'], cat['cat_bool_p'], cat['cat_bool_v'],
            cat['cat_str_p'], cat['cat_src'], cat['cat_play'], packed['op_card_index'],
            self._bytes(packed['player_revealed_count']),self._bytes(packed['player_revealed_side']),packed['player_revealed_instance'],
            self._bytes(packed['card_is_view']),packed['card_physical_index'],extra_protected.view(t.uint8),
        )
        if len(pointers) != len(_PTR_TYPES):
            raise RuntimeError('Sampler pointer arity drifted')
        info_active=t.zeros(rows,dtype=t.bool,device=self.device) if self.guard else None
        info=(self._information_guard.prepare(sources,outputs,viewers,info_active) if self.guard else None)
        return PreparedSample(self, packed, outputs, viewers, seeds, active, pointers, sources, scratch,info,info_active,extra_protected)

    def launch(self, batch):
        t = self.t
        if self.closed:
            raise RuntimeError('SoA sampler is closed')
        if not isinstance(batch, PreparedSample) or batch.owner is not self:
            raise ValueError('Prepared buffers must belong to this sampler backend')
        rows = int(batch.viewers.shape[0])
        for name,value in batch.fields.items():
            value.copy_(batch.sources[name])
        for name,value in batch.outputs.items():
            value.copy_(batch.sources[name])
        batch.scratch.zero_()
        source_error=batch.outputs['row_error']
        bad=((batch.active!=0)&(~batch.sources['row_valid']
             |(batch.sources['catalog_digest']!=self._catalog_digest[None,:]).any(1)
             |(batch.active<0)|(batch.active>1)))
        source_error.copy_(t.where(bad&(source_error==0),INVALID_INPUT,source_error))
        f=batch.sources;K=self.layout.max_cards
        physical=f['card_physical_index'].long();safe=physical.clamp(0,K-1)
        bad_view=(physical<0)|(physical>=K)|(physical==t.arange(K,device=self.device)[None,:])
        bad_view|=~f['card_valid'].gather(1,safe)|f['card_is_view'].gather(1,safe)
        for name in ('card_entity_slot','card_instance_slot','card_side'):
            bad_view|=f[name]!=f[name].gather(1,safe)
        unsupported_view = (batch.active != 0) & (f['card_is_view']&bad_view).any(1)
        source_error.copy_(t.where(unsupported_view & (source_error == 0), UNSUPPORTED_ROOT, source_error))
        # Keep every externally referenced card snapshot/alias immutable while
        # determinizing hidden zones. These are typed physical references.
        mask_refs=[];mask_live=[];CH=self.layout.max_choice_cards;M=self.layout.max_deferred_marks;DF=self.layout.max_deferred
        def protect(refs,live):mask_refs.append(refs.long().reshape(rows,-1));mask_live.append(live.reshape(rows,-1))
        protect(f['op_card_index'][:,None],f['operation_present'][:,None]&(f['op_card_index'][:,None]>=0))
        protect(f['op_choice_card_index'],f['operation_present'][:,None]&(t.arange(CH,device=self.device)[None,:]<f['op_choice_card_count'][:,None]))
        mark=(f['mark_queue_presence']==2)[:,None]&(t.arange(M,device=self.device)[None,:]<f['mark_queue_count'][:,None])&f['mark_operation_present']
        protect(f['mark_op_card_index'],mark&(f['mark_op_card_index']>=0))
        protect(f['mark_op_choice_card_index'],mark[...,None]&(t.arange(CH,device=self.device)[None,None,:]<f['mark_op_choice_card_count'][...,None]))
        protect(f['deferred_card_index'],(f['deferred_presence']==2)[:,None]&(t.arange(DF,device=self.device)[None,:]<f['deferred_count'][:,None])&(f['deferred_card_presence']==2))
        refs=t.cat(mask_refs,1);live=t.cat(mask_live,1);safe_refs=refs.clamp(0,K-1)
        invalid=(live&((refs<0)|(refs>=K)|~f['card_valid'].gather(1,safe_refs))).any(-1)
        invalid|=f['operation_present']&((f['op_choice_card_count']<0)|(f['op_choice_card_count']>CH))
        invalid|=(f['mark_queue_presence']==2)&((f['mark_queue_count']<0)|(f['mark_queue_count']>M))
        invalid|=(f['deferred_presence']==2)&((f['deferred_count']<0)|(f['deferred_count']>DF))
        source_error.copy_(t.where((batch.active!=0)&invalid&(source_error==0),INVALID_INPUT,source_error))
        counts=t.zeros((rows,K),dtype=t.int32,device=self.device);counts.scatter_add_(1,safe_refs,live.int())
        batch.extra_protected.copy_(counts>0)
        L = self.layout
        scalars = (
            rows, L.max_characters, L.max_cards, L.max_zone, L.max_choice_cards,
            N_PINT, N_CBOOL, N_KINT, N_KBOOL, N_KSTR, self.max_draws,
            L.max_string_slots, L.max_string_bytes, self.scratch_width, VOCAB_CARDS + 1,
        )
        error=batch.outputs['row_error']
        if self._module is not None:
            self._module.launch(
                'domain_soa_sample', list(scalars) + list(batch.pointers),
                grid=((rows + 63) // 64,), block=(64,), capture_safe=True)
        else:
            args = list(scalars) + [ctypes.c_void_p(item.data_ptr()) for item in batch.pointers]
            self._kernel(*args)
        changed=batch.outputs['card_valid']&(batch.outputs['card_entity_slot']!=batch.sources['card_entity_slot'])
        for name in ('card_is_view','card_is_snapshot'):batch.outputs[name].masked_fill_(changed,False)
        batch.outputs['card_physical_index'].masked_fill_(changed,0)
        if self.guard:
            batch.information_active.copy_(batch.active!=0)
            checked=self._information_guard.launch(batch.information)
            error.copy_(t.where(batch.active!=0,checked['errors'],error))
        result = dict(batch.outputs)
        # A failed sample is never a partially committed world. Keep the error
        # code but restore every other column, including inactive/history rows.
        failed = error != 0
        for name,value in batch.outputs.items():
            if name != 'row_error':
                mask=failed.view((rows,)+(1,)*(value.ndim-1))
                t.where(mask,batch.sources[name],value,out=value)
        return result

    def determinize(self, fields, viewers, seeds, active):
        return self.launch(self.prepare(fields, viewers, seeds, active))

    def close(self):
        if self.closed:
            return
        if self._information_guard is not None:
            self._information_guard.close()
        if self._module is not None:
            self._module.close()
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
