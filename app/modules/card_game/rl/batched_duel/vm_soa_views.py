"""Native SoA object-view bridge over CompiledRecordSchema field-major buffers.

Language-heap headers for domain objects keep real tags 5..10, a sentinel
offset, and extra=global_ref. Authoritative values stay in typed columns
indexed as offset*rows+row. The generic graph holds only language temporaries
and interned boxed scalar/key reads. Unbound heaps (soa == NULL) keep the
original scratch semantics. Full-rule and CUDA qualification stay false until
PRIMARY corpus/search tests pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType

from .record_soa import (
    ABSENT,
    BOOL,
    FLOAT,
    INT,
    NULL,
    REF,
    TEXT,
    CompiledRecordSchema,
    TAG_CARD,
    TAG_CHARACTER,
    TAG_DICT,
    TAG_LIST,
    TAG_PLAYER,
    TAG_TUPLE,
)

VIEW_OFFSET_SENTINEL = -2147483647 - 1
MISSING_REF = VIEW_OFFSET_SENTINEL
ATTACH_POINT = 'DOMAIN_SOA_ATTACH_POINT'
HELPER_NAMES = (
    'soa_attach',
    'soa_view_ref',
    'soa_global_ref',
    'soa_promote',
    'soa_is_view',
    'soa_clone',
    'soa_reset_caches',
)
supports_full_duel_rules = False
cuda_qualified = False

_KIND = {'record': 0, 'list': 1, 'map': 2, 'tuple': 3}
_GAME_KEYS = ('rng', 'sides', 'next_entity')


class SoAViewError(ValueError):
    pass


def _require_schema(schema):
    if not isinstance(schema, CompiledRecordSchema):
        raise TypeError('CompiledRecordSchema required')
    return schema


def iter_value_columns(schema):
    schema = _require_schema(schema)
    for pool in schema.pools:
        for field in pool.fields:
            yield pool, field, 'field'
        if pool.keys is not None:
            yield pool, pool.keys, 'keys'
        for value in pool.values:
            yield pool, value, 'values'


def total_record_fields(schema):
    return sum(1 for _ in iter_value_columns(_require_schema(schema)))


def value_cache_extent(schema):
    schema = _require_schema(schema)
    return int(schema.string_bytes) + int(schema.max_strings) + total_record_fields(schema)


def soa_view_layout(schema, *, constant_node_count=0):
    """Host/device buffer shapes for DomainSoAContext, excluding row count."""
    schema = _require_schema(schema)
    if type(constant_node_count) is not int or constant_node_count < 0:
        raise SoAViewError('Nonnegative constant_node_count required')
    fields = total_record_fields(schema)
    return MappingProxyType({
        'total_objects': schema.total_objects,
        'value_cache': value_cache_extent(schema),
        'clone_map': schema.total_objects,
        'clone_queue': schema.total_objects,
        'constant_adopt': constant_node_count,
        'total_record_fields': fields,
        'string_bytes': int(schema.string_bytes),
        'max_strings': schema.max_strings,
        'max_string_bytes': schema.max_string_bytes,
        'view_offset_sentinel': VIEW_OFFSET_SENTINEL,
        'index': 'offset*rows+row',
        'channels': MappingProxyType({
            'ABSENT': ABSENT, 'NULL': NULL, 'BOOL': BOOL, 'INT': INT,
            'FLOAT': FLOAT, 'TEXT': TEXT, 'REF': REF,
        }),
        'supports_full_duel_rules': False,
        'cuda_qualified': False,
        'schema_identity': schema.identity,
    })


@dataclass(frozen=True)
class SoAViewMetadata:
    layout: object
    helpers: tuple
    attach_point: str
    schema_identity: str


def emit_soa_metadata(schema, *, image=None):
    constant_node_count = 0
    if image is not None:
        nodes = getattr(image, 'nodes', None)
        if nodes is None:
            raise SoAViewError('VM image with constant nodes required')
        constant_node_count = int(getattr(nodes, 'shape', [0])[0])
    layout = soa_view_layout(schema, constant_node_count=constant_node_count)
    return SoAViewMetadata(layout, HELPER_NAMES, ATTACH_POINT, schema.identity)


def _c_int(value):
    value = int(value)
    if value <= -2147483648:
        return '(-2147483647-1)'
    return str(value)


def _c_u64(value):
    return str(int(value) & ((1 << 64) - 1)) + 'ULL'


def _c_bytes(data):
    if not data:
        return '0'
    return ','.join(str(b) for b in data)


def _c_list(values):
    return ','.join(_c_int(v) for v in values) if values else '0'


def _field_key(column):
    name = column.name
    if '.' in name:
        return name.split('.', 1)[1]
    return name


def _root_has_game_keys(schema):
    pool = schema.pools[schema.root]
    if pool.kind != 'record':
        return False
    names = {_field_key(field) for field in pool.fields}
    return all(key in names for key in _GAME_KEYS)


def _column_tables(schema):
    columns = []
    refs = []
    names = []
    name_bytes = []
    pool_field_start = []
    pool_key_col = []
    pool_value_start = []
    pool_value_count = []
    for pool in schema.pools:
        pool_field_start.append(len(columns))
        for field in pool.fields:
            key = _field_key(field)
            raw = key.encode('utf-8')
            names.append((len(name_bytes), len(raw)))
            name_bytes.extend(raw)
            columns.append(field)
            refs.append(field.references)
        if pool.keys is None:
            pool_key_col.append(-1)
        else:
            pool_key_col.append(len(columns))
            names.append((-1, 0))
            columns.append(pool.keys)
            refs.append(pool.keys.references)
        pool_value_start.append(len(columns))
        pool_value_count.append(len(pool.values))
        for value in pool.values:
            names.append((-1, 0))
            columns.append(value)
            refs.append(value.references)
    return columns, refs, names, bytes(name_bytes), pool_field_start, pool_key_col, pool_value_start, pool_value_count


def _emit_schema_tables(schema):
    columns, refs, names, name_bytes, field_start, key_col, value_start, value_count = _column_tables(schema)
    ref_flat = []
    ref_start = []
    for item in refs:
        ref_start.append(len(ref_flat))
        ref_flat.extend(item)
    allowed = []
    allowed_n = []
    for pool in schema.pools:
        allowed_n.append(len(pool.tags))
        allowed.extend(pool.tags)
        allowed.extend([0] * (4 - len(pool.tags)))
    parts = [
        'enum { SOA_ABSENT=' + str(ABSENT) + ',SOA_NULL=' + str(NULL) + ',SOA_BOOL=' + str(BOOL) + ',',
        ' SOA_INT=' + str(INT) + ',SOA_FLOAT=' + str(FLOAT) + ',SOA_TEXT=' + str(TEXT) + ',SOA_REF=' + str(REF) + ' };',
        'enum { SOA_KIND_RECORD=0,SOA_KIND_LIST=1,SOA_KIND_MAP=2,SOA_KIND_TUPLE=3 };',
        'const int SOA_VIEW_OFFSET=(-2147483647-1);',
        'const int SOA_MISSING=(-2147483647-1);',
        'const int SOA_POOL_COUNT=' + _c_int(len(schema.pools)) + ';',
        'const int SOA_ROOT=' + _c_int(schema.root) + ';',
        'const int SOA_TOTAL_OBJECTS=' + _c_int(schema.total_objects) + ';',
        'const int SOA_MAX_STRINGS=' + _c_int(schema.max_strings) + ';',
        'const int SOA_MAX_STRING_BYTES=' + _c_int(schema.max_string_bytes) + ';',
        'const int SOA_STRING_COUNT=' + _c_int(schema.strings_count) + ';',
        'const int SOA_BYTES_USED=' + _c_int(schema.bytes_used) + ';',
        'const int SOA_STRING_OFFSETS=' + _c_int(schema.string_offsets) + ';',
        'const int SOA_STRING_LENGTHS=' + _c_int(schema.string_lengths) + ';',
        'const int SOA_STRING_STATIC_REFS=' + _c_int(schema.string_static_refs) + ';',
        'const int SOA_STRING_HASHES=' + _c_int(schema.string_hashes) + ';',
        'const int SOA_STRING_BYTES=' + _c_int(schema.string_bytes) + ';',
        'const int SOA_TOTAL_RECORD_FIELDS=' + _c_int(len(columns)) + ';',
        'const int SOA_VALUE_CACHE_EXTENT=' + _c_int(value_cache_extent(schema)) + ';',
        'const int SOA_LEN_I64=' + _c_int(schema.lengths['i64']) + ';',
        'const int SOA_LEN_F64=' + _c_int(schema.lengths['f64']) + ';',
        'const int SOA_LEN_I32=' + _c_int(schema.lengths['i32']) + ';',
        'const int SOA_LEN_U8=' + _c_int(schema.lengths['u8']) + ';',
        'const int SOA_HAS_GAME_KEYS=' + ('1' if _root_has_game_keys(schema) else '0') + ';',
        'const int SOA_POOL_KIND[]={' + _c_list(_KIND[p.kind] for p in schema.pools) + '};',
        'const int SOA_POOL_CAP[]={' + _c_list(p.capacity for p in schema.pools) + '};',
        'const int SOA_POOL_BASE[]={' + _c_list(p.base for p in schema.pools) + '};',
        'const int SOA_POOL_USED[]={' + _c_list(p.used for p in schema.pools) + '};',
        'const int SOA_POOL_TAG[]={' + _c_list(p.tag for p in schema.pools) + '};',
        'const int SOA_POOL_COUNT_COL[]={' + _c_list(p.count for p in schema.pools) + '};',
        'const int SOA_POOL_ORDER[]={' + _c_list(p.order for p in schema.pools) + '};',
        'const int SOA_POOL_MAX_ITEMS[]={' + _c_list(p.max_items for p in schema.pools) + '};',
        'const int SOA_POOL_NFIELDS[]={' + _c_list(len(p.fields) for p in schema.pools) + '};',
        'const int SOA_POOL_FIELD_START[]={' + _c_list(field_start) + '};',
        'const int SOA_POOL_KEY_COL[]={' + _c_list(key_col) + '};',
        'const int SOA_POOL_VALUE_START[]={' + _c_list(value_start) + '};',
        'const int SOA_POOL_VALUE_COUNT[]={' + _c_list(value_count) + '};',
        'const int SOA_POOL_NTAGS[]={' + _c_list(allowed_n) + '};',
        'const int SOA_POOL_TAGS[]={' + _c_list(allowed) + '};',
    ]
    if columns:
        parts.extend([
            'const int SOA_COL_TAG[]={' + _c_list(c.tag for c in columns) + '};',
            'const int SOA_COL_INTEGER[]={' + _c_list(c.integer for c in columns) + '};',
            'const int SOA_COL_REAL[]={' + _c_list(c.real for c in columns) + '};',
            'const int SOA_COL_WORD[]={' + _c_list(c.word for c in columns) + '};',
            'const int SOA_COL_STRIDE[]={' + _c_list(c.stride for c in columns) + '};',
            'const int SOA_COL_EXTENT[]={' + _c_list(c.extent for c in columns) + '};',
            'const int SOA_COL_SCALAR[]={' + _c_list(c.scalar_mask for c in columns) + '};',
            'const int SOA_COL_NREFS[]={' + _c_list(len(r) for r in refs) + '};',
            'const int SOA_COL_REF_START[]={' + _c_list(ref_start) + '};',
            'const int SOA_NAME_OFF[]={' + _c_list(n[0] for n in names) + '};',
            'const int SOA_NAME_LEN[]={' + _c_list(n[1] for n in names) + '};',
        ])
        parts.append('const int SOA_COL_REFS[]={' + _c_list(ref_flat) + '};')
        parts.append('const unsigned char SOA_NAME_BYTES[]={' + _c_bytes(name_bytes) + '};')
    else:
        parts.extend([
            'const int SOA_COL_TAG[]={0};const int SOA_COL_INTEGER[]={-1};const int SOA_COL_REAL[]={-1};',
            'const int SOA_COL_WORD[]={-1};const int SOA_COL_STRIDE[]={1};const int SOA_COL_EXTENT[]={0};',
            'const int SOA_COL_SCALAR[]={0};const int SOA_COL_NREFS[]={0};const int SOA_COL_REF_START[]={0};',
            'const int SOA_COL_REFS[]={0};const int SOA_NAME_OFF[]={-1};const int SOA_NAME_LEN[]={0};',
            'const unsigned char SOA_NAME_BYTES[]={0};',
        ])
    rng = sides = nxt = -1
    if _root_has_game_keys(schema):
        root = schema.pools[schema.root]
        names_map = {_field_key(f): i for i, f in enumerate(root.fields)}
        rng, sides, nxt = names_map['rng'], names_map['sides'], names_map['next_entity']
    parts.append('const int SOA_GAME_RNG_FIELD=' + _c_int(rng) + ';')
    parts.append('const int SOA_GAME_SIDES_FIELD=' + _c_int(sides) + ';')
    parts.append('const int SOA_GAME_NEXT_ENTITY_FIELD=' + _c_int(nxt) + ';')
    return '\n'.join(parts) + '\n'


def _emit_runtime_head():
    return r'''
struct DomainSoAContext {
 long long* i64; double* f64; int* i32; unsigned char* u8;
 int rows; int row;
 int* views; int* value_cache; int* clone_map; int* clone_queue; int* constant_adopt;
 int constant_node_count;
};
__device__ inline int make_none(Heap* heap);
__device__ inline int make_bool(Heap* heap, int val);
__device__ inline int make_int64(Heap* heap, long long val);
__device__ inline int make_float64(Heap* heap, double val);
__device__ inline int make_utf8(Heap* heap, const char* str, int byte_len);
__device__ inline int read_int64(Heap* heap, int ref, long long* out_val);
__device__ inline int read_float64(Heap* heap, int ref, double* out_val);
__device__ inline const unsigned char* get_payload_ptr(const Heap* heap, const NodeView* node);
__device__ inline int get_node(const Heap* heap, int ref, NodeView* out_node);
__device__ inline int get_edge(const Heap* heap, const NodeView* node, int idx, int* out_k, int* out_v);
__device__ inline int key_equal(Heap* heap, int ref_a, int ref_b);
__device__ inline int soa_fail(Heap* heap,int code){if(heap&&heap->error==ERROR_OK)heap->error=code;return -1;}
__device__ inline DomainSoAContext* soa_ctx(const Heap* heap){return heap?heap->soa:0;}
__device__ inline unsigned char soa_load_u8(const DomainSoAContext* s,int off){return s->u8[(long long)off*s->rows+s->row];}
__device__ inline void soa_store_u8(DomainSoAContext* s,int off,unsigned char v){s->u8[(long long)off*s->rows+s->row]=v;}
__device__ inline int soa_load_i32(const DomainSoAContext* s,int off){return s->i32[(long long)off*s->rows+s->row];}
__device__ inline void soa_store_i32(DomainSoAContext* s,int off,int v){s->i32[(long long)off*s->rows+s->row]=v;}
__device__ inline long long soa_load_i64(const DomainSoAContext* s,int off){return s->i64[(long long)off*s->rows+s->row];}
__device__ inline void soa_store_i64(DomainSoAContext* s,int off,long long v){s->i64[(long long)off*s->rows+s->row]=v;}
__device__ inline double soa_load_f64(const DomainSoAContext* s,int off){return s->f64[(long long)off*s->rows+s->row];}
__device__ inline void soa_store_f64(DomainSoAContext* s,int off,double v){s->f64[(long long)off*s->rows+s->row]=v;}
__device__ inline int soa_load_view(const DomainSoAContext* s,int gid){return s->views[(long long)gid*s->rows+s->row];}
__device__ inline void soa_store_view(DomainSoAContext* s,int gid,int ref){s->views[(long long)gid*s->rows+s->row]=ref;}
__device__ inline int soa_load_cache(const DomainSoAContext* s,int idx){return s->value_cache[(long long)idx*s->rows+s->row];}
__device__ inline void soa_store_cache(DomainSoAContext* s,int idx,int ref){s->value_cache[(long long)idx*s->rows+s->row]=ref;}
__device__ inline int soa_cache_ok(const Heap* heap,int ref){return ref>=0&&ref<heap->counts[0];}

__device__ inline int soa_ready(const Heap* heap){DomainSoAContext* s=soa_ctx(heap);return s&&s->rows>0&&s->row>=0&&s->row<s->rows&&s->i64&&s->f64&&s->i32&&s->u8&&s->views&&s->value_cache&&s->clone_map&&s->clone_queue;}
__device__ inline int soa_locate(int global_ref,int* pool,int* slot){
 if(global_ref<1)return 0;
 for(int i=0;i<SOA_POOL_COUNT;i++){
  if(SOA_POOL_BASE[i]<global_ref && global_ref<=SOA_POOL_BASE[i]+SOA_POOL_CAP[i]){
   *pool=i;*slot=global_ref-SOA_POOL_BASE[i]-1;return 1;
  }
 }
 return 0;
}
__device__ inline int soa_pool_allows_tag(int pool,int tag){
 int n=SOA_POOL_NTAGS[pool];
 for(int i=0;i<n;i++)if(SOA_POOL_TAGS[pool*4+i]==tag)return 1;
 return 0;
}
__device__ inline int soa_header_view(const Heap* heap,int ref){
 if(!heap||ref<0||ref>=heap->counts[0])return 0;
 return heap->nodes[ref*4]>=TAG_LIST&&heap->nodes[ref*4]<=TAG_CARD_ENTITY&&heap->nodes[ref*4+1]==SOA_VIEW_OFFSET;
}
__device__ inline int soa_is_view(Heap* heap,int vm_ref){
 return soa_ready(heap)&&soa_header_view(heap,vm_ref);
}
__device__ inline void soa_attach(Heap* heap,DomainSoAContext* soa){if(heap)heap->soa=soa;}
__device__ inline void soa_reset_range(int* buf,int count,DomainSoAContext* s){
 if(!buf)return;
 for(int i=0;i<count;i++)buf[(long long)i*s->rows+s->row]=SOA_MISSING;
}
__device__ inline void soa_reset_caches(Heap* heap){
 DomainSoAContext* s=soa_ctx(heap);if(!s||s->rows<=0||s->row<0||s->row>=s->rows)return;
 soa_reset_range(s->views,SOA_TOTAL_OBJECTS,s);
 soa_reset_range(s->value_cache,SOA_VALUE_CACHE_EXTENT,s);
 soa_reset_range(s->clone_map,SOA_TOTAL_OBJECTS,s);
 soa_reset_range(s->clone_queue,SOA_TOTAL_OBJECTS,s);
 if(s->constant_adopt&&s->constant_node_count>0)soa_reset_range(s->constant_adopt,s->constant_node_count,s);
}
__device__ inline int soa_alloc_header(Heap* heap,int tag,int global_ref,int count){
 if(heap->counts[0]+1>heap->max_nodes)return soa_fail(heap,ERROR_NODE_CAPACITY);
 int ref=heap->counts[0]++;
 heap->nodes[ref*4+0]=tag;
 heap->nodes[ref*4+1]=SOA_VIEW_OFFSET;
 heap->nodes[ref*4+2]=count;
 heap->nodes[ref*4+3]=global_ref;
 return ref;
}
__device__ inline int soa_refresh_header(Heap* heap,int ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!s||!soa_header_view(heap,ref))return 0;
 int gid=heap->nodes[ref*4+3],pool=0,slot=0;
 if(!soa_locate(gid,&pool,&slot)){soa_fail(heap,ERROR_INVALID_REF);return 0;}
 int used=soa_load_i32(s,SOA_POOL_USED[pool]);
 int tag=soa_load_u8(s,SOA_POOL_TAG[pool]+slot);
 int count=soa_load_i32(s,SOA_POOL_COUNT_COL[pool]+slot);
 if(slot>=used||!soa_pool_allows_tag(pool,tag)||count<0||count>SOA_POOL_MAX_ITEMS[pool]){soa_fail(heap,ERROR_INVALID_REF);return 0;}
 heap->nodes[ref*4+0]=tag;
 heap->nodes[ref*4+2]=count;
 heap->nodes[ref*4+3]=gid;
 return 1;
}
__device__ inline int soa_view_ref(Heap* heap,int global_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_ready(heap))return soa_fail(heap,ERROR_INVALID_REF);
 int pool=0,slot=0;if(!soa_locate(global_ref,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 int gid=global_ref-1;
 int interned=soa_load_view(s,gid);
 if(soa_cache_ok(heap,interned)&&soa_header_view(heap,interned)&&heap->nodes[interned*4+3]==global_ref){
  soa_refresh_header(heap,interned);return interned;
 }
 int used=soa_load_i32(s,SOA_POOL_USED[pool]);
 if(slot>=used)return soa_fail(heap,ERROR_INVALID_REF);
 int tag=soa_load_u8(s,SOA_POOL_TAG[pool]+slot);
 int count=soa_load_i32(s,SOA_POOL_COUNT_COL[pool]+slot);
 int ref=soa_alloc_header(heap,tag,global_ref,count);
 if(ref<0)return -1;
 soa_store_view(s,gid,ref);
 return ref;
}
__device__ inline int soa_global_ref(Heap* heap,int vm_ref){
 if(!soa_is_view(heap,vm_ref))return soa_fail(heap,ERROR_INVALID_REF);
 return heap->nodes[vm_ref*4+3];
}
__device__ inline int soa_try_get_node(const Heap* heap,int ref,NodeView* out_node){
 Heap* h=(Heap*)heap;
 if(!soa_ready(h)||!soa_header_view(h,ref))return 0;
 if(!soa_refresh_header(h,ref)){if(out_node){out_node->tag=0;out_node->offset=0;out_node->count=0;out_node->extra=0;out_node->is_constant=0;}return 1;}
 out_node->tag=h->nodes[ref*4+0];
 out_node->offset=SOA_VIEW_OFFSET;
 out_node->count=h->nodes[ref*4+2];
 out_node->extra=h->nodes[ref*4+3];
 out_node->is_constant=0;
 return 1;
}
__device__ inline long long soa_hash_bytes(const unsigned char* p,int n){
 unsigned long long h=14695981039346656037ULL;
 for(int i=0;i<n;i++)h=(h^(unsigned long long)p[i])*1099511628211ULL;
 return (long long)h;
}
__device__ inline int soa_intern_bytes(Heap* heap,const unsigned char* p,int n){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_ready(heap)||n<0)return soa_fail(heap,ERROR_INVALID_REF);
 int count=soa_load_i32(s,SOA_STRING_COUNT),used=soa_load_i32(s,SOA_BYTES_USED);
 long long h=soa_hash_bytes(p,n);
 for(int i=0;i<count;i++){
  if(soa_load_i32(s,SOA_STRING_LENGTHS+i)!=n||soa_load_i64(s,SOA_STRING_HASHES+i)!=h)continue;
  int off=soa_load_i32(s,SOA_STRING_OFFSETS+i);int ok=1;
  for(int j=0;j<n;j++)if(soa_load_u8(s,SOA_STRING_BYTES+off+j)!=p[j]){ok=0;break;}
  if(ok)return i+1;
 }
 if(count>=SOA_MAX_STRINGS||used+n>SOA_MAX_STRING_BYTES)return soa_fail(heap,ERROR_PAYLOAD_CAPACITY);
 soa_store_i32(s,SOA_STRING_OFFSETS+count,used);
 soa_store_i32(s,SOA_STRING_LENGTHS+count,n);
 soa_store_i64(s,SOA_STRING_HASHES+count,h);
 for(int j=0;j<n;j++)soa_store_u8(s,SOA_STRING_BYTES+used+j,p[j]);
 soa_store_i32(s,SOA_STRING_COUNT,count+1);
 soa_store_i32(s,SOA_BYTES_USED,used+n);
 soa_store_i32(s,SOA_STRING_STATIC_REFS+count,SOA_MISSING);
 return count+1;
}
__device__ inline int soa_box_string_id(Heap* heap,int sid){
 DomainSoAContext* s=soa_ctx(heap);
 if(sid<1||sid>soa_load_i32(s,SOA_STRING_COUNT))return soa_fail(heap,ERROR_INVALID_REF);
 int frozen=soa_load_i32(s,SOA_STRING_STATIC_REFS+sid-1);
 if(frozen!=SOA_MISSING){
  NodeView n;if(soa_cache_ok(heap,frozen)&&get_node(heap,frozen,&n)&&n.tag==TAG_STR)return frozen;
 }
 int cache=SOA_STRING_BYTES+(sid-1);
 int interned=soa_load_cache(s,cache);
 if(soa_cache_ok(heap,interned)){
  NodeView n;if(get_node(heap,interned,&n)&&n.tag==TAG_STR)return interned;
 }
 int off=soa_load_i32(s,SOA_STRING_OFFSETS+sid-1),len=soa_load_i32(s,SOA_STRING_LENGTHS+sid-1);
 if(heap->counts[2]+len>heap->max_payload)return soa_fail(heap,ERROR_PAYLOAD_CAPACITY);
 char tmp[1024];const char* ptr=0;
 if(len<=1024){for(int i=0;i<len;i++)tmp[i]=(char)soa_load_u8(s,SOA_STRING_BYTES+off+i);ptr=tmp;}
 else {
  if(heap->counts[2]+len>heap->max_payload)return soa_fail(heap,ERROR_PAYLOAD_CAPACITY);
  int payload=heap->counts[2];
  for(int i=0;i<len;i++)heap->payload[payload+i]=soa_load_u8(s,SOA_STRING_BYTES+off+i);
  heap->counts[2]+=len;
  if(heap->counts[0]+1>heap->max_nodes){heap->counts[2]=payload;return soa_fail(heap,ERROR_NODE_CAPACITY);}
  int ref=heap->counts[0]++;heap->nodes[ref*4+0]=TAG_STR;heap->nodes[ref*4+1]=payload;heap->nodes[ref*4+2]=len;heap->nodes[ref*4+3]=0;
  soa_store_cache(s,cache,ref);return ref;
 }
 int boxed=make_utf8(heap,ptr,len);if(boxed<0)return -1;soa_store_cache(s,cache,boxed);return boxed;
}
__device__ inline int soa_name_ref(Heap* heap,int col){
 DomainSoAContext* s=soa_ctx(heap);
 int off=SOA_NAME_OFF[col],len=SOA_NAME_LEN[col];
 if(off<0||len<0)return soa_fail(heap,ERROR_INVALID_REF);
 int cache=SOA_STRING_BYTES+SOA_MAX_STRINGS+col;
 int interned=soa_load_cache(s,cache);
 if(soa_cache_ok(heap,interned)){
  NodeView n;if(get_node(heap,interned,&n)&&n.tag==TAG_STR)return interned;
 }
 int sid=soa_intern_bytes(heap,SOA_NAME_BYTES+off,len);if(sid<1)return -1;
 int boxed=soa_box_string_id(heap,sid);if(boxed<0)return -1;soa_store_cache(s,cache,boxed);return boxed;
}
__device__ inline int soa_clear_cell_cache(DomainSoAContext* s,int col,int index){
 int tag_off=SOA_COL_TAG[col]+index;if(tag_off<0)return 0;
 soa_store_cache(s,tag_off,SOA_MISSING);return 1;
}
__device__ inline int soa_box_cell(Heap* heap,int col,int index){
 DomainSoAContext* s=soa_ctx(heap);
 int kind=soa_load_u8(s,SOA_COL_TAG[col]+index);
 if(kind==SOA_ABSENT)return soa_fail(heap,ERROR_INVALID_REF);
 if(kind==SOA_NULL){
  int cache=SOA_COL_TAG[col]+index,interned=soa_load_cache(s,cache);
  if(soa_cache_ok(heap,interned)){NodeView n;if(get_node(heap,interned,&n)&&n.tag==TAG_NONE)return interned;}
  int boxed=make_none(heap);if(boxed<0)return -1;soa_store_cache(s,cache,boxed);return boxed;
 }
 int cache=SOA_COL_TAG[col]+index,interned=soa_load_cache(s,cache);
 if(kind==SOA_REF){
  int gid=soa_load_i32(s,SOA_COL_WORD[col]+index);
  return soa_view_ref(heap,gid);
 }
 if(soa_cache_ok(heap,interned)){
  NodeView n;if(get_node(heap,interned,&n)){
   int want=kind==SOA_BOOL?TAG_BOOL:kind==SOA_INT?TAG_INT:kind==SOA_FLOAT?TAG_FLOAT:kind==SOA_TEXT?TAG_STR:-1;
   if(want>=0&&n.tag==want)return interned;
  }
 }
 int boxed=-1;
 if(kind==SOA_BOOL){
  if(SOA_COL_INTEGER[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  boxed=make_bool(heap,soa_load_i64(s,SOA_COL_INTEGER[col]+index)?1:0);
 }else if(kind==SOA_INT){
  if(SOA_COL_INTEGER[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  boxed=make_int64(heap,soa_load_i64(s,SOA_COL_INTEGER[col]+index));
 }else if(kind==SOA_FLOAT){
  if(SOA_COL_REAL[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  boxed=make_float64(heap,soa_load_f64(s,SOA_COL_REAL[col]+index));
 }else if(kind==SOA_TEXT){
  boxed=soa_box_string_id(heap,soa_load_i32(s,SOA_COL_WORD[col]+index));
 }else return soa_fail(heap,ERROR_TYPE_MISMATCH);
 if(boxed<0)return -1;soa_store_cache(s,cache,boxed);return boxed;
}
__device__ inline int soa_promote(Heap* heap,int vm_ref,int type_id);
__device__ inline int soa_write_cell(Heap* heap,int col,int index,int value);
__device__ inline int soa_write_encoded(Heap* heap,int col,int index,int value,int* memo_old,int* memo_new,int* memo_n,int* active,int* active_n);
__device__ inline int soa_item_index(int pool,int slot,int ordinal){return slot*SOA_POOL_MAX_ITEMS[pool]+ordinal;}
__device__ inline int soa_get_edge(const Heap* heap,const NodeView* node,int idx,int* out_k,int* out_v){
 Heap* h=(Heap*)heap;DomainSoAContext* s=soa_ctx(h);
 if(!s||!node||node->offset!=SOA_VIEW_OFFSET)return 0;
 int pool=0,slot=0;if(!soa_locate(node->extra,&pool,&slot)){soa_fail(h,ERROR_INVALID_REF);return 0;}
 if(idx<0||idx>=node->count)return 0;
 int kind=SOA_POOL_KIND[pool];
 if(kind==SOA_KIND_RECORD){
  int order=soa_load_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,idx));
  if(order<0||order>=SOA_POOL_NFIELDS[pool]){soa_fail(h,ERROR_INVALID_REF);return 0;}
  int col=SOA_POOL_FIELD_START[pool]+order;
  int key=soa_name_ref(h,col);if(key<0)return 0;
  int val=soa_box_cell(h,col,slot);if(val<0)return 0;
  *out_k=key;*out_v=val;return 1;
 }
 if(kind==SOA_KIND_MAP){
  int index=soa_item_index(pool,slot,idx);
  int key=soa_box_cell(h,SOA_POOL_KEY_COL[pool],index);if(key<0)return 0;
  int val=soa_box_cell(h,SOA_POOL_VALUE_START[pool],index);if(val<0)return 0;
  *out_k=key;*out_v=val;return 1;
 }
 int col=SOA_POOL_VALUE_START[pool]+(kind==SOA_KIND_TUPLE?idx:0);
 int index=kind==SOA_KIND_TUPLE?slot:soa_item_index(pool,slot,idx);
 int val=soa_box_cell(h,col,index);if(val<0)return 0;
 *out_k=val;*out_v=-1;return 1;
}
__device__ inline int soa_text_equals_name(Heap* heap,int ref,int col){
 NodeView n;if(!get_node(heap,ref,&n)||n.tag!=TAG_STR)return 0;
 const unsigned char* p=get_payload_ptr(heap,&n);int len=SOA_NAME_LEN[col];
 if(n.count!=len)return 0;int off=SOA_NAME_OFF[col];
 for(int i=0;i<len;i++)if(p[i]!=SOA_NAME_BYTES[off+i])return 0;
 return 1;
}
__device__ inline int soa_dict_find(Heap* heap,int dict_ref,int key_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,dict_ref))return -2;
 int gid=heap->nodes[dict_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 if(SOA_POOL_KIND[pool]==SOA_KIND_LIST||SOA_POOL_KIND[pool]==SOA_KIND_TUPLE)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 soa_refresh_header(heap,dict_ref);int count=heap->nodes[dict_ref*4+2];
 if(SOA_POOL_KIND[pool]==SOA_KIND_RECORD){
  for(int i=0;i<count;i++){
   int order=soa_load_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,i));
   int col=SOA_POOL_FIELD_START[pool]+order;
   if(soa_text_equals_name(heap,key_ref,col))return i;
   int name=soa_name_ref(heap,col);if(name<0)return -1;
   int eq=key_equal(heap,name,key_ref);if(heap->error)return -1;if(eq)return i;
  }
  return -1;
 }
 int key_col=SOA_POOL_KEY_COL[pool];
 for(int i=0;i<count;i++){
  int boxed=soa_box_cell(heap,key_col,soa_item_index(pool,slot,i));if(boxed<0)return -1;
  int eq=key_equal(heap,boxed,key_ref);if(heap->error)return -1;if(eq)return i;
 }
 return -1;
}
__device__ inline int soa_dict_get(Heap* heap,int dict_ref,int key_ref,int default_ref){
 int idx=soa_dict_find(heap,dict_ref,key_ref);if(idx==-2)return -2;if(heap->error)return -1;
 if(idx<0)return default_ref;
 NodeView n;if(!get_node(heap,dict_ref,&n))return soa_fail(heap,ERROR_INVALID_REF);
 int k=0,v=0;if(!soa_get_edge(heap,&n,idx,&k,&v))return -1;return v;
}
__device__ inline int soa_col_accepts_ref(int col,int pool){
 int n=SOA_COL_NREFS[col],start=SOA_COL_REF_START[col];
 for(int i=0;i<n;i++)if(SOA_COL_REFS[start+i]==pool)return 1;
 return 0;
}
__device__ inline int soa_write_cell(Heap* heap,int col,int index,int value){
 DomainSoAContext* s=soa_ctx(heap);
 NodeView n;if(!get_node(heap,value,&n))return soa_fail(heap,ERROR_INVALID_REF);
 soa_clear_cell_cache(s,col,index);
 if(n.tag==TAG_NONE){soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_NULL);return 0;}
 if(n.tag==TAG_BOOL){
  if(!(SOA_COL_SCALAR[col]&(1<<SOA_BOOL))||SOA_COL_INTEGER[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  soa_store_i64(s,SOA_COL_INTEGER[col]+index,n.extra?1:0);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_BOOL);return 0;
 }
 if(n.tag==TAG_INT){
  if(!(SOA_COL_SCALAR[col]&(1<<SOA_INT))||SOA_COL_INTEGER[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  long long v=0;if(read_int64(heap,value,&v)<0)return -1;
  soa_store_i64(s,SOA_COL_INTEGER[col]+index,v);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_INT);return 0;
 }
 if(n.tag==TAG_FLOAT){
  if(!(SOA_COL_SCALAR[col]&(1<<SOA_FLOAT))||SOA_COL_REAL[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  double v=0;if(read_float64(heap,value,&v)<0)return -1;
  soa_store_f64(s,SOA_COL_REAL[col]+index,v);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_FLOAT);return 0;
 }
 if(n.tag==TAG_STR){
  if(!(SOA_COL_SCALAR[col]&(1<<SOA_TEXT))||SOA_COL_WORD[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  const unsigned char* p=get_payload_ptr(heap,&n);
  int sid=soa_intern_bytes(heap,p?p:(const unsigned char*)"",n.count);if(sid<1)return -1;
  soa_store_i32(s,SOA_COL_WORD[col]+index,sid);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_TEXT);return 0;
 }
 if(n.tag<TAG_LIST||n.tag>TAG_CARD_ENTITY)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 if(!SOA_COL_NREFS[col]||SOA_COL_WORD[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 int child=-1;
 if(soa_header_view(heap,value)){
  int gid=heap->nodes[value*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
  if(!soa_col_accepts_ref(col,pool))return soa_fail(heap,ERROR_TYPE_MISMATCH);
  child=gid;
 }else{
  int chosen=-1;
  int start=SOA_COL_REF_START[col];
  for(int i=0;i<SOA_COL_NREFS[col];i++){
   int tid=SOA_COL_REFS[start+i];
   if(!soa_pool_allows_tag(tid,n.tag))continue;
   if(chosen<0||SOA_POOL_NFIELDS[tid]<SOA_POOL_NFIELDS[chosen])chosen=tid;
  }
  if(chosen<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  int promoted=soa_promote(heap,value,chosen);if(promoted<0)return -1;
  child=soa_global_ref(heap,promoted);if(child<1)return -1;
 }
 soa_store_i32(s,SOA_COL_WORD[col]+index,child);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_REF);return 0;
}
__device__ inline int soa_dict_set(Heap* heap,int dict_ref,int key_ref,int val_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,dict_ref))return -2;
 if(dict_ref<0)return soa_fail(heap,ERROR_CONSTANT_WRITE);
 int gid=heap->nodes[dict_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 int kind=SOA_POOL_KIND[pool];
 if(kind==SOA_KIND_LIST||kind==SOA_KIND_TUPLE)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 soa_refresh_header(heap,dict_ref);int count=heap->nodes[dict_ref*4+2];
 if(kind==SOA_KIND_RECORD){
  int field=-1;
  for(int i=0;i<SOA_POOL_NFIELDS[pool];i++)if(soa_text_equals_name(heap,key_ref,SOA_POOL_FIELD_START[pool]+i)){field=i;break;}
  if(field<0){
   for(int i=0;i<SOA_POOL_NFIELDS[pool];i++){
    int name=soa_name_ref(heap,SOA_POOL_FIELD_START[pool]+i);if(name<0)return -1;
    int eq=key_equal(heap,name,key_ref);if(heap->error)return -1;if(eq){field=i;break;}
   }
  }
  if(field<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  int col=SOA_POOL_FIELD_START[pool]+field;
  int present=soa_load_u8(s,SOA_COL_TAG[col]+slot)!=SOA_ABSENT;
  if(soa_write_cell(heap,col,slot,val_ref)<0)return -1;
  if(!present){
   if(count>=SOA_POOL_MAX_ITEMS[pool])return soa_fail(heap,ERROR_EDGE_CAPACITY);
   soa_store_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,count),field);
   count++;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[dict_ref*4+2]=count;
  }
  return 0;
 }
 int idx=soa_dict_find(heap,dict_ref,key_ref);if(heap->error)return -1;
 if(idx<0){
  if(count>=SOA_POOL_MAX_ITEMS[pool])return soa_fail(heap,ERROR_EDGE_CAPACITY);
  idx=count;count++;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[dict_ref*4+2]=count;
 }
 int index=soa_item_index(pool,slot,idx);
 if(soa_write_cell(heap,SOA_POOL_KEY_COL[pool],index,key_ref)<0)return -1;
 if(soa_write_cell(heap,SOA_POOL_VALUE_START[pool],index,val_ref)<0)return -1;
 return 0;
}
__device__ inline int soa_dict_delete(Heap* heap,int dict_ref,int key_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,dict_ref))return -2;
 if(dict_ref<0)return soa_fail(heap,ERROR_CONSTANT_WRITE);
 int gid=heap->nodes[dict_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 int kind=SOA_POOL_KIND[pool];
 if(kind==SOA_KIND_LIST||kind==SOA_KIND_TUPLE)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 int idx=soa_dict_find(heap,dict_ref,key_ref);if(heap->error)return -1;if(idx<0)return soa_fail(heap,ERROR_KEY_NOT_FOUND);
 soa_refresh_header(heap,dict_ref);int count=heap->nodes[dict_ref*4+2];
 if(kind==SOA_KIND_RECORD){
  int order=soa_load_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,idx));
  int col=SOA_POOL_FIELD_START[pool]+order;
  soa_store_u8(s,SOA_COL_TAG[col]+slot,SOA_ABSENT);soa_clear_cell_cache(s,col,slot);
  for(int i=idx;i<count-1;i++)soa_store_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,i),soa_load_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,i+1)));
  count--;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[dict_ref*4+2]=count;return 0;
 }
 for(int i=idx;i<count-1;i++){
  int dst=soa_item_index(pool,slot,i),src=soa_item_index(pool,slot,i+1);
  int kcol=SOA_POOL_KEY_COL[pool],vcol=SOA_POOL_VALUE_START[pool];
  soa_store_u8(s,SOA_COL_TAG[kcol]+dst,soa_load_u8(s,SOA_COL_TAG[kcol]+src));
  soa_store_u8(s,SOA_COL_TAG[vcol]+dst,soa_load_u8(s,SOA_COL_TAG[vcol]+src));
  if(SOA_COL_INTEGER[kcol]>=0)soa_store_i64(s,SOA_COL_INTEGER[kcol]+dst,soa_load_i64(s,SOA_COL_INTEGER[kcol]+src));
  if(SOA_COL_INTEGER[vcol]>=0)soa_store_i64(s,SOA_COL_INTEGER[vcol]+dst,soa_load_i64(s,SOA_COL_INTEGER[vcol]+src));
  if(SOA_COL_REAL[kcol]>=0)soa_store_f64(s,SOA_COL_REAL[kcol]+dst,soa_load_f64(s,SOA_COL_REAL[kcol]+src));
  if(SOA_COL_REAL[vcol]>=0)soa_store_f64(s,SOA_COL_REAL[vcol]+dst,soa_load_f64(s,SOA_COL_REAL[vcol]+src));
  if(SOA_COL_WORD[kcol]>=0)soa_store_i32(s,SOA_COL_WORD[kcol]+dst,soa_load_i32(s,SOA_COL_WORD[kcol]+src));
  if(SOA_COL_WORD[vcol]>=0)soa_store_i32(s,SOA_COL_WORD[vcol]+dst,soa_load_i32(s,SOA_COL_WORD[vcol]+src));
  soa_clear_cell_cache(s,kcol,dst);soa_clear_cell_cache(s,vcol,dst);
 }
 int last=soa_item_index(pool,slot,count-1);
 soa_store_u8(s,SOA_COL_TAG[SOA_POOL_KEY_COL[pool]]+last,SOA_ABSENT);
 soa_store_u8(s,SOA_COL_TAG[SOA_POOL_VALUE_START[pool]]+last,SOA_ABSENT);
 count--;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[dict_ref*4+2]=count;return 0;
}
__device__ inline int soa_list_get(Heap* heap,int list_ref,int index){
 if(!soa_is_view(heap,list_ref))return -2;
 NodeView n;if(!get_node(heap,list_ref,&n))return soa_fail(heap,ERROR_INVALID_REF);
 if(n.tag!=TAG_LIST&&n.tag!=TAG_TUPLE)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 if(index<0)index=n.count+index;if(index<0||index>=n.count)return soa_fail(heap,ERROR_INDEX_OUT_OF_BOUNDS);
 int k=0,v=0;if(!soa_get_edge(heap,&n,index,&k,&v))return -1;return k;
}
__device__ inline int soa_list_set(Heap* heap,int list_ref,int index,int val_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,list_ref))return -2;
 if(list_ref<0)return soa_fail(heap,ERROR_CONSTANT_WRITE);
 int gid=heap->nodes[list_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 if(SOA_POOL_KIND[pool]!=SOA_KIND_LIST)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 soa_refresh_header(heap,list_ref);int count=heap->nodes[list_ref*4+2];
 if(index<0)index=count+index;if(index<0||index>=count)return soa_fail(heap,ERROR_INDEX_OUT_OF_BOUNDS);
 return soa_write_cell(heap,SOA_POOL_VALUE_START[pool],soa_item_index(pool,slot,index),val_ref);
}
__device__ inline int soa_list_append(Heap* heap,int list_ref,int val_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,list_ref))return -2;
 if(list_ref<0)return soa_fail(heap,ERROR_CONSTANT_WRITE);
 int gid=heap->nodes[list_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 if(SOA_POOL_KIND[pool]!=SOA_KIND_LIST)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 soa_refresh_header(heap,list_ref);int count=heap->nodes[list_ref*4+2];
 if(count>=SOA_POOL_MAX_ITEMS[pool])return soa_fail(heap,ERROR_EDGE_CAPACITY);
 if(soa_write_cell(heap,SOA_POOL_VALUE_START[pool],soa_item_index(pool,slot,count),val_ref)<0)return -1;
 count++;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[list_ref*4+2]=count;return 0;
}
__device__ inline int soa_list_pop(Heap* heap,int list_ref,int* out_val_ref){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,list_ref))return -2;
 if(list_ref<0)return soa_fail(heap,ERROR_CONSTANT_WRITE);
 int gid=heap->nodes[list_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 if(SOA_POOL_KIND[pool]!=SOA_KIND_LIST)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 soa_refresh_header(heap,list_ref);int count=heap->nodes[list_ref*4+2];
 if(count<=0)return soa_fail(heap,ERROR_INDEX_OUT_OF_BOUNDS);
 int last=soa_item_index(pool,slot,count-1);
 int val=soa_box_cell(heap,SOA_POOL_VALUE_START[pool],last);if(val<0)return -1;
 *out_val_ref=val;
 soa_store_u8(s,SOA_COL_TAG[SOA_POOL_VALUE_START[pool]]+last,SOA_ABSENT);
 soa_clear_cell_cache(s,SOA_POOL_VALUE_START[pool],last);
 count--;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[list_ref*4+2]=count;return 0;
}
__device__ inline int soa_copy_cell(DomainSoAContext* s,int col,int src,int dst){
 unsigned char kind=soa_load_u8(s,SOA_COL_TAG[col]+src);soa_store_u8(s,SOA_COL_TAG[col]+dst,kind);
 if(SOA_COL_INTEGER[col]>=0)soa_store_i64(s,SOA_COL_INTEGER[col]+dst,soa_load_i64(s,SOA_COL_INTEGER[col]+src));
 if(SOA_COL_REAL[col]>=0)soa_store_f64(s,SOA_COL_REAL[col]+dst,soa_load_f64(s,SOA_COL_REAL[col]+src));
 if(SOA_COL_WORD[col]>=0)soa_store_i32(s,SOA_COL_WORD[col]+dst,soa_load_i32(s,SOA_COL_WORD[col]+src));
 soa_clear_cell_cache(s,col,dst);return kind;
}
__device__ inline int soa_clone_gid(Heap* heap,int global_ref);
__device__ inline int soa_clone_cell_refs(Heap* heap,int col,int index){
 DomainSoAContext* s=soa_ctx(heap);
 if(soa_load_u8(s,SOA_COL_TAG[col]+index)!=SOA_REF)return 0;
 int child=soa_load_i32(s,SOA_COL_WORD[col]+index);
 int cloned=soa_clone_gid(heap,child);if(cloned<1)return -1;
 soa_store_i32(s,SOA_COL_WORD[col]+index,cloned);return 0;
}
__device__ inline int soa_tuple_is_pure(Heap* heap,int global,int depth){
 if(depth>128)return 0;
 int pool=0,slot=0;if(!soa_locate(global,&pool,&slot))return 0;
 DomainSoAContext* s=heap->soa;
 if(soa_load_u8(s,SOA_POOL_TAG[pool]+slot)!=TAG_TUPLE)return 0;
 int count=soa_load_i32(s,SOA_POOL_COUNT_COL[pool]+slot),kind=SOA_POOL_KIND[pool];
 if(count<0||count>SOA_POOL_MAX_ITEMS[pool])return 0;
 for(int i=0;i<count;i++){
  int col=SOA_POOL_VALUE_START[pool]+(kind==SOA_KIND_TUPLE?i:0);
  int index=kind==SOA_KIND_TUPLE?slot:soa_item_index(pool,slot,i);
  int tag=soa_load_u8(s,SOA_COL_TAG[col]+index);
  if(tag==SOA_ABSENT)return 0;
  if(tag==SOA_REF&&!soa_tuple_is_pure(heap,soa_load_i32(s,SOA_COL_WORD[col]+index),depth+1))return 0;
 }
 return 1;
}
__device__ inline int soa_clone_gid(Heap* heap,int global_ref){
 DomainSoAContext* s=soa_ctx(heap);
 int pool=0,slot=0;if(!soa_locate(global_ref,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 int gid=global_ref-1;int mapped=s->clone_map[(long long)gid*s->rows+s->row];
 if(mapped>0)return mapped;
 if(soa_tuple_is_pure(heap,global_ref,0)){
  s->clone_map[(long long)gid*s->rows+s->row]=global_ref;return global_ref;
 }
 int used=soa_load_i32(s,SOA_POOL_USED[pool]);
 if(used>=SOA_POOL_CAP[pool])return soa_fail(heap,ERROR_NODE_CAPACITY);
 int dst=used;soa_store_i32(s,SOA_POOL_USED[pool],used+1);
 int neu=SOA_POOL_BASE[pool]+dst+1;
 s->clone_map[(long long)gid*s->rows+s->row]=neu;
 soa_store_u8(s,SOA_POOL_TAG[pool]+dst,soa_load_u8(s,SOA_POOL_TAG[pool]+slot));
 int count=soa_load_i32(s,SOA_POOL_COUNT_COL[pool]+slot);
 soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+dst,count);
 int kind=SOA_POOL_KIND[pool];
 if(kind==SOA_KIND_RECORD){
  for(int i=0;i<count;i++)soa_store_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,dst,i),soa_load_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,i)));
  for(int f=0;f<SOA_POOL_NFIELDS[pool];f++){
   int col=SOA_POOL_FIELD_START[pool]+f;soa_copy_cell(s,col,slot,dst);
   if(soa_clone_cell_refs(heap,col,dst)<0)return -1;
  }
 }else if(kind==SOA_KIND_TUPLE){
  for(int i=0;i<SOA_POOL_VALUE_COUNT[pool];i++){
   int col=SOA_POOL_VALUE_START[pool]+i;soa_copy_cell(s,col,slot,dst);
   if(soa_clone_cell_refs(heap,col,dst)<0)return -1;
  }
 }else{
  if(kind==SOA_KIND_MAP){
   int kcol=SOA_POOL_KEY_COL[pool];
   for(int i=0;i<count;i++)soa_copy_cell(s,kcol,soa_item_index(pool,slot,i),soa_item_index(pool,dst,i));
  }
  int vcol=SOA_POOL_VALUE_START[pool];
  for(int i=0;i<count;i++){
   int di=soa_item_index(pool,dst,i);soa_copy_cell(s,vcol,soa_item_index(pool,slot,i),di);
   if(soa_clone_cell_refs(heap,vcol,di)<0)return -1;
  }
 }
 return neu;
}
__device__ inline int soa_clone(Heap* heap,int vm_ref,int reset_memo){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_ready(heap)||!soa_is_view(heap,vm_ref))return soa_fail(heap,ERROR_INVALID_REF);
 if(reset_memo){for(int i=0;i<SOA_TOTAL_OBJECTS;i++)s->clone_map[(long long)i*s->rows+s->row]=0;}
 int gid=heap->nodes[vm_ref*4+3];
 int cloned=soa_clone_gid(heap,gid);if(cloned<1)return -1;
 return soa_view_ref(heap,cloned);
}
__device__ inline int soa_memo_find(const int* olds,const int* news,int n,int ref){for(int i=0;i<n;i++)if(olds[i]==ref)return news[i];return 0;}
__device__ inline int soa_scratch_fits(Heap* heap,const NodeView* n,int ref,int pool);
__device__ inline int soa_encode(Heap* heap,int vm_ref,int type_id,int* memo_old,int* memo_new,int* memo_n,int* active,int* active_n);
__device__ inline int soa_write_encoded(Heap* heap,int col,int index,int value,int* memo_old,int* memo_new,int* memo_n,int* active,int* active_n){
 NodeView n;if(!get_node(heap,value,&n))return soa_fail(heap,ERROR_INVALID_REF);
 if(n.tag<=TAG_STR)return soa_write_cell(heap,col,index,value);
 if(!SOA_COL_NREFS[col]||SOA_COL_WORD[col]<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 int chosen=-1;int start=SOA_COL_REF_START[col];
 if(soa_header_view(heap,value)){
  int gid=heap->nodes[value*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
  if(!soa_col_accepts_ref(col,pool))return soa_fail(heap,ERROR_TYPE_MISMATCH);
  DomainSoAContext* s=soa_ctx(heap);soa_clear_cell_cache(s,col,index);
  soa_store_i32(s,SOA_COL_WORD[col]+index,gid);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_REF);return 0;
 }
 for(int i=0;i<SOA_COL_NREFS[col];i++){
  int tid=SOA_COL_REFS[start+i];
  if(soa_scratch_fits(heap,&n,value,tid)&&(chosen<0||SOA_POOL_NFIELDS[tid]<SOA_POOL_NFIELDS[chosen]))chosen=tid;
 }
 if(chosen<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 int gid=soa_encode(heap,value,chosen,memo_old,memo_new,memo_n,active,active_n);if(gid<1)return -1;
 DomainSoAContext* s=soa_ctx(heap);soa_clear_cell_cache(s,col,index);
 soa_store_i32(s,SOA_COL_WORD[col]+index,gid);soa_store_u8(s,SOA_COL_TAG[col]+index,SOA_REF);return 0;
}
__device__ inline int soa_scratch_fits(Heap* heap,const NodeView* n,int ref,int pool){
 if(!soa_pool_allows_tag(pool,n->tag))return 0;
 int kind=SOA_POOL_KIND[pool];
 if(kind==SOA_KIND_TUPLE)return n->count==SOA_POOL_MAX_ITEMS[pool];
 if(kind==SOA_KIND_LIST||kind==SOA_KIND_MAP)return n->count>=0&&n->count<=SOA_POOL_MAX_ITEMS[pool];
 if(kind!=SOA_KIND_RECORD)return 0;
 for(int i=0;i<n->count;i++){
  int k=0,v=0;if(!get_edge(heap,n,i,&k,&v))return 0;
  int matched=0;
  for(int f=0;f<SOA_POOL_NFIELDS[pool];f++)if(soa_text_equals_name((Heap*)heap,k,SOA_POOL_FIELD_START[pool]+f)){matched=1;break;}
  if(!matched)return 0;
 }
 return 1;
}
__device__ inline int soa_choose_pool(Heap* heap,const NodeView* n,int ref,int type_id){
 if(type_id>=0)return soa_scratch_fits(heap,n,ref,type_id)?type_id:-1;
 int chosen=-1;
 for(int i=0;i<SOA_POOL_COUNT;i++){
  if(!soa_scratch_fits(heap,n,ref,i))continue;
  if(chosen<0||SOA_POOL_NFIELDS[i]<SOA_POOL_NFIELDS[chosen])chosen=i;
 }
 return chosen;
}
__device__ inline int soa_encode(Heap* heap,int vm_ref,int type_id,int* memo_old,int* memo_new,int* memo_n,int* active,int* active_n){
 DomainSoAContext* s=soa_ctx(heap);
 if(soa_header_view(heap,vm_ref)){
  int gid=heap->nodes[vm_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
  if(type_id>=0&&pool!=type_id)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  return gid;
 }
 int seen=soa_memo_find(memo_old,memo_new,*memo_n,vm_ref);if(seen)return seen;
 for(int i=0;i<*active_n;i++)if(active[i]==vm_ref)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 NodeView n;if(!get_node(heap,vm_ref,&n))return soa_fail(heap,ERROR_INVALID_REF);
 if(n.tag<TAG_LIST||n.tag>TAG_CARD_ENTITY)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 int pool=soa_choose_pool(heap,&n,vm_ref,type_id);if(pool<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 int used=soa_load_i32(s,SOA_POOL_USED[pool]);
 if(used>=SOA_POOL_CAP[pool])return soa_fail(heap,ERROR_NODE_CAPACITY);
 int slot=used;soa_store_i32(s,SOA_POOL_USED[pool],used+1);
 int gid=SOA_POOL_BASE[pool]+slot+1;
 soa_store_u8(s,SOA_POOL_TAG[pool]+slot,n.tag);
 soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,n.count);
 if(*memo_n>=512||*active_n>=512)return soa_fail(heap,ERROR_STACK_OVERFLOW);
 memo_old[*memo_n]=vm_ref;memo_new[*memo_n]=gid;(*memo_n)++;
 active[(*active_n)++]=vm_ref;
 int kind=SOA_POOL_KIND[pool];
 if(kind==SOA_KIND_RECORD){
  for(int f=0;f<SOA_POOL_NFIELDS[pool];f++)soa_store_u8(s,SOA_COL_TAG[SOA_POOL_FIELD_START[pool]+f]+slot,SOA_ABSENT);
  for(int i=0;i<n.count;i++){
   int k=0,v=0;if(!get_edge(heap,&n,i,&k,&v))return soa_fail(heap,ERROR_INVALID_REF);
   int field=-1;
   for(int f=0;f<SOA_POOL_NFIELDS[pool];f++)if(soa_text_equals_name(heap,k,SOA_POOL_FIELD_START[pool]+f)){field=f;break;}
   if(field<0)return soa_fail(heap,ERROR_TYPE_MISMATCH);
   soa_store_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,i),field);
   if(soa_write_encoded(heap,SOA_POOL_FIELD_START[pool]+field,slot,v,memo_old,memo_new,memo_n,active,active_n)<0)return -1;
  }
 }else if(kind==SOA_KIND_MAP){
  for(int i=0;i<n.count;i++){
   int k=0,v=0;if(!get_edge(heap,&n,i,&k,&v))return soa_fail(heap,ERROR_INVALID_REF);
   int index=soa_item_index(pool,slot,i);
   if(soa_write_encoded(heap,SOA_POOL_KEY_COL[pool],index,k,memo_old,memo_new,memo_n,active,active_n)<0)return -1;
   if(soa_write_encoded(heap,SOA_POOL_VALUE_START[pool],index,v,memo_old,memo_new,memo_n,active,active_n)<0)return -1;
  }
 }else{
  for(int i=0;i<n.count;i++){
   int k=0,v=0;if(!get_edge(heap,&n,i,&k,&v))return soa_fail(heap,ERROR_INVALID_REF);
   int col=SOA_POOL_VALUE_START[pool]+(kind==SOA_KIND_TUPLE?i:0);
   int index=kind==SOA_KIND_TUPLE?slot:soa_item_index(pool,slot,i);
   if(soa_write_encoded(heap,col,index,k,memo_old,memo_new,memo_n,active,active_n)<0)return -1;
  }
 }
 (*active_n)--;
 return gid;
}
__device__ inline int soa_install_view(Heap* heap,int vm_ref,int gid){
 DomainSoAContext* s=soa_ctx(heap);
 int pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 int tag=soa_load_u8(s,SOA_POOL_TAG[pool]+slot);
 int count=soa_load_i32(s,SOA_POOL_COUNT_COL[pool]+slot);
 int dest=vm_ref;
 if(vm_ref<0){
  int idx=-1-vm_ref;
  if(s->constant_adopt&&idx>=0&&idx<s->constant_node_count){
   int adopted=s->constant_adopt[(long long)idx*s->rows+s->row];
   if(soa_cache_ok(heap,adopted)&&soa_header_view(heap,adopted)&&heap->nodes[adopted*4+3]==gid)return adopted;
  }
  dest=soa_alloc_header(heap,tag,gid,count);if(dest<0)return -1;
  if(s->constant_adopt&&idx>=0&&idx<s->constant_node_count)s->constant_adopt[(long long)idx*s->rows+s->row]=dest;
 }else{
  heap->nodes[dest*4+0]=tag;heap->nodes[dest*4+1]=SOA_VIEW_OFFSET;heap->nodes[dest*4+2]=count;heap->nodes[dest*4+3]=gid;
 }
 soa_store_view(s,gid-1,dest);
 return dest;
}
__device__ inline int soa_promote(Heap* heap,int vm_ref,int type_id){
 if(!soa_ready(heap))return soa_fail(heap,ERROR_INVALID_REF);
 if(soa_header_view(heap,vm_ref)){
  int gid=heap->nodes[vm_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
  if(type_id>=0&&pool!=type_id)return soa_fail(heap,ERROR_TYPE_MISMATCH);
  soa_refresh_header(heap,vm_ref);return vm_ref>=0?vm_ref:soa_view_ref(heap,gid);
 }
 DomainSoAContext* s=soa_ctx(heap);
 int used_save[SOA_POOL_COUNT];
 for(int i=0;i<SOA_POOL_COUNT;i++)used_save[i]=soa_load_i32(s,SOA_POOL_USED[i]);
 int strings=soa_load_i32(s,SOA_STRING_COUNT),bytes=soa_load_i32(s,SOA_BYTES_USED);
 int memo_old[512],memo_new[512],active[512],memo_n=0,active_n=0;
 int gid=soa_encode(heap,vm_ref,type_id,memo_old,memo_new,&memo_n,active,&active_n);
 if(gid<1){
  for(int i=0;i<SOA_POOL_COUNT;i++)soa_store_i32(s,SOA_POOL_USED[i],used_save[i]);
  soa_store_i32(s,SOA_STRING_COUNT,strings);soa_store_i32(s,SOA_BYTES_USED,bytes);
  return -1;
 }
 int root_view=-1;
 for(int i=0;i<memo_n;i++){
  int installed=soa_install_view(heap,memo_old[i],memo_new[i]);if(installed<0)return -1;
  if(memo_old[i]==vm_ref)root_view=installed;
 }
 if(root_view<0)root_view=soa_view_ref(heap,gid);
 return root_view;
}
__device__ inline int soa_scratch_has_text_key(Heap* heap,int dict_ref,const char* key,int n){
 NodeView node;if(!get_node(heap,dict_ref,&node))return 0;
 if(node.tag!=TAG_DICT&&(node.tag<8||node.tag>10))return 0;
 for(int i=0;i<node.count;i++){
  int k=0,v=0;if(!get_edge(heap,&node,i,&k,&v))return 0;
  NodeView kn;if(!get_node(heap,k,&kn)||kn.tag!=TAG_STR||kn.count!=n)continue;
  const unsigned char* p=get_payload_ptr(heap,&kn);int ok=1;
  for(int j=0;j<n;j++)if(p[j]!=(unsigned char)key[j]){ok=0;break;}
  if(ok)return v;
 }
 return 0;
}
__device__ inline int soa_looks_like_game(Heap* heap,int ref){
 if(!SOA_HAS_GAME_KEYS||!soa_ready(heap)||soa_header_view(heap,ref))return 0;
 NodeView n;if(!get_node(heap,ref,&n))return 0;
 if(n.tag!=TAG_DICT&&(n.tag<8||n.tag>10))return 0;
 int rng=soa_scratch_has_text_key(heap,ref,"rng",3);
 int sides=soa_scratch_has_text_key(heap,ref,"sides",5);
 int nxt=soa_scratch_has_text_key(heap,ref,"next_entity",11);
 if(!rng||!sides||!nxt)return 0;
 NodeView rn;if(!get_node(heap,rng,&rn)||rn.tag!=TAG_INT)return 0;
 long long value=0;if(read_int64(heap,rng,&value)<0){heap->error=ERROR_OK;return 0;}
 if(value<0)return 0;
 return 1;
}
__device__ inline int soa_maybe_promote_game(Heap* heap,int ref){
 if(!soa_looks_like_game(heap,ref))return ref;
 return soa_promote(heap,ref,SOA_ROOT);
}
__device__ inline int soa_promote_args(Heap* heap,int* args,int nargs){
 if(!soa_ready(heap)||!args||nargs<0)return 0;
 for(int i=0;i<nargs;i++)args[i]=soa_maybe_promote_game(heap,args[i]);
 return 0;
}
__device__ inline int soa_list_concat(Heap* heap,int dest,int src){
 if(!soa_is_view(heap,dest))return -2;
 NodeView n;if(!get_node(heap,src,&n))return soa_fail(heap,ERROR_INVALID_REF);
 if(n.tag!=TAG_LIST&&n.tag!=TAG_TUPLE)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 for(int i=0;i<n.count;i++){
  int k=0,v=0;if(!get_edge(heap,&n,i,&k,&v))return soa_fail(heap,ERROR_INVALID_REF);
  if(soa_list_append(heap,dest,k)<0)return -1;
 }
 return dest;
}
__device__ inline int soa_list_repeat(Heap* heap,int dest,long long times){
 if(!soa_is_view(heap,dest))return -2;
 if(times<0)times=0;
 soa_refresh_header(heap,dest);int count=heap->nodes[dest*4+2];
 if(times==1)return dest;
 if(times==0){
  DomainSoAContext* s=soa_ctx(heap);int gid=heap->nodes[dest*4+3],pool=0,slot=0;
  if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
  soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,0);heap->nodes[dest*4+2]=0;return dest;
 }
 int boxed[64];if(count>64)return soa_fail(heap,ERROR_STACK_OVERFLOW);
 for(int i=0;i<count;i++){boxed[i]=soa_list_get(heap,dest,i);if(boxed[i]<0)return -1;}
 DomainSoAContext* s=soa_ctx(heap);int gid=heap->nodes[dest*4+3],pool=0,slot=0;
 if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,0);heap->nodes[dest*4+2]=0;
 for(long long t=0;t<times;t++)for(int i=0;i<count;i++)if(soa_list_append(heap,dest,boxed[i])<0)return -1;
 return dest;
}
__device__ inline int soa_push_pin(int* out,int maxn,int* n,int ref){if(ref<0)return 1;if(*n>=maxn)return 0;out[(*n)++]=ref;return 1;}
__device__ inline int soa_pin_cell(Heap* heap,int col,int index,int* out,int maxn,int* n){
 DomainSoAContext* s=soa_ctx(heap);
 int boxed=soa_load_cache(s,SOA_COL_TAG[col]+index);if(soa_cache_ok(heap,boxed)&&!soa_push_pin(out,maxn,n,boxed))return 0;
 if(soa_load_u8(s,SOA_COL_TAG[col]+index)==SOA_REF){
  int child=soa_load_i32(s,SOA_COL_WORD[col]+index);int view=soa_load_view(s,child-1);
  if(soa_cache_ok(heap,view)&&!soa_push_pin(out,maxn,n,view))return 0;
 }
 return 1;
}
__device__ inline int soa_collect_pins(Heap* heap,int ref,int* out,int maxn){
 if(!soa_is_view(heap,ref)||!out||maxn<1)return 0;
 DomainSoAContext* s=soa_ctx(heap);
 int gid=heap->nodes[ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return 0;
 soa_refresh_header(heap,ref);int count=heap->nodes[ref*4+2];int kind=SOA_POOL_KIND[pool];int n=0;
 if(!soa_push_pin(out,maxn,&n,ref))return 0;
 if(kind==SOA_KIND_RECORD){
  for(int f=0;f<SOA_POOL_NFIELDS[pool];f++){
   int col=SOA_POOL_FIELD_START[pool]+f;
   int name=soa_load_cache(s,SOA_STRING_BYTES+SOA_MAX_STRINGS+col);
   if(soa_cache_ok(heap,name)&&!soa_push_pin(out,maxn,&n,name))return 0;
   if(!soa_pin_cell(heap,col,slot,out,maxn,&n))return 0;
  }
 }else if(kind==SOA_KIND_TUPLE){
  for(int i=0;i<SOA_POOL_VALUE_COUNT[pool];i++)if(!soa_pin_cell(heap,SOA_POOL_VALUE_START[pool]+i,slot,out,maxn,&n))return 0;
 }else{
  if(kind==SOA_KIND_MAP)for(int i=0;i<count;i++)if(!soa_pin_cell(heap,SOA_POOL_KEY_COL[pool],soa_item_index(pool,slot,i),out,maxn,&n))return 0;
  for(int i=0;i<count;i++)if(!soa_pin_cell(heap,SOA_POOL_VALUE_START[pool],soa_item_index(pool,slot,i),out,maxn,&n))return 0;
 }
 return n;
}
__device__ inline int soa_list_delete(Heap* heap,int list_ref,int index){
 DomainSoAContext* s=soa_ctx(heap);
 if(!soa_is_view(heap,list_ref))return -2;
 int gid=heap->nodes[list_ref*4+3],pool=0,slot=0;if(!soa_locate(gid,&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 if(SOA_POOL_KIND[pool]!=SOA_KIND_LIST)return soa_fail(heap,ERROR_TYPE_MISMATCH);
 soa_refresh_header(heap,list_ref);int count=heap->nodes[list_ref*4+2];
 if(index<0)index=count+index;if(index<0||index>=count)return soa_fail(heap,ERROR_INDEX_OUT_OF_BOUNDS);
 int vcol=SOA_POOL_VALUE_START[pool];
 for(int i=index;i<count-1;i++)soa_copy_cell(s,vcol,soa_item_index(pool,slot,i+1),soa_item_index(pool,slot,i));
 soa_store_u8(s,SOA_COL_TAG[vcol]+soa_item_index(pool,slot,count-1),SOA_ABSENT);
 soa_clear_cell_cache(s,vcol,soa_item_index(pool,slot,count-1));
 count--;soa_store_i32(s,SOA_POOL_COUNT_COL[pool]+slot,count);heap->nodes[list_ref*4+2]=count;return 0;
}
'''


def emit_soa_view_source(schema):
    schema = _require_schema(schema)
    source=_emit_schema_tables(schema) + _emit_runtime_head()
    return re.sub(r'(?<!__device__ )\b(const (?:int|unsigned char) SOA_\w+\[\])',r'__device__ \1',source)


def _once(source, old, new, name):
    count = source.count(old)
    if count < 1:
        raise SoAViewError('Missing ' + name)
    return source.replace(old, new)


def _replace_all(source, old, new, name):
    if old not in source:
        raise SoAViewError('Missing ' + name)
    return source.replace(old, new)


def _insert_before(source, marker, text, name):
    if marker not in source:
        raise SoAViewError('Missing ' + name)
    return source.replace(marker, text + marker, 1)


def integrate_soa_source(source, schema, image=None):
    """Patch emit_native_array_source output with a native SoA view bridge."""
    if type(source) is not str or not source.strip():
        raise TypeError('Generated native-array C++ source required')
    schema = _require_schema(schema)
    if image is not None and getattr(image, 'nodes', None) is None:
        raise SoAViewError('VM image with constant nodes required')
    helpers = emit_soa_view_source(schema)
    heap_marker = 'struct Heap {'
    if heap_marker not in source:
        raise SoAViewError('Heap ABI missing from generated source')
    source = source.replace(heap_marker, 'struct DomainSoAContext;\n' + heap_marker, 1)
    if 'DomainSoAContext* soa' not in source:
        if source.count('    int error;') < 1:
            raise SoAViewError('Heap error field missing')
        source = source.replace('    int error;', '    int error;\n    DomainSoAContext* soa;', 1)
    source = source.replace('heap->error = ERROR_OK;', 'heap->soa = 0;\n    heap->error = ERROR_OK;', 1)
    hook = '// Internal node resolution'
    if hook not in source:
        hook = '__device__ inline int get_node(const Heap* heap, int ref, NodeView* out_node) {'
    source = source.replace(hook, helpers + hook, 1)
    get_node = '__device__ inline int get_node(const Heap* heap, int ref, NodeView* out_node) {\n    if (ref >= 0) {'
    get_node_new = '__device__ inline int get_node(const Heap* heap, int ref, NodeView* out_node) {\n    if (soa_try_get_node(heap, ref, out_node)) return heap->error?0:1;\n    if (ref >= 0) {'
    source = _replace_all(source, get_node, get_node_new, 'get_node')
    get_edge = '__device__ inline int get_edge(const Heap* heap, const NodeView* node, int idx, int* out_k, int* out_v) {\n    if (idx < 0 || idx >= node->count) return 0;'
    get_edge_new = '__device__ inline int get_edge(const Heap* heap, const NodeView* node, int idx, int* out_k, int* out_v) {\n    if (node && node->offset==SOA_VIEW_OFFSET) return soa_get_edge(heap,node,idx,out_k,out_v);\n    if (idx < 0 || idx >= node->count) return 0;'
    source = _replace_all(source, get_edge, get_edge_new, 'get_edge')
    source = _patch_dict_list(source)
    source = _patch_clone_gc_numeric(source)
    source = _patch_context(source)
    source = _patch_execute(source)
    if '__device__ inline int device_list_pop_at(' in source:
        from .vm_soa_methods import integrate_soa_methods
        source=integrate_soa_methods(source)
    return source


def _patch_dict_list(source):
    sets = (
        ('__device__ inline int dict_set(Heap* heap, int dict_ref, int key_ref, int val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (dict_ref < 0) {',
         '__device__ inline int dict_set(Heap* heap, int dict_ref, int key_ref, int val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, dict_ref)) return soa_dict_set(heap, dict_ref, key_ref, val_ref);\n    if (dict_ref < 0) {'),
        ('__device__ inline int dict_delete(Heap* heap, int dict_ref, int key_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (dict_ref < 0) {',
         '__device__ inline int dict_delete(Heap* heap, int dict_ref, int key_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, dict_ref)) return soa_dict_delete(heap, dict_ref, key_ref);\n    if (dict_ref < 0) {'),
        ('__device__ inline int dict_find(Heap* heap, int dict_ref, int key_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    NodeView node;',
         '__device__ inline int dict_find(Heap* heap, int dict_ref, int key_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, dict_ref)) return soa_dict_find(heap, dict_ref, key_ref);\n    NodeView node;'),
        ('__device__ inline int dict_get(Heap* heap, int dict_ref, int key_ref, int default_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    int idx = dict_find(heap, dict_ref, key_ref);',
         '__device__ inline int dict_get(Heap* heap, int dict_ref, int key_ref, int default_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, dict_ref)) return soa_dict_get(heap, dict_ref, key_ref, default_ref);\n    int idx = dict_find(heap, dict_ref, key_ref);'),
        ('__device__ inline int list_set(Heap* heap, int list_ref, int index, int val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (list_ref < 0) {',
         '__device__ inline int list_set(Heap* heap, int list_ref, int index, int val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, list_ref)) return soa_list_set(heap, list_ref, index, val_ref);\n    if (list_ref < 0) {'),
        ('__device__ inline int list_append(Heap* heap, int list_ref, int val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (list_ref < 0) {',
         '__device__ inline int list_append(Heap* heap, int list_ref, int val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, list_ref)) return soa_list_append(heap, list_ref, val_ref);\n    if (list_ref < 0) {'),
        ('__device__ inline int list_pop(Heap* heap, int list_ref, int* out_val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (list_ref < 0) {',
         '__device__ inline int list_pop(Heap* heap, int list_ref, int* out_val_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, list_ref)) return soa_list_pop(heap, list_ref, out_val_ref);\n    if (list_ref < 0) {'),
        ('__device__ inline int list_get(Heap* heap, int list_ref, int index) {\n    if (heap->error != ERROR_OK) return -1;\n    NodeView node;',
         '__device__ inline int list_get(Heap* heap, int list_ref, int index) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, list_ref)) return soa_list_get(heap, list_ref, index);\n    NodeView node;'),
    )
    for old, new in sets:
        source = _replace_all(source, old, new, old.split('(')[0])
    slice_fn = '__device__ inline int collection_store_slice(Context* c,int owner,int key,int val){'
    if slice_fn in source:
        source = source.replace(slice_fn, slice_fn + '\n    if(soa_is_view(&c->h,owner)){c->h.error=ERROR_TYPE_MISMATCH;return 0;}', 1)
    store_fn = '__device__ inline int collection_store(Context* c, int owner, int key, int val) {'
    if store_fn in source:
        source = source.replace(store_fn, store_fn + '\n    if(soa_is_view(&c->h,owner)&&kind(c,key)==T_SLICE){c->h.error=ERROR_TYPE_MISMATCH;return 0;}', 1)
    delete_fn = '__device__ inline int collection_delete(Context* c, int owner, int key) {'
    if delete_fn in source:
        source = source.replace(delete_fn, delete_fn + '\n    if(soa_is_view(&c->h,owner)&&kind(c,owner)==TAG_LIST){long long idx=0;if(read_int64(&c->h,key,&idx)<0)return 0;if(idx<(-2147483647-1LL)||idx>2147483647LL){c->error=BAD_PROGRAM;return 0;}return soa_list_delete(&c->h,owner,(int)idx)==0;}', 1)
    return source


def _patch_clone_gc_numeric(source):
    deep = '__device__ inline int deep_clone(Heap* heap, int root_ref) {\n    if (heap->error != ERROR_OK) return -1;'
    deep_new = '__device__ inline int deep_clone(Heap* heap, int root_ref) {\n    if (heap->error != ERROR_OK) return -1;\n    if (soa_is_view(heap, root_ref)) return soa_clone(heap, root_ref, 1);\n    if (soa_ready(heap)) {DomainSoAContext* s=heap->soa;for(int i=0;i<SOA_TOTAL_OBJECTS;i++)s->clone_map[(long long)i*s->rows+s->row]=0;}'
    if deep in source:
        source = _replace_all(source, deep, deep_new, 'deep_clone')
    discover_heap = '        if (!get_node(heap, curr_ref, &node)) {\n            heap->error = ERROR_INVALID_REF;\n            break;\n        }'
    discover_heap_new = discover_heap + '\n        if (soa_is_view(heap, curr_ref)) {\n            int cloned = soa_clone(heap, curr_ref, 0);\n            if (cloned<0) break;\n            memo[memo_count].old_ref = curr_ref;\n            memo[memo_count].new_ref = cloned;\n            memo_count++;\n            continue;\n        }'
    if discover_heap in source:
        source = source.replace(discover_heap, discover_heap_new, 1)
    phase2 = '        if (old_n.tag > TAG_STR && old_n.tag <= TAG_CARD_ENTITY) {'
    phase2_new = '        if (old_n.offset==SOA_VIEW_OFFSET) continue;\n        if (old_n.tag > TAG_STR && old_n.tag <= TAG_CARD_ENTITY) {'
    if phase2 in source:
        source = source.replace(phase2, phase2_new, 1)
    native = '__device__ inline int native_deepcopy(Context* c,int root){\n if(c->error || c->h.error)return MISSING;'
    native_new = '__device__ inline int native_deepcopy(Context* c,int root){\n if(c->error || c->h.error)return MISSING;\n if(soa_is_view(&c->h,root))return soa_clone(&c->h,root,1);'
    if native in source:
        source = source.replace(native, native_new)
    discover = '__device__ inline int deepcopy_discover(Context* c,int ref,int old_nodes,\n int old_edges,int old_payload,int* tail){\n if(!deepcopy_tick(c))return 0;'
    discover_new = '__device__ inline int deepcopy_discover(Context* c,int ref,int old_nodes,\n int old_edges,int old_payload,int* tail){\n if(!deepcopy_tick(c))return 0;\n if(soa_is_view(&c->h,ref)){\n  int cloned=soa_clone(&c->h,ref,0);if(cloned<0)return 0;\n  int key=deepcopy_key(c,ref,old_nodes);if(key<0)return 0;deepcopy_memo_set(c,key,cloned);return 1;}\n'
    if discover in source:
        source = source.replace(discover, discover_new)
    src_edge = '__device__ inline int deepcopy_source_edge(Context* c,const NodeView* n,int i,int side){\n // source_node validated the complete range against the frozen input counts.\n const int* edges=n->is_constant?c->h.constant_edges:c->h.edges;'
    src_edge_new = '__device__ inline int deepcopy_source_edge(Context* c,const NodeView* n,int i,int side){\n if(n && n->offset==SOA_VIEW_OFFSET){int k=0,v=0;if(!get_edge(&c->h,n,i,&k,&v)){c->h.error=ERROR_INVALID_REF;return MISSING;}return side?v:k;}\n const int* edges=n->is_constant?c->h.constant_edges:c->h.edges;'
    if src_edge in source:
        source = source.replace(src_edge, src_edge_new)
    # deepcopy_source_node bound-check: skip sentinel views
    bound = ' }else{\n  bound=n->is_constant?c->h.constant_edge_count:old_edges;\n  if(n->offset<0 || n->count<0 || (long long)n->offset+n->count>bound){'
    bound_new = ' }else if(n->offset==SOA_VIEW_OFFSET){\n  if(!soa_ready(&c->h)){c->h.error=ERROR_INVALID_REF;return 0;}\n }else{\n  bound=n->is_constant?c->h.constant_edge_count:old_edges;\n  if(n->offset<0 || n->count<0 || (long long)n->offset+n->count>bound){'
    if bound in source:
        source = source.replace(bound, bound_new)
    compact = 'if(n.tag<TAG_NONE || n.tag>TAG_CARD_ENTITY){target->error=ERROR_INVALID_TAG;break;}'
    compact_new = 'if(n.offset==SOA_VIEW_OFFSET){target->error=ERROR_TYPE_MISMATCH;break;}\n        if(n.tag<TAG_NONE || n.tag>TAG_CARD_ENTITY){target->error=ERROR_INVALID_TAG;break;}'
    if compact in source:
        source = source.replace(compact, compact_new)
    seq = '  if(seq!=MISSING){\n   long long repeat=0;if(numeric_int(c,times,&repeat)<0)return MISSING;'
    seq_new = '  if(seq!=MISSING){\n   if(soa_is_view(&c->h,seq)){\n    long long repeat=0;if(op==OP_INPLACE_ADD||op==OP_BINARY_ADD)return soa_list_concat(&c->h,seq,b==seq?a: (seq==a?b:a));\n    if(numeric_int(c,times,&repeat)<0)return MISSING;return soa_list_repeat(&c->h,seq,repeat);\n   }\n   long long repeat=0;if(numeric_int(c,times,&repeat)<0)return MISSING;'
    if seq in source:
        source = source.replace(seq, seq_new, 1)
    add = '  if((ta==TAG_LIST&&tb==TAG_LIST)||(ta==TAG_TUPLE&&tb==TAG_TUPLE)){'
    add_new = '  if((soa_is_view(&c->h,a)||soa_is_view(&c->h,b))&&(op==OP_INPLACE_ADD||op==OP_BINARY_ADD)){\n   if(soa_is_view(&c->h,a)&&ta==TAG_LIST)return soa_list_concat(&c->h,a,b);\n   c->error=UNSUPPORTED_TYPE;return MISSING;\n  }\n  if((ta==TAG_LIST&&tb==TAG_LIST)||(ta==TAG_TUPLE&&tb==TAG_TUPLE)){'
    if add in source:
        source = source.replace(add, add_new, 1)
    return source


def _patch_context(source):
    node = '__device__ inline const int* node(Context* c,int ref){\n if(ref>=0){if(ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}return c->h.nodes+ref*4;}'
    node_new = '__device__ inline const int* node(Context* c,int ref){\n if(ref>=0){if(ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}if(c->h.soa&&c->h.nodes[ref*4+1]==SOA_VIEW_OFFSET)soa_refresh_header(&c->h,ref);return c->h.nodes+ref*4;}'
    if node in source:
        source = source.replace(node, node_new)
    edge = '__device__ inline int edge(Context* c,int container,int index,int value=0){\n const int* n=node(c,container);if(!n || index<0 || index>=n[2]){c->error=BAD_PROGRAM;return MISSING;}'
    edge_new = '__device__ inline int edge(Context* c,int container,int index,int value=0){\n if(soa_is_view(&c->h,container)){NodeView nv;if(!get_node(&c->h,container,&nv)){c->error=BAD_PROGRAM;return MISSING;}int k=0,v=0;if(!get_edge(&c->h,&nv,index,&k,&v)){c->error=BAD_PROGRAM;return MISSING;}return value?v:k;}\n const int* n=node(c,container);if(!n || index<0 || index>=n[2]){c->error=BAD_PROGRAM;return MISSING;}'
    if edge in source:
        source = source.replace(edge, edge_new)
    gc = '  int ref=c->gc_queue[head++];const int* n=node(c,ref);if(!n)break;int tag=n[0];'
    # Every materialized view/scalar cache is rooted below. Domain children
    # remain typed references, not generator/frame objects in scratch edges.
    gc_new = '  int ref=c->gc_queue[head++];if(soa_is_view(&c->h,ref))continue;const int* n=node(c,ref);if(!n)break;int tag=n[0];'
    if gc in source:
        source = source.replace(gc, gc_new)
    pending = ' gc_ref(c,pending_root,&tail);'
    pending_new = ' gc_ref(c,pending_root,&tail);\n if(soa_ready(&c->h)){DomainSoAContext* s=c->h.soa;for(int i=0;i<SOA_TOTAL_OBJECTS;i++)gc_ref(c,s->views[(long long)i*s->rows+s->row],&tail);for(int i=0;i<SOA_VALUE_CACHE_EXTENT;i++)gc_ref(c,s->value_cache[(long long)i*s->rows+s->row],&tail);}'
    if '__device__ inline void collect_suspended(' in source:
        source = _once(source,pending,pending_new,'complete SoA scratch GC roots')
    mutable = '__device__ inline int* mutable_node(Context* c,int ref){\n if(ref<0 || ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}return c->h.nodes+ref*4;\n}'
    mutable_new = '__device__ inline int* mutable_node(Context* c,int ref){\n if(ref<0 || ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}if(soa_is_view(&c->h,ref)){c->h.error=ERROR_TYPE_MISMATCH;return 0;}return c->h.nodes+ref*4;\n}'
    if mutable in source:
        source = source.replace(mutable, mutable_new, 1)
    call = '__device__ inline void call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names=MISSING,const int* kw_values=0,int nkw=0){\n const int* n=node(c,callable);if(!n)return;'
    call_new = '__device__ inline void call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names=MISSING,const int* kw_values=0,int nkw=0){\n int promoted[VM_STACK];if(soa_ready(&c->h)&&args&&nargs>0){int narg=nargs<VM_STACK?nargs:VM_STACK;for(int i=0;i<narg;i++)promoted[i]=soa_maybe_promote_game(&c->h,args[i]);args=promoted;nargs=narg;}\n const int* n=node(c,callable);if(!n)return;'
    if call in source:
        source = source.replace(call, call_new, 1)
    frame = ' if(fn[5]>VM_LOCALS || fn[6]>VM_STACK){c->error=BAD_CALL;return -1;}'
    frame_new = ' if(fn[5]>VM_LOCALS || fn[6]>VM_STACK){c->error=BAD_CALL;return -1;}\n int promoted[VM_STACK];if(soa_ready(&c->h)&&args&&nargs>0){int narg=nargs<VM_STACK?nargs:VM_STACK;for(int i=0;i<narg;i++)promoted[i]=soa_maybe_promote_game(&c->h,args[i]);args=promoted;nargs=narg;}'
    if frame in source:
        source = source.replace(frame, frame_new, 1)
    return source


def _patch_execute(source):
    setup = 'c.h.constant_node_count=constant_node_count;c.h.constant_edge_count=constant_edge_count;c.h.constant_payload_count=constant_payload_count;c.h.error=0;'
    setup_new = setup + 'c.h.soa=0;'
    if setup in source:
        source = source.replace(setup, setup_new)
    entry = ' c.current=new_frame(&c,entry,args+(long long)tid*nargs,nargs,-1,3);'
    entry_new = ' DOMAIN_SOA_ATTACH_POINT;\n c.current=new_frame(&c,entry,args+(long long)tid*nargs,nargs,-1,3);'
    if entry in source:
        source = source.replace(entry, entry_new)
    if 'DOMAIN_SOA_ATTACH_POINT' in source and '#ifndef DOMAIN_SOA_ATTACH_POINT' not in source:
        define = '#ifndef DOMAIN_SOA_ATTACH_POINT\n#define DOMAIN_SOA_ATTACH_POINT ((void)0)\n#endif\n'
        source = source.replace('namespace duel_vm_heap {', define + 'namespace duel_vm_heap {', 1)
        if '#ifndef DOMAIN_SOA_ATTACH_POINT' not in source:
            source = define + source
    return source
