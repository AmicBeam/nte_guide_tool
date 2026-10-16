"""Typed-pool reachability GC for field-major RecordSoA columns.

``compact_rows`` is the CPU array-only reference. ``emit_compact_source``
emits the same algorithm as CUDA C++. ``CudaRecordSoACompact`` is CUDA-only
and lazy; that path has no CPU fallback. Distinct equal objects stay distinct,
aliases are retained, and only TEXT channels (including dynamic map keys) are
string-remapped.
"""
from __future__ import annotations

import math

import numpy as np

from .record_soa import (
    ABSENT, BOOL, FLOAT, INT, MISSING, NULL, REF, TEXT,
    CompiledRecordSchema, RecordSoABatch, RecordSoAError, RecordSoATypeError,
    TorchRecordSoABatch,
)

STATUS_OK, STATUS_BOUNDS, STATUS_TAG, STATUS_COUNT, STATUS_CAPACITY, STATUS_CYCLE, STATUS_UNSUPPORTED = range(7)
_STATUS = ('ok', 'bounds', 'tag', 'count', 'capacity', 'cycle', 'unsupported')
_KIND = {'record': 0, 'list': 1, 'map': 2, 'tuple': 3}

CudaModuleKernelSpec = {
    'record_soa_compact': (
        'int64*', 'float64*', 'int32*', 'uint8*', 'int32*',
        'int64*', 'float64*', 'int32*', 'uint8*', 'int32*',
        'int32*', 'int32*', 'uint8*', 'int32*', 'int32*', 'int32',
    ),
}


def compact_rows(batch):
    """Return a fresh batch with unreachable pool slots and strings dropped."""
    if not isinstance(batch, RecordSoABatch):
        raise RecordSoATypeError('Typed domain record batch required')
    dest = RecordSoABatch(batch.schema, batch.rows)
    for row in range(batch.rows):
        status = _compact_row(batch, dest, row)
        if status != STATUS_OK:
            raise RecordSoAError('Record SoA compact %s' % _STATUS[status])
    return dest


def _locate(schema, ref):
    if type(ref) is not int or ref < 1:
        return None
    for pool in schema.pools:
        if pool.base < ref <= pool.base + pool.capacity:
            return pool, ref - pool.base - 1
    return None


def _child_limit(pool, count):
    return count * 2 if pool.kind == 'map' else count


def _child(pool, slot, cursor, src_i32):
    if pool.kind == 'record':
        field_id = int(src_i32[pool.order + slot * pool.max_items + cursor])
        return pool.fields[field_id], slot
    if pool.kind == 'tuple':
        return pool.values[cursor], slot
    ordinal = cursor // 2 if pool.kind == 'map' else cursor
    index = slot * pool.max_items + ordinal
    if pool.kind == 'map' and cursor % 2 == 0:
        return pool.keys, index
    return pool.values[0], index


def _check_record(pool, slot, count, src_u8, src_i32):
    present = 0
    for field in pool.fields:
        if int(src_u8[field.tag + slot]) != ABSENT:
            present += 1
    if present != count:
        return STATUS_COUNT
    seen = set()
    for ordinal in range(count):
        field_id = int(src_i32[pool.order + slot * pool.max_items + ordinal])
        if field_id < 0 or field_id >= len(pool.fields) or field_id in seen:
            return STATUS_COUNT
        seen.add(field_id)
        if int(src_u8[pool.fields[field_id].tag + slot]) == ABSENT:
            return STATUS_COUNT
    return STATUS_OK


def _mark_string(schema, sid, old_count, old_bytes, src_i32, str_map):
    if sid < 1 or sid > old_count:
        return STATUS_BOUNDS
    start = int(src_i32[schema.string_offsets + sid - 1])
    size = int(src_i32[schema.string_lengths + sid - 1])
    if start < 0 or size < 0 or start + size > old_bytes:
        return STATUS_BOUNDS
    str_map[sid - 1] = 1
    return STATUS_OK


def _follow_value(column, index, schema, src, old_count, old_bytes, str_map):
    kind = int(src['u8'][column.tag + index])
    if kind in (ABSENT, NULL):
        return STATUS_OK, None
    if kind in (BOOL, INT):
        if not column.scalar_mask & (1 << kind) or column.integer < 0:
            return STATUS_UNSUPPORTED, None
        value = int(src['i64'][column.integer + index])
        if kind == BOOL and value not in (0, 1):
            return STATUS_BOUNDS, None
        return STATUS_OK, None
    if kind == FLOAT:
        if not column.scalar_mask & (1 << FLOAT) or column.real < 0:
            return STATUS_UNSUPPORTED, None
        if not math.isfinite(float(src['f64'][column.real + index])):
            return STATUS_BOUNDS, None
        return STATUS_OK, None
    if kind == TEXT:
        if not column.scalar_mask & (1 << TEXT) or column.word < 0:
            return STATUS_UNSUPPORTED, None
        status = _mark_string(schema, int(src['i32'][column.word + index]),
                              old_count, old_bytes, src['i32'], str_map)
        return status, None
    if kind == REF:
        if not column.references or column.word < 0:
            return STATUS_UNSUPPORTED, None
        ref = int(src['i32'][column.word + index])
        located = _locate(schema, ref)
        if located is None:
            return STATUS_BOUNDS, None
        if located[0].index not in column.references:
            return STATUS_TAG, None
        return STATUS_OK, ref
    return STATUS_UNSUPPORTED, None


def _enter(schema, ref, src, color, obj_map, stack, sp):
    located = _locate(schema, ref)
    if located is None:
        return STATUS_BOUNDS, sp
    pool, slot = located
    used = int(src['i32'][pool.used])
    if not 0 <= used <= pool.capacity or slot >= used:
        return STATUS_BOUNDS, sp
    idx = pool.base + slot
    if color[idx] == 1:
        return STATUS_CYCLE, sp
    if color[idx] == 2:
        return STATUS_OK, sp
    tag = int(src['u8'][pool.tag + slot])
    if tag not in pool.tags:
        return STATUS_TAG, sp
    count = int(src['i32'][pool.count + slot])
    if not 0 <= count <= pool.max_items:
        return STATUS_COUNT, sp
    if pool.kind == 'tuple' and count != pool.max_items:
        return STATUS_COUNT, sp
    if pool.kind == 'record':
        status = _check_record(pool, slot, count, src['u8'], src['i32'])
        if status != STATUS_OK:
            return status, sp
    if sp >= schema.total_objects:
        return STATUS_CAPACITY, sp
    color[idx] = 1
    stack[sp] = ref
    obj_map[idx] = 0
    return STATUS_OK, sp + 1


def _discover(schema, src, root, color, obj_map, stack, str_map):
    old_count = int(src['i32'][schema.strings_count])
    old_bytes = int(src['i32'][schema.bytes_used])
    if not 0 <= old_count <= schema.max_strings or not 0 <= old_bytes <= schema.max_string_bytes:
        return STATUS_BOUNDS
    for pool in schema.pools:
        used = int(src['i32'][pool.used])
        if not 0 <= used <= pool.capacity:
            return STATUS_BOUNDS
    status, sp = _enter(schema, root, src, color, obj_map, stack, 0)
    while status == STATUS_OK and sp:
        ref = int(stack[sp - 1])
        pool, slot = _locate(schema, ref)
        idx = pool.base + slot
        count = int(src['i32'][pool.count + slot])
        cursor = int(obj_map[idx])
        limit = _child_limit(pool, count)
        if cursor >= limit:
            color[idx] = 2
            sp -= 1
            continue
        obj_map[idx] = cursor + 1
        column, index = _child(pool, slot, cursor, src['i32'])
        status, child = _follow_value(column, index, schema, src, old_count, old_bytes, str_map)
        if status == STATUS_OK and child is not None:
            status, sp = _enter(schema, child, src, color, obj_map, stack, sp)
    return status


def _assign_maps(schema, src, color, obj_map, dst):
    for pool in schema.pools:
        new_used = 0
        old_used = int(src['i32'][pool.used])
        for slot in range(old_used):
            idx = pool.base + slot
            if color[idx] != 2:
                continue
            obj_map[idx] = pool.base + new_used + 1
            new_used += 1
        dst['i32'][pool.used] = new_used
    return STATUS_OK


def _string_bytes(schema, buffers, index):
    start = int(buffers['i32'][schema.string_offsets + index])
    size = int(buffers['i32'][schema.string_lengths + index])
    return (int(buffers['i64'][schema.string_hashes + index]),
            buffers['u8'][schema.string_bytes + start:schema.string_bytes + start + size])


def _compact_strings(schema, src, dst, str_map):
    old_count = int(src['i32'][schema.strings_count])
    new_count = 0
    bytes_used = 0
    for old in range(old_count):
        if int(str_map[old]) != 1:
            str_map[old] = 0
            continue
        hash_value, raw = _string_bytes(schema, src, old)
        found = -1
        for prior in range(new_count):
            prior_hash, prior_raw = _string_bytes(schema, dst, prior)
            if prior_hash == hash_value and prior_raw.tobytes() == raw.tobytes():
                found = prior
                break
        if found >= 0:
            str_map[old] = found + 1
            continue
        if new_count >= schema.max_strings or bytes_used + len(raw) > schema.max_string_bytes:
            return STATUS_CAPACITY
        dst['i32'][schema.string_offsets + new_count] = bytes_used
        dst['i32'][schema.string_lengths + new_count] = len(raw)
        dst['i64'][schema.string_hashes + new_count] = hash_value
        dst['i32'][schema.string_static_refs + new_count] = src['i32'][schema.string_static_refs + old]
        dst['u8'][schema.string_bytes + bytes_used:schema.string_bytes + bytes_used + len(raw)] = raw
        str_map[old] = new_count + 1
        new_count += 1
        bytes_used += len(raw)
    dst['i32'][schema.strings_count] = new_count
    dst['i32'][schema.bytes_used] = bytes_used
    for rest in range(new_count, schema.max_strings):
        dst['i32'][schema.string_static_refs + rest] = MISSING
    return STATUS_OK


def _copy_value(column, old_index, new_index, src, dst, obj_map, str_map):
    kind = int(src['u8'][column.tag + old_index])
    dst['u8'][column.tag + new_index] = kind
    if kind in (ABSENT, NULL):
        return STATUS_OK
    if kind in (BOOL, INT):
        dst['i64'][column.integer + new_index] = src['i64'][column.integer + old_index]
        return STATUS_OK
    if kind == FLOAT:
        dst['f64'][column.real + new_index] = src['f64'][column.real + old_index]
        return STATUS_OK
    if kind == TEXT:
        sid = int(str_map[int(src['i32'][column.word + old_index]) - 1])
        if sid < 1:
            return STATUS_BOUNDS
        dst['i32'][column.word + new_index] = sid
        return STATUS_OK
    if kind != REF:
        return STATUS_UNSUPPORTED
    located = _locate(src['schema'], int(src['i32'][column.word + old_index]))
    if located is None:
        return STATUS_BOUNDS
    mapped = int(obj_map[located[0].base + located[1]])
    if mapped < 1:
        return STATUS_BOUNDS
    dst['i32'][column.word + new_index] = mapped
    return STATUS_OK


def _copy_object(pool, old_slot, new_slot, src, dst, obj_map, str_map):
    dst['u8'][pool.tag + new_slot] = src['u8'][pool.tag + old_slot]
    count = int(src['i32'][pool.count + old_slot])
    dst['i32'][pool.count + new_slot] = count
    if pool.kind == 'record':
        for ordinal in range(count):
            field_id = int(src['i32'][pool.order + old_slot * pool.max_items + ordinal])
            dst['i32'][pool.order + new_slot * pool.max_items + ordinal] = field_id
            status = _copy_value(pool.fields[field_id], old_slot, new_slot, src, dst, obj_map, str_map)
            if status != STATUS_OK:
                return status
        return STATUS_OK
    for ordinal in range(count):
        if pool.kind == 'tuple':
            status = _copy_value(pool.values[ordinal], old_slot, new_slot, src, dst, obj_map, str_map)
        else:
            old_index = old_slot * pool.max_items + ordinal
            new_index = new_slot * pool.max_items + ordinal
            if pool.kind == 'map':
                status = _copy_value(pool.keys, old_index, new_index, src, dst, obj_map, str_map)
                if status != STATUS_OK:
                    return status
            status = _copy_value(pool.values[0], old_index, new_index, src, dst, obj_map, str_map)
        if status != STATUS_OK:
            return status
    return STATUS_OK


def _copy_live(schema, src, dst, color, obj_map, str_map):
    for pool in schema.pools:
        new_slot = 0
        old_used = int(src['i32'][pool.used])
        for old_slot in range(old_used):
            if color[pool.base + old_slot] != 2:
                continue
            status = _copy_object(pool, old_slot, new_slot, src, dst, obj_map, str_map)
            if status != STATUS_OK:
                return status
            new_slot += 1
    return STATUS_OK


def _compact_row(batch, dest, row):
    schema = batch.schema
    src = {key: batch.buffers[key][:, row] for key in batch.buffers}
    dst = {key: dest.buffers[key][:, row] for key in dest.buffers}
    src['schema'] = schema
    total = schema.total_objects
    color = np.zeros(total, dtype=np.uint8)
    obj_map = np.zeros(total, dtype=np.int32)
    stack = np.zeros(total, dtype=np.int32)
    str_map = np.zeros(schema.max_strings, dtype=np.int32)
    status = _discover(schema, src, int(batch.roots[row]), color, obj_map, stack, str_map)
    if status != STATUS_OK:
        return status
    status = _assign_maps(schema, src, color, obj_map, dst)
    if status != STATUS_OK:
        return status
    status = _compact_strings(schema, src, dst, str_map)
    if status != STATUS_OK:
        return status
    status = _copy_live(schema, src, dst, color, obj_map, str_map)
    if status != STATUS_OK:
        return status
    located = _locate(schema, int(batch.roots[row]))
    if located is None:
        return STATUS_BOUNDS
    dest.roots[row] = obj_map[located[0].base + located[1]]
    if int(dest.roots[row]) < 1:
        return STATUS_BOUNDS
    return STATUS_OK


def _emit_tables(schema):
    columns, refs = [], []
    pools = []
    for pool in schema.pools:
        field_start = len(columns)
        columns.extend(pool.fields)
        keys = -1
        if pool.keys is not None:
            keys = len(columns)
            columns.append(pool.keys)
        value_start = len(columns)
        columns.extend(pool.values)
        tags = list(pool.tags) + [0, 0, 0, 0]
        pools.append(
            '{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}' % (
                _KIND[pool.kind], pool.capacity, pool.base, pool.used, pool.tag,
                pool.count, pool.order, pool.max_items, len(pool.tags),
                tags[0], tags[1], tags[2], tags[3], len(pool.fields),
                field_start, keys, len(pool.values), value_start))
    col_rows = []
    for column in columns:
        start = len(refs)
        refs.extend(column.references)
        col_rows.append('{%d,%d,%d,%d,%d,%d,%d,%d}' % (
            column.tag, column.integer, column.real, column.word, column.stride,
            column.scalar_mask, start, len(column.references)))
    if not col_rows:
        col_rows.append('{0,-1,-1,-1,1,0,0,0}')
    if not refs:
        refs.append(0)
    return '\n'.join((
        'static const int kNPools=%d;' % len(schema.pools),
        'static const int kTotalObjects=%d;' % schema.total_objects,
        'static const int kMaxStrings=%d;' % schema.max_strings,
        'static const int kMaxStringBytes=%d;' % schema.max_string_bytes,
        'static const int kStringsCount=%d;' % schema.strings_count,
        'static const int kBytesUsed=%d;' % schema.bytes_used,
        'static const int kStringOffsets=%d;' % schema.string_offsets,
        'static const int kStringLengths=%d;' % schema.string_lengths,
        'static const int kStringStaticRefs=%d;' % schema.string_static_refs,
        'static const int kStringHashes=%d;' % schema.string_hashes,
        'static const int kStringBytes=%d;' % schema.string_bytes,
        'static const int kMissing=%d;' % MISSING,
        'static const PoolRec kPools[]={%s};' % ','.join(pools),
        'static const ColRec kCols[]={%s};' % ','.join(col_rows),
        'static const int kColRefs[]={%s};' % ','.join(str(int(v)) for v in refs),
    ))


_COMPACT_PREFIX = r'''
enum { kAbsent=0,kNull=1,kBool=2,kInt=3,kFloat=4,kText=5,kRef=6 };
enum { kOk=0,kBounds=1,kTag=2,kCount=3,kCapacity=4,kCycle=5,kUnsupported=6 };
enum { kRecord=0,kList=1,kMap=2,kTuple=3 };
struct PoolRec {
  int kind,capacity,base,used,tag,count,order,max_items;
  int n_tags,tags0,tags1,tags2,tags3;
  int n_fields,field_start,keys,n_values,value_start;
};
struct ColRec {
  int tag,integer,real,word,stride,scalar_mask,ref_off,nrefs;
};
#define RSA(P,C,R) ((P)[((long long)(C)*(long long)(rows))+(long long)(R)])
__device__ inline int tag_ok(const PoolRec& p,int tag){
  return (p.n_tags>0&&p.tags0==tag)||(p.n_tags>1&&p.tags1==tag)||
         (p.n_tags>2&&p.tags2==tag)||(p.n_tags>3&&p.tags3==tag);
}
__device__ inline int locate_pool(int ref,int* slot){
  if(ref<1)return -1;
  for(int p=0;p<kNPools;++p){
    if(ref>kPools[p].base&&ref<=kPools[p].base+kPools[p].capacity){
      *slot=ref-kPools[p].base-1;return p;
    }
  }
  return -1;
}
__device__ inline int finite_d(double x){
  union{double d;unsigned long long u;}v;v.d=x;
  return ((v.u>>52)&0x7ffULL)!=0x7ffULL;
}
__device__ inline int child_limit(const PoolRec& p,int count){
  return p.kind==kMap?count*2:count;
}
__device__ inline int child_col(const PoolRec& p,int slot,int cursor,int rows,int row,
                               const int* src_i32,int* index,int* status){
  if(p.kind==kRecord){
    int fid=RSA(src_i32,p.order+slot*p.max_items+cursor,row);
    if(fid<0||fid>=p.n_fields){*status=kCount;return -1;}
    *index=slot;return p.field_start+fid;
  }
  if(p.kind==kTuple){*index=slot;return p.value_start+cursor;}
  int ordinal=p.kind==kMap?cursor/2:cursor;
  *index=slot*p.max_items+ordinal;
  if(p.kind==kMap&&(cursor%2)==0)return p.keys;
  return p.value_start;
}
__device__ inline int ref_allowed(const ColRec& c,int pool_index){
  for(int i=0;i<c.nrefs;++i)if(kColRefs[c.ref_off+i]==pool_index)return 1;
  return 0;
}
'''

_COMPACT_DISCOVER = r'''
__device__ inline int check_record(const PoolRec& p,int slot,int count,int rows,int row,
                                   const unsigned char* src_u8,const int* src_i32){
  int present=0;
  for(int f=0;f<p.n_fields;++f){
    if(RSA(src_u8,kCols[p.field_start+f].tag+slot,row)!=kAbsent)present++;
  }
  if(present!=count)return kCount;
  for(int o=0;o<count;++o){
    int fid=RSA(src_i32,p.order+slot*p.max_items+o,row);
    if(fid<0||fid>=p.n_fields)return kCount;
    for(int j=0;j<o;++j){
      if(RSA(src_i32,p.order+slot*p.max_items+j,row)==fid)return kCount;
    }
    if(RSA(src_u8,kCols[p.field_start+fid].tag+slot,row)==kAbsent)return kCount;
  }
  return kOk;
}
__device__ inline int mark_string(int sid,int old_count,int old_bytes,int rows,int row,
                                  const int* src_i32,int* str_map){
  if(sid<1||sid>old_count)return kBounds;
  int start=RSA(src_i32,kStringOffsets+sid-1,row);
  int size=RSA(src_i32,kStringLengths+sid-1,row);
  if(start<0||size<0||start+size>old_bytes)return kBounds;
  RSA(str_map,sid-1,row)=1;
  return kOk;
}
__device__ inline int follow_value(const ColRec& c,int index,int rows,int row,
                                  const unsigned char* src_u8,const long long* src_i64,
                                  const double* src_f64,const int* src_i32,
                                  int old_count,int old_bytes,int* str_map,int* child){
  int kind=RSA(src_u8,c.tag+index,row);*child=-1;
  if(kind==kAbsent||kind==kNull)return kOk;
  if(kind==kBool||kind==kInt){
    if(!(c.scalar_mask&(1<<kind))||c.integer<0)return kUnsupported;
    long long value=RSA(src_i64,c.integer+index,row);
    if(kind==kBool&&value!=0&&value!=1)return kBounds;
    return kOk;
  }
  if(kind==kFloat){
    if(!(c.scalar_mask&(1<<kFloat))||c.real<0)return kUnsupported;
    return finite_d(RSA(src_f64,c.real+index,row))?kOk:kBounds;
  }
  if(kind==kText){
    if(!(c.scalar_mask&(1<<kText))||c.word<0)return kUnsupported;
    return mark_string(RSA(src_i32,c.word+index,row),old_count,old_bytes,rows,row,src_i32,str_map);
  }
  if(kind!=kRef)return kUnsupported;
  if(c.nrefs==0||c.word<0)return kUnsupported;
  int ref=RSA(src_i32,c.word+index,row),slot=0,pool=locate_pool(ref,&slot);
  if(pool<0)return kBounds;
  if(!ref_allowed(c,pool))return kTag;
  *child=ref;return kOk;
}
'''

_COMPACT_WALK = r'''
__device__ inline int enter_ref(int ref,int rows,int row,
                                const unsigned char* src_u8,const int* src_i32,
                                unsigned char* color,int* obj_map,int* stack,int sp,int* status){
  int slot=0,p=locate_pool(ref,&slot);
  if(p<0){*status=kBounds;return sp;}
  const PoolRec& pool=kPools[p];
  int used=RSA(src_i32,pool.used,row);
  if(used<0||used>pool.capacity||slot>=used){*status=kBounds;return sp;}
  int idx=pool.base+slot;
  unsigned char seen=RSA(color,idx,row);
  if(seen==1){*status=kCycle;return sp;}
  if(seen==2)return sp;
  int tag=RSA(src_u8,pool.tag+slot,row);
  if(!tag_ok(pool,tag)){*status=kTag;return sp;}
  int count=RSA(src_i32,pool.count+slot,row);
  if(count<0||count>pool.max_items){*status=kCount;return sp;}
  if(pool.kind==kTuple&&count!=pool.max_items){*status=kCount;return sp;}
  if(pool.kind==kRecord){
    int rec=check_record(pool,slot,count,rows,row,src_u8,src_i32);
    if(rec!=kOk){*status=rec;return sp;}
  }
  if(sp>=kTotalObjects){*status=kCapacity;return sp;}
  RSA(color,idx,row)=1;RSA(stack,sp,row)=ref;RSA(obj_map,idx,row)=0;
  return sp+1;
}
__device__ inline int discover(int root,int rows,int row,
                               const unsigned char* src_u8,const long long* src_i64,
                               const double* src_f64,const int* src_i32,
                               unsigned char* color,int* obj_map,int* stack,int* str_map){
  int old_count=RSA(src_i32,kStringsCount,row),old_bytes=RSA(src_i32,kBytesUsed,row);
  if(old_count<0||old_count>kMaxStrings||old_bytes<0||old_bytes>kMaxStringBytes)return kBounds;
  for(int p=0;p<kNPools;++p){
    int used=RSA(src_i32,kPools[p].used,row);
    if(used<0||used>kPools[p].capacity)return kBounds;
  }
  int status=kOk,sp=enter_ref(root,rows,row,src_u8,src_i32,color,obj_map,stack,0,&status);
  while(status==kOk&&sp){
    int ref=RSA(stack,sp-1,row),slot=0,p=locate_pool(ref,&slot);
    const PoolRec& pool=kPools[p];
    int idx=pool.base+slot,count=RSA(src_i32,pool.count+slot,row);
    int cursor=RSA(obj_map,idx,row),limit=child_limit(pool,count);
    if(cursor>=limit){RSA(color,idx,row)=2;sp--;continue;}
    RSA(obj_map,idx,row)=cursor+1;
    int index=0,cid=child_col(pool,slot,cursor,rows,row,src_i32,&index,&status);
    if(status!=kOk)break;
    int child=-1;
    status=follow_value(kCols[cid],index,rows,row,src_u8,src_i64,src_f64,src_i32,
                        old_count,old_bytes,str_map,&child);
    if(status==kOk&&child>=0)sp=enter_ref(child,rows,row,src_u8,src_i32,color,obj_map,stack,sp,&status);
  }
  return status;
}
__device__ inline int assign_maps(int rows,int row,const int* src_i32,unsigned char* color,
                                  int* obj_map,int* dst_i32){
  for(int p=0;p<kNPools;++p){
    const PoolRec& pool=kPools[p];
    int new_used=0,old_used=RSA(src_i32,pool.used,row);
    for(int slot=0;slot<old_used;++slot){
      int idx=pool.base+slot;
      if(RSA(color,idx,row)!=2)continue;
      RSA(obj_map,idx,row)=pool.base+new_used+1;
      new_used++;
    }
    RSA(dst_i32,pool.used,row)=new_used;
  }
  return kOk;
}
'''

_COMPACT_COPY = r'''
__device__ inline int strings_equal(int old_i,int new_i,int rows,int row,
                                    const int* src_i32,const long long* src_i64,const unsigned char* src_u8,
                                    const int* dst_i32,const long long* dst_i64,const unsigned char* dst_u8){
  if(RSA(src_i64,kStringHashes+old_i,row)!=RSA(dst_i64,kStringHashes+new_i,row))return 0;
  int old_n=RSA(src_i32,kStringLengths+old_i,row),new_n=RSA(dst_i32,kStringLengths+new_i,row);
  if(old_n!=new_n)return 0;
  int old_s=RSA(src_i32,kStringOffsets+old_i,row),new_s=RSA(dst_i32,kStringOffsets+new_i,row);
  for(int i=0;i<old_n;++i){
    if(RSA(src_u8,kStringBytes+old_s+i,row)!=RSA(dst_u8,kStringBytes+new_s+i,row))return 0;
  }
  return 1;
}
__device__ inline int compact_strings(int rows,int row,
                                      const int* src_i32,const long long* src_i64,const unsigned char* src_u8,
                                      int* dst_i32,long long* dst_i64,unsigned char* dst_u8,int* str_map){
  int old_count=RSA(src_i32,kStringsCount,row),new_count=0,bytes_used=0;
  for(int old=0;old<old_count;++old){
    if(RSA(str_map,old,row)!=1){RSA(str_map,old,row)=0;continue;}
    int found=-1;
    for(int prior=0;prior<new_count;++prior){
      if(strings_equal(old,prior,rows,row,src_i32,src_i64,src_u8,dst_i32,dst_i64,dst_u8)){found=prior;break;}
    }
    if(found>=0){RSA(str_map,old,row)=found+1;continue;}
    int start=RSA(src_i32,kStringOffsets+old,row),size=RSA(src_i32,kStringLengths+old,row);
    if(new_count>=kMaxStrings||bytes_used+size>kMaxStringBytes)return kCapacity;
    RSA(dst_i32,kStringOffsets+new_count,row)=bytes_used;
    RSA(dst_i32,kStringLengths+new_count,row)=size;
    RSA(dst_i64,kStringHashes+new_count,row)=RSA(src_i64,kStringHashes+old,row);
    RSA(dst_i32,kStringStaticRefs+new_count,row)=RSA(src_i32,kStringStaticRefs+old,row);
    for(int i=0;i<size;++i)RSA(dst_u8,kStringBytes+bytes_used+i,row)=RSA(src_u8,kStringBytes+start+i,row);
    RSA(str_map,old,row)=new_count+1;new_count++;bytes_used+=size;
  }
  RSA(dst_i32,kStringsCount,row)=new_count;RSA(dst_i32,kBytesUsed,row)=bytes_used;
  for(int rest=new_count;rest<kMaxStrings;++rest)RSA(dst_i32,kStringStaticRefs+rest,row)=kMissing;
  return kOk;
}
__device__ inline int copy_value(const ColRec& c,int old_index,int new_index,int rows,int row,
                                 const unsigned char* src_u8,const long long* src_i64,
                                 const double* src_f64,const int* src_i32,
                                 unsigned char* dst_u8,long long* dst_i64,double* dst_f64,int* dst_i32,
                                 const int* obj_map,const int* str_map){
  int kind=RSA(src_u8,c.tag+old_index,row);
  RSA(dst_u8,c.tag+new_index,row)=(unsigned char)kind;
  if(kind==kAbsent||kind==kNull)return kOk;
  if(kind==kBool||kind==kInt){RSA(dst_i64,c.integer+new_index,row)=RSA(src_i64,c.integer+old_index,row);return kOk;}
  if(kind==kFloat){RSA(dst_f64,c.real+new_index,row)=RSA(src_f64,c.real+old_index,row);return kOk;}
  if(kind==kText){
    int sid=RSA(str_map,RSA(src_i32,c.word+old_index,row)-1,row);
    if(sid<1)return kBounds;RSA(dst_i32,c.word+new_index,row)=sid;return kOk;
  }
  if(kind!=kRef)return kUnsupported;
  int slot=0,p=locate_pool(RSA(src_i32,c.word+old_index,row),&slot);
  if(p<0)return kBounds;
  int mapped=RSA(obj_map,kPools[p].base+slot,row);
  if(mapped<1)return kBounds;
  RSA(dst_i32,c.word+new_index,row)=mapped;return kOk;
}
'''

_COMPACT_KERNEL = r'''
__device__ inline int copy_object(const PoolRec& pool,int old_slot,int new_slot,int rows,int row,
                                  const unsigned char* src_u8,const long long* src_i64,
                                  const double* src_f64,const int* src_i32,
                                  unsigned char* dst_u8,long long* dst_i64,double* dst_f64,int* dst_i32,
                                  const int* obj_map,const int* str_map){
  RSA(dst_u8,pool.tag+new_slot,row)=RSA(src_u8,pool.tag+old_slot,row);
  int count=RSA(src_i32,pool.count+old_slot,row);
  RSA(dst_i32,pool.count+new_slot,row)=count;
  if(pool.kind==kRecord){
    for(int o=0;o<count;++o){
      int fid=RSA(src_i32,pool.order+old_slot*pool.max_items+o,row);
      RSA(dst_i32,pool.order+new_slot*pool.max_items+o,row)=fid;
      int st=copy_value(kCols[pool.field_start+fid],old_slot,new_slot,rows,row,
                        src_u8,src_i64,src_f64,src_i32,dst_u8,dst_i64,dst_f64,dst_i32,obj_map,str_map);
      if(st!=kOk)return st;
    }
    return kOk;
  }
  for(int o=0;o<count;++o){
    int st;
    if(pool.kind==kTuple){
      st=copy_value(kCols[pool.value_start+o],old_slot,new_slot,rows,row,
                    src_u8,src_i64,src_f64,src_i32,dst_u8,dst_i64,dst_f64,dst_i32,obj_map,str_map);
    }else{
      int old_i=old_slot*pool.max_items+o,new_i=new_slot*pool.max_items+o;
      if(pool.kind==kMap){
        st=copy_value(kCols[pool.keys],old_i,new_i,rows,row,
                      src_u8,src_i64,src_f64,src_i32,dst_u8,dst_i64,dst_f64,dst_i32,obj_map,str_map);
        if(st!=kOk)return st;
      }
      st=copy_value(kCols[pool.value_start],old_i,new_i,rows,row,
                    src_u8,src_i64,src_f64,src_i32,dst_u8,dst_i64,dst_f64,dst_i32,obj_map,str_map);
    }
    if(st!=kOk)return st;
  }
  return kOk;
}
__device__ inline int copy_live(int rows,int row,
                                const unsigned char* src_u8,const long long* src_i64,
                                const double* src_f64,const int* src_i32,
                                unsigned char* dst_u8,long long* dst_i64,double* dst_f64,int* dst_i32,
                                const unsigned char* color,const int* obj_map,const int* str_map){
  for(int p=0;p<kNPools;++p){
    const PoolRec& pool=kPools[p];
    int new_slot=0,old_used=RSA(src_i32,pool.used,row);
    for(int old_slot=0;old_slot<old_used;++old_slot){
      if(RSA(color,pool.base+old_slot,row)!=2)continue;
      int st=copy_object(pool,old_slot,new_slot,rows,row,src_u8,src_i64,src_f64,src_i32,
                         dst_u8,dst_i64,dst_f64,dst_i32,obj_map,str_map);
      if(st!=kOk)return st;
      new_slot++;
    }
  }
  return kOk;
}
extern "C" __global__ void record_soa_compact(
 const long long* src_i64,const double* src_f64,const int* src_i32,const unsigned char* src_u8,
 const int* src_roots,long long* dst_i64,double* dst_f64,int* dst_i32,unsigned char* dst_u8,
 int* dst_roots,int* obj_map,int* stack,unsigned char* color,int* str_map,int* status,int rows){
  int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(row>=rows)return;
  for(int i=0;i<kTotalObjects;++i){RSA(color,i,row)=0;RSA(obj_map,i,row)=0;}
  for(int i=0;i<kMaxStrings;++i)RSA(str_map,i,row)=0;
  int st=discover(src_roots[row],rows,row,src_u8,src_i64,src_f64,src_i32,color,obj_map,stack,str_map);
  if(st==kOk)st=assign_maps(rows,row,src_i32,color,obj_map,dst_i32);
  if(st==kOk)st=compact_strings(rows,row,src_i32,src_i64,src_u8,dst_i32,dst_i64,dst_u8,str_map);
  if(st==kOk)st=copy_live(rows,row,src_u8,src_i64,src_f64,src_i32,dst_u8,dst_i64,dst_f64,dst_i32,color,obj_map,str_map);
  if(st==kOk){
    int slot=0,p=locate_pool(src_roots[row],&slot);
    if(p<0)st=kBounds;
    else{
      int mapped=RSA(obj_map,kPools[p].base+slot,row);
      if(mapped<1)st=kBounds;else dst_roots[row]=mapped;
    }
  }
  status[row]=st;
}
#ifndef __CUDACC__
extern "C" void host_record_soa_compact(
 const long long* src_i64,const double* src_f64,const int* src_i32,const unsigned char* src_u8,
 const int* src_roots,long long* dst_i64,double* dst_f64,int* dst_i32,unsigned char* dst_u8,
 int* dst_roots,int* obj_map,int* stack,unsigned char* color,int* str_map,int* status,int rows){
  for(int row=0;row<rows;++row){
    native_array_configure_row(row,rows);
    record_soa_compact(src_i64,src_f64,src_i32,src_u8,src_roots,dst_i64,dst_f64,dst_i32,dst_u8,
                       dst_roots,obj_map,stack,color,str_map,status,rows);
  }
}
#endif
'''


def emit_compact_source(schema):
    """CUDA C++ for the same typed-pool compact algorithm as compact_rows."""
    if not isinstance(schema, CompiledRecordSchema):
        raise RecordSoAError('Compiled schema required')
    structs, helpers = _COMPACT_PREFIX.split('__device__', 1)
    return (structs + _emit_tables(schema) + '__device__' + helpers +
            _COMPACT_DISCOVER + _COMPACT_WALK + _COMPACT_COPY + _COMPACT_KERNEL)


class CudaRecordSoACompact:
    """CUDA-only lazy compact. Importing this module does not load torch."""

    gpu_qualified = False

    def __init__(self, schema, device='cuda'):
        if not isinstance(schema, CompiledRecordSchema):
            raise RecordSoAError('Compiled schema required')
        self.schema = schema
        self.device = device
        self._module = None
        self._keepalive = []
        self._last_workspace = None

    def _ensure(self):
        if self._module is not None:
            return self._module
        import torch
        from .cuda_runtime import CudaModule
        if not str(self.device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Actual CUDA required, no CPU compact fallback')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare typed compact kernel before CUDA capture')
        self._module = CudaModule(emit_compact_source(self.schema), CudaModuleKernelSpec, device=self.device)
        return self._module

    def compact(self, batch):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Host compact verification cannot run during CUDA capture')
        result, status, event = self.compact_device(batch)
        if event is None:
            torch.cuda.current_stream(result.roots.device).synchronize()
        else:
            event.synchronize()
        codes = status.detach().cpu().numpy()
        bad = np.flatnonzero(codes)
        if len(bad):
            raise RecordSoAError('Record SoA compact %s' % _STATUS[int(codes[bad[0]])])
        self._last_workspace=None
        return result

    def compact_device(self, batch):
        """Capturable typed collection; errors remain on device for the caller."""
        workspace=self.prepare_workspace(batch.rows)
        import torch
        if torch.cuda.is_current_stream_capturing():
            self._keepalive.append(workspace)
        else:
            self._last_workspace=workspace
        return self.compact_into_device(batch,workspace)

    def prepare_workspace(self,rows):
        from types import SimpleNamespace
        if type(rows) is not int or rows<0:
            raise RecordSoAError('Nonnegative static compact row count required')
        module=self._ensure()
        import torch
        make=lambda shape,dtype:torch.zeros(shape,dtype=dtype,device=module.device)
        buffers={key:make((self.schema.lengths[key],rows),dtype) for key,dtype in (
            ('i64',torch.int64),('f64',torch.float64),('i32',torch.int32),('u8',torch.uint8))}
        roots=make((rows,),torch.int32)
        return SimpleNamespace(schema_identity=self.schema.identity,rows=rows,
            destination=TorchRecordSoABatch(self.schema,buffers,roots),
            obj_map=make((self.schema.total_objects,rows),torch.int32),
            stack=make((self.schema.total_objects,rows),torch.int32),
            color=make((self.schema.total_objects,rows),torch.uint8),
            strings=make((self.schema.max_strings,rows),torch.int32),status=make((rows,),torch.int32))

    def compact_into_device(self,batch,workspace):
        """Reuse a prepared destination; consume it before the next collection."""
        if not isinstance(batch, TorchRecordSoABatch):
            raise RecordSoATypeError('Torch typed domain record batch required')
        if batch.schema.identity != self.schema.identity:
            raise RecordSoAError('Matching compiled schema required')
        if workspace.schema_identity!=self.schema.identity or workspace.rows!=batch.rows:
            raise RecordSoAError('Matching typed collection workspace required')
        module = self._ensure()
        rows=batch.rows
        dest_buffers=workspace.destination.buffers
        for value in dest_buffers.values():
            value.zero_()
        dest_buffers['i32'][self.schema.string_static_refs:self.schema.string_static_refs+self.schema.max_strings] = MISSING
        dest_roots=workspace.destination.roots
        dest_roots.zero_()
        obj_map,stack,color,str_map,status=(getattr(workspace,key) for key in ('obj_map','stack','color','strings','status'))
        for value in (obj_map,stack,color,str_map,status):
            value.zero_()
        if rows == 0:
            return workspace.destination,status,None
        event = module.launch('record_soa_compact', [
            batch.buffers['i64'], batch.buffers['f64'], batch.buffers['i32'], batch.buffers['u8'], batch.roots,
            dest_buffers['i64'], dest_buffers['f64'], dest_buffers['i32'], dest_buffers['u8'], dest_roots,
            obj_map, stack, color, str_map, status, rows,
        ], grid=(max(rows, 1) + 127) // 128, block=128,capture_safe=True)
        return workspace.destination,status,event

    def close(self):
        if self._module is not None:
            self._module.close()
            self._module = None
        self._keepalive.clear()
        self._last_workspace=None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
