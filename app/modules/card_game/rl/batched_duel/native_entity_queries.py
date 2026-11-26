"""Bounded immutable string-tuple membership in the shared runtime only.

Entity/registry queries repeatedly test dynamic character IDs against frozen
orders. Only an all-string frozen tuple and a plain string needle use this
index. Mutable dictionaries, typed fields, subclasses and mixed tuples retain
their canonical queries; no live entity or effect result is retained.
"""
from dataclasses import replace
import re

from .native_array_codegen import _matching_brace_end
from .vm_image import VmImage


def _membership_rows(image):
    rows = []
    for index, (tag, offset, count, _) in enumerate(image.nodes):
        tag, offset, count = int(tag), int(offset), int(count)
        if tag != 6 or not 4 <= count <= 256:
            continue
        if offset < 0 or offset + count > len(image.edges):
            raise ValueError('Frozen membership edges outside image')
        members = {}
        for ref in image.edges[offset:offset + count, 0]:
            ref = int(ref)
            if ref >= 0 or -1-ref >= len(image.nodes):
                break
            kind, begin, size, _ = map(int, image.nodes[-1-ref])
            if kind != 4:
                break
            if begin < 0 or size < 0 or begin + size > len(image.payload):
                raise ValueError('Frozen membership string outside image')
            members.setdefault(image.payload[begin:begin+size].tobytes(), ref)
        else:
            rows.append((-1-index, tuple(members[raw] for raw in sorted(members))))
    selected = []
    entries = 0
    for ref, members in sorted(rows, key=lambda row: (-len(row[1]), row[0])):
        if len(selected) == 64:
            break
        if entries + len(members) <= 2048:
            selected.append((ref, members))
            entries += len(members)
    return sorted(selected)


def specialize_entity_queries(unit, image):
    """Specialize one runtime owner while preserving all common inline bodies."""
    if unit.name != 'runtime':
        raise ValueError('Entity queries require the shared runtime unit')
    if not isinstance(image, VmImage):
        raise TypeError('Numeric VM image required')
    image.verify()
    if 'native_entity_string_membership' in unit.source:
        raise ValueError('Entity membership already specialized')
    pattern = re.compile(r'(?m)^__device__ __noinline__ int contains\(Context\* c,int needle,int owner\)\s*\{')
    matches = list(pattern.finditer(unit.source))
    if len(matches) != 1:
        raise ValueError('Unique shared membership owner required')
    match = matches[0]
    end = _matching_brace_end(unit.source, match.end()-1)
    body = unit.source[match.start():end]
    anchor = ' int tag=kind(c,owner);'
    canonical = ' int n=length(c,owner);for(int i=0;i<n;i++)if(vm_equal(c,needle,edge(c,owner,i)))return 1;return 0;'
    if body.count(anchor) != 1 or body.count(canonical) != 1:
        raise ValueError('Canonical sequence membership boundary changed')
    rows = _membership_rows(image)
    metadata, members = [], []
    for ref, values in rows:
        metadata.extend((ref, len(members), len(values)))
        members.extend(values)
    def table(name, values):
        return '__device__ const int ' + name + '[]={' + (','.join(map(str, values)) or '0') + '};\n'
    helper = table('native_entity_string_membership_meta', metadata)
    helper += table('native_entity_string_membership_values', members)
    helper += '''__device__ inline int native_entity_string_membership(Context* c,int owner,int needle){
 if(owner>=0||c->error||c->h.error)return -2;
 int lo=0,hi=ENTITY_MEMBERSHIP_COUNT;
 while(lo<hi){int mid=lo+(hi-lo)/2;
  if(native_entity_string_membership_meta[mid*3]<owner)lo=mid+1;else hi=mid;}
 if(lo>=ENTITY_MEMBERSHIP_COUNT||native_entity_string_membership_meta[lo*3]!=owner)return -2;
 int record=lo;
 // Probe without changing VM/heap error ordering. Invalid references must
 // enter the original equality loop, which owns their precise error status.
 NodeView query;if(!get_node(&c->h,needle,&query)||query.tag!=TAG_STR)return -2;
 const unsigned char* bytes=get_payload_ptr(&c->h,&query);
 if(!bytes)return -2;
 int begin=native_entity_string_membership_meta[record*3+1];
 lo=begin;hi=begin+native_entity_string_membership_meta[record*3+2];
 while(lo<hi){int mid=lo+(hi-lo)/2;NodeView candidate;
  if(!get_node(&c->h,native_entity_string_membership_values[mid],&candidate))return -2;
  const unsigned char* other=get_payload_ptr(&c->h,&candidate);
  if(!other)return -2;
  int common=candidate.count<query.count?candidate.count:query.count;
  int order=vm_memcmp(other,bytes,common);
  if(!order)order=candidate.count<query.count?-1:(candidate.count>query.count?1:0);
  if(!order)return 1;
  if(order<0)lo=mid+1;else hi=mid;
 }
 return 0;
}
'''.replace('ENTITY_MEMBERSHIP_COUNT', str(len(rows)))
    hook = '''
 if(tag==TAG_TUPLE&&owner<0){
  int found=native_entity_string_membership(c,owner,needle);if(found!=-2)return found;
 }
'''
    body = body.replace(anchor, anchor + hook, 1)
    return replace(unit, source=unit.source[:match.start()] + helper + body + unit.source[end:])
