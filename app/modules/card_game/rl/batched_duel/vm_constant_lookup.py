"""Static GPU index for exact immutable string-key hits in a frozen VM image.

Only negative constant references are indexed. Misses, mutable dictionaries,
dynamic strings and non-string keys retain the existing Python-equality path.
The index never caches mutable heap data or changes the rule program.
"""
from dataclasses import dataclass
from hashlib import sha256
from .vm_image import VmImage


@dataclass(frozen=True)
class ConstantLookup:
    source: str
    identity: str
    dictionaries: int
    indexed_keys: int


def emit_constant_lookup(image):
    if not isinstance(image, VmImage):
        raise TypeError('Numeric VM image required')
    image.verify()
    records = []
    keys = []
    for node_index, (tag, offset, count, _) in enumerate(image.nodes):
        if int(tag) != 7 or not int(count):
            continue
        offset, count = int(offset), int(count)
        if offset < 0 or count < 0 or offset + count > len(image.edges):
            raise ValueError('Constant dictionary edges outside image')
        first = {}
        aliases = {}
        for index, (key, _) in enumerate(image.edges[offset:offset + count]):
            key = int(key)
            if key >= 0 or -1-key >= len(image.nodes):
                continue
            kind, begin, size, _ = map(int, image.nodes[-1-key])
            if kind != 4:
                continue
            if begin < 0 or size < 0 or begin + size > len(image.payload):
                raise ValueError('Constant string outside image')
            raw = image.payload[begin:begin+size].tobytes()
            first.setdefault(raw, index)
            aliases[key] = raw
        if not aliases:
            continue
        # Restrict acceleration to all-string dictionaries: an earlier custom
        # or numeric key must not be bypassed by an identity-only lookup.
        if len(aliases) != count:
            continue
        entries = sorted((key, first[raw]) for key, raw in aliases.items())
        records.append((-1-node_index, entries))
    records.sort()
    metadata = []
    for ref, entries in records:
        metadata.append((ref, len(keys), len(entries)))
        keys.extend(entries)
    def table(name, rows, width):
        values = ','.join(str(value) for row in rows for value in row) or '0'
        return '__device__ const int '+name+'['+str(max(1,len(rows)*width))+']={'+values+'};\n'
    source = table('frozen_dict_index', metadata, 3) + table('frozen_dict_keys', keys, 2)
    source += '''__device__ inline int frozen_dict_hit(int dict_ref,int key_ref){
 if(dict_ref>=0||key_ref>=0)return -2;
 int lo=0,hi=FROZEN_DICT_COUNT;
 while(lo<hi){int mid=lo+(hi-lo)/2;int ref=frozen_dict_index[mid*3];
  if(ref<dict_ref)lo=mid+1;else hi=mid;}
 if(lo>=FROZEN_DICT_COUNT||frozen_dict_index[lo*3]!=dict_ref)return -2;
 int begin=frozen_dict_index[lo*3+1],end=begin+frozen_dict_index[lo*3+2];lo=begin;hi=end;
 while(lo<hi){int mid=lo+(hi-lo)/2;int ref=frozen_dict_keys[mid*2];
  if(ref<key_ref)lo=mid+1;else hi=mid;}
 return lo<end&&frozen_dict_keys[lo*2]==key_ref?frozen_dict_keys[lo*2+1]:-2;
}
'''.replace('FROZEN_DICT_COUNT', str(len(metadata)))
    return ConstantLookup(source, sha256(source.encode()).hexdigest(), len(metadata), len(keys))


def integrate_constant_lookup(heap_source, plan):
    """Insert after existing dict validation; default heap emission is untouched."""
    marker = '// 13. dict_find:'
    if heap_source.count(marker) != 1:
        raise ValueError('Frozen dictionary index integration point changed')
    source = heap_source.replace(marker, plan.source + '\n' + marker, 1)
    start = source.index('__device__ inline int dict_find(')
    point = source.index('    for (int i = 0; i < node.count; i++) {', start)
    source = source[:point] + '''    if (dict_ref < 0 && key_ref < 0) {
        int hit = frozen_dict_hit(dict_ref, key_ref);
        if (hit >= 0) return hit;
    }
''' + source[point:]
    return source
