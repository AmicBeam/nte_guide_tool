"""Exact indexes for immutable class namespaces and their frozen C3 order.

No instance attributes, getters or bindings are evaluated here. The existing
object handler consumes the same value reference and descriptor category.
"""
from hashlib import sha256


def emit_frozen_attributes(image, orders):
    image.verify()
    strings = {}
    refs = {}
    for index, (tag, offset, count, _) in enumerate(image.nodes):
        if int(tag) == 4:
            raw = image.payload[int(offset):int(offset)+int(count)].tobytes()
            strings[-1-index] = raw
            refs.setdefault(raw, -1-index)
    canonical = [refs[strings[-1-index]] if -1-index in strings else -2147483648
                 for index in range(len(image.nodes))]

    def own_namespace(cls):
        result = {}
        for column in (5, 2, 3, 4, 1):
            ref = int(cls[column])
            if ref >= 0:
                return None
            tag, offset, count, _ = map(int, image.nodes[-1-ref])
            if tag != 7:
                return None
            for key, value in image.edges[offset:offset+count]:
                key = int(key)
                if key not in strings:
                    return None
                result.setdefault(strings[key], (int(value), column))
        return result

    own = [own_namespace(cls) for cls in image.classes]
    inherited = []
    for order in orders:
        merged = {}
        for cid in order:
            if own[cid] is None:
                merged = None
                break
            for raw, value in own[cid].items():
                merged.setdefault(raw, value)
        inherited.append(merged)

    def table(name, rows, width):
        data = ','.join(str(x) for row in rows for x in row) or '0'
        return f'__device__ const int {name}[{max(1, len(rows)*width)}]={{'+data+'};\n'

    def index(name, namespaces):
        metadata, entries, values = [], [], []
        for namespace in namespaces:
            if namespace is None:
                metadata.append((0, -1, 0))
                continue
            metadata.append((len(entries), len(namespace), len(values)))
            entries.extend((refs[raw], *value) for raw, value in
                           sorted(namespace.items(), key=lambda pair: refs[pair[0]]))
            values.extend((refs[raw], *namespace[raw]) for raw in sorted(namespace))
        return (table(name+'_meta', metadata, 3) + table(name+'_entries', entries, 3) +
                table(name+'_values', values, 3))

    source = (table('frozen_attr_canonical', [(r,) for r in canonical], 1) +
              index('frozen_attr_own', own) + index('frozen_attr_class', inherited))
    source += r'''
// Return 1 only for a supported, immutable all-string namespace and string
// query. An absent name is then exact; all other cases use the old lookup.
__device__ inline int frozen_attr_lookup(Context* c,int cid,int name,int* category,
 int* result,const int* metadata,const int* entries,const int* values){
 if(cid<0||cid>=FROZEN_ATTR_CLASSES)return 0;
 int count=metadata[cid*3+1];if(count<0)return 0;
 NodeView query;if(!get_node(&c->h,name,&query)||query.tag!=TAG_STR)return 0;
 int lo=0,hi=count;
 if(name<0){
  long long index=-1LL-name;if(index>=FROZEN_ATTR_NODES)return 0;
  int canonical=frozen_attr_canonical[index];
  int begin=metadata[cid*3];
  while(lo<hi){int mid=lo+(hi-lo)/2;int key=entries[(begin+mid)*3];
   if(key<canonical)lo=mid+1;else hi=mid;}
  if(lo<count&&entries[(begin+lo)*3]==canonical){
   *result=entries[(begin+lo)*3+1];*category=entries[(begin+lo)*3+2];
  }else *result=MISSING;
  return 1;
 }
 const unsigned char* bytes=get_payload_ptr(&c->h,&query);
 if(!bytes){c->h.error=ERROR_INVALID_REF;*result=MISSING;return 1;}
 int begin=metadata[cid*3+2];
 while(lo<hi){int mid=lo+(hi-lo)/2;NodeView candidate;
  if(!get_node(&c->h,values[(begin+mid)*3],&candidate)){
   c->h.error=ERROR_INVALID_REF;*result=MISSING;return 1;}
  const unsigned char* other=get_payload_ptr(&c->h,&candidate);
  if(!other){c->h.error=ERROR_INVALID_REF;*result=MISSING;return 1;}
  int shared=candidate.count<query.count?candidate.count:query.count;
  int compare=vm_memcmp(other,bytes,shared);
  if(!compare)compare=candidate.count<query.count?-1:(candidate.count>query.count?1:0);
  if(!compare){*result=values[(begin+mid)*3+1];*category=values[(begin+mid)*3+2];return 1;}
  if(compare<0)lo=mid+1;else hi=mid;
 }
 *result=MISSING;return 1;
}
'''.replace('FROZEN_ATTR_CLASSES', str(len(image.classes))).replace(
        'FROZEN_ATTR_NODES', str(len(image.nodes)))
    return source, sha256(source.encode()).hexdigest()


def integrate_frozen_attributes(source, image, orders):
    plan, _ = emit_frozen_attributes(image, orders)
    own = ' const int* cls=c->classes+cid*6;'
    inherited = '__device__ inline int lookup_class(Context* c,int cid,int name,int* category){\n'
    if source.count(own) != 1 or source.count(inherited) != 1:
        raise ValueError('Frozen attribute integration point changed')
    source = source.replace(own, ''' int result=MISSING;
 if(frozen_attr_lookup(c,cid,name,category,&result,frozen_attr_own_meta,
  frozen_attr_own_entries,frozen_attr_own_values))return result;
''' + own, 1)
    source = source.replace(inherited, inherited + ''' int result=MISSING;
 if(frozen_attr_lookup(c,cid,name,category,&result,frozen_attr_class_meta,
  frozen_attr_class_entries,frozen_attr_class_values))return result;
''', 1)
    return plan + source
