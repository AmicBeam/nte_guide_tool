"""Resolve exact constant and dynamic UTF-8 field names without boxing keys."""


def emit_record_field_lookup(schema,image):
    if image is None:
        raise ValueError('Frozen constant image required for direct field lookup')
    names=tuple(dict.fromkeys(field.name.split('.',1)[1] for pool in schema.pools for field in pool.fields))
    codes={name:index for index,name in enumerate(names)}
    keys=[]
    for tag,offset,count,_ in image.nodes:
        if int(tag)!=4:
            keys.append(-2)
            continue
        value=image.payload[int(offset):int(offset)+int(count)].tobytes().decode('utf-8')
        keys.append(codes.get(value,-1))
    fields=[]
    for pool in schema.pools:
        members={field.name.split('.',1)[1]:index for index,field in enumerate(pool.fields)}
        fields.extend(members.get(name,-1) for name in names)
    ordered=sorted(codes, key=lambda name:name.encode('utf-8'))
    offsets=[];lengths=[];raw=bytearray()
    for name in ordered:
        value=name.encode('utf-8');offsets.append(len(raw));lengths.append(len(value));raw.extend(value)
    name_index={name:i for i,name in enumerate(ordered)}
    starts=[];counts=[];dynamic_names=[];dynamic_fields=[]
    for pool in schema.pools:
        members={field.name.split('.',1)[1]:i for i,field in enumerate(pool.fields)}
        starts.append(len(dynamic_names));counts.append(len(members))
        for name in sorted(members,key=lambda value:value.encode('utf-8')):
            dynamic_names.append(name_index[name]);dynamic_fields.append(members[name])
    def numbers(values):return ','.join(map(str,values)) or '0'
    tables='''
__device__ const int SOA_FAST_KEY[]={'''+','.join(map(str,keys))+'''};
__device__ const short SOA_FAST_FIELD[]={'''+','.join(map(str,fields))+'''};
__device__ inline int soa_constant_field(int pool,int key){
 if(key>=0||key<(-'''+str(len(keys))+'''))return -2;
 int code=SOA_FAST_KEY[-1-key];if(code<0)return code;
 return SOA_FAST_FIELD[pool*'''+str(len(names))+'''+code];
}
'''
    tables+='''
__device__ const int SOA_FIELD_NAME_OFF[]={'''+numbers(offsets)+'''};
__device__ const int SOA_FIELD_NAME_LEN[]={'''+numbers(lengths)+'''};
__device__ const unsigned char SOA_FIELD_NAME_BYTES[]={'''+numbers(raw)+'''};
__device__ const int SOA_DYNAMIC_FIELD_START[]={'''+numbers(starts)+'''};
__device__ const int SOA_DYNAMIC_FIELD_COUNT[]={'''+numbers(counts)+'''};
__device__ const int SOA_DYNAMIC_FIELD_NAME[]={'''+numbers(dynamic_names)+'''};
__device__ const int SOA_DYNAMIC_FIELD_INDEX[]={'''+numbers(dynamic_fields)+'''};
__device__ inline int soa_record_field(Heap* heap,int pool,int key){
 int fixed=soa_constant_field(pool,key);if(fixed!=-2)return fixed;
 NodeView query;if(!get_node(heap,key,&query)||query.tag!=TAG_STR)return -2;
 const unsigned char* bytes=get_payload_ptr(heap,&query);if(!bytes)return -2;
 int lo=SOA_DYNAMIC_FIELD_START[pool],hi=lo+SOA_DYNAMIC_FIELD_COUNT[pool];
 while(lo<hi){
  int mid=lo+(hi-lo)/2,name=SOA_DYNAMIC_FIELD_NAME[mid];
  int len=SOA_FIELD_NAME_LEN[name],off=SOA_FIELD_NAME_OFF[name];
  int shared=len<query.count?len:query.count,compare=0;
  for(int i=0;i<shared;i++)if(SOA_FIELD_NAME_BYTES[off+i]!=bytes[i]){
   compare=SOA_FIELD_NAME_BYTES[off+i]<bytes[i]?-1:1;break;
  }
  if(!compare)compare=len<query.count?-1:(len>query.count?1:0);
  if(!compare)return SOA_DYNAMIC_FIELD_INDEX[mid];
  if(compare<0)lo=mid+1;else hi=mid;
 }
 return -1;
}
__device__ inline int soa_record_contains(Heap* heap,int owner,int key){
 if(!soa_is_view(heap,owner))return -2;
 int pool=0,slot=0;
 if(!soa_locate(heap->nodes[owner*4+3],&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
 if(SOA_POOL_KIND[pool]!=SOA_KIND_RECORD)return -2;
 if(!soa_refresh_header(heap,owner))return 0;
 int field=soa_record_field(heap,pool,key);if(field==-2)return -2;
 if(field<0)return 0;
 return soa_load_u8(heap->soa,SOA_COL_TAG[SOA_POOL_FIELD_START[pool]+field]+slot)!=SOA_ABSENT;
}
'''
    return tables


def integrate_constant_fields(source,schema,image):
    tables=emit_record_field_lookup(schema,image)
    marker='__device__ inline int soa_dict_find(Heap* heap,int dict_ref,int key_ref){'
    if source.count(marker)!=1:
        raise ValueError('Typed field lookup boundary changed')
    source=source.replace(marker,tables+'\n'+marker,1)
    scan=''' if(SOA_POOL_KIND[pool]==SOA_KIND_RECORD){
  for(int i=0;i<count;i++){'''
    fast=''' if(SOA_POOL_KIND[pool]==SOA_KIND_RECORD){
  int fixed=soa_record_field(heap,pool,key_ref);
  if(fixed==-1)return -1;
  if(fixed>=0){
   if(soa_load_u8(s,SOA_COL_TAG[SOA_POOL_FIELD_START[pool]+fixed]+slot)==SOA_ABSENT)return -1;
   for(int i=0;i<count;i++)if(soa_load_i32(s,SOA_POOL_ORDER[pool]+soa_item_index(pool,slot,i))==fixed)return i;
   return soa_fail(heap,ERROR_INVALID_REF);
  }
  for(int i=0;i<count;i++){'''
    if source.count(scan)!=1:
        raise ValueError('Typed record lookup body changed')
    source=source.replace(scan,fast,1)
    marker='__device__ inline int dict_get(Heap* heap, int dict_ref, int key_ref, int default_ref) {'
    direct='''
 if(soa_is_view(heap,dict_ref)){
  int pool=0,slot=0;
  if(!soa_locate(heap->nodes[dict_ref*4+3],&pool,&slot))return soa_fail(heap,ERROR_INVALID_REF);
  if(SOA_POOL_KIND[pool]==SOA_KIND_RECORD){
   // Dynamic keys previously passed through dict_find's header validation.
   if(key_ref>=0&&!soa_refresh_header(heap,dict_ref))return -1;
   int field=soa_record_field(heap,pool,key_ref);
   if(field==-1)return default_ref;
   if(field>=0){
    int col=SOA_POOL_FIELD_START[pool]+field;
    if(soa_load_u8(heap->soa,SOA_COL_TAG[col]+slot)==SOA_ABSENT)return default_ref;
    return soa_box_cell(heap,col,slot);
   }
  }
 }
'''
    if source.count(marker)!=1:
        raise ValueError('Canonical dictionary read boundary changed')
    source=source.replace(marker,marker+direct,1)
    scan='''  int field=-1;
  for(int i=0;i<SOA_POOL_NFIELDS[pool];i++)if(soa_text_equals_name(heap,key_ref,SOA_POOL_FIELD_START[pool]+i)){field=i;break;}'''
    fast='''  int field=soa_record_field(heap,pool,key_ref);
  if(field<0)for(int i=0;i<SOA_POOL_NFIELDS[pool];i++)if(soa_text_equals_name(heap,key_ref,SOA_POOL_FIELD_START[pool]+i)){field=i;break;}'''
    if source.count(scan)!=1:
        raise ValueError('Typed dictionary write boundary changed')
    source=source.replace(scan,fast,1)
    membership=''' int tag=kind(c,owner);if(tag==80)return dict_find(&c->h,node(c,owner)[1],needle)>=0;'''
    direct=''' int tag=kind(c,owner);
 if(owner>=0&&(tag==TAG_DICT||(tag>=8&&tag<=10))){
  int found=soa_record_contains(&c->h,owner,needle);if(found>=0)return found;
 }
 if(tag==80)return dict_find(&c->h,node(c,owner)[1],needle)>=0;'''
    if source.count(membership)!=1:
        raise ValueError('Canonical record membership boundary changed')
    source=source.replace(membership,direct,1)
    checks=(
        ('''  int matched=0;
  for(int f=0;f<SOA_POOL_NFIELDS[pool];f++)if(soa_text_equals_name((Heap*)heap,k,SOA_POOL_FIELD_START[pool]+f)){matched=1;break;}''',
         '''  int matched=soa_record_field(heap,pool,k)>=0;'''),
        ('''   int field=-1;
   for(int f=0;f<SOA_POOL_NFIELDS[pool];f++)if(soa_text_equals_name(heap,k,SOA_POOL_FIELD_START[pool]+f)){field=f;break;}''',
         '''   int field=soa_record_field(heap,pool,k);'''),
    )
    for old,new in checks:
        if source.count(old)!=1:
            raise ValueError('Typed record adoption boundary changed')
        source=source.replace(old,new,1)
    return source
