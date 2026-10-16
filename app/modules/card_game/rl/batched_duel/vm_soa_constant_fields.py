"""Resolve frozen field-name constants to direct domain column offsets."""


def integrate_constant_fields(source,schema,image):
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
    tables='''
__device__ const int SOA_FAST_KEY[]={'''+','.join(map(str,keys))+'''};
__device__ const short SOA_FAST_FIELD[]={'''+','.join(map(str,fields))+'''};
__device__ inline int soa_constant_field(int pool,int key){
 if(key>=0||key<(-'''+str(len(keys))+'''))return -2;
 int code=SOA_FAST_KEY[-1-key];if(code<0)return code;
 return SOA_FAST_FIELD[pool*'''+str(len(names))+'''+code];
}
'''
    marker='__device__ inline int soa_dict_find(Heap* heap,int dict_ref,int key_ref){'
    if source.count(marker)!=1:
        raise ValueError('Typed field lookup boundary changed')
    source=source.replace(marker,tables+'\n'+marker,1)
    scan=''' if(SOA_POOL_KIND[pool]==SOA_KIND_RECORD){
  for(int i=0;i<count;i++){'''
    fast=''' if(SOA_POOL_KIND[pool]==SOA_KIND_RECORD){
  int fixed=soa_constant_field(pool,key_ref);
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
   int field=soa_constant_field(pool,key_ref);
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
    fast='''  int field=soa_constant_field(pool,key_ref);
  if(field<0)for(int i=0;i<SOA_POOL_NFIELDS[pool];i++)if(soa_text_equals_name(heap,key_ref,SOA_POOL_FIELD_START[pool]+i)){field=i;break;}'''
    if source.count(scan)!=1:
        raise ValueError('Typed dictionary write boundary changed')
    return source.replace(scan,fast,1)
