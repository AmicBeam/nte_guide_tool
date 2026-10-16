"""View-aware language method operations used by the canonical rule program."""
import re


_METHODS = r'''
__device__ inline int soa_method_pop(Context* c,int owner,int index,int* out){
 int count=length(c,owner);if(index<0)index+=count;
 if(index<0||index>=count){c->h.error=ERROR_INDEX_OUT_OF_BOUNDS;return -1;}
 *out=edge(c,owner,index);if(c->error||c->h.error)return -1;
 return soa_list_delete(&c->h,owner,index);
}
__device__ inline int soa_method_insert(Context* c,int owner,int index,int item){
 int count=length(c,owner);if(index<0){index+=count;if(index<0)index=0;}
 if(index>count)index=count;
 if(soa_list_append(&c->h,owner,item)<0)return -1;
 for(int i=count;i>index;i--){int value=edge(c,owner,i-1);if(c->error||c->h.error)return -1;
  if(soa_list_set(&c->h,owner,i,value)<0)return -1;}
 return soa_list_set(&c->h,owner,index,item);
}
__device__ inline int soa_method_remove(Context* c,int owner,int item){
 int count=length(c,owner);
 for(int i=0;i<count;i++){
  int value=edge(c,owner,i);if(c->error||c->h.error)return -1;
  if(vm_equal(c,value,item))return soa_list_delete(&c->h,owner,i);
  if(c->error||c->h.error)return -1;
 }
 c->error=BAD_CALL;return -1;
}
__device__ inline int soa_method_reverse(Context* c,int owner){
 int count=length(c,owner);
 for(int i=0;i<count/2;i++){
  int left=edge(c,owner,i),right=edge(c,owner,count-1-i);
  if(c->error||c->h.error)return -1;
  if(soa_list_set(&c->h,owner,i,right)<0||soa_list_set(&c->h,owner,count-1-i,left)<0)return -1;
 }
 return 0;
}
__device__ inline int soa_method_repeat(Context* c,int owner,long long times){
 int count=length(c,owner);if(times<0)times=0;
 int global=soa_global_ref(&c->h,owner),pool=0,slot=0;
 if(!soa_locate(global,&pool,&slot)){c->error=BAD_PROGRAM;return MISSING;}
 if(count && times>SOA_POOL_MAX_ITEMS[pool]/count){c->h.error=ERROR_EDGE_CAPACITY;return MISSING;}
 int saved=make_list(&c->h);if(saved<0)return MISSING;
 for(int i=0;i<count;i++)if(list_append(&c->h,saved,edge(c,owner,i))<0)return MISSING;
 soa_store_i32(c->h.soa,SOA_POOL_COUNT_COL[pool]+slot,0);
 c->h.nodes[owner*4+2]=0;
 for(long long i=0;i<times*count;i++)if(soa_list_append(&c->h,owner,edge(c,saved,(int)(i%count)))<0)return MISSING;
 return owner;
}
'''

_IMMUTABLE = r'''
__device__ inline int soa_tuple_immutable(Heap* heap,int global,int depth){
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
  if(tag==SOA_REF&&!soa_tuple_immutable(heap,soa_load_i32(s,SOA_COL_WORD[col]+index),depth+1))return 0;
 }
 return 1;
}
'''


def integrate_soa_methods(source):
    source=re.sub(r'(?<!__device__ )\b(const (?:int|unsigned char) SOA_\w+\[\])',r'__device__ \1',source)
    source=source.replace(' int constant_node_count;',' int constant_node_count; int failed_pool;',1)
    source=source.replace('return heap->nodes[ref*4+1]==SOA_VIEW_OFFSET;',
        'return heap->nodes[ref*4]>=TAG_LIST&&heap->nodes[ref*4]<=TAG_CARD_ENTITY&&heap->nodes[ref*4+1]==SOA_VIEW_OFFSET;')
    source=source.replace('if(used>=SOA_POOL_CAP[pool])return soa_fail(heap,ERROR_NODE_CAPACITY);',
        'if(used>=SOA_POOL_CAP[pool]){s->failed_pool=pool;return soa_fail(heap,ERROR_NODE_CAPACITY);}')
    numeric='__device__ inline int numeric_binary(Context* c,int a,int b,int op){'
    source=source.replace(numeric,'__device__ inline int soa_method_repeat(Context* c,int owner,long long times);\n'+numeric,1)
    marker='__device__ inline int device_list_pop_at(Context* c, int list_ref, int index, int* out_val) {'
    if source.count(marker)!=1:
        raise ValueError('Canonical list method boundary changed')
    source=source.replace(marker,_METHODS+'\n'+marker,1)
    clone='__device__ inline int soa_clone_gid(Heap* heap,int global_ref){'
    source=source.replace(clone,_IMMUTABLE+'\n'+clone+'''
 if(soa_tuple_immutable(heap,global_ref,0)){
  heap->soa->clone_map[(long long)(global_ref-1)*heap->soa->rows+heap->soa->row]=global_ref;
  return global_ref;
 }
''',1)
    deepcopy='__device__ inline int native_deepcopy(Context* c,int root){'
    if deepcopy in source:
        # Atomic roots already return before this point. A typed root owns its
        # reset in soa_clone; only a nonempty transient graph can contain typed
        # children that need a fresh cross-view memo for this copy invocation.
        copy_marker=' int tail=0;\n if(++c->deepcopy_epoch==0)'
        if source.count(copy_marker)!=1:
            raise ValueError('Canonical non-atomic deepcopy boundary changed')
        source=source.replace(copy_marker,'''
 if(root_node.count>0 && soa_ready(&c->h))for(int i=0;i<SOA_TOTAL_OBJECTS;i++)
  c->h.soa->clone_map[(long long)i*c->h.soa->rows+c->h.soa->row]=0;
'''+copy_marker,1)
    for declaration, body in (
        (marker,'return soa_method_pop(c,list_ref,index,out_val);'),
        ('__device__ inline int device_list_insert(Context* c, int list_ref, int index, int item_ref) {',
         'return soa_method_insert(c,list_ref,index,item_ref);'),
        ('__device__ inline int device_list_remove(Context* c, int list_ref, int item_ref) {',
         'return soa_method_remove(c,list_ref,item_ref);'),
        ('__device__ inline int device_list_reverse(Context* c, int list_ref) {',
         'return soa_method_reverse(c,list_ref);')):
        if source.count(declaration)!=1:
            raise ValueError('Canonical list mutation boundary changed')
        source=source.replace(declaration,declaration+'\n if(soa_is_view(&c->h,list_ref)){'+body+'}',1)
    # Arrays passed through the call/frame ABI must outlive the conditional
    # that initializes them; later dispatch still reads those argument refs.
    source=source.replace('if(soa_ready(&c->h)&&args&&nargs>0){int promoted[VM_STACK];',
        'int promoted[VM_STACK];if(soa_ready(&c->h)&&args&&nargs>0){')
    call='__device__ inline void call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names=MISSING,const int* kw_values=0,int nkw=0){'
    source=source.replace(call,call+'\n if(nargs<0||nargs>VM_STACK){c->error=BAD_CALL;return;}',1)
    frame='int kw_names=MISSING,const int* kw_values=0,int nkw=0,int kwdefaults_override=MISSING,int pending_root=MISSING){'
    source=source.replace(frame,frame+'\n if(nargs<0||nargs>VM_STACK){c->error=BAD_CALL;return -1;}',1)
    source=source.replace('int narg=nargs<VM_STACK?nargs:VM_STACK;','int narg=nargs;')
    # A recognizable, representable Game cannot silently continue in the
    # generic language heap after typed promotion fails (e.g. pool overflow).
    source=source.replace(' return soa_scratch_fits(heap,&n,ref,SOA_ROOT);', ' return 1;')
    source=source.replace(' int saved=heap->error;\n int promoted=soa_promote(heap,ref,SOA_ROOT);\n if(promoted<0){heap->error=saved;return ref;}\n return promoted;',
        ' return soa_promote(heap,ref,SOA_ROOT);')
    source=source.replace('int used_save[64];if(SOA_POOL_COUNT>64)return soa_fail(heap,ERROR_STACK_OVERFLOW);',
                          'int used_save[SOA_POOL_COUNT];')
    # Retain the canonical RNG draw order; only list storage access changes.
    source=source.replace('if (offset < 0 || offset + n > c->h.counts[1]) {',
        'if (!soa_is_view(&c->h,list_ref) && (offset < 0 || offset + n > c->h.counts[1])) {')
    swap='''int tmp = c->h.edges[(offset + i) * 2 + 0];
                        c->h.edges[(offset + i) * 2 + 0] = c->h.edges[(offset + (int)j) * 2 + 0];
                        c->h.edges[(offset + (int)j) * 2 + 0] = tmp;'''
    source=source.replace(swap,'''if(soa_is_view(&c->h,list_ref)){
                         int a=edge(c,list_ref,i),b=edge(c,list_ref,(int)j);
                         if(soa_list_set(&c->h,list_ref,i,b)<0||soa_list_set(&c->h,list_ref,(int)j,a)<0)return 1;
                        }else{'''+swap+'}')
    # Binary +/* create new sequences; only in-place operations mutate lists.
    start='   if(soa_is_view(&c->h,seq)){'
    if start in source:
        pos=source.index(start)
        end=source.index('   long long repeat=0;if(numeric_int(c,times,&repeat)<0)return MISSING;',pos)
        source=source[:pos]+'''   if(soa_is_view(&c->h,seq)&&op==OP_INPLACE_MULTIPLY&&kind(c,seq)==TAG_LIST){
    long long n=0;if(numeric_int(c,times,&n)<0)return MISSING;return soa_method_repeat(c,seq,n);
   }
'''+source[end:]
    block='''  if((soa_is_view(&c->h,a)||soa_is_view(&c->h,b))&&(op==OP_INPLACE_ADD||op==OP_BINARY_ADD)){
   if(soa_is_view(&c->h,a)&&ta==TAG_LIST)return soa_list_concat(&c->h,a,b);
   c->error=UNSUPPORTED_TYPE;return MISSING;
  }
'''
    source=source.replace(block,'')
    source=source.replace('    if(soa_is_view(&c->h,owner)){c->h.error=ERROR_TYPE_MISMATCH;return 0;}','')
    source=source.replace('    if(soa_is_view(&c->h,owner)&&kind(c,key)==T_SLICE){c->h.error=ERROR_TYPE_MISMATCH;return 0;}','')
    marker='''    c->h.counts[1]+=(int)size;
    c->h.nodes[owner*4+1]=offset;'''
    replacement='''    c->h.counts[1]+=(int)size;
    if(soa_is_view(&c->h,owner)){
     int global=soa_global_ref(&c->h,owner),pool=0,slot=0;
     if(!soa_locate(global,&pool,&slot)||size>SOA_POOL_MAX_ITEMS[pool]){c->h.error=ERROR_EDGE_CAPACITY;return 0;}
     soa_store_i32(c->h.soa,SOA_POOL_COUNT_COL[pool]+slot,(int)size);c->h.nodes[owner*4+2]=(int)size;
     for(int i=0;i<(int)size;i++)if(soa_list_set(&c->h,owner,i,c->h.edges[(offset+i)*2])<0)return 0;
     return 1;
    }
    c->h.nodes[owner*4+1]=offset;'''
    if marker not in source:
        raise ValueError('Canonical slice assignment boundary changed')
    source=source.replace(marker,replacement,1)
    return source
