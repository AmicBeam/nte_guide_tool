"""Device deepcopy for the codec's exact scalar, container and Entity tags.

Insert the emitted source after Context/node/edge and before call(). The caller
must provide zero-initialized gc_marks with max_nodes + constant_node_count
int64 entries per row, and gc_queue with max_nodes entries. Memo epochs
distinguish repeated copies within one VM invocation. This primitive does not
invoke Python or user __deepcopy__ hooks.

Traversal is iterative, so nesting consumes the declared graph/work budget
rather than a fixed recursive stack. Cyclic mutable containers are supported
inside the VM; the final state codec may still reject their export. Pure tuple
subgraphs retain their original identity, including nested/shared tuples.
"""

SUPPORTED_DEEPCOPY_NAMES = ('copy.deepcopy',)


def emit_deepcopy_source(intrinsic_names) -> str:
    names = tuple(intrinsic_names)
    native_id = names.index('copy.deepcopy') if 'copy.deepcopy' in names else -1
    return _SOURCE.replace('DEEPCOPY_NATIVE_ID', str(native_id))


_SOURCE = r'''
// MISSING is never a legal reference. Reserve its neighbour for the temporary
// "needs copying" state; the checked memo size excludes that constant ref.
const int DEEPCOPY_CHANGED = (-2147483647);
const int NATIVE_DEEPCOPY = DEEPCOPY_NATIVE_ID;

__device__ inline int deepcopy_tick(Context* c){
 if(c->error || c->h.error)return 0;
 if(c->steps>=c->step_limit){c->error=STEP_LIMIT;return 0;}
 ++c->steps;
 if(c->deadline && clock_ns()>=(unsigned long long)c->deadline){c->error=DEADLINE;return 0;}
 return 1;
}

__device__ inline int deepcopy_key(Context* c,int ref,int old_nodes){
 long long key=ref;
 if(ref<0){
  long long id=-1LL-ref;
  if(id>=c->h.constant_node_count){c->h.error=ERROR_INVALID_REF;return -1;}
  key=(long long)old_nodes+id;
 }else if(ref>=old_nodes){c->h.error=ERROR_INVALID_REF;return -1;}
 return (int)key;
}

__device__ inline int deepcopy_source_node(Context* c,int ref,int old_nodes,
 int old_edges,int old_payload,NodeView* n){
 if(deepcopy_key(c,ref,old_nodes)<0)return 0;
 if(!get_node(&c->h,ref,n)){c->h.error=ERROR_INVALID_REF;return 0;}
 if(n->tag<0 || n->tag>TAG_CARD_ENTITY){c->error=UNSUPPORTED_TYPE;return 0;}
 int bound=n->is_constant?c->h.constant_payload_count:old_payload;
 if(n->tag==TAG_NONE || n->tag==TAG_BOOL){
  if(n->offset!=0 || n->count!=0 ||
   (n->tag==TAG_NONE?n->extra!=0:(n->extra!=0 && n->extra!=1))){
   c->h.error=ERROR_TYPE_MISMATCH;return 0;
  }
 }else if(n->tag<=TAG_STR){
  if(n->offset<0 || n->count<0 || (long long)n->offset+n->count>bound ||
   (n->tag==TAG_INT && n->count<1) || (n->tag==TAG_FLOAT && n->count!=8)){
   c->h.error=ERROR_INVALID_REF;return 0;
  }
 }else{
  bound=n->is_constant?c->h.constant_edge_count:old_edges;
  if(n->offset<0 || n->count<0 || (long long)n->offset+n->count>bound){
   c->h.error=ERROR_INVALID_REF;return 0;
  }
 }
 return 1;
}

__device__ inline int deepcopy_source_edge(Context* c,const NodeView* n,int i,int side){
 // source_node validated the complete range against the frozen input counts.
 const int* edges=n->is_constant?c->h.constant_edges:c->h.edges;
 return edges[((long long)n->offset+i)*2+side];
}

// High bits identify one deepcopy invocation; low bits retain the exact signed ref.
// GC writes only 0/1, which carry epoch zero and cannot match an active copy.
__device__ inline int deepcopy_memo_get(Context* c,int key){
 unsigned long long packed=(unsigned long long)c->gc_marks[key];
 return (unsigned int)(packed>>32)==c->deepcopy_epoch?(int)(unsigned int)packed:MISSING;
}
__device__ inline void deepcopy_memo_set(Context* c,int key,int value){
 c->gc_marks[key]=(long long)(((unsigned long long)c->deepcopy_epoch<<32)|(unsigned int)value);
}

__device__ inline int deepcopy_discover(Context* c,int ref,int old_nodes,
 int old_edges,int old_payload,int* tail){
 if(!deepcopy_tick(c))return 0;
 int key=deepcopy_key(c,ref,old_nodes);if(key<0)return 0;
 if(deepcopy_memo_get(c,key)!=MISSING)return 1;
 NodeView n;if(!deepcopy_source_node(c,ref,old_nodes,old_edges,old_payload,&n))return 0;
 // Atomic immutable objects retain identity and their exact payload bytes.
 if(n.tag<=TAG_STR){deepcopy_memo_set(c,key,ref);return 1;}
 if(*tail>=c->h.max_nodes){c->error=STACK_OVERFLOW;return 0;}
 deepcopy_memo_set(c,key,n.tag==TAG_TUPLE?ref:DEEPCOPY_CHANGED);
 c->gc_queue[(*tail)++]=ref;
 return 1;
}

__device__ inline int deepcopy_mapped(Context* c,int ref,int old_nodes){
 int key=deepcopy_key(c,ref,old_nodes);if(key<0)return MISSING;
 int result=deepcopy_memo_get(c,key);
 if(result==MISSING || result==DEEPCOPY_CHANGED){c->h.error=ERROR_INVALID_REF;return MISSING;}
 return result;
}

__device__ inline int native_deepcopy(Context* c,int root){
 if(c->error || c->h.error)return MISSING;
 const int old_nodes=c->h.counts[0],old_edges=c->h.counts[1],old_payload=c->h.counts[2];
 const long long memo_size=(long long)old_nodes+c->h.constant_node_count;
 if(old_nodes<0 || old_nodes>c->h.max_nodes || old_edges<0 || old_edges>c->h.max_edges ||
  old_payload<0 || old_payload>c->h.max_payload || c->h.constant_node_count<0 ||
  c->h.constant_edge_count<0 || c->h.constant_payload_count<0 ||
  memo_size>2147483646LL || !c->gc_marks || !c->gc_queue){
  c->error=BAD_PROGRAM;return MISSING;
 }
 // Atomic immutables need no memo. Public projections copy hundreds of these;
 // clearing the entire heap memo for each scalar made work grow with unrelated objects.
 NodeView root_node;
 if(!deepcopy_source_node(c,root,old_nodes,old_edges,old_payload,&root_node))return MISSING;
 if(root_node.tag<=TAG_STR)return deepcopy_tick(c)?root:MISSING;
 int tail=0;
 if(++c->deepcopy_epoch==0){c->error=BAD_PROGRAM;return MISSING;}
 if(!deepcopy_discover(c,root,old_nodes,old_edges,old_payload,&tail))return MISSING;

 // Phase one: visit every original object exactly once, including dict keys.
 // No allocations occur until the full reachable graph has been validated.
 for(int head=0;head<tail;head++){
  if(!deepcopy_tick(c))return MISSING;
  int ref=c->gc_queue[head];NodeView n;
  if(!deepcopy_source_node(c,ref,old_nodes,old_edges,old_payload,&n))return MISSING;
  for(int i=0;i<n.count;i++){
   if(!deepcopy_discover(c,deepcopy_source_edge(c,&n,i,0),old_nodes,old_edges,old_payload,&tail))return MISSING;
   if(n.tag>=TAG_DICT &&
    !deepcopy_discover(c,deepcopy_source_edge(c,&n,i,1),old_nodes,old_edges,old_payload,&tail))return MISSING;
  }
 }

 // Python returns the original tuple iff all of its deepcopied items retain
 // identity. Propagate mutable descendants to a fixed point, with no depth
 // cap and no allocation for pure tuples. Reverse order resolves most graphs
 // in one pass; shared children can require additional bounded passes.
 int changed=1;
 while(changed){
  changed=0;
  for(int pos=tail-1;pos>=0;pos--){
   if(!deepcopy_tick(c))return MISSING;
   int ref=c->gc_queue[pos],key=deepcopy_key(c,ref,old_nodes);NodeView n;
   if(key<0 || !deepcopy_source_node(c,ref,old_nodes,old_edges,old_payload,&n))return MISSING;
   if(n.tag!=TAG_TUPLE || deepcopy_memo_get(c,key)==DEEPCOPY_CHANGED)continue;
   for(int i=0;i<n.count;i++){
    if(!deepcopy_tick(c))return MISSING;
    int child=deepcopy_source_edge(c,&n,i,0),child_key=deepcopy_key(c,child,old_nodes);
    if(child_key<0)return MISSING;
    if(deepcopy_memo_get(c,child_key)==MISSING){c->h.error=ERROR_INVALID_REF;return MISSING;}
    if(deepcopy_memo_get(c,child_key)!=child){deepcopy_memo_set(c,key,DEEPCOPY_CHANGED);changed=1;break;}
   }
  }
 }

 // Reserve all destination shells before writing any child reference. The
 // original/constant memo indexing remains based on old_nodes, never on the
 // growing heap node count, so new node IDs cannot alias constant memo keys.
 for(int pos=0;pos<tail;pos++){
  if(!deepcopy_tick(c))break;
  int ref=c->gc_queue[pos],key=deepcopy_key(c,ref,old_nodes);NodeView n;
  if(key<0 || !deepcopy_source_node(c,ref,old_nodes,old_edges,old_payload,&n))break;
  if(deepcopy_memo_get(c,key)!=DEEPCOPY_CHANGED)continue;
  if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;break;}
  if((long long)c->h.counts[1]+n.count>c->h.max_edges){c->h.error=ERROR_EDGE_CAPACITY;break;}
  int dest=c->h.counts[0]++,offset=c->h.counts[1];c->h.counts[1]+=n.count;
  int* out=c->h.nodes+(long long)dest*4;
  out[0]=n.tag;out[1]=offset;out[2]=n.count;out[3]=n.extra;
  deepcopy_memo_set(c,key,dest);
 }

 // Phase two: fill ordered edges using the complete memo, preserving aliases
 // and cycles. Only newly allocated node/edge ranges are ever written.
 for(int pos=0;pos<tail && !c->error && !c->h.error;pos++){
  if(!deepcopy_tick(c))break;
  int ref=c->gc_queue[pos],dest=deepcopy_mapped(c,ref,old_nodes);NodeView n;
  if(c->error || c->h.error || !deepcopy_source_node(c,ref,old_nodes,old_edges,old_payload,&n))break;
  if(dest==ref)continue;
  if(dest<old_nodes || dest>=c->h.counts[0]){c->h.error=ERROR_INVALID_REF;break;}
  int offset=c->h.nodes[(long long)dest*4+1];
  for(int i=0;i<n.count;i++){
   if(!deepcopy_tick(c))break;
   int a=deepcopy_mapped(c,deepcopy_source_edge(c,&n,i,0),old_nodes),b=-1;
   if(n.tag>=TAG_DICT)b=deepcopy_mapped(c,deepcopy_source_edge(c,&n,i,1),old_nodes);
   if(c->error || c->h.error)break;
   c->h.edges[((long long)offset+i)*2]=a;
   c->h.edges[((long long)offset+i)*2+1]=b;
  }
 }
 if(c->error || c->h.error){
  c->h.counts[0]=old_nodes;c->h.counts[1]=old_edges;c->h.counts[2]=old_payload;
  return MISSING;
 }
 int result=deepcopy_mapped(c,root,old_nodes);
 if(c->error || c->h.error){
  c->h.counts[0]=old_nodes;c->h.counts[1]=old_edges;c->h.counts[2]=old_payload;
  return MISSING;
 }
 return result;
}

__device__ inline int deepcopy_call(Context* c,int callee,const int* args,int nargs,int* out){
 const int* n=node(c,callee);
 if(!n)return 0;
 if(NATIVE_DEEPCOPY<0 || n[0]!=T_NATIVE || n[3]!=NATIVE_DEEPCOPY)return 0;
 if((nargs!=1 && nargs!=2) || !args || !out){c->error=BAD_CALL;return 1;}
 // An explicit memo affects Python object identity across independent calls.
 // Only the default None is supported; never silently ignore a caller memo.
 if(nargs==2){
  int tag=kind(c,args[1]);
  if(c->error || c->h.error)return 1;
  if(tag!=TAG_NONE){c->error=UNSUPPORTED_TYPE;return 1;}
 }
 int result=native_deepcopy(c,args[0]);
 if(!c->error && !c->h.error)*out=result;
 return 1;
}
'''
