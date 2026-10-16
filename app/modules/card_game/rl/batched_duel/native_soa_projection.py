"""Export only canonical public projections from SoA views to transient graphs.

The numeric public encoder and action leases use their existing graph ABI.
Authoritative Game records are explicitly refused by this exporter, including
when nested in a packet. Rule execution always retains typed field storage.
"""
from .vm_heap import emit_vm_heap_source


def emit_soa_metadata_source():
    from .device_rule_results import _RESULT_SOURCE
    start = _RESULT_SOURCE.index('extern "C" __global__ void decode_metadata(')
    end = _RESULT_SOURCE.index('extern "C" __global__ void decode_pair(', start)
    prefix, kernel = _RESULT_SOURCE[:start], _RESULT_SOURCE[start:end]
    kernel = kernel.replace('void decode_metadata(', 'void soa_decode_metadata(', 1)
    opening = kernel.index('{')
    extra = '''long long* si64,double* sf64,int* si32,unsigned char* su8,
 int* views,int* value_cache,int* clone_map,int* clone_queue,int* constant_adopt,
 const int* typed_roots'''
    kernel = kernel[:opening].rstrip()[:-1]+','+extra+')'+kernel[opening:]
    attach_after = 'payload+(long long)row*maxp,counts+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);'
    if kernel.count(attach_after) != 1:
        raise ValueError('Metadata scalar decoder attachment changed')
    kernel = kernel.replace(attach_after, attach_after+'''
 DomainSoAContext domain;
 domain.i64=si64;domain.f64=sf64;domain.i32=si32;domain.u8=su8;
 domain.rows=rows;domain.row=row;domain.views=views;domain.value_cache=value_cache;
 domain.clone_map=clone_map;domain.clone_queue=clone_queue;domain.constant_adopt=constant_adopt;domain.constant_node_count=cnc;
 soa_attach(&h,&domain);int game_ref=soa_view_ref(&h,typed_roots[row]);
''', 1)
    kernel = kernel.replace('field_ref(&h,roots[row],', 'field_ref(&h,game_ref,')
    other = _RESULT_SOURCE[end:].replace('void decode_pair(', 'void soa_decode_pair(', 1).replace(
        'void decode_float(', 'void soa_decode_float(', 1)
    return '\nnamespace soa_metadata_result {\n'+prefix+kernel+other+'\n}\n'


def emit_soa_projection_source(schema):
    heap = emit_vm_heap_source()
    start = heap.index('__device__ inline int compact_index(')
    end = heap.index('} // namespace duel_vm_heap', start)
    helpers = heap[start:end]
    helpers = helpers.replace('compact_index', 'soa_export_index').replace(
        'compact_enqueue', 'soa_export_enqueue').replace('compact_into', 'soa_export_into')
    helpers = helpers.replace('source->counts[0]+(int)constant', 'source->max_nodes+(int)constant')
    helpers = helpers.replace('long long total=(long long)source->counts[0]+source->constant_node_count;',
        'long long total=(long long)source->max_nodes+source->constant_node_count;')
    helpers = helpers.replace('if(n.tag<TAG_NONE || n.tag>TAG_CARD_ENTITY)',
        'if(soa_is_view(source,old) && soa_global_ref(source,old)>'+str(schema.pools[schema.root].base)+
        ' && soa_global_ref(source,old)<='+str(schema.pools[schema.root].base+schema.pools[schema.root].capacity)+
        '){target->error=ERROR_TYPE_MISMATCH;break;}\n        if(n.tag<TAG_NONE || n.tag>TAG_CARD_ENTITY)')
    helpers = helpers.replace('n.tag==TAG_LIST?0:n.extra',
                              '(n.tag==TAG_LIST || n.offset==SOA_VIEW_OFFSET)?0:n.extra')
    return '\nnamespace duel_vm_heap {\n'+helpers+'\n}\n'+r'''
extern "C" __global__ void soa_export_projection(
 int* sn,int* se,unsigned char* sp,int* sc,const int* roots,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 long long* si64,double* sf64,int* si32,unsigned char* su8,
 int* views,int* value_cache,int* clone_map,int* clone_queue,int* constant_adopt,
 int* dn,int* de,unsigned char* dp,int* dc,int* dr,int* mapping,int* queue,
 int* errors,const unsigned char* active,int rows,int maxn,int maxe,int maxp,
 int destn,int deste,int destp){
 using namespace duel_vm_heap;
 int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(row>=rows)return;
 errors[row]=0;dr[row]=-1;dc[row*3]=dc[row*3+1]=dc[row*3+2]=0;
 if(!active[row])return;
 Heap source,target;
 init_heap(&source,sn+(long long)row*maxn*4,se+(long long)row*maxe*2,
  sp+(long long)row*maxp,sc+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 init_heap(&target,dn+(long long)row*destn*4,de+(long long)row*deste*2,
  dp+(long long)row*destp,dc+row*3,destn,deste,destp,0,0,0,0,0,0);
 DomainSoAContext domain;
 domain.i64=si64;domain.f64=sf64;domain.i32=si32;domain.u8=su8;
 domain.rows=rows;domain.row=row;domain.views=views;domain.value_cache=value_cache;
 domain.clone_map=clone_map;domain.clone_queue=clone_queue;domain.constant_adopt=constant_adopt;domain.constant_node_count=cnc;
 soa_attach(&source,&domain);
 int capacity=maxn+cnc;
 dr[row]=soa_export_into(&source,&target,roots[row],mapping+(long long)row*capacity,
  queue+(long long)row*capacity,capacity);
 errors[row]=source.error?source.error:target.error;
}
'''+emit_soa_select_source()+emit_soa_metadata_source()


def emit_soa_select_source():
    return r'''
extern "C" __global__ void soa_select_result(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* outputs,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 long long* si64,double* sf64,int* si32,unsigned char* su8,
 int* views,int* value_cache,int* clone_map,int* clone_queue,int* constant_adopt,
 const int* path,int depth,int* selected,int* typed,int* errors,const unsigned char* active,
 int rows,int maxn,int maxe,int maxp){
 using namespace duel_vm_heap;
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 selected[row]=(-2147483647-1);typed[row]=0;errors[row]=0;
 if(!active[row])return;
 if(depth<0||depth>8){errors[row]=ERROR_INDEX_OUT_OF_BOUNDS;return;}
 Heap h;
 init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,counts+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 DomainSoAContext domain;
 domain.i64=si64;domain.f64=sf64;domain.i32=si32;domain.u8=su8;
 domain.rows=rows;domain.row=row;domain.views=views;domain.value_cache=value_cache;
 domain.clone_map=clone_map;domain.clone_queue=clone_queue;
 domain.constant_adopt=constant_adopt;domain.constant_node_count=cnc;
 soa_attach(&h,&domain);
 int ref=outputs[row];
 for(int i=0;i<depth;i++){
  NodeView n;int key=0,value=0;
  if(!get_node(&h,ref,&n)||(n.tag!=TAG_LIST&&n.tag!=TAG_TUPLE)
     ||path[i]<0||path[i]>=n.count||!get_edge(&h,&n,path[i],&key,&value)){
   errors[row]=h.error?h.error:ERROR_INDEX_OUT_OF_BOUNDS;return;
  }
  ref=key;
 }
 selected[row]=ref;
 if(soa_is_view(&h,ref))typed[row]=soa_global_ref(&h,ref);
 errors[row]=h.error;
}
'''
