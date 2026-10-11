"""Actual CUDA interpreter for frozen rule IR, with explicit capability gates.

The current instruction set is a checked execution subset. A missing capability
raises before launch; this module never calls the Python engine as a fallback.
Full duel simulation is NOT available until all simulate/observe/legality
capabilities and formal-rule GPU parity tests pass.
"""
from dataclasses import dataclass
import time
import os
import numpy as np
from .vm_image import VmImage
from .vm_heap import emit_vm_heap_source
from .vm_collections import emit_collection_source, SUPPORTED_COLLECTION_OPS
from .vm_objects import emit_object_source, OBJECT_OPS
from .vm_builtins import emit_builtin_source,SUPPORTED_BUILTIN_NAMES
from .vm_numeric import emit_numeric_source,NUMERIC_OPS
from .vm_reducers import emit_reducer_source,SUPPORTED_REDUCER_NAMES
from .vm_frame_gc import emit_frame_gc_source
from .vm_random import emit_random_vm_source,SUPPORTED_RANDOM_NAMES
from .random_source import emit_random_source
from .vm_exceptions import emit_exception_source,EXCEPTION_NAMES,EXCEPTION_OPS
from .vm_strings import emit_string_source,STRING_OPS,STRING_INTRINSICS
from .vm_ryu_source import emit_ryu_source
from .vm_deepcopy import emit_deepcopy_source,SUPPORTED_DEEPCOPY_NAMES
from .vm_text_methods import emit_text_method_source
from .vm_decimal import emit_decimal_source,SUPPORTED_DECIMAL_NAMES
from .vm_context import emit_context_source,SUPPORTED_CONTEXT_NAMES
from .vm_stdlib import emit_stdlib_source,SUPPORTED_STDLIB_NAMES
from .vm_numpy import emit_numpy_source,SUPPORTED_NUMPY_NAMES
from .vm_json import emit_json_source,SUPPORTED_JSON_NAMES

SUPPORTED_OPS=frozenset(('NOP','EXTENDED_ARG','LOAD_CONST','LOAD_FAST','STORE_FAST','LOAD_GLOBAL',
    'DELETE_FAST','LOAD_DEREF','STORE_DEREF','LOAD_CLOSURE','BINARY_SUBSCR','COMPARE_OP','CONTAINS_OP','IS_OP',
    'UNARY_NOT','UNARY_NEGATIVE','BINARY_ADD','BINARY_SUBTRACT','BINARY_MULTIPLY',
    'BUILD_TUPLE','BUILD_LIST','MAKE_FUNCTION','GET_ITER','FOR_ITER','GEN_START','YIELD_VALUE',
    'CALL_FUNCTION','CALL_FUNCTION_KW','CALL_FUNCTION_EX','POP_TOP','DUP_TOP','ROT_TWO','JUMP_ABSOLUTE','JUMP_FORWARD',
    'POP_JUMP_IF_FALSE','POP_JUMP_IF_TRUE','JUMP_IF_FALSE_OR_POP','JUMP_IF_TRUE_OR_POP','RETURN_VALUE',
    'LOAD_ASSERTION_ERROR','SETUP_WITH','WITH_EXCEPT_START')) | SUPPORTED_COLLECTION_OPS | OBJECT_OPS | NUMERIC_OPS | EXCEPTION_OPS | STRING_OPS
SUPPORTED_INTRINSICS=frozenset(('next','object',*SUPPORTED_BUILTIN_NAMES,*SUPPORTED_REDUCER_NAMES,*SUPPORTED_RANDOM_NAMES,*EXCEPTION_NAMES,*STRING_INTRINSICS,*SUPPORTED_DEEPCOPY_NAMES,*SUPPORTED_DECIMAL_NAMES,*SUPPORTED_CONTEXT_NAMES,*SUPPORTED_STDLIB_NAMES,*SUPPORTED_NUMPY_NAMES,*SUPPORTED_JSON_NAMES))
DENIED_STANDALONE_INTRINSICS=frozenset(('uuid.uuid4',))


class VmCapabilityError(RuntimeError):pass
class VmExecutionError(RuntimeError):pass


@dataclass(frozen=True)
class VmLimits:
    frames:int=64
    locals:int=128
    stack:int=256
    cells:int=64
    steps:int=2_000_000
    def __post_init__(self):
        for name in ('frames','locals','stack','cells','steps'):
            value=getattr(self,name)
            if type(value) is not int or value<1 or value>2**30:raise ValueError('Positive bounded VM limits required')


def emit_vm_source(image,limits=VmLimits(),*,compiled_blocks=None):
    image.verify()
    names=set(image.opcode_names)|set(SUPPORTED_OPS)
    ids={n:i for i,n in enumerate(image.opcode_names)}
    missing=sorted(names-set(ids))
    ids.update({n:-1000-i for i,n in enumerate(missing)})
    enum='enum Op { '+','.join('OP_'+n+'='+str(ids[n]) for n in sorted(names))+' };\n'
    prefix=f'''\nnamespace duel_rule_vm {{
using namespace duel_vm_heap;
{enum}
const int VM_FRAMES={limits.frames},VM_LOCALS={limits.locals},VM_STACK={limits.stack},VM_CELLS={limits.cells};
const int HEADER=14, LBASE=HEADER, SBASE=LBASE+VM_LOCALS, CBASE=SBASE+VM_STACK, STRIDE=CBASE+VM_CELLS;
const int MISSING=(-2147483647-1), NIL=-1, FALSE_REF=-2, TRUE_REF=-3;
const int T_FUNCTION=11,T_CODE=12,T_CELL=16,T_ITER=17,T_GEN=19,T_DYNFN=33,T_NATIVE=15;
const int NATIVE_NEXT={image.intrinsic_names.index('next') if 'next' in image.intrinsic_names else -1};
const int NATIVE_OBJECT={image.intrinsic_names.index('object') if 'object' in image.intrinsic_names else -2};
enum VmError {{ BAD_PROGRAM=100,STACK_OVERFLOW=101,FRAME_OVERFLOW=102,UNBOUND_LOCAL=103,
 BAD_CALL=104,UNSUPPORTED_OP=105,UNSUPPORTED_TYPE=106,STEP_LIMIT=107,DEADLINE=108,STOP_ITERATION=109 }};
'''
    compiled_plan = None
    if compiled_blocks is not None:
        if not isinstance(compiled_blocks, dict):
            raise TypeError('Explicit compiled-block budget mapping required')
        from .vm_compiled_blocks import emit_compiled_blocks
        compiled_plan = emit_compiled_blocks(image, **compiled_blocks)
    source=_VM_SOURCE
    if compiled_plan is not None:
        declaration='__device__ inline void execute(Context* c){'
        source=source.replace(declaration,compiled_plan.source+'\n'+declaration,1)
        hook='  if(++c->steps>c->step_limit){c->error=STEP_LIMIT;break;}'
        source=source.replace(hook,'  if(run_compiled_block(c))continue;\n'+hook,1)

    device_kernel=source[source.index('extern "C" __global__ void vm_execute('):]
    device_kernel=device_kernel.replace('void vm_execute(', 'void vm_execute_device(', 1)
    device_kernel=device_kernel.replace('int* gc_queue){', 'int* gc_queue,const unsigned char* active){', 1)
    device_kernel=device_kernel.replace('if(tid>=rows)return;', 'if(tid>=rows)return;\n if(!active[tid]||deadlines[tid]<0){outputs[tid]=args[(long long)tid*nargs];statuses[tid*4]=active[tid]?108:0;statuses[tid*4+1]=entry;statuses[tid*4+2]=-1;statuses[tid*4+3]=0;return;}', 1)
    source+=device_kernel
    compact_kernel=_VM_SOURCE[_VM_SOURCE.index('extern "C" __global__ void vm_compact('):_VM_SOURCE.index('extern "C" __global__ void vm_execute(')]
    compact_kernel=compact_kernel.replace('void vm_compact(', 'void vm_compact_device(', 1)
    compact_kernel=compact_kernel.replace('int maxp,int capacity){',
        'int maxp,int capacity,int destn,int deste,int destp,const unsigned char* active,const int* call_status){',1)
    compact_kernel=compact_kernel.replace('if(row>=rows)return;',
        'if(row>=rows)return; status[row]=0; if(!active[row])return; '
        'if(call_status[row*4]){status[row]=call_status[row*4];return;} '
        'dc[row*3]=0;dc[row*3+1]=0;dc[row*3+2]=0;',1)
    compact_kernel=compact_kernel.replace('dn+(long long)row*maxn*4,de+(long long)row*maxe*2,dp+(long long)row*maxp,\n  dc+row*3,maxn,maxe,maxp',
        'dn+(long long)row*destn*4,de+(long long)row*deste*2,dp+(long long)row*destp,\n  dc+row*3,destn,deste,destp',1)
    source+=compact_kernel
    source+=_GRAPH_TRANSFER_SOURCE
    return '#define BMT_DEVICE __device__\n'+emit_random_source()+emit_ryu_source()+emit_vm_heap_source()+prefix+source.replace('// COLLECTION_HANDLERS',emit_collection_source()).replace('// OBJECT_HANDLERS',emit_object_source(image)).replace('// BUILTIN_HANDLERS',emit_builtin_source(image.intrinsic_names)).replace('// NUMERIC_HANDLERS',emit_numeric_source()).replace('// REDUCER_HANDLERS',emit_reducer_source(image.intrinsic_names)).replace('// FRAME_GC_HANDLERS',emit_frame_gc_source()).replace('// RANDOM_HANDLERS',emit_random_vm_source(image.intrinsic_names)).replace('// EXCEPTION_HANDLERS',emit_exception_source(image)).replace('// STRING_HANDLERS',emit_string_source(image.intrinsic_names)).replace('// DEEPCOPY_HANDLERS',emit_deepcopy_source(image.intrinsic_names)).replace('// TEXT_HANDLERS',emit_text_method_source()).replace('// DECIMAL_HANDLERS',emit_decimal_source(image.intrinsic_names)).replace('// CONTEXT_HANDLERS',emit_context_source(image)).replace('// STDLIB_HANDLERS','// STDLIB_HANDLERS\n'+emit_stdlib_source(image)).replace('// NUMPY_HANDLERS','// NUMPY_HANDLERS\n'+emit_numpy_source(image.intrinsic_names)+emit_json_source(image.intrinsic_names))


_VM_SOURCE=r'''
// Frame: function,pc,sp,caller,status,return_kind,return_target,default,globals,closure,generator,cells_count.
struct Context {
 Heap h; const int* classes;int class_count;const int* modules;int module_count; const int* instructions; const int* functions; int function_count; int* frames;
 long long* gc_marks;int* gc_queue;unsigned int deepcopy_epoch;int raised;int error_value;int error_kind;int current; int output; int error; int steps; int step_limit; long long deadline; int context_values;
};
__device__ inline unsigned long long clock_ns(){unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));return value;}
__device__ inline int* frame(Context* c,int id){return c->frames+id*STRIDE;}
__device__ inline const int* function(Context* c,int id){
 if(id<0 || id>=c->function_count){c->error=BAD_PROGRAM;return 0;}return c->functions+id*18;
}
__device__ inline int* mutable_node(Context* c,int ref){
 if(ref<0 || ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}return c->h.nodes+ref*4;
}
__device__ inline const int* node(Context* c,int ref){
 if(ref>=0){if(ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}return c->h.nodes+ref*4;}
 long long id=-1LL-ref;if(id<0 || id>=c->h.constant_node_count){c->error=BAD_PROGRAM;return 0;}
 return c->h.constant_nodes+id*4;
}
__device__ inline int kind(Context* c,int ref){const int* n=node(c,ref);return n?n[0]:-1;}
__device__ inline int edge(Context* c,int container,int index,int value=0){
 const int* n=node(c,container);if(!n || index<0 || index>=n[2]){c->error=BAD_PROGRAM;return MISSING;}
 long long slot=(long long)n[1]+index;
 if(container>=0){if(slot<0 || slot>=c->h.counts[1]){c->error=BAD_PROGRAM;return MISSING;}return c->h.edges[slot*2+value];}
 if(slot<0 || slot>=c->h.constant_edge_count){c->error=BAD_PROGRAM;return MISSING;}return c->h.constant_edges[slot*2+value];
}
__device__ inline long long range_size(Context* c,int ref);
__device__ inline int numpy_length(Context* c,int ref,int* out);
__device__ inline int numpy_truth(Context* c,int ref,int* out);
__device__ inline int numpy_iterator(Context* c,int ref);
__device__ inline int numpy_iterator_next(Context* c,int ref);
__device__ inline int numpy_numeric(Context* c,int a,int b,int op,int* out);
__device__ inline int numpy_gc_trace(Context* c,int ref,int* tail);
__device__ inline int numpy_try_subscript(Context* c,int owner,int key,int* out);
__device__ inline int numpy_try_store(Context* c,int owner,int key,int value);
__device__ inline int length(Context* c,int ref){
 int np_length=0;if(numpy_length(c,ref,&np_length))return np_length;
 const int* n=node(c,ref);if(!n)return 0;
 if(n[0]==TAG_NONE)return 0;
 if(n[0]==80)return length(c,n[1]);
 if(n[0]==TAG_LIST || n[0]==TAG_TUPLE || n[0]==TAG_DICT || (n[0]>=8&&n[0]<=10) || n[0]==21 || n[0]==22 || n[0]==73)return n[2];
 c->error=UNSUPPORTED_TYPE;return 0;
}
__device__ inline int meta(Context* c,int tag,int offset,int count,int extra){
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 int id=c->h.counts[0]++;int* n=c->h.nodes+id*4;n[0]=tag;n[1]=offset;n[2]=count;n[3]=extra;return id;
}
__device__ inline void push(Context* c,int* f,int value){
 if(f[2]<0 || f[2]>=VM_STACK){c->error=STACK_OVERFLOW;return;}f[SBASE+f[2]++]=value;
}
__device__ inline int pop(Context* c,int* f){
 if(f[2]<=0 || f[2]>VM_STACK){c->error=BAD_PROGRAM;return MISSING;}return f[SBASE+--f[2]];
}
__device__ inline int peek(Context* c,int* f){if(f[2]<=0){c->error=BAD_PROGRAM;return MISSING;}return f[SBASE+f[2]-1];}
__device__ inline int decimal_truth(Context* c,int ref);
__device__ inline int decimal_equal(Context* c,int a,int b,int* out);
__device__ inline int decimal_numeric(Context* c,int ref,int as_float,int* out);
__device__ inline int decimal_string(Context* c,int ref);
__device__ inline int truth_value(Context* c,int value){
 int np_truth=0;if(numpy_truth(c,value,&np_truth))return np_truth;
 int tag=kind(c,value);if(tag==80)return length(c,value)!=0;if(tag==28)return decimal_truth(c,value);if(tag==26)return range_size(c,value)!=0;if(tag==21||tag==22||tag==73)return length(c,value)!=0;if(tag>=11)return 1;return duel_vm_heap::truth(&c->h,value);

}
// FRAME_GC_HANDLERS
__device__ inline int new_frame(Context* c,int fid,const int* args,int nargs,int caller,int return_kind,
 int globals_override=MISSING,int closure_override=MISSING,int defaults_override=MISSING,
 int kw_names=MISSING,const int* kw_values=0,int nkw=0,int kwdefaults_override=MISSING,int pending_root=MISSING){
 const int* fn=function(c,fid);if(!fn)return -1;
 if(fn[5]>VM_LOCALS || fn[6]>VM_STACK){c->error=BAD_CALL;return -1;}
 int id=-1;for(int i=0;i<VM_FRAMES;i++)if(frame(c,i)[4]==0){id=i;break;}
 if(id<0){
  collect_suspended(c,args,nargs,globals_override,closure_override,defaults_override,kw_names,kw_values,nkw,kwdefaults_override,pending_root);
  if(c->error||c->h.error)return -1;
  for(int i=0;i<VM_FRAMES;i++)if(frame(c,i)[4]==0){id=i;break;}
  if(id<0){c->error=FRAME_OVERFLOW;return -1;}
 }
 int* f=frame(c,id);for(int i=0;i<STRIDE;i++)f[i]=MISSING;
 f[0]=fid;f[1]=fn[0];f[2]=0;f[3]=caller;f[4]=1;f[5]=return_kind;f[6]=-1;f[7]=MISSING;
 f[8]=globals_override==MISSING?fn[8]:globals_override;f[9]=closure_override==MISSING?fn[15]:closure_override;f[10]=-1;f[12]=NIL;f[13]=NIL;
 int defaults=defaults_override==MISSING?fn[13]:defaults_override;
 int count_defaults=length(c,defaults);int argc=fn[2],kwonly=fn[4],params=argc+kwonly;
 int varargs=fn[7]&4,varkw=fn[7]&8,extra=params+(varargs?1:0);
 if(params+(varargs?1:0)+(varkw?1:0)>fn[5] || nargs<0 || nkw<0 || (nargs>argc&&!varargs)){c->error=BAD_CALL;return -1;}
 for(int i=0;i<argc && i<nargs;i++)f[LBASE+i]=args[i];
 if(varargs)f[LBASE+params]=make_tuple(&c->h,args+(nargs<argc?nargs:argc),nargs>argc?nargs-argc:0);
 if(varkw)f[LBASE+extra]=make_dict(&c->h);
 if(nkw && (kw_names==MISSING || length(c,kw_names)!=nkw)){c->error=BAD_CALL;return -1;}
 for(int i=0;i<nkw && !c->error && !c->h.error;i++){
  int name=edge(c,kw_names,i),slot=-1;
  if(kind(c,name)!=TAG_STR){c->error=BAD_CALL;break;}
  for(int j=fn[3];j<params;j++)if(key_equal(&c->h,name,edge(c,fn[10],j))){slot=j;break;}
  if(slot>=0){if(f[LBASE+slot]!=MISSING)c->error=BAD_CALL;else f[LBASE+slot]=kw_values[i];}
  else if(varkw){int kwdict=f[LBASE+extra];if(dict_find(&c->h,kwdict,name)>=0)c->error=BAD_CALL;else dict_set(&c->h,kwdict,name,kw_values[i]);}
  else c->error=BAD_CALL;
 }
 for(int i=0;i<argc;i++)if(f[LBASE+i]==MISSING){
  if(i<argc-count_defaults){c->error=BAD_CALL;return -1;}f[LBASE+i]=edge(c,defaults,i-(argc-count_defaults));
 }
 int kwdefaults=kwdefaults_override==MISSING?fn[14]:kwdefaults_override;
 for(int i=argc;i<params;i++)if(f[LBASE+i]==MISSING){
  int name=edge(c,fn[10],i),value=kind(c,kwdefaults)==TAG_DICT?dict_get(&c->h,kwdefaults,name,MISSING):MISSING;
  if(value==MISSING){c->error=BAD_CALL;return -1;}f[LBASE+i]=value;
 }
 if(c->error||c->h.error)return -1;
 int nc=length(c,fn[11]),nf=length(c,fn[12]);f[11]=nc+nf;
 if(nc+nf>VM_CELLS){c->error=STACK_OVERFLOW;return -1;}
 for(int i=0;i<nc;i++){
  int name=edge(c,fn[11],i),local=MISSING;
  for(int j=0;j<fn[5];j++)if(key_equal(&c->h,name,edge(c,fn[10],j))){local=f[LBASE+j];break;}
  f[CBASE+i]=meta(c,T_CELL,local,0,0);
 }
 for(int i=0;i<nf;i++){
  int value=edge(c,f[9],i);
  // Runtime LOAD_CLOSURE shares cells; frozen closures contain their captured values.
  f[CBASE+nc+i]=kind(c,value)==T_CELL?value:meta(c,T_CELL,value,0,0);
 }
 return id;
}
__device__ inline int iterator(Context* c,int owner){
 if(kind(c,owner)==64)return owner;
 if(kind(c,owner)==60)return numpy_iterator(c,owner);
 int tag=kind(c,owner);if(tag==T_ITER || tag==T_GEN)return owner;
 if(tag==70||tag==71||tag==72)return owner;
 if(tag!=26 && tag!=TAG_LIST && tag!=TAG_TUPLE && tag!=TAG_DICT && !(tag>=8&&tag<=10) && tag!=21 && tag!=22 && tag!=73 && tag!=80){c->error=UNSUPPORTED_TYPE;return MISSING;}
 return meta(c,T_ITER,owner,0,0);
}
__device__ inline int iterator_next(Context* c,int ref){
 if(kind(c,ref)==64)return numpy_iterator_next(c,ref);
 int* it=mutable_node(c,ref);if(!it || it[0]!=T_ITER){c->error=BAD_CALL;return MISSING;}
 int owner=it[1],index=it[2];if(kind(c,owner)==80)owner=node(c,owner)[1];const int* n=node(c,owner);if(!n)return MISSING;
 if(n[0]==26){
  long long count=range_size(c,owner);if(index>=count)return MISSING;
  long long start=0,step=0;read_int64(&c->h,edge(c,owner,0),&start);read_int64(&c->h,edge(c,owner,2),&step);
  long long value=(long long)((unsigned long long)start+(unsigned long long)index*(unsigned long long)step);
  if(index==2147483647){c->error=STEP_LIMIT;return MISSING;}it[2]++;return make_int64(&c->h,value);
 }
 if(index>=n[2])return MISSING;it[2]++;return edge(c,owner,index);
}
__device__ inline void resume_generator(Context* c,int ref,int caller,int mode,int target,int fallback){
 int* gen=mutable_node(c,ref);if(!gen || gen[0]!=T_GEN){c->error=BAD_CALL;return;}
 int id=gen[3];
 if(id<0){
  int* parent=frame(c,caller);
  if(mode==2){pop(c,parent);parent[1]=target;}else if(fallback!=MISSING)push(c,parent,fallback);else c->error=STOP_ITERATION;
  return;
 }
 if(id>=VM_FRAMES){c->error=BAD_PROGRAM;return;}int* f=frame(c,id);
 if(f[4]!=2){c->error=BAD_CALL;return;}f[3]=caller;f[5]=mode;f[6]=target;f[7]=fallback;f[4]=1;
 push(c,f,NIL);c->current=id;
}
__device__ inline long long range_size(Context* c,int ref){
 long long start=0,stop=0,step=0;read_int64(&c->h,edge(c,ref,0),&start);read_int64(&c->h,edge(c,ref,1),&stop);read_int64(&c->h,edge(c,ref,2),&step);
 if(!step){c->error=BAD_CALL;return 0;}unsigned long long span=0,increment=0;
 if(step>0){if(start>=stop)return 0;span=(unsigned long long)stop-(unsigned long long)start;increment=(unsigned long long)step;}
 else{if(start<=stop)return 0;span=(unsigned long long)start-(unsigned long long)stop;increment=0ULL-(unsigned long long)step;}
 unsigned long long count=(span-1)/increment+1;if(count>9223372036854775807ULL){c->h.error=ERROR_INT_OVERFLOW;return 0;}return (long long)count;
}
// NUMERIC_HANDLERS
// OBJECT_HANDLERS
// BUILTIN_HANDLERS
// REDUCER_HANDLERS
// RANDOM_HANDLERS
// EXCEPTION_HANDLERS
// STRING_HANDLERS
// DEEPCOPY_HANDLERS
// TEXT_HANDLERS
// DECIMAL_HANDLERS
// CONTEXT_HANDLERS
// STDLIB_HANDLERS
// NUMPY_HANDLERS
__device__ inline void call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names=MISSING,const int* kw_values=0,int nkw=0){
 const int* n=node(c,callable);if(!n)return;
 int bound_args[VM_STACK];
 if(n[0]==29){
  // Python special-method invocation looks on the type, ignoring instance shadows.
  int instance=callable,cls=class_ref(c,instance),category=-1;
  int name=make_utf8(&c->h,"__call__",8);
  int method=cls==MISSING?MISSING:lookup_class(c,node(c,cls)[3],name,&category);
  if(method==MISSING){c->error=BAD_CALL;return;}
  if(category==2)callable=bind_method(c,method,instance);
  else if(category==4)callable=bind_method(c,method,cls);
  else if(category==3)callable=method;
  else{c->error=UNSUPPORTED_TYPE;return;}
  n=node(c,callable);if(!n)return;
 }
 if(n[0]==18){
  if(nargs+1>VM_STACK){c->error=STACK_OVERFLOW;return;}
  bound_args[0]=edge(c,callable,1);for(int i=0;i<nargs;i++)bound_args[i+1]=args[i];
  callable=edge(c,callable,0);args=bound_args;nargs++;n=node(c,callable);if(!n)return;
 }
 if(n[0]==13){
  int cid=n[3];if(!class_constructible(cid)){c->error=UNSUPPORTED_TYPE;return;}
  int instance=MISSING;int etag=class_entity_tag(cid);
  if(etag==8||etag==9||etag==10)instance=make_entity(&c->h,etag);
  else if(class_dict_subclass(cid))instance=make_dict(&c->h);
  else{
   int attrs=make_dict(&c->h),values[2]={callable,attrs};instance=make_tuple(&c->h,values,2);
   int* obj=mutable_node(c,instance);if(!obj)return;obj[0]=29;
  }
  int name=make_utf8(&c->h,"__init__",8),category=-1,init=lookup_class(c,cid,name,&category);
  if(init==MISSING){
   if(class_dict_subclass(cid)&&(nargs||nkw)){
    int bound=stdlib_bind(c,instance,STDLIB_METHOD_DICT_INIT),out=MISSING;
    if(!stdlib_bound_apply(c,bound,args,nargs,kw_names,kw_values,nkw,&out)&&!c->error)c->error=BAD_CALL;
    if(!c->error&&!c->h.error)push(c,frame(c,caller),instance);return;
   }
   if(nargs||nkw)c->error=BAD_CALL;else push(c,frame(c,caller),instance);return;}
  if(category!=2||kind(c,init)!=T_FUNCTION||nargs+1>VM_STACK){c->error=UNSUPPORTED_TYPE;return;}
  int args_with_self[VM_STACK];args_with_self[0]=instance;for(int i=0;i<nargs;i++)args_with_self[i+1]=args[i];
  int id=new_frame(c,node(c,init)[3],args_with_self,nargs+1,caller,4,MISSING,MISSING,MISSING,kw_names,kw_values,nkw);
  if(id>=0){frame(c,id)[6]=instance;c->current=id;}return;
 }
 if(stdlib_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw))return;
 if(reducer_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw))return;
 int native_result=MISSING;
 if(numpy_call(c,callable,args,nargs,kw_names,kw_values,nkw,&native_result)||
    json_call(c,callable,args,nargs,kw_names,kw_values,nkw,&native_result)){
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_result);return;
 }
 if(decimal_call(c,callable,args,nargs,kw_names,kw_values,nkw,&native_result)){
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_result);return;
 }
 if(context_call(c,callable,args,nargs,kw_names,kw_values,nkw,&native_result)){
  if(!c->error&&!c->h.error&&c->raised==MISSING)push(c,frame(c,caller),native_result);return;
 }
 if(dict_keyword_call(c,callable,args,nargs,kw_names,kw_values,nkw,&native_result)){
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_result);return;
 }
 if(nkw && (n[0]==T_NATIVE||n[0]==34||n[0]==41||n[0]==45||n[0]==46||n[0]==47)){c->error=BAD_CALL;return;}
 if(text_iterable_call(c,callable,args,nargs,caller))return;
 if(deepcopy_call(c,callable,args,nargs,&native_result)||text_call(c,callable,args,nargs,&native_result)||string_call(c,callable,args,nargs,&native_result)||exception_call(c,callable,args,nargs,&native_result)||random_call(c,callable,args,nargs,&native_result)||native_builtin_call(c,callable,args,nargs,&native_result)){
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_result);return;
 }
 if(n[0]==T_NATIVE){
  if(n[3]==NATIVE_OBJECT){if(nargs||nkw)c->error=BAD_CALL;else push(c,frame(c,caller),meta(c,36,0,0,0));return;}
  if(nkw || n[3]!=NATIVE_NEXT || nargs<1 || nargs>2){c->error=BAD_CALL;return;}
  int type=kind(c,args[0]);int fallback=nargs==2?args[1]:MISSING;
  if(type==T_GEN)resume_generator(c,args[0],caller,1,-1,fallback);
  else if(type==T_ITER){int value=iterator_next(c,args[0]);if(value==MISSING)value=fallback;
   if(value==MISSING)c->error=STOP_ITERATION;else push(c,frame(c,caller),value);}
  else if(type==70||type==71||type==72){
   if(type==70){int ot=kind(c,edge(c,args[0],0));if(ot==T_GEN){c->error=UNSUPPORTED_TYPE;return;}}
   int value=stdlib_iter_next(c,args[0]);if(value==MISSING)value=fallback;
   if(value==MISSING)c->error=STOP_ITERATION;else push(c,frame(c,caller),value);}
  else c->error=BAD_CALL;return;
 }
 int fid=-1,globals=MISSING,closure=MISSING,defaults=MISSING,kwdefaults=MISSING;
 if(n[0]==T_FUNCTION)fid=n[3];
 else if(n[0]==T_DYNFN){fid=n[3];globals=edge(c,callable,0);closure=edge(c,callable,1);defaults=edge(c,callable,2);kwdefaults=edge(c,callable,3);}
 else{c->error=BAD_CALL;return;}
 int id=new_frame(c,fid,args,nargs,caller,0,globals,closure,defaults,kw_names,kw_values,nkw,kwdefaults);if(id<0)return;
 if(function(c,fid)[7]&32){
  int gen=meta(c,T_GEN,0,0,id);frame(c,id)[4]=2;frame(c,id)[10]=gen;push(c,frame(c,caller),gen);
 }else c->current=id;
}
__device__ inline void return_value(Context* c,int value,int yielded){
 int id=c->current;int* f=frame(c,id);int caller=f[3],mode=f[5],target=f[6],fallback=f[7];
 if(yielded){
  if(f[10]<0 || caller<0){c->error=BAD_CALL;return;}f[4]=2;if(mode==5){reducer_resume(c,f[10],caller,target,value,0);return;}
  if(mode==7){stdlib_resume(c,f[10],caller,target,value,0);return;}
  if(mode==8){text_join_resume(c,f[10],caller,target,value,0);return;}push(c,frame(c,caller),value);c->current=caller;return;
 }
 if(f[10]>=0){
  int* gen=mutable_node(c,f[10]);if(gen)gen[3]=-1;
  f[4]=0;c->current=caller;
  if(caller<0){c->error=BAD_CALL;return;}
  if(mode==5){reducer_resume(c,f[10],caller,target,MISSING,1);return;}
  if(mode==7){stdlib_resume(c,f[10],caller,target,MISSING,1);return;}
  if(mode==8){text_join_resume(c,f[10],caller,target,MISSING,1);return;}
  if(mode==2){pop(c,frame(c,caller));frame(c,caller)[1]=target;}
  else if(fallback!=MISSING)push(c,frame(c,caller),fallback);else c->error=STOP_ITERATION;
 }else{
  f[4]=0;c->current=caller;
  if(mode==4){if(kind(c,value)!=TAG_NONE)c->error=BAD_CALL;else if(caller<0)c->error=BAD_PROGRAM;else push(c,frame(c,caller),target);}
  else if(mode==6){int* parent=frame(c,caller);block_push(c,parent,0,target,parent[2]);push(c,parent,value);}
  else if(mode==7){stdlib_resume(c,MISSING,caller,target,value,0);return;}
  else if(caller<0)c->output=value;else push(c,frame(c,caller),value);
 }
}
// COLLECTION_HANDLERS
__device__ inline int subscript(Context* c,int owner,int key){
 int np_out=MISSING;if(numpy_try_subscript(c,owner,key,&np_out))return np_out;
 int tag=kind(c,owner);
 if(tag==TAG_DICT || (tag>=8&&tag<=10)){
  int value=dict_get(&c->h,owner,key,MISSING);if(value==MISSING && !c->h.error)c->h.error=ERROR_KEY_NOT_FOUND;return value;
 }
 if(tag==TAG_LIST || tag==TAG_TUPLE){long long index=0;if(read_int64(&c->h,key,&index)<0)return MISSING;
  if(index<(-2147483647-1LL)||index>2147483647LL){c->error=BAD_PROGRAM;return MISSING;}return list_get(&c->h,owner,(int)index);}
 c->error=UNSUPPORTED_TYPE;return MISSING;
}
__device__ inline int contains(Context* c,int needle,int owner){
 int tag=kind(c,owner);if(tag==80)return dict_find(&c->h,node(c,owner)[1],needle)>=0; if(tag==TAG_DICT || (tag>=8&&tag<=10) || tag==73)return dict_find(&c->h,owner,needle)>=0;
 if(tag==21||tag==22)return set_contains(c,needle,owner);
 if(tag==TAG_STR){
  if(kind(c,needle)!=TAG_STR){c->error=UNSUPPORTED_TYPE;return 0;}
  NodeView a,b;get_node(&c->h,owner,&a);get_node(&c->h,needle,&b);
  const unsigned char* ap=get_payload_ptr(&c->h,&a),*bp=get_payload_ptr(&c->h,&b);
  for(int i=0;i<=a.count-b.count;i++){int equal=1;for(int j=0;j<b.count;j++)if(ap[i+j]!=bp[j]){equal=0;break;}if(equal)return 1;}return 0;
 }
 if(tag!=TAG_LIST && tag!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 0;}
 int n=length(c,owner);for(int i=0;i<n;i++)if(vm_equal(c,needle,edge(c,owner,i)))return 1;return 0;
}
__device__ inline int compare(Context* c,int a,int b,int mode){
 int stdlib_result=0;if(stdlib_compare(c,a,b,mode,&stdlib_result))return stdlib_result;
 int decimal_result=0;if(decimal_compare(c,a,b,mode,&decimal_result))return decimal_result;
 if(mode==2 || mode==3){int eq=vm_equal(c,a,b);return mode==2?eq:!eq;}
 int ta=kind(c,a),tb=kind(c,b),cmp=0;
 if((ta==TAG_TUPLE&&tb==TAG_TUPLE)||(ta==TAG_LIST&&tb==TAG_LIST)){
  int left=length(c,a),right=length(c,b),limit=left<right?left:right;
  for(int i=0;i<limit;i++){
   int av=edge(c,a,i),bv=edge(c,b,i);
   if(vm_equal(c,av,bv))continue;
   if(c->error||c->h.error)return 0;
   // Python compares the first unequal pair with the requested operator;
   // length is relevant only after an entirely equal common prefix.
   return compare(c,av,bv,mode);
  }
  cmp=left<right?-1:left>right?1:0;
 }
 else if(ta==TAG_STR&&tb==TAG_STR)cmp=str_compare(&c->h,a,b);
 else if((ta==TAG_INT||ta==TAG_BOOL)&&(tb==TAG_INT||tb==TAG_BOOL)){
  long long aa=0,bb=0;numeric_int(c,a,&aa);numeric_int(c,b,&bb);cmp=aa<bb?-1:aa>bb?1:0;
 }else if(ta==TAG_FLOAT||tb==TAG_FLOAT){
  if((ta!=TAG_FLOAT&&ta!=TAG_INT&&ta!=TAG_BOOL)||(tb!=TAG_FLOAT&&tb!=TAG_INT&&tb!=TAG_BOOL)){c->error=UNSUPPORTED_TYPE;return 0;}
  if(ta==TAG_FLOAT&&tb==TAG_FLOAT){double aa=0,bb=0;numeric_double(c,a,&aa);numeric_double(c,b,&bb);cmp=aa<bb?-1:aa>bb?1:0;}
  else{int integer_ref=ta==TAG_FLOAT?b:a,float_ref=ta==TAG_FLOAT?a:b;long long integer=0;double floating=0;numeric_int(c,integer_ref,&integer);numeric_double(c,float_ref,&floating);
   if(floating>=9223372036854775808.0)cmp=-1;else if(floating< -9223372036854775808.0)cmp=1;
   else{long long truncated=(long long)floating;cmp=integer<truncated?-1:integer>truncated?1:((double)truncated<floating?-1:(double)truncated>floating?1:0);}
   if(ta==TAG_FLOAT)cmp=-cmp;
  }
 }else{c->error=UNSUPPORTED_TYPE;return 0;}
 if(mode==0)return cmp<0;if(mode==1)return cmp<=0;if(mode==4)return cmp>0;if(mode==5)return cmp>=0;
 c->error=UNSUPPORTED_OP;return 0;
}
__device__ inline void execute(Context* c){
 while(c->current>=0){
  if(c->error==STOP_ITERATION){c->error=0;c->raised=exception_create(c,exception_type(c,7),make_tuple(&c->h,0,0));}
  if(c->error)break;
  if(c->h.error==ERROR_KEY_NOT_FOUND&&c->error_value!=MISSING){
   int key=c->error_value;c->h.error=0;c->raised=exception_create(c,exception_type(c,3),make_tuple(&c->h,&key,1));
  }else if(c->h.error==ERROR_INDEX_OUT_OF_BOUNDS){
   c->h.error=0;const char* message=c->error_kind==2?"list assignment index out of range":c->error_kind==3?"pop from empty list":c->error_kind==4?"pop index out of range":"list index out of range";
   int length=0;while(message[length])length++;int text=make_utf8(&c->h,message,length);
   c->raised=exception_create(c,exception_type(c,4),make_tuple(&c->h,&text,1));
  }
  if(c->h.error)break;
  if(c->raised!=MISSING){unwind_exception(c);continue;}

  if(++c->steps>c->step_limit){c->error=STEP_LIMIT;break;}
  if(c->deadline && (c->steps&63)==0 && clock_ns()>=(unsigned long long)c->deadline){c->error=DEADLINE;break;}
  c->error_value=MISSING;c->error_kind=0;
  int* f=frame(c,c->current);const int* fn=function(c,f[0]);if(!fn)break;
  int pc=f[1];if(pc<fn[0]||pc>=fn[0]+fn[1]){c->error=BAD_PROGRAM;break;}
  const int* ins=c->instructions+pc*4;int op=ins[0],arg=ins[1],value=ins[2],target=ins[3];f[1]++;
  if(handle_strings(c,f,op,arg))continue;
  if(handle_exceptions(c,f,op,arg,value,target))continue;
  if(stdlib_handle_opcode(c,f,op,arg,value,target))continue;
  if(handle_collections(c,f,op,arg,value,target))continue;
  switch(op){
   case OP_NOP:case OP_EXTENDED_ARG:break;
   case OP_LOAD_CONST:push(c,f,value);break;
   case OP_LOAD_FAST:if(arg<0||arg>=fn[5])c->error=BAD_PROGRAM;else if(f[LBASE+arg]==MISSING)c->error=UNBOUND_LOCAL;else push(c,f,f[LBASE+arg]);break;
   case OP_STORE_FAST:if(arg<0||arg>=fn[5])c->error=BAD_PROGRAM;else f[LBASE+arg]=pop(c,f);break;
   case OP_DELETE_FAST:if(arg<0||arg>=fn[5])c->error=BAD_PROGRAM;else f[LBASE+arg]=MISSING;break;
   case OP_LOAD_GLOBAL:{int v=dict_get(&c->h,f[8],value,MISSING);if(v==MISSING)c->error=UNBOUND_LOCAL;else push(c,f,v);break;}
   case OP_LOAD_DEREF:case OP_LOAD_CLOSURE:case OP_STORE_DEREF:{
    if(arg<0||arg>=f[11]){c->error=BAD_PROGRAM;break;}int ref=f[CBASE+arg];int* cell=mutable_node(c,ref);
    if(!cell || cell[0]!=T_CELL){c->error=BAD_PROGRAM;break;}
    if(op==OP_LOAD_CLOSURE)push(c,f,ref);else if(op==OP_LOAD_DEREF){if(cell[1]==MISSING)c->error=UNBOUND_LOCAL;else push(c,f,cell[1]);}else cell[1]=pop(c,f);break;
   }
   case OP_POP_TOP:pop(c,f);break;
   case OP_DUP_TOP:push(c,f,peek(c,f));break;
   case OP_ROT_TWO:{int a=pop(c,f),b=pop(c,f);push(c,f,a);push(c,f,b);break;}
   case OP_BUILD_LIST:case OP_BUILD_TUPLE:{
    if(arg<0||arg>f[2]){c->error=BAD_PROGRAM;break;}int ref;
    if(op==OP_BUILD_TUPLE)ref=make_tuple(&c->h,f+SBASE+f[2]-arg,arg);
    else{ref=make_list(&c->h);for(int i=f[2]-arg;i<f[2];i++)list_append(&c->h,ref,f[SBASE+i]);}
    f[2]-=arg;push(c,f,ref);break;
   }
   case OP_MAKE_FUNCTION:{
    int qual=pop(c,f),code=pop(c,f);const int* n=node(c,code);
    if(!n||n[0]!=T_CODE||arg&~15){c->error=BAD_CALL;break;}
    int closure=arg&8?pop(c,f):NIL;
    if(arg&4)pop(c,f);int kw=arg&2?pop(c,f):NIL;int defaults=arg&1?pop(c,f):NIL;
    int vals[6]={f[8],closure,defaults,kw,qual,code};int ref=make_tuple(&c->h,vals,6);int* out=mutable_node(c,ref);
    if(out){out[0]=T_DYNFN;out[3]=n[3];push(c,f,ref);}break;
   }
   case OP_CALL_FUNCTION:case OP_CALL_FUNCTION_KW:{
    int names=MISSING,nkw=0;if(op==OP_CALL_FUNCTION_KW){names=pop(c,f);if(kind(c,names)!=TAG_TUPLE){c->error=BAD_CALL;break;}nkw=length(c,names);}
    if(arg<0||arg+1>f[2]||nkw>arg){c->error=BAD_CALL;break;}int pos=f[2]-arg-1,callee=f[SBASE+pos];int args[VM_STACK];
    for(int i=0;i<arg;i++)args[i]=f[SBASE+pos+1+i];f[2]=pos;call(c,callee,args,arg-nkw,c->current,names,args+arg-nkw,nkw);break;
   }
   case OP_CALL_FUNCTION_EX:{
    if(arg&~1){c->error=BAD_CALL;break;}int kwargs=arg&1?pop(c,f):MISSING,argv=pop(c,f),callee=pop(c,f);
    int tag=kind(c,argv);if(tag!=TAG_LIST&&tag!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;break;}
    int nargs=length(c,argv),nkw=kwargs==MISSING?0:length(c,kwargs);int args[VM_STACK],keys[VM_STACK],vals[VM_STACK];
    if(nargs>VM_STACK||nkw>VM_STACK||(kwargs!=MISSING&&kind(c,kwargs)!=TAG_DICT)){c->error=BAD_CALL;break;}
    for(int i=0;i<nargs;i++)args[i]=edge(c,argv,i);
    for(int i=0;i<nkw;i++){keys[i]=edge(c,kwargs,i);vals[i]=edge(c,kwargs,i,1);}
    int names=nkw?make_tuple(&c->h,keys,nkw):MISSING;
    call(c,callee,args,nargs,c->current,names,vals,nkw);break;
   }
   case OP_IMPORT_NAME:{pop(c,f);pop(c,f);if(kind(c,value)!=14)c->error=BAD_PROGRAM;else push(c,f,value);break;}
   case OP_IMPORT_FROM:case OP_LOAD_ATTR:case OP_LOAD_METHOD:{
    int owner=op==OP_IMPORT_FROM?peek(c,f):pop(c,f),getter=0;
    int attr=numpy_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=object_attribute(c,owner,value,&getter);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=stdlib_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=builtin_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=random_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=exception_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=text_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=decimal_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=context_attribute(c,owner,value);
    if(attr==MISSING){c->error=UNSUPPORTED_TYPE;break;}
    if(getter){if(op==OP_LOAD_METHOD){c->error=UNSUPPORTED_TYPE;break;}call(c,attr,0,0,c->current);}
    else{if(op==OP_LOAD_METHOD)push(c,f,NIL);push(c,f,attr);}break;
   }
   case OP_CALL_METHOD:{
    if(arg<0||arg+2>f[2]){c->error=BAD_CALL;break;}int pos=f[2]-arg-2,callee=f[SBASE+pos+1],args[VM_STACK];
    if(f[SBASE+pos]!=NIL){c->error=BAD_CALL;break;}for(int i=0;i<arg;i++)args[i]=f[SBASE+pos+2+i];
    f[2]=pos;call(c,callee,args,arg,c->current);break;
   }
   case OP_STORE_ATTR:{int owner=pop(c,f),v=pop(c,f);object_store(c,owner,value,v);break;}
   case OP_GET_ITER:push(c,f,iterator(c,pop(c,f)));break;
   case OP_FOR_ITER:{int ref=peek(c,f);if(kind(c,ref)==T_GEN)resume_generator(c,ref,c->current,2,target,MISSING);
    else{int next=iterator_next(c,ref);if(next==MISSING){pop(c,f);f[1]=target;}else push(c,f,next);}break;}
   case OP_GEN_START:if(pop(c,f)!=NIL)c->error=BAD_CALL;break;
   case OP_YIELD_VALUE:return_value(c,pop(c,f),1);break;
   case OP_RETURN_VALUE:return_value(c,pop(c,f),0);break;
   case OP_BINARY_SUBSCR:{int key=pop(c,f),owner=pop(c,f);push(c,f,collection_subscript(c,owner,key));break;}
   case OP_COMPARE_OP:{int b=pop(c,f),a=pop(c,f);push(c,f,compare(c,a,b,arg)?TRUE_REF:FALSE_REF);break;}
   case OP_CONTAINS_OP:{int owner=pop(c,f),needle=pop(c,f);int yes=contains(c,needle,owner);push(c,f,(arg?!yes:yes)?TRUE_REF:FALSE_REF);break;}
   case OP_IS_OP:{int b=pop(c,f),a=pop(c,f);int ta=kind(c,a),tb=kind(c,b);int yes=a==b;
    if(ta==TAG_NONE&&tb==TAG_NONE)yes=1;else if(ta==TAG_BOOL&&tb==TAG_BOOL)yes=node(c,a)[3]==node(c,b)[3];
    push(c,f,(arg?!yes:yes)?TRUE_REF:FALSE_REF);break;}
   case OP_UNARY_NOT:{int yes=truth_value(c,pop(c,f));push(c,f,yes?FALSE_REF:TRUE_REF);break;}
   case OP_UNARY_NEGATIVE:{int ref=pop(c,f);if(kind(c,ref)==TAG_FLOAT){double a=0;read_float64(&c->h,ref,&a);push(c,f,make_float64(&c->h,-a));}
    else{long long a=0;numeric_int(c,ref,&a);if(a==(-9223372036854775807LL-1))c->h.error=ERROR_INT_OVERFLOW;else push(c,f,make_int64(&c->h,-a));}break;}
   case OP_BINARY_ADD:case OP_BINARY_SUBTRACT:case OP_BINARY_MULTIPLY:case OP_INPLACE_ADD:case OP_INPLACE_SUBTRACT:case OP_INPLACE_MULTIPLY:
   case OP_BINARY_TRUE_DIVIDE:case OP_BINARY_FLOOR_DIVIDE:case OP_BINARY_AND:
   case OP_BINARY_MODULO:case OP_BINARY_OR:case OP_BINARY_XOR:case OP_INPLACE_OR:case OP_INPLACE_AND:case OP_INPLACE_XOR:
    case OP_INPLACE_FLOOR_DIVIDE:case OP_INPLACE_TRUE_DIVIDE:case OP_INPLACE_MODULO:{
    int b=pop(c,f),a=pop(c,f);int stdlib_out=MISSING;if(stdlib_binary(c,a,b,op,&stdlib_out))push(c,f,stdlib_out);else push(c,f,numeric_binary(c,a,b,op));break;

   }
   case OP_JUMP_ABSOLUTE:case OP_JUMP_FORWARD:f[1]=target;break;
   case OP_POP_JUMP_IF_FALSE:case OP_POP_JUMP_IF_TRUE:{int yes=truth_value(c,pop(c,f));if(yes==(op==OP_POP_JUMP_IF_TRUE))f[1]=target;break;}
   case OP_JUMP_IF_FALSE_OR_POP:case OP_JUMP_IF_TRUE_OR_POP:{int yes=truth_value(c,peek(c,f));if(yes==(op==OP_JUMP_IF_TRUE_OR_POP))f[1]=target;else pop(c,f);break;}
   case OP_LOAD_ASSERTION_ERROR:push(c,f,exception_type(c,12));break;
   case OP_SETUP_WITH:{
    int mgr=pop(c,f),self_id=c->current,getter=0;
    int exit_name=make_utf8(&c->h,"__exit__",8),enter_name=make_utf8(&c->h,"__enter__",9);
    int exit_attr=object_attribute(c,mgr,exit_name,&getter);
    if(exit_attr==MISSING&&!c->error&&!c->h.error)exit_attr=context_attribute(c,mgr,exit_name);
    if(exit_attr==MISSING||getter){c->error=UNSUPPORTED_TYPE;break;}push(c,f,exit_attr);
    int enter_attr=object_attribute(c,mgr,enter_name,&getter);
    if(enter_attr==MISSING&&!c->error&&!c->h.error)enter_attr=context_attribute(c,mgr,enter_name);
    if(enter_attr==MISSING||getter){c->error=UNSUPPORTED_TYPE;break;}
    call(c,enter_attr,0,0,self_id);
    if(c->error||c->h.error||c->raised!=MISSING)break;
    if(c->current==self_id){int res=pop(c,f);block_push(c,f,0,target,f[2]);push(c,f,res);}
    else if(c->current>=0){frame(c,c->current)[5]=6;frame(c,c->current)[6]=target;}
    else c->error=UNSUPPORTED_TYPE;break;
   }
   case OP_WITH_EXCEPT_START:{
    if(f[2]<7){c->error=BAD_PROGRAM;break;}
    int type=f[SBASE+f[2]-1],val=f[SBASE+f[2]-2],tb=f[SBASE+f[2]-3],exit_func=f[SBASE+f[2]-7];
    int args[3]={type,val,tb};call(c,exit_func,args,3,c->current);break;
   }
   default:c->error=UNSUPPORTED_OP;break;
  }
 }
}
} // namespace duel_rule_vm
extern "C" __global__ void vm_clock(long long* output){if(blockIdx.x==0&&threadIdx.x==0)output[0]=(long long)duel_rule_vm::clock_ns();}
extern "C" __global__ void vm_compact(
 int* sn,int* se,unsigned char* sp,int* sc,const int* cn,const int* ce,const unsigned char* cp,
 int cnc,int cec,int cpc,const int* roots,int* dn,int* de,unsigned char* dp,int* dc,int* dr,
 int* map,int* queue,int* status,int rows,int maxn,int maxe,int maxp,int capacity){
 using namespace duel_vm_heap;int row=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(row>=rows)return;
 Heap source,target;
 init_heap(&source,sn+(long long)row*maxn*4,se+(long long)row*maxe*2,sp+(long long)row*maxp,
  sc+row*3,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 init_heap(&target,dn+(long long)row*maxn*4,de+(long long)row*maxe*2,dp+(long long)row*maxp,
  dc+row*3,maxn,maxe,maxp,0,0,0,0,0,0);
 dr[row]=compact_into(&source,&target,roots[row],map+(long long)row*capacity,queue+(long long)row*capacity,capacity);
 status[row]=target.error;
}
extern "C" __global__ void vm_execute(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* args,int nargs,
 const int* constants_nodes,const int* constants_edges,const unsigned char* constants_payload,
 const int* instructions,const int* functions,int function_count,
 int constant_node_count,int constant_edge_count,int constant_payload_count,
 int* frames,int* outputs,int* statuses,const long long* deadlines,
 int rows,int max_nodes,int max_edges,int max_payload,int entry,int step_limit,const int* classes,int class_count,const int* modules,int module_count,long long* gc_marks,int* gc_queue){
 using namespace duel_rule_vm;int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(tid>=rows)return;
 Context c;c.deepcopy_epoch=0;c.gc_marks=gc_marks+(long long)tid*(max_nodes+constant_node_count);c.gc_queue=gc_queue+(long long)tid*max_nodes;c.classes=classes;c.class_count=class_count;c.modules=modules;c.module_count=module_count;c.h.nodes=nodes+(long long)tid*max_nodes*4;c.h.edges=edges+(long long)tid*max_edges*2;
 c.h.payload=payload+(long long)tid*max_payload;c.h.counts=counts+tid*3;
 c.h.max_nodes=max_nodes;c.h.max_edges=max_edges;c.h.max_payload=max_payload;
 c.h.constant_nodes=constants_nodes;c.h.constant_edges=constants_edges;c.h.constant_payload=constants_payload;
 c.h.constant_node_count=constant_node_count;c.h.constant_edge_count=constant_edge_count;c.h.constant_payload_count=constant_payload_count;c.h.error=0;
 c.instructions=instructions;c.functions=functions;c.function_count=function_count;c.frames=frames+(long long)tid*VM_FRAMES*STRIDE;
 for(int i=0;i<VM_FRAMES;i++)frame(&c,i)[4]=0;
 c.current=-1;c.output=MISSING;c.error=0;c.raised=MISSING;c.error_value=MISSING;c.error_kind=0;c.steps=0;c.step_limit=step_limit;c.deadline=deadlines[tid];c.context_values=MISSING;
 c.current=new_frame(&c,entry,args+(long long)tid*nargs,nargs,-1,3);
 if(c.current>=0)execute(&c);
 outputs[tid]=c.output;statuses[tid*4]=c.h.error?c.h.error:c.error;
 statuses[tid*4+1]=c.current>=0?frame(&c,c.current)[0]:entry;
 statuses[tid*4+2]=c.current>=0?frame(&c,c.current)[1]:-1;statuses[tid*4+3]=c.steps;
}
'''


_GRAPH_TRANSFER_SOURCE=r'''
extern "C" __global__ void vm_load_graph(
 const int* sn,const int* se,const unsigned char* sp,const int* sc,const int* sr,
 int* dn,int* de,unsigned char* dp,int* dc,int* dr,const unsigned char* active,int* errors,
 int rows,int smn,int sme,int smp,int dmn,int dme,int dmp){
 using namespace duel_vm_heap;
 int row=blockIdx.x;if(row>=rows)return;
 __shared__ int valid,n,e,p;
 if(threadIdx.x==0){
  n=sc[row*3];e=sc[row*3+1];p=sc[row*3+2];valid=0;errors[row]=0;
  if(active[row]){
   if(n<1||n>smn||e<0||e>sme||p<0||p>smp||sr[row]<0||sr[row]>=n)errors[row]=ERROR_INVALID_REF;
   else if(n>dmn)errors[row]=ERROR_NODE_CAPACITY;
   else if(e>dme)errors[row]=ERROR_EDGE_CAPACITY;
   else if(p>dmp)errors[row]=ERROR_PAYLOAD_CAPACITY;
   else valid=1;
  }
  if(!valid){dc[row*3]=0;dc[row*3+1]=0;dc[row*3+2]=0;dr[row]=0;}
 }
 __syncthreads();if(!valid)return;
 for(long long i=threadIdx.x;i<(long long)n*4;i+=blockDim.x)dn[(long long)row*dmn*4+i]=sn[(long long)row*smn*4+i];
 for(long long i=threadIdx.x;i<(long long)e*2;i+=blockDim.x)de[(long long)row*dme*2+i]=se[(long long)row*sme*2+i];
 for(long long i=threadIdx.x;i<p;i+=blockDim.x)dp[(long long)row*dmp+i]=sp[(long long)row*smp+i];
 __syncthreads();if(threadIdx.x==0){dc[row*3]=n;dc[row*3+1]=e;dc[row*3+2]=p;dr[row]=sr[row];}
}
'''


class DeviceGraph:
    """Owned GPU graph buffers. Ingress copying is not rule computation."""
    def __init__(self,layout,nodes,edges,payload,counts,roots):
        self.layout=layout;self.nodes=nodes;self.edges=edges;self.payload=payload;self.counts=counts;self.roots=roots
    @property
    def rows(self):return int(self.roots.shape[0])
    def fork(self):return DeviceGraph(self.layout,*(getattr(self,k).clone() for k in ('nodes','edges','payload','counts','roots')))


@dataclass(frozen=True)
class DeviceVmWorkspace:
    """Reusable borrowed VM scratch, independent of persistent root count.

    The graph is mutated in place. Calls and their outputs borrow these buffers
    until the next use; compact or consume the result before reusing the lanes.
    This is execution storage, not a completed-rule or concurrency receipt.
    """
    executor:object
    arena:DeviceGraph
    frames:object
    outputs:object
    statuses:object
    marks:object
    queue:object


@dataclass(frozen=True)
class DeviceCompactionWorkspace:
    """Reusable reachable-graph projection into smaller temporary lanes."""
    executor:object
    source:DeviceGraph
    arena:DeviceGraph
    mapping:object
    queue:object
    errors:object


class DeviceCall:
    def __init__(self,executor,arena,outputs,statuses,frames,event,scratch_buffers=()):
        self.executor=executor;self.arena=arena;self.outputs=outputs;self.statuses=statuses;self.frames=frames;self.event=event;self.scratch_buffers=scratch_buffers
    def check(self):
        if self.event is None:
            self.executor.torch.cuda.current_stream(self.executor.device).synchronize()
        else:
            self.event.synchronize()
        statuses=self.statuses.cpu().numpy()
        bad=np.flatnonzero(statuses[:,0])
        if len(bad):
            row=int(bad[0]);code,fid,pc,steps=map(int,statuses[row]);name=self.executor.image.function_ids[fid] if 0<=fid<len(self.executor.image.function_ids) else 'unknown'
            path=os.environ.get('NTE_CUDA_VM_FAILURE_REPORT')
            if path:
                self.write_failure_report(path,row,fid,pc)
            raise VmExecutionError(f'CUDA VM row {row}: status={code}, function={name}, pc={pc}, steps={steps}')
        return statuses
    def write_failure_report(self,path,row,fid,pc):
        """Explicit bounded post-failure diagnostics; never a search hot path.

        Only local references and short scalar text are retained. This does not
        resume rules, decode whole states or turn failure into a valid result.
        """
        import json
        from pathlib import Path
        ex=self.executor
        frames=self.frames[row].cpu().numpy()
        nodes=self.arena.nodes[row].cpu().numpy()
        payload=self.arena.payload[row].cpu().numpy()
        def describe(ref):
            ref=int(ref)
            if ref==-2147483648:return dict(ref=ref,missing=True)
            source=ex.image.nodes if ref<0 else nodes
            index=-1-ref if ref<0 else ref
            if index<0 or index>=len(source):return dict(ref=ref,invalid=True)
            tag,off,count,extra=map(int,source[index]);result=dict(ref=ref,tag=tag,extra=extra)
            data=ex.image.payload if ref<0 else payload
            if tag==4 and 0<=off and 0<=count<=128 and off+count<=len(data):
                result['text']=data[off:off+count].tobytes().decode('utf8',errors='replace')
            return result
        report=dict(row=row,function=fid,pc=pc,frames=[])
        for index,f in enumerate(frames):
            if not f[4]:continue
            fn=int(f[0]);meta=ex.image.functions[fn] if 0<=fn<len(ex.image.functions) else None
            count=min(int(meta[5]) if meta is not None else 0,ex.limits.locals)
            report['frames'].append(dict(frame=index,function=fn,pc=int(f[1]),status=int(f[4]),
                locals=[describe(v) for v in f[14:14+count]]))
        Path(path).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    def decode(self):
        """Explicit diagnostic export. Does not run Python rule functions."""
        self.check();refs=self.outputs.cpu().tolist();ex=self.executor;torch=ex.torch
        a=self.arena;local=[i for i,r in enumerate(refs) if r>=0]
        nodes={}
        if local:
            row=torch.tensor(local,dtype=torch.long,device=ex.device);ids=torch.tensor([refs[i] for i in local],dtype=torch.long,device=ex.device)
            if bool((ids>=a.counts[row,0]).any()):raise VmExecutionError('Output reference outside graph')
            values=a.nodes[row,ids].cpu().numpy()
            nodes={i:values[j] for j,i in enumerate(local)}
        for i,r in enumerate(refs):
            if r<0:
                index=-1-r
                if index>=len(ex.image.nodes):raise VmExecutionError('Output constant outside program')
                nodes[i]=ex.image.nodes[index]
        if all(int(n[0]) in range(5) for n in nodes.values()):
            # Scalar output transfers only its payload, not an entire 1600-row heap.
            selected=[i for i in local if int(nodes[i][0]) in (2,3,4)];buffers={}
            if selected:
                width=max(int(nodes[i][2]) for i in selected)
                if width:
                    rows=torch.tensor(selected,dtype=torch.long,device=ex.device)
                    offsets=torch.tensor([int(nodes[i][1]) for i in selected],dtype=torch.long,device=ex.device)
                    positions=offsets[:,None]+torch.arange(width,device=ex.device)[None,:]
                    # Padding bytes need not belong to an object, but always remain inside this row's allocation.
                    raw=a.payload[rows[:,None],positions.clamp(0,a.layout.max_payload_bytes-1)].cpu().numpy()
                    buffers={i:raw[j,:int(nodes[i][2])].tobytes() for j,i in enumerate(selected)}
            result=[]
            import struct
            for i,r in enumerate(refs):
                tag,offset,count,extra=map(int,nodes[i]);data=(ex.image.payload[offset:offset+count].tobytes() if r<0 else buffers.get(i,b''))
                if tag==0:value=None
                elif tag==1:value=bool(extra)
                elif tag==2:value=int.from_bytes(data,'little',signed=True)
                elif tag==3:value=struct.unpack('<d',data)[0]
                else:value=data.decode('utf-8')
                result.append(value)
            return result
        from .codec import unpack_states
        return unpack_states(self.export())

    def export(self):
        """Collect reachable output graph on GPU, retaining entities and aliases.

        Only explicit diagnostic export transfers full rows. Rule execution and
        garbage collection both remain on device; the codec validates the result.
        """
        self.check();ex=self.executor;torch=ex.torch;a=self.arena;l=a.layout;rows=a.rows;b=ex._buffers
        capacity=l.max_nodes+len(ex.image.nodes)
        dn=torch.zeros_like(a.nodes);de=torch.zeros_like(a.edges);dp=torch.zeros_like(a.payload);dc=torch.zeros_like(a.counts)
        dr=torch.empty_like(a.roots);mapping=torch.empty((rows,capacity),dtype=torch.int32,device=ex.device);queue=torch.empty_like(mapping)
        status=torch.empty(rows,dtype=torch.int32,device=ex.device)
        event=ex.module.launch('vm_compact',[a.nodes,a.edges,a.payload,a.counts,b['nodes'],b['edges'],b['payload'],
            len(ex.image.nodes),len(ex.image.edges),len(ex.image.payload),self.outputs,dn,de,dp,dc,dr,mapping,queue,status,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes,capacity],grid=(rows+127)//128,block=128)
        event.synchronize()
        values=status.cpu().numpy();bad=np.flatnonzero(values)
        if len(bad):raise VmExecutionError(f'GPU output graph export failed row={int(bad[0])}, status={int(values[bad[0]])}')
        from .schema import PackedStateBatch
        return PackedStateBatch(l,*(value.cpu().numpy() for value in (dn,de,dp,dc,dr)))


class CudaVm:
    """CUDA-only IR execution. Capability checks are entry-specific and mandatory."""
    supports_full_duel_rules=False
    rule_threads_per_block=1
    def __init__(self,image,*,device='cuda',limits=VmLimits(),guard_standalone_entities=False,
                 rule_threads_per_block=1,compiled_blocks=None):
        if not isinstance(image,VmImage):raise TypeError('Numeric VM image required')
        image.verify()
        if not isinstance(limits,VmLimits):raise TypeError('VmLimits required')
        if type(guard_standalone_entities) is not bool:raise TypeError('Boolean standalone-entity guard required')
        if type(rule_threads_per_block) is not int or not 1<=rule_threads_per_block<=1024:
            raise ValueError('Positive bounded rule thread block size required')
        if compiled_blocks is not None and not isinstance(compiled_blocks,dict):
            raise TypeError('Explicit compiled-block budget mapping required')
        self.compiled_blocks_config = None if compiled_blocks is None else dict(compiled_blocks)
        self.compiled_block_plan = None
        if compiled_blocks is not None:
            from .vm_compiled_blocks import emit_compiled_blocks
            self.compiled_block_plan=emit_compiled_blocks(image,**compiled_blocks)
        self.guard_standalone_entities=guard_standalone_entities
        # Each VM row is a large, irregular scalar interpreter. A one-thread
        # block lets independent scratch rows spread across SMs; packing a small
        # batch into one 128-thread block confines all rule work to one SM.
        # This is a launch policy, not a measured speedup or capacity approval.
        self.rule_threads_per_block=rule_threads_per_block
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Actual CUDA required, no CPU VM fallback')
        from .cuda_runtime import CudaModule
        self.image=image;self.limits=limits;self.torch=torch;self.device=torch.device(device)
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self._buffers={k:torch.tensor(np.array(getattr(image,k),copy=True),device=self.device) for k in
            ('instructions','functions','nodes','edges','payload','classes','modules')}
        signature=['int32*','int32*','uint8*','int32*','int32*','int32',
            'int32*','int32*','uint8*','int32*','int32*','int32','int32','int32','int32',
            'int32*','int32*','int32*','int64*','int32','int32','int32','int32','int32','int32','int32*','int32','int32*','int32','int64*','int32*']
        compact_signature=['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*',
            'int32','int32','int32','int32*','int32*','int32*','uint8*','int32*','int32*','int32*','int32*','int32*',
            'int32','int32','int32','int32','int32']
        self.module=CudaModule(emit_vm_source(image,limits,compiled_blocks=self.compiled_blocks_config),{
            'vm_execute':signature,'vm_execute_device':signature+['uint8*'],
            'vm_clock':['int64*'],'vm_compact':compact_signature,
            'vm_compact_device':compact_signature+['int32','int32','int32','uint8*','int32*'],
            'vm_load_graph':['int32*','int32*','uint8*','int32*','int32*']*2
                            +['uint8*','int32*']+['int32']*7},device=str(self.device))
        try:
            self.module.ensure_stack()
        except BaseException:
            self.module.close()
            raise
        self.closed=False
    def capability(self,entry):
        missing=self.image.missing_capabilities(entry,opcodes=SUPPORTED_OPS,intrinsics=SUPPORTED_INTRINSICS)
        guarded=sorted(set(missing['intrinsics'])&DENIED_STANDALONE_INTRINSICS) if self.guard_standalone_entities else []
        blocked=bool(missing['opcodes'] or set(missing['intrinsics'])-set(guarded))
        return dict(entry=entry,ready=not blocked,missing=missing,runtime_denied_intrinsics=guarded,
                    full_duel_rules=False,actual_backend='cuda',
                    rule_threads_per_block=self.rule_threads_per_block)
    def clock_ns(self):
        if self.closed:raise RuntimeError('VM closed')
        out=self.torch.empty(1,dtype=self.torch.int64,device=self.device)
        event=self.module.launch('vm_clock',[out],grid=1,block=1)
        event.synchronize()
        return int(out.cpu()[0])
    def ingest(self,packed):
        if self.closed:raise RuntimeError('VM closed')
        from .schema import PackedStateBatch
        from .codec import unpack_states
        if not isinstance(packed,PackedStateBatch):raise TypeError('Packed graph ingress required')
        unpack_states(packed)  # Validate the ingress graph; never calculate a rule result here.
        torch=self.torch
        return DeviceGraph(packed.layout,*(torch.tensor(np.array(getattr(packed,k),copy=True),device=self.device)
                                          for k in ('nodes','edges','payload','counts','roots')))
    def launch(self,arena,entry,*,argument_refs=None,gpu_deadlines=None):
        if self.closed:raise RuntimeError('VM closed')
        self.image.verify();cap=self.capability(entry)
        if not cap['ready']:raise VmCapabilityError(str(cap['missing']))
        if not isinstance(arena,DeviceGraph):raise TypeError('Device graph required')
        torch=self.torch;rows=arena.rows
        if rows<1:raise ValueError('Nonempty device batch required')
        if argument_refs is None:argument_refs=arena.roots[:,None]
        if (not isinstance(argument_refs,torch.Tensor) or argument_refs.device!=self.device
                or argument_refs.dtype!=torch.int32 or argument_refs.ndim!=2 or argument_refs.shape[0]!=rows
                or not argument_refs.is_contiguous()):raise ValueError('Contiguous CUDA int32 argument refs required')
        if gpu_deadlines is None:gpu_deadlines=torch.zeros(rows,dtype=torch.int64,device=self.device)
        if (not isinstance(gpu_deadlines,torch.Tensor) or gpu_deadlines.shape!=(rows,)
                or gpu_deadlines.device!=self.device or gpu_deadlines.dtype!=torch.int64 or not gpu_deadlines.is_contiguous()):raise ValueError('CUDA int64 deadline array required')
        if bool((gpu_deadlines<0).any()):raise ValueError('Invalid GPU deadline')
        scratch=arena.fork();stride=14+self.limits.locals+self.limits.stack+self.limits.cells
        frames=torch.empty((rows,self.limits.frames,stride),dtype=torch.int32,device=self.device)
        outputs=torch.empty(rows,dtype=torch.int32,device=self.device);statuses=torch.zeros((rows,4),dtype=torch.int32,device=self.device)
        b=self._buffers;l=arena.layout
        gc_marks=torch.zeros((rows,l.max_nodes+len(self.image.nodes)),dtype=torch.int64,device=self.device);gc_queue=torch.empty((rows,l.max_nodes),dtype=torch.int32,device=self.device)
        event=self.module.launch('vm_execute',[scratch.nodes,scratch.edges,scratch.payload,scratch.counts,argument_refs,argument_refs.shape[1],
            b['nodes'],b['edges'],b['payload'],b['instructions'],b['functions'],len(self.image.functions),
            len(self.image.nodes),len(self.image.edges),len(self.image.payload),frames,outputs,statuses,gpu_deadlines,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes,self.image.entries[entry],self.limits.steps,b['classes'],len(self.image.classes),b['modules'],len(self.image.modules),gc_marks,gc_queue],grid=(rows+self.rule_threads_per_block-1)//self.rule_threads_per_block,block=self.rule_threads_per_block)
        return DeviceCall(self,scratch,outputs,statuses,frames,event,(gc_marks,gc_queue))
    def launch_device(self,arena,entry,*,argument_refs,active,gpu_deadlines):
        """Masked capturable rule call with device-owned errors and no checks.

        Exact static entry/shape validation remains mandatory. Inactive rows
        preserve their graph and never invoke rules. The caller checks statuses
        after the complete device batch; this method is not full-game approval.
        """
        if self.closed:raise RuntimeError('VM closed')
        if not self.capability(entry)['ready']:raise VmCapabilityError('Unsupported frozen device entry')
        if not isinstance(arena,DeviceGraph):raise TypeError('Device graph required')
        t=self.torch;rows=arena.rows
        for value,dtype,shape in ((active,t.uint8,(rows,)),(gpu_deadlines,t.int64,(rows,))):
            if (not isinstance(value,t.Tensor) or value.device!=self.device or value.dtype!=dtype
                    or value.shape!=shape or not value.is_contiguous()):raise ValueError('Exact device selector/deadline vector required')
        if (not isinstance(argument_refs,t.Tensor) or argument_refs.device!=self.device
                or argument_refs.dtype!=t.int32 or argument_refs.ndim!=2 or argument_refs.shape[0]!=rows
                or argument_refs.shape[1]<1 or not argument_refs.is_contiguous()):raise ValueError('Exact device argument matrix required')
        scratch=arena.fork();stride=14+self.limits.locals+self.limits.stack+self.limits.cells
        frames=t.empty((rows,self.limits.frames,stride),dtype=t.int32,device=self.device)
        outputs=t.empty(rows,dtype=t.int32,device=self.device)
        statuses=t.zeros((rows,4),dtype=t.int32,device=self.device)
        b=self._buffers;l=arena.layout
        marks=t.zeros((rows,l.max_nodes+len(self.image.nodes)),dtype=t.int64,device=self.device)
        queue=t.empty((rows,l.max_nodes),dtype=t.int32,device=self.device)
        self.module.launch('vm_execute_device',[scratch.nodes,scratch.edges,scratch.payload,scratch.counts,argument_refs,argument_refs.shape[1],
            b['nodes'],b['edges'],b['payload'],b['instructions'],b['functions'],len(self.image.functions),
            len(self.image.nodes),len(self.image.edges),len(self.image.payload),frames,outputs,statuses,gpu_deadlines,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes,self.image.entries[entry],self.limits.steps,b['classes'],len(self.image.classes),b['modules'],len(self.image.modules),marks,queue,active],
            grid=(rows+self.rule_threads_per_block-1)//self.rule_threads_per_block,
            block=self.rule_threads_per_block,capture_safe=True)
        return DeviceCall(self,scratch,outputs,statuses,frames,None,(marks,queue))

    def prepare_device_workspace(self, arena):
        """Allocate scratch once, before capture, for GPU-selected work lanes.

        Persistent roots live elsewhere. This method does not fork the graph or
        allocate a heap for every root; the caller owns and refills these lanes.
        """
        if self.closed:raise RuntimeError('VM closed')
        t=self.torch
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare reusable VM storage before CUDA capture')
        self._validate_device_graph(arena)
        rows=arena.rows;l=arena.layout
        stride=14+self.limits.locals+self.limits.stack+self.limits.cells
        return DeviceVmWorkspace(self,arena,
            t.empty((rows,self.limits.frames,stride),dtype=t.int32,device=self.device),
            t.empty(rows,dtype=t.int32,device=self.device),
            t.zeros((rows,4),dtype=t.int32,device=self.device),
            t.zeros((rows,l.max_nodes+len(self.image.nodes)),dtype=t.int64,device=self.device),
            t.empty((rows,l.max_nodes),dtype=t.int32,device=self.device))

    def _validate_device_graph(self, arena):
        if not isinstance(arena,DeviceGraph):raise TypeError('Device graph required')
        t=self.torch;l=arena.layout;rows=arena.rows
        if not 0<rows<=l.game_capacity:raise ValueError('Bounded graph rows required')
        for value,dtype,shape in (
                (arena.nodes,t.int32,(rows,l.max_nodes,4)),
                (arena.edges,t.int32,(rows,l.max_edges,2)),
                (arena.payload,t.uint8,(rows,l.max_payload_bytes)),
                (arena.counts,t.int32,(rows,3)),(arena.roots,t.int32,(rows,))):
            if (not isinstance(value,t.Tensor) or value.dtype!=dtype
                    or value.device!=self.device or value.shape!=shape
                    or not value.is_contiguous()):
                raise ValueError('Exact contiguous CUDA graph buffers required')

    def launch_into_device(self, workspace, entry, *, argument_refs, active, gpu_deadlines):
        """Run masked frozen rules in reusable lanes without heap allocation.

        No host row dispatch, fork, synchronization or numeric selector read is
        performed. Error/status vectors stay on device. The caller must compact
        successful results before another call overwrites the borrowed buffers.
        """
        if self.closed:raise RuntimeError('VM closed')
        if not isinstance(workspace,DeviceVmWorkspace) or workspace.executor is not self:
            raise ValueError('Workspace must belong to this frozen VM')
        if not self.capability(entry)['ready']:
            raise VmCapabilityError('Unsupported frozen device entry')
        arena=workspace.arena;self._validate_device_graph(arena)
        t=self.torch;rows=arena.rows
        stride=14+self.limits.locals+self.limits.stack+self.limits.cells
        for value,dtype,shape in (
                (workspace.frames,t.int32,(rows,self.limits.frames,stride)),
                (workspace.outputs,t.int32,(rows,)),(workspace.statuses,t.int32,(rows,4)),
                (workspace.marks,t.int64,(rows,arena.layout.max_nodes+len(self.image.nodes))),
                (workspace.queue,t.int32,(rows,arena.layout.max_nodes))):
            if (not isinstance(value,t.Tensor) or value.device!=self.device
                    or value.dtype!=dtype or value.shape!=shape or not value.is_contiguous()):
                raise ValueError('Exact bound VM scratch buffers required')
        for value,dtype,shape in ((active,t.uint8,(rows,)),(gpu_deadlines,t.int64,(rows,))):
            if (not isinstance(value,t.Tensor) or value.device!=self.device
                    or value.dtype!=dtype or value.shape!=shape or not value.is_contiguous()):
                raise ValueError('Exact device selector/deadline vector required')
        if (not isinstance(argument_refs,t.Tensor) or argument_refs.device!=self.device
                or argument_refs.dtype!=t.int32 or argument_refs.ndim!=2
                or argument_refs.shape[0]!=rows or argument_refs.shape[1]<1
                or not argument_refs.is_contiguous()):
            raise ValueError('Exact device argument matrix required')
        # Deepcopy epochs restart at one for each VM call. Reused marks must not
        # retain an earlier call's epoch, including references to program constants.
        workspace.marks.zero_()
        b=self._buffers;l=arena.layout
        self.module.launch('vm_execute_device',[
            arena.nodes,arena.edges,arena.payload,arena.counts,argument_refs,argument_refs.shape[1],
            b['nodes'],b['edges'],b['payload'],b['instructions'],b['functions'],len(self.image.functions),
            len(self.image.nodes),len(self.image.edges),len(self.image.payload),
            workspace.frames,workspace.outputs,workspace.statuses,gpu_deadlines,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes,self.image.entries[entry],self.limits.steps,
            b['classes'],len(self.image.classes),b['modules'],len(self.image.modules),workspace.marks,
            workspace.queue,active],grid=(rows+self.rule_threads_per_block-1)//self.rule_threads_per_block,
            block=self.rule_threads_per_block,capture_safe=True)
        return DeviceCall(self,arena,workspace.outputs,workspace.statuses,
                          workspace.frames,None,(workspace.marks,workspace.queue))

    def load_graph_into_device(self, workspace, source, *, active, errors):
        """Copy used compact graph bytes into larger reusable rule scratch.

        No full-heap fork or host unpack occurs. References/aliases retain their
        exact row-local indices. Failed/inactive lanes expose an empty graph;
        consumers still gate rule calls with all upstream errors.
        """
        if self.closed:raise RuntimeError('VM closed')
        if not isinstance(workspace,DeviceVmWorkspace) or workspace.executor is not self:
            raise ValueError('Workspace must belong to this frozen VM')
        target=workspace.arena;self._validate_device_graph(source);self._validate_device_graph(target)
        t=self.torch;rows=source.rows
        if rows!=target.rows:raise ValueError('Matching graph lane rows required')
        for value,dtype in ((active,t.uint8),(errors,t.int32)):
            if (not isinstance(value,t.Tensor) or value.device!=self.device
                    or value.dtype!=dtype or value.shape!=(rows,) or not value.is_contiguous()):
                raise ValueError('Exact graph lane selector/error vector required')
        if any(getattr(source,key).untyped_storage().data_ptr()==getattr(target,key).untyped_storage().data_ptr()
               for key in ('nodes','edges','payload','counts','roots')):
            raise ValueError('Separate graph transfer storage required')
        a=source.layout;d=target.layout
        self.module.launch('vm_load_graph',[
            source.nodes,source.edges,source.payload,source.counts,source.roots,
            target.nodes,target.edges,target.payload,target.counts,target.roots,active,errors,
            rows,a.max_nodes,a.max_edges,a.max_payload_bytes,d.max_nodes,d.max_edges,d.max_payload_bytes],
            grid=rows,block=128,capture_safe=True)
        return target

    def prepare_compaction_workspace(self, source, destination):
        """Bind reusable maps/queue before capture; destination is scratch.

        The destination may be much smaller than the rule heap. Overflow remains
        a device error; no fields are truncated and failed output cannot be stored
        as a persistent state. Existing live leased graphs must not be targets.
        """
        if self.closed:raise RuntimeError('VM closed')
        t=self.torch
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('Prepare compaction storage before CUDA capture')
        self._validate_device_graph(source);self._validate_device_graph(destination)
        if source.rows!=destination.rows:raise ValueError('Matching compaction rows required')
        if any(getattr(source,k).data_ptr()==getattr(destination,k).data_ptr()
               for k in ('nodes','edges','payload','counts','roots')):
            raise ValueError('Separate compaction destination required')
        rows=source.rows;capacity=source.layout.max_nodes+len(self.image.nodes)
        return DeviceCompactionWorkspace(self,source,destination,
            t.empty((rows,capacity),dtype=t.int32,device=self.device),
            t.empty((rows,capacity),dtype=t.int32,device=self.device),
            t.zeros(rows,dtype=t.int32,device=self.device))

    def compact_into_device(self, call, workspace, *, active):
        """Keep reachable result fields/aliases on GPU before VM lanes reuse.

        Returns the bound workspace. Mapping converts source graph references
        into destination references; it is borrowed until the next compaction.
        Failed calls and inactive rows preserve the destination and set errors.
        """
        if self.closed:raise RuntimeError('VM closed')
        if (not isinstance(workspace,DeviceCompactionWorkspace) or workspace.executor is not self
                or call.executor is not self or call.arena is not workspace.source):
            raise ValueError('Compaction must bind the exact source VM lanes')
        source=workspace.source;target=workspace.arena;t=self.torch;rows=source.rows
        self._validate_device_graph(source);self._validate_device_graph(target)
        if rows!=target.rows:raise ValueError('Matching compaction rows required')
        if any(getattr(source,k).data_ptr()==getattr(target,k).data_ptr()
               for k in ('nodes','edges','payload','counts','roots')):
            raise ValueError('Separate compaction destination required')
        capacity=source.layout.max_nodes+len(self.image.nodes)
        for value,dtype,shape in ((active,t.uint8,(rows,)),
                (call.outputs,t.int32,(rows,)),(call.statuses,t.int32,(rows,4)),
                (workspace.mapping,t.int32,(rows,capacity)),(workspace.queue,t.int32,(rows,capacity)),
                (workspace.errors,t.int32,(rows,))):
            if (not isinstance(value,t.Tensor) or value.device!=self.device
                    or value.dtype!=dtype or value.shape!=shape or not value.is_contiguous()):
                raise ValueError('Exact device compaction selectors required')
        b=self._buffers;l=source.layout;d=target.layout
        self.module.launch('vm_compact_device',[
            source.nodes,source.edges,source.payload,source.counts,b['nodes'],b['edges'],b['payload'],
            len(self.image.nodes),len(self.image.edges),len(self.image.payload),call.outputs,
            target.nodes,target.edges,target.payload,target.counts,target.roots,
            workspace.mapping,workspace.queue,workspace.errors,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes,workspace.mapping.shape[1],
            d.max_nodes,d.max_edges,d.max_payload_bytes,active,call.statuses],
            grid=(rows+127)//128,block=128,capture_safe=True)
        return workspace
    def close(self):
        if self.closed:return
        self.module.close();self._buffers.clear();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
