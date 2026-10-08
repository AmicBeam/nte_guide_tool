"""Actual CUDA interpreter for frozen rule IR, with explicit capability gates.

The current instruction set is a checked execution subset. A missing capability
raises before launch; this module never calls the Python engine as a fallback.
Full duel simulation is NOT available until all simulate/observe/legality
capabilities and formal-rule GPU parity tests pass.
"""
from dataclasses import dataclass
import time
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

SUPPORTED_OPS=frozenset(('NOP','EXTENDED_ARG','LOAD_CONST','LOAD_FAST','STORE_FAST','LOAD_GLOBAL',
    'DELETE_FAST','LOAD_DEREF','STORE_DEREF','LOAD_CLOSURE','BINARY_SUBSCR','COMPARE_OP','CONTAINS_OP','IS_OP',
    'UNARY_NOT','UNARY_NEGATIVE','BINARY_ADD','BINARY_SUBTRACT','BINARY_MULTIPLY',
    'BUILD_TUPLE','BUILD_LIST','MAKE_FUNCTION','GET_ITER','FOR_ITER','GEN_START','YIELD_VALUE',
    'CALL_FUNCTION','CALL_FUNCTION_KW','CALL_FUNCTION_EX','POP_TOP','DUP_TOP','ROT_TWO','JUMP_ABSOLUTE','JUMP_FORWARD',
    'POP_JUMP_IF_FALSE','POP_JUMP_IF_TRUE','JUMP_IF_FALSE_OR_POP','JUMP_IF_TRUE_OR_POP','RETURN_VALUE')) | SUPPORTED_COLLECTION_OPS | OBJECT_OPS | NUMERIC_OPS
SUPPORTED_INTRINSICS=frozenset(('next','object',*SUPPORTED_BUILTIN_NAMES,*SUPPORTED_REDUCER_NAMES,*SUPPORTED_RANDOM_NAMES))


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


def emit_vm_source(image,limits=VmLimits()):
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
const int HEADER=12, LBASE=HEADER, SBASE=LBASE+VM_LOCALS, CBASE=SBASE+VM_STACK, STRIDE=CBASE+VM_CELLS;
const int MISSING=(-2147483647-1), NIL=-1, FALSE_REF=-2, TRUE_REF=-3;
const int T_FUNCTION=11,T_CODE=12,T_CELL=16,T_ITER=17,T_GEN=19,T_DYNFN=33,T_NATIVE=15;
const int NATIVE_NEXT={image.intrinsic_names.index('next') if 'next' in image.intrinsic_names else -1};
const int NATIVE_OBJECT={image.intrinsic_names.index('object') if 'object' in image.intrinsic_names else -2};
enum VmError {{ BAD_PROGRAM=100,STACK_OVERFLOW=101,FRAME_OVERFLOW=102,UNBOUND_LOCAL=103,
 BAD_CALL=104,UNSUPPORTED_OP=105,UNSUPPORTED_TYPE=106,STEP_LIMIT=107,DEADLINE=108,STOP_ITERATION=109 }};
'''
    return '#define BMT_DEVICE __device__\n'+emit_random_source()+emit_vm_heap_source()+prefix+_VM_SOURCE.replace('// COLLECTION_HANDLERS',emit_collection_source()).replace('// OBJECT_HANDLERS',emit_object_source(image)).replace('// BUILTIN_HANDLERS',emit_builtin_source(image.intrinsic_names)).replace('// NUMERIC_HANDLERS',emit_numeric_source()).replace('// REDUCER_HANDLERS',emit_reducer_source(image.intrinsic_names)).replace('// FRAME_GC_HANDLERS',emit_frame_gc_source()).replace('// RANDOM_HANDLERS',emit_random_vm_source(image.intrinsic_names))


_VM_SOURCE=r'''
// Frame: function,pc,sp,caller,status,return_kind,return_target,default,globals,closure,generator,cells_count.
struct Context {
 Heap h; const int* classes;int class_count;const int* modules;int module_count; const int* instructions; const int* functions; int function_count; int* frames;
 int* gc_marks;int* gc_queue;int current; int output; int error; int steps; int step_limit; long long deadline;
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
__device__ inline int length(Context* c,int ref){
 const int* n=node(c,ref);if(!n)return 0;
 if(n[0]==TAG_NONE)return 0;
 if(n[0]==TAG_LIST || n[0]==TAG_TUPLE || n[0]==TAG_DICT || (n[0]>=8&&n[0]<=10) || n[0]==21 || n[0]==22)return n[2];
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
__device__ inline int truth_value(Context* c,int value){
 int tag=kind(c,value);if(tag==26)return range_size(c,value)!=0;if(tag==21||tag==22)return length(c,value)!=0;if(tag>=11)return 1;return duel_vm_heap::truth(&c->h,value);
}
// FRAME_GC_HANDLERS
__device__ inline int new_frame(Context* c,int fid,const int* args,int nargs,int caller,int return_kind,
 int globals_override=MISSING,int closure_override=MISSING,int defaults_override=MISSING,
 int kw_names=MISSING,const int* kw_values=0,int nkw=0,int kwdefaults_override=MISSING){
 const int* fn=function(c,fid);if(!fn)return -1;
 if(fn[5]>VM_LOCALS || fn[6]>VM_STACK){c->error=BAD_CALL;return -1;}
 int id=-1;for(int i=0;i<VM_FRAMES;i++)if(frame(c,i)[4]==0){id=i;break;}
 if(id<0){
  collect_suspended(c,args,nargs,globals_override,closure_override,defaults_override,kw_names,kw_values,nkw,kwdefaults_override);
  if(c->error||c->h.error)return -1;
  for(int i=0;i<VM_FRAMES;i++)if(frame(c,i)[4]==0){id=i;break;}
  if(id<0){c->error=FRAME_OVERFLOW;return -1;}
 }
 int* f=frame(c,id);for(int i=0;i<STRIDE;i++)f[i]=MISSING;
 f[0]=fid;f[1]=fn[0];f[2]=0;f[3]=caller;f[4]=1;f[5]=return_kind;f[6]=-1;f[7]=MISSING;
 f[8]=globals_override==MISSING?fn[8]:globals_override;f[9]=closure_override==MISSING?fn[15]:closure_override;f[10]=-1;
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
 int tag=kind(c,owner);if(tag==T_ITER || tag==T_GEN)return owner;
 if(tag!=26 && tag!=TAG_LIST && tag!=TAG_TUPLE && tag!=TAG_DICT && !(tag>=8&&tag<=10)){c->error=UNSUPPORTED_TYPE;return MISSING;}
 return meta(c,T_ITER,owner,0,0);
}
__device__ inline int iterator_next(Context* c,int ref){
 int* it=mutable_node(c,ref);if(!it || it[0]!=T_ITER){c->error=BAD_CALL;return MISSING;}
 int owner=it[1],index=it[2];const int* n=node(c,owner);if(!n)return MISSING;
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
__device__ inline void call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names=MISSING,const int* kw_values=0,int nkw=0){
 const int* n=node(c,callable);if(!n)return;
 int bound_args[VM_STACK];
 if(n[0]==18){
  if(nargs+1>VM_STACK){c->error=STACK_OVERFLOW;return;}
  bound_args[0]=edge(c,callable,1);for(int i=0;i<nargs;i++)bound_args[i+1]=args[i];
  callable=edge(c,callable,0);args=bound_args;nargs++;n=node(c,callable);if(!n)return;
 }
 if(n[0]==13){
  int cid=n[3];if(!class_constructible(cid)){c->error=UNSUPPORTED_TYPE;return;}
  int attrs=make_dict(&c->h),values[2]={callable,attrs},instance=make_tuple(&c->h,values,2);
  int* obj=mutable_node(c,instance);if(!obj)return;obj[0]=29;
  int name=make_utf8(&c->h,"__init__",8),category=-1,init=lookup_class(c,cid,name,&category);
  if(init==MISSING){if(nargs||nkw)c->error=BAD_CALL;else push(c,frame(c,caller),instance);return;}
  if(category!=2||kind(c,init)!=T_FUNCTION||nargs+1>VM_STACK){c->error=UNSUPPORTED_TYPE;return;}
  int args_with_self[VM_STACK];args_with_self[0]=instance;for(int i=0;i<nargs;i++)args_with_self[i+1]=args[i];
  int id=new_frame(c,node(c,init)[3],args_with_self,nargs+1,caller,4,MISSING,MISSING,MISSING,kw_names,kw_values,nkw);
  if(id>=0){frame(c,id)[6]=instance;c->current=id;}return;
 }
 if(reducer_call(c,callable,args,nargs,caller,kw_names,kw_values,nkw))return;
 int native_result=MISSING;
 if(nkw && (n[0]==T_NATIVE||n[0]==34||n[0]==41)){c->error=BAD_CALL;return;}
 if(random_call(c,callable,args,nargs,&native_result)||native_builtin_call(c,callable,args,nargs,&native_result)){
  if(!c->error&&!c->h.error)push(c,frame(c,caller),native_result);return;
 }
 if(n[0]==T_NATIVE){
  if(n[3]==NATIVE_OBJECT){if(nargs||nkw)c->error=BAD_CALL;else push(c,frame(c,caller),meta(c,36,0,0,0));return;}
  if(nkw || n[3]!=NATIVE_NEXT || nargs<1 || nargs>2){c->error=BAD_CALL;return;}
  int type=kind(c,args[0]);int fallback=nargs==2?args[1]:MISSING;
  if(type==T_GEN)resume_generator(c,args[0],caller,1,-1,fallback);
  else if(type==T_ITER){int value=iterator_next(c,args[0]);if(value==MISSING)value=fallback;
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
  if(f[10]<0 || caller<0){c->error=BAD_CALL;return;}f[4]=2;if(mode==5){reducer_resume(c,f[10],caller,target,value,0);return;}push(c,frame(c,caller),value);c->current=caller;return;
 }
 if(f[10]>=0){
  int* gen=mutable_node(c,f[10]);if(gen)gen[3]=-1;
  f[4]=0;c->current=caller;
  if(caller<0){c->error=BAD_CALL;return;}
  if(mode==5){reducer_resume(c,f[10],caller,target,MISSING,1);return;}
  if(mode==2){pop(c,frame(c,caller));frame(c,caller)[1]=target;}
  else if(fallback!=MISSING)push(c,frame(c,caller),fallback);else c->error=STOP_ITERATION;
 }else{
  f[4]=0;c->current=caller;
  if(mode==4){if(kind(c,value)!=TAG_NONE)c->error=BAD_CALL;else if(caller<0)c->error=BAD_PROGRAM;else push(c,frame(c,caller),target);}
  else if(caller<0)c->output=value;else push(c,frame(c,caller),value);
 }
}
// COLLECTION_HANDLERS
__device__ inline int subscript(Context* c,int owner,int key){
 int tag=kind(c,owner);
 if(tag==TAG_DICT || (tag>=8&&tag<=10)){
  int value=dict_get(&c->h,owner,key,MISSING);if(value==MISSING && !c->h.error)c->h.error=ERROR_KEY_NOT_FOUND;return value;
 }
 if(tag==TAG_LIST || tag==TAG_TUPLE){long long index=0;if(read_int64(&c->h,key,&index)<0)return MISSING;
  if(index<(-2147483647-1LL)||index>2147483647LL){c->error=BAD_PROGRAM;return MISSING;}return list_get(&c->h,owner,(int)index);}
 c->error=UNSUPPORTED_TYPE;return MISSING;
}
__device__ inline int contains(Context* c,int needle,int owner){
 int tag=kind(c,owner);if(tag==TAG_DICT || (tag>=8&&tag<=10))return dict_find(&c->h,owner,needle)>=0;
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
 if(mode==2 || mode==3){int eq=vm_equal(c,a,b);return mode==2?eq:!eq;}
 int ta=kind(c,a),tb=kind(c,b),cmp=0;
 if(ta==TAG_STR&&tb==TAG_STR)cmp=str_compare(&c->h,a,b);
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
 while(c->current>=0 && !c->error && !c->h.error){
  if(++c->steps>c->step_limit){c->error=STEP_LIMIT;break;}
  if(c->deadline && (c->steps&63)==0 && clock_ns()>=(unsigned long long)c->deadline){c->error=DEADLINE;break;}
  int* f=frame(c,c->current);const int* fn=function(c,f[0]);if(!fn)break;
  int pc=f[1];if(pc<fn[0]||pc>=fn[0]+fn[1]){c->error=BAD_PROGRAM;break;}
  const int* ins=c->instructions+pc*4;int op=ins[0],arg=ins[1],value=ins[2],target=ins[3];f[1]++;
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
    int attr=object_attribute(c,owner,value,&getter);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=builtin_attribute(c,owner,value);
    if(attr==MISSING&&!c->error&&!c->h.error)attr=random_attribute(c,owner,value);
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
   case OP_BINARY_TRUE_DIVIDE:case OP_BINARY_FLOOR_DIVIDE:case OP_BINARY_AND:{
    int b=pop(c,f),a=pop(c,f);push(c,f,numeric_binary(c,a,b,op));break;
   }
   case OP_JUMP_ABSOLUTE:case OP_JUMP_FORWARD:f[1]=target;break;
   case OP_POP_JUMP_IF_FALSE:case OP_POP_JUMP_IF_TRUE:{int yes=truth_value(c,pop(c,f));if(yes==(op==OP_POP_JUMP_IF_TRUE))f[1]=target;break;}
   case OP_JUMP_IF_FALSE_OR_POP:case OP_JUMP_IF_TRUE_OR_POP:{int yes=truth_value(c,peek(c,f));if(yes==(op==OP_JUMP_IF_TRUE_OR_POP))f[1]=target;else pop(c,f);break;}
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
 int rows,int max_nodes,int max_edges,int max_payload,int entry,int step_limit,const int* classes,int class_count,const int* modules,int module_count,int* gc_marks,int* gc_queue){
 using namespace duel_rule_vm;int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(tid>=rows)return;
 Context c;c.gc_marks=gc_marks+(long long)tid*max_nodes;c.gc_queue=gc_queue+(long long)tid*max_nodes;c.classes=classes;c.class_count=class_count;c.modules=modules;c.module_count=module_count;c.h.nodes=nodes+(long long)tid*max_nodes*4;c.h.edges=edges+(long long)tid*max_edges*2;
 c.h.payload=payload+(long long)tid*max_payload;c.h.counts=counts+tid*3;
 c.h.max_nodes=max_nodes;c.h.max_edges=max_edges;c.h.max_payload=max_payload;
 c.h.constant_nodes=constants_nodes;c.h.constant_edges=constants_edges;c.h.constant_payload=constants_payload;
 c.h.constant_node_count=constant_node_count;c.h.constant_edge_count=constant_edge_count;c.h.constant_payload_count=constant_payload_count;c.h.error=0;
 c.instructions=instructions;c.functions=functions;c.function_count=function_count;c.frames=frames+(long long)tid*VM_FRAMES*STRIDE;
 for(int i=0;i<VM_FRAMES;i++)frame(&c,i)[4]=0;
 c.current=-1;c.output=MISSING;c.error=0;c.steps=0;c.step_limit=step_limit;c.deadline=deadlines[tid];
 c.current=new_frame(&c,entry,args+(long long)tid*nargs,nargs,-1,3);
 if(c.current>=0)execute(&c);
 outputs[tid]=c.output;statuses[tid*4]=c.error?c.error:c.h.error;
 statuses[tid*4+1]=c.current>=0?frame(&c,c.current)[0]:entry;
 statuses[tid*4+2]=c.current>=0?frame(&c,c.current)[1]:-1;statuses[tid*4+3]=c.steps;
}
'''


class DeviceGraph:
    """Owned GPU graph buffers. Ingress copying is not rule computation."""
    def __init__(self,layout,nodes,edges,payload,counts,roots):
        self.layout=layout;self.nodes=nodes;self.edges=edges;self.payload=payload;self.counts=counts;self.roots=roots
    @property
    def rows(self):return int(self.roots.shape[0])
    def fork(self):return DeviceGraph(self.layout,*(getattr(self,k).clone() for k in ('nodes','edges','payload','counts','roots')))


class DeviceCall:
    def __init__(self,executor,arena,outputs,statuses,frames,event,scratch_buffers=()):
        self.executor=executor;self.arena=arena;self.outputs=outputs;self.statuses=statuses;self.frames=frames;self.event=event;self.scratch_buffers=scratch_buffers
    def check(self):
        self.event.synchronize();statuses=self.statuses.cpu().numpy()
        bad=np.flatnonzero(statuses[:,0])
        if len(bad):
            row=int(bad[0]);code,fid,pc,steps=map(int,statuses[row]);name=self.executor.image.function_ids[fid] if 0<=fid<len(self.executor.image.function_ids) else 'unknown'
            raise VmExecutionError(f'CUDA VM row {row}: status={code}, function={name}, pc={pc}, steps={steps}')
        return statuses
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
    def __init__(self,image,*,device='cuda',limits=VmLimits()):
        if not isinstance(image,VmImage):raise TypeError('Numeric VM image required')
        image.verify()
        if not isinstance(limits,VmLimits):raise TypeError('VmLimits required')
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Actual CUDA required, no CPU VM fallback')
        from .cuda_runtime import CudaModule
        self.image=image;self.limits=limits;self.torch=torch;self.device=torch.device(device)
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        self._buffers={k:torch.tensor(np.array(getattr(image,k),copy=True),device=self.device) for k in
            ('instructions','functions','nodes','edges','payload','classes','modules')}
        signature=['int32*','int32*','uint8*','int32*','int32*','int32',
            'int32*','int32*','uint8*','int32*','int32*','int32','int32','int32','int32',
            'int32*','int32*','int32*','int64*','int32','int32','int32','int32','int32','int32','int32*','int32','int32*','int32','int32*','int32*']
        compact_signature=['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*',
            'int32','int32','int32','int32*','int32*','int32*','uint8*','int32*','int32*','int32*','int32*','int32*',
            'int32','int32','int32','int32','int32']
        self.module=CudaModule(emit_vm_source(image,limits),{'vm_execute':signature,'vm_clock':['int64*'],'vm_compact':compact_signature},device=str(self.device))
        self.closed=False
    def capability(self,entry):
        missing=self.image.missing_capabilities(entry,opcodes=SUPPORTED_OPS,intrinsics=SUPPORTED_INTRINSICS)
        return dict(entry=entry,ready=not any(missing.values()),missing=missing,full_duel_rules=False,actual_backend='cuda')
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
        scratch=arena.fork();stride=12+self.limits.locals+self.limits.stack+self.limits.cells
        frames=torch.empty((rows,self.limits.frames,stride),dtype=torch.int32,device=self.device)
        outputs=torch.empty(rows,dtype=torch.int32,device=self.device);statuses=torch.zeros((rows,4),dtype=torch.int32,device=self.device)
        b=self._buffers;l=arena.layout
        gc_marks=torch.empty((rows,l.max_nodes),dtype=torch.int32,device=self.device);gc_queue=torch.empty_like(gc_marks)
        event=self.module.launch('vm_execute',[scratch.nodes,scratch.edges,scratch.payload,scratch.counts,argument_refs,argument_refs.shape[1],
            b['nodes'],b['edges'],b['payload'],b['instructions'],b['functions'],len(self.image.functions),
            len(self.image.nodes),len(self.image.edges),len(self.image.payload),frames,outputs,statuses,gpu_deadlines,
            rows,l.max_nodes,l.max_edges,l.max_payload_bytes,self.image.entries[entry],self.limits.steps,b['classes'],len(self.image.classes),b['modules'],len(self.image.modules),gc_marks,gc_queue],grid=(rows+127)//128,block=128)
        return DeviceCall(self,scratch,outputs,statuses,frames,event,(gc_marks,gc_queue))
    def close(self):
        if self.closed:return
        self.module.close();self._buffers.clear();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
