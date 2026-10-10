"""Checked CPython 3.10 exception-block protocol on the CUDA VM.

Source protocol: python/cpython v3.10.14 Python/ceval.c, exception_unwind and
POP_EXCEPT. This independent implementation preserves type/args/cause/context;
traceback objects, lasti introspection and custom exception classes are omitted.
VM capacity, deadline, unsupported capabilities and numeric faults stay fatal.
"""
EXCEPTION_NAMES=('BaseException','Exception','ValueError','KeyError','IndexError','TypeError','RuntimeError','StopIteration','ZeroDivisionError','OverflowError','AttributeError','LookupError','AssertionError')
EXCEPTION_OPS=frozenset(('SETUP_FINALLY','POP_BLOCK','POP_EXCEPT','RERAISE','RAISE_VARARGS','JUMP_IF_NOT_EXC_MATCH','ROT_FOUR','ROT_N'))


def emit_exception_source(image):
    identities=[];refs=[]
    for code,name in enumerate(EXCEPTION_NAMES):
        identities.append(f'case {-4000-code}:return {code};')
        if name in image.intrinsic_names:
            index=image.intrinsic_names.index(name);identities.append(f'case {index}:return {code};')
            found=[-1-i for i,n in enumerate(image.nodes) if n[0]==15 and n[3]==index]
            if found:refs.append(f'case {code}:return {found[0]};')
    source=r'''
const int T_EXCEPTION=23,T_BLOCK=44,UNHANDLED_PYTHON_EXCEPTION=110;
__device__ inline void stdlib_sort_unwind(Context* c,int accumulator);
__device__ inline int exception_class_code(Context* c,int ref){
 if(kind(c,ref)!=T_NATIVE)return -1;switch(node(c,ref)[3]){ @IDENTITIES@ default:return -1;}
}
__device__ inline int exception_type(Context* c,int code){
 if(code<0||code>=13){c->error=BAD_PROGRAM;return MISSING;}
 switch(code){ @REFS@ default:return meta(c,T_NATIVE,0,0,-4000-code);}
}
// Frames store only their OWN exception; inheritance follows the active caller chain.
// This also lets a suspended generator inherit its new caller on resumption.
__device__ inline int active_exception(Context* c,int id){
 for(int depth=0;id>=0&&depth<VM_FRAMES;depth++){
  if(id>=VM_FRAMES){c->error=BAD_PROGRAM;return NIL;}int* f=frame(c,id);
  if(f[13]!=NIL)return f[13];id=f[3];
 }return NIL;
}
__device__ inline int exception_create(Context* c,int type,int arguments,int cause=MISSING){
 if(exception_class_code(c,type)<0||kind(c,arguments)!=TAG_TUPLE){c->error=BAD_CALL;return MISSING;}
 int refs[5]={type,arguments,cause==MISSING?NIL:cause,NIL,NIL};int ref=make_tuple(&c->h,refs,5);
 int* n=mutable_node(c,ref);if(n){n[0]=T_EXCEPTION;n[3]=cause!=MISSING;}return ref;
}
__device__ inline int exception_normalize(Context* c,int ref){
 if(kind(c,ref)==T_EXCEPTION){if(exception_class_code(c,edge(c,ref,0))>=0)return ref;}
 else if(exception_class_code(c,ref)>=0)return exception_create(c,ref,make_tuple(&c->h,0,0));
 c->error=BAD_CALL;return MISSING;
}
__device__ inline int exception_call(Context* c,int callable,const int* args,int nargs,int* out){
 if(exception_class_code(c,callable)<0)return 0;
 *out=exception_create(c,callable,make_tuple(&c->h,args,nargs));return 1;
}
__device__ inline int exception_attribute(Context* c,int owner,int name){
 if(kind(c,owner)!=T_EXCEPTION)return MISSING;
 if(vm_text_equals(c,name,"args"))return edge(c,owner,1);
 if(vm_text_equals(c,name,"__cause__"))return edge(c,owner,2);
 if(vm_text_equals(c,name,"__context__"))return edge(c,owner,3);
 if(vm_text_equals(c,name,"__class__"))return edge(c,owner,0);
 if(vm_text_equals(c,name,"__suppress_context__"))return node(c,owner)[3]?TRUE_REF:FALSE_REF;
 return MISSING;
}
__device__ inline void exception_set(Context* c,int ref,int field,int value){
 int* n=mutable_node(c,ref);if(!n||n[0]!=T_EXCEPTION||field<0||field>=5){c->error=BAD_PROGRAM;return;}
 c->h.edges[(n[1]+field)*2]=value;
}
__device__ inline void raise_exception(Context* c,int ref,int cause=MISSING){
 int exception=exception_normalize(c,ref);if(c->error||c->h.error)return;
 int previous=active_exception(c,c->current);
 if(previous!=NIL){int value=edge(c,previous,1);if(value!=exception)exception_set(c,exception,3,value);}
 if(cause!=MISSING){
  int normalized=kind(c,cause)==TAG_NONE?NIL:exception_normalize(c,cause);
  if(c->error||c->h.error)return;exception_set(c,exception,2,normalized);mutable_node(c,exception)[3]=1;
 }
 c->raised=exception;
}
__device__ inline int exception_match_one(Context* c,int actual,int wanted){
 int a=exception_class_code(c,actual),b=exception_class_code(c,wanted);
 if(a<0||b<0){c->error=BAD_CALL;return 0;}
 return a==b||b==0||(b==1&&a!=0)||(b==11&&(a==3||a==4));
}
__device__ inline int exception_matches(Context* c,int actual,int wanted){
 int pending[64],top=1,matched=0;pending[0]=wanted;
 while(top&&!c->error&&!c->h.error){int ref=pending[--top];
  if(kind(c,ref)==TAG_TUPLE){int count=length(c,ref);if(top+count>64){c->error=STACK_OVERFLOW;return 0;}
   for(int i=0;i<count;i++)pending[top++]=edge(c,ref,i);
  }else if(exception_match_one(c,actual,ref))matched=1;
 }return c->error||c->h.error?0:matched;
}
__device__ inline void block_push(Context* c,int* f,int type,int target,int level){
 if(level<0||level>VM_STACK){c->error=BAD_PROGRAM;return;}
 int fields[4]={type,target,level,f[12]};int ref=make_tuple(&c->h,fields,4);int* n=mutable_node(c,ref);
 if(n){n[0]=T_BLOCK;f[12]=ref;}
}
__device__ inline int block_pop(Context* c,int* f){
 int ref=f[12];const int* n=node(c,ref);
 if(!n||n[0]!=T_BLOCK||n[2]!=4){c->error=BAD_PROGRAM;return MISSING;}
 f[12]=edge(c,ref,3);return ref;
}
__device__ inline void restore_handler(Context* c,int* f,int level,int unwinding){
 if(f[2]<level+3||(!unwinding&&f[2]>level+4)){c->error=BAD_PROGRAM;return;}
 if(unwinding)while(f[2]>level+3)pop(c,f);
 int type=pop(c,f),value=pop(c,f),tb=pop(c,f);
 if(type==NIL&&value==NIL&&tb==NIL)f[13]=NIL;
 else{int refs[3]={tb,value,type};f[13]=make_tuple(&c->h,refs,3);}
}
__device__ inline void unwind_exception(Context* c){
 while(c->current>=0&&c->raised!=MISSING&&!c->error&&!c->h.error){
  int* f=frame(c,c->current);
  while(f[12]!=NIL&&!c->error&&!c->h.error){
   int block=block_pop(c,f);if(c->error)break;
   int type=edge(c,block,0),target=edge(c,block,1),level=edge(c,block,2);
   if(type==1){restore_handler(c,f,level,1);continue;}
   if(type!=0||level>f[2]){c->error=BAD_PROGRAM;return;}
   while(f[2]>level)pop(c,f);
   int previous=f[13],old[3]={NIL,NIL,NIL};
   if(previous!=NIL)for(int i=0;i<3;i++)old[i]=edge(c,previous,i);
   block_push(c,f,1,f[1]-1,level);
   for(int i=0;i<3;i++)push(c,f,old[i]);
   int exc=c->raised,klass=edge(c,exc,0),refs[3]={NIL,exc,klass};f[13]=make_tuple(&c->h,refs,3);
   push(c,f,NIL);push(c,f,exc);push(c,f,klass);f[1]=target;c->raised=MISSING;return;
  }
  if(c->error||c->h.error)return;
  if(f[10]>=0){
   int* gen=mutable_node(c,f[10]);if(gen)gen[3]=-1;
   if(exception_class_code(c,edge(c,c->raised,0))==7){
    int original=c->raised,text=make_utf8(&c->h,"generator raised StopIteration",30),args=make_tuple(&c->h,&text,1);
    c->raised=exception_create(c,exception_type(c,6),args,original);exception_set(c,c->raised,3,original);
   }
  }
  if(f[5]==7)stdlib_sort_unwind(c,f[6]);
  if(c->error||c->h.error)return;
  int caller=f[3];f[4]=0;c->current=caller;
 }
 if(c->raised!=MISSING&&!c->error&&!c->h.error)c->error=UNHANDLED_PYTHON_EXCEPTION;
}
__device__ inline int handle_exceptions(Context* c,int* f,int op,int arg,int value,int target){
 switch(op){
 case OP_SETUP_FINALLY:block_push(c,f,0,target,f[2]);return 1;
 case OP_POP_BLOCK:{int b=block_pop(c,f);if(!c->error&&edge(c,b,0)!=0)c->error=BAD_PROGRAM;return 1;}
 case OP_POP_EXCEPT:{int b=block_pop(c,f);if(c->error)return 1;if(edge(c,b,0)!=1)c->error=BAD_PROGRAM;else restore_handler(c,f,edge(c,b,2),0);return 1;}
 case OP_JUMP_IF_NOT_EXC_MATCH:{int wanted=pop(c,f),actual=pop(c,f);if(!exception_matches(c,actual,wanted))f[1]=target;return 1;}
 case OP_RAISE_VARARGS:{
  if(arg<0||arg>2){c->error=BAD_PROGRAM;return 1;}
  if(!arg){int active=active_exception(c,c->current);
   if(active!=NIL)c->raised=edge(c,active,1);
   else{int text=make_utf8(&c->h,"No active exception to reraise",30);c->raised=exception_create(c,exception_type(c,6),make_tuple(&c->h,&text,1));}
  }else{int cause=arg==2?pop(c,f):MISSING,exc=pop(c,f);raise_exception(c,exc,cause);}return 1;
 }
 case OP_RERAISE:{int type=pop(c,f),exc=pop(c,f);pop(c,f);if(kind(c,exc)!=T_EXCEPTION||exception_class_code(c,type)<0)c->error=BAD_PROGRAM;else c->raised=exc;return 1;}
 case OP_ROT_FOUR:case OP_ROT_N:{int count=op==OP_ROT_FOUR?4:arg;
  if(count<1||count>f[2]){c->error=BAD_PROGRAM;return 1;}int top=f[SBASE+f[2]-1];
  for(int i=f[2]-1;i>f[2]-count;i--)f[SBASE+i]=f[SBASE+i-1];f[SBASE+f[2]-count]=top;return 1;
 }
 default:return 0;
 }
}
'''
    return source.replace('@IDENTITIES@',''.join(identities)).replace('@REFS@',''.join(refs))
