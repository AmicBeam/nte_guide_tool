"""Per-execution ContextVar bindings and the exact no-patches context manager.

Integration contract (the caller owns the VM and WITH continuations):

* Add ``int context_values`` to Context and initialize it to MISSING for every
  root invocation. This lazily becomes a heap list of [variable, value] tuples;
  variables compare by reference identity, never their names or defaults.
* Include context_values among frame-GC roots. Traverse tag 43 and tag 25 as
  sequences, and tags 46/47 through n[1]. A token holds its binding-table ref,
  which both preserves reachability and detects a different execution context.
* Emit this source after exception helpers and vm_text_equals. Invoke
  context_call before generic native keyword rejection. Do not push its result
  while c->raised is present. Normal and exceptional WITH exits must invoke the
  manager's __exit__; VM fatal capacity/deadline errors remain fatal.
* Variable tag 24 keeps [name, default]. n[3]==1 means known default present,
  n[3]==2 known absent, and n[3]==0 is the historical frozen representation:
  a non-None default is known, while None is ambiguous and fails explicitly if
  an unbound get() would need that default. The image emitter can retain presence
  in this extra word; no constant graph mutation occurs here.

Token tag 25 holds [variable, old binding or MISSING, binding table], with used
in n[3]. Manager tag 43 holds [variable, token or MISSING], with lifecycle in
n[3]. Binding tags 46/47 store owner in n[1] and method code in n[3]. Managers
implement only the exact frozen simulation._without_patches intrinsic. Generic
contextlib managers, context copies/tasks, Token.MISSING introspection and
context-manager reuse are outside this component's scope and fail explicitly.
There are no process-wide mutable CUDA bindings and no CPU callbacks.
"""

SUPPORTED_CONTEXT_NAMES = (
    'contextvars.ContextVar',
    'app.modules.card_game.engine.duel_v2.simulation._without_patches',
)
_SIMULATION_MODULE = 'app.modules.card_game.engine.duel_v2.simulation'


def _image_text(image, ref):
    """Read an immutable program-name literal, never a runtime rule value."""
    if ref >= 0 or -1 - ref >= len(image.nodes):
        return None
    tag, offset, count, _ = map(int, image.nodes[-1 - ref])
    if tag != 4 or offset < 0 or count < 0 or offset + count > len(image.payload):
        return None
    return image.payload[offset:offset + count].tobytes().decode('utf-8')


def _image_binding(image, mapping, name):
    if mapping >= 0 or -1 - mapping >= len(image.nodes):
        return None
    tag, offset, count, _ = map(int, image.nodes[-1 - mapping])
    if tag != 7 or offset < 0 or count < 0 or offset + count > len(image.edges):
        return None
    found = []
    for key, value in image.edges[offset:offset + count]:
        if _image_text(image, int(key)) == name:
            found.append(int(value))
    if len(found) > 1:
        raise ValueError('Duplicate frozen ContextVar binding')
    return found[0] if found else None


def emit_context_source(image) -> str:
    """Emit CUDA context primitives using only identities in the frozen image."""
    image.verify()
    names = tuple(image.intrinsic_names)
    ids = [names.index(name) if name in names else -7101 - i
           for i, name in enumerate(SUPPORTED_CONTEXT_NAMES)]
    candidates = set()
    for index, name in enumerate(image.module_names):
        if name == _SIMULATION_MODULE:
            ref = _image_binding(image, int(image.modules[index][0]), '_collect_patches')
            if ref is not None:
                candidates.add(ref)
    prefix = 'func:' + _SIMULATION_MODULE + ':collecting_patches:'
    for index, identity in enumerate(image.function_ids):
        if identity.startswith(prefix):
            ref = _image_binding(image, int(image.functions[index][8]), '_collect_patches')
            if ref is not None:
                candidates.add(ref)
    if len(candidates) > 1:
        raise ValueError('Conflicting frozen simulation ContextVar identities')
    collect = next(iter(candidates)) if candidates else -2147483648
    if collect != -2147483648:
        if collect >= 0 or int(image.nodes[-1 - collect][0]) != 24:
            raise ValueError('Frozen simulation patch switch is not a ContextVar')
    return (_SOURCE.replace('@VAR_ID@', str(ids[0]))
            .replace('@MANAGER_ID@', str(ids[1]))
            .replace('@COLLECT_REF@', str(collect)))


_SOURCE = r'''
const int T_CONTEXT_VAR=24,T_CONTEXT_TOKEN=25,T_PATCH_MANAGER=43,
 T_CONTEXT_BOUND=46,T_CONTEXT_MANAGER_BOUND=47;
const int NATIVE_CONTEXT_VAR=@VAR_ID@,NATIVE_WITHOUT_PATCHES=@MANAGER_ID@,
 CONTEXT_COLLECT_PATCHES=@COLLECT_REF@;
enum ContextMethod { CONTEXT_GET=1,CONTEXT_SET=2,CONTEXT_RESET=3,
 CONTEXT_ENTER=1,CONTEXT_EXIT=2 };
__device__ inline int context_error_arg(Context* c,int code,int arg){
 int args=make_tuple(&c->h,&arg,1);if(c->h.error)return MISSING;
 int exc=exception_create(c,exception_type(c,code),args);
 if(!c->error&&!c->h.error)raise_exception(c,exc);return MISSING;
}
__device__ inline int context_error(Context* c,int code,const char* message){
 int size=0;while(message[size])size++;int arg=make_utf8(&c->h,message,size);
 if(c->h.error)return MISSING;return context_error_arg(c,code,arg);
}
__device__ inline int context_sequence(Context* c,int tag,const int* refs,int count,int extra){
 int ref=make_tuple(&c->h,refs,count);if(c->h.error)return MISSING;
 int* n=mutable_node(c,ref);if(!n)return MISSING;n[0]=tag;n[3]=extra;return ref;
}
__device__ inline int context_variable(Context* c,int variable){
 const int* n=node(c,variable);if(!n)return 0;
 if(n[0]!=T_CONTEXT_VAR||n[2]!=2){context_error(c,5,"Expected a ContextVar");return 0;}
 return 1;
}
__device__ inline int context_table(Context* c){
 if(c->context_values==MISSING){
  int ref=make_list(&c->h);if(c->h.error)return MISSING;c->context_values=ref;
 }
 if(kind(c,c->context_values)!=TAG_LIST){c->error=BAD_PROGRAM;return MISSING;}
 return c->context_values;
}
__device__ inline int context_binding(Context* c,int variable){
 if(c->context_values==MISSING)return MISSING;
 const int* table=node(c,c->context_values);
 if(!table||table[0]!=TAG_LIST||table[2]<0){c->error=BAD_PROGRAM;return MISSING;}
 for(int i=0;i<table[2];i++){
  int entry=edge(c,c->context_values,i);const int* n=node(c,entry);
  if(!n||n[0]!=TAG_TUPLE||n[2]!=2){c->error=BAD_PROGRAM;return MISSING;}
  if(edge(c,entry,0)==variable)return entry;
 }return MISSING;
}
__device__ inline int context_get(Context* c,int variable,int fallback=MISSING){
 if(!context_variable(c,variable))return MISSING;
 int binding=context_binding(c,variable);if(c->error||c->h.error)return MISSING;
 if(binding!=MISSING){int value=edge(c,binding,1);if(value!=MISSING)return value;}
 if(fallback!=MISSING)return fallback;
 const int* var=node(c,variable);int default_value=edge(c,variable,1);
 if(c->error||c->h.error)return MISSING;
 if(var[3]==2)return context_error_arg(c,11,variable);
 if(var[3]!=0&&var[3]!=1){c->error=BAD_PROGRAM;return MISSING;}
 if(var[3]==0&&default_value==NIL){c->error=UNSUPPORTED_TYPE;return MISSING;}
 return default_value;
}
__device__ inline int context_set(Context* c,int variable,int value){
 if(!context_variable(c,variable))return MISSING;
 if(value==MISSING||!node(c,value)){c->error=BAD_PROGRAM;return MISSING;}
 int table=context_table(c);if(table==MISSING)return MISSING;
 int binding=context_binding(c,variable);if(c->error||c->h.error)return MISSING;
 int old=binding==MISSING?MISSING:edge(c,binding,1);
 int fields[3]={variable,old,table};
 int token=context_sequence(c,T_CONTEXT_TOKEN,fields,3,0);if(token==MISSING)return MISSING;
 if(binding==MISSING){
  int pair[2]={variable,value};binding=make_tuple(&c->h,pair,2);
  if(c->h.error||list_append(&c->h,table,binding)!=0)return MISSING;
 }else{
  int* n=mutable_node(c,binding);if(!n)return MISSING;
  c->h.edges[((long long)n[1]+1)*2]=value;
 }
 return token;
}
__device__ inline int context_reset(Context* c,int variable,int token){
 if(!context_variable(c,variable))return MISSING;
 const int* n=node(c,token);if(!n)return MISSING;
 if(n[0]!=T_CONTEXT_TOKEN){return context_error(c,5,"Expected a ContextVar Token");}
 if(token<0||n[2]!=3){c->error=BAD_PROGRAM;return MISSING;}
 if(n[3])return context_error(c,6,"Token has already been used once");
 if(edge(c,token,0)!=variable)return context_error(c,2,"Token was created by a different ContextVar");
 if(edge(c,token,2)!=c->context_values)return context_error(c,2,"Token was created in a different Context");
 int binding=context_binding(c,variable);if(c->error||c->h.error)return MISSING;
 if(binding==MISSING){c->error=BAD_PROGRAM;return MISSING;}
 int old=edge(c,token,1);int* entry=mutable_node(c,binding);if(!entry)return MISSING;
 c->h.edges[((long long)entry[1]+1)*2]=old;mutable_node(c,token)[3]=1;return NIL;
}
__device__ inline int context_enter(Context* c,int manager){
 int* n=mutable_node(c,manager);if(!n)return MISSING;
 if(n[0]!=T_PATCH_MANAGER||n[2]!=2){c->error=BAD_PROGRAM;return MISSING;}
 // Reentering/reusing the contextlib generator manager is outside this primitive.
 if(n[3]!=0){c->error=UNSUPPORTED_TYPE;return MISSING;}
 int token=context_set(c,edge(c,manager,0),FALSE_REF);
 if(c->error||c->h.error||c->raised!=MISSING)return MISSING;
 c->h.edges[((long long)n[1]+1)*2]=token;n[3]=1;return NIL;
}
__device__ inline int context_exit(Context* c,int manager){
 int* n=mutable_node(c,manager);if(!n)return MISSING;
 if(n[0]!=T_PATCH_MANAGER||n[2]!=2){c->error=BAD_PROGRAM;return MISSING;}
 if(n[3]!=1){c->error=UNSUPPORTED_TYPE;return MISSING;}
 int result=context_reset(c,edge(c,manager,0),edge(c,manager,1));
 if(result==MISSING||c->error||c->h.error)return MISSING;n[3]=2;return FALSE_REF;
}
__device__ inline int context_attribute(Context* c,int owner,int name){
 int type=kind(c,owner),code=0;
 if(type==T_CONTEXT_VAR){
  if(vm_text_equals(c,name,"name"))return edge(c,owner,0);
  if(vm_text_equals(c,name,"get"))code=CONTEXT_GET;
  else if(vm_text_equals(c,name,"set"))code=CONTEXT_SET;
  else if(vm_text_equals(c,name,"reset"))code=CONTEXT_RESET;
  if(code)return meta(c,T_CONTEXT_BOUND,owner,0,code);
 }else if(type==T_PATCH_MANAGER){
  if(vm_text_equals(c,name,"__enter__"))code=CONTEXT_ENTER;
  else if(vm_text_equals(c,name,"__exit__"))code=CONTEXT_EXIT;
  if(code)return meta(c,T_CONTEXT_MANAGER_BOUND,owner,0,code);
 }else if(type==T_CONTEXT_TOKEN){
  if(vm_text_equals(c,name,"var"))return edge(c,owner,0);
  if(vm_text_equals(c,name,"old_value")){
   int old=edge(c,owner,1);if(old==MISSING)c->error=UNSUPPORTED_TYPE;return old;
  }
 }return MISSING;
}
__device__ inline int context_call(Context* c,int callable,const int* args,int nargs,
 int kw_names,const int* kw_values,int nkw,int* out){
 const int* n=node(c,callable);if(!n)return 0;int type=n[0],id=n[3];
 if(type!=T_CONTEXT_BOUND&&type!=T_CONTEXT_MANAGER_BOUND&&
  !(type==T_NATIVE&&(id==NATIVE_CONTEXT_VAR||id==NATIVE_WITHOUT_PATCHES)))return 0;
 *out=MISSING;
 if(nargs<0||nkw<0||(nargs&&!args)||(nkw&&!kw_values)){
  c->error=BAD_CALL;return 1;
 }
 if(type==T_NATIVE&&id==NATIVE_CONTEXT_VAR){
  if(nargs!=1||nkw>1){*out=context_error(c,5,"ContextVar requires a name and optional default keyword");return 1;}
  if(kind(c,args[0])!=TAG_STR){*out=context_error(c,5,"ContextVar name must be a string");return 1;}
  int default_value=NIL,has_default=0;
  if(nkw){
   if(kind(c,kw_names)!=TAG_TUPLE||length(c,kw_names)!=1||
    !vm_text_equals(c,edge(c,kw_names,0),"default")){
    *out=context_error(c,5,"Invalid ContextVar keyword");return 1;
   }
   default_value=kw_values[0];has_default=1;
  }
  int fields[2]={args[0],default_value};
  *out=context_sequence(c,T_CONTEXT_VAR,fields,2,has_default?1:2);return 1;
 }
 if(nkw){*out=context_error(c,5,"Context method does not accept keyword arguments");return 1;}
 if(type==T_NATIVE){
  if(nargs){*out=context_error(c,5,"_without_patches takes no arguments");return 1;}
  if(CONTEXT_COLLECT_PATCHES==MISSING){c->error=UNSUPPORTED_TYPE;return 1;}
  if(!context_variable(c,CONTEXT_COLLECT_PATCHES))return 1;
  int fields[2]={CONTEXT_COLLECT_PATCHES,MISSING};
  *out=context_sequence(c,T_PATCH_MANAGER,fields,2,0);return 1;
 }
 int owner=n[1];
 if(type==T_CONTEXT_BOUND){
  if(id==CONTEXT_GET){
   if(nargs>1)*out=context_error(c,5,"ContextVar.get accepts at most one argument");
   else *out=context_get(c,owner,nargs?args[0]:MISSING);
  }else if(id==CONTEXT_SET){
   if(nargs!=1)*out=context_error(c,5,"ContextVar.set requires one argument");else *out=context_set(c,owner,args[0]);
  }else if(id==CONTEXT_RESET){
   if(nargs!=1)*out=context_error(c,5,"ContextVar.reset requires one argument");else *out=context_reset(c,owner,args[0]);
  }else c->error=BAD_PROGRAM;
 }else{
  if(id==CONTEXT_ENTER){
   if(nargs)*out=context_error(c,5,"__enter__ takes no arguments");else *out=context_enter(c,owner);
  }else if(id==CONTEXT_EXIT){
   if(nargs!=3)*out=context_error(c,5,"__exit__ requires three arguments");else *out=context_exit(c,owner);
  }else c->error=BAD_PROGRAM;
 }
 return 1;
}
'''
