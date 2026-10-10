"""Device standard-library operations for frozen formal-rule functions.

Implements the standard-library names used by captured Duel V2 formal
functions. This is not a full Python runtime and does not claim full duel-rule
support. Callback keys, getters and generator consumers execute in the VM
(mode 7). Set and frozenset values are stored in first-seen insertion order;
hash-randomized order-sensitive formal paths are unsupported and fail closed.
uuid.uuid4 is intentionally omitted from SUPPORTED_STDLIB_NAMES: missing
entity identities must not be forged.

Tags 70-79 are reserved here (NumPy uses 60-69):
70 enumerate, 71 combinations, 72 permutations, 73 Counter,
74 super proxy, 75 stdlib accumulator, 78 stdlib bound method.
"""
from __future__ import annotations

SUPPORTED_STDLIB_NAMES: tuple[str, ...] = (
    'enumerate',
    'isinstance',
    'getattr',
    'hasattr',
    'super',
    'sorted',
    'set',
    'frozenset',
    'collections.Counter',
    'Counter',
    'itertools.combinations',
    'itertools.permutations',
    'math.isfinite',
    'isfinite',
    'enum.Enum',
    'enum.Enum.__repr__',
    'enum.Enum._generate_next_value_',
    'repr',
    'app.modules.card_game.engine.duel_v2.entities.clone_state',
    'app.modules.card_game.content.duel_v2.registry._ensure_kits',
)


def _node(image, ref):
    if ref >= 0:
        return None
    index = -1 - int(ref)
    if index < 0 or index >= len(image.nodes):
        return None
    return image.nodes[index]


def _payload_str(image, ref):
    node = _node(image, ref)
    if node is None or int(node[0]) != 4:
        return None
    offset, count = int(node[1]), int(node[2])
    return bytes(image.payload[offset:offset + count]).decode('utf-8')


def _kits_precondition(image):
    """Bake registry module index and KITS length. Runtime no-op only if complete."""
    names = tuple(image.module_names)
    target = 'app.modules.card_game.content.duel_v2.registry'
    if target not in names:
        return -1, 0
    module_index = names.index(target)
    module_ref = int(image.modules[module_index, 0])
    node = _node(image, module_ref)
    if node is None or int(node[0]) != 7:
        return module_index, 0
    offset, count = int(node[1]), int(node[2])
    kits_ref = None
    for i in range(count):
        key = int(image.edges[offset + i, 0])
        value = int(image.edges[offset + i, 1])
        if _payload_str(image, key) == 'KITS':
            kits_ref = value
            break
    if kits_ref is None:
        return module_index, 0
    kits_node = _node(image, kits_ref)
    if kits_node is None or int(kits_node[0]) != 7:
        return module_index, 0
    return module_index, int(kits_node[2])


def _nid(names, name, sentinel):
    return names.index(name) if name in names else sentinel


def emit_stdlib_source(image) -> str:
    names = tuple(image.intrinsic_names)
    kits_module, kits_count = _kits_precondition(image)
    mapping = {
        'ID_ENUMERATE': _nid(names, 'enumerate', -7001),
        'ID_ISINSTANCE': _nid(names, 'isinstance', -7002),
        'ID_GETATTR': _nid(names, 'getattr', -7003),
        'ID_HASATTR': _nid(names, 'hasattr', -7004),
        'ID_SUPER': _nid(names, 'super', -7005),
        'ID_SORTED': _nid(names, 'sorted', -7006),
        'ID_SET': _nid(names, 'set', -7007),
        'ID_FROZENSET': _nid(names, 'frozenset', -7008),
        'ID_COUNTER': _nid(names, 'collections.Counter', -7009),
        'ID_COUNTER_SHORT': _nid(names, 'Counter', -7010),
        'ID_COMBINATIONS': _nid(names, 'itertools.combinations', -7011),
        'ID_PERMUTATIONS': _nid(names, 'itertools.permutations', -7012),
        'ID_ISFINITE': _nid(names, 'math.isfinite', -7013),
        'ID_ISFINITE_SHORT': _nid(names, 'isfinite', -7014),
        'ID_ENUM': _nid(names, 'enum.Enum', -7015),
        'ID_ENUM_REPR': _nid(names, 'enum.Enum.__repr__', -7016),
        'ID_CLONE': _nid(names, 'app.modules.card_game.engine.duel_v2.entities.clone_state', -7017),
        'ID_ENSURE_KITS': _nid(names, 'app.modules.card_game.content.duel_v2.registry._ensure_kits', -7018),
        'ID_LIST': _nid(names, 'list', -7019),
        'ID_TUPLE': _nid(names, 'tuple', -7020),
        'ID_DICT': _nid(names, 'dict', -7021),
        'ID_MIN': _nid(names, 'min', -7022),
        'ID_MAX': _nid(names, 'max', -7023),
        'ID_UUID': _nid(names, 'uuid.uuid4', -7024),
        'ID_INT': _nid(names, 'int', -7025),
        'ID_FLOAT': _nid(names, 'float', -7026),
        'ID_BOOL': _nid(names, 'bool', -7027),
        'ID_STR': _nid(names, 'str', -7028),
        'ID_OBJECT': _nid(names, 'object', -7029),
        'ID_ENUM_NEXT': _nid(names, 'enum.Enum._generate_next_value_', -7030),
        'ID_REPR': _nid(names, 'repr', -7031),
        'KITS_MODULE_INDEX': kits_module,
        'KITS_EXPECTED_COUNT': kits_count,
    }
    source = _SOURCE
    # Replace longer tokens first: ID_COUNTER is a prefix of ID_COUNTER_SHORT.
    for key in sorted(mapping, key=len, reverse=True):
        source = source.replace(key, str(mapping[key]))
    return source


_SOURCE = r'''
// --- VM standard library (vm_stdlib.py). Tags 70-79. Mode 7 continuations. ---
// Set/frozenset iteration is first-seen insertion order. CPython hash order
// is not reproduced; order-sensitive formal paths remain unsupported.
#ifndef T_STDLIB_ENUMERATE
const int T_STDLIB_ENUMERATE=70,T_STDLIB_COMB=71,T_STDLIB_PERM=72,T_STDLIB_COUNTER=73;
const int T_STDLIB_SUPER=74,T_STDLIB_ACC=75,T_STDLIB_BOUND=78;
#endif
const int NATIVE_STDLIB_ENUMERATE=ID_ENUMERATE,NATIVE_STDLIB_ISINSTANCE=ID_ISINSTANCE;
const int NATIVE_STDLIB_GETATTR=ID_GETATTR,NATIVE_STDLIB_HASATTR=ID_HASATTR,NATIVE_STDLIB_SUPER=ID_SUPER;
const int NATIVE_STDLIB_SORTED=ID_SORTED,NATIVE_STDLIB_SET=ID_SET,NATIVE_STDLIB_FROZENSET=ID_FROZENSET;
const int NATIVE_STDLIB_COUNTER=ID_COUNTER,NATIVE_STDLIB_COUNTER_SHORT=ID_COUNTER_SHORT;
const int NATIVE_STDLIB_COMBINATIONS=ID_COMBINATIONS,NATIVE_STDLIB_PERMUTATIONS=ID_PERMUTATIONS;
const int NATIVE_STDLIB_ISFINITE=ID_ISFINITE,NATIVE_STDLIB_ISFINITE_SHORT=ID_ISFINITE_SHORT;
const int NATIVE_STDLIB_ENUM=ID_ENUM,NATIVE_STDLIB_ENUM_REPR=ID_ENUM_REPR;
const int NATIVE_STDLIB_CLONE=ID_CLONE,NATIVE_STDLIB_ENSURE_KITS=ID_ENSURE_KITS;
const int NATIVE_STDLIB_LIST=ID_LIST,NATIVE_STDLIB_TUPLE=ID_TUPLE,NATIVE_STDLIB_DICT=ID_DICT;
const int NATIVE_STDLIB_MIN=ID_MIN,NATIVE_STDLIB_MAX=ID_MAX,NATIVE_STDLIB_UUID=ID_UUID;
const int NATIVE_STDLIB_INT=ID_INT,NATIVE_STDLIB_FLOAT=ID_FLOAT,NATIVE_STDLIB_BOOL=ID_BOOL;
const int NATIVE_STDLIB_STR=ID_STR,NATIVE_STDLIB_OBJECT=ID_OBJECT;
const int NATIVE_STDLIB_ENUM_NEXT=ID_ENUM_NEXT,NATIVE_STDLIB_REPR=ID_REPR;
const int STDLIB_KITS_MODULE=KITS_MODULE_INDEX,STDLIB_KITS_COUNT=KITS_EXPECTED_COUNT,STDLIB_POOL_CAP=64;
const int STDLIB_METHOD_DICT_INIT=1,STDLIB_METHOD_OBJECT_INIT=2,STDLIB_METHOD_SET_ADD=3;
const int STDLIB_METHOD_SET_UPDATE=4,STDLIB_METHOD_COUNTER_GET=5,STDLIB_METHOD_COUNTER_UPDATE=6,STDLIB_METHOD_COUNTER_COPY=7,STDLIB_METHOD_ENUM_REPR=8;
const int STDLIB_ACC_LIST=1,STDLIB_ACC_TUPLE=2,STDLIB_ACC_SET=3,STDLIB_ACC_FROZEN=4,STDLIB_ACC_SORTED=5;
const int STDLIB_ACC_SORTED_KEYS=6,STDLIB_ACC_COUNTER=7,STDLIB_ACC_MINKEY=8,STDLIB_ACC_MAXKEY=9;
const int STDLIB_ACC_COMB_POOL=10,STDLIB_ACC_PERM_POOL=11,STDLIB_ACC_EXTEND=12;

__device__ inline int make_set(Context* c);
__device__ inline int set_add(Context* c,int set_ref,int value);
__device__ inline int set_contains(Context* c,int needle,int set_ref);
__device__ inline int native_deepcopy(Context* c,int ref);
__device__ inline int device_dict_update(Context* c,int dict_ref,int other_ref);
__device__ inline int lookup_super(Context* c,int start_cls,int obj,int name,int* getter);
__device__ inline int class_mro_contains(Context* c,int cid,int type_ref);
__device__ inline int class_entity_tag(int cid);
__device__ inline int class_dict_subclass(int cid);
__device__ inline int stdlib_call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names,const int* kw_values,int nkw);
__device__ inline int stdlib_iter_next(Context* c,int ref);
__device__ inline int stdlib_for_iter(Context* c,int ref,int target);
__device__ inline int stdlib_attribute(Context* c,int owner,int name);
__device__ inline void stdlib_resume(Context* c,int generator,int caller,int accumulator,int item,int exhausted);
__device__ inline int stdlib_compare(Context* c,int a,int b,int mode,int* out);
__device__ inline int stdlib_binary(Context* c,int a,int b,int op,int* out);
__device__ inline int stdlib_subscript(Context* c,int owner,int key,int* out);
__device__ inline int stdlib_store(Context* c,int owner,int key,int val);
__device__ inline int stdlib_handle_opcode(Context* c,int* f,int op,int arg,int value,int target);
__device__ inline int concat_strings(Context* c,const int* refs,int count);

__device__ inline int stdlib_is_counter_id(int id){return (NATIVE_STDLIB_COUNTER>=0&&id==NATIVE_STDLIB_COUNTER)||(NATIVE_STDLIB_COUNTER_SHORT>=0&&id==NATIVE_STDLIB_COUNTER_SHORT);}
__device__ inline int stdlib_is_isfinite_id(int id){return (NATIVE_STDLIB_ISFINITE>=0&&id==NATIVE_STDLIB_ISFINITE)||(NATIVE_STDLIB_ISFINITE_SHORT>=0&&id==NATIVE_STDLIB_ISFINITE_SHORT);}
__device__ inline int stdlib_new_counter(Context* c){int d=make_dict(&c->h);int* p=mutable_node(c,d);if(p)p[0]=T_STDLIB_COUNTER;return d;}
__device__ inline int stdlib_kw(Context* c,int kw_names,const int* kw_values,int nkw,const char* name,int* present){
 *present=0;if(nkw<=0||kw_names==MISSING)return MISSING;
 for(int i=0;i<nkw;i++){if(vm_text_equals(c,edge(c,kw_names,i),name)){*present=1;return kw_values[i];}}return MISSING;
}
__device__ inline int stdlib_unknown_kw(Context* c,int kw_names,int nkw,const char* a,const char* b,const char* d){
 for(int i=0;i<nkw;i++){int k=edge(c,kw_names,i);int ok=(a&&vm_text_equals(c,k,a))||(b&&vm_text_equals(c,k,b))||(d&&vm_text_equals(c,k,d));if(!ok){c->error=BAD_CALL;return 1;}}return 0;
}
__device__ inline void stdlib_acc_set(Context* c,int acc,int field,int val){
 const int* n=node(c,acc);if(!n||n[0]!=T_STDLIB_ACC||field<0||field>=n[2]||acc<0){c->error=BAD_PROGRAM;return;}
 long long slot=(long long)n[1]+field;if(slot<0||slot>=c->h.counts[1]){c->error=BAD_PROGRAM;return;}c->h.edges[slot*2]=val;
}
__device__ inline int stdlib_make_acc(Context* c,int op,int caller,int source,int result,int keyfn,int extra){
 int fields[10]={op,caller,source,result,keyfn,MISSING,0,extra,MISSING,0};
 int acc=make_tuple(&c->h,fields,10);int* n=mutable_node(c,acc);if(!n)return MISSING;n[0]=T_STDLIB_ACC;return acc;
}
__device__ inline int stdlib_push(Context* c,int caller,int value){if(caller<0)c->output=value;else push(c,frame(c,caller),value);return 1;}
__device__ inline int stdlib_bind(Context* c,int owner,int code){return meta(c,T_STDLIB_BOUND,owner,0,code);}
__device__ inline int stdlib_edge_set(Context* c,int ref,int i,int val){
 const int* n=node(c,ref);if(!n||ref<0||i<0||i>=n[2]){c->error=BAD_PROGRAM;return 0;}
 long long slot=(long long)n[1]+i;if(slot<0||slot>=c->h.counts[1]){c->error=BAD_PROGRAM;return 0;}c->h.edges[slot*2]=val;return 1;
}
__device__ inline int stdlib_counter_add(Context* c,int counter,int key,int delta_ref){
 long long delta=1;if(delta_ref!=MISSING){if(read_int64(&c->h,delta_ref,&delta)<0)return 0;}
 int found=dict_find(&c->h,counter,key);if(c->h.error)return 0;long long current=0;
 if(found>=0){if(read_int64(&c->h,edge(c,counter,found,1),&current)<0)return 0;}
 long long next=(long long)((unsigned long long)current+(unsigned long long)delta);if((delta>0&&next<current)||(delta<0&&next>current)){c->h.error=ERROR_INT_OVERFLOW;return 0;}
 return dict_set(&c->h,counter,key,make_int64(&c->h,next))==0;
}
__device__ inline int stdlib_async_src(Context* c,int src){
 int tag=kind(c,src);if(tag==T_GEN||tag==T_ITER)return 1;
 if(tag==T_STDLIB_ENUMERATE){int ot=kind(c,edge(c,src,0));return ot==T_GEN||ot==T_ITER;}return 0;
}
__device__ inline int stdlib_iget(Context* c,int ref,int i){long long v=0;if(read_int64(&c->h,edge(c,ref,i),&v)<0)return -1;return (int)v;}
__device__ inline void stdlib_iset(Context* c,int ref,int i,int v){stdlib_edge_set(c,ref,i,make_int64(&c->h,(long long)v));}
__device__ inline int stdlib_combo_init(Context* c,int pool,int r,int perm){
 int n=length(c,pool);if(n>STDLIB_POOL_CAP){c->error=UNSUPPORTED_TYPE;return MISSING;}if(r<0){c->error=BAD_CALL;return MISSING;}
 if(r>n){int done=make_tuple(&c->h,0,0);int* p=mutable_node(c,done);if(p){p[0]=perm?T_STDLIB_PERM:T_STDLIB_COMB;p[3]=2;}return done;}
 int extra=perm?n+r:r;int refs[2+STDLIB_POOL_CAP+STDLIB_POOL_CAP];refs[0]=pool;refs[1]=make_int64(&c->h,(long long)r);
 if(perm){for(int i=0;i<n;i++)refs[2+i]=make_int64(&c->h,(long long)i);for(int i=0;i<r;i++)refs[2+n+i]=make_int64(&c->h,(long long)(n-i));}
 else{for(int i=0;i<r;i++)refs[2+i]=make_int64(&c->h,(long long)i);}
 int ref=make_tuple(&c->h,refs,2+extra);int* p=mutable_node(c,ref);if(!p)return MISSING;p[0]=perm?T_STDLIB_PERM:T_STDLIB_COMB;p[3]=0;return ref;
}
__device__ inline int stdlib_take_pool(Context* c,int ref,int r){
 int pool=edge(c,ref,0);int items[STDLIB_POOL_CAP];for(int i=0;i<r;i++){int idx=stdlib_iget(c,ref,2+i);if(idx<0)return MISSING;items[i]=edge(c,pool,idx);}return make_tuple(&c->h,items,r);
}
__device__ inline int stdlib_combo_advance(Context* c,int ref){
 int* np=mutable_node(c,ref);if(!np)return MISSING;if(np[3]==2)return MISSING;
 long long r=0;read_int64(&c->h,edge(c,ref,1),&r);int n=length(c,edge(c,ref,0)),rr=(int)r;
 if(rr==0){if(np[3]==0){np[3]=1;return make_tuple(&c->h,0,0);}np[3]=2;return MISSING;}
 if(np[3]==0){np[3]=1;return stdlib_take_pool(c,ref,rr);}
 for(int i=rr-1;i>=0;i--){int idx=stdlib_iget(c,ref,2+i);if(idx!=i+n-rr){stdlib_iset(c,ref,2+i,idx+1);for(int j=i+1;j<rr;j++)stdlib_iset(c,ref,2+j,stdlib_iget(c,ref,2+j-1)+1);return stdlib_take_pool(c,ref,rr);}}
 np[3]=2;return MISSING;
}
__device__ inline int stdlib_perm_advance(Context* c,int ref){
 int* np=mutable_node(c,ref);if(!np)return MISSING;if(np[3]==2)return MISSING;
 long long r=0;read_int64(&c->h,edge(c,ref,1),&r);int n=length(c,edge(c,ref,0)),rr=(int)r;
 if(rr==0){if(np[3]==0){np[3]=1;return make_tuple(&c->h,0,0);}np[3]=2;return MISSING;}
 if(np[3]==0){np[3]=1;return stdlib_take_pool(c,ref,rr);}if(!n){np[3]=2;return MISSING;}
 for(int i=rr-1;i>=0;i--){int cycles=stdlib_iget(c,ref,2+n+i)-1;stdlib_iset(c,ref,2+n+i,cycles);
  if(cycles==0){int first=stdlib_iget(c,ref,2+i);for(int j=i;j<n-1;j++)stdlib_iset(c,ref,2+j,stdlib_iget(c,ref,2+j+1));stdlib_iset(c,ref,2+n-1,first);stdlib_iset(c,ref,2+n+i,n-i);}
  else{int j=n-cycles;int a=stdlib_iget(c,ref,2+i),b=stdlib_iget(c,ref,2+j);stdlib_iset(c,ref,2+i,b);stdlib_iset(c,ref,2+j,a);return stdlib_take_pool(c,ref,rr);}}
 np[3]=2;return MISSING;
}
__device__ inline int stdlib_iter_next(Context* c,int ref){
 int tag=kind(c,ref);
 if(tag==T_STDLIB_ENUMERATE){int owner=edge(c,ref,0),ot=kind(c,owner);if(ot==T_GEN||ot==T_ITER)return MISSING;
  long long start=0;if(edge(c,ref,2)!=NIL&&kind(c,edge(c,ref,2))!=TAG_NONE)read_int64(&c->h,edge(c,ref,2),&start);
  int idx=node(c,ref)[3],pos=idx-(int)start,n=length(c,owner);if(pos<0||pos>=n)return MISSING;
  int pair[2]={make_int64(&c->h,(long long)idx),edge(c,owner,pos)};mutable_node(c,ref)[3]=idx+1;return make_tuple(&c->h,pair,2);}
 if(tag==T_STDLIB_COMB)return stdlib_combo_advance(c,ref);
 if(tag==T_STDLIB_PERM)return stdlib_perm_advance(c,ref);
 return MISSING;
}
__device__ inline int stdlib_for_iter(Context* c,int ref,int target){
 int tag=kind(c,ref);if(tag!=T_STDLIB_ENUMERATE&&tag!=T_STDLIB_COMB&&tag!=T_STDLIB_PERM)return 0;
 if(tag==T_STDLIB_ENUMERATE){int owner=edge(c,ref,0),ot=kind(c,owner);
  if(ot==T_GEN){if(edge(c,ref,1)==NIL)stdlib_edge_set(c,ref,1,make_int64(&c->h,(long long)target));resume_generator(c,owner,c->current,7,ref,MISSING);return 1;}
  if(ot==T_ITER){int item=iterator_next(c,owner);if(item==MISSING){pop(c,frame(c,c->current));frame(c,c->current)[1]=target;return 1;}
   int idx=node(c,ref)[3];int pair[2]={make_int64(&c->h,(long long)idx),item};mutable_node(c,ref)[3]=idx+1;push(c,frame(c,c->current),make_tuple(&c->h,pair,2));return 1;}}
 int next=stdlib_iter_next(c,ref);if(next==MISSING){pop(c,frame(c,c->current));frame(c,c->current)[1]=target;}else push(c,frame(c,c->current),next);return 1;
}
__device__ inline int compare(Context* c,int a,int b,int mode);
__device__ inline int native_builtin_call(Context* c,int callable,const int* args,int nargs,int* out);
__device__ inline int object_attribute(Context* c,int owner,int name,int* getter);
__device__ inline int builtin_attribute(Context* c,int owner,int name);
__device__ inline int stdlib_is_none(Context* c,int ref){return ref==MISSING||ref==NIL||kind(c,ref)==TAG_NONE;}
// Python clone_state hydrates identities too. Reject inputs for which hydration
// would change identity, aliasing, class tags, side ownership or the ID counter.
__device__ inline int stdlib_hydrated_entity(Context* c,int ref,int tag,int side,int* refs,int* ids,int* count,long long* high){
 if(kind(c,ref)!=tag)return 0;
 int idname=make_utf8(&c->h,"entity_id",9),sidename=make_utf8(&c->h,"side",4);
 int eid=dict_get(&c->h,ref,idname,MISSING),actual_side=dict_get(&c->h,ref,sidename,MISSING);
 if(eid==MISSING||kind(c,eid)!=TAG_STR||node(c,eid)[2]==0||actual_side==MISSING||!vm_equal(c,side,actual_side))return 0;
 for(int i=0;i<*count;i++)if(key_equal(&c->h,eid,ids[i]))return ref==refs[i];
 if(*count>=512){c->error=STACK_OVERFLOW;return 0;}
 refs[*count]=ref;ids[(*count)++]=eid;
 NodeView nv;if(!get_node(&c->h,eid,&nv))return 0;const unsigned char* p=get_payload_ptr(&c->h,&nv);
 if(nv.count>1&&p[0]=='e'){
  long long number=0;int digits=1;for(int i=1;i<nv.count;i++)if(p[i]<'0'||p[i]>'9'){digits=0;break;}
  if(digits){for(int i=1;i<nv.count;i++){int digit=p[i]-'0';if(number>(9223372036854775807LL-digit)/10)return 0;number=number*10+digit;}if(number>*high)*high=number;}
 }
 if(tag==TAG_CARD_ENTITY){
  int inst_name=make_utf8(&c->h,"instance_id",11),inst=dict_get(&c->h,ref,inst_name,MISSING);
  if(inst==MISSING||kind(c,inst)!=TAG_STR||node(c,inst)[2]==0)return 0;
  for(int i=0;i<*count-1;i++)if(kind(c,refs[i])==TAG_CARD_ENTITY){int old=dict_get(&c->h,refs[i],inst_name,MISSING);if(old!=MISSING&&key_equal(&c->h,old,inst))return 0;}
 }
 return !c->error&&!c->h.error;
}
__device__ inline int stdlib_clone_hydrated(Context* c,int state){
 if(kind(c,state)!=TAG_DICT)return MISSING;
 int sides=dict_text_get(c,state,"sides"),next=dict_text_get(c,state,"next_entity");
 if(sides==MISSING||kind(c,sides)!=TAG_DICT||next==MISSING||kind(c,next)!=TAG_INT)return MISSING;
 long long high=0,sequence=0;if(read_int64(&c->h,next,&sequence)<0)return MISSING;
 int refs[512],ids[512],count=0;
 const char* zones[4]={"hand","deck","discard","removed"};
 for(int s=0;s<length(c,sides);s++){
  int side=edge(c,sides,s),team=edge(c,sides,s,1);
  if(!stdlib_hydrated_entity(c,team,TAG_PLAYER_ENTITY,side,refs,ids,&count,&high))return MISSING;
  int chars=dict_get(&c->h,team,make_utf8(&c->h,"characters",10),MISSING);if(chars==MISSING||kind(c,chars)!=TAG_DICT)return MISSING;
  for(int j=0;j<length(c,chars);j++)if(!stdlib_hydrated_entity(c,edge(c,chars,j,1),TAG_CHARACTER_ENTITY,side,refs,ids,&count,&high))return MISSING;
  for(int z=0;z<4;z++){int cards=dict_get(&c->h,team,make_utf8(&c->h,zones[z],z>=2?7:4),MISSING);if(cards==MISSING)continue;if(kind(c,cards)!=TAG_LIST)return MISSING;
   for(int j=0;j<length(c,cards);j++){int card=edge(c,cards,j),inst=dict_get(&c->h,card,make_utf8(&c->h,"instance_id",11),MISSING);if(inst!=MISSING&&truth_value(c,inst)&&!stdlib_hydrated_entity(c,card,TAG_CARD_ENTITY,side,refs,ids,&count,&high))return MISSING;}}
 }
 const char* transients[2]={"operation","pending_choice"};
 for(int t=0;t<2;t++){int data=dict_text_get(c,state,transients[t]);if(stdlib_is_none(c,data))continue;if(kind(c,data)!=TAG_DICT)return MISSING;int side=dict_text_get(c,data,"side");
  int cards=t?dict_text_get(c,data,"cards"):MISSING,card=t?MISSING:dict_text_get(c,data,"card");
  int n=cards==MISSING?0:length(c,cards);if(!t&&card!=MISSING&&!stdlib_is_none(c,card))n=1;
  for(int j=0;j<n;j++){int item=t?edge(c,cards,j):card,inst=dict_get(&c->h,item,make_utf8(&c->h,"instance_id",11),MISSING);if(inst!=MISSING&&truth_value(c,inst)&&!stdlib_hydrated_entity(c,item,TAG_CARD_ENTITY,side,refs,ids,&count,&high))return MISSING;}
 }
 if(c->error||c->h.error||sequence<high)return MISSING;
 return native_deepcopy(c,state);
}
__device__ inline int stdlib_set_eq(Context* c,int a,int b){int na=length(c,a),nb=length(c,b);if(na!=nb)return 0;for(int i=0;i<na;i++)if(!set_contains(c,edge(c,a,i),b))return 0;return 1;}
__device__ inline int stdlib_enum_eq(Context* c,int a,int b){
 int ta=kind(c,a),tb=kind(c,b);
 int other=ta==27?tb:ta;if(other>4&&other!=27)return 0;
 return key_equal(&c->h,a,b);
}
__device__ inline int stdlib_mapping_eq(Context* c,int a,int b){
 if(kind(c,a)==T_STDLIB_COUNTER&&kind(c,b)==T_STDLIB_COUNTER){
  // CPython 3.10 Counter equality treats absent counts as zero.
  int zero=make_int64(&c->h,0);
  for(int direction=0;direction<2;direction++){int left=direction?b:a,right=direction?a:b;
   for(int i=0;i<length(c,left);i++){int value=dict_get(&c->h,right,edge(c,left,i),zero);
    if(!vm_equal(c,edge(c,left,i,1),value))return 0;}}
  return !c->error&&!c->h.error;
 }
 int na=length(c,a);if(na!=length(c,b))return 0;
 for(int i=0;i<na;i++){int k=edge(c,a,i),vb=dict_get(&c->h,b,k,MISSING);if(vb==MISSING)return 0;if(!vm_equal(c,edge(c,a,i,1),vb))return 0;}return 1;
}
__device__ inline int stdlib_seq_cmp(Context* c,int a,int b){
 int na=length(c,a),nb=length(c,b),n=na<nb?na:nb;
 for(int i=0;i<n;i++){int x=edge(c,a,i),y=edge(c,b,i);if(compare(c,x,y,0))return -1;if(c->error||c->h.error)return 0;if(compare(c,y,x,0))return 1;if(c->error||c->h.error)return 0;}
 return na<nb?-1:na>nb?1:0;
}
__device__ inline void stdlib_reverse_list(Context* c,int ref){int n=length(c,ref);for(int i=0;i<n/2;i++){int a=edge(c,ref,i),b=edge(c,ref,n-1-i);stdlib_edge_set(c,ref,i,b);stdlib_edge_set(c,ref,n-1-i,a);}}
__device__ inline void stdlib_sort_lists(Context* c,int items,int keys,int reverse){
 int use_keys=!(keys==MISSING||keys==NIL||kind(c,keys)==TAG_NONE);
 if(reverse){stdlib_reverse_list(c,items);if(use_keys)stdlib_reverse_list(c,keys);}
 int n=length(c,items);
 for(int i=1;i<n;i++){int item=edge(c,items,i),key=use_keys?edge(c,keys,i):item,j=i;
  while(j>0){int prev_item=edge(c,items,j-1),prev_key=use_keys?edge(c,keys,j-1):prev_item;if(!compare(c,key,prev_key,0))break;if(c->error||c->h.error)return;stdlib_edge_set(c,items,j,prev_item);if(use_keys)stdlib_edge_set(c,keys,j,prev_key);j--;}
  stdlib_edge_set(c,items,j,item);if(use_keys)stdlib_edge_set(c,keys,j,key);}
 if(reverse){stdlib_reverse_list(c,items);if(use_keys)stdlib_reverse_list(c,keys);}
}
__device__ inline int stdlib_scalar_repr(Context* c,int value){
 if(kind(c,value)!=TAG_STR)return scalar_string(c,value);
 NodeView v;if(!get_node(&c->h,value,&v))return MISSING;
 const unsigned char* p=get_payload_ptr(&c->h,&v);int single=0,dbl=0;
 for(int i=0;i<v.count;i++){if(p[i]=='\'')single=1;if(p[i]=='"')dbl=1;}
 char quote=single&&!dbl?'"':'\'';int parts=0;char text[4096];text[parts++]=quote;
 for(int i=0;i<v.count;i++){
  unsigned char ch=p[i];if(ch<32||ch>=127){c->error=UNSUPPORTED_TYPE;return MISSING;}
  if(parts+3>=4096){c->error=UNSUPPORTED_TYPE;return MISSING;}
  if(ch=='\\'||ch==quote)text[parts++]='\\';text[parts++]=(char)ch;
 }
 text[parts++]=quote;return make_utf8(&c->h,text,parts);
}
__device__ inline int stdlib_enum_repr(Context* c,int en){
 if(kind(c,en)!=27){c->error=UNSUPPORTED_TYPE;return MISSING;}
 int open_b=make_utf8(&c->h,"<",1),colon=make_utf8(&c->h,": ",2),close_b=make_utf8(&c->h,">",1),text=edge(c,en,3),value=edge(c,en,1),valstr=stdlib_scalar_repr(c,value);
 int parts[5]={open_b,text,colon,valstr,close_b};return concat_strings(c,parts,5);
}
__device__ inline int stdlib_match_type(Context* c,int obj,int type_ref){
 int tt=kind(c,type_ref),ot=kind(c,obj);
 if(tt==TAG_TUPLE){int n=length(c,type_ref);for(int i=0;i<n;i++)if(stdlib_match_type(c,obj,edge(c,type_ref,i)))return 1;return 0;}
 if(tt==T_NATIVE){int id=node(c,type_ref)[3];
  if(NATIVE_STDLIB_INT>=0&&id==NATIVE_STDLIB_INT)return ot==TAG_INT||ot==TAG_BOOL;
  if(NATIVE_STDLIB_FLOAT>=0&&id==NATIVE_STDLIB_FLOAT)return ot==TAG_FLOAT;
  if(NATIVE_STDLIB_BOOL>=0&&id==NATIVE_STDLIB_BOOL)return ot==TAG_BOOL;
  if(NATIVE_STDLIB_STR>=0&&id==NATIVE_STDLIB_STR){if(ot==TAG_STR)return 1;if(ot==27){int cls=edge(c,obj,0);return kind(c,cls)==13&&class_str_subclass(node(c,cls)[3]);}return 0;}
  if(NATIVE_STDLIB_OBJECT>=0&&id==NATIVE_STDLIB_OBJECT)return 1;
  if(NATIVE_STDLIB_DICT>=0&&id==NATIVE_STDLIB_DICT){if(ot==TAG_DICT||(ot>=8&&ot<=10)||ot==T_STDLIB_COUNTER)return 1;if(ot==29){int cls=class_ref(c,obj);return cls!=MISSING&&kind(c,cls)==13&&class_dict_subclass(node(c,cls)[3]);}return 0;}
  if(NATIVE_STDLIB_LIST>=0&&id==NATIVE_STDLIB_LIST)return ot==TAG_LIST;
  if(NATIVE_STDLIB_TUPLE>=0&&id==NATIVE_STDLIB_TUPLE)return ot==TAG_TUPLE;
  if(NATIVE_STDLIB_SET>=0&&id==NATIVE_STDLIB_SET)return ot==21;
  if(NATIVE_STDLIB_FROZENSET>=0&&id==NATIVE_STDLIB_FROZENSET)return ot==22;
  if(stdlib_is_counter_id(id))return ot==T_STDLIB_COUNTER;
  if(NATIVE_STDLIB_ENUM>=0&&id==NATIVE_STDLIB_ENUM){if(ot==27)return 1;int cls=class_ref(c,obj);return cls!=MISSING&&kind(c,cls)==13&&class_enum_subclass(node(c,cls)[3]);}
  return 0;}
 if(tt==13){int cls=class_ref(c,obj);if(cls==MISSING||kind(c,cls)!=13)return 0;return class_mro_contains(c,node(c,cls)[3],type_ref);}return 0;
}
__device__ inline int stdlib_attribute_name(Context* c,int name){
 if(kind(c,name)==27){
  int cls=class_ref(c,name);
  if(cls!=MISSING&&kind(c,cls)==13&&class_str_subclass(node(c,cls)[3]))name=edge(c,name,1);
 }
 if(kind(c,name)!=TAG_STR){c->error=BAD_CALL;return MISSING;}
 return name;
}
__device__ inline int stdlib_lookup(Context* c,int owner,int name,int* getter){
 *getter=0;int attr=object_attribute(c,owner,name,getter);if(attr!=MISSING||c->error||c->h.error)return attr;
 attr=stdlib_attribute(c,owner,name);if(attr!=MISSING||c->error||c->h.error)return attr;
 attr=builtin_attribute(c,owner,name);if(attr!=MISSING||c->error||c->h.error)return attr;
 return MISSING;
}
__device__ inline void stdlib_attr_missing(Context* c){c->raised=exception_create(c,exception_type(c,10),make_tuple(&c->h,0,0));}
__device__ inline void stdlib_finish(Context* c,int acc);
__device__ inline int stdlib_invoke(Context* c,int fn,int arg,int caller,int acc);
__device__ inline void stdlib_sorted_keys(Context* c,int acc);
__device__ inline void stdlib_minmax_next(Context* c,int acc);
__device__ inline int stdlib_bound_apply(Context* c,int callable,const int* args,int nargs,int kw_names,const int* kw_values,int nkw,int* out);
__device__ inline void stdlib_finish(Context* c,int acc){
 if(c->error||c->h.error)return;
 int op=edge(c,acc,0),caller=edge(c,acc,1),result=edge(c,acc,3),keyfn=edge(c,acc,4),keys=edge(c,acc,5),extra=edge(c,acc,7);
 if(op==STDLIB_ACC_EXTEND)result=NIL;
 else if(op==STDLIB_ACC_SORTED){if(!stdlib_is_none(c,keyfn)){stdlib_acc_set(c,acc,0,STDLIB_ACC_SORTED_KEYS);stdlib_acc_set(c,acc,5,make_list(&c->h));stdlib_acc_set(c,acc,6,0);stdlib_sorted_keys(c,acc);return;}stdlib_sort_lists(c,result,MISSING,extra);}
 else if(op==STDLIB_ACC_SORTED_KEYS)stdlib_sort_lists(c,result,keys,extra);
 else if(op==STDLIB_ACC_TUPLE){int* n=mutable_node(c,result);if(n)n[0]=TAG_TUPLE;}
 else if(op==STDLIB_ACC_FROZEN){int* n=mutable_node(c,result);if(n)n[0]=22;}
 else if(op==STDLIB_ACC_COMB_POOL||op==STDLIB_ACC_PERM_POOL){int* n=mutable_node(c,result);if(n)n[0]=TAG_TUPLE;result=stdlib_combo_init(c,result,extra,op==STDLIB_ACC_PERM_POOL);}
 else if(op==STDLIB_ACC_MINKEY||op==STDLIB_ACC_MAXKEY){if(!edge(c,acc,6)){int def=edge(c,acc,8);if(stdlib_is_none(c,def)){c->error=BAD_CALL;return;}result=def;}}
 stdlib_push(c,caller,result);c->current=caller;
}
__device__ inline void stdlib_sorted_keys(Context* c,int acc){int items=edge(c,acc,3),idx=edge(c,acc,6),n=length(c,items);if(idx>=n){stdlib_finish(c,acc);return;}stdlib_invoke(c,edge(c,acc,4),edge(c,items,idx),edge(c,acc,1),acc);}
__device__ inline void stdlib_minmax_next(Context* c,int acc){
 int src=edge(c,acc,2),tag=kind(c,src),item=MISSING;
 if(tag==T_STDLIB_ENUMERATE||tag==T_STDLIB_COMB||tag==T_STDLIB_PERM)item=stdlib_iter_next(c,src);
 else if(tag==T_ITER)item=iterator_next(c,src);
 else if(tag==26){int it=iterator(c,src);stdlib_acc_set(c,acc,2,it);item=iterator_next(c,it);}
 else{int idx=node(c,acc)[3],n=length(c,src);if(idx>=n){stdlib_finish(c,acc);return;}item=edge(c,src,idx);mutable_node(c,acc)[3]=idx+1;}
 if(item==MISSING){stdlib_finish(c,acc);return;}stdlib_acc_set(c,acc,9,item);stdlib_invoke(c,edge(c,acc,4),item,edge(c,acc,1),acc);
}
__device__ inline int stdlib_append_raw(Context* c,int acc,int item){
 int op=edge(c,acc,0),result=edge(c,acc,3);
 if(op==STDLIB_ACC_EXTEND||op==STDLIB_ACC_LIST||op==STDLIB_ACC_TUPLE||op==STDLIB_ACC_SORTED||op==STDLIB_ACC_COMB_POOL||op==STDLIB_ACC_PERM_POOL)return list_append(&c->h,result,item)==0;
 if(op==STDLIB_ACC_SET||op==STDLIB_ACC_FROZEN)return set_add(c,result,item);
 if(op==STDLIB_ACC_COUNTER)return stdlib_counter_add(c,result,item,MISSING);
 c->error=BAD_PROGRAM;return 0;
}
__device__ inline int stdlib_consume(Context* c,int src,int op,int caller,int keyfn,int extra,int result_override=MISSING){
 int result=result_override!=MISSING?result_override:(op==STDLIB_ACC_SET||op==STDLIB_ACC_FROZEN)?make_set(c):(op==STDLIB_ACC_COUNTER?stdlib_new_counter(c):make_list(&c->h));
 int acc=stdlib_make_acc(c,op,caller,src,result,keyfn,extra);int tag=kind(c,src);
 if(tag==T_GEN){if(node(c,src)[3]<0)stdlib_finish(c,acc);else resume_generator(c,src,caller,7,acc,MISSING);return 1;}
 if(tag==T_STDLIB_ENUMERATE&&kind(c,edge(c,src,0))==T_GEN){resume_generator(c,edge(c,src,0),caller,7,acc,MISSING);return 1;}
 if(tag==T_ITER){for(;;){int item=iterator_next(c,src);if(item==MISSING)break;if(!stdlib_append_raw(c,acc,item))return 1;}stdlib_finish(c,acc);return 1;}
 if(tag==T_STDLIB_ENUMERATE||tag==T_STDLIB_COMB||tag==T_STDLIB_PERM){for(;;){int item=stdlib_iter_next(c,src);if(item==MISSING)break;if(!stdlib_append_raw(c,acc,item))return 1;}stdlib_finish(c,acc);return 1;}
 int n=length(c,src);for(int i=0;i<n;i++)if(!stdlib_append_raw(c,acc,edge(c,src,i)))return 1;stdlib_finish(c,acc);return 1;
}
__device__ inline int stdlib_bound_apply(Context* c,int callable,const int* args,int nargs,int kw_names,const int* kw_values,int nkw,int* out){
 const int* n=node(c,callable);if(!n||n[0]!=T_STDLIB_BOUND)return 0;int owner=n[1],code=n[3];
 if(code==STDLIB_METHOD_DICT_INIT){if(nargs>1){c->error=BAD_CALL;return 1;}if(nargs==1){int src=args[0],st=kind(c,src);
  if(st==TAG_DICT||(st>=8&&st<=10)||st==T_STDLIB_COUNTER){if(device_dict_update(c,owner,src))return 1;}
  else if(st==TAG_LIST||st==TAG_TUPLE){int cnt=length(c,src);for(int i=0;i<cnt;i++){int pair=edge(c,src,i);const int* pn=node(c,pair);if(!pn||(pn[0]!=TAG_LIST&&pn[0]!=TAG_TUPLE)||pn[2]!=2){c->error=BAD_CALL;return 1;}if(dict_set(&c->h,owner,edge(c,pair,0),edge(c,pair,1)))return 1;}}
  else if(!stdlib_is_none(c,src)&&!(st==TAG_TUPLE&&length(c,src)==0)){c->error=UNSUPPORTED_TYPE;return 1;}}
  for(int i=0;i<nkw;i++)if(dict_set(&c->h,owner,edge(c,kw_names,i),kw_values[i]))return 1;*out=NIL;return 1;}
 if(code==STDLIB_METHOD_OBJECT_INIT){if(nargs||nkw){c->error=BAD_CALL;return 1;}*out=NIL;return 1;}
 if(code==STDLIB_METHOD_SET_ADD){if(nargs!=1||nkw){c->error=BAD_CALL;return 1;}set_add(c,owner,args[0]);*out=NIL;return 1;}
 if(code==STDLIB_METHOD_SET_UPDATE){if(nargs!=1||nkw){c->error=BAD_CALL;return 1;}int src=args[0],st=kind(c,src);if(st==T_GEN||st==T_ITER){c->error=UNSUPPORTED_TYPE;return 1;}int cnt=length(c,src);for(int i=0;i<cnt;i++)set_add(c,owner,edge(c,src,i));*out=NIL;return 1;}
 if(code==STDLIB_METHOD_COUNTER_UPDATE){if(nargs>1){c->error=BAD_CALL;return 1;}if(nargs==1){int src=args[0],st=kind(c,src);
  if(st==T_STDLIB_COUNTER||st==TAG_DICT||(st>=8&&st<=10)){int cnt=length(c,src);for(int i=0;i<cnt;i++)if(!stdlib_counter_add(c,owner,edge(c,src,i),edge(c,src,i,1)))return 1;}
  else if(st==TAG_LIST||st==TAG_TUPLE||st==21||st==22){int cnt=length(c,src);for(int i=0;i<cnt;i++)if(!stdlib_counter_add(c,owner,edge(c,src,i),MISSING))return 1;}
  else{c->error=UNSUPPORTED_TYPE;return 1;}}
  for(int i=0;i<nkw;i++)if(!stdlib_counter_add(c,owner,edge(c,kw_names,i),kw_values[i]))return 1;*out=NIL;return 1;}
 if(code==STDLIB_METHOD_COUNTER_COPY){if(nargs||nkw){c->error=BAD_CALL;return 1;}int d=stdlib_new_counter(c);if(device_dict_update(c,d,owner))return 1;*out=d;return 1;}
 if(code==8){if(nargs||nkw){c->error=BAD_CALL;return 1;}*out=stdlib_enum_repr(c,owner);return 1;}
 return 0;
}
__device__ inline int stdlib_invoke(Context* c,int fn,int arg,int caller,int acc){
 const int* n=node(c,fn);if(!n){c->error=BAD_CALL;return 0;}int args_buf[2];const int* args=&arg;int nargs=1;
 if(n[0]==18){args_buf[0]=edge(c,fn,1);args_buf[1]=arg;args=args_buf;nargs=2;fn=edge(c,fn,0);n=node(c,fn);if(!n)return 0;}
 if(n[0]==T_FUNCTION||n[0]==T_DYNFN){int fid=n[3],globals=MISSING,closure=MISSING,defaults=MISSING,kwdefaults=MISSING;
  if(n[0]==T_DYNFN){globals=edge(c,fn,0);closure=edge(c,fn,1);defaults=edge(c,fn,2);kwdefaults=edge(c,fn,3);}
  int id=new_frame(c,fid,args,nargs,caller,7,globals,closure,defaults,MISSING,0,0,kwdefaults);if(id<0)return 0;frame(c,id)[6]=acc;
  if(function(c,fid)[7]&32){int gen=meta(c,T_GEN,0,0,id);frame(c,id)[4]=2;frame(c,id)[10]=gen;stdlib_resume(c,MISSING,caller,acc,gen,0);return 1;}
  c->current=id;return 1;}
 int out=MISSING;
 if(n[0]==T_STDLIB_BOUND){if(!stdlib_bound_apply(c,fn,nargs==2?args+1:&arg,1,MISSING,0,0,&out)){c->error=BAD_CALL;return 0;}if(!c->error&&!c->h.error)stdlib_resume(c,MISSING,caller,acc,out,0);return 1;}
 if(native_builtin_call(c,fn,args,nargs,&out)){if(!c->error&&!c->h.error)stdlib_resume(c,MISSING,caller,acc,out,0);return 1;}
 c->error=UNSUPPORTED_TYPE;return 0;
}
__device__ inline void stdlib_resume(Context* c,int generator,int caller,int target,int item,int exhausted){
 if(c->error||c->h.error)return;
 if(kind(c,target)==T_STDLIB_ENUMERATE){if(exhausted){long long pc=0;if(edge(c,target,1)!=NIL&&kind(c,edge(c,target,1))!=TAG_NONE)read_int64(&c->h,edge(c,target,1),&pc);pop(c,frame(c,caller));frame(c,caller)[1]=(int)pc;c->current=caller;return;}
  int idx=node(c,target)[3];int pair[2]={make_int64(&c->h,(long long)idx),item};mutable_node(c,target)[3]=idx+1;push(c,frame(c,caller),make_tuple(&c->h,pair,2));c->current=caller;return;}
 if(kind(c,target)!=T_STDLIB_ACC){c->error=BAD_PROGRAM;return;}int op=edge(c,target,0),src=edge(c,target,2);
 if(op==STDLIB_ACC_SORTED_KEYS){if(exhausted){c->error=BAD_PROGRAM;return;}list_append(&c->h,edge(c,target,5),item);stdlib_acc_set(c,target,6,edge(c,target,6)+1);stdlib_sorted_keys(c,target);return;}
 if(op==STDLIB_ACC_MINKEY||op==STDLIB_ACC_MAXKEY){if(exhausted){stdlib_finish(c,target);return;}if(generator==MISSING){int has=edge(c,target,6),best=edge(c,target,5),cur=edge(c,target,9),is_max=edge(c,target,7);
  if(!has){stdlib_acc_set(c,target,3,cur);stdlib_acc_set(c,target,5,item);stdlib_acc_set(c,target,6,1);}else if(compare(c,item,best,is_max?4:0)){stdlib_acc_set(c,target,3,cur);stdlib_acc_set(c,target,5,item);}if(c->error||c->h.error)return;
  if(kind(c,src)==T_GEN)resume_generator(c,src,caller,7,target,MISSING);else stdlib_minmax_next(c,target);return;}
  stdlib_acc_set(c,target,9,item);stdlib_invoke(c,edge(c,target,4),item,caller,target);return;}
 if(!exhausted){if(kind(c,src)==T_STDLIB_ENUMERATE&&generator!=MISSING){int idx=node(c,src)[3];int pair[2]={make_int64(&c->h,(long long)idx),item};mutable_node(c,src)[3]=idx+1;item=make_tuple(&c->h,pair,2);}
  if(!stdlib_append_raw(c,target,item))return;if(generator!=MISSING&&kind(c,generator)==T_GEN){resume_generator(c,generator,caller,7,target,MISSING);return;}
  if(kind(c,src)==T_STDLIB_ENUMERATE&&kind(c,edge(c,src,0))==T_GEN){resume_generator(c,edge(c,src,0),caller,7,target,MISSING);return;}c->error=BAD_PROGRAM;return;}
 stdlib_finish(c,target);
}
__device__ inline int stdlib_call(Context* c,int callable,const int* args,int nargs,int caller,int kw_names,const int* kw_values,int nkw){
 const int* n=node(c,callable);if(!n)return 0;
 if(n[0]==34&&n[3]==METHOD_LIST_EXTEND&&nargs==1&&!nkw){
  int src=args[0],tag=kind(c,src);
  if(tag==T_GEN||tag==T_ITER||tag==T_STDLIB_ENUMERATE||tag==T_STDLIB_COMB||tag==T_STDLIB_PERM){
   int owner=n[1];if(kind(c,owner)!=TAG_LIST||owner<0){c->error=BAD_CALL;return 1;}
   return stdlib_consume(c,src,STDLIB_ACC_EXTEND,caller,MISSING,0,owner);
  }
 }
 if(n[0]==T_STDLIB_BOUND){int out=MISSING;if(!stdlib_bound_apply(c,callable,args,nargs,kw_names,kw_values,nkw,&out)){c->error=BAD_CALL;return 1;}if(!c->error&&!c->h.error)stdlib_push(c,caller,out);return 1;}
 if(n[0]!=T_NATIVE)return 0;int id=n[3];
 if(NATIVE_STDLIB_UUID>=0&&id==NATIVE_STDLIB_UUID){
  // uuid.uuid4 is not a supported intrinsic. Missing entity identities must stay typed; random UUIDs are not forged.
  c->error=UNSUPPORTED_TYPE;return 1;}
 if(NATIVE_STDLIB_REPR>=0&&id==NATIVE_STDLIB_REPR){
  if(nkw||nargs!=1){c->error=BAD_CALL;return 1;}
  stdlib_push(c,caller,kind(c,args[0])==27?stdlib_enum_repr(c,args[0]):stdlib_scalar_repr(c,args[0]));return 1;
 }
 if(NATIVE_STDLIB_ENUM_NEXT>=0&&id==NATIVE_STDLIB_ENUM_NEXT){
  if(nkw||nargs!=4){c->error=BAD_CALL;return 1;}
  int values=args[3],tag=kind(c,values);if(tag!=TAG_LIST&&tag!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 1;}
  int result=args[1];for(int i=length(c,values)-1;i>=0;i--){int last=edge(c,values,i),t=kind(c,last);
   if(t==TAG_INT||t==TAG_FLOAT||t==TAG_BOOL){result=numeric_binary(c,last,make_int64(&c->h,1),OP_BINARY_ADD);break;}}
  stdlib_push(c,caller,result);return 1;
 }
 if((NATIVE_STDLIB_LIST>=0&&id==NATIVE_STDLIB_LIST)||(NATIVE_STDLIB_TUPLE>=0&&id==NATIVE_STDLIB_TUPLE)){if(nkw){c->error=BAD_CALL;return 1;}if(nargs==1){int tag=kind(c,args[0]);if(tag==T_GEN||tag==T_ITER||tag==T_STDLIB_ENUMERATE||tag==T_STDLIB_COMB||tag==T_STDLIB_PERM){stdlib_consume(c,args[0],id==NATIVE_STDLIB_TUPLE?STDLIB_ACC_TUPLE:STDLIB_ACC_LIST,caller,MISSING,0);return 1;}}return 0;}
 if((NATIVE_STDLIB_MIN>=0&&id==NATIVE_STDLIB_MIN)||(NATIVE_STDLIB_MAX>=0&&id==NATIVE_STDLIB_MAX)){int has_key=0,has_def=0,keyfn=stdlib_kw(c,kw_names,kw_values,nkw,"key",&has_key),def=stdlib_kw(c,kw_names,kw_values,nkw,"default",&has_def);if(stdlib_unknown_kw(c,kw_names,nkw,"key","default",0))return 1;if(!has_key)return 0;if(nargs==0||(nargs>1&&has_def)){c->error=BAD_CALL;return 1;}
  int src=nargs==1?args[0]:make_list(&c->h);if(nargs>1){for(int i=0;i<nargs;i++)list_append(&c->h,src,args[i]);}int acc=stdlib_make_acc(c,id==NATIVE_STDLIB_MAX?STDLIB_ACC_MAXKEY:STDLIB_ACC_MINKEY,caller,src,MISSING,keyfn,id==NATIVE_STDLIB_MAX?1:0);stdlib_acc_set(c,acc,8,has_def?def:MISSING);if(kind(c,src)==T_GEN)resume_generator(c,src,caller,7,acc,MISSING);else stdlib_minmax_next(c,acc);return 1;}
 if(NATIVE_STDLIB_ENUMERATE>=0&&id==NATIVE_STDLIB_ENUMERATE){if(stdlib_unknown_kw(c,kw_names,nkw,"start",0,0)||nargs<1||nargs>2){c->error=BAD_CALL;return 1;}int has_start=0,start=nargs==2?args[1]:stdlib_kw(c,kw_names,kw_values,nkw,"start",&has_start);if(nargs==2&&has_start){c->error=BAD_CALL;return 1;}if(stdlib_is_none(c,start))start=make_int64(&c->h,0);long long s=0;if(read_int64(&c->h,start,&s)<0)return 1;int refs[3]={args[0],NIL,start};int ref=make_tuple(&c->h,refs,3);int* p=mutable_node(c,ref);if(p){p[0]=T_STDLIB_ENUMERATE;p[3]=(int)s;}stdlib_push(c,caller,ref);return 1;}
 if(NATIVE_STDLIB_ISINSTANCE>=0&&id==NATIVE_STDLIB_ISINSTANCE){if(nkw||nargs!=2){c->error=BAD_CALL;return 1;}stdlib_push(c,caller,stdlib_match_type(c,args[0],args[1])?TRUE_REF:FALSE_REF);return 1;}
 if(NATIVE_STDLIB_GETATTR>=0&&id==NATIVE_STDLIB_GETATTR){if(nkw||nargs<2||nargs>3){c->error=BAD_CALL;return 1;}int name=stdlib_attribute_name(c,args[1]);if(c->error||c->h.error)return 1;int getter=0,attr=stdlib_lookup(c,args[0],name,&getter);if(c->error||c->h.error)return 1;if(getter){c->error=UNSUPPORTED_TYPE;return 1;}if(attr!=MISSING){stdlib_push(c,caller,attr);return 1;}if(nargs==3){stdlib_push(c,caller,args[2]);return 1;}stdlib_attr_missing(c);return 1;}
 if(NATIVE_STDLIB_HASATTR>=0&&id==NATIVE_STDLIB_HASATTR){if(nkw||nargs!=2){c->error=BAD_CALL;return 1;}int name=stdlib_attribute_name(c,args[1]);if(c->error||c->h.error)return 1;int getter=0,attr=stdlib_lookup(c,args[0],name,&getter);if(c->error||c->h.error)return 1;stdlib_push(c,caller,attr==MISSING?FALSE_REF:TRUE_REF);return 1;}
 if(NATIVE_STDLIB_SUPER>=0&&id==NATIVE_STDLIB_SUPER){if(nkw||(nargs!=0&&nargs!=2)){c->error=BAD_CALL;return 1;}int type_ref=MISSING,obj=MISSING;if(nargs==2){type_ref=args[0];obj=args[1];}else{int* f=frame(c,caller);const int* fn=function(c,f[0]);if(!fn)return 1;obj=f[LBASE];int nc=length(c,fn[11]),nf=length(c,fn[12]);for(int i=0;i<nc+nf;i++){int name=i<nc?edge(c,fn[11],i):edge(c,fn[12],i-nc);if(vm_text_equals(c,name,"__class__")){int cell=f[CBASE+i];const int* cn=node(c,cell);if(cn&&cn[0]==T_CELL)type_ref=cn[1];break;}}if(stdlib_is_none(c,type_ref)){c->error=BAD_CALL;return 1;}}int refs[2]={type_ref,obj};int sp=make_tuple(&c->h,refs,2);int* p=mutable_node(c,sp);if(p)p[0]=T_STDLIB_SUPER;stdlib_push(c,caller,sp);return 1;}
 if(NATIVE_STDLIB_SORTED>=0&&id==NATIVE_STDLIB_SORTED){int has_key=0,has_rev=0,keyfn=stdlib_kw(c,kw_names,kw_values,nkw,"key",&has_key),rev=stdlib_kw(c,kw_names,kw_values,nkw,"reverse",&has_rev);if(stdlib_unknown_kw(c,kw_names,nkw,"key","reverse",0)||nargs!=1){c->error=BAD_CALL;return 1;}int reverse=0;if(has_rev)reverse=truth_value(c,rev);stdlib_consume(c,args[0],STDLIB_ACC_SORTED,caller,has_key?keyfn:MISSING,reverse);return 1;}
 if((NATIVE_STDLIB_SET>=0&&id==NATIVE_STDLIB_SET)||(NATIVE_STDLIB_FROZENSET>=0&&id==NATIVE_STDLIB_FROZENSET)){if(nkw||nargs>1){c->error=BAD_CALL;return 1;}int frozen=id==NATIVE_STDLIB_FROZENSET;if(nargs==0){int s=make_set(c);if(frozen){int* p=mutable_node(c,s);if(p)p[0]=22;}stdlib_push(c,caller,s);return 1;}stdlib_consume(c,args[0],frozen?STDLIB_ACC_FROZEN:STDLIB_ACC_SET,caller,MISSING,0);return 1;}
 if(stdlib_is_counter_id(id)){if(nargs>1){c->error=BAD_CALL;return 1;}if(nargs==1&&nkw&&stdlib_async_src(c,args[0])){c->error=BAD_CALL;return 1;}if(nargs==1&&stdlib_async_src(c,args[0])){stdlib_consume(c,args[0],STDLIB_ACC_COUNTER,caller,MISSING,0);return 1;}int d=stdlib_new_counter(c);if(nargs==1){int src=args[0],st=kind(c,src);if(st==T_STDLIB_COUNTER||st==TAG_DICT||(st>=8&&st<=10)){int cnt=length(c,src);for(int i=0;i<cnt;i++)if(!stdlib_counter_add(c,d,edge(c,src,i),edge(c,src,i,1)))return 1;}else if(st==TAG_LIST||st==TAG_TUPLE||st==21||st==22){int cnt=length(c,src);for(int i=0;i<cnt;i++)if(!stdlib_counter_add(c,d,edge(c,src,i),MISSING))return 1;}else{c->error=UNSUPPORTED_TYPE;return 1;}}for(int i=0;i<nkw;i++)if(!stdlib_counter_add(c,d,edge(c,kw_names,i),kw_values[i]))return 1;stdlib_push(c,caller,d);return 1;}
 if((NATIVE_STDLIB_COMBINATIONS>=0&&id==NATIVE_STDLIB_COMBINATIONS)||(NATIVE_STDLIB_PERMUTATIONS>=0&&id==NATIVE_STDLIB_PERMUTATIONS)){int perm=id==NATIVE_STDLIB_PERMUTATIONS,has_r=0,rref=nargs>=2?args[1]:stdlib_kw(c,kw_names,kw_values,nkw,"r",&has_r);if(stdlib_unknown_kw(c,kw_names,nkw,"r",0,0)||nargs<1||nargs>2||(nargs==2&&has_r)){c->error=BAD_CALL;return 1;}if(stdlib_async_src(c,args[0])){long long r=0;if(stdlib_is_none(c,rref)){c->error=UNSUPPORTED_TYPE;return 1;}if(read_int64(&c->h,rref,&r)<0)return 1;stdlib_consume(c,args[0],perm?STDLIB_ACC_PERM_POOL:STDLIB_ACC_COMB_POOL,caller,MISSING,(int)r);return 1;}int pool=args[0];if(kind(c,pool)!=TAG_TUPLE&&kind(c,pool)!=TAG_LIST){c->error=UNSUPPORTED_TYPE;return 1;}long long r=0;if(stdlib_is_none(c,rref)){if(!perm){c->error=BAD_CALL;return 1;}r=length(c,pool);}else if(read_int64(&c->h,rref,&r)<0)return 1;stdlib_push(c,caller,stdlib_combo_init(c,pool,(int)r,perm));return 1;}
 if(stdlib_is_isfinite_id(id)){if(nkw||nargs!=1){c->error=BAD_CALL;return 1;}int t=kind(c,args[0]);if(t==TAG_INT||t==TAG_BOOL){stdlib_push(c,caller,TRUE_REF);return 1;}if(t==TAG_FLOAT){double v=0;if(read_float64(&c->h,args[0],&v))return 1;stdlib_push(c,caller,vm_isfinite(v)?TRUE_REF:FALSE_REF);return 1;}c->error=UNSUPPORTED_TYPE;return 1;}
 if(NATIVE_STDLIB_ENUM_REPR>=0&&id==NATIVE_STDLIB_ENUM_REPR){if(nkw||nargs!=1){c->error=BAD_CALL;return 1;}stdlib_push(c,caller,stdlib_enum_repr(c,args[0]));return 1;}
 if(NATIVE_STDLIB_CLONE>=0&&id==NATIVE_STDLIB_CLONE){if(nkw||nargs!=1){c->error=BAD_CALL;return 1;}int out=stdlib_clone_hydrated(c,args[0]);if(out==MISSING&&!c->error&&!c->h.error)c->error=UNSUPPORTED_TYPE;if(!c->error&&!c->h.error)stdlib_push(c,caller,out);return 1;}
 if(NATIVE_STDLIB_ENSURE_KITS>=0&&id==NATIVE_STDLIB_ENSURE_KITS){if(nkw||nargs){c->error=BAD_CALL;return 1;}if(STDLIB_KITS_MODULE<0||STDLIB_KITS_COUNT<=0||STDLIB_KITS_MODULE>=c->module_count){c->error=UNSUPPORTED_TYPE;return 1;}int kits=dict_text_get(c,c->modules[STDLIB_KITS_MODULE],"KITS");if(kits==MISSING||kind(c,kits)!=TAG_DICT||length(c,kits)!=STDLIB_KITS_COUNT){c->error=UNSUPPORTED_TYPE;return 1;}stdlib_push(c,caller,NIL);return 1;}
 if(NATIVE_STDLIB_ENUM>=0&&id==NATIVE_STDLIB_ENUM){c->error=UNSUPPORTED_TYPE;return 1;}
 return 0;
}
__device__ inline int stdlib_attribute(Context* c,int owner,int name){
 int tag=kind(c,owner);
 if(tag==27){if(vm_text_equals(c,name,"name"))return edge(c,owner,2);if(vm_text_equals(c,name,"value"))return edge(c,owner,1);if(vm_text_equals(c,name,"__repr__"))return stdlib_bind(c,owner,8);return MISSING;}
 if(tag==T_STDLIB_SUPER){int getter=0,found=lookup_super(c,edge(c,owner,0),edge(c,owner,1),name,&getter);if(found!=MISSING||c->error||c->h.error)return found;if(vm_text_equals(c,name,"__init__")){int obj=edge(c,owner,1),cls=class_ref(c,obj);if(cls!=MISSING&&kind(c,cls)==13&&class_dict_subclass(node(c,cls)[3]))return stdlib_bind(c,obj,STDLIB_METHOD_DICT_INIT);return stdlib_bind(c,obj,STDLIB_METHOD_OBJECT_INIT);}return MISSING;}
 if(tag==T_STDLIB_COUNTER){if(vm_text_equals(c,name,"update"))return stdlib_bind(c,owner,STDLIB_METHOD_COUNTER_UPDATE);if(vm_text_equals(c,name,"copy"))return stdlib_bind(c,owner,STDLIB_METHOD_COUNTER_COPY);if(vm_text_equals(c,name,"get"))return meta(c,34,owner,0,1);if(vm_text_equals(c,name,"keys"))return meta(c,34,owner,0,2);if(vm_text_equals(c,name,"values"))return meta(c,34,owner,0,3);if(vm_text_equals(c,name,"items"))return meta(c,34,owner,0,4);return MISSING;}
 if(tag==21){if(vm_text_equals(c,name,"add"))return stdlib_bind(c,owner,STDLIB_METHOD_SET_ADD);if(vm_text_equals(c,name,"update"))return stdlib_bind(c,owner,STDLIB_METHOD_SET_UPDATE);return MISSING;}
 if(tag==T_STDLIB_BOUND&&node(c,owner)[3]==0&&vm_text_equals(c,name,"__call__"))return owner;
 return MISSING;
}
__device__ inline int stdlib_compare(Context* c,int a,int b,int mode,int* out){
 int ta=kind(c,a),tb=kind(c,b);
 if(ta==27||tb==27){if(mode==2||mode==3){*out=stdlib_enum_eq(c,a,b);if(mode==3)*out=!*out;return 1;}c->error=UNSUPPORTED_TYPE;*out=0;return 1;}
 if((ta==21||ta==22)&&(tb==21||tb==22)){if(mode==2||mode==3){*out=stdlib_set_eq(c,a,b);if(mode==3)*out=!*out;return 1;}c->error=UNSUPPORTED_TYPE;*out=0;return 1;}
 if(ta==T_STDLIB_COUNTER&&tb==T_STDLIB_COUNTER&&(mode==2||mode==3)){*out=stdlib_mapping_eq(c,a,b);if(mode==3)*out=!*out;return 1;return 1;}
 if((ta==TAG_TUPLE||ta==TAG_LIST)&&ta==tb&&mode!=2&&mode!=3){int cmp=stdlib_seq_cmp(c,a,b);if(mode==0)*out=cmp<0;else if(mode==1)*out=cmp<=0;else if(mode==4)*out=cmp>0;else if(mode==5)*out=cmp>=0;else{c->error=UNSUPPORTED_OP;*out=0;}return 1;}
 return 0;
}
__device__ inline int stdlib_binary(Context* c,int a,int b,int op,int* out){
 int difference=op==OP_BINARY_SUBTRACT||op==OP_INPLACE_SUBTRACT;
 int intersection=op==OP_BINARY_AND||op==OP_INPLACE_AND;
 int union_=op==OP_BINARY_OR||op==OP_INPLACE_OR;
 int symmetric=op==OP_BINARY_XOR||op==OP_INPLACE_XOR;
 if(!difference&&!intersection&&!union_&&!symmetric)return 0;
 int ta=kind(c,a),tb=kind(c,b);if(ta!=21&&ta!=22)return 0;
 if(tb!=21&&tb!=22){c->error=UNSUPPORTED_TYPE;return 1;}
 int dest=make_set(c),n=length(c,a);
 for(int i=0;i<n&&!c->error&&!c->h.error;i++){
  int item=edge(c,a,i),present=set_contains(c,item,b);
  if(union_||(intersection?present:!present))set_add(c,dest,item);
 }
 if(union_||symmetric)for(int i=0;i<length(c,b)&&!c->error&&!c->h.error;i++){
  int item=edge(c,b,i);if(union_||!set_contains(c,item,a))set_add(c,dest,item);
 }
 if(c->error||c->h.error)return 1;
 int inplace=op==OP_INPLACE_SUBTRACT||op==OP_INPLACE_AND||op==OP_INPLACE_OR||op==OP_INPLACE_XOR;
 if(inplace&&ta==21){
  int* original=mutable_node(c,a);if(!original)return 1;original[2]=0;
  for(int i=0;i<length(c,dest)&&!c->error&&!c->h.error;i++)set_add(c,a,edge(c,dest,i));
  *out=a;
 }else{if(ta==22){int* p=mutable_node(c,dest);if(p)p[0]=22;}*out=dest;}
 return 1;
}
__device__ inline int stdlib_subscript(Context* c,int owner,int key,int* out){if(kind(c,owner)!=T_STDLIB_COUNTER)return 0;int v=dict_get(&c->h,owner,key,MISSING);if(c->h.error)return 1;if(v==MISSING)v=make_int64(&c->h,0);*out=v;return 1;}
__device__ inline int stdlib_store(Context* c,int owner,int key,int val){if(kind(c,owner)!=T_STDLIB_COUNTER)return 0;long long n=0;if(read_int64(&c->h,val,&n)<0)return 1;dict_set(&c->h,owner,key,val);return 1;}
__device__ inline int stdlib_handle_opcode(Context* c,int* f,int op,int arg,int value,int target){
 if(op==OP_FOR_ITER)return stdlib_for_iter(c,peek(c,f),target);
 if(op==OP_GET_ITER){int owner=peek(c,f),tag=kind(c,owner);if(tag==T_STDLIB_ENUMERATE||tag==T_STDLIB_COMB||tag==T_STDLIB_PERM){pop(c,f);push(c,f,owner);return 1;}return 0;}
 if(op==OP_STORE_SUBSCR){if(f[2]<3)return 0;int owner=f[SBASE+f[2]-2];if(kind(c,owner)!=T_STDLIB_COUNTER)return 0;int key=pop(c,f);owner=pop(c,f);int val=pop(c,f);return stdlib_store(c,owner,key,val);}
 if(op==OP_BINARY_SUBSCR){if(f[2]<2)return 0;int owner=f[SBASE+f[2]-2];if(kind(c,owner)!=T_STDLIB_COUNTER)return 0;int key=pop(c,f);owner=pop(c,f);int out=MISSING;if(!stdlib_subscript(c,owner,key,&out))return 0;push(c,f,out);return 1;}
 return 0;
}
'''
