"""Opt-in, read-only typed hydration proof in one native rule owner.

The canonical function remains the MISS path. Native scan work consumes VM
steps in addition to ordinary opcodes. HIT skips temporary canonical allocations;
this is a distinct program source identity, not allocation/opcode equivalence.
No CUDA execution or training qualification follows from enabling this pass.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import ast
import inspect
import json
from pathlib import Path
import re
import textwrap

from .native_array_codegen import _matching_brace_end
from .native_compile_units import NativeCompileUnit
from .record_soa import CompiledRecordSchema

_MODULE = 'app.modules.card_game.engine.duel_v2.entities'
_NAMESPACE = 'native_hydration_probe_detail'
# Structural contracts for the supported algorithm, independent of function
# numbering, source line numbers, whole-program identity, and pool addresses.
_CANONICAL_AST = 'abda4173d6df74b0968e7142d36966e7ba7c41d02eb70bc3f98168212870c978'
_CANONICAL_FUNCTIONS = {'CardEntity.template_id': '83301e43777603be4776ee43de736da0ddcc9448c68b264860ef7be79e3dbf2b',
 'CharacterEntity.template_id': '237c293e64d7822dab5db08b5f68854f33dfa90d8ee9a5907b9a1aae04553f26',
 'Entity.__init__': 'fa7c1a60ba468b7eb792250d7b7639c1e1485a726a96df4a0b8dd106bb6c4153',
 'Entity.entity_id': 'd343e5e77b7fd51119735e8af4826bf2d565887ce98245b040bd8a69d3782b84',
 'Entity.is_same_entity': '8d9d6538136b9c835195ae24f1039ddb4214f97fa6528c72111aa58d7543045a',
 'Entity.side': '43b434f2b4443374e4c95b7558edb90a09e65ca38bf89d7a60a7d659ec6308d6',
 'allocate_entity_id': '3e1ef412d1e705c976176492cafc4d198708a24cfb665127a751eaf05d7e1de8',
 'hydrate_entities': 'cd46701e884473f014d052ff74257242976ae24723dcb92457378d3ea1e092a1'}
_CANONICAL_CLASSES = {'CardEntity': '29a1f1072d3adad5a5e69ed85702021d6a133102d145beb68b56943e0f7be228',
 'CharacterEntity': '11c5844bd08a415fe5f8e6a48310dba40c461aa76a6dc9efcecc2e4b4d44b607',
 'Entity': '8c5329b53b37484556689b8b5401797ab33fabd332e4b4b683eb9e14ce331d49',
 'PlayerEntity': 'e873ec4cd314e5b3fdc564b30e1f8baedf1473dc2744028dd49f43f3c7e14bcd'}


def _hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def _reference_name(identifier):
    if type(identifier) is not str:raise ValueError('Canonical symbolic reference required')
    parts=identifier.split(':')
    if len(parts)<4 or parts[0] not in ('func','class','code'):
        raise ValueError('Canonical symbolic reference required')
    return ':'.join(parts[:3])


def _normalize(value, constants, seen=()):
    if isinstance(value,list):return [_normalize(x,constants,seen)for x in value]
    if not isinstance(value,dict):return value
    kind=value.get('kind')
    if kind=='constant':
        key=value.get('id')
        if key in seen or key not in constants:raise ValueError('Finite canonical constants required')
        item={k:v for k,v in constants[key].items()if k!='id'}
        return _normalize(item,constants,seen+(key,))
    if kind in ('function','class','code'):
        return dict(kind=kind,id=_reference_name(value['id']))
    return {k:_normalize(v,constants,seen)for k,v in value.items()}


def _function_structure(fn, constants):
    keys=('module','qualname','name','argcount','posonlyargcount','kwonlyargcount',
          'nlocals','stacksize','varnames','freevars','cellvars','flags',
          'instructions','globals','defaults','kwdefaults','closure')
    result={k:_normalize(fn[k],constants)for k in keys}
    values=[_normalize(x,constants)for x in fn['constants']]
    # Only an unreferenced leading string is a docstring. None at index zero
    # is often executed (e.g. constructor side checks and implicit returns).
    loaded_zero=any(row['op']=='LOAD_CONST'and row['arg']==0 for row in fn['instructions'])
    if values and isinstance(values[0],dict) and values[0].get('kind')=='literal' and values[0].get('type')=='str' and not loaded_zero:
        values[0]=None
    result['constants']=values
    return result


def _class_structure(cls, constants):
    return _normalize({k:v for k,v in cls.items()if k!='id'},constants)


def _stable_ast(value):
    """Version-independent semantic AST, including explicit empty/None fields.

    Python 3.12 added empty type_params to ordinary definitions. Nonempty type
    parameters change the supported contract and are rejected, not erased.
    """
    if isinstance(value,ast.AST):
        fields={}
        for name in value._fields:
            item=getattr(value,name,None)
            if name=='type_params':
                if item not in(None,[]):raise ValueError('Generic hydration AST is unsupported')
                continue
            fields[name]=_stable_ast(item)
        return dict(node=type(value).__name__,fields=fields)
    if isinstance(value,(list,tuple)):return [_stable_ast(x)for x in value]
    if value is None or type(value) in(bool,int,float,str):return value
    raise ValueError('Unsupported canonical AST field')


def _canonical_ast():
    from ...engine.duel_v2.entities import hydrate_entities
    tree=ast.parse(textwrap.dedent(inspect.getsource(hydrate_entities)))
    fn=tree.body[0]
    if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):
        fn.body.pop(0)
    return _hash(_stable_ast(tree))


_NATIVE_SCAN = r'''
// All stores are confined to this activation's automatic variables. No node(),
// edge(), boxing, dictionary, view refresh, cache, or promotion helpers called.
struct Probe {
 Context* c;const DomainSoAContext* s;int work,limit;bool stopped;
 __device__ bool tick(){
  if(stopped)return false;
  if(work>=limit){stopped=true;return false;}
  if((work&63)==0 && c->deadline && clock_ns()>=(unsigned long long)c->deadline){stopped=true;return false;}
  ++work;return true;
 }
 __device__ int i32(int off){return s->i32[(long long)off*s->rows+s->row];}
 __device__ int u8(int off){return s->u8[(long long)off*s->rows+s->row];}
 __device__ long long i64(int off){return s->i64[(long long)off*s->rows+s->row];}
 __device__ int tag(Col col,int at){if(!tick())return -1;return u8(col.tag+at);}
 __device__ int word(Col col,int at){if(col.word<0||!tick())return -1;return i32(col.word+at);}
 __device__ bool slot(Pool p,int gid,int want_tag,int* out){
  if(!tick())return false;
  int used=i32(p.used);long long wide=(long long)gid-p.base-1;
  if(wide<0||wide>=p.capacity)return false;int n=(int)wide;
  if(used<0||used>p.capacity||n<0||n>=used||u8(p.tag+n)!=want_tag)return false;
  int count=i32(p.count+n);
  if(count<0||count>p.max_items)return false;
  // Validate the record order/presence invariant before relying on columns.
  if(p.nfields){
   bool seen[256]={};
   for(int i=0;i<count;i++){
    if(!tick())return false;int f=i32(p.order+n*p.max_items+i);
    if(f<0||f>=p.nfields||seen[f])return false;seen[f]=true;
   }
   for(int i=0;i<p.nfields;i++){
    if(!tick())return false;int t=u8(p.fields[i].tag+n);
    if(t<0||t>6||bool(t)!=seen[i])return false;
   }
  }
  *out=n;return true;
 }
 __device__ bool ref(Col col,int at,int* out){
  if(tag(col,at)!=6)return false;*out=word(col,at);return !stopped&&*out>0;
 }
 __device__ bool text(int sid,int* start,int* length){
  if(!tick())return false;
  int ns=i32(strings_count),nb=i32(bytes_used);
  if(ns<0||ns>max_strings||nb<0||nb>max_string_bytes||sid<=0||sid>ns)return false;
  int p=i32(string_offsets+sid-1),n=i32(string_lengths+sid-1);
  if(p<0||n<0||p>nb||n>nb-p)return false;
  *start=p;*length=n;return true;
 }
 __device__ int byte(int start,int at){return u8(string_bytes+start+at);}
 __device__ bool ascii(int sid,int maxlen){
  int p,n;if(!text(sid,&p,&n)||n<=0||n>maxlen)return false;
  for(int i=0;i<n;i++){if(!tick())return false;int b=byte(p,i);if(b<=0||b>127)return false;}return true;
 }
 __device__ bool equal(int a,int b){
  int p,n,q,m;if(!text(a,&p,&n)||!text(b,&q,&m)||n!=m)return false;
  for(int i=0;i<n;i++){if(!tick()||byte(p,i)!=byte(q,i))return false;}return true;
 }
 __device__ bool side(Col col,int at,int expected){
  if(tag(col,at)!=5)return false;int p,n;
  return text(word(col,at),&p,&n)&&n==1&&tick()&&byte(p,0)==('a'+expected);
 }
 __device__ bool eid(Col col,int at,long long* result){
  if(tag(col,at)!=5)return false;int p,n;
  if(!text(word(col,at),&p,&n)||n<2||n>20||!tick()||byte(p,0)!='e')return false;
  long long value=0;
  for(int i=1;i<n;i++){
   if(!tick())return false;int d=byte(p,i)-'0';
   if(d<0||d>9||(i==1&&d==0)||value>(9223372036854775807LL-d)/10)return false;
   value=value*10+d;
  }
  *result=value;return value>0;
 }
};
struct Seen {int gid,kind,side,key;long long eid;};
struct Scan {
 Probe p;Seen seen[128];int size;long long high;
 __device__ bool entity(int gid,int kind,int side,int key){
  int slot;Pool pool=kind==0?pool_Player:(kind==1?pool_Character:pool_Card);
  if(!p.slot(pool,gid,kind==0?8:(kind==1?9:10),&slot))return false;
  Col eidcol=kind==0?Player_entity_id:(kind==1?Character_entity_id:Card_entity_id);
  Col sidecol=kind==0?Player_side:(kind==1?Character_side:Card_side);
  long long eid;if(!p.eid(eidcol,slot,&eid)||eid>high)return false;
  for(int i=0;i<size;i++){
   if(!p.tick())return false;Seen v=seen[i];
   bool same=v.kind==kind && (kind==2 || v.side==side);
   if(same&&kind!=0)same=p.equal(v.key,key);
   if(p.stopped)return false;
   if(same){
    // Only card slots may repeat a logical identity. A distinct ref needs
    // canonical alias replacement, even when its bytes happen to be equal.
    return kind==2 && v.gid==gid && v.eid==eid;
   }
   if(v.eid==eid)return false;
  }
  if(size>=128||!p.side(sidecol,slot,side))return false;
  seen[size++]={gid,kind,side,key,eid};return true;
 }
 __device__ bool card(int gid,int side){
  int slot;if(!p.slot(pool_Card,gid,10,&slot)||p.tag(Card_instance_id,slot)!=5)return false;
  int key=p.word(Card_instance_id,slot);
  return p.ascii(key,128)&&entity(gid,2,side,key);
 }
 __device__ bool cards(int gid,int side){
  int slot;if(!p.slot(pool_CardList,gid,5,&slot))return false;
  int n=p.i32(pool_CardList.count+slot);
  for(int i=0;i<n;i++){int ref;if(!p.ref(CardList_values,slot*pool_CardList.max_items+i,&ref)||!card(ref,side))return false;}return true;
 }
 __device__ bool zone(Col col,int slot,int side){
  int t=p.tag(col,slot);if(t==0)return true;int gid;
  return t==6&&p.ref(col,slot,&gid)&&cards(gid,side);
 }
 __device__ bool team(int gid,int side){
  if(!entity(gid,0,side,0))return false;int slot,chars,cs;
  if(!p.slot(pool_Player,gid,8,&slot)||!p.ref(Player_characters,slot,&chars)||!p.slot(pool_CharacterMap,chars,7,&cs))return false;
  int n=p.i32(pool_CharacterMap.count+cs);
  for(int i=0;i<n;i++){
   int at=cs*pool_CharacterMap.max_items+i,key,child;
   if(p.tag(CharacterMap_keys,at)!=5)return false;key=p.word(CharacterMap_keys,at);
   if(!p.ascii(key,128)||!p.ref(CharacterMap_values,at,&child)||!entity(child,1,side,key))return false;
  }
  return zone(Player_hand,slot,side)&&zone(Player_deck,slot,side)&&zone(Player_discard,slot,side)&&zone(Player_removed,slot,side);
 }
 __device__ bool optional(int root,Col field,bool choice){
  int t=p.tag(field,root);if(t==0||t==1)return true;
  int gid,slot;if(t!=6||!p.ref(field,root,&gid))return false;
  Pool pool=choice?pool_PendingChoice:pool_Operation;
  if(!p.slot(pool,gid,7,&slot))return false;
  if(p.i32(pool.count+slot)==0)return true;
  Col sidecol=choice?PendingChoice_side:Operation_side;
  Col valuecol=choice?PendingChoice_cards:Operation_card;
  t=p.tag(valuecol,slot);if(t==0||(!choice&&t==1))return true;
  int child;if(!p.ref(valuecol,slot,&child))return false;
  int cs;
  if(choice){
   if(!p.slot(pool_CardList,child,5,&cs))return false;
   if(p.i32(pool_CardList.count+cs)==0)return true;
  }
  int side=p.side(sidecol,slot,0)?0:(p.side(sidecol,slot,1)?1:-1);
  if(side<0)return false;
  return choice?cards(child,side):card(child,side);
 }
 __device__ bool root(int ref){
  Context* c=p.c;
  if(c->h.counts[0]<0||c->h.counts[0]>c->h.max_nodes||ref<0||ref>=c->h.counts[0])return false;
  const int* node=c->h.nodes+((long long)ref)*4;
  if(node[0]!=7||node[1]!=(-2147483647-1))return false;
  int slot,gid=node[3],sides,ss;
  if(!p.slot(pool_Game,gid,7,&slot)||p.tag(Game_next_entity,slot)!=3||!p.tick())return false;
  high=p.i64(Game_next_entity.integer+slot);if(high<0)return false;
  if(!p.ref(Game_sides,slot,&sides)||!p.slot(pool_Sides,sides,7,&ss)||p.i32(pool_Sides.count+ss)!=2)return false;
  // Use canonical record insertion order, not hardcoded a then b.
  for(int i=0;i<2;i++){
   if(!p.tick())return false;int side=p.i32(pool_Sides.order+ss*pool_Sides.max_items+i),teamref;
   if(side<0||side>1||!p.ref(side==0?Sides_a:Sides_b,ss,&teamref)||!team(teamref,side))return false;
  }
  return optional(slot,Game_operation,false)&&optional(slot,Game_pending_choice,true)&&!p.stopped;
 }
};
__device__ __noinline__ bool probe(Context* c,int state,int* work){
 *work=0;DomainSoAContext* s=c->h.soa;
 if(!s||!s->i64||!s->i32||!s->u8||s->rows<=0||s->row<0||s->row>=s->rows||!c->h.nodes||!c->h.counts)return false;
 // 64 spare admissions prevent starting a scan at an already exhausted edge.
 // Scan stops without errors before spending more than the actual remaining.
 if(c->steps<0||c->step_limit<c->steps||c->step_limit-c->steps<64)return false;
 int remaining=c->step_limit-c->steps;
 Scan scan;scan.p={c,s,0,remaining<32768?remaining:32768,false};scan.size=0;scan.high=0;
 bool hit=scan.root(state);*work=scan.p.work;return hit&&!scan.p.stopped;
}
'''


def _validate_schema(schema):
    if not isinstance(schema,CompiledRecordSchema):
        raise ValueError('Compiled closed typed schema required')
    # Rebuild declarative offsets to reject mutated/stale pool/column metadata.
    caps={p.name:p.capacity for p in schema.pools}
    caps.update({p.name+'.items':p.max_items for p in schema.pools if p.kind in('map','list')})
    fresh=CompiledRecordSchema(schema.definitions,root=schema.names[schema.root],
        capacities=caps,max_strings=schema.max_strings,max_string_bytes=schema.max_string_bytes)
    attrs=('names','root','pools','type_ids','lengths','total_objects',
           'strings_count','bytes_used','string_offsets','string_lengths','string_static_refs',
           'string_hashes','string_bytes','max_strings','max_string_bytes')
    if any(getattr(schema,k)!=getattr(fresh,k)for k in attrs):
        raise ValueError('Typed schema metadata/layout is not canonical')
    names=('Game','Sides','Player','CharacterMap','Character','CardList','Card','Operation','PendingChoice')
    if schema.names[schema.root]!='Game' or any(n not in schema.type_ids for n in names):
        raise ValueError('Required hydration pools/root missing')
    pools={n:schema.pools[schema.type_ids[n]]for n in names}
    expected={'Game':('record',(7,)),'Sides':('record',(7,)),
        'Player':('record',(8,7)),'Character':('record',(9,7)),'Card':('record',(10,7)),
        'CharacterMap':('map',(7,)),'CardList':('list',(5,)),
        'Operation':('record',(7,)),'PendingChoice':('record',(7,))}
    for name,pool in pools.items():
        kind,tags=expected[name]
        if pool.kind!=kind or pool.tags!=tags or pool.max_items>256 or len(pool.fields)>256:
            raise ValueError('Unsupported hydration pool shape: '+name)
    maximum=2**31-1
    def integer(value,low,high,label):
        if type(value) is not int or not low<=value<=high:
            raise ValueError('Bounded int32 schema metadata required: '+label)
    for kind in('i64','f64','i32','u8'):
        integer(schema.lengths[kind],1,maximum,'length '+kind)
    integer(schema.total_objects,1,maximum,'total objects')
    def region(offset,extent,kind,label,nullable=False):
        integer(offset,-1 if nullable else 0,maximum,label)
        integer(extent,1,maximum,label+' extent')
        if offset!=-1 and (offset>=schema.lengths[kind] or extent>schema.lengths[kind]-offset):
            raise ValueError('Schema region outside typed buffer: '+label)
    for name,p in pools.items():
        integer(p.base,0,maximum,name+' base');integer(p.capacity,1,maximum,name+' capacity')
        if p.base>schema.total_objects-p.capacity:raise ValueError('Pool outside typed object range')
        integer(p.max_items,1,256,name+' items')
        region(p.used,1,'i32',name+' used')
        region(p.tag,p.capacity,'u8',name+' tag');region(p.count,p.capacity,'i32',name+' count')
        integer(p.order,-1,maximum,name+' order')
        if p.fields:region(p.order,p.capacity*p.max_items,'i32',name+' order')
        elif p.order!=-1:raise ValueError('Unexpected sequence order metadata')
        for c in(*p.fields,*(()if p.keys is None else(p.keys,)),*p.values):
            if c in p.fields and (not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',c.name) or not c.name.startswith(name+'.')):
                raise ValueError('ASCII C++ field identifier required: '+str(c.name))
            integer(c.stride,1,maximum,c.name+' stride');integer(c.extent,1,maximum,c.name+' extent')
            region(c.tag,c.extent,'u8',c.name+' tag')
            region(c.integer,c.extent,'i64',c.name+' integer',nullable=True)
            region(c.word,c.extent,'i32',c.name+' word',nullable=True)
    integer(schema.max_strings,1,maximum,'max strings');integer(schema.max_string_bytes,1,maximum,'max bytes')
    for field,extent in(('strings_count',1),('bytes_used',1),('string_offsets',schema.max_strings),('string_lengths',schema.max_strings)):
        region(getattr(schema,field),extent,'i32',field)
    region(schema.string_bytes,schema.max_string_bytes,'u8','string bytes')
    if tuple(f.name for f in pools['Sides'].fields)!=('Sides.a','Sides.b'):
        raise ValueError('Exactly ordered a/b sides required')
    fields={'Game':{'next_entity':(8,()),'sides':(0,('Sides',)),
            'operation':(0,('Operation',)),'pending_choice':(0,('PendingChoice',))},
        'Sides':{'a':(0,('Player',)),'b':(0,('Player',))},
        'Player':{'entity_id':(32,()),'side':(32,()),'characters':(0,('CharacterMap',)),
            **{n:(0,('CardList',))for n in('hand','deck','discard','removed')}},
        'Character':{'entity_id':(32,()),'side':(32,())},
        'Card':{'entity_id':(32,()),'side':(32,()),'instance_id':(32,())},
        'Operation':{'side':(32,()),'card':(0,('Card',))},
        'PendingChoice':{'side':(32,()),'cards':(0,('CardList',))}}
    def column(col,mask,refs):
        return col.scalar_mask==mask and col.references==tuple(schema.type_ids[n]for n in refs)
    for name,required in fields.items():
        columns={c.name.split('.',1)[1]:c for c in pools[name].fields}
        if any(n not in columns or not column(columns[n],*spec)for n,spec in required.items()):
            raise ValueError('Unsupported hydration field type/reference: '+name)
    if not column(pools['CharacterMap'].keys,32,()) or not column(pools['CharacterMap'].values[0],0,('Character',)):
        raise ValueError('Canonical character map required')
    if not column(pools['CardList'].values[0],0,('Card',)):
        raise ValueError('Canonical card list required')
    return pools


def _helper_source(schema):
    pools=_validate_schema(schema)
    lines=['\nnamespace '+_NAMESPACE+' {','struct Col {int tag,integer,word;};',
        'struct Pool {int base,capacity,used,tag,count,order,max_items,nfields;const Col* fields;};']
    def col(c):return '{'+','.join(map(str,(c.tag,c.integer,c.word)))+'}'
    for name,p in pools.items():
        if p.fields:
            lines.append('__device__ const Col fields_'+name+'[] = {'+','.join(col(c)for c in p.fields)+'};')
        lines.append('__device__ const Pool pool_'+name+' = {'+
            ','.join(map(str,(p.base,p.capacity,p.used,p.tag,p.count,p.order,p.max_items,len(p.fields))))+
            ', '+('fields_'+name if p.fields else '0')+'};')
        for c in p.fields:lines.append('__device__ const Col '+c.name.replace('.','_')+' = '+col(c)+';')
        if p.keys:lines.append('__device__ const Col '+name+'_keys = '+col(p.keys)+';')
        if p.values:lines.append('__device__ const Col '+name+'_values = '+col(p.values[0])+';')
    for attr in('strings_count','bytes_used','string_offsets','string_lengths','string_bytes','max_strings','max_string_bytes'):
        lines.append('const int '+attr+'='+str(getattr(schema,attr))+';')
    return '\n'.join(lines)+_NATIVE_SCAN+'\n}\n'


def _resolve_program(program,image):
    from ..rule_ir.portable_program import FrozenRuleProgram
    from .vm_image import lower_program
    if not isinstance(program,FrozenRuleProgram):
        raise ValueError('Enabled hydration probe requires the actual FrozenRuleProgram')
    program.validate();image.verify()
    if image.program_identity!=program.identity or lower_program(program).identity!=image.identity:
        raise ValueError('Exact program/image lowering binding required')
    if _canonical_ast()!=_CANONICAL_AST:
        raise ValueError('Canonical hydration algorithm AST changed')
    data=program.to_dict();constants={c['id']:c for c in data['constants']}
    functions=data['functions']+data['codes']
    selected={}
    for name,digest in _CANONICAL_FUNCTIONS.items():
        found=[f for f in functions if f['module']==_MODULE and f['qualname']==name]
        if len(found)!=1 or _hash(_function_structure(found[0],constants))!=digest:
            raise ValueError('Canonical hydration dependency function changed: '+name)
        selected[name]=found[0]
    classes={}
    for name,digest in _CANONICAL_CLASSES.items():
        found=[c for c in data['classes']if c['module']==_MODULE and c['qualname']==name]
        if len(found)!=1 or _hash(_class_structure(found[0],constants))!=digest:
            raise ValueError('Canonical entity class/descriptor changed: '+name)
        classes[name]=found[0]
    # A canonical-looking ID string is not sufficient: portable IR validates
    # uniqueness/reachability, but IDs can be forged independently of a target
    # descriptor's module/qualname. Resolve dependency edges to their objects.
    tables={'function':{f['id']:f for f in data['functions']},
        'class':{c['id']:c for c in data['classes']},'code':{c['id']:c for c in data['codes']}}
    prefixes={'function':'func','class':'class','code':'code'}
    def descriptor_id(kind,identifier):
        if type(identifier) is not str or identifier not in tables[kind]:
            raise ValueError('Canonical dependency descriptor is missing')
        target=tables[kind][identifier]
        if _reference_name(identifier)!=prefixes[kind]+':'+target['module']+':'+target['qualname']:
            raise ValueError('Canonical dependency reference/descriptor identity mismatch')
        if target['module']==_MODULE:
            verified=selected if kind=='function'else classes if kind=='class'else{}
            chosen=verified.get(target['qualname'])
            if chosen is None or identifier!=chosen['id']:
                raise ValueError('Dependency must use the exact verified selected descriptor ID')
    def edges(value,seen=()):
        if isinstance(value,list):
            for item in value:edges(item,seen)
        elif isinstance(value,dict):
            kind=value.get('kind')
            if type(kind) is str and kind in tables:
                descriptor_id(kind,value.get('id'));return
            if kind=='constant':
                key=value.get('id')
                if key in seen or key not in constants:raise ValueError('Finite dependency constants required')
                edges(constants[key],seen+(key,));return
            for item in value.values():edges(item,seen)
    for kind,descriptors in(('function',selected.values()),('class',classes.values())):
        for descriptor in descriptors:
            descriptor_id(kind,descriptor['id']);edges(descriptor)
    fn=selected['hydrate_entities']
    for name in('PlayerEntity','CharacterEntity','CardEntity'):
        if fn['globals'].get(name)!={'kind':'class','id':classes[name]['id']}:
            raise ValueError('Canonical hydration class global changed')
    if fn['globals'].get('allocate_entity_id')!={'kind':'function','id':selected['allocate_entity_id']['id']}:
        raise ValueError('Canonical hydration allocator global changed')
    if fn['flags']&32 or fn['argcount']!=1 or fn['varnames'][0]!='state' or fn['closure'] or fn['defaults'] or fn['kwdefaults']:
        raise ValueError('Canonical synchronous state argument required')
    matches=[i for i,fid in enumerate(image.function_ids)if fid==fn['id']]
    if len(matches)!=1:raise ValueError('Unique canonical hydration function in image required')
    fid=matches[0];row=image.functions[fid]
    if int(row[7])&32 or int(row[2])!=1 or int(row[5])!=fn['nlocals']:
        raise ValueError('Canonical hydration native frame metadata required')
    return fid,fn['varnames'].index('state'),fn


def validate_hydration_probe_inputs(program,image,schema):
    """Fail closed before CUDA allocation/compilation; executes no rule calls."""
    resolved=_resolve_program(program,image)
    _validate_schema(schema)
    return resolved


def _hook(fid,local):
    state='f[LBASE+'+str(local)+']'
    return ('\n // Read-only proof work is additional to original VM opcodes.\n'
        ' if(pc==fn[0] && f[0]=='+str(fid)+' && f[2]==0 && f[4]==1 && f[5]==0 && f[10]<0 && !c->error && !c->h.error && c->raised==MISSING){\n'
        '  int hydration_probe_work=0;\n'
        '  bool hydration_probe_hit='+_NAMESPACE+'::probe(c,'+state+',&hydration_probe_work);\n'
        '  c->steps+=hydration_probe_work;\n'
        '  if(hydration_probe_hit){return_value(c,'+state+',0);return;}\n }\n')


@dataclass(frozen=True)
class HydrationProbeBinding:
    program_identity: str
    image_identity: str
    schema_identity: str
    function_identity: str
    function_index: int
    state_local: int
    owner_unit: str
    canonical_ast: str
    function_structure: str
    recipe_sha256: str
    original_units: tuple
    transformed_units: tuple
    identity: str
    budget_semantics: str = 'additional bounded native work on HIT and MISS; not original opcode/allocation equivalence'
    gpu_qualified: bool = False

    def to_dict(self):
        return asdict(self)


def specialize_hydration_probe(units,image,schema,*,program=None,enabled=False):
    """Return units plus binding; disabled preserves the exact original object.

    Enabled is a source transform only. Callers must compile under the actual
    changed ordered unit sources and retain this binding alongside emitted SHA.
    """
    if type(enabled) is not bool:raise ValueError('Explicit hydration probe bool required')
    if not enabled:return units,None
    if type(units) not in(list,tuple) or not units or any(not isinstance(u,NativeCompileUnit)for u in units):
        raise ValueError('Complete separate native compilation units required')
    if len({u.name for u in units})!=len(units) or any(_NAMESPACE in u.source for u in units):
        raise ValueError('Unique untouched native owner units required')
    fid,local,fn=validate_hydration_probe_inputs(program,image,schema)
    helper=_helper_source(schema)
    entry='__device__ __noinline__ void native_function_'+str(fid)+'(Context* c,int* f,const int* fn,int pc){'
    owners=[i for i,u in enumerate(units)if entry in u.source]
    if len(owners)!=1:raise ValueError('Unique complete canonical native function owner required')
    index=owners[0];unit=units[index]
    if not re.fullmatch(r'rules_\d+',unit.name)or unit.source.count(entry)!=1:
        raise ValueError('Canonical native rule owner required')
    at=unit.source.index(entry);end=_matching_brace_end(unit.source,at+len(entry)-1)
    body=unit.source[at:end]
    start=int(image.functions[fid,0])
    expected_pcs=set(range(start,start+int(image.functions[fid,1])))
    labels=[int(x)for x in re.findall(r'(?m)^op_(\d+):\{',body)]
    admissions=[int(x)for x in re.findall(r'(?m)^admit_(\d+):if\(!native_function_admit',body)]
    if len(labels)!=len(expected_pcs)or set(labels)!=expected_pcs or len(admissions)!=len(expected_pcs)or set(admissions)!=expected_pcs:
        raise ValueError('Complete native body/admission coverage required')
    if not body.startswith(entry+'\n const int owner=c->current;\n switch(pc){')or 'case '+str(start)+':goto op_'+str(start)+';'not in body:
        raise ValueError('Complete canonical native function entry changed')
    first=re.search(r'(?m)^__device__ __noinline__ void native_function_\d+\([^;{}]*\)\{',unit.source)
    if first is None or unit.source.count('struct Context {')!=1 or unit.source.count('struct DomainSoAContext {')!=1 or first.start()>at:
        raise ValueError('Complete unchanged native ABI header required')
    if unit.source.find('struct Context {')>first.start()or unit.source.find('struct DomainSoAContext {')>first.start():
        raise ValueError('Native ABI declarations must precede private helper')
    from .cuda_vm import _VM_SOURCE
    from .vm_soa_views import _emit_runtime_head
    for prototype,name in((_VM_SOURCE,'Context'),(_emit_runtime_head(),'DomainSoAContext')):
        marker='struct '+name+' {'
        begin=prototype.index(marker)
        declaration=prototype[begin:_matching_brace_end(prototype,prototype.index('{',begin))]
        declarations=(declaration,)
        if name=='DomainSoAContext':
            # vm_soa_methods adds this diagnostic member in the full SoA
            # pipeline. Both recognized declarations use named field access.
            declarations+=(declaration.replace(' int constant_node_count;',
                ' int constant_node_count; int failed_pool;',1),)
        if sum(unit.source.count(d)for d in declarations)!=1:
            raise ValueError('Canonical native ABI structure changed: '+name)
    hook=_hook(fid,local)
    source=unit.source[:first.start()]+helper+unit.source[first.start():]
    source=source.replace(entry,entry+hook,1)
    if source.replace(helper,'',1).replace(entry+hook,entry,1)!=unit.source:
        raise ValueError('Exact complete canonical fallback body restoration failed')
    result=list(units);result[index]=replace(unit,source=source)
    before=tuple((u.name,u.sha256)for u in units);after=tuple((u.name,u.sha256)for u in result)
    recipe_sha=sha256(Path(__file__).read_bytes()).hexdigest()
    binding=dict(program_identity=program.identity,image_identity=image.identity,schema_identity=schema.identity,
        function_identity=fn['id'],function_index=fid,state_local=local,owner_unit=unit.name,
        canonical_ast=_CANONICAL_AST,function_structure=_CANONICAL_FUNCTIONS['hydrate_entities'],
        recipe_sha256=recipe_sha,original_units=before,transformed_units=after)
    return tuple(result),HydrationProbeBinding(**binding,identity=_hash(binding))
