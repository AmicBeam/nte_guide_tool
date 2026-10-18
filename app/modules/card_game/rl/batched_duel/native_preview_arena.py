"""Opt-in, ABI-pinned typed allocation scope for the exact scalar rule preview.

This is a derived backend, never the canonical cache. Only one split unit is
changed. The frozen program, private deep-copy implementation, frame GC and all
shared helpers are pinned together. Heap nodes and strings remain allocated.
No CUDA qualification is implied by transforming or CPU-testing this source.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType

from .native_compile_units import NativeCompileUnit, _LEXICAL, _end
from .record_soa import CompiledRecordSchema
from .vm_image import VmImage

PROGRAM_SHA256 = '6afa1c73067cef1ff694ba6cf22b71110f28cf858b9147ea5a20b472afdb475c'
PROGRAM_IDENTITY = 'a7fbbe222097b9197b5bc06f67b7381d67c4810b4097d7815b20326ec106801a'
IMAGE_IDENTITY = '790b4afc3e31b136e2838ec4dab3112260ffe96a61176c6f37d7f79aa2a8f91c'
SCHEMA_IDENTITY = '75073840444a5492c7db2ac626c26efbc31674d08c9d7b53cf21b779787eded9'
SCHEMA_LAYOUT_SHA256 = '01151a68786b8516f8780bbd43c3ad33d4781670d32b000e8c5eb1d62e16fe4a'
PREVIEW_AST_SHA256 = 'af5ae119e1753ee88b0b579b124ec170716d9f7d1663470684c5aafc9f66f5db'
PREVIEW_SOURCE_SHA256 = '27253988b49c6af0944020ec2f8c70855e4d8612d86fe642bc58121cedc3648d'
TARGET_FUNCTION = 597
TARGET_UNIT = 'rules_009'
BACKEND_ABI = 'exact_scalar_preview_typed_arena_v1'
CANONICAL_UNIT_SHA256 = MappingProxyType({'runtime': '947f5e02d107e06b3ce75354c90e582cad8296ce8843a5d067c5b13550138ce6',
 'rules_000': 'aba2fb0caf9ef4e1a786d2558c40c59ac2cb628b6e29df4cf58f0234a1b690dc',
 'rules_001': '510a2e997239236dc56858ce57f892f8c064934f7a6b5668213591bb5eececde',
 'rules_002': '59f9af3610788361dbed9b3ee55977f16229e32197dce905e8e3dd6e811ff935',
 'rules_003': 'e5ec7644ba79aa14ceae2a96088dac9b99a6deeff91e9471c043a6d1c6e813c3',
 'rules_004': 'a7fa394715451c0cc8f21dbfeb5248ac8bb61c1f10beb5067aa185fddd4af92c',
 'rules_005': '0d50db2e05455602290ef0d0d89af82d32938a9a89af391114f4aa5663deb7ca',
 'rules_006': 'b74f0cf5ddddd4016f421aef9b0b790f8f62cef25e29055db135c47c9c897fb9',
 'rules_007': '20c6f35320b1b34f2c8db10b26dbd5fbdde2e495ef1055725d77dc25bdd8f779',
 'rules_008': 'ad9defd3ca5827145e555c3babe61257196e95186aa93db5048987e0266011f0',
 'rules_009': '604fc7f177004dc7cfaa115b8eda669197ebf7dbf378497f45ea715775dba7e4',
 'rules_010': '5b9cbd32f7939af418dea1b2f2a1ed7534febdb8e4437cda7377ba9cf18b276c',
 'rules_011': 'fc430bf48dc523d55e5cbe30aa8c33e0d2bfd374e272b5b220d513cd8b5022e7',
 'rules_012': '24aa2826dbdc9fa960aac8a9ef6bf034be6deb5d0b33420b3173d48af3d15884',
 'rules_013': '28f4459097aa9a0b0cef22a16fe6d3412cf9b2fc8b60b64e16ac2bba52e09032',
 'rules_014': 'df540bf645a36010750cc882bc83e6806cb0f6e414aa7a46e4cda8c9d55ae21b',
 'rules_015': 'c2ef935d84482939ed09204cd01511acaaeb5c0a1c593abbab39fe02abeb0842',
 'rules_016': 'cdf1c4bff93f94e18d99663a5abe81462201896f706f238666287b88f973848f',
 'values_000': 'abdbe5f084fff05be644d4062238fab06f1f0b973ca4e74275b416e9eb1f1dea',
 'values_001': 'abb805365e1ab7cc239b541df86727886b3033a96cf4f47d4b5dc6a00d589a84',
 'values_002': '7d515c5447f98a30811b3d25e73bb38ec2d52801021841bf1f69cf5d0db06099',
 'values_003': '0e8e9ad15f12a2532be8970f981ebb4dd3a58a326b936367c287fe47b345ed1d'})


class PreviewArenaError(ValueError):
    pass


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':')).encode()).hexdigest()


def _stable_ast(value):
    """Retain every semantic field across Python3.10/3.13 AST layouts.

    Ordinary definitions gained an empty type_params field in Python3.12.
    Generic definitions remain unsupported and are never normalized away.
    """
    if isinstance(value,ast.AST):
        fields={}
        for name in value._fields:
            item=getattr(value,name,None)
            if name=='type_params':
                if item not in(None,[]):raise PreviewArenaError('Generic preview AST is unsupported')
                continue
            fields[name]=_stable_ast(item)
        return dict(node=type(value).__name__,fields=fields)
    if isinstance(value,(list,tuple)):return [_stable_ast(item)for item in value]
    if value is None or type(value)in(bool,int,float,str):return value
    raise PreviewArenaError('Unsupported canonical preview AST field')


def _schema_layout(schema):
    return dict(pools=[asdict(p) for p in schema.pools], lengths=dict(schema.lengths),
                **{k: getattr(schema, k) for k in (
                    'total_objects', 'root', 'max_strings', 'max_string_bytes', 'strings_count',
                    'bytes_used', 'string_offsets', 'string_lengths', 'string_static_refs',
                    'string_hashes', 'string_bytes')})


def _unit_hashes(units):
    # The frozen generator may be loaded under the same module name before or
    # after this adapter. Check actual source bytes, not Python class identity
    # or a caller-supplied ``sha256`` property.
    result = {}
    for unit in units:
        if (type(getattr(unit, 'name', None)) is not str
                or type(getattr(unit, 'source', None)) is not str or unit.name in result):
            raise PreviewArenaError('Unique named native source units required')
        result[unit.name] = sha256(unit.source.encode()).hexdigest()
    return result


def verify_frozen_preview(program, image, schema, *, projection_source=None):
    """Reject any different rules/callgraph/bytecode/globals, image or schema.

    A supplied source is only a guard, never executed. By default inspect the
    local canonical projection source without importing the rule runtime.
    """
    data = program.data if hasattr(program, 'data') else program
    if not isinstance(data, dict) or _digest(data) != PROGRAM_SHA256:
        raise PreviewArenaError('Unsupported frozen program / preview callgraph')
    if data.get('identity') != PROGRAM_IDENTITY:
        raise PreviewArenaError('Frozen program identity mismatch')
    if not isinstance(image, VmImage) or not isinstance(schema, CompiledRecordSchema):
        raise PreviewArenaError('Actual numeric image and compiled typed schema required')
    image.verify()
    if image.identity != IMAGE_IDENTITY or image.program_identity != PROGRAM_IDENTITY:
        raise PreviewArenaError('Frozen image mismatch')
    if (schema.identity != SCHEMA_IDENTITY or len(schema.pools) != 83
            or _digest(_schema_layout(schema)) != SCHEMA_LAYOUT_SHA256):
        raise PreviewArenaError('Frozen typed schema mismatch')
    if projection_source is None:
        projection_source = (Path(__file__).resolve().parents[2] /
                             'engine/duel_v2/projection.py').read_text(encoding='utf8')
    if sha256(projection_source.encode()).hexdigest() != PREVIEW_SOURCE_SHA256:
        raise PreviewArenaError('Projection source mismatch')
    matches = [n for n in ast.parse(projection_source).body
               if isinstance(n, ast.FunctionDef) and n.name == 'rule_battle_preview']
    if len(matches) != 1 or _digest(_stable_ast(matches[0])) != PREVIEW_AST_SHA256:
        raise PreviewArenaError('Scalar preview AST mismatch')
    fn = data['functions'][TARGET_FUNCTION]
    if (image.function_ids[TARGET_FUNCTION] != fn['id'] or int(image.functions[TARGET_FUNCTION, 5]) != 9
            or fn['nlocals'] != 9 or fn['instructions'][-1]['op'] != 'RETURN_VALUE'
            or sum(i['op'] == 'RETURN_VALUE' for i in fn['instructions']) != 1):
        raise PreviewArenaError('Preview frame/return layout mismatch')
    return int(image.functions[TARGET_FUNCTION, 0])


@dataclass(frozen=True)
class PreviewArenaTransform:
    units: tuple
    binding: object

    @property
    def identity(self):
        return _digest(dict(self.binding))

    def verify(self):
        actual = _unit_hashes(self.units)
        fixed = dict(backend=BACKEND_ABI, program=PROGRAM_IDENTITY, image=IMAGE_IDENTITY,
                     schema=SCHEMA_IDENTITY, reclaim='private typed slots only; heap and strings retained',
                     cuda_qualified=False)
        if (set(self.binding) != set(fixed) | {'canonical_units', 'derived_units'}
                or any(self.binding[k] != v for k, v in fixed.items())):
            raise PreviewArenaError('Derived backend binding changed')
        if actual != dict(self.binding['derived_units']):
            raise PreviewArenaError('Derived units changed')
        if (dict(self.binding['canonical_units']) != dict(CANONICAL_UNIT_SHA256)
                or set(actual) != set(CANONICAL_UNIT_SHA256)
                or any(actual[k] != v for k, v in CANONICAL_UNIT_SHA256.items() if k != TARGET_UNIT)
                or actual[TARGET_UNIT] == CANONICAL_UNIT_SHA256[TARGET_UNIT]):
            raise PreviewArenaError('Canonical cache reuse boundary changed')
        return True


def transform_units(units, image, schema, program, *, projection_source=None):
    """Explicitly derive one unit; require all 22 canonical source SHA matches.

    Caller must use ``result.identity`` as a distinct backend/cache binding and
    compile/link the returned units. This function performs no compilation.
    """
    entry = verify_frozen_preview(program, image, schema, projection_source=projection_source)
    units = tuple(units)
    if len(units) != 22 or _unit_hashes(units) != dict(CANONICAL_UNIT_SHA256):
        raise PreviewArenaError('Exact canonical 22-unit ABI required')
    owner = next(u for u in units if u.name == TARGET_UNIT)
    source = _transform_owner(owner.source, schema, entry)
    derived = tuple(NativeCompileUnit(u.name, source) if u.name == TARGET_UNIT else u for u in units)
    binding = MappingProxyType(dict(
        backend=BACKEND_ABI, program=PROGRAM_IDENTITY, image=IMAGE_IDENTITY,
        schema=SCHEMA_IDENTITY, canonical_units=dict(CANONICAL_UNIT_SHA256),
        derived_units=_unit_hashes(derived),
        reclaim='private typed slots only; heap and strings retained',
        cuda_qualified=False))
    result = PreviewArenaTransform(derived, binding)
    result.verify()
    return result


def _transform_owner(source, schema, entry):
    """Internal emitter; public callers must pass the complete pinned ABI guard."""
    signature = '__device__ __noinline__ void native_function_597(Context* c,int* f,const int* fn,int pc){'
    if source.count(signature) != 1 or 'preview_arena_' in source:
        raise PreviewArenaError('Unique unmodified complete preview body required')
    # Counts are raw integers in physically reserved, unaddressable local tail.
    layout = re.search(r'const int VM_FRAMES=\d+,VM_LOCALS=(\d+),VM_STACK=\d+,VM_CELLS=\d+;', source)
    if not layout or int(layout[1]) < 9 + len(schema.pools) + 1:
        raise PreviewArenaError('Insufficient physical frame tail')
    if ('const int HEADER=14, LBASE=HEADER, SBASE=LBASE+VM_LOCALS, CBASE=SBASE+VM_STACK, STRIDE=CBASE+VM_CELLS;'
            not in source or 'for(int i=0;i<fn[5];i++)gc_ref(c,f[LBASE+i],tail);' not in source):
        raise PreviewArenaError('Frame ABI / exact GC local roots changed')
    start = source.index(signature)
    mask = _LEXICAL.sub(lambda m: re.sub(r'[^\n]', ' ', m[0]), source)
    end = _end(mask, start + len(signature) - 1)
    body = source[start:end]
    # Complete lowering delegates RETURN_VALUE to the canonical shared opcode.
    # Keep that opcode intact: validate/reclaim while the result is still the
    # top stack root, then let the original instruction pop and return it.
    returns = re.findall(r'native_step<OP_RETURN_VALUE>\(c,f,fn,-?\d+,-?\d+,-?\d+\);', body)
    direct = 'return_value(c,pop(c,f),0);'
    if len(returns) + body.count(direct) != 1 or f'case {entry}:goto op_{entry};' not in body:
        raise PreviewArenaError('Exact preview entry / semantic return required')
    old = returns[0] if returns else direct
    body = body.replace(signature, signature + '\n if(!preview_arena_enter(c,f,pc))return;', 1)
    body = body.replace(old, 'int preview_result=peek(c,f);\n'
                        ' if(!preview_arena_leave(c,f,preview_result))return;\n'
                        ' ' + old, 1)
    return source[:start] + _emit_helpers(schema, entry) + body + source[end:]


def _emit_helpers(schema, entry):
    # Generate exact scalar-cache ranges for every physical slot, including
    # unused list capacity. Do not clear caches belonging to old live objects.
    cases = []
    for i, pool in enumerate(schema.pools):
        columns = (*pool.fields, *((pool.keys,) if pool.keys is not None else ()), *pool.values)
        rows = []
        for col in columns:
            stride = col.stride
            stores = [f'soa_store_u8(s,{col.tag}+slot*{stride}+j,SOA_ABSENT);']
            for channel, offset, zero in (('i64', col.integer, '0'), ('f64', col.real, '0.0'), ('i32', col.word, '0')):
                if offset >= 0:
                    stores.append(f'soa_store_{channel}(s,{offset}+slot*{stride}+j,{zero});')
            rows.append(f'''for(int j=0;j<{stride};j++){{
 if(!preview_arena_charge(c))return 0;
 soa_store_cache(s,{col.tag}+slot*{stride}+j,SOA_MISSING);
 {' '.join(stores)}
 }}''')
        if pool.order >= 0:
            rows.append(f'''for(int j=0;j<{pool.max_items};j++){{
 if(!preview_arena_charge(c))return 0;
 soa_store_i32(s,{pool.order}+slot*{pool.max_items}+j,0);
 }}''')
        cases.append(f'case {i}:{{' + '\n'.join(rows) + '}break;')
    return _HELPERS.replace('@ENTRY@', str(entry)).replace('@COUNT@', str(len(schema.pools))).replace('@CASES@', '\n'.join(cases))


_HELPERS = r'''
// Scope belongs to the real VM frame, not a C++ activation (calls may suspend).
__device__ inline int preview_arena_charge(Context* c){
 if(c->error||c->h.error||c->raised!=MISSING)return 0;
 if(++c->steps>c->step_limit){c->error=STEP_LIMIT;return 0;}
 if(c->deadline&&clock_ns()>=(unsigned long long)c->deadline){c->error=DEADLINE;return 0;}
 return 1;
}
__device__ inline int preview_arena_enter(Context* c,int* f,int pc){
 const int* code=function(c,597);
 if(!code||f[0]!=597||code[5]!=9||!soa_ready(&c->h)){c->error=BAD_PROGRAM;return 0;}
 DomainSoAContext* s=soa_ctx(&c->h);
 if(pc==@ENTRY@){
  for(int p=0;p<@COUNT@;p++){
   if(!preview_arena_charge(c))return 0;
   int used=soa_load_i32(s,SOA_POOL_USED[p]);
   if(used<0||used>SOA_POOL_CAP[p]){c->error=BAD_PROGRAM;return 0;}
   f[LBASE+9+p]=used;
  }
  f[LBASE+9+@COUNT@]=19088743;
 }else if(f[LBASE+9+@COUNT@]!=19088743){c->error=BAD_PROGRAM;return 0;}
 return 1;
}
__device__ inline int preview_arena_released(Context* c,int* f,int ref){
 if(!soa_header_view(&c->h,ref))return 0;
 int p=0,slot=0;
 if(!soa_locate(c->h.nodes[ref*4+3],&p,&slot)){c->error=BAD_PROGRAM;return 0;}
 return slot>=f[LBASE+9+p];
}
__device__ inline int preview_arena_leave(Context* c,int* f,int result){
 if(f[LBASE+9+@COUNT@]!=19088743||!preview_arena_charge(c))return 0;
 // The ONLY escaping value is an ordinary dictionary of whitelisted numeric
 // scores. No typed view, even an old live view, is accepted as its payload.
 if(result<0||soa_header_view(&c->h,result)){c->error=BAD_PROGRAM;return 0;}
 const int* out=node(c,result);
 if(!out||out[0]!=TAG_DICT||out[2]<0||out[2]>4){c->error=BAD_PROGRAM;return 0;}
 int seen=0;
 for(int i=0;i<out[2];i++){
  if(!preview_arena_charge(c))return 0;
  int key=edge(c,result,i,0),value=edge(c,result,i,1),bit=0;
  if(soa_header_view(&c->h,key)||soa_header_view(&c->h,value)){c->error=BAD_PROGRAM;return 0;}
  const int* v=node(c,value);
  if(!v||(v[0]!=TAG_INT&&v[0]!=TAG_FLOAT)){c->error=BAD_PROGRAM;return 0;}
  if(vm_text_equals(c,key,"attack"))bit=1;
  else if(vm_text_equals(c,key,"counter"))bit=2;
  else if(vm_text_equals(c,key,"攻击次数"))bit=4;
  else if(vm_text_equals(c,key,"创生花伤害"))bit=8;
  if(!bit||(seen&bit)||c->error||c->h.error){c->error=BAD_PROGRAM;return 0;}
  seen|=bit;
 }
 DomainSoAContext* s=soa_ctx(&c->h);
 // Validate every range before changing any allocation counter.
 for(int p=0;p<@COUNT@;p++){
  if(!preview_arena_charge(c))return 0;
  int old=f[LBASE+9+p],used=soa_load_i32(s,SOA_POOL_USED[p]);
  if(old<0||used<old||used>SOA_POOL_CAP[p]){c->error=BAD_PROGRAM;return 0;}
 }
 // Constant adoption may have interned a newly allocated typed view. Clear
 // only those entries; preserve existing object headers and Python id().
 for(int i=0;i<s->constant_node_count;i++){
  if(!preview_arena_charge(c))return 0;
  long long at=(long long)i*s->rows+s->row;
  if(s->constant_adopt&&preview_arena_released(c,f,s->constant_adopt[at]))s->constant_adopt[at]=SOA_MISSING;
  if(c->error)return 0;
 }
 for(int p=0;p<@COUNT@;p++){
  int used=soa_load_i32(s,SOA_POOL_USED[p]);
  for(int slot=f[LBASE+9+p];slot<used;slot++){
   if(!preview_arena_charge(c))return 0;
   soa_store_view(s,SOA_POOL_BASE[p]+slot,SOA_MISSING);
   soa_store_u8(s,SOA_POOL_TAG[p]+slot,SOA_ABSENT);
   soa_store_i32(s,SOA_POOL_COUNT_COL[p]+slot,0);
   switch(p){@CASES@ default:c->error=BAD_PROGRAM;return 0;}
  }
 }
 // Memo epochs, including wrap, are budgeted. No live view cache is reset.
 if(s->clone_epoch==2147483647){
  for(int i=0;i<SOA_TOTAL_OBJECTS;i++){
   if(!preview_arena_charge(c))return 0;
   s->clone_queue[(long long)i*s->rows+s->row]=SOA_MISSING;
  }
  s->clone_epoch=1;
 }else s->clone_epoch++;
 for(int p=0;p<@COUNT@;p++){
  if(!preview_arena_charge(c))return 0;
  soa_store_i32(s,SOA_POOL_USED[p],f[LBASE+9+p]);
 }
 f[LBASE+9+@COUNT@]=0;
 return !c->error&&!c->h.error;
}
'''
