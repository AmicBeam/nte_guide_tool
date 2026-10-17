"""Optional primitive comparison routing in the shared runtime owner only.

Boxed references, allocation, vm_equal, numeric_int and logical budgets retain
their canonical implementations. No live field/effect result is cached.
"""
from dataclasses import replace
from hashlib import sha256
import re

from .native_array_codegen import _matching_brace_end


_SHAPES = {
    'compare': 'dfa1941e8241968252be0a24f4e0f71f859d4059363f6c217b765e1c2f964e3b',
    'stdlib_compare': 'f9899badad4e14313f066178f0b5b991d867363b9a862b9aeae045f67643caa0',
    'decimal_compare': '22a5a390bf0aa5eff2b2b42be847e1c0dba55cbae54a1e0894708547b1dd5a5b',
}


def _definition(source, name, qualifier):
    pattern = re.compile(r'(?m)^__device__ '+qualifier+r' int '+name+r'\([^;{}]*\)\s*\{')
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError('Unique shared comparison definition required: '+name)
    match = matches[0]
    end = _matching_brace_end(source, match.end()-1)
    return match.start(), end, source[match.start():end]


def specialize_runtime_compare(unit):
    """Skip rejecting library handlers only for validated primitive refs.

    Wider integer payloads, floats, Decimal, custom objects, malformed refs and
    unsupported modes retain the complete original route. The shape guards
    bind the assumption that stdlib/Decimal reject these primitive pairs.
    """
    if unit.name != 'runtime':
        raise ValueError('Primitive comparison routing requires runtime unit')
    source = unit.source
    if 'native_runtime_primitive_compare' in source:
        raise ValueError('Primitive comparisons already specialized')
    for name, expected in _SHAPES.items():
        qualifier = '__noinline__' if name == 'compare' else 'inline'
        _, _, body = _definition(source, name, qualifier)
        canonical = body.replace('__device__ __noinline__', '__device__ inline', 1)
        if sha256(canonical.encode()).hexdigest() != expected:
            raise ValueError('Canonical comparison behavior changed: '+name)
    begin, end, body = _definition(source, 'compare', '__noinline__')
    integer = ('long long aa=0,bb=0;numeric_int(c,a,&aa);numeric_int(c,b,&bb);'
               'cmp=aa<bb?-1:aa>bb?1:0;')
    equality = 'if(mode==2 || mode==3){int eq=vm_equal(c,a,b);return mode==2?eq:!eq;}'
    if body.count(integer) != 1 or body.count(equality) != 1:
        raise ValueError('Canonical primitive comparison branches changed')
    helper = '''
// Nonmutating validation: invalid refs enter the original error-owning route.
__device__ inline int native_runtime_primitive_compare(Context* c,int ref,NodeView* n){
 const int* raw=0;int constant=ref<0;
 if(constant){long long index=-1LL-ref;if(index<0||index>=c->h.constant_node_count)return 0;
  raw=c->h.constant_nodes+index*4;
 }else{if(ref>=c->h.counts[0]||ref>=c->h.max_nodes)return 0;raw=c->h.nodes+ref*4;}
 // Do not call get_node on unknown views: its SoA adapter refreshes headers.
 if(raw[0]!=TAG_BOOL&&raw[0]!=TAG_INT&&raw[0]!=TAG_STR)return 0;
 n->tag=raw[0];n->offset=raw[1];n->count=raw[2];n->extra=raw[3];n->is_constant=constant;
 if(n->tag==TAG_BOOL)return n->offset==0&&n->count==0&&(n->extra==0||n->extra==1);
 if(n->tag!=TAG_INT&&n->tag!=TAG_STR)return 0;
 if(n->tag==TAG_INT&&(n->count<1||n->count>8))return 0;
 if(n->count<0||n->offset<0)return 0;
 long long end=(long long)n->offset+n->count;
 return end<=(n->is_constant?c->h.constant_payload_count:c->h.counts[2]);
}
'''
    hook = '''
 if(!c->error&&!c->h.error&&mode>=0&&mode<=5){
  NodeView left,right;
  if(native_runtime_primitive_compare(c,a,&left)&&native_runtime_primitive_compare(c,b,&right)){
   int numeric=(left.tag==TAG_INT||left.tag==TAG_BOOL)&&(right.tag==TAG_INT||right.tag==TAG_BOOL);
   if(numeric||((mode==2||mode==3)&&left.tag==TAG_STR&&right.tag==TAG_STR)){
    '''+equality+'''
    int cmp=0;'''+integer+'''
    if(mode==0)return cmp<0;if(mode==1)return cmp<=0;if(mode==4)return cmp>0;return cmp>=0;
   }
  }
 }
'''
    opening = body.index('{')+1
    body = body[:opening]+hook+body[opening:]
    return replace(unit, source=source[:begin]+helper+body+source[end:])
