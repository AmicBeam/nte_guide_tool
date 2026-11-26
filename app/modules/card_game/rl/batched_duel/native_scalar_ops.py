"""Guarded native scalar helpers for straight-line value regions.

Integer/bool fast paths reuse the canonical numeric integer arithmetic tail so
overflow, mixed bool/int boxing and heap budgets cannot drift. Fallbacks keep
stdlib_binary then numeric_binary, compare, exact UNARY_NEGATIVE, and
truth_value. Compare/not return raw truth; binary/negative still box.
Primary wires these into the native_values region compiler.
"""
from .vm_numeric import emit_numeric_source


_INT_TAIL_BEGIN = (
    'long long aa=0,bb=0,out=0;if(numeric_int(c,a,&aa)<0||numeric_int(c,b,&bb)<0)return MISSING;'
)
_INT_TAIL_END = 'return c->h.error?MISSING:make_int64(&c->h,out);'
_UNARY_NEGATIVE_FALLBACK = (
    'if(kind(c,ref)==TAG_FLOAT){double a=0;read_float64(&c->h,ref,&a);return make_float64(&c->h,-a);}\n'
    ' else{long long a=0;numeric_int(c,ref,&a);if(a==(-9223372036854775807LL-1))c->h.error=ERROR_INT_OVERFLOW;else return make_int64(&c->h,-a);}\n'
    ' return MISSING;'
)


def _integer_arithmetic_tail():
    source = emit_numeric_source()
    if source.count(_INT_TAIL_BEGIN) != 1 or source.count(_INT_TAIL_END) != 1:
        raise ValueError('Canonical integer arithmetic tail changed')
    begin = source.index(_INT_TAIL_BEGIN)
    end = source.index(_INT_TAIL_END, begin) + len(_INT_TAIL_END)
    tail = source[begin:end]
    closer = source[end:end + 8]
    if 'make_int64(&c->h,out)' not in tail or '}' not in closer:
        raise ValueError('Canonical integer arithmetic tail lost boxing')
    return tail


def emit_native_scalar_source():
    from .cuda_vm import _VM_SOURCE
    canonical_negative = (
        'case OP_UNARY_NEGATIVE:{int ref=pop(c,f);if(kind(c,ref)==TAG_FLOAT){double a=0;'
        'read_float64(&c->h,ref,&a);push(c,f,make_float64(&c->h,-a));}\n'
        '    else{long long a=0;numeric_int(c,ref,&a);if(a==(-9223372036854775807LL-1))'
        'c->h.error=ERROR_INT_OVERFLOW;else push(c,f,make_int64(&c->h,-a));}break;}'
    )
    if _VM_SOURCE.count(canonical_negative) != 1:
        raise ValueError('Canonical unary negation changed')
    tail = _integer_arithmetic_tail()
    return (
        'template<int OP>\n'
        '__device__ __noinline__ int native_scalar_binary(Context* c,int a,int b){\n'
        ' int ta=kind(c,a),tb=kind(c,b);\n'
        ' if((ta==TAG_BOOL||ta==TAG_INT)&&(tb==TAG_BOOL||tb==TAG_INT)'
        '&&OP!=OP_BINARY_TRUE_DIVIDE&&OP!=OP_INPLACE_TRUE_DIVIDE){\n'
        '  int op=OP;\n'
        '  ' + tail + '\n'
        ' }\n'
        ' int stdlib_out=MISSING;if(stdlib_binary(c,a,b,OP,&stdlib_out))return stdlib_out;\n'
        ' return numeric_binary(c,a,b,OP);\n'
        '}\n'
        'template<int MODE>\n'
        '__device__ __noinline__ int native_scalar_compare(Context* c,int a,int b){\n'
        ' int ta=kind(c,a),tb=kind(c,b);\n'
        ' if((ta==TAG_BOOL||ta==TAG_INT)&&(tb==TAG_BOOL||tb==TAG_INT)&&MODE>=0&&MODE<=5){\n'
        '  long long aa=0,bb=0;numeric_int(c,a,&aa);numeric_int(c,b,&bb);\n'
        '  int cmp=aa<bb?-1:aa>bb?1:0;\n'
        '  if(MODE==0)return cmp<0;if(MODE==1)return cmp<=0;if(MODE==2)return cmp==0;\n'
        '  if(MODE==3)return cmp!=0;if(MODE==4)return cmp>0;return cmp>=0;\n'
        ' }\n'
        ' return compare(c,a,b,MODE);\n'
        '}\n'
        '__device__ inline int native_scalar_negative(Context* c,int ref){\n'
        ' int tag=kind(c,ref);\n'
        ' if(tag==TAG_BOOL||tag==TAG_INT){\n'
        '  long long a=0;numeric_int(c,ref,&a);\n'
        '  if(a==(-9223372036854775807LL-1))c->h.error=ERROR_INT_OVERFLOW;else return make_int64(&c->h,-a);\n'
        '  return MISSING;\n'
        ' }\n'
        ' ' + _UNARY_NEGATIVE_FALLBACK + '\n'
        '}\n'
        '__device__ inline int native_scalar_not(Context* c,int ref){\n'
        ' int tag=kind(c,ref);\n'
        ' if(tag==TAG_BOOL||tag==TAG_INT){\n'
        '  int yes=duel_vm_heap::truth(&c->h,ref);return yes?0:1;\n'
        ' }\n'
        ' return truth_value(c,ref)?0:1;\n'
        '}\n'
    )
