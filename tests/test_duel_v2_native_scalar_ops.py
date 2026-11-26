"""Independently compiled native scalar helpers versus canonical primitives."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from app.modules.card_game.rl.batched_duel.native_scalar_ops import (
    _INT_TAIL_BEGIN,
    _INT_TAIL_END,
    emit_native_scalar_source,
)
from app.modules.card_game.rl.batched_duel.vm_heap import emit_vm_heap_source
from app.modules.card_game.rl.batched_duel.vm_numeric import emit_numeric_source

MISSING = -2147483647 - 1
INT64_MIN = -9223372036854775807 - 1
INT64_MAX = 9223372036854775807
TAG_BOOL = 1
TAG_INT = 2
TAG_FLOAT = 3
TAG_STR = 4
TAG_LIST = 5
ERROR_INT_OVERFLOW = 7
ERROR_NODE_CAPACITY = 1

OP_BINARY_ADD = 1
OP_BINARY_SUBTRACT = 2
OP_BINARY_MULTIPLY = 3
OP_BINARY_TRUE_DIVIDE = 4
OP_BINARY_FLOOR_DIVIDE = 5
OP_BINARY_AND = 6
OP_BINARY_OR = 7
OP_BINARY_XOR = 8
OP_BINARY_MODULO = 9
OP_INPLACE_ADD = 10
OP_INPLACE_TRUE_DIVIDE = 18


def _context_stub():
    return (
        '#include <cstdint>\n#include <cstring>\n#define __device__\n#define __noinline__ __attribute__((noinline))\n'
        + emit_vm_heap_source()
        + r'''
namespace duel_rule_vm {
using namespace duel_vm_heap;
enum Op {
 OP_BINARY_ADD=1,OP_BINARY_SUBTRACT=2,OP_BINARY_MULTIPLY=3,OP_BINARY_TRUE_DIVIDE=4,
 OP_BINARY_FLOOR_DIVIDE=5,OP_BINARY_AND=6,OP_BINARY_OR=7,OP_BINARY_XOR=8,OP_BINARY_MODULO=9,
 OP_INPLACE_ADD=10,OP_INPLACE_SUBTRACT=11,OP_INPLACE_MULTIPLY=12,OP_INPLACE_FLOOR_DIVIDE=13,
 OP_INPLACE_AND=14,OP_INPLACE_OR=15,OP_INPLACE_XOR=16,OP_INPLACE_MODULO=17,OP_INPLACE_TRUE_DIVIDE=18
};
const int MISSING=(-2147483647-1), NIL=-1;
const int T_STDLIB_COUNTER=73;
enum VmError { BAD_PROGRAM=100,STACK_OVERFLOW=101,FRAME_OVERFLOW=102,UNBOUND_LOCAL=103, BAD_CALL=104,UNSUPPORTED_OP=105,UNSUPPORTED_TYPE=106 };
struct Context { Heap h; int error; };
__device__ inline int* mutable_node(Context* c,int ref){ if(ref<0||ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;} return c->h.nodes+ref*4; }
__device__ inline const int* node(Context* c,int ref){
 if(ref>=0){if(ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return 0;}return c->h.nodes+ref*4;}
 long long id=-1LL-ref;if(id<0||id>=c->h.constant_node_count){c->error=BAD_PROGRAM;return 0;} return c->h.constant_nodes+id*4;
}
__device__ inline int kind(Context* c,int ref){const int* n=node(c,ref);return n?n[0]:-1;}
__device__ inline int edge(Context* c,int container,int index,int value=0){
 const int* n=node(c,container);if(!n||index<0||index>=n[2]){c->error=BAD_PROGRAM;return MISSING;}
 long long slot=(long long)n[1]+index;
 if(container>=0){if(slot<0||slot>=c->h.counts[1]){c->error=BAD_PROGRAM;return MISSING;}return c->h.edges[slot*2+value];}
 if(slot<0||slot>=c->h.constant_edge_count){c->error=BAD_PROGRAM;return MISSING;}return c->h.constant_edges[slot*2+value];
}
__device__ inline int length(Context* c,int ref){
 const int* n=node(c,ref);if(!n)return 0;
 if(n[0]==TAG_NONE)return 0;
 if(n[0]==TAG_LIST||n[0]==TAG_TUPLE||n[0]==TAG_DICT||(n[0]>=8&&n[0]<=10)||n[0]==21||n[0]==22||n[0]==T_STDLIB_COUNTER)return n[2];
 c->error=UNSUPPORTED_TYPE;return 0;
}
__device__ inline int meta(Context* c,int tag,int offset,int count,int extra){
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 int id=c->h.counts[0]++;int* n=c->h.nodes+id*4;n[0]=tag;n[1]=offset;n[2]=count;n[3]=extra;return id;
}
__device__ inline int numpy_numeric(Context* c,int a,int b,int op,int* out){(void)a;(void)b;(void)op;*out=MISSING;return 0;}
__device__ inline int numpy_truth(Context* c,int ref,int* out){(void)ref;*out=0;return 0;}
__device__ inline int decimal_equal(Context* c,int a,int b,int* out){(void)a;(void)b;*out=0;return 0;}
__device__ inline int decimal_compare(Context* c,int a,int b,int mode,int* out){(void)a;(void)b;(void)mode;*out=0;return 0;}
'''
    )


def _original_helpers():
    return emit_numeric_source() + r'''
__device__ inline int stdlib_binary(Context* c,int a,int b,int op,int* out){
 int ta=kind(c,a),tb=kind(c,b);
 if(ta!=T_STDLIB_COUNTER&&tb!=T_STDLIB_COUNTER)return 0;
 if(ta!=T_STDLIB_COUNTER||tb!=T_STDLIB_COUNTER){c->error=UNSUPPORTED_TYPE;return 1;}
 *out=make_int64(&c->h,73000+(op&255));return 1;
}
__device__ inline int stdlib_compare(Context* c,int a,int b,int mode,int* out){
 int ta=kind(c,a),tb=kind(c,b);
 if(ta==T_STDLIB_COUNTER&&tb==T_STDLIB_COUNTER&&(mode==2||mode==3)){
  *out=a==b;if(mode==3)*out=!*out;return 1;
 }
 return 0;
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
__device__ inline int truth_value(Context* c,int value){
 int np_truth=0;if(numpy_truth(c,value,&np_truth))return np_truth;
 int tag=kind(c,value);if(tag==21||tag==22||tag==T_STDLIB_COUNTER)return length(c,value)!=0;if(tag>=11)return 1;
 return duel_vm_heap::truth(&c->h,value);
}
__device__ inline int canonical_negative(Context* c,int ref){
 if(kind(c,ref)==TAG_FLOAT){double a=0;read_float64(&c->h,ref,&a);return make_float64(&c->h,-a);}
 else{long long a=0;numeric_int(c,ref,&a);if(a==(-9223372036854775807LL-1))c->h.error=ERROR_INT_OVERFLOW;else return make_int64(&c->h,-a);}
 return MISSING;
}
__device__ inline int canonical_binary(Context* c,int a,int b,int op){
 int stdlib_out=MISSING;if(stdlib_binary(c,a,b,op,&stdlib_out))return stdlib_out;
 return numeric_binary(c,a,b,op);
}
'''


def _wrapper():
    return r'''
#define NATIVE_BIN(OP) case OP: return native_scalar_binary<OP>(&G,a,b);
#define NATIVE_CMP(MODE) case MODE: return native_scalar_compare<MODE>(&G,a,b);
static Context G;
static int NODES[4096], EDGES[8192], COUNTS[4];
static unsigned char PAYLOAD[65536];
static int SNAP_NODES[4096], SNAP_EDGES[8192], SNAP_COUNTS[4];
static unsigned char SNAP_PAYLOAD[65536];
static int SNAP_ERR, SNAP_HERR;
extern "C" int h_err(){return G.error;}
extern "C" int h_herr(){return G.h.error;}
extern "C" int h_ncount(){return G.h.counts[0];}
extern "C" int h_ecount(){return G.h.counts[1];}
extern "C" int h_pcount(){return G.h.counts[2];}
extern "C" int h_reset(int mn,int me,int mp){
 if(mn<8||me<8||mp<32||mn>4096||me>8192||mp>65536) return -1;
 std::memset(NODES,0,sizeof(NODES));std::memset(EDGES,0,sizeof(EDGES));std::memset(PAYLOAD,0,sizeof(PAYLOAD));std::memset(COUNTS,0,sizeof(COUNTS));
 init_heap(&G.h,NODES,EDGES,PAYLOAD,COUNTS,mn,me,mp,0,0,0,0,0,0);G.error=0;return 0;
}
extern "C" int h_set_max(int mn,int me,int mp){G.h.max_nodes=mn;G.h.max_edges=me;G.h.max_payload=mp;return 0;}
extern "C" void h_snap(){
 std::memcpy(SNAP_NODES,NODES,sizeof(NODES));std::memcpy(SNAP_EDGES,EDGES,sizeof(EDGES));
 std::memcpy(SNAP_PAYLOAD,PAYLOAD,sizeof(PAYLOAD));std::memcpy(SNAP_COUNTS,COUNTS,sizeof(COUNTS));
 SNAP_ERR=G.error;SNAP_HERR=G.h.error;
}
extern "C" void h_restore(){
 std::memcpy(NODES,SNAP_NODES,sizeof(NODES));std::memcpy(EDGES,SNAP_EDGES,sizeof(EDGES));
 std::memcpy(PAYLOAD,SNAP_PAYLOAD,sizeof(PAYLOAD));std::memcpy(COUNTS,SNAP_COUNTS,sizeof(COUNTS));
 G.error=SNAP_ERR;G.h.error=SNAP_HERR;
}
extern "C" int h_none(){return make_none(&G.h);}
extern "C" int h_bool(int v){return make_bool(&G.h,v);}
extern "C" int h_int(long long v){return make_int64(&G.h,v);}
extern "C" int h_float(double v){return make_float64(&G.h,v);}
extern "C" int h_str(const char* s,int n){return make_utf8(&G.h,s,n);}
extern "C" int h_list(){return make_list(&G.h);}
extern "C" int h_append(int l,int v){return list_append(&G.h,l,v);}
extern "C" int h_counter(){return meta(&G,T_STDLIB_COUNTER,0,0,0);}
extern "C" int h_kind(int ref){return kind(&G,ref);}
extern "C" int h_len(int ref){return length(&G,ref);}
extern "C" int h_edge(int ref,int i){return edge(&G,ref,i);}
extern "C" int h_read_i(int ref,long long* o){return read_int64(&G.h,ref,o);}
extern "C" int h_read_f(int ref,double* o){return read_float64(&G.h,ref,o);}
extern "C" int h_read_s(int ref,char* buf,int cap){
 NodeView v;if(!get_node(&G.h,ref,&v)||v.tag!=TAG_STR||v.count>=cap)return -1;
 const unsigned char* p=get_payload_ptr(&G.h,&v);if(v.count&&!p)return -1;for(int i=0;i<v.count;i++)buf[i]=(char)p[i];buf[v.count]=0;return v.count;
}
extern "C" int h_native_bin(int op,int a,int b){
 switch(op){
  NATIVE_BIN(OP_BINARY_ADD) NATIVE_BIN(OP_BINARY_SUBTRACT) NATIVE_BIN(OP_BINARY_MULTIPLY)
  NATIVE_BIN(OP_BINARY_TRUE_DIVIDE) NATIVE_BIN(OP_BINARY_FLOOR_DIVIDE)
  NATIVE_BIN(OP_BINARY_AND) NATIVE_BIN(OP_BINARY_OR) NATIVE_BIN(OP_BINARY_XOR) NATIVE_BIN(OP_BINARY_MODULO)
  NATIVE_BIN(OP_INPLACE_ADD) NATIVE_BIN(OP_INPLACE_SUBTRACT) NATIVE_BIN(OP_INPLACE_MULTIPLY)
  NATIVE_BIN(OP_INPLACE_FLOOR_DIVIDE) NATIVE_BIN(OP_INPLACE_AND) NATIVE_BIN(OP_INPLACE_OR)
  NATIVE_BIN(OP_INPLACE_XOR) NATIVE_BIN(OP_INPLACE_MODULO) NATIVE_BIN(OP_INPLACE_TRUE_DIVIDE)
  default: G.error=UNSUPPORTED_OP; return MISSING;
 }
}
extern "C" int h_orig_bin(int op,int a,int b){return canonical_binary(&G,a,b,op);}
extern "C" int h_native_cmp(int mode,int a,int b){
 switch(mode){ NATIVE_CMP(0) NATIVE_CMP(1) NATIVE_CMP(2) NATIVE_CMP(3) NATIVE_CMP(4) NATIVE_CMP(5)
  default: return compare(&G,a,b,mode); }
}
extern "C" int h_orig_cmp(int mode,int a,int b){return compare(&G,a,b,mode);}
extern "C" int h_native_neg(int ref){return native_scalar_negative(&G,ref);}
extern "C" int h_orig_neg(int ref){return canonical_negative(&G,ref);}
extern "C" int h_native_not(int ref){return native_scalar_not(&G,ref);}
extern "C" int h_orig_not(int ref){return truth_value(&G,ref)?0:1;}
}
'''


def _compile_lib():
    compiler = None
    for cand in ('clang++', 'c++', 'g++'):
        path = shutil.which(cand)
        if path:
            compiler = path
            break
    if not compiler:
        raise unittest.SkipTest('No C++ compiler found; skipping host scalar tests')
    src = _context_stub() + _original_helpers() + emit_native_scalar_source() + _wrapper()
    tmp = tempfile.TemporaryDirectory()
    cpp = os.path.join(tmp.name, 'native_scalar_ops_host.cpp')
    lib_ext = '.dylib' if sys.platform == 'darwin' else '.so'
    so = os.path.join(tmp.name, 'libnative_scalar_ops_host' + lib_ext)
    with open(cpp, 'w', encoding='ascii') as fh:
        fh.write(src)
    result = subprocess.run(
        [compiler, '-std=c++17', '-shared', '-fPIC', '-O0', cpp, '-o', so],
        capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        tmp.cleanup()
        raise AssertionError('C++ compile failed:\n' + result.stdout + result.stderr)
    lib = ctypes.CDLL(so)
    lib.h_reset.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.h_set_max.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.h_int.argtypes = [ctypes.c_longlong]
    lib.h_float.argtypes = [ctypes.c_double]
    lib.h_str.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.h_read_i.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    lib.h_read_f.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.h_read_s.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    for name in (
        'h_err', 'h_herr', 'h_ncount', 'h_ecount', 'h_pcount', 'h_reset', 'h_set_max',
        'h_none', 'h_bool', 'h_int', 'h_float', 'h_str', 'h_list', 'h_append', 'h_counter',
        'h_kind', 'h_len', 'h_edge', 'h_read_i', 'h_read_s',
        'h_native_bin', 'h_orig_bin', 'h_native_cmp', 'h_orig_cmp',
        'h_native_neg', 'h_orig_neg', 'h_native_not', 'h_orig_not',
    ):
        getattr(lib, name).restype = ctypes.c_int
    lib.h_snap.restype = None
    lib.h_restore.restype = None
    lib._tmp = tmp
    return lib


class Host:
    def __init__(self, lib):
        self.lib = lib
        self.reset()

    def reset(self, mn=512, me=2048, mp=16384):
        if self.lib.h_reset(mn, me, mp) != 0:
            raise AssertionError('reset')

    def state(self):
        return (self.lib.h_err(), self.lib.h_herr(), self.lib.h_ncount(),
                self.lib.h_ecount(), self.lib.h_pcount())

    def snap(self):
        self.lib.h_snap()

    def restore(self):
        self.lib.h_restore()

    def i(self, v):
        return self.lib.h_int(v)

    def b(self, v):
        return self.lib.h_bool(1 if v else 0)

    def f(self, v):
        return self.lib.h_float(v)

    def s(self, text):
        raw = text.encode('utf-8')
        return self.lib.h_str(raw, len(raw))

    def lst(self, values):
        ref = self.lib.h_list()
        for item in values:
            self.lib.h_append(ref, item)
        return ref

    def read_i(self, ref):
        value = ctypes.c_longlong()
        rc = self.lib.h_read_i(ref, ctypes.byref(value))
        return rc, value.value

    def read_f(self, ref):
        value = ctypes.c_double()
        rc = self.lib.h_read_f(ref, ctypes.byref(value))
        return rc, value.value

    def read_s(self, ref):
        buf = ctypes.create_string_buffer(256)
        n = self.lib.h_read_s(ref, buf, 256)
        if n < 0:
            return None
        return buf.value.decode('utf-8')

    def pair(self, native, orig):
        self.snap()
        left = native()
        left_state = self.state()
        self.restore()
        right = orig()
        right_state = self.state()
        return left, left_state, right, right_state


class NativeScalarOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = _compile_lib()
        cls.host = Host(cls.lib)

    @classmethod
    def tearDownClass(cls):
        cls.lib._tmp.cleanup()

    def setUp(self):
        self.host.reset()

    def test_source_guards_reuse_tail_and_fallbacks(self):
        source = emit_native_scalar_source()
        numeric = emit_numeric_source()
        begin = numeric.index(_INT_TAIL_BEGIN)
        end = numeric.index(_INT_TAIL_END, begin) + len(_INT_TAIL_END)
        tail = numeric[begin:end]
        self.assertIn(tail, source)
        self.assertIn('template<int OP>', source)
        self.assertIn('__device__ __noinline__ int native_scalar_binary(Context* c,int a,int b)', source)
        self.assertIn('template<int MODE>', source)
        self.assertIn('__device__ __noinline__ int native_scalar_compare(Context* c,int a,int b)', source)
        self.assertIn('__device__ inline int native_scalar_negative(Context* c,int ref)', source)
        self.assertIn('__device__ inline int native_scalar_not(Context* c,int ref)', source)
        binary = source[source.index('native_scalar_binary'):source.index('native_scalar_compare')]
        self.assertLess(binary.index('stdlib_binary'), binary.index('numeric_binary'))
        compare = source[source.index('native_scalar_compare'):source.index('native_scalar_negative')]
        self.assertIn('return compare(c,a,b,MODE);', compare)
        self.assertNotIn('TRUE_REF', compare)
        self.assertNotIn('FALSE_REF', compare)
        unary = source[source.index('native_scalar_negative'):source.index('native_scalar_not')]
        self.assertIn('if(kind(c,ref)==TAG_FLOAT){double a=0;read_float64(&c->h,ref,&a);return make_float64(&c->h,-a);}', unary)
        self.assertIn('numeric_int(c,ref,&a);if(a==(-9223372036854775807LL-1))c->h.error=ERROR_INT_OVERFLOW;else return make_int64(&c->h,-a);', unary)
        not_src = source[source.index('native_scalar_not'):]
        self.assertIn('return truth_value(c,ref)?0:1;', not_src)
        self.assertNotIn('TRUE_REF', not_src)
        self.assertIn('make_int64(&c->h,out)', binary)
        self.assertNotIn('TRUE_REF', binary)

    def _bin(self, op, a, b):
        h = self.host
        return h.pair(lambda: h.lib.h_native_bin(op, a, b), lambda: h.lib.h_orig_bin(op, a, b))

    def _same_box(self, op, a, b):
        left, ls, right, rs = self._bin(op, a, b)
        self.assertEqual(ls, rs)
        self.assertEqual(left, right)
        return left, ls

    def test_integer_edges_overflow_and_bool(self):
        h = self.host
        a, b = h.i(3), h.i(4)
        ref, state = self._same_box(OP_BINARY_ADD, a, b)
        self.assertEqual(h.lib.h_kind(ref), TAG_INT)
        self.assertEqual(h.read_i(ref), (0, 7))
        self.assertGreater(state[2], 2)
        self._same_box(OP_BINARY_SUBTRACT, h.i(3), h.i(9))
        self._same_box(OP_BINARY_MULTIPLY, h.i(-6), h.i(7))
        self._same_box(OP_BINARY_AND, h.i(12), h.i(10))
        self._same_box(OP_BINARY_OR, h.i(12), h.i(10))
        self._same_box(OP_BINARY_XOR, h.i(12), h.i(10))
        self._same_box(OP_BINARY_MODULO, h.i(-7), h.i(3))
        self._same_box(OP_BINARY_FLOOR_DIVIDE, h.i(-7), h.i(3))
        for op, av, bv in (
            (OP_BINARY_ADD, INT64_MAX, 1),
            (OP_BINARY_MULTIPLY, INT64_MIN, -1),
            (OP_BINARY_FLOOR_DIVIDE, INT64_MIN, -1),
        ):
            h.reset()
            left, right = h.i(av), h.i(bv)
            ref, state = self._same_box(op, left, right)
            self.assertEqual(ref, MISSING)
            self.assertEqual(state[1], ERROR_INT_OVERFLOW)
        h.reset()
        t = h.b(True)
        ref, _ = self._same_box(OP_BINARY_ADD, t, t)
        self.assertEqual(h.lib.h_kind(ref), TAG_INT)
        self.assertEqual(h.read_i(ref), (0, 2))
        ref, _ = self._same_box(OP_BINARY_AND, t, t)
        self.assertEqual(h.lib.h_kind(ref), TAG_INT)
        self.assertEqual(h.read_i(ref), (0, 1))
        self._same_box(OP_BINARY_ADD, t, h.i(4))
        self._same_box(OP_INPLACE_ADD, h.i(8), h.i(1))
        first, _ = self._same_box(OP_BINARY_ADD, h.i(1), h.i(1))
        second, _ = self._same_box(OP_BINARY_ADD, h.i(1), h.i(1))
        self.assertNotEqual(first, second)

    def test_mixed_float_sequence_string_counter_fallback(self):
        h = self.host
        ref, _ = self._same_box(OP_BINARY_TRUE_DIVIDE, h.i(7), h.i(2))
        self.assertEqual(h.lib.h_kind(ref), TAG_FLOAT)
        self.assertEqual(h.read_f(ref)[1], 3.5)
        ref, _ = self._same_box(OP_BINARY_ADD, h.i(1), h.f(0.5))
        self.assertEqual(h.lib.h_kind(ref), TAG_FLOAT)
        self.assertAlmostEqual(h.read_f(ref)[1], 1.5)
        ref, _ = self._same_box(OP_BINARY_ADD, h.s('ab'), h.s('cd'))
        self.assertEqual(h.lib.h_kind(ref), TAG_STR)
        self.assertEqual(h.read_s(ref), 'abcd')
        left = h.lst([h.i(1), h.i(2)])
        right = h.lst([h.i(3)])
        ref, _ = self._same_box(OP_BINARY_ADD, left, right)
        self.assertEqual(h.lib.h_kind(ref), TAG_LIST)
        self.assertEqual(h.lib.h_len(ref), 3)
        seq = h.lst([h.i(9)])
        ref, _ = self._same_box(OP_BINARY_MULTIPLY, seq, h.i(3))
        self.assertEqual(h.lib.h_len(ref), 3)
        ca, cb = h.lib.h_counter(), h.lib.h_counter()
        ref, _ = self._same_box(OP_BINARY_AND, ca, cb)
        self.assertEqual(h.read_i(ref), (0, 73000 + OP_BINARY_AND))

    def test_compare_not_negative_and_capacity(self):
        h = self.host
        a, b = h.i(3), h.i(5)
        for mode, expected in ((0, 1), (1, 1), (2, 0), (3, 1), (4, 0), (5, 0)):
            left, ls, right, rs = h.pair(
                lambda mode=mode: h.lib.h_native_cmp(mode, a, b),
                lambda mode=mode: h.lib.h_orig_cmp(mode, a, b))
            self.assertEqual((left, ls), (right, rs))
            self.assertEqual(left, expected)
            self.assertEqual(ls[2], 2)
        left, ls, right, rs = h.pair(
            lambda: h.lib.h_native_cmp(2, h.b(True), h.i(1)),
            lambda: h.lib.h_orig_cmp(2, h.b(True), h.i(1)))
        self.assertEqual((left, ls), (right, rs))
        self.assertEqual(left, 1)
        mixed = h.pair(
            lambda: h.lib.h_native_cmp(0, h.i(2), h.f(2.5)),
            lambda: h.lib.h_orig_cmp(0, h.i(2), h.f(2.5)))
        self.assertEqual(mixed[0], mixed[2])
        self.assertEqual(mixed[1], mixed[3])
        self.assertEqual(mixed[0], 1)
        text = h.pair(
            lambda: h.lib.h_native_cmp(0, h.s('a'), h.s('b')),
            lambda: h.lib.h_orig_cmp(0, h.s('a'), h.s('b')))
        self.assertEqual(text[0], text[2])
        self.assertEqual(text[0], 1)
        neg = h.pair(lambda: h.lib.h_native_neg(h.i(5)), lambda: h.lib.h_orig_neg(h.i(5)))
        self.assertEqual(neg[0], neg[2])
        self.assertEqual(neg[1], neg[3])
        self.assertEqual(h.read_i(neg[0]), (0, -5))
        overflow = h.pair(lambda: h.lib.h_native_neg(h.i(INT64_MIN)), lambda: h.lib.h_orig_neg(h.i(INT64_MIN)))
        self.assertEqual(overflow[0], MISSING)
        self.assertEqual(overflow[0], overflow[2])
        self.assertEqual(overflow[1][1], ERROR_INT_OVERFLOW)
        self.assertEqual(overflow[1], overflow[3])
        h.reset()
        flag = h.b(True)
        bool_neg = h.pair(lambda: h.lib.h_native_neg(flag), lambda: h.lib.h_orig_neg(flag))
        self.assertEqual(h.lib.h_kind(bool_neg[0]), TAG_INT)
        self.assertEqual(h.read_i(bool_neg[0]), (0, -1))
        float_neg = h.pair(lambda: h.lib.h_native_neg(h.f(2.5)), lambda: h.lib.h_orig_neg(h.f(2.5)))
        self.assertEqual(float_neg[1], float_neg[3])
        self.assertAlmostEqual(h.read_f(float_neg[0])[1], -2.5)
        for value, expected in ((h.b(True), 0), (h.b(False), 1), (h.i(0), 1), (h.i(8), 0), (h.lst([h.i(1)]), 0)):
            left, ls, right, rs = h.pair(
                lambda value=value: h.lib.h_native_not(value),
                lambda value=value: h.lib.h_orig_not(value))
            self.assertEqual((left, ls), (right, rs))
            self.assertEqual(left, expected)
        h.reset(8, 16, 64)
        a, b = h.i(1), h.i(1)
        h.lib.h_set_max(h.lib.h_ncount(), 16, 64)
        ref, state = self._same_box(OP_BINARY_ADD, a, b)
        self.assertEqual(state[1], ERROR_NODE_CAPACITY)
        self.assertIn(ref, (MISSING, -1))

if __name__ == '__main__':
    unittest.main()
