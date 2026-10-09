"""Host compilation tests for the bounded device NumPy subset."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel.vm_heap import emit_vm_heap_source
from app.modules.card_game.rl.batched_duel.vm_numpy import (
    NP_DTYPE_BOOL,
    SUPPORTED_NUMPY_METHODS,
    SUPPORTED_NUMPY_NAMES,
    TAG_NDARRAY,
    TAG_NDARRAY_ITER,
    TAG_NDARRAY_METHOD,
    TAG_NP_DTYPE,
    TAG_NP_SCALAR,
    emit_numpy_source,
)

MISSING = -2147483647 - 1
UNSUPPORTED_TYPE = 106
BAD_CALL = 104
NP_OP_ADD = 1
NP_OP_MUL = 3


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _context_stub() -> str:
    return (
        "#define __device__\n#include <cstdint>\n#include <cstring>\n"
        + emit_vm_heap_source()
        + r"""
namespace duel_rule_vm {
using namespace duel_vm_heap;
const int MISSING=(-2147483647-1), NIL=-1;
const int T_NATIVE=15, T_SLICE=20;
enum VmError { BAD_PROGRAM=100,STACK_OVERFLOW=101,FRAME_OVERFLOW=102,UNBOUND_LOCAL=103, BAD_CALL=104,UNSUPPORTED_OP=105,UNSUPPORTED_TYPE=106,STEP_LIMIT=107,DEADLINE=108,STOP_ITERATION=109 };
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
 if(n[0]==TAG_LIST||n[0]==TAG_TUPLE||n[0]==TAG_DICT||(n[0]>=8&&n[0]<=10))return n[2];
 c->error=UNSUPPORTED_TYPE;return 0;
}
__device__ inline int meta(Context* c,int tag,int offset,int count,int extra){
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 int id=c->h.counts[0]++;int* n=c->h.nodes+id*4;n[0]=tag;n[1]=offset;n[2]=count;n[3]=extra;return id;
}
__device__ inline int numeric_int(Context* c,int ref,long long* value){
 if(kind(c,ref)==TAG_BOOL){*value=node(c,ref)[3]?1:0;return 0;} return read_int64(&c->h,ref,value);
}
__device__ inline int numeric_double(Context* c,int ref,double* value){
 if(kind(c,ref)==TAG_FLOAT)return read_float64(&c->h,ref,value);
 long long integer=0;if(numeric_int(c,ref,&integer)<0)return -1;*value=(double)integer;return 0;
}
__device__ inline int vm_text_equals(Context* c,int ref,const char* text){
 const int* n=node(c,ref);if(!n||n[0]!=TAG_STR)return 0;NodeView v;if(!get_node(&c->h,ref,&v))return 0;
 const unsigned char* p=get_payload_ptr(&c->h,&v);int size=0;while(text[size])size++;
 if(size!=v.count)return 0;if(size&&!p)return 0;for(int i=0;i<size;i++)if(p[i]!=(unsigned char)text[i])return 0;return 1;
}
static int GC_LAST=MISSING;
__device__ inline void gc_ref(Context* c,int ref,int* tail){ GC_LAST=ref; if(tail) (*tail)++; (void)c; }
"""
    )


def _wrapper() -> str:
    return r"""
static Context G;
static int NODES[8192];
static int EDGES[16384];
static unsigned char PAYLOAD[262144];
static int COUNTS[4];
extern "C" int h_missing(){return MISSING;}
extern "C" int h_err(){return G.error;}
extern "C" int h_herr(){return G.h.error;}
extern "C" int h_reset(int mn,int me,int mp){
 if(mn<8||me<8||mp<64||mn>8192||me>16384||mp>262144) return -1;
 std::memset(NODES,0,sizeof(NODES));std::memset(EDGES,0,sizeof(EDGES));std::memset(PAYLOAD,0,sizeof(PAYLOAD));std::memset(COUNTS,0,sizeof(COUNTS));
 init_heap(&G.h,NODES,EDGES,PAYLOAD,COUNTS,mn,me,mp,0,0,0,0,0,0);G.error=0;GC_LAST=MISSING;return 0;
}
extern "C" int h_none(){return make_none(&G.h);}
extern "C" int h_bool(int v){return make_bool(&G.h,v);}
extern "C" int h_int(long long v){return make_int64(&G.h,v);}
extern "C" int h_float(double v){return make_float64(&G.h,v);}
extern "C" int h_str(const char* s,int n){return make_utf8(&G.h,s,n);}
extern "C" int h_list(){return make_list(&G.h);}
extern "C" int h_tuple(const int* r,int n){return make_tuple(&G.h,r,n);}
extern "C" int h_dict(){return make_dict(&G.h);}
extern "C" int h_append(int l,int v){return list_append(&G.h,l,v);}
extern "C" int h_dset(int d,int k,int v){return dict_set(&G.h,d,k,v);}
extern "C" int h_slice(int a,int b,int p){int refs[3]={a,b,p};int t=make_tuple(&G.h,refs,3);if(t<0)return t;int* nd=mutable_node(&G,t);if(!nd)return MISSING;nd[0]=T_SLICE;return t;}
extern "C" int h_kind(int ref){return kind(&G,ref);}
extern "C" int h_native(int extra){return meta(&G,T_NATIVE,0,0,extra);}
extern "C" int h_call_np(int callee,const int* args,int nargs,int kn,const int* kv,int nkw){
 int out=MISSING;int hit=numpy_call(&G,callee,args,nargs,kn,kv,nkw,&out);
 if(!hit&&!G.error&&!G.h.error) G.error=UNSUPPORTED_TYPE; return out;
}
extern "C" int h_attr(int owner,const char* name){int nm=make_utf8(&G.h,name,(int)std::strlen(name));return numpy_attribute(&G,owner,nm);}
extern "C" int h_sub(int owner,int key){int out=MISSING;int hit=numpy_try_subscript(&G,owner,key,&out);if(!hit&&!G.error&&!G.h.error)G.error=UNSUPPORTED_TYPE;return out;}
extern "C" int h_store(int owner,int key,int value){int hit=numpy_try_store(&G,owner,key,value);if(!hit&&!G.error&&!G.h.error)G.error=UNSUPPORTED_TYPE;return hit;}
extern "C" int h_len(int ref){int out=0;int hit=numpy_length(&G,ref,&out);if(!hit){G.error=UNSUPPORTED_TYPE;return MISSING;}return out;}
extern "C" int h_truth(int ref){int out=0;int hit=numpy_truth(&G,ref,&out);if(!hit){G.error=UNSUPPORTED_TYPE;return MISSING;}return out;}
extern "C" int h_iter(int owner){return numpy_iterator(&G,owner);}
extern "C" int h_next(int it){return numpy_iterator_next(&G,it);}
extern "C" int h_num(int a,int b,int op){int out=MISSING;int hit=numpy_numeric(&G,a,b,op,&out);if(!hit&&!G.error&&!G.h.error)G.error=UNSUPPORTED_TYPE;return out;}
extern "C" int h_info(int ref,int* ndim,int* s0,int* s1,int* dtype,int* root){
 NpArr a;if(np_load(&G,ref,&a)<=0)return -1;*ndim=a.ndim;*s0=a.shape0;*s1=a.ndim==1?1:a.shape1;*dtype=a.dtype;*root=a.root;return 0;
}
extern "C" int h_item(int ref,int i,int j,float* o){NpArr a;if(np_load(&G,ref,&a)<=0)return -1;return np_get_f32(&G,&a,i,j,o)?0:-1;}
extern "C" unsigned int h_scalar_bits(int ref){const int* n=node(&G,ref);if(!n||n[0]!=T_NP_SCALAR)return 0;return (unsigned int)n[3];}
extern "C" int h_align_mod(int ref){NpArr a;if(np_load(&G,ref,&a)<=0)return -1;return (int)((unsigned long long)(G.h.payload+a.data)&3ull);}
extern "C" int h_gc(int ref){int tail=0;GC_LAST=MISSING;int hit=numpy_gc_trace(&G,ref,&tail);return hit?GC_LAST:MISSING;}
extern "C" int h_native_zeros(){return NATIVE_NP_ZEROS;}
extern "C" int h_native_asarray(){return NATIVE_NP_ASARRAY;}
extern "C" int h_native_concat(){return NATIVE_NP_CONCAT;}
extern "C" int h_native_equal(){return NATIVE_NP_EQUAL;}
extern "C" int h_native_float32(){return NATIVE_NP_FLOAT32;}
extern "C" int h_native_isfinite(){return NATIVE_NP_ISFINITE;}
extern "C" int h_bool_val(int ref){const int* n=node(&G,ref);if(!n||n[0]!=TAG_BOOL)return -1;return n[3]?1:0;}
}
"""


def _compile_lib():
    compiler = None
    for cand in ("clang++", "c++", "g++"):
        path = shutil.which(cand)
        if path:
            compiler = path
            break
    if not compiler:
        raise unittest.SkipTest("No C++ compiler found; skipping host NumPy VM tests")
    src = _context_stub() + emit_numpy_source(SUPPORTED_NUMPY_NAMES) + _wrapper()
    tmp = tempfile.TemporaryDirectory()
    cpp = os.path.join(tmp.name, "vm_numpy_host.cpp")
    lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
    so = os.path.join(tmp.name, "libvm_numpy_host" + lib_ext)
    with open(cpp, "w", encoding="ascii") as fh:
        fh.write(src)
    result = subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O0", cpp, "-o", so], capture_output=True, text=True)
    if result.returncode != 0:
        tmp.cleanup()
        raise AssertionError("C++ compile failed:\n" + result.stdout + result.stderr)
    lib = ctypes.CDLL(so)
    lib.h_reset.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.h_str.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.h_tuple.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.h_call_np.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.h_attr.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.h_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.h_item.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.h_int.argtypes = [ctypes.c_longlong]
    lib.h_float.argtypes = [ctypes.c_double]
    for name in (
        "h_missing", "h_err", "h_herr", "h_none", "h_bool", "h_int", "h_float", "h_str", "h_list", "h_tuple",
        "h_dict", "h_append", "h_dset", "h_slice", "h_kind", "h_native", "h_call_np", "h_attr", "h_sub", "h_store",
        "h_len", "h_truth", "h_iter", "h_next", "h_num", "h_align_mod", "h_gc", "h_native_zeros", "h_native_asarray",
        "h_native_concat", "h_native_equal", "h_native_float32", "h_native_isfinite", "h_reset",
        "h_bool_val",
    ):
        getattr(lib, name).restype = ctypes.c_int
    lib.h_scalar_bits.restype = ctypes.c_uint
    lib._tmp = tmp
    return lib


class Host:
    def __init__(self, lib) -> None:
        self.lib = lib
        self.reset()

    def reset(self, mn: int = 1024, me: int = 4096, mp: int = 65536) -> None:
        self.lib.h_reset(mn, me, mp)

    def check(self, ref: int) -> int:
        if self.lib.h_err() or self.lib.h_herr() or ref == MISSING:
            raise AssertionError("vm=%s heap=%s ref=%s" % (self.lib.h_err(), self.lib.h_herr(), ref))
        return ref

    def i(self, v: int) -> int:
        return self.check(self.lib.h_int(v))

    def f(self, v: float) -> int:
        return self.check(self.lib.h_float(v))

    def b(self, v: bool) -> int:
        return self.check(self.lib.h_bool(1 if v else 0))

    def s(self, text: str) -> int:
        raw = text.encode("utf-8")
        return self.check(self.lib.h_str(raw, len(raw)))

    def lst(self, values) -> int:
        ref = self.check(self.lib.h_list())
        for item in values:
            if self.lib.h_append(ref, item) != 0 or self.lib.h_err() or self.lib.h_herr():
                raise AssertionError("append failed")
        return ref

    def tup(self, values) -> int:
        arr = (ctypes.c_int * len(values))(*values)
        return self.check(self.lib.h_tuple(arr, len(values)))

    def native(self, name: str) -> int:
        ids = {
            "numpy.zeros": self.lib.h_native_zeros(),
            "numpy.asarray": self.lib.h_native_asarray(),
            "numpy.concatenate": self.lib.h_native_concat(),
            "numpy.array_equal": self.lib.h_native_equal(),
            "numpy.float32": self.lib.h_native_float32(),
            "numpy.isfinite": self.lib.h_native_isfinite(),
        }
        return self.check(self.lib.h_native(ids[name]))

    def call(self, name: str, args, kwargs=None) -> int:
        callee = self.native(name)
        arr = (ctypes.c_int * len(args))(*args) if args else None
        kn = MISSING
        kv = None
        nkw = 0
        if kwargs:
            names = [self.s(key) for key in kwargs]
            kn = self.tup(names)
            kv = (ctypes.c_int * len(kwargs))(*kwargs.values())
            nkw = len(kwargs)
        return self.check(self.lib.h_call_np(callee, arr, len(args), kn, kv, nkw))

    def method(self, owner: int, name: str, *args) -> int:
        bound = self.check(self.lib.h_attr(owner, name.encode("ascii")))
        arr = (ctypes.c_int * len(args))(*args) if args else None
        return self.check(self.lib.h_call_np(bound, arr, len(args), MISSING, None, 0))

    def dtype_f32(self) -> int:
        return self.native("numpy.float32")

    def zeros(self, shape) -> int:
        shape_ref = self.tup([self.i(v) for v in shape]) if isinstance(shape, tuple) else self.i(shape)
        return self.call("numpy.zeros", [shape_ref], {"dtype": self.dtype_f32()})

    def asarray(self, seq_ref: int) -> int:
        return self.call("numpy.asarray", [seq_ref], {"dtype": self.dtype_f32()})

    def to_numpy(self, ref: int) -> np.ndarray:
        ndim = ctypes.c_int()
        s0 = ctypes.c_int()
        s1 = ctypes.c_int()
        dtype = ctypes.c_int()
        root = ctypes.c_int()
        if self.lib.h_info(ref, ctypes.byref(ndim), ctypes.byref(s0), ctypes.byref(s1), ctypes.byref(dtype), ctypes.byref(root)) != 0:
            raise AssertionError("not an ndarray")
        rows, cols = s0.value, s1.value
        data = np.empty((rows, cols) if ndim.value == 2 else (rows,), dtype=np.float32)
        item = ctypes.c_float()
        if ndim.value == 1:
            for i in range(rows):
                if self.lib.h_item(ref, i, 0, ctypes.byref(item)) != 0:
                    raise AssertionError("item")
                data[i] = item.value
        else:
            for i in range(rows):
                for j in range(cols):
                    if self.lib.h_item(ref, i, j, ctypes.byref(item)) != 0:
                        raise AssertionError("item")
                    data[i, j] = item.value
        if dtype.value == NP_DTYPE_BOOL:
            data = data.astype(bool)
        return data

    def scalar_f32(self, ref: int) -> float:
        bits = self.lib.h_scalar_bits(ref)
        return ctypes.c_float.from_buffer(ctypes.c_uint(bits)).value


class TestVmNumpySource(unittest.TestCase):
    def test_frozen_names_and_tags(self) -> None:
        self.assertEqual(
            SUPPORTED_NUMPY_NAMES,
            ("numpy.zeros", "numpy.asarray", "numpy.concatenate", "numpy.array_equal", "numpy.float32", "numpy.isfinite"),
        )
        self.assertEqual(SUPPORTED_NUMPY_METHODS, ("reshape", "all", "tolist"))
        self.assertEqual((TAG_NDARRAY, TAG_NDARRAY_METHOD, TAG_NP_DTYPE, TAG_NP_SCALAR, TAG_NDARRAY_ITER), (60, 61, 62, 63, 64))
        src = emit_numpy_source(SUPPORTED_NUMPY_NAMES)
        for hook in (
            "numpy_call", "numpy_attribute", "numpy_try_subscript", "numpy_try_store",
            "numpy_length", "numpy_truth", "numpy_iterator", "numpy_iterator_next", "numpy_numeric", "numpy_gc_trace",
        ):
            self.assertIn(hook + "(", src)
        self.assertIn("T_NDARRAY=60", src)
        self.assertIn("NATIVE_NP_ZEROS=0", src)
        self.assertIn("NP_OP_ADD", src)

    def test_cuda_parity_skipped_without_gpu(self) -> None:
        if _cuda_available():
            self.skipTest("CUDA present; device parity belongs to the primary VM integration")
        self.skipTest("CUDA GPU unavailable on this host; NumPy device parity skipped")


class TestVmNumpyHost(unittest.TestCase):
    lib = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = _compile_lib()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.lib is not None and hasattr(cls.lib, "_tmp"):
            cls.lib._tmp.cleanup()

    def setUp(self) -> None:
        self.host = Host(self.lib)

    def test_zeros_asarray_reshape_concat_match_numpy(self) -> None:
        h = self.host
        self.assertTrue(np.array_equal(h.to_numpy(h.zeros(5)), np.zeros(5, dtype=np.float32)))
        z2 = h.zeros((3, 4))
        self.assertEqual(h.to_numpy(z2).shape, (3, 4))
        self.assertEqual(h.lib.h_len(z2), 3)
        empty = h.method(h.asarray(h.lst([])), "reshape", h.i(-1), h.i(11))
        self.assertEqual(h.to_numpy(empty).shape, (0, 11))
        oracle = np.asarray([1.25, 2, True], dtype=np.float32)
        arr = h.asarray(h.lst([h.f(1.25), h.i(2), h.b(True)]))
        self.assertTrue(np.array_equal(h.to_numpy(arr), oracle))
        cat = h.call("numpy.concatenate", [h.tup([arr, arr])])
        self.assertTrue(np.array_equal(h.to_numpy(cat), np.concatenate((oracle, oracle))))
        left = h.zeros((2, 2))
        right = h.asarray(h.lst([h.lst([h.f(1), h.f(2)]), h.lst([h.f(3), h.f(4)])]))
        both = h.call("numpy.concatenate", [h.tup([left, right])], {"axis": h.i(1)})
        want = np.concatenate((np.zeros((2, 2), dtype=np.float32), np.asarray([[1, 2], [3, 4]], dtype=np.float32)), axis=1)
        self.assertTrue(np.array_equal(h.to_numpy(both), want))

    def test_encode_like_indexing_and_isfinite(self) -> None:
        h = self.host
        n_cards, n_actions, extra_dim = 8, 3, 12
        extra = h.zeros((n_actions, extra_dim))
        marks = h.zeros((3, n_cards))
        h.check(h.lib.h_store(marks, h.tup([h.i(2), h.i(1)]), h.f(0.5)))
        h.check(h.lib.h_store(extra, h.tup([h.i(0), h.i(1)]), h.f(0.5)))
        sl = h.lib.h_slice(h.i(4), h.i(8), h.i(1))
        h.check(h.lib.h_store(extra, h.tup([h.i(1), sl]), h.lst([h.f(1), h.f(0), h.f(1), h.f(0)])))
        base_vals = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] for _ in range(n_actions)]
        base = h.asarray(h.lst([h.lst([h.i(v) for v in row]) for row in base_vals]))
        h.check(h.lib.h_store(base, h.i(0), h.lst([h.i(9), h.i(-1), h.i(-1), h.i(-1), h.i(-1), h.i(0), h.i(0), h.i(0), h.i(0), h.i(0), h.i(0)])))
        state = h.call("numpy.concatenate", [h.tup([h.asarray(h.lst([h.f(0.1), h.f(0.2), h.f(0.3)])), h.method(marks, "reshape", h.i(-1))])])
        cand = h.call("numpy.concatenate", [h.tup([base, extra])], {"axis": h.i(1)})
        self.assertTrue(h.method(h.call("numpy.isfinite", [state]), "all"))
        extra_np = np.zeros((n_actions, extra_dim), dtype=np.float32)
        marks_np = np.zeros((3, n_cards), dtype=np.float32)
        marks_np[2, 1] += 0.5
        extra_np[0, 1] += 0.5
        extra_np[1, 4:8] = [1.0, 0.0, 1.0, 0.0]
        base_np = np.asarray(base_vals, dtype=np.float32)
        base_np[0] = [9, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        state_np = np.concatenate((np.asarray([0.1, 0.2, 0.3], dtype=np.float32), marks_np.reshape(-1)))
        cand_np = np.concatenate((base_np, extra_np), axis=1)
        self.assertTrue(np.array_equal(h.to_numpy(state), state_np))
        self.assertTrue(np.array_equal(h.to_numpy(cand), cand_np))

    def test_scalar_add_view_alias_signed_zero_and_alignment(self) -> None:
        h = self.host
        h.s("x")
        arr = h.zeros(4)
        self.assertEqual(h.lib.h_align_mod(arr), 0)
        scalar = h.call("numpy.float32", [h.f(1.25)])
        added = h.check(h.lib.h_num(scalar, h.f(0.5), NP_OP_ADD))
        self.assertEqual(h.lib.h_kind(added), TAG_NP_SCALAR)
        self.assertEqual(np.float32(h.scalar_f32(added)), np.float32(1.25) + np.float32(0.5))
        view = h.method(arr, "reshape", h.i(2), h.i(2))
        h.check(h.lib.h_store(view, h.tup([h.i(0), h.i(1)]), h.f(3.5)))
        self.assertEqual(h.to_numpy(arr)[1], np.float32(3.5))
        self.assertEqual(h.lib.h_gc(view), arr)
        z = h.zeros(2)
        h.check(h.lib.h_store(z, h.i(0), h.f(0.0)))
        h.check(h.lib.h_store(z, h.i(1), h.f(-0.0)))
        eq = h.call("numpy.array_equal", [z, h.zeros(2)])
        self.assertEqual(h.lib.h_bool_val(eq), 1)
        inf = h.call("numpy.float32", [h.f(1e40)])
        inf_arr = h.zeros(1)
        h.check(h.lib.h_store(inf_arr, h.i(0), inf))
        self.assertFalse(bool(h.to_numpy(h.call("numpy.isfinite", [inf_arr])).astype(bool)[0]))
        nan = h.check(h.lib.h_num(inf, h.f(0.0), NP_OP_MUL))
        nan_arr = h.zeros(1)
        h.check(h.lib.h_store(nan_arr, h.i(0), nan))
        self.assertEqual(h.lib.h_bool_val(h.call("numpy.array_equal", [nan_arr, nan_arr])), 0)
        physical = h.zeros(8)
        nxt = h.check(h.lib.h_num(h.check(h.lib.h_sub(physical, h.i(3))), h.f(0.5), NP_OP_ADD))
        h.check(h.lib.h_store(physical, h.i(3), nxt))
        oracle = np.zeros(8, dtype=np.float32)
        oracle[3] += 0.5
        self.assertTrue(np.array_equal(h.to_numpy(physical), oracle))
        self.assertEqual(h.lib.h_kind(h.method(physical, "tolist")), 5)

    def test_error_paths(self) -> None:
        h = self.host
        h.lib.h_call_np(h.native("numpy.zeros"), (ctypes.c_int * 1)(h.i(2)), 1, MISSING, None, 0)
        self.assertEqual(h.lib.h_err(), UNSUPPORTED_TYPE)
        h.reset()
        kn = h.tup([h.s("order")])
        kv = (ctypes.c_int * 1)(h.dtype_f32())
        h.lib.h_call_np(h.native("numpy.zeros"), (ctypes.c_int * 1)(h.i(2)), 1, kn, kv, 1)
        self.assertEqual(h.lib.h_err(), BAD_CALL)
        h.reset()
        h.lib.h_truth(h.zeros((2, 3)))
        self.assertEqual(h.lib.h_err(), UNSUPPORTED_TYPE)
        h.reset()
        stale = h.zeros(3)
        h.lib.h_num(stale, h.f(1.0), NP_OP_ADD)
        self.assertEqual(h.lib.h_err(), UNSUPPORTED_TYPE)
        h.reset()
        h.lib.h_reset(16, 16, 96)
        dtype = h.lib.h_native(h.lib.h_native_float32())
        names = h.lib.h_tuple((ctypes.c_int * 1)(h.lib.h_str(b"dtype", 5)), 1)
        h.lib.h_call_np(h.lib.h_native(h.lib.h_native_zeros()), (ctypes.c_int * 1)(h.lib.h_int(64)), 1, names, (ctypes.c_int * 1)(dtype), 1)
        self.assertTrue(h.lib.h_herr() or h.lib.h_err())


if __name__ == "__main__":
    unittest.main()
