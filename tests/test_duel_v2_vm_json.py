"""Host compilation tests for device json.dumps used by recovery own-deck keys."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from app.modules.card_game.rl.batched_duel.vm_heap import emit_vm_heap_source
from app.modules.card_game.rl.batched_duel.vm_json import (
    SUPPORTED_JSON_NAMES,
    emit_json_source,
)

MISSING = -2147483647 - 1
BAD_CALL = 104
UNSUPPORTED_TYPE = 106


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _context_stub() -> str:
    return (
        "#define __device__\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n"
        + emit_vm_heap_source()
        + r"""
namespace duel_rule_vm {
using namespace duel_vm_heap;
const int MISSING=(-2147483647-1), NIL=-1;
const int T_NATIVE=15;
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
__device__ inline int integer_string(Context* c,int ref,int plus){
 long long v=0;if(numeric_int(c,ref,&v)<0)return MISSING; char buf[48]; int n=std::snprintf(buf,sizeof(buf), plus?(v>=0?"+%lld":"%lld"):"%lld", v);
 if(n<0||n>=(int)sizeof(buf)){c->error=UNSUPPORTED_TYPE;return MISSING;} return make_utf8(&c->h,buf,n);
}
__device__ inline int float_string(Context* c,int ref){
 double v=0;if(read_float64(&c->h,ref,&v)<0)return MISSING; char buf[64];
 if(v==0.0){const char* s=std::signbit(v)?"-0.0":"0.0";return make_utf8(&c->h,s,(int)std::strlen(s));}
 int n=std::snprintf(buf,sizeof(buf),"%.17g",v); if(n<0||n>=(int)sizeof(buf)){c->error=UNSUPPORTED_TYPE;return MISSING;}
 int has=0; for(int i=0;i<n;i++) if(buf[i]=='.'||buf[i]=='e'||buf[i]=='E') has=1;
 if(!has && n+2<(int)sizeof(buf)){buf[n++]='.';buf[n++]='0';buf[n]=0;} return make_utf8(&c->h,buf,n);
}
"""
    )


def _wrapper() -> str:
    return r"""
static Context G;
static int NODES[4096];
static int EDGES[8192];
static unsigned char PAYLOAD[65536];
static int COUNTS[4];
extern "C" int h_err(){return G.error;}
extern "C" int h_herr(){return G.h.error;}
extern "C" int h_reset(int mn,int me,int mp){
 if(mn<8||me<8||mp<32||mn>4096||me>8192||mp>65536) return -1;
 std::memset(NODES,0,sizeof(NODES));std::memset(EDGES,0,sizeof(EDGES));std::memset(PAYLOAD,0,sizeof(PAYLOAD));std::memset(COUNTS,0,sizeof(COUNTS));
 init_heap(&G.h,NODES,EDGES,PAYLOAD,COUNTS,mn,me,mp,0,0,0,0,0,0);G.error=0;return 0;
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
extern "C" int h_kind(int ref){return kind(&G,ref);}
extern "C" int h_dumps(){return meta(&G,T_NATIVE,0,0,NATIVE_JSON_DUMPS);}
extern "C" int h_call(int callee,const int* args,int nargs,int kn,const int* kv,int nkw){
 int out=MISSING;int hit=json_call(&G,callee,args,nargs,kn,kv,nkw,&out);
 if(!hit&&!G.error&&!G.h.error) G.error=UNSUPPORTED_TYPE; return out;
}
extern "C" int h_read_str(int ref,char* buf,int cap){
 NodeView v;if(!get_node(&G.h,ref,&v)||v.tag!=TAG_STR||v.count>=cap)return -1;
 const unsigned char* p=get_payload_ptr(&G.h,&v);if(v.count&&!p)return -1;for(int i=0;i<v.count;i++)buf[i]=(char)p[i];buf[v.count]=0;return v.count;
}
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
        raise unittest.SkipTest("No C++ compiler found; skipping host JSON VM tests")
    src = _context_stub() + emit_json_source(SUPPORTED_JSON_NAMES) + _wrapper()
    tmp = tempfile.TemporaryDirectory()
    cpp = os.path.join(tmp.name, "vm_json_host.cpp")
    lib_ext = ".dylib" if sys.platform == "darwin" else ".so"
    so = os.path.join(tmp.name, "libvm_json_host" + lib_ext)
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
    lib.h_call.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.h_read_str.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.h_int.argtypes = [ctypes.c_longlong]
    lib.h_float.argtypes = [ctypes.c_double]
    for name in ("h_err", "h_herr", "h_reset", "h_none", "h_bool", "h_int", "h_float", "h_str", "h_list", "h_tuple", "h_dict", "h_append", "h_dset", "h_kind", "h_dumps", "h_call", "h_read_str"):
        getattr(lib, name).restype = ctypes.c_int
    lib._tmp = tmp
    return lib


class Host:
    def __init__(self, lib) -> None:
        self.lib = lib
        self.reset()

    def reset(self, mn: int = 512, me: int = 2048, mp: int = 16384) -> None:
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

    def none(self) -> int:
        return self.check(self.lib.h_none())

    def lst(self, values) -> int:
        ref = self.check(self.lib.h_list())
        for item in values:
            if self.lib.h_append(ref, item) != 0 or self.lib.h_err() or self.lib.h_herr():
                raise AssertionError("append failed")
        return ref

    def tup(self, values) -> int:
        arr = (ctypes.c_int * len(values))(*values)
        return self.check(self.lib.h_tuple(arr, len(values)))

    def dct(self, items) -> int:
        ref = self.check(self.lib.h_dict())
        for key, value in items:
            if self.lib.h_dset(ref, key, value) != 0 or self.lib.h_err() or self.lib.h_herr():
                raise AssertionError("dict_set failed")
        return ref

    def dumps(self, obj: int, kwargs=None) -> str:
        callee = self.check(self.lib.h_dumps())
        args = (ctypes.c_int * 1)(obj)
        kn = MISSING
        kv = None
        nkw = 0
        if kwargs:
            names = [self.s(key) for key in kwargs]
            kn = self.tup(names)
            kv = (ctypes.c_int * len(kwargs))(*kwargs.values())
            nkw = len(kwargs)
        out = self.check(self.lib.h_call(callee, args, 1, kn, kv, nkw))
        buf = ctypes.create_string_buffer(4096)
        n = self.lib.h_read_str(out, buf, 4096)
        if n < 0:
            raise AssertionError("read str")
        return buf.value.decode("utf-8")

    def py(self, value):
        if value is None:
            return self.none()
        if isinstance(value, bool):
            return self.b(value)
        if isinstance(value, int):
            return self.i(value)
        if isinstance(value, float):
            return self.f(value)
        if isinstance(value, str):
            return self.s(value)
        if isinstance(value, tuple):
            return self.tup([self.py(item) for item in value])
        if isinstance(value, list):
            return self.lst([self.py(item) for item in value])
        if isinstance(value, dict):
            return self.dct([(self.s(str(k)), self.py(v)) for k, v in value.items()])
        raise TypeError(type(value))


class TestVmJsonSource(unittest.TestCase):
    def test_frozen_names_without_private_heap_tags(self) -> None:
        self.assertEqual(SUPPORTED_JSON_NAMES, ("json.dumps",))
        src = emit_json_source(SUPPORTED_JSON_NAMES)
        self.assertIn("json_call(", src)
        self.assertIn("NATIVE_JSON_DUMPS=0", src)
        self.assertIn("sort_keys", src)
        self.assertNotIn("TAG_JSON", src)
        for tag in range(70, 80):
            self.assertNotIn("=%s;" % tag, src.replace(" ", ""))

    def test_cuda_parity_skipped_without_gpu(self) -> None:
        if _cuda_available():
            self.skipTest("CUDA present; device parity belongs to the primary VM integration")
        self.skipTest("CUDA GPU unavailable on this host; JSON device parity skipped")


class TestVmJsonHost(unittest.TestCase):
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

    def test_recovery_own_deck_keys_match_cpython(self) -> None:
        fixtures = [
            ("A03", {"ephemeral_turn": 3}),
            ("A03", {"copy": True, "ephemeral_turn": 3}),
            ("C02", {"expires_turn": 4, "jingu_mark": {"delta": -1, "source_entity_id": "e12"}}),
            ("NF01", {}),
            ("Q05", {"copy": False}),
            ("x", {"a": None, "b": True, "c": 0}),
        ]
        for item in fixtures:
            got = self.host.dumps(self.host.py(item), {"sort_keys": self.host.b(True)})
            want = json.dumps(item, sort_keys=True)
            self.assertEqual(got, want)
        shuffled = list(fixtures)
        shuffled.sort(key=lambda c: json.dumps(c, sort_keys=True))
        device_sorted = list(fixtures)
        device_sorted.sort(key=lambda c: self.host.dumps(self.host.py(c), {"sort_keys": self.host.b(True)}))
        self.assertEqual(device_sorted, shuffled)

    def test_escaping_unicode_separators_and_floats(self) -> None:
        h = self.host
        self.assertEqual(h.dumps(h.s("quote\\and\n")), json.dumps("quote\\and\n"))
        self.assertEqual(h.dumps(h.s("\x01")), json.dumps("\x01"))
        self.assertEqual(h.dumps(h.s("\u4e2d")), json.dumps("\u4e2d"))
        self.assertEqual(h.dumps(h.s("\u4e2d"), {"ensure_ascii": h.b(False)}), json.dumps("\u4e2d", ensure_ascii=False))
        compact = h.dumps(h.py({"b": 1, "a": 2}), {"sort_keys": h.b(True), "separators": h.tup([h.s(","), h.s(":")])})
        self.assertEqual(compact, json.dumps({"b": 1, "a": 2}, sort_keys=True, separators=(",", ":")))
        self.assertEqual(h.dumps(h.f(1.0)), json.dumps(1.0))
        self.assertEqual(h.dumps(h.f(-0.0)), json.dumps(-0.0))
        self.assertEqual(h.dumps(h.py([None, True, False, 0, -2])), json.dumps([None, True, False, 0, -2]))

    def test_error_paths(self) -> None:
        h = self.host
        callee = h.check(h.lib.h_dumps())
        kn = h.tup([h.s("indent")])
        kv = (ctypes.c_int * 1)(h.i(2))
        h.lib.h_call(callee, (ctypes.c_int * 1)(h.s("x")), 1, kn, kv, 1)
        self.assertEqual(h.lib.h_err(), BAD_CALL)
        h.reset()
        cyclic = h.lst([])
        h.lib.h_append(cyclic, cyclic)
        callee = h.check(h.lib.h_dumps())
        h.lib.h_call(callee, (ctypes.c_int * 1)(cyclic), 1, MISSING, None, 0)
        self.assertEqual(h.lib.h_err(), UNSUPPORTED_TYPE)
        h.reset()
        bad = h.dct([(h.i(1), h.i(2))])
        callee = h.check(h.lib.h_dumps())
        h.lib.h_call(callee, (ctypes.c_int * 1)(bad), 1, MISSING, None, 0)
        self.assertEqual(h.lib.h_err(), UNSUPPORTED_TYPE)
        h.reset()
        h.lib.h_reset(16, 16, 40)
        callee = h.lib.h_dumps()
        h.lib.h_call(callee, (ctypes.c_int * 1)(h.lib.h_str(b"abcdefghijklmnopqrstuvwxyz0123456789", 36)), 1, MISSING, None, 0)
        self.assertTrue(h.lib.h_herr() or h.lib.h_err())


if __name__ == "__main__":
    unittest.main()
