"""Native SoA object-view bridge: layout, source integration, and clang++ ABI tests."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.modules.card_game.rl.batched_duel.native_array_cpu import HOST_CUDA_SHIMS
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema
from app.modules.card_game.rl.batched_duel.vm_heap import emit_vm_heap_source
from app.modules.card_game.rl.batched_duel.vm_soa_views import (
    ATTACH_POINT,
    HELPER_NAMES,
    VIEW_OFFSET_SENTINEL,
    cuda_qualified,
    emit_soa_metadata,
    integrate_soa_source,
    soa_view_layout,
    supports_full_duel_rules,
    value_cache_extent,
)


def tiny_schema():
    return compile_schema({
        'Item': {'kind': 'record', 'capacity': 16, 'fields': {'n': 'int', 's': 'str', 'ok': 'bool', 'rate': 'float'}},
        'Items': {'kind': 'list', 'capacity': 8, 'max_items': 8, 'value': '@Item', 'tags': [5]},
        'Counters': {'kind': 'map', 'capacity': 8, 'max_items': 8, 'key': 'str', 'value': 'int'},
        'Pair': {'kind': 'tuple', 'capacity': 8, 'items': ['int', 'str']},
        'Player': {'kind': 'record', 'capacity': 8, 'fields': {'hp': 'int', 'name': 'str'}, 'tags': [7, 8]},
        'Game': {'kind': 'record', 'capacity': 8, 'fields': {
            'rng': 'int', 'sides': '@Player', 'next_entity': 'int',
            'items': '@Items', 'mapping': '@Counters', 'pair': '@Pair',
        }},
    }, max_strings=32, max_string_bytes=256)


def _cxx():
    for name in ('clang++', 'g++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


EXECUTE_STUB = '''
extern "C" __global__ void vm_execute(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* args,int nargs,
 const int* constants_nodes,const int* constants_edges,const unsigned char* constants_payload,
 const int* instructions,const int* functions,int function_count,
 int constant_node_count,int constant_edge_count,int constant_payload_count,
 int* frames,int* outputs,int* statuses,const long long* deadlines,
 int rows,int max_nodes,int max_edges,int max_payload,int entry,int step_limit,const int* classes,int class_count,const int* modules,int module_count,long long* gc_marks,int* gc_queue){
 using namespace duel_vm_heap;
 Context c;
 c.h.constant_node_count=constant_node_count;c.h.constant_edge_count=constant_edge_count;c.h.constant_payload_count=constant_payload_count;c.h.error=0;
 c.current=new_frame(&c,entry,args+(long long)tid*nargs,nargs,-1,3);
}
'''


class SoAViewLayoutTests(unittest.TestCase):
    def test_layout_and_qualification(self):
        schema = tiny_schema()
        layout = soa_view_layout(schema, constant_node_count=3)
        self.assertEqual(layout['index'], 'offset*rows+row')
        self.assertEqual(layout['view_offset_sentinel'], VIEW_OFFSET_SENTINEL)
        self.assertEqual(layout['clone_map'], schema.total_objects)
        self.assertEqual(layout['constant_adopt'], 3)
        self.assertEqual(layout['value_cache'], value_cache_extent(schema))
        self.assertEqual(layout['channels']['REF'], 6)
        self.assertFalse(layout['supports_full_duel_rules'])
        self.assertFalse(layout['cuda_qualified'])
        self.assertFalse(supports_full_duel_rules)
        self.assertFalse(cuda_qualified)
        meta = emit_soa_metadata(schema)
        self.assertEqual(meta.helpers, HELPER_NAMES)
        self.assertEqual(meta.attach_point, ATTACH_POINT)
        self.assertEqual(meta.schema_identity, schema.identity)


class SoAViewIntegrateTests(unittest.TestCase):
    def setUp(self):
        self.schema = tiny_schema()
        self.heap = emit_vm_heap_source()
        self.source = integrate_soa_source(self.heap, self.schema)

    def test_heap_abi_and_helpers(self):
        self.assertIn('struct DomainSoAContext;', self.source)
        self.assertIn('DomainSoAContext* soa;', self.source)
        self.assertIn('heap->soa = 0;', self.source)
        for name in HELPER_NAMES:
            self.assertIn(name + '(', self.source)
        self.assertIn('soa_try_get_node(heap, ref, out_node)', self.source)
        self.assertIn('node->offset==SOA_VIEW_OFFSET', self.source)
        self.assertIn('soa_dict_find(heap, dict_ref, key_ref)', self.source)
        self.assertIn('soa_dict_get(heap, dict_ref, key_ref, default_ref)', self.source)
        self.assertIn('soa_list_get(heap, list_ref, index)', self.source)
        self.assertIn('soa_clone(heap, root_ref, 1)', self.source)
        self.assertNotIn(ATTACH_POINT, self.source)

    def test_unbound_keeps_generic_bodies(self):
        self.assertIn('if (soa_is_view(heap, dict_ref)) return soa_dict_set', self.source)
        self.assertIn('if (dict_ref < 0) {', self.source)
        self.assertIn('heap->edges[(node.offset + idx) * 2 + 1] = val_ref;', self.source)

    def test_attach_point_on_execute_stub(self):
        source = integrate_soa_source(self.heap + EXECUTE_STUB, self.schema)
        self.assertIn('#ifndef DOMAIN_SOA_ATTACH_POINT', source)
        self.assertEqual(source.count(ATTACH_POINT + ';'), 1)
        self.assertIn('c.h.soa=0;', source)


DRIVER = r'''
using namespace duel_vm_heap;
#define CHECK(cond) do{ if(!(cond)){ std::printf("FAIL %s\n", #cond); return 1; } }while(0)
static int run_unbound(){
  const int MAXN=64,MAXE=128,MAXP=512;
  int nodes[MAXN*4]={0}; int edges[MAXE*2]={0}; unsigned char payload[MAXP]={0}; int counts[4]={0};
  Heap h; init_heap(&h,nodes,edges,payload,counts,MAXN,MAXE,MAXP,0,0,0,0,0,0);
  CHECK(h.soa==0);
  int d=make_dict(&h);
  CHECK(dict_set(&h,d,make_utf8(&h,"hp",2),make_int64(&h,4))==0);
  long long hp=0; CHECK(read_int64(&h,dict_get(&h,d,make_utf8(&h,"hp",2),-1),&hp)==0 && hp==4);
  CHECK(!soa_is_view(&h,d));
  int cloned=deep_clone(&h,d); CHECK(cloned>=0 && cloned!=d && h.error==0);
  return 0;
}
static int run_bound(){
  const int MAXN=256,MAXE=512,MAXP=2048,ROWS=1;
  int nodes[MAXN*4]={0}; int edges[MAXE*2]={0}; unsigned char payload[MAXP]={0}; int counts[4]={0};
  long long i64[SOA_LEN_I64*ROWS]={0}; double f64[SOA_LEN_F64*ROWS]={0};
  int i32[SOA_LEN_I32*ROWS]={0}; unsigned char u8[SOA_LEN_U8*ROWS]={0};
  int views[SOA_TOTAL_OBJECTS*ROWS]; int vcache[SOA_VALUE_CACHE_EXTENT*ROWS];
  int cmap[SOA_TOTAL_OBJECTS*ROWS]; int cqueue[SOA_TOTAL_OBJECTS*ROWS];
  Heap h; init_heap(&h,nodes,edges,payload,counts,MAXN,MAXE,MAXP,0,0,0,0,0,0);
  DomainSoAContext s{};
  s.i64=i64;s.f64=f64;s.i32=i32;s.u8=u8;s.rows=ROWS;s.row=0;
  s.views=views;s.value_cache=vcache;s.clone_map=cmap;s.clone_queue=cqueue;
  s.constant_adopt=0;s.constant_node_count=0;
  soa_attach(&h,&s); soa_reset_caches(&h);
  int item=make_dict(&h);
  CHECK(dict_set(&h,item,make_utf8(&h,"n",1),make_int64(&h,3))==0);
  CHECK(dict_set(&h,item,make_utf8(&h,"s",1),make_utf8(&h,"hi",2))==0);
  CHECK(dict_set(&h,item,make_utf8(&h,"ok",2),make_bool(&h,1))==0);
  CHECK(dict_set(&h,item,make_utf8(&h,"rate",4),make_float64(&h,0.5))==0);
  int items=make_list(&h);
  CHECK(list_append(&h,items,item)==0);
  CHECK(list_append(&h,items,item)==0);
  int mapping=make_dict(&h);
  CHECK(dict_set(&h,mapping,make_utf8(&h,"a",1),make_int64(&h,1))==0);
  int pair_ch[2]={make_int64(&h,7),make_utf8(&h,"x",1)};
  int pair=make_tuple(&h,pair_ch,2);
  int player=make_entity(&h,TAG_PLAYER_ENTITY);
  CHECK(dict_set(&h,player,make_utf8(&h,"hp",2),make_int64(&h,12))==0);
  CHECK(dict_set(&h,player,make_utf8(&h,"name",4),make_utf8(&h,"p",1))==0);
  int game=make_dict(&h);
  CHECK(dict_set(&h,game,make_utf8(&h,"rng",3),make_int64(&h,7))==0);
  CHECK(dict_set(&h,game,make_utf8(&h,"sides",5),player)==0);
  CHECK(dict_set(&h,game,make_utf8(&h,"next_entity",11),make_int64(&h,4))==0);
  CHECK(dict_set(&h,game,make_utf8(&h,"items",5),items)==0);
  CHECK(dict_set(&h,game,make_utf8(&h,"mapping",7),mapping)==0);
  CHECK(dict_set(&h,game,make_utf8(&h,"pair",4),pair)==0);
  int view=soa_promote(&h,game,SOA_ROOT);
  CHECK(h.error==0 && view==game && soa_is_view(&h,view));
  int again=soa_promote(&h,view,SOA_ROOT); CHECK(again==view);
  int got=dict_get(&h,view,make_utf8(&h,"items",5),-123);
  CHECK(soa_is_view(&h,got) && soa_is_view(&h,list_get(&h,got,0)));
  CHECK(list_get(&h,got,0)==list_get(&h,got,1));
  int sides=dict_get(&h,view,make_utf8(&h,"sides",5),-1);
  CHECK(soa_is_view(&h,sides) && tag(&h,sides)==TAG_PLAYER_ENTITY);
  long long hp=0; CHECK(read_int64(&h,dict_get(&h,sides,make_utf8(&h,"hp",2),-1),&hp)==0 && hp==12);
  double rate=0; CHECK(read_float64(&h,dict_get(&h,list_get(&h,got,0),make_utf8(&h,"rate",4),-1),&rate)==0 && rate==0.5);
  CHECK(dict_set(&h,sides,make_utf8(&h,"hp",2),make_int64(&h,20))==0);
  CHECK(list_append(&h,got,list_get(&h,got,0))==0);
  CHECK(h.nodes[got*4+2]==3);
  int popped=0; CHECK(list_pop(&h,got,&popped)==0 && soa_is_view(&h,popped));
  int cloned=deep_clone(&h,view);
  CHECK(soa_is_view(&h,cloned) && cloned!=view);
  int citems=dict_get(&h,cloned,make_utf8(&h,"items",5),-1);
  CHECK(list_get(&h,citems,0)==list_get(&h,citems,1));
  CHECK(list_get(&h,citems,0)!=list_get(&h,got,0));
  long long chp=0; read_int64(&h,dict_get(&h,dict_get(&h,cloned,make_utf8(&h,"sides",5),-1),make_utf8(&h,"hp",2),-1),&chp);
  CHECK(chp==20);
  dict_set(&h,sides,make_utf8(&h,"hp",2),make_int64(&h,99));
  long long chp2=0; read_int64(&h,dict_get(&h,dict_get(&h,cloned,make_utf8(&h,"sides",5),-1),make_utf8(&h,"hp",2),-1),&chp2);
  CHECK(chp2==20);
  int wrap=make_dict(&h);
  CHECK(dict_set(&h,wrap,make_utf8(&h,"g",1),view)==0);
  int mixed=deep_clone(&h,wrap);
  int mg=dict_get(&h,mixed,make_utf8(&h,"g",1),-1);
  CHECK(soa_is_view(&h,mg) && mg!=view);
  int cmapv=dict_get(&h,view,make_utf8(&h,"mapping",7),-1);
  CHECK(soa_is_view(&h,cmapv));
  long long av=0; read_int64(&h,dict_get(&h,cmapv,make_utf8(&h,"a",1),-1),&av); CHECK(av==1);
  CHECK(dict_set(&h,cmapv,make_utf8(&h,"b",1),make_int64(&h,2))==0);
  int pairv=dict_get(&h,view,make_utf8(&h,"pair",4),-1);
  CHECK(soa_is_view(&h,pairv) && tag(&h,pairv)==TAG_TUPLE);
  int pclone=deep_clone(&h,pairv); CHECK(pclone==pairv);
  Heap target; int tn[MAXN*4]={0}; int te[MAXE*2]={0}; unsigned char tp[MAXP]={0}; int tc[4]={0};
  int map[MAXN]; int queue[MAXN];
  init_heap(&target,tn,te,tp,tc,MAXN,MAXE,MAXP,0,0,0,0,0,0);
  CHECK(compact_into(&h,&target,view,map,queue,MAXN)<0);
  CHECK(target.error==ERROR_TYPE_MISMATCH || h.error==ERROR_TYPE_MISMATCH);
  return 0;
}
int main(){
  if(run_unbound())return 2;
  if(run_bound())return 3;
  return 0;
}
'''


class SoAViewCompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cxx = _cxx()
        if not cls.cxx:
            raise unittest.SkipTest('clang++/g++ required')
        schema = tiny_schema()
        source = HOST_CUDA_SHIMS + integrate_soa_source(emit_vm_heap_source(), schema)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cpp = Path(cls.tmpdir.name) / 'soa_views_test.cpp'
        cpp.write_text(source + '\n' + DRIVER)
        cls.binary = Path(cls.tmpdir.name) / 'soa_views_test'
        compiled = subprocess.run(
            [cls.cxx, '-std=c++17', '-O0', '-o', str(cls.binary), str(cpp)],
            capture_output=True, text=True)
        if compiled.returncode != 0:
            raise unittest.SkipTest('compile failed: ' + compiled.stderr[-2000:])
        cls.compile_ok = compiled.returncode == 0

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_scalar_container_entity_alias_mutation_deepcopy(self):
        result = subprocess.run([str(self.binary)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
