"""CPU source/ABI checks and a compiled small-pool harness, never CUDA evidence."""
import copy
import ast
from dataclasses import replace
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.batched_duel import native_preview_arena as arena
from app.modules.card_game.rl.batched_duel.native_compile_units import NativeCompileUnit
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema
from app.modules.card_game.rl.batched_duel.vm_soa_views import _emit_schema_tables


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / 'artifacts/rl-evals/soa-capacity-1017-01/teacher-native-program-102.json'


def small_schema():
    return compile_schema({
        'Game': dict(kind='record', capacity=4, fields={'hp': 'int', 'items': '@Items', 'meta': '@Meta', 'pair': '@Pair'}),
        'Items': dict(kind='list', capacity=4, max_items=2, value='number'),
        'Meta': dict(kind='map', capacity=4, max_items=2, key='str', value='number'),
        'Pair': dict(kind='tuple', capacity=4, items=['int','@Items']),
    }, max_strings=16, max_string_bytes=128)


def owner_fixture():
    return '''const int VM_FRAMES=16,VM_LOCALS=128,VM_STACK=32,VM_CELLS=16;
const int HEADER=14, LBASE=HEADER, SBASE=LBASE+VM_LOCALS, CBASE=SBASE+VM_STACK, STRIDE=CBASE+VM_CELLS;
void gc_fixture(){for(int i=0;i<fn[5];i++)gc_ref(c,f[LBASE+i],tail);}
__device__ __noinline__ void native_function_597(Context* c,int* f,const int* fn,int pc){
 const int owner=c->current;
 switch(pc){case 10:goto op_10;case 12:goto op_12;default:return;}
 op_10:{child_call(c);if(!native_function_active(c,f,owner,597))return;}
 op_12:{return_value(c,pop(c,f),0);}
}
'''


class TransformTests(unittest.TestCase):
    def test_cpp_keys_are_ascii_and_exact_utf8_bytes(self):
        helper=arena._emit_helpers(small_schema(),10)
        self.assertTrue(helper.isascii())
        literals=re.findall(r'vm_text_equals\(c,key,("[^"\n]*")\)',helper)
        self.assertEqual(len(literals),4)
        for literal,key in zip(literals,('attack','counter','攻击次数','创生花伤害')):
            escapes=re.findall(r'\\([0-7]{3})',literal[1:-1])
            self.assertEqual(literal[1:-1],''.join('\\'+part for part in escapes))
            self.assertEqual(bytes(int(part,8)for part in escapes),key.encode('utf8'))

    def test_scope_helpers_are_noinline_and_all_runtime_loops_stay_rolled(self):
        helper=arena._emit_helpers(small_schema(),10)
        for name in('charge','enter','released','leave','clear_column','clear_slot'):
            self.assertIn('__device__ __noinline__ int preview_arena_'+name+'(',helper)
        for match in re.finditer(r'(?m)^\s*for\(',helper):
            self.assertRegex(helper[:match.start()].rstrip(),r'#pragma unroll 1$')

    def test_emitted_body_is_constant_size_not_a_schema_expansion(self):
        from types import SimpleNamespace
        small=arena._emit_helpers(small_schema(),10)
        large=arena._emit_helpers(SimpleNamespace(pools=(None,)*83),10)
        self.assertLess(abs(len(small)-len(large)),10)
        self.assertNotIn('switch(p)',large)
        self.assertIn('SOA_POOL_FIELD_START[pool]+i',large)
        self.assertIn('SOA_COL_STRIDE[col]',large)

    def test_stable_ast_ignores_only_empty_type_parameters(self):
        fn=ast.parse('def example(a):\n return a+1\n').body[0]
        baseline=arena._digest(arena._stable_ast(fn))
        extended=copy.deepcopy(fn)
        if 'type_params'not in extended._fields:extended._fields=(*extended._fields,'type_params')
        extended.type_params=[]
        self.assertEqual(baseline,arena._digest(arena._stable_ast(extended)))
        extended.type_params=[ast.Name(id='T',ctx=ast.Load())]
        with self.assertRaises(arena.PreviewArenaError):arena._stable_ast(extended)
        changed=copy.deepcopy(fn);changed.name='different'
        self.assertNotEqual(baseline,arena._digest(arena._stable_ast(changed)))

    def test_only_real_return_is_hooked_and_resume_does_not_reset(self):
        original = owner_fixture()
        source = arena._transform_owner(original, small_schema(), 10)
        body = source[source.index('__device__ __noinline__ void native_function_597'):]
        self.assertIn('preview_arena_enter(c,f,pc)', body)
        self.assertIn('child_call(c);if(!native_function_active(c,f,owner,597))return;', body)
        self.assertEqual(body.count('preview_arena_leave(c,f,preview_result)'), 1)
        self.assertLess(body.index('preview_arena_leave'), body.index('return_value(c,pop(c,f),0)'))
        self.assertIn('if(pc==10)', source)
        self.assertNotIn('soa_reset_caches(', source)
        self.assertNotIn('SOA_STRING_COUNT', source)
        delegated = original.replace('return_value(c,pop(c,f),0);', 'native_step<OP_RETURN_VALUE>(c,f,fn,0,0,-1);')
        transformed = arena._transform_owner(delegated, small_schema(), 10)
        self.assertIn('preview_arena_leave(c,f,preview_result)', transformed)
        self.assertEqual(transformed.count('native_step<OP_RETURN_VALUE>(c,f,fn,0,0,-1);'), 1)

    def test_bad_gc_layout_frame_space_entry_and_double_transform_rejected(self):
        for source in (owner_fixture().replace('VM_LOCALS=128', 'VM_LOCALS=10'),
                       owner_fixture().replace('i<fn[5]', 'i<VM_LOCALS'),
                       owner_fixture().replace('case 10:', 'case 11:'),
                       owner_fixture().replace('return_value(c,pop(c,f),0);', 'return;')):
            with self.assertRaises(arena.PreviewArenaError):
                arena._transform_owner(source, small_schema(), 10)
        with self.assertRaises(arena.PreviewArenaError):
            arena._transform_owner(arena._transform_owner(owner_fixture(), small_schema(), 10), small_schema(), 10)

    def test_full_interface_preserves_other_21_unit_objects_and_new_binding(self):
        # Deliberately explicit synthetic ABI. Public production pins are never
        # inferred from its caller's units; they cannot accept this fixture.
        units = tuple(NativeCompileUnit(f'rules_{i:03d}', owner_fixture() if i == 9 else str(i))
                      for i in range(22))
        expected = {u.name: u.sha256 for u in units}
        with patch.object(arena, 'verify_frozen_preview', return_value=10), \
                patch.object(arena, 'CANONICAL_UNIT_SHA256', expected):
            result = arena.transform_units(units, None, small_schema(), None)
            self.assertTrue(result.verify())
            self.assertEqual(sum(a is b for a, b in zip(units, result.units)), 21)
            self.assertNotEqual(result.units[9].sha256, units[9].sha256)
            self.assertEqual(len(result.identity), 64)
            with self.assertRaises(arena.PreviewArenaError):
                arena.transform_units((*units[:-1], replace(units[-1], source='changed')), None, small_schema(), None)
            bad = replace(result, units=(*result.units[:-1], replace(result.units[-1], source='changed')))
            with self.assertRaises(arena.PreviewArenaError):
                bad.verify()

    @unittest.skipUnless(PROGRAM.exists(), 'Frozen functional program fixture unavailable')
    def test_actual_frozen_program_callgraph_and_metadata_guard(self):
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        program = FrozenRuleProgram.from_json(PROGRAM.read_text())
        image = lower_program(program)
        schema = compile_execution_schema()
        self.assertIsInstance(arena.verify_frozen_preview(program, image, schema), int)
        changed = copy.deepcopy(program.data)
        changed['functions'][560]['instructions'][0]['arg'] = 999
        with self.assertRaises(arena.PreviewArenaError):
            arena.verify_frozen_preview(changed, image, schema)
        with self.assertRaises(arena.PreviewArenaError):
            arena.verify_frozen_preview(program, image, small_schema())
        corrupt_schema = copy.copy(schema)
        corrupt_schema.pools = (replace(schema.pools[0], capacity=schema.pools[0].capacity + 1), *schema.pools[1:])
        with self.assertRaises(arena.PreviewArenaError):
            arena.verify_frozen_preview(program, image, corrupt_schema)
        source = (ROOT / 'app/modules/card_game/engine/duel_v2/projection.py').read_text()
        with self.assertRaises(arena.PreviewArenaError):
            arena.verify_frozen_preview(program, image, schema, projection_source=source.replace('return result', 'return state'))
        with self.assertRaises(arena.PreviewArenaError):
            arena.transform_units((NativeCompileUnit('rules_009', owner_fixture()),), image, schema, program)


class CompiledCPUHarnessTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'Host C++ compiler unavailable')
    def test_emitted_scope_reuse_alias_identity_escape_and_budget(self):
        schema = small_schema()
        cells=sum(sum(c.stride for c in (*p.fields,*((p.keys,)if p.keys is not None else()),*p.values))
                  +(p.max_items if p.order>=0 else 0)for p in schema.pools)
        pools=len(schema.pools)
        # One enter charge/pool, leave admission+score+range checks+three
        # constant roots+headers+cells+counter restore, for full then empty.
        leave_fixed=1+1+pools+3+pools
        expected_steps=12*((pools+leave_fixed+2*pools+2*cells)+(pools+leave_fixed+pools+cells))
        source = (_HARNESS_HEAD + _emit_schema_tables(schema) + _HARNESS_RUNTIME +
                  arena._emit_helpers(schema, 10) + _HARNESS_MAIN.replace('@EXPECTED_STEPS@',str(expected_steps)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scope.cpp'
            path.write_text(source)
            exe = Path(directory) / 'scope'
            compiled = subprocess.run(['c++', '-std=c++17', '-O0', str(path), '-o', str(exe)],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=5)
            self.assertEqual(ran.returncode, 0, ran.stderr)


_HARNESS_HEAD = r'''
#include <cassert>
#include <cstring>
#include <iostream>
#define __device__
#define __noinline__ __attribute__((noinline))
enum { TAG_INT=2,TAG_FLOAT=3,TAG_DICT=7,MISSING=-999,STEP_LIMIT=100,DEADLINE=101,BAD_PROGRAM=102,LBASE=14 };
struct DomainSoAContext {int rows=1,row=0,clone_epoch=7,constant_node_count=3;
 int used[512]={},views[512]={},value_cache[2048]={},adopt[3]={},clone_queue[512]={};
 unsigned char u8[2048]={};long long i64[2048]={};double f64[2048]={};int* constant_adopt=adopt;};
struct Heap {int error=0,nodes[4000]={};DomainSoAContext* soa;};
struct Context {Heap h;int error=0,raised=MISSING,steps=0,step_limit=1000000;long long deadline=0;};
unsigned long long clock_ns(){return 100;}
'''
_HARNESS_RUNTIME = r'''
int soa_ready(Heap*h){return h->soa!=nullptr;}
DomainSoAContext* soa_ctx(Heap*h){return h->soa;}
int soa_load_i32(DomainSoAContext*s,int i){return s->used[i];}
void soa_store_i32(DomainSoAContext*s,int i,int v){s->used[i]=v;}
void soa_store_u8(DomainSoAContext*s,int i,unsigned char v){s->u8[i]=v;}
void soa_store_i64(DomainSoAContext*s,int i,long long v){s->i64[i]=v;}
void soa_store_f64(DomainSoAContext*s,int i,double v){s->f64[i]=v;}
void soa_store_view(DomainSoAContext*s,int i,int v){s->views[i]=v;}
void soa_store_cache(DomainSoAContext*s,int i,int v){s->value_cache[i]=v;}
int fn[18]={0,0,0,0,0,9};const int* function(Context*,int){return fn;}
int* node(Context*c,int r){return r<0?nullptr:c->h.nodes+r*4;}
int soa_header_view(Heap*h,int r){return r>=0&&h->nodes[r*4+1]==SOA_VIEW_OFFSET;}
int soa_locate(int gid,int*p,int*slot){for(int i=0;i<SOA_POOL_COUNT;i++)if(gid>SOA_POOL_BASE[i]&&gid<=SOA_POOL_BASE[i]+SOA_POOL_CAP[i]){*p=i;*slot=gid-SOA_POOL_BASE[i]-1;return 1;}return 0;}
int edge(Context*,int,int i,int which){return which==0?20+i:30+i;}
const char* fixture_keys[]={"attack","counter","\346\224\273\345\207\273\346\254\241\346\225\260","\345\210\233\347\224\237\350\212\261\344\274\244\345\256\263"};
bool vm_text_equals(Context*,int key,const char*s){return key>=20&&key<=23&&std::strcmp(s,fixture_keys[key-20])==0;}
'''
_HARNESS_MAIN = r'''
void setup(Context&c,DomainSoAContext&s,int*f){c.h.soa=&s;f[0]=597;
 for(int p=0;p<SOA_POOL_COUNT;p++)s.used[SOA_POOL_USED[p]]=1;
 for(int i=0;i<512;i++)s.views[i]=77;
 for(int i=0;i<2048;i++)s.value_cache[i]=77;
 for(int col=0;col<SOA_TOTAL_RECORD_FIELDS;col++)for(int j=0;j<SOA_COL_STRIDE[col];j++){
  s.u8[SOA_COL_TAG[col]+j]=123;
  if(SOA_COL_INTEGER[col]>=0)s.i64[SOA_COL_INTEGER[col]+j]=777;
  if(SOA_COL_REAL[col]>=0)s.f64[SOA_COL_REAL[col]+j]=-2.5;
  if(SOA_COL_WORD[col]>=0)s.used[SOA_COL_WORD[col]+j]=444;
 }
 c.h.nodes[40]=TAG_DICT;c.h.nodes[42]=1;c.h.nodes[120]=TAG_INT;
 // Ref 40 is the pre-existing caller object; ref41 belongs to this scope.
 c.h.nodes[160]=TAG_DICT;c.h.nodes[161]=SOA_VIEW_OFFSET;c.h.nodes[163]=1;
 c.h.nodes[164]=TAG_DICT;c.h.nodes[165]=SOA_VIEW_OFFSET;c.h.nodes[167]=2;
 s.adopt[0]=40;s.adopt[1]=41;s.adopt[2]=MISSING;
}
int main(){
 Context c{};DomainSoAContext s{};int f[160]={};setup(c,s,f);
 for(int repeat=0;repeat<12;repeat++){
  assert(preview_arena_enter(&c,f,10));
  for(int p=0;p<SOA_POOL_COUNT;p++)s.used[SOA_POOL_USED[p]]=3;
  // Dirty every field/channel as a previous full object would. A following
  // empty/sparse adoption must never see its old optional keys or payload.
  for(int i=0;i<SOA_TOTAL_RECORD_FIELDS;i++){
   int at=SOA_COL_TAG[i]+SOA_COL_STRIDE[i];s.u8[at]=SOA_REF;
   if(SOA_COL_INTEGER[i]>=0)s.i64[SOA_COL_INTEGER[i]+SOA_COL_STRIDE[i]]=123;
   if(SOA_COL_REAL[i]>=0)s.f64[SOA_COL_REAL[i]+SOA_COL_STRIDE[i]]=1.25;
   if(SOA_COL_WORD[i]>=0)s.used[SOA_COL_WORD[i]+SOA_COL_STRIDE[i]]=999;
  }
  assert(preview_arena_enter(&c,f,12)); // resumed child call cannot resnapshot
  assert(f[LBASE+9]==1);
  assert(preview_arena_leave(&c,f,10));
  assert(s.used[SOA_POOL_USED[0]]==1&&s.used[SOA_POOL_USED[1]]==1);
  assert(s.views[0]==77&&s.views[1]==SOA_MISSING&&s.views[2]==SOA_MISSING);
  for(int p=0;p<SOA_POOL_COUNT;p++)assert(s.views[SOA_POOL_BASE[p]]==77);
  assert(s.value_cache[SOA_COL_TAG[0]]==77);
  assert(s.value_cache[SOA_COL_TAG[0]+1]==SOA_MISSING);
  for(int i=0;i<SOA_TOTAL_RECORD_FIELDS;i++){
   assert(s.u8[SOA_COL_TAG[i]+SOA_COL_STRIDE[i]]==SOA_ABSENT);
   if(SOA_COL_INTEGER[i]>=0)assert(s.i64[SOA_COL_INTEGER[i]+SOA_COL_STRIDE[i]]==0);
   if(SOA_COL_REAL[i]>=0)assert(s.f64[SOA_COL_REAL[i]+SOA_COL_STRIDE[i]]==0.0);
   if(SOA_COL_WORD[i]>=0)assert(s.used[SOA_COL_WORD[i]+SOA_COL_STRIDE[i]]==0);
   for(int j=0;j<SOA_COL_STRIDE[i];j++){
    assert(s.u8[SOA_COL_TAG[i]+j]==123);
    assert(s.u8[SOA_COL_TAG[i]+SOA_COL_STRIDE[i]+j]==SOA_ABSENT);
    if(SOA_COL_INTEGER[i]>=0)assert(s.i64[SOA_COL_INTEGER[i]+j]==777);
    if(SOA_COL_REAL[i]>=0)assert(s.f64[SOA_COL_REAL[i]+j]==-2.5);
    if(SOA_COL_WORD[i]>=0)assert(s.used[SOA_COL_WORD[i]+j]==444);
   }
  }
  assert(s.used[SOA_POOL_COUNT_COL[0]+1]==0);
  // Reallocate the released position as an EMPTY record/list: an allocator
  // need only publish its new tag/count, not inherit previous optional cells.
  assert(preview_arena_enter(&c,f,10));
  for(int p=0;p<SOA_POOL_COUNT;p++){
   s.used[SOA_POOL_USED[p]]=2;s.u8[SOA_POOL_TAG[p]+1]=p==0?TAG_DICT:5;
   s.used[SOA_POOL_COUNT_COL[p]+1]=0;
  }
  for(int i=0;i<SOA_TOTAL_RECORD_FIELDS;i++)assert(s.u8[SOA_COL_TAG[i]+SOA_COL_STRIDE[i]]==SOA_ABSENT);
  // Complete that private empty allocation before the next synthetic call.
  assert(preview_arena_leave(&c,f,10));
  assert(s.adopt[0]==40&&s.adopt[1]==SOA_MISSING);
  assert(c.h.nodes[160]==TAG_DICT&&c.h.nodes[163]==1); // old VM id preserved
  assert(c.h.nodes[120]==TAG_INT); // escaped score still boxed and alive
 }
 assert(s.clone_epoch==31&&c.steps==@EXPECTED_STEPS@);
 // Exercise all four actual key comparisons (independent fixture bytes),
 // preserving each boxed numeric score after reclamation.
 Context scores{};DomainSoAContext sc{};int fs[160]={};setup(scores,sc,fs);
 scores.h.nodes[42]=4;
 for(int i=0;i<4;i++){scores.h.nodes[(30+i)*4]=i==1?TAG_FLOAT:TAG_INT;scores.h.nodes[(30+i)*4+3]=100+i;}
 assert(preview_arena_enter(&scores,fs,10));sc.used[SOA_POOL_USED[0]]=3;
 assert(preview_arena_leave(&scores,fs,10));
 for(int i=0;i<4;i++)assert(scores.h.nodes[(30+i)*4+3]==100+i);
 // A typed score, even an old one, fails before reclaiming any slots.
 Context bad{};DomainSoAContext b{};int fb[160]={};setup(bad,b,fb);
 assert(preview_arena_enter(&bad,fb,10));b.used[SOA_POOL_USED[0]]=2;
 bad.h.nodes[121]=SOA_VIEW_OFFSET;bad.h.nodes[123]=1;
 assert(!preview_arena_leave(&bad,fb,10));assert(bad.error==BAD_PROGRAM);
 assert(b.used[SOA_POOL_USED[0]]==2);
 // Resuming an unrelated/uninitialized frame cannot approve a return.
 Context uninit{};DomainSoAContext u{};int fu[160]={};setup(uninit,u,fu);
 assert(!preview_arena_enter(&uninit,fu,12));assert(uninit.error==BAD_PROGRAM);
 // Guard failure stays sticky; it does not rewind a partially processed scope.
 Context limited{};DomainSoAContext l{};int fl[160]={};setup(limited,l,fl);
 assert(preview_arena_enter(&limited,fl,10));l.used[SOA_POOL_USED[0]]=2;
 limited.step_limit=limited.steps;
 assert(!preview_arena_leave(&limited,fl,10));assert(limited.error==STEP_LIMIT);
 assert(l.used[SOA_POOL_USED[0]]==2);
 Context timed{};DomainSoAContext t{};int ft[160]={};setup(timed,t,ft);
 timed.deadline=99;assert(!preview_arena_enter(&timed,ft,10));assert(timed.error==DEADLINE);
 // A corrupt allocation counter is rejected before any pool is rewound.
 Context corrupt{};DomainSoAContext q{};int fq[160]={};setup(corrupt,q,fq);
 assert(preview_arena_enter(&corrupt,fq,10));q.used[SOA_POOL_USED[1]]=0;
 assert(!preview_arena_leave(&corrupt,fq,10));assert(corrupt.error==BAD_PROGRAM);
 assert(q.used[SOA_POOL_USED[0]]==1&&q.used[SOA_POOL_USED[1]]==0);
 // Full memo wrap is safe and charged, rather than silently leaving old gids.
 Context wrap{};DomainSoAContext w{};int fw[160]={};setup(wrap,w,fw);
 w.clone_epoch=2147483647;for(int i=0;i<SOA_TOTAL_OBJECTS;i++)w.clone_queue[i]=2147483647;
 assert(preview_arena_enter(&wrap,fw,10));assert(preview_arena_leave(&wrap,fw,10));
 assert(w.clone_epoch==1);for(int i=0;i<SOA_TOTAL_OBJECTS;i++)assert(w.clone_queue[i]==SOA_MISSING);
 std::cout<<"CPU synthetic typed arena checks passed\n";
}
'''


if __name__ == '__main__':
    unittest.main()
