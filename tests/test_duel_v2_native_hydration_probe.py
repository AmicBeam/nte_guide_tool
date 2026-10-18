"""Portable native hydration source binding and compiled read-only guard tests."""
from pathlib import Path
import ctypes as C
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

import numpy as np

from app.modules.card_game.engine.duel_v2.entities import PlayerEntity, CharacterEntity, CardEntity, hydrate_entities
from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
from app.modules.card_game.rl.batched_duel.cuda_vm import _VM_SOURCE
from app.modules.card_game.rl.batched_duel import native_hydration_probe as probe


def fixture():
    a=CardEntity({'card_id':'card_a','instance_id':'card-1'},entity_id='e3',side='a')
    b=CardEntity({'card_id':'card_b','instance_id':'card-2'},entity_id='e6',side='b')
    return {'sides':{
        'a':PlayerEntity({'characters':{'alpha':CharacterEntity({'id':'alpha'},entity_id='e2',side='a')},'hand':[a],'deck':[],'discard':[],'removed':[]},entity_id='e1',side='a'),
        'b':PlayerEntity({'characters':{'beta':CharacterEntity({'id':'beta'},entity_id='e5',side='b')},'hand':[b],'deck':[],'discard':[],'removed':[]},entity_id='e4',side='b')},
        'next_entity':6,'operation':None,'pending_choice':None}


def graph(value):
    seen={}
    def visit(v):
        if isinstance(v,(dict,list,tuple)):
            if id(v) in seen:return ('alias',seen[id(v)])
            n=len(seen);seen[id(v)]=n
            if isinstance(v,dict):return (type(v).__name__,n,tuple((k,visit(x))for k,x in v.items()))
            return(type(v).__name__,n,tuple(visit(x)for x in v))
        return(type(v).__name__,v)
    return visit(value)


def function(source, signature):
    at=source.index(signature);opening=source.index('{',at);depth=1;i=opening+1
    while depth:
        depth+=(source[i]=='{')-(source[i]=='}');i+=1
    return source[at:i]


def shim(schema):
    real_return=function(_VM_SOURCE,'__device__ inline void return_value(')
    return r'''
#define __device__
#define __noinline__ __attribute__((noinline))
const int STRIDE=128,LBASE=14,SBASE=30,MISSING=(-2147483647-1),BAD_CALL=44,BAD_PROGRAM=45,STOP_ITERATION=46,TAG_NONE=0;
struct DomainSoAContext {long long*i64;double*f64;int*i32;unsigned char*u8;int rows,row;int*views,*value_cache,*clone_map,*clone_queue,*constant_adopt;int constant_node_count,clone_epoch;};
struct Heap {int*nodes,*counts;DomainSoAContext*soa;int error,max_nodes;};
struct Context {Heap h;int*frames;int current,output,error,steps,step_limit;long long deadline;int raised;};
unsigned long long host_now;int host_clock_calls;
unsigned long long clock_ns(){++host_clock_calls;return host_now++;}
int*frame(Context*c,int id){return c->frames+id*STRIDE;}
void push(Context*,int*f,int v){f[SBASE+f[2]++]=v;}
int pop(Context*,int*f){return f[SBASE+--f[2]];}
int*mutable_node(Context*c,int r){return c->h.nodes+4*r;}
int kind(Context*c,int r){return c->h.nodes[4*r];}
void reducer_resume(Context*,int,int,int,int,int){}
void stdlib_resume(Context*,int,int,int,int,int){}
void text_join_resume(Context*,int,int,int,int,int){}
void block_push(Context*,int*,int,int,int){}
'''+real_return+probe._helper_source(schema)+r'''
int host_fellthrough;
void native_function_37(Context* c,int* f,const int* fn,int pc){
'''+probe._hook(37,0)+r'''
 ++host_fellthrough;
}
extern "C" void run_host_probe(long long*i64,double*f64,int*i32,unsigned char*u8,int rows,int row,int gid,int steps,int limit,long long deadline,long long now,int tweak,int*out){
 DomainSoAContext s={i64,f64,i32,u8,rows,row,0,0,0,0,0,0,0};
 int nodes[8]={7,MISSING,0,gid,7,MISSING,0,gid},counts[3]={2,0,0},frames[256]={};
 Context c={{nodes,counts,&s,0,2},frames,1,MISSING,0,steps,limit,deadline,MISSING};
 int*f=frame(&c,1);f[0]=37;f[2]=0;f[3]=0;f[4]=1;f[5]=0;f[10]=-1;f[LBASE]=0;
 int fn[1]={100};int pc=100;
 if(tweak==1)pc=101;
 if(tweak==2)f[5]=4;
 if(tweak==3)f[10]=0;
 if(tweak==4)c.error=22;
 if(tweak==5)c.h.error=23;
 if(tweak==6)c.raised=0;
 if(tweak==7)f[3]=-1;
 if(tweak==8)nodes[1]=0;
 if(tweak==9)f[2]=1;
 if(tweak==10)nodes[3]=(-2147483647-1);
 host_now=now;host_clock_calls=0;host_fellthrough=0;
 native_function_37(&c,f,fn,pc);
 out[0]=host_fellthrough==0;out[1]=c.steps-steps;out[2]=c.steps;out[3]=c.current;out[4]=f[4];out[5]=frames[2];out[6]=frames[SBASE];out[7]=c.output;out[8]=c.error;out[9]=c.h.error;out[10]=c.raised;out[11]=host_clock_calls;out[12]=f[2];out[13]=f[LBASE];out[14]=nodes[3];out[15]=counts[0];
}
'''


class NativeHydrationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema=compile_execution_schema();cls.tmp=tempfile.TemporaryDirectory(prefix='native-hydration-')
        cpp=Path(cls.tmp.name)/'probe.cpp';cpp.write_text(shim(cls.schema));so=cpp.with_suffix('.so')
        subprocess.run(['c++','-std=c++14','-O1','-shared','-fPIC',str(cpp),'-o',str(so)],check=True,capture_output=True,timeout=30)
        cls.lib=C.CDLL(str(so));cls.native_run=cls.lib.run_host_probe
        cls.native_run.argtypes=[C.c_void_p]*4+[C.c_int]*5+[C.c_longlong]*2+[C.c_int,C.c_void_p]
        cls.native_run.restype=None
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def check(self,state,hit,*,steps=10,limit=48000000,deadline=0,now=0,tweak=0,mutate=None):
        original=graph(state);batch=pack_game_states([state],self.schema)
        if mutate:mutate(batch)
        before={k:v.tobytes()for k,v in batch.buffers.items()};out=np.zeros(16,np.int32)
        self.native_run(*(batch.buffers[k].ctypes.data for k in('i64','f64','i32','u8')),1,0,int(batch.roots[0]),steps,limit,deadline,now,tweak,out.ctypes.data)
        self.assertEqual(bool(out[0]),hit,out.tolist())
        self.assertEqual(graph(state),original)
        self.assertEqual({k:v.tobytes()for k,v in batch.buffers.items()},before)
        self.assertEqual(out[2],steps+out[1]);self.assertLessEqual(out[2],limit)
        self.assertEqual(out[8],22 if tweak==4 else 0);self.assertEqual(out[9],23 if tweak==5 else 0)
        self.assertEqual(out[10],0 if tweak==6 else -2147483648)
        self.assertEqual(out[15],2);self.assertEqual(out[13],0)
        self.assertEqual(out[14],-2147483648 if tweak==10 else int(batch.roots[0]))
        if hit:
            expected=deepcopy(state);hydrate_entities(expected);self.assertEqual(graph(state),graph(expected))
            self.assertEqual(out[3],-1 if tweak==7 else 0);self.assertEqual(out[4],0)
            self.assertEqual(out[7] if tweak==7 else out[6],0)
        else:self.assertEqual(out[3],1);self.assertEqual(out[4],1)
        return out,batch
    def test_normal_and_native_frame_return(self):
        out,_=self.check(fixture(),True,deadline=100,now=1);self.assertGreater(out[1],0);self.assertGreater(out[11],0)
        self.check(fixture(),True,tweak=7)
    def test_aliases_operation_pending_and_canonical_side_skip(self):
        state=fixture();card=state['sides']['a']['hand'][0]
        state['operation']={'card':card,'side':'b'}
        state['pending_choice']={'cards':[card,card],'side':'b'}
        self.check(state,True)
        # Actual insertion order matters; b-before-a remains canonical.
        state=fixture();state['sides']=dict(reversed(list(state['sides'].items())));self.check(state,True)
    def test_same_logical_different_reference_needs_alias_repair(self):
        state=fixture();card=state['sides']['a']['hand'][0]
        state['operation']={'card':deepcopy(card),'side':'a'};self.check(state,False)
        normalized=deepcopy(state);hydrate_entities(normalized)
        self.assertIs(normalized['operation']['card'],normalized['sides']['a']['hand'][0])
        state['operation']['card']['entity_id']='e7';state['next_entity']=7;self.check(state,False)
        with self.assertRaises(ValueError):hydrate_entities(deepcopy(state))
    def test_duplicate_eid_and_shared_ref_different_logical(self):
        state=fixture();state['sides']['b']['characters']['beta']['entity_id']='e2';self.check(state,False)
        with self.assertRaises(ValueError):hydrate_entities(deepcopy(state))
        state=fixture();chars=state['sides']['a']['characters'];chars['other']=chars['alpha'];self.check(state,False)
        with self.assertRaises(ValueError):hydrate_entities(deepcopy(state))
    def test_missing_empty_nonascii_and_noncanonical_ids(self):
        for value in(None,'','e01','e０１','not-numeric','e9223372036854775808'):
            with self.subTest(value=value):
                state=fixture();state['sides']['a']['hand'][0]['entity_id']=value;self.check(state,False)
        state=fixture();del state['sides']['a']['hand'][0]['entity_id'];self.check(state,False)
        state=fixture();state['sides']['a']['hand'][0]['instance_id']='';self.check(state,False)
        state=fixture();state['sides']['a']['hand'][0]['instance_id']='牌';self.check(state,False)
    def test_counter_class_side_repairs(self):
        for value in(0,):
            state=fixture();state['next_entity']=value;self.check(state,False)
        for tag in(1,2,4,5,6):
            def nonint(b,tag=tag):
                col=next(f for f in self.schema.pools[self.schema.root].fields if f.name=='Game.next_entity');b.buffers['u8'][col.tag,0]=tag
            self.check(fixture(),False,mutate=nonint)
        for cls,key in(('card','hand'),('player','a'),('character','alpha')):
            state=fixture()
            if cls=='card':state['sides']['a']['hand'][0]=dict(state['sides']['a']['hand'][0])
            elif cls=='player':state['sides']['a']=dict(state['sides']['a'])
            else:state['sides']['a']['characters']['alpha']=dict(state['sides']['a']['characters']['alpha'])
            self.check(state,False)
        state=fixture();state['sides']['a']['hand'][0]['side']='b';self.check(state,False)
        state=fixture();del state['sides']['a']['side'];self.check(state,False)
    def test_budget_deadline_and_entry_admission(self):
        self.check(fixture(),False,steps=100,limit=100)
        self.check(fixture(),False,steps=100,limit=163)
        self.check(fixture(),False,steps=2147483600,limit=2147483647)
        out,_=self.check(fixture(),False,limit=100);self.assertGreater(out[1],0)
        out,_=self.check(fixture(),False,deadline=1,now=1);self.assertEqual(out[1],0)
        out,_=self.check(fixture(),False,deadline=2,now=1);self.assertEqual(out[1],64)
        for tweak in(1,2,3,4,5,6,8,9,10):
            with self.subTest(tweak=tweak):self.check(fixture(),False,tweak=tweak)
    def test_every_native_work_budget_cut_and_no_integer_wrap(self):
        full,batch=self.check(fixture(),True)
        work=int(full[1]);before={k:v.tobytes()for k,v in batch.buffers.items()}
        for limit in range(work+2):
            out=np.zeros(16,np.int32)
            self.native_run(*(batch.buffers[k].ctypes.data for k in('i64','f64','i32','u8')),1,0,int(batch.roots[0]),0,limit,0,0,0,out.ctypes.data)
            self.assertEqual(bool(out[0]),limit>=work,(limit,work,out.tolist()))
            self.assertLessEqual(out[2],limit);self.assertEqual(out[8],0);self.assertEqual(out[9],0)
        self.assertEqual({k:v.tobytes()for k,v in batch.buffers.items()},before)
        full,_=self.check(fixture(),True,steps=2147483647-work,limit=2147483647)
        self.assertEqual(full[2],2147483647)
    def test_seen_capacity_and_transient_empty_records(self):
        state=fixture();self.check(state,True)
        state['operation']={};state['pending_choice']={'cards':[]};self.check(state,True)
        state=fixture();deck=[]
        for i in range(94):deck.append(CardEntity({'card_id':'c','instance_id':'extra-'+str(i)},entity_id='e'+str(100+i),side='a'))
        state['sides']['a']['deck']=deck
        state['sides']['b']['deck']=[CardEntity({'card_id':'c','instance_id':'more-'+str(i)},entity_id='e'+str(200+i),side='b')for i in range(40)]
        state['next_entity']=239;self.check(state,False)
    def test_bad_typed_metadata_is_read_only_miss(self):
        def corrupt(b):
            p=self.schema.pools[self.schema.type_ids['Sides']];b.buffers['i32'][p.order+1,0]=0
        self.check(fixture(),False,mutate=corrupt)
        def corrupt_tag(b):
            p=self.schema.pools[self.schema.type_ids['Player']];b.buffers['i32'][p.used,0]=-1
        self.check(fixture(),False,mutate=corrupt_tag)
    def test_field_major_rows_are_independent(self):
        states=[fixture() for _ in range(3)]
        states[1]['sides']['b']['characters']['beta']['entity_id']='e2'
        states[2]['operation']={'card':states[2]['sides']['a']['hand'][0],'side':'a'}
        batch=pack_game_states(states,self.schema)
        before={k:v.tobytes()for k,v in batch.buffers.items()}
        for row,want in enumerate((True,False,True)):
            out=np.zeros(16,np.int32)
            self.native_run(*(batch.buffers[k].ctypes.data for k in('i64','f64','i32','u8')),3,row,int(batch.roots[row]),0,48000000,0,0,0,out.ctypes.data)
            self.assertEqual(bool(out[0]),want,(row,out.tolist()))
        self.assertEqual({k:v.tobytes()for k,v in batch.buffers.items()},before)



from dataclasses import replace
from unittest.mock import patch
from app.modules.card_game.rl.batched_duel.native_compile_units import NativeCompileUnit
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram, compute_identity


def rebound(data):
    data['identity']=compute_identity(data)
    return FrozenRuleProgram.from_dict(data)


def owner_units(program,image):
    from app.modules.card_game.rl.batched_duel.native_array_codegen import native_array_plan
    from app.modules.card_game.rl.batched_duel.native_functions import _emit_function, _admission_guards
    from app.modules.card_game.rl.batched_duel.vm_soa_views import _emit_runtime_head
    fid,_,_=probe._resolve_program(program,image)
    plan=native_array_plan(image)
    fn=next(f for f in plan.coverage['functions']if f['index']==fid)
    rows=plan.coverage['instructions'][fn['start']:fn['start']+fn['count']]
    body=_emit_function(fn,rows,_admission_guards(),complete=True)
    def struct(source,name):
        marker='struct '+name+' {';begin=source.index(marker);opening=source.index('{',begin)
        return source[begin:probe._matching_brace_end(source,opening)]+';\n'
    prefix=struct(_VM_SOURCE,'Context')+struct(_emit_runtime_head(),'DomainSoAContext')
    prefix+='__device__ inline int shared_unchanged(){return 7;}\n'
    return (NativeCompileUnit('runtime',prefix),NativeCompileUnit('rules_123',prefix+body),NativeCompileUnit('rules_124',prefix))


class DisabledHydrationTests(unittest.TestCase):
    def test_disabled_preserves_exact_object_and_never_resolves_inputs(self):
        units=[object()]
        with patch.object(probe,'_resolve_program',side_effect=AssertionError('unexpected parse')):
            result,binding=probe.specialize_hydration_probe(units,object(),object())
        self.assertIs(result,units);self.assertIsNone(binding)
        for bad in(None,0,1,'true'):
            with self.assertRaises(ValueError):probe.specialize_hydration_probe(units,None,None,enabled=bad)
    def test_stable_ast_normalizes_only_empty_new_type_parameters(self):
        import ast
        before=ast.parse('def ordinary(value):\n    return value\n')
        expected=probe._stable_ast(before)
        clone=ast.parse('def ordinary(value):\n    return value\n')
        node=clone.body[0]
        node._fields=tuple(n for n in node._fields if n!='type_params')+('type_params',)
        node.type_params=[]
        self.assertEqual(probe._stable_ast(clone),expected)
        node.type_params=[ast.Name(id='T',ctx=ast.Load())]
        with self.assertRaisesRegex(ValueError,'Generic'):probe._stable_ast(clone)

    def test_configuration_rejects_before_cuda_and_defaults_remain_off(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        import inspect
        for cls in(CudaSoARules,NativeSoACUDAQuery):
            self.assertIs(inspect.signature(cls).parameters['native_hydration_probe'].default,False)
            self.assertIsNone(inspect.signature(cls).parameters['hydration_program'].default)
        for opts in(dict(native_hydration_probe=1),dict(native_hydration_probe=True),
                     dict(native_hydration_probe=True,native_functions=True,separate_compilation=True)):
            with self.assertRaises(ValueError):CudaSoARules(None,None,**opts)


    def test_opt_in_rejects_execute_only_before_cuda_or_metadata_allocation(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        with patch.object(probe,'validate_hydration_probe_inputs',side_effect=AssertionError('no metadata allocation')):
            for execute_only in(True,None,0):
                with self.subTest(execute_only=execute_only),self.assertRaisesRegex(ValueError,'full module'):
                    CudaSoARules(None,None,native_hydration_probe=True,native_functions=True,
                        separate_compilation=True,hydration_program=object(),execute_only=execute_only,stack_bytes=65536)

    def test_query_forwards_explicit_program_and_binding_without_claiming_qualification(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_duel import native_soa_cuda_query as query
        image=SimpleNamespace(**{n:np.zeros(1,dtype=np.int32)for n in('nodes','edges','payload')})
        program=object();binding=object()
        executor=SimpleNamespace(source_sha256='emitter-only',hydration_probe_binding=binding,
            canonical_rule_source_sha256='canonical-rule',rule_source_sha256='canonical-rule',
            preview_arena_binding=None,preview_arena_identity=None)
        with patch.object(query,'CudaSoARules',return_value=executor)as ctor:
            result=query.NativeSoACUDAQuery(image,object(),native_functions=True,separate_compilation=True,
                native_hydration_probe=True,hydration_program=program)
        self.assertIs(ctor.call_args.kwargs['hydration_program'],program)
        self.assertIs(ctor.call_args.kwargs['native_hydration_probe'],True)
        self.assertIs(result.hydration_probe_binding,binding)
        self.assertFalse(result.gpu_throughput_qualified)


@unittest.skipUnless(sys.version_info[:2]==(3,10),'CPython3.10 portable capture contract')
class HydrationBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.modules.card_game.rl.rule_ir.portable_program import capture_program
        cls.program=capture_program({'hydrate':hydrate_entities})
        cls.image=lower_program(cls.program);cls.schema=compile_execution_schema()
        cls.units=owner_units(cls.program,cls.image)
    def transform(self,program=None,image=None,units=None,schema=None):
        return probe.specialize_hydration_probe(self.units if units is None else units,
            self.image if image is None else image,self.schema if schema is None else schema,
            program=self.program if program is None else program,enabled=True)
    def test_dynamic_function_owner_full_fallback_restore_and_source_binding(self):
        result,binding=self.transform()
        self.assertEqual(binding.owner_unit,'rules_123')
        self.assertEqual(self.image.function_ids[binding.function_index],binding.function_identity)
        self.assertIs(result[0],self.units[0]);self.assertIs(result[2],self.units[2])
        entry='__device__ __noinline__ void native_function_'+str(binding.function_index)+'(Context* c,int* f,const int* fn,int pc){'
        restored=result[1].source.replace(probe._helper_source(self.schema),'',1).replace(entry+probe._hook(binding.function_index,binding.state_local),entry,1)
        self.assertEqual(restored,self.units[1].source)
        self.assertEqual(binding.original_units,tuple((u.name,u.sha256)for u in self.units))
        self.assertEqual(binding.transformed_units,tuple((u.name,u.sha256)for u in result))
        self.assertNotEqual(binding.original_units,binding.transformed_units)
        self.assertEqual(binding.recipe_sha256,__import__('hashlib').sha256(Path(probe.__file__).read_bytes()).hexdigest())
        self.assertFalse(binding.gpu_qualified)
        with self.assertRaises(ValueError):self.transform(units=result)
    def test_capture_order_changes_function_id_without_changing_algorithm(self):
        data=self.program.to_dict();data['functions'].reverse();program=rebound(data);image=lower_program(program)
        fid,_,_=probe._resolve_program(program,image)
        self.assertNotEqual(fid,probe._resolve_program(self.program,self.image)[0])
        units=owner_units(program,image);_,binding=self.transform(program,image,units)
        self.assertEqual(binding.function_index,fid)
    def test_changed_program_image_ir_globals_defaults_and_classes_rejected(self):
        mutators=(
            lambda d,f:f['instructions'][0].update(arg=1),
            lambda d,f:f['globals'].update(PlayerEntity=f['globals']['CharacterEntity']),
            lambda d,f:f['globals'].update(isinstance={'kind':'intrinsic','name':'int'}),
            lambda d,f:f['globals'].update(allocate_entity_id=f['globals']['isinstance']),
            lambda d,f:f.update(defaults=[{'kind':'singleton','name':'None'}]),
            lambda d,f:f.update(flags=f['flags']|32),
            lambda d,f:next(c for c in d['classes']if c['module']==probe._MODULE and c['qualname']=='PlayerEntity')['methods'].update(__setitem__={'kind':'function','id':f['id']}),
            lambda d,f:next(c for c in d['classes']if c['module']==probe._MODULE and c['qualname']=='PlayerEntity')['attributes'].update(id={'kind':'singleton','name':'None'}),
        )
        for mutate in mutators:
            data=self.program.to_dict();fn=next(f for f in data['functions']if f['module']==probe._MODULE and f['qualname']=='hydrate_entities')
            mutate(data,fn);program=rebound(data);image=lower_program(program)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.transform(program,image)
        data=self.program.to_dict();data['source_hashes']['unused']='0'*64
        with self.assertRaises(ValueError):self.transform(rebound(data),self.image)
        with patch.object(probe,'_canonical_ast',return_value='changed'):
            with self.assertRaises(ValueError):self.transform()
        with self.assertRaises(ValueError):probe.specialize_hydration_probe(self.units,self.image,self.schema,enabled=True)
    def test_canonical_looking_reference_id_must_match_actual_descriptor(self):
        data=self.program.to_dict()
        original=next(f for f in data['functions']if f['module']==probe._MODULE and f['qualname']=='Entity.entity_id')
        forged=deepcopy(original)
        forged['id']='func:'+probe._MODULE+':Entity.entity_id:9999:9999'
        forged['name']=forged['qualname']='other_getter'
        side=next(c for c in data['constants']if c.get('kind')=='literal'and c.get('value')=='side')
        forged['constants'][1]={'kind':'constant','id':side['id']}
        data['functions'].append(forged)
        entity=next(c for c in data['classes']if c['module']==probe._MODULE and c['qualname']=='Entity')
        entity['properties']['entity_id']['fget']['id']=forged['id']
        program=rebound(data)
        with self.assertRaisesRegex(ValueError,'reference/descriptor'):self.transform(program,lower_program(program))
        data=self.program.to_dict();duplicate=deepcopy(original)
        duplicate['id']='func:'+probe._MODULE+':Entity.entity_id:8888:8888'
        data['functions'].append(duplicate)
        entity=next(c for c in data['classes']if c['module']==probe._MODULE and c['qualname']=='Entity')
        entity['properties']['entity_id']['fget']['id']=duplicate['id']
        program=rebound(data)
        with self.assertRaisesRegex(ValueError,'Entity.entity_id'):self.transform(program,lower_program(program))

    def test_executed_constant_zero_is_bound_but_unused_docstring_is_not(self):
        data=self.program.to_dict()
        fn=next(f for f in data['functions']if f['module']==probe._MODULE and f['qualname']=='Entity.__init__')
        self.assertTrue(any(i['op']=='LOAD_CONST'and i['arg']==0 for i in fn['instructions']))
        integer=next(c for c in data['constants']if c.get('kind')=='literal'and c.get('type')=='int'and c.get('value')==1)
        fn['constants'][0]={'kind':'constant','id':integer['id']}
        program=rebound(data);image=lower_program(program)
        with self.assertRaisesRegex(ValueError,'Entity.__init__'):self.transform(program,image)
        data=self.program.to_dict()
        fn=next(f for f in data['functions']if f['module']==probe._MODULE and f['qualname']=='hydrate_entities')
        self.assertFalse(any(i['op']=='LOAD_CONST'and i['arg']==0 for i in fn['instructions']))
        string=next(c for c in data['constants']if c.get('kind')=='literal'and c.get('type')=='str'and c.get('value')=='sides')
        fn['constants'][0]={'kind':'constant','id':string['id']}
        program=rebound(data)
        probe._resolve_program(program,lower_program(program))

    def test_missing_duplicate_source_unknown_frame_and_partial_body_rejected(self):
        for units in(self.units[:1],self.units+(self.units[1],),
            (self.units[0],replace(self.units[1],source=self.units[1].source.replace('const int owner=c->current;','const int owner=-1;')),self.units[2]),
            (self.units[0],replace(self.units[1],source=self.units[1].source.replace('struct Context {','struct UnknownContext {')),self.units[2])):
            with self.assertRaises(ValueError):self.transform(units=units)
        fid,_,_=probe._resolve_program(self.program,self.image);start=int(self.image.functions[fid,0])
        partial=replace(self.units[1],source=self.units[1].source.replace('op_'+str(start)+':{','removed_label:{',1))
        with self.assertRaises(ValueError):self.transform(units=(self.units[0],partial,self.units[2]))
    def test_schema_types_references_order_bounds_and_stale_offsets_rejected(self):
        from app.modules.card_game.rl.batched_duel.record_soa import CompiledRecordSchema
        original=self.schema
        data=json.loads(json.dumps(dict(original.definitions)))
        data['Player']['fields']['side']='int'
        wrong=CompiledRecordSchema(data)
        with self.assertRaises(ValueError):self.transform(schema=wrong)
        data=json.loads(json.dumps(dict(original.definitions)));data['CardList']['value']='@Character'
        with self.assertRaises(ValueError):self.transform(schema=CompiledRecordSchema(data))
        data=json.loads(json.dumps(dict(original.definitions)));data['Sides']['fields']={'b':'@Player','a':'@Player'}
        with self.assertRaises(ValueError):self.transform(schema=CompiledRecordSchema(data))
        data=json.loads(json.dumps(dict(original.definitions)));data['CardList']['max_items']=257
        with self.assertRaises(ValueError):self.transform(schema=CompiledRecordSchema(data))
        wrong=compile_execution_schema();pool=wrong.pools[wrong.type_ids['Game']]
        column=pool.fields[0];changed=replace(pool,fields=(replace(column,tag=column.tag+1),)+pool.fields[1:])
        wrong.pools=tuple(changed if p.name=='Game'else p for p in wrong.pools)
        with self.assertRaises(ValueError):self.transform(schema=wrong)
    def test_schema_invalid_generated_identifier_and_int32_metadata_rejected(self):
        from app.modules.card_game.rl.batched_duel.record_soa import CompiledRecordSchema
        data=json.loads(json.dumps(dict(self.schema.definitions)))
        data['Game']['fields']['bad-field']='int'
        with self.assertRaisesRegex(ValueError,'identifier'):self.transform(schema=CompiledRecordSchema(data))
        wrong=compile_execution_schema();pool=wrong.pools[wrong.type_ids['Game']]
        col=pool.fields[0];changed=replace(pool,fields=(replace(col,tag=2**31),)+pool.fields[1:])
        wrong.pools=tuple(changed if p.name=='Game'else p for p in wrong.pools)
        with self.assertRaises(ValueError):self.transform(schema=wrong)

    def test_recognized_full_pipeline_pool_diagnostic_member(self):
        owner=self.units[1]
        full=replace(owner,source=owner.source.replace(' int constant_node_count;',
            ' int constant_node_count; int failed_pool;',1))
        _,binding=self.transform(units=(self.units[0],full,self.units[2]))
        self.assertEqual(binding.owner_unit,owner.name)
        unknown=replace(owner,source=owner.source.replace(' int constant_node_count;',
            ' int constant_node_count; int unknown_member;',1))
        with self.assertRaises(ValueError):self.transform(units=(self.units[0],unknown,self.units[2]))
    def test_address_capacity_changes_get_distinct_actual_binding(self):
        schema=compile_execution_schema(pool_scale=2)
        result,binding=self.transform(schema=schema)
        baseline,original=self.transform()
        self.assertNotEqual(result[1].sha256,baseline[1].sha256)
        self.assertNotEqual(binding.identity,original.identity)


if __name__=='__main__':unittest.main()
