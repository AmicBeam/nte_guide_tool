"""Actual CUDA execution of source bytecode; not a full-game capacity test."""
from pathlib import Path
import subprocess
import os,json,hashlib
import sys
import unittest
from app.modules.card_game.rl.rule_ir.portable_program import capture_program,FrozenRuleProgram
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,VmLimits,emit_vm_source,SUPPORTED_OPS,VmCapabilityError
from app.modules.card_game.rl.batched_duel.codec import pack_states
from app.modules.card_game.rl.batched_duel.schema import GraphLayout


def arithmetic(value,offset=2):
    if value<0:return -value
    return value+offset


def unsupported(value):return value**2


def collections_probe(value):
    doubled=[item*2 for item in value]
    a,b=(7,8)
    mapping={**{'a':1},'a':a,'b':b}
    if doubled:
        doubled[-1]=33
        del doubled[0]
    members={1,1,2}
    return {'list':doubled,'tuple':(a,b),'slice':value[1:-1:2],
            'reverse':value[::-1],'text':'真红🙂x'[::-1],
            'map':mapping,'comprehension':list({item:item*2 for item in value}.items()),
            'members':(2 in members,3 in members),'alias':[value,value]}


def entity_echo(value):
    value['unknown']['trace'][0]=9
    return {'entities':[value,value],'unknown':value['unknown']}


def signature_target(a,/,b=2,*items,k=3,**kw):
    return {'a':a,'b':b,'items':items,'k':k,'kw':kw}


def signature_probe(value):
    def runtime(*,scale=value):return scale
    return (signature_target(1,4,8,9,k=5,tail=7),
            signature_target(2,k=4,**{'tail':6}),
            signature_target(*(3,5,8),**{'k':9,'tail':4}),runtime(),runtime(scale=13))


class AttributeBox:
    factor=3
    def __init__(self,bias):self.bias=bias
    def run(self,value):return value+self.bias
    @property
    def scaled(self):return self.bias*2
    @classmethod
    def tagvalue(cls):return cls.factor
    @staticmethod
    def twice(value):return value*2


ATTRIBUTE_BOX=AttributeBox(7)


def attribute_probe(value):
    return ATTRIBUTE_BOX.run(value)+ATTRIBUTE_BOX.twice(value)+ATTRIBUTE_BOX.tagvalue()+ATTRIBUTE_BOX.scaled


def import_probe(value):
    from app.modules.card_game.engine.duel_v2.state import other
    return other(value)


def constructor_probe(value):
    box=AttributeBox(value)
    box.bias=box.bias+1
    return box.run(3),box.scaled,box.tagvalue()


def numeric_probe(value):
    a,b=value
    changed=a
    changed+=b
    changed-=b
    changed*=b
    return a+b,a-b,a*b,a//b,a/b,a<b,a>=b,-a,changed,int(a)&7,True+2,True<2.0


def precision_compare(value):
    a,b=value
    return a<b,a>b,a==b,a<=b,a>=b


def exact_equality(value):return value[0]==value[1]


def reducer_probe(value):
    generator=(item for item in value)
    found=any(generator)
    after=next(generator,None)
    return (all(item>0 for item in value),found,after,
        sum((item*.5 for item in value),start=1.25),min((item for item in value),default=-3),
        max((item for item in value),default=7),min(3,7,-1),max(3,7,-1),sum(range(4)),all([]),any([]))


def generator_collection_probe(count):
    kept=(item for item in (0,5,9))
    first=any(kept)
    total=0
    for i in range(count):
        if any(item for item in (1,2,3)):total+=1
        total+=sum(item for item in (1,2))
    return total,first,next(kept,None),next(kept,None)


def random_probe(seed):
    from random import Random
    rng=Random(seed)
    items=[{'id':'a'},{'id':'b'},{'id':'c'},{'id':'d'}]
    rng.shuffle(items)
    chosen=rng.choice(items)
    bits=rng.getrandbits(64)
    zero=rng.getrandbits(0)
    numbers=(rng.randrange(5),rng.randrange(-11,15,3),rng.randrange(8,-12,-3),rng.randint(-100,100),rng.random(),rng.getrandbits(63))
    rng.seed(seed)
    repeated=rng.getrandbits(64)
    return items,chosen,bits,zero,numbers,repeated


def equality_probe(value):
    first={'k':[value,1.0]};second={'k':[value,1]}
    items=[value];alias=items;items+=[3]
    nested=[first,second];nested.remove({'k':[value,1]})
    text='真';text+='红'
    pair=('a',);pair+=('b',)
    return (first==second,first['k'] in [second['k']],items is alias,items,nested,text,pair,
        '红' in '真红队','' in '', '异' not in '真红队','a' in {'a','b'})


def native_probe(value):
    mapping=dict(value)
    mapping.setdefault('added',7)
    popped=mapping.pop('missing',13)
    mapping.update({'after':9})
    copied=mapping.copy()
    items=list(mapping.items())
    values=list(mapping.values())
    keys=list(mapping.keys())
    sequence=[1,2,3]
    sequence.append(4)
    sequence.extend(sequence)
    first=sequence.pop(0)
    last=sequence.pop()
    sequence.insert(-1,8)
    sequence.remove(2)
    sequence.reverse()
    clone=sequence.copy()
    return (mapping,copied,items,values,keys,popped,first,last,clone,
            len(sequence),sequence.count(3),sequence.index(8),
            [i for i in range(4,-8,-3)],list(range(-9223372036854775808,9223372036854775807,9223372036854775807)),
            bool(range(0)),bool(range(1)),len('真红🙂'),int(' -32 '),float(3),int(-3.75))


def test_program(stem,entry,function,*,own=False):
    if sys.version_info[:2]==(3,10):
        kwargs=dict(scope_character_ids=None,scope_card_ids=None)
        if own:
            from app.modules.card_game.rl.rule_ir.portable_program import DEFAULT_ALLOWED_MODULE_PREFIXES
            kwargs['allowed_module_prefixes']=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,)
        return capture_program({entry:function},**kwargs)
    directory=os.environ.get('NTE_VM_TEST_PROGRAM_DIR')
    if not directory:raise RuntimeError('CPython3.13 consumes explicit 3.10 frozen fixtures; no native recapture')
    fixture=Path(directory)/('all.json' if os.environ.get('NTE_VM_COMBINED_PROGRAM')=='1' else stem+'.json')
    program=FrozenRuleProgram.from_json(fixture.read_text())
    program.validate();root=Path(__file__).resolve().parents[1]
    from app.modules.card_game.rl.offline_sources import _allowed_source
    for name,digest in program.source_hashes.items():
        path=root/name
        if not _allowed_source(name) or path.is_symlink() or root not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen test source mismatch: '+name)
    expected=set(ALL_VM_ENTRIES) if os.environ.get('NTE_VM_COMBINED_PROGRAM')=='1' else {entry}
    if set(program.entries)!=expected or entry not in program.entries:raise ValueError('Frozen test entry mismatch')
    return program


ALL_VM_ENTRIES=('add','u','actor','collections','echo','signature','attributes','import','native','construct','numeric','precision','equality','hero','alive','order','exact','reducers','frame_gc','random')


class SourceTest(unittest.TestCase):
    def test_import_does_not_load_torch_and_source_has_no_cpu_runtime(self):
        code="import sys;from app.modules.card_game.rl.batched_duel import cuda_vm;assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
        p=test_program('arithmetic','add',arithmetic,own=True)
        image=lower_program(p);source=emit_vm_source(image)
        self.assertIn('extern "C" __global__ void vm_execute',source)
        self.assertIn('globaltimer',source);self.assertFalse(CudaVm.supports_full_duel_rules)
        self.assertIn('case OP_CALL_FUNCTION',source)

    def test_limits_reject_unbounded_or_bool_values(self):
        for field in ('frames','locals','stack','cells','steps'):
            with self.assertRaises(ValueError):VmLimits(**{field:True})
            with self.assertRaises(ValueError):VmLimits(**{field:0})


class CudaExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():raise unittest.SkipTest('Actual CUDA required')
        cls._executors={}

    @classmethod
    def tearDownClass(cls):
        for vm in cls._executors.values():vm.close()

    def executor(self,image):
        from contextlib import nullcontext
        if image.identity not in self._executors:self._executors[image.identity]=CudaVm(image)
        return nullcontext(self._executors[image.identity])

    def test_actual_cuda_integer_control_flow_defaults_and_capacity(self):
        p=test_program('arithmetic','add',arithmetic,own=True)
        image=lower_program(p)
        values=[i-800 for i in range(1600)];packed=pack_states(values,GraphLayout(1600,64,64,1024))
        with self.executor(image) as vm:
            arena=vm.ingest(packed);before=arena.nodes.clone();call=vm.launch(arena,'add')
            self.assertEqual(call.decode(),[arithmetic(v) for v in values])
            self.assertTrue(vm.torch.equal(before,arena.nodes));self.assertEqual(call.check().shape,(1600,4))

    def test_formal_actor_all_phases_closure_generator_and_no_source_calls(self):
        from app.modules.card_game.engine.duel_v2 import flow,new_game
        from app.modules.card_game.rl.rule_ir.portable_program import DEFAULT_ALLOWED_MODULE_PREFIXES
        from unittest.mock import patch
        p=test_program('actor','actor',flow.acting_side)
        image=lower_program(p);states=[]
        for i in range(10):
            s=new_game(seed=33300+i,first_side='a' if i%2 else 'b',skip_mulligan=bool(i%2))
            if not i%2:s['mulligan_done']=['a'] if i%4 else []
            states.append(s)
        finished=new_game(seed=33320,skip_mulligan=True);finished['phase']='finished';states.append(finished)
        choice=new_game(seed=33321,skip_mulligan=True);choice['phase']='choice';choice['pending_choice']={'side':'b'};states.append(choice)
        expected=[flow.acting_side(s) for s in states]
        packed=pack_states(states,GraphLayout(len(states),8192,16384,131072))
        with self.executor(image) as vm:
            self.assertTrue(vm.capability('actor')['ready'])
            arena=vm.ingest(packed)
            with patch.object(flow,'acting_side',side_effect=AssertionError('CPU rule fallback')):
                self.assertEqual(vm.launch(arena,'actor').decode(),expected)

    def test_missing_entry_opcodes_fail_before_device_execution(self):
        p=test_program('unsupported','u',unsupported,own=True)
        with self.executor(lower_program(p)) as vm:
            self.assertFalse(vm.capability('u')['ready'])
            arena=vm.ingest(pack_states([[1,2]],GraphLayout(1,64,64,1024)))
            with self.assertRaises(VmCapabilityError):vm.launch(arena,'u')

    def test_collections_unicode_slices_mutation_comprehensions_and_aliases(self):
        p=test_program('collections','collections',collections_probe,own=True)
        values=[[],[1],[1,2,3,4,5],[-3,-2,0,7]]
        packed=pack_states(values,GraphLayout(4,4096,4096,32768))
        with self.executor(lower_program(p)) as vm:
            self.assertTrue(vm.capability('collections')['ready'])
            results=vm.launch(vm.ingest(packed),'collections').decode()
        self.assertEqual(results,[collections_probe(v) for v in values])
        for result in results:self.assertIs(result['alias'][0],result['alias'][1])

    def test_structured_device_export_preserves_entities_unknown_fields_and_input(self):
        from app.modules.card_game.engine.duel_v2.entities import CardEntity
        p=test_program('entity_echo','echo',entity_echo,own=True)
        value=CardEntity(card_id='N01',entity_id='card_17',side='a',unknown={'trace':[1,2],'future':True})
        with self.executor(lower_program(p)) as vm:
            arena=vm.ingest(pack_states([value],GraphLayout(1,512,512,4096)))
            before=arena.nodes.clone();result=vm.launch(arena,'echo').decode()[0]
            self.assertTrue(vm.torch.equal(before,arena.nodes))
        self.assertIsInstance(result['entities'][0],CardEntity)
        self.assertIs(result['entities'][0],result['entities'][1])
        self.assertIs(result['unknown'],result['entities'][0]['unknown'])
        self.assertEqual(result['unknown']['trace'],[9,2]);self.assertEqual(value['unknown']['trace'],[1,2])

    def test_keywords_varargs_kwonly_and_runtime_kwdefaults(self):
        p=test_program('signature','signature',signature_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            self.assertTrue(vm.capability('signature')['ready'])
            result=vm.launch(vm.ingest(pack_states([11],GraphLayout(1,4096,4096,32768))),'signature').decode()
        self.assertEqual(result,[signature_probe(11)])

    def test_frozen_instance_method_classmethod_staticmethod_and_property(self):
        p=test_program('attributes','attributes',attribute_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            self.assertTrue(vm.capability('attributes')['ready'])
            result=vm.launch(vm.ingest(pack_states([4,9],GraphLayout(2,2048,2048,16384))),'attributes').decode()
        self.assertEqual(result,[attribute_probe(4),attribute_probe(9)])

    def test_frozen_import_calls_formal_rule_on_device(self):
        from app.modules.card_game.engine.duel_v2 import state
        from unittest.mock import patch
        p=test_program('import','import',import_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            self.assertTrue(vm.capability('import')['ready'])
            arena=vm.ingest(pack_states(['a','b'],GraphLayout(2,512,512,4096)))
            with patch.object(state,'other',side_effect=AssertionError('CPU fallback')):
                self.assertEqual(vm.launch(arena,'import').decode(),['b','a'])

    def test_instance_construction_initialization_and_mutable_attributes(self):
        p=test_program('construct','construct',constructor_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            self.assertTrue(vm.capability('construct')['ready'])
            actual=vm.launch(vm.ingest(pack_states([2,5],GraphLayout(2,2048,2048,16384))),'construct').decode()
        self.assertEqual(actual,[constructor_probe(2),constructor_probe(5)])

    def test_native_mapping_list_unicode_and_signed_extreme_range_semantics(self):
        p=test_program('native','native',native_probe,own=True)
        values=[{'existing':4},{}]
        with self.executor(lower_program(p)) as vm:
            self.assertTrue(vm.capability('native')['ready'])
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(2,8192,8192,65536))),'native').decode()
        self.assertEqual(actual,[native_probe(v) for v in values])

    def test_numeric_types_floor_sign_and_precise_mixed_comparisons(self):
        import struct
        values=[[-11,3],[11,-3],[-11,-3],[1.0,.1],[-1.0,.1],[3.3,.1],[0.0,-3.0],[-0.0,3.0],[3.25,.5]]
        p=test_program('numeric','numeric',numeric_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(len(values),2048,2048,16384))),'numeric').decode()
        expected=[numeric_probe(v) for v in values]
        self.assertEqual(actual,expected)
        for row,oracle in zip(actual,expected):
            if isinstance(oracle[3],float):self.assertEqual(struct.pack('<d',row[3]),struct.pack('<d',oracle[3]))
        values=[[2**53+1,float(2**53)],[2**63-1,float(2**63)],[-2**63,float(-2**63)],[-2**63,-float(2**63)-2048.]]
        p=test_program('precision','precision',precision_compare,own=True)
        with self.executor(lower_program(p)) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(4,2048,2048,16384))),'precision').decode()
        self.assertEqual(actual,[precision_compare(v) for v in values])

    def test_recursive_equality_contains_and_inplace_alias_semantics(self):
        p=test_program('equality','equality',equality_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            actual=vm.launch(vm.ingest(pack_states([2,5.0],GraphLayout(2,4096,4096,32768))),'equality').decode()
        self.assertEqual(actual,[equality_probe(2),equality_probe(5.0)])

    def test_exact_integer_float_equality_beyond_int64(self):
        values=[[2**80,float(2**80)],[2**80+1,float(2**80)],[-2**80,-float(2**80)],[-2**80+1,-float(2**80)],
            [float(2**63),2**63-1],[float(-2**63),-2**63],[True,1],[False,0]]
        p=test_program('exact','exact',exact_equality,own=True)
        with self.executor(lower_program(p)) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(len(values),2048,2048,16384))),'exact').decode()
        self.assertEqual(actual,[exact_equality(v) for v in values])

    def test_reducers_suspend_resume_shortcircuit_and_empty_defaults(self):
        p=test_program('reducers','reducers',reducer_probe,own=True)
        values=[[],[0,1,2],[-3,0,5],[1,2,3]]
        with self.executor(lower_program(p)) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(4,8192,16384,65536))),'reducers').decode()
        self.assertEqual(actual,[reducer_probe(v) for v in values])

    def test_abandoned_generators_reclaimed_without_losing_live_generator(self):
        p=test_program('frame_gc','frame_gc',generator_collection_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            actual=vm.launch(vm.ingest(pack_states([100],GraphLayout(1,16384,32768,131072))),'frame_gc').decode()
        self.assertEqual(actual,[generator_collection_probe(100)])

    def test_random_seed_mutation_shuffle_bits_and_odd_row_alignment(self):
        import random
        from unittest.mock import patch
        seeds=[-2**63,-1,0,1,2**63-1]
        expected=[random_probe(seed) for seed in seeds]
        p=test_program('random','random',random_probe,own=True)
        with self.executor(lower_program(p)) as vm:
            arena=vm.ingest(pack_states(seeds,GraphLayout(5,4096,8192,32769)))
            with patch.object(random,'Random',side_effect=AssertionError('CPU Random fallback')):
                actual=vm.launch(arena,'random').decode()
        self.assertEqual(actual,expected)
        for result in actual:self.assertTrue(any(result[1] is item for item in result[0]))

    def test_formal_state_helpers_run_from_device_arguments(self):
        from app.modules.card_game.engine.duel_v2 import state,new_game
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS,validate_deck
        from copy import deepcopy
        rows=[]
        for deck in STARTER_DECKS:
            if deck['id'] not in ('starter','weave-rush','quick-rush','zhenhong','murk'):continue
            game=new_game(seed=7200+len(rows),skip_mulligan=True,decks={'a':validate_deck(deepcopy(deck)),'b':validate_deck(deepcopy(deck))})
            for side in ('a','b'):
                cid=next(iter(game['sides'][side]['characters']));game['sides'][side]['characters'][cid]['hp']=3.25
                if side=='b':game['sides'][side].pop('order',None)
                rows.append([game,side,cid])
        packed=pack_states(rows,GraphLayout(len(rows),8192,16384,131072))
        for entry,fn,width in (('hero',state.hero,3),('alive',state.alive,3),('order',state.team_order,2)):
            expected=[fn(*row[:width]) for row in rows]
            p=test_program(entry,entry,fn)
            with self.executor(lower_program(p)) as vm:
                arena=vm.ingest(packed);torch=vm.torch
                positions=arena.nodes[torch.arange(len(rows),device=vm.device),arena.roots.long(),1].long()
                indices=positions[:,None]+torch.arange(width,device=vm.device)[None,:]
                args=arena.edges[torch.arange(len(rows),device=vm.device)[:,None],indices,0].contiguous()
                with patch.object(state,fn.__name__,side_effect=AssertionError('CPU rule fallback')):
                    actual=vm.launch(arena,entry,argument_refs=args).decode()
            self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
