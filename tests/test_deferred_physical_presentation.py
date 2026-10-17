"""CPU contracts and typed transport only; no actual CUDA/rule qualification."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch as t

from app.modules.card_game.rl.batched_search import deferred_physical_presentation as contract
from app.modules.card_game.rl.batched_search.device_games import DeviceGameLoop
from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules
from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import DeviceTiledSoARules
from app.modules.card_game.rl.batched_search.device_tiered_soa_rules import DeviceTieredSoARules
from app.modules.card_game.rl.batched_search.device_raw_action_tape import DeviceRawActionTape
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceDecision,DeviceStateBatch
from app.modules.card_game.rl.batched_duel.vm_image import VmImage,_image_identity


def numeric_image():
    names=tuple('func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:seven_'+entry+':1:0'
        for entry in ('apply_action','simulate_action','clean'))
    ints=lambda width:np.zeros((0,width),dtype='<i4')
    image=VmImage('synthetic-test-only','',ints(4),ints(18),ints(6),ints(3),ints(4),ints(2),
        np.zeros(0,dtype=np.uint8),(),(),names,(),(),dict(apply_action=0,simulate_action=1,clean=2),{})
    return replace(image,identity=_image_identity(image))


class CrossVersionDeferredAstTests(unittest.TestCase):
    def tree(self):
        import ast
        return ast.parse('class A:\n def f(self,x=1):\n  return x\n')
    def add_type_params(self,tree,value):
        import ast
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                node._fields=tuple(name for name in node._fields if name!='type_params')+('type_params',)
                node.type_params=list(value)
        return tree
    def test_empty_new_fields_match_legacy_without_mutating_ast_classes(self):
        import ast,copy
        original=self.tree();expected=contract._contract_ast_dump(copy.deepcopy(original))
        class_fields={cls:cls._fields for cls in (ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)}
        modern=self.add_type_params(copy.deepcopy(original),[])
        self.assertEqual(contract._contract_ast_dump(modern),expected)
        for cls,fields in class_fields.items():self.assertIs(cls._fields,fields)
        self.assertEqual(contract._contract_ast_dump(modern),expected) # idempotent
        modern.body[0].type_params=[ast.Name(id='T',ctx=ast.Load())]
        with self.assertRaises(ValueError):contract._contract_ast_dump(modern)
    def test_nonempty_or_unknown_field_value_fails_closed(self):
        import ast
        with self.assertRaisesRegex(ValueError,'nonempty type_params'):
            contract._contract_ast_dump(self.add_type_params(self.tree(),[ast.Name(id='T',ctx=ast.Load())]))
        tree=self.add_type_params(self.tree(),[]);tree.body[0].type_params=None
        with self.assertRaises(ValueError):contract._contract_ast_dump(tree)
    def test_modern_dump_is_told_to_preserve_all_other_empty_fields(self):
        import ast
        original=ast.dump;seen=[]
        def modern(node,annotate_fields=True,include_attributes=False,*,indent=None,show_empty=False):
            seen.append(show_empty)
            if not show_empty:raise AssertionError('Other empty fields must remain checked')
            return original(node,annotate_fields=annotate_fields,include_attributes=include_attributes,indent=indent)
        with patch.object(contract.ast,'dump',modern):
            contract._contract_ast_dump(self.tree())
        self.assertEqual(seen,[True])
    def test_other_ast_changes_are_not_normalized_and_existing14_hashes_stay(self):
        import ast,copy
        tree=self.add_type_params(self.tree(),[]);other=copy.deepcopy(tree)
        other.body[0].body[0].body[0].value=ast.Constant(value=2)
        self.assertNotEqual(contract._contract_ast_dump(tree),contract._contract_ast_dump(other))
        self.assertEqual(contract._current_contract(),contract._AST)


class DeferredSourceContractTests(unittest.TestCase):
    def setUp(self):t.set_num_threads(1)
    def rules(self):
        rules=object.__new__(DeviceSoARules);rules.rows=1;rules.device=t.device('cpu')
        rules.raw_tape=DeviceRawActionTape(1,2,32,32,256,device='cpu',allow_cpu_reference=True)
        self.addCleanup(rules.raw_tape.close)
        rules.executor=NS(image=numeric_image(),closed=False,execute_only=False);rules.physical_loop=None
        return rules
    def admit_synthetic_image(self,rules):
        # Test-only whitelist patch; production has only independently frozen images.
        context=patch.object(contract,'_IMAGES',frozenset(((rules.executor.image.program_identity,rules.executor.image.identity),)))
        context.start();self.addCleanup(context.stop)
    def bound_loop(self,rules):
        loop=object.__new__(DeviceGameLoop)
        loop.rules=rules;loop.rows=rules.rows;loop.device=rules.device
        loop.defer_physical_presentation=True
        loop._deferred_physical_contract=contract.validate_deferred_rules(rules)
        loop._deferred_raw_tape=rules.raw_tape;loop._deferred_step_open=False
        return loop
    def test_real_source_shapes_and_unknown_image_rejected(self):
        self.assertEqual(len(contract._current_contract()),14)
        rules=self.rules()
        with self.assertRaisesRegex(ValueError,'Unknown frozen'):contract.validate_deferred_rules(rules)
        self.admit_synthetic_image(rules)
        self.assertEqual(contract.validate_deferred_rules(rules),('synthetic-test-only',rules.executor.image.identity))
        self.assertFalse(rules.supports_full_duel_rules)
        with patch('app.modules.card_game.engine.duel_v2.simulation.simulate_action',lambda *a:None):
            with self.assertRaisesRegex(ValueError,'source shape'):contract.validate_deferred_rules(rules)
    def test_missing_entry_corrupt_image_contextvar_and_raw_fail_closed(self):
        rules=self.rules();self.admit_synthetic_image(rules)
        image=rules.executor.image
        rules.executor.image=replace(image,entries={})
        with self.assertRaises(ValueError):contract.validate_deferred_rules(rules)
        rules.executor.image=image
        rules.executor.execute_only=True
        with self.assertRaisesRegex(ValueError,'Full native'):contract.validate_deferred_rules(rules)
        rules.executor.execute_only=False
        from app.modules.card_game.engine.duel_v2 import simulation
        with simulation._without_patches():
            with self.assertRaisesRegex(ValueError,'ContextVar'):contract.validate_deferred_rules(rules)
        rules.raw_tape=None
        with self.assertRaisesRegex(ValueError,'raw action tape'):contract.validate_deferred_rules(rules)
    def test_binding_requires_actual_enabled_owner_and_immediate_clean_window(self):
        rules=self.rules();self.admit_synthetic_image(rules)
        with self.assertRaisesRegex(ValueError,'Bind'):contract.require_deferred_step(rules)
        with self.assertRaisesRegex(ValueError,'DeviceGameLoop'):rules.bind_game_loop(NS(rows=1,device=rules.device,defer_physical_presentation=True))
        loop=self.bound_loop(rules);rules.bind_game_loop(loop)
        with self.assertRaisesRegex(ValueError,'immediate-clean'):contract.require_deferred_step(rules)
        loop._deferred_step_open=True;contract.require_deferred_step(rules)
        loop._deferred_raw_tape=None
        with self.assertRaisesRegex(ValueError,'binding changed'):contract.require_deferred_step(rules)
        loop._deferred_raw_tape=rules.raw_tape;loop.rules=object()
        with self.assertRaisesRegex(ValueError,'different rule'):contract.require_deferred_step(rules)
    def test_default_step_public_semantics_and_private_controls(self):
        rules=object.__new__(DeviceSoARules);calls=[]
        rules._step_device=lambda *a,**kw:calls.append(kw)
        rules.step_device(None,None,None,None,physical=True)
        rules.step_deferred_physical_device(None,None,None,None)
        self.assertEqual(calls,[dict(physical=True),dict(physical=True,_defer=True)])
        rules=object.__new__(DeviceSoARules)
        with self.assertRaises(TypeError):rules._step_device(None,None,None,None,_defer=1)
        with self.assertRaises(ValueError):rules._step_device(None,None,None,None,_defer=True)


class DeferredTypedStepTests(unittest.TestCase):
    def make(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
        from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
        from app.modules.card_game.rl.batched_duel.device_sparse_soa_storage import DeviceSparseSoAStore
        from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
        from tests import test_duel_v2_device_graph_storage as graphs
        t.set_num_threads(1);schema=compile_execution_schema();rows=3
        rules=object.__new__(DeviceSoARules);rules.t=t;rules.rows=rows;rules.device=t.device('cpu')
        rules.store=DeviceSparseSoAStore(schema,16,device='cpu',max_bytes=8<<20,allow_cpu_reference=True)
        self.addCleanup(rules.store.close)
        owners=t.arange(rows,dtype=t.int64)
        states=rules.store.put_device(pack_game_states([new_game(seed=30+i) for i in range(rows)],schema).to_torch('cpu'),
            owners,t.ones(rows,dtype=t.bool))
        rules.action_store=CompactDeviceGraphStore(slots=8,node_pages=16,edge_pages=16,payload_pages=16,
            node_page_bytes=256,edge_page_bytes=256,payload_page_bytes=256,
            max_node_pages_per_slot=4,max_edge_pages_per_slot=4,max_payload_pages_per_slot=4,
            device='cpu',allow_cpu_reference=True)
        self.addCleanup(rules.action_store.close)
        graph=graphs._graph([{'type':'end_turn'} for _ in range(rows)],'cpu',max_nodes=16,max_edges=32,max_payload=256)
        action=rules.action_store.put_device(graph,owners,t.ones(rows,dtype=t.bool))
        rules.layout=rules.action_layout=graph.layout
        rules._shared_batch=schema.allocate(rows).to_torch('cpu');rules._shared_scratch=graphs._scratch(graph)
        image=numeric_image();calls=[]
        rules.executor=NS(image=image,closed=False,execute_only=False,reset_workspace=lambda *a:None,
            compact_transient=lambda arena,roots,active:NS(arena=arena,errors=t.zeros(rows,dtype=t.int32)))
        # Numeric fake execute/commit, with real typed gather/store and raw graph
        # arenas. This verifies transport/routing, never claims CUDA rule parity.
        def invoke(entry,batch,scratch,refs,active):
            calls.append((entry,active.clone()));return NS(state=batch)
        rules._invoke=invoke
        rules._commit=lambda call,who,active,errors:rules.store.put_device(call.state,who,active.bool(),errors=errors)
        rules.binder=NS(bind_action_into_device=lambda *a,**kw:kw['argument_refs'].zero_())
        rules._keepalive=[];rules._raw_capture_hook=None;rules.physical_loop=None
        rules.raw_tape=DeviceRawActionTape(rows,8,128,128,1024,device='cpu',allow_cpu_reference=True)
        self.addCleanup(rules.raw_tape.close);rules.raw_tape.begin_device(owners,t.arange(rows,dtype=t.int64)+90)
        decision=DeviceDecision(states,rules.action_store,action.slots,action.generations,
            graph.roots[:,None].expand(-1,2).clone(),t.zeros((rows,2),dtype=t.int64),
            t.full((rows,),2,dtype=t.int64),t.ones((rows,2),dtype=t.bool),
            t.empty((rows,0)),t.empty((rows,2,0)),t.zeros(rows,dtype=t.int32))
        context=patch.object(contract,'_IMAGES',frozenset(((image.program_identity,image.identity),)))
        context.start();self.addCleanup(context.stop)
        loop=object.__new__(DeviceGameLoop);loop.rules=rules;loop.rows=rows;loop.device=rules.device
        loop.defer_physical_presentation=True;loop._deferred_step_open=True
        loop._deferred_physical_contract=contract.validate_deferred_rules(rules);loop._deferred_raw_tape=rules.raw_tape
        loop._steps=t.zeros(rows,dtype=t.int64);loop._source_versions=t.zeros(rows,dtype=t.int64)
        rules.bind_game_loop(loop)
        return rules,states,decision,calls
    def test_only_private_step_switches_entry_raw_once_and_typed_parent_is_immutable(self):
        from app.modules.card_game.rl.batched_duel.record_soa import unpack_game_states
        from scripts.validate_duel_v2_native_soa import compare_boundary
        for deferred in (False,True):
            with self.subTest(deferred=deferred):
                rules,states,decision,calls=self.make()
                original=unpack_game_states(rules.store.gather_device(states).to_host())
                states.active[1]=False;states.errors[2]=77
                args=(states,t.zeros(3,dtype=t.int64),decision,t.zeros(3,dtype=t.int64))
                with patch.object(t.cuda,'is_current_stream_capturing',return_value=True):
                    out=rules.step_deferred_physical_device(*args) if deferred else rules.step_device(*args,physical=True)
                self.assertEqual(calls[0][0],'simulate_action' if deferred else 'apply_action')
                self.assertEqual(calls[0][1].tolist(),[1,0,0])
                self.assertEqual(out.errors.tolist(),[0,0,77])
                self.assertEqual(int(rules.raw_tape._record_count),1)
                self.assertEqual(rules.raw_tape.steps_device.tolist(),[1,0,0])
                self.assertEqual(states.errors.tolist(),[0,0,77])
                request=replace(states,active=t.ones(3,dtype=t.bool),errors=t.zeros(3,dtype=t.int32))
                compare_boundary(unpack_game_states(rules.store.gather_device(request).to_host()),original,
                    'typed parent full state/RNG/aliases unchanged')
                self.assertTrue(bool((rules.action_store.live[decision.action_slots]==1).all()))
                self.assertFalse(rules.supports_full_duel_rules)


class DeferredGameLoopTests(unittest.TestCase):
    def make(self,defer=False,mid_error=False,mixed=False):
        from tests import test_duel_v2_device_games as fixture_module
        base,fixture,sink,roots,seeds,ids,train=fixture_module.DeviceGameLoopTest().make(rows=4)
        fixture.raw_tape=NS();events=[];raw=t.zeros(4,dtype=t.int64)
        step=fixture.step_device;clean=fixture.clean_device
        loop_ref=[]
        def ordinary(states,*args,physical=False):
            if physical:events.append(('apply',states.active.clone()));raw.add_(states.active.long())
            return step(states,*args,physical=physical)
        def deferred(states,*args):
            self.assertTrue(loop_ref[0]._deferred_step_open)
            events.append(('simulate',states.active.clone()));raw.add_(states.active.long())
            output=step(states,*args,physical=True)
            if mid_error:
                output=replace(output,errors=output.errors.clone());output.errors[0]=1
            return output
        def cleaned(states):
            if loop_ref and loop_ref[0]._ready:events.append(('clean',states.active.clone()))
            return clean(states)
        fixture.step_device=ordinary;fixture.step_deferred_physical_device=deferred;fixture.clean_device=cleaned
        # Explicit synthetic oracle: production source/type admission is tested above.
        with patch.object(contract,'validate_deferred_rules',return_value=('synthetic','oracle')):
            loop=DeviceGameLoop(fixture,base.search,sink,fixture_module.FixtureWorldSeeds(),max_decisions=8,
                device='cpu',allow_cpu_reference=True,allow_synthetic=True,inner_max_ticks=400,
                defer_physical_presentation=defer)
        loop_ref.append(loop)
        if mixed:
            roots.active[1]=False;fixture.data[roots.slots[2],0]=3;roots.errors[3]=55
        return loop,fixture,sink,roots,seeds,ids,train,events,raw
    def finish(self,items):
        loop,fixture,sink,roots,seeds,ids,train,*_=items
        loop.initialize(roots,seeds,ids,ids,train)
        for _ in range(12):
            if not bool(loop.unfinished_device.any()):break
            loop.tick()
        return loop.result_device()
    def test_default_and_explicit_closed_path_same_full_outcome_and_lease_cleanup(self):
        for defer in (False,True):
            items=self.make(defer);before=items[1].data[items[3].slots].clone();result=self.finish(items)
            self.assertEqual(result.steps.tolist(),[3]*4);self.assertEqual(result.errors.tolist(),[0]*4)
            self.assertEqual(items[-1].tolist(),[3]*4)
            physical=[name for name,active in items[-2] if name in ('apply','simulate')]
            self.assertEqual(physical,(['simulate'] if defer else ['apply'])*3)
            t.testing.assert_close(items[1].data[items[3].slots],before)
            self.assertEqual(int(items[1].live.sum()),4)
            self.assertFalse(items[0]._deferred_step_open)
    def test_finished_padding_original_errors_and_failed_midstep_do_not_advance(self):
        items=self.make(True,mixed=True);out=self.finish(items)
        self.assertEqual(out.steps.tolist(),[3,0,0,0]);self.assertEqual(items[-1].tolist(),[3,0,0,0])
        self.assertEqual(int(out.errors[3]),55)
        items=self.make(True,mid_error=True);out=self.finish(items)
        self.assertEqual(int(out.steps[0]),0);self.assertNotEqual(int(out.errors[0]),0)
        self.assertEqual(int(items[-1][0]),1)
        self.assertFalse(items[0]._deferred_step_open)
        # Failed physical output is retired; other rows retain only their final roots.
        self.assertEqual(int(items[1].live.sum()),4)
    def test_static_step_exception_closes_the_immediate_clean_window(self):
        items=self.make(True)
        def failed(*args):raise RuntimeError('synthetic static step failure')
        items[1].step_deferred_physical_device=failed
        with self.assertRaisesRegex(RuntimeError,'static step failure'):self.finish(items)
        self.assertFalse(items[0]._deferred_step_open)

    def test_constructor_boolean_and_missing_source_rejection(self):
        for value in (1,None,'yes'):
            with self.assertRaises(TypeError):DeviceGameLoop(None,None,None,None,max_decisions=1,defer_physical_presentation=value)
        with self.assertRaises(TypeError):DeviceGameLoop(NS(),None,None,None,max_decisions=1,defer_physical_presentation=True)


class DeferredAdapterDispatchTests(unittest.TestCase):
    def test_tiled_uses_same_queue_and_raw_hook_without_touching_public_step(self):
        from tests import test_duel_v2_tiered_soa_rules as fixtures
        factory=fixtures.TieredSoARuleTests();rules,roots,actors,_=factory.make();self.addCleanup(factory.doCleanups)
        tiled=rules.slow;calls=[]
        tiled.physical_loop=NS(steps_device=t.zeros(7,dtype=t.int64),source_versions_device=t.zeros(7,dtype=t.int64))
        original=tiled.inner._raw_capture_hook
        def inner(states,who,decision,indices,**kw):
            calls.append((states,kw,tiled.inner._raw_capture_hook))
            self.assertIs(kw['_deferred_owner'],tiled)
            return tiled.inner.step_device(states,who,decision,indices,physical=True)
        tiled.inner._step_device=inner
        decision,chosen,_=rules.rule_decision_device(roots,actors)
        with patch.object(contract,'require_deferred_step') as admit:
            out=tiled.step_deferred_physical_device(roots,actors,decision,chosen)
        self.assertEqual(admit.call_count,1);self.assertTrue(calls)
        self.assertTrue(all(kw['physical'] and kw['_defer'] and callable(hook) for _,kw,hook in calls))
        self.assertIs(tiled.inner._raw_capture_hook,original)
        self.assertEqual(tiled.raw_writes,int((roots.active&(roots.errors==0)).sum()))
        self.assertEqual(int(out.errors[2]),77)
    def test_tiered_deferred_physical_always_slow_once_without_retry(self):
        tiered=object.__new__(DeviceTieredSoARules);calls=[]
        tiered.slow=NS(step_deferred_physical_device=lambda *a:calls.append(a) or 'result')
        tiered.fast=NS(step_deferred_physical_device=lambda *a:self.fail('fast physical path'))
        with patch.object(contract,'require_deferred_step') as admit:
            self.assertEqual(tiered.step_deferred_physical_device(1,2,3,4),'result')
        self.assertEqual(calls,[(1,2,3,4)]);admit.assert_called_once_with(tiered)


if __name__=='__main__':unittest.main()
