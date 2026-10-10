"""Actual masked VM and wide CPython MT seeds; not complete search evidence."""
import os
from pathlib import Path
import unittest
from random import Random
from app.modules.card_game.rl.rule_ir.portable_program import capture_program, FrozenRuleProgram, DEFAULT_ALLOWED_MODULE_PREFIXES
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def wide_seed_probe(seed):
    rng = Random(seed)
    items = list(range(12))
    rng.shuffle(items)
    return rng.getrandbits(32), rng.random(), rng.randrange(10000), rng.choice(items), items


def increment(value):
    value['calls'] += 1
    return value


def program():
    fixture=os.environ.get('NTE_DEVICE_VM_TEST_PROGRAM')
    if fixture:return FrozenRuleProgram.from_json(Path(fixture).read_text())
    return capture_program({'wide':wide_seed_probe,'increment':increment},
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),scope_character_ids=None,scope_card_ids=None)


class DeviceVmTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def test_actual_cuda_wide_integer_seed_exact_python_stream(self):
        from hashlib import sha256
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        root=(1<<256)|int.from_bytes(sha256(b'recovery:20261011:17:a').digest(),'big')
        values=[0,-1,1<<63, -(1<<63),root,root+31*104729,-root,(1<<257)-1+31*104729]
        with CudaVm(lower_program(program())) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(len(values),4096,8192,65536))),'wide').decode()
        self.assertEqual(actual,[wide_seed_probe(value) for value in values])

    def test_actual_cuda_masked_call_graph_never_executes_inactive_rows(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        with CudaVm(lower_program(program())) as vm:
            arena=vm.ingest(pack_states([{'calls':0},{'calls':4}],GraphLayout(2,128,256,4096)))
            active=torch.tensor([1,0],dtype=torch.uint8,device=vm.device)
            deadlines=torch.zeros(2,dtype=torch.int64,device=vm.device)
            args=arena.roots[:,None].contiguous()
            vm.launch_device(arena,'increment',argument_refs=args,active=active,gpu_deadlines=deadlines)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call=vm.launch_device(arena,'increment',argument_refs=args,active=active,gpu_deadlines=deadlines)
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(call.decode(),[{'calls':1},{'calls':4}])
            active.copy_(torch.tensor([0,1],dtype=torch.uint8,device=vm.device))
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(call.decode(),[{'calls':0},{'calls':5}])

    def test_reusable_workspace_capture_has_no_heap_fork_and_reads_new_selectors(self):
        import torch
        from contextlib import ExitStack
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,DeviceGraph
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        with CudaVm(lower_program(program())) as vm:
            original=vm.ingest(pack_states([{'calls':0},{'calls':4}],GraphLayout(2,128,256,4096)))
            arena=original.fork()
            workspace=vm.prepare_device_workspace(arena)
            active=torch.tensor([1,0],dtype=torch.uint8,device=vm.device)
            deadlines=torch.zeros(2,dtype=torch.int64,device=vm.device)
            args=arena.roots[:,None].contiguous()
            pointers=[getattr(workspace,k).data_ptr() for k in ('frames','outputs','statuses','marks','queue')]
            vm.launch_into_device(workspace,'increment',argument_refs=args,active=active,gpu_deadlines=deadlines)
            torch.cuda.synchronize()
            for name in ('nodes','edges','payload','counts','roots'):
                getattr(arena,name).copy_(getattr(original,name))
            graph=torch.cuda.CUDAGraph()
            with ExitStack() as stack:
                stack.enter_context(patch.object(DeviceGraph,'fork',side_effect=AssertionError('Heap fork')))
                for name in ('empty','zeros','empty_like','zeros_like'):
                    stack.enter_context(patch.object(torch,name,side_effect=AssertionError('Workspace allocation')))
                for name in ('cpu','numpy','item','tolist','__bool__'):
                    stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError('Host selector')))
                with torch.cuda.graph(graph):
                    call=vm.launch_into_device(workspace,'increment',argument_refs=args,active=active,gpu_deadlines=deadlines)
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(call.decode(),[{'calls':1},{'calls':4}])
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(call.decode(),[{'calls':2},{'calls':4}])
            active.copy_(torch.tensor([0,1],dtype=torch.uint8,device=vm.device))
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(call.decode(),[{'calls':2},{'calls':5}])
            self.assertIs(call.arena,arena)
            self.assertEqual(pointers,[getattr(workspace,k).data_ptr() for k in ('frames','outputs','statuses','marks','queue')])
            before=[getattr(arena,name).clone() for name in ('nodes','edges','payload','counts','roots')]
            deadlines.fill_(-1);graph.replay();torch.cuda.synchronize()
            self.assertEqual(call.statuses[:,0].tolist(),[0,108])
            for name,value in zip(('nodes','edges','payload','counts','roots'),before):
                self.assertTrue(torch.equal(getattr(arena,name),value))

    def test_device_compaction_smaller_lanes_preserves_aliases_and_failed_rows(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,DeviceGraph
        from app.modules.card_game.rl.batched_duel.codec import pack_states,unpack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout,PackedStateBatch
        from unittest.mock import patch
        shared={'unknown':'retain'}
        values=[{'calls':0,'extra':[shared,shared]},{'calls':4}]
        with CudaVm(lower_program(program())) as vm:
            source=vm.ingest(pack_states(values,GraphLayout(2,128,256,4096)))
            target=vm.ingest(pack_states([None,{'retained':True}],GraphLayout(2,32,64,512)))
            compact=vm.prepare_compaction_workspace(source,target)
            work=vm.prepare_device_workspace(source)
            active=torch.tensor([1,0],dtype=torch.uint8,device=vm.device)
            args=source.roots[:,None].contiguous()
            deadline=torch.zeros(2,dtype=torch.int64,device=vm.device)
            call=vm.launch_into_device(work,'increment',argument_refs=args,active=active,gpu_deadlines=deadline)
            torch.cuda.synchronize()
            before=[getattr(source,k).clone() for k in ('nodes','edges','payload','counts','roots')]
            def decoded():
                return unpack_states(PackedStateBatch(target.layout,*[
                    getattr(target,k).cpu().numpy() for k in ('nodes','edges','payload','counts','roots')]))
            graph=torch.cuda.CUDAGraph()
            with patch.object(DeviceGraph,'fork',side_effect=AssertionError('Compaction heap clone')):
                with torch.cuda.graph(graph):
                    returned=vm.compact_into_device(call,compact,active=active)
            graph.replay();torch.cuda.synchronize()
            self.assertIs(returned,compact)
            self.assertEqual(compact.errors.tolist(),[0,0])
            got=decoded()
            self.assertEqual(got,[{'calls':1,'extra':[shared,shared]},{'retained':True}])
            self.assertIs(got[0]['extra'][0],got[0]['extra'][1])
            for name,value in zip(('nodes','edges','payload','counts','roots'),before):
                self.assertTrue(torch.equal(getattr(source,name),value))
            # Failed rules cannot overwrite a valid destination even when active.
            call.statuses[:,0].fill_(108);active.fill_(1)
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(compact.errors.tolist(),[108,108])
            self.assertEqual(decoded(),got)
            call.statuses.zero_()
            tiny=vm.ingest(pack_states([None,None],GraphLayout(2,1,1,1)))
            overflow=vm.prepare_compaction_workspace(source,tiny)
            vm.compact_into_device(call,overflow,active=active);torch.cuda.synchronize()
            self.assertTrue(bool((overflow.errors!=0).all()))
