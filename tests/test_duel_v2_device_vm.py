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
