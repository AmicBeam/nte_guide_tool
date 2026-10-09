"""Callable instances and bounded immutable deepcopy regression fixtures."""
import os
import sys
import unittest
from pathlib import Path
from app.modules.card_game.rl.rule_ir.portable_program import capture_program,FrozenRuleProgram,DEFAULT_ALLOWED_MODULE_PREFIXES
from app.modules.card_game.rl.batched_duel.gpu_rule_program import lower_gpu_program
from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,VmLimits,emit_vm_source
from app.modules.card_game.rl.batched_duel.codec import pack_states
from app.modules.card_game.rl.batched_duel.schema import GraphLayout

class CallableBase:
    def __call__(self,value,*,bonus=2):
        return value+bonus

class CallableChild(CallableBase):
    pass

def callable_probe(value):
    obj=CallableChild()
    obj.__call__=0
    return obj(value,bonus=3)

def atomic_copy_probe(value):
    from copy import deepcopy
    # Large unrelated data remains in the input heap; copying a scalar is O(1).
    return [deepcopy(value[0]) for _ in range(20)]

def repeated_copy_probe(value):
    from copy import deepcopy
    first=deepcopy(value)
    second=deepcopy(value)
    first[0].append(9)
    for _ in range(90):
        abandoned=(item for item in (1,))
        abandoned=None
    after_gc=deepcopy(second)
    return first,second,after_gc,second[0] is second[1],after_gc[0] is after_gc[1],first is second

def set_probe(value):
    a=set(value)
    b={2,3,7}
    output=[sorted(a&b),sorted(a|b),sorted(a^b),sorted(a-b)]
    alias=a
    a &= b
    output.append(sorted(alias))
    a |= {9}
    output.append(sorted(alias))
    a ^= {2,8}
    output.append(sorted(alias))
    a -= {7,8}
    output.append(sorted(alias))
    frozen=frozenset(value)&b
    return output,isinstance(frozen,frozenset),sorted(frozen)

def program():
    folder=os.environ.get('NTE_VM_TEST_PROGRAM_DIR')
    if sys.version_info[:2]!=(3,10):
        if not folder:raise RuntimeError('Frozen CPython3.10 callable/copy fixture required')
        return FrozenRuleProgram.from_json((Path(folder)/'callable_copy.json').read_text(encoding='utf8'))
    return capture_program({'callable':callable_probe,'atomic':atomic_copy_probe,'copies':repeated_copy_probe,'sets':set_probe},allowed_module_prefixes=(*DEFAULT_ALLOWED_MODULE_PREFIXES,__name__))

class CallableCopySourceTest(unittest.TestCase):
    def test_frozen_inherited_method_and_scalar_fast_path(self):
        source=emit_vm_source(lower_gpu_program(program()))
        self.assertIn('lookup_class(c,node(c,cls)[3],name,&category)',source)
        self.assertNotIn('for(int i=0;i<memo_size;i++)',source)
        self.assertIn('deepcopy_memo_get(c,key)',source)
        self.assertEqual(callable_probe(4),7)
        for value in (None,True,7,-0.0,'甲'):
            self.assertEqual(atomic_copy_probe([value,[]]),[value]*20)

try:
    import torch
    CUDA=torch.cuda.is_available()
except ImportError:CUDA=False

@unittest.skipUnless(CUDA,'Actual CUDA required')
class CallableCopyCudaTest(unittest.TestCase):
    def test_inherited_call_ignores_instance_shadow_and_forwards_keywords(self):
        with CudaVm(lower_gpu_program(program())) as vm:
            arena=vm.ingest(pack_states([4],GraphLayout(1,1024,2048,8192)))
            self.assertEqual(vm.launch(arena,'callable').decode()[0],7)
    def test_set_operations_and_inplace_aliases_match_python(self):
        with CudaVm(lower_gpu_program(program())) as vm:
            values=[[],[1,2,7],[2,3,9]]
            arena=vm.ingest(pack_states(values,GraphLayout(3,4096,8192,32768)))
            self.assertEqual(vm.launch(arena,'sets').decode(),[set_probe(v) for v in values])

    def test_repeated_copy_aliases_and_gc_do_not_reuse_stale_memo(self):
        shared=[3,4]
        value=[shared,shared]
        with CudaVm(lower_gpu_program(program())) as vm:
            arena=vm.ingest(pack_states([value],GraphLayout(1,4096,8192,32768)))
            actual=vm.launch(arena,'copies').decode()[0]
        self.assertEqual(actual,repeated_copy_probe(value))
        self.assertEqual(value,[[3,4],[3,4]])
        self.assertIs(actual[1][0],actual[1][1])
        self.assertIs(actual[2][0],actual[2][1])
        self.assertIsNot(actual[0],actual[1])
        self.assertIsNot(actual[1],actual[2])

    def test_atomic_copy_fits_budget_independent_of_unrelated_heap(self):
        with CudaVm(lower_gpu_program(program()),limits=VmLimits(steps=1000)) as vm:
            for scalar in (None,True,7,-0.0,'甲'):
                value=[scalar,list(range(1200))]
                arena=vm.ingest(pack_states([value],GraphLayout(1,4096,8192,32768)))
                self.assertEqual(vm.launch(arena,'atomic').decode()[0],[scalar]*20)

if __name__=='__main__':unittest.main()
