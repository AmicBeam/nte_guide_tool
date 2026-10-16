"""Direct primitive routing preserves original bodies, budgets and status."""
import unittest,hashlib
from tests.test_duel_v2_vm_compiled_blocks import program,arithmetic,loop,aliases
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source,VmLimits


class DirectDispatchContractTest(unittest.TestCase):
    def test_default_unchanged_and_direct_route_after_budget_before_original_switch(self):
        image=lower_program(program());base=emit_vm_source(image)
        self.assertEqual(base,emit_vm_source(image,direct_stack_dispatch=False))
        fast=emit_vm_source(image,direct_stack_dispatch=True)
        marker=fast.index('// DIRECT_STACK_DISPATCH:')
        self.assertLess(fast.index('if(++c->steps>c->step_limit)',fast.index('__device__ inline void execute')),marker)
        self.assertLess(marker,fast.index('switch(op)',marker))
        start=base.index('  switch(op){',base.index('__device__ inline void execute'))
        end=base.index('extern "C" __global__ void vm_clock',start)
        self.assertIn(base[start:end],fast)
        with self.assertRaises(TypeError):emit_vm_source(image,direct_stack_dispatch=1)

    def test_four_direct_ops_have_no_early_library_interception(self):
        image=lower_program(program())
        from app.modules.card_game.rl.batched_duel.vm_strings import emit_string_source
        from app.modules.card_game.rl.batched_duel.vm_exceptions import emit_exception_source
        from app.modules.card_game.rl.batched_duel.vm_stdlib import emit_stdlib_source
        from app.modules.card_game.rl.batched_duel.vm_collections import emit_collection_source
        sources=(emit_string_source(image.intrinsic_names),emit_exception_source(image),emit_stdlib_source(image),emit_collection_source())
        for source in sources:
            for name in ('LOAD_CONST','LOAD_FAST','STORE_FAST','POP_TOP'):self.assertNotIn('OP_'+name,source)


class DirectDispatchCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def options(self):
        return dict(constant_lookup=True,compact_frame_init=True,device_call_boundaries=True,
                    library_call_boundaries=True,mutable_lookup_cache=True,frozen_class_lookup=True)

    def test_actual_cuda_values_aliases_and_compiled_source_binding(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program())
        for entry,values,formal in (('arithmetic',[1,-5,17],arithmetic),('loop',[0,5,25],loop),('aliases',[[1,2,3],[4],[]],aliases)):
            for direct in (False,True):
                options=dict(self.options(),direct_stack_dispatch=direct)
                with CudaVm(image,**options) as vm:
                    expected_sha=hashlib.sha256(emit_vm_source(image,**options).encode()).hexdigest()
                    self.assertEqual(vm.module.source_sha,expected_sha)
                    graph=vm.ingest(pack_states(values,GraphLayout(3,4096,8192,32768)))
                    result=vm.launch(graph,entry).decode()
                    self.assertEqual(result,[formal(value) for value in values])
                    if entry=='aliases':self.assertIs(result[0][0][0],result[0][0][1])

    def test_actual_cuda_step_deadline_and_overflow_status_identical(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program())
        for entry,value,steps,deadline,code in (('arithmetic',2,2,0,107),('loop',1000,2000000,1,108),('arithmetic',2**63-1,2000000,0,7)):
            results=[]
            for direct in (False,True):
                with CudaVm(image,limits=VmLimits(steps=steps),direct_stack_dispatch=direct,**self.options()) as vm:
                    graph=vm.ingest(pack_states([value],GraphLayout(1,4096,8192,32768)))
                    deadlines=torch.tensor([deadline],dtype=torch.int64,device=vm.device)
                    call=vm.launch(graph,entry,gpu_deadlines=deadlines);torch.cuda.synchronize()
                    results.append(call.statuses.cpu().tolist())
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0][0][0],code)


if __name__=='__main__':unittest.main()
