"""Compiled block CPU contracts and CUDA equivalence; no speedup claims."""
from pathlib import Path
import os,sys,unittest
from app.modules.card_game.rl.batched_duel.vm_compiled_blocks import emit_compiled_blocks,CompiledBlockError
from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source,VmLimits
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def arithmetic(value):
    return (value+3)*2-1


def loop(value):
    total=0
    for i in range(value):
        total+=i*3
    return total


def aliases(value):
    return [value,value],value[1:],sum(value)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
    path=os.environ.get('NTE_VM_COMPILED_BLOCKS_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    return capture_program({'arithmetic':arithmetic,'loop':loop,'aliases':aliases},allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),scope_character_ids=None,scope_card_ids=None)


class CompiledBlocksContractTest(unittest.TestCase):
    def test_deterministic_bounded_code_and_fences(self):
        image=lower_program(program());a=emit_compiled_blocks(image);b=emit_compiled_blocks(image)
        self.assertEqual(a.identity,b.identity);self.assertTrue(a.verify())
        self.assertGreater(a.coverage['compiled_op_count'],0)
        self.assertLessEqual(len(a.coverage['compiled_blocks']),64)
        self.assertTrue(any(f['reason'] in ('call','return','jump') for f in a.coverage['fenced_ops']))
        for block in a.coverage['compiled_blocks']:
            self.assertFalse(any(op.startswith('CALL') or op.startswith('JUMP') or op=='RETURN_VALUE' for op in block['ops']))
        self.assertFalse(a.coverage['cpu_fallback']);self.assertFalse(a.coverage['measured_speedup'])

    def test_bound_and_image_rejection_and_small_budget(self):
        image=lower_program(program())
        for updates in (dict(max_blocks=True),dict(max_functions=0),dict(max_source_chars=-1)):
            with self.assertRaises(ValueError):emit_compiled_blocks(image,**updates)
        with self.assertRaises(TypeError):emit_compiled_blocks(object())
        with self.assertRaises(CompiledBlockError):emit_compiled_blocks(image,entries=('missing',))
        small=emit_compiled_blocks(image,max_blocks=1,max_ops=2)
        self.assertLessEqual(len(small.coverage['compiled_blocks']),1);self.assertLessEqual(small.coverage['compiled_op_count'],2)

    def test_integrated_source_default_unchanged_and_guard_order(self):
        image=lower_program(program());base=emit_vm_source(image)
        self.assertEqual(base,emit_vm_source(image,compiled_blocks=None))
        compiled=emit_vm_source(image,compiled_blocks={})
        self.assertEqual(compiled.count('__device__ inline int run_compiled_block(Context* c)'),1)
        entry=compiled.index('__device__ inline void execute(Context* c)')
        hook=compiled.index('if(run_compiled_block(c))continue;',entry)
        self.assertLess(compiled.index('if(c->raised!=MISSING)',entry),hook)
        self.assertLess(hook,compiled.index('if(++c->steps>c->step_limit)',hook))
        self.assertIn('f[1]=next_pc',compiled)


class CompiledBlocksCudaTest(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')

    def test_same_outputs_aliases_and_per_instruction_step_limit(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program())
        for entry,values in [('arithmetic',[1,-5,17]),('loop',[0,5,25]),('aliases',[[1,2,3],[4],[]])]:
            results=[]
            for config in (None,{}):
                with CudaVm(image,compiled_blocks=config) as vm:
                    packed=pack_states(values,GraphLayout(3,4096,8192,32768));graph=vm.ingest(packed)
                    call=vm.launch(graph,entry);decoded=call.decode();results.append(decoded)
                    if entry=='aliases':self.assertIs(decoded[0][0][0],decoded[0][0][1])
            self.assertEqual(results[0],results[1])
        statuses=[]
        for config in (None,{}):
            with CudaVm(image,limits=VmLimits(steps=2),compiled_blocks=config) as vm:
                graph=vm.ingest(pack_states([2],GraphLayout(1,128,256,1024)));call=vm.launch(graph,'arithmetic')
                vm.torch.cuda.synchronize();statuses.append(call.statuses.cpu().tolist())
        self.assertEqual(statuses[0],statuses[1]);self.assertEqual(statuses[0][0][0],107)

    def test_periodic_deadline_and_overflow_status_match_interpreter(self):
        import torch
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program())
        for entry,value,deadline,expected in [('loop',1000,1,108),('arithmetic',2**63-1,0,7)]:
            statuses=[]
            for config in (None,{}):
                with CudaVm(image,compiled_blocks=config) as vm:
                    graph=vm.ingest(pack_states([value],GraphLayout(1,4096,8192,32768)))
                    call=vm.launch(graph,entry,gpu_deadlines=torch.tensor([deadline],dtype=torch.int64,device=vm.device))
                    torch.cuda.synchronize();statuses.append(call.statuses.cpu().tolist())
            self.assertEqual(statuses[0],statuses[1]);self.assertEqual(statuses[0][0][0],expected)


if __name__=='__main__':unittest.main()
