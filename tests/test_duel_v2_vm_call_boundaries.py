"""Generated call boundaries and actual CUDA output/status equivalence."""
import os,sys,unittest
from pathlib import Path
from app.modules.card_game.rl.batched_duel.vm_call_boundaries import separate_calls_source
from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source,CudaVm,VmLimits
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from tests.test_duel_v2_cuda_vm import signature_probe,generator_collection_probe


def add(value):return (value+3)*2
def nested(value):return add(value)+add(value+1)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
    path=os.environ.get('NTE_VM_CALL_BOUNDARIES_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    return capture_program({'nested':nested,'signature':signature_probe,'generators':generator_collection_probe},allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,'tests.test_duel_v2_cuda_vm'),scope_character_ids=None,scope_card_ids=None)


class CallBoundaryTest(unittest.TestCase):
    def test_exact_qualifier_transform_comments_strings_and_default(self):
        source='// __device__ inline void call();\n/*\n__device__ inline void execute();\n*/\nconst char* x="__device__ inline int new_frame();";\n__device__ inline void call();\n__device__ inline int new_frame(){return 7;}\n__device__ inline void execute(){call();}\n'
        changed=separate_calls_source(source)
        self.assertEqual(changed.replace('__noinline__','inline'),source)
        self.assertEqual(changed.count('__noinline__'),3)
        self.assertIn('"__device__ inline int new_frame();"',changed)
        image=lower_program(program());base=emit_vm_source(image)
        self.assertEqual(base,emit_vm_source(image,device_call_boundaries=False))
        separated=emit_vm_source(image,device_call_boundaries=True)
        self.assertEqual(separated.replace('__noinline__','inline'),base)
        for names in ((),('unknown',),('call','call')):
            with self.assertRaises((TypeError,ValueError)):separate_calls_source(source,names)
        with self.assertRaises(ValueError):separate_calls_source('// missing')

    def test_actual_cuda_outputs_aliases_gc_and_step_statuses(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program());results=[]
        for flag in (False,True):
            rows=[]
            with CudaVm(image,limits=VmLimits(frames=12),device_call_boundaries=flag) as vm:
                for entry,value in [('nested',13),('signature',7),('generators',100)]:
                    call=vm.launch(vm.ingest(pack_states([value],GraphLayout(1,16384,32768,131072))),entry)
                    rows.append((call.decode(),call.statuses.cpu().tolist()))
            results.append(rows)
        self.assertEqual(results[0],results[1])
        statuses=[]
        for flag in (False,True):
            with CudaVm(image,limits=VmLimits(steps=7),device_call_boundaries=flag) as vm:
                call=vm.launch(vm.ingest(pack_states([3],GraphLayout(1,4096,8192,32768))),'nested')
                torch.cuda.synchronize();statuses.append(call.statuses.cpu().tolist())
        self.assertEqual(statuses[0],statuses[1]);self.assertEqual(statuses[1][0][0],107)


if __name__=='__main__':unittest.main()
