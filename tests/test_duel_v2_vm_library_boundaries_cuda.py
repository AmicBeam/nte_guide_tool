"""Actual CUDA equivalence of optional large-library call boundaries."""
import os,sys,unittest
from pathlib import Path
from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,VmLimits,emit_vm_source
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from tests.test_duel_v2_cuda_vm import signature_probe,generator_collection_probe


def sorted_probe(value):
    entries=[value+3,value-2,value+1]
    return sorted(entries),min(entries),max(entries)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
    path=os.environ.get('NTE_VM_LIBRARY_BOUNDARIES_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 fixture required')
    return capture_program({'sorted':sorted_probe,'signature':signature_probe,'generators':generator_collection_probe},allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,'tests.test_duel_v2_cuda_vm'),scope_character_ids=None,scope_card_ids=None)


class LibraryCudaTest(unittest.TestCase):
    def test_default_identical_and_inverse_transform(self):
        image=lower_program(program());base=emit_vm_source(image)
        self.assertEqual(base,emit_vm_source(image,library_call_boundaries=False))
        variant=emit_vm_source(image,library_call_boundaries=True)
        self.assertEqual(variant.replace('__noinline__','inline'),base)
        with self.assertRaises(TypeError):emit_vm_source(image,library_call_boundaries=1)

    def test_actual_cuda_outputs_gc_and_step_status(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program());results=[]
        for option in (False,True):
            with CudaVm(image,limits=VmLimits(frames=12),device_call_boundaries=True,library_call_boundaries=option) as vm:
                rows=[]
                for entry,value in [('sorted',9),('signature',7),('generators',100)]:
                    call=vm.launch(vm.ingest(pack_states([value],GraphLayout(1,16384,32768,131072))),entry)
                    rows.append((call.decode(),call.statuses.cpu().tolist()))
                results.append(rows)
        self.assertEqual(results[0],results[1])
        statuses=[]
        for option in (False,True):
            with CudaVm(image,limits=VmLimits(steps=7),device_call_boundaries=True,library_call_boundaries=option) as vm:
                call=vm.launch(vm.ingest(pack_states([3],GraphLayout(1,4096,8192,32768))),'sorted')
                torch.cuda.synchronize();statuses.append(call.statuses.cpu().tolist())
        self.assertEqual(statuses[0],statuses[1]);self.assertEqual(statuses[1][0][0],107)


if __name__=='__main__':unittest.main()
