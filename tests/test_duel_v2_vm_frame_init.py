"""Compare reusable frames, keyword calls, closures and suspended-frame GC."""
from pathlib import Path
import os
import sys
import unittest
from app.modules.card_game.rl.batched_duel.vm_frame_init import integrate_frame_init, HOOK
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from tests.test_duel_v2_cuda_vm import signature_probe, generator_collection_probe


def variable(flag):
    if flag:
        value = 42
    return value


def reuse(value):
    for i in range(value):
        variable(True)
    return variable(False)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
    path = os.environ.get('NTE_VM_FRAME_INIT_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2] != (3,10):
        raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    return capture_program({'signature':signature_probe,'generators':generator_collection_probe,'reuse':reuse},allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,'tests.test_duel_v2_cuda_vm'),scope_character_ids=None,scope_card_ids=None)


class FrameInitTest(unittest.TestCase):
    def test_hook_is_unique_and_default_source_unchanged(self):
        from app.modules.card_game.rl.batched_duel.cuda_vm import emit_vm_source
        image=lower_program(program());source=emit_vm_source(image)
        self.assertEqual(source,emit_vm_source(image,compact_frame_init=False))
        self.assertIn(HOOK,source)
        self.assertNotIn(HOOK,emit_vm_source(image,compact_frame_init=True))
        for bad in ('',HOOK+HOOK):
            with self.assertRaises(ValueError):integrate_frame_init(bad)
        with self.assertRaises(TypeError):emit_vm_source(image,compact_frame_init=1)

    def test_real_cuda_reused_slots_cells_and_gc_and_unbound_error(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm, VmLimits
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program())
        cases=[('signature',7,signature_probe(7)),('generators',100,generator_collection_probe(100)),('reuse',100,None)]
        observed=[]
        for flag in (False,True):
            rows=[]
            with CudaVm(image,limits=VmLimits(frames=12),compact_frame_init=flag) as vm:
                for entry,value,expected in cases:
                    call=vm.launch(vm.ingest(pack_states([value],GraphLayout(1,16384,32768,131072))),entry)
                    torch.cuda.synchronize();status=call.statuses.cpu().tolist()
                    rows.append((status,None if status[0][0] else call.decode()))
            observed.append(rows)
        self.assertEqual(observed[0],observed[1])
        for (entry,value,expected),(status,actual) in zip(cases,observed[1]):
            if entry=='reuse':
                self.assertEqual(status[0][0],103)
            else:
                self.assertEqual(actual,[expected])



if __name__=='__main__':
    unittest.main()
