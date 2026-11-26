"""Generator join continuation fixture; source emission alone is not CUDA proof."""
from pathlib import Path
import os,sys,unittest
from app.modules.card_game.rl.rule_ir.portable_program import capture_program,FrozenRuleProgram,DEFAULT_ALLOWED_MODULE_PREFIXES
from app.modules.card_game.rl.batched_duel.gpu_rule_program import lower_gpu_program
from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,emit_vm_source
from app.modules.card_game.rl.batched_duel.codec import pack_states
from app.modules.card_game.rl.batched_duel.schema import GraphLayout
from app.modules.card_game.rl.batched_duel.vm_frame_gc import emit_frame_gc_source

def join_probe(value):
    generator=(item for item in value)
    mapping=dict(existing='0')
    mapping.update({'existing':'1'},joined='，'.join(generator))
    first=mapping['joined']
    exhausted='-'.join(generator)
    return first,exhausted

def program():
    if sys.version_info[:2]==(3,10):
        return capture_program({'join':join_probe},allowed_module_prefixes=(*DEFAULT_ALLOWED_MODULE_PREFIXES,__name__))
    folder=os.environ.get('NTE_VM_TEST_PROGRAM_DIR')
    if not folder:raise RuntimeError('Frozen CPython3.10 join fixture required')
    return FrozenRuleProgram.from_json((Path(folder)/'join_probe.json').read_text(encoding='utf8'))

class JoinSourceTest(unittest.TestCase):
    def test_continuation_is_registered_and_protected_by_gc(self):
        source=emit_vm_source(lower_gpu_program(program()))
        self.assertIn('text_iterable_call(c,callable,args,nargs,caller)',source)
        self.assertIn('if(mode==8){text_join_resume(',source)
        self.assertIn('f[5]==8',emit_frame_gc_source())
        self.assertEqual(join_probe(['甲','乙']),('甲，乙',''))
        self.assertEqual(join_probe([]),('',''))
        with self.assertRaises(TypeError):join_probe(['a',2])

try:
    import torch
    CUDA=torch.cuda.is_available()
except ImportError:CUDA=False

@unittest.skipUnless(CUDA,'Actual CUDA required')
class JoinCudaTest(unittest.TestCase):
    def test_empty_unicode_and_exhausted_generator_match_python(self):
        image=lower_gpu_program(program())
        with CudaVm(image) as vm:
            for value in ([],['甲','乙'],['','x','']):
                arena=vm.ingest(pack_states([value],GraphLayout(1,1024,2048,8192)))
                self.assertEqual(vm.launch(arena,'join').decode()[0],join_probe(value))

if __name__=='__main__':unittest.main()
