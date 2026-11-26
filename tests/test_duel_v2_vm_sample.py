"""Exact sequence sampling and following MT draws on the actual CUDA VM."""
from pathlib import Path
import os,sys,unittest
from random import Random


def sample_probe(packet):
    rng=Random(packet['seed']);population=packet['population']
    samples=[rng.sample(population,k) for k in packet['sizes']]
    if packet.get('mutate') and samples and samples[0]:samples[0][0]['changed']=True
    return samples,population,rng.getrandbits(64)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import capture_program,FrozenRuleProgram,DEFAULT_ALLOWED_MODULE_PREFIXES
    path=os.environ.get('NTE_VM_SAMPLE_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Explicit frozen fixture required')
    return capture_program({'sample':sample_probe},allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),scope_character_ids=None,scope_card_ids=None)


class RandomSampleCudaTest(unittest.TestCase):
    def test_actual_cuda_pool_set_zero_duplicates_alias_and_following_rng(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from copy import deepcopy
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        cases=[]
        for n,k in ((0,0),(4,0),(4,1),(4,4),(21,5),(22,5),(85,6),(86,6),(1000,2),(500,150)):
            for seed in (0,913,2**100+7):
                cases.append(dict(seed=seed,population=list(range(n)),sizes=[k,k,0]))
        cases.append(dict(seed=73,population=[{'n':i} for i in range(8)],sizes=[4,2],mutate=True))
        cases.append(dict(seed=-271,population=['same','same','other','same'],sizes=[3,4]))
        with CudaVm(lower_program(program()),constant_lookup=True,compact_frame_init=True,
                    device_call_boundaries=True,library_call_boundaries=True) as vm:
            for start in range(0,len(cases),8):
                batch=cases[start:start+8]
                expected=[sample_probe(deepcopy(v)) for v in batch]
                arena=vm.ingest(pack_states(batch,GraphLayout(len(batch),16384,32768,131072)))
                self.assertEqual(vm.launch(arena,'sample').decode(),expected)
