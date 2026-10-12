"""Fused mapping operations preserve interpreter dispatch and diagnostics."""
from pathlib import Path
import os,sys,unittest
from app.modules.card_game.rl.batched_duel.vm_compiled_blocks import emit_compiled_blocks,CompiledBlockError
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def lookup(data):
    value=data['key']
    return value, 'key' in data, 'missing' not in data


def sequences(data):
    return data['text']+'x',data['items']+[4],data['items'][-1],data['items'][1:]


def caught(data):
    try:
        return data['missing']
    except KeyError:
        try:
            return data['items'][9]
        except IndexError:
            return 'caught'


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram,capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
    path=os.environ.get('NTE_VM_COMPILED_MAPPING_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise unittest.SkipTest('Same-source CPython3.10 fixture export required')
    return capture_program({'lookup':lookup,'sequences':sequences,'caught':caught},allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),scope_character_ids=None,scope_card_ids=None)


class CompiledMappingTest(unittest.TestCase):
    def config(self,image):
        return dict(function_ids=list(image.entries.values()),max_functions=3,max_prefix_ops=256)

    def test_explicit_selection_rejection_and_source_interception(self):
        image=lower_program(program());config=self.config(image)
        plan=emit_compiled_blocks(image,**config);self.assertTrue(plan.verify())
        self.assertEqual(plan.coverage['selected_functions'],tuple(image.function_ids[v] for v in config['function_ids']))
        ops={op for b in plan.coverage['compiled_blocks'] for op in b['ops']}
        self.assertIn('BINARY_SUBSCR',ops);self.assertIn('CONTAINS_OP',ops)
        for hook in ('handle_strings','handle_exceptions','stdlib_handle_opcode','handle_collections','c->raised!=MISSING','c->current!=id'):
            self.assertIn(hook,plan.source)
        for ids in ([],[True],[-1],[len(image.functions)],[0,0],list(range(4))):
            with self.assertRaises(CompiledBlockError):emit_compiled_blocks(image,function_ids=ids,max_functions=3)
        with self.assertRaises(CompiledBlockError):emit_compiled_blocks(image,function_ids=[0],entries=('lookup',))

    def test_actual_cuda_mapping_sequences_exception_and_step_boundaries(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm,VmLimits
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        image=lower_program(program());layout=GraphLayout(3,4096,8192,32768)
        cases=[{'key':7,'text':'ab','items':[1,2,3]},{'key':-2,'text':'','items':[8]},{'key':0,'text':'中文','items':[]}]
        configs=(None,self.config(image));outputs=[]
        for config in configs:
            with CudaVm(image,compiled_blocks=config) as vm:
                result={}
                for entry in image.entries:
                    inputs=cases if entry!='sequences' else [cases[0],cases[1],dict(cases[2],items=[5,6])]
                    call=vm.launch(vm.ingest(pack_states(inputs,layout)),entry);torch.cuda.synchronize()
                    statuses=call.statuses.cpu().tolist()
                    result[entry]=(statuses,call.decode() if all(row[0]==0 for row in statuses) else None)
                outputs.append(result)
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[0]['caught'][1],['caught']*3)
        # Failing instructions and every early return must retain exact PC/step.
        for limit in (1,2,3,4,5,6,7,8,12,16,24,32):
            status=[]
            for config in configs:
                with CudaVm(image,limits=VmLimits(steps=limit),compiled_blocks=config) as vm:
                    call=vm.launch(vm.ingest(pack_states(cases,layout)),'caught');torch.cuda.synchronize()
                    status.append(call.statuses.cpu().tolist())
            self.assertEqual(status[0],status[1],limit)


if __name__=='__main__':unittest.main()
