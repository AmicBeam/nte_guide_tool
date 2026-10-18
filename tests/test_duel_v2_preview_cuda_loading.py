"""CPU loading contracts only; no CUDA or rule execution qualification."""
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tests.test_duel_v2_soa_cuda_loading import scalar_fixture


class PreviewLoadingTest(unittest.TestCase):
    def program(self):
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
        return FrozenRuleProgram.from_json((Path(__file__).parent/'fixtures'/'duel_v2_scalar_compare_program.json').read_text())

    @contextmanager
    def environment(self,transform=None):
        import torch
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        from app.modules.card_game.rl.batched_duel.native_compile_units import NativeCompileUnit
        image,schema=scalar_fixture()
        units=tuple(NativeCompileUnit('unit%02d'%i,'source%02d'%i)for i in range(22))
        created=[];tensor=torch.tensor;full=torch.full
        class Module:
            def __init__(self,source,specs,**kwargs):
                self.source=source;self.kwargs=kwargs;self._compilation_profile={};created.append(self)
            def ensure_stack(self,size):self.stack=size
        with ExitStack()as stack:
            stack.enter_context(patch('torch.cuda.is_available',return_value=True))
            stack.enter_context(patch('torch.tensor',side_effect=lambda data,**kw:tensor(data,**dict(kw,device='cpu'))))
            stack.enter_context(patch('torch.full',side_effect=lambda shape,value,**kw:full(shape,value,**dict(kw,device='cpu'))))
            stack.enter_context(patch('app.modules.card_game.rl.batched_duel.native_soa_cuda.emit_native_soa_source',return_value=('canonical-monolithic',object())))
            stack.enter_context(patch('app.modules.card_game.rl.batched_duel.native_compile_units.split_native_source',return_value=units))
            stack.enter_context(patch('app.modules.card_game.rl.batched_duel.cuda_runtime.CudaModule',Module))
            stack.enter_context(patch.object(CudaSoARules,'prepare_result_path'))
            transform_mock=stack.enter_context(patch('app.modules.card_game.rl.batched_duel.native_preview_arena.transform_units',side_effect=transform))
            yield CudaSoARules,image,schema,units,created,transform_mock

    def test_public_optin_defaults_are_explicit_and_disabled(self):
        import inspect
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        for cls in (CudaSoARules,NativeSoACUDAQuery):
            signature=inspect.signature(cls)
            self.assertIs(signature.parameters['native_preview_arena'].default,False)
            self.assertIsNone(signature.parameters['preview_program'].default)

    def test_default_does_not_parse_program_or_transform_or_change_any_unit(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError('Disabled program must not be parsed')
        with self.environment()as(cls,image,schema,units,created,transform):
            executor=cls(image,schema,device='cuda:0',native_functions=True,separate_compilation=True,
                         preview_program=Poison())
            transform.assert_not_called()
            self.assertEqual(created[0].kwargs['compilation_units'],[(u.name,u.source)for u in units])
            self.assertEqual(executor.rule_source_sha256,executor.canonical_rule_source_sha256)
            self.assertEqual(executor.source_sha256,executor.rule_source_sha256)
            self.assertNotIn('native_preview_arena',created[0]._compilation_profile)
            self.assertIsNone(executor.preview_arena_binding)

    def test_enabled_uses_actual_post_specialization_units_and_distinct_binding(self):
        from app.modules.card_game.rl.batched_duel.native_compile_units import NativeCompileUnit
        verified=[]
        def specialize(unit,*args,**kw):return NativeCompileUnit(unit.name,unit.source+' specialized')
        def transform(units,image,schema,program):
            self.assertEqual(units[0].source,'source00 specialized')
            self.assertIs(program,actual_program)
            derived=tuple(NativeCompileUnit(u.name,u.source+' preview')if i==9 else u for i,u in enumerate(units))
            hashes=lambda values:{u.name:sha256(u.source.encode()).hexdigest()for u in values}
            return SimpleNamespace(units=derived,identity='d'*64,
                binding=dict(canonical_units=hashes(units),derived_units=hashes(derived),cuda_qualified=False),
                verify=lambda:verified.append(True))
        actual_program=self.program()
        with self.environment(transform)as(cls,image,schema,units,created,mock):
            with patch('app.modules.card_game.rl.batched_duel.native_runtime_index.index_runtime_attributes',side_effect=specialize):
                executor=cls(image,schema,device='cuda:0',native_functions=True,separate_compilation=True,
                    indexed_attributes=True,native_preview_arena=True,preview_program=actual_program)
            self.assertEqual(verified,[True]);mock.assert_called_once()
            self.assertEqual(executor.rule_source_sha256,'d'*64)
            self.assertEqual(executor.canonical_rule_source_sha256,sha256(b'canonical-monolithic').hexdigest())
            self.assertEqual(executor.source_sha256,executor.canonical_rule_source_sha256)
            actual=created[0].kwargs['compilation_units']
            self.assertEqual(actual[9],('unit09','source09 preview'))
            self.assertEqual(actual[0],('unit00','source00 specialized'))
            self.assertEqual(actual[10],('unit10','source10'))
            profile=created[0]._compilation_profile['native_preview_arena']
            self.assertEqual(profile['identity'],executor.rule_source_sha256)
            self.assertEqual(profile['binding'],executor.preview_arena_binding)
            self.assertFalse(profile['binding']['cuda_qualified'])

    def test_invalid_options_fail_before_cuda_setup(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import CudaSoARules
        base=dict(native_preview_arena=True,native_functions=True,separate_compilation=True,preview_program=self.program())
        variants=[dict(native_preview_arena=1),dict(native_functions=False),dict(separate_compilation=False),
                  dict(execute_only=True),dict(preview_program=None),dict(preview_program={}),dict(native_hydration_probe=True)]
        with patch('torch.cuda.is_available',side_effect=AssertionError('CUDA must not be consulted')):
            for variant in variants:
                with self.subTest(variant=variant),self.assertRaises(ValueError):
                    CudaSoARules(None,None,**dict(base,**variant))

    def test_transform_rejection_has_no_compilation_or_fallback(self):
        def reject(*args,**kwargs):raise ValueError('canonical22 mismatch')
        with self.environment(reject)as(cls,image,schema,_,created,mock):
            with self.assertRaisesRegex(ValueError,'canonical22 mismatch'):
                cls(image,schema,device='cuda:0',native_functions=True,separate_compilation=True,
                    native_preview_arena=True,preview_program=self.program())
            mock.assert_called_once();self.assertEqual(created,[])

    def test_query_forwards_optin_and_exports_actual_binding(self):
        import json
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        image,schema=scalar_fixture();program=self.program()
        executor=SimpleNamespace(source_sha256='m'*64,rule_source_sha256='d'*64,
            canonical_rule_source_sha256='c'*64,hydration_probe_binding=None,
            preview_arena_binding=dict(derived_units={'unit09':'9'*64},cuda_qualified=False),preview_arena_identity='d'*64)
        with tempfile.TemporaryDirectory()as folder, \
             patch('app.modules.card_game.rl.batched_duel.native_soa_cuda_query.CudaSoARules',return_value=executor)as factory, \
             patch('app.modules.card_game.rl.batched_duel.native_soa_source.emit_native_soa_source',return_value=('monolithic',None)):
            query=NativeSoACUDAQuery(image,schema,directory=folder,native_preview_arena=True,
                preview_program=program,native_functions=True,separate_compilation=True)
            self.assertTrue(factory.call_args.kwargs['native_preview_arena'])
            self.assertIs(factory.call_args.kwargs['preview_program'],program)
            self.assertEqual(query.rule_source_sha256,'d'*64)
            binding=json.loads((Path(folder)/'native_preview_arena_binding.json').read_text())
            self.assertEqual(binding['actual_rule_source_sha256'],'d'*64)
            self.assertEqual(binding['canonical_rule_source_sha256'],'c'*64)
            self.assertEqual(binding['actual_post_split_binding'],executor.preview_arena_binding)
            self.assertEqual((Path(folder)/'native_soa.cu').read_text(),'monolithic')


if __name__=='__main__':unittest.main()
