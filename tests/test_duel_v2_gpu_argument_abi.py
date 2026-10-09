"""Verify source declaration, registered ABI and call agree before CUDA launch."""
import ast
import inspect
import re
import textwrap
import unittest
from app.modules.card_game.rl.batched_duel.gpu_rule_arguments import (
    GraphArgumentKernel, _APPEND_SPEC, _META_SPEC, emit_argument_source,
)

class ArgumentAbiTest(unittest.TestCase):
    def test_all_argument_kernel_declarations_match_registered_types(self):
        source=emit_argument_source()
        for name,spec in [('graph_append_arguments',_APPEND_SPEC),('graph_read_meta',_META_SPEC)]:
            declaration=re.search(r'void '+name+r'\s*\((.*?)\)\s*\{',source,re.S).group(1)
            types=[]
            for parameter in declaration.split(','):
                pointer='*' in parameter
                types.append(('uint8' if 'unsigned char' in parameter else 'int32')+('*' if pointer else ''))
            self.assertEqual(tuple(types),spec,name)
    def test_append_call_arity_matches_source_and_registration(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(GraphArgumentKernel.bind)))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='launch']
        append=next(node for node in calls if isinstance(node.args[0],ast.Constant)
                    and node.args[0].value=='graph_append_arguments')
        self.assertIsInstance(append.args[1],ast.List)
        self.assertEqual(len(append.args[1].elts),len(_APPEND_SPEC))
        self.assertEqual(len(_APPEND_SPEC),28)

class GuardedEntryTest(unittest.TestCase):
    def test_standalone_guard_matches_constructor_without_permitting_other_gaps(self):
        from types import SimpleNamespace
        from app.modules.card_game.rl.batched_duel.gpu_rule_backend import GpuRuleBackend,GpuRuleCapabilityError
        backend=object.__new__(GpuRuleBackend)
        backend.image=SimpleNamespace(entries={'public_legal_decision':0})
        backend.entry_names=('public_legal_decision',)
        backend._capability_report={'public_legal_decision':dict(declared_capabilities_ready=False,
             missing={'opcodes':[],'intrinsics':['uuid.uuid4']})}
        backend.guard_standalone_entities=True
        self.assertEqual(backend._require_entry('decision'),'public_legal_decision')
        backend.guard_standalone_entities=False
        with self.assertRaises(GpuRuleCapabilityError):backend._require_entry('decision')
        backend.guard_standalone_entities=True
        backend._capability_report['public_legal_decision']['missing']['intrinsics'].append('unsupported')
        with self.assertRaises(GpuRuleCapabilityError):backend._require_entry('decision')

if __name__=='__main__':unittest.main()
