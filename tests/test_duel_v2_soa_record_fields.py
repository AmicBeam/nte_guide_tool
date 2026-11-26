"""Record lookup must keep exact text, missing fields and insertion order."""
import copy
import os
from pathlib import Path
import sys
import unittest

from app.modules.card_game.rl.batched_duel.record_soa import compile_schema, pack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits


def edit_field(state, parts):
    key = ''.join(parts)
    before = (key in state, state.get(key, -999), list(state))
    missing = None
    if key in state:
        old = state[key]
        del state[key]
        missing = (key in state, state.get(key, -999))
        state[key] = old + 1
    return state, before, missing, list(state)


def literal_fields(state):
    return ('hp' in state, 'field_63' in state, 'missing' in state,
            1 in state, None in state, False in state, state.get('field_63', -999), list(state))


def repeat_field_reads(state, parts, repeats):
    key = ''.join(parts)
    total = 0
    for _ in range(repeats):
        if key in state:
            total += state.get(key, 0)
    return total


def two_record_pools(state, parts):
    key = ''.join(parts)
    child = state['child']
    return state.get(key, -999), child.get(key, -999), key in state, key in child


def schema():
    fields = {f'field_{i:02d}': 'int' for i in range(64)}
    fields.update({key: 'int' for key in ('hp', '汉字', 'é', 'e\u0301', '')})
    return compile_schema({'Game': dict(kind='record', capacity=8, fields=fields)},
                          max_strings=128, max_string_bytes=4096)


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_SOA_RECORD_FIELDS_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 source freeze required')
    return capture_program({'edit_field': edit_field, 'literal_fields': literal_fields,
                            'repeat_field_reads': repeat_field_reads},
        scope_character_ids=None, scope_card_ids=None,
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))


def inputs():
    source = {'field_63': 63, 'hp': 0, 'é': 7, '汉字': 9, '': 11, 'e\u0301': 13}
    keys = ('field_63', 'hp', '汉字', 'é', 'e\u0301', '', 'field_00', 'missing', 'field_6', '汉')
    return [copy.deepcopy(source) for _ in keys], [list(key) for key in keys]


class RecordFieldTests(unittest.TestCase):
    @unittest.skipUnless(sys.version_info[:2] == (3, 10), 'CPython3.10 source freeze required')
    def test_dynamic_names_stay_inside_the_correct_record_pool(self):
        from app.modules.card_game.rl.rule_ir.portable_program import capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES
        from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
        frozen = capture_program({'two_record_pools': two_record_pools},
            scope_character_ids=None, scope_card_ids=None,
            allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))
        layout = compile_schema({
            'Game': dict(kind='record', capacity=8, fields={'hp': 'int', 'child': '@Child'}),
            'Child': dict(kind='record', capacity=8, fields={'hp': 'int', 'only_child': 'int', 'é': 'int'}),
        }, max_strings=32, max_string_bytes=1024)
        state = {'hp': 0, 'child': {'only_child': 5, 'hp': 7, 'é': 11}}
        keys = ('hp', 'child', 'only_child', 'é', 'missing', '')
        with NativeSoACPU(lower_program(frozen), layout, limits=VmLimits(steps=100000),
                          max_nodes=8192, max_edges=16384, max_payload_bytes=65536) as backend:
            call = backend.call('two_record_pools', pack_game_states([state]*len(keys), layout),
                                [[None, list(k)] for k in keys])
            self.assertEqual(call.decode(), [two_record_pools(state, list(k)) for k in keys])

    def check_backend(self, backend):
        states, parts = inputs()
        actual = backend.call('edit_field', pack_game_states(states, backend.schema),
                              [[None, p] for p in parts]).decode()
        for got, state, p in zip(actual, states, parts):
            want = edit_field(copy.deepcopy(state), p)
            self.assertEqual(got, want)
            self.assertEqual(list(got[0]), list(want[0]))
        call = backend.call('literal_fields', pack_game_states(states, backend.schema),
                            [[None] for _ in states])
        self.assertEqual(call.decode(), [literal_fields(state) for state in states])
        result = backend.call('repeat_field_reads', pack_game_states(states, backend.schema),
                              [[None, p, 8] for p in parts]).decode()
        self.assertEqual(result, [repeat_field_reads(s, p, 8) for s, p in zip(states, parts)])

    def test_native_dynamic_utf8_membership_reads_delete_and_reinsert(self):
        from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
        with NativeSoACPU(lower_program(program()), schema(),
                limits=VmLimits(steps=100000), max_nodes=8192, max_edges=16384,
                max_payload_bytes=65536) as backend:
            self.check_backend(backend)

    def test_actual_cuda_record_field_operations(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        with NativeSoACUDAQuery(lower_program(program()), schema(),
                limits=VmLimits(steps=100000), max_nodes=8192, max_edges=16384,
                max_payload_bytes=65536, seconds=15) as backend:
            self.check_backend(backend)


if __name__ == '__main__':
    unittest.main()
