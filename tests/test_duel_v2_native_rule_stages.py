"""Real rule helpers on native control flow; not a full-game/search test."""
import copy
import os
from pathlib import Path
import sys
import unittest

from app.modules.card_game.engine.duel_v2.state import (
    hero, alive, collapsed, finished, clear_harmony_source,
)
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.domain_soa_records import compile_execution_schema
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU
from app.modules.card_game.rl.batched_duel.record_soa import pack_game_states
from app.modules.card_game.rl.batched_duel.vm_image import lower_program


def real_rule_stage(state, side, cid):
    character = hero(state, side, cid)
    status = alive(state, side, cid), collapsed(character), finished(state)
    clear_harmony_source(state, side, cid)
    return state, character, status


def fixture_program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        FrozenRuleProgram, capture_program, DEFAULT_ALLOWED_MODULE_PREFIXES)
    path = os.environ.get('NTE_NATIVE_RULE_STAGES_PROGRAM')
    if path:
        return FrozenRuleProgram.from_json(Path(path).read_text(encoding='utf8'))
    if sys.version_info[:2] != (3, 10):
        raise unittest.SkipTest('CPython3.10 frozen fixture required')
    return capture_program({'stage': real_rule_stage}, scope_character_ids=None,
        scope_card_ids=None, allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,))


def fixture_cases():
    from app.modules.card_game.engine.duel_v2 import new_game
    cases = []
    for i in range(2):
        state = new_game(seed=71+i, skip_mulligan=True)
        side = 'a' if not i else 'b'
        cid = state['sides'][side]['order'][0]
        state['sides'][side]['last_front'] = cid if not i else None
        cases.append(dict(state=state, side=side, cid=cid))
    return cases


def assert_stage(backend, cases):
    schema = backend.schema
    result = backend.call('stage', pack_game_states([c['state'] for c in cases], schema),
                          [[None, c['side'], c['cid']] for c in cases]).decode()
    expected = [real_rule_stage(copy.deepcopy(c['state']), c['side'], c['cid']) for c in cases]
    if result != expected:
        raise AssertionError('Real rule stage state/result mismatch')
    for (state, character, _), case in zip(result, cases):
        if character is not state['sides'][case['side']]['characters'][case['cid']]:
            raise AssertionError('Real rule stage alias mismatch')


class NativeRuleStageTests(unittest.TestCase):
    def test_real_rule_stage_native_cpu(self):
        image, schema = lower_program(fixture_program()), compile_execution_schema()
        with NativeSoACPU(image, schema, native_functions=True, native_values=True,
                limits=VmLimits(steps=100000), max_nodes=16384,
                max_edges=32768, max_payload_bytes=131072) as backend:
            assert_stage(backend, fixture_cases())

    def test_real_rule_stage_actual_cuda(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.native_soa_cuda_query import NativeSoACUDAQuery
        image, schema = lower_program(fixture_program()), compile_execution_schema()
        with NativeSoACUDAQuery(image, schema, native_functions=True, native_values=True, execute_only=True,
                limits=VmLimits(steps=100000), seconds=10, max_nodes=16384,
                max_edges=32768, max_payload_bytes=131072) as backend:
            assert_stage(backend, fixture_cases())


if __name__ == '__main__':
    unittest.main()
