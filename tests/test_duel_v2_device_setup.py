"""Explicit GPU canonical setup parity; zero complete games/training evidence."""
from copy import deepcopy
from pathlib import Path
import os
import sys
import unittest


class DeviceSetupCudaTest(unittest.TestCase):
    def test_canonical_seed_entities_decks_mulligan_and_public_board(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required for canonical setup')
        from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.batched_search.evaluator import runtime_matrix_keys
        from app.modules.card_game.rl.batched_duel.gpu_rule_program import capture_gpu_rule_program,verify_source_hashes
        from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        fixture=os.environ.get('NTE_GPU_DEVICE_SETUP_PROGRAM')
        if fixture:
            program=FrozenRuleProgram.from_json(Path(fixture).read_text());verify_source_hashes(program)
        elif sys.version_info[:2]==(3,10):
            program=capture_gpu_rule_program(['initialize'],runtime='seven')
        else:
            self.fail('Explicit CPython3.10 frozen canonical setup fixture required')
        keys=runtime_matrix_keys('seven');parameters=[]
        decks={deck['id']:deck for deck in STARTER_DECKS}
        for index,key in enumerate(keys):
            for skip in (False,True):
                parameters.append(dict(seed=20261011+index,decks={'a':deepcopy(decks[key]),
                    'b':deepcopy(decks[keys[(index+1)%len(keys)]])},first_side='a' if skip else 'b',
                    skip_mulligan=skip,escalation=True))
        expected=[new_game(**row) for row in parameters]
        with CudaVm(lower_program(program),guard_standalone_entities=True) as vm:
            self.assertTrue(vm.capability('initialize')['ready'])
            self.assertIn('uuid.uuid4',vm.capability('initialize')['runtime_denied_intrinsics'])
            arena=vm.ingest(pack_states(parameters,GraphLayout(len(parameters),262144,524288,32*1024*1024)))
            actual=vm.launch(arena,'initialize').decode()
        self.assertEqual(actual,expected)
        for row in actual:
            from app.modules.card_game.engine.duel_v2.entities import PlayerEntity,CharacterEntity,CardEntity
            for side in ('a','b'):
                self.assertIsInstance(row['sides'][side],PlayerEntity)
                self.assertTrue(all(isinstance(hero,CharacterEntity) for hero in row['sides'][side]['characters'].values()))
                self.assertTrue(all(isinstance(card,CardEntity) for card in row['sides'][side]['deck']))
