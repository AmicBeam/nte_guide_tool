"""Explicit freeze/adapter binding; no CUDA performance qualification."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from app.modules.card_game.rl.batched_search.device_soa_rules import _value_observation_path


class ValueObservationEntryTests(unittest.TestCase):
    def test_old_images_and_unknown_value_entry(self):
        image=SimpleNamespace(entries={},function_ids=())
        self.assertEqual(_value_observation_path(image),('public_legal_decision',(1,)))
        for index,names in ((True,('bad',)),(-1,('bad',)),(0,('bad',)),(1,('bad',))):
            image.entries={'value_observation':index};image.function_ids=names
            with self.assertRaisesRegex(ValueError,'Exact frozen'):
                _value_observation_path(image)
        image.entries={'value_observation':0}
        image.function_ids=('func:app.modules.card_game.rl.batched_duel.gpu_value_projection:cached_policy_value_observation:hash',)
        self.assertEqual(_value_observation_path(image),('value_observation',()))

    @unittest.skipUnless(sys.version_info[:2]==(3,10),'Portable freeze requires CPython 3.10')
    def test_real_freeze_loader_and_protocol_binding(self):
        from app.modules.card_game.rl.batched_duel.native_array_program import (
            capture_native_array_program,load_native_array_program)
        from app.modules.card_game.rl.rule_ir.portable_program import compute_identity,ValidationError
        with self.assertRaises(TypeError):capture_native_array_program(value_observation_entry=1)
        with self.assertRaisesRegex(ValueError,'indexed'):
            capture_native_array_program(value_observation_entry=True)
        program=capture_native_array_program(policy_query_cache=True,value_observation_entry=True)
        self.assertIn('value_observation',program.entries)
        self.assertTrue(program.data['native_array']['value_observation_entry'])
        self.assertIn('app/modules/card_game/rl/batched_duel/gpu_value_projection.py',program.data['source_hashes'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'program.json';path.write_text(program.to_json())
            self.assertEqual(load_native_array_program(path).identity,program.identity)
            for bad in (False,1):
                program.data['native_array']['value_observation_entry']=bad
                program.data['identity']=compute_identity(program.data);path.write_text(program.to_json())
                with self.assertRaises(ValidationError):load_native_array_program(path)


if __name__=='__main__':unittest.main()
