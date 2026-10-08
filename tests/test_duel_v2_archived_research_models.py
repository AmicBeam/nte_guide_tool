"""The missing five-cross numeric models are archived without serving approval."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'app/modules/card_game/engine/ai/models'
ARCHIVE = MODELS / 'research/five-cross-20260924'
EXPECTED = {
    'zhenhong': '3c1258b60d9e985a30da55b6d98bc6526064c450a0292ece64c378ebc7f428e1',
    'murk': '744b4b9d71cca135df6f84ecb0479979cfc1ae6e922e8f62a535a97f902d9f51',
}


class ArchivedResearchModelsTest(unittest.TestCase):
    def test_missing_models_preserve_identity_and_stay_out_of_serving_root(self):
        for key, expected in EXPECTED.items():
            with self.subTest(key=key):
                manifest = json.loads((ARCHIVE / f'{key}.json').read_text(encoding='utf-8'))
                digest = hashlib.sha256((ARCHIVE / f'{key}.npz').read_bytes()).hexdigest()
                self.assertEqual(digest, expected)
                self.assertEqual(manifest['sha256'], expected)
                self.assertEqual(manifest['schema'], 'cross_five_grounded_wdl_v1')
                self.assertIs(manifest['automatic_serving_approval'], False)
                self.assertFalse((MODELS / f'{key}.json').exists())
                self.assertFalse((MODELS / f'{key}.npz').exists())


if __name__ == '__main__':
    unittest.main()
