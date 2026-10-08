"""Independent CUDA execution parity for source-level collection semantics."""
import subprocess,sys,unittest
from app.modules.card_game.rl.batched_duel.vm_collections import SUPPORTED_COLLECTION_OPS,emit_collection_source

class CollectionImportTest(unittest.TestCase):
    def test_import_without_torch(self):
        subprocess.run([sys.executable,'-c',"import sys;from app.modules.card_game.rl.batched_duel import vm_collections;assert 'torch' not in sys.modules"],check=True,capture_output=True)
    def test_handler_does_not_claim_hash_iteration_parity(self):
        from app.modules.card_game.rl.batched_duel.vm_collections import IMPLEMENTATION_STATUS
        self.assertIn('NOT implemented',IMPLEMENTATION_STATUS['set_semantics'])
        self.assertTrue(SUPPORTED_COLLECTION_OPS)
        self.assertIn('handle_collections',emit_collection_source())

if __name__=='__main__':unittest.main()
