"""Authenticated seven-to-seven research migration, without readiness inheritance."""
from pathlib import Path
from hashlib import sha256
import json,shutil,tempfile,unittest
from app.modules.card_game.rl import seven_runtime as rt
from app.modules.card_game.rl.seven_source import migrate_seven
from app.modules.card_game.rl.next_training import plan
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'Optional training Torch required')
class SevenMigrationTest(unittest.TestCase):
    def test_all_tensors_preserved_and_authenticated_source_changes_rejected(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';module=repo/'app/modules/card_game';manifest={}
            paths={*(module/'engine/duel_v2').rglob('*.py'),*(module/'content/duel_v2').rglob('*.py'),module/'content/duel_v2/catalog.json'}
            for name in ('league_schema.py','league_observation.py','seven_lineup.py','seven_grounded.py','seven_runtime.py','seven_policy.py','murk_lineup.py','information_search.py','search_policy.py'):
                paths.add(module/'rl'/name)
            for path in paths:
                name=path.relative_to(repo).as_posix();data=path.read_bytes();target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                manifest[name]=dict(sha256=sha256(data).hexdigest(),bytes=len(data))
            (source/'source-manifest.json').write_text(json.dumps(manifest))
            key='collapse';build=plan()['builds'][key];initial=root/'initial';output=root/'migrated'
            net=rt.create_network(key,16,'cpu');rt.export(net,initial,key,build,dict(synthetic_fixture=True))
            weights=initial/(key+'.npz');metadata=initial/(key+'.json')
            kwargs=dict(source_snapshot=source,expected_sha256=sha256(weights.read_bytes()).hexdigest(),expected_manifest_sha256=sha256(metadata.read_bytes()).hexdigest(),device='cpu')
            migrated,origin=migrate_seven(initial,key,build,output,**kwargs)
            self.assertTrue(all(torch.equal(w,migrated.state_dict()[n]) for n,w in net.state_dict().items()))
            receipt=origin['research_migration']
            self.assertFalse(receipt['prior_ready_adopted']);self.assertFalse(receipt['prior_quality_adopted'])
            self.assertTrue(receipt['optimizer_reset']);self.assertTrue(receipt['experience_reset'])
            target=source/'app/modules/card_game/engine/duel_v2/modifiers.py'
            target.write_bytes(target.read_bytes()+b'\n# changed source\n')
            with self.assertRaises(ValueError):migrate_seven(initial,key,build,root/'rejected',**kwargs)
            self.assertFalse((root/'rejected').exists())

if __name__=='__main__':unittest.main()
