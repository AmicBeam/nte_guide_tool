import hashlib,json,tempfile,unittest,shutil
from pathlib import Path
from app.modules.card_game.rl import seven_covered_value as scope

class ComputationBindingTest(unittest.TestCase):
    def test_sampler_change_is_separate_but_value_target_change_is_not(self):
        root=Path(scope.__file__).parent
        with tempfile.TemporaryDirectory()as directory:
            target=Path(directory)
            names=('seven_covered_value.py','covered_rollout.py','policies.py','seven_auxiliary_jobs.py',
                   'seven_runtime.py','outcome_runtime.py','seven_policy.py','seven_lineup.py',
                   'seven_grounded.py','information_search.py','seven_sampler.py')
            for name in names:shutil.copy2(root/name,target/name)
            original=scope.computation_identity(target)
            (target/'seven_sampler.py').write_text('different world sampling')
            self.assertEqual(scope.computation_identity(target),original)
            p=target/'seven_covered_value.py';p.write_text(p.read_text().replace('if step>=6:','if step>=7:'))
            self.assertNotEqual(scope.computation_identity(target),original)

    def test_receipt_binds_immutable_report_and_exact_current_computation(self):
        with tempfile.TemporaryDirectory()as directory:
            folder=Path(directory);report=dict(algorithm_sha256='old',rule_hash='rules',models={'starter':'weights'})
            (folder/'scope.json').write_text(json.dumps(report));root=Path(scope.__file__).parent
            receipt=dict(schema='covered_value_unchanged_computation_v1',original_report_sha256=hashlib.sha256((folder/'scope.json').read_bytes()).hexdigest(),original_algorithm_sha256='old',rule_hash='rules',models=report['models'],value_computation_sha256=scope.computation_identity(),original_value_computation_sha256=scope.computation_identity(),accepted_sampler_sha256=hashlib.sha256((root/'seven_sampler.py').read_bytes()).hexdigest())
            path=folder/'value-computation-compatibility.json';path.write_text(json.dumps(receipt));self.assertTrue(scope._algorithm_binding_matches(folder,report))
            for field in ('accepted_sampler_sha256','value_computation_sha256','original_value_computation_sha256','original_report_sha256'):
                changed=dict(receipt,**{field:'wrong'});path.write_text(json.dumps(changed));self.assertFalse(scope._algorithm_binding_matches(folder,report))
            path.write_text(json.dumps(receipt));(folder/'scope.json').write_text(json.dumps(dict(report,passed=True)))
            self.assertFalse(scope._algorithm_binding_matches(folder,report))

class CollectorSourceBindingTest(unittest.TestCase):
    def test_sampler_source_change_invalidates_future_collection_contract(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.batched_search.training_checkpoint import collection_contract
        original=collection_contract([],{},{});read=Path.read_bytes
        def changed(path):
            content=read(path)
            return content+b'\n# changed sampler contract'if path.name=='seven_sampler.py'else content
        with patch.object(Path,'read_bytes',changed):self.assertNotEqual(collection_contract([],{},{}),original)
