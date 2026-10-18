"""Tiny CPU AdamW fixtures with mocked CUDA/proofs; never GPU qualification."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from app.modules.card_game.rl import device_auxiliary_calibration as fit
from app.modules.card_game.rl.seven_schedule import ROSTER
from tests import test_duel_v2_device_auxiliary_calibration as fixtures


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy=torch.nn.Linear(1,1)
        self.value_net=torch.nn.Linear(1,1)
        self.wdl=torch.nn.Linear(1,3)


class AuxiliaryFitResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.AuxiliaryCalibrationIngressTest.setUpClass()

    def environment(self, folder, *, pause_update=False, broken=False):
        fixture=fixtures.AuxiliaryCalibrationIngressTest();source=folder/'source';source.mkdir()
        (source/'frozen.json').write_text('{"fixture":true}')
        torch.manual_seed(4);original={k:deepcopy(Tiny().state_dict()) for k in ROSTER}
        counts=dict(update=0,proof=0); clock=[0.]
        def restore(directory,key,device):
            net=Tiny()
            if Path(directory).resolve()==source.resolve(): state=original[key]
            else:
                with np.load(Path(directory)/(key+'.npz'),allow_pickle=False) as data:
                    state={n:torch.from_numpy(data[n].copy()) for n in data.files}
            net.load_state_dict(state)
            return net,fixture.decks[key],dict(fixture=True)
        def export(net,directory,key,build,origin):
            directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/(key+'.npz'),**{n:t.detach().numpy() for n,t in net.state_dict().items()})
            (directory/(key+'.json')).write_text(json.dumps(origin,sort_keys=True))
            self.assertTrue(torch.equal(net.policy.weight,original[key]['policy.weight']))
        def update(net,opt,rows,value_only):
            self.assertTrue(value_only);counts['update']+=1
            opt.zero_grad();net.wdl(net.value_net(torch.ones(1,1))).square().sum().backward();opt.step()
            if broken: raise RuntimeError('uncertain update')
            if pause_update and counts['update']==1: clock[0]+=145.
        def proof(segment,**kwargs):
            counts['proof']+=1
            return fixture.exported(segment,**kwargs)
        def models(directory,key):return SimpleNamespace(serving_deck=fixture.decks[key])
        def metrics(games,models,**kwargs):
            return {k:dict(normal_games=8,brier=.4,log_loss=.8) for k in models}
        def scope(directory):return {},json.loads((Path(directory)/'scope.json').read_text())
        stack=ExitStack()
        for target,value in (
            ('torch.cuda.is_available',lambda:True),
            ('torch.cuda.get_rng_state_all',lambda:[]),('torch.cuda.set_rng_state_all',lambda state:None),
            ('app.modules.card_game.rl.device_auxiliary_calibration.time.monotonic',lambda:clock[0]),
            ('app.modules.card_game.rl.device_auxiliary_collection.verified_auxiliary_games',proof),
            ('app.modules.card_game.rl.seven_runtime.restore',restore),
            ('app.modules.card_game.rl.seven_runtime.export',export),
            ('app.modules.card_game.rl.seven_runtime.model',models),
            ('app.modules.card_game.rl.seven_runtime.update',update),
            ('app.modules.card_game.rl.seven_round.value_metrics',metrics),
            ('app.modules.card_game.rl.seven_round.versions',lambda directory:{k:'fixture' for k in ROSTER}),
            ('app.modules.card_game.rl.seven_covered_value.load_scope',scope)):
            stack.enter_context(patch(target,side_effect=value))
        kwargs=dict(source=source,output=folder/'fit',checkpoint_dir=folder/'cache',
            starting_decks=fixture.decks,seed_reservation=fixture.seeds,
            phase_segments=fixture.phases(),deadline=fixture.deadline,process_seconds=150.)
        return stack,kwargs,counts,clock

    def test_resume_does_not_repeat_proofs_or_adamw_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            stack,kwargs,counts,clock=self.environment(Path(temp),pause_update=True)
            with stack:
                with self.assertRaises(fit.AuxiliaryFitPaused):fit.fit_device_auxiliary_calibration(**kwargs)
                self.assertEqual(counts,dict(proof=3,update=1))
                self.assertFalse((kwargs['checkpoint_dir']/'RUNNING.json').exists())
                clock[0]=0
                _,report=fit.fit_device_auxiliary_calibration(**kwargs,resume=True)
                self.assertEqual(counts,dict(proof=3,update=56))
                self.assertEqual(report['auxiliary_optimizer_updates'],56)
                self.assertTrue(report['passed']);self.assertEqual(report['real_learner_updates'],0)
                fit.fit_device_auxiliary_calibration(**kwargs,resume=True)
                self.assertEqual(counts,dict(proof=3,update=56))
            expected={}
            for key in ROSTER:
                with np.load(kwargs['output']/'models'/(key+'.npz')) as arrays:
                    expected[key]={n:arrays[n].copy() for n in arrays.files}
            other=Path(temp)/'uninterrupted';other.mkdir()
            stack,kwargs,counts,_=self.environment(other)
            with stack:
                fit.fit_device_auxiliary_calibration(**kwargs)
            self.assertEqual(counts,dict(proof=3,update=56))
            for key in ROSTER:
                with np.load(kwargs['output']/'models'/(key+'.npz')) as arrays:
                    for name,value in expected[key].items():np.testing.assert_array_equal(value,arrays[name])

    def test_uncertain_update_is_sticky(self):
        with tempfile.TemporaryDirectory() as temp:
            stack,kwargs,counts,_=self.environment(Path(temp),broken=True)
            with stack:
                with self.assertRaisesRegex(RuntimeError,'uncertain'):fit.fit_device_auxiliary_calibration(**kwargs)
                self.assertTrue((kwargs['checkpoint_dir']/'REJECTED.json').is_file())
                with self.assertRaisesRegex(ValueError,'uncertain'):
                    fit.fit_device_auxiliary_calibration(**kwargs,resume=True)
                self.assertEqual(counts['update'],1)

    def test_output_raw_binding_and_deadline_extension_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            stack,kwargs,counts,clock=self.environment(Path(temp),pause_update=True)
            with stack:
                with self.assertRaises(fit.AuxiliaryFitPaused):fit.fit_device_auxiliary_calibration(**kwargs)
                clock[0]=0
                with self.assertRaisesRegex(ValueError,'authorized'):
                    fit.fit_device_auxiliary_calibration(**dict(kwargs,deadline=kwargs['deadline']+100),resume=True)
                from app.modules.card_game.rl import seven_runtime as rt
                original_update=rt.update
                def pause_next(*args,**options):
                    result=original_update(*args,**options);clock[0]+=145.;return result
                extended=dict(kwargs,deadline=kwargs['deadline']+100)
                with patch.object(rt,'update',side_effect=pause_next):
                    with self.assertRaises(fit.AuxiliaryFitPaused):
                        fit.fit_device_auxiliary_calibration(**extended,resume=True,
                            resume_authorization='explicit fixture deadline extension')
                self.assertEqual(counts['update'],2)
                self.assertEqual(len(json.loads((kwargs['checkpoint_dir']/'head.json').read_text())['budgets']),1)
                clock[0]=0
                (kwargs['output']/'training-0-private.json.gz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'output changed'):
                    fit.fit_device_auxiliary_calibration(**extended,resume=True)
                self.assertEqual(counts['update'],2)

    def test_source_tamper_and_concurrent_resume_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            stack,kwargs,counts,clock=self.environment(Path(temp),pause_update=True)
            with stack:
                with self.assertRaises(fit.AuxiliaryFitPaused):fit.fit_device_auxiliary_calibration(**kwargs)
                clock[0]=0
                claim=kwargs['checkpoint_dir'].with_name('cache.fit-owner');claim.write_text('other worker')
                with self.assertRaisesRegex(ValueError,'active or uncertain'):
                    fit.fit_device_auxiliary_calibration(**kwargs,resume=True)
                claim.unlink()
                (kwargs['source']/'frozen.json').write_text('changed')
                with self.assertRaisesRegex(ValueError,'binding changed'):
                    fit.fit_device_auxiliary_calibration(**kwargs,resume=True)
                self.assertEqual(counts['update'],1)

    def test_ledger_proofs_are_loaded_only_at_admitted_cohort_boundary(self):
        # Mock ledger bytes and replay proof explicitly; this only verifies
        # scheduling and must never qualify this synthetic data for CUDA use.
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);stack,kwargs,counts,clock=self.environment(folder)
            fixture=fixtures.AuxiliaryCalibrationIngressTest(); phases=fixture.phases()
            ledger=object.__new__(fit.DeviceAuxiliaryCalibrationLedger)
            ledger.contract=dict(physical_plans={p:[fit._physical_job(j) for j in jobs]
                for p,jobs in fixture.plans.items()},source_model_versions={k:'fixture' for k in ROSTER},
                source_binding=phases['training'][0].source_binding,
                seed_reservation=fixture.seeds,starting_decks=fixture.decks)
            ledger.contract_sha256='fixture';ledger._manifest_file_sha256='fixture'
            entries=[]
            for phase,jobs in fixture.plans.items():
                path=folder/(phase+'-original');path.write_bytes(b'bound original bytes')
                entries.append(dict(phase=phase,offset=0,count=len(jobs),status='COMPLETE',
                    checkpoints=[dict(path=path.name,sha256=sha256(path.read_bytes()).hexdigest())]))
            ledger.manifest=dict(cohorts=entries)
            ledger._assert_current=lambda:None
            ledger._checkpoint_path=lambda name:folder/name
            def load(entry,**options):
                self.assertFalse(options['verify']);return phases[entry['phase']][0]
            with stack, patch.object(ledger,'completed_phase_segments',side_effect=AssertionError('unbounded proof')) as all_games, \
                    patch.object(ledger,'_load_segment',side_effect=load) as loaded, \
                    patch.object(ledger,'_clean',return_value=True), \
                    patch('app.modules.card_game.rl.device_auxiliary_continuation.dump_auxiliary_checkpoint',
                          side_effect=lambda segment,path:Path(path).write_bytes(b'synthetic archived fixture')):
                original=fixture.exported
                def bounded_proof(segment,**options):
                    counts['proof']+=1;clock[0]+=145.;return original(segment,**options)
                with patch('app.modules.card_game.rl.device_auxiliary_collection.verified_auxiliary_games',
                           side_effect=bounded_proof):
                    with self.assertRaises(fit.AuxiliaryFitPaused):
                        fit.fit_device_auxiliary_calibration(**dict(kwargs,phase_segments=None,ledger=ledger))
                all_games.assert_not_called();self.assertEqual(loaded.call_count,1)
                self.assertEqual(counts,dict(proof=1,update=0))

    def test_budget_and_option_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            stack,kwargs,_,_=self.environment(Path(temp))
            with stack:
                with self.assertRaisesRegex(ValueError,'process_seconds'):
                    fit.fit_device_auxiliary_calibration(**dict(kwargs,process_seconds=151))
                with self.assertRaisesRegex(ValueError,'checkpoint_dir'):
                    fit.fit_device_auxiliary_calibration(**dict(kwargs,checkpoint_dir=None))


if __name__=='__main__':unittest.main()
