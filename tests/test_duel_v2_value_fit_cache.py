"""Real tiny CPU value fitting; mocked collection/metrics are NOT GPU evidence."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from app.modules.card_game.rl import seven_round as rr,seven_runtime as rt
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.rl.batched_search.device_pure_segments import PureSegmentPaused


class ValueFitCheckpointTest(unittest.TestCase):
    def setUp(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.build_acceptance import canonical_build,SEED_PHASES
        from app.modules.card_game.rl.seven_lineup import identity
        torch.set_num_threads(1);torch.manual_seed(719)
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'source';self.out=Path(self.temp.name)/'value'
        preset={d['id']:d for d in STARTER_DECKS}
        self.decks={k:canonical_build(preset[k])for k in ROSTER}
        for key in ROSTER:rt.export(rt.create_network(key,16,'cpu'),self.source,key,self.decks[key],dict(cpu_fixture=True))
        state=new_game(seed=5,decks={'a':preset['starter'],'b':preset['starter']})
        _,(x,_) = rt.decision(state,'a')
        self.games=[dict(complete=True,winner='a',rows=[],value_rows=[
            dict(x=x.copy(),z=1.,side='a',key=key,value_actor=1,step=0),
            dict(x=(x+.001).astype(np.float32),z=-1.,side='b',key=key,value_actor=0,step=1)])for key in ROSTER]
        self.config=dict(value_fit_checkpoint=True,value_full_tower=True,value_replicas=1,value_check_replicas=1,
            value_metric_backend='numpy',device='cpu',recovery_source=str(self.source),rule_hash=identity(),
            seeds={name:10000+i*10000 for i,name in enumerate(SEED_PHASES)})
        self.deadline=time.time()+600

    def nets(self):
        restored={k:rt.restore(self.source,k,'cpu')for k in ROSTER}
        return{k:v[0]for k,v in restored.items()},{k:v[2]for k,v in restored.items()}

    def fit(self,collect,metrics):
        nets,origins=self.nets()
        with patch.object(rr,'collect',side_effect=collect),patch.object(rr,'value_metrics',side_effect=metrics):
            result=rr.warm_value(None,nets,self.decks,origins,self.source,self.out,
                self.config,0,self.deadline)
        return result,nets

    def collect(self,pool,jobs,folder):return deepcopy(self.games)

    @staticmethod
    def metrics(games,policies):
        return{k:dict(normal_games=8,brier=.2,log_loss=.3)for k in policies}

    def test_heldout_pause_new_process_uses_identical_bank_once_and_cached_metrics(self):
        first=[True]
        def paused(pool,jobs,folder):
            if Path(folder).name=='heldout'and first[0]:
                first[0]=False;raise PureSegmentPaused('synthetic-natural-prefix')
            return self.collect(pool,jobs,folder)
        original_update=rt.update
        with patch.object(rt,'update',wraps=original_update)as update:
            with self.assertRaises(PureSegmentPaused):self.fit(paused,self.metrics)
            self.assertEqual(update.call_count,7*8)
            marker=json.loads((self.out/'value-fit-checkpoint.json').read_text())
            original_bank=rr._value_fit_bank_hash(self.out/'models')
            self.assertEqual(set(marker['temperatures']),set(ROSTER))
            self.assertTrue(all(v['total_attempted']==8 and v['selected']==1 for v in marker['updates'].values()))
            (models,report),nets=self.fit(self.collect,self.metrics)
            self.assertEqual(update.call_count,7*8)
            self.assertEqual(rr._value_fit_bank_hash(models),original_bank)

            self.assertEqual(report['updates'],marker['updates'])
            self.assertTrue(report['passed'])
            for key,net in nets.items():
                model=rt.model(models,key)
                for name,value in net.state_dict().items():np.testing.assert_array_equal(value.detach().numpy(),model.weights[name])
            report_bytes=(self.out/'value-precheck.json').read_bytes()
            def no_new_metric(*args,**kwargs):raise AssertionError('Already committed metrics must not be regenerated')
            (_,again),_=self.fit(self.collect,no_new_metric)
            self.assertEqual(again,report);self.assertEqual(update.call_count,7*8)
            self.assertEqual((self.out/'value-precheck.json').read_bytes(),report_bytes)
            self.assertEqual(rr._value_fit_bank_hash(models),original_bank)

    def test_collection_only_pauses_before_optimizer_and_resumes_existing_evidence(self):
        self.config['value_fit_collect_only']=True
        with patch.object(rt,'update',side_effect=AssertionError('Collection boundary must not fit')):
            with self.assertRaises(rr.ValueFitReady)as pause:self.fit(self.collect,self.metrics)
        self.assertEqual(pause.exception.phase_games,dict(training=7,selection=7,heldout=0))
        self.assertFalse((self.out/'value-fit-start.json').exists())
        self.assertFalse((self.out/'value-fit-checkpoint.json').exists())
        self.config['value_fit_collect_only']=False
        (models,report),_=self.fit(self.collect,self.metrics)
        self.assertTrue(report['passed'])
        bank=rr._value_fit_bank_hash(models)
        self.config['value_fit_collect_only']=True
        with patch.object(rt,'update',side_effect=AssertionError('Committed fit must not repeat')):
            (again,_),_=self.fit(self.collect,self.metrics)
        self.assertEqual(rr._value_fit_bank_hash(again),bank)

    def test_source_evidence_model_and_incomplete_marker_refuse_refitting(self):
        def held_pause(pool,jobs,folder):
            if Path(folder).name=='heldout':raise PureSegmentPaused('fixture')
            return self.collect(pool,jobs,folder)
        with self.assertRaises(PureSegmentPaused):self.fit(held_pause,self.metrics)
        with patch.object(rt,'update',side_effect=AssertionError('must not refit')):
            previous=self.games[0]['value_rows'][0]['x'][0].copy()
            self.games[0]['value_rows'][0]['x'][0]+=1
            with self.assertRaisesRegex(ValueError,'source/seeds/build/protocol/evidence'):self.fit(self.collect,self.metrics)
            self.games[0]['value_rows'][0]['x'][0]=previous
            selected=self.out/'models/starter.npz';original=selected.read_bytes();selected.write_bytes(original+b'tampered')
            with self.assertRaises(ValueError):self.fit(self.collect,self.metrics)
            selected.write_bytes(original)
            path=self.source/'starter.json';path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'source/seeds/build/protocol/evidence'):self.fit(self.collect,self.metrics)
        self.out.joinpath('value-fit-checkpoint.json').unlink()
        with patch.object(rt,'update',side_effect=AssertionError('must not refit')),self.assertRaisesRegex(ValueError,'Incomplete/orphan'):
            self.fit(self.collect,self.metrics)

    def test_fit_interruption_is_not_inferred_successful_or_recounted(self):
        with patch.object(rt,'update',side_effect=RuntimeError('actual fit interrupted')):
            with self.assertRaisesRegex(RuntimeError,'fit interrupted'):self.fit(self.collect,self.metrics)
        self.assertTrue((self.out/'value-fit-start.json').exists())
        self.assertFalse((self.out/'value-fit-checkpoint.json').exists())
        with patch.object(rt,'update',side_effect=AssertionError('must not replay')),self.assertRaisesRegex(ValueError,'Incomplete/orphan'):
            self.fit(self.collect,self.metrics)


    def test_disabling_flag_cannot_wash_an_incomplete_persistent_fit(self):
        with patch.object(rt,'update',side_effect=RuntimeError('first actual fit interrupted')):
            with self.assertRaises(RuntimeError):self.fit(self.collect,self.metrics)
        self.config['value_fit_checkpoint']=False
        with patch.object(rt,'update')as update,self.assertRaisesRegex(ValueError,'cannot be bypassed'):
            self.fit(self.collect,self.metrics)
        update.assert_not_called()

    def test_head_only_checkpoint_reuses_uncalibrated_selection_identity(self):
        self.config.update(value_full_tower=False,value_passes=1)
        def paused(pool,jobs,folder):
            if Path(folder).name=='heldout':raise PureSegmentPaused('head-only fixture')
            return self.collect(pool,jobs,folder)
        with patch.object(rt,'update',wraps=rt.update)as update:
            with self.assertRaises(PureSegmentPaused):self.fit(paused,self.metrics)
            original=rr._value_fit_bank_hash(self.out/'models')
            self.assertEqual(update.call_count,7)
            (_,report),_=self.fit(self.collect,self.metrics)
            self.assertEqual(update.call_count,7)
            self.assertEqual(rr._value_fit_bank_hash(self.out/'models'),original)
            self.assertEqual(report['selected_epochs'],{})
            self.assertTrue(all(v['total_attempted']==v['selected']==1 for v in report['updates'].values()))

    def test_default_one_shot_still_refits_without_creating_fit_markers(self):
        self.config.update(value_fit_checkpoint=False,value_full_tower=False,value_passes=1)
        with patch.object(rt,'update',wraps=rt.update)as update:
            self.fit(self.collect,self.metrics);self.assertEqual(update.call_count,7)
            self.fit(self.collect,self.metrics);self.assertEqual(update.call_count,14)
        self.assertFalse((self.out/'value-fit-start.json').exists())
        self.assertFalse((self.out/'value-fit-checkpoint.json').exists())


    def test_numeric_batch_and_temperature_pauses_resume_exact_adam_once(self):
        # Compare the real existing one-shot learner to staged numeric recovery.
        (_,baseline),_=self.fit(self.collect,self.metrics)
        expected=rr._value_fit_bank_hash(self.out/'models')
        self.out=Path(self.temp.name)/'incremental'
        self.config.update(value_fit_process_seconds=150,value_fit_resume_authorization='explicit numeric continuation')
        original_admit=rr._ValueFitProcess._admit
        def before_first_temperature(session,label,reserve=5.):
            if label.startswith('temperature-'):session._pause()
            return original_admit(session,label,reserve)
        def after_first_temperature(session,label,reserve=5.):
            if label.startswith('temperature-')and session.state['temperature']==1:session._pause()
            return original_admit(session,label,reserve)
        with patch.object(rt,'update',wraps=rt.update)as update:
            with patch.object(rr._ValueFitProcess,'_admit',before_first_temperature),self.assertRaises(rr.ValueFitPaused):
                self.fit(self.collect,self.metrics)
            self.assertEqual(update.call_count,1)
            first=json.loads((self.out/'value-fit-process/head.json').read_text())
            self.assertEqual(first['status'],'READY')
            with patch.object(rr._ValueFitProcess,'_admit',after_first_temperature),self.assertRaises(rr.ValueFitPaused):
                self.fit(self.collect,self.metrics)
            self.assertEqual(update.call_count,1)
            def after_first_held_metric(session,label,reserve=5.):
                if label=='heldout-weave-rush':session._pause()
                return original_admit(session,label,reserve)
            with patch.object(rr._ValueFitProcess,'_admit',after_first_held_metric),self.assertRaises(rr.ValueFitPaused):
                self.fit(self.collect,self.metrics)
            self.assertEqual(update.call_count,56)
            saved=json.loads((self.out/'value-fit-process/head.json').read_text())
            from app.modules.card_game.rl.batched_search.pure_checkpoint import read_payload
            payload=read_payload(self.out/'value-fit-process'/saved['checkpoint'],saved['contract'],kind='main_value_fit')
            self.assertEqual(payload['metric_cursor'],1)
            self.assertEqual(set(payload['metrics']),{'starter'})
            (_,report),_=self.fit(self.collect,self.metrics)
            self.assertEqual(update.call_count,56)
            self.assertEqual(rr._value_fit_bank_hash(self.out/'models'),expected)
            self.assertEqual(report['updates'],baseline['updates'])
            self.assertEqual(report['metrics'],baseline['metrics'])
            self.assertFalse((self.out/'value-fit-process/RUNNING.json').exists())
            self.assertFalse((self.out/'value-fit-process/REJECTED.json').exists())
            self.fit(self.collect,lambda *a,**k:(_ for _ in()).throw(AssertionError('completed metrics replay')))
            self.assertEqual(update.call_count,56)


    def test_numeric_running_or_rejected_batch_cannot_replay(self):
        self.config.update(value_fit_process_seconds=150)
        original=rt.update
        calls=[]
        def interrupted(*args,**kwargs):
            calls.append(1);original(*args,**kwargs);raise RuntimeError('actual Adam step then lost checkpoint')
        with patch.object(rt,'update',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'Adam step'):
            self.fit(self.collect,self.metrics)
        self.assertEqual(calls,[1])
        self.assertTrue((self.out/'value-fit-process/RUNNING.json').is_file())
        self.assertTrue((self.out/'value-fit-process/REJECTED.json').is_file())
        with patch.object(rt,'update')as update,self.assertRaisesRegex(ValueError,'Uncertain/rejected'):
            self.fit(self.collect,self.metrics)
        update.assert_not_called()

    def test_numeric_checkpoint_arrays_tampering_is_rejected_before_update(self):
        self.config.update(value_fit_process_seconds=150)
        original=rr._ValueFitProcess._admit
        def paused(session,label,reserve=5.):
            if label.startswith('temperature-'):session._pause()
            return original(session,label,reserve)
        with patch.object(rr._ValueFitProcess,'_admit',paused),self.assertRaises(rr.ValueFitPaused):
            self.fit(self.collect,self.metrics)
        head=json.loads((self.out/'value-fit-process/head.json').read_text())
        array=self.out/'value-fit-process'/head['checkpoint']/'arrays.npz'
        array.write_bytes(array.read_bytes()+b'changed')
        with patch.object(rt,'update')as update,self.assertRaisesRegex(ValueError,'numerical integrity'):
            self.fit(self.collect,self.metrics)
        update.assert_not_called()


if __name__=='__main__':unittest.main()
