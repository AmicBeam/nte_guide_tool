"""Synthetic CPU archives/oracles only; never historical-data or CUDA evidence."""
from copy import deepcopy
from dataclasses import replace
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl.build_acceptance import canonical_build,SEED_PHASES
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.rl.seven_auxiliary_jobs import auxiliary_jobs
from app.modules.card_game.rl import seven_auxiliary_reattestation as ar


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def oracle():
    def new_game(**kwargs):return dict(step=0,phase='playing')
    def apply(state,actor,action):
        step=state['step']+1
        return dict(step=step,phase='finished'if step==8 else'playing',winner='a',reason='player_health')
    def observe(state,side,**kwargs):
        return dict(step=state['step'],side=side,legal_actions=[dict(action=dict(type='end_turn'))])
    def actor(state):return'a'if state['step']%2==0 else'b'
    def encode(view,actions):return np.array([view['step'],int(view['side']=='b')],np.float32),None
    return new_game,apply,observe,actor,lambda v:v,encode,lambda s,a:([dict(type='end_turn')],0),lambda v,a:0


def fixture_game(job):
    return dict(complete=True,seed=job['seed'],first=job['first'],rows=[],winner='a',
        actions=[dict(side='a'if i%2==0 else'b',action=dict(type='end_turn'))for i in range(8)],
        value_rows=[dict(step=i,side=s,key=job['keys'][s],value_actor=int((i%2==0)==(s=='a')),
            x=[i,int(s=='b')],z=1 if s=='a'else-1)for i in range(6,8)for s in ('a','b')])


class SevenReattestationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'source';self.source.mkdir()
        decks={d['id']:d for d in STARTER_DECKS}
        self.starts={k:canonical_build(decks[k])for k in ROSTER}
        seeds={k:10000+i*10000 for i,k in enumerate(SEED_PHASES)}
        counts=dict(training=168,selection=56,heldout=56)
        config=dict(seeds=seeds,phase_counts=counts,deadline=5000000000.)
        write(self.source/'config.json',config);files={}
        for phase in counts:
            plan=auxiliary_jobs(seeds[ar._BASES[phase]],counts[phase]//56,decks=self.starts,deadline=config['deadline']-30)
            name=phase+'.json';write(self.source/name,dict(complete=True,games=len(plan),
                seeds=[j['seed']for j in plan],first=[j['first']for j in plan]))
            files[name]=ar._digest(self.source/name)
            for i,job in enumerate(plan):
                name=f'{phase}-{i}-private.json.gz'
                with gzip.open(self.source/name,'wt')as stream:json.dump(fixture_game(job),stream)
                files[name]=ar._digest(self.source/name)
        models={}
        for key in ROSTER:
            write(self.source/'models'/(key+'.json'),dict(fixture=True))
            array=self.source/'models'/(key+'.npz');array.write_bytes(('NOT WEIGHTS '+key).encode())
            models[key]=ar._digest(array)
        scope=dict(schema='seven_expert_value_v2',passed=True,phase_counts=counts,
            objective='visible_rule_after_six_exploration_actions',real_learner_updates=0,
            primary_policy_arrays_unchanged=True,models=models,files=files,rule_hash='old',
            algorithm_sha256='old',choices={k:dict(epoch=2,temperature=4.)for k in ROSTER})
        write(self.source/'scope.json',scope)
        self.plan=ar.prepare_reattestation(self.source,starting_decks=self.starts)
        self.deadline=time.time()+100

    def test_exact_rows_and_current_observation_drift_reject(self):
        job=self.plan.jobs[0][2];game=fixture_game(job)
        with patch.object(ar,'_oracle',side_effect=oracle):
            self.assertEqual(ar._verify_game(game,job,deadline=self.deadline),(8,4))
            for mutate in (lambda g:g['value_rows'].append(deepcopy(g['value_rows'][0])),
                           lambda g:g['value_rows'].pop(),
                           lambda g:g['value_rows'][0]['x'].__setitem__(0,21),
                           lambda g:g['value_rows'][0].__setitem__('z',0),
                           lambda g:g['actions'][0]['action'].__setitem__('type','concede')):
                bad=deepcopy(game);mutate(bad)
                with self.assertRaises(ValueError):ar._verify_game(bad,job,deadline=self.deadline)

    def test_bounded_shard_and_incomplete_finalization_never_stage(self):
        with patch.object(ar,'_oracle',side_effect=oracle):
            receipt=ar.verify_reattestation_shard(self.plan,start=0,max_jobs=2,deadline=self.deadline)
        self.assertEqual(len(receipt.jobs),2)
        with patch.object(ar,'stage_reattestation_models')as stage,self.assertRaisesRegex(ValueError,'Every declared'):
            ar.finalize_reattestation(self.plan,[receipt],Path(self.temp.name)/'out',
                deadline=self.deadline,teacher_directory='fixture-teachers')
        stage.assert_not_called()
        with self.assertRaises(ValueError):
            ar.finalize_reattestation(self.plan,[receipt,receipt],Path(self.temp.name)/'out',
                deadline=self.deadline,teacher_directory='fixture-teachers')

    def test_hash_paths_and_source_mutation_rejected(self):
        with self.assertRaises(ValueError):ar._safe(self.source,'../outside.json')
        with self.assertRaises(ValueError):ar._safe(self.source,str(self.source/'scope.json'))
        write(self.source/'scope.json',dict(self.plan.scope,rule_hash='changed'))
        with self.assertRaisesRegex(ValueError,'Source proof'):
            ar.verify_reattestation_shard(self.plan,start=0,max_jobs=1,deadline=self.deadline)

    def test_metadata_staging_rejects_incompatible_schema_without_output(self):
        output=Path(self.temp.name)/'out'
        with self.assertRaisesRegex(ValueError,'feature/schema'):
            ar.stage_reattestation_models(self.plan,output,teacher_directory='fixture-teachers')
        self.assertFalse(output.exists())

    def test_phase_summary_and_count_drift_cannot_be_rehashed(self):
        scope=deepcopy(self.plan.scope)
        summary=json.loads((self.source/'selection.json').read_text());summary['seeds'][0]+=1
        write(self.source/'selection.json',summary)
        scope['files']['selection.json']=ar._digest(self.source/'selection.json')
        write(self.source/'scope.json',scope)
        with self.assertRaisesRegex(ValueError,'phase endpoints'):
            ar.prepare_reattestation(self.source,starting_decks=self.starts)

    def test_full_fixture_receipts_still_require_current_calibration(self):
        # Full synthetic receipt coverage tests the final gate, not historical equivalence.
        items=tuple(dict(absolute=i,phase=p,ordinal=o,file_sha256=d,actions=8,value_rows=4)
                    for i,(p,o,j,d)in enumerate(self.plan.jobs))
        from app.modules.card_game.rl.seven_lineup import identity
        receipt=ar.ReattestationShard(self.plan.plan_sha256,identity(),ar.cv.identity(),ar._verifier_identity(),items)
        output=Path(self.temp.name)/'out'
        def stage(*args,**kwargs):output.mkdir();return{k:object()for k in ROSTER}
        failed={k:dict(normal_games=8,brier=.8,log_loss=.5)for k in ROSTER}
        with patch.object(ar,'stage_reattestation_models',side_effect=stage),patch(
                'app.modules.card_game.rl.seven_round.value_metrics',return_value=failed),self.assertRaises(ValueError):
            ar.finalize_reattestation(self.plan,[receipt],output,deadline=self.deadline,teacher_directory='fixture')
        report=json.loads((output/'scope.json').read_text())
        self.assertFalse(report['passed']);self.assertEqual(report['choices'],self.plan.scope['choices'])
        self.assertFalse(report['reattestation']['fresh_confirmation'])
        self.assertFalse(report['reattestation']['fresh_gpu_collection'])
        self.assertEqual(report['reattestation']['weight_updates'],0)


if __name__=='__main__':unittest.main()
