"""Protocol tests: no real optimizer runs and no claims of model strength."""
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from app.modules.card_game.rl.build_acceptance import (
    canonical_build, build_identity, canonical_candidates, claim_seeds,
    phase_deadlines, paired_result, delivery_decision, check_source)
from app.modules.card_game.rl.league_schema import deck, PRESETS


def evidence(n=1024, *, candidate_win=True, baseline_win=False):
    return [dict(variant=v,seed=seed,foe=foe,first=first,complete=True,
                 winner='a' if (candidate_win if v=='candidate' else baseline_win) else 'b')
            for v in ('baseline','candidate') for seed in range(n)
            for foe in ('left','right') for first in ('a','b')]


class AcceptanceTest(unittest.TestCase):
    def builds(self):
        b=deck('starter');c=deepcopy(b);c['card_ids'].remove('J03');c['card_ids'].append('J01')
        return b,c

    def test_permutations_are_one_build_and_one_real_shuffle(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        b,_=self.builds();c=deepcopy(b);c['card_ids'].reverse();c['id']='another';c['name']='another'
        self.assertEqual(build_identity(b),build_identity(c));self.assertEqual(len(canonical_candidates([b,c])),1)
        a=new_game(seed=32,decks={'a':canonical_build(b),'b':canonical_build(b)})
        z=new_game(seed=32,decks={'a':canonical_build(c),'b':canonical_build(b)})
        for zone in ('hand','deck'):
            self.assertEqual([(x['card_id'],x['instance_id']) for x in a['sides']['a'][zone]],
                             [(x['card_id'],x['instance_id']) for x in z['sides']['a'][zone]])
        changed=deepcopy(b);changed['character_ids'].reverse()
        self.assertNotEqual(build_identity(b),build_identity(changed))

    def test_generated_candidates_have_unique_multisets(self):
        from app.modules.card_game.rl.full_cycle_experiment import candidates
        for k in PRESETS:
            rows=candidates(deck(k),43);self.assertGreater(len(rows),50)
            self.assertEqual(len(rows),len({build_identity(b) for b in rows}))
            self.assertTrue(all(b['card_ids']==sorted(b['card_ids']) for b in rows))

    def test_original_baseline_survives_every_round_and_screen_even_when_ranked_last(self):
        import time
        import app.modules.card_game.rl.full_cycle_experiment as e
        builds={k:canonical_build(deck(k)) for k in PRESETS};seen=[]
        def compare(pool,out,key,dirs,bs,models,reference,seed,end,pairs):
            baseline=build_identity(builds[key])
            self.assertIn(baseline,{build_identity(b) for b in bs.values()})
            seen.append((key,seed))
            return {label:dict(complete=True,n=4*pairs,wins=0 if build_identity(b)==baseline else 4*pairs) for label,b in bs.items()}
        with tempfile.TemporaryDirectory() as tmp,patch.object(e,'compare',compare):
            e.tune(None,Path(tmp),'models',builds,time.time()+100,dict(smoke=False,seeds={'search':10000000000},deadlines={'shutdown':time.time()+100}))
        self.assertEqual(len(seen),36)

    def test_persistent_seed_reservations_never_reuse_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ledger.json';a=claim_seeds(p,'a');b=claim_seeds(p,'b')
            self.assertEqual(len(set(a.values())),7)
            self.assertGreater(min(b.values()),max(a.values()))
            with self.assertRaises(ValueError):claim_seeds(p,'a')
            self.assertEqual(len(json.loads(p.read_text())['runs']),2)

    def test_reserved_budget_cannot_be_consumed_by_training(self):
        d=phase_deadlines(100,21600)
        self.assertEqual(d['shutdown'],21700)
        self.assertLess(d['candidate_adaptation'],d['confirmation'])
        self.assertLess(d['confirmation'],d['audit'])
        self.assertGreaterEqual(d['shutdown']-d['candidate_adaptation'],21600*.29)
        with self.assertRaises(ValueError):phase_deadlines(100,21601)

    def result(self,rows,n=1024):
        return paired_result(rows,expected_seeds=range(n),opponents=('left','right'))

    def test_confirmation_uses_seed_clusters_and_rejects_bad_data(self):
        rows=evidence();r=self.result(rows)
        self.assertEqual(r['seed_clusters'],1024);self.assertEqual(r['decision'],'accepted')
        self.assertEqual(r['delta'],1);self.assertEqual(len(r['groups']),4)
        for broken in (rows[:-1],rows+[rows[0]],[dict(rows[0],complete=False)]+rows[1:],
                       [dict(rows[0],seed=999999)]+rows[1:]):
            self.assertFalse(self.result(broken)['complete'])
        self.assertEqual(self.result(evidence(candidate_win=False,baseline_win=True))['decision'],'rejected')
        self.assertEqual(self.result(evidence(candidate_win=True,baseline_win=True))['decision'],'inconclusive')

    def test_equal_group_scoring_does_not_add_worst_matchup_weight(self):
        rows=evidence()
        for r in rows:
            if r['foe']=='left' and r['first']=='a':r['winner']='a' if r['variant']=='baseline' else 'b'
        result=self.result(rows)
        self.assertEqual(result['delta'],.5)
        self.assertEqual(result['decision'],'accepted')

    def test_audit_regression_overrides_screen_and_confirmation_winner(self):
        b,c=self.builds();confirm=self.result(evidence());audit=self.result(evidence(candidate_win=False,baseline_win=True))
        result=delivery_decision(b,c,confirm,audit,fair_adaptation=True,tactical_passed=True)
        self.assertEqual(result['recommendation_status'],'retained_baseline')
        self.assertEqual(build_identity(result['recommended_build']),build_identity(b))
        self.assertFalse(result['build_quality_approved'])

    def test_inconclusive_unfair_incomplete_and_smoke_never_promote(self):
        b,c=self.builds();good=self.result(evidence());tie=self.result(evidence(candidate_win=True,baseline_win=True))
        for confirm,audit,fair,tactic,smoke in [(tie,good,True,True,False),(good,good,False,True,False),
                  (good,dict(complete=False),True,True,False),(good,good,True,False,False),(good,good,True,True,True)]:
            self.assertFalse(delivery_decision(b,c,confirm,audit,fair_adaptation=fair,tactical_passed=tactic,smoke=smoke)['build_quality_approved'])
        accepted=delivery_decision(b,c,good,good,fair_adaptation=True,tactical_passed=True)
        self.assertTrue(accepted['build_quality_approved']);self.assertFalse(accepted['automatic_serving_approval'])

    def test_discarded_run_requires_explicit_research_source_and_new_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);models=p/'verified/adaptation/latest';models.mkdir(parents=True)
            (p/'disposition.json').write_text(json.dumps(dict(default_reuse_allowed=False)))
            with self.assertRaises(ValueError):check_source(models)
            with self.assertRaises(ValueError):check_source(models,research_warm_start=True)
            self.assertIsNotNone(check_source(models,research_warm_start=True,baseline_builds={'starter':{}}))


class FullCycleProtocolTest(unittest.TestCase):
    def test_full_orchestration_runs_equal_adaptation_and_rolls_back_after_audit(self):
        """Real orchestration with bounded deterministic adapters, no Torch training."""
        from types import SimpleNamespace
        import sys
        from contextlib import ExitStack
        import app.modules.card_game.rl.full_cycle_experiment as e
        builds={k:canonical_build(deck(k)) for k in PRESETS};proposals=deepcopy(builds)
        proposals['starter']['card_ids'].remove('J03');proposals['starter']['card_ids'].append('J01')
        calls=[];now=100.;metadata={}
        class Model:
            def __init__(self,directory,key):
                self.serving_deck=metadata.get((str(directory),key),builds[key]);self.version='same-start'
        def export(net,out,key,build,**kwargs):
            out=Path(out);out.mkdir(exist_ok=True,parents=True);metadata[str(out),key]=build
            (out/(key+'.npz')).write_bytes(b'model');(out/(key+'.json')).write_text('{}')
        def migrate(src,target):
            for k in PRESETS:export(None,target,k,builds[k])
        def train(pool,out,source,reference,bs,rbs,deadline,config,stage,probe,update_target=None):
            calls.append((stage,str(source),str(reference),deepcopy(rbs),update_target))
            directory=out/stage;directory.mkdir()
            e.write(directory/'summary.json',dict(totals={k:dict(steps=update_target or 0) for k in PRESETS}))
            for folder in ('initial','latest'):
                for k in PRESETS:export(None,directory/folder,k,bs[k])
            return directory/'latest'
        def tune(*args):
            folder=args[1]/'cards';folder.mkdir();e.write(folder/'status.json',dict(workflow_complete=True));return proposals
        def gate(*args):
            phase=args[-2]
            return dict(complete=True,decision='accepted',delta=.2,lower=.1) if phase=='confirmation' else dict(complete=True,delta=-.1,lower=-.2)
        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
        fake_torch=SimpleNamespace(set_num_threads=lambda n:None,manual_seed=lambda n:None)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            out=Path(tmp);config=dict(device='cpu',initial='source',research_warm_start=False,rule_hash='rules',
                seeds={'probes':1},deadlines={k:10000 for k in phase_deadlines(100,600)},workers=1,
                adaptation_steps=32,smoke=False,audit_margin=.05)
            for name,value in [('LeagueModel',Model),('migrate_numeric',migrate),('load_numeric',lambda *args:(None,{})),
                  ('export_model',export),('probes',lambda *args:{}),('train',train),('tune',tune),
                  ('gate_comparison',gate),('proposal_tactics',lambda *args:dict(passed=True,complete=True)),
                  ('evaluate',lambda *args:True),('ProcessPoolExecutor',Pool),('rule_hash',lambda:'rules')]:
                stack.enter_context(patch.object(e,name,value))
            stack.enter_context(patch.dict(sys.modules,{'torch':fake_torch}))
            e.run(out,config)
            baseline,candidate=calls[1:];self.assertEqual(baseline[1:],candidate[1:])
            result=json.loads((out/'delivery-decision.json').read_text())
            self.assertEqual(result['starter']['reason'],'audit_regression_or_insufficient_evidence')
            self.assertEqual(build_identity(result['starter']['recommended_build']),build_identity(builds['starter']))
            status=json.loads((out/'status.json').read_text());self.assertTrue(status['workflow_complete']);self.assertFalse(status['quality_approved'])

class RealEvaluationNormalizationTest(unittest.TestCase):
    def test_frozen_numeric_policy_sees_same_game_for_permuted_input(self):
        import gzip,hashlib,time
        import numpy as np
        from app.modules.card_game.rl.league_schema import SCHEMA,SEATS,CARD_IDS,CAND_DIM,feature_names,rule_hash,build_hash
        from app.modules.card_game.rl.league_policy import shapes
        from app.modules.card_game.rl.full_cycle_experiment import evaluate_job
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=root/'models';models.mkdir();(root/'raw').mkdir();(root/'replays').mkdir()
            for key in ('starter','weave-rush'):
                p=models/f'{key}.npz';np.savez_compressed(p,**{k:np.zeros(v,np.float32) for k,v in shapes(4).items()})
                b=deck(key);m=dict(schema=SCHEMA,deck=key,rule_hash=rule_hash(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    seat_ids=list(SEATS),card_ids=list(CARD_IDS),features=feature_names(),cand_dim=CAND_DIM,hidden=4,
                    build=b,build_sha256=build_hash(b),validation={})
                (models/f'{key}.json').write_text(json.dumps(m))
            b=deck('starter');permuted=deepcopy(b);permuted['card_ids'].reverse()
            job=dict(cell='canonical-check',seed=111,first='a',deadline=time.time()+20,
                     policies={'a':(str(models),'starter'),'b':(str(models),'weave-rush')},output=str(root))
            rows=[]
            for i,build in enumerate((b,permuted)):
                result=evaluate_job(dict(job,id=str(i),decks={'a':build,'b':deck('weave-rush')}));self.assertTrue(result['complete'])
                rows.append(json.loads(gzip.decompress((root/'raw'/f'{i}.json.gz').read_bytes())))
            self.assertEqual(rows[0]['actions'],rows[1]['actions'])
            self.assertEqual(rows[0]['decks'],rows[1]['decks'])
            self.assertEqual((rows[0]['winner'],rows[0]['turn']),(rows[1]['winner'],rows[1]['turn']))

    def test_copied_discarded_weights_still_match_registry(self):
        from app.modules.card_game.rl.build_acceptance import source_disposition
        registry=json.loads(Path('app/modules/card_game/rl/experiment_dispositions.json').read_text())
        weight=next(iter(registry['records'][0]['weights']))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'starter.json').write_text(json.dumps(dict(sha256=weight)))
            self.assertFalse(source_disposition(p)['default_reuse_allowed'])
            with self.assertRaises(ValueError):check_source(p)
