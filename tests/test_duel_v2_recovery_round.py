import json
from pathlib import Path
import tempfile
import time
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl import recovery_round as round, recovery_runtime as rt
from app.modules.card_game.rl.cross_schedule import ROSTER
from app.modules.card_game.rl.cross_lineup import identity
from scripts.train_duel_v2_current_round import preset_decks, validate_round_launch


class RecoveryRoundTest(unittest.TestCase):
    def test_single_learner_schedule_keeps_opponents_out_of_training_rows(self):
        decks = preset_decks()
        jobs = round.jobs_for('models', decks, 100, float('inf'), pure=False, keys=['murk'])
        self.assertEqual(len(jobs), 10)
        self.assertEqual({j['first'] for j in jobs}, {'a', 'b'})
        for job in jobs:
            self.assertTrue(job['train_sides'])
            self.assertTrue(all(job['policies'][s][1] == 'murk' for s in job['train_sides']))
        compared = round.compare_jobs('new', 'old', decks, 200, float('inf'), 5, keys=['murk'])
        self.assertEqual(len(compared), 40)
        self.assertEqual({j['key'] for j in compared}, {'murk'})
        self.assertEqual({j['policies']['b' if j['learner']=='a' else 'a'][1] for j in compared}, set(ROSTER))
        for invalid in ([], ['murk', 'murk'], ['unknown']):
            with self.assertRaises(ValueError):
                round.jobs_for('models', decks, 100, float('inf'), keys=invalid)

    def test_single_learner_rejects_frozen_opponent_rows_before_update(self):
        jobs = round.jobs_for('models', preset_decks(), 100, float('inf'), pure=False)
        with self.assertRaisesRegex(ValueError, 'Frozen opponent'):
            round.update_batch({}, {}, {}, jobs, [], None, {'learning_keys':['murk']}, float('inf'), 'unused')

    def test_single_learner_gate_does_not_require_learning_health_from_frozen_teams(self):
        health = {'murk':dict(finite=True, gradient=True, coverage=1, policy_rows=8)}
        row = dict(legal_play=10, plays=5, candidate=6, reference=4, games=8, changed=2)
        result = round.gate_decision({'murk':row}, {'murk':row}, health, no_gain=False,
            search={'murk':row}, search_confirm={'murk':row}, keys=['murk'])
        self.assertEqual(result['promoted_keys'], ['murk'])
        self.assertEqual(set(result['retained_keys']), set(ROSTER)-{'murk'})

    def test_full_value_calibration_preserves_policy_and_never_selects_on_heldout(self):
        import torch
        from app.modules.card_game.engine.duel_v2 import new_game
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);nets,decks,origins=round.initialize(root/'initial',16,'cpu',29)
            before={k:{n:t.clone() for n,t in net.state_dict().items()} for k,net in nets.items()}
            for net in nets.values():
                for name,parameter in net.named_parameters():
                    if name.startswith(('state_net.','cand_net.','score.')):
                        parameter.grad=torch.ones_like(parameter)*100
            phases=[]
            for phase in ('train','selection','heldout'):
                games=[]
                for ki,key in enumerate(ROSTER):
                    state=new_game(seed=90+ki,first_side='a',decks={'a':decks[key],'b':decks[key]})
                    _,(x,c)=rt.decision(state,'a')
                    games.append(dict(complete=True,value_rows=[dict(x=x,key=key,z=1,value_actor=1),
                        dict(x=x,key=key,z=-1,value_actor=0)]))
                phases.append(games)
            calls=[];original=round.value_metrics
            def metrics(games,policies):
                calls.append(id(games));return original(games,policies)
            config=dict(value_full_tower=True,value_replicas=2,value_check_replicas=3,
                seeds=dict(adaptation=100000,selection=200000),rule_hash=identity())
            with patch.object(round,'collect',side_effect=phases) as collect,patch.object(round,'value_metrics',side_effect=metrics):
                models,report=round.warm_value(None,nets,decks,origins,root/'initial',root/'value',config,1,float('inf'))
            self.assertEqual([len(call.args[1]) for call in collect.call_args_list],[60,90,90])
            self.assertEqual(calls.count(id(phases[2])),1)
            self.assertEqual(calls[-1],id(phases[2]))
            train_seeds={j['seed'] for j in collect.call_args_list[0].args[1]}
            selected_seeds={j['seed'] for j in collect.call_args_list[1].args[1]}
            held_seeds={j['seed'] for j in collect.call_args_list[2].args[1]}
            self.assertFalse(train_seeds & selected_seeds or train_seeds & held_seeds or selected_seeds & held_seeds)
            for key,net in nets.items():
                self.assertIn(report['selected_epochs'][key],(1,2,4,8))
                self.assertEqual(report['updates'][key]['total_attempted'],8)
                self.assertEqual(report['updates'][key]['selected'],report['selected_epochs'][key])
                self.assertTrue(any(not torch.equal(t,net.state_dict()[n]) for n,t in before[key].items() if n.startswith('value_net.')))
                for n,t in before[key].items():
                    if not n.startswith(('value_net.','wdl.')):self.assertTrue(torch.equal(t,net.state_dict()[n]),n)
                self.assertTrue(all(p.grad is None for n,p in net.named_parameters()
                                    if n.startswith(('state_net.','cand_net.','score.'))))
            self.assertTrue(report['policy_preserved'])

    def test_value_calibration_protocol_is_explicit_and_bounded(self):
        self.assertEqual(round.value_calibration_protocol({})['epoch_candidates'],[16])
        full=round.value_calibration_protocol(dict(value_full_tower=True,value_replicas=12,value_check_replicas=3))
        self.assertEqual(full['epoch_candidates'],[1,2,4,8])
        self.assertEqual(full['train_replicas'],12)
        self.assertEqual(full['check_replicas'],3)

    def test_expanded_preflight_still_has_a_bound_and_smoke_cannot_inherit_it(self):
        config=dict(runtime='recovery',rule_hash=identity(),smoke=True,preflight=True,
            preflight_replicas=3,planned_batches=1,workers=12,simulations=32,
            started_at=0,hard_deadline=7200,train_until=7100,max_row_reuse=8,policy_passes=8)
        round.validate_launch(config)
        for changes in ({'hard_deadline':7201}, {'preflight':False}, {'preflight_replicas':1},
                        {'preflight_replicas':4}, {'preflight_replicas':True}):
            with self.assertRaises(ValueError):round.validate_launch(dict(config,**changes))

    def test_candidate_alias_loss_floor_reports_unlearnable_labels(self):
        from app.modules.card_game.rl.recovery_capacity import candidate_alias_metrics
        row=dict(c=np.array([[1.,2.],[1.,2.],[3.,4.]]),pi=np.array([.8,0.,.2]))
        result=candidate_alias_metrics([row])
        self.assertAlmostEqual(result['minimum_kl'],.8*np.log(2))
        self.assertEqual(result['conflicting_label_rows'],1)
        self.assertEqual(result['duplicate_candidates'],1)
        self.assertFalse(result['rule_equivalence_verified'])
        balanced={**row,'pi':np.array([.4,.4,.2])}
        self.assertEqual(candidate_alias_metrics([balanced])['minimum_kl'],0.)
        distinct={**row,'c':np.array([[1.,2.],[1.,3.],[3.,4.]])}
        self.assertEqual(candidate_alias_metrics([distinct])['minimum_kl'],0.)
        with self.assertRaises(ValueError):
            candidate_alias_metrics([{**row,'pi':np.array([1.,-1.,1.])}])

    def test_capacity_rejects_missing_results_before_loading_source(self):
        from app.modules.card_game.rl.recovery_capacity import fit
        with self.assertRaisesRegex(ValueError,'Training job/result'):
            fit('unused',[{}],[],{},float('inf'))
        with self.assertRaisesRegex(ValueError,'Both heldout'):
            fit('unused',[{}],[{}],{},float('inf'),held_jobs=[{}])
        with self.assertRaisesRegex(ValueError,'Heldout job/result'):
            fit('unused',[{}],[{}],{},float('inf'),held_jobs=[{}],held_games=[])

    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_unknown_target_uses_residual_whole_root_and_keeps_supported_teacher(self):
        import torch
        from app.modules.card_game.rl import preserved_policy
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.league_learning import tensors
        decks=preset_decks()
        with tempfile.TemporaryDirectory() as root:
            nets,_,_=round.initialize(root,16,'cpu',12)
            supported=new_game(seed=3,skip_mulligan=True,decks={'a':decks['starter'],'b':decks['weave-rush']})
            _,(x,c)=rt.decision(supported,'a')
            strict,_=preserved_policy.create_network('starter',hidden=16)
            preserved_policy.export(strict,Path(root)/'strict','starter',decks['starter'],{})
            np.testing.assert_array_equal(rt.model(root,'starter').scores(x,c),rt.model(Path(root)/'strict','starter').scores(x,c))
            unknown=c.copy();unknown[0,4]=13
            # The unsupported target must not accidentally become a legacy
            # target via ordinal mapping, even for other actions in this root.
            with self.assertRaises(ValueError): rt.model(Path(root)/'strict','starter').scores(x,unknown)
            numeric=rt.model(root,'starter').scores(x,unknown)
            np.testing.assert_array_equal(numeric,np.zeros(len(c)))
            xx,cc,mask=tensors([(x,unknown)],'cpu')
            logits,_=nets['starter'](xx,cc,mask)
            np.testing.assert_array_equal(logits.detach().numpy()[0],numeric)
            self.assertEqual(rt.model(root,'zhenhong').manifest['origin']['initialization'],'fresh_residual_no_teacher')
            self.assertEqual(rt.model(root,'murk').manifest['origin']['initialization'],'fresh_residual_no_teacher')

    def test_schedule_and_historical_sample_ownership(self):
        decks=preset_decks()
        jobs=round.jobs_for('now',decks,100,time.time()+60,pure=False,report={})
        self.assertEqual(len(jobs),30)
        self.assertEqual(len(set((j['left'],j['right'],j['first']) for j in jobs)),30)
        for a,b in zip(jobs[::2],jobs[1::2]): self.assertEqual(a['seed'],b['seed'])
        history=round.jobs_for('now',decks,100,time.time()+60,pure=False,historical='old')
        self.assertEqual(len(history),60)
        for job in history:
            self.assertEqual(len(job['train_sides']),1)
            learner=job['train_sides'][0]
            self.assertEqual(job['policies'][learner][0],'now')
            self.assertEqual(job['policies']['b' if learner=='a' else 'a'][0],'old')
        covered=round.jobs_for('now',decks,100,time.time()+60,pure=False,teacher_mode='covered_expert')
        self.assertTrue(all(j['teacher_mode']=='covered_expert' for j in covered))
        comparisons=round.compare_jobs('new','old',decks,100,time.time()+60,1,search=True,teacher_mode='covered_expert')
        self.assertTrue(all(j['teacher_mode']=='covered_expert' for j in comparisons))

    def test_selective_promotion_keeps_unchanged_teams_and_requires_each_search_block(self):
        health={k:dict(finite=True,gradient=True,coverage=1.,policy_rows=64) for k in ROSTER}
        baseline={k:dict(candidate=4.,reference=4.,games=8,changed=0,plays=10,legal_play=40) for k in ROSTER}
        better=deepcopy(baseline)
        for k in ('zhenhong','murk'):better[k].update(candidate=6.,changed=3)
        decision=round.gate_decision(better,better,health,no_gain=True,search=better,search_confirm=better)
        self.assertEqual(decision['decision'],'promote')
        self.assertEqual(decision['promoted_keys'],['zhenhong','murk'])
        self.assertEqual(set(decision['retained_keys']),set(ROSTER[:3]))
        failed=deepcopy(better);failed['murk']['candidate']=3
        decision=round.gate_decision(better,better,health,no_gain=True,search=better,search_confirm=failed)
        self.assertEqual(decision['promoted_keys'],['zhenhong'])

    def test_selective_copy_preserves_other_numeric_models_exactly(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            ref=Path(root)/'reference';candidate=Path(root)/'candidate'
            nets,decks,origins=round.initialize(ref,16,'cpu',5)
            for key,net in nets.items():
                with torch.no_grad():net.wdl.bias.add_(.1)
                rt.export(net,candidate,key,decks[key],origins[key])
            selected=round.select_models(candidate,ref,['zhenhong','murk'],Path(root)/'selected'/'models')
            for key in ROSTER:
                source=candidate if key in ('zhenhong','murk') else ref
                self.assertEqual(rt.model(selected,key).version,rt.model(source,key).version)
                for name,array in rt.model(source,key).weights.items():
                    np.testing.assert_array_equal(rt.model(selected,key).weights[name],array)
            with self.assertRaises(ValueError):round.select_models(candidate,ref,[],Path(root)/'bad')

    def test_gate_stops_no_learning_and_never_promotes_unchanged_actions(self):
        health={k:dict(finite=True,gradient=True,coverage=1.,policy_rows=64) for k in ROSTER}
        pure={k:dict(candidate=4.,reference=4.,games=8,changed=0,decisions=50,
                     plays=10,reference_plays=10,legal_play=40) for k in ROSTER}
        self.assertEqual(round.gate_decision(pure,pure,health,no_gain=False)['decision'],'keep')
        self.assertEqual(round.gate_decision(pure,pure,health,no_gain=True)['decision'],'plateau')
        better=deepcopy(pure)
        for r in better.values():r.update(candidate=6.,changed=3)
        self.assertEqual(round.gate_decision(better,better,health,no_gain=False)['decision'],'keep')
        self.assertEqual(round.gate_decision(better,better,health,no_gain=False,search=better,search_confirm=better)['decision'],'promote')
        broken=deepcopy(better);broken['zhenhong']['plays']=0
        self.assertEqual(round.gate_decision(broken,broken,health,no_gain=False)['decision'],'p0')
        health['murk']['coverage']=.99
        self.assertEqual(round.gate_decision(better,better,health,no_gain=False)['decision'],'p0')

    def test_mixed_versions_require_separate_value_evidence(self):
        from types import SimpleNamespace
        a=SimpleNamespace(manifest={'deck':'starter'},version='a')
        b=SimpleNamespace(manifest={'deck':'starter'},version='b')
        def report(version):
            return dict(schema='recovery_value_precheck_v1',complete=True,heldout=True,passed=True,
                        rule_hash=identity(),models={'starter':version},
                        metrics={'starter':dict(normal_games=12,brier=.4,log_loss=.8)})
        self.assertFalse(rt.value_precheck_ok(report('a'),{'a':a,'b':b}))
        combined=round.merged_reports(report('a'),report('b'))
        self.assertTrue(rt.value_precheck_ok(combined,{'a':a,'b':b}))
        combined['by_version']['b']['metrics']['brier']=.8
        self.assertFalse(rt.value_precheck_ok(combined,{'a':a,'b':b}))

    def test_formal_flags_cannot_replace_receipt(self):
        config=dict(runtime='recovery',rule_hash=identity(),smoke=False,device='cuda',
                    formal_launch_approved=True,throughput_passed=True,recovery_source='unused',
                    simulations=32,gumbel_candidates=16,terminal_horizon=3,rewards=0,planned_batches=6,gate_roots=32,search_gate_roots=2,gates=dict(first=1,mid=2,late=4))
        with self.assertRaisesRegex(ValueError,'receipt required'):validate_round_launch(config)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'ready.json';path.write_text(json.dumps({'schema':'five_recovery_readiness_v1','device':'cpu'}))
            with self.assertRaisesRegex(ValueError,'identity/age/config'):round.validate_launch(dict(config,readiness_report=str(path)))

    def test_completed_training_only_and_value_rows_keep_terminal_view(self):
        import torch
        from app.modules.card_game.rl.episode_replay import EpisodePool
        from app.modules.card_game.engine.duel_v2 import new_game
        with tempfile.TemporaryDirectory() as root:
            nets,decks,_=round.initialize(Path(root)/'models',16,'cpu',23)
            opts={k:torch.optim.Adam(n.parameters(),lr=.001) for k,n in nets.items()}
            pools={k:EpisodePool(max_row_reuse=4) for k in ROSTER}
            jobs=[];games=[]
            for key in ROSTER:
                state=new_game(seed=3,decks={'a':decks[key],'b':decks[key]})
                _,(x,c)=rt.decision(state,'a');pi=np.zeros(len(c));pi[0]=1
                rows=[dict(x=x,c=c,pi=pi,selected=0,z=1,side='a',step=0,key=key),
                      dict(x=x,c=c,pi=pi,selected=0,z=-1,side='b',step=1,key=key)]
                values=[dict(x=x,z=1,side='a',step=1,key=key,value_actor=0)]
                jobs.append(dict(id=key,train_sides=['a','b'],policies={'a':('m',key),'b':('m',key)}))
                games.append(dict(complete=True,decisions=2,episode_actions=[{},{}],rows=rows,value_rows=values,winner='a'))
            health,rows=round.update_batch(nets,opts,pools,jobs,games,np.random.default_rng(7),
                                          {'policy_passes':4},float('inf'),root)
            for key in ROSTER:
                self.assertTrue(health[key]['gradient'])
                self.assertEqual(health[key]['coverage'],1)
                self.assertEqual(health[key]['value_rows'],1)
                self.assertEqual(len(rows[key]),2)
            from app.modules.card_game.rl.episode_replay import dump_episode
            with self.assertRaises(ValueError):dump_episode(Path(root)/'bad',jobs[0],dict(games[0],complete=False))

    def test_value_only_cannot_move_policy_after_adam_has_momentum(self):
        import torch
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.league_learning import tensors
        with tempfile.TemporaryDirectory() as root:
            nets,decks,_=round.initialize(root,16,'cpu',19)
            for key in ROSTER:
                net=nets[key];opt=torch.optim.Adam(net.parameters(),lr=.003)
                state=new_game(seed=9,first_side='a',skip_mulligan=True,decks={'a':decks[key],'b':decks[key]})
                _,(x,c)=rt.decision(state,'a');pi=np.zeros(len(c));pi[-1]=1
                # Fill policy Adam moments before the value-only step: zero
                # policy gradients must not let momentum move the policy.
                rt.update(net,opt,[dict(x=x,c=c,pi=pi,z=1,value_actor=1)])
                xx,cc,mask=tensors([(x,c)],'cpu')
                with torch.no_grad():before=net(xx,cc,mask)[0].clone()
                policy_before={n:t.clone() for n,t in net.state_dict().items()
                               if n.startswith(('state_net.','cand_net.','score.','legacy.'))}
                rt.update(net,opt,[dict(x=x,z=-1,value_actor=0)],value_only=True)
                with torch.no_grad():after=net(xx,cc,mask)[0]
                self.assertTrue(torch.equal(before,after),key)
                for n,t in policy_before.items():self.assertTrue(torch.equal(t,net.state_dict()[n]),n)
                rt.export(net,Path(root)/'updated',key,decks[key],{})
                numeric=rt.model(Path(root)/'updated',key)
                np.testing.assert_allclose(after.numpy()[0],numeric.scores(x,c),atol=2e-4,rtol=2e-4)
                h=net.value_net(xx)
                with torch.no_grad():prob=net.wdl(torch.cat((h,torch.zeros((1,1))),-1)).softmax(-1)
                np.testing.assert_allclose(prob.numpy()[0],numeric.wdl(x,False),atol=2e-4,rtol=2e-4)

    def test_isolation_fit_groups_paired_seeds_and_never_changes_source(self):
        from app.modules.card_game.rl.recovery_capacity import fit
        from app.modules.card_game.engine.duel_v2 import new_game
        with tempfile.TemporaryDirectory() as root:
            _,decks,_=round.initialize(root,16,'cpu',19)
            signature=round.versions(root);jobs=[];games=[]
            for seed in (10,11,12):
                for key in ROSTER:
                    state=new_game(seed=seed,first_side='a',decks={'a':decks[key],'b':decks[key]})
                    _,(x,c)=rt.decision(state,'a')
                    scores=rt.model(root,key).scores(x,c).astype(float)
                    pi=np.exp(scores-np.logaddexp.reduce(scores))
                    jobs.append(dict(seed=seed,training=True,train_sides=['a']))
                    games.append(dict(complete=True,rows=[dict(x=x,c=c,pi=pi,z=1,side='a',key=key,value_actor=1) for _ in range(131)]))
            result=fit(root,jobs,games,dict(device='cpu',seeds={'probes':31}),time.time()+30,steps=2)
            self.assertEqual(signature,round.versions(root))
            self.assertTrue(result['complete']);self.assertFalse(result['strength_approved'])
            for row in result['metrics'].values():
                self.assertFalse(set(row['train_seed_groups'])&set(row['heldout_seed_groups']))
                self.assertEqual(row['train_seed_groups'],[10])
                self.assertEqual(row['heldout_seed_groups'],[11,12])
                self.assertEqual(row['train_rows'],131)
                self.assertEqual(row['heldout_rows'],262)
                self.assertEqual(row['row_uses'],262)
                self.assertEqual(row['updates'],4)

    def test_cuda_preflight_refuses_before_creating_output(self):
        import torch
        if torch.cuda.is_available():self.skipTest('Only unavailable-CUDA refusal is under test')
        import subprocess,sys
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'not-created'
            result=subprocess.run([sys.executable,'scripts/train_duel_v2_current_round.py',
                                   '--preflight','--device','cuda','--seconds','300',
                                   '--output',str(out),'--run'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('CUDA unavailable',result.stderr)
            self.assertFalse(out.exists())

    def test_receipt_excludes_mutable_guard_and_refuses_completed_resume(self):
        import torch
        from app.modules.card_game.rl.episode_replay import EpisodePool
        with tempfile.TemporaryDirectory() as root:
            out=Path(root);models=out/'models'
            nets,_,_=round.initialize(models,16,'cpu',23)
            opts={k:torch.optim.Adam(n.parameters()) for k,n in nets.items()}
            pools={k:EpisodePool() for k in ROSTER}
            config=dict(train_until=time.time()+100,hard_deadline=time.time()+200,planned_batches=2)
            round.save_round(root,nets,opts,pools,models,np.random.default_rng(3),config,1,models,True)
            (out/'guard.json').write_text('{"phase":"running"}')
            before=round.immutable_receipt_files(out)
            (out/'guard.json').write_text('{"phase":"exited"}')
            (out/'status.json').write_text('{"phase":"complete"}')
            self.assertEqual(before,round.immutable_receipt_files(out))
            with self.assertRaisesRegex(ValueError,'already stopped'):
                round.restore_round(root,'cpu',config)

    def test_new_diagnosis_uses_new_window_without_extending_old_training(self):
        from datetime import datetime,timezone,timedelta
        from scripts.check_duel_v2_recovery_capacity import diagnostic_deadline
        tz=timezone(timedelta(hours=8))
        morning=datetime(2026,9,27,8,59,tzinfo=tz).timestamp()
        self.assertEqual(diagnostic_deadline(morning,1800),morning+60)
        afternoon=datetime(2026,9,27,14,0,tzinfo=tz).timestamp()
        self.assertEqual(diagnostic_deadline(afternoon,1800),afternoon+1800)
        # No source-run deadline is an argument to this new, separately
        # seeded diagnostic. Exact training resume still uses restore_round.

    def test_balanced_capacity_rejects_overlap_and_missing_opponents(self):
        from app.modules.card_game.rl.recovery_capacity import fit
        from app.modules.card_game.engine.duel_v2 import new_game
        with tempfile.TemporaryDirectory() as root:
            _,decks,_=round.initialize(root,16,'cpu',19)
            jobs=[];games=[]
            for seed in (10,11,12):
                for key in ROSTER:
                    state=new_game(seed=seed,first_side='a',decks={'a':decks[key],'b':decks[key]})
                    _,(x,c)=rt.decision(state,'a')
                    scores=rt.model(root,key).scores(x,c).astype(float)
                    pi=np.exp(scores-np.logaddexp.reduce(scores))
                    jobs.append(dict(seed=seed,training=True,train_sides=['a'],left=key,right=key,first='a'))
                    games.append(dict(complete=True,rows=[dict(x=x,c=c,pi=pi,z=1,side='a',key=key,value_actor=1)]))
            held_jobs=[dict(j,seed=j['seed']+100,training=False) for j in jobs]
            kwargs=dict(config=dict(device='cpu',seeds={'probes':31}),deadline=float('inf'),steps=2)
            with self.assertRaisesRegex(ValueError,'overlap'):
                fit(root,jobs,games,held_jobs=jobs,held_games=games,**kwargs)
            bad=deepcopy(held_jobs);bad[0]['first']='b'
            with self.assertRaisesRegex(ValueError,'coverage differs'):
                fit(root,jobs,games,held_jobs=bad,held_games=games,**kwargs)
            result=fit(root,jobs,games,held_jobs=held_jobs,held_games=games,**kwargs)
            self.assertEqual(result['partition'],'balanced_new_paired_seeds')
            for row in result['metrics'].values():
                self.assertEqual(row['train_seed_groups'],[10,11,12])
                self.assertEqual(row['heldout_seed_groups'],[110,111,112])

    def test_sampled_world_proof_does_not_poison_shared_information_value(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl import recovery_search
        from app.modules.card_game.rl import cross_runtime
        from app.modules.card_game.rl.information_search import Node
        decks=preset_decks()
        state=new_game(seed=3,first_side='a',skip_mulligan=True,
                       decks={'a':decks['starter'],'b':decks['starter']})
        class Uniform:
            def scores_only(self,x,c):return np.zeros(len(c),dtype=np.float32)
            def scores_value(self,x,c):return self.scores_only(x,c),0.
            def wdl(self,x,actor):return np.array([.3,.4,.3],dtype=np.float32)
        original=Node.create
        for mode in ('natural_wdl','legacy'):
            nodes=[]
            def remember(*args,**kwargs):
                node=original(*args,**kwargs);nodes.append(node);return node
            with patch.object(Node,'create',side_effect=remember), patch.object(
                    recovery_search.core,'short_terminal_value',return_value=1.) as proof:
                result=recovery_search.search(state,'a',{'a':Uniform(),'b':Uniform()},
                    seed=17,runtime=cross_runtime,simulations=4,max_depth=10,q_scale=mode)
            self.assertTrue(result['complete']);self.assertGreater(proof.call_count,0)
            if mode=='natural_wdl':self.assertTrue(all(node.raw_value==0. for node in nodes))
            else:self.assertTrue(any(node.raw_value==1. for node in nodes))
            # A proven +1 may still be backed up as a sampled outcome.
            self.assertGreater(sum(node.total.sum() for node in nodes),0)

    def test_all_five_private_worlds_natural_search_and_numeric_export(self):
        import torch
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.information_search import sample_world
        from app.modules.card_game.rl.recovery_search import search
        from app.modules.card_game.rl.league_learning import tensors
        with tempfile.TemporaryDirectory() as root:
            nets,decks,_=round.initialize(root,16,'cpu',21)
            for item in round.learning_schedule():
                a,b=item['left'],item['right']
                state=new_game(seed=5,first_side='a',skip_mulligan=True,decks={'a':decks[a],'b':decks[b]})
                actions,(x,c)=rt.decision(state,'a')
                world=sample_world(state,'a',71,runtime=rt)
                again,(xx,cc)=rt.decision(world,'a')
                self.assertEqual(actions,again)
                np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)
                tx,tc,mask=tensors([(x,c)],'cpu')
                with torch.no_grad(): logits,_=nets[a](tx,tc,mask)
                np.testing.assert_allclose(logits.numpy()[0],rt.model(root,a).scores(x,c),atol=2e-4,rtol=2e-4)
                result=search(state,'a',{'a':rt.model(root,a),'b':rt.model(root,b)},seed=81,
                              simulations=2,terminal_horizon=1,runtime=rt)
                self.assertTrue(result['complete']);self.assertAlmostEqual(float(result['pi'].sum()),1,places=5)

    def test_evaluation_uses_real_telemetry_and_verified_public_replay(self):
        import gzip
        from app.modules.card_game.rl.full_cycle_experiment import evaluate_job
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);nets,decks,_=round.initialize(folder/'models',16,'cpu',29)
            (folder/'eval/raw').mkdir(parents=True);(folder/'eval/replays').mkdir()
            job=dict(id='check',runtime='recovery',first='a',seed=73,deadline=time.time()+30,
                     output=str(folder/'eval'),replay=True,cell='cross/zhenhong/murk',
                     decks={'a':decks['zhenhong'],'b':decks['murk']},
                     policies={'a':(str(folder/'models'),'zhenhong'),'b':(str(folder/'models'),'murk')})
            result=evaluate_job(job)
            self.assertTrue(result['complete']);self.assertTrue(result['replay_verified'])
            with gzip.open(folder/'eval/raw/check.json.gz','rt') as handle:raw=json.load(handle)
            self.assertIn('telemetry',raw);self.assertTrue(raw['actions'])
            self.assertTrue((folder/'eval/replays/check.json.gz').is_file())
            with self.assertRaisesRegex(ValueError,'Matching WDL'):
                evaluate_job(dict(job,search_sides=['a','b']))

    def test_resume_keeps_optimizer_pools_rng_and_original_deadlines(self):
        import torch
        from app.modules.card_game.rl.episode_replay import EpisodePool
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);models=folder/'models'
            nets,_,_=round.initialize(models,16,'cpu',23)
            opts={k:torch.optim.Adam(n.parameters(),lr=.001) for k,n in nets.items()}
            pools={k:EpisodePool() for k in ROSTER};rng=np.random.default_rng(37)
            config=dict(train_until=time.time()+100,hard_deadline=time.time()+200)
            round.save_round(root,nets,opts,pools,models,rng,config,1,models,True)
            nn,oo,pp,rr,pointer,no_gain=round.restore_round(root,'cpu',config)
            self.assertTrue(no_gain);self.assertEqual(pointer['cycle'],1)
            np.testing.assert_array_equal(rng.integers(100,size=10),rr.integers(100,size=10))
            for k in ROSTER:
                for n,v in nets[k].state_dict().items():self.assertTrue(torch.equal(v,nn[k].state_dict()[n]))
            with self.assertRaisesRegex(ValueError,'deadline'):round.restore_round(root,'cpu',dict(config,hard_deadline=config['hard_deadline']+1))


if __name__=='__main__':unittest.main()
