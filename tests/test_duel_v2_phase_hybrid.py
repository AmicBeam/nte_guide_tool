"""Local integration of phase isolation; CPU evidence never grants CUDA readiness."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl import seven_runtime as old, policy_runtime as dispatch
from app.modules.card_game.rl.next_training import plan
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.engine.duel_v2 import new_game


class PhaseHybridTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(61031)

    def bank(self, root, keys=('lingke',)):
        from app.modules.card_game.rl.phase_runtime import migrate
        decks = plan()['builds']
        for k in keys:
            net = old.create_network(k, 16)
            old.export(net, root/'old', k, decks[k], {'fixture': 'phase CPU only'})
            migrate(root/'old', root/'phase', k)
        return decks

    def rows(self, key, deck):
        result = []
        for i, opening in enumerate((True, False)):
            state = new_game(seed=61035, first_side='a', skip_mulligan=not opening, decks={'a': deck, 'b': deck})
            actions, (x, c) = old.decision(state, 'a')
            pi = np.zeros(len(c), np.float32); pi[0] = 1
            result.append(dict(x=x, c=c, pi=pi, side='a', key=key, step=i, z=1, value_actor=1,
                               phase='mulligan' if opening else 'playing', selected=0, action=actions[0]))
        return result

    def test_batches_cover_same_rows_without_cross_phase_or_tiny_remainder(self):
        from app.modules.card_game.rl.phase_replay import PhaseEpisodePool
        rows = self.rows('lingke', plan()['builds']['lingke'])
        pool = PhaseEpisodePool(max_row_reuse=4); pool.begin_fresh_batch()
        pool.add('example', dict(rows=[dict(rows[0], step=i) for i in range(16)] +
                                     [dict(rows[1], step=i+16) for i in range(499)]))
        batches = list(pool.policy_pass_batches(np.random.default_rng(3), passes=4))
        self.assertEqual(len(batches), 20)
        self.assertEqual(sorted(len(b[0]) for b in batches), [16]*4 + [124]*4 + [125]*12)
        self.assertEqual(set(pool.row_uses.values()), {4})
        self.assertEqual(sum(len(b[0]) for b in batches), 2060)
        for batch, _, stats in batches:
            self.assertEqual({r['phase']=='mulligan' for r in batch}, {stats['policy_phase']=='opening'})
        restored = PhaseEpisodePool.from_state(pool.export_state())
        self.assertEqual(restored.row_uses, pool.row_uses)
        self.assertFalse(list(restored.policy_pass_batches(np.random.default_rng(4), passes=4)))

    def test_update_does_not_move_inactive_parameters_or_adam_momentum(self):
        import torch
        from app.modules.card_game.rl.phase_checkpoint import create_optimizers
        from app.modules.card_game.rl.phase_update import update
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); deck=self.bank(root)['lingke']; net,_,_=dispatch.restore(root/'phase','lingke')
            opts=create_optimizers(net); opening,playing=self.rows('lingke',deck)
            update(net,opts,[opening]); update(net,opts,[playing])
            for active, row, inactive in [('playing',playing,'opening.'),('opening',opening,('state_net.','cand_net.','score.'))]:
                before={n:v.clone() for n,v in net.state_dict().items() if n.startswith(inactive)}
                other='opening' if active=='playing' else 'playing'
                adam=deepcopy(opts[other].state_dict())
                update(net,opts,[row])
                for n,v in before.items(): self.assertTrue(torch.equal(v,net.state_dict()[n]), n)
                now=opts[other].state_dict()
                for k,v in adam['state'].items():
                    for f,t in v.items(): self.assertTrue(torch.equal(t,now['state'][k][f]))
            before={n:v.clone() for n,v in net.state_dict().items() if not n.startswith(('value_net.','wdl.'))}
            update(net,opts,[dict(playing,value_actor=0)],value_only=True)
            for n,v in before.items(): self.assertTrue(torch.equal(v,net.state_dict()[n]),n)
            wrong=dict(opts,opening=opts['playing'])
            with self.assertRaisesRegex(ValueError,'ownership'):update(net,wrong,[opening])
            with self.assertRaisesRegex(ValueError,'Group policy'):update(net,opts,[opening,playing])

    def test_existing_value_calibration_preserves_both_policy_branches(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);deck=self.bank(root)['lingke'];net,_,_=dispatch.restore(root/'phase','lingke')
            before={n:v.clone()for n,v in net.state_dict().items()if not n.startswith(('value_net.','wdl.'))}
            rows=self.rows('lingke',deck)
            opt=torch.optim.Adam(net.value_parameters(),lr=.0003)
            dispatch.update(net,opt,rows,value_only=True)
            dispatch.update(net,opt,rows,warmup=True)
            for name,value in before.items():self.assertTrue(torch.equal(value,net.state_dict()[name]),name)

    def test_batched_inference_routes_on_host_and_matches_numpy(self):
        from app.modules.card_game.rl.batched_search.inference import BatchedInference, InferenceRequest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);deck=self.bank(root)['lingke']; model=dispatch.model(root/'phase','lingke')
            engine=BatchedInference({'lingke':model},device='cpu');rows=self.rows('lingke',deck)
            requests=[InferenceRequest(i,'lingke',model.version,r['x'],r['c'],True)for i,r in enumerate(rows)]
            net=engine._networks['lingke'];calls=[];original=net.policy_logits
            def record(x,c,mask,phase,**kwargs):calls.append(phase);return original(x,c,mask,phase,**kwargs)
            with patch.object(net,'policy_logits',side_effect=record),patch.object(net,'forward',side_effect=AssertionError('no mixed forward in fastpath')),patch('app.modules.card_game.rl.seven_policy.canonical_scores_tensor',side_effect=AssertionError('no CUDA-to-host candidate copy')):
                results=engine.predict(requests)
            self.assertEqual(calls,['opening','playing'])
            for r,result in zip(rows,results):
                np.testing.assert_allclose(result.logits,model.scores(r['x'],r['c']),rtol=2e-4,atol=2e-4)
                np.testing.assert_allclose(result.wdl,model.wdl(r['x'],True),rtol=2e-4,atol=2e-4)
            value_request = InferenceRequest('value','lingke',model.version,rows[0]['x'],rows[0]['c'][:0],False,value_only=True)
            with patch.object(net,'policy_logits',side_effect=AssertionError('value queries skip policy')):
                value = engine.predict([value_request])[0]
            self.assertEqual(len(value.logits),0)
            np.testing.assert_allclose(value.wdl,model.wdl(rows[0]['x'],False),atol=2e-4,rtol=2e-4)

    def test_frozen_teacher_fastpath_handles_supported_and_unknown_roots(self):
        from app.modules.card_game.rl import phase_runtime
        from app.modules.card_game.rl.batched_search.inference import BatchedInference, InferenceRequest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);decks=plan()['builds'];net=old.create_network('starter',16,legacy=True)
            old.export(net,root/'old','starter',decks['starter'],{'fixture':'teacher CPU'})
            phase_runtime.migrate(root/'old',root/'phase','starter')
            model=dispatch.model(root/'phase','starter');engine=BatchedInference({'starter':model})
            rows=self.rows('starter',decks['starter'])+self.rows('lingke',decks['lingke'])
            requests=[InferenceRequest(i,'starter',model.version,r['x'],r['c'],True)for i,r in enumerate(rows)]
            loaded=engine._networks['starter']
            with patch.object(loaded.legacy,'forward',side_effect=AssertionError('no legacy CUDA support sync')),patch('app.modules.card_game.rl.seven_policy.canonical_scores_tensor',side_effect=AssertionError('no candidate D2H')):
                results=engine.predict(requests)
            for row,result in zip(rows,results):
                np.testing.assert_allclose(result.logits,model.scores(row['x'],row['c']),atol=2e-4,rtol=2e-4)

    def test_explicit_website_binding_loads_phase_model_without_changing_live_assets(self):
        import json
        from hashlib import sha256
        from app.modules.card_game.engine.ai import advanced_model as ai
        from app.modules.card_game.engine.duel_v2 import legal_actions
        from app.modules.card_game.rl import recovery_search, seven_lineup, phase_policy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.bank(root)
            release = root/'seven-20261024'
            (root/'phase').rename(release)
            manifest = json.loads((release/'lingke.json').read_text())
            binding = dict(kind='explicit_user_candidate', request='local fixture only', quality_approved=False,
                           runtime_rule_hash=seven_lineup.identity(), models={'lingke': dict(
                           manifest_sha256=sha256((release/'lingke.json').read_bytes()).hexdigest(),
                           model_sha256=manifest['sha256'], build_sha256=manifest['build_sha256'])})
            (release/'serving.json').write_text(json.dumps(binding))
            with patch.object(ai.config, 'DUEL_AI_MODEL_DIR', root):
                model = ai.load_model('lingke')
                self.assertEqual(model.schema, phase_policy.SCHEMA)
                self.assertEqual(ai.model_build_hash(model, model.serving_deck), manifest['build_sha256'])
                ai.require_human_deck(model, model.serving_deck, 'lingke')
                state = new_game(seed=61035, first_side='a', decks={'a':model.serving_deck,'b':model.serving_deck})
                state['ai_profile'] = dict(deck_id='lingke', version=model.version, build_sha256=model.serving_build_sha256)
                actual = recovery_search.search
                records = []
                def record(*args, **kwargs):
                    result = actual(*args, **kwargs); records.append(result); return result
                with patch.object(recovery_search, 'search', side_effect=record):
                    action = ai.choose_action(state, 'a')
                self.assertIn(action, legal_actions(state, 'a'))
                self.assertTrue(records and records[0]['simulations'] > 0)
                with patch.object(phase_policy, 'fingerprint', return_value='changed'):
                    with self.assertRaises(ai.RuleValidationError): ai.load_model('lingke')

    def test_all_seven_once_only_update_and_restoration(self):
        from app.modules.card_game.rl import phase_hybrid, hybrid_joint_update
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);decks=self.bank(root,ROSTER);jobs=[];games=[]
            for k in ROSTER:
                rows=self.rows(k,decks[k]);values=[dict(x=r['x'].copy(),side=s,key=k,step=r['step'],z=1 if s=='a' else -1,value_actor=int(s=='a')) for r in rows for s in ('a','b')]
                jobs.append(dict(id='controlled-'+k,policies={s:(str(root/'phase'),k)for s in ('a','b')},train_sides=['a','b']))
                games.append(dict(complete=True,decisions=2,episode_actions=[dict(side='a',action=r['action'])for r in rows],rows=rows,value_rows=values,winner='a'))
            config=dict(device='cpu',policy_layout=dispatch.LAYOUT,learning_keys=list(ROSTER),policy_passes=1,max_row_reuse=4,seed=3,train_until=time.time()+180,hard_deadline=time.time()+240)
            with self.assertRaises(ValueError):
                hybrid_joint_update.run_joint_update(root/'phase',jobs,games,root/'rejected',config,deadline=time.time()+120)
            receipt=hybrid_joint_update._run_once(root/'phase',jobs,games,root/'stage',config,deadline=time.time()+120,contract='synthetic-not-readiness',executor=phase_hybrid._execute_update)
            self.assertFalse(receipt['quality_approved'])
            nets,opts,pools,rng,saved=hybrid_joint_update.load_joint_state(root/'stage')
            self.assertEqual(set(opts),set(ROSTER))
            for k in ROSTER:
                self.assertEqual(set(opts[k]),{'opening','playing','value'})
                self.assertEqual(receipt['health'][k]['coverage'],1)
                self.assertTrue(receipt['restore_checks'][k]['adam_next_step_equal'])
            again=hybrid_joint_update._run_once(root/'phase',jobs,games,root/'stage',config,deadline=time.time()-1,contract='synthetic-not-readiness',executor=phase_hybrid._execute_update)
            self.assertEqual(again['files'],receipt['files'])


if __name__ == '__main__':unittest.main()
