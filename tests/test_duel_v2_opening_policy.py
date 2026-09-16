"""Opening contracts; synthetic networks demonstrate capability, not trained strength."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.modules.card_game.engine.duel_v2 import new_game,observe,legal_actions,apply_action,acting_side
from app.modules.card_game.rl.capability import preset_opponent_deck
from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA,OPENING_CAND_DIM,encode_opening


class OpeningObservationTest(unittest.TestCase):
    def game(self,foe='weave-rush',first='a'):
        return new_game(seed=7,first_side=first,skip_mulligan=False,
                        decks={'a':preset_opponent_deck('starter'),'b':preset_opponent_deck(foe)})

    def encoded(self,state):
        actions=[a for a in legal_actions(state,'a') if a['type']=='mulligan']
        return encode_opening(observe(state,'a'),actions),actions

    def test_own_hand_opponent_roles_and_seat_are_visible_but_hidden_cards_are_not(self):
        import numpy as np
        state=self.game();before,actions=self.encoded(state)
        self.assertEqual(before[1].shape,(26,OPENING_CAND_DIM))
        for row,action in zip(before[1],actions):self.assertEqual(row[11:].sum(),len(action['card_ids'])/2)
        changed=deepcopy(state);changed['rng']=91234
        for side in ('a','b'):changed['sides'][side]['deck'].reverse()
        for card in changed['sides']['b']['hand']:card['card_id']='N01'
        after,_=self.encoded(changed)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(before[0],self.encoded(self.game('starter'))[0][0]))
        self.assertFalse(np.array_equal(before[0],self.encoded(self.game(first='b'))[0][0]))

    def test_check_reports_joint_schema_without_loading_torch(self):
        import contextlib,io,sys
        from scripts.train_duel_v2_public import main
        with patch.dict(sys.modules,{'torch':None}),contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--check'])
        result=json.loads(output.getvalue())
        self.assertEqual(result['schema'],OPENING_SCHEMA)
        self.assertTrue(result['learn_mulligan'])
        self.assertEqual(result['candidate_dim'],OPENING_CAND_DIM)
        self.assertFalse(result['training_started'])

    def test_opening_telemetry_is_private_and_separates_replacement_from_first_turn_draw(self):
        from app.modules.card_game.rl.report_telemetry import GameTelemetry
        from app.modules.card_game.rl.offline_analysis import _public_payload
        state=self.game(first='b');opening=deepcopy(state);t=GameTelemetry(state)
        for side in ('a','b'):
            action={'type':'mulligan','card_ids':[state['sides'][side]['hand'][0]['instance_id']]}
            after=apply_action(state,side,action);t.after(state,after,action,side);state=after
        data=t.finish(state)
        for side in ('a','b'):
            self.assertEqual(len(data['opening'][side]['kept']),4)
            self.assertEqual(len(data['opening'][side]['replaced']),1)
            self.assertEqual(len(data['opening'][side]['replacement_draws']),1)
        for viewer in ('a','b'):
            view=observe(state,viewer)
            for events in (view['events'],view['presentation']['events']):
                for event in events:
                    if event.get('type')=='mulligan' and event.get('side')!=viewer:
                        self.assertNotIn('card_ids',event)
                    elif event.get('type')=='mulligan':self.assertEqual(len(event['card_ids']),1)
        public=json.dumps(_public_payload(opening,state,game_id='fixture'))
        self.assertNotIn('replacement_draws',public)
        self.assertNotIn('initial_hand',public)
        self.assertNotIn('first_turn_actions',public)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
class OpeningNetworkTest(unittest.TestCase):
    def test_v2_warm_start_preserves_play_scores_and_new_opening_features_can_learn(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
        from app.modules.card_game.rl.public_schema import SCHEMA,CAND_DIM,state_dim
        from app.modules.card_game.rl.public_training import warm_start_weights
        from app.modules.card_game.rl.public_observation import encode_public
        torch.manual_seed(19)
        old=CompactScorer(state_dim(),CAND_DIM,8)
        new=CompactScorer(state_dim(),OPENING_CAND_DIM,8)
        warm_start_weights({'schema':SCHEMA,'deck':'starter','gpu_lock':GPU_LOCK,'model':old.state_dict()},new,'starter')
        state=new_game(seed=7,first_side='a',skip_mulligan=True)
        actions=[a for a in legal_actions(state,'a') if a['type']!='concede'];view=observe(state,'a')
        x,c=encode_public(view,actions);x2,c2=encode_opening(view,actions)
        mask=torch.ones(1,len(actions),dtype=torch.bool)
        a=old(torch.tensor(x)[None],torch.tensor(c)[None],mask)[0]
        b=new(torch.tensor(x2)[None],torch.tensor(c2)[None],mask)[0]
        torch.testing.assert_close(a,b)
        state=new_game(seed=7,first_side='a',skip_mulligan=False)
        actions=[a for a in legal_actions(state,'a') if a['type']=='mulligan'];x,c=encode_opening(observe(state,'a'),actions)
        logits,_=new(torch.tensor(x)[None],torch.tensor(c)[None],torch.ones(1,len(actions),dtype=torch.bool))
        loss=-torch.log_softmax(logits,1)[0,-1];loss.backward()
        self.assertGreater(float(new.cand_net[0].weight.grad[:,11:].abs().sum()),0)

    def test_exported_joint_policy_scores_mulligan_while_legacy_keeps_all(self):
        import torch
        import numpy as np
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
        from app.modules.card_game.rl.public_schema import state_dim,rule_identity
        from app.modules.card_game.rl.public_export import export_checkpoint
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        with torch.random.fork_rng(), tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(20260916)
            root=Path(directory);net=CompactScorer(state_dim(),OPENING_CAND_DIM,8)
            torch.save({'schema':OPENING_SCHEMA,'deck':'starter','gpu_lock':GPU_LOCK,'model':net.state_dict(),
                'hidden':8,'state_dim':state_dim(),'cand_dim':OPENING_CAND_DIM,'update':1,
                'rule_identity':{'rule_hash':rule_identity(OPENING_SCHEMA)},'learn_mulligan':True,
                'opening_policy':'learned_joint_v1'},root/'synthetic.pt')
            export_checkpoint(root/'synthetic.pt',root/'export')
            model=FrozenModel(root/'export','starter')
            state=new_game(seed=7,skip_mulligan=False)
            actions=[a for a in legal_actions(state,'a') if a['type']=='mulligan']
            view=observe(state,'a')
            with patch.object(model,'scores',side_effect=lambda x,c:c[:,11:].sum(1)) as scored:
                selected=model.select_public_action(view,actions)
                self.assertEqual(len(actions[selected]['card_ids']),3);self.assertTrue(scored.called)
            # Admission fixtures are synthetic, not measured game strength.
            import itertools
            from app.modules.card_game.rl.capability import PUBLIC_CHARACTERS
            from app.modules.card_game.rl.offline_analysis import frozen_model_policy
            from app.modules.card_game.rl.public_export import approve_candidate
            teams=list(itertools.combinations(PUBLIC_CHARACTERS,4))
            report={'model':frozen_model_policy(model).model_metadata,'preset':'starter','sampled_opponent':True,
                'by_opponent_team':{str(i):{} for i in range(15)},'pairs':32,
                'engine':{'escalation':True,'first_turn_draw':True,'skip_mulligan':False},
                'games_index':[{'pair':i,'position':position,'opponent_team':teams[i%15],
                    'terminated':True,'winner':'a','action_types':['mulligan','end_turn']}
                    for i in range(32) for position in ('first','second')]}
            report_path=root/'admission-fixture.json'
            report['model']['opening_override']='keep_all';report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'actual learned-mulligan'):
                approve_candidate(root/'export/starter.npz',report_path)
            del report['model']['opening_override'];report_path.write_text(json.dumps(report))
            approve_candidate(root/'export/starter.npz',report_path)
            self.assertTrue(FrozenModel(root/'export','starter').capability['human_public_custom'])
            model.schema='resident_public_v2'
            with patch.object(model,'scores',side_effect=AssertionError('legacy must not infer mulligan')):
                self.assertEqual(actions[model.select_public_action(view,actions)]['card_ids'],[])

    def test_native_rollout_contains_mulligan_and_pinned_peer_survives_history_eviction(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer,collect_rollout
        from app.modules.card_game.rl.public_schema import state_dim
        from app.modules.card_game.rl.public_training import FrozenLeague,PublicTrainingEnv
        from app.modules.card_game.rl.opening_observation import opening_observe
        net=CompactScorer(state_dim(),OPENING_CAND_DIM,8)
        net._trained_deck='starter'
        peer=deepcopy(net);peer._trained_deck='weave-rush'
        league=FrozenLeague(1,learn_mulligan=True);league.add(net);league.add_pinned(peer)
        pinned=league.policies[0];league.add(net);league.add(net)
        self.assertIs(league.policies[0],pinned)
        with tempfile.TemporaryDirectory() as directory:
            env=PublicTrainingEnv(4,deck_id='starter',league=league,device='cpu',compiled_dir=directory,learn_mulligan=True,random_team_fraction=0,rule_fraction=0)
            try:
                for label,assignment in zip(env.opponent_lineup.tolist(),env.assignment.tolist()):
                    self.assertEqual(league.policies[assignment-1]._trained_deck,('starter','weave-rush')[label])
                before={k:v.clone() for k,v in net.state_dict().items()}
                result=collect_rollout(env,net,2,observe_fn=opening_observe)
                self.assertEqual(result['cand'].shape[-1],OPENING_CAND_DIM)
                phase=result['state'][:,3]
                self.assertTrue(bool((phase==0).any()))
                self.assertGreater(sum(sum(r) for r in env.opening_summary()['mulligan_by_position_and_count']),0)
                self.assertTrue(all(torch.equal(before[k],v) for k,v in net.state_dict().items()))
            finally:env.close()


class OpeningReportTest(unittest.TestCase):
    def test_real_keep_all_games_have_opening_and_first_turn_summary(self):
        from app.modules.card_game.rl.offline_analysis import run_offline_analysis
        from app.modules.card_game.rl.opening_report import summarize_opening,write_opening_report
        def end_policy(view,actions):
            return next(i for i,a in enumerate(actions) if (a['type']=='mulligan' and not a['card_ids']) or a['type']=='end_turn')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            report=run_offline_analysis(root/'games',end_policy,opponent_policy=end_policy,
                opponent_deck=preset_opponent_deck('weave-rush'),pairs=1,max_actions=160,max_seconds=30,seed_base=11)
            self.assertTrue(report['complete'])
            summary=summarize_opening(root/'games')
            self.assertTrue(summary['complete'])
            for position,data in summary['by_position'].items():
                self.assertEqual(data['mulligan_count_games'],{0:1})
                self.assertEqual(data['before_first_turn_end']['mean_ap'],1 if position=='first' else 2)
            self.assertFalse(write_opening_report(root,[{'learner':'starter','opponent':'weave-rush','mode':'learned','summary':summary}]))
            self.assertIn('首动作',(root/'opening-report.md').read_text())

    def test_eight_cell_driver_pairs_same_models_and_rejects_changed_controls(self):
        # Synthetic deterministic scorer exercises the real official-engine/report path.
        import contextlib,io
        from types import SimpleNamespace,MethodType
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        from app.modules.card_game.rl.opening_report import complete_opening_matrix
        from scripts.analyze_duel_v2_opening import main
        def model_factory(directory,key):
            deck=preset_opponent_deck(key)
            model=SimpleNamespace(schema=OPENING_SCHEMA,version='synthetic-'+key,serving_deck=deck,
                manifest={'deck':key,'opening_policy':'learned_joint_v1','rule_identity':{'rule_hash':'synthetic'},'learner_deck':deck},
                scores=lambda state,candidates:candidates[:,11:].sum(1))
            model.select_public_action=MethodType(FrozenModel.select_public_action,model)
            return model
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'report'
            with patch('app.modules.card_game.engine.ai.advanced_model.FrozenModel',side_effect=model_factory), \
                 patch('app.modules.card_game.rl.offline_analysis.resolve_numeric_model_dir',side_effect=lambda path,deck_id:(path,deck_id)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--starter-model','synthetic-starter','--weave-model','synthetic-weave',
                    '--pairs','1','--seconds','90','--output',str(root)]),0)
            result=json.loads((root/'opening-summary.json').read_text())
            self.assertTrue(result['complete']);self.assertEqual(len(result['cells']),8)
            for row in result['cells']:
                expected='3' if row['mode']=='learned' else '0'
                for data in row['summary']['by_position'].values():
                    self.assertEqual(data['mulligan_count_games'],{expected:1})
            changed=deepcopy(result['cells']);changed[0]['seed_base']+=1
            self.assertFalse(complete_opening_matrix(changed))
            changed=deepcopy(result['cells']);changed[0]['summary']['opponent_model']['sha256']='different'
            self.assertFalse(complete_opening_matrix(changed))
            self.assertFalse(complete_opening_matrix(result['cells'][:-1]))
