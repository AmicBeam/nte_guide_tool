"""Fixed-team construction, synthetic learning, and bounded battle evaluation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class TuningCliTest(unittest.TestCase):
    def test_check_does_not_import_training(self):
        import contextlib,io,sys
        from scripts.train_duel_v2_cards import main
        with patch.dict(sys.modules,{'torch':None}),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--check','--team','weave-rush']),0)
        result=json.loads(output.getvalue())
        self.assertFalse(result['learns_characters'])
        self.assertFalse(result['training_started'])
        self.assertEqual(result['character_ids'],['bohe','baicang','zero','iloy'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required for card-tuning unit checks')
class CardTuningTest(unittest.TestCase):
    def test_battle_checkpoint_is_frozen_and_rejects_stale_rules(self):
        import torch
        from app.modules.card_game.rl.card_tuning import load_battle_policy
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
        from app.modules.card_game.rl.public_schema import SCHEMA, state_dim, CAND_DIM, rule_identity
        network = CompactScorer(state_dim(), CAND_DIM, 8)
        checkpoint = dict(schema=SCHEMA, hidden=8, gpu_lock=GPU_LOCK,
                          rule_identity={'rule_hash': rule_identity()}, model=network.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            torch.save(checkpoint, path)
            frozen, metadata = load_battle_policy(path, 'cpu')
            self.assertFalse(frozen.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            self.assertEqual(len(metadata['sha256']), 64)
            checkpoint['rule_identity']['rule_hash'] = 'stale'
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_battle_policy(path, 'cpu')

    def test_every_public_team_has_legal_fixed_card_candidates(self):
        from itertools import combinations
        from app.modules.card_game.rl.card_tuning import CardTuner
        from app.modules.card_game.rl.capability import PUBLIC_CHARACTERS
        for team in combinations(PUBLIC_CHARACTERS, 4):
            tuner = CardTuner(team, 8)
            for row in tuner.sample(3)['rows']:
                self.assertEqual(tuner.deck(row)['character_ids'], list(team))

    def test_policy_keeps_team_and_learns_only_card_choices(self):
        import torch
        from app.modules.card_game.rl.card_tuning import CardTuner,update_tuner
        team=['bohe','baicang','zero','iloy']
        tuner=CardTuner(team,16)
        optimizer=torch.optim.Adam(tuner.parameters(),lr=.01)
        before={k:p.detach().clone() for k,p in tuner.named_parameters()}
        torch.manual_seed(17)
        for _ in range(2):
            batch=tuner.sample(24)
            self.assertTrue(torch.equal(batch['rows'][:,:4],tuner.seats[None].expand(24,-1)))
            for row in batch['rows']:self.assertEqual(tuner.deck(row)['character_ids'],team)
            self.assertTrue(batch['log_prob'].requires_grad)
            self.assertTrue(torch.isfinite(batch['entropy']).all())
            # Synthetic varied returns test the gradient path, not game strength.
            stats=update_tuner(tuner,optimizer,batch,torch.arange(24,dtype=torch.float32)/12-1,
                               torch.ones(24,dtype=torch.bool),entropy_coef=0.0)
            self.assertTrue(stats['updated'])
        self.assertTrue(any(not torch.equal(before[k],p) for k,p in tuner.named_parameters()))
        self.assertTrue(all(k.startswith('net.') for k in tuner.state_dict()))

    def test_truncations_never_update_policy(self):
        import torch
        from app.modules.card_game.rl.card_tuning import CardTuner,update_tuner
        tuner=CardTuner(['nanali','iloy','zero','jiuyuan'],8)
        optimizer=torch.optim.Adam(tuner.parameters())
        before={k:v.clone() for k,v in tuner.state_dict().items()}
        result=update_tuner(tuner,optimizer,tuner.sample(2),torch.ones(2),torch.zeros(2,dtype=torch.bool))
        self.assertFalse(result['updated'])
        self.assertTrue(all(torch.equal(v,tuner.state_dict()[k]) for k,v in before.items()))

    def test_native_battles_are_bounded_and_do_not_train_play_policy(self):
        import torch
        from app.modules.card_game.rl.card_tuning import CardTuner,CardBattleEvaluator
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.public_schema import state_dim,CAND_DIM
        net=CompactScorer(state_dim(),CAND_DIM,8).eval()
        before={k:v.clone() for k,v in net.state_dict().items()}
        tuner=CardTuner(['nanali','iloy','zero','jiuyuan'],8)
        with tempfile.TemporaryDirectory() as directory:
            evaluator=CardBattleEvaluator(net,device='cpu',compiled_dir=directory)
            try:
                scores=evaluator.evaluate(tuner.sample(2)['rows'],pairs=1,max_actions=2)
                self.assertEqual(scores['rewards'].shape,(2,))
                self.assertFalse(scores['valid'].any())
                self.assertEqual(int(scores['truncated'].sum()),4)
                self.assertTrue(all(torch.equal(v,net.state_dict()[k]) for k,v in before.items()))
            finally:evaluator.close()

    def test_finished_native_games_produce_terminal_scores(self):
        import torch
        from app.modules.card_game.rl.card_tuning import CardTuner,CardBattleEvaluator
        class EndPolicy(torch.nn.Module):
            def forward(self,state,candidates,mask):
                logits=(candidates[:,:,0]==0).float().masked_fill(~mask,-torch.inf)
                return logits,torch.zeros(len(state))
        with tempfile.TemporaryDirectory() as directory:
            evaluator=CardBattleEvaluator(EndPolicy(),device='cpu',compiled_dir=directory)
            try:
                tuner=CardTuner(['bohe','baicang','zero','iloy'],8)
                torch.manual_seed(4)
                result=evaluator.evaluate(tuner.sample(2)['rows'],pairs=1,max_actions=160)
                self.assertTrue(result['valid'].all())
                self.assertEqual(int(result['truncated'].sum()),0)
                torch.testing.assert_close(result['rewards']*result['trials'],(result['wins']-result['losses']).float())
            finally:evaluator.close()


class CandidateAnalysisTest(unittest.TestCase):
    def test_custom_cards_use_existing_analysis_and_cannot_approve_web_model(self):
        from app.modules.card_game.rl.capability import preset_opponent_deck
        from app.modules.card_game.rl.offline_analysis import run_offline_analysis,export_game_replay
        from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
        from app.modules.card_game.rl.public_export import approve_candidate
        deck=preset_opponent_deck('starter')
        deck['card_ids']=['N01' if c=='N02' else c for c in deck['card_ids']]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            report=run_offline_analysis(root/'report',VisibleEngineRulePolicy(),learner_deck=deck,
                pairs=1,max_actions=2,max_seconds=10,seed_base=1)
            self.assertFalse(report['learner_is_preset'])
            self.assertEqual(report['learner_build']['card_ids'],deck['card_ids'])
            export_game_replay(root/'report',report['games_index'][0]['game_id'],root/'replay.json')
            self.assertEqual(json.loads((root/'replay.json').read_text())['visibility'],'spectator_public_v1')
            (root/'starter.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'tuned-deck report'):
                approve_candidate(root/'starter.npz',root/'report/report.json')
