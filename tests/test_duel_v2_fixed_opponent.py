"""Fixed-matchup PPO uses the requested deck and a single immutable opponent."""
import unittest
from unittest.mock import patch

class FixedOpponentTest(unittest.TestCase):
    def test_reset_preserves_fixed_cards_and_policy_across_seats(self):
        import torch
        from app.modules.card_game.rl.public_training import PublicTrainingEnv, FrozenLeague
        from app.modules.card_game.rl.capability import preset_opponent_deck
        from app.modules.card_game.rl.gpu_duel.catalog import encode_public_deck_row
        class Backend:
            def __init__(self,*a,**k): self.calls=[]
            def reset_public_gpu_state(self,state,seeds,a,b,**kwargs):
                self.calls.append((a.clone(),b.clone(),kwargs))
            def close(self): pass
        class Env(PublicTrainingEnv):
            def prepare_decision(self): pass
        foe=preset_opponent_deck('weave-rush')
        league=FrozenLeague(learn_mulligan=True,greedy=True)
        peer=torch.nn.Linear(1,1);peer._trained_deck='weave-rush';league.add_pinned(peer)
        with patch('app.modules.card_game.rl.gpu_duel.compiled_backend.CompiledStarterBackend',Backend):
            env=Env(8,deck_id='starter',league=league,device='cpu',learn_mulligan=True,
                    random_team_fraction=0,rule_fraction=0,fixed_opponent_deck=foe)
            env.learner=torch.arange(8,dtype=torch.int32)%2
            for _ in range(3):
                env.reset();a,b,kw=env.compiled.calls[-1]
                actual=torch.where((env.learner==0)[:,None],b,a)
                expected=torch.tensor(encode_public_deck_row(foe)).expand(8,-1)
                self.assertTrue(torch.equal(actual,expected))
                self.assertTrue(bool((env.assignment==1).all()))
                self.assertTrue(bool((env.opponent_lineup==1).all()))
                self.assertTrue(kw['escalation']);self.assertTrue(kw['mulligan'])
            with self.assertRaises(ValueError):
                Env(8,deck_id='starter',league=league,device='cpu',learn_mulligan=True,
                    random_team_fraction=.2,rule_fraction=0,fixed_opponent_deck=foe)

    def test_n06_enumerates_all_legal_replacements_only(self):
        from collections import Counter
        from scripts.train_duel_v2_fixed_n06 import enumerate_builds
        from app.modules.card_game.rl.capability import preset_opponent_deck
        b=preset_opponent_deck('starter')
        b['card_ids']=[c for c in b['card_ids'] if not c.startswith('N')]+['N02']*2+['N03']*2+['N04']*2+['N07','N08']
        rows=enumerate_builds(b)
        self.assertEqual(Counter(r['copies'] for r in rows),{0:1,1:5,2:13})
        for row in rows:
            self.assertEqual(Counter(c for c in row['build']['card_ids'] if not c.startswith('N')),
                             Counter(c for c in b['card_ids'] if not c.startswith('N')))
            self.assertEqual(row['build']['card_ids'].count('N06'),row['copies'])
