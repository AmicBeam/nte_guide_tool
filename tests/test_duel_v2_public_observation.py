from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from app.modules.card_game.engine.duel_v2 import new_game,observe,legal_actions
from app.modules.card_game.rl.capability import preset_opponent_deck,sample_public_deck
from app.modules.card_game.rl.public_observation import encode_public
from app.modules.card_game.rl.public_schema import SCHEMA,state_dim,rule_identity,CAND_DIM


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
class PublicObservationTest(unittest.TestCase):
    def state(self):
        return new_game(seed=7,first_side='a',skip_mulligan=True,
                        decks={'a':preset_opponent_deck('starter'),'b':sample_public_deck(3)})

    def encoded(self,state):
        actions=[a for a in legal_actions(state,'a') if a['type']!='concede']
        return encode_public(observe(state,'a'),actions)

    def test_escalation_and_persistent_flags_are_distinguishable(self):
        import numpy as np
        state=self.state();state['turn']=6
        baseline=self.encoded(state)[0]
        self.assertEqual(len(baseline),state_dim())
        for edit in ('ultimates_used','hp_floor','pending_atk','next_followup','escalation'):
            changed=deepcopy(state)
            if edit=='ultimates_used':changed['sides']['a']['ultimates_used']=1
            elif edit=='escalation':changed['escalation_enabled']=False
            else:changed['sides']['a']['characters']['nanali']['flags'][edit]=1
            self.assertFalse(np.array_equal(baseline,self.encoded(changed)[0]),edit)

    def test_hidden_cards_rng_and_deck_order_do_not_leak(self):
        import numpy as np
        state=self.state();before=self.encoded(state)
        changed=deepcopy(state);changed['rng']=999
        for side in ('a','b'):changed['sides'][side]['deck'].reverse()
        changed['sides']['b']['hand'].reverse()
        for card in changed['sides']['b']['hand']:card['card_id']='N04'
        after=self.encoded(changed)
        for x,y in zip(before,after):np.testing.assert_array_equal(x,y)

    def test_known_enemy_card_changes_public_observation(self):
        import numpy as np
        state=self.state();before=self.encoded(state)[0]
        state['sides']['b']['revealed_ids']=[state['sides']['b']['hand'][0]['instance_id']]
        self.assertFalse(np.array_equal(before,self.encoded(state)[0]))

    def test_revealed_copy_is_distinguishable_as_an_action(self):
        import numpy as np
        from app.modules.card_game.engine.duel_v2.state import card_instance
        state=self.state()
        cards=[card_instance(state,'N03','a') for _ in range(2)]
        state['sides']['a']['hand']=cards
        state['sides']['a']['revealed_ids']=[cards[0]['instance_id']]
        actions=[a for a in legal_actions(state,'a') if a['type']=='play_card']
        _,candidates=encode_public(observe(state,'a'),actions)
        self.assertEqual(candidates.shape,(2,CAND_DIM))
        np.testing.assert_array_equal(candidates[0,:10],candidates[1,:10])
        self.assertEqual(set(candidates[:,-1]),{0.,1.})

    def test_legacy_visible_encoder_matches_existing_weights_contract(self):
        import numpy as np
        from app.modules.card_game.engine.ai.advanced_model import encode
        from app.modules.card_game.engine.duel_v2 import apply_action,acting_side,choose_action
        state=self.state()
        for _ in range(12):
            side=acting_side(state)
            if side is None:break
            actions=[a for a in legal_actions(state,side) if a['type']!='concede']
            old=encode(state,side,actions)
            public=encode_public(observe(state,side),actions,schema='resident_public_v1')
            for a,b in zip(old,public):np.testing.assert_array_equal(a,b)
            state=apply_action(state,side,choose_action(state,side))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch export verification')
class ExportTest(unittest.TestCase):
    def test_legacy_transfer_preserves_shared_feature_scores(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,CARD_IDS,SEATS
        from app.modules.card_game.rl.public_schema import LEGACY_SIDE,LEGACY_CHAR
        from app.modules.card_game.rl.public_training import warm_start_weights
        legacy_dim=2+2*len(LEGACY_SIDE)+2*len(SEATS)*len(LEGACY_CHAR)+3*len(CARD_IDS)
        old=CompactScorer(legacy_dim,10,8)
        new=CompactScorer(state_dim(),CAND_DIM,8)
        warm_start_weights({'schema':'resident_public_v1','deck':'starter','gpu_lock':GPU_LOCK,
                            'model':old.state_dict()},new,'starter')
        state=new_game(seed=7,skip_mulligan=True,first_side='a',
                       decks={'a':preset_opponent_deck('starter'),'b':sample_public_deck(3)})
        actions=[a for a in legal_actions(state,'a') if a['type']!='concede']
        v=observe(state,'a')
        x0,c0=encode_public(v,actions,schema='resident_public_v1')
        x1,c1=encode_public(v,actions)
        mask=torch.ones(1,len(actions),dtype=torch.bool)
        with torch.no_grad():
            a,_=old(torch.tensor(x0)[None],torch.tensor(c0)[None],mask)
            b,_=new(torch.tensor(x1)[None],torch.tensor(c1)[None],mask)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_gpu_sampler_and_frozen_history_contract(self):
        import torch
        from app.modules.card_game.rl.public_training import tensor_public_decks,FrozenLeague
        from app.modules.card_game.rl.gpu_duel.catalog import SEATS,CARD_IDS
        from app.modules.card_game.content.duel_v2 import validate_deck
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        torch.manual_seed(10)
        for index,row in enumerate(tensor_public_decks(64,'cpu').tolist()):
            validate_deck({'id':str(index),'name':'sampler',
                           'character_ids':[SEATS[k] for k in row[:4]],
                           'card_ids':[CARD_IDS[k] for k in row[4:]]})
        net=CompactScorer(state_dim(),CAND_DIM,8)
        league=FrozenLeague(2);league.add(net)
        first=next(league.history[0].parameters()).clone()
        with torch.no_grad():next(net.parameters()).add_(10)
        self.assertTrue(torch.equal(first,next(league.history[0].parameters())))
        self.assertTrue(all(not p.requires_grad for p in league.history[0].parameters()))
        league.add(net);league.add(net)
        self.assertEqual(len(league.history),2)

    def test_export_is_numeric_and_stays_unvalidated(self):
        import torch
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK
        from app.modules.card_game.rl.public_export import export_checkpoint
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel,require_human_deck
        from app.errors import RuleValidationError
        net=CompactScorer(state_dim(),CAND_DIM,16)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint=root/'test.pt'
            torch.save({'model':net.state_dict(),'schema':SCHEMA,'deck':'starter','hidden':16,
                        'state_dim':state_dim(),'cand_dim':CAND_DIM,'gpu_lock':GPU_LOCK,
                        'update':0,'rule_identity':{'rule_hash':rule_identity()}},checkpoint)
            result=export_checkpoint(checkpoint,root/'export')
            self.assertFalse(result['validated_for_serving'])
            model=FrozenModel(root/'export','starter')
            state=new_game(seed=7,skip_mulligan=True,first_side='a',
                           decks={'a':preset_opponent_deck('starter'),'b':sample_public_deck(8)})
            actions=[a for a in legal_actions(state,'a') if a['type']!='concede']
            choice=model.select_public_action(observe(state,'a'),actions)
            self.assertLess(choice,len(actions))
            with self.assertRaises(RuleValidationError):
                require_human_deck(model,preset_opponent_deck('starter'),'starter')
