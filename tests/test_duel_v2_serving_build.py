"""Synthetic manifest/evidence fixtures test binding, not game strength."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
class ServingBuildTest(unittest.TestCase):
    def fixture(self, root):
        import numpy as np
        from app.modules.card_game.rl.public_schema import SCHEMA,state_dim,rule_identity,CAND_DIM
        from app.modules.card_game.rl.capability import PUBLIC_CHARACTERS,preset_opponent_deck
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,CARD_IDS,SEATS
        deck=preset_opponent_deck('starter');deck['card_ids']=['N01' if c=='N02' else c for c in deck['card_ids']]
        shapes={'state_net.0.weight':(1,state_dim()),'state_net.0.bias':(1,),'state_net.2.weight':(1,1),'state_net.2.bias':(1,),
                'cand_net.0.weight':(1,CAND_DIM),'cand_net.0.bias':(1,),'cand_net.2.weight':(1,1),'cand_net.2.bias':(1,),
                'score.weight':(1,1),'score.bias':(1,)}
        model=root/'starter.npz';np.savez_compressed(model,**{k:np.zeros(v,np.float32) for k,v in shapes.items()})
        digest=hashlib.sha256(model.read_bytes()).hexdigest();identity=rule_identity()
        manifest={'schema':SCHEMA,'deck':'starter','gpu_lock':GPU_LOCK,'card_ids':list(CARD_IDS),'seat_ids':list(SEATS),
                  'sha256':digest,'rule_identity':{'rule_hash':identity},'hidden':1,'state_dim':state_dim(),'cand_dim':CAND_DIM,
                  'update':1,'learner_deck':deck,'capability':{'kind':'public_candidate_v1','bot_preset':'starter','human_public_custom':False}}
        model.with_suffix('.json').write_text(json.dumps(manifest))
        teams=list(itertools.combinations(PUBLIC_CHARACTERS,4))
        report={'model':{'sha256':digest,'schema':SCHEMA,'rule_hash':identity},'preset':'starter','learner_build':deck,
                'learner_is_preset':False,'sampled_opponent':True,'by_opponent_team':{str(i):{} for i in range(15)},'pairs':32,
                'engine':{'escalation':True,'first_turn_draw':True},'games_index':[
                    {'pair':i,'position':position,'opponent_team':teams[i%15],'terminated':True,'winner':'a'}
                    for i in range(32) for position in ('first','second')]}
        path=root/'report.json';path.write_text(json.dumps(report))
        return model,path,deck

    def test_explicit_binding_preserves_training_identity(self):
        from app.modules.card_game.rl.public_export import approve_candidate
        from app.modules.card_game.rl.capability import serving_deck_hash
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel,serving_deck
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,report,deck=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'custom learner'):approve_candidate(model,report)
            evidence=approve_candidate(model,report,serving_build=deck)
            frozen=FrozenModel(root,'starter')
            self.assertEqual(evidence['serving_build_sha256'],serving_deck_hash(deck))
            self.assertEqual(frozen.manifest['learner_deck'],deck)
            self.assertEqual(serving_deck(frozen,'starter')['card_ids'],deck['card_ids'])
            copied=serving_deck(frozen,'starter');copied['card_ids'].clear()
            self.assertEqual(len(frozen.serving_deck['card_ids']),32)

    def test_report_and_fixed_roles_must_match(self):
        from app.modules.card_game.rl.public_export import approve_candidate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,report,deck=self.fixture(root)
            other=deepcopy(deck);other['card_ids']=['N05' if c=='N01' else c for c in other['card_ids']]
            with self.assertRaisesRegex(ValueError,'exact serving build'):approve_candidate(model,report,serving_build=other)
            other=deepcopy(deck);other['character_ids'].reverse()
            with self.assertRaisesRegex(ValueError,'characters and order'):approve_candidate(model,report,serving_build=other)

    def test_tampered_or_removed_serving_build_fails_and_cache_rechecks(self):
        from app.modules.card_game.rl.public_export import approve_candidate
        from app.modules.card_game.engine.ai import advanced_model
        from app.errors import RuleValidationError
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,report,deck=self.fixture(root)
            approve_candidate(model,report,serving_build=deck)
            manifest_path=model.with_suffix('.json');manifest=json.loads(manifest_path.read_text())
            with patch.object(advanced_model.config,'DUEL_AI_MODEL_DIR',str(root)):
                advanced_model.load_model('starter')
                changed=deepcopy(manifest);changed['serving_build']['card_ids']=['N05' if c=='N01' else c for c in deck['card_ids']]
                manifest_path.write_text(json.dumps(changed))
                with self.assertRaises(RuleValidationError):advanced_model.load_model('starter')
            changed=deepcopy(manifest);del changed['serving_build'];manifest_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'missing'):advanced_model.FrozenModel(root,'starter')
            advanced_model._CACHE.clear()

    def test_active_room_rejects_changed_build(self):
        from app.modules.card_game.rl.public_export import approve_candidate
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel,choose_action
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.errors import RuleValidationError
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,report,deck=self.fixture(root);approve_candidate(model,report,serving_build=deck)
            frozen=FrozenModel(root,'starter');state=new_game(seed=1,decks={'a':deck,'b':deck})
            state['ai_profile']={'deck_id':'starter','version':frozen.version,'build_sha256':'changed'}
            with patch('app.modules.card_game.engine.ai.advanced_model.load_model',return_value=frozen):
                with self.assertRaisesRegex(RuleValidationError,'构筑版本'):choose_action(state,'b')

    def test_runtime_revalidation_preserves_training_rule_identity(self):
        from app.modules.card_game.rl.public_export import approve_candidate
        from app.modules.card_game.rl.public_schema import rule_identity
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,report,deck=self.fixture(root)
            path=model.with_suffix('.json');manifest=json.loads(path.read_text())
            manifest['rule_identity']={'rule_hash':'historical-training-rules'}
            manifest['runtime_rule_hash']=rule_identity()
            path.write_text(json.dumps(manifest))
            old_report=json.loads(report.read_text())
            old_report['model']['rule_hash']='historical-training-rules'
            stale=root/'old-report.json';stale.write_text(json.dumps(old_report))
            with self.assertRaisesRegex(ValueError,'does not match'):
                approve_candidate(model,stale,serving_build=deck)
            approve_candidate(model,report,serving_build=deck)
            frozen=FrozenModel(root,'starter')
            self.assertEqual(frozen.manifest['rule_identity']['rule_hash'],'historical-training-rules')
            self.assertEqual(frozen.manifest['validation']['rule_hash'],rule_identity())
