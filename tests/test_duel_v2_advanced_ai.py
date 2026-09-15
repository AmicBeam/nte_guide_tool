"""Advanced AI room lifecycle and CPU-only inference boundaries."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib
import importlib.util
import tempfile
import unittest

from tests.test_solo_room_flow import RoomFlowTestCase


class AdvancedRoomTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.service = importlib.import_module('app.modules.card_game.engine.application.v2_service')
        self.storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        self.repository = importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        self.token = self._issue_login_and_get_token('advanced-ai-test')
    def tearDown(self):
        self.storage.flush(5)
        super().tearDown()
    def test_advanced_room_persists_and_dispatches_model(self):
        from app.modules.card_game.engine.duel_v2 import choose_action
        module = 'app.modules.card_game.engine.ai.advanced_model.'
        with patch(module+'load_model', return_value=SimpleNamespace(version='test-version')), patch(module+'choose_action', side_effect=choose_action) as model:
            result = self._post('/api/duel-v2/start', {'mode':'advanced','ai_deck':'weave-rush'}, token=self.token)
            self.assertEqual(result['room']['mode'],'advanced')
            game = result['game']
            self.assertEqual(game['sides']['b']['name'],'高级人机')
            self.assertEqual({h['id'] for h in game['sides']['a']['characters']},{'bohe','baicang','zero','iloy'})
            self.assertEqual({h['id'] for h in game['sides']['b']['characters']},{'bohe','baicang','zero','iloy'})
            code = result['room']['room_code']
            action = next(e['action'] for e in game['legal_actions'] if e['action']['type']=='mulligan' and not e['action']['card_ids'])
            result = self._post('/api/duel-v2/action', {'room_code':code,'expected_version':game['version'],'request_id':'ai-mulligan','action':action}, token=self.token)
            game = result['game']
            payload = {'room_code':code,'expected_version':game['version'],'request_id':'ai-turn','action':{'type':'end_turn'}}
            result = self._post('/api/duel-v2/action', payload, token=self.token)
            self.assertTrue(model.called)
            again = self._post('/api/duel-v2/action', payload, token=self.token)
            self.assertEqual(result['game']['version'],again['game']['version'])
            state = self._get('/api/duel-v2/state',token=self.token)
            self.assertEqual(state['room']['mode'],'advanced')
            room = self.repository.room_by_code(code)
            profile = self.storage.load(room.id)['game']['ai_profile']
            self.assertEqual(profile,{'deck_id':'weave-rush','version':'test-version'})
            switched=self.client.post('/api/duel-v2/start',json={'mode':'advanced','ai_deck':'starter'},headers={'Authorization':'Bearer '+self.token})
            self.assertEqual(switched.status_code,409)
    @unittest.skipUnless(importlib.util.find_spec('numpy') and (Path(__file__).resolve().parents[1]/'app/modules/card_game/engine/ai/models/starter.npz').exists(), 'bundled serving models and NumPy required')
    def test_real_models_finish_both_presets_and_save_replays(self):
        from app.modules.card_game.engine.duel_v2 import choose_action
        for deck in ('starter','weave-rush'):
            result=self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':deck},token=self.token)
            code=result['room']['room_code']
            for index in range(180):
                game=result['game']
                if game['phase']=='finished':break
                room=self.repository.room_by_code(code)
                state=self.storage.load(room.id)['game']
                action=choose_action(state,'a')
                result=self._post('/api/duel-v2/action',{'room_code':code,'expected_version':game['version'],
                    'request_id':f'{deck}-{index}','action':action},token=self.token)
            self.assertEqual(result['game']['phase'],'finished')
            self.assertTrue(result['room']['has_replay'])
            self.assertEqual(result['game']['sides']['b']['name'],'高级人机')
            self._post('/api/duel-v2/leave',token=self.token)

    def test_missing_models_fail_before_room_creation(self):
        from app.errors import RuleValidationError
        with patch('app.modules.card_game.engine.ai.advanced_model.load_model', side_effect=RuleValidationError('模型暂不可用')):
            response=self.client.post('/api/duel-v2/start',json={'mode':'advanced'},headers={'Authorization':'Bearer '+self.token})
        self.assertEqual(response.status_code,400)
        player=self.dao_module.get_or_create_player('advanced-ai-test')[0]
        self.assertIsNone(self.repository.current_room(player))
    def test_unsupported_preset_is_rejected(self):
        response=self.client.post('/api/duel-v2/start',json={'mode':'advanced','ai_deck':'other'},headers={'Authorization':'Bearer '+self.token})
        self.assertEqual(response.status_code,400)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy inference dependency')
class CpuEncoderTest(unittest.TestCase):
    def test_bundled_models_are_complete_and_loadable(self):
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        directory=Path(__file__).resolve().parents[1]/'app/modules/card_game/engine/ai/models'
        self.assertEqual({p.name for p in directory.iterdir()},
                         {'starter.npz','starter.json','weave-rush.npz','weave-rush.json'})
        for deck in ('starter','weave-rush'):
            self.assertTrue(FrozenModel(directory,deck).version)

    def test_runtime_rule_compatibility_still_rejects_other_rules(self):
        import json
        from unittest.mock import patch
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        directory = Path(__file__).resolve().parents[1] / 'app/modules/card_game/engine/ai/models'
        for deck in ('starter', 'weave-rush'):
            manifest = json.loads((directory / f'{deck}.json').read_text())
            self.assertNotEqual(manifest['runtime_rule_hash'], manifest['rule_identity']['rule_hash'])
            with patch('app.modules.card_game.rl.rule_ir.completeness.compiled_rule_hash', return_value='unrelated'):
                with self.assertRaisesRegex(ValueError, 'rule version'):
                    FrozenModel(directory, deck)

    def test_hidden_cards_do_not_change_encoding(self):
        import numpy as np
        from app.modules.card_game.engine.duel_v2 import new_game,legal_actions,acting_side
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        from app.modules.card_game.engine.ai.advanced_model import encode
        state=new_game(seed=4,skip_mulligan=True,decks={'a':STARTER_DECK,'b':STARTER_DECK})
        side=acting_side(state);foe='b' if side=='a' else 'a'
        actions=[a for a in legal_actions(state,side) if a['type']!='concede']
        before=encode(state,side,actions)
        changed=deepcopy(state)
        changed['sides'][foe]['hand'].reverse()
        for card in changed['sides'][foe]['hand']:card['card_id']='N04'
        for team in changed['sides'].values():team['deck'].reverse()
        changed['rng']=999
        after=encode(changed,side,actions)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    def test_numeric_export_rejects_bad_checksum(self):
        import json
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,CARD_IDS,SEATS
        from app.modules.card_game.rl.rule_ir.completeness import compiled_rule_hash
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'starter.npz').write_bytes(b'invalid')
            (root/'starter.json').write_text(json.dumps(dict(schema='resident_public_v1',deck='starter',gpu_lock=GPU_LOCK,
                                                             card_ids=list(CARD_IDS),seat_ids=list(SEATS),sha256='bad',rule_identity={'rule_hash':compiled_rule_hash('starter')})))
            with self.assertRaisesRegex(ValueError,'checksum'):FrozenModel(root,'starter')
