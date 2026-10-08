from tests.duel_v2_test_decks import test_character_deck
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
        from app.modules.card_game.rl.capability import capability_ready_manifest
        with patch(module+'load_model', return_value=SimpleNamespace(version='test-version', capability=capability_ready_manifest('weave-rush'))), patch(module+'choose_action', side_effect=choose_action) as model:
            result = self._post('/api/duel-v2/start', {'mode':'advanced','ai_deck':'weave-rush'}, token=self.token)
            self.assertEqual(result['room']['mode'],'advanced')
            game = result['game']
            self.assertEqual(game['sides']['b']['name'],'高级人机')
            catalog=self._get('/api/duel-v2/catalog', token=self.token)
            human=set(catalog['saved_build']['character_ids'])
            self.assertEqual({h['id'] for h in game['sides']['a']['characters']}, human)
            self.assertEqual({h['id'] for h in game['sides']['b']['characters']},{'bohe','baicang','zero','iloy'})
            self.assertEqual(catalog['saved_build']['character_ids'], self._get('/api/duel-v2/catalog', token=self.token)['saved_build']['character_ids'])
            code = result['room']['room_code']
            action = next(e['action'] for e in game['legal_actions'] if e['action']['type']=='mulligan' and not e['action']['card_ids'])
            result = self._post('/api/duel-v2/action', {'room_code':code,'expected_version':game['version'],'request_id':'ai-mulligan','action':action}, token=self.token)
            for step in range(40):
                if not result['room']['ai_pending']:break
                result=self._post('/api/duel-v2/ai-step',{'room_code':code,'expected_version':result['game']['version'],'request_id':f'opening-step-{step}'},token=self.token)
            game = result['game']
            payload = {'room_code':code,'expected_version':game['version'],'request_id':'ai-turn','action':{'type':'end_turn'}}
            result = self._post('/api/duel-v2/action', payload, token=self.token)
            self.assertTrue(model.called)
            again = self._post('/api/duel-v2/action', payload, token=self.token)
            self.assertEqual(result['game']['version'],again['game']['version'])
            if result['room']['ai_pending']:
                previous=result['game']['version']
                result=self._post('/api/duel-v2/ai-step',{'room_code':code,'expected_version':previous,'request_id':'explicit-ai-step'},token=self.token)
                self.assertEqual(result['game']['version'],previous+1)
            state = self._get('/api/duel-v2/state',token=self.token)
            self.assertEqual(state['room']['mode'],'advanced')
            room = self.repository.room_by_code(code)
            profile = self.storage.load(room.id)['game']['ai_profile']
            self.assertEqual(profile['deck_id'],'weave-rush')
            self.assertEqual(profile['version'],'test-version')
            self.assertIn('capability', profile)
            switched=self.client.post('/api/duel-v2/start',json={'mode':'advanced','ai_deck':'starter'},headers={'Authorization':'Bearer '+self.token})
            self.assertEqual(switched.status_code,409)
    @unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
    def test_joint_opening_policy_is_dispatched_and_opponent_swap_ids_stay_private(self):
        # Synthetic scorer tests API plumbing, not trained strength.
        from types import MethodType
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA
        from app.modules.card_game.rl.capability import capability_ready_manifest,preset_opponent_deck,serving_deck_hash
        deck=preset_opponent_deck('starter')
        model=SimpleNamespace(version='opening-fixture',schema=OPENING_SCHEMA,capability=capability_ready_manifest('starter'),
            serving_deck=deck,serving_build_sha256=serving_deck_hash(deck),scores=lambda state,candidates:candidates[:,11:].sum(1))
        model.select_public_action=MethodType(FrozenModel.select_public_action,model)
        with patch('app.modules.card_game.engine.ai.advanced_model.load_model',return_value=model):
            result=self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':'starter'},token=self.token)
            code=result['room']['room_code'];game=result['game']
            result=self._post('/api/duel-v2/action',{'room_code':code,'expected_version':game['version'],
                'request_id':'joint-opening','action':{'type':'mulligan','card_ids':[]}},token=self.token)
            result=self._post('/api/duel-v2/ai-step',{'room_code':code,'expected_version':result['game']['version'],'request_id':'joint-ai-opening'},token=self.token)
            game=result['game']
            events=[e for e in game['presentation']['events'] if e.get('type')=='mulligan' and e.get('side')=='b']
            self.assertEqual(len(events),1)
            self.assertEqual(events[0]['amount'],3)
            self.assertNotIn('card_ids',events[0])
            room=self.repository.room_by_code(code)
            stored=self.storage.load(room.id)['game']
            raw=next(e for e in stored['events'] if e.get('type')=='mulligan' and e.get('side')=='b')
            self.assertEqual(len(raw['card_ids']),3)

    @unittest.skipUnless(importlib.util.find_spec('numpy') and (Path(__file__).resolve().parents[1]/'app/modules/card_game/engine/ai/models/starter.npz').exists(), 'bundled serving models and NumPy required')
    @patch('app.modules.card_game.engine.ai.advanced_search.choose',side_effect=lambda state,side,model,fallback:fallback)
    def test_real_models_finish_all_presets_and_save_replays(self, _planner):
        from app.modules.card_game.engine.duel_v2 import choose_action
        for deck in ('starter','weave-rush','quick-rush'):
            result=self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':deck},token=self.token)
            code=result['room']['room_code']
            for index in range(800):
                game=result['game']
                if game['phase']=='finished':break
                if result['room']['ai_pending']:
                    result=self._post('/api/duel-v2/ai-step',{'room_code':code,'expected_version':game['version'],'request_id':f'{deck}-ai-{index}'},token=self.token)
                    continue
                room=self.repository.room_by_code(code)
                state=self.storage.load(room.id)['game']
                action=choose_action(state,'a')
                result=self._post('/api/duel-v2/action',{'room_code':code,'expected_version':game['version'],
                    'request_id':f'{deck}-{index}','action':action},token=self.token)
            self.assertEqual(result['game']['phase'],'finished')
            self.assertTrue(result['room']['has_replay'])
            self.assertEqual(result['game']['sides']['b']['name'],'实验性高级人机')
            self._post('/api/duel-v2/leave',token=self.token)

    def test_evaluated_bot_allocation_and_mirror_match_without_changing_saved_build(self):
        from collections import Counter
        from app.modules.card_game.rl.capability import capability_ready_manifest,preset_opponent_deck,serving_deck_hash
        from app.modules.card_game.engine.duel_v2 import choose_action
        original=preset_opponent_deck('starter')
        self._post('/api/duel-v2/build',original,token=self.token)
        custom=deepcopy(original);custom['card_ids']=['N01' if c=='N02' else c for c in custom['card_ids']]
        module='app.modules.card_game.engine.ai.advanced_model.'
        model=SimpleNamespace(version='paired-version',capability=capability_ready_manifest('starter'),serving_deck=custom)
        with patch(module+'load_model',return_value=model) as loader,patch(module+'choose_action',side_effect=choose_action):
            self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':'starter','ai_player_deck':'mirror'},token=self.token)
            loader.assert_called_once_with('starter')
            player=self.dao_module.get_or_create_player('advanced-ai-test')[0]
            room=self.repository.current_room(player)
            state=self.storage.load(room.id)['game']
            for side in ('a','b'):
                cards=[c['card_id'] for zone in ('hand','deck','discard') for c in state['sides'][side][zone]]
                self.assertEqual(Counter(cards),Counter(custom['card_ids']))
            self.assertEqual(state['ai_profile']['build_sha256'],serving_deck_hash(custom))
            saved=self._get('/api/duel-v2/catalog',token=self.token)['saved_build']
            self.assertEqual(saved['card_ids'],original['card_ids'])

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required for serving')
    def test_published_models_support_public_saved_builds(self):
        from app.modules.card_game.rl.capability import sample_public_deck
        from app.modules.card_game.engine.ai.advanced_model import load_model
        custom=sample_public_deck(9016,character_ids=['nanali','bohe','baicang','jiuyuan'])
        self._post('/api/duel-v2/build',custom,token=self.token)
        for key in ('starter','weave-rush'):
            self.assertTrue(load_model(key).capability['human_public_custom'])
            started=self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':key,'ai_player_deck':'saved'},token=self.token)
            room=self.repository.room_by_code(started['room']['room_code'])
            state=self.storage.load(room.id)['game']
            self.assertEqual(list(state['sides']['a']['characters']),custom['character_ids'])
            action=next(item['action'] for item in started['game']['legal_actions'] if item['action']['type']=='mulligan' and not item['action']['card_ids'])
            self._post('/api/duel-v2/action',{'room_code':room.room_code,'expected_version':started['game']['version'],
                'request_id':key+'-public-mulligan','action':action},token=self.token)
            self.assertEqual(self._get('/api/duel-v2/catalog',token=self.token)['saved_build']['card_ids'],custom['card_ids'])
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

    def test_removed_midrange_rejected_without_room_or_saved_build_change(self):
        before = self._get('/api/duel-v2/catalog', token=self.token)['saved_build']
        response=self.client.post('/api/duel-v2/start',json={'mode':'advanced','ai_deck':'midrange'},headers={'Authorization':'Bearer '+self.token})
        self.assertEqual(response.status_code,400)
        player=self.dao_module.get_or_create_player('advanced-ai-test')[0]
        self.assertIsNone(self.repository.current_room(player))
        self.assertEqual(before,self._get('/api/duel-v2/catalog',token=self.token)['saved_build'])

    def test_legacy_rejects_saved_custom_before_room_creation(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        custom = deepcopy(STARTER_DECK)
        custom['card_ids'] = ['N01' if c == 'N02' else c for c in custom['card_ids']]
        self._post('/api/duel-v2/build', custom, token=self.token)
        with patch('app.modules.card_game.engine.ai.advanced_model.load_model',
                   return_value=SimpleNamespace(version='legacy')):
            result = self._post_error('/api/duel-v2/start',
                {'mode':'advanced', 'ai_deck':'starter', 'ai_player_deck':'saved'}, token=self.token)
        self.assertIn('尚未开放', result['error'])
        player = self.dao_module.get_or_create_player('advanced-ai-test')[0]
        self.assertIsNone(self.repository.current_room(player))
        self.assertEqual(self._get('/api/duel-v2/catalog', token=self.token)['saved_build']['card_ids'], custom['card_ids'])

    def test_invalid_player_deck_selection_is_rejected(self):
        with patch('app.modules.card_game.engine.ai.advanced_model.load_model',
                   return_value=SimpleNamespace(version='legacy')):
            result = self._post_error('/api/duel-v2/start',
                {'mode':'advanced', 'ai_player_deck':'unknown'}, token=self.token)
        self.assertIn('我方构筑', result['error'])

    def test_advanced_uses_saved_public_deck_not_bot_preset(self):
        from app.modules.card_game.rl.capability import capability_ready_manifest
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        custom = deepcopy(content.STARTER_DECK)
        custom['id'] = 'deck-public-custom'
        custom['name'] = '公开自组'
        custom['card_ids'] = [card_id if card_id != 'N02' else 'N01' for card_id in custom['card_ids']]
        saved = self._post('/api/duel-v2/build', custom, token=self.token)
        self.assertEqual(saved['saved_build']['id'], 'deck-public-custom')
        module = 'app.modules.card_game.engine.ai.advanced_model.'
        with patch(module+'load_model', return_value=SimpleNamespace(version='test-version', capability=capability_ready_manifest('weave-rush'))):
            result = self._post('/api/duel-v2/start', {'mode':'advanced','ai_deck':'weave-rush'}, token=self.token)
        self.assertEqual({h['id'] for h in result['game']['sides']['a']['characters']}, set(custom['character_ids']))
        self.assertEqual({h['id'] for h in result['game']['sides']['b']['characters']}, {'bohe','baicang','zero','iloy'})
        catalog = self._get('/api/duel-v2/catalog', token=self.token)
        self.assertEqual(catalog['saved_build']['id'], 'deck-public-custom')
        self.assertEqual(catalog['saved_build']['card_ids'], custom['card_ids'])

    def test_privileged_account_cannot_use_test_characters_in_advanced(self):
        token = self._issue_login_and_get_token('advanced-privileged')
        models = importlib.import_module('app.models')
        models.Player.update(shaft_test_whitelisted=True).where(
            models.Player.player_uid == 'advanced-privileged'
        ).execute()
        content = importlib.import_module('app.modules.card_game.content.duel_v2')
        requiem = test_character_deck('requiem')
        self._post('/api/duel-v2/build', requiem, token=token)
        module = 'app.modules.card_game.engine.ai.advanced_model.'
        with patch(module+'load_model', return_value=SimpleNamespace(version='test-version', capability={'kind':'legacy_exact_mirror'})):
            error = self._post_error('/api/duel-v2/start', {'mode':'advanced','ai_deck':'starter','ai_player_deck':'saved'}, token=token)
        self.assertIn('公开', error['error'])
        catalog = self._get('/api/duel-v2/catalog', token=token)
        self.assertTrue(any(item['name'] == '浊燃预组' for item in catalog['saved_builds']))

    def test_refresh_keeps_asymmetric_advanced_room(self):
        module = 'app.modules.card_game.engine.ai.advanced_model.'
        with patch(module+'load_model', return_value=SimpleNamespace(version='test-version', capability={'kind':'legacy_exact_mirror'})):
            started = self._post('/api/duel-v2/start', {'mode':'advanced','ai_deck':'starter','ai_player_deck':'saved'}, token=self.token)
            refreshed = self._get('/api/duel-v2/state', token=self.token)
        self.assertEqual(started['room']['room_code'], refreshed['room']['room_code'])
        self.assertEqual(started['game']['sides']['a']['characters'][0]['id'], refreshed['game']['sides']['a']['characters'][0]['id'])
        self.assertEqual({h['id'] for h in refreshed['game']['sides']['b']['characters']}, {'nanali','iloy','zero','jiuyuan'})



@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy inference dependency')
class CpuEncoderTest(unittest.TestCase):
    def _legacy_fixture(self, root):
        import hashlib,json,numpy as np
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,CARD_IDS,SEATS,N_SEATS
        from app.modules.card_game.rl.gpu_duel.resident_schema import SIDE_FIELDS,CHAR_FIELDS
        from app.modules.card_game.rl.rule_ir.completeness import compiled_rule_hash
        width=2+2*len(SIDE_FIELDS)+2*N_SEATS*len(CHAR_FIELDS)+3*len(CARD_IDS)
        shapes={'state_net.0.weight':(1,width),'state_net.0.bias':(1,),'state_net.2.weight':(1,1),'state_net.2.bias':(1,),
                'cand_net.0.weight':(1,10),'cand_net.0.bias':(1,),'cand_net.2.weight':(1,1),'cand_net.2.bias':(1,),
                'score.weight':(1,1),'score.bias':(1,)}
        np.savez_compressed(root/'starter.npz',**{key:np.zeros(shape,np.float32) for key,shape in shapes.items()})
        manifest={'schema':'resident_public_v1','deck':'starter','gpu_lock':GPU_LOCK,'card_ids':list(CARD_IDS),
                  'seat_ids':list(SEATS),'sha256':hashlib.sha256((root/'starter.npz').read_bytes()).hexdigest(),
                  'rule_identity':{'rule_hash':'historical-fixture'},'runtime_rule_hash':compiled_rule_hash('starter'),
                  'hidden':1,'state_dim':width,'cand_dim':10}
        (root/'starter.json').write_text(json.dumps(manifest),encoding='utf-8')
        return manifest

    def test_bundled_models_are_complete_and_loadable(self):
        from app.modules.card_game.engine.ai.advanced_model import load_model
        from app.modules.card_game.engine.ai.recovery_model import SCHEMAS, RELEASE
        directory=Path(__file__).resolve().parents[1]/'app/modules/card_game/engine/ai/models'/RELEASE
        keys=('starter','weave-rush','quick-rush','zhenhong','murk')
        self.assertEqual({p.name for p in directory.iterdir()},
                         {f'{key}.{ext}' for key in keys for ext in ('json','npz')} | {'serving.json', '.gitattributes'})
        for key in keys:
            model=load_model(key)
            self.assertIn(model.schema,SCHEMAS)
            self.assertTrue(model.version)
            self.assertTrue(model.serving_authorized)

    def test_runtime_rule_compatibility_still_rejects_other_rules(self):
        from app.modules.card_game.engine.ai.fixed_model import FixedServingModel
        directory=Path(__file__).resolve().parents[1]/'app/modules/card_game/engine/ai/models'
        for key in ('starter','weave-rush','quick-rush'):
            with patch('app.modules.card_game.rl.fixed_lineup.identity', return_value='unrelated'):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    FixedServingModel(directory,key)

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
    def test_unknown_public_card_encoding_fails_closed(self):
        from app.modules.card_game.engine.duel_v2 import new_game,legal_actions,acting_side
        from app.modules.card_game.content.duel_v2 import STARTER_DECK
        from app.modules.card_game.engine.ai.advanced_model import encode
        state=new_game(seed=4,skip_mulligan=True,decks={'a':STARTER_DECK,'b':STARTER_DECK})
        side=acting_side(state)
        actions=[a for a in legal_actions(state,side) if a['type']!='concede']
        state['sides'][side]['hand'][0]['card_id']='X99'
        with self.assertRaisesRegex(ValueError,'cannot represent'):
            encode(state,side,actions)

    def test_legacy_manifest_without_capability_still_loads(self):
        import json
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        from app.modules.card_game.rl.capability import LEGACY_MIRROR_KIND
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=self._legacy_fixture(root)
            model=FrozenModel(root,'starter')
            self.assertEqual(model.capability['kind'], LEGACY_MIRROR_KIND)
            self.assertFalse(model.capability['human_public_custom'])
            self.assertNotIn('capability',manifest)

    def test_capability_gate_rejects_claiming_public_custom_on_legacy_weights(self):
        import json
        from app.modules.card_game.engine.ai.advanced_model import FrozenModel
        from app.modules.card_game.rl.gpu_duel.catalog import GPU_LOCK,CARD_IDS,SEATS
        from app.modules.card_game.rl.rule_ir.completeness import compiled_rule_hash
        directory=Path(__file__).resolve().parents[1]/'app/modules/card_game/engine/ai/models'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest=self._legacy_fixture(root)
            manifest['capability']={'kind':'legacy_exact_mirror','human_public_custom':True}
            (root/'starter.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'public custom'):
                FrozenModel(root,'starter')


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
