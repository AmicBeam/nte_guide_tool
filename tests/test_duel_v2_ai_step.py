"""Advanced AI commits exactly one requested operation with normal receipts."""
import importlib
from types import SimpleNamespace
from unittest.mock import patch
from tests.test_solo_room_flow import RoomFlowTestCase


class AdvancedStepTest(RoomFlowTestCase):
    def setUp(self):
        super().setUp()
        self.service = importlib.import_module('app.modules.card_game.engine.application.v2_service')
        self.storage = importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        self.token = self._issue_login_and_get_token('step-test')

    def tearDown(self):
        self.storage.flush(5)
        super().tearDown()

    def test_each_ai_operation_is_visible_and_idempotent(self):
        from app.modules.card_game.engine.duel_v2 import choose_action
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        root = 'app.modules.card_game.engine.ai.advanced_model.'
        fixture=SimpleNamespace(version='step-fixture',schema='fixed_ten_v1',
            serving_deck=next(d for d in STARTER_DECKS if d['id']=='starter'),
            validate_human=lambda deck:{'kind':'test_fixture','human_public_custom':True})
        with patch(root+'load_model', return_value=fixture), patch(root+'choose_action',side_effect=choose_action) as planner:
            result=self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':'starter'},token=self.token)
            game=result['game'];code=result['room']['room_code']
            self.assertFalse(planner.called)
            result=self._post('/api/duel-v2/action',dict(room_code=code,expected_version=game['version'],request_id='human-opening',action={'type':'mulligan','card_ids':[]}),token=self.token)
            self.assertFalse(planner.called)
            self.assertTrue(result['room']['ai_pending'])
            version=result['game']['version']
            request=dict(room_code=code,expected_version=version,request_id='one-bot-operation')
            step=self._post('/api/duel-v2/ai-step',request,token=self.token)
            self.assertEqual(step['game']['version'],version+1)
            self.assertEqual(planner.call_count,1)
            again=self._post('/api/duel-v2/ai-step',request,token=self.token)
            self.assertTrue(again['replayed_request'])
            self.assertEqual(again['game']['version'],step['game']['version'])
            self.assertEqual(planner.call_count,1)
            stale=self.client.post('/api/duel-v2/ai-step',json={**request,'request_id':'stale-tab'},headers={'Authorization':'Bearer '+self.token})
            self.assertEqual(stale.status_code,409)
            self.assertEqual(planner.call_count,1)
            wrong_room=self.client.post('/api/duel-v2/ai-step',json={**request,'room_code':'WRONG','request_id':'old-room'},headers={'Authorization':'Bearer '+self.token})
            self.assertEqual(wrong_room.status_code,409)

    def test_player_cannot_forge_bot_operation_or_use_it_in_solo(self):
        result=self._post('/api/duel-v2/start',{'mode':'solo'},token=self.token)
        request=dict(room_code=result['room']['room_code'],expected_version=result['game']['version'],request_id='not-advanced')
        error=self.client.post('/api/duel-v2/ai-step',json=request,headers={'Authorization':'Bearer '+self.token})
        self.assertEqual(error.status_code,400)
        forged=self.client.post('/api/duel-v2/action',json={**request,'request_id':'forged','action':{'type':'_ai_step'}},headers={'Authorization':'Bearer '+self.token})
        self.assertEqual(forged.status_code,400)
