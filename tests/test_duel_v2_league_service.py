"""Exercise v5 through a real temporary API room with synthetic numeric weights."""
from pathlib import Path
import importlib,json,tempfile
from unittest.mock import patch
from tests.test_solo_room_flow import RoomFlowTestCase
from tests.test_duel_v2_league import zero_policy

class LeagueRoomTest(RoomFlowTestCase):
    def test_eight_character_numeric_model_roundtrip(self):
        storage=importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        token=self._issue_login_and_get_token('league-room-test')
        try:
            with tempfile.TemporaryDirectory() as d:
                zero_policy(d,'quick-rush');p=Path(d)/'quick-rush.json';m=json.loads(p.read_text())
                # Synthetic acceptance metadata is confined to this isolated test.
                m['validation']={'approved':True,'complete':True,'model_sha256':m['sha256'],'rule_hash':m['rule_hash'],'build_sha256':m['build_sha256']}
                p.write_text(json.dumps(m),encoding='utf-8')
                with patch('app.config.DUEL_AI_MODEL_DIR',d):
                    room=self._post('/api/duel-v2/start',{'mode':'advanced','ai_deck':'quick-rush','ai_player_deck':'mirror'},token=token)
                    game=room['game'];self.assertEqual([h['id'] for h in game['sides']['b']['characters']],['xiaozhi','zero','bohe','haiyue'])
                    action=next(e['action'] for e in game['legal_actions'] if e['action']['type']=='mulligan' and not e['action']['card_ids'])
                    result=self._post('/api/duel-v2/action',{'room_code':room['room']['room_code'],'expected_version':game['version'],'request_id':'league-opening','action':action},token=token)
                    self.assertEqual(result['game']['phase'],'playing');self.assertEqual(result['game']['active_side'],'a')
        finally:storage.flush(5)
