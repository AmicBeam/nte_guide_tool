"""Shared telemetry/public replay on physical records, no search-strength claim."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
from app.modules.card_game.rl import recovery_runtime
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl.batched_duel.recording import PhysicalGameRecord,state_digest


class PhysicalRecordTest(unittest.TestCase):
    def test_real_actions_shared_telemetry_replay_and_no_overwrite(self):
        state=new_game(seed=8133,skip_mulligan=True)
        opening=deepcopy(state);record=PhysicalGameRecord(state)
        for _ in range(3):
            side=acting_side(state);actions,_=recovery_runtime.decision(state,side)
            action=next((a for a in actions if a['type']=='attack'),actions[0])
            after=apply_action(state,side,action)
            record.append(state,after,side,action,actions,search_stats={'simulations':2})
            state=clean(after)
        side=acting_side(state);action={'type':'concede'}
        after=apply_action(state,side,action)
        record.append(state,after,side,action,[action]);state=clean(after)
        with tempfile.TemporaryDirectory() as directory:
            result=record.finish(state,{'id':'fixture'},directory,public_replay=True)
            self.assertTrue(result['complete']);self.assertTrue(result['replay_verified'])
            self.assertEqual(result['search_simulations'],6)
            with gzip.open(Path(directory)/'raw/fixture.json.gz','rt',encoding='utf-8') as f:raw=json.load(f)
            self.assertEqual(len(raw['actions']),4);self.assertIn('cards',raw['telemetry'])
            self.assertTrue((Path(directory)/'replays/fixture.json.gz').exists())
            with gzip.open(Path(directory)/'replays/fixture.json.gz','rt',encoding='utf-8') as f:public=json.load(f)
            def check_public(value):
                if isinstance(value,dict):
                    self.assertFalse(set(value)&{'deck','rng','seed','private_card'})
                    if isinstance(value.get('hand'),list):
                        for card in value['hand']:
                            if 'card_id' in card or 'instance_id' in card:
                                self.assertTrue(card.get('copy') or card.get('revealed') or card.get('derived'))
                    for child in value.values():check_public(child)
                elif isinstance(value,list):
                    for child in value:check_public(child)
            check_public(public)
            self.assertEqual(raw['actions'][-1]['state_sha256'],state_digest(state))
            with self.assertRaises(RuntimeError):record.finish(state,{'id':'fixture'},directory)
        self.assertEqual(opening['version'],0)

    def test_real_legal_play_opportunities_survive_recording_and_trigger_idle_gate(self):
        from app.modules.card_game.rl.batched_search.behavior import BehaviorGate,InactivePolicyError
        state=new_game(seed=8135,skip_mulligan=True);record=PhysicalGameRecord(state)
        for _ in range(6):
            side=acting_side(state);actions,_=recovery_runtime.decision(state,side)
            action=next(a for a in actions if a['type']=='end_turn')
            after=apply_action(state,side,action);record.append(state,after,side,action,actions);state=clean(after)
        side=acting_side(state);action={'type':'concede'};after=apply_action(state,side,action)
        record.append(state,after,side,action,[action]);state=clean(after)
        with tempfile.TemporaryDirectory() as directory:
            result=record.finish(state,{'id':'idle-fixture','policies':{'a':'test','b':'test'}},directory)
            self.assertTrue(result['complete'])
            self.assertGreaterEqual(result['behavior']['a']['legal_play_decisions'],2)
            self.assertGreaterEqual(result['behavior']['a']['legal_play_turns'],2)
            self.assertEqual(result['behavior']['a']['manual_play_card_actions'],0)
            with self.assertRaises(InactivePolicyError):BehaviorGate(stop_games=1).observe([result])

    def test_trace_mismatch_and_path_escape_are_rejected(self):
        state=new_game(seed=8134,skip_mulligan=True);record=PhysicalGameRecord(state)
        side=acting_side(state);actions,_=recovery_runtime.decision(state,side)
        after=apply_action(state,side,actions[0]);bad=deepcopy(state);bad['version']+=1
        with self.assertRaises(ValueError):record.append(bad,after,side,actions[0],actions)
        with tempfile.TemporaryDirectory() as output,self.assertRaises(ValueError):
            record.finish(state,{'id':'../escape'},output)


if __name__=='__main__':unittest.main()
