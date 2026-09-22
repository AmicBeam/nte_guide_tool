import gzip
import json
from pathlib import Path
import tempfile
import unittest
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side,legal_actions
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl.zhenhong_tactical_audit import audit_file


class TacticalAuditTest(unittest.TestCase):
    def test_replay_verifies_actual_outcome_without_inventing_startup(self):
        b=next(b for b in STARTER_DECKS if b['id']=='zhenhong');decks={'a':b,'b':b}
        state=new_game(seed=8291,first_side='a',decks=decks);actions=[]
        for _ in range(500):
            if state['phase']=='finished':break
            side=acting_side(state);legal=[a for a in legal_actions(state,side) if a['type']!='concede']
            action=next((a for a in legal if a['type']=='end_turn'),legal[0])
            actions.append(dict(side=side,action=action));state=clean(apply_action(state,side,action))
        self.assertEqual(state['phase'],'finished')
        data=dict(id='audit-test',seed=8291,first='a',decks=decks,actions=actions,complete=True,
                  winner=state['winner'],passive_triggers={'a':0,'b':0})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.json.gz'
            def save():
                with gzip.open(path,'wt') as f:json.dump(data,f)
            save();result=audit_file(path,counterfactuals=True)
            self.assertTrue(result['replayed'])
            for counts in result['counts'].values():self.assertEqual(counts['starts_after_ready'],0)
            data['winner']='b' if state['winner']=='a' else 'a';save()
            with self.assertRaises(ValueError):audit_file(path)
