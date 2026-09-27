import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl import covered_value as cv


class CoveredValueTest(unittest.TestCase):
    def test_manual_pass_flag_cannot_replace_raw_phase_endpoints(self):
        from app.modules.card_game.rl.cross_schedule import ROSTER
        from app.modules.card_game.rl.cross_lineup import identity
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);files={}
            for phase in ('training','selection','heldout'):
                p=folder/(phase+'.json');p.write_text('{}');files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            (folder/'scope.json').write_text(json.dumps(dict(schema='covered_expert_value_v1',passed=True,
                rule_hash=identity(),algorithm_sha256=cv.identity(),objective='visible_rule_after_six_exploration_actions',
                models={k:'invented' for k in ROSTER},files=files)))
            with self.assertRaisesRegex(ValueError,'endpoints incomplete'):cv.load_scope(folder)

    def test_expert_value_leaf_keeps_viewer_and_deadline_never_supplies_value(self):
        state={'phase':'playing','active_side':'b'};seen=[]
        class Runtime:
            def value_for(self,state,viewer,values):seen.append(viewer);return -.3
        with patch('app.modules.card_game.rl.covered_rollout.rule_decision',return_value=([{}],0)),\
             patch('app.modules.card_game.engine.duel_v2.simulation.simulate_action',return_value=state):
            result=cv.expert_leaf(state,'a',{},Runtime(),float('inf'))
        self.assertEqual(result,(-.3,True));self.assertEqual(seen,['a'])
        self.assertEqual(cv.expert_leaf(state,'a',{},Runtime(),0),(None,False))

    def test_auxiliary_rows_only_after_opening_and_have_true_observer_terminal_labels(self):
        state={'phase':'playing','active_side':'a','turn':0}
        def act(state,actor,action):
            result=dict(state,turn=state['turn']+1)
            if result['turn']>=5:result['active_side']='b'
            if result['turn']>=8:result.update(phase='finished',winner='a')
            return result
        with patch('app.modules.card_game.engine.duel_v2.new_game',return_value=state),\
             patch('app.modules.card_game.rl.covered_rollout.rule_decision',return_value=([{}],0)),\
             patch('app.modules.card_game.engine.duel_v2.observe',return_value={}),\
             patch('app.modules.card_game.rl.cross_lineup.encode',return_value=(np.zeros(2),[])),\
             patch('app.modules.card_game.engine.duel_v2.simulation.simulate_action',side_effect=act):
            game=cv.collect_game(dict(seed=1,first='a',decks={},keys={'a':'starter','b':'murk'},deadline=float('inf')))
        self.assertTrue(game['complete']);self.assertEqual(len(game['value_rows']),4)
        for row in game['value_rows']:
            self.assertGreaterEqual(row['step'],6)
            self.assertEqual(row['z'],1 if row['side']=='a' else -1)
            self.assertEqual(row['value_actor'],int(row['side']=='b'))


if __name__=='__main__':unittest.main()
