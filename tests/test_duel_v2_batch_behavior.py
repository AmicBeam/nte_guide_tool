"""Behavior qualification never modifies a score, action, model, or reward."""
from copy import deepcopy
import unittest
from app.modules.card_game.rl.batched_search.behavior import BehaviorGate,InactivePolicyError


def record(gid,attack=0,complete=True):
    side=dict(legal_play_decisions=5,legal_play_turns=4,manual_play_card_actions=0,ordinary_attack_actions=attack,ultimate_actions=0)
    return dict(id=gid,complete=complete,replay_verified=True,error=None,policies={'a':'model','b':'model'},behavior={'a':side,'b':dict(side,manual_play_card_actions=3)})


class BehaviorTest(unittest.TestCase):
    def test_full_game_idle_failure_is_distinct_from_physical_capacity(self):
        gate=BehaviorGate();rows=[record(str(i)) for i in range(8)];before=deepcopy(rows)
        with self.assertRaises(InactivePolicyError):gate.observe(rows)
        self.assertEqual(rows,before);self.assertEqual(gate.report()['inactive_games'],8)
        self.assertFalse(gate.report()['behavior_approved']);self.assertTrue(gate.report()['p0_stop'])
        # Final partial evidence can still be persisted after the first P0 signal.
        gate.observe([record('partial',complete=False)]);self.assertEqual(gate.report()['verified_normal_games'],8)

    def test_both_idle_sides_are_one_game_not_two(self):
        gate=BehaviorGate();rows=[]
        for i in range(4):
            row=record(str(i));row['behavior']['b']=dict(row['behavior']['a']);rows.append(row)
        gate.observe(rows);self.assertEqual(gate.report()['inactive_sides'],8)
        self.assertEqual(gate.report()['inactive_games'],4);self.assertFalse(gate.report()['p0_stop'])

    def test_real_attacks_without_cards_and_one_turn_losses_are_not_idle(self):
        gate=BehaviorGate();rows=[record('attacker',attack=2),record('short')]
        rows[1]['behavior']['a']['legal_play_turns']=1
        gate.observe(rows);self.assertTrue(gate.report()['behavior_approved'])

    def test_missing_or_duplicate_evidence_is_rejected(self):
        gate=BehaviorGate();row=record('one',attack=1);gate.observe([row])
        with self.assertRaises(ValueError):gate.observe([row])
        missing=record('missing');missing.pop('behavior')
        with self.assertRaises(ValueError):BehaviorGate().observe([missing])


if __name__=='__main__':unittest.main()
