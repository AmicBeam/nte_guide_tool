"""2026-10-09 user rule: collapsed espers cannot provide entry harmony."""
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import apply_action, new_game, observe
from app.modules.card_game.engine.duel_v2.combat import harmony_marks
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import card_instance, force_collapse, hero, move_out


class CollapseHarmonyTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=9, first_side='a', skip_mulligan=True, escalation=False)
        for team in s['sides'].values():
            team.update(hand=[], ap=10)
        s['sides']['a'].update(front='nanali', last_front='nanali')
        hero(s, 'a', 'nanali')['harmony'] = 3
        return s

    def test_front_and_benched_source_block_preview_projection_and_payment(self):
        for benched in (False, True):
            with self.subTest(benched=benched):
                s = self.game()
                if benched:
                    move_out(s, 'a', 'nanali')
                force_collapse(s, 'a', 'nanali', by='b')
                s = json.loads(json.dumps(s))
                opening = capture_public_board(s)
                seq = s['event_seq']
                self.assertEqual(harmony_marks(s, 'a'), dict(payer=None, ready={}, available=False))
                self.assertFalse(opening['sides']['a']['harmony_available'])
                view = observe(s, 'a')
                chars = view['sides']['a']['characters']
                self.assertTrue(all(not h['harmony_source'] and not h['harmony_ready'] for h in chars))
                action = dict(type='attack', character_id='zero')
                row = next(row for row in view['legal_actions'] if row['action'] == action)
                self.assertEqual(row['preview']['harmony'], '不触发')
                after = apply_action(s, 'a', action)
                events = [e for e in after['events'] if e['seq'] > seq]
                self.assertFalse(any(e['type'] == 'harmony' for e in events))
                self.assertEqual(hero(after, 'a', 'nanali')['harmony'], 3)
                self.assertFalse(after['sides']['a']['harmonized'].get('zero'))
                self.assertEqual(replay_public_board(opening, events), capture_public_board(after))

    def test_collapsed_departure_cannot_create_residue_after_expiry(self):
        s = self.game()
        s['sides']['a']['last_front'] = None
        force_collapse(s, 'a', 'nanali', by='b')
        move_out(s, 'a', 'nanali')
        self.assertIsNone(s['sides']['a']['last_front'])
        hero(s, 'a', 'nanali')['flags'].pop('collapse')
        self.assertFalse(harmony_marks(s, 'a')['available'])
        self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 3)

    def test_natural_expiry_reenables_unchanged_source(self):
        for benched in (False, True):
            with self.subTest(benched=benched):
                s = self.game()
                if benched:
                    move_out(s, 'a', 'nanali')
                force_collapse(s, 'a', 'nanali', by='b')
                self.assertFalse(harmony_marks(s, 'a')['available'])
                # Reach the applier's next turn end without entering another esper.
                begin_turn(s, 'b')
                s = apply_action(s, 'b', dict(type='end_turn'))
                self.assertTrue(harmony_marks(s, 'a')['available'])
                s = apply_action(s, 'a', dict(type='attack', character_id='zero'))
                self.assertEqual(hero(s, 'a', 'nanali')['harmony'], 0)

    def test_free_entry_harmony_cannot_borrow_collapsed_front(self):
        ids = ['anhunqu', 'canhong', 'haiyue', 'zero']
        deck = dict(id='test', name='test', character_ids=ids,
                    card_ids=[cid for owner in ids for cid, card in CARDS.items()
                              if card['character_id'] == owner and not card.get('derived')])
        for collapsed in (False, True):
            with self.subTest(collapsed=collapsed):
                s = new_game(seed=9, first_side='a', skip_mulligan=True, escalation=False,
                             decks={'a': deck, 'b': deck})
                for team in s['sides'].values():
                    team.update(hand=[], ap=10)
                s['sides']['a']['front'] = 'canhong'
                hero(s, 'a', 'canhong')['harmony'] = 1
                if collapsed:
                    force_collapse(s, 'a', 'canhong', by='b')
                card = card_instance(s, 'A06', 'a')
                s['sides']['a']['hand'] = [card]
                s = apply_action(s, 'a', dict(type='play_card', card_id=card['instance_id']))
                self.assertEqual(s['sides']['a']['harmonized'].get('anhunqu', False), not collapsed)
                self.assertEqual(hero(s, 'a', 'canhong')['harmony'], 1)


if __name__ == '__main__':
    unittest.main()
