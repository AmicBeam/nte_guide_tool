"""User-specified additive shields, with existing damage and expiry boundaries."""
import unittest

from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero, move_out, shield


class ShieldStackingTest(unittest.TestCase):
    def game(self):
        s = new_game(seed=52, first_side='a', skip_mulligan=True, escalation=False)
        for team in s['sides'].values():
            team.update(hand=[], ap=20, shield=0)
        return s

    def test_multiple_grants_add_to_remaining_shield(self):
        s = self.game(); h = hero(s, 'a', 'zero'); hp = h['hp']
        shield(s, 'a', 'zero', 3)
        damage(s, 'a', 'zero', 1)
        shield(s, 'a', 'zero', 1)
        shield(s, 'a', 'zero', 3)
        self.assertEqual(h['shield'], 6)
        damage(s, 'a', 'zero', 7)
        self.assertEqual((h['shield'], h['hp']), (0, hp - 1))

    def test_consecutive_battle_cards_stack_and_expire_on_own_turn(self):
        s = self.game()
        for _ in range(2):
            card = card_instance(s, 'Z03', 'a')
            s['sides']['a']['hand'].append(card)
            s = apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        self.assertEqual(hero(s, 'a', 'zero')['shield'], 4)
        view = observe(s, 'a', include_previews=False)
        self.assertEqual(next(h for h in view['sides']['a']['characters'] if h['id'] == 'zero')['shield'], 4)
        grants = [e for e in s['events'] if e['type'] == 'shield' and e.get('target') == 'a:zero']
        self.assertEqual(grants[-1]['before']['shield'], 2)
        self.assertEqual(grants[-1]['after']['shield'], 4)
        s = apply_action(s, 'a', {'type': 'end_turn'})
        self.assertEqual(hero(s, 'a', 'zero')['shield'], 4)
        s = apply_action(s, 'b', {'type': 'end_turn'})
        self.assertEqual(hero(s, 'a', 'zero')['shield'], 0)

    def test_return_and_knockdown_clear_accumulated_shield(self):
        for knocked_down in (False, True):
            with self.subTest(knocked_down=knocked_down):
                s = self.game(); s['sides']['a']['front'] = 'zero'
                shield(s, 'a', 'zero', 2); shield(s, 'a', 'zero', 3)
                if knocked_down:
                    damage(s, 'a', 'zero', 99)
                else:
                    move_out(s, 'a', 'zero')
                self.assertEqual(hero(s, 'a', 'zero')['shield'], 0)
