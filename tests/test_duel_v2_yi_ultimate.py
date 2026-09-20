"""Yi's user-specified tabletop ultimate adds one turn to current beast fangs."""
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.state import hero


class YiUltimateTest(unittest.TestCase):
    def game(self, fangs):
        team = ['yi', 'zero', 'jiuyuan', 'nanali']
        deck = dict(id='yi-test', name='终结测试', character_ids=team,
                    card_ids=[key for cid in team for key, card in CARDS.items()
                              if card['character_id'] == cid and not card.get('derived')])
        s = new_game(seed=61, first_side='a', skip_mulligan=True,
                     decks={'a': deck, 'b': deck}, escalation=False)
        for side in s['sides'].values():
            side.update(hand=[], shield=0)
        hero(s, 'a', 'yi').update(energy=5, beast_fangs=fangs)
        return s

    def test_ultimate_adds_one_instead_of_refreshing_minimum(self):
        # The +1 is a tabletop rule supplied by the user, not an Everness value.
        for before in (0, 1, 4, 6, 9):
            with self.subTest(before=before):
                s = apply_action(self.game(before), 'a', {'type': 'ultimate', 'character_id': 'yi'})
                h = hero(s, 'a', 'yi')
                self.assertEqual((h['beast_fangs'], h['ultimate_turns'], h['energy']), (before + 1, 2, 0))
                s = json.loads(json.dumps(s))
                view = observe(s, 'a', include_previews=False)
                shown = next(h for h in view['sides']['a']['characters'] if h['id'] == 'yi')
                self.assertEqual(shown['beast_fangs'], before + 1)

    def test_countdown_and_later_ultimate_addition_are_independent(self):
        s = apply_action(self.game(4), 'a', {'type': 'ultimate', 'character_id': 'yi'})
        begin_turn(s, 'b')
        self.assertEqual(hero(s, 'a', 'yi')['beast_fangs'], 5)
        begin_turn(s, 'a')
        self.assertEqual(hero(s, 'a', 'yi')['beast_fangs'], 4)
        self.assertEqual(hero(s, 'a', 'yi')['ultimate_turns'], 1)
        hero(s, 'a', 'yi')['energy'] = 5
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'yi'})
        self.assertEqual(hero(s, 'a', 'yi')['beast_fangs'], 5)
        self.assertEqual(hero(s, 'a', 'yi')['ultimate_turns'], 2)
