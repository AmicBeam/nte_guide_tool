"""User-authored K06: an ordinary one-AP play plus a mandatory two-AP surcharge."""
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, get_catalog
from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.modifiers import add_effect
from app.modules.card_game.engine.duel_v2.state import hero
from tests.test_duel_v2_lingke import game, give, play


class AdditionalApCostTest(unittest.TestCase):
    def prepare(self, ap, *, instant=False, ordinary_zero=False, spent=False):
        state = game()
        state['sides']['a']['ap'] = ap
        state['sides']['a']['used']['instant'] = spent
        owner = hero(state, 'a', 'lingke')
        if instant:
            add_effect(owner, 'test.instant', 'card.instant', True, op='allow', clear_on=())
        if ordinary_zero:
            add_effect(owner, 'test.cost', 'card.cost', 0, op='set', clear_on=())
        action = give(state, 'a', 'K06')
        return state, action

    def test_catalog_separates_ordinary_and_extra_costs(self):
        self.assertEqual((CARDS['K06']['cost'], CARDS['K06']['additional_ap_cost']), (1, 2))
        shown = next(card for card in get_catalog()['cards'] if card['id'] == 'K06')
        self.assertEqual(shown['additional_ap_cost'], 2)
        self.assertTrue(shown['description'].startswith('额外消耗 2 点行动力。'))

    def test_legality_preview_and_actual_payment_agree_at_boundaries(self):
        for instant, zero, spent, required in ((False, False, False, 3), (True, False, False, 2),
                                               (True, False, True, 3), (False, True, False, 2),
                                               (True, True, False, 2)):
            for ap in (required - 1, required):
                with self.subTest(instant=instant, zero=zero, spent=spent, ap=ap):
                    state, action = self.prepare(ap, instant=instant, ordinary_zero=zero, spent=spent)
                    before = json.dumps(state, sort_keys=True)
                    self.assertEqual(action in legal_actions(state, 'a'), ap >= required)
                    view = observe(state, 'a')
                    card = next(c for c in view['sides']['a']['hand'] if c['card_id'] == 'K06')
                    self.assertFalse(card['action_point_free'])
                    if ap < required:
                        self.assertEqual(card['unavailable_reason'], '还差 1 点行动力')
                        with self.assertRaises(ValueError):
                            apply_action(state, 'a', action)
                    else:
                        entry = next(e for e in view['legal_actions'] if e['action'] == action)
                        self.assertEqual(entry['preview']['ap_cost'], required)
                        after = apply_action(json.loads(json.dumps(state)), 'a', entry['entity_action'])
                        self.assertEqual(after['sides']['a']['ap'], 0)
                        self.assertEqual(after['sides']['a']['used']['instant'], spent or (instant and not zero))
                        hits = [e for e in after['events'] if e['type'] == 'combat' and e.get('source') == 'a:lingke']
                        self.assertEqual([e['attribute'] for e in hits], ['光', '咒', '灵'])
                    self.assertEqual(json.dumps(state, sort_keys=True), before)

    def test_k03_substitution_does_not_inherit_original_surcharge(self):
        state = play(game(), 'K03')
        state['sides']['a']['ap'] = 0
        action = give(state, 'a', 'K06')
        after = apply_action(state, 'a', action)
        self.assertEqual(after['sides']['a']['ap'], 1)
        self.assertTrue(after['sides']['a']['used']['instant'])


if __name__ == '__main__':
    unittest.main()
