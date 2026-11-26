"""Battle release fills the actor, before combat and without a second after-use fill."""
import unittest
from unittest.mock import patch
from tests.test_duel_v2_latest_card_adjustments import game
from app.modules.card_game.engine.duel_v2 import apply_action, combat, flow
from app.modules.card_game.engine.duel_v2.state import card_instance, hero
from app.modules.card_game.engine.events import GameEvent


class BattleReleaseHarmonyTest(unittest.TestCase):
    def test_both_kits_fill_on_release_before_attack_and_never_afterwards(self):
        for cid, card_id in (('zero', 'Z03'), ('heiyu', 'P01')):
            with self.subTest(character=cid):
                s = game(('zero', 'heiyu', 'nanali', 'jiuyuan'))
                card = card_instance(s, card_id, 'a');s['sides']['a']['hand'] = [card]
                observed = []
                original_flow = flow.fire
                original_combat = combat.fire
                def observe_release(context, event, **data):
                    result = original_flow(context, event, **data)
                    if event in (GameEvent.V2_CARD_USED, GameEvent.V2_CARD_RELEASED):
                        observed.append((event, hero(context.state, 'a', cid)['harmony']))
                    return result
                def spend_before_hit(context, event, **data):
                    original_combat(context, event, **data)
                    if event == 'before_hits':
                        observed.append(('before_hits', hero(context.state, 'a', cid)['harmony']))
                        hero(context.state, 'a', cid)['harmony'] = 0
                with patch.object(flow, 'fire', side_effect=observe_release), \
                     patch.object(combat, 'fire', side_effect=spend_before_hit):
                    s = apply_action(s, 'a', dict(type='play_card', card_id=card['instance_id']))
                self.assertEqual(observed, [(GameEvent.V2_CARD_USED, 1),
                                           (GameEvent.V2_CARD_RELEASED, 3), ('before_hits', 3)])
                self.assertEqual(hero(s, 'a', cid)['harmony'], 0)
                other = 'zero' if cid == 'heiyu' else 'heiyu'
                self.assertEqual(hero(s, 'a', other)['harmony'], 0)

    def test_direct_battle_release_fills_without_cost_or_creating_source(self):
        for cid, card_id in (('zero', 'Z03'), ('heiyu', 'P01')):
            with self.subTest(character=cid):
                s = game(('zero', 'heiyu', 'nanali', 'jiuyuan'))
                s['sides']['a']['ap'] = 0
                flow.resolve_release(s, 'a', card_instance(s, card_id, 'a'))
                self.assertEqual(hero(s, 'a', cid)['harmony'], 3)
                self.assertEqual(s['sides']['a']['ap'], 0)
                self.assertIsNone(s['sides']['a']['last_front'])
