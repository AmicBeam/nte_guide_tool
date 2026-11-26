"""Team energy on solitary and Nanali's two harmony triggers."""
import json
import unittest

from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.events import GameEvent
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.state import hero, damage
from tests.test_duel_v2_latest_card_adjustments import game, play


class TeamEnergyWeaveTest(unittest.TestCase):
    def test_solitary_recovers_team_once_before_front_entry(self):
        s = game(('zhenhong', 'zero', 'nanali', 'iloy'))
        c = operation(s, 'a', 'zero')
        fire(c, GameEvent.V2_SURPLUS)
        self.assertEqual([hero(s, 'a', cid)['energy'] for cid in
                          ('zhenhong', 'zero', 'nanali', 'iloy')], [0, 1, 1, 1])
        self.assertIsNone(s['sides']['a']['front'])
        fire(c, GameEvent.V2_SURPLUS)
        finish_operation(s)
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 1)
        self.assertEqual(hero(s, 'b', 'zero')['energy'], 0)
        s = json.loads(json.dumps(s))
        from app.modules.card_game.engine.duel_v2.entities import hydrate_entities
        hydrate_entities(s)
        s['turn'] += 2
        fire(operation(s, 'a', 'zero'), GameEvent.V2_SURPLUS)
        self.assertEqual(hero(s, 'a', 'zhenhong')['energy'], 1)
        self.assertEqual(hero(s, 'a', 'zero')['energy'], 2)

    def test_solitary_obeys_caps_down_and_special_energy(self):
        s = game(('zhenhong', 'zero', 'nanali', 'xun'))
        s['sides']['a']['front'] = 'zhenhong'
        hero(s, 'a', 'zhenhong')['awakened'] = True
        hero(s, 'a', 'zero')['energy'] = 5
        hero(s, 'a', 'xun')['energy'] = 0
        damage(s, 'a', 'nanali', 100)
        fire(operation(s, 'a', 'zero'), GameEvent.V2_SURPLUS)
        self.assertEqual([hero(s, 'a', cid)['energy'] for cid in
                          ('zhenhong', 'zero', 'nanali', 'xun')], [0, 5, 0, 0])

    def test_weave_actor_and_payer_share_opening_three_card_limit(self):
        s = game(('nanali', 'baicang', 'zero', 'iloy'))
        self.assertEqual(s['sides']['a']['nanali_family'], 1)
        hero(s, 'a', 'baicang')['harmony'] = 2
        s['sides']['a']['front'] = 'baicang'
        s = play(s, 'B03', 'a:nanali')
        self.assertEqual(s['sides']['a']['nanali_family'], 2)
        hero(s, 'a', 'nanali')['harmony'] = 3
        s['sides']['a'].update(last_front='nanali',
            last_actor_entity_id=hero(s, 'a', 'nanali').entity_id)
        from app.modules.card_game.engine.duel_v2 import apply_action
        s = apply_action(s, 'a', dict(type='attack', character_id='baicang'))
        self.assertEqual(s['sides']['a']['nanali_family'], 3)
        self.assertEqual(sum(c['card_id'] == 'NF01' for c in s['sides']['a']['hand']), 2)
        fire(operation(s, 'a', 'nanali'), GameEvent.V2_HARMONY_TRIGGERED,
             kind='创生', entering='nanali', payer='zero')
        self.assertEqual(s['sides']['a']['nanali_family'], 3)

    def test_uninvolved_nanali_and_other_harmony_do_not_grant_family(self):
        s = game(('nanali', 'baicang', 'zero', 'iloy'))
        c = operation(s, 'a', 'zero')
        fire(c, GameEvent.V2_HARMONY_TRIGGERED,
             kind='覆纹', entering='baicang', payer='iloy')
        fire(c, GameEvent.V2_HARMONY_TRIGGERED,
             kind='延滞', entering='nanali', payer='zero')
        self.assertEqual(s['sides']['a']['nanali_family'], 1)
