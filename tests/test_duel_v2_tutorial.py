import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, new_game, observe


class DuelV2TutorialTest(unittest.TestCase):
    def test_level_one_board_script(self) -> None:
        state = new_game(scenario='tutorial_l01_board')
        self.assertEqual(state['sides']['a']['hp'], 4)
        self.assertEqual(state['sides']['b']['hp'], 4)
        self.assertEqual(state['sides']['a']['shield'], 0)
        self.assertEqual(state['sides']['b']['shield'], 0)
        self.assertEqual(state['sides']['a']['ap'], 1)
        self.assertEqual(state['sides']['b']['ap'], 0)
        self.assertEqual(list(state['sides']['a']['order']), ['tutorial_protagonist', 'tutorial_bohe'])
        view = observe(state, 'a')
        self.assertTrue(view['tutorial']['enabled'])
        self.assertEqual(view['tutorial']['step_id'], 'l01_intro')
        self.assertEqual(view['tutorial']['placement'], 'center')
        self.assertIn('回合制卡牌对战', view['tutorial']['modal']['body'])
        self.assertIn('bench:a', view['tutorial']['spotlights'])
        types = {item['action']['type'] for item in view['legal_actions']}
        self.assertEqual(types, {'tutorial_ack'})
        self.assertNotIn('passive', view['sides']['a']['characters'][0])
        self.assertEqual(view['sides']['a']['hand'], [])

        for expected in ('l01_bench', 'l01_front', 'l01_life', 'l01_ap', 'l01_turn'):
            state = apply_action(state, 'a', {'type': 'tutorial_ack'})
            self.assertEqual(observe(state, 'a')['tutorial']['step_id'], expected)
        self.assertEqual(observe(state, 'a')['tutorial']['placement'], 'bottom-right')
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        self.assertEqual(observe(state, 'a')['tutorial']['step_id'], 'l01_p1_attack')
        self.assertEqual(observe(state, 'a')['tutorial']['cue'], '按住主角，并拖动到对战区')
        actions = [item['action'] for item in observe(state, 'a')['legal_actions']]
        self.assertEqual(actions, [{'type': 'attack', 'character_id': 'tutorial_protagonist'}])

        state = apply_action(state, 'a', {'type': 'attack', 'character_id': 'tutorial_protagonist'})
        self.assertEqual(state['sides']['b']['hp'], 2)
        self.assertEqual(state['sides']['b']['shield'], 0)
        self.assertEqual(state['sides']['a']['characters']['tutorial_protagonist']['harmony'], 0)
        self.assertEqual(state['sides']['a']['characters']['tutorial_protagonist']['energy'], 0)
        self.assertEqual(observe(state, 'a')['tutorial']['step_id'], 'l01_p1_end')
        self.assertEqual(
            [item['action']['type'] for item in observe(state, 'a')['legal_actions']],
            ['end_turn'],
        )

        state = apply_action(state, 'a', {'type': 'end_turn'})
        self.assertEqual(state['sides']['a']['characters']['tutorial_protagonist']['hp'], 3)
        self.assertEqual(state['sides']['b']['characters']['tutorial_nanali']['hp'], 3)
        self.assertEqual(state['active_side'], 'a')
        self.assertEqual(observe(state, 'a')['tutorial']['step_id'], 'l01_o1_watch')
        self.assertTrue(any('娜娜莉攻击主角' in text for text in observe(state, 'a')['logs']))

        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        self.assertEqual(observe(state, 'a')['tutorial']['step_id'], 'l01_p2_return')
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        self.assertEqual(state['winner'], 'a')
        self.assertEqual(state['reason'], 'tutorial_complete')
        self.assertTrue(any('本关完成' in text for text in observe(state, 'a')['logs']))

    def test_later_levels_are_playable(self) -> None:
        from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS, load_scenario
        for scenario_id in CAMPAIGN_LEVELS:
            load_scenario(scenario_id)
            state = new_game(scenario=scenario_id)
            self.assertEqual(state['flags']['tutorial']['scenario'], scenario_id)

    def test_level_three_plays_battle_card(self) -> None:
        state = new_game(scenario='tutorial_l02_hand')
        self.assertEqual(len(state['sides']['a']['hand']), 1)
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        plays = [item['action'] for item in observe(state, 'a')['legal_actions']
                 if item['action']['type'] == 'play_card']
        self.assertTrue(plays)
        state = apply_action(state, 'a', plays[0])
        self.assertEqual(state['sides']['b']['hp'], 9)

    def test_level_one_rejects_end_turn_before_attack(self) -> None:
        state = new_game(scenario='tutorial_l01_board')
        while observe(state, 'a')['tutorial']['step_id'] != 'l01_p1_attack':
            state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        types = {item['action']['type'] for item in observe(state, 'a')['legal_actions']}
        self.assertNotIn('end_turn', types)
        self.assertNotIn({'type': 'end_turn'}, legal_actions(state, 'a'))
