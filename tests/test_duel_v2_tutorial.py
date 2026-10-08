import unittest

from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, new_game, observe


class DuelV2TutorialTest(unittest.TestCase):
    def test_all_campaign_scripts_complete(self) -> None:
        from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS
        for scenario_id in CAMPAIGN_LEVELS:
            with self.subTest(scenario=scenario_id):
                state = new_game(scenario=scenario_id)
                for step in range(100):
                    if state['phase'] == 'finished':
                        break
                    actions = legal_actions(state, 'a')
                    self.assertTrue(actions)
                    state = apply_action(state, 'a', actions[0])
                self.assertEqual(state['reason'], 'tutorial_complete')

    def test_campaign_titles_use_current_terminology(self) -> None:
        from app.modules.card_game.content.duel_v2.tutorial.pack import CAMPAIGN_LEVELS, LEVEL_META, SCENARIOS
        self.assertEqual(len(CAMPAIGN_LEVELS), 13)
        self.assertEqual([level['id'] for level in LEVEL_META], list(CAMPAIGN_LEVELS))
        for level in LEVEL_META:
            self.assertEqual(level['title'], SCENARIOS[level['id']]['title'])
            self.assertNotIn('贯穿', level['title'])
        state = new_game(scenario='tutorial_l11_pierce')
        tutorial = observe(state, 'a')['tutorial']
        self.assertEqual(tutorial['title'], '第十关 穿透')
        self.assertEqual(tutorial['modal']['title'], '穿透')

    def test_roster_tutorial_explains_draw_and_fixed_deck_size(self) -> None:
        state = new_game(scenario='tutorial_l12_roster4')
        state = apply_action(state, 'a', {'type': 'mulligan', 'card_ids': []})
        tutorial = observe(state, 'a')['tutorial']
        self.assertEqual(tutorial['step_id'], 'l12_roster')
        self.assertIn('牌库共 32 张', tutorial['modal']['body'])
        self.assertIn('牌库仍为教学专用', tutorial['modal']['body'])
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        tutorial = observe(state, 'a')['tutorial']
        self.assertEqual(tutorial['step_id'], 'l12_empty')
        self.assertIn('自己的回合开始时，抽 1 张牌。', tutorial['modal']['body'])
        self.assertIn('抽牌时若牌库已空，会立刻失败。', tutorial['modal']['body'])
        self.assertNotIn('先手', tutorial['modal']['body'])
        self.assertNotIn('构筑时要留足牌库', tutorial['modal']['body'])
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        self.assertEqual(state['reason'], 'tutorial_complete')

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

        for expected in ('l01_bench', 'l01_opponent_front', 'l01_player_front', 'l01_life', 'l01_ap', 'l01_turn'):
            state = apply_action(state, 'a', {'type': 'tutorial_ack'})
            tutorial = observe(state, 'a')['tutorial']
            self.assertEqual(tutorial['step_id'], expected)
            if expected in ('l01_opponent_front', 'l01_player_front'):
                side = 'b' if expected == 'l01_opponent_front' else 'a'
                self.assertEqual(tutorial['spotlights'], [f'zone:{side}:front'])
                self.assertTrue(tutorial['ack_required'])
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
        self.assertEqual(observe(state, 'a')['tutorial']['step_id'], 'l01_p2_attack')
        state = apply_action(state, 'a', {'type': 'attack', 'character_id': 'tutorial_protagonist'})
        self.assertEqual(state['sides']['a']['characters']['tutorial_protagonist']['hp'], 1)
        self.assertEqual(state['sides']['b']['characters']['tutorial_nanali']['hp'], 1)
        self.assertEqual(state['sides']['a']['hp'], 4)
        self.assertEqual(state['sides']['b']['hp'], 2)
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


    def test_escalation_lesson_only_plays_second_player_turns_four_to_six(self):
        state = new_game(scenario='tutorial_l13_escalation')
        self.assertEqual((state['first_side'], state['active_side'], state['turn']), ('b', 'a', 4))
        self.assertEqual(state['sides']['a']['turn_count'], 2)
        self.assertEqual(state['sides']['b']['turn_count'], 2)
        self.assertEqual(observe(state, 'a')['sides']['a']['ultimate_remaining'], 1)
        energies = lambda s: [s['sides']['a']['characters'][cid]['energy'] for cid in ('tutorial_protagonist', 'tutorial_bohe')]
        self.assertEqual(energies(state), [4, 5])
        for _ in range(2):
            state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        state = apply_action(state, 'a', {'type': 'end_turn'})
        self.assertEqual(state['turn'], 6)
        self.assertEqual(energies(state), [5, 5])
        self.assertEqual(observe(state, 'a')['sides']['a']['ultimate_remaining'], 2)
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        for index, cid in enumerate(('tutorial_protagonist', 'tutorial_bohe')):
            self.assertEqual(legal_actions(state, 'a'), [{'type': 'ultimate', 'character_id': cid}])
            state = apply_action(state, 'a', {'type': 'ultimate', 'character_id': cid})
            self.assertEqual(state['sides']['a']['ap'], 2)
            self.assertEqual(observe(state, 'a')['sides']['a']['ultimate_remaining'], 1-index)
        self.assertEqual(energies(state), [0, 0])
        self.assertIn('第 13 回合', observe(state, 'a')['tutorial']['modal']['body'])
        self.assertIn('第 20 回合', observe(state, 'a')['tutorial']['modal']['body'])
        state = apply_action(state, 'a', {'type': 'tutorial_ack'})
        self.assertEqual(state['reason'], 'tutorial_complete')
        self.assertEqual(state['turn'], 6)

    def test_only_escalation_lesson_enables_escalation(self):
        from app.modules.card_game.content.duel_v2.tutorial import CAMPAIGN_LEVELS
        for scenario in CAMPAIGN_LEVELS:
            self.assertEqual(new_game(scenario=scenario)['escalation_enabled'], scenario == 'tutorial_l13_escalation')


    def test_ranged_lesson_preserves_front_and_one_hp_after_json_restore(self):
        import json
        from app.modules.card_game.content.duel_v2.catalog import CHARACTERS
        from app.modules.card_game.engine.duel_v2.projection import preview
        for _ in range(5):
            state = new_game(scenario='tutorial_l14_ranged')
            self.assertNotIn('tutorial_haiyue', CHARACTERS)
            state = apply_action(state, 'a', {'type': 'tutorial_ack'})
            state = apply_action(state, 'a', {'type': 'play_card', 'card_id': 'a-1'})
            self.assertEqual(state['sides']['a']['front'], 'tutorial_bohe')
            self.assertEqual(state['sides']['a']['ap'], 1)
            state = apply_action(state, 'a', {'type': 'ultimate', 'character_id': 'tutorial_haiyue'})
            state = json.loads(json.dumps(state))
            action = {'type': 'attack', 'character_id': 'tutorial_haiyue'}
            self.assertEqual(legal_actions(state, 'a'), [action])
            predicted = preview(state, 'a', action)
            self.assertFalse(predicted['will_switch'])
            self.assertEqual(predicted['counter'], 0)
            seq = state['event_seq']
            state = apply_action(state, 'a', action)
            own = state['sides']['a']
            self.assertEqual(own['front'], 'tutorial_bohe')
            self.assertEqual(own['characters']['tutorial_haiyue']['hp'], 1)
            self.assertEqual(own['ap'], 0)
            self.assertFalse(own['normal_attack_available'])
            events = [e for e in state['events'] if e['seq'] > seq]
            attack = next(e for e in events if e['type'] == 'attack')
            self.assertEqual(attack['target'], 'b:tutorial_nanali')
            self.assertEqual(attack['counter'], 0)
            self.assertFalse(any(e['type'] in ('enter', 'move', 'harmony')
                                 and e.get('actor', '').startswith('a:') for e in events))
            state = apply_action(state, 'a', {'type': 'tutorial_ack'})
            self.assertEqual(state['reason'], 'tutorial_complete')
