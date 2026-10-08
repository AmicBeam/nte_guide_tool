import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.content.duel_v2.catalog import LIGHT_ENERGY_MAX
from app.modules.card_game.engine.duel_v2 import apply_action, choose_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.projection import preview
from app.modules.card_game.engine.duel_v2.state import card_instance, damage, hero, new_character


class DuelV2PresentationTest(unittest.TestCase):
    def test_front_knockdown_clears_harmony_marks_in_event_patches(self):
        from app.modules.card_game.engine.duel_v2.state import damage
        state = self.game()
        state['sides']['a']['front'] = 'zero'
        hero(state, 'a', 'zero')['harmony'] = 2
        before = capture_public_board(state)
        state['_public_board'] = deepcopy(before)
        self.assertTrue(before['sides']['a']['harmony_available'])
        self.assertTrue(any(c['harmony_ready'] for c in before['sides']['a']['characters']))
        state['events'] = []
        damage(state, 'a', 'zero', 100, source='b:nanali', kind='combat')
        down_index = next(i for i, e in enumerate(state['events']) if e['type'] == 'down')
        board = replay_public_board(before, state['events'][:down_index + 1])
        team = board['sides']['a']
        self.assertIsNone(team['front'])
        self.assertFalse(team['harmony_available'])
        self.assertTrue(all(not c['harmony_source'] and not c['harmony_ready'] for c in team['characters']))
        self.assertEqual(board['turn'], before['turn'])

    def game(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        state['sides']['a']['ap'] = 2
        state['sides']['b']['shield'] = 0
        for side in ('a', 'b'):
            if 'xun' not in state['sides'][side]['characters']:
                state['sides'][side]['characters']['xun'] = new_character('xun')
                state['sides'][side]['order'].append('xun')
        return state

    def hand(self, state, card_id, side='a', copy=False):
        card = card_instance(state, card_id, side, copy=copy)
        state['sides'][side]['hand'] = [card]
        return card

    def play(self, state, card_id, target=None, side='a'):
        card = self.hand(state, card_id, side)
        action = {'type': 'play_card', 'card_id': card['instance_id']}
        if target:
            action['target_id'] = target
        return apply_action(state, side, action), card

    def events(self, view, kind=None):
        items = view['presentation']['events']
        return [event for event in items if kind is None or event['type'] == kind]

    def test_play_and_combat_include_public_card_and_values(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        after, card = self.play(state, 'N03')
        own = observe(after, 'a')
        enemy = observe(after, 'b')
        plays = self.events(own, 'play') + self.events(enemy, 'play')
        self.assertTrue(plays)
        for event in plays:
            self.assertEqual(event['card']['name'], card['name'])
            self.assertEqual(event['card']['instance_id'], card['instance_id'])
            self.assertEqual(event['card']['cost'], card['cost'])
            self.assertEqual(event['actor'], 'a:nanali')
            self.assertIn('patch', event)
        attacks = self.events(own, 'attack')
        self.assertEqual(len(attacks), 1)
        self.assertEqual(attacks[0]['attack'], attacks[0]['amount'])
        self.assertIn('counter', attacks[0])
        self.assertTrue(attacks[0]['group_id'])
        combats = [event for event in self.events(own) if event['type'] == 'combat']
        self.assertGreaterEqual(len(combats), 1)
        groups = {event['group_id'] for event in combats}
        self.assertEqual(groups, {attacks[0]['group_id']})
        for event in combats:
            self.assertIn('before', event)
            self.assertIn('after', event)
            self.assertIn('hp', event['before'])
            self.assertIn('hp', event['after'])

    def test_double_hit_uses_separate_groups(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        # Generic multi-hit contract, independent of the retired X05 effect.
        from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
        after = deepcopy(state)
        operation(after, 'a', 'nanali').sortie(hits=2, no_counter=True)
        finish_operation(after)
        own = observe(after, 'a')
        attacks = self.events(own, 'attack')
        self.assertEqual(len(attacks), 2)
        self.assertNotEqual(attacks[0]['group_id'], attacks[1]['group_id'])
        self.assertEqual(attacks[0]['hit_index'], 0)
        self.assertEqual(attacks[1]['hit_index'], 1)
        first = [event for event in self.events(own, 'combat') if event.get('group_id') == attacks[0]['group_id']]
        second = [event for event in self.events(own, 'combat') if event.get('group_id') == attacks[1]['group_id']]
        self.assertTrue(first)
        self.assertTrue(second)

    def test_private_draw_and_inspect_are_isolated(self):
        state = self.game()
        after, _ = self.play(state, 'J02')
        own = observe(after, 'a')
        enemy = observe(after, 'b')
        draws = [event for event in self.events(own, 'draw') if event.get('side') == 'a']
        self.assertTrue(draws)
        self.assertIn('card', draws[-1])
        self.assertTrue(draws[-1]['card']['name'])
        self.assertIn(draws[-1]['card']['name'], draws[-1]['text'])
        own_log = next(event for event in own['events'] if event['type'] == 'draw' and event.get('side') == 'a')
        self.assertIn('card', own_log)
        self.assertIn(own_log['card']['name'], own_log['text'])
        hidden = next(event for event in self.events(enemy, 'draw') if event.get('side') == 'a')
        self.assertNotIn('card', hidden)
        self.assertNotIn('private_card', hidden)
        self.assertNotIn('「', hidden['text'])
        enemy_log = next(event for event in enemy['events'] if event['type'] == 'draw' and event.get('side') == 'a')
        self.assertNotIn('card', enemy_log)
        self.assertNotIn('「', enemy_log['text'])
        inspect = self.play(after, 'J01')[0]
        own_view, enemy_view = observe(inspect, 'a'), observe(inspect, 'b')
        secret = own_view['pending_choice']['choices'][0]['card']['name']
        dumped = json.dumps(enemy_view, ensure_ascii=False)
        self.assertNotIn(secret, dumped)
        self.assertNotIn(secret, json.dumps(enemy_view['presentation'], ensure_ascii=False))
        self.assertEqual(enemy_view['pending_choice']['choices'], [])

    def test_turn_start_draw_is_after_turn_and_marked_present(self):
        state = self.game()
        after = apply_action(state, 'a', {'type': 'end_turn'})
        view = observe(after, 'b')
        events = view['presentation']['events']
        turn_at = next(index for index, event in enumerate(events)
                       if event['type'] == 'turn' and event.get('side') == 'b')
        draw_at = next(index for index, event in enumerate(events)
                       if event['type'] == 'draw' and event.get('side') == 'b' and index > turn_at)
        self.assertGreater(draw_at, turn_at)
        self.assertTrue(events[turn_at].get('present'))
        self.assertTrue(events[draw_at].get('present'))
        self.assertIn('card', events[draw_at])
        self.assertTrue(all(
            event['type'] != 'draw' or event.get('side') != 'b'
            for event in events[turn_at + 1:draw_at]
        ))

    def test_resource_and_finish_are_queued_not_silent(self):
        from app.modules.card_game.engine.duel_v2.presentation import SILENT_PRESENT_TYPES
        self.assertNotIn('resource', SILENT_PRESENT_TYPES)
        self.assertNotIn('finish', SILENT_PRESENT_TYPES)
        self.assertNotIn('initiative', SILENT_PRESENT_TYPES)
        self.assertNotIn('mulligan', SILENT_PRESENT_TYPES)

    def test_turn_banner_precedes_recovery_return_and_start_effects(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        hero(state, 'b', 'zero')['shield'] = 2
        hero(state, 'b', 'jiuyuan').update(hp=0, down_turns=1)
        self.sync_public_board(state)
        after = apply_action(state, 'a', {'type': 'end_turn'})
        for viewer in ('a', 'b'):
            events = self.replay_action_board(state, after, viewer)
            turn_at = next(i for i, event in enumerate(events) if event['type'] == 'turn')
            for kind in ('revive', 'move', 'gain', 'draw'):
                event_at = next(i for i, event in enumerate(events) if event['type'] == kind)
                self.assertGreater(event_at, turn_at, kind)
            banner_board = replay_public_board(capture_public_board(state), events[:turn_at + 1])
            self.assertEqual(banner_board['active_side'], 'b')
            self.assertEqual(banner_board['sides']['b']['front'], 'zero')
            characters = {h['id']: h for h in banner_board['sides']['b']['characters']}
            self.assertEqual(characters['zero']['shield'], 2)
            self.assertEqual(characters['jiuyuan']['hp'], 0)

    def test_interaction_hints_cover_current_card_policies(self):
        state = self.game()
        mapping = {'N03': 'cast', 'N07': 'form', 'N06': 'cast',
                   'Z01': 'cast', 'Y06': 'target', 'J06': 'cast', 'X04': 'cast'}
        for card_id, kind in mapping.items():
            with self.subTest(card=card_id):
                current = deepcopy(state)
                current['sides']['a'].update(ap=2, front='zero', surplus=True)
                if card_id == 'Y06':
                    hero(current, 'a', 'jiuyuan').update(hp=0, down_turns=3)
                card = self.hand(current, card_id)
                entry = next(item for item in observe(current, 'a')['legal_actions']
                             if item['action']['type'] == 'play_card'
                             and item['action']['card_id'] == card['instance_id'])
                self.assertEqual(entry['interaction']['kind'], kind)
                self.assertEqual(entry['interaction']['actor_id'], f"a:{CARDS[card_id]['character_id']}")
                if kind == 'form':
                    self.assertEqual(entry['interaction']['target_ids'], ['a:nanali'])
        attack = next(item for item in observe(state, 'a')['legal_actions'] if item['action']['type'] == 'attack')
        self.assertEqual(attack['interaction']['kind'], 'attack')
        self.assertTrue(attack['interaction']['actor_id'].startswith('a:'))

    def test_y06_target_hints_match_each_canonical_revival_action(self):
        state = self.game()
        for cid in ('zero', 'jiuyuan'):
            hero(state, 'a', cid).update(hp=0, down_turns=3)
        hero(state, 'b', 'zero').update(hp=0, down_turns=3)
        card = self.hand(state, 'Y06')
        actions = [action for action in legal_actions(state, 'a')
                   if action.get('type') == 'play_card' and action.get('card_id') == card['instance_id']]
        self.assertEqual({action['target_id'] for action in actions}, {'a:zero', 'a:jiuyuan'})
        entries = [item for item in observe(state, 'a')['legal_actions'] if item['action'] in actions]
        self.assertEqual(len(entries), len(actions))
        for entry in entries:
            self.assertEqual(entry['interaction']['kind'], 'target')
            self.assertEqual(entry['interaction']['actor_id'], 'a:iloy')
            self.assertEqual(entry['interaction']['target_ids'], [entry['action']['target_id']])

    def test_n06_is_targetless_self_revival_even_while_downed(self):
        state = self.game()
        hero(state, 'a', 'nanali').update(hp=0, down_turns=3)
        hero(state, 'a', 'zero').update(hp=0, down_turns=3)
        card = self.hand(state, 'N06')
        entries = [item for item in observe(state, 'a')['legal_actions']
                   if item['action'].get('type') == 'play_card'
                   and item['action'].get('card_id') == card['instance_id']]
        self.assertEqual(len(entries), 1)
        self.assertNotIn('target_id', entries[0]['action'])
        self.assertEqual(entries[0]['interaction']['kind'], 'cast')
        self.assertEqual(entries[0]['interaction']['actor_id'], 'a:nanali')
        result = apply_action(state, 'a', entries[0]['action'])
        self.assertEqual(hero(result, 'a', 'nanali')['hp'], hero(result, 'a', 'nanali')['max_hp'])
        self.assertEqual(hero(result, 'a', 'zero')['hp'], 0)

    def sync_public_board(self, state):
        if '_public_board' in state:
            state['_public_board'] = capture_public_board(state)
        return state

    def replay_action_board(self, before_state, after_state, viewer='a'):
        before_seq = before_state['event_seq']
        before_board = capture_public_board(before_state)
        after_board = capture_public_board(after_state)
        events = [event for event in observe(after_state, viewer)['presentation']['events']
                  if event['seq'] > before_seq]
        replayed = replay_public_board(before_board, events)
        self.maxDiff = None
        self.assertEqual(replayed, after_board)
        return events

    def test_presentation_patches_replay_public_board(self):
        state = self.game()
        state['sides']['a']['front'] = 'nanali'
        state['sides']['b']['front'] = 'zero'
        state['sides']['b']['shield'] = 4
        hero(state, 'b', 'zero').update(hp=4, shield=1)
        self.sync_public_board(state)
        attack_action = next(action for action in legal_actions(state, 'a') if action['type'] == 'attack')
        after_attack = apply_action(state, 'a', attack_action)
        self.replay_action_board(state, after_attack)
        mixed = self.events(observe(after_attack, 'a'), 'combat')
        self.assertTrue(any(event['before'].get('shield') and event['after'].get('hp') < event['before'].get('hp')
                            for event in mixed))

        form_state = deepcopy(after_attack)
        form_state['sides']['a']['ap'] = 2
        self.sync_public_board(form_state)
        form_before = deepcopy(form_state)
        form_after, _ = self.play(form_state, 'N07')
        form_events = self.replay_action_board(form_before, form_after)
        self.assertTrue(any(event['type'] == 'shape' for event in form_events))

        n06_state = deepcopy(form_after)
        n06_state['sides']['a']['ap'] = 2
        damage(n06_state, 'a', 'nanali', 99, source='b:zero')
        self.sync_public_board(n06_state)
        n06_before = deepcopy(n06_state)
        n06_after, _ = self.play(n06_state, 'N06')
        n06_events = self.replay_action_board(n06_before, n06_after)
        self.assertTrue(any(event['type'] == 'revive' and event['target'] == 'a:nanali' for event in n06_events))

        x05_state = deepcopy(n06_after)
        x05_state['sides']['a']['ap'] = 2
        x05_state['sides']['a']['front'] = 'xun'
        self.sync_public_board(x05_state)
        x05_before = deepcopy(x05_state)
        x05_after, _ = self.play(x05_state, 'X05')
        x05_events = self.replay_action_board(x05_before, x05_after)
        attacks = [event for event in x05_events if event['type'] == 'attack']
        self.assertEqual(attacks, [])
        self.assertTrue(any(event['type'] == 'move' and event['target'] == 'a:xun' for event in x05_events))
        self.assertIsNone(x05_after['sides']['a']['front'])
        self.assertFalse(any(event.get('card', {}).get('copy') for event in x05_events))

        stale = deepcopy(x05_after)
        for event in stale['events']:
            for key in ('action_id', 'action_version', 'actor', 'patch', 'card', 'before', 'after', 'group_id'):
                event.pop(key, None)
        stale.pop('_public_board', None)
        stale.pop('current_action_id', None)
        stale['sides']['a']['ap'] = 2
        stale_before = deepcopy(stale)
        stale_after, _ = self.play(stale, 'N01')
        self.replay_action_board(stale_before, stale_after)

    def test_ai_multi_action_keeps_distinct_action_ids(self):
        state = self.game()
        before = state['event_seq']
        versions = []
        ids = []
        current = state
        for _ in range(3):
            if current['phase'] == 'finished':
                break
            action = choose_action(current, 'a')
            current = apply_action(current, 'a', action)
            versions.append(current['version'])
            ids.append(current['current_action_id'])
        self.assertEqual(len(set(ids)), len(ids))
        view = observe(current, 'b')
        fresh = [event for event in view['presentation']['events'] if event['seq'] > before]
        self.assertTrue(fresh)
        seen = {event['action_id'] for event in fresh}
        self.assertGreaterEqual(len(seen), 2)
        self.assertTrue(all(event['action_id'] for event in fresh))
        plays = [event for event in fresh if event['type'] == 'play']
        for event in plays:
            self.assertIn('card', event)
            self.assertTrue(event['card']['name'])

    def test_old_snapshot_without_metadata_still_projects(self):
        state = self.game()
        state['sides']['b']['front'] = 'zero'
        after, card = self.play(state, 'N03')
        raw = deepcopy(after)
        for event in raw['events']:
            for key in ('action_id', 'action_version', 'actor', 'patch', 'card', 'before', 'after', 'group_id'):
                event.pop(key, None)
        raw.pop('_public_board', None)
        raw.pop('current_action_id', None)
        view = observe(raw, 'b')
        self.assertEqual(view['presentation']['schema_version'], 1)
        self.assertEqual(view['presentation']['cursor'], raw['event_seq'])
        play = next(event for event in view['presentation']['events'] if event['type'] == 'play')
        self.assertTrue(play['action_id'])
        self.assertNotIn('card', play)
        self.assertNotIn(card['name'], json.dumps(play, ensure_ascii=False).replace(play['text'], ''))
        self.assertIn('「', play['text'])

    def test_after_seq_filters_presentation_only(self):
        state = self.game()
        after, _ = self.play(state, 'N01')
        cursor = after['event_seq'] - 1
        full = observe(after, 'a')
        filtered = observe(after, 'a', after_seq=cursor)
        self.assertEqual(full['sides'], filtered['sides'])
        self.assertEqual(full['legal_actions'], filtered['legal_actions'])
        self.assertTrue(filtered['presentation']['events'])
        self.assertTrue(all(event['seq'] > cursor for event in filtered['presentation']['events']))
        self.assertGreater(len(full['presentation']['events']), len(filtered['presentation']['events']))

    def test_preview_and_observe_do_not_leak_future_or_hidden_cards(self):
        state = self.game()
        state['sides']['a']['front'] = 'nanali'
        hero(state, 'a', 'nanali')['harmony'] = 2
        state['sides']['b']['front'] = 'zero'
        secret = 'SECRET-FUTURE-CARD'
        for card in state['sides']['a']['deck'] + state['sides']['b']['deck'] + state['sides']['b']['hand']:
            card['name'] = secret
        view = observe(state, 'a')
        dumped = json.dumps(view, ensure_ascii=False)
        self.assertNotIn(secret, dumped)
        action = next(item['action'] for item in view['legal_actions'] if item['action']['type'] == 'attack')
        dumped_preview = json.dumps(preview(state, 'a', action), ensure_ascii=False)
        self.assertNotIn(secret, dumped_preview)

    def test_ultimate_weapon_family_growth_knockdown_and_revival_events(self):
        state = self.game()
        nanali = hero(state, 'a', 'nanali')
        nanali.update(energy=LIGHT_ENERGY_MAX, shape='N07')
        after = apply_action(state, 'a', {'type': 'ultimate', 'character_id': 'nanali'})
        awaken = self.events(observe(after, 'a'), 'ultimate')[-1]
        self.assertEqual(awaken['actor'], 'a:nanali')
        self.assertEqual(awaken['after']['awakened'], True)
        self.assertEqual(awaken['after']['ultimate_turns'], 2)
        after['sides']['a']['ap'] = 2
        after, _ = self.play(after, 'N07')
        shape = self.events(observe(after, 'a'), 'shape')[-1]
        self.assertEqual(shape['card']['type'], 'form')
        after['sides']['a']['front'] = 'zero'
        after['sides']['a']['ap'] = 2
        hero(after, 'a', 'zero')['harmony'] = 2
        after['sides']['b']['front'] = 'zero'
        hero(after, 'b', 'zero')['hp'] = 1
        after, _ = self.play(after, 'N03')
        view = observe(after, 'a')
        self.assertTrue(any(e.get('card', {}).get('card_id') == 'NF01' for e in self.events(view, 'gain')))
        self.assertFalse(self.events(view, 'record'))
        self.assertTrue(self.events(view, 'down'))
        self.assertTrue(self.events(view, 'enter'))
        self.assertFalse(any(event.get('card', {}).get('copy') for event in self.events(view, 'gain')))
        after['sides']['a']['ap'] = 2
        after, _ = self.play(after, 'NF01')
        self.assertEqual(hero(after, 'a', 'nanali')['permanent_growth'], {'attack': 1, 'max_hp': 1})
        self.assertTrue(any('永久攻击 +1' in event['text'] for event in self.events(observe(after, 'a'))))
        revived = deepcopy(after)
        hero(revived, 'a', 'zero').update(hp=0, down_turns=1)
        revived['sides']['a']['ap'] = 0
        revived = apply_action(revived, 'a', {'type': 'end_turn'})
        revived = apply_action(revived, 'b', {'type': 'end_turn'})
        revive = self.events(observe(revived, 'a'), 'revive')
        self.assertTrue(revive)
        self.assertEqual(revive[-1]['target'], 'a:zero')



    def test_http_after_seq_query_matches_engine_filter(self):
        from tests.test_duel_v2_api import DuelV2ApiTest
        case = DuelV2ApiTest('test_catalog_and_build_are_versioned_and_strict')
        case.setUp()
        try:
            token, view = case._pvp()
            cursor = view['presentation']['cursor']
            payload = {'room_code': case.code, 'request_id': 'presentation-after-seq',
                       'expected_version': view['version'], 'action': case._action(view, 'attack')}
            updated = case._post('/api/duel-v2/action', payload, token=token)['game']
            filtered = case._get(f'/api/duel-v2/state?after_seq={cursor}', token=token)['game']
            self.assertTrue(all(event['seq'] > cursor for event in filtered['presentation']['events']))
            self.assertEqual(filtered['version'], updated['version'])
            self.assertIn('interaction', filtered['legal_actions'][0])
            self.assertTrue(any(event['type'] in ('attack', 'play', 'enter', 'combat')
                                for event in filtered['presentation']['events']))
        finally:
            case.tearDown()


if __name__ == '__main__':
    unittest.main()
