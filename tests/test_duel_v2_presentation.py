import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import apply_action, choose_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.projection import preview
from app.modules.card_game.engine.duel_v2.state import card_instance, hero


class DuelV2PresentationTest(unittest.TestCase):
    def game(self):
        state = new_game(seed=9, first_side='a', skip_mulligan=True)
        state['sides']['a']['ap'] = 2
        state['sides']['b']['shield'] = 0
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
        after, _ = self.play(state, 'X08')
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
        after, _ = self.play(state, 'N01')
        own = observe(after, 'a')
        enemy = observe(after, 'b')
        draws = self.events(own, 'draw')
        self.assertTrue(draws)
        self.assertIn('card', draws[-1])
        self.assertTrue(draws[-1]['card']['name'])
        hidden = self.events(enemy, 'draw')[-1]
        self.assertNotIn('card', hidden)
        self.assertNotIn('private_card', hidden)
        inspect = self.play(after, 'J03')[0]
        own_view, enemy_view = observe(inspect, 'a'), observe(inspect, 'b')
        secret = own_view['pending_choice']['choices'][0]['card']['name']
        dumped = json.dumps(enemy_view, ensure_ascii=False)
        self.assertNotIn(secret, dumped)
        self.assertNotIn(secret, json.dumps(enemy_view['presentation'], ensure_ascii=False))
        self.assertEqual(enemy_view['pending_choice']['choices'], [])

    def test_interaction_hints_cover_card_policies(self):
        state = self.game()
        mapping = {
            'N03': 'cast',
            'N02': 'form',
            'N06': 'target',
            'Z01': 'target',
            'J06': 'choice',
            'X04': 'choice',
        }
        for card_id, kind in mapping.items():
            current = deepcopy(state)
            current['sides']['a']['ap'] = 2
            if card_id == 'X04':
                current['sides']['a']['discard'].append(card_instance(current, 'X01', 'a'))
            if card_id in ('Z01', 'J06'):
                current['sides']['b']['front'] = 'zero'
            if card_id == 'J06':
                current['sides']['b']['hand'] = [card_instance(current, 'N01', 'b')]
            card = self.hand(current, card_id)
            entry = next(item for item in observe(current, 'a')['legal_actions']
                         if item['action']['type'] == 'play_card' and item['action']['card_id'] == card['instance_id'])
            self.assertEqual(entry['interaction']['kind'], kind)
            self.assertEqual(entry['interaction']['actor_id'], f"a:{CARDS[card_id]['character_id']}")
            if kind == 'target' and card_id == 'N06':
                self.assertEqual(entry['interaction']['target_ids'], [entry['action']['target_id']])
                self.assertNotIn('a:nanali', entry['interaction']['target_ids'])
            if kind == 'form':
                self.assertEqual(entry['interaction']['target_ids'], ['a:nanali'])
        attack = next(item for item in observe(state, 'a')['legal_actions'] if item['action']['type'] == 'attack')
        self.assertEqual(attack['interaction']['kind'], 'attack')
        self.assertTrue(attack['interaction']['actor_id'].startswith('a:'))

    def n06_entries(self, state, side='a'):
        return [item for item in observe(state, side)['legal_actions']
                if item['action'].get('type') == 'play_card'
                and next((card for card in state['sides'][side]['hand']
                          if card['instance_id'] == item['action']['card_id']), {}).get('card_id') == 'N06']

    def test_n06_target_ids_match_each_canonical_action(self):
        state = self.game()
        hero(state, 'a', 'zero')['hp'] = 0
        card = self.hand(state, 'N06')
        actions = legal_actions(state, 'a')
        n06_actions = [action for action in actions
                       if action.get('type') == 'play_card' and action.get('card_id') == card['instance_id']]
        self.assertEqual({action['target_id'] for action in n06_actions}, {'a:jiuyuan', 'a:xun'})
        entries = self.n06_entries(state)
        self.assertEqual(len(entries), len(n06_actions))
        seen = []
        for entry in entries:
            target = entry['action']['target_id']
            self.assertEqual(entry['interaction']['kind'], 'target')
            self.assertEqual(entry['interaction']['actor_id'], 'a:nanali')
            self.assertEqual(entry['interaction']['target_ids'], [target])
            self.assertNotIn('a:nanali', entry['interaction']['target_ids'])
            self.assertNotIn('a:zero', entry['interaction']['target_ids'])
            seen.append(target)
        self.assertEqual(set(seen), {'a:jiuyuan', 'a:xun'})
        copied = deepcopy(state)
        copy_card = card_instance(copied, 'N06', 'a', copy=True)
        copied['sides']['a']['hand'] = [copy_card]
        copied['sides']['a']['ap'] = 2
        copy_entries = self.n06_entries(copied)
        self.assertTrue(copy_entries)
        for entry in copy_entries:
            self.assertEqual(entry['interaction']['actor_id'], 'a:xun')
            self.assertEqual(entry['interaction']['target_ids'], [entry['action']['target_id']])
            self.assertNotIn('a:xun', entry['interaction']['target_ids'])
            self.assertTrue(entry['action']['target_id'].startswith('a:'))
            self.assertNotEqual(entry['action']['target_id'], 'b:zero')

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
        hero(state, 'b', 'zero').update(hp=4, shield=2)
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
        form_after, _ = self.play(form_state, 'N02')
        form_events = self.replay_action_board(form_before, form_after)
        self.assertTrue(any(event['type'] == 'shape' for event in form_events))

        n06_state = deepcopy(form_after)
        n06_state['sides']['a']['ap'] = 2
        hero(n06_state, 'a', 'jiuyuan')['hp'] = 3
        self.sync_public_board(n06_state)
        n06_before = deepcopy(n06_state)
        n06_after, _ = self.play(n06_state, 'N06', target='a:jiuyuan')
        n06_events = self.replay_action_board(n06_before, n06_after)
        self.assertTrue(any(event['type'] == 'heal' and event['target'] == 'a:jiuyuan' for event in n06_events))

        x08_state = deepcopy(n06_after)
        x08_state['sides']['a']['ap'] = 2
        x08_state['sides']['a']['front'] = 'xun'
        x08_state['sides']['a']['records'] = [
            {'card_id': 'N01', 'name': CARDS['N01']['name'], 'type': 'tactic', 'character_id': 'nanali'}]
        self.sync_public_board(x08_state)
        x08_before = deepcopy(x08_state)
        x08_after, _ = self.play(x08_state, 'X08')
        x08_events = self.replay_action_board(x08_before, x08_after)
        attacks = [event for event in x08_events if event['type'] == 'attack']
        self.assertEqual(len(attacks), 2)
        self.assertNotEqual(attacks[0]['group_id'], attacks[1]['group_id'])
        self.assertTrue(any(event.get('card', {}).get('copy') for event in x08_events if event['type'] == 'gain'))

        stale = deepcopy(x08_after)
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
        hero(state, 'a', 'zero')['harmony'] = 2
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

    def test_awaken_shape_growth_record_copy_down_revive(self):
        state = self.game()
        nanali = hero(state, 'a', 'nanali')
        nanali.update(energy=4, shape='N02')
        after = apply_action(state, 'a', {'type': 'awaken', 'character_id': 'nanali'})
        awaken = self.events(observe(after, 'a'), 'awaken')[-1]
        self.assertEqual(awaken['actor'], 'a:nanali')
        self.assertEqual(awaken['after']['awakened'], True)
        after['sides']['a']['ap'] = 2
        after, _ = self.play(after, 'N02')
        shape = self.events(observe(after, 'a'), 'shape')[-1]
        self.assertEqual(shape['card']['type'], 'form')
        after['sides']['a']['front'] = 'zero'
        after['sides']['a']['ap'] = 2
        hero(after, 'a', 'nanali')['harmony'] = 2
        after['sides']['b']['front'] = 'zero'
        hero(after, 'b', 'zero')['hp'] = 1
        after, _ = self.play(after, 'N03')
        view = observe(after, 'a')
        self.assertTrue(self.events(view, 'growth'))
        self.assertTrue(self.events(view, 'record'))
        self.assertTrue(self.events(view, 'down'))
        self.assertTrue(self.events(view, 'enter'))
        copies = [event for event in view['presentation']['events'] if event['type'] == 'gain' and event.get('card', {}).get('copy')]
        self.assertTrue(self.events(view, 'record')[-1]['card']['name'])
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
