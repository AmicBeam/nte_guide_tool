"""2026-09-21 user tabletop rules; not Everness numerical claims."""
import json
import unittest
from copy import deepcopy

from tests import test_duel_v2_surplus_rework as fixtures
from app.modules.card_game.engine.duel_v2 import apply_action, observe, legal_actions
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.state import (
    hero, card_instance, damage, damage_immune, damage_limit, front_debuff, move_out,
)
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.combat import harmony_marks


class ZhenhongBuffTest(unittest.TestCase):
    game = fixtures.SurplusReworkTest.game
    hold = fixtures.SurplusReworkTest.hold
    play = fixtures.SurplusReworkTest.play

    def protect(self, state, side='a'):
        fire(EffectContext(state, side, 'zhenhong', {'side': side}), 'on_surplus', actor_only=True)

    def test_surplus_attack_stacks_and_expires_with_damage_limit_preserving_other_buffs(self):
        from app.modules.card_game.engine.duel_v2.state import add_atk_buff, attack_value
        s = self.game()
        h = hero(s, 'a', 'zhenhong')
        add_atk_buff(h, 3)
        base = attack_value(h, s, 'a')
        self.protect(s)
        self.protect(s)
        s['deferred_actions'] = []
        move_out(s, 'a', 'zhenhong')
        self.assertEqual(attack_value(h, s, 'a'), base + 2)
        s = clone_state(json.loads(json.dumps(s)))
        h = hero(s, 'a', 'zhenhong')
        begin_turn(s, 'b')
        self.assertEqual(attack_value(h, s, 'a'), base + 2)
        self.protect(s)  # Opponent-turn trigger has the same next-own-turn expiry.
        s['deferred_actions'] = []
        self.assertEqual(attack_value(h, s, 'a'), base + 3)
        opening, seq = capture_public_board(s), s['event_seq']
        s['_public_board'] = deepcopy(opening)
        begin_turn(s, 'a')
        self.assertEqual(attack_value(h, s, 'a'), base)
        self.assertIsNone(damage_limit(s, 'a', 'zhenhong'))
        self.assertNotIn('atk_buff_expiring', h['flags'])
        for viewer in ('a', 'b'):
            board = observe(s, viewer, include_previews=False)
            shown = next(h for h in board['sides']['a']['characters'] if h['id'] == 'zhenhong')
            self.assertEqual(shown['attack'], base)
        events = [e for e in observe(s, 'a')['presentation']['events'] if e['seq'] > seq]
        replay = replay_public_board(opening, events)
        shown = next(h for h in replay['sides']['a']['characters'] if h['id'] == 'zhenhong')
        self.assertEqual(shown['attack'], base)

    def test_typed_draws_are_unconditional_and_do_not_use_instant(self):
        for key, wanted in [('R02', 'I03'), ('R03', 'Z08'), ('R04', 'Z03')]:
            for surplus in (False, True):
                with self.subTest(key=key, surplus=surplus):
                    s = self.game()
                    team = s['sides']['a']
                    team.update(surplus=surplus, ap=1)
                    team['used']['instant'] = surplus  # An already used instant does not block these paid cards.
                    ids = ['I03', 'Z08', 'Z03']
                    team['deck'] = [card_instance(s, k, 'a') for k in ids]
                    before = deepcopy(team['deck'])
                    held = self.hold(s, key)
                    view = observe(s, 'a', include_previews=False)
                    shown = next(c for c in view['sides']['a']['hand'] if c['card_id'] == key)
                    self.assertFalse(shown.get('instant'))
                    self.assertFalse(shown['action_point_free'])
                    s = apply_action(s, 'a', {'type': 'play_card', 'card_id': held['instance_id']})
                    team = s['sides']['a']
                    self.assertEqual(team['ap'], 0)
                    self.assertEqual(bool(team['used'].get('instant')), surplus)
                    self.assertEqual([c['card_id'] for c in team['hand']], [wanted])
                    self.assertEqual(team['deck'], [c for c in before if c['card_id'] != wanted])
                    attacks = [e for e in s['events'] if e['type'] == 'attack']
                    self.assertEqual(attacks[-1]['amount'], {'R02': 1, 'R03': 2, 'R04': 3}[key])
                    if surplus:
                        held = self.hold(s, key)
                        s['sides']['a']['ap'] = 0
                        self.assertNotIn({'type': 'play_card', 'card_id': held['instance_id']}, legal_actions(s, 'a'))

    def test_typed_draw_missing_type_does_not_draw_wrong_card_or_lose(self):
        for surplus in (False, True):
            for ids in ([], ['Z03']):
                s = self.game()
                s['sides']['a'].update(surplus=surplus, deck=[card_instance(s, k, 'a') for k in ids])
                s = self.play(s, 'R02')
                self.assertEqual(s['phase'], 'playing')
                self.assertEqual(s['sides']['a']['hand'], [])
                self.assertEqual([c['card_id'] for c in s['sides']['a']['deck']], ids)

    def test_typed_draw_is_private_and_full_hand_discards(self):
        s = self.game()
        s['sides']['a'].update(surplus=True, deck=[card_instance(s, 'I03', 'a')])
        s = self.play(s, 'R02')
        other_view = observe(s, 'b')
        self.assertTrue(all(c.get('hidden') for c in other_view['sides']['a']['hand']))
        self.assertFalse(any(e.get('card', {}).get('card_id') == 'I03' for e in other_view['events']))
        s = self.game()
        s['sides']['a']['hand'] = [card_instance(s, 'R01', 'a') for _ in range(10)]
        s['sides']['a']['deck'] = [card_instance(s, 'I03', 'a')]
        EffectContext(s, 'a', 'zhenhong', {}).draw(1, card_type='tactic')
        self.assertEqual(len(s['sides']['a']['hand']), 10)
        self.assertEqual(s['sides']['a']['discard'][-1]['card_id'], 'I03')

    def test_rf01_draw_is_unconditional_and_empty_deck_stops_attack(self):
        for surplus in (False, True):
            s = self.game()
            s['sides']['a'].update(surplus=surplus, deck=[card_instance(s, 'I03', 'a')])
            s = self.play(s, 'RF01')
            self.assertEqual(s['sides']['a']['hand'][0]['card_id'], 'I03')
            attacks = [e for e in s['events'] if e['type'] == 'attack']
            self.assertEqual(attacks[-1]['amount'], 6)
            self.assertEqual(hero(s, 'a', 'zhenhong')['shield'], 4)
        s = self.game()
        s['sides']['a']['deck'] = []
        s = self.play(s, 'RF01')
        self.assertEqual(s['winner'], 'b')
        self.assertFalse(any(e['type'] == 'attack' for e in s['events']))

    def test_limit_caps_each_damage_after_stain_and_resistance_before_shields(self):
        for kind in ('combat', 'damage', 'genesis', 'followup', 'burn', 'star', 'nightmare'):
            with self.subTest(kind=kind):
                s = self.game(team=('zhenhong', 'haiyue', 'zero', 'jiuyuan'))
                self.protect(s)
                h = hero(s, 'a', 'zhenhong')
                h['shield'] = 3
                s['sides']['a']['front'] = 'zhenhong'
                front_debuff(s, 'a')['stain'] = {'by': 'b', 'expires_side': 'a'}
                h['next_damage_resistance'] = {'魂': -2}
                self.assertEqual(damage(s, 'a', 'zhenhong', 50, kind=kind, source='b:haiyue'), 0)
                self.assertEqual((h['hp'], h['shield']), (6, 2))
                self.assertEqual(s['events'][-1]['amount'], 1)
                self.assertEqual(h['next_damage_resistance'], {})
                h['shield'] = 0
                self.assertEqual(damage(s, 'a', 'zhenhong', 50, kind=kind, source='b:haiyue'), 1)
                self.assertEqual(h['hp'], 5)
                self.assertEqual(damage(s, 'a', 'zhenhong', 50, source='a:zero', bypass=True), 1)
                self.assertEqual(h['hp'], 4)
                move_out(s, 'a', 'zhenhong')
                self.assertEqual(damage_limit(s, 'a', 'zhenhong'), 1)
                self.assertFalse(damage_immune(s, 'a', 'zhenhong'))
                self.assertEqual(damage(s, 'a', 'zero', 2), 2)

    def test_limit_keeps_zero_damage_zero_and_support_counter_immune(self):
        s = self.game(team=('zhenhong', 'haiyue', 'zero', 'jiuyuan'))
        self.protect(s)
        h = hero(s, 'a', 'zhenhong')
        h['next_damage_resistance'] = {'魂': -2}
        self.assertEqual(damage(s, 'a', 'zhenhong', 0, source='b:haiyue'), 0)
        self.assertEqual(h['next_damage_resistance'], {'魂': -2})
        s['sides']['a']['front'] = 'zhenhong'
        front_debuff(s, 'a')['stain'] = {'by': 'b', 'expires_side': 'a'}
        self.assertEqual(damage(s, 'a', 'zhenhong', 0, kind='combat',
                                source='b:haiyue', counter_immunity='support'), 0)
        self.assertEqual(h['hp'], 6)
        self.assertEqual(h['next_damage_resistance'], {'魂': -2})
        self.assertEqual(s['events'][-1]['counter_immunity'], 'support')

    def test_limit_expires_before_own_start_damage_but_not_enemy_start(self):
        s = self.game()
        self.protect(s)
        s['deferred_actions'] = []
        s['sides']['a']['front'] = 'zhenhong'
        hero(s, 'b', 'yi').update(shape='I07', beast_fangs=6, awakened=True, ultimate_turns=2)
        begin_turn(s, 'b')
        self.assertEqual(damage_limit(s, 'a', 'zhenhong'), 1)
        self.assertEqual(hero(s, 'a', 'zhenhong')['hp'], 5)
        begin_turn(s, 'a')
        self.assertIsNone(damage_limit(s, 'a', 'zhenhong'))
        self.assertEqual(hero(s, 'a', 'zhenhong')['hp'], 3)
        # An opponent-turn trigger expires at the immediately following own start.
        s = self.game()
        begin_turn(s, 'b')
        self.protect(s)
        s['deferred_actions'] = []
        begin_turn(s, 'a')
        self.assertIsNone(damage_limit(s, 'a', 'zhenhong'))

    def test_limited_front_preserves_uncapped_penetration_and_death_clears_effect(self):
        from app.modules.card_game.engine.duel_v2.state import knockdowns
        s = self.game(team=('zhenhong', 'bohe', 'zero', 'jiuyuan'))
        self.protect(s, 'b')
        s['deferred_actions'] = []
        s['sides']['b']['front'] = 'zhenhong'
        hero(s, 'a', 'bohe')['base_attack'] = 20
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'bohe'})
        self.assertEqual(s['sides']['b']['hp'], 86)
        self.assertEqual(hero(s, 'b', 'zhenhong')['hp'], 5)
        # A lethal capped hit still uses the pre-hit life for overkill.
        s = self.game(team=('zhenhong', 'bohe', 'zero', 'jiuyuan'))
        self.protect(s, 'b')
        s['deferred_actions'] = []
        s['sides']['b']['front'] = 'zhenhong'
        hero(s, 'a', 'bohe')['base_attack'] = 20
        hero(s, 'b', 'zhenhong')['hp'] = 1
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'bohe'})
        self.assertEqual(s['sides']['b']['hp'], 81)
        self.assertEqual(hero(s, 'b', 'zhenhong')['down_turns'], 3)
        self.assertIsNone(damage_limit(s, 'b', 'zhenhong'))
        hero(s, 'b', 'zhenhong')['hp'] = 0
        knockdowns(s)
        self.assertNotIn('damage_immunity_until', hero(s, 'b', 'zhenhong')['flags'])
        self.assertNotIn('atk_buff_expiring', hero(s, 'b', 'zhenhong')['flags'])

    def test_eight_attack_vs_five_life_limit_preview_and_replay(self):
        for shield, player_shield, expected_overflow in [(0, 0, 3), (2, 0, 1), (4, 0, 0), (0, 2, 3)]:
            with self.subTest(shield=shield, player_shield=player_shield):
                s = self.game(team=('zhenhong', 'bohe', 'zero', 'jiuyuan'))
                self.protect(s, 'b')
                s['deferred_actions'] = []
                s['sides']['b'].update(front='zhenhong', hp=30, shield=player_shield)
                h = hero(s, 'b', 'zhenhong')
                h.update(hp=5, shield=shield)
                hero(s, 'a', 'bohe')['base_attack'] = 8
                opening, seq = capture_public_board(s), s['event_seq']
                s['_public_board'] = deepcopy(opening)
                action = {'type': 'attack', 'character_id': 'bohe'}
                expected_loss = max(0, expected_overflow-player_shield)
                preview = next(e['preview'] for e in observe(s, 'a')['legal_actions'] if e['action'] == action)
                self.assertEqual(preview['对手玩家生命变化'], -expected_loss)
                s = apply_action(s, 'a', action)
                h = hero(s, 'b', 'zhenhong')
                self.assertEqual(h['hp'], 5 if shield else 4)
                self.assertEqual(h['shield'], max(0, shield-1))
                self.assertEqual(s['sides']['b']['hp'], 30-expected_loss)
                hits = [e for e in s['events'] if e['seq'] > seq and e['type'] == 'penetration']
                self.assertEqual([e['amount'] for e in hits], [expected_overflow] if expected_overflow else [])
                events = [e for e in observe(s, 'a')['presentation']['events'] if e['seq'] > seq]
                replay = replay_public_board(opening, events)
                shown = next(h for h in replay['sides']['b']['characters'] if h['id'] == 'zhenhong')
                self.assertEqual(shown['hp'], h['hp'])
                self.assertEqual(replay['sides']['b']['hp'], 30-expected_loss)

    def test_ignore_shield_penetrates_before_front_limit(self):
        s = self.game(team=('zhenhong', 'bohe', 'zero', 'jiuyuan'))
        self.protect(s, 'b')
        s['deferred_actions'] = []
        s['sides']['b']['front'] = 'zhenhong'
        hero(s, 'b', 'zhenhong').update(hp=5, shield=4)
        hero(s, 'a', 'bohe')['base_attack'] = 6  # M03 adds 2.
        s = self.play(s, 'M03')
        self.assertEqual(hero(s, 'b', 'zhenhong')['hp'], 4)
        self.assertEqual(hero(s, 'b', 'zhenhong')['shield'], 4)
        self.assertEqual(s['sides']['b']['hp'], 97)

    def test_limit_snapshot_live_and_replay_no_external_duration_marker(self):
        s = self.game()
        opening, seq = capture_public_board(s), s['event_seq']
        self.protect(s)
        s = clone_state(json.loads(json.dumps(s)))
        self.assertEqual(damage_limit(s, 'a', 'zhenhong'), 1)
        for side in ('a', 'b'):
            h = next(h for h in observe(s, side, include_previews=False)['sides']['a']['characters'] if h['id'] == 'zhenhong')
            self.assertFalse(h['damage_immune'])
            self.assertEqual(h['damage_limit'], 1)
            self.assertFalse(any('免' in m['name'] for m in h['effect_markers']))
        events = [e for e in observe(s, 'a')['presentation']['events'] if e['seq'] > seq]
        h = next(h for h in replay_public_board(opening, events)['sides']['a']['characters'] if h['id'] == 'zhenhong')
        self.assertFalse(h['damage_immune'])
        self.assertEqual(h['damage_limit'], 1)

    def test_ultimate_moves_and_harmonizes_without_attack_or_combat_resources(self):
        s = self.game()
        s['sides']['a']['front'] = 'jiuyuan'
        hero(s, 'a', 'jiuyuan')['harmony'] = 2
        hero(s, 'a', 'zhenhong')['energy'] = 5
        before = s['sides']['a']['ap']
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zhenhong'})
        self.assertEqual(s['sides']['a']['front'], 'zhenhong')
        self.assertEqual(hero(s, 'a', 'jiuyuan')['harmony'], 0)
        self.assertEqual(hero(s, 'a', 'zhenhong')['harmony'], 1)
        self.assertTrue(s['sides']['a']['normal_attack_available'])
        self.assertEqual(s['sides']['a']['ap'], before)
        self.assertEqual(s['sides']['b']['hp'], 98)
        self.assertFalse(any(e['type'] == 'attack' for e in s['events']))
        self.assertTrue(all(h['energy'] == 0 for h in s['sides']['a']['characters'].values()))

    def test_ultimate_already_front_no_reentry_and_surplus_entry_drains_extra_attack(self):
        s = self.game()
        s['sides']['a']['front'] = 'zhenhong'
        hero(s, 'a', 'zhenhong')['energy'] = 5
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zhenhong'})
        self.assertFalse(any(e['type'] in ('enter', 'attack', 'harmony') for e in s['events']))
        s = self.game()
        s['sides']['a']['front'] = 'jiuyuan'
        hero(s, 'a', 'jiuyuan')['harmony'] = 2
        hero(s, 'a', 'zhenhong')['energy'] = 5
        front_debuff(s, 'b')['delay'] = {'left': 2, 'by': 'a', 'tick_side': 'a'}
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zhenhong'})
        self.assertEqual(hero(s, 'a', 'zhenhong')['extra_attacks'], 1)
        self.assertEqual(hero(s, 'a', 'zhenhong')['energy'], 0)
        self.assertEqual(damage_limit(s, 'a', 'zhenhong'), 1)
        self.assertFalse(s.get('deferred_actions'))

    def test_movement_checks_owner_turn_and_never_attacks(self):
        for active in ('a', 'b'):
            s = self.game()
            s['active_side'] = active
            s['sides']['a']['front'] = 'jiuyuan'
            hero(s, 'a', 'jiuyuan')['harmony'] = 2
            self.assertEqual(harmony_marks(s, 'a')['available'], active == 'a')
            EffectContext(s, 'a', 'zhenhong', {'side': 'a'}).enter_front()
            self.assertEqual(s['sides']['a']['front'], 'zhenhong')
            self.assertEqual(hero(s, 'a', 'jiuyuan')['harmony'], 0 if active == 'a' else 2)
            self.assertFalse(any(e['type'] == 'attack' for e in s['events']))

    def test_opponent_response_enters_without_consuming_harmony(self):
        s = self.game(team=('nanali', 'zero', 'jiuyuan', 'iloy'))
        s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['harmony'] = 2
        self.hold(s, 'N02', 'b')
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'zero'})
        self.assertEqual(s['sides']['b']['front'], 'nanali')
        self.assertEqual(hero(s, 'b', 'zero')['harmony'], 2)
        self.assertEqual(hero(s, 'b', 'nanali')['energy'], 0)
        self.assertFalse(any(e['type'] == 'harmony' and e['side'] == 'b' for e in s['events']))

    def test_enemy_turn_end_active_response_does_not_harmonize(self):
        from app.modules.card_game.engine.duel_v2.flow import trigger_responses
        s = self.game(team=('haiyue', 'yi', 'zero', 'jiuyuan'))
        s['sides']['b']['last_front'] = 'yi'
        hero(s, 'b', 'yi')['harmony'] = 2
        self.hold(s, 'U01', 'b')
        trigger_responses(s, 'b', 'enemy_turn_end_empty')
        self.assertEqual(hero(s, 'b', 'yi')['harmony'], 2)
        self.assertFalse(any(e['type'] == 'harmony' and e['side'] == 'b' for e in s['events']))
        self.assertTrue(any(e['type'] == 'attack' and e['side'] == 'b' for e in s['events']))

    def test_typed_draw_precedes_entry_that_first_triggers_surplus(self):
        s = self.game()
        s['sides']['a']['front'] = 'jiuyuan'
        hero(s, 'a', 'jiuyuan')['harmony'] = 2
        front_debuff(s, 'b')['delay'] = {'left': 2, 'by': 'a', 'tick_side': 'a'}
        s['sides']['a'].update(ap=1, deck=[card_instance(s, 'I03', 'a')])
        s = self.play(s, 'R02')
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['used'].get('instant'))
        self.assertEqual([c['card_id'] for c in s['sides']['a']['hand']], ['I03'])
        self.assertTrue(s['sides']['a']['surplus'])


if __name__ == '__main__':
    unittest.main()
