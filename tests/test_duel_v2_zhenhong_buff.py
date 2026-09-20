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
    hero, card_instance, damage, damage_immune, front_debuff, move_out,
)
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.combat import harmony_marks


class ZhenhongBuffTest(unittest.TestCase):
    game = fixtures.SurplusReworkTest.game
    hold = fixtures.SurplusReworkTest.hold
    play = fixtures.SurplusReworkTest.play

    def protect(self, state, side='a'):
        fire(EffectContext(state, side, 'zhenhong', {'side': side}), 'on_surplus', actor_only=True)

    def test_conditional_draws_keep_type_order_and_share_instant(self):
        for key, wanted in [('R02', 'I03'), ('R03', 'Z08'), ('R04', 'Z03')]:
            for surplus in (False, True):
                with self.subTest(key=key, surplus=surplus):
                    s = self.game()
                    team = s['sides']['a']
                    team.update(surplus=surplus, ap=1)
                    ids = ['I03', 'Z08', 'Z03']
                    team['deck'] = [card_instance(s, k, 'a') for k in ids]
                    before = deepcopy(team['deck'])
                    held = self.hold(s, key)
                    view = observe(s, 'a', include_previews=False)
                    shown = next(c for c in view['sides']['a']['hand'] if c['card_id'] == key)
                    self.assertEqual(bool(shown.get('instant')), surplus)
                    self.assertFalse(shown['action_point_free'])
                    s = apply_action(s, 'a', {'type': 'play_card', 'card_id': held['instance_id']})
                    team = s['sides']['a']
                    self.assertEqual(team['ap'], 1 if surplus else 0)
                    self.assertEqual(bool(team['used'].get('instant')), surplus)
                    self.assertEqual([c['card_id'] for c in team['hand']], [wanted] if surplus else [])
                    self.assertEqual(team['deck'], [c for c in before if not surplus or c['card_id'] != wanted])
                    attacks = [e for e in s['events'] if e['type'] == 'attack']
                    self.assertEqual(attacks[-1]['amount'], {'R02': 2, 'R03': 3, 'R04': 4}[key])
                    if surplus:
                        held = self.hold(s, key)
                        s['sides']['a']['ap'] = 0
                        self.assertNotIn({'type': 'play_card', 'card_id': held['instance_id']}, legal_actions(s, 'a'))

    def test_typed_draw_missing_type_does_not_draw_wrong_card_or_lose(self):
        for ids in ([], ['Z03']):
            s = self.game()
            s['sides']['a'].update(surplus=True, deck=[card_instance(s, k, 'a') for k in ids])
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

    def test_immunity_blocks_all_damage_including_stain_and_preserves_shield(self):
        s = self.game(team=('zhenhong', 'haiyue', 'zero', 'jiuyuan'))
        self.protect(s)
        h = hero(s, 'a', 'zhenhong')
        h['shield'] = 3
        s['sides']['a']['front'] = 'zhenhong'
        front_debuff(s, 'a')['stain'] = {'by': 'b', 'expires_side': 'a'}
        h['next_damage_resistance'] = {'魂': -2}
        for kind in ('combat', 'damage', 'genesis', 'followup', 'burn', 'star', 'nightmare'):
            self.assertEqual(damage(s, 'a', 'zhenhong', 50, kind=kind, source='b:haiyue'), 0)
            self.assertEqual((h['hp'], h['shield']), (5, 3))
        self.assertEqual(h['next_damage_resistance'], {'魂': -2})
        self.assertEqual(damage(s, 'a', 'zhenhong', 50, source='a:zero', bypass=True), 0)
        move_out(s, 'a', 'zhenhong')
        self.assertTrue(damage_immune(s, 'a', 'zhenhong'))
        self.assertEqual(damage(s, 'a', 'zero', 1), 1)

    def test_immunity_expires_before_own_start_damage_but_not_enemy_start(self):
        s = self.game()
        self.protect(s)
        s['deferred_actions'] = []
        s['sides']['a']['front'] = 'zhenhong'
        front_debuff(s, 'a')['burn'] = {'left': 2, 'by': 'b'}
        begin_turn(s, 'b')
        self.assertTrue(damage_immune(s, 'a', 'zhenhong'))
        begin_turn(s, 'a')
        self.assertFalse(damage_immune(s, 'a', 'zhenhong'))
        self.assertEqual(hero(s, 'a', 'zhenhong')['hp'], 4)
        # An opponent-turn trigger expires at the immediately following own start.
        s = self.game()
        begin_turn(s, 'b')
        self.protect(s)
        s['deferred_actions'] = []
        begin_turn(s, 'a')
        self.assertFalse(damage_immune(s, 'a', 'zhenhong'))

    def test_immune_front_does_not_leak_penetration_and_death_clears_flag(self):
        from app.modules.card_game.engine.duel_v2.state import knockdowns
        s = self.game(team=('zhenhong', 'bohe', 'zero', 'jiuyuan'))
        self.protect(s, 'b')
        s['deferred_actions'] = []
        s['sides']['b']['front'] = 'zhenhong'
        hero(s, 'a', 'bohe')['base_attack'] = 20
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'bohe'})
        self.assertEqual(s['sides']['b']['hp'], 100)
        self.assertEqual(hero(s, 'b', 'zhenhong')['hp'], 5)
        hero(s, 'b', 'zhenhong')['hp'] = 0
        knockdowns(s)
        self.assertNotIn('damage_immunity_until', hero(s, 'b', 'zhenhong')['flags'])

    def test_immunity_snapshot_live_and_replay_no_external_duration_marker(self):
        s = self.game()
        opening, seq = capture_public_board(s), s['event_seq']
        self.protect(s)
        s = clone_state(json.loads(json.dumps(s)))
        self.assertTrue(damage_immune(s, 'a', 'zhenhong'))
        for side in ('a', 'b'):
            h = next(h for h in observe(s, side, include_previews=False)['sides']['a']['characters'] if h['id'] == 'zhenhong')
            self.assertTrue(h['damage_immune'])
            self.assertFalse(any('免' in m['name'] for m in h['effect_markers']))
        events = [e for e in observe(s, 'a')['presentation']['events'] if e['seq'] > seq]
        h = next(h for h in replay_public_board(opening, events)['sides']['a']['characters'] if h['id'] == 'zhenhong')
        self.assertTrue(h['damage_immune'])

    def test_ultimate_moves_and_harmonizes_without_attack_or_combat_resources(self):
        s = self.game()
        s['sides']['a']['front'] = 'jiuyuan'
        hero(s, 'a', 'jiuyuan')['harmony'] = 2
        hero(s, 'a', 'zhenhong')['energy'] = 5
        before = s['sides']['a']['ap']
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zhenhong'})
        self.assertEqual(s['sides']['a']['front'], 'zhenhong')
        self.assertEqual(hero(s, 'a', 'jiuyuan')['harmony'], 0)
        self.assertEqual(hero(s, 'a', 'zhenhong')['harmony'], 0)
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
        self.assertTrue(damage_immune(s, 'a', 'zhenhong'))
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

    def test_condition_is_not_retroactive_when_card_entry_first_triggers_surplus(self):
        s = self.game()
        s['sides']['a']['front'] = 'jiuyuan'
        hero(s, 'a', 'jiuyuan')['harmony'] = 2
        front_debuff(s, 'b')['delay'] = {'left': 2, 'by': 'a', 'tick_side': 'a'}
        s['sides']['a'].update(ap=1, deck=[card_instance(s, 'I03', 'a')])
        s = self.play(s, 'R02')
        self.assertEqual(s['sides']['a']['ap'], 0)
        self.assertFalse(s['sides']['a']['used'].get('instant'))
        self.assertEqual(s['sides']['a']['hand'], [])
        self.assertTrue(s['sides']['a']['surplus'])


if __name__ == '__main__':
    unittest.main()
