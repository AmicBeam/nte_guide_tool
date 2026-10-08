"""User-authored Dafutier rules, using formal actions and public replay patches."""
import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog, validate_deck
from app.modules.card_game.content.duel_v2.characters.common import revive_character
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.combat import resolve_harmony_effect
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.continuous import apply_burn
from app.modules.card_game.engine.duel_v2.flow import begin_turn
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.engine.duel_v2.state import (
    add_collapse, card_instance, collapse_max, collapsed, force_collapse, hero, knockdowns,
)


def game():
    ids = ['dafutier', 'canhong', 'haiyue', 'zero']
    cards = [cid for owner in ids for cid, card in CARDS.items()
             if card['character_id'] == owner and not card.get('derived')]
    deck = validate_deck(dict(id='dafutier-test', name='达芙蒂尔测试', character_ids=ids, card_ids=cards))
    s = new_game(seed=73, first_side='a', skip_mulligan=True,
                 decks={'a': deck, 'b': deck}, escalation=False)
    for team in s['sides'].values():
        team.update(hand=[], ap=20, shield=0, hp=100)
        for h in team['characters'].values():
            h.update(hp=40, max_hp=40)
    return s


def give(s, cid, side='a'):
    card = card_instance(s, cid, side)
    s['sides'][side]['hand'].append(card)
    return dict(type='play_card', card_id=card['instance_id'])


def play(s, cid, target=None, side='a'):
    action = give(s, cid, side)
    if target:
        action['target_id'] = target
    return apply_action(s, side, action)


def context(s, side='a', cid='dafutier'):
    return EffectContext(s, side, cid, {'side': side})


def tokens(s, side='a'):
    return [c for c in s['sides'][side]['hand'] if c['card_id'] == 'FF01']


class DafutierTest(unittest.TestCase):
    def test_catalog_names_stats_and_public_access(self):
        self.assertEqual((CHARACTERS['dafutier']['attack'], CHARACTERS['dafutier']['max_hp']), (2, 5))
        self.assertEqual(CHARACTERS['dafutier']['energy_max'], 6)
        self.assertEqual([CARDS[f'F{i:02d}']['name'] for i in range(1, 9)],
                         ['目', '机械游行', '造物之众', '于隐匿中', '完美真相', '湖心花', '见此终幕', '漆黑青春妄想'])
        self.assertIn('dafutier', [c['id'] for c in get_catalog(include_test_characters=False)['characters']])
        self.assertTrue(CARDS['FF01']['derived'])
        self.assertEqual(CARDS['F02']['hp'], 6)
        self.assertEqual((CARDS['F08']['attack'], CARDS['F08']['hp']), (3, 6))

    def test_insight_once_each_global_turn_and_only_friendly_enemy_collapse(self):
        s = game()
        for cid in ('zero', 'haiyue'):
            force_collapse(s, 'b', cid, by='a')
        self.assertEqual(len(tokens(s)), 1)
        force_collapse(s, 'a', 'zero', by='a')
        self.assertEqual(len(tokens(s)), 1)
        begin_turn(s, 'b')
        force_collapse(s, 'b', 'canhong', by='a')
        self.assertEqual(len(tokens(s)), 2)

    def test_natural_and_forced_collapse_share_trigger_but_refresh_does_not(self):
        s = game()
        add_collapse(s, 'b', 'zero', 5, by='a')
        force_collapse(s, 'b', 'zero', by='a')
        add_collapse(s, 'b', 'zero', 5, by='a')
        self.assertEqual(len(tokens(s)), 1)
        self.assertEqual(sum(e['text'] == '零进入倾陷。' for e in s['events']), 1)

    def test_passive_down_and_revival_does_not_reset_turn_limit(self):
        s = game()
        force_collapse(s, 'b', 'zero', by='a')
        hero(s, 'a', 'dafutier')['hp'] = 0
        knockdowns(s)
        force_collapse(s, 'b', 'haiyue', by='a')
        revive_character(context(s), 'a', 'dafutier')
        force_collapse(s, 'b', 'canhong', by='a')
        self.assertEqual(len(tokens(s)), 1)

    def test_both_dissonance_paths_stack_twice_and_survive_down_json_revival(self):
        s = game()
        s['sides']['b']['front'] = 'zero'
        apply_burn(s, 'b', by='a', source_cid='dafutier')
        resolve_harmony_effect(context(s), '黯星')
        self.assertEqual(collapse_max(hero(s, 'b', 'zero')), 4)
        apply_burn(s, 'b', by='a', source_cid='dafutier')
        resolve_harmony_effect(context(s), '黯星')
        self.assertEqual(collapse_max(hero(s, 'b', 'zero')), 3)
        self.assertEqual(hero(s, 'a', 'dafutier')['dissonance_layers'], 2)
        hero(s, 'b', 'zero')['hp'] = 0
        knockdowns(s)
        s = json.loads(json.dumps(s))
        revive_character(context(s), 'b', 'zero')
        self.assertEqual(collapse_max(hero(s, 'b', 'zero')), 3)
        self.assertEqual(collapse_max(hero(s, 'a', 'zero')), 5)

    def test_dissonance_without_front_reduces_downed_targets_too(self):
        s = game()
        hero(s, 'b', 'zero')['hp'] = 0
        knockdowns(s)
        resolve_harmony_effect(context(s), '黯星')
        apply_burn(s, 'b', by='a', source_cid='dafutier')
        self.assertEqual(collapse_max(hero(s, 'b', 'zero')), 4)
        self.assertEqual(collapse_max(hero(s, 'b', 'haiyue')), 4)

    def test_lowering_limit_immediately_collapses_existing_full_meter(self):
        s = game()
        hero(s, 'b', 'haiyue')['flags']['collapse_count'] = 4
        resolve_harmony_effect(context(s), '黯星')
        apply_burn(s, 'b', by='a', source_cid='dafutier')
        self.assertTrue(collapsed(hero(s, 'b', 'haiyue')))
        self.assertEqual(len(tokens(s)), 1)

    def test_creations_recovers_one_deck_then_discard_per_collapse(self):
        s = game()
        a = card_instance(s, 'F03', 'a'); b = card_instance(s, 'F03', 'a')
        s['sides']['a']['deck'] = [a]
        s['sides']['a']['discard'] = [b]
        force_collapse(s, 'b', 'zero', by='a')
        self.assertIn(a['instance_id'], [c['instance_id'] for c in s['sides']['a']['hand']])
        self.assertEqual(len(s['sides']['a']['discard']), 1)
        force_collapse(s, 'b', 'haiyue', by='a')
        self.assertIn(b['instance_id'], [c['instance_id'] for c in s['sides']['a']['hand']])
        self.assertEqual(len(tokens(s)), 1)

    def test_creations_cannot_recover_current_resolving_instance(self):
        s = game(); s['sides']['a']['deck'] = []; s['sides']['a']['discard'] = []
        add_collapse(s, 'b', 'zero', 4, by='a')
        s = play(s, 'F03', 'b:zero')
        self.assertEqual([c['card_id'] for c in s['sides']['a']['hand']], ['FF01'])
        self.assertEqual([c['card_id'] for c in s['sides']['a']['discard']], ['F03'])

    def test_creations_keeps_ally_target_and_recovery_separate_from_insight(self):
        s = game(); s['sides']['a']['deck'] = [card_instance(s, 'F03', 'a')]
        add_collapse(s, 'a', 'zero', 4, by='a')
        s = play(s, 'F03', 'a:zero')
        self.assertTrue(collapsed(hero(s, 'a', 'zero')))
        self.assertEqual(len(tokens(s)), 0)
        self.assertEqual([c['card_id'] for c in s['sides']['a']['hand']], ['F03'])

    def test_user_october9_creations_adds_two_and_collapses_at_five(self):
        for side in ('a', 'b'):
            with self.subTest(side=side):
                s = play(game(), 'F03', f'{side}:zero')
                self.assertEqual(hero(s, side, 'zero')['flags']['collapse_count'], 2)
                self.assertFalse(collapsed(hero(s, side, 'zero')))
                add_collapse(s, side, 'zero', 1, by='a')
                s = play(s, 'F03', f'{side}:zero')
                self.assertTrue(collapsed(hero(s, side, 'zero')))
                self.assertEqual(len(tokens(s)), 1 if side == 'b' else 0)

    def test_insight_is_repeatably_free_without_spending_instant_opportunity(self):
        for instant_spent in (False, True):
            with self.subTest(instant_spent=instant_spent):
                s = game(); force_collapse(s, 'b', 'zero', by='a')
                give(s, 'FF01')
                s['sides']['a']['ap'] = 0
                s['sides']['a']['used']['instant'] = instant_spent
                for _ in range(2):
                    insight = tokens(s)[0]
                    actions = [a for a in legal_actions(s, 'a') if a.get('card_id') == insight['instance_id']]
                    self.assertEqual([a['target_id'] for a in actions], ['b:zero'])
                    shown = next(c for c in observe(s, 'a', include_previews=False)['sides']['a']['hand']
                                 if c['instance_id'] == insight['instance_id'])
                    self.assertTrue(shown['action_point_free'])
                    self.assertTrue(shown['description'].startswith('不消耗行动力。'))
                    s = apply_action(s, 'a', actions[0])
                    self.assertEqual(s['sides']['a']['ap'], 0)
                    self.assertEqual(s['sides']['a']['used']['instant'], instant_spent)
                self.assertEqual(hero(s, 'b', 'zero')['hp'], 36)
                self.assertEqual(s['sides']['a']['insight_played'], 2)

    def test_lake_manual_and_auto_response_targets_attacker_including_own_front(self):
        s = play(game(), 'F06', 'b:haiyue')
        self.assertEqual(hero(s, 'b', 'haiyue')['flags']['collapse_count'], 4)
        s = game(); s['sides']['b']['front'] = 'dafutier'
        give(s, 'F06', 'b')
        s = apply_action(s, 'a', dict(type='attack', character_id='zero'))
        self.assertEqual(hero(s, 'a', 'zero')['flags']['collapse_count'], 4)
        self.assertEqual(s['sides']['b']['ap'], 19)
        self.assertFalse(any(c['card_id'] == 'F06' for c in s['sides']['b']['hand']))

    def test_lake_cannot_respond_without_payment_or_while_collapsed(self):
        for disabled in ('payment', 'collapse'):
            s = game(); s['sides']['b']['front'] = 'zero'; give(s, 'F06', 'b')
            if disabled == 'payment': s['sides']['b']['ap'] = 0
            else: force_collapse(s, 'b', 'dafutier', by='a')
            s = apply_action(s, 'a', dict(type='attack', character_id='zero'))
            self.assertEqual(hero(s, 'a', 'zero')['flags'].get('collapse_count', 0), 0)

    def test_black_weapon_punishes_attacks_even_from_bench(self):
        s = play(game(), 'F08')
        s = apply_action(s, 'a', dict(type='end_turn'))
        s = apply_action(s, 'b', dict(type='attack', character_id='zero'))
        self.assertEqual(hero(s, 'b', 'zero')['flags']['collapse_count'], 3)
        self.assertEqual(hero(s, 'a', 'dafutier')['hp'], 6)

    def test_black_weapon_triggers_on_ranged_sortie_but_not_counter(self):
        s = play(game(), 'F08')
        s['sides']['a']['front'] = 'zero'
        s = apply_action(s, 'a', dict(type='end_turn'))
        hero(s, 'b', 'haiyue')['energy'] = 6
        s = apply_action(s, 'b', dict(type='ultimate', character_id='haiyue'))
        s = apply_action(s, 'b', dict(type='attack', character_id='haiyue'))
        self.assertEqual(hero(s, 'b', 'haiyue')['flags']['collapse_count'], 3)
        self.assertIsNone(s['sides']['b']['front'])
        self.assertEqual(hero(s, 'a', 'zero')['hp'], 37)

        s = play(game(), 'F08')
        s['sides']['b']['front'] = 'zero'
        s = apply_action(s, 'a', dict(type='attack', character_id='canhong'))
        self.assertEqual(hero(s, 'b', 'zero')['flags'].get('collapse_count', 0), 0)
        self.assertEqual(hero(s, 'a', 'canhong')['hp'], 38)

    def test_black_weapon_and_lake_stack_after_completed_sortie_damage(self):
        s = play(game(), 'F08')
        s['sides']['a']['front'] = 'zero'
        give(s, 'F06')
        s = apply_action(s, 'a', dict(type='end_turn'))
        s = apply_action(s, 'b', dict(type='attack', character_id='zero'))
        self.assertTrue(collapsed(hero(s, 'b', 'zero')))
        self.assertEqual(hero(s, 'a', 'zero')['hp'], 38)
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 38)
        self.assertTrue(any(e['type'] == 'attack' and e.get('actor') == 'b:zero'
                             for e in s['events']))
        self.assertEqual(len(tokens(s)), 1)
        self.assertEqual(hero(s, 'b', 'zero')['harmony'], 1)
        self.assertEqual(hero(s, 'b', 'zero')['energy'], 2)
        events = s['events']
        attack_index = next(i for i, e in enumerate(events) if e['type'] == 'attack' and e.get('actor') == 'b:zero')
        collapse_index = next(i for i, e in enumerate(events) if e['text'] == '零进入倾陷。')
        self.assertLess(attack_index, collapse_index)

    def test_black_weapon_waits_for_all_hits_and_triggers_once_per_sortie(self):
        from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
        s = play(game(), 'F08')
        s['sides']['a']['front'] = 'zero'
        s = apply_action(s, 'a', dict(type='end_turn'))
        cursor = s['event_seq']
        c = operation(s, 'b', 'zero')
        c.sortie(hits=2)
        finish_operation(s)
        events = [e for e in s['events'] if e['seq'] > cursor]
        attacks = [e for e in events if e['type'] == 'attack' and e.get('actor') == 'b:zero']
        self.assertEqual(len(attacks), 2)
        self.assertEqual(hero(s, 'a', 'zero')['hp'], 36)
        self.assertEqual(hero(s, 'b', 'zero')['flags']['collapse_count'], 3)
        collapse = next(e for e in events if e['type'] == 'collapse')
        self.assertGreater(collapse['seq'], attacks[-1]['seq'])

    def test_black_weapon_triggers_after_additional_attack_damage(self):
        from app.modules.card_game.engine.duel_v2.combat import additional_attack
        s = play(game(), 'F08')
        s['sides']['a']['front'] = 'zero'
        hero(s, 'b', 'zero')['flags']['collapse_count'] = 3
        cursor = s['event_seq']
        additional_attack(context(s, 'b', 'zero'), '光')
        events = [e for e in s['events'] if e['seq'] > cursor]
        self.assertEqual(hero(s, 'a', 'zero')['hp'], 38)
        self.assertTrue(collapsed(hero(s, 'b', 'zero')))
        self.assertLess(next(e['seq'] for e in events if e['type'] == 'combat'),
                        next(e['seq'] for e in events if e['text'] == '零进入倾陷。'))

    def test_black_weapon_does_not_trigger_for_sortie_interrupted_before_damage(self):
        s = play(game(), 'F08')
        s['sides']['a']['front'] = 'zero'
        give(s, 'F06')
        hero(s, 'b', 'zero')['flags']['collapse_count'] = 1
        s = apply_action(s, 'a', dict(type='end_turn'))
        cursor = s['event_seq']
        s = apply_action(s, 'b', dict(type='attack', character_id='zero'))
        events = [e for e in s['events'] if e['seq'] > cursor]
        self.assertEqual(hero(s, 'a', 'zero')['hp'], 40)
        self.assertTrue(collapsed(hero(s, 'b', 'zero')))
        self.assertFalse(any(e['type'] == 'attack' for e in events))
        self.assertEqual([e['amount'] for e in events if e['type'] == 'collapse' and 'amount' in e], [4])

    def test_black_weapon_cannot_trigger_if_holder_dies_during_sortie(self):
        s = play(game(), 'F08')
        s['sides']['a']['front'] = 'dafutier'
        s = apply_action(s, 'a', dict(type='end_turn'))
        hero(s, 'b', 'zero')['base_attack'] = 10
        s = apply_action(s, 'b', dict(type='attack', character_id='zero'))
        self.assertEqual(hero(s, 'a', 'dafutier')['hp'], 0)
        self.assertEqual(hero(s, 'b', 'zero')['flags'].get('collapse_count', 0), 0)

    def test_eye_counts_current_entry_support_and_caps_bonus_at_two(self):
        for previous, attack, meter in ((0, 4, 2), (2, 5, 3)):
            s = game(); s['sides']['a']['front'] = 'canhong'; s['sides']['b']['front'] = 'zero'
            s['sides']['a'].update(support_turn=s['turn'], support_count=previous)
            s = play(s, 'F01')
            hit = next(e for e in reversed(s['events']) if e['type'] == 'attack')
            self.assertEqual(hit['attack'], attack)
            self.assertEqual(hero(s, 'b', 'zero')['flags']['collapse_count'], meter)
            self.assertEqual(hero(s, 'a', 'dafutier')['shield'], 1)

    def test_eye_without_support_has_base_frame_and_no_extra_meter(self):
        s = game(); s['sides']['b']['front'] = 'zero'
        s = play(s, 'F01')
        hit = next(e for e in reversed(s['events']) if e['type'] == 'attack')
        self.assertEqual(hit['attack'], 3)
        self.assertEqual(hero(s, 'b', 'zero')['flags'].get('collapse_count', 0), 0)

    def test_summons_do_not_receive_eye_bonus_or_generate_insight(self):
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        s = game(); summon_front(s, 'b', name='靶子', attack=0, hp=20)
        target = s['sides']['b']['front']
        s['sides']['a'].update(support_turn=s['turn'], support_count=2)
        s = play(s, 'F01')
        self.assertEqual(hero(s, 'b', target)['flags'].get('collapse_count', 0), 0)
        force_collapse(s, 'b', target, by='a')
        self.assertEqual(len(tokens(s)), 0)

    def test_ultimate_after_support_retargets_after_knockdown_and_machine_refills(self):
        s = play(game(), 'F02')
        hero(s, 'a', 'dafutier')['energy'] = 6
        s = apply_action(s, 'a', dict(type='ultimate', character_id='dafutier'))
        s['sides']['a']['front'] = 'canhong'; s['sides']['b']['front'] = 'zero'
        hero(s, 'b', 'zero')['hp'] = 1
        s = play(s, 'F01')
        self.assertEqual(s['sides']['b']['hp'], 98)
        self.assertEqual(hero(s, 'a', 'dafutier')['energy'], 6)
        self.assertFalse(any(a.get('type') == 'ultimate' and a.get('character_id') == 'dafutier'
                             for a in legal_actions(s, 'a')))

    def test_ultimate_expires_before_second_following_turn_start(self):
        s = game(); hero(s, 'a', 'dafutier')['energy'] = 6
        s = apply_action(s, 'a', dict(type='ultimate', character_id='dafutier'))
        for side in ('b', 'a', 'b'):
            begin_turn(s, side)
        self.assertTrue(hero(s, 'a', 'dafutier')['awakened'])
        begin_turn(s, 'a')
        self.assertFalse(hero(s, 'a', 'dafutier')['awakened'])

    def test_borrowed_dafutier_support_triggers_ultimate_once_and_counts_once(self):
        from app.modules.card_game.engine.duel_v2.state import new_character
        s = game()
        s['sides']['a']['characters']['lingke'] = new_character('lingke')
        s['sides']['a']['order'].append('lingke')
        hero(s, 'a', 'dafutier')['energy'] = 6
        s = apply_action(s, 'a', dict(type='ultimate', character_id='dafutier'))
        s = play(s, 'K05', 'a:dafutier')
        self.assertEqual(s['sides']['a']['support_count'], 1)
        # The sync's dark-resistance -1 applies to its 3 damage and the
        # borrowed Dafutier's 2-damage ultimate followup: 4 + 3.
        self.assertEqual(s['sides']['b']['hp'], 93)
        hits = [e for e in s['events'] if '达芙蒂尔终结的援护追加伤害：' in e['text']]
        self.assertEqual(len(hits), 1)

    def test_permanent_limit_is_visible_in_live_and_replay_after_source_and_target_down(self):
        s = game()
        resolve_harmony_effect(context(s), '黯星')
        apply_burn(s, 'b', by='a', source_cid='dafutier')
        hero(s, 'a', 'dafutier')['hp'] = hero(s, 'b', 'zero')['hp'] = 0
        knockdowns(s)
        for view in (observe(s, 'a', include_previews=False), capture_public_board(s)):
            target = next(h for h in view['sides']['b']['characters'] if h['id'] == 'zero')
            self.assertEqual(target['collapse_max'], 4)
            marker = next(m for m in target['effect_markers'] if m['id'] == 'collapse_reduction')
            self.assertFalse(marker['clears_on_down'])
        revive_character(context(s), 'a', 'dafutier')
        self.assertEqual(hero(s, 'a', 'dafutier')['dissonance_layers'], 1)

    def test_insight_is_not_constructible(self):
        ids = ['dafutier', 'canhong', 'haiyue', 'zero']
        cards = [cid for owner in ids for cid, card in CARDS.items()
                 if card['character_id'] == owner and not card.get('derived')]
        cards[0] = 'FF01'
        with self.assertRaisesRegex(ValueError, '衍生牌不可构筑'):
            validate_deck(dict(id='bad', name='bad', character_ids=ids, card_ids=cards))

    def test_hidden_increases_all_enemy_meters_and_stacks_insight_damage(self):
        s = play(game(), 'F04'); s = play(s, 'F04')
        self.assertEqual(s['sides']['a']['insight_bonus'], 2)
        self.assertTrue(all(hero(s, 'b', cid)['flags']['collapse_count'] == 4 for cid in s['sides']['b']['order']))
        force_collapse(s, 'b', 'zero', by='a')
        s = play(s, 'FF01', 'b:zero')
        self.assertEqual(hero(s, 'b', 'zero')['hp'], 36)

    def test_truth_one_plus_played_insights_randomizes_each_hit_and_preserves_input(self):
        s = game(); s['sides']['a'].update(insight_bonus=1, insight_played=3)
        action = give(s, 'F05'); before = deepcopy(s)
        result = apply_action(s, 'a', action)
        hits = [e for e in result['events'] if e['type'] == 'damage' and e['text'].startswith('「完美真相」：')]
        self.assertEqual([e['amount'] for e in hits], [3] * 4)
        self.assertEqual(s, before)
        again = apply_action(json.loads(json.dumps(s)), 'a', action)
        self.assertEqual([(e['target'], e['amount']) for e in hits],
                         [(e['target'], e['amount']) for e in again['events']
                          if e['type'] == 'damage' and e['text'].startswith('「完美真相」：')])

    def test_finale_randomly_collapses_one_living_esper_per_side_without_damage(self):
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        s = game()
        for side in ('a', 'b'):
            hero(s, side, 'zero')['hp'] = 0
            knockdowns(s)
            summon_front(s, side, name='靶子', attack=0, hp=4)
            for h in s['sides'][side]['characters'].values():
                h['shield'] = 100
        action = give(s, 'F07')
        before = deepcopy(s)
        opening, seq = capture_public_board(s), s['event_seq']
        after = apply_action(s, 'a', action)
        again = apply_action(json.loads(json.dumps(s)), 'a', action)
        self.assertEqual(s, before)
        self.assertNotEqual(after['rng'], before['rng'])
        for side in ('a', 'b'):
            targets = [cid for cid, h in after['sides'][side]['characters'].items() if collapsed(h)]
            self.assertEqual(len(targets), 1)
            self.assertNotEqual(targets[0], 'zero')
            self.assertFalse(hero(after, side, targets[0]).get('summoned'))
            self.assertEqual(targets, [cid for cid, h in again['sides'][side]['characters'].items() if collapsed(h)])
            for cid, h in after['sides'][side]['characters'].items():
                self.assertEqual((h['hp'], h['shield']), (hero(before, side, cid)['hp'], 100))
        self.assertEqual(len(tokens(after)), 1)
        self.assertEqual(replay_public_board(opening, [e for e in after['events'] if e['seq'] > seq]),
                         capture_public_board(after))
        self.assertEqual(capture_public_board(simulate_action(s, 'a', action)), capture_public_board(after))

    def test_finale_still_collapses_friendly_side_when_no_enemy_is_alive(self):
        s = game()
        for cid in s['sides']['b']['order']:
            hero(s, 'b', cid)['hp'] = 0
        knockdowns(s)
        s = play(s, 'F07')
        self.assertEqual(sum(collapsed(h) for h in s['sides']['a']['characters'].values()), 1)
        self.assertFalse(any(collapsed(h) for h in s['sides']['b']['characters'].values()))

    def test_projection_dynamic_text_counters_permanent_markers_and_replay(self):
        s = game(); give(s, 'F01'); give(s, 'FF01'); give(s, 'F05')
        s['sides']['a'].update(insight_bonus=2, insight_played=3, support_turn=s['turn'], support_count=2)
        card_texts = [c['description'] for c in observe(s, 'a', include_previews=False)['sides']['a']['hand']]
        self.assertTrue(any('当前为 2' in text for text in card_texts))
        self.assertTrue(any('造成 4 点伤害' in text for text in card_texts))
        self.assertTrue(any('共 4 次' in text for text in card_texts))
        action = give(s, 'F04'); opening = capture_public_board(s); seq = s['event_seq']
        after = apply_action(s, 'a', action)
        self.assertEqual(replay_public_board(opening, [e for e in after['events'] if e['seq'] > seq]),
                         capture_public_board(after))
        sandbox = simulate_action(s, 'a', action)
        self.assertEqual(capture_public_board(sandbox), capture_public_board(after))


if __name__ == '__main__':
    unittest.main()
