"""持续伤害、浊燃、噩梦、倾陷时长，以及四名测试角色的关键结算。"""
import json
import unittest

from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.continuous import apply_burn, apply_dot, distribute
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.flow import begin_turn, end_turn
from app.modules.card_game.engine.duel_v2.state import attack_value, card_instance, force_collapse, front_debuff, hero, knockdowns


def _deck(characters, prefixes):
    cards = []
    for prefix in prefixes:
        cards.extend([f'{prefix}01', f'{prefix}01', f'{prefix}02', f'{prefix}02',
                      f'{prefix}03', f'{prefix}04', f'{prefix}05', f'{prefix}08'])
    return {'id': 'cast', 'name': '测试', 'character_ids': list(characters), 'card_ids': cards}


class ContinuousCastTest(unittest.TestCase):
    def game(self):
        deck = _deck(('nanali', 'anhunqu', 'canhong', 'zaowu'), ('N', 'A', 'C', 'S'))
        foe = _deck(('zero', 'iloy', 'jiuyuan', 'adler'), ('Z', 'Y', 'J', 'D'))
        return new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': foe})

    def play(self, state, card_id, *, side='a', target=None):
        state['sides'][side]['ap'] = 10
        state['sides'][side]['hand'] = [card_instance(state, card_id, side)]
        held = state['sides'][side]['hand'][0]
        action = {'type': 'play_card', 'card_id': held['instance_id']}
        if target:
            action['target_id'] = target
        return apply_action(state, side, action)

    def test_vinyl_discard_presents_only_the_actual_discarded_card_for_both_views_and_replay(self):
        from app.modules.card_game.engine.duel_v2.state import damage
        from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
        s = self.play(self.game(), 'S01')
        summon = s['sides']['a']['front']
        cards = [card_instance(s, key, 'b') for key in ('Z03', 'Y04')]
        s['sides']['b']['hand'] = cards
        opening, seq = capture_public_board(s), s['event_seq']
        s['_public_board'] = opening
        damage(s, 'a', summon, 1, source='a:zaowu')
        discarded = s['sides']['b']['discard'][-1]
        retained = s['sides']['b']['hand'][0]
        for viewer in ('a', 'b'):
            events = [e for e in observe(s, viewer)['presentation']['events'] if e['seq'] > seq]
            notice, = [e for e in events if e['type'] == 'effect' and e.get('present')]
            self.assertIn(discarded['name'], notice['text'])
            self.assertNotIn(retained['name'], notice['text'])
            self.assertEqual(notice['target'], 'b:player')
            self.assertEqual(len([e for e in events if e['type'] == 'discard']), 1)
            replay = replay_public_board(opening, events)
            self.assertEqual(replay['sides']['b']['hand_count'], 1)
            self.assertEqual(len(s['sides']['b']['discard']), 1)

    def test_vinyl_empty_hand_or_zero_damage_has_no_discard_notice(self):
        from app.modules.card_game.engine.duel_v2.state import damage
        for empty, amount in ((True, 1), (False, 0)):
            s = self.play(self.game(), 'S01')
            summon = s['sides']['a']['front']
            s['sides']['b']['hand'] = [] if empty else [card_instance(s, 'Z03', 'b')]
            seq = s['event_seq']
            damage(s, 'a', summon, amount, source='a:zaowu')
            self.assertFalse(any(e['type'] == 'effect' and e.get('present')
                                 for e in s['events'] if e['seq'] > seq))

    def test_s05_hits_allied_front_and_opposing_espers_without_requiring_gulang(self):
        from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        for front in (None, 'nanali', 'zaowu', 'gulang', 'other_summon'):
            with self.subTest(front=front):
                s = self.game()
                if front == 'gulang':
                    s = self.play(s, 'S01')
                elif front == 'other_summon':
                    summon_front(s, 'a', name='塔吉多', attack=2, hp=4)
                else:
                    s['sides']['a']['front'] = front
                own_front = s['sides']['a']['front']
                foe_summon = summon_front(s, 'b', name='对方召唤物', attack=2, hp=4)
                hero(s, 'b', 'jiuyuan').update(hp=0, down_turns=3)
                before = {(side, cid): h['hp'] for side in ('a', 'b')
                          for cid, h in s['sides'][side]['characters'].items()}
                s['sides']['a']['ap'] = 10
                held = card_instance(s, 'S05', 'a')
                s['sides']['a']['hand'] = [held]
                action = {'type': 'play_card', 'card_id': held['instance_id']}
                self.assertIn(action, legal_actions(s, 'a'))
                opening, seq = capture_public_board(s), s['event_seq']
                s = apply_action(s, 'a', action)
                replay = replay_public_board(opening, [e for e in s['events'] if e['seq'] > seq])
                for (side, cid), hp in before.items():
                    hit = (side == 'a' and cid == own_front or
                           side == 'b' and cid != foe_summon and hp > 0)
                    expected = hp - 2 if hit else hp
                    self.assertEqual(hero(s, side, cid)['hp'], expected)
                    shown = next(h for h in replay['sides'][side]['characters'] if h['id'] == cid)
                    self.assertEqual(shown['hp'], expected)
                self.assertEqual(s['sides']['a']['hp'], 30)
                self.assertEqual(s['sides']['b']['hp'], 30)
                if front == 'gulang':
                    self.assertEqual(len(s['sides']['b']['hand']), 4)
                if front != 'gulang':
                    s['sides']['a']['hand'] = [card_instance(s, 'S06', 'a')]
                    self.assertFalse(any(a['type'] == 'play_card' for a in legal_actions(s, 'a')))

    def test_s05_finishes_enemy_damage_after_its_own_front_falls(self):
        s = self.game()
        s['sides']['a']['front'] = 'zaowu'
        s['sides']['b']['front'] = 'zero'
        hero(s, 'a', 'zaowu')['hp'] = 1
        before = {cid: h['hp'] for cid, h in s['sides']['b']['characters'].items()}
        s = self.play(s, 'S05')
        self.assertEqual(hero(s, 'a', 'zaowu')['hp'], 0)
        self.assertIsNone(s['sides']['a']['front'])
        for cid, hp in before.items():
            self.assertEqual(hero(s, 'b', cid)['hp'], hp - 2)

    def test_gulang_details_describe_each_item_in_live_legacy_snapshot_and_replay(self):
        from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
        cases = (
            ('S01', '黑胶唱片', '随机弃置对方 1 张手牌'),
            ('S02', '好狗狗的项圈', '随机使 1 名对方异能者进入倾陷'),
            ('S03', '气味奇怪的液体', '随机对 1 名对方异能者造成等同于本次所受伤害的伤害'),
        )
        for card_id, item, effect in cases:
            with self.subTest(card_id=card_id):
                s = self.game()
                opening, seq = capture_public_board(s), s['event_seq']
                s = self.play(s, card_id)
                cid = s['sides']['a']['front']
                replay = replay_public_board(opening, [e for e in s['events'] if e['seq'] > seq])
                h = hero(s, 'a', cid)
                self.assertIn(effect, h['passive'])
                # Existing rooms stored only an opaque trigger sentence.
                h['passive'] = f'吞噬了{item}。受到伤害时触发。倒地或离开战斗区时消失。'
                s = json.loads(json.dumps(s))
                views = [observe(s, side, include_previews=False) for side in ('a', 'b')]
                views += [capture_public_board(s), replay]
                for view in views:
                    shown = next(h for h in view['sides']['a']['characters'] if h['id'] == cid)
                    self.assertIn(item, shown['passive'])
                    self.assertIn(effect, shown['passive'])
                    self.assertNotIn('受到伤害时触发', shown['passive'])

    def test_keyword_text_reaches_catalog_and_public_hand(self):
        from app.modules.card_game.content.duel_v2 import CARDS
        for card in CARDS.values():
            if card.get('instant'):
                self.assertIn('瞬发', card['description'], card['id'])
            if card.get('retain'):
                self.assertIn('可重复使用', card['description'], card['id'])
            if card['cost'] == 0 and not card.get('spend_all_ap'):
                self.assertIn('不消耗行动力', card['description'], card['id'])
        s = self.game()
        s['sides']['a']['hand'] = [card_instance(s, 'S04', 'a'), card_instance(s, 'A03', 'a')]
        descriptions = {c['card_id']: c['description'] for c in observe(s, 'a')['sides']['a']['hand']}
        self.assertIn('瞬发。可重复使用。', descriptions['S04'])
        self.assertTrue(descriptions['A03'].startswith('不消耗行动力。'))

    def test_tomato_adds_derived_card_to_hand_without_attacking(self):
        s = self.game()
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=5)
        before = sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        s = self.play(s, 'A04')
        self.assertTrue(any(c['card_id'] == 'AF01' for c in s['sides']['a']['hand']))
        self.assertIsNone(s['sides']['a']['front'])
        self.assertEqual(sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters']), before)

    def test_icecream_triggers_rose_and_removes_nightmare(self):
        s = self.play(self.game(), 'A08')
        for cid in ('zero', 'jiuyuan', 'adler'):
            hero(s, 'b', cid)['hp'] = 0
        hero(s, 'b', 'iloy').update(hp=5, max_hp=5, shield=0)
        hero(s, 'a', 'anhunqu')['hp'] = 1
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=2)
        s = self.play(s, 'A03')
        self.assertEqual((hero(s, 'b', 'iloy')['hp'], hero(s, 'b', 'iloy')['max_hp']), (2, 3))
        self.assertEqual(hero(s, 'a', 'anhunqu')['hp'], 3)
        self.assertNotIn('nightmare', front_debuff(s, 'b')['dots'])
        self.assertFalse(s['sides']['a']['used'].get('instant'))

    def test_nightmare_kill_no_longer_adds_player_damage_with_gilded_symphony(self):
        s = self.play(self.game(), 'A07')
        for cid in ('zero', 'jiuyuan', 'adler'):
            hero(s, 'b', cid)['hp'] = 0
        hero(s, 'b', 'iloy').update(hp=1, shield=0)
        s['sides']['b']['shield'] = 0
        s['rng'] = 0  # Both nightmare points hit the sole living esper.
        before = s['sides']['b']['hp']
        s = self.play(s, 'A03')
        self.assertEqual(hero(s, 'b', 'iloy')['hp'], 0)
        self.assertEqual(s['sides']['b']['hp'], before)
        self.assertFalse(any('「噩梦」消灭了对方异能者' in e['text'] for e in s['events']))

    def test_rose_still_heals_when_nightmare_kills_target(self):
        s = self.play(self.game(), 'A08')
        for cid in ('zero', 'jiuyuan', 'adler'):
            hero(s, 'b', cid)['hp'] = 0
        hero(s, 'b', 'iloy').update(hp=1, shield=0)
        hero(s, 'a', 'anhunqu')['hp'] = 1
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=2)
        s = self.play(s, 'A03')
        self.assertEqual(hero(s, 'a', 'anhunqu')['hp'], 3)

    def test_random_distribution_is_not_necessarily_continuous_damage(self):
        s = self.game()
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 1}
        apply_dot(s, 'b', 'etch', by='a', source_cid='canhong', stacks=5)
        before = sum(h['hp'] for h in s['sides']['b']['characters'].values())
        distribute(s, 'b', 5, source='a:nanali', by='a', continuous=False)
        self.assertEqual(before - sum(h['hp'] for h in s['sides']['b']['characters'].values()), 5)

    def test_downed_source_does_not_remove_nightmare_or_change_damage_source(self):
        s = self.game()
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=3)
        hero(s, 'a', 'anhunqu')['hp'] = 0
        knockdowns(s)
        start = s['event_seq']
        end_turn(s, 'a')
        hits = [e for e in s['events'] if e['seq'] > start and e['type'] == 'damage'
                and e.get('source') == 'a:anhunqu']
        self.assertEqual(sum(e['amount'] for e in hits), 3)

    def test_target_wording_controls_both_sides_and_opposing_player(self):
        s = self.game()
        for card_id, target in [('A02', 'a:canhong'), ('A05', 'b:player')]:
            with self.subTest(card=card_id):
                s['sides']['a']['hand'] = [card_instance(s, card_id, 'a')]
                held = s['sides']['a']['hand'][0]
                s['sides']['a']['ap'] = 10
                action = {'type': 'play_card', 'card_id': held['instance_id'], 'target_id': target}
                self.assertIn(action, legal_actions(s, 'a'))
                entry = next(e for e in observe(s, 'a')['legal_actions'] if e['action'] == action)
                self.assertEqual(entry['interaction']['kind'], 'target')
        before = s['sides']['b']['hp']
        s['sides']['b']['shield'] = 0
        s = self.play(s, 'A05', target='b:player')
        self.assertEqual(s['sides']['b']['hp'], before - 3)
        self.assertIsNone(s['sides']['a']['front'])

    def test_adler_full_team_shield_includes_player(self):
        s = self.game()
        end_turn(s, 'a')
        before = s['sides']['b']['shield']
        s = self.play(s, 'D02', side='b')
        self.assertEqual(s['sides']['b']['shield'], before + 3)
        self.assertTrue(all(h['shield'] >= 3 for h in s['sides']['b']['characters'].values()))

    def test_repeatable_card_returns_and_second_play_pays_action_point(self):
        s = self.play(self.game(), 'S01')
        summon = s['sides']['a']['front']
        s = self.play(s, 'S04', target=f'a:{summon}')
        held = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'S04')
        self.assertTrue(s['sides']['a']['used']['instant'])
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        summon = summon_front(s, 'a', name='测试召唤物', attack=0, hp=10)
        before = s['sides']['a']['ap']
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': held['instance_id'], 'target_id': f'a:{summon}'})
        self.assertEqual(s['sides']['a']['ap'], before - 1)
        self.assertTrue(any(c['instance_id'] == held['instance_id'] for c in s['sides']['a']['hand']))

    def test_mirage_lasts_three_friendly_turns_and_refreshes_after_battle_card(self):
        from app.modules.card_game.content.duel_v2.characters.canhong import _mirage
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        for refresh in (False, True):
            with self.subTest(refresh=refresh):
                s = self.play(self.game(), 'C04')
                self.assertIn('残虹进入幻境，持续 3 个己方回合。', [e['text'] for e in s['events']])
                s = json.loads(json.dumps(s))
                for step in range(1, 4):
                    begin_turn(s, 'b')
                    self.assertTrue(_mirage(EffectContext(s, 'a', 'canhong', {'side': 'a'})))
                    begin_turn(s, 'a')
                    self.assertEqual(_mirage(EffectContext(s, 'a', 'canhong', {'side': 'a'})),
                                     step < 3 or refresh)
                    if refresh and step == 1:
                        s = self.play(s, 'C04')
                if refresh:
                    begin_turn(s, 'b')
                    begin_turn(s, 'a')
                    self.assertFalse(_mirage(EffectContext(s, 'a', 'canhong', {'side': 'a'})))

    def test_mirage_dot_damage_grows_all_dots_without_splitting_animation(self):
        from app.modules.card_game.content.duel_v2.characters.canhong import _enter_mirage
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        s = self.game()
        _enter_mirage(EffectContext(s, 'a', 'canhong', {'side': 'a'}))
        apply_dot(s, 'b', 'etch', by='a', source_cid='canhong', stacks=5)
        apply_dot(s, 'b', 'venom', by='a', source_cid='canhong', stacks=5)
        for h in s['sides']['b']['characters'].values():
            h.update(hp=20, max_hp=20)
        start = s['event_seq']
        attack = attack_value(hero(s, 'a', 'canhong'), s, 'a')
        distribute(s, 'b', 8, source='a:canhong', by='a', reason='幻境持续伤害')
        events = [e for e in s['events'] if e['seq'] > start]
        hits = [e for e in events if e['type'] == 'damage']
        self.assertGreater(len(hits), 1)
        self.assertEqual(len(hits), len({e['target'] for e in hits}))
        self.assertEqual(len({e['group_id'] for e in events}), 1)
        dots = front_debuff(s, 'b')['dots']
        self.assertEqual(dots['etch']['stacks'], min(10, 5 + len(hits)))
        self.assertEqual(dots['venom']['stacks'], min(10, 5 + len(hits)))
        self.assertEqual(attack_value(hero(s, 'a', 'canhong'), s, 'a'), attack)

    def test_canhong_initial_harmony_is_visible_for_both_sides_before_mulligan(self):
        # The initial 2 harmony is a user-authored tabletop rule (2026-10-07).
        deck = _deck(('anhunqu', 'canhong', 'zaowu', 'adler'), ('A', 'C', 'S', 'D'))
        for first in ('a', 'b'):
            with self.subTest(first=first):
                s = new_game(seed=3, first_side=first, decks={'a': deck, 'b': deck})
                self.assertEqual(s['phase'], 'mulligan')
                for side in ('a', 'b'):
                    for cid in deck['character_ids']:
                        h = hero(s, side, cid)
                        self.assertEqual(h['harmony'], 2 if cid == 'canhong' else 0)
                        self.assertEqual(h['energy'], 0)
                    public = observe(s, side)['sides'][side]['characters']
                    self.assertEqual(next(h for h in public if h['id'] == 'canhong')['harmony'], 2)
                    opening = s['_public_board']['sides'][side]['characters']
                    self.assertEqual(next(h for h in opening if h['id'] == 'canhong')['harmony'], 2)

    def test_canhong_burn_can_stack_three_times_in_one_turn_without_attack_growth(self):
        s = self.game()
        h = hero(s, 'a', 'canhong')
        base = attack_value(h, s, 'a')
        for expected in (1, 2, 3, 3):
            apply_burn(s, 'b', by='a', source_cid='anhunqu')
            self.assertEqual(front_debuff(s, 'b')['burn']['stacks'], expected)
            self.assertEqual(attack_value(h, s, 'a'), base)

    def test_canhong_dots_do_not_increase_attack_on_either_players_turn(self):
        s = self.game()
        h = hero(s, 'a', 'canhong')
        base = attack_value(h, s, 'a')
        for side in ('a', 'b', 'a'):
            self.assertEqual(s['active_side'], side)
            apply_dot(s, 'b', 'etch', by='a', source_cid='canhong', stacks=1)
            apply_dot(s, 'b', 'venom', by='a', source_cid='canhong', stacks=1)
            self.assertEqual(attack_value(h, s, 'a'), base)
            end_turn(s, side)
        self.assertFalse(any('持续伤害，残虹的异能触发：攻击 +1' in e.get('text', '')
                             for e in s['events']))

    def test_canhong_initial_harmony_pays_for_first_replacement_and_does_not_refill(self):
        s = self.game()
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'canhong'})
        self.assertEqual(hero(s, 'a', 'canhong')['harmony'], 2)
        self.assertNotIn('burn', front_debuff(s, 'b'))  # No partner on first entry.
        s = self.play(s, 'AF01')
        self.assertEqual(hero(s, 'a', 'canhong')['harmony'], 0)
        self.assertEqual(front_debuff(s, 'b')['burn']['stacks'], 1)
        self.assertEqual(attack_value(hero(s, 'a', 'canhong'), s, 'a'), 2)
        end_turn(s, 'a')
        end_turn(s, 'b')
        self.assertEqual(hero(s, 'a', 'canhong')['harmony'], 0)
        s = clone_state(json.loads(json.dumps(s)))
        s = self.play(s, 'C08')
        h = hero(s, 'a', 'canhong')
        self.assertEqual(h['harmony'], 0)
        h['hp'] = 0
        knockdowns(s)
        s = self.play(s, 'C07')
        h = hero(s, 'a', 'canhong')
        self.assertEqual(h['harmony'], 0)
        attack = attack_value(h, s, 'a')
        apply_dot(s, 'b', 'venom', by='a', source_cid='canhong', stacks=1)
        self.assertEqual(attack_value(h, s, 'a'), attack)

    def test_canhong_existing_harmony_survives_json_and_revival(self):
        s = clone_state(json.loads(json.dumps(self.game())))
        h = hero(s, 'a', 'canhong')
        self.assertEqual(h['harmony'], 2)
        h['hp'] = 0
        knockdowns(s)
        self.assertEqual(h['harmony'], 2)
        s = self.play(s, 'C07')
        self.assertEqual(hero(s, 'a', 'canhong')['harmony'], 2)

    def test_copied_canhong_neither_grants_initial_harmony_nor_dot_attack(self):
        ours = _deck(('anhunqu', 'zero', 'zaowu', 'adler'), ('A', 'Z', 'S', 'D'))
        theirs = _deck(('nanali', 'canhong', 'iloy', 'jiuyuan'), ('N', 'C', 'Y', 'J'))
        s = new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': ours, 'b': theirs})
        s = self.play(s, 'A01', target='b:canhong')
        self.assertEqual(hero(s, 'a', 'anhunqu')['harmony'], 0)
        for kind in ('etch', 'venom'):
            apply_dot(s, 'b', kind, by='a', source_cid='anhunqu', stacks=1)
            for side, cid in (('a', 'anhunqu'), ('b', 'canhong')):
                self.assertEqual(attack_value(hero(s, side, cid), s, side), 2)

    def test_mimic_family_is_owned_and_permanently_grows_anhunqu(self):
        ours = _deck(('anhunqu', 'canhong', 'zaowu', 'adler'), ('A', 'C', 'S', 'D'))
        theirs = _deck(('nanali', 'zero', 'iloy', 'jiuyuan'), ('N', 'Z', 'Y', 'J'))
        s = new_game(seed=3, first_side='a', skip_mulligan=True, decks={'a': ours, 'b': theirs})
        s = self.play(s, 'A01', target='b:nanali')
        card = next(c for c in s['sides']['a']['hand'] if c['card_id'] == 'NF01')
        self.assertEqual(card['character_id'], 'anhunqu')
        shown = next(c for c in observe(s, 'a')['sides']['a']['hand'] if c['card_id'] == 'NF01')
        self.assertIn('安魂曲永久获得', shown['description'])
        self.assertNotIn('娜娜莉', shown['description'])
        s = apply_action(s, 'a', {'type': 'play_card', 'card_id': card['instance_id']})
        h = hero(s, 'a', 'anhunqu')
        self.assertEqual(h['permanent_growth'], {'attack': 1, 'max_hp': 1})
        self.assertEqual(hero(s, 'b', 'nanali')['permanent_growth'], {'attack': 0, 'max_hp': 0})
        s['sides']['b']['shield'] = 0
        s = apply_action(s, 'a', {'type': 'attack', 'character_id': 'anhunqu'})
        self.assertNotIn('nightmare', front_debuff(s, 'b').get('dots') or {})

    def test_mimic_does_not_replace_ultimate_or_own_weapon_turn_hook(self):
        s = self.play(self.game(), 'A07')
        h = hero(s, 'a', 'anhunqu')
        h['flags']['mimic'] = {'id': 'adler'}
        from app.modules.card_game.content.duel_v2.registry import fire
        from app.modules.card_game.engine.duel_v2.context import EffectContext
        fire(EffectContext(s, 'a', 'anhunqu', {'side':'a'}), 'on_turn_begin', actor_only=True)
        self.assertEqual(h['shield'], 2)
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 4)
        fire(EffectContext(s, 'a', 'anhunqu', {'side':'a'}), 'on_ultimate', actor_only=True)
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 7)

    def test_early_shift_has_no_attack_penalty_and_ignores_harmony_fullness(self):
        for attribute in ('魂', '咒'):
            for harmony in (0, 1, 2):
                with self.subTest(attribute=attribute, harmony=harmony):
                    s = self.game()
                    s['sides']['a']['front'] = 'canhong'
                    payer = hero(s, 'a', 'canhong')
                    payer.update(attribute=attribute, harmony=harmony)
                    s['sides']['b']['front'] = 'zero'
                    hero(s, 'b', 'zero').update(hp=20, max_hp=20, shield=0)
                    s = self.play(s, 'A06')
                    hit = next(e for e in reversed(s['events']) if e['type'] == 'attack')
                    self.assertEqual(hit['attack'], 2)
                    self.assertEqual(hit['counter'], 0)
                    self.assertEqual(hero(s, 'a', 'anhunqu')['hp'], 5)
                    self.assertEqual(hero(s, 'a', 'canhong')['harmony'], harmony)
                    self.assertEqual(s['sides']['a']['support_count'], 1)
                    zone = front_debuff(s, 'b')
                    self.assertIn('star' if attribute == '魂' else 'burn', zone)
                    self.assertNotIn('nightmare', zone.get('dots') or {})

    def test_gilded_symphony_adds_two_nightmare_on_equip_and_turn_start(self):
        s = self.play(self.game(), 'A07')
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 2)
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=3)
        begin_turn(s, 'b')
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 5)
        begin_turn(s, 'a')
        # The shared nightmare clock halves five before the weapon adds two.
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 4)
        self.assertIn('己方回合开始，「鎏金交响诗」施加 2 层「噩梦」。', [e['text'] for e in s['events']])

    def test_distribution_merges_one_number_per_character(self):
        s = self.game()
        for cid in s['sides']['b']['characters']:
            hero(s, 'b', cid)['hp'] = 10
        killed = distribute(s, 'b', 5, source='a:anhunqu', by='a', reason='测试')
        events = [e for e in s['events'] if e.get('text', '').startswith('测试')]
        self.assertEqual(len(events), len({e['target'] for e in events}))
        self.assertEqual(sum(e['amount'] for e in events), 5)
        self.assertTrue(all(e.get('group_id') == events[0]['group_id'] for e in events))
        self.assertEqual(killed, [])

    def test_canhong_auto_weapon_resolves_after_she_is_downed(self):
        s = self.game()
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 20}
        s['sides']['a']['hand'].append(card_instance(s, 'C07', 'a'))
        hero(s, 'a', 'canhong')['hp'] = 0
        knockdowns(s)
        revived = hero(s, 'a', 'canhong')
        self.assertEqual(revived['shape'], 'C07')
        self.assertGreater(revived['hp'], 0)
        self.assertEqual(revived['down_turns'], 0)

    def test_burn_lasts_two_applier_turn_ends(self):
        s = self.game()
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 2, 'source': 'a:canhong'}
        before = sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        end_turn(s, 'a')
        mid = sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        self.assertEqual(before - mid, 2)
        self.assertEqual(front_debuff(s, 'b')['burn']['left'], 1)
        end_turn(s, 'b')
        end_turn(s, 'a')
        self.assertNotIn('burn', front_debuff(s, 'b'))

    def test_nightmare_halves_on_the_next_own_turn(self):
        s = self.game()
        apply_dot(s, 'b', 'nightmare', by='a', source_cid='anhunqu', stacks=10)
        end_turn(s, 'a')
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 10)
        end_turn(s, 'b')
        self.assertEqual(front_debuff(s, 'b')['dots']['nightmare']['stacks'], 5)

    def test_etch_and_venom_keep_stacks_without_damage_even_in_old_snapshots(self):
        from app.modules.card_game.engine.duel_v2.continuous import kind_count, total_stacks, trigger_dot, tick_owner_end
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                s = self.game()
                for kind in ('etch', 'venom'):
                    apply_dot(s, 'b', kind, by='a', source_cid='canhong', stacks=10)
                    if legacy:
                        front_debuff(s, 'b')['dots'][kind]['mode'] = 'every5'
                s = json.loads(json.dumps(s))
                before = [(side, cid, h['hp'], h['shield']) for side in ('a', 'b')
                          for cid, h in s['sides'][side]['characters'].items()]
                rng, seq = s['rng'], s['event_seq']
                for kind in ('etch', 'venom'):
                    self.assertEqual(trigger_dot(s, 'b', kind, front_debuff(s, 'b')['dots'][kind]), [])
                tick_owner_end(s, 'a')
                for side, cid, hp, shield in before:
                    self.assertEqual((hero(s, side, cid)['hp'], hero(s, side, cid)['shield']), (hp, shield))
                self.assertEqual(s['rng'], rng)
                self.assertFalse(any(e['type'] == 'damage' for e in s['events'] if e['seq'] > seq))
                self.assertEqual(kind_count(s, 'a'), 2)
                self.assertEqual(total_stacks(s, 'a'), 20)
                end_turn(s, 'a')
                self.assertEqual(total_stacks(s, 'a'), 20)
                self.assertTrue(all(hero(s, side, cid)['hp'] == hp for side, cid, hp, _ in before))

    def test_non_damaging_dots_still_enable_canhong_card_scaling(self):
        from app.modules.card_game.engine.duel_v2.continuous import kind_count, total_stacks
        s = self.game()
        for kind in ('etch', 'venom'):
            apply_dot(s, 'b', kind, by='a', source_cid='canhong', stacks=5)
        self.assertEqual((kind_count(s, 'a'), total_stacks(s, 'a')), (2, 10))
        s = self.play(s, 'C04')
        hit = next(e for e in reversed(s['events']) if e['type'] == 'attack')
        self.assertEqual(hit['attack'], 4)
        self.assertEqual(hero(s, 'a', 'canhong')['shield'], 2)

    def test_collapse_lasts_until_applier_next_turn_end(self):
        s = self.game()
        force_collapse(s, 'b', 'zero', by='a')
        end_turn(s, 'a')
        self.assertTrue(hero(s, 'b', 'zero')['flags'].get('collapse'))
        begin_turn(s, 'a')
        self.assertTrue(hero(s, 'b', 'zero')['flags'].get('collapse'))
        end_turn(s, 'a')
        self.assertNotIn('collapse', hero(s, 'b', 'zero').get('flags') or {})

    def test_adler_attack_matches_shield(self):
        s = self.game()
        end_turn(s, 'a')
        adler = hero(s, 'b', 'adler')
        from app.modules.card_game.engine.duel_v2.state import attack_value
        self.assertEqual(attack_value(adler, s, 'b'), adler['shield'])
        self.assertEqual(adler['shield'], 2)

    def test_adler_ultimate_strips_all_enemy_shields_and_guard_ticks_twice(self):
        from copy import deepcopy
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
        from app.modules.card_game.engine.duel_v2.state import energy_max
        for burn in (False, True):
            with self.subTest(burn=burn):
                s = self.game()
                end_turn(s, 'a')
                summon = summon_front(s, 'a', name='护盾召唤物', attack=0, hp=4)
                s['sides']['a']['shield'] = 4
                for h in s['sides']['a']['characters'].values():
                    h['shield'] = 3
                ally_shields = {cid: h['shield'] for cid, h in s['sides']['b']['characters'].items()}
                if burn:
                    front_debuff(s, 'a')['burn'] = {'left': 2, 'by': 'b', 'stacks': 1}
                hero(s, 'b', 'adler')['energy'] = energy_max(hero(s, 'b', 'adler'))
                s['sides']['b']['ap'] = 0
                hp = sum(h['hp'] for h in s['sides']['a']['characters'].values())
                opening, seq = capture_public_board(s), s['event_seq']
                s['_public_board'] = deepcopy(opening)
                s = apply_action(s, 'b', {'type': 'ultimate', 'character_id': 'adler'})
                self.assertEqual((s['sides']['b']['ap'], hero(s, 'b', 'adler')['energy']), (0, 0))
                self.assertEqual(sum(h['hp'] for h in s['sides']['a']['characters'].values()), hp)
                self.assertEqual(s['sides']['a']['shield'], 0)
                self.assertTrue(all(h['shield'] == 0 for h in s['sides']['a']['characters'].values()))
                self.assertEqual({cid: h['shield'] for cid, h in s['sides']['b']['characters'].items()}, ally_shields)
                self.assertEqual(front_debuff(s, 'a')['dots']['guard']['left'], 2)
                events = [e for e in observe(s, 'b')['presentation']['events'] if e['seq'] > seq]
                replay = replay_public_board(opening, events)
                self.assertEqual(replay['sides']['a']['shield'], 0)
                self.assertTrue(all(h['shield'] == 0 for h in replay['sides']['a']['characters']))
                self.assertFalse(any(e['type'] == 'damage' for e in events))
                # Isolate the periodic effect and prove later shields do not absorb it.
                front_debuff(s, 'a').pop('burn', None)
                for h in s['sides']['a']['characters'].values():
                    h['shield'] = 3
                s = clone_state(json.loads(json.dumps(s)))
                start = s['event_seq']
                end_turn(s, 'b')
                hits = [e for e in s['events'] if e['seq'] > start and e['type'] == 'damage'
                        and e.get('source') == 'b:adler']
                self.assertEqual(len(hits), 1)
                self.assertNotEqual(hits[0]['target'], f'a:{summon}')
                self.assertEqual(hits[0]['before']['shield'], hits[0]['after']['shield'])
                self.assertEqual(sum(h['hp'] for h in s['sides']['a']['characters'].values()), hp - 1)
                self.assertEqual(front_debuff(s, 'a')['dots']['guard']['left'], 1)
                end_turn(s, 'a')
                end_turn(s, 'b')
                self.assertEqual(sum(h['hp'] for h in s['sides']['a']['characters'].values()), hp - 2)
                self.assertNotIn('guard', front_debuff(s, 'a')['dots'])

    def test_adler_ultimate_refreshes_existing_guard_without_stacking(self):
        from app.modules.card_game.engine.duel_v2.state import energy_max
        s = self.game()
        end_turn(s, 'a')
        apply_dot(s, 'a', 'guard', by='b', source_cid='adler', stacks=1, duration=3)
        hero(s, 'b', 'adler')['energy'] = energy_max(hero(s, 'b', 'adler'))
        s = apply_action(s, 'b', {'type': 'ultimate', 'character_id': 'adler'})
        guard = front_debuff(s, 'a')['dots']['guard']
        self.assertEqual((guard['stacks'], guard['left'], guard['source']), (1, 2, 'b:adler'))

    def test_zaowu_ultimate_buffs_allied_summons_in_live_state_and_replay(self):
        from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
        from app.modules.card_game.engine.duel_v2.state import energy_max
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        for summon_kind in ('gulang', 'other'):
            for equipped, bonus in ((False, 2), (True, 3)):
                with self.subTest(summon=summon_kind, equipped=equipped):
                    s = self.game()
                    if summon_kind == 'gulang':
                        s = self.play(s, 'S01')
                        cid = s['sides']['a']['front']
                    else:
                        cid = summon_front(s, 'a', name='塔吉多', attack=2, hp=4)
                    if equipped:
                        s = self.play(s, 'S08')
                    foe = summon_front(s, 'b', name='对方召唤物', attack=2, hp=4)
                    hero(s, 'a', 'canhong').update(hp=0, down_turns=3)
                    targets = [('a', cid), ('a', 'nanali'), ('a', 'anhunqu'),
                               ('a', 'zaowu'), ('a', 'canhong'), ('b', foe), ('b', 'zero')]
                    before = {(side, target): attack_value(hero(s, side, target), s, side)
                              for side, target in targets}
                    hero(s, 'a', 'zaowu')['energy'] = energy_max(hero(s, 'a', 'zaowu'))
                    opening, seq = capture_public_board(s), s['event_seq']
                    s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zaowu'})
                    s = clone_state(json.loads(json.dumps(s)))
                    replay = replay_public_board(opening, [e for e in s['events'] if e['seq'] > seq])
                    views = [observe(s, viewer, include_previews=False) for viewer in ('a', 'b')] + [replay]
                    for side, target in targets:
                        expected = before[side, target] + (bonus if side == 'a' and target in
                                   (cid, 'nanali', 'anhunqu') else 0)
                        self.assertEqual(attack_value(hero(s, side, target), s, side), expected)
                        for view in views:
                            shown = next(h for h in view['sides'][side]['characters'] if h['id'] == target)
                            self.assertEqual(shown['attack'], expected)
                    self.assertTrue(any('己方召唤物' in e['text'] for e in s['events'] if e['seq'] > seq))

    def test_zaowu_ultimate_does_not_transfer_bonus_to_later_summons(self):
        from app.modules.card_game.engine.duel_v2.state import energy_max
        s = self.play(self.game(), 'S01')
        old = s['sides']['a']['front']
        hero(s, 'a', 'zaowu')['energy'] = energy_max(hero(s, 'a', 'zaowu'))
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zaowu'})
        self.assertEqual(attack_value(hero(s, 'a', old), s, 'a'), 2)
        s = self.play(s, 'S02')
        new = s['sides']['a']['front']
        self.assertNotEqual(new, old)
        self.assertNotIn(old, s['sides']['a']['order'])
        self.assertEqual(attack_value(hero(s, 'a', new), s, 'a'), 0)

    def test_zaowu_summon_bonus_refreshes_without_stacking_and_expires(self):
        from app.modules.card_game.engine.duel_v2.state import energy_max
        s = self.play(self.game(), 'S01')
        cid = s['sides']['a']['front']
        base = attack_value(hero(s, 'a', cid), s, 'a')
        hero(s, 'a', 'zaowu')['energy'] = energy_max(hero(s, 'a', 'zaowu'))
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zaowu'})
        self.assertEqual(attack_value(hero(s, 'a', cid), s, 'a'), base + 2)
        end_turn(s, 'a')
        end_turn(s, 'b')
        self.assertEqual(attack_value(hero(s, 'a', cid), s, 'a'), base + 2)
        s = self.play(s, 'S08')
        hero(s, 'a', 'zaowu')['energy'] = energy_max(hero(s, 'a', 'zaowu'))
        s = apply_action(s, 'a', {'type': 'ultimate', 'character_id': 'zaowu'})
        self.assertEqual(attack_value(hero(s, 'a', cid), s, 'a'), base + 3)
        s = clone_state(json.loads(json.dumps(s)))
        end_turn(s, 'a')
        end_turn(s, 'b')
        self.assertEqual(attack_value(hero(s, 'a', cid), s, 'a'), base + 3)
        end_turn(s, 'a')
        end_turn(s, 'b')
        self.assertEqual(s['sides']['a']['front'], cid)
        self.assertEqual(attack_value(hero(s, 'a', cid), s, 'a'), base)

    def test_zaowu_bonus_floors_per_kind(self):
        s = self.game()
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 1}
        apply_dot(s, 'b', 'etch', by='a', source_cid='canhong', stacks=5)
        before = sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        distribute(s, 'b', 4, source='a:canhong', by='a', reason='增幅')
        lost = before - sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        self.assertEqual(lost, 6)
