"""User-authored Heiyu card rules and private choice boundaries."""
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog, validate_deck
from app.modules.card_game.engine.duel_v2 import apply_action, new_game, observe
from app.modules.card_game.engine.duel_v2.entities import clone_state
from app.modules.card_game.engine.duel_v2.state import card_instance, hero

TEAM = ['heiyu', 'zero', 'nanali', 'jiuyuan']


def game():
    deck = validate_deck(dict(id='heiyu-test', name='黑羽测试', character_ids=TEAM,
        card_ids=[cid for owner in TEAM for cid, card in CARDS.items()
                  if card['character_id'] == owner and not card.get('derived')]))
    s = new_game(seed=42, first_side='a', skip_mulligan=True,
                 decks={'a': deck, 'b': deck}, escalation=False)
    for team in s['sides'].values():
        team.update(ap=10, hp=100, shield=0, hand=[])
    return s


def give(s, cid):
    card = card_instance(s, cid, 'a')
    s['sides']['a']['hand'].append(card)
    return card


def play(s, cid, target=None):
    card = give(s, cid)
    action = dict(type='play_card', card_id=card['instance_id'])
    if target:
        action['target_id'] = target
    return apply_action(s, 'a', action)


def top(s, ids):
    cards = [card_instance(s, cid, 'a') for cid in ids]
    s['sides']['a']['deck'] = cards
    return cards


def confirm(s, bottom=None):
    cards = s['pending_choice']['cards']
    bottom_card = next((c for c in cards if c['card_id'] == bottom), None)
    return apply_action(s, 'a', dict(type='choose_deck_order',
        card_ids=[c['instance_id'] for c in cards if c is not bottom_card],
        bottom_card_id=bottom_card['instance_id'] if bottom_card else None))


def witch(s):
    hero(s, 'a', 'heiyu')['energy'] = 5
    return apply_action(s, 'a', dict(type='ultimate', character_id='heiyu'))


class HeiyuTest(unittest.TestCase):
    def test_five_energy_compact_mechanisms_and_costume_names(self):
        self.assertEqual(CHARACTERS['heiyu']['energy_max'], 5)
        self.assertEqual([CARDS[c]['name'] for c in ('P07', 'P08')], ['牧星者', '不凋羽'])
        for cid in ('P07', 'P08'):
            self.assertEqual((CARDS[cid]['attack'], CARDS[cid]['hp']), (3, 6))
        self.assertEqual(CARDS['PF01']['description'], '瞬发。「占卜」（至少置底 1 张），然后抽 1 张牌。')
        self.assertTrue(CARDS['PF01']['derived'])
        self.assertFalse(any('【魔女效果】' in CARDS[f'P{i:02d}']['description'] for i in range(1, 9)))
        self.assertIn('heiyu', {c['id'] for c in get_catalog(include_test_characters=False)['characters']})

    def test_basic_sortie_inserts_omen_and_battle_refills_harmony(self):
        s = game()
        self.assertEqual(s['phase'], 'playing')
        before = len(s['sides']['a']['deck'])
        hero(s, 'a', 'heiyu')['harmony'] = 0
        s = play(s, 'P01')
        self.assertEqual(len(s['sides']['a']['deck']), before + 1)
        self.assertEqual(hero(s, 'a', 'heiyu')['harmony'], 2)

    def test_omen_waits_for_confirmation_counts_once_and_moves_bottom_to_hand(self):
        s = game()
        cards = top(s, ['P01', 'P03', 'Z01', 'P05'])
        s = play(s, 'PF01')
        self.assertEqual(s['sides']['a']['heiyu_divinations'], 1)
        self.assertEqual(s['sides']['a']['hand'], [])
        self.assertEqual(observe(s, 'b')['pending_choice']['choices'], [])
        s = confirm(clone_state(json.loads(json.dumps(s))), 'P05')
        self.assertEqual(s['sides']['a']['heiyu_divinations'], 1)
        self.assertEqual({c['entity_id'] for c in s['sides']['a']['hand']},
                         {cards[0]['entity_id'], cards[3]['entity_id']})

    def test_witch_uses_one_top_effect_and_free_exit_keeps_buff(self):
        s = witch(game())
        top(s, ['P01', 'P01'])
        s = apply_action(s, 'a', dict(type='attack', character_id='heiyu'))
        self.assertEqual(s['sides']['b']['hp'], 96)
        self.assertEqual([c['card_id'] for c in s['sides']['a']['deck']], ['P01'])
        self.assertEqual(hero(s, 'a', 'heiyu')['harmony'], 1)
        before = observe(s, 'a')['sides']['a']
        s = apply_action(s, 'a', dict(type='exit_ultimate', character_id='heiyu'))
        after = observe(s, 'a')['sides']['a']
        self.assertEqual(after['ap'], before['ap'])
        self.assertEqual([c['attack'] for c in after['characters']], [c['attack'] for c in before['characters']])

    def test_witch_p04_releases_after_original_attack_and_triggers_next_top(self):
        s = witch(game())
        top(s, ['P04', 'P01'])
        s = apply_action(s, 'a', dict(type='attack', character_id='heiyu'))
        self.assertEqual(s['sides']['b']['hp'], 93)
        self.assertEqual(s['sides']['a']['deck'], [])
        self.assertEqual(len([e for e in s['events'] if e['type'] == 'attack' and e.get('actor') == 'a:heiyu']), 2)

    def test_p03_draw_then_place_one_hand_card_at_either_end(self):
        from app.modules.card_game.engine.duel_v2.entities import entity_action
        for bottom in (False, True):
            for select_drawn in (False, True):
                with self.subTest(bottom=bottom, select_drawn=select_drawn):
                    s = game()
                    held = give(s, 'Z03')
                    drawn, rest = top(s, ['P05', 'Z02'])
                    s = play(s, 'P03')
                    self.assertEqual(s['pending_choice']['kind'], 'hand_deck_placement')
                    self.assertEqual(s['sides']['a']['ap'], 9)
                    self.assertEqual(s['sides']['a']['front'], None)
                    self.assertEqual({c['entity_id'] for c in s['pending_choice']['cards']},
                                     {held['entity_id'], drawn['entity_id']})
                    view = observe(s, 'a')
                    self.assertTrue(view['pending_choice']['hand_placement'])
                    self.assertEqual(view['sides']['a']['deck_count'], 1)
                    self.assertEqual(len([e for e in view['legal_actions'] if e['action']['type'] == 'choose_deck_order']), 4)
                    self.assertEqual(observe(s, 'b')['pending_choice']['choices'], [])
                    chosen = drawn if select_drawn else held
                    action = dict(type='choose_deck_order',
                                  card_ids=[] if bottom else [chosen['instance_id']],
                                  bottom_card_ids=[chosen['instance_id']] if bottom else [])
                    restored = clone_state(json.loads(json.dumps(s)))
                    cursor = restored['event_seq']
                    s = apply_action(restored, 'a', entity_action(restored, 'a', action))
                    self.assertEqual(s['phase'], 'playing')
                    self.assertIsNone(s['pending_choice'])
                    self.assertEqual(s['sides']['a']['front'], 'heiyu')
                    self.assertEqual(s['sides']['a']['ap'], 9)
                    self.assertEqual(hero(s, 'a', 'heiyu')['harmony'], 2)
                    self.assertEqual(s['sides']['a'].get('heiyu_divinations', 0), 0)
                    # Sortie inserts one omen; the existing two-card relative order is retained.
                    deck = [c['entity_id'] for c in s['sides']['a']['deck'] if c['card_id'] != 'PF01']
                    self.assertEqual(deck, [rest['entity_id'], chosen['entity_id']] if bottom else
                                     [chosen['entity_id'], rest['entity_id']])
                    kept = held if select_drawn else drawn
                    self.assertEqual([c['entity_id'] for c in s['sides']['a']['hand']], [kept['entity_id']])
                    self.assertFalse(any(e['type'] == 'reveal' for e in s['events'] if e['seq'] > cursor))

    def test_p03_requires_exactly_one_valid_hand_instance(self):
        s = game()
        held = give(s, 'Z03')
        drawn = top(s, ['Z03'])[0]
        s = play(s, 'P03')
        for top_ids, bottom_ids in (([], []), ([held['instance_id']], [drawn['instance_id']]),
                                    (['foreign'], []), ([held['instance_id']], [held['instance_id']])):
            with self.subTest(top=top_ids, bottom=bottom_ids), self.assertRaises(ValueError):
                apply_action(s, 'a', dict(type='choose_deck_order', card_ids=top_ids, bottom_card_ids=bottom_ids))

    def test_p03_does_not_trigger_bottomed_cards_divination_benefit(self):
        s = game()
        drawn = top(s, ['P05'])[0]
        s = play(s, 'P03')
        s = apply_action(s, 'a', dict(type='choose_deck_order', card_ids=[], bottom_card_ids=[drawn['instance_id']]))
        self.assertEqual(s['sides']['a']['hand'], [])
        self.assertTrue(any(c['entity_id'] == drawn['entity_id'] for c in s['sides']['a']['deck']))

    def test_p03_witch_sortie_still_checks_placed_top_card(self):
        s = witch(game())
        drawn = top(s, ['P05', 'Z02'])[0]
        s = play(s, 'P03')
        s = apply_action(s, 'a', dict(type='choose_deck_order', card_ids=[drawn['instance_id']], bottom_card_ids=[]))
        self.assertEqual(s['sides']['a']['front'], 'heiyu')
        self.assertEqual([c['entity_id'] for c in s['sides']['a']['hand']], [drawn['entity_id']])
        self.assertEqual(s['sides']['a'].get('heiyu_divinations', 0), 0)
        self.assertTrue(any(e['type'] == 'reveal' and e['card']['entity_id'] == drawn['entity_id'] for e in s['events']))

    def test_p03_empty_draw_ends_game_before_choice_or_sortie(self):
        s = game()
        give(s, 'Z03')
        top(s, [])
        s = play(s, 'P03')
        self.assertEqual(s['phase'], 'finished')
        self.assertEqual(s['winner'], 'b')
        self.assertIsNone(s['pending_choice'])
        self.assertIsNone(s['sides']['a']['front'])

    def test_p05_uses_previous_count_then_divines_and_p08_advances_owned_card(self):
        s = game()
        s['sides']['a']['heiyu_divinations'] = 3
        before = sum(h['hp'] for h in s['sides']['b']['characters'].values())
        s = play(s, 'P05')
        self.assertEqual(before - sum(h['hp'] for h in s['sides']['b']['characters'].values()), 6)
        self.assertEqual(s['sides']['a']['heiyu_divinations'], 4)
        s = confirm(s)
        s = play(s, 'P08')
        owned = top(s, ['Z01', 'P04', 'Z02', 'Z03'])[1]
        s = play(s, 'PF01')
        self.assertEqual(s['pending_choice']['cards'][0]['entity_id'], owned['entity_id'])

    def test_omen_rejects_zero_bottom_and_multiple_bottom_effects_resolve(self):
        from app.modules.card_game.engine.duel_v2.entities import entity_action
        s = game()
        hero(s, 'a', 'nanali').update(hp=0, down_turns=2)
        cards = top(s, ['Z01', 'P06', 'P06'])
        s = play(s, 'PF01')
        self.assertEqual(s['pending_choice']['min_bottom'], 1)
        self.assertEqual(s['pending_choice']['max_bottom'], 3)
        self.assertEqual(len(s['pending_choice']['cards']), 3)
        with self.assertRaises(ValueError):
            apply_action(s, 'a', dict(type='choose_deck_order',
                card_ids=[c['instance_id'] for c in cards], bottom_card_ids=[]))
        action = dict(type='choose_deck_order', card_ids=[cards[0]['instance_id']],
                      bottom_card_ids=[cards[1]['instance_id'], cards[2]['instance_id']])
        s = apply_action(clone_state(json.loads(json.dumps(s))), 'a', entity_action(s, 'a', action))
        self.assertEqual(hero(s, 'a', 'nanali')['down_turns'], 0)
        self.assertEqual(hero(s, 'a', 'nanali')['hp'], hero(s, 'a', 'nanali')['max_hp'])
        self.assertEqual(s['sides']['a']['heiyu_divinations'], 1)
        self.assertEqual([c['entity_id'] for c in s['sides']['a']['hand']], [cards[0]['entity_id']])

    def test_all_inspected_cards_may_go_to_bottom_in_chosen_order(self):
        s = game()
        cards = top(s, ['Z01', 'Z02', 'Z03', 'Z04'])
        s = play(s, 'PF01')
        order = list(reversed(cards))
        s = apply_action(s, 'a', dict(type='choose_deck_order', card_ids=[],
            bottom_card_ids=[c['instance_id'] for c in order]))
        self.assertEqual([c['entity_id'] for c in s['sides']['a']['hand']], [order[0]['entity_id']])
        self.assertEqual([c['entity_id'] for c in s['sides']['a']['deck']], [c['entity_id'] for c in order[1:]])
