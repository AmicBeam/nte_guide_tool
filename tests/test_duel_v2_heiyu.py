"""User-authored compact Heiyu rules; this revision was not run at user request."""
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
        self.assertEqual(CARDS['PF01']['description'], '瞬发。占卜，然后抽 1 张牌。')
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

    def test_p03_hand_top_then_divination_then_sortie(self):
        s = game()
        held = give(s, 'Z03')
        s = play(s, 'P03')
        self.assertEqual(s['pending_choice']['kind'], 'kit')
        s = apply_action(s, 'a', dict(type='choose', choice_id=held['instance_id']))
        self.assertEqual(s['pending_choice']['kind'], 'heiyu_divination')
        self.assertEqual(s['pending_choice']['cards'][0]['entity_id'], held['entity_id'])
        s = confirm(s)
        self.assertEqual(s['sides']['a']['front'], 'heiyu')
        self.assertEqual(hero(s, 'a', 'heiyu')['harmony'], 2)

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
