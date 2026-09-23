"""持续伤害、浊燃、噩梦、倾陷时长，以及四名测试角色的关键结算。"""
import unittest

from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.continuous import apply_dot, distribute
from app.modules.card_game.engine.duel_v2.flow import begin_turn, end_turn
from app.modules.card_game.engine.duel_v2.state import card_instance, force_collapse, front_debuff, hero


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

    def test_zaowu_bonus_floors_per_kind(self):
        s = self.game()
        front_debuff(s, 'b')['burn'] = {'left': 2, 'by': 'a', 'stacks': 1}
        apply_dot(s, 'b', 'etch', by='a', source_cid='canhong', stacks=5)
        before = sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        distribute(s, 'b', 4, source='a:canhong', by='a', reason='增幅')
        lost = before - sum(hero(s, 'b', cid)['hp'] for cid in s['sides']['b']['characters'])
        self.assertEqual(lost, 6)
