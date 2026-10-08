"""All use-time effects precede the used card's frame and effect resolution."""
import unittest
from app.modules.card_game.content.duel_v2 import CARDS, validate_deck
from app.modules.card_game.engine.duel_v2 import apply_action, new_game
from app.modules.card_game.engine.duel_v2.state import card_instance, front_debuff, hero, knockdowns
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board


def game():
    ids = ['haniya', 'zero', 'lingke', 'canhong']
    cards = [cid for owner in ids for cid, c in CARDS.items()
             if c['character_id'] == owner and not c.get('derived')]
    deck = validate_deck(dict(id='timing', name='timing', character_ids=ids, card_ids=cards))
    s = new_game(seed=73, first_side='a', skip_mulligan=True, decks={'a': deck, 'b': deck}, escalation=False)
    for team in s['sides'].values():
        team.update(hand=[], ap=20, shield=0)
        for h in team['characters'].values():
            h.update(hp=40, max_hp=40)
    return s


def play(s, cid):
    card = card_instance(s, cid, 'a'); s['sides']['a']['hand'].append(card)
    return apply_action(s, 'a', dict(type='play_card', card_id=card['instance_id']))


class CardUseTimingTest(unittest.TestCase):
    def test_zero_weapon_use_draw_precedes_equipment_for_passive_and_ultimate(self):
        for ultimate in (False, True):
            with self.subTest(ultimate=ultimate):
                s = game(); zero = hero(s, 'a', 'zero')
                zero.update(shape=None if ultimate else 'Z07', awakened=ultimate)
                seq = s['event_seq']; opening = capture_public_board(s)
                after = play(s, 'C06' if ultimate else 'Z08')
                events = [e for e in after['events'] if e['seq'] > seq]
                draw = next(i for i, e in enumerate(events) if e['type'] == 'draw')
                shape = next(i for i, e in enumerate(events) if e['type'] == 'shape')
                self.assertLess(draw, shape)
                self.assertEqual(replay_public_board(opening, events), capture_public_board(after))

    def test_zero_empty_draw_ends_game_before_weapon_effect(self):
        s = game(); hero(s, 'a', 'zero')['shape'] = 'Z07'
        s['sides']['a']['deck'] = []; seq = s['event_seq']
        after = play(s, 'Z08')
        self.assertEqual(after['phase'], 'finished')
        self.assertEqual(hero(after, 'a', 'zero')['shape'], 'Z07')
        self.assertFalse(any(e['type'] == 'shape' for e in after['events'] if e['seq'] > seq))

    def test_first_weapon_cannot_enable_its_own_use_draw(self):
        s = game(); seq = s['event_seq']; after = play(s, 'Z07')
        self.assertFalse(any(e['type'] == 'draw' for e in after['events'] if e['seq'] > seq))

    def test_revival_weapon_use_draw_also_precedes_reviving_its_owner(self):
        s = game(); hero(s, 'a', 'zero')['awakened'] = True
        hero(s, 'a', 'canhong').update(hp=0, down_turns=3)
        s['sides']['a']['deck'] = []; seq = s['event_seq']
        after = play(s, 'C07')
        self.assertEqual(after['phase'], 'finished')
        self.assertEqual(hero(after, 'a', 'canhong')['hp'], 0)
        self.assertFalse(any(e['type'] == 'shape' for e in after['events'] if e['seq'] > seq))

    def test_automatic_revival_weapon_only_releases_without_use_triggers(self):
        s = game(); team = s['sides']['a']
        actor = hero(s, 'a', 'canhong')
        hero(s, 'a', 'zero')['awakened'] = True
        hero(s, 'a', 'lingke')['lingke_chain'] = dict(entity_id=actor.entity_id,
                                                      character_id='canhong', count=2)
        team.update(hand=[card_instance(s, 'C07', 'a')], deck=[], ap=0)
        front_debuff(s, 'b')['burn'] = dict(left=2, by='a', stacks=20)
        actor['hp'] = 0; seq = s['event_seq']
        knockdowns(s)
        events = [e for e in s['events'] if e['seq'] > seq]
        played = [e for e in events if e['type'] == 'play' and (e.get('card') or {}).get('card_id') == 'C07']
        self.assertEqual(len(played), 1)
        self.assertTrue(played[0]['release_only'])
        self.assertIn('自动释放', played[0]['text'])
        self.assertFalse(any(e['type'] in ('draw', 'attack') for e in events))
        self.assertEqual(hero(s, 'a', 'lingke')['lingke_chain']['count'], 2)
        self.assertEqual(s['phase'], 'playing')
        self.assertEqual(actor['shape'], 'C07')
        self.assertGreater(actor['hp'], 0)
        self.assertEqual(team['ap'], 0)
        self.assertFalse(team['used'].get('instant'))
        self.assertEqual([c['card_id'] for c in team['discard']], ['C07'])

    def test_automatic_release_preserves_the_interrupted_parent_card(self):
        s = game(); actor = hero(s, 'a', 'canhong')
        actor.update(hp=1, harmony=0)
        s['sides']['b']['front'] = 'zero'
        s['sides']['a']['hand'] = [card_instance(s, 'C07', 'a')]
        front_debuff(s, 'b')['burn'] = dict(left=2, by='a', stacks=20)
        after = play(s, 'C02')
        self.assertEqual(sorted(c['card_id'] for c in after['sides']['a']['discard']), ['C02', 'C07'])
        self.assertEqual(hero(after, 'a', 'lingke')['lingke_chain']['count'], 1)

    def test_haniya_weapon_use_consumes_aura_before_card_frame_once(self):
        s = game(); hero(s, 'a', 'haniya').update(shape='V08', protagonist_aura=4)
        seq = s['event_seq']; after = play(s, 'Z03')
        events = [e for e in after['events'] if e['seq'] > seq]
        aura = [i for i, e in enumerate(events) if '消耗 4 点主角光环' in e['text']]
        self.assertEqual(len(aura), 1)
        frame = next(i for i, e in enumerate(events) if e['type'] == 'shield' and e.get('target') == 'a:zero')
        self.assertLess(aura[0], frame)
        self.assertEqual(hero(after, 'a', 'haniya')['protagonist_aura'], 0)
        hit = next(e for e in events if e['type'] == 'attack' and e.get('actor') == 'a:zero')
        self.assertTrue(any(m['label'] == '本次出击' and m['value'] == 4 for m in hit['attack_modifiers']))

    def test_canhong_use_mirage_precedes_frame_and_this_cards_damage(self):
        s = game(); seq = s['event_seq']; after = play(s, 'C01')
        events = [e for e in after['events'] if e['seq'] > seq]
        mirage = next(i for i, e in enumerate(events) if '残虹进入幻境' in e['text'])
        attack = next(i for i, e in enumerate(events) if e['type'] == 'attack')
        self.assertLess(mirage, attack)
        self.assertTrue(any('蚀心' in e['text'] for e in events))


if __name__ == '__main__':
    unittest.main()
