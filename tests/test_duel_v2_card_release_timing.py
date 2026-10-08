"""Divination releases effects without triggering card-use abilities."""
import json
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, validate_deck
from app.modules.card_game.engine.duel_v2 import apply_action, new_game
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board, replay_public_board
from app.modules.card_game.engine.duel_v2.state import card_instance, hero


def game():
    ids = ['heiyu', 'lingke', 'haniya', 'zero']
    deck = validate_deck(dict(id='release', name='release', character_ids=ids,
        card_ids=[cid for owner in ids for cid,c in CARDS.items()
                  if c['character_id'] == owner and not c.get('derived')]))
    s = new_game(seed=73, first_side='a', skip_mulligan=True, decks={'a':deck,'b':deck}, escalation=False)
    for team in s['sides'].values():
        team.update(hand=[],ap=20,shield=0,hp=100)
        for h in team['characters'].values(): h.update(hp=40,max_hp=40)
    actor = hero(s,'a','heiyu')
    hero(s,'a','lingke')['lingke_chain'] = dict(entity_id=actor.entity_id,character_id='heiyu',count=1)
    hero(s,'a','haniya').update(shape='V08',protagonist_aura=4)
    return s


class CardReleaseTimingTest(unittest.TestCase):
    def test_basic_divination_releases_do_not_count_as_using_the_revealed_card(self):
        for cid in ('P01','P02','P04'):
            with self.subTest(card=cid):
                s=game(); team=s['sides']['a']
                revealed=card_instance(s,cid,'a')
                team['deck']=[revealed]+[card_instance(s,k,'a') for k in ('Z01','Z02','Z03')]
                omen=card_instance(s,'PF01','a'); team['hand']=[omen]
                s=apply_action(s,'a',dict(type='play_card',card_id=omen['instance_id']))
                pending=s['pending_choice']; seq=s['event_seq']; opening=capture_public_board(s)
                action=dict(type='choose_deck_order',card_ids=[c['instance_id'] for c in pending['cards'] if c['card_id'] != cid],
                            bottom_card_ids=[revealed['instance_id']])
                after=apply_action(json.loads(json.dumps(s)),'a',action)
                events=[e for e in after['events'] if e['seq']>seq]
                releases=[e for e in events if e['type']=='play' and (e.get('card') or {}).get('card_id')==cid]
                self.assertEqual(len(releases),1); self.assertTrue(releases[0]['release_only'])
                self.assertFalse(releases[0]['present'])
                self.assertEqual(hero(after,'a','lingke')['lingke_chain']['count'],2)
                self.assertEqual(hero(after,'a','haniya')['protagonist_aura'],4)
                self.assertFalse(any(e['type']=='attack' and e.get('actor')=='a:lingke' for e in events))
                # Released battle cards gain only the ordinary combat resource;
                # they do not invoke Heiyu's "after using a battle card" refill.
                self.assertEqual(hero(after,'a','heiyu')['harmony'],0 if cid=='P02' else 1)
                self.assertEqual([c['card_id'] for c in after['sides']['a']['discard']].count(cid),1)
                self.assertEqual(replay_public_board(opening,events),capture_public_board(after))

    def test_witch_delayed_p04_is_still_release_not_use(self):
        s=game(); actor=hero(s,'a','heiyu'); actor['awakened']=True
        hero(s,'a','lingke')['lingke_chain']['count']=2
        hero(s,'a','haniya')['shape']=None
        card=card_instance(s,'P04','a'); s['sides']['a']['deck']=[card]
        seq=s['event_seq']; after=apply_action(s,'a',dict(type='attack',character_id='heiyu'))
        events=[e for e in after['events'] if e['seq']>seq]
        releases=[e for e in events if e['type']=='play' and (e.get('card') or {}).get('card_id')=='P04']
        self.assertEqual(len(releases),1); self.assertTrue(releases[0]['release_only'])
        self.assertEqual(hero(after,'a','lingke')['lingke_chain']['count'],2)
        self.assertEqual(sum(e['type']=='attack' and e.get('actor')=='a:heiyu' for e in events),2)
        self.assertFalse(any(e['type']=='attack' and e.get('actor')=='a:lingke' for e in events))

    def test_manually_playing_the_same_battle_card_is_a_use(self):
        s=game(); hero(s,'a','lingke')['lingke_chain']['count']=2
        card=card_instance(s,'P01','a'); s['sides']['a']['hand']=[card]
        seq=s['event_seq']; after=apply_action(s,'a',dict(type='play_card',card_id=card['instance_id']))
        events=[e for e in after['events'] if e['seq']>seq]
        self.assertEqual(hero(after,'a','lingke')['lingke_chain']['count'],0)
        self.assertEqual(hero(after,'a','haniya')['protagonist_aura'],0)
        self.assertEqual(hero(after,'a','heiyu')['harmony'],2)
        self.assertTrue(any(e['type']=='attack' and e.get('actor')=='a:lingke' for e in events))


if __name__=='__main__': unittest.main()
