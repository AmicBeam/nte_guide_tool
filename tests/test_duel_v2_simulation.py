import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from random import Random
from app.modules.card_game.engine.duel_v2 import new_game,apply_action,legal_actions,acting_side,observe
from app.modules.card_game.engine.duel_v2.simulation import simulate_action,_without_patches,collecting_patches
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.content.duel_v2 import STARTER_DECKS,CARDS,CHARACTERS,validate_deck


def mechanics(s):
    return {k:v for k,v in s.items() if k not in ('events','_public_board')}


def events(s):
    return [{k:v for k,v in e.items() if k!='patch'} for e in s['events']]


class SimulationTest(unittest.TestCase):
    def test_full_catalog_rosters_in_random_legal_trajectories(self):
        ids=list(CHARACTERS);builds=list(STARTER_DECKS)
        for start in range(0,len(ids),4):
            chars=[ids[(start+i)%len(ids)] for i in range(4)]
            cards=[k for cid in chars for k,c in CARDS.items() if c['character_id']==cid and not c.get('derived')]
            if len(cards)!=32:raise AssertionError((chars,len(cards)))
            builds.append(validate_deck(dict(id='parity',name='parity',character_ids=chars,card_ids=cards)))
        for i,b in enumerate(builds):
            slow=new_game(seed=606+i,first_side=('a','b')[i%2],decks={'a':b,'b':builds[(i+1)%len(builds)]})
            fast=deepcopy(slow);rng=Random(300+i)
            for step in range(60):
                if slow['phase']=='finished':break
                side=acting_side(slow);a=[x for x in legal_actions(slow,side) if x['type']!='concede']
                self.assertEqual(a,[x for x in legal_actions(fast,side) if x['type']!='concede'])
                action=rng.choice(a)
                slow=apply_action(slow,side,action);fast=simulate_action(fast,side,action)
                self.assertEqual(mechanics(slow),mechanics(fast),(i,step,action))
                self.assertEqual(events(slow),events(fast),(i,step,action))
                for viewer in ('a','b'):
                    self.assertEqual(observe(clean(deepcopy(slow)),viewer,include_previews=False),observe(clean(deepcopy(fast)),viewer,include_previews=False))
                slow=clean(slow);fast=clean(fast)

    def test_input_unchanged_default_replay_and_error_scope_restored(self):
        s=new_game(seed=40);before=deepcopy(s);a=legal_actions(s,'a')[0]
        fast=simulate_action(s,'a',a);self.assertEqual(s,before)
        self.assertFalse(any('patch' in e for e in fast['events'][len(s['events']):]))
        ordinary=apply_action(s,'a',a);self.assertTrue(any('patch' in e for e in ordinary['events']))
        with self.assertRaises(ValueError):simulate_action(s,'a',{'type':'invalid'})
        self.assertTrue(collecting_patches())
        ordinary=apply_action(s,'a',a);self.assertTrue(any(e.get('patch') for e in ordinary['events']))

    def test_scoped_setting_does_not_change_other_threads(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            with _without_patches():
                self.assertFalse(collecting_patches())
                self.assertTrue(executor.submit(collecting_patches).result())
                with _without_patches():self.assertFalse(collecting_patches())
                self.assertFalse(collecting_patches())
            self.assertTrue(collecting_patches())
