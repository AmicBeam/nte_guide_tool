"""September 18 user-authored clock/arid rules and hidden-state boundaries."""
from app.modules.card_game.engine.duel_v2.equipment import permanent_attack
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe, legal_actions
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, damage, energy, heal, move_out
from app.modules.card_game.engine.duel_v2.entities import entity_action
from app.modules.card_game.engine.duel_v2.flow import begin_turn


class XunReworkTest(unittest.TestCase):
    def game(self):
        deck = dict(id='xun-test', name='浔', character_ids=['xun','nanali','zero','jiuyuan'],
                    card_ids=[p+str(i).zfill(2) for p in ('X','N','Z','J') for i in range(1,5) for _ in range(2)])
        s = new_game(seed=21, first_side='a', skip_mulligan=True, decks={'a':deck,'b':deck})
        for team in s['sides'].values(): team['hand']=[]; team['shield']=0
        s['sides']['a']['ap']=2
        return s

    def play(self,s,cid):
        c=card_instance(s,cid,'a'); s['sides']['a']['hand'].append(c)
        return apply_action(s,'a',{'type':'play_card','card_id':c['instance_id']})

    def choose(self,s,key,value):
        c=next(c for c in s['pending_choice']['cards'] if c.get(key)==value)
        return apply_action(s,'a',entity_action(s,'a',{'type':'choose','choice_id':c['instance_id']}))

    def next(self,s):
        s=apply_action(s,'a',{'type':'end_turn'})
        return apply_action(s,'b',{'type':'end_turn'})

    def test_clock_not_energy_and_downed_tick(self):
        s=self.game();h=hero(s,'a','xun')
        self.assertEqual((permanent_attack(h),h['max_hp'],h['energy']),(3,6,1))
        energy(s,'a','xun',4);self.assertEqual(h['energy'],1)
        damage(s,'a','xun',99);s=self.next(s)
        self.assertEqual(hero(s,'a','xun')['energy'],2)
        s['turn']=12;hero(s,'a','xun')['energy']=5;begin_turn(s,'a')
        v=observe(s,'a')['sides']['a']['characters'][0]
        self.assertEqual((v['energy'],v['energy_max'],v['resource_name']),(4,4,'时计'))

    def test_sortie_requires_other_espers_down(self):
        s=self.game();a={'type':'attack','character_id':'xun'}
        self.assertNotIn(a,legal_actions(s,'a'))
        for cid in ('nanali','zero','jiuyuan'):damage(s,'a',cid,99)
        self.assertIn(a,legal_actions(s,'a'))
        s=apply_action(s,'a',a);self.assertEqual(s['sides']['b']['hp'],27)
        self.assertEqual(hero(s,'a','xun')['energy'],1)

    def test_unique_arid_support_and_ultimate(self):
        s=self.game();s['sides']['a'].update(front='zero',ap=8)
        hero(s,'a','zero')['harmony']=2
        s=self.play(s,'N03')
        self.assertEqual(hero(s,'a','xun')['arid'],1)
        hero(s,'a','nanali')['energy']=5
        s=apply_action(s,'a',{'type':'ultimate','character_id':'nanali'})
        self.assertEqual(hero(s,'a','xun')['arid'],1)
        s=self.next(s);hero(s,'a','zero')['energy']=5
        s=apply_action(s,'a',{'type':'ultimate','character_id':'zero'})
        self.assertEqual(hero(s,'a','xun')['arid'],2)

    def test_ultimate_second_choice_tiers_and_gear(self):
        for arid,gear,main,aoe in [(0,False,4,0),(1,False,6,0),(2,False,8,2),(3,False,10,2),(3,True,16,5)]:
            with self.subTest(arid=arid,gear=gear):
                s=self.game();h=hero(s,'a','xun');h.update(energy=5,arid=arid,shape='X08' if gear else None)
                for t in s['sides']['b']['characters'].values():t.update(hp=30,max_hp=30)
                s=apply_action(s,'a',{'type':'ultimate','character_id':'xun'})
                self.assertEqual(s['phase'],'choice');self.assertEqual(hero(s,'b','zero')['hp'],30)
                self.assertFalse(observe(s,'b')['pending_choice']['choices'])
                s=self.choose(s,'ultimate_target','zero')
                self.assertEqual(hero(s,'b','zero')['hp'],30-main)
                self.assertEqual(hero(s,'b','nanali')['hp'],30-aoe)
                self.assertEqual(hero(s,'a','xun').get('arid'),0)
                self.assertEqual(s['phase'],'playing')

    def test_ultimate_overflow_shields_and_bench(self):
        s=self.game();hero(s,'a','xun').update(energy=5,arid=3)
        for h in s['sides']['b']['characters'].values():h.update(hp=20,max_hp=20)
        hero(s,'b','zero').update(hp=2,shield=1)
        s=apply_action(s,'a',{'type':'ultimate','character_id':'xun'})
        s=self.choose(s,'ultimate_target','zero')
        self.assertEqual(s['sides']['b']['hp'],25)
        self.assertEqual(hero(s,'b','zero')['down_turns'],3)

    def test_rewind_preserves_transport_and_permanent_removal(self):
        s=self.game();rewind=card_instance(s,'X01','a');s['sides']['a']['hand']=[rewind]
        # Seed the actual turn-start snapshot with this physical card.
        s['turn_snapshots']['a'][0]['sides']['a']['hand']=[deepcopy(rewind)]
        old=deepcopy(s['turn_snapshots']['a'][0]['sides'])
        s=apply_action(s,'a',{'type':'attack','character_id':'nanali'});s=self.next(s)
        seq,version=s['event_seq'],s['version']
        s=apply_action(s,'a',{'type':'play_card','card_id':rewind['instance_id']})
        self.assertEqual(s['version'],version+1);self.assertGreater(s['event_seq'],seq)
        self.assertEqual(s['turn'],1);self.assertEqual(s['sides']['b']['hp'],old['b']['hp'])
        self.assertEqual(s['sides']['a']['deck'],old['a']['deck'])
        self.assertIn(rewind['entity_id'],s['time_removed'])
        self.assertNotIn(rewind['entity_id'],[c['entity_id'] for c in s['sides']['a']['hand']])
        s=self.next(s);s=self.play(s,'X01')
        self.assertEqual(len(s['time_removed']),2)
        self.assertFalse(any(c['entity_id'] in s['time_removed'] for c in s['sides']['a']['hand']))
        self.assertEqual(json.loads(json.dumps(s))['turn'],1)
        for side in ('a','b'):
            v=json.dumps(observe(s,side));self.assertNotIn('turn_snapshots',v);self.assertNotIn('time_removed',v)

    def test_snapshot_limit_no_recursive_history(self):
        s=self.game()
        for _ in range(3):s=self.next(s)
        self.assertEqual(len(s['turn_snapshots']['a']),2)
        for rows in s['turn_snapshots'].values():
            for snap in rows:
                self.assertNotIn('turn_snapshots',snap);self.assertNotIn('events',snap)
        plain=new_game(first_side='a',skip_mulligan=True)
        self.assertNotIn('turn_snapshots',plain)

    def test_life_lock_and_private_deck_expire(self):
        s=self.game();s['sides']['a']['hp']=20
        s=self.play(s,'X02');damage(s,'a','player',7);heal(s,'a','player',5)
        self.assertEqual(s['sides']['a']['hp'],20)
        s=self.play(s,'X03')
        self.assertEqual([c['instance_id'] for c in observe(s,'a')['sides']['a']['visible_deck']],
                         [c['instance_id'] for c in s['sides']['a']['deck']])
        self.assertNotIn('visible_deck',observe(s,'b')['sides']['a'])
        from app.modules.card_game.engine.duel_v2.replay_log import public_replay_view
        self.assertNotIn('visible_deck',json.dumps(public_replay_view(s,'a')))
        s=apply_action(s,'a',{'type':'end_turn'})
        self.assertNotIn('visible_deck',observe(s,'a')['sides']['a'])
        damage(s,'a','player',3);self.assertEqual(s['sides']['a']['hp'],20)
        s=apply_action(s,'b',{'type':'end_turn'});damage(s,'a','player',3)
        self.assertEqual(s['sides']['a']['hp'],17)

    def test_antique_cost_weights_and_zero_conversion(self):
        for ap in (1,2,3,5):
            s=self.game();s['sides']['a']['ap']=ap;s=self.play(s,'X04')
            self.assertEqual(s['sides']['a']['ap'],max(0,ap-3))
            c=next(c for c in s['sides']['a']['hand'] if c['card_id']=='XA01')
            self.assertEqual(c['antique_investment'],min(3,ap))
            with patch('app.modules.card_game.content.duel_v2.characters.xun.Random') as rng:
                rng.return_value.choices.return_value=[0];rng.return_value.getrandbits.return_value=42
                s=apply_action(s,'a',{'type':'play_card','card_id':c['instance_id']})
                from app.modules.card_game.content.duel_v2.characters.xun import ANTIQUE_WEIGHTS
                self.assertEqual(rng.return_value.choices.call_args.kwargs['weights'],ANTIQUE_WEIGHTS[min(3,ap)])
            self.assertEqual(s['sides']['a']['ap'],max(0,ap-3)+1)
        s=self.game();s['sides']['a']['ap']=0
        with self.assertRaises(ValueError):self.play(s,'X04')

    def test_family_virtual_play_and_filtered_text(self):
        s=self.game();hero(s,'a','xun')['hp']=1;s=self.play(s,'X06')
        self.assertEqual(hero(s,'a','xun')['hp'],6)
        c=next(c for c in s['sides']['a']['hand'] if c['card_id']=='XF01')
        self.assertNotIn('阿德勒',next(c for c in observe(s,'a')['sides']['a']['hand'] if c['card_id']=='XF01')['description'])
        s=apply_action(s,'a',{'type':'play_card','card_id':c['instance_id']})
        s=self.choose(s,'family_option','nanali')
        self.assertEqual(s['sides']['a']['front'],'nanali')
        self.assertFalse(any(c['card_id']=='N03' for c in s['sides']['a']['discard']))
        self.assertEqual(hero(s,'a','xun')['energy'],1)

    def test_family_description_tracks_current_options(self):
        from app.modules.card_game.engine.duel_v2.state import add_collapse
        s = self.game()
        s['sides']['a']['hand'].append(card_instance(s, 'XF01', 'a'))

        def description():
            return next(c['description'] for c in observe(s, 'a')['sides']['a']['hand']
                        if c['card_id'] == 'XF01')

        self.assertEqual(description(), '二选一：视为打出一张「要叫大姐头」；或视为打出一张「探悉天职」。')
        damage(s, 'a', 'jiuyuan', 99)
        self.assertEqual(description(), '三选一：视为打出一张「要叫大姐头」；或视为打出一张「探悉天职」；或在战斗区召唤一个 2/4 的塔吉多。')
        damage(s, 'a', 'nanali', 99)
        self.assertEqual(description(), '二选一：视为打出一张「探悉天职」；或在战斗区召唤一个 2/4 的塔吉多。')
        add_collapse(s, 'a', 'zero', 5)
        self.assertEqual(description(), '在战斗区召唤一个 2/4 的塔吉多。')
        for cid in ('jiuyuan', 'nanali'):
            hero(s, 'a', cid).update(hp=4, down_turns=0)
        self.assertEqual(description(), '视为打出一张「要叫大姐头」。')
        add_collapse(s, 'a', 'nanali', 5)
        self.assertEqual(description(), '当前无可用选项。')

    def test_summon_persists_then_disappears(self):
        s=self.game();damage(s,'a','jiuyuan',99);s=self.play(s,'XF01')
        s=self.choose(s,'family_option','summon');cid=s['sides']['a']['front']
        self.assertEqual((hero(s,'a',cid)['hp'],permanent_attack(hero(s,'a',cid))),(4,2))
        s=self.next(s);self.assertEqual(s['sides']['a']['front'],cid)
        observe(s,'a');s=apply_action(s,'a',{'type':'attack','character_id':'zero'})
        self.assertNotIn(cid,s['sides']['a']['order']);observe(s,'a')
        s=self.play(s,'XF01');s=self.choose(s,'family_option','summon');cid=s['sides']['a']['front']
        damage(s,'a',cid,4);self.assertIsNone(s['sides']['a']['front'])
        self.assertNotIn(cid,s['sides']['a']['order']);observe(s,'a')

    def test_faction_response_and_collapse(self):
        s=self.game();s['sides']['a']['front']='nanali'
        s['sides']['a']['hand']=[card_instance(s,'X05','a')]
        s=apply_action(s,'a',{'type':'end_turn'})
        s=apply_action(s,'b',{'type':'attack','character_id':'zero'})
        self.assertIsNone(s['sides']['a']['front']);self.assertEqual(hero(s,'a','nanali')['hp'],5)
        s=apply_action(s,'b',{'type':'end_turn'});s=self.play(s,'X07')
        self.assertTrue(hero(s,'a','xun')['flags'].get('collapse'))
        self.assertEqual(sum(c['card_id']=='XF01' for c in s['sides']['a']['hand']),2)
        # Collapse lasts through the next own turn, until that turn ends.
        s=self.next(s);self.assertTrue(hero(s,'a','xun')['flags'].get('collapse'))
        s=apply_action(s,'a',{'type':'end_turn'})
        self.assertFalse(hero(s,'a','xun')['flags'].get('collapse'))


class XunBoundaryTest(unittest.TestCase):
    game = XunReworkTest.game
    next = XunReworkTest.next
    def test_rewind_filtered_from_training_and_rule_ai(self):
        from app.modules.card_game.rl.backend import V2Backend
        from app.modules.card_game.engine.duel_v2 import choose_action
        s=self.next(self.game());card=card_instance(s,'X01','a');s['sides']['a']['hand']=[card]
        a={'type':'play_card','card_id':card['instance_id']}
        self.assertIn(a,legal_actions(s,'a'))
        backend=V2Backend()
        self.assertNotIn(a,backend.legal_actions(s,'a'))
        self.assertNotEqual(a,choose_action(s,'a'))
        with self.assertRaises(ValueError):backend.apply_action(s,'a',a)

    def test_summon_removal_replay_patch(self):
        from app.modules.card_game.engine.duel_v2.summons import summon_front
        from app.modules.card_game.engine.duel_v2.presentation import capture_public_board,diff_public,merge_patch
        s=self.game();cid=summon_front(s,'a',name='测试',attack=2,hp=4)
        before=capture_public_board(s);move_out(s,'a',cid);after=capture_public_board(s)
        self.assertEqual(merge_patch(before,diff_public(before,after)),after)

if __name__=='__main__':unittest.main()
