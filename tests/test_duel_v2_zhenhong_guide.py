"""Public setup guidance respects resources, safety and hidden information."""
from copy import deepcopy
import unittest
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe, acting_side, legal_actions
from app.modules.card_game.engine.duel_v2.state import hero, front_debuff
from app.modules.card_game.engine.ai.zhenhong_guide import recommend

class ZhenhongGuideTest(unittest.TestCase):
    def game(self, cards):
        deck=next(d for d in STARTER_DECKS if d['id']=='zhenhong')
        s=new_game(seed=913,first_side='a',skip_mulligan=True,escalation=False,decks={'a':deck,'b':deck})
        for team in s['sides'].values():
            team['deck'] += team['hand'];team['hand']=[];team['ap']=2
        for key in cards:
            c=next(c for c in s['sides']['a']['deck'] if c['card_id']==key)
            s['sides']['a']['deck'].remove(c);s['sides']['a']['hand'].append(c)
        return s
    def guide(self,s,a):
        before=deepcopy(s);r=recommend(s,'a',a,seed=77881,seconds=5,max_plans=48)
        self.assertEqual(s,before)
        return r
    def test_missing_fangs_prefers_free_setup_over_unprepared_attack(self):
        s=self.game(['I04']);a=dict(type='attack',character_id='zero')
        r=self.guide(s,a);self.assertEqual(r['selected'],dict(type='play_card',card_id=s['sides']['a']['hand'][0]['instance_id']))
        after=apply_action(s,'a',r['selected']);self.assertGreater(hero(after,'a','yi')['beast_fangs'],0)
        self.assertEqual(after['sides']['a']['ap'],2)
    def test_fangs_then_light_to_yi_arms_delay(self):
        s=self.game([]);hero(s,'a','yi')['beast_fangs']=4
        s['sides']['a']['last_front']='zero';hero(s,'a','zero')['harmony']=2
        r=self.guide(s,dict(type='attack',character_id='zhenhong'))
        after=apply_action(s,'a',r['selected'])
        self.assertTrue(front_debuff(after,'b').get('delay'))
    def test_zero_battle_then_yi_battle_sets_up_fangs_and_delay_together(self):
        s=self.game(['I04','Z04','I01'])
        held={c['card_id']:c['instance_id'] for c in s['sides']['a']['hand']}
        baseline=dict(type='play_card',card_id=held['I04'])
        r=self.guide(s,baseline);self.assertEqual(r['selected'],dict(type='play_card',card_id=held['Z04']))
        self.assertIn([dict(type='play_card',card_id=held['Z04']),dict(type='play_card',card_id=held['I01'])],r['priority_routes'])
        s=apply_action(s,'a',r['selected']);self.assertEqual(hero(s,'a','zero')['harmony'],2)
        self.assertFalse(hero(s,'a','yi').get('beast_fangs'))
        r=self.guide(s,baseline);self.assertEqual(r['selected'],dict(type='play_card',card_id=held['I01']))
        s=apply_action(s,'a',r['selected'])
        self.assertEqual(hero(s,'a','yi')['beast_fangs'],4)
        self.assertEqual(front_debuff(s,'b')['delay']['by'],'a')
        self.assertEqual(hero(s,'a','zero')['harmony'],0)
        self.assertEqual(s['sides']['a']['ap'],0)
    def test_macro_does_not_bypass_ap_or_dead_harmony_payer(self):
        for mode in ('one_ap','fatal_zero'):
            s=self.game(['I04','Z04','I01'])
            if mode=='one_ap':s['sides']['a']['ap']=1
            else:
                s['sides']['b']['front']='zero';foe=hero(s,'b','zero')
                foe['base_stats']['attack']=20;foe.update(hp=50,max_hp=50)
            baseline=dict(type='play_card',card_id=s['sides']['a']['hand'][0]['instance_id'])
            r=self.guide(s,baseline);self.assertEqual(r['priority_routes'],[])
    def test_free_fangs_remains_a_fallback_without_yi_battle(self):
        s=self.game(['I04','Z04']);baseline=dict(type='attack',character_id='zero')
        r=self.guide(s,baseline);self.assertEqual(r['priority_routes'],[])
        self.assertEqual(r['selected'],dict(type='play_card',card_id=s['sides']['a']['hand'][0]['instance_id']))
    def test_existing_delay_cashout_beats_paid_fangs(self):
        s=self.game(['I01']);s['sides']['a']['ap']=1;s['sides']['a']['last_front']='zero';hero(s,'a','zero')['harmony']=2
        front_debuff(s,'b')['delay']=dict(left=2,by='a',tick_side='a')
        baseline=dict(type='play_card',card_id=s['sides']['a']['hand'][0]['instance_id'])
        r=self.guide(s,baseline);after=apply_action(s,'a',r['selected'])
        self.assertGreater(hero(after,'a','zhenhong').get('surplus_passive_triggers',0),0)
    def test_two_action_delay_and_red_charge_line_is_found(self):
        s=self.game(['R02']);s['sides']['a']['last_front']='yi';hero(s,'a','yi').update(harmony=2,beast_fangs=4)
        r=self.guide(s,dict(type='end_turn'));self.assertEqual(r['selected']['type'],'play_card')
        after=apply_action(s,'a',r['selected']);self.assertTrue(front_debuff(after,'b').get('delay'))
        r2=self.guide(after,dict(type='end_turn'));after=apply_action(after,'a',r2['selected'])
        self.assertGreater(hero(after,'a','zhenhong').get('surplus_passive_triggers',0),0)
    def test_immediate_win_is_preserved(self):
        s=self.game(['I04']);s['sides']['b'].update(hp=1,shield=0);a=dict(type='attack',character_id='zero')
        r=self.guide(s,a);self.assertEqual(r['selected'],a);self.assertEqual(r['reason'],'preserve_winning_action')
    def test_unprotected_fatal_red_attack_is_not_used_for_setup(self):
        s=self.game(['R08']);s['sides']['b']['front']='zero'
        foe=hero(s,'b','zero');foe['base_stats']['attack']=20;foe.update(hp=50,max_hp=50)
        baseline=dict(type='attack',character_id='zhenhong')
        self.assertEqual(hero(apply_action(s,'a',baseline),'a','zhenhong')['hp'],0)
        r=self.guide(s,baseline)
        self.assertGreater(hero(apply_action(s,'a',r['selected']),'a','zhenhong')['hp'],0)
    def test_unknown_opponent_card_order_does_not_change_advice(self):
        s=self.game(['I04']);b=s['sides']['b'];b['hand']=b['deck'][:5];b['deck']=b['deck'][5:]
        t=deepcopy(s);pool=t['sides']['b']['hand']+t['sides']['b']['deck'];pool.reverse();t['sides']['b']['hand']=pool[:5];t['sides']['b']['deck']=pool[5:]
        self.assertEqual(observe(s,'a',include_previews=False),observe(t,'a',include_previews=False))
        a=dict(type='attack',character_id='zero');left=self.guide(s,a);right=self.guide(t,a)
        self.assertEqual(left['selected'],right['selected']);self.assertEqual(left['best_rank'],right['best_rank'])
    def test_dead_or_ultimate_red_and_budget_keep_model(self):
        for mode in ('dead','ultimate','budget'):
            s=self.game(['I04']);a=dict(type='end_turn')
            if mode=='dead':hero(s,'a','zhenhong')['hp']=0
            if mode=='ultimate':hero(s,'a','zhenhong')['awakened']=True
            r=recommend(s,'a',a,seconds=0 if mode=='budget' else 5)
            self.assertEqual(r['selected'],a);self.assertFalse(r['changed'])


class ZhenhongOpeningGuideTest(unittest.TestCase):
    def game(self, cards, side='a'):
        deck=next(d for d in STARTER_DECKS if d['id']=='zhenhong')
        s=new_game(seed=725,first_side=side,skip_mulligan=False,decks={'a':deck,'b':deck})
        while acting_side(s)!=side:s=apply_action(s,acting_side(s),dict(type='mulligan',card_ids=[]))
        team=s['sides'][side];team['deck']+=team['hand'];team['hand']=[]
        for key in cards:
            c=next(c for c in team['deck'] if c['card_id']==key);team['deck'].remove(c);team['hand'].append(c)
        self.assertEqual(len(team['hand']),5)
        return s
    def repair(self,s,side,keys):
        ids=[c['instance_id'] for c in s['sides'][side]['hand'] if c['card_id'] in keys][:3]
        baseline=dict(type='mulligan',card_ids=ids);self.assertIn(baseline,legal_actions(s,side))
        before=deepcopy(s);r=recommend(s,side,baseline)
        self.assertEqual(s,before);self.assertIn(r['selected'],legal_actions(s,side));self.assertEqual(r['worlds'],0)
        return r
    def test_fangs_and_zero_battle_are_both_retained_in_both_seats(self):
        for side in ('a','b'):
            s=self.game(['I01','Z04','R02','Y03','Y06'],side)
            r=self.repair(s,side,('I01','Z04','Y06'))
            self.assertEqual(r['selected']['card_ids'],[c['instance_id'] for c in s['sides'][side]['hand'] if c['card_id']=='Y06'])
            after=apply_action(s,side,r['selected']);held={c['instance_id'] for c in after['sides'][side]['hand']}
            self.assertTrue(all(c['instance_id'] in held for c in s['sides'][side]['hand'] if c['card_id'] in ('I01','Z04')))
    def test_keep_yi_battle_to_pair_with_zero_battle(self):
        s=self.game(['I04','I03','I01','Z03','R02']);r=self.repair(s,'a',('I04','I03','I01'))
        self.assertEqual({c['card_id'] for c in r['protected']},{'I01','Z03'})
        outgoing={c['card_id'] for c in s['sides']['a']['hand'] if c['instance_id'] in r['selected']['card_ids']}
        self.assertEqual(outgoing,{'I04','I03'})
    def test_free_fangs_preference_without_zero_battle(self):
        s=self.game(['I04','I03','I01','R02','Y03']);r=self.repair(s,'a',('I04','I03','I01'))
        self.assertEqual({c['card_id'] for c in r['protected']},{'I04'})
    def test_draw_backup_is_kept_when_core_is_missing(self):
        for keys,backup in [(['I01','R01','R02','Y03','Y06'],'R01'),(['Z04','Y02','R02','Y03','Y06'],'Y02')]:
            s=self.game(keys);r=self.repair(s,'a',tuple(keys[:3]));self.assertIn(backup,{c['card_id'] for c in r['protected']})
    def test_backup_is_optional_when_both_primary_pieces_are_present(self):
        s=self.game(['I01','Z04','R01','Y03','Y06']);r=self.repair(s,'a',('R01','Y03','Y06'))
        self.assertFalse(r['changed'])
    def test_no_candidate_does_not_invent_a_keep_or_read_opponent(self):
        s=self.game(['R02','R04','R06','Y03','Y06']);r=self.repair(s,'a',('R02','R04','R06'))
        self.assertFalse(r['changed']);self.assertEqual(r['protected'],[])
