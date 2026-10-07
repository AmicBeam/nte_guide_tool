"""Public setup guidance respects resources, safety and hidden information."""
from copy import deepcopy
import unittest
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
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
