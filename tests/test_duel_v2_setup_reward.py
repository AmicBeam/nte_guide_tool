import unittest,time
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.setup_reward import ready,coefficient,learning_return
from app.modules.card_game.rl import ten_search_runtime as runtime
from app.modules.card_game.rl.information_search import search_game,search
from tests.test_duel_v2_information_search import Uniform

class SetupRewardTest(unittest.TestCase):
    def test_real_sequence_only_counts_joint_own_delay_and_fangs(self):
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        from app.modules.card_game.engine.duel_v2 import new_game,apply_action
        deck=next(d for d in STARTER_DECKS if d['id']=='zhenhong')
        s=new_game(seed=81,first_side='a',skip_mulligan=True,decks={'a':deck,'b':deck})
        team=s['sides']['a'];team['ap']=3
        for card_id in ['Z04','I04']:
            if not any(c['card_id']==card_id for c in team['hand']):
                c=next(c for c in team['deck'] if c['card_id']==card_id);team['deck'].remove(c);team['hand'].append(c)
        foe=s['sides']['b'];foe['deck']+=foe['hand'];foe['hand']=[]
        ids={c['card_id']:c['instance_id'] for c in team['hand']}
        s=apply_action(s,'a',dict(type='play_card',card_id=ids['Z04']));self.assertFalse(ready(s,'a'))
        s=apply_action(s,'a',dict(type='attack',character_id='yi'));self.assertFalse(ready(s,'a'))
        s=apply_action(s,'a',dict(type='play_card',card_id=ids['I04']));self.assertTrue(ready(s,'a'))
        s['sides']['a']['characters']['yi']['hp']=0;self.assertTrue(ready(s,'a'))
        s['sides']['b']['front_debuff']['delay']['by']='b';self.assertFalse(ready(s,'a'))
        s['sides']['b']['front_debuff']['delay'].update(by='a',left=0);self.assertFalse(ready(s,'a'))

    def test_decay_and_replay_reweighting(self):
        self.assertEqual(coefficient(.1,0,64),.1)
        self.assertEqual(coefficient(.1,32,64),.05)
        self.assertEqual(coefficient(.1,64,64),0)
        self.assertEqual(coefficient(.1,128,64),0)
        r=dict(z=1,reward_to_go=10.4,setup_future=1,setup_reward=.1)
        self.assertAlmostEqual(learning_return(r),10.5)
        self.assertAlmostEqual(learning_return(r,.05),10.45)
        self.assertAlmostEqual(learning_return(r,0),10.4)
        with self.assertRaises(ValueError):coefficient(.1,0,0)

    def play(self,training=True,key='zhenhong',train_sides=('a',),limit=4):
        s={'phase':'playing','active_side':'a','turn':1,'sides':{'a':{'characters':{'zhenhong':{},'yi':{'beast_fangs':1}}},'b':{'characters':{},'front_debuff':{}}}}
        def act(state,side,action):
            n=deepcopy(state);n['turn']+=1
            n['sides']['b']['front_debuff']={'delay':{'by':'a','left':2}} if n['turn'] in (2,4,5) else {}
            if n['turn']==5:n.update(phase='finished',winner='a')
            return n
        result=dict(complete=True,actions=[{'type':'test'}],pi=np.array([1.],np.float32),x=np.zeros(1),c=np.zeros((1,1)),search_choice=0,raw_choice=0,roots_covered=1,nodes=1,simulations=1,value=0.)
        job=dict(runtime='ten',seed=1,deadline=time.time()+10,max_actions=limit,decks={},policies={'a':('unused',key),'b':('unused','starter')},training=training,collect=True,setup_rewards={'a':.1},train_sides=list(train_sides))
        with patch('app.modules.card_game.rl.information_search.new_game',return_value=s),patch.object(runtime,'model',return_value=Uniform()),patch('app.modules.card_game.rl.information_search.search',return_value=result),patch('app.modules.card_game.rl.information_search.apply_action',side_effect=act):return search_game(job)

    def test_first_achievement_only_and_credit_only_future(self):
        result=self.play();self.assertEqual([r['setup_future'] for r in result['rows']],[1,0,0,0])
        self.assertEqual([learning_return(r) for r in result['rows']],[10.1,10,10,10])
        self.assertEqual(result['setup_rewards']['a'],.1)
        self.assertEqual(self.play(limit=2)['rows'],[])

    def test_evaluation_other_team_and_frozen_actor_have_no_bonus(self):
        for options in [dict(training=False),dict(key='starter'),dict(train_sides=('b',))]:
            result=self.play(**options)
            self.assertTrue(all('setup_future' not in r for r in result['rows']))
            self.assertEqual(result['setup_rewards']['a'],0)

    def test_search_receives_setup_credit_without_changing_terminal_value(self):
        from types import SimpleNamespace
        s={'phase':'playing','active_side':'a','turn':1,'sides':{'a':{'characters':{'yi':{'beast_fangs':1}}},'b':{'characters':{},'front_debuff':{}}}}
        def decision(state,side):return [{'type':'setup'},{'type':'nothing'}],(np.zeros(1),np.zeros((2,1)))
        adapter=SimpleNamespace(decision=decision,determinize=lambda s,side,seed:deepcopy(s),passive_count=lambda s,side:0)
        def act(state,side,action):
            n=deepcopy(state);n.update(phase='finished',winner='draw')
            if action['type']=='setup':n['sides']['b']['front_debuff']={'delay':{'by':'a','left':2}}
            return n
        with patch('app.modules.card_game.rl.information_search.apply_action',side_effect=act):
            r=search(s,'a',{'a':Uniform(),'b':Uniform()},seed=5,simulations=32,runtime=adapter,setup_reward=.1)
            self.assertAlmostEqual(r['mean_values'][0],.01);self.assertEqual(r['mean_values'][1],0)
            r=search(s,'a',{'a':Uniform(),'b':Uniform()},seed=5,simulations=32,runtime=adapter,setup_reward=.1,setup_already_achieved=True)
            np.testing.assert_array_equal(r['mean_values'],[0,0])

    def test_decay_progress_survives_same_policy_warmstart(self):
        from pathlib import Path
        import tempfile,json
        from app.modules.card_game.rl.fixed_lineup import save_weights,tensor_shapes
        from app.modules.card_game.rl.ten_search_runtime import migrate_source
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS
        builds={d['id']:d for d in STARTER_DECKS}
        weights={k:np.zeros(shape,np.float32) for k,shape in tensor_shapes(1).items()}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            save_weights(weights,p/'source','zhenhong',builds['zhenhong'],{'setup_reward_updates':32})
            migrate_source(p/'source',p/'target',{'zhenhong':builds['zhenhong']})
            m=json.loads((p/'target/zhenhong.json').read_text())
            self.assertEqual(m['origin']['setup_reward_updates'],32)
            save_weights(weights,p/'starter-only','starter',builds['starter'],{'setup_reward_updates':55})
            migrate_source(p/'starter-only',p/'new-character',{'zhenhong':builds['zhenhong']})
            m=json.loads((p/'new-character/zhenhong.json').read_text())
            self.assertEqual(m['origin']['setup_reward_updates'],0)
