import importlib.util
import time
import unittest
from copy import deepcopy
import numpy as np
from app.modules.card_game.rl.league_search import prove_turn, paired_evidence
from app.modules.card_game.rl.league_rollout import decision
from app.modules.card_game.rl.league_schema import deck
from app.modules.card_game.engine.duel_v2 import new_game,apply_action
from app.modules.card_game.engine.duel_v2.state import card_instance,hero


class SearchTest(unittest.TestCase):
    def game(self):
        s=new_game(seed=973000123,first_side='a',skip_mulligan=True,
                   decks={'a':deck('quick-rush'),'b':deck('starter')})
        s['sides']['a']['ap']=2
        s['sides']['b']['hp']=9;s['sides']['b']['shield']=0
        hero(s,'a','xiaozhi')['jingu']=3
        s['sides']['a']['hand']=[card_instance(s,'Q01','a')]
        return s

    def test_all_roots_and_two_action_win_without_peeking(self):
        s=self.game();before=deepcopy(s)
        one=prove_turn(s,'a',depth=1)
        two=prove_turn(s,'a',depth=2)
        self.assertFalse(one['paths']);self.assertTrue(two['paths'])
        self.assertEqual(two['roots_visited'],len(decision(s,'a')[0]))
        for path in two['paths'].values():
            t=s
            for action in path:t=apply_action(t,'a',action)
            self.assertEqual(t['winner'],'a')
        self.assertEqual(s,before)
        altered=deepcopy(s);altered['rng']=999;altered['sides']['b']['deck'].reverse()
        for c in altered['sides']['b']['hand']:c['card_id']='N02';c['response']='ally_attacked'
        other=prove_turn(altered,'a',depth=2)
        self.assertEqual(two['paths'],other['paths'])

    def test_possible_paid_response_and_deadline_fail_closed(self):
        s=self.game();s['sides']['b']['ap']=1
        self.assertFalse(prove_turn(s,'a')['eligible'])
        s['sides']['b']['ap']=0
        r=prove_turn(s,'a',deadline=time.time()-1)
        self.assertTrue(r['exhausted']);self.assertFalse(r['paths'])
        r=prove_turn(s,'a',node_budget=1)
        self.assertTrue(r['exhausted']);self.assertEqual(r['roots_visited'],1)

    def test_paired_confirmation_rejects_noise_truncation_and_mixed_gaps(self):
        def games(q):return [dict(complete=True,reward=x) for x in q]
        self.assertFalse(paired_evidence(games([10]*8),games([-10]*8))['accepted'])
        a=games([10]*32);b=games([10]*31+[-10])
        self.assertFalse(paired_evidence(a,b)['accepted'])
        b=games([-10]*8+[10]*24)
        self.assertTrue(paired_evidence(a,b)['accepted'])
        self.assertFalse(paired_evidence(b,a)['accepted'])
        b[0]['complete']=False
        self.assertFalse(paired_evidence(a,b)['accepted'])
        b=games([0,-10]+[10]*30)
        self.assertFalse(paired_evidence(a,b)['accepted'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch environment required')
    def test_full_candidate_distillation_teaches_low_probability_proof(self):
        import torch
        from app.modules.card_game.rl.league_search import distill_step
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([3.,1.,-4.]));self.v=torch.nn.Parameter(torch.tensor(2.))
            def forward(self,x,c,m):return (c*self.w).sum(-1).masked_fill(~m,-1e9),self.v.expand(x.shape[0])
        net=Tiny();opt=torch.optim.SGD(net.parameters(),lr=1)
        sample=dict(x=np.zeros(1,dtype=np.float32),c=np.eye(3,dtype=np.float32),targets=[2])
        for _ in range(30):distill_step(net,opt,[sample])
        self.assertEqual(int(net.w.argmax()),2);self.assertEqual(float(net.v.detach()),2)

    def test_curriculum_preserves_deck_accounting_and_disjoint_splits(self):
        from collections import Counter
        from app.modules.card_game.rl.search_pilot import position,corpus
        for key in ('starter','weave-rush','quick-rush'):
            s,side=position(key,973200005)
            own=s['sides'][side]
            self.assertEqual(Counter(c['card_id'] for c in own['hand']+own['deck'] if not c.get('derived')),Counter(deck(key)['card_ids']))
            self.assertFalse(any(c.get('derived') for c in own['deck']))
        seen=set()
        a,_,_=corpus('quick-rush',973300000,2,seen,time.time()+10)
        b,_,_=corpus('quick-rush',973300000,2,seen,time.time()+10)
        self.assertEqual(len(a),2);self.assertEqual(len(b),2)
        self.assertFalse({s['fingerprint'] for s in a}&{s['fingerprint'] for s in b})

    def test_searched_actions_cannot_masquerade_as_ppo_samples(self):
        from app.modules.card_game.rl.league_rollout import play
        with self.assertRaisesRegex(ValueError,'on-policy'):
            play({'tactical_search':True,'collect':True})

    def test_free_hidden_response_prevents_zero_ap_shortcut(self):
        s=new_game(seed=973900009,first_side='a',skip_mulligan=True,
                   decks={'a':deck('quick-rush'),'b':deck('quick-rush')})
        self.assertEqual(s['sides']['b']['ap'],0)
        self.assertFalse(prove_turn(s,'a')['eligible'])

    def test_all_actions_screened_then_only_one_challenger_gets_fresh_confirmation(self):
        from unittest.mock import patch
        from app.modules.card_game.rl.league_search import compare_root
        calls=[]
        def mapper(fn,jobs):
            jobs=list(jobs);calls.append(jobs)
            return [dict(complete=True,reward=10 if j['action']['index']==2 else -10) for j in jobs]
        actions=[dict(type='attack',index=i) for i in range(3)]
        with patch('app.modules.card_game.rl.league_search.decision',return_value=(actions,(np.zeros(1),np.eye(3)))), patch('app.modules.card_game.rl.league_search.prove_turn',return_value={'paths':{}}):
            r=compare_root({},'a',{},[10,0,-10],seed=10,deadline=time.time()+10,map_jobs=mapper)
        self.assertEqual(r['targets'],[2]);self.assertTrue(r['evidence']['accepted'])
        self.assertEqual(len(calls[0]),24);self.assertEqual(len(calls[1]),64)
        self.assertEqual({j['action']['index'] for j in calls[0]},{0,1,2})
        self.assertEqual({j['action']['index'] for j in calls[1]},{0,2})
        self.assertFalse({j['seed'] for j in calls[0]} & {j['seed'] for j in calls[1]})

    def test_pilot_quality_rejects_forgetting_and_match_regression(self):
        from app.modules.card_game.rl.search_pilot import quality_report
        keys=('starter','weave-rush','quick-rush')
        summary={k:{'scores':{'test':{'forgotten':0}},'heldout_gates':{'final':{'passed':True}}} for k in keys}
        rows=[dict(key=k,variant=v,wins=5,n=16) for k in keys for v in ('baseline','final')]
        self.assertTrue(quality_report(summary,rows)['passed'])
        summary['starter']['scores']['test']['forgotten']=1
        rows[-1]['wins']=4
        q=quality_report(summary,rows)
        self.assertFalse(q['passed']);self.assertEqual(len(q['reasons']),2)
        self.assertFalse(q['automatic_serving_approval'])
