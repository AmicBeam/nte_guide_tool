"""Search refactor parity on real five-team rules, not a strength evaluation.

Policies are small deterministic public-observation test functions. No weights,
training samples, final-evaluation seeds or GPU fallback are used in these tests.
"""
from copy import deepcopy
import unittest
import numpy as np

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, acting_side
from app.modules.card_game.rl import recovery_runtime, recovery_search
from app.modules.card_game.rl.batched_search.oracle_backend import CPUOracleBackend
from app.modules.card_game.rl.batched_search.traversal import search_traversal
from app.modules.card_game.rl.batched_search.cooperative import (
    cooperative_search, CooperativeScheduler, RootJob,
)
from app.modules.card_game.rl.batched_search.inference import InferenceResult


class PublicTestPolicy:
    def __init__(self, key):
        self.version='public-test-function-v1-'+key
        self.manifest={'deck':key}

    def scores_only(self,x,c):
        weights=np.linspace(-.031,.047,c.shape[1],dtype=np.float64)
        result=np.einsum('ij,j->i',np.asarray(c,dtype=np.float64),weights,optimize=False)
        if not np.isfinite(result).all():raise ValueError('Non-finite unit-policy scores')
        return result

    def wdl(self,x,is_actor):
        delta=float(np.tanh(np.sum(x,dtype=np.float64)*.001))*.2
        if not is_actor:delta=-delta
        return np.array([.4-delta,.2,.4+delta],dtype=np.float64)

    def scores_value(self,x,c):
        prob=self.wdl(x,True)
        return self.scores_only(x,c),float(prob[2]-prob[0])


class NumpyOracleInference:
    def __init__(self,models):self.models=models
    def predict(self,requests):
        result=[]
        for request in requests:
            model=self.models[request.model_key]
            if request.model_version!=model.version:raise ValueError('Wrong oracle model')
            logits=(np.zeros(0) if request.value_only else
                    model.scores_only(request.x,request.candidates))
            result.append(InferenceResult(request.request_id,request.model_key,
                model.version,logits,model.wdl(request.x,request.value_actor)))
        return result


def direct(generator,models):
    try:
        query=next(generator)
        while True:
            model=models[query.policy_key]
            scores=np.zeros(0) if query.value_only else model.scores_only(query.x,query.c)
            query=generator.send((scores,model.wdl(query.x,query.value_actor)))
    except StopIteration as stop:
        return stop.value


class RealRuleTraversalParityTest(unittest.TestCase):
    def assert_result_equal(self,actual,expected):
        for key in ('actions','complete','simulations','requested','roots_covered',
                    'search_choice','raw_choice','nodes','worlds','max_depth'):
            self.assertEqual(actual[key],expected[key],key)
        for key in ('x','c','pi','visits','mean_values','visit_pi','root_prior',
                    'root_gumbel','terminal_checks','terminal_wins'):
            np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
        self.assertEqual(actual['value'],expected['value'])

    def test_all_five_actual_rule_roots_match_both_q_scales(self):
        decks={deck['id']:deck for deck in STARTER_DECKS}
        keys=('starter','weave-rush','quick-rush','zhenhong','murk')
        models={key:PublicTestPolicy(key) for key in keys}
        backend=CPUOracleBackend(runtime=recovery_runtime)
        jobs=[];expected={};states=[]
        for index,key in enumerate(keys):
            foe=keys[(index+1)%len(keys)]
            state=new_game(seed=7100+index,decks={'a':decks[key],'b':decks[foe]},
                           first_side='b' if index%2 else 'a',skip_mulligan=True)
            before=deepcopy(state);states.append((state,before))
            viewer=acting_side(state)
            policies={'a':models[key],'b':models[foe]}
            for mode in ('legacy','natural_wdl'):
                root=key+':'+mode
                options=dict(seed=7300+index,simulations=32,gumbel_candidates=16,
                             terminal_horizon=3,max_depth=10,noise=True,q_scale=mode)
                reference=recovery_search.search(state,viewer,policies,
                    runtime=recovery_runtime,**options)
                actual=direct(search_traversal(state,viewer,policies,
                    backend=backend,**options),models)
                self.assert_result_equal(actual,reference)
                expected[root]=reference
                jobs.append(RootJob(root,'game-'+root,1,cooperative_search(
                    state,viewer,policies,backend=backend,root_id=root,**options)))
        run=CooperativeScheduler(NumpyOracleInference(models),
            allow_cpu_oracle=True).run(jobs)
        self.assertEqual(run['root_count'],10)
        self.assertEqual(run['peak_outstanding_roots'],10)
        for root,result in run['results'].items():
            self.assert_result_equal(result,expected[root])
        for state,before in states:self.assertEqual(state,before)


if __name__=='__main__':unittest.main()
