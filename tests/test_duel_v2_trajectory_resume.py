import gzip,json,tempfile,time,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from app.modules.card_game.rl.information_search import search_game,search
from app.modules.card_game.rl import ten_search_runtime as runtime
from app.modules.card_game.content.duel_v2 import STARTER_DECKS

class Policy:
    version='frozen-test-model'
    def scores_value(self,x,c):return np.zeros(len(c)),0.

class ResumeTest(unittest.TestCase):
    def test_real_game_resume_preserves_actions_labels_and_rng(self):
        for algorithm in ('puct','gumbel'):
            with self.subTest(algorithm=algorithm):
                self.check_resume(algorithm)

    def check_resume(self,algorithm):
        # Deterministic cheap search selects a distribution over actual legal moves.
        # Rule progression, state serialization, budget and action RNG are real.
        def cheap(s,side,policies,**kw):
            actions,(x,c)=runtime.decision(s,side);p=np.ones(len(actions),np.float32)/len(actions)
            return dict(complete=True,actions=actions,x=x,c=c,pi=p,search_choice=0,raw_choice=0,
                        roots_covered=len(actions),nodes=1,simulations=kw['simulations'],value=0.)
        b=next(d for d in STARTER_DECKS if d['id']=='zhenhong')
        job=dict(search_algorithm=algorithm,runtime='ten',seed=18191,first='b',decks={'a':b,'b':b},policies={'a':('same','zhenhong'),'b':('same','zhenhong')},
                 deadline=time.time()+60,training=True,collect=True,record_episode=True,train_sides=['a','b'],
                 search_mix=dict(low=2,high=8,deep_probability=.25),max_actions=400)
        with tempfile.TemporaryDirectory() as folder,patch.object(runtime,'model',return_value=Policy()),patch('app.modules.card_game.rl.information_search.search',side_effect=cheap):
            full=search_game(job);self.assertTrue(full['complete'])
            path=Path(folder)/'progress.json.gz';sliced={**job,'resume_path':str(path),'slice_actions':3}
            pauses=0
            while True:
                part=search_game(sliced)
                if part['complete']:break
                pauses+=1;self.assertTrue(part['resumable']);self.assertEqual(part['rows'],[])
                saved=json.loads(gzip.open(path,'rt').read());self.assertTrue(all('z' not in r for r in saved['progress']['rows']))
                self.assertLess(pauses,150)
            self.assertGreater(pauses,1);self.assertEqual(full['episode_actions'],part['episode_actions'])
            self.assertEqual(full['winner'],part['winner']);self.assertEqual(full['decisions'],part['decisions'])
            self.assertEqual(len(full['rows']),len(part['rows']));self.assertEqual(len(full['value_rows']),len(part['value_rows']))
            for left,right in zip(full['rows']+full['value_rows'],part['rows']+part['value_rows']):
                self.assertEqual(left.keys(),right.keys())
                for k in left:
                    if isinstance(left[k],np.ndarray):np.testing.assert_array_equal(left[k],right[k])
                    else:self.assertEqual(left[k],right[k])
            with self.assertRaises(ValueError):search_game({**sliced,'deadline':job['deadline']+1})
            with self.assertRaises(ValueError):search_game({**sliced,'seed':job['seed']+1})
            with self.assertRaises(ValueError):search_game({**sliced,'search_algorithm':'gumbel' if algorithm=='puct' else 'puct'})
            with patch.object(Policy,'version','different-model'):
                with self.assertRaises(ValueError):search_game(sliced)

    def test_expired_root_checkpoint_has_no_unfinished_value_row(self):
        b=next(d for d in STARTER_DECKS if d['id']=='zhenhong')
        job=dict(runtime='ten',seed=9,first='a',decks={'a':b,'b':b},policies={'a':('same','zhenhong'),'b':('same','zhenhong')},deadline=time.time()+30,
                 training=True,collect=True,record_episode=True,train_sides=['a','b'],search_mix=dict(low=2,high=8,deep_probability=1))
        with tempfile.TemporaryDirectory() as folder,patch.object(runtime,'model',return_value=Policy()),patch('app.modules.card_game.rl.information_search.search',return_value={'complete':False}):
            path=Path(folder)/'p.json.gz';r=search_game({**job,'resume_path':str(path)})
            d=json.loads(gzip.open(path,'rt').read())['progress'];self.assertEqual(d['value_rows'],[]);self.assertEqual(d['decisions'],0)
            expected=np.random.default_rng(job['seed'] ^ 0x5A17).bit_generator.state
            self.assertEqual(expected,d['budget_rng']);self.assertEqual(r['rows'],[])

class EncodedInferenceTest(unittest.TestCase):
    def test_optimized_search_matches_original_and_reduces_observations(self):
        from app.modules.card_game.rl import outcome_runtime as rt
        from app.modules.card_game.rl.fixed_lineup import tensor_shapes
        from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side
        b=next(d for d in STARTER_DECKS if d['id']=='zhenhong');r=np.random.default_rng(51)
        model=rt.OutcomeModel.__new__(rt.OutcomeModel);model.hidden=8
        model.weights={n:r.normal(0,.1,shape).astype(np.float32) for n,shape in tensor_shapes(8).items()}
        model.weights.update({'wdl.weight':r.normal(0,.1,(3,9)).astype(np.float32),'wdl.bias':r.normal(0,.1,3).astype(np.float32)})
        state=new_game(seed=912,decks={'a':b,'b':b})
        for i in range(3):
            side=acting_side(state)
            with patch.object(rt,'value_for',wraps=rt.value_for) as count:
                old=search(state,side,{'a':model,'b':model},seed=915,simulations=40,runtime=rt,reuse_inference=False)
                before=count.call_count
            with patch.object(rt,'value_for',wraps=rt.value_for) as count:
                new=search(state,side,{'a':model,'b':model},seed=915,simulations=40,runtime=rt,reuse_inference=True)
                self.assertLess(count.call_count,before)
            np.testing.assert_array_equal(old['visits'],new['visits']);np.testing.assert_array_equal(old['mean_values'],new['mean_values'])
            self.assertEqual(old['value'],new['value']);self.assertEqual(old['search_choice'],new['search_choice'])
            state=apply_action(state,side,old['actions'][old['search_choice']])

class SchedulingTest(unittest.TestCase):
    def test_paused_jobs_keep_same_identity_and_do_not_create_extra_games(self):
        import sys
        from types import SimpleNamespace
        from concurrent.futures import ThreadPoolExecutor
        from collections import Counter
        from app.modules.card_game.rl import selfplay_trial as trial
        from app.modules.card_game.rl.episode_replay import EpisodePool
        calls=[];seen=Counter()
        def game(job):
            calls.append(deepcopy(job));seen[job['id']]+=1
            if seen[job['id']]==1:return dict(complete=False,resumable=True,reason='slice_limit',decisions=1,rows=[])
            return dict(complete=True,resumable=False,decisions=1,searched=1,rows=[],value_rows=[],winner='a',
                episode_actions=[{'side':'a','action':{'type':'end_turn'}}],search_stats=[{'policy_label':False,'simulations':1}])
        fake_torch=SimpleNamespace(save=lambda data,path:Path(path).write_text('{}'))
        dummy=SimpleNamespace(state_dict=lambda:{})
        base={k:SimpleNamespace(serving_deck={'id':k}) for k in ('zhenhong',*trial.FOES)}
        with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(max_workers=3) as executor,patch.dict(sys.modules,{'torch':fake_torch}),patch.object(trial,'search_game',side_effect=game):
            out=Path(d);steps={a:0 for a in trial.ARMS};totals={a:dict(games=0,samples=0,wins=0) for a in trial.ARMS}
            current,groups=trial.learn(executor,out,dict(smoke=True,train_until=time.time()+300,simulations=8,seeds={'foundation':100}),
                base,out,out,{a:dummy for a in trial.ARMS},{a:dummy for a in trial.ARMS},{a:EpisodePool() for a in trial.ARMS},steps,totals,lambda *a,**k:None,{})
            self.assertEqual(groups,3);self.assertEqual(set(seen.values()),{2})
            for eid in seen:
                pair=[j for j in calls if j['id']==eid];self.assertEqual(pair[0],pair[1])
            self.assertTrue(all(t['games']==1 and t['resumed_slices']==1 for t in totals.values()))
