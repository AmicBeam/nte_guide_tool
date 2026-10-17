"""CPU partition oracle for Gumbel32, never CUDA width/capacity evidence.

The independent fixture executes each partition at its actual padded width.
Production's device-only lane selector/scatter needs its separate CUDA test.
"""
from dataclasses import fields,is_dataclass,replace
import unittest


def partition_fixture(fixture,width):
    """Install bounded gather/evaluate/scatter around a synthetic tensor tree."""
    import torch as t
    from tests.test_duel_v2_device_search import TensorTreeFixture
    from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch,DeviceDecision
    names=('metadata_device','decision_device','sample_world_device','step_device',
           'clean_device','value_observation_device','release_device')
    counts=dict(tails=0,active_rows=0)
    def select(value,ids,valid):
        if isinstance(value,t.Tensor):return value.index_select(0,ids)
        if isinstance(value,DeviceStateBatch):
            return replace(value,slots=t.where(valid,value.slots[ids],-1),
                generations=t.where(valid,value.generations[ids],0),
                owners=t.where(valid,value.owners[ids],0),active=value.active[ids]&valid,
                errors=t.where(valid,value.errors[ids],0))
        if isinstance(value,DeviceDecision):
            return type(value)(**{f.name:select(getattr(value,f.name),ids,valid) for f in fields(value)})
        return value
    def merge(parts):
        first=parts[0][0]
        if isinstance(first,t.Tensor):return t.cat([value[:n] for value,n in parts])
        if is_dataclass(first):
            return type(first)(**{f.name:merge([(getattr(value,f.name),n) for value,n in parts])
                                  for f in fields(first)})
        return fixture if isinstance(first,TensorTreeFixture) else first
    for name in names:
        def invoke(states,*args,_name=name,**kwargs):
            # CPU-only shortcut for an entirely inactive capture site. CUDA
            # production uses _ActiveSoALanes.pending to skip this same work.
            if not bool(states.active.any()):
                return getattr(TensorTreeFixture,_name)(fixture,states,*args,**kwargs)
            parts=[]
            for start in range(0,fixture.rows,width):
                count=min(width,fixture.rows-start)
                ids=t.arange(start,start+width).clamp_max(fixture.rows-1)
                valid=t.arange(width)<count
                local=select(states,ids,valid)
                view=object.__new__(TensorTreeFixture)
                view.__dict__.update({k:v for k,v in fixture.__dict__.items() if k not in names})
                view.rows=width
                for key in ('viewer','base','cursor','sample_seeds'):
                    setattr(view,key,getattr(fixture,key)[ids].clone())
                output=getattr(TensorTreeFixture,_name)(view,local,
                    *[select(arg,ids,valid) for arg in args],**kwargs)
                fixture.cursor[ids[:count]]=view.cursor[:count]
                fixture.sample_seeds[ids[:count]]=view.sample_seeds[:count]
                counts['tails']+=count<width
                counts['active_rows']+=int((local.active & (local.errors==0)).sum())
                if output is not None:parts.append((output,count))
            return merge(parts) if parts else None
        setattr(fixture,name,invoke)
    return counts


class WideSoASearchOracleTests(unittest.TestCase):
    def test_full32_distinct_roots_tail_inactive_error_rng_and_leases(self):
        import numpy as np
        import torch as t
        from tests.test_duel_v2_device_search import TensorTreeFixture,FixtureInference,InjectiveFixtureKeys
        from app.modules.card_game.rl.batched_search.device_random import DevicePCG64
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        from app.modules.card_game.rl.batched_search.device_statistics import DeviceStatisticsPool
        t.set_num_threads(1);results=[];random_states=[]
        rows=65
        for width in (None,2,8,32,64):
            fixture=TensorTreeFixture(rows,actions=3)
            counts=None if width is None else partition_fixture(fixture,width)
            bank=DevicePCG64([np.random.PCG64(913+i).state for i in range(rows)],device='cpu',
                max_rows=rows,max_draws=3,max_choices=3,allow_cpu_reference=True)
            self.addCleanup(bank.close)
            stats=DeviceStatisticsPool(rows*33,3,device='cpu',max_batch=rows,max_path_len=4,allow_cpu_reference=True)
            search=DeviceGumbelSearch(fixture,FixtureInference(),FixtureInference(),stats,InjectiveFixtureKeys(),
                rows=rows,max_actions=3,simulations=32,max_depth=4,terminal_horizon=0,
                teacher_mode='network_value',opponent_mode='policy',random=bank,device='cpu',
                allow_cpu_reference=True,allow_synthetic=True,group_ready_leaves=True,
                reuse_interior_observation=True,reuse_root_observation=True,
                reuse_sampled_world=True,reuse_root_decision=True)
            roots=fixture.roots();roots.active[2]=False;roots.active[33]=False;roots.errors[3]=77
            seeds=t.zeros((rows,5),dtype=t.int64);seeds[:,0]=t.arange(rows)+1700;seeds[:,4]=1
            ids=t.arange(rows*2,dtype=t.int64).reshape(rows,2)
            search.initialize(roots,fixture.viewer,seeds,ids,ids+100)
            for _ in range(2000):
                if not bool(search.unfinished_device.any()):break
                search.tick()
            else:self.fail('Bounded Gumbel32 partition oracle did not finish')
            result=search.result_device()
            expected=t.full((rows,),32,dtype=t.int64);expected[[2,3,33]]=0
            t.testing.assert_close(result.completed,expected)
            self.assertEqual(int(fixture.live.sum()),rows,width)
            self.assertEqual(result.errors[3].item(),77)
            t.testing.assert_close(fixture.data[roots.slots],t.tensor([[0,-1]]*rows))
            results.append({name:getattr(result,name).clone() for name in (
                'actions','policy','visits','information_keys','errors','completed','requested','complete')})
            random_states.append(bank.export_numpy_states())
            if counts is not None:
                self.assertGreater(counts['tails'],0);self.assertGreater(counts['active_rows'],0)
        for output in results[1:]:
            for name,value in results[0].items():t.testing.assert_close(value,output[name],rtol=0,atol=0)
        self.assertTrue(all(state==random_states[0] for state in random_states[1:]))


if __name__=='__main__':unittest.main()
