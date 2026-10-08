#!/usr/bin/env python3
"""Explicit synthetic node-pool CUDA validation; never a game or training run."""
from pathlib import Path
import argparse,json,time,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--roots',type=int,nargs='+',default=[64,256,512,1600])
    p.add_argument('--max-actions',type=int,default=256)
    p.add_argument('--seconds',type=float,default=120.)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args(argv)
    if not a.roots or len(a.roots)!=len(set(a.roots)) or any(n<1 or n>1600 for n in a.roots):p.error('Distinct root counts 1..1600 required')
    if not 3<=a.max_actions<=1024:p.error('Full-action capacity must be 3..1024')
    if not 0<a.seconds<=600:p.error('Finite positive seconds up to 600 required')
    if a.run and a.output is None:p.error('--run requires a new output directory')
    return a


def main(argv=None):
    a=parse_args(argv)
    config=dict(kind='synthetic_node_pool_only',roots=a.roots,max_actions=a.max_actions,simulations=32,
                device=a.device,seconds=a.seconds,game_executions=0,gradient_updates=0,full_five_team_validation=False)
    if not a.run:
        print(json.dumps(config));return
    if a.output.exists():raise ValueError('New output directory required; no implicit restart')
    a.output.mkdir(parents=True)
    import torch,numpy as np
    from app.modules.card_game.rl.batched_search.nodes import BatchedNodePool
    from app.modules.card_game.rl import search_policy,recovery_search
    from app.modules.card_game.rl.information_search import Node
    torch.set_num_threads(1)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refuse CPU fallback')
    def write(name,data):
        p=a.output/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
    config.update(torch=torch.__version__,device_name=torch.cuda.get_device_name(0) if a.device=='cuda' else 'CPU oracle')
    write('config.json',config)
    start=time.monotonic();absolute=time.time()+a.seconds;results=[]
    for n in a.roots:
        if time.monotonic()-start>=a.seconds or time.time()>=absolute:raise TimeoutError('Synthetic validation deadline')
        for mode in ('legacy','natural_wdl'):
            if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter();pool=BatchedNodePool(n*33,a.max_actions,device=a.device,q_scale=mode)
            if a.device=='cuda':torch.cuda.synchronize()
            allocation_wall=time.perf_counter()-begin
            priors=[np.array([.1,.3,.6],dtype=np.float64)]*n
            refs=pool.allocate(priors,[.2]*n,['root-'+str(i) for i in range(n)])
            reference=Node.create(priors[0].copy(),.2);noise=np.zeros((n,a.max_actions))
            schedule=search_policy.visit_schedule(3,32)
            begin=time.perf_counter();device_start=device_end=None
            if a.device=='cuda':
                device_start=torch.cuda.Event(enable_timing=True);device_end=torch.cuda.Event(enable_timing=True);device_start.record()
            for index in range(32):
                if time.monotonic()-start>=a.seconds or time.time()>=absolute:raise TimeoutError('Synthetic validation deadline')
                selected=pool.select_root(refs,noise,[schedule[index]]*n)
                expected=recovery_search.choose_root(reference,np.zeros(3),schedule[index],mode)
                actual=selected.cpu().numpy()
                if not np.all(actual==expected):raise ValueError('CUDA/NumPy root choice mismatch')
                value=(-.75,.125,.875)[index%3]
                pool.backup([[(ref,int(action))] for ref,action in zip(refs,actual)],[value]*n)
                reference.visits[expected]+=1;reference.total[expected]+=value
            if a.device=='cuda':device_end.record();device_end.synchronize()
            wall=time.perf_counter()-begin
            q=pool.completed_q(refs).cpu().numpy()[:,:3]
            pi=pool.policy(refs).cpu().numpy()[:,:3]
            np.testing.assert_allclose(q,np.broadcast_to(recovery_search.completed_q(reference,mode),(n,3)),rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(pi,np.broadcast_to(recovery_search.improved_policy(reference,mode),(n,3)),rtol=1e-10,atol=1e-10)
            row=dict(roots=n,q_scale=mode,node_capacity=pool.capacity,statistics_bytes=pool.allocated_bytes,
                     allocation_wall_seconds=allocation_wall,simulations_per_root=32,total_synthetic_simulations=n*32,
                     traversal_wall_seconds=wall,device_stream_milliseconds=device_start.elapsed_time(device_end) if device_start else None,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved() if a.device=='cuda' else None,
                     numpy_choices_policy_q_equal=True,game_executions=0,gradient_updates=0)
            results.append(row);write('progress.json',dict(phase='running',results=results));print(json.dumps(row),flush=True)
            pool.release(refs)
            if pool.active_nodes:raise ValueError('Node leak after release')
            del pool
            if a.device=='cuda':torch.cuda.empty_cache()
    write('result.json',dict(complete=True,kind=config['kind'],results=results,wall_seconds=time.monotonic()-start,
                            game_executions=0,gradient_updates=0,full_five_team_validation=False))
    write('progress.json',dict(phase='complete',results=results))


if __name__=='__main__':main()
