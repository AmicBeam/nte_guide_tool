#!/usr/bin/env python3
"""Bounded end-to-end resident throughput benchmark; no reusable weights saved."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,nargs='+',default=[1024,2048,4096])
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from app.modules.card_game.rl.gpu_duel.env import GpuDuelEnv
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer,collect_rollout,ppo_update,rollout_to_batch
    from app.modules.card_game.rl.gpu_duel.resident_obs import resident_observe
    torch.set_num_threads(1)
    results=[]
    for deck in ('starter','weave-rush'):
        for n in args.n:
            torch.manual_seed(123)
            env=GpuDuelEnv(n,device='cuda',deck=deck,compiled_backend='cuda')
            try:
                env.reset(seeds=list(range(n)))
                state,cand,_=resident_observe(env.state)
                net=CompactScorer(state.shape[-1],cand.shape[-1],hidden=128).cuda()
                opt=torch.optim.Adam(net.parameters(),lr=3e-4)
                torch.cuda.reset_peak_memory_stats()
                for trial in range(3):
                    start=time.perf_counter()
                    rollout=collect_rollout(env,net,args.horizon,observe_fn=resident_observe)
                    torch.cuda.synchronize();collected=time.perf_counter()
                    batch=rollout_to_batch(rollout)
                    ppo_update(net,opt,batch,epochs=2,minibatch=1024)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    row=dict(deck=deck,n=n,trial=trial,horizon=args.horizon,seconds=elapsed,
                             collect_seconds=collected-start,steps_per_second=n*args.horizon/elapsed,
                             peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
                    results.append(row);print(json.dumps(row),flush=True)
                    del rollout,batch
                del net,opt,state,cand
            finally:env.compiled.close()
            del env
            torch.cuda.empty_cache()
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
