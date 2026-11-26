"""Bounded real-CUDA direct-domain component check; never a game benchmark.

Default is plan-only. --run measures synthetic 400-row storage/action/zone/damage
fixtures. No models, optimizer, VM, game outcomes or full-rule qualification.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',action='store_true');p.add_argument('--output',type=Path)
 p.add_argument('--roots',type=int,default=400);p.add_argument('--repeats',type=int,default=50)
 p.add_argument('--copy-backend',choices=('direct','tensor'),default='direct')
 a=p.parse_args(argv)
 if not 1<=a.roots<=400 or not 1<=a.repeats<=200:p.error('roots1..400 / repeats1..200')
 if not a.run:
  print(json.dumps(dict(scope='synthetic CUDA components only',roots=a.roots,
   full_game_benchmark=False,full_duel_qualified=False,vm=False,jax=False)))
  return 0
 if a.output is None:p.error('--run requires a new --output directory')
 if a.output.exists():p.error('output must be new')
 a.output.mkdir(parents=True)
 started=time.monotonic();report=dict(passed=False,roots=a.roots,vm=False,jax=False,
  full_duel_qualified=False,full_game_benchmark=False,scope='synthetic CUDA components only',measurements={},copy_backend=a.copy_backend)
 kernel=None;pool=None
 try:
  import torch as t
  import numpy as np
  from dataclasses import fields
  from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout,array_specs
  from app.modules.card_game.rl.batched_duel.domain_soa_state_pool import DomainSoAStatePool
  from app.modules.card_game.rl.batched_duel.domain_soa_transition import FieldSpec
  from app.modules.card_game.rl.batched_duel.domain_soa_action_batch import DomainSoAActionBatch,DomainActionJobResults
  from app.modules.card_game.rl.batched_duel.domain_soa_cuda_damage import DomainCudaDamage
  from app.modules.card_game.rl.batched_duel.domain_soa_rules import damage_fields
  from app.modules.card_game.rl.batched_duel.domain_soa_zones import move_card
  if not t.cuda.is_available():raise RuntimeError('Actual CUDA required; no CPU fallback')
  t.set_num_threads(1);t.backends.cuda.matmul.allow_tf32=False
  device=t.device('cuda',t.cuda.current_device());b=a.roots
  specs=tuple(FieldSpec(n,np.dtype(dtype).name,tail)for n,(dtype,tail)in array_specs(DomainSoALayout()).items())
  pool=DomainSoAStatePool(b*33,device=device,specs=specs,fused_cuda_copy=a.copy_backend=='direct')
  initial={spec.name:t.zeros((b,)+spec.tail,dtype=pool.columns[spec.name].dtype,device=device)for spec in specs}
  initial['player_hp'][:,0]=t.arange(b,device=device,dtype=t.int32)%30+1
  owners=t.arange(b,device=device,dtype=t.int64)+1;active=t.ones(b,dtype=t.bool,device=device)
  roots=pool.put_device(initial,owners,active)
  report.update(persistent_state_bytes=pool.allocated_bytes,state_fields=len(specs),packed_storage_groups=len(pool._packed))
  def memory_guard():
   free,total=t.cuda.mem_get_info()
   if total-free>14336*1024**2:raise MemoryError('Whole GPU memory guard 14336MiB')
   if time.monotonic()-started>120:raise TimeoutError('120s component budget')
  def measure(name,operation,repeats):
   for _ in range(3):operation()
   t.cuda.synchronize();memory_guard()
   graph=t.cuda.CUDAGraph()
   capture_start=time.monotonic()
   with t.cuda.graph(graph):result=operation()
   t.cuda.synchronize();capture=time.monotonic()-capture_start
   samples=[]
   for _ in range(3):
    begin=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):graph.replay()
    end.record();end.synchronize();samples.append(begin.elapsed_time(end)*1000/repeats)
    memory_guard()
   report['measurements'][name]=dict(microseconds_per_batch=samples,repeats=repeats,capture_seconds=capture)
   return result
  def fork():
   child=pool.clone_device(roots);pool.release_device(child);return child.errors
  error=measure('fork_and_release_full_field_states',fork,a.repeats)
  if bool(error.any().cpu()):raise AssertionError('State fork/release lease error')
  observed=pool.gather_device(roots)
  if not all(t.equal(observed[k],v)for k,v in initial.items()):raise AssertionError('Parent state changed')
  kernel=DomainCudaDamage(device=device)
  hp=t.full((b,2,4),6,dtype=t.int32,device=device);shield=t.ones_like(hp)
  amount=t.arange(b*8,device=device,dtype=t.int32).reshape(b,2,4)%9
  bit=t.zeros_like(hp,dtype=t.bool);on=t.ones_like(bit)
  args=(hp,shield,amount,bit,bit,bit,t.zeros_like(hp),t.full_like(hp,-1),on)
  batch=kernel.prepare(*args);expected=damage_fields(*args)
  def damage():kernel.launch(batch);return batch.result
  result=measure('fused_prepared_damage',damage,a.repeats*5)
  if not all(t.equal(getattr(result,f.name),getattr(expected,f.name))for f in fields(expected)):
   raise AssertionError('Direct damage differs from tensor oracle')
  arena=DomainSoAActionBatch([(1,1)],max_roots=b,max_candidates=800,device=device)
  tokens=t.ones((b,800),dtype=t.int32,device=device);counts=t.full((b,),800,dtype=t.int32,device=device)
  mask=t.ones_like(tokens,dtype=t.bool);selected=t.zeros_like(mask);selected[:,799]=True
  generation=t.ones(b,dtype=t.int64,device=device)
  def action_batch():
   q=arena.group_selected_actions(tokens,counts,mask,selected,active,owners,generation)
   result=DomainActionJobResults(q.present,q.candidate_ordinal,q.root_index,q.action_index,
    q.owners,q.generations,q.handler_id,q.candidate_ordinal,q.epoch)
   return arena.scatter_results(q,result)
  action_result=measure('group_and_scatter_800_legal_candidates',action_batch,max(1,a.repeats//5))
  if bool(action_result.errors.any().cpu())or not bool(action_result.present[:,799].all().cpu()):
   raise AssertionError('Action queue identity/order error')
  cards=t.full((b,2,4,96),-1,dtype=t.int32,device=device);lengths=t.zeros((b,2,4),dtype=t.int32,device=device)
  cards[:,0,0,:3]=t.arange(3,device=device,dtype=t.int32);cards[:,0,1,:4]=t.arange(3,7,device=device,dtype=t.int32)
  lengths[:,0,0]=3;lengths[:,0,1]=4
  zero=t.zeros(b,dtype=t.int64,device=device);one=t.ones_like(zero);three=t.full_like(zero,3)
  result=measure('ordered_zone_move',lambda:move_card(cards,lengths,zero,one,zero,zero,three,active,max_cards=192),a.repeats)
  if bool(result.errors.any().cpu())or not bool((result.cards[:,0,0,3]==3).all().cpu()):
   raise AssertionError('Zone ordering/identity error')
  report.update(passed=True,action_workspace_bytes=arena.capacity_bytes,
   peak_torch_reserved_bytes=t.cuda.max_memory_reserved(),device_name=t.cuda.get_device_name())
 except BaseException as e:
  import traceback
  report.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  if kernel is not None:kernel.close()
  if pool is not None:pool.close()
  report['seconds']=time.monotonic()-started
  (a.output/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report))
 return 0 if report['passed']else 2


if __name__=='__main__':raise SystemExit(main())
