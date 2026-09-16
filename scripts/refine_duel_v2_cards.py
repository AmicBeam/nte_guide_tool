#!/usr/bin/env python3
"""Explicit bounded one/two-card local search with frozen battle policies."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--initial',type=Path)
    p.add_argument('--policy-checkpoint',type=Path,required=True)
    p.add_argument('--opponent-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--two-limit',type=int,default=128)
    p.add_argument('--screen-pairs',type=int,default=8)
    p.add_argument('--final-pairs',type=int,default=128)
    p.add_argument('--seed',type=int,default=420000000)
    a=p.parse_args(argv)
    if not 1<=a.seconds<=3600:p.error('seconds must be 1..3600')
    if a.output.exists():p.error('Use a new output directory')
    from app.modules.card_game.rl.card_tuning import load_battle_policy,CardBattleEvaluator
    from app.modules.card_game.rl.candidate_selection import evaluate_tensor_candidates
    from app.modules.card_game.rl.local_card_search import refine_locally
    baseline=json.loads(a.baseline.read_text(encoding='utf-8'))
    initial=json.loads(a.initial.read_text(encoding='utf-8')) if a.initial else baseline
    own,own_id=load_battle_policy(a.policy_checkpoint,'cuda')
    peer,peer_id=load_battle_policy(a.opponent_checkpoint,'cuda')
    evaluator=CardBattleEvaluator(own,opponents=[peer],device='cuda')
    start=time.monotonic()
    def stop():
        if time.monotonic()-start>=a.seconds:raise TimeoutError('Local search budget expired')
    try:
        result=refine_locally(baseline,lambda decks,plan:evaluate_tensor_candidates(evaluator,decks,plan,stop_check=stop),
            initial=initial,rounds=a.rounds,two_limit=a.two_limit,screen_pairs=a.screen_pairs,final_pairs=a.final_pairs,seed_base=a.seed,stop_check=stop)
    finally:evaluator.close()
    result.update(policy=own_id,opponent=peer_id,seconds=time.monotonic()-start,optimizer_updates=0)
    a.output.mkdir(parents=True)
    (a.output/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.output/'candidate.json').write_text(json.dumps(result['build'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('complete','selection','completed_rounds','seconds')},ensure_ascii=False))
    return 0 if result['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
