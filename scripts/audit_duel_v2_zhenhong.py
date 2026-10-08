#!/usr/bin/env python3
"""Read-only sequence and harvesting audit of completed evaluation recordings."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.modules.card_game.rl.zhenhong_tactical_audit import audit_file
from app.modules.card_game.rl.fixed_lineup import identity,write


def worker(job):
    path,counterfactuals=job
    return audit_file(path,counterfactuals=counterfactuals)


def run(args):
    config=json.loads((args.source/'config.json').read_text())
    if config['rule_hash']!=identity():raise ValueError('Use the matching frozen rules for audit')
    folder=args.source/'evaluation';index=folder/'index.jsonl'
    rows=[json.loads(line) for line in index.read_text().splitlines()] if index.exists() else json.loads((folder/'index.json').read_text())
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation ids')
    groups=defaultdict(list)
    for row in rows:
        if not row['complete']:raise ValueError('Incomplete evaluation')
        groups[(row.get('kind'),row['cell'])].append(row)
    chosen=[]
    for group in groups.values():
        ordered=sorted(group,key=lambda r:(r['seed'],r['first'],r['id']))
        chosen.extend(ordered[:args.per_cell] if args.per_cell else ordered)
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for result in pool.map(worker,[(folder/'raw'/(r['id']+'.json.gz'),args.counterfactuals) for r in chosen]):results.append(result)
    totals=defaultdict(Counter)
    for row in results:
        for side,counts in row['counts'].items():
            variant=(row['row'] if side==row['first'] else row['col']) if row['kind']=='matrix' else row['variant']
            c=totals[str(row['kind'])+'/'+str(variant)];c.update(counts);c['perspectives']+=1;c['wins']+=int(row['winner']==side)
            c['games_started']+=int(counts.get('passive_triggers',0)>0)
            c['games_harvested_after_start']+=int(counts.get('later_zhenhong_attacks',0)>0)
    write(args.output/'cases.json',results)
    write(args.output/'summary.json',dict(rule_hash=identity(),source=str(args.source),total_index_games=len(rows),audited_games=len(results),
        per_cell=args.per_cell,counterfactuals=args.counterfactuals,metrics={k:dict(v) for k,v in totals.items()},
        no_training_use=True,no_empty_front_penalty=True,activation_opportunities_are_not_automatic_errors=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--per-cell',type=int,default=0);p.add_argument('--counterfactuals',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=8 or a.per_cell<0:p.error('Invalid audit bound')
    run(a)
