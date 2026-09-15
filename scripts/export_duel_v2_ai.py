#!/usr/bin/env python3
"""Export trusted training checkpoints to numeric CPU serving files + parity cases."""
import argparse, copy, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import torch
    from app.modules.card_game.rl.gpu_duel.catalog import CARD_IDS,SEATS,preset_by_id
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
    from app.modules.card_game.rl.gpu_duel.resident_obs import resident_observe
    from app.modules.card_game.rl.gpu_duel.state import empty_state
    from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
    from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row,bind_gpu_rows
    from app.modules.card_game.rl.rule_ir.compiled_oracle import map_python_action
    from app.modules.card_game.engine.duel_v2 import new_game,apply_action,acting_side,legal_actions
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    for deck_id in ('starter','weave-rush'):
        path=args.root/deck_id/'best.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        from app.modules.card_game.rl.capability import resolve_model_capability
        capability = resolve_model_capability(checkpoint, deck_id)
        if capability['human_public_custom']:
            raise ValueError('resident_public_v1 exporter only supports mirrors; public custom needs a matching export/observation contract')
        net=CompactScorer(checkpoint['state_dim'],checkpoint['cand_dim'],checkpoint['hidden'])
        net.load_state_dict(checkpoint['model']);net.eval()
        dest=args.output/f'{deck_id}.npz'
        np.savez_compressed(dest,**{k:v.numpy() for k,v in checkpoint['model'].items() if not k.startswith('value.')})
        manifest={k:checkpoint[k] for k in ('schema','deck','hidden','state_dim','cand_dim','gpu_lock','update','rule_identity')}
        manifest.update(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),card_ids=list(CARD_IDS),seat_ids=list(SEATS),
                        source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        capability=capability)
        (args.output/f'{deck_id}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        cases=[]
        backend=CompiledStarterBackend(args.output/'native-cache',deck_id=deck_id)
        try:
            tensor_state=empty_state(1,'cpu');rows=bind_gpu_rows(tensor_state)
            for seed in (123,456):
                state=new_game(seed=seed,skip_mulligan=True,decks={'a':preset_by_id(deck_id),'b':preset_by_id(deck_id)})
                rng=random.Random(seed)
                for step in range(24):
                    if state['phase']=='finished':break
                    side=acting_side(state)
                    actions=[a for a in legal_actions(state,side) if a['type']!='concede']
                    row=backend.step_lists([pack_python_row(state)],[-1])[0]
                    mapped=[map_python_action(row,state,a) for a in actions]
                    rows.copy_(torch.tensor([row],dtype=torch.int32))
                    values,candidates,mask=resident_observe(tensor_state)
                    with torch.no_grad():scores,_=net(values,candidates,mask)
                    cleaned=copy.deepcopy(state)
                    for k in ('events','logs','_replay_events','_public_replay','_public_board'):cleaned.pop(k,None)
                    cases.append(dict(state=cleaned,side=side,actions=actions,values=values[0].tolist(),
                                      candidates=candidates[0,mapped].tolist(),scores=scores[0,mapped].tolist()))
                    state=apply_action(state,side,rng.choice(actions))
        finally:backend.close()
        (args.output/f'{deck_id}-parity.json').write_text(json.dumps(cases,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(dict(deck=deck_id,file=str(dest),bytes=dest.stat().st_size,cases=len(cases))))
    return 0
if __name__=='__main__':raise SystemExit(main())
