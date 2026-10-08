#!/usr/bin/env python3
"""Bounded native/CUDA and public observation parity. Never creates an optimizer."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('native','cuda'),default='native')
    p.add_argument('--cases',type=int,default=32)
    args=p.parse_args(argv)
    if not 1 <= args.cases <= 1024:p.error('cases must be 1..1024')
    import numpy as np
    import torch
    if args.backend=='cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; no fallback')
    from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
    from app.modules.card_game.rl.rule_ir.pack_row import pack_python_row,bind_gpu_rows
    from app.modules.card_game.rl.rule_ir.compiled_oracle import rebuild,map_python_action,canonical_row
    from app.modules.card_game.rl.gpu_duel.state import empty_state
    from app.modules.card_game.rl.public_observation import public_observe,encode_public
    from app.modules.card_game.rl.public_training import PublicTrainingEnv,FrozenLeague
    from app.modules.card_game.rl.public_schema import state_dim,rule_identity,CAND_DIM
    from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
    from app.modules.card_game.rl.capability import preset_opponent_deck,sample_public_deck
    from app.modules.card_game.engine.duel_v2 import new_game,observe,legal_actions
    device='cuda' if args.backend=='cuda' else 'cpu'
    with tempfile.TemporaryDirectory() as tmp:
        native=CompiledStarterBackend(tmp,backend='native',deck_id='starter')
        actual=CompiledStarterBackend(tmp,backend=args.backend,deck_id='starter')
        state_t=empty_state(1,device)
        rows_t=bind_gpu_rows(state_t)
        count=0
        try:
            for seed in range(args.cases):
                state=new_game(seed=seed,first_side='a',skip_mulligan=True,
                               decks={'a':preset_opponent_deck('starter'),'b':sample_public_deck(seed)})
                state['turn']=(1,6,13,20)[seed%4]
                row=rebuild(native,pack_python_row(state))
                rows_t.copy_(torch.tensor([row],dtype=torch.int32,device=device))
                actions=[a for a in legal_actions(state,'a') if a['type']!='concede']
                mapped=[map_python_action(row,state,a) for a in actions]
                expected_state,expected_cand=encode_public(observe(state,'a'),actions)
                got_state,got_cand,mask=public_observe(state_t)
                np.testing.assert_allclose(got_state[0].cpu().numpy(),expected_state,atol=1e-7)
                np.testing.assert_allclose(got_cand[0,mapped].cpu().numpy(),expected_cand,atol=1e-7)
                if not bool(mask[0,mapped].all()):raise AssertionError('Legal mask mismatch')
                index=mapped[seed%len(mapped)]
                expected=native.step_lists([row],[index])[0]
                actual.step_gpu_state(state_t,[index])
                if canonical_row(rows_t[0].cpu().tolist())!=canonical_row(expected):
                    raise AssertionError(f'{args.backend} differs from native case {seed}')
                count+=1
            # Exercise public device reset, a frozen model opponent, and stable learner decisions.
            league=FrozenLeague(2)
            league.add(CompactScorer(state_dim(),CAND_DIM,8).to(device))
            env=PublicTrainingEnv(8,deck_id='starter',league=league,device=device,compiled_dir=tmp)
            try:
                for _ in range(3):
                    env.reset_finished();env.prepare_decision()
                    _,_,mask=public_observe(env.state)
                    if not bool(mask.any(1).all()):raise AssertionError('Missing live action')
                    env.step_learner(mask.int().argmax(1).int())
            finally:env.close()
        finally:
            native.close();actual.close()
    print(json.dumps({'ok':True,'backend':args.backend,'cases':count,
                      'rule_hash':rule_identity(),'training_started':False}))
    return 0


if __name__=='__main__':raise SystemExit(main())
