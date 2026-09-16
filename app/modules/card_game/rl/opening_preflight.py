"""Explicit native/CUDA opening parity check, never trains or runs on import."""
def verify_opening(*, device='cpu'):
    import tempfile
    import torch
    import numpy as np
    from app.modules.card_game.engine.duel_v2 import new_game,observe,legal_actions,acting_side
    from .gpu_duel.compiled_backend import CompiledStarterBackend
    from .gpu_duel.state import empty_state
    from .rule_ir.pack_row import bind_gpu_rows,pack_python_row
    from .rule_ir.compiled_oracle import rebuild,map_python_action
    from .opening_observation import encode_opening,opening_observe
    with tempfile.TemporaryDirectory() as directory:
        native=CompiledStarterBackend(directory,backend='native')
        actual=CompiledStarterBackend(directory+'/target',backend='cuda' if device=='cuda' else 'native')
        try:
            cases=0
            from .gpu_duel.catalog import encode_public_deck_row
            from .capability import preset_opponent_deck
            deck=torch.tensor([encode_public_deck_row(preset_opponent_deck('starter'))],device=device,dtype=torch.int32)
            for seed in (0,1):
                initial=empty_state(1,device)
                actual.reset_public_gpu_state(initial,torch.tensor([seed],device=device,dtype=torch.int32),deck,deck,mulligan=True)
                if int(initial.phase[0])!=0 or initial.hand_n[0].cpu().tolist()!=[5,5] or int(initial.turn[0])!=0:
                    raise ValueError('Opening reset omitted mulligan or drew first-turn cards early')
                for _ in range(2):actual.step_gpu_state(initial,torch.zeros(1,device=device,dtype=torch.int32))
                if int(initial.phase[0])!=1 or int(initial.turn[0])!=1:raise ValueError('Opening transition failed')
            for first in ('a','b'):
                state=new_game(seed=11,first_side=first,skip_mulligan=False)
                side=acting_side(state)
                row=rebuild(native,pack_python_row(state))
                packed=empty_state(1,device);tensor=bind_gpu_rows(packed)
                tensor.copy_(torch.tensor([row],device=device,dtype=torch.int32))
                actions=[a for a in legal_actions(state,side) if a['type']=='mulligan']
                values,candidates,mask=opening_observe(packed)
                expected_values,expected_candidates=encode_opening(observe(state,side),actions)
                np.testing.assert_allclose(values[0].cpu().numpy(),expected_values)
                for i,action in enumerate(actions):
                    index=map_python_action(row,state,action)
                    if index!=i:raise ValueError('Opening tie-break order differs from official engine')
                    np.testing.assert_allclose(candidates[0,index].cpu().numpy(),expected_candidates[i])
                    tensor.copy_(torch.tensor([row],device=device,dtype=torch.int32))
                    expected=native.step_lists([row],[index])[0]
                    actual.step_gpu_state(packed,torch.tensor([index],device=device,dtype=torch.int32))
                    if tensor[0].cpu().tolist()!=expected:raise ValueError('Opening native/target mismatch')
                    cases+=1
            return {'backend':device,'opening_cases':cases,'trained':False}
        finally:
            native.close();actual.close()
