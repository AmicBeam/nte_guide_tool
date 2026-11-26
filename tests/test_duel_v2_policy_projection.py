"""Encoder/visibility/purity parity for derived policy-only projection."""
from copy import deepcopy
from random import Random
import unittest
import numpy as np
from app.modules.card_game.engine.duel_v2 import new_game,acting_side,legal_actions,observe
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.rl.league_rollout import clean
from app.modules.card_game.rl.batched_duel.gpu_policy_projection import policy_public_legal_decision
from app.modules.card_game.rl.seven_lineup import encode
from app.modules.card_game.content.duel_v2 import STARTER_DECKS


def without_card_display(value):
    if isinstance(value,dict):
        return {k:without_card_display(v) for k,v in value.items() if k not in (
            'name','description','resource_name','faction','avatar','portrait',
            'passive','awakened_passive','flavor','action_point_free','unavailable_reason')}
    if isinstance(value,list):return [without_card_display(v) for v in value]
    return value


def without_character_display(character):
    return without_card_display({k:v for k,v in character.items() if k not in (
        'attack_breakdown','stat_layers','equipment','effect_markers','shape','shape_description')})


def trace_states(max_steps=48):
    keys=('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke')
    decks={d['id']:d for d in STARTER_DECKS}
    for i,key in enumerate(keys):
        for seat in ('a','b'):
            seed=20261017001+i*13+int(seat=='b');rng=Random(seed)
            state=new_game(seed=seed,first_side=seat,decks={'a':decks[key],'b':decks[keys[(i+1)%7]]})
            for step in range(max_steps):
                yield state
                if state['phase']=='finished':break
                actor=acting_side(state);actions=[a for a in legal_actions(state,actor) if a['type']!='concede']
                if not actions:raise AssertionError('Canonical trace has no legal action')
                expanded=[]
                for action in actions:
                    expanded.extend([action]*({'play_card':2,'attack':2,'ultimate':3}.get(action['type'],1)))
                state=clean(simulate_action(state,actor,rng.choice(expanded)))


class PolicyProjectionTest(unittest.TestCase):
    def test_real_trace_numerical_legal_visibility_and_purity_both_viewers(self):
        seen=set();states=0;accepted=0;rejected=0
        for state in trace_states():
            before=deepcopy(state);seen.add(state['phase']);states+=1
            for side in ('a','b'):
                canonical=observe(state,side,include_previews=False)
                actions,view=policy_public_legal_decision(state,side)
                self.assertEqual(actions,[e['action'] for e in canonical['legal_actions']])
                self.assertEqual(state,before)
                for ps in ('a','b'):
                    self.assertNotIn('name',view['sides'][ps])
                    for character in view['sides'][ps]['characters']:
                        self.assertFalse(set(character)&{'name','avatar','portrait','passive',
                                                       'awakened_passive','flavor','faction','resource_name'})
                    self.assertEqual([without_character_display(c) for c in view['sides'][ps]['characters']],
                                     [without_character_display(c) for c in canonical['sides'][ps]['characters']])
                    self.assertEqual(without_card_display(view['sides'][ps]['hand']),without_card_display(canonical['sides'][ps]['hand']))
                    self.assertEqual(without_card_display(view['sides'][ps]['discard']),without_card_display(canonical['sides'][ps]['discard']))
                self.assertEqual(without_card_display(view.get('pending_choice')),without_card_display(canonical.get('pending_choice')))
                self.assertEqual(without_card_display(view.get('resolving_card')),without_card_display(canonical.get('resolving_card')))
                legal=[a for a in actions if a['type']!='concede']
                try:x,c=encode(canonical,legal)
                except (ValueError,KeyError,TypeError) as error:
                    with self.assertRaises(type(error)):encode(view,legal)
                    rejected+=1;continue
                xx,cc=encode(view,legal)
                self.assertEqual(x.tobytes(),xx.tobytes());self.assertEqual(c.tobytes(),cc.tobytes());accepted+=1
            self.assertEqual(state,before)
        self.assertGreater(states,200);self.assertGreater(accepted,400)
        self.assertTrue({'playing','choice','mulligan'}.issubset(seen))
        self.assertEqual(rejected,0,'Current encoder rejects a real canonical trace')

    def test_hidden_enemy_reallocation_does_not_change_policy_tensors(self):
        from app.modules.card_game.rl.batched_duel.gpu_zhenhong_guide import guide_sample_world
        for index,state in enumerate(trace_states(max_steps=8)):
            if state['phase']!='playing':continue
            side=acting_side(state);sampled=guide_sample_world(state,side,913+index)
            a,v=policy_public_legal_decision(state,side);aa,vv=policy_public_legal_decision(sampled,side)
            self.assertEqual(a,aa)
            a=[q for q in a if q['type']!='concede'];aa=[q for q in aa if q['type']!='concede']
            x,c=encode(v,a);xx,cc=encode(vv,aa)
            self.assertEqual(x.tobytes(),xx.tobytes());self.assertEqual(c.tobytes(),cc.tobytes())


class PolicyProjectionCudaTest(unittest.TestCase):
    def test_actual_cuda_derived_projection_matches_full_canonical_and_keeps_input(self):
        import os
        from pathlib import Path
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        program_path=os.environ.get('NTE_POLICY_PROJECTION_PROGRAM')
        if not program_path:self.skipTest('Explicit frozen policy projection required')
        from app.modules.card_game.rl.batched_duel.gpu_rule_program import load_program
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        states=[];seen=set()
        for index,state in enumerate(trace_states()):
            identity=(tuple(state['sides']['a']['characters']),state['first_side'],state['phase'])
            if index%23==0 or identity not in seen:states.append(state)
            seen.add(identity)
        packets=[dict(state=s,side=side) for s in states for side in ('a','b')]
        program=load_program(Path(program_path),runtime='seven',repo_root=Path.cwd())
        with CudaVm(lower_program(program),guard_standalone_entities=True,constant_lookup=True,
                    compact_frame_init=True,device_call_boundaries=True,library_call_boundaries=True,
                    mutable_lookup_cache=True,frozen_class_lookup=True) as vm:
            for start in range(0,len(packets),8):
                batch=packets[start:start+8]
                arena=vm.ingest(pack_states(batch,GraphLayout(len(batch),262144,524288,33554432)))
                actual=vm.launch(arena,'policy_projection_probe').decode()
                for packet,(actions,view,unchanged) in zip(batch,actual):
                    canonical=observe(packet['state'],packet['side'],include_previews=False)
                    self.assertEqual(unchanged,packet['state'])
                    self.assertEqual(actions,[e['action'] for e in canonical['legal_actions']])
                    legal=[a for a in actions if a['type']!='concede']
                    x,c=encode(canonical,legal);xx,cc=encode(view,legal)
                    self.assertEqual(x.tobytes(),xx.tobytes());self.assertEqual(c.tobytes(),cc.tobytes())
                    for ps in ('a','b'):
                        self.assertEqual(without_card_display(view['sides'][ps]['hand']),without_card_display(canonical['sides'][ps]['hand']))
