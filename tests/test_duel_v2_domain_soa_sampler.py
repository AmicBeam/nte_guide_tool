"""Direct domain SoA sampler vs seven_sampler oracle and public-identity invariance."""
from copy import deepcopy
import unittest
import shutil

import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.rl.seven_sampler import determinize as oracle_determinize
from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states
from app.modules.card_game.rl.batched_duel.domain_soa_schema import (
    ARRAY_NAMES,
    CARD_VOCAB,
    DomainSoALayout,
)
from app.modules.card_game.rl.batched_duel.domain_soa_sampler import (
    DomainSoASampler,
    MISSING_STAGES,
    PARTICLE_CHANGED,
    REQUIRED,
    UNSUPPORTED_ROOT,
    coverage,
    cpu_oracle as _cpu_oracle,
    PUBLIC_GUARD_UNAVAILABLE,
    INVALID_INPUT,
    sampler_source,
    supports_full_duel_rules,
)

_INV = {index: card_id for card_id, index in CARD_VOCAB.items()}


def cpu_oracle(*args,**kwargs):
    if not (shutil.which('clang++')or shutil.which('g++')):
        raise unittest.SkipTest('C++ compiler unavailable; actual CUDA tests remain independent')
    return _cpu_oracle(*args,**kwargs)


def _decks():
    return {item['id']: item for item in STARTER_DECKS}


def _state(seed=7, first='a', a='starter', b='starter', skip=True):
    decks = _decks()
    return new_game(seed=seed, first_side=first, skip_mulligan=skip, decks={'a': decks[a], 'b': decks[b]})


def _torch_fields(states, layout, device='cpu'):
    packed = encode_domain_states(states, layout, presentation='drop')
    return packed, {
        name: torch.as_tensor(getattr(packed, name).copy(), device=device)
        for name in ARRAY_NAMES
    }


def _zone_ids(fields, row, side, zone):
    count = int(fields['zone_count'][row, side, zone])
    ids = []
    for pos in range(count):
        index = int(fields['zone_card_index'][row, side, zone, pos])
        ids.append(_INV[int(fields['card_template'][row, index])])
    return ids


def _eph(fields, row, side, zone):
    from app.modules.card_game.rl.batched_duel.domain_soa_schema import CARD_INT_FIELDS, PRESENCE_VALUE
    field = CARD_INT_FIELDS.index('ephemeral_turn')
    count = int(fields['zone_count'][row, side, zone])
    flags = []
    for pos in range(count):
        index = int(fields['zone_card_index'][row, side, zone, pos])
        present = int(fields['card_int_presence'][row, index, field]) == PRESENCE_VALUE
        flags.append(int(fields['card_int_value'][row, index, field]) if present else None)
    return flags


class DomainSoASamplerTest(unittest.TestCase):
    def test_coverage_refuses_full_rules_and_lists_gaps(self):
        self.assertFalse(supports_full_duel_rules)
        self.assertFalse(DomainSoASampler.supports_full_duel_rules)
        info = coverage()
        self.assertFalse(info['supports_full_duel_rules'])
        self.assertTrue(info['full_information_set_guard'])
        self.assertFalse(info['full_sampling_qualified']);self.assertFalse(info['physical_rules_available']);self.assertFalse(info['h3_available'])
        self.assertEqual(info['oracle'], 'app.modules.card_game.rl.seven_sampler')
        self.assertTrue(info['missing_stages'])
        self.assertEqual(tuple(info['missing_stages']), MISSING_STAGES)
        source = sampler_source()
        self.assertNotIn('vm_', source)
        self.assertNotIn('opcode', source)
        self.assertIn('bmt_seed_words', source)
        self.assertIn('generation allocation including recovery/deck-generator retry', info['implemented'])

    def test_revealed_hand_and_visible_own_deck_are_supported(self):
        states=[_state(),_state()]
        states[0]['sides']['b']['revealed_ids']=[states[0]['sides']['b']['hand'][0]['instance_id']]
        states[1]['sides']['a']['deck_visible_turn']=states[1]['turn']
        layout=DomainSoALayout(game_capacity=2)
        _,fields=_torch_fields(states,layout)
        with cpu_oracle(layout,guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(2,dtype=torch.int64),torch.tensor([11,12],dtype=torch.int64),torch.ones(2,dtype=torch.int32))
            self.assertEqual(out['row_error'].tolist(),[0,0])
            gold=oracle_determinize(states[1],'a',12)
            self.assertEqual(_zone_ids(out,1,0,1),[c['card_id']for c in gold['sides']['a']['deck']])
            self.assertEqual(int(out['rng'][1]),gold['rng'])

    def test_physical_views_reject_even_diagnostic_sampling(self):
        layout = DomainSoALayout(game_capacity=1)
        _, fields = _torch_fields([_state()], layout)
        fields['card_is_view'][0,0] = True
        with cpu_oracle(layout, guard=False) as sampler:
            result = sampler.determinize(fields, torch.tensor([0],dtype=torch.int64),
                                    torch.tensor([91],dtype=torch.int64), torch.tensor([1],dtype=torch.int32))
            self.assertEqual(int(result['row_error'][0]), UNSUPPORTED_ROOT)
            self.assertTrue(torch.equal(result['card_is_view'], fields['card_is_view']))

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA view guard required')
    def test_cuda_physical_views_reject_without_mutating(self):
        layout = DomainSoALayout(game_capacity=1)
        _, fields = _torch_fields([_state()], layout, 'cuda')
        fields['card_is_view'][0,0] = True
        with DomainSoASampler(layout=layout, device='cuda', guard=False) as sampler:
            prepared = sampler.prepare(fields, torch.zeros(1,dtype=torch.int64,device='cuda'),
                                       torch.ones(1,dtype=torch.int64,device='cuda'),
                                       torch.ones(1,dtype=torch.int32,device='cuda'))
            sampler.launch(prepared)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = sampler.launch(prepared)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(result['row_error'][0]), UNSUPPORTED_ROOT)
            self.assertTrue(torch.equal(result['zone_card_index'],fields['zone_card_index']))

    def test_missing_schema_fields_reject(self):
        layout = DomainSoALayout(game_capacity=2)
        packed, fields = _torch_fields([_state()], layout)
        fields = dict(fields)
        del fields['zone_card_index']
        with cpu_oracle(layout,guard=False) as sampler:
            with self.assertRaisesRegex(TypeError, 'Missing domain SoA fields'):
                sampler.determinize(
                    fields, torch.zeros(1, dtype=torch.int64),
                    torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.int32))

    def test_unsupported_root_is_row_error(self):
        layout = DomainSoALayout(game_capacity=2)
        state = _state()
        state['phase'] = 'finished'
        packed, fields = _torch_fields([state], layout)
        with cpu_oracle(layout,guard=False) as sampler:
            out = sampler.determinize(
                fields, torch.zeros(1, dtype=torch.int64),
                torch.tensor([17], dtype=torch.int64), torch.ones(1, dtype=torch.int32))
        self.assertEqual(int(out['row_error'][0]), UNSUPPORTED_ROOT)

    def test_opening_matches_seven_sampler_oracle(self):
        layout = DomainSoALayout(game_capacity=7, max_cards=256, max_zone=128)
        states, oracles, viewers, seeds = [], [], [], []
        for i, key in enumerate(('starter', 'weave-rush', 'quick-rush', 'zhenhong','murk','collapse','lingke')):
            state = _state(seed=11 + i, first='a' if i % 2 == 0 else 'b', a=key, b=key)
            viewer = 'a' if i % 2 == 0 else 'b'
            seed = 23 + i * 17
            states.append(state)
            oracles.append(oracle_determinize(deepcopy(state), viewer, seed))
            viewers.append(0 if viewer == 'a' else 1)
            seeds.append(seed)
        packed, fields = _torch_fields(states, layout)
        snapshot = {name: fields[name].clone() for name in ('card_template', 'zone_card_index', 'rng')}
        with cpu_oracle(layout,guard=False) as sampler:
            out = sampler.determinize(
                fields, torch.tensor(viewers, dtype=torch.int64),
                torch.tensor(seeds, dtype=torch.int64), torch.ones(len(states), dtype=torch.int32))
        self.assertTrue(torch.equal(out['row_error'], torch.zeros(len(states), dtype=torch.int32)))
        for name, value in snapshot.items():
            self.assertTrue(torch.equal(fields[name], value), name)
        for row, (state, world, viewer) in enumerate(zip(states, oracles, viewers)):
            foe = 1 - viewer
            self.assertEqual(_zone_ids(out, row, foe, 0), [c['card_id'] for c in world['sides']['b' if viewer == 0 else 'a']['hand']])
            self.assertEqual(_zone_ids(out, row, foe, 1), [c['card_id'] for c in world['sides']['b' if viewer == 0 else 'a']['deck']])
            self.assertEqual(_zone_ids(out, row, viewer, 1), [c['card_id'] for c in world['sides']['a' if viewer == 0 else 'b']['deck']])
            self.assertEqual(int(out['rng'][row]), world['rng'])
            self.assertEqual(int(out['next_instance'][row]), world['next_instance'])
            self.assertEqual(int(out['next_entity'][row]), world['next_entity'])
            self.assertEqual(
                _eph(out, row, foe, 0),
                [c.get('ephemeral_turn') for c in world['sides']['b' if viewer == 0 else 'a']['hand']])

    def test_hidden_identity_does_not_change_sample(self):
        layout = DomainSoALayout(game_capacity=2, max_cards=256, max_zone=128)
        state = _state(seed=19, a='murk', b='murk')
        packed, fields = _torch_fields([state, deepcopy(state)], layout)
        for zone in (0, 1):
            for pos in range(int(fields['zone_count'][1, 1, zone])):
                index = int(fields['zone_card_index'][1, 1, zone, pos])
                fields['card_template'][1, index] = CARD_VOCAB['A01']
        with cpu_oracle(layout,guard=False) as sampler:
            out = sampler.determinize(
                fields, torch.zeros(2, dtype=torch.int64),
                torch.tensor([41, 41], dtype=torch.int64), torch.ones(2, dtype=torch.int32))
        self.assertTrue(torch.equal(out['row_error'], torch.zeros(2, dtype=torch.int32)))
        self.assertEqual(_zone_ids(out, 0, 1, 0), _zone_ids(out, 1, 1, 0))
        self.assertEqual(_zone_ids(out, 0, 1, 1), _zone_ids(out, 1, 1, 1))
        world = oracle_determinize(state, 'a', 41)
        self.assertEqual(_zone_ids(out, 0, 1, 0), [c['card_id'] for c in world['sides']['b']['hand']])
        self.assertEqual(int(out['rng'][0]), world['rng'])
        self.assertNotEqual(PARTICLE_CHANGED, int(out['row_error'][0]))

    def test_generated_surplus_fixtures_match_oracle(self):
        layout = DomainSoALayout(game_capacity=4, max_cards=256, max_zone=128)
        decks = _decks()
        murk = new_game(seed=7, first_side='b', skip_mulligan=True, decks={'a': decks['murk'], 'b': decks['murk']})
        team = murk['sides']['b']
        card = next((c for zone in ('hand', 'deck') for c in team[zone] if c['card_id'] == 'A04'), None)
        team[card and ('hand' if card in team['hand'] else 'deck')].remove(card)
        team['discard'] = [card, card_instance(murk, 'AF01', 'b')]
        rush = new_game(seed=7, first_side='b', skip_mulligan=True,
                        decks={'a': decks['quick-rush'], 'b': decks['quick-rush']})
        rteam = rush['sides']['b']
        rteam['discard'] += rteam['hand']
        rteam['hand'] = []
        for _ in range(4):
            rteam['deck'].append(card_instance(rush, 'Q05', 'b'))
        states = [murk, rush]
        packed, fields = _torch_fields(states, layout)
        seeds = [3, 5]
        worlds = [oracle_determinize(states[0], 'a', seeds[0]), oracle_determinize(states[1], 'a', seeds[1])]
        with cpu_oracle(layout,guard=False) as sampler:
            out = sampler.determinize(
                fields, torch.zeros(2, dtype=torch.int64),
                torch.tensor(seeds, dtype=torch.int64), torch.ones(2, dtype=torch.int32))
        self.assertTrue(torch.equal(out['row_error'], torch.zeros(2, dtype=torch.int32)), out['row_error'].tolist())
        for row, world in enumerate(worlds):
            self.assertEqual(_zone_ids(out, row, 1, 0), [c['card_id'] for c in world['sides']['b']['hand']])
            self.assertEqual(_zone_ids(out, row, 1, 1), [c['card_id'] for c in world['sides']['b']['deck']])
            self.assertEqual(int(out['rng'][row]), world['rng'])

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required; CPU oracle is not device qualification')
    def test_cuda_opening_matches_cpu_oracle(self):
        layout = DomainSoALayout(game_capacity=2, max_cards=256, max_zone=128)
        states = [_state(seed=3), _state(seed=5, a='murk', b='quick-rush')]
        packed, cpu_fields = _torch_fields(states, layout, 'cpu')
        viewers = torch.zeros(2, dtype=torch.int64)
        seeds = torch.tensor([11, 13], dtype=torch.int64)
        active = torch.ones(2, dtype=torch.int32)
        worlds=[oracle_determinize(state,'a',int(seeds[i]))for i,state in enumerate(states)]
        gpu_fields = {name: value.cuda() for name, value in cpu_fields.items()}
        with DomainSoASampler(layout=layout, device='cuda',guard=False) as gpu:
            got = gpu.determinize(gpu_fields, viewers.cuda(), seeds.cuda(), active.cuda())
            torch.cuda.synchronize()
            self.assertEqual(got['row_error'].cpu().tolist(),[0,0])
            for row,world in enumerate(worlds):
                self.assertEqual(_zone_ids(got,row,1,0),[c['card_id']for c in world['sides']['b']['hand']])
                self.assertEqual(_zone_ids(got,row,1,1),[c['card_id']for c in world['sides']['b']['deck']])
                self.assertEqual(int(got['rng'][row]),world['rng'])
                self.assertEqual(int(got['next_entity'][row]),world['next_entity'])
                self.assertEqual(int(got['next_instance'][row]),world['next_instance'])

    def test_multiple_generated_kinds_keep_python_counter_insertion_order(self):
        state=_state(seed=38,a='collapse',b='collapse')
        team=state['sides']['b']
        parent=next(c for c in team['hand']+team['deck']if c['card_id']=='A04')
        for name in ('hand','deck'):
            if parent in team[name]:team[name].remove(parent);break
        team['discard'].append(parent)
        team['hand'].extend([card_instance(state,'FF01','b'),card_instance(state,'PF01','b')])
        layout=DomainSoALayout(game_capacity=24,max_cards=256,max_zone=128)
        _,fields=_torch_fields([deepcopy(state)for _ in range(24)],layout)
        with cpu_oracle(layout,guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(24,dtype=torch.int64),
                torch.arange(24,dtype=torch.int64)+1,torch.ones(24,dtype=torch.int32))
            self.assertEqual(out['row_error'].tolist(),[0]*24)
            for row in range(24):
                world=oracle_determinize(state,'a',row+1)
                for z,name in ((0,'hand'),(1,'deck')):
                    self.assertEqual(_zone_ids(out,row,1,z),[c['card_id']for c in world['sides']['b'][name]],msg=str(row+1))
                self.assertEqual(int(out['rng'][row]),world['rng'])

    def test_recovery_budget_does_not_start_one_half_turn_early(self):
        states=[]
        for turn in (20,21):
            state=_state(seed=88,first='b',a='murk',b='murk')
            state['turn']=turn;state['active_side']='a'if turn%2==0 else 'b'
            state['sides']['b']['hand'].append(card_instance(state,'AF01','b'))
            states.append(state)
        with self.assertRaisesRegex(ValueError,'No public-consistent generation allocation'):
            oracle_determinize(states[0],'a',101)
        world=oracle_determinize(states[1],'a',102)
        layout=DomainSoALayout(game_capacity=2,max_cards=256,max_zone=128)
        _,fields=_torch_fields(states,layout)
        with cpu_oracle(layout,guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(2,dtype=torch.int64),
                torch.tensor([101,102],dtype=torch.int64),torch.ones(2,dtype=torch.int32))
            self.assertEqual(out['row_error'].tolist(),[6,0])
            for z,name in ((0,'hand'),(1,'deck')):
                self.assertEqual(_zone_ids(out,1,1,z),[c['card_id']for c in world['sides']['b'][name]])
            self.assertEqual(int(out['rng'][1]),world['rng'])

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA recovery-boundary test required')
    def test_cuda_recovery_budget_boundary_matches_formal_sampler(self):
        states=[]
        for turn in (20,21):
            state=_state(seed=88,first='b',a='murk',b='murk')
            state['turn']=turn;state['active_side']='a'if turn%2==0 else 'b'
            state['sides']['b']['hand'].append(card_instance(state,'AF01','b'))
            states.append(state)
        with self.assertRaisesRegex(ValueError,'No public-consistent generation allocation'):
            oracle_determinize(states[0],'a',101)
        world=oracle_determinize(states[1],'a',102)
        layout=DomainSoALayout(game_capacity=2,max_cards=256,max_zone=128)
        _,fields=_torch_fields(states,layout,'cuda')
        with DomainSoASampler(layout=layout,device='cuda',guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(2,dtype=torch.int64,device='cuda'),
                torch.tensor([101,102],dtype=torch.int64,device='cuda'),torch.ones(2,dtype=torch.int32,device='cuda'))
            torch.cuda.synchronize()
            host={name:value.cpu()for name,value in out.items()}
            self.assertEqual(host['row_error'].tolist(),[6,0])
            self.assertEqual(int(host['rng'][1]),world['rng'])
            self.assertEqual(int(host['next_entity'][1]),world['next_entity'])
            for z,name in ((0,'hand'),(1,'deck')):
                self.assertEqual(_zone_ids(host,1,1,z),[c['card_id']for c in world['sides']['b'][name]])

    def test_default_public_guard_uses_complete_observations_and_original_candidates(self):
        layout=DomainSoALayout(game_capacity=1,max_cards=256,max_zone=128)
        _,fields=_torch_fields([_state(seed=17)],layout)
        with cpu_oracle(layout)as sampler:
            batch=sampler.prepare(fields,torch.zeros(1,dtype=torch.int64),
                torch.tensor([31],dtype=torch.int64),torch.ones(1,dtype=torch.int32))
            pointer=batch.outputs['row_error'].data_ptr()
            for viewer in (0,1):
                batch.viewers.fill_(viewer)
                out=sampler.launch(batch)
                self.assertEqual(int(out['row_error'][0]),0)
                self.assertEqual(out['row_error'].data_ptr(),pointer)
                self.assertEqual(set(out),set(fields))
                world=oracle_determinize(_state(seed=17),('a','b')[viewer],31)
                self.assertEqual(int(out['rng'][0]),world['rng'])
                self.assertEqual(_zone_ids(out,0,1-viewer,0),[c['card_id']for c in world['sides'][('a','b')[1-viewer]]['hand']])
            fields['row_error'][0]=9
            out=sampler.launch(batch)
            self.assertEqual(int(out['row_error'][0]),9)

    def test_full_state_noncontiguous_inputs_refresh_and_failure_rolls_back(self):
        layout=DomainSoALayout(game_capacity=2,max_cards=256,max_zone=128)
        _,fields=_torch_fields([_state(seed=21),_state(seed=22)],layout)
        hp=torch.empty((2,4),dtype=torch.int32)[:,::2]
        hp.copy_(fields['player_hp']);fields['player_hp']=hp
        with cpu_oracle(layout,guard=False)as sampler:
            batch=sampler.prepare(fields,torch.zeros(2,dtype=torch.int64),
                torch.tensor([31,32],dtype=torch.int64),torch.ones(2,dtype=torch.int32))
            out=sampler.launch(batch)
            self.assertEqual(set(out),set(fields))
            self.assertEqual(out['row_error'].tolist(),[0,0])
            out['player_hp'][0,0]-=1
            self.assertEqual(int(fields['player_hp'][0,0]),30)
            fields['player_hp'][0,0]=17
            fields['zone_count'][1,1,1]=-1
            out=sampler.launch(batch)
            self.assertEqual(int(out['player_hp'][0,0]),17)
            self.assertEqual(out['row_error'].tolist(),[0,INVALID_INPUT])
            for name in fields:
                if name!='row_error':self.assertTrue(torch.equal(out[name][1],fields[name][1]),msg=name)

    def test_catalog_binding_and_existing_error_are_not_cleared(self):
        layout=DomainSoALayout(game_capacity=2,max_cards=256,max_zone=128)
        _,fields=_torch_fields([_state(seed=34),_state(seed=35)],layout)
        fields['catalog_digest'][0,0]^=1
        fields['row_error'][1]=13
        with cpu_oracle(layout,guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(2,dtype=torch.int64),
                torch.tensor([91,92],dtype=torch.int64),torch.ones(2,dtype=torch.int32))
            self.assertEqual(out['row_error'].tolist(),[INVALID_INPUT,13])
            for name in fields:
                if name!='row_error':self.assertTrue(torch.equal(out[name],fields[name]),msg=name)

    def test_choice_and_operation_card_remain_visible_and_cannot_be_reallocated(self):
        state=_state(seed=44)
        team=state['sides']['a'];held=team['hand'].pop()
        inspected=team['deck'][:2];del team['deck'][:2]
        state.update(phase='choice',pending_choice=dict(side='a',kind='inspect_top',prompt='inspect',cards=inspected),
                     operation=dict(side='a',actor=team['order'][0],card=held,participants=[],
                                    target_id=None,harmony=None,first_bonus=0,redeem_bonus=0,record_candidate=False))
        world=oracle_determinize(deepcopy(state),'a',77)
        layout=DomainSoALayout(game_capacity=1,max_cards=256,max_zone=128)
        _,fields=_torch_fields([state],layout)
        with cpu_oracle(layout,guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(1,dtype=torch.int64),
                torch.tensor([77],dtype=torch.int64),torch.ones(1,dtype=torch.int32))
            self.assertEqual(int(out['row_error'][0]),0)
            self.assertEqual(int(out['rng'][0]),world['rng'])
            self.assertEqual(int(out['next_entity'][0]),world['next_entity'])
            self.assertEqual(int(out['next_instance'][0]),world['next_instance'])
            refs=[int(out['op_card_index'][0])]+out['choice_card_index'][0,:2].tolist()
            originals=[held]+inspected
            for ref,original in zip(refs,originals):
                self.assertEqual(int(out['card_entity_slot'][0,ref]),int(original['entity_id'][1:]))
                self.assertEqual(_INV[int(out['card_template'][0,ref])],original['card_id'])
            for z,name in ((0,'hand'),(1,'deck')):
                self.assertEqual(_zone_ids(out,0,1,z),[c['card_id']for c in world['sides']['b'][name]])

    @unittest.skipUnless(torch.cuda.is_available(),'Actual 400-row CUDA sampler graph required')
    def test_400_resident_graph_refreshes_inputs_and_matches_formal_sampler(self):
        keys=('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke')
        states=[]
        for i in range(400):
            key='collapse' if i%11==0 else keys[i%7]
            state=_state(seed=100+i,first='ab'[i%2],a=key,b=key)
            if i%11==0:
                side='ab'[1-i%2];team=state['sides'][side]
                parent=next(c for c in team['hand']+team['deck']if c['card_id']=='A04')
                for name in ('hand','deck'):
                    if parent in team[name]:team[name].remove(parent);break
                team['discard'].append(parent)
                team['hand'].extend([card_instance(state,'FF01',side),card_instance(state,'PF01',side)])
            states.append(state)
        layout=DomainSoALayout(game_capacity=400,max_cards=256,max_zone=128)
        _,fields=_torch_fields(states,layout,'cuda')
        # A real state-pool column can be a noncontiguous view.
        hp=torch.empty((400,4),dtype=torch.int32,device='cuda')[:,::2]
        hp.copy_(fields['player_hp']);fields['player_hp']=hp
        viewers=torch.tensor([i%2 for i in range(400)],dtype=torch.int64,device='cuda')
        seeds=torch.arange(400,dtype=torch.int64,device='cuda')+1900
        active=torch.ones(400,dtype=torch.int32,device='cuda')
        with DomainSoASampler(layout=layout,device='cuda',guard=False)as sampler:
            batch=sampler.prepare(fields,viewers,seeds,active)
            sampler.launch(batch);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=sampler.launch(batch)
            graph.replay();torch.cuda.synchronize()
            host={name:value.cpu()for name,value in result.items()}
            self.assertEqual(set(host),set(ARRAY_NAMES))
            self.assertEqual(host['row_error'].tolist(),[0]*400)
            for row,state in enumerate(states):
                viewer='ab'[row%2];foe=1-row%2
                world=oracle_determinize(state,viewer,1900+row)
                for z,name in ((0,'hand'),(1,'deck')):
                    self.assertEqual(_zone_ids(host,row,foe,z),[c['card_id']for c in world['sides']['ab'[foe]][name]])
                self.assertEqual(int(host['rng'][row]),world['rng'])
                self.assertEqual(int(host['next_instance'][row]),world['next_instance'])
                self.assertEqual(int(host['next_entity'][row]),world['next_entity'])
            fields['player_hp'][0,0]=17
            fields['row_error'][1]=9
            fields['zone_count'][2,1,1]=-1
            graph.replay();torch.cuda.synchronize()
            self.assertEqual(int(result['player_hp'][0,0]),17)
            self.assertEqual(result['row_error'][:3].cpu().tolist(),[0,9,INVALID_INPUT])
            for name in fields:
                if name!='row_error':
                    self.assertTrue(torch.equal(result[name][1:3],fields[name][1:3]),msg=name)
            self.assertEqual(int(fields['player_hp'][0,0]),17)
            del graph
            torch.cuda.synchronize()
        # Default guard checks complete model inputs; it does not qualify full duel rules.
        with DomainSoASampler(layout=DomainSoALayout(game_capacity=1,max_cards=256,max_zone=128),device='cuda')as guarded:
            one={name:value[:1].contiguous()for name,value in fields.items()}
            sample=guarded.prepare(one,viewers[:1],seeds[:1],active[:1])
            pointer=sample.outputs['row_error'].data_ptr()
            guarded.launch(sample);torch.cuda.synchronize()
            self.assertEqual(int(sample.outputs['row_error'][0]),0)
            self.assertEqual(sample.outputs['row_error'].data_ptr(),pointer)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA choice-root sampling required')
    def test_cuda_choice_roots_preserve_visible_aliases_and_reject_wrong_kind(self):
        states=[]
        for kind in ('inspect_top','enemy_hand','heiyu_divination','hand_deck_placement','deck_order'):
            state=_state(seed=44)
            team=state['sides']['a'];held=team['hand'].pop()
            if kind=='enemy_hand':inspected=list(state['sides']['b']['hand'])
            else:inspected=team['deck'][:2];del team['deck'][:2]
            pending=dict(side='a',kind=kind,prompt='inspect',cards=inspected)
            if kind not in ('inspect_top','enemy_hand'):pending.update(deck_order=True,deck_cards=True,multi_bottom=True)
            if kind=='hand_deck_placement':pending['hand_placement']=True
            state.update(phase='choice',pending_choice=pending,
                         operation=dict(side='a',actor=team['order'][0],card=held,participants=[],
                                        target_id=None,harmony=None,first_bonus=0,redeem_bonus=0,record_candidate=False))
            states.append(state)
        worlds=[oracle_determinize(state,'a',91+i)for i,state in enumerate(states[:4])]
        with self.assertRaises(ValueError):oracle_determinize(states[4],'a',95)
        layout=DomainSoALayout(game_capacity=5,max_cards=256,max_zone=128)
        _,fields=_torch_fields(states,layout,'cuda')
        with DomainSoASampler(layout=layout,device='cuda',guard=False)as sampler:
            out=sampler.determinize(fields,torch.zeros(5,dtype=torch.int64,device='cuda'),
                torch.arange(5,dtype=torch.int64,device='cuda')+91,
                torch.ones(5,dtype=torch.int32,device='cuda'))
            torch.cuda.synchronize()
            host={name:value.cpu()for name,value in out.items()}
            self.assertEqual(host['row_error'].tolist(),[0,0,0,0,UNSUPPORTED_ROOT])
            for row,world in enumerate(worlds):
                self.assertEqual(int(host['rng'][row]),world['rng'])
                self.assertEqual(int(host['next_instance'][row]),world['next_instance'])
                self.assertEqual(int(host['next_entity'][row]),world['next_entity'])
                for z,name in ((0,'hand'),(1,'deck')):
                    self.assertEqual(_zone_ids(host,row,1,z),[c['card_id']for c in world['sides']['b'][name]])
                refs=[int(fields['op_card_index'][row])]+fields['choice_card_index'][row,:int(fields['choice_card_count'][row])].cpu().tolist()
                for ref in refs:
                    self.assertEqual(int(host['card_entity_slot'][row,ref]),int(fields['card_entity_slot'][row,ref]))
                    self.assertEqual(int(host['card_template'][row,ref]),int(fields['card_template'][row,ref]))
            for name in fields:
                if name!='row_error':self.assertTrue(torch.equal(out[name][4],fields[name][4]),msg=name)

    def test_inactive_rows_keep_input_and_zero_error(self):
        layout = DomainSoALayout(game_capacity=2, max_cards=256, max_zone=128)
        packed, fields = _torch_fields([_state(seed=8), _state(seed=9)], layout)
        before = fields['card_template'].clone()
        with cpu_oracle(layout,guard=False) as sampler:
            out = sampler.determinize(
                fields, torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([1, 2], dtype=torch.int64), torch.tensor([0, 0], dtype=torch.int32))
        self.assertTrue(torch.equal(out['row_error'], torch.zeros(2, dtype=torch.int32)))
        self.assertTrue(torch.equal(out['card_template'], before))
        self.assertTrue(torch.equal(out['rng'], fields['rng']))


if __name__ == '__main__':
    unittest.main()


class RevealedSamplerTest(unittest.TestCase):
    def fixtures(self):
        states=[]
        for i,key in enumerate(('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke')):
            st=_state(seed=11+i,a=key,b=key)
            for side in ('a','b'):
                team=st['sides'][side];team['revealed_ids']=[team['hand'][2]['instance_id'],team['hand'][0]['instance_id'],team['hand'][2]['instance_id']]
            states.append(st)
        return states

    def verify(self,states,viewers,seeds,fields,out):
        self.assertFalse(out['row_error'].any())
        for b,(st,v,seed)in enumerate(zip(states,viewers,seeds)):
            gold=oracle_determinize(st,('a','b')[v],seed)
            for side,zone in ((v,1),(1-v,0),(1-v,1)):
                self.assertEqual(_zone_ids(out,b,side,zone),[c['card_id']for c in gold['sides'][('a','b')[side]][('hand','deck')[zone]]])
            self.assertEqual(int(out['rng'][b]),gold['rng']);self.assertEqual(int(out['next_entity'][b]),gold['next_entity']);self.assertEqual(int(out['next_instance'][b]),gold['next_instance'])
            foe=1-v;refs=[int(fields['zone_card_index'][b,foe,0,j])for j in (0,2)]
            for k in fields:
                if k.startswith('card_'):self.assertTrue(torch.equal(out[k][b,refs],fields[k][b,refs]),str((b,k)))

    def test_revealed_count_order_and_whole_world_information(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard
        from app.modules.card_game.rl.batched_duel.domain_soa_card_order import DomainCardOrder
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
        from app.modules.card_game.rl.batched_duel.domain_soa_codec import decode_domain_states
        states=[s for s in self.fixtures()for _ in range(2)];viewers=[i%2 for i in range(len(states))];seeds=list(range(70,70+len(states)));L=DomainSoALayout(game_capacity=len(states));_,f=_torch_fields(states,L)
        with cpu_oracle(L,guard=False)as sampler,DomainInformationGuard(L,device='cpu',allow_cpu_oracle=True,max_actions=64)as guard,DomainCardOrder(L,device='cpu',allow_cpu_oracle=True)as order:
            vi=torch.tensor(viewers);ac=torch.ones(len(states),dtype=torch.bool);out=sampler.determinize(f,vi,torch.tensor(seeds),ac.int());self.verify(states,viewers,seeds,f,out)
            checked=guard.launch(guard.prepare(f,out,vi,ac));self.assertTrue(checked['accepted'].all())
            ordered=order.launch(order.prepare(out,ac));self.assertFalse(ordered['errors'].any())
            worlds=decode_domain_states(DomainSoABatch(layout=L,**{k:v.numpy()for k,v in ordered['fields'].items()}))
            for b,w in enumerate(worlds):
                gold=oracle_determinize(states[b],('a','b')[viewers[b]],seeds[b]);gold.pop('events',None);self.assertEqual(w,gold)

    def test_invalid_revealed_template_refuses_before_catalog_access(self):
        states=self.fixtures()[:2];L=DomainSoALayout(game_capacity=2);_,f=_torch_fields(states,L)
        slot=int(f['zone_card_index'][0,1,0,0]);f['card_template'][0,slot]=32767
        with cpu_oracle(L,guard=False)as sampler:
            out=sampler.determinize(f,torch.zeros(2,dtype=torch.int64),torch.tensor([7,9]),torch.ones(2,dtype=torch.int32))
            self.assertEqual(out['row_error'].tolist(),[INVALID_INPUT,0])
            for k in f:
                if k!='row_error':self.assertTrue(torch.equal(out[k][0],f[k][0]),k)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_revealed_sampling_guard_and_new_seed_replay(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard
        states=[self.fixtures()[i%7]for i in range(400)];L=DomainSoALayout(game_capacity=400);_,f=_torch_fields(states,L,'cuda')
        vi=torch.arange(400,device='cuda')%2;seed=torch.arange(400,device='cuda')+70;ac=torch.ones(400,dtype=torch.bool,device='cuda')
        with DomainSoASampler(L,device='cuda',guard=False)as sampler,DomainInformationGuard(L,device='cuda',max_actions=64)as guard:
            p=sampler.prepare(f,vi,seed,ac.int());sampler.launch(p);check=guard.prepare(f,p.outputs,vi,ac);guard.launch(check);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=sampler.launch(p);checked=guard.launch(check)
            g.replay();torch.cuda.synchronize();self.assertTrue(checked['accepted'].all());self.verify(states[:14],vi[:14].cpu().tolist(),seed[:14].cpu().tolist(),f,out)
            seed.add_(1000);vi.copy_(1-vi);g.replay();torch.cuda.synchronize();self.assertTrue(checked['accepted'].all());self.verify(states[:14],vi[:14].cpu().tolist(),seed[:14].cpu().tolist(),f,out)
            foe=1-int(vi[0]);ref=int(f['zone_card_index'][0,foe,0,0]);f['card_template'][0,ref]=32767
            g.replay();torch.cuda.synchronize();self.assertEqual(int(out['row_error'][0]),INVALID_INPUT);self.assertFalse(checked['accepted'][0]);self.assertTrue(checked['accepted'][1:].all())


class VisibleDeckSamplerTest(unittest.TestCase):
    def fixtures(self):
        states=[];viewers=[]
        for i,key in enumerate(('starter','weave-rush','quick-rush','zhenhong','murk','collapse','lingke')):
            for v in (0,1):
                for active in (True,False):
                    st=_state(seed=40+i,a=key,b=key,first=('a','b')[v if active else 1-v])
                    st['sides'][('a','b')[v]]['deck_visible_turn']=st['turn']
                    st['sides'][('a','b')[1-v]]['deck_visible_turn']=st['turn']
                    states.append(st);viewers.append(v)
        return states,viewers

    def verify(self,states,viewers,seeds,out):
        self.assertFalse(out['row_error'].any())
        for b,(st,v,seed)in enumerate(zip(states,viewers,seeds)):
            gold=oracle_determinize(st,('a','b')[v],seed)
            for side,zone in ((v,1),(1-v,0),(1-v,1)):
                self.assertEqual(_zone_ids(out,b,side,zone),[c['card_id']for c in gold['sides'][('a','b')[side]][('hand','deck')[zone]]],str((b,side,zone)))
            self.assertEqual(int(out['rng'][b]),gold['rng']);self.assertEqual(int(out['next_entity'][b]),gold['next_entity']);self.assertEqual(int(out['next_instance'][b]),gold['next_instance'])

    def test_visible_deck_only_when_viewer_active_and_strict_export(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard
        from app.modules.card_game.rl.batched_duel.domain_soa_card_order import DomainCardOrder
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
        from app.modules.card_game.rl.batched_duel.domain_soa_codec import decode_domain_states
        states,viewers=self.fixtures();B=len(states);seeds=list(range(70,70+B));L=DomainSoALayout(game_capacity=B);_,f=_torch_fields(states,L)
        with cpu_oracle(L,guard=False)as sampler,DomainInformationGuard(L,device='cpu',allow_cpu_oracle=True,max_actions=64)as guard,DomainCardOrder(L,device='cpu',allow_cpu_oracle=True)as order:
            vi=torch.tensor(viewers);ac=torch.ones(B,dtype=torch.bool);out=sampler.determinize(f,vi,torch.tensor(seeds),ac.int());self.verify(states,viewers,seeds,out)
            checked=guard.launch(guard.prepare(f,out,vi,ac));self.assertTrue(checked['accepted'].all())
            ordered=order.launch(order.prepare(out,ac));self.assertFalse(ordered['errors'].any())
            worlds=decode_domain_states(DomainSoABatch(layout=L,**{k:v.numpy()for k,v in ordered['fields'].items()}))
            for b,w in enumerate(worlds):
                gold=oracle_determinize(states[b],('a','b')[viewers[b]],seeds[b]);gold.pop('events',None);self.assertEqual(w,gold)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_visible_deck_graph_refreshes_visibility_and_rng(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import PLAYER_INT_FIELDS
        base,vs=self.fixtures();states=[base[i%len(base)]for i in range(400)];viewers=[vs[i%len(vs)]for i in range(400)];L=DomainSoALayout(game_capacity=400);_,f=_torch_fields(states,L,'cuda')
        vi=torch.tensor(viewers,device='cuda');seeds=torch.arange(400,device='cuda')+70;ac=torch.ones(400,dtype=torch.bool,device='cuda')
        with DomainSoASampler(L,device='cuda',guard=False)as sampler,DomainInformationGuard(L,device='cuda',max_actions=64)as guard:
            p=sampler.prepare(f,vi,seeds,ac.int());sampler.launch(p);check=guard.prepare(f,p.outputs,vi,ac);guard.launch(check);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=sampler.launch(p);checked=guard.launch(check)
            g.replay();torch.cuda.synchronize();self.assertTrue(checked['accepted'].all());self.verify(states[:28],viewers[:28],seeds[:28].cpu().tolist(),out)
            key=PLAYER_INT_FIELDS.index('deck_visible_turn');f['player_int_value'][:,:,key]-=1
            replacements=deepcopy(states[:28])
            for st in replacements:
                for team in st['sides'].values():team['deck_visible_turn']-=1
            seeds.add_(1000);g.replay();torch.cuda.synchronize();self.assertTrue(checked['accepted'].all());self.verify(replacements,viewers[:28],seeds[:28].cpu().tolist(),out)


class EffectiveViewSamplerTest(unittest.TestCase):
    def fixtures(self):
        from tests import test_duel_v2_domain_soa_card_order as view_cases
        return view_cases.EffectiveViewOrderTest().fixtures()

    def check_worlds(self,states,viewers,seeds,L,out,active,device):
        from app.modules.card_game.rl.batched_duel.domain_soa_card_order import DomainCardOrder
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
        from app.modules.card_game.rl.batched_duel.domain_soa_codec import decode_domain_states
        with DomainCardOrder(L,device=device,allow_cpu_oracle=device=='cpu')as order:
            ordered=order.launch(order.prepare(out,active));self.assertFalse(ordered['errors'].any())
            worlds=decode_domain_states(DomainSoABatch(layout=L,**{k:v[:len(states)].cpu().numpy()for k,v in ordered['fields'].items()}))
            for b,w in enumerate(worlds):
                gold=oracle_determinize(states[b],('a','b')[viewers[b]],seeds[b]);gold.pop('events',None);self.assertEqual(w,gold)
                view=w['pending_choice']['cards'][0];self.assertIs(view,w['operation']['card']);self.assertIs(view['_physical_card'],w['operation']['choice_cards'][1])

    def test_effective_view_retains_physical_row_and_shared_visible_identity(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard
        states=self.fixtures();L=DomainSoALayout(game_capacity=2);_,f=_torch_fields(states,L);vi=torch.zeros(2,dtype=torch.int64);seeds=torch.tensor([5,19]);ac=torch.ones(2,dtype=torch.bool)
        with cpu_oracle(L,guard=False)as sampler,DomainInformationGuard(L,device='cpu',allow_cpu_oracle=True,max_actions=64)as guard:
            out=sampler.determinize(f,vi,seeds,ac.int());self.assertFalse(out['row_error'].any());checked=guard.launch(guard.prepare(f,out,vi,ac));self.assertTrue(checked['accepted'].all());self.check_worlds(states,[0,0],[5,19],L,out,ac,'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_effective_view_sampling_guard_graph_and_aliases(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import DomainInformationGuard
        base=self.fixtures();states=[base[i%2]for i in range(400)];L=DomainSoALayout(game_capacity=400);_,f=_torch_fields(states,L,'cuda');vi=torch.zeros(400,dtype=torch.int64,device='cuda');seeds=torch.arange(400,device='cuda')+5;ac=torch.ones(400,dtype=torch.bool,device='cuda')
        with DomainSoASampler(L,device='cuda',guard=False)as sampler,DomainInformationGuard(L,device='cuda',max_actions=64)as guard:
            p=sampler.prepare(f,vi,seeds,ac.int());sampler.launch(p);check=guard.prepare(f,p.outputs,vi,ac);guard.launch(check);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=sampler.launch(p);checked=guard.launch(check)
            g.replay();torch.cuda.synchronize();self.assertFalse(out['row_error'].any());self.assertTrue(checked['accepted'].all());self.check_worlds(states[:14],[0]*14,seeds[:14].cpu().tolist(),L,out,ac,'cuda')
            seeds.add_(1000);g.replay();torch.cuda.synchronize();self.assertTrue(checked['accepted'].all());self.check_worlds(states[:14],[0]*14,seeds[:14].cpu().tolist(),L,out,ac,'cuda')
            ref=int(f['choice_card_index'][0,0]);f['card_physical_index'][0,ref]=ref;g.replay();torch.cuda.synchronize();self.assertEqual(int(out['row_error'][0]),UNSUPPORTED_ROOT);self.assertFalse(checked['accepted'][0]);self.assertTrue(checked['accepted'][1:].all())


class IntegratedGuardSamplerTest(unittest.TestCase):
    def test_invariant_failure_rolls_back_every_state_column(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_information_guard import INFORMATION_CHANGED
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import CARD_INT_FIELDS
        st=_state(110);st['active_side']='a';st['sides']['a']['ap']=100
        from app.modules.card_game.engine.duel_v2.flow import legal_actions
        plays=[a for a in legal_actions(st,'a')if a['type']=='play_card'];self.assertTrue(plays)
        instance=int(plays[-1]['card_id'].split('-')[-1]);L=DomainSoALayout(game_capacity=2);_,f=_torch_fields([st,st],L)
        with cpu_oracle(L)as sampler:
            p=sampler.prepare(f,torch.zeros(2,dtype=torch.int64),torch.tensor([31,32]),torch.ones(2,dtype=torch.int32));native=sampler._kernel
            def corrupt(*args):
                native(*args)
                ref=int(((p.outputs['card_instance_slot'][0]==instance)&(p.outputs['card_side'][0]==0)).nonzero()[0]);key=CARD_INT_FIELDS.index('additional_ap_cost')
                p.outputs['card_int_presence'][0,ref,key]=2;p.outputs['card_int_value'][0,ref,key]=1
            sampler._kernel=corrupt;out=sampler.launch(p)
            self.assertEqual(out['row_error'].tolist(),[INFORMATION_CHANGED,0])
            for k in f:
                if k!='row_error':self.assertTrue(torch.equal(out[k][0],f[k][0]),k)
            sampler._kernel=native;out=sampler.launch(p);self.assertFalse(out['row_error'].any())
            self.assertFalse(sampler.supports_full_duel_rules);self.assertFalse(sampler.gpu_qualified)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_default_full_800_action_guard_capture(self):
        states=RevealedSamplerTest().fixtures()+EffectiveViewSamplerTest().fixtures()+VisibleDeckSamplerTest().fixtures()[0]
        states=[states[i%len(states)]for i in range(400)];L=DomainSoALayout(game_capacity=400);_,f=_torch_fields(states,L,'cuda');vi=torch.zeros(400,dtype=torch.int64,device='cuda');seed=torch.arange(400,device='cuda')+51;ac=torch.ones(400,dtype=torch.int32,device='cuda')
        # Choice roots are owned by a; other roots accept both observers.
        with DomainSoASampler(L,device='cuda')as sampler:
            p=sampler.prepare(f,vi,seed,ac);sampler.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=sampler.launch(p)
            g.replay();torch.cuda.synchronize();self.assertFalse(out['row_error'].any())
            for b in range(14):
                gold=oracle_determinize(states[b],'a',int(seed[b]));self.assertEqual(int(out['rng'][b]),gold['rng']);self.assertEqual(_zone_ids(out,b,1,0),[c['card_id']for c in gold['sides']['b']['hand']])
            seed.add_(1000);g.replay();torch.cuda.synchronize();self.assertFalse(out['row_error'].any())
            f['row_error'][0]=64;f['catalog_digest'][1,0]^=1;ac[2]=0
            g.replay();torch.cuda.synchronize();self.assertEqual(out['row_error'][:3].cpu().tolist(),[64,INVALID_INPUT,0])
            for k in f:
                if k!='row_error':self.assertTrue(torch.equal(out[k][:3],f[k][:3]),k)


class A03PublicTemporarySamplerTest(unittest.TestCase):
    def fixture(self,derived=False):
        st=_state(seed=7,first='b',a='murk',b='murk');team=st['sides']['b']
        for cid in ('A04','A03'):
            for zone in ('hand','deck'):
                card=next((c for c in team[zone]if c['card_id']==cid),None)
                if card:team[zone].remove(card);team['discard'].append(card);break
        team['discard'].append(card_instance(st,'AF01','b'));extra=card_instance(st,'A03','b');extra['ephemeral_turn']=st['turn']
        if derived:extra['derived']=True
        team['discard'].append(extra);return st

    def verify(self,states,seeds,out):
        self.assertFalse(out['row_error'].any())
        for b,(st,seed)in enumerate(zip(states,seeds)):
            gold=oracle_determinize(st,'a',seed)
            for side,zone in ((0,1),(1,0),(1,1)):
                self.assertEqual(_zone_ids(out,b,side,zone),[c['card_id']for c in gold['sides'][('a','b')[side]][('hand','deck')[zone]]],str((b,side,zone)))
            self.assertEqual(_eph(out,b,1,2),[c.get('ephemeral_turn')for c in gold['sides']['b']['discard']])
            self.assertEqual(int(out['rng'][b]),gold['rng']);self.assertEqual(int(out['next_entity'][b]),gold['next_entity']);self.assertEqual(int(out['next_instance'][b]),gold['next_instance'])

    def test_public_temporary_is_removed_from_original_counts_and_derived_is_not_original(self):
        states=[self.fixture(derived=d)for d in(False,True)for _ in range(8)];seeds=list(range(8))*2;L=DomainSoALayout(game_capacity=16);_,f=_torch_fields(states,L)
        with cpu_oracle(L)as sampler:
            out=sampler.determinize(f,torch.zeros(16,dtype=torch.int64),torch.tensor(seeds),torch.ones(16,dtype=torch.int32));self.verify(states,seeds,out)
        gold=[oracle_determinize(self.fixture(),'a',seed)for seed in range(8)]
        count=[sum(c.get('ephemeral_turn')is not None for c in g['sides']['b']['discard']if c['card_id']=='A03')for g in gold]
        self.assertIn(0,count);self.assertIn(1,count)

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_public_temporary_default_guard_replay(self):
        states=[self.fixture(derived=bool(i%2))for i in range(400)];L=DomainSoALayout(game_capacity=400);_,f=_torch_fields(states,L,'cuda');vi=torch.zeros(400,dtype=torch.int64,device='cuda');seeds=torch.arange(400,device='cuda')%8;ac=torch.ones(400,dtype=torch.int32,device='cuda')
        with DomainSoASampler(L,device='cuda')as sampler:
            p=sampler.prepare(f,vi,seeds,ac);sampler.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=sampler.launch(p)
            g.replay();torch.cuda.synchronize();self.verify(states[:16],seeds[:16].cpu().tolist(),out)
            seeds.add_(8);g.replay();torch.cuda.synchronize();self.verify(states[:16],seeds[:16].cpu().tolist(),out)


class AdvancedSamplerCapacityTest(unittest.TestCase):
    def fixture(self):
        from tests.test_duel_v2_domain_soa_codec import _natural_states
        roots=[s for s in _natural_states(120)if s['phase']in('playing','choice')]
        st=max(roots,key=lambda s:max(len(t['characters'])for t in s['sides'].values()))
        self.assertGreater(max(len(t['characters'])for t in st['sides'].values()),8)
        self.assertEqual(st['phase'],'playing');return st

    def verify(self,states,viewers,seeds,L,out,active,device):
        from app.modules.card_game.rl.batched_duel.domain_soa_card_order import DomainCardOrder
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
        from app.modules.card_game.rl.batched_duel.domain_soa_codec import decode_domain_states
        self.assertFalse(out['row_error'].any())
        with DomainCardOrder(L,device=device,allow_cpu_oracle=device=='cpu')as order:
            result=order.launch(order.prepare(out,active));self.assertFalse(result['errors'].any())
            worlds=decode_domain_states(DomainSoABatch(layout=L,**{k:v[:len(states)].cpu().numpy()for k,v in result['fields'].items()}))
        for b,w in enumerate(worlds):
            gold=oracle_determinize(states[b],('a','b')[viewers[b]],seeds[b]);gold.pop('events',None);self.assertEqual(w,gold)
            for side in('a','b'):self.assertEqual(list(w['sides'][side]['characters']),list(states[b]['sides'][side]['characters']))

    def test_larger_storage_keeps_disappeared_summon_identities(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoACapacityError
        st=self.fixture()
        with self.assertRaises(DomainSoACapacityError):_torch_fields([st],DomainSoALayout(game_capacity=1))
        L=DomainSoALayout(game_capacity=2,max_characters=16);_,f=_torch_fields([st,st],L);vi=torch.tensor([0,1]);seeds=torch.tensor([710096,710097]);active=torch.ones(2,dtype=torch.int32)
        with cpu_oracle(L)as sampler:
            out=sampler.determinize(f,vi,seeds,active);self.verify([st,st],[0,1],seeds.tolist(),L,out,active.bool(),'cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA required')
    def test_cuda_400_larger_storage_full_guard_and_complete_world_export(self):
        st=self.fixture();L=DomainSoALayout(game_capacity=400,max_characters=16);_,f=_torch_fields([st]*400,L,'cuda');vi=torch.arange(400,device='cuda')%2;seeds=torch.arange(400,device='cuda')+710096;active=torch.ones(400,dtype=torch.int32,device='cuda')
        with DomainSoASampler(L,device='cuda')as sampler:
            p=sampler.prepare(f,vi,seeds,active);sampler.launch(p);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=sampler.launch(p)
            g.replay();torch.cuda.synchronize();self.verify([st]*32,vi[:32].cpu().tolist(),seeds[:32].cpu().tolist(),L,out,active.bool(),'cuda')
            seeds.add_(1000);g.replay();torch.cuda.synchronize();self.verify([st]*32,vi[:32].cpu().tolist(),seeds[:32].cpu().tolist(),L,out,active.bool(),'cuda')
