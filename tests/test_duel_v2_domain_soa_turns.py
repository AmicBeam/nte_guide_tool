"""Direct replacement stage versus formal flow; not full-game qualification."""
from copy import deepcopy
import shutil
import unittest

import torch

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action
from app.modules.card_game.engine.duel_v2.state import draw, shuffle
from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states
from app.modules.card_game.rl.batched_duel.domain_soa_schema import ARRAY_NAMES, DomainSoALayout
from app.modules.card_game.rl.batched_duel.domain_soa_turns import (
    DomainSoATurns, KIND_MULLIGAN, KIND_REDRAW, INVALID_INPUT, DUPLICATE_SELECTION,
    UNKNOWN_HOOK, EMPTY_DRAW, CONT_BEGIN_TURN, CONT_FINISH_OPERATION,
)


def state(seed=7, preset=0):
    deck = STARTER_DECKS[preset % len(STARTER_DECKS)]
    return new_game(seed=seed, first_side='a', decks={'a': deck, 'b': deck})


def fields(states, device='cpu'):
    layout = DomainSoALayout(game_capacity=len(states))
    packed = encode_domain_states(states, layout, presentation='drop')
    return {name: torch.as_tensor(getattr(packed, name).copy(), device=device) for name in ARRAY_NAMES}


def controls(cols, sides, selections, kind=KIND_MULLIGAN):
    device = cols['phase'].device
    n = len(sides)
    ids = torch.zeros((n, 3), dtype=torch.int32, device=device)
    counts = []
    for row, selected in enumerate(selections):
        counts.append(len(selected))
        if selected:
            ids[row, :len(selected)] = torch.tensor(selected, device=device)
    return (torch.tensor(sides, dtype=torch.int32, device=device),
            torch.tensor(counts, dtype=torch.int32, device=device), ids,
            torch.full((n,), kind, dtype=torch.int32, device=device),
            torch.ones(n, dtype=torch.bool, device=device))


def selected_slots(cols, row, side, count):
    return [int(cols['card_instance_slot'][row, int(cols['zone_card_index'][row, side, 0, j])])
            for j in range(count)]


def zone_instances(result, original, row, side, zone):
    return [int(original['card_instance_slot'][row, int(result.zone_card_index[row, side, zone, j])])
            for j in range(int(result.zone_count[row, side, zone]))]


def formal_instance_slots(before, after, cols, row, side, zone):
    # Match stable physical identities through the original codec's slot table.
    mapping = {}
    for s, sk in enumerate(('a', 'b')):
        for z, zk in enumerate(('hand', 'deck', 'discard', 'banished')):
            for j, card in enumerate(before['sides'][sk].get(zk, [])):
                index = int(cols['zone_card_index'][row, s, z, j])
                mapping[card['instance_id']] = int(cols['card_instance_slot'][row, index])
    return [mapping[c['instance_id']] for c in after['sides'][('a','b')[side]][('hand','deck','discard','banished')[zone]]]


def oracle_backend(device):
    if device == 'cpu' and not (shutil.which('clang++') or shutil.which('g++')):
        raise unittest.SkipTest('Explicit compiler oracle unavailable')
    return DomainSoATurns(device=device, allow_cpu_oracle=device == 'cpu')


class TurnsTest(unittest.TestCase):
    def test_first_mulligan_matches_formal_all_presets_and_counts(self):
        states = [state(43 + j, j // 4) for j in range(4 * len(STARTER_DECKS))]
        cols = fields(states)
        selected = [selected_slots(cols, j, 0, j % 4)[::-1] for j in range(len(states))]
        with oracle_backend('cpu') as backend:
            result = backend.apply(cols, *controls(cols, [0]*len(states), selected))
            self.assertEqual(result.errors.tolist(), [0]*len(states))
            for j, before in enumerate(states):
                ids = [c['instance_id'] for c in before['sides']['a']['hand'][:j%4]][::-1]
                after = apply_action(before, 'a', {'type':'mulligan', 'card_ids':ids})
                for z in (0, 1):
                    self.assertEqual(zone_instances(result, cols, j, 0, z), formal_instance_slots(before, after, cols, j, 0, z))
                for key in ('rng','version','current_action_version','combat_group','event_seq'):
                    self.assertEqual(int(getattr(result, key)[j]), after[key], key)
                self.assertEqual(int(result.mulligan_count[j]), 1)
            self.assertFalse(backend.supports_full_duel_rules)
            self.assertFalse(backend.gpu_qualified)

    def test_second_mulligan_is_explicit_unfinished_continuation(self):
        before = apply_action(state(), 'a', {'type':'mulligan','card_ids':[]})
        cols = fields([before])
        with oracle_backend('cpu') as backend:
            result = backend.apply(cols, *controls(cols, [1], [[]]))
            self.assertEqual(int(result.errors[0]), 0)
            self.assertEqual(int(result.continuation_kind[0]), CONT_BEGIN_TURN)
            self.assertEqual(int(result.phase[0]), 0)  # begin_turn has NOT happened
            self.assertEqual(int(result.mulligan_count[0]), 2)
            self.assertEqual(int(result.continuation_side[0]), 0)

    def test_redraw_prefix_matches_formal_draw_then_shuffle(self):
        before = new_game(seed=123, skip_mulligan=True)
        before['phase'] = 'choice'
        before['pending_choice'] = {'kind':'hand_redraw','side':'a','prompt':'手牌调度','max_select':3,'cards':before['sides']['a']['hand'][:]}
        cols = fields([before])
        chosen = selected_slots(cols, 0, 0, 3)[::-1]
        expected = deepcopy(before)
        team = expected['sides']['a']
        removed, team['hand'] = team['hand'][:3], team['hand'][3:]
        draw(expected, 'a', 3)
        team['deck'].extend(removed)
        shuffle(expected, team['deck'])
        with oracle_backend('cpu') as backend:
            result = backend.apply(cols, *controls(cols, [0], [chosen], KIND_REDRAW))
            self.assertEqual(int(result.errors[0]), 0)
            self.assertEqual(int(result.continuation_kind[0]), CONT_FINISH_OPERATION)
            self.assertEqual(int(result.rng[0]), expected['rng'])
            for z in (0, 1):
                self.assertEqual(zone_instances(result, cols, 0, 0, z), formal_instance_slots(before, expected, cols, 0, 0, z))

    def test_failures_preserve_source_and_prior_error(self):
        cols = fields([state() for _ in range(4)])
        chosen = [selected_slots(cols, j, 0, 1) for j in range(4)]
        chosen[0] *= 2
        cols['row_error'][1] = 83
        cols['mulligan_count'][2] = 2  # never write past two completion slots
        cols['zone_count'][3, 0, 1] = 0
        with oracle_backend('cpu') as backend:
            result = backend.apply(cols, *controls(cols, [0]*4, chosen))
            self.assertEqual(result.errors.tolist(), [DUPLICATE_SELECTION,83,INVALID_INPUT,EMPTY_DRAW])
            for attr in ('zone_card_index','zone_count','rng','mulligan_count','version','event_seq'):
                self.assertTrue(torch.equal(getattr(result, attr), cols[attr]), attr)

    def test_malformed_native_types_rejected_before_launch(self):
        cols = fields([state()])
        cols['character_hp'] = cols['character_hp'].to(torch.int64)
        with oracle_backend('cpu') as backend:
            with self.assertRaises(ValueError):
                backend.prepare(cols, *controls(cols, [0], [[]]))

    def test_noncontiguous_live_columns_refresh_and_output_independence(self):
        raw = fields([state(), state(11)])
        cols = {}
        for name, value in raw.items():
            storage = torch.empty((4,)+value.shape[1:], dtype=value.dtype)
            view = storage[::2]
            view.copy_(value)
            cols[name] = view
        args = controls(cols, [0,0], [[],[]])
        with oracle_backend('cpu') as backend:
            prepared = backend.prepare(cols, *args)
            result = backend.launch(prepared)
            old_rng = result.rng.clone()
            cols['rng'].add_(3)
            backend.launch(prepared)
            self.assertFalse(torch.equal(result.rng, old_rng))
            self.assertFalse(result.rng.data_ptr() == cols['rng'].data_ptr())

    def test_rng_budget_and_stale_catalog_roll_back(self):
        cols = fields([state(j) for j in range(16)])
        cols['catalog_digest'][0, 0] ^= 1
        if not (shutil.which('clang++') or shutil.which('g++')):
            self.skipTest('Explicit compiler oracle unavailable')
        with DomainSoATurns(device='cpu', allow_cpu_oracle=True, max_draws=1) as backend:
            result = backend.apply(cols, *controls(cols, [0]*16, [[] for _ in range(16)]))
            self.assertNotEqual(int(result.errors[0]), 0)
            self.assertTrue(bool((result.errors[1:] == 5).any()))
            failed = result.errors != 0
            self.assertTrue(torch.equal(result.rng[failed], cols['rng'][failed]))
            self.assertTrue(torch.equal(result.zone_card_index[failed], cols['zone_card_index'][failed]))

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_cuda_redraw_and_hook_rejection(self):
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import CHARACTER_BOOL_FIELDS, CHARACTER_VOCAB
        before = new_game(seed=123, skip_mulligan=True)
        before['phase'] = 'choice'
        before['pending_choice'] = {'kind':'hand_redraw','side':'a','prompt':'调度','cards':before['sides']['a']['hand'][:]}
        cpu = fields([before, before, before])
        cols = {k:v.to('cuda') for k,v in cpu.items()}
        selections = [selected_slots(cpu, j, 0, 2) for j in range(3)]
        aw = CHARACTER_BOOL_FIELDS.index('awakened')
        cols['character_template'][1,0,0] = CHARACTER_VOCAB['xiaozhi']
        cols['character_bool_presence'][1,0,0,aw] = 2
        cols['character_bool_value'][1,0,0,aw] = True
        cols['catalog_digest'][2,0] ^= 1
        with oracle_backend('cuda') as backend:
            result = backend.apply(cols, *controls(cols, [0]*3, selections, KIND_REDRAW))
            torch.cuda.synchronize()
            self.assertEqual(result.errors.cpu().tolist(), [0,UNKNOWN_HOOK,INVALID_INPUT])
            self.assertEqual(int(result.continuation_kind[0]), CONT_FINISH_OPERATION)
            self.assertTrue(torch.equal(result.zone_card_index[1:], cols['zone_card_index'][1:]))
            expected = deepcopy(before)
            hand = expected['sides']['a']['hand']
            removed = hand[:2]
            expected['sides']['a']['hand'] = hand[2:]
            draw(expected, 'a', 2)
            expected['sides']['a']['deck'].extend(removed)
            shuffle(expected, expected['sides']['a']['deck'])
            self.assertEqual(int(result.rng[0]), expected['rng'])
            for z in (0,1):
                self.assertEqual(zone_instances(result, cols, 0, 0, z), formal_instance_slots(before, expected, cpu, 0, 0, z))

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA required')
    def test_cuda_400_rows_graph_live_inputs_and_errors(self):
        states = [state(300+j, j) for j in range(400)]
        cpu = fields(states)
        cols = {k:v.to('cuda') for k,v in cpu.items()}
        selected = [selected_slots(cpu, j, 0, j % 4) for j in range(400)]
        args = controls(cols, [0]*400, selected)
        with oracle_backend('cuda') as backend:
            prepared = backend.prepare(cols, *args)
            backend.launch(prepared)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = backend.launch(prepared)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(result.errors.cpu().tolist(), [0]*400)
            for j in (0,1,2,3,53,399):
                ids = [c['instance_id'] for c in states[j]['sides']['a']['hand'][:j%4]]
                expected = apply_action(states[j], 'a', {'type':'mulligan','card_ids':ids})
                self.assertEqual(int(result.rng[j]), expected['rng'])
                for z in (0,1):
                    self.assertEqual(zone_instances(result, cols, j, 0, z), formal_instance_slots(states[j], expected, cpu, j, 0, z))
            cols['row_error'][0] = 91
            cols['mulligan_count'][1] = 2
            args[2][2, 0] = 2147483647
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(result.errors[:3].cpu().tolist(), [91,INVALID_INPUT,2])
            self.assertEqual(int(result.version[0]), int(cols['version'][0]))


if __name__ == '__main__':
    unittest.main()
