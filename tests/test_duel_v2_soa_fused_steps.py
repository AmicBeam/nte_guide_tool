"""CPU control-flow/lease fakes for real fused methods; no GPU qualification."""
from copy import deepcopy
import inspect
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch as t

from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules, _validate_fused_clean
from app.modules.card_game.rl.batched_search.device_tiled_soa_rules import DeviceTiledSoARules


def canonical_image():
    return NS(entries={'clean': 0}, function_ids=(
        'func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:seven_clean:120:0',
        'func:app.modules.card_game.rl.league_rollout:clean:90:1'))


class LeaseStore:
    """Independent CPU leases: validate generations/owners and never alias puts."""
    def __init__(self, rows, log, failure):
        self.rows, self.log, self.next_slot = rows, log, 100
        self.live = {(i, 1): (1000+i, 101+i) for i in range(rows)}
        self.columns = {(i, 1): {'i64': t.tensor([200+i])} for i in range(rows)}
        self.puts, self.failure = 0, failure
        self.released = []

    def put_device(self, batch, owners, active, *, errors):
        self.puts += 1
        errors = errors.clone()
        if self.puts == 2 and self.failure == 'put' and bool(active[7]):
            errors[7] |= 21
        slots = t.full((self.rows,), -1, dtype=t.int64)
        generations = t.zeros(self.rows, dtype=t.int64)
        accepted = active.bool() & (errors == 0)
        for row in t.nonzero(accepted).flatten().tolist():
            slot = self.next_slot
            self.next_slot += 1
            generation = slot+17
            slots[row], generations[row] = slot, generation
            self.live[slot, generation] = int(owners[row]), int(batch.roots[row])
            self.columns[slot, generation] = {'i64': batch.buffers['i64'][:, row].clone()}
        self.log.append(('put', accepted.clone(), slots.clone()))
        return DeviceStateBatch(self, slots, generations, owners.clone(), accepted.clone(), errors.clone())

    def gather_into_device(self, states, batch, *, shared_reads):
        assert shared_reads is True
        for row in t.nonzero(states.active).flatten().tolist():
            key = int(states.slots[row]), int(states.generations[row])
            owner, root = self.live[key]
            assert owner == int(states.owners[row])
            batch.roots[row] = root
            batch.buffers['i32'][0, row] = 2  # compacted live EventList count
            batch.buffers['i64'][:, row].copy_(self.columns[key]['i64'])
        self.log.append(('gather', states.active.clone(), batch.roots.clone()))
        return batch

    def release_device(self, states):
        released = []
        for row in t.nonzero(states.active).flatten().tolist():
            key = int(states.slots[row]), int(states.generations[row])
            assert self.live[key][0] == int(states.owners[row])
            del self.live[key]
            del self.columns[key]
            self.released.append(key)
            released.append(row)
        self.log.append(('release', released))


def fixture(failure='compact'):
    rows, log = 9, []
    rules = object.__new__(DeviceSoARules)
    rules.t, rules.device, rules.rows, rules._keepalive = t, t.device('cpu'), rows, []
    rules._fused_clean_checked = True
    rules.store = LeaseStore(rows, log, failure)
    schema = NS(root=0, pools=(NS(name='Game', base=100, capacity=100),
                              NS(name='EventList', used=0, capacity=8)))
    batch = NS(schema=schema, roots=t.arange(101, 110, dtype=t.int32),
               buffers={'i32': t.tensor([[4, 8, 8, 8, 8, 8, 7, 8, 8]], dtype=t.int32),
                        'i64': t.arange(400, 409)[None]})
    scratch = NS(roots=t.full((rows,), -777, dtype=t.int32))
    rules._shared_batch, rules._shared_scratch = batch, scratch
    rules._compact_workspace = object()
    active = t.tensor([True, True, True, True, True, False, True, True, True])
    states = DeviceStateBatch(rules.store, t.arange(rows), t.ones(rows, dtype=t.int64),
        t.arange(1000, 1009), active, t.tensor([0, 0, 4, 0, 0, 0, 0, 0, 0], dtype=t.int32))

    def call(state, roots, statuses):
        result = NS(state=state, scratch=scratch, roots=roots, statuses=statuses)
        result.typed_result = lambda: NS(schema=schema, buffers=state.buffers, roots=result.roots)
        return result

    def reset(state, transient):
        log.append(('reset', state.roots.clone()))
        transient.roots.zero_()
    rules.executor = NS(image=canonical_image(), reset_workspace=reset)

    def compact(state, workspace):
        assert workspace is rules._compact_workspace
        log.append(('compact', state.roots.clone(), deepcopy(state.buffers)))
        # Real compact_rows rejects root=0 with STATUS_BOUNDS=1. Its caller
        # must mask this secondary status using pre-compaction eligibility.
        status = (state.roots == 0).to(t.int32)
        if (len([item for item in log if item[0] == 'compact']) == 2
                and failure == 'compact' and int(state.roots[7]) != 0):
            status[7] = 21
        return NS(schema=schema, buffers=deepcopy(state.buffers), roots=state.roots.clone()), status, object()
    rules.compactor = NS(compact_into_device=compact)

    def step(self, incoming, actors, decision, indices, *, physical=False, commit=True):
        assert physical is False
        errors = incoming.errors.clone()
        mask = (incoming.active & (errors == 0)).view(t.uint8)
        log.append(('step', commit, mask.clone()))
        status = t.zeros((rows, 4), dtype=t.int32)
        assert commit is False, 'fused capacity fallback must not execute the action again'
        assert len([item for item in log if item[0] == 'step']) == 1
        roots = t.arange(111, 120, dtype=t.int32)
        roots[4] = 0
        status[3, 0] = 9
        self._first_call = call(batch, roots, status)
        return self._first_call, errors, mask
    rules._step_device = MethodType(step, rules)

    def invoke(self, entry, state, transient, refs, mask):
        assert entry == 'clean'
        assert bool((refs == 0).all()), 'stale transient references survived reset'
        log.append(('clean', mask.clone(), state.roots.clone()))
        # Like a masked device kernel, clean changes only its active typed
        # columns. Full-pool rows retain the first action's exact result.
        state.buffers['i32'][0].add_(mask.to(t.int32))
        state.buffers['i64'][0].add_(mask.to(t.int64)*100)
        status = t.zeros((rows, 4), dtype=t.int32)
        if len([item for item in log if item[0] == 'clean']) == 1:
            status[6, 0] = 13
        else:
            status[8, 0] = 22
        # CUDA skips inactive rows: never synthesize errors on masked rows.
        status[:, 0] *= mask.to(t.int32)
        roots = state.roots.clone()
        roots[status[:, 0] != 0] = 0
        return call(state, roots, status)
    rules._invoke = MethodType(invoke, rules)
    return rules, states, batch, log


class SoAFusedStepTests(unittest.TestCase):
    def test_mixed_rows_exact_final_errors_roots_and_independent_lease_cleanup(self):
        for failure in ('compact', 'put'):
            with self.subTest(failure=failure):
                self.check_mixed_rows(failure)

    def check_mixed_rows(self, failure):
        rules, states, batch, log = fixture(failure)
        original = tuple(getattr(states, name).clone() for name in ('slots', 'generations', 'owners', 'active', 'errors'))
        with patch.object(t.cuda, 'is_current_stream_capturing', return_value=True):
            result = rules.step_clean_device(states, object(), object(), object())
        self.assertEqual(result.errors.tolist(), [0, 0, 4, 9, 150, 0, 13, 21, 22])
        self.assertEqual(result.active.tolist(), [True, True, False, False, False, False, False, False, False])
        t.testing.assert_close(result.owners, states.owners)
        self.assertEqual(rules.store.live[int(result.slots[0]), int(result.generations[0])], (1000, 111))
        self.assertEqual(rules.store.live[int(result.slots[1]), int(result.generations[1])], (1001, 112))
        # Direct fallback commits the first step's columns, not original input
        # columns or a second action's result. Final clean adds exactly 100.
        self.assertEqual(rules.store.columns[int(result.slots[1]), int(result.generations[1])]['i64'].tolist(), [501])
        self.assertEqual(len(rules.store.live), 11)  # all nine originals + two finals
        self.assertEqual(len(rules.store.released), 2)  # successful and failed-clean intermediates
        self.assertTrue(all(key[0] >= 100 for key in rules.store.released))
        for name, value in zip(('slots', 'generations', 'owners', 'active', 'errors'), original):
            t.testing.assert_close(getattr(states, name), value)
        steps = [entry for entry in log if entry[0] == 'step']
        self.assertEqual([entry[1] for entry in steps], [False])
        clean = [entry for entry in log if entry[0] == 'clean']
        self.assertEqual(clean[0][1].bool().tolist(), [True, False, False, False, False, False, True, False, False])
        self.assertEqual(clean[1][1].bool().tolist(), [False, True, False, False, False, False, False, False, True])
        # Root selection happens before clearing stale scratch/memo state.
        reset = next(entry for entry in log if entry[0] == 'reset')
        self.assertEqual(reset[1].tolist(), [111, 102, 103, 104, 105, 106, 117, 108, 109])
        compactions = [entry for entry in log if entry[0] == 'compact']
        self.assertEqual([entry[1].tolist() for entry in compactions], [
            [111, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 112, 0, 0, 0, 0, 0, 118, 119],
            [0, 112, 0, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(compactions[1][2]['i32'][0, [1, 7, 8]].tolist(), [8, 8, 8])
        self.assertEqual(compactions[1][2]['i64'][0, [1, 7, 8]].tolist(), [401, 407, 408])
        self.assertEqual(rules._first_call.statuses[:, 0].tolist(), [0, 0, 0, 9, 0, 0, 0, 0, 0])
        self.assertEqual(rules._first_call.roots.tolist(), [111, 112, 113, 114, 0, 116, 117, 118, 119])
        self.assertEqual([entry[1] for entry in log if entry[0] == 'release'], [[1, 8]])

    def test_physical_and_non_boolean_modes_rejected_before_any_rule_or_tape(self):
        for value in (True, 0, 1, None):
            rules, states, _, log = fixture()
            rules.raw_tape = NS(append_device=lambda *a: self.fail('physical tape must not run'))
            with self.assertRaises(ValueError):
                rules.step_clean_device(states, object(), object(), object(), physical=value)
            self.assertEqual(log, [])

    def test_empty_capacity_fallback_never_traverses_or_commits_first_call_roots(self):
        rules, states, batch, log = fixture()
        batch.buffers['i32'].fill_(4)
        with patch.object(t.cuda, 'is_current_stream_capturing', return_value=True):
            result = rules.step_clean_device(states, object(), object(), object())
        self.assertEqual(len([item for item in log if item[0] == 'step']), 1)
        compactions = [item for item in log if item[0] == 'compact']
        self.assertTrue(bool((compactions[1][1] == 0).all()))
        self.assertTrue(bool((compactions[2][1] == 0).all()))
        puts = [item for item in log if item[0] == 'put']
        self.assertFalse(bool(puts[1][1].any()))
        self.assertFalse(bool(puts[2][1].any()))
        self.assertEqual(rules.store.released, [])
        self.assertEqual(result.errors.tolist(), [0, 0, 4, 9, 150, 0, 13, 0, 0])
        self.assertEqual(result.active.tolist(), [True, True, False, False, False, False, False, True, True])
        self.assertEqual(rules._first_call.statuses[3, 0].item(), 9)

    def test_first_vm_error_two_and_clean_error_22_do_not_gain_root_or_bounds_errors(self):
        rules, states, _, log = fixture()
        original = rules._step_device
        def failed(*args, **kwargs):
            result = original(*args, **kwargs)
            result[0].statuses[3, 0] = 2
            result[0].roots[3] = 0
            return result
        rules._step_device = failed
        with patch.object(t.cuda, 'is_current_stream_capturing', return_value=True):
            result = rules.step_clean_device(states, object(), object(), object())
        self.assertEqual(int(result.errors[3]), 2)
        self.assertEqual(int(result.errors[8]), 22)
        self.assertEqual(int(result.errors[4]), 150)  # error-free invalid root
        self.assertEqual(int(rules._first_call.statuses[3, 0]), 2)

    def test_canonical_allocation_proof_rejects_changed_clean_and_wrong_frozen_entry(self):
        _validate_fused_clean(canonical_image())
        for source in ("def clean(s):\n s['events']=[1]\n s.pop('_public_board',None)\n return s\n",
                       "def clean(s):\n s['events']=[]\n s['other']=[]\n s.pop('_public_board',None)\n return s\n",
                       "def clean(s):\n s['events']=[]\n return s.copy()\n"):
            with patch('inspect.getsource', return_value=source), self.assertRaises(ValueError):
                _validate_fused_clean(canonical_image())
        images = [NS(entries={'clean': True}, function_ids=canonical_image().function_ids),
                  NS(entries={'clean': 2}, function_ids=canonical_image().function_ids),
                  NS(entries={'clean': 0}, function_ids=('func:elsewhere:clean:1:0',)),
                  NS(entries={'clean': 0}, function_ids=canonical_image().function_ids[:1])]
        for image in images:
            with self.assertRaises(ValueError):
                _validate_fused_clean(image)
        rules, states, _, log = fixture()
        rules._fused_clean_checked = False
        rules.executor.image = images[-1]
        with self.assertRaises(ValueError):
            rules.step_clean_device(states, object(), object(), object())
        self.assertEqual(log, [])

    def test_missing_event_pool_and_upper_bound_root_fail_before_clean(self):
        rules, states, batch, log = fixture()
        batch.schema.pools = batch.schema.pools[:1]
        with self.assertRaisesRegex(ValueError, 'EventList'):
            rules.step_clean_device(states, object(), object(), object())
        self.assertEqual([entry[0] for entry in log], ['step'])
        rules, states, batch, log = fixture()
        original_step = rules._step_device
        def bad_root(*args, **kwargs):
            result = original_step(*args, **kwargs)
            if kwargs.get('commit') is False:
                result[0].roots[0] = 201  # Game base=100, capacity=100
            return result
        rules._step_device = bad_root
        with patch.object(t.cuda, 'is_current_stream_capturing', return_value=True):
            result = rules.step_clean_device(states, object(), object(), object())
        self.assertEqual(int(result.errors[0]), 150)
        self.assertFalse(bool(result.active[0]))
        self.assertTrue(all(not bool(entry[1][0]) for entry in log if entry[0] == 'clean'))
        self.assertEqual(len([entry for entry in log if entry[0] == 'step']), 1)
        self.assertFalse(bool([entry for entry in log if entry[0] == 'put'][1][1][0]))

    def test_tile_preparation_is_explicit_idempotent_and_refuses_capture(self):
        from app.modules.card_game.rl.batched_search.device_search import DeviceGumbelSearch
        self.assertIs(inspect.signature(DeviceGumbelSearch.__init__).parameters['fused_rule_steps'].default, False)
        rules = object.__new__(DeviceTiledSoARules)
        rules.t, rules.closed = t, False
        prepared, allocated = [], []
        rules.inner = NS(prepare_fused_steps=lambda: prepared.append(True))
        rules._sites, rules._site_index = {'step': [object()]}, {'step': 0}
        def queue():
            value = object()
            allocated.append(value)
            return value
        rules._new_queue = queue
        self.assertNotIn('step_clean', rules._sites)
        with patch.object(t.cuda, 'is_current_stream_capturing', return_value=False):
            rules.prepare_fused_steps(); rules.prepare_fused_steps()
        self.assertEqual(len(allocated), 1)
        self.assertEqual(rules._sites['step_clean'], allocated)
        with patch.object(t.cuda, 'is_current_stream_capturing', return_value=True), self.assertRaises(RuntimeError):
            rules.prepare_fused_steps()
        for value in (True, 0):
            with self.assertRaises(ValueError):
                rules.step_clean_device(object(), object(), object(), object(), physical=value)


if __name__ == '__main__':
    unittest.main()
