"""Explicit synthetic CPU rule-controller protocol; no GPU/aux qualification."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from app.modules.card_game.rl.batched_duel.codec import pack_states
from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
from app.modules.card_game.rl.batched_duel.schema import GraphLayout
from app.modules.card_game.rl.batched_duel.device_graph_storage import CompactDeviceGraphStore
from app.modules.card_game.rl.batched_duel.device_rule_protocol import (
    DeviceStateBatch, DeviceDecision, DeviceMeta, DeviceValueObservation,
)
from app.modules.card_game.rl.batched_search.device_rule_policy import DeviceRulePolicy
from tests.test_duel_v2_device_pure_value_tape import CheckedPool


class Rules:
    def __init__(self):
        self.store = CheckedPool()
        self.actors = torch.tensor([0, 1])
        self.versions = torch.tensor([3, 7])
        self.phases = torch.tensor([1, 0], dtype=torch.int32)
        self.finished = torch.zeros(2, dtype=torch.bool)
        self.meta_errors = torch.zeros(2, dtype=torch.int32)
        self.rule_errors = torch.zeros_like(self.meta_errors)
        self.value_errors = torch.zeros_like(self.meta_errors)
        self.chosen = torch.tensor([2, 1])
        self.x = torch.tensor([[12., 34., 56.], [78., 91., 23.]])
        self.action_store = CompactDeviceGraphStore(slots=4, node_pages=16, edge_pages=16,
            payload_pages=16, node_page_bytes=128, edge_page_bytes=128, payload_page_bytes=128,
            max_node_pages_per_slot=8, max_edge_pages_per_slot=8, max_payload_pages_per_slot=8,
            device='cpu', allow_cpu_reference=True)
        actions = [[{'type': 'concede'}, {'type': 'end_turn'}, {'type': 'attack'}] for _ in range(2)]
        packed = pack_states(actions, GraphLayout(2, 32, 32, 128))
        graph = DeviceGraph(packed.layout, *(torch.tensor(np.array(getattr(packed, n), copy=True))
            for n in ('nodes', 'edges', 'payload', 'counts', 'roots')))
        owners = torch.tensor([40, 41]);active = torch.ones(2, dtype=torch.bool)
        self.lease = self.action_store.put_device(graph, owners, active)
        self.refs = torch.full((2, 4), -999, dtype=torch.int32)
        for row in (0, 1):
            root = int(graph.roots[row]);offset = int(graph.nodes[row, root, 1])
            self.refs[row, :3] = graph.edges[row, offset:offset+3, 0]
        self.counts = torch.tensor([3, 2])
        self.valid = torch.arange(4)[None, :] < self.counts[:, None]
        self.hook = None;self.value_hook = None;self.calls = []

    def roots(self):
        return DeviceStateBatch(self.store, torch.tensor([0, 1]), torch.ones(2, dtype=torch.int64),
            torch.tensor([40, 41]), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.int32))

    def metadata_device(self, roots):
        self.calls.append(('meta', roots))
        return DeviceMeta(self.actors, self.phases, torch.full((2,), -2), self.finished,
            self.versions, self.meta_errors)

    def rule_decision_device(self, roots, actors):
        self.calls.append(('rule', roots))
        decision = DeviceDecision(roots, self.action_store, self.lease.slots.clone(),
            self.lease.generations.clone(), self.refs.clone(), torch.full((2, 4), -1),
            self.counts, self.valid, torch.empty((2, 0)), torch.empty((2, 4, 0)),
            torch.zeros(2, dtype=torch.int32))
        if self.hook is not None:decision = self.hook(decision)
        self.last_raw = decision
        return decision, self.chosen, self.rule_errors

    def value_observation_device(self, roots, viewers):
        self.calls.append(('value', roots))
        value = DeviceValueObservation(self.x, torch.ones((2, 1)), torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.float64), self.value_errors)
        if self.value_hook is not None:value = self.value_hook(value)
        return value


class RulePolicyTests(unittest.TestCase):
    def fixture(self):
        rules = Rules();self.addCleanup(rules.action_store.close)
        policy = DeviceRulePolicy(rules, rows=2, max_actions=4, device='cpu',
            allow_cpu_reference=True, allow_synthetic=True)
        return policy, rules, rules.roots()

    def run_policy(self, policy, rules, roots):
        ids = torch.zeros((2, 2), dtype=torch.int64)
        policy.initialize(roots, rules.actors, torch.zeros((2, 5), dtype=torch.int64), ids, ids)
        return policy.result_device()

    def exploration_fixture(self, *, retries=128):
        rules = Rules();self.addCleanup(rules.action_store.close)
        seeds = (202610180931, 202610180932)
        policy = DeviceRulePolicy(rules, rows=2, max_actions=4, device='cpu',
            allow_cpu_reference=True, allow_synthetic=True, explore_first_six=True,
            exploration_seeds=seeds, exploration_max_retries=retries)
        self.addCleanup(policy.close)
        roots = rules.roots();steps = torch.zeros(2, dtype=torch.int64)
        loop = SimpleNamespace(search=policy, rules=rules, rows=2, device=torch.device('cpu'),
            _steps=steps, steps_device=steps, _job_seeds=torch.tensor(seeds), _owners=roots.owners.clone())
        return policy, rules, roots, loop, seeds

    def test_actual_rule_index_order_observation_and_zero_search_schedule(self):
        policy, rules, roots = self.fixture()
        with self.assertRaises(RuntimeError):policy.result_device()
        for step in range(4):
            rules.chosen.copy_(torch.tensor([0 if step % 2 else 2, 1]))
            result = self.run_policy(policy, rules, roots)
            self.assertEqual(result.actions.tolist(), rules.chosen.tolist())
            self.assertEqual(result.complete.tolist(), [True, True])
            self.assertTrue(torch.equal(result.decision.observations, rules.x))
            self.assertIs(result.decision.observations, rules.x)
            self.assertEqual(result.decision.observations.shape, (2, 3))
            self.assertEqual(rules.last_raw.observations.shape, (2, 0))
            for name in ('action_refs', 'action_types', 'action_slots', 'action_generations', 'candidates'):
                self.assertIs(getattr(result.decision, name), getattr(rules.last_raw, name))
            self.assertIs(result.decision.action_pool, rules.action_store)
            self.assertEqual(result.policy.argmax(-1).tolist(), rules.chosen.tolist())
            self.assertEqual(result.policy.sum(-1).tolist(), [1., 1.])
            for name in ('requested', 'completed', 'visits', 'visit_policy'):
                self.assertEqual(getattr(result, name).sum().item(), 0)
            self.assertEqual(policy.unfinished_device.tolist(), [0, 0]);policy.tick()
            rules.versions += 1;rules.actors.copy_(1-rules.actors)
        self.assertEqual(sum(name == 'rule' for name, _ in rules.calls), 4)
        self.assertEqual(sum(name == 'value' for name, _ in rules.calls), 4)
        self.assertFalse(policy.noise or policy.counts_as_search or policy.gpu_qualified)

    def test_padding_and_inactive_rows_do_not_create_choices_or_errors(self):
        policy, rules, roots = self.fixture()
        roots.active[1] = False
        rules.chosen[1] = -999;rules.counts[1] = 0
        rules.valid[1].zero_();rules.refs[1].fill_(-999)
        rules.rule_errors[1] = 512;rules.value_errors[1] = 1024;rules.x[1].fill_(float('nan'))
        result = self.run_policy(policy, rules, roots)
        self.assertEqual(result.actions.tolist(), [2, -1])
        self.assertEqual(result.complete.tolist(), [True, False])
        self.assertEqual(result.errors.tolist(), [0, 0])
        self.assertEqual(result.policy[1].sum().item(), 0)

    def test_lease_owner_generation_refs_actor_and_rule_errors_fail_closed(self):
        for mode in ('stale_root', 'wrong_owner', 'stale_action', 'dead_action', 'foreign_action_owner',
                     'bad_ref', 'bad_choice', 'bad_count', 'bad_prefix', 'wrong_actor', 'finished',
                     'foreign_decision', 'decision_error', 'rule_error', 'root_error'):
            with self.subTest(mode=mode):
                policy, rules, roots = self.fixture()
                if mode == 'stale_root':roots.generations[0] += 1
                if mode == 'wrong_owner':roots.owners[0] = 999
                if mode == 'stale_action':rules.lease.generations[0] += 1
                if mode == 'dead_action':rules.action_store.live[0] = 0
                if mode == 'foreign_action_owner':rules.action_store.owner[0] = 999
                if mode == 'bad_ref':rules.refs[0, 0] = 999
                if mode == 'bad_choice':rules.chosen[0] = 4
                if mode == 'bad_count':rules.counts[0] = 5
                if mode == 'bad_prefix':rules.valid[0, 1] = False
                if mode == 'wrong_actor':rules.actors[0] = 9
                if mode == 'finished':rules.finished[0] = True
                if mode == 'foreign_decision':
                    rules.hook = lambda d:replace(d, states=replace(d.states, owners=d.states.owners+torch.tensor([1, 0])))
                if mode == 'decision_error':
                    rules.hook = lambda d:replace(d, errors=torch.tensor([2048, 0], dtype=torch.int32))
                if mode == 'rule_error':rules.rule_errors[0] = 4096
                if mode == 'root_error':roots.errors[0] = 8192
                result = self.run_policy(policy, rules, roots)
                self.assertFalse(result.complete[0].item())
                self.assertEqual(result.actions[0].item(), -1)
                self.assertNotEqual(result.errors[0].item(), 0)
                if mode == 'rule_error':self.assertTrue(result.errors[0].item() & 4096)
                if mode == 'root_error':self.assertTrue(result.errors[0].item() & 8192)
                self.assertEqual(result.actions[1].item(), 1)

    def test_value_query_errors_context_finite_terminal_and_revision(self):
        for mode in ('nan', 'context', 'terminal', 'terminal_value', 'error', 'revision', 'phase', 'actor',
                     'root_lease_invalidated', 'action_lease_invalidated'):
            policy, rules, roots = self.fixture()
            def alter(value):
                if mode == 'nan':value.observations[0, 0] = float('inf')
                if mode == 'context':value.viewer_is_actor[0, 0] = 0
                if mode == 'terminal':value.terminal[0] = True
                if mode == 'terminal_value':value.terminal_value[0] = float('nan')
                if mode == 'error':value.errors[0] = 16384
                if mode == 'revision':rules.versions[0] += 1
                if mode == 'phase':rules.phases[0] += 1
                if mode == 'actor':rules.actors[0] = 1-rules.actors[0]
                if mode == 'root_lease_invalidated':rules.store.live[0] = False
                if mode == 'action_lease_invalidated':rules.action_store.generation[0] += 1
                return value
            rules.value_hook = alter
            result = self.run_policy(policy, rules, roots)
            self.assertFalse(result.complete[0].item(), mode)
            self.assertEqual(result.actions[0].item(), -1)
            self.assertTrue(result.complete[1].item())

    def test_exact_protocol_cpu_controls_foreign_pool_and_no_host_reads(self):
        policy, rules, roots = self.fixture()
        for reference, synthetic in ((False, False), (False, True), (True, False)):
            with self.assertRaises(ValueError):DeviceRulePolicy(rules, rows=2, max_actions=4,
                device='cpu', allow_cpu_reference=reference, allow_synthetic=synthetic)
        rules.hook = lambda d:replace(d, action_pool=object())
        with self.assertRaises(ValueError):self.run_policy(policy, rules, roots)
        rules.hook = lambda d:replace(d, counts=d.counts.int())
        with self.assertRaises(ValueError):self.run_policy(policy, rules, roots)
        rules.hook = None
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU read')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar read')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('truth read')):
            result = self.run_policy(policy, rules, roots)
        self.assertEqual(result.actions.tolist(), [2, 1])

    def test_constructor_bounds_and_formal_frozen_teacher_binding(self):
        from app.modules.card_game.rl.batched_search.device_soa_rules import DeviceSoARules
        rules = Rules();self.addCleanup(rules.action_store.close)
        for kwargs in ({'rows': True}, {'rows': 0}, {'rows': 4097},
                       {'max_actions': False}, {'max_actions': 0}, {'max_actions': 16385}):
            args = dict(rows=2, max_actions=4);args.update(kwargs)
            with self.assertRaises(ValueError):
                DeviceRulePolicy(rules, **args, device='cpu',
                    allow_cpu_reference=True, allow_synthetic=True)
        with self.assertRaises(TypeError):
            DeviceRulePolicy(rules, rows=2, max_actions=4, device='cpu',
                allow_cpu_reference=1, allow_synthetic=True)
        # Type/configuration admission is tested with CUDA availability mocked;
        # allocation is forbidden, so this never executes or emulates CUDA.
        formal = DeviceSoARules.__new__(DeviceSoARules)
        formal.require_teacher = True;formal.rows = 2;formal.max_actions = 4
        formal.device = torch.device('cuda:0');formal.store = object();formal.action_store = object()
        prefix = 'func:app.modules.card_game.rl.batched_duel.gpu_rule_entries:'
        good_image = SimpleNamespace(entries={'teacher_continue': 0, 'rule_decision': 1},
            function_ids=(prefix+'teacher_continue:1:0', prefix+'seven_rule_decision:1:1'))
        formal.executor = SimpleNamespace(closed=False, execute_only=False, image=good_image,
            prepare_result_path=lambda *a:None, select_result=lambda *a:None, export_projection=lambda *a:None)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch, 'zeros', side_effect=AssertionError('allocation admission reached')):
            with self.assertRaisesRegex(TypeError, 'actual DeviceSoARules'):
                DeviceRulePolicy(rules, rows=2, max_actions=4, device='cuda:0')
            for field, bad in (('require_teacher', False), ('rows', 1), ('max_actions', 3),
                               ('device', torch.device('cuda:1'))):
                original = getattr(formal, field);setattr(formal, field, bad)
                with self.assertRaisesRegex(ValueError, 'Exact teacher-enabled'):
                    DeviceRulePolicy(formal, rows=2, max_actions=4, device='cuda:0')
                setattr(formal, field, original)
            formal.executor.image = SimpleNamespace(entries={}, function_ids=())
            with self.assertRaisesRegex(ValueError, 'frozen teacher_continue'):
                DeviceRulePolicy(formal, rows=2, max_actions=4, device='cuda:0')
            formal.executor.image = SimpleNamespace(entries=good_image.entries,
                function_ids=('foreign:teacher', 'foreign:decision'))
            with self.assertRaisesRegex(ValueError, 'canonical frozen'):
                DeviceRulePolicy(formal, rows=2, max_actions=4, device='cuda:0')
            formal.executor.image = good_image
            with self.assertRaisesRegex(AssertionError, 'allocation admission reached'):
                DeviceRulePolicy(formal, rows=2, max_actions=4, device='cuda:0')

    def test_first_six_exact_numpy_integer_schedule_all_six_state_fields(self):
        policy, rules, roots, loop, seeds = self.exploration_fixture()
        policy.bind_game_loop(loop)
        expected = [np.random.default_rng(seed) for seed in seeds]
        self.assertEqual(policy._exploration_rng.export_numpy_states(), tuple(g.bit_generator.state for g in expected))
        for step, phases, activity in (
                (0, (1, 1), (True, True)), (1, (1, 0), (True, True)),
                (2, (0, 1), (True, True)), (3, (1, 1), (True, False)),
                (4, (1, 1), (True, True)), (5, (1, 1), (True, True)),
                (6, (1, 1), (True, True)), (7, (1, 1), (True, True))):
            loop._steps.fill_(step);rules.phases.copy_(torch.tensor(phases, dtype=torch.int32))
            roots.active.copy_(torch.tensor(activity))
            rules.counts.copy_(torch.tensor([1 if step == 4 else 3, 2]))
            rules.valid.copy_(torch.arange(4)[None, :] < rules.counts[:, None])
            rules.chosen.copy_(torch.tensor([0 if step == 4 else 2, 1]))
            expected_choices = rules.chosen.tolist();mask = []
            for row in (0, 1):
                eligible = activity[row] and step < 6 and phases[row] == 1
                mask.append(eligible)
                if eligible:expected_choices[row] = int(expected[row].integers(int(rules.counts[row])))
                if not activity[row]:expected_choices[row] = -1
            result = self.run_policy(policy, rules, roots)
            self.assertEqual(result.actions.tolist(), expected_choices)
            self.assertEqual(policy.rule_choices_device.tolist(), rules.chosen.tolist())
            self.assertEqual(policy.explored_device.tolist(), mask)
            self.assertEqual(result.errors.tolist(), [0, 0])
            self.assertEqual(policy._exploration_rng.export_numpy_states(), tuple(g.bit_generator.state for g in expected))
            self.assertIs(result.decision.observations, rules.x)
            self.assertEqual(result.requested.sum().item()+result.completed.sum().item()+result.visits.sum().item(), 0)
            rules.versions += 1

    def test_exploration_rejection_error_is_sticky_and_state_transactional(self):
        policy, rules, roots, loop, _ = self.exploration_fixture(retries=1)
        policy.bind_game_loop(loop)
        # A real uint32 cached zero must be rejected for high=3. The actual
        # PCG64 CPU reference kernel rolls back all six fields on retry error7.
        bank = policy._exploration_rng
        bank._has_uint32[0] = 1;bank._uinteger[0] = 0
        before = bank.export_numpy_states()
        result = self.run_policy(policy, rules, roots)
        self.assertEqual(result.errors.tolist(), [7, 0])
        self.assertEqual(policy.exploration_errors_device.tolist(), [7, 0])
        self.assertEqual(result.actions.tolist(), [-1, 1])
        self.assertEqual(policy.rule_choices_device.tolist(), [2, 1])
        self.assertEqual(bank.export_numpy_states(), before)
        self.assertFalse(result.complete[0].item())
        self.assertFalse(policy.explored_device.any().item())

    def test_exploration_seed_counter_owner_binding_and_lifecycle(self):
        for bad in ([1, 2], (1,), (True, 2), (-1, 2), (2**63, 2)):
            policy, rules, roots = self.fixture()
            with self.assertRaises(ValueError):DeviceRulePolicy(rules, rows=2, max_actions=4,
                device='cpu', allow_cpu_reference=True, allow_synthetic=True,
                explore_first_six=True, exploration_seeds=bad)
        for retries in (True, 0, 1025):
            with self.assertRaises(ValueError):DeviceRulePolicy(rules, rows=2, max_actions=4,
                device='cpu', allow_cpu_reference=True, allow_synthetic=True,
                exploration_max_retries=retries)
        with self.assertRaises(ValueError):DeviceRulePolicy(rules, rows=2, max_actions=4,
            device='cpu', allow_cpu_reference=True, allow_synthetic=True, exploration_seeds=(1, 2))
        policy, rules, roots, loop, _ = self.exploration_fixture()
        with self.assertRaises(RuntimeError):self.run_policy(policy, rules, roots)
        for field, bad in (('search', object()), ('rules', object()), ('rows', 1),
                           ('device', torch.device('cuda:0')), ('steps_device', torch.zeros(2, dtype=torch.int32))):
            previous = getattr(loop, field);setattr(loop, field, bad)
            with self.assertRaises((ValueError, TypeError)):policy.bind_game_loop(loop)
            setattr(loop, field, previous)
        loop.steps_device = loop._steps.clone()
        with self.assertRaises(ValueError):policy.bind_game_loop(loop)
        loop.steps_device = loop._steps
        policy.bind_game_loop(loop)
        with self.assertRaises(RuntimeError):policy.bind_game_loop(loop)
        before = policy._exploration_rng.export_numpy_states()
        for field in ('seed', 'negative_step', 'owner'):
            if field == 'seed':loop._job_seeds[0] += 1
            if field == 'negative_step':loop._steps[0] = -1
            if field == 'owner':loop._owners[0] += 1
            result = self.run_policy(policy, rules, roots)
            self.assertFalse(result.complete[0].item())
            self.assertEqual(policy._exploration_rng.export_numpy_states(), before)
            if field == 'seed':loop._job_seeds[0] -= 1
            if field == 'negative_step':loop._steps[0] = 0
            if field == 'owner':loop._owners[0] -= 1
        loop._job_seeds = loop._job_seeds.clone()
        with self.assertRaises(ValueError):self.run_policy(policy, rules, roots)
        policy.close();policy.close()
        self.assertFalse(rules.action_store.closed)
        with self.assertRaises(RuntimeError):self.run_policy(policy, rules, roots)
        default, rules, roots = self.fixture()
        self.assertIsNone(default.exploration_state_device)
        default.close();default.close()
        with self.assertRaises(RuntimeError):self.run_policy(default, rules, roots)


if __name__ == '__main__':unittest.main()
