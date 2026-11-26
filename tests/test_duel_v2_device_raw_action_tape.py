"""CPU/CUDA tests for the physical raw-action graph archive; not full-rules evidence."""
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity, CharacterEntity, PlayerEntity,
)
from app.modules.card_game.rl.batched_duel.codec import pack_states, unpack_states
from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch
from app.modules.card_game.rl.batched_search.device_raw_action_tape import (
    ACTOR_ERROR, CAPACITY_ERROR, CAVEAT, DUPLICATE_ERROR, DeviceRawActionTape,
    INVALID_ID, OWNER_ERROR, RANGE_ERROR, STALE_ERROR, STEP_ERROR,
)


def _actions():
    shared = ['alias', 7]
    player = PlayerEntity({'hp': 3, 'bag': shared}, entity_id='p1')
    character = CharacterEntity({'id': 'lingke', 'hp': 8}, entity_id='c1')
    card = CardEntity({'card_id': 'k03', 'cost': 1}, entity_id='k1')
    return [
        {'type': 'play', 'card': 'k03', 'cost': 1, 'player': player, 'also': shared,
         'character': character, 'pair': (1, 2), 'flag': True, 'none': None, 'score': 1.5},
        {'type': 'attack', 'target': 0, 'xs': [shared, shared], 'empty': [], 'card': card},
    ]


def _graph(states, device, max_nodes=64, max_edges=64, max_payload=256):
    import torch
    packed = pack_states(states, GraphLayout(len(states), max_nodes, max_edges, max_payload))
    return DeviceGraph(packed.layout, *(
        torch.tensor(np.array(getattr(packed, name), copy=True), device=device)
        for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))


def _scratch(graph):
    return DeviceGraph(graph.layout, graph.nodes.new_zeros(graph.nodes.shape),
                       graph.edges.new_zeros(graph.edges.shape),
                       graph.payload.new_zeros(graph.payload.shape),
                       graph.counts.new_zeros(graph.counts.shape),
                       graph.roots.new_zeros(graph.roots.shape))


def _packed(graph):
    return PackedStateBatch(graph.layout, *(
        np.ascontiguousarray(getattr(graph, name).detach().cpu().numpy())
        for name in ('nodes', 'edges', 'payload', 'counts', 'roots')))


def _vectors(graph, *, games=None, owners=None, steps=None, versions=None, actors=None, active=None):
    import torch
    device = graph.nodes.device
    lanes = graph.rows
    def tensor(values, default, dtype):
        if values is None:
            return default
        return torch.tensor(values, dtype=dtype, device=device)
    game = tensor(games, torch.arange(lanes, dtype=torch.int64, device=device), torch.int64)
    own = tensor(owners, torch.arange(40, 40 + lanes, dtype=torch.int64, device=device), torch.int64)
    step = tensor(steps, torch.zeros(lanes, dtype=torch.int64, device=device), torch.int64)
    version = tensor(versions, torch.arange(lanes, dtype=torch.int64, device=device) + 1, torch.int64)
    actor = tensor(actors, torch.zeros(lanes, dtype=torch.int64, device=device), torch.int64)
    mask = tensor(active, torch.ones(lanes, dtype=torch.bool, device=device), torch.bool)
    return game, own, step, version, actor, mask


class DeviceRawActionTapeTest(unittest.TestCase):
    def tape(self, rows=2, max_records=8, node_capacity=256, edge_capacity=256,
             payload_capacity=1024, device='cpu'):
        return DeviceRawActionTape(
            rows, max_records, node_capacity, edge_capacity, payload_capacity,
            device=device, allow_cpu_reference=device == 'cpu')

    def begin(self, tape, rows=None, owners=None, seeds=None):
        import torch
        rows = tape.rows if rows is None else rows
        own = torch.arange(40, 40 + rows, dtype=torch.int64, device=tape.device)
        if owners is not None:
            own = torch.tensor(owners, dtype=torch.int64, device=tape.device)
        job = torch.arange(rows, dtype=torch.int64, device=tape.device)
        if seeds is not None:
            job = torch.tensor(seeds, dtype=torch.int64, device=tape.device)
        return tape.begin_device(own, job), own, job

    def test_constructor_requires_cuda_and_refuses_readiness_claims(self):
        tape = self.tape()
        with self.assertRaises(ValueError):
            DeviceRawActionTape(2, 4, 16, 16, 64, device='cpu')
        self.assertFalse(tape.supports_full_duel_rules)
        self.assertFalse(hasattr(tape, 'device_validation_receipt'))
        self.assertEqual(CAVEAT.find('ordinals are insufficient'), -1)
        self.assertIn('Ordinals are insufficient', CAVEAT)
        self.assertIn('qualify', CAVEAT)

    def test_cpu_pack_append_gather_preserves_copy_after_source_mutation(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        graph.nodes[0, 0, 3] = 0x13579BDF
        original = (graph.nodes.clone(), graph.edges.clone(), graph.payload.clone(),
                    graph.counts.clone(), graph.roots.clone())
        tape = self.tape()
        self.begin(tape)
        game, owners, steps, versions, actors, active = _vectors(graph)
        errors = tape.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertEqual(errors.tolist(), [0, 0])
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(tape._rec_game[:2].tolist(), [0, 1])
        self.assertEqual(tape._rec_owner[:2].tolist(), [40, 41])
        self.assertEqual(tape._rec_seed[:2].tolist(), [0, 1])
        self.assertEqual(tape._rec_step[:2].tolist(), [0, 0])
        graph.nodes.fill_(7)
        graph.edges.fill_(9)
        graph.payload.fill_(3)
        dest = _scratch(graph)
        gathered = tape.gather_into_device(torch.arange(2, dtype=torch.int64), dest, active)
        self.assertEqual(gathered.tolist(), [0, 0])
        n0, e0, p0 = (int(v) for v in original[3][0].tolist())
        n1, e1, p1 = (int(v) for v in original[3][1].tolist())
        torch.testing.assert_close(dest.nodes[0, :n0], original[0][0, :n0])
        torch.testing.assert_close(dest.edges[0, :e0], original[1][0, :e0])
        torch.testing.assert_close(dest.payload[0, :p0], original[2][0, :p0])
        torch.testing.assert_close(dest.nodes[1, :n1], original[0][1, :n1])
        torch.testing.assert_close(dest.counts, original[3])
        torch.testing.assert_close(dest.roots, original[4])
        self.assertEqual(int(dest.nodes[0, 0, 3]), 0x13579BDF)
        unpacked = unpack_states(_packed(dest))
        self.assertIs(unpacked[0]['player']['bag'], unpacked[0]['also'])
        self.assertIs(unpacked[1]['xs'][0], unpacked[1]['xs'][1])
        self.assertIsInstance(unpacked[0]['player'], PlayerEntity)
        self.assertIsInstance(unpacked[0]['character'], CharacterEntity)
        self.assertIsInstance(unpacked[1]['card'], CardEntity)
        self.assertEqual(unpacked[0]['pair'], (1, 2))
        self.assertEqual(unpacked[0]['type'], 'play')
        self.assertEqual(unpacked[1]['type'], 'attack')

    def test_inactive_mask_does_not_write_and_zero_selected_is_noop(self):
        import torch
        graph = _graph(_actions() + [{'type': 'end'}], 'cpu')
        tape = self.tape(rows=3)
        self.begin(tape, rows=3)
        game, owners, steps, versions, actors, active = _vectors(
            graph, active=[True, False, True])
        before_nodes = tape._nodes.clone()
        errors = tape.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertEqual(errors.tolist(), [0, 0, 0])
        self.assertEqual(int(tape._record_count), 2)
        self.assertEqual(tape._rec_game[:2].tolist(), [0, 2])
        n0 = int(graph.counts[0, 0])
        n2 = int(graph.counts[2, 0])
        off0 = int(tape._rec_node_off[0])
        off1 = int(tape._rec_node_off[1])
        self.assertTrue(torch.equal(tape._nodes[off0:off0 + n0], graph.nodes[0, :n0]))
        self.assertTrue(torch.equal(tape._nodes[off1:off1 + n2], graph.nodes[2, :n2]))
        dest = _scratch(graph)
        dest.nodes.fill_(9)
        gathered = tape.gather_into_device(
            torch.tensor([0, 1, 0], dtype=torch.int64), dest, active)
        self.assertEqual(gathered.tolist(), [0, 0, 0])
        torch.testing.assert_close(dest.nodes[0, :n0], graph.nodes[0, :n0])
        self.assertTrue(bool((dest.nodes[1] == 0).all()))
        self.assertTrue(bool((dest.counts[1] == 0).all()))
        torch.testing.assert_close(dest.nodes[2, :n0], graph.nodes[0, :n0])
        inactive = torch.zeros(3, dtype=torch.bool)
        count = int(tape._record_count)
        used = int(tape._node_used)
        sentinel = tape._nodes[tape.node_capacity].clone()
        again = tape.append_device(graph, game, owners, steps, versions, actors, inactive)
        self.assertEqual(again.tolist(), [0, 0, 0])
        self.assertEqual(int(tape._record_count), count)
        self.assertEqual(int(tape._node_used), used)
        self.assertTrue(torch.equal(tape._nodes[tape.node_capacity], sentinel))
        self.assertFalse(torch.equal(tape._nodes[:used], before_nodes[:used]))

    def test_capacity_is_all_or_none_across_records_and_nodes(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        need_n = int(graph.counts[:, 0].sum())
        need_e = int(graph.counts[:, 1].sum())
        need_p = int(graph.counts[:, 2].sum())
        game, owners, steps, versions, actors, active = _vectors(graph)
        records = self.tape(max_records=1, node_capacity=need_n, edge_capacity=need_e,
                            payload_capacity=need_p)
        self.begin(records)
        packed_before = records._nodes.clone()
        overflow = records.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertTrue(bool((overflow & CAPACITY_ERROR).all()))
        self.assertEqual(int(records._record_count), 0)
        self.assertEqual(int(records._node_used), 0)
        self.assertEqual(records._expected_step.tolist(), [0, 0])
        self.assertTrue(torch.equal(records._nodes[:-1], packed_before[:-1]))
        nodes = self.tape(max_records=2, node_capacity=need_n - 1, edge_capacity=need_e,
                          payload_capacity=need_p)
        self.begin(nodes)
        overflow_n = nodes.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertTrue(bool((overflow_n & CAPACITY_ERROR).all()))
        self.assertEqual(int(nodes._record_count), 0)
        leftover = self.tape(rows=3, max_records=1, node_capacity=need_n, edge_capacity=need_e,
                             payload_capacity=need_p)
        self.begin(leftover, rows=3)
        leftover.append_device(graph, game, owners, steps, versions, actors, active)
        extra = _graph([{'type': 'wait'}], 'cpu')
        g2, o2, s2, v2, a2, m2 = _vectors(extra, games=[2], owners=[42], versions=[3])
        ok = leftover.append_device(extra, g2, o2, s2, v2, a2, m2)
        self.assertEqual(ok.tolist(), [0])
        self.assertEqual(int(leftover._record_count), 1)
        self.assertEqual(int(leftover._rec_game[0]), 2)

    def test_invalid_ids_owner_actor_stale_skip_and_duplicate(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        ids = self.tape()
        self.begin(ids)
        game, owners, steps, versions, actors, active = _vectors(graph, games=[0, 99])
        bad_ids = ids.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertEqual(int(bad_ids[0]), 0)
        self.assertTrue(int(bad_ids[1]) & INVALID_ID)
        self.assertEqual(int(ids._record_count), 1)
        foreign = self.tape()
        self.begin(foreign, owners=[40, 41])
        game, owners, steps, versions, actors, active = _vectors(graph, owners=[40, 99])
        foreign_err = foreign.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertEqual(int(foreign_err[0]), 0)
        self.assertTrue(int(foreign_err[1]) & OWNER_ERROR)
        self.assertEqual(int(foreign._record_count), 1)
        blocked_game, blocked_owners, blocked_steps, blocked_versions, blocked_actors, blocked_active = _vectors(
            graph, games=[1, 0], owners=[41, 40], active=[True, False])
        blocked = foreign.append_device(
            graph, blocked_game, blocked_owners, blocked_steps, blocked_versions,
            blocked_actors, blocked_active)
        self.assertNotEqual(int(blocked[0]), 0)
        self.assertEqual(int(foreign._expected_step[1]), 0)
        actor = self.tape()
        self.begin(actor)
        game, owners, steps, versions, actors, active = _vectors(graph, actors=[0, 2])
        actor_err = actor.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertTrue(int(actor_err[1]) & ACTOR_ERROR)
        skip = self.tape()
        self.begin(skip)
        game, owners, steps, versions, actors, active = _vectors(graph)
        self.assertEqual(
            skip.append_device(graph, game, owners, steps, versions, actors, active).tolist(),
            [0, 0])
        skip_err = skip.append_device(
            graph, game, owners, torch.tensor([2, 1]), versions + 2, actors, active)
        self.assertTrue(int(skip_err[0]) & STEP_ERROR)
        self.assertEqual(int(skip_err[1]), 0)
        self.assertEqual(skip._rec_step[:3].tolist(), [0, 0, 1])
        stale = self.tape()
        self.begin(stale)
        game, owners, steps, versions, actors, active = _vectors(graph, versions=[5, 6])
        stale.append_device(graph, game, owners, steps, versions, actors, active)
        stale_err = stale.append_device(
            graph, game, owners, torch.ones(2, dtype=torch.int64),
            torch.tensor([4, 7]), actors, torch.tensor([True, False]))
        self.assertTrue(int(stale_err[0]) & STALE_ERROR)
        self.assertEqual(int(stale._record_count), 2)
        dup = self.tape()
        self.begin(dup)
        game, owners, steps, versions, actors, active = _vectors(graph, games=[0, 0])
        dup_err = dup.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertTrue(bool(((dup_err & DUPLICATE_ERROR) != 0).all()))
        self.assertEqual(int(dup._record_count), 0)

    def test_inactive_duplicate_lane_cannot_erase_latest_version(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        tape = self.tape()
        self.begin(tape)
        vectors = _vectors(graph, games=[0, 0], owners=[40, 40],
                           versions=[7, 0], active=[True, False])
        self.assertEqual(tape.append_device(graph, *vectors).tolist(), [0, 0])
        self.assertEqual(tape._last_version.tolist(), [7, -1])
        vectors = _vectors(graph, games=[0, 0], owners=[40, 40], steps=[1, 0],
                           versions=[7, 0], active=[True, False])
        errors = tape.append_device(graph, *vectors)
        self.assertTrue(int(errors[0]) & STALE_ERROR)
        self.assertEqual(int(tape._record_count), 1)

    def test_malformed_used_range_is_rejected_before_copy(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        tape = self.tape()
        self.begin(tape)
        game, owners, steps, versions, actors, active = _vectors(graph)
        graph.roots[0] = 999
        errors = tape.append_device(graph, game, owners, steps, versions, actors, active)
        self.assertTrue(int(errors[0]) & RANGE_ERROR)
        self.assertEqual(int(errors[1]), 0)
        self.assertEqual(int(tape._record_count), 1)
        self.assertEqual(int(tape._rec_game[0]), 1)
        n1 = int(graph.counts[1, 0])
        off = int(tape._rec_node_off[0])
        self.assertTrue(torch.equal(tape._nodes[off:off + n1], graph.nodes[1, :n1]))
        graph.counts[1, 0] = 0
        graph.roots[0] = 0
        used = int(tape._node_used)
        again = tape.append_device(
            graph, game, owners, torch.ones(2, dtype=torch.int64), versions + 1,
            actors, torch.tensor([False, True]))
        self.assertTrue(int(again[1]) & RANGE_ERROR)
        self.assertEqual(int(tape._record_count), 1)
        self.assertEqual(int(tape._node_used), used)

    def test_gather_invalid_record_and_destination_capacity_fail_closed(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        tape = self.tape()
        self.begin(tape)
        game, owners, steps, versions, actors, active = _vectors(graph)
        tape.append_device(graph, game, owners, steps, versions, actors, active)
        dest = _scratch(graph)
        dest.nodes.fill_(4)
        bad = tape.gather_into_device(
            torch.tensor([-1, 7], dtype=torch.int64), dest, active)
        self.assertTrue(bool(((bad & INVALID_ID) != 0).all()))
        self.assertTrue(bool((dest.nodes == 0).all()))
        self.assertTrue(bool((dest.counts == 0).all()))
        n0 = int(graph.counts[0, 0])
        self.assertGreater(n0, 1)
        tiny_layout = GraphLayout(2, max(1, n0 - 1), graph.layout.max_edges,
                                  graph.layout.max_payload_bytes)
        tiny = DeviceGraph(
            tiny_layout,
            graph.nodes.new_zeros((2, tiny_layout.max_nodes, 4)),
            graph.edges.new_zeros((2, tiny_layout.max_edges, 2)),
            graph.payload.new_zeros((2, tiny_layout.max_payload_bytes)),
            graph.counts.new_zeros((2, 3)),
            graph.roots.new_zeros((2,)))
        overflow = tape.gather_into_device(torch.arange(2, dtype=torch.int64), tiny, active)
        self.assertTrue(int(overflow[0]) & CAPACITY_ERROR)
        self.assertTrue(bool((tiny.counts[0] == 0).all()))
        self.assertTrue(bool((tiny.nodes[0] == 0).all()))

    def test_hot_path_has_no_host_tensor_reads(self):
        import torch
        graph = _graph(_actions(), 'cpu')
        tape = self.tape()
        game, owners, steps, versions, actors, active = _vectors(graph)
        dest = _scratch(graph)
        records = torch.arange(2, dtype=torch.int64)
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU copy')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy copy')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar copy')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('Host list')), \
             patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('Dynamic index')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Host branch')):
            self.begin(tape)
            tape.append_device(graph, game, owners, steps, versions, actors, active)
            tape.gather_into_device(records, dest, active)

    def test_actual_cuda_capture_replays_synthetic_raw_action_dicts(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; synthetic raw action archive only')
        graph = _graph(_actions(), 'cuda')
        tape = self.tape(device='cuda')
        game, owners, steps, versions, actors, active = _vectors(graph)
        dest = _scratch(graph)
        records = torch.arange(2, dtype=torch.int64, device=graph.nodes.device)
        begin_owners = torch.arange(40, 42, dtype=torch.int64, device=graph.nodes.device)
        seeds = torch.arange(2, dtype=torch.int64, device=graph.nodes.device)
        tape.begin_device(begin_owners, seeds)
        tape.append_device(graph, game, owners, steps, versions, actors, active)
        tape.gather_into_device(records, dest, active)
        torch.cuda.synchronize()
        cap = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cap):
            tape.begin_device(begin_owners, seeds)
            tape.append_device(graph, game, owners, steps, versions, actors, active)
            tape.gather_into_device(records, dest, active)
        graph.nodes[0, 0, 3] = 0x24
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Host loop predicate')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host root stepping')), \
             patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('Dynamic index')):
            cap.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(tape._record_count), 2)
        n0 = int(graph.counts[0, 0])
        self.assertEqual(int(dest.nodes[0, 0, 3]), 0x24)
        torch.testing.assert_close(dest.nodes[0, :n0], graph.nodes[0, :n0])
        torch.testing.assert_close(dest.counts, graph.counts)
        # A real lane plus inactive padding for the same game must preserve
        # its version marker; the next stale action is refused on device.
        game.zero_();owners.fill_(40);active[1] = False
        versions.copy_(torch.tensor([7, 0], dtype=torch.int64, device=tape.device))
        next_steps = torch.tensor([1, 0], dtype=torch.int64, device=tape.device)
        next_versions = torch.tensor([6, 0], dtype=torch.int64, device=tape.device)
        outcomes = []
        padded = torch.cuda.CUDAGraph()
        with torch.cuda.graph(padded):
            tape.begin_device(begin_owners, seeds)
            tape.append_device(graph, game, owners, steps, versions, actors, active)
            outcomes.append(tape.append_device(graph, game, owners, next_steps,
                                               next_versions, actors, active))
        padded.replay();torch.cuda.synchronize()
        self.assertEqual(tape._last_version.cpu().tolist(), [7, -1])
        self.assertTrue(int(outcomes[0][0]) & STALE_ERROR)
        self.assertEqual(int(tape._record_count), 1)
        tape.close()


if __name__ == '__main__':
    unittest.main()

