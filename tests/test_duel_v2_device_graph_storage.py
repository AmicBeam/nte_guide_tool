"""Compact device graph store tests. Numerical fixtures do not execute game rules."""
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity, CharacterEntity, PlayerEntity,
)
from app.modules.card_game.rl.batched_duel.codec import pack_states, unpack_states
from app.modules.card_game.rl.batched_duel.cuda_vm import DeviceGraph
from app.modules.card_game.rl.batched_duel.device_graph_pool import (
    CORRUPT_PAGES,DUPLICATE_SLOT, INVALID_HANDLE, INVALID_SIZE, PAGE_CAPACITY,
)
from app.modules.card_game.rl.batched_duel.device_graph_storage import (
    SLOT_CAPACITY, CompactDeviceGraphStore,
)
from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch


def _states():
    shared = ['alias', 7]
    player = PlayerEntity({'hp': 3, 'bag': shared}, entity_id='p1')
    character = CharacterEntity({'id': 'lingke', 'hp': 8}, entity_id='c1')
    card = CardEntity({'card_id': 'k03', 'cost': 1}, entity_id='k1')
    return [
        {'player': player, 'also': shared, 'character': character, 'card': card,
         'pair': (1, 2), 'flag': True, 'none': None, 'score': 1.5},
        {'n': 1, 'xs': [shared, shared], 'empty': []},
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


class DeviceGraphStorageTest(unittest.TestCase):
    def test_shared_reads_copy_independent_rows_without_relaxing_write_leases(self):
        import torch
        from dataclasses import replace
        graph=_graph([{'shared':[1,2]}],'cpu')
        with self.store() as store:
            owners,active=self.vectors(graph)
            saved=store.put_device(graph,owners,active)
            repeated=DeviceStateBatch(store,saved.slots.repeat(2),saved.generations.repeat(2),
                saved.owners.repeat(2),saved.active.repeat(2),saved.errors.repeat(2))
            target=_graph([{},{}],'cpu')
            self.assertEqual(store.gather_into_device(repeated,target).errors.tolist(),[DUPLICATE_SLOT]*2)
            self.assertEqual(store.gather_into_device(repeated,target,shared_reads=True).errors.tolist(),[0,0])
            self.assertEqual(unpack_states(_packed(target)),[{'shared':[1,2]}]*2)
            bad=replace(repeated,owners=torch.tensor([int(saved.owners[0]),99]))
            self.assertEqual(store.gather_into_device(bad,target,shared_reads=True).errors.tolist(),[0,INVALID_HANDLE])
            store.release_device(repeated)
            self.assertEqual(repeated.errors.tolist(),[DUPLICATE_SLOT]*2)
            self.assertEqual(int(store.live.sum()),1)
            store.release_device(saved)
            self.assertEqual(int(store.live.sum()),0)

    def test_invalid_root_is_not_admitted(self):
        graph=_graph([{'a':1},{'a':2}],'cpu');graph.roots[0]=999
        with self.store() as store:
            owners,active=self.vectors(graph)
            states=store.put_device(graph,owners,active)
            self.assertEqual(states.errors.tolist(),[INVALID_SIZE,0])
            self.assertEqual(int(store.live.sum()),1)
            store.release_device(states)
            self.assertEqual(int(store.live.sum()),0)

    def test_corrupt_component_fails_gather_and_atomic_release_before_any_arena_changes(self):
        import torch
        from dataclasses import replace
        graph=_graph(_states(),'cpu')
        with self.store() as store:
            owners,active=self.vectors(graph)
            states=store.put_device(graph,owners,active)
            slot=int(states.slots[0]);old=int(store.payload.page_table[slot,0])
            store.payload.page_table[slot,0]=-1
            result=store.gather_into_device(states,_scratch(graph))
            self.assertEqual(result.errors[0].item(),CORRUPT_PAGES)
            self.assertEqual(result.graph.counts[0].tolist(),[0,0,0])
            first=replace(states,active=torch.tensor([True,False]))
            counts=store.counts.clone();gens=store.generation.clone()
            component_before=[(pool.live.clone(),pool.page_count.clone(),pool.generation.clone())
                              for pool in (store.nodes,store.edges,store.payload)]
            store.release_device(first)
            self.assertEqual(first.errors[0].item(),CORRUPT_PAGES)
            self.assertTrue(torch.equal(store.counts,counts))
            self.assertTrue(torch.equal(store.generation,gens))
            for pool,before in zip((store.nodes,store.edges,store.payload),component_before):
                for name,old_values in zip(('live','page_count','generation'),before):
                    self.assertTrue(torch.equal(getattr(pool,name),old_values))
            store.payload.page_table[slot,0]=old
            # Sticky search/corruption errors do not prevent releasing repaired valid leases.
            store.release_device(states)
            self.assertEqual(int(store.live.sum()),0)

    def test_component_generation_and_foreign_pool_are_checked(self):
        from dataclasses import replace
        graph=_graph(_states(),'cpu')
        with self.store() as store,self.store() as foreign:
            owners,active=self.vectors(graph)
            states=store.put_device(graph,owners,active)
            invalid=replace(states,pool=foreign)
            with self.assertRaises(ValueError):store.gather_into_device(invalid,_scratch(graph))
            with self.assertRaises(ValueError):store.release_device(invalid)
            slot=int(states.slots[0]);store.nodes.generation[slot]+=1
            result=store.gather_into_device(states,_scratch(graph))
            self.assertEqual(result.errors[0].item(),CORRUPT_PAGES)
            self.assertEqual(result.graph.counts[0].tolist(),[0,0,0])
            store.nodes.generation[slot]-=1
            states.errors[0]=107
            result=store.gather_into_device(states,_scratch(graph))
            self.assertEqual(result.errors[0].item(),107)
            self.assertEqual(result.graph.counts[0].tolist(),[0,0,0])
            store.release_device(states)
            self.assertEqual(int(store.live.sum()),0)

    def store(self, device="cpu", slots=4, pages=16, page_bytes=128, per_slot=8):
        return CompactDeviceGraphStore(
            slots=slots, node_pages=pages, edge_pages=pages, payload_pages=pages,
            node_page_bytes=page_bytes, edge_page_bytes=page_bytes,
            payload_page_bytes=page_bytes, max_node_pages_per_slot=per_slot,
            max_edge_pages_per_slot=per_slot, max_payload_pages_per_slot=per_slot,
            device=device, allow_cpu_reference=device == "cpu")

    def vectors(self, graph, owners=None, active=None):
        import torch
        b = graph.rows
        own = torch.arange(40, 40 + b, dtype=torch.int64, device=graph.nodes.device)
        if owners is not None:
            own = torch.tensor(owners, dtype=torch.int64, device=graph.nodes.device)
        mask = torch.ones(b, dtype=torch.bool, device=graph.nodes.device)
        if active is not None:
            mask = torch.tensor(active, dtype=torch.bool, device=graph.nodes.device)
        return own, mask

    def test_pack_store_gather_codec_aliases_and_unknown_fields(self):
        import torch
        graph = _graph(_states(), "cpu")
        graph.nodes[0, 0, 3] = 0x13579BDF
        original = graph.nodes.clone(), graph.edges.clone(), graph.payload.clone()
        store = self.store()
        owners, active = self.vectors(graph)
        states = store.put_device(graph, owners, active)
        self.assertEqual(states.errors.tolist(), [0, 0])
        dest = _scratch(graph)
        gathered = store.gather_into_device(states, dest)
        self.assertEqual(gathered.errors.tolist(), [0, 0])
        torch.testing.assert_close(graph.nodes, original[0])
        torch.testing.assert_close(graph.edges, original[1])
        torch.testing.assert_close(graph.payload, original[2])
        n0, e0, p0 = (int(v) for v in graph.counts[0].tolist())
        n1, e1, p1 = (int(v) for v in graph.counts[1].tolist())
        torch.testing.assert_close(dest.nodes[0, :n0], graph.nodes[0, :n0])
        torch.testing.assert_close(dest.edges[0, :e0], graph.edges[0, :e0])
        torch.testing.assert_close(dest.payload[0, :p0], graph.payload[0, :p0])
        torch.testing.assert_close(dest.nodes[1, :n1], graph.nodes[1, :n1])
        self.assertEqual(int(dest.nodes[0, 0, 3]), 0x13579BDF)
        unpacked = unpack_states(_packed(dest))
        self.assertIs(unpacked[0]["player"]["bag"], unpacked[0]["also"])
        self.assertIs(unpacked[1]["xs"][0], unpacked[1]["xs"][1])
        self.assertIsInstance(unpacked[0]["player"], PlayerEntity)
        self.assertIsInstance(unpacked[0]["character"], CharacterEntity)
        self.assertIsInstance(unpacked[0]["card"], CardEntity)
        self.assertEqual(unpacked[0]["pair"], (1, 2))
        store.release_device(states)
        self.assertEqual(int(store.live.sum()), 0)

    def test_inactive_mask_does_not_expose_another_slot(self):
        import torch
        graph = _graph(_states(), "cpu")
        store = self.store()
        owners, active = self.vectors(graph, active=[True, False])
        states = store.put_device(graph, owners, active)
        self.assertEqual(states.errors.tolist(), [0, 0])
        self.assertEqual(int(states.slots[1]), -1)
        self.assertEqual(int(store.live.sum()), 1)
        dest = _scratch(graph)
        dest.nodes.fill_(9)
        gathered = store.gather_into_device(states, dest)
        self.assertEqual(gathered.errors.tolist(), [0, 0])
        n0 = int(graph.counts[0, 0])
        torch.testing.assert_close(dest.nodes[0, :n0], graph.nodes[0, :n0])
        self.assertTrue(bool((dest.nodes[1] == 0).all()))
        self.assertTrue(bool((dest.counts[1] == 0).all()))
        store.release_device(states)

    def test_stale_owner_and_duplicate_slot_do_not_mutate(self):
        import torch
        graph = _graph(_states()[:1], "cpu")
        store = self.store()
        owners, active = self.vectors(graph)
        states = store.put_device(graph, owners, active)
        live = store.live.clone()
        gen = store.generation.clone()
        pages = store.nodes.page_owner.clone()
        dest = _scratch(graph)
        stale = DeviceStateBatch(
            store, states.slots, states.generations, states.owners + 1,
            states.active, states.errors.clone())
        self.assertEqual(store.gather_into_device(stale, dest).errors.tolist(),
                         [INVALID_HANDLE])
        self.assertTrue(bool((dest.nodes == 0).all()))
        store.release_device(stale)
        torch.testing.assert_close(store.live, live)
        torch.testing.assert_close(store.generation, gen)
        torch.testing.assert_close(store.nodes.page_owner, pages)
        dup = DeviceStateBatch(
            store, torch.stack((states.slots[0], states.slots[0])),
            torch.stack((states.generations[0], states.generations[0])),
            torch.stack((states.owners[0], states.owners[0])),
            torch.tensor([True, True]), torch.zeros(2, dtype=torch.int32))
        dest2 = DeviceGraph(graph.layout, dest.nodes.new_zeros((2,) + graph.nodes.shape[1:]),
                            dest.edges.new_zeros((2,) + graph.edges.shape[1:]),
                            dest.payload.new_zeros((2,) + graph.payload.shape[1:]),
                            dest.counts.new_zeros((2, 3)), dest.roots.new_zeros((2,)))
        self.assertEqual(store.gather_into_device(dup, dest2).errors.tolist(),
                         [DUPLICATE_SLOT, DUPLICATE_SLOT])
        store.release_device(dup)
        torch.testing.assert_close(store.live, live)
        torch.testing.assert_close(store.generation, gen)
        store.release_device(states)
        self.assertEqual(int(store.generation[int(states.slots[0])]), 1)
        self.assertEqual(int(store.live.sum()), 0)
        store.release_device(states)
        self.assertEqual(states.errors.tolist(), [INVALID_HANDLE])
        self.assertEqual(int(store.generation[int(states.slots[0])]), 1)

    def test_overflow_rolls_back_without_touching_live_slots(self):
        import torch
        graph = _graph(_states()[:1], "cpu")
        store = self.store(slots=3, pages=4, page_bytes=256, per_slot=4)
        first = store.put_device(graph, *self.vectors(graph, owners=[11]))
        self.assertEqual(first.errors.tolist(), [0])
        before = [v.clone() for v in (store.live, store.nodes.page_owner,
                                      store.edges.page_owner, store.payload.page_owner,
                                      store.generation, store.nodes.generation)]
        pair = _graph(_states(), "cpu")
        overflow = store.put_device(pair, *self.vectors(pair, owners=[12, 13]))
        self.assertTrue(all(code in (PAGE_CAPACITY, SLOT_CAPACITY)
                            for code in overflow.errors.tolist()))
        self.assertEqual(overflow.errors.tolist()[0], overflow.errors.tolist()[1])
        for actual, expected in zip(
                (store.live, store.nodes.page_owner, store.edges.page_owner,
                 store.payload.page_owner, store.generation, store.nodes.generation), before):
            torch.testing.assert_close(actual, expected)
        dest = _scratch(graph)
        gathered = store.gather_into_device(first, dest)
        self.assertEqual(gathered.errors.tolist(), [0])
        n0 = int(graph.counts[0, 0])
        torch.testing.assert_close(dest.nodes[0, :n0], graph.nodes[0, :n0])
        store.release_device(first)

    def test_reuse_generation_and_failed_search_still_releases(self):
        import torch
        graph = _graph(_states()[:1], "cpu")
        store = self.store()
        first = store.put_device(graph, *self.vectors(graph, owners=[21]))
        slot = int(first.slots[0])
        store.release_device(first)
        self.assertEqual(int(store.generation[slot]), 1)
        again = store.put_device(graph, *self.vectors(graph, owners=[22]))
        self.assertEqual(int(again.slots[0]), slot)
        self.assertEqual(int(again.generations[0]), 1)
        dest = _scratch(graph)
        self.assertEqual(store.gather_into_device(first, dest).errors.tolist(),
                         [INVALID_HANDLE])
        self.assertTrue(bool((dest.counts == 0).all()))
        self.assertEqual(store.gather_into_device(again, dest).errors.tolist(), [0])
        again.errors.fill_(99)
        store.release_device(again)
        self.assertEqual(int(again.errors[0]), 99)
        self.assertEqual(int(store.live.sum()), 0)
        self.assertEqual(int(store.generation[slot]), 2)

    def test_cpu_oracle_required_and_hot_path_has_no_scalar_host_reads(self):
        import torch
        with self.assertRaises(ValueError):
            CompactDeviceGraphStore(
                slots=2, node_pages=2, edge_pages=2, payload_pages=2,
                node_page_bytes=16, edge_page_bytes=16, payload_page_bytes=16,
                max_node_pages_per_slot=1, max_edge_pages_per_slot=1,
                max_payload_pages_per_slot=1, device="cpu")
        graph = _graph(_states(), "cpu")
        store = self.store()
        owners, active = self.vectors(graph)
        dest = _scratch(graph)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU copy")), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy copy")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("Scalar copy")), \
             patch.object(torch.Tensor, "tolist", side_effect=AssertionError("List copy")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("Host branch")):
            states = store.put_device(graph, owners, active)
            store.gather_into_device(states, dest)
            store.metrics_device()
            store.release_device(states)
            states = store.put_device(graph, owners, active)
            store.release_device(states)

    def test_invalid_size_and_zero_edge_graphs(self):
        graph = _graph([1, {"a": 1}], "cpu")
        store = self.store()
        owners, active = self.vectors(graph)
        ok = store.put_device(graph, owners, active)
        self.assertEqual(ok.errors.tolist(), [0, 0])
        self.assertEqual(int(graph.counts[0, 1]), 0)
        store.release_device(ok)
        graph.counts[0, 0] = 0
        bad = store.put_device(graph, owners, active)
        self.assertEqual(bad.errors.tolist(), [INVALID_SIZE, 0])
        self.assertEqual(int(store.live.sum()), 1)
        store.release_device(bad)
        self.assertEqual(int(store.live.sum()), 0)

    def test_actual_cuda_store_gather_release_and_graph_replay(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required; fixture does not execute game rules')
        graph = _graph(_states(), "cuda")
        store = self.store(device="cuda")
        owners, active = self.vectors(graph)
        dest = _scratch(graph)
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                states = store.put_device(graph, owners, active)
                store.gather_into_device(states, dest)
                store.release_device(states)
        torch.cuda.current_stream().wait_stream(warm)
        graph_cap = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_cap):
            states = store.put_device(graph, owners, active)
            gathered = store.gather_into_device(states, dest)
            store.release_device(states)
        for _ in range(3):
            graph_cap.replay()
        torch.cuda.synchronize()
        self.assertFalse(bool((states.errors != 0).any()))
        self.assertFalse(bool((gathered.errors != 0).any()))
        self.assertEqual(int(store.live.sum()), 0)
        store.close()


if __name__ == '__main__':
    unittest.main()

