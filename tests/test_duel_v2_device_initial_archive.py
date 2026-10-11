"""Original state copy CUDA oracle; no actual duel execution or qualification."""
import unittest
from tests import test_duel_v2_device_graph_storage as graphs


class InitialArchiveTest(unittest.TestCase):
    def test_requires_actual_cuda_and_separate_arenas(self):
        from app.modules.card_game.rl.batched_search.device_initial_archive import DeviceInitialArchive
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        with graphs.DeviceGraphStorageTest().store() as source:
            with self.assertRaises(ValueError):
                DeviceInitialArchive(source, source, rows=2, layout=GraphLayout(2, 64, 64, 256))

    def test_actual_cuda_original_survives_physical_release_and_reuse(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required; initial-copy primitive only')
        from app.modules.card_game.rl.batched_search.device_initial_archive import DeviceInitialArchive
        from app.modules.card_game.rl.batched_search.cuda_conditional_graph import CudaConditionalWhile
        from app.modules.card_game.rl.batched_duel.codec import unpack_states
        from app.modules.card_game.rl.batched_duel.device_rule_protocol import DeviceStateBatch
        data = graphs._states() + [{'seed': 123, 'private': ['fixture']}]
        graph = graphs._graph(data, 'cuda')
        case = graphs.DeviceGraphStorageTest()
        with case.store('cuda', slots=8, pages=64) as source, case.store('cuda', slots=8, pages=64) as saved:
            owners = torch.arange(3, dtype=torch.int64, device=source.device)
            roots = source.put_device(graph, owners, torch.ones(3, dtype=torch.uint8, device=source.device))
            selected = DeviceStateBatch(source, roots.slots, roots.generations, roots.owners,
                torch.tensor([True, False, True], device=source.device), roots.errors)
            copy = DeviceInitialArchive(source, saved, rows=3, lanes=2, layout=graph.layout)
            self.addCleanup(copy.close)
            loop = CudaConditionalWhile(device=str(source.device), max_ticks=2, seconds=60)
            self.addCleanup(loop.close)
            pending = torch.ones(1, dtype=torch.uint8, device=source.device)
            def body():copy.capture_device(selected);pending.zero_()
            loop.capture(pending, body, initialize=lambda:pending.fill_(1))
            loop.replay();torch.cuda.synchronize()
            self.assertEqual(copy.output.errors.cpu().tolist(), [0, 0, 0])
            self.assertEqual(int(saved.live.sum()), 2)
            source.release_device(roots)
            graph.nodes.zero_();graph.edges.zero_();graph.payload.zero_()
            scratch = graphs._scratch(graph)
            gathered = saved.gather_into_device(copy.output, scratch)
            self.assertEqual(gathered.errors.cpu().tolist(), [0, 0, 0])
            from app.modules.card_game.rl.batched_duel.schema import GraphLayout, PackedStateBatch
            ids = torch.tensor([0, 2], device=source.device)
            host = [value.index_select(0, ids).cpu().numpy() for value in
                    (scratch.nodes, scratch.edges, scratch.payload, scratch.counts, scratch.roots)]
            restored = unpack_states(PackedStateBatch(GraphLayout(2, 64, 64, 256), *host))
            self.assertEqual(restored[0], data[0])
            self.assertEqual(restored[1], data[2])
            saved.release_device(copy.output)
            self.assertEqual(int(saved.live.sum()), 0)


if __name__ == '__main__':unittest.main()
