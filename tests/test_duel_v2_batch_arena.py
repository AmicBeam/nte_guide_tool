"""Unit tests for batched duel CPU state arena.

Validates layout parameter validation, numpy int32 dtype and shape safety,
aliasing isolation, workspace fork/reset independence, generational ABA protection,
cross-arena isolation, capacity exhaustion, active workspace release protection,
and 1600-slot memory accounting.
"""

from dataclasses import FrozenInstanceError, replace
import unittest
import numpy as np

from app.modules.card_game.rl.batched_duel import (
    ArenaBusyError,
    ArenaCapacityError,
    ArenaError,
    ArenaInvalidRefError,
    ArenaLayout,
    ArenaStateDtypeError,
    ArenaStateShapeError,
    CpuStateArena,
    GameRef,
    WorkspaceRef,
)


class TestArenaLayout(unittest.TestCase):
    """Test validation of ArenaLayout parameter types and bounds."""

    def test_valid_layout(self) -> None:
        layout = ArenaLayout(game_capacity=16, workspace_capacity=32, state_width=128)
        self.assertEqual(layout.game_capacity, 16)
        self.assertEqual(layout.workspace_capacity, 32)
        self.assertEqual(layout.state_width, 128)

    def test_layout_rejects_bool(self) -> None:
        for kwargs in [
            {"game_capacity": True, "workspace_capacity": 32, "state_width": 128},
            {"game_capacity": 16, "workspace_capacity": False, "state_width": 128},
            {"game_capacity": 16, "workspace_capacity": 32, "state_width": True},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    ArenaLayout(**kwargs)

    def test_layout_rejects_non_int(self) -> None:
        for kwargs in [
            {"game_capacity": 16.5, "workspace_capacity": 32, "state_width": 128},
            {"game_capacity": "16", "workspace_capacity": 32, "state_width": 128},
            {"game_capacity": 16, "workspace_capacity": None, "state_width": 128},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    ArenaLayout(**kwargs)

    def test_layout_rejects_non_positive(self) -> None:
        for kwargs in [
            {"game_capacity": 0, "workspace_capacity": 32, "state_width": 128},
            {"game_capacity": -5, "workspace_capacity": 32, "state_width": 128},
            {"game_capacity": 16, "workspace_capacity": 0, "state_width": 128},
            {"game_capacity": 16, "workspace_capacity": 32, "state_width": -1},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ArenaLayout(**kwargs)

    def test_layout_is_frozen(self) -> None:
        layout = ArenaLayout(game_capacity=8, workspace_capacity=16, state_width=64)
        with self.assertRaises(FrozenInstanceError):
            layout.game_capacity = 32  # type: ignore


class TestStateValidationAndTypeSafety(unittest.TestCase):
    """Test strict int32 and 1D shape checks, preventing truncation or bad shapes."""

    def setUp(self) -> None:
        self.layout = ArenaLayout(game_capacity=4, workspace_capacity=4, state_width=8)
        self.arena = CpuStateArena(self.layout)

    def test_reject_non_ndarray(self) -> None:
        for invalid in [[0] * 8, (0,) * 8, 123, "not_an_array"]:
            with self.subTest(invalid=type(invalid)):
                with self.assertRaises(TypeError):
                    self.arena.allocate_game(invalid)  # type: ignore

    def test_reject_non_int32_dtype(self) -> None:
        # Rejects float64, float32, int64 without silent truncation
        dtypes = [np.float32, np.float64, np.int64, np.int16, np.uint32]
        for dt in dtypes:
            arr = np.ones(8, dtype=dt)
            with self.subTest(dtype=dt):
                with self.assertRaises((ArenaStateDtypeError, TypeError)):
                    self.arena.allocate_game(arr)

    def test_reject_incorrect_ndim(self) -> None:
        # 2D arrays rejected even if total element count matches
        arr_2d = np.zeros((1, 8), dtype=np.int32)
        with self.assertRaises((ArenaStateShapeError, ValueError)):
            self.arena.allocate_game(arr_2d)

        arr_col = np.zeros((8, 1), dtype=np.int32)
        with self.assertRaises((ArenaStateShapeError, ValueError)):
            self.arena.allocate_game(arr_col)

    def test_reject_incorrect_width(self) -> None:
        arr_short = np.zeros(7, dtype=np.int32)
        with self.assertRaises((ArenaStateShapeError, ValueError)):
            self.arena.allocate_game(arr_short)

        arr_long = np.zeros(9, dtype=np.int32)
        with self.assertRaises((ArenaStateShapeError, ValueError)):
            self.arena.allocate_game(arr_long)


class TestAliasingAndDataIsolation(unittest.TestCase):
    """Test defensive copies prevent input/output aliasing and mutable view leakage."""

    def setUp(self) -> None:
        self.layout = ArenaLayout(game_capacity=4, workspace_capacity=4, state_width=8)
        self.arena = CpuStateArena(self.layout)

    def test_allocate_game_defensive_copy(self) -> None:
        external_state = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        ref = self.arena.allocate_game(external_state)

        # Mutate caller array
        external_state[0] = 999

        # Arena array must remain unaffected
        arena_read = self.arena.read_game(ref)
        self.assertEqual(arena_read[0], 1)

    def test_read_game_returns_independent_copy(self) -> None:
        init_state = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        ref = self.arena.allocate_game(init_state)

        read1 = self.arena.read_game(ref)
        # Mutate caller copy
        read1[0] = 777

        read2 = self.arena.read_game(ref)
        self.assertEqual(read2[0], 10)

    def test_write_game_defensive_copy(self) -> None:
        init_state = np.zeros(8, dtype=np.int32)
        ref = self.arena.allocate_game(init_state)

        new_state = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int32)
        self.arena.write_game(ref, new_state)

        new_state[0] = -1
        self.assertEqual(self.arena.read_game(ref)[0], 5)

    def test_workspace_read_write_defensive_copy(self) -> None:
        game_ref = self.arena.allocate_game(np.ones(8, dtype=np.int32))
        ws_ref = self.arena.fork_workspace(game_ref)

        ws_read1 = self.arena.read_workspace(ws_ref)
        ws_read1[0] = 42

        ws_read2 = self.arena.read_workspace(ws_ref)
        self.assertEqual(ws_read2[0], 1)

        write_buf = np.array([9, 8, 7, 6, 5, 4, 3, 2], dtype=np.int32)
        self.arena.write_workspace(ws_ref, write_buf)
        write_buf[0] = 0
        self.assertEqual(self.arena.read_workspace(ws_ref)[0], 9)

    def test_slot_isolation(self) -> None:
        state_a = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        state_b = np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)

        ref_a = self.arena.allocate_game(state_a)
        ref_b = self.arena.allocate_game(state_b)

        self.assertNotEqual(ref_a.slot, ref_b.slot)
        np.testing.assert_array_equal(self.arena.read_game(ref_a), state_a)
        np.testing.assert_array_equal(self.arena.read_game(ref_b), state_b)


class TestWorkspaceForkAndIndependence(unittest.TestCase):
    """Test workspace fork uses numpy copying, operates independently, and supports reset."""

    def setUp(self) -> None:
        self.layout = ArenaLayout(game_capacity=4, workspace_capacity=4, state_width=8)
        self.arena = CpuStateArena(self.layout)

    def test_fork_copies_values_and_maintains_independence(self) -> None:
        init_state = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        game_ref = self.arena.allocate_game(init_state)

        ws_ref = self.arena.fork_workspace(game_ref)
        np.testing.assert_array_equal(self.arena.read_workspace(ws_ref), init_state)

        # Mutate workspace -> game remains unchanged
        ws_mod = np.array([99, 99, 99, 99, 99, 99, 99, 99], dtype=np.int32)
        self.arena.write_workspace(ws_ref, ws_mod)
        np.testing.assert_array_equal(self.arena.read_game(game_ref), init_state)

        # Mutate game -> workspace remains unchanged
        game_mod = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        self.arena.write_game(game_ref, game_mod)
        np.testing.assert_array_equal(self.arena.read_workspace(ws_ref), ws_mod)

    def test_reset_workspace_from_game(self) -> None:
        game_ref = self.arena.allocate_game(np.zeros(8, dtype=np.int32))
        ws_ref = self.arena.fork_workspace(game_ref)

        # Update game with new progress
        new_game_state = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        self.arena.write_game(game_ref, new_game_state)

        # Reset workspace to match updated game
        self.arena.reset_workspace_from_game(ws_ref)
        np.testing.assert_array_equal(
            self.arena.read_workspace(ws_ref), new_game_state
        )


class TestGenerationalAbaAndSlotReuse(unittest.TestCase):
    """Test generation advancing protects against ABA stale reference reuse."""

    def setUp(self) -> None:
        self.layout = ArenaLayout(game_capacity=2, workspace_capacity=2, state_width=8)
        self.arena = CpuStateArena(self.layout)

    def test_game_generation_advances_on_free(self) -> None:
        state_v1 = np.ones(8, dtype=np.int32)
        old_ref = self.arena.allocate_game(state_v1)
        slot = old_ref.slot
        gen_v1 = old_ref.generation

        self.arena.free_game(old_ref)

        # Re-allocate into the same slot
        state_v2 = np.full(8, 2, dtype=np.int32)
        new_ref = self.arena.allocate_game(state_v2)
        self.assertEqual(new_ref.slot, slot)
        self.assertGreater(new_ref.generation, gen_v1)

        # Old ref must be completely rejected on all operations
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.read_game(old_ref)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.write_game(old_ref, state_v1)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.free_game(old_ref)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.fork_workspace(old_ref)

    def test_workspace_generation_advances_on_free(self) -> None:
        game_ref = self.arena.allocate_game(np.ones(8, dtype=np.int32))
        old_ws_ref = self.arena.fork_workspace(game_ref)
        ws_slot = old_ws_ref.slot
        ws_gen_v1 = old_ws_ref.generation

        self.arena.free_workspace(old_ws_ref)

        # Re-allocate workspace slot
        new_ws_ref = self.arena.fork_workspace(game_ref)
        self.assertEqual(new_ws_ref.slot, ws_slot)
        self.assertGreater(new_ws_ref.generation, ws_gen_v1)

        # Old workspace ref rejected
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.read_workspace(old_ws_ref)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.write_workspace(old_ws_ref, np.zeros(8, dtype=np.int32))
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.free_workspace(old_ws_ref)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.reset_workspace_from_game(old_ws_ref)


class TestCrossArenaRejection(unittest.TestCase):
    """Test operations reject references originating from a different arena."""

    def setUp(self) -> None:
        self.layout = ArenaLayout(game_capacity=2, workspace_capacity=2, state_width=8)
        self.arena_a = CpuStateArena(self.layout)
        self.arena_b = CpuStateArena(self.layout)

    def test_cross_arena_game_ref_rejected(self) -> None:
        ref_a = self.arena_a.allocate_game(np.ones(8, dtype=np.int32))

        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.read_game(ref_a)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.write_game(ref_a, np.ones(8, dtype=np.int32))
        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.free_game(ref_a)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.fork_workspace(ref_a)

    def test_cross_arena_workspace_ref_rejected(self) -> None:
        game_a = self.arena_a.allocate_game(np.ones(8, dtype=np.int32))
        ws_a = self.arena_a.fork_workspace(game_a)

        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.read_workspace(ws_a)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.write_workspace(ws_a, np.ones(8, dtype=np.int32))
        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.free_workspace(ws_a)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena_b.reset_workspace_from_game(ws_a)

    def test_malformed_and_changed_owner_refs_rejected(self) -> None:
        game = self.arena_a.allocate_game(np.zeros(8, dtype=np.int32))
        workspace = self.arena_a.fork_workspace(game)
        for ref in (replace(game, slot=False), replace(game, generation=1.),
                    replace(game, arena_id=True)):
            with self.subTest(ref=ref), self.assertRaises(ArenaInvalidRefError):
                self.arena_a.read_game(ref)
        for ref in (replace(workspace, slot=False),
                    replace(workspace, generation=1.),
                    replace(workspace, owner_game_slot=True),
                    replace(workspace, owner_game_generation=2)):
            with self.subTest(ref=ref), self.assertRaises(ArenaInvalidRefError):
                self.arena_a.read_workspace(ref)

    def test_arena_identity_is_not_a_worker_local_counter(self) -> None:
        # A tiny ordinal is unsafe when another worker creates its first arena.
        self.assertGreater(self.arena_a.arena_id.bit_length(), 64)
        self.assertNotEqual(self.arena_a.arena_id, self.arena_b.arena_id)


class TestCapacityExhaustionAndDeterministicAllocation(unittest.TestCase):
    """Test capacity limits, deterministic slot reuse, and clean error raising."""

    def test_game_capacity_limit_and_reuse(self) -> None:
        layout = ArenaLayout(game_capacity=2, workspace_capacity=2, state_width=8)
        arena = CpuStateArena(layout)

        ref0 = arena.allocate_game(np.zeros(8, dtype=np.int32))
        ref1 = arena.allocate_game(np.zeros(8, dtype=np.int32))
        self.assertEqual(ref0.slot, 0)
        self.assertEqual(ref1.slot, 1)

        # Capacity exhausted
        with self.assertRaises(ArenaCapacityError):
            arena.allocate_game(np.zeros(8, dtype=np.int32))

        # Free slot 0 and verify deterministic reuse
        arena.free_game(ref0)
        ref_reused = arena.allocate_game(np.zeros(8, dtype=np.int32))
        self.assertEqual(ref_reused.slot, 0)

    def test_workspace_capacity_limit_and_reuse(self) -> None:
        layout = ArenaLayout(game_capacity=2, workspace_capacity=2, state_width=8)
        arena = CpuStateArena(layout)
        game_ref = arena.allocate_game(np.zeros(8, dtype=np.int32))

        ws0 = arena.fork_workspace(game_ref)
        ws1 = arena.fork_workspace(game_ref)
        self.assertEqual(ws0.slot, 0)
        self.assertEqual(ws1.slot, 1)

        with self.assertRaises(ArenaCapacityError):
            arena.fork_workspace(game_ref)

        arena.free_workspace(ws0)
        ws_reused = arena.fork_workspace(game_ref)
        self.assertEqual(ws_reused.slot, 0)


class TestDoubleFreeAndLifecycleDependencies(unittest.TestCase):
    """Test rejection of double-free and preventing game free with active workspaces."""

    def setUp(self) -> None:
        self.layout = ArenaLayout(game_capacity=2, workspace_capacity=4, state_width=8)
        self.arena = CpuStateArena(self.layout)

    def test_double_free_game_rejected(self) -> None:
        ref = self.arena.allocate_game(np.zeros(8, dtype=np.int32))
        self.arena.free_game(ref)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.free_game(ref)

    def test_double_free_workspace_rejected(self) -> None:
        game_ref = self.arena.allocate_game(np.zeros(8, dtype=np.int32))
        ws_ref = self.arena.fork_workspace(game_ref)
        self.arena.free_workspace(ws_ref)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.free_workspace(ws_ref)

    def test_free_game_with_active_workspaces_rejected(self) -> None:
        game_ref = self.arena.allocate_game(np.zeros(8, dtype=np.int32))
        ws_ref = self.arena.fork_workspace(game_ref)

        # Rejection when workspace is still active
        with self.assertRaises(ArenaBusyError):
            self.arena.free_game(game_ref)

        # Free workspace first -> game release succeeds
        self.arena.free_workspace(ws_ref)
        self.arena.free_game(game_ref)
        self.assertEqual(self.arena.active_game_count, 0)

    def test_out_of_bounds_ref_rejected(self) -> None:
        bogus_game = GameRef(arena_id=self.arena.arena_id, slot=999, generation=1)
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.read_game(bogus_game)

        bogus_ws = WorkspaceRef(
            arena_id=self.arena.arena_id,
            slot=-1,
            generation=1,
            owner_game_slot=0,
            owner_game_generation=1,
        )
        with self.assertRaises(ArenaInvalidRefError):
            self.arena.read_workspace(bogus_ws)


class Test1600SlotAllocationAndMemoryReporting(unittest.TestCase):
    """Test 1600-slot configuration memory accounting without starting simulations."""

    def test_1600_capacity_layout_and_nbytes(self) -> None:
        # 1600 real games, 3200 workspaces, state width 512 int32
        game_cap = 1600
        ws_cap = 3200
        width = 512
        layout = ArenaLayout(
            game_capacity=game_cap,
            workspace_capacity=ws_cap,
            state_width=width,
        )
        arena = CpuStateArena(layout)

        expected_game_bytes = game_cap * width * 4  # np.int32 is 4 bytes
        expected_ws_bytes = ws_cap * width * 4
        expected_total = expected_game_bytes + expected_ws_bytes

        self.assertEqual(arena.allocated_bytes, expected_total)
        self.assertEqual(arena.active_game_count, 0)
        self.assertEqual(arena.active_workspace_count, 0)

        # Verify C-contiguous layout
        self.assertTrue(arena._game_rows.flags.c_contiguous)
        self.assertTrue(arena._workspace_rows.flags.c_contiguous)

        games = []
        workspaces = []
        for index in range(game_cap):
            init_row = np.zeros(width, dtype=np.int32)
            init_row[0] = index
            init_row[-1] = index + 1
            ref = arena.allocate_game(init_row)
            games.append(ref)
            workspaces.extend((arena.fork_workspace(ref), arena.fork_workspace(ref)))
        self.assertEqual(arena.active_game_count, game_cap)
        self.assertEqual(arena.active_workspace_count, ws_cap)
        with self.assertRaises(ArenaCapacityError):
            arena.allocate_game(np.zeros(width, dtype=np.int32))
        with self.assertRaises(ArenaCapacityError):
            arena.fork_workspace(games[0])
        for index in (0, game_cap // 2, game_cap - 1):
            self.assertEqual(arena.read_game(games[index])[0], index)
            self.assertEqual(arena.read_workspace(workspaces[2 * index])[-1], index + 1)
        for ref in workspaces:
            arena.free_workspace(ref)
        for ref in games:
            arena.free_game(ref)
        self.assertEqual(arena.active_game_count, 0)
        self.assertEqual(arena.active_workspace_count, 0)


class TestRefImmutabilityAndHashability(unittest.TestCase):
    """Test that refs are frozen dataclasses without arrays and can be used in sets/dicts."""

    def test_refs_are_hashable_and_immutable(self) -> None:
        gref = GameRef(arena_id=1, slot=0, generation=1)
        wsref = WorkspaceRef(
            arena_id=1,
            slot=0,
            generation=1,
            owner_game_slot=0,
            owner_game_generation=1,
        )

        ref_set = {gref}
        self.assertIn(gref, ref_set)

        ws_map = {wsref: "active"}
        self.assertEqual(ws_map[wsref], "active")

        with self.assertRaises(FrozenInstanceError):
            gref.slot = 1  # type: ignore

        with self.assertRaises(FrozenInstanceError):
            wsref.generation = 2  # type: ignore


if __name__ == "__main__":
    unittest.main()
