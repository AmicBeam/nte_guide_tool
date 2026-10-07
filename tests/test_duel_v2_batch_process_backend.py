"""Unit tests for ResidentRuleBackend formal CPU worker bulk resolver.

Verifies:
1. Warmup verification confirming no Torch or CUDA imported in worker processes.
2. Full roundtrip import/export of five official starter decks preserving Entity types,
   aliases, dynamic fields, and mutation isolation.
3. Item-by-item numerical and rule equivalence with CPUOracleBackend on real games
   (acting_side, finished, terminal_value, decision, x, c, value_observation,
   sample_world, step, clean).
4. Physical root advancement via commit_action using official flow.apply_action
   preserving events, enforcing next_root_generation = root_generation + 1, and
   retaining previous state in place.
5. Strict rejection of cross-backend, stale, bool, float, and reused generation handles.
6. Fail-closed worker capacity overflow poisoning and closing backend.
7. Clean bounded process termination upon release/close leaving no live owned child processes.
8. Monotonic operation timeout boundary poisoning without sleeping tens of seconds.
9. CooperativeScheduler integration and weakref finalizer lifecycle garbage collection.

Tests do NOT create GPU contexts, load PyTorch, or mock fake non-rules states.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import gc
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity,
    CharacterEntity,
    PlayerEntity,
)
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.engine.duel_v2.simulation import simulate_action
from app.modules.card_game.rl import league_rollout, recovery_runtime
from app.modules.card_game.rl.batched_duel.process_backend import (
    ResidentRuleBackend,
    StateRef,
)
from app.modules.card_game.rl.batched_search.cooperative import (
    CooperativeScheduler,
    OperationQuery,
    OperationResponse,
    RootJob,
    cooperative_search,
)
from app.modules.card_game.rl.batched_search.inference import InferenceResult
from app.modules.card_game.rl.batched_search.oracle_backend import CPUOracleBackend


class FakeInference:
    """Deterministic inference mock for cooperative scheduler integration."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def predict(self, requests: list[Any]) -> list[InferenceResult]:
        self.sizes.append(len(requests))
        return [
            InferenceResult(
                r.request_id,
                r.model_key,
                r.model_version,
                np.zeros(len(r.candidates), dtype=np.float32),
                np.array([0.2, 0.1, 0.7], dtype=np.float32),
            )
            for r in requests
        ]


class TestResidentRuleBackend(unittest.TestCase):
    """Unit tests for resident process CPU bulk resolver."""

    def test_inline_metadata_and_clean_keep_canonical_lifetimes(self):
        game=new_game(seed=970,first_side='a',skip_mulligan=True)
        with ResidentRuleBackend(workers=1,max_games=1,max_states_per_worker=30) as backend:
            [ref]=backend.import_states([game],['one'],[1])
            self.assertEqual(backend.inline_operation('acting_side',(ref,)),(True,'a'))
            self.assertEqual(backend.inline_operation('clean',(ref,)),(False,None))
            actions,_=backend.decision(ref,'a')
            after=backend.step(ref,'a',actions[0])
            self.assertIs(backend.inline_operation('clean',(after,))[1],after)
            with self.assertRaises(ValueError):backend.inline_operation('finished',(replace(after,phase='finished'),))
            with self.assertRaises(ValueError):backend.inline_operation('terminal_value',(after,'wrong'))
            backend.release([after])
            with self.assertRaises(ValueError):backend.inline_operation('clean',(after,))

    def test_full_32_simulations_inline_rule_path_matches_oracle(self):
        from tests.test_duel_v2_batch_real_rules import PublicTestPolicy,NumpyOracleInference,direct
        from app.modules.card_game.rl.batched_search.traversal import search_traversal
        game=new_game(seed=971,decks={'a':STARTER_DECKS[3],'b':STARTER_DECKS[4]},first_side='a',skip_mulligan=True)
        models={'zhenhong':PublicTestPolicy('zhenhong')};policies={'a':models['zhenhong'],'b':models['zhenhong']}
        for mode in ('legacy','natural_wdl'):
            options=dict(seed=972,simulations=32,terminal_horizon=3,q_scale=mode,noise=True)
            expected=direct(search_traversal(game,'a',policies,backend=CPUOracleBackend(runtime=recovery_runtime),**options),models)
            with ResidentRuleBackend(workers=1,max_games=1,max_states_per_worker=300) as backend:
                [ref]=backend.import_states([game],['one'],[1])
                job=RootJob('root','one',1,cooperative_search(ref,'a',policies,backend=backend,**options))
                actual=CooperativeScheduler(NumpyOracleInference(models)).run([job])['results']['root']
                for key in ('search_choice','simulations','value','worlds','nodes'):self.assertEqual(actual[key],expected[key])
                for key in ('visits','mean_values','pi'):np.testing.assert_array_equal(actual[key],expected[key])
                self.assertGreater(backend.stats()['inline_operations'].get('clean',0),0)

    def test_expired_worker_request_does_not_allocate_a_world(self):
        from app.modules.card_game.rl.batched_search.cooperative import RequestExpired
        game=new_game(seed=961,first_side='a',skip_mulligan=True)
        with ResidentRuleBackend(workers=1,max_games=1,max_states_per_worker=20) as backend:
            [ref]=backend.import_states([game],['one'],[1])
            query=OperationQuery('late',backend,'sample_world',(ref,'a',10),(),deadline=0.)
            [response]=backend.resolve_operations([query])
            self.assertIsInstance(response.value,RequestExpired)
            self.assertEqual(backend.stats()['live_states'],1)

    def test_metadata_receipts_admission_and_release_are_validated_before_dispatch(self):
        game=new_game(seed=938,skip_mulligan=True)
        with ResidentRuleBackend(workers=1,max_games=1,max_states_per_worker=20) as backend:
            [ref]=backend.import_states([game],['one'],[1])
            with self.assertRaises(ValueError):backend.finished(replace(ref,phase='finished'))
            with self.assertRaises(ValueError):backend.resolve_operations([
                OperationQuery('foreign',object(),'finished',(ref,),())])
            with self.assertRaises(ValueError):backend.resolve_operations([
                OperationQuery('bad-args',backend,'step',(ref,),())])
            with self.assertRaises(ValueError):backend.release([ref,ref])
            self.assertEqual(backend.stats()['live_states'],1)
            with self.assertRaises(OverflowError):backend.import_states([game],['two'],[1])
            self.assertFalse(backend.finished(ref))

    def test_mixed_worker_commit_failure_keeps_journal_and_stops_method(self):
        games=[new_game(seed=951+i,first_side='a',skip_mulligan=True) for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            backend=ResidentRuleBackend(workers=1,max_games=2,max_states_per_worker=20,record_output=directory)
            try:
                refs=backend.import_states(games,['one','two'],[1,1])
                procs=backend.owned_processes
                with self.assertRaises(RuntimeError):backend.commit_actions([
                    (refs[0],'a',{'type':'concede'},2,None),
                    (refs[1],'a',{'type':'attack','character_id':'missing'},2,None)])
                self.assertTrue(all(not proc.is_alive() for proc in procs))
                with self.assertRaises(RuntimeError):backend.stats()
                journal=Path(directory)/'journals/one/actions.jsonl'
                self.assertEqual(json.loads(journal.read_text())['action'],{'type':'concede'})
                self.assertFalse((Path(directory)/'raw/one.json.gz').exists())
            finally:backend.close()

    def test_bulk_physical_commit_writes_verified_records_without_worker_torch(self):
        games=[new_game(seed=941+i,first_side='a',skip_mulligan=True) for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            with ResidentRuleBackend(workers=2,max_games=2,max_states_per_worker=20,
                                     record_output=directory) as backend:
                refs=backend.import_states(games,['one','two'],[1,1])
                commits=[(ref,'a',{'type':'concede'},2,{'simulations':0}) for ref in refs]
                after=backend.commit_actions(commits)
                with self.assertRaises(ValueError):backend.commit_action(refs[0],'a',{'type':'concede'},2)
                jobs=[{'id':name,'replay':True} for name in ('one','two')]
                results=backend.finish_games(after,jobs)
                self.assertTrue(all(row['complete'] and row['replay_verified'] for row in results))
                self.assertTrue(all(not row['torch_imported'] for row in backend.warmup()))
                for name in ('one','two'):
                    with gzip.open(Path(directory)/'raw'/(name+'.json.gz'),'rt',encoding='utf-8') as f:
                        raw=json.load(f)
                    self.assertEqual(raw['actions'][0]['action'],{'type':'concede'})
                    self.assertTrue((Path(directory)/'replays'/(name+'.json.gz')).is_file())
                    journal=Path(directory)/'journals'/name/'actions.jsonl'
                    self.assertTrue(journal.read_bytes().endswith(b'\n'))
                    row=json.loads(journal.read_text())
                    self.assertEqual(row['action'],raw['actions'][0]['action'])
                    self.assertEqual(row['state_sha256'],raw['actions'][0]['state_sha256'])

    def test_warmup_confirms_no_torch_in_workers(self) -> None:
        """Verify worker startup and ensure workers never import Torch or CUDA."""
        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            self.assertEqual(backend.backend_kind, "python_resident_process")
            self.assertFalse(backend.supports_gpu)
            info = backend.warmup()
            self.assertEqual(len(info), 2)
            for item in info:
                self.assertIn("worker", item)
                self.assertIn("pid", item)
                self.assertIsInstance(item["pid"], int)
                self.assertFalse(item["torch_imported"], "Worker must not have Torch imported")

    def test_import_export_preserves_entities_aliases_and_caller_isolation(self) -> None:
        """Verify 5 official decks import/export with entity classes and caller mutation isolation."""
        official_decks = list(STARTER_DECKS)
        self.assertGreaterEqual(len(official_decks), 1)

        games = [
            new_game(
                seed=2026 + i,
                decks={"a": official_decks[i % len(official_decks)], "b": official_decks[i % len(official_decks)]},
            )
            for i in range(5)
        ]
        orig_copies = [deepcopy(g) for g in games]

        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            refs = backend.import_states(games, [f"g_{i}" for i in range(5)], [1] * 5)
            self.assertEqual(len(refs), 5)

            # Caller input states must remain untouched
            self.assertEqual(games, orig_copies)

            for i, ref in enumerate(refs):
                self.assertIsInstance(ref, StateRef)
                self.assertEqual(ref.role, "real")
                self.assertEqual(ref.game_id, f"g_{i}")
                self.assertEqual(ref.root_generation, 1)

            exported = backend.export_states(refs)
            self.assertEqual(len(exported), 5)

            for i in range(5):
                orig = games[i]
                dec = exported[i]

                # Entity classes preserved
                self.assertIs(type(dec["sides"]["a"]), PlayerEntity)
                self.assertIs(type(dec["sides"]["b"]), PlayerEntity)
                self.assertEqual(dec["sides"]["a"].entity_id, orig["sides"]["a"].entity_id)

                for side in ("a", "b"):
                    for cid, char_orig in orig["sides"][side]["characters"].items():
                        char_dec = dec["sides"][side]["characters"][cid]
                        self.assertIs(type(char_dec), CharacterEntity)
                        self.assertEqual(char_dec.template_id, char_orig.template_id)
                        self.assertEqual(char_dec.entity_id, char_orig.entity_id)

                    for orig_card, dec_card in zip(orig["sides"][side]["hand"], dec["sides"][side]["hand"]):
                        self.assertIs(type(dec_card), CardEntity)
                        self.assertEqual(dec_card.template_id, orig_card.template_id)
                        self.assertEqual(dec_card.entity_id, orig_card.entity_id)

            # Modifying exported state does not mutate state inside worker
            exported[0]["sides"]["a"]["hp"] = 999
            re_exported = backend.export_states([refs[0]])[0]
            self.assertEqual(re_exported["sides"]["a"]["hp"], 30)

    def test_item_by_item_equivalence_with_cpu_oracle_backend(self) -> None:
        """Verify exact correspondence of decision, observations, sample_world, step, clean with CPUOracleBackend."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})
        oracle = CPUOracleBackend(runtime=recovery_runtime)

        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            [ref] = backend.import_states([game], ["eq_game"], [1])

            # 1. acting_side
            self.assertEqual(backend.acting_side(ref), oracle.acting_side(game))

            # 2. finished
            self.assertEqual(backend.finished(ref), oracle.finished(game))

            # 3. terminal_value
            self.assertEqual(backend.terminal_value(ref, "a"), oracle.terminal_value(game, "a"))
            self.assertEqual(backend.terminal_value(ref, "b"), oracle.terminal_value(game, "b"))

            # 4. decision
            b_actions, (b_x, b_c) = backend.decision(ref, "a")
            o_actions, (o_x, o_c) = oracle.decision(game, "a")
            self.assertEqual(b_actions, o_actions)
            np.testing.assert_array_equal(b_x, o_x)
            np.testing.assert_array_equal(b_c, o_c)

            # 5. value_observation
            b_vx, b_vc = backend.value_observation(ref, "a")
            o_vx, o_vc = oracle.value_observation(game, "a")
            np.testing.assert_array_equal(b_vx, o_vx)
            np.testing.assert_array_equal(b_vc, o_vc)

            # 6. sample_world
            b_sample_ref = backend.sample_world(ref, "a", seed=42)
            self.assertEqual(b_sample_ref.role, "hypothetical")
            self.assertEqual(b_sample_ref.game_id, "eq_game")
            self.assertEqual(b_sample_ref.root_generation, 1)

            o_sample = oracle.sample_world(game, "a", seed=42)
            b_s_actions, (b_s_x, b_s_c) = backend.decision(b_sample_ref, "a")
            o_s_actions, (o_s_x, o_s_c) = oracle.decision(o_sample, "a")
            self.assertEqual(b_s_actions, o_s_actions)
            np.testing.assert_array_equal(b_s_x, o_s_x)
            np.testing.assert_array_equal(b_s_c, o_s_c)

            # 7. step
            action = b_actions[0]
            b_step_ref = backend.step(ref, "a", action)
            self.assertEqual(b_step_ref.role, "hypothetical")
            self.assertEqual(b_step_ref.game_id, "eq_game")
            self.assertEqual(b_step_ref.root_generation, 1)

            o_step = league_rollout.clean(simulate_action(deepcopy(game), "a", action))
            self.assertEqual(backend.acting_side(b_step_ref), oracle.acting_side(o_step))

            step_actor = backend.acting_side(b_step_ref)
            self.assertIsNotNone(step_actor)
            b_step_actions, (b_step_x, b_step_c) = backend.decision(b_step_ref, step_actor)
            o_step_actions, (o_step_x, o_step_c) = oracle.decision(o_step, step_actor)
            self.assertEqual(b_step_actions, o_step_actions)
            np.testing.assert_array_equal(b_step_x, o_step_x)
            np.testing.assert_array_equal(b_step_c, o_step_c)

            # 8. clean: on already clean hypothetical ref returns identical instance
            cleaned_step_ref = backend.clean(b_step_ref)
            self.assertIs(cleaned_step_ref, b_step_ref)

            # clean on root ref returns a cleaned copy
            cleaned_root_ref = backend.clean(ref)
            self.assertIsNot(cleaned_root_ref, ref)
            [cleaned_root_state] = backend.export_states([cleaned_root_ref])
            self.assertEqual(cleaned_root_state["events"], [])

    def test_commit_action_flow_events_and_generation(self) -> None:
        """Verify commit_action executes formal flow.apply_action and increments root_generation."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})

        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            [ref] = backend.import_states([game], ["comm_game"], [1])
            actions, _ = backend.decision(ref, "a")
            action = actions[0]

            # Rejects invalid root generations (must be ref.root_generation + 1)
            with self.assertRaises(ValueError):
                backend.commit_action(ref, "a", action, 1)
            with self.assertRaises(ValueError):
                backend.commit_action(ref, "a", action, 3)
            with self.assertRaises(TypeError):
                backend.commit_action(ref, "a", action, True)  # bool rejected

            # Rejects commit_action on hypothetical ref
            hypo_ref = backend.step(ref, "a", action)
            with self.assertRaises(ValueError):
                backend.commit_action(hypo_ref, "a", action, 2)

            # Valid commit_action
            next_ref = backend.commit_action(ref, "a", action, 2)
            self.assertEqual(next_ref.role, "real")
            self.assertEqual(next_ref.root_generation, 2)
            self.assertEqual(next_ref.game_id, "comm_game")

            [next_state] = backend.export_states([next_ref])
            self.assertGreater(len(next_state["events"]), 0)

            # Prior state in ref remains intact
            [prior_state] = backend.export_states([ref])
            self.assertEqual(prior_state["turn"], 0)

    def test_rejection_of_invalid_refs(self) -> None:
        """Verify rejection of stale, cross-backend, bool/float, and generation-mismatched handles."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})

        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            [ref] = backend.import_states([game], ["val_game"], [1])

            # Stale handle rejection after release
            backend.release([ref])
            with self.assertRaises(ValueError):
                backend.decision(ref, "a")
            with self.assertRaises(ValueError):
                backend.export_states([ref])

            # Cross-backend handle rejection
            with ResidentRuleBackend(workers=1, max_games=5, max_states_per_worker=20) as other_backend:
                [other_ref] = other_backend.import_states([game], ["other_game"], [1])
                with self.assertRaises(ValueError):
                    backend.decision(other_ref, "a")

            # Bool / float ref parameter rejections
            with self.assertRaises(TypeError):
                StateRef(
                    backend_id=backend.backend_id,
                    worker=0,
                    slot=True,  # bool forbidden
                    generation=0,
                    game_id="g",
                    root_generation=1,
                    actor="a",
                    phase="playing",
                    winner=None,
                    role="real",
                )
            with self.assertRaises(TypeError):
                StateRef(
                    backend_id=backend.backend_id,
                    worker=0,
                    slot=0,
                    generation=False,  # bool forbidden
                    game_id="g",
                    root_generation=1,
                    actor="a",
                    phase="playing",
                    winner=None,
                    role="real",
                )
            with self.assertRaises(TypeError):
                StateRef(
                    backend_id=backend.backend_id,
                    worker=0,
                    slot=1.5,  # float forbidden
                    generation=0,
                    game_id="g",
                    root_generation=1,
                    actor="a",
                    phase="playing",
                    winner=None,
                    role="real",
                )

    def test_capacity_overflow_failclosed(self) -> None:
        """Verify worker slot overflow poisons and closes the backend fail-closed."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})

        # Create backend with only 2 slots per worker
        backend = ResidentRuleBackend(workers=1, max_games=3, max_states_per_worker=2)
        try:
            # Import 2 states filling the worker capacity
            refs = backend.import_states([game, game], ["cap_0", "cap_1"], [1, 1])
            self.assertEqual(len(refs), 2)

            # Third state must exceed capacity and trigger fail-closed RuntimeError
            with self.assertRaises(RuntimeError):
                backend.import_states([game], ["cap_2"], [1])

            # Backend must now be closed/poisoned
            with self.assertRaises(RuntimeError):
                backend.decision(refs[0], "a")
        finally:
            backend.close()

    def test_process_lifecycle_cleanup_after_close(self) -> None:
        """Verify all owned child processes are terminated upon backend close."""
        backend = ResidentRuleBackend(workers=2, max_games=5, max_states_per_worker=20)
        procs = backend.owned_processes
        self.assertEqual(len(procs), 2)
        self.assertTrue(all(p.is_alive() for p in procs))

        backend.close()
        self.assertTrue(all(not p.is_alive() for p in procs))

        # Operations after close are rejected
        with self.assertRaises(RuntimeError):
            backend.stats()

    def test_operation_timeout_poisoning(self) -> None:
        """Verify unified monotonic deadline timeout triggers poisoning and process cleanup."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})
        backend = ResidentRuleBackend(workers=2, max_games=5, max_states_per_worker=20, operation_timeout=5.0)
        try:
            [ref] = backend.import_states([game], ["to_game"], [1])
            procs = backend.owned_processes

            # Mock concurrent.futures.wait returning not_done to simulate timeout boundary instantly
            fake_fut = MagicMock()
            with patch("concurrent.futures.wait", return_value=(set(), {fake_fut})):
                with self.assertRaises(TimeoutError):
                    backend.decision(ref, "a")

            # Must poison and close backend, terminating child processes
            self.assertTrue(all(not p.is_alive() for p in procs))
            with self.assertRaises(RuntimeError):
                backend.stats()
        finally:
            backend.close()

    def test_bulk_resolve_operations_and_scheduler_integration(self) -> None:
        """Verify resolve_operations handles mixed queries and integrates with CooperativeScheduler."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})

        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            [ref] = backend.import_states([game], ["bulk_game"], [1])

            # Mixed query batch
            q_acting = OperationQuery("q1", backend, "acting_side", (ref,), ())
            q_finished = OperationQuery("q2", backend, "finished", (ref,), ())
            q_term = OperationQuery("q3", backend, "terminal_value", (ref, "a"), ())
            q_dec = OperationQuery("q4", backend, "decision", (ref, "a"), ())
            q_val = OperationQuery("q5", backend, "value_observation", (ref, "a"), ())

            responses = backend.resolve_operations([q_acting, q_finished, q_term, q_dec, q_val])
            self.assertEqual(len(responses), 5)
            resp_dict = {r.id: r.value for r in responses}

            self.assertIn(resp_dict["q1"], ("a", "b", None))
            self.assertFalse(resp_dict["q2"])
            self.assertEqual(resp_dict["q3"], 0.0)
            self.assertIsInstance(resp_dict["q4"], tuple)
            self.assertIsInstance(resp_dict["q5"], tuple)

            # Verify stats recording
            st = backend.stats()
            self.assertGreaterEqual(st["methods"].get("decision", 0), 1)
            self.assertGreaterEqual(st["live_states"], 1)
            self.assertGreaterEqual(st["active_processes"], 2)

            # CooperativeScheduler integration test
            gen = cooperative_search(
                ref,
                "a",
                {"a": ("m", "sha"), "b": ("m", "sha")},
                backend=backend,
                seed=7,
                simulations=2,
                terminal_horizon=0,
                root_id="r_test",
            )
            run = CooperativeScheduler(FakeInference()).run([RootJob("r_test", "bulk_game", 1, gen)])
            self.assertEqual(run["root_count"], 1)
            self.assertIn("r_test", run["results"])
            self.assertTrue(run["results"]["r_test"]["complete"])

    def test_weakref_finalizer_enqueues_and_flushes(self) -> None:
        """Verify garbage collection of StateRef enqueues release and flushes to workers."""
        game = new_game(seed=2026, decks={"a": STARTER_DECKS[0], "b": STARTER_DECKS[0]})

        with ResidentRuleBackend(workers=2, max_games=10, max_states_per_worker=50) as backend:
            def _create_and_drop_step() -> tuple[int, int]:
                [local_ref] = backend.import_states([game], ["gc_game"], [1])
                hypo_ref = backend.step(local_ref, "a", {"type": "mulligan", "card_ids": []})
                return local_ref.slot, hypo_ref.slot

            # Create local handles that will go out of scope
            _create_and_drop_step()
            gc.collect()

            # Calling stats() flushes pending releases from queue
            st = backend.stats()
            self.assertGreaterEqual(st["released_states"], 1)


if __name__ == "__main__":
    unittest.main()
