"""Unit tests for Duel V2 batched cooperative evaluation.

Tests cover:
1. Default CLI execution emits pure configuration check without Torch or workers.
2. CLI and evaluator input validation on paths, capacities, and budgets.
3. 25-cell / 1600-game balanced matrix generation, seat alternation, paired seeds, and dedup.
4. Bounded end-to-end smoke test on real Duel V2 engine with 2 workers, NumpyOracleInference,
   and PublicTestPolicy, verifying partial physical trace recording, replay verification,
   and clean slot lifecycle.
"""
from __future__ import annotations

import collections
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.rl import recovery_policy, recovery_runtime
from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
from app.modules.card_game.rl.batched_search.evaluator import (
    evaluate_jobs,
    generate_matrix_jobs,
    compute_matrix_summary,
    write_matrix_csv,
    generate_report_markdown,
    wilson_score_interval,
)
from scripts.evaluate_duel_v2_batched import main, parse_args
from tests.test_duel_v2_batch_real_rules import PublicTestPolicy, NumpyOracleInference


class TestDuelV2BatchedEvaluator(unittest.TestCase):
    def test_cli_default_config_check_no_run(self) -> None:
        """Default CLI must only print config check without starting processes or loading weights."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = main([])
        self.assertEqual(ret, 0)
        output = buf.getvalue().strip()
        data = json.loads(output)
        self.assertEqual(data["action"], "config_check_only")
        self.assertFalse(data["run"])
        self.assertFalse(data["processes_started"])
        self.assertFalse(data["weights_loaded"])
        self.assertEqual(data["workers"], 16)
        self.assertEqual(data["games"], 1600)
        self.assertEqual(data["seconds"], 900)

    def test_cli_validation_invalid_args(self) -> None:
        """CLI must reject invalid workers, games, seconds, and missing run parameters."""
        with self.assertRaises(SystemExit):
            parse_args(["--workers", "0"])

        with self.assertRaises(SystemExit):
            parse_args(["--workers", "65"])

        with self.assertRaises(SystemExit):
            parse_args(["--games", "0"])

        with self.assertRaises(SystemExit):
            parse_args(["--games", "1601"])

        with self.assertRaises(SystemExit):
            parse_args(["--seconds", "0"])

        with self.assertRaises(SystemExit):
            parse_args(["--seconds", "14401"])

        # --run requires --output, --seed-base, and --budget-mode
        with self.assertRaises(SystemExit):
            parse_args(["--run"])

        with self.assertRaises(SystemExit):
            parse_args(["--run", "--output", "out_dir"])

        with self.assertRaises(SystemExit):
            parse_args(["--run", "--output", "out_dir", "--seed-base", "1000"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            existing = Path(tmp_dir) / "already_exists"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                main(["--run", "--output", str(existing), "--seed-base", "1000", "--budget-mode", "simulations"])

    def test_evaluator_input_validation(self) -> None:
        """evaluate_jobs must reject empty jobs, invalid budgets, duplicate IDs, and out-of-range limits."""
        dummy_models = {"starter": PublicTestPolicy("starter")}
        dummy_inf = NumpyOracleInference(dummy_models)
        dummy_backend = None

        with self.assertRaises(ValueError):
            evaluate_jobs([], dummy_models, dummy_inf, dummy_backend, budget_mode="simulations")

        with self.assertRaises(ValueError):
            evaluate_jobs([{"id": "g1"}], dummy_models, dummy_inf, dummy_backend, budget_mode="unsupported")

        with self.assertRaises(ValueError):
            evaluate_jobs([{"id": "g1"}], dummy_models, dummy_inf, dummy_backend, budget_mode="simulations", simulations=0)

        with self.assertRaises(ValueError):
            evaluate_jobs(
                [{"id": "g1"}],
                dummy_models,
                dummy_inf,
                dummy_backend,
                budget_mode="simulations",
                simulations=32,
                decision_seconds=-1.0,
            )

        # Duplicate game IDs
        dup_jobs = [
            {"id": "g_dup", "cell": 0, "seed": 1, "first": "a", "decks": {"a": {}, "b": {}}},
            {"id": "g_dup", "cell": 0, "seed": 2, "first": "b", "decks": {"a": {}, "b": {}}},
        ]
        with self.assertRaises(ValueError):
            evaluate_jobs(dup_jobs, dummy_models, dummy_inf, dummy_backend, budget_mode="simulations")

        # Duplicate cell/seed/first configuration
        dup_config_jobs = [
            {"id": "g1", "cell": 0, "seed": 100, "first": "a", "decks": {"a": {}, "b": {}}},
            {"id": "g2", "cell": 0, "seed": 100, "first": "a", "decks": {"a": {}, "b": {}}},
        ]
        with self.assertRaises(ValueError):
            evaluate_jobs(dup_config_jobs, dummy_models, dummy_inf, dummy_backend, budget_mode="simulations")

    def test_balanced_25_matrix_1600_jobs(self) -> None:
        """Matrix generator must yield balanced 5x5 cells with seat alternation and paired seed dedup."""
        keys = ("starter", "weave-rush", "quick-rush", "zhenhong", "murk")
        decks = {d["id"]: d for d in STARTER_DECKS}

        jobs = generate_matrix_jobs(keys, decks, seed_base=10000, total_games=1600)
        self.assertEqual(len(jobs), 1600)

        # Verify all 25 cells have exactly 64 jobs
        cell_counts = collections.Counter(j["cell"] for j in jobs)
        self.assertEqual(len(cell_counts), 25)
        for c in range(25):
            self.assertEqual(cell_counts[c], 64)

        # Verify unique game IDs
        game_ids = [j["id"] for j in jobs]
        self.assertEqual(len(set(game_ids)), 1600)

        # Verify deduplication key (cell, seed, first) is unique across all 1600 jobs
        dedup_keys = [(j["cell"], j["seed"], j["first"]) for j in jobs]
        self.assertEqual(len(set(dedup_keys)), 1600)

        # Verify pairing: jobs in the same cell at pair_index 2k and 2k+1 share identical seed
        for cell in range(25):
            cell_jobs = [j for j in jobs if j["cell"] == cell]
            self.assertEqual(len(cell_jobs), 64)

            # 32 first='a' and 32 first='b'
            a_first = [j for j in cell_jobs if j["first"] == "a"]
            b_first = [j for j in cell_jobs if j["first"] == "b"]
            self.assertEqual(len(a_first), 32)
            self.assertEqual(len(b_first), 32)

            row_model = keys[cell // 5]
            col_model = keys[cell % 5]

            for j in cell_jobs:
                self.assertEqual(j["matrix_row"], row_model)
                self.assertEqual(j["column"], col_model)

                pair_idx = j["pair_index"]
                expected_seed = 10000 + cell * 1_000_000 + pair_idx // 2
                self.assertEqual(j["seed"], expected_seed)

                # Alternation check
                if pair_idx % 2 == 0:
                    self.assertEqual(j["first"], "a")
                    self.assertEqual(j["first_engine_side"], "a")
                    self.assertEqual(j["policies"]["a"], row_model)
                    self.assertEqual(j["policies"]["b"], col_model)
                else:
                    self.assertEqual(j["first"], "b")
                    self.assertEqual(j["first_engine_side"], "b")
                    self.assertEqual(j["policies"]["b"], row_model)
                    self.assertEqual(j["policies"]["a"], col_model)

    def test_real_engine_bounded_smoke_partial_recording(self) -> None:
        """Smoke test on real Duel V2 engine: 2 workers, max_actions=2, verifying partial record emission."""
        keys = ("starter", "weave-rush")
        decks = {d["id"]: d for d in STARTER_DECKS}

        m_starter = PublicTestPolicy("starter")
        m_starter.schema = recovery_policy.RESIDUAL_SCHEMA
        m_weave = PublicTestPolicy("weave-rush")
        m_weave.schema = recovery_policy.RESIDUAL_SCHEMA

        models = {"starter": m_starter, "weave-rush": m_weave}
        inference = NumpyOracleInference(models)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            backend = ResidentRuleBackend(
                workers=2,
                max_games=2,
                runtime=recovery_runtime,
                record_output=out_path,
            )

            jobs = [
                {
                    "id": "smoke_g0001",
                    "cell": 0,
                    "seed": 88001,
                    "first": "a",
                    "decks": {"a": decks["starter"], "b": decks["weave-rush"]},
                    "policies": {"a": "starter", "b": "weave-rush"},
                    "matrix_row": "starter",
                    "column": "weave-rush",
                    "replay": True,
                },
                {
                    "id": "smoke_g0002",
                    "cell": 1,
                    "seed": 88002,
                    "first": "b",
                    "decks": {"a": decks["starter"], "b": decks["weave-rush"]},
                    "policies": {"a": "weave-rush", "b": "starter"},
                    "matrix_row": "starter",
                    "column": "weave-rush",
                    "replay": True,
                },
            ]

            with backend:
                res = evaluate_jobs(
                    jobs=jobs,
                    models=models,
                    inference=inference,
                    backend=backend,
                    budget_mode="simulations",
                    decision_seconds=2.0,
                    simulations=4,
                    proof_cap=2,
                    max_actions=2,  # Bounded to 2 real physical actions
                    output_dir=out_path,
                )

            # Assert evaluation structure
            self.assertEqual(res["total_games"], 2)
            self.assertEqual(res["completed_games"], 0)  # Interrupted by max_actions limit
            self.assertEqual(res["uncompleted_games"], 2)
            self.assertFalse(res["game_execution_complete"])
            self.assertFalse(res["concurrency_1600_full_games_complete"])
            self.assertFalse(res["quality_approved"])
            self.assertEqual(res["training_updates"], 0)
            self.assertFalse(res["gpu_rule_execution"])
            self.assertFalse(res["training_workflow_complete"])
            self.assertGreater(res["peak_outstanding_roots"], 0)

            # Verify index.jsonl contains partial records with complete=False and replay_verified=True
            index_file = out_path / "index.jsonl"
            self.assertTrue(index_file.exists())
            lines = [json.loads(line) for line in index_file.read_text(encoding="utf-8").strip().split("\n")]
            self.assertEqual(len(lines), 2)
            for item in lines:
                self.assertFalse(item["complete"])
                self.assertTrue(item["replay_verified"])
                self.assertIn("search_simulations", item)

            # Verify raw trace files were written to disk
            raw1 = out_path / "raw" / "smoke_g0001.json.gz"
            raw2 = out_path / "raw" / "smoke_g0002.json.gz"
            self.assertTrue(raw1.exists())
            self.assertTrue(raw2.exists())

            with gzip.open(raw1, "rt", encoding="utf-8") as f:
                payload1 = json.load(f)
                self.assertFalse(payload1["complete"])
                self.assertTrue(payload1["replay_verified"])
                self.assertEqual(len(payload1["actions"]), 2)
                self.assertEqual(payload1["completion_reason"], "max_actions")

            # Verify progress.json exists
            prog_file = out_path / "progress.json"
            self.assertTrue(prog_file.exists())
            prog_data = json.loads(prog_file.read_text(encoding="utf-8"))
            self.assertEqual(prog_data["phase"], "finished")
            self.assertEqual(prog_data["total_games"], 2)
            self.assertEqual(prog_data["uncompleted_games"], 2)
            self.assertFalse(res["evaluation_accepted"])
            self.assertTrue(res["behavior"]["behavior_approved"])

    def test_matrix_summary_and_wilson_ci(self) -> None:
        """Wilson score interval and summary aggregation must correctly compute intervals."""
        # Edge cases
        self.assertEqual(wilson_score_interval(0, 0), (None, None))
        lower, upper = wilson_score_interval(32, 64)
        self.assertAlmostEqual(lower, 0.3804, delta=0.01)
        self.assertAlmostEqual(upper, 0.6196, delta=0.01)

        # Mock results test
        keys = ("starter", "weave-rush", "quick-rush", "zhenhong", "murk")
        mock_results = [
            {"matrix_row": "starter", "column": "weave-rush", "winner": "a", "first": "a", "complete": True},
            {"matrix_row": "starter", "column": "weave-rush", "winner": "b", "first": "b", "complete": True},
            {"matrix_row": "starter", "column": "weave-rush", "winner": "b", "first": "a", "complete": True},
            {"matrix_row": "starter", "column": "weave-rush", "complete": False},
        ]
        summary = compute_matrix_summary(keys, mock_results)
        cell = next(c for c in summary["cells"] if c["first_model"] == "starter" and c["second_model"] == "weave-rush")
        self.assertEqual(cell["wins"], 2)
        self.assertEqual(cell["losses"], 1)
        self.assertEqual(cell["draws"], 0)
        self.assertEqual(cell["completed"], 3)
        self.assertEqual(cell["uncompleted"], 1)
        self.assertAlmostEqual(cell["win_rate"], 2 / 3, places=3)
        self.assertIsNotNone(cell["wilson_ci_lower"])
        self.assertIsNotNone(cell["wilson_ci_upper"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "matrix.csv"
            write_matrix_csv(summary, csv_path)
            self.assertTrue(csv_path.exists())
            content = csv_path.read_text(encoding="utf-8")
            self.assertIn("starter,weave-rush,2,0,1,3,1", content)

        report = generate_report_markdown(summary, {"completed_games": 3, "total_games": 4}, {})
        self.assertIn("Duel V2 Batched Cooperative Evaluation Report", report)
        self.assertIn("quality_approved", report)
        self.assertIn("concurrency_1600_full_games_complete", report)


if __name__ == "__main__":
    unittest.main()
