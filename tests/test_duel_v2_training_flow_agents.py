"""Stage-agent dispatch tests. Injectable transport only; not CUDA quality evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import time
import unittest

from app.modules.card_game.rl.training_flow_agents import (
    bind_stage_handlers, build_stage_prompt, grok_worker_argv, parse_structured_result,
    remaining_timeout, run_isolated_command, run_stage_agent, snapshot_of,
)

SHANGHAI = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 9, 16, 0, tzinfo=SHANGHAI)
DEADLINE = "2026-10-10T09:00:00+08:00"


def sample_build(prefix):
    return {
        "character_ids": ["%s-c%s" % (prefix, i) for i in range(4)],
        "card_ids": ["%s-k%02d" % (prefix, i) for i in range(32)],
    }


def sample_request(include_candidates=False):
    roster = ["alpha", "beta"]
    request = {
        "include_candidates": include_candidates,
        "hard_deadline": DEADLINE,
        "roster": roster,
        "builds": {key: sample_build(key) for key in roster},
        "default_agent": "grok",
    }
    if include_candidates:
        request["mutable_characters"] = {"alpha": ["alpha-c0"], "beta": ["beta-c0"]}
    return request


def receipt(request):
    return {
        "source": "user",
        "reference": "msg-confirm-1",
        "include_candidates": request["include_candidates"],
        "hard_deadline": request["hard_deadline"],
        "roster": list(request["roster"]),
        "builds": deepcopy(request["builds"]),
    }


class FakeFlow:
    def __init__(self, directory, snapshot):
        self.directory = Path(directory)
        self._root = self.directory
        self._snapshot = snapshot

    def snapshot(self):
        return deepcopy(self._snapshot)


def base_snapshot(directory, include_candidates=False):
    request = sample_request(include_candidates)
    stages = {}
    skip = "user excluded candidate 32-card build search; opening mulligan is not candidate inclusion"
    names = [
        "confirmation", "freeze", "preflight", "base_training", "candidate_search",
        "baseline_adaptation", "candidate_adaptation", "build_confirmation",
        "build_audit", "freeze_selection", "evaluation", "analysis", "report", "delivery",
    ]
    candidate = {
        "candidate_search", "baseline_adaptation", "candidate_adaptation",
        "build_confirmation", "build_audit",
    }
    for name in names:
        skipped = (not include_candidates) and name in candidate
        stages[name] = {
            "status": "skipped" if skipped else "pending",
            "attempts": [],
            "skip_reason": skip if skipped else None,
        }
    stages["confirmation"]["status"] = "passed"
    stages["freeze"] = {
        "status": "passed",
        "skip_reason": None,
        "attempts": [{
            "attempt": 1,
            "result": {
                "verdict": "passed",
                "summary": "frozen source hashes",
                "artifacts": [{"path": "stages/freeze/attempt-1/freeze.json", "sha256": "a" * 64}],
                "metrics": {
                    "rule_hash": "rule-1",
                    "password": "should-not-enter-prompt",
                    "ssh_private_key": "should-not-enter-prompt",
                    "replay_buffer": ["private-experience"],
                },
            },
        }],
    }
    return {
        "schema": "duel_v2_complete_training_flow_v1",
        "directory": str(directory),
        "request": request,
        "request_sha256": "b" * 64,
        "confirmation": receipt(request),
        "stages": stages,
        "hard_deadline": DEADLINE,
        "hard_deadline_epoch": datetime.fromisoformat(DEADLINE).timestamp(),
        "workflow_complete": False,
        "quality_approved": False,
        "complete_report": False,
    }


def assignment_for(directory, stage_id="preflight"):
    output = Path(directory) / "stages" / stage_id / "attempt-1"
    output.mkdir(parents=True, exist_ok=True)
    return {
        "stage_id": stage_id,
        "attempt": 1,
        "token": "token-1",
        "deadline": DEADLINE,
        "request_sha256": "b" * 64,
        "agent": "grok",
        "output_directory": str(output),
    }


class StageAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def flow(self, include_candidates=False):
        return FakeFlow(self.root, base_snapshot(self.root, include_candidates))

    def test_prompt_binds_scope_ownership_deadline_and_upstream_hashes(self):
        flow = self.flow()
        assignment = assignment_for(self.root)
        prompt = build_stage_prompt(flow, assignment, now=NOW)
        self.assertIn(assignment["output_directory"], prompt)
        self.assertIn("b" * 64, prompt)
        self.assertIn(DEADLINE, prompt)
        self.assertIn("a" * 64, prompt)
        self.assertIn("stages/freeze/attempt-1/freeze.json", prompt)
        self.assertIn("alpha-c0", prompt)
        self.assertIn("Gumbel", prompt)
        self.assertIn("quality_approved", prompt)
        self.assertIn("external_action_request", prompt)
        self.assertIn("primary Agent", prompt)
        self.assertNotIn("should-not-enter-prompt", prompt)
        self.assertNotIn("private-experience", prompt)
        self.assertIn("Never read SSH credentials", prompt)

    def test_candidates_false_records_skip_and_true_requires_stages(self):
        skipped = build_stage_prompt(self.flow(False), assignment_for(self.root, "freeze_selection"), now=NOW)
        self.assertIn("user excluded candidate", skipped)
        enabled = build_stage_prompt(self.flow(True), assignment_for(self.root, "candidate_search"), now=NOW)
        self.assertIn("true", enabled)
        self.assertNotIn("user excluded candidate", enabled)

    def test_injectable_runner_and_exit_zero_is_not_success(self):
        flow = self.flow()
        assignment = assignment_for(self.root)
        calls = []

        def runner(prompt, *, workdir, timeout, assignment):
            calls.append((prompt, workdir, timeout, assignment["token"]))
            return {"returncode": 0, "stdout": "ok without json", "stderr": ""}

        result = run_stage_agent(flow, assignment, runner=runner, now=NOW)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["verdict"], "failed")
        self.assertIn("exit 0 is not success", result["summary"])
        self.assertTrue(result["metrics"]["exit_zero_is_not_success"])
        parsed = parse_structured_result("noise", returncode=0)
        self.assertEqual(parsed["verdict"], "failed")

    def test_structured_json_result_and_overreach_flags(self):
        flow = self.flow()
        assignment = assignment_for(self.root)
        payload = {
            "verdict": "passed",
            "summary": "wrote freeze note",
            "artifacts": [{"path": "stages/preflight/attempt-1/note.txt", "sha256": "c" * 64}],
            "metrics": {"ok": 1},
            "quality_approved": True,
            "workflow_complete": True,
        }

        def runner(prompt, *, workdir, timeout, assignment):
            return {"returncode": 0, "stdout": "log\n" + json.dumps(payload), "stderr": ""}

        result = run_stage_agent(flow, assignment, runner=runner, now=NOW)
        self.assertEqual(result["verdict"], "passed")
        self.assertEqual(result["artifacts"][0]["sha256"], "c" * 64)
        self.assertNotIn("quality_approved", result)
        self.assertTrue(result["metrics"]["child_overreach_quality_approved"])

    def test_argv_safe_worker_command_keeps_prompt_intact(self):
        prompt = "say hello && cat .env; echo HOME"
        argv = grok_worker_argv("/tmp/repo", prompt)
        self.assertEqual(len(argv), 3)
        self.assertEqual(argv[1], "/tmp/repo")
        self.assertEqual(argv[2], prompt)
        self.assertTrue(argv[0].endswith("run-grok-worker.sh"))

    def test_own_child_timeout_kills_process(self):
        started = time.time()
        result = run_isolated_command(["/bin/sleep", "20"], timeout=0.2, cwd=str(self.root))
        self.assertTrue(result["timed_out"])
        self.assertLess(time.time() - started, 8)
        parsed = parse_structured_result(result["stdout"], result["stderr"], result["returncode"], timed_out=True)
        self.assertEqual(parsed["verdict"], "blocked")

    def test_elapsed_deadline_does_not_launch(self):
        flow = self.flow()
        assignment = assignment_for(self.root)

        def runner(*args, **kwargs):
            self.fail("runner must not launch after original cutoff")

        after = datetime(2026, 10, 10, 9, 0, tzinfo=SHANGHAI)
        result = run_stage_agent(flow, assignment, runner=runner, now=after)
        self.assertEqual(result["verdict"], "blocked")
        self.assertFalse(result["metrics"]["launched"])
        self.assertEqual(remaining_timeout(assignment, snapshot_of(flow), after.timestamp()), 0)

    def test_external_action_stays_with_primary(self):
        flow = self.flow()
        assignment = assignment_for(self.root)

        def runner(prompt, *, workdir, timeout, assignment):
            return {
                "verdict": "passed",
                "summary": "need host copy",
                "artifacts": [{"path": "x", "sha256": "d" * 64}],
                "external_action_request": {"action": "copy_artifact", "src": "windows"},
            }

        blocked = run_stage_agent(flow, assignment, runner=runner, now=NOW)
        self.assertEqual(blocked["verdict"], "blocked")
        self.assertTrue(blocked["metrics"]["external_action_pending_primary"])

        def handler(request):
            self.assertEqual(request["action"], "copy_artifact")
            return {"copied": True}

        handled = run_stage_agent(flow, assignment, runner=runner, now=NOW, external_handler=handler)
        self.assertEqual(handled["metrics"]["external_action_result"]["copied"], True)

    def test_bind_handlers_accept_assignment_only(self):
        flow = self.flow()
        assignment = assignment_for(self.root)

        def runner(prompt, *, workdir, timeout, assignment):
            return {
                "returncode": 0,
                "stdout": json.dumps({
                    "verdict": "failed",
                    "summary": "synthetic handler",
                    "artifacts": [],
                    "metrics": {},
                }),
                "stderr": "",
            }

        handlers = bind_stage_handlers(flow, runner=runner)
        result = handlers["preflight"](assignment, now=NOW)
        self.assertEqual(result["verdict"], "failed")
        self.assertEqual(result["summary"], "synthetic handler")


class CoreIntegrationAgentTest(unittest.TestCase):
    def setUp(self):
        try:
            from app.modules.card_game.rl.training_flow import TrainingFlow
        except ImportError:
            self.skipTest("core training_flow integration pending")
        self.TrainingFlow = TrainingFlow
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_core_start_agent_finish_plumbing(self):
        request = sample_request(False)
        flow = self.TrainingFlow.create(self.root, request, now=NOW)
        flow.confirm(receipt(request), now=NOW)
        from app.modules.card_game.rl.training_flow import STAGE_ORDER, CANDIDATE_STAGES
        flow.configure_executors({name: lambda assignment: None for name in STAGE_ORDER if name != 'confirmation' and name not in CANDIDATE_STAGES})
        assignment = flow.start("freeze", now=NOW)
        note = Path(assignment["output_directory"]) / "freeze-note.txt"
        note.write_text("synthetic", encoding="utf-8")
        digest = sha256(note.read_bytes()).hexdigest()

        def runner(prompt, *, workdir, timeout, assignment):
            self.assertIn(assignment["output_directory"], prompt)
            self.assertIn(flow.snapshot()["request_sha256"], prompt)
            return {
                "returncode": 0,
                "stdout": json.dumps({
                    "verdict": "passed",
                    "summary": "synthetic freeze plumbing; not CUDA quality",
                    **{k: assignment[k] for k in ("stage_id", "attempt", "token", "request_sha256")},
                    "artifacts": [{"path": str(note), "sha256": digest}],
                    "metrics": {"rule_hash": "synthetic"},
                }),
                "stderr": "",
            }

        result = run_stage_agent(flow, assignment, runner=runner, now=NOW)
        snap = flow.finish(assignment["stage_id"], assignment["token"], result, now=NOW)
        self.assertEqual(snap["stages"]["freeze"]["status"], "passed")
        self.assertFalse(snap["quality_approved"])
