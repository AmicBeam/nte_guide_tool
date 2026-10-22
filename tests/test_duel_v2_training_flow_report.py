"""Chinese flow-report tests. Synthetic fixtures are not CUDA quality evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.modules.card_game.rl.training_flow_report import render_flow_report, write_flow_report

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


def empty_stages(include_candidates=False):
    names = [
        "confirmation", "freeze", "preflight", "base_training", "candidate_search",
        "baseline_adaptation", "candidate_adaptation", "build_confirmation",
        "build_audit", "freeze_selection", "evaluation", "analysis", "report", "delivery",
    ]
    candidate = {
        "candidate_search", "baseline_adaptation", "candidate_adaptation",
        "build_confirmation", "build_audit",
    }
    skip = "user excluded candidate 32-card build search; opening mulligan is not candidate inclusion"
    stages = {}
    for name in names:
        skipped = (not include_candidates) and name in candidate
        stages[name] = {
            "status": "skipped" if skipped else "pending",
            "attempts": [],
            "skip_reason": skip if skipped else None,
        }
    stages["confirmation"]["status"] = "passed"
    return stages


def passed(metrics, artifacts=None, summary="synthetic"):
    return {
        "status": "passed",
        "skip_reason": None,
        "attempts": [{
            "attempt": 1,
            "result": {
                "verdict": "passed",
                "summary": summary,
                "artifacts": artifacts or [{"path": "note.txt", "sha256": "e" * 64}],
                "metrics": metrics,
            },
        }],
    }


class FlowReportTest(unittest.TestCase):
    def test_global_games_are_not_presented_as_each_teams_games(self):
        flow = self.make(metrics={
            "complete_training_games": 56,
            "formal_policy_updates": {"alpha": 20, "beta": 24},
        })
        text = render_flow_report(flow)
        self.assertIn("| alpha | 56（全轮总计） | 20 |", text)
        self.assertNotIn("| alpha | 56 | 20 |", text)

    def test_global_updates_keep_their_scope_with_per_team_games(self):
        flow = self.make(metrics={
            "complete_training_games": {"alpha": 14, "beta": 12},
            "formal_policy_updates": 44,
        })
        text = render_flow_report(flow)
        self.assertIn("| beta | 12 | 44（全轮总计） |", text)

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "run"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, include_candidates=False, stages=None, metrics=None, flags=None):
        request = sample_request(include_candidates)
        snapshot = {
            "schema": "duel_v2_complete_training_flow_v1",
            "directory": str(self.root),
            "request": request,
            "request_sha256": "req" + "0" * 61,
            "confirmation": receipt(request),
            "stages": stages or empty_stages(include_candidates),
            "hard_deadline": DEADLINE,
            "hard_deadline_epoch": datetime.fromisoformat(DEADLINE).timestamp(),
            "workflow_complete": False,
            "quality_approved": False,
            "complete_report": False,
            "metrics": metrics or {},
        }
        if flags:
            snapshot.update(flags)
        return FakeFlow(self.root, snapshot)

    def test_required_chinese_sections_and_card_counts(self):
        text = render_flow_report(self.make(False))
        for heading in (
            "范围与截止", "实际构筑张数表", "候选变更与接受决定",
            "Gumbel / 终局 WDL / 更新身份", "初始 / 选定 / 最新模型身份",
            "N×N 评估分母", "逐卡 / 异能者 / 召唤 / 使用 / 释放 / 机制",
            "原始记录 / 公开录像 / 模型交付", "阶段尝试与缺失证据",
            "流程完整性与质量批准", "最终审计依据",
        ):
            self.assertIn(heading, text)
        self.assertIn("alpha-c0", text)
        self.assertIn("32", text)
        self.assertIn("CUDA/Gumbel", text)
        self.assertIn("按用户确认跳过", text)
        self.assertIn("32-card build", text)

    def test_missing_is_uncompleted_not_zero(self):
        text = render_flow_report(self.make(False))
        self.assertIn("未完成", text)
        self.assertNotIn("0.0%", text)
        self.assertNotIn("| 0%", text)
        self.assertIn("workflow_complete", text)
        self.assertIn("quality_approved", text)
        self.assertIn("相互独立", text)

    def test_candidate_changes_and_zero_updates(self):
        stages = empty_stages(True)
        stages["candidate_search"] = passed({
            "candidate_changes": [{
                "build": "alpha",
                "removed": "alpha-k00",
                "added": "alpha-k99",
                "parent": "original",
                "decision": "retain_baseline",
            }],
            "acceptance_decision": "retain_baseline",
        })
        stages["base_training"] = passed({
            "complete_training_games": {"alpha": 0, "beta": 12},
            "formal_policy_updates": {"alpha": 0, "beta": 4},
            "learning": {
                "algorithm": "gumbel",
                "simulations": 32,
                "gumbel_candidates": 16,
                "reward": "terminal_wdl",
                "terminal_wdl": True,
            },
            "models": {
                "alpha": {"initial": "i1", "selected": "s1", "latest": "l1"},
                "beta": {"initial": "i2", "selected": "s2", "latest": "l2"},
            },
        })
        text = render_flow_report(self.make(True, stages=stages))
        self.assertIn("alpha-k00", text)
        self.assertIn("retain_baseline", text)
        self.assertIn("未完成（0）", text)
        self.assertIn("gumbel", text)
        self.assertIn("terminal_wdl", text)
        self.assertIn("i1", text)
        self.assertIn("s1", text)
        self.assertIn("l1", text)

    def test_nxn_and_detail_topics_and_metadata_only_delivery(self):
        stages = empty_stages(False)
        stages["evaluation"] = passed({
            "evaluation": {
                "modes": {
                    "pure": {"planned_cells": 4, "completed_cells": 4},
                    "search": {"planned_cells": 4, "completed_cells": 0},
                },
                "baseline": {"planned_cells": 2, "completed_cells": 2},
                "new_old": {"planned_cells": 2, "completed_cells": 0},
                "mulligan": None,
                "varied_opponents": {"planned_cells": 8, "completed_cells": 1},
            },
        })
        stages["analysis"] = passed({
            "analysis": {
                "card": {"complete": True, "rows": [1]},
                "esper": {"complete": False},
                "summon": {"status": "missing"},
                "use": {"games": 3},
                "release": {},
                "mechanism": {"complete": True, "rows": [1]},
            },
        })
        stages["delivery"] = passed({
            "delivery": {
                "raw": [],
                "public_replays": [{"path": "replays/a.json", "sha256": "f" * 64}],
                "models": [{"path": "models/alpha.json", "sha256": "1" * 64}],
                "metadata_only": True,
            },
        })
        text = render_flow_report(self.make(False, stages=stages))
        self.assertIn("纯网络", text)
        self.assertIn("带搜索", text)
        self.assertIn("未完成（0/4）", text)
        self.assertIn("基线对照", text)
        self.assertIn("新旧模型对照", text)
        self.assertIn("起手换牌对照", text)
        self.assertIn("变化对手", text)
        self.assertIn("逐卡", text)
        self.assertIn("异能者", text)
        self.assertIn("召唤", text)
        self.assertIn("仅有清单，无数值文件", text)
        self.assertIn("1 个哈希文件", text)
        self.assertIn("未完成", text)

    def test_quality_flag_does_not_hide_gaps(self):
        flow = self.make(False, flags={"workflow_complete": True, "quality_approved": True, "complete_report": True})
        text = render_flow_report(flow)
        self.assertIn("与缺失证据矛盾", text)
        self.assertIn("不能填补", text)
        self.assertIn("不能晋升模型", text)

    def test_write_stays_in_run_root_and_rejects_traversal(self):
        flow = self.make(False)
        saved = write_flow_report(flow)
        path = self.root / saved["path"]
        self.assertTrue(path.is_file())
        self.assertEqual(sha256(path.read_bytes()).hexdigest(), saved["sha256"])
        self.assertIn("完整训练流程报告", path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            write_flow_report(flow, path="../outside.md")


class CoreIntegrationReportTest(unittest.TestCase):
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

    def test_render_and_write_from_core_snapshot(self):
        request = sample_request(False)
        flow = self.TrainingFlow.create(self.root, request, now=NOW)
        flow.confirm(receipt(request), now=NOW)
        text = render_flow_report(flow)
        self.assertIn("范围与截止", text)
        self.assertIn("alpha-c0,alpha-c1,alpha-c2,alpha-c3", text)
        self.assertIn("按用户确认跳过", text)
        saved = write_flow_report(flow)
        self.assertEqual((self.root / saved["path"]).read_text(encoding="utf-8"), text if text.endswith("\n") else text)
