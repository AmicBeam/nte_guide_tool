"""Cooperative real-game evaluation orchestrator for Duel V2 batched search.

Drives batches of up to 1,600 resident real games using official Duel V2 rules
on worker processes, cooperative Gumbel serving search, and batch neural inference.
Never falls back to CPU or fake rules; never executes GPU rules.
"""
from __future__ import annotations

import collections
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4
import numpy as np

from app.modules.card_game.engine.duel_v2 import new_game
from .cooperative import CooperativeScheduler, RootJob
from .serving import serving_search
from .traversal import PolicyIdentity


def wilson_score_interval(wins: int, total: int, confidence: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Compute Wilson score interval for binomial proportion."""
    if total <= 0:
        return None, None
    z = 1.96  # 95% confidence
    p = wins / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    variance_term = (p * (1.0 - p) / total) + (z * z) / (4.0 * total * total)
    spread = (z * math.sqrt(max(0.0, variance_term))) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(lower, 4), round(upper, 4)


def generate_matrix_jobs(
    keys: Sequence[str],
    serving_decks: Mapping[str, Any],
    seed_base: int,
    total_games: int = 1600,
    replay: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Generate 5x5 balanced matrix evaluation jobs with paired engine seat alternation.

    Matrix cell is index % 25 (row = first player model, col = second player model).
    pairIndex is index // 25, alternating first_engine_side between 'a' and 'b'.
    Both seats in a pair share the integer seed (seed_base + cell*1000000 + pairIndex//2).
    """
    if type(total_games) is not int or total_games < 1 or total_games > 1600:
        raise ValueError(f"total_games must be an integer 1..1600, got {total_games!r}")
    if len(keys) != 5:
        raise ValueError(f"Exactly 5 keys required for 5x5 matrix, got {len(keys)}")
    if type(seed_base) is not int:
        raise TypeError("seed_base must be an integer")

    jobs = []
    seen_dedup = set()
    for index in range(total_games):
        cell = index % 25
        row_idx = cell // 5
        col_idx = cell % 5
        row_model = keys[row_idx]
        col_model = keys[col_idx]

        pair_index = index // 25
        first_engine_side = 'a' if (pair_index % 2 == 0) else 'b'

        if first_engine_side == 'a':
            first = 'a'
            policies = {'a': row_model, 'b': col_model}
            decks = {'a': serving_decks[row_model], 'b': serving_decks[col_model]}
        else:
            first = 'b'
            policies = {'a': col_model, 'b': row_model}
            decks = {'a': serving_decks[col_model], 'b': serving_decks[row_model]}

        seed = seed_base + cell * 1_000_000 + pair_index // 2
        game_id = f"g{index:04d}_c{cell:02d}_p{pair_index:02d}_{first_engine_side}"

        dedup_key = (cell, seed, first)
        if dedup_key in seen_dedup:
            raise ValueError(f"Duplicate job configuration generated for key {dedup_key}")
        seen_dedup.add(dedup_key)

        job = {
            'id': game_id,
            'index': index,
            'cell': cell,
            'pair_index': pair_index,
            'seed': seed,
            'first': first,
            'first_engine_side': first_engine_side,
            'decks': decks,
            'policies': policies,
            'matrix_row': row_model,
            'column': col_model,
            'replay': replay,
        }
        if output_dir is not None:
            job['output'] = str(output_dir)
        jobs.append(job)

    return jobs


def compute_matrix_summary(keys: Sequence[str], results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute 5x5 matrix summary from game results, where row is first and col is second."""
    cells_data = {}
    for r_idx, row_key in enumerate(keys):
        for c_idx, col_key in enumerate(keys):
            cell_id = r_idx * len(keys) + c_idx
            cells_data[(row_key, col_key)] = {
                'cell_id': cell_id,
                'row_model': row_key,
                'col_model': col_key,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'completed': 0,
                'uncompleted': 0,
                'total_scheduled': 0,
            }

    for res in results:
        row_key = res.get('matrix_row')
        col_key = res.get('column')
        if (row_key, col_key) not in cells_data:
            continue
        c = cells_data[(row_key, col_key)]
        c['total_scheduled'] += 1
        if not res.get('complete'):
            c['uncompleted'] += 1
            continue
        c['completed'] += 1
        winner = res.get('winner')
        first = res.get('first')
        # row_key is always the first-moving player in the match
        if (winner == 'a' and first == 'a') or (winner == 'b' and first == 'b'):
            c['wins'] += 1
        elif (winner == 'b' and first == 'a') or (winner == 'a' and first == 'b'):
            c['losses'] += 1
        else:
            c['draws'] += 1

    cell_list = []
    matrix_grid = []
    for r_idx, row_key in enumerate(keys):
        row_grid = []
        for c_idx, col_key in enumerate(keys):
            c = cells_data[(row_key, col_key)]
            completed = c['completed']
            wins = c['wins']
            losses = c['losses']
            draws = c['draws']
            uncompleted = c['uncompleted']
            win_rate = (wins / completed) if completed > 0 else None
            ci_lower, ci_upper = wilson_score_interval(wins, completed)
            if completed == 64 and uncompleted == 0:
                notes = "balanced 64-game cell"
            elif completed < 64:
                notes = "small sample; paired cluster insufficient"
            else:
                notes = "completed"

            entry = {
                'cell_id': c['cell_id'],
                'first_model': row_key,
                'second_model': col_key,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'completed': completed,
                'uncompleted': uncompleted,
                'total_scheduled': c['total_scheduled'],
                'win_rate': round(win_rate, 4) if win_rate is not None else None,
                'wilson_ci_lower': ci_lower,
                'wilson_ci_upper': ci_upper,
                'notes': notes,
            }
            cell_list.append(entry)
            row_grid.append(entry)
        matrix_grid.append(row_grid)

    return {
        'cells': cell_list,
        'grid': matrix_grid,
        'keys': list(keys),
    }


def write_matrix_csv(matrix_summary: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write matrix.csv with first as row, second as col, counts, and Wilson CI."""
    path = Path(path)
    lines = [
        "row_first_model,col_second_model,wins,draws,losses,total_completed,uncompleted,win_rate,wilson_ci_lower,wilson_ci_upper,notes"
    ]
    for c in matrix_summary['cells']:
        wr = f"{c['win_rate']:.4f}" if c['win_rate'] is not None else ""
        cil = f"{c['wilson_ci_lower']:.4f}" if c['wilson_ci_lower'] is not None else ""
        ciu = f"{c['wilson_ci_upper']:.4f}" if c['wilson_ci_upper'] is not None else ""
        lines.append(
            f"{c['first_model']},{c['second_model']},{c['wins']},{c['draws']},{c['losses']},"
            f"{c['completed']},{c['uncompleted']},{wr},{cil},{ciu},\"{c['notes']}\""
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def generate_report_markdown(
    matrix_summary: Dict[str, Any],
    eval_result: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Format markdown evaluation report with full disclaimer and 5x5 matrix table."""
    keys = matrix_summary['keys']
    lines = [
        "# Duel V2 Batched Cooperative Evaluation Report",
        "",
        "## 状态与交付说明",
        "",
        f"- `game_execution_complete`: **{str(eval_result.get('game_execution_complete', False)).lower()}**",
        f"- `concurrency_1600_full_games_complete`: **{str(eval_result.get('concurrency_1600_full_games_complete', False)).lower()}**",
        f"- `evaluation_accepted`: **{str(eval_result.get('evaluation_accepted', False)).lower()}**",
        f"- `behavior_approved`: **{str((eval_result.get('behavior') or {}).get('behavior_approved', False)).lower()}**",
        "- `quality_approved`: **false**",
        "- `training_updates`: **0**",
        "- `gpu_rule_execution`: **false**",
        "- `training_workflow_complete`: **false**",
        "",
        "> [!IMPORTANT]",
        "> 本评测仅评估冻结策略对决主矩阵，不代表训练全流程交付或质量批准。",
        "> 其余全规范项目未运行并标记为 missing：",
        "> - `historical_models_comparison`: **missing**",
        "> - `card_swaps_comparison`: **missing**",
        "> - `lineup_variations_comparison`: **missing**",
        "> - `tactical_cases_gate`: **missing**",
        "",
        "## 5×5 互搏胜率主矩阵（行：先手，列：后手）",
        "",
        "表格每格格式为：`胜-平-负 (胜局率% [参考 Wilson 95% 区间]) | 未完成数`。区间未作成对根聚类修正，不能当独立确认门槛。",
        "",
    ]

    header = "| 先手 \\ 后手 | " + " | ".join(keys) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(keys)) + " |"
    lines.append(header)
    lines.append(sep)

    grid = matrix_summary['grid']
    for r_idx, row_key in enumerate(keys):
        row_cells = [f"**{row_key}**"]
        for c_idx, col_key in enumerate(keys):
            cell = grid[r_idx][c_idx]
            if cell['completed'] > 0:
                pct = f"{cell['win_rate'] * 100:.1f}%"
                ci = f"[{cell['wilson_ci_lower'] * 100:.1f}%, {cell['wilson_ci_upper'] * 100:.1f}%]" if cell['wilson_ci_lower'] is not None else ""
                val = f"{cell['wins']}-{cell['draws']}-{cell['losses']} ({pct} {ci})"
            else:
                val = "N/A"
            if cell['uncompleted'] > 0:
                val += f" (未完成: {cell['uncompleted']})"
            row_cells.append(val)
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.extend([
        "",
        "## 运行与延迟统计",
        "",
        f"- 总计划局数: {eval_result.get('total_games', 0)}",
        f"- 完成局数: {eval_result.get('completed_games', 0)}",
        f"- 未完成局数: {eval_result.get('uncompleted_games', 0)}",
        f"- 峰值在途根数 (`peak_outstanding_roots`): {eval_result.get('peak_outstanding_roots', 0)}",
        "",
    ])

    stats = eval_result.get('stats', {})
    if stats:
        lines.extend([
            "### 搜索延迟 (秒)",
            f"- p50: {stats.get('search_latency_p50')}",
            f"- p95: {stats.get('search_latency_p95')}",
            f"- p99: {stats.get('search_latency_p99')}",
            f"- max: {stats.get('search_latency_max')}",
            "",
            "### 提交延迟 (秒)",
            f"- p50: {stats.get('commit_latency_p50')}",
            f"- p95: {stats.get('commit_latency_p95')}",
            f"- p99: {stats.get('commit_latency_p99')}",
            f"- max: {stats.get('commit_latency_max')}",
            "",
            "### 模拟与公开证明量",
            f"- 总公开证明检查: {stats.get('proof_checks_total')}",
            f"- 总树模拟次数: {stats.get('tree_simulations_total')}",
            "",
            "### 决策停止 / 回退原因分布",
            "```json",
            json.dumps(stats.get('fallback_reasons', {}), indent=2, ensure_ascii=False),
            "```",
            "",
        ])

    backend_stats = eval_result.get('backend_stats')
    if backend_stats:
        lines.extend([
            "### 后端运行时统计",
            f"- Workers: {backend_stats.get('workers')}",
            f"- 活跃进程数: {backend_stats.get('active_processes')}",
            f"- 峰值状态槽位: {backend_stats.get('peak_states')}",
            f"- 释放状态槽位: {backend_stats.get('released_states')}",
            f"- Worker 执行纳秒: {backend_stats.get('worker_execution_ns')}",
            f"- 父进程 RPC 墙钟纳秒: {backend_stats.get('parent_rpc_wall_ns')}",
            "",
        ])

    return '\n'.join(lines) + '\n'


def write_result_json(
    output_path: Union[str, Path],
    eval_result: Dict[str, Any],
    matrix_summary: Dict[str, Any],
    config: Dict[str, Any],
    source_snapshot: Dict[str, Any],
) -> None:
    """Write complete result.json with spec audit states and metrics."""
    payload = {
        'game_execution_complete': eval_result.get('game_execution_complete', False),
        'concurrency_1600_full_games_complete': eval_result.get('concurrency_1600_full_games_complete', False),
        'quality_approved': False,
        'evaluation_accepted':eval_result.get('evaluation_accepted',False),
        'training_updates': 0,
        'gpu_rule_execution': False,
        'training_workflow_complete': False,
        'full_spec_report_complete': False,
        'missing_evaluations': {
            'historical_models_comparison': 'missing',
            'card_swaps_comparison': 'missing',
            'lineup_variations_comparison': 'missing',
            'tactical_cases_gate': 'missing',
        },
        'config': config,
        'source_snapshot': source_snapshot,
        'matrix': matrix_summary['grid'],
        'cells': matrix_summary['cells'],
        'total_games': eval_result.get('total_games', 0),
        'completed_games': eval_result.get('completed_games', 0),
        'uncompleted_games': eval_result.get('uncompleted_games', 0),
        'peak_outstanding_roots': eval_result.get('peak_outstanding_roots', 0),
        'stats': eval_result.get('stats', {}),
        'backend_stats': eval_result.get('backend_stats'),
        'device_statistics':eval_result.get('device_statistics'),
        'device_memory':eval_result.get('device_memory'),
        'method_error':eval_result.get('method_error'),
        'behavior':eval_result.get('behavior'),
        'device_stats': {
            'cpu_peak_memory_bytes': None,
            'gpu_peak_memory_bytes': None,
        },
    }
    p = Path(output_path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)


def make_source_snapshot(models: Mapping[str, Any], budget_mode: str, models_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Capture runtime source SHA snapshot and verify source manifest if present."""
    from app.modules.card_game.rl import cross_lineup
    from app.modules.card_game.rl.batched_duel import process_backend, recording, serving_protocol
    from app.modules.card_game.rl.batched_search import serving, cooperative, traversal, inference, statistics, nodes, behavior

    search_files = [
        serving.__file__,
        cooperative.__file__,
        traversal.__file__,
        inference.__file__,
        statistics.__file__, nodes.__file__, behavior.__file__,
        process_backend.__file__, recording.__file__, serving_protocol.__file__,
        __file__,
    ]
    search_hasher = sha256()
    for sf in search_files:
        if not sf:
            continue
        sf_path = Path(sf)
        if sf_path.suffix == '.pyc':
            py_path = sf_path.with_suffix('.py')
            sf_path = py_path if py_path.exists() else sf_path
        if sf_path.exists():
            search_hasher.update(sf_path.read_bytes())
    search_code_sha = search_hasher.hexdigest()

    rules_hash = cross_lineup.identity()

    model_shas = {}
    manifest_shas = {}
    build_shas = {}

    for key, model in models.items():
        m_version = getattr(model, 'version', None)
        model_shas[key] = m_version
        manifest = getattr(model, 'manifest', {}) or {}
        manifest_path=Path(models_dir)/(key+'.json') if models_dir is not None else None
        manifest_shas[key] = sha256(manifest_path.read_bytes()).hexdigest() if manifest_path is not None else None
        build_shas[key] = getattr(model, 'serving_build_sha256', None) or manifest.get('build_sha256')

    snapshot = {
        'rules_hash': rules_hash,
        'search_code_sha': search_code_sha,
        'budget_mode': budget_mode,
        'model_shas': model_shas,
        'manifest_shas': manifest_shas,
        'build_shas': build_shas,
    }

    if models_dir is not None:
        source_manifest_path = Path(models_dir) / 'source_manifest.json'
        if source_manifest_path.exists():
            source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
            snapshot['source_manifest'] = source_manifest
            expected_models = source_manifest.get('models', {})
            for k, expected_m in expected_models.items():
                if k in model_shas and expected_m.get('sha256') != model_shas[k]:
                    raise ValueError(
                        f"Source manifest mismatch for model {k}: expected {expected_m.get('sha256')}, got {model_shas[k]}"
                    )

    return snapshot


def evaluate_jobs(
    jobs: Sequence[Dict[str, Any]],
    models: Mapping[str, Any],
    inference: Any,
    backend: Any,
    *,
    budget_mode: str,
    decision_seconds: float = 3.0,
    simulations: int = 32,
    proof_cap: int = 8,
    deadline: Optional[float] = None,
    max_actions: int = 800,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    statistics=None,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Execute complete real-game cooperative evaluation across resident worker states.

    Import up to 1600 games once into backend workers. In each round, construct
    RootJob instances using serving_search for all active games, batch resolve via
    CooperativeScheduler, and apply single-operation physical commits on backend.
    """
    if budget_mode not in ('simulations', 'wall_clock'):
        raise ValueError(f"Explicit serving budget mode ('simulations' or 'wall_clock') required, got {budget_mode!r}")
    if type(simulations) is not int or simulations < 1:
        raise ValueError(f"simulations must be a positive integer, got {simulations!r}")
    if type(proof_cap) is not int or not 0 <= proof_cap < simulations:
        raise ValueError(f"proof_cap must be between 0 and {simulations - 1}, got {proof_cap!r}")
    if isinstance(decision_seconds, bool) or not isinstance(decision_seconds, (float, int)) or not math.isfinite(decision_seconds) or decision_seconds <= 0:
        raise ValueError(f"decision_seconds must be a positive finite number, got {decision_seconds!r}")
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError(f"max_actions must be a positive integer, got {max_actions!r}")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError(f"deadline must be a finite monotonic timestamp, got {deadline!r}")

    jobs = list(jobs)
    if not jobs or len(jobs) > 1600:
        raise ValueError(f"Non-empty sequence of at most 1600 jobs required, got {len(jobs)}")

    # Strict game id and configuration deduplication check
    seen_ids = set()
    seen_dedup = set()
    for j in jobs:
        gid = j.get('id')
        if not isinstance(gid, str) or not gid:
            raise ValueError(f"Job id must be a non-empty string, got {gid!r}")
        if gid in seen_ids:
            raise ValueError(f"Duplicate game id detected: {gid}")
        seen_ids.add(gid)

        cell = j.get('cell', j.get('matrix_row', 0))
        seed = j.get('seed')
        first = j.get('first')
        if first not in ('a', 'b'):
            raise ValueError(f"Job first must be 'a' or 'b', got {first!r}")
        if type(seed) is not int:
            raise TypeError(f"Job seed must be an integer, got {seed!r}")

        dedup_key = (cell, seed, first)
        if dedup_key in seen_dedup:
            raise ValueError(f"Duplicate job configuration: cell={cell}, seed={seed}, seat={first}")
        seen_dedup.add(dedup_key)

    # Resolve output directory
    if output_dir is None:
        output_dir = getattr(backend, 'record_output', None)
    if output_dir is None and jobs:
        output_dir = jobs[0].get('output')
    out_path = Path(output_dir).resolve() if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / 'raw').mkdir(parents=True, exist_ok=True)
        (out_path / 'replays').mkdir(parents=True, exist_ok=True)

    # Pre-reserve 30 seconds before global deadline for disk persistence
    search_deadline = (deadline - 30.0) if deadline is not None else None

    # Initialize all games in Python before importing once into worker slots
    states = []
    for j in jobs:
        decks = j.get('decks')
        if not isinstance(decks, dict) or set(decks.keys()) != {'a', 'b'}:
            raise ValueError(f"Job {j['id']} requires both 'a' and 'b' decks")
        st = new_game(seed=j['seed'], decks=decks, first_side=j['first'])
        states.append(st)

    game_ids = [j['id'] for j in jobs]
    root_generations = [1] * len(jobs)
    initial_refs = backend.import_states(states, game_ids, root_generations)

    active_games: Dict[str, Tuple[Dict[str, Any], Any]] = {
        ref.game_id: (job, ref) for job, ref in zip(jobs, initial_refs)
    }
    action_counts: Dict[str, int] = {j['id']: 0 for j in jobs}
    game_job_map: Dict[str, Dict[str, Any]] = {j['id']: j for j in jobs}

    finished_results: Dict[str, Dict[str, Any]] = {}
    uncompleted_results: Dict[str, Dict[str, Any]] = {}
    persisted_game_ids: set[str] = set()

    search_latencies: List[float] = []
    search_batch_latencies: List[float] = []
    commit_latencies: List[float] = []
    proof_checks_list: List[int] = []
    tree_simulations_list: List[int] = []
    total_simulations_list: List[int] = []
    fallback_reasons: collections.Counter[str] = collections.Counter()
    planner_outcomes: collections.Counter[str] = collections.Counter()
    peak_outstanding_roots = 0

    start_monotonic = clock()

    def _flush_records(batch_results: Sequence[Dict[str, Any]]) -> None:
        if out_path is None:
            behavior_gate.observe(batch_results)
            return
        fresh=[]
        index_file = out_path / 'index.jsonl'
        with index_file.open('a', encoding='utf-8') as f:
            for res in batch_results:
                gid = res.get('id')
                if gid not in persisted_game_ids:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
                    persisted_game_ids.add(gid)
                    fresh.append(res)

        behavior_gate.observe(fresh)

    def _update_progress(phase: str = 'evaluating') -> None:
        prog = {
            'phase': phase,
            'total_games': len(jobs),
            'completed_games': sum(1 for r in finished_results.values() if r.get('complete')),
            'uncompleted_games': len(uncompleted_results) + sum(1 for r in finished_results.values() if not r.get('complete')),
            'in_progress_games': len(active_games),
            'peak_outstanding_roots': peak_outstanding_roots,
            'persisted_games': len(persisted_game_ids),
            'physical_actions':sum(action_counts.values()),
            'max_actions_taken':max(action_counts.values(),default=0),
            'elapsed_seconds': clock() - start_monotonic,
            'deadline': deadline,
            'last_updated': clock(),
            'behavior_approved':behavior_gate.report()['behavior_approved'],
            'inactive_games':behavior_gate.report()['inactive_games'],
        }
        if out_path is not None:
            prog_file = out_path / 'progress.json'
            tmp_file = prog_file.with_suffix('.tmp')
            tmp_file.write_text(json.dumps(prog, indent=2, ensure_ascii=False), encoding='utf-8')
            tmp_file.replace(prog_file)
        if progress_callback is not None:
            progress_callback(prog)

    from .behavior import BehaviorGate
    behavior_gate=BehaviorGate()
    method_error=None
    try:
        while active_games:
            # 1. Harvest any games whose resident state reached 'finished'
            ready_finish_refs = []
            ready_finish_jobs = []
            for gid, (job, ref) in list(active_games.items()):
                if ref.phase == 'finished':
                    ready_finish_refs.append(ref)
                    ready_finish_jobs.append(job)
                    del active_games[gid]

            if ready_finish_refs:
                summaries = backend.finish_games(ready_finish_refs, ready_finish_jobs)
                backend.release(ready_finish_refs)
                for res in summaries:
                    finished_results[res['id']] = res
                _flush_records(summaries)
                _update_progress()

            if not active_games:
                break

            # 2. Check global search deadline (reserving 30s for disk output)
            time_now = clock()
            if search_deadline is not None and time_now >= search_deadline:
                break

            # 3. Check for games reaching max_actions limit
            exceeded_gids = {
                gid for gid, count in action_counts.items()
                if gid in active_games and count >= max_actions
            }
            if len(exceeded_gids) == len(active_games):
                # All active games exhausted action budget
                break

            # 4. Construct RootJob for each active unexhausted game
            games_to_search = []
            root_jobs = []
            queued_at = clock()

            for gid, (job, ref) in active_games.items():
                if action_counts[gid] >= max_actions:
                    continue

                actor = ref.actor
                phase = ref.phase
                # P0 sanity check on worker metadata
                if phase != 'finished' and actor not in ('a', 'b'):
                    raise RuntimeError(f"P0: Abnormal actor {actor!r} or phase {phase!r} on game {gid}")
                if not phase:
                    raise RuntimeError(f"P0: Missing phase on game {gid}")

                policy_spec = job['policies'][actor]
                if policy_spec in models:
                    model = models[policy_spec]
                elif isinstance(policy_spec, PolicyIdentity) or hasattr(policy_spec, 'version'):
                    model = policy_spec
                elif isinstance(policy_spec, tuple) and len(policy_spec) == 2:
                    model = policy_spec
                else:
                    model = models.get(policy_spec)
                    if model is None:
                        raise ValueError(f"Unknown policy key {policy_spec!r} for actor {actor}")

                root_token = f"{ref.game_id}:r{ref.root_generation}:{uuid4().hex[:8]}"

                gen = serving_search(
                    ref,
                    actor,
                    model,
                    backend=backend,
                    budget_mode=budget_mode,
                    seconds=decision_seconds,
                    simulations=simulations,
                    proof_cap=proof_cap,
                    deadline=search_deadline,
                    root_id=root_token,
                    queued_at=queued_at,
                    statistics=statistics,
                    clock=clock,
                )

                root_job = RootJob(
                    root_id=root_token,
                    game_id=ref.game_id,
                    generation=ref.root_generation,
                    generator=gen,
                    deadline=(min(queued_at+decision_seconds,search_deadline) if search_deadline is not None else queued_at+decision_seconds)
                        if budget_mode=='wall_clock' else search_deadline,
                )
                root_jobs.append(root_job)
                games_to_search.append((gid, job, ref, actor, root_token))

            if not root_jobs:
                break

            peak_outstanding_roots = max(peak_outstanding_roots, len(root_jobs))

            # 5. Batch resolve all cooperative search operations
            search_t0 = clock()
            scheduler = CooperativeScheduler(inference, max_roots=len(root_jobs), allow_cpu_oracle=False)
            sched_output = scheduler.run(root_jobs,cancelled=(
                (lambda:clock()>=search_deadline) if search_deadline is not None else None))
            search_dur = clock() - search_t0
            search_batch_latencies.append(search_dur)
            peak_outstanding_roots = max(peak_outstanding_roots, sched_output.get('peak_outstanding_roots', len(root_jobs)))

            results_by_root = sched_output['results']

            # 6. Validate search decisions and prepare batch physical commits
            commits = []
            old_refs_to_release = []
            policy_timeouts=[]

            for gid, job, ref, actor, root_token in games_to_search:
                search_res = results_by_root.get(root_token)
                if search_res is None:
                    raise RuntimeError(f"Scheduler dropped result for root {root_token}")

                selected_action = search_res.get('selected_action')
                actions = search_res.get('actions', [])
                p_checks = search_res.get('proof_checks', 0)
                t_sims = search_res.get('tree_simulations', 0)
                tot_sims = search_res.get('total_simulations', 0)
                stop_reason = search_res.get('stop_reason')
                used_fallback = search_res.get('used_fallback', False)

                if type(tot_sims) is not int or tot_sims < 0 or tot_sims > simulations:
                    raise ValueError(f"Invalid total_simulations {tot_sims} (must be 0..{simulations})")
                if type(p_checks) is not int or p_checks < 0:
                    raise ValueError(f"Invalid proof_checks {p_checks}")
                if type(t_sims) is not int or t_sims < 0:
                    raise ValueError(f"Invalid tree_simulations {t_sims}")
                if p_checks>proof_cap or p_checks+t_sims!=tot_sims:
                    raise ValueError('Public proofs and tree simulations exceeded their shared budget')
                latency=search_res.get('wall_seconds')
                if isinstance(latency,bool) or not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<0:
                    raise ValueError('Finite actual per-root search latency required')
                search_latencies.append(latency)

                proof_checks_list.append(p_checks)
                tree_simulations_list.append(t_sims)
                total_simulations_list.append(tot_sims)
                if stop_reason:planner_outcomes[stop_reason]+=1
                if stop_reason and used_fallback:
                    fallback_reasons[stop_reason] += 1

                if selected_action is None:
                    # A root never obtains a fresh budget for the same physical
                    # decision after missing its 3-second policy window.
                    policy_timeouts.append((gid,job,ref))
                    continue

                if selected_action not in actions:
                    raise ValueError(f"P0: Selected action {selected_action} not in legal actions {actions}")

                compact_stats = {
                    'simulations': tot_sims,
                    'tree_simulations': t_sims,
                    'proof_checks': p_checks,
                    'proof_attempts':search_res.get('proof_attempts'),
                    'used_fallback': used_fallback,
                    'stop_reason': stop_reason,
                    'tree_stop_reason':search_res.get('tree_stop_reason'),
                    'deadline_reached':search_res.get('deadline_reached'),
                    'wall_seconds': search_res.get('wall_seconds'),
                    'surrogate_model_key': search_res.get('surrogate_model_key'),
                    'surrogate_model_version': search_res.get('surrogate_model_version'),
                }

                commits.append((ref, actor, selected_action, ref.root_generation + 1, compact_stats))
                old_refs_to_release.append(ref)

            # 7. Apply physical commits before deadline cutoff
            if search_deadline is not None and clock() >= search_deadline:
                # Do not commit past deadline
                break

            if policy_timeouts:
                timeout_refs=[ref for _,_,ref in policy_timeouts]
                timeout_jobs=[dict(job,completion_reason='root_deadline_without_policy')
                              for _,job,_ in policy_timeouts]
                summaries=backend.finish_games(timeout_refs,timeout_jobs)
                backend.release(timeout_refs)
                for row in summaries:uncompleted_results[row['id']]=row
                _flush_records(summaries)
                for gid,_,_ in policy_timeouts:active_games.pop(gid,None)

            if commits:
                commit_t0 = clock()
                new_refs = backend.commit_actions(commits)
                commit_dur = clock() - commit_t0
                commit_latencies.append(commit_dur)

                for old_ref, new_ref in zip(old_refs_to_release, new_refs):
                    gid = old_ref.game_id
                    job = game_job_map[gid]
                    active_games[gid] = (job, new_ref)
                    action_counts[gid] += 1

                backend.release(old_refs_to_release)

            _update_progress()

    except Exception as exc:
        method_error=repr(exc)
    finally:
        # Wrap up any remaining active games (due to deadline cutoff or max_actions)
        if active_games:
            uncompleted_refs = [ref for _, ref in active_games.values()]
            uncompleted_jobs = []
            for job,ref in active_games.values():
                reason=('terminal' if ref.phase=='finished' else 'method_error' if method_error is not None
                        else 'max_actions' if action_counts[job['id']]>=max_actions
                        else 'global_search_deadline' if search_deadline is not None and clock()>=search_deadline
                        else 'evaluation_stopped')
                uncompleted_jobs.append(dict(job,completion_reason=reason))
            try:
                # Attempt to finalize partial records via backend if still healthy
                uncompleted_summaries = backend.finish_games(uncompleted_refs, uncompleted_jobs)
                backend.release(uncompleted_refs)
                for res in uncompleted_summaries:
                    uncompleted_results[res['id']] = res
                _flush_records(uncompleted_summaries)
            except Exception as exc:
                # Backend poisoned or failed; record uncompleted error state without pretending finish succeeded
                fallback_batch = []
                for job, ref in active_games.values():
                    fallback_res = {
                        **job,
                        'complete': False,
                        'replay_verified': False,
                        'error': f'Unrecovered backend state: {exc!r}',
                        'completion_reason':'backend_unrecoverable',
                        'winner': None,
                        'turn': getattr(ref, 'turn', None),
                    }
                    uncompleted_results[job['id']] = fallback_res
                    fallback_batch.append(fallback_res)
                _flush_records(fallback_batch)
            active_games.clear()

        _update_progress(phase='failed' if method_error is not None else
                         'finished' if behavior_gate.report()['behavior_approved'] else 'rejected')

    all_results = []
    for j in jobs:
        gid = j['id']
        if gid in finished_results:
            all_results.append(finished_results[gid])
        elif gid in uncompleted_results:
            all_results.append(uncompleted_results[gid])
        else:
            all_results.append({
                **j,
                'complete': False,
                'replay_verified': False,
                'error': 'Missing game result record',
            })

    completed_count = sum(1 for r in all_results if r.get('complete') is True)
    uncompleted_count = len(all_results) - completed_count
    game_execution_complete = (method_error is None and completed_count == len(jobs) and all(r.get('replay_verified') is True for r in all_results))
    concurrency_1600_full_games_complete = (len(jobs) == 1600 and game_execution_complete)

    def _percentile(data: List[Union[int, float]], p: float) -> Optional[float]:
        if not data:
            return None
        return round(float(np.percentile(data, p)), 6)

    stats = {
        'search_latency_p50': _percentile(search_latencies, 50),
        'search_latency_p95': _percentile(search_latencies, 95),
        'search_latency_p99': _percentile(search_latencies, 99),
        'search_latency_max': round(float(max(search_latencies)), 6) if search_latencies else None,
        'commit_latency_p50': _percentile(commit_latencies, 50),
        'commit_latency_p95': _percentile(commit_latencies, 95),
        'commit_latency_p99': _percentile(commit_latencies, 99),
        'commit_latency_max': round(float(max(commit_latencies)), 6) if commit_latencies else None,
        'total_wall_seconds': round(clock() - start_monotonic, 4),
        'peak_outstanding_roots': peak_outstanding_roots,
        'proof_checks_total': sum(proof_checks_list),
        'proof_checks_p50': _percentile(proof_checks_list, 50),
        'proof_checks_max': max(proof_checks_list) if proof_checks_list else 0,
        'tree_simulations_total': sum(tree_simulations_list),
        'tree_simulations_p50': _percentile(tree_simulations_list, 50),
        'tree_simulations_max': max(tree_simulations_list) if tree_simulations_list else 0,
        'fallback_reasons': dict(fallback_reasons),
        'planner_outcomes':dict(planner_outcomes),
        'search_batch_latency_p50':_percentile(search_batch_latencies,50),
        'search_batch_latency_p95':_percentile(search_batch_latencies,95),
    }

    try:backend_stats = backend.stats() if hasattr(backend, 'stats') else None
    except RuntimeError:backend_stats=None

    try:device_stats=statistics.stats() if statistics is not None else None
    except RuntimeError:device_stats=None

    return {
        'results': all_results,
        'total_games': len(jobs),
        'completed_games': completed_count,
        'uncompleted_games': uncompleted_count,
        'game_execution_complete': game_execution_complete,
        'concurrency_1600_full_games_complete': concurrency_1600_full_games_complete,
        'quality_approved': False,
        'training_updates': 0,
        'gpu_rule_execution': False,
        'training_workflow_complete': False,
        'peak_outstanding_roots': peak_outstanding_roots,
        'stats': stats,
        'backend_stats': backend_stats,
        'device_statistics':device_stats,
        'method_error':method_error,
        'behavior':behavior_gate.report(),
        'evaluation_accepted':game_execution_complete and behavior_gate.report()['behavior_approved'],
    }
