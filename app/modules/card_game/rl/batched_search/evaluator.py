"""Cooperative real-game evaluation orchestrator for Duel V2 batched search.

Drives batches of up to 1,600 resident real games using official Duel V2 rules
on worker processes, cooperative Gumbel serving search, and batch neural inference.
Default rule execution stays resident CPU. GPU rule execution is opt-in via
an explicit cuda_vm backend and is independent of inference --device. Missing
CUDA, unsupported entries, or runtime/schema mismatch fail closed with no CPU
fallback. Device verification stays false until actual device evidence.
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



RULE_BACKEND_RESIDENT_CPU = 'resident_cpu'
RULE_BACKEND_CUDA_VM = 'cuda_vm'
RUNTIME_RECOVERY = 'recovery'
RUNTIME_SEVEN = 'seven'
RECOVERY_MATRIX_KEYS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
MATRIX_ORDER = 'row_first_col_second'
COMPLETED_DENOMINATOR = 'completed_normal_games'


def runtime_matrix_keys(runtime_name: str) -> Tuple[str, ...]:
    if runtime_name == RUNTIME_RECOVERY:
        return RECOVERY_MATRIX_KEYS
    if runtime_name == RUNTIME_SEVEN:
        from app.modules.card_game.rl.seven_lineup import KEYS
        return tuple(KEYS)
    raise ValueError(f'Unsupported runtime {runtime_name!r}; expected recovery or seven')


def resolve_runtime_module(runtime_name: str):
    if runtime_name == RUNTIME_RECOVERY:
        from app.modules.card_game.rl import recovery_runtime
        return recovery_runtime
    if runtime_name == RUNTIME_SEVEN:
        from app.modules.card_game.rl import seven_runtime
        return seven_runtime
    raise ValueError(f'Unsupported runtime {runtime_name!r}; expected recovery or seven')


def report_rule_backend(backend: Any, explicit: Optional[str] = None) -> str:
    kind = explicit or getattr(backend, 'backend_kind', None) or RULE_BACKEND_RESIDENT_CPU
    if kind in (RULE_BACKEND_RESIDENT_CPU, 'python_resident_process'):
        return RULE_BACKEND_RESIDENT_CPU
    if kind == RULE_BACKEND_CUDA_VM:
        return RULE_BACKEND_CUDA_VM
    raise ValueError(f'Unknown rule backend {kind!r}')


def _require_kwarg(func: Any, name: str, label: str) -> None:
    import inspect
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'{label} is not inspectable: {exc}. No CPU fallback.') from exc
    if name not in signature.parameters:
        raise RuntimeError(
            f'{label} does not accept {name}=; integration is pending. '
            'No CPU fallback and no source-only ready gate.'
        )


def _resolve_capacity_planner(capacity_planner: Any = None):
    if capacity_planner is not None:
        return capacity_planner
    try:
        from app.modules.card_game.rl.batched_duel.gpu_rule_backend import plan_gpu_rule_capacity
        return plan_gpu_rule_capacity
    except ImportError:
        pass
    try:
        from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import plan_gpu_rule_capacity
        return plan_gpu_rule_capacity
    except ImportError as exc:
        raise RuntimeError(
            'plan_gpu_rule_capacity is not available; B-side capacity planner integration is pending. '
            'No CPU fallback.'
        ) from exc


def _load_gpu_rule_program(program_api: Any, path: Any, runtime: str):
    load = getattr(program_api, 'load_program', None)
    if load is None:
        raise RuntimeError('gp.load_program is missing; A-side program API integration is pending. No CPU fallback.')
    _require_kwarg(load, 'runtime', 'gp.load_program')
    return load(path, runtime=runtime)


def _capture_or_build_gpu_rule_program(program_api: Any, runtime: str):
    capture_available = getattr(program_api, 'source_capture_available', lambda: False)
    if not capture_available():
        raise RuntimeError(
            'Frozen GPU rule program JSON is required when CPython 3.10 source capture is unavailable. '
            'No CPU fallback.'
        )
    if hasattr(program_api, 'build_gpu_rule_program'):
        build = program_api.build_gpu_rule_program
        _require_kwarg(build, 'runtime', 'gp.build_gpu_rule_program')
        program, status = build(runtime=runtime)
        return program, status
    capture = getattr(program_api, 'capture_gpu_rule_program', None)
    if capture is None:
        raise RuntimeError('gp.capture/build are missing; A-side program API integration is pending. No CPU fallback.')
    _require_kwarg(capture, 'runtime', 'gp.capture_gpu_rule_program')
    return capture(runtime=runtime), {'source_capture': True, 'device_verified': False}


def statistics_capacity(rule_backend, search_roots, capacity_plan=None):
    """Bind actual statistics allocation to the same limits used in the plan."""
    if type(search_roots) is not int or not 1 <= search_roots <= 1600:
        raise ValueError('search_roots must be an integer in 1..1600')
    if rule_backend == RULE_BACKEND_RESIDENT_CPU:
        return 33 * search_roots, 2048
    if rule_backend != RULE_BACKEND_CUDA_VM:
        raise ValueError('Unknown rule backend')
    from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import validate_capacity_plan
    validate_capacity_plan(capacity_plan)
    node_capacity = capacity_plan['node_capacity']
    candidate_capacity = capacity_plan['candidate_capacity']
    if type(node_capacity) is not int or node_capacity < 1:
        raise ValueError('Positive planned node capacity required')
    if type(candidate_capacity) is not int or candidate_capacity < 1:
        raise ValueError('Positive planned candidate capacity required')
    return node_capacity, candidate_capacity


def select_rule_backend(
    *,
    rule_backend: str,
    runtime: str,
    workers: int,
    max_games: int,
    search_roots: int,
    record_output: Any = None,
    gpu_rule_program: Any = None,
    guard_standalone_entities: bool = False,
    gpu_memory_budget_mib: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_payload_bytes: Optional[int] = None,
    max_candidates: Optional[int] = None,
    resident_backend_cls: Any = None,
    gpu_backend_cls: Any = None,
    capacity_planner: Any = None,
    program_api: Any = None,
    fake_label: Optional[str] = None,
    query_batch_rows: Optional[int] = None,
) -> Any:
    """Construct the explicit rule backend. cuda_vm never falls back to resident CPU."""
    if rule_backend not in (RULE_BACKEND_RESIDENT_CPU, RULE_BACKEND_CUDA_VM):
        raise ValueError(f'Unsupported rule backend {rule_backend!r}')
    if runtime not in (RUNTIME_RECOVERY, RUNTIME_SEVEN):
        raise ValueError(f'Unsupported runtime {runtime!r}')
    runtime_module = resolve_runtime_module(runtime)
    if rule_backend == RULE_BACKEND_RESIDENT_CPU:
        if fake_label:
            raise ValueError('fake_label is only valid for labeled GPU constructor tests')
        cls = resident_backend_cls
        if cls is None:
            from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
            cls = ResidentRuleBackend
        backend = cls(
            workers=workers,
            max_games=max_games,
            runtime=runtime_module,
            record_output=record_output,
        )
        backend.requested_search_roots = search_roots
        return backend

    if fake_label is not None:
        label = str(fake_label).lower()
        if 'fake' not in label and 'synthetic' not in label:
            raise ValueError('GPU constructor fakes must be labeled fake or synthetic; no real CUDA claim')
        if gpu_backend_cls is None:
            raise ValueError('Labeled GPU fakes require an explicit gpu_backend_cls')

    cls = gpu_backend_cls
    if cls is None:
        from app.modules.card_game.rl.batched_duel.gpu_rule_backend import GpuRuleBackend
        cls = GpuRuleBackend
    api = program_api
    if api is None:
        from app.modules.card_game.rl.batched_duel import gpu_rule_program as api
    if gpu_rule_program is not None:
        program = _load_gpu_rule_program(api, gpu_rule_program, runtime)
        program_status = {'loaded': True, 'device_verified': False}
    else:
        program, program_status = _capture_or_build_gpu_rule_program(api, runtime)
    planner = _resolve_capacity_planner(capacity_planner)
    _require_kwarg(planner, 'runtime', 'plan_gpu_rule_capacity')
    memory_budget_bytes = None if gpu_memory_budget_mib is None else int(gpu_memory_budget_mib) * 1024 * 1024
    plan_options = dict(
        memory_budget_bytes=memory_budget_bytes, max_nodes=max_nodes, max_edges=max_edges,
        max_payload_bytes=max_payload_bytes, max_candidates=max_candidates, runtime=runtime,
    )
    if query_batch_rows is not None:
        if type(query_batch_rows) is not int or not 1 <= query_batch_rows <= 32:
            raise ValueError('Bounded GPU query batch required')
        plan_options['query_batch_rows'] = query_batch_rows
    plan = planner(
        max_games,
        search_roots,
        **plan_options,
    )
    _require_kwarg(cls.__init__, 'runtime', 'GpuRuleBackend')
    _require_kwarg(cls.__init__, 'capacity_plan', 'GpuRuleBackend')
    backend = cls(
        program,
        runtime=runtime,
        capacity_plan=plan,
        guard_standalone_entities=bool(guard_standalone_entities),
        record_output=record_output,
        max_games=max_games,
    )
    backend.requested_search_roots = search_roots
    backend.capacity_plan = getattr(backend, 'capacity_plan', plan)
    backend.program_status = program_status

    if getattr(backend, 'runtime_name', None) not in (RUNTIME_RECOVERY, RUNTIME_SEVEN):
        backend.runtime_name = runtime
    if fake_label is not None:
        backend.fake_label = fake_label
        if getattr(backend, 'device_verified', False):
            raise RuntimeError('Labeled GPU fakes cannot claim device_verified')
    return backend


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

    team_count = len(keys)
    if team_count not in (5, 7):
        raise ValueError(f"Matrix keys must be the five-team or seven-team roster, got {team_count}")
    if team_count == 7:
        from app.modules.card_game.rl.seven_lineup import KEYS as SEVEN_KEYS
        if set(keys) != set(SEVEN_KEYS):
            raise ValueError(f'Seven-team matrix keys must match seven_lineup.KEYS, got {tuple(keys)}')
    if type(seed_base) is not int:
        raise TypeError("seed_base must be an integer")

    jobs = []
    seen_dedup = set()
    cell_count = team_count * team_count
    for index in range(total_games):
        cell = index % cell_count
        row_idx = cell // team_count
        col_idx = cell % team_count
        row_model = keys[row_idx]
        col_model = keys[col_idx]

        pair_index = index // cell_count
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

        'matrix_order': MATRIX_ORDER,
        'denominator': COMPLETED_DENOMINATOR,
        'team_count': len(keys),
        'partial_cells': sum(1 for cell in cell_list if cell['uncompleted'] > 0 or cell['completed'] == 0),
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
        f"- `resident_1600_full_games_complete`: **{str(eval_result.get('resident_1600_full_games_complete', False)).lower()}**",
        f"- `fixed_budget_complete`: **{str(eval_result.get('fixed_budget_complete', False)).lower()}**",
        f"- 同时准入搜索上限：**{eval_result.get('search_root_limit', 'unknown')}**；驻留对局数：**{eval_result.get('total_games', 0)}**",
        "- 离线3秒窗口从搜索准入计时；驻留等待及动作端到端延迟另报，不能等同网站3秒响应。",
        f"- `evaluation_accepted`: **{str(eval_result.get('evaluation_accepted', False)).lower()}**",
        f"- `behavior_approved`: **{str((eval_result.get('behavior') or {}).get('behavior_approved', False)).lower()}**",
        "- `quality_approved`: **false**",
        "- `training_updates`: **0**",
        f"- `rule_backend`: **{eval_result.get('rule_backend', config.get('rule_backend', RULE_BACKEND_RESIDENT_CPU))}**",
        f"- `runtime`: **{eval_result.get('runtime', config.get('runtime', RUNTIME_RECOVERY))}**",
        f"- `gpu_rule_execution`: **{str(eval_result.get('gpu_rule_execution', False)).lower()}**",
        f"- `device_verified`: **{str(eval_result.get('device_verified', False)).lower()}**",
        f"- requested search roots: **{eval_result.get('requested_search_roots', eval_result.get('search_root_limit', config.get('search_roots', 'unknown')))}**",
        f"- observed search-root peak: **{eval_result.get('observed_search_root_peak', eval_result.get('peak_outstanding_roots', 0))}**",
        f"- denominator: **{eval_result.get('denominator', COMPLETED_DENOMINATOR)}**; uncompleted/partial games are not mixed into the normal-cell rate",
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
        f"## {len(keys)}×{len(keys)} 互搏胜率主矩阵（行：先手，列：后手）",
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
        'gpu_rule_execution': bool(eval_result.get('gpu_rule_execution', False)),
        'rule_backend': eval_result.get('rule_backend', config.get('rule_backend', RULE_BACKEND_RESIDENT_CPU)),
        'runtime': eval_result.get('runtime', config.get('runtime', RUNTIME_RECOVERY)),
        'device_verified': bool(eval_result.get('device_verified', False)),
        'requested_search_roots': eval_result.get('requested_search_roots', config.get('search_roots')),
        'observed_search_root_peak': eval_result.get('observed_search_root_peak', eval_result.get('peak_outstanding_roots', 0)),
        'capacity_plan': eval_result.get('capacity_plan', config.get('capacity_plan')),
        'gpu_memory_budget_mib': eval_result.get('gpu_memory_budget_mib', config.get('gpu_memory_budget_mib')),
        'denominator': eval_result.get('denominator', COMPLETED_DENOMINATOR),
        'matrix_order': matrix_summary.get('matrix_order', MATRIX_ORDER),
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


def make_source_snapshot(
    models: Mapping[str, Any],
    budget_mode: str,
    models_dir: Optional[Union[str, Path]] = None,
    *,
    rule_backend: str = RULE_BACKEND_RESIDENT_CPU,
    runtime: str = RUNTIME_RECOVERY,
    search_roots: Optional[int] = None,
    games: Optional[int] = None,
    capacity_plan: Any = None,
    gpu_program_identity: Any = None,
    gpu_memory_budget_mib: Optional[int] = None,
    guard_standalone_entities: Optional[bool] = None,
    device_verified: bool = False,
) -> Dict[str, Any]:
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
    search_files.append(str(Path(__file__).with_name('training_rollout.py')))
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

        'rule_backend': rule_backend,
        'runtime': runtime,
        'requested_search_roots': search_roots,
        'requested_games': games,
        'capacity_plan': capacity_plan,
        'gpu_program_identity': gpu_program_identity,
        'gpu_memory_budget_mib': gpu_memory_budget_mib,
        'guard_standalone_entities': guard_standalone_entities,
        'device_verified': bool(device_verified),
        'concurrency_1600_ready': False,
        'matrix_order': MATRIX_ORDER,
        'denominator': COMPLETED_DENOMINATOR,
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
    search_roots: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    statistics=None,
    clock: Callable[[], float] = time.monotonic,

    rule_backend: Optional[str] = None,
    runtime_name: Optional[str] = None,
    capacity_plan: Any = None,
    gpu_memory_budget_mib: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete real-game cooperative evaluation across resident worker states.

    Import up to 1600 games once into backend workers. In each round, construct
    RootJob instances are admitted into a bounded rolling search window. Finished
    roots commit immediately, before waiting for slower roots or refilling slots.
    """
    evaluation_started = clock()
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

    if search_roots is None:
        search_roots = min(len(jobs), 2 * getattr(backend, 'workers', 16))
    if type(search_roots) is not int or not 1 <= search_roots <= 1600:
        raise ValueError('search_roots must be an integer in 1..1600')
    search_roots = min(search_roots, len(jobs))

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
    del states  # Workers own the immutable initial states and replay journals now.

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
    queue_latencies = []
    decision_latencies = []
    budget_failures = collections.Counter()
    committed_budget_complete = 0
    committed_decisions = 0
    scheduler_stats = None

    resident_setup_seconds = clock() - evaluation_started
    start_monotonic = evaluation_started

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
    last_progress_write = float('-inf')
    # Resident games waiting for admission do not own a search deadline. This
    # offline protocol reports their wait separately from the admitted 3s window.
    ready = collections.deque((gid, clock()) for gid in active_games)
    searching = {}

    def retire(refs, job_rows):
        if not refs:
            return
        summaries = backend.finish_games(refs, job_rows)
        backend.release(refs)
        for row in summaries:
            target = finished_results if row.get('complete') else uncompleted_results
            target[row['id']] = row
            active_games.pop(row['id'], None)
        _flush_records(summaries)

    def admit():
        nonlocal peak_outstanding_roots
        while ready:
            gid, ready_at = ready.popleft()
            job, ref = active_games[gid]
            if ref.phase == 'finished':
                retire([ref], [job])
                continue
            if action_counts[gid] >= max_actions:
                retire([ref], [dict(job, completion_reason='max_actions')])
                continue
            actor = ref.actor
            if not ref.phase or actor not in ('a', 'b'):
                raise RuntimeError(f'P0: Abnormal resident metadata for game {gid}')
            policy_spec = job['policies'][actor]
            if isinstance(policy_spec, str):
                model = models[policy_spec]
            elif isinstance(policy_spec, PolicyIdentity) or hasattr(policy_spec, 'version') or isinstance(policy_spec, tuple) and len(policy_spec) == 2:
                model = policy_spec
            else:
                raise ValueError(f'Unknown policy key {policy_spec!r}')
            admitted_at = clock()
            queue_seconds = admitted_at - ready_at
            queue_latencies.append(queue_seconds)
            root_token = f'{ref.game_id}:r{ref.root_generation}:{uuid4().hex[:8]}'
            searching[root_token] = (gid, job, ref, actor, ready_at, queue_seconds)
            peak_outstanding_roots = max(peak_outstanding_roots, len(searching))
            gen = serving_search(ref, actor, model, backend=backend,
                budget_mode=budget_mode, seconds=decision_seconds,
                simulations=simulations, proof_cap=proof_cap, deadline=search_deadline,
                root_id=root_token, queued_at=admitted_at, statistics=statistics, clock=clock)
            root_deadline = search_deadline
            if budget_mode == 'wall_clock':
                root_deadline = min(admitted_at + decision_seconds, search_deadline) if search_deadline is not None else admitted_at + decision_seconds
            return RootJob(root_token, ref.game_id, ref.root_generation, gen, root_deadline)
        return None

    def completed(rows):
        nonlocal committed_budget_complete, committed_decisions, last_progress_write
        commits = []; old_refs = []; rejected_refs = []; rejected_jobs = []
        commit_context = []
        for root, search_res in rows:
            gid, job, ref, actor, ready_at, queue_seconds = searching.pop(root.root_id)
            p_checks = search_res.get('proof_checks', 0)
            t_sims = search_res.get('tree_simulations', 0)
            tot_sims = search_res.get('total_simulations', 0)
            if any(type(n) is not int or n < 0 for n in (p_checks, t_sims, tot_sims)) or p_checks > proof_cap or p_checks + t_sims != tot_sims or tot_sims > simulations:
                raise ValueError('Invalid completed search work accounting')
            latency = search_res.get('wall_seconds', max(0., clock() - ready_at - queue_seconds))
            if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
                raise ValueError('Finite actual per-root search latency required')
            search_latencies.append(latency)
            proof_checks_list.append(p_checks); tree_simulations_list.append(t_sims); total_simulations_list.append(tot_sims)
            reason = search_res.get('stop_reason')
            if reason: planner_outcomes[reason] += 1
            if reason and search_res.get('used_fallback'): fallback_reasons[reason] += 1
            selected = search_res.get('selected_action')
            budget_complete = (
                reason == 'forced_action' and len(search_res.get('actions', [])) == 1
                or reason == 'public_terminal_win' and p_checks > 0
                or reason == 'tree_search' and search_res.get('tree_complete') is True
                   and tot_sims == simulations
            )
            if search_res.get('budget_complete', False) != budget_complete:
                raise ValueError('Search budget declaration disagrees with completed work')
            reject_reason = None
            if selected is None:
                reject_reason = 'root_deadline_without_policy'
            elif budget_mode == 'simulations' and not budget_complete:
                reject_reason = 'incomplete_search_budget'
            if reject_reason:
                budget_failures[reject_reason] += 1
                rejected_refs.append(ref); rejected_jobs.append(dict(job, completion_reason=reject_reason))
                continue
            if selected not in search_res.get('actions', []):
                raise ValueError(f'P0: Selected action outside legal root on game {gid}')
            # A global cutoff never grants a fresh policy window or commits late work.
            if search_deadline is not None and clock() >= search_deadline:
                continue
            compact = {key: search_res.get(key) for key in (
                'proof_attempts', 'used_fallback', 'stop_reason', 'tree_stop_reason',
                'deadline_reached', 'wall_seconds', 'surrogate_model_key', 'surrogate_model_version')}
            compact.update(simulations=tot_sims, tree_simulations=t_sims, proof_checks=p_checks,
                budget_complete=budget_complete, budget_mode=budget_mode,
                queue_seconds=queue_seconds, deadline_scope='search_admission')
            commits.append((ref, actor, selected, ref.root_generation + 1, compact))
            old_refs.append(ref); commit_context.append((gid, ready_at, budget_complete))
        if commits:
            t0 = clock(); new_refs = backend.commit_actions(commits)
            commit_latencies.append(clock() - t0)
            if len(new_refs) != len(commits): raise ValueError('Physical commit lost a game')
            backend.release(old_refs)
            for ref, (gid, ready_at, budget_complete) in zip(new_refs, commit_context):
                active_games[gid] = (game_job_map[gid], ref)
                action_counts[gid] += 1
                committed_decisions += 1; committed_budget_complete += int(budget_complete)
                decision_latencies.append(clock() - ready_at)
                ready.append((gid, clock()))
        retire(rejected_refs, rejected_jobs)
        if clock() - last_progress_write >= 1.:
            _update_progress()
            last_progress_write = clock()

    try:
        _update_progress()
        scheduler = CooperativeScheduler(inference, max_roots=search_roots, allow_cpu_oracle=False)
        scheduler_stats = scheduler.run_stream(admit, completed, cancelled=(
            (lambda: clock() >= search_deadline) if search_deadline is not None else None))
        peak_outstanding_roots = scheduler_stats['peak_outstanding_roots']

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
    concurrency_1600_full_games_complete = (len(jobs) == 1600 and game_execution_complete and peak_outstanding_roots == 1600)
    fixed_budget_complete = budget_mode == 'simulations' and game_execution_complete and committed_decisions == committed_budget_complete and not budget_failures

    def _percentile(data: List[Union[int, float]], p: float) -> Optional[float]:
        if not data:
            return None
        return round(float(np.percentile(data, p)), 6)

    stats = {
        'resident_queue_latency_p50': _percentile(queue_latencies, 50),
        'resident_queue_latency_p95': _percentile(queue_latencies, 95),
        'resident_queue_latency_p99': _percentile(queue_latencies, 99),
        'decision_end_to_end_latency_p50': _percentile(decision_latencies, 50),
        'decision_end_to_end_latency_p95': _percentile(decision_latencies, 95),
        'decision_end_to_end_latency_p99': _percentile(decision_latencies, 99),
        'search_root_limit': search_roots,
        'resident_games': len(jobs),
        'resident_setup_seconds': resident_setup_seconds,
        'deadline_scope': 'search_admission',
        'committed_decisions': committed_decisions,
        'committed_budget_complete_decisions': committed_budget_complete,
        'incomplete_budget_roots': dict(budget_failures),
        'scheduler': scheduler_stats,
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


    resolved_rule_backend = report_rule_backend(backend, rule_backend)

    raw_runtime = runtime_name or getattr(backend, 'runtime_name', None) or getattr(backend, 'runtime', None)
    if raw_runtime in (RUNTIME_RECOVERY, RUNTIME_SEVEN):
        resolved_runtime = raw_runtime
    elif getattr(raw_runtime, '__name__', '').endswith('seven_runtime'):
        resolved_runtime = RUNTIME_SEVEN
    elif getattr(raw_runtime, '__name__', '').endswith('recovery_runtime'):
        resolved_runtime = RUNTIME_RECOVERY
    else:
        if raw_runtime is None:
            resolved_runtime = RUNTIME_RECOVERY
        else:
            raise ValueError(f'Cannot identify evaluation runtime from {raw_runtime!r}')
    resolved_capacity = capacity_plan if capacity_plan is not None else getattr(backend, 'capacity_plan', None)
    resolved_device_verified = bool(getattr(backend, 'device_verified', False))
    gpu_rule_execution = resolved_rule_backend == RULE_BACKEND_CUDA_VM

    return {
        'results': all_results,
        'total_games': len(jobs),
        'completed_games': completed_count,
        'uncompleted_games': uncompleted_count,
        'game_execution_complete': game_execution_complete,
        'concurrency_1600_full_games_complete': concurrency_1600_full_games_complete,
        'resident_1600_full_games_complete': len(jobs) == 1600 and game_execution_complete,
        'fixed_budget_complete': fixed_budget_complete,
        'search_root_limit': search_roots,
        'quality_approved': False,
        'training_updates': 0,
        'gpu_rule_execution': gpu_rule_execution,
        'rule_backend': resolved_rule_backend,
        'runtime': resolved_runtime,
        'device_verified': resolved_device_verified,
        'requested_search_roots': search_roots,
        'observed_search_root_peak': peak_outstanding_roots,
        'capacity_plan': resolved_capacity,
        'gpu_memory_budget_mib': gpu_memory_budget_mib,
        'denominator': COMPLETED_DENOMINATOR,
        'matrix_order': MATRIX_ORDER,
        'training_workflow_complete': False,
        'peak_outstanding_roots': peak_outstanding_roots,
        'stats': stats,
        'backend_stats': backend_stats,
        'device_statistics':device_stats,
        'method_error':method_error,
        'behavior':behavior_gate.report(),
        'evaluation_accepted':game_execution_complete and behavior_gate.report()['behavior_approved'] and (budget_mode != 'simulations' or fixed_budget_complete),
    }
