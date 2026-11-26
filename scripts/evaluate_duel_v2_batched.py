#!/usr/bin/env python3
"""CLI entry point for Duel V2 batched cooperative evaluation.

Default invocation outputs a pure configuration check without importing PyTorch,
loading neural weights, or launching worker processes. Full evaluation requires
explicit --run with a fresh output directory, seed-base, and budget-mode.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duel V2 Batched Cooperative Evaluation Driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the real cooperative evaluation (default only prints configuration check).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fresh output directory for reports, index, and physical traces (required if --run).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device: 'cpu' or 'cuda'.",
    )
    parser.add_argument('--q-scale', choices=('natural_wdl', 'mctx_default'), default='natural_wdl',
                        help='Frozen Q profile; requires matching same-profile device preflight evidence.')
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of dedicated resident CPU rule worker processes (1..64).",
    )
    parser.add_argument(
        "--search-roots",
        type=int,
        default=None,
        help="Simultaneous admitted searches (1..1600; default min(games, workers*2)); resident games wait separately.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1600,
        help="Total games in the evaluation matrix (1..1600; 1600 represents 64 games/cell).",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=900,
        help="Total wall-clock budget in seconds (1..14400), including 30s persistence reserve.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Base integer seed allocated from shared ledger (required if --run).",
    )
    parser.add_argument(
        "--budget-mode",
        choices=("simulations", "wall_clock"),
        default=None,
        help="Search budget: fixed 32 work units or 3s from search admission (resident queue wait reported separately).",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Directory containing RecoveryServingModel weights and serving binding.",
    )

    parser.add_argument(
        "--rule-backend",
        choices=("resident_cpu", "cuda_vm"),
        default="resident_cpu",
        help="Rule execution backend. Independent of inference --device. Default keeps resident CPU workers.",
    )
    parser.add_argument(
        "--runtime",
        choices=("seven",),
        default="seven",
        help="Current evaluation uses the unified seven-team schema.",
    )
    parser.add_argument(
        "--gpu-rule-program",
        type=Path,
        default=None,
        help="Frozen GPU rule program JSON. Required for cuda_vm when CPython 3.10 source capture is unavailable.",
    )
    parser.add_argument(
        "--guard-standalone-entities",
        action="store_true",
        default=False,
        help="Explicit CUDA VM standalone-entity guard. Rejected unless --rule-backend cuda_vm.",
    )
    parser.add_argument(
        "--gpu-memory-budget-mib",
        type=int,
        default=None,
        help="Device memory budget in MiB for GPU rule capacity planning. Not inferred.",
    )
    parser.add_argument(
        "--gpu-max-nodes",
        type=int,
        default=None,
        help="Bounded GPU graph node capacity. Omitted values are not silently defaulted by this driver.",
    )
    parser.add_argument(
        "--gpu-max-edges",
        type=int,
        default=None,
        help="Bounded GPU graph edge capacity.",
    )
    parser.add_argument(
        "--gpu-max-payload-bytes",
        type=int,
        default=None,
        help="Bounded GPU graph payload bytes.",
    )
    parser.add_argument(
        "--gpu-max-candidates",
        type=int,
        default=None,
        help="Bounded GPU candidate capacity.",
    )

    args = parser.parse_args(argv)

    if not 1 <= args.workers <= 64:
        parser.error(f"--workers must be between 1 and 64, got {args.workers}")
    if not 1 <= args.games <= 1600:
        parser.error(f"--games must be between 1 and 1600, got {args.games}")
    if args.search_roots is not None and not 1 <= args.search_roots <= 1600:
        parser.error("--search-roots must be between 1 and 1600")
    args.search_roots = min(args.games, args.search_roots or args.workers * 2)
    if not 1 <= args.seconds <= 14400:
        parser.error(f"--seconds must be a positive integer <= 14400, got {args.seconds}")


    gpu_only = []
    if args.gpu_rule_program is not None:
        gpu_only.append("--gpu-rule-program")
    if args.guard_standalone_entities:
        gpu_only.append("--guard-standalone-entities")
    if args.gpu_memory_budget_mib is not None:
        gpu_only.append("--gpu-memory-budget-mib")
    if args.gpu_max_nodes is not None:
        gpu_only.append("--gpu-max-nodes")
    if args.gpu_max_edges is not None:
        gpu_only.append("--gpu-max-edges")
    if args.gpu_max_payload_bytes is not None:
        gpu_only.append("--gpu-max-payload-bytes")
    if args.gpu_max_candidates is not None:
        gpu_only.append("--gpu-max-candidates")
    if args.rule_backend != "cuda_vm" and gpu_only:
        parser.error(
            "GPU rule flags require --rule-backend cuda_vm; no silent ignore: " + ", ".join(gpu_only)
        )
    if args.rule_backend == "cuda_vm" and args.gpu_rule_program is None and sys.version_info[:2] != (3, 10):
        parser.error(
            "--gpu-rule-program frozen JSON is required for --rule-backend cuda_vm on this host "
            "(not CPython 3.10). No CPU fallback."
        )
    for name, value in (
        ("--gpu-memory-budget-mib", args.gpu_memory_budget_mib),
        ("--gpu-max-nodes", args.gpu_max_nodes),
        ("--gpu-max-edges", args.gpu_max_edges),
        ("--gpu-max-payload-bytes", args.gpu_max_payload_bytes),
        ("--gpu-max-candidates", args.gpu_max_candidates),
    ):
        if value is not None and (type(value) is not int or value < 1):
            parser.error(f"{name} must be a positive integer, got {value}")

    if args.run:
        if args.output is None:
            parser.error("--output is required when --run is specified")
        if args.seed_base is None:
            parser.error("--seed-base is required when --run is specified")
        if args.budget_mode is None:
            parser.error("--budget-mode is required when --run is specified")

    return args


def main(argv: list[str] | None = None, *, device_driver=None) -> int:
    args = parse_args(argv)

    if not args.run:
        # Default mode: configuration check only. Never import Torch, never load weights, never spawn workers.
        capacity_plan = None
        if args.rule_backend == 'cuda_vm':
            from app.modules.card_game.rl.batched_duel.gpu_rule_capacity import plan_gpu_rule_capacity
            graph_limits = {
                name: value for name, value in (
                    ('max_nodes', args.gpu_max_nodes), ('max_edges', args.gpu_max_edges),
                    ('max_payload_bytes', args.gpu_max_payload_bytes),
                    ('max_candidates', args.gpu_max_candidates)) if value is not None}
            capacity_plan = plan_gpu_rule_capacity(
                args.games, args.search_roots, runtime=args.runtime,
                memory_budget_bytes=None if args.gpu_memory_budget_mib is None else args.gpu_memory_budget_mib * 1024 * 1024,
                **graph_limits)
        torch_imported = ("torch" in sys.modules) or ("torch.cuda" in sys.modules)
        config_check = {
            "action": "config_check_only",
            "run": False,
            "device": args.device,
            "q_scale": args.q_scale,

            "rule_backend": args.rule_backend,
            "runtime": args.runtime,
            "gpu_rule_program": str(args.gpu_rule_program) if args.gpu_rule_program else None,
            "guard_standalone_entities": bool(args.guard_standalone_entities),
            "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
            "gpu_max_nodes": args.gpu_max_nodes,
            "gpu_max_edges": args.gpu_max_edges,
            "gpu_max_payload_bytes": args.gpu_max_payload_bytes,
            "gpu_max_candidates": args.gpu_max_candidates,
            "workers": args.workers,
            "games": args.games,
            "seconds": args.seconds,
            "seed_base": args.seed_base,
            "budget_mode": args.budget_mode,
            "models_dir": str(args.models) if args.models else None,
            "output_dir": str(args.output) if args.output else None,
            "torch_imported": torch_imported,
            "processes_started": False,
            "weights_loaded": False,
            "requested_resident_games": args.games,
            "search_roots": args.search_roots,
            "deadline_scope": "search_admission",
            "concurrency_1600_ready": False,

            "device_verified": False,
            "gpu_rule_execution": False,
            "matrix_keys": 5 if args.runtime == "recovery" else 7,
            "matrix_order": "row_first_col_second",
            "capacity_plan": capacity_plan,
            "denominator": "completed_normal_games",
            "host_python": sys.version.split()[0],
            "source_capture_python_required": "3.10",
        }
        print(json.dumps(config_check, indent=2))
        return 2 if capacity_plan is not None and not capacity_plan['valid'] else 0

    # The historical CUDA VM backend is still a host cooperative dispatcher.
    # No output, weights, seed allocation or worker launch precedes this gate.
    from app.modules.card_game.rl.batched_search.device_execution_contract import (
        DeviceExecutionNotReady, require_closed_device_driver,
    )
    try:
        driver = require_closed_device_driver(device_driver, purpose='winrate_evaluation', q_scale=args.q_scale)
    except DeviceExecutionNotReady as exc:
        print(json.dumps(dict(action='refused_formal_device_execution', reason=str(exc),
                              outputs_created=False, weights_loaded=False,
                              processes_started=False, quality_approved=False), ensure_ascii=False))
        return 2
    driver.validate_frozen_inputs(args)
    result = driver.run_device_batch(args)
    if not isinstance(result, dict):
        raise TypeError('Closed device evaluator must return an evidence mapping')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get('device_execution_complete') is True else 2


def _historical_cooperative_run(args) -> int:
    """Historical host dispatcher, unreachable from the formal CLI entry.

    Kept for source reproduction only. It cannot fulfill the current closed
    GPU-search requirement or be selected by --device/--rule-backend flags.
    """
    start_monotonic = time.monotonic()
    deadline = start_monotonic + float(args.seconds)
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Fresh directory required.")

    source_manifest=ROOT/'source-manifest.json'
    source_revision=ROOT/'source-revision.json'
    if not source_manifest.is_file() or not source_revision.is_file():
        raise ValueError('Run a full evaluation from an explicit frozen source snapshot')
    from app.modules.card_game.rl.offline_sources import _allowed_source
    manifest_bytes=source_manifest.read_bytes()
    for name,item in json.loads(manifest_bytes).items():
        path=ROOT/name
        if (not _allowed_source(name) or path.is_symlink() or ROOT not in path.resolve().parents
                or sha256(path.read_bytes()).hexdigest()!=item['sha256']):
            raise ValueError('Frozen source identity mismatch: '+name)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "replays").mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False. Refuse CPU fallback.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()

    from app.modules.card_game.rl.batched_search.inference import BatchedInference
    from app.modules.card_game.rl.batched_search.statistics import BatchedStatistics
    from app.modules.card_game.rl.batched_search.evaluator import (
        evaluate_jobs,
        generate_matrix_jobs,
        compute_matrix_summary,
        write_matrix_csv,
        generate_report_markdown,
        write_result_json,
        make_source_snapshot,
        runtime_matrix_keys,
        select_rule_backend,
        statistics_capacity,
    )

    keys = runtime_matrix_keys(args.runtime)
    if args.runtime == "seven":
        from app.modules.card_game.rl import policy_runtime
        default_models = ROOT / "app/modules/card_game/engine/ai/models" / "seven"
        models_dir = args.models.resolve() if args.models else default_models
        models = {key: policy_runtime.model(models_dir, key) for key in keys}
    else:
        from app.modules.card_game.engine.ai.recovery_model import RecoveryServingModel, RELEASE
        models_dir = args.models.resolve() if args.models else (ROOT / "app/modules/card_game/engine/ai/models" / RELEASE)
        models = {key: RecoveryServingModel(models_dir, key) for key in keys}
    serving_decks = {key: models[key].serving_deck for key in keys}

    inference = BatchedInference(models, device=args.device)

    backend = select_rule_backend(
        rule_backend=args.rule_backend,
        runtime=args.runtime,
        workers=args.workers,
        max_games=args.games,
        search_roots=args.search_roots,
        record_output=output_dir,
        gpu_rule_program=args.gpu_rule_program,
        guard_standalone_entities=bool(args.guard_standalone_entities),
        gpu_memory_budget_mib=args.gpu_memory_budget_mib,
        max_nodes=args.gpu_max_nodes,
        max_edges=args.gpu_max_edges,
        max_payload_bytes=args.gpu_max_payload_bytes,
        max_candidates=args.gpu_max_candidates,
    )

    jobs = generate_matrix_jobs(
        keys=keys,
        serving_decks=serving_decks,
        seed_base=args.seed_base,
        total_games=args.games,
        output_dir=output_dir,
    )

    source_snapshot = make_source_snapshot(
        models,
        args.budget_mode,
        models_dir,
        rule_backend=args.rule_backend,
        runtime=args.runtime,
        search_roots=args.search_roots,
        games=args.games,
        capacity_plan=getattr(backend, "capacity_plan", None),
        gpu_program_identity=getattr(getattr(backend, "program", None), "identity", None),
        gpu_memory_budget_mib=args.gpu_memory_budget_mib,
        guard_standalone_entities=bool(args.guard_standalone_entities) if args.rule_backend == "cuda_vm" else None,
        device_verified=bool(getattr(backend, "device_verified", False)),
    )
    stat_nodes, stat_candidates = statistics_capacity(
        args.rule_backend, args.search_roots, getattr(backend, 'capacity_plan', None))
    with backend,BatchedStatistics(stat_nodes,stat_candidates,device=args.device) as statistics:
        eval_result = evaluate_jobs(
            jobs=jobs,
            models=models,
            inference=inference,
            backend=backend,
            budget_mode=args.budget_mode,
            decision_seconds=3.0,
            simulations=32,
            proof_cap=8,
            deadline=deadline,
            max_actions=800,
            search_roots=args.search_roots,
            output_dir=output_dir,
            statistics=statistics,

            rule_backend=args.rule_backend,
            runtime_name=args.runtime,
            capacity_plan=getattr(backend, "capacity_plan", None),
            gpu_memory_budget_mib=args.gpu_memory_budget_mib,
        )

    config_dict = {
        "device": args.device,
        "workers": args.workers,
        "search_roots": args.search_roots,
        "deadline_scope": "search_admission",
        "games": args.games,
        "seconds": args.seconds,
        "seed_base": args.seed_base,
        "budget_mode": args.budget_mode,
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
        "source_manifest_sha256":sha256(manifest_bytes).hexdigest(),
        "source_revision":json.loads(source_revision.read_text(encoding='utf-8')),

        "rule_backend": args.rule_backend,
        "runtime": args.runtime,
        "gpu_rule_program": str(args.gpu_rule_program) if args.gpu_rule_program else None,
        "guard_standalone_entities": bool(args.guard_standalone_entities) if args.rule_backend == "cuda_vm" else None,
        "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
        "gpu_max_nodes": args.gpu_max_nodes,
        "gpu_max_edges": args.gpu_max_edges,
        "gpu_max_payload_bytes": args.gpu_max_payload_bytes,
        "gpu_max_candidates": args.gpu_max_candidates,
        "capacity_plan": getattr(backend, "capacity_plan", None),
        "device_verified": bool(getattr(backend, "device_verified", False)),
        "matrix_order": "row_first_col_second",
        "denominator": "completed_normal_games",
    }

    matrix_summary = compute_matrix_summary(keys, eval_result["results"])
    if args.device=='cuda':
        torch.cuda.synchronize()
        eval_result['device_memory']=dict(allocated_peak_bytes=int(torch.cuda.max_memory_allocated()),
            reserved_peak_bytes=int(torch.cuda.max_memory_reserved()),
            scope='Calling process Torch allocator; excludes driver and unrelated applications')
    else:eval_result['device_memory']=None

    write_matrix_csv(matrix_summary, output_dir / "matrix.csv")
    write_result_json(output_dir / "result.json", eval_result, matrix_summary, config_dict, source_snapshot)
    report_md = generate_report_markdown(matrix_summary, eval_result, config_dict)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Evaluation completed: {eval_result['completed_games']}/{eval_result['total_games']} games.")
    print(f"Results written to: {output_dir}")
    return 0 if eval_result['evaluation_accepted'] else 1


if __name__ == "__main__":
    sys.exit(main())
