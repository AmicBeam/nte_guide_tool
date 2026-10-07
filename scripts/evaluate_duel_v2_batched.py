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
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of dedicated resident CPU rule worker processes (1..64).",
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
        help="Search budget mode: 'simulations' (fixed 32) or 'wall_clock' (3.0s shared).",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="Directory containing RecoveryServingModel weights and serving binding.",
    )

    args = parser.parse_args(argv)

    if not 1 <= args.workers <= 64:
        parser.error(f"--workers must be between 1 and 64, got {args.workers}")
    if not 1 <= args.games <= 1600:
        parser.error(f"--games must be between 1 and 1600, got {args.games}")
    if not 1 <= args.seconds <= 14400:
        parser.error(f"--seconds must be a positive integer <= 14400, got {args.seconds}")

    if args.run:
        if args.output is None:
            parser.error("--output is required when --run is specified")
        if args.seed_base is None:
            parser.error("--seed-base is required when --run is specified")
        if args.budget_mode is None:
            parser.error("--budget-mode is required when --run is specified")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.run:
        # Default mode: configuration check only. Never import Torch, never load weights, never spawn workers.
        torch_imported = ("torch" in sys.modules) or ("torch.cuda" in sys.modules)
        config_check = {
            "action": "config_check_only",
            "run": False,
            "device": args.device,
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
            "requested_concurrent_games": args.games,
            "concurrency_1600_ready": False,
        }
        print(json.dumps(config_check, indent=2))
        return 0

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

    from app.modules.card_game.engine.ai.recovery_model import RecoveryServingModel, RELEASE
    from app.modules.card_game.rl.batched_duel.process_backend import ResidentRuleBackend
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
    )
    from app.modules.card_game.rl import recovery_runtime

    models_dir = args.models.resolve() if args.models else (ROOT / 'app/modules/card_game/engine/ai/models' / RELEASE)
    keys = ("starter", "weave-rush", "quick-rush", "zhenhong", "murk")

    models = {key: RecoveryServingModel(models_dir, key) for key in keys}
    serving_decks = {key: models[key].serving_deck for key in keys}

    inference = BatchedInference(models, device=args.device)

    backend = ResidentRuleBackend(
        workers=args.workers,
        max_games=args.games,
        runtime=recovery_runtime,
        record_output=output_dir,
    )

    jobs = generate_matrix_jobs(
        keys=keys,
        serving_decks=serving_decks,
        seed_base=args.seed_base,
        total_games=args.games,
        output_dir=output_dir,
    )

    source_snapshot = make_source_snapshot(models, args.budget_mode, models_dir)
    with backend,BatchedStatistics(33*args.games,2048,device=args.device) as statistics:
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
            output_dir=output_dir,
            statistics=statistics,
        )

    config_dict = {
        "device": args.device,
        "workers": args.workers,
        "games": args.games,
        "seconds": args.seconds,
        "seed_base": args.seed_base,
        "budget_mode": args.budget_mode,
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
        "source_manifest_sha256":sha256(manifest_bytes).hexdigest(),
        "source_revision":json.loads(source_revision.read_text(encoding='utf-8')),
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
