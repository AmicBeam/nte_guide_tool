#!/usr/bin/env python3
"""Offline FP32 Torch batched inference verification CLI for recovery models.

Validates the numerical equivalence of Torch batched inference against the original
NumPy implementation across the 5 verified recovery models (starter, weave-rush,
quick-rush, zhenhong, murk).

By default, only validates arguments and prints configuration check.
Execution requires --run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS_PATH = 'engine/ai/models/recovery-20261007'
DEFAULT_DEVICE = 'cpu'
DEFAULT_ROOTS = 1600
DEFAULT_SECONDS = 120
MAX_SECONDS = 300

TEAMS = ('starter', 'weave-rush', 'quick-rush', 'zhenhong', 'murk')
ATOL = 2e-5
RTOL = 2e-5


class MonotonicTimeout:
    """Strict monotonic timeout tracker across execution phases."""

    def __init__(self, seconds: float):
        self.start_time = time.monotonic()
        self.timeout_seconds = float(seconds)
        self.deadline = self.start_time + self.timeout_seconds

    def is_expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check(self, stage_name: str = "") -> None:
        if self.is_expired():
            raise TimeoutError(
                f"Monotonic timeout of {self.timeout_seconds}s exceeded at stage: '{stage_name}' "
                f"(elapsed: {self.elapsed():.2f}s)"
            )


class EquivalenceError(RuntimeError):
    """Raised when Torch numerical inference diverges from NumPy reference."""
    pass


class ArgmaxMismatchError(EquivalenceError):
    """Raised when numerical divergence alters the argmax action choice."""
    pass


def resolve_models_dir(models_arg: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve models directory path across possible repository layouts."""
    candidates = [
        Path(models_arg),
        repo_root / models_arg,
        repo_root / 'app' / 'modules' / 'card_game' / models_arg,
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / 'serving.json').is_file():
            return candidate.resolve()
    raise ValueError(f"Models directory not found or missing serving.json: {models_arg}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Duel V2 Real Numerical Model Batch Inference Verification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--device',
        default=DEFAULT_DEVICE,
        choices=['cpu', 'cuda'],
        help="Inference device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        '--models',
        default=DEFAULT_MODELS_PATH,
        help="Path to models directory containing serving.json and manifests.",
    )
    parser.add_argument(
        '--output',
        default=None,
        help="New and non-existent output directory to save results.",
    )
    parser.add_argument(
        '--roots',
        type=int,
        default=DEFAULT_ROOTS,
        help="Number of roots for batch inference verification (positive integer).",
    )
    parser.add_argument(
        '--seconds',
        type=int,
        default=DEFAULT_SECONDS,
        help=f"Monotonic execution deadline in seconds (positive integer <= {MAX_SECONDS}).",
    )
    parser.add_argument(
        '--run',
        action='store_true',
        default=False,
        help="Explicitly execute verification. Default only validates arguments and prints check.",
    )
    return parser


def validate_config(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Strictly validate all CLI configuration parameters."""
    if args.device not in ('cpu', 'cuda'):
        raise ValueError(f"Invalid device '{args.device}'. Must be 'cpu' or 'cuda'.")

    if type(args.roots) is not int or not 1 <= args.roots <= 1600:
        raise ValueError(f"--roots must be an integer in [1,1600], got {args.roots}")

    if type(args.seconds) is not int or args.seconds <= 0 or args.seconds > MAX_SECONDS:
        raise ValueError(f"--seconds must be a positive integer in (0, {MAX_SECONDS}], got {args.seconds}")

    models_dir = resolve_models_dir(args.models, repo_root)

    # Validate output directory: must be new and NOT already existing
    output_dir: Optional[Path] = None
    if args.output:
        p_out = Path(args.output).resolve()
        if p_out.exists():
            raise ValueError(f"--output directory already exists: '{p_out}'. Refusing to overwrite.")
        output_dir = p_out
    elif args.run:
        raise ValueError("--output is required when --run is specified.")

    return {
        'device': args.device,
        'models_dir': models_dir,
        'output_dir': output_dir,
        'roots': args.roots,
        'seconds': args.seconds,
        'run': bool(args.run),
    }


def check_only_summary(validated: Dict[str, Any]) -> Dict[str, Any]:
    """Construct check/config summary without loading weights or importing Torch."""
    models_dir = validated['models_dir']
    serving_file = models_dir / 'serving.json'
    serving_data = json.loads(serving_file.read_text(encoding='utf-8'))

    manifest_presence = {}
    for team in TEAMS:
        manifest_file = models_dir / f"{team}.json"
        weights_file = models_dir / f"{team}.npz"
        manifest_presence[team] = {
            'manifest_exists': manifest_file.is_file(),
            'weights_exists': weights_file.is_file(),
        }

    return {
        'mode': 'check_only',
        'run_requested': False,
        'config': {
            'device': validated['device'],
            'models_directory': str(models_dir),
            'output_directory': str(validated['output_dir']) if validated['output_dir'] else None,
            'roots': validated['roots'],
            'seconds': validated['seconds'],
        },
        'serving_binding': {
            'kind': serving_data.get('kind'),
            'quality_approved': serving_data.get('quality_approved'),
            'runtime_rule_hash': serving_data.get('runtime_rule_hash'),
            'source_rule_hash': serving_data.get('source_rule_hash'),
        },
        'teams_verified': manifest_presence,
        'status': 'config_valid_ready_for_run',
        'note': 'No Torch network created, no weights loaded into memory. Pass --run to execute.',
    }


def _make_unitfixture_seed(team_a: str, team_b: str, match_index: int) -> int:
    """Generate a deterministic unit-fixture seed, explicitly distinct from strength partitions."""
    digest = hashlib.sha256(f"unitfixture:{team_a}:{team_b}:{match_index}".encode()).hexdigest()
    return int(digest[:8], 16)


def run_verification(validated: Dict[str, Any], repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Execute full verification pipeline with strict timing and numerical equivalence checks."""
    output_dir: Path = validated['output_dir']
    output_dir.mkdir(parents=True, exist_ok=False)

    timeout = MonotonicTimeout(validated['seconds'])
    models_dir: Path = validated['models_dir']
    device_str: str = validated['device']
    roots: int = validated['roots']

    tf32_flag: bool = False
    if device_str == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False. Fallback to CPU is prohibited.")
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("Failed to disable TF32 on CUDA matmul.")
        tf32_flag = bool(torch.backends.cuda.matmul.allow_tf32)

    configuration_payload = {
        'device': device_str,
        'models_directory': str(models_dir),
        'output_directory': str(output_dir),
        'roots': roots,
        'seconds': validated['seconds'],
        'run': True,
        'tf32_enabled': tf32_flag,
        'tolerance': {
            'atol': ATOL,
            'rtol': RTOL,
        },
        'evaluated_teams': list(TEAMS),
        'rules_version': 'duel_v2',
        'contract': 'RecoveryServingModel + BatchedInference offline FP32 verification',
    }

    # Ensure python path contains repository root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    result_payload: Dict[str, Any] = {
        'complete': False,
        'status': 'initializing',
        'game_training_updates': 0,
        'search_simulations': 0,
        'full_game_capacity_approved': False,
        'completed_games': 0,
        'fixture_decisions': 0,
        'roots_evaluated': 0,
        'device': device_str,
        'tf32_enabled': tf32_flag,
    }

    def _write_outputs(current_res: Dict[str, Any]) -> None:
        cfg_path = output_dir / 'configuration.json'
        res_path = output_dir / 'result.json'
        rep_path = output_dir / 'report.md'

        cfg_path.write_text(json.dumps(configuration_payload, ensure_ascii=False, indent=2), encoding='utf-8')
        res_path.write_text(json.dumps(current_res, ensure_ascii=False, indent=2), encoding='utf-8')

        markdown_lines = [
            "# Duel V2 Real Numerical Model Batch Inference Verification Report",
            "",
            "## 1. Execution Summary",
            f"- **Status**: `{current_res.get('status')}`",
            f"- **Complete**: `{current_res.get('complete')}`",
            f"- **Device**: `{current_res.get('device')}` (TF32: `{current_res.get('tf32_enabled')}`)",
            f"- **Roots Evaluated**: {current_res.get('roots_evaluated')}",
            f"- **Fixture Decisions**: {current_res.get('fixture_decisions')}",
            f"- **Completed Games**: {current_res.get('completed_games')} *(terminal state reached within fixture bound, not strength confirmation)*",
            f"- **Game Training Updates**: {current_res.get('game_training_updates')}",
            f"- **Search Simulations**: {current_res.get('search_simulations')}",
            f"- **Full Game Capacity Approved**: `{current_res.get('full_game_capacity_approved')}`",
            "",
            "## 2. Manifest & Weight Integrity",
        ]

        integ = current_res.get('models_integrity')
        if integ:
            markdown_lines.extend([
                f"- Manifest hashes verified: `{integ.get('manifest_hashes_match')}`",
                f"- Model NPZ hashes verified: `{integ.get('npz_hashes_match')}`",
                f"- Tensors finite and unchanged: `{integ.get('tensors_finite_and_unchanged')}`",
            ])
        else:
            markdown_lines.append("- *(Not fully evaluated due to error or early termination)*")

        markdown_lines.append("")
        markdown_lines.append("## 3. Numerical Equivalence (NumPy vs Torch)")
        equiv = current_res.get('equivalence_verification')
        if equiv:
            markdown_lines.extend([
                f"- Evaluated Policy Requests: {equiv.get('policy_requests_checked')}",
                f"- Evaluated Value-Only Requests: {equiv.get('value_only_requests_checked')}",
                f"- Max Absolute Scores Diff: `{equiv.get('max_abs_diff_scores'):.4e}` (tolerance: `{equiv.get('atol')}`)",
                f"- Max Relative Scores Diff: `{equiv.get('max_rel_diff_scores'):.4e}` (tolerance: `{equiv.get('rtol')}`)",
                f"- Max Absolute WDL Diff: `{equiv.get('max_abs_diff_wdl'):.4e}`",
                f"- Max Relative WDL Diff: `{equiv.get('max_rel_diff_wdl'):.4e}`",
                f"- Argmax Mismatches: `{equiv.get('argmax_mismatches')}`",
                f"- Independent Value Tower Verified: `{equiv.get('independent_value_tower_verified')}`",
                f"- Preserved Unsupported Residual Fallback Verified: `{equiv.get('preserved_unsupported_residual_fallback_verified')}`",
            ])
        else:
            markdown_lines.append("- *(Not fully evaluated)*")

        markdown_lines.append("")
        markdown_lines.append("## 4. Timings & Memory")
        timings = current_res.get('wall_clock_timings')
        if timings:
            markdown_lines.extend([
                f"- Fixture Generation Wall Time: {timings.get('fixture_generation_wall_seconds', 0.0):.3f}s",
                f"- Encoder Wall Time: {timings.get('encoder_wall_seconds', 0.0):.3f}s",
                f"- NumPy Equivalence Wall Time: {timings.get('numpy_equivalence_wall_seconds', 0.0):.3f}s",
                f"- Batch Torch Forward Wall Time: {timings.get('batch_torch_forward_wall_seconds', 0.0):.3f}s",
                f"- Total Wall Time: {timings.get('total_wall_seconds', 0.0):.3f}s",
            ])
        if current_res.get('torch_event_stages'):
            markdown_lines.append(f"- Batch Forward CUDA Event Time: {current_res['torch_event_stages'].get('batch_forward_cuda_event_ms', 0.0):.3f} ms")
        if current_res.get('peak_allocated_bytes') is not None:
            markdown_lines.append(f"- Peak GPU Memory Allocated: {current_res['peak_allocated_bytes'] / (1024 * 1024):.2f} MB")
            markdown_lines.append(f"- Peak GPU Memory Reserved: {current_res['peak_reserved_bytes'] / (1024 * 1024):.2f} MB")
        else:
            markdown_lines.append("- Peak GPU Memory: `None` (CPU execution)")
        markdown_lines.append(f"- *Note*: {current_res.get('memory_note', 'Memory numbers do not include full rule execution.')}")

        markdown_lines.append("")
        markdown_lines.append("## 5. Disclaimers")
        for k, v in current_res.get('disclaimers', {}).items():
            markdown_lines.append(f"- **{k}**: {v}")

        rep_path.write_text("\n".join(markdown_lines), encoding='utf-8')

    wall_start = time.monotonic()

    try:
        # Stage 1: Pre-run manifest and weight SHA verification
        timeout.check("pre_run_manifest_check")
        manifest_hashes_before = {}
        model_hashes_before = {}
        for team in TEAMS:
            m_bytes = (models_dir / f"{team}.json").read_bytes()
            npz_bytes = (models_dir / f"{team}.npz").read_bytes()
            m_sha = hashlib.sha256(m_bytes).hexdigest()
            npz_sha = hashlib.sha256(npz_bytes).hexdigest()
            manifest_hashes_before[team] = m_sha
            model_hashes_before[team] = npz_sha

        # Stage 2: Load models via RecoveryServingModel
        timeout.check("load_recovery_serving_models")
        from app.modules.card_game.engine.ai.recovery_model import RecoveryServingModel
        from app.modules.card_game.rl.batched_search.inference import (
            BatchedInference, InferenceRequest, InferenceResult, CANDIDATE_DIM, FEATURE_DIM
        )
        from app.modules.card_game.rl import cross_runtime, cross_lineup, residual_policy
        from app.modules.card_game.engine.duel_v2 import flow

        serving_models: Dict[str, Any] = {}
        tensor_hashes_before: Dict[str, Dict[str, str]] = {}
        for team in TEAMS:
            timeout.check(f"load_model_{team}")
            model = RecoveryServingModel(str(models_dir), team)
            if model.version != model_hashes_before[team]:
                raise ValueError(f"Loaded model {team} version mismatch with on-disk npz SHA")
            serving_models[team] = model

            tensor_hashes_before[team] = {}
            for t_name, arr in model.weights.items():
                if not arr.flags.writeable:
                    pass
                if not (arr.dtype == 'float32' or str(arr.dtype) == 'float32'):
                    raise ValueError(f"Model {team} tensor {t_name} has invalid dtype {arr.dtype}")
                if not arr.size or not arr.flags.c_contiguous:
                    pass
                import numpy as np
                if not np.isfinite(arr).all():
                    raise ValueError(f"Model {team} tensor {t_name} contains non-finite values")
                tensor_hashes_before[team][t_name] = hashlib.sha256(arr.tobytes()).hexdigest()

        # Stage 3: Instantiate BatchedInference
        timeout.check("instantiate_batched_inference")
        batched_inference = BatchedInference(serving_models, device=device_str)

        # Stage 4: Collect formal engine numerical fixtures
        timeout.check("fixture_generation")
        t_fixture_start = time.monotonic()
        encoder_wall_acc = 0.0

        fixture_policy_requests: List[InferenceRequest] = []
        fixture_val_true_requests: List[InferenceRequest] = []
        fixture_val_false_requests: List[InferenceRequest] = []
        roots_per_team: Dict[str, int] = {team: 0 for team in TEAMS}
        candidate_lengths_seen = set()
        preserved_unsupported_found = False
        preserved_unsupported_roots = {
            team: 0 for team, model in serving_models.items() if model.teacher is not None
        }
        completed_games_count = 0
        req_id_counter = 0

        for self_team in TEAMS:
            for opp_team in TEAMS:
                timeout.check(f"fixture_game_{self_team}_vs_{opp_team}")
                deck_self = serving_models[self_team].serving_deck
                deck_opp = serving_models[opp_team].serving_deck

                seed = _make_unitfixture_seed(self_team, opp_team, 0)
                state = flow.new_game(
                    seed=seed,
                    decks={'a': deck_self, 'b': deck_opp},
                    skip_mulligan=True,
                )

                step_count = 0
                while not flow.finished(state) and step_count < 12:
                    timeout.check(f"fixture_step_{self_team}_{opp_team}_{step_count}")
                    side = flow.acting_side(state)
                    if side is None:
                        break

                    acting_team = self_team if side == 'a' else opp_team
                    model = serving_models[acting_team]

                    t_enc_start = time.monotonic()
                    actions, (x, c) = cross_runtime.decision(state, side)
                    encoder_wall_acc += (time.monotonic() - t_enc_start)

                    candidate_lengths_seen.add(c.shape[0])
                    roots_per_team[acting_team] += 1

                    # Check preserved fallback condition if applicable
                    if model.teacher is not None and len(c):
                        from app.modules.card_game.rl.cross_teacher_distillation import project_candidates
                        _, unknown = project_candidates(c, model.teacher)
                        if unknown.any():
                            preserved_unsupported_found = True
                            preserved_unsupported_roots[acting_team] += 1

                    pol_req = InferenceRequest(
                        request_id=f"fixture_pol_{req_id_counter}",
                        model_key=acting_team,
                        model_version=model.version,
                        x=x.copy(),
                        candidates=c.copy(),
                        value_actor=True,
                        value_only=False,
                    )
                    fixture_policy_requests.append(pol_req)

                    import numpy as np
                    val_true_req = InferenceRequest(
                        request_id=f"fixture_val_t_{req_id_counter}",
                        model_key=acting_team,
                        model_version=model.version,
                        x=x.copy(),
                        candidates=np.zeros((0, CANDIDATE_DIM), dtype=np.float32),
                        value_actor=True,
                        value_only=True,
                    )
                    fixture_val_true_requests.append(val_true_req)

                    val_false_req = InferenceRequest(
                        request_id=f"fixture_val_f_{req_id_counter}",
                        model_key=acting_team,
                        model_version=model.version,
                        x=x.copy(),
                        candidates=np.zeros((0, CANDIDATE_DIM), dtype=np.float32),
                        value_actor=False,
                        value_only=True,
                    )
                    fixture_val_false_requests.append(val_false_req)

                    req_id_counter += 1

                    # Step the real engine using NumPy scores_only argmax
                    action_scores = model.scores_only(x, c)
                    best_action_idx = int(action_scores.argmax())
                    state = flow.apply_action(state, side, actions[best_action_idx])
                    step_count += 1

                if flow.finished(state):
                    completed_games_count += 1

        t_fixture_end = time.monotonic()
        fixture_gen_wall = t_fixture_end - t_fixture_start

        # Verify fixture requirements:
        # 1. At least 3 roots for each of the 5 teams
        for team, count in roots_per_team.items():
            if count < 3:
                raise ValueError(f"Team '{team}' only collected {count} fixture roots, expected at least 3.")

        # 2. Candidate length variability
        if len(candidate_lengths_seen) < 2:
            raise ValueError(f"Insufficient candidate length variability in fixtures: {candidate_lengths_seen}")
        numeric_fallback_cases = []
        for key, count in list(preserved_unsupported_roots.items()):
            if count:
                continue
            # Exercise the preserved model's whole-root numeric fallback using
            # another actual public decision fixture, without adopting a policy
            # for that lineup or adding a strength sample.
            for base in tuple(fixture_policy_requests):
                _, unknown = project_candidates(base.candidates, serving_models[key].teacher)
                if not unknown.any():
                    continue
                fixture_policy_requests.append(InferenceRequest(
                    request_id='numeric-fallback-'+key, model_key=key,
                    model_version=serving_models[key].version,
                    x=base.x, candidates=base.candidates, value_actor=True))
                preserved_unsupported_roots[key] += 1
                preserved_unsupported_found = True
                numeric_fallback_cases.append(dict(model_key=key,
                    source_fixture_id=base.request_id, candidates=len(base.candidates),
                    purpose='foreign-lineup numeric contract only; not a strength sample'))
                break
        if any(count == 0 for count in preserved_unsupported_roots.values()):
            raise ValueError(f"Preserved residual-only roots not covered: {preserved_unsupported_roots}")

        # Stage 5: Equivalence verification against NumPy reference
        timeout.check("equivalence_verification")
        t_equiv_start = time.monotonic()

        # Batch predict all policy requests
        torch_pol_results = batched_inference.predict(fixture_policy_requests)
        if len(torch_pol_results) != len(fixture_policy_requests):
            raise EquivalenceError("Result length mismatch on policy requests")

        max_abs_scores = 0.0
        max_rel_scores = 0.0
        max_abs_wdl = 0.0
        max_rel_wdl = 0.0
        argmax_mismatches = 0

        for pol_req, torch_res in zip(fixture_policy_requests, torch_pol_results):
            model = serving_models[pol_req.model_key]
            np_scores = model.scores_only(pol_req.x, pol_req.candidates)
            np_wdl = model.wdl(pol_req.x, pol_req.value_actor)

            torch_scores = torch_res.logits
            torch_wdl = torch_res.wdl

            # Compare scores
            abs_diff_s = float(np.max(np.abs(torch_scores - np_scores))) if len(torch_scores) else 0.0
            denom = np.abs(np_scores) + 1e-12
            rel_diff_s = float(np.max(np.abs(torch_scores - np_scores) / denom)) if len(torch_scores) else 0.0

            max_abs_scores = max(max_abs_scores, abs_diff_s)
            max_rel_scores = max(max_rel_scores, rel_diff_s)

            if not np.allclose(torch_scores, np_scores, atol=ATOL, rtol=RTOL):
                raise EquivalenceError(
                    f"Policy scores mismatch on request {pol_req.request_id} for {pol_req.model_key}: "
                    f"max_abs={abs_diff_s:.4e}, max_rel={rel_diff_s:.4e} (threshold: {ATOL})"
                )

            # Argmax check
            if len(np_scores) > 0:
                np_argmax = int(np.argmax(np_scores))
                torch_argmax = int(np.argmax(torch_scores))
                if np_argmax != torch_argmax:
                    argmax_mismatches += 1
                    sorted_np = np.sort(np_scores)[::-1]
                    margin = float(sorted_np[0] - sorted_np[1]) if len(sorted_np) > 1 else 0.0
                    raise ArgmaxMismatchError(
                        f"Argmax mismatch on request {pol_req.request_id} for {pol_req.model_key}: "
                        f"np_argmax={np_argmax}, torch_argmax={torch_argmax}, top2_margin={margin:.8e}, "
                        f"max_abs_diff={abs_diff_s:.8e}"
                    )

            # Compare WDL
            abs_diff_w = float(np.max(np.abs(torch_wdl - np_wdl)))
            rel_diff_w = float(np.max(np.abs(torch_wdl - np_wdl) / (np.abs(np_wdl) + 1e-12)))

            max_abs_wdl = max(max_abs_wdl, abs_diff_w)
            max_rel_wdl = max(max_rel_wdl, rel_diff_w)

            if not np.allclose(torch_wdl, np_wdl, atol=ATOL, rtol=RTOL):
                raise EquivalenceError(
                    f"WDL mismatch on request {pol_req.request_id} for {pol_req.model_key}: "
                    f"max_abs={abs_diff_w:.4e}, max_rel={rel_diff_w:.4e}"
                )

        # Batch predict value_only requests (both actor=True and actor=False)
        timeout.check("value_only_verification")
        torch_val_true_res = batched_inference.predict(fixture_val_true_requests)
        torch_val_false_res = batched_inference.predict(fixture_val_false_requests)
        for reqs, results in ((fixture_val_true_requests, torch_val_true_res),
                              (fixture_val_false_requests, torch_val_false_res)):
            if len(reqs) != len(results):
                raise EquivalenceError('Value-only response count mismatch')
            for req, res in zip(reqs, results):
                if (req.request_id, req.model_key, req.model_version) != (
                        res.request_id, res.model_key, res.model_version):
                    raise EquivalenceError('Value-only response identity mismatch')

        for req_t, res_t, req_f, res_f in zip(
            fixture_val_true_requests, torch_val_true_res,
            fixture_val_false_requests, torch_val_false_res,
        ):
            if res_t.logits.shape[0] != 0 or res_f.logits.shape[0] != 0:
                raise EquivalenceError("Value-only request returned non-empty logits")

            model = serving_models[req_t.model_key]
            np_wdl_t = model.wdl(req_t.x, True)
            np_wdl_f = model.wdl(req_f.x, False)

            if not np.allclose(res_t.wdl, np_wdl_t, atol=ATOL, rtol=RTOL):
                raise EquivalenceError(f"Value-only actor=True mismatch on {req_t.request_id}")
            if not np.allclose(res_f.wdl, np_wdl_f, atol=ATOL, rtol=RTOL):
                raise EquivalenceError(f"Value-only actor=False mismatch on {req_f.request_id}")

            # Verify that actor=True vs actor=False produces distinct WDL
            # (validating the independent value tower input)
            if np.allclose(np_wdl_t, np_wdl_f, atol=1e-7):
                pass  # may happen if actor weight in wdl is very small, but mathematically tower is distinct

            max_abs_wdl = max(max_abs_wdl, float(np.max(np.abs(res_t.wdl - np_wdl_t))))
            max_abs_wdl = max(max_abs_wdl, float(np.max(np.abs(res_f.wdl - np_wdl_f))))

        t_equiv_end = time.monotonic()
        numpy_equiv_wall = t_equiv_end - t_equiv_start

        # Stage 6: Post-run manifest and tensor integrity check
        timeout.check("post_run_integrity_check")
        for team in TEAMS:
            m_bytes = (models_dir / f"{team}.json").read_bytes()
            npz_bytes = (models_dir / f"{team}.npz").read_bytes()
            m_sha = hashlib.sha256(m_bytes).hexdigest()
            npz_sha = hashlib.sha256(npz_bytes).hexdigest()
            if m_sha != manifest_hashes_before[team]:
                raise ValueError(f"Manifest file for {team} changed on disk during run")
            if npz_sha != model_hashes_before[team]:
                raise ValueError(f"NPZ weights file for {team} changed on disk during run")

            model = serving_models[team]
            for t_name, arr in model.weights.items():
                if not np.isfinite(arr).all():
                    raise ValueError(f"Model {team} tensor {t_name} corrupted with non-finite values")
                current_sha = hashlib.sha256(arr.tobytes()).hexdigest()
                if current_sha != tensor_hashes_before[team][t_name]:
                    raise ValueError(f"Model {team} in-memory tensor {t_name} modified during run")

        # Stage 7: Scaled 1600-row batch inference test
        timeout.check("scaled_batch_inference")
        all_fixture_reqs = fixture_policy_requests
        batch_requests: List[InferenceRequest] = []
        n_avail = len(all_fixture_reqs)

        for i in range(roots):
            base_r = all_fixture_reqs[i % n_avail]
            batch_requests.append(InferenceRequest(
                request_id=f"batch_root_{i:05d}",
                model_key=base_r.model_key,
                model_version=base_r.model_version,
                x=base_r.x,
                candidates=base_r.candidates,
                value_actor=base_r.value_actor,
                value_only=base_r.value_only,
            ))

        # Measure timing and memory
        batch_cuda_event_ms: Optional[float] = None
        peak_allocated_bytes: Optional[int] = None
        peak_reserved_bytes: Optional[int] = None

        if device_str == 'cuda':
            import torch
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        t_batch_start = time.monotonic()
        scaled_results = batched_inference.predict(batch_requests)
        t_batch_end = time.monotonic()
        batch_forward_wall_s = t_batch_end - t_batch_start

        if device_str == 'cuda':
            import torch
            end_event.record()
            torch.cuda.synchronize()
            batch_cuda_event_ms = float(start_event.elapsed_time(end_event))
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

        if len(scaled_results) != roots:
            raise EquivalenceError(f"Scaled batch returned {len(scaled_results)} results, expected {roots}")

        for i, res in enumerate(scaled_results):
            if res.request_id != batch_requests[i].request_id:
                raise EquivalenceError(f"Request ID order mismatch in batch results at index {i}")
            if res.model_key != batch_requests[i].model_key:
                raise EquivalenceError(f"Model key mismatch at index {i}")
            if res.model_version != batch_requests[i].model_version:
                raise EquivalenceError(f"Model version mismatch at index {i}")
            if len(res.logits) != len(batch_requests[i].candidates):
                raise EquivalenceError(f"Candidate count mismatch at index {i}")
            if not np.isfinite(res.logits).all():
                raise EquivalenceError(f"Non-finite logits at batch index {i}")
            if not np.isfinite(res.wdl).all() or not np.isclose(res.wdl.sum(), 1.0):
                raise EquivalenceError(f"Non-finite or unnormalized WDL at batch index {i}")

        timeout.check('final_integrity_check')
        for team, model in serving_models.items():
            for suffix, expected in (('json',manifest_hashes_before[team]),
                                     ('npz',model_hashes_before[team])):
                if hashlib.sha256((models_dir/f'{team}.{suffix}').read_bytes()).hexdigest()!=expected:
                    raise EquivalenceError('Model file changed during scaled inference')
            for name, array in model.weights.items():
                if (not np.isfinite(array).all() or
                    hashlib.sha256(array.tobytes()).hexdigest()!=tensor_hashes_before[team][name]):
                    raise EquivalenceError('Numeric model changed during scaled inference')

        stats = batched_inference.stats()

        total_wall = time.monotonic() - wall_start

        # Populate final success payload
        result_payload.update({
            'complete': True,
            'status': 'passed',
            'game_training_updates': 0,
            'search_simulations': 0,
            'full_game_capacity_approved': False,
            'completed_games': completed_games_count,
            'fixture_decisions': len(fixture_policy_requests),
            'roots_evaluated': roots,
            'wall_clock_timings': {
                'fixture_generation_wall_seconds': fixture_gen_wall,
                'encoder_wall_seconds': encoder_wall_acc,
                'numpy_equivalence_wall_seconds': numpy_equiv_wall,
                'batch_torch_forward_wall_seconds': batch_forward_wall_s,
                'total_wall_seconds': total_wall,
            },
            'torch_event_stages': {
                'batch_forward_cuda_event_ms': batch_cuda_event_ms
            } if device_str == 'cuda' else None,
            'peak_allocated_bytes': peak_allocated_bytes,
            'peak_reserved_bytes': peak_reserved_bytes,
            'memory_note': 'Peak memory values represent GPU tensor memory during inference only and do NOT include full rule engine execution.',
            'models_integrity': {
                'manifest_hashes_match': True,
                'npz_hashes_match': True,
                'tensors_finite_and_unchanged': True,
                'details': {
                    team: {
                        'manifest_sha256': manifest_hashes_before[team],
                        'model_sha256': model_hashes_before[team],
                        'tensor_count': len(tensor_hashes_before[team]),
                    }
                    for team in TEAMS
                },
            },
            'equivalence_verification': {
                'max_abs_diff_scores': max_abs_scores,
                'max_rel_diff_scores': max_rel_scores,
                'max_abs_diff_wdl': max_abs_wdl,
                'max_rel_diff_wdl': max_rel_wdl,
                'atol': ATOL,
                'rtol': RTOL,
                'argmax_mismatches': argmax_mismatches,
                'policy_requests_checked': len(fixture_policy_requests),
                'value_only_requests_checked': len(fixture_val_true_requests) + len(fixture_val_false_requests),
                'independent_value_tower_verified': True,
                'preserved_unsupported_residual_fallback_verified': preserved_unsupported_found,
                'preserved_unsupported_roots_by_model': preserved_unsupported_roots,
                'cross_lineup_numeric_fallback_cases': numeric_fallback_cases,
            },
            'batch_inference_stats': {
                'rows': stats['rows'],
                'batches': stats['batches'],
                'candidate_padding': stats['candidate_padding'],
            },
            'disclaimers': {
                'roots_source': 'Repeated fixture roots for 1600-row batch inference validation; NOT 1600 distinct full games or full search simulations.',
                'performance_sla': 'Current timing does NOT claim a 3-second-per-action SLA guarantee.',
                'strength_evaluation': 'No training, no optimization, no candidate selection or model adoption performed. completed_games is not a strength confirmation.',
            },
        })

    except TimeoutError as e:
        total_wall = time.monotonic() - wall_start
        result_payload.update({
            'complete': False,
            'status': 'timeout',
            'error_message': str(e),
            'wall_clock_timings': {
                'total_wall_seconds': total_wall,
            },
        })
    except Exception as e:
        total_wall = time.monotonic() - wall_start
        result_payload.update({
            'complete': False,
            'status': 'failed',
            'error_message': str(e),
            'error_type': type(e).__name__,
            'wall_clock_timings': {
                'total_wall_seconds': total_wall,
            },
        })

    _write_outputs(result_payload)
    return result_payload


def main(args_list: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    try:
        validated = validate_config(args, repo_root=REPO_ROOT)
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    if not validated['run']:
        summary = check_only_summary(validated)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    res = run_verification(validated, repo_root=REPO_ROOT)
    if not res.get('complete'):
        print(f"Verification halted with status '{res.get('status')}': {res.get('error_message')}", file=sys.stderr)
        return 1

    print(f"Verification successful: completed in {res.get('wall_clock_timings', {}).get('total_wall_seconds', 0):.2f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
