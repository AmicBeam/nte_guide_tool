"""Post-run evidence checks for device-resident Gumbel execution.

This module never runs rules or loads CUDA. It rejects incomplete evidence;
consistent records still require independent examination of the CUDA trace.
In-flight root counters alone are intentionally insufficient.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


SCHEMA = 'duel_device_search_evidence_v1'
SEARCH_STAGES = frozenset({
    'sample_world', 'legal_observation', 'inference', 'root_selection',
    'interior_selection', 'rule_step', 'backup', 'final_policy',
})


@dataclass(frozen=True)
class DeviceSearchContract:
    concurrent_roots: int = 1600
    simulations: int = 32
    gumbel_candidates: int = 16
    runtime: str = 'seven'
    q_scale: str = 'natural_wdl'

    def __post_init__(self):
        for value in (self.concurrent_roots, self.simulations, self.gumbel_candidates):
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer device-search limits required')
        if self.runtime != 'seven' or self.q_scale != 'natural_wdl':
            raise ValueError('Current seven-team natural-WDL contract required')


def _identities(value):
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(v, str) or not v for v in value):
        return None
    if len(set(value)) != len(value):
        return None
    return set(value)


def _artifact_verified(record, root):
    if not isinstance(record, dict):
        return False
    name, digest = record.get('path'), record.get('sha256')
    if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
        return False
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        return False
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return False
    if target.suffix not in ('.json', '.jsonl', '.csv', '.gz'):
        return False
    hasher = sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest() == digest


def audit_device_search(report, *, artifact_root, contract=DeviceSearchContract()):
    """Audit terminal evidence; never equate schema consistency with CUDA proof.

    ``dispatches`` must come from actual device execution/trace records, with
    the active root identities observed on device. ``search_driver`` means
    device control of search traversal, not the host cooperative scheduler.
    Roots may diverge in phase; individual stage batches need not have the
    same width as the active device-controlled cohort.
    """
    if not isinstance(report, dict) or not isinstance(contract, DeviceSearchContract):
        raise TypeError('Evidence mapping and DeviceSearchContract required')
    root = Path(artifact_root).resolve()
    reasons = []
    expected = dict(schema=SCHEMA, runtime=contract.runtime, algorithm='gumbel',
                    q_scale=contract.q_scale, simulations=contract.simulations,
                    gumbel_candidates=contract.gumbel_candidates,
                    rule_device='cuda', control_plane='device_resident',
                    source_kind='actual_cuda')
    for key, value in expected.items():
        if report.get(key) != value or isinstance(report.get(key), bool):
            reasons.append('identity:' + key)
    for key in ('cpu_rule_calls', 'host_root_steps', 'hot_path_d2h_bytes',
                'device_errors', 'budget_failures'):
        if type(report.get(key)) is not int or report[key] != 0:
            reasons.append('boundary:' + key)
    for key in ('rule_parity_passed', 'search_parity_passed', 'replay_verified'):
        if report.get(key) is not True:
            reasons.append('verification:' + key)
    for key in ('trace_artifact', 'episodes_artifact', 'source_manifest_artifact'):
        if not _artifact_verified(report.get(key), root):
            reasons.append('artifact:' + key)

    roots = report.get('search_roots')
    root_ids = None
    if isinstance(roots, list):
        root_ids = _identities([r.get('root_id') if isinstance(r, dict) else None for r in roots])
    completed = set()
    if root_ids is None:
        reasons.append('roots:missing_or_duplicate')
    else:
        for record in roots:
            count = record.get('completed_simulations')
            # Capacity evidence uses full-budget, nontrivial real-game roots.
            # One-action/proved-win shortcuts remain valid game behavior but
            # do not establish this full Gumbel capacity test.
            if (record.get('real_game') is True and record.get('complete') is True
                    and type(record.get('legal_actions')) is int and record['legal_actions'] > 1
                    and type(count) is int and count == contract.simulations
                    and record.get('policy_target_complete') is True
                    and record.get('terminal_game_complete') is True):
                completed.add(record['root_id'])
        if len(completed) != len(roots):
            reasons.append('roots:incomplete_budget_policy_or_game')

    dispatches = report.get('dispatches')
    stages = set()
    peak_device_roots = 0
    if not isinstance(dispatches, list) or not dispatches:
        reasons.append('dispatches:missing')
    else:
        for dispatch in dispatches:
            if not isinstance(dispatch, dict):
                reasons.append('dispatches:malformed')
                continue
            ids = _identities(dispatch.get('active_root_ids'))
            if (ids is None or root_ids is None or not ids <= root_ids
                    or type(dispatch.get('active_rows')) is not int
                    or dispatch['active_rows'] != len(ids)
                    or dispatch.get('device') != 'cuda'
                    or dispatch.get('event_completed') is not True
                    or dispatch.get('observed_on_device') is not True
                    or not isinstance(dispatch.get('kernel'), str) or not dispatch['kernel']):
                reasons.append('dispatches:unobserved_or_invalid')
                continue
            stage = dispatch.get('stage')
            if not isinstance(stage, str):
                reasons.append('dispatches:invalid_stage')
                continue
            stages.add(stage)
            if stage == 'search_driver' and ids <= completed:
                peak_device_roots = max(peak_device_roots, len(ids))
        missing = SEARCH_STAGES - stages
        reasons.extend('stage:' + stage for stage in sorted(missing))
    if peak_device_roots < contract.concurrent_roots:
        reasons.append('capacity:device_search_driver_roots')
    peak = report.get('peak_device_memory_bytes')
    budget = report.get('device_memory_budget_bytes')
    if (type(peak) is not int or type(budget) is not int
            or not 0 < peak <= budget):
        reasons.append('capacity:unmeasured_or_over_budget_memory')
    return dict(schema=SCHEMA, evidence_consistent=not reasons,
                reasons=sorted(set(reasons)), observed_device_search_roots=peak_device_roots,
                requested_device_search_roots=contract.concurrent_roots,
                independent_cuda_review_required=True, quality_approved=False)
