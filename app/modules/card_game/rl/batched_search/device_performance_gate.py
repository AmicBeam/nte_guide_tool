"""Post-batch throughput checks and bounded probe budgets; never authorization.

A caller supplies synchronized whole-GPU-batch measurements. Compilation and
warmup are reported separately. This gate does not qualify a rule executor or
approve formal learning, and never derives success from a process exit code.
"""
from dataclasses import dataclass
import math
import time


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite ' + name + ' required')
    return float(value)


@dataclass(frozen=True)
class ProbeBudget:
    setup_seconds: float = 300
    execution_seconds: float = 120

    def __post_init__(self):
        _positive(self.setup_seconds, 'setup seconds')
        _positive(self.execution_seconds, 'execution seconds')

    def deadline(self, phase, *, epoch=None, monotonic=None):
        if phase not in ('setup', 'execution'):
            raise ValueError('Known probe phase required')
        duration = getattr(self, phase + '_seconds')
        return (time.time() if epoch is None else epoch) + duration, (time.monotonic() if monotonic is None else monotonic) + duration

    @staticmethod
    def expired(deadline, *, epoch=None, monotonic=None):
        if len(deadline) != 2 or any(not math.isfinite(v) for v in deadline):
            raise ValueError('Finite epoch and monotonic deadlines required')
        return (time.time() if epoch is None else epoch) >= deadline[0] or (time.monotonic() if monotonic is None else monotonic) >= deadline[1]


@dataclass(frozen=True)
class SearchBatchMeasurement:
    batch_id: str
    source_identity: str
    scope: str
    roots: int
    distinct_initial_seeds: int
    peak_device_roots: int
    requested_min: int
    requested_max: int
    completed_min: int
    completed_max: int
    completed_total: int
    search_error_rows: int
    game_error_rows: int
    simulation_budget: int
    candidate_cap: int
    horizon: int
    steady_seconds: float
    setup_seconds: float
    host_rule_calls: int = 0
    host_per_root_search_calls: int = 0
    complete_policy_candidates: bool = False
    device_resident_search: bool = False
    timed_out: bool = False

    def __post_init__(self):
        if not isinstance(self.batch_id, str) or not self.batch_id or not isinstance(self.source_identity, str) or not self.source_identity or self.scope not in ('actual_duel_rules', 'synthetic'):
            raise ValueError('Measured batch, source identity and explicit scope required')
        counts = ('roots', 'distinct_initial_seeds', 'peak_device_roots', 'requested_min', 'requested_max',
                  'completed_min', 'completed_max', 'completed_total', 'search_error_rows', 'game_error_rows',
                  'simulation_budget', 'candidate_cap', 'horizon', 'host_rule_calls', 'host_per_root_search_calls')
        for name in counts:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('Nonnegative integer ' + name + ' required')
        _positive(self.steady_seconds, 'steady synchronized search seconds')
        if isinstance(self.setup_seconds, bool) or not math.isfinite(self.setup_seconds) or self.setup_seconds < 0:
            raise ValueError('Nonnegative setup seconds required')
        for name in ('complete_policy_candidates', 'device_resident_search', 'timed_out'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('Explicit boolean ' + name + ' required')
        if not self.roots or self.completed_min > self.completed_max or self.requested_min > self.requested_max:
            raise ValueError('Valid nonempty cohort and count ranges required')
        if not self.roots * self.completed_min <= self.completed_total <= self.roots * self.completed_max:
            raise ValueError('Measured completion total must fit the per-root range')
        if self.completed_max > self.requested_max or max(self.search_error_rows, self.game_error_rows) > self.roots:
            raise ValueError('Completion and errors must fit the actual cohort')

    def metrics(self):
        return dict(batch_id=self.batch_id, source_identity=self.source_identity, scope=self.scope,
                    roots=self.roots, completed_simulations=self.completed_total,
                    completed_simulations_per_second=self.completed_total / self.steady_seconds,
                    complete_root_decisions_per_second=(self.roots / self.steady_seconds
                        if self.completed_min == self.completed_max == self.simulation_budget
                        and self.search_error_rows == self.game_error_rows == 0 and not self.timed_out else 0),
                    steady_seconds=self.steady_seconds, setup_seconds=self.setup_seconds)


def assess_search_throughput(batches, *, max_batch_seconds=None, required_roots=1600, consecutive_batches=3):
    if type(required_roots) is not int or required_roots < 1 or type(consecutive_batches) is not int or consecutive_batches < 1:
        raise ValueError('Positive root and repeated-batch requirements required')
    if max_batch_seconds is not None:
        max_batch_seconds = _positive(max_batch_seconds, 'agreed batch time limit')
    batches = tuple(batches)
    if not all(isinstance(b, SearchBatchMeasurement) for b in batches):
        raise TypeError('Whole-batch measurements required')
    reasons = []
    if max_batch_seconds is None:
        reasons.append('throughput_threshold_not_set')
    if len(batches) < consecutive_batches:
        reasons.append('insufficient_repeated_batches')
    if len({b.batch_id for b in batches}) != len(batches):
        reasons.append('duplicate_batch_evidence')
    if len({b.source_identity for b in batches}) > 1:
        reasons.append('mixed_source_identities')
    for batch in batches:
        checks = dict(actual_rules=batch.scope == 'actual_duel_rules',
                      roots=batch.roots == batch.distinct_initial_seeds == batch.peak_device_roots == required_roots,
                      budget=(batch.simulation_budget, batch.candidate_cap, batch.horizon) == (32, 16, 3),
                      requested=batch.requested_min == batch.requested_max == 32,
                      completed=batch.completed_min == batch.completed_max == 32 and batch.completed_total == required_roots * 32,
                      zero_errors=batch.search_error_rows == batch.game_error_rows == 0,
                      gpu_only=batch.host_rule_calls == batch.host_per_root_search_calls == 0 and batch.device_resident_search,
                      full_policy_candidates=batch.complete_policy_candidates,
                      no_timeout=not batch.timed_out,
                      time_limit=max_batch_seconds is not None and batch.steady_seconds <= max_batch_seconds)
        reasons.extend(batch.batch_id + ':' + key for key, passed in checks.items() if not passed)
    return dict(throughput_passed=not reasons, reasons=reasons,
                measured=[batch.metrics() for batch in batches], max_batch_seconds=max_batch_seconds,
                rule_executor_qualified=False, formal_learning_authorized=False)


def require_training_performance(batches, *, max_batch_seconds=None, **requirements):
    result = assess_search_throughput(batches, max_batch_seconds=max_batch_seconds, **requirements)
    if not result['throughput_passed']:
        raise RuntimeError('GPU search throughput gate closed: ' + ', '.join(result['reasons']))
    return result
