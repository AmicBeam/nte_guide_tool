"""Qualified device collection boundary for the existing seven-team trainer.

No executor is manufactured here. The real driver supplies complete CUDA
episodes and frozen evaluation records; this context refuses CPU rule maps.
Numerical updates, selection and checkpoint semantics remain in seven_round.
"""


class ClosedDeviceLearningContext:
    def __init__(self, driver, *, q_scale=None):
        from .device_execution_contract import require_closed_device_driver
        self.driver = require_closed_device_driver(driver, purpose='training', q_scale=q_scale)
        for name in ('collect', 'evaluate_jobs'):
            if not callable(getattr(self.driver, name, None)):
                raise ValueError('Closed device learning requires ' + name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # The outer service owns the CUDA arenas and frozen model bank.
        return False

    def map(self, *args, **kwargs):
        raise RuntimeError('CPU rule callbacks are forbidden in closed device learning')

    @property
    def collector(self):
        return self

    @property
    def last_diagnostics(self):
        return getattr(self.driver, 'last_diagnostics', None)

    def _checked_results(self, jobs, results):
        results = list(results)
        ids = [job.get('id') for job in jobs]
        if any(not isinstance(id, str) or not id for id in ids) or len(set(ids)) != len(ids):
            raise ValueError('Unique nonempty device job identities required')
        if len(results) != len(jobs):
            raise ValueError('Closed device execution omitted or duplicated jobs')
        for job, result in zip(jobs, results):
            if not isinstance(result, dict) or any(result.get(key) != job[key]
                    for key in ('id', 'seed', 'first', 'left', 'right') if key in job):
                raise ValueError('Closed device result identity/order mismatch')
        return results

    def collect(self, jobs):
        self._check_q_profiles(jobs)
        return self._checked_results(jobs, self.driver.collect(jobs))

    def evaluate_jobs(self, jobs):
        self._check_q_profiles(jobs)
        return self._checked_results(jobs, self.driver.evaluate_jobs(jobs))

    def _check_q_profiles(self, jobs):
        """Reject a different search contract before the device service runs."""
        expected = getattr(self.driver, 'q_scale', 'natural_wdl')
        for job in jobs:
            if job.get('pure') is True:
                continue
            profile = job.get('q_scale', 'natural_wdl')
            if type(profile) is not str or profile not in ('natural_wdl', 'mctx_default') or profile != expected:
                raise ValueError('Search job Q profile differs from the frozen device driver')


class DeviceValueCalibrationContext:
    """Live argmax-only bootstrap for warm_value, without Gumbel approval.

    The real GPU sampler is deliberately separate from formal policy learning.
    It cannot run search jobs, comparison telemetry or host rule callbacks.
    """
    def __init__(self, collector, *, cache_dir=None, continuation=None, max_segments=None,
                 max_new_batches=None):
        from .device_pure_collection import DevicePureCollector
        if not isinstance(collector, DevicePureCollector):
            raise TypeError('Live CUDA pure collector required for value calibration')
        self._collector = collector
        if cache_dir is not None and getattr(collector, 'segment_decisions', None) is None:
            raise ValueError('Persistent calibration cache requires segmented collection')
        if cache_dir is None and (continuation is not None or max_segments is not None or max_new_batches is not None):
            raise ValueError('Continuation/process bounds require explicit calibration cache_dir')
        if max_new_batches is not None and (type(max_new_batches) is not int or max_new_batches < 1):
            raise ValueError('Explicit positive new-batch process limit required')
        self.cache_dir, self.continuation, self.max_segments = cache_dir, continuation, max_segments
        self.max_new_batches = max_new_batches
        self._new_batches_in_window = 0
        self._diagnostics = dict(batches=[], completed_games=0)

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def map(self, *args, **kwargs):
        raise RuntimeError('CPU rule callbacks are forbidden in device value calibration')

    @property
    def collector(self): return self

    @property
    def last_diagnostics(self): return self._diagnostics

    def collect(self, jobs):
        if not jobs or any(j.get('pure') is not True or j.get('training') is not False for j in jobs):
            raise ValueError('Value calibration context accepts only explicit pure jobs')
        from .training_rollout import models_for_jobs
        from copy import deepcopy
        from .soa_pure_jobs import adapt_pure_jobs
        models = models_for_jobs(jobs)
        # Check duplicate physical jobs/identities across chunk boundaries too.
        adapt_pure_jobs(jobs, models)
        self._diagnostics = dict(batches=[], completed_games=0)
        results = []
        new_batches = self._new_batches_in_window
        # Segment restore retains both full device archives and their uploaded
        # history; bound simultaneous roots separately from the phase job count.
        batch_rows = 8 if getattr(self._collector, 'segment_decisions', None) is not None else 32
        for start in range(0, len(jobs), batch_rows):
            batch = jobs[start:start+batch_rows]
            attempted, cached = False, False
            try:
                options = {}
                if self.cache_dir is not None:
                    from .pure_checkpoint import binding_for, batch_cache_path
                    binding = binding_for(adapt_pure_jobs(batch, models), self._collector.executor,
                                          self._collector.max_decisions)
                    path = batch_cache_path(self.cache_dir, binding)
                    receipt = self.continuation(binding, path) if callable(self.continuation) else self.continuation
                    cached = (path/'completed').exists()
                    if not cached and self.max_new_batches is not None and new_batches >= self.max_new_batches:
                        from .device_pure_segments import PureSegmentPaused
                        self._diagnostics['new_batches'] = new_batches
                        self._diagnostics['paused_before_batch'] = start
                        raise PureSegmentPaused(str(path))
                    options = dict(checkpoint_dir=path, continuation=receipt, max_segments=self.max_segments)
                attempted = True
                returned = self._collector.collect(batch, models, **options)
                if self.cache_dir is not None and not cached:
                    new_batches += 1
                    self._new_batches_in_window = new_batches
                    self._diagnostics['new_batches'] = new_batches
                results.extend(ClosedDeviceLearningContext._checked_results(self, batch, returned))
                self._diagnostics['completed_games'] += sum(r.get('complete') is True for r in returned)
            finally:
                self._diagnostics['batches'].append(dict(offset=start, jobs=len(batch),
                    attempted=attempted, cache_hit=cached,
                    diagnostic=deepcopy(getattr(self._collector, 'last_diagnostics', None)) if attempted else None))
        return results
