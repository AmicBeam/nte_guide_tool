"""Qualified device collection boundary for the existing seven-team trainer.

No executor is manufactured here. The real driver supplies complete CUDA
episodes and frozen evaluation records; this context refuses CPU rule maps.
Numerical updates, selection and checkpoint semantics remain in seven_round.
"""


class ClosedDeviceLearningContext:
    def __init__(self, driver):
        from .device_execution_contract import require_closed_device_driver
        self.driver = require_closed_device_driver(driver, purpose='training')
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
        return self._checked_results(jobs, self.driver.collect(jobs))

    def evaluate_jobs(self, jobs):
        return self._checked_results(jobs, self.driver.evaluate_jobs(jobs))


class DeviceValueCalibrationContext:
    """Live argmax-only bootstrap for warm_value, without Gumbel approval.

    The real GPU sampler is deliberately separate from formal policy learning.
    It cannot run search jobs, comparison telemetry or host rule callbacks.
    """
    def __init__(self, collector):
        from .device_pure_collection import DevicePureCollector
        if not isinstance(collector, DevicePureCollector):
            raise TypeError('Live CUDA pure collector required for value calibration')
        self._collector = collector
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
        for start in range(0, len(jobs), 32):
            batch = jobs[start:start+32]
            try:
                returned = self._collector.collect(batch, models)
                results.extend(ClosedDeviceLearningContext._checked_results(self, batch, returned))
                self._diagnostics['completed_games'] += sum(r.get('complete') is True for r in returned)
            finally:
                self._diagnostics['batches'].append(dict(offset=start, jobs=len(batch),
                    diagnostic=deepcopy(getattr(self._collector, 'last_diagnostics', None))))
        return results
