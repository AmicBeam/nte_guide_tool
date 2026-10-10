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
