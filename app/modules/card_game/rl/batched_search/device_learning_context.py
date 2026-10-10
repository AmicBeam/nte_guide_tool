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
        return self.driver

    def evaluate_jobs(self, jobs):
        results = list(self.driver.evaluate_jobs(jobs))
        if len(results) != len(jobs):
            raise ValueError('Closed device evaluation omitted or duplicated jobs')
        return results
