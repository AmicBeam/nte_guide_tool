"""Isolated rule fixtures committing through the native array backend.

Tests retain Python setup/oracles. Only their local apply_action binding is
replaced, never a production module. Every successful native commit is checked
against formal full state before returning it to the fixture. Host mock hooks
and out-of-scope character fixtures are reported as excluded, not coverage.
"""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types
import unittest


MODULES = (
    'test_duel_v2_surplus_rework', 'test_duel_v2_heiyu',
    'test_duel_v2_dafutier', 'test_duel_v2_lingke',
    'test_duel_v2_xiaozhi_haiyue', 'test_duel_v2_zone_status',
    'test_duel_v2_continuous_cast', 'test_duel_v2_collapse_harmony',
    'test_duel_v2_engine', 'test_duel_v2_card_polish',
    'test_duel_v2_card_use_timing', 'test_duel_v2_mimic_generated_choices',
    'test_duel_v2_turn_boundaries', 'test_duel_v2_jiuyuan_rework',
    'test_duel_v2_iloy_knockdown', 'test_duel_v2_auto_release',
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def run_rule_conformance(runtime, output, *, modules=MODULES, budget=None):
    from ...content.duel_v2 import PUBLIC_CHARACTER_IDS
    from ...engine.duel_v2 import apply_action as formal_apply
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[5]
    records = []
    commits = []
    rejected = []
    active_test = [None]
    def native_apply(state, side, action):
        if budget is not None:
            budget()
        characters = {cid for team in state['sides'].values() for cid, h in team['characters'].items()
                      if not h.get('summoned')}
        if characters - set(PUBLIC_CHARACTER_IDS):
            raise unittest.SkipTest('Fixture uses characters outside native public-seven scope')
        before = deepcopy(state)
        packet = runtime.call('checked_apply_action', [[state, side, action]])[0]
        if not packet[0]:
            rejected.append(dict(test=active_test[0], side=side, action=action, error=packet[1:]))
            cls = {'ValueError': ValueError, 'TypeError': TypeError}.get(packet[1])
            if cls is None:
                raise AssertionError('Unknown native checked exception')
            if state != before or _digest(state) != _digest(before):
                raise AssertionError('Native rejection changed source fixture')
            raise cls(*packet[2])
        actual = packet[1]
        expected = formal_apply(state, side, action)
        if actual != expected or _digest(actual) != _digest(expected):
            mismatch = output/('mismatch-'+str(len(records))+'.json')
            mismatch.write_text(json.dumps(dict(test=active_test[0], side=side, action=action,
                before=before, actual=actual, expected=expected), ensure_ascii=False, indent=2), encoding='utf8')
            raise AssertionError(['native fixture full state', action, str(mismatch)])
        if state != before or _digest(state) != _digest(before):
            raise AssertionError('Native commit changed source fixture')
        commits.append(dict(test=active_test[0], side=side, action=action,
                            state_sha256=_digest(actual)))
        return actual
    class Result(unittest.TextTestResult):
        def startTest(self, test):
            if budget is not None:
                budget()
            active_test[0] = test.id()
            self._before_commits = len(commits)
            super().startTest(test)
        def stopTest(self, test):
            records.append(dict(test=test.id(), native_commits=len(commits)-self._before_commits))
            active_test[0] = None
            super().stopTest(test)
    suite = unittest.TestSuite()
    registered = []
    try:
        for name in modules:
            path = repo/'tests'/(name+'.py')
            module_name = 'native_array_fixture_'+name
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            sys.modules[module_name] = module
            registered.append(module_name)
            exec(compile(path.read_text(encoding='utf8'), str(path), 'exec'), module.__dict__)
            for alias, value in list(vars(module).items()):
                if value is formal_apply:
                    setattr(module, alias, native_apply)
            loaded = unittest.defaultTestLoader.loadTestsFromModule(module)
            def uses_native(fn, cls, seen=None):
                fn = getattr(fn, '__func__', fn)
                if not inspect.isfunction(fn) or fn.__globals__ is not module.__dict__:
                    return False
                seen = set() if seen is None else seen
                if id(fn) in seen:
                    return False
                seen.add(id(fn))
                codes = [fn.__code__]
                names = set()
                while codes:
                    code = codes.pop()
                    names.update(code.co_names)
                    codes.extend(item for item in code.co_consts if inspect.iscode(item))
                for key in names:
                    value = fn.__globals__.get(key)
                    if value is native_apply:
                        return True
                    if uses_native(value, cls, seen):
                        return True
                    if uses_native(getattr(cls, key, None), cls, seen):
                        return True
                return False
            def add(tests):
                for test in tests:
                    if isinstance(test, unittest.TestSuite):
                        add(test)
                    else:
                        method = getattr(test, test._testMethodName)
                        source = inspect.getsource(method)
                        if 'patch.object' in source or 'with patch(' in source:
                            def excluded(*args):
                                raise unittest.SkipTest('Host mock instrumentation is not a native rule callback')
                            setattr(test, test._testMethodName, excluded)
                        elif not uses_native(method, type(test)):
                            def no_commit(*args):
                                raise unittest.SkipTest('Fixture has no routed native physical commit')
                            setattr(test, test._testMethodName, no_commit)
                        suite.addTest(test)
            add(loaded)
        with (output/'tests.log').open('w', encoding='utf8') as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Result).run(suite)
        report = dict(scope='rule fixtures with native physical commits and independent Python oracles',
            passed=result.wasSuccessful(), tests=result.testsRun, skipped=len(result.skipped),
            failures=len(result.failures), errors=len(result.errors),
            native_commits=len(commits), native_rejections=len(rejected),
            tests_with_native_commits=sum(row['native_commits'] > 0 for row in records),
            excluded=[dict(test=test.id(), reason=reason) for test, reason in result.skipped],
            rows=records, gpu_qualified=False)
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        (output/'commits.json').write_text(json.dumps(commits, ensure_ascii=False, indent=2), encoding='utf8')
        (output/'rejections.json').write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding='utf8')
        return report
    finally:
        for name in registered:
            sys.modules.pop(name, None)
