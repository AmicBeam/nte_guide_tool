"""Explicit CPU oracle driver over the compiled native array rules.

Only ingress/egress and numerical model evaluation run in Python. All game
transitions, legal actions, public projection and chance transitions call the
compiled source. This host driver is for offline parity, never GPU throughput
or a closed-device training service. No Python engine fallback is available.
"""
import types
import numpy as np


class NativeArrayReference:
    supports_full_duel_rules = False
    gpu_qualified = False
    host_rule_callbacks = False

    def __init__(self, executor, *, max_batch_rows=8):
        if not callable(getattr(executor, 'call', None)):
            raise TypeError('Compiled native CPU executor required')
        self.executor = executor
        if type(max_batch_rows) is not int or max_batch_rows < 1:
            raise ValueError('Positive native reference batch bound required')
        self.max_batch_rows = max_batch_rows
        self.dispatches = {}
        self.rows = {}

    def call(self, entry, arguments):
        self.rows[entry] = self.rows.get(entry, 0) + len(arguments)
        output = []
        for start in range(0, len(arguments), self.max_batch_rows):
            self.dispatches[entry] = self.dispatches.get(entry, 0) + 1
            output.extend(self.executor.call(entry, arguments[start:start+self.max_batch_rows]))
        return output

    def initialize(self, parameters):
        return self.call('initialize', [[p] for p in parameters])

    def actor(self, state):
        return self.call('acting_side', [[state]])[0]

    def public(self, state, observer):
        actions, view = self.call('public_legal_decision', [[state, observer]])[0]
        expected = [entry['action'] for entry in view['legal_actions']]
        if actions != expected:
            raise ValueError('Native full legal ordering disagrees with native public view')
        return actions, view

    def decision(self, state, side):
        from ..seven_lineup import encode
        actions, view = self.public(state, side)
        actions = [a for a in actions if a['type'] != 'concede']
        return actions, encode(view, actions)

    def determinize(self, state, viewer, seed):
        from ..seven_lineup import encode
        world, guards = self.call('sample_world_guard_views_packet', [[state, viewer, seed]])[0]
        for before, after in guards:
            left_actions, left_view = before
            right_actions, right_view = after
            x, c = encode(left_view, left_actions)
            xx, cc = encode(right_view, right_actions)
            if not np.array_equal(x, xx) or not np.array_equal(c, cc):
                raise ValueError('Native sampled world changed complete information encoding')
        return world

    def apply(self, state, side, action):
        return self.call('apply_action', [[state, side, action]])[0]

    def simulate(self, state, side, action):
        return self.call('simulate_action', [[state, side, action]])[0]

    def clean(self, state):
        return self.call('clean', [[state]])[0]

    def value_for(self, state, viewer, policies):
        from ..seven_lineup import encode
        _, view = self.public(state, viewer)
        x, _ = encode(view, [])
        p = policies[viewer].wdl(x, self.actor(state) == viewer)
        return float(p[2] - p[0])

    @staticmethod
    def predict_encoded(policy, x, c):
        return policy.scores_value(x, c)

    @staticmethod
    def action_scores(policy, x, c):
        return policy.scores(x, c)

    def search(self, state, viewer, policies, **options):
        """Original Gumbel implementation with isolated native rule bindings.

        Mathematical helpers, information keys, visit budgets and RNG remain
        original. No global module or function is patched, including tests
        running concurrently against the canonical Python oracle.
        """
        from .. import recovery_search, information_search
        if options.get('teacher_mode', 'network') != 'network' or options.get('terminal_horizon', 0) != 0:
            raise ValueError('Native array reference has no teacher continuations')
        options['teacher_mode'] = 'network'
        options['terminal_horizon'] = 0
        options['runtime'] = self
        core = types.SimpleNamespace(**vars(information_search))
        core.acting_side = self.actor
        core.apply_action = self.apply
        core.simulate_action = self.simulate
        core.clean = self.clean
        core.sample_world = lambda state, side, seed, **kw: self.determinize(state, side, seed)
        def leaf_value(state, side, policies, **kw):
            if state['phase'] == 'finished':
                return information_search.terminal_value(state, side)
            return self.value_for(state, side, policies)
        core.leaf_value = leaf_value
        def no_teacher(*args, **kw):
            raise RuntimeError('Teacher continuation is absent from native array search')
        core.short_terminal_value = no_teacher
        original = recovery_search.search
        namespace = dict(original.__globals__, core=core)
        prepared = types.FunctionType(original.__code__, namespace, original.__name__,
                                      original.__defaults__, original.__closure__)
        prepared.__kwdefaults__ = dict(original.__kwdefaults__ or {})
        return prepared(state, viewer, policies, **options)
