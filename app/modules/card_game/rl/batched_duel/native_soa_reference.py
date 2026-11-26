"""Explicit CPU oracle driver over persistent typed NativeSoA columns.

Only ingress/egress and numerical model evaluation run in Python. All game
transitions, legal actions, public projection and chance transitions call the
compiled source. This host driver is for offline software parity, never GPU
throughput or a closed-device training service. No Python engine fallback.
"""
import types
import numpy as np

from .record_soa import RecordSoABatch, RecordSoAError, _DecodeRow
from .record_soa_compact import compact_rows

_METADATA = ('phase', 'winner', 'turn', 'version')


class StateHandle:
    """Typed persistent world; root scalars only, never a Game dict graph."""

    __slots__ = ('batch',)

    def __init__(self, batch):
        if not isinstance(batch, RecordSoABatch) or batch.rows != 1:
            raise RecordSoAError('StateHandle requires a single typed row')
        self.batch = batch

    def _scalar(self, name):
        schema, batch = self.batch.schema, self.batch
        pool, slot = schema.locate(int(batch.roots[0]))
        if pool.index != schema.root:
            raise RecordSoAError('StateHandle root is not a Game record')
        field = next(f for f in pool.fields if f.name == pool.name + '.' + name)
        return _DecodeRow(batch, 0).value(field, slot)

    def __getitem__(self, key):
        if key not in _METADATA:
            raise KeyError('StateHandle allows only phase/winner/turn/version')
        return self._scalar(key)

    def get(self, key, default=None):
        if key not in _METADATA:
            return default
        value = self._scalar(key)
        return default if value is None else value


class NativeSoAReference:
    supports_full_duel_rules = False
    gpu_qualified = False
    host_rule_callbacks = False

    def __init__(self, executor, *, max_batch_rows=8, schema=None):
        if not callable(getattr(executor, 'call', None)):
            raise TypeError('Compiled native SoA CPU executor required')
        self.executor = executor
        if type(max_batch_rows) is not int or max_batch_rows < 1:
            raise ValueError('Positive native reference batch bound required')
        self.max_batch_rows = max_batch_rows
        self.schema = schema if schema is not None else getattr(executor, 'schema', None)
        self.dispatches = {}
        self.rows = {}

    def _invoke(self, entry, state, arguments, state_positions=(0,)):
        if not isinstance(state, RecordSoABatch):
            raise RecordSoAError('Matching persistent domain columns required')
        if state.rows != len(arguments) or state.rows > self.max_batch_rows:
            raise RecordSoAError('Bounded matching typed argument batch required')
        for row in arguments:
            if any(isinstance(value, dict) and {'sides', 'rng', 'next_entity'} <= set(value)
                   for value in row):
                raise RecordSoAError('Game graph ingress is forbidden in persistent rule calls')
        self.rows[entry] = self.rows.get(entry, 0) + state.rows
        self.dispatches[entry] = self.dispatches.get(entry, 0) + 1
        return self.executor.call(entry, state, arguments, state_positions=state_positions)

    def call(self, entry, state, arguments, *, state_positions=(0,)):
        scratch = state.clone_rows(list(range(state.rows)))
        return self._invoke(entry, scratch, arguments, state_positions)

    def _scratch(self, handle):
        return handle.batch.clone_rows([0])

    def _commit(self, result, path=()):
        return StateHandle(compact_rows(result.select_state(path)))

    def initialize(self, parameters):
        if self.schema is None:
            raise RecordSoAError('Compiled domain schema required')
        output = []
        for start in range(0, len(parameters), self.max_batch_rows):
            chunk = parameters[start:start + self.max_batch_rows]
            blank = self.schema.allocate(len(chunk))
            result = self._invoke('initialize', blank, [[p] for p in chunk], ())
            selected = compact_rows(result.select_state())
            output.extend(StateHandle(selected.clone_rows([i])) for i in range(selected.rows))
        return output

    def actor(self, state):
        return self._invoke('acting_side', self._scratch(state), [[None]]).decode()[0]

    def public(self, state, observer):
        actions, view = self._invoke(
            'public_legal_decision', self._scratch(state), [[None, observer]]).decode()[0]
        expected = [entry['action'] for entry in view['legal_actions']]
        if actions != expected:
            raise ValueError('Native full legal ordering disagrees with native public view')
        return actions, view

    def decision(self, state, side):
        from ..seven_lineup import encode
        actions, view = self.public(state, side)
        actions = [a for a in actions if a['type'] != 'concede']
        return actions, encode(view, actions)

    def sample(self, state, viewer, seed):
        from ..seven_lineup import encode
        packet = self._invoke(
            'sample_world_guard_views_packet', self._scratch(state), [[None, viewer, seed]])
        world = self._commit(packet, (0,))
        for before, after in packet.decode(path=(1,))[0]:
            left_actions, left_view = before
            right_actions, right_view = after
            x, c = encode(left_view, left_actions)
            xx, cc = encode(right_view, right_actions)
            if not np.array_equal(x, xx) or not np.array_equal(c, cc):
                raise ValueError('Native sampled world changed complete information encoding')
        return world

    def determinize(self, state, viewer, seed):
        return self.sample(state, viewer, seed)

    def apply(self, state, side, action):
        return self._commit(self._invoke('apply_action', self._scratch(state), [[None, side, action]]))

    def simulate(self, state, side, action):
        return self._commit(self._invoke('simulate_action', self._scratch(state), [[None, side, action]]))

    def clean(self, state):
        return self._commit(self._invoke('clean', self._scratch(state), [[None]]))

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
        """Original Gumbel implementation with isolated native typed bindings."""
        from .. import recovery_search, information_search
        if options.get('teacher_mode', 'network') != 'network' or options.get('terminal_horizon', 0) != 0:
            raise ValueError('Native SoA reference has no teacher continuations')
        options['teacher_mode'] = 'network'
        options['terminal_horizon'] = 0
        options['runtime'] = self
        core = types.SimpleNamespace(**vars(information_search))
        core.acting_side = self.actor
        core.apply_action = self.apply
        core.simulate_action = self.simulate
        core.clean = self.clean
        core.sample_world = lambda world, side, seed, **kw: self.sample(world, side, seed)
        def leaf_value(world, side, policies, **kw):
            if world['phase'] == 'finished':
                return information_search.terminal_value(world, side)
            return self.value_for(world, side, policies)
        core.leaf_value = leaf_value
        def no_teacher(*args, **kw):
            raise RuntimeError('Teacher continuation is absent from native SoA search')
        core.short_terminal_value = no_teacher
        original = recovery_search.search
        namespace = dict(original.__globals__, core=core)
        prepared = types.FunctionType(original.__code__, namespace, original.__name__,
                                      original.__defaults__, original.__closure__)
        prepared.__kwdefaults__ = dict(original.__kwdefaults__ or {})
        return prepared(state, viewer, policies, **options)
