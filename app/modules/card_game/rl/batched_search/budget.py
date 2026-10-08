"""Single-owner per-root accounting for future batched search.

Reservations include queued work. Call ``complete`` only after the device event
has completed; enqueueing is not a completed simulation. This component neither
implements Gumbel selection nor claims a hard real-time GPU deadline.
"""
from dataclasses import dataclass
import math
import time
from uuid import uuid4


class BudgetExhausted(RuntimeError):
    """The root is closed, expired, or has no remaining work allowance."""


@dataclass(frozen=True)
class SearchLimits:
    seconds: float = 3.0
    simulations: int = 32
    proof_cap: int = 8
    gumbel_candidates: int = 16
    terminal_horizon: int = 3
    max_depth: int = 10

    def __post_init__(self):
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)) or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('Positive finite decision seconds required')
        for name in ('simulations', 'gumbel_candidates', 'max_depth'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer required: ' + name)
        for name in ('proof_cap', 'terminal_horizon'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('Nonnegative integer required: ' + name)
        if self.proof_cap >= self.simulations:
            raise ValueError('Proof allowance must leave at least one tree simulation')


@dataclass(frozen=True)
class SimulationTicket:
    budget_id: str
    game_id: str
    root_generation: int
    sequence: int
    kind: str


class DecisionBudget:
    """Own one root on one scheduler thread; not a thread-safe shared counter.

    Only one simulation may be outstanding per root, retaining the original
    sequential visit-update dependency. Concurrency comes from different roots.
    Aborted reservations are not refunded, so cancellations cannot oversubscribe
    the proof-plus-tree allowance. A fresh decision needs a fresh budget object.
    """

    def __init__(self, game_id, root_generation, *, limits=None, clock=time.monotonic, started_at=None):
        if not isinstance(game_id, str) or not game_id:
            raise ValueError('Nonempty game identity required')
        if type(root_generation) is not int or root_generation < 1:
            raise ValueError('Positive root generation required')
        self.limits = SearchLimits() if limits is None else limits
        if not isinstance(self.limits, SearchLimits):
            raise TypeError('SearchLimits required')
        if not callable(clock):
            raise TypeError('Clock must be callable')
        self._clock = clock
        self.game_id = game_id
        self.root_generation = root_generation
        self.started_at = self._finite_time(clock() if started_at is None else started_at)
        self.deadline = self.started_at + self.limits.seconds
        if not math.isfinite(self.deadline):
            raise ValueError('Finite deadline required')
        self._last_time = self.started_at
        self._identity = uuid4().hex
        self._pending = {}
        self._reserved = {'proof': 0, 'tree': 0}
        self._started = {'proof': 0, 'tree': 0}
        self._completed = {'proof': 0, 'tree': 0}
        self._aborted = 0
        self._discarded = 0
        self._closed = False

    @staticmethod
    def _finite_time(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite clock value required')
        return float(value)

    def _now(self):
        now = self._finite_time(self._clock())
        if now < self._last_time:
            raise ValueError('Decision clock moved backwards')
        self._last_time = now
        return now

    def _open(self):
        if self._closed or self._now() >= self.deadline:
            raise BudgetExhausted('Root closed or expired')

    def _lookup(self, ticket):
        if not isinstance(ticket, SimulationTicket) or ticket.budget_id != self._identity:
            raise ValueError('Foreign simulation ticket')
        if type(ticket.sequence) is not int or type(ticket.root_generation) is not int:
            raise ValueError('Ticket sequence and generation must be integers, not bool/float')
        record = self._pending.get(ticket.sequence)
        if record is None or record['ticket'] != ticket:
            raise ValueError('Unknown, altered or already consumed ticket')
        return record

    def reserve(self, kind):
        if kind not in ('proof', 'tree'):
            raise ValueError('Simulation kind must be proof or tree')
        self._open()
        if self._pending:
            raise RuntimeError('Finish or abort the current simulation before reserving another')
        reserved = sum(self._reserved.values())
        if reserved >= self.limits.simulations:
            raise BudgetExhausted('Combined simulation allowance exhausted')
        if kind == 'proof':
            if self._reserved['tree']:
                raise RuntimeError('Proof checks must precede tree simulations')
            if self._reserved['proof'] >= self.limits.proof_cap:
                raise BudgetExhausted('Proof allowance exhausted')
        ticket = SimulationTicket(self._identity, self.game_id, self.root_generation, reserved, kind)
        self._reserved[kind] += 1
        self._pending[ticket.sequence] = {'ticket': ticket, 'started': False}
        return ticket

    def mark_started(self, ticket):
        record = self._lookup(ticket)
        self._open()
        if record['started']:
            raise ValueError('Simulation already started')
        record['started'] = True
        self._started[ticket.kind] += 1

    def complete(self, ticket):
        """Return whether a device-completed result still belongs to a live root."""
        record = self._lookup(ticket)
        if not record['started']:
            raise ValueError('Unstarted work cannot be completed')
        accepted = not self._closed and self._now() < self.deadline
        del self._pending[ticket.sequence]
        if accepted:
            self._completed[ticket.kind] += 1
        else:
            self._discarded += 1
        return accepted

    def abort(self, ticket):
        self._lookup(ticket)
        del self._pending[ticket.sequence]
        self._aborted += 1

    def close(self):
        """Stop new admission; outstanding GPU completions will be rejected."""
        self._closed = True

    def snapshot(self):
        return dict(game_id=self.game_id, root_generation=self.root_generation,
                    started_at=self.started_at, deadline=self.deadline,
                    reserved=dict(self._reserved), started=dict(self._started),
                    completed=dict(self._completed), aborted=self._aborted,
                    discarded=self._discarded, pending=len(self._pending),
                    closed=self._closed, expired=self._now() >= self.deadline)
