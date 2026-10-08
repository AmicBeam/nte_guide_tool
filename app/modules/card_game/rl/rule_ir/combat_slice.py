"""Single combat-slice rule program: front-priority, simultaneous hits, then knockdown/victory."""
from __future__ import annotations

from .ast import Bind, ClampMin, Cmp, Const, Name, Program, Select, Store, Bin
from .schema import SLICE_VERSION

N = Name
C = Const


def _and(*xs):
    acc = xs[0]
    for item in xs[1:]:
        acc = Bin('and', acc, item)
    return acc


def _or(*xs):
    acc = xs[0]
    for item in xs[1:]:
        acc = Bin('or', acc, item)
    return acc


# Skip if finished, attacker has no HP, or attacker is not in front.
SKIP = _or(Cmp('ne', N('done'), C(0)), Cmp('le', N('ah'), C(0)), Cmp('eq', N('af'), C(0)))

BINDS = (
    Bind('has_front', Cmp('ne', N('df'), C(0))),
    Bind('amount', ClampMin(Bin('add', N('atk'), N('bonus')), 0)),
    Bind('counter', Select(_and(N('has_front'), Cmp('eq', N('no_counter'), C(0))), N('datk'), C(0))),
    Bind('th', Select(N('has_front'), N('dh'), N('dphp'))),
    Bind('ts', Select(N('has_front'), N('dsh'), N('dpsh'))),
    Bind('absorbed', Bin('min', N('ts'), N('amount'))),
    Bind('nh', ClampMin(Bin('sub', N('th'), Bin('sub', N('amount'), N('absorbed'))), 0)),
    Bind('ns', Bin('sub', N('ts'), N('absorbed'))),
    Bind('self_abs', Bin('min', N('ash'), N('counter'))),
    Bind('self_hp', ClampMin(Bin('sub', N('ah'), Bin('sub', N('counter'), N('self_abs'))), 0)),
    Bind('self_sh', Bin('sub', N('ash'), N('self_abs'))),
    Bind('overflow', Select(
        _and(N('has_front'), Cmp('ne', N('penetrate'), C(0))),
        ClampMin(Bin('sub', Bin('sub', N('amount'), N('ts')), N('th')), 0),
        C(0),
    )),
    Bind('pabs', Bin('min', N('dpsh'), N('overflow'))),
    Bind('php', Select(
        N('has_front'),
        ClampMin(Bin('sub', N('dphp'), Bin('sub', N('overflow'), N('pabs'))), 0),
        N('nh'),
    )),
    Bind('psh', Select(N('has_front'), Bin('sub', N('dpsh'), N('pabs')), N('ns'))),
    Bind('dead_a', Cmp('eq', N('self_hp'), C(0))),
    Bind('dead_d', _and(N('has_front'), Cmp('eq', N('nh'), C(0)))),
    Bind('new_pa', Select(Cmp('eq', N('side'), C(1)), N('php'), N('pa_hp'))),
    Bind('new_pb', Select(Cmp('eq', N('side'), C(0)), N('php'), N('pb_hp'))),
    Bind('over', _or(Cmp('le', N('new_pa'), C(0)), Cmp('le', N('new_pb'), C(0)))),
)

STORES = (
    Store('ah', N('self_hp')),
    Store('ash', Select(N('dead_a'), C(0), N('self_sh'))),
    Store('dh', Select(N('has_front'), N('nh'), N('dh'))),
    Store('dsh', Select(N('dead_d'), C(0), Select(N('has_front'), N('ns'), N('dsh')))),
    Store('dphp', N('php')),
    Store('dpsh', N('psh')),
    Store('af', Select(N('dead_a'), C(0), N('af'))),
    Store('df', Select(N('dead_d'), C(0), N('df'))),
    Store('ad', Select(N('dead_a'), C(3), N('ad'))),
    Store('dd', Select(N('dead_d'), C(3), N('dd'))),
    Store('done', Select(N('over'), C(1), N('done'))),
    Store('winner', Select(
        N('over'),
        Select(
            _and(Cmp('le', N('new_pa'), C(0)), Cmp('le', N('new_pb'), C(0))),
            C(2),
            Select(Cmp('le', N('new_pa'), C(0)), C(1), C(0)),
        ),
        N('winner'),
    )),
)

COMBAT_SLICE = Program(
    name='combat_zero_vs_zero',
    version=SLICE_VERSION,
    skip_when=SKIP,
    binds=BINDS,
    stores=STORES,
)
