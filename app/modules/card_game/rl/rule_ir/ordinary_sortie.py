"""Common ordinary-sortie IR. Optional dispatcher rejects special mechanics.

Combat, entry and resource stores live here once, for CPU and CUDA emitters.
Inputs are side-normalized: side=0 is the attacker. Launch exactly one step.
"""
from .ast import Program, Bind, Store, Name as N, Const as C, Bin, Cmp, Select
from .combat_slice import BINDS, STORES, SKIP
from .schema import FIELDS, SIDE_ALIASES


def ordinary_program(seats):
    fields = FIELDS + ('actor', 'old_front', 'last_front', 'old_front_shield', 'harmony')
    fields += tuple(f'{prefix}_{i}' for i in range(seats) for prefix in ('alive', 'energy', 'cap'))
    binds = BINDS + (Bind('moved', Cmp('ne', N('old_front'), N('actor'))),)
    stores = STORES + (
        Store('last_front', Select(N('moved'), C(-1), N('last_front'))),
        Store('old_front_shield', Select(N('moved'), C(0), N('old_front_shield'))),
        Store('harmony', Select(N('over'), N('harmony'), Bin('min', Bin('add', N('harmony'), C(1)), C(2)))),
    )
    for i in range(seats):
        add = Bin('add', C(1), Cmp('eq', N('actor'), C(i)))
        stores += (Store(f'energy_{i}', Select(Bin('and', N(f'alive_{i}'), Cmp('eq', N('over'), C(0))),
                      Bin('min', Bin('add', N(f'energy_{i}'), add), N(f'cap_{i}')), N(f'energy_{i}'))),)
    return Program('ordinary_sortie_common', 'ordinary_common_v1', SKIP, binds, stores, fields, SIDE_ALIASES)
