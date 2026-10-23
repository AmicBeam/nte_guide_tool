"""Per-root temporary storage for serial physical rules, outside game state.

Queries finish before callbacks, and damage IDs are consumed before damage
callbacks. Neither buffer is live across recursive rule execution. Immutable
parallel query kernels deliberately retain their own per-thread storage.
"""


def query_capacity(layout):
    return 2*(layout.max_characters+1)*(layout.max_effects+layout.max_timed_attack+8)+2*layout.max_characters


def prepare_workspace(torch, layout, rows, device):
    capacity=query_capacity(layout)
    # Candidate is {double value; int layer, op}; float64 provides alignment.
    return (torch.empty((rows,2*capacity),dtype=torch.float64,device=device),
            torch.empty((rows,capacity),dtype=torch.int32,device=device))


def use_query_workspace(source):
    old='Candidate a[MAXQ];int n=0;'
    if source.count(old)!=1:
        raise ValueError('Physical query workspace binding requires one query body')
    return source.replace(old,'Candidate* a=(Candidate*)f->query_scratch;int n=0;')
