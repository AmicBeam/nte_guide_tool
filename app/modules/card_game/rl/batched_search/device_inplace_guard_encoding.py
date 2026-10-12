"""Source-derived private encoding with two fresh-output allocations removed.

The original encoder and globals stay untouched. Only the final multiply and
invalid-row where of freshly constructed output tensors become in-place.
All validation, equations, candidate order, dtype and error masks stay put.
"""
import ast
import inspect
from hashlib import sha256
from ..batched_duel import gpu_seven_candidates as encoder

_PREPARED = None


def _rewrite(source, needle, replacement):
    tree=ast.parse(source)
    expected=ast.dump(ast.parse(needle).body[0],include_attributes=False)
    changed=0
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal changed
            if ast.dump(node,include_attributes=False)==expected:
                changed+=1
                return ast.copy_location(ast.parse(replacement).body[0],node)
            return self.generic_visit(node)
    tree=Rewrite().visit(tree)
    if changed!=1:raise RuntimeError('Unique final fresh candidate output assignment required')
    return ast.fix_missing_locations(tree)


def prepare_inplace_guard_encoding():
    global _PREPARED
    if _PREPARED is not None:return _PREPARED[1]
    t=encoder._require_torch()
    if t.cuda.is_available() and t.cuda.is_current_stream_capturing():
        raise RuntimeError('Prepare source-derived guard encoding before capture')
    private=inspect.getsource(encoder._encode_token_arrays)
    public=inspect.getsource(encoder.encode_tokens_device)
    namespace=dict(encoder.__dict__)
    first=_rewrite(private,'candidates = candidates * mask.unsqueeze(-1).float()',
                   'candidates.mul_(mask.unsqueeze(-1).float())')
    second=_rewrite(public,'candidates = t.where(output_good[:, None, None], candidates, 0.)',
                    'candidates.masked_fill_(~output_good[:, None, None], 0.)')
    exec(compile(first,encoder.__file__,'exec'),namespace)
    exec(compile(second,encoder.__file__,'exec'),namespace)
    identity=sha256((private+public+ast.dump(first)+ast.dump(second)).encode()).hexdigest()
    _PREPARED=(namespace['encode_tokens_device'],identity)
    return identity


def encode_guard_tokens_device(*args,**kwargs):
    if _PREPARED is None:raise RuntimeError('Guard encoder must be prepared outside capture')
    return _PREPARED[0](*args,**kwargs)
