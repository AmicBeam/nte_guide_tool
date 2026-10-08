"""Read-only numeric adaptation of the failed archive for diagnosis only.

Historical rule identity is retained. This adapter is not a warm start,
training source, or website loader. Layout and tensor checks are strict.
"""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from . import cross_grounded
from .cross_lineup import all_features, candidate_names
from .cross_runtime import CrossModel
from .league_schema import build_hash


class ArchivedDiagnosticModel(CrossModel):
    def __init__(self, directory, key):
        root = Path(directory); path = root / f'{key}.npz'
        self.manifest = m = json.loads((root / f'{key}.json').read_text())
        self.version = sha256(path.read_bytes()).hexdigest(); self.hidden = m['hidden']
        if (m.get('schema') != cross_grounded.SCHEMA or m.get('sha256') != self.version
                or m.get('deck') != key or not m.get('rule_hash')
                or m.get('features') != all_features() or m.get('candidates') != candidate_names()
                or m.get('candidate_transform_sha256') != cross_grounded.fingerprint()
                or m.get('wdl_order') != ['loss', 'draw', 'win']
                or m.get('value_context') != ['viewer_is_actor']
                or m.get('automatic_serving_approval') is not False
                or type(self.hidden) is not int or not 1 <= self.hidden <= 256):
            raise ValueError('Historical diagnostic layout/identity mismatch')
        # Match the exact legacy architecture; no missing tensors are filled.
        h = self.hidden
        expected = {
            'state_net.0.weight': (h, len(all_features())), 'state_net.0.bias': (h,),
            'state_net.2.weight': (h, h), 'state_net.2.bias': (h,),
            'cand_net.0.weight': (h, cross_grounded.DIM), 'cand_net.0.bias': (h,),
            'cand_net.2.weight': (h, h), 'cand_net.2.bias': (h,),
            'score.weight': (1, h), 'score.bias': (1,),
            'value.0.weight': (h, h), 'value.0.bias': (h,),
            'value.2.weight': (1, h), 'value.2.bias': (1,),
            'wdl.weight': (3, h + 1), 'wdl.bias': (3,),
        }
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError('Historical diagnostic tensor set mismatch')
            self.weights = {name: arrays[name].copy() for name in expected}
        for name, shape in expected.items():
            if self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all():
                raise ValueError('Invalid historical diagnostic tensor')
            self.weights[name].flags.writeable = False
        self.serving_deck = m['build']
        if build_hash(self.serving_deck) != m.get('build_sha256'):
            raise ValueError('Historical diagnostic build mismatch')
        self.historical_rule_hash = m['rule_hash']
        self.diagnostic_only = True

    def validate_human(self, build):
        raise ValueError('Failed archive is diagnostic-only')

    def wdl(self, x, is_actor):
        probabilities = super().wdl(x, is_actor)
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.):
            raise ValueError('Non-finite historical diagnostic WDL')
        return probabilities

    def wdl_checked_float64(self, x, is_actor):
        """Independent stable arithmetic for auditing BLAS warnings, not a silent replacement."""
        def linear(value, name):
            return np.einsum('ij,j->i', self.weights[name + '.weight'].astype(np.float64), value,
                             optimize=False) + self.weights[name + '.bias'].astype(np.float64)
        h = np.maximum(linear(np.asarray(x, dtype=np.float64), 'state_net.0'), 0)
        h = np.maximum(linear(h, 'state_net.2'), 0)
        logits = linear(np.append(h, float(bool(is_actor))), 'wdl')
        if not np.isfinite(logits).all():
            raise ValueError('Non-finite independent float64 WDL')
        p = np.exp(logits - logits.max())
        return p / p.sum()
