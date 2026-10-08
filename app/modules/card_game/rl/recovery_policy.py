"""Independent terminal-value tower for the five-team recovery schema.

Value-only steps cannot move policy scores. Policy still fits complete Gumbel
labels; no heuristic action preference or extra environment reward is added.
"""
from hashlib import sha256
from pathlib import Path

RESIDUAL_SCHEMA = 'cross_five_decoupled_residual_wdl_v1'
PRESERVED_SCHEMA = 'cross_five_decoupled_partial_preserved_wdl_v1'
SCHEMAS = (RESIDUAL_SCHEMA, PRESERVED_SCHEMA)


def fingerprint():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def attach(net):
    from copy import deepcopy
    import types
    import torch
    from .preserved_policy import PARTIAL_SCHEMA
    net.add_module('value_net', deepcopy(net.state_net))
    original = net.forward

    # A callable closing over a bound original method would make deepcopy
    # retain the original network. Bind the unbound method to this instance.
    method = original.__func__

    def forward(self, x, c, mask):
        scores, _ = method(self, x, c, mask)
        h = self.value_net(x)
        probabilities = self.wdl(torch.cat((h, h.new_ones((len(x), 1))), -1)).softmax(-1)
        return scores, probabilities[:, 2] - probabilities[:, 0]

    net.forward = types.MethodType(forward, net)
    net.policy_schema = PRESERVED_SCHEMA if net.policy_schema == PARTIAL_SCHEMA else RESIDUAL_SCHEMA
    return net


def extra_shapes(hidden):
    from .residual_policy import tensor_shapes
    return {name.replace('state_net.', 'value_net.', 1): shape
            for name, shape in tensor_shapes(hidden).items() if name.startswith('state_net.')}


def value_weights(weights):
    # Same visible encoding and mathematical tower, independent arrays.
    return dict(weights, **{name.replace('value_net.', 'state_net.', 1): value
                           for name, value in weights.items() if name.startswith('value_net.')})
