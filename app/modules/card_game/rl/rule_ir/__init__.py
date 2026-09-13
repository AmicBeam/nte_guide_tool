"""Shared rule IR. Combat slice and starter-card effects compile to tensor/CUDA backends."""

from .combat_slice import COMBAT_SLICE
from .compile import compile_program
from .schema import FIELDS, N_FIELDS, SLICE_VERSION
from .starter import STARTER_APPEARING, STARTER_CARDS, STARTER_HOOKS, validate_starter_ir

__all__ = [
    'COMBAT_SLICE', 'FIELDS', 'N_FIELDS', 'SLICE_VERSION', 'compile_program',
    'STARTER_APPEARING', 'STARTER_CARDS', 'STARTER_HOOKS', 'validate_starter_ir',
]
