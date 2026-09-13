"""Shared combat-slice IR. CPU and CUDA are generated; Python engine stays oracle."""

from .combat_slice import COMBAT_SLICE
from .compile import compile_program
from .schema import FIELDS, N_FIELDS, SLICE_VERSION

__all__ = ['COMBAT_SLICE', 'FIELDS', 'N_FIELDS', 'SLICE_VERSION', 'compile_program']
