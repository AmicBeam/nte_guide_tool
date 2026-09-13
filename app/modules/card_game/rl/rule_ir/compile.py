"""Compile one IR program to CPU and optionally CUDA."""
from __future__ import annotations

from .ast import Program, validate
from .combat_slice import COMBAT_SLICE
from .lower_cpu import lower_cpu


def compile_program(program: Program | None = None, *, cuda: bool = False):
    program = program or COMBAT_SLICE
    validate(program)
    cpu = lower_cpu(program)
    gpu = None
    if cuda:
        from .lower_cuda import compile_cuda
        gpu = compile_cuda(program)
    return {'cpu': cpu, 'cuda': gpu, 'program': program}
