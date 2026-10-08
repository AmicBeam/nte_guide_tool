"""Compile one IR program to CPU and optionally CUDA."""
from __future__ import annotations

from .ast import Program, validate
from .combat_slice import COMBAT_SLICE


def compile_program(program: Program | None = None, *, cuda: bool = False):
    from .lower_cpu import lower_cpu

    program = program or COMBAT_SLICE
    validate(program)
    cpu = lower_cpu(program)

    def cpu_steps(x, steps: int = 4):
        y = x
        for k in range(int(steps)):
            y = cpu(y, k % 2)
        return y

    gpu = None
    if cuda:
        from .lower_cuda import compile_cuda
        gpu = compile_cuda(program)
    return {'cpu': cpu, 'cpu_steps': cpu_steps, 'cuda': gpu, 'program': program}
