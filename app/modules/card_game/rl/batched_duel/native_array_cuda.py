"""Native compiled control flow over resident state arrays, CUDA only.

Reuses checked allocation, leases, kernel ABI and compaction from CudaVm;
execution source and instruction storage are replaced. The portable instruction
image is compile-time input. No opcode stream is uploaded for execution and
there is no interpreter fallback. Actual device parity remains a separate gate.
"""
import numpy as np
from .cuda_vm import CudaVm


class CudaNativeArray(CudaVm):
    execution_mode = 'native_array_aot'
    runtime_instruction_fetch = False
    interpreter_fallback = False
    supports_full_duel_rules = False

    def _instruction_input(self, image):
        # Preserve the checked kernel ABI with an inert placeholder. Poisoning
        # it must not change native execution, including callbacks and jumps.
        return np.full((1, 4), -2147483648, dtype=np.int32)

    def _emit_execution_source(self, image, limits, **options):
        from .native_array_codegen import emit_native_array_source, native_array_plan
        if options.get('compiled_blocks') is not None:
            raise ValueError('Native array execution requires complete compilation')
        self.native_plan = native_array_plan(image)
        return emit_native_array_source(image, limits, **options)

    def capability(self, entry):
        result = super().capability(entry)
        result.update(execution_mode=self.execution_mode,
                      runtime_instruction_fetch=False, interpreter_fallback=False,
                      native_plan=dict(identity=self.native_plan.identity,
                          compiled_instructions=self.native_plan.coverage['compiled_op_count'],
                          compiled_functions=self.native_plan.coverage['compiled_function_count'],
                          interpreter_fallback=False))
        return result
