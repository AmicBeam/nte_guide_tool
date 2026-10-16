"""Static native-array codegen contracts; no speedup or SoA claims."""
from types import MappingProxyType
import inspect
import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits, emit_vm_source
from app.modules.card_game.rl.batched_duel.native_array_codegen import (
    NativeArrayCodegenError,
    NativeArrayPlan,
    emit_native_array_source,
    native_array_plan,
    _FETCH_LINE,
    _OPCODE_SWITCH,
    HELPER_NAME,
)
from app.modules.card_game.rl.batched_duel.vm_image import VmImage, _image_identity


OPCODES = (
    'NOP', 'LOAD_CONST', 'LOAD_FAST', 'JUMP_ABSOLUTE', 'POP_TOP',
    'RETURN_VALUE', 'CALL_FUNCTION', 'FOR_ITER', 'YIELD_VALUE', 'STORE_SUBSCR',
)


def make_image():
    instructions = np.asarray(
        [
            [1, 0, 7, -1],
            [2, 0, 0, -1],
            [3, 0, 0, 4],
            [0, 0, 0, -1],
            [4, 0, 0, -1],
            [5, 0, 0, -1],
            [6, 1, 0, -1],
            [7, 0, 0, 6],
            [8, 0, 0, -1],
            [9, 0, 0, -1],
        ],
        dtype='<i4',
    )
    functions = np.zeros((2, 18), dtype='<i4')
    functions[0, 1] = 6
    functions[0, 5] = 8
    functions[0, 6] = 16
    functions[1, 0] = 6
    functions[1, 1] = 4
    functions[1, 5] = 8
    functions[1, 6] = 16
    image = VmImage(
        program_identity='native-array-test',
        identity='',
        instructions=instructions,
        functions=functions,
        classes=np.zeros((0, 6), dtype='<i4'),
        modules=np.zeros((0, 1), dtype='<i4'),
        nodes=np.zeros((1, 4), dtype='<i4'),
        edges=np.zeros((0, 2), dtype='<i4'),
        payload=np.zeros((0,), dtype=np.uint8),
        opcode_names=OPCODES,
        intrinsic_names=('next', 'object'),
        function_ids=('f0', 'f1'),
        class_ids=(),
        module_names=(),
        entries=MappingProxyType({'f0': 0}),
        required=MappingProxyType({
            'f0': MappingProxyType({'opcodes': OPCODES, 'intrinsics': ()}),
        }),
    )
    object.__setattr__(image, 'identity', _image_identity(image))
    image.verify()
    return image


def kernel_signature(source, name):
    token = 'extern "C" __global__ void ' + name + '('
    start = source.index(token)
    return source[start:source.index('{', start)]


def extract_execute(source):
    token = 'void execute(Context* c){'
    body = source.index(token)
    decl = source.rfind('__device__', 0, body)
    depth = 0
    for index, ch in enumerate(source[body:], body):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return source[decl:index + 1]
    raise AssertionError('execute() was not closed')


class NativeArrayPlanTest(unittest.TestCase):
    def test_complete_instruction_and_function_coverage_is_deterministic(self):
        image = make_image()
        first = native_array_plan(image)
        second = native_array_plan(image)
        self.assertIsInstance(first, NativeArrayPlan)
        self.assertFalse(first.interpreter_fallback)
        self.assertFalse(first.coverage['interpreter_fallback'])
        self.assertFalse(first.coverage['cpu_fallback'])
        self.assertFalse(first.coverage['measured_speedup'])
        self.assertFalse(first.coverage['soa_layout'])
        self.assertTrue(first.verify())
        self.assertEqual(first.identity, second.identity)
        self.assertEqual(first.coverage['compiled_pc'], tuple(range(len(image.instructions))))
        self.assertEqual(first.coverage['compiled_function_indexes'], (0, 1))
        self.assertEqual(first.coverage['compiled_function_ids'], ('f0', 'f1'))
        self.assertEqual(first.coverage['compiled_op_count'], len(image.instructions))
        self.assertEqual(first.coverage['compiled_function_count'], len(image.functions))
        self.assertEqual(first.coverage['unknown_opcodes'], ())
        self.assertEqual(first.coverage['helper_opcodes'], ('STORE_SUBSCR',))
        ops = tuple(row['op'] for row in first.coverage['instructions'])
        self.assertEqual(
            ops,
            tuple(OPCODES[int(image.instructions[pc, 0])] for pc in range(len(image.instructions))),
        )
        self.assertEqual(set(ops), set(OPCODES))
        self.assertEqual(first.coverage['functions'][0]['pcs'], tuple(range(6)))
        self.assertEqual(first.coverage['functions'][1]['pcs'], tuple(range(6, 10)))

    def test_malformed_image_and_jumps_fail_closed(self):
        with self.assertRaises(TypeError):
            native_array_plan(object())
        image = make_image()
        broken = np.array(image.instructions, copy=True)
        broken[2, 3] = 9
        object.__setattr__(image, 'instructions', broken)
        object.__setattr__(image, 'identity', _image_identity(image))
        with self.assertRaises(NativeArrayCodegenError):
            native_array_plan(image)
        image = make_image()
        names = list(image.opcode_names)
        names[2] = 'FAKE_OP'
        object.__setattr__(image, 'opcode_names', tuple(names))
        object.__setattr__(image, 'identity', _image_identity(image))
        with self.assertRaisesRegex(NativeArrayCodegenError, 'Unsupported frozen opcode'):
            emit_native_array_source(image)


class NativeArraySourceTest(unittest.TestCase):
    def test_kernel_abi_matches_emit_vm_source(self):
        image = make_image()
        base = emit_vm_source(image)
        native = emit_native_array_source(image)
        for name in ('vm_execute', 'vm_execute_device', 'vm_compact', 'vm_compact_device', 'vm_clock', 'vm_load_graph'):
            self.assertEqual(kernel_signature(base, name), kernel_signature(native, name))
        self.assertIn('c.instructions=instructions', native)

    def test_original_opcode_bodies_are_retained_without_instruction_fetch(self):
        image = make_image()
        native = emit_native_array_source(image)
        self.assertEqual(native, emit_native_array_source(image))
        self.assertIn(_OPCODE_SWITCH, native)
        self.assertIn('case OP_LOAD_CONST:push(c,f,value);break;', native)
        self.assertIn('case OP_RETURN_VALUE:return_value(c,pop(c,f),0);break;', native)
        self.assertIn('case OP_JUMP_ABSOLUTE:case OP_JUMP_FORWARD:f[1]=target;break;', native)
        self.assertIn('default:c->error=UNSUPPORTED_OP;break;', native)
        self.assertNotIn(_FETCH_LINE, native)
        self.assertNotIn('const int* ins=c->instructions', native)
        execute = extract_execute(native)
        self.assertNotIn('c->instructions', execute)
        self.assertNotIn('switch(op)', execute)
        self.assertEqual(execute.count('f[1]++'), 1)
        self.assertIn(HELPER_NAME + '(c,f,fn,pc);', execute)
        self.assertIn('if(++c->steps>c->step_limit)', execute)
        self.assertIn('if(c->raised!=MISSING)', execute)
        self.assertNotIn('run_compiled_block', native)

    def test_every_frozen_pc_is_static_including_control_flow_and_native_calls(self):
        image = make_image()
        native = emit_native_array_source(image)
        self.assertIn('/* stack LOAD_CONST */', native)
        self.assertIn('push(c,f,7);', native)
        self.assertIn('/* stack LOAD_FAST */', native)
        self.assertIn('/* stack POP_TOP */', native)
        self.assertIn('native_step<OP_JUMP_ABSOLUTE>(c,f,fn,0,0,4);', native)
        self.assertIn('native_step<OP_RETURN_VALUE>(c,f,fn,0,0,-1);', native)
        self.assertIn('native_step<OP_CALL_FUNCTION>(c,f,fn,1,0,-1);', native)
        self.assertIn('native_step<OP_FOR_ITER>(c,f,fn,0,0,6);', native)
        self.assertIn('native_step<OP_YIELD_VALUE>(c,f,fn,0,0,-1);', native)
        self.assertIn('native_step<OP_STORE_SUBSCR>(c,f,fn,0,0,-1);', native)
        for pc in range(len(image.instructions)):
            self.assertIn('case ' + str(pc) + ':', native)
        self.assertIn('default:c->error=BAD_PROGRAM;return;', native)
        self.assertIn('pc<fn[0]||pc>=fn[0]+fn[1]){c->error=BAD_PROGRAM;break;}', native)

    def test_options_match_emit_vm_source_and_reject_unsupported_values(self):
        image = make_image()
        native = emit_native_array_source(image)
        self.assertIn('__device__ inline void execute(Context* c){', native)
        noinline = emit_native_array_source(image, device_call_boundaries=True)
        self.assertIn('__device__ __noinline__ void execute(Context* c){', noinline)
        fast = emit_native_array_source(image, direct_stack_dispatch=True)
        load = fast[fast.index('/* stack LOAD_CONST */'):fast.index('/* stack LOAD_FAST */')]
        self.assertIn('push(c,f,7);', load)
        self.assertNotIn('native_handlers<OP_LOAD_CONST>', load)
        self.assertIn('native_handlers<OP_LOAD_CONST>', native)
        with self.assertRaises(TypeError):
            emit_native_array_source(image, constant_lookup=1)
        with self.assertRaises(TypeError):
            emit_native_array_source(image, compiled_blocks=[])
        with self.assertRaises(TypeError):
            emit_native_array_source(image, not_an_option=True)
        accepted = emit_native_array_source(
            image,
            limits=VmLimits(frames=8),
            compiled_blocks={},
            constant_lookup=False,
            compact_frame_init=True,
            device_call_boundaries=False,
            library_call_boundaries=False,
            mutable_lookup_cache=False,
            frozen_class_lookup=False,
            direct_stack_dispatch=False,
        )
        self.assertIn('VM_FRAMES=8', accepted)
        self.assertNotIn(_FETCH_LINE, accepted)
        self.assertNotIn('run_compiled_block', accepted)
        vm_params = set(inspect.signature(emit_vm_source).parameters) - {'image', 'limits'}
        values = {name: False if name != 'compiled_blocks' else None for name in vm_params}
        inspect.signature(emit_native_array_source).bind(image, VmLimits(), **values)
        inspect.signature(emit_vm_source).bind(image, VmLimits(), **values)


class NativeArrayFrozenProgramTest(unittest.TestCase):
    def test_captured_program_compiles_every_instruction(self):
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        from tests.test_duel_v2_vm_compiled_blocks import program
        image = lower_program(program())
        plan = native_array_plan(image)
        native = emit_native_array_source(image)
        self.assertFalse(plan.interpreter_fallback)
        self.assertEqual(plan.coverage['compiled_op_count'], len(image.instructions))
        self.assertEqual(plan.coverage['compiled_function_count'], len(image.functions))
        self.assertEqual(plan.coverage['compiled_pc'], tuple(range(len(image.instructions))))
        self.assertNotIn(_FETCH_LINE, native)
        self.assertIn(_OPCODE_SWITCH, native)
        execute = extract_execute(native)
        self.assertNotIn('c->instructions', execute)
        self.assertNotIn('switch(op)', execute)
        for pc, row in enumerate(image.instructions):
            op = image.opcode_names[int(row[0])]
            self.assertIn('case ' + str(pc) + ':', native)
            self.assertIn('OP_' + op, native)


if __name__ == '__main__':
    unittest.main()
