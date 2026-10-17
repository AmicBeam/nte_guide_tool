"""CUDA Driver relocatable CUBIN linker; fake ctypes driver, no GPU required."""
import ctypes
import unittest

from app.modules.card_game.rl.batched_duel.cuda_linker import (
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INPUT_CUBIN,
    CU_JIT_OPTIMIZATION_LEVEL,
    LOG_BUFFER_SIZE,
    link_cuda_objects,
)
from app.modules.card_game.rl.batched_duel.cuda_runtime import CudaDriverError


class CFunc:
    def __init__(self, impl):
        self._impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._impl(*args)


def _handle(state):
    if isinstance(state, ctypes.c_void_p):
        return state.value
    if isinstance(state, int):
        return state
    return ctypes.cast(state, ctypes.c_void_p).value


def _copy(ptr, size):
    if hasattr(ptr, 'raw'):
        return bytes(ptr.raw[:size])
    address = ptr if isinstance(ptr, int) else ctypes.addressof(ptr)
    return ctypes.string_at(address, size)


class FakeDriver:
    HANDLE = 0xC0FFEE

    def __init__(
        self,
        *,
        v2_create=True,
        v2_add=True,
        fallback_create=False,
        fallback_add=False,
        create_rc=0,
        add_rcs=None,
        complete_rc=0,
        destroy_rc=0,
        cubin=b'\x7fELF\x00linked\x00image',
        error_name=b'CUDA_ERROR_UNKNOWN',
        error_string=b'unknown error',
        destroy_raises=None,
        complete_empty=False,
        write_logs_on=None,
    ):
        self.calls = []
        self.create_options = None
        self.create_values = None
        self.added = []
        self.added_refs = []
        self.alive_at_complete = None
        self.alive_at_destroy = None
        self.cubin = cubin
        self.cubin_buf = ctypes.create_string_buffer(cubin, len(cubin))
        self.create_rc = create_rc
        self.add_rcs = list(add_rcs) if add_rcs is not None else []
        self.complete_rc = complete_rc
        self.destroy_rc = destroy_rc
        self.error_name = error_name
        self.error_string = error_string
        self.destroy_raises = destroy_raises
        self.complete_empty = complete_empty
        self.write_logs_on = write_logs_on
        self.destroy_calls = 0

        if v2_create:
            self.cuLinkCreate_v2 = CFunc(self._create)
        if fallback_create:
            self.cuLinkCreate = CFunc(self._create)
        if v2_add:
            self.cuLinkAddData_v2 = CFunc(self._add_data)
        if fallback_add:
            self.cuLinkAddData = CFunc(self._add_data)
        self.cuLinkComplete = CFunc(self._complete)
        self.cuLinkDestroy = CFunc(self._destroy)
        self.cuGetErrorName = CFunc(self._error_name)
        self.cuGetErrorString = CFunc(self._error_string)

    def _snapshot(self):
        objects = []
        for name, data, size in self.added_refs:
            name_bytes = name.value if hasattr(name, 'value') else ctypes.string_at(name)
            objects.append((name_bytes, _copy(data, size), size))
        logs = None
        if self.create_values is not None:
            logs = (
                _copy(self.create_values[0], LOG_BUFFER_SIZE),
                _copy(self.create_values[2], LOG_BUFFER_SIZE),
            )
        return objects, logs

    def _maybe_write_logs(self, stage):
        if self.write_logs_on != stage or self.create_values is None:
            return
        ctypes.memmove(self.create_values[0], b'info from linker\0', 17)
        ctypes.memmove(self.create_values[2], b'error from linker\0', 18)

    def _create(self, count, options, values, state_out):
        self.create_options = [options[i] for i in range(count)]
        self.create_values = [values[i] for i in range(count)]
        self.calls.append(('create', count, list(self.create_options), list(self.create_values)))
        self._maybe_write_logs('create')
        ctypes.cast(state_out, ctypes.POINTER(ctypes.c_void_p))[0] = self.HANDLE
        return self.create_rc

    def _add_data(self, state, kind, data, size, name, nopt, opts, vals):
        size_value = int(size.value if isinstance(size, ctypes.c_size_t) else size)
        name_bytes = name.value if hasattr(name, 'value') else ctypes.string_at(name)
        data_bytes = _copy(data, size_value)
        self.added_refs.append((name, data, size_value))
        self.added.append((name_bytes, data_bytes, size_value))
        self.calls.append(('add', _handle(state), kind, size_value, name_bytes, nopt, opts, vals))
        self._maybe_write_logs('add')
        if self.add_rcs:
            return self.add_rcs.pop(0)
        return 0

    def _complete(self, state, cubin_out, size_out):
        self.alive_at_complete = self._snapshot()
        self.calls.append(('complete', _handle(state)))
        self._maybe_write_logs('complete')
        if self.complete_empty:
            ctypes.cast(cubin_out, ctypes.POINTER(ctypes.c_void_p))[0] = 0
            ctypes.cast(size_out, ctypes.POINTER(ctypes.c_size_t))[0] = 0
            return self.complete_rc
        ctypes.cast(cubin_out, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(self.cubin_buf)
        ctypes.cast(size_out, ctypes.POINTER(ctypes.c_size_t))[0] = len(self.cubin)
        return self.complete_rc

    def _destroy(self, state):
        self.destroy_calls += 1
        self.alive_at_destroy = self._snapshot()
        self.calls.append(('destroy', _handle(state)))
        if self.destroy_raises is not None:
            raise self.destroy_raises
        return self.destroy_rc

    def _error_name(self, rc, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_char_p))[0] = self.error_name
        return 0

    def _error_string(self, rc, out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_char_p))[0] = self.error_string
        return 0


class CudaLinkerTests(unittest.TestCase):
    def objects(self):
        return [('alpha', b'\x7fELFa\x00one'), ('beta', b'\x7fELFb\x00two')]

    def test_pointer_size_signatures_and_copied_output(self):
        driver = FakeDriver()
        cubin = driver.cubin
        result = link_cuda_objects(driver, self.objects(), jit_level=2)
        self.assertEqual(result, cubin)
        driver.cubin_buf.raw = b'X' * len(cubin)
        self.assertEqual(result, cubin)
        self.assertIsInstance(result, bytes)

        for name in ('cuLinkCreate_v2', 'cuLinkAddData_v2', 'cuLinkComplete', 'cuLinkDestroy'):
            self.assertEqual(getattr(driver, name).restype, ctypes.c_int)
        self.assertEqual(driver.cuLinkCreate_v2.argtypes, [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ])
        self.assertEqual(driver.cuLinkAddData_v2.argtypes, [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ])
        self.assertEqual(
            ctypes.sizeof(driver.cuLinkAddData_v2.argtypes[3]),
            ctypes.sizeof(ctypes.c_void_p),
        )
        self.assertEqual(driver.cuLinkComplete.argtypes, [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ])
        self.assertEqual(
            ctypes.sizeof(driver.cuLinkComplete.argtypes[2]._type_),
            ctypes.sizeof(ctypes.c_void_p),
        )
        self.assertEqual(driver.cuLinkDestroy.argtypes, [ctypes.c_void_p])
        self.assertEqual(
            [step[0] for step in driver.calls],
            ['create', 'add', 'add', 'complete', 'destroy'],
        )
        self.assertEqual(driver.calls[-1][1], FakeDriver.HANDLE)
        self.assertEqual(driver.destroy_calls, 1)

    def test_prefers_v2_symbols_and_falls_back(self):
        both = FakeDriver(v2_create=True, v2_add=True, fallback_create=True, fallback_add=True)
        link_cuda_objects(both, self.objects())
        self.assertIsNotNone(both.cuLinkCreate_v2.argtypes)
        self.assertIsNotNone(both.cuLinkCreate.argtypes)
        self.assertEqual(
            [step[0] for step in both.calls],
            ['create', 'add', 'add', 'complete', 'destroy'],
        )

        fallback = FakeDriver(v2_create=False, v2_add=False, fallback_create=True, fallback_add=True)
        self.assertFalse(hasattr(fallback, 'cuLinkCreate_v2'))
        self.assertFalse(hasattr(fallback, 'cuLinkAddData_v2'))
        link_cuda_objects(fallback, [('only', b'cubin-object')])
        self.assertEqual(fallback.cuLinkCreate.argtypes[0], ctypes.c_uint)
        self.assertEqual(fallback.cuLinkAddData.argtypes[3], ctypes.c_size_t)
        self.assertEqual(
            [step[0] for step in fallback.calls],
            ['create', 'add', 'complete', 'destroy'],
        )

        mixed = FakeDriver(v2_create=False, v2_add=True, fallback_create=True, fallback_add=False)
        link_cuda_objects(mixed, [('only', b'cubin-object')])
        self.assertEqual(mixed.cuLinkCreate.restype, ctypes.c_int)
        self.assertEqual(mixed.cuLinkAddData_v2.argtypes[3], ctypes.c_size_t)

    def test_jit_options_and_log_buffers_stay_alive(self):
        driver = FakeDriver(write_logs_on='complete')
        link_cuda_objects(driver, self.objects())
        self.assertEqual(driver.create_options, [
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ])
        self.assertEqual(driver.create_values[1], LOG_BUFFER_SIZE)
        self.assertEqual(driver.create_values[3], LOG_BUFFER_SIZE)

        objects, logs = driver.alive_at_complete
        self.assertEqual(objects[0], (b'alpha', b'\x7fELFa\x00one', 9))
        self.assertEqual(objects[1], (b'beta', b'\x7fELFb\x00two', 9))
        self.assertTrue(logs[0].startswith(b'\x00'))
        destroy_objects, destroy_logs = driver.alive_at_destroy
        self.assertEqual(destroy_objects, objects)
        self.assertTrue(destroy_logs[0].startswith(b'info from linker\x00'))
        self.assertTrue(destroy_logs[1].startswith(b'error from linker\x00'))

        leveled = FakeDriver()
        link_cuda_objects(leveled, [('one', b'payload')], jit_level=4)
        self.assertEqual(leveled.create_options[-1], CU_JIT_OPTIMIZATION_LEVEL)
        self.assertEqual(leveled.create_values[-1], 4)

        zero = FakeDriver()
        link_cuda_objects(zero, [('one', b'payload')], jit_level=0)
        self.assertEqual(zero.create_options[-1], CU_JIT_OPTIMIZATION_LEVEL)
        self.assertIsNone(zero.create_values[-1])

    def test_object_names_are_utf8_and_data_is_not_borrowed(self):
        driver = FakeDriver()
        result = link_cuda_objects(driver, [('目标', b'\x7fELFsrc')])
        self.assertEqual(driver.added[0][0], '目标'.encode('utf-8'))
        self.assertEqual(driver.alive_at_complete[0][0][1], b'\x7fELFsrc')
        self.assertEqual(driver.alive_at_destroy[0][0][1], b'\x7fELFsrc')
        self.assertEqual(result, driver.cubin)
        self.assertEqual(driver.calls[1][2], CU_JIT_INPUT_CUBIN)
        self.assertEqual(driver.calls[1][5], 0)
        self.assertIsNone(driver.calls[1][6])
        self.assertIsNone(driver.calls[1][7])

    def test_link_errors_include_name_string_and_logs(self):
        driver = FakeDriver(add_rcs=[9], write_logs_on='add')
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(driver, self.objects())
        message = str(ctx.exception)
        self.assertIn('cuLinkAddData_v2 failed with CUDA driver error code 9', message)
        self.assertIn('CUDA_ERROR_UNKNOWN', message)
        self.assertIn('unknown error', message)
        self.assertIn('error from linker', message)
        self.assertIn('info from linker', message)
        self.assertEqual(driver.destroy_calls, 1)

        fallback = FakeDriver(
            v2_create=False, fallback_create=True, create_rc=4, write_logs_on='create'
        )
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(fallback, [('one', b'payload')])
        self.assertIn('cuLinkCreate failed with CUDA driver error code 4', str(ctx.exception))
        self.assertEqual(fallback.destroy_calls, 0)

    def test_lifecycle_on_every_failure_preserves_original_error(self):
        create_fail = FakeDriver(create_rc=3)
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(create_fail, self.objects())
        self.assertIn('cuLinkCreate_v2', str(ctx.exception))
        self.assertEqual(create_fail.destroy_calls, 0)

        add_fail = FakeDriver(add_rcs=[0, 5], destroy_raises=RuntimeError('cleanup'))
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(add_fail, self.objects())
        self.assertIn('cuLinkAddData_v2 failed with CUDA driver error code 5', str(ctx.exception))
        self.assertEqual(add_fail.destroy_calls, 1)

        complete_fail = FakeDriver(complete_rc=8, destroy_rc=11)
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(complete_fail, self.objects())
        self.assertIn('cuLinkComplete failed with CUDA driver error code 8', str(ctx.exception))
        self.assertEqual(complete_fail.destroy_calls, 1)

        empty = FakeDriver(complete_empty=True)
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(empty, self.objects())
        self.assertIn('empty CUBIN', str(ctx.exception))
        self.assertEqual(empty.destroy_calls, 1)

        destroy_fail = FakeDriver(destroy_rc=6)
        with self.assertRaises(CudaDriverError) as ctx:
            link_cuda_objects(destroy_fail, [('one', b'payload')])
        self.assertIn('cuLinkDestroy failed with CUDA driver error code 6', str(ctx.exception))

    def test_invalid_inputs_never_touch_the_driver(self):
        driver = FakeDriver()
        cases = [
            (TypeError, [('alpha', b'ok'), b'not-a-pair']),
            (TypeError, [(b'alpha', b'ok')]),
            (TypeError, [('alpha', bytearray(b'ok'))]),
            (ValueError, []),
            (ValueError, [('', b'ok')]),
            (ValueError, [('a\0b', b'ok')]),
            (ValueError, [('dup', b'one'), ('dup', b'two')]),
            (ValueError, [('alpha', b'')]),
        ]
        for error, objects in cases:
            with self.subTest(objects=objects):
                with self.assertRaises(error):
                    link_cuda_objects(driver, objects)
                self.assertEqual(driver.calls, [])
                self.assertEqual(driver.destroy_calls, 0)

        with self.assertRaises(TypeError):
            link_cuda_objects(driver, self.objects(), True)
        with self.assertRaises(TypeError):
            link_cuda_objects(driver, self.objects(), jit_level=True)
        self.assertEqual(driver.calls, [])

        with self.assertRaises(CudaDriverError):
            link_cuda_objects(object(), self.objects())


if __name__ == '__main__':
    unittest.main()
