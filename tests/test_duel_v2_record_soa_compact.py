"""Typed RecordSoA compact: CPU reference, host-shim CUDA source, optional CUDA."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.modules.card_game.rl.batched_duel.native_array_cpu import (
    HOST_CUDA_SHIMS, _cptr, compile_host_native_library,
)
from app.modules.card_game.rl.batched_duel.record_soa import (
    REF, TAG_DICT, TAG_PLAYER, RecordSoABatch, RecordSoAError,
    compile_schema, pack_game_states, unpack_game_states,
)
from app.modules.card_game.rl.batched_duel.record_soa_compact import (
    CudaModuleKernelSpec, CudaRecordSoACompact, compact_rows, emit_compact_source,
)


def schema():
    return compile_schema({
        'Game': {'kind': 'record', 'capacity': 4, 'tags': (TAG_DICT, TAG_PLAYER), 'fields': {
            'hp': 'int', 'rate': 'float', 'ready': 'bool', 'name': 'str',
            'left': '@Item', 'right': '@Item', 'items': '@Items',
            'mapping': '@Counters', 'pair': '@Pair',
        }},
        'Item': {'kind': 'record', 'capacity': 8, 'fields': {'n': 'int'}},
        'Items': {'kind': 'list', 'capacity': 4, 'max_items': 8, 'value': '@Item'},
        'Counters': {'kind': 'map', 'capacity': 4, 'max_items': 8, 'key': 'str', 'value': 'int'},
        'Pair': {'kind': 'tuple', 'capacity': 4, 'items': ['int', 'str']},
        'Node': {'kind': 'record', 'capacity': 4, 'fields': {'next': '@Node'}},
    }, max_strings=32, max_string_bytes=512)


def sample(name='真红'):
    item = {'n': 5}
    return {'name': name, 'right': item, 'left': item, 'hp': 3, 'rate': 0.125,
            'ready': True, 'items': [item, item], 'mapping': {'b': 2, 'a': 1},
            'pair': (7, 'x')}


def _cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class RecordSoACompactTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        code = (
            "import sys\n"
            "from app.modules.card_game.rl.batched_duel.record_soa_compact import compact_rows\n"
            "assert 'torch' not in sys.modules\n"
            "assert compact_rows.__name__ == 'compact_rows'\n"
        )
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)

    def test_aliases_types_order_and_classes(self):
        packed = pack_game_states([sample()], schema())
        game = packed.schema.pools[packed.schema.root]
        packed.buffers['u8'][game.tag, 0] = TAG_PLAYER
        result = unpack_game_states(compact_rows(packed))[0]
        source = sample()
        self.assertEqual(result, source)
        self.assertEqual(list(result), list(source))
        self.assertEqual(list(result['mapping']), ['b', 'a'])
        self.assertIs(result['left'], result['right'])
        self.assertIs(result['items'][1], result['right'])
        self.assertIs(type(result['pair']), tuple)
        self.assertIs(type(result['items']), list)
        self.assertEqual(int(compact_rows(packed).buffers['u8'][game.tag, 0]), TAG_PLAYER)
        distinct = pack_game_states([{'left': {'n': 1}, 'right': {'n': 1}}], schema())
        out = unpack_game_states(compact_rows(distinct))[0]
        self.assertEqual(out['left'], out['right'])
        self.assertIsNot(out['left'], out['right'])

    def test_absent_null_and_independent_rows(self):
        packed = pack_game_states([{}, {'hp': None, 'name': 'a'}, sample()], schema())
        compacted = compact_rows(packed)
        self.assertEqual(unpack_game_states(compacted)[:2], [{}, {'hp': None, 'name': 'a'}])
        item = packed.schema.pools[packed.schema.type_ids['Item']]
        packed.buffers['i32'][item.used, 0] = 2
        packed.buffers['u8'][item.tag + 1, 0] = 99
        compacted = compact_rows(packed)
        self.assertEqual(int(compacted.buffers['i32'][item.used, 0]), 0)
        self.assertEqual(int(compacted.buffers['i32'][item.used, 2]), 1)

    def test_dead_strings_are_dropped_and_duplicates_merge(self):
        packed = pack_game_states([{'name': 'hello', 'pair': (1, 'hello')}], schema())
        sch = packed.schema
        off = int(packed.buffers['i32'][sch.string_offsets, 0])
        ln = int(packed.buffers['i32'][sch.string_lengths, 0])
        used = int(packed.buffers['i32'][sch.bytes_used, 0])
        packed.buffers['i32'][sch.strings_count, 0] = 2
        packed.buffers['i32'][sch.string_offsets + 1, 0] = used
        packed.buffers['i32'][sch.string_lengths + 1, 0] = ln
        packed.buffers['i64'][sch.string_hashes + 1, 0] = packed.buffers['i64'][sch.string_hashes, 0]
        packed.buffers['u8'][sch.string_bytes + used:sch.string_bytes + used + ln, 0] = (
            packed.buffers['u8'][sch.string_bytes + off:sch.string_bytes + off + ln, 0])
        packed.buffers['i32'][sch.bytes_used, 0] = used + ln
        pair = packed.schema.pools[packed.schema.type_ids['Pair']]
        packed.buffers['i32'][pair.values[1].word, 0] = 2
        compacted = compact_rows(packed)
        self.assertEqual(int(compacted.buffers['i32'][sch.strings_count, 0]), 1)
        self.assertEqual(unpack_game_states(compacted), [{'name': 'hello', 'pair': (1, 'hello')}])

    def test_roots_and_poison_fail_closed(self):
        packed = pack_game_states([{'hp': 1, 'left': {'n': 1}}], schema())
        game = packed.schema.pools[packed.schema.root]
        left = next(f for f in game.fields if f.name.endswith('.left'))
        cases = []
        bad_used = packed.clone_rows([0])
        bad_used.buffers['i32'][game.used, 0] = 99
        cases.append(bad_used)
        bad_tag = packed.clone_rows([0])
        bad_tag.buffers['u8'][game.tag, 0] = 3
        cases.append(bad_tag)
        bad_count = packed.clone_rows([0])
        bad_count.buffers['i32'][game.count, 0] = 99
        cases.append(bad_count)
        bad_ref = packed.clone_rows([0])
        bad_ref.buffers['i32'][left.word, 0] = 0
        cases.append(bad_ref)
        bad_root = packed.clone_rows([0])
        bad_root.roots[0] = 0
        cases.append(bad_root)
        far = packed.clone_rows([0])
        far.roots[0] = 99999
        cases.append(far)
        for batch in cases:
            with self.subTest(), self.assertRaises(RecordSoAError):
                compact_rows(batch)

    def test_cycle_is_rejected(self):
        node_schema = compile_schema(
            {'Node': {'kind': 'record', 'capacity': 4, 'fields': {'next': '@Node'}}},
            root='Node', max_strings=8, max_string_bytes=64)
        packed = pack_game_states([{'next': None}], node_schema)
        node = packed.schema.pools[0]
        field = node.fields[0]
        packed.buffers['u8'][field.tag, 0] = REF
        packed.buffers['i32'][field.word, 0] = int(packed.roots[0])
        packed.buffers['i32'][node.count, 0] = 1
        packed.buffers['i32'][node.order, 0] = 0
        with self.assertRaisesRegex(RecordSoAError, 'cycle'):
            compact_rows(packed)


def _bind_host(library):
    fn = library.host_record_soa_compact
    fn.argtypes = (ctypes.c_void_p,) * 15 + (ctypes.c_int,)
    fn.restype = None
    return fn


def _host_compact(fn, packed):
    schema = packed.schema
    dest = RecordSoABatch(schema, packed.rows)
    obj_map = np.zeros((schema.total_objects, packed.rows), np.int32)
    stack = np.zeros((schema.total_objects, packed.rows), np.int32)
    color = np.zeros((schema.total_objects, packed.rows), np.uint8)
    str_map = np.zeros((schema.max_strings, packed.rows), np.int32)
    status = np.zeros(packed.rows, np.int32)
    fn(_cptr(packed.buffers['i64']), _cptr(packed.buffers['f64']), _cptr(packed.buffers['i32']),
       _cptr(packed.buffers['u8']), _cptr(packed.roots),
       _cptr(dest.buffers['i64']), _cptr(dest.buffers['f64']), _cptr(dest.buffers['i32']),
       _cptr(dest.buffers['u8']), _cptr(dest.roots),
       _cptr(obj_map), _cptr(stack), _cptr(color), _cptr(str_map), _cptr(status), packed.rows)
    return dest, status


class RecordSoACompactHostSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = schema()
        source = emit_compact_source(cls.schema)
        if 'extern "C" __global__ void record_soa_compact' not in source:
            raise AssertionError('compact kernel missing')
        if 'host_record_soa_compact' not in source:
            raise AssertionError('host launcher missing')
        cls._temp = tempfile.TemporaryDirectory(prefix='record_soa_compact_')
        cls.library = compile_host_native_library(HOST_CUDA_SHIMS + source, Path(cls._temp.name))
        cls.fn = _bind_host(cls.library)

    @classmethod
    def tearDownClass(cls):
        cls.library = None
        cls.fn = None
        cls._temp.cleanup()

    def test_emitted_source_uses_staged_device_helpers(self):
        source = emit_compact_source(self.schema)
        for name in ('discover(', 'assign_maps(', 'compact_strings(', 'copy_live('):
            self.assertIn(name, source)
        self.assertEqual(tuple(CudaModuleKernelSpec), ('record_soa_compact',))

    def test_host_shim_matches_cpu_reference(self):
        packed = pack_game_states([sample(), {}, {'hp': None, 'name': 'a'}], self.schema)
        item = packed.schema.pools[packed.schema.type_ids['Item']]
        packed.buffers['i32'][item.used, 1] = 1
        packed.buffers['u8'][item.tag, 1] = 99
        reference = compact_rows(packed)
        dest, status = _host_compact(self.fn, packed)
        self.assertTrue(np.all(status == 0))
        for key in packed.buffers:
            self.assertTrue(np.array_equal(dest.buffers[key], reference.buffers[key]), key)
        self.assertTrue(np.array_equal(dest.roots, reference.roots))

    def test_host_shim_reports_poison_status(self):
        packed = pack_game_states([{'hp': 1}], self.schema)
        packed.roots[0] = 0
        dest, status = _host_compact(self.fn, packed)
        self.assertNotEqual(int(status[0]), 0)
        with self.assertRaises(RecordSoAError):
            compact_rows(packed)


@unittest.skipUnless(_cuda_available(), 'CUDA GPU not available')
class RecordSoACompactCudaTests(unittest.TestCase):
    gpu_qualified = False

    def test_cuda_module_matches_cpu_reference(self):
        compiled = schema()
        packed = pack_game_states([sample(), {'hp': None}], compiled)
        reference = compact_rows(packed)
        runtime = CudaRecordSoACompact(compiled)
        try:
            result = runtime.compact(packed.to_torch('cuda')).to_host()
        finally:
            runtime.close()
        for key in packed.buffers:
            self.assertTrue(np.array_equal(result.buffers[key], reference.buffers[key]), key)
        self.assertTrue(np.array_equal(result.roots, reference.roots))
        self.assertFalse(CudaRecordSoACompact.gpu_qualified)


if __name__ == '__main__':
    unittest.main()
