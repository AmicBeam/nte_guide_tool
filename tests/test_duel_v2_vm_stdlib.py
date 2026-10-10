"""CPU/source coverage for frozen-rule standard-library VM ops.

Device execution is skipped unless CUDA is present. Later CUDA fixtures use the
same layout as tests/test_duel_v2_cuda_vm.py: capture on CPython 3.10, then on
other hosts set NTE_VM_TEST_PROGRAM_DIR to a directory of per-stem JSON programs
(or all.json with NTE_VM_COMBINED_PROGRAM=1). This module never claims full
Python or full duel-rule support. Set/frozenset iteration is first-seen
insertion order; hash-randomized order-sensitive paths are unsupported.
"""
from pathlib import Path
import os
import sys
import unittest
from collections import Counter
from enum import Enum
from itertools import combinations, permutations
from math import isfinite

from app.modules.card_game.rl.batched_duel.cuda_vm import (
    CudaVm, SUPPORTED_INTRINSICS, SUPPORTED_OPS, VmCapabilityError, emit_vm_source,
)
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.vm_stdlib import (
    SUPPORTED_STDLIB_NAMES, _kits_precondition, emit_stdlib_source,
)
from app.modules.card_game.rl.rule_ir.portable_program import FrozenRuleProgram, capture_program


class Color(Enum):
    RED = 1
    BLUE = 2


class Flag(str, Enum):
    A = 'a'
    B = 'b'


class BaseBox:
    def __init__(self, n):
        self.n = n


class ChildBox(BaseBox):
    def __init__(self, n):
        super().__init__(n)
        self.m = n + 1

    def doubled(self):
        return self.n * 2


def constructors_probe(value):
    items = list(value)
    enumerated = list(enumerate(items, start=1))
    members = set(items)
    frozen = frozenset(items)
    members.add(9)
    members.update([9, 8])
    counts = Counter(items)
    counts.update([items[0]] if items else [])
    copied = counts.copy()
    missing = counts['__absent__']
    finite = isfinite(1.0) and isfinite(0)
    combos = list(combinations(items[:3] if len(items) >= 3 else [1, 2, 3], 2))
    perms = list(permutations(items[:2] if len(items) >= 2 else [1, 2]))
    empty_combo = list(combinations(items[:2] if items else [1, 2], 0))
    return {
        'enum': enumerated,
        'set': sorted(members),
        'frozen': sorted(frozen),
        'in_set': (items[0] in members) if items else False,
        # Exported state dictionaries use string keys; integer Counter keys stay
        # valid inside the VM and cross the codec as explicit (key, value) pairs.
        'counts': list(copied.items()),
        'missing': missing,
        'len_counter': len(counts),
        'empty_counter': not Counter(),
        'finite': finite,
        'combos': combos,
        'perms': perms,
        'empty_combo': empty_combo,
        'isinstance': isinstance(1, int) and isinstance('x', str) and not isinstance(1, str),
        'getattr': getattr(counts, 'copy') is not None,
        'hasattr': hasattr(counts, 'update') and not hasattr(counts, 'nope'),
        'diff': sorted({1, 2, 3} - set([2])),
    }


def callback_probe(value):
    items = list(value)
    keyed = sorted(items, key=lambda item: -item)
    stable = sorted([(1, 'b'), (1, 'a'), (2, 'c')], key=lambda pair: pair[0])
    generated = sorted((item for item in items), key=lambda item: item if item >= 0 else -item)
    lowest = min(items, key=lambda item: -item) if items else None
    highest = max((item for item in items), key=lambda item: item) if items else None
    listed = list(enumerate(item for item in items))
    return keyed, stable, generated, lowest, highest, listed


def nonfinite_probe(value):
    return isfinite(float('inf'))


def class_enum_super_probe(value):
    child = ChildBox(value)
    return {
        'n': child.n,
        'm': child.m,
        'doubled': child.doubled(),
        'color_eq': Color.RED == Color.RED and Color.RED != Color.BLUE and Color.RED != 1,
        'flag_eq': Flag.A == 'a' and Flag.A != Flag.B,
        'isinstance_enum': isinstance(Color.RED, Color) and isinstance(Color.RED, Enum),
        'name': Color.RED.name,
        'enum_value': Color.RED.value,
        'repr': repr(Color.RED),
        'getattr': getattr(child, 'n'),
        'hasattr': hasattr(child, 'm') and not hasattr(child, 'missing'),
        'getattr_default': getattr(child, 'missing', 7),
    }


def clone_probe(value):
    from app.modules.card_game.engine.duel_v2.entities import clone_state
    copied = clone_state(value)
    return copied, copied is not value


def ensure_probe(value):
    from app.modules.card_game.content.duel_v2.registry import _ensure_kits
    _ensure_kits()
    return value


def uuid_probe(value):
    from uuid import uuid4
    return uuid4().hex, value


def keyword_probe(value):
    return sorted(value, key=lambda item: item, unexpected=True)


def inplace_sort_probe(mode):
    # The sampler imports json locally and captures that module in its key.
    import json
    items=[('B',{'z':2}),('A',{'z':1}),('A',{'z':0})]
    alias=items
    result=items.sort(key=lambda c:json.dumps(c,sort_keys=True),reverse=bool(mode))
    return items,alias,result,items is alias


def inplace_sort_keys_probe(mode):
    items=[(2,'b'),(1,'z'),(2,'a'),(1,'x')]
    seen=[];alias=items
    def key(value):
        seen.append((value[0],len(items)))
        return value[0]
    result=items.sort(key=key,reverse=bool(mode))
    return items,alias,result,seen,items is alias


def inplace_sort_exception_probe(mode):
    items=[3,1,2];alias=items;seen=[]
    def key(value):
        seen.append(value)
        if mode==0 and value==1:raise ValueError('key error')
        if mode==1:items.append(9)
        return value
    error=None
    try:
        items.sort(key=key)
    except ValueError as exc:
        error=exc.args
    return items,alias,error,seen,items is alias


def unpacked_generator_probe(values):
    # Same opcode pattern as encode_base's tuple of hand + generated discards.
    parts=([-1],*([v,v+1] for v in values))
    expanded=[-2,*(v for v in values),99]
    reused=(v*2 for v in values)
    first=(*reused,);second=(*reused,)
    return parts,expanded,first,second,(*range(len(values)),),[*iter(values),42]


def type_identity_probe(value):
    types=(bool,int,float,str,list,tuple,dict,set,frozenset,range,Counter)
    values=(True,1,1.25,'x',[1],(1,),{'x':1},set(),frozenset(),range(2),Counter())
    return [type(item) is types[i] for i,item in enumerate(values)],type(True) is int,type(value) is int


def test_program(stem, entry, function, *, own=False):
    if sys.version_info[:2] == (3, 10):
        kwargs = dict(scope_character_ids=None, scope_card_ids=None)
        if own:
            from app.modules.card_game.rl.rule_ir.portable_program import DEFAULT_ALLOWED_MODULE_PREFIXES
            kwargs['allowed_module_prefixes'] = DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,)
        return capture_program({entry: function}, **kwargs)
    directory = os.environ.get('NTE_VM_TEST_PROGRAM_DIR')
    if not directory:
        raise RuntimeError('CPython3.13 consumes explicit 3.10 frozen fixtures; set NTE_VM_TEST_PROGRAM_DIR')
    fixture = Path(directory) / ('stdlib-all.json' if os.environ.get('NTE_VM_COMBINED_PROGRAM') == '1' else 'stdlib-' + stem + '.json')
    program = FrozenRuleProgram.from_json(fixture.read_text())
    program.validate()
    from app.modules.card_game.rl.batched_duel.gpu_rule_program import verify_source_hashes
    verify_source_hashes(program, repo_root=Path(__file__).resolve().parents[1])
    if entry not in program.entries:
        raise ValueError('Frozen standard-library entry mismatch')
    return program


class _FakeImage:
    def __init__(self, *, names=(), module_names=(), modules=(), nodes=(), edges=(), payload=b''):
        import numpy as np
        self.intrinsic_names = tuple(names)
        self.module_names = tuple(module_names)
        self.modules = np.array(modules, dtype=object) if modules is not None else np.zeros((0, 1), dtype=object)
        self.nodes = list(nodes)
        edge_array = np.array(edges, dtype=object)
        self.edges = edge_array if edge_array.ndim == 2 else np.zeros((0, 2), dtype=object)
        self.payload = payload


class SourceTest(unittest.TestCase):
    def test_supported_names_are_real_and_uuid_stays_unresolved(self):
        for name in SUPPORTED_STDLIB_NAMES:
            self.assertIn(name, SUPPORTED_INTRINSICS)
        self.assertNotIn('uuid.uuid4', SUPPORTED_STDLIB_NAMES)
        self.assertNotIn('uuid.uuid4', SUPPORTED_INTRINSICS)
        self.assertIn('enum.Enum._generate_next_value_', SUPPORTED_STDLIB_NAMES)
        self.assertFalse(CudaVm.supports_full_duel_rules)

    def test_emit_contains_stdlib_handlers_and_numpy_slot(self):
        program = test_program('constructors', 'constructors', constructors_probe, own=True)
        image = lower_program(program)
        source = emit_vm_source(image)
        self.assertIn('// STDLIB_HANDLERS', source)
        self.assertIn('// NUMPY_HANDLERS', source)
        self.assertIn('T_STDLIB_ENUMERATE=70', source)
        self.assertIn('T_STDLIB_COUNTER=73', source)
        self.assertIn('stdlib_call', source)
        self.assertIn('stdlib_resume', source)
        self.assertIn('mode==7', source.replace(' ', ''))
        self.assertIn('first-seen insertion order', source)
        self.assertIn('random UUIDs are not forged', source)
        self.assertIn('STDLIB_POOL_CAP=64', source)
        self.assertNotIn('emit_numpy_source', source)
        self.assertFalse(CudaVm.supports_full_duel_rules)

    def test_kits_precondition_fails_unless_registry_is_complete(self):
        empty = _FakeImage()
        self.assertEqual(_kits_precondition(empty), (-1, 0))
        payload = b'KITS'
        # module dict at ref -1 / node 0; KITS key at node 1; empty mapping at node 2.
        incomplete = _FakeImage(
            names=('app.modules.card_game.content.duel_v2.registry._ensure_kits',),
            module_names=('app.modules.card_game.content.duel_v2.registry',),
            modules=[[-1]],
            nodes=[[7, 0, 1, 0], [4, 0, 4, 0], [7, 1, 0, 0]],
            edges=[[-2, -3]],
            payload=payload,
        )
        self.assertEqual(_kits_precondition(incomplete), (0, 0))
        complete = _FakeImage(
            names=incomplete.intrinsic_names,
            module_names=incomplete.module_names,
            modules=incomplete.modules,
            nodes=[[7, 0, 1, 0], [4, 0, 4, 0], [7, 1, 3, 0]],
            edges=incomplete.edges,
            payload=payload,
        )
        self.assertEqual(_kits_precondition(complete), (0, 3))
        source = emit_stdlib_source(incomplete)
        self.assertIn('const int STDLIB_KITS_MODULE=0,STDLIB_KITS_COUNT=0', source)

    def test_synthetic_entries_are_capability_ready_except_uuid(self):
        cases = (
            ('constructors', constructors_probe),
            ('callbacks', callback_probe),
            ('classes', class_enum_super_probe),
            ('clone', clone_probe),
            ('ensure', ensure_probe),
            ('inplace_sort',inplace_sort_probe),
            ('inplace_sort_keys',inplace_sort_keys_probe),
            ('inplace_sort_exception',inplace_sort_exception_probe),
            ('unpacked_generator',unpacked_generator_probe),
            ('type_identity',type_identity_probe),
        )
        for entry, fn in cases:
            program = test_program(entry, entry, fn, own=True)
            image = lower_program(program)
            missing = image.missing_capabilities(entry, opcodes=SUPPORTED_OPS, intrinsics=SUPPORTED_INTRINSICS)
            self.assertFalse(any(missing.values()), msg=(entry, missing))
        program = test_program('uuid', 'uuid', uuid_probe, own=True)
        image = lower_program(program)
        missing = image.missing_capabilities('uuid', opcodes=SUPPORTED_OPS, intrinsics=SUPPORTED_INTRINSICS)
        self.assertIn('uuid.uuid4', missing['intrinsics'])

    def test_unknown_keywords_are_not_ignored_in_source(self):
        program = test_program('keywords', 'keywords', keyword_probe, own=True)
        image = lower_program(program)
        source = emit_vm_source(image)
        self.assertIn('stdlib_unknown_kw', source)
        self.assertIn('BAD_CALL', source)


class CudaExecutionTest(unittest.TestCase):
    def test_generator_star_expansion_keeps_stack_and_exhaustion(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        rows=[[],[7],[3,1,4]]
        program=test_program('unpacked_generator','unpacked_generator',unpacked_generator_probe,own=True)
        with self.executor(lower_program(program)) as vm:
            actual=vm.launch(vm.ingest(pack_states(rows,GraphLayout(3,8192,16384,131072))),'unpacked_generator').decode()
        self.assertEqual(actual,[unpacked_generator_probe(row) for row in rows])

    def test_type_returns_canonical_type_identity_without_erasing_bool(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        rows=[1,True,'word']
        program=test_program('type_identity','type_identity',type_identity_probe,own=True)
        with self.executor(lower_program(program)) as vm:
            actual=vm.launch(vm.ingest(pack_states(rows,GraphLayout(3,8192,16384,131072))),'type_identity').decode()
        self.assertEqual(actual,[type_identity_probe(row) for row in rows])

    def test_inplace_sort_local_module_closure_stability_and_exception_restore(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        for stem,probe in (('inplace_sort',inplace_sort_probe),
                ('inplace_sort_keys',inplace_sort_keys_probe),
                ('inplace_sort_exception',inplace_sort_exception_probe)):
            program=test_program(stem,stem,probe,own=True)
            packed=pack_states([0,1],GraphLayout(2,8192,16384,131072))
            with self.executor(lower_program(program)) as vm:
                actual=vm.launch(vm.ingest(packed),stem).decode()
            self.assertEqual(actual,[probe(0),probe(1)])
            for result in actual:
                self.assertIs(result[0],result[1])

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except Exception as exc:
            raise unittest.SkipTest('Actual CUDA required; later fixtures: NTE_VM_TEST_PROGRAM_DIR') from exc
        if not torch.cuda.is_available():
            raise unittest.SkipTest('Actual CUDA required; later fixtures: NTE_VM_TEST_PROGRAM_DIR')
        cls._executors = {}

    @classmethod
    def tearDownClass(cls):
        for vm in cls._executors.values():
            vm.close()

    def executor(self, image):
        from contextlib import nullcontext
        if image.identity not in self._executors:
            self._executors[image.identity] = CudaVm(image)
        return nullcontext(self._executors[image.identity])

    def test_constructors_counter_itertools_and_set_difference(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        rows = [[], [3], [3, 1, 2, 1]]
        program = test_program('constructors', 'constructors', constructors_probe, own=True)
        packed = pack_states(rows, GraphLayout(len(rows), 8192, 16384, 131072))
        with self.executor(lower_program(program)) as vm:
            self.assertTrue(vm.capability('constructors')['ready'])
            actual = vm.launch(vm.ingest(packed), 'constructors').decode()
        self.assertEqual(actual, [constructors_probe(row) for row in rows])

    def test_nested_callback_generator_keys_and_stable_ties(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        rows = [[], [4, -1, 2], [2, 2, -3, 1]]
        program = test_program('callbacks', 'callbacks', callback_probe, own=True)
        packed = pack_states(rows, GraphLayout(len(rows), 8192, 16384, 131072))
        with self.executor(lower_program(program)) as vm:
            actual = vm.launch(vm.ingest(packed), 'callbacks').decode()
        self.assertEqual(actual, [callback_probe(row) for row in rows])

    def test_class_enum_super_and_getattr(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        rows = [0, 4, -2]
        program = test_program('classes', 'classes', class_enum_super_probe, own=True)
        packed = pack_states(rows, GraphLayout(len(rows), 4096, 8192, 65536))
        with self.executor(lower_program(program)) as vm:
            actual = vm.launch(vm.ingest(packed), 'classes').decode()
        self.assertEqual(actual, [class_enum_super_probe(row) for row in rows])

    def test_clone_state_copies_without_aliasing_input(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.content.duel_v2 import STARTER_DECKS, validate_deck
        from copy import deepcopy
        deck = next(item for item in STARTER_DECKS if item['id'] == 'starter')
        rows = [new_game(seed=seed, skip_mulligan=True,
                         decks={'a': validate_deck(deepcopy(deck)), 'b': validate_deck(deepcopy(deck))})
                for seed in (8101, 8102)]
        program = test_program('clone', 'clone', clone_probe, own=True)
        packed = pack_states(rows, GraphLayout(len(rows), 4096, 8192, 65536))
        with self.executor(lower_program(program)) as vm:
            arena = vm.ingest(packed)
            before = arena.nodes.clone()
            actual = vm.launch(arena, 'clone').decode()
            self.assertTrue(vm.torch.equal(before, arena.nodes))
        self.assertEqual(actual, [clone_probe(row) for row in rows])
        for row, result in zip(rows, actual):
            self.assertEqual(result[0], row)
            self.assertTrue(result[1])

    def test_uuid_capability_fails_before_device_execution(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        program = test_program('uuid', 'uuid', uuid_probe, own=True)
        with self.executor(lower_program(program)) as vm:
            self.assertFalse(vm.capability('uuid')['ready'])
            arena = vm.ingest(pack_states([1], GraphLayout(1, 64, 64, 1024)))
            with self.assertRaises(VmCapabilityError):
                vm.launch(arena, 'uuid')

    def test_unknown_sorted_keyword_and_pool_capacity_fail_closed(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        program = test_program('keywords', 'keywords', keyword_probe, own=True)
        with self.executor(lower_program(program)) as vm:
            arena = vm.ingest(pack_states([[1, 2]], GraphLayout(1, 1024, 2048, 8192)))
            with self.assertRaises(Exception):
                vm.launch(arena, 'keywords').check()

    def test_ensure_kits_matches_baked_precondition(self):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        program = test_program('ensure', 'ensure', ensure_probe, own=True)
        image = lower_program(program)
        module_index, kits_count = _kits_precondition(image)
        packed = pack_states([3], GraphLayout(1, 1024, 2048, 8192))
        with self.executor(image) as vm:
            if module_index < 0 or kits_count <= 0:
                with self.assertRaises(Exception):
                    vm.launch(vm.ingest(packed), 'ensure').check()
            else:
                self.assertEqual(vm.launch(vm.ingest(packed), 'ensure').decode(), [3])
