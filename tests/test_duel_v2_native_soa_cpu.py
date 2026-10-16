"""Execute compiled canonical Python operations against authoritative columns."""
import copy
from random import Random
import unittest
import re
import sys

from app.modules.card_game.rl.rule_ir.portable_program import DEFAULT_ALLOWED_MODULE_PREFIXES, capture_program
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema, pack_game_states, unpack_game_states
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU


def mutate_hp(state, amount):
    state['hp'] += amount
    return state


def mutate_alias(state):
    state['left']['n'] += 4
    same = state['left'] is state['right']
    return state, same


def mutate_containers(state):
    state['items'].append({'n': 7})
    state['items'].insert(0, {'n': 8})
    state['items'].extend([{'n': 9}])
    popped = state['items'].pop(1)
    state['mapping']['new'] = 11
    state['mapping'].setdefault('other', 12)
    del state['mapping']['old']
    return state, popped, list(state['mapping'])


def clone_alias(state):
    other = copy.deepcopy(state)
    other['left']['n'] += 20
    return state, other, other['left'] is other['right'], state['left'] is other['left']


def scalar_result(state):
    return state['hp']*2


def sequence_operators(state):
    original=state['items']
    added=original+original
    repeated=original*2
    original*=2
    original[1:3]=[{'n':9}]
    original.reverse()
    return state,added,repeated


def shuffle_items(state):
    rng=Random(state['rng'])
    rng.shuffle(state['items'])
    state['rng']=rng.getrandbits(63)
    return state


def clone_tuple(state):
    other=copy.deepcopy(state)
    return state['pair'] is other['pair']


def clone_transient_views_twice(state):
    wrapper={'left':state['left'],'right':state['right']}
    first=copy.deepcopy(wrapper)
    atomic=copy.deepcopy(state['hp'])
    second=copy.deepcopy(wrapper)
    first['left']['n']+=10
    return state,first,second,first['left'] is first['right'],first['left'] is second['left'],atomic is state['hp']


def clone_empty_twice(state):
    first=copy.deepcopy(state['items'])
    second=copy.deepcopy(state['items'])
    first.append({'n':9})
    return state,first,second,first is second


def public_item(state):
    return {'left':state['left'],'right':state['right']}


def inplace_concat(state):
    original=state['items']
    original+=original
    return state,original is state['items']


def view_schema():
    return compile_schema({
        'Game': {'kind': 'record', 'capacity': 8, 'fields': {
            'hp': 'int', 'rng':'int', 'pair':'@Pair', 'left': '@Item', 'right': '@Item', 'items': '@Items', 'mapping': '@Counters'}},
        'Item': {'kind': 'record', 'capacity': 64, 'fields': {'n': 'int'}},
        'Items': {'kind': 'list', 'capacity': 8, 'max_items': 16, 'value': '@Item'},
        'Counters': {'kind': 'map', 'capacity': 8, 'max_items': 16, 'key': 'str', 'value': 'int'},
        'Pair': {'kind':'tuple','capacity':8,'items':['int','str']},
    }, max_strings=64, max_string_bytes=1024)


@unittest.skipUnless(sys.version_info[:2]==(3,10),'Tiny source capture uses CPython 3.10; other hosts load a frozen program')
class NativeSoACPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        program = capture_program({fn.__name__: fn for fn in (
            mutate_hp, mutate_alias, mutate_containers, clone_alias, scalar_result,
            sequence_operators,shuffle_items,clone_tuple,clone_transient_views_twice,
            clone_empty_twice,public_item,inplace_concat)},
            scope_character_ids=None, scope_card_ids=None,
            allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,))
        cls.schema = view_schema()
        cls.backend = NativeSoACPU(lower_program(program), cls.schema,
            limits=VmLimits(frames=64, locals=128, stack=128, cells=64, steps=100000),
            max_nodes=8192, max_edges=16384, max_payload_bytes=262144)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_direct_column_mutation_and_repeated_persistent_calls(self):
        batch = pack_game_states([{'hp': 1}, {'hp': 7}], self.schema)
        for _ in range(20):
            result = self.backend.call('mutate_hp', batch, [[None, 2], [None, -1]])
            self.assertIs(result.state_result(), batch)
        self.assertEqual(unpack_game_states(batch), [{'hp': 41}, {'hp': -13}])
        self.assertEqual(batch.column('Game', 'hp')[:, 0].tolist(), [41, -13])
        self.assertEqual(self.backend.call('scalar_result', batch, [[None], [None]]).decode(), [82, -26])

    def test_alias_identity(self):
        item = {'n': 2}
        state = {'left': item, 'right': item}
        batch = pack_game_states([state], self.schema)
        actual, same = self.backend.call('mutate_alias', batch, [[None]]).decode()[0]
        self.assertTrue(same)
        self.assertIs(actual['left'], actual['right'])
        self.assertEqual(actual['left']['n'], 6)

    def test_container_writes_preserve_order(self):
        state = {'items': [{'n': 1}], 'mapping': {'old': 2, 'stay': 3}}
        expected = mutate_containers(copy.deepcopy(state))
        batch = pack_game_states([state], self.schema)
        actual = self.backend.call('mutate_containers', batch, [[None]]).decode()[0]
        self.assertEqual(actual, expected)
        self.assertEqual(list(actual[0]['mapping']), ['stay', 'new', 'other'])

    def test_deepcopy_preserves_alias_and_branch_independence(self):
        item = {'n': 3}
        batch = pack_game_states([{'left': item, 'right': item}], self.schema)
        before, after, same, shared = self.backend.call('clone_alias', batch, [[None]]).decode()[0]
        self.assertTrue(same)
        self.assertFalse(shared)
        self.assertEqual(before['left']['n'], 3)
        self.assertEqual(after['left']['n'], 23)

    def test_sequence_operations_preserve_python_alias_semantics(self):
        source={'items':[{'n':1},{'n':2},{'n':3}]}
        expected=sequence_operators(copy.deepcopy(source))
        batch=pack_game_states([source],self.schema)
        actual=self.backend.call('sequence_operators',batch,[[None]]).decode()[0]
        self.assertEqual(actual,expected)
        self.assertIsNot(actual[0]['items'],actual[1])

    def test_shuffle_keeps_exact_random_stream(self):
        source={'rng':1016,'items':[{'n':i} for i in range(10)]}
        expected=shuffle_items(copy.deepcopy(source))
        batch=pack_game_states([source],self.schema)
        actual=self.backend.call('shuffle_items',batch,[[None]]).decode()[0]
        self.assertEqual(actual,expected)

    def test_immutable_tuple_retains_deepcopy_identity(self):
        batch=pack_game_states([{'pair':(7,'x')}],self.schema)
        self.assertEqual(self.backend.call('clone_tuple',batch,[[None]]).decode(),[True])

    def test_transient_copies_with_typed_children_have_independent_memos(self):
        item={'n':3}
        batch=pack_game_states([{'hp':1<<40,'left':item,'right':item}],self.schema)
        state,first,second,aliased,shared,atomic=self.backend.call('clone_transient_views_twice',batch,[[None]]).decode()[0]
        self.assertTrue(aliased);self.assertFalse(shared);self.assertTrue(atomic)
        self.assertEqual([state['left']['n'],first['left']['n'],second['left']['n']],[3,13,3])

    def test_empty_typed_copies_stay_independent(self):
        batch=pack_game_states([{'items':[]}],self.schema)
        state,first,second,shared=self.backend.call('clone_empty_twice',batch,[[None]]).decode()[0]
        self.assertFalse(shared)
        self.assertEqual(state['items'],[]);self.assertEqual(first,[{'n':9}]);self.assertEqual(second,[])

    def test_cuda_launch_signatures_match_generated_kernels(self):
        from app.modules.card_game.rl.batched_duel.native_soa_source import _kernel
        from app.modules.card_game.rl.batched_duel.native_soa_cuda import _SIGNATURE,_EXPORT_SIGNATURE,_SELECT_SIGNATURE
        from app.modules.card_game.rl.batched_duel.device_rule_results import _SPEC,_VALUE_PREFIX
        signatures={'vm_soa_execute':_SIGNATURE,'soa_export_projection':_EXPORT_SIGNATURE,
            'soa_select_result':_SELECT_SIGNATURE,
            'soa_decode_metadata':_SPEC+['int64*','float64*','int32*','uint8*']+['int32*']*6,
            'soa_decode_pair':_VALUE_PREFIX+['int32*']*3+['int32']*4,
            'soa_decode_float':_VALUE_PREFIX+['float64*','int32*']+['int32']*4}
        names={'int*':'int32*','long long*':'int64*','unsigned char*':'uint8*',
               'double*':'float64*','int':'int32'}
        for name,expected in signatures.items():
            kernel=_kernel(self.backend.source,name)
            signature=kernel[:kernel.index('{')]
            params=signature[signature.index('(')+1:signature.rfind(')')]
            actual=[]
            for param in params.split(','):
                dtype=re.sub(r'\b\w+\s*$','',param.strip()).replace('const ','').strip()
                actual.append(names[dtype])
            self.assertEqual(actual,expected,name)
        for line in self.backend.source.splitlines():
            if re.match(r'const (?:int|unsigned char) SOA_\w+\[\]',line.strip()):
                self.fail('CUDA schema table lacks device storage: '+line[:100])

    def test_projection_export_keeps_aliases_and_refuses_game(self):
        from app.modules.card_game.rl.batched_duel.codec import unpack_states
        from app.modules.card_game.rl.batched_duel.record_soa import RecordSoAError
        item={'n':7}
        batch=pack_game_states([{'left':item,'right':item}],self.schema)
        call=self.backend.call('public_item',batch,[[None]])
        actual=unpack_states(call.export_projection())[0]
        self.assertEqual(actual,{'left':{'n':7},'right':{'n':7}})
        self.assertIs(actual['left'],actual['right'])
        call=self.backend.call('mutate_alias',batch,[[None]])
        with self.assertRaises(RecordSoAError):
            call.export_projection()

    def test_inplace_concatenation_uses_columns_and_keeps_identity(self):
        batch=pack_game_states([{'items':[{'n':1},{'n':2}]}],self.schema)
        state,same=self.backend.call('inplace_concat',batch,[[None]]).decode()[0]
        self.assertTrue(same)
        self.assertEqual(state['items'],[{'n':1},{'n':2},{'n':1},{'n':2}])
        self.assertIs(state['items'][0],state['items'][2])


if __name__ == '__main__':
    unittest.main()
