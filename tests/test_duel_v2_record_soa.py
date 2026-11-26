import unittest

import numpy as np

from app.modules.card_game.rl.batched_duel.record_soa import (
    RecordSoAError, RecordSoACapacityError, RecordSoATypeError,
    compile_schema, pack_game_states, unpack_game_states,
)


def schema(**kwargs):
    return compile_schema({
        'Game': {'kind': 'record', 'capacity': 4, 'fields': {
            'hp': 'int', 'rate': 'float', 'ready': 'bool', 'name': 'str',
            'left': '@Item', 'right': '@Item', 'items': '@Items',
            'mapping': '@Counters', 'pair': '@Pair',
        }},
        'Item': {'kind': 'record', 'capacity': 8, 'fields': {'n': 'int'}},
        'Items': {'kind': 'list', 'capacity': 4, 'max_items': 8, 'value': '@Item'},
        'Counters': {'kind': 'map', 'capacity': 4, 'max_items': 8, 'key': 'str', 'value': 'int'},
        'Pair': {'kind': 'tuple', 'capacity': 4, 'items': ['int', 'str']},
    }, max_strings=32, max_string_bytes=512, **kwargs)


class RecordSoATests(unittest.TestCase):
    def test_alias_order_and_scalar_types(self):
        item = {'n': 5}
        source = {'name': '真红', 'right': item, 'left': item, 'hp': -(1 << 63),
                  'rate': 0.125, 'ready': True, 'items': [item, item],
                  'mapping': {'b': 2, 'a': 1}, 'pair': (7, 'x')}
        packed = pack_game_states([source], schema())
        result = unpack_game_states(packed)[0]
        self.assertEqual(result, source)
        self.assertEqual(list(result), list(source))
        self.assertEqual(list(result['mapping']), ['b', 'a'])
        self.assertIs(result['left'], result['right'])
        self.assertIs(result['items'][1], result['right'])
        self.assertIs(type(result['ready']), bool)
        self.assertIs(type(result['hp']), int)
        self.assertIs(type(result['rate']), float)
        self.assertIs(type(result['pair']), tuple)

    def test_columns_are_authoritative_and_rows_independent(self):
        packed = pack_game_states([{'hp': 3}, {'hp': 6}], schema())
        column = packed.column('Game', 'hp')
        column[:, 0] += [10, 20]
        self.assertEqual(unpack_game_states(packed), [{'hp': 13}, {'hp': 26}])
        branch = packed.clone_rows([0, 0, 1])
        branch.column('Game', 'hp')[1, 0] = 99
        self.assertEqual(unpack_game_states(branch), [{'hp': 13}, {'hp': 99}, {'hp': 26}])
        self.assertEqual(unpack_game_states(packed)[0]['hp'], 13)
        self.assertEqual(packed.clone_rows([]).rows, 0)

    def test_absent_and_null_differ(self):
        packed = pack_game_states([{}, {'hp': None}], schema())
        self.assertEqual(unpack_game_states(packed), [{}, {'hp': None}])

    def test_schema_type_and_integer_bounds_fail_closed(self):
        for source in ({'hp': True}, {'ready': 1}, {'hp': 1 << 63},
                       {'rate': float('nan')}, {'hp': 1, 'unknown': 2}):
            with self.subTest(source=source), self.assertRaises(RecordSoATypeError):
                pack_game_states([source], schema())
        with self.assertRaises(RecordSoAError):
            schema(capacities={'Typo': 1})
        with self.assertRaises(RecordSoAError):
            compile_schema({'Game': {'kind': 'record', 'fields': {'x': 'any'}}})

    def test_capacity_and_corruption_fail_closed(self):
        with self.assertRaises(RecordSoACapacityError):
            pack_game_states([{'items': [{'n': n} for n in range(9)]}], schema())
        packed = pack_game_states([{'hp': 7}], schema())
        game = packed.schema.pools[packed.schema.root]
        packed.buffers['i32'][game.order, 0] = 99
        with self.assertRaises(RecordSoAError):
            unpack_game_states(packed)
        for bad in ([True], [0.5], [-1], [1]):
            with self.subTest(bad=bad), self.assertRaises(RecordSoAError):
                packed.clone_rows(bad)


if __name__ == '__main__':
    unittest.main()
