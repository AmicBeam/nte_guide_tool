"""Unit tests for batched duel lossless graph codec.

Covers full roundtrips of five official new_game states, dynamic fields and
complex types (effects, unknown flags, nested private choices, aliases, tuples,
Unicode/Chinese strings, integers > 2^63), input/output mutation isolation,
cycle and unsupported type rejections, decode boundary checks on corrupt buffers,
clone_rows row indexing, and high-volume packing of 1,600 synthetic states.
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from app.modules.card_game.content.duel_v2.catalog import STARTER_DECKS
from app.modules.card_game.engine.duel_v2.entities import (
    CardEntity,
    CharacterEntity,
    Entity,
    PlayerEntity,
)
from app.modules.card_game.engine.duel_v2.flow import new_game
from app.modules.card_game.rl.batched_duel.codec import (
    CodecCapacityError,
    CodecCycleError,
    CodecDecodeError,
    CodecError,
    CodecUnsupportedTypeError,
    GraphLayout,
    PackedStateBatch,
    clone_rows,
    pack_states,
    unpack_states,
)


def assert_lossless_equal(original: object, decoded: object, path: str = "$") -> None:
    """Strictly assert recursive type, value, insertion order, and entity class equality."""
    if type(original) is not type(decoded):
        raise AssertionError(
            f"Type mismatch at {path}: expected {type(original)}, got {type(decoded)}"
        )

    if original is None:
        return

    if type(original) is bool:
        if original is not decoded:
            raise AssertionError(f"Bool value mismatch at {path}: {original} != {decoded}")
        return

    if type(original) is int:
        if original != decoded:
            raise AssertionError(f"Int value mismatch at {path}: {original} != {decoded}")
        return

    if type(original) is float:
        if math.isnan(original):
            if not math.isnan(decoded):
                raise AssertionError(f"Float NaN mismatch at {path}")
        elif original != decoded:
            raise AssertionError(f"Float value mismatch at {path}: {original} != {decoded}")
        return

    if type(original) is str:
        if original != decoded:
            raise AssertionError(f"String mismatch at {path}: {original!r} != {decoded!r}")
        return

    if type(original) in (list, tuple):
        if len(original) != len(decoded):
            raise AssertionError(
                f"Sequence length mismatch at {path}: {len(original)} != {len(decoded)}"
            )
        for i, (orig_elem, dec_elem) in enumerate(zip(original, decoded)):
            assert_lossless_equal(orig_elem, dec_elem, f"{path}[{i}]")
        return

    if isinstance(original, dict):
        orig_keys = list(original.keys())
        dec_keys = list(decoded.keys())
        if orig_keys != dec_keys:
            raise AssertionError(
                f"Dict key order/content mismatch at {path}:\nExpected: {orig_keys}\nGot:      {dec_keys}"
            )
        for k in orig_keys:
            assert_lossless_equal(original[k], decoded[k], f"{path}.{k}")

        if isinstance(original, Entity):
            if original.entity_id != decoded.entity_id:
                raise AssertionError(
                    f"Entity ID mismatch at {path}: {original.entity_id} != {decoded.entity_id}"
                )
            if original.side != decoded.side:
                raise AssertionError(
                    f"Entity side mismatch at {path}: {original.side} != {decoded.side}"
                )
        if isinstance(original, CharacterEntity):
            if original.template_id != decoded.template_id:
                raise AssertionError(
                    f"Character template ID mismatch at {path}: {original.template_id} != {decoded.template_id}"
                )
        if isinstance(original, CardEntity):
            if original.template_id != decoded.template_id:
                raise AssertionError(
                    f"Card template ID mismatch at {path}: {original.template_id} != {decoded.template_id}"
                )
        return

    raise AssertionError(f"Unhandled type at {path}: {type(original)}")


class DuelV2BatchCodecTest(unittest.TestCase):
    def test_noncanonical_graph_and_depth_fail_closed(self):
        layout=GraphLayout(1,32,32,256,max_depth=4)
        with self.assertRaises(ValueError):
            GraphLayout(1,32,32,256,max_depth=257)
        with self.assertRaises(ValueError):
            GraphLayout(1,2**31,32,256)
        with self.assertRaises(CodecCapacityError):
            pack_states([[[[[[]]]]]],layout)

        # A wire graph can contain a cycle even though the ingress encoder rejects it.
        packet=pack_states([[1]],layout)
        packet.edges[0,0,0]=packet.roots[0]
        with self.assertRaises(CodecDecodeError):
            unpack_states(packet)

        packet=pack_states([{'a':1,'b':2}],layout)
        root=int(packet.roots[0]);offset=int(packet.nodes[0,root,1])
        packet.edges[0,offset+1,0]=packet.edges[0,offset,0]
        with self.assertRaises(CodecDecodeError):
            unpack_states(packet)

        packet=pack_states([None],layout)
        packet.counts[0,0]=2
        with self.assertRaises(CodecDecodeError):
            unpack_states(packet)

        class UnknownPlayer(PlayerEntity):
            pass
        value=UnknownPlayer.__new__(UnknownPlayer);dict.__init__(value,side='a')
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([value],layout)

    def setUp(self) -> None:
        self.standard_layout = GraphLayout(
            game_capacity=8,
            max_nodes=16384,
            max_edges=16384,
            max_payload_bytes=131072,
        )

    def test_layout_validation(self) -> None:
        """GraphLayout rejects booleans, non-integers, and non-positive capacities."""
        with self.assertRaises(TypeError):
            GraphLayout(game_capacity=True, max_nodes=100, max_edges=100, max_payload_bytes=100)
        with self.assertRaises(TypeError):
            GraphLayout(game_capacity=10, max_nodes=False, max_edges=100, max_payload_bytes=100)
        with self.assertRaises(TypeError):
            GraphLayout(game_capacity=10, max_nodes=100, max_edges=1.5, max_payload_bytes=100)
        with self.assertRaises(TypeError):
            GraphLayout(game_capacity=10, max_nodes=100, max_edges=100, max_payload_bytes="100")

        with self.assertRaises(ValueError):
            GraphLayout(game_capacity=0, max_nodes=100, max_edges=100, max_payload_bytes=100)
        with self.assertRaises(ValueError):
            GraphLayout(game_capacity=10, max_nodes=-1, max_edges=100, max_payload_bytes=100)
        with self.assertRaises(ValueError):
            GraphLayout(game_capacity=10, max_nodes=100, max_edges=0, max_payload_bytes=100)
        with self.assertRaises(ValueError):
            GraphLayout(game_capacity=10, max_nodes=100, max_edges=100, max_payload_bytes=-5)

        layout = GraphLayout(10, 100, 200, 300)
        self.assertEqual(layout.game_capacity, 10)
        self.assertEqual(layout.max_nodes, 100)
        self.assertEqual(layout.max_edges, 200)
        self.assertEqual(layout.max_payload_bytes, 300)

    def test_official_five_new_game_roundtrip(self) -> None:
        """Verify full roundtrip for five official new_game states with recursive type checks."""
        official_decks = list(STARTER_DECKS)
        self.assertGreaterEqual(len(official_decks), 1)

        games = []
        for i in range(5):
            deck = official_decks[i % len(official_decks)]
            st = new_game(seed=2026 + i, decks={'a': deck, 'b': deck})
            games.append(st)

        batch = pack_states(games, self.standard_layout)
        self.assertIsInstance(batch, PackedStateBatch)
        self.assertEqual(batch.n_rows, 5)
        self.assertGreater(batch.allocated_bytes, 0)

        unpacked = unpack_states(batch)
        self.assertEqual(len(unpacked), 5)

        for i in range(5):
            orig = games[i]
            dec = unpacked[i]

            # Detailed root and container class assertions
            self.assertIs(type(dec['sides']['a']), PlayerEntity)
            self.assertIs(type(dec['sides']['b']), PlayerEntity)
            self.assertEqual(dec['sides']['a'].kind, 'player')
            self.assertEqual(dec['sides']['a'].entity_id, orig['sides']['a'].entity_id)

            for side in ('a', 'b'):
                for cid, char_orig in orig['sides'][side]['characters'].items():
                    char_dec = dec['sides'][side]['characters'][cid]
                    self.assertIs(type(char_dec), CharacterEntity)
                    self.assertEqual(char_dec.kind, 'character')
                    self.assertEqual(char_dec.template_id, char_orig.template_id)
                    self.assertEqual(char_dec.entity_id, char_orig.entity_id)

                for zone in ('hand', 'deck'):
                    for orig_card, dec_card in zip(orig['sides'][side][zone], dec['sides'][side][zone]):
                        self.assertIs(type(dec_card), CardEntity)
                        self.assertEqual(dec_card.kind, 'card')
                        self.assertEqual(dec_card.template_id, orig_card.template_id)
                        self.assertEqual(dec_card.entity_id, orig_card.entity_id)

            # Full recursive type, value, and ordering equality
            assert_lossless_equal(orig, dec)

    def test_dynamic_fields_and_complex_types(self) -> None:
        """Verify explicit dynamic fields, unknown flags, choice, aliases, tuples, Chinese, >2^63 ints."""
        base_game = new_game(seed=42)
        shared_card = base_game['sides']['a']['hand'][0]
        self.assertIsInstance(shared_card, CardEntity)

        # 1. Effects
        base_game['effects'] = [
            {'effect_id': 'dynamic_atk_buff', 'amount': 4, 'duration': 2, 'modifiers': [{'factor': 1.25}]}
        ]

        # 2. Unknown flags & deferred metadata
        base_game['flags'] = {
            'custom_collapse_marker': True,
            'secret_round_flag': 'v2_flag_xyz',
            'deferred_meta': {'step': 12, 'sub': ('arg1', 'arg2')},
        }

        # 3. Nested private choice referencing shared card
        base_game['pending_choice'] = {
            'side': 'a',
            'kind': 'inspect_top',
            'prompt': '请选择一张牌',
            'cards': [shared_card],
            'options': ('opt_alpha', 'opt_beta', ('nested_1', 'nested_2')),
        }

        # 4. Operation sharing the same CardEntity
        base_game['operation'] = {
            'side': 'a',
            'action': 'play_card',
            'card': shared_card,
        }

        # 5. Tuple vs List distinction
        base_game['tuple_field'] = (10, 'text', 2.71828, (True, False, None))
        base_game['list_field'] = [10, 'text', 2.71828, [True, False, None]]

        # 6. Chinese text and symbols
        base_game['chinese_meta'] = {
            '测试角色': '娜娜莉・灵感迸发',
            '描述': '在对方回合受到伤害时获得 1 点环合值。',
            '状态': '活跃',
            '特效': '✨💥🌟',
        }

        # 7. Greater than 2^63 RNG and large integers
        base_game['rng'] = (1 << 63) + 9876543210
        base_game['huge_pos_int'] = (1 << 120) - 17
        base_game['huge_neg_int'] = -((1 << 120) - 17)

        batch = pack_states([base_game], self.standard_layout)
        unpacked = unpack_states(batch)[0]

        assert_lossless_equal(base_game, unpacked)

        # Verify shared container alias identity
        dec_hand_card = unpacked['sides']['a']['hand'][0]
        dec_choice_card = unpacked['pending_choice']['cards'][0]
        dec_op_card = unpacked['operation']['card']

        self.assertIs(dec_hand_card, dec_choice_card)
        self.assertIs(dec_hand_card, dec_op_card)
        self.assertIsInstance(dec_hand_card, CardEntity)
        self.assertEqual(dec_hand_card.entity_id, shared_card.entity_id)
        self.assertEqual(dec_hand_card.template_id, shared_card.template_id)

        # Verify tuple vs list
        self.assertIs(type(unpacked['tuple_field']), tuple)
        self.assertIs(type(unpacked['tuple_field'][3]), tuple)
        self.assertIs(type(unpacked['list_field']), list)
        self.assertIs(type(unpacked['list_field'][3]), list)

        # Verify 63-bit+ ints
        self.assertEqual(unpacked['rng'], (1 << 63) + 9876543210)
        self.assertEqual(unpacked['huge_pos_int'], (1 << 120) - 17)
        self.assertEqual(unpacked['huge_neg_int'], -((1 << 120) - 17))

    def test_input_and_output_modification_isolation(self) -> None:
        """Mutations to inputs after packing or outputs after unpacking must not corrupt state."""
        st = new_game(seed=77)
        batch = pack_states([st], self.standard_layout)

        # Mutate original input state
        st['turn'] = 9999
        st['sides']['a']['hp'] = -100
        st['injected_key'] = 'tainted'
        st['sides']['a']['hand'].clear()

        # Unpack first copy
        copy1 = unpack_states(batch)[0]
        self.assertEqual(copy1['turn'], 0)
        self.assertEqual(copy1['sides']['a']['hp'], 30)
        self.assertNotIn('injected_key', copy1)
        self.assertGreater(len(copy1['sides']['a']['hand']), 0)

        # Mutate unpacked copy1
        copy1['turn'] = 8888
        copy1['sides']['a']['hp'] = 5
        copy1['sides']['a']['hand'].pop()

        # Unpack second copy from the same batch
        copy2 = unpack_states(batch)[0]
        self.assertEqual(copy2['turn'], 0)
        self.assertEqual(copy2['sides']['a']['hp'], 30)
        self.assertNotEqual(len(copy1['sides']['a']['hand']), len(copy2['sides']['a']['hand']))
        self.assertIsNot(copy1, copy2)

    def test_cycle_and_unsupported_rejection(self) -> None:
        """Codec strictly rejects circular references, sets, bytes, non-finite floats, non-str keys."""
        # 1. Circular dict
        cycle_dict: dict = {}
        cycle_dict['self'] = cycle_dict
        with self.assertRaises((CodecCycleError, CodecUnsupportedTypeError)):
            pack_states([cycle_dict], self.standard_layout)

        # 2. Circular list
        cycle_list: list = []
        cycle_list.append(cycle_list)
        with self.assertRaises((CodecCycleError, CodecUnsupportedTypeError)):
            pack_states([cycle_list], self.standard_layout)

        # 3. Indirect cycle
        a: list = []
        b: list = [a]
        a.append(b)
        with self.assertRaises((CodecCycleError, CodecUnsupportedTypeError)):
            pack_states([a], self.standard_layout)

        # 4. Sets
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'invalid_set': {1, 2, 3}}], self.standard_layout)

        # 5. Bytes
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'raw_bytes': b'not_json'}], self.standard_layout)

        # 6. Non-finite floats
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'nan': float('nan')}], self.standard_layout)
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'inf': float('inf')}], self.standard_layout)
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'neginf': float('-inf')}], self.standard_layout)

        # 7. Non-string dict keys
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{123: 'integer_key'}], self.standard_layout)
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{(1, 2): 'tuple_key'}], self.standard_layout)

        # 8. Custom unsupported class
        class CustomObj:
            pass
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'custom': CustomObj()}], self.standard_layout)

        # 9. Unknown Entity subclass
        class UnknownEntity(Entity):
            pass
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'entity': UnknownEntity({'entity_id': 'unk'})}], self.standard_layout)

        # 10. NumPy scalars
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'np_int': np.int32(42)}], self.standard_layout)
        with self.assertRaises(CodecUnsupportedTypeError):
            pack_states([{'np_float': np.float64(3.14)}], self.standard_layout)

    def test_capacity_errors(self) -> None:
        """Capacity limits must raise CodecCapacityError and reject the entire batch."""
        tiny_nodes = GraphLayout(game_capacity=2, max_nodes=5, max_edges=10, max_payload_bytes=100)
        with self.assertRaises(CodecCapacityError):
            pack_states([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}], tiny_nodes)

        tiny_edges = GraphLayout(game_capacity=2, max_nodes=50, max_edges=2, max_payload_bytes=100)
        with self.assertRaises(CodecCapacityError):
            pack_states([{'a': 1, 'b': 2, 'c': 3}], tiny_edges)

        tiny_payload = GraphLayout(game_capacity=2, max_nodes=50, max_edges=50, max_payload_bytes=5)
        with self.assertRaises(CodecCapacityError):
            pack_states([{'text': 'a long string exceeding limit'}], tiny_payload)

        tiny_games = GraphLayout(game_capacity=2, max_nodes=50, max_edges=50, max_payload_bytes=100)
        with self.assertRaises(CodecCapacityError):
            pack_states([{'a': 1}, {'b': 2}, {'c': 3}], tiny_games)

    def test_decode_validation_on_corrupt_buffers(self) -> None:
        """Corrupted arrays must be rejected with CodecDecodeError without crashing or out-of-bounds reads."""
        batch = pack_states([{'a': [1, 2], 'name': 'test'}], self.standard_layout)

        # 1. Corrupt root index
        corrupt_roots = batch.roots.copy()
        corrupt_roots[0] = 9999
        b_bad_root = PackedStateBatch(
            batch.layout, batch.nodes.copy(), batch.edges.copy(), batch.payload.copy(), batch.counts.copy(), corrupt_roots
        )
        with self.assertRaises(CodecDecodeError):
            unpack_states(b_bad_root)

        # 2. Corrupt negative node count
        corrupt_counts = batch.counts.copy()
        corrupt_counts[0, 0] = -1
        b_bad_counts = PackedStateBatch(
            batch.layout, batch.nodes.copy(), batch.edges.copy(), batch.payload.copy(), corrupt_counts, batch.roots.copy()
        )
        with self.assertRaises(CodecDecodeError):
            unpack_states(b_bad_counts)

        # 3. Corrupt node count exceeding layout max_nodes
        corrupt_counts2 = batch.counts.copy()
        corrupt_counts2[0, 0] = self.standard_layout.max_nodes + 1
        b_bad_counts2 = PackedStateBatch(
            batch.layout, batch.nodes.copy(), batch.edges.copy(), batch.payload.copy(), corrupt_counts2, batch.roots.copy()
        )
        with self.assertRaises(CodecDecodeError):
            unpack_states(b_bad_counts2)

        # 4. Corrupt invalid tag
        corrupt_nodes = batch.nodes.copy()
        corrupt_nodes[0, 0, 0] = 99
        b_bad_tag = PackedStateBatch(
            batch.layout, corrupt_nodes, batch.edges.copy(), batch.payload.copy(), batch.counts.copy(), batch.roots.copy()
        )
        with self.assertRaises(CodecDecodeError):
            unpack_states(b_bad_tag)

        # 5. Corrupt edge child reference
        corrupt_edges = batch.edges.copy()
        corrupt_edges[0, 0, 0] = 9999
        b_bad_edge = PackedStateBatch(
            batch.layout, batch.nodes.copy(), corrupt_edges, batch.payload.copy(), batch.counts.copy(), batch.roots.copy()
        )
        with self.assertRaises(CodecDecodeError):
            unpack_states(b_bad_edge)

        # 6. Corrupt payload offset
        corrupt_nodes_offset = batch.nodes.copy()
        corrupt_nodes_offset[0, 1, 1] = 99999
        b_bad_offset = PackedStateBatch(
            batch.layout, corrupt_nodes_offset, batch.edges.copy(), batch.payload.copy(), batch.counts.copy(), batch.roots.copy()
        )
        with self.assertRaises(CodecDecodeError):
            unpack_states(b_bad_offset)

    def test_clone_rows(self) -> None:
        """clone_rows produces independent C-contiguous batches with out-of-order and repeated indices."""
        sA = {'name': 'A', 'items': [1, 2]}
        sB = {'name': 'B', 'items': [3, 4]}
        sC = {'name': 'C', 'items': [5, 6]}

        batch = pack_states([sA, sB, sC], self.standard_layout)
        self.assertEqual(batch.n_rows, 3)

        cloned = clone_rows(batch, [2, 0, 2, 1])
        self.assertEqual(cloned.n_rows, 4)
        self.assertTrue(cloned.nodes.flags.c_contiguous)
        self.assertTrue(cloned.edges.flags.c_contiguous)
        self.assertTrue(cloned.payload.flags.c_contiguous)
        self.assertTrue(cloned.counts.flags.c_contiguous)
        self.assertTrue(cloned.roots.flags.c_contiguous)

        unpacked = unpack_states(cloned)
        self.assertEqual(len(unpacked), 4)
        assert_lossless_equal(sC, unpacked[0])
        assert_lossless_equal(sA, unpacked[1])
        assert_lossless_equal(sC, unpacked[2])
        assert_lossless_equal(sB, unpacked[3])

        # Verify rows 0 and 2 are distinct independent objects in memory
        self.assertIsNot(unpacked[0], unpacked[2])
        unpacked[0]['items'].append(999)
        self.assertEqual(unpacked[2]['items'], [5, 6])

        # Strict validation on invalid indices
        with self.assertRaises(IndexError):
            clone_rows(batch, [-1])
        with self.assertRaises(IndexError):
            clone_rows(batch, [3])
        with self.assertRaises(TypeError):
            clone_rows(batch, [True])
        with self.assertRaises(TypeError):
            clone_rows(batch, [1.5])

        tiny_cap = GraphLayout(game_capacity=2, max_nodes=50, max_edges=50, max_payload_bytes=100)
        batch_tiny = pack_states([sA, sB], tiny_cap)
        with self.assertRaises(CodecCapacityError):
            clone_rows(batch_tiny, [0, 1, 0])

    def test_synthetic_1600_batch_pack_unpack(self) -> None:
        """Verify packing and unpacking of 1,600 small synthetic states in true contiguous arrays."""
        synthetic_states = []
        for i in range(1600):
            st = {
                'id': i,
                'code': f'state_{i}',
                'active': bool(i % 2 == 0),
                'rate': float(i) * 0.125,
                'scores': [i, i + 1, i + 2],
                'pair': ('fixed', i),
                'nested': {'val': i * 10, 'flag': None},
            }
            synthetic_states.append(st)

        layout = GraphLayout(
            game_capacity=1600,
            max_nodes=32,
            max_edges=32,
            max_payload_bytes=256,
        )

        batch = pack_states(synthetic_states, layout)
        self.assertEqual(batch.n_rows, 1600)
        self.assertGreater(batch.allocated_bytes, 0)
        self.assertEqual(batch.nodes.shape, (1600, 32, 4))
        self.assertEqual(batch.edges.shape, (1600, 32, 2))
        self.assertEqual(batch.payload.shape, (1600, 256))
        self.assertEqual(batch.counts.shape, (1600, 3))
        self.assertEqual(batch.roots.shape, (1600,))

        unpacked = unpack_states(batch)
        self.assertEqual(len(unpacked), 1600)

        # Spot check diverse row indices
        for check_idx in (0, 1, 50, 100, 500, 1000, 1599):
            assert_lossless_equal(synthetic_states[check_idx], unpacked[check_idx])


if __name__ == '__main__':
    unittest.main()
