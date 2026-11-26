"""Frozen entity MRO tags and actual constructor parity, without UUID fallback."""
import json
import os
from pathlib import Path
import unittest

from app.modules.card_game.engine.duel_v2.entities import CardEntity, CharacterEntity, PlayerEntity
from app.modules.card_game.engine.events import GameEvent
from app.modules.card_game.rl.rule_ir.portable_program import capture_program, FrozenRuleProgram, DEFAULT_ALLOWED_MODULE_PREFIXES
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.vm_objects import emit_object_source


def construct_entities(value):
    card = CardEntity({'card_id': 'N01', 'unknown': value}, entity_id='card-1', side='a')
    character = CharacterEntity({'id': 'nanally'}, entity_id='character-1', side='b')
    player = PlayerEntity({'hp': 20}, entity_id='player-1', side='a')
    return card, character, player, card.entity_id, character.side, card is card


class HookBox:
    def on_card_used(self, value):
        return value + 7


HOOK_BOX = HookBox()


def enum_attribute_probe(value):
    return (getattr(HOOK_BOX, GameEvent.V2_CARD_USED)(value),
            hasattr(HOOK_BOX, GameEvent.V2_CARD_USED),
            getattr(HOOK_BOX, GameEvent.V2_SURPLUS, 'missing'))


def json_action_key_probe(value):
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def tuple_key_probe(value):
    return max(range(len(value)), key=lambda index: (value[index], -index))


def sequence_order_probe(value):
    left,right=value
    return left<right,left<=right,left>right,left>=right,left==right,left!=right


def keys_view_probe(value):
    keys=value.keys()
    before=list(keys)
    clock=sorted({'turn','remaining'} & keys)
    equal_before=keys==set(before)
    sorted_before=sorted(keys)
    empty_before=bool(keys)
    value['added']=1
    after=list(keys)
    return before,clock,empty_before,after,bool(keys),'added' in keys,equal_before,sorted_before,keys==set(before)


def program():
    fixture = os.environ.get('NTE_ENTITY_TEST_PROGRAM')
    if fixture:
        return FrozenRuleProgram.from_json(Path(fixture).read_text())
    return capture_program({'construct_entities': construct_entities, 'enum_attributes': enum_attribute_probe,
                           'json_action_key': json_action_key_probe, 'tuple_key': tuple_key_probe,
                           'sequence_order': sequence_order_probe, 'keys_view': keys_view_probe},
                           allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES + (__name__,),
                           scope_character_ids=None, scope_card_ids=None)


class EntityConstructionTest(unittest.TestCase):
    def test_frozen_qualified_entity_ids_map_to_correct_heap_tags(self):
        image = lower_program(program())
        source = emit_object_source(image)
        for name, tag in (('CardEntity', 10), ('CharacterEntity', 9), ('PlayerEntity', 8)):
            cid = next(i for i, key in enumerate(image.class_ids) if '.entities:' + name + ':' in key)
            ref = next(-1-i for i, n in enumerate(image.nodes) if n[0] == 13 and n[3] == cid)
            self.assertIn('case ' + str(cid) + ':return ' + str(tag) + ';', source)
            self.assertIn('if(tag==' + str(tag) + ')return ' + str(ref) + ';', source)

    def test_actual_cuda_constructor_mro_fields_and_properties(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        values = [{'future': [1, 2]}, {'future': [3]}]
        with CudaVm(lower_program(program()), guard_standalone_entities=True) as vm:
            actual = vm.launch(vm.ingest(pack_states(values, GraphLayout(2, 4096, 8192, 65536))), 'construct_entities').decode()
        self.assertEqual(actual, [construct_entities(value) for value in values])

    def test_actual_cuda_string_enum_hook_attribute_lookup(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        with CudaVm(lower_program(program()), guard_standalone_entities=True) as vm:
            actual=vm.launch(vm.ingest(pack_states([3,5],GraphLayout(2,4096,8192,65536))),'enum_attributes').decode()
        self.assertEqual(actual,[enum_attribute_probe(3),enum_attribute_probe(5)])

    def test_actual_cuda_rule_action_key_accepts_unused_default_str(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        values=[{'type':'play_card','targets':['b:薄荷'],'cost':0.5,'optional':None},
                {'type':'attack','character_id':'zhenhong','nested':{'shield':2,'instant':False}}]
        with CudaVm(lower_program(program()),guard_standalone_entities=True) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(2,4096,8192,65536))),'json_action_key').decode()
        self.assertEqual(actual,[json_action_key_probe(v) for v in values])

    def test_actual_cuda_teacher_tuple_ranking_and_nested_sequence_order(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        values=[[0.5,0.5,0.4],[-2,-1,-1],[0.0,-0.0,0.0]]
        pairs=[((1,2),(1,3)),((1,),(1,0)),([1,4],[1,3]),
               ((('a',2),1),(('a',2),0)),((1.0,0),(True,0))]
        with CudaVm(lower_program(program()),guard_standalone_entities=True) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(3,4096,8192,65536))),'tuple_key').decode()
            self.assertEqual(actual,[tuple_key_probe(v) for v in values])
            actual=vm.launch(vm.ingest(pack_states(pairs,GraphLayout(5,4096,8192,65536))),'sequence_order').decode()
            self.assertEqual(actual,[sequence_order_probe(v) for v in pairs])

    def test_actual_cuda_live_dictionary_key_view_and_effect_clock_intersection(self):
        from copy import deepcopy
        import torch
        if not torch.cuda.is_available():self.skipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        values=[{}, {'phase':'end','turn':2}, {'remaining':3,'turn':1}]
        with CudaVm(lower_program(program()),guard_standalone_entities=True) as vm:
            actual=vm.launch(vm.ingest(pack_states(values,GraphLayout(3,4096,8192,65536))),'keys_view').decode()
        self.assertEqual(actual,[keys_view_probe(deepcopy(v)) for v in values])
