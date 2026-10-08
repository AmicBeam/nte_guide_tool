"""Numeric source lowering is not a rule execution/1600-capacity claim."""
from copy import deepcopy
import math
import struct
import unittest
from app.modules.card_game.rl.rule_ir.portable_program import capture_program,FrozenRuleProgram,compute_identity
from app.modules.card_game.rl.batched_duel.vm_image import lower_program,ImageError
from app.modules.card_game.rl.batched_duel.schema import TAG_FLOAT


def arithmetic(value,offset=2):
    if value<0:return -value
    return value+offset


def closure(value):
    return sum(x+value for x in (1,2))


def signed_zero():return -0.0,0.0


class ImageTest(unittest.TestCase):
    def capture(self,**entries):return capture_program(entries,allowed_module_prefixes=(__name__,),scope_character_ids=None,scope_card_ids=None)

    def test_all_instructions_jumps_closures_and_metadata_survive(self):
        p=self.capture(arithmetic=arithmetic,closure=closure);im=lower_program(p)
        self.assertTrue(im.verify());self.assertEqual(len(im.instructions),sum(len(f['instructions']) for f in p.functions+p.codes))
        self.assertEqual(im.functions.shape[1],18)
        for row in im.functions:
            start,n=map(int,row[:2])
            for target in im.instructions[start:start+n,3]:self.assertTrue(target==-1 or start<=target<start+n)
        caps=im.required['closure'];self.assertIn('YIELD_VALUE',caps['opcodes']);self.assertIn('sum',caps['intrinsics'])
        self.assertTrue(im.missing_capabilities('closure')['opcodes'])
        self.assertEqual(im.missing_capabilities('closure',opcodes=caps['opcodes'],intrinsics=caps['intrinsics']),{'opcodes':[],'intrinsics':[]})

    def test_source_and_all_image_storage_are_independent_and_immutable(self):
        p=self.capture(arithmetic=arithmetic);im=lower_program(p);again=lower_program(FrozenRuleProgram(deepcopy(p.data)))
        self.assertEqual(im.identity,again.identity)
        for name in ('instructions','functions','classes','modules','nodes','edges','payload'):
            with self.assertRaises(ValueError):getattr(im,name).setflags(write=True)
        with self.assertRaises(TypeError):im.entries['bad']=1
        p.data['entries']['arithmetic']='bad';self.assertTrue(im.verify())

    def test_negative_zero_bit_pattern_is_preserved(self):
        im=lower_program(self.capture(zero=signed_zero));values=[]
        for tag,offset,count,extra in im.nodes:
            if tag==TAG_FLOAT:values.append(struct.unpack('<d',im.payload[offset:offset+count].tobytes())[0])
        self.assertIn(-1.,[math.copysign(1.,x) for x in values]);self.assertIn(1.,[math.copysign(1.,x) for x in values])

    def test_invalid_source_identity_is_never_lowered(self):
        p=self.capture(arithmetic=arithmetic);p.data['functions'][0]['instructions'][0]['op']='DROP_UNKNOWN_FIELDS'
        with self.assertRaises(Exception):lower_program(p)
        with self.assertRaises(TypeError):lower_program({})


if __name__=='__main__':unittest.main()
