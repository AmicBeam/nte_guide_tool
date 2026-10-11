"""Source transformation scope and fail-closed validation."""
import unittest
from app.modules.card_game.rl.batched_duel.vm_library_boundaries import LIBRARY_FUNCTIONS,separate_library_calls_source

class LibraryBoundaryTest(unittest.TestCase):
    def test_prototypes_definitions_and_noncode_exact(self):
        source='// __device__ inline int stdlib_call();\nconst char* text="__device__ inline void object_store();";\n/*\n__device__ inline int lookup_super();\n*/\n'
        for name in LIBRARY_FUNCTIONS:
            source+='__device__ inline int '+name+'(int value);\n__device__ inline int '+name+'(int value){return value+1;}\n'
        source+='__device__ inline int untouched(){return 3;}\n'
        variant=separate_library_calls_source(source)
        self.assertEqual(variant.replace('__noinline__','inline'),source)
        self.assertEqual(variant.count('__noinline__'),2*len(LIBRARY_FUNCTIONS))
        self.assertIn('__device__ inline int untouched()',variant)
        one=separate_library_calls_source(source,('stdlib_call',))
        self.assertEqual(one.count('__noinline__'),2)

    def test_invalid_selection_and_source(self):
        for names in [(),('wrong',),('stdlib_call','stdlib_call')]:
            with self.assertRaises((ValueError,TypeError)):separate_library_calls_source('',names)
        for value in [None,3,{}]:
            with self.assertRaises(TypeError):separate_library_calls_source(value)
        for names in ['stdlib_call',None,(3,)]:
            with self.assertRaises(TypeError):separate_library_calls_source('',names)
        with self.assertRaises(ValueError):separate_library_calls_source('// no hooks')

if __name__=='__main__':unittest.main()
