"""Compiled PC/step/error equivalence for optional native stack continuations."""
import sys
import unittest
import numpy as np

from app.modules.card_game.rl.rule_ir.portable_program import capture_program,DEFAULT_ALLOWED_MODULE_PREFIXES
from app.modules.card_game.rl.batched_duel.vm_image import lower_program
from app.modules.card_game.rl.batched_duel.record_soa import compile_schema,pack_game_states,unpack_game_states
from app.modules.card_game.rl.batched_duel.cuda_vm import VmLimits
from app.modules.card_game.rl.batched_duel.native_soa_cpu import NativeSoACPU


def stack_branch(state,n):
    a=7;b=3;c=n;d=a;e=b;f=c
    if n<0:
        f=-f
    else:
        d=e
    for i in range(f%5):
        d=d+i
    state['hp']=state['hp']+d+e
    return state,d,e,f


def stack_unbound(state,n):
    if n:
        value=3
    result=value
    return state,result


def stack_exception(state,n):
    a=3;b=4;c=5;d=a;e=b;f=c
    try:
        if n<0:
            raise ValueError('negative')
        result=d+e+f
    except ValueError:
        result=-n
    finally:
        state['hp']=state['hp']+1
    return state,result


def stack_schema():
    return compile_schema({'Game':dict(kind='record',capacity=8,fields={'hp':'int'})},
                          max_strings=32,max_string_bytes=1024)


@unittest.skipUnless(sys.version_info[:2]==(3,10),'Source freeze requires CPython 3.10')
class NativeStackRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program=capture_program({f.__name__:f for f in (stack_branch,stack_unbound,stack_exception)},
            scope_character_ids=None,scope_card_ids=None,allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,))
        cls.image=lower_program(cls.program);cls.schema=stack_schema()
        limits=VmLimits(frames=32,locals=64,stack=64,cells=32,steps=10000)
        options=dict(limits=limits,max_nodes=8192,max_edges=16384,max_payload_bytes=262144)
        cls.baseline=NativeSoACPU(cls.image,cls.schema,**options)
        cls.fused=NativeSoACPU(cls.image,cls.schema,stack_runs=True,**options)

    @classmethod
    def tearDownClass(cls):
        cls.baseline.close();cls.fused.close()

    def test_compiled_branches_loops_exceptions_and_every_logical_step_cut(self):
        self.assertIn('native stack continuation',self.fused.source)
        self.assertNotIn('native stack continuation',self.baseline.source)
        for entry in ('stack_branch','stack_unbound','stack_exception'):
            for limit in (*range(1,90),10000):
                states=[dict(hp=2),dict(hp=7),dict(hp=11)]
                left=self.baseline.call(entry,pack_game_states(states,self.schema),[[None,n] for n in (-2,0,4)],steps=limit)
                right=self.fused.call(entry,pack_game_states(states,self.schema),[[None,n] for n in (-2,0,4)],steps=limit)
                np.testing.assert_array_equal(left.statuses,right.statuses,err_msg=(entry,limit))
                self.assertEqual(unpack_game_states(left.state),unpack_game_states(right.state),(entry,limit))
                if not left.statuses[:,0].any():
                    self.assertEqual(left.decode(),right.decode(),(entry,limit))

    def test_default_source_stays_identical_and_flag_is_explicit(self):
        from app.modules.card_game.rl.batched_duel.native_soa_source import emit_native_soa_source
        default,_=emit_native_soa_source(self.image,self.schema,self.baseline.limits)
        off,_=emit_native_soa_source(self.image,self.schema,self.baseline.limits,stack_runs=False)
        self.assertEqual(default,off)
        with self.assertRaises(TypeError):emit_native_soa_source(self.image,self.schema,self.baseline.limits,stack_runs=1)


if __name__=='__main__':unittest.main()
