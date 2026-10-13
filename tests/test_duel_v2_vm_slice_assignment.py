"""Actual CUDA slice writes and formal effect-list alias regression."""
from pathlib import Path
import os,sys,unittest


def assign(packet):
    values=packet['values'];alias=values
    rhs=values if packet.get('self',False) else packet['rhs']
    values[packet['start']:packet['stop']:packet['step']]=rhs
    if packet.get('append',False):values.append(99)
    return {'values':values,'alias':alias}


def mismatch(packet):
    values=packet['values'];alias=values
    try:
        values[::2]=packet['rhs']
    except ValueError:
        return {'values':values,'alias':alias,'rejected':True}
    return {'values':values,'alias':alias,'rejected':False}


def refresh(holder):
    from app.modules.card_game.engine.duel_v2.modifiers import add_effect
    add_effect(holder,'probe','attack.panel',2,stacking='refresh',source_key='same')
    alias=holder['effects']
    add_effect(holder,'probe','attack.panel',7,stacking='refresh',source_key='same')
    add_effect(holder,'probe','attack.panel',4,max_stacks=1,source_key='same')
    return {'holder':holder,'alias':alias}


def program():
    from app.modules.card_game.rl.rule_ir.portable_program import (
        capture_program,FrozenRuleProgram,DEFAULT_ALLOWED_MODULE_PREFIXES)
    path=os.environ.get('NTE_VM_SLICE_ASSIGNMENT_PROGRAM')
    if path:return FrozenRuleProgram.from_json(Path(path).read_text())
    if sys.version_info[:2]!=(3,10):raise RuntimeError('Explicit same-source CPython3.10 fixture required')
    return capture_program(dict(assign=assign,mismatch=mismatch,refresh=refresh),
        allowed_module_prefixes=DEFAULT_ALLOWED_MODULE_PREFIXES+(__name__,),
        scope_character_ids=None,scope_card_ids=None)


class SliceAssignmentCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():raise unittest.SkipTest('Actual CUDA required')
        from app.modules.card_game.rl.batched_duel.cuda_vm import CudaVm
        from app.modules.card_game.rl.batched_duel.vm_image import lower_program
        cls.vm=CudaVm(lower_program(program()),guard_standalone_entities=True,
            constant_lookup=True,compact_frame_init=True,device_call_boundaries=True,library_call_boundaries=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'vm'):cls.vm.close()

    def execute(self,entry,values):
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout
        arena=self.vm.ingest(pack_states(values,GraphLayout(len(values),8192,16384,65536)))
        return self.vm.launch(arena,entry).decode()

    def test_insert_replace_remove_self_tuple_and_extended_writes_preserve_alias(self):
        from copy import deepcopy
        cases=[dict(values=[0,1,2,3,4],start=a,stop=b,step=s,rhs=r,append=True)
            for a,b,s,r in [(None,None,None,[8]),(1,3,1,[8,9,10]),(4,1,1,[8]),
                (-99,99,1,()),(None,None,-1,[5,4,3,2,1]),(1,5,2,(8,9)),
                (None,None,10,[8]),(None,None,-10,[8])]]
        cases += [dict(values=[0,1,2],start=1,stop=2,step=1,self=True,append=True),
            dict(values=[],start=None,stop=None,step=None,rhs=[{'unknown':[1,2]}])]
        expected=[assign(deepcopy(p)) for p in cases]
        actual=self.execute('assign',cases);self.assertEqual(actual,expected)
        for row in actual:self.assertIs(row['values'],row['alias'])

    def test_extended_size_rejection_keeps_original_alias_and_contents(self):
        rows=[dict(values=[1,2,3,4],rhs=[9]),dict(values=[1,2,3,4],rhs=[9,8])]
        from copy import deepcopy
        actual=self.execute('mismatch',rows)
        self.assertEqual(actual,[mismatch(deepcopy(p)) for p in rows])
        for row in actual:self.assertIs(row['values'],row['alias'])

    def test_formal_modifier_refresh_and_stack_limit_preserve_effect_list(self):
        from copy import deepcopy
        rows=[{'entity_id':'holder_1'}, {'entity_id':'holder_2','unknown':{'future':[1,2]}}]
        actual=self.execute('refresh',rows)
        self.assertEqual(actual,[refresh(deepcopy(p)) for p in rows])
        for row in actual:self.assertIs(row['holder']['effects'],row['alias'])

    def test_capacity_failure_keeps_original_list_header_and_edges(self):
        import numpy as np
        from app.modules.card_game.rl.batched_duel.codec import pack_states
        from app.modules.card_game.rl.batched_duel.schema import GraphLayout,TAG_LIST
        packet=dict(values=[1,2,3],start=None,stop=None,step=None,rhs=list(range(100)))
        packed=pack_states([packet],GraphLayout(1,512,128,4096))
        ids=np.flatnonzero((packed.nodes[0,:,0]==TAG_LIST)&(packed.nodes[0,:,2]==3))
        self.assertEqual(len(ids),1);ref=int(ids[0]);header=packed.nodes[0,ref].copy()
        offset,count=map(int,header[1:3]);edges=packed.edges[0,offset:offset+count].copy()
        call=self.vm.launch(self.vm.ingest(packed),'assign')
        self.assertEqual(int(call.statuses[0,0].cpu()),2)
        np.testing.assert_array_equal(call.arena.nodes[0,ref].cpu().numpy(),header)
        np.testing.assert_array_equal(call.arena.edges[0,offset:offset+count].cpu().numpy(),edges)


if __name__=='__main__':unittest.main()
