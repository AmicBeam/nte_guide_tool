"""Formal-state comparisons for emitted direct C++ primitive writes."""
from copy import deepcopy
import unittest
import numpy as np
from app.modules.card_game.engine.duel_v2.modifiers import add_effect,evaluate
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.batched_duel.domain_soa_execution_probe import DomainPrimitiveOracle
from tests.test_duel_v2_domain_soa_character_queries import state,pack


class DomainExecutionPrimitivesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L,unused=pack([state(96)])
        cls.core=DomainPrimitiveOracle(cls.L,allow_cpu_oracle=True)
    @classmethod
    def tearDownClass(cls):cls.core.close()
    def batch(self,st):
        L,fields=pack([st]);self.assertEqual(L,self.L)
        return DomainSoABatch(layout=L,**{k:v.numpy().copy()for k,v in fields.items()})
    def without_presentation(self,st):
        st=deepcopy(st);st.pop('events',None);st.pop('_public_board',None);return st
    def test_compacted_zone_fields_insert_and_replace(self):
        for zone in ({},{'left':1},{'slow':2,'by':'b'},{'slow_turn':7,'tick_side':'a','expires_side':'b'}):
            st=state(96);st['sides']['a']['front_debuff']['delay']=deepcopy(zone);b=self.batch(st)
            self.assertEqual(self.core.run(b,0,0,0,31),0)
            st['sides']['a']['front_debuff']['delay'].update(slow_turn=31,by='b',slow=3,left=2)
            self.assertEqual(_decode_row(b,0),self.without_presentation(st))
            self.assertEqual(self.core.run(b,7,0,0,0),31)
    def test_effect_writes_and_query_match_formal_modifiers(self):
        st=state(96);b=self.batch(st);cid=next(iter(st['sides']['a']['characters']));h=st['sides']['a']['characters'][cid]
        for amount in (2,-3,1):
            self.assertEqual(self.core.run(b,2,0,0,amount),0)
            add_effect(h,'probe.attack','attack.panel',amount,source_entity_id=h['entity_id'],source_key='ability')
            self.assertEqual(_decode_row(b,0),self.without_presentation(st))
        self.assertEqual(self.core.run(b,3,0,0,10),evaluate(st,'a',h,'attack.panel',10)['value'])
        self.assertEqual(self.core.run(b,5,0,0,0),0);h['effects']=[]
        self.assertEqual(_decode_row(b,0),self.without_presentation(st))
    def test_stat_presence_is_preserved_on_mutation(self):
        st=state(96);b=self.batch(st);h=next(iter(st['sides']['b']['characters'].values()))
        self.assertEqual(self.core.run(b,4,1,0,2),0);h.update(hp=2,shield=2,energy=2)
        self.assertEqual(_decode_row(b,0),self.without_presentation(st))
    def test_invalid_side_rolls_back_all_columns(self):
        b=self.batch(state(96));before={k:v.copy()for k,v in ((k,getattr(b,k))for k in self.core.specs)}
        self.assertNotEqual(self.core.run(b,0,7,0,2),0)
        for key,a in before.items():
            if key!='row_error':np.testing.assert_array_equal(getattr(b,key),a,err_msg=key)
    def test_collection_preserves_named_state_and_reclaims_strings(self):
        st=state(96);b=self.batch(st);original=int(b.string_count[0])
        for unused in range(20):self.assertEqual(self.core.run(b,10,0,0,0),0)
        self.assertEqual(int(b.string_count[0]),original+1)
        self.assertEqual(self.core.run(b,9,0,0,0),0)
        self.assertEqual(int(b.string_count[0]),original)
        self.assertEqual(_decode_row(b,0),self.without_presentation(st))
    def test_effect_maximum_preserves_previous_source_metadata(self):
        st=state(96);b=self.batch(st);h=next(iter(st['sides']['a']['characters'].values()))
        for amount in (5,2):
            self.assertEqual(self.core.run(b,11,0,0,amount),0)
            add_effect(h,'probe.maximum','attack.panel',amount,stacking='max',stack_key='maximum',source_entity_id=h['entity_id'],source_key='first'if amount>2 else'second',label='larger'if amount>2 else'smaller')
            self.assertEqual(_decode_row(b,0),self.without_presentation(st))
    def test_real_damage_attribute_conditions_and_bounded_stacking(self):
        st=state(96);b=self.batch(st);h=next(iter(st['sides']['a']['characters'].values()))
        self.assertEqual(self.core.run(b,12,0,0,0),0)
        add_effect(h,'probe.resistance','damage.resistance',-2,source_entity_id=h['entity_id'],source_key='ability',consume_on='damage',conditions=[{'id':'damage_attribute','args':{'attribute':'光'}}])
        self.assertEqual(_decode_row(b,0),self.without_presentation(st))
        for attribute,code in (('光',1),('暗',2)):
            self.assertEqual(self.core.run(b,13,0,0,code),evaluate(st,'a',h,'damage.resistance',0,query={'attribute':attribute})['value'])
        for amount in (1,2,3):
            self.assertEqual(self.core.run(b,14,0,0,amount),0)
            add_effect(h,'probe.bounded','attack.panel',amount,max_stacks=2,clear_on=(),source_entity_id=h['entity_id'],source_key='ability')
            self.assertEqual(_decode_row(b,0),self.without_presentation(st))
    def test_collection_retains_deep_copied_marked_card_queue(self):
        from tests.test_duel_v2_domain_soa_mark_queue import MarkQueueContractTest
        st=MarkQueueContractTest().formal();b=self.batch(st)
        self.assertEqual(self.core.run(b,10,0,0,0),0)
        self.assertEqual(self.core.run(b,9,0,0,0),0)
        self.assertEqual(_decode_row(b,0),self.without_presentation(st))
    def test_explicit_cpu_only_and_unqualified(self):
        self.assertFalse(self.core.supports_full_duel_rules);self.assertFalse(self.core.gpu_qualified)
        with self.assertRaises(ValueError):DomainPrimitiveOracle(self.L)

if __name__=='__main__':unittest.main()
