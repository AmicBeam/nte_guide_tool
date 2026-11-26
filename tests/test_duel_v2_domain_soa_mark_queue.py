"""Formal deferred marked-card snapshots retain distinct CardEntity identity."""
from copy import deepcopy
import unittest
from app.modules.card_game.engine.duel_v2.state import card_instance
from app.modules.card_game.engine.duel_v2.flow import operation,finish_operation
from app.modules.card_game.rl.batched_duel.domain_soa_codec import encode_domain_states,_decode_row
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout,DomainSoACapacityError,DomainSoAUnsupportedError
from tests.test_duel_v2_domain_soa_character_queries import state

class MarkQueueContractTest(unittest.TestCase):
    def formal(self):
        st=state(143,a='quick-rush');h=st['sides']['a']['characters']['xiaozhi']
        card=card_instance(st,'Q01','a');card['jingu_mark']={'delta':-1,'source_entity_id':h['entity_id']}
        operation(st,'a','xiaozhi',card);st['_response_intercept']=True
        finish_operation(st)
        self.assertEqual(len(st['_deferred_card_marks']),1)
        return st
    def test_formal_snapshot_and_zone_card_remain_distinct(self):
        st=self.formal();b=encode_domain_states([st],presentation='drop');out=_decode_row(b,0)
        queue=out['_deferred_card_marks'];self.assertEqual(queue,st['_deferred_card_marks'])
        queued=queue[0]['card'];physical=next(c for c in out['sides']['a']['discard']if c['entity_id']==queued['entity_id'])
        self.assertIsNot(physical,queued);self.assertEqual(sum(b.card_is_snapshot[0]),1)
        self.assertEqual(out['operation'],st['operation'])
    def test_queue_empty_absence_and_capacity_are_distinct(self):
        for present in (False,True):
            st=state(143)
            if present:st['_deferred_card_marks']=[]
            b=encode_domain_states([st],presentation='drop');out=_decode_row(b,0)
            self.assertEqual('_deferred_card_marks'in out,present)
        st=self.formal();st['_deferred_card_marks']*=2
        with self.assertRaises(DomainSoACapacityError):encode_domain_states([st],DomainSoALayout(max_deferred_marks=1),presentation='drop')
    def test_queue_support_does_not_weaken_duplicate_physical_rejection(self):
        st=self.formal();st['sides']['a']['hand'].append(st['sides']['a']['discard'][-1])
        with self.assertRaises(DomainSoAUnsupportedError):encode_domain_states([st],presentation='drop')
    def test_entity_return_marker_roundtrip(self):
        st=state(143);cid=next(iter(st['sides']['a']['characters']));operation(st,'a',cid)
        st['operation']['return_after_resources']=st['sides']['a']['characters'][cid]['entity_id']
        b=encode_domain_states([st],presentation='drop');self.assertEqual(_decode_row(b,0)['operation'],st['operation'])

if __name__=='__main__':unittest.main()
