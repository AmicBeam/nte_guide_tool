"""Lossless typed in-flight sortie contract across choice suspension."""
from copy import deepcopy
import unittest
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row,encode_domain_states
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoALayout,SORTIE_INT_KEYS,SORTIE_BOOL_KEYS,SORTIE_ATTRIBUTES
from tests.test_duel_v2_domain_soa_character_queries import state

class PendingSortieContractTest(unittest.TestCase):
    def test_all_spec_fields_and_storage_target_roundtrip(self):
        for side in ('a','b'):
            for attribute in SORTIE_ATTRIBUTES:
                st=state(106);actor=next(iter(st['sides'][side]['characters']));foe='b'if side=='a'else'a';target=next(iter(st['sides'][foe]['characters']))
                spec={key:i+257 for i,key in enumerate(SORTIE_INT_KEYS)}
                spec.update({key:bool(i%2)for i,key in enumerate(SORTIE_BOOL_KEYS)});spec.update(attribute=attribute,target=(foe,target))
                st['operation']={'side':side,'actor':actor,'first_bonus':0,'redeem_bonus':0,'record_candidate':False,'participants':[],'card':None,'pending_sortie':{'spec':spec,'stay_bench':True}}
                b=encode_domain_states([st],DomainSoALayout(),presentation="drop");got=_decode_row(b,0)
                self.assertEqual(got['operation']['pending_sortie'],st['operation']['pending_sortie'])
    def test_empty_and_explicit_zero_are_distinct(self):
        for spec in ({},{'bonus':0,'no_counter':False},{'target':('b','player')}):
            st=state(106);actor=next(iter(st['sides']['a']['characters']));st['operation']={'side':'a','actor':actor,'first_bonus':0,'redeem_bonus':0,'record_candidate':False,'participants':[],'card':None,'pending_sortie':{'spec':deepcopy(spec),'stay_bench':False}}
            b=encode_domain_states([st],DomainSoALayout(),presentation="drop");self.assertEqual(_decode_row(b,0)['operation']['pending_sortie'],st['operation']['pending_sortie'])
    def test_unknown_and_wrong_typed_spec_rejected(self):
        for spec in ({'unknown':1},{'no_counter':1},{'hits':True},{'penetrate_brand':2**40},{'target':('b','missing')}):
            st=state(106);actor=next(iter(st['sides']['a']['characters']));st['operation']={'side':'a','actor':actor,'first_bonus':0,'redeem_bonus':0,'record_candidate':False,'participants':[],'card':None,'pending_sortie':{'spec':spec,'stay_bench':False}}
            with self.assertRaises((ValueError,TypeError,OverflowError)):encode_domain_states([st],DomainSoALayout(),presentation="drop")

if __name__=='__main__':unittest.main()
