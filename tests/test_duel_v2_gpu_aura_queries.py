"""Opt-in generator-only source/CPU proof; never actual CUDA qualification."""
import ast
import copy
import sys
import unittest
from unittest.mock import patch

from app.modules.card_game.engine.duel_v2 import modifiers,new_game
from app.modules.card_game.engine.duel_v2.context import EffectContext
from app.modules.card_game.engine.duel_v2.state import new_character
from app.modules.card_game.content.duel_v2 import CARDS,PUBLIC_CHARACTER_IDS
from app.modules.card_game.content.duel_v2.registry import KITS
from app.modules.card_game.rl.batched_duel import gpu_aura_queries as aura
from scripts.validate_duel_v2_native_soa import compare_boundary


def inventory(cid='zhenhong'):
    game=new_game(seed=20261018)
    ids=list(dict.fromkeys((cid,'bohe','xiaozhi','adler','zhenhong','nanali')))[:4]
    for side,team in game['sides'].items():
        team['characters']={name:new_character(name,entity_id=side+':aura:'+name,side=side)for name in ids}
        team['order']=ids[:];team['front']=ids[0];team['last_front']=ids[0]
        for name,hero in team['characters'].items():
            hero['effects']=[modifiers.contribution('test.stored','attack.panel',2,source_entity_id=hero['entity_id']),
                modifiers.contribution('test.cost','card.cost',-1,source_entity_id=hero['entity_id'])]
            hero['flags'].update(atk_buff=1,timed_attack=[dict(left=2,amount=1)])
            if name=='bohe':hero['shape']=next(key for key,card in CARDS.items()if card['character_id']=='bohe' and 'enemy_turn_attack'in card)
            if name=='xiaozhi':hero.update(shape='Q08',jingu=6)
            if name=='adler':hero['shape']='D07'
            if name=='zhenhong':hero['awakened']=True
    subject=game['sides']['a']['characters'][cid]
    game['aura_alias_probe']=[subject,subject,subject['effects']]
    return game,subject


class AuraQueryTests(unittest.TestCase):
    def compare(self,game,cid,property,query=None,extra=()):
        subject=game['sides']['a']['characters'][cid];before=copy.deepcopy(game)
        expected=modifiers.evaluate(game,'a',subject,property,7,query=query,extra=extra)
        actual=aura.query_evaluate(game,'a',subject,property,7,query=query,extra=extra)
        frozen=aura.capture_aura_evaluate()(game,'a',subject,property,7,query=query,extra=extra)
        compare_boundary(actual,expected,'CPU aura complete result/contribution order')
        compare_boundary(frozen,expected,'frozen-body CPU aura complete result')
        compare_boundary(game,before,'real inventory full input graph/RNG/alias purity')
    def test_all17_actual_character_inventories_properties_and_order(self):
        self.assertEqual(len(PUBLIC_CHARACTER_IDS),17)
        for cid in PUBLIC_CHARACTER_IDS:
            game,_=inventory(cid)
            for property in ('attack.panel','attack.override','attack.hit','card.cost','card.instant',
                             'rule.gain_energy','rule.damage_immune','rule.damage_limit','unmatched.property'):
                with self.subTest(cid=cid,property=property):
                    query=dict(attacker_id='xiaozhi',operation=dict(side='a',card=dict(type='battle')))
                    extra=(modifiers.contribution('test.extra',property,3,op='add',layer=2),)
                    self.compare(game,cid,property,query,extra)
    def test_empty_generators_skipped_but_all_context_constructions_retained(self):
        game,subject=inventory();ctor=EffectContext.__init__.__code__
        codes={item[2]for item in aura._PROVIDERS}
        from app.modules.card_game.content.duel_v2.characters.zhenhong import Zhenhong
        other=Zhenhong.aura_effects.__code__
        def counted(fn):
            counts=dict(context=0,known=0,other=0)
            def profile(frame,event,arg):
                if event=='call':
                    if frame.f_code is ctor:counts['context']+=1
                    if frame.f_code in codes:counts['known']+=1
                    if frame.f_code is other:counts['other']+=1
            previous=sys.getprofile();sys.setprofile(profile)
            try:result=fn(game,'a',subject,'card.cost',7,query={})
            finally:sys.setprofile(previous)
            return result,counts
        expected,old=counted(modifiers.evaluate);actual,new=counted(aura.query_evaluate)
        compare_boundary(actual,expected,'generator skip result')
        self.assertEqual(old['context'],new['context']);self.assertEqual(new['context'],8)
        self.assertEqual(old['known'],6);self.assertEqual(new['known'],0)
        self.assertEqual(old['other'],new['other']);self.assertGreater(old['other'],0)
    def test_matching_properties_and_unknown_provider_do_not_skip(self):
        for cls,provider,_,property in aura._PROVIDERS:
            self.assertFalse(aura._frozen_skip_aura(provider,property,{},None,True))
            self.assertTrue(aura._frozen_skip_aura(provider,'unmatched',{},None,True))
        calls=[]
        class Unknown:
            @staticmethod
            def aura_effects(c,property,query):
                calls.append((property,c.cid));c.state['unknown_aura_calls']=c.state.get('unknown_aura_calls',0)+1
                yield modifiers.contribution('unknown',property,1,kind='aura',source_entity_id=c.entity_id)
        Unknown.aura_effects.__name__=aura._BOHE.__name__ # same name grants nothing
        self.assertFalse(aura._frozen_skip_aura(Unknown.aura_effects,'other',{},None,True))
        game,subject=inventory();game['sides']['a']['characters']['unknown']=new_character('bohe',entity_id='unknown',side='a')
        with patch.dict(KITS,unknown=Unknown):
            left=copy.deepcopy(game);right=copy.deepcopy(game)
            a=modifiers.evaluate(left,'a',left['sides']['a']['characters']['zhenhong'],'card.cost',7)
            b=aura.query_evaluate(right,'a',right['sides']['a']['characters']['zhenhong'],'card.cost',7)
        compare_boundary(b,a,'unknown provider full result');compare_boundary(right,left,'unknown mutation full state/RNG/aliases')
        self.assertEqual(len(calls),2)
    def test_custom_property_query_operation_and_get_side_effect_counts_match(self):
        game,subject=inventory()
        def run(fn,make):
            trace=[];property,query=make(trace)
            result=fn(game,'a',subject,property,7,query=query)
            return result,trace
        class Property(str):
            def __ne__(self,other):self.trace.append('property_ne');return super().__ne__(other)
        class Query(dict):
            def __bool__(self):self.trace.append('query_bool');return True
            def get(self,key,default=None):self.trace.append('query_get:'+key);return super().get(key,default)
        class Operation(dict):
            def __bool__(self):self.trace.append('operation_bool');return True
            def get(self,key,default=None):self.trace.append('operation_get:'+key);return super().get(key,default)
        def custom_property(trace):
            value=Property('unmatched');value.trace=trace;return value,{}
        def custom_query(trace):
            value=Query();value.trace=trace;return 'unmatched',value
        def custom_operation(trace):
            value=Operation(card=dict(type='battle'));value.trace=trace;return 'unmatched',dict(operation=value)
        # Even an exact dict can invoke custom key equality from get('operation').
        def colliding_key(trace):
            class Key:
                def __hash__(self):return hash('operation')
                def __eq__(self,other):trace.append('operation_key_eq');return other=='operation'
            return 'unmatched',{Key():dict(card=dict(type='battle'))}
        for make in (custom_property,custom_query,custom_operation,colliding_key):
            with self.subTest(make=make.__name__):
                expected,old=run(modifiers.evaluate,make);actual,new=run(aura.query_evaluate,make)
                compare_boundary(actual,expected,'custom input result');self.assertEqual(new,old)
                frozen,new=run(aura.capture_aura_evaluate(),make)
                compare_boundary(frozen,expected,'frozen-body custom input');self.assertEqual(new,old)
    def test_get_exceptions_keep_type_message_call_count_and_input_graph(self):
        game,subject=inventory();before=copy.deepcopy(game)
        class Operation(dict):
            def get(self,key,default=None):
                self.trace.append(key);raise LookupError('operation-get-marker')
        for fn in (modifiers.evaluate,aura.query_evaluate,aura.capture_aura_evaluate()):
            operation=Operation(card=1);operation.trace=[]
            with self.assertRaisesRegex(LookupError,'operation-get-marker'):
                fn(game,'a',subject,'unmatched',7,query={'operation':operation})
            self.assertEqual(operation.trace,['card'])
            compare_boundary(game,before,'constructor get exception input purity')

    def test_explicit_frozen_body_has_no_unsupported_runtime_code_introspection(self):
        body=aura.capture_aura_evaluate()
        self.assertNotIn('__code__',body.__code__.co_names)
        predicate=body.__globals__['_aura_skip']
        for name in ('__code__','__new__','__setattr__','inspect'):
            self.assertNotIn(name,predicate.__code__.co_names)
        self.assertIn('_aura_ctor',body.__code__.co_varnames)
        self.assertFalse(aura.validate_aura_query_sources()['default_capture_enabled'])

    def test_evaluate_object_code_and_constructor_replacement_are_rejected(self):
        def replacement(*args,**kwargs):return 0
        with patch.object(modifiers,'evaluate',replacement):
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        original=aura._ORIGINAL_EVALUATE.__code__
        aura._ORIGINAL_EVALUATE.__code__=replacement.__code__
        try:
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        finally:aura._ORIGINAL_EVALUATE.__code__=original
        with patch.object(EffectContext,'__init__',replacement):
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()

    def test_source_object_code_ctor_descriptors_and_near_guard_changes_fail_closed(self):
        class PretendEqual:
            def __eq__(self,other):return True
        with patch.object(aura,'_BOHE',PretendEqual()):
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        original=aura._BOHE.__code__
        def changed(c,property,query):
            if False:yield None
        aura._BOHE.__code__=changed.__code__
        try:
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        finally:aura._BOHE.__code__=original
        with patch.object(aura.Bohe,'aura_effects',staticmethod(changed)):
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        with patch.object(EffectContext,'operation',property(lambda self:None),create=True):
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        with patch.object(EffectContext,'__setattr__',lambda self,k,v:object.__setattr__(self,k,v)):
            with self.assertRaises(aura.AuraQuerySpecializationError):aura.capture_aura_evaluate()
        from app.modules.card_game.content.duel_v2.characters.zhenhong import Zhenhong
        with self.assertRaises(aura.AuraQuerySpecializationError):aura._empty_property(Zhenhong.aura_effects)
        for text in ("def p(c,property,query):\n if property != 'x':\n  return 1\n yield 2", 
                     "def p(c,property,query):\n if c and property != 'x':\n  return\n yield 2"):
            tree=ast.parse(text)
            with patch.object(aura,'_source_tree',return_value=tree):
                with self.assertRaises(aura.AuraQuerySpecializationError):aura._empty_property(aura._BOHE)
        self.assertFalse(aura.validate_aura_query_sources()['gpu_qualified'])


if __name__=='__main__':unittest.main()
