import json
import unittest
from copy import deepcopy
from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, observe
from app.modules.card_game.engine.duel_v2.entities import (
    PlayerEntity, CharacterEntity, CardEntity, clone_state, hydrate_entities, resolve_entity,
)
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, damage, new_character


class EntityTest(unittest.TestCase):
    def game(self):
        deck = next(d for d in STARTER_DECKS if d['id'] == 'weave-rush')
        return new_game(seed=9, first_side='a', skip_mulligan=True, decks={'a':deck, 'b':deck})

    def test_all_objects_are_distinct_entities_and_deterministic(self):
        s=self.game()
        self.assertEqual(s,self.game())
        entities=[]
        for team in s['sides'].values():
            self.assertIsInstance(team,PlayerEntity);entities.append(team)
            for h in team['characters'].values():
                self.assertIsInstance(h,CharacterEntity);entities.append(h)
            for zone in ('hand','deck','discard','removed'):
                for c in team[zone]:
                    self.assertIsInstance(c,CardEntity);entities.append(c)
        self.assertEqual(len(entities),len({e.entity_id for e in entities}))
        a,b=hero(s,'a','baicang'),hero(s,'b','baicang')
        self.assertEqual(a.template_id,b.template_id)
        self.assertFalse(a.is_same_entity(b))
        before=deepcopy(s)
        restored=clone_state(json.loads(json.dumps(s)))
        self.assertEqual(restored,s)
        self.assertIsNot(hero(restored,'a','baicang'),a)
        self.assertTrue(hero(restored,'a','baicang').is_same_entity(a))
        self.assertEqual(s,before)

    def test_card_moves_and_generated_copies_keep_separate_identity(self):
        s=self.game();team=s['sides']['a']
        card=team['deck'].pop();eid=card.entity_id;team['hand'].append(card)
        self.assertIs(resolve_entity(s,eid),card)
        team['hand'].remove(card);team['discard'].append(card)
        restored=clone_state(json.loads(json.dumps(s)))
        self.assertEqual(resolve_entity(restored,eid).entity_id,eid)
        one=card_instance(s,'M01','a',copy=True);two=card_instance(s,'M01','a',copy=True)
        self.assertNotEqual(one.entity_id,two.entity_id)
        self.assertEqual(one.template_id,two.template_id)

    def test_old_json_hydration_is_stable_and_counter_does_not_collide(self):
        s=json.loads(json.dumps(self.game()))
        def strip(value):
            if isinstance(value,dict):
                value.pop('entity_id',None)
                for child in value.values():strip(child)
            elif isinstance(value,list):
                for child in value:strip(child)
        strip(s);s.pop('next_entity',None)
        first=clone_state(s);second=clone_state(s)
        self.assertEqual(first,second)
        eid=card_instance(first,'M01','a').entity_id
        self.assertNotIn(eid,{c.entity_id for t in second['sides'].values() for c in t['hand']+t['deck']})
        self.assertNotIn('entity_id',s['sides']['a'])

    def test_pending_card_restores_same_object_as_zone_reference(self):
        s=self.game();card=s['sides']['a']['hand'][0]
        s['pending_choice']={'side':'a','cards':[card]}
        restored=clone_state(json.loads(json.dumps(s)))
        self.assertIs(restored['pending_choice']['cards'][0],restored['sides']['a']['hand'][0])

    def test_duplicate_ids_are_rejected(self):
        s=self.game();hero(s,'b','baicang')['entity_id']=hero(s,'a','baicang').entity_id
        with self.assertRaises(ValueError):hydrate_entities(s)

    def test_entity_actions_and_hidden_cards(self):
        s=self.game();before=deepcopy(s)
        view=observe(s,'a')
        attack=next(e for e in view['legal_actions'] if e['action']=={'type':'attack','character_id':'baicang'})
        self.assertEqual(apply_action(s,'a',attack['entity_action']),apply_action(s,'a',attack['action']))
        self.assertEqual(s,before)
        self.assertTrue(all('entity_id' not in c for c in view['sides']['b']['hand']))
        with self.assertRaises(ValueError):apply_action(s,'a',{'type':'attack','actor_entity_id':hero(s,'b','baicang').entity_id})
        with self.assertRaises(ValueError):apply_action(s,'a',{'type':'attack','actor_entity_id':'missing'})

    def test_b04_hits_opposing_baicang_and_ultimate_executes_same_template(self):
        for side in ('a','b'):
            with self.subTest(side=side):
                s=self.game();s['active_side']=side;s['sides'][side]['ap']=2
                foe='b' if side=='a' else 'a'
                hero(s,side,'baicang')['hp']=4
                for t in s['sides'].values():t['hand']=[]
                card=card_instance(s,'B04',side);s['sides'][side]['hand']=[card]
                nxt=apply_action(s,side,{'type':'play_card','card_entity_id':card.entity_id})
                self.assertEqual(hero(nxt,side,'baicang')['hp'],4)
                for owner,t in nxt['sides'].items():
                    for cid,h in t['characters'].items():
                        if h.entity_id!=hero(s,side,'baicang').entity_id:
                            self.assertEqual(h['hp'],hero(s,owner,cid)['hp']-2)
                hero(s,side,'baicang')['awakened']=True
                hero(s,foe,'baicang')['hp']=3
                damage(s,foe,'baicang',1,source=side+':baicang')
                self.assertEqual(hero(s,foe,'baicang')['down_turns'],3)

    def test_baicang_ultimate_never_executes_allies(self):
        for side in ('a','b'):
            with self.subTest(side=side):
                s=self.game();s['active_side']=side;s['sides'][side]['ap']=2
                foe='b' if side=='a' else 'a'
                for t in s['sides'].values():t['hand']=[]
                hero(s,side,'baicang').update(hp=4,awakened=True)
                hero(s,side,'zero')['hp']=3
                hero(s,foe,'zero')['hp']=3
                hero(s,foe,'baicang')['hp']=3
                card=card_instance(s,'B04',side);s['sides'][side]['hand']=[card]
                nxt=apply_action(s,side,{'type':'play_card','card_entity_id':card.entity_id})
                self.assertEqual(hero(nxt,side,'zero')['hp'],1)
                self.assertEqual(hero(nxt,side,'zero')['down_turns'],0)
                self.assertEqual(hero(nxt,foe,'zero')['down_turns'],3)
                self.assertEqual(hero(nxt,foe,'baicang')['down_turns'],3)



    def test_entity_mulligan_and_player_target(self):
        s=new_game(seed=5,first_side='a')
        view=observe(s,'a')
        entry=next(e for e in view['legal_actions'] if e['action']['type']=='mulligan' and len(e['action']['card_ids'])==1)
        self.assertIn('card_entity_ids',entry['entity_action'])
        self.assertEqual(apply_action(s,'a',entry['entity_action']),apply_action(s,'a',entry['action']))
        s=self.game();s['sides']['a']['hp']=25;s['sides']['a']['ap']=2
        card=card_instance(s,'Y03','a');s['sides']['a']['hand']=[card]
        nxt=apply_action(s,'a',{'type':'play_card','card_entity_id':card.entity_id,
                             'target_entity_id':s['sides']['a'].entity_id})
        self.assertEqual(nxt['sides']['a']['hp'],30)

    def test_inspect_choice_survives_json_with_card_entity_address(self):
        s=new_game(seed=6,first_side='a',skip_mulligan=True)
        card=card_instance(s,'J01','a');s['sides']['a']['hand']=[card]
        s=apply_action(s,'a',{'type':'play_card','card_entity_id':card.entity_id})
        self.assertEqual(s['phase'],'choice')
        restored=clone_state(json.loads(json.dumps(s)))
        selected=restored['pending_choice']['cards'][0]
        self.assertIsInstance(selected,CardEntity)
        eid=selected.entity_id
        nxt=apply_action(restored,'a',{'type':'choose','choice_entity_id':eid})
        self.assertEqual(nxt['phase'],'playing')
        self.assertTrue(any(c.entity_id==eid for c in nxt['sides']['a']['hand']))



from tests.test_solo_room_flow import RoomFlowTestCase


class EntityPersistenceTest(RoomFlowTestCase):
    def test_old_json_database_load_restores_entity_classes(self):
        import importlib
        repo=importlib.import_module('app.modules.card_game.engine.application.v2_repository')
        storage=importlib.import_module('app.modules.card_game.engine.application.v2_persistence')
        entities=importlib.import_module('app.modules.card_game.engine.duel_v2.entities')
        engine=importlib.import_module('app.modules.card_game.engine.duel_v2')
        state=json.loads(json.dumps(engine.new_game(seed=1,skip_mulligan=True)))
        for team in state['sides'].values():
            team.pop('entity_id')
            for hero in team['characters'].values():hero.pop('entity_id')
            for zone in ('hand','deck','discard','removed'):
                for card in team[zone]:card.pop('entity_id')
        state.pop('next_entity')
        with self.db_module.atomic_transaction():
            player,_=self.dao_module.get_or_create_player('entity-storage')
            room=repo.create_room(player,'pvp')
            repo.replace_run(room,{'game':state,'requests':[]})
        loaded=storage.load(room.id)['game']
        self.assertIsInstance(loaded['sides']['a'],entities.PlayerEntity)
        self.assertIsInstance(next(iter(loaded['sides']['a']['characters'].values())),entities.CharacterEntity)
        self.assertIsInstance(loaded['sides']['a']['hand'][0],entities.CardEntity)
        self.assertEqual(loaded,storage.load(room.id)['game'])
        with self.db_module.atomic_transaction():storage.initialize(room,{'game':loaded,'requests':[]})
        storage._cache.clear()
        self.assertEqual(loaded,storage.load(room.id)['game'])


if __name__ == "__main__":
    unittest.main()
