"""User-defined 2026-09-18 rules; values are not original-game data."""
import unittest
from app.modules.card_game.content.duel_v2 import CHARACTERS
from app.modules.card_game.engine.duel_v2 import apply_action, legal_actions, new_game
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, attack_value, damage
from app.modules.card_game.engine.duel_v2.flow import begin_turn


def scenario(side='a'):
    s = new_game(seed=17, first_side=side, skip_mulligan=True)
    for t in s['sides'].values():
        t['hand'] = []
        t['ap'] = 10
        t['shield'] = 0
    return s


def give(s, side, cid):
    card = card_instance(s, cid, side)
    s['sides'][side]['hand'].append(card)
    return {'type':'play_card', 'card_id':card['instance_id']}


class September18RulesTest(unittest.TestCase):
    def test_y07_equips_then_heals_only_injured_living_ally(self):
        for side in ('a','b'):
            s=scenario(side);h=hero(s,side,'zero');h['hp']=h['max_hp']-1
            hero(s,side,'nanali').update(hp=0,down_turns=2)
            before=attack_value(h);action=give(s,side,'Y07')
            result=apply_action(s,side,action)
            self.assertEqual(hero(result,side,'zero')['hp'],h['max_hp'])
            self.assertEqual(attack_value(hero(result,side,'zero')),before+1)
            self.assertEqual(hero(result,side,'nanali')['hp'],0)
            self.assertEqual(hero(result,side,'iloy')['hp'],7)
            self.assertEqual(hero(result,side,'iloy')['max_hp'],7)

    def test_y07_no_candidate_no_heal_or_attack_gain(self):
        s=scenario();before={cid:attack_value(h) for cid,h in s['sides']['a']['characters'].items()}
        result=apply_action(s,'a',give(s,'a','Y07'))
        self.assertEqual({cid:attack_value(h) for cid,h in result['sides']['a']['characters'].items()},before)

    def test_y07_heals_once_when_delayed_energy_resolves_not_on_ultimate_button(self):
        s=scenario();hero(s,'a','iloy').update(shape='Y07',hp=7,max_hp=7,energy=5)
        hero(s,'a','zero')['hp']=1
        s=apply_action(s,'a',{'type':'ultimate','character_id':'iloy'})
        self.assertEqual(hero(s,'a','zero')['hp'],1)
        begin_turn(s,'a')
        self.assertEqual(hero(s,'a','zero')['hp'],3)
        begin_turn(s,'a')
        self.assertEqual(hero(s,'a','zero')['hp'],3)

    def test_y07_no_ally_death_damage(self):
        s=scenario();hero(s,'a','iloy')['shape']='Y07'
        damage(s,'a','zero',100,source='b:zero')
        self.assertEqual(s['sides']['b']['hp'],30)
        self.assertFalse(any(e['type']=='damage' and e.get('source')=='a:iloy' for e in s['events']))

    def test_n04_bonus_precedes_own_entry_and_stacks_until_turn_end(self):
        s=scenario()
        # Match the engine's last-departed field through a real normal sortie.
        s=apply_action(s,'a',{'type':'attack','character_id':'zero'})
        hero(s,'a','zero')['harmony']=2
        seq=s['event_seq'];s=apply_action(s,'a',give(s,'a','N04'))
        hits=[e for e in s['events'] if e['seq']>seq and e['type']=='genesis']
        self.assertEqual([e['amount'] for e in hits],[2,2])
        self.assertEqual(s['sides']['a']['genesis_damage_bonus'],1)
        s=apply_action(s,'a',give(s,'a','N04'))
        self.assertEqual(s['sides']['a']['genesis_damage_bonus'],2)
        s=apply_action(s,'a',{'type':'end_turn'})
        self.assertEqual(s['sides']['a']['genesis_damage_bonus'],0)

    def test_n05_targets_enemy_bench_and_is_followup(self):
        for weapon,expected in ((None,4),('N08',6)):
            s=scenario();action=give(s,'a','N05')
            self.assertFalse(any(a['type']=='play_card' and a.get('card_id')==action['card_id'] for a in legal_actions(s)))
            s['sides']['a']['harmony_damage']=True;hero(s,'a','nanali')['shape']=weapon
            targets=[a.get('target_id') for a in legal_actions(s) if a['type']=='play_card' and a.get('card_id')==action['card_id']]
            self.assertEqual(set(targets),{'b:'+cid for cid in s['sides']['b']['order']})
            hero(s,'b','iloy').update(hp=10,max_hp=10)
            r=apply_action(s,'a',{**action,'target_id':'b:iloy'})
            self.assertEqual(hero(r,'b','iloy')['hp'],10-expected)
            self.assertTrue(any(e['type']=='followup' and e['amount']==expected for e in r['events']))
            self.assertIsNone(r['sides']['a']['front'])

    def test_y04_uses_shared_instant_allowance(self):
        for used in (False,True):
            s=scenario();s['sides']['a']['ap']=1;s['sides']['a']['used']['instant']=used
            r=apply_action(s,'a',give(s,'a','Y04'))
            self.assertEqual(r['sides']['a']['ap'],0 if used else 1)
            self.assertTrue(r['sides']['a']['used']['instant'])
            self.assertEqual(hero(r,'a','iloy')['flags']['next_bonus'],1)
            self.assertEqual(hero(r,'a','iloy')['flags']['next_shield'],2)

    def test_bohe_duration_is_at_end(self):
        self.assertEqual(CHARACTERS['bohe']['awakened_passive'],'薄荷免疫对方的非战斗伤害，持续 2 个己方回合。')
