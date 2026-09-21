"""User-authored Haniya tabletop rules, not Everness combat numbers."""
from app.modules.card_game.engine.duel_v2.modifiers import compatibility_flags
import json
import unittest
from copy import deepcopy
from pathlib import Path

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, damage, move_out, attack_value
from app.modules.card_game.engine.duel_v2.projection import preview
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


class HaniyaTest(unittest.TestCase):
    def game(self, skip=True, team=('haniya','zero','anhunqu','xiaozhi')):
        cards = [key for cid in team for key, card in CARDS.items()
                 if card['character_id'] == cid and not card.get('derived')]
        deck = dict(id='haniya-test', name='哈尼娅测试', character_ids=list(team), card_ids=cards)
        s = new_game(seed=71, first_side='a', skip_mulligan=skip, decks={'a':deck,'b':deck})
        if skip:
            for t in s['sides'].values():
                t['hand'] = []; t['shield'] = 0
            s['sides']['a']['ap'] = 10
        return s

    def play(self, s, cid, target=None):
        card = card_instance(s, cid, 'a'); s['sides']['a']['hand'].append(card)
        action = {'type':'play_card','card_id':card['instance_id']}
        if card.get('play_options'): action['option_id'] = card['play_options'][0]['id']
        if target: action['target_id'] = target
        return apply_action(s, 'a', action)

    def attack(self, s, cid):
        return apply_action(s, 'a', {'type':'attack','character_id':cid})

    def next(self, s):
        return apply_action(apply_action(s, 'a', {'type':'end_turn'}), 'b', {'type':'end_turn'})

    def test_catalog_and_initial_resource(self):
        s = self.game(False)
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],0)
        self.assertEqual((CHARACTERS['haniya']['attribute'], CHARACTERS['haniya']['attack'], CHARACTERS['haniya']['max_hp']), ('魂',2,4))
        self.assertEqual(CHARACTERS['haniya']['energy_max'],6)
        self.assertNotIn('haniya', {h['id'] for h in get_catalog(include_test_characters=False)['characters']})
        cards = [c for c in CARDS.values() if c['character_id']=='haniya']
        self.assertEqual(len(cards),8)
        self.assertEqual([c['name'] for c in cards][-2:], ['名为哈尼娅的旋律','引爆全场'])
        for field in ('avatar','portrait'):
            self.assertTrue(Path('app'+CHARACTERS['haniya'][field]).is_file())
        self.assertEqual(hero(self.game(),'a','haniya')['protagonist_aura'],1)

    def test_normal_entry_spends_all_preview_matches_and_front_retry_also_triggers(self):
        s = self.game(); hero(s,'a','haniya')['protagonist_aura']=4
        original=deepcopy(s)
        p=preview(s,'a',{'type':'attack','character_id':'zero'})
        self.assertEqual(s,original)
        s=self.attack(s,'zero')
        self.assertEqual(s['sides']['b']['hp'],24)
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],0)
        self.assertIn('6',str(p))
        s=self.play(s,'V01')
        s=self.attack(s,'zero')
        self.assertEqual(s['sides']['b']['hp'],21)
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],0)

    def test_battle_only_with_weapon_and_set_attack_overwrites(self):
        for weapon, card, attack, remain in [(None,'Z04',4,4), ('V08','Z04',8,0), ('V08','Q01',4,0)]:
            with self.subTest(weapon=weapon,card=card):
                s=self.game(); hero(s,'a','haniya').update(shape=weapon,protagonist_aura=4)
                s=self.play(s,card)
                self.assertEqual(s['sides']['b']['hp'],30-attack)
                self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],remain)
        s=self.game(); hero(s,'a','haniya').update(shape='V08',protagonist_aura=4)
        s['sides']['a']['front']='zero';s=self.play(s,'Z04')
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],0)
        self.assertEqual(s['sides']['b']['hp'],22)

    def test_ultimate_doubles_all_gains_caps_and_expires(self):
        s=self.game();hero(s,'a','haniya')['energy']=6
        s=apply_action(s,'a',{'type':'ultimate','character_id':'haniya'})
        s=self.play(s,'V01');self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],3)
        s=self.play(s,'V02','a:zero');self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],6)
        hero(s,'a','haniya')['protagonist_aura']=0
        s=self.next(s);self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],2)
        s=self.next(s);self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],3)
        self.assertFalse(hero(s,'a','haniya')['awakened'])
        s=self.next(s);self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],4)

    def test_down_retains_but_disables_gaining_and_support_then_recovers(self):
        s=self.game();hero(s,'a','haniya')['protagonist_aura']=4
        damage(s,'a','haniya',99)
        s=self.attack(s,'zero');self.assertEqual(s['sides']['b']['hp'],28)
        s=self.next(s);self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],4)
        s=self.next(s);self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],4)
        s=self.next(s);self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],5)

    def test_selected_free_attacks_reset_and_expiry(self):
        s=self.game();s=self.play(s,'V02','a:zero');s['sides']['a']['ap']=0
        s=self.attack(s,'zero');self.assertEqual(s['sides']['a']['ap'],0)
        self.assertFalse(s['sides']['a']['normal_attack_available'])
        s=self.play(s,'V01');self.assertTrue(s['sides']['a']['normal_attack_available'])
        self.assertNotIn({'type':'attack','character_id':'haniya'},legal_actions(s,'a'))
        s=self.attack(s,'zero');self.assertEqual(s['sides']['a']['ap'],0)
        s=self.next(s);s['sides']['a']['ap']=0
        self.assertNotIn({'type':'attack','character_id':'zero'},legal_actions(s,'a'))

    def test_debuff_stacks_and_survives_turn_move_and_equipment_clears_on_down(self):
        s=self.game();s=self.play(s,'V03','b:zero');s=self.play(s,'V03','b:zero')
        h=hero(s,'b','zero');self.assertEqual(compatibility_flags(h)['atk_debuff'],4)
        s['sides']['b']['front']='zero';move_out(s,'b','zero');h['shape']='Z08'
        s=self.next(s);h=hero(s,'b','zero')
        self.assertEqual(compatibility_flags(h)['atk_debuff'],4);self.assertEqual(attack_value(h,s,'b'),0)
        damage(s,'b','zero',99);self.assertNotIn('atk_debuff',h['flags'])

    def test_dark_star_count_retained_while_down_and_weapon_does_not_spend(self):
        s=self.game();hero(s,'a','haniya').update(shape='V06',protagonist_aura=4)
        s['sides']['a']['front']='anhunqu';hero(s,'a','anhunqu')['harmony']=2
        s=self.attack(s,'haniya')
        self.assertEqual(s['sides']['a']['dark_star_count'],1)
        self.assertEqual(s['sides']['b']['hp'],23)
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],4)
        hero(s,'a','haniya')['protagonist_aura']=0;s=self.play(s,'V04')
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],2)
        damage(s,'a','haniya',99);s=self.next(s)
        self.assertEqual(s['sides']['a']['dark_star_count'],1)

    def test_random_damage_conserves_points_includes_player_and_is_replayable(self):
        s=self.game();hero(s,'a','haniya')['protagonist_aura']=6
        for cid in s['sides']['b']['characters']: damage(s,'b',cid,99)
        original=json.loads(json.dumps(s))
        a=self.play(s,'V05');b=self.play(original,'V05')
        self.assertEqual(a['sides']['b']['hp'],18)
        self.assertEqual(a['rng'],b['rng'])
        self.assertEqual(hero(a,'a','haniya')['protagonist_aura'],0)
        self.assertEqual(a['sides']['a']['deck'],b['sides']['a']['deck'])
        s=self.game();hero(s,'a','haniya')['protagonist_aura']=6
        seq=s['event_seq'];s=self.play(s,'V05')
        hits=[e for e in s['events'] if e['seq']>seq and e.get('source')=='a:haniya' and 'before' in e and 'hp' in e['before'] and 'amount' in e]
        self.assertEqual(sum(e['amount'] for e in hits),12)
        self.assertGreater(len({e['target'] for e in hits}),1)

    def test_weapon_draw_only_actual_attack_player_damage_and_overflow(self):
        s=self.game();hero(s,'a','haniya').update(shape='V07',protagonist_aura=0)
        n=len(s['sides']['a']['deck']);s=self.attack(s,'zero')
        self.assertEqual(len(s['sides']['a']['deck']),n-1)
        s=self.game();hero(s,'a','haniya')['shape']='V07';s['sides']['b']['shield']=99
        n=len(s['sides']['a']['deck']);s=self.attack(s,'zero')
        self.assertEqual(len(s['sides']['a']['deck']),n)
        damage(s,'b','player',1,bypass=True,source='a:zero')
        self.assertEqual(len(s['sides']['a']['deck']),n)
        s=self.game(team=('haniya','bohe','zero','xiaozhi'));hero(s,'a','haniya')['shape']='V07'
        s['sides']['b']['front']='zero';hero(s,'b','zero')['hp']=1
        n=len(s['sides']['a']['deck']);s=self.attack(s,'bohe')
        self.assertEqual(len(s['sides']['a']['deck']),n-1)

    def test_response_battle_bonus_and_xiaozhi_override(self):
        for responder, card, expected in [('nanali', 'N02', 5), ('xiaozhi', 'Q04', 5)]:
            with self.subTest(card=card):
                s=self.game(team=('haniya','nanali','zero','xiaozhi'))
                hero(s,'b','haniya').update(shape='V08',protagonist_aura=3)
                s['sides']['b'].update(front='xiaozhi' if responder=='xiaozhi' else 'zero',ap=1)
                s['sides']['b']['hand']=[card_instance(s,card,'b')]
                s=self.attack(s,'zero')
                hit=next(e for e in reversed(s['events']) if e['type']=='attack')
                self.assertEqual(hit['counter'],expected)
                self.assertEqual(hero(s,'b','haniya')['protagonist_aura'],0)

    def test_auto_sortie_does_not_consume_and_dark_star_does_not_reuse_old_status(self):
        s=self.game(team=('haniya','bohe','zero','anhunqu'))
        hero(s,'a','haniya')['protagonist_aura']=3
        hero(s,'a','bohe')['shape']='M08'
        s=self.next(s)
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],4)
        s=self.game();hero(s,'a','haniya').update(shape='V06',protagonist_aura=3)
        s['sides']['b']['front_debuff']['star']={'left':2,'by':'a'}
        s=self.attack(s,'zero')
        self.assertEqual(hero(s,'a','haniya')['protagonist_aura'],0)

    def test_public_resource_survives_json_and_replay(self):
        s=self.game();hero(s,'a','haniya')['protagonist_aura']=5
        s=apply_action(json.loads(json.dumps(s)),'a',{'type':'end_turn'})
        for viewer in ('a','b'):
            h=next(h for h in observe(s,viewer)['sides']['a']['characters'] if h['id']=='haniya')
            self.assertEqual(h['protagonist_aura'],5)
        self.assertEqual(next(h for h in capture_public_board(s)['sides']['a']['characters'] if h['id']=='haniya')['protagonist_aura'],5)

if __name__=='__main__': unittest.main()
