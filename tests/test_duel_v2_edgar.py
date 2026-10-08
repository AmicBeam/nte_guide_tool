import json
import unittest
from copy import deepcopy

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS, get_catalog
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.entities import entity_action
from app.modules.card_game.engine.duel_v2.state import card_instance, hero, damage, heal, draw, front_debuff


class EdgarTest(unittest.TestCase):
    def deck(self):
        return dict(id='edgar-test',name='埃德嘉测试',character_ids=['edgar','zero','jiuyuan','hathor'],
                    card_ids=[p+str(i).zfill(2) for p in ('E','Z','J','H') for i in range(1,5) for _ in range(2)])

    def game(self, skip=True):
        d=self.deck();s=new_game(seed=71,first_side='a',skip_mulligan=skip,decks={'a':d,'b':d})
        if skip:
            for t in s['sides'].values():t['hand']=[];t['shield']=0
            s['sides']['a']['ap']=6
        return s

    def play(self,s,cid):
        card=card_instance(s,cid,'a');s['sides']['a']['hand'].append(card)
        return apply_action(s,'a',{'type':'play_card','card_id':card['instance_id']})

    def next(self,s):
        s=apply_action(s,'a',{'type':'end_turn'})
        return apply_action(s,'b',{'type':'end_turn'})

    def choose(self,s):
        a=next(a for a in legal_actions(s,'a') if a['type']=='choose')
        return apply_action(s,'a',entity_action(s,'a',a))

    def test_catalog_initial_hand_and_visibility(self):
        s=self.game(skip=False)
        self.assertEqual(len(s['sides']['a']['hand']),6)
        self.assertEqual(len(s['sides']['b']['hand']),6)
        self.assertEqual(hero(s,'a','edgar')['truth_keys'],0)
        self.assertEqual((CHARACTERS['edgar']['attack'],CHARACTERS['edgar']['max_hp']),(1,6))
        self.assertEqual(len([a for a in legal_actions(s,'a') if a['type']=='mulligan']),42)
        self.assertNotIn('edgar',{h['id'] for h in get_catalog(include_test_characters=False)['characters']})
        self.assertEqual(CARDS['E08']['name'],'扭曲之城的呼唤')

    def test_support_keys_self_and_team_clear_on_down(self):
        s=self.game();s['sides']['a']['front']='zero';hero(s,'a','zero')['harmony']=2
        s=apply_action(s,'a',{'type':'attack','character_id':'jiuyuan'})
        self.assertEqual(hero(s,'a','edgar')['truth_keys'],1)
        # Full harmony must exist when Jiuyuan leaves the front, before turn return.
        hero(s,'a','jiuyuan')['harmony']=2;s=self.next(s)
        s=apply_action(s,'a',{'type':'attack','character_id':'edgar'})
        self.assertEqual(hero(s,'a','edgar')['truth_keys'],2)
        damage(s,'a','edgar',99)
        self.assertEqual(hero(s,'a','edgar')['truth_keys'],0)

    def test_ultimate_front_and_player_extends_then_ends(self):
        s=self.game();h=hero(s,'a','edgar');h.update(energy=5,truth_keys=2)
        s['sides']['a']['front']='zero';hero(s,'a','zero')['hp']=1
        s=apply_action(s,'a',{'type':'ultimate','character_id':'edgar'})
        self.assertEqual(hero(s,'a','zero')['hp'],3)
        self.assertEqual(hero(s,'a','edgar')['ultimate_turns'],3)
        self.assertEqual(hero(s,'a','edgar')['truth_keys'],0)
        s['sides']['a']['front']=None;s['sides']['a']['hp']=20
        for expected,left in [(22,2),(24,1),(24,0)]:
            s=self.next(s)
            self.assertEqual(s['sides']['a']['hp'],expected)
            self.assertEqual(hero(s,'a','edgar')['ultimate_turns'],left)
        s=self.next(s);self.assertEqual(s['sides']['a']['hp'],24)

    def test_ultimate_down_cancels_and_new_keys_dont_extend_current(self):
        s=self.game();hero(s,'a','edgar').update(energy=5,truth_keys=1)
        s=apply_action(s,'a',{'type':'ultimate','character_id':'edgar'})
        s=self.play(s,'E06');self.assertEqual(hero(s,'a','edgar')['ultimate_turns'],2)
        damage(s,'a','edgar',99);s['sides']['a']['hp']=20
        s=self.next(s);self.assertEqual(s['sides']['a']['hp'],20)

    def test_surplus_once_per_game_and_deck_out_win_after_down(self):
        s=self.game()
        with self.assertRaises(ValueError):self.play(s,'E01')
        s['sides']['a']['front']='zero';hero(s,'a','zero')['harmony']=2
        s['sides']['b']['front']='zero'
        front_debuff(s,'b')['delay']={'by':'a','left':1}
        s=apply_action(s,'a',{'type':'attack','character_id':'jiuyuan'})
        self.assertTrue(s['sides']['a']['surplus_ever'])
        s=self.next(s);self.assertFalse(s['sides']['a']['surplus'])
        s=self.play(s,'E01');damage(s,'a','edgar',99)
        s['sides']['a']['deck']=[];draw(s,'a')
        self.assertEqual((s['phase'],s['winner'],s['reason']),('finished','a','empty_draw_win'))

    def test_two_draws_empty_deck_normally_loses(self):
        s=self.game();before=len(s['sides']['a']['deck']);s=self.play(s,'E02')
        self.assertEqual(len(s['sides']['a']['deck']),before-2)
        s['sides']['a']['deck']=s['sides']['a']['deck'][:1]
        s=self.play(s,'E02');self.assertEqual(s['winner'],'b')

    def test_hand_redraw_six_cards_private_and_excludes_spell(self):
        s=self.game();t=s['sides']['a'];t['hand']=[card_instance(s,'Z03','a') for _ in range(8)]
        s=self.play(s,'E03');p=s['pending_choice'];original=deepcopy(s)
        self.assertEqual(len(p['cards']),6)
        self.assertFalse(any(c['card_id']=='E03' for c in p['cards']))
        self.assertEqual(observe(s,'b')['pending_choice']['choices'],[])
        ids=[c['instance_id'] for c in p['cards'][:3]]
        expected=[c['entity_id'] for c in t['deck'][:3]]
        action=entity_action(s,'a',{'type':'choose_cards','card_ids':list(reversed(ids))})
        s=apply_action(json.loads(json.dumps(s)),'a',action)
        self.assertEqual(s['phase'],'playing');self.assertEqual(len(s['sides']['a']['hand']),8)
        hand_ids={c['instance_id'] for c in s['sides']['a']['hand']}
        self.assertFalse(set(ids)&hand_ids)
        self.assertTrue(set(expected)<={c['entity_id'] for c in s['sides']['a']['hand']})
        self.assertEqual(s['sides']['a']['ap'],original['sides']['a']['ap'])
        self.assertTrue(s['sides']['a']['used']['instant'])
        self.assertEqual(original['phase'],'choice')

    def test_redraw_no_other_hand_rejected_and_zero_valid(self):
        s=self.game()
        with self.assertRaises(ValueError):self.play(s,'E03')
        s['sides']['a']['hand']=[card_instance(s,'Z03','a')];s=self.play(s,'E03')
        original=deepcopy(s['sides']['a']['deck'])
        s=apply_action(s,'a',{'type':'choose_cards','card_ids':[]})
        self.assertEqual(s['phase'],'playing')
        # Zero replacements should not change the deck order or RNG.
        self.assertEqual(s['sides']['a']['deck'],original)

    def test_energy_counts_other_lost_hp_including_down(self):
        s=self.game();hero(s,'a','zero')['hp']=3;hero(s,'a','jiuyuan')['hp']=3
        hero(s,'a','edgar')['hp']=1
        s=self.play(s,'E04');self.assertEqual(hero(s,'a','edgar')['energy'],3)
        s=self.next(s);damage(s,'a','hathor',99)
        s=self.play(s,'E04');self.assertEqual(hero(s,'a','edgar')['energy'],5)

    def test_random_owned_cards_gain_instance_instant(self):
        s=self.game();s=self.play(s,'E05')
        cards=s['sides']['a']['hand']
        self.assertEqual({c['character_id'] for c in cards},{'zero','jiuyuan','hathor'})
        self.assertTrue(all(c['instant'] and not c.get('copy') for c in cards))
        self.assertTrue(all(c['card_id'] in CARDS and not c.get('derived') for c in cards))

    def test_battle_heals_lowest_role_before_sortie_and_grants_key(self):
        s=self.game();s['sides']['a']['hp']=1
        s=self.play(s,'E06')
        self.assertEqual(s['sides']['a']['hp'],6)
        self.assertEqual(s['sides']['a']['front'],'edgar')
        self.assertEqual(hero(s,'a','edgar')['truth_keys'],1)
        self.assertEqual(s['sides']['b']['hp'],29)

    def test_heal_bonus_only_own_heals_draw_any_allied_actual_heal_once(self):
        s=self.game();s=self.play(s,'E08');t=s['sides']['a'];t['hp']=20
        hero(s,'a','edgar')['energy']=5;before=len(t['deck'])
        s=apply_action(s,'a',{'type':'ultimate','character_id':'edgar'})
        self.assertEqual(s['sides']['a']['hp'],23)
        self.assertEqual(len(s['sides']['a']['deck']),before-1)
        heal(s,'a','zero',2);self.assertEqual(len(s['sides']['a']['deck']),before-1)
        s=self.play(s,'E08');hero(s,'a','zero')['hp']=2
        heal(s,'a','zero',1);self.assertEqual(hero(s,'a','zero')['hp'],3)
        self.assertEqual(len(s['sides']['a']['deck']),before-1)
        s=apply_action(s,'a',{'type':'end_turn'});before=len(s['sides']['a']['deck'])
        heal(s,'a','zero',1);self.assertEqual(len(s['sides']['a']['deck']),before-1)

    def test_turn_draw_inspection_continues_turn_and_only_replaces_normal_draw(self):
        s=self.game();s=self.play(s,'E07');s=apply_action(s,'a',{'type':'end_turn'})
        before=deepcopy(s['sides']['a']['deck'])
        s=apply_action(s,'b',{'type':'end_turn'})
        self.assertEqual(s['phase'],'choice');self.assertEqual(s['pending_choice']['kind'],'inspect_top')
        self.assertEqual(len(s['pending_choice']['cards']),3)
        from app.modules.card_game.engine.duel_v2.replay_log import public_replay_view
        self.assertEqual(public_replay_view(s,'a')['pending_choice']['choices'],[])
        self.assertEqual(len(observe(s,'a')['pending_choice']['choices']),3)
        self.assertEqual(observe(s,'a')['sides']['a']['deck_count'],len(before))
        chosen=s['pending_choice']['cards'][1]
        s=apply_action(s,'a',{'type':'choose','choice_id':chosen['instance_id']})
        self.assertEqual(s['phase'],'playing');self.assertNotIn('turn_start_pending',s)
        self.assertEqual(s['sides']['a']['ap'],2)
        self.assertEqual([c['instance_id'] for c in s['sides']['a']['deck'][-2:]],
                         [before[0]['instance_id'],before[2]['instance_id']])
        s=self.play(s,'E02');self.assertEqual(s['phase'],'playing')

    def test_empty_turn_draw_with_inspection_still_resolves_deckout(self):
        for wins in (False,True):
            s=self.game();s=self.play(s,'E07');s['sides']['a'].update(deck=[],empty_draw_wins=wins)
            s=self.next(s);self.assertEqual(s['winner'],'a' if wins else 'b')

    def test_one_card_inspection_and_xun_checkpoint_after_choice(self):
        from app.modules.card_game.engine.duel_v2.state import new_character
        s=self.game();s['sides']['a']['characters']['xun']=new_character('xun');s['sides']['a']['order'].append('xun')
        s=self.play(s,'E07');s=self.next(s)
        self.assertEqual(s['phase'],'choice');s=self.choose(s)
        self.assertEqual(len(s['turn_snapshots']['a']),1)
        self.assertEqual(s['turn_snapshots']['a'][0]['phase'],'playing')
        s['sides']['a']['deck']=s['sides']['a']['deck'][:1]
        s=self.next(s);self.assertEqual(s['phase'],'playing')
        self.assertEqual(len(s['sides']['a']['deck']),0)

if __name__=='__main__':unittest.main()
