"""Current authored card changes, including resource and entry ordering."""
import unittest

from app.modules.card_game.content.duel_v2 import CARDS, CHARACTERS
from app.modules.card_game.content.duel_v2.registry import fire
from app.modules.card_game.engine.events import GameEvent
from app.modules.card_game.engine.duel_v2 import new_game, apply_action, legal_actions, observe
from app.modules.card_game.engine.duel_v2.flow import operation, finish_operation
from app.modules.card_game.engine.duel_v2.genesis import hit
from app.modules.card_game.engine.duel_v2.state import (hero, card_instance, damage, collapse_max,
    collapse_border_cut, add_collapse, collapsed, front_debuff, attack_value)
from app.modules.card_game.engine.duel_v2.presentation import capture_public_board


def game(ids=('nanali', 'zero', 'jiuyuan', 'iloy')):
    deck = dict(id='current', name='current', character_ids=list(ids), card_ids=[
        key for cid in ids for key, card in CARDS.items()
        if card['character_id'] == cid and not card.get('derived')])
    s = new_game(seed=29, decks={'a':deck,'b':deck}, first_side='a', skip_mulligan=True, escalation=False)
    for team in s['sides'].values():
        team.update(hand=[], ap=20, hp=100, shield=0)
        for h in team['characters'].values(): h.update(hp=30,max_hp=30)
    return s


def play(s, key, target=None):
    card = card_instance(s,key,'a');s['sides']['a']['hand'].append(card)
    action = dict(type='play_card',card_id=card['instance_id'])
    if target: action['target_id']=target
    return apply_action(s,'a',action)


class LatestCardAdjustmentsTest(unittest.TestCase):
    def test_nanali_tracks_actual_genesis_overlay_including_shields_not_overkill(self):
        s=game();target=hero(s,'b','zero');target.update(shield=2,hp=3)
        damage(s,'b','zero',10,source='a:iloy',kind='genesis')
        self.assertEqual(s['sides']['a']['genesis_weave_damage'],5)
        damage(s,'b','player',5,source='a:zero',kind='overlay')
        damage(s,'b','player',5,source='a:zero',kind='flower')
        self.assertEqual(s['sides']['a']['genesis_weave_damage'],10)
        s=play(s,'N04')
        attack=next(e for e in reversed(s['events']) if e['type']=='attack')
        self.assertEqual(attack['attack'],CHARACTERS['nanali']['attack']+3)
        self.assertEqual(hero(s,'a','nanali')['shield'],0)
        self.assertFalse(s['sides']['a'].get('genesis_damage_bonus'))
        card=card_instance(s,'N04','a');s['sides']['a']['hand'].append(card)
        view=observe(s,'a',include_previews=False)
        self.assertTrue(view['sides']['a']['hand'][-1]['description'].endswith('（当前为1）'))

    def test_zero_heals_and_draws(self):
        s=game();s['sides']['a'].update(front='nanali',surplus=True);hero(s,'a','nanali')['hp']=1
        count=len(s['sides']['a']['deck']);s=play(s,'Z01')
        self.assertEqual(hero(s,'a','nanali')['hp'],30)
        self.assertEqual(len(s['sides']['a']['deck']),count-1)

    def test_jiuyuan_liquidates_marked_only_once_and_recovers_energy(self):
        s=game();j=hero(s,'a','jiuyuan');j.update(awakened=True,energy=0)
        for cid in ('zero','iloy'):hero(s,'b',cid)['flags']['pact']=True
        hit(operation(s,'a','nanali'))
        self.assertEqual(hero(s,'b','zero')['hp'],28)
        self.assertEqual(hero(s,'b','iloy')['hp'],28)
        self.assertEqual(hero(s,'b','nanali')['hp'],30)
        self.assertEqual(s['sides']['b']['hp'],99)
        self.assertEqual(j['energy'],4)
        hero(s,'b','zero')['flags']['pact']=True
        hit(operation(s,'a','nanali'))
        self.assertEqual(hero(s,'b','zero')['hp'],28)
        self.assertTrue(hero(s,'b','zero')['flags']['pact'])
        s['turn']+=2;hit(operation(s,'a','nanali'))
        self.assertEqual(hero(s,'b','zero')['hp'],26)

    def test_baicang_paid_harmony_before_entry_no_panel_attack_or_quota(self):
        s=game(('baicang','nanali','zero','iloy'));b=hero(s,'a','baicang');b.update(hp=27,harmony=2)
        s['sides']['a']['front']='baicang'
        s=play(s,'B03','a:nanali')
        self.assertEqual(s['sides']['a']['front'],'nanali')
        self.assertEqual(hero(s,'a','baicang')['harmony'],0)
        self.assertEqual(s['sides']['a']['support_count'],1)
        self.assertIn('weave',front_debuff(s,'b'))
        self.assertEqual(s['sides']['b']['hp'],97)
        self.assertFalse(any(e['type']=='attack' for e in s['events']))
        self.assertTrue(s['sides']['a']['normal_attack_available'])
        card=card_instance(s,'B03','a');s['sides']['a']['hand'].append(card)
        self.assertFalse(any(a.get('card_id')==card['instance_id'] for a in legal_actions(s,'a')))

    def test_q06_auto_releases_one_real_card_free_from_either_zone(self):
        for zone in ('hand','deck'):
            s=game(('xiaozhi','zero','nanali','iloy'));team=s['sides']['a'];x=hero(s,'a','xiaozhi')
            first=card_instance(s,'Q06','a');second=card_instance(s,'Q06','a');team[zone]=[first,second];team['ap']=0
            ctx=operation(s,'a','zero');parent=s['operation'];fire(ctx,GameEvent.V2_SURPLUS)
            self.assertIs(s['operation'],parent)
            self.assertEqual(x['energy'],x['energy_max'])
            self.assertEqual(team['ap'],0);self.assertEqual(x['harmony'],0)
            self.assertEqual(len(team[zone]),1)
            self.assertIn(first['instance_id'],[c['instance_id'] for c in team['discard']])
            events=[e for e in s['events'] if e['type']=='play' and e.get('card',{}).get('card_id')=='Q06']
            self.assertEqual(len(events),1);self.assertTrue(events[0]['release_only'])

    def test_zhenhong_initial_and_once_per_turn_solitary_grants_shield(self):
        s=game(('zhenhong','zero','nanali','iloy'));z=hero(s,'a','zhenhong');self.assertEqual(z['harmony'],2)
        z['hp']=7
        for _ in range(2):fire(operation(s,'a','zero'),GameEvent.V2_SURPLUS);finish_operation(s)
        self.assertEqual(z['hp'],7);self.assertEqual(z['shield'],4)
        self.assertEqual(z['surplus_passive_triggers'],1)
        self.assertEqual(s['sides']['b']['hp'],98)
        s['turn']+=2;fire(operation(s,'a','zero'),GameEvent.V2_SURPLUS);finish_operation(s)
        self.assertEqual(z['shield'],8);self.assertEqual(s['sides']['b']['hp'],96)

    def test_redraw_gets_one_more_and_training_has_no_frame_changes(self):
        s=game(('zhenhong','zero','nanali','iloy'));team=s['sides']['a']
        team['hand']=[card_instance(s,'R01','a'),card_instance(s,'R03','a')]
        s=play(s,'R05');self.assertEqual(len(s['sides']['a']['hand']),3)
        s=game(('zhenhong','zero','nanali','iloy'));s=play(s,'R02')
        attack=next(e for e in reversed(s['events']) if e['type']=='attack')
        self.assertEqual(attack['attack'],CHARACTERS['zhenhong']['attack'])
        self.assertEqual(hero(s,'a','zhenhong')['shield'],0)

    def test_collapse_three_frame_progress_and_nonstacking_reduction(self):
        s=game(('dafutier','zero','nanali','iloy'));t=hero(s,'b','zero')
        self.assertEqual(collapse_max(t),3);self.assertEqual(collapse_border_cut(t),0)
        add_collapse(s,'b','zero',1,by='a')
        self.assertAlmostEqual(collapse_border_cut(t),100/3)
        for board in (observe(s,'a',include_previews=False),capture_public_board(s)):
            row=next(h for h in board['sides']['b']['characters'] if h['id']=='zero')
            self.assertAlmostEqual(row['collapse_border_cut'],100/3)
        ctx=operation(s,'a','dafutier')
        for _ in range(2):fire(ctx,GameEvent.V2_DISSONANCE,enemy='b',target_id='zero')
        self.assertEqual(collapse_max(t),2)
        add_collapse(s,'b','zero',1,by='a')
        self.assertTrue(collapsed(t));self.assertEqual(collapse_border_cut(t),100)

    def test_anhunqu_eliminates_all_collapsed_only(self):
        s=game(('anhunqu','zero','nanali','iloy'))
        for cid in ('zero','iloy'):hero(s,'b',cid)['flags']['collapse']={'side':'a','until':3}
        fire(operation(s,'a','anhunqu'),GameEvent.V2_DISSONANCE,enemy='b',target_id='nanali')
        self.assertTrue(hero(s,'b','zero')['down_turns']);self.assertTrue(hero(s,'b','iloy')['down_turns'])
        self.assertEqual(hero(s,'b','nanali')['hp'],30)

    def test_haiyue_remote_while_star_exists_and_u04_gets_ultimate_bonus(self):
        s=game(('haiyue','zero','nanali','iloy'));hero(s,'a','haiyue')['awakened']=True
        front_debuff(s,'b')['star']={'left':1,'by':'a','source_cid':'zero'}
        s['sides']['a']['front']='zero'
        s=play(s,'U04','b:nanali')
        self.assertEqual(hero(s,'b','nanali')['hp'],23)
        self.assertEqual(s['sides']['a']['front'],'zero')
        self.assertEqual(hero(s,'a','haiyue')['hp'],30)

    def test_adler_constant_base_and_haniya_star_gain(self):
        s=game(('adler','haniya','canhong','zero'));a=hero(s,'a','adler');a['shield']=9
        self.assertEqual(attack_value(a,s,'a'),1)
        self.assertEqual(hero(s,'a','canhong')['harmony'],3)
        h=hero(s,'a','haniya');before=h['protagonist_aura']
        damage(s,'b','player',1,source='a:zero',kind='star')
        self.assertEqual(h['protagonist_aura'],before+1)
