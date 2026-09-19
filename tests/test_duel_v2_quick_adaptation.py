import unittest,tempfile,json
from pathlib import Path
from app.modules.card_game.rl.quick_adaptation import contribution_events,select_candidate,migrate_numeric
from tests.test_duel_v2_league import zero_policy
from app.modules.card_game.rl.league_policy import LeagueModel
class AdaptationTest(unittest.TestCase):
 def test_actual_loss_and_sources(self):
  e=[dict(actor='a:xiaozhi',target='b:player',type='combat',amount=20,before=dict(hp=3,shield=2),after=dict(hp=0,shield=0)),dict(actor='a:zero',target='a:zero',type='damage',before=dict(hp=5),after=dict(hp=3)),dict(actor='a:bohe',target='a:zero',type='heal',before=dict(hp=2),after=dict(hp=4))]
  r=contribution_events(e,['xiaozhi','zero','bohe','haiyue']);self.assertEqual(r['xiaozhi'],dict(player_hp_damage=3,enemy_shield_absorbed=2));self.assertEqual(r['zero'],{});self.assertEqual(r['bohe']['healing'],2)
 def test_selection_guard(self):
  self.assertTrue(select_candidate(dict(complete=True,candidate=10,best=8),True));self.assertFalse(select_candidate(dict(complete=False,candidate=10,best=8),True));self.assertFalse(select_candidate(dict(complete=True,candidate=10,best=8),False))
 def test_migration_contract(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'source';src.mkdir()
   for key in ('starter','weave-rush','quick-rush'):
    zero_policy(src,key);p=src/f'{key}.json';m=json.loads(p.read_text());m['rule_hash']='prior-rule';m['validation']={'approved':True};p.write_text(json.dumps(m))
   migrate_numeric(src,Path(d)/'new');m=LeagueModel(Path(d)/'new','quick-rush');self.assertFalse(m.approved);self.assertFalse(m.user_authorized);self.assertEqual(m.manifest['origin']['source_rule_hash'],'prior-rule');self.assertTrue(all((w==0).all() for w in m.weights.values()))
   p=src/'quick-rush.json';j=json.loads(p.read_text());j['features']=[];p.write_text(json.dumps(j))
   with self.assertRaises(ValueError):migrate_numeric(src,Path(d)/'reject')

 def test_probes_use_explicit_empty_opponent_hand(self):
  import time
  from app.modules.card_game.rl.quick_adaptation import adaptation_probes
  with tempfile.TemporaryDirectory() as d:
   probes=adaptation_probes(d,time.time()+20)['quick-rush'];self.assertEqual(len(probes),12)
   for p in probes:
    foe='b' if p['side']=='a' else 'a';self.assertEqual(p['state']['sides'][foe]['hand'],[]);self.assertTrue(p['targets'])
