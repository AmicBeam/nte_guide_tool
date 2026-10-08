import hashlib,json,tempfile,time,unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from app.modules.card_game.rl.league_schema import *
from app.modules.card_game.rl.league_observation import encode_league
from app.modules.card_game.rl.league_rollout import determinize,decision,play,clean
from app.modules.card_game.rl.league_policy import LeagueModel,shapes
from app.modules.card_game.engine.duel_v2 import new_game,observe,legal_actions,apply_action
from app.modules.card_game.engine.duel_v2.state import card_instance,hero


def zero_policy(root,key):
    path=Path(root)/(key+'.npz');np.savez(path,**{n:np.zeros(s,dtype=np.float32) for n,s in shapes(8).items()})
    build=deck(key);m={'schema':SCHEMA,'deck':key,'rule_hash':rule_hash(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seat_ids':list(SEATS),'card_ids':list(CARD_IDS),'features':feature_names(),'cand_dim':CAND_DIM,'hidden':8,'build':build,'build_sha256':build_hash(build),'validation':{'approved':False}}
    path.with_suffix('.json').write_text(json.dumps(m),encoding='utf-8')

class LeagueTest(unittest.TestCase):
    def game(self):return new_game(seed=921000000,skip_mulligan=True,first_side='a',decks={'a':deck('quick-rush'),'b':deck('starter')})
    def test_every_public_holdout_team_has_a_valid_deck(self):
        from itertools import combinations
        from app.modules.card_game.rl.league_experiment import sample_deck
        from app.modules.card_game.content.duel_v2 import validate_deck
        seen=set()
        for i,team in enumerate(combinations(SEATS,4)):
            build=sample_deck(945090000+i,team)
            self.assertEqual(validate_deck(build),build)
            self.assertTrue(build['id']);seen.add(tuple(build['character_ids']))
        self.assertEqual(len(seen),70)

    def test_all_presets_and_fast_projection_match(self):
        self.assertEqual(len(feature_names(old=True)),666)
        for k in PRESETS:
            s=new_game(seed=921000001,decks={'a':deck(k),'b':deck('quick-rush')})
            actions=[a for a in legal_actions(s,'a') if a['type']!='concede']
            x,c=encode_league(observe(s,'a'),actions);xx,cc=encode_league(observe(s,'a',include_previews=False),actions)
            np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)
            self.assertEqual(x.shape,(len(feature_names()),));self.assertEqual(c.shape[1],CAND_DIM)
    def test_options_and_private_marks_are_distinct(self):
        s=self.game();hero(s,'a','xiaozhi')['jingu']=6;s['sides']['a']['ap']=2
        hand=[card_instance(s,'Q03','a'),card_instance(s,'Q03','a')]
        for c,d in zip(hand,(-1,1)):c['jingu_mark']={'delta':d,'source_entity_id':hero(s,'a','xiaozhi')['entity_id']}
        s['sides']['a']['hand']=hand
        a,(x,c)=decision(s,'a');rows=[c[i] for i,act in enumerate(a) if act['type']=='play_card']
        self.assertEqual(len(rows),6);self.assertEqual(len(set(tuple(r) for r in rows)),6)
    def test_hidden_cards_order_ids_rng_do_not_change_samples(self):
        s=self.game();altered=deepcopy(s);foe=altered['sides']['b'];foe['deck'].reverse()
        for c in foe['deck']+foe['hand']:
            if not c.get('derived'):
                c['card_id']='N04' if c['card_id']!='N04' else 'N05'
            c['instance_id']='b-9999';c['entity_id']='e9999'
        altered['sides']['a']['deck'].reverse();altered['rng']=123456789;altered['next_entity']=99999
        a=determinize(s,'a',38);b=determinize(altered,'a',38)
        for side in ('a','b'):
            for zone in ('deck','hand'):
                self.assertEqual([(c['card_id'],c['instance_id'],c['entity_id']) for c in a['sides'][side][zone]],[(c['card_id'],c['instance_id'],c['entity_id']) for c in b['sides'][side][zone]])
        self.assertEqual(a['rng'],b['rng'])
    def test_q05_generated_pool_accounting(self):
        s=self.game();s['sides']['a']['hand']=[card_instance(s,'Q05','a')]+s['sides']['a']['hand']
        # Preserve a legal original pool by replacing an original Q05 instance from deck.
        original=next(c for c in s['sides']['a']['deck'] if c['card_id']=='Q05');s['sides']['a']['deck'].remove(original)
        action=next(a for a in legal_actions(s,'a') if a.get('card_id')==s['sides']['a']['hand'][0]['instance_id'])
        s=apply_action(s,'a',action);s=apply_action(s,'a',{'type':'end_turn'})
        sampled=determinize(s,'b',41)
        self.assertEqual(sum(len(sampled['sides']['a'][z]) for z in ('hand','deck','discard')),33)
    def test_numpy_model_guard_and_terminal_branches(self):
        with tempfile.TemporaryDirectory() as d:
            for key in ('starter','quick-rush'):zero_policy(d,key)
            with self.assertRaises(ValueError):LeagueModel(d,'quick-rush',require_approved=True)
            from unittest.mock import patch
            from app.errors import RuleValidationError
            from app.modules.card_game.engine.ai.advanced_model import load_model
            with patch('app.config.DUEL_AI_MODEL_DIR', d), self.assertRaises(RuleValidationError):load_model('quick-rush')
            s=self.game();s['sides']['a']['hand']=[card_instance(s,'Q01','a')];s['sides']['b']['hp']=7;hero(s,'a','xiaozhi')['jingu']=3
            job={'root':s,'viewer':'a','deadline':time.time()+30,'seed':19,'policies':{'a':(d,'quick-rush'),'b':(d,'starter')}}
            q=[]
            for opt in ('base','boost'):
                g=play({**job,'action':{'type':'play_card','card_id':s['sides']['a']['hand'][0]['instance_id'],'option_id':opt}})
                self.assertTrue(g['complete']);q.append(g['reward'])
            self.assertEqual(q,[-10,10])
            result=play({**job,'max_actions':0,'action':{'type':'play_card','card_id':s['sides']['a']['hand'][0]['instance_id'],'option_id':'base'}})
            self.assertFalse(result['complete']);self.assertIsNone(result['reward'])
    def test_explicit_candidate_authorization_keeps_quality_gate_false(self):
        with tempfile.TemporaryDirectory() as d:
            zero_policy(d,'quick-rush');p=Path(d)/'quick-rush.json'
            m=json.loads(p.read_text())
            authorization={'kind':'explicit_user_candidate','request':'Enable candidate',
                           'report':'test-report.md','model_sha256':m['sha256'],
                           'build_sha256':m['build_sha256'],'rule_hash':m['rule_hash']}
            m['serving_authorization']=authorization
            p.write_text(json.dumps(m))
            model=LeagueModel(d,'quick-rush',require_approved=True)
            self.assertFalse(model.approved)
            self.assertTrue(model.user_authorized)
            self.assertTrue(model.validate_human(deck('starter'))['human_public_custom'])
            for field in ('kind','request','report','model_sha256','build_sha256','rule_hash'):
                with self.subTest(field=field):
                    m['serving_authorization']={**authorization,field:''}
                    p.write_text(json.dumps(m))
                    with self.assertRaises(ValueError):LeagueModel(d,'quick-rush',require_approved=True)

    def test_serving_lineup_is_independent_of_current_gift_preset(self):
        with tempfile.TemporaryDirectory() as d:
            zero_policy(d, 'quick-rush')
            p = Path(d) / 'quick-rush.json'
            m = json.loads(p.read_text())
            m['build']['character_ids'][-1] = 'haiyue'
            m['build']['card_ids'] = [c for c in m['build']['card_ids'] if not c.startswith('J')] + ['U01', 'U02', 'U03', 'U03', 'U04', 'U06', 'U06', 'U08']
            m['build_sha256'] = build_hash(m['build'])
            p.write_text(json.dumps(m))
            model = LeagueModel(d, 'quick-rush')
            self.assertEqual(model.serving_deck, m['build'])
            self.assertIn('haiyue', model.serving_deck['character_ids'])
            self.assertIn('jiuyuan', deck('quick-rush')['character_ids'])
            m['build']['card_ids'][0] = 'Z01'
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Build identity mismatch'):
                LeagueModel(d, 'quick-rush')

    def test_explicit_runtime_compatibility_preserves_training_identity(self):
        with tempfile.TemporaryDirectory() as d:
            zero_policy(d, 'starter')
            p = Path(d) / 'starter.json'
            m = json.loads(p.read_text())
            runtime = m['rule_hash']
            m['rule_hash'] = 'historical-training-rules'
            m['serving_authorization'] = {
                'kind': 'explicit_user_candidate', 'request': 'Enable candidate',
                'report': 'test-report.md', 'model_sha256': m['sha256'],
                'build_sha256': m['build_sha256'], 'rule_hash': m['rule_hash']}
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'rule/schema mismatch'):
                LeagueModel(d, 'starter', require_approved=True)
            compatibility = {
                'kind': 'explicit_user_compatibility', 'request': 'Accept minor rule changes',
                'source_rule_hash': m['rule_hash'], 'runtime_rule_hash': runtime,
                'model_sha256': m['sha256'], 'build_sha256': m['build_sha256']}
            m['runtime_compatibility'] = compatibility
            p.write_text(json.dumps(m))
            model = LeagueModel(d, 'starter', require_approved=True)
            self.assertTrue(model.runtime_compatible)
            self.assertFalse(model.approved)
            self.assertEqual(model.manifest['rule_hash'], 'historical-training-rules')
            for field in compatibility:
                with self.subTest(field=field):
                    m['runtime_compatibility'] = {**compatibility, field: ''}
                    p.write_text(json.dumps(m))
                    with self.assertRaises(ValueError):
                        LeagueModel(d, 'starter', require_approved=True)

    def test_branch_gradient_and_numeric_export(self):
        try:import torch
        except ImportError:self.skipTest('Torch tested on Windows')
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.league_learning import branch_update
        from app.modules.card_game.rl.league_policy import export_model
        torch.manual_seed(3);net=CompactScorer(len(feature_names()),CAND_DIM,8);opt=torch.optim.Adam(net.parameters(),lr=.01)
        s=self.game();actions,(x,c)=decision(s,'a');c=c[:2]
        xx=torch.tensor(x)[None];cc=torch.tensor(c)[None];mask=torch.ones(1,2,dtype=torch.bool)
        before=net(xx,cc,mask)[0].softmax(1)[0,1].item()
        for _ in range(10):self.assertEqual(branch_update(net,opt,x,c,[-10,10])['updates'],1)
        after=net(xx,cc,mask)[0].softmax(1)[0,1].item();self.assertGreater(after,before)
        with tempfile.TemporaryDirectory() as d:
            export_model(net,d,'quick-rush',deck('quick-rush'));m=LeagueModel(d,'quick-rush')
            np.testing.assert_allclose(m.scores(x,c),net(xx,cc,mask)[0].detach().numpy()[0],rtol=1e-5,atol=1e-5)
