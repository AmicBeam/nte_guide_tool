"""Synthetic candidate comparisons validate selection and lifecycle, not strength."""
import contextlib
import importlib.util
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.modules.card_game.rl.candidate_selection import metrics,run_tournament,select_with_baseline,PROFILES
from app.modules.card_game.rl.capability import preset_opponent_deck


def groups(pairs, wins):
    return {k:{p:{'wins':wins,'losses':pairs-wins,'games':pairs,'truncated':0} for p in ('first','second')}
            for k in ('starter','weave-rush')}


def schedule(pairs, **kwargs):
    return [{'lineup':k,'pair':i,'seed':0,'opponent_model':'peer-0','deck':preset_opponent_deck(k)}
            for k in ('starter','weave-rush') for i in range(pairs)]


class CandidateSelectionTest(unittest.TestCase):
    def test_higher_overall_score_can_trade_off_one_matchup_or_seat(self):
        baseline={'id':'original','complete':True,'groups':groups(100,50)}
        bad={'id':'bad','complete':True,'groups':groups(100,100)}
        bad['groups']['weave-rush']['first']['wins']=30
        for r in (baseline,bad):r['metrics']=metrics(r['groups'])
        self.assertGreater(bad['metrics']['macro'],baseline['metrics']['macro'])
        self.assertEqual(select_with_baseline([baseline,bad])['id'],'bad')
        self.assertAlmostEqual(bad['metrics']['utility'], .825)
        bad['groups']['weave-rush']['first']['wins']=60;bad['metrics']=metrics(bad['groups'])
        self.assertEqual(select_with_baseline([baseline,bad])['id'],'bad')
        bad['groups']=deepcopy(baseline['groups']);bad['metrics']=metrics(bad['groups'])
        self.assertEqual(select_with_baseline([baseline,bad])['id'],'original')

    def test_same_average_keeps_baseline_despite_different_worst_seat(self):
        baseline={'id':'original','complete':True,'groups':groups(100,50)}
        candidate={'id':'new','complete':True,'groups':groups(100,50)}
        candidate['groups']['starter']['first']['wins']=80
        candidate['groups']['weave-rush']['second']['wins']=20
        # Historical scores must not override the newly selected scoring rule.
        baseline['metrics']={'utility':0}
        candidate['metrics']={'utility':1}
        self.assertEqual(select_with_baseline([baseline,candidate])['id'],'original')
        self.assertEqual(candidate['metrics']['utility'],.5)

    def test_previous_local_candidate_is_included_and_deduplicated(self):
        from app.modules.card_game.rl.candidate_selection import candidate_pool,allocation_key
        deck=preset_opponent_deck('starter')
        candidate=deepcopy(deck)
        candidate['card_ids']=['N01' if c=='N02' else c for c in candidate['card_ids']]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'original-deck.json').write_text(json.dumps(deck))
            (root/'seed-candidates.json').write_text(json.dumps([candidate,deck]))
            rows=candidate_pool(root,'starter')
            self.assertIn(allocation_key(candidate),[allocation_key(r['build']) for r in rows])
            self.assertEqual(len(rows),len({allocation_key(r['build']) for r in rows}))

    def test_stage_budget_and_seeds_and_incomplete_fallback(self):
        deck=preset_opponent_deck('starter')
        candidates=[{'id':'original','build':deck},{'id':'new','build':{**deck,'name':'candidate'}}]
        seen=[]
        def evaluate(builds, plan):
            seen.append(plan);pairs=len(plan)//2
            return [{'complete':True,'groups':groups(pairs,pairs//2 if b['name']==deck['name'] else pairs)} for b in builds]
        result=run_tournament(candidates,evaluate,profile='comprehensive',schedule_factory=schedule)
        self.assertTrue(result['complete']);self.assertEqual(result['selection'],'new')
        self.assertEqual([len(p) for p in seen],[16,64,256])
        seeds=[{p['seed']+seat for p in plan for seat in (0,1)} for plan in seen]
        self.assertFalse(seeds[0]&seeds[1] or seeds[0]&seeds[2] or seeds[1]&seeds[2])
        self.assertTrue(all(202609160 not in s for s in seeds))
        bad=lambda builds,plan:[{'complete':False,'groups':{}} for _ in builds]
        result=run_tournament(candidates,bad,schedule_factory=schedule)
        self.assertFalse(result['complete']);self.assertEqual(result['selection'],'original')
        def timeout():raise TimeoutError()
        result=run_tournament(candidates,evaluate,stop_check=timeout,schedule_factory=schedule)
        self.assertFalse(result['complete']);self.assertEqual(result['selection'],'original')

    def test_false_completed_flag_cannot_hide_missing_samples(self):
        candidates=[{'id':'original','build':preset_opponent_deck('starter')}]
        result=run_tournament(candidates,lambda decks,plan:[{'complete':True,'groups':groups(1,1)}],
                              schedule_factory=schedule,profile='comprehensive')
        self.assertFalse(result['complete'])

    def test_check_reports_bounded_plan_without_ml(self):
        from scripts.smoke_duel_v2_full import main
        with contextlib.redirect_stdout(io.StringIO()) as out:
            main(['--check','--profile','comprehensive','--seconds','21600'])
        plan=json.loads(out.getvalue())
        self.assertFalse(plan['training_started'])
        self.assertAlmostEqual(sum(plan['seconds_per_preset'].values())*2+plan['reserve_seconds'],21600)
        self.assertEqual(plan['candidate_plan']['pairs_per_lineup'],[8,32,128])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_pipeline_prepares_both_bases_and_adapts_selected_cards(self):
        from scripts.smoke_duel_v2_full import main
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run'
            def train(command,**kwargs):
                out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
                calls.append((out.parent.name,out.name,command))
                (out/'latest.pt').write_text('synthetic')
                (out/'metrics.jsonl').write_text('{"update":1}\n')
                if out.name=='cards':
                    self.assertTrue((root/'starter/base/latest.pt').exists())
                    self.assertTrue((root/'weave-rush/base/latest.pt').exists())
                    self.assertIn('--opponent-checkpoint',command)
                    deck=preset_opponent_deck(out.parent.name)
                    old,new=('N02','N01') if out.parent.name=='starter' else ('M01','M02')
                    deck['card_ids']=[new if c==old else c for c in deck['card_ids']]
                    (out/'candidate.json').write_text(json.dumps(deck))
                return SimpleNamespace(returncode=0)
            def fake_tournament(candidates,evaluate,**kwargs):
                # Pipeline passes the real comparison adapter; exercise it with fixed opponents.
                evaluate([c['build'] for c in candidates],schedule(2))
                winner=candidates[-1]
                results=[{**c,'complete':True,'groups':groups(2,2),'metrics':metrics(groups(2,2))} for c in candidates]
                return {'complete':True,'selection':winner['id'],'build':winner['build'],
                        'stages':[{'results':results}],'reason':'synthetic pipeline selection'}
            def offline(out,policy,**kwargs):
                out.mkdir(parents=True)
                result={'complete':True,'pairs':kwargs['pairs'],'games':kwargs['pairs']*2,
                        'by_position':{'first':{'win':1},'second':{'win':1}},
                        'model':policy.model_metadata,'learner_build':kwargs['learner_deck'],
                        'games_index':[{'game_id':'fixture'}]}
                (out/'report.json').write_text(json.dumps(result));return result
            with patch('subprocess.run',side_effect=train), \
                 patch('app.modules.card_game.rl.public_export.export_checkpoint'), \
                 patch('app.modules.card_game.engine.ai.advanced_model.FrozenModel',return_value=object()), \
                 patch('app.modules.card_game.rl.offline_analysis.frozen_model_policy',return_value=SimpleNamespace(model_metadata={'sha256':'fake'})), \
                 patch('app.modules.card_game.rl.offline_analysis.run_offline_analysis',side_effect=offline), \
                 patch('app.modules.card_game.rl.offline_analysis.export_game_replay'), \
                 patch('app.modules.card_game.rl.full_report.write_full_report',return_value={'report_complete':True}), \
                 patch('app.modules.card_game.rl.card_tuning.load_battle_policy',return_value=(object(),{'sha256':'fake'})), \
                 patch('app.modules.card_game.rl.card_tuning.CardBattleEvaluator'), \
                 patch('app.modules.card_game.rl.candidate_selection.evaluate_tensor_candidates',return_value=[]), \
                 patch('app.modules.card_game.rl.local_card_search.refine_locally',side_effect=lambda baseline,evaluate,**kw:fake_tournament([{'id':'original','build':baseline},{'id':'local','build':kw['initial']}],evaluate)), \
                 patch('app.modules.card_game.rl.candidate_selection.run_tournament',side_effect=fake_tournament):
                with contextlib.redirect_stdout(io.StringIO()):main(['--output',str(root),'--phase','train'])
                with contextlib.redirect_stdout(io.StringIO()):main(['--output',str(root),'--phase','evaluate'])
                self.assertEqual(len(json.loads((root/'evaluation-index.json').read_text())),34)
                cache=root/'evaluations/starter-original-rule-starter/cache-identity.json'
                saved=json.loads(cache.read_text())
                for field,value in [('seed',-1),('opponent',{'sha256':'different'}),('opponent_build',{})]:
                    cache.write_text(json.dumps({**saved,field:value}))
                    with self.assertRaisesRegex(RuntimeError,'Cached opponent/seed/rule identity changed'), contextlib.redirect_stdout(io.StringIO()):
                        main(['--output',str(root),'--phase','evaluate'])
                    cache.write_text(json.dumps(saved))
            self.assertEqual([(k,s) for k,s,_ in calls][:2],[('starter','base'),('weave-rush','base')])
            for key,stage,command in calls:
                if stage=='adapt':self.assertEqual(command[command.index('--learner-deck')+1],str(root/key/'selected-deck.json'))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_native_batch_reports_real_lineups_and_both_positions(self):
        import torch
        from app.modules.card_game.rl.card_tuning import CardBattleEvaluator
        from app.modules.card_game.rl.candidate_opponents import build_opponent_schedule
        from app.modules.card_game.rl.candidate_selection import evaluate_tensor_candidates
        class EndPolicy(torch.nn.Module):
            def forward(self,state,candidates,mask):
                logits=(candidates[:,:,0]==0).float().masked_fill(~mask,-torch.inf)
                return logits,torch.zeros(len(state))
        net=EndPolicy()
        plan=build_opponent_schedule(1,seed=75,random_fraction=0,rule_fraction=0,model_ids=('learner','peer-0'))
        with tempfile.TemporaryDirectory() as directory:
            evaluator=CardBattleEvaluator(net,opponents=[net],device='cpu',compiled_dir=directory)
            try:
                result=evaluate_tensor_candidates(evaluator,[preset_opponent_deck('starter')],plan,stop_check=lambda:None,max_actions=160)
            finally:evaluator.close()
        self.assertTrue(result[0]['complete'])
        self.assertEqual(set(result[0]['groups']),{'starter','weave-rush'})
        for group in result[0]['groups'].values():
            for value in group.values():
                self.assertEqual(value['games'],1)
                self.assertEqual(value['truncated'],0)
