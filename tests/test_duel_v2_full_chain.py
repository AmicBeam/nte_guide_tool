"""Focused lifecycle, counter and offline telemetry contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
class TrainingChainTest(unittest.TestCase):
    def test_terminal_buckets_reset_without_double_counting(self):
        import torch
        from app.modules.card_game.rl.training_metrics import TerminalCounters
        c=TerminalCounters(4,'cpu')
        state=SimpleNamespace(phase=torch.tensor([3,3,3,1]),winner=torch.tensor([0,0,2,-1]),first=torch.tensor([0,0,0,1]))
        learner=torch.tensor([0,1,0,0]);opponent=torch.tensor([0,1,1,0])
        c.record(state,learner,opponent);c.record(state,learner,opponent)
        self.assertEqual(int(c.counts.sum()),3)
        self.assertEqual(c.summary()['frozen_models']['second']['loss'],1)
        c.reset(torch.tensor([True,False,False,False]));c.record(state,learner,opponent)
        self.assertEqual(int(c.counts.sum()),4)

    def test_candidate_keeps_exact_characters(self):
        from app.modules.card_game.rl.public_training import training_deck,fixed_deck_rows
        from app.modules.card_game.rl.capability import preset_opponent_deck
        deck=preset_opponent_deck('starter')
        deck['card_ids']=['N01' if x=='N02' else x for x in deck['card_ids']]
        self.assertEqual(training_deck('starter',deck)['card_ids'],deck['card_ids'])
        self.assertEqual(tuple(fixed_deck_rows('starter',3,'cpu',deck).shape),(3,36))
        deck['character_ids']=list(reversed(deck['character_ids']))
        with self.assertRaisesRegex(ValueError,'characters'):training_deck('starter',deck)


class ReportTelemetryTest(unittest.TestCase):
    def test_sources_and_response_and_harmony_are_distinct(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.report_telemetry import GameTelemetry
        state=new_game(seed=1,skip_mulligan=True)
        t=GameTelemetry(state);card=deepcopy(state['sides']['a']['hand'][0]);cid=card['card_id']
        card['response']='self_attacked'
        after=deepcopy(state);seq=state['event_seq']
        after['events'] += [
            {'seq':seq+1,'type':'draw','side':'a','private_card':card,'patch':{'turn':3}},
            {'seq':seq+2,'type':'play','side':'a','card':card,'patch':{'turn':3}},
            {'seq':seq+3,'type':'harmony','side':'a','actor':'a:nanali','target':'a:zero',
             'before':{'harmony':2},'after':{'harmony':0},'text':'入场，触发创生。','patch':{'turn':3}},
            {'seq':seq+4,'type':'harmony','side':'a','text':'创生后续效果'}]
        t.after(state,after,{'type':'mulligan'},'a')
        bucket=t.cards['a'][cid]
        self.assertEqual(bucket['sources']['mulligan'],1)
        self.assertNotIn('draw',bucket['sources'])
        self.assertEqual(bucket['response_plays'],1)
        self.assertEqual(bucket['first_original_play_turn'],3)
        self.assertEqual(len(t.mechanisms),1)
        self.assertEqual(t.mechanisms[0]['provider'],'a:zero')
        self.assertEqual(t.mechanisms[0]['harmony_type'],'创生')

    def test_extra_report_keeps_telemetry_private(self):
        from app.modules.card_game.rl.offline_analysis import run_offline_analysis
        from app.modules.card_game.rl.policies import VisibleEngineRulePolicy
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'report'
            report=run_offline_analysis(out,VisibleEngineRulePolicy(),pairs=1,max_actions=2,max_seconds=10,seed_base=1)
            game=report['games_index'][0]
            raw=json.loads((out/game['raw_log']).read_text())
            self.assertIsNotNone(raw['telemetry'])
            self.assertIsNone(raw['telemetry']['prediction_probability'])
            public=(out/game['public_replay']).read_text()
            self.assertNotIn('telemetry',public)
            self.assertNotIn('private_card',public)

class ReportBundleTest(unittest.TestCase):
    def test_bundle_has_checksums_and_excludes_private_games(self):
        import hashlib,zipfile
        from app.modules.card_game.rl.report_bundle import package_report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';root.mkdir()
            (root/'status.json').write_text(json.dumps({'phase':'complete'}))
            decisions={key:{'selection':'original'} for key in ('starter','weave-rush')}
            (root/'construction-decisions.json').write_text(json.dumps(decisions))
            for key in decisions:
                export=root/key/'base-export';export.mkdir(parents=True)
                (export/(key+'.npz')).write_bytes(b'synthetic numeric model fixture')
                (export/(key+'.json')).write_text('{}')
                (root/key/'original-deck.json').write_text('{}')
            private=root/'evaluations'/'sample'/'raw';private.mkdir(parents=True)
            (private/'game.json').write_text('{"private_card":"excluded fixture"}')
            selection_private=root/'selection/screen/case/raw';selection_private.mkdir(parents=True)
            (selection_private/'game.json').write_text('{"private_card":"also excluded"}')
            (root/'report.md').write_text('# Synthetic report fixture')
            target=Path(directory)/'bundle.zip'
            result=package_report(root,target)
            self.assertEqual(result['sha256'],hashlib.sha256(target.read_bytes()).hexdigest())
            with zipfile.ZipFile(target) as archive:
                names=archive.namelist()
                self.assertNotIn('evaluations/sample/raw/game.json',names)
                self.assertNotIn('selection/screen/case/raw/game.json',names)
                self.assertIn('final/starter/deck.json',names)
                manifest=json.loads(archive.read('delivery-manifest.json'))
                for name,digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest)
            with self.assertRaises(FileExistsError):package_report(root,target)
            (root/'status.json').write_text(json.dumps({'phase':'failed'}))
            with self.assertRaises(ValueError):package_report(root,Path(directory)/'failed.zip')


class PipelineGuardTest(unittest.TestCase):
    def test_successful_exit_without_update_is_not_a_pass(self):
        from scripts.smoke_duel_v2_full import validate_training_output
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.jsonl'
            for content in ('', '{"updated":false}\n'):
                path.write_text(content)
                with self.assertRaisesRegex(RuntimeError,'without an optimizer update'):
                    validate_training_output(directory)
            path.write_text('{"update":1}\n')
            self.assertEqual(validate_training_output(directory),1)

class FinalReportSelectionTest(unittest.TestCase):
    def test_selection_survives_empty_mechanism_buckets_and_does_not_mutate_matrix(self):
        from app.modules.card_game.rl.capability import preset_opponent_deck
        from app.modules.card_game.rl.full_report import write_full_report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def write(name,value):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            zero=dict(games=0,win=0,loss=0,draw=0,error=0,truncated=0)
            one={**zero,'games':1,'win':1}
            index=[];decisions={};chosen={};screen={};shortlist={};extra=[]
            for key in ('starter','weave-rush'):
                deck=preset_opponent_deck(key)
                write(f'{key}/original-deck.json',deck);write(f'{key}/cards/candidate.json',deck)
                for stage in ('base','cards','adapted'):
                    write(f'{key}/{stage}/status.json',{'update':1,'seconds':1})
                    write(f'{key}/{stage}/metrics.jsonl',{'update':1,'seconds':1,'finished_games':1})
                    write(f'{key}/{stage}/position-results.json',{'opponents':{'rule':{'first':one,'second':zero}}})
                write(f'{key}/cards/candidate-history.jsonl',{'candidate_id':deck['id'],'valid':True,'trials':1,'truncated':0,'score':1})
                label=key+'-original-rule-varied';path=f'evaluations/{label}/report.json'
                write(f'evaluations/{label}/raw/game.json',{'telemetry':{'cards':{'a':{}},'mechanisms':[],
                      'opportunities':[],'turn_states':[],'final_turn':2}})
                write(path,{'learner_build':deck,'model':{'sha256':'synthetic'},'opponent_model':{'type':'rule'},
                    'complete':True,'by_position':{'first':one,'second':zero},'games_index':[{'game_id':'fixture',
                    'winner':'a','first_side':'a','position':'first','terminated':True,'error':None,'truncated':False,
                    'final_turn':2,'raw_log':'raw/game.json','opponent_team':deck['character_ids']}]})
                index.append({'label':label,'preset':key,'variant':'original','report':path})
                decisions[key]={'selection':'original','original_win_rate':1,'candidate_win_rate':1,'sample_games_each':1,'reason':'fixture'}
                chosen[key]={'id':'original','model':'base-export','model_sha256':'synthetic','build':deck}
                screen[key]={'candidates':[{'id':'original','wins':1,'games':1,'cells':[{'opponent':'rule','report':path}]}]}
                shortlist[key]={'history_rows':1,'unique_allocations':1,'shortlist':[{}]}
                extra.append({'label':key+'-selected-rule-varied','preset':key,'report':path})
            write('config.json',{'kind':'comprehensive','rule_hash':'fixture'})
            write('evaluation-index.json',index);write('construction-decisions.json',decisions)
            write('selection/selected.json',{'complete':True,'choices':chosen,'reports':extra})
            write('selection/screen-results.json',screen);write('selection/shortlist.json',shortlist)
            summary=write_full_report(root)
            self.assertEqual(summary['serving_selection'],chosen)
            self.assertIn('最终模型与优选构筑',(root/'report.md').read_text())
            self.assertEqual(json.loads((root/'evaluation-index.json').read_text()),index)
