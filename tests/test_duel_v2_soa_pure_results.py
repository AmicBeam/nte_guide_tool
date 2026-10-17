"""Synthetic CPU receipts with explicit fake oracles; never CUDA evidence."""
from copy import deepcopy
import hashlib
import pickle
import time
import unittest
from unittest.mock import patch

import numpy as np

from app.modules.card_game.rl.batched_search.soa_pure_jobs import adapt_pure_jobs
from app.modules.card_game.rl.batched_search.soa_pure_results import verified_pure_calibration_games
from app.modules.card_game.rl.batched_search.soa_training_jobs import policy_source_alias
from tests.test_duel_v2_soa_pure_jobs import calibration, comparison, models, FOLDER, HISTORICAL
from tests.test_duel_v2_soa_training_results import observe_stub, actor_stub, clean_stub, encode_stub, apply_stub


def initial_stub(parameters):
    assert parameters['skip_mulligan'] is False
    shared = ['alias']
    return dict(version=0, phase='mulligan', active_side=parameters['first_side'],
        first_side=parameters['first_side'], winner=None, reason=None, turn=1, rng=parameters['seed'],
        sides={side: dict(order=deepcopy(parameters['decks'][side]['character_ids']),
                         original_cards=deepcopy(parameters['decks'][side]['card_ids'])) for side in ('a', 'b')},
        alias_left=shared, alias_right=shared,
        events=[dict(type='initial-event')], _public_board=dict(initial_cache=True))


def apply_oracle(state, side, action):
    result = apply_stub(state, side, action)
    if result['phase'] == 'finished':
        result['reason'] = 'player_hp'
    return result


def scores_stub(x, c):
    return -np.arange(len(c), dtype=np.float32)


ORACLE = observe_stub, actor_stub, apply_oracle, clean_stub, encode_stub


def fixture():
    bank = models()
    for model in bank.values(): model.scores = scores_stub
    submitted = calibration()[:2]
    submitted[0]['train_sides'] = ['a']
    metadata = adapt_pure_jobs(submitted, bank)
    initial = [initial_stub(parameters) for parameters in metadata.initializers]
    values, raw, references, final = [], [], {}, []
    for owner, job in enumerate(metadata.jobs):
        state = deepcopy(initial[owner])
        for step in range(2):
            side = actor_stub(state);actor = int(side == 'b')
            view = observe_stub(state, side, include_previews=False)
            actions = [v['action'] for v in view['legal_actions'] if v['action']['type'] != 'concede']
            x, c = encode_stub(view, actions)
            logits = scores_stub(x, c);selected = int(np.argmax(logits))
            words = np.frombuffer(hashlib.sha256(x.tobytes()+c.tobytes()).digest(), dtype='>u4').astype(np.int64)
            references[owner, step] = dict(owner=owner, step=step, actor=actor, job_seed=job['seed'],
                version=state['version'], encode_error=0, inference_error=0, requested=0, completed=0,
                inference_backend='cuda', actions=deepcopy(actions), x=x, c=c, logits=logits,
                selected=selected, info_hash=words, source_alias=policy_source_alias(*job['policies'][side]),
                model_version=metadata.model_versions[owner][side])
            raw.append(dict(game=owner, step=step, owner=owner, actor=actor, seed=job['seed'],
                version=state['version'], chosen=selected, raw_action=deepcopy(actions[selected])))
            for viewer, viewer_side in enumerate(('a', 'b')):
                vx, _ = encode_stub(observe_stub(state, viewer_side, include_previews=False), [])
                values.append(dict(owner=owner, step=step, actor=viewer, viewer=viewer, physical_actor=actor,
                    x=vx, value_actor=float(viewer == actor), z=1. if job['first'] == viewer_side else -1.,
                    version=state['version'], job_seed=job['seed'], pure_value=True, slot=owner, generation=1,
                    learning_eligible=metadata.train_masks[owner][viewer]))
            state = apply_oracle(state, side, actions[selected])
        final.append(state)
    return dict(dataset=dict(complete=True, error_rows=0, optimizer_steps=0, truncated=False, policy=[],
                    execution=deepcopy(metadata.execution), model_versions=deepcopy(list(metadata.model_versions)), values=values),
        initial=initial, final=final, raw_records=raw, metadata=metadata, models=bank, policy_reference=references)


class SoAPureResultsTests(unittest.TestCase):
    def checked(self, args):
        with patch('app.modules.card_game.rl.batched_search.soa_pure_results._oracle', return_value=ORACLE), \
             patch('app.modules.card_game.rl.batched_search.soa_pure_results._initial_state', side_effect=initial_stub):
            return verified_pure_calibration_games(**args)

    def test_calibration_rows_masked_audit_complete_and_inputs_unchanged(self):
        args = fixture();before = pickle.dumps(args)
        games = self.checked(args)
        self.assertEqual(pickle.dumps(args), before)
        for owner, game in enumerate(games):
            self.assertEqual(game['job'], args['metadata'].jobs[owner])
            self.assertEqual(game['metadata'], args['metadata'].identities[owner])
            self.assertEqual(game['model_versions'], args['metadata'].model_versions[owner])
            self.assertEqual((game['rows'], game['searched'], game['decisions']), ([], 0, 2))
            self.assertNotIn('behavior', game)
            self.assertEqual(game['reason'], 'player_hp')
            self.assertFalse(game['learning_eligible'])
            self.assertFalse(game['formal_training_qualified']);self.assertFalse(game['gpu_qualified'])
            self.assertEqual(len(game['audit_value_rows']), 4)
            self.assertEqual(len(game['value_rows']), 2*sum(args['metadata'].train_masks[owner]))
            self.assertTrue(all(row['learning_eligible'] for row in game['value_rows']))
            self.assertIsNone(game['job']['value_precheck'])
        self.assertEqual({row['side'] for row in games[0]['value_rows']}, {'a'})
        self.assertEqual({row['side'] for row in games[0]['audit_value_rows']}, {'a', 'b'})
        games[0]['value_rows'][0]['x'][0] = 999
        games[0]['job']['decks']['a']['card_ids'].reverse()
        self.assertEqual(pickle.dumps(args), before)

    def test_missing_duplicate_or_wrong_coordinate_actor_seed_and_version_rejected(self):
        def missing(a):a['dataset']['values'].pop()
        def duplicate(a):a['dataset']['values'].append(deepcopy(a['dataset']['values'][0]))
        def raw_duplicate(a):a['raw_records'].append(deepcopy(a['raw_records'][0]))
        def sequence(a):a['raw_records'][0],a['raw_records'][1]=a['raw_records'][1],a['raw_records'][0]
        def actor(a):a['raw_records'][0]['actor']=1
        def seed(a):a['raw_records'][0]['seed']+=1
        def version(a):a['dataset']['values'][0]['version']+=1
        def viewer(a):a['dataset']['values'][0]['viewer']=True
        def reference(a):del a['policy_reference'][0,0]
        def lease(a):a['dataset']['values'][1]['generation']+=1
        def missing_lease(a):del a['dataset']['values'][0]['slot']
        def raw_search(a):a['raw_records'][0]['completed']=32
        def raw_type(a):a['raw_records'][0]['raw_action']['target']=False
        for mutate in (missing, duplicate, raw_duplicate, sequence, actor, seed, version, viewer, reference,
                       lease, missing_lease, raw_search, raw_type):
            args=fixture();mutate(args);before=pickle.dumps(args)
            with self.subTest(mutate=mutate.__name__),self.assertRaises(ValueError):self.checked(args)
            self.assertEqual(pickle.dumps(args),before)

    def test_wrong_context_targets_encodings_models_argmax_and_information_rejected(self):
        def context(a):a['dataset']['values'][0]['value_actor']=0.
        def z(a):a['dataset']['values'][0]['z']*=-1
        def eligible(a):a['dataset']['values'][1]['learning_eligible']=True
        def observation(a):a['dataset']['values'][0]['x'][0]+=1
        def candidates(a):a['policy_reference'][0,0]['c'][0,0]+=1
        def action_order(a):a['policy_reference'][0,0]['actions'].reverse()
        def hash_(a):a['policy_reference'][0,0]['info_hash'][0]^=1
        def source(a):a['policy_reference'][0,0]['source_alias']='wrong'
        def version(a):a['policy_reference'][0,0]['model_version']='0'*64
        def logits(a):a['policy_reference'][0,0]['logits']*=-1
        def chosen(a):
            a['policy_reference'][0,0]['selected']=1
            a['raw_records'][0].update(chosen=1,raw_action=dict(type='end_turn'))
        def numpy(a):
            for model in a['models'].values():model.scores=lambda x,c:np.ones(len(c),np.float32)*5
        for mutate in (context,z,eligible,observation,candidates,action_order,hash_,source,version,logits,chosen,numpy):
            args=fixture();mutate(args)
            with self.subTest(mutate=mutate.__name__),self.assertRaises(ValueError):self.checked(args)

    def test_partial_error_search_labels_initial_order_rng_alias_and_deadline_rejected(self):
        def complete(a):a['dataset']['complete']=False
        def truncated(a):a['dataset']['truncated']=True
        def errors(a):a['dataset']['error_rows']=1
        def optimizer(a):a['dataset']['optimizer_steps']=1
        def labels(a):a['dataset']['policy']=[{}]
        def half(a):a['final'][0]['phase']='playing'
        def reason(a):a['final'][0]['reason']='timeout'
        def rng(a):a['final'][0]['rng']+=1
        def aliases(a):a['final'][0]['alias_right']=list(a['final'][0]['alias_right'])
        def cards(a):a['initial'][0]['sides']['a']['original_cards'].reverse()
        def initial_rng(a):a['initial'][0]['rng']+=1
        def models_(a):a['dataset']['model_versions'][0]['a']='0'*64
        for mutate in (complete,truncated,errors,optimizer,labels,half,reason,rng,aliases,cards,initial_rng,models_):
            args=fixture();mutate(args)
            with self.subTest(mutate=mutate.__name__),self.assertRaises(ValueError):self.checked(args)
        args=fixture();args['deadline']=time.monotonic()-1
        with self.assertRaises(TimeoutError):self.checked(args)

    def test_comparison_roles_rejected_even_when_pure(self):
        args=fixture();bank=models(FOLDER,HISTORICAL)
        args['metadata']=adapt_pure_jobs(comparison()[:1],bank,reference_models=models(HISTORICAL))
        args['models']=bank
        with self.assertRaisesRegex(ValueError,'comparison'):self.checked(args)

    def test_real_initializer_preserves_events_and_public_cache_without_clean(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.batched_search.soa_pure_results import _initial_state
        from app.modules.card_game.rl.batched_search.soa_training_results import _same_graph
        args=fixture()
        parameters=args['metadata'].initializers[0]
        before=pickle.dumps(parameters)
        with patch('app.modules.card_game.rl.league_rollout.clean',
                   side_effect=AssertionError('Initial state must not be cleaned')):
            actual=_initial_state(parameters)
        self.assertTrue(actual['events'])
        self.assertIn('_public_board',actual)
        _same_graph(actual,new_game(**deepcopy(parameters)))
        self.assertEqual(pickle.dumps(parameters),before)
        # The fake-oracle batch also retains explicit nonempty initialization
        # events/cache; deleting them cannot be repaired by proof-time cleaning.
        for field in ('events','_public_board'):
            bad=fixture()
            if field=='events':bad['initial'][0][field]=[]
            else:del bad['initial'][0][field]
            before=pickle.dumps(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):self.checked(bad)
            self.assertEqual(pickle.dumps(bad),before)


if __name__=='__main__':unittest.main()
