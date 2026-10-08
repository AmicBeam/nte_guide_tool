import json
import tempfile
import unittest
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import numpy as np

from app.modules.card_game.rl.information_search import Node, search, action_identity
from app.modules.card_game.rl.search_label_counterfactual import (
    INITIATIVES, MAX_WALL_SECONDS, PAIRED_ROOT_RUNTIME, ROSTER, SEARCH,
    accept_root_diagnostics, dry_run, evaluate_paired_roots, pair_forced_root,
    planned_roots, prepare_run, root_is_eligible, select_paid_alternative,
    validate_output_path, validate_seed_reservation,
)
from app.modules.card_game.rl.search_policy import visit_schedule, choose_root


class Prior:
    def scores_value(self, x, c):
        scores = np.linspace(5., 0., len(c), dtype=np.float64)
        return scores, 0.


class Many:
    n = 20

    @staticmethod
    def decision(s, side):
        actions = [{'type': 'wait', 'i': index} for index in range(Many.n)]
        return actions, (np.array([s.get('step', 0)], np.float32), np.eye(Many.n, dtype=np.float32))

    @staticmethod
    def determinize(s, viewer, seed):
        return {**s, 'latent': seed % 2}


def _act(s, side, action):
    state = deepcopy(s)
    state['step'] = state.get('step', 0) + 1
    if state['step'] > 4:
        state.update(phase='finished', winner='a')
    return state


def _search(**extra):
    with patch('app.modules.card_game.rl.information_search.apply_action', side_effect=_act):
        return search(dict(phase='playing', active_side='a', turn=1), 'a',
                      {'a': Prior(), 'b': Prior()}, runtime=Many, seed=11,
                      simulations=SEARCH['simulations'], fast_simulation=False,
                      algorithm='gumbel', gumbel_candidates=SEARCH['gumbel_candidates'],
                      **extra)


def _reservation(start=12_520_000_000):
    from app.modules.card_game.rl.build_acceptance import SEED_PHASES, SEED_WIDTH
    return {phase: start + index * SEED_WIDTH for index, phase in enumerate(SEED_PHASES)}


def _ledger(*runs):
    return dict(runs=list(runs))


def _write_models(root, rule_hash='rule'):
    import hashlib
    from app.modules.card_game.rl.cross_grounded import fingerprint
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for key in ROSTER:
        weights = root / f'{key}.npz'
        np.savez(weights, x=np.array([1.0], dtype=np.float32))
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        build = dict(id=key, character_ids=['a', 'b', 'c', 'd'], card_ids=['x'] * 32)
        build_digest = hashlib.sha256(json.dumps(
            {name: build[name] for name in ('character_ids', 'card_ids')},
            sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest()
        (root / f'{key}.json').write_text(json.dumps(dict(
            schema='cross_five_grounded_wdl_v1', rule_hash=rule_hash, sha256=digest,
            deck=key, build_sha256=build_digest,
            candidate_transform_sha256=fingerprint(), build=build,
        )), encoding='utf-8')
    return root


def _state(ap=2, rng=7, hidden=True):
    foe_hand = [{'hidden': True}] if hidden else [{'card_id': 'secret', 'name': 'hidden-card'}]
    return dict(
        phase='playing', active_side='a', turn=3, rng=rng, winner=None,
        sides={
            'a': dict(hp=30, shield=0, ap=ap, front='hero',
                      characters={'hero': dict(id='hero', hp=8, shield=1)}),
            'b': dict(hp=30, shield=4, ap=2, front='foe',
                      characters={'foe': dict(id='foe', hp=6, shield=2)},
                      hand=foe_hand),
        },
    )


def _actions():
    extras = [{'type': 'play_card', 'card_id': f'free-{index}'} for index in range(16)]
    return [
        {'type': 'end_turn'},
        {'type': 'attack', 'character_id': 'hero'},
        {'type': 'play_card', 'card_id': 'paid-card'},
        *extras,
    ]


def _diagnostics(actions, admitted=None, selected=0, completed_q=None):
    n = len(actions)
    admitted = [True] * min(16, n) + [False] * max(0, n - 16) if admitted is None else list(admitted)
    prior = np.linspace(0.2, 0.01, n)
    prior = (prior / prior.sum()).tolist()
    visits = [2 if flag else 0 for flag in admitted]
    q = [0.1 * index for index in range(n)] if completed_q is None else list(completed_q)
    return dict(
        action_identities=[action_identity(action) for action in actions],
        prior=prior,
        admitted_to_max16=admitted,
        visits=visits,
        completed_q=q,
        selected_index=selected,
        selected_identity=action_identity(actions[selected]),
        requested_simulations=32,
        completed_simulations=32,
        candidate_cap=16,
    )


def _search_result(actions, diagnostics, selected=0):
    pi = np.array(diagnostics['prior'], dtype=np.float32)
    return dict(
        complete=True,
        actions=actions,
        pi=pi,
        simulations=32,
        search_choice=selected,
        root_diagnostics=diagnostics,
    )


class Policy:
    def __init__(self, key):
        self.serving_deck = dict(id=key, character_ids=['a', 'b', 'c', 'd'], card_ids=['x'] * 32)


class SearchLabelCounterfactualTest(unittest.TestCase):
    def test_diagnostics_absent_keep_output_and_rng(self):
        plain = _search()
        diagnosed = _search(diagnostics=True)
        self.assertNotIn('root_diagnostics', plain)
        self.assertIn('root_diagnostics', diagnosed)
        for key, value in plain.items():
            other = diagnosed[key]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other)
            else:
                self.assertEqual(value, other)
        np.testing.assert_array_equal(plain['root_gumbel'], diagnosed['root_gumbel'])
        np.testing.assert_array_equal(plain['visits'], diagnosed['visits'])
        self.assertEqual(plain['search_choice'], diagnosed['search_choice'])

    def test_candidate_admission_covers_all_legal_actions_and_max16(self):
        result = _search(diagnostics=True)
        diagnostics = accept_root_diagnostics(result)
        self.assertEqual(len(diagnostics['action_identities']), Many.n)
        self.assertEqual(diagnostics['action_identities'], [action_identity(action) for action in result['actions']])
        self.assertEqual(diagnostics['candidate_cap'], 16)
        admitted = [index for index, flag in enumerate(diagnostics['admitted_to_max16']) if flag]
        self.assertEqual(len(admitted), 16)
        self.assertEqual(admitted, list(range(16)))
        self.assertTrue(all(diagnostics['visits'][index] > 0 for index in admitted))
        self.assertTrue(all(diagnostics['visits'][index] == 0 for index in range(16, Many.n)))
        self.assertEqual(int(np.sum(result['visits'] > 0)), 16)

    def test_gumbel_top_k_matches_zero_noise_admission(self):
        prior = np.linspace(0.2, 0.01, 20)
        prior = prior / prior.sum()
        node = Node.create(prior, 0.)
        schedule = visit_schedule(16, 32)
        for visit in schedule:
            chosen = choose_root(node, np.zeros(20), visit)
            node.visits[chosen] += 1
        self.assertEqual(sorted(np.flatnonzero(node.visits > 0).tolist()), list(range(16)))

    def test_nonfinite_and_incomplete_diagnostics_are_rejected(self):
        incomplete = _search(diagnostics=True, deadline=0)
        self.assertFalse(incomplete['complete'])
        with self.assertRaises(ValueError):
            accept_root_diagnostics(incomplete)
        result = _search(diagnostics=True)
        broken = deepcopy(result)
        broken['root_diagnostics']['completed_q'] = list(broken['root_diagnostics']['completed_q'])
        admitted = next(index for index, flag in enumerate(broken['root_diagnostics']['admitted_to_max16']) if flag)
        broken['root_diagnostics']['completed_q'][admitted] = float('nan')
        with self.assertRaises(ValueError):
            accept_root_diagnostics(broken)
        missing = dict(result)
        missing.pop('root_diagnostics')
        with self.assertRaises(ValueError):
            accept_root_diagnostics(missing)

    def test_duplicate_and_old_seeds_are_rejected_but_own_reservation_is_allowed(self):
        seeds = _reservation(12_030_000_000)
        ledger = _ledger(dict(run_id='five-cross-20260924-formal', start=12_030_000_000,
                              end=12_100_000_000, seeds=seeds))
        with self.assertRaises(ValueError):
            validate_seed_reservation(seeds, ledger)
        own = validate_seed_reservation(seeds, ledger, run_id='five-cross-20260924-formal')
        self.assertEqual(own['run_id'], 'five-cross-20260924-formal')
        self.assertEqual(len(own['planned']), 20)
        duplicate = _reservation()
        duplicate['audit'] = duplicate['confirmation']
        with self.assertRaises(ValueError):
            validate_seed_reservation(duplicate, ledger)
        ok = validate_seed_reservation(_reservation(), ledger)
        self.assertFalse(ok['archived_overlap'])
        self.assertEqual(len(ok['planned']), 20)

    def test_plan_covers_ten_cross_pairs_and_both_first_directions(self):
        plan = planned_roots(_reservation())
        self.assertEqual(len(plan), 20)
        pairs = [(row['left'], row['right']) for row in plan]
        self.assertEqual(len(set(pairs)), 10)
        self.assertEqual({team for row in plan[:8] for team in (row['left'], row['right'])}, set(ROSTER))
        self.assertEqual({row['first'] for row in plan}, set(INITIATIVES))
        self.assertEqual([row['first'] for row in plan], ['a', 'b'] * 10)
        for index in range(0, 20, 2):
            self.assertEqual((plan[index]['left'], plan[index]['right']),
                             (plan[index + 1]['left'], plan[index + 1]['right']))
            self.assertEqual(plan[index]['seed'], plan[index + 1]['seed'])
            self.assertEqual((plan[index]['first'], plan[index + 1]['first']), ('a', 'b'))
        self.assertNotIn('starter', [row['left'] for row in plan if row['right'] == 'starter'])

    def test_root_legality_requires_ap2_end_turn_and_paid_action(self):
        actions = _actions()
        cost = lambda state, side, action, view=None: 1 if action['type'] in ('attack', 'play_card') and not str(action.get('card_id', '')).startswith('free-') else 0
        self.assertTrue(root_is_eligible(_state(ap=2), 'a', actions, action_cost=cost))
        self.assertFalse(root_is_eligible(_state(ap=1), 'a', actions, action_cost=cost))
        self.assertFalse(root_is_eligible(_state(ap=2), 'a', [actions[0]], action_cost=cost))
        self.assertFalse(root_is_eligible(_state(ap=2, rng=1) | dict(phase='choice'), 'a', actions, action_cost=cost))

    def test_cloned_rng_identity_and_hidden_info_boundary(self):
        actions = _actions()
        diagnostics = _diagnostics(actions)
        result = _search_result(actions, diagnostics)
        worlds = []

        def sample_world(state, viewer, seed):
            world = deepcopy(state)
            world['rng'] = 99
            world['private'] = {'b': 'secret-hand'}
            worlds.append(world)
            return world

        def apply_action(state, side, action):
            nxt = deepcopy(state)
            nxt['rng'] = state['rng'] + (0 if action['type'] == 'end_turn' else 3)
            if action['type'] == 'attack':
                nxt['sides']['b']['hp'] -= 2
                nxt['sides']['a']['ap'] -= 1
            if action['type'] == 'end_turn':
                nxt['sides']['a']['ap'] = 0
            return nxt

        def observe(state, side):
            view = deepcopy(state)
            view['sides']['b']['hand'] = [{'hidden': True}]
            if side != 'a':
                raise AssertionError('selector must not observe as the opponent')
            return view

        def cost(state, side, action, view=None):
            if view is not None and any(card.get('card_id') for card in view['sides']['b']['hand']):
                raise AssertionError('hidden cards leaked into cost')
            return 1 if action['type'] in ('attack', 'play_card') and not str(action.get('card_id', '')).startswith('free-') else 0

        def continuation(state, label, deadline):
            return dict(complete=True, state=state, steps=0)

        paired = pair_forced_root(
            _state(), 'a', result, diagnostics, sample_world=sample_world, apply_action=apply_action,
            clean=lambda state: state, observe=observe, action_cost=cost, snapshot=lambda state: {
                'phase': state['phase'], 'winner': state.get('winner'), 'rng': state['rng'],
                'sides': {side: dict(hp=team['hp'], shield=team['shield'], ap=team['ap'],
                                     characters={cid: dict(hp=hero['hp'], shield=hero['shield'], alive=hero['hp'] > 0)
                                                 for cid, hero in team['characters'].items()})
                          for side, team in state['sides'].items()},
            }, world_seed=123, continuation=continuation,
        )
        self.assertTrue(paired['paired'])
        self.assertTrue(paired['cloned_rng_identical'])
        self.assertEqual(paired['cloned_rng'], 99)
        self.assertEqual(worlds[0]['rng'], 99)
        self.assertTrue(paired['immediate']['diverged'])
        self.assertEqual(paired['immediate']['ap_spend'], 1)
        self.assertTrue(paired['immediate']['attacker_survived'])
        self.assertTrue(paired['visible_only'])
        self.assertFalse(paired['full_game_win_rate'])

        leaked = pair_forced_root(
            _state(hidden=False), 'a', result, diagnostics, sample_world=sample_world, apply_action=apply_action,
            clean=lambda state: state, observe=lambda state, side: state, action_cost=cost,
            snapshot=lambda state: state, world_seed=1, continuation=continuation,
        )
        self.assertFalse(leaked['paired'])
        self.assertIn('hidden', leaked['invalid_reason'])

    def test_excluded_top16_forced_choice_is_reported(self):
        actions = _actions()
        admitted = [False] * len(actions)
        admitted[0] = True
        diagnostics = _diagnostics(actions, admitted=admitted, selected=0)
        result = _search_result(actions, diagnostics)
        paid = select_paid_alternative(
            _state(), 'a', result, diagnostics,
            observe=lambda state, side: {'sides': {'b': {'hand': [{'hidden': True}]}}},
            action_cost=lambda state, side, action, view=None: 1 if action['type'] in ('attack', 'play_card') and not str(action.get('card_id', '')).startswith('free-') else 0,
        )
        self.assertTrue(paid['excluded_top16'])
        self.assertEqual(paid['action']['type'], 'attack')

    def test_incomplete_q_and_wall_limit_reject_pairing(self):
        result = _search(diagnostics=True)
        broken = deepcopy(result)
        broken['root_diagnostics']['completed_q'] = list(broken['root_diagnostics']['completed_q'])
        admitted = next(index for index, flag in enumerate(broken['root_diagnostics']['admitted_to_max16']) if flag)
        broken['root_diagnostics']['completed_q'][admitted] = float('nan')
        with self.assertRaises(ValueError):
            accept_root_diagnostics(broken)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            output = root / 'incomplete'
            with self.assertRaises(ValueError):
                evaluate_paired_roots(seed_reservation=_reservation(), models=models, output=output,
                                      ledger=_ledger(), rule_hash='rule', seconds=30, max_games=1, max_roots=1,
                                      hooks=_hooks(fail_search=True))
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['failed'])
            self.assertFalse(report['complete'])
            self.assertFalse(report['optimizer'])
            later = root / 'wall'
            wall_report = evaluate_paired_roots(seed_reservation=_reservation(), models=models, output=later,
                                                ledger=_ledger(), rule_hash='rule', seconds=1, max_games=20,
                                                max_roots=20, hooks=_hooks(clock_limit=True), clock=_Clock(step=0.4))
            self.assertTrue(wall_report['denominators']['wall_limited'])
            self.assertGreaterEqual(wall_report['denominators']['roots_unpaired'], 1)
            self.assertFalse(wall_report['complete'])
            self.assertFalse(wall_report['optimizer'])
            self.assertFalse(wall_report['training'])
            self.assertFalse(wall_report['full_game_win_rate'])
            self.assertTrue((later / 'roots.jsonl').exists())

    def test_dry_run_validates_identities_and_creates_no_output(self):
        self.assertTrue(PAIRED_ROOT_RUNTIME)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            seeds_path = root / 'seeds.json'
            seeds_path.write_text(json.dumps(_reservation()), encoding='utf-8')
            output = root / 'new-run'
            ledger = _ledger(dict(run_id='old', start=10_000_000_000, end=12_520_000_000,
                                  seeds=_reservation(10_000_000_000)))
            result = dry_run(seed_reservation=seeds_path, models=models, output=output,
                             ledger=ledger, rule_hash='rule')
            self.assertFalse(output.exists())
            self.assertFalse(result['output_created'])
            self.assertEqual(result['games_played'], 0)
            self.assertEqual(result['roster'], list(ROSTER))
            self.assertEqual(result['initiatives'], list(INITIATIVES))
            self.assertEqual(len(result['planned']), 20)
            self.assertTrue(all(item['rule_hash'] == 'rule' for item in result['identities']['models'].values()))
            self.assertFalse(result['optimizer'])
            validate_output_path(output)
            output.mkdir()
            with self.assertRaises(ValueError):
                dry_run(seed_reservation=seeds_path, models=models, output=output,
                        ledger=ledger, rule_hash='rule')

    def test_dry_run_rejects_build_or_transform_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            seeds = _reservation()
            ledger = _ledger()
            manifest_path = models / 'starter.json'
            original = json.loads(manifest_path.read_text(encoding='utf-8'))
            changed = dict(original, build_sha256='wrong')
            manifest_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Build SHA mismatch'):
                dry_run(seed_reservation=seeds, models=models, output=root / 'out',
                        ledger=ledger, rule_hash='rule')
            changed = dict(original, candidate_transform_sha256='wrong')
            manifest_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Candidate transform mismatch'):
                dry_run(seed_reservation=seeds, models=models, output=root / 'out',
                        ledger=ledger, rule_hash='rule')

    def test_cli_dry_run_prints_plan_without_creating_output(self):
        from scripts.check_duel_v2_search_label_counterfactual import main
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            seeds_path = root / 'seeds.json'
            seeds_path.write_text(json.dumps(_reservation()), encoding='utf-8')
            ledger_path = root / 'ledger.json'
            ledger_path.write_text(json.dumps(_ledger(dict(run_id='old', start=1, end=2, seeds={'foundation': 1}))), encoding='utf-8')
            output = root / 'cli-out'
            with patch('sys.stdout', StringIO()):
                result = main(['--seed-reservation', str(seeds_path), '--models', str(models),
                               '--output', str(output), '--seed-ledger', str(ledger_path), '--rule-hash', 'rule'])
            self.assertFalse(output.exists())
            self.assertEqual(result['runtime_enabled'], True)
            self.assertEqual(len(result['planned']), 20)
            self.assertEqual(result['games_played'], 0)

    def test_evaluate_pairs_twenty_jobs_without_optimizer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            models = _write_models(root / 'models')
            output = root / 'eval'
            report = evaluate_paired_roots(
                seed_reservation=_reservation(), models=models, output=output, ledger=_ledger(),
                rule_hash='rule', seconds=30, max_games=20, max_roots=20, hooks=_hooks(),
            )
            self.assertEqual(report['denominators']['games_played'], 20)
            self.assertEqual(report['denominators']['roots_paired'], 20)
            self.assertEqual(report['planned'], 20)
            self.assertFalse(report['optimizer'])
            self.assertFalse(report['training'])
            self.assertFalse(report['serving_export'])
            self.assertFalse(report['full_game_win_rate'])
            self.assertLessEqual(report['seconds_limit'], MAX_WALL_SECONDS)
            lines = (output / 'roots.jsonl').read_text(encoding='utf-8').strip().splitlines()
            self.assertEqual(len(lines), 20)
            firsts = [json.loads(line)['job']['first'] for line in lines]
            self.assertEqual(firsts, ['a', 'b'] * 10)
            self.assertTrue(all(json.loads(line)['paired'] for line in lines))
            self.assertTrue(any(json.loads(line).get('excluded_top16') for line in lines))
            with self.assertRaises(ValueError):
                prepare_run(seed_reservation=_reservation(), models=models, output=root / 'too-long',
                            ledger=_ledger(), rule_hash='rule', seconds=3601)

    def test_default_pairing_waits_for_a_search_selected_end(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = evaluate_paired_roots(
                seed_reservation=_reservation(), models=_write_models(root / 'models'),
                output=root / 'eval', ledger=_ledger(), rule_hash='rule',
                seconds=1, max_games=1, max_roots=1, hooks=_hooks(selected=1),
                clock=_Clock(step=0.01),
            )
            self.assertGreaterEqual(report['denominators']['roots_eligible'], 1)
            self.assertEqual(report['denominators']['roots_end_selected'], 0)
            self.assertEqual(report['denominators']['roots_paired'], 0)


class _Clock:
    def __init__(self, step=1):
        self.now = 0
        self.step = step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def _hooks(fail_search=False, clock_limit=False, selected=0):
    actions = _actions()
    diagnostics = _diagnostics(actions, selected=selected)
    result = _search_result(actions, diagnostics, selected=selected)
    incomplete = dict(complete=False, actions=actions, pi=np.array(diagnostics['prior'], dtype=np.float32),
                      simulations=1, search_choice=0)

    class Counter:
        searches = 0

    def new_game(seed, first_side, decks):
        state = _state(rng=seed)
        state['first'] = first_side
        state['seed'] = seed
        return state

    def search(state, viewer, policies, **kwargs):
        Counter.searches += 1
        if viewer != state.get('active_side', viewer):
            raise AssertionError('search viewer is not the acting player')
        if viewer != 'a' and any(not card.get('hidden') for card in (state.get('sides', {}).get('a', {}).get('hand') or []) if isinstance(card, dict)):
            raise AssertionError('opponent search saw hidden acting-player cards')
        if fail_search:
            return incomplete
        if clock_limit:
            return incomplete
        payload = deepcopy(result)
        if Counter.searches % 7 == 0:
            payload['root_diagnostics']['admitted_to_max16'] = [False] * len(actions)
            payload['root_diagnostics']['admitted_to_max16'][0] = True
            payload['root_diagnostics']['visits'] = [2 if index == 0 else 0 for index in range(len(actions))]
        return payload

    def apply_action(state, side, action):
        nxt = deepcopy(state)
        nxt['rng'] = state.get('rng', 0) + (0 if action.get('type') == 'end_turn' else 4)
        if action.get('type') == 'attack':
            foe = 'b' if side == 'a' else 'a'
            nxt['sides'][foe]['hp'] -= 2
            nxt['sides'][side]['ap'] = max(0, int(nxt['sides'][side]['ap']) - 1)
        elif action.get('type') == 'end_turn':
            nxt['sides'][side]['ap'] = 0
            nxt['active_side'] = 'b' if side == 'a' else 'a'
            nxt['phase'] = 'playing'
        return nxt

    def observe(state, side):
        view = deepcopy(state)
        view['sides']['b']['hand'] = [{'hidden': True}]
        return view

    def choose_visible(state, side, policies):
        if side == state.get('active_side'):
            raise AssertionError('opponent policy used the acting player')
        return {'type': 'end_turn'}

    return dict(
        new_game=new_game,
        acting_side=lambda state: state.get('active_side', 'a'),
        apply_action=apply_action,
        clean=lambda state: state,
        observe=observe,
        search=search,
        sample_world=lambda state, viewer, seed: dict(deepcopy(state), rng=state.get('rng', 0), world=seed),
        load_policies=lambda directory, left, right: {'a': Policy(left), 'b': Policy(right)},
        action_cost=lambda state, side, action, view=None: 1 if action.get('type') in ('attack', 'play_card') and not str(action.get('card_id', '')).startswith('free-') else 0,
        choose_visible=choose_visible,
        snapshot=lambda state: {
            'phase': state.get('phase'), 'winner': state.get('winner'), 'rng': state.get('rng'),
            'sides': {side: dict(hp=team['hp'], shield=team['shield'], ap=team['ap'],
                                 characters={cid: dict(hp=hero['hp'], shield=hero['shield'], alive=hero['hp'] > 0)
                                             for cid, hero in team['characters'].items()})
                      for side, team in state['sides'].items()},
        },
    )


if __name__ == '__main__':
    unittest.main()
