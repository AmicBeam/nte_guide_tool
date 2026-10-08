import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from app.modules.card_game.rl.cross_teacher_distillation import (
    ACCEPTANCE_BOUNDARY, KL_GRID, NO_OLD_TEACHER, REPO_TEACHERS, TEAMS,
    agreement_record, current_candidates, current_features, dry_run,
    holdout_seeds, named_index_map, overlapping_root, project_candidates,
    project_observation, report_without_old_teacher, select_variant,
    split_paired_roots, summarize_agreement,
    teacher_kl_loss, unsupported_action_mask, update_student,
    validate_kl_grid, verify_old_teacher, verify_repository_teachers,
)
from app.modules.card_game.rl.cross_lineup import identity as current_rule_identity
from app.modules.card_game.rl.fixed_lineup import CARD_IDS as OLD_CARD_IDS, SEATS as OLD_SEATS
from app.modules.card_game.rl.league_schema import ACT_ATTACK, ACT_END, ACT_PLAY


def _games(seeds):
    rows = []
    for seed in seeds:
        for first in ('a', 'b'):
            rows.append(dict(seed=seed, first=first, training=True, rows=[]))
    return rows


def _teacher_stub(real):
    return dict(
        features=list(real['features']),
        candidates=list(real['candidates']),
        observation_index=list(real['observation_index']),
        candidate_index=list(real['candidate_index']),
        weights=real['weights'],
        hidden=real['hidden'],
    )


class CrossTeacherDistillationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teachers = verify_repository_teachers(REPO_TEACHERS)
        cls.starter = cls.teachers['starter']

    def test_repository_teachers_keep_sha_schema_and_build(self):
        current = current_rule_identity()
        for key in TEAMS:
            item = self.teachers[key]
            self.assertEqual(item['schema'], 'fixed_ten_v1')
            self.assertEqual(item['sha256'], item['file_sha256'])
            self.assertEqual(item['sha256'], hashlib.sha256((REPO_TEACHERS / f'{key}.npz').read_bytes()).hexdigest())
            self.assertFalse(item['rule_hash_rewritten'])
            self.assertNotEqual(item['source_rule_hash'], current)
            self.assertEqual(len(item['added_features']), 20)
            self.assertTrue(all(name.startswith('fixed:damage_immune:') for name in item['added_features']))
            self.assertFalse(item['automatic_serving_approval'])
        for key in NO_OLD_TEACHER:
            with self.assertRaisesRegex(ValueError, 'no old teacher'):
                verify_old_teacher(REPO_TEACHERS, key)
            report = report_without_old_teacher(key)
            self.assertFalse(report['claimed_improved'])

    def test_named_columns_map_and_unsupported_actions_are_rejected(self):
        teacher = _teacher_stub(self.starter)
        current_x = np.arange(len(current_features()), dtype=np.float32)
        projected = project_observation(current_x, teacher)
        self.assertEqual(len(projected), len(teacher['features']))
        old_index = teacher['features'].index('turn')
        new_index = current_features().index('turn')
        self.assertEqual(projected[old_index], current_x[new_index])
        immune = current_features().index('fixed:damage_immune:0:nanali')
        self.assertNotIn('fixed:damage_immune:0:nanali', teacher['features'])
        self.assertEqual(current_x[immune], immune)
        option_new = current_candidates().index('option:base')
        option_old = teacher['candidates'].index('option:base')
        self.assertNotEqual(option_new, option_old)
        cand = np.zeros((3, len(current_candidates())), dtype=np.float32)
        cand[0, 0] = ACT_END
        cand[1, 0] = ACT_PLAY
        cand[1, 2] = OLD_CARD_IDS.index('N01')
        cand[1, option_new] = 1
        cand[2, 0] = ACT_PLAY
        cand[2, 1] = len(OLD_SEATS)
        cand[2, 2] = len(OLD_CARD_IDS)
        extra = current_candidates().index('mulligan:A01')
        cand[2, extra] = 1
        projected_c, rejected = project_candidates(cand, teacher)
        self.assertEqual(tuple(rejected), (False, False, True))
        self.assertEqual(projected_c[1, option_old], 1)
        self.assertEqual(unsupported_action_mask(cand, teacher).tolist(), [False, False, True])
        with self.assertRaisesRegex(ValueError, 'Unsupported source column'):
            named_index_map(['missing'], current_features())

    def test_overlapping_root_keeps_terminal_wdl_and_hides_opponent_cards(self):
        teacher = self.starter
        x = np.zeros(len(current_features()), dtype=np.float32)
        cand = np.zeros((2, len(current_candidates())), dtype=np.float32)
        cand[0, 0] = ACT_ATTACK
        cand[1, 0] = ACT_END
        row = dict(key='starter', x=x, c=cand, z=1, pi=np.array([0.2, 0.8]))
        mapped = overlapping_root(row, teacher)
        self.assertTrue(mapped['overlapping'])
        self.assertEqual(mapped['z'], 1)
        self.assertEqual(mapped['rejected'], 0)
        self.assertEqual(len(mapped['keep']), 2)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            overlapping_root(dict(row, reward_to_go=4.0), teacher)
        with self.assertRaisesRegex(ValueError, 'hidden opponent'):
            overlapping_root(dict(row, hidden_opponent_cards=True), teacher)
        with self.assertRaisesRegex(ValueError, 'no old teacher'):
            overlapping_root(dict(row, key='zhenhong'), teacher)

    def test_paired_seeds_keep_confirmation_out_of_selection(self):
        games = _games(range(10))
        parts = split_paired_roots(games)
        self.assertEqual(parts['paired_seeds'], 10)
        self.assertEqual(parts['both_seats'], 10)
        train_seeds = {game['seed'] for game in parts['train']}
        self.assertFalse(train_seeds & set(parts['selection_seeds']))
        self.assertFalse(train_seeds & set(parts['confirmation_seeds']))
        self.assertFalse(set(parts['selection_seeds']) & set(parts['confirmation_seeds']))
        self.assertEqual(set(parts['confirmation_seeds']),
                         holdout_seeds(sorted(set(range(10)) - set(parts['selection_seeds']))))
        with self.assertRaisesRegex(ValueError, 'Frozen evaluation'):
            split_paired_roots([dict(seed=1, first='a', training=False)])
        with self.assertRaisesRegex(ValueError, 'must not select'):
            select_variant([dict(kl_coefficient=0.1, passed=True)], confirmation_used=True)
        self.assertIsNone(select_variant([dict(kl_coefficient=0.0, passed=True, exact_action=9)]))
        chosen = select_variant([
            dict(kl_coefficient=0.05, passed=True, exact_action=3, cross_entropy=1.0),
            dict(kl_coefficient=0.1, passed=True, exact_action=4, cross_entropy=0.8),
        ])
        self.assertEqual(chosen['kl_coefficient'], 0.1)
        self.assertEqual(validate_kl_grid(), KL_GRID)
        with self.assertRaisesRegex(ValueError, 'must include zero'):
            validate_kl_grid((0.1, 0.2))

    def test_agreement_reports_each_team_denominator_and_miscalls(self):
        x = np.zeros(len(current_features()), dtype=np.float32)
        cand = np.zeros((3, len(current_candidates())), dtype=np.float32)
        cand[0, 0] = ACT_PLAY
        cand[0, 2] = 0
        cand[1, 0] = ACT_ATTACK
        cand[2, 0] = ACT_END
        row = dict(c=cand, z=1)
        mapped = dict(overlapping=True, rejected=0, keep=np.array([0, 1, 2]), teacher_top=1, z=1)
        teacher_prob = np.array([0.1, 0.8, 0.1])
        attack_as_play = agreement_record(row, mapped, np.array([3.0, 1.0, 0.0]), teacher_prob)
        self.assertTrue(attack_as_play['legal_play'])
        self.assertFalse(attack_as_play['exact_action'])
        self.assertTrue(attack_as_play['student_play'])
        self.assertEqual(attack_as_play['teacher_kind'], 'attack')
        end_as_play = agreement_record(row, dict(mapped, teacher_top=2), np.array([2.0, 0.0, 1.0]), teacher_prob)
        exact = agreement_record(row, mapped, np.array([0.0, 4.0, 1.0]), teacher_prob)
        self.assertTrue(exact['exact_action'])
        summary = summarize_agreement([attack_as_play, end_as_play, exact])
        self.assertEqual(summary['n'], 3)
        self.assertEqual(summary['legal_play'], 3)
        self.assertEqual(summary['attack_called_play'], 1)
        self.assertEqual(summary['end_called_play'], 1)
        self.assertEqual(summary['exact_action'], 1)
        self.assertEqual(ACCEPTANCE_BOUNDARY['starter']['new_wins'], 0)
        self.assertEqual(ACCEPTANCE_BOUNDARY['quick-rush']['new_wins'], 19)

    def test_dry_run_validates_inputs_without_training_or_serving_edits(self):
        import gzip
        before = {key: hashlib.sha256((REPO_TEACHERS / f'{key}.npz').read_bytes()).hexdigest() for key in TEAMS}
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            episodes = root / 'episodes'
            episodes.mkdir()
            for seed in range(4):
                for first in ('a', 'b'):
                    payload = dict(
                        job=dict(seed=12030000000 + seed, first=first, left='starter',
                                 right='weave-rush', training=True),
                        rows=[dict(key='starter', z=1)],
                        winner='a',
                    )
                    with gzip.open(episodes / f'train-{seed}-{first}.json.gz', 'wt', encoding='utf-8') as handle:
                        json.dump(payload, handle)
            output = root / 'out'
            result = dry_run(teachers=REPO_TEACHERS, episodes=episodes, output=output)
            self.assertTrue(result['dry_run'])
            self.assertFalse(output.exists())
            self.assertFalse(result['training'])
            self.assertFalse(result['optimizer'])
            self.assertFalse(result['serving_export'])
            self.assertFalse(result['serving_files_rewritten'])
            self.assertFalse(result['hidden_opponent_information'])
            self.assertFalse(result['confirmation_selects_variant'])
            self.assertEqual(result['claimed_improved'], {'zhenhong': False, 'murk': False})
            self.assertEqual(result['acceptance_boundary']['starter']['new_wins'], 0)
            self.assertEqual(result['episodes'], 8)
            self.assertEqual(result['both_seats'], 4)
            from scripts.check_duel_v2_cross_teacher import main
            printed = main(['--teachers', str(REPO_TEACHERS), '--episodes', str(episodes),
                            '--output', str(root / 'cli-out')])
            self.assertFalse((root / 'cli-out').exists())
            self.assertEqual(printed['training'], False)
        after = {key: hashlib.sha256((REPO_TEACHERS / f'{key}.npz').read_bytes()).hexdigest() for key in TEAMS}
        self.assertEqual(before, after)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch optional locally')
    def test_zero_kl_is_noop_and_nonzero_kl_reaches_policy(self):
        import torch
        from copy import deepcopy
        from app.modules.card_game.rl.cross_runtime import create_network
        from app.modules.card_game.rl.outcome_runtime import update as original_update
        torch.set_num_threads(1)
        x = np.zeros(len(current_features()), dtype=np.float32)
        cand = np.zeros((2, len(current_candidates())), dtype=np.float32)
        cand[0, 0] = ACT_PLAY
        cand[0, 2] = 0
        cand[1, 0] = ACT_END
        row = dict(key='starter', x=x, c=cand, z=1, value_actor=1, pi=np.array([0.5, 0.5], np.float32))
        net = create_network(8, 'cpu')
        reference = deepcopy(net)
        opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.05)
        reference_opt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.05)
        before_score = net.score.weight.detach().clone()
        before_cand = net.cand_net[0].weight.detach().clone()
        zero = update_student(net, opt, [row], {'starter': self.starter}, kl_coefficient=0.0)
        original_update(reference, reference_opt, [row])
        self.assertEqual(zero['kl_coefficient'], 0.0)
        self.assertEqual(zero['kl_loss'], 0.0)
        self.assertFalse(zero['serving_export'])
        self.assertEqual(zero['policy_samples'], 1)
        for name, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
        filled = update_student(net, opt, [row], {'starter': self.starter}, kl_coefficient=0.2)
        self.assertGreater(filled['kl_loss'], 0)
        self.assertGreater(filled['policy_samples'], 0)
        self.assertTrue(any(value > 0 for name, value in filled['policy_gradient_norms'].items()
                            if name.startswith(('cand_net', 'score'))))
        self.assertFalse(torch.equal(before_score, net.score.weight.detach())
                         and torch.equal(before_cand, net.cand_net[0].weight.detach()))
        top = update_student(net, opt, [row], {'starter': self.starter},
                             teacher_top_weight=1.0)
        self.assertGreater(top['teacher_top_loss'], 0)
        self.assertEqual(top['teacher_top_weight'], 1.0)
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        mask = torch.tensor([[True, True], [True, False]])
        teacher = torch.tensor([[0.2, 0.1], [0.3, 0.0]])
        noop, stats = teacher_kl_loss(logits, mask, teacher, 0.0)
        self.assertEqual(float(noop), 0.0)
        self.assertEqual(stats['kl_samples'], 0)
        self.assertIsNone(noop.grad_fn)


if __name__ == '__main__':
    unittest.main()
