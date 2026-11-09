import unittest
from pathlib import Path

from flask import Flask
from app.modules.guide import blueprint, service
from app.modules.guide.diyboss_scoring import calculate


class DiyBossScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        app = Flask(__name__, template_folder=str(root / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()
        cls.record = service.detail('diyboss', 'config-3')

    def stage(self, identifier):
        return next(s for s in self.record['stages'] if s['id'] == identifier)

    def maximum(self, identifier, no_hp=False):
        return [g['options'][-1]['id'] for g in self.stage(identifier)['buff_groups']
                if not (no_hp and '生命值' in g['name'])]

    def targets(self, identifier):
        return [t['id'] for t in self.stage(identifier)['targets']]

    def test_known_combinations_and_special_goal_scales(self):
        for stage_id, buffs, targets, expected in [
            ('2', self.maximum('2', True), self.targets('2'), 6375),
            ('7', self.maximum('7', True), [], 27000),
            ('7', self.maximum('7', True), self.targets('7')[::2], 34000),
            ('7', self.maximum('7'), self.targets('7'), 41000),
            ('8', self.maximum('8', True), self.targets('8'), 29625)]:
            with self.subTest(stage=stage_id, expected=expected):
                result = calculate(self.record, stage_id, '4', buffs, targets)
                self.assertEqual(result['total'], expected)
        for identifier, expected in [('1', 25), ('7', 700), ('8', 175)]:
            self.assertEqual([t['score'] for t in self.stage(identifier)['targets']], [expected] * 3)

    def test_none_selected_difficulty_change_and_old_periods(self):
        self.assertEqual(calculate(self.record, '1', '1')['total'], 500)
        self.assertEqual(calculate(self.record, '1', '4')['total'], 2500)
        old = service.detail('diyboss', 'config-1')
        self.assertEqual(calculate(old, '1', '2', ['TimeOP001'])['total'], 1100)

    def test_invalid_selections_are_rejected(self):
        group = self.stage('7')['buff_groups'][0]
        for buffs, targets in [([o['id'] for o in group['options'][:2]], []),
                               ([group['options'][0]['id']] * 2, []),
                               (['invented'], []), ([], self.targets('1')),
                               ([], self.targets('7')[:1] * 2), ('TimeOP001', [])]:
            with self.assertRaises(ValueError):
                calculate(self.record, '7', '4', buffs, targets)
        import copy
        unknown = copy.deepcopy(self.record)
        stage = next(s for s in unknown['stages'] if s['id'] == '7')
        stage['targets'][0]['score'] = None
        with self.assertRaisesRegex(ValueError, '缺少积分'):
            calculate(unknown, '7', '4', [], [stage['targets'][0]['id']])
        with self.assertRaises(LookupError):
            calculate(self.record, '999', '1')
        with self.assertRaises(LookupError):
            calculate(self.record, '1', '999')

    def test_read_only_api_validates_and_returns_breakdown(self):
        response = self.client.post('/api/guide/diyboss/config-3/score', json={
            'stage': '7', 'difficulty': '4', 'buffs': self.maximum('7', True),
            'targets': self.targets('7')[::2]})
        self.assertEqual(response.status_code, 200)
        result = response.json['data']
        self.assertEqual(result['target_score'], 1400)
        self.assertEqual(result['buff_score'], 4900)
        self.assertEqual(result['total'], 34000)
        for payload in [[], {'stage': True}, {'buffs': ['invented']}, {'targets': [7]}]:
            self.assertEqual(self.client.post('/api/guide/diyboss/config-3/score', json=payload).status_code, 400)
        self.assertEqual(self.client.post('/api/guide/diyboss/missing/score', json={}).status_code, 404)
