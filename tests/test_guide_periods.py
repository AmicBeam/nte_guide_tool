"""All dated combat collections share chronological ordering and live-window highlighting."""
from datetime import datetime
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from flask import Flask
from app.modules.guide import blueprint, service
from app.modules.guide.challenge_calendar import is_current_period


class GuidePeriodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        app = Flask(__name__, template_folder=str(root / 'app/templates'))
        app.register_blueprint(blueprint)
        cls.client = app.test_client()

    def test_explicit_window_boundaries_and_timezone(self):
        record = {'begin_at': '2026-11-05T10:00:00+08:00', 'end_at': '2026-12-17T05:59:00+08:00'}
        for timestamp, expected in [
            ('2026-11-05T09:59:59+08:00', False), ('2026-11-05T10:00:00+08:00', True),
            ('2026-11-05T02:00:00+00:00', True), ('2026-12-17T05:58:59+08:00', True),
            ('2026-12-17T05:59:00+08:00', False)]:
            self.assertEqual(is_current_period(record, datetime.fromisoformat(timestamp)), expected)
        for changes in [{'is_permanent': True}, {'begin_at': None}, {'end_at': 'invalid'},
                        {'begin_at': '2026-11-05T10:00:00'}, {'end_at': record['begin_at']}]:
            self.assertFalse(is_current_period({**record, **changes}, datetime.fromisoformat('2026-11-09T12:00:00+08:00')))

    def test_date_precision_and_challenge_five_am(self):
        record = {'kind': 'diyboss', 'begin_date': '2026-11-05', 'end_date': '2026-11-09'}
        self.assertTrue(is_current_period(record, datetime.fromisoformat('2026-11-09T23:59:59+08:00')))
        self.assertFalse(is_current_period(record, datetime.fromisoformat('2026-11-10T00:00:00+08:00')))
        record['kind'] = 'challenges'
        self.assertTrue(is_current_period(record, datetime.fromisoformat('2026-11-09T04:59:59+08:00')))
        self.assertFalse(is_current_period(record, datetime.fromisoformat('2026-11-09T05:00:00+08:00')))

    def test_all_collections_sort_by_date_and_highlight_current_not_latest(self):
        real_records = service.public_records
        for kind in ('challenges', 'diyboss', 'endless'):
            records = []
            # Deliberately misleading IDs/season numbers, input order and status.
            for identifier, start, end, season in [('older', '2026-09-01', '2026-09-20', 99),
                    ('future', '2026-12-01', '2026-12-20', 1),
                    ('current', '2026-11-01', '2026-11-20', 2)]:
                records.append({'id': identifier, 'name': identifier, 'season': season, 'description': '',
                    'is_permanent': False, 'begin_date': start, 'end_date': end,
                    'begin_at': start + 'T10:00:00+08:00', 'end_at': end + 'T05:59:00+08:00',
                    'floor_begin': 1, 'floor_end': 50, 'stage_count': 50, 'status': 'ended'})
            with self.subTest(kind=kind), patch('app.modules.guide.service.public_records',
                    side_effect=lambda selected: records if selected == kind else real_records(selected)), \
                    patch('app.modules.guide.challenge_calendar.current_time',
                          return_value=datetime.fromisoformat('2026-11-09T12:00:00+08:00')) as clock:
                listing = self.client.get('/api/guide/' + kind).json['items']
                self.assertEqual([row['id'] for row in listing], ['future', 'current', 'older'])
                page = self.client.get('/guide?kind=' + kind).text
                links = re.findall(r'class="[^"]*guide-challenge-current[^"]*" href="([^"]+)"', page)
                self.assertEqual(links, [f'/guide/{kind}/current'])
                clock.return_value = datetime.fromisoformat('2027-01-01T12:00:00+08:00')
                self.assertNotIn('guide-challenge-current', self.client.get('/guide?kind=' + kind).text)

    def test_live_data_reverse_order_and_no_current_endless(self):
        with patch('app.modules.guide.challenge_calendar.current_time',
                   return_value=datetime.fromisoformat('2026-11-09T12:00:00+08:00')):
            listing = self.client.get('/api/guide/diyboss').json['items']
            self.assertEqual([row['id'] for row in listing], ['config-3', 'config-2', 'config-1'])
            self.assertIn('guide-challenge-current', self.client.get('/guide?kind=diyboss').text)
            self.assertNotIn('guide-challenge-current', self.client.get('/guide?kind=endless').text)
